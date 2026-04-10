//! Runtime execution of a compiled render graph.
//!
//! Records barriers, manages dynamic rendering state, and invokes pass
//! callbacks into a [`CommandBuffer`](ignis_command::command_buffer::CommandBuffer).

use ash::vk;
use ignis_command::barriers::ImageBarrierInfo;
use ignis_command::command_buffer::CommandBuffer;

use crate::allocate::PhysicalResources;
use crate::compile::{access_info, CompiledGraph, ResourceBarrier};
use crate::pass::{AccessType, PassDeclaration};
use crate::resource::{ResourceDeclaration, ResourceInfo};

/// Records a compiled render graph into a command buffer.
///
/// Uses [`PhysicalResources`] for physical image handles and views.
/// Graphics passes are automatically wrapped in dynamic rendering
/// (Vulkan 1.3 `vkCmdBeginRendering`/`vkCmdEndRendering`).
pub struct GraphExecutor;

impl GraphExecutor {
    /// Record the full graph execution.
    ///
    /// For each step in execution order:
    /// 1. Emits pre-barriers for resource transitions
    /// 2. Begins dynamic rendering for graphics passes (auto-configured from resource accesses)
    /// 3. Invokes the pass callback
    /// 4. Ends dynamic rendering
    ///
    /// After all passes, emits a final barrier to transition the backbuffer
    /// to `PRESENT_SRC_KHR`.
    pub fn record(
        graph: &CompiledGraph,
        cmd: &mut CommandBuffer,
        resources: &PhysicalResources,
    ) {
        let images = resources.images();

        for (step, &pass_idx) in graph.pass_order.iter().enumerate() {
            emit_barriers(cmd, &graph.barriers[step], &graph.resources, images);

            let pass = &graph.passes[pass_idx];

            if !pass.is_compute {
                begin_pass_rendering(cmd, pass, resources);
            }

            if let Some(ref callback) = pass.callback {
                callback.build_render_pass(cmd);
            }

            if !pass.is_compute {
                // SAFETY: device is valid, rendering was begun.
                cmd.end_rendering();
            }
        }

        // Transition backbuffer to present layout.
        if let Some(bb) = graph.backbuffer {
            emit_present_barrier(graph, cmd, bb, images);
        }
    }
}

/// Build and begin dynamic rendering for a graphics pass.
///
/// Inspects the pass's resource accesses to determine color and depth/stencil
/// attachments, then calls `cmd_begin_rendering` with appropriate load/store ops.
/// Clear values are sourced from the pass callback (if present), falling back
/// to black for color and 1.0/0 for depth/stencil.
fn begin_pass_rendering(
    cmd: &mut CommandBuffer,
    pass: &PassDeclaration,
    resources: &PhysicalResources,
) {
    let mut color_attachments: Vec<vk::RenderingAttachmentInfo<'_>> = Vec::new();
    let mut depth_attachment: Option<vk::RenderingAttachmentInfo<'_>> = None;
    let mut stencil_attachment: Option<vk::RenderingAttachmentInfo<'_>> = None;
    let mut render_extent = vk::Extent2D { width: 0, height: 0 };
    let mut color_index: usize = 0;

    let views = resources.views();
    let extents = resources.extents();
    let formats = resources.formats();
    let callback = pass.callback.as_deref();

    for access in &pass.accesses {
        let idx = access.resource.index as usize;
        let view = views[idx];
        let extent = extents[idx];

        match access.access_type {
            AccessType::ColorOutput => {
                render_extent = max_extent(render_extent, extent);
                let clear_color = callback
                    .map(|cb| cb.clear_color(color_index))
                    .unwrap_or(vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    });
                color_index += 1;
                color_attachments.push(
                    vk::RenderingAttachmentInfo::default()
                        .image_view(view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue { color: clear_color }),
                );
            }
            AccessType::ColorInput => {
                render_extent = max_extent(render_extent, extent);
                color_attachments.push(
                    vk::RenderingAttachmentInfo::default()
                        .image_view(view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::LOAD)
                        .store_op(vk::AttachmentStoreOp::STORE),
                );
            }
            AccessType::DepthStencilOutput => {
                render_extent = max_extent(render_extent, extent);
                let format = formats[idx];
                let clear_ds = callback
                    .map(|cb| cb.clear_depth_stencil())
                    .unwrap_or(vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    });
                let attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        depth_stencil: clear_ds,
                    });
                depth_attachment = Some(attachment);
                if has_stencil_component(format) {
                    stencil_attachment = Some(attachment);
                }
            }
            AccessType::DepthStencilInput => {
                render_extent = max_extent(render_extent, extent);
                let format = formats[idx];
                let attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::NONE);
                depth_attachment = Some(attachment);
                if has_stencil_component(format) {
                    stencil_attachment = Some(attachment);
                }
            }
            // Texture, storage, and attachment inputs don't produce rendering attachments.
            _ => {}
        }
    }

    let mut rendering_info = vk::RenderingInfo::default()
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: render_extent,
        })
        .layer_count(1)
        .color_attachments(&color_attachments);

    if let Some(ref depth) = depth_attachment {
        rendering_info = rendering_info.depth_attachment(depth);
    }
    if let Some(ref stencil) = stencil_attachment {
        rendering_info = rendering_info.stencil_attachment(stencil);
    }

    // SAFETY: device is valid, rendering_info is well-formed.
    cmd.begin_rendering(&rendering_info);
}

fn max_extent(a: vk::Extent2D, b: vk::Extent2D) -> vk::Extent2D {
    vk::Extent2D {
        width: a.width.max(b.width),
        height: a.height.max(b.height),
    }
}

fn has_stencil_component(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT
            | vk::Format::S8_UINT
    )
}

fn emit_barriers(
    cmd: &mut CommandBuffer,
    barriers: &[ResourceBarrier],
    resources: &[ResourceDeclaration],
    images: &[vk::Image],
) {
    if barriers.is_empty() {
        return;
    }

    let mut image_barriers = Vec::new();
    let mut buf_src_stage = vk::PipelineStageFlags2::NONE;
    let mut buf_src_access = vk::AccessFlags2::NONE;
    let mut buf_dst_stage = vk::PipelineStageFlags2::NONE;
    let mut buf_dst_access = vk::AccessFlags2::NONE;
    let mut has_buffer_barrier = false;

    for barrier in barriers {
        let res_idx = barrier.resource.index as usize;

        match &resources[res_idx].info {
            ResourceInfo::Attachment(info) => {
                let (src_stage, src_acc, old_layout) = access_info(barrier.src_access);
                let (dst_stage, dst_acc, new_layout) = access_info(barrier.dst_access);

                image_barriers.push(ImageBarrierInfo {
                    image: images[res_idx],
                    old_layout,
                    new_layout,
                    src_stage,
                    dst_stage,
                    src_access: src_acc,
                    dst_access: dst_acc,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: aspect_for_format(info.format),
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    },
                    src_queue_family: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family: vk::QUEUE_FAMILY_IGNORED,
                });
            }
            ResourceInfo::Buffer(_) => {
                let (src_s, src_a, _) = access_info(barrier.src_access);
                let (dst_s, dst_a, _) = access_info(barrier.dst_access);
                has_buffer_barrier = true;
                buf_src_stage |= src_s;
                buf_src_access |= src_a;
                buf_dst_stage |= dst_s;
                buf_dst_access |= dst_a;
            }
        }
    }

    if !image_barriers.is_empty() {
        cmd.image_barriers(&image_barriers);
    }

    if has_buffer_barrier {
        cmd.memory_barrier(buf_src_stage, buf_src_access, buf_dst_stage, buf_dst_access);
    }
}

fn emit_present_barrier(
    graph: &CompiledGraph,
    cmd: &mut CommandBuffer,
    backbuffer: crate::resource::ResourceHandle,
    images: &[vk::Image],
) {
    let bb_idx = backbuffer.index as usize;
    if bb_idx >= images.len() || images[bb_idx] == vk::Image::null() {
        return;
    }

    // Find the last access type to determine the source state.
    let src_access = find_last_access(graph, backbuffer);
    let Some(src) = src_access else { return };

    let (src_stage, src_acc, old_layout) = access_info(src);

    cmd.image_barrier(&ImageBarrierInfo {
        image: images[bb_idx],
        old_layout,
        new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        src_stage,
        src_access: src_acc,
        dst_stage: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
        dst_access: vk::AccessFlags2::NONE,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
        src_queue_family: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family: vk::QUEUE_FAMILY_IGNORED,
    });
}

fn find_last_access(
    graph: &CompiledGraph,
    resource: crate::resource::ResourceHandle,
) -> Option<AccessType> {
    for &pass_idx in graph.pass_order.iter().rev() {
        for access in &graph.passes[pass_idx].accesses {
            if access.resource == resource {
                return Some(access.access_type);
            }
        }
    }
    None
}

fn aspect_for_format(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
            vk::ImageAspectFlags::DEPTH
        }
        vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        _ => vk::ImageAspectFlags::COLOR,
    }
}
