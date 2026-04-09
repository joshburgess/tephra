//! Runtime execution of a compiled render graph.
//!
//! Records barriers and invokes pass callbacks into a
//! [`CommandBuffer`](ignis_command::command_buffer::CommandBuffer).

use ash::vk;
use ignis_command::barriers::ImageBarrierInfo;
use ignis_command::command_buffer::CommandBuffer;

use crate::compile::{access_info, CompiledGraph, ResourceBarrier};
use crate::pass::AccessType;
use crate::resource::{ResourceDeclaration, ResourceInfo};

/// Records a compiled render graph into a command buffer.
///
/// The caller is responsible for allocating physical images and providing
/// them via the `images` slice (indexed by resource handle). Buffer
/// resources should have `vk::Image::null()` in their slot.
pub struct GraphExecutor;

impl GraphExecutor {
    /// Record the full graph execution.
    ///
    /// For each step in execution order: emits pre-barriers, then invokes
    /// the pass callback. After all passes, emits a final barrier to
    /// transition the backbuffer to `PRESENT_SRC_KHR`.
    pub fn record(graph: &CompiledGraph, cmd: &mut CommandBuffer, images: &[vk::Image]) {
        for (step, &pass_idx) in graph.pass_order.iter().enumerate() {
            emit_barriers(cmd, &graph.barriers[step], &graph.resources, images);

            if let Some(ref callback) = graph.passes[pass_idx].callback {
                callback.build_render_pass(cmd);
            }
        }

        // Transition backbuffer to present layout.
        if let Some(bb) = graph.backbuffer {
            emit_present_barrier(graph, cmd, bb, images);
        }
    }
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
