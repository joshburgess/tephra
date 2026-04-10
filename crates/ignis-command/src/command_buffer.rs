//! Command buffer wrapper with convenience methods for recording.
//!
//! [`CommandBuffer`] wraps a `vk::CommandBuffer` and provides typed methods for
//! barriers, copies, draws, dispatches, and render passes. In later phases,
//! it will gain state tracking for lazy pipeline compilation and descriptor
//! set management.

use ash::vk;

use crate::barriers::ImageBarrierInfo;

/// Queue type this command buffer was allocated from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandBufferType {
    /// Graphics queue (supports graphics, compute, and transfer).
    Graphics,
    /// Async compute queue.
    AsyncCompute,
    /// Transfer/DMA queue.
    Transfer,
}

/// A recorded command buffer with convenience methods.
///
/// Created via `Device::request_command_buffer()`. Must be submitted back to the
/// device within the same frame it was requested.
pub struct CommandBuffer {
    pub(crate) raw: vk::CommandBuffer,
    pub(crate) cb_type: CommandBufferType,
    pub(crate) device: ash::Device,
    pub(crate) debug_utils: Option<ash::ext::debug_utils::Device>,
    pub(crate) mesh_shader: Option<ash::ext::mesh_shader::Device>,
    pub(crate) acceleration_structure: Option<ash::khr::acceleration_structure::Device>,
    pub(crate) device_generated_commands: Option<ash::nv::device_generated_commands::Device>,
}

impl CommandBuffer {
    /// Create a `CommandBuffer` wrapper from a raw Vulkan command buffer.
    ///
    /// The caller is responsible for ensuring the raw handle is valid
    /// and has been begun for recording.
    pub fn from_raw(
        raw: vk::CommandBuffer,
        cb_type: CommandBufferType,
        device: ash::Device,
    ) -> Self {
        Self {
            raw,
            cb_type,
            device,
            debug_utils: None,
            mesh_shader: None,
            acceleration_structure: None,
            device_generated_commands: None,
        }
    }

    /// Create a `CommandBuffer` wrapper with extension loaders.
    ///
    /// Optional extension loaders enable additional command recording:
    /// - `debug_utils`: debug label methods visible in RenderDoc/Nsight
    /// - `mesh_shader`: mesh shader draw commands
    /// - `acceleration_structure`: RT acceleration structure build/copy commands
    /// - `device_generated_commands`: GPU-driven command generation (NVIDIA)
    #[allow(clippy::too_many_arguments)]
    pub fn from_raw_with_extensions(
        raw: vk::CommandBuffer,
        cb_type: CommandBufferType,
        device: ash::Device,
        debug_utils: Option<ash::ext::debug_utils::Device>,
        mesh_shader: Option<ash::ext::mesh_shader::Device>,
        acceleration_structure: Option<ash::khr::acceleration_structure::Device>,
        device_generated_commands: Option<ash::nv::device_generated_commands::Device>,
    ) -> Self {
        Self {
            raw,
            cb_type,
            device,
            debug_utils,
            mesh_shader,
            acceleration_structure,
            device_generated_commands,
        }
    }

    /// The raw Vulkan command buffer handle.
    pub fn raw(&self) -> vk::CommandBuffer {
        self.raw
    }

    /// The queue type this command buffer was allocated for.
    pub fn cb_type(&self) -> CommandBufferType {
        self.cb_type
    }

    // ---- Barrier helpers ----

    /// Insert a full pipeline barrier with image memory barriers.
    pub fn image_barriers(&mut self, barriers: &[ImageBarrierInfo]) {
        if barriers.is_empty() {
            return;
        }

        let image_barriers: Vec<vk::ImageMemoryBarrier2<'_>> = barriers
            .iter()
            .map(|b| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(b.src_stage)
                    .src_access_mask(b.src_access)
                    .dst_stage_mask(b.dst_stage)
                    .dst_access_mask(b.dst_access)
                    .old_layout(b.old_layout)
                    .new_layout(b.new_layout)
                    .image(b.image)
                    .subresource_range(b.subresource_range)
                    .src_queue_family_index(b.src_queue_family)
                    .dst_queue_family_index(b.dst_queue_family)
            })
            .collect();

        let dep_info = vk::DependencyInfo::default().image_memory_barriers(&image_barriers);

        // SAFETY: command buffer and all image handles in barriers are valid.
        unsafe {
            self.device.cmd_pipeline_barrier2(self.raw, &dep_info);
        }
    }

    /// Insert a single image layout transition barrier.
    pub fn image_barrier(&mut self, barrier: &ImageBarrierInfo) {
        self.image_barriers(std::slice::from_ref(barrier));
    }

    /// Insert a memory barrier (no image/buffer specifics).
    pub fn memory_barrier(
        &mut self,
        src_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
        dst_access: vk::AccessFlags2,
    ) {
        let memory_barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access);

        let dep_info =
            vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&memory_barrier));

        // SAFETY: command buffer is valid.
        unsafe {
            self.device.cmd_pipeline_barrier2(self.raw, &dep_info);
        }
    }

    // ---- Copy commands ----

    /// Copy data between buffers.
    pub fn copy_buffer(&mut self, src: vk::Buffer, dst: vk::Buffer, regions: &[vk::BufferCopy]) {
        // SAFETY: command buffer and buffer handles are valid.
        unsafe {
            self.device.cmd_copy_buffer(self.raw, src, dst, regions);
        }
    }

    /// Copy data from a buffer to an image.
    pub fn copy_buffer_to_image(
        &mut self,
        buffer: vk::Buffer,
        image: vk::Image,
        layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
    ) {
        // SAFETY: command buffer, buffer, and image handles are valid.
        unsafe {
            self.device
                .cmd_copy_buffer_to_image(self.raw, buffer, image, layout, regions);
        }
    }

    /// Copy data from an image to a buffer.
    pub fn copy_image_to_buffer(
        &mut self,
        image: vk::Image,
        layout: vk::ImageLayout,
        buffer: vk::Buffer,
        regions: &[vk::BufferImageCopy],
    ) {
        // SAFETY: command buffer, image, and buffer handles are valid.
        unsafe {
            self.device
                .cmd_copy_image_to_buffer(self.raw, image, layout, buffer, regions);
        }
    }

    // ---- Draw commands ----

    /// Issue a non-indexed draw call.
    pub fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        // SAFETY: command buffer is valid and inside a render pass.
        unsafe {
            self.device.cmd_draw(
                self.raw,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
    }

    /// Issue an indexed draw call.
    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        // SAFETY: command buffer is valid and inside a render pass.
        unsafe {
            self.device.cmd_draw_indexed(
                self.raw,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }

    // ---- Indirect draw commands ----

    /// Issue an indirect draw call.
    pub fn draw_indirect(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        // SAFETY: command buffer and buffer are valid, inside a render pass.
        unsafe {
            self.device
                .cmd_draw_indirect(self.raw, buffer, offset, draw_count, stride);
        }
    }

    /// Issue an indirect indexed draw call.
    pub fn draw_indexed_indirect(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        // SAFETY: command buffer and buffer are valid, inside a render pass.
        unsafe {
            self.device
                .cmd_draw_indexed_indirect(self.raw, buffer, offset, draw_count, stride);
        }
    }

    /// Issue an indirect draw call with a GPU-driven draw count (Vulkan 1.2).
    pub fn draw_indirect_count(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        count_buffer: vk::Buffer,
        count_offset: vk::DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) {
        // SAFETY: command buffer and buffers are valid, inside a render pass.
        unsafe {
            self.device.cmd_draw_indirect_count(
                self.raw,
                buffer,
                offset,
                count_buffer,
                count_offset,
                max_draw_count,
                stride,
            );
        }
    }

    /// Issue an indirect indexed draw call with a GPU-driven draw count (Vulkan 1.2).
    pub fn draw_indexed_indirect_count(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        count_buffer: vk::Buffer,
        count_offset: vk::DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) {
        // SAFETY: command buffer and buffers are valid, inside a render pass.
        unsafe {
            self.device.cmd_draw_indexed_indirect_count(
                self.raw,
                buffer,
                offset,
                count_buffer,
                count_offset,
                max_draw_count,
                stride,
            );
        }
    }

    // ---- Compute ----

    /// Dispatch a compute shader.
    pub fn dispatch(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        // SAFETY: command buffer is valid and not inside a render pass.
        unsafe {
            self.device
                .cmd_dispatch(self.raw, group_count_x, group_count_y, group_count_z);
        }
    }

    /// Dispatch a compute shader indirectly from a buffer.
    pub fn dispatch_indirect(&mut self, buffer: vk::Buffer, offset: vk::DeviceSize) {
        // SAFETY: command buffer and buffer are valid, not inside a render pass.
        unsafe {
            self.device.cmd_dispatch_indirect(self.raw, buffer, offset);
        }
    }

    // ---- Render pass ----

    /// Begin a render pass with the given info.
    pub fn begin_render_pass(
        &mut self,
        render_pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
        render_area: vk::Rect2D,
        clear_values: &[vk::ClearValue],
    ) {
        let begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .render_area(render_area)
            .clear_values(clear_values);

        // SAFETY: command buffer, render pass, and framebuffer are valid.
        unsafe {
            self.device
                .cmd_begin_render_pass(self.raw, &begin_info, vk::SubpassContents::INLINE);
        }
    }

    /// End the current render pass.
    pub fn end_render_pass(&mut self) {
        // SAFETY: command buffer is valid and inside a render pass.
        unsafe {
            self.device.cmd_end_render_pass(self.raw);
        }
    }

    // ---- Dynamic rendering (Vulkan 1.3 / VK_KHR_dynamic_rendering) ----

    /// Begin dynamic rendering with the given rendering info.
    ///
    /// This replaces the traditional `VkRenderPass` + `VkFramebuffer` approach.
    /// Requires `VK_KHR_dynamic_rendering` or Vulkan 1.3.
    pub fn begin_rendering(&mut self, rendering_info: &vk::RenderingInfo<'_>) {
        // SAFETY: command buffer is valid, rendering info is well-formed.
        unsafe {
            self.device.cmd_begin_rendering(self.raw, rendering_info);
        }
    }

    /// End the current dynamic rendering scope.
    pub fn end_rendering(&mut self) {
        // SAFETY: command buffer is valid and inside a dynamic rendering scope.
        unsafe {
            self.device.cmd_end_rendering(self.raw);
        }
    }

    // ---- Binding commands ----

    /// Bind a graphics pipeline.
    pub fn bind_pipeline(&mut self, bind_point: vk::PipelineBindPoint, pipeline: vk::Pipeline) {
        // SAFETY: command buffer and pipeline are valid.
        unsafe {
            self.device
                .cmd_bind_pipeline(self.raw, bind_point, pipeline);
        }
    }

    /// Bind vertex buffers.
    pub fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
    ) {
        // SAFETY: command buffer and buffer handles are valid.
        unsafe {
            self.device
                .cmd_bind_vertex_buffers(self.raw, first_binding, buffers, offsets);
        }
    }

    /// Bind an index buffer.
    pub fn bind_index_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        // SAFETY: command buffer and buffer handle are valid.
        unsafe {
            self.device
                .cmd_bind_index_buffer(self.raw, buffer, offset, index_type);
        }
    }

    /// Set the viewport dynamically.
    pub fn set_viewport(&mut self, first_viewport: u32, viewports: &[vk::Viewport]) {
        // SAFETY: command buffer is valid.
        unsafe {
            self.device
                .cmd_set_viewport(self.raw, first_viewport, viewports);
        }
    }

    /// Set the scissor rectangle dynamically.
    pub fn set_scissor(&mut self, first_scissor: u32, scissors: &[vk::Rect2D]) {
        // SAFETY: command buffer is valid.
        unsafe {
            self.device
                .cmd_set_scissor(self.raw, first_scissor, scissors);
        }
    }

    /// Push constants from a raw byte slice.
    pub fn push_constants(
        &mut self,
        layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        data: &[u8],
    ) {
        // SAFETY: command buffer, layout are valid; data is correctly sized.
        unsafe {
            self.device
                .cmd_push_constants(self.raw, layout, stage_flags, offset, data);
        }
    }

    /// Push constants from a typed value.
    ///
    /// Converts `data` to bytes via [`bytemuck`] and pushes at offset 0.
    pub fn push_constants_typed<T: bytemuck::Pod>(
        &mut self,
        layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        data: &T,
    ) {
        self.push_constants(layout, stage_flags, 0, bytemuck::bytes_of(data));
    }

    // ---- Blit / Clear / Fill commands ----

    /// Blit (scaled copy) from one image to another.
    ///
    /// Supports format conversion and scaling. Both images must be in
    /// appropriate layouts (`TRANSFER_SRC_OPTIMAL` / `TRANSFER_DST_OPTIMAL`).
    pub fn blit_image(
        &mut self,
        src: vk::Image,
        src_layout: vk::ImageLayout,
        dst: vk::Image,
        dst_layout: vk::ImageLayout,
        regions: &[vk::ImageBlit],
        filter: vk::Filter,
    ) {
        // SAFETY: command buffer, images are valid, layouts are correct.
        unsafe {
            self.device
                .cmd_blit_image(self.raw, src, src_layout, dst, dst_layout, regions, filter);
        }
    }

    /// Generate a full mipmap chain for an image.
    ///
    /// Assumes mip level 0 contains the source data and is in
    /// `TRANSFER_DST_OPTIMAL` layout. Each subsequent level is generated
    /// by blitting from the previous level with linear filtering.
    /// All mip levels end in `SHADER_READ_ONLY_OPTIMAL`.
    pub fn generate_mipmap(&mut self, image: vk::Image, width: u32, height: u32, mip_levels: u32) {
        let mut mip_width = width as i32;
        let mut mip_height = height as i32;

        for i in 1..mip_levels {
            // Transition level i-1: TRANSFER_DST → TRANSFER_SRC
            self.image_barrier(&ImageBarrierInfo {
                image,
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_stage: vk::PipelineStageFlags2::TRANSFER,
                src_access: vk::AccessFlags2::TRANSFER_WRITE,
                dst_stage: vk::PipelineStageFlags2::TRANSFER,
                dst_access: vk::AccessFlags2::TRANSFER_READ,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: i - 1,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_queue_family: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family: vk::QUEUE_FAMILY_IGNORED,
            });

            let next_width = (mip_width / 2).max(1);
            let next_height = (mip_height / 2).max(1);

            let blit = vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i - 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_width,
                        y: mip_height,
                        z: 1,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: next_width,
                        y: next_height,
                        z: 1,
                    },
                ],
            };

            // SAFETY: command buffer and image are valid.
            unsafe {
                self.device.cmd_blit_image(
                    self.raw,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit],
                    vk::Filter::LINEAR,
                );
            }

            // Transition level i-1: TRANSFER_SRC → SHADER_READ_ONLY
            self.image_barrier(&ImageBarrierInfo {
                image,
                old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_stage: vk::PipelineStageFlags2::TRANSFER,
                src_access: vk::AccessFlags2::TRANSFER_READ,
                dst_stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                dst_access: vk::AccessFlags2::SHADER_READ,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: i - 1,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_queue_family: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family: vk::QUEUE_FAMILY_IGNORED,
            });

            mip_width = next_width;
            mip_height = next_height;
        }

        // Transition last mip level: TRANSFER_DST → SHADER_READ_ONLY
        self.image_barrier(&ImageBarrierInfo {
            image,
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_stage: vk::PipelineStageFlags2::TRANSFER,
            src_access: vk::AccessFlags2::TRANSFER_WRITE,
            dst_stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            dst_access: vk::AccessFlags2::SHADER_READ,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: mip_levels - 1,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_queue_family: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family: vk::QUEUE_FAMILY_IGNORED,
        });
    }

    /// Clear a color image to a uniform value.
    ///
    /// The image must be in `GENERAL` or `TRANSFER_DST_OPTIMAL` layout.
    pub fn clear_color_image(
        &mut self,
        image: vk::Image,
        layout: vk::ImageLayout,
        clear_value: &vk::ClearColorValue,
        ranges: &[vk::ImageSubresourceRange],
    ) {
        // SAFETY: command buffer and image are valid, layout is correct.
        unsafe {
            self.device
                .cmd_clear_color_image(self.raw, image, layout, clear_value, ranges);
        }
    }

    /// Clear a depth/stencil image to a uniform value.
    ///
    /// The image must be in `GENERAL` or `TRANSFER_DST_OPTIMAL` layout.
    pub fn clear_depth_stencil_image(
        &mut self,
        image: vk::Image,
        layout: vk::ImageLayout,
        clear_value: &vk::ClearDepthStencilValue,
        ranges: &[vk::ImageSubresourceRange],
    ) {
        // SAFETY: command buffer and image are valid, layout is correct.
        unsafe {
            self.device
                .cmd_clear_depth_stencil_image(self.raw, image, layout, clear_value, ranges);
        }
    }

    /// Fill a buffer region with a 32-bit value.
    ///
    /// `offset` and `size` must be multiples of 4. Use `vk::WHOLE_SIZE` for size
    /// to fill from offset to end.
    pub fn fill_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        data: u32,
    ) {
        // SAFETY: command buffer and buffer are valid, offset/size are aligned.
        unsafe {
            self.device
                .cmd_fill_buffer(self.raw, buffer, offset, size, data);
        }
    }

    /// Inline-update a buffer region with arbitrary data.
    ///
    /// Limited to 65536 bytes. `offset` must be a multiple of 4, and `data.len()`
    /// must be a multiple of 4.
    pub fn update_buffer(&mut self, buffer: vk::Buffer, offset: vk::DeviceSize, data: &[u8]) {
        // SAFETY: command buffer and buffer are valid, data size constraints met.
        unsafe {
            self.device
                .cmd_update_buffer(self.raw, buffer, offset, data);
        }
    }

    /// Resolve a multisample image into a single-sample image.
    pub fn resolve_image(
        &mut self,
        src: vk::Image,
        src_layout: vk::ImageLayout,
        dst: vk::Image,
        dst_layout: vk::ImageLayout,
        regions: &[vk::ImageResolve],
    ) {
        // SAFETY: command buffer and images are valid.
        unsafe {
            self.device
                .cmd_resolve_image(self.raw, src, src_layout, dst, dst_layout, regions);
        }
    }

    /// Copy regions between two images.
    pub fn copy_image(
        &mut self,
        src: vk::Image,
        src_layout: vk::ImageLayout,
        dst: vk::Image,
        dst_layout: vk::ImageLayout,
        regions: &[vk::ImageCopy],
    ) {
        // SAFETY: command buffer and images are valid.
        unsafe {
            self.device
                .cmd_copy_image(self.raw, src, src_layout, dst, dst_layout, regions);
        }
    }

    // ---- Dynamic state commands ----

    /// Set depth bias dynamically.
    pub fn set_depth_bias(&mut self, constant_factor: f32, clamp: f32, slope_factor: f32) {
        // SAFETY: command buffer is valid.
        unsafe {
            self.device
                .cmd_set_depth_bias(self.raw, constant_factor, clamp, slope_factor);
        }
    }

    /// Set blend constants dynamically.
    pub fn set_blend_constants(&mut self, constants: &[f32; 4]) {
        // SAFETY: command buffer is valid.
        unsafe {
            self.device.cmd_set_blend_constants(self.raw, constants);
        }
    }

    /// Set stencil compare mask dynamically.
    pub fn set_stencil_compare_mask(&mut self, face_mask: vk::StencilFaceFlags, mask: u32) {
        // SAFETY: command buffer is valid.
        unsafe {
            self.device
                .cmd_set_stencil_compare_mask(self.raw, face_mask, mask);
        }
    }

    /// Set stencil write mask dynamically.
    pub fn set_stencil_write_mask(&mut self, face_mask: vk::StencilFaceFlags, mask: u32) {
        // SAFETY: command buffer is valid.
        unsafe {
            self.device
                .cmd_set_stencil_write_mask(self.raw, face_mask, mask);
        }
    }

    /// Set stencil reference value dynamically.
    pub fn set_stencil_reference(&mut self, face_mask: vk::StencilFaceFlags, reference: u32) {
        // SAFETY: command buffer is valid.
        unsafe {
            self.device
                .cmd_set_stencil_reference(self.raw, face_mask, reference);
        }
    }

    // ---- Debug label commands ----

    /// Begin a debug label region in the command buffer.
    ///
    /// Appears in tools like RenderDoc and Nsight. No-op if `VK_EXT_debug_utils`
    /// is not available.
    pub fn begin_debug_label(&mut self, name: &str, color: [f32; 4]) {
        if let Some(debug_utils) = &self.debug_utils {
            let c_name = std::ffi::CString::new(name).unwrap_or_default();
            let label = vk::DebugUtilsLabelEXT::default()
                .label_name(&c_name)
                .color(color);
            // SAFETY: command buffer is valid, label is well-formed.
            unsafe {
                debug_utils.cmd_begin_debug_utils_label(self.raw, &label);
            }
        }
    }

    /// End the current debug label region.
    ///
    /// No-op if `VK_EXT_debug_utils` is not available.
    pub fn end_debug_label(&mut self) {
        if let Some(debug_utils) = &self.debug_utils {
            // SAFETY: command buffer is valid and inside a debug label region.
            unsafe {
                debug_utils.cmd_end_debug_utils_label(self.raw);
            }
        }
    }

    /// Insert a single debug label point.
    ///
    /// No-op if `VK_EXT_debug_utils` is not available.
    pub fn insert_debug_label(&mut self, name: &str, color: [f32; 4]) {
        if let Some(debug_utils) = &self.debug_utils {
            let c_name = std::ffi::CString::new(name).unwrap_or_default();
            let label = vk::DebugUtilsLabelEXT::default()
                .label_name(&c_name)
                .color(color);
            // SAFETY: command buffer is valid, label is well-formed.
            unsafe {
                debug_utils.cmd_insert_debug_utils_label(self.raw, &label);
            }
        }
    }

    // ---- Query commands ----

    /// Write a GPU timestamp at the given pipeline stage.
    pub fn write_timestamp(
        &mut self,
        stage: vk::PipelineStageFlags2,
        query_pool: vk::QueryPool,
        query: u32,
    ) {
        // SAFETY: command buffer and query pool are valid, query index is in range.
        unsafe {
            self.device
                .cmd_write_timestamp2(self.raw, stage, query_pool, query);
        }
    }

    /// Begin a query.
    pub fn begin_query(
        &mut self,
        query_pool: vk::QueryPool,
        query: u32,
        flags: vk::QueryControlFlags,
    ) {
        // SAFETY: command buffer and query pool are valid.
        unsafe {
            self.device
                .cmd_begin_query(self.raw, query_pool, query, flags);
        }
    }

    /// End a query.
    pub fn end_query(&mut self, query_pool: vk::QueryPool, query: u32) {
        // SAFETY: command buffer and query pool are valid.
        unsafe {
            self.device.cmd_end_query(self.raw, query_pool, query);
        }
    }

    /// Reset a range of queries in a pool from the command buffer.
    pub fn reset_query_pool(&mut self, query_pool: vk::QueryPool, first: u32, count: u32) {
        // SAFETY: command buffer and query pool are valid.
        unsafe {
            self.device
                .cmd_reset_query_pool(self.raw, query_pool, first, count);
        }
    }

    // ---- Event commands (split barriers) ----

    /// Set an event with pipeline stage and memory dependency info.
    ///
    /// Used with [`wait_events2`](Self::wait_events2) for split barriers.
    /// Requires `VK_KHR_synchronization2` or Vulkan 1.3.
    pub fn set_event2(&mut self, event: vk::Event, dependency_info: &vk::DependencyInfo<'_>) {
        // SAFETY: command buffer and event are valid.
        unsafe {
            self.device.cmd_set_event2(self.raw, event, dependency_info);
        }
    }

    /// Wait on one or more events with dependency info.
    ///
    /// Pairs with [`set_event2`](Self::set_event2) for split barriers.
    pub fn wait_events2(
        &mut self,
        events: &[vk::Event],
        dependency_infos: &[vk::DependencyInfo<'_>],
    ) {
        // SAFETY: command buffer, events, and dependency infos are valid.
        unsafe {
            self.device
                .cmd_wait_events2(self.raw, events, dependency_infos);
        }
    }

    /// Reset an event from the command buffer.
    pub fn reset_event2(&mut self, event: vk::Event, stage_mask: vk::PipelineStageFlags2) {
        // SAFETY: command buffer and event are valid.
        unsafe {
            self.device.cmd_reset_event2(self.raw, event, stage_mask);
        }
    }

    // ---- Secondary command buffer execution ----

    /// Execute one or more secondary command buffers.
    ///
    /// The secondary command buffers must have been recorded with
    /// `RENDER_PASS_CONTINUE_BIT` if called inside a render pass.
    pub fn execute_commands(&mut self, secondary_buffers: &[vk::CommandBuffer]) {
        // SAFETY: command buffer and secondary buffers are valid.
        unsafe {
            self.device
                .cmd_execute_commands(self.raw, secondary_buffers);
        }
    }

    // ---- Mesh shader commands ----

    /// Dispatch mesh shader work groups.
    ///
    /// No-op if `VK_EXT_mesh_shader` is not available.
    pub fn draw_mesh_tasks(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        if let Some(mesh) = &self.mesh_shader {
            // SAFETY: command buffer is valid and a mesh shader pipeline is bound.
            unsafe {
                mesh.cmd_draw_mesh_tasks(self.raw, group_count_x, group_count_y, group_count_z);
            }
        }
    }

    /// Dispatch mesh shader work groups with parameters from a buffer.
    ///
    /// No-op if `VK_EXT_mesh_shader` is not available.
    pub fn draw_mesh_tasks_indirect(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        if let Some(mesh) = &self.mesh_shader {
            // SAFETY: command buffer and buffer are valid.
            unsafe {
                mesh.cmd_draw_mesh_tasks_indirect(self.raw, buffer, offset, draw_count, stride);
            }
        }
    }

    /// Dispatch mesh shader work groups with count from a buffer.
    ///
    /// No-op if `VK_EXT_mesh_shader` is not available.
    pub fn draw_mesh_tasks_indirect_count(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        count_buffer: vk::Buffer,
        count_offset: vk::DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) {
        if let Some(mesh) = &self.mesh_shader {
            // SAFETY: command buffer, buffer, and count buffer are valid.
            unsafe {
                mesh.cmd_draw_mesh_tasks_indirect_count(
                    self.raw,
                    buffer,
                    offset,
                    count_buffer,
                    count_offset,
                    max_draw_count,
                    stride,
                );
            }
        }
    }

    // ---- Acceleration structure commands ----

    /// Build acceleration structures.
    ///
    /// No-op if `VK_KHR_acceleration_structure` is not available.
    pub fn build_acceleration_structures(
        &mut self,
        infos: &[vk::AccelerationStructureBuildGeometryInfoKHR<'_>],
        build_range_infos: &[&[vk::AccelerationStructureBuildRangeInfoKHR]],
    ) {
        if let Some(accel) = &self.acceleration_structure {
            // SAFETY: command buffer and build info are valid.
            unsafe {
                accel.cmd_build_acceleration_structures(self.raw, infos, build_range_infos);
            }
        }
    }

    /// Copy an acceleration structure.
    ///
    /// Used for compaction (compact mode) or cloning (clone mode).
    /// No-op if `VK_KHR_acceleration_structure` is not available.
    pub fn copy_acceleration_structure(&mut self, info: &vk::CopyAccelerationStructureInfoKHR<'_>) {
        if let Some(accel) = &self.acceleration_structure {
            // SAFETY: command buffer and copy info are valid.
            unsafe {
                accel.cmd_copy_acceleration_structure(self.raw, info);
            }
        }
    }

    /// Write acceleration structure properties to a query pool.
    ///
    /// Commonly used to query compacted size after building with
    /// `ALLOW_COMPACTION` flag. No-op if `VK_KHR_acceleration_structure`
    /// is not available.
    pub fn write_acceleration_structures_properties(
        &mut self,
        structures: &[vk::AccelerationStructureKHR],
        query_type: vk::QueryType,
        query_pool: vk::QueryPool,
        first_query: u32,
    ) {
        if let Some(accel) = &self.acceleration_structure {
            // SAFETY: command buffer, structures, and query pool are valid.
            unsafe {
                accel.cmd_write_acceleration_structures_properties(
                    self.raw,
                    structures,
                    query_type,
                    query_pool,
                    first_query,
                );
            }
        }
    }

    // ---- Device-generated commands ----

    /// Preprocess device-generated commands.
    ///
    /// Prepares the generated commands buffer for later execution.
    /// No-op if `VK_NV_device_generated_commands` is not available.
    pub fn preprocess_generated_commands(&mut self, info: &vk::GeneratedCommandsInfoNV<'_>) {
        if let Some(dgc) = &self.device_generated_commands {
            // SAFETY: command buffer and generated commands info are valid.
            unsafe {
                (dgc.fp().cmd_preprocess_generated_commands_nv)(self.raw, info);
            }
        }
    }

    /// Execute device-generated commands.
    ///
    /// Executes commands that were previously preprocessed (or generates and
    /// executes in one step if `is_preprocessed` is false).
    /// No-op if `VK_NV_device_generated_commands` is not available.
    pub fn execute_generated_commands(
        &mut self,
        is_preprocessed: bool,
        info: &vk::GeneratedCommandsInfoNV<'_>,
    ) {
        if let Some(dgc) = &self.device_generated_commands {
            // SAFETY: command buffer and generated commands info are valid.
            unsafe {
                (dgc.fp().cmd_execute_generated_commands_nv)(
                    self.raw,
                    vk::Bool32::from(is_preprocessed),
                    info,
                );
            }
        }
    }

    // ---- Binding commands ----

    /// Bind descriptor sets.
    pub fn bind_descriptor_sets(
        &mut self,
        bind_point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first_set: u32,
        descriptor_sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        // SAFETY: command buffer, layout, and descriptor sets are valid.
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                self.raw,
                bind_point,
                layout,
                first_set,
                descriptor_sets,
                dynamic_offsets,
            );
        }
    }
}

/// Begin a secondary command buffer for recording.
///
/// Secondary command buffers can be recorded on worker threads and executed
/// from a primary command buffer via [`CommandBuffer::execute_commands`].
pub fn begin_secondary(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    usage: vk::CommandBufferUsageFlags,
    inheritance: &vk::CommandBufferInheritanceInfo<'_>,
) -> Result<(), vk::Result> {
    let begin_info = vk::CommandBufferBeginInfo::default()
        .flags(usage)
        .inheritance_info(inheritance);

    // SAFETY: device and cmd are valid, begin_info is well-formed.
    unsafe { device.begin_command_buffer(cmd, &begin_info) }
}
