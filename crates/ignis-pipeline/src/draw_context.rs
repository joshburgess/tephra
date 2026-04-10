//! High-level draw context integrating command buffers, descriptors, and pipelines.
//!
//! [`DrawContext`] wraps a [`CommandBuffer`] and adds automatic pipeline
//! compilation, descriptor set flushing, and render pass management.
//! This is the Granite-style convenience layer where calling `draw()` or
//! `dispatch()` triggers lazy pipeline resolution and descriptor binding.

use ash::vk;
use log::debug;
use thiserror::Error;

use ignis_command::command_buffer::CommandBuffer;
use ignis_command::state::StaticPipelineState;
use ignis_descriptors::binding_table::{BindingTable, MAX_DESCRIPTOR_SETS};
use ignis_descriptors::cache::DescriptorSetCache;

use crate::framebuffer_cache::FramebufferCache;
use crate::pipeline::{PipelineCompiler, VertexInputLayout};
use crate::program::Program;
use crate::render_pass::{RenderPassCache, RenderPassInfo};

/// Errors from draw context operations.
#[derive(Debug, Error)]
pub enum DrawError {
    /// A Vulkan operation failed.
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
    /// Not inside a render pass when attempting to draw.
    #[error("not inside a render pass")]
    NotInRenderPass,
}

/// Shared pipeline and caching resources for a frame.
///
/// Holds the mutable references to the pipeline compiler, render pass cache,
/// framebuffer cache, and descriptor set caches. Created once per frame and
/// passed to [`DrawContext`] for each command buffer.
pub struct FrameResources {
    /// Pipeline compiler for lazy pipeline lookup/compilation.
    pub pipeline_compiler: PipelineCompiler,
    /// Render pass cache for auto-creating render passes.
    pub render_pass_cache: RenderPassCache,
    /// Framebuffer cache for auto-creating framebuffers.
    pub framebuffer_cache: FramebufferCache,
    /// Per-frame descriptor set caches (one per set index).
    pub descriptor_caches: [DescriptorSetCache; MAX_DESCRIPTOR_SETS],
}

impl FrameResources {
    /// Create new frame resources with the given `VkPipelineCache`.
    pub fn new(pipeline_cache: vk::PipelineCache) -> Self {
        Self {
            pipeline_compiler: PipelineCompiler::new(pipeline_cache),
            render_pass_cache: RenderPassCache::new(),
            framebuffer_cache: FramebufferCache::new(),
            descriptor_caches: std::array::from_fn(|_| DescriptorSetCache::new()),
        }
    }

    /// Reset per-frame caches. Call at the start of each frame.
    ///
    /// Note: Framebuffer cache is NOT reset per-frame — framebuffers are
    /// long-lived and only need to be invalidated on swapchain recreation.
    pub fn reset_frame(&mut self, _device: &ash::Device) {
        for cache in &mut self.descriptor_caches {
            cache.reset();
        }
    }

    /// Destroy all Vulkan resources.
    pub fn destroy(&mut self, device: &ash::Device) {
        self.pipeline_compiler.destroy(device);
        self.render_pass_cache.destroy(device);
        self.framebuffer_cache.destroy(device);
        // Descriptor set caches don't own Vulkan resources (pools are in allocators)
    }
}

/// High-level draw context integrating command buffers, descriptors, and pipelines.
///
/// Wraps a [`CommandBuffer`] and provides Granite-style convenience: set your
/// pipeline state, bind your resources, and call `draw()` — the context
/// automatically resolves the pipeline, flushes dirty descriptor sets, and
/// issues the Vulkan commands.
///
/// # Usage
///
/// ```ignore
/// let mut ctx = DrawContext::new(&mut cmd, &device, &mut resources);
/// ctx.begin_render_pass(&rp_info, extent, &clear_values, &image_views)?;
/// ctx.set_depth_test(true, true);
/// ctx.set_texture(0, 0, view, sampler);
/// ctx.draw(&mut program, &vertex_layout, 3, 1, 0, 0)?;
/// ctx.end_render_pass();
/// ```
pub struct DrawContext<'a> {
    cmd: &'a mut CommandBuffer,
    device: &'a ash::Device,
    resources: &'a mut FrameResources,

    // Mutable state
    bindings: BindingTable,
    state: StaticPipelineState,
    bound_pipeline: vk::Pipeline,
    bound_program_hash: u64,

    // Render pass state
    in_render_pass: bool,
    current_render_pass: vk::RenderPass,
    current_render_pass_hash: u64,
    current_subpass: u32,
}

impl<'a> DrawContext<'a> {
    /// Create a new draw context wrapping the given command buffer.
    pub fn new(
        cmd: &'a mut CommandBuffer,
        device: &'a ash::Device,
        resources: &'a mut FrameResources,
    ) -> Self {
        Self {
            cmd,
            device,
            resources,
            bindings: BindingTable::new(),
            state: StaticPipelineState::default(),
            bound_pipeline: vk::Pipeline::null(),
            bound_program_hash: 0,
            in_render_pass: false,
            current_render_pass: vk::RenderPass::null(),
            current_render_pass_hash: 0,
            current_subpass: 0,
        }
    }

    // ---- Render pass management ----

    /// Begin a render pass, auto-creating the `VkRenderPass` and `VkFramebuffer`.
    ///
    /// The render pass is looked up or created via the [`RenderPassCache`].
    /// The framebuffer is looked up or created via the [`FramebufferCache`].
    pub fn begin_render_pass(
        &mut self,
        info: &RenderPassInfo,
        extent: vk::Extent2D,
        clear_values: &[vk::ClearValue],
        attachments: &[vk::ImageView],
    ) -> Result<(), DrawError> {
        let rp_hash = RenderPassCache::compatible_hash(info);
        let render_pass = self
            .resources
            .render_pass_cache
            .get_or_create(self.device, info)?;
        let framebuffer = self.resources.framebuffer_cache.get_or_create(
            self.device,
            render_pass,
            attachments,
            extent.width,
            extent.height,
        )?;

        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };

        self.cmd
            .begin_render_pass(render_pass, framebuffer, render_area, clear_values);

        // Set viewport and scissor to match the render area
        self.cmd.set_viewport(
            0,
            &[vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
        self.cmd.set_scissor(0, &[render_area]);

        self.in_render_pass = true;
        self.current_render_pass = render_pass;
        self.current_render_pass_hash = rp_hash;
        self.current_subpass = 0;

        // Invalidate bound pipeline since render pass changed
        self.bound_pipeline = vk::Pipeline::null();

        Ok(())
    }

    /// End the current render pass.
    pub fn end_render_pass(&mut self) {
        if self.in_render_pass {
            self.cmd.end_render_pass();
            self.in_render_pass = false;
            self.current_render_pass = vk::RenderPass::null();
        }
    }

    // ---- Pipeline state setters ----

    /// Set the primitive topology.
    pub fn set_topology(&mut self, topology: vk::PrimitiveTopology) {
        self.state.topology = topology;
    }

    /// Set the face culling mode.
    pub fn set_cull_mode(&mut self, cull_mode: vk::CullModeFlags) {
        self.state.cull_mode = cull_mode;
    }

    /// Set the front face winding order.
    pub fn set_front_face(&mut self, front_face: vk::FrontFace) {
        self.state.front_face = front_face;
    }

    /// Set the polygon rasterization mode.
    pub fn set_polygon_mode(&mut self, mode: vk::PolygonMode) {
        self.state.polygon_mode = mode;
    }

    /// Set depth test and depth write enables.
    pub fn set_depth_test(&mut self, test: bool, write: bool) {
        self.state.depth_test = test;
        self.state.depth_write = write;
    }

    /// Set the depth comparison operator.
    pub fn set_depth_compare(&mut self, op: vk::CompareOp) {
        self.state.depth_compare = op;
    }

    /// Set whether stencil testing is enabled.
    pub fn set_stencil_test(&mut self, enable: bool) {
        self.state.stencil_test = enable;
    }

    /// Set blending state for the first color attachment.
    pub fn set_blend(
        &mut self,
        enable: bool,
        src_color: vk::BlendFactor,
        dst_color: vk::BlendFactor,
        color_op: vk::BlendOp,
        src_alpha: vk::BlendFactor,
        dst_alpha: vk::BlendFactor,
        alpha_op: vk::BlendOp,
    ) {
        self.state.blend_enable = enable;
        self.state.src_color_blend = src_color;
        self.state.dst_color_blend = dst_color;
        self.state.color_blend_op = color_op;
        self.state.src_alpha_blend = src_alpha;
        self.state.dst_alpha_blend = dst_alpha;
        self.state.alpha_blend_op = alpha_op;
    }

    /// Set the color write mask.
    pub fn set_color_write_mask(&mut self, mask: vk::ColorComponentFlags) {
        self.state.color_write_mask = mask;
    }

    /// Enable standard alpha blending (src_alpha, one_minus_src_alpha).
    pub fn set_alpha_blend(&mut self) {
        self.set_blend(
            true,
            vk::BlendFactor::SRC_ALPHA,
            vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            vk::BlendOp::ADD,
            vk::BlendFactor::ONE,
            vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            vk::BlendOp::ADD,
        );
    }

    /// Get a mutable reference to the full pipeline state.
    pub fn state_mut(&mut self) -> &mut StaticPipelineState {
        &mut self.state
    }

    /// Get a reference to the current pipeline state.
    pub fn state(&self) -> &StaticPipelineState {
        &self.state
    }

    // ---- Descriptor binding ----

    /// Bind a uniform buffer.
    pub fn set_uniform_buffer(
        &mut self,
        set: u32,
        binding: u32,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) {
        self.bindings
            .set_uniform_buffer(set, binding, buffer, offset, range);
    }

    /// Bind a storage buffer.
    pub fn set_storage_buffer(
        &mut self,
        set: u32,
        binding: u32,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) {
        self.bindings
            .set_storage_buffer(set, binding, buffer, offset, range);
    }

    /// Bind a combined image sampler (texture).
    pub fn set_texture(
        &mut self,
        set: u32,
        binding: u32,
        view: vk::ImageView,
        sampler: vk::Sampler,
    ) {
        self.bindings.set_texture(set, binding, view, sampler);
    }

    /// Bind a storage image.
    pub fn set_storage_image(&mut self, set: u32, binding: u32, view: vk::ImageView) {
        self.bindings.set_storage_image(set, binding, view);
    }

    /// Bind an input attachment.
    pub fn set_input_attachment(&mut self, set: u32, binding: u32, view: vk::ImageView) {
        self.bindings.set_input_attachment(set, binding, view);
    }

    /// Push constants from a raw byte slice.
    pub fn push_constants(
        &mut self,
        program: &Program,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        data: &[u8],
    ) {
        self.cmd
            .push_constants(program.pipeline_layout(), stage_flags, offset, data);
    }

    /// Push constants from a typed value.
    pub fn push_constants_typed<T: bytemuck::Pod>(
        &mut self,
        program: &Program,
        stage_flags: vk::ShaderStageFlags,
        data: &T,
    ) {
        self.cmd
            .push_constants_typed(program.pipeline_layout(), stage_flags, data);
    }

    // ---- Vertex input ----

    /// Bind vertex buffers.
    pub fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
    ) {
        self.cmd.bind_vertex_buffers(first_binding, buffers, offsets);
    }

    /// Bind an index buffer.
    pub fn bind_index_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        self.cmd.bind_index_buffer(buffer, offset, index_type);
    }

    // ---- Draw commands ----

    /// Issue a non-indexed draw call.
    ///
    /// Automatically resolves the pipeline from the current state and program,
    /// flushes dirty descriptor sets, and issues `vkCmdDraw`.
    pub fn draw(
        &mut self,
        program: &mut Program,
        vertex_layout: &VertexInputLayout,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<(), DrawError> {
        if !self.in_render_pass {
            return Err(DrawError::NotInRenderPass);
        }

        self.resolve_graphics_pipeline(program, vertex_layout)?;
        self.flush_descriptor_sets(program)?;

        self.cmd
            .draw(vertex_count, instance_count, first_vertex, first_instance);
        Ok(())
    }

    /// Issue an indexed draw call.
    ///
    /// Automatically resolves the pipeline and flushes descriptor sets.
    pub fn draw_indexed(
        &mut self,
        program: &mut Program,
        vertex_layout: &VertexInputLayout,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) -> Result<(), DrawError> {
        if !self.in_render_pass {
            return Err(DrawError::NotInRenderPass);
        }

        self.resolve_graphics_pipeline(program, vertex_layout)?;
        self.flush_descriptor_sets(program)?;

        self.cmd.draw_indexed(
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        );
        Ok(())
    }

    /// Issue a compute dispatch.
    ///
    /// Automatically resolves the compute pipeline and flushes descriptor sets.
    /// Must NOT be called inside a render pass.
    pub fn dispatch(
        &mut self,
        program: &mut Program,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) -> Result<(), DrawError> {
        let pipeline = self
            .resources
            .pipeline_compiler
            .get_or_compile_compute(self.device, program)?;

        if pipeline != self.bound_pipeline {
            self.cmd
                .bind_pipeline(vk::PipelineBindPoint::COMPUTE, pipeline);
            self.bound_pipeline = pipeline;
            self.bound_program_hash = program.layout_hash();
            // Program changed — rebind all descriptor sets
            self.bindings.mark_all_dirty();
        }

        self.flush_descriptor_sets(program)?;
        self.cmd
            .dispatch(group_count_x, group_count_y, group_count_z);
        Ok(())
    }

    // ---- Raw access ----

    /// Get a mutable reference to the underlying command buffer.
    ///
    /// Use this for operations not covered by `DrawContext` (e.g., manual
    /// barriers, copies, or custom render pass management).
    pub fn cmd(&mut self) -> &mut CommandBuffer {
        self.cmd
    }

    // ---- Internal ----

    fn resolve_graphics_pipeline(
        &mut self,
        program: &mut Program,
        vertex_layout: &VertexInputLayout,
    ) -> Result<(), DrawError> {
        let pipeline = self.resources.pipeline_compiler.get_or_compile_graphics(
            self.device,
            program,
            &self.state,
            self.current_render_pass,
            self.current_render_pass_hash,
            self.current_subpass,
            vertex_layout,
        )?;

        if pipeline != self.bound_pipeline {
            self.cmd
                .bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pipeline);
            self.bound_pipeline = pipeline;
        }

        // If program changed, rebind all descriptor sets
        if program.layout_hash() != self.bound_program_hash {
            self.bound_program_hash = program.layout_hash();
            self.bindings.mark_all_dirty();
            debug!(
                "Program changed to {:#x}, marking all descriptor sets dirty",
                program.layout_hash()
            );
        }

        Ok(())
    }

    fn flush_descriptor_sets(&mut self, program: &mut Program) -> Result<(), DrawError> {
        let dirty = self.bindings.dirty_sets();
        if dirty == 0 {
            return Ok(());
        }

        let bind_point = if self.in_render_pass {
            vk::PipelineBindPoint::GRAPHICS
        } else {
            vk::PipelineBindPoint::COMPUTE
        };

        for set_idx in 0..MAX_DESCRIPTOR_SETS as u32 {
            if dirty & (1 << set_idx) == 0 {
                continue;
            }

            let set_bindings = self.bindings.get_set(set_idx).clone();

            // Skip sets with no active bindings
            if set_bindings.is_empty() {
                self.bindings.clear_dirty(set_idx);
                continue;
            }

            let Some(allocator) = program.set_allocator_mut(set_idx as usize) else {
                // Program doesn't use this set — skip
                self.bindings.clear_dirty(set_idx);
                continue;
            };

            let cache = &mut self.resources.descriptor_caches[set_idx as usize];
            let descriptor_set = cache.get_or_allocate(self.device, allocator, &set_bindings)?;

            self.cmd.bind_descriptor_sets(
                bind_point,
                program.pipeline_layout(),
                set_idx,
                &[descriptor_set],
                &[],
            );

            self.bindings.clear_dirty(set_idx);
        }

        Ok(())
    }
}
