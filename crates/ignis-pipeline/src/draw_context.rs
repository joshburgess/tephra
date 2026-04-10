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
use ignis_descriptors::cache::{DescriptorSetCache, PreparedDescriptorWrites};

use crate::framebuffer_cache::FramebufferCache;
use crate::pipeline::{PipelineCompiler, VertexInputLayout};
use crate::program::Program;
use crate::render_pass::{AttachmentLoadOp, AttachmentStoreOp, RenderPassCache, RenderPassInfo};

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

/// Attachment description for dynamic rendering.
///
/// Used with [`DrawContext::begin_rendering`] to specify color and depth/stencil
/// attachments without creating a `VkRenderPass` or `VkFramebuffer`.
#[derive(Clone)]
pub struct RenderingAttachment {
    /// Image view for this attachment.
    pub view: vk::ImageView,
    /// Format of this attachment (used for pipeline key).
    pub format: vk::Format,
    /// Load operation at the start of rendering.
    pub load_op: AttachmentLoadOp,
    /// Store operation at the end of rendering.
    pub store_op: AttachmentStoreOp,
    /// Clear value (used when load_op is Clear).
    pub clear_value: vk::ClearValue,
}

/// Snapshot of draw context state for save/restore.
///
/// Created by [`DrawContext::save_state`] and applied by [`DrawContext::restore_state`].
/// Captures the current pipeline state, descriptor bindings, and bound pipeline
/// so you can temporarily change state (e.g., for UI overlay or debug drawing)
/// and restore the previous state afterward.
#[derive(Clone)]
pub struct SavedDrawState {
    /// Descriptor binding table snapshot.
    pub bindings: BindingTable,
    /// Static pipeline state snapshot.
    pub state: StaticPipelineState,
    /// Currently bound pipeline handle.
    pub bound_pipeline: vk::Pipeline,
    /// Hash of the currently bound program.
    pub bound_program_hash: u64,
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

    // Dynamic rendering state
    use_dynamic_rendering: bool,
    dynamic_color_formats: Vec<vk::Format>,
    dynamic_depth_format: vk::Format,
    dynamic_stencil_format: vk::Format,

    // Push descriptor state
    push_descriptor_device: Option<&'a ash::khr::push_descriptor::Device>,
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
            use_dynamic_rendering: false,
            dynamic_color_formats: Vec::new(),
            dynamic_depth_format: vk::Format::UNDEFINED,
            dynamic_stencil_format: vk::Format::UNDEFINED,
            push_descriptor_device: None,
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
        if self.in_render_pass && !self.use_dynamic_rendering {
            self.cmd.end_render_pass();
            self.in_render_pass = false;
            self.current_render_pass = vk::RenderPass::null();
        }
    }

    // ---- Dynamic rendering (Vulkan 1.3 / VK_KHR_dynamic_rendering) ----

    /// Begin dynamic rendering without a `VkRenderPass` or `VkFramebuffer`.
    ///
    /// This is the modern alternative to [`begin_render_pass`](Self::begin_render_pass).
    /// Requires the device to support `VK_KHR_dynamic_rendering` or Vulkan 1.3.
    ///
    /// Attachment formats are extracted from the provided attachments and used
    /// for pipeline key hashing (via `VkPipelineRenderingCreateInfo`).
    pub fn begin_rendering(
        &mut self,
        extent: vk::Extent2D,
        color_attachments: &[RenderingAttachment],
        depth_attachment: Option<&RenderingAttachment>,
    ) -> Result<(), DrawError> {
        // Build Vulkan color attachment infos
        let vk_color_attachments: Vec<vk::RenderingAttachmentInfo<'_>> = color_attachments
            .iter()
            .map(|a| {
                let mut info = vk::RenderingAttachmentInfo::default()
                    .image_view(a.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(a.load_op.into())
                    .store_op(a.store_op.into());
                if a.load_op == AttachmentLoadOp::Clear {
                    info = info.clear_value(a.clear_value);
                }
                info
            })
            .collect();

        // Build depth attachment info
        let vk_depth_attachment;
        let vk_stencil_attachment;

        if let Some(depth) = depth_attachment {
            let is_stencil_format = matches!(
                depth.format,
                vk::Format::D16_UNORM_S8_UINT
                    | vk::Format::D24_UNORM_S8_UINT
                    | vk::Format::D32_SFLOAT_S8_UINT
            );

            let mut info = vk::RenderingAttachmentInfo::default()
                .image_view(depth.view)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(depth.load_op.into())
                .store_op(depth.store_op.into());
            if depth.load_op == AttachmentLoadOp::Clear {
                info = info.clear_value(depth.clear_value);
            }
            vk_depth_attachment = Some(info);

            // If the depth format includes stencil, use the same view for stencil
            if is_stencil_format {
                let mut stencil_info = vk::RenderingAttachmentInfo::default()
                    .image_view(depth.view)
                    .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .load_op(depth.load_op.into())
                    .store_op(depth.store_op.into());
                if depth.load_op == AttachmentLoadOp::Clear {
                    stencil_info = stencil_info.clear_value(depth.clear_value);
                }
                vk_stencil_attachment = Some(stencil_info);
            } else {
                vk_stencil_attachment = None;
            }
        } else {
            vk_depth_attachment = None;
            vk_stencil_attachment = None;
        }

        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };

        let mut rendering_info = vk::RenderingInfo::default()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(&vk_color_attachments);

        if let Some(ref depth) = vk_depth_attachment {
            rendering_info = rendering_info.depth_attachment(depth);
        }
        if let Some(ref stencil) = vk_stencil_attachment {
            rendering_info = rendering_info.stencil_attachment(stencil);
        }

        self.cmd.begin_rendering(&rendering_info);

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

        // Store format info for pipeline key
        self.dynamic_color_formats = color_attachments.iter().map(|a| a.format).collect();
        self.dynamic_depth_format = depth_attachment
            .map(|a| a.format)
            .unwrap_or(vk::Format::UNDEFINED);
        self.dynamic_stencil_format = if depth_attachment.is_some_and(|a| {
            matches!(
                a.format,
                vk::Format::D16_UNORM_S8_UINT
                    | vk::Format::D24_UNORM_S8_UINT
                    | vk::Format::D32_SFLOAT_S8_UINT
            )
        }) {
            self.dynamic_depth_format
        } else {
            vk::Format::UNDEFINED
        };

        self.in_render_pass = true;
        self.use_dynamic_rendering = true;
        self.bound_pipeline = vk::Pipeline::null();

        Ok(())
    }

    /// End the current dynamic rendering scope.
    pub fn end_rendering(&mut self) {
        if self.in_render_pass && self.use_dynamic_rendering {
            self.cmd.end_rendering();
            self.in_render_pass = false;
            self.use_dynamic_rendering = false;
            self.dynamic_color_formats.clear();
            self.dynamic_depth_format = vk::Format::UNDEFINED;
            self.dynamic_stencil_format = vk::Format::UNDEFINED;
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

    /// Enable or disable depth bias (constant/clamp/slope are dynamic state).
    pub fn set_depth_bias_enable(&mut self, enable: bool) {
        self.state.depth_bias_enable = enable;
    }

    /// Set front-face stencil operations.
    pub fn set_stencil_front(
        &mut self,
        fail_op: vk::StencilOp,
        pass_op: vk::StencilOp,
        depth_fail_op: vk::StencilOp,
        compare_op: vk::CompareOp,
    ) {
        self.state.stencil_front = ignis_command::state::StencilFaceState {
            fail_op,
            depth_fail_op,
            pass_op,
            compare_op,
        };
    }

    /// Set back-face stencil operations.
    pub fn set_stencil_back(
        &mut self,
        fail_op: vk::StencilOp,
        pass_op: vk::StencilOp,
        depth_fail_op: vk::StencilOp,
        compare_op: vk::CompareOp,
    ) {
        self.state.stencil_back = ignis_command::state::StencilFaceState {
            fail_op,
            depth_fail_op,
            pass_op,
            compare_op,
        };
    }

    /// Set the rasterization sample count.
    pub fn set_rasterization_samples(&mut self, samples: vk::SampleCountFlags) {
        self.state.rasterization_samples = samples;
    }

    /// Enable or disable alpha-to-coverage multisampling.
    pub fn set_alpha_to_coverage(&mut self, enable: bool) {
        self.state.alpha_to_coverage = enable;
    }

    /// Enable or disable conservative rasterization.
    ///
    /// When enabled, rasterization uses overestimation mode. Requires
    /// `VK_EXT_conservative_rasterization`.
    pub fn set_conservative_rasterization(&mut self, enable: bool) {
        self.state.conservative_rasterization = enable;
    }

    /// Set the required subgroup size for compute/mesh/task shader stages.
    ///
    /// The required size is `1 << log2`. Pass 0 to use the driver default.
    /// Requires `VK_EXT_subgroup_size_control`.
    pub fn set_subgroup_size_log2(&mut self, log2: u8) {
        self.state.subgroup_size_log2 = log2;
    }

    /// Set a specialization constant value.
    ///
    /// `index` must be in `0..8`. The value is reinterpreted as float/int/bool
    /// depending on the shader's spec constant type.
    pub fn set_specialization_constant(&mut self, index: u32, value: u32) {
        assert!(index < 8, "specialization constant index must be < 8");
        self.state.spec_constant_mask |= 1 << index;
        self.state.spec_constants[index as usize] = value;
    }

    /// Clear all specialization constants.
    pub fn clear_specialization_constants(&mut self) {
        self.state.spec_constant_mask = 0;
        self.state.spec_constants = [0; 8];
    }

    /// Get a mutable reference to the full pipeline state.
    pub fn state_mut(&mut self) -> &mut StaticPipelineState {
        &mut self.state
    }

    /// Get a reference to the current pipeline state.
    pub fn state(&self) -> &StaticPipelineState {
        &self.state
    }

    // ---- State save/restore ----

    /// Snapshot the current draw state (bindings, pipeline state, bound pipeline).
    ///
    /// Returns a [`SavedDrawState`] that can be passed to [`restore_state`](Self::restore_state).
    /// Useful for UI overlays, debug draws, or any pattern that temporarily
    /// changes draw state and needs to restore it afterward.
    pub fn save_state(&self) -> SavedDrawState {
        SavedDrawState {
            bindings: self.bindings.clone(),
            state: self.state.clone(),
            bound_pipeline: self.bound_pipeline,
            bound_program_hash: self.bound_program_hash,
        }
    }

    /// Restore a previously saved draw state.
    ///
    /// Marks all descriptor sets as dirty so they will be re-flushed on the next draw.
    pub fn restore_state(&mut self, saved: &SavedDrawState) {
        self.bindings = saved.bindings.clone();
        self.state = saved.state.clone();
        self.bound_pipeline = saved.bound_pipeline;
        self.bound_program_hash = saved.bound_program_hash;
        self.bindings.mark_all_dirty();
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

    /// Bind an acceleration structure for ray queries.
    pub fn set_acceleration_structure(
        &mut self,
        set: u32,
        binding: u32,
        handle: vk::AccelerationStructureKHR,
    ) {
        self.bindings
            .set_acceleration_structure(set, binding, handle);
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

    /// Issue an indirect draw call.
    ///
    /// Automatically resolves the pipeline and flushes descriptor sets.
    pub fn draw_indirect(
        &mut self,
        program: &mut Program,
        vertex_layout: &VertexInputLayout,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> Result<(), DrawError> {
        if !self.in_render_pass {
            return Err(DrawError::NotInRenderPass);
        }

        self.resolve_graphics_pipeline(program, vertex_layout)?;
        self.flush_descriptor_sets(program)?;

        self.cmd.draw_indirect(buffer, offset, draw_count, stride);
        Ok(())
    }

    /// Issue an indirect indexed draw call.
    ///
    /// Automatically resolves the pipeline and flushes descriptor sets.
    pub fn draw_indexed_indirect(
        &mut self,
        program: &mut Program,
        vertex_layout: &VertexInputLayout,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> Result<(), DrawError> {
        if !self.in_render_pass {
            return Err(DrawError::NotInRenderPass);
        }

        self.resolve_graphics_pipeline(program, vertex_layout)?;
        self.flush_descriptor_sets(program)?;

        self.cmd
            .draw_indexed_indirect(buffer, offset, draw_count, stride);
        Ok(())
    }

    /// Issue a mesh shader draw call.
    ///
    /// Automatically resolves the pipeline (with no vertex input state) and
    /// flushes descriptor sets. Requires `VK_EXT_mesh_shader`.
    pub fn draw_mesh_tasks(
        &mut self,
        program: &mut Program,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) -> Result<(), DrawError> {
        if !self.in_render_pass {
            return Err(DrawError::NotInRenderPass);
        }

        // Mesh shader pipelines use an empty vertex layout
        let empty_layout = VertexInputLayout::default();
        self.resolve_graphics_pipeline(program, &empty_layout)?;
        self.flush_descriptor_sets(program)?;

        self.cmd
            .draw_mesh_tasks(group_count_x, group_count_y, group_count_z);
        Ok(())
    }

    /// Issue an indirect mesh shader draw call.
    ///
    /// Automatically resolves the pipeline and flushes descriptor sets.
    /// Requires `VK_EXT_mesh_shader`.
    pub fn draw_mesh_tasks_indirect(
        &mut self,
        program: &mut Program,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) -> Result<(), DrawError> {
        if !self.in_render_pass {
            return Err(DrawError::NotInRenderPass);
        }

        let empty_layout = VertexInputLayout::default();
        self.resolve_graphics_pipeline(program, &empty_layout)?;
        self.flush_descriptor_sets(program)?;

        self.cmd
            .draw_mesh_tasks_indirect(buffer, offset, draw_count, stride);
        Ok(())
    }

    /// Issue an indirect mesh shader draw call with a count buffer.
    ///
    /// Automatically resolves the pipeline and flushes descriptor sets.
    /// Requires `VK_EXT_mesh_shader`.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_mesh_tasks_indirect_count(
        &mut self,
        program: &mut Program,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        count_buffer: vk::Buffer,
        count_offset: vk::DeviceSize,
        max_draw_count: u32,
        stride: u32,
    ) -> Result<(), DrawError> {
        if !self.in_render_pass {
            return Err(DrawError::NotInRenderPass);
        }

        let empty_layout = VertexInputLayout::default();
        self.resolve_graphics_pipeline(program, &empty_layout)?;
        self.flush_descriptor_sets(program)?;

        self.cmd.draw_mesh_tasks_indirect_count(
            buffer,
            offset,
            count_buffer,
            count_offset,
            max_draw_count,
            stride,
        );
        Ok(())
    }

    /// Issue an indirect compute dispatch.
    ///
    /// Automatically resolves the compute pipeline and flushes descriptor sets.
    /// Must NOT be called inside a render pass.
    pub fn dispatch_indirect(
        &mut self,
        program: &mut Program,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
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
            self.bindings.mark_all_dirty();
        }

        self.flush_descriptor_sets(program)?;
        self.cmd.dispatch_indirect(buffer, offset);
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

    /// Set the push descriptor extension loader.
    ///
    /// Required for programs created with [`Program::create_with_push_descriptors`].
    /// The loader can be obtained from [`Context::push_descriptor_device()`](ignis_core::context::Context::push_descriptor_device).
    pub fn set_push_descriptor_device(
        &mut self,
        device: &'a ash::khr::push_descriptor::Device,
    ) {
        self.push_descriptor_device = Some(device);
    }

    /// Bind a persistent bindless descriptor set at the given set index.
    ///
    /// This bypasses the normal descriptor binding/caching path. The bindless
    /// set is bound directly to the command buffer and remains valid for all
    /// subsequent draw/dispatch calls using the same pipeline layout.
    ///
    /// Typically bound at a high set index (e.g., set 3) to avoid conflicts
    /// with per-draw descriptor sets at lower indices.
    pub fn bind_bindless_set(
        &mut self,
        pipeline_layout: vk::PipelineLayout,
        set_index: u32,
        bindless_set: vk::DescriptorSet,
    ) {
        let bind_point = if self.in_render_pass {
            vk::PipelineBindPoint::GRAPHICS
        } else {
            vk::PipelineBindPoint::COMPUTE
        };

        self.cmd
            .bind_descriptor_sets(bind_point, pipeline_layout, set_index, &[bindless_set], &[]);
    }

    // ---- Internal ----

    fn resolve_graphics_pipeline(
        &mut self,
        program: &mut Program,
        vertex_layout: &VertexInputLayout,
    ) -> Result<(), DrawError> {
        let pipeline = if self.use_dynamic_rendering {
            self.resources
                .pipeline_compiler
                .get_or_compile_graphics_dynamic(
                    self.device,
                    program,
                    &self.state,
                    &self.dynamic_color_formats,
                    self.dynamic_depth_format,
                    self.dynamic_stencil_format,
                    vertex_layout,
                )?
        } else {
            self.resources.pipeline_compiler.get_or_compile_graphics(
                self.device,
                program,
                &self.state,
                self.current_render_pass,
                self.current_render_pass_hash,
                self.current_subpass,
                vertex_layout,
            )?
        };

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

            // Push descriptor path: write descriptors inline on the command buffer
            if program.is_push_descriptor_set(set_idx) {
                if let Some(push_device) = self.push_descriptor_device {
                    let prepared = PreparedDescriptorWrites::from_bindings(&set_bindings);
                    if !prepared.is_empty() {
                        let write_set = prepared.build_writes(vk::DescriptorSet::null());
                        // SAFETY: command buffer, pipeline layout, and descriptor data are valid.
                        unsafe {
                            push_device.cmd_push_descriptor_set(
                                self.cmd.raw(),
                                bind_point,
                                program.pipeline_layout(),
                                set_idx,
                                write_set.writes(),
                            );
                        }
                    }
                    self.bindings.clear_dirty(set_idx);
                    continue;
                }
                // Fall through to regular path if push descriptor device not set
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
