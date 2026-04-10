//! Lazy pipeline compilation with hash-and-cache.
//!
//! On draw/dispatch, the pipeline key is hashed from the current program,
//! static render state, compatible render pass, and vertex input layout.
//! Cache misses trigger pipeline compilation via `vkCreateGraphicsPipelines`.

use std::hash::{Hash, Hasher};

use ash::vk;
use rustc_hash::{FxHashMap, FxHasher};

use ignis_command::state::StaticPipelineState;

use crate::fossilize::FossilizeRecorder;
use crate::program::Program;

/// Key for looking up a compiled pipeline (legacy render pass path).
#[derive(Clone, PartialEq, Eq, Hash)]
struct PipelineKey {
    program_hash: u64,
    static_state: StaticPipelineState,
    render_pass_hash: u64,
    subpass: u32,
    vertex_layout_hash: u64,
}

/// Key for looking up a compiled pipeline (dynamic rendering path).
#[derive(Clone, PartialEq, Eq, Hash)]
struct DynamicPipelineKey {
    program_hash: u64,
    static_state: StaticPipelineState,
    color_formats: Vec<vk::Format>,
    depth_format: vk::Format,
    stencil_format: vk::Format,
    vertex_layout_hash: u64,
}

/// Vertex buffer binding description for pipeline creation.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct VertexBinding {
    /// Binding slot index.
    pub binding: u32,
    /// Stride in bytes between consecutive elements.
    pub stride: u32,
    /// Per-vertex or per-instance stepping.
    pub input_rate: vk::VertexInputRate,
}

/// Vertex attribute description for pipeline creation.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct VertexAttribute {
    /// Shader input location.
    pub location: u32,
    /// Binding slot this attribute reads from.
    pub binding: u32,
    /// Data format.
    pub format: vk::Format,
    /// Byte offset within the vertex.
    pub offset: u32,
}

/// Full vertex input layout for pipeline creation.
///
/// Combines vertex buffer bindings with per-attribute descriptions.
/// The shader reflection provides locations and formats; the application
/// supplies bindings, strides, and offsets.
#[derive(Debug, Clone, Default)]
pub struct VertexInputLayout {
    /// Vertex buffer bindings.
    pub bindings: Vec<VertexBinding>,
    /// Vertex attributes.
    pub attributes: Vec<VertexAttribute>,
}

impl VertexInputLayout {
    /// Compute a hash of the vertex input layout for pipeline key comparison.
    pub fn compute_hash(&self) -> u64 {
        let mut hasher = FxHasher::default();
        self.bindings.hash(&mut hasher);
        self.attributes.hash(&mut hasher);
        hasher.finish()
    }
}

/// Compiles and caches graphics and compute pipelines.
///
/// Pipeline keys are hashed from the current rendering state. Cache misses
/// trigger `vkCreateGraphicsPipelines` or `vkCreateComputePipelines`.
pub struct PipelineCompiler {
    graphics_cache: FxHashMap<u64, vk::Pipeline>,
    dynamic_graphics_cache: FxHashMap<u64, vk::Pipeline>,
    compute_cache: FxHashMap<u64, vk::Pipeline>,
    pipeline_cache: vk::PipelineCache,
    fossilize_recorder: Option<FossilizeRecorder>,
}

impl PipelineCompiler {
    /// Create a new pipeline compiler with the given `VkPipelineCache`.
    pub fn new(pipeline_cache: vk::PipelineCache) -> Self {
        Self {
            graphics_cache: FxHashMap::default(),
            dynamic_graphics_cache: FxHashMap::default(),
            compute_cache: FxHashMap::default(),
            pipeline_cache,
            fossilize_recorder: None,
        }
    }

    /// Attach a Fossilize recorder for pipeline state serialization.
    ///
    /// When set, each newly compiled pipeline's state is recorded for later
    /// replay to warm up the pipeline cache on subsequent launches.
    pub fn set_fossilize_recorder(&mut self, recorder: FossilizeRecorder) {
        self.fossilize_recorder = Some(recorder);
    }

    /// Access the Fossilize recorder, if attached.
    pub fn fossilize_recorder(&self) -> Option<&FossilizeRecorder> {
        self.fossilize_recorder.as_ref()
    }

    /// Look up or compile a graphics pipeline.
    #[allow(clippy::too_many_arguments)]
    pub fn get_or_compile_graphics(
        &mut self,
        device: &ash::Device,
        program: &Program,
        state: &StaticPipelineState,
        render_pass: vk::RenderPass,
        render_pass_hash: u64,
        subpass: u32,
        vertex_layout: &VertexInputLayout,
    ) -> Result<vk::Pipeline, vk::Result> {
        let key = PipelineKey {
            program_hash: program.layout_hash(),
            static_state: state.clone(),
            render_pass_hash,
            subpass,
            vertex_layout_hash: vertex_layout.compute_hash(),
        };

        let key_hash = {
            let mut hasher = FxHasher::default();
            key.hash(&mut hasher);
            hasher.finish()
        };

        if let Some(&pipeline) = self.graphics_cache.get(&key_hash) {
            return Ok(pipeline);
        }

        let pipeline =
            self.compile_graphics(device, program, state, render_pass, subpass, vertex_layout)?;

        log::debug!(
            "Compiled graphics pipeline (total cached: {})",
            self.graphics_cache.len() + 1
        );

        if let Some(ref mut recorder) = self.fossilize_recorder {
            recorder.record_graphics_pipeline(key_hash, key_hash.to_le_bytes().to_vec());
        }

        self.graphics_cache.insert(key_hash, pipeline);
        Ok(pipeline)
    }

    /// Look up or compile a graphics pipeline using dynamic rendering.
    ///
    /// Instead of a `VkRenderPass`, the pipeline is created with
    /// `VkPipelineRenderingCreateInfo` specifying attachment formats directly.
    /// Requires `VK_KHR_dynamic_rendering` or Vulkan 1.3.
    #[allow(clippy::too_many_arguments)]
    pub fn get_or_compile_graphics_dynamic(
        &mut self,
        device: &ash::Device,
        program: &Program,
        state: &StaticPipelineState,
        color_formats: &[vk::Format],
        depth_format: vk::Format,
        stencil_format: vk::Format,
        vertex_layout: &VertexInputLayout,
    ) -> Result<vk::Pipeline, vk::Result> {
        let key = DynamicPipelineKey {
            program_hash: program.layout_hash(),
            static_state: state.clone(),
            color_formats: color_formats.to_vec(),
            depth_format,
            stencil_format,
            vertex_layout_hash: vertex_layout.compute_hash(),
        };

        let key_hash = {
            let mut hasher = FxHasher::default();
            key.hash(&mut hasher);
            hasher.finish()
        };

        if let Some(&pipeline) = self.dynamic_graphics_cache.get(&key_hash) {
            return Ok(pipeline);
        }

        let pipeline = self.compile_graphics_dynamic(
            device,
            program,
            state,
            color_formats,
            depth_format,
            stencil_format,
            vertex_layout,
        )?;

        log::debug!(
            "Compiled dynamic rendering pipeline (total cached: {})",
            self.dynamic_graphics_cache.len() + 1
        );

        if let Some(ref mut recorder) = self.fossilize_recorder {
            recorder.record_graphics_pipeline(key_hash, key_hash.to_le_bytes().to_vec());
        }

        self.dynamic_graphics_cache.insert(key_hash, pipeline);
        Ok(pipeline)
    }

    /// Look up or compile a compute pipeline.
    pub fn get_or_compile_compute(
        &mut self,
        device: &ash::Device,
        program: &Program,
    ) -> Result<vk::Pipeline, vk::Result> {
        let key_hash = program.layout_hash();

        if let Some(&pipeline) = self.compute_cache.get(&key_hash) {
            return Ok(pipeline);
        }

        let pipeline = self.compile_compute(device, program)?;

        log::debug!(
            "Compiled compute pipeline (total cached: {})",
            self.compute_cache.len() + 1
        );

        if let Some(ref mut recorder) = self.fossilize_recorder {
            recorder.record_compute_pipeline(key_hash, key_hash.to_le_bytes().to_vec());
        }

        self.compute_cache.insert(key_hash, pipeline);
        Ok(pipeline)
    }

    /// Destroy all cached pipelines.
    ///
    /// Flushes the Fossilize recorder (if attached) before destroying pipelines.
    pub fn destroy(&mut self, device: &ash::Device) {
        if let Some(ref recorder) = self.fossilize_recorder {
            if let Err(e) = recorder.flush() {
                log::warn!("Failed to flush Fossilize recorder: {e}");
            }
        }

        for (_, pipeline) in self.graphics_cache.drain() {
            // SAFETY: device is valid, pipeline is valid, GPU is idle.
            unsafe {
                device.destroy_pipeline(pipeline, None);
            }
        }
        for (_, pipeline) in self.dynamic_graphics_cache.drain() {
            // SAFETY: device is valid, pipeline is valid, GPU is idle.
            unsafe {
                device.destroy_pipeline(pipeline, None);
            }
        }
        for (_, pipeline) in self.compute_cache.drain() {
            // SAFETY: device is valid, pipeline is valid, GPU is idle.
            unsafe {
                device.destroy_pipeline(pipeline, None);
            }
        }
    }

    fn compile_graphics(
        &self,
        device: &ash::Device,
        program: &Program,
        state: &StaticPipelineState,
        render_pass: vk::RenderPass,
        subpass: u32,
        vertex_layout: &VertexInputLayout,
    ) -> Result<vk::Pipeline, vk::Result> {
        let shaders: Vec<_> = program
            .shaders()
            .iter()
            .map(|s| (s.module, s.stage))
            .collect();
        build_graphics_pipeline(
            device,
            self.pipeline_cache,
            &shaders,
            program.pipeline_layout(),
            state,
            render_pass,
            subpass,
            vertex_layout,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_graphics_dynamic(
        &self,
        device: &ash::Device,
        program: &Program,
        state: &StaticPipelineState,
        color_formats: &[vk::Format],
        depth_format: vk::Format,
        stencil_format: vk::Format,
        vertex_layout: &VertexInputLayout,
    ) -> Result<vk::Pipeline, vk::Result> {
        let shaders: Vec<_> = program
            .shaders()
            .iter()
            .map(|s| (s.module, s.stage))
            .collect();
        build_dynamic_graphics_pipeline(
            device,
            self.pipeline_cache,
            &shaders,
            program.pipeline_layout(),
            state,
            color_formats,
            depth_format,
            stencil_format,
            vertex_layout,
        )
    }

    fn compile_compute(
        &self,
        device: &ash::Device,
        program: &Program,
    ) -> Result<vk::Pipeline, vk::Result> {
        let shader = program
            .shaders()
            .iter()
            .find(|s| s.stage == vk::ShaderStageFlags::COMPUTE)
            .expect("compute program must have a compute shader stage");

        build_compute_pipeline(
            device,
            self.pipeline_cache,
            shader.module,
            program.pipeline_layout(),
        )
    }
}

// --- Standalone pipeline compilation functions ---
// Used by both PipelineCompiler (sync) and AsyncPipelineCompiler (async).

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_graphics_pipeline(
    device: &ash::Device,
    pipeline_cache: vk::PipelineCache,
    shaders: &[(vk::ShaderModule, vk::ShaderStageFlags)],
    pipeline_layout: vk::PipelineLayout,
    state: &StaticPipelineState,
    render_pass: vk::RenderPass,
    subpass: u32,
    vertex_layout: &VertexInputLayout,
) -> Result<vk::Pipeline, vk::Result> {
    let entry_name = c"main";

    // Detect mesh shader pipeline — no vertex input or input assembly state
    let is_mesh_pipeline = shaders
        .iter()
        .any(|(_, stage)| *stage == vk::ShaderStageFlags::MESH_EXT);

    let binding_descs: Vec<vk::VertexInputBindingDescription> = vertex_layout
        .bindings
        .iter()
        .map(|b| {
            vk::VertexInputBindingDescription::default()
                .binding(b.binding)
                .stride(b.stride)
                .input_rate(b.input_rate)
        })
        .collect();

    let attr_descs: Vec<vk::VertexInputAttributeDescription> = vertex_layout
        .attributes
        .iter()
        .map(|a| {
            vk::VertexInputAttributeDescription::default()
                .location(a.location)
                .binding(a.binding)
                .format(a.format)
                .offset(a.offset)
        })
        .collect();

    let vertex_input_ci = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&binding_descs)
        .vertex_attribute_descriptions(&attr_descs);

    let input_assembly_ci = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(state.topology)
        .primitive_restart_enable(state.primitive_restart);

    let viewport_ci = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let mut conservative_raster_ci =
        vk::PipelineRasterizationConservativeStateCreateInfoEXT::default()
            .conservative_rasterization_mode(vk::ConservativeRasterizationModeEXT::OVERESTIMATE);

    let mut rasterization_ci = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(state.polygon_mode)
        .cull_mode(state.cull_mode)
        .front_face(state.front_face)
        .depth_bias_enable(state.depth_bias_enable)
        .line_width(1.0);

    if state.conservative_rasterization {
        rasterization_ci = rasterization_ci.push_next(&mut conservative_raster_ci);
    }

    let multisample_ci = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(state.rasterization_samples)
        .sample_shading_enable(state.sample_shading)
        .alpha_to_coverage_enable(state.alpha_to_coverage)
        .alpha_to_one_enable(state.alpha_to_one);

    let depth_stencil_ci = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(state.depth_test)
        .depth_write_enable(state.depth_write)
        .depth_compare_op(state.depth_compare)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(state.stencil_test)
        .front(state.stencil_front.to_vk())
        .back(state.stencil_back.to_vk());

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(state.blend_enable)
        .src_color_blend_factor(state.src_color_blend)
        .dst_color_blend_factor(state.dst_color_blend)
        .color_blend_op(state.color_blend_op)
        .src_alpha_blend_factor(state.src_alpha_blend)
        .dst_alpha_blend_factor(state.dst_alpha_blend)
        .alpha_blend_op(state.alpha_blend_op)
        .color_write_mask(state.color_write_mask);

    let blend_attachments = [color_blend_attachment];

    let color_blend_ci = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(&blend_attachments);

    let mut dynamic_states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    if state.depth_bias_enable {
        dynamic_states.push(vk::DynamicState::DEPTH_BIAS);
    }
    if state.stencil_test {
        dynamic_states.push(vk::DynamicState::STENCIL_COMPARE_MASK);
        dynamic_states.push(vk::DynamicState::STENCIL_WRITE_MASK);
        dynamic_states.push(vk::DynamicState::STENCIL_REFERENCE);
    }

    let dynamic_state_ci =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    // Build specialization info if any constants are active
    let spec_map_entries: Vec<vk::SpecializationMapEntry> = (0..8u32)
        .filter(|i| state.spec_constant_mask & (1 << i) != 0)
        .map(|i| vk::SpecializationMapEntry {
            constant_id: i,
            offset: i * 4,
            size: 4,
        })
        .collect();

    let spec_info = if !spec_map_entries.is_empty() {
        Some(
            vk::SpecializationInfo::default()
                .map_entries(&spec_map_entries)
                .data(bytemuck::cast_slice(&state.spec_constants)),
        )
    } else {
        None
    };

    // Build required subgroup size info if set
    let mut subgroup_size_ci = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default();
    if state.subgroup_size_log2 > 0 {
        subgroup_size_ci.required_subgroup_size = 1u32 << state.subgroup_size_log2;
    }

    // Build stage_cis with specialization info and subgroup size attached
    let mut stage_cis_owned: Vec<vk::PipelineShaderStageCreateInfo<'_>> = shaders
        .iter()
        .map(|(module, stage)| {
            let mut ci = vk::PipelineShaderStageCreateInfo::default()
                .stage(*stage)
                .module(*module)
                .name(entry_name);
            if let Some(ref spec) = spec_info {
                ci = ci.specialization_info(spec);
            }
            ci
        })
        .collect();

    // Chain subgroup size requirement on compute/mesh/task stages
    if state.subgroup_size_log2 > 0 {
        for ci in &mut stage_cis_owned {
            if ci.stage == vk::ShaderStageFlags::COMPUTE
                || ci.stage == vk::ShaderStageFlags::MESH_EXT
                || ci.stage == vk::ShaderStageFlags::TASK_EXT
            {
                ci.p_next = &subgroup_size_ci as *const _ as *const std::ffi::c_void;
            }
        }
    }

    // Mesh shader pipelines must not have vertex input or input assembly state
    let mut pipeline_ci = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stage_cis_owned)
        .viewport_state(&viewport_ci)
        .rasterization_state(&rasterization_ci)
        .multisample_state(&multisample_ci)
        .depth_stencil_state(&depth_stencil_ci)
        .color_blend_state(&color_blend_ci)
        .dynamic_state(&dynamic_state_ci)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(subpass);

    if !is_mesh_pipeline {
        pipeline_ci = pipeline_ci
            .vertex_input_state(&vertex_input_ci)
            .input_assembly_state(&input_assembly_ci);
    }

    // SAFETY: device is valid, all pipeline create info is well-formed.
    let pipelines =
        unsafe { device.create_graphics_pipelines(pipeline_cache, &[pipeline_ci], None) }
            .map_err(|(_, err)| err)?;

    Ok(pipelines[0])
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_dynamic_graphics_pipeline(
    device: &ash::Device,
    pipeline_cache: vk::PipelineCache,
    shaders: &[(vk::ShaderModule, vk::ShaderStageFlags)],
    pipeline_layout: vk::PipelineLayout,
    state: &StaticPipelineState,
    color_formats: &[vk::Format],
    depth_format: vk::Format,
    stencil_format: vk::Format,
    vertex_layout: &VertexInputLayout,
) -> Result<vk::Pipeline, vk::Result> {
    let entry_name = c"main";

    // Detect mesh shader pipeline — no vertex input or input assembly state
    let is_mesh_pipeline = shaders
        .iter()
        .any(|(_, stage)| *stage == vk::ShaderStageFlags::MESH_EXT);

    let binding_descs: Vec<vk::VertexInputBindingDescription> = vertex_layout
        .bindings
        .iter()
        .map(|b| {
            vk::VertexInputBindingDescription::default()
                .binding(b.binding)
                .stride(b.stride)
                .input_rate(b.input_rate)
        })
        .collect();

    let attr_descs: Vec<vk::VertexInputAttributeDescription> = vertex_layout
        .attributes
        .iter()
        .map(|a| {
            vk::VertexInputAttributeDescription::default()
                .location(a.location)
                .binding(a.binding)
                .format(a.format)
                .offset(a.offset)
        })
        .collect();

    let vertex_input_ci = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&binding_descs)
        .vertex_attribute_descriptions(&attr_descs);

    let input_assembly_ci = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(state.topology)
        .primitive_restart_enable(state.primitive_restart);

    let viewport_ci = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let mut conservative_raster_ci =
        vk::PipelineRasterizationConservativeStateCreateInfoEXT::default()
            .conservative_rasterization_mode(vk::ConservativeRasterizationModeEXT::OVERESTIMATE);

    let mut rasterization_ci = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(state.polygon_mode)
        .cull_mode(state.cull_mode)
        .front_face(state.front_face)
        .depth_bias_enable(state.depth_bias_enable)
        .line_width(1.0);

    if state.conservative_rasterization {
        rasterization_ci = rasterization_ci.push_next(&mut conservative_raster_ci);
    }

    let multisample_ci = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(state.rasterization_samples)
        .sample_shading_enable(state.sample_shading)
        .alpha_to_coverage_enable(state.alpha_to_coverage)
        .alpha_to_one_enable(state.alpha_to_one);

    let depth_stencil_ci = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(state.depth_test)
        .depth_write_enable(state.depth_write)
        .depth_compare_op(state.depth_compare)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(state.stencil_test)
        .front(state.stencil_front.to_vk())
        .back(state.stencil_back.to_vk());

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(state.blend_enable)
        .src_color_blend_factor(state.src_color_blend)
        .dst_color_blend_factor(state.dst_color_blend)
        .color_blend_op(state.color_blend_op)
        .src_alpha_blend_factor(state.src_alpha_blend)
        .dst_alpha_blend_factor(state.dst_alpha_blend)
        .alpha_blend_op(state.alpha_blend_op)
        .color_write_mask(state.color_write_mask);

    let blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> =
        vec![color_blend_attachment; color_formats.len().max(1)];

    let color_blend_ci = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(&blend_attachments);

    let mut dynamic_states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    if state.depth_bias_enable {
        dynamic_states.push(vk::DynamicState::DEPTH_BIAS);
    }
    if state.stencil_test {
        dynamic_states.push(vk::DynamicState::STENCIL_COMPARE_MASK);
        dynamic_states.push(vk::DynamicState::STENCIL_WRITE_MASK);
        dynamic_states.push(vk::DynamicState::STENCIL_REFERENCE);
    }

    let dynamic_state_ci =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    // Build specialization info if any constants are active
    let spec_map_entries: Vec<vk::SpecializationMapEntry> = (0..8u32)
        .filter(|i| state.spec_constant_mask & (1 << i) != 0)
        .map(|i| vk::SpecializationMapEntry {
            constant_id: i,
            offset: i * 4,
            size: 4,
        })
        .collect();

    let spec_info = if !spec_map_entries.is_empty() {
        Some(
            vk::SpecializationInfo::default()
                .map_entries(&spec_map_entries)
                .data(bytemuck::cast_slice(&state.spec_constants)),
        )
    } else {
        None
    };

    // Build required subgroup size info if set
    let mut subgroup_size_ci = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default();
    if state.subgroup_size_log2 > 0 {
        subgroup_size_ci.required_subgroup_size = 1u32 << state.subgroup_size_log2;
    }

    // Build stage_cis with specialization info and subgroup size attached
    let mut stage_cis_owned: Vec<vk::PipelineShaderStageCreateInfo<'_>> = shaders
        .iter()
        .map(|(module, stage)| {
            let mut ci = vk::PipelineShaderStageCreateInfo::default()
                .stage(*stage)
                .module(*module)
                .name(entry_name);
            if let Some(ref spec) = spec_info {
                ci = ci.specialization_info(spec);
            }
            ci
        })
        .collect();

    // Chain subgroup size requirement on compute/mesh/task stages
    if state.subgroup_size_log2 > 0 {
        for ci in &mut stage_cis_owned {
            if ci.stage == vk::ShaderStageFlags::COMPUTE
                || ci.stage == vk::ShaderStageFlags::MESH_EXT
                || ci.stage == vk::ShaderStageFlags::TASK_EXT
            {
                ci.p_next = &subgroup_size_ci as *const _ as *const std::ffi::c_void;
            }
        }
    }

    let mut rendering_ci = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(color_formats)
        .depth_attachment_format(depth_format)
        .stencil_attachment_format(stencil_format);

    // Mesh shader pipelines must not have vertex input or input assembly state
    let mut pipeline_ci = vk::GraphicsPipelineCreateInfo::default()
        .push_next(&mut rendering_ci)
        .stages(&stage_cis_owned)
        .viewport_state(&viewport_ci)
        .rasterization_state(&rasterization_ci)
        .multisample_state(&multisample_ci)
        .depth_stencil_state(&depth_stencil_ci)
        .color_blend_state(&color_blend_ci)
        .dynamic_state(&dynamic_state_ci)
        .layout(pipeline_layout)
        .render_pass(vk::RenderPass::null())
        .subpass(0);

    if !is_mesh_pipeline {
        pipeline_ci = pipeline_ci
            .vertex_input_state(&vertex_input_ci)
            .input_assembly_state(&input_assembly_ci);
    }

    // SAFETY: device is valid, all pipeline create info is well-formed.
    let pipelines =
        unsafe { device.create_graphics_pipelines(pipeline_cache, &[pipeline_ci], None) }
            .map_err(|(_, err)| err)?;

    Ok(pipelines[0])
}

pub(crate) fn build_compute_pipeline(
    device: &ash::Device,
    pipeline_cache: vk::PipelineCache,
    module: vk::ShaderModule,
    pipeline_layout: vk::PipelineLayout,
) -> Result<vk::Pipeline, vk::Result> {
    build_compute_pipeline_with_state(
        device,
        pipeline_cache,
        module,
        pipeline_layout,
        &StaticPipelineState::default(),
    )
}

/// Build a compute pipeline with specialization constants and subgroup size control.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_compute_pipeline_with_state(
    device: &ash::Device,
    pipeline_cache: vk::PipelineCache,
    module: vk::ShaderModule,
    pipeline_layout: vk::PipelineLayout,
    state: &StaticPipelineState,
) -> Result<vk::Pipeline, vk::Result> {
    let entry_name = c"main";

    // Build specialization info if any constants are active
    let spec_map_entries: Vec<vk::SpecializationMapEntry> = (0..8u32)
        .filter(|i| state.spec_constant_mask & (1 << i) != 0)
        .map(|i| vk::SpecializationMapEntry {
            constant_id: i,
            offset: i * 4,
            size: 4,
        })
        .collect();

    let spec_info = if !spec_map_entries.is_empty() {
        Some(
            vk::SpecializationInfo::default()
                .map_entries(&spec_map_entries)
                .data(bytemuck::cast_slice(&state.spec_constants)),
        )
    } else {
        None
    };

    let mut subgroup_size_ci = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default();
    if state.subgroup_size_log2 > 0 {
        subgroup_size_ci.required_subgroup_size = 1u32 << state.subgroup_size_log2;
    }

    let mut stage_ci = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(entry_name);

    if let Some(ref spec) = spec_info {
        stage_ci = stage_ci.specialization_info(spec);
    }

    if state.subgroup_size_log2 > 0 {
        stage_ci.p_next = &subgroup_size_ci as *const _ as *const std::ffi::c_void;
    }

    let pipeline_ci = vk::ComputePipelineCreateInfo::default()
        .stage(stage_ci)
        .layout(pipeline_layout);

    // SAFETY: device is valid, pipeline create info is well-formed.
    let pipelines =
        unsafe { device.create_compute_pipelines(pipeline_cache, &[pipeline_ci], None) }
            .map_err(|(_, err)| err)?;

    Ok(pipelines[0])
}
