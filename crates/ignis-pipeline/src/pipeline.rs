//! Lazy pipeline compilation with hash-and-cache.
//!
//! On draw/dispatch, the pipeline key is hashed from the current program,
//! static render state, compatible render pass, and vertex input layout.
//! Cache misses trigger pipeline compilation via `vkCreateGraphicsPipelines`.

use std::hash::{Hash, Hasher};

use ash::vk;
use rustc_hash::{FxHashMap, FxHasher};

use ignis_command::state::StaticPipelineState;

use crate::program::Program;

/// Key for looking up a compiled pipeline.
#[derive(Clone, PartialEq, Eq, Hash)]
struct PipelineKey {
    program_hash: u64,
    static_state: StaticPipelineState,
    render_pass_hash: u64,
    subpass: u32,
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
    fn compute_hash(&self) -> u64 {
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
    compute_cache: FxHashMap<u64, vk::Pipeline>,
    pipeline_cache: vk::PipelineCache,
}

impl PipelineCompiler {
    /// Create a new pipeline compiler with the given `VkPipelineCache`.
    pub fn new(pipeline_cache: vk::PipelineCache) -> Self {
        Self {
            graphics_cache: FxHashMap::default(),
            compute_cache: FxHashMap::default(),
            pipeline_cache,
        }
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

        let pipeline = self.compile_graphics(
            device,
            program,
            state,
            render_pass,
            subpass,
            vertex_layout,
        )?;

        log::debug!(
            "Compiled graphics pipeline (total cached: {})",
            self.graphics_cache.len() + 1
        );

        self.graphics_cache.insert(key_hash, pipeline);
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

        self.compute_cache.insert(key_hash, pipeline);
        Ok(pipeline)
    }

    /// Destroy all cached pipelines.
    pub fn destroy(&mut self, device: &ash::Device) {
        for (_, pipeline) in self.graphics_cache.drain() {
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
        let entry_name = c"main";

        // Shader stages
        let stage_cis: Vec<vk::PipelineShaderStageCreateInfo<'_>> = program
            .shaders()
            .iter()
            .map(|s| {
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::from_raw(s.stage.as_raw()))
                    .module(s.module)
                    .name(entry_name)
            })
            .collect();

        // Vertex input
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

        // Viewport and scissor are dynamic
        let viewport_ci = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization_ci = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(state.polygon_mode)
            .cull_mode(state.cull_mode)
            .front_face(state.front_face)
            .depth_bias_enable(false)
            .line_width(1.0);

        let multisample_ci = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false);

        let depth_stencil_ci = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(state.depth_test)
            .depth_write_enable(state.depth_write)
            .depth_compare_op(state.depth_compare)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(state.stencil_test);

        // One blend attachment state per color attachment
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(state.blend_enable)
            .src_color_blend_factor(state.src_color_blend)
            .dst_color_blend_factor(state.dst_color_blend)
            .color_blend_op(state.color_blend_op)
            .src_alpha_blend_factor(state.src_alpha_blend)
            .dst_alpha_blend_factor(state.dst_alpha_blend)
            .alpha_blend_op(state.alpha_blend_op)
            .color_write_mask(state.color_write_mask);

        // Default to 1 color attachment; the render pass dictates the real count
        let blend_attachments = [color_blend_attachment];

        let color_blend_ci = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_ci =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let pipeline_ci = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stage_cis)
            .vertex_input_state(&vertex_input_ci)
            .input_assembly_state(&input_assembly_ci)
            .viewport_state(&viewport_ci)
            .rasterization_state(&rasterization_ci)
            .multisample_state(&multisample_ci)
            .depth_stencil_state(&depth_stencil_ci)
            .color_blend_state(&color_blend_ci)
            .dynamic_state(&dynamic_state_ci)
            .layout(program.pipeline_layout())
            .render_pass(render_pass)
            .subpass(subpass);

        // SAFETY: device is valid, all pipeline create info is well-formed.
        let pipelines = unsafe {
            device.create_graphics_pipelines(self.pipeline_cache, &[pipeline_ci], None)
        }
        .map_err(|(_, err)| err)?;

        Ok(pipelines[0])
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

        let entry_name = c"main";
        let stage_ci = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader.module)
            .name(entry_name);

        let pipeline_ci = vk::ComputePipelineCreateInfo::default()
            .stage(stage_ci)
            .layout(program.pipeline_layout());

        // SAFETY: device is valid, pipeline create info is well-formed.
        let pipelines = unsafe {
            device.create_compute_pipelines(self.pipeline_cache, &[pipeline_ci], None)
        }
        .map_err(|(_, err)| err)?;

        Ok(pipelines[0])
    }
}
