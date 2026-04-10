//! Shader module loading and SPIR-V reflection.
//!
//! Uses `spirv-reflect` to extract descriptor set layouts, push constant ranges,
//! and vertex input attributes from SPIR-V shader modules.

use ash::vk;

/// Maximum descriptor sets supported (Vulkan 1.0 minimum guarantee).
pub const MAX_DESCRIPTOR_SETS: usize = 4;

/// Describes a single binding within a descriptor set layout.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DescriptorBindingInfo {
    /// The binding number.
    pub binding: u32,
    /// The descriptor type.
    pub descriptor_type: vk::DescriptorType,
    /// Number of descriptors in this binding.
    pub descriptor_count: u32,
    /// Shader stages that access this binding.
    pub stage_flags: vk::ShaderStageFlags,
}

/// Reflected descriptor set layout (list of bindings for a single set).
#[derive(Debug, Clone, Default)]
pub struct ReflectedSetLayout {
    /// The bindings in this set.
    pub bindings: Vec<DescriptorBindingInfo>,
}

/// Reflected information about a vertex shader input.
#[derive(Debug, Clone)]
pub struct VertexInputAttribute {
    /// The input location.
    pub location: u32,
    /// The vertex attribute format.
    pub format: vk::Format,
}

/// Reflection data extracted from a SPIR-V shader module.
pub struct ShaderReflection {
    /// Descriptor set layouts used by this shader.
    pub descriptor_sets: [ReflectedSetLayout; MAX_DESCRIPTOR_SETS],
    /// Push constant range, if any.
    pub push_constant_range: Option<vk::PushConstantRange>,
    /// Vertex input attributes (only for vertex shaders).
    pub vertex_inputs: Vec<VertexInputAttribute>,
}

/// A compiled shader module with its reflection data.
pub struct Shader {
    pub(crate) module: vk::ShaderModule,
    pub(crate) stage: vk::ShaderStageFlags,
    pub(crate) reflection: ShaderReflection,
}

impl Shader {
    /// Create a shader module from SPIR-V bytecode.
    ///
    /// Loads the SPIR-V, creates a `VkShaderModule`, and extracts reflection
    /// data (descriptor set layouts, push constant ranges, vertex inputs).
    pub fn create(
        device: &ash::Device,
        stage: vk::ShaderStageFlags,
        spirv: &[u32],
    ) -> Result<Self, String> {
        let reflection = reflect_spirv(stage, spirv)?;

        let module_ci = vk::ShaderModuleCreateInfo::default().code(spirv);

        // SAFETY: device is valid, SPIR-V data is well-formed.
        let module = unsafe {
            device
                .create_shader_module(&module_ci, None)
                .map_err(|e| format!("Failed to create shader module: {e}"))?
        };

        log::debug!("Created shader module (stage={:?})", stage);

        Ok(Self {
            module,
            stage,
            reflection,
        })
    }

    /// Destroy the shader module.
    pub fn destroy(&mut self, device: &ash::Device) {
        if self.module != vk::ShaderModule::null() {
            // SAFETY: device is valid, module is valid, GPU is idle.
            unsafe {
                device.destroy_shader_module(self.module, None);
            }
            self.module = vk::ShaderModule::null();
        }
    }

    /// The raw Vulkan shader module handle.
    pub fn module(&self) -> vk::ShaderModule {
        self.module
    }

    /// The shader stage.
    pub fn stage(&self) -> vk::ShaderStageFlags {
        self.stage
    }

    /// The reflection data.
    pub fn reflection(&self) -> &ShaderReflection {
        &self.reflection
    }
}

/// Extract reflection data from SPIR-V bytecode.
pub(crate) fn reflect_spirv(
    stage: vk::ShaderStageFlags,
    spirv: &[u32],
) -> Result<ShaderReflection, String> {
    let module = spirv_reflect::ShaderModule::load_u32_data(spirv)
        .map_err(|e| format!("Failed to load SPIR-V for reflection: {e}"))?;

    let descriptor_sets = reflect_descriptor_sets(&module, stage)?;
    let push_constant_range = reflect_push_constants(&module, stage)?;

    let vertex_inputs = if stage == vk::ShaderStageFlags::VERTEX {
        reflect_vertex_inputs(&module)?
    } else {
        Vec::new()
    };

    Ok(ShaderReflection {
        descriptor_sets,
        push_constant_range,
        vertex_inputs,
    })
}

fn reflect_descriptor_sets(
    module: &spirv_reflect::ShaderModule,
    stage: vk::ShaderStageFlags,
) -> Result<[ReflectedSetLayout; MAX_DESCRIPTOR_SETS], String> {
    let bindings = module
        .enumerate_descriptor_bindings(None)
        .map_err(|e| format!("Failed to enumerate descriptor bindings: {e}"))?;

    let mut sets: [ReflectedSetLayout; MAX_DESCRIPTOR_SETS] =
        std::array::from_fn(|_| ReflectedSetLayout::default());

    for binding in &bindings {
        let set_idx = binding.set as usize;
        if set_idx >= MAX_DESCRIPTOR_SETS {
            log::warn!(
                "Descriptor set {} exceeds MAX_DESCRIPTOR_SETS ({}), skipping",
                set_idx,
                MAX_DESCRIPTOR_SETS
            );
            continue;
        }

        let descriptor_type = convert_descriptor_type(binding.descriptor_type);
        let descriptor_count = if binding.count > 0 { binding.count } else { 1 };

        sets[set_idx].bindings.push(DescriptorBindingInfo {
            binding: binding.binding,
            descriptor_type,
            descriptor_count,
            stage_flags: stage,
        });
    }

    for set in &mut sets {
        set.bindings.sort_by_key(|b| b.binding);
    }

    Ok(sets)
}

fn reflect_push_constants(
    module: &spirv_reflect::ShaderModule,
    stage: vk::ShaderStageFlags,
) -> Result<Option<vk::PushConstantRange>, String> {
    let blocks = module
        .enumerate_push_constant_blocks(None)
        .map_err(|e| format!("Failed to enumerate push constant blocks: {e}"))?;

    if blocks.is_empty() {
        return Ok(None);
    }

    // Use the largest block size (push constant blocks are overlapping ranges).
    let total_size: u32 = blocks.iter().map(|b| b.padded_size).max().unwrap_or(0);

    if total_size == 0 {
        return Ok(None);
    }

    Ok(Some(
        vk::PushConstantRange::default()
            .stage_flags(stage)
            .offset(0)
            .size(total_size),
    ))
}

fn reflect_vertex_inputs(
    module: &spirv_reflect::ShaderModule,
) -> Result<Vec<VertexInputAttribute>, String> {
    let inputs = module
        .enumerate_input_variables(None)
        .map_err(|e| format!("Failed to enumerate input variables: {e}"))?;

    let mut attrs: Vec<VertexInputAttribute> = inputs
        .iter()
        .filter(|v| {
            !v.decoration_flags
                .contains(spirv_reflect::types::ReflectDecorationFlags::BUILT_IN)
        })
        .map(|v| VertexInputAttribute {
            location: v.location,
            format: convert_format(v.format),
        })
        .collect();

    attrs.sort_by_key(|a| a.location);
    Ok(attrs)
}

fn convert_descriptor_type(dt: spirv_reflect::types::ReflectDescriptorType) -> vk::DescriptorType {
    use spirv_reflect::types::ReflectDescriptorType;
    match dt {
        ReflectDescriptorType::Sampler => vk::DescriptorType::SAMPLER,
        ReflectDescriptorType::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        ReflectDescriptorType::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
        ReflectDescriptorType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
        ReflectDescriptorType::UniformTexelBuffer => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
        ReflectDescriptorType::StorageTexelBuffer => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
        ReflectDescriptorType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
        ReflectDescriptorType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
        ReflectDescriptorType::UniformBufferDynamic => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
        ReflectDescriptorType::StorageBufferDynamic => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
        ReflectDescriptorType::InputAttachment => vk::DescriptorType::INPUT_ATTACHMENT,
        ReflectDescriptorType::AccelerationStructureNV => {
            vk::DescriptorType::ACCELERATION_STRUCTURE_KHR
        }
        ReflectDescriptorType::Undefined => vk::DescriptorType::SAMPLER,
    }
}

fn convert_format(fmt: spirv_reflect::types::ReflectFormat) -> vk::Format {
    use spirv_reflect::types::ReflectFormat;
    match fmt {
        ReflectFormat::R32_UINT => vk::Format::R32_UINT,
        ReflectFormat::R32_SINT => vk::Format::R32_SINT,
        ReflectFormat::R32_SFLOAT => vk::Format::R32_SFLOAT,
        ReflectFormat::R32G32_UINT => vk::Format::R32G32_UINT,
        ReflectFormat::R32G32_SINT => vk::Format::R32G32_SINT,
        ReflectFormat::R32G32_SFLOAT => vk::Format::R32G32_SFLOAT,
        ReflectFormat::R32G32B32_UINT => vk::Format::R32G32B32_UINT,
        ReflectFormat::R32G32B32_SINT => vk::Format::R32G32B32_SINT,
        ReflectFormat::R32G32B32_SFLOAT => vk::Format::R32G32B32_SFLOAT,
        ReflectFormat::R32G32B32A32_UINT => vk::Format::R32G32B32A32_UINT,
        ReflectFormat::R32G32B32A32_SINT => vk::Format::R32G32B32A32_SINT,
        ReflectFormat::R32G32B32A32_SFLOAT => vk::Format::R32G32B32A32_SFLOAT,
        ReflectFormat::Undefined => vk::Format::UNDEFINED,
    }
}
