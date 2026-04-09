//! Shader module loading and SPIR-V reflection.

use ash::vk;

use crate::render_pass::DescriptorSetLayout;

// Re-export for convenience
pub use crate::render_pass::DescriptorBindingInfo;

/// Maximum descriptor sets supported.
pub const MAX_DESCRIPTOR_SETS: usize = 4;

/// Reflected information about a vertex shader input.
#[derive(Debug, Clone)]
pub struct VertexInputAttribute {
    /// The input location.
    pub location: u32,
    /// The vertex attribute format.
    pub format: vk::Format,
    /// Byte offset within the vertex.
    pub offset: u32,
}

/// Reflection data extracted from a SPIR-V shader module.
pub struct ShaderReflection {
    /// Descriptor set layouts used by this shader.
    pub descriptor_sets: [DescriptorSetLayout; MAX_DESCRIPTOR_SETS],
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
