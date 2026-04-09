//! Multi-stage shader program linking.
//!
//! A [`Program`] combines multiple [`Shader`](crate::shader::Shader) stages,
//! merges their reflection data, and creates the Vulkan pipeline layout.

use ash::vk;
use crate::shader::MAX_DESCRIPTOR_SETS;

/// A linked shader program (e.g., vertex + fragment, or compute).
pub struct Program {
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) descriptor_set_layouts: [Option<vk::DescriptorSetLayout>; MAX_DESCRIPTOR_SETS],
    pub(crate) push_constant_range: Option<vk::PushConstantRange>,
    pub(crate) layout_hash: u64,
}

/// Handle to a program, usable for pipeline compilation.
pub struct ProgramHandle {
    // TODO: Phase 4, Iteration 4.2
    _private: (),
}
