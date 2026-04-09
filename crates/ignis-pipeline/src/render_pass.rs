//! Render pass auto-generation and caching.
//!
//! Hashes render pass compatibility keys and caches `VkRenderPass` and
//! `VkFramebuffer` objects.

use ash::vk;

/// A descriptor set layout description (list of bindings).
#[derive(Default)]
pub struct DescriptorSetLayout {
    /// The bindings in this set.
    pub bindings: Vec<DescriptorBindingInfo>,
}

/// Owned version of descriptor set layout binding info (no lifetime).
#[derive(Debug, Clone)]
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

/// Parameters describing a render pass configuration.
pub struct RenderPassInfo {
    /// Number of color attachments.
    pub color_attachment_count: u32,
    /// Formats of color attachments.
    pub color_formats: Vec<vk::Format>,
    /// Depth/stencil format, if any.
    pub depth_stencil_format: Option<vk::Format>,
    /// Sample count.
    pub samples: vk::SampleCountFlags,
}

/// Cache for render passes and framebuffers.
pub struct RenderPassCache {
    // TODO: Phase 4, Iteration 4.3
    // - FxHashMap<RenderPassKey, vk::RenderPass>
    // - FxHashMap<FramebufferKey, vk::Framebuffer>
    _private: (),
}
