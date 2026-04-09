//! Barrier helper utilities for pipeline and image layout transitions.

use ash::vk;

/// Description of a pipeline barrier to insert.
pub struct BarrierInfo {
    /// Source pipeline stage.
    pub src_stage: vk::PipelineStageFlags2,
    /// Destination pipeline stage.
    pub dst_stage: vk::PipelineStageFlags2,
    /// Source access mask.
    pub src_access: vk::AccessFlags2,
    /// Destination access mask.
    pub dst_access: vk::AccessFlags2,
}

/// Description of an image layout transition.
pub struct ImageBarrierInfo {
    /// The image to transition.
    pub image: vk::Image,
    /// Old layout.
    pub old_layout: vk::ImageLayout,
    /// New layout.
    pub new_layout: vk::ImageLayout,
    /// Source pipeline stage.
    pub src_stage: vk::PipelineStageFlags2,
    /// Destination pipeline stage.
    pub dst_stage: vk::PipelineStageFlags2,
    /// Source access mask.
    pub src_access: vk::AccessFlags2,
    /// Destination access mask.
    pub dst_access: vk::AccessFlags2,
    /// Subresource range to transition.
    pub subresource_range: vk::ImageSubresourceRange,
}
