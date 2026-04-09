//! Barrier helper utilities for pipeline and image layout transitions.

use ash::vk;

/// Description of an image layout transition barrier.
#[derive(Clone)]
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

impl ImageBarrierInfo {
    /// Convenience: transition an image from undefined to a target layout.
    /// Useful for newly created images.
    pub fn undefined_to(
        image: vk::Image,
        new_layout: vk::ImageLayout,
        dst_stage: vk::PipelineStageFlags2,
        dst_access: vk::AccessFlags2,
        aspect_mask: vk::ImageAspectFlags,
    ) -> Self {
        Self {
            image,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout,
            src_stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
            src_access: vk::AccessFlags2::NONE,
            dst_stage,
            dst_access,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            },
        }
    }

    /// Convenience: transition a color attachment to present layout.
    pub fn color_to_present(image: vk::Image) -> Self {
        Self {
            image,
            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            src_stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            src_access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            dst_stage: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            dst_access: vk::AccessFlags2::NONE,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        }
    }

    /// Convenience: transition from undefined to color attachment optimal.
    pub fn undefined_to_color_attachment(image: vk::Image) -> Self {
        Self::undefined_to(
            image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::ImageAspectFlags::COLOR,
        )
    }
}
