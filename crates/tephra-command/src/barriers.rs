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
    /// Source queue family for ownership transfer (`QUEUE_FAMILY_IGNORED` if not transferring).
    pub src_queue_family: u32,
    /// Destination queue family for ownership transfer (`QUEUE_FAMILY_IGNORED` if not transferring).
    pub dst_queue_family: u32,
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
            src_queue_family: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family: vk::QUEUE_FAMILY_IGNORED,
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
            src_queue_family: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family: vk::QUEUE_FAMILY_IGNORED,
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

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;

    fn dummy_image() -> vk::Image {
        vk::Image::from_raw(42)
    }

    // --- undefined_to ---

    #[test]
    fn undefined_to_sets_layouts() {
        let barrier = ImageBarrierInfo::undefined_to(
            dummy_image(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_READ,
            vk::ImageAspectFlags::COLOR,
        );
        assert_eq!(barrier.old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(barrier.new_layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }

    #[test]
    fn undefined_to_sets_stages() {
        let barrier = ImageBarrierInfo::undefined_to(
            dummy_image(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::ImageAspectFlags::COLOR,
        );
        assert_eq!(barrier.src_stage, vk::PipelineStageFlags2::TOP_OF_PIPE);
        assert_eq!(barrier.dst_stage, vk::PipelineStageFlags2::TRANSFER);
        assert_eq!(barrier.src_access, vk::AccessFlags2::NONE);
        assert_eq!(barrier.dst_access, vk::AccessFlags2::TRANSFER_WRITE);
    }

    #[test]
    fn undefined_to_sets_full_subresource() {
        let barrier = ImageBarrierInfo::undefined_to(
            dummy_image(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_READ,
            vk::ImageAspectFlags::DEPTH,
        );
        assert_eq!(barrier.subresource_range.aspect_mask, vk::ImageAspectFlags::DEPTH);
        assert_eq!(barrier.subresource_range.base_mip_level, 0);
        assert_eq!(barrier.subresource_range.level_count, vk::REMAINING_MIP_LEVELS);
        assert_eq!(barrier.subresource_range.base_array_layer, 0);
        assert_eq!(
            barrier.subresource_range.layer_count,
            vk::REMAINING_ARRAY_LAYERS
        );
    }

    #[test]
    fn undefined_to_no_queue_transfer() {
        let barrier = ImageBarrierInfo::undefined_to(
            dummy_image(),
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_WRITE,
            vk::ImageAspectFlags::COLOR,
        );
        assert_eq!(barrier.src_queue_family, vk::QUEUE_FAMILY_IGNORED);
        assert_eq!(barrier.dst_queue_family, vk::QUEUE_FAMILY_IGNORED);
    }

    // --- color_to_present ---

    #[test]
    fn color_to_present_layouts() {
        let barrier = ImageBarrierInfo::color_to_present(dummy_image());
        assert_eq!(barrier.old_layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        assert_eq!(barrier.new_layout, vk::ImageLayout::PRESENT_SRC_KHR);
    }

    #[test]
    fn color_to_present_stages() {
        let barrier = ImageBarrierInfo::color_to_present(dummy_image());
        assert_eq!(
            barrier.src_stage,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
        );
        assert_eq!(barrier.dst_stage, vk::PipelineStageFlags2::BOTTOM_OF_PIPE);
        assert_eq!(
            barrier.src_access,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
        );
        assert_eq!(barrier.dst_access, vk::AccessFlags2::NONE);
    }

    #[test]
    fn color_to_present_single_layer_single_mip() {
        let barrier = ImageBarrierInfo::color_to_present(dummy_image());
        assert_eq!(barrier.subresource_range.level_count, 1);
        assert_eq!(barrier.subresource_range.layer_count, 1);
        assert_eq!(
            barrier.subresource_range.aspect_mask,
            vk::ImageAspectFlags::COLOR
        );
    }

    // --- undefined_to_color_attachment ---

    #[test]
    fn undefined_to_color_attachment_layouts() {
        let barrier = ImageBarrierInfo::undefined_to_color_attachment(dummy_image());
        assert_eq!(barrier.old_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(barrier.new_layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }

    #[test]
    fn undefined_to_color_attachment_stages() {
        let barrier = ImageBarrierInfo::undefined_to_color_attachment(dummy_image());
        assert_eq!(barrier.src_stage, vk::PipelineStageFlags2::TOP_OF_PIPE);
        assert_eq!(
            barrier.dst_stage,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
        );
        assert_eq!(
            barrier.dst_access,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
        );
    }

    // --- Image handle preserved ---

    #[test]
    fn barrier_preserves_image_handle() {
        let img = dummy_image();
        let b1 = ImageBarrierInfo::color_to_present(img);
        let b2 = ImageBarrierInfo::undefined_to_color_attachment(img);
        assert_eq!(b1.image, img);
        assert_eq!(b2.image, img);
    }

    // --- Clone ---

    #[test]
    fn barrier_is_cloneable() {
        let barrier = ImageBarrierInfo::color_to_present(dummy_image());
        let cloned = barrier.clone();
        assert_eq!(cloned.old_layout, barrier.old_layout);
        assert_eq!(cloned.new_layout, barrier.new_layout);
        assert_eq!(cloned.image, barrier.image);
    }
}
