//! Image creation helpers, staging, and metadata.

use ash::vk;
use crate::memory::ImageDomain;

/// Metadata describing an image's properties.
#[derive(Debug, Clone)]
pub struct ImageInfo {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Depth (1 for 2D images).
    pub depth: u32,
    /// Pixel format.
    pub format: vk::Format,
    /// Number of mip levels.
    pub mip_levels: u32,
    /// Number of array layers.
    pub array_layers: u32,
    /// Sample count.
    pub samples: vk::SampleCountFlags,
    /// Image type (1D, 2D, 3D).
    pub image_type: vk::ImageType,
}

/// Parameters for creating a new image.
pub struct ImageCreateInfo {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Depth (1 for 2D images).
    pub depth: u32,
    /// Pixel format.
    pub format: vk::Format,
    /// Vulkan image usage flags.
    pub usage: vk::ImageUsageFlags,
    /// Number of mip levels.
    pub mip_levels: u32,
    /// Number of array layers.
    pub array_layers: u32,
    /// Sample count.
    pub samples: vk::SampleCountFlags,
    /// Image type (1D, 2D, 3D).
    pub image_type: vk::ImageType,
    /// Initial layout.
    pub initial_layout: vk::ImageLayout,
    /// Memory domain.
    pub domain: ImageDomain,
}

impl ImageCreateInfo {
    /// Convenience constructor for an immutable 2D image (e.g., a texture).
    pub fn immutable_2d(width: u32, height: u32, format: vk::Format) -> Self {
        Self {
            width,
            height,
            depth: 1,
            format,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            image_type: vk::ImageType::TYPE_2D,
            initial_layout: vk::ImageLayout::UNDEFINED,
            domain: ImageDomain::Physical,
        }
    }

    /// Convenience constructor for a color render target.
    pub fn render_target(width: u32, height: u32, format: vk::Format) -> Self {
        Self {
            width,
            height,
            depth: 1,
            format,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            image_type: vk::ImageType::TYPE_2D,
            initial_layout: vk::ImageLayout::UNDEFINED,
            domain: ImageDomain::Physical,
        }
    }

    /// Convenience constructor for a depth/stencil attachment.
    pub fn depth_stencil(width: u32, height: u32, format: vk::Format) -> Self {
        Self {
            width,
            height,
            depth: 1,
            format,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            image_type: vk::ImageType::TYPE_2D,
            initial_layout: vk::ImageLayout::UNDEFINED,
            domain: ImageDomain::Physical,
        }
    }

    /// Convenience constructor for a transient attachment (lazily allocated).
    pub fn transient_attachment(width: u32, height: u32, format: vk::Format) -> Self {
        Self {
            width,
            height,
            depth: 1,
            format,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            image_type: vk::ImageType::TYPE_2D,
            initial_layout: vk::ImageLayout::UNDEFINED,
            domain: ImageDomain::Transient,
        }
    }
}

/// Parameters for creating an image view.
pub struct ImageViewCreateInfo {
    /// The view type.
    pub view_type: vk::ImageViewType,
    /// Format (may differ from the image format for format-compatible views).
    pub format: vk::Format,
    /// Subresource range.
    pub subresource_range: vk::ImageSubresourceRange,
}
