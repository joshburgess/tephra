//! Image creation helpers, staging, and metadata.

use crate::memory::ImageDomain;
use ash::vk;

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
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            image_type: vk::ImageType::TYPE_2D,
            initial_layout: vk::ImageLayout::UNDEFINED,
            domain: ImageDomain::Transient,
        }
    }

    // ---- Builder-style setters for chained construction ----

    /// Set the usage flags.
    pub fn usage(mut self, usage: vk::ImageUsageFlags) -> Self {
        self.usage = usage;
        self
    }

    /// Set the number of mip levels.
    pub fn mip_levels(mut self, mip_levels: u32) -> Self {
        self.mip_levels = mip_levels;
        self
    }

    /// Set the number of array layers.
    pub fn array_layers(mut self, array_layers: u32) -> Self {
        self.array_layers = array_layers;
        self
    }

    /// Set the sample count.
    pub fn samples(mut self, samples: vk::SampleCountFlags) -> Self {
        self.samples = samples;
        self
    }

    /// Set the memory domain.
    pub fn domain(mut self, domain: ImageDomain) -> Self {
        self.domain = domain;
        self
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immutable_2d_defaults() {
        let info = ImageCreateInfo::immutable_2d(512, 256, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(info.width, 512);
        assert_eq!(info.height, 256);
        assert_eq!(info.depth, 1);
        assert_eq!(info.format, vk::Format::R8G8B8A8_UNORM);
        assert!(info.usage.contains(vk::ImageUsageFlags::SAMPLED));
        assert!(info.usage.contains(vk::ImageUsageFlags::TRANSFER_DST));
        assert_eq!(info.mip_levels, 1);
        assert_eq!(info.array_layers, 1);
        assert_eq!(info.samples, vk::SampleCountFlags::TYPE_1);
        assert_eq!(info.image_type, vk::ImageType::TYPE_2D);
        assert_eq!(info.initial_layout, vk::ImageLayout::UNDEFINED);
        assert_eq!(info.domain, ImageDomain::Physical);
    }

    #[test]
    fn render_target_defaults() {
        let info = ImageCreateInfo::render_target(1920, 1080, vk::Format::B8G8R8A8_SRGB);
        assert_eq!(info.width, 1920);
        assert_eq!(info.height, 1080);
        assert!(info.usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT));
        assert!(info.usage.contains(vk::ImageUsageFlags::SAMPLED));
        assert!(!info.usage.contains(vk::ImageUsageFlags::TRANSFER_DST));
        assert_eq!(info.domain, ImageDomain::Physical);
    }

    #[test]
    fn depth_stencil_defaults() {
        let info = ImageCreateInfo::depth_stencil(800, 600, vk::Format::D32_SFLOAT);
        assert!(
            info.usage
                .contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        );
        assert!(!info.usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT));
        assert_eq!(info.domain, ImageDomain::Physical);
    }

    #[test]
    fn transient_attachment_defaults() {
        let info = ImageCreateInfo::transient_attachment(640, 480, vk::Format::R8G8B8A8_UNORM);
        assert!(
            info.usage
                .contains(vk::ImageUsageFlags::TRANSIENT_ATTACHMENT)
        );
        assert!(info.usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT));
        assert_eq!(info.domain, ImageDomain::Transient);
    }

    #[test]
    fn builder_usage() {
        let info = ImageCreateInfo::immutable_2d(64, 64, vk::Format::R8G8B8A8_UNORM)
            .usage(vk::ImageUsageFlags::STORAGE);
        assert_eq!(info.usage, vk::ImageUsageFlags::STORAGE);
    }

    #[test]
    fn builder_mip_levels() {
        let info =
            ImageCreateInfo::immutable_2d(256, 256, vk::Format::R8G8B8A8_UNORM).mip_levels(9);
        assert_eq!(info.mip_levels, 9);
    }

    #[test]
    fn builder_array_layers() {
        let info =
            ImageCreateInfo::immutable_2d(64, 64, vk::Format::R8G8B8A8_UNORM).array_layers(6);
        assert_eq!(info.array_layers, 6);
    }

    #[test]
    fn builder_samples() {
        let info = ImageCreateInfo::render_target(64, 64, vk::Format::R8G8B8A8_UNORM)
            .samples(vk::SampleCountFlags::TYPE_4);
        assert_eq!(info.samples, vk::SampleCountFlags::TYPE_4);
    }

    #[test]
    fn builder_domain() {
        let info = ImageCreateInfo::immutable_2d(64, 64, vk::Format::R8G8B8A8_UNORM)
            .domain(ImageDomain::Transient);
        assert_eq!(info.domain, ImageDomain::Transient);
    }

    #[test]
    fn builder_chaining() {
        let info = ImageCreateInfo::immutable_2d(128, 128, vk::Format::R8G8B8A8_UNORM)
            .mip_levels(5)
            .array_layers(6)
            .samples(vk::SampleCountFlags::TYPE_2)
            .domain(ImageDomain::Transient);
        assert_eq!(info.mip_levels, 5);
        assert_eq!(info.array_layers, 6);
        assert_eq!(info.samples, vk::SampleCountFlags::TYPE_2);
        assert_eq!(info.domain, ImageDomain::Transient);
    }
}
