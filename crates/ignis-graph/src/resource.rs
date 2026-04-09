//! Virtual resource declarations for the render graph.
//!
//! Resources are virtual handles until physical images/buffers are assigned
//! during graph compilation and execution.

use ash::vk;

/// Opaque handle to a virtual resource in the render graph.
///
/// Resources are virtual until physical images/buffers are assigned during
/// graph compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceHandle {
    pub(crate) index: u32,
}

/// How a resource's size is specified.
#[derive(Debug, Clone, Copy)]
pub enum SizeClass {
    /// Absolute pixel dimensions.
    Absolute(u32),
    /// Relative to the swapchain extent (e.g., 1.0 = full, 0.5 = half).
    SwapchainRelative(f32),
}

/// Description of an image attachment resource.
#[derive(Debug, Clone)]
pub struct AttachmentInfo {
    /// The image format.
    pub format: vk::Format,
    /// Width specification.
    pub width: SizeClass,
    /// Height specification.
    pub height: SizeClass,
    /// Multisample count.
    pub samples: vk::SampleCountFlags,
}

impl AttachmentInfo {
    /// Create an attachment with absolute dimensions.
    pub fn absolute(format: vk::Format, width: u32, height: u32) -> Self {
        Self {
            format,
            width: SizeClass::Absolute(width),
            height: SizeClass::Absolute(height),
            samples: vk::SampleCountFlags::TYPE_1,
        }
    }

    /// Create an attachment sized relative to the swapchain.
    pub fn swapchain_relative(format: vk::Format, scale: f32) -> Self {
        Self {
            format,
            width: SizeClass::SwapchainRelative(scale),
            height: SizeClass::SwapchainRelative(scale),
            samples: vk::SampleCountFlags::TYPE_1,
        }
    }

    /// Resolve dimensions given the swapchain extent.
    pub fn resolve_extent(&self, swapchain_extent: vk::Extent2D) -> vk::Extent2D {
        let w = match self.width {
            SizeClass::Absolute(v) => v,
            SizeClass::SwapchainRelative(s) => (swapchain_extent.width as f32 * s) as u32,
        };
        let h = match self.height {
            SizeClass::Absolute(v) => v,
            SizeClass::SwapchainRelative(s) => (swapchain_extent.height as f32 * s) as u32,
        };
        vk::Extent2D {
            width: w.max(1),
            height: h.max(1),
        }
    }
}

/// Description of a buffer resource.
#[derive(Debug, Clone)]
pub struct BufferInfo {
    /// Buffer size in bytes.
    pub size: vk::DeviceSize,
    /// Buffer usage flags.
    pub usage: vk::BufferUsageFlags,
}

/// Internal resource declaration in the graph.
pub(crate) enum ResourceInfo {
    Attachment(AttachmentInfo),
    Buffer(BufferInfo),
}

/// A declared resource in the graph.
pub(crate) struct ResourceDeclaration {
    pub name: String,
    pub info: ResourceInfo,
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- AttachmentInfo::absolute --

    #[test]
    fn absolute_attachment_dimensions() {
        let info = AttachmentInfo::absolute(vk::Format::R8G8B8A8_UNORM, 1920, 1080);
        let extent = info.resolve_extent(vk::Extent2D {
            width: 800,
            height: 600,
        });
        // Absolute dimensions ignore swapchain extent
        assert_eq!(extent.width, 1920);
        assert_eq!(extent.height, 1080);
    }

    // -- AttachmentInfo::swapchain_relative --

    #[test]
    fn swapchain_relative_full_scale() {
        let info = AttachmentInfo::swapchain_relative(vk::Format::R8G8B8A8_UNORM, 1.0);
        let extent = info.resolve_extent(vk::Extent2D {
            width: 1920,
            height: 1080,
        });
        assert_eq!(extent.width, 1920);
        assert_eq!(extent.height, 1080);
    }

    #[test]
    fn swapchain_relative_half_scale() {
        let info = AttachmentInfo::swapchain_relative(vk::Format::R8G8B8A8_UNORM, 0.5);
        let extent = info.resolve_extent(vk::Extent2D {
            width: 1920,
            height: 1080,
        });
        assert_eq!(extent.width, 960);
        assert_eq!(extent.height, 540);
    }

    #[test]
    fn swapchain_relative_minimum_clamp() {
        // Even with a tiny scale, dimensions should clamp to at least 1
        let info = AttachmentInfo::swapchain_relative(vk::Format::R8G8B8A8_UNORM, 0.001);
        let extent = info.resolve_extent(vk::Extent2D {
            width: 100,
            height: 100,
        });
        assert!(extent.width >= 1);
        assert!(extent.height >= 1);
    }

    #[test]
    fn resolve_extent_mixed_size_class() {
        // Width absolute, height relative
        let info = AttachmentInfo {
            format: vk::Format::R8G8B8A8_UNORM,
            width: SizeClass::Absolute(512),
            height: SizeClass::SwapchainRelative(0.5),
            samples: vk::SampleCountFlags::TYPE_1,
        };
        let extent = info.resolve_extent(vk::Extent2D {
            width: 1920,
            height: 1080,
        });
        assert_eq!(extent.width, 512);
        assert_eq!(extent.height, 540);
    }

    // -- Default samples --

    #[test]
    fn absolute_defaults_to_single_sample() {
        let info = AttachmentInfo::absolute(vk::Format::R8G8B8A8_UNORM, 100, 100);
        assert_eq!(info.samples, vk::SampleCountFlags::TYPE_1);
    }

    #[test]
    fn swapchain_relative_defaults_to_single_sample() {
        let info = AttachmentInfo::swapchain_relative(vk::Format::R8G8B8A8_UNORM, 1.0);
        assert_eq!(info.samples, vk::SampleCountFlags::TYPE_1);
    }

    // -- ResourceHandle equality --

    #[test]
    fn resource_handle_equality() {
        let a = ResourceHandle { index: 0 };
        let b = ResourceHandle { index: 0 };
        let c = ResourceHandle { index: 1 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn resource_handle_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ResourceHandle { index: 5 });
        assert!(set.contains(&ResourceHandle { index: 5 }));
        assert!(!set.contains(&ResourceHandle { index: 6 }));
    }
}
