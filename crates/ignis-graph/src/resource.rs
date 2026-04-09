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
