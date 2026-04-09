//! Opaque handle types for Vulkan resources.
//!
//! Handles wrap raw Vulkan objects with their associated allocation and metadata.
//! They are owned values — not reference counted — and register with the
//! [`DeletionQueue`](crate::frame_context::DeletionQueue) on destruction.

use ash::vk;
use crate::buffer::BufferInfo;
use crate::image::ImageInfo;

/// An opaque handle to a Vulkan buffer and its memory allocation.
pub struct BufferHandle {
    pub(crate) raw: vk::Buffer,
    pub(crate) allocation: Option<gpu_allocator::vulkan::Allocation>,
    pub(crate) info: BufferInfo,
}

impl BufferHandle {
    /// The raw Vulkan buffer handle.
    pub fn raw(&self) -> vk::Buffer {
        self.raw
    }

    /// Metadata about this buffer.
    pub fn info(&self) -> &BufferInfo {
        &self.info
    }
}

/// An opaque handle to a Vulkan image and its memory allocation.
pub struct ImageHandle {
    pub(crate) raw: vk::Image,
    pub(crate) allocation: Option<gpu_allocator::vulkan::Allocation>,
    pub(crate) info: ImageInfo,
}

impl ImageHandle {
    /// The raw Vulkan image handle.
    pub fn raw(&self) -> vk::Image {
        self.raw
    }

    /// Metadata about this image.
    pub fn info(&self) -> &ImageInfo {
        &self.info
    }
}

/// An opaque handle to a Vulkan image view.
pub struct ImageViewHandle {
    pub(crate) raw: vk::ImageView,
    pub(crate) image_info: ImageInfo,
}

impl ImageViewHandle {
    /// The raw Vulkan image view handle.
    pub fn raw(&self) -> vk::ImageView {
        self.raw
    }
}

/// An opaque handle to a Vulkan sampler.
pub struct SamplerHandle {
    pub(crate) raw: vk::Sampler,
}

impl SamplerHandle {
    /// The raw Vulkan sampler handle.
    pub fn raw(&self) -> vk::Sampler {
        self.raw
    }
}
