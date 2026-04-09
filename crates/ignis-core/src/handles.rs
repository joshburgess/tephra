//! Opaque handle types for Vulkan resources.
//!
//! Handles wrap raw Vulkan objects with their associated allocation and metadata.
//! They are owned values — not reference counted — and should be returned to the
//! [`Device`](crate::device::Device) for deferred destruction.

use ash::vk;
use gpu_allocator::vulkan as vma;

use crate::buffer::BufferInfo;
use crate::image::ImageInfo;

/// An opaque handle to a Vulkan buffer and its memory allocation.
pub struct BufferHandle {
    pub(crate) raw: vk::Buffer,
    pub(crate) allocation: Option<vma::Allocation>,
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

    /// Access the mapped memory slice, if the buffer is host-visible.
    ///
    /// Returns `None` if the buffer is device-local only.
    pub fn mapped_slice(&self) -> Option<&[u8]> {
        self.allocation.as_ref()?.mapped_slice()
    }

    /// Access the mapped memory slice mutably, if the buffer is host-visible.
    ///
    /// Returns `None` if the buffer is device-local only.
    pub fn mapped_slice_mut(&mut self) -> Option<&mut [u8]> {
        self.allocation.as_mut()?.mapped_slice_mut()
    }
}

/// An opaque handle to a Vulkan image and its memory allocation.
pub struct ImageHandle {
    pub(crate) raw: vk::Image,
    pub(crate) allocation: Option<vma::Allocation>,
    pub(crate) info: ImageInfo,
    /// Default image view (created for convenience).
    pub(crate) default_view: vk::ImageView,
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

    /// The default image view (covers all mips and layers).
    pub fn default_view(&self) -> vk::ImageView {
        self.default_view
    }
}

/// An opaque handle to a Vulkan image view.
pub struct ImageViewHandle {
    pub(crate) raw: vk::ImageView,
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
