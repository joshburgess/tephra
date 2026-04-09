//! Slot-based descriptor binding model.
//!
//! Tracks current bindings for all descriptor sets. When a binding changes,
//! the set is marked dirty and will be re-allocated and written on the next
//! draw or dispatch.

use ash::vk;

/// Maximum number of descriptor sets (Vulkan 1.0 minimum guarantee).
pub const MAX_DESCRIPTOR_SETS: usize = 4;

/// Maximum bindings per descriptor set.
pub const MAX_BINDINGS_PER_SET: usize = 16;

/// A single binding slot in a descriptor set.
#[derive(Clone)]
pub enum BindingSlot {
    /// No binding set.
    None,
    /// Uniform buffer binding.
    UniformBuffer {
        /// The buffer handle.
        buffer: vk::Buffer,
        /// Byte offset into the buffer.
        offset: vk::DeviceSize,
        /// Byte range of the binding.
        range: vk::DeviceSize,
    },
    /// Storage buffer binding.
    StorageBuffer {
        /// The buffer handle.
        buffer: vk::Buffer,
        /// Byte offset into the buffer.
        offset: vk::DeviceSize,
        /// Byte range of the binding.
        range: vk::DeviceSize,
    },
    /// Combined image sampler binding.
    CombinedImageSampler {
        /// The image view.
        view: vk::ImageView,
        /// The sampler.
        sampler: vk::Sampler,
        /// The image layout at time of access.
        layout: vk::ImageLayout,
    },
    /// Storage image binding.
    StorageImage {
        /// The image view.
        view: vk::ImageView,
        /// The image layout at time of access.
        layout: vk::ImageLayout,
    },
}

/// Bindings for a single descriptor set.
pub(crate) struct DescriptorSetBindings {
    pub bindings: [BindingSlot; MAX_BINDINGS_PER_SET],
    pub active_mask: u32,
}

/// Tracks all descriptor set bindings and dirty state.
pub struct BindingTable {
    pub(crate) sets: [DescriptorSetBindings; MAX_DESCRIPTOR_SETS],
    pub(crate) dirty_sets: u32,
}
