//! Slot-based descriptor binding model.
//!
//! Tracks current bindings for all descriptor sets. When a binding changes,
//! the set is marked dirty and will be re-allocated and written on the next
//! draw or dispatch.

use std::hash::{Hash, Hasher};

use ash::vk;

/// Maximum number of descriptor sets (Vulkan 1.0 minimum guarantee).
pub const MAX_DESCRIPTOR_SETS: usize = 4;

/// Maximum bindings per descriptor set.
pub const MAX_BINDINGS_PER_SET: usize = 16;

/// A single binding slot in a descriptor set.
#[derive(Clone, Copy)]
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
    /// Input attachment binding (for subpass inputs).
    InputAttachment {
        /// The image view.
        view: vk::ImageView,
        /// The image layout at time of access.
        layout: vk::ImageLayout,
    },
}

impl Hash for BindingSlot {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            BindingSlot::None => {}
            BindingSlot::UniformBuffer {
                buffer,
                offset,
                range,
            } => {
                buffer.hash(state);
                offset.hash(state);
                range.hash(state);
            }
            BindingSlot::StorageBuffer {
                buffer,
                offset,
                range,
            } => {
                buffer.hash(state);
                offset.hash(state);
                range.hash(state);
            }
            BindingSlot::CombinedImageSampler {
                view,
                sampler,
                layout,
            } => {
                view.hash(state);
                sampler.hash(state);
                (*layout).hash(state);
            }
            BindingSlot::StorageImage { view, layout } => {
                view.hash(state);
                (*layout).hash(state);
            }
            BindingSlot::InputAttachment { view, layout } => {
                view.hash(state);
                (*layout).hash(state);
            }
        }
    }
}

impl PartialEq for BindingSlot {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (BindingSlot::None, BindingSlot::None) => true,
            (
                BindingSlot::UniformBuffer { buffer: b1, offset: o1, range: r1 },
                BindingSlot::UniformBuffer { buffer: b2, offset: o2, range: r2 },
            ) => b1 == b2 && o1 == o2 && r1 == r2,
            (
                BindingSlot::StorageBuffer { buffer: b1, offset: o1, range: r1 },
                BindingSlot::StorageBuffer { buffer: b2, offset: o2, range: r2 },
            ) => b1 == b2 && o1 == o2 && r1 == r2,
            (
                BindingSlot::CombinedImageSampler { view: v1, sampler: s1, layout: l1 },
                BindingSlot::CombinedImageSampler { view: v2, sampler: s2, layout: l2 },
            ) => v1 == v2 && s1 == s2 && l1 == l2,
            (
                BindingSlot::StorageImage { view: v1, layout: l1 },
                BindingSlot::StorageImage { view: v2, layout: l2 },
            ) => v1 == v2 && l1 == l2,
            (
                BindingSlot::InputAttachment { view: v1, layout: l1 },
                BindingSlot::InputAttachment { view: v2, layout: l2 },
            ) => v1 == v2 && l1 == l2,
            _ => false,
        }
    }
}

impl Eq for BindingSlot {}

/// Bindings for a single descriptor set.
#[derive(Clone)]
pub struct DescriptorSetBindings {
    pub(crate) bindings: [BindingSlot; MAX_BINDINGS_PER_SET],
    pub(crate) active_mask: u32,
}

impl Default for DescriptorSetBindings {
    fn default() -> Self {
        Self {
            bindings: [BindingSlot::None; MAX_BINDINGS_PER_SET],
            active_mask: 0,
        }
    }
}

impl Hash for DescriptorSetBindings {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.active_mask.hash(state);
        // Only hash active bindings
        let mut mask = self.active_mask;
        while mask != 0 {
            let idx = mask.trailing_zeros() as usize;
            idx.hash(state);
            self.bindings[idx].hash(state);
            mask &= mask - 1;
        }
    }
}

impl PartialEq for DescriptorSetBindings {
    fn eq(&self, other: &Self) -> bool {
        if self.active_mask != other.active_mask {
            return false;
        }
        let mut mask = self.active_mask;
        while mask != 0 {
            let idx = mask.trailing_zeros() as usize;
            if self.bindings[idx] != other.bindings[idx] {
                return false;
            }
            mask &= mask - 1;
        }
        true
    }
}

impl Eq for DescriptorSetBindings {}

/// Tracks all descriptor set bindings and dirty state.
///
/// The binding table holds the current slot-based bindings for up to
/// [`MAX_DESCRIPTOR_SETS`] descriptor sets. Each `set_*` call marks the
/// affected set as dirty. On draw/dispatch, dirty sets are resolved
/// (hashed, cached, or freshly allocated and written).
pub struct BindingTable {
    sets: [DescriptorSetBindings; MAX_DESCRIPTOR_SETS],
    dirty_sets: u32,
}

impl BindingTable {
    /// Create an empty binding table.
    pub fn new() -> Self {
        Self {
            sets: std::array::from_fn(|_| DescriptorSetBindings::default()),
            dirty_sets: 0,
        }
    }

    /// Set a uniform buffer binding.
    pub fn set_uniform_buffer(
        &mut self,
        set: u32,
        binding: u32,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) {
        self.set_binding(
            set,
            binding,
            BindingSlot::UniformBuffer {
                buffer,
                offset,
                range,
            },
        );
    }

    /// Set a storage buffer binding.
    pub fn set_storage_buffer(
        &mut self,
        set: u32,
        binding: u32,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) {
        self.set_binding(
            set,
            binding,
            BindingSlot::StorageBuffer {
                buffer,
                offset,
                range,
            },
        );
    }

    /// Set a combined image sampler binding.
    pub fn set_texture(
        &mut self,
        set: u32,
        binding: u32,
        view: vk::ImageView,
        sampler: vk::Sampler,
    ) {
        self.set_binding(
            set,
            binding,
            BindingSlot::CombinedImageSampler {
                view,
                sampler,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
        );
    }

    /// Set a storage image binding.
    pub fn set_storage_image(&mut self, set: u32, binding: u32, view: vk::ImageView) {
        self.set_binding(
            set,
            binding,
            BindingSlot::StorageImage {
                view,
                layout: vk::ImageLayout::GENERAL,
            },
        );
    }

    /// Set an input attachment binding.
    pub fn set_input_attachment(&mut self, set: u32, binding: u32, view: vk::ImageView) {
        self.set_binding(
            set,
            binding,
            BindingSlot::InputAttachment {
                view,
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
        );
    }

    /// Clear all bindings for a given set.
    pub fn clear_set(&mut self, set: u32) {
        let s = set as usize;
        if s < MAX_DESCRIPTOR_SETS {
            self.sets[s] = DescriptorSetBindings::default();
            self.dirty_sets |= 1 << set;
        }
    }

    /// Clear all bindings in all sets.
    pub fn clear_all(&mut self) {
        for set in &mut self.sets {
            *set = DescriptorSetBindings::default();
        }
        self.dirty_sets = (1 << MAX_DESCRIPTOR_SETS) - 1;
    }

    /// Get the bindings for a descriptor set.
    pub fn get_set(&self, set: u32) -> &DescriptorSetBindings {
        &self.sets[set as usize]
    }

    /// Which sets are dirty (bitflag).
    pub fn dirty_sets(&self) -> u32 {
        self.dirty_sets
    }

    /// Clear the dirty flag for a set.
    pub fn clear_dirty(&mut self, set: u32) {
        self.dirty_sets &= !(1 << set);
    }

    /// Clear all dirty flags.
    pub fn clear_all_dirty(&mut self) {
        self.dirty_sets = 0;
    }

    fn set_binding(&mut self, set: u32, binding: u32, slot: BindingSlot) {
        let s = set as usize;
        let b = binding as usize;
        debug_assert!(s < MAX_DESCRIPTOR_SETS, "set index out of range");
        debug_assert!(b < MAX_BINDINGS_PER_SET, "binding index out of range");

        let set_bindings = &mut self.sets[s];
        set_bindings.bindings[b] = slot;
        set_bindings.active_mask |= 1 << binding;
        self.dirty_sets |= 1 << set;
    }
}

impl Default for BindingTable {
    fn default() -> Self {
        Self::new()
    }
}
