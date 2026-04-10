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
#[derive(Debug, Clone, Copy)]
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
    /// Acceleration structure binding (for ray queries).
    AccelerationStructure {
        /// The acceleration structure handle.
        handle: vk::AccelerationStructureKHR,
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
            BindingSlot::AccelerationStructure { handle } => {
                handle.hash(state);
            }
        }
    }
}

impl PartialEq for BindingSlot {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (BindingSlot::None, BindingSlot::None) => true,
            (
                BindingSlot::UniformBuffer {
                    buffer: b1,
                    offset: o1,
                    range: r1,
                },
                BindingSlot::UniformBuffer {
                    buffer: b2,
                    offset: o2,
                    range: r2,
                },
            ) => b1 == b2 && o1 == o2 && r1 == r2,
            (
                BindingSlot::StorageBuffer {
                    buffer: b1,
                    offset: o1,
                    range: r1,
                },
                BindingSlot::StorageBuffer {
                    buffer: b2,
                    offset: o2,
                    range: r2,
                },
            ) => b1 == b2 && o1 == o2 && r1 == r2,
            (
                BindingSlot::CombinedImageSampler {
                    view: v1,
                    sampler: s1,
                    layout: l1,
                },
                BindingSlot::CombinedImageSampler {
                    view: v2,
                    sampler: s2,
                    layout: l2,
                },
            ) => v1 == v2 && s1 == s2 && l1 == l2,
            (
                BindingSlot::StorageImage {
                    view: v1,
                    layout: l1,
                },
                BindingSlot::StorageImage {
                    view: v2,
                    layout: l2,
                },
            ) => v1 == v2 && l1 == l2,
            (
                BindingSlot::InputAttachment {
                    view: v1,
                    layout: l1,
                },
                BindingSlot::InputAttachment {
                    view: v2,
                    layout: l2,
                },
            ) => v1 == v2 && l1 == l2,
            (
                BindingSlot::AccelerationStructure { handle: h1 },
                BindingSlot::AccelerationStructure { handle: h2 },
            ) => h1 == h2,
            _ => false,
        }
    }
}

impl Eq for BindingSlot {}

/// Bindings for a single descriptor set.
#[derive(Debug, Clone)]
pub struct DescriptorSetBindings {
    pub(crate) bindings: [BindingSlot; MAX_BINDINGS_PER_SET],
    pub(crate) active_mask: u32,
}

impl DescriptorSetBindings {
    /// Bitmask of which binding slots are active (have a non-None value).
    pub fn active_mask(&self) -> u32 {
        self.active_mask
    }

    /// Whether this set has any active bindings.
    pub fn is_empty(&self) -> bool {
        self.active_mask == 0
    }
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
#[derive(Clone)]
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

    /// Set an acceleration structure binding (for ray queries).
    pub fn set_acceleration_structure(
        &mut self,
        set: u32,
        binding: u32,
        handle: vk::AccelerationStructureKHR,
    ) {
        self.set_binding(set, binding, BindingSlot::AccelerationStructure { handle });
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

    /// The bindings for a descriptor set.
    pub fn set(&self, set: u32) -> &DescriptorSetBindings {
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

    /// Mark a specific set as dirty (forces re-allocation on next flush).
    pub fn mark_dirty(&mut self, set: u32) {
        self.dirty_sets |= 1 << set;
    }

    /// Mark all sets as dirty.
    pub fn mark_all_dirty(&mut self) {
        self.dirty_sets = (1 << MAX_DESCRIPTOR_SETS) - 1;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{Hash, Hasher};

    use ash::vk::{self, Handle};
    use rustc_hash::FxHasher;

    fn hash_of(bindings: &DescriptorSetBindings) -> u64 {
        let mut hasher = FxHasher::default();
        bindings.hash(&mut hasher);
        hasher.finish()
    }

    // --- BindingSlot Hash / Equality ---

    #[test]
    fn none_slots_are_equal() {
        assert_eq!(BindingSlot::None, BindingSlot::None);
    }

    #[test]
    fn different_variant_types_not_equal() {
        let ub = BindingSlot::UniformBuffer {
            buffer: vk::Buffer::from_raw(1),
            offset: 0,
            range: 64,
        };
        let sb = BindingSlot::StorageBuffer {
            buffer: vk::Buffer::from_raw(1),
            offset: 0,
            range: 64,
        };
        assert_ne!(ub, sb);
    }

    #[test]
    fn uniform_buffer_equality() {
        let a = BindingSlot::UniformBuffer {
            buffer: vk::Buffer::from_raw(1),
            offset: 0,
            range: 256,
        };
        let b = BindingSlot::UniformBuffer {
            buffer: vk::Buffer::from_raw(1),
            offset: 0,
            range: 256,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn uniform_buffer_different_offset() {
        let a = BindingSlot::UniformBuffer {
            buffer: vk::Buffer::from_raw(1),
            offset: 0,
            range: 256,
        };
        let b = BindingSlot::UniformBuffer {
            buffer: vk::Buffer::from_raw(1),
            offset: 64,
            range: 256,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn combined_image_sampler_equality() {
        let a = BindingSlot::CombinedImageSampler {
            view: vk::ImageView::from_raw(10),
            sampler: vk::Sampler::from_raw(20),
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let b = BindingSlot::CombinedImageSampler {
            view: vk::ImageView::from_raw(10),
            sampler: vk::Sampler::from_raw(20),
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn combined_image_sampler_different_sampler() {
        let a = BindingSlot::CombinedImageSampler {
            view: vk::ImageView::from_raw(10),
            sampler: vk::Sampler::from_raw(20),
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        let b = BindingSlot::CombinedImageSampler {
            view: vk::ImageView::from_raw(10),
            sampler: vk::Sampler::from_raw(99),
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
        assert_ne!(a, b);
    }

    // --- DescriptorSetBindings Hash / Equality ---

    #[test]
    fn empty_bindings_equal() {
        let a = DescriptorSetBindings::default();
        let b = DescriptorSetBindings::default();
        assert_eq!(a, b);
        assert_eq!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn empty_bindings_is_empty() {
        let b = DescriptorSetBindings::default();
        assert!(b.is_empty());
        assert_eq!(b.active_mask(), 0);
    }

    #[test]
    fn identical_single_binding_same_hash() {
        let make = || {
            let mut b = DescriptorSetBindings::default();
            b.bindings[0] = BindingSlot::UniformBuffer {
                buffer: vk::Buffer::from_raw(42),
                offset: 0,
                range: 128,
            };
            b.active_mask = 1;
            b
        };
        let a = make();
        let b = make();
        assert_eq!(a, b);
        assert_eq!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn different_buffer_handle_different_hash() {
        let mut a = DescriptorSetBindings::default();
        a.bindings[0] = BindingSlot::UniformBuffer {
            buffer: vk::Buffer::from_raw(1),
            offset: 0,
            range: 64,
        };
        a.active_mask = 1;

        let mut b = DescriptorSetBindings::default();
        b.bindings[0] = BindingSlot::UniformBuffer {
            buffer: vk::Buffer::from_raw(2),
            offset: 0,
            range: 64,
        };
        b.active_mask = 1;

        assert_ne!(a, b);
        assert_ne!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn different_binding_slot_different_hash() {
        // Same data in binding[0] vs binding[1]
        let mut a = DescriptorSetBindings::default();
        a.bindings[0] = BindingSlot::UniformBuffer {
            buffer: vk::Buffer::from_raw(1),
            offset: 0,
            range: 64,
        };
        a.active_mask = 1; // bit 0

        let mut b = DescriptorSetBindings::default();
        b.bindings[1] = BindingSlot::UniformBuffer {
            buffer: vk::Buffer::from_raw(1),
            offset: 0,
            range: 64,
        };
        b.active_mask = 2; // bit 1

        assert_ne!(a, b);
        assert_ne!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn different_range_different_hash() {
        let mut a = DescriptorSetBindings::default();
        a.bindings[0] = BindingSlot::UniformBuffer {
            buffer: vk::Buffer::from_raw(1),
            offset: 0,
            range: 64,
        };
        a.active_mask = 1;

        let mut b = DescriptorSetBindings::default();
        b.bindings[0] = BindingSlot::UniformBuffer {
            buffer: vk::Buffer::from_raw(1),
            offset: 0,
            range: 128,
        };
        b.active_mask = 1;

        assert_ne!(a, b);
        assert_ne!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn uniform_vs_storage_different_hash() {
        let mut a = DescriptorSetBindings::default();
        a.bindings[0] = BindingSlot::UniformBuffer {
            buffer: vk::Buffer::from_raw(1),
            offset: 0,
            range: 64,
        };
        a.active_mask = 1;

        let mut b = DescriptorSetBindings::default();
        b.bindings[0] = BindingSlot::StorageBuffer {
            buffer: vk::Buffer::from_raw(1),
            offset: 0,
            range: 64,
        };
        b.active_mask = 1;

        assert_ne!(a, b);
        assert_ne!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn multiple_bindings_hash_stability() {
        let make = || {
            let mut b = DescriptorSetBindings::default();
            b.bindings[0] = BindingSlot::UniformBuffer {
                buffer: vk::Buffer::from_raw(1),
                offset: 0,
                range: 64,
            };
            b.bindings[3] = BindingSlot::CombinedImageSampler {
                view: vk::ImageView::from_raw(10),
                sampler: vk::Sampler::from_raw(20),
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };
            b.active_mask = (1 << 0) | (1 << 3);
            b
        };
        let a = make();
        let b = make();
        assert_eq!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn inactive_slots_ignored_in_equality() {
        let mut a = DescriptorSetBindings::default();
        a.bindings[0] = BindingSlot::UniformBuffer {
            buffer: vk::Buffer::from_raw(1),
            offset: 0,
            range: 64,
        };
        a.active_mask = 1;
        // Leave bindings[5] as junk but not in active_mask
        a.bindings[5] = BindingSlot::StorageBuffer {
            buffer: vk::Buffer::from_raw(99),
            offset: 0,
            range: 256,
        };

        let mut b = DescriptorSetBindings::default();
        b.bindings[0] = BindingSlot::UniformBuffer {
            buffer: vk::Buffer::from_raw(1),
            offset: 0,
            range: 64,
        };
        b.active_mask = 1;

        // They should be equal because only active slots are compared
        assert_eq!(a, b);
        assert_eq!(hash_of(&a), hash_of(&b));
    }

    // --- BindingTable Dirty Tracking ---

    #[test]
    fn new_table_not_dirty() {
        let table = BindingTable::new();
        assert_eq!(table.dirty_sets(), 0);
    }

    #[test]
    fn set_uniform_buffer_marks_dirty() {
        let mut table = BindingTable::new();
        table.set_uniform_buffer(0, 0, vk::Buffer::from_raw(1), 0, 64);
        assert_ne!(table.dirty_sets() & 1, 0); // set 0 is dirty
    }

    #[test]
    fn set_texture_marks_dirty() {
        let mut table = BindingTable::new();
        table.set_texture(
            2,
            0,
            vk::ImageView::from_raw(1),
            vk::Sampler::from_raw(2),
        );
        assert_ne!(table.dirty_sets() & (1 << 2), 0); // set 2 is dirty
    }

    #[test]
    fn clear_dirty_clears_single_set() {
        let mut table = BindingTable::new();
        table.set_uniform_buffer(0, 0, vk::Buffer::from_raw(1), 0, 64);
        table.set_uniform_buffer(1, 0, vk::Buffer::from_raw(2), 0, 64);
        assert_ne!(table.dirty_sets() & 0b11, 0);

        table.clear_dirty(0);
        assert_eq!(table.dirty_sets() & 1, 0); // set 0 no longer dirty
        assert_ne!(table.dirty_sets() & 2, 0); // set 1 still dirty
    }

    #[test]
    fn clear_all_dirty_clears_all() {
        let mut table = BindingTable::new();
        table.set_uniform_buffer(0, 0, vk::Buffer::from_raw(1), 0, 64);
        table.set_uniform_buffer(1, 0, vk::Buffer::from_raw(2), 0, 64);
        table.set_uniform_buffer(2, 0, vk::Buffer::from_raw(3), 0, 64);
        table.clear_all_dirty();
        assert_eq!(table.dirty_sets(), 0);
    }

    #[test]
    fn mark_dirty_forces_dirty() {
        let mut table = BindingTable::new();
        table.mark_dirty(3);
        assert_ne!(table.dirty_sets() & (1 << 3), 0);
    }

    #[test]
    fn mark_all_dirty() {
        let mut table = BindingTable::new();
        table.mark_all_dirty();
        assert_eq!(table.dirty_sets(), (1 << MAX_DESCRIPTOR_SETS) - 1);
    }

    #[test]
    fn clear_set_marks_dirty_and_empties_bindings() {
        let mut table = BindingTable::new();
        table.set_uniform_buffer(1, 0, vk::Buffer::from_raw(1), 0, 64);
        table.clear_all_dirty();

        table.clear_set(1);
        assert_ne!(table.dirty_sets() & (1 << 1), 0); // clearing marks dirty
        assert!(table.set(1).is_empty());
    }

    #[test]
    fn clear_all_marks_all_dirty() {
        let mut table = BindingTable::new();
        table.set_uniform_buffer(0, 0, vk::Buffer::from_raw(1), 0, 64);
        table.clear_all_dirty();

        table.clear_all();
        assert_eq!(table.dirty_sets(), (1 << MAX_DESCRIPTOR_SETS) - 1);
        for i in 0..MAX_DESCRIPTOR_SETS {
            assert!(table.set(i as u32).is_empty());
        }
    }

    #[test]
    fn set_storage_buffer_updates_active_mask() {
        let mut table = BindingTable::new();
        table.set_storage_buffer(0, 5, vk::Buffer::from_raw(1), 0, 128);
        assert_ne!(table.set(0).active_mask() & (1 << 5), 0);
    }

    #[test]
    fn set_storage_image_updates_active_mask() {
        let mut table = BindingTable::new();
        table.set_storage_image(0, 2, vk::ImageView::from_raw(1));
        assert_ne!(table.set(0).active_mask() & (1 << 2), 0);
    }

    #[test]
    fn set_input_attachment_updates_active_mask() {
        let mut table = BindingTable::new();
        table.set_input_attachment(0, 7, vk::ImageView::from_raw(1));
        assert_ne!(table.set(0).active_mask() & (1 << 7), 0);
    }

    #[test]
    fn set_acceleration_structure_updates_active_mask() {
        let mut table = BindingTable::new();
        table.set_acceleration_structure(
            0,
            0,
            vk::AccelerationStructureKHR::from_raw(1),
        );
        assert_ne!(table.set(0).active_mask() & 1, 0);
    }

    #[test]
    fn overwrite_binding_preserves_active_mask() {
        let mut table = BindingTable::new();
        table.set_uniform_buffer(0, 0, vk::Buffer::from_raw(1), 0, 64);
        table.set_uniform_buffer(0, 0, vk::Buffer::from_raw(2), 0, 128);
        // Still active at binding 0
        assert_ne!(table.set(0).active_mask() & 1, 0);
    }

    #[test]
    fn hash_consistency_across_table_access() {
        let mut table = BindingTable::new();
        table.set_uniform_buffer(0, 0, vk::Buffer::from_raw(1), 0, 64);
        table.set_texture(
            0,
            1,
            vk::ImageView::from_raw(10),
            vk::Sampler::from_raw(20),
        );

        let h1 = hash_of(table.set(0));
        let h2 = hash_of(table.set(0));
        assert_eq!(h1, h2);
    }
}
