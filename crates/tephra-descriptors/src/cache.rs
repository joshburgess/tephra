//! Hash-and-cache layer for descriptor sets.
//!
//! On draw/dispatch, dirty sets are hashed and looked up in a per-frame cache.
//! Cache misses allocate a new set from the slab allocator and write descriptors.

use std::hash::{Hash, Hasher};

use ash::vk;
use rustc_hash::{FxHashMap, FxHasher};

use crate::binding_table::{BindingSlot, DescriptorSetBindings};
use crate::set_allocator::DescriptorSetAllocator;

/// Intermediate entry for building descriptor writes.
struct WriteEntry {
    binding: u32,
    descriptor_type: vk::DescriptorType,
    buffer_info_idx: Option<usize>,
    image_info_idx: Option<usize>,
    accel_struct_idx: Option<usize>,
}

/// Prepared descriptor writes from a set of bindings.
///
/// Owns the backing `DescriptorBufferInfo` and `DescriptorImageInfo` arrays
/// so that `build_writes()` can produce `WriteDescriptorSet` references that
/// borrow from this struct.
pub struct PreparedDescriptorWrites {
    buffer_infos: Vec<vk::DescriptorBufferInfo>,
    image_infos: Vec<vk::DescriptorImageInfo>,
    accel_structs: Vec<vk::AccelerationStructureKHR>,
    entries: Vec<WriteEntry>,
}

impl PreparedDescriptorWrites {
    /// Build prepared writes from the given binding state.
    pub fn from_bindings(bindings: &DescriptorSetBindings) -> Self {
        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
        let mut image_infos: Vec<vk::DescriptorImageInfo> = Vec::new();
        let mut accel_structs: Vec<vk::AccelerationStructureKHR> = Vec::new();
        let mut entries: Vec<WriteEntry> = Vec::new();

        let mut mask = bindings.active_mask;
        while mask != 0 {
            let idx = mask.trailing_zeros() as usize;
            match &bindings.bindings[idx] {
                BindingSlot::None => {}
                BindingSlot::UniformBuffer {
                    buffer,
                    offset,
                    range,
                } => {
                    let buf_idx = buffer_infos.len();
                    buffer_infos.push(
                        vk::DescriptorBufferInfo::default()
                            .buffer(*buffer)
                            .offset(*offset)
                            .range(*range),
                    );
                    entries.push(WriteEntry {
                        binding: idx as u32,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        buffer_info_idx: Some(buf_idx),
                        image_info_idx: None,
                        accel_struct_idx: None,
                    });
                }
                BindingSlot::StorageBuffer {
                    buffer,
                    offset,
                    range,
                } => {
                    let buf_idx = buffer_infos.len();
                    buffer_infos.push(
                        vk::DescriptorBufferInfo::default()
                            .buffer(*buffer)
                            .offset(*offset)
                            .range(*range),
                    );
                    entries.push(WriteEntry {
                        binding: idx as u32,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        buffer_info_idx: Some(buf_idx),
                        image_info_idx: None,
                        accel_struct_idx: None,
                    });
                }
                BindingSlot::CombinedImageSampler {
                    view,
                    sampler,
                    layout,
                } => {
                    let img_idx = image_infos.len();
                    image_infos.push(
                        vk::DescriptorImageInfo::default()
                            .image_view(*view)
                            .sampler(*sampler)
                            .image_layout(*layout),
                    );
                    entries.push(WriteEntry {
                        binding: idx as u32,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        buffer_info_idx: None,
                        image_info_idx: Some(img_idx),
                        accel_struct_idx: None,
                    });
                }
                BindingSlot::StorageImage { view, layout } => {
                    let img_idx = image_infos.len();
                    image_infos.push(
                        vk::DescriptorImageInfo::default()
                            .image_view(*view)
                            .image_layout(*layout),
                    );
                    entries.push(WriteEntry {
                        binding: idx as u32,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        buffer_info_idx: None,
                        image_info_idx: Some(img_idx),
                        accel_struct_idx: None,
                    });
                }
                BindingSlot::InputAttachment { view, layout } => {
                    let img_idx = image_infos.len();
                    image_infos.push(
                        vk::DescriptorImageInfo::default()
                            .image_view(*view)
                            .image_layout(*layout),
                    );
                    entries.push(WriteEntry {
                        binding: idx as u32,
                        descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                        buffer_info_idx: None,
                        image_info_idx: Some(img_idx),
                        accel_struct_idx: None,
                    });
                }
                BindingSlot::AccelerationStructure { handle } => {
                    let as_idx = accel_structs.len();
                    accel_structs.push(*handle);
                    entries.push(WriteEntry {
                        binding: idx as u32,
                        descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                        buffer_info_idx: None,
                        image_info_idx: None,
                        accel_struct_idx: Some(as_idx),
                    });
                }
            }
            mask &= mask - 1;
        }

        Self {
            buffer_infos,
            image_infos,
            accel_structs,
            entries,
        }
    }

    /// Whether there are any writes to apply.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Build `VkWriteDescriptorSet` array targeting the given descriptor set.
    ///
    /// For push descriptors, pass `vk::DescriptorSet::null()`.
    /// Returns a [`DescriptorWriteSet`] that owns the writes and any acceleration
    /// structure chain structs needed for their lifetimes.
    pub fn build_writes(&self, dst_set: vk::DescriptorSet) -> DescriptorWriteSet<'_> {
        // Pre-build acceleration structure write infos so they outlive the writes
        let mut accel_write_infos: Vec<vk::WriteDescriptorSetAccelerationStructureKHR<'_>> =
            Vec::new();
        for entry in &self.entries {
            if let Some(idx) = entry.accel_struct_idx {
                accel_write_infos.push(
                    vk::WriteDescriptorSetAccelerationStructureKHR::default()
                        .acceleration_structures(&self.accel_structs[idx..idx + 1]),
                );
            }
        }

        let mut writes = Vec::with_capacity(self.entries.len());
        let mut accel_idx = 0;
        for entry in &self.entries {
            let mut write = vk::WriteDescriptorSet::default()
                .dst_set(dst_set)
                .dst_binding(entry.binding)
                .dst_array_element(0)
                .descriptor_type(entry.descriptor_type);

            if let Some(idx) = entry.buffer_info_idx {
                write = write.buffer_info(&self.buffer_infos[idx..idx + 1]);
            }
            if let Some(idx) = entry.image_info_idx {
                write = write.image_info(&self.image_infos[idx..idx + 1]);
            }
            if entry.accel_struct_idx.is_some() {
                // Chain the acceleration structure info via p_next.
                // SAFETY: accel_write_infos outlives writes within DescriptorWriteSet.
                write.descriptor_count = 1;
                write.p_next = &accel_write_infos[accel_idx]
                    as *const vk::WriteDescriptorSetAccelerationStructureKHR<'_>
                    as *const std::ffi::c_void;
                accel_idx += 1;
            }

            writes.push(write);
        }

        DescriptorWriteSet {
            writes,
            _accel_write_infos: accel_write_infos,
        }
    }

    /// Build writes and apply them to a descriptor set via `vkUpdateDescriptorSets`.
    pub fn apply(&self, device: &ash::Device, dst_set: vk::DescriptorSet) {
        if self.is_empty() {
            return;
        }
        let write_set = self.build_writes(dst_set);
        // SAFETY: device, set, and all referenced handles are valid.
        unsafe {
            device.update_descriptor_sets(write_set.writes(), &[]);
        }
    }
}

/// Owns descriptor writes and any chained extension structs needed for their lifetimes.
///
/// Acceleration structure descriptor writes require a
/// `WriteDescriptorSetAccelerationStructureKHR` chained via `p_next`. This struct
/// keeps both the writes and the chain structs alive together.
pub struct DescriptorWriteSet<'a> {
    writes: Vec<vk::WriteDescriptorSet<'a>>,
    _accel_write_infos: Vec<vk::WriteDescriptorSetAccelerationStructureKHR<'a>>,
}

impl<'a> DescriptorWriteSet<'a> {
    /// Get the descriptor writes as a slice.
    pub fn writes(&self) -> &[vk::WriteDescriptorSet<'a>] {
        &self.writes
    }

    /// Whether there are any writes.
    pub fn is_empty(&self) -> bool {
        self.writes.is_empty()
    }
}

/// Per-frame descriptor set cache for a single layout.
///
/// Caches descriptor sets by their binding state hash. Reset when the
/// frame context recycles (which also resets the underlying pool).
pub struct DescriptorSetCache {
    cache: FxHashMap<u64, vk::DescriptorSet>,
}

impl DescriptorSetCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            cache: FxHashMap::default(),
        }
    }

    /// Look up or allocate+write a descriptor set for the given bindings.
    ///
    /// Returns the `VkDescriptorSet` to bind.
    pub fn get_or_allocate(
        &mut self,
        device: &ash::Device,
        allocator: &mut DescriptorSetAllocator,
        bindings: &DescriptorSetBindings,
    ) -> Result<vk::DescriptorSet, vk::Result> {
        let hash = Self::hash_bindings(bindings);

        if let Some(&set) = self.cache.get(&hash) {
            return Ok(set);
        }

        // Cache miss: allocate a new set and write descriptors
        let set = allocator.allocate(device)?;
        let prepared = PreparedDescriptorWrites::from_bindings(bindings);
        prepared.apply(device, set);
        self.cache.insert(hash, set);

        Ok(set)
    }

    /// Clear the cache. Called when the frame context resets.
    pub fn reset(&mut self) {
        self.cache.clear();
    }

    fn hash_bindings(bindings: &DescriptorSetBindings) -> u64 {
        let mut hasher = FxHasher::default();
        bindings.hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for DescriptorSetCache {
    fn default() -> Self {
        Self::new()
    }
}
