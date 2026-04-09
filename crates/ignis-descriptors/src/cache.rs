//! Hash-and-cache layer for descriptor sets.
//!
//! On draw/dispatch, dirty sets are hashed and looked up in a per-frame cache.
//! Cache misses allocate a new set from the slab allocator and write descriptors.

use std::hash::{Hash, Hasher};

use ash::vk;
use rustc_hash::{FxHashMap, FxHasher};

use crate::binding_table::{BindingSlot, DescriptorSetBindings};
use crate::set_allocator::DescriptorSetAllocator;

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
        Self::write_descriptors(device, set, bindings);
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

    fn write_descriptors(
        device: &ash::Device,
        set: vk::DescriptorSet,
        bindings: &DescriptorSetBindings,
    ) {
        // Collect descriptor writes
        // We need to keep the info structs alive while building the writes
        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
        let mut image_infos: Vec<vk::DescriptorImageInfo> = Vec::new();
        // Track which indices correspond to which writes
        struct WriteEntry {
            binding: u32,
            descriptor_type: vk::DescriptorType,
            buffer_info_idx: Option<usize>,
            image_info_idx: Option<usize>,
        }
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
                    });
                }
            }
            mask &= mask - 1;
        }

        if entries.is_empty() {
            return;
        }

        // Build the VkWriteDescriptorSet array
        let writes: Vec<vk::WriteDescriptorSet<'_>> = entries
            .iter()
            .map(|entry| {
                let mut write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(entry.binding)
                    .dst_array_element(0)
                    .descriptor_type(entry.descriptor_type);

                if let Some(idx) = entry.buffer_info_idx {
                    write = write.buffer_info(&buffer_infos[idx..idx + 1]);
                }
                if let Some(idx) = entry.image_info_idx {
                    write = write.image_info(&image_infos[idx..idx + 1]);
                }

                write
            })
            .collect();

        // SAFETY: device, set, and all referenced handles are valid.
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }
}

impl Default for DescriptorSetCache {
    fn default() -> Self {
        Self::new()
    }
}
