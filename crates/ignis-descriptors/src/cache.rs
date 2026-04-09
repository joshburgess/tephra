//! Hash-and-cache layer for descriptor sets.
//!
//! On draw/dispatch, dirty sets are hashed and looked up in a per-frame cache.
//! Cache misses allocate a new set and write descriptors.

/// Per-frame descriptor set cache.
pub struct DescriptorSetCache {
    // TODO: Phase 3, Iteration 3.3
    // - FxHashMap<u64, vk::DescriptorSet>
    _private: (),
}

impl DescriptorSetCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self { _private: () }
    }
}
