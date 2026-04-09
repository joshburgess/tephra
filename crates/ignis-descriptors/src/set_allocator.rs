//! Per-layout descriptor set slab allocator.
//!
//! Each unique `VkDescriptorSetLayout` gets its own allocator that manages
//! a list of `VkDescriptorPool` slabs. Pools are reset per-frame rather
//! than freeing individual sets.

/// Allocator for descriptor sets of a single layout.
pub struct DescriptorSetAllocator {
    // TODO: Phase 3, Iteration 3.1
    // - layout: vk::DescriptorSetLayout
    // - slabs: Vec<vk::DescriptorPool>
    // - pool_sizes derived from layout
    _private: (),
}
