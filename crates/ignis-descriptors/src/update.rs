//! Descriptor update templates and helpers.
//!
//! Wraps `vkUpdateDescriptorSetWithTemplate` for efficient descriptor writes.

/// Cached descriptor update template for a specific layout.
pub struct DescriptorUpdateTemplate {
    // TODO: Phase 3, Iteration 3.3
    // - vk::DescriptorUpdateTemplate handle
    // - layout info
    _private: (),
}
