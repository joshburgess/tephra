//! Temporal resource aliasing.
//!
//! Resources with non-overlapping lifetimes can share the same physical
//! `VkImage`, reducing memory usage.

/// Analyzes resource lifetimes and assigns aliasing groups.
pub struct ResourceAliaser {
    // TODO: Phase 6, Iteration 6.5
    _private: (),
}
