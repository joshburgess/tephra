//! Subpass merging for tile-based deferred renderers (TBDR).
//!
//! Identifies consecutive passes that can be merged into a single
//! `VkRenderPass` with multiple subpasses, critical for mobile and
//! Apple Silicon GPUs.

/// Analyzes passes for subpass merge opportunities.
pub struct SubpassMerger {
    // TODO: Phase 6, Iteration 6.4
    _private: (),
}
