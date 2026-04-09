//! `VkPipelineCache` disk persistence.
//!
//! Serializes the Vulkan pipeline cache to disk on shutdown and loads it
//! on startup to avoid redundant pipeline compilations.

/// Manages `VkPipelineCache` creation, loading, and saving.
pub struct PipelineCacheManager {
    // TODO: Phase 4, Iteration 4.5
    _private: (),
}
