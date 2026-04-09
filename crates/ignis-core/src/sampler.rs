//! Sampler cache and stock sampler presets.
//!
//! Provides a hash-and-cache layer for Vulkan samplers and a set of commonly
//! used sampler presets via [`StockSampler`].

use ash::vk;

/// Commonly used sampler configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StockSampler {
    /// Nearest filtering, clamp to edge.
    NearestClamp,
    /// Nearest filtering, repeat wrapping.
    NearestWrap,
    /// Linear filtering, clamp to edge.
    LinearClamp,
    /// Linear filtering, repeat wrapping.
    LinearWrap,
    /// Trilinear (linear + mipmap linear) filtering, clamp to edge.
    TrilinearClamp,
    /// Trilinear filtering, repeat wrapping.
    TrilinearWrap,
    /// Nearest filtering with depth comparison, for shadow maps.
    NearestShadow,
    /// Linear filtering with depth comparison, for shadow maps.
    LinearShadow,
}

/// Cache for Vulkan samplers, keyed by their create info.
pub(crate) struct SamplerCache {
    // TODO: Phase 1, Iteration 1.4
    // - FxHashMap<SamplerKey, vk::Sampler>
    // - stock_samplers: [vk::Sampler; StockSampler variant count]
    _private: (),
}

impl SamplerCache {
    /// Create a new empty sampler cache.
    pub fn new() -> Self {
        Self { _private: () }
    }
}
