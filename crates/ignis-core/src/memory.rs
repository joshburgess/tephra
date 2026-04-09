//! Memory domain enums and allocation helpers.
//!
//! Provides a simplified memory model mapping to Vulkan memory properties.

/// Where a resource's memory should be allocated.
///
/// This maps to Vulkan memory property flags via `gpu-allocator`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryDomain {
    /// GPU-only memory (`DEVICE_LOCAL`). Fastest for GPU access, not CPU-visible.
    Device,
    /// CPU-to-GPU upload memory (`HOST_VISIBLE | HOST_COHERENT`).
    Host,
    /// GPU-to-CPU readback memory (`HOST_VISIBLE | HOST_CACHED`).
    CachedHost,
}

/// Where an image's memory should be allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageDomain {
    /// Physical image backed by device-local memory.
    Physical,
    /// Transient attachment — may use lazily allocated memory.
    Transient,
}
