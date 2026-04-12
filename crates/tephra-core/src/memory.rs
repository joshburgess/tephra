//! Memory domain enums and allocation helpers.
//!
//! Provides a simplified memory model mapping to Vulkan memory properties
//! through `gpu-allocator`.

use gpu_allocator::MemoryLocation;

/// Where a buffer's memory should be allocated.
///
/// This maps to `gpu-allocator` memory locations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryDomain {
    /// GPU-only memory (`DEVICE_LOCAL`). Fastest for GPU access, not CPU-visible.
    Device,
    /// CPU-to-GPU upload memory (`HOST_VISIBLE | HOST_COHERENT`).
    Host,
    /// GPU-to-CPU readback memory (`HOST_VISIBLE | HOST_CACHED`).
    CachedHost,
}

impl MemoryDomain {
    /// Convert to `gpu-allocator` memory location.
    pub fn to_gpu_allocator(self) -> MemoryLocation {
        match self {
            MemoryDomain::Device => MemoryLocation::GpuOnly,
            MemoryDomain::Host => MemoryLocation::CpuToGpu,
            MemoryDomain::CachedHost => MemoryLocation::GpuToCpu,
        }
    }
}

/// Where an image's memory should be allocated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageDomain {
    /// Physical image backed by device-local memory.
    Physical,
    /// Transient attachment — may use lazily allocated memory.
    Transient,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_maps_to_gpu_only() {
        assert_eq!(
            MemoryDomain::Device.to_gpu_allocator(),
            MemoryLocation::GpuOnly
        );
    }

    #[test]
    fn host_maps_to_cpu_to_gpu() {
        assert_eq!(
            MemoryDomain::Host.to_gpu_allocator(),
            MemoryLocation::CpuToGpu
        );
    }

    #[test]
    fn cached_host_maps_to_gpu_to_cpu() {
        assert_eq!(
            MemoryDomain::CachedHost.to_gpu_allocator(),
            MemoryLocation::GpuToCpu
        );
    }

    #[test]
    fn memory_domain_equality() {
        assert_eq!(MemoryDomain::Device, MemoryDomain::Device);
        assert_ne!(MemoryDomain::Device, MemoryDomain::Host);
        assert_ne!(MemoryDomain::Host, MemoryDomain::CachedHost);
    }

    #[test]
    fn image_domain_equality() {
        assert_eq!(ImageDomain::Physical, ImageDomain::Physical);
        assert_ne!(ImageDomain::Physical, ImageDomain::Transient);
    }

    #[test]
    fn memory_domain_is_hashable() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(MemoryDomain::Device);
        set.insert(MemoryDomain::Host);
        set.insert(MemoryDomain::CachedHost);
        assert_eq!(set.len(), 3);
        // Inserting duplicate doesn't increase count
        set.insert(MemoryDomain::Device);
        assert_eq!(set.len(), 3);
    }
}
