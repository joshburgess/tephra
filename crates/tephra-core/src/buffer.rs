//! Buffer creation helpers and metadata.

use crate::memory::MemoryDomain;
use ash::vk;

/// Metadata describing a buffer's properties.
#[derive(Debug, Clone)]
pub struct BufferInfo {
    /// Size in bytes.
    pub size: vk::DeviceSize,
    /// Vulkan buffer usage flags.
    pub usage: vk::BufferUsageFlags,
    /// Memory domain for this buffer.
    pub domain: MemoryDomain,
}

/// Parameters for creating a new buffer.
pub struct BufferCreateInfo {
    /// Size in bytes.
    pub size: vk::DeviceSize,
    /// Vulkan buffer usage flags.
    pub usage: vk::BufferUsageFlags,
    /// Memory domain.
    pub domain: MemoryDomain,
}

impl BufferCreateInfo {
    /// Convenience: a device-local vertex buffer.
    pub fn vertex(size: vk::DeviceSize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            domain: MemoryDomain::Device,
        }
    }

    /// Convenience: a device-local index buffer.
    pub fn index(size: vk::DeviceSize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            domain: MemoryDomain::Device,
        }
    }

    /// Convenience: a host-visible uniform buffer.
    pub fn uniform(size: vk::DeviceSize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            domain: MemoryDomain::Host,
        }
    }

    /// Convenience: a host-visible staging buffer.
    pub fn staging(size: vk::DeviceSize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            domain: MemoryDomain::Host,
        }
    }

    /// Convenience: a device-local storage buffer.
    pub fn storage(size: vk::DeviceSize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            domain: MemoryDomain::Device,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_buffer_info() {
        let info = BufferCreateInfo::vertex(1024);
        assert_eq!(info.size, 1024);
        assert!(info.usage.contains(vk::BufferUsageFlags::VERTEX_BUFFER));
        assert!(info.usage.contains(vk::BufferUsageFlags::TRANSFER_DST));
        assert_eq!(info.domain, MemoryDomain::Device);
    }

    #[test]
    fn index_buffer_info() {
        let info = BufferCreateInfo::index(512);
        assert_eq!(info.size, 512);
        assert!(info.usage.contains(vk::BufferUsageFlags::INDEX_BUFFER));
        assert!(info.usage.contains(vk::BufferUsageFlags::TRANSFER_DST));
        assert_eq!(info.domain, MemoryDomain::Device);
    }

    #[test]
    fn uniform_buffer_info() {
        let info = BufferCreateInfo::uniform(256);
        assert_eq!(info.size, 256);
        assert!(info.usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER));
        assert_eq!(info.domain, MemoryDomain::Host);
    }

    #[test]
    fn staging_buffer_info() {
        let info = BufferCreateInfo::staging(4096);
        assert_eq!(info.size, 4096);
        assert!(info.usage.contains(vk::BufferUsageFlags::TRANSFER_SRC));
        assert_eq!(info.domain, MemoryDomain::Host);
    }

    #[test]
    fn storage_buffer_info() {
        let info = BufferCreateInfo::storage(2048);
        assert_eq!(info.size, 2048);
        assert!(info.usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER));
        assert!(info.usage.contains(vk::BufferUsageFlags::TRANSFER_DST));
        assert_eq!(info.domain, MemoryDomain::Device);
    }

    #[test]
    fn vertex_not_transfer_src() {
        let info = BufferCreateInfo::vertex(64);
        assert!(!info.usage.contains(vk::BufferUsageFlags::TRANSFER_SRC));
    }

    #[test]
    fn staging_not_transfer_dst() {
        let info = BufferCreateInfo::staging(64);
        assert!(!info.usage.contains(vk::BufferUsageFlags::TRANSFER_DST));
    }
}
