//! Buffer creation helpers and metadata.

use ash::vk;
use crate::memory::MemoryDomain;

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
