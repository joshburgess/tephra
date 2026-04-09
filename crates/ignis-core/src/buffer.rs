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
