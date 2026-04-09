//! Bump allocator for transient per-frame data (VBO, IBO, UBO).
//!
//! Each frame context owns a [`LinearAllocatorPool`] that provides fast,
//! zero-overhead allocations from pre-mapped host-visible buffers. All
//! allocations are reset at the start of each frame.

use ash::vk;
use std::ptr::NonNull;

/// A single linear (bump) allocator backed by a mapped Vulkan buffer.
pub(crate) struct LinearAllocator {
    pub buffer: vk::Buffer,
    pub mapped_ptr: *mut u8,
    pub offset: usize,
    pub capacity: usize,
}

/// Result of a transient allocation from a linear allocator.
pub struct TransientAllocation {
    /// Pointer to the mapped memory region.
    pub ptr: NonNull<u8>,
    /// The Vulkan buffer containing this allocation.
    pub buffer: vk::Buffer,
    /// Byte offset within the buffer.
    pub offset: vk::DeviceSize,
    /// Size of the allocation in bytes.
    pub size: vk::DeviceSize,
}

/// Pool of linear allocators, recycled per frame.
pub(crate) struct LinearAllocatorPool {
    // TODO: Phase 2, Iteration 2.2
    // - active: Vec<LinearAllocator>
    // - free_list: Vec<LinearAllocator>
    // - default_capacity: usize
    _private: (),
}

impl LinearAllocatorPool {
    /// Create a new empty allocator pool.
    pub fn new() -> Self {
        Self { _private: () }
    }
}
