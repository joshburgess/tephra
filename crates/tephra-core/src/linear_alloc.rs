//! Bump allocator for transient per-frame data (VBO, IBO, UBO).
//!
//! Each frame context owns a `LinearAllocatorPool` that provides fast,
//! zero-overhead allocations from pre-mapped host-visible buffers. All
//! allocations are reset at the start of each frame.

use std::ptr::NonNull;

use ash::vk;
use gpu_allocator::vulkan as vma;

use crate::device::DeviceError;

/// Default buffer size for each linear allocator block: 256 KB.
const DEFAULT_BLOCK_SIZE: usize = 256 * 1024;

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

/// A single linear (bump) allocator backed by a mapped Vulkan buffer.
pub(crate) struct LinearAllocator {
    buffer: vk::Buffer,
    allocation: vma::Allocation,
    mapped_ptr: NonNull<u8>,
    offset: usize,
    capacity: usize,
}

impl LinearAllocator {
    /// Try to allocate `size` bytes with the given alignment.
    ///
    /// Returns `None` if there isn't enough space in this allocator.
    fn allocate(&mut self, size: usize, alignment: usize) -> Option<TransientAllocation> {
        // Align the current offset up
        let aligned_offset = (self.offset + alignment - 1) & !(alignment - 1);
        if aligned_offset + size > self.capacity {
            return None;
        }

        // SAFETY: mapped_ptr is valid and the range [aligned_offset..aligned_offset+size]
        // is within the buffer's mapped region. The pointer arithmetic is in-bounds.
        let ptr = unsafe { NonNull::new_unchecked(self.mapped_ptr.as_ptr().add(aligned_offset)) };

        let result = TransientAllocation {
            ptr,
            buffer: self.buffer,
            offset: aligned_offset as vk::DeviceSize,
            size: size as vk::DeviceSize,
        };

        self.offset = aligned_offset + size;
        Some(result)
    }

    /// Reset the allocator for reuse (sets offset to 0).
    fn reset(&mut self) {
        self.offset = 0;
    }
}

/// Pool of linear allocators, recycled per frame.
///
/// Active allocators serve allocation requests. When a frame resets, all
/// active allocators move to the free list for reuse.
pub(crate) struct LinearAllocatorPool {
    active: Vec<LinearAllocator>,
    free_list: Vec<LinearAllocator>,
    block_size: usize,
}

impl LinearAllocatorPool {
    /// Create a new empty allocator pool.
    pub fn new() -> Self {
        Self {
            active: Vec::new(),
            free_list: Vec::new(),
            block_size: DEFAULT_BLOCK_SIZE,
        }
    }

    /// Allocate `size` bytes with the given alignment from the pool.
    ///
    /// Tries the current active allocator first. If it's full, promotes a
    /// free-list allocator or creates a new one.
    pub fn allocate(
        &mut self,
        device: &ash::Device,
        allocator: &mut vma::Allocator,
        size: usize,
        alignment: usize,
    ) -> Result<TransientAllocation, DeviceError> {
        // Try the last active allocator
        if let Some(active) = self.active.last_mut() {
            if let Some(alloc) = active.allocate(size, alignment) {
                return Ok(alloc);
            }
        }

        // Current block is full (or none exists). Get a new one.
        let block_size = self.block_size.max(size);
        let mut new_block = self.get_or_create_block(device, allocator, block_size)?;
        let result = new_block
            .allocate(size, alignment)
            .expect("fresh block should have enough space");
        self.active.push(new_block);
        Ok(result)
    }

    /// Reset all active allocators and move them to the free list.
    pub fn reset(&mut self) {
        for mut alloc in self.active.drain(..) {
            alloc.reset();
            self.free_list.push(alloc);
        }
    }

    /// Destroy all allocators, freeing their Vulkan resources.
    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut vma::Allocator) {
        for block in self.active.drain(..).chain(self.free_list.drain(..)) {
            // SAFETY: device is valid, buffer is valid, GPU is idle.
            unsafe {
                device.destroy_buffer(block.buffer, None);
            }
            allocator.free(block.allocation).ok();
        }
    }

    fn get_or_create_block(
        &mut self,
        device: &ash::Device,
        allocator: &mut vma::Allocator,
        size: usize,
    ) -> Result<LinearAllocator, DeviceError> {
        // Try to reuse a block from the free list that's large enough
        if let Some(idx) = self.free_list.iter().position(|b| b.capacity >= size) {
            return Ok(self.free_list.swap_remove(idx));
        }

        // Create a new block
        Self::create_block(device, allocator, size)
    }

    fn create_block(
        device: &ash::Device,
        allocator: &mut vma::Allocator,
        size: usize,
    ) -> Result<LinearAllocator, DeviceError> {
        let buffer_ci = vk::BufferCreateInfo::default()
            .size(size as vk::DeviceSize)
            .usage(
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: device is valid, buffer_ci is well-formed.
        let buffer = unsafe { device.create_buffer(&buffer_ci, None)? };

        // SAFETY: device and buffer are valid.
        let mem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = allocator
            .allocate(&vma::AllocationCreateDesc {
                name: "linear_allocator_block",
                requirements: mem_reqs,
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: vma::AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| DeviceError::AllocationFailed(e.to_string()))?;

        // SAFETY: buffer, memory, and offset are valid and compatible.
        unsafe {
            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        }

        let mapped_ptr = allocation
            .mapped_ptr()
            .expect("CpuToGpu allocation must be mapped")
            .cast::<u8>();

        log::debug!("Created linear allocator block ({} KB)", size / 1024);

        Ok(LinearAllocator {
            buffer,
            allocation,
            mapped_ptr,
            offset: 0,
            capacity: size,
        })
    }
}
