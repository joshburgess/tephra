//! Streaming buffer pools for per-frame sub-allocation.
//!
//! [`BufferPool`] provides ring-buffer style sub-allocation from large mapped
//! buffers. Each pool targets a specific usage (vertex, index, uniform, staging)
//! and allocates from pre-mapped host-visible buffers with a bump pointer.
//!
//! Blocks are recycled per-frame: at the start of each frame, blocks used
//! `FRAME_OVERLAP` frames ago are returned to the free list.

use std::ptr::NonNull;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan as vma;

use crate::device::DeviceError;

/// Result of a buffer pool sub-allocation.
pub struct BufferSubAllocation {
    /// The Vulkan buffer handle.
    pub buffer: vk::Buffer,
    /// Byte offset within the buffer.
    pub offset: vk::DeviceSize,
    /// Size of the allocation in bytes.
    pub size: vk::DeviceSize,
    /// Pointer to the mapped memory region.
    pub ptr: NonNull<u8>,
}

/// A single buffer block used for sub-allocation.
struct BufferBlock {
    buffer: vk::Buffer,
    allocation: vma::Allocation,
    mapped_ptr: NonNull<u8>,
    offset: usize,
    capacity: usize,
}

impl BufferBlock {
    fn try_allocate(&mut self, size: usize, alignment: usize) -> Option<BufferSubAllocation> {
        let aligned = (self.offset + alignment - 1) & !(alignment - 1);
        if aligned + size > self.capacity {
            return None;
        }

        // SAFETY: mapped_ptr is valid, the range is within the allocation.
        let ptr = unsafe { NonNull::new_unchecked(self.mapped_ptr.as_ptr().add(aligned)) };

        let result = BufferSubAllocation {
            buffer: self.buffer,
            offset: aligned as vk::DeviceSize,
            size: size as vk::DeviceSize,
            ptr,
        };

        self.offset = aligned + size;
        Some(result)
    }

    fn reset(&mut self) {
        self.offset = 0;
    }
}

/// Streaming buffer pool for a specific usage type.
///
/// Pre-allocates large mapped buffers and sub-allocates from them with a bump
/// pointer. Blocks are recycled per-frame after the GPU finishes using them.
pub struct BufferPool {
    usage: vk::BufferUsageFlags,
    memory_location: MemoryLocation,
    block_size: usize,
    active_block: Option<BufferBlock>,
    full_blocks: Vec<BufferBlock>,
    free_blocks: Vec<BufferBlock>,
}

impl BufferPool {
    /// Create a new buffer pool.
    ///
    /// - `usage`: Vulkan buffer usage flags (e.g., `VERTEX_BUFFER`, `UNIFORM_BUFFER`).
    /// - `memory_location`: Where to allocate (usually `CpuToGpu` for streaming).
    /// - `block_size`: Size of each backing buffer in bytes.
    pub fn new(
        usage: vk::BufferUsageFlags,
        memory_location: MemoryLocation,
        block_size: usize,
    ) -> Self {
        Self {
            usage,
            memory_location,
            block_size,
            active_block: None,
            full_blocks: Vec::new(),
            free_blocks: Vec::new(),
        }
    }

    /// Sub-allocate `size` bytes with the given alignment.
    ///
    /// Returns a [`BufferSubAllocation`] with the buffer, offset, and mapped pointer.
    /// The caller can write data directly through the pointer.
    pub fn allocate(
        &mut self,
        device: &ash::Device,
        allocator: &mut vma::Allocator,
        size: usize,
        alignment: usize,
    ) -> Result<BufferSubAllocation, DeviceError> {
        // Try the active block first
        if let Some(block) = &mut self.active_block {
            if let Some(alloc) = block.try_allocate(size, alignment) {
                return Ok(alloc);
            }
            // Active block is full — move it to full list
        }

        // Move full active block aside
        if let Some(block) = self.active_block.take() {
            self.full_blocks.push(block);
        }

        // Try to reuse a free block
        if let Some(mut block) = self.free_blocks.pop() {
            block.reset();
            if let Some(alloc) = block.try_allocate(size, alignment) {
                self.active_block = Some(block);
                return Ok(alloc);
            }
            // Free block too small (shouldn't happen with uniform block sizes)
            self.free_blocks.push(block);
        }

        // Allocate a new block
        let actual_size = self.block_size.max(size + alignment);
        let mut block = self.create_block(device, allocator, actual_size)?;
        let alloc = block
            .try_allocate(size, alignment)
            .expect("freshly created block must fit the allocation");
        self.active_block = Some(block);
        Ok(alloc)
    }

    /// Reset the pool for a new frame.
    ///
    /// Moves all used blocks to the free list. Call this when the GPU has
    /// finished using the data (after fence/timeline wait).
    pub fn reset(&mut self) {
        if let Some(mut block) = self.active_block.take() {
            block.reset();
            self.free_blocks.push(block);
        }
        for mut block in self.full_blocks.drain(..) {
            block.reset();
            self.free_blocks.push(block);
        }
    }

    /// Destroy all blocks and free GPU memory.
    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut vma::Allocator) {
        let blocks = self
            .active_block
            .take()
            .into_iter()
            .chain(self.full_blocks.drain(..))
            .chain(self.free_blocks.drain(..));

        for block in blocks {
            // SAFETY: device is valid, buffer and allocation are valid.
            unsafe {
                device.destroy_buffer(block.buffer, None);
            }
            allocator.free(block.allocation).ok();
        }
    }

    fn create_block(
        &self,
        device: &ash::Device,
        allocator: &mut vma::Allocator,
        size: usize,
    ) -> Result<BufferBlock, DeviceError> {
        let buffer_ci = vk::BufferCreateInfo::default()
            .size(size as vk::DeviceSize)
            .usage(self.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: device is valid, buffer_ci is well-formed.
        let buffer =
            unsafe { device.create_buffer(&buffer_ci, None) }.map_err(DeviceError::Vulkan)?;

        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = allocator
            .allocate(&vma::AllocationCreateDesc {
                name: "buffer_pool_block",
                requirements,
                location: self.memory_location,
                linear: true,
                allocation_scheme: vma::AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| DeviceError::AllocationFailed(e.to_string()))?;

        // SAFETY: buffer and allocation are valid.
        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }
            .map_err(DeviceError::Vulkan)?;

        let mapped_ptr = allocation
            .mapped_ptr()
            .ok_or_else(|| DeviceError::AllocationFailed("buffer pool block not mapped".into()))?;

        log::debug!(
            "BufferPool: allocated new {:.0}KB block (usage={:?})",
            size as f64 / 1024.0,
            self.usage
        );

        Ok(BufferBlock {
            buffer,
            allocation,
            mapped_ptr: mapped_ptr.cast(),
            offset: 0,
            capacity: size,
        })
    }
}

/// Pre-configured buffer pool sizes.
pub mod pool_sizes {
    /// Default VBO block size: 4 MB.
    pub const VBO_BLOCK_SIZE: usize = 4 * 1024 * 1024;
    /// Default IBO block size: 2 MB.
    pub const IBO_BLOCK_SIZE: usize = 2 * 1024 * 1024;
    /// Default UBO block size: 256 KB.
    pub const UBO_BLOCK_SIZE: usize = 256 * 1024;
    /// Default staging block size: 8 MB.
    pub const STAGING_BLOCK_SIZE: usize = 8 * 1024 * 1024;
}
