//! Buffer-based descriptor management via `VK_EXT_descriptor_buffer`.
//!
//! An alternative descriptor management path where descriptors are written
//! directly into GPU-visible buffers instead of using `VkDescriptorSet` objects.
//! This can reduce CPU overhead for descriptor updates in some implementations.
//!
//! # Usage
//!
//! 1. Query descriptor sizes via [`DescriptorBufferProperties`].
//! 2. Create a [`DescriptorBuffer`] with sufficient capacity.
//! 3. Write descriptors into the buffer at computed offsets.
//! 4. Bind the buffer with `vkCmdBindDescriptorBuffersEXT` and set offsets
//!    with `vkCmdSetDescriptorBufferOffsetsEXT`.
//!
//! # Extension Requirements
//!
//! Requires `VK_EXT_descriptor_buffer` to be enabled at device creation.

use ash::vk;

/// Cached descriptor buffer properties from the physical device.
///
/// These sizes are needed to compute buffer layouts and offsets for
/// descriptor buffer usage.
#[derive(Debug, Clone, Copy)]
pub struct DescriptorBufferProperties {
    /// Size of a sampler descriptor in bytes.
    pub sampler_descriptor_size: usize,
    /// Size of a combined image sampler descriptor in bytes.
    pub combined_image_sampler_descriptor_size: usize,
    /// Size of a sampled image descriptor in bytes.
    pub sampled_image_descriptor_size: usize,
    /// Size of a storage image descriptor in bytes.
    pub storage_image_descriptor_size: usize,
    /// Size of a uniform texel buffer descriptor in bytes.
    pub uniform_texel_buffer_descriptor_size: usize,
    /// Size of a storage texel buffer descriptor in bytes.
    pub storage_texel_buffer_descriptor_size: usize,
    /// Size of a uniform buffer descriptor in bytes.
    pub uniform_buffer_descriptor_size: usize,
    /// Size of a storage buffer descriptor in bytes.
    pub storage_buffer_descriptor_size: usize,
    /// Size of an acceleration structure descriptor in bytes.
    pub acceleration_structure_descriptor_size: usize,
    /// Required alignment for descriptor buffer offsets.
    pub descriptor_buffer_offset_alignment: vk::DeviceSize,
}

impl DescriptorBufferProperties {
    /// Query descriptor buffer properties from the physical device.
    ///
    /// Requires `VK_EXT_descriptor_buffer` support.
    pub fn query(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Self {
        let mut desc_buf_props = vk::PhysicalDeviceDescriptorBufferPropertiesEXT::default();
        let mut props2 =
            vk::PhysicalDeviceProperties2::default().push_next(&mut desc_buf_props);

        // SAFETY: instance and physical_device are valid.
        unsafe {
            instance.get_physical_device_properties2(physical_device, &mut props2);
        }

        Self {
            sampler_descriptor_size: desc_buf_props.sampler_descriptor_size,
            combined_image_sampler_descriptor_size: desc_buf_props
                .combined_image_sampler_descriptor_size,
            sampled_image_descriptor_size: desc_buf_props.sampled_image_descriptor_size,
            storage_image_descriptor_size: desc_buf_props.storage_image_descriptor_size,
            uniform_texel_buffer_descriptor_size: desc_buf_props
                .uniform_texel_buffer_descriptor_size,
            storage_texel_buffer_descriptor_size: desc_buf_props
                .storage_texel_buffer_descriptor_size,
            uniform_buffer_descriptor_size: desc_buf_props.uniform_buffer_descriptor_size,
            storage_buffer_descriptor_size: desc_buf_props.storage_buffer_descriptor_size,
            acceleration_structure_descriptor_size: desc_buf_props
                .acceleration_structure_descriptor_size,
            descriptor_buffer_offset_alignment: desc_buf_props.descriptor_buffer_offset_alignment,
        }
    }

    /// Get the descriptor size for a given descriptor type.
    pub fn descriptor_size(&self, ty: vk::DescriptorType) -> usize {
        match ty {
            vk::DescriptorType::SAMPLER => self.sampler_descriptor_size,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                self.combined_image_sampler_descriptor_size
            }
            vk::DescriptorType::SAMPLED_IMAGE => self.sampled_image_descriptor_size,
            vk::DescriptorType::STORAGE_IMAGE => self.storage_image_descriptor_size,
            vk::DescriptorType::UNIFORM_TEXEL_BUFFER => {
                self.uniform_texel_buffer_descriptor_size
            }
            vk::DescriptorType::STORAGE_TEXEL_BUFFER => {
                self.storage_texel_buffer_descriptor_size
            }
            vk::DescriptorType::UNIFORM_BUFFER => self.uniform_buffer_descriptor_size,
            vk::DescriptorType::STORAGE_BUFFER => self.storage_buffer_descriptor_size,
            vk::DescriptorType::ACCELERATION_STRUCTURE_KHR => {
                self.acceleration_structure_descriptor_size
            }
            _ => {
                log::warn!("Unknown descriptor type {:?}, using uniform buffer size", ty);
                self.uniform_buffer_descriptor_size
            }
        }
    }
}

/// A GPU buffer used for storing descriptors via `VK_EXT_descriptor_buffer`.
///
/// Descriptors are written directly into the buffer's mapped memory at
/// offsets computed from the set layout.
pub struct DescriptorBuffer {
    buffer: vk::Buffer,
    allocation: Option<gpu_allocator::vulkan::Allocation>,
    device_address: vk::DeviceAddress,
    mapped_ptr: *mut u8,
    size: vk::DeviceSize,
    offset: vk::DeviceSize,
    usage: DescriptorBufferUsage,
}

// SAFETY: The descriptor buffer is GPU memory with a mapped pointer.
// Access is synchronized by the frame context (one writer at a time).
unsafe impl Send for DescriptorBuffer {}
// SAFETY: Same as Send.
unsafe impl Sync for DescriptorBuffer {}

/// What kind of descriptors this buffer holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorBufferUsage {
    /// Resource descriptors (UBO, SSBO, images, etc.).
    Resource,
    /// Sampler descriptors only.
    Sampler,
    /// Both resource and sampler descriptors.
    ResourceAndSampler,
}

impl DescriptorBufferUsage {
    fn to_buffer_usage(self) -> vk::BufferUsageFlags {
        match self {
            DescriptorBufferUsage::Resource => {
                vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            }
            DescriptorBufferUsage::Sampler => {
                vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            }
            DescriptorBufferUsage::ResourceAndSampler => {
                vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
                    | vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            }
        }
    }
}

impl DescriptorBuffer {
    /// Create a new descriptor buffer.
    ///
    /// `size` is the total buffer size in bytes.
    pub fn new(
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        size: vk::DeviceSize,
        usage: DescriptorBufferUsage,
    ) -> Result<Self, vk::Result> {
        let buffer_ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage.to_buffer_usage())
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: device is valid, buffer_ci is well-formed.
        let buffer = unsafe { device.create_buffer(&buffer_ci, None)? };

        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = allocator
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "descriptor_buffer",
                requirements,
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|_| vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;

        // SAFETY: device, buffer, and allocation are valid.
        unsafe {
            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        }

        let mapped_ptr = allocation
            .mapped_ptr()
            .map(|p| p.as_ptr() as *mut u8)
            .unwrap_or(std::ptr::null_mut());

        let addr_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        // SAFETY: device and buffer are valid, buffer was created with SHADER_DEVICE_ADDRESS.
        let device_address = unsafe { device.get_buffer_device_address(&addr_info) };

        log::debug!(
            "Created descriptor buffer: {} bytes, address={:#x}",
            size,
            device_address
        );

        Ok(Self {
            buffer,
            allocation: Some(allocation),
            device_address,
            mapped_ptr,
            size,
            offset: 0,
            usage,
        })
    }

    /// The raw Vulkan buffer handle.
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    /// The buffer device address.
    pub fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }

    /// Total buffer size.
    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    /// Current write offset (bump allocator position).
    pub fn offset(&self) -> vk::DeviceSize {
        self.offset
    }

    /// The buffer usage type.
    pub fn usage(&self) -> DescriptorBufferUsage {
        self.usage
    }

    /// Reset the bump allocator for reuse (e.g., per-frame reset).
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Sub-allocate space for `count` descriptors of the given size,
    /// respecting the required alignment.
    ///
    /// Returns the byte offset within the buffer, or `None` if the buffer
    /// is full.
    pub fn allocate(
        &mut self,
        descriptor_size: usize,
        count: u32,
        alignment: vk::DeviceSize,
    ) -> Option<vk::DeviceSize> {
        let aligned_offset = (self.offset + alignment - 1) & !(alignment - 1);
        let total_size = descriptor_size as vk::DeviceSize * count as vk::DeviceSize;
        if aligned_offset + total_size > self.size {
            return None;
        }
        self.offset = aligned_offset + total_size;
        Some(aligned_offset)
    }

    /// Write raw descriptor data at the given offset.
    ///
    /// # Safety
    ///
    /// The caller must ensure `offset + data.len()` does not exceed the
    /// buffer size, and the data is a valid descriptor for the target layout.
    pub unsafe fn write(&self, offset: vk::DeviceSize, data: &[u8]) {
        debug_assert!(!self.mapped_ptr.is_null(), "descriptor buffer not mapped");
        debug_assert!(
            offset + data.len() as vk::DeviceSize <= self.size,
            "descriptor buffer write out of bounds"
        );
        // SAFETY: mapped_ptr is valid for the buffer size, caller ensures bounds.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.mapped_ptr.add(offset as usize), data.len());
        }
    }

    /// Destroy the descriptor buffer and free its memory.
    pub fn destroy(
        &mut self,
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
    ) {
        if let Some(alloc) = self.allocation.take() {
            allocator.free(alloc).ok();
        }
        if self.buffer != vk::Buffer::null() {
            // SAFETY: device is valid, buffer is valid.
            unsafe {
                device.destroy_buffer(self.buffer, None);
            }
            self.buffer = vk::Buffer::null();
        }
    }
}

/// Get the descriptor set layout size for use with descriptor buffers.
///
/// This tells you how many bytes a set layout requires in a descriptor buffer.
pub fn get_descriptor_set_layout_size(
    descriptor_buffer_loader: &ash::ext::descriptor_buffer::Device,
    layout: vk::DescriptorSetLayout,
) -> vk::DeviceSize {
    let mut size: vk::DeviceSize = 0;
    // SAFETY: loader and layout are valid, size pointer is valid.
    unsafe {
        (descriptor_buffer_loader
            .fp()
            .get_descriptor_set_layout_size_ext)(
            descriptor_buffer_loader.device(),
            layout,
            &mut size,
        );
    }
    size
}

/// Get the offset of a binding within a descriptor set layout.
///
/// Used to compute where to write a specific binding's descriptor data
/// within the buffer region for that set.
pub fn get_descriptor_set_layout_binding_offset(
    descriptor_buffer_loader: &ash::ext::descriptor_buffer::Device,
    layout: vk::DescriptorSetLayout,
    binding: u32,
) -> vk::DeviceSize {
    let mut offset: vk::DeviceSize = 0;
    // SAFETY: loader and layout are valid, offset pointer is valid.
    unsafe {
        (descriptor_buffer_loader
            .fp()
            .get_descriptor_set_layout_binding_offset_ext)(
            descriptor_buffer_loader.device(),
            layout,
            binding,
            &mut offset,
        );
    }
    offset
}
