//! Resource creation and destruction methods on [`Device`].

use ash::vk;
use gpu_allocator::vulkan as vma;

use crate::buffer::{BufferCreateInfo, BufferInfo};
use crate::device::{Device, DeviceError};
use crate::frame_context::DeferredDeletion;
use crate::handles::{BufferHandle, ImageHandle, ImageViewHandle};
use crate::image::{ImageCreateInfo, ImageInfo, ImageViewCreateInfo};
use crate::memory::ImageDomain;

impl Device {
    /// Create a buffer.
    pub fn create_buffer(&mut self, info: &BufferCreateInfo) -> Result<BufferHandle, DeviceError> {
        let buffer_ci = vk::BufferCreateInfo::default()
            .size(info.size)
            .usage(info.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: device is valid, buffer_ci is well-formed.
        let raw = unsafe { self.raw().create_buffer(&buffer_ci, None)? };

        // SAFETY: buffer and device are valid.
        let mem_reqs = unsafe { self.raw().get_buffer_memory_requirements(raw) };

        let allocation = {
            let mut guard = self.context.allocator().lock();
            let allocator = guard.as_mut().ok_or(DeviceError::AllocatorUnavailable)?;
            allocator
                .allocate(&vma::AllocationCreateDesc {
                    name: "buffer",
                    requirements: mem_reqs,
                    location: info.domain.to_gpu_allocator(),
                    linear: true,
                    allocation_scheme: vma::AllocationScheme::GpuAllocatorManaged,
                })
                .map_err(|e| DeviceError::AllocationFailed(e.to_string()))?
        };

        // SAFETY: buffer, memory, and offset are valid and compatible.
        unsafe {
            self.raw()
                .bind_buffer_memory(raw, allocation.memory(), allocation.offset())?;
        }

        Ok(BufferHandle {
            raw,
            allocation: Some(allocation),
            info: BufferInfo {
                size: info.size,
                usage: info.usage,
                domain: info.domain,
            },
        })
    }

    /// Create a buffer and immediately fill it with data.
    ///
    /// For device-local buffers, this creates a staging buffer, copies the data,
    /// and schedules the staging buffer for deferred deletion.
    /// For host-visible buffers, the data is written directly.
    pub fn create_buffer_with_data(
        &mut self,
        info: &BufferCreateInfo,
        data: &[u8],
    ) -> Result<BufferHandle, DeviceError> {
        use crate::memory::MemoryDomain;

        assert!(
            data.len() as u64 <= info.size,
            "data ({} bytes) exceeds buffer size ({} bytes)",
            data.len(),
            info.size,
        );

        match info.domain {
            MemoryDomain::Host | MemoryDomain::CachedHost => {
                let mut buffer = self.create_buffer(info)?;
                if let Some(slice) = buffer.mapped_slice_mut() {
                    slice[..data.len()].copy_from_slice(data);
                }
                Ok(buffer)
            }
            MemoryDomain::Device => {
                // Create the device-local target buffer
                let buffer = self.create_buffer(info)?;

                // Create a staging buffer
                let staging_info = BufferCreateInfo::staging(info.size);
                let mut staging = self.create_buffer(&staging_info)?;

                // Copy data to staging
                if let Some(slice) = staging.mapped_slice_mut() {
                    slice[..data.len()].copy_from_slice(data);
                }

                // Record and submit a copy command
                self.copy_buffer_immediate(&staging, &buffer, info.size)?;

                // Schedule staging buffer for deferred deletion
                self.destroy_buffer(staging);

                Ok(buffer)
            }
        }
    }

    /// Destroy a buffer (deferred until the current frame's fence signals).
    pub fn destroy_buffer(&mut self, buffer: BufferHandle) {
        let BufferHandle {
            raw, allocation, ..
        } = buffer;
        if let Some(alloc) = allocation {
            self.schedule_deletion(DeferredDeletion::Buffer(raw, alloc));
        }
    }

    /// Create an image.
    pub fn create_image(&mut self, info: &ImageCreateInfo) -> Result<ImageHandle, DeviceError> {
        let mut usage = info.usage;
        if info.domain == ImageDomain::Transient {
            usage |= vk::ImageUsageFlags::TRANSIENT_ATTACHMENT;
        }

        let image_ci = vk::ImageCreateInfo::default()
            .image_type(info.image_type)
            .format(info.format)
            .extent(vk::Extent3D {
                width: info.width,
                height: info.height,
                depth: info.depth,
            })
            .mip_levels(info.mip_levels)
            .array_layers(info.array_layers)
            .samples(info.samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(info.initial_layout);

        // SAFETY: device is valid, image_ci is well-formed.
        let raw = unsafe { self.raw().create_image(&image_ci, None)? };

        // SAFETY: image and device are valid.
        let mem_reqs = unsafe { self.raw().get_image_memory_requirements(raw) };

        let location = match info.domain {
            ImageDomain::Physical => gpu_allocator::MemoryLocation::GpuOnly,
            ImageDomain::Transient => gpu_allocator::MemoryLocation::GpuOnly,
        };

        let allocation = {
            let mut guard = self.context.allocator().lock();
            let allocator = guard.as_mut().ok_or(DeviceError::AllocatorUnavailable)?;
            allocator
                .allocate(&vma::AllocationCreateDesc {
                    name: "image",
                    requirements: mem_reqs,
                    location,
                    linear: false,
                    allocation_scheme: vma::AllocationScheme::GpuAllocatorManaged,
                })
                .map_err(|e| DeviceError::AllocationFailed(e.to_string()))?
        };

        // SAFETY: image, memory, and offset are valid and compatible.
        unsafe {
            self.raw()
                .bind_image_memory(raw, allocation.memory(), allocation.offset())?;
        }

        let image_info = ImageInfo {
            width: info.width,
            height: info.height,
            depth: info.depth,
            format: info.format,
            mip_levels: info.mip_levels,
            array_layers: info.array_layers,
            samples: info.samples,
            image_type: info.image_type,
        };

        // Create a default image view
        let aspect_mask = format_to_aspect_mask(info.format);
        let view_type = match info.image_type {
            vk::ImageType::TYPE_1D => {
                if info.array_layers > 1 {
                    vk::ImageViewType::TYPE_1D_ARRAY
                } else {
                    vk::ImageViewType::TYPE_1D
                }
            }
            vk::ImageType::TYPE_2D => {
                if info.array_layers > 1 {
                    vk::ImageViewType::TYPE_2D_ARRAY
                } else {
                    vk::ImageViewType::TYPE_2D
                }
            }
            vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
            _ => vk::ImageViewType::TYPE_2D,
        };

        let view_ci = vk::ImageViewCreateInfo::default()
            .image(raw)
            .view_type(view_type)
            .format(info.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: info.mip_levels,
                base_array_layer: 0,
                layer_count: info.array_layers,
            });

        // SAFETY: device and image are valid, view_ci is well-formed.
        let default_view = unsafe { self.raw().create_image_view(&view_ci, None)? };

        Ok(ImageHandle {
            raw,
            allocation: Some(allocation),
            info: image_info,
            default_view,
        })
    }

    /// Create an additional image view for an existing image.
    pub fn create_image_view(
        &self,
        image: &ImageHandle,
        info: &ImageViewCreateInfo,
    ) -> Result<ImageViewHandle, DeviceError> {
        let view_ci = vk::ImageViewCreateInfo::default()
            .image(image.raw)
            .view_type(info.view_type)
            .format(info.format)
            .subresource_range(info.subresource_range);

        // SAFETY: device and image are valid, view_ci is well-formed.
        let raw = unsafe { self.raw().create_image_view(&view_ci, None)? };

        Ok(ImageViewHandle { raw })
    }

    /// Destroy an image (deferred).
    pub fn destroy_image(&mut self, image: ImageHandle) {
        let ImageHandle {
            raw,
            allocation,
            default_view,
            ..
        } = image;

        self.schedule_deletion(DeferredDeletion::ImageView(default_view));
        if let Some(alloc) = allocation {
            self.schedule_deletion(DeferredDeletion::Image(raw, alloc));
        }
    }

    /// Destroy an image view (deferred).
    pub fn destroy_image_view(&mut self, view: ImageViewHandle) {
        self.schedule_deletion(DeferredDeletion::ImageView(view.raw));
    }

    /// Record and submit a buffer-to-buffer copy on the graphics queue.
    fn copy_buffer_immediate(
        &self,
        src: &BufferHandle,
        dst: &BufferHandle,
        size: vk::DeviceSize,
    ) -> Result<(), DeviceError> {
        let pool = self.current_command_pool();

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        // SAFETY: device and pool are valid.
        let cmd = unsafe { self.raw().allocate_command_buffers(&alloc_info)? }[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        // SAFETY: cmd is valid and freshly allocated.
        unsafe {
            self.raw().begin_command_buffer(cmd, &begin_info)?;
            self.raw().cmd_copy_buffer(
                cmd,
                src.raw,
                dst.raw,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size,
                }],
            );
            self.raw().end_command_buffer(cmd)?;
        }

        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));

        let queue = self.context.queue(crate::context::QueueType::Graphics).queue;

        // SAFETY: queue, cmd, and fence are valid.
        unsafe {
            self.raw()
                .queue_submit(queue, &[submit_info], vk::Fence::null())?;
            self.raw().queue_wait_idle(queue)?;
        }

        Ok(())
    }
}

/// Determine the aspect mask from an image format.
fn format_to_aspect_mask(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
            vk::ImageAspectFlags::DEPTH
        }
        vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        _ => vk::ImageAspectFlags::COLOR,
    }
}
