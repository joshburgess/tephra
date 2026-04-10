//! Headless (offscreen) rendering without a window surface.
//!
//! Provides a [`HeadlessSwapchain`] that emulates a swapchain using plain
//! Vulkan images, allowing the same frame flow to be used for offscreen
//! rendering (e.g., server-side rendering, screenshot capture, testing).

use ash::vk;

/// A "swapchain" backed by plain images for headless rendering.
///
/// This allows using the same `begin_frame` / `end_frame` flow without
/// requiring a window surface or `VK_KHR_swapchain`.
pub struct HeadlessSwapchain {
    images: Vec<vk::Image>,
    allocations: Vec<Option<gpu_allocator::vulkan::Allocation>>,
    image_views: Vec<vk::ImageView>,
    format: vk::Format,
    extent: vk::Extent2D,
    current_index: u32,
}

impl HeadlessSwapchain {
    /// Create a new headless swapchain with the given parameters.
    ///
    /// Creates `image_count` images of the specified size and format.
    pub fn new(
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        width: u32,
        height: u32,
        format: vk::Format,
        image_count: u32,
    ) -> Result<Self, vk::Result> {
        let extent = vk::Extent2D { width, height };
        let mut images = Vec::with_capacity(image_count as usize);
        let mut allocations = Vec::with_capacity(image_count as usize);
        let mut image_views = Vec::with_capacity(image_count as usize);

        for _ in 0..image_count {
            let image_ci = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);

            // SAFETY: device is valid, image_ci is well-formed.
            let image = unsafe { device.create_image(&image_ci, None)? };

            let requirements = unsafe { device.get_image_memory_requirements(image) };

            let allocation = allocator
                .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                    name: "headless_swapchain_image",
                    requirements,
                    location: gpu_allocator::MemoryLocation::GpuOnly,
                    linear: false,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                })
                .map_err(|_| vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;

            // SAFETY: device, image, and allocation are valid.
            unsafe {
                device.bind_image_memory(image, allocation.memory(), allocation.offset())?;
            }

            let view_ci = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            // SAFETY: device and image are valid.
            let view = unsafe { device.create_image_view(&view_ci, None)? };

            images.push(image);
            allocations.push(Some(allocation));
            image_views.push(view);
        }

        log::debug!(
            "Created headless swapchain: {}x{}, {:?}, {} images",
            width,
            height,
            format,
            image_count
        );

        Ok(Self {
            images,
            allocations,
            image_views,
            format,
            extent,
            current_index: 0,
        })
    }

    /// Acquire the next image (round-robin).
    ///
    /// Returns the image index. No synchronization is needed since there's
    /// no presentation engine.
    pub fn acquire_next_image(&mut self) -> u32 {
        let index = self.current_index;
        self.current_index = (self.current_index + 1) % self.images.len() as u32;
        index
    }

    /// Get the image at the given index.
    pub fn image(&self, index: u32) -> vk::Image {
        self.images[index as usize]
    }

    /// Get the image view at the given index.
    pub fn image_view(&self, index: u32) -> vk::ImageView {
        self.image_views[index as usize]
    }

    /// The image format.
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// The image extent.
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// The number of images.
    pub fn image_count(&self) -> u32 {
        self.images.len() as u32
    }

    /// Destroy all resources.
    pub fn destroy(
        &mut self,
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
    ) {
        for view in self.image_views.drain(..) {
            // SAFETY: device is valid, view is valid, GPU is idle.
            unsafe {
                device.destroy_image_view(view, None);
            }
        }
        for (image, allocation) in self.images.drain(..).zip(self.allocations.drain(..)) {
            if let Some(alloc) = allocation {
                allocator.free(alloc).ok();
            }
            // SAFETY: device is valid, image is valid, GPU is idle.
            unsafe {
                device.destroy_image(image, None);
            }
        }
    }
}
