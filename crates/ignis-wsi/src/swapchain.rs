//! Swapchain creation, recreation, and image management.
//!
//! Manages a `VkSwapchainKHR` and its associated images and image views.
//! Handles format selection (prefer sRGB), present mode selection (prefer
//! mailbox), and recreation on window resize.

use ash::vk;

/// Information about a swapchain image for rendering.
pub struct SwapchainImage {
    /// The swapchain image handle.
    pub image: vk::Image,
    /// The image view for the swapchain image.
    pub view: vk::ImageView,
    /// The image index within the swapchain.
    pub index: u32,
}

/// Manages a `VkSwapchainKHR` and its images/views.
pub struct Swapchain {
    handle: vk::SwapchainKHR,
    loader: ash::khr::swapchain::Device,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    present_mode: vk::PresentModeKHR,
}

impl Swapchain {
    /// Create a new swapchain.
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        surface_loader: &ash::khr::surface::Instance,
        width: u32,
        height: u32,
        old_swapchain: vk::SwapchainKHR,
    ) -> Result<Self, vk::Result> {
        let loader = ash::khr::swapchain::Device::new(instance, device);

        // Query surface capabilities
        // SAFETY: physical_device and surface are valid.
        let capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
        };

        // Choose format (prefer B8G8R8A8_SRGB)
        // SAFETY: physical_device and surface are valid.
        let formats = unsafe {
            surface_loader.get_physical_device_surface_formats(physical_device, surface)?
        };
        let format = Self::choose_format(&formats);

        // Choose present mode (prefer Mailbox > Fifo)
        // SAFETY: physical_device and surface are valid.
        let present_modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?
        };
        let present_mode = Self::choose_present_mode(&present_modes);

        // Determine extent
        let extent = Self::choose_extent(&capabilities, width, height);

        // Image count: prefer min+1, capped by max (0 = unlimited)
        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count > 0 && image_count > capabilities.max_image_count {
            image_count = capabilities.max_image_count;
        }

        let swapchain_ci = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);

        // SAFETY: device and surface are valid, swapchain_ci is well-formed.
        let handle = unsafe { loader.create_swapchain(&swapchain_ci, None)? };

        // Get swapchain images
        // SAFETY: device and swapchain are valid.
        let images = unsafe { loader.get_swapchain_images(handle)? };

        // Create image views
        let image_views = Self::create_image_views(device, &images, format.format)?;

        log::debug!(
            "Created swapchain: {}x{}, {:?}, {:?}, {} images",
            extent.width,
            extent.height,
            format.format,
            present_mode,
            images.len()
        );

        Ok(Self {
            handle,
            loader,
            images,
            image_views,
            format,
            extent,
            present_mode,
        })
    }

    /// Recreate the swapchain (e.g., after window resize).
    pub fn recreate(
        &mut self,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        surface_loader: &ash::khr::surface::Instance,
        width: u32,
        height: u32,
    ) -> Result<(), vk::Result> {
        // Destroy old image views
        self.destroy_image_views(device);

        let old_swapchain = self.handle;

        // Query updated capabilities
        // SAFETY: physical_device and surface are valid.
        let capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
        };

        let extent = Self::choose_extent(&capabilities, width, height);

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count > 0 && image_count > capabilities.max_image_count {
            image_count = capabilities.max_image_count;
        }

        let swapchain_ci = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(self.format.format)
            .image_color_space(self.format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(self.present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);

        // SAFETY: device and surface are valid.
        self.handle = unsafe { self.loader.create_swapchain(&swapchain_ci, None)? };

        // Destroy old swapchain
        // SAFETY: old swapchain is no longer in use.
        unsafe {
            self.loader.destroy_swapchain(old_swapchain, None);
        }

        // SAFETY: device and new swapchain are valid.
        self.images = unsafe { self.loader.get_swapchain_images(self.handle)? };
        self.image_views = Self::create_image_views(device, &self.images, self.format.format)?;
        self.extent = extent;

        log::debug!(
            "Recreated swapchain: {}x{}, {} images",
            extent.width,
            extent.height,
            self.images.len()
        );

        Ok(())
    }

    /// Acquire the next swapchain image.
    ///
    /// Returns the image index or an error indicating the swapchain is out of date.
    pub fn acquire_next_image(
        &self,
        semaphore: vk::Semaphore,
        timeout: u64,
    ) -> Result<(u32, bool), vk::Result> {
        // SAFETY: swapchain and semaphore are valid.
        unsafe {
            self.loader
                .acquire_next_image(self.handle, timeout, semaphore, vk::Fence::null())
        }
    }

    /// Present the given image index.
    pub fn present(
        &self,
        queue: vk::Queue,
        image_index: u32,
        wait_semaphore: vk::Semaphore,
    ) -> Result<bool, vk::Result> {
        let swapchains = [self.handle];
        let image_indices = [image_index];
        let wait_semaphores = [wait_semaphore];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        // SAFETY: queue, swapchain, and semaphore are valid.
        unsafe { self.loader.queue_present(queue, &present_info) }
    }

    /// The swapchain image format.
    pub fn format(&self) -> vk::Format {
        self.format.format
    }

    /// The swapchain extent.
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// The number of swapchain images.
    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    /// Get a swapchain image by index.
    pub fn image(&self, index: u32) -> SwapchainImage {
        let i = index as usize;
        SwapchainImage {
            image: self.images[i],
            view: self.image_views[i],
            index,
        }
    }

    /// Destroy the swapchain and its image views.
    pub fn destroy(&mut self, device: &ash::Device) {
        self.destroy_image_views(device);
        if self.handle != vk::SwapchainKHR::null() {
            // SAFETY: device is valid, swapchain is valid, GPU is idle.
            unsafe {
                self.loader.destroy_swapchain(self.handle, None);
            }
            self.handle = vk::SwapchainKHR::null();
        }
    }

    fn destroy_image_views(&mut self, device: &ash::Device) {
        for view in self.image_views.drain(..) {
            // SAFETY: device is valid, view is valid, GPU is idle.
            unsafe {
                device.destroy_image_view(view, None);
            }
        }
    }

    fn choose_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
        // Prefer B8G8R8A8_SRGB with SRGB_NONLINEAR color space
        for f in formats {
            if f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                return *f;
            }
        }
        // Fallback: prefer any SRGB format
        for f in formats {
            if f.format == vk::Format::R8G8B8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                return *f;
            }
        }
        // Last resort: use the first available format
        formats[0]
    }

    fn choose_present_mode(modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
        // Prefer mailbox (triple buffering) for low-latency without tearing
        if modes.contains(&vk::PresentModeKHR::MAILBOX) {
            return vk::PresentModeKHR::MAILBOX;
        }
        // FIFO is guaranteed to be available
        vk::PresentModeKHR::FIFO
    }

    fn choose_extent(
        capabilities: &vk::SurfaceCapabilitiesKHR,
        width: u32,
        height: u32,
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            // Driver specifies the exact extent
            capabilities.current_extent
        } else {
            // We choose within the allowed range
            vk::Extent2D {
                width: width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        }
    }

    fn create_image_views(
        device: &ash::Device,
        images: &[vk::Image],
        format: vk::Format,
    ) -> Result<Vec<vk::ImageView>, vk::Result> {
        images
            .iter()
            .map(|&image| {
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
                unsafe { device.create_image_view(&view_ci, None) }
            })
            .collect()
    }
}
