//! WSI manager: frame acquisition, presentation, and swapchain lifecycle.
//!
//! The [`WSI`] struct is the top-level object for applications that render to
//! a window. It owns a [`Device`] and [`Swapchain`], managing the frame
//! acquire/present flow.

use ash::vk;
use thiserror::Error;

use tephra_core::context::{ContextConfig, ContextError, QueueType};
use tephra_core::device::{Device, DeviceError};

use crate::platform::{SurfaceError, WSIPlatform};
use crate::swapchain::{Swapchain, SwapchainImage};

/// Errors from WSI operations.
#[derive(Debug, Error)]
pub enum WSIError {
    /// An error from context/device creation.
    #[error(transparent)]
    Context(#[from] ContextError),

    /// An error from device operations.
    #[error(transparent)]
    Device(#[from] DeviceError),

    /// A Vulkan API call failed.
    #[error("Vulkan error: {0}")]
    Vulkan(vk::Result),

    /// Failed to create the surface.
    #[error(transparent)]
    Surface(#[from] SurfaceError),

    /// The swapchain is out of date and needs recreation.
    #[error("swapchain out of date")]
    OutOfDate,

    /// The surface has zero extent (window minimized).
    #[error("surface has zero extent")]
    ZeroExtent,
}

impl From<vk::Result> for WSIError {
    fn from(result: vk::Result) -> Self {
        Self::Vulkan(result)
    }
}

/// Configuration for WSI creation.
pub struct WSIConfig {
    /// Application name.
    pub app_name: std::ffi::CString,
    /// Application version.
    pub app_version: u32,
    /// Enable Vulkan validation layers.
    pub enable_validation: bool,
}

impl Default for WSIConfig {
    fn default() -> Self {
        Self {
            app_name: std::ffi::CString::new("tephra application").unwrap(),
            app_version: vk::make_api_version(0, 1, 0, 0),
            enable_validation: cfg!(debug_assertions),
        }
    }
}

/// The WSI manager, owning a device, surface, and swapchain.
///
/// This is the top-level entry point for windowed applications. It handles
/// the full begin_frame/end_frame lifecycle including swapchain image
/// acquisition and presentation.
pub struct WSI {
    device: Device,
    surface: vk::SurfaceKHR,
    surface_loader: ash::khr::surface::Instance,
    swapchain: Swapchain,
    acquire_semaphores: Vec<vk::Semaphore>,
    release_semaphores: Vec<vk::Semaphore>,
    current_image_index: u32,
    current_semaphore_index: usize,
    swapchain_suboptimal: bool,
    width: u32,
    height: u32,
}

impl WSI {
    /// Create a new WSI manager from a platform and configuration.
    ///
    /// This creates the Vulkan context, device, surface, and swapchain.
    pub fn new(platform: &dyn WSIPlatform, config: &WSIConfig) -> Result<Self, WSIError> {
        let (width, height) = platform.get_extent();

        // Build context config with platform's required extensions
        let mut required_extensions: Vec<*const std::ffi::c_char> =
            platform.required_instance_extensions().to_vec();

        // VK_KHR_surface is always needed
        required_extensions.push(ash::khr::surface::NAME.as_ptr());

        // Deduplicate
        required_extensions.sort();
        required_extensions.dedup();

        let context_config = ContextConfig {
            app_name: config.app_name.clone(),
            app_version: config.app_version,
            enable_validation: config.enable_validation,
            required_instance_extensions: required_extensions,
        };

        // Create device
        let device = Device::new(&context_config)?;

        // Create surface
        let surface =
            platform.create_surface(device.context().entry(), device.context().instance())?;

        let surface_loader =
            ash::khr::surface::Instance::new(device.context().entry(), device.context().instance());

        // Verify the graphics queue supports presentation to this surface
        let graphics_family = device.context().queue(QueueType::Graphics).family_index;
        // SAFETY: physical_device, surface are valid.
        let supports_present = unsafe {
            surface_loader.get_physical_device_surface_support(
                device.context().physical_device(),
                graphics_family,
                surface,
            )
        }?;

        if !supports_present {
            log::error!(
                "Graphics queue family {} does not support presentation to the surface",
                graphics_family
            );
            // Clean up surface before returning error
            // SAFETY: surface is valid.
            unsafe {
                surface_loader.destroy_surface(surface, None);
            }
            return Err(WSIError::Surface(SurfaceError::CreationFailed(
                "graphics queue does not support presentation".into(),
            )));
        }

        // Create swapchain
        let swapchain = Swapchain::new(
            device.context().instance(),
            device.raw(),
            device.context().physical_device(),
            surface,
            &surface_loader,
            width,
            height,
            vk::SwapchainKHR::null(),
        )?;

        // Create per-swapchain-image acquire/release semaphores.
        // We need at least as many as swapchain images to avoid signaling a
        // semaphore that the presentation engine is still using.
        let sem_count = swapchain.image_count();
        let semaphore_ci = vk::SemaphoreCreateInfo::default();
        let mut acquire_semaphores = Vec::with_capacity(sem_count);
        let mut release_semaphores = Vec::with_capacity(sem_count);

        for _ in 0..sem_count {
            // SAFETY: device is valid, semaphore_ci is well-formed.
            let acquire = unsafe { device.raw().create_semaphore(&semaphore_ci, None)? };
            let release = unsafe { device.raw().create_semaphore(&semaphore_ci, None)? };
            acquire_semaphores.push(acquire);
            release_semaphores.push(release);
        }

        log::info!(
            "WSI initialized: {}x{}, format={:?}",
            width,
            height,
            swapchain.format()
        );

        Ok(Self {
            device,
            surface,
            surface_loader,
            swapchain,
            acquire_semaphores,
            release_semaphores,
            current_image_index: 0,
            current_semaphore_index: 0,
            swapchain_suboptimal: false,
            width,
            height,
        })
    }

    /// Begin a new frame: advance frame context and acquire a swapchain image.
    ///
    /// Returns the acquired swapchain image info for rendering.
    pub fn begin_frame(&mut self) -> Result<SwapchainImage, WSIError> {
        self.device.begin_frame()?;

        let acquire_sem = self.acquire_semaphores[self.current_semaphore_index];

        let (image_index, suboptimal) =
            match self.swapchain.acquire_next_image(acquire_sem, u64::MAX) {
                Ok((idx, suboptimal)) => (idx, suboptimal),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain()?;
                    let (idx, suboptimal) =
                        self.swapchain.acquire_next_image(acquire_sem, u64::MAX)?;
                    (idx, suboptimal)
                }
                Err(e) => return Err(WSIError::Vulkan(e)),
            };

        self.swapchain_suboptimal = suboptimal;
        self.current_image_index = image_index;

        Ok(self.swapchain.image(image_index))
    }

    /// End the current frame: submit the command buffer and present.
    ///
    /// The `cmd` must have been recorded during this frame. It will be submitted
    /// to the graphics queue with the acquire semaphore as a wait dependency and
    /// the release semaphore signaled for presentation.
    pub fn end_frame(&mut self, cmd: vk::CommandBuffer) -> Result<(), WSIError> {
        let acquire_sem = self.acquire_semaphores[self.current_semaphore_index];
        let release_sem = self.release_semaphores[self.current_semaphore_index];

        // Submit with acquire wait + release signal + frame fence
        self.device.submit_command_buffer_for_frame(
            cmd,
            &[acquire_sem],
            &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
            &[release_sem],
        )?;

        // Present
        let queue = self.device.context().queue(QueueType::Graphics).queue;
        let suboptimal = match self
            .swapchain
            .present(queue, self.current_image_index, release_sem)
        {
            Ok(suboptimal) => suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.swapchain_suboptimal = true;
                true
            }
            Err(e) => return Err(WSIError::Vulkan(e)),
        };

        if suboptimal || self.swapchain_suboptimal {
            self.recreate_swapchain()?;
        }

        self.current_semaphore_index =
            (self.current_semaphore_index + 1) % self.acquire_semaphores.len();

        Ok(())
    }

    /// Notify the WSI that the window has been resized.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.swapchain_suboptimal = true;
    }

    /// Access the underlying device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Mutable access to the underlying device.
    pub fn device_mut(&mut self) -> &mut Device {
        &mut self.device
    }

    /// The current swapchain format.
    pub fn swapchain_format(&self) -> vk::Format {
        self.swapchain.format()
    }

    /// The current swapchain extent.
    pub fn swapchain_extent(&self) -> vk::Extent2D {
        self.swapchain.extent()
    }

    fn recreate_swapchain(&mut self) -> Result<(), WSIError> {
        if self.width == 0 || self.height == 0 {
            return Err(WSIError::ZeroExtent);
        }

        // Wait for all GPU work to finish before recreating
        // SAFETY: device is valid.
        unsafe {
            self.device.raw().device_wait_idle()?;
        }

        self.swapchain.recreate(
            self.device.raw(),
            self.device.context().physical_device(),
            self.surface,
            &self.surface_loader,
            self.width,
            self.height,
        )?;

        // Recreate semaphores if image count changed
        let new_count = self.swapchain.image_count();
        if new_count != self.acquire_semaphores.len() {
            // Destroy old semaphores (GPU is idle from device_wait_idle above)
            for sem in &self.acquire_semaphores {
                // SAFETY: device is valid, semaphore is valid, GPU is idle.
                unsafe {
                    self.device.raw().destroy_semaphore(*sem, None);
                }
            }
            for sem in &self.release_semaphores {
                // SAFETY: device is valid, semaphore is valid, GPU is idle.
                unsafe {
                    self.device.raw().destroy_semaphore(*sem, None);
                }
            }

            let semaphore_ci = vk::SemaphoreCreateInfo::default();
            self.acquire_semaphores.clear();
            self.release_semaphores.clear();
            for _ in 0..new_count {
                // SAFETY: device is valid.
                let acquire = unsafe { self.device.raw().create_semaphore(&semaphore_ci, None)? };
                let release = unsafe { self.device.raw().create_semaphore(&semaphore_ci, None)? };
                self.acquire_semaphores.push(acquire);
                self.release_semaphores.push(release);
            }
        }

        self.current_semaphore_index = 0;
        self.swapchain_suboptimal = false;

        log::debug!("Swapchain recreated: {}x{}", self.width, self.height);

        Ok(())
    }
}

impl Drop for WSI {
    fn drop(&mut self) {
        // SAFETY: Wait for GPU idle before tearing down.
        unsafe {
            self.device.raw().device_wait_idle().ok();
        }

        // Destroy swapchain
        self.swapchain.destroy(self.device.raw());

        // Destroy semaphores
        for sem in &self.acquire_semaphores {
            // SAFETY: device is valid, semaphore is valid.
            unsafe {
                self.device.raw().destroy_semaphore(*sem, None);
            }
        }
        for sem in &self.release_semaphores {
            // SAFETY: device is valid, semaphore is valid.
            unsafe {
                self.device.raw().destroy_semaphore(*sem, None);
            }
        }

        // Destroy surface
        // SAFETY: surface is valid, instance is valid.
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
        }

        // Device is dropped automatically via its own Drop impl
    }
}
