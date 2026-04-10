//! Platform abstraction for surface creation.
//!
//! The [`WSIPlatform`] trait allows different windowing backends.
//! A [`WinitPlatform`] implementation is provided for `winit` windows.

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

/// Trait for platform-specific surface creation.
pub trait WSIPlatform {
    /// Create a Vulkan surface for this platform.
    fn create_surface(
        &self,
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<vk::SurfaceKHR, SurfaceError>;

    /// Get the current window extent in pixels.
    fn get_extent(&self) -> (u32, u32);

    /// Instance extensions required by this platform (as raw C string pointers).
    fn required_instance_extensions(&self) -> &[*const std::ffi::c_char];
}

/// Errors from surface creation.
#[derive(Debug, thiserror::Error)]
pub enum SurfaceError {
    /// Failed to create the Vulkan surface.
    #[error("failed to create surface: {0}")]
    CreationFailed(String),
}

/// Winit-based platform implementation using `ash-window`.
pub struct WinitPlatform<'a> {
    window: &'a winit::window::Window,
    required_extensions: Vec<*const std::ffi::c_char>,
}

impl<'a> WinitPlatform<'a> {
    /// Create a new winit platform from a window reference.
    pub fn new(window: &'a winit::window::Window) -> Result<Self, SurfaceError> {
        let display_handle = window
            .display_handle()
            .map_err(|e| SurfaceError::CreationFailed(e.to_string()))?;

        let required_extensions =
            ash_window::enumerate_required_extensions(display_handle.as_raw())
                .map_err(|e| SurfaceError::CreationFailed(e.to_string()))?
                .to_vec();

        Ok(Self {
            window,
            required_extensions,
        })
    }
}

impl WSIPlatform for WinitPlatform<'_> {
    fn create_surface(
        &self,
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<vk::SurfaceKHR, SurfaceError> {
        let display_handle = self
            .window
            .display_handle()
            .map_err(|e| SurfaceError::CreationFailed(e.to_string()))?;
        let window_handle = self
            .window
            .window_handle()
            .map_err(|e| SurfaceError::CreationFailed(e.to_string()))?;

        // SAFETY: entry, instance are valid; display/window handles are valid for
        // the lifetime of the window.
        unsafe {
            ash_window::create_surface(
                entry,
                instance,
                display_handle.as_raw(),
                window_handle.as_raw(),
                None,
            )
        }
        .map_err(|e| SurfaceError::CreationFailed(e.to_string()))
    }

    fn get_extent(&self) -> (u32, u32) {
        let size = self.window.inner_size();
        (size.width, size.height)
    }

    fn required_instance_extensions(&self) -> &[*const std::ffi::c_char] {
        &self.required_extensions
    }
}
