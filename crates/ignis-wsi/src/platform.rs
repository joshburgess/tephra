//! Platform abstraction for surface creation.
//!
//! The [`WSIPlatform`] trait allows different windowing backends.
//! A winit implementation is provided.

use ash::vk;
use std::ffi::CStr;

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

    /// Instance extensions required by this platform.
    fn required_extensions(&self) -> Vec<&'static CStr>;
}

/// Errors from surface creation.
#[derive(Debug, thiserror::Error)]
pub enum SurfaceError {
    /// Failed to create the Vulkan surface.
    #[error("failed to create surface: {0}")]
    CreationFailed(String),
}
