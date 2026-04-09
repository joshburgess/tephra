//! WSI manager: frame acquisition, presentation, and swapchain lifecycle.

use thiserror::Error;

/// Errors from WSI operations.
#[derive(Debug, Error)]
pub enum WSIError {
    /// The swapchain is out of date and must be recreated.
    #[error("swapchain out of date")]
    OutOfDate,
    /// Failed to acquire a swapchain image.
    #[error("failed to acquire swapchain image: {0}")]
    AcquireFailed(String),
    /// Failed to present.
    #[error("presentation failed: {0}")]
    PresentFailed(String),
}

/// The WSI manager, owning a device and swapchain.
pub struct WSI {
    // TODO: Phase 5, Iteration 5.1
    // - device: Device
    // - swapchain: Swapchain
    // - acquire/release semaphores
    // - current_image_index
    _private: (),
}
