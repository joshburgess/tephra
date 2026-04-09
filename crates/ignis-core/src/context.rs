//! Vulkan instance, physical device selection, logical device creation, and queue discovery.
//!
//! The [`Context`] struct owns the Vulkan instance, device, and queues. It is the
//! entry point for all Vulkan operations.

use ash::vk;
use thiserror::Error;

/// Errors that can occur during context creation.
#[derive(Debug, Error)]
pub enum ContextError {
    /// Failed to load the Vulkan library.
    #[error("failed to load Vulkan library: {0}")]
    LoadingError(String),

    /// Failed to create the Vulkan instance.
    #[error("failed to create Vulkan instance: {0}")]
    InstanceCreation(vk::Result),

    /// No suitable physical device found.
    #[error("no suitable Vulkan physical device found")]
    NoSuitableDevice,

    /// Failed to create the logical device.
    #[error("failed to create logical device: {0}")]
    DeviceCreation(vk::Result),

    /// Failed to create the memory allocator.
    #[error("failed to create GPU allocator: {0}")]
    AllocatorCreation(String),
}

/// Configuration for creating a [`Context`].
pub struct ContextConfig {
    /// Application name reported to the Vulkan driver.
    pub app_name: String,
    /// Application version reported to the Vulkan driver.
    pub app_version: u32,
    /// Whether to enable validation layers and debug messenger.
    pub enable_validation: bool,
    /// Additional instance extensions to enable (e.g., surface extensions).
    pub required_instance_extensions: Vec<std::ffi::CString>,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            app_name: "ignis application".to_owned(),
            app_version: vk::make_api_version(0, 1, 0, 0),
            enable_validation: cfg!(debug_assertions),
            required_instance_extensions: Vec::new(),
        }
    }
}

/// Information about a device queue.
pub struct QueueInfo {
    /// The raw Vulkan queue handle.
    pub queue: vk::Queue,
    /// The queue family index this queue belongs to.
    pub family_index: u32,
}

/// Queue types supported by the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueueType {
    /// Graphics-capable queue (also supports compute and transfer).
    Graphics,
    /// Dedicated or shared compute queue.
    Compute,
    /// Dedicated or shared transfer queue.
    Transfer,
}

/// Summary of device features enabled at creation time.
pub struct DeviceFeatures {
    /// Whether timeline semaphores are supported (Vulkan 1.2+).
    pub timeline_semaphore: bool,
    /// Whether synchronization2 is supported (Vulkan 1.3+).
    pub synchronization2: bool,
    /// Whether dynamic rendering is supported (Vulkan 1.3+).
    pub dynamic_rendering: bool,
}

/// The core Vulkan context owning instance, device, queues, and allocator.
///
/// Created via [`Context::new`]. This is the foundational object from which
/// all other ignis objects are derived.
pub struct Context {
    // TODO: Phase 1, Iteration 1.1
    // - ash::Entry
    // - ash::Instance
    // - ash::Device
    // - vk::PhysicalDevice
    // - QueueInfo for graphics, compute, transfer
    // - gpu_allocator::vulkan::Allocator (behind Mutex)
    // - Debug messenger (debug builds)
    // - PhysicalDeviceProperties, DeviceFeatures
    _private: (),
}

impl Context {
    /// Create a new Vulkan context with the given configuration.
    ///
    /// This initializes the Vulkan library, creates an instance and logical device,
    /// discovers queues, and sets up the GPU memory allocator.
    pub fn new(_config: &ContextConfig) -> Result<Self, ContextError> {
        todo!("Phase 1, Iteration 1.1: implement context creation")
    }
}
