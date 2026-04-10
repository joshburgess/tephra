//! Central device abstraction wrapping [`Context`] with frame management.
//!
//! The [`Device`] owns the Vulkan context and frame context manager. All resource
//! creation, command buffer requests, and submission flow through this type.

use ash::vk;
use thiserror::Error;

use crate::context::{Context, ContextConfig, ContextError, QueueType};
use crate::frame_context::{DeferredDeletion, FRAME_OVERLAP, FrameContextManager};
use crate::linear_alloc::TransientAllocation;
use crate::resource_manager::{ResourceHandle, ResourceManager, ResourceState};
use crate::sampler::{SamplerCache, SamplerCreateInfo, StockSampler};
use crate::sync::{FencePool, SemaphorePool};

/// Errors that can occur during device operations.
#[derive(Debug, Error)]
pub enum DeviceError {
    /// An error from context creation.
    #[error(transparent)]
    Context(#[from] ContextError),

    /// A Vulkan API call failed.
    #[error("Vulkan error: {0}")]
    Vulkan(vk::Result),

    /// The GPU allocator is not available (device is shutting down).
    #[error("allocator unavailable")]
    AllocatorUnavailable,

    /// A GPU allocation failed.
    #[error("allocation failed: {0}")]
    AllocationFailed(String),
}

impl From<vk::Result> for DeviceError {
    fn from(result: vk::Result) -> Self {
        Self::Vulkan(result)
    }
}

/// The central device abstraction.
///
/// Wraps [`Context`] and adds frame context management, resource creation
/// convenience methods, and deferred deletion. This is the primary API surface
/// for interacting with the GPU.
pub struct Device {
    pub(crate) context: Context,
    frame_manager: FrameContextManager,
    sampler_cache: SamplerCache,
    pub(crate) semaphore_pool: SemaphorePool,
    pub(crate) fence_pool: FencePool,
    resource_manager: Option<ResourceManager>,
}

impl Device {
    /// Create a new device with default configuration.
    pub fn new(config: &ContextConfig) -> Result<Self, DeviceError> {
        let context = Context::new(config)?;
        Self::init(context)
    }

    /// Create a device from an existing context.
    pub fn from_context(context: Context) -> Result<Self, DeviceError> {
        Self::init(context)
    }

    fn init(context: Context) -> Result<Self, DeviceError> {
        let graphics_family = context.queue(QueueType::Graphics).family_index;
        let compute_family = context.queue(QueueType::Compute).family_index;
        let transfer_family = context.queue(QueueType::Transfer).family_index;

        let dedicated_compute = (compute_family != graphics_family).then_some(compute_family);
        let dedicated_transfer = (transfer_family != graphics_family).then_some(transfer_family);

        let frame_manager = FrameContextManager::new(
            context.device(),
            graphics_family,
            dedicated_compute,
            dedicated_transfer,
            FRAME_OVERLAP,
        )
        .map_err(DeviceError::Vulkan)?;

        let sampler_cache = SamplerCache::new(context.device()).map_err(DeviceError::Vulkan)?;

        Ok(Self {
            context,
            frame_manager,
            sampler_cache,
            semaphore_pool: SemaphorePool::new(),
            fence_pool: FencePool::new(),
            resource_manager: None,
        })
    }

    /// Begin a new frame. Waits on the frame fence, flushes deferred deletions,
    /// and resets the command pool.
    pub fn begin_frame(&mut self) -> Result<(), DeviceError> {
        self.frame_manager
            .begin_frame(self.context.device(), self.context.allocator())
            .map_err(DeviceError::Vulkan)?;

        // Log pending resource loads if a resource manager is attached
        if let Some(ref rm) = self.resource_manager {
            let pending = rm.pending_resources();
            if !pending.is_empty() {
                log::debug!("ResourceManager: {} resources pending load", pending.len());
            }
        }

        Ok(())
    }

    /// End the current frame.
    pub fn end_frame(&mut self) -> Result<(), DeviceError> {
        // Submission and presentation will be handled here once WSI is integrated.
        Ok(())
    }

    /// Schedule a resource for deferred deletion. The resource will be freed
    /// when the current frame's fence signals (i.e., when this frame context
    /// is recycled after `FRAME_OVERLAP` frames).
    pub(crate) fn schedule_deletion(&mut self, deletion: DeferredDeletion) {
        self.frame_manager.schedule_deletion(deletion);
    }

    /// The current frame fence, for use in queue submissions.
    pub fn current_fence(&self) -> vk::Fence {
        self.frame_manager.current_frame().fence
    }

    /// Mark the current frame's fence as submitted.
    pub fn mark_fence_submitted(&mut self) {
        self.frame_manager.mark_fence_submitted();
    }

    /// The current frame's graphics command pool.
    pub fn current_command_pool(&self) -> vk::CommandPool {
        self.frame_manager.current_frame().graphics_command_pool
    }

    /// The current frame's command pool for the given queue type.
    ///
    /// Falls back to the graphics pool if no dedicated pool exists.
    pub fn command_pool_for_queue(&self, queue_type: QueueType) -> vk::CommandPool {
        self.frame_manager.current_frame().command_pool(queue_type)
    }

    /// The monotonically increasing frame count.
    pub fn frame_count(&self) -> u64 {
        self.frame_manager.frame_count()
    }

    /// Access the underlying Vulkan context.
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// The raw Vulkan device handle.
    pub fn raw(&self) -> &ash::Device {
        self.context.device()
    }

    /// Look up or create a sampler from a create info.
    pub fn sampler(&mut self, info: &SamplerCreateInfo) -> Result<vk::Sampler, DeviceError> {
        self.sampler_cache
            .get_or_create(self.context.device(), info)
            .map_err(DeviceError::Vulkan)
    }

    /// Get a pre-created stock sampler.
    pub fn stock_sampler(&self, stock: StockSampler) -> vk::Sampler {
        self.sampler_cache.stock(stock)
    }

    /// Allocate transient data from the current frame's linear allocator pool.
    ///
    /// The returned allocation is valid until the frame context is recycled
    /// (after `FRAME_OVERLAP` frames). The buffer is host-visible and coherent,
    /// suitable for vertex, index, or uniform data.
    pub fn allocate_transient(
        &mut self,
        size: usize,
        alignment: usize,
    ) -> Result<TransientAllocation, DeviceError> {
        let device = self.context.device().clone();
        let mut alloc_guard = self.context.allocator().lock();
        let allocator = alloc_guard
            .as_mut()
            .ok_or(DeviceError::AllocatorUnavailable)?;
        self.frame_manager
            .current_frame_mut()
            .linear_allocator_pool
            .allocate(&device, allocator, size, alignment)
    }

    /// Query format properties for a given format.
    pub fn format_properties(&self, format: vk::Format) -> vk::FormatProperties {
        // SAFETY: physical device is valid.
        unsafe {
            self.context
                .instance()
                .get_physical_device_format_properties(self.context.physical_device(), format)
        }
    }

    /// Query image format properties for a given combination.
    ///
    /// Returns `None` if the format/type/tiling/usage combination is unsupported.
    pub fn image_format_properties(
        &self,
        format: vk::Format,
        image_type: vk::ImageType,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        flags: vk::ImageCreateFlags,
    ) -> Option<vk::ImageFormatProperties> {
        // SAFETY: physical device is valid.
        unsafe {
            self.context
                .instance()
                .get_physical_device_image_format_properties(
                    self.context.physical_device(),
                    format,
                    image_type,
                    tiling,
                    usage,
                    flags,
                )
        }
        .ok()
    }

    /// Check whether a format supports the given tiling and usage flags.
    pub fn is_format_supported(
        &self,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::FormatFeatureFlags,
    ) -> bool {
        let props = self.format_properties(format);
        let features = match tiling {
            vk::ImageTiling::OPTIMAL => props.optimal_tiling_features,
            vk::ImageTiling::LINEAR => props.linear_tiling_features,
            _ => vk::FormatFeatureFlags::empty(),
        };
        features.contains(usage)
    }

    /// Enable timeline semaphore-based frame synchronization.
    ///
    /// Creates a timeline semaphore and configures the frame context manager to
    /// use it for frame pacing instead of per-frame fences. Submitters should
    /// include the timeline semaphore in their signal list.
    ///
    /// The per-frame fence is still maintained for use with swapchain operations
    /// and explicit synchronization.
    pub fn enable_timeline_sync(&mut self) -> Result<(), DeviceError> {
        let timeline = crate::sync::TimelineSemaphore::new(self.context.device())
            .map_err(DeviceError::Vulkan)?;
        self.frame_manager.enable_timeline(timeline);
        Ok(())
    }

    /// The timeline semaphore's raw handle, if timeline sync is enabled.
    pub fn timeline_semaphore(&self) -> Option<vk::Semaphore> {
        self.frame_manager.timeline().map(|t| t.raw())
    }

    /// Advance and return the next timeline signal value for the current frame.
    ///
    /// Returns `None` if timeline sync is not enabled.
    pub fn timeline_signal_value(&mut self) -> Option<u64> {
        self.frame_manager.timeline_signal_value()
    }

    /// Attach a resource manager to the device.
    ///
    /// The resource manager tracks asynchronous resource loading (textures,
    /// buffers) and provides fallback views while resources are in flight.
    pub fn set_resource_manager(&mut self, rm: ResourceManager) {
        self.resource_manager = Some(rm);
    }

    /// Access the resource manager, if one is attached.
    pub fn resource_manager(&self) -> Option<&ResourceManager> {
        self.resource_manager.as_ref()
    }

    /// Mutably access the resource manager, if one is attached.
    pub fn resource_manager_mut(&mut self) -> Option<&mut ResourceManager> {
        self.resource_manager.as_mut()
    }

    /// Query the state of a managed resource.
    ///
    /// Returns `None` if no resource manager is attached.
    pub fn resource_state(&self, handle: ResourceHandle) -> Option<ResourceState> {
        self.resource_manager
            .as_ref()
            .map(|rm| rm.resource_state(handle))
    }

    /// Set a debug name on a Vulkan object.
    ///
    /// No-op if `VK_EXT_debug_utils` is not enabled.
    pub fn set_name<T: vk::Handle>(&self, handle: T, name: &str) {
        self.context.set_name(handle, name);
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: Wait for all GPU work to finish before tearing down.
        unsafe {
            self.context.device().device_wait_idle().ok();
        }

        // Flush all pending deletions across all frame contexts
        self.frame_manager
            .flush_all(self.context.device(), self.context.allocator());

        // Destroy frame context Vulkan objects (fences, command pools, linear allocators)
        self.frame_manager
            .destroy(self.context.device(), self.context.allocator());

        // Destroy cached samplers and sync pools
        self.sampler_cache.destroy(self.context.device());
        self.semaphore_pool.destroy(self.context.device());
        self.fence_pool.destroy(self.context.device());

        // Context Drop handles device, instance, allocator, debug messenger
    }
}
