//! Central device abstraction wrapping [`Context`] with frame management.
//!
//! The [`Device`] owns the Vulkan context and frame context manager. All resource
//! creation, command buffer requests, and submission flow through this type.

use ash::vk;
use thiserror::Error;

use crate::context::{Context, ContextConfig, ContextError, QueueType};
use crate::frame_context::{DeferredDeletion, FrameContextManager, FRAME_OVERLAP};
use crate::linear_alloc::TransientAllocation;
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

        let sampler_cache =
            SamplerCache::new(context.device()).map_err(DeviceError::Vulkan)?;

        Ok(Self {
            context,
            frame_manager,
            sampler_cache,
            semaphore_pool: SemaphorePool::new(),
            fence_pool: FencePool::new(),
        })
    }

    /// Begin a new frame. Waits on the frame fence, flushes deferred deletions,
    /// and resets the command pool.
    pub fn begin_frame(&mut self) -> Result<(), DeviceError> {
        self.frame_manager
            .begin_frame(self.context.device(), self.context.allocator())
            .map_err(DeviceError::Vulkan)
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
    pub fn get_sampler(&mut self, info: &SamplerCreateInfo) -> Result<vk::Sampler, DeviceError> {
        self.sampler_cache
            .get_or_create(self.context.device(), info)
            .map_err(DeviceError::Vulkan)
    }

    /// Get a pre-created stock sampler.
    pub fn get_stock_sampler(&self, stock: StockSampler) -> vk::Sampler {
        self.sampler_cache.get_stock(stock)
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
        let allocator = alloc_guard.as_mut().ok_or(DeviceError::AllocatorUnavailable)?;
        self.frame_manager
            .current_frame_mut()
            .linear_allocator_pool
            .allocate(&device, allocator, size, alignment)
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
