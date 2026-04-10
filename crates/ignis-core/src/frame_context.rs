//! Frame context ring buffer and deferred deletion.
//!
//! Manages a fixed number of in-flight frames, each with its own command pool(s),
//! fence, and deletion queue. Resources scheduled for deletion are freed when
//! the corresponding frame's fence signals.

use ash::vk;
use gpu_allocator::vulkan as vma;
use parking_lot::Mutex;

use crate::context::QueueType;
use crate::linear_alloc::LinearAllocatorPool;
use crate::sync::TimelineSemaphore;

/// Default number of frames that can be in flight simultaneously.
pub const FRAME_OVERLAP: usize = 2;

/// A resource scheduled for deferred deletion.
#[allow(dead_code)]
pub(crate) enum DeferredDeletion {
    /// A buffer and its memory allocation.
    Buffer(vk::Buffer, vma::Allocation),
    /// An image and its memory allocation.
    Image(vk::Image, vma::Allocation),
    /// An image view.
    ImageView(vk::ImageView),
    /// A sampler.
    Sampler(vk::Sampler),
    /// A framebuffer.
    Framebuffer(vk::Framebuffer),
    /// A pipeline.
    Pipeline(vk::Pipeline),
    /// A descriptor pool.
    DescriptorPool(vk::DescriptorPool),
}

/// Queue of resources pending deletion.
pub(crate) struct DeletionQueue {
    queue: Vec<DeferredDeletion>,
}

impl DeletionQueue {
    pub fn new() -> Self {
        Self {
            queue: Vec::with_capacity(64),
        }
    }

    /// Schedule a resource for deferred deletion.
    pub fn push(&mut self, deletion: DeferredDeletion) {
        self.queue.push(deletion);
    }

    /// Drain all pending deletions for processing.
    pub fn drain(&mut self) -> std::vec::Drain<'_, DeferredDeletion> {
        self.queue.drain(..)
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

/// A single frame's resources: fence, command pools, deletion queue, and linear allocator pool.
pub(crate) struct FrameContext {
    /// Fence signaled when this frame's GPU work completes.
    pub fence: vk::Fence,
    /// Command pool for this frame's graphics command buffers.
    pub graphics_command_pool: vk::CommandPool,
    /// Command pool for async compute (if dedicated compute queue exists).
    pub compute_command_pool: Option<vk::CommandPool>,
    /// Command pool for transfer operations (if dedicated transfer queue exists).
    pub transfer_command_pool: Option<vk::CommandPool>,
    /// Deferred deletions to process when the fence signals.
    pub deletion_queue: DeletionQueue,
    /// Bump allocator pool for transient per-frame data.
    pub linear_allocator_pool: LinearAllocatorPool,
    /// Whether the fence has been submitted and may be in a signaled state.
    pub fence_submitted: bool,
}

impl FrameContext {
    /// Create a new frame context with the given fence and command pools.
    fn new(
        fence: vk::Fence,
        graphics_command_pool: vk::CommandPool,
        compute_command_pool: Option<vk::CommandPool>,
        transfer_command_pool: Option<vk::CommandPool>,
    ) -> Self {
        Self {
            fence,
            graphics_command_pool,
            compute_command_pool,
            transfer_command_pool,
            deletion_queue: DeletionQueue::new(),
            linear_allocator_pool: LinearAllocatorPool::new(),
            fence_submitted: false,
        }
    }

    /// Get the command pool for the given queue type.
    ///
    /// Falls back to the graphics pool if no dedicated pool exists.
    pub fn command_pool(&self, queue_type: QueueType) -> vk::CommandPool {
        match queue_type {
            QueueType::Graphics => self.graphics_command_pool,
            QueueType::Compute => self
                .compute_command_pool
                .unwrap_or(self.graphics_command_pool),
            QueueType::Transfer => self
                .transfer_command_pool
                .unwrap_or(self.graphics_command_pool),
        }
    }
}

/// Manages the ring buffer of [`FrameContext`]s.
///
/// Handles frame advancement, fence synchronization, command pool resets,
/// and deferred resource deletion.
///
/// Optionally uses a timeline semaphore for frame synchronization instead of
/// per-frame fences. When enabled, submit signals timeline value N, and
/// `begin_frame` waits on value N - FRAME_OVERLAP.
pub(crate) struct FrameContextManager {
    frames: Vec<FrameContext>,
    current_index: usize,
    frame_count: u64,
    /// Optional timeline semaphore for frame pacing (replaces per-frame fences).
    timeline: Option<TimelineSemaphore>,
}

impl FrameContextManager {
    /// Create a new frame context manager.
    ///
    /// Creates `frame_overlap` frame contexts, each with its own fence and
    /// command pool(s). Dedicated compute/transfer pools are only created
    /// when their queue families differ from the graphics family.
    pub fn new(
        device: &ash::Device,
        graphics_family: u32,
        compute_family: Option<u32>,
        transfer_family: Option<u32>,
        frame_overlap: usize,
    ) -> Result<Self, vk::Result> {
        let mut frames = Vec::with_capacity(frame_overlap);

        for _ in 0..frame_overlap {
            let fence_ci = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            // SAFETY: device is valid, fence_ci is well-formed.
            let fence = unsafe { device.create_fence(&fence_ci, None)? };

            let pool_ci = vk::CommandPoolCreateInfo::default()
                .queue_family_index(graphics_family)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);
            // SAFETY: device is valid, pool_ci specifies a valid queue family.
            let graphics_pool = unsafe { device.create_command_pool(&pool_ci, None)? };

            let compute_pool = if let Some(family) = compute_family {
                let ci = vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT);
                // SAFETY: device is valid, ci specifies a valid queue family.
                Some(unsafe { device.create_command_pool(&ci, None)? })
            } else {
                None
            };

            let transfer_pool = if let Some(family) = transfer_family {
                let ci = vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT);
                // SAFETY: device is valid, ci specifies a valid queue family.
                Some(unsafe { device.create_command_pool(&ci, None)? })
            } else {
                None
            };

            frames.push(FrameContext::new(
                fence,
                graphics_pool,
                compute_pool,
                transfer_pool,
            ));
        }

        if compute_family.is_some() {
            log::info!("Created dedicated compute command pools");
        }
        if transfer_family.is_some() {
            log::info!("Created dedicated transfer command pools");
        }

        Ok(Self {
            frames,
            current_index: 0,
            frame_count: 0,
            timeline: None,
        })
    }

    /// Advance to the next frame, waiting on its fence and flushing deletions.
    ///
    /// This must be called at the start of each frame. It:
    /// 1. Advances the ring buffer index
    /// 2. Waits on the frame's fence (ensuring GPU is done with old resources)
    /// 3. Processes deferred deletions
    /// 4. Resets all command pools
    pub fn begin_frame(
        &mut self,
        device: &ash::Device,
        allocator: &Mutex<Option<vma::Allocator>>,
    ) -> Result<(), vk::Result> {
        let overlap = self.frames.len() as u64;
        self.current_index = (self.current_index + 1) % self.frames.len();
        self.frame_count += 1;

        // Synchronize with GPU: either via timeline semaphore or per-frame fence.
        if let Some(ref timeline) = self.timeline {
            // Wait for the timeline to reach the value that was signaled when this
            // frame context was last used (frame_count - FRAME_OVERLAP).
            if self.frame_count > overlap {
                let wait_value = self.frame_count - overlap;
                timeline.wait(device, wait_value, u64::MAX)?;
            }
        } else {
            let frame = &mut self.frames[self.current_index];
            // Fence-based path (default)
            if frame.fence_submitted {
                // SAFETY: fence is valid and was previously submitted.
                unsafe {
                    device.wait_for_fences(&[frame.fence], true, u64::MAX)?;
                }
                frame.fence_submitted = false;
            }
        }

        let frame = &mut self.frames[self.current_index];

        // Always reset the fence so it's unsignaled for the next queue_submit.
        // On the first use, the fence was created SIGNALED and needs resetting.
        // SAFETY: fence is valid; resetting an already-unsignaled fence is a no-op.
        unsafe {
            device.reset_fences(&[frame.fence])?;
        }

        // Flush deferred deletions now that the GPU is done
        Self::flush_deletions(device, allocator, &mut frame.deletion_queue);

        // Reset linear allocator pool — all transient allocations from this frame are recycled
        frame.linear_allocator_pool.reset();

        // Reset command pools — all command buffers from this frame are now invalid
        // SAFETY: device and command pools are valid; we've waited on the fence.
        unsafe {
            device.reset_command_pool(
                frame.graphics_command_pool,
                vk::CommandPoolResetFlags::empty(),
            )?;
            if let Some(pool) = frame.compute_command_pool {
                device.reset_command_pool(pool, vk::CommandPoolResetFlags::empty())?;
            }
            if let Some(pool) = frame.transfer_command_pool {
                device.reset_command_pool(pool, vk::CommandPoolResetFlags::empty())?;
            }
        }

        Ok(())
    }

    /// Mark the current frame's fence as submitted.
    pub fn mark_fence_submitted(&mut self) {
        self.frames[self.current_index].fence_submitted = true;
    }

    /// The current frame context.
    pub fn current_frame(&self) -> &FrameContext {
        &self.frames[self.current_index]
    }

    /// The current frame context (mutable).
    pub fn current_frame_mut(&mut self) -> &mut FrameContext {
        &mut self.frames[self.current_index]
    }

    /// The current frame index (monotonically increasing).
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Schedule a resource for deferred deletion in the current frame.
    pub fn schedule_deletion(&mut self, deletion: DeferredDeletion) {
        self.frames[self.current_index]
            .deletion_queue
            .push(deletion);
    }

    /// Enable timeline semaphore-based frame synchronization.
    ///
    /// When enabled, `begin_frame` waits on the timeline instead of per-frame
    /// fences. Submitters should signal the timeline via
    /// [`timeline_signal_value`](Self::timeline_signal_value) after queue submit.
    /// The per-frame fence is still maintained for compatibility.
    pub fn enable_timeline(&mut self, timeline: TimelineSemaphore) {
        self.timeline = Some(timeline);
        log::info!("Frame sync: timeline semaphore enabled");
    }

    /// The timeline semaphore, if enabled.
    pub fn timeline(&self) -> Option<&TimelineSemaphore> {
        self.timeline.as_ref()
    }

    /// The timeline semaphore (mutable), if enabled.
    #[allow(dead_code)]
    pub fn timeline_mut(&mut self) -> Option<&mut TimelineSemaphore> {
        self.timeline.as_mut()
    }

    /// Get the next timeline value to signal on the current frame's submit.
    ///
    /// Returns `None` if timeline sync is not enabled.
    pub fn timeline_signal_value(&mut self) -> Option<u64> {
        self.timeline.as_mut().map(|t| t.next_value())
    }

    /// Flush all deletions across all frame contexts. Called during shutdown.
    pub fn flush_all(&mut self, device: &ash::Device, allocator: &Mutex<Option<vma::Allocator>>) {
        for frame in &mut self.frames {
            Self::flush_deletions(device, allocator, &mut frame.deletion_queue);
        }
    }

    /// Destroy all owned Vulkan objects. Called during shutdown after device_wait_idle.
    pub fn destroy(&mut self, device: &ash::Device, allocator: &Mutex<Option<vma::Allocator>>) {
        if let Some(ref mut timeline) = self.timeline {
            timeline.destroy(device);
        }

        let mut alloc_guard = allocator.lock();
        for frame in &mut self.frames {
            // Destroy linear allocator pools
            if let Some(alloc) = alloc_guard.as_mut() {
                frame.linear_allocator_pool.destroy(device, alloc);
            }

            // SAFETY: device is valid, we've waited idle, these handles are valid.
            unsafe {
                device.destroy_fence(frame.fence, None);
                device.destroy_command_pool(frame.graphics_command_pool, None);
                if let Some(pool) = frame.compute_command_pool {
                    device.destroy_command_pool(pool, None);
                }
                if let Some(pool) = frame.transfer_command_pool {
                    device.destroy_command_pool(pool, None);
                }
            }
        }
        self.frames.clear();
    }

    fn flush_deletions(
        device: &ash::Device,
        allocator: &Mutex<Option<vma::Allocator>>,
        queue: &mut DeletionQueue,
    ) {
        if queue.is_empty() {
            return;
        }

        let mut alloc_guard = allocator.lock();
        let allocator = match alloc_guard.as_mut() {
            Some(a) => a,
            None => return,
        };

        for deletion in queue.drain() {
            // SAFETY: All handles are valid and the GPU is done with them
            // (we waited on the fence before calling this).
            unsafe {
                match deletion {
                    DeferredDeletion::Buffer(buffer, allocation) => {
                        device.destroy_buffer(buffer, None);
                        allocator.free(allocation).ok();
                    }
                    DeferredDeletion::Image(image, allocation) => {
                        device.destroy_image(image, None);
                        allocator.free(allocation).ok();
                    }
                    DeferredDeletion::ImageView(view) => {
                        device.destroy_image_view(view, None);
                    }
                    DeferredDeletion::Sampler(sampler) => {
                        device.destroy_sampler(sampler, None);
                    }
                    DeferredDeletion::Framebuffer(fb) => {
                        device.destroy_framebuffer(fb, None);
                    }
                    DeferredDeletion::Pipeline(pipeline) => {
                        device.destroy_pipeline(pipeline, None);
                    }
                    DeferredDeletion::DescriptorPool(pool) => {
                        device.destroy_descriptor_pool(pool, None);
                    }
                }
            }
        }
    }
}
