//! Frame context ring buffer and deferred deletion.
//!
//! Manages a fixed number of in-flight frames, each with its own command pool,
//! fence, and deletion queue. Resources scheduled for deletion are freed when
//! the corresponding frame's fence signals.

use ash::vk;
use gpu_allocator::vulkan as vma;
use parking_lot::Mutex;

use crate::linear_alloc::LinearAllocatorPool;

/// Default number of frames that can be in flight simultaneously.
pub const FRAME_OVERLAP: usize = 2;

/// A resource scheduled for deferred deletion.
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

/// A single frame's resources: fence, command pool, deletion queue, and linear allocator pool.
pub(crate) struct FrameContext {
    /// Fence signaled when this frame's GPU work completes.
    pub fence: vk::Fence,
    /// Command pool for this frame's graphics command buffers.
    pub graphics_command_pool: vk::CommandPool,
    /// Deferred deletions to process when the fence signals.
    pub deletion_queue: DeletionQueue,
    /// Bump allocator pool for transient per-frame data.
    pub linear_allocator_pool: LinearAllocatorPool,
    /// Whether the fence has been submitted and may be in a signaled state.
    pub fence_submitted: bool,
}

impl FrameContext {
    /// Create a new frame context with the given fence and command pool.
    fn new(fence: vk::Fence, command_pool: vk::CommandPool) -> Self {
        Self {
            fence,
            graphics_command_pool: command_pool,
            deletion_queue: DeletionQueue::new(),
            linear_allocator_pool: LinearAllocatorPool::new(),
            fence_submitted: false,
        }
    }
}

/// Manages the ring buffer of [`FrameContext`]s.
///
/// Handles frame advancement, fence synchronization, command pool resets,
/// and deferred resource deletion.
pub(crate) struct FrameContextManager {
    frames: Vec<FrameContext>,
    current_index: usize,
    frame_count: u64,
}

impl FrameContextManager {
    /// Create a new frame context manager.
    ///
    /// Creates `frame_overlap` frame contexts, each with its own fence and
    /// command pool on the graphics queue family.
    pub fn new(
        device: &ash::Device,
        graphics_family_index: u32,
        frame_overlap: usize,
    ) -> Result<Self, vk::Result> {
        let mut frames = Vec::with_capacity(frame_overlap);

        for _ in 0..frame_overlap {
            let fence_ci = vk::FenceCreateInfo::default()
                .flags(vk::FenceCreateFlags::SIGNALED);
            // SAFETY: device is valid, fence_ci is well-formed.
            let fence = unsafe { device.create_fence(&fence_ci, None)? };

            let pool_ci = vk::CommandPoolCreateInfo::default()
                .queue_family_index(graphics_family_index)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);
            // SAFETY: device is valid, pool_ci specifies a valid queue family.
            let command_pool = unsafe { device.create_command_pool(&pool_ci, None)? };

            frames.push(FrameContext::new(fence, command_pool));
        }

        Ok(Self {
            frames,
            current_index: 0,
            frame_count: 0,
        })
    }

    /// Advance to the next frame, waiting on its fence and flushing deletions.
    ///
    /// This must be called at the start of each frame. It:
    /// 1. Advances the ring buffer index
    /// 2. Waits on the frame's fence (ensuring GPU is done with old resources)
    /// 3. Processes deferred deletions
    /// 4. Resets the command pool
    pub fn begin_frame(
        &mut self,
        device: &ash::Device,
        allocator: &Mutex<Option<vma::Allocator>>,
    ) -> Result<(), vk::Result> {
        self.current_index = (self.current_index + 1) % self.frames.len();
        self.frame_count += 1;

        let frame = &mut self.frames[self.current_index];

        // Wait for the fence from when this frame context was last used
        if frame.fence_submitted {
            // SAFETY: fence is valid and was previously submitted.
            unsafe {
                device.wait_for_fences(&[frame.fence], true, u64::MAX)?;
                device.reset_fences(&[frame.fence])?;
            }
            frame.fence_submitted = false;
        }

        // Flush deferred deletions now that the GPU is done
        Self::flush_deletions(device, allocator, &mut frame.deletion_queue);

        // Reset linear allocator pool — all transient allocations from this frame are recycled
        frame.linear_allocator_pool.reset();

        // Reset command pool — all command buffers from this frame are now invalid
        // SAFETY: device and command pool are valid; we've waited on the fence.
        unsafe {
            device.reset_command_pool(
                frame.graphics_command_pool,
                vk::CommandPoolResetFlags::empty(),
            )?;
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
        self.frames[self.current_index].deletion_queue.push(deletion);
    }

    /// Flush all deletions across all frame contexts. Called during shutdown.
    pub fn flush_all(
        &mut self,
        device: &ash::Device,
        allocator: &Mutex<Option<vma::Allocator>>,
    ) {
        for frame in &mut self.frames {
            Self::flush_deletions(device, allocator, &mut frame.deletion_queue);
        }
    }

    /// Destroy all owned Vulkan objects. Called during shutdown after device_wait_idle.
    pub fn destroy(&mut self, device: &ash::Device, allocator: &Mutex<Option<vma::Allocator>>) {
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
