//! Frame context ring buffer and deferred deletion.
//!
//! Manages a fixed number of in-flight frames, each with its own command pool,
//! fence, and deletion queue. Resources scheduled for deletion are freed when
//! the corresponding frame's fence signals.

use ash::vk;

/// Default number of frames that can be in flight simultaneously.
pub const FRAME_OVERLAP: usize = 2;

/// A single frame's resources: fence, command pool, and deletion queue.
pub(crate) struct FrameContext {
    /// Fence signaled when this frame's GPU work completes.
    pub fence: vk::Fence,
    /// Command pool for this frame's graphics command buffers.
    pub command_pool: vk::CommandPool,
    /// Deferred deletions to process when the fence signals.
    pub deletion_queue: DeletionQueue,
    /// Whether the fence has been signaled at least once.
    pub signaled: bool,
}

/// A resource scheduled for deferred deletion.
pub(crate) enum DeferredDeletion {
    /// A buffer and its memory allocation.
    Buffer(vk::Buffer, gpu_allocator::vulkan::Allocation),
    /// An image and its memory allocation.
    Image(vk::Image, gpu_allocator::vulkan::Allocation),
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
        Self { queue: Vec::new() }
    }

    /// Schedule a resource for deferred deletion.
    pub fn push(&mut self, deletion: DeferredDeletion) {
        self.queue.push(deletion);
    }

    /// Drain all pending deletions for processing.
    pub fn drain(&mut self) -> std::vec::Drain<'_, DeferredDeletion> {
        self.queue.drain(..)
    }
}

/// Manages the ring buffer of [`FrameContext`]s.
pub(crate) struct FrameContextManager {
    // TODO: Phase 1, Iteration 1.2
    // - frames: [FrameContext; FRAME_OVERLAP] or Vec<FrameContext>
    // - current_frame_index: usize
    _private: (),
}

impl FrameContextManager {
    /// Advance to the next frame, waiting on its fence and flushing deletions.
    pub fn begin_frame(&mut self) {
        todo!("Phase 1, Iteration 1.2: wait fence, flush deletions, reset command pool")
    }

    /// Finalize the current frame (expanded later for submission).
    pub fn end_frame(&mut self) {
        todo!("Phase 1, Iteration 1.2: end frame")
    }
}
