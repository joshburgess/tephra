//! Command buffer wrapper with state tracking and convenience methods.

use ash::vk;

/// A recorded command buffer with state tracking.
///
/// Wraps a `vk::CommandBuffer` and provides convenience methods for
/// barriers, copies, draws, and dispatches. Tracks current pipeline state
/// for lazy pipeline compilation (Phase 4).
pub struct CommandBuffer {
    pub(crate) raw: vk::CommandBuffer,
    // TODO: Phase 2, Iteration 2.1
    // - queue_type
    // - linear allocator reference
    // - binding table (Phase 3)
    // - pipeline state (Phase 4)
}

impl CommandBuffer {
    /// The raw Vulkan command buffer handle.
    pub fn raw(&self) -> vk::CommandBuffer {
        self.raw
    }
}
