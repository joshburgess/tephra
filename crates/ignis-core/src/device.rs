//! Central device abstraction wrapping [`Context`] with frame management.
//!
//! The [`Device`] owns the Vulkan context and frame context manager. All resource
//! creation, command buffer requests, and submission flow through this type.

use crate::context::Context;

/// The central device abstraction.
///
/// Wraps [`Context`] and adds frame context management, resource creation
/// convenience methods, and deferred deletion.
pub struct Device {
    pub(crate) context: Context,
    // TODO: Phase 1, Iteration 1.2+
    // - FrameContextManager
    // - SamplerCache
    // - Resource creation methods
}

impl Device {
    /// Create a new device from an existing context.
    pub fn new(context: Context) -> Self {
        Self {
            context,
            // TODO: initialize frame context manager, sampler cache
        }
    }

    /// Access the underlying Vulkan context.
    pub fn context(&self) -> &Context {
        &self.context
    }
}
