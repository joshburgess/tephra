//! Pass declarations and render callbacks.

use crate::resource::ResourceHandle;

/// Builder for declaring a render pass and its resource dependencies.
pub struct PassBuilder<'a> {
    // TODO: Phase 6, Iteration 6.1
    _lifetime: std::marker::PhantomData<&'a ()>,
}

impl PassBuilder<'_> {
    /// Declare a color output attachment for this pass.
    pub fn add_color_output(&mut self, _name: &str) -> ResourceHandle {
        todo!("Phase 6, Iteration 6.1")
    }

    /// Declare a texture input (sampled read) for this pass.
    pub fn add_texture_input(&mut self, _resource: ResourceHandle) {
        todo!("Phase 6, Iteration 6.1")
    }
}

/// Callback trait for recording commands within a render pass.
pub trait RenderPassCallback {
    /// Record draw commands for this pass.
    fn build_render_pass(&self, cmd: &mut ignis_command::command_buffer::CommandBuffer);

    /// Whether this callback needs a render pass (false for compute-only).
    fn need_render_pass(&self) -> bool {
        true
    }
}
