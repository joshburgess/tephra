//! Pass declarations and render callbacks.
//!
//! Each pass declares its resource dependencies (what it reads and writes).
//! The graph compiler uses these declarations to determine execution order
//! and barrier placement.

use ignis_command::command_buffer::CommandBuffer;

use crate::resource::ResourceHandle;

/// How a pass accesses a resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AccessType {
    /// Write to a color attachment.
    ColorOutput,
    /// Read-modify-write a color attachment.
    ColorInput,
    /// Write to the depth/stencil attachment.
    DepthStencilOutput,
    /// Read from the depth/stencil attachment.
    DepthStencilInput,
    /// Sample as a texture (read-only).
    TextureInput,
    /// Read from a storage buffer/image.
    StorageRead,
    /// Write to a storage buffer/image.
    StorageWrite,
    /// Read as a subpass input attachment.
    AttachmentInput,
}

impl AccessType {
    /// Whether this access type writes to the resource.
    pub fn is_write(self) -> bool {
        matches!(
            self,
            AccessType::ColorOutput
                | AccessType::ColorInput
                | AccessType::DepthStencilOutput
                | AccessType::StorageWrite
        )
    }
}

/// A single resource access declared by a pass.
#[derive(Debug, Clone)]
pub(crate) struct ResourceAccess {
    pub resource: ResourceHandle,
    pub access_type: AccessType,
}

/// A declared render pass in the graph.
pub(crate) struct PassDeclaration {
    pub name: String,
    pub accesses: Vec<ResourceAccess>,
    pub callback: Option<Box<dyn RenderPassCallback>>,
    /// Whether this is a compute-only pass (no render pass needed).
    pub is_compute: bool,
}

/// Callback trait for recording commands within a render pass.
///
/// Implement this to provide the actual draw/dispatch commands for a pass.
pub trait RenderPassCallback {
    /// Record draw/dispatch commands for this pass.
    fn build_render_pass(&self, cmd: &mut CommandBuffer);

    /// Whether this callback needs a render pass (false for compute-only).
    fn need_render_pass(&self) -> bool {
        true
    }
}
