//! Render graph builder and top-level API.
//!
//! Build a render graph by adding passes and declaring resources, then call
//! [`bake`](RenderGraph::bake) to compile it into an executable form.

use crate::compile::CompiledGraph;
use crate::pass::{AccessType, PassDeclaration, RenderPassCallback, ResourceAccess};
use crate::resource::{
    AttachmentInfo, BufferInfo, ResourceDeclaration, ResourceHandle, ResourceInfo,
};

/// A render graph describing passes and their resource dependencies.
///
/// Build the graph by adding passes and declaring resources, then call
/// [`bake`](RenderGraph::bake) to compile it into an executable form.
pub struct RenderGraph {
    pub(crate) passes: Vec<PassDeclaration>,
    pub(crate) resources: Vec<ResourceDeclaration>,
    pub(crate) backbuffer: Option<ResourceHandle>,
}

impl RenderGraph {
    /// Create a new empty render graph.
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            resources: Vec::new(),
            backbuffer: None,
        }
    }

    /// Add a rendering pass to the graph. Returns a builder for declaring
    /// the pass's resource dependencies.
    pub fn add_pass(&mut self, name: &str) -> PassBuilder<'_> {
        let pass_index = self.passes.len();
        self.passes.push(PassDeclaration {
            name: name.to_string(),
            accesses: Vec::new(),
            callback: None,
            is_compute: false,
        });
        PassBuilder {
            graph: self,
            pass_index,
        }
    }

    /// Set which resource represents the final backbuffer output.
    ///
    /// The compiler uses this to determine which passes are reachable
    /// and to set the final image layout to `PRESENT_SRC_KHR`.
    pub fn set_backbuffer_source(&mut self, resource: ResourceHandle) {
        self.backbuffer = Some(resource);
    }

    /// Compile the graph into an executable form.
    ///
    /// Performs dependency analysis, topological ordering, barrier placement,
    /// subpass merge detection, and resource aliasing analysis.
    pub fn bake(self) -> CompiledGraph {
        CompiledGraph::compile(self)
    }
}

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for declaring a render pass and its resource dependencies.
pub struct PassBuilder<'a> {
    graph: &'a mut RenderGraph,
    pass_index: usize,
}

impl PassBuilder<'_> {
    /// Declare a color output attachment created by this pass.
    pub fn add_color_output(
        &mut self,
        name: &str,
        info: AttachmentInfo,
    ) -> ResourceHandle {
        let handle = self.create_resource(name, ResourceInfo::Attachment(info));
        self.add_access(handle, AccessType::ColorOutput);
        handle
    }

    /// Declare a color input (read-modify-write) for this pass.
    pub fn add_color_input(&mut self, resource: ResourceHandle) {
        self.add_access(resource, AccessType::ColorInput);
    }

    /// Declare a depth/stencil output attachment created by this pass.
    pub fn add_depth_stencil_output(
        &mut self,
        name: &str,
        info: AttachmentInfo,
    ) -> ResourceHandle {
        let handle = self.create_resource(name, ResourceInfo::Attachment(info));
        self.add_access(handle, AccessType::DepthStencilOutput);
        handle
    }

    /// Declare a depth/stencil input for this pass.
    pub fn add_depth_stencil_input(&mut self, resource: ResourceHandle) {
        self.add_access(resource, AccessType::DepthStencilInput);
    }

    /// Declare a texture input (sampled read) for this pass.
    pub fn add_texture_input(&mut self, resource: ResourceHandle) {
        self.add_access(resource, AccessType::TextureInput);
    }

    /// Declare a storage buffer/image output created by this pass.
    pub fn add_storage_output(
        &mut self,
        name: &str,
        info: BufferInfo,
    ) -> ResourceHandle {
        let handle = self.create_resource(name, ResourceInfo::Buffer(info));
        self.add_access(handle, AccessType::StorageWrite);
        handle
    }

    /// Declare a storage buffer/image read for this pass.
    pub fn add_storage_input(&mut self, resource: ResourceHandle) {
        self.add_access(resource, AccessType::StorageRead);
    }

    /// Declare a subpass input attachment read for this pass.
    pub fn add_attachment_input(&mut self, resource: ResourceHandle) {
        self.add_access(resource, AccessType::AttachmentInput);
    }

    /// Mark this pass as compute-only (no render pass needed).
    pub fn set_compute(&mut self) {
        self.graph.passes[self.pass_index].is_compute = true;
    }

    /// Set the render callback for this pass.
    pub fn set_render_callback(&mut self, callback: Box<dyn RenderPassCallback>) {
        self.graph.passes[self.pass_index].callback = Some(callback);
    }

    fn create_resource(&mut self, name: &str, info: ResourceInfo) -> ResourceHandle {
        let index = self.graph.resources.len() as u32;
        self.graph.resources.push(ResourceDeclaration {
            name: name.to_string(),
            info,
        });
        ResourceHandle { index }
    }

    fn add_access(&mut self, resource: ResourceHandle, access_type: AccessType) {
        self.graph.passes[self.pass_index]
            .accesses
            .push(ResourceAccess {
                resource,
                access_type,
            });
    }
}
