//! Render graph builder and top-level API.

use crate::pass::PassBuilder;
use crate::resource::ResourceHandle;

/// A render graph describing passes and their resource dependencies.
///
/// Build the graph by adding passes and declaring resources, then call
/// [`bake`](RenderGraph::bake) to compile it into an executable form.
pub struct RenderGraph {
    // TODO: Phase 6, Iteration 6.1
    // - passes: Vec<PassDeclaration>
    // - resources: Vec<ResourceDeclaration>
    _private: (),
}

impl RenderGraph {
    /// Create a new empty render graph.
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Add a rendering pass to the graph.
    pub fn add_pass(&mut self, _name: &str) -> PassBuilder<'_> {
        todo!("Phase 6, Iteration 6.1")
    }

    /// Set which resource represents the final backbuffer output.
    pub fn set_backbuffer_source(&mut self, _resource: ResourceHandle) {
        todo!("Phase 6, Iteration 6.1")
    }
}
