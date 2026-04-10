//! Render graph: automatic pass ordering, barrier placement, and resource management.
//!
//! Declare rendering passes and their resource dependencies, then let the graph
//! compiler determine execution order, insert barriers, merge subpasses for TBDR,
//! and alias transient resources.
//!
//! # Usage
//!
//! ```ignore
//! let mut graph = RenderGraph::new();
//!
//! let mut shadow = graph.add_pass("shadow");
//! let shadow_map = shadow.add_depth_stencil_output("shadow_map", depth_info);
//!
//! let mut lighting = graph.add_pass("lighting");
//! lighting.add_texture_input(shadow_map);
//! let hdr = lighting.add_color_output("hdr", hdr_info);
//!
//! graph.set_backbuffer_source(hdr);
//! let compiled = graph.bake();
//! ```

pub mod alias;
pub mod allocate;
pub mod compile;
pub mod execute;
pub mod graph;
pub mod pass;
pub mod resource;
pub mod subpass_merge;

pub use allocate::PhysicalResources;
pub use compile::CompiledGraph;
pub use execute::GraphExecutor;
pub use graph::{PassBuilder, RenderGraph};
pub use pass::RenderPassCallback;
pub use resource::{AttachmentInfo, BufferInfo, ResourceHandle, SizeClass};
