//! Render graph: automatic pass ordering, barrier placement, and resource management.
//!
//! Declare rendering passes and their resource dependencies, then let the graph
//! compiler determine execution order, insert barriers, merge subpasses for TBDR,
//! and alias transient resources.

pub mod alias;
pub mod compile;
pub mod execute;
pub mod graph;
pub mod pass;
pub mod resource;
pub mod subpass_merge;
