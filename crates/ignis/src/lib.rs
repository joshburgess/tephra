//! ignis — A Granite-inspired mid-level Vulkan abstraction layer for Rust.
//!
//! This is the umbrella crate re-exporting all ignis sub-crates for convenience.
//!
//! # Crate Organization
//!
//! - [`core`] — Context, device, frame management, resource creation
//! - [`command`] — Command buffer recording, linear allocators, barriers
//! - [`descriptors`] — Descriptor set allocation, binding, caching
//! - [`pipeline`] — Shader reflection, program linking, pipeline compilation
//! - [`wsi`] — Windowing, swapchain, platform integration
//! - [`graph`] — Render graph with automatic pass ordering and barriers

pub use ignis_core as core;
pub use ignis_command as command;
pub use ignis_descriptors as descriptors;
pub use ignis_pipeline as pipeline;
pub use ignis_wsi as wsi;
pub use ignis_graph as graph;
