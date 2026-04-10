//! tephra — A Granite-inspired mid-level Vulkan abstraction layer for Rust.
//!
//! This is the umbrella crate re-exporting all tephra sub-crates for convenience.
//!
//! # Crate Organization
//!
//! - [`core`] — Context, device, frame management, resource creation
//! - [`command`] — Command buffer recording, linear allocators, barriers
//! - [`descriptors`] — Descriptor set allocation, binding, caching
//! - [`pipeline`] — Shader reflection, program linking, pipeline compilation
//! - [`wsi`] — Windowing, swapchain, platform integration
//! - [`graph`] — Render graph with automatic pass ordering and barriers
//! - [`renderer`] — High-level pipeline context combining compilation and caching

pub use tephra_command as command;
pub use tephra_core as core;
pub use tephra_descriptors as descriptors;
pub use tephra_graph as graph;
pub use tephra_pipeline as pipeline;
pub use tephra_wsi as wsi;

pub mod prelude;
pub mod renderer;
