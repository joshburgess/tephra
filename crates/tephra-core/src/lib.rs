//! Core Vulkan abstractions for the tephra rendering library.
//!
//! This crate provides the foundational types: Vulkan context initialization,
//! device abstraction, frame context ring buffer with deferred deletion,
//! buffer/image creation helpers, and sampler caching.

pub mod buffer;
pub mod buffer_pool;
pub mod context;
pub mod cookie;
pub mod device;
pub mod external;
pub mod format;
pub mod frame_context;
pub mod handles;
pub mod image;
pub mod linear_alloc;
pub mod memory;
pub mod post_mortem;
pub mod quirks;
pub mod renderdoc;
pub mod resource_manager;
mod resources;
pub mod rtas;
pub mod sampler;
mod submission;
pub mod sync;
pub mod video;
