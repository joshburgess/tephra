//! Core Vulkan abstractions for the ignis rendering library.
//!
//! This crate provides the foundational types: Vulkan context initialization,
//! device abstraction, frame context ring buffer with deferred deletion,
//! buffer/image creation helpers, and sampler caching.

pub mod buffer;
pub mod context;
pub mod device;
pub mod frame_context;
pub mod handles;
pub mod image;
pub mod linear_alloc;
pub mod memory;
mod resources;
pub mod sampler;
mod submission;
pub mod sync;
