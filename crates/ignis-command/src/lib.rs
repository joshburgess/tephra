//! Command buffer recording, linear allocators, and barrier helpers.
//!
//! Provides a [`CommandBuffer`](command_buffer::CommandBuffer) wrapper with
//! typed methods for barriers, copies, draws, dispatches, and render passes.
//! The [`linear_alloc`] module provides per-frame bump allocators for transient
//! vertex, index, and uniform data.

pub mod barriers;
pub mod command_buffer;
pub mod linear_alloc;
pub mod state;
