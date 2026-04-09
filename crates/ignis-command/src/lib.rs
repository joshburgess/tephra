//! Command buffer recording, linear allocators, and barrier helpers.
//!
//! Provides a [`CommandBuffer`] wrapper with state tracking, transient data
//! allocation via bump allocators, and barrier helper utilities.

pub mod barriers;
pub mod command_buffer;
pub mod linear_alloc;
pub mod state;
