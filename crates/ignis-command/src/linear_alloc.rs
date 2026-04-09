//! Bump allocator for transient per-frame data (VBO, IBO, UBO).
//!
//! The actual allocator implementation lives in `ignis-core`. This module
//! re-exports the [`TransientAllocation`] type for convenience.

pub use ignis_core::linear_alloc::TransientAllocation;
