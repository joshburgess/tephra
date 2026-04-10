//! Bump allocator for transient per-frame data (VBO, IBO, UBO).
//!
//! The actual allocator implementation lives in `tephra-core`. This module
//! re-exports the [`TransientAllocation`] type for convenience.

pub use tephra_core::linear_alloc::TransientAllocation;
