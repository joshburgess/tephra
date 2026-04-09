//! Descriptor set management: allocation, binding, and caching.
//!
//! Provides a slot-based descriptor binding model with automatic set allocation,
//! hashing, and per-frame caching.

pub mod binding_table;
pub mod cache;
pub mod set_allocator;
