//! Windowing system integration: swapchain management and platform abstraction.
//!
//! Provides the [`WSI`] manager that owns a [`Device`](ignis_core::device::Device)
//! and handles swapchain creation, recreation, and frame presentation.

pub mod platform;
pub mod swapchain;
pub mod wsi;
