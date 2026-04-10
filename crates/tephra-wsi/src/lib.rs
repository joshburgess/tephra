//! Windowing system integration: swapchain management and platform abstraction.
//!
//! Provides the [`WSI`](wsi::WSI) manager that owns a
//! [`Device`](tephra_core::device::Device) and handles swapchain creation,
//! recreation, and frame presentation.

pub mod hdr;
pub mod headless;
pub mod platform;
pub mod pre_rotation;
pub mod swapchain;
pub mod wsi;
