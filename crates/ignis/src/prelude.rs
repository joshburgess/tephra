//! Convenience re-exports of the most commonly used ignis types.
//!
//! Use `use ignis::prelude::*` to import all common types at once.
//!
//! This module re-exports the types most frequently needed when writing
//! rendering code: device, command buffers, pipeline management, resource
//! creation, and WSI integration.

// Core types
pub use ignis_core::buffer::BufferCreateInfo;
pub use ignis_core::context::{ContextConfig, QueueType};
pub use ignis_core::device::Device;
pub use ignis_core::handles::{BufferHandle, ImageHandle};
pub use ignis_core::image::ImageCreateInfo;
pub use ignis_core::memory::{ImageDomain, MemoryDomain};
pub use ignis_core::sampler::StockSampler;

// Command buffer types
pub use ignis_command::barriers::ImageBarrierInfo;
pub use ignis_command::command_buffer::{CommandBuffer, CommandBufferType};

// Pipeline types
pub use ignis_pipeline::draw_context::{DrawContext, FrameResources, RenderingAttachment};
pub use ignis_pipeline::pipeline::VertexInputLayout;
pub use ignis_pipeline::program::Program;
pub use ignis_pipeline::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
pub use ignis_pipeline::shader::Shader;

// WSI types
pub use ignis_wsi::platform::WinitPlatform;
pub use ignis_wsi::wsi::{WSI, WSIConfig};
