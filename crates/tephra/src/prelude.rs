//! Convenience re-exports of the most commonly used tephra types.
//!
//! Use `use tephra::prelude::*` to import all common types at once.
//!
//! This module re-exports the types most frequently needed when writing
//! rendering code: device, command buffers, pipeline management, resource
//! creation, and WSI integration.

// Core types
pub use tephra_core::buffer::BufferCreateInfo;
pub use tephra_core::context::{ContextConfig, QueueType};
pub use tephra_core::device::Device;
pub use tephra_core::handles::{BufferHandle, ImageHandle};
pub use tephra_core::image::ImageCreateInfo;
pub use tephra_core::memory::{ImageDomain, MemoryDomain};
pub use tephra_core::sampler::StockSampler;

// Command buffer types
pub use tephra_command::barriers::ImageBarrierInfo;
pub use tephra_command::command_buffer::{CommandBuffer, CommandBufferType};

// Pipeline types
pub use tephra_pipeline::draw_context::{DrawContext, FrameResources, RenderingAttachment};
pub use tephra_pipeline::pipeline::VertexInputLayout;
pub use tephra_pipeline::program::Program;
pub use tephra_pipeline::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
pub use tephra_pipeline::shader::Shader;

// WSI types
pub use tephra_wsi::platform::WinitPlatform;
pub use tephra_wsi::wsi::{WSI, WSIConfig};
