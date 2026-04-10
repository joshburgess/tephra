//! Shader reflection, program linking, and lazy pipeline compilation.
//!
//! Loads SPIR-V shaders, extracts reflection data via `spirv-reflect`,
//! auto-generates pipeline layouts, and compiles pipelines on demand
//! with hash-and-cache.

pub mod async_pipeline;
pub mod draw_context;
pub mod framebuffer_cache;
pub mod pipeline;
pub mod pipeline_cache;
pub mod program;
pub mod render_pass;
pub mod render_state;
pub mod shader;
pub mod shader_manager;
