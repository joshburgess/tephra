//! High-level rendering context combining pipeline compilation and render pass caching.
//!
//! [`PipelineContext`] wraps [`PipelineCompiler`], [`PipelineCacheManager`], and
//! [`RenderPassCache`] into a single object that manages all pipeline-related
//! state. Use alongside a [`Device`](tephra_core::device::Device) or
//! [`WSI`](tephra_wsi::wsi::WSI).

use std::path::Path;

use ash::vk;

use tephra_command::state::StaticPipelineState;
use tephra_pipeline::pipeline::{PipelineCompiler, VertexInputLayout};
use tephra_pipeline::pipeline_cache::PipelineCacheManager;
use tephra_pipeline::program::Program;
use tephra_pipeline::render_pass::{RenderPassCache, RenderPassInfo};

/// Combines pipeline compilation, caching, and render pass management.
///
/// This is the recommended way to manage pipelines in tephra. Create one
/// per device and use it for all pipeline lookups.
///
/// # Example
///
/// ```ignore
/// let mut ctx = PipelineContext::new(device.raw(), Some(Path::new("pipeline_cache.bin")))?;
///
/// let pipeline = ctx.graphics_pipeline(
///     device.raw(), &program, &state, &rp_info, 0, &vertex_layout,
/// )?;
///
/// // At shutdown:
/// ctx.save(device.raw());
/// ctx.destroy(device.raw());
/// ```
pub struct PipelineContext {
    compiler: PipelineCompiler,
    cache_manager: PipelineCacheManager,
    render_pass_cache: RenderPassCache,
}

impl PipelineContext {
    /// Create a new pipeline context.
    ///
    /// If `cache_path` is provided and the file exists, the `VkPipelineCache`
    /// is seeded from disk to avoid redundant compilations.
    pub fn new(
        device: &ash::Device,
        cache_path: Option<&Path>,
    ) -> Result<Self, PipelineContextError> {
        let cache_manager =
            PipelineCacheManager::new(device, cache_path).map_err(PipelineContextError::Vulkan)?;
        let compiler = PipelineCompiler::new(cache_manager.cache());
        let render_pass_cache = RenderPassCache::new();

        Ok(Self {
            compiler,
            cache_manager,
            render_pass_cache,
        })
    }

    /// Look up or compile a graphics pipeline.
    ///
    /// Automatically resolves the render pass from the given info, then
    /// hashes the full pipeline key for cache lookup.
    #[allow(clippy::too_many_arguments)]
    pub fn graphics_pipeline(
        &mut self,
        device: &ash::Device,
        program: &Program,
        state: &StaticPipelineState,
        render_pass_info: &RenderPassInfo,
        subpass: u32,
        vertex_layout: &VertexInputLayout,
    ) -> Result<(vk::Pipeline, vk::RenderPass), PipelineContextError> {
        let render_pass = self
            .render_pass_cache
            .get_or_create(device, render_pass_info)
            .map_err(PipelineContextError::Vulkan)?;

        let rp_hash = RenderPassCache::compatible_hash(render_pass_info);

        let pipeline = self
            .compiler
            .get_or_compile_graphics(
                device,
                program,
                state,
                render_pass,
                rp_hash,
                subpass,
                vertex_layout,
            )
            .map_err(PipelineContextError::Vulkan)?;

        Ok((pipeline, render_pass))
    }

    /// Look up or compile a compute pipeline.
    pub fn compute_pipeline(
        &mut self,
        device: &ash::Device,
        program: &Program,
    ) -> Result<vk::Pipeline, PipelineContextError> {
        self.compiler
            .get_or_compile_compute(device, program)
            .map_err(PipelineContextError::Vulkan)
    }

    /// Look up or create a render pass for the given configuration.
    pub fn render_pass(
        &mut self,
        device: &ash::Device,
        info: &RenderPassInfo,
    ) -> Result<vk::RenderPass, PipelineContextError> {
        self.render_pass_cache
            .get_or_create(device, info)
            .map_err(PipelineContextError::Vulkan)
    }

    /// Save the pipeline cache to disk for cross-run reuse.
    pub fn save(&self, device: &ash::Device) {
        if let Err(e) = self.cache_manager.save(device) {
            log::warn!("Failed to save pipeline cache: {}", e);
        }
    }

    /// Destroy all Vulkan resources owned by this context.
    pub fn destroy(&mut self, device: &ash::Device) {
        self.save(device);
        self.compiler.destroy(device);
        self.cache_manager.destroy(device);
        self.render_pass_cache.destroy(device);
    }
}

/// Errors from pipeline context operations.
#[derive(Debug, thiserror::Error)]
pub enum PipelineContextError {
    /// A Vulkan API call failed.
    #[error("Vulkan error: {0}")]
    Vulkan(vk::Result),
}
