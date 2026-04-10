//! `VkPipelineCache` disk persistence.
//!
//! Creates a `VkPipelineCache` at startup, optionally loading cached data
//! from a file. Saves the cache data to disk on shutdown to avoid redundant
//! pipeline compilations across runs.

use std::path::{Path, PathBuf};

use ash::vk;

/// Manages `VkPipelineCache` creation, loading, and saving.
///
/// If a cache file exists on disk, it is loaded at creation time to seed the
/// driver's pipeline cache. On shutdown, call [`save`](Self::save) to persist
/// newly compiled pipelines.
pub struct PipelineCacheManager {
    cache: vk::PipelineCache,
    cache_path: Option<PathBuf>,
}

impl PipelineCacheManager {
    /// Create a new pipeline cache manager.
    ///
    /// If `cache_path` is provided and the file exists, the cache is initialized
    /// from the file's contents. Otherwise, an empty cache is created.
    pub fn new(device: &ash::Device, cache_path: Option<&Path>) -> Result<Self, vk::Result> {
        let initial_data = cache_path.and_then(|p| std::fs::read(p).ok());

        let cache_ci = if let Some(ref data) = initial_data {
            log::debug!("Loading pipeline cache from disk ({} bytes)", data.len());
            vk::PipelineCacheCreateInfo::default().initial_data(data)
        } else {
            vk::PipelineCacheCreateInfo::default()
        };

        // SAFETY: device is valid, cache_ci is well-formed.
        let cache = unsafe { device.create_pipeline_cache(&cache_ci, None)? };

        Ok(Self {
            cache,
            cache_path: cache_path.map(Path::to_path_buf),
        })
    }

    /// The raw `VkPipelineCache` handle.
    pub fn cache(&self) -> vk::PipelineCache {
        self.cache
    }

    /// Save the pipeline cache to disk.
    ///
    /// Call this before destroying the cache (typically at shutdown).
    pub fn save(&self, device: &ash::Device) -> Result<(), Box<dyn std::error::Error>> {
        let Some(ref path) = self.cache_path else {
            return Ok(());
        };

        // SAFETY: device and cache are valid.
        let data = unsafe { device.get_pipeline_cache_data(self.cache)? };

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(path, &data)?;

        log::debug!("Saved pipeline cache to disk ({} bytes)", data.len());

        Ok(())
    }

    /// Destroy the pipeline cache Vulkan object.
    pub fn destroy(&mut self, device: &ash::Device) {
        if self.cache != vk::PipelineCache::null() {
            // SAFETY: device is valid, cache is valid, GPU is idle.
            unsafe {
                device.destroy_pipeline_cache(self.cache, None);
            }
            self.cache = vk::PipelineCache::null();
        }
    }
}
