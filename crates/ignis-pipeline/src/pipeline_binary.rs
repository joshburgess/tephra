//! Pipeline binary caching via `VK_KHR_pipeline_binary`.
//!
//! An alternative to `VkPipelineCache` that provides direct access to compiled
//! pipeline binaries. This enables more efficient caching strategies:
//! - Per-pipeline binary blobs instead of a monolithic cache
//! - Fine-grained invalidation (only recompile changed pipelines)
//! - Parallel warm-up of specific pipelines
//!
//! # Extension Requirements
//!
//! Requires `VK_KHR_pipeline_binary` to be enabled at device creation.

/// A key identifying a specific pipeline binary.
///
/// Pipeline binary keys are opaque driver-specific identifiers returned
/// when creating pipeline binaries. They can be used to check if a cached
/// binary is still valid for the current driver version.
#[derive(Debug, Clone)]
pub struct PipelineBinaryKey {
    /// The raw key data (driver-specific format).
    pub data: Vec<u8>,
}

/// A compiled pipeline binary blob.
///
/// Contains the compiled GPU code for a single pipeline stage or the full
/// pipeline, depending on the implementation.
#[derive(Debug, Clone)]
pub struct PipelineBinaryData {
    /// The key identifying this binary.
    pub key: PipelineBinaryKey,
    /// The compiled binary data.
    pub data: Vec<u8>,
}

/// Manages pipeline binaries for disk caching.
///
/// Provides higher-level operations on top of `VK_KHR_pipeline_binary`
/// for storing and loading compiled pipeline binaries.
pub struct PipelineBinaryCache {
    path: std::path::PathBuf,
    binaries: Vec<PipelineBinaryData>,
}

impl PipelineBinaryCache {
    /// Create a new pipeline binary cache backed by the given directory.
    pub fn new(cache_dir: impl Into<std::path::PathBuf>) -> Self {
        let path = cache_dir.into();
        Self {
            path,
            binaries: Vec::new(),
        }
    }

    /// The cache directory path.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// The number of cached binaries.
    pub fn binary_count(&self) -> usize {
        self.binaries.len()
    }

    /// Add a binary to the cache.
    pub fn add_binary(&mut self, binary: PipelineBinaryData) {
        self.binaries.push(binary);
    }

    /// Get all cached binaries.
    pub fn binaries(&self) -> &[PipelineBinaryData] {
        &self.binaries
    }

    /// Save all binaries to disk.
    ///
    /// Each binary is written as a separate file named by a hash of its key.
    pub fn save(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.path)?;

        for (i, binary) in self.binaries.iter().enumerate() {
            let filename = format!("pipeline_{:016x}.bin", hash_key(&binary.key));
            let filepath = self.path.join(filename);

            use std::io::Write;
            let mut file = std::fs::File::create(&filepath)?;

            // Write key length + key data + binary data
            file.write_all(&(binary.key.data.len() as u32).to_le_bytes())?;
            file.write_all(&binary.key.data)?;
            file.write_all(&binary.data)?;

            log::debug!(
                "Saved pipeline binary {} ({} bytes) to {:?}",
                i,
                binary.data.len(),
                filepath
            );
        }

        Ok(())
    }

    /// Load all binaries from disk.
    pub fn load(&mut self) -> std::io::Result<()> {
        let read_dir = match std::fs::read_dir(&self.path) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e),
        };

        for entry in read_dir {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "bin") {
                match self.load_binary_file(&path) {
                    Ok(binary) => self.binaries.push(binary),
                    Err(e) => {
                        log::warn!("Failed to load pipeline binary {:?}: {}", path, e);
                    }
                }
            }
        }

        log::debug!(
            "Loaded {} pipeline binaries from {:?}",
            self.binaries.len(),
            self.path
        );

        Ok(())
    }

    fn load_binary_file(
        &self,
        path: &std::path::Path,
    ) -> std::io::Result<PipelineBinaryData> {
        use std::io::Read;

        let mut file = std::fs::File::open(path)?;

        let mut key_len_bytes = [0u8; 4];
        file.read_exact(&mut key_len_bytes)?;
        let key_len = u32::from_le_bytes(key_len_bytes) as usize;

        let mut key_data = vec![0u8; key_len];
        file.read_exact(&mut key_data)?;

        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        Ok(PipelineBinaryData {
            key: PipelineBinaryKey { data: key_data },
            data,
        })
    }
}

fn hash_key(key: &PipelineBinaryKey) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = rustc_hash::FxHasher::default();
    key.data.hash(&mut hasher);
    hasher.finish()
}

// Note: Direct Vulkan API wrappers for VK_KHR_pipeline_binary are not
// available in ash 0.38. When the extension is added to ash, add:
// - create_pipeline_binaries_from_pipeline()
// - destroy_pipeline_binaries()
// - get_pipeline_binary_data()
//
// For now, the PipelineBinaryCache provides a file-based caching layer
// that can be used with any serialization format for pipeline state.
