//! Fossilize pipeline state recording and replay.
//!
//! [Fossilize](https://github.com/ValveSoftware/Fossilize) is a pipeline state
//! serialization format used by Steam/Proton to record and pre-compile pipeline
//! states for faster startup times. This module provides an interface for
//! recording pipeline creation parameters and replaying them to warm up the
//! pipeline cache.
//!
//! # Architecture
//!
//! Pipeline states are serialized as a stream of tagged records:
//! - Shader modules (SPIR-V blobs)
//! - Sampler create infos
//! - Descriptor set layouts
//! - Pipeline layouts
//! - Render passes
//! - Graphics/compute pipeline create infos
//!
//! During normal operation, these records are captured as pipelines are created.
//! On subsequent launches, the recorded states are replayed in a background
//! thread to pre-warm the `VkPipelineCache`.
//!
//! # Usage
//!
//! ```ignore
//! let recorder = FossilizeRecorder::new("pipeline_cache.foz")?;
//! // ... create pipelines normally, recorder captures state ...
//! recorder.flush()?;
//!
//! // On next launch:
//! let replayer = FossilizeReplayer::new("pipeline_cache.foz", device, pipeline_cache)?;
//! replayer.replay_all()?; // Pre-compiles all recorded pipelines
//! ```

use std::path::{Path, PathBuf};

use ash::vk;
use rustc_hash::FxHashMap;

/// Magic bytes for the Fossilize database format header.
const FOSSILIZE_MAGIC: &[u8; 16] = b"FOSSILIZE_DB\x00\x00\x00\x01";

/// Tag identifying the type of a recorded object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum FossilizeTag {
    /// SPIR-V shader module.
    ShaderModule = 1,
    /// Sampler create info.
    Sampler = 2,
    /// Descriptor set layout.
    DescriptorSetLayout = 3,
    /// Pipeline layout.
    PipelineLayout = 4,
    /// Render pass (or render pass 2).
    RenderPass = 5,
    /// Graphics pipeline create info.
    GraphicsPipeline = 6,
    /// Compute pipeline create info.
    ComputePipeline = 7,
    /// Ray tracing pipeline create info.
    RayTracingPipeline = 8,
}

/// A recorded pipeline state entry.
#[derive(Debug, Clone)]
pub struct FossilizeEntry {
    /// The tag identifying the object type.
    pub tag: FossilizeTag,
    /// A content-based hash identifying this unique state combination.
    pub hash: u64,
    /// The serialized state data.
    pub data: Vec<u8>,
}

/// Records pipeline creation state for later replay.
///
/// Captures shader modules, pipeline layouts, render passes, and pipeline
/// create infos as they are created, serializing them to a database file.
pub struct FossilizeRecorder {
    path: PathBuf,
    entries: FxHashMap<(FossilizeTag, u64), Vec<u8>>,
}

impl FossilizeRecorder {
    /// Create a new recorder that will write to the given path.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_owned(),
            entries: FxHashMap::default(),
        }
    }

    /// Record a shader module (SPIR-V blob).
    pub fn record_shader_module(&mut self, hash: u64, spirv: &[u32]) {
        let data = bytemuck::cast_slice(spirv).to_vec();
        self.entries
            .insert((FossilizeTag::ShaderModule, hash), data);
    }

    /// Record a pipeline layout hash and its serialized create info.
    pub fn record_pipeline_layout(&mut self, hash: u64, data: Vec<u8>) {
        self.entries
            .insert((FossilizeTag::PipelineLayout, hash), data);
    }

    /// Record a graphics pipeline's serialized state.
    pub fn record_graphics_pipeline(&mut self, hash: u64, data: Vec<u8>) {
        self.entries
            .insert((FossilizeTag::GraphicsPipeline, hash), data);
    }

    /// Record a compute pipeline's serialized state.
    pub fn record_compute_pipeline(&mut self, hash: u64, data: Vec<u8>) {
        self.entries
            .insert((FossilizeTag::ComputePipeline, hash), data);
    }

    /// The number of recorded entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Flush all recorded entries to the database file.
    pub fn flush(&self) -> std::io::Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(&self.path)?;
        file.write_all(FOSSILIZE_MAGIC)?;

        // Write entry count
        let count = self.entries.len() as u32;
        file.write_all(&count.to_le_bytes())?;

        // Write each entry: tag (u32) + hash (u64) + data_len (u32) + data
        for ((tag, hash), data) in &self.entries {
            file.write_all(&(*tag as u32).to_le_bytes())?;
            file.write_all(&hash.to_le_bytes())?;
            file.write_all(&(data.len() as u32).to_le_bytes())?;
            file.write_all(data)?;
        }

        log::debug!("Fossilize: flushed {} entries to {:?}", count, self.path);

        Ok(())
    }
}

/// Replays recorded pipeline states to pre-warm the pipeline cache.
pub struct FossilizeReplayer {
    entries: Vec<FossilizeEntry>,
}

impl FossilizeReplayer {
    /// Load a Fossilize database from disk.
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Self> {
        use std::io::Read;

        let mut file = std::fs::File::open(path.as_ref())?;

        // Verify magic
        let mut magic = [0u8; 16];
        file.read_exact(&mut magic)?;
        if &magic != FOSSILIZE_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not a Fossilize database",
            ));
        }

        // Read entry count
        let mut count_bytes = [0u8; 4];
        file.read_exact(&mut count_bytes)?;
        let count = u32::from_le_bytes(count_bytes) as usize;

        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            let mut tag_bytes = [0u8; 4];
            file.read_exact(&mut tag_bytes)?;
            let tag_u32 = u32::from_le_bytes(tag_bytes);

            let mut hash_bytes = [0u8; 8];
            file.read_exact(&mut hash_bytes)?;
            let hash = u64::from_le_bytes(hash_bytes);

            let mut len_bytes = [0u8; 4];
            file.read_exact(&mut len_bytes)?;
            let data_len = u32::from_le_bytes(len_bytes) as usize;

            let mut data = vec![0u8; data_len];
            file.read_exact(&mut data)?;

            let tag = match tag_u32 {
                1 => FossilizeTag::ShaderModule,
                2 => FossilizeTag::Sampler,
                3 => FossilizeTag::DescriptorSetLayout,
                4 => FossilizeTag::PipelineLayout,
                5 => FossilizeTag::RenderPass,
                6 => FossilizeTag::GraphicsPipeline,
                7 => FossilizeTag::ComputePipeline,
                8 => FossilizeTag::RayTracingPipeline,
                _ => continue, // skip unknown tags
            };

            entries.push(FossilizeEntry { tag, hash, data });
        }

        log::debug!(
            "Fossilize: loaded {} entries from {:?}",
            entries.len(),
            path.as_ref()
        );

        Ok(Self { entries })
    }

    /// Get all recorded entries.
    pub fn entries(&self) -> &[FossilizeEntry] {
        &self.entries
    }

    /// Get entries of a specific tag type.
    pub fn entries_by_tag(&self, tag: FossilizeTag) -> impl Iterator<Item = &FossilizeEntry> {
        self.entries.iter().filter(move |e| e.tag == tag)
    }

    /// Get all shader module entries (SPIR-V blobs).
    pub fn shader_modules(&self) -> impl Iterator<Item = &FossilizeEntry> {
        self.entries_by_tag(FossilizeTag::ShaderModule)
    }

    /// Get all graphics pipeline entries.
    pub fn graphics_pipelines(&self) -> impl Iterator<Item = &FossilizeEntry> {
        self.entries_by_tag(FossilizeTag::GraphicsPipeline)
    }

    /// Get all compute pipeline entries.
    pub fn compute_pipelines(&self) -> impl Iterator<Item = &FossilizeEntry> {
        self.entries_by_tag(FossilizeTag::ComputePipeline)
    }
}

/// Replay a set of pipeline states into a pipeline cache.
///
/// This is a placeholder for the full replay implementation. A complete
/// implementation would deserialize the create infos and call
/// `vkCreateGraphicsPipelines` / `vkCreateComputePipelines` with the
/// provided pipeline cache for disk persistence.
pub fn replay_into_cache(
    _device: &ash::Device,
    _pipeline_cache: vk::PipelineCache,
    _replayer: &FossilizeReplayer,
) -> Result<u32, vk::Result> {
    // Full implementation would:
    // 1. Reconstruct shader modules from recorded SPIR-V
    // 2. Reconstruct pipeline layouts from recorded state
    // 3. Reconstruct render passes from recorded state
    // 4. Create pipelines with the pipeline cache
    // 5. Immediately destroy the pipelines (they're cached)
    log::debug!(
        "Fossilize: replay_into_cache is a stub — full replay requires state deserialization"
    );
    Ok(0)
}
