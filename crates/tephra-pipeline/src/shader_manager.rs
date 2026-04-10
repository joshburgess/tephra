//! Shader lifecycle management with optional hot-reload.
//!
//! The [`ShaderManager`] tracks loaded shaders by path and provides a
//! convenience API for loading SPIR-V from disk. When the `hot-reload`
//! feature is enabled, it watches shader directories for changes and
//! signals when recompilation is needed.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ash::vk;

use crate::shader::Shader;

/// Manages shader modules with optional file-watching for hot-reload.
///
/// # Without `hot-reload`
///
/// Acts as a simple registry: load shaders by path and retrieve them later.
///
/// # With `hot-reload`
///
/// Watches registered shader directories for file modifications. Call
/// [`poll_changes`](ShaderManager::poll_changes) each frame to check for
/// invalidated shaders.
pub struct ShaderManager {
    shaders: HashMap<PathBuf, ManagedShader>,
    #[cfg(feature = "hot-reload")]
    watcher: Option<HotReloadWatcher>,
    #[cfg(feature = "hot-reload")]
    changed: Vec<PathBuf>,
}

struct ManagedShader {
    shader: Shader,
    stage: vk::ShaderStageFlags,
}

impl ShaderManager {
    /// Create a new shader manager.
    pub fn new() -> Self {
        Self {
            shaders: HashMap::new(),
            #[cfg(feature = "hot-reload")]
            watcher: None,
            #[cfg(feature = "hot-reload")]
            changed: Vec::new(),
        }
    }

    /// Load a SPIR-V shader from disk and register it.
    ///
    /// If the shader was already loaded from this path, the old module is
    /// destroyed and replaced.
    pub fn load(
        &mut self,
        device: &ash::Device,
        path: &Path,
        stage: vk::ShaderStageFlags,
    ) -> Result<&Shader, ShaderManagerError> {
        let spirv_bytes =
            std::fs::read(path).map_err(|e| ShaderManagerError::Io(path.to_path_buf(), e))?;

        if spirv_bytes.len() % 4 != 0 {
            return Err(ShaderManagerError::InvalidSpirv(
                path.to_path_buf(),
                "file size is not a multiple of 4".into(),
            ));
        }

        // SAFETY: We verified the byte slice length is a multiple of 4.
        // The alignment is handled by the Vec's allocator.
        let spirv: Vec<u32> = spirv_bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let shader = Shader::create(device, stage, &spirv)
            .map_err(|e| ShaderManagerError::InvalidSpirv(path.to_path_buf(), e))?;

        // Destroy old shader if replacing
        if let Some(mut old) = self.shaders.remove(path) {
            old.shader.destroy(device);
        }

        let canonical = path.to_path_buf();
        self.shaders
            .insert(canonical.clone(), ManagedShader { shader, stage });

        log::debug!("Loaded shader: {} (stage={:?})", path.display(), stage);

        Ok(&self.shaders[&canonical].shader)
    }

    /// Get a previously loaded shader by path.
    pub fn get(&self, path: &Path) -> Option<&Shader> {
        self.shaders.get(path).map(|m| &m.shader)
    }

    /// Reload a shader from disk (e.g., after detecting a file change).
    pub fn reload(
        &mut self,
        device: &ash::Device,
        path: &Path,
    ) -> Result<&Shader, ShaderManagerError> {
        let stage = self.shaders.get(path).map(|m| m.stage).ok_or_else(|| {
            ShaderManagerError::InvalidSpirv(
                path.to_path_buf(),
                "shader not previously loaded".into(),
            )
        })?;
        self.load(device, path, stage)
    }

    /// Destroy all managed shader modules.
    pub fn destroy(&mut self, device: &ash::Device) {
        for (_, mut managed) in self.shaders.drain() {
            managed.shader.destroy(device);
        }
    }
}

impl Default for ShaderManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Hot-reload support (behind feature flag)
// ---------------------------------------------------------------------------

#[cfg(feature = "hot-reload")]
mod hot_reload {
    use std::path::{Path, PathBuf};
    use std::sync::mpsc;

    use notify::{RecommendedWatcher, RecursiveMode, Watcher};

    pub(super) struct HotReloadWatcher {
        _watcher: RecommendedWatcher,
        rx: mpsc::Receiver<PathBuf>,
    }

    impl HotReloadWatcher {
        pub fn new(watch_dirs: &[&Path]) -> Result<Self, notify::Error> {
            let (tx, rx) = mpsc::channel();

            let mut watcher = notify::recommended_watcher(move |res: Result<notify::Event, _>| {
                if let Ok(event) = res {
                    if event.kind.is_modify() || event.kind.is_create() {
                        for path in event.paths {
                            if path
                                .extension()
                                .is_some_and(|ext| ext == "spv" || ext == "spirv")
                            {
                                tx.send(path).ok();
                            }
                        }
                    }
                }
            })?;

            for dir in watch_dirs {
                watcher.watch(dir, RecursiveMode::Recursive)?;
            }

            Ok(Self {
                _watcher: watcher,
                rx,
            })
        }

        pub fn poll(&self) -> Vec<PathBuf> {
            let mut changed = Vec::new();
            while let Ok(path) = self.rx.try_recv() {
                if !changed.contains(&path) {
                    changed.push(path);
                }
            }
            changed
        }
    }
}

#[cfg(feature = "hot-reload")]
use hot_reload::HotReloadWatcher;

#[cfg(feature = "hot-reload")]
impl ShaderManager {
    /// Enable hot-reload by watching the given directories for SPIR-V changes.
    pub fn enable_hot_reload(&mut self, watch_dirs: &[&Path]) -> Result<(), ShaderManagerError> {
        let watcher = HotReloadWatcher::new(watch_dirs)
            .map_err(|e| ShaderManagerError::WatchError(e.to_string()))?;
        self.watcher = Some(watcher);
        log::info!(
            "Shader hot-reload enabled for {} directories",
            watch_dirs.len()
        );
        Ok(())
    }

    /// Poll for shader file changes. Returns paths that were modified since the last poll.
    ///
    /// Call this once per frame. Use the returned paths with [`reload`](ShaderManager::reload)
    /// to update shader modules, then invalidate any programs/pipelines that reference them.
    pub fn poll_changes(&mut self) -> &[PathBuf] {
        self.changed.clear();
        if let Some(ref watcher) = self.watcher {
            self.changed = watcher.poll();
        }
        &self.changed
    }
}

/// Errors from shader manager operations.
#[derive(Debug, thiserror::Error)]
pub enum ShaderManagerError {
    /// Failed to read a shader file.
    #[error("failed to read shader {0}: {1}")]
    Io(PathBuf, std::io::Error),

    /// The SPIR-V data is invalid.
    #[error("invalid SPIR-V in {0}: {1}")]
    InvalidSpirv(PathBuf, String),

    /// Failed to set up file watching.
    #[error("failed to set up file watcher: {0}")]
    WatchError(String),
}
