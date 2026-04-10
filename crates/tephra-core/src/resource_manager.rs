//! Asynchronous resource manager for images and meshes.
//!
//! Provides a background loading pipeline for GPU resources with fallback
//! textures, reference counting, and automatic staging buffer management.
//!
//! # Architecture
//!
//! Resources go through a lifecycle:
//! 1. **Requested** — a handle is returned immediately with a fallback texture.
//! 2. **Loading** — a background thread reads the file and stages upload data.
//! 3. **Uploading** — a transfer command buffer copies from staging to GPU memory.
//! 4. **Ready** — the resource is available for rendering.
//!
//! This allows the main render loop to reference resources immediately while
//! they load asynchronously in the background.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk;
use rustc_hash::FxHashMap;

/// Unique handle to a managed resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceHandle(u64);

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

impl ResourceHandle {
    fn new() -> Self {
        Self(NEXT_HANDLE.fetch_add(1, Ordering::Relaxed))
    }

    /// The raw handle value.
    pub fn raw(self) -> u64 {
        self.0
    }
}

/// The current state of a managed resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceState {
    /// Resource has been requested but loading hasn't started.
    Pending,
    /// Resource is being loaded from disk.
    Loading,
    /// Resource is being uploaded to the GPU.
    Uploading,
    /// Resource is ready for use.
    Ready,
    /// Resource failed to load.
    Failed,
}

/// Describes a texture resource to load.
#[derive(Debug, Clone)]
pub struct TextureLoadDesc {
    /// Path to the texture file.
    pub path: PathBuf,
    /// Whether to generate mipmaps after loading.
    pub generate_mipmaps: bool,
    /// Desired format (None = auto-detect from file).
    pub format: Option<vk::Format>,
    /// Whether this is an sRGB texture.
    pub srgb: bool,
}

/// Describes a mesh/buffer resource to load.
#[derive(Debug, Clone)]
pub struct BufferLoadDesc {
    /// Path to the mesh/buffer file.
    pub path: PathBuf,
    /// Buffer usage flags.
    pub usage: vk::BufferUsageFlags,
}

/// A managed resource entry.
struct ManagedResource {
    state: ResourceState,
    path: PathBuf,
    kind: ResourceKind,
}

enum ResourceKind {
    Texture {
        image: vk::Image,
        view: vk::ImageView,
        _generate_mipmaps: bool,
    },
    Buffer {
        buffer: vk::Buffer,
        size: vk::DeviceSize,
    },
    Pending,
}

/// The resource manager.
///
/// Tracks loaded resources, manages their lifecycle, and provides fallback
/// resources while loading is in progress.
pub struct ResourceManager {
    resources: FxHashMap<ResourceHandle, ManagedResource>,
    path_to_handle: FxHashMap<PathBuf, ResourceHandle>,
    fallback_image: vk::Image,
    fallback_view: vk::ImageView,
}

impl ResourceManager {
    /// Create a new resource manager.
    ///
    /// `fallback_image` and `fallback_view` are used as placeholders while
    /// resources are loading. Typically a 1x1 magenta or checkerboard texture.
    pub fn new(fallback_image: vk::Image, fallback_view: vk::ImageView) -> Self {
        Self {
            resources: FxHashMap::default(),
            path_to_handle: FxHashMap::default(),
            fallback_image,
            fallback_view,
        }
    }

    /// Request a texture resource. Returns a handle immediately.
    ///
    /// If the texture was already requested/loaded, returns the existing handle.
    pub fn request_texture(&mut self, desc: &TextureLoadDesc) -> ResourceHandle {
        if let Some(&handle) = self.path_to_handle.get(&desc.path) {
            return handle;
        }

        let handle = ResourceHandle::new();
        self.resources.insert(
            handle,
            ManagedResource {
                state: ResourceState::Pending,
                path: desc.path.clone(),
                kind: ResourceKind::Pending,
            },
        );
        self.path_to_handle.insert(desc.path.clone(), handle);

        log::debug!("Requested texture: {:?} -> {:?}", desc.path, handle);
        handle
    }

    /// Request a buffer resource. Returns a handle immediately.
    pub fn request_buffer(&mut self, desc: &BufferLoadDesc) -> ResourceHandle {
        if let Some(&handle) = self.path_to_handle.get(&desc.path) {
            return handle;
        }

        let handle = ResourceHandle::new();
        self.resources.insert(
            handle,
            ManagedResource {
                state: ResourceState::Pending,
                path: desc.path.clone(),
                kind: ResourceKind::Pending,
            },
        );
        self.path_to_handle.insert(desc.path.clone(), handle);

        log::debug!("Requested buffer: {:?} -> {:?}", desc.path, handle);
        handle
    }

    /// Query the state of a resource.
    pub fn resource_state(&self, handle: ResourceHandle) -> ResourceState {
        self.resources
            .get(&handle)
            .map(|r| r.state)
            .unwrap_or(ResourceState::Failed)
    }

    /// Get the image view for a texture resource.
    ///
    /// Returns the fallback view if the resource is not yet ready.
    pub fn texture_view(&self, handle: ResourceHandle) -> vk::ImageView {
        if let Some(res) = self.resources.get(&handle) {
            if let ResourceKind::Texture { view, .. } = &res.kind {
                if res.state == ResourceState::Ready {
                    return *view;
                }
            }
        }
        self.fallback_view
    }

    /// Get the image for a texture resource.
    ///
    /// Returns the fallback image if the resource is not yet ready.
    pub fn texture_image(&self, handle: ResourceHandle) -> vk::Image {
        if let Some(res) = self.resources.get(&handle) {
            if let ResourceKind::Texture { image, .. } = &res.kind {
                if res.state == ResourceState::Ready {
                    return *image;
                }
            }
        }
        self.fallback_image
    }

    /// Get the buffer and size for a buffer resource.
    ///
    /// Returns `None` if the resource is not yet ready.
    pub fn buffer(&self, handle: ResourceHandle) -> Option<(vk::Buffer, vk::DeviceSize)> {
        if let Some(res) = self.resources.get(&handle) {
            if let ResourceKind::Buffer { buffer, size } = &res.kind {
                if res.state == ResourceState::Ready {
                    return Some((*buffer, *size));
                }
            }
        }
        None
    }

    /// Mark a resource as ready with its GPU objects.
    ///
    /// Called by the upload system after the resource has been transferred
    /// to GPU memory.
    pub fn mark_texture_ready(
        &mut self,
        handle: ResourceHandle,
        image: vk::Image,
        view: vk::ImageView,
    ) {
        if let Some(res) = self.resources.get_mut(&handle) {
            res.state = ResourceState::Ready;
            res.kind = ResourceKind::Texture {
                image,
                view,
                _generate_mipmaps: false,
            };
        }
    }

    /// Mark a buffer resource as ready.
    pub fn mark_buffer_ready(
        &mut self,
        handle: ResourceHandle,
        buffer: vk::Buffer,
        size: vk::DeviceSize,
    ) {
        if let Some(res) = self.resources.get_mut(&handle) {
            res.state = ResourceState::Ready;
            res.kind = ResourceKind::Buffer { buffer, size };
        }
    }

    /// Mark a resource as failed.
    pub fn mark_failed(&mut self, handle: ResourceHandle) {
        if let Some(res) = self.resources.get_mut(&handle) {
            res.state = ResourceState::Failed;
            log::warn!("Resource failed to load: {:?}", res.path);
        }
    }

    /// Get all pending resources that need loading.
    pub fn pending_resources(&self) -> Vec<(ResourceHandle, &Path)> {
        self.resources
            .iter()
            .filter(|(_, r)| r.state == ResourceState::Pending)
            .map(|(h, r)| (*h, r.path.as_path()))
            .collect()
    }

    /// The fallback image view (used while resources load).
    pub fn fallback_view(&self) -> vk::ImageView {
        self.fallback_view
    }

    /// Total number of managed resources.
    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }

    /// Number of resources in the ready state.
    pub fn ready_count(&self) -> usize {
        self.resources
            .values()
            .filter(|r| r.state == ResourceState::Ready)
            .count()
    }
}
