//! Framebuffer caching by render pass and attachment views.
//!
//! Framebuffers are keyed by the render pass, image views, and dimensions.
//! They are cached and reused when the same combination is requested again.

use std::hash::{Hash, Hasher};

use ash::vk;
use rustc_hash::{FxHashMap, FxHasher};

/// Key for looking up a cached framebuffer.
#[derive(Clone, PartialEq, Eq)]
struct FramebufferKey {
    render_pass: vk::RenderPass,
    attachments: Vec<vk::ImageView>,
    width: u32,
    height: u32,
    layers: u32,
}

impl Hash for FramebufferKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.render_pass.hash(state);
        self.attachments.hash(state);
        self.width.hash(state);
        self.height.hash(state);
        self.layers.hash(state);
    }
}

/// Cache for `VkFramebuffer` objects.
///
/// Framebuffers are expensive to create and should be reused when the same
/// render pass, attachment views, and dimensions are requested again. This
/// cache is typically reset per-frame or when the swapchain is recreated.
pub struct FramebufferCache {
    cache: FxHashMap<u64, vk::Framebuffer>,
}

impl FramebufferCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            cache: FxHashMap::default(),
        }
    }

    /// Get or create a framebuffer for the given render pass and attachments.
    pub fn get_or_create(
        &mut self,
        device: &ash::Device,
        render_pass: vk::RenderPass,
        attachments: &[vk::ImageView],
        width: u32,
        height: u32,
    ) -> Result<vk::Framebuffer, vk::Result> {
        let key = FramebufferKey {
            render_pass,
            attachments: attachments.to_vec(),
            width,
            height,
            layers: 1,
        };

        let hash = {
            let mut hasher = FxHasher::default();
            key.hash(&mut hasher);
            hasher.finish()
        };

        if let Some(&fb) = self.cache.get(&hash) {
            return Ok(fb);
        }

        let fb_ci = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(attachments)
            .width(width)
            .height(height)
            .layers(1);

        // SAFETY: device is valid, render_pass and attachments are valid.
        let fb = unsafe { device.create_framebuffer(&fb_ci, None)? };

        log::debug!(
            "Created framebuffer {}x{} (total cached: {})",
            width,
            height,
            self.cache.len() + 1
        );

        self.cache.insert(hash, fb);
        Ok(fb)
    }

    /// Clear the cache, destroying all framebuffers.
    pub fn reset(&mut self, device: &ash::Device) {
        for (_, fb) in self.cache.drain() {
            // SAFETY: device is valid, framebuffer is valid, GPU is idle.
            unsafe {
                device.destroy_framebuffer(fb, None);
            }
        }
    }

    /// Destroy all cached framebuffers.
    pub fn destroy(&mut self, device: &ash::Device) {
        self.reset(device);
    }
}

impl Default for FramebufferCache {
    fn default() -> Self {
        Self::new()
    }
}
