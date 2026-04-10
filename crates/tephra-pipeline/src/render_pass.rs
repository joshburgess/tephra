//! Render pass auto-generation and caching.
//!
//! Hashes render pass compatibility keys and caches `VkRenderPass` objects.
//! Automates subpass dependency generation and image layout transitions.

use std::hash::{Hash, Hasher};

use ash::vk;
use rustc_hash::{FxHashMap, FxHasher};

/// Load operation for an attachment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttachmentLoadOp {
    /// Clear the attachment at the start of the render pass.
    Clear,
    /// Preserve existing contents.
    Load,
    /// Contents are undefined (don't care).
    DontCare,
}

impl From<AttachmentLoadOp> for vk::AttachmentLoadOp {
    fn from(op: AttachmentLoadOp) -> Self {
        match op {
            AttachmentLoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
            AttachmentLoadOp::Load => vk::AttachmentLoadOp::LOAD,
            AttachmentLoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
        }
    }
}

/// Store operation for an attachment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttachmentStoreOp {
    /// Store the attachment results.
    Store,
    /// Contents are undefined after the render pass (don't care).
    DontCare,
}

impl From<AttachmentStoreOp> for vk::AttachmentStoreOp {
    fn from(op: AttachmentStoreOp) -> Self {
        match op {
            AttachmentStoreOp::Store => vk::AttachmentStoreOp::STORE,
            AttachmentStoreOp::DontCare => vk::AttachmentStoreOp::DONT_CARE,
        }
    }
}

/// Description of a single color attachment in a render pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorAttachmentInfo {
    /// The format of the color attachment.
    pub format: vk::Format,
    /// Load operation at the start of the render pass.
    pub load_op: AttachmentLoadOp,
    /// Store operation at the end of the render pass.
    pub store_op: AttachmentStoreOp,
}

/// Description of the depth/stencil attachment in a render pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthStencilAttachmentInfo {
    /// The format of the depth/stencil attachment.
    pub format: vk::Format,
    /// Depth load operation.
    pub depth_load_op: AttachmentLoadOp,
    /// Depth store operation.
    pub depth_store_op: AttachmentStoreOp,
    /// Stencil load operation.
    pub stencil_load_op: AttachmentLoadOp,
    /// Stencil store operation.
    pub stencil_store_op: AttachmentStoreOp,
}

/// Parameters describing a render pass configuration (Granite-style).
///
/// Users describe what they want — color attachments, depth/stencil, load/store
/// ops, and sample count — and the cache generates the `VkRenderPass` with
/// correct subpass dependencies and layout transitions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RenderPassInfo {
    /// Color attachment descriptions.
    pub color_attachments: Vec<ColorAttachmentInfo>,
    /// Depth/stencil attachment, if any.
    pub depth_stencil: Option<DepthStencilAttachmentInfo>,
    /// Multisample count.
    pub samples: vk::SampleCountFlags,
}

/// Cache for `VkRenderPass` objects.
///
/// Render passes are hashed by their full description and reused when
/// an identical configuration is requested again.
pub struct RenderPassCache {
    cache: FxHashMap<u64, vk::RenderPass>,
}

impl RenderPassCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            cache: FxHashMap::default(),
        }
    }

    /// Get or create a render pass for the given info.
    pub fn get_or_create(
        &mut self,
        device: &ash::Device,
        info: &RenderPassInfo,
    ) -> Result<vk::RenderPass, vk::Result> {
        let hash = Self::hash_info(info);

        if let Some(&rp) = self.cache.get(&hash) {
            return Ok(rp);
        }

        let rp = Self::create_render_pass(device, info)?;
        self.cache.insert(hash, rp);

        log::debug!("Created render pass (total cached: {})", self.cache.len());

        Ok(rp)
    }

    /// Compute the compatible render pass hash for pipeline key computation.
    pub fn compatible_hash(info: &RenderPassInfo) -> u64 {
        Self::hash_info(info)
    }

    /// Destroy all cached render passes.
    pub fn destroy(&mut self, device: &ash::Device) {
        for (_, rp) in self.cache.drain() {
            // SAFETY: device is valid, render pass is valid, GPU is idle.
            unsafe {
                device.destroy_render_pass(rp, None);
            }
        }
    }

    fn hash_info(info: &RenderPassInfo) -> u64 {
        let mut hasher = FxHasher::default();
        info.hash(&mut hasher);
        hasher.finish()
    }

    fn create_render_pass(
        device: &ash::Device,
        info: &RenderPassInfo,
    ) -> Result<vk::RenderPass, vk::Result> {
        let mut attachments: Vec<vk::AttachmentDescription> = Vec::new();
        let mut color_refs: Vec<vk::AttachmentReference> = Vec::new();

        // Color attachments
        for (i, color) in info.color_attachments.iter().enumerate() {
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(color.format)
                    .samples(info.samples)
                    .load_op(color.load_op.into())
                    .store_op(color.store_op.into())
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(if color.load_op == AttachmentLoadOp::Load {
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                    } else {
                        vk::ImageLayout::UNDEFINED
                    })
                    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            );

            color_refs.push(vk::AttachmentReference {
                attachment: i as u32,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            });
        }

        // Depth/stencil attachment
        let depth_ref;
        if let Some(depth) = &info.depth_stencil {
            let attachment_idx = attachments.len() as u32;
            let has_stencil = matches!(
                depth.format,
                vk::Format::D16_UNORM_S8_UINT
                    | vk::Format::D24_UNORM_S8_UINT
                    | vk::Format::D32_SFLOAT_S8_UINT
                    | vk::Format::S8_UINT
            );

            attachments.push(
                vk::AttachmentDescription::default()
                    .format(depth.format)
                    .samples(info.samples)
                    .load_op(depth.depth_load_op.into())
                    .store_op(depth.depth_store_op.into())
                    .stencil_load_op(if has_stencil {
                        depth.stencil_load_op.into()
                    } else {
                        vk::AttachmentLoadOp::DONT_CARE
                    })
                    .stencil_store_op(if has_stencil {
                        depth.stencil_store_op.into()
                    } else {
                        vk::AttachmentStoreOp::DONT_CARE
                    })
                    .initial_layout(if depth.depth_load_op == AttachmentLoadOp::Load {
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                    } else {
                        vk::ImageLayout::UNDEFINED
                    })
                    .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
            );

            depth_ref = Some(vk::AttachmentReference {
                attachment: attachment_idx,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            });
        } else {
            depth_ref = None;
        }

        // Single subpass
        let mut subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs);

        if let Some(ref depth) = depth_ref {
            subpass = subpass.depth_stencil_attachment(depth);
        }

        // Subpass dependencies: external -> subpass 0
        let dependencies = [vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )];

        let render_pass_ci = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(&dependencies);

        // SAFETY: device is valid, render_pass_ci is well-formed.
        let render_pass = unsafe { device.create_render_pass(&render_pass_ci, None)? };

        Ok(render_pass)
    }
}

impl Default for RenderPassCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- AttachmentLoadOp / StoreOp conversions ---

    #[test]
    fn load_op_conversion() {
        assert_eq!(vk::AttachmentLoadOp::from(AttachmentLoadOp::Clear), vk::AttachmentLoadOp::CLEAR);
        assert_eq!(vk::AttachmentLoadOp::from(AttachmentLoadOp::Load), vk::AttachmentLoadOp::LOAD);
        assert_eq!(
            vk::AttachmentLoadOp::from(AttachmentLoadOp::DontCare),
            vk::AttachmentLoadOp::DONT_CARE
        );
    }

    #[test]
    fn store_op_conversion() {
        assert_eq!(
            vk::AttachmentStoreOp::from(AttachmentStoreOp::Store),
            vk::AttachmentStoreOp::STORE
        );
        assert_eq!(
            vk::AttachmentStoreOp::from(AttachmentStoreOp::DontCare),
            vk::AttachmentStoreOp::DONT_CARE
        );
    }

    // --- RenderPassInfo hashing ---

    fn make_single_color_pass(format: vk::Format, load: AttachmentLoadOp) -> RenderPassInfo {
        RenderPassInfo {
            color_attachments: vec![ColorAttachmentInfo {
                format,
                load_op: load,
                store_op: AttachmentStoreOp::Store,
            }],
            depth_stencil: None,
            samples: vk::SampleCountFlags::TYPE_1,
        }
    }

    #[test]
    fn identical_pass_info_same_hash() {
        let a = make_single_color_pass(vk::Format::B8G8R8A8_SRGB, AttachmentLoadOp::Clear);
        let b = make_single_color_pass(vk::Format::B8G8R8A8_SRGB, AttachmentLoadOp::Clear);
        assert_eq!(
            RenderPassCache::compatible_hash(&a),
            RenderPassCache::compatible_hash(&b)
        );
    }

    #[test]
    fn different_format_different_hash() {
        let a = make_single_color_pass(vk::Format::B8G8R8A8_SRGB, AttachmentLoadOp::Clear);
        let b = make_single_color_pass(vk::Format::R8G8B8A8_UNORM, AttachmentLoadOp::Clear);
        assert_ne!(
            RenderPassCache::compatible_hash(&a),
            RenderPassCache::compatible_hash(&b)
        );
    }

    #[test]
    fn different_load_op_different_hash() {
        let a = make_single_color_pass(vk::Format::B8G8R8A8_SRGB, AttachmentLoadOp::Clear);
        let b = make_single_color_pass(vk::Format::B8G8R8A8_SRGB, AttachmentLoadOp::Load);
        assert_ne!(
            RenderPassCache::compatible_hash(&a),
            RenderPassCache::compatible_hash(&b)
        );
    }

    #[test]
    fn different_sample_count_different_hash() {
        let a = RenderPassInfo {
            color_attachments: vec![ColorAttachmentInfo {
                format: vk::Format::B8G8R8A8_SRGB,
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::Store,
            }],
            depth_stencil: None,
            samples: vk::SampleCountFlags::TYPE_1,
        };
        let b = RenderPassInfo {
            samples: vk::SampleCountFlags::TYPE_4,
            ..a.clone()
        };
        assert_ne!(
            RenderPassCache::compatible_hash(&a),
            RenderPassCache::compatible_hash(&b)
        );
    }

    #[test]
    fn with_depth_different_from_without() {
        let without = make_single_color_pass(vk::Format::B8G8R8A8_SRGB, AttachmentLoadOp::Clear);
        let with = RenderPassInfo {
            depth_stencil: Some(DepthStencilAttachmentInfo {
                format: vk::Format::D32_SFLOAT,
                depth_load_op: AttachmentLoadOp::Clear,
                depth_store_op: AttachmentStoreOp::DontCare,
                stencil_load_op: AttachmentLoadOp::DontCare,
                stencil_store_op: AttachmentStoreOp::DontCare,
            }),
            ..without.clone()
        };
        assert_ne!(
            RenderPassCache::compatible_hash(&without),
            RenderPassCache::compatible_hash(&with)
        );
    }

    #[test]
    fn two_color_attachments_different_from_one() {
        let one = make_single_color_pass(vk::Format::B8G8R8A8_SRGB, AttachmentLoadOp::Clear);
        let two = RenderPassInfo {
            color_attachments: vec![
                ColorAttachmentInfo {
                    format: vk::Format::B8G8R8A8_SRGB,
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                },
                ColorAttachmentInfo {
                    format: vk::Format::R16G16B16A16_SFLOAT,
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                },
            ],
            depth_stencil: None,
            samples: vk::SampleCountFlags::TYPE_1,
        };
        assert_ne!(
            RenderPassCache::compatible_hash(&one),
            RenderPassCache::compatible_hash(&two)
        );
    }

    #[test]
    fn hash_stability() {
        let info = make_single_color_pass(vk::Format::B8G8R8A8_SRGB, AttachmentLoadOp::Clear);
        let h1 = RenderPassCache::compatible_hash(&info);
        let h2 = RenderPassCache::compatible_hash(&info);
        assert_eq!(h1, h2);
    }

    #[test]
    fn cache_starts_empty() {
        let cache = RenderPassCache::new();
        assert_eq!(cache.cache.len(), 0);
    }

    // --- DepthStencilAttachmentInfo ---

    #[test]
    fn depth_stencil_different_formats_different_hash() {
        let make = |fmt| RenderPassInfo {
            color_attachments: vec![],
            depth_stencil: Some(DepthStencilAttachmentInfo {
                format: fmt,
                depth_load_op: AttachmentLoadOp::Clear,
                depth_store_op: AttachmentStoreOp::DontCare,
                stencil_load_op: AttachmentLoadOp::DontCare,
                stencil_store_op: AttachmentStoreOp::DontCare,
            }),
            samples: vk::SampleCountFlags::TYPE_1,
        };
        assert_ne!(
            RenderPassCache::compatible_hash(&make(vk::Format::D32_SFLOAT)),
            RenderPassCache::compatible_hash(&make(vk::Format::D24_UNORM_S8_UINT))
        );
    }
}
