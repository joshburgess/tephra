//! Physical resource allocation for compiled render graphs.
//!
//! Allocates actual Vulkan images and views for virtual resources declared
//! in the render graph. The backbuffer slot is filled externally with the
//! swapchain image each frame.

use ash::vk;
use log::debug;

use ignis_core::device::{Device, DeviceError};
use ignis_core::handles::ImageHandle;
use ignis_core::image::ImageCreateInfo;
use ignis_core::memory::ImageDomain;

use crate::compile::CompiledGraph;
use crate::pass::AccessType;
use crate::resource::{ResourceHandle, ResourceInfo};

/// Physical images and views for a compiled render graph.
///
/// Created once (or on swapchain resize) via [`PhysicalResources::allocate`],
/// then updated per-frame with [`set_backbuffer`](PhysicalResources::set_backbuffer)
/// before passing to [`GraphExecutor::record`](crate::execute::GraphExecutor::record).
pub struct PhysicalResources {
    /// Raw images indexed by resource handle.
    images: Vec<vk::Image>,
    /// Default views indexed by resource handle.
    views: Vec<vk::ImageView>,
    /// Resolved image formats indexed by resource handle.
    formats: Vec<vk::Format>,
    /// Resolved image extents indexed by resource handle.
    extents: Vec<vk::Extent2D>,
    /// Owned image handles for cleanup (excludes backbuffer).
    owned: Vec<(usize, ImageHandle)>,
    /// Index of the backbuffer resource, if any.
    backbuffer_index: Option<usize>,
}

impl PhysicalResources {
    /// Allocate physical images for all attachment resources in the graph.
    ///
    /// Buffer resources and the backbuffer slot are left as null handles.
    /// Call [`set_backbuffer`](PhysicalResources::set_backbuffer) each frame
    /// to inject the swapchain image before recording.
    pub fn allocate(
        device: &mut Device,
        graph: &CompiledGraph,
        swapchain_extent: vk::Extent2D,
        swapchain_format: vk::Format,
    ) -> Result<Self, DeviceError> {
        let num_resources = graph.resources.len();
        let mut images = vec![vk::Image::null(); num_resources];
        let mut views = vec![vk::ImageView::null(); num_resources];
        let mut formats = vec![vk::Format::UNDEFINED; num_resources];
        let mut extents = vec![
            vk::Extent2D {
                width: 0,
                height: 0
            };
            num_resources
        ];
        let mut owned = Vec::new();
        let backbuffer_index = graph.backbuffer.map(|h| h.index as usize);

        let usage_flags = compute_usage_flags(graph, num_resources);

        for (i, res) in graph.resources.iter().enumerate() {
            // Backbuffer is externally provided via set_backbuffer().
            if Some(i) == backbuffer_index {
                formats[i] = swapchain_format;
                extents[i] = swapchain_extent;
                debug!(
                    "Resource [{}] \"{}\" is backbuffer — skipping allocation",
                    i, res.name
                );
                continue;
            }

            match &res.info {
                ResourceInfo::Attachment(info) => {
                    let extent = info.resolve_extent(swapchain_extent);
                    let usage = usage_flags[i];

                    let create_info = ImageCreateInfo {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                        format: info.format,
                        usage,
                        mip_levels: 1,
                        array_layers: 1,
                        samples: info.samples,
                        image_type: vk::ImageType::TYPE_2D,
                        initial_layout: vk::ImageLayout::UNDEFINED,
                        domain: ImageDomain::Physical,
                    };

                    let handle = device.create_image(&create_info)?;
                    images[i] = handle.raw();
                    views[i] = handle.default_view();
                    formats[i] = info.format;
                    extents[i] = extent;

                    debug!(
                        "Allocated resource [{}] \"{}\" {}x{} {:?} usage={:?}",
                        i, res.name, extent.width, extent.height, info.format, usage
                    );

                    owned.push((i, handle));
                }
                ResourceInfo::Buffer(_) => {
                    debug!(
                        "Resource [{}] \"{}\" is a buffer — skipping image allocation",
                        i, res.name
                    );
                }
            }
        }

        debug!(
            "Allocated {} physical images for render graph ({} total resources)",
            owned.len(),
            num_resources
        );

        Ok(Self {
            images,
            views,
            formats,
            extents,
            owned,
            backbuffer_index,
        })
    }

    /// Set the backbuffer image and view for this frame.
    ///
    /// Must be called each frame before
    /// [`GraphExecutor::record`](crate::execute::GraphExecutor::record).
    pub fn set_backbuffer(&mut self, image: vk::Image, view: vk::ImageView) {
        if let Some(idx) = self.backbuffer_index {
            self.images[idx] = image;
            self.views[idx] = view;
        }
    }

    /// Raw image handles indexed by resource handle.
    pub fn images(&self) -> &[vk::Image] {
        &self.images
    }

    /// Image views indexed by resource handle.
    pub fn views(&self) -> &[vk::ImageView] {
        &self.views
    }

    /// Image formats indexed by resource handle.
    pub fn formats(&self) -> &[vk::Format] {
        &self.formats
    }

    /// Resolved extents indexed by resource handle.
    pub fn extents(&self) -> &[vk::Extent2D] {
        &self.extents
    }

    /// The image view for a specific resource.
    pub fn view(&self, handle: ResourceHandle) -> vk::ImageView {
        self.views[handle.index as usize]
    }

    /// The raw image for a specific resource.
    pub fn image(&self, handle: ResourceHandle) -> vk::Image {
        self.images[handle.index as usize]
    }

    /// The format for a specific resource.
    pub fn format(&self, handle: ResourceHandle) -> vk::Format {
        self.formats[handle.index as usize]
    }

    /// The resolved extent for a specific resource.
    pub fn extent(&self, handle: ResourceHandle) -> vk::Extent2D {
        self.extents[handle.index as usize]
    }

    /// Destroy all owned images. Call after the GPU is idle.
    pub fn destroy(self, device: &mut Device) {
        for (_, handle) in self.owned {
            device.destroy_image(handle);
        }
    }
}

/// Compute image usage flags for each resource based on how passes access them.
fn compute_usage_flags(graph: &CompiledGraph, num_resources: usize) -> Vec<vk::ImageUsageFlags> {
    let mut flags = vec![vk::ImageUsageFlags::empty(); num_resources];

    for pass in &graph.passes {
        for access in &pass.accesses {
            let idx = access.resource.index as usize;
            flags[idx] |= access_type_to_usage(access.access_type);
        }
    }

    flags
}

/// Map an access type to the required image usage flag.
fn access_type_to_usage(access: AccessType) -> vk::ImageUsageFlags {
    match access {
        AccessType::ColorOutput | AccessType::ColorInput => vk::ImageUsageFlags::COLOR_ATTACHMENT,
        AccessType::DepthStencilOutput | AccessType::DepthStencilInput => {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        }
        AccessType::TextureInput => vk::ImageUsageFlags::SAMPLED,
        AccessType::StorageRead | AccessType::StorageWrite => vk::ImageUsageFlags::STORAGE,
        AccessType::AttachmentInput => vk::ImageUsageFlags::INPUT_ATTACHMENT,
    }
}
