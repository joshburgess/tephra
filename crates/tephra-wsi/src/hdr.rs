//! HDR output and advanced surface format selection.
//!
//! Provides types for requesting HDR surface formats (HDR10, scRGB, etc.)
//! and setting HDR metadata via `VK_EXT_hdr_metadata`.

use ash::vk;

/// Desired backbuffer color space / transfer function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackbufferFormat {
    /// Standard sRGB (8-bit per channel, non-linear).
    /// Prefers `B8G8R8A8_SRGB` / `R8G8B8A8_SRGB` with `SRGB_NONLINEAR`.
    Srgb,

    /// Unorm sRGB (8-bit per channel, linear in shader, sRGB color space).
    /// Prefers `B8G8R8A8_UNORM` with `SRGB_NONLINEAR`.
    SrgbUnorm,

    /// HDR10 (10-bit per channel, PQ/ST.2084 transfer).
    /// Prefers `A2B10G10R10_UNORM_PACK32` with `HDR10_ST2084_EXT`.
    Hdr10,

    /// scRGB (16-bit float per channel, linear).
    /// Prefers `R16G16B16A16_SFLOAT` with `EXTENDED_SRGB_LINEAR_EXT`.
    ScRgb,

    /// Dolby Vision (if supported).
    /// Prefers `R16G16B16A16_SFLOAT` with `DOLBYVISION_EXT`.
    DolbyVision,
}

impl BackbufferFormat {
    /// Select the best matching surface format from the available formats.
    ///
    /// Returns the preferred format if available, otherwise falls back to
    /// standard sRGB, then to the first available format.
    pub fn select(self, available: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
        let preferences = self.preferred_formats();

        // Try each preference in order
        for (format, color_space) in &preferences {
            for avail in available {
                if avail.format == *format && avail.color_space == *color_space {
                    return *avail;
                }
            }
        }

        // Fall back to sRGB if we requested HDR but it's not available
        if self != BackbufferFormat::Srgb {
            log::warn!(
                "Requested {:?} format not available, falling back to sRGB",
                self
            );
            return BackbufferFormat::Srgb.select(available);
        }

        // Last resort
        available[0]
    }

    /// Returns ordered list of (format, color_space) preferences.
    fn preferred_formats(self) -> Vec<(vk::Format, vk::ColorSpaceKHR)> {
        match self {
            BackbufferFormat::Srgb => vec![
                (vk::Format::B8G8R8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR),
                (vk::Format::R8G8B8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR),
            ],
            BackbufferFormat::SrgbUnorm => vec![
                (
                    vk::Format::B8G8R8A8_UNORM,
                    vk::ColorSpaceKHR::SRGB_NONLINEAR,
                ),
                (
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ColorSpaceKHR::SRGB_NONLINEAR,
                ),
            ],
            BackbufferFormat::Hdr10 => vec![
                (
                    vk::Format::A2B10G10R10_UNORM_PACK32,
                    vk::ColorSpaceKHR::HDR10_ST2084_EXT,
                ),
                (
                    vk::Format::A2R10G10B10_UNORM_PACK32,
                    vk::ColorSpaceKHR::HDR10_ST2084_EXT,
                ),
            ],
            BackbufferFormat::ScRgb => vec![(
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
            )],
            BackbufferFormat::DolbyVision => vec![(
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ColorSpaceKHR::DOLBYVISION_EXT,
            )],
        }
    }

    /// Returns `true` if this format is an HDR format.
    pub fn is_hdr(self) -> bool {
        matches!(
            self,
            BackbufferFormat::Hdr10 | BackbufferFormat::ScRgb | BackbufferFormat::DolbyVision
        )
    }
}

/// HDR display metadata (SMPTE ST 2086 mastering display + CTA-861.3 content light level).
///
/// Set via `VK_EXT_hdr_metadata`. Values use the same units as the Vulkan spec.
#[derive(Debug, Clone, Copy)]
pub struct HdrMetadata {
    /// Red primary chromaticity (CIE 1931 xy).
    pub display_primary_red: (f32, f32),
    /// Green primary chromaticity (CIE 1931 xy).
    pub display_primary_green: (f32, f32),
    /// Blue primary chromaticity (CIE 1931 xy).
    pub display_primary_blue: (f32, f32),
    /// White point chromaticity (CIE 1931 xy).
    pub white_point: (f32, f32),
    /// Maximum luminance of the mastering display in nits.
    pub max_luminance: f32,
    /// Minimum luminance of the mastering display in nits.
    pub min_luminance: f32,
    /// Maximum content light level in nits.
    pub max_content_light_level: f32,
    /// Maximum frame-average light level in nits.
    pub max_frame_average_light_level: f32,
}

impl HdrMetadata {
    /// Standard BT.2020 / Display P3 mastering display metadata.
    ///
    /// Typical HDR10 mastering display with 1000 nit peak brightness.
    pub fn bt2020_1000nit() -> Self {
        Self {
            display_primary_red: (0.708, 0.292),
            display_primary_green: (0.170, 0.797),
            display_primary_blue: (0.131, 0.046),
            white_point: (0.3127, 0.3290),
            max_luminance: 1000.0,
            min_luminance: 0.001,
            max_content_light_level: 1000.0,
            max_frame_average_light_level: 400.0,
        }
    }

    /// Convert to the Vulkan `VkHdrMetadataEXT` structure.
    pub fn to_vk(self) -> vk::HdrMetadataEXT<'static> {
        vk::HdrMetadataEXT::default()
            .display_primary_red(vk::XYColorEXT {
                x: self.display_primary_red.0,
                y: self.display_primary_red.1,
            })
            .display_primary_green(vk::XYColorEXT {
                x: self.display_primary_green.0,
                y: self.display_primary_green.1,
            })
            .display_primary_blue(vk::XYColorEXT {
                x: self.display_primary_blue.0,
                y: self.display_primary_blue.1,
            })
            .white_point(vk::XYColorEXT {
                x: self.white_point.0,
                y: self.white_point.1,
            })
            .max_luminance(self.max_luminance)
            .min_luminance(self.min_luminance)
            .max_content_light_level(self.max_content_light_level)
            .max_frame_average_light_level(self.max_frame_average_light_level)
    }
}

/// Set HDR metadata on swapchain(s).
///
/// Requires `VK_EXT_hdr_metadata` to be enabled. The `hdr_metadata_loader`
/// should be created from the instance and device at initialization time
/// via `ash::ext::hdr_metadata::Device::new(instance, device)`.
pub fn set_hdr_metadata(
    hdr_metadata_loader: &ash::ext::hdr_metadata::Device,
    swapchains: &[vk::SwapchainKHR],
    metadata: &HdrMetadata,
) {
    let vk_metadata = metadata.to_vk();
    let metadatas = vec![vk_metadata; swapchains.len()];

    // SAFETY: loader is valid, swapchains are valid, metadata is well-formed.
    unsafe {
        (hdr_metadata_loader.fp().set_hdr_metadata_ext)(
            hdr_metadata_loader.device(),
            swapchains.len() as u32,
            swapchains.as_ptr(),
            metadatas.as_ptr(),
        );
    }
}

/// Preferred present mode selection with priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PresentModePriority {
    /// Low latency without tearing (Mailbox > Fifo).
    LowLatency,
    /// Lowest latency, allows tearing (Immediate > Mailbox > Fifo).
    LowestLatency,
    /// Power efficient, guaranteed no tearing (Fifo only).
    PowerSaving,
    /// Adaptive sync (FifoRelaxed > Fifo).
    AdaptiveSync,
}

impl PresentModePriority {
    /// Select the best present mode from available modes.
    pub fn select(self, available: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
        let preference = self.preference_order();
        for mode in &preference {
            if available.contains(mode) {
                return *mode;
            }
        }
        // FIFO is always available per spec
        vk::PresentModeKHR::FIFO
    }

    fn preference_order(self) -> Vec<vk::PresentModeKHR> {
        match self {
            PresentModePriority::LowLatency => {
                vec![vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::FIFO]
            }
            PresentModePriority::LowestLatency => vec![
                vk::PresentModeKHR::IMMEDIATE,
                vk::PresentModeKHR::MAILBOX,
                vk::PresentModeKHR::FIFO,
            ],
            PresentModePriority::PowerSaving => vec![vk::PresentModeKHR::FIFO],
            PresentModePriority::AdaptiveSync => {
                vec![vk::PresentModeKHR::FIFO_RELAXED, vk::PresentModeKHR::FIFO]
            }
        }
    }
}
