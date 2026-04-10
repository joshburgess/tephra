//! Texture format utilities and mip chain computation.
//!
//! Provides helpers for querying block sizes, block dimensions, and component
//! counts for all Vulkan formats including compressed formats (BC, ETC2, ASTC).
//! Also computes mip chain layouts for texture uploading.

use ash::vk;

/// Information about a texture format's block encoding.
#[derive(Debug, Clone, Copy)]
pub struct FormatBlockInfo {
    /// Block width in texels (1 for non-compressed formats).
    pub block_width: u32,
    /// Block height in texels (1 for non-compressed formats).
    pub block_height: u32,
    /// Bytes per block (or per texel for non-compressed formats).
    pub block_size: u32,
}

/// Get block encoding info for a Vulkan format.
///
/// Returns block dimensions and size for compressed and uncompressed formats.
pub fn format_block_info(format: vk::Format) -> FormatBlockInfo {
    match format {
        // --- Uncompressed 1-byte ---
        vk::Format::R8_UNORM
        | vk::Format::R8_SNORM
        | vk::Format::R8_UINT
        | vk::Format::R8_SINT
        | vk::Format::R8_SRGB => FormatBlockInfo {
            block_width: 1,
            block_height: 1,
            block_size: 1,
        },

        // --- Uncompressed 2-byte ---
        vk::Format::R8G8_UNORM
        | vk::Format::R8G8_SNORM
        | vk::Format::R8G8_UINT
        | vk::Format::R8G8_SINT
        | vk::Format::R8G8_SRGB
        | vk::Format::R16_UNORM
        | vk::Format::R16_SNORM
        | vk::Format::R16_UINT
        | vk::Format::R16_SINT
        | vk::Format::R16_SFLOAT
        | vk::Format::D16_UNORM
        | vk::Format::R5G6B5_UNORM_PACK16
        | vk::Format::B5G6R5_UNORM_PACK16
        | vk::Format::R4G4B4A4_UNORM_PACK16
        | vk::Format::B4G4R4A4_UNORM_PACK16
        | vk::Format::R5G5B5A1_UNORM_PACK16
        | vk::Format::A1R5G5B5_UNORM_PACK16 => FormatBlockInfo {
            block_width: 1,
            block_height: 1,
            block_size: 2,
        },

        // --- Uncompressed 3-byte ---
        vk::Format::R8G8B8_UNORM
        | vk::Format::R8G8B8_SNORM
        | vk::Format::R8G8B8_UINT
        | vk::Format::R8G8B8_SINT
        | vk::Format::R8G8B8_SRGB
        | vk::Format::B8G8R8_UNORM
        | vk::Format::B8G8R8_SNORM
        | vk::Format::B8G8R8_UINT
        | vk::Format::B8G8R8_SINT
        | vk::Format::B8G8R8_SRGB
        | vk::Format::D16_UNORM_S8_UINT => FormatBlockInfo {
            block_width: 1,
            block_height: 1,
            block_size: 3,
        },

        // --- Uncompressed 4-byte ---
        vk::Format::R8G8B8A8_UNORM
        | vk::Format::R8G8B8A8_SNORM
        | vk::Format::R8G8B8A8_UINT
        | vk::Format::R8G8B8A8_SINT
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::B8G8R8A8_UNORM
        | vk::Format::B8G8R8A8_SNORM
        | vk::Format::B8G8R8A8_UINT
        | vk::Format::B8G8R8A8_SINT
        | vk::Format::B8G8R8A8_SRGB
        | vk::Format::A2R10G10B10_UNORM_PACK32
        | vk::Format::A2B10G10R10_UNORM_PACK32
        | vk::Format::R16G16_UNORM
        | vk::Format::R16G16_SNORM
        | vk::Format::R16G16_UINT
        | vk::Format::R16G16_SINT
        | vk::Format::R16G16_SFLOAT
        | vk::Format::R32_UINT
        | vk::Format::R32_SINT
        | vk::Format::R32_SFLOAT
        | vk::Format::B10G11R11_UFLOAT_PACK32
        | vk::Format::E5B9G9R9_UFLOAT_PACK32
        | vk::Format::D32_SFLOAT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::X8_D24_UNORM_PACK32 => FormatBlockInfo {
            block_width: 1,
            block_height: 1,
            block_size: 4,
        },

        // --- Uncompressed 5-byte ---
        vk::Format::D32_SFLOAT_S8_UINT => FormatBlockInfo {
            block_width: 1,
            block_height: 1,
            block_size: 5,
        },

        // --- Uncompressed 8-byte ---
        vk::Format::R16G16B16A16_UNORM
        | vk::Format::R16G16B16A16_SNORM
        | vk::Format::R16G16B16A16_UINT
        | vk::Format::R16G16B16A16_SINT
        | vk::Format::R16G16B16A16_SFLOAT
        | vk::Format::R32G32_UINT
        | vk::Format::R32G32_SINT
        | vk::Format::R32G32_SFLOAT
        | vk::Format::R64_UINT
        | vk::Format::R64_SINT
        | vk::Format::R64_SFLOAT => FormatBlockInfo {
            block_width: 1,
            block_height: 1,
            block_size: 8,
        },

        // --- Uncompressed 12-byte ---
        vk::Format::R32G32B32_UINT | vk::Format::R32G32B32_SINT | vk::Format::R32G32B32_SFLOAT => {
            FormatBlockInfo {
                block_width: 1,
                block_height: 1,
                block_size: 12,
            }
        }

        // --- Uncompressed 16-byte ---
        vk::Format::R32G32B32A32_UINT
        | vk::Format::R32G32B32A32_SINT
        | vk::Format::R32G32B32A32_SFLOAT
        | vk::Format::R64G64_UINT
        | vk::Format::R64G64_SINT
        | vk::Format::R64G64_SFLOAT => FormatBlockInfo {
            block_width: 1,
            block_height: 1,
            block_size: 16,
        },

        // --- BC (S3TC / RGTC / BPTC) — 4x4 blocks ---
        vk::Format::BC1_RGB_UNORM_BLOCK
        | vk::Format::BC1_RGB_SRGB_BLOCK
        | vk::Format::BC1_RGBA_UNORM_BLOCK
        | vk::Format::BC1_RGBA_SRGB_BLOCK
        | vk::Format::BC4_UNORM_BLOCK
        | vk::Format::BC4_SNORM_BLOCK => FormatBlockInfo {
            block_width: 4,
            block_height: 4,
            block_size: 8,
        },
        vk::Format::BC2_UNORM_BLOCK
        | vk::Format::BC2_SRGB_BLOCK
        | vk::Format::BC3_UNORM_BLOCK
        | vk::Format::BC3_SRGB_BLOCK
        | vk::Format::BC5_UNORM_BLOCK
        | vk::Format::BC5_SNORM_BLOCK
        | vk::Format::BC6H_UFLOAT_BLOCK
        | vk::Format::BC6H_SFLOAT_BLOCK
        | vk::Format::BC7_UNORM_BLOCK
        | vk::Format::BC7_SRGB_BLOCK => FormatBlockInfo {
            block_width: 4,
            block_height: 4,
            block_size: 16,
        },

        // --- ETC2 — 4x4 blocks ---
        vk::Format::ETC2_R8G8B8_UNORM_BLOCK
        | vk::Format::ETC2_R8G8B8_SRGB_BLOCK
        | vk::Format::ETC2_R8G8B8A1_UNORM_BLOCK
        | vk::Format::ETC2_R8G8B8A1_SRGB_BLOCK
        | vk::Format::EAC_R11_UNORM_BLOCK
        | vk::Format::EAC_R11_SNORM_BLOCK => FormatBlockInfo {
            block_width: 4,
            block_height: 4,
            block_size: 8,
        },
        vk::Format::ETC2_R8G8B8A8_UNORM_BLOCK
        | vk::Format::ETC2_R8G8B8A8_SRGB_BLOCK
        | vk::Format::EAC_R11G11_UNORM_BLOCK
        | vk::Format::EAC_R11G11_SNORM_BLOCK => FormatBlockInfo {
            block_width: 4,
            block_height: 4,
            block_size: 16,
        },

        // --- ASTC — variable block sizes, all 16 bytes per block ---
        vk::Format::ASTC_4X4_UNORM_BLOCK | vk::Format::ASTC_4X4_SRGB_BLOCK => FormatBlockInfo {
            block_width: 4,
            block_height: 4,
            block_size: 16,
        },
        vk::Format::ASTC_5X4_UNORM_BLOCK | vk::Format::ASTC_5X4_SRGB_BLOCK => FormatBlockInfo {
            block_width: 5,
            block_height: 4,
            block_size: 16,
        },
        vk::Format::ASTC_5X5_UNORM_BLOCK | vk::Format::ASTC_5X5_SRGB_BLOCK => FormatBlockInfo {
            block_width: 5,
            block_height: 5,
            block_size: 16,
        },
        vk::Format::ASTC_6X5_UNORM_BLOCK | vk::Format::ASTC_6X5_SRGB_BLOCK => FormatBlockInfo {
            block_width: 6,
            block_height: 5,
            block_size: 16,
        },
        vk::Format::ASTC_6X6_UNORM_BLOCK | vk::Format::ASTC_6X6_SRGB_BLOCK => FormatBlockInfo {
            block_width: 6,
            block_height: 6,
            block_size: 16,
        },
        vk::Format::ASTC_8X5_UNORM_BLOCK | vk::Format::ASTC_8X5_SRGB_BLOCK => FormatBlockInfo {
            block_width: 8,
            block_height: 5,
            block_size: 16,
        },
        vk::Format::ASTC_8X6_UNORM_BLOCK | vk::Format::ASTC_8X6_SRGB_BLOCK => FormatBlockInfo {
            block_width: 8,
            block_height: 6,
            block_size: 16,
        },
        vk::Format::ASTC_8X8_UNORM_BLOCK | vk::Format::ASTC_8X8_SRGB_BLOCK => FormatBlockInfo {
            block_width: 8,
            block_height: 8,
            block_size: 16,
        },
        vk::Format::ASTC_10X5_UNORM_BLOCK | vk::Format::ASTC_10X5_SRGB_BLOCK => FormatBlockInfo {
            block_width: 10,
            block_height: 5,
            block_size: 16,
        },
        vk::Format::ASTC_10X6_UNORM_BLOCK | vk::Format::ASTC_10X6_SRGB_BLOCK => FormatBlockInfo {
            block_width: 10,
            block_height: 6,
            block_size: 16,
        },
        vk::Format::ASTC_10X8_UNORM_BLOCK | vk::Format::ASTC_10X8_SRGB_BLOCK => FormatBlockInfo {
            block_width: 10,
            block_height: 8,
            block_size: 16,
        },
        vk::Format::ASTC_10X10_UNORM_BLOCK | vk::Format::ASTC_10X10_SRGB_BLOCK => FormatBlockInfo {
            block_width: 10,
            block_height: 10,
            block_size: 16,
        },
        vk::Format::ASTC_12X10_UNORM_BLOCK | vk::Format::ASTC_12X10_SRGB_BLOCK => FormatBlockInfo {
            block_width: 12,
            block_height: 10,
            block_size: 16,
        },
        vk::Format::ASTC_12X12_UNORM_BLOCK | vk::Format::ASTC_12X12_SRGB_BLOCK => FormatBlockInfo {
            block_width: 12,
            block_height: 12,
            block_size: 16,
        },

        // Unknown / unsupported — assume 1x1, 4 bytes
        _ => FormatBlockInfo {
            block_width: 1,
            block_height: 1,
            block_size: 4,
        },
    }
}

/// Whether a format is a compressed block format.
pub fn is_compressed(format: vk::Format) -> bool {
    let info = format_block_info(format);
    info.block_width > 1 || info.block_height > 1
}

/// Whether a format has a depth component.
pub fn is_depth_format(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D16_UNORM
            | vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT
            | vk::Format::D32_SFLOAT_S8_UINT
            | vk::Format::X8_D24_UNORM_PACK32
    )
}

/// Whether a format has a stencil component.
pub fn is_stencil_format(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::S8_UINT
            | vk::Format::D16_UNORM_S8_UINT
            | vk::Format::D24_UNORM_S8_UINT
            | vk::Format::D32_SFLOAT_S8_UINT
    )
}

/// Whether a format has both depth and stencil components.
pub fn is_depth_stencil_format(format: vk::Format) -> bool {
    is_depth_format(format) && is_stencil_format(format)
}

/// Description of a single mip level in a texture layout.
#[derive(Debug, Clone, Copy)]
pub struct MipLevelLayout {
    /// Width in texels at this mip level.
    pub width: u32,
    /// Height in texels at this mip level.
    pub height: u32,
    /// Depth in texels at this mip level (1 for 2D textures).
    pub depth: u32,
    /// Width in blocks (= ceil(width / block_width)).
    pub block_count_x: u32,
    /// Height in blocks (= ceil(height / block_height)).
    pub block_count_y: u32,
    /// Size in bytes of this mip level (one layer).
    pub size: vk::DeviceSize,
    /// Byte offset from the start of the texture data.
    pub offset: vk::DeviceSize,
}

/// Full mip chain layout for a texture.
#[derive(Debug, Clone)]
pub struct TextureFormatLayout {
    /// The texture format.
    pub format: vk::Format,
    /// Block encoding info.
    pub block_info: FormatBlockInfo,
    /// Number of array layers.
    pub array_layers: u32,
    /// Per-mip-level layout info.
    pub mip_levels: Vec<MipLevelLayout>,
    /// Total size in bytes (all mips, one layer).
    pub total_size: vk::DeviceSize,
}

impl TextureFormatLayout {
    /// Compute the mip chain layout for a 2D texture.
    pub fn new_2d(
        format: vk::Format,
        width: u32,
        height: u32,
        mip_levels: u32,
        array_layers: u32,
    ) -> Self {
        Self::new_3d(format, width, height, 1, mip_levels, array_layers)
    }

    /// Compute the mip chain layout for a 3D texture.
    pub fn new_3d(
        format: vk::Format,
        width: u32,
        height: u32,
        depth: u32,
        mip_levels: u32,
        array_layers: u32,
    ) -> Self {
        let block_info = format_block_info(format);
        let mut levels = Vec::with_capacity(mip_levels as usize);
        let mut offset: vk::DeviceSize = 0;

        for mip in 0..mip_levels {
            let w = (width >> mip).max(1);
            let h = (height >> mip).max(1);
            let d = (depth >> mip).max(1);

            let bx = w.div_ceil(block_info.block_width);
            let by = h.div_ceil(block_info.block_height);

            let size = (bx as vk::DeviceSize)
                * (by as vk::DeviceSize)
                * (d as vk::DeviceSize)
                * (block_info.block_size as vk::DeviceSize);

            levels.push(MipLevelLayout {
                width: w,
                height: h,
                depth: d,
                block_count_x: bx,
                block_count_y: by,
                size,
                offset,
            });

            offset += size;
        }

        Self {
            format,
            block_info,
            array_layers,
            mip_levels: levels,
            total_size: offset,
        }
    }

    /// Compute the maximum number of mip levels for the given dimensions.
    pub fn max_mip_levels(width: u32, height: u32, depth: u32) -> u32 {
        let max_dim = width.max(height).max(depth);
        if max_dim == 0 {
            return 0;
        }
        32 - max_dim.leading_zeros()
    }

    /// Row pitch in bytes for a given mip level (block_count_x * block_size).
    pub fn row_pitch(&self, mip: u32) -> vk::DeviceSize {
        let level = &self.mip_levels[mip as usize];
        level.block_count_x as vk::DeviceSize * self.block_info.block_size as vk::DeviceSize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba8_block_info() {
        let info = format_block_info(vk::Format::R8G8B8A8_UNORM);
        assert_eq!(info.block_width, 1);
        assert_eq!(info.block_height, 1);
        assert_eq!(info.block_size, 4);
    }

    #[test]
    fn bc7_block_info() {
        let info = format_block_info(vk::Format::BC7_UNORM_BLOCK);
        assert_eq!(info.block_width, 4);
        assert_eq!(info.block_height, 4);
        assert_eq!(info.block_size, 16);
    }

    #[test]
    fn astc_8x8_block_info() {
        let info = format_block_info(vk::Format::ASTC_8X8_UNORM_BLOCK);
        assert_eq!(info.block_width, 8);
        assert_eq!(info.block_height, 8);
        assert_eq!(info.block_size, 16);
    }

    #[test]
    fn is_compressed_formats() {
        assert!(!is_compressed(vk::Format::R8G8B8A8_UNORM));
        assert!(is_compressed(vk::Format::BC7_UNORM_BLOCK));
        assert!(is_compressed(vk::Format::ETC2_R8G8B8_UNORM_BLOCK));
        assert!(is_compressed(vk::Format::ASTC_4X4_UNORM_BLOCK));
    }

    #[test]
    fn depth_format_detection() {
        assert!(is_depth_format(vk::Format::D32_SFLOAT));
        assert!(is_depth_format(vk::Format::D24_UNORM_S8_UINT));
        assert!(!is_depth_format(vk::Format::R8G8B8A8_UNORM));
    }

    #[test]
    fn depth_stencil_detection() {
        assert!(is_depth_stencil_format(vk::Format::D24_UNORM_S8_UINT));
        assert!(is_depth_stencil_format(vk::Format::D32_SFLOAT_S8_UINT));
        assert!(!is_depth_stencil_format(vk::Format::D32_SFLOAT));
    }

    #[test]
    fn max_mip_levels_power_of_two() {
        assert_eq!(TextureFormatLayout::max_mip_levels(256, 256, 1), 9);
        assert_eq!(TextureFormatLayout::max_mip_levels(1024, 1024, 1), 11);
        assert_eq!(TextureFormatLayout::max_mip_levels(1, 1, 1), 1);
    }

    #[test]
    fn max_mip_levels_non_power_of_two() {
        assert_eq!(TextureFormatLayout::max_mip_levels(300, 200, 1), 9);
    }

    #[test]
    fn mip_chain_layout_256x256_rgba8() {
        let layout = TextureFormatLayout::new_2d(vk::Format::R8G8B8A8_UNORM, 256, 256, 9, 1);
        assert_eq!(layout.mip_levels.len(), 9);
        assert_eq!(layout.mip_levels[0].width, 256);
        assert_eq!(layout.mip_levels[0].height, 256);
        assert_eq!(layout.mip_levels[0].size, 256 * 256 * 4);
        assert_eq!(layout.mip_levels[1].width, 128);
        assert_eq!(layout.mip_levels[1].height, 128);
        assert_eq!(layout.mip_levels[8].width, 1);
        assert_eq!(layout.mip_levels[8].height, 1);
        assert_eq!(layout.mip_levels[8].size, 4);
    }

    #[test]
    fn mip_chain_layout_compressed() {
        let layout = TextureFormatLayout::new_2d(vk::Format::BC7_UNORM_BLOCK, 256, 256, 9, 1);
        assert_eq!(layout.mip_levels[0].block_count_x, 64);
        assert_eq!(layout.mip_levels[0].block_count_y, 64);
        assert_eq!(layout.mip_levels[0].size, 64 * 64 * 16);
        // 2x2 mip: ceil(2/4) = 1 block in each direction
        assert_eq!(layout.mip_levels[7].width, 2);
        assert_eq!(layout.mip_levels[7].block_count_x, 1);
        assert_eq!(layout.mip_levels[7].block_count_y, 1);
    }

    #[test]
    fn offsets_are_contiguous() {
        let layout = TextureFormatLayout::new_2d(vk::Format::R8G8B8A8_UNORM, 128, 128, 8, 1);
        let mut expected_offset = 0u64;
        for level in &layout.mip_levels {
            assert_eq!(level.offset, expected_offset);
            expected_offset += level.size;
        }
        assert_eq!(layout.total_size, expected_offset);
    }

    #[test]
    fn row_pitch() {
        let layout = TextureFormatLayout::new_2d(vk::Format::R8G8B8A8_UNORM, 256, 256, 1, 1);
        assert_eq!(layout.row_pitch(0), 256 * 4);
    }
}
