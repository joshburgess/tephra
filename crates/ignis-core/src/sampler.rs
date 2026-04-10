//! Sampler cache and stock sampler presets.
//!
//! Provides a hash-and-cache layer for Vulkan samplers and a set of commonly
//! used sampler presets via [`StockSampler`].

use ash::vk;
use rustc_hash::FxHashMap;

/// Commonly used sampler configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StockSampler {
    /// Nearest filtering, clamp to edge.
    NearestClamp,
    /// Nearest filtering, repeat wrapping.
    NearestWrap,
    /// Linear filtering, clamp to edge.
    LinearClamp,
    /// Linear filtering, repeat wrapping.
    LinearWrap,
    /// Trilinear (linear + mipmap linear) filtering, clamp to edge.
    TrilinearClamp,
    /// Trilinear filtering, repeat wrapping.
    TrilinearWrap,
    /// Nearest filtering with depth comparison, for shadow maps.
    NearestShadow,
    /// Linear filtering with depth comparison, for shadow maps.
    LinearShadow,
}

/// Key for caching samplers, derived from `VkSamplerCreateInfo` fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SamplerKey {
    mag_filter: vk::Filter,
    min_filter: vk::Filter,
    mipmap_mode: vk::SamplerMipmapMode,
    address_mode_u: vk::SamplerAddressMode,
    address_mode_v: vk::SamplerAddressMode,
    address_mode_w: vk::SamplerAddressMode,
    anisotropy_enable: bool,
    max_anisotropy_bits: u32,
    compare_enable: bool,
    compare_op: vk::CompareOp,
    min_lod_bits: u32,
    max_lod_bits: u32,
    border_color: vk::BorderColor,
}

impl SamplerKey {
    fn from_create_info(ci: &SamplerCreateInfo) -> Self {
        Self {
            mag_filter: ci.mag_filter,
            min_filter: ci.min_filter,
            mipmap_mode: ci.mipmap_mode,
            address_mode_u: ci.address_mode_u,
            address_mode_v: ci.address_mode_v,
            address_mode_w: ci.address_mode_w,
            anisotropy_enable: ci.anisotropy_enable,
            max_anisotropy_bits: ci.max_anisotropy.to_bits(),
            compare_enable: ci.compare_enable,
            compare_op: ci.compare_op,
            min_lod_bits: ci.min_lod.to_bits(),
            max_lod_bits: ci.max_lod.to_bits(),
            border_color: ci.border_color,
        }
    }
}

/// Parameters for creating or looking up a sampler.
pub struct SamplerCreateInfo {
    /// Magnification filter.
    pub mag_filter: vk::Filter,
    /// Minification filter.
    pub min_filter: vk::Filter,
    /// Mipmap filtering mode.
    pub mipmap_mode: vk::SamplerMipmapMode,
    /// Addressing mode for U coordinates.
    pub address_mode_u: vk::SamplerAddressMode,
    /// Addressing mode for V coordinates.
    pub address_mode_v: vk::SamplerAddressMode,
    /// Addressing mode for W coordinates.
    pub address_mode_w: vk::SamplerAddressMode,
    /// Whether anisotropic filtering is enabled.
    pub anisotropy_enable: bool,
    /// Maximum anisotropy level.
    pub max_anisotropy: f32,
    /// Whether comparison mode is enabled (for shadow samplers).
    pub compare_enable: bool,
    /// Comparison operator.
    pub compare_op: vk::CompareOp,
    /// Minimum LOD clamp.
    pub min_lod: f32,
    /// Maximum LOD clamp.
    pub max_lod: f32,
    /// Border color for clamp-to-border addressing.
    pub border_color: vk::BorderColor,
}

impl Default for SamplerCreateInfo {
    /// Default sampler: linear filtering, clamp-to-edge, no anisotropy.
    fn default() -> Self {
        Self {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            anisotropy_enable: false,
            max_anisotropy: 1.0,
            compare_enable: false,
            compare_op: vk::CompareOp::NEVER,
            min_lod: 0.0,
            max_lod: vk::LOD_CLAMP_NONE,
            border_color: vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
        }
    }
}

impl SamplerCreateInfo {
    /// Set magnification and minification filters.
    pub fn filter(mut self, mag: vk::Filter, min: vk::Filter) -> Self {
        self.mag_filter = mag;
        self.min_filter = min;
        self
    }

    /// Set the mipmap filtering mode.
    pub fn mipmap_mode(mut self, mode: vk::SamplerMipmapMode) -> Self {
        self.mipmap_mode = mode;
        self
    }

    /// Set all three address modes at once.
    pub fn address_mode(mut self, mode: vk::SamplerAddressMode) -> Self {
        self.address_mode_u = mode;
        self.address_mode_v = mode;
        self.address_mode_w = mode;
        self
    }

    /// Enable anisotropic filtering with the given maximum level.
    pub fn anisotropy(mut self, max_anisotropy: f32) -> Self {
        self.anisotropy_enable = true;
        self.max_anisotropy = max_anisotropy;
        self
    }

    /// Enable comparison mode (e.g., for shadow samplers).
    pub fn compare(mut self, op: vk::CompareOp) -> Self {
        self.compare_enable = true;
        self.compare_op = op;
        self
    }

    /// Set the LOD clamp range.
    pub fn lod_range(mut self, min: f32, max: f32) -> Self {
        self.min_lod = min;
        self.max_lod = max;
        self
    }

    /// Set the border color for clamp-to-border addressing.
    pub fn border_color(mut self, color: vk::BorderColor) -> Self {
        self.border_color = color;
        self
    }
}

impl StockSampler {
    fn to_create_info(self) -> SamplerCreateInfo {
        let (mag, min, mip, addr, compare, compare_op) = match self {
            Self::NearestClamp => (
                vk::Filter::NEAREST,
                vk::Filter::NEAREST,
                vk::SamplerMipmapMode::NEAREST,
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
                false,
                vk::CompareOp::NEVER,
            ),
            Self::NearestWrap => (
                vk::Filter::NEAREST,
                vk::Filter::NEAREST,
                vk::SamplerMipmapMode::NEAREST,
                vk::SamplerAddressMode::REPEAT,
                false,
                vk::CompareOp::NEVER,
            ),
            Self::LinearClamp => (
                vk::Filter::LINEAR,
                vk::Filter::LINEAR,
                vk::SamplerMipmapMode::NEAREST,
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
                false,
                vk::CompareOp::NEVER,
            ),
            Self::LinearWrap => (
                vk::Filter::LINEAR,
                vk::Filter::LINEAR,
                vk::SamplerMipmapMode::NEAREST,
                vk::SamplerAddressMode::REPEAT,
                false,
                vk::CompareOp::NEVER,
            ),
            Self::TrilinearClamp => (
                vk::Filter::LINEAR,
                vk::Filter::LINEAR,
                vk::SamplerMipmapMode::LINEAR,
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
                false,
                vk::CompareOp::NEVER,
            ),
            Self::TrilinearWrap => (
                vk::Filter::LINEAR,
                vk::Filter::LINEAR,
                vk::SamplerMipmapMode::LINEAR,
                vk::SamplerAddressMode::REPEAT,
                false,
                vk::CompareOp::NEVER,
            ),
            Self::NearestShadow => (
                vk::Filter::NEAREST,
                vk::Filter::NEAREST,
                vk::SamplerMipmapMode::NEAREST,
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
                true,
                vk::CompareOp::LESS_OR_EQUAL,
            ),
            Self::LinearShadow => (
                vk::Filter::LINEAR,
                vk::Filter::LINEAR,
                vk::SamplerMipmapMode::NEAREST,
                vk::SamplerAddressMode::CLAMP_TO_EDGE,
                true,
                vk::CompareOp::LESS_OR_EQUAL,
            ),
        };

        SamplerCreateInfo {
            mag_filter: mag,
            min_filter: min,
            mipmap_mode: mip,
            address_mode_u: addr,
            address_mode_v: addr,
            address_mode_w: addr,
            anisotropy_enable: false,
            max_anisotropy: 1.0,
            compare_enable: compare,
            compare_op,
            min_lod: 0.0,
            max_lod: vk::LOD_CLAMP_NONE,
            border_color: vk::BorderColor::FLOAT_OPAQUE_BLACK,
        }
    }
}

/// Cache for Vulkan samplers, keyed by their create info.
pub(crate) struct SamplerCache {
    cache: FxHashMap<SamplerKey, vk::Sampler>,
    stock_samplers: [vk::Sampler; 8],
}

impl SamplerCache {
    /// Create a new sampler cache, pre-creating all stock samplers.
    pub fn new(device: &ash::Device) -> Result<Self, vk::Result> {
        let mut cache = FxHashMap::default();
        let stock_variants = [
            StockSampler::NearestClamp,
            StockSampler::NearestWrap,
            StockSampler::LinearClamp,
            StockSampler::LinearWrap,
            StockSampler::TrilinearClamp,
            StockSampler::TrilinearWrap,
            StockSampler::NearestShadow,
            StockSampler::LinearShadow,
        ];

        let mut stock_samplers = [vk::Sampler::null(); 8];

        for (i, &variant) in stock_variants.iter().enumerate() {
            let ci = variant.to_create_info();
            let sampler = Self::create_vk_sampler(device, &ci)?;
            let key = SamplerKey::from_create_info(&ci);
            cache.insert(key, sampler);
            stock_samplers[i] = sampler;
        }

        log::debug!("Pre-created {} stock samplers", stock_variants.len());

        Ok(Self {
            cache,
            stock_samplers,
        })
    }

    /// Look up or create a sampler from a create info.
    pub fn get_or_create(
        &mut self,
        device: &ash::Device,
        info: &SamplerCreateInfo,
    ) -> Result<vk::Sampler, vk::Result> {
        let key = SamplerKey::from_create_info(info);

        if let Some(&sampler) = self.cache.get(&key) {
            return Ok(sampler);
        }

        let sampler = Self::create_vk_sampler(device, info)?;
        self.cache.insert(key, sampler);
        log::debug!("Created new cached sampler (total: {})", self.cache.len());
        Ok(sampler)
    }

    /// Get a pre-created stock sampler.
    pub fn stock(&self, stock: StockSampler) -> vk::Sampler {
        self.stock_samplers[stock as usize]
    }

    /// Destroy all cached samplers.
    pub fn destroy(&mut self, device: &ash::Device) {
        for (_, sampler) in self.cache.drain() {
            // SAFETY: device is valid, sampler is valid, GPU is idle.
            unsafe {
                device.destroy_sampler(sampler, None);
            }
        }
    }

    fn create_vk_sampler(
        device: &ash::Device,
        info: &SamplerCreateInfo,
    ) -> Result<vk::Sampler, vk::Result> {
        let sampler_ci = vk::SamplerCreateInfo::default()
            .mag_filter(info.mag_filter)
            .min_filter(info.min_filter)
            .mipmap_mode(info.mipmap_mode)
            .address_mode_u(info.address_mode_u)
            .address_mode_v(info.address_mode_v)
            .address_mode_w(info.address_mode_w)
            .anisotropy_enable(info.anisotropy_enable)
            .max_anisotropy(info.max_anisotropy)
            .compare_enable(info.compare_enable)
            .compare_op(info.compare_op)
            .min_lod(info.min_lod)
            .max_lod(info.max_lod)
            .border_color(info.border_color);

        // SAFETY: device is valid, sampler_ci is well-formed.
        unsafe { device.create_sampler(&sampler_ci, None) }
    }
}
