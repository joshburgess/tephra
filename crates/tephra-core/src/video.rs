//! Video decode/encode queue support via `VK_KHR_video_queue`.
//!
//! Provides types and helpers for hardware-accelerated video decoding
//! and encoding using Vulkan video extensions. Supported codecs include
//! H.264 (AVC), H.265 (HEVC), and AV1.
//!
//! # Extension Requirements
//!
//! - `VK_KHR_video_queue` — base video queue support
//! - `VK_KHR_video_decode_queue` — video decoding
//! - `VK_KHR_video_encode_queue` — video encoding
//! - Codec-specific extensions (e.g., `VK_KHR_video_decode_h264`)
//!
//! # Architecture
//!
//! Video operations use a dedicated video queue (if available) or the
//! graphics queue. The workflow is:
//! 1. Query video capabilities and profiles
//! 2. Create a video session with the desired codec and parameters
//! 3. Allocate session memory and DPB (decoded picture buffer) images
//! 4. Record decode/encode commands into a command buffer
//! 5. Submit to the video queue

use ash::vk;

/// Supported video codec types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VideoCodec {
    /// H.264 / AVC.
    H264,
    /// H.265 / HEVC.
    H265,
    /// AV1.
    Av1,
}

impl VideoCodec {
    /// Convert to `VkVideoCodecOperationFlagBitsKHR` for decode.
    pub fn decode_operation(self) -> vk::VideoCodecOperationFlagsKHR {
        match self {
            VideoCodec::H264 => vk::VideoCodecOperationFlagsKHR::DECODE_H264,
            VideoCodec::H265 => vk::VideoCodecOperationFlagsKHR::DECODE_H265,
            VideoCodec::Av1 => vk::VideoCodecOperationFlagsKHR::DECODE_AV1,
        }
    }

    /// Convert to `VkVideoCodecOperationFlagBitsKHR` for encode.
    ///
    /// Note: AV1 encode is not yet available in ash 0.38, so it falls back
    /// to H265 encode. Check driver support before using.
    pub fn encode_operation(self) -> vk::VideoCodecOperationFlagsKHR {
        match self {
            VideoCodec::H264 => vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
            VideoCodec::H265 => vk::VideoCodecOperationFlagsKHR::ENCODE_H265,
            VideoCodec::Av1 => {
                log::warn!("AV1 encode not available in ash 0.38, using H265 as fallback");
                vk::VideoCodecOperationFlagsKHR::ENCODE_H265
            }
        }
    }
}

/// Video session configuration.
#[derive(Debug, Clone)]
pub struct VideoSessionConfig {
    /// The codec to use.
    pub codec: VideoCodec,
    /// Whether this is a decode or encode session.
    pub operation: VideoOperation,
    /// Maximum coded extent (width, height).
    pub max_coded_extent: vk::Extent2D,
    /// Picture format for decoded/encoded frames.
    pub picture_format: vk::Format,
    /// Reference picture format (for DPB).
    pub reference_format: vk::Format,
    /// Maximum number of DPB slots.
    pub max_dpb_slots: u32,
    /// Maximum number of active reference pictures.
    pub max_active_reference_pictures: u32,
}

/// Whether a video session is for decoding or encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoOperation {
    /// Video decoding.
    Decode,
    /// Video encoding.
    Encode,
}

/// Capabilities for a specific video profile.
#[derive(Debug, Clone)]
pub struct VideoCapabilities {
    /// Minimum coded extent supported.
    pub min_coded_extent: vk::Extent2D,
    /// Maximum coded extent supported.
    pub max_coded_extent: vk::Extent2D,
    /// Maximum number of DPB slots.
    pub max_dpb_slots: u32,
    /// Maximum number of active reference pictures.
    pub max_active_reference_pictures: u32,
    /// Supported picture formats.
    pub picture_formats: Vec<vk::Format>,
}

/// Query video decode capabilities for a codec.
///
/// The `video_queue_loader` should be created from the entry and instance
/// at initialization time via `ash::khr::video_queue::Instance::new(entry, instance)`.
///
/// Returns `None` if the codec is not supported for decoding on this device.
pub fn query_decode_capabilities(
    video_queue_loader: &ash::khr::video_queue::Instance,
    physical_device: vk::PhysicalDevice,
    codec: VideoCodec,
) -> Option<VideoCapabilities> {
    let profile_info = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(codec.decode_operation())
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
        .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8);

    let mut caps = vk::VideoCapabilitiesKHR::default();

    // SAFETY: loader, physical_device, and profile_info are valid.
    let result = unsafe {
        (video_queue_loader
            .fp()
            .get_physical_device_video_capabilities_khr)(
            physical_device, &profile_info, &mut caps
        )
    };

    if result != vk::Result::SUCCESS {
        return None;
    }

    Some(VideoCapabilities {
        min_coded_extent: caps.min_coded_extent,
        max_coded_extent: caps.max_coded_extent,
        max_dpb_slots: caps.max_dpb_slots,
        max_active_reference_pictures: caps.max_active_reference_pictures,
        picture_formats: Vec::new(), // Would need additional queries
    })
}

/// Check if a video queue family is available on the device.
pub fn find_video_queue_family(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    operation: VideoOperation,
) -> Option<u32> {
    // SAFETY: instance and physical_device are valid.
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    let required_flag = match operation {
        VideoOperation::Decode => vk::QueueFlags::VIDEO_DECODE_KHR,
        VideoOperation::Encode => vk::QueueFlags::VIDEO_ENCODE_KHR,
    };

    queue_families
        .iter()
        .enumerate()
        .find(|(_, props)| props.queue_flags.contains(required_flag))
        .map(|(index, _)| index as u32)
}

/// List of device extensions required for video decode support.
pub fn decode_device_extensions(codec: VideoCodec) -> Vec<&'static std::ffi::CStr> {
    let mut exts = vec![
        ash::khr::video_queue::NAME,
        ash::khr::video_decode_queue::NAME,
    ];

    match codec {
        VideoCodec::H264 => exts.push(ash::khr::video_decode_h264::NAME),
        VideoCodec::H265 => exts.push(ash::khr::video_decode_h265::NAME),
        VideoCodec::Av1 => exts.push(ash::khr::video_decode_av1::NAME),
    }

    exts
}

/// List of device extensions required for video encode support.
///
/// Note: AV1 encode is not yet available in ash 0.38. Returns `None`
/// for unsupported codecs.
pub fn encode_device_extensions(codec: VideoCodec) -> Option<Vec<&'static std::ffi::CStr>> {
    let mut exts = vec![
        ash::khr::video_queue::NAME,
        ash::khr::video_encode_queue::NAME,
    ];

    match codec {
        VideoCodec::H264 => exts.push(ash::khr::video_encode_h264::NAME),
        VideoCodec::H265 => exts.push(ash::khr::video_encode_h265::NAME),
        VideoCodec::Av1 => {
            log::warn!("AV1 encode extension not available in ash 0.38");
            return None;
        }
    }

    Some(exts)
}
