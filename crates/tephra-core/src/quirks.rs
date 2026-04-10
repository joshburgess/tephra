//! Driver quirks detection and workarounds.
//!
//! Detects GPU vendor, driver version, and known issues. Applies workarounds
//! for driver bugs and missing features based on the physical device properties.

use ash::vk;

/// Known GPU vendors identified by PCI vendor ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    /// NVIDIA Corporation.
    Nvidia,
    /// Advanced Micro Devices.
    Amd,
    /// Intel Corporation.
    Intel,
    /// Arm (Mali GPUs).
    Arm,
    /// Qualcomm (Adreno GPUs).
    Qualcomm,
    /// Apple (MoltenVK / Apple Silicon).
    Apple,
    /// Samsung (Xclipse GPUs).
    Samsung,
    /// Unknown vendor.
    Unknown(u32),
}

impl GpuVendor {
    /// Identify vendor from PCI vendor ID.
    pub fn from_vendor_id(id: u32) -> Self {
        match id {
            0x10DE => Self::Nvidia,
            0x1002 => Self::Amd,
            0x8086 => Self::Intel,
            0x13B5 => Self::Arm,
            0x5143 => Self::Qualcomm,
            0x106B => Self::Apple,
            0x144D => Self::Samsung,
            other => Self::Unknown(other),
        }
    }

    /// Whether this is a mobile/tile-based deferred rendering GPU.
    pub fn is_tbdr(&self) -> bool {
        matches!(
            self,
            Self::Arm | Self::Qualcomm | Self::Apple | Self::Samsung
        )
    }
}

/// Known driver-specific quirks and workarounds.
///
/// These flags are set during context initialization based on the detected
/// GPU vendor and driver version. Code paths can check these flags to apply
/// appropriate workarounds.
#[derive(Debug, Clone)]
pub struct ImplementationQuirks {
    /// Detected GPU vendor.
    pub vendor: GpuVendor,
    /// Vulkan API version reported by the device.
    pub api_version: u32,
    /// Driver version (vendor-specific encoding).
    pub driver_version: u32,
    /// Device type (discrete, integrated, virtual, CPU, etc.).
    pub device_type: vk::PhysicalDeviceType,
    /// Device name as reported by the driver.
    pub device_name: String,

    /// Whether to emulate VkEvent as a full pipeline barrier.
    ///
    /// Some NVIDIA drivers have broken VkEvent implementation.
    /// When true, `set_event` + `wait_events` should be replaced
    /// with a regular pipeline barrier.
    pub emulate_event_as_barrier: bool,

    /// Whether to force host-cached memory for readbacks.
    ///
    /// Some AMD drivers perform poorly with host-coherent memory
    /// for GPU->CPU readbacks.
    pub force_host_cached_readback: bool,

    /// Whether pipeline cache control is broken.
    ///
    /// Some drivers crash or produce corrupt data with
    /// `VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT`.
    pub broken_pipeline_cache_control: bool,

    /// Whether the driver has broken subgroup operations in certain stages.
    pub broken_subgroup_in_compute: bool,

    /// Recommended frame overlap count for this device.
    ///
    /// TBDR GPUs (mobile, Apple) benefit from 3 frames;
    /// desktop GPUs typically use 2.
    pub recommended_frame_overlap: usize,
}

impl ImplementationQuirks {
    /// Detect quirks from physical device properties.
    pub fn detect(properties: &vk::PhysicalDeviceProperties) -> Self {
        let vendor = GpuVendor::from_vendor_id(properties.vendor_id);
        let api_version = properties.api_version;
        let driver_version = properties.driver_version;
        let device_type = properties.device_type;

        // Extract device name
        let device_name = {
            let raw = &properties.device_name;
            let len = raw.iter().position(|&c| c == 0).unwrap_or(raw.len());
            // SAFETY: device_name is a null-terminated UTF-8 string from the driver.
            let bytes: Vec<u8> = raw[..len].iter().map(|&c| c as u8).collect();
            String::from_utf8_lossy(&bytes).into_owned()
        };

        let mut quirks = Self {
            vendor,
            api_version,
            driver_version,
            device_type,
            device_name,
            emulate_event_as_barrier: false,
            force_host_cached_readback: false,
            broken_pipeline_cache_control: false,
            broken_subgroup_in_compute: false,
            recommended_frame_overlap: if vendor.is_tbdr() { 3 } else { 2 },
        };

        quirks.apply_vendor_workarounds();
        quirks
    }

    fn apply_vendor_workarounds(&mut self) {
        match self.vendor {
            GpuVendor::Nvidia => {
                // Older NVIDIA drivers have buggy VkEvent support
                // Driver version encoding: (major << 22) | (minor << 14) | patch
                let major = self.driver_version >> 22;
                if major < 520 {
                    self.emulate_event_as_barrier = true;
                }
            }
            GpuVendor::Amd => {
                // AMD RDNA drivers prefer host-cached for readbacks
                self.force_host_cached_readback = true;
            }
            GpuVendor::Apple => {
                // MoltenVK doesn't support VkEvent properly
                self.emulate_event_as_barrier = true;
            }
            GpuVendor::Arm | GpuVendor::Qualcomm => {
                // Mobile drivers: conservative settings
                self.emulate_event_as_barrier = true;
            }
            _ => {}
        }

        log::info!(
            "GPU: {} ({:?}, vendor={:?}, driver={:#x}, API={}.{}.{})",
            self.device_name,
            self.device_type,
            self.vendor,
            self.driver_version,
            vk::api_version_major(self.api_version),
            vk::api_version_minor(self.api_version),
            vk::api_version_patch(self.api_version),
        );

        if self.emulate_event_as_barrier {
            log::info!("Quirk: emulating VkEvent as pipeline barrier");
        }
        if self.force_host_cached_readback {
            log::info!("Quirk: forcing host-cached memory for readbacks");
        }
    }
}
