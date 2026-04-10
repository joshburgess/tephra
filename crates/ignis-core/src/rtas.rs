//! Ray tracing acceleration structure types and helpers.
//!
//! Provides types for creating and managing bottom-level (BLAS) and top-level
//! (TLAS) acceleration structures. Requires `VK_KHR_acceleration_structure`.

use ash::vk;

/// Type of acceleration structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelerationStructureType {
    /// Bottom-level acceleration structure (geometry).
    BottomLevel,
    /// Top-level acceleration structure (instances).
    TopLevel,
}

impl From<AccelerationStructureType> for vk::AccelerationStructureTypeKHR {
    fn from(ty: AccelerationStructureType) -> Self {
        match ty {
            AccelerationStructureType::BottomLevel => {
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL
            }
            AccelerationStructureType::TopLevel => vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        }
    }
}

/// Build mode for acceleration structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelerationStructureBuildMode {
    /// Full build from scratch.
    Build,
    /// Update an existing acceleration structure in place.
    Update,
}

impl From<AccelerationStructureBuildMode> for vk::BuildAccelerationStructureModeKHR {
    fn from(mode: AccelerationStructureBuildMode) -> Self {
        match mode {
            AccelerationStructureBuildMode::Build => vk::BuildAccelerationStructureModeKHR::BUILD,
            AccelerationStructureBuildMode::Update => vk::BuildAccelerationStructureModeKHR::UPDATE,
        }
    }
}

/// Flags controlling acceleration structure build behavior.
#[derive(Debug, Clone, Copy)]
pub struct AccelerationStructureBuildFlags {
    /// Allow the structure to be updated after initial build.
    pub allow_update: bool,
    /// Allow the structure to be compacted after build.
    pub allow_compaction: bool,
    /// Prefer faster trace performance over build time.
    pub prefer_fast_trace: bool,
    /// Prefer faster build time over trace performance.
    pub prefer_fast_build: bool,
    /// Minimize memory usage.
    pub low_memory: bool,
}

impl Default for AccelerationStructureBuildFlags {
    fn default() -> Self {
        Self {
            allow_update: false,
            allow_compaction: false,
            prefer_fast_trace: true,
            prefer_fast_build: false,
            low_memory: false,
        }
    }
}

impl From<AccelerationStructureBuildFlags> for vk::BuildAccelerationStructureFlagsKHR {
    fn from(flags: AccelerationStructureBuildFlags) -> Self {
        let mut vk_flags = vk::BuildAccelerationStructureFlagsKHR::empty();
        if flags.allow_update {
            vk_flags |= vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE;
        }
        if flags.allow_compaction {
            vk_flags |= vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION;
        }
        if flags.prefer_fast_trace {
            vk_flags |= vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE;
        }
        if flags.prefer_fast_build {
            vk_flags |= vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD;
        }
        if flags.low_memory {
            vk_flags |= vk::BuildAccelerationStructureFlagsKHR::LOW_MEMORY;
        }
        vk_flags
    }
}

/// Build sizes returned by `get_acceleration_structure_build_sizes`.
#[derive(Debug, Clone, Copy)]
pub struct AccelerationStructureBuildSizes {
    /// Size in bytes for the acceleration structure buffer.
    pub acceleration_structure_size: vk::DeviceSize,
    /// Size in bytes for the build scratch buffer.
    pub build_scratch_size: vk::DeviceSize,
    /// Size in bytes for the update scratch buffer.
    pub update_scratch_size: vk::DeviceSize,
}

/// A handle to a Vulkan acceleration structure with its backing buffer.
pub struct AccelerationStructure {
    /// The raw Vulkan acceleration structure handle.
    pub handle: vk::AccelerationStructureKHR,
    /// The backing buffer for this acceleration structure.
    pub buffer: vk::Buffer,
    /// Device address of the acceleration structure.
    pub device_address: vk::DeviceAddress,
    /// Type of this acceleration structure.
    pub structure_type: AccelerationStructureType,
}

impl AccelerationStructure {
    /// Destroy the acceleration structure.
    ///
    /// The caller must ensure the acceleration structure is not in use by the GPU.
    /// The backing buffer must be freed separately.
    pub fn destroy(&mut self, as_device: &ash::khr::acceleration_structure::Device) {
        if self.handle != vk::AccelerationStructureKHR::null() {
            // SAFETY: as_device is valid, handle is valid, GPU is idle.
            unsafe {
                as_device.destroy_acceleration_structure(self.handle, None);
            }
            self.handle = vk::AccelerationStructureKHR::null();
            self.device_address = 0;
        }
    }
}
