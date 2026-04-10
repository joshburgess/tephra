//! External memory and semaphore interop.
//!
//! Provides types and helpers for sharing resources between processes or APIs
//! (e.g., video decode, D3D12 interop, DMA-BUF for Wayland). Requires
//! `VK_KHR_external_memory_fd` / `VK_KHR_external_semaphore_fd` (POSIX) or
//! their Windows equivalents.

use ash::vk;

/// Handle type for external memory or semaphore resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalHandleType {
    /// POSIX file descriptor (DMA-BUF, sync fd).
    OpaqueFd,
    /// DMA-BUF file descriptor (Linux).
    DmaBuf,
    /// Windows NT handle.
    OpaqueWin32,
    /// Windows KMT handle (legacy).
    OpaqueWin32Kmt,
}

impl From<ExternalHandleType> for vk::ExternalMemoryHandleTypeFlags {
    fn from(ty: ExternalHandleType) -> Self {
        match ty {
            ExternalHandleType::OpaqueFd => vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD,
            ExternalHandleType::DmaBuf => vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
            ExternalHandleType::OpaqueWin32 => vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32,
            ExternalHandleType::OpaqueWin32Kmt => {
                vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KMT
            }
        }
    }
}

impl From<ExternalHandleType> for vk::ExternalSemaphoreHandleTypeFlags {
    fn from(ty: ExternalHandleType) -> Self {
        match ty {
            ExternalHandleType::OpaqueFd => vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD,
            ExternalHandleType::DmaBuf => {
                // DMA-BUF is memory-only; fall back to opaque fd for semaphores
                vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD
            }
            ExternalHandleType::OpaqueWin32 => {
                vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32
            }
            ExternalHandleType::OpaqueWin32Kmt => {
                vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32_KMT
            }
        }
    }
}

/// External memory import information.
#[derive(Debug, Clone)]
pub struct ExternalMemoryImportInfo {
    /// The handle type being imported.
    pub handle_type: ExternalHandleType,
    /// Size in bytes of the memory to import.
    pub size: vk::DeviceSize,
    /// Memory type index to use.
    pub memory_type_index: u32,
}

/// External semaphore import information.
#[derive(Debug, Clone)]
pub struct ExternalSemaphoreImportInfo {
    /// The handle type being imported.
    pub handle_type: ExternalHandleType,
    /// Whether to use a temporary import (released after first wait).
    pub temporary: bool,
}

/// Query external memory capabilities for a buffer.
///
/// Returns the compatible handle types, export features, and import features
/// for the given external buffer configuration.
pub fn query_external_buffer_properties(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    usage: vk::BufferUsageFlags,
    handle_type: ExternalHandleType,
) -> vk::ExternalBufferProperties<'static> {
    let external_info = vk::PhysicalDeviceExternalBufferInfo::default()
        .usage(usage)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::from(handle_type));

    let mut properties = vk::ExternalBufferProperties::default();

    // SAFETY: instance and physical_device are valid.
    unsafe {
        instance.get_physical_device_external_buffer_properties(
            physical_device,
            &external_info,
            &mut properties,
        );
    }

    properties
}

/// Query external semaphore capabilities.
///
/// Returns the compatible handle types, export features, and import features
/// for the given external semaphore configuration.
pub fn query_external_semaphore_properties(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    handle_type: ExternalHandleType,
) -> vk::ExternalSemaphoreProperties<'static> {
    let external_info = vk::PhysicalDeviceExternalSemaphoreInfo::default()
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::from(handle_type));

    let mut properties = vk::ExternalSemaphoreProperties::default();

    // SAFETY: instance and physical_device are valid.
    unsafe {
        instance.get_physical_device_external_semaphore_properties(
            physical_device,
            &external_info,
            &mut properties,
        );
    }

    properties
}

/// POSIX file descriptor-based external memory operations.
///
/// Available on Linux, macOS (via MoltenVK), and other POSIX platforms.
#[cfg(unix)]
pub mod fd {
    use ash::vk;

    /// Get a POSIX file descriptor for an exported memory allocation.
    pub fn get_memory_fd(
        ext_mem_fd: &ash::khr::external_memory_fd::Device,
        memory: vk::DeviceMemory,
        handle_type: vk::ExternalMemoryHandleTypeFlags,
    ) -> Result<std::os::unix::io::RawFd, vk::Result> {
        let get_info = vk::MemoryGetFdInfoKHR::default()
            .memory(memory)
            .handle_type(handle_type);

        // SAFETY: ext_mem_fd loader and memory are valid.
        unsafe { ext_mem_fd.get_memory_fd(&get_info) }
    }

    /// Get a POSIX file descriptor for an exported semaphore.
    pub fn get_semaphore_fd(
        ext_sem_fd: &ash::khr::external_semaphore_fd::Device,
        semaphore: vk::Semaphore,
        handle_type: vk::ExternalSemaphoreHandleTypeFlags,
    ) -> Result<std::os::unix::io::RawFd, vk::Result> {
        let get_info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(handle_type);

        // SAFETY: ext_sem_fd loader and semaphore are valid.
        unsafe { ext_sem_fd.get_semaphore_fd(&get_info) }
    }

    /// Import a POSIX file descriptor as a semaphore.
    pub fn import_semaphore_fd(
        ext_sem_fd: &ash::khr::external_semaphore_fd::Device,
        semaphore: vk::Semaphore,
        fd: std::os::unix::io::RawFd,
        handle_type: vk::ExternalSemaphoreHandleTypeFlags,
        temporary: bool,
    ) -> Result<(), vk::Result> {
        let mut import_info = vk::ImportSemaphoreFdInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(handle_type)
            .fd(fd);

        if temporary {
            import_info.flags = vk::SemaphoreImportFlags::TEMPORARY;
        }

        // SAFETY: ext_sem_fd loader and semaphore are valid, fd is a valid handle.
        unsafe { ext_sem_fd.import_semaphore_fd(&import_info) }
    }
}
