//! GPU crash dump / post-mortem debugging support.
//!
//! Provides interfaces for collecting GPU crash information after a device
//! lost event, including NVIDIA Nsight Aftermath integration and generic
//! breadcrumb-based debugging.
//!
//! # Aftermath Integration
//!
//! When running on NVIDIA GPUs with the Aftermath SDK available, this module
//! can enable GPU crash dump collection. After a `VK_ERROR_DEVICE_LOST`,
//! crash dumps are written to disk for analysis in Nsight Graphics.
//!
//! # Generic Breadcrumbs
//!
//! For non-NVIDIA GPUs, a simple breadcrumb buffer approach is provided.
//! Write marker values to a host-visible buffer at key points in command
//! buffer recording. After device lost, read the buffer to determine
//! which commands completed before the crash.

use ash::vk;

/// Configuration for post-mortem debugging.
#[derive(Debug, Clone)]
pub struct PostMortemConfig {
    /// Enable NVIDIA Aftermath crash dump collection (if available).
    pub enable_aftermath: bool,
    /// Directory for crash dump files.
    pub crash_dump_dir: std::path::PathBuf,
    /// Enable breadcrumb markers in command buffers.
    pub enable_breadcrumbs: bool,
    /// Number of breadcrumb entries per command buffer.
    pub breadcrumb_count: u32,
}

impl Default for PostMortemConfig {
    fn default() -> Self {
        Self {
            enable_aftermath: true,
            crash_dump_dir: std::path::PathBuf::from("crash_dumps"),
            enable_breadcrumbs: true,
            breadcrumb_count: 256,
        }
    }
}

/// Breadcrumb marker for tracking command buffer progress.
///
/// Write these values to a host-visible buffer at key points in command
/// recording. After a device lost, read back the buffer to see which
/// commands completed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BreadcrumbMarker {
    /// Identifier for this marker (e.g., draw call index).
    pub id: u32,
    /// Human-readable label for debugging.
    pub label: &'static str,
}

/// Manages a breadcrumb buffer for post-mortem debugging.
///
/// Uses `vkCmdFillBuffer` or mapped writes to track command buffer progress.
/// After device lost, the last completed marker indicates where the GPU
/// stalled.
pub struct BreadcrumbBuffer {
    buffer: vk::Buffer,
    allocation: Option<gpu_allocator::vulkan::Allocation>,
    mapped_ptr: *mut u32,
    capacity: u32,
    next_marker: u32,
    markers: Vec<BreadcrumbMarker>,
}

// SAFETY: The breadcrumb buffer uses host-visible memory accessed through
// atomic-compatible writes. Access is synchronized by frame context.
unsafe impl Send for BreadcrumbBuffer {}
// SAFETY: Same as Send.
unsafe impl Sync for BreadcrumbBuffer {}

impl BreadcrumbBuffer {
    /// Create a new breadcrumb buffer.
    pub fn new(
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
        capacity: u32,
    ) -> Result<Self, vk::Result> {
        let size = capacity as vk::DeviceSize * std::mem::size_of::<u32>() as vk::DeviceSize;

        let buffer_ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        // SAFETY: device is valid, buffer_ci is well-formed.
        let buffer = unsafe { device.create_buffer(&buffer_ci, None)? };

        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = allocator
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "breadcrumb_buffer",
                requirements,
                location: gpu_allocator::MemoryLocation::GpuToCpu,
                linear: true,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|_| vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;

        // SAFETY: device, buffer, and allocation are valid.
        unsafe {
            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        }

        let mapped_ptr = allocation
            .mapped_ptr()
            .map(|p| p.as_ptr() as *mut u32)
            .unwrap_or(std::ptr::null_mut());

        // Zero-initialize the buffer
        if !mapped_ptr.is_null() {
            // SAFETY: mapped_ptr is valid for capacity u32 entries.
            unsafe {
                std::ptr::write_bytes(mapped_ptr, 0, capacity as usize);
            }
        }

        Ok(Self {
            buffer,
            allocation: Some(allocation),
            mapped_ptr,
            capacity,
            next_marker: 0,
            markers: Vec::with_capacity(capacity as usize),
        })
    }

    /// Reset the breadcrumb buffer for a new frame.
    pub fn reset(&mut self) {
        self.next_marker = 0;
        self.markers.clear();

        if !self.mapped_ptr.is_null() {
            // SAFETY: mapped_ptr is valid for capacity entries.
            unsafe {
                std::ptr::write_bytes(self.mapped_ptr, 0, self.capacity as usize);
            }
        }
    }

    /// Record a breadcrumb marker.
    ///
    /// Writes the marker ID to the breadcrumb buffer via `vkCmdFillBuffer`.
    /// The marker value is also stored locally for later lookup.
    pub fn write_marker(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        marker: BreadcrumbMarker,
    ) {
        if self.next_marker >= self.capacity {
            return;
        }

        let offset =
            self.next_marker as vk::DeviceSize * std::mem::size_of::<u32>() as vk::DeviceSize;

        // SAFETY: device, cmd, and buffer are valid. offset is within bounds.
        unsafe {
            device.cmd_fill_buffer(
                cmd,
                self.buffer,
                offset,
                std::mem::size_of::<u32>() as vk::DeviceSize,
                marker.id,
            );
        }

        self.markers.push(marker);
        self.next_marker += 1;
    }

    /// Read back breadcrumb values after a device lost event.
    ///
    /// Returns the markers that were completed (non-zero values in the buffer).
    pub fn read_completed_markers(&self) -> Vec<&BreadcrumbMarker> {
        if self.mapped_ptr.is_null() {
            return Vec::new();
        }

        let mut completed = Vec::new();
        for (i, marker) in self.markers.iter().enumerate() {
            // SAFETY: mapped_ptr is valid, i is within capacity.
            let value = unsafe { *self.mapped_ptr.add(i) };
            if value != 0 {
                completed.push(marker);
            }
        }
        completed
    }

    /// Find the last completed marker (the furthest point the GPU reached).
    pub fn last_completed_marker(&self) -> Option<&BreadcrumbMarker> {
        if self.mapped_ptr.is_null() {
            return None;
        }

        let mut last = None;
        for (i, marker) in self.markers.iter().enumerate() {
            // SAFETY: mapped_ptr is valid, i is within capacity.
            let value = unsafe { *self.mapped_ptr.add(i) };
            if value != 0 {
                last = Some(marker);
            }
        }
        last
    }

    /// The underlying buffer handle (for barrier purposes).
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    /// Destroy the breadcrumb buffer.
    pub fn destroy(
        &mut self,
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
    ) {
        if let Some(alloc) = self.allocation.take() {
            allocator.free(alloc).ok();
        }
        if self.buffer != vk::Buffer::null() {
            // SAFETY: device is valid, buffer is valid.
            unsafe {
                device.destroy_buffer(self.buffer, None);
            }
            self.buffer = vk::Buffer::null();
        }
    }
}

/// Aftermath state (NVIDIA-specific).
///
/// This is a configuration struct. Full Aftermath integration requires
/// linking against the Aftermath SDK and registering callbacks via
/// `GFSDK_Aftermath_EnableGpuCrashDumps`.
#[derive(Debug, Clone)]
pub struct AftermathConfig {
    /// Enable GPU crash dumps.
    pub enable_crash_dumps: bool,
    /// Enable shader debug info for source correlation.
    pub enable_shader_debug_info: bool,
    /// Enable automatic marker tracking.
    pub enable_markers: bool,
    /// Enable resource tracking.
    pub enable_resource_tracking: bool,
}

impl Default for AftermathConfig {
    fn default() -> Self {
        Self {
            enable_crash_dumps: true,
            enable_shader_debug_info: true,
            enable_markers: true,
            enable_resource_tracking: true,
        }
    }
}

/// Check if NVIDIA Aftermath is available on the system.
///
/// This checks for the Aftermath runtime library without loading it.
pub fn is_aftermath_available() -> bool {
    #[cfg(target_os = "windows")]
    {
        // Check for GFSDK_Aftermath_Lib.x64.dll
        std::path::Path::new("GFSDK_Aftermath_Lib.x64.dll").exists()
    }
    #[cfg(target_os = "linux")]
    {
        // Check for libGFSDK_Aftermath_Lib.x64.so
        std::path::Path::new("/usr/lib/libGFSDK_Aftermath_Lib.x64.so").exists()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        false
    }
}
