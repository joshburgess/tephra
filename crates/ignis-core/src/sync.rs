//! Semaphore and fence helpers with per-frame recycling.
//!
//! Semaphores and fences are pooled rather than created/destroyed each frame.
//! When a frame context resets, its sync objects return to the pool for reuse.

use ash::vk;

use crate::device::{Device, DeviceError};

/// A recycled Vulkan semaphore.
///
/// Returned to the recycling pool when the frame context resets.
pub struct SemaphoreHandle {
    pub(crate) raw: vk::Semaphore,
}

impl SemaphoreHandle {
    /// The raw Vulkan semaphore handle.
    pub fn raw(&self) -> vk::Semaphore {
        self.raw
    }
}

/// Pool of recycled semaphores.
pub(crate) struct SemaphorePool {
    free: Vec<vk::Semaphore>,
}

impl SemaphorePool {
    pub fn new() -> Self {
        Self {
            free: Vec::with_capacity(8),
        }
    }

    /// Get a semaphore from the pool or create a new one.
    pub fn request(&mut self, device: &ash::Device) -> Result<vk::Semaphore, vk::Result> {
        if let Some(sem) = self.free.pop() {
            return Ok(sem);
        }

        let ci = vk::SemaphoreCreateInfo::default();
        // SAFETY: device is valid, ci is well-formed.
        let sem = unsafe { device.create_semaphore(&ci, None)? };
        log::debug!("Created new semaphore");
        Ok(sem)
    }

    /// Return a semaphore to the pool for reuse.
    pub fn recycle(&mut self, semaphore: vk::Semaphore) {
        self.free.push(semaphore);
    }

    /// Destroy all pooled semaphores.
    pub fn destroy(&mut self, device: &ash::Device) {
        for sem in self.free.drain(..) {
            // SAFETY: device is valid, semaphore is valid, GPU is idle.
            unsafe {
                device.destroy_semaphore(sem, None);
            }
        }
    }
}

/// Pool of recycled fences.
pub(crate) struct FencePool {
    free: Vec<vk::Fence>,
}

impl FencePool {
    pub fn new() -> Self {
        Self {
            free: Vec::with_capacity(4),
        }
    }

    /// Get a fence from the pool or create a new one (in unsignaled state).
    pub fn request(&mut self, device: &ash::Device) -> Result<vk::Fence, vk::Result> {
        if let Some(fence) = self.free.pop() {
            // Reset fence to unsignaled before reuse
            // SAFETY: device is valid, fence is valid and not pending.
            unsafe { device.reset_fences(&[fence])? };
            return Ok(fence);
        }

        let ci = vk::FenceCreateInfo::default();
        // SAFETY: device is valid, ci is well-formed.
        let fence = unsafe { device.create_fence(&ci, None)? };
        log::debug!("Created new fence");
        Ok(fence)
    }

    /// Return a fence to the pool for reuse.
    pub fn recycle(&mut self, fence: vk::Fence) {
        self.free.push(fence);
    }

    /// Destroy all pooled fences.
    pub fn destroy(&mut self, device: &ash::Device) {
        for fence in self.free.drain(..) {
            // SAFETY: device is valid, fence is valid, GPU is idle.
            unsafe {
                device.destroy_fence(fence, None);
            }
        }
    }
}

/// Sync-related methods on Device.
impl Device {
    /// Request a semaphore from the recycling pool.
    pub fn request_semaphore(&mut self) -> Result<SemaphoreHandle, DeviceError> {
        let sem = self.semaphore_pool.request(self.context.device())?;
        Ok(SemaphoreHandle { raw: sem })
    }

    /// Return a semaphore to the recycling pool.
    pub fn recycle_semaphore(&mut self, semaphore: SemaphoreHandle) {
        self.semaphore_pool.recycle(semaphore.raw);
    }

    /// Request a fence from the recycling pool.
    pub fn request_fence(&mut self) -> Result<vk::Fence, DeviceError> {
        let fence = self.fence_pool.request(self.context.device())?;
        Ok(fence)
    }

    /// Return a fence to the recycling pool.
    pub fn recycle_fence(&mut self, fence: vk::Fence) {
        self.fence_pool.recycle(fence);
    }

    /// Wait for a fence to signal.
    pub fn wait_fence(&self, fence: vk::Fence, timeout_ns: u64) -> Result<bool, DeviceError> {
        // SAFETY: device and fence are valid.
        let result = unsafe { self.raw().wait_for_fences(&[fence], true, timeout_ns) };
        match result {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(e) => Err(DeviceError::Vulkan(e)),
        }
    }
}
