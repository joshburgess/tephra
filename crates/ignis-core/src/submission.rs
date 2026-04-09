//! Command buffer allocation and queue submission on [`Device`].

use ash::vk;

use crate::context::QueueType;
use crate::device::{Device, DeviceError};

/// Submission-related methods on Device.
impl Device {
    /// Allocate a command buffer from the current frame's command pool and begin recording.
    ///
    /// The returned command buffer uses `ONE_TIME_SUBMIT_BIT`. It must be submitted
    /// back via [`submit_command_buffer`](Device::submit_command_buffer) within
    /// the same frame.
    pub fn request_command_buffer_raw(
        &self,
        queue_type: QueueType,
    ) -> Result<vk::CommandBuffer, DeviceError> {
        let pool = self.command_pool_for_queue(queue_type);

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        // SAFETY: device and pool are valid.
        let cmd = unsafe { self.raw().allocate_command_buffers(&alloc_info)? }[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        // SAFETY: cmd is valid and freshly allocated.
        unsafe {
            self.raw().begin_command_buffer(cmd, &begin_info)?;
        }

        Ok(cmd)
    }

    /// End recording and submit a command buffer to the specified queue.
    ///
    /// Optionally signals a fence and/or wait/signal semaphores.
    pub fn submit_command_buffer(
        &mut self,
        cmd: vk::CommandBuffer,
        queue_type: QueueType,
        wait_semaphores: &[vk::Semaphore],
        wait_stages: &[vk::PipelineStageFlags],
        signal_semaphores: &[vk::Semaphore],
        fence: vk::Fence,
    ) -> Result<(), DeviceError> {
        // SAFETY: cmd is valid and has been recorded.
        unsafe {
            self.raw().end_command_buffer(cmd)?;
        }

        let submit_info = vk::SubmitInfo::default()
            .command_buffers(std::slice::from_ref(&cmd))
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .signal_semaphores(signal_semaphores);

        let queue = self.context.queue(queue_type).queue;

        // SAFETY: queue, cmd, semaphores, and fence are valid.
        unsafe {
            self.raw()
                .queue_submit(queue, &[submit_info], fence)?;
        }

        if fence != vk::Fence::null() && fence == self.current_fence() {
            self.mark_fence_submitted();
        }

        Ok(())
    }

    /// Submit a command buffer to the graphics queue, signaling the current frame fence.
    pub fn submit_command_buffer_for_frame(
        &mut self,
        cmd: vk::CommandBuffer,
        wait_semaphores: &[vk::Semaphore],
        wait_stages: &[vk::PipelineStageFlags],
        signal_semaphores: &[vk::Semaphore],
    ) -> Result<(), DeviceError> {
        let fence = self.current_fence();
        self.submit_command_buffer(
            cmd,
            QueueType::Graphics,
            wait_semaphores,
            wait_stages,
            signal_semaphores,
            fence,
        )
    }

    /// Wait for a queue to become idle.
    pub fn wait_queue_idle(&self, queue_type: QueueType) -> Result<(), DeviceError> {
        let queue = self.context.queue(queue_type).queue;
        // SAFETY: queue is valid.
        unsafe { self.raw().queue_wait_idle(queue)? };
        Ok(())
    }
}
