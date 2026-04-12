//! Integration tests for the full frame loop lifecycle.
//!
//! Tests multi-frame rendering, deferred deletion timing, fence synchronization,
//! and resource cleanup with validation layers enabled.

mod validation_harness;

use ash::vk;

use tephra::core::buffer::BufferCreateInfo;
use tephra::core::context::QueueType;
use tephra::core::image::ImageCreateInfo;
use tephra::core::memory::MemoryDomain;

use validation_harness::{
    assert_no_validation_errors, clear_errors, create_headless_device, init_capture_logger,
};

/// Run multiple frames with empty submissions to verify fence lifecycle.
///
/// Exercises: fence creation (SIGNALED), begin_frame wait, fence reset,
/// command pool reset, frame index advancement.
#[test]
fn multi_frame_empty_submissions() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("multi_frame_empty");

    for _ in 0..10 {
        device.begin_frame().expect("begin_frame");
        let cmd = device
            .request_command_buffer_raw(QueueType::Graphics)
            .expect("request cmd");
        let fence = device.current_fence();
        device
            .submit_command_buffer(cmd, QueueType::Graphics, &[], &[], &[], fence)
            .expect("submit");
    }

    drop(device);
    assert_no_validation_errors();
}

/// Create and destroy buffers across multiple frames, verifying deferred deletion
/// doesn't trigger validation errors.
///
/// Resources destroyed in frame N are not actually freed until frame N + FRAME_OVERLAP.
#[test]
fn deferred_buffer_deletion() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("deferred_buf_del");

    // Frame 1: create buffers and destroy some
    device.begin_frame().expect("begin_frame");
    let buf_a = device
        .create_buffer(&BufferCreateInfo::uniform(256))
        .expect("buf_a");
    let buf_b = device
        .create_buffer(&BufferCreateInfo::uniform(512))
        .expect("buf_b");
    // Destroy buf_a — goes into frame 1's deletion queue
    device.destroy_buffer(buf_a);

    let cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd");
    let fence = device.current_fence();
    device
        .submit_command_buffer(cmd, QueueType::Graphics, &[], &[], &[], fence)
        .expect("submit");

    // Frame 2: buf_a's deletion should be pending, not yet flushed
    device.begin_frame().expect("begin_frame");
    device.destroy_buffer(buf_b);
    let cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd");
    let fence = device.current_fence();
    device
        .submit_command_buffer(cmd, QueueType::Graphics, &[], &[], &[], fence)
        .expect("submit");

    // Frame 3: frame 1's deletion queue (buf_a) gets flushed here
    device.begin_frame().expect("begin_frame");
    let cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd");
    let fence = device.current_fence();
    device
        .submit_command_buffer(cmd, QueueType::Graphics, &[], &[], &[], fence)
        .expect("submit");

    // Frame 4: frame 2's deletion queue (buf_b) gets flushed
    device.begin_frame().expect("begin_frame");

    drop(device);
    assert_no_validation_errors();
}

/// Create and destroy images across frames.
#[test]
fn deferred_image_deletion() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("deferred_img_del");

    device.begin_frame().expect("begin_frame");
    let img = device
        .create_image(&ImageCreateInfo::render_target(
            64,
            64,
            vk::Format::R8G8B8A8_UNORM,
        ))
        .expect("image");

    let cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd");
    let fence = device.current_fence();
    device
        .submit_command_buffer(cmd, QueueType::Graphics, &[], &[], &[], fence)
        .expect("submit");

    // Destroy in frame 2 — deferred
    device.begin_frame().expect("begin_frame");
    device.destroy_image(img);
    let cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd");
    let fence = device.current_fence();
    device
        .submit_command_buffer(cmd, QueueType::Graphics, &[], &[], &[], fence)
        .expect("submit");

    // Frame 3: deletion gets flushed
    device.begin_frame().expect("begin_frame");

    drop(device);
    assert_no_validation_errors();
}

/// Create resources, use them in GPU commands, then destroy them.
/// Verifies that deferred deletion waits until the GPU is done.
#[test]
fn resource_use_then_destroy() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("use_then_destroy");

    // Frame 1: create a staging buffer with data, copy to device-local buffer
    device.begin_frame().expect("begin_frame");

    let data: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
    let staging_info = BufferCreateInfo::staging(256);
    let mut staging = device.create_buffer(&staging_info).expect("staging");
    if let Some(slice) = staging.mapped_slice_mut() {
        slice[..256].copy_from_slice(&data);
    }

    let dst_info = BufferCreateInfo {
        size: 256,
        usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER,
        domain: MemoryDomain::Device,
    };
    let dst = device.create_buffer(&dst_info).expect("dst");

    // Record a copy command
    let cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd");
    // SAFETY: cmd, src, dst are valid
    unsafe {
        device.raw().cmd_copy_buffer(
            cmd,
            staging.raw(),
            dst.raw(),
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: 256,
            }],
        );
    }
    let fence = device.current_fence();
    device
        .submit_command_buffer(cmd, QueueType::Graphics, &[], &[], &[], fence)
        .expect("submit");

    // Destroy staging buffer — deferred, GPU may still be reading it
    device.destroy_buffer(staging);

    // Frame 2: GPU is now done with the copy (fence waited in begin_frame)
    device.begin_frame().expect("begin_frame");
    device.destroy_buffer(dst);

    let cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd");
    let fence = device.current_fence();
    device
        .submit_command_buffer(cmd, QueueType::Graphics, &[], &[], &[], fence)
        .expect("submit");

    // Frame 3: both deletion queues flushed
    device.begin_frame().expect("begin_frame");

    drop(device);
    assert_no_validation_errors();
}

/// Run many frames creating and destroying resources each frame.
/// Stress-tests the ring buffer recycling and deletion queue flushing.
#[test]
fn sustained_frame_loop_with_churn() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("sustained_churn");

    for i in 0..20 {
        device.begin_frame().expect("begin_frame");

        // Create a buffer each frame
        let buf = device
            .create_buffer(&BufferCreateInfo::uniform(64 + (i * 16) as u64))
            .expect("buffer");

        // Create an image every other frame
        if i % 2 == 0 {
            let img = device
                .create_image(&ImageCreateInfo::render_target(
                    32,
                    32,
                    vk::Format::R8G8B8A8_UNORM,
                ))
                .expect("image");
            device.destroy_image(img);
        }

        // Submit a frame
        let cmd = device
            .request_command_buffer_raw(QueueType::Graphics)
            .expect("cmd");
        let fence = device.current_fence();
        device
            .submit_command_buffer(cmd, QueueType::Graphics, &[], &[], &[], fence)
            .expect("submit");

        // Destroy the buffer — deferred
        device.destroy_buffer(buf);
    }

    // Final frames to flush remaining deletion queues
    for _ in 0..3 {
        device.begin_frame().expect("begin_frame");
        let cmd = device
            .request_command_buffer_raw(QueueType::Graphics)
            .expect("cmd");
        let fence = device.current_fence();
        device
            .submit_command_buffer(cmd, QueueType::Graphics, &[], &[], &[], fence)
            .expect("submit");
    }

    drop(device);
    assert_no_validation_errors();
}

/// Verify frame count increments correctly and command pools are usable each frame.
#[test]
fn frame_count_advances() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("frame_count");

    assert_eq!(device.frame_count(), 0);

    for expected in 1..=5u64 {
        device.begin_frame().expect("begin_frame");
        assert_eq!(device.frame_count(), expected);

        // Verify we can allocate and submit a command buffer each frame
        let cmd = device
            .request_command_buffer_raw(QueueType::Graphics)
            .expect("cmd");
        let fence = device.current_fence();
        device
            .submit_command_buffer(cmd, QueueType::Graphics, &[], &[], &[], fence)
            .expect("submit");
    }

    drop(device);
    assert_no_validation_errors();
}

/// Submit using submit_command_buffer_for_frame (convenience wrapper).
#[test]
fn submit_for_frame_convenience() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("submit_for_frame");

    for _ in 0..5 {
        device.begin_frame().expect("begin_frame");
        let cmd = device
            .request_command_buffer_raw(QueueType::Graphics)
            .expect("cmd");
        device
            .submit_command_buffer_for_frame(cmd, &[], &[], &[])
            .expect("submit");
    }

    drop(device);
    assert_no_validation_errors();
}

/// Create a buffer with immediate data upload (staging + copy).
/// Verifies the internal staging buffer lifecycle works correctly.
#[test]
fn buffer_with_data_upload() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("buf_data_upload");

    device.begin_frame().expect("begin_frame");

    let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let info = BufferCreateInfo {
        size: 8,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        domain: MemoryDomain::Device,
    };
    let buf = device
        .create_buffer_with_data(&info, &data)
        .expect("buffer with data");

    let cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd");
    let fence = device.current_fence();
    device
        .submit_command_buffer(cmd, QueueType::Graphics, &[], &[], &[], fence)
        .expect("submit");

    device.begin_frame().expect("begin_frame");
    device.destroy_buffer(buf);

    // Flush deletion
    device.begin_frame().expect("begin_frame");

    drop(device);
    assert_no_validation_errors();
}

/// Create an image with data upload via staging buffer.
#[test]
fn image_with_data_upload() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("img_data_upload");

    device.begin_frame().expect("begin_frame");

    // 4x4 RGBA8 image = 64 bytes
    let pixels = vec![0xFFu8; 4 * 4 * 4];
    let info = ImageCreateInfo::immutable_2d(4, 4, vk::Format::R8G8B8A8_UNORM);
    let img = device
        .create_image_with_data(&info, &pixels)
        .expect("image with data");

    let cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd");
    let fence = device.current_fence();
    device
        .submit_command_buffer(cmd, QueueType::Graphics, &[], &[], &[], fence)
        .expect("submit");

    device.begin_frame().expect("begin_frame");
    device.destroy_image(img);

    // Flush deletion
    device.begin_frame().expect("begin_frame");

    drop(device);
    assert_no_validation_errors();
}

/// Verify that Device drop correctly cleans up even with pending deletions.
#[test]
fn drop_with_pending_deletions() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("drop_pending");

    device.begin_frame().expect("begin_frame");

    // Create several resources
    let buf = device
        .create_buffer(&BufferCreateInfo::uniform(128))
        .expect("buffer");
    let img = device
        .create_image(&ImageCreateInfo::render_target(
            32,
            32,
            vk::Format::R8G8B8A8_UNORM,
        ))
        .expect("image");

    // Destroy them — they go into the deletion queue
    device.destroy_buffer(buf);
    device.destroy_image(img);

    // Drop without advancing frames — Device::drop should handle flush_all
    drop(device);
    assert_no_validation_errors();
}

/// Verify read-back from a host-visible buffer after GPU write.
#[test]
fn buffer_readback_after_copy() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("buf_readback");

    device.begin_frame().expect("begin_frame");

    let size = 64u64;
    let src_info = BufferCreateInfo {
        size,
        usage: vk::BufferUsageFlags::TRANSFER_SRC,
        domain: MemoryDomain::Host,
    };
    let mut src = device.create_buffer(&src_info).expect("src");

    // Fill source with pattern
    if let Some(slice) = src.mapped_slice_mut() {
        for (i, byte) in slice.iter_mut().enumerate().take(size as usize) {
            *byte = (i * 3 % 256) as u8;
        }
    }

    let dst_info = BufferCreateInfo {
        size,
        usage: vk::BufferUsageFlags::TRANSFER_DST,
        domain: MemoryDomain::CachedHost,
    };
    let dst = device.create_buffer(&dst_info).expect("dst");

    // Copy src → dst
    let cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd");
    // SAFETY: cmd, buffers are valid
    unsafe {
        device.raw().cmd_copy_buffer(
            cmd,
            src.raw(),
            dst.raw(),
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            }],
        );
    }
    let fence = device.current_fence();
    device
        .submit_command_buffer(cmd, QueueType::Graphics, &[], &[], &[], fence)
        .expect("submit");

    // Wait for GPU to finish — must wait explicitly since begin_frame with
    // FRAME_OVERLAP=2 won't necessarily wait on this frame's fence yet
    device
        .wait_queue_idle(QueueType::Graphics)
        .expect("wait idle");

    // Verify data
    let dst_slice = dst.mapped_slice().expect("dst mapped");
    for (i, byte) in dst_slice.iter().enumerate().take(size as usize) {
        assert_eq!(*byte, (i * 3 % 256) as u8, "mismatch at index {i}");
    }

    device.destroy_buffer(src);
    device.destroy_buffer(dst);

    drop(device);
    assert_no_validation_errors();
}
