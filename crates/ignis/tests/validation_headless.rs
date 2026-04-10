//! Headless GPU validation tests.
//!
//! Creates a headless device with validation layers, runs various GPU operations,
//! and asserts zero validation errors. Tests cover buffer operations, image
//! operations, mipmap generation, and command buffer recording.

mod validation_harness;

use ash::vk;

use ignis::command::barriers::ImageBarrierInfo;
use ignis::command::command_buffer::{CommandBuffer, CommandBufferType};
use ignis::core::buffer::BufferCreateInfo;
use ignis::core::context::QueueType;
use ignis::core::image::ImageCreateInfo;
use ignis::core::memory::{ImageDomain, MemoryDomain};

use validation_harness::*;

/// Helper: submit a command buffer and wait for completion.
fn submit_and_wait(device: &mut ignis::core::device::Device, cmd: &CommandBuffer) {
    let fence = device.current_fence();
    device
        .submit_command_buffer(cmd.raw(), QueueType::Graphics, &[], &[], &[], fence)
        .expect("submit");
    // SAFETY: fence is valid.
    unsafe {
        device
            .raw()
            .wait_for_fences(&[fence], true, u64::MAX)
            .expect("wait");
    }
}

#[test]
fn buffer_create_fill_readback() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("validation_buffer_test");

    // Create a host-visible buffer and write data
    let size = 256u64;
    let info = BufferCreateInfo {
        size,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        domain: MemoryDomain::Host,
    };
    let mut buffer = device.create_buffer(&info).expect("buffer creation");

    // Write pattern
    if let Some(slice) = buffer.mapped_slice_mut() {
        for (i, byte) in slice[..size as usize].iter_mut().enumerate() {
            *byte = (i & 0xFF) as u8;
        }
    }

    // Read back and verify
    let slice = buffer.mapped_slice().expect("buffer not mapped");
    for (i, &byte) in slice[..size as usize].iter().enumerate() {
        assert_eq!(byte, (i & 0xFF) as u8, "mismatch at byte {i}");
    }

    // Cleanup
    // SAFETY: no GPU work in flight.
    unsafe {
        device.raw().device_wait_idle().ok();
    }
    device.destroy_buffer(buffer);

    assert_no_validation_errors();
}

#[test]
fn image_create_and_transition() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("validation_image_test");

    // Create a 2D image
    let image_ci = ImageCreateInfo {
        width: 64,
        height: 64,
        depth: 1,
        format: vk::Format::R8G8B8A8_UNORM,
        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED,
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        image_type: vk::ImageType::TYPE_2D,
        initial_layout: vk::ImageLayout::UNDEFINED,
        domain: ImageDomain::Physical,
    };

    let image = device.create_image(&image_ci).expect("image creation");

    // Record layout transitions
    device.begin_frame().expect("begin_frame");
    let raw_cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd alloc");
    let mut cmd =
        CommandBuffer::from_raw(raw_cmd, CommandBufferType::Graphics, device.raw().clone());

    // UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
    cmd.image_barrier(&ImageBarrierInfo::undefined_to(
        image.raw(),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        vk::ImageAspectFlags::COLOR,
    ));

    // COLOR_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
    cmd.image_barrier(&ImageBarrierInfo {
        image: image.raw(),
        old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        src_stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        src_access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        dst_stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
        dst_access: vk::AccessFlags2::SHADER_READ,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
        src_queue_family: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family: vk::QUEUE_FAMILY_IGNORED,
    });

    submit_and_wait(&mut device, &cmd);

    // Cleanup
    // SAFETY: GPU is idle.
    unsafe {
        device.raw().device_wait_idle().ok();
    }
    device.destroy_image(image);

    assert_no_validation_errors();
}

#[test]
fn mipmap_generation() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("validation_mipmap_test");

    let width = 128u32;
    let height = 128u32;
    let mip_count = (width.max(height) as f32).log2().floor() as u32 + 1;

    // Create image with mips
    let image_ci = ImageCreateInfo {
        width,
        height,
        depth: 1,
        format: vk::Format::R8G8B8A8_UNORM,
        usage: vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED,
        mip_levels: mip_count,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        image_type: vk::ImageType::TYPE_2D,
        initial_layout: vk::ImageLayout::UNDEFINED,
        domain: ImageDomain::Physical,
    };
    let image = device.create_image(&image_ci).expect("image");

    // Create staging buffer with test data
    let data_size = (width * height * 4) as u64;
    let staging_info = BufferCreateInfo {
        size: data_size,
        usage: vk::BufferUsageFlags::TRANSFER_SRC,
        domain: MemoryDomain::Host,
    };
    let mut staging = device.create_buffer(&staging_info).expect("staging");
    if let Some(slice) = staging.mapped_slice_mut() {
        // Fill with a gradient pattern
        for (i, byte) in slice.iter_mut().enumerate().take(data_size as usize) {
            *byte = (i % 256) as u8;
        }
    }

    // Record commands
    device.begin_frame().expect("begin_frame");
    let raw_cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd alloc");
    let mut cmd =
        CommandBuffer::from_raw(raw_cmd, CommandBufferType::Graphics, device.raw().clone());

    // UNDEFINED -> TRANSFER_DST for all mips
    cmd.image_barrier(&ImageBarrierInfo {
        image: image.raw(),
        old_layout: vk::ImageLayout::UNDEFINED,
        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        src_stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
        src_access: vk::AccessFlags2::NONE,
        dst_stage: vk::PipelineStageFlags2::TRANSFER,
        dst_access: vk::AccessFlags2::TRANSFER_WRITE,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: mip_count,
            base_array_layer: 0,
            layer_count: 1,
        },
        src_queue_family: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family: vk::QUEUE_FAMILY_IGNORED,
    });

    // Copy staging -> mip 0
    let region = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
    };
    // SAFETY: all handles are valid.
    unsafe {
        device.raw().cmd_copy_buffer_to_image(
            cmd.raw(),
            staging.raw(),
            image.raw(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }

    // Generate mipmaps
    cmd.generate_mipmap(image.raw(), width, height, mip_count);

    submit_and_wait(&mut device, &cmd);

    // Cleanup
    // SAFETY: GPU is idle.
    unsafe {
        device.raw().device_wait_idle().ok();
    }
    device.destroy_buffer(staging);
    device.destroy_image(image);

    assert_no_validation_errors();
}

#[test]
fn buffer_copy() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("validation_buffer_copy_test");

    let size = 1024u64;

    // Source buffer
    let src_info = BufferCreateInfo {
        size,
        usage: vk::BufferUsageFlags::TRANSFER_SRC,
        domain: MemoryDomain::Host,
    };
    let mut src = device.create_buffer(&src_info).expect("src buffer");
    if let Some(slice) = src.mapped_slice_mut() {
        for (i, byte) in slice.iter_mut().enumerate().take(size as usize) {
            *byte = (i % 256) as u8;
        }
    }

    // Destination buffer
    let dst_info = BufferCreateInfo {
        size,
        usage: vk::BufferUsageFlags::TRANSFER_DST,
        domain: MemoryDomain::CachedHost,
    };
    let dst = device.create_buffer(&dst_info).expect("dst buffer");

    // Record copy
    device.begin_frame().expect("begin_frame");
    let raw_cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd alloc");
    let cmd = CommandBuffer::from_raw(raw_cmd, CommandBufferType::Graphics, device.raw().clone());

    let copy_region = vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size,
    };
    // SAFETY: all handles are valid.
    unsafe {
        device
            .raw()
            .cmd_copy_buffer(cmd.raw(), src.raw(), dst.raw(), &[copy_region]);
    }

    submit_and_wait(&mut device, &cmd);

    // Verify
    let dst_slice = dst.mapped_slice().expect("dst not mapped");
    for (i, byte) in dst_slice.iter().enumerate().take(size as usize) {
        assert_eq!(
            *byte,
            (i % 256) as u8,
            "buffer copy mismatch at byte {i}"
        );
    }

    // Cleanup
    // SAFETY: GPU is idle.
    unsafe {
        device.raw().device_wait_idle().ok();
    }
    device.destroy_buffer(src);
    device.destroy_buffer(dst);

    assert_no_validation_errors();
}
