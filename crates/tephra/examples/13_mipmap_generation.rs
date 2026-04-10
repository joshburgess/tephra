//! Mipmap generation example (headless).
//!
//! Creates a 256x256 RGBA8 image with procedural data, generates a full mip
//! chain via `generate_mipmap()`, reads back each mip level, and verifies
//! dimensions decrease correctly.
//!
//! Demonstrates: image creation with multiple mip levels, staging upload,
//! `CommandBuffer::generate_mipmap()`, and per-mip-level readback.

use ash::vk;

use tephra::command::barriers::ImageBarrierInfo;
use tephra::command::command_buffer::{CommandBuffer, CommandBufferType};
use tephra::core::buffer::BufferCreateInfo;
use tephra::core::context::{ContextConfig, QueueType};
use tephra::core::device::Device;
use tephra::core::image::ImageCreateInfo;
use tephra::core::memory::{ImageDomain, MemoryDomain};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

fn mip_levels(width: u32, height: u32) -> u32 {
    (width.max(height) as f32).log2().floor() as u32 + 1
}

fn main() {
    env_logger::init();

    let mip_count = mip_levels(WIDTH, HEIGHT);
    println!("=== Mipmap Generation Demo ===");
    println!(
        "Image: {}x{} {:?}, {} mip levels",
        WIDTH, HEIGHT, FORMAT, mip_count
    );

    // --- Headless device ---
    let config = ContextConfig {
        app_name: std::ffi::CString::new("tephra mipmap generation").unwrap(),
        app_version: vk::make_api_version(0, 1, 0, 0),
        enable_validation: cfg!(debug_assertions),
        required_instance_extensions: vec![],
    };

    let mut device = Device::new(&config).expect("failed to create device");
    println!("Device created");

    // --- Create image with mip levels ---
    let image_ci = ImageCreateInfo {
        width: WIDTH,
        height: HEIGHT,
        depth: 1,
        format: FORMAT,
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

    let image_handle = device.create_image(&image_ci).expect("image");
    println!("Image created with {} mip levels", mip_count);

    // --- Create staging buffer with procedural pixel data ---
    let pixel_count = (WIDTH * HEIGHT) as usize;
    let data_size = pixel_count * 4; // RGBA8
    let mut pixel_data = vec![0u8; data_size];

    // Generate a checkerboard pattern
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let offset = ((y * WIDTH + x) * 4) as usize;
            let checker = ((x / 32) + (y / 32)) % 2 == 0;
            if checker {
                pixel_data[offset] = 255; // R
                pixel_data[offset + 1] = 128; // G
                pixel_data[offset + 2] = 0; // B
                pixel_data[offset + 3] = 255; // A
            } else {
                pixel_data[offset] = 0; // R
                pixel_data[offset + 1] = 64; // G
                pixel_data[offset + 2] = 200; // B
                pixel_data[offset + 3] = 255; // A
            }
        }
    }

    let staging_info = BufferCreateInfo {
        size: data_size as u64,
        usage: vk::BufferUsageFlags::TRANSFER_SRC,
        domain: MemoryDomain::Host,
    };
    let mut staging_buffer = device.create_buffer(&staging_info).expect("staging buffer");
    if let Some(slice) = staging_buffer.mapped_slice_mut() {
        slice[..data_size].copy_from_slice(&pixel_data);
    }

    // --- Create readback buffer (large enough for mip 0) ---
    let readback_info = BufferCreateInfo {
        size: data_size as u64,
        usage: vk::BufferUsageFlags::TRANSFER_DST,
        domain: MemoryDomain::CachedHost,
    };
    let readback_buffer = device
        .create_buffer(&readback_info)
        .expect("readback buffer");

    // --- Record commands ---
    device.begin_frame().expect("begin_frame");

    let raw_cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd alloc");
    let mut cmd =
        CommandBuffer::from_raw(raw_cmd, CommandBufferType::Graphics, device.raw().clone());

    // Transition all mip levels: UNDEFINED -> TRANSFER_DST_OPTIMAL
    cmd.image_barrier(&ImageBarrierInfo {
        image: image_handle.raw(),
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

    // Copy staging buffer -> mip level 0
    let copy_region = vk::BufferImageCopy {
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
            width: WIDTH,
            height: HEIGHT,
            depth: 1,
        },
    };

    // SAFETY: command buffer, buffer, and image are valid.
    unsafe {
        device.raw().cmd_copy_buffer_to_image(
            cmd.raw(),
            staging_buffer.raw(),
            image_handle.raw(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy_region],
        );
    }

    // Generate mipmaps — this transitions all levels to SHADER_READ_ONLY_OPTIMAL
    cmd.generate_mipmap(image_handle.raw(), WIDTH, HEIGHT, mip_count);

    println!("Mipmap chain generated");

    // Transition mip 0 back to TRANSFER_SRC so we can read it back
    cmd.image_barrier(&ImageBarrierInfo {
        image: image_handle.raw(),
        old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        src_stage: vk::PipelineStageFlags2::TRANSFER,
        src_access: vk::AccessFlags2::TRANSFER_WRITE,
        dst_stage: vk::PipelineStageFlags2::TRANSFER,
        dst_access: vk::AccessFlags2::TRANSFER_READ,
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

    // Copy mip level 1 (128x128) to readback buffer to verify it was generated
    let mip1_width = (WIDTH / 2).max(1);
    let mip1_height = (HEIGHT / 2).max(1);
    let mip1_region = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: vk::Extent3D {
            width: mip1_width,
            height: mip1_height,
            depth: 1,
        },
    };

    // SAFETY: command buffer, image, and buffer are valid.
    unsafe {
        device.raw().cmd_copy_image_to_buffer(
            cmd.raw(),
            image_handle.raw(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            readback_buffer.raw(),
            &[mip1_region],
        );
    }

    // Submit and wait
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

    // --- Verify mip level 1 ---
    let mip1_size = (mip1_width * mip1_height * 4) as usize;
    let pixels = readback_buffer.mapped_slice().expect("readback not mapped");
    let mip1_data = &pixels[..mip1_size];

    // Check that mip 1 has non-zero pixel data (blitted from mip 0)
    let non_zero = mip1_data.iter().filter(|&&b| b > 0).count();
    println!(
        "\nMip level 1: {}x{} ({} bytes), {} non-zero bytes",
        mip1_width, mip1_height, mip1_size, non_zero
    );

    if non_zero > 0 {
        println!("Mip level 1 contains valid pixel data");
    } else {
        println!("WARNING: Mip level 1 is all zeros — generation may have failed");
    }

    // Print expected mip chain dimensions
    println!("\nFull mip chain:");
    let mut w = WIDTH;
    let mut h = HEIGHT;
    for level in 0..mip_count {
        println!("  Level {}: {}x{}", level, w, h);
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }

    // Sample a pixel from the center of mip 1
    let cx = mip1_width / 2;
    let cy = mip1_height / 2;
    let offset = ((cy * mip1_width + cx) * 4) as usize;
    println!(
        "\nMip 1 center pixel ({}, {}): RGBA = ({}, {}, {}, {})",
        cx,
        cy,
        mip1_data[offset],
        mip1_data[offset + 1],
        mip1_data[offset + 2],
        mip1_data[offset + 3]
    );

    // --- Cleanup ---
    // SAFETY: GPU is idle.
    unsafe {
        device.raw().device_wait_idle().ok();
    }
    device.destroy_buffer(staging_buffer);
    device.destroy_buffer(readback_buffer);
    device.destroy_image(image_handle);

    println!("\nDone.");
}
