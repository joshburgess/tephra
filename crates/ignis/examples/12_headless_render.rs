//! Headless offscreen rendering with pixel readback.
//!
//! Renders a colored triangle into an offscreen image using dynamic rendering,
//! copies the result to a host-visible buffer, and verifies the pixels.
//! No window or swapchain is created.
//!
//! Demonstrates: headless device setup, offscreen render target creation,
//! dynamic rendering, image-to-buffer copy, and CPU readback.
//!
//! Requires: compiled SPIR-V shaders `shaders/triangle.{vert,frag}.spv`.

use ash::vk;

use ignis::command::barriers::ImageBarrierInfo;
use ignis::command::command_buffer::{CommandBuffer, CommandBufferType};
use ignis::core::buffer::BufferCreateInfo;
use ignis::core::context::{ContextConfig, QueueType};
use ignis::core::device::Device;
use ignis::core::image::ImageCreateInfo;
use ignis::core::memory::{ImageDomain, MemoryDomain};
use ignis::pipeline::draw_context::{DrawContext, FrameResources, RenderingAttachment};
use ignis::pipeline::pipeline::VertexInputLayout;
use ignis::pipeline::program::Program;
use ignis::pipeline::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use ignis::pipeline::shader::Shader;

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

fn spirv_from_bytes(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len() % 4 == 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    env_logger::init();

    // --- Headless device ---
    let config = ContextConfig {
        app_name: std::ffi::CString::new("ignis headless render").unwrap(),
        app_version: vk::make_api_version(0, 1, 0, 0),
        enable_validation: cfg!(debug_assertions),
        required_instance_extensions: vec![],
    };

    let mut device = Device::new(&config).expect("failed to create device");
    println!("Headless device created");

    // --- Create offscreen render target ---
    let image_ci = ImageCreateInfo {
        width: WIDTH,
        height: HEIGHT,
        depth: 1,
        format: FORMAT,
        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        image_type: vk::ImageType::TYPE_2D,
        initial_layout: vk::ImageLayout::UNDEFINED,
        domain: ImageDomain::Physical,
    };

    let image_handle = device
        .create_image(&image_ci)
        .expect("failed to create offscreen image");
    println!("Offscreen image created: {}x{} {:?}", WIDTH, HEIGHT, FORMAT);

    // --- Create readback buffer ---
    let pixel_size = 4u64; // RGBA8
    let readback_size = WIDTH as u64 * HEIGHT as u64 * pixel_size;
    let readback_info = BufferCreateInfo {
        size: readback_size,
        usage: vk::BufferUsageFlags::TRANSFER_DST,
        domain: MemoryDomain::CachedHost,
    };
    let readback_buffer = device
        .create_buffer(&readback_info)
        .expect("readback buffer");

    // --- Load shaders ---
    let vert_spirv = spirv_from_bytes(include_bytes!("../shaders/triangle.vert.spv"));
    let frag_spirv = spirv_from_bytes(include_bytes!("../shaders/triangle.frag.spv"));
    let vert_shader = Shader::create(device.raw(), vk::ShaderStageFlags::VERTEX, &vert_spirv)
        .expect("vertex shader");
    let frag_shader = Shader::create(device.raw(), vk::ShaderStageFlags::FRAGMENT, &frag_spirv)
        .expect("fragment shader");
    let mut program =
        Program::create(device.raw(), &[&vert_shader, &frag_shader]).expect("program");

    // --- Record commands ---
    device.begin_frame().expect("begin_frame");

    let raw_cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd alloc");
    let mut cmd =
        CommandBuffer::from_raw(raw_cmd, CommandBufferType::Graphics, device.raw().clone());

    let mut frame_resources = FrameResources::new(vk::PipelineCache::null());

    // Transition image: UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
    cmd.image_barrier(&ImageBarrierInfo::undefined_to(
        image_handle.raw(),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        vk::ImageAspectFlags::COLOR,
    ));

    // Draw triangle using DrawContext
    {
        let extent = vk::Extent2D {
            width: WIDTH,
            height: HEIGHT,
        };

        let mut ctx = DrawContext::new(&mut cmd, device.raw(), &mut frame_resources);

        let color_attachment = RenderingAttachment {
            view: image_handle.default_view(),
            format: FORMAT,
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            clear_value: vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            resolve_view: None,
        };

        ctx.begin_rendering(extent, &[color_attachment], None)
            .expect("begin_rendering");

        // The triangle shader has CW winding; disable culling so it renders.
        ctx.set_cull_mode(vk::CullModeFlags::NONE);

        let vertex_layout = VertexInputLayout::default();
        ctx.draw(&mut program, &vertex_layout, 3, 1, 0, 0)
            .expect("draw");

        ctx.end_rendering();
    }

    // Transition image: COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_SRC_OPTIMAL
    cmd.image_barrier(&ImageBarrierInfo {
        image: image_handle.raw(),
        old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        src_stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        src_access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        dst_stage: vk::PipelineStageFlags2::TRANSFER,
        dst_access: vk::AccessFlags2::TRANSFER_READ,
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

    // Copy image to readback buffer
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
            width: WIDTH,
            height: HEIGHT,
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
            &[region],
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

    // --- Read back pixels ---
    let pixels = readback_buffer.mapped_slice().expect("readback not mapped");
    let pixel_data = &pixels[..readback_size as usize];

    // Check center pixel (should be part of the triangle, non-black)
    let center_x = WIDTH / 2;
    let center_y = HEIGHT / 2;
    let center_offset = ((center_y * WIDTH + center_x) * 4) as usize;
    let r = pixel_data[center_offset];
    let g = pixel_data[center_offset + 1];
    let b = pixel_data[center_offset + 2];
    let a = pixel_data[center_offset + 3];
    println!(
        "Center pixel ({}, {}): RGBA = ({}, {}, {}, {})",
        center_x, center_y, r, g, b, a
    );

    // The triangle should have non-zero color at the center
    let has_color = r > 0 || g > 0 || b > 0;
    if has_color {
        println!("Triangle rendered successfully at center pixel");
    } else {
        println!("WARNING: Center pixel is black — triangle may not be visible");
    }

    // Check a corner pixel (should be the clear color: black)
    let corner_r = pixel_data[0];
    let corner_g = pixel_data[1];
    let corner_b = pixel_data[2];
    println!(
        "Corner pixel (0, 0): RGBA = ({}, {}, {}, {})",
        corner_r, corner_g, corner_b, pixel_data[3]
    );

    // Count non-black pixels to estimate triangle coverage
    let mut non_black = 0u32;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let offset = ((y * WIDTH + x) * 4) as usize;
            if pixel_data[offset] > 0 || pixel_data[offset + 1] > 0 || pixel_data[offset + 2] > 0 {
                non_black += 1;
            }
        }
    }
    let total = WIDTH * HEIGHT;
    let coverage = non_black as f64 / total as f64 * 100.0;
    println!(
        "Triangle coverage: {}/{} pixels ({:.1}%)",
        non_black, total, coverage
    );

    // --- Cleanup ---
    // SAFETY: GPU is idle.
    unsafe {
        device.raw().device_wait_idle().ok();
    }
    program.destroy(device.raw());
    let mut vert_shader = vert_shader;
    let mut frag_shader = frag_shader;
    vert_shader.destroy(device.raw());
    frag_shader.destroy(device.raw());
    frame_resources.destroy(device.raw());
    device.destroy_buffer(readback_buffer);
    device.destroy_image(image_handle);

    println!("\nDone.");
}
