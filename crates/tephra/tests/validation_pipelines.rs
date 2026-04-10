//! Pipeline compilation validation tests.
//!
//! Compiles graphics and compute pipelines with various state combinations
//! and asserts zero validation errors.

mod validation_harness;

use ash::vk;

use tephra::command::barriers::ImageBarrierInfo;
use tephra::command::command_buffer::{CommandBuffer, CommandBufferType};
use tephra::core::context::QueueType;
use tephra::core::image::ImageCreateInfo;
use tephra::core::memory::{ImageDomain, MemoryDomain};
use tephra::pipeline::draw_context::{DrawContext, FrameResources, RenderingAttachment};
use tephra::pipeline::pipeline::VertexInputLayout;
use tephra::pipeline::program::Program;
use tephra::pipeline::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use tephra::pipeline::shader::Shader;

use validation_harness::*;

fn spirv_from_bytes(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len() % 4 == 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn graphics_pipeline_default_state() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("validation_pipeline_default");

    // Create offscreen render target
    let image_ci = ImageCreateInfo {
        width: 64,
        height: 64,
        depth: 1,
        format: vk::Format::R8G8B8A8_UNORM,
        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        image_type: vk::ImageType::TYPE_2D,
        initial_layout: vk::ImageLayout::UNDEFINED,
        domain: ImageDomain::Physical,
    };
    let image = device.create_image(&image_ci).expect("image");

    // Load shaders
    let vert_spirv = spirv_from_bytes(include_bytes!("../shaders/triangle.vert.spv"));
    let frag_spirv = spirv_from_bytes(include_bytes!("../shaders/triangle.frag.spv"));
    let vert = Shader::create(device.raw(), vk::ShaderStageFlags::VERTEX, &vert_spirv)
        .expect("vert shader");
    let frag = Shader::create(device.raw(), vk::ShaderStageFlags::FRAGMENT, &frag_spirv)
        .expect("frag shader");
    let mut program = Program::create(device.raw(), &[&vert, &frag]).expect("program");

    // Record
    device.begin_frame().expect("begin_frame");
    let raw_cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd alloc");
    let mut cmd =
        CommandBuffer::from_raw(raw_cmd, CommandBufferType::Graphics, device.raw().clone());

    let mut frame_resources = FrameResources::new(vk::PipelineCache::null());

    cmd.image_barrier(&ImageBarrierInfo::undefined_to(
        image.raw(),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        vk::ImageAspectFlags::COLOR,
    ));

    {
        let extent = vk::Extent2D {
            width: 64,
            height: 64,
        };
        let mut ctx = DrawContext::new(&mut cmd, device.raw(), &mut frame_resources);

        let attachment = RenderingAttachment {
            view: image.default_view(),
            format: vk::Format::R8G8B8A8_UNORM,
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            clear_value: vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            resolve_view: None,
        };

        ctx.begin_rendering(extent, &[attachment], None)
            .expect("begin_rendering");

        // Draw with default state (triggers pipeline compilation)
        ctx.set_cull_mode(vk::CullModeFlags::NONE);
        let vertex_layout = VertexInputLayout::default();
        ctx.draw(&mut program, &vertex_layout, 3, 1, 0, 0)
            .expect("draw");

        ctx.end_rendering();
    }

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

    // Cleanup
    // SAFETY: GPU is idle.
    unsafe {
        device.raw().device_wait_idle().ok();
    }
    program.destroy(device.raw());
    let mut vert = vert;
    let mut frag = frag;
    vert.destroy(device.raw());
    frag.destroy(device.raw());
    frame_resources.destroy(device.raw());
    device.destroy_image(image);

    assert_no_validation_errors();
}

#[test]
fn graphics_pipeline_multiple_state_combos() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("validation_pipeline_combos");

    let image_ci = ImageCreateInfo {
        width: 64,
        height: 64,
        depth: 1,
        format: vk::Format::R8G8B8A8_UNORM,
        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        image_type: vk::ImageType::TYPE_2D,
        initial_layout: vk::ImageLayout::UNDEFINED,
        domain: ImageDomain::Physical,
    };
    let image = device.create_image(&image_ci).expect("image");

    let vert_spirv = spirv_from_bytes(include_bytes!("../shaders/triangle.vert.spv"));
    let frag_spirv = spirv_from_bytes(include_bytes!("../shaders/triangle.frag.spv"));
    let vert =
        Shader::create(device.raw(), vk::ShaderStageFlags::VERTEX, &vert_spirv).expect("vert");
    let frag =
        Shader::create(device.raw(), vk::ShaderStageFlags::FRAGMENT, &frag_spirv).expect("frag");
    let mut program = Program::create(device.raw(), &[&vert, &frag]).expect("program");

    device.begin_frame().expect("begin_frame");
    let raw_cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd alloc");
    let mut cmd =
        CommandBuffer::from_raw(raw_cmd, CommandBufferType::Graphics, device.raw().clone());

    let mut frame_resources = FrameResources::new(vk::PipelineCache::null());

    cmd.image_barrier(&ImageBarrierInfo::undefined_to(
        image.raw(),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        vk::ImageAspectFlags::COLOR,
    ));

    {
        let extent = vk::Extent2D {
            width: 64,
            height: 64,
        };
        let mut ctx = DrawContext::new(&mut cmd, device.raw(), &mut frame_resources);

        let attachment = RenderingAttachment {
            view: image.default_view(),
            format: vk::Format::R8G8B8A8_UNORM,
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            clear_value: vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            resolve_view: None,
        };

        ctx.begin_rendering(extent, &[attachment], None)
            .expect("begin_rendering");

        let vertex_layout = VertexInputLayout::default();

        // Combo 1: no culling (default topology)
        ctx.set_cull_mode(vk::CullModeFlags::NONE);
        ctx.draw(&mut program, &vertex_layout, 3, 1, 0, 0)
            .expect("draw combo 1");

        // Combo 2: front culling
        ctx.set_cull_mode(vk::CullModeFlags::FRONT);
        ctx.draw(&mut program, &vertex_layout, 3, 1, 0, 0)
            .expect("draw combo 2");

        // Combo 3: alpha blending
        ctx.set_cull_mode(vk::CullModeFlags::NONE);
        ctx.set_transparent_sprite_state();
        ctx.set_depth_test(false, false);
        ctx.draw(&mut program, &vertex_layout, 3, 1, 0, 0)
            .expect("draw combo 3");

        // Combo 4: additive blending
        ctx.set_additive_blend_state();
        ctx.set_depth_test(false, false);
        ctx.draw(&mut program, &vertex_layout, 3, 1, 0, 0)
            .expect("draw combo 4");

        ctx.end_rendering();
    }

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

    // Cleanup
    // SAFETY: GPU is idle.
    unsafe {
        device.raw().device_wait_idle().ok();
    }
    program.destroy(device.raw());
    let mut vert = vert;
    let mut frag = frag;
    vert.destroy(device.raw());
    frag.destroy(device.raw());
    frame_resources.destroy(device.raw());
    device.destroy_image(image);

    assert_no_validation_errors();
}

#[test]
fn graphics_pipeline_with_depth() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("validation_pipeline_depth");

    let image_ci = ImageCreateInfo {
        width: 64,
        height: 64,
        depth: 1,
        format: vk::Format::R8G8B8A8_UNORM,
        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        image_type: vk::ImageType::TYPE_2D,
        initial_layout: vk::ImageLayout::UNDEFINED,
        domain: ImageDomain::Physical,
    };
    let image = device.create_image(&image_ci).expect("image");

    let depth_ci = ImageCreateInfo::depth_stencil(64, 64, vk::Format::D32_SFLOAT);
    let depth_image = device.create_image(&depth_ci).expect("depth image");

    let vert_spirv = spirv_from_bytes(include_bytes!("../shaders/triangle.vert.spv"));
    let frag_spirv = spirv_from_bytes(include_bytes!("../shaders/triangle.frag.spv"));
    let vert =
        Shader::create(device.raw(), vk::ShaderStageFlags::VERTEX, &vert_spirv).expect("vert");
    let frag =
        Shader::create(device.raw(), vk::ShaderStageFlags::FRAGMENT, &frag_spirv).expect("frag");
    let mut program = Program::create(device.raw(), &[&vert, &frag]).expect("program");

    device.begin_frame().expect("begin_frame");
    let raw_cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd alloc");
    let mut cmd =
        CommandBuffer::from_raw(raw_cmd, CommandBufferType::Graphics, device.raw().clone());

    let mut frame_resources = FrameResources::new(vk::PipelineCache::null());

    cmd.image_barrier(&ImageBarrierInfo::undefined_to(
        image.raw(),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        vk::ImageAspectFlags::COLOR,
    ));
    cmd.image_barrier(&ImageBarrierInfo::undefined_to(
        depth_image.raw(),
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        vk::ImageAspectFlags::DEPTH,
    ));

    {
        let extent = vk::Extent2D {
            width: 64,
            height: 64,
        };
        let mut ctx = DrawContext::new(&mut cmd, device.raw(), &mut frame_resources);

        let color_attachment = RenderingAttachment {
            view: image.default_view(),
            format: vk::Format::R8G8B8A8_UNORM,
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            clear_value: vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            resolve_view: None,
        };

        let depth_attachment = RenderingAttachment {
            view: depth_image.default_view(),
            format: vk::Format::D32_SFLOAT,
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::DontCare,
            clear_value: vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
            resolve_view: None,
        };

        ctx.begin_rendering(extent, &[color_attachment], Some(&depth_attachment))
            .expect("begin_rendering");

        // Draw with depth test enabled
        ctx.set_cull_mode(vk::CullModeFlags::NONE);
        ctx.set_depth_test(true, true);
        let vertex_layout = VertexInputLayout::default();
        ctx.draw(&mut program, &vertex_layout, 3, 1, 0, 0)
            .expect("draw with depth");

        ctx.end_rendering();
    }

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
        device.raw().device_wait_idle().ok();
    }

    program.destroy(device.raw());
    let mut vert = vert;
    let mut frag = frag;
    vert.destroy(device.raw());
    frag.destroy(device.raw());
    frame_resources.destroy(device.raw());
    device.destroy_image(image);
    device.destroy_image(depth_image);

    assert_no_validation_errors();
}

/// Draw with line-list topology (tests non-default topology pipeline compilation).
#[test]
fn graphics_pipeline_line_list_topology() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("validation_pipeline_line_list");

    let image_ci = ImageCreateInfo {
        width: 64,
        height: 64,
        depth: 1,
        format: vk::Format::R8G8B8A8_UNORM,
        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        image_type: vk::ImageType::TYPE_2D,
        initial_layout: vk::ImageLayout::UNDEFINED,
        domain: ImageDomain::Physical,
    };
    let image = device.create_image(&image_ci).expect("image");

    let vert_spirv = spirv_from_bytes(include_bytes!("../shaders/triangle.vert.spv"));
    let frag_spirv = spirv_from_bytes(include_bytes!("../shaders/triangle.frag.spv"));
    let vert =
        Shader::create(device.raw(), vk::ShaderStageFlags::VERTEX, &vert_spirv).expect("vert");
    let frag =
        Shader::create(device.raw(), vk::ShaderStageFlags::FRAGMENT, &frag_spirv).expect("frag");
    let mut program = Program::create(device.raw(), &[&vert, &frag]).expect("program");

    device.begin_frame().expect("begin_frame");
    let raw_cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd alloc");
    let mut cmd =
        CommandBuffer::from_raw(raw_cmd, CommandBufferType::Graphics, device.raw().clone());

    let mut frame_resources = FrameResources::new(vk::PipelineCache::null());

    cmd.image_barrier(&ImageBarrierInfo::undefined_to(
        image.raw(),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        vk::ImageAspectFlags::COLOR,
    ));

    {
        let extent = vk::Extent2D {
            width: 64,
            height: 64,
        };
        let mut ctx = DrawContext::new(&mut cmd, device.raw(), &mut frame_resources);

        let attachment = RenderingAttachment {
            view: image.default_view(),
            format: vk::Format::R8G8B8A8_UNORM,
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            clear_value: vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            resolve_view: None,
        };

        ctx.begin_rendering(extent, &[attachment], None)
            .expect("begin_rendering");

        ctx.set_cull_mode(vk::CullModeFlags::NONE);
        ctx.set_topology(vk::PrimitiveTopology::LINE_LIST);
        let vertex_layout = VertexInputLayout::default();
        // 2 vertices = 1 line segment
        ctx.draw(&mut program, &vertex_layout, 2, 1, 0, 0)
            .expect("draw line list");

        ctx.end_rendering();
    }

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
        device.raw().device_wait_idle().ok();
    }

    program.destroy(device.raw());
    let mut vert = vert;
    let mut frag = frag;
    vert.destroy(device.raw());
    frag.destroy(device.raw());
    frame_resources.destroy(device.raw());
    device.destroy_image(image);

    assert_no_validation_errors();
}

#[test]
fn compute_pipeline_compilation() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("validation_compute_pipeline");

    // Just compile and dispatch — tests the compute pipeline path
    let spirv = spirv_from_bytes(include_bytes!("../shaders/double.comp.spv"));
    let shader =
        Shader::create(device.raw(), vk::ShaderStageFlags::COMPUTE, &spirv).expect("shader");
    let mut program = Program::create(device.raw(), &[&shader]).expect("program");

    let size = 256u64;
    let buf_info = tephra::core::buffer::BufferCreateInfo {
        size,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        domain: MemoryDomain::Host,
    };
    let mut buf_a = device.create_buffer(&buf_info).expect("buf_a");
    let buf_b = device.create_buffer(&buf_info).expect("buf_b");

    if let Some(slice) = buf_a.mapped_slice_mut() {
        for (i, byte) in slice.iter_mut().enumerate().take(size as usize) {
            *byte = (i % 256) as u8;
        }
    }

    device.begin_frame().expect("begin_frame");
    let raw_cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd alloc");
    let mut cmd =
        CommandBuffer::from_raw(raw_cmd, CommandBufferType::Graphics, device.raw().clone());

    let mut frame_resources = FrameResources::new(vk::PipelineCache::null());

    {
        let mut ctx = DrawContext::new(&mut cmd, device.raw(), &mut frame_resources);
        ctx.set_storage_buffer(0, 0, buf_a.raw(), 0, size);
        ctx.set_storage_buffer(0, 1, buf_b.raw(), 0, size);
        ctx.dispatch(&mut program, 1, 1, 1).expect("dispatch");
    }

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

    // Cleanup
    // SAFETY: GPU is idle.
    unsafe {
        device.raw().device_wait_idle().ok();
    }
    program.destroy(device.raw());
    let mut shader = shader;
    shader.destroy(device.raw());
    frame_resources.destroy(device.raw());
    device.destroy_buffer(buf_a);
    device.destroy_buffer(buf_b);

    assert_no_validation_errors();
}
