//! Descriptor path validation tests.
//!
//! Exercises descriptor set allocation, binding table, caching, and compute
//! dispatch with bound descriptors. Asserts zero validation errors.

mod validation_harness;

use ash::vk;

use ignis::command::command_buffer::{CommandBuffer, CommandBufferType};
use ignis::core::buffer::BufferCreateInfo;
use ignis::core::context::QueueType;
use ignis::core::memory::MemoryDomain;
use ignis::pipeline::draw_context::{DrawContext, FrameResources};
use ignis::pipeline::program::Program;
use ignis::pipeline::shader::Shader;

use validation_harness::*;

fn spirv_from_bytes(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len() % 4 == 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn compute_dispatch_with_storage_buffers() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("validation_descriptor_compute");

    // Create storage buffers
    let size = 256u64;
    let input_info = BufferCreateInfo {
        size,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        domain: MemoryDomain::Host,
    };
    let mut input = device.create_buffer(&input_info).expect("input buffer");
    if let Some(slice) = input.mapped_slice_mut() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        slice[..bytes.len()].copy_from_slice(bytes);
    }

    let output_info = BufferCreateInfo {
        size,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        domain: MemoryDomain::CachedHost,
    };
    let output = device.create_buffer(&output_info).expect("output buffer");

    // Load compute shader
    let spirv = spirv_from_bytes(include_bytes!("../shaders/double.comp.spv"));
    let shader = Shader::create(device.raw(), vk::ShaderStageFlags::COMPUTE, &spirv)
        .expect("compute shader");
    let mut program = Program::create(device.raw(), &[&shader]).expect("program");

    // Record and dispatch
    device.begin_frame().expect("begin_frame");
    let raw_cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd alloc");
    let mut cmd = CommandBuffer::from_raw(
        raw_cmd,
        CommandBufferType::Graphics,
        device.raw().clone(),
    );

    let mut frame_resources = FrameResources::new(vk::PipelineCache::null());

    {
        let mut ctx = DrawContext::new(&mut cmd, device.raw(), &mut frame_resources);
        ctx.set_storage_buffer(0, 0, input.raw(), 0, size);
        ctx.set_storage_buffer(0, 1, output.raw(), 0, size);
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

    // Verify results
    let out_slice = output.mapped_slice().expect("output not mapped");
    let results: &[f32] = bytemuck::cast_slice(&out_slice[..size as usize]);
    for (i, &val) in results.iter().enumerate() {
        let expected = i as f32 * 2.0;
        assert!(
            (val - expected).abs() < f32::EPSILON,
            "mismatch at [{i}]: got {val}, expected {expected}"
        );
    }

    // Cleanup
    // SAFETY: GPU is idle.
    unsafe { device.raw().device_wait_idle().ok(); }
    program.destroy(device.raw());
    let mut shader = shader;
    shader.destroy(device.raw());
    frame_resources.destroy(device.raw());
    device.destroy_buffer(input);
    device.destroy_buffer(output);

    assert_no_validation_errors();
}

#[test]
fn multiple_dispatches_reuse_descriptors() {
    init_capture_logger();
    clear_errors();

    let mut device = create_headless_device("validation_descriptor_reuse");

    let size = 256u64;

    // Create buffers
    let buf_info = BufferCreateInfo {
        size,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        domain: MemoryDomain::Host,
    };
    let mut buf_a = device.create_buffer(&buf_info).expect("buf_a");
    let buf_b = device.create_buffer(&buf_info).expect("buf_b");
    let buf_c = device.create_buffer(&buf_info).expect("buf_c");

    if let Some(slice) = buf_a.mapped_slice_mut() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        slice[..bytes.len()].copy_from_slice(bytes);
    }

    // Load shader and program
    let spirv = spirv_from_bytes(include_bytes!("../shaders/double.comp.spv"));
    let shader = Shader::create(device.raw(), vk::ShaderStageFlags::COMPUTE, &spirv)
        .expect("shader");
    let mut program = Program::create(device.raw(), &[&shader]).expect("program");

    // Record multiple dispatches with different buffer bindings
    device.begin_frame().expect("begin_frame");
    let raw_cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd alloc");
    let mut cmd = CommandBuffer::from_raw(
        raw_cmd,
        CommandBufferType::Graphics,
        device.raw().clone(),
    );

    let mut frame_resources = FrameResources::new(vk::PipelineCache::null());

    {
        let mut ctx = DrawContext::new(&mut cmd, device.raw(), &mut frame_resources);

        // Dispatch 1: A -> B
        ctx.set_storage_buffer(0, 0, buf_a.raw(), 0, size);
        ctx.set_storage_buffer(0, 1, buf_b.raw(), 0, size);
        ctx.dispatch(&mut program, 1, 1, 1).expect("dispatch 1");

        // Dispatch 2: B -> C (reuses the cached pipeline, new descriptor set)
        ctx.set_storage_buffer(0, 0, buf_b.raw(), 0, size);
        ctx.set_storage_buffer(0, 1, buf_c.raw(), 0, size);
        ctx.dispatch(&mut program, 1, 1, 1).expect("dispatch 2");
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

    // Verify C = A * 4 (doubled twice)
    let c_slice = buf_c.mapped_slice().expect("buf_c not mapped");
    let results: &[f32] = bytemuck::cast_slice(&c_slice[..size as usize]);
    for (i, &val) in results.iter().enumerate() {
        let expected = i as f32 * 4.0;
        assert!(
            (val - expected).abs() < f32::EPSILON,
            "chain mismatch at [{i}]: got {val}, expected {expected}"
        );
    }

    // Cleanup
    // SAFETY: GPU is idle.
    unsafe { device.raw().device_wait_idle().ok(); }
    program.destroy(device.raw());
    let mut shader = shader;
    shader.destroy(device.raw());
    frame_resources.destroy(device.raw());
    device.destroy_buffer(buf_a);
    device.destroy_buffer(buf_b);
    device.destroy_buffer(buf_c);

    assert_no_validation_errors();
}
