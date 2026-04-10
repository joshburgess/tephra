//! Compute pipeline example (headless).
//!
//! Dispatches a compute shader that doubles every element in a storage buffer.
//! Demonstrates: compute shader loading, storage buffer binding via DrawContext,
//! compute pipeline dispatch, and GPU→CPU readback.
//!
//! No window is created — this is a pure GPU compute workload.
//!
//! Requires: compiled SPIR-V shader `shaders/double.comp.spv`.
//! Compile with: `glslc shaders/double.comp -o shaders/double.comp.spv`

use ash::vk;

use tephra::command::command_buffer::{CommandBuffer, CommandBufferType};
use tephra::core::buffer::BufferCreateInfo;
use tephra::core::context::{ContextConfig, QueueType};
use tephra::core::device::Device;
use tephra::core::memory::MemoryDomain;
use tephra::pipeline::draw_context::{DrawContext, FrameResources};
use tephra::pipeline::program::Program;
use tephra::pipeline::shader::Shader;

const ELEMENT_COUNT: usize = 64;

fn spirv_from_bytes(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len() % 4 == 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    env_logger::init();

    // --- Create headless device (no window extensions) ---
    let config = ContextConfig {
        app_name: std::ffi::CString::new("tephra compute example").unwrap(),
        app_version: vk::make_api_version(0, 1, 0, 0),
        enable_validation: cfg!(debug_assertions),
        required_instance_extensions: vec![],
    };

    let mut device = Device::new(&config).expect("failed to create device");
    println!("Device created (headless compute)");

    // --- Prepare input data: [1.0, 2.0, 3.0, ..., 64.0] ---
    let input_data: Vec<f32> = (1..=ELEMENT_COUNT as u32).map(|i| i as f32).collect();
    let data_size = (ELEMENT_COUNT * std::mem::size_of::<f32>()) as u64;

    // --- Create storage buffers ---
    let input_info = BufferCreateInfo {
        size: data_size,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        domain: MemoryDomain::Host,
    };
    let mut input_buffer = device.create_buffer(&input_info).expect("input buffer");
    if let Some(slice) = input_buffer.mapped_slice_mut() {
        let bytes: &[u8] = bytemuck::cast_slice(&input_data);
        slice[..bytes.len()].copy_from_slice(bytes);
    }

    let output_info = BufferCreateInfo {
        size: data_size,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        domain: MemoryDomain::CachedHost,
    };
    let output_buffer = device.create_buffer(&output_info).expect("output buffer");

    println!(
        "Buffers created: {} elements ({} bytes each)",
        ELEMENT_COUNT, data_size
    );

    // --- Load compute shader & create program ---
    let comp_spirv = spirv_from_bytes(include_bytes!("../shaders/double.comp.spv"));
    let comp_shader = Shader::create(device.raw(), vk::ShaderStageFlags::COMPUTE, &comp_spirv)
        .expect("compute shader");
    let mut program = Program::create(device.raw(), &[&comp_shader]).expect("compute program");

    println!("Compute shader loaded, program created");

    // --- Begin frame (sets up command pools) ---
    device.begin_frame().expect("begin_frame");

    // --- Record compute dispatch ---
    let raw_cmd = device
        .request_command_buffer_raw(QueueType::Graphics)
        .expect("cmd alloc");
    let mut cmd =
        CommandBuffer::from_raw(raw_cmd, CommandBufferType::Graphics, device.raw().clone());

    let mut frame_resources = FrameResources::new(vk::PipelineCache::null());

    {
        let mut ctx = DrawContext::new(&mut cmd, device.raw(), &mut frame_resources);
        ctx.set_storage_buffer(0, 0, input_buffer.raw(), 0, data_size);
        ctx.set_storage_buffer(0, 1, output_buffer.raw(), 0, data_size);
        ctx.dispatch(&mut program, 1, 1, 1)
            .expect("dispatch failed");
    }

    println!("Dispatched compute: {} threads", ELEMENT_COUNT);

    // --- Submit and wait ---
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

    println!("GPU work complete");

    // --- Read back results ---
    let output_slice = output_buffer.mapped_slice().expect("output not mapped");
    let results: &[f32] = bytemuck::cast_slice(&output_slice[..data_size as usize]);

    let mut all_correct = true;
    for (i, &val) in results.iter().enumerate() {
        let expected = input_data[i] * 2.0;
        if (val - expected).abs() > f32::EPSILON {
            println!("  MISMATCH at [{}]: got {} expected {}", i, val, expected);
            all_correct = false;
        }
    }

    if all_correct {
        println!(
            "All {} results correct! (each element doubled)",
            ELEMENT_COUNT
        );
        println!("  Input:  {:?}", &input_data[..8]);
        println!("  Output: {:?}", &results[..8]);
    } else {
        println!("ERRORS detected in compute output");
    }

    // --- Cleanup ---
    // SAFETY: GPU is idle (we waited on fence above).
    unsafe {
        device.raw().device_wait_idle().ok();
    }

    program.destroy(device.raw());
    // Shader module can be destroyed immediately (not needed after pipeline creation)
    let mut comp_shader = comp_shader;
    comp_shader.destroy(device.raw());
    frame_resources.destroy(device.raw());
    device.destroy_buffer(input_buffer);
    device.destroy_buffer(output_buffer);

    println!("Cleanup complete");
}
