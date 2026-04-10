//! Compute dispatch with shader variant management (headless).
//!
//! Demonstrates the `ShaderVariantRegistry` for managing shader permutations,
//! combined with a compute dispatch. Registers a shader template with defines,
//! stores compiled variants, and dispatches a compute shader that scales buffer
//! elements by a push-constant factor.
//!
//! Requires: compiled SPIR-V shaders `shaders/double.comp.spv` and `shaders/scale.comp.spv`.

use ash::vk;

use ignis::command::command_buffer::{CommandBuffer, CommandBufferType};
use ignis::core::buffer::BufferCreateInfo;
use ignis::core::context::{ContextConfig, QueueType};
use ignis::core::device::Device;
use ignis::core::memory::MemoryDomain;
use ignis::pipeline::draw_context::{DrawContext, FrameResources};
use ignis::pipeline::program::Program;
use ignis::pipeline::shader::Shader;
use ignis::pipeline::shader_variant::{DefineSet, ShaderTemplate, ShaderVariantRegistry};

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

    // --- Demonstrate ShaderVariantRegistry ---
    println!("=== Shader Variant Registry Demo ===\n");

    let mut registry = ShaderVariantRegistry::new();

    // Register a compute shader template with available defines
    let template = ShaderTemplate::new("shaders/scale.comp.glsl", vk::ShaderStageFlags::COMPUTE)
        .define_bool("USE_ABS")
        .define_int("WORKGROUP_SIZE", 64);
    let template_id = registry.register_template(template);
    println!("Registered template {:?}", template_id);

    // Create two different variant configurations
    let mut variant_a = DefineSet::new();
    variant_a.set_bool("USE_ABS", false);
    variant_a.set_int("WORKGROUP_SIZE", 64);

    let mut variant_b = DefineSet::new();
    variant_b.set_bool("USE_ABS", true);
    variant_b.set_int("WORKGROUP_SIZE", 128);

    println!("Variant A preamble:\n{}", variant_a.to_preamble());
    println!("Variant B preamble:\n{}", variant_b.to_preamble());

    let key_a = registry.variant_key(template_id, &variant_a);
    let key_b = registry.variant_key(template_id, &variant_b);
    println!("Variant A key hash: {:#018x}", key_a.define_hash);
    println!("Variant B key hash: {:#018x}", key_b.define_hash);
    assert_ne!(
        key_a.define_hash, key_b.define_hash,
        "Different defines must have different hashes"
    );

    // Store pre-compiled SPIR-V for variant A (in a real app, this would come from glslc)
    let spirv_a = spirv_from_bytes(include_bytes!("../shaders/double.comp.spv"));
    registry.store_compiled(key_a, spirv_a);
    assert!(registry.is_compiled(&key_a));
    assert!(!registry.is_compiled(&key_b));
    println!(
        "\nRegistry: {} templates, {} compiled variants",
        registry.template_count(),
        registry.compiled_count()
    );

    // --- Now dispatch a compute shader (same pattern as 05_compute) ---
    println!("\n=== Compute Dispatch ===\n");

    let config = ContextConfig {
        app_name: std::ffi::CString::new("ignis compute variants").unwrap(),
        app_version: vk::make_api_version(0, 1, 0, 0),
        enable_validation: cfg!(debug_assertions),
        required_instance_extensions: vec![],
    };

    let mut device = Device::new(&config).expect("failed to create device");
    println!("Device created (headless compute)");

    // Use the compiled variant SPIR-V from registry
    let spirv = registry
        .get_compiled(&key_a)
        .expect("variant A should be compiled");
    let comp_shader =
        Shader::create(device.raw(), vk::ShaderStageFlags::COMPUTE, spirv).expect("compute shader");
    let mut program = Program::create(device.raw(), &[&comp_shader]).expect("compute program");

    // Prepare buffers
    let input_data: Vec<f32> = (1..=ELEMENT_COUNT as u32).map(|i| i as f32).collect();
    let data_size = (ELEMENT_COUNT * std::mem::size_of::<f32>()) as u64;

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

    // Record and dispatch
    device.begin_frame().expect("begin_frame");

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

    // Read back and verify (the "double" shader multiplies by 2)
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
            "All {} results correct! (variant A: each element doubled)",
            ELEMENT_COUNT
        );
        println!("  Input:  {:?}", &input_data[..8]);
        println!("  Output: {:?}", &results[..8]);
    } else {
        println!("ERRORS detected in compute output");
    }

    // Cleanup
    // SAFETY: GPU is idle.
    unsafe {
        device.raw().device_wait_idle().ok();
    }
    program.destroy(device.raw());
    let mut comp_shader = comp_shader;
    comp_shader.destroy(device.raw());
    frame_resources.destroy(device.raw());
    device.destroy_buffer(input_buffer);
    device.destroy_buffer(output_buffer);

    println!("\nDone.");
}
