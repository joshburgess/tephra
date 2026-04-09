//! Compute pipeline example.
//!
//! Demonstrates compute shader dispatch with storage buffers using the
//! PipelineContext for pipeline caching and the dedicated compute queue.

fn main() {
    env_logger::init();

    println!("05_compute: compute pipeline dispatch demo");
    println!();
    println!("This example requires a compute shader (compute.comp.spv).");
    println!();
    println!("API usage pattern:");
    println!("  1. Create Device (discovers dedicated compute queue)");
    println!("  2. Load compute shader: ShaderManager::load(device, path, COMPUTE)");
    println!("  3. Create program: Program::create(device, &[&compute_shader])");
    println!("  4. Create storage buffers (input + output)");
    println!("  5. Allocate descriptor set, bind storage buffers");
    println!("  6. Get compute pipeline: PipelineContext::get_compute_pipeline(device, &program)");
    println!("  7. Record commands:");
    println!("     - cmd = device.request_command_buffer_raw(QueueType::Compute)");
    println!("     - cmd.bind_pipeline(COMPUTE, pipeline)");
    println!("     - cmd.bind_descriptor_sets(...)");
    println!("     - cmd.dispatch(group_x, group_y, group_z)");
    println!("  8. Submit to compute queue with fence");
    println!("  9. Wait for fence, read back results");
}
