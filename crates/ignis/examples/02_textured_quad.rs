//! Phase 3+4 validation: shader reflection and pipeline creation.
//!
//! Demonstrates SPIR-V shader loading, program linking with automatic
//! reflection, pipeline compilation, and descriptor set allocation.
//! Requires pre-compiled SPIR-V shaders in the `shaders/` directory.

fn main() {
    env_logger::init();

    println!("02_textured_quad: shader reflection + pipeline compilation demo");
    println!();
    println!("This example requires pre-compiled SPIR-V shaders.");
    println!("Place quad.vert.spv and quad.frag.spv in a shaders/ directory,");
    println!("then uncomment the rendering code below.");
    println!();
    println!("API usage pattern:");
    println!("  1. ShaderManager::load(device, path, stage)");
    println!("  2. Program::create(device, &[&vert, &frag])");
    println!("  3. PipelineContext::get_graphics_pipeline(...)");
    println!("  4. cmd.bind_pipeline() + cmd.draw()");

    // When shaders are available, the flow would be:
    //
    //   let mut shader_mgr = ShaderManager::new();
    //   let vert = shader_mgr.load(device, "shaders/quad.vert.spv", VERTEX)?;
    //   let frag = shader_mgr.load(device, "shaders/quad.frag.spv", FRAGMENT)?;
    //   let program = Program::create(device, &[vert, frag])?;
    //
    //   let mut ctx = PipelineContext::new(device, Some("pipeline_cache.bin"))?;
    //   let (pipeline, rp) = ctx.get_graphics_pipeline(
    //       device, &program, &state, &rp_info, 0, &vertex_layout,
    //   )?;
    //
    //   cmd.bind_pipeline(GRAPHICS, pipeline);
    //   cmd.bind_descriptor_sets(...);
    //   cmd.draw(6, 1, 0, 0);
}
