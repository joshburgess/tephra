//! Phase 5 validation: two-pass rendering with manual barriers.
//!
//! Demonstrates rendering to an offscreen target, then sampling it
//! when rendering to the swapchain — using manual barrier placement
//! (before the render graph automates this in Phase 6).

fn main() {
    env_logger::init();

    println!("03_multi_pass: manual multi-pass rendering demo");
    println!();
    println!("This example requires pre-compiled SPIR-V shaders and a Vulkan runtime.");
    println!();
    println!("API usage pattern:");
    println!("  1. Create WSI (window + device + swapchain)");
    println!("  2. Create offscreen render target (Image + ImageView)");
    println!("  3. Pass 1: render scene to offscreen target");
    println!("     - cmd.image_barrier(UNDEFINED -> COLOR_ATTACHMENT)");
    println!("     - cmd.begin_render_pass(offscreen_rp, offscreen_fb)");
    println!("     - cmd.draw(...)");
    println!("     - cmd.end_render_pass()");
    println!("  4. Manual barrier: COLOR_ATTACHMENT -> SHADER_READ_ONLY");
    println!("  5. Pass 2: sample offscreen target, render to swapchain");
    println!("     - cmd.begin_render_pass(swapchain_rp, swapchain_fb)");
    println!("     - cmd.bind_descriptor_sets(offscreen_texture)");
    println!("     - cmd.draw(fullscreen_quad)");
    println!("     - cmd.end_render_pass()");
    println!("  6. Barrier: COLOR_ATTACHMENT -> PRESENT_SRC");
}
