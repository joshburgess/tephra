//! Phase 6 validation: render graph compilation and execution.
//!
//! Builds a multi-pass render graph with shadow mapping, main color pass,
//! and post-processing — then compiles it to verify automatic dependency
//! analysis, topological sorting, barrier placement, and dead pass culling.

use ash::vk;

use ignis::graph::{
    AttachmentInfo, BufferInfo, CompiledGraph, RenderGraph, SizeClass,
};
use ignis::graph::alias::find_alias_groups;
use ignis::graph::subpass_merge::find_merge_groups;

fn main() {
    env_logger::init();

    let mut graph = RenderGraph::new();

    // --- Shadow map pass ---
    let mut shadow = graph.add_pass("shadow");
    let shadow_map = shadow.add_depth_stencil_output(
        "shadow_map",
        AttachmentInfo {
            format: vk::Format::D32_SFLOAT,
            width: SizeClass::Absolute(2048),
            height: SizeClass::Absolute(2048),
            samples: vk::SampleCountFlags::TYPE_1,
        },
    );

    // --- G-buffer pass ---
    let mut gbuffer = graph.add_pass("gbuffer");
    let albedo = gbuffer.add_color_output(
        "albedo",
        AttachmentInfo::swapchain_relative(vk::Format::R8G8B8A8_UNORM, 1.0),
    );
    let normals = gbuffer.add_color_output(
        "normals",
        AttachmentInfo::swapchain_relative(vk::Format::R16G16B16A16_SFLOAT, 1.0),
    );
    let gbuffer_depth = gbuffer.add_depth_stencil_output(
        "gbuffer_depth",
        AttachmentInfo::swapchain_relative(vk::Format::D32_SFLOAT, 1.0),
    );

    // --- Lighting pass (reads shadow map + G-buffer) ---
    let mut lighting = graph.add_pass("lighting");
    lighting.add_texture_input(shadow_map);
    lighting.add_texture_input(albedo);
    lighting.add_texture_input(normals);
    lighting.add_depth_stencil_input(gbuffer_depth);
    let hdr = lighting.add_color_output(
        "hdr",
        AttachmentInfo::swapchain_relative(vk::Format::R16G16B16A16_SFLOAT, 1.0),
    );

    // --- Tone mapping (post-process) ---
    let mut tonemap = graph.add_pass("tonemap");
    tonemap.add_texture_input(hdr);
    let ldr = tonemap.add_color_output(
        "ldr",
        AttachmentInfo::swapchain_relative(vk::Format::B8G8R8A8_SRGB, 1.0),
    );

    // --- An intentionally dead pass (should be culled) ---
    let mut _debug = graph.add_pass("debug_overlay");
    _debug.add_storage_output(
        "debug_buffer",
        BufferInfo {
            size: 1024,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        },
    );

    // Set the final output
    graph.set_backbuffer_source(ldr);

    // --- Compile ---
    println!("Compiling render graph...");
    let compiled: CompiledGraph = graph.bake();

    println!("\nExecution order ({} steps):", compiled.step_count());
    for step in 0..compiled.step_count() {
        let kind = if compiled.step_is_compute(step) {
            "compute"
        } else {
            "graphics"
        };
        println!("  [{}] \"{}\" ({})", step, compiled.step_name(step), kind);
    }

    // --- Analyze subpass merge opportunities ---
    let merge_groups = find_merge_groups(&compiled);
    if merge_groups.is_empty() {
        println!("\nNo subpass merge opportunities detected.");
    } else {
        println!("\nSubpass merge groups:");
        for (i, group) in merge_groups.iter().enumerate() {
            println!("  Group {}: {} passes", i, group.steps.len());
        }
    }

    // --- Analyze resource aliasing ---
    let alias_groups = find_alias_groups(&compiled);
    if alias_groups.is_empty() {
        println!("No resource aliasing opportunities detected.");
    } else {
        println!("\nAlias groups:");
        for (i, group) in alias_groups.iter().enumerate() {
            println!("  Group {}: {} resources can share memory", i, group.resources.len());
        }
    }

    // Verify the dead pass was culled
    let has_debug = (0..compiled.step_count())
        .any(|s| compiled.step_name(s) == "debug_overlay");
    println!(
        "\nDead pass culling: debug_overlay {}",
        if has_debug { "PRESENT (bug!)" } else { "correctly culled" }
    );

    println!("\nRender graph compilation successful.");
}
