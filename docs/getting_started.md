# Getting Started with ignis

This guide walks through rendering a triangle from scratch.

## Prerequisites

- Rust 1.85+ (edition 2024)
- Vulkan runtime (MoltenVK on macOS, GPU drivers on Linux/Windows)
- `glslc` (part of the Vulkan SDK) for compiling GLSL to SPIR-V

## Project Setup

Add ignis to your `Cargo.toml`:

```toml
[dependencies]
ignis = { path = "crates/ignis" }
ash = "0.38"
winit = "0.30"
```

## Shaders

Create `shaders/triangle.vert`:

```glsl
#version 450

const vec2 positions[3] = vec2[](
    vec2( 0.0, -0.5),
    vec2( 0.5,  0.5),
    vec2(-0.5,  0.5)
);

const vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragColor = colors[gl_VertexIndex];
}
```

Create `shaders/triangle.frag`:

```glsl
#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}
```

Compile them:

```bash
glslc shaders/triangle.vert -o shaders/triangle.vert.spv
glslc shaders/triangle.frag -o shaders/triangle.frag.spv
```

## Minimal Rendering Code

The core rendering loop uses these ignis types:

```rust
use ignis::prelude::*;
```

### 1. Create a Window and WSI

```rust
// WSI (Window System Integration) owns the Device and Swapchain.
let config = WSIConfig {
    app_name: "Triangle".into(),
    width: 1280,
    height: 720,
    vsync: true,
    ..Default::default()
};

let platform = WinitPlatform::new(&window);
let mut wsi = WSI::new(&platform, &config).expect("WSI init");
```

### 2. Load Shaders and Create a Program

```rust
let device = wsi.device().raw();

let vert = Shader::create(device, vk::ShaderStageFlags::VERTEX, &vert_spirv)
    .expect("vertex shader");
let frag = Shader::create(device, vk::ShaderStageFlags::FRAGMENT, &frag_spirv)
    .expect("fragment shader");

// A Program links shaders and auto-generates the pipeline layout
// from SPIR-V reflection (push constants, descriptor set layouts).
let mut program = Program::create(device, &[&vert, &frag])
    .expect("program");
```

### 3. Create Frame Resources

```rust
// FrameResources holds the pipeline compiler, render pass cache,
// and descriptor set caches. Created once, reused every frame.
let mut frame_resources = FrameResources::new(vk::PipelineCache::null());
```

### 4. The Frame Loop

```rust
// Acquire the next swapchain image
let frame_info = wsi.begin_frame().expect("begin_frame");

// Get a command buffer from the current frame's pool
let raw_cmd = wsi.device_mut()
    .request_command_buffer_raw(QueueType::Graphics)
    .expect("cmd alloc");
let mut cmd = CommandBuffer::from_raw(
    raw_cmd,
    CommandBufferType::Graphics,
    wsi.device().raw().clone(),
);
```

### 5. Record Commands via DrawContext

```rust
// Transition the swapchain image to COLOR_ATTACHMENT_OPTIMAL
cmd.image_barrier(&ImageBarrierInfo::undefined_to(
    frame_info.image,
    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
    vk::ImageAspectFlags::COLOR,
));

// DrawContext wraps the command buffer with automatic pipeline
// compilation and descriptor flushing.
let mut ctx = DrawContext::new(&mut cmd, wsi.device().raw(), &mut frame_resources);

let attachment = RenderingAttachment {
    view: frame_info.view,
    format: wsi.swapchain_format(),
    load_op: AttachmentLoadOp::Clear,
    store_op: AttachmentStoreOp::Store,
    clear_value: vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.1, 0.1, 0.1, 1.0],
        },
    },
    resolve_view: None,
};

// begin_rendering uses VK_KHR_dynamic_rendering (no VkRenderPass needed).
// It automatically sets viewport and scissor to match the extent.
ctx.begin_rendering(extent, &[attachment], None).expect("begin_rendering");

// Set pipeline state. The first draw() will compile the pipeline.
ctx.set_cull_mode(vk::CullModeFlags::NONE);
let vertex_layout = VertexInputLayout::default();
ctx.draw(&mut program, &vertex_layout, 3, 1, 0, 0).expect("draw");

ctx.end_rendering();

// Transition for presentation
cmd.image_barrier(&ImageBarrierInfo {
    image: frame_info.image,
    old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
    // ...barrier fields...
});
```

### 6. Submit and Present

```rust
wsi.end_frame(cmd.raw()).expect("end_frame");
```

`end_frame` submits the command buffer, presents the swapchain image, and handles resize.

## Key Concepts

### Lazy Pipeline Compilation

You never create pipelines explicitly. When you call `ctx.draw()`, the DrawContext:
1. Hashes the current state (program + topology + blend + depth + render pass + vertex layout)
2. Looks up the hash in the pipeline cache
3. On miss: compiles a new `VkPipeline` and caches it

Subsequent draws with the same state are just hash lookups.

### Descriptor Binding

Bind resources before drawing:

```rust
ctx.set_uniform_buffer(0, 0, ubo_buffer, 0, ubo_size);
ctx.set_texture(0, 1, texture_view, sampler);
ctx.set_storage_buffer(1, 0, ssbo_buffer, 0, ssbo_size);
```

The first number is the descriptor set index, the second is the binding index. Dirty sets are flushed automatically on `draw()`/`dispatch()`.

### State Presets

Common state configurations are available as presets:

```rust
ctx.set_opaque_state();              // back-cull, depth test+write, no blend
ctx.set_transparent_sprite_state();  // no cull, alpha blend, depth test only
ctx.set_additive_blend_state();      // no cull, additive blend
ctx.set_wireframe_state();           // line polygon mode
ctx.set_quad_state();                // no cull, no depth, no blend
```

### State Save/Restore

Temporarily change state (e.g., for a debug overlay) and restore:

```rust
let saved = ctx.save_state();
ctx.set_wireframe_state();
ctx.draw(&mut debug_program, &layout, 36, 1, 0, 0)?;
ctx.restore_state(&saved);
// Continue drawing with the original state
```

## Running with Validation

Always develop with validation layers enabled:

```bash
# Linux
VK_LAYER_PATH=/usr/share/vulkan/explicit_layer.d cargo run --example 01_triangle

# macOS (MoltenVK)
DYLD_LIBRARY_PATH=/opt/homebrew/lib \
VK_ICD_FILENAMES=/opt/homebrew/Cellar/molten-vk/1.4.1/etc/vulkan/icd.d/MoltenVK_icd.json \
VK_LAYER_PATH=/opt/homebrew/opt/vulkan-validationlayers/share/vulkan/explicit_layer.d \
cargo run --example 01_triangle
```

Zero validation errors is mandatory.

## Next Steps

- See `examples/02_textured_quad.rs` for texture loading and vertex buffers
- See `examples/03_multi_pass.rs` for multi-pass rendering
- See `examples/07_render_graph.rs` for automatic pass ordering
- See `docs/architecture.md` for the full architectural overview
