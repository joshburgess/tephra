# ignis

A Granite-inspired mid-level Vulkan abstraction layer for Rust.

Built on [`ash`](https://crates.io/crates/ash) + [`gpu-allocator`](https://crates.io/crates/gpu-allocator) + [`spirv-reflect`](https://crates.io/crates/spirv-reflect) + [`winit`](https://crates.io/crates/winit).

## Goals

- OpenGL/D3D11-style convenience with Vulkan's power and explicitness
- Automatic resource lifetime management via deferred deletion
- Hash-and-cache descriptor sets and pipelines
- Render graph with automatic barrier placement and subpass merging
- Zero-allocation transient data via per-frame bump allocators

## Crates

| Crate | Description |
|---|---|
| `ignis-core` | Context, device, frame management, buffers, images, samplers |
| `ignis-command` | Command buffer recording, linear allocators, barriers |
| `ignis-descriptors` | Descriptor set allocation, binding, caching |
| `ignis-pipeline` | Shader reflection, program linking, pipeline compilation |
| `ignis-wsi` | Windowing, swapchain, platform integration |
| `ignis-graph` | Render graph with automatic pass ordering and barriers |
| `ignis` | Umbrella crate re-exporting everything |

## Requirements

- Rust 1.85+ (edition 2024)
- Vulkan 1.2+ runtime

## License

MIT OR Apache-2.0
