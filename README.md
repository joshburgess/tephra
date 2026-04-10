# tephra

A [Granite](https://github.com/Themaister/Granite)-inspired mid-level Vulkan abstraction layer for Rust.

## Why "tephra"?

In volcanology, *tephra* is the fragmented material ejected during a volcanic eruption — raw, sharp pieces flung from the source. The Vulkan API is similar: powerful but chaotic, with memory management, synchronization, and pipeline state all exposed as sharp fragments you must assemble correctly.

Tephra (the library) catches those fragments mid-air and organizes them into usable pieces — frame contexts, deferred deletion, automatic pipeline compilation, descriptor caching — while remaining fundamentally Vulkan-level. You're still thinking in command buffers, render passes, and barrier transitions, not in materials, meshes, and lights. Structured volcanic debris, not yet compressed into solid rock.

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
| `tephra-core` | Context, device, frame management, buffers, images, samplers |
| `tephra-command` | Command buffer recording, linear allocators, barriers |
| `tephra-descriptors` | Descriptor set allocation, binding, caching |
| `tephra-pipeline` | Shader reflection, program linking, pipeline compilation |
| `tephra-wsi` | Windowing, swapchain, platform integration |
| `tephra-graph` | Render graph with automatic pass ordering and barriers |
| `tephra` | Umbrella crate re-exporting everything |

## Requirements

- Rust 1.85+ (edition 2024)
- Vulkan 1.2+ runtime

## License

MIT OR Apache-2.0
