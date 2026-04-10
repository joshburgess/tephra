# ignis Architecture

ignis is a mid-level Vulkan abstraction layer inspired by [Granite](https://github.com/Themaister/Granite)'s `vulkan/` layer. It sits between raw Vulkan and a full engine, providing OpenGL/D3D11-style convenience while retaining Vulkan's explicit control.

## Crate Dependency Graph

```
ignis (umbrella)
  |
  +-- ignis-graph         Render graph
  |     +-- ignis-pipeline
  |     +-- ignis-command
  |     +-- ignis-core
  |
  +-- ignis-wsi           Window system integration
  |     +-- ignis-core
  |
  +-- ignis-pipeline      Shaders, programs, pipelines
  |     +-- ignis-command
  |     +-- ignis-descriptors
  |     +-- ignis-core
  |
  +-- ignis-command        Command buffers, barriers
  |     +-- ignis-core
  |
  +-- ignis-descriptors   Descriptor sets, binding
  |     +-- ignis-core
  |
  +-- ignis-core          Context, device, memory, images, buffers
```

Dependencies flow strictly downward. `ignis-core` has no ignis dependencies.

## Key Design Decisions

### Frame Overlap

The engine uses a ring buffer of `FRAME_OVERLAP` (default 2) frame contexts. Each frame context owns:

- A `VkFence` for CPU-GPU synchronization
- Per-queue command pools (graphics, optional compute/transfer)
- A deferred deletion queue
- A linear allocator pool for transient per-frame data

`begin_frame()` advances the ring index, waits on the old fence, flushes deferred deletions, and resets command pools.

### Deferred Deletion

Resources are never destroyed immediately. When you call `device.destroy_buffer(handle)`, the buffer and its allocation are placed in the current frame's `DeletionQueue`. They are freed when the fence for that frame context signals (i.e., after `FRAME_OVERLAP` frames).

This eliminates use-after-free hazards without requiring reference counting on the hot path.

### Descriptor Set Model

Descriptors use a slot-based binding table:

1. **BindingTable** tracks which resources are bound at each (set, binding) pair and maintains per-set dirty flags.
2. On `draw()`/`dispatch()`, dirty sets are hashed and looked up in the **DescriptorSetCache**.
3. Cache misses allocate from a per-set **DescriptorSetAllocator** and write descriptors.
4. **Push descriptors** (`VK_KHR_push_descriptor`) skip allocation entirely.
5. **Bindless** tables use `VK_EXT_descriptor_indexing` with a persistent large set.

### Pipeline Compilation

Pipelines are compiled lazily on first use:

1. The **DrawContext** collects pipeline state (topology, blend, depth, etc.) and the current **Program**.
2. On `draw()`, the state is hashed into a pipeline key.
3. The **PipelineCompiler** looks up the key in an `FxHashMap`. Misses trigger `vkCreateGraphicsPipelines`.
4. A `VkPipelineCache` provides disk persistence for compiled pipelines.
5. Optional **FossilizeRecorder** captures pipeline state for cross-run pre-warming.

The separation of `StaticPipelineState` (hashed, baked into the key) from dynamic state (viewport, scissor, depth bias) minimizes pipeline permutations.

### Command Buffer Lifecycle

```
begin_frame()
  |
  v
request_command_buffer_raw(QueueType)
  |  -- allocates from the current frame's pool, begins recording
  v
CommandBuffer::from_raw(raw, type, device)
  |  -- typed wrapper for recording
  v
DrawContext::new(&mut cmd, device, &mut resources)
  |  -- high-level recording: set state, bind resources, draw/dispatch
  v
submit_command_buffer(cmd, queue, waits, signals, fence)
  |  -- ends recording, submits to the queue
  v
end_frame()
```

### Render Graph

The render graph automates barrier placement and execution ordering:

1. **Declare** passes with their resource inputs/outputs using the graph builder.
2. **Compile**: topological sort, dead-pass culling, barrier insertion.
3. **Subpass merge**: adjacent passes with attachment I/O can be merged into Vulkan subpasses (important for tile-based GPUs).
4. **Resource aliasing**: non-overlapping transient resources share memory.
5. **Execute**: the graph executor records commands, inserts barriers, and handles resource transitions.

### WSI Integration

The **WSI** module owns both the `Device` and the `Swapchain`. The frame loop:

```
wsi.begin_frame()           -- acquires swapchain image, calls device.begin_frame()
  |
  v
record commands, draw       -- user code
  |
  v
wsi.end_frame()             -- submits, presents, handles resize
```

Swapchain recreation (on `VK_ERROR_OUT_OF_DATE_KHR` or `VK_SUBOPTIMAL_KHR`) is handled automatically.

## Module Reference

| Module | Purpose |
|--------|---------|
| `core::context` | Vulkan instance, device, queues |
| `core::device` | Central abstraction wrapping context + frame management |
| `core::frame_context` | Frame ring buffer, deferred deletion |
| `core::buffer` / `core::image` | Resource creation parameters |
| `core::handles` | Owned `BufferHandle` / `ImageHandle` with metadata |
| `core::sampler` | Sampler cache with stock samplers |
| `core::memory` | Memory domain abstraction over gpu-allocator |
| `core::sync` | Semaphore/fence pools, timeline semaphores |
| `command::command_buffer` | Typed command recording |
| `command::barriers` | Pipeline barrier helpers |
| `command::state` | `StaticPipelineState` for pipeline key hashing |
| `descriptors::binding_table` | Slot-based descriptor tracking |
| `descriptors::set_allocator` | Per-set descriptor pool management |
| `descriptors::cache` | Hash-and-cache descriptor set reuse |
| `descriptors::bindless` | Large bindless descriptor tables |
| `pipeline::shader` | SPIR-V loading and reflection |
| `pipeline::program` | Multi-stage shader linking |
| `pipeline::pipeline` | Lazy pipeline compilation and caching |
| `pipeline::draw_context` | High-level draw API (Granite-style) |
| `wsi::wsi` | Swapchain + device ownership |
| `wsi::platform` | winit platform integration |
| `graph` | Render graph builder, compiler, executor |
