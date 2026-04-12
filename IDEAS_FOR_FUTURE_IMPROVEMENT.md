# Ideas for Future Improvement

Minor gaps and optional enhancements left over after the initial build-out. None are blocking; all are nice-to-haves.

## Bevy integration plugin

A `BevyTephraPlugin` that exposes `WSI` + `Device` as Bevy resources, extracts ECS data (transforms, meshes, materials) into a render world, and submits draw calls through tephra command buffers. Tagged as a stretch goal in the original plan.

## Descriptor update templates

The descriptor cache currently writes via `vkUpdateDescriptorSets`. Switching to `vkUpdateDescriptorSetWithTemplate` would cut per-set update cost in the hot path. Requires creating a `VkDescriptorUpdateTemplate` per layout and caching it alongside the layout.

## Enforce `#![deny(missing_docs)]`

Public items already carry doc comments, but the lint isn't enabled. Turn it on per-crate to prevent regressions.

## MSAA resolve in the render graph

`PassBuilder::add_resolve_output()` isn't implemented. Needed for MSAA targets that resolve into a single-sample attachment as part of a render pass.

## Swapchain resize stress testing

WSI handles `VK_ERROR_OUT_OF_DATE_KHR` / `VK_SUBOPTIMAL_KHR` and recreates the swapchain, but it hasn't been hammered with rapid resizing. Worth a dedicated stress test.
