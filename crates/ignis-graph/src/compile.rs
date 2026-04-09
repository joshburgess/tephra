//! Graph compilation: dependency analysis, topological ordering, and barrier placement.
//!
//! The compiler takes a [`RenderGraph`](crate::graph::RenderGraph) and produces a
//! [`CompiledGraph`] with a topologically sorted execution order (unreachable passes
//! culled) and per-pass barrier lists for correct synchronization.

use ash::vk;
use log::{debug, warn};

use crate::graph::RenderGraph;
use crate::pass::{AccessType, PassDeclaration};
use crate::resource::{ResourceDeclaration, ResourceHandle};

/// Barrier describing a resource state transition between passes.
#[derive(Debug, Clone)]
pub struct ResourceBarrier {
    /// The resource being transitioned.
    pub(crate) resource: ResourceHandle,
    /// Access type in the source (previous) pass.
    pub(crate) src_access: AccessType,
    /// Access type in the destination (current) pass.
    pub(crate) dst_access: AccessType,
}

/// A compiled render graph ready for execution.
///
/// Contains the topologically sorted pass order, per-pass barriers,
/// and all resource/pass metadata needed for execution.
pub struct CompiledGraph {
    /// Indices of passes in execution order (indices into `passes`).
    pub(crate) pass_order: Vec<usize>,
    /// Pre-barriers for each step in execution order.
    /// `barriers[i]` contains barriers to execute before `pass_order[i]`.
    pub(crate) barriers: Vec<Vec<ResourceBarrier>>,
    /// All pass declarations (consumed from the graph).
    pub(crate) passes: Vec<PassDeclaration>,
    /// All resource declarations.
    pub(crate) resources: Vec<ResourceDeclaration>,
    /// The backbuffer resource, if set.
    pub(crate) backbuffer: Option<ResourceHandle>,
}

impl CompiledGraph {
    /// Compile a render graph into execution order with barriers.
    ///
    /// Performs dependency analysis (RAW, WAW, WAR hazards), dead pass culling
    /// via backbuffer reachability, topological sorting, and barrier placement.
    pub(crate) fn compile(graph: RenderGraph) -> Self {
        let num_passes = graph.passes.len();
        let num_resources = graph.resources.len();

        let deps = build_dependencies(&graph.passes, num_resources);

        let reachable = find_reachable(&graph.passes, &deps, graph.backbuffer, num_passes);
        let reachable_count = reachable.iter().filter(|&&r| r).count();
        debug!(
            "Render graph: {}/{} passes reachable from backbuffer",
            reachable_count, num_passes
        );

        let pass_order = topological_sort(&deps, &reachable, num_passes);

        let barriers = place_barriers(&graph.passes, &pass_order, num_resources);

        for (step, &pass_idx) in pass_order.iter().enumerate() {
            debug!(
                "  step[{}]: \"{}\" ({} pre-barriers)",
                step,
                graph.passes[pass_idx].name,
                barriers[step].len(),
            );
        }

        CompiledGraph {
            pass_order,
            barriers,
            passes: graph.passes,
            resources: graph.resources,
            backbuffer: graph.backbuffer,
        }
    }

    /// Number of execution steps (passes in sorted order).
    pub fn step_count(&self) -> usize {
        self.pass_order.len()
    }

    /// Name of the pass at the given execution step.
    pub fn step_name(&self, step: usize) -> &str {
        &self.passes[self.pass_order[step]].name
    }

    /// Whether the pass at the given step is compute-only.
    pub fn step_is_compute(&self, step: usize) -> bool {
        self.passes[self.pass_order[step]].is_compute
    }

    /// The backbuffer resource handle, if set.
    pub fn backbuffer(&self) -> Option<ResourceHandle> {
        self.backbuffer
    }
}

/// Vulkan synchronization info for a given access type.
///
/// Returns `(pipeline_stage, access_flags, image_layout)`.
pub(crate) fn access_info(
    access: AccessType,
) -> (vk::PipelineStageFlags2, vk::AccessFlags2, vk::ImageLayout) {
    match access {
        AccessType::ColorOutput => (
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ),
        AccessType::ColorInput => (
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ),
        AccessType::DepthStencilOutput => (
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ),
        AccessType::DepthStencilInput => (
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        ),
        AccessType::TextureInput => (
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_READ,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ),
        AccessType::StorageRead => (
            vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_READ,
            vk::ImageLayout::GENERAL,
        ),
        AccessType::StorageWrite => (
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::ImageLayout::GENERAL,
        ),
        AccessType::AttachmentInput => (
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::INPUT_ATTACHMENT_READ,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ),
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build dependency adjacency list: `deps[i]` = passes that pass `i` depends on.
fn build_dependencies(passes: &[PassDeclaration], num_resources: usize) -> Vec<Vec<usize>> {
    let num_passes = passes.len();
    let mut deps: Vec<Vec<usize>> = vec![Vec::new(); num_passes];

    let mut last_writer: Vec<Option<usize>> = vec![None; num_resources];
    let mut readers_since_write: Vec<Vec<usize>> = vec![Vec::new(); num_resources];

    for (pass_idx, pass) in passes.iter().enumerate() {
        for access in &pass.accesses {
            let res = access.resource.index as usize;

            if access.access_type.is_write() {
                // WAW: depend on previous writer.
                if let Some(prev) = last_writer[res] {
                    if prev != pass_idx {
                        push_unique(&mut deps[pass_idx], prev);
                    }
                }
                // WAR: depend on all readers since the last write.
                for &reader in &readers_since_write[res] {
                    if reader != pass_idx {
                        push_unique(&mut deps[pass_idx], reader);
                    }
                }
                last_writer[res] = Some(pass_idx);
                readers_since_write[res].clear();
            } else {
                // RAW: depend on the last writer.
                if let Some(writer) = last_writer[res] {
                    if writer != pass_idx {
                        push_unique(&mut deps[pass_idx], writer);
                    }
                }
                readers_since_write[res].push(pass_idx);
            }
        }
    }

    deps
}

/// Walk backwards from the backbuffer to find all reachable passes.
fn find_reachable(
    passes: &[PassDeclaration],
    deps: &[Vec<usize>],
    backbuffer: Option<ResourceHandle>,
    num_passes: usize,
) -> Vec<bool> {
    let mut reachable = vec![false; num_passes];

    let Some(bb) = backbuffer else {
        // No backbuffer declared — assume all passes are needed.
        reachable.fill(true);
        return reachable;
    };

    // Seed with passes that write the backbuffer.
    let mut stack: Vec<usize> = Vec::new();
    for (pass_idx, pass) in passes.iter().enumerate() {
        for access in &pass.accesses {
            if access.resource == bb && access.access_type.is_write() {
                stack.push(pass_idx);
            }
        }
    }

    while let Some(pass_idx) = stack.pop() {
        if reachable[pass_idx] {
            continue;
        }
        reachable[pass_idx] = true;
        for &dep in &deps[pass_idx] {
            if !reachable[dep] {
                stack.push(dep);
            }
        }
    }

    reachable
}

/// Kahn's algorithm: topological sort of reachable passes.
fn topological_sort(
    deps: &[Vec<usize>],
    reachable: &[bool],
    num_passes: usize,
) -> Vec<usize> {
    let mut in_degree = vec![0u32; num_passes];
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); num_passes];

    for pass_idx in 0..num_passes {
        if !reachable[pass_idx] {
            continue;
        }
        for &dep in &deps[pass_idx] {
            if reachable[dep] {
                in_degree[pass_idx] += 1;
                successors[dep].push(pass_idx);
            }
        }
    }

    let mut queue: Vec<usize> = (0..num_passes)
        .filter(|&i| reachable[i] && in_degree[i] == 0)
        .collect();

    let mut order = Vec::with_capacity(num_passes);

    while let Some(pass_idx) = queue.pop() {
        order.push(pass_idx);
        for &succ in &successors[pass_idx] {
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                queue.push(succ);
            }
        }
    }

    let expected = reachable.iter().filter(|&&r| r).count();
    if order.len() != expected {
        warn!(
            "Render graph has a dependency cycle — {} of {} reachable passes emitted",
            order.len(),
            expected,
        );
    }

    order
}

/// Determine barriers needed between consecutive accesses to each resource.
fn place_barriers(
    passes: &[PassDeclaration],
    pass_order: &[usize],
    num_resources: usize,
) -> Vec<Vec<ResourceBarrier>> {
    let mut last_access: Vec<Option<AccessType>> = vec![None; num_resources];
    let mut barriers = Vec::with_capacity(pass_order.len());

    for &pass_idx in pass_order {
        let mut step_barriers = Vec::new();

        for access in &passes[pass_idx].accesses {
            let res = access.resource.index as usize;

            if let Some(src) = last_access[res] {
                if needs_barrier(src, access.access_type) {
                    step_barriers.push(ResourceBarrier {
                        resource: access.resource,
                        src_access: src,
                        dst_access: access.access_type,
                    });
                }
            }

            last_access[res] = Some(access.access_type);
        }

        barriers.push(step_barriers);
    }

    barriers
}

/// Whether a barrier is required between two consecutive accesses.
fn needs_barrier(src: AccessType, dst: AccessType) -> bool {
    if src != dst {
        return true;
    }
    // Same access type: barrier only needed if it writes (WAW hazard).
    src.is_write()
}

fn push_unique(vec: &mut Vec<usize>, val: usize) {
    if !vec.contains(&val) {
        vec.push(val);
    }
}
