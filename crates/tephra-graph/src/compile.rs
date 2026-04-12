//! Graph compilation: dependency analysis, topological ordering, and barrier placement.
//!
//! The compiler takes a [`RenderGraph`] and produces a
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
fn topological_sort(deps: &[Vec<usize>], reachable: &[bool], num_passes: usize) -> Vec<usize> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::RenderGraph;
    use crate::resource::{AttachmentInfo, BufferInfo};

    fn color_attachment() -> AttachmentInfo {
        AttachmentInfo::absolute(vk::Format::R8G8B8A8_UNORM, 256, 256)
    }

    fn depth_attachment() -> AttachmentInfo {
        AttachmentInfo::absolute(vk::Format::D32_SFLOAT, 256, 256)
    }

    // -- needs_barrier --

    #[test]
    fn needs_barrier_different_access_types() {
        assert!(needs_barrier(
            AccessType::ColorOutput,
            AccessType::TextureInput
        ));
        assert!(needs_barrier(
            AccessType::StorageWrite,
            AccessType::StorageRead
        ));
        assert!(needs_barrier(
            AccessType::DepthStencilOutput,
            AccessType::DepthStencilInput
        ));
    }

    #[test]
    fn needs_barrier_same_write_access() {
        assert!(needs_barrier(
            AccessType::ColorOutput,
            AccessType::ColorOutput
        ));
        assert!(needs_barrier(
            AccessType::StorageWrite,
            AccessType::StorageWrite
        ));
    }

    #[test]
    fn no_barrier_same_read_access() {
        assert!(!needs_barrier(
            AccessType::TextureInput,
            AccessType::TextureInput
        ));
        assert!(!needs_barrier(
            AccessType::StorageRead,
            AccessType::StorageRead
        ));
        assert!(!needs_barrier(
            AccessType::DepthStencilInput,
            AccessType::DepthStencilInput
        ));
    }

    // -- access_info --

    #[test]
    fn access_info_color_output() {
        let (stage, acc, layout) = access_info(AccessType::ColorOutput);
        assert_eq!(stage, vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
        assert!(acc.contains(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE));
        assert_eq!(layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    }

    #[test]
    fn access_info_texture_input() {
        let (stage, _acc, layout) = access_info(AccessType::TextureInput);
        assert_eq!(stage, vk::PipelineStageFlags2::FRAGMENT_SHADER);
        assert_eq!(layout, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }

    #[test]
    fn access_info_storage_write() {
        let (_stage, acc, layout) = access_info(AccessType::StorageWrite);
        assert!(acc.contains(vk::AccessFlags2::SHADER_WRITE));
        assert_eq!(layout, vk::ImageLayout::GENERAL);
    }

    #[test]
    fn access_info_depth_stencil() {
        let (_, _, layout_out) = access_info(AccessType::DepthStencilOutput);
        let (_, _, layout_in) = access_info(AccessType::DepthStencilInput);
        assert_eq!(
            layout_out,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        );
        assert_eq!(layout_in, vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    }

    // -- Empty graph --

    #[test]
    fn empty_graph() {
        let graph = RenderGraph::new();
        let compiled = graph.bake();
        assert_eq!(compiled.step_count(), 0);
        assert!(compiled.backbuffer().is_none());
    }

    // -- Single pass --

    #[test]
    fn single_pass() {
        let mut graph = RenderGraph::new();
        let mut pass = graph.add_pass("only");
        let r = pass.add_color_output("out", color_attachment());
        graph.set_backbuffer_source(r);
        let compiled = graph.bake();

        assert_eq!(compiled.step_count(), 1);
        assert_eq!(compiled.step_name(0), "only");
    }

    // -- Linear chain: A → B → C --

    #[test]
    fn linear_chain_ordering() {
        let mut graph = RenderGraph::new();

        let mut a = graph.add_pass("A");
        let r1 = a.add_color_output("r1", color_attachment());

        let mut b = graph.add_pass("B");
        b.add_texture_input(r1);
        let r2 = b.add_color_output("r2", color_attachment());

        let mut c = graph.add_pass("C");
        c.add_texture_input(r2);
        let r3 = c.add_color_output("r3", color_attachment());

        graph.set_backbuffer_source(r3);
        let compiled = graph.bake();

        assert_eq!(compiled.step_count(), 3);
        assert_eq!(compiled.step_name(0), "A");
        assert_eq!(compiled.step_name(1), "B");
        assert_eq!(compiled.step_name(2), "C");
    }

    // -- Linear chain barrier placement --

    #[test]
    fn linear_chain_barriers() {
        let mut graph = RenderGraph::new();

        let mut a = graph.add_pass("A");
        let r1 = a.add_color_output("r1", color_attachment());

        let mut b = graph.add_pass("B");
        b.add_texture_input(r1);
        let r2 = b.add_color_output("r2", color_attachment());

        graph.set_backbuffer_source(r2);
        let compiled = graph.bake();

        // Step 0 (A): no barriers (first access to r1)
        assert!(compiled.barriers[0].is_empty());
        // Step 1 (B): barrier on r1 (ColorOutput → TextureInput)
        assert_eq!(compiled.barriers[1].len(), 1);
        assert_eq!(compiled.barriers[1][0].src_access, AccessType::ColorOutput);
        assert_eq!(compiled.barriers[1][0].dst_access, AccessType::TextureInput);
    }

    // -- Diamond dependency: A writes r1+r2, B reads r1, C reads r2, D reads both --

    #[test]
    fn diamond_dependency() {
        let mut graph = RenderGraph::new();

        let mut a = graph.add_pass("A");
        let r1 = a.add_color_output("r1", color_attachment());
        let r2 = a.add_color_output("r2", color_attachment());

        let mut b = graph.add_pass("B");
        b.add_texture_input(r1);
        let r3 = b.add_color_output("r3", color_attachment());

        let mut c = graph.add_pass("C");
        c.add_texture_input(r2);
        let r4 = c.add_color_output("r4", color_attachment());

        let mut d = graph.add_pass("D");
        d.add_texture_input(r3);
        d.add_texture_input(r4);
        let r5 = d.add_color_output("r5", color_attachment());

        graph.set_backbuffer_source(r5);
        let compiled = graph.bake();

        assert_eq!(compiled.step_count(), 4);

        // A must be first, D must be last
        assert_eq!(compiled.step_name(0), "A");
        assert_eq!(compiled.step_name(3), "D");

        // B and C must be in the middle (order between them is unspecified)
        let middle: Vec<&str> = (1..3).map(|i| compiled.step_name(i)).collect();
        assert!(middle.contains(&"B"));
        assert!(middle.contains(&"C"));
    }

    // -- Dead pass culling --

    #[test]
    fn dead_pass_culled() {
        let mut graph = RenderGraph::new();

        let mut live = graph.add_pass("live");
        let r = live.add_color_output("out", color_attachment());

        // This pass is not connected to the backbuffer
        let mut dead = graph.add_pass("dead");
        dead.add_color_output("dead_out", color_attachment());

        graph.set_backbuffer_source(r);
        let compiled = graph.bake();

        assert_eq!(compiled.step_count(), 1);
        assert_eq!(compiled.step_name(0), "live");
    }

    #[test]
    fn no_backbuffer_all_reachable() {
        let mut graph = RenderGraph::new();

        let mut a = graph.add_pass("A");
        a.add_color_output("r1", color_attachment());

        let mut b = graph.add_pass("B");
        b.add_color_output("r2", color_attachment());

        // No backbuffer set — all passes should be included
        let compiled = graph.bake();
        assert_eq!(compiled.step_count(), 2);
    }

    // -- Compute pass --

    #[test]
    fn compute_pass_flagged() {
        let mut graph = RenderGraph::new();

        let mut pass = graph.add_pass("comp");
        pass.set_compute();
        let r = pass.add_storage_output(
            "buf",
            BufferInfo {
                size: 1024,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            },
        );
        graph.set_backbuffer_source(r);
        let compiled = graph.bake();

        assert_eq!(compiled.step_count(), 1);
        assert!(compiled.step_is_compute(0));
    }

    // -- Depth/stencil dependency --

    #[test]
    fn depth_stencil_dependency() {
        let mut graph = RenderGraph::new();

        let mut depth_pass = graph.add_pass("depth_prepass");
        let depth = depth_pass.add_depth_stencil_output("depth", depth_attachment());

        let mut color_pass = graph.add_pass("color");
        color_pass.add_depth_stencil_input(depth);
        let color = color_pass.add_color_output("color", color_attachment());

        graph.set_backbuffer_source(color);
        let compiled = graph.bake();

        assert_eq!(compiled.step_count(), 2);
        assert_eq!(compiled.step_name(0), "depth_prepass");
        assert_eq!(compiled.step_name(1), "color");
    }

    // -- Read-modify-write (ColorInput) --

    #[test]
    fn color_input_read_modify_write() {
        let mut graph = RenderGraph::new();

        let mut a = graph.add_pass("A");
        let r = a.add_color_output("target", color_attachment());

        let mut b = graph.add_pass("B");
        b.add_color_input(r);
        // B modifies r in-place, no new output — use r as backbuffer
        // But we need some output to be the backbuffer. Let's add one.
        let out = b.add_color_output("final", color_attachment());

        graph.set_backbuffer_source(out);
        let compiled = graph.bake();

        assert_eq!(compiled.step_count(), 2);
        assert_eq!(compiled.step_name(0), "A");
        assert_eq!(compiled.step_name(1), "B");
    }

    // -- Multiple readers from one writer --

    #[test]
    fn multiple_readers() {
        let mut graph = RenderGraph::new();

        let mut writer = graph.add_pass("writer");
        let r = writer.add_color_output("shared", color_attachment());

        let mut reader1 = graph.add_pass("reader1");
        reader1.add_texture_input(r);
        let out1 = reader1.add_color_output("out1", color_attachment());

        let mut reader2 = graph.add_pass("reader2");
        reader2.add_texture_input(r);
        let out2 = reader2.add_color_output("out2", color_attachment());

        let mut merge = graph.add_pass("merge");
        merge.add_texture_input(out1);
        merge.add_texture_input(out2);
        let final_out = merge.add_color_output("final", color_attachment());

        graph.set_backbuffer_source(final_out);
        let compiled = graph.bake();

        assert_eq!(compiled.step_count(), 4);
        // Writer must come first
        assert_eq!(compiled.step_name(0), "writer");
        // Merge must come last
        assert_eq!(compiled.step_name(3), "merge");
    }

    // -- Transitive culling --

    #[test]
    fn transitive_culling() {
        let mut graph = RenderGraph::new();

        // Live chain: A → B (backbuffer)
        let mut a = graph.add_pass("A");
        let r = a.add_color_output("r", color_attachment());

        let mut b = graph.add_pass("B");
        b.add_texture_input(r);
        let out = b.add_color_output("out", color_attachment());

        // Dead chain: C → D (not connected to backbuffer)
        let mut c = graph.add_pass("C");
        let dead_r = c.add_color_output("dead_r", color_attachment());

        let mut d = graph.add_pass("D");
        d.add_texture_input(dead_r);
        d.add_color_output("dead_out", color_attachment());

        graph.set_backbuffer_source(out);
        let compiled = graph.bake();

        assert_eq!(compiled.step_count(), 2);
        let names: Vec<&str> = (0..compiled.step_count())
            .map(|i| compiled.step_name(i))
            .collect();
        assert!(names.contains(&"A"));
        assert!(names.contains(&"B"));
        assert!(!names.contains(&"C"));
        assert!(!names.contains(&"D"));
    }
}
