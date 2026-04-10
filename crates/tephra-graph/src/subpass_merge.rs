//! Subpass merging for tile-based deferred renderers (TBDR).
//!
//! Identifies consecutive passes that can be merged into a single
//! `VkRenderPass` with multiple subpasses, critical for mobile and
//! Apple Silicon GPUs.

use log::debug;

use crate::compile::CompiledGraph;
use crate::pass::AccessType;

/// A group of consecutive passes merged into a single `VkRenderPass`.
#[derive(Debug, Clone)]
pub struct MergeGroup {
    /// Indices into `CompiledGraph::pass_order` for passes in this group.
    pub steps: Vec<usize>,
}

/// Analyze compiled pass order for subpass merge opportunities.
///
/// Two consecutive graphics passes can merge when the second reads an
/// attachment written by the first via [`AccessType::AttachmentInput`]
/// (subpass input attachment). Both passes must be graphics (not compute).
pub fn find_merge_groups(graph: &CompiledGraph) -> Vec<MergeGroup> {
    if graph.pass_order.is_empty() {
        return Vec::new();
    }

    let mut groups: Vec<MergeGroup> = Vec::new();
    let mut current_group = MergeGroup { steps: vec![0] };

    for step in 1..graph.pass_order.len() {
        let prev_pass_idx = graph.pass_order[step - 1];
        let curr_pass_idx = graph.pass_order[step];
        let prev_pass = &graph.passes[prev_pass_idx];
        let curr_pass = &graph.passes[curr_pass_idx];

        // Both must be graphics passes.
        if prev_pass.is_compute || curr_pass.is_compute {
            flush_group(&mut groups, &mut current_group);
            current_group = MergeGroup { steps: vec![step] };
            continue;
        }

        // The current pass must read (via AttachmentInput) a resource that the
        // previous pass wrote.
        let can_merge = curr_pass.accesses.iter().any(|curr_acc| {
            curr_acc.access_type == AccessType::AttachmentInput
                && prev_pass.accesses.iter().any(|prev_acc| {
                    prev_acc.resource == curr_acc.resource && prev_acc.access_type.is_write()
                })
        });

        if can_merge {
            current_group.steps.push(step);
        } else {
            flush_group(&mut groups, &mut current_group);
            current_group = MergeGroup { steps: vec![step] };
        }
    }

    flush_group(&mut groups, &mut current_group);

    for group in &groups {
        let names: Vec<&str> = group
            .steps
            .iter()
            .map(|&s| graph.passes[graph.pass_order[s]].name.as_str())
            .collect();
        debug!("Subpass merge group: {:?}", names);
    }

    groups
}

fn flush_group(groups: &mut Vec<MergeGroup>, current: &mut MergeGroup) {
    if current.steps.len() > 1 {
        groups.push(current.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::RenderGraph;
    use crate::resource::AttachmentInfo;
    use ash::vk;

    fn color_attachment() -> AttachmentInfo {
        AttachmentInfo::absolute(vk::Format::R8G8B8A8_UNORM, 256, 256)
    }

    #[test]
    fn empty_graph_no_merge() {
        let graph = RenderGraph::new();
        let compiled = graph.bake();
        let groups = find_merge_groups(&compiled);
        assert!(groups.is_empty());
    }

    #[test]
    fn single_pass_no_merge() {
        let mut graph = RenderGraph::new();
        let mut p = graph.add_pass("P");
        let r = p.add_color_output("out", color_attachment());
        graph.set_backbuffer_source(r);
        let compiled = graph.bake();

        let groups = find_merge_groups(&compiled);
        assert!(groups.is_empty());
    }

    #[test]
    fn texture_input_no_merge() {
        // TextureInput is NOT AttachmentInput — should not merge
        let mut graph = RenderGraph::new();

        let mut a = graph.add_pass("A");
        let r = a.add_color_output("r", color_attachment());

        let mut b = graph.add_pass("B");
        b.add_texture_input(r);
        let out = b.add_color_output("out", color_attachment());

        graph.set_backbuffer_source(out);
        let compiled = graph.bake();

        let groups = find_merge_groups(&compiled);
        assert!(groups.is_empty());
    }

    #[test]
    fn attachment_input_merges() {
        // B reads A's output via AttachmentInput — should merge
        let mut graph = RenderGraph::new();

        let mut a = graph.add_pass("A");
        let r = a.add_color_output("r", color_attachment());

        let mut b = graph.add_pass("B");
        b.add_attachment_input(r);
        let out = b.add_color_output("out", color_attachment());

        graph.set_backbuffer_source(out);
        let compiled = graph.bake();

        let groups = find_merge_groups(&compiled);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].steps.len(), 2);
    }

    #[test]
    fn compute_pass_breaks_merge() {
        // A (graphics) → B (compute) → C (graphics with AttachmentInput from A)
        // Compute break should prevent merging
        let mut graph = RenderGraph::new();

        let mut a = graph.add_pass("A");
        let r1 = a.add_color_output("r1", color_attachment());

        let mut b = graph.add_pass("B");
        b.set_compute();
        b.add_storage_output(
            "buf",
            crate::resource::BufferInfo {
                size: 1024,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            },
        );
        // B reads r1 as storage to create a dependency so it's not culled
        b.add_texture_input(r1);
        let buf = b.add_storage_output(
            "buf2",
            crate::resource::BufferInfo {
                size: 1024,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            },
        );

        let mut c = graph.add_pass("C");
        c.add_storage_input(buf);
        let out = c.add_color_output("out", color_attachment());

        graph.set_backbuffer_source(out);
        let compiled = graph.bake();

        // Even though there are consecutive passes, compute breaks merging
        let groups = find_merge_groups(&compiled);
        // No pair qualifies for merge (B is compute, so A-B can't merge, B-C can't merge)
        assert!(groups.is_empty());
    }

    #[test]
    fn three_pass_chain_merge() {
        // A → B (AttachmentInput from A) → C (AttachmentInput from B)
        // All three should form one merge group
        let mut graph = RenderGraph::new();

        let mut a = graph.add_pass("A");
        let r1 = a.add_color_output("r1", color_attachment());

        let mut b = graph.add_pass("B");
        b.add_attachment_input(r1);
        let r2 = b.add_color_output("r2", color_attachment());

        let mut c = graph.add_pass("C");
        c.add_attachment_input(r2);
        let r3 = c.add_color_output("r3", color_attachment());

        graph.set_backbuffer_source(r3);
        let compiled = graph.bake();

        let groups = find_merge_groups(&compiled);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].steps.len(), 3);
    }

    #[test]
    fn non_consecutive_attachment_input_no_merge() {
        // A → B (texture read, not attachment) → C (attachment input from A)
        // C reads from A but they're not consecutive — no merge
        let mut graph = RenderGraph::new();

        let mut a = graph.add_pass("A");
        let r1 = a.add_color_output("r1", color_attachment());

        let mut b = graph.add_pass("B");
        b.add_texture_input(r1);
        let r2 = b.add_color_output("r2", color_attachment());

        let mut c = graph.add_pass("C");
        c.add_texture_input(r2);
        // C also reads r1 via attachment input, but A is not the previous pass
        c.add_attachment_input(r1);
        let r3 = c.add_color_output("r3", color_attachment());

        graph.set_backbuffer_source(r3);
        let compiled = graph.bake();

        let groups = find_merge_groups(&compiled);
        // B-C: C reads r2 via texture (not attachment), and r1 via attachment but A didn't write
        // at step B — so the merge check looks at whether prev pass (B) wrote the resource
        // that C reads via AttachmentInput. B didn't write r1, so no merge.
        // A-B: B reads via TextureInput, not AttachmentInput — no merge.
        assert!(groups.is_empty());
    }
}
