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
