//! Temporal resource aliasing.
//!
//! Resources with non-overlapping lifetimes can share the same physical
//! `VkImage` or `VkDeviceMemory` allocation, reducing memory usage.
//! Uses greedy interval-graph colouring on attachment resource lifetimes.

use log::debug;

use crate::compile::CompiledGraph;
use crate::resource::ResourceInfo;

/// Lifetime of a resource in execution-step space.
#[derive(Debug, Clone)]
pub struct ResourceLifetime {
    /// Resource index.
    pub(crate) resource: u32,
    /// First execution step that accesses this resource.
    pub(crate) first_use: usize,
    /// Last execution step that accesses this resource.
    pub(crate) last_use: usize,
}

/// A group of resources whose lifetimes don't overlap and can share memory.
#[derive(Debug, Clone)]
pub struct AliasGroup {
    /// Resource indices that can alias.
    pub resources: Vec<u32>,
}

/// Compute the lifetime (first use, last use) of each resource.
pub fn compute_lifetimes(graph: &CompiledGraph) -> Vec<ResourceLifetime> {
    let num_resources = graph.resources.len();
    let mut first_use = vec![usize::MAX; num_resources];
    let mut last_use = vec![0usize; num_resources];
    let mut used = vec![false; num_resources];

    for (step, &pass_idx) in graph.pass_order.iter().enumerate() {
        for access in &graph.passes[pass_idx].accesses {
            let res = access.resource.index as usize;
            used[res] = true;
            first_use[res] = first_use[res].min(step);
            last_use[res] = last_use[res].max(step);
        }
    }

    (0..num_resources)
        .filter(|&i| used[i])
        .map(|i| ResourceLifetime {
            resource: i as u32,
            first_use: first_use[i],
            last_use: last_use[i],
        })
        .collect()
}

/// Find groups of attachment resources that can share the same memory.
///
/// Uses a greedy interval-colouring approach: for each resource, try to
/// place it in an existing group where no lifetime overlaps.
pub fn find_alias_groups(graph: &CompiledGraph) -> Vec<AliasGroup> {
    let lifetimes = compute_lifetimes(graph);

    // Build lookup: resource index -> lifetime.
    let mut lt_lookup: Vec<Option<usize>> = vec![None; graph.resources.len()];
    for (i, lt) in lifetimes.iter().enumerate() {
        lt_lookup[lt.resource as usize] = Some(i);
    }

    // Only alias attachment resources (not buffers).
    let attachment_lt_indices: Vec<usize> = lifetimes
        .iter()
        .enumerate()
        .filter(|(_, lt)| {
            matches!(
                graph.resources[lt.resource as usize].info,
                ResourceInfo::Attachment(_)
            )
        })
        .map(|(i, _)| i)
        .collect();

    let mut groups: Vec<AliasGroup> = Vec::new();

    for &lt_idx in &attachment_lt_indices {
        let lt = &lifetimes[lt_idx];

        let mut placed = false;
        for group in &mut groups {
            let overlaps = group.resources.iter().any(|&existing_res| {
                if let Some(existing_lt_idx) = lt_lookup[existing_res as usize] {
                    let existing_lt = &lifetimes[existing_lt_idx];
                    lt.first_use <= existing_lt.last_use && existing_lt.first_use <= lt.last_use
                } else {
                    false
                }
            });

            if !overlaps {
                group.resources.push(lt.resource);
                placed = true;
                break;
            }
        }

        if !placed {
            groups.push(AliasGroup {
                resources: vec![lt.resource],
            });
        }
    }

    // Only keep groups with actual aliasing (2+ resources).
    groups.retain(|g| g.resources.len() > 1);

    for group in &groups {
        let names: Vec<&str> = group
            .resources
            .iter()
            .map(|&r| graph.resources[r as usize].name.as_str())
            .collect();
        debug!("Alias group: {:?}", names);
    }

    groups
}
