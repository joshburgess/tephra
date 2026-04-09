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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::RenderGraph;
    use crate::resource::AttachmentInfo;
    use ash::vk;

    fn color_attachment() -> AttachmentInfo {
        AttachmentInfo::absolute(vk::Format::R8G8B8A8_UNORM, 256, 256)
    }

    // -- compute_lifetimes --

    #[test]
    fn lifetimes_single_pass() {
        let mut graph = RenderGraph::new();
        let mut p = graph.add_pass("P");
        let r = p.add_color_output("out", color_attachment());
        graph.set_backbuffer_source(r);
        let compiled = graph.bake();

        let lifetimes = compute_lifetimes(&compiled);
        assert_eq!(lifetimes.len(), 1);
        assert_eq!(lifetimes[0].first_use, 0);
        assert_eq!(lifetimes[0].last_use, 0);
    }

    #[test]
    fn lifetimes_linear_chain() {
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

        let lifetimes = compute_lifetimes(&compiled);
        // r1 used in step 0 (write) and step 1 (read)
        let r1_lt = lifetimes.iter().find(|l| l.resource == r1.index).unwrap();
        assert_eq!(r1_lt.first_use, 0);
        assert_eq!(r1_lt.last_use, 1);
        // r3 only used in step 2
        let r3_lt = lifetimes.iter().find(|l| l.resource == r3.index).unwrap();
        assert_eq!(r3_lt.first_use, 2);
        assert_eq!(r3_lt.last_use, 2);
    }

    #[test]
    fn lifetimes_empty_graph() {
        let graph = RenderGraph::new();
        let compiled = graph.bake();
        let lifetimes = compute_lifetimes(&compiled);
        assert!(lifetimes.is_empty());
    }

    // -- find_alias_groups --

    #[test]
    fn no_aliasing_with_overlapping_lifetimes() {
        // A writes r1, B reads r1 and writes r2 — lifetimes overlap at step 1
        let mut graph = RenderGraph::new();

        let mut a = graph.add_pass("A");
        let r1 = a.add_color_output("r1", color_attachment());

        let mut b = graph.add_pass("B");
        b.add_texture_input(r1);
        let r2 = b.add_color_output("r2", color_attachment());

        graph.set_backbuffer_source(r2);
        let compiled = graph.bake();

        let groups = find_alias_groups(&compiled);
        // r1 lives [0,1], r2 lives [1,1] — they overlap at step 1
        // No alias group with 2+ resources possible
        assert!(groups.is_empty());
    }

    #[test]
    fn aliasing_with_non_overlapping_lifetimes() {
        // A writes r1, B reads r1 writes r2, C reads r2 writes r3
        // r1 lives [0,1], r3 lives [2,2] — non-overlapping, can alias
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

        let groups = find_alias_groups(&compiled);
        // r1 [0,1] and r3 [2,2] don't overlap — should alias
        assert!(!groups.is_empty());
        let aliased: Vec<u32> = groups.iter().flat_map(|g| &g.resources).copied().collect();
        assert!(aliased.contains(&r1.index));
        assert!(aliased.contains(&r3.index));
    }

    #[test]
    fn buffers_not_aliased() {
        // Only attachment resources are aliased, not buffers.
        let mut graph = RenderGraph::new();

        let mut a = graph.add_pass("A");
        a.set_compute();
        let buf1 = a.add_storage_output(
            "buf1",
            crate::resource::BufferInfo {
                size: 1024,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            },
        );

        let mut b = graph.add_pass("B");
        b.set_compute();
        b.add_storage_input(buf1);
        let buf2 = b.add_storage_output(
            "buf2",
            crate::resource::BufferInfo {
                size: 1024,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            },
        );

        graph.set_backbuffer_source(buf2);
        let compiled = graph.bake();

        let groups = find_alias_groups(&compiled);
        // Buffers shouldn't be aliased — groups should be empty
        assert!(groups.is_empty());
    }

    #[test]
    fn single_resource_no_group() {
        // A single attachment can't form an alias group (need 2+).
        let mut graph = RenderGraph::new();
        let mut p = graph.add_pass("P");
        let r = p.add_color_output("only", color_attachment());
        graph.set_backbuffer_source(r);
        let compiled = graph.bake();

        let groups = find_alias_groups(&compiled);
        assert!(groups.is_empty());
    }
}
