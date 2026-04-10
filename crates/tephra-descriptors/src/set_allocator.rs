//! Per-layout descriptor set slab allocator.
//!
//! Each unique `VkDescriptorSetLayout` gets its own [`DescriptorSetAllocator`]
//! that manages a list of `VkDescriptorPool` slabs. Pools are reset per-frame
//! rather than freeing individual sets.

use ash::vk;

/// Number of descriptor sets per pool slab.
const SETS_PER_SLAB: u32 = 64;

/// Describes the descriptor types needed for a layout, used to size pool slabs.
#[derive(Debug, Clone)]
pub struct DescriptorSetLayoutInfo {
    /// The Vulkan descriptor set layout handle.
    pub layout: vk::DescriptorSetLayout,
    /// Pool sizes — one entry per descriptor type used in the layout.
    pub pool_sizes: Vec<vk::DescriptorPoolSize>,
}

/// A single pool slab that can allocate a fixed number of descriptor sets.
struct PoolSlab {
    pool: vk::DescriptorPool,
    allocated: u32,
    capacity: u32,
}

/// Allocator for descriptor sets of a single layout.
///
/// Manages a chain of `VkDescriptorPool` slabs. When a slab is exhausted,
/// a new one is created. All slabs are reset when the frame context recycles.
pub struct DescriptorSetAllocator {
    layout_info: DescriptorSetLayoutInfo,
    slabs: Vec<PoolSlab>,
    current_slab: usize,
}

impl DescriptorSetAllocator {
    /// Create a new allocator for the given layout.
    pub fn new(layout_info: DescriptorSetLayoutInfo) -> Self {
        Self {
            layout_info,
            slabs: Vec::new(),
            current_slab: 0,
        }
    }

    /// Allocate a descriptor set from this allocator.
    pub fn allocate(&mut self, device: &ash::Device) -> Result<vk::DescriptorSet, vk::Result> {
        // Try current slab
        if let Some(slab) = self.slabs.get_mut(self.current_slab) {
            if slab.allocated < slab.capacity {
                let set = self.allocate_from_slab(device, self.current_slab)?;
                return Ok(set);
            }
            // Current slab is full, try next
            self.current_slab += 1;
        }

        // Try subsequent slabs that might have space (after a reset)
        while self.current_slab < self.slabs.len() {
            let slab = &self.slabs[self.current_slab];
            if slab.allocated < slab.capacity {
                let set = self.allocate_from_slab(device, self.current_slab)?;
                return Ok(set);
            }
            self.current_slab += 1;
        }

        // All slabs exhausted — create a new one
        self.create_slab(device)?;
        let set = self.allocate_from_slab(device, self.current_slab)?;
        Ok(set)
    }

    /// Reset all slabs for reuse. Called when the frame context recycles.
    pub fn reset(&mut self, device: &ash::Device) -> Result<(), vk::Result> {
        for slab in &mut self.slabs {
            // SAFETY: device and pool are valid.
            unsafe {
                device.reset_descriptor_pool(slab.pool, vk::DescriptorPoolResetFlags::empty())?;
            }
            slab.allocated = 0;
        }
        self.current_slab = 0;
        Ok(())
    }

    /// Destroy all pool slabs.
    pub fn destroy(&mut self, device: &ash::Device) {
        for slab in self.slabs.drain(..) {
            // SAFETY: device is valid, pool is valid, GPU is idle.
            unsafe {
                device.destroy_descriptor_pool(slab.pool, None);
            }
        }
    }

    fn allocate_from_slab(
        &mut self,
        device: &ash::Device,
        slab_idx: usize,
    ) -> Result<vk::DescriptorSet, vk::Result> {
        let slab = &mut self.slabs[slab_idx];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(slab.pool)
            .set_layouts(std::slice::from_ref(&self.layout_info.layout));

        // SAFETY: device, pool, and layout are valid.
        let sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };
        slab.allocated += 1;
        Ok(sets[0])
    }

    fn create_slab(&mut self, device: &ash::Device) -> Result<(), vk::Result> {
        // Scale pool sizes by sets-per-slab
        let pool_sizes: Vec<vk::DescriptorPoolSize> = self
            .layout_info
            .pool_sizes
            .iter()
            .map(|ps| vk::DescriptorPoolSize {
                ty: ps.ty,
                descriptor_count: ps.descriptor_count * SETS_PER_SLAB,
            })
            .collect();

        // If layout has no descriptors (empty set), still need a pool
        let pool_ci = if pool_sizes.is_empty() {
            vk::DescriptorPoolCreateInfo::default().max_sets(SETS_PER_SLAB)
        } else {
            vk::DescriptorPoolCreateInfo::default()
                .max_sets(SETS_PER_SLAB)
                .pool_sizes(&pool_sizes)
        };

        // SAFETY: device is valid, pool_ci is well-formed.
        let pool = unsafe { device.create_descriptor_pool(&pool_ci, None)? };

        log::debug!(
            "Created descriptor pool slab (total slabs: {})",
            self.slabs.len() + 1
        );

        self.current_slab = self.slabs.len();
        self.slabs.push(PoolSlab {
            pool,
            allocated: 0,
            capacity: SETS_PER_SLAB,
        });

        Ok(())
    }
}
