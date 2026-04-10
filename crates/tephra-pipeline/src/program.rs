//! Multi-stage shader program linking.
//!
//! A [`Program`] combines multiple [`Shader`](crate::shader::Shader) stages,
//! merges their reflection data, and creates the Vulkan pipeline layout and
//! per-set descriptor allocators.

use std::hash::{Hash, Hasher};

use ash::vk;
use rustc_hash::FxHasher;

use tephra_descriptors::set_allocator::{
    DescriptorSetAllocator, DescriptorSetLayoutInfo as AllocatorLayoutInfo,
};

use crate::shader::{MAX_DESCRIPTOR_SETS, ReflectedSetLayout, Shader};

/// A linked shader program (e.g., vertex + fragment, or compute).
///
/// Owns the `VkPipelineLayout`, `VkDescriptorSetLayout`s, and per-set
/// descriptor set allocators.
pub struct Program {
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layouts: [Option<vk::DescriptorSetLayout>; MAX_DESCRIPTOR_SETS],
    set_allocators: [Option<DescriptorSetAllocator>; MAX_DESCRIPTOR_SETS],
    push_constant_range: Option<vk::PushConstantRange>,
    layout_hash: u64,
    shaders: Vec<ShaderStageInfo>,
    /// Bitmask of descriptor sets using push descriptors.
    push_descriptor_set_mask: u32,
}

/// Info about a shader stage retained for pipeline compilation.
pub(crate) struct ShaderStageInfo {
    pub module: vk::ShaderModule,
    pub stage: vk::ShaderStageFlags,
}

impl Program {
    /// Create a linked program from one or more shader stages.
    ///
    /// Merges reflection data from all shaders, creates `VkDescriptorSetLayout`s,
    /// `VkPipelineLayout`, and per-set `DescriptorSetAllocator`s.
    pub fn create(device: &ash::Device, shaders: &[&Shader]) -> Result<Self, vk::Result> {
        Self::create_internal(device, shaders, 0)
    }

    /// Create a linked program with push descriptor support.
    ///
    /// Sets indicated by `push_descriptor_set_mask` use `VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR`
    /// and skip descriptor pool allocation. Their descriptors are pushed inline
    /// via `vkCmdPushDescriptorSetKHR` instead.
    ///
    /// Requires `VK_KHR_push_descriptor` to be enabled on the device.
    pub fn create_with_push_descriptors(
        device: &ash::Device,
        shaders: &[&Shader],
        push_descriptor_set_mask: u32,
    ) -> Result<Self, vk::Result> {
        Self::create_internal(device, shaders, push_descriptor_set_mask)
    }

    fn create_internal(
        device: &ash::Device,
        shaders: &[&Shader],
        push_descriptor_set_mask: u32,
    ) -> Result<Self, vk::Result> {
        let merged_sets = Self::merge_descriptor_sets(shaders);
        let merged_push_constants = Self::merge_push_constants(shaders);

        // Create VkDescriptorSetLayouts and allocators
        let mut vk_set_layouts: [Option<vk::DescriptorSetLayout>; MAX_DESCRIPTOR_SETS] =
            [None; MAX_DESCRIPTOR_SETS];
        let mut set_allocators: [Option<DescriptorSetAllocator>; MAX_DESCRIPTOR_SETS] =
            std::array::from_fn(|_| None);

        for (i, set_info) in merged_sets.iter().enumerate() {
            if set_info.bindings.is_empty() {
                continue;
            }

            let is_push_set = push_descriptor_set_mask & (1 << i) != 0;

            let vk_bindings: Vec<vk::DescriptorSetLayoutBinding> = set_info
                .bindings
                .iter()
                .map(|b| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b.binding)
                        .descriptor_type(b.descriptor_type)
                        .descriptor_count(b.descriptor_count)
                        .stage_flags(b.stage_flags)
                })
                .collect();

            let mut layout_flags = vk::DescriptorSetLayoutCreateFlags::empty();
            if is_push_set {
                layout_flags |= vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR;
            }

            let layout_ci = vk::DescriptorSetLayoutCreateInfo::default()
                .flags(layout_flags)
                .bindings(&vk_bindings);

            // SAFETY: device is valid, layout_ci is well-formed.
            let layout = unsafe { device.create_descriptor_set_layout(&layout_ci, None)? };
            vk_set_layouts[i] = Some(layout);

            // Push descriptor sets don't need a pool allocator
            if !is_push_set {
                let pool_sizes: Vec<vk::DescriptorPoolSize> = set_info
                    .bindings
                    .iter()
                    .map(|b| vk::DescriptorPoolSize {
                        ty: b.descriptor_type,
                        descriptor_count: b.descriptor_count,
                    })
                    .collect();

                set_allocators[i] = Some(DescriptorSetAllocator::new(AllocatorLayoutInfo {
                    layout,
                    pool_sizes,
                }));
            }
        }

        // Collect contiguous non-None layouts starting from set 0
        let mut all_layouts: Vec<vk::DescriptorSetLayout> = Vec::new();
        for layout_opt in &vk_set_layouts {
            match layout_opt {
                Some(layout) => all_layouts.push(*layout),
                None => break,
            }
        }

        let mut layout_ci = vk::PipelineLayoutCreateInfo::default().set_layouts(&all_layouts);

        let pc_range;
        if let Some(ref range) = merged_push_constants {
            pc_range = [*range];
            layout_ci = layout_ci.push_constant_ranges(&pc_range);
        }

        // SAFETY: device is valid, layout_ci is well-formed.
        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_ci, None)? };

        let layout_hash = Self::compute_hash(shaders);

        let shader_infos: Vec<ShaderStageInfo> = shaders
            .iter()
            .map(|s| ShaderStageInfo {
                module: s.module,
                stage: s.stage,
            })
            .collect();

        log::debug!(
            "Created program with {} stages, {} descriptor set layouts, hash={:#x}",
            shaders.len(),
            all_layouts.len(),
            layout_hash
        );

        Ok(Self {
            pipeline_layout,
            descriptor_set_layouts: vk_set_layouts,
            set_allocators,
            push_constant_range: merged_push_constants,
            layout_hash,
            shaders: shader_infos,
            push_descriptor_set_mask,
        })
    }

    /// The Vulkan pipeline layout.
    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    /// The descriptor set layout for a given set index.
    pub fn descriptor_set_layout(&self, set: usize) -> Option<vk::DescriptorSetLayout> {
        self.descriptor_set_layouts.get(set).copied().flatten()
    }

    /// Mutable access to the descriptor set allocator for a given set.
    pub fn set_allocator_mut(&mut self, set: usize) -> Option<&mut DescriptorSetAllocator> {
        self.set_allocators.get_mut(set).and_then(|a| a.as_mut())
    }

    /// The merged push constant range.
    pub fn push_constant_range(&self) -> Option<&vk::PushConstantRange> {
        self.push_constant_range.as_ref()
    }

    /// The layout hash for pipeline key computation.
    pub fn layout_hash(&self) -> u64 {
        self.layout_hash
    }

    /// Whether a given descriptor set uses push descriptors.
    pub fn is_push_descriptor_set(&self, set: u32) -> bool {
        self.push_descriptor_set_mask & (1 << set) != 0
    }

    /// The shader stage infos for pipeline compilation.
    pub(crate) fn shaders(&self) -> &[ShaderStageInfo] {
        &self.shaders
    }

    /// Reset all per-set descriptor set allocators (called per-frame).
    pub fn reset_allocators(&mut self, device: &ash::Device) -> Result<(), vk::Result> {
        for alloc in self.set_allocators.iter_mut().flatten() {
            alloc.reset(device)?;
        }
        Ok(())
    }

    /// Destroy all Vulkan resources owned by this program.
    pub fn destroy(&mut self, device: &ash::Device) {
        for alloc in self.set_allocators.iter_mut().flatten() {
            alloc.destroy(device);
        }
        // SAFETY: device is valid, handles are valid, GPU is idle.
        unsafe {
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            for layout in self.descriptor_set_layouts.iter().flatten() {
                device.destroy_descriptor_set_layout(*layout, None);
            }
        }
    }

    fn merge_descriptor_sets(shaders: &[&Shader]) -> [ReflectedSetLayout; MAX_DESCRIPTOR_SETS] {
        let mut merged: [ReflectedSetLayout; MAX_DESCRIPTOR_SETS] =
            std::array::from_fn(|_| ReflectedSetLayout::default());

        for shader in shaders {
            for (set_idx, set_info) in shader.reflection.descriptor_sets.iter().enumerate() {
                for binding in &set_info.bindings {
                    if let Some(existing) = merged[set_idx]
                        .bindings
                        .iter_mut()
                        .find(|b| b.binding == binding.binding)
                    {
                        // Same set+binding across stages: merge stage flags
                        existing.stage_flags |= binding.stage_flags;
                        debug_assert_eq!(
                            existing.descriptor_type, binding.descriptor_type,
                            "Descriptor type mismatch at set={}, binding={}",
                            set_idx, binding.binding
                        );
                    } else {
                        merged[set_idx].bindings.push(binding.clone());
                    }
                }
            }
        }

        for set in &mut merged {
            set.bindings.sort_by_key(|b| b.binding);
        }

        merged
    }

    fn merge_push_constants(shaders: &[&Shader]) -> Option<vk::PushConstantRange> {
        let mut merged_stage = vk::ShaderStageFlags::empty();
        let mut max_size: u32 = 0;

        for shader in shaders {
            if let Some(range) = &shader.reflection.push_constant_range {
                merged_stage |= range.stage_flags;
                max_size = max_size.max(range.size);
            }
        }

        if max_size > 0 {
            Some(
                vk::PushConstantRange::default()
                    .stage_flags(merged_stage)
                    .offset(0)
                    .size(max_size),
            )
        } else {
            None
        }
    }

    fn compute_hash(shaders: &[&Shader]) -> u64 {
        let mut hasher = FxHasher::default();
        for shader in shaders {
            shader.module.hash(&mut hasher);
            shader.stage.hash(&mut hasher);
        }
        hasher.finish()
    }
}
