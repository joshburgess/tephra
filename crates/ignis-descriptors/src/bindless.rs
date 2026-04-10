//! Bindless descriptor table for GPU-driven rendering.
//!
//! A [`BindlessTable`] provides a persistent descriptor set with large arrays
//! of textures and storage buffers, indexed by integer handles in shaders.
//! Uses `VK_EXT_descriptor_indexing` features (core in Vulkan 1.2):
//! - `UPDATE_AFTER_BIND`: descriptors can be updated without rebinding
//! - `PARTIALLY_BOUND`: unused array slots need not be valid
//!
//! # Shader usage
//! ```glsl
//! layout(set = 3, binding = 0) uniform sampler2D textures[];
//! layout(set = 3, binding = 1) buffer Buffers { uint data[]; } buffers[];
//!
//! // In fragment shader:
//! vec4 color = texture(textures[nonuniformEXT(push.texture_index)], uv);
//! ```

use ash::vk;
use log::debug;

/// Maximum number of bindless texture slots.
pub const MAX_BINDLESS_TEXTURES: u32 = 16384;
/// Maximum number of bindless buffer slots.
pub const MAX_BINDLESS_BUFFERS: u32 = 4096;

/// Persistent bindless descriptor set with texture and buffer arrays.
///
/// Allocate once, register/unregister resources dynamically, and bind the
/// set at a fixed index (typically set 3). The set uses `UPDATE_AFTER_BIND`
/// so updates don't require rebinding.
pub struct BindlessTable {
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    set: vk::DescriptorSet,
    free_texture_indices: Vec<u32>,
    free_buffer_indices: Vec<u32>,
    texture_count: u32,
    buffer_count: u32,
}

impl BindlessTable {
    /// Create a new bindless descriptor table.
    ///
    /// Requires Vulkan 1.2 descriptor indexing features to be enabled:
    /// - `descriptor_binding_partially_bound`
    /// - `descriptor_binding_sampled_image_update_after_bind`
    /// - `descriptor_binding_storage_buffer_update_after_bind`
    /// - `runtime_descriptor_array`
    pub fn new(device: &ash::Device) -> Result<Self, vk::Result> {
        Self::with_capacity(device, MAX_BINDLESS_TEXTURES, MAX_BINDLESS_BUFFERS)
    }

    /// Create a bindless table with custom capacity.
    pub fn with_capacity(
        device: &ash::Device,
        max_textures: u32,
        max_buffers: u32,
    ) -> Result<Self, vk::Result> {
        // Build bindings and flags for only non-zero-count types
        let mut active_bindings = Vec::new();
        let mut active_flags = Vec::new();

        if max_textures > 0 {
            active_bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(max_textures)
                    .stage_flags(vk::ShaderStageFlags::ALL),
            );
            active_flags.push(
                vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            );
        }

        if max_buffers > 0 {
            active_bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(max_buffers)
                    .stage_flags(vk::ShaderStageFlags::ALL),
            );
            active_flags.push(
                vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            );
        }

        let mut binding_flags_ci =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                .binding_flags(&active_flags);

        let layout_ci = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&active_bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .push_next(&mut binding_flags_ci);

        // SAFETY: device is valid, layout_ci is well-formed with correct pNext chain.
        let layout = unsafe { device.create_descriptor_set_layout(&layout_ci, None)? };

        // Pool — only include non-zero pool sizes
        let mut pool_sizes = Vec::new();
        if max_textures > 0 {
            pool_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: max_textures,
            });
        }
        if max_buffers > 0 {
            pool_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: max_buffers,
            });
        }

        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);

        // SAFETY: device is valid, pool_ci is well-formed.
        let pool = unsafe { device.create_descriptor_pool(&pool_ci, None)? };

        // Allocate the single persistent set
        let layouts = [layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);

        // SAFETY: device, pool, and layout are valid and compatible.
        let sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };
        let set = sets[0];

        // Build free lists (reverse order so pop gives lowest indices first)
        let free_texture_indices: Vec<u32> = (0..max_textures).rev().collect();
        let free_buffer_indices: Vec<u32> = (0..max_buffers).rev().collect();

        debug!(
            "Created bindless table: {} texture slots, {} buffer slots",
            max_textures, max_buffers
        );

        Ok(Self {
            pool,
            layout,
            set,
            free_texture_indices,
            free_buffer_indices,
            texture_count: max_textures,
            buffer_count: max_buffers,
        })
    }

    /// Register a texture and return its bindless index for shader access.
    ///
    /// Returns `None` if the table is full.
    pub fn register_texture(
        &mut self,
        device: &ash::Device,
        view: vk::ImageView,
        sampler: vk::Sampler,
        layout: vk::ImageLayout,
    ) -> Option<u32> {
        let index = self.free_texture_indices.pop()?;

        let image_info = vk::DescriptorImageInfo::default()
            .image_view(view)
            .sampler(sampler)
            .image_layout(layout);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(0)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&image_info));

        // SAFETY: device and descriptor set are valid, write is well-formed.
        unsafe { device.update_descriptor_sets(&[write], &[]) };

        debug!("Registered bindless texture at index {}", index);
        Some(index)
    }

    /// Unregister a texture, freeing its slot for reuse.
    ///
    /// The slot becomes invalid for shader access. With `PARTIALLY_BOUND`,
    /// accessing an unregistered slot is undefined behavior in the shader.
    pub fn unregister_texture(&mut self, index: u32) {
        debug_assert!(
            index < self.texture_count,
            "texture index {} out of range (max {})",
            index,
            self.texture_count
        );
        self.free_texture_indices.push(index);
    }

    /// Register a storage buffer and return its bindless index.
    ///
    /// Returns `None` if the table is full.
    pub fn register_buffer(
        &mut self,
        device: &ash::Device,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) -> Option<u32> {
        let index = self.free_buffer_indices.pop()?;

        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(offset)
            .range(range);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(1)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info));

        // SAFETY: device and descriptor set are valid, write is well-formed.
        unsafe { device.update_descriptor_sets(&[write], &[]) };

        debug!("Registered bindless buffer at index {}", index);
        Some(index)
    }

    /// Unregister a storage buffer, freeing its slot for reuse.
    pub fn unregister_buffer(&mut self, index: u32) {
        debug_assert!(
            index < self.buffer_count,
            "buffer index {} out of range (max {})",
            index,
            self.buffer_count
        );
        self.free_buffer_indices.push(index);
    }

    /// The persistent descriptor set to bind in the command buffer.
    pub fn descriptor_set(&self) -> vk::DescriptorSet {
        self.set
    }

    /// The descriptor set layout (needed for pipeline layout creation).
    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }

    /// Number of free texture slots remaining.
    pub fn free_texture_count(&self) -> u32 {
        self.free_texture_indices.len() as u32
    }

    /// Number of free buffer slots remaining.
    pub fn free_buffer_count(&self) -> u32 {
        self.free_buffer_indices.len() as u32
    }

    /// Destroy all Vulkan resources owned by this table.
    pub fn destroy(&mut self, device: &ash::Device) {
        // SAFETY: device is valid, handles are valid, GPU is idle.
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}
