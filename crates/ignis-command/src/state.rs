//! Pipeline state tracking and save/restore for command buffers.

use ash::vk;

/// Stencil operation state for one face.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StencilFaceState {
    /// Operation on stencil test fail.
    pub fail_op: vk::StencilOp,
    /// Operation on stencil pass + depth fail.
    pub depth_fail_op: vk::StencilOp,
    /// Operation on both stencil and depth pass.
    pub pass_op: vk::StencilOp,
    /// Stencil comparison operator.
    pub compare_op: vk::CompareOp,
}

impl Default for StencilFaceState {
    fn default() -> Self {
        Self {
            fail_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            compare_op: vk::CompareOp::ALWAYS,
        }
    }
}

impl StencilFaceState {
    /// Convert to Vulkan `StencilOpState`.
    ///
    /// Reference, write mask, and compare mask are dynamic state.
    pub fn to_vk(&self) -> vk::StencilOpState {
        vk::StencilOpState {
            fail_op: self.fail_op,
            pass_op: self.pass_op,
            depth_fail_op: self.depth_fail_op,
            compare_op: self.compare_op,
            compare_mask: 0xFF,
            write_mask: 0xFF,
            reference: 0,
        }
    }
}

/// Static pipeline state that contributes to the pipeline hash key.
///
/// All fields that affect `VkGraphicsPipelineCreateInfo` are tracked here.
/// On draw, this state is hashed together with the active program and render pass
/// to look up or compile the correct pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StaticPipelineState {
    /// Primitive topology.
    pub topology: vk::PrimitiveTopology,
    /// Face culling mode.
    pub cull_mode: vk::CullModeFlags,
    /// Front face winding order.
    pub front_face: vk::FrontFace,
    /// Polygon rasterization mode.
    pub polygon_mode: vk::PolygonMode,
    /// Whether depth testing is enabled.
    pub depth_test: bool,
    /// Whether depth writes are enabled.
    pub depth_write: bool,
    /// Depth comparison operator.
    pub depth_compare: vk::CompareOp,
    /// Whether stencil testing is enabled.
    pub stencil_test: bool,
    /// Front-face stencil operations.
    pub stencil_front: StencilFaceState,
    /// Back-face stencil operations.
    pub stencil_back: StencilFaceState,
    /// Whether depth bias is enabled (constant/clamp/slope are dynamic state).
    pub depth_bias_enable: bool,
    /// Whether blending is enabled.
    pub blend_enable: bool,
    /// Source color blend factor.
    pub src_color_blend: vk::BlendFactor,
    /// Destination color blend factor.
    pub dst_color_blend: vk::BlendFactor,
    /// Color blend operation.
    pub color_blend_op: vk::BlendOp,
    /// Source alpha blend factor.
    pub src_alpha_blend: vk::BlendFactor,
    /// Destination alpha blend factor.
    pub dst_alpha_blend: vk::BlendFactor,
    /// Alpha blend operation.
    pub alpha_blend_op: vk::BlendOp,
    /// Color write mask.
    pub color_write_mask: vk::ColorComponentFlags,
    /// Whether primitive restart is enabled.
    pub primitive_restart: bool,
    /// Whether alpha-to-coverage multisampling is enabled.
    pub alpha_to_coverage: bool,
    /// Whether alpha-to-one multisampling is enabled.
    pub alpha_to_one: bool,
    /// Whether sample shading is enabled.
    pub sample_shading: bool,
    /// Rasterization sample count.
    pub rasterization_samples: vk::SampleCountFlags,
    /// Specialization constant bitmask — which of the 8 slots are active.
    pub spec_constant_mask: u32,
    /// Up to 8 specialization constant values (reinterpreted as float/int/bool).
    pub spec_constants: [u32; 8],
}

impl Default for StaticPipelineState {
    fn default() -> Self {
        Self {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            polygon_mode: vk::PolygonMode::FILL,
            depth_test: false,
            depth_write: false,
            depth_compare: vk::CompareOp::LESS_OR_EQUAL,
            stencil_test: false,
            stencil_front: StencilFaceState::default(),
            stencil_back: StencilFaceState::default(),
            depth_bias_enable: false,
            blend_enable: false,
            src_color_blend: vk::BlendFactor::ONE,
            dst_color_blend: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend: vk::BlendFactor::ONE,
            dst_alpha_blend: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
            primitive_restart: false,
            alpha_to_coverage: false,
            alpha_to_one: false,
            sample_shading: false,
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            spec_constant_mask: 0,
            spec_constants: [0; 8],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    fn hash_state(state: &StaticPipelineState) -> u64 {
        let mut hasher = DefaultHasher::new();
        state.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn default_state_values() {
        let state = StaticPipelineState::default();
        assert_eq!(state.topology, vk::PrimitiveTopology::TRIANGLE_LIST);
        assert_eq!(state.cull_mode, vk::CullModeFlags::BACK);
        assert_eq!(state.front_face, vk::FrontFace::COUNTER_CLOCKWISE);
        assert_eq!(state.polygon_mode, vk::PolygonMode::FILL);
        assert!(!state.depth_test);
        assert!(!state.depth_write);
        assert_eq!(state.depth_compare, vk::CompareOp::LESS_OR_EQUAL);
        assert!(!state.stencil_test);
        assert!(!state.blend_enable);
        assert!(!state.primitive_restart);
    }

    #[test]
    fn equal_states_same_hash() {
        let a = StaticPipelineState::default();
        let b = StaticPipelineState::default();
        assert_eq!(a, b);
        assert_eq!(hash_state(&a), hash_state(&b));
    }

    #[test]
    fn different_topology_different_hash() {
        let a = StaticPipelineState::default();
        let b = StaticPipelineState {
            topology: vk::PrimitiveTopology::LINE_LIST,
            ..Default::default()
        };
        assert_ne!(a, b);
        assert_ne!(hash_state(&a), hash_state(&b));
    }

    #[test]
    fn different_cull_mode_different_hash() {
        let a = StaticPipelineState::default();
        let b = StaticPipelineState {
            cull_mode: vk::CullModeFlags::FRONT,
            ..Default::default()
        };
        assert_ne!(a, b);
        assert_ne!(hash_state(&a), hash_state(&b));
    }

    #[test]
    fn depth_test_toggle_different_hash() {
        let a = StaticPipelineState::default();
        let b = StaticPipelineState {
            depth_test: true,
            depth_write: true,
            ..Default::default()
        };
        assert_ne!(a, b);
        assert_ne!(hash_state(&a), hash_state(&b));
    }

    #[test]
    fn blend_state_different_hash() {
        let a = StaticPipelineState::default();
        let b = StaticPipelineState {
            blend_enable: true,
            src_color_blend: vk::BlendFactor::SRC_ALPHA,
            dst_color_blend: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            ..Default::default()
        };
        assert_ne!(a, b);
        assert_ne!(hash_state(&a), hash_state(&b));
    }

    #[test]
    fn hash_stability() {
        // Same state should always produce the same hash within a run.
        let state = StaticPipelineState::default();
        let h1 = hash_state(&state);
        let h2 = hash_state(&state);
        assert_eq!(h1, h2);
    }

    #[test]
    fn clone_produces_equal() {
        let a = StaticPipelineState::default();
        let b = a.clone();
        assert_eq!(a, b);
        assert_eq!(hash_state(&a), hash_state(&b));
    }
}
