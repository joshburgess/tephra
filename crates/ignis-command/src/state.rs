//! Pipeline state tracking and save/restore for command buffers.

use ash::vk;

/// Static pipeline state that contributes to the pipeline hash key.
///
/// All fields that affect `VkGraphicsPipelineCreateInfo` are tracked here.
/// On draw, this state is hashed together with the active program and render pass
/// to look up or compile the correct pipeline.
#[derive(Clone, PartialEq, Eq, Hash)]
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
            blend_enable: false,
            src_color_blend: vk::BlendFactor::ONE,
            dst_color_blend: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend: vk::BlendFactor::ONE,
            dst_alpha_blend: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
            primitive_restart: false,
        }
    }
}
