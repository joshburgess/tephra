//! Device-generated commands layout and types.
//!
//! Provides types for defining indirect command layouts used with
//! `VK_EXT_device_generated_commands`. The GPU reads tokens from a buffer
//! and generates draw/dispatch commands, reducing CPU overhead for large scenes.

use ash::vk;

/// A token in an indirect command layout.
///
/// Each token describes one operation the GPU performs when processing
/// an indirect command sequence.
#[derive(Debug, Clone)]
pub enum IndirectLayoutToken {
    /// Draw call token — dispatches `vkCmdDraw` or `vkCmdDrawIndexed`.
    Draw {
        /// Byte offset in the indirect buffer.
        offset: u32,
    },
    /// Draw indexed token.
    DrawIndexed {
        /// Byte offset in the indirect buffer.
        offset: u32,
    },
    /// Dispatch token — dispatches `vkCmdDispatch`.
    Dispatch {
        /// Byte offset in the indirect buffer.
        offset: u32,
    },
    /// Draw mesh tasks token — dispatches `vkCmdDrawMeshTasksEXT`.
    DrawMeshTasks {
        /// Byte offset in the indirect buffer.
        offset: u32,
    },
    /// Push constant update token.
    PushConstant {
        /// Byte offset in the indirect buffer.
        offset: u32,
        /// Pipeline layout for push constant update.
        layout: vk::PipelineLayout,
        /// Shader stage flags for the push constant range.
        stage_flags: vk::ShaderStageFlags,
        /// Byte offset within push constants.
        push_constant_offset: u32,
        /// Size in bytes of the push constant data.
        push_constant_size: u32,
    },
    /// Vertex buffer binding token.
    VertexBuffer {
        /// Byte offset in the indirect buffer.
        offset: u32,
        /// Vertex buffer binding index.
        binding: u32,
    },
    /// Index buffer binding token.
    IndexBuffer {
        /// Byte offset in the indirect buffer.
        offset: u32,
        /// Index type (UINT16 or UINT32).
        index_type: vk::IndexType,
    },
}

/// Description for creating an indirect commands layout.
///
/// Defines the sequence of operations the GPU performs when executing
/// device-generated commands.
#[derive(Debug, Clone)]
pub struct IndirectCommandsLayoutDesc {
    /// The type of pipeline this layout targets.
    pub pipeline_bind_point: vk::PipelineBindPoint,
    /// Tokens defining the indirect command sequence.
    pub tokens: Vec<IndirectLayoutToken>,
    /// Total stride in bytes of one indirect command in the buffer.
    pub stride: u32,
}
