//! Virtual resource declarations for the render graph.

/// Opaque handle to a virtual resource in the render graph.
///
/// Resources are virtual until physical images/buffers are assigned during
/// graph compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceHandle {
    pub(crate) index: u32,
}
