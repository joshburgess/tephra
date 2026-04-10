//! Surface pre-rotation for Android and TBDR GPUs.
//!
//! On some platforms (especially Android), the display may be rotated relative
//! to the native surface orientation. Without pre-rotation, the compositor must
//! perform an extra copy/rotation, wasting bandwidth on TBDR architectures.
//!
//! This module provides helpers to detect the current surface transform and
//! compute the correction matrix that should be applied in the vertex shader
//! (typically via a specialization constant or push constant).

use ash::vk;

/// The current surface pre-rotation state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SurfaceRotation {
    /// No rotation (identity).
    Identity,
    /// 90 degrees clockwise.
    Rotate90,
    /// 180 degrees.
    Rotate180,
    /// 270 degrees clockwise (90 degrees counter-clockwise).
    Rotate270,
}

impl SurfaceRotation {
    /// Detect the current surface rotation from the surface transform flags.
    pub fn from_transform(transform: vk::SurfaceTransformFlagsKHR) -> Self {
        if transform.contains(vk::SurfaceTransformFlagsKHR::ROTATE_90) {
            SurfaceRotation::Rotate90
        } else if transform.contains(vk::SurfaceTransformFlagsKHR::ROTATE_180) {
            SurfaceRotation::Rotate180
        } else if transform.contains(vk::SurfaceTransformFlagsKHR::ROTATE_270) {
            SurfaceRotation::Rotate270
        } else {
            SurfaceRotation::Identity
        }
    }

    /// Convert to `VkSurfaceTransformFlagBitsKHR` for swapchain creation.
    pub fn to_transform(self) -> vk::SurfaceTransformFlagsKHR {
        match self {
            SurfaceRotation::Identity => vk::SurfaceTransformFlagsKHR::IDENTITY,
            SurfaceRotation::Rotate90 => vk::SurfaceTransformFlagsKHR::ROTATE_90,
            SurfaceRotation::Rotate180 => vk::SurfaceTransformFlagsKHR::ROTATE_180,
            SurfaceRotation::Rotate270 => vk::SurfaceTransformFlagsKHR::ROTATE_270,
        }
    }

    /// Returns the rotation angle in degrees.
    pub fn degrees(self) -> f32 {
        match self {
            SurfaceRotation::Identity => 0.0,
            SurfaceRotation::Rotate90 => 90.0,
            SurfaceRotation::Rotate180 => 180.0,
            SurfaceRotation::Rotate270 => 270.0,
        }
    }

    /// Compute a 2D rotation matrix (column-major, as a flat `[f32; 4]`)
    /// that pre-rotates clip-space coordinates.
    ///
    /// Multiply this with the projection matrix or apply in the vertex shader:
    /// ```glsl
    /// gl_Position.xy = preRotationMatrix * gl_Position.xy;
    /// ```
    ///
    /// The returned matrix is stored as `[m00, m10, m01, m11]` (column-major).
    pub fn rotation_matrix(self) -> [f32; 4] {
        match self {
            SurfaceRotation::Identity => [1.0, 0.0, 0.0, 1.0],
            SurfaceRotation::Rotate90 => [0.0, 1.0, -1.0, 0.0],
            SurfaceRotation::Rotate180 => [-1.0, 0.0, 0.0, -1.0],
            SurfaceRotation::Rotate270 => [0.0, -1.0, 1.0, 0.0],
        }
    }

    /// Returns `true` if the rotation swaps width and height.
    pub fn swaps_dimensions(self) -> bool {
        matches!(self, SurfaceRotation::Rotate90 | SurfaceRotation::Rotate270)
    }

    /// Adjust an extent for pre-rotation (swap width/height if needed).
    pub fn adjust_extent(self, extent: vk::Extent2D) -> vk::Extent2D {
        if self.swaps_dimensions() {
            vk::Extent2D {
                width: extent.height,
                height: extent.width,
            }
        } else {
            extent
        }
    }
}

/// Query the current surface transform for pre-rotation.
pub fn query_surface_rotation(
    surface_loader: &ash::khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> Result<SurfaceRotation, vk::Result> {
    // SAFETY: physical_device and surface are valid.
    let caps = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
    };
    Ok(SurfaceRotation::from_transform(caps.current_transform))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_matrix() {
        let m = SurfaceRotation::Identity.rotation_matrix();
        assert_eq!(m, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn rotate90_swaps_dimensions() {
        assert!(SurfaceRotation::Rotate90.swaps_dimensions());
        assert!(!SurfaceRotation::Identity.swaps_dimensions());
        assert!(!SurfaceRotation::Rotate180.swaps_dimensions());
        assert!(SurfaceRotation::Rotate270.swaps_dimensions());
    }

    #[test]
    fn adjust_extent_swap() {
        let extent = vk::Extent2D {
            width: 1920,
            height: 1080,
        };
        let rotated = SurfaceRotation::Rotate90.adjust_extent(extent);
        assert_eq!(rotated.width, 1080);
        assert_eq!(rotated.height, 1920);
    }

    #[test]
    fn adjust_extent_identity() {
        let extent = vk::Extent2D {
            width: 1920,
            height: 1080,
        };
        let same = SurfaceRotation::Identity.adjust_extent(extent);
        assert_eq!(same.width, 1920);
        assert_eq!(same.height, 1080);
    }

    #[test]
    fn from_transform_flags() {
        assert_eq!(
            SurfaceRotation::from_transform(vk::SurfaceTransformFlagsKHR::IDENTITY),
            SurfaceRotation::Identity
        );
        assert_eq!(
            SurfaceRotation::from_transform(vk::SurfaceTransformFlagsKHR::ROTATE_90),
            SurfaceRotation::Rotate90
        );
        assert_eq!(
            SurfaceRotation::from_transform(vk::SurfaceTransformFlagsKHR::ROTATE_180),
            SurfaceRotation::Rotate180
        );
        assert_eq!(
            SurfaceRotation::from_transform(vk::SurfaceTransformFlagsKHR::ROTATE_270),
            SurfaceRotation::Rotate270
        );
    }
}
