//! Unique monotonic resource IDs for O(1) dirty tracking.
//!
//! Every GPU resource (buffer, image, sampler, image view) is assigned a unique
//! [`Cookie`] at creation time. The descriptor binding system can compare cookies
//! instead of full descriptor state to detect changes.

use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_COOKIE: AtomicU64 = AtomicU64::new(1);

/// A unique, monotonically increasing resource identifier.
///
/// Cookies are generated via [`Cookie::next`] and never recycled.
/// Two resources with different cookies are guaranteed to be distinct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cookie(u64);

impl Cookie {
    /// Generate the next unique cookie.
    pub fn next() -> Self {
        Self(NEXT_COOKIE.fetch_add(1, Ordering::Relaxed))
    }

    /// The raw numeric value.
    pub fn value(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cookies_are_unique() {
        let a = Cookie::next();
        let b = Cookie::next();
        assert_ne!(a, b);
        assert!(b.value() > a.value());
    }

    #[test]
    fn cookies_are_monotonic() {
        let start = Cookie::next();
        let end = Cookie::next();
        assert!(end.value() > start.value());
    }
}
