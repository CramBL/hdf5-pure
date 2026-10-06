//! Byte ranges in a file.

use std::ops::Range;

/// Returns `true` if `a` and `b` share at least one byte.
pub fn overlaps(a: &Range<u64>, b: &Range<u64>) -> bool {
    a.start < b.end && b.start < a.end
}
