//! Byte ranges in a file.

use std::ops::Range;

/// Returns `true` if `a` and `b` share at least one byte.
pub fn overlaps(a: &Range<u64>, b: &Range<u64>) -> bool {
    a.start < b.end && b.start < a.end
}

/// Returns the number of bytes `extent` shares with the `(addr, len)` pairs in `sections`,
/// summed over the pairs.
pub fn covered_len(sections: &[(u64, u64)], extent: &Range<u64>) -> u64 {
    sections
        .iter()
        .map(|&(addr, len)| {
            (addr + len)
                .min(extent.end)
                .saturating_sub(addr.max(extent.start))
        })
        .sum()
}
