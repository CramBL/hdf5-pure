//! Byte-preservation assertions for file publication tests.

/// Asserts that the bytes outside a publication range are unchanged.
///
/// The compared suffix ends at `before.len()`. Appended bytes in `after` are outside the
/// comparison.
///
/// # Panics
///
/// Panics if the publication range overflows or exceeds `before.len()`, if `after` is shorter
/// than `before`, or if a compared byte differs.
///
/// # Examples
///
/// ```
/// use test_util::publication;
///
/// let before = b"prefixoldtail";
/// let after = b"prefixnewtailappended";
/// publication::assert_preserved_bytes(before, after, 6, 3);
/// assert_ne!(&before[6..9], &after[6..9]);
/// assert_eq!(&after[before.len()..], b"appended");
/// ```
#[track_caller]
pub fn assert_preserved_bytes(
    before: &[u8],
    after: &[u8],
    publication_offset: usize,
    publication_len: usize,
) {
    let end = publication_offset.checked_add(publication_len).unwrap();
    assert!(end <= before.len());
    assert!(after.len() >= before.len());
    assert_eq!(&after[..publication_offset], &before[..publication_offset]);
    assert_eq!(&after[end..before.len()], &before[end..]);
}
