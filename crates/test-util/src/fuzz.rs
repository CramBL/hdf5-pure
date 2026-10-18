//! Loads committed malformed-file inputs for regression tests.

use crate::bytes;
use crate::superblock::v0::Fields;

/// Loads a regression fixture and extends it to the end-of-file address stored in its superblock.
///
/// The fixture is read from `tests/data/fuzz/` relative to the repository root.
///
/// # Panics
///
/// Panics if the fixture cannot be read, its superblock cannot be parsed, or its recorded extent
/// is smaller than the fixture or cannot be represented as a buffer length.
pub fn fixture_with_recorded_extent(filename: &str) -> Vec<u8> {
    let mut image = std::fs::read(crate::data(&format!("fuzz/{filename}"))).unwrap();
    let Fields {
        widths,
        base_address,
        eof_address_at,
        root_header_address: _,
    } = Fields::read(&image, 0);
    assert_eq!(base_address, 0);
    let stored_eof = bytes::uint_at(&image, eof_address_at, widths.offset);
    assert!(stored_eof >= image.len() as u64);
    image.resize(usize::try_from(stored_eof).unwrap(), 0);
    image
}
