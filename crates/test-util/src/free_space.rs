//! The free-space manager's structures: section
//! `subsec_fmt4_infra_freespaceindex`, version 4.0.

/// The free-space manager header, which states how much space a manager
/// tracks and where its sections live.
pub const SIGNATURE: &[u8; 4] = b"FSHD";

/// The section-info block the header points at, which holds the free sections
/// themselves.
pub const SECTIONS_SIGNATURE: &[u8; 4] = b"FSSE";

/// Returns the header HDF5 1.14.6 wrote at 619 for a manager of one section, 1600 bytes at 2848,
/// whose section list is at 701.
pub fn single_section_header() -> Vec<u8> {
    hex(
        "46534844000140060000000000000100000000000000010000000000000000000000000000000300500078003f00ffffffffffffff7fbd0200000000000023000000000000002300000000000000ea133710",
    )
}

/// Returns the section list at 701 that [`single_section_header`] points at.
pub fn single_section_info() -> Vec<u8> {
    hex("46535345006b02000000000000014006000000000000200b000000000000005c797631")
}

/// Returns the header HDF5 1.14.6 wrote at 736 for a manager of two sections, 16 bytes at 871
/// and 893 bytes at 1155, whose section list is at 818.
pub fn two_section_header() -> Vec<u8> {
    hex(
        "4653484400018d030000000000000200000000000000020000000000000000000000000000000300500078003f00ffffffffffffff7f320300000000000035000000000000003500000000000000d681e354",
    )
}

/// Returns the section list at 818 that [`two_section_header`] points at.
pub fn two_section_info() -> Vec<u8> {
    hex(
        "4653534500e002000000000000011000000000000000670300000000000000017d03000000000000830400000000000000910245b2",
    )
}

/// Parses `digits` as hexadecimal, two digits per byte.
fn hex(digits: &str) -> Vec<u8> {
    (0..digits.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&digits[i..i + 2], 16).unwrap())
        .collect()
}
