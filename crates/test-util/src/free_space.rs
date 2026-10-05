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

/// Returns the header HDF5 2.2.0 wrote at 535, in a file with 4-byte addresses and lengths, for a
/// manager of one section, 1600 bytes at 3648, whose section list is at 585.
pub fn four_byte_widths_header() -> Vec<u8> {
    hex(
        "465348440001400600000100000001000000000000000300500078003f00ffffffff490200001f0000001f000000c79f2cac",
    )
}

/// Returns the section list at 585 that [`four_byte_widths_header`] points at.
pub fn four_byte_widths_section_info() -> Vec<u8> {
    hex("465353450017020000014006000000000000400e000000000000003801289b")
}

/// Returns the header HDF5 2.2.0 wrote at 627, in a file with 8-byte addresses and 4-byte lengths,
/// for a manager of one section, 1600 bytes at 3648, whose section list is at 681.
pub fn four_byte_lengths_header() -> Vec<u8> {
    hex(
        "465348440001400600000100000001000000000000000300500078003f00ffffffffa90200000000000023000000230000000ac4c817",
    )
}

/// Returns the section list at 681 that [`four_byte_lengths_header`] points at.
pub fn four_byte_lengths_section_info() -> Vec<u8> {
    hex("46535345007302000000000000014006000000000000400e000000000000007a0f4061")
}

/// Returns the header HDF5 2.2.0 wrote at 485, in a file with 2-byte addresses and lengths, for a
/// manager of one section, 1600 bytes at 3648, whose section list is at 519.
pub fn two_byte_widths_header() -> Vec<u8> {
    hex("46534844000140060100010000000300500078003f00ffff07021d001d0006c1adf5")
}

/// Returns the section list at 519 that [`two_byte_widths_header`] points at.
pub fn two_byte_widths_section_info() -> Vec<u8> {
    hex("4653534500e501014006000000000000400e000000000000000a661e9c")
}

/// Returns the header HDF5 1.12.3, 1.14.6, and 2.2.0 wrote at 48 for a manager of ten sections,
/// whose 197-byte section list is at 645.
pub fn oversized_section_list_header() -> Vec<u8> {
    hex(
        "4653484400014f050000000000000a000000000000000a0000000000000000000000000000000300500078003f00ffffffffffffff7f8502000000000000c500000000000000c500000000000000ff32959e",
    )
}

/// Returns the section list at 645 that [`oversized_section_list_header`] points at, which holds 9
/// zero bytes between its last section and its checksum.
pub fn oversized_section_list() -> Vec<u8> {
    hex(
        "46535345003000000000000000020a000000000000004a0300000000000000ff1400000000000000011800000000000000da0100000000000000011b000000000000008c0900000000000000012f000000000000003b0600000000000000014a0000000000000082000000000000000001c2000000000000003a0a0000000000000001fe00000000000000d10e000000000000000146010000000000006204000000000000000189010000000000002111000000000000000000000000000000003f5c5cae",
    )
}

/// Parses `digits` as hexadecimal, two digits per byte.
fn hex(digits: &str) -> Vec<u8> {
    (0..digits.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&digits[i..i + 2], 16).unwrap())
        .collect()
}
