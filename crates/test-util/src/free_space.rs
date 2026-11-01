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

/// Returns a checksummed manager header and section list with 8-byte addresses and lengths.
///
/// Sections are `(stored_address, length)` pairs, grouped by length in the section list. The
/// fixed header fields and section-list prefix are copied from [`single_section_header`] and
/// [`single_section_info`]. Encodes group counts using the fewest bytes that hold the total
/// section count, with a one-byte minimum. Section extents may be invalid.
///
/// # Panics
///
/// Panics if section lengths sum to more than [`u64::MAX`].
///
/// # Examples
///
/// ```
/// use test_util::free_space;
///
/// assert_eq!(
///     free_space::raw_manager_eight(619, 701, &[(2848, 1600)]),
///     (free_space::single_section_header(), free_space::single_section_info()),
/// );
/// ```
pub fn raw_manager_eight(
    header_address: u64,
    list_address: u64,
    sections: &[(u64, u64)],
) -> (Vec<u8>, Vec<u8>) {
    use std::collections::BTreeMap;

    /// The file client's simple section class, `H5MF_FSPACE_SECT_SIMPLE` in `H5MFpkg.h`,
    /// HDF5 2.2.0.
    const SECTION_CLASS_SIMPLE: u8 = 0;

    let count = u64::try_from(sections.len()).unwrap();
    let count_width = usize::try_from(
        (u64::BITS - count.leading_zeros())
            .div_ceil(u8::BITS)
            .max(1),
    )
    .unwrap();
    let mut sets: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for &(address, length) in sections {
        sets.entry(length).or_default().push(address);
    }
    let mut list = single_section_info()[..13].to_vec();
    list[5..13].copy_from_slice(&header_address.to_le_bytes());
    for (length, addresses) in sets {
        crate::bytes::push_uint(
            &mut list,
            u64::try_from(addresses.len()).unwrap(),
            count_width,
        );
        list.extend_from_slice(&length.to_le_bytes());
        for address in addresses {
            list.extend_from_slice(&address.to_le_bytes());
            list.push(SECTION_CLASS_SIMPLE);
        }
    }
    crate::checksum::append(&mut list);
    let mut header = single_section_header();
    let total = sections
        .iter()
        .try_fold(0u64, |total, &(_, length)| total.checked_add(length))
        .unwrap();
    for (at, value) in [
        (6, total),
        (14, count),
        (22, count),
        (54, list_address),
        (62, u64::try_from(list.len()).unwrap()),
        (70, u64::try_from(list.len()).unwrap()),
    ] {
        crate::bytes::set_slice_at(&mut header, at, &value.to_le_bytes());
    }
    let length = header.len();
    crate::checksum::restamp(&mut header, 0, length);
    (header, list)
}

/// Parses `digits` as hexadecimal, two digits per byte.
fn hex(digits: &str) -> Vec<u8> {
    (0..digits.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&digits[i..i + 2], 16).unwrap())
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::one_byte(255, 1)]
    #[case::two_bytes(256, 2)]
    #[case::last_two_bytes(65_535, 2)]
    #[case::three_bytes(65_536, 3)]
    fn section_sets_use_the_total_count_width(#[case] count: u64, #[case] width: usize) {
        let sections = (0..count)
            .map(|index| (index * 4, index % 2 + 1))
            .collect::<Vec<_>>();
        let (_, list) = raw_manager_eight(619, 701, &sections);
        let prefix = SECTIONS_SIGNATURE.len() + size_of::<u8>() + size_of::<u64>();
        let record = size_of::<u64>() + size_of::<u8>();
        let fields = width + size_of::<u64>();
        let first = count.div_ceil(2);
        assert_eq!(crate::bytes::uint_at(&list, prefix, width), first);
        let second = prefix + fields + usize::try_from(first).unwrap() * record;
        assert_eq!(crate::bytes::uint_at(&list, second, width), count / 2);
        assert_eq!(
            list.len(),
            prefix + 2 * fields + usize::try_from(count).unwrap() * record + size_of::<u32>()
        );
    }
}
