use std::path::Path;

use crate::bytes;
use crate::checksum;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeaderStats {
    pub secondary_block_count: u64,
    pub secondary_block_size: u64,
    pub data_block_count: u64,
    pub data_block_size: u64,
    pub max_index_set: u64,
    pub element_count: u64,
}

#[track_caller]
pub fn header_stats(path: impl AsRef<Path>) -> HeaderStats {
    let file = crate::read_file(path.as_ref());
    let header = bytes::sole_signature(&file, SIGNATURE);
    // Signature(4) + version(1) + client ID(1) + element size(1) + maximum
    // elements bits(1) + index block elements(1) + data block minimum
    // elements(1) + secondary block minimum data pointers(1) + maximum data
    // block page elements bits(1), then the six statistics in the order the
    // header stores them: section `subsec_fmt4_appendixc_extarr`, version 4.0.
    let stat = |field: usize| {
        bytes::u64_at(
            &file,
            header + SIGNATURE.len() + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + field * LENGTH,
        )
    };
    HeaderStats {
        secondary_block_count: stat(0),
        secondary_block_size: stat(1),
        data_block_count: stat(2),
        data_block_size: stat(3),
        max_index_set: stat(4),
        element_count: stat(5),
    }
}

/// Builds an index block and its sole allocated secondary block with eight-byte addresses.
///
/// The secondary block holds 16,384 undefined data block pointers and their zeroed page bitmaps.
/// The caller supplies four encoded, unfiltered inline elements and places the secondary block
/// immediately after the index block. Both blocks include their checksums.
///
/// Returns [`None`] if `inline_elements` is not 32 bytes or the secondary block address overflows.
pub fn large_secondary_blocks(
    header_address: u64,
    index_address: u64,
    inline_elements: &[u8],
) -> Option<(Vec<u8>, Vec<u8>)> {
    if inline_elements.len() != INLINE_BYTES {
        return None;
    }
    let secondary_address = index_address.checked_add(u64::try_from(INDEX_BYTES).ok()?)?;
    let mut index = INDEX_SIGNATURE.to_vec();
    index.extend_from_slice(&[VERSION, UNFILTERED_CLIENT]);
    index.extend_from_slice(&header_address.to_le_bytes());
    index.extend_from_slice(inline_elements);
    for pointer in 0..DIRECT_POINTERS + SECONDARY_POINTERS {
        index.extend_from_slice(
            &if pointer == DIRECT_POINTERS + SECONDARY_ORDINAL {
                secondary_address
            } else {
                u64::MAX
            }
            .to_le_bytes(),
        );
    }
    checksum::append(&mut index);
    let mut secondary = SECONDARY_SIGNATURE.to_vec();
    secondary.extend_from_slice(&[VERSION, UNFILTERED_CLIENT]);
    secondary.extend_from_slice(&header_address.to_le_bytes());
    secondary.extend_from_slice(&SECONDARY_OFFSET.to_le_bytes());
    secondary.resize(secondary.len() + DATA_BLOCKS * BITMAP_BYTES, 0);
    secondary.resize(secondary.len() + DATA_BLOCKS * LENGTH, 0xff);
    checksum::append(&mut secondary);
    Some((index, secondary))
}

/// The checksum width defined in "The Extensible Array Index" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
pub const CHECKSUM_BYTES: usize = 4;
/// The version 0 index block signature defined in "The Extensible Array Index" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
pub const INDEX_SIGNATURE: &[u8; 4] = b"EAIB";
// Section `subsec_fmt4_appendixc_extarr`, version 4.0, defines the secondary-block signature.
const SECONDARY_SIGNATURE: &[u8; 4] = b"EASB";
// Section `subsec_fmt4_appendixc_extarr`, version 4.0, defines version zero.
const VERSION: u8 = 0;
// Section `subsec_fmt4_appendixc_extarr`, version 4.0, assigns client zero to unfiltered chunks.
const UNFILTERED_CLIENT: u8 = 0;
const INLINE_BYTES: usize = 4 * LENGTH;
const DIRECT_POINTERS: usize = 6;
const SECONDARY_POINTERS: usize = 57;
const SECONDARY_ORDINAL: usize = 24;
const SECONDARY_OFFSET: u64 = 4_294_967_280;
const DATA_BLOCKS: usize = 16_384;
const BITMAP_BYTES: usize = 32;
const INDEX_BYTES: usize =
    14 + INLINE_BYTES + (DIRECT_POINTERS + SECONDARY_POINTERS) * LENGTH + CHECKSUM_BYTES;

// Each statistic is a length field, eight bytes wide in every file this crate
// writes and in every file it is compared against.
const LENGTH: usize = 8;

/// The version 0 header signature defined in "The Extensible Array Index" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
pub const SIGNATURE: &[u8; 4] = b"EAHD";

#[cfg(test)]
mod tests {
    use crate::extensible_array::{self, HeaderStats};
    use crate::temp;

    #[test]
    fn reads_the_six_statistics_in_stored_order() {
        let mut file = vec![0u8; 8];
        file.extend_from_slice(extensible_array::SIGNATURE);
        file.extend_from_slice(&[0, 0, 4, 32, 4, 16, 4, 16]);
        for stat in 1..=6u64 {
            file.extend_from_slice(&stat.to_le_bytes());
        }
        let path = temp::temp_path("extensible_array.h5");
        std::fs::write(&path, &file).expect("write the fixture");

        assert_eq!(
            extensible_array::header_stats(&path),
            HeaderStats {
                secondary_block_count: 1,
                secondary_block_size: 2,
                data_block_count: 3,
                data_block_size: 4,
                max_index_set: 5,
                element_count: 6,
            }
        );
    }
}
