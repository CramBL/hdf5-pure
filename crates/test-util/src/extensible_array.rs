use std::path::Path;

use crate::bytes;
use crate::checksum;

/// The values for an independently constructed version 0 Extensible Array header.
#[derive(Clone, Copy, Debug)]
pub struct Header {
    offsets: Width,
    lengths: Width,
    client: Client,
    creation: Creation,
    index: Option<Index>,
}

impl Header {
    /// Creates a fixture with zero statistics and an undefined index-block address.
    pub fn new(offsets: Width, lengths: Width, client: Client, creation: Creation) -> Self {
        Self {
            offsets,
            lengths,
            client,
            creation,
            index: None,
        }
    }

    /// Stores a defined index-block address and the header statistics if they fit their widths.
    ///
    /// Returns [`None`] when any field is too large or `address` equals the undefined marker.
    pub fn with_index(mut self, address: u64, stats: HeaderStats) -> Option<Self> {
        if address >= self.offsets.maximum()
            || stats
                .fields()
                .iter()
                .any(|&value| value > self.lengths.maximum())
        {
            return None;
        }
        self.index = Some(Index { address, stats });
        Some(self)
    }

    /// Builds the header, including its checksum.
    pub fn build(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.encoded_len());
        bytes.extend_from_slice(SIGNATURE);
        bytes.extend_from_slice(&[
            VERSION,
            self.client.id(),
            self.client.element_size(self.offsets),
        ]);
        let [max_bits, inline, min_pointers, min_elements, page_bits] =
            self.creation.layout_fields();
        bytes.extend_from_slice(&[max_bits, inline, min_elements, min_pointers, page_bits]);
        let (stats, address) = match self.index {
            Some(index) => (index.stats.fields(), Some(index.address)),
            None => ([0; STATISTIC_COUNT], None),
        };
        for value in stats {
            bytes::push_uint(&mut bytes, value, usize::from(self.lengths.get()));
        }
        bytes::push_address(&mut bytes, address, usize::from(self.offsets.get()));
        checksum::append(&mut bytes);
        bytes
    }

    /// Returns the encoded length from the signature through the checksum.
    pub fn encoded_len(self) -> usize {
        PREFIX_LEN
            + STATISTIC_COUNT * usize::from(self.lengths.get())
            + usize::from(self.offsets.get())
            + CHECKSUM_BYTES
    }
}

/// The chunk element encodings that the fixture can describe.
#[derive(Clone, Copy, Debug)]
pub enum Client {
    /// Filtered chunks, with the width of the stored chunk-size field.
    Filtered(Width),
    /// Unfiltered chunks, whose entries hold only an address.
    Unfiltered,
}

impl Client {
    /// Returns the client ID the header stores.
    pub fn id(self) -> u8 {
        match self {
            Self::Unfiltered => UNFILTERED_CLIENT,
            Self::Filtered(_) => FILTERED_CLIENT,
        }
    }

    /// Returns the entry size in bytes at the supplied address width.
    pub fn element_size(self, offsets: Width) -> u8 {
        offsets.get()
            + match self {
                Self::Unfiltered => 0,
                Self::Filtered(chunk_size_width) => chunk_size_width.get() + FILTER_MASK_LEN,
            }
    }
}

/// Valid creation parameter sets for header and layout message fixtures.
#[derive(Clone, Copy, Debug)]
pub enum Creation {
    /// The default geometry for dataset chunks.
    Canonical,
    /// Reserves 32 inline entries in the index block.
    LargeInline,
    /// Reserves no inline entries in the index block.
    NoInline,
    /// Sets the maximum element bit count to 16 and reserves eight inline entries.
    Small,
    /// Reserves two inline entries and limits each page to two elements.
    SmallPages,
}

impl Creation {
    /// Returns the five creation fields in data layout message order.
    pub fn layout_fields(self) -> [u8; 5] {
        match self {
            // The H5D_EARRAY_* constants in H5Dpkg.h, HDF5 2.2.0, set the default geometry.
            Self::Canonical => [32, 4, 4, 16, 10],
            Self::LargeInline => [24, 32, 8, 32, 12],
            Self::NoInline => [16, 0, 2, 8, 8],
            Self::Small => [16, 8, 2, 8, 8],
            Self::SmallPages => [8, 2, 2, 1, 1],
        }
    }
}

/// The allowed fixture widths for addresses, lengths, and stored chunk sizes.
#[derive(Clone, Copy, Debug)]
pub enum Width {
    /// Two bytes.
    Two,
    /// Four bytes.
    Four,
    /// Eight bytes.
    Eight,
}

impl Width {
    /// Returns the number of bytes in the field.
    pub fn get(self) -> u8 {
        match self {
            Self::Two => 2,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }

    fn maximum(self) -> u64 {
        match self {
            Self::Two => u64::from(u16::MAX),
            Self::Four => u64::from(u32::MAX),
            Self::Eight => u64::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Index {
    address: u64,
    stats: HeaderStats,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeaderStats {
    pub secondary_block_count: u64,
    pub secondary_block_size: u64,
    pub data_block_count: u64,
    pub data_block_size: u64,
    pub max_index_set: u64,
    pub element_count: u64,
}

impl HeaderStats {
    fn fields(self) -> [u64; STATISTIC_COUNT] {
        [
            self.secondary_block_count,
            self.secondary_block_size,
            self.data_block_count,
            self.data_block_size,
            self.max_index_set,
            self.element_count,
        ]
    }
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

/// The length in bytes of the signature and the eight single-byte header fields.
///
/// "The Extensible Array Index" of the [format specification, version 4.0][spec] defines
/// these fields and the statistics that follow.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
pub const PREFIX_LEN: usize = 12;

/// The number of statistics stored at the file's length width before the index-block address.
pub const STATISTIC_COUNT: usize = 6;
// "The Extensible Array Index" in the format specification, version 4.0, assigns client one
// to filtered chunks.
const FILTERED_CLIENT: u8 = 1;
// "The Extensible Array Index" in the format specification, version 4.0, defines the four-byte
// filter mask for filtered chunks.
const FILTER_MASK_LEN: u8 = 4;

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
