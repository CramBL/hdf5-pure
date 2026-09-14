//! The Data Layout message (type 0x0008): a dataset's storage class and the
//! properties of that class.
//!
//! [`DataLayout::parse`] reads a version 3 or a version 4 message. The two
//! encode a compact and a contiguous dataset's properties alike, and a version 4
//! message stores a chunked dataset's indexing type as well, which the parse
//! turns into a [`ChunkIndexLayout`]. The message is defined in "The Data Layout
//! Message" of the [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::bytes::{ensure_len, read_length, read_optional_offset};
use crate::error::FormatError;

/// Parsed HDF5 data layout message.
#[derive(Debug, Clone, PartialEq)]
pub enum DataLayout {
    /// Compact: data stored inline in the message.
    Compact {
        /// The inline raw data bytes.
        data: Vec<u8>,
    },
    /// Contiguous: data stored at a single address in the file.
    Contiguous {
        /// File address of the data, or `None` if undefined (all 0xFF).
        address: Option<StoredAddress>,
        /// Size of the data in bytes.
        size: u64,
    },
    /// Chunked: data stored in chunks located through a chunk index.
    Chunked {
        /// Chunk dimension sizes, one per dataset dimension and then the
        /// element size in bytes.
        chunk_dimensions: Vec<u32>,
        /// The indexing type the message stores, and the address stored with it.
        index: ChunkIndexLayout,
    },
    /// Virtual dataset layout (v4 only).
    Virtual,
}

impl DataLayout {
    /// Parses a data layout message from its body.
    ///
    /// `offset_size` and `length_size` are the superblock's address and length
    /// widths.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidLayoutVersion`] if the version byte is
    /// neither 3 nor 4, [`FormatError::InvalidLayoutClass`] if the class byte
    /// is outside the storage classes that version defines,
    /// [`FormatError::InvalidChunkIndexType`] if a version 4 chunked message
    /// stores an indexing type outside 1 to 5, and
    /// [`FormatError::UnexpectedEof`] if the message body ends inside a field.
    pub fn parse(data: &[u8], offset_size: u8, length_size: u8) -> Result<DataLayout, FormatError> {
        ensure_len(data, 0, 2)?;
        let version = data[0];
        let layout_class = data[1];

        match version {
            LAYOUT_VERSION_3 => Self::parse_v3(data, layout_class, offset_size, length_size),
            LAYOUT_VERSION_4 => Self::parse_v4(data, layout_class, offset_size, length_size),
            _ => Err(FormatError::InvalidLayoutVersion(version)),
        }
    }

    fn parse_v3(
        data: &[u8],
        layout_class: u8,
        offset_size: u8,
        length_size: u8,
    ) -> Result<DataLayout, FormatError> {
        let pos = LAYOUT_PROPERTIES_OFFSET;
        match layout_class {
            LAYOUT_CLASS_COMPACT => Self::parse_compact(data),
            LAYOUT_CLASS_CONTIGUOUS => Self::parse_contiguous(data, offset_size, length_size),
            LAYOUT_CLASS_CHUNKED => {
                ensure_len(data, pos, 1)?;
                let dimensionality = data[pos] as usize;
                let mut p = pos + 1;
                // btree address first
                let os = offset_size as usize;
                ensure_len(data, p, os)?;
                let address = read_optional_offset(data, p, offset_size)?.map(StoredAddress::new);
                p += os;
                // chunk dim sizes: dimensionality × 4 bytes each
                ensure_len(data, p, dimensionality * 4)?;
                let mut chunk_dimensions = Vec::with_capacity(dimensionality);
                for _ in 0..dimensionality {
                    let dim = u32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]);
                    chunk_dimensions.push(dim);
                    p += 4;
                }
                Ok(DataLayout::Chunked {
                    chunk_dimensions,
                    index: ChunkIndexLayout::BTreeV1 { address },
                })
            }
            _ => Err(FormatError::InvalidLayoutClass(layout_class)),
        }
    }

    fn parse_v4(
        data: &[u8],
        layout_class: u8,
        offset_size: u8,
        length_size: u8,
    ) -> Result<DataLayout, FormatError> {
        let pos = LAYOUT_PROPERTIES_OFFSET;
        match layout_class {
            LAYOUT_CLASS_COMPACT => Self::parse_compact(data),
            LAYOUT_CLASS_CONTIGUOUS => Self::parse_contiguous(data, offset_size, length_size),
            LAYOUT_CLASS_CHUNKED => {
                ensure_len(data, pos, 3)?;
                let flags = data[pos];
                let dimensionality = data[pos + 1] as usize;
                let dim_size_encoded_length = data[pos + 2] as usize;
                let mut p = pos + 3;

                // dimension sizes
                ensure_len(data, p, dimensionality * dim_size_encoded_length)?;
                let mut chunk_dimensions = Vec::with_capacity(dimensionality);
                for _ in 0..dimensionality {
                    let val = match dim_size_encoded_length {
                        1 => data[p] as u32,
                        2 => u16::from_le_bytes([data[p], data[p + 1]]) as u32,
                        4 => u32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]),
                        8 => {
                            // Truncate to u32
                            u32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]])
                        }
                        _ => {
                            return Err(FormatError::UnexpectedEof {
                                expected: p + dim_size_encoded_length,
                                available: data.len(),
                            });
                        }
                    };
                    chunk_dimensions.push(val);
                    p += dim_size_encoded_length;
                }

                Ok(DataLayout::Chunked {
                    chunk_dimensions,
                    index: parse_chunk_index(data, p, flags, offset_size, length_size)?,
                })
            }
            LAYOUT_CLASS_VIRTUAL => Ok(DataLayout::Virtual),
            _ => Err(FormatError::InvalidLayoutClass(layout_class)),
        }
    }

    fn parse_compact(data: &[u8]) -> Result<DataLayout, FormatError> {
        let pos = LAYOUT_PROPERTIES_OFFSET;
        ensure_len(data, pos, 2)?;
        let data_size = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        ensure_len(data, COMPACT_DATA_OFFSET, data_size)?;
        Ok(DataLayout::Compact {
            data: data[COMPACT_DATA_OFFSET..COMPACT_DATA_OFFSET + data_size].to_vec(),
        })
    }

    fn parse_contiguous(
        data: &[u8],
        offset_size: u8,
        length_size: u8,
    ) -> Result<DataLayout, FormatError> {
        let pos = LAYOUT_PROPERTIES_OFFSET;
        let os = offset_size as usize;
        ensure_len(data, pos, os + length_size as usize)?;
        Ok(DataLayout::Contiguous {
            address: read_optional_offset(data, pos, offset_size)?.map(StoredAddress::new),
            size: read_length(data, pos + os, length_size)?,
        })
    }
}

/// The index a chunked dataset's chunks are located through, and the address
/// the message stores for it.
///
/// A version 3 message indexes every chunked dataset with a version 1 B-tree. A
/// version 4 message stores one of the five types the Chunk Indexing Type table
/// of "The Data Layout Message" lists, in the [format specification, version
/// 4.0][spec]. libhdf5 picks between them by the dataset's shape, its unlimited
/// dimensions, its allocation time, and its filters
/// (`H5D__layout_set_latest_indexing` in `H5Dlayout.c`, HDF5 1.14.6). An address
/// is `None` where the message stores the undefined address, which is storage
/// that was never allocated.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ChunkIndexLayout {
    /// A version 1 B-tree indexes the chunks, the index of every version 3
    /// message, which is what libhdf5 writes for a chunked dataset outside the
    /// latest format bounds.
    BTreeV1 {
        /// Address of the B-tree's root node.
        address: Option<StoredAddress>,
    },
    /// One chunk holds the whole dataset, and the message stores the address of
    /// that chunk itself.
    SingleChunk {
        /// The chunk's stored size and filter mask, present where the layout
        /// flags mark the chunk as filtered.
        filtered: Option<FilteredSingleChunk>,
        /// Address of the single chunk.
        address: Option<StoredAddress>,
    },
    /// The chunks lie in one array in the order of their coordinates, and a
    /// reader computes a chunk's address from its position.
    Implicit {
        /// Base address of the array of chunks.
        address: Option<StoredAddress>,
    },
    /// A fixed array indexes the chunks of a dataspace of fixed maximum
    /// dimensions.
    FixedArray {
        /// Address of the fixed array header.
        address: Option<StoredAddress>,
    },
    /// An extensible array indexes the chunks of a dataspace with one unlimited
    /// dimension.
    ExtensibleArray {
        /// Address of the extensible array header.
        address: Option<StoredAddress>,
    },
    /// A version 2 B-tree indexes the chunks of a dataspace with more than one
    /// unlimited dimension.
    BTreeV2 {
        /// Address of the B-tree's header.
        address: Option<StoredAddress>,
    },
}

impl ChunkIndexLayout {
    /// Returns the address stored with the index, whichever index it is.
    pub(crate) fn address(self) -> Option<StoredAddress> {
        match self {
            ChunkIndexLayout::BTreeV1 { address }
            | ChunkIndexLayout::SingleChunk { address, .. }
            | ChunkIndexLayout::Implicit { address }
            | ChunkIndexLayout::FixedArray { address }
            | ChunkIndexLayout::ExtensibleArray { address }
            | ChunkIndexLayout::BTreeV2 { address } => address,
        }
    }

    /// Returns `true` if this crate can walk the index chunk by chunk, which an
    /// in-place overwrite and an in-place copy both need.
    ///
    /// Every index but the version 2 B-tree has a walker in
    /// [`crate::chunked_read`], and this mirrors that dispatch.
    pub(crate) fn enumerable(self) -> bool {
        !matches!(self, ChunkIndexLayout::BTreeV2 { .. })
    }
}

/// The Single Chunk indexing information of a filtered chunk.
///
/// The message stores both fields or neither, under the
/// [`SINGLE_INDEX_WITH_FILTER`] flag, so a caller that has one has the other.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FilteredSingleChunk {
    /// The chunk's size as stored, after its filters ran.
    pub(crate) filtered_size: u64,
    /// The filters skipped for this chunk, one bit per pipeline entry, counted
    /// from the first filter.
    pub(crate) filter_mask: u32,
}

/// Parses the chunk indexing type byte at `pos`, with the indexing information
/// and the address that follow it.
///
/// # Errors
///
/// Returns [`FormatError::InvalidChunkIndexType`] if the byte is outside 1 to 5.
/// `H5O__layout_decode` rejects the same bytes, and a version 1 B-tree in a
/// version 4 message with them (`H5Olayout.c`, HDF5 1.14.6).
fn parse_chunk_index(
    data: &[u8],
    pos: usize,
    flags: u8,
    offset_size: u8,
    length_size: u8,
) -> Result<ChunkIndexLayout, FormatError> {
    ensure_len(data, pos, 1)?;
    let index_type = data[pos];
    let info = pos + 1;
    // Every index type ends with its address, and `info_len` is how many bytes
    // of indexing information precede it. Only the Single Chunk index's
    // information is kept: a Fixed Array and an Extensible Array repeat their
    // creation parameters in their own header, which is where the walkers read
    // them, and no walker reads a version 2 B-tree.
    let address = |info_len: usize| {
        Ok::<_, FormatError>(
            read_optional_offset(data, info + info_len, offset_size)?.map(StoredAddress::new),
        )
    };
    Ok(match index_type {
        CHUNK_INDEX_SINGLE_CHUNK => {
            let filtered = read_filtered_single_chunk(data, info, flags, length_size)?;
            let info_len = filtered.map_or(0, |_| length_size as usize + FILTER_MASK_LEN);
            ChunkIndexLayout::SingleChunk {
                filtered,
                address: address(info_len)?,
            }
        }
        CHUNK_INDEX_IMPLICIT => ChunkIndexLayout::Implicit {
            address: address(0)?,
        },
        CHUNK_INDEX_FIXED_ARRAY => ChunkIndexLayout::FixedArray {
            address: address(FIXED_ARRAY_INFO_LEN)?,
        },
        CHUNK_INDEX_EXTENSIBLE_ARRAY => ChunkIndexLayout::ExtensibleArray {
            address: address(EXTENSIBLE_ARRAY_INFO_LEN)?,
        },
        CHUNK_INDEX_BTREE_V2 => ChunkIndexLayout::BTreeV2 {
            address: address(BTREE_V2_INFO_LEN)?,
        },
        other => return Err(FormatError::InvalidChunkIndexType(other)),
    })
}

/// Reads the Single Chunk indexing information at `pos`, which the message
/// stores only where `flags` marks the chunk as filtered.
fn read_filtered_single_chunk(
    data: &[u8],
    pos: usize,
    flags: u8,
    length_size: u8,
) -> Result<Option<FilteredSingleChunk>, FormatError> {
    if flags & SINGLE_INDEX_WITH_FILTER == 0 {
        return Ok(None);
    }
    let mask_at = pos + length_size as usize;
    ensure_len(data, mask_at, FILTER_MASK_LEN)?;
    Ok(Some(FilteredSingleChunk {
        filtered_size: read_length(data, pos, length_size)?,
        filter_mask: u32::from_le_bytes([
            data[mask_at],
            data[mask_at + 1],
            data[mask_at + 2],
            data[mask_at + 3],
        ]),
    }))
}

/// Byte offset of a layout message's class-specific property description, past
/// the version and layout class bytes the message opens with. From "The Data
/// Layout Message", version 4.0.
const LAYOUT_PROPERTIES_OFFSET: usize = 2;

/// Byte offset of a compact layout's inline data within the layout message
/// body: the property description, past its two-byte data size field. Version 3
/// and version 4 encode the compact class identically, so one constant covers
/// both.
///
/// Exported because the data is addressed in the *file* as well as parsed out
/// of it (repointing a stored object reference writes over these bytes in
/// place, issue #324), and a second derivation of the same offset would be
/// free to drift from the parser's.
pub(crate) const COMPACT_DATA_OFFSET: usize = LAYOUT_PROPERTIES_OFFSET + 2;

/// The layout message version libhdf5 1.6.3 and later write outside the latest
/// format bounds, from the Version table of "The Data Layout Message", version
/// 4.0.
const LAYOUT_VERSION_3: u8 = 3;

/// The layout message version whose chunked property description stores an
/// indexing type, and the first that defines the virtual storage class, from the
/// same table as [`LAYOUT_VERSION_3`].
const LAYOUT_VERSION_4: u8 = 4;

/// Compact storage, from the Layout Class table of "The Data Layout Message",
/// version 4.0.
const LAYOUT_CLASS_COMPACT: u8 = 0;

/// Contiguous storage, from the same table as [`LAYOUT_CLASS_COMPACT`].
const LAYOUT_CLASS_CONTIGUOUS: u8 = 1;

/// Chunked storage, from the same table as [`LAYOUT_CLASS_COMPACT`].
const LAYOUT_CLASS_CHUNKED: u8 = 2;

/// Virtual storage, from the same table as [`LAYOUT_CLASS_COMPACT`].
const LAYOUT_CLASS_VIRTUAL: u8 = 3;

/// The Single Chunk indexing type, from the Chunk Indexing Type table of "The
/// Data Layout Message", version 4.0.
const CHUNK_INDEX_SINGLE_CHUNK: u8 = 1;

/// The Implicit indexing type, from the same table as
/// [`CHUNK_INDEX_SINGLE_CHUNK`].
const CHUNK_INDEX_IMPLICIT: u8 = 2;

/// The Fixed Array indexing type, from the same table as
/// [`CHUNK_INDEX_SINGLE_CHUNK`].
const CHUNK_INDEX_FIXED_ARRAY: u8 = 3;

/// The Extensible Array indexing type, from the same table as
/// [`CHUNK_INDEX_SINGLE_CHUNK`].
const CHUNK_INDEX_EXTENSIBLE_ARRAY: u8 = 4;

/// The version 2 B-tree indexing type, from the same table as
/// [`CHUNK_INDEX_SINGLE_CHUNK`].
const CHUNK_INDEX_BTREE_V2: u8 = 5;

/// The chunked layout flag under which the message stores a filtered chunk's
/// size and filter mask ahead of a Single Chunk index's address.
///
/// The Flags table of "The Data Layout Message", version 4.0, names bit 1
/// `SINGLE_INDEX_WITH_FILTER`, while its Single Chunk paragraph points at bit 0.
/// libhdf5 writes and reads bit 1: `H5O_LAYOUT_CHUNK_SINGLE_INDEX_WITH_FILTER`
/// in `H5Oprivate.h`, and `H5O__layout_decode` in `H5Olayout.c`, HDF5 1.14.6.
const SINGLE_INDEX_WITH_FILTER: u8 = 0x02;

/// Width of the Filters for chunk field of the Single Chunk indexing
/// information, and of a chunk record's filter mask everywhere else.
const FILTER_MASK_LEN: usize = 4;

/// Width of the Fixed Array indexing information: the Page Bits parameter.
const FIXED_ARRAY_INFO_LEN: usize = 1;

/// Width of the Extensible Array indexing information: the Max Bits, Index
/// Elements, Min Pointers, Min Elements and Page Bits parameters.
const EXTENSIBLE_ARRAY_INFO_LEN: usize = 5;

/// Width of the version 2 B-tree indexing information: a four-byte Node Size
/// and the one-byte Split Percent and Merge Percent.
const BTREE_V2_INFO_LEN: usize = 6;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v3_compact() {
        let mut buf = vec![3u8, 0]; // version=3, class=0 (compact)
        buf.extend_from_slice(&5u16.to_le_bytes()); // data_size=5
        buf.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD, 0xEE]); // data
        let layout = DataLayout::parse(&buf, 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Compact {
                data: vec![0xAA, 0xBB, 0xCC, 0xDD, 0xEE]
            }
        );
    }

    #[test]
    fn v3_contiguous() {
        let mut buf = vec![3u8, 1]; // version=3, class=1 (contiguous)
        buf.extend_from_slice(&0x1000u64.to_le_bytes()); // address
        buf.extend_from_slice(&256u64.to_le_bytes()); // size
        let layout = DataLayout::parse(&buf, 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Contiguous {
                address: Some(StoredAddress::new(0x1000)),
                size: 256,
            }
        );
    }

    #[test]
    fn v3_contiguous_undefined_address() {
        let mut buf = vec![3u8, 1];
        buf.extend_from_slice(&[0xFF; 8]); // undefined address
        buf.extend_from_slice(&0u64.to_le_bytes()); // size
        let layout = DataLayout::parse(&buf, 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Contiguous {
                address: None,
                size: 0,
            }
        );
    }

    #[test]
    fn v3_chunked() {
        let mut buf = vec![3u8, 2]; // version=3, class=2 (chunked)
        buf.push(3); // dimensionality=3 (rank+1)
        buf.extend_from_slice(&0x2000u64.to_le_bytes()); // btree address
        // 3 chunk dim sizes × 4 bytes
        buf.extend_from_slice(&100u32.to_le_bytes());
        buf.extend_from_slice(&200u32.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes()); // last = element size
        let layout = DataLayout::parse(&buf, 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Chunked {
                chunk_dimensions: vec![100, 200, 8],
                index: ChunkIndexLayout::BTreeV1 {
                    address: Some(StoredAddress::new(0x2000))
                },
            }
        );
    }

    #[test]
    fn v4_compact() {
        let mut buf = vec![4u8, 0]; // version=4, class=0
        buf.extend_from_slice(&3u16.to_le_bytes());
        buf.extend_from_slice(&[1, 2, 3]);
        let layout = DataLayout::parse(&buf, 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Compact {
                data: vec![1, 2, 3]
            }
        );
    }

    #[test]
    fn v4_contiguous() {
        let mut buf = vec![4u8, 1];
        buf.extend_from_slice(&0x5000u64.to_le_bytes());
        buf.extend_from_slice(&512u64.to_le_bytes());
        let layout = DataLayout::parse(&buf, 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Contiguous {
                address: Some(StoredAddress::new(0x5000)),
                size: 512,
            }
        );
    }

    fn v4_chunked(flags: u8, index_type: u8, info: &[u8]) -> Vec<u8> {
        let mut buf = vec![4u8, 2]; // version=4, class=2 (chunked)
        buf.push(flags);
        buf.push(1); // dimensionality=1
        buf.push(4); // dim_size_encoded_length=4
        buf.extend_from_slice(&64u32.to_le_bytes()); // dim 0
        buf.push(index_type);
        buf.extend_from_slice(info);
        buf.extend_from_slice(&0x3000u64.to_le_bytes()); // address
        buf
    }

    #[test]
    fn v4_chunked_single_chunk_no_filters() {
        let layout = DataLayout::parse(&v4_chunked(0, 1, &[]), 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Chunked {
                chunk_dimensions: vec![64],
                index: ChunkIndexLayout::SingleChunk {
                    filtered: None,
                    address: Some(StoredAddress::new(0x3000)),
                },
            }
        );
    }

    #[test]
    fn v4_chunked_single_chunk_with_filters() {
        let mut info = 1024u64.to_le_bytes().to_vec();
        info.extend_from_slice(&0x0000_0003u32.to_le_bytes());
        let layout = DataLayout::parse(&v4_chunked(0x02, 1, &info), 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Chunked {
                chunk_dimensions: vec![64],
                index: ChunkIndexLayout::SingleChunk {
                    filtered: Some(FilteredSingleChunk {
                        filtered_size: 1024,
                        filter_mask: 3,
                    }),
                    address: Some(StoredAddress::new(0x3000)),
                },
            }
        );
    }

    #[test]
    fn v4_chunked_implicit_index_ignores_the_filter_flag() {
        let layout = DataLayout::parse(&v4_chunked(0x02, 2, &[]), 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Chunked {
                chunk_dimensions: vec![64],
                index: ChunkIndexLayout::Implicit {
                    address: Some(StoredAddress::new(0x3000))
                },
            }
        );
    }

    #[test]
    fn v4_chunked_fixed_array_skips_page_bits() {
        let layout = DataLayout::parse(&v4_chunked(0, 3, &[10]), 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Chunked {
                chunk_dimensions: vec![64],
                index: ChunkIndexLayout::FixedArray {
                    address: Some(StoredAddress::new(0x3000))
                },
            }
        );
    }

    #[test]
    fn v4_chunked_extensible_array_skips_creation_parameters() {
        let layout = DataLayout::parse(&v4_chunked(0, 4, &[32, 4, 4, 16, 10]), 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Chunked {
                chunk_dimensions: vec![64],
                index: ChunkIndexLayout::ExtensibleArray {
                    address: Some(StoredAddress::new(0x3000))
                },
            }
        );
    }

    #[test]
    fn v4_chunked_btree_v2_skips_node_parameters() {
        let mut info = 2048u32.to_le_bytes().to_vec();
        info.extend_from_slice(&[100, 40]);
        let layout = DataLayout::parse(&v4_chunked(0, 5, &info), 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Chunked {
                chunk_dimensions: vec![64],
                index: ChunkIndexLayout::BTreeV2 {
                    address: Some(StoredAddress::new(0x3000))
                },
            }
        );
    }

    #[test]
    fn v4_chunked_undefined_index_address_parses_as_none() {
        let mut buf = v4_chunked(0, 4, &[32, 4, 4, 16, 10]);
        let addr_at = buf.len() - 8;
        buf[addr_at..].copy_from_slice(&[0xFF; 8]);
        let layout = DataLayout::parse(&buf, 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Chunked {
                chunk_dimensions: vec![64],
                index: ChunkIndexLayout::ExtensibleArray { address: None },
            }
        );
    }

    #[test]
    fn v4_chunked_unknown_index_type_is_rejected() {
        let err = DataLayout::parse(&v4_chunked(0, 6, &[]), 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidChunkIndexType(6));
    }

    #[test]
    fn v4_chunked_truncated_index_type_reports_eof() {
        let buf = vec![4u8, 2, 0, 1, 4, 64, 0, 0, 0];
        let err = DataLayout::parse(&buf, 8, 8).unwrap_err();
        assert_eq!(
            err,
            FormatError::UnexpectedEof {
                expected: 10,
                available: 9,
            }
        );
    }

    #[test]
    fn invalid_version() {
        let buf = vec![5u8, 0, 0, 0];
        let err = DataLayout::parse(&buf, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidLayoutVersion(5));
    }

    #[test]
    fn invalid_class_v3() {
        let buf = vec![3u8, 5];
        let err = DataLayout::parse(&buf, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidLayoutClass(5));
    }

    #[test]
    fn invalid_class_v4() {
        let buf = vec![4u8, 7];
        let err = DataLayout::parse(&buf, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidLayoutClass(7));
    }

    #[test]
    fn v3_contiguous_4byte_offsets() {
        let mut buf = vec![3u8, 1];
        buf.extend_from_slice(&0x800u32.to_le_bytes());
        buf.extend_from_slice(&24u32.to_le_bytes());
        let layout = DataLayout::parse(&buf, 4, 4).unwrap();
        assert_eq!(
            layout,
            DataLayout::Contiguous {
                address: Some(StoredAddress::new(0x800)),
                size: 24,
            }
        );
    }

    #[test]
    fn v4_virtual() {
        let buf = vec![4u8, 3];
        let layout = DataLayout::parse(&buf, 8, 8).unwrap();
        assert_eq!(layout, DataLayout::Virtual);
    }
}
