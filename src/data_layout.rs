//! HDF5 Data Layout message parsing (message type 0x0008).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

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
        address: Option<u64>,
        /// Size of the data in bytes.
        size: u64,
    },
    /// Chunked: data stored in chunks via a B-tree.
    Chunked {
        /// Chunk dimension sizes.
        chunk_dimensions: Vec<u32>,
        /// B-tree address, or `None` if undefined.
        btree_address: Option<u64>,
        /// Layout version (3 or 4).
        version: u8,
        /// Chunk index type (v4 only).
        chunk_index_type: Option<u8>,
        /// Filtered size for v4 single chunk with filters.
        single_chunk_filtered_size: Option<u64>,
        /// Filter mask for v4 single chunk with filters.
        single_chunk_filter_mask: Option<u32>,
    },
    /// Virtual dataset layout (v4 only).
    Virtual,
}

/// Byte offset of a compact layout's inline data within the layout message
/// body: version(1) + layout class(1) + data size(2). Version 3 and version 4
/// encode the compact class identically, so one constant covers both.
///
/// Exported because the data is addressed in the *file* as well as parsed out of
/// it — repointing a stored object reference writes over these bytes in place
/// (issue #324) — and a second derivation of the same offset would be free to
/// drift from the parser's.
pub(crate) const COMPACT_DATA_OFFSET: usize = 4;

impl DataLayout {
    /// Parse a data layout message from raw message bytes.
    ///
    /// `offset_size` and `length_size` come from the superblock.
    pub fn parse(data: &[u8], offset_size: u8, length_size: u8) -> Result<DataLayout, FormatError> {
        ensure_len(data, 0, 2)?;
        let version = data[0];
        let layout_class = data[1];

        match version {
            3 => Self::parse_v3(data, layout_class, offset_size, length_size),
            4 => Self::parse_v4(data, layout_class, offset_size, length_size),
            _ => Err(FormatError::InvalidLayoutVersion(version)),
        }
    }

    fn parse_v3(
        data: &[u8],
        layout_class: u8,
        offset_size: u8,
        length_size: u8,
    ) -> Result<DataLayout, FormatError> {
        let pos = 2;
        match layout_class {
            0 => Self::parse_compact(data),
            1 => Self::parse_contiguous(data, offset_size, length_size),
            2 => {
                // Chunked
                ensure_len(data, pos, 1)?;
                let dimensionality = data[pos] as usize;
                let mut p = pos + 1;
                // btree address first
                let os = offset_size as usize;
                ensure_len(data, p, os)?;
                let btree_address = read_optional_offset(data, p, offset_size)?;
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
                    btree_address,
                    version: 3,
                    chunk_index_type: None,
                    single_chunk_filtered_size: None,
                    single_chunk_filter_mask: None,
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
        let pos = 2;
        match layout_class {
            0 => Self::parse_compact(data),
            1 => Self::parse_contiguous(data, offset_size, length_size),
            2 => {
                // Chunked v4
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

                // chunk index type
                ensure_len(data, p, 1)?;
                let chunk_index_type = data[p];
                p += 1;

                // Parse index-specific fields
                let mut single_chunk_filtered_size = None;
                let mut single_chunk_filter_mask = None;
                let btree_address = match chunk_index_type {
                    1 => {
                        // Single chunk
                        // H5O_LAYOUT_CHUNK_SINGLE_INDEX_WITH_FILTER = 0x02
                        let filters_present = flags & 0x02 != 0;
                        if filters_present {
                            // filtered_size(length_size) + filter_mask(4) + address(offset_size)
                            let ls = length_size as usize;
                            let os = offset_size as usize;
                            ensure_len(data, p, ls + 4 + os)?;
                            single_chunk_filtered_size = Some(read_length(data, p, length_size)?);
                            p += ls;
                            single_chunk_filter_mask = Some(u32::from_le_bytes([
                                data[p],
                                data[p + 1],
                                data[p + 2],
                                data[p + 3],
                            ]));
                            p += 4;
                            read_optional_offset(data, p, offset_size)?
                        } else {
                            // just address(offset_size)
                            ensure_len(data, p, offset_size as usize)?;
                            read_optional_offset(data, p, offset_size)?
                        }
                    }
                    2 => {
                        // Implicit: just address
                        ensure_len(data, p, offset_size as usize)?;
                        read_optional_offset(data, p, offset_size)?
                    }
                    3 => {
                        // Fixed Array: max_dblk_page_nelmts_bits(1) + address(offset_size)
                        ensure_len(data, p, 1 + offset_size as usize)?;
                        p += 1; // skip max_dblk_page_nelmts_bits
                        read_optional_offset(data, p, offset_size)?
                    }
                    4 => {
                        // Extensible Array: 5 creation params + address(offset_size)
                        ensure_len(data, p, 5 + offset_size as usize)?;
                        p += 5; // skip EA creation parameters
                        read_optional_offset(data, p, offset_size)?
                    }
                    5 => {
                        // B-tree v2: node_size(4) + split_percent(1) + merge_percent(1) + address
                        ensure_len(data, p, 6 + offset_size as usize)?;
                        p += 6;
                        read_optional_offset(data, p, offset_size)?
                    }
                    _ => {
                        // Unknown index type: try just address
                        ensure_len(data, p, offset_size as usize)?;
                        read_optional_offset(data, p, offset_size)?
                    }
                };

                Ok(DataLayout::Chunked {
                    chunk_dimensions,
                    btree_address,
                    version: 4,
                    chunk_index_type: Some(chunk_index_type),
                    single_chunk_filtered_size,
                    single_chunk_filter_mask,
                })
            }
            3 => Ok(DataLayout::Virtual),
            _ => Err(FormatError::InvalidLayoutClass(layout_class)),
        }
    }

    fn parse_compact(data: &[u8]) -> Result<DataLayout, FormatError> {
        let pos = 2;
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
        let pos = 2;
        let os = offset_size as usize;
        ensure_len(data, pos, os + length_size as usize)?;
        Ok(DataLayout::Contiguous {
            address: read_optional_offset(data, pos, offset_size)?,
            size: read_length(data, pos + os, length_size)?,
        })
    }
}

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
                address: Some(0x1000),
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
                btree_address: Some(0x2000),
                version: 3,
                chunk_index_type: None,
                single_chunk_filtered_size: None,
                single_chunk_filter_mask: None,
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
                address: Some(0x5000),
                size: 512,
            }
        );
    }

    #[test]
    fn v4_chunked_single_chunk_no_filters() {
        let mut buf = vec![4u8, 2]; // version=4, class=2
        buf.push(0); // flags (no filters)
        buf.push(2); // dimensionality=2
        buf.push(4); // dim_size_encoded_length=4
        buf.extend_from_slice(&64u32.to_le_bytes()); // dim 0
        buf.extend_from_slice(&32u32.to_le_bytes()); // dim 1
        buf.push(1); // chunk_index_type=1 (single chunk)
        buf.extend_from_slice(&0x3000u64.to_le_bytes()); // chunk address
        let layout = DataLayout::parse(&buf, 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Chunked {
                chunk_dimensions: vec![64, 32],
                btree_address: Some(0x3000),
                version: 4,
                chunk_index_type: Some(1),
                single_chunk_filtered_size: None,
                single_chunk_filter_mask: None,
            }
        );
    }

    #[test]
    fn v4_chunked_single_chunk_with_filters() {
        let mut buf = vec![4u8, 2]; // version=4, class=2
        buf.push(0x02); // flags bit 1 = single chunk with filter
        buf.push(1); // dimensionality=1
        buf.push(4); // dim_size_encoded_length=4
        buf.extend_from_slice(&128u32.to_le_bytes()); // dim 0
        buf.push(1); // chunk_index_type=1 (single chunk)
        // filters present: filtered_size(8) + filter_mask(4) + address(8)
        buf.extend_from_slice(&1024u64.to_le_bytes()); // filtered size
        buf.extend_from_slice(&0u32.to_le_bytes()); // filter mask
        buf.extend_from_slice(&0x4000u64.to_le_bytes()); // address
        let layout = DataLayout::parse(&buf, 8, 8).unwrap();
        assert_eq!(
            layout,
            DataLayout::Chunked {
                chunk_dimensions: vec![128],
                btree_address: Some(0x4000),
                version: 4,
                chunk_index_type: Some(1),
                single_chunk_filtered_size: Some(1024),
                single_chunk_filter_mask: Some(0),
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
                address: Some(0x800),
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
