//! Parses the version 2 B-tree header and nodes and computes internal-node child-pointer widths.
//!
//! "Version 2 B-trees" of the [format specification, version 4.0][spec] defines the `BTHD` header,
//! `BTIN` internal nodes, and `BTLF` leaf nodes. The header's "Node Size" field is the size in
//! bytes of every B-tree node. The header itself has a separate variable-length layout whose
//! address and length fields use the superblock widths. [`BTreeV2NodeInfo`] computes the
//! record-count widths that the internal-node child-pointer layout leaves variable.
//!
//! [`HugeObjectRecord`] and [`AttributeRecord`] are the records of the huge-object index of a
//! fractal heap and of the attribute indexes, with their encoders. A reader parses a huge-object
//! record with [`BTreeV2Record::huge_object`], and reads the heap ID and the creation order of the
//! other records with the other methods of [`BTreeV2Record`]. [`BTreeV2ChunkRecordContext`]
//! decodes dataset chunk-index client types 10 and 11 into [`BTreeV2ChunkRecord`] values.
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2

use core::num::{NonZeroU16, NonZeroU64};

use alloc::vec;
use alloc::vec::Vec;

#[cfg(feature = "checksum")]
use byteorder::ByteOrder;
#[cfg(feature = "checksum")]
use byteorder::LittleEndian;

use crate::address::StoredAddress;
use crate::bytes;
use crate::chunk_record;
use crate::convert::Narrow;
use crate::data_layout::LayoutVersion;
use crate::error::FormatError;
use crate::message_flags::MessageFlags;
use crate::metadata_source::MetadataSource;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// A version 2 B-tree header, signature `BTHD`: the geometry every node of the tree shares, and
/// the address of the root node.
///
/// [`parse`](Self::parse) accepts version 0, the one version the specification defines, and keeps
/// the fields from the type to the total number of records, except the split and merge percentages.
///
/// The header is defined in "Version 2 B-trees" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BTreeV2Header {
    /// The type of the tree, such as 5 for the name index of a group's links or 8 for the name
    /// index of an object's attributes. The layout of the records depends on the type.
    pub tree_type: u8,
    /// The size in bytes of every node of the tree.
    pub node_size: u32,
    /// The size in bytes of one record.
    pub record_size: u16,
    /// The depth of the tree, 0 where the root node is a leaf.
    pub depth: u16,
    /// The address of the root node.
    pub root_node_address: StoredAddress,
    /// The number of records in the root node.
    pub num_records_in_root: u16,
    /// The number of records in the whole tree.
    pub total_records: u64,
}

/// One record of a version 2 B-tree, with its bytes as the node stores them.
///
/// The layout of the bytes depends on the [`tree_type`](BTreeV2Header::tree_type) of the tree, and
/// a caller decodes them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BTreeV2Record {
    /// The `record_size` bytes of the record.
    pub data: Vec<u8>,
}

/// Returns the fewest bytes that hold `max_nrec`, and 1 for 0.
///
/// A child pointer stores each of its record counts in the width this returns for the largest
/// count the field can hold.
fn bytes_for_max_records(max_nrec: u64) -> usize {
    if max_nrec == 0 {
        return 1;
    }
    let bits = 64 - max_nrec.leading_zeros() as usize;
    bits.div_ceil(8)
}

/// Reads the little-endian unsigned integer of `width` bytes at `pos`, for a width of 1 to 8.
fn read_var_uint(data: &[u8], pos: usize, width: usize) -> Result<u64, FormatError> {
    bytes::ensure_len(data, pos, width)?;
    let mut val = 0u64;
    for i in 0..width {
        val |= (data[pos + i] as u64) << (i * 8);
    }
    Ok(val)
}

/// Returns the encoded size of a version 2 B-tree's version 0 `BTHD` header.
///
/// "Version 2 B-trees" of the [format specification, version 4.0][spec] lays out the fields through
/// "Merge Percent" in 16 bytes. "Root Node Address" follows at the superblock's "Size of Offsets",
/// then the 2-byte "Number of Records in Root Node", "Total Number of Records in B-tree" at the
/// superblock's "Size of Lengths", and the 4-byte checksum. `offset_width` and `length_width`
/// supply those two variable widths.
///
/// The encoded length is `16 + offset_width + 2 + length_width + 4`. The header's "Node Size" field
/// does not contribute another variable-sized region: it describes `BTIN` and `BTLF` nodes.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
pub const fn btree_v2_header_size(offset_width: OffsetWidth, length_width: LengthWidth) -> usize {
    // signature(4) + version(1) + type(1) + node size(4) + record size(2) +
    // depth(2) + split %(1) + merge %(1) + root address + records in root(2) +
    // total records + checksum(4)
    4 + 1
        + 1
        + 4
        + 2
        + 2
        + 1
        + 1
        + offset_width.get() as usize
        + 2
        + length_width.get() as usize
        + 4
}

impl BTreeV2Header {
    /// Parses the version 2 B-tree header at `offset` in `file_data`.
    ///
    /// `offset_size` and `length_size` are the superblock's "Size of Offsets" and "Size of
    /// Lengths" bytes. With the `checksum` feature, `parse` also verifies the header's checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeV2Signature`] if the header does not begin with `BTHD`,
    /// [`FormatError::InvalidBTreeV2Version`] if its version is not 0,
    /// [`FormatError::UnexpectedEof`] if it runs past the end of `file_data`,
    /// [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if a width is not
    /// 2, 4, or 8, and, with the `checksum` feature, [`FormatError::ChecksumMismatch`] if the
    /// stored checksum differs from the computed one.
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<BTreeV2Header, FormatError> {
        bytes::ensure_len(file_data, offset, 4)?;
        if &file_data[offset..offset + 4] != b"BTHD" {
            return Err(FormatError::InvalidBTreeV2Signature);
        }

        bytes::ensure_len(file_data, offset, 4 + 1 + 1 + 4 + 2 + 2 + 1 + 1)?;
        let version = file_data[offset + 4];
        if version != 0 {
            return Err(FormatError::InvalidBTreeV2Version(version));
        }

        let tree_type = file_data[offset + 5];
        let node_size = u32::from_le_bytes([
            file_data[offset + 6],
            file_data[offset + 7],
            file_data[offset + 8],
            file_data[offset + 9],
        ]);
        let record_size = u16::from_le_bytes([file_data[offset + 10], file_data[offset + 11]]);
        let depth = u16::from_le_bytes([file_data[offset + 12], file_data[offset + 13]]);
        let _split_percent = file_data[offset + 14];
        let _merge_percent = file_data[offset + 15];

        let mut pos = offset + 16;
        let root_node_address =
            StoredAddress::new(bytes::read_offset(file_data, pos, offset_size)?);
        pos += offset_size as usize;

        bytes::ensure_len(file_data, pos, 2)?;
        let num_records_in_root = u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);
        pos += 2;

        let total_records = bytes::read_length(file_data, pos, length_size)?;
        #[allow(unused_assignments)]
        {
            pos += length_size as usize;
        }

        // Validate header checksum
        #[cfg(feature = "checksum")]
        {
            bytes::ensure_len(file_data, pos, 4)?;
            let stored = LittleEndian::read_u32(&file_data[pos..pos + 4]);
            let computed = crate::checksum::jenkins_lookup3(&file_data[offset..pos]);
            if computed != stored {
                return Err(FormatError::ChecksumMismatch {
                    expected: stored,
                    computed,
                });
            }
        }

        Ok(BTreeV2Header {
            tree_type,
            node_size,
            record_size,
            depth,
            root_node_address,
            num_records_in_root,
            total_records,
        })
    }

    /// Parses the version 2 B-tree header at `address` in `source`.
    ///
    /// Reads at most 64 bytes, which hold the header at every width the format allows, and fewer
    /// where the file ends first, then parses them as [`parse`](Self::parse) does.
    ///
    /// # Errors
    ///
    /// Returns the errors [`parse`](Self::parse) returns, and the error `source` returns if a read
    /// fails.
    pub fn parse_from_source(
        source: &(impl MetadataSource + ?Sized),
        address: u64,
        offset_size: u8,
        length_size: u8,
    ) -> Result<BTreeV2Header, FormatError> {
        // 16 fixed prefix bytes + root address + 2 (records in the root) + total-records
        // field + 4 checksum: <= 64 with 8-byte offsets/lengths.
        const MAX_HEADER: u64 = 64;
        let window = MAX_HEADER
            .min(source.len().saturating_sub(address))
            .to_usize()?;
        let buf = source.read_metadata_at(address, window)?;
        Self::parse(&buf, 0, offset_size, length_size)
    }
}

/// The dataset context needed to decode a version 2 B-tree chunk record.
///
/// Types 10 and 11 store scaled chunk coordinates for the dataset dimensions. Type 10 omits
/// chunk size and filter-mask fields because its chunks are unfiltered and have the fixed logical
/// chunk size. Type 11 stores both. For version 4 layout messages, the stored-size field is one
/// byte wider than the fewest bytes needed for the logical chunk size, capped at 8 bytes.
/// `H5D_BT2_COMPUTE_CHUNK_SIZE_LEN` in `H5Dbtree2.c` (HDF5 2.2.0) applies that rule.
///
/// The two record layouts are defined in "Version 2 B-trees" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BTreeV2ChunkRecordContext {
    tree_type: u8,
    offset_width: OffsetWidth,
    rank: usize,
    fixed_chunk_size: NonZeroU64,
    stored_size_width: usize,
    record_size: usize,
}

impl BTreeV2ChunkRecordContext {
    /// Creates a context for a type 10 or type 11 chunk index.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidBTreeNodeType`] if `tree_type` is not 10 or 11,
    /// [`FormatError::InvalidLayoutVersion`] if `layout_version` is not 4, and
    /// [`FormatError::InvalidChunkGeometry`] if the record size cannot be represented.
    pub fn new(
        tree_type: u8,
        offset_width: OffsetWidth,
        rank: usize,
        fixed_chunk_size: NonZeroU64,
        layout_version: LayoutVersion,
    ) -> Result<Self, FormatError> {
        if !matches!(tree_type, BTREE_V2_CHUNK | BTREE_V2_FILTERED_CHUNK) {
            return Err(FormatError::InvalidBTreeNodeType(tree_type));
        }
        if layout_version != LayoutVersion::Four {
            return Err(FormatError::InvalidLayoutVersion(layout_version.get()));
        }

        let stored_size_width = if tree_type == BTREE_V2_FILTERED_CHUNK {
            chunk_record::chunk_element_encoding(fixed_chunk_size.get(), offset_width, true)
                .chunk_size_bytes
        } else {
            0
        };
        let coordinate_bytes =
            rank.checked_mul(CHUNK_COORDINATE_LEN)
                .ok_or(FormatError::InvalidChunkGeometry(
                    "version 2 B-tree chunk record size overflows",
                ))?;
        let record_size = usize::from(offset_width.get())
            .checked_add(stored_size_width)
            .and_then(|size| {
                size.checked_add(if tree_type == BTREE_V2_FILTERED_CHUNK {
                    FILTER_MASK_LEN
                } else {
                    0
                })
            })
            .and_then(|size| size.checked_add(coordinate_bytes))
            .ok_or(FormatError::InvalidChunkGeometry(
                "version 2 B-tree chunk record size overflows",
            ))?;
        if record_size > usize::from(u16::MAX) {
            return Err(FormatError::InvalidChunkGeometry(
                "version 2 B-tree chunk record is too large",
            ));
        }

        Ok(Self {
            tree_type,
            offset_width,
            rank,
            fixed_chunk_size,
            stored_size_width,
            record_size,
        })
    }

    /// Returns the exact encoded size of one record.
    pub fn record_size(self) -> usize {
        self.record_size
    }

    /// Requires the B-tree header's declared record size to match this record layout.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DataSizeMismatch`] if `actual` differs from the exact record size.
    pub fn require_record_size(self, actual: u16) -> Result<(), FormatError> {
        let actual = usize::from(actual);
        if actual != self.record_size {
            return Err(FormatError::DataSizeMismatch {
                expected: self.record_size,
                actual,
            });
        }
        Ok(())
    }

    /// Decodes a raw record as the chunk-index client type this context describes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DataSizeMismatch`] unless the record has exactly the expected
    /// length, [`FormatError::ChunkedReadError`] for an undefined chunk address, and
    /// [`FormatError::InvalidChunkGeometry`] for a zero stored size in a filtered record.
    pub fn decode(self, record: &BTreeV2Record) -> Result<BTreeV2ChunkRecord, FormatError> {
        if record.data.len() != self.record_size {
            return Err(FormatError::DataSizeMismatch {
                expected: self.record_size,
                actual: record.data.len(),
            });
        }

        let address = bytes::read_optional_offset_width(&record.data, 0, self.offset_width)?
            .ok_or_else(|| {
                FormatError::ChunkedReadError(
                    "version 2 B-tree chunk record has an undefined chunk address".into(),
                )
            })?;
        let mut pos = usize::from(self.offset_width.get());
        let (stored_size, filter_mask) = if self.tree_type == BTREE_V2_FILTERED_CHUNK {
            let stored_size = read_var_uint(&record.data, pos, self.stored_size_width)?;
            if stored_size == 0 {
                return Err(FormatError::InvalidChunkGeometry(
                    "filtered version 2 B-tree chunk record has zero stored size",
                ));
            }
            pos += self.stored_size_width;
            let mut fields = bytes::Fields::new(&record.data, pos);
            let filter_mask = fields.u32()?;
            pos = fields.pos();
            (stored_size, filter_mask)
        } else {
            (self.fixed_chunk_size.get(), 0)
        };

        let mut scaled_offsets = Vec::with_capacity(self.rank);
        for _ in 0..self.rank {
            scaled_offsets.push(read_var_uint(&record.data, pos, CHUNK_COORDINATE_LEN)?);
            pos += CHUNK_COORDINATE_LEN;
        }

        Ok(BTreeV2ChunkRecord {
            address: StoredAddress::new(address),
            stored_size,
            filter_mask,
            scaled_offsets,
        })
    }
}

/// A decoded type 10 or type 11 record of a version 2 B-tree chunk index.
///
/// The B-tree stores chunk coordinates scaled by the dataset's chunk dimensions. The reader
/// converts [`scaled_offsets`](Self::scaled_offsets) to ordinary element offsets before exposing
/// the chunk.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BTreeV2ChunkRecord {
    /// The address of the chunk in the file.
    pub address: StoredAddress,
    /// The size of the chunk in the file in bytes, after its filters.
    pub stored_size: u64,
    /// The filters skipped for the chunk: bit `i` is set where filter `i` was not applied.
    pub filter_mask: u32,
    /// The chunk coordinates, in units of whole chunks, for each dataset dimension.
    pub scaled_offsets: Vec<u64>,
}

impl BTreeV2Record {
    /// Parses the record as a record of the huge-object index of a fractal heap, type 1.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the record ends before its three fields.
    pub fn huge_object(
        &self,
        offset_width: OffsetWidth,
        length_width: LengthWidth,
    ) -> Result<HugeObjectRecord, FormatError> {
        let os = usize::from(offset_width.get());
        let ls = usize::from(length_width.get());
        Ok(HugeObjectRecord {
            address: StoredAddress::new(bytes::read_offset_width(&self.data, 0, offset_width)?),
            length: bytes::read_length_width(&self.data, os, length_width)?,
            id: bytes::read_length_width(&self.data, os + ls, length_width)?,
        })
    }

    /// Parses the record as a record of the huge-object index of a fractal heap, type 3.
    ///
    /// Type 3 is the non-filtered layout used when heap IDs hold the huge object's address and
    /// length directly. The B-tree still tracks every huge object so the heap can enumerate and
    /// delete all of them.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the record ends before its two fields.
    pub fn huge_object_direct(
        &self,
        offset_width: OffsetWidth,
        length_width: LengthWidth,
    ) -> Result<HugeObjectDirectRecord, FormatError> {
        let os = usize::from(offset_width.get());
        Ok(HugeObjectDirectRecord {
            address: StoredAddress::new(bytes::read_offset_width(&self.data, 0, offset_width)?),
            length: bytes::read_length_width(&self.data, os, length_width)?,
        })
    }

    /// Returns the heap ID in the record of a link index of `tree_type`, or `None` if the record
    /// ends before an ID of `heap_id_length` bytes.
    ///
    /// The heap ID follows the 4-byte name hash in a type 5 record of a name index, and the
    /// 8-byte creation order in a type 6 record of a creation-order index. A record of any other
    /// `tree_type` is read as type 6.
    pub fn link_heap_id(&self, tree_type: u8, heap_id_length: u16) -> Option<&[u8]> {
        let at = if tree_type == BTREE_V2_LINK_NAME {
            usize::from(NAME_HASH_LEN)
        } else {
            LINK_CREATION_ORDER_LEN
        };
        self.data.get(at..at + usize::from(heap_id_length))
    }

    /// Returns the heap ID that opens a type 8 or type 9 record of an attribute index, or `None`
    /// if the record is shorter than `heap_id_length` bytes.
    pub fn attribute_heap_id(&self, heap_id_length: u16) -> Option<&[u8]> {
        self.data.get(..usize::from(heap_id_length))
    }

    /// Returns the creation order in a type 8 or type 9 record of an attribute index, past a heap
    /// ID of `heap_id_length` bytes and the message flags, or `None` if the record ends before it.
    pub fn attribute_creation_order(&self, heap_id_length: u16) -> Option<u32> {
        let at = usize::from(heap_id_length) + usize::from(MESSAGE_FLAGS_LEN);
        self.data
            .get(at..)
            .and_then(|rest| rest.first_chunk())
            .map(|&bytes| u32::from_le_bytes(bytes))
    }
}

/// A record of the version 2 B-tree that indexes the huge objects of a fractal heap, type 1: the
/// objects the heap does not filter and whose heap IDs store the key of their record.
///
/// The record is defined in "Version 2 B-trees" of the [format specification, version
/// 4.0][spec], which calls the key the Huge Object ID.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HugeObjectRecord {
    /// The address of the object.
    pub address: StoredAddress,
    /// The length of the object in bytes.
    pub length: u64,
    /// The key of the object, which its heap ID stores.
    pub id: u64,
}

/// A record of a fractal heap huge-object B-tree, type 3: the object address and length.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HugeObjectDirectRecord {
    /// The address of the object.
    pub address: StoredAddress,
    /// The length of the object in bytes.
    pub length: u64,
}

impl HugeObjectDirectRecord {
    /// Returns the encoded length of a type-3 huge-object record.
    pub const fn size(offset_width: OffsetWidth, length_width: LengthWidth) -> u16 {
        offset_width.get() as u16 + length_width.get() as u16
    }
}

impl HugeObjectRecord {
    /// Returns the length in bytes of a record whose address is `offset_width` bytes wide and
    /// whose length and key are `length_width` bytes wide each.
    pub const fn size(offset_width: OffsetWidth, length_width: LengthWidth) -> u16 {
        offset_width.get() as u16 + 2 * length_width.get() as u16
    }

    /// Appends the record to `buf`, with the address in `offset_width` bytes and the length and
    /// the key in `length_width` bytes each.
    pub fn encode(&self, buf: &mut Vec<u8>, offset_width: OffsetWidth, length_width: LengthWidth) {
        let Self {
            address,
            length,
            id,
        } = *self;
        bytes::write_offset(buf, address.get(), offset_width);
        bytes::write_length(buf, length, length_width);
        bytes::write_length(buf, id, length_width);
    }
}

/// A record of a version 2 B-tree that indexes the attributes of an object in dense storage, type
/// 8 in the name index and type 9 in the creation-order index.
///
/// A type 9 record stores the three fields, and a type 8 record adds the hash of the attribute
/// name. Both are defined in "Version 2 B-trees" of the [format specification, version
/// 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttributeRecord<'a> {
    /// The heap ID of the Attribute message in the attribute heap of the object.
    pub heap_id: &'a [u8],
    /// The object header message flags of the Attribute message.
    pub flags: MessageFlags,
    /// The creation order of the attribute.
    pub creation_order: u32,
}

impl AttributeRecord<'_> {
    /// Returns the length in bytes of a type 8 record whose heap ID is `heap_id_length` bytes
    /// long.
    pub const fn name_record_size(heap_id_length: u16) -> u16 {
        Self::creation_order_record_size(heap_id_length) + NAME_HASH_LEN
    }

    /// Returns the length in bytes of a type 9 record whose heap ID is `heap_id_length` bytes
    /// long.
    pub const fn creation_order_record_size(heap_id_length: u16) -> u16 {
        heap_id_length + MESSAGE_FLAGS_LEN + ATTRIBUTE_CREATION_ORDER_LEN
    }

    /// Appends the type 8 record of the attribute to `buf`, with `name_hash` as the hash of its
    /// name.
    pub fn encode_name_record(&self, buf: &mut Vec<u8>, name_hash: u32) {
        self.encode_creation_order_record(buf);
        buf.extend_from_slice(&name_hash.to_le_bytes());
    }

    /// Appends the type 9 record of the attribute to `buf`.
    pub fn encode_creation_order_record(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(self.heap_id);
        buf.push(self.flags.get());
        buf.extend_from_slice(&self.creation_order.to_le_bytes());
    }
}

/// Returns the number of records a leaf of `node_size` bytes holds, or 0 where the leaf's prefix
/// and checksum fill the node or `record_size` is 0.
fn max_records_leaf(node_size: u32, record_size: u16) -> u64 {
    // Leaf overhead: signature(4) + version(1) + type(1) + checksum(4) = 10
    let overhead = 10u32;
    if node_size <= overhead || record_size == 0 {
        return 0;
    }
    ((node_size - overhead) / record_size as u32) as u64
}

/// The capacities of the nodes of a version 2 B-tree at each depth, and the widths of the record
/// counts in its child pointers.
///
/// A child pointer holds the address of a child node, the number of records in the child, and,
/// above depth 1, the number of records in the child's subtree. The file does not store the widths
/// of the two counts, and a reader computes them from the node size, the record size, the width of
/// an address, and the depth. The specification describes the subtree capacity as the product of
/// the capacities below it. `H5B2__hdr_init` in `H5B2hdr.c` (HDF5 2.2.0) computes it as
/// `cum_max_nrec[u] = (max_nrec[u] + 1) * cum_max_nrec[u - 1] + max_nrec[u]`, and the table follows
/// the C library.
///
/// [`BTreeV2Plan`](crate::btree_v2_write::BTreeV2Plan) lays out a tree from the same table, so a
/// written tree has the widths a reader computes for it.
///
/// The child pointers are defined in "Version 2 B-trees" of the [format specification, version
/// 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
pub struct BTreeV2NodeInfo {
    /// The width in bytes of the "Number of Records in Child Node" field of a child pointer. The C
    /// library sizes it for the leaf capacity, the largest of any depth, and uses the one width at
    /// every depth.
    max_nrec_size: usize,
    /// The width in bytes of the "Total Number of Records in Child Node" field of a child pointer,
    /// by the depth of the child. Entry 0 is 0, since a pointer to a leaf has no such field.
    cum_max_nrec_size: Vec<usize>,
    /// The number of records a full node holds, by depth. Entry 0 is the leaf capacity, and no
    /// entry is larger than the one before it, which `H5B2__hdr_init` asserts.
    max_nrec: Vec<u64>,
    /// The number of records a full subtree holds, by the depth of its root.
    cum_max_nrec: Vec<u64>,
}

impl BTreeV2NodeInfo {
    /// Returns the table of a tree whose root is a leaf.
    fn leaf_only(node_size: u32, record_size: u16) -> BTreeV2NodeInfo {
        let max_nrec0 = max_records_leaf(node_size, record_size);
        BTreeV2NodeInfo {
            max_nrec_size: bytes_for_max_records(max_nrec0),
            cum_max_nrec_size: vec![0],
            max_nrec: vec![max_nrec0],
            cum_max_nrec: vec![max_nrec0],
        }
    }

    /// Adds the capacities and the widths of one internal depth above the deepest in the table.
    fn push_level(&mut self, node_size: u32, record_size: u16, offset_size: u8) {
        let u = self.max_nrec.len();
        // Internal-pointer size at this level uses the *previous* level's
        // subtree-total width (H5B2_INT_POINTER_SIZE).
        let int_ptr = offset_size as usize + self.max_nrec_size + self.cum_max_nrec_size[u - 1];
        // Records that fit an internal node at this level (H5B2_NUM_INT_REC).
        let avail = (node_size as usize).saturating_sub(10 + int_ptr);
        let denom = record_size as usize + int_ptr;
        let max_nrec_u = avail.checked_div(denom).unwrap_or(0) as u64;
        let cum = max_nrec_u
            .saturating_add(1)
            .saturating_mul(self.cum_max_nrec[u - 1])
            .saturating_add(max_nrec_u);
        self.cum_max_nrec_size.push(bytes_for_max_records(cum));
        self.max_nrec.push(max_nrec_u);
        self.cum_max_nrec.push(cum);
    }

    /// Computes the table for a tree of `depth`, with nodes of `node_size` bytes and records of
    /// `record_size` bytes.
    ///
    /// `offset_size` is the superblock's "Size of Offsets" byte.
    pub fn compute(
        node_size: u32,
        record_size: u16,
        offset_size: u8,
        depth: u16,
    ) -> BTreeV2NodeInfo {
        let mut info = BTreeV2NodeInfo::leaf_only(node_size, record_size);
        for _ in 1..=depth {
            info.push_level(node_size, record_size, offset_size);
        }
        info
    }

    /// Computes the table for the shallowest tree that holds `records`, and returns it with the
    /// depth of that tree.
    ///
    /// Returns `None` if no tree of these node and record sizes holds `records`, which happens
    /// where an internal node fits 0 records.
    pub fn for_record_count(
        node_size: u32,
        record_size: u16,
        offset_size: u8,
        records: u64,
    ) -> Option<(BTreeV2NodeInfo, u16)> {
        let mut info = BTreeV2NodeInfo::leaf_only(node_size, record_size);
        let mut depth = 0u16;
        while *info.cum_max_nrec.last().expect("table is never empty") < records {
            info.push_level(node_size, record_size, offset_size);
            depth += 1;
            if *info.max_nrec.last().expect("just pushed") == 0 {
                return None;
            }
        }
        Some((info, depth))
    }

    /// Returns the number of records a full node at `depth` holds.
    ///
    /// # Panics
    ///
    /// Panics if `depth` is deeper than the tree the table was computed for.
    pub fn max_nrec(&self, depth: u16) -> u64 {
        self.max_nrec[depth as usize]
    }

    /// Returns the number of records a full subtree rooted at `depth` holds.
    pub(crate) fn cum_max_nrec(&self, depth: u16) -> u64 {
        self.cum_max_nrec[depth as usize]
    }

    /// Returns the width in bytes of the count of records in the child, one width at every depth.
    pub(crate) fn max_nrec_size(&self) -> usize {
        self.max_nrec_size
    }

    /// Returns the width in bytes of the count of records in the child's subtree, in the child
    /// pointers of a node at `depth`.
    ///
    /// The children are one depth below, so the width is `cum_max_nrec_size[depth - 1]`, and 0 at
    /// depth 1, where the children are leaves and the pointers have no such field.
    pub(crate) fn total_nrec_size(&self, depth: u16) -> usize {
        depth
            .checked_sub(1)
            .and_then(|below| self.cum_max_nrec_size.get(usize::from(below)))
            .copied()
            .unwrap_or(0)
    }

    /// Returns the width in bytes of one child pointer in a node at `depth`.
    fn child_ptr_size(&self, depth: u16, offset_size: u8) -> usize {
        offset_size as usize + self.max_nrec_size + self.total_nrec_size(depth)
    }
}

/// Parses the `num_records` records of the version 2 B-tree leaf node at `offset` in `file_data`.
///
/// With the `checksum` feature, the function also verifies the checksum that follows the records,
/// where `file_data` holds it.
///
/// The node is defined in "Version 2 B-trees" of the [format specification, version 4.0][spec].
///
/// # Errors
///
/// Returns [`FormatError::InvalidBTreeV2Signature`] if the node does not begin with `BTLF`,
/// [`FormatError::UnexpectedEof`] if the records run past the end of `file_data`, and, with the
/// `checksum` feature, [`FormatError::ChecksumMismatch`] if the stored checksum differs from the
/// computed one.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
pub fn parse_btree_v2_leaf_records(
    file_data: &[u8],
    offset: usize,
    num_records: u16,
    record_size: u16,
) -> Result<Vec<BTreeV2Record>, FormatError> {
    // signature(4) + version(1) + type(1) = 6 bytes header
    bytes::ensure_len(file_data, offset, 6)?;
    if &file_data[offset..offset + 4] != b"BTLF" {
        return Err(FormatError::InvalidBTreeV2Signature);
    }

    let pos = offset + 6;
    let rs = record_size as usize;
    let total = num_records as usize * rs;
    bytes::ensure_len(file_data, pos, total)?;

    #[cfg(feature = "checksum")]
    {
        let checksum_pos = pos + total;
        if file_data.len() >= checksum_pos + 4 {
            let stored = LittleEndian::read_u32(&file_data[checksum_pos..checksum_pos + 4]);
            let computed = crate::checksum::jenkins_lookup3(&file_data[offset..checksum_pos]);
            if computed != stored {
                return Err(FormatError::ChecksumMismatch {
                    expected: stored,
                    computed,
                });
            }
        }
    }

    let mut records = Vec::with_capacity(num_records as usize);
    for i in 0..num_records as usize {
        let start = pos + i * rs;
        records.push(BTreeV2Record {
            data: file_data[start..start + rs].to_vec(),
        });
    }
    Ok(records)
}

/// Parses the child pointers of the version 2 B-tree internal node at the start of `node`, and
/// returns the address and the record count of each child.
///
/// The node holds `num_records` records of `record_size` bytes and one more child pointer than
/// records. `depth` is the depth of the node, and `node_info` the table of its tree, which gives
/// the widths of the counts in each pointer. `offset_size` is the superblock's "Size of Offsets"
/// byte. The records sit at `node[6 + i * record_size..]`, where a caller reads them.
///
/// The node is defined in "Version 2 B-trees" of the [format specification, version 4.0][spec].
///
/// # Errors
///
/// Returns [`FormatError::InvalidBTreeV2Signature`] if the node does not begin with `BTIN`,
/// [`FormatError::UnexpectedEof`] if the records, child pointers, or checksum run past the end of
/// `node`, [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8, and, with the
/// `checksum` feature, [`FormatError::ChecksumMismatch`] if the stored checksum differs from the
/// computed one.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
pub fn parse_btree_v2_internal_child_pointers(
    node: &[u8],
    num_records: u16,
    depth: NonZeroU16,
    record_size: u16,
    offset_size: u8,
    node_info: &BTreeV2NodeInfo,
) -> Result<Vec<(StoredAddress, u16)>, FormatError> {
    // signature(4) + version(1) + type(1) = 6
    bytes::ensure_len(node, 0, 6)?;
    if &node[0..4] != b"BTIN" {
        return Err(FormatError::InvalidBTreeV2Signature);
    }

    let nr = num_records as usize;
    let rs = record_size as usize;
    // Records come first, then the child pointers.
    let mut pos = 6;
    bytes::ensure_len(node, pos, nr * rs)?;
    pos += nr * rs;

    let nrec_width = node_info.max_nrec_size;
    let total_nrec_width = node_info.total_nrec_size(depth.get());

    let num_children = nr + 1;
    let child_ptr_size = node_info.child_ptr_size(depth.get(), offset_size);
    bytes::ensure_len(node, pos, num_children * child_ptr_size)?;

    let mut children = Vec::with_capacity(num_children);
    for _ in 0..num_children {
        let addr = StoredAddress::new(bytes::read_offset(node, pos, offset_size)?);
        pos += offset_size as usize;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "`H5B2__hdr_init` in `H5B2hdr.c` (HDF5 2.2.0) asserts that `max_nrec_size` is at \
                      most 2 bytes, so the count in a file libhdf5 writes fits `u16`"
        )]
        let child_nrec = read_var_uint(node, pos, nrec_width)? as u16;
        pos += nrec_width;
        pos += total_nrec_width; // skip total-records-in-subtree
        children.push((addr, child_nrec));
    }

    let checksum_end = pos
        .checked_add(4)
        .ok_or(FormatError::InvalidBTreeV2Signature)?;
    bytes::ensure_len(node, pos, 4)?;
    crate::checksum::verify_trailing(&node[..checksum_end])?;

    Ok(children)
}

/// The type of a version 2 B-tree that indexes the huge objects of a fractal heap that the heap
/// does not filter and whose heap IDs store a key.
///
/// The type is defined in the Type table of "Version 2 B-trees" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v2
pub const BTREE_V2_HUGE_OBJECT: u8 = 1;

/// The type of a version 2 B-tree that indexes non-filtered huge objects whose heap IDs carry the
/// object address and length directly.
pub const BTREE_V2_HUGE_OBJECT_DIRECT: u8 = 3;

/// The type of a version 2 B-tree that indexes the link names of a group, from the same table as
/// [`BTREE_V2_HUGE_OBJECT`].
pub const BTREE_V2_LINK_NAME: u8 = 5;

/// The type of a version 2 B-tree that indexes the link creation order of a group, from the same
/// table as [`BTREE_V2_HUGE_OBJECT`].
pub const BTREE_V2_LINK_CREATION_ORDER: u8 = 6;

/// The type of a version 2 B-tree that indexes the attribute names of an object, from the same
/// table as [`BTREE_V2_HUGE_OBJECT`].
pub const BTREE_V2_ATTRIBUTE_NAME: u8 = 8;

/// The type of a version 2 B-tree that indexes the attribute creation order of an object, from
/// the same table as [`BTREE_V2_HUGE_OBJECT`].
pub const BTREE_V2_ATTRIBUTE_CREATION_ORDER: u8 = 9;

/// The type of a version 2 B-tree that indexes unfiltered dataset chunks, from the same table as
/// [`BTREE_V2_HUGE_OBJECT`].
pub const BTREE_V2_CHUNK: u8 = 10;

/// The type of a version 2 B-tree that indexes filtered dataset chunks, from the same table as
/// [`BTREE_V2_HUGE_OBJECT`].
pub const BTREE_V2_FILTERED_CHUNK: u8 = 11;

/// The width of one scaled chunk coordinate in a type 10 or type 11 record.
const CHUNK_COORDINATE_LEN: usize = 8;

/// The width of the filter mask in a type 11 record.
const FILTER_MASK_LEN: usize = 4;

/// The width of the creation order that opens a type 6 record, from the same section as
/// [`BTREE_V2_HUGE_OBJECT`].
const LINK_CREATION_ORDER_LEN: usize = 8;

/// The width of the message flags in a type 8 or type 9 record.
const MESSAGE_FLAGS_LEN: u16 = 1;

/// The width of the creation order in a type 8 or type 9 record.
const ATTRIBUTE_CREATION_ORDER_LEN: u16 = 4;

/// The width of the name hash that opens a type 5 record and closes a type 8 record.
const NAME_HASH_LEN: u16 = 4;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::btree_v2;
    use test_util::bytes as test_bytes;
    use test_util::widths::Widths;

    use super::*;

    #[test]
    fn header_size_follows_file_widths() {
        assert_eq!(
            btree_v2_header_size(OffsetWidth::Four, LengthWidth::Four),
            30
        );
        assert_eq!(
            btree_v2_header_size(OffsetWidth::Four, LengthWidth::Eight),
            34
        );
        assert_eq!(
            btree_v2_header_size(OffsetWidth::Eight, LengthWidth::Eight),
            38
        );
    }

    #[test]
    fn a_header_parses_to_its_fields() {
        let data = btree_v2::Header::new(5, 11, 0x1000, 3).build(WIDTHS);
        let hdr = BTreeV2Header::parse(&data, 0, 8, 8).unwrap();
        assert_eq!(hdr.tree_type, 5);
        assert_eq!(hdr.node_size, 512);
        assert_eq!(hdr.record_size, 11);
        assert_eq!(hdr.depth, 0);
        assert_eq!(hdr.root_node_address, StoredAddress::new(0x1000));
        assert_eq!(hdr.num_records_in_root, 3);
        assert_eq!(hdr.total_records, 3);
    }

    #[test]
    fn type_10_chunk_records_decode_scaled_offsets() {
        let context = BTreeV2ChunkRecordContext::new(
            BTREE_V2_CHUNK,
            OffsetWidth::Four,
            3,
            NonZeroU64::new(80).unwrap(),
            LayoutVersion::Four,
        )
        .unwrap();
        assert_eq!(context.record_size(), 4 + 3 * 8);

        let mut data = Vec::new();
        test_bytes::push_address(&mut data, Some(0x1020_3040), 4);
        test_bytes::push_uint(&mut data, 2, 8);
        test_bytes::push_uint(&mut data, 7, 8);
        test_bytes::push_uint(&mut data, 11, 8);
        let record = context.decode(&BTreeV2Record { data }).unwrap();

        assert_eq!(record.address, StoredAddress::new(0x1020_3040));
        assert_eq!(record.stored_size, 80);
        assert_eq!(record.filter_mask, 0);
        assert_eq!(record.scaled_offsets, vec![2, 7, 11]);
    }

    #[test]
    fn type_10_chunk_records_support_8_byte_offsets_and_rank_1() {
        let context = BTreeV2ChunkRecordContext::new(
            BTREE_V2_CHUNK,
            OffsetWidth::Eight,
            1,
            NonZeroU64::new(32).unwrap(),
            LayoutVersion::Four,
        )
        .unwrap();
        assert_eq!(context.record_size(), 16);

        let mut data = Vec::new();
        test_bytes::push_address(&mut data, Some(0x1020_3040_5060_7080), 8);
        test_bytes::push_uint(&mut data, 9, 8);
        let record = context.decode(&BTreeV2Record { data }).unwrap();

        assert_eq!(record.address, StoredAddress::new(0x1020_3040_5060_7080));
        assert_eq!(record.scaled_offsets, vec![9]);
    }

    #[test]
    fn type_11_chunk_records_decode_size_mask_and_scaled_offsets() {
        // An 80-byte logical chunk uses a 2-byte stored-size field in a version 4 layout.
        let context = BTreeV2ChunkRecordContext::new(
            BTREE_V2_FILTERED_CHUNK,
            OffsetWidth::Eight,
            2,
            NonZeroU64::new(80).unwrap(),
            LayoutVersion::Four,
        )
        .unwrap();
        assert_eq!(context.record_size(), 8 + 2 + 4 + 2 * 8);

        let mut data = Vec::new();
        test_bytes::push_address(&mut data, Some(0x1234), 8);
        test_bytes::push_uint(&mut data, 57, 2);
        data.extend_from_slice(&0x0000_0005u32.to_le_bytes());
        test_bytes::push_uint(&mut data, 3, 8);
        test_bytes::push_uint(&mut data, 4, 8);
        let record = context.decode(&BTreeV2Record { data }).unwrap();

        assert_eq!(record.address, StoredAddress::new(0x1234));
        assert_eq!(record.stored_size, 57);
        assert_eq!(record.filter_mask, 5);
        assert_eq!(record.scaled_offsets, vec![3, 4]);
    }

    #[test]
    fn chunk_record_context_rejects_declared_record_size_mismatches() {
        let context = BTreeV2ChunkRecordContext::new(
            BTREE_V2_CHUNK,
            OffsetWidth::Eight,
            2,
            NonZeroU64::new(64).unwrap(),
            LayoutVersion::Four,
        )
        .unwrap();
        let expected = context.record_size();

        assert_eq!(
            context
                .require_record_size(u16::try_from(expected - 1).unwrap())
                .unwrap_err(),
            FormatError::DataSizeMismatch {
                expected,
                actual: expected - 1,
            }
        );
        assert_eq!(
            context
                .require_record_size(u16::try_from(expected + 1).unwrap())
                .unwrap_err(),
            FormatError::DataSizeMismatch {
                expected,
                actual: expected + 1,
            }
        );
    }

    #[test]
    fn filtered_chunk_records_reject_zero_size_and_undefined_address() {
        let context = BTreeV2ChunkRecordContext::new(
            BTREE_V2_FILTERED_CHUNK,
            OffsetWidth::Four,
            1,
            NonZeroU64::new(32).unwrap(),
            LayoutVersion::Four,
        )
        .unwrap();

        let mut zero_size = Vec::new();
        test_bytes::push_address(&mut zero_size, Some(0x100), 4);
        test_bytes::push_uint(&mut zero_size, 0, 2);
        zero_size.extend_from_slice(&0u32.to_le_bytes());
        test_bytes::push_uint(&mut zero_size, 0, 8);
        assert_eq!(
            context
                .decode(&BTreeV2Record { data: zero_size })
                .unwrap_err(),
            FormatError::InvalidChunkGeometry(
                "filtered version 2 B-tree chunk record has zero stored size"
            )
        );

        let mut undefined = Vec::new();
        test_bytes::push_undefined_address(&mut undefined, 4);
        test_bytes::push_uint(&mut undefined, 1, 2);
        undefined.extend_from_slice(&0u32.to_le_bytes());
        test_bytes::push_uint(&mut undefined, 0, 8);
        assert_eq!(
            context
                .decode(&BTreeV2Record { data: undefined })
                .unwrap_err(),
            FormatError::ChunkedReadError(
                "version 2 B-tree chunk record has an undefined chunk address".into()
            )
        );
    }

    #[test]
    fn chunk_record_context_rejects_non_chunk_tree_types() {
        assert_eq!(
            BTreeV2ChunkRecordContext::new(
                BTREE_V2_ATTRIBUTE_NAME,
                OffsetWidth::Eight,
                1,
                NonZeroU64::new(32).unwrap(),
                LayoutVersion::Four,
            )
            .unwrap_err(),
            FormatError::InvalidBTreeNodeType(BTREE_V2_ATTRIBUTE_NAME)
        );
    }

    #[test]
    fn node_info_matches_hdf5_widths() {
        // The name index of a dense group: 512-byte nodes, 11-byte records, and 8-byte offsets,
        // with the widths computed by hand from `H5B2__hdr_init`. A child pointer of a depth-3
        // root is 11 bytes: the address (8), `max_nrec_size` (1), and `cum_max_nrec_size[2]` (2),
        // since `cum_max_nrec[2]` is 26449.
        let ni = BTreeV2NodeInfo::compute(512, 11, 8, 3);
        assert_eq!(ni.max_nrec_size, 1);
        assert_eq!(ni.cum_max_nrec_size, vec![0, 2, 2, 3]);
        assert_eq!(ni.child_ptr_size(1, 8), 9); // depth-1 children are leaves
        assert_eq!(ni.child_ptr_size(2, 8), 11);
        assert_eq!(ni.child_ptr_size(3, 8), 11);

        // A leaf of (4096 - 10) / 8 = 510 records needs a 2-byte record count.
        let big = BTreeV2NodeInfo::compute(4096, 8, 8, 1);
        assert_eq!(big.max_nrec_size, 2);

        let (node_info, depth) = BTreeV2NodeInfo::for_record_count(512, 17, 8, 30).unwrap();
        assert_eq!(depth, 1);
        assert_eq!(node_info.max_nrec(0), 29);
    }

    #[test]
    fn a_header_without_bthd_is_an_invalid_signature() {
        let mut data = btree_v2::Header::new(5, 11, 0, 0).build(WIDTHS);
        data[0] = b'X';
        let err = BTreeV2Header::parse(&data, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeV2Signature);
    }

    #[test]
    fn a_header_of_version_1_is_an_invalid_version() {
        let mut data = btree_v2::Header::new(5, 11, 0, 0).build(WIDTHS);
        data[4] = 1;
        let err = BTreeV2Header::parse(&data, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeV2Version(1));
    }

    #[test]
    fn nodes_with_4_byte_addresses_parse_to_their_fields() {
        let widths = Widths::new(4, 8);
        let header = btree_v2::Header::new(5, 11, 0x200, 1)
            .depth(1)
            .total_records(3)
            .build(widths);
        let leaf = btree_v2::leaf(5, &[vec![1; 11], vec![2; 11]]);
        let child = |address| btree_v2::Child {
            address,
            records: 1,
            records_width: 1,
            subtree: None,
        };
        let internal = btree_v2::internal(5, &[vec![3; 11]], &[child(0x300), child(0x400)], widths);
        let expected = BTreeV2Header {
            tree_type: 5,
            node_size: 512,
            record_size: 11,
            depth: 1,
            root_node_address: StoredAddress::new(0x200),
            num_records_in_root: 1,
            total_records: 3,
        };

        assert_eq!(BTreeV2Header::parse(&header, 0, 4, 8), Ok(expected.clone()));
        assert_eq!(
            BTreeV2Header::parse_from_source(header.as_slice(), 0, 4, 8),
            Ok(expected)
        );
        assert_eq!(
            parse_btree_v2_leaf_records(&leaf, 0, 2, 11),
            Ok(vec![
                BTreeV2Record { data: vec![1; 11] },
                BTreeV2Record { data: vec![2; 11] },
            ])
        );
        assert_eq!(
            parse_btree_v2_internal_child_pointers(
                &internal,
                1,
                NonZeroU16::MIN,
                11,
                4,
                &BTreeV2NodeInfo::compute(512, 11, 4, 1),
            ),
            Ok(vec![
                (StoredAddress::new(0x300), 1),
                (StoredAddress::new(0x400), 1),
            ])
        );
    }

    const WIDTHS: Widths = Widths::EIGHT;

    #[rstest]
    fn an_encoded_huge_object_record_decodes_to_its_fields(
        #[values(OffsetWidth::Four, OffsetWidth::Eight)] offset_width: OffsetWidth,
        #[values(LengthWidth::Four, LengthWidth::Eight)] length_width: LengthWidth,
    ) {
        let record = HugeObjectRecord {
            address: StoredAddress::new(0x1234),
            length: 700,
            id: 3,
        };
        let mut data = Vec::new();
        record.encode(&mut data, offset_width, length_width);

        assert_eq!(
            data.len(),
            usize::from(HugeObjectRecord::size(offset_width, length_width))
        );
        let [os, ls] = [offset_width.get(), length_width.get()].map(usize::from);
        let mut expected = 0x1234u64.to_le_bytes()[..os].to_vec();
        expected.extend_from_slice(&700u64.to_le_bytes()[..ls]);
        expected.extend_from_slice(&3u64.to_le_bytes()[..ls]);
        assert_eq!(data, expected);
        assert_eq!(
            BTreeV2Record { data }.huge_object(offset_width, length_width),
            Ok(record)
        );
    }

    #[rstest]
    fn a_direct_huge_object_record_decodes_to_its_fields(
        #[values(OffsetWidth::Four, OffsetWidth::Eight)] offset_width: OffsetWidth,
        #[values(LengthWidth::Four, LengthWidth::Eight)] length_width: LengthWidth,
    ) {
        let [os, ls] = [offset_width.get(), length_width.get()].map(usize::from);
        let mut data = 0x1234u64.to_le_bytes()[..os].to_vec();
        data.extend_from_slice(&700u64.to_le_bytes()[..ls]);

        assert_eq!(
            data.len(),
            usize::from(HugeObjectDirectRecord::size(offset_width, length_width))
        );
        assert_eq!(
            BTreeV2Record { data }.huge_object_direct(offset_width, length_width),
            Ok(HugeObjectDirectRecord {
                address: StoredAddress::new(0x1234),
                length: 700,
            })
        );
    }

    #[test]
    fn a_huge_object_record_shorter_than_its_fields_is_unexpected_eof() {
        let record = BTreeV2Record { data: vec![0; 20] };
        assert_eq!(
            record.huge_object(OffsetWidth::Eight, LengthWidth::Eight),
            Err(FormatError::UnexpectedEof {
                expected: 24,
                available: 20,
            })
        );
    }

    #[test]
    fn an_attribute_name_record_holds_its_heap_id_flags_creation_order_and_hash() {
        let mut data = Vec::new();
        AttributeRecord {
            heap_id: &[1, 2, 3, 4, 5, 6, 7, 8],
            flags: MessageFlags::CONSTANT,
            creation_order: 0x0A0B,
        }
        .encode_name_record(&mut data, 0x1122_3344);

        assert_eq!(
            data,
            [
                1, 2, 3, 4, 5, 6, 7, 8, 0x01, 0x0B, 0x0A, 0, 0, 0x44, 0x33, 0x22, 0x11
            ]
        );
        assert_eq!(
            data.len(),
            usize::from(AttributeRecord::name_record_size(8))
        );
        let record = BTreeV2Record { data };
        assert_eq!(
            record.attribute_heap_id(8),
            Some(&[1, 2, 3, 4, 5, 6, 7, 8][..])
        );
        assert_eq!(record.attribute_creation_order(8), Some(0x0A0B));
    }

    #[test]
    fn an_attribute_creation_order_record_omits_the_hash() {
        let mut data = Vec::new();
        AttributeRecord {
            heap_id: &[9; 8],
            flags: MessageFlags::NONE,
            creation_order: 7,
        }
        .encode_creation_order_record(&mut data);

        assert_eq!(data, [9, 9, 9, 9, 9, 9, 9, 9, 0, 7, 0, 0, 0]);
        assert_eq!(
            data.len(),
            usize::from(AttributeRecord::creation_order_record_size(8))
        );
    }

    #[rstest]
    #[case::without_a_creation_order(vec![1, 2, 3, 4, 5, 6, 7, 8, 0], Some(&[1, 2, 3, 4, 5, 6, 7, 8][..]), None)]
    #[case::shorter_than_a_heap_id(vec![1, 2, 3], None, None)]
    fn a_short_attribute_record_yields_the_fields_it_holds(
        #[case] data: Vec<u8>,
        #[case] heap_id: Option<&[u8]>,
        #[case] creation_order: Option<u32>,
    ) {
        let record = BTreeV2Record { data };
        assert_eq!(record.attribute_heap_id(8), heap_id);
        assert_eq!(record.attribute_creation_order(8), creation_order);
    }

    #[rstest]
    #[case::a_name_index(BTREE_V2_LINK_NAME, &[0xA, 0xB, 0xC, 0xD, 1, 2, 3, 4, 5, 6, 7], Some(&[1, 2, 3, 4, 5, 6, 7][..]))]
    #[case::a_creation_order_index(6, &[0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7], Some(&[1, 2, 3, 4, 5, 6, 7][..]))]
    #[case::a_record_shorter_than_its_heap_id(BTREE_V2_LINK_NAME, &[0xA, 0xB, 0xC, 0xD, 1, 2], None)]
    fn a_link_record_yields_the_heap_id_past_its_key(
        #[case] tree_type: u8,
        #[case] data: &[u8],
        #[case] heap_id: Option<&[u8]>,
    ) {
        let record = BTreeV2Record {
            data: data.to_vec(),
        };
        assert_eq!(record.link_heap_id(tree_type, 7), heap_id);
    }
}
