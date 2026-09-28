//! The version 2 B-tree parsers: the header, the leaf and internal nodes, and the table of node
//! capacities, which gives the widths of the record counts in an internal node's child pointers.

use core::num::NonZeroU16;

use alloc::vec;
use alloc::vec::Vec;

#[cfg(feature = "checksum")]
use byteorder::ByteOrder;
#[cfg(feature = "checksum")]
use byteorder::LittleEndian;

use crate::address::StoredAddress;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::metadata_source::MetadataSource;

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
/// [`BTreeV2Plan`](crate::BTreeV2Plan) lays out a tree from the same table, so a written tree has
/// the widths a reader computes for it.
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
/// [`FormatError::UnexpectedEof`] if the records or the child pointers run past the end of `node`,
/// and [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8.
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

    Ok(children)
}

#[cfg(test)]
mod tests {
    use test_util::btree_v2;
    use test_util::widths::Widths;

    use super::*;

    #[test]
    fn parse_header() {
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
    }

    #[test]
    fn invalid_signature() {
        let mut data = btree_v2::Header::new(5, 11, 0, 0).build(WIDTHS);
        data[0] = b'X';
        let err = BTreeV2Header::parse(&data, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeV2Signature);
    }

    #[test]
    fn invalid_version() {
        let mut data = btree_v2::Header::new(5, 11, 0, 0).build(WIDTHS);
        data[4] = 1; // bad version
        let err = BTreeV2Header::parse(&data, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeV2Version(1));
    }

    const WIDTHS: Widths = Widths::EIGHT;
}
