use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::metadata_source::MetadataSource;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

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
    HEADER_FIXED_LEN + offset_width.get() as usize + length_width.get() as usize
}

impl BTreeV2Header {
    /// Parses a version 2 B-tree header using the width bytes from the superblock.
    ///
    /// Pass the "Size of Offsets" byte as `offset_size` and the "Size of Lengths" byte as
    /// `length_size`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] for an
    /// unsupported width, or an error from [`parse_with_widths`](Self::parse_with_widths).
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<BTreeV2Header, FormatError> {
        Self::parse_with_widths(
            file_data,
            offset,
            FormatWidths::from_sizes(offset_size, length_size)?,
        )
    }

    /// Checks that the version 2 B-tree header fits in the slice before parsing its fields.
    ///
    /// Requires the four-byte checksum field. With the `checksum` feature, verifies the checksum
    /// after checking the signature and version.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the header extends past `file_data`. Reports
    /// the exclusive end of the header, capped at [`usize::MAX`], and the length of `file_data`.
    ///
    /// Returns [`FormatError::InvalidBTreeV2Signature`] for a signature other than `BTHD`,
    /// or [`FormatError::InvalidBTreeV2Version`] for a header version other than 0. With the
    /// `checksum` feature, returns [`FormatError::ChecksumMismatch`] if verification fails.
    pub fn parse_with_widths(
        file_data: &[u8],
        offset: usize,
        widths: FormatWidths,
    ) -> Result<BTreeV2Header, FormatError> {
        let size = btree_v2_header_size(widths.offsets, widths.lengths);
        let header = offset
            .checked_add(size)
            .and_then(|end| file_data.get(offset..end))
            .ok_or_else(|| FormatError::UnexpectedEof {
                expected: offset.saturating_add(size),
                available: file_data.len(),
            })?;
        Self::parse_header(header, widths)
    }

    /// Reads a version 2 B-tree header using the width bytes from the superblock.
    ///
    /// Takes the same width bytes as [`parse`](Self::parse).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] for an
    /// unsupported width, or an error from
    /// [`parse_from_source_with_widths`](Self::parse_from_source_with_widths).
    pub fn parse_from_source(
        source: &(impl MetadataSource + ?Sized),
        address: u64,
        offset_size: u8,
        length_size: u8,
    ) -> Result<BTreeV2Header, FormatError> {
        Self::parse_from_source_with_widths(
            source,
            address,
            FormatWidths::from_sizes(offset_size, length_size)?,
        )
    }

    /// Reads exactly the bytes of a version 2 B-tree header in one metadata request.
    ///
    /// Parses the header as [`parse_with_widths`](Self::parse_with_widths) does.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hdf5_pure_format::__private::{BTreeV2Header, FormatWidths};
    /// # use hdf5_pure_format::FormatError;
    /// let file = b"\0\0BTHD";
    /// let widths = FormatWidths::from_sizes(4, 8)?;
    /// assert_eq!(
    ///     BTreeV2Header::parse_from_source_with_widths(file.as_slice(), 2, widths),
    ///     Err(FormatError::UnexpectedSourceEof {
    ///         expected: 36,
    ///         available: 6,
    ///     }),
    /// );
    /// # Ok::<(), FormatError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the end of the header exceeds [`u64::MAX`], or
    /// [`FormatError::UnexpectedSourceEof`] if the source is too short or returns fewer bytes than
    /// requested. The truncation error reports the exclusive end of the header and the available
    /// extent in the source's coordinates.
    ///
    /// Returns errors from the source if a read fails, or the signature, version, and checksum
    /// errors of [`parse_with_widths`](Self::parse_with_widths) if parsing fails.
    pub fn parse_from_source_with_widths(
        source: &(impl MetadataSource + ?Sized),
        address: u64,
        widths: FormatWidths,
    ) -> Result<BTreeV2Header, FormatError> {
        let size = btree_v2_header_size(widths.offsets, widths.lengths);
        let end = address
            .checked_add(size.to_u64())
            .ok_or(FormatError::OffsetOverflow {
                offset: address,
                length: size.to_u64(),
            })?;
        let available = source.len();
        if end > available {
            return Err(FormatError::UnexpectedSourceEof {
                expected: end,
                available,
            });
        }
        let acquired = source.read_metadata(address, size)?;
        let header = acquired
            .get(..size)
            .ok_or_else(|| FormatError::UnexpectedSourceEof {
                expected: end,
                available: address + acquired.len().to_u64(),
            })?;
        Self::parse_header(header, widths)
    }

    /// Parses a header from an exact-length slice that includes the checksum field.
    ///
    /// # Errors
    ///
    /// Returns the signature, version, and checksum errors of
    /// [`parse_with_widths`](Self::parse_with_widths).
    fn parse_header(header: &[u8], widths: FormatWidths) -> Result<Self, FormatError> {
        let mut cursor = bytes::ByteCursor::new(header);
        if cursor.take_array()? != HEADER_SIGNATURE {
            return Err(FormatError::InvalidBTreeV2Signature);
        }
        let version = cursor.u8()?;
        if version != HEADER_VERSION {
            return Err(FormatError::InvalidBTreeV2Version(version));
        }

        let tree_type = cursor.u8()?;
        let node_size = cursor.u32()?;
        let record_size = cursor.u16()?;
        let depth = cursor.u16()?;
        let _split_percent = cursor.u8()?;
        let _merge_percent = cursor.u8()?;
        let root_node_address = cursor.address(widths.offsets)?;
        let num_records_in_root = cursor.u16()?;
        let total_records = cursor.length(widths.lengths)?;
        checksum::verify_trailing(header)?;

        Ok(Self {
            tree_type,
            node_size,
            record_size,
            depth,
            root_node_address,
            num_records_in_root,
            total_records,
        })
    }
}

// The fixed-width fields of the version 0 header, including its checksum, are defined in
// "Version 2 B-trees" of the format specification, version 4.0.
const HEADER_FIXED_LEN: usize = 22;
const HEADER_SIGNATURE: &[u8; 4] = b"BTHD";
const HEADER_VERSION: u8 = 0;

#[cfg(test)]
#[path = "tests/header.rs"]
mod tests;
