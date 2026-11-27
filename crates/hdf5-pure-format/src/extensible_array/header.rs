use hdf5_pure_core::__private::{FileRange, FileRegion};

use super::ExtensibleArrayIndexBlockRef;
use super::block::{CHECKSUM_LEN, Owner, VERSION};
use super::creation::{
    BlockOffsetWidth, ExtensibleArrayBitGeometry, ExtensibleArrayCreationParameters, MaxIndexBits,
    PageElementCount,
};
use super::geometry::{ExtensibleArrayGeometry, Layout};
use super::statistics::ExtensibleArrayStats;
use crate::address::{BaseAddress, BaseAddressExt, StoredAddress};
use crate::bytes;
use crate::checksum;
use crate::chunk_record::{ArrayClient, ElementEncoding, InvalidElementEncoding};
use crate::codec::Decode;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::FormatWidths;

/// The address of an Extensible Array header, from the chunk index address of a data layout
/// message.
///
/// A caller reads the bytes of [`parse_range`](Self::parse_range) and passes them to
/// [`parse`](Self::parse), which returns the header and any index-block reference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayHeaderLocator {
    widths: FormatWidths,
    base: BaseAddress,
    address: StoredAddress,
}

impl ExtensibleArrayHeaderLocator {
    /// Creates the locator of the header at the base-relative `address`.
    pub const fn new(widths: FormatWidths, base: BaseAddress, address: StoredAddress) -> Self {
        Self {
            widths,
            base,
            address,
        }
    }

    /// Parses the complete header from [`Self::parse_range`] within `region`.
    ///
    /// Uses the same framing and checksum policy as [`ExtensibleArrayHeader::parse_with_widths`].
    /// Resolves a defined index-block address against the base and derives the block's range
    /// from the checked creation parameters and element encoding.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the header address or end exceeds [`u64::MAX`],
    /// or [`FormatError::InsufficientRegion`] if `region` omits any part of the header range.
    /// Within that range, reports the header field and checksum errors documented by
    /// [`ExtensibleArrayHeader::parse_with_widths`]. After decoding a header with a defined
    /// index-block address, returns [`FormatError::OffsetOverflow`] if the resolved index-block
    /// address or end exceeds [`u64::MAX`].
    pub fn parse(&self, region: FileRegion<'_>) -> Result<ExtensibleArray, FormatError> {
        let range = self.parse_range()?;
        let bytes = region.subregion(range)?.bytes();
        let header = ExtensibleArrayHeader::decode(bytes, self.widths)?;
        let index_block_address = header.index_block_address();
        let index_block = if index_block_address.is_undefined(header.encoding.offsets()) {
            None
        } else {
            let owner = Owner::from_header(self.base, range.start(), &header);
            Some(ExtensibleArrayIndexBlockRef::new(
                owner,
                index_block_address,
            )?)
        };
        Ok(ExtensibleArray {
            range,
            header,
            index_block,
        })
    }

    /// Returns the range a caller reads for [`parse`](Self::parse), the whole header.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the base address plus `address`, or the end of
    /// the header, is past [`u64::MAX`].
    pub fn parse_range(&self) -> Result<FileRange, FormatError> {
        let Self {
            widths,
            base,
            address,
        } = *self;
        FileRange::new(base.resolve(address)?, header_len(widths))
    }
}

/// The parsed header of an Extensible Array and the reference to its index block, from
/// [`ExtensibleArrayHeaderLocator::parse`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArray {
    range: FileRange,
    header: ExtensibleArrayHeader,
    index_block: Option<ExtensibleArrayIndexBlockRef>,
}

impl ExtensibleArray {
    /// Returns the physical range of the header.
    pub const fn range(&self) -> FileRange {
        self.range
    }

    /// Returns the stored header fields.
    ///
    /// The [`ExtensibleArrayHeader::index_block_address`] is base-relative or undefined.
    pub const fn header(&self) -> &ExtensibleArrayHeader {
        &self.header
    }

    /// Returns the block geometry defined by the header's creation parameters.
    ///
    /// Includes the geometry of unallocated blocks.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if a derived block offset
    /// exceeds [`u64::MAX`].
    pub fn geometry(&self) -> Result<ExtensibleArrayGeometry, FormatError> {
        ExtensibleArrayGeometry::from_header(&self.header)
    }

    /// Returns whether the array indexes filtered chunks, the header's client ID 1.
    pub const fn is_filtered(&self) -> bool {
        matches!(
            self.header.encoding.client(),
            ArrayClient::FilteredChunks { .. }
        )
    }

    /// Returns the index-block reference, or [`None`] when the header stores the undefined address.
    pub const fn index_block(&self) -> Option<ExtensibleArrayIndexBlockRef> {
        self.index_block
    }
}

/// An Extensible Array header, signature `EAHD`, version 0.
///
/// The maximum element bit count is in `1..=64`, and the page exponent is at most that count
/// and at most 63. The page element count fits a [`u64`], and the block offset is 1 to 8 bytes
/// wide.
///
/// "The Extensible Array Index" of the [format specification, version 4.0][spec]
/// defines the header.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayHeader {
    encoding: ElementEncoding,
    element_size: u8,
    creation: ExtensibleArrayCreationParameters,
    stored_statistics: ExtensibleArrayStats,
    index_block_address: StoredAddress,
}

impl ExtensibleArrayHeader {
    /// Parses the Extensible Array header at `offset` in `file_data`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] for an unsupported `offset_size` or
    /// [`FormatError::InvalidLengthSize`] for an unsupported `length_size`. The supported
    /// widths are 2, 4, and 8 bytes. Checks the offset width before the length width and
    /// checks both before reading the header. See [`Self::parse_with_widths`] for errors
    /// from the header bytes.
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<Self, FormatError> {
        Self::parse_with_widths(
            file_data,
            offset,
            FormatWidths::from_sizes(offset_size, length_size)?,
        )
    }

    /// Parses the complete header at `offset` and ignores bytes after its checksum.
    ///
    /// Reads the signature and version before checking the checksum. With the `checksum`
    /// feature, verifies the four-byte Jenkins checksum over all preceding header bytes.
    /// Then checks the client ID, the element size for filtered chunks, and the creation parameters
    /// in that order.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the complete header is unavailable. This
    /// error reports the required end relative to the start of `file_data` and the input length.
    /// The parser caps the required end at [`usize::MAX`].
    ///
    /// Returns [`FormatError::InvalidExtensibleArraySignature`] for a signature other than
    /// `EAHD`, [`FormatError::InvalidExtensibleArrayVersion`] for a version other than 0,
    /// or [`FormatError::ChecksumMismatch`] if verification is enabled and the checksum differs.
    /// Returns [`FormatError::InvalidExtensibleArrayClient`] for a client ID other than 0 or 1,
    /// or [`FormatError::InvalidExtensibleArrayGeometry`] for an unsupported filtered element
    /// size or creation parameter.
    pub fn parse_with_widths(
        file_data: &[u8],
        offset: usize,
        widths: FormatWidths,
    ) -> Result<Self, FormatError> {
        let input = file_data
            .get(offset..)
            .ok_or_else(|| FormatError::UnexpectedEof {
                expected: offset.saturating_add(Self::serialized_size_with_widths(widths)),
                available: file_data.len(),
            })?;
        Self::decode(input, widths).map_err(|error| match error {
            FormatError::UnexpectedEof { expected, .. } => FormatError::UnexpectedEof {
                expected: offset.saturating_add(expected),
                available: file_data.len(),
            },
            error => error,
        })
    }

    /// Returns the length in bytes of a header with `offset_size`-byte addresses and
    /// `length_size`-byte lengths.
    pub fn serialized_size(offset_size: u8, length_size: u8) -> usize {
        4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 6 * length_size as usize + offset_size as usize + 4
    }

    /// Returns the header length for validated format widths.
    pub fn serialized_size_with_widths(widths: FormatWidths) -> usize {
        HEADER_PREFIX_LEN
            + ExtensibleArrayStats::fields_len(widths.lengths)
            + usize::from(widths.offsets.get())
            + CHECKSUM_LEN
    }

    /// Returns 0 for unfiltered chunks or 1 for filtered chunks.
    pub const fn client_id(&self) -> u8 {
        self.encoding.client().id()
    }

    /// Returns the stored element size in bytes.
    ///
    /// For unfiltered chunks, the reader uses the file's offset width to read the entries.
    pub const fn element_size(&self) -> u8 {
        self.element_size
    }

    /// Returns the checked creation parameters from the header.
    pub const fn creation_parameters(&self) -> ExtensibleArrayCreationParameters {
        self.creation
    }

    /// Returns the parsed maximum element bit count and data block page exponent.
    pub fn bit_geometry(&self) -> ExtensibleArrayBitGeometry {
        self.creation.bit_geometry()
    }

    /// Returns the "Max Nelmts Bits" field, in `1..=64`.
    pub fn max_index_bits(&self) -> MaxIndexBits {
        self.bit_geometry().max_index_bits()
    }

    /// Returns the number of elements in a data block page, a nonzero power of two.
    pub const fn page_nelmts(&self) -> PageElementCount {
        self.creation.bit_geometry().page_nelmts()
    }

    /// Returns the width in bytes of a block offset, in `1..=8`.
    pub fn block_offset_width(&self) -> BlockOffsetWidth {
        self.bit_geometry().block_offset_width()
    }

    /// Returns the six statistics as stored in the header.
    ///
    /// The parser reads these values from the header bytes alone. Use
    /// [`ExtensibleArrayStats::encode_fields`] to reproduce the fields at the file's length width.
    ///
    /// # Examples
    ///
    /// Reads and re-encodes the statistics of a header with two-byte lengths and four-byte addresses.
    ///
    /// ```
    /// use hdf5_pure_format::__private::{ExtensibleArrayHeader, FormatWidths};
    ///
    /// let bytes = [
    ///     b'E', b'A', b'H', b'D', 0, 0, 4, 32, 4, 16, 4, 10,
    ///     1, 0, 0x80, 1, 3, 0, 0, 9, 5, 0, 6, 0,
    ///     0, 0x10, 0, 0, 0x73, 0x63, 0xec, 0x2b,
    /// ];
    /// let widths = FormatWidths::from_sizes(4, 2)?;
    /// let header = ExtensibleArrayHeader::parse_with_widths(&bytes, 0, widths)?;
    /// let stats = header.stored_statistics();
    /// assert_eq!(stats.max_idx_set, 5);
    /// assert_eq!(stats.nelmts, 6);
    ///
    /// let mut fields = [0; 12];
    /// assert_eq!(stats.encode_fields(&mut fields, widths.lengths)?, fields.len());
    /// assert_eq!(fields, bytes[12..24]);
    /// # Ok::<(), hdf5_pure_format::FormatError>(())
    /// ```
    pub const fn stored_statistics(&self) -> &ExtensibleArrayStats {
        &self.stored_statistics
    }

    /// Returns the stored index-block address, including the file-width undefined address.
    pub const fn index_block_address(&self) -> StoredAddress {
        self.index_block_address
    }

    pub(super) fn layout(&self) -> Layout {
        self.creation.layout(self.encoding)
    }
}

impl<'input> Decode<'input> for ExtensibleArrayHeader {
    type Context = FormatWidths;
    type Error = FormatError;

    fn decode(input: &'input [u8], widths: FormatWidths) -> Result<Self, Self::Error> {
        let header =
            bytes::ByteCursor::new(input).take(Self::serialized_size_with_widths(widths))?;
        let mut cursor = bytes::ByteCursor::new(header);
        if *cursor.take_array()? != HEADER_SIGNATURE {
            return Err(FormatError::InvalidExtensibleArraySignature);
        }
        let [
            version,
            client_id,
            element_size,
            max_index_bits,
            inline_elements,
            min_elements,
            min_data_blocks,
            page_bits,
        ] = cursor.array::<HEADER_BYTE_FIELDS>()?;
        if version != VERSION {
            return Err(FormatError::InvalidExtensibleArrayVersion(version));
        }
        checksum::verify_trailing(header)?;

        let encoding =
            ElementEncoding::new(widths.offsets, client_id, element_size).map_err(|error| {
                match error {
                    InvalidElementEncoding::Client(id) => {
                        FormatError::InvalidExtensibleArrayClient(id)
                    }
                    InvalidElementEncoding::EntrySize => {
                        FormatError::InvalidExtensibleArrayGeometry
                    }
                }
            })?;
        let creation = ExtensibleArrayCreationParameters::from_fields([
            max_index_bits,
            inline_elements,
            min_data_blocks,
            min_elements,
            page_bits,
        ])
        .ok_or(FormatError::InvalidExtensibleArrayGeometry)?;
        let stored_statistics = ExtensibleArrayStats::decode_fields(&mut cursor, widths.lengths)?;
        let index_block_address = cursor.address(widths.offsets)?;
        Ok(Self {
            encoding,
            element_size,
            creation,
            stored_statistics,
            index_block_address,
        })
    }
}

pub(super) fn header_len(widths: FormatWidths) -> u64 {
    ExtensibleArrayHeader::serialized_size_with_widths(widths).to_u64()
}

pub(super) const HEADER_SIGNATURE: [u8; 4] = *b"EAHD";

// The version, client ID, element size, and the five creation parameters of a header.
pub(super) const HEADER_BYTE_FIELDS: usize = 8;
pub(super) const HEADER_PREFIX_LEN: usize = HEADER_SIGNATURE.len() + HEADER_BYTE_FIELDS;

#[cfg(test)]
#[path = "tests/header.rs"]
mod tests;
