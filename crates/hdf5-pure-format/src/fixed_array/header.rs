use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;

use super::FixedArrayPlacement;
use super::FixedArrayPlan;
use super::data_block::BlockStorage;
use super::data_block::FixedArrayDataBlockRef;
use super::framing::CHECKSUM_LEN;
use super::framing::SIGNATURE_LEN;
use super::framing::VERSION;
use super::geometry;
use super::geometry::Geometry;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum;
use crate::chunk_record::ArrayClient;
use crate::error::FormatError;
use crate::width::FormatWidths;

/// The address of a Fixed Array header, from the chunk index address of a data layout message.
///
/// A caller reads the bytes of [`parse_range`](Self::parse_range) and passes them to
/// [`parse`](Self::parse). Use [`FixedArrayHeader::data_block`] to locate an allocated block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedArrayHeaderLocator {
    widths: FormatWidths,
    base: BaseAddress,
    address: StoredAddress,
}

impl FixedArrayHeaderLocator {
    /// Creates the locator of the header at the base-relative `address`.
    pub const fn new(widths: FormatWidths, base: BaseAddress, address: StoredAddress) -> Self {
        Self {
            widths,
            base,
            address,
        }
    }

    /// Parses the header in `region`.
    ///
    /// An unfiltered entry size must equal the file's offset width. A filtered entry includes
    /// the address, a size field of up to eight bytes, and a four-byte filter mask.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`parse_range`](Self::parse_range), [`FormatError::InvalidFixedArraySignature`] if the
    /// bytes do not start with `FAHD`, [`FormatError::InvalidFixedArrayVersion`] if the version is
    /// not 0, [`FormatError::InvalidFixedArrayClient`] if the client ID is 2 or more, and
    /// [`FormatError::InvalidFixedArrayGeometry`] if the page bits, the entry size, or the entry
    /// count is out of range. Returns [`FormatError::OffsetOverflow`] if the base address plus
    /// `address` or plus the data block address, or the end of the header or of the data block,
    /// is past [`u64::MAX`], and, under the `checksum` feature, [`FormatError::ChecksumMismatch`]
    /// if the checksum does not match the header's bytes.
    pub fn parse(&self, region: FileRegion<'_>) -> Result<FixedArrayHeader, FormatError> {
        let Self {
            widths,
            base,
            address: _,
        } = *self;
        let range = self.parse_range()?;
        let bytes = region.subregion(range)?.bytes();
        let mut cursor = bytes::ByteCursor::new(bytes);
        if cursor.array::<{ HEADER_SIGNATURE.len() }>()? != HEADER_SIGNATURE {
            return Err(FormatError::InvalidFixedArraySignature);
        }
        let version = cursor.u8()?;
        if version != VERSION {
            return Err(FormatError::InvalidFixedArrayVersion(version));
        }
        let client_id = cursor.u8()?;
        let entry_size = cursor.u8()?;
        let page_bits = cursor.u8()?;
        let entries = cursor.length(widths.lengths)?;
        let data_block = cursor.decoded_address(widths.offsets)?;
        checksum::verify_trailing(bytes)?;

        let encoding = geometry::element_encoding(widths.offsets, client_id, entry_size)?;
        if entry_size != encoding.stride().get() {
            return Err(FormatError::InvalidFixedArrayGeometry);
        }
        let geometry = Geometry::new(encoding, entries, page_bits)?;
        let address = match data_block {
            DecodedStoredAddress::Defined(address) => Some(base.resolve(address)?),
            DecodedStoredAddress::Undefined => None,
        };
        Ok(FixedArrayHeader {
            range,
            widths,
            base,
            storage: BlockStorage::new(geometry, address)?,
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

/// A version 0 Fixed Array header, from [`FixedArrayHeaderLocator::parse`].
///
/// Retains the recorded widths, base, entry encoding, and page geometry, including when the
/// data-block address is undefined. A writer can prepare empty storage with
/// [`initialization_at`](Self::initialization_at).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedArrayHeader {
    range: FileRange,
    widths: FormatWidths,
    base: BaseAddress,
    storage: BlockStorage,
}

impl FixedArrayHeader {
    /// Returns whether the array's client format stores filtered chunk sizes and masks.
    pub const fn is_filtered(&self) -> bool {
        matches!(
            self.storage.geometry().encoding().client(),
            ArrayClient::FilteredChunks { .. }
        )
    }

    /// Returns the physical range of the header.
    pub const fn range(&self) -> FileRange {
        self.range
    }

    /// Returns the reference to the data block, or `None` if the header stores the undefined
    /// address, in an array without chunks.
    pub const fn data_block(&self) -> Option<FixedArrayDataBlockRef> {
        self.storage.at(self.base, self.range.start())
    }

    /// Plans an empty data block at `data_block` for this header's recorded geometry.
    ///
    /// Retains the header's stored address, widths, client, entry size, entry count, and page bits.
    /// The caller establishes ownership of the header and data-block allocations before writing.
    /// An existing data-block address does not affect the plan.
    ///
    /// # Errors
    ///
    /// Returns the placement errors of [`FixedArrayPlan::place`],
    /// [`FormatError::InvalidFixedArrayGeometry`] if the block lengths or their sum overflow, or
    /// [`FormatError::AddressBelowBase`] if the header precedes its recorded base.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{BaseAddress, PhysicalAddress, StoredAddress};
    /// use hdf5_pure_core::__private::{BaseAddressExt, FileRange, FileRegion, StoredAddressExt};
    /// use hdf5_pure_format::{__private as format, FormatError};
    ///
    /// let widths = format::FormatWidths::from_sizes(2, 4)?;
    /// let placement = format::FixedArrayPlan::new(3, 16, widths, false)?
    ///     .with_page_bits(1)?
    ///     .place(StoredAddress::new(64), StoredAddress::new(256))?;
    /// let bytes = placement.serialize_header()?;
    /// let range = FileRange::new(PhysicalAddress::new(64), bytes.len() as u64)?;
    /// let locator = format::FixedArrayHeaderLocator::new(
    ///     widths, BaseAddress::ZERO, StoredAddress::new(64),
    /// );
    /// let header = locator.parse(FileRegion::new(range, &bytes)?)?;
    /// assert_eq!(header.entries(), 3);
    /// assert_eq!(header.page_bits(), 1);
    /// let plan = header.initialization_plan()?;
    /// assert_eq!(plan.block_lengths().collect::<Vec<_>>(), [18, 27]);
    /// let replacement = header.initialization_at(StoredAddress::new(512))?;
    /// let bytes = replacement.serialize_header()?;
    /// let updated = locator.parse(FileRegion::new(range, &bytes)?)?;
    /// assert_eq!(updated.entries(), header.entries());
    /// assert_eq!(updated.page_bits(), header.page_bits());
    /// assert_eq!(updated.data_block().unwrap().range().start(), PhysicalAddress::new(512));
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn initialization_at(
        &self,
        data_block: StoredAddress,
    ) -> Result<FixedArrayPlacement, FormatError> {
        self.initialization_plan()?
            .place(self.base.relative(self.range.start())?, data_block)
    }

    /// Plans empty storage with the recorded widths, entry encoding, and page bits.
    ///
    /// Does not choose addresses or establish allocation ownership. See
    /// [`Self::initialization_at`] for an example of reusing parsed geometry.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFixedArrayGeometry`] if a block length or their sum overflows.
    pub fn initialization_plan(&self) -> Result<FixedArrayPlan, FormatError> {
        FixedArrayPlan::from_geometry(self.widths.lengths, self.storage.geometry())
    }

    /// Returns the array's maximum entry count, including unallocated slots.
    pub const fn entries(&self) -> u64 {
        self.storage.geometry().entries()
    }

    /// Returns the base 2 logarithm of the maximum entries per page.
    pub const fn page_bits(&self) -> u8 {
        self.storage.geometry().page_bits()
    }
}

/// Returns the length in bytes of a Fixed Array header in a file with the address and length
/// widths `widths`.
pub(super) fn header_len(FormatWidths { offsets, lengths }: FormatWidths) -> u64 {
    SIGNATURE_LEN
        + HEADER_BYTE_FIELDS
        + u64::from(lengths.get())
        + u64::from(offsets.get())
        + CHECKSUM_LEN
}

pub(super) const HEADER_SIGNATURE: [u8; 4] = *b"FAHD";
/// The version, client ID, entry size, and page bits bytes of a header.
const HEADER_BYTE_FIELDS: u64 = 4;
