use alloc::format;

use hdf5_pure_core::__private::{DecodedStoredAddress, FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;

use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::FormatWidths;

use super::address_space::AddressSpaceBits;
use super::section_list::FreeSpaceManagerSectionListRef;

/// A physical reference to a version 0 free-space manager header.
///
/// The caller resolves the stored header address before constructing this reference. Parsing
/// validates common framing. Client-specific header validation is a separate operation.
///
/// # Examples
///
/// The caller supplies header bytes, resolves the section-list address, and supplies the used
/// section-list bytes. The borrowed record bytes require client-specific validation.
///
/// ```
/// use hdf5_pure_core::{BaseAddress, StoredAddress};
/// use hdf5_pure_core::__private::{BaseAddressExt, DecodedStoredAddress, FileRegion,
///     StoredAddressExt};
/// use hdf5_pure_format::__private as format;
/// use hdf5_pure_format::__private::FormatWidths;
/// use hdf5_pure_format::__private::FreeSection;
/// use hdf5_pure_format::__private::{FreeSpaceManagerHeader, FreeSpaceManagerHeaderRef};
///
/// let widths = FormatWidths::from_sizes(8, 8)?;
/// let header_addr = StoredAddress::new(64);
/// # let list_addr = StoredAddress::new(146);
/// # let (header_bytes, list_bytes) = format::serialize_free_space_manager(
/// #     widths, &[FreeSection { addr: StoredAddress::new(256), size: 32 }],
/// #     header_addr, list_addr, format::SECTION_CLASS_SIMPLE,
/// # )?;
/// let base = BaseAddress::new(512);
/// let header_ref = FreeSpaceManagerHeaderRef::new(widths, base.resolve(header_addr)?)?;
/// let header_region = FileRegion::new(header_ref.parse_range(), &header_bytes)?;
/// let frame = header_ref.parse(header_region)?;
/// let header = FreeSpaceManagerHeader::from_frame(&frame)?;
/// let DecodedStoredAddress::Defined(list_addr) = frame.section_list_addr() else {
///     panic!("expected a defined section list address")
/// };
/// let list_ref = frame.section_list_ref(header_addr, base.resolve(list_addr)?)?;
/// let list_region = FileRegion::new(list_ref.parse_range(), &list_bytes)?;
/// let view = list_ref.parse(list_region)?;
/// assert_eq!(header.total_sections(), 1);
/// assert_eq!(view.records().len(), 18);
/// # Ok::<(), hdf5_pure_format::FormatError>(())
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FreeSpaceManagerHeaderRef {
    range: FileRange,
    widths: FormatWidths,
}

impl FreeSpaceManagerHeaderRef {
    /// Creates a header reference at the resolved physical address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the header range ends past [`u64::MAX`].
    pub fn new(widths: FormatWidths, address: PhysicalAddress) -> Result<Self, FormatError> {
        Ok(Self {
            range: FileRange::new(address, free_space_manager_header_len(widths))?,
            widths,
        })
    }

    /// Parses common header framing from the referenced bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`parse_range`](Self::parse_range). Other failures are the errors of
    /// [`FreeSpaceManagerHeaderFrame::parse`].
    pub fn parse(
        &self,
        region: FileRegion<'_>,
    ) -> Result<FreeSpaceManagerHeaderFrame, FormatError> {
        let Self { range, widths } = *self;
        FreeSpaceManagerHeaderFrame::parse(widths, region.subregion(range)?.bytes())
    }

    /// Parses common header framing and verifies the checksum regardless of the `checksum` feature.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ChecksumMismatch`] if the stored checksum differs from the
    /// computed one. Other failures are the errors of [`parse`](Self::parse).
    pub fn parse_checked(
        &self,
        region: FileRegion<'_>,
    ) -> Result<FreeSpaceManagerHeaderFrame, FormatError> {
        #[cfg(not(feature = "checksum"))]
        checksum::verify_trailing_required(region.subregion(self.parse_range())?.bytes())?;
        self.parse(region)
    }

    /// Returns the complete physical header allocation.
    pub const fn range(&self) -> FileRange {
        self.range
    }

    /// Returns the physical bytes required to parse the header.
    ///
    /// This range equals [`range`](Self::range).
    pub const fn parse_range(&self) -> FileRange {
        self.range
    }
}

/// The common fields of a version 0 free-space manager header.
///
/// Parsing checks the signature, version, address-space width, and section-list lengths. With
/// the `checksum` feature, parsing also verifies the header checksum. Client-specific section
/// counts, classes, and records require separate validation. The stored section-list address
/// is classified as defined or undefined at the file's offset width.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FreeSpaceManagerHeaderFrame {
    client_id: u8,
    total_space: u64,
    total_sections: u64,
    serialized_sections: u64,
    unserialized_sections: u64,
    section_classes: u16,
    shrink_percent: u16,
    expand_percent: u16,
    addr_space_bits: AddressSpaceBits,
    max_section_size: u64,
    section_list_addr: DecodedStoredAddress,
    section_list_used: u64,
    section_list_allocated: u64,
    widths: FormatWidths,
}

impl FreeSpaceManagerHeaderFrame {
    /// Parses common framing at the start of `data`.
    ///
    /// The caller passes the file's offset and length widths. Trailing bytes are ignored. The
    /// client ID and section counts are retained for client-specific validation.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `data` is shorter than the header,
    /// [`FormatError::ChecksumMismatch`] if the checksum differs with the `checksum` feature
    /// enabled, [`FormatError::UnsupportedVersion`] if the version is not 0, or
    /// [`FormatError::InvalidFreeSpaceManager`] if the signature is invalid, the address space
    /// is outside 1 to 64 bits, or the used section-list length exceeds its allocated length.
    pub fn parse(
        widths @ FormatWidths { offsets, lengths }: FormatWidths,
        data: &[u8],
    ) -> Result<Self, FormatError> {
        let block = data
            .get(..header_len(widths))
            .ok_or(FormatError::UnexpectedEof {
                expected: header_len(widths),
                available: data.len(),
            })?;
        let mut cursor = bytes::ByteCursor::new(block);
        let signature = cursor.array::<{ FSHD_SIGNATURE.len() }>()?;
        if &signature != FSHD_SIGNATURE {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the header signature is b\"{}\", not b\"{}\"",
                signature.escape_ascii(),
                FSHD_SIGNATURE.escape_ascii()
            )));
        }
        checksum::verify_trailing(block)?;
        let version = cursor.u8()?;
        if version != FSHD_VERSION {
            return Err(FormatError::UnsupportedVersion(version));
        }
        let client_id = cursor.u8()?;
        let total_space = cursor.length(lengths)?;
        let total_sections = cursor.length(lengths)?;
        let serialized_sections = cursor.length(lengths)?;
        let unserialized_sections = cursor.length(lengths)?;
        let section_classes = cursor.u16()?;
        let (shrink_percent, expand_percent) = (cursor.u16()?, cursor.u16()?);
        let bits = cursor.u16()?;
        let addr_space_bits = AddressSpaceBits::new(bits).ok_or_else(|| {
            FormatError::InvalidFreeSpaceManager(format!(
                "an address space of {bits} bits is not 1 to 64 bits wide"
            ))
        })?;
        let max_section_size = cursor.length(lengths)?;
        let section_list_addr =
            crate::width::decode_stored_address(offsets, block, cursor.position())?;
        cursor.address(offsets)?;
        let section_list_used = cursor.length(lengths)?;
        let section_list_allocated = cursor.length(lengths)?;
        if section_list_used > section_list_allocated {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the section list uses {section_list_used} bytes, more than the {section_list_allocated} bytes \
                 allocated to it"
            )));
        }
        Ok(Self {
            client_id,
            total_space,
            total_sections,
            serialized_sections,
            unserialized_sections,
            section_classes,
            shrink_percent,
            expand_percent,
            addr_space_bits,
            max_section_size,
            section_list_addr,
            section_list_used,
            section_list_allocated,
            widths,
        })
    }

    /// Creates a section-list reference at the resolved physical address.
    ///
    /// The caller resolves [`section_list_addr`](Self::section_list_addr) to `address` and
    /// passes this header's stored address as `header_addr` to check the address in the section list.
    /// The complete allocation and the used range are recorded separately.
    /// Construction checks range arithmetic without checking source bounds or ownership.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFreeSpaceManager`] if the section-list address or
    /// `header_addr` is undefined, or the used length is shorter than the prefix and checksum.
    /// Returns [`FormatError::AddressTooLarge`] if `header_addr` exceeds the offset width, or
    /// [`FormatError::OffsetOverflow`] if either physical range ends past [`u64::MAX`].
    pub fn section_list_ref(
        &self,
        header_addr: StoredAddress,
        address: PhysicalAddress,
    ) -> Result<FreeSpaceManagerSectionListRef, FormatError> {
        FreeSpaceManagerSectionListRef::new(self, header_addr, address)
    }

    pub(crate) fn widths(&self) -> FormatWidths {
        self.widths
    }

    pub(super) fn address_space(&self) -> AddressSpaceBits {
        self.addr_space_bits
    }

    /// Returns the client ID for client-specific validation.
    pub fn client_id(&self) -> u8 {
        self.client_id
    }

    /// Returns the number of free bytes recorded in "Total Space Tracked".
    pub fn total_space(&self) -> u64 {
        self.total_space
    }

    /// Returns the count recorded in "Total Number of Sections".
    pub fn total_sections(&self) -> u64 {
        self.total_sections
    }

    /// Returns the count recorded in "Number of Serialized Sections".
    pub fn serialized_sections(&self) -> u64 {
        self.serialized_sections
    }

    /// Returns the count recorded in "Number of Un-Serialized Sections".
    pub fn unserialized_sections(&self) -> u64 {
        self.unserialized_sections
    }

    /// Returns the count recorded in "Number of Section Classes".
    pub fn section_classes(&self) -> u16 {
        self.section_classes
    }

    /// Returns the percentage recorded in "Shrink Percent".
    pub fn shrink_percent(&self) -> u16 {
        self.shrink_percent
    }

    /// Returns the percentage recorded in "Expand Percent".
    pub fn expand_percent(&self) -> u16 {
        self.expand_percent
    }

    /// Returns the bit count recorded in "Size of Address Space".
    pub fn addr_space_bits(&self) -> u16 {
        self.addr_space_bits.get()
    }

    /// Returns the size limit recorded in "Maximum Section Size".
    pub fn max_section_size(&self) -> u64 {
        self.max_section_size
    }

    /// Returns the stored section-list address with its undefined value classified.
    pub fn section_list_addr(&self) -> DecodedStoredAddress {
        self.section_list_addr
    }

    /// Returns the used section-list length in bytes, including its checksum.
    pub fn section_list_used(&self) -> u64 {
        self.section_list_used
    }

    /// Returns the complete section-list allocation length in bytes.
    pub fn section_list_allocated(&self) -> u64 {
        self.section_list_allocated
    }
}

/// Returns the length in bytes of a header at `widths`, 82 for 8-byte addresses and lengths.
pub fn free_space_manager_header_len(widths: FormatWidths) -> u64 {
    header_len(widths).to_u64()
}

/// Returns the length in bytes of a header at `widths`, checksum included.
pub(super) const fn header_len(FormatWidths { offsets, lengths }: FormatWidths) -> usize {
    let (os, ls) = (offsets.get() as usize, lengths.get() as usize);
    // sig(4) ver(1) client(1) + 4*L + classes/shrink/expand/abits (2 each) +
    // max(L) + fsse_addr(O) + used(L) + alloc(L) + checksum(4)
    4 + 1 + 1 + 4 * ls + 2 * 4 + ls + os + ls + ls + CHECKSUM_LEN
}

pub(super) const FSHD_SIGNATURE: &[u8; 4] = b"FSHD";

/// The version of a free-space manager header, the one version the format specification describes.
pub(super) const FSHD_VERSION: u8 = 0;

pub(super) const CHECKSUM_LEN: usize = 4;

#[cfg(test)]
#[path = "tests/framing.rs"]
mod tests;
