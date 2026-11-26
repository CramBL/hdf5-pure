use alloc::format;

use hdf5_pure_core::__private::{DecodedStoredAddress, FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;

use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::OffsetWidth;

use super::header::CHECKSUM_LEN;
use super::header::FreeSpaceManagerHeaderFrame;

/// A physical reference to a version 0 free-space section list.
///
/// The complete allocation may be longer than the used bytes required for parsing. A
/// successful parse validates common framing and the header address. Client-specific record
/// validation and storage ownership are separate checks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FreeSpaceManagerSectionListRef {
    range: FileRange,
    parse_range: FileRange,
    offsets: OffsetWidth,
    header_addr: StoredAddress,
}

impl FreeSpaceManagerSectionListRef {
    pub(super) fn new(
        frame: &FreeSpaceManagerHeaderFrame,
        header_addr: StoredAddress,
        address: PhysicalAddress,
    ) -> Result<Self, FormatError> {
        if frame.section_list_addr() == DecodedStoredAddress::Undefined {
            return Err(FormatError::InvalidFreeSpaceManager(
                "the header stores no section list address".into(),
            ));
        }
        if header_addr.get() > StoredAddress::undefined(frame.widths().offsets).get() {
            return Err(FormatError::AddressTooLarge {
                address: header_addr.get(),
                offset_size: frame.widths().offsets.get(),
            });
        }
        if header_addr.is_undefined(frame.widths().offsets) {
            return Err(FormatError::InvalidFreeSpaceManager(
                "the section list's expected header address is undefined".into(),
            ));
        }
        let minimum = (section_list_prefix_len(frame.widths().offsets) + CHECKSUM_LEN).to_u64();
        if frame.section_list_used() < minimum {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the section list uses {} bytes, fewer than its {minimum}-byte prefix and checksum",
                frame.section_list_used()
            )));
        }
        Ok(Self {
            range: FileRange::new(address, frame.section_list_allocated())?,
            parse_range: FileRange::new(address, frame.section_list_used())?,
            offsets: frame.widths().offsets,
            header_addr,
        })
    }

    /// Borrows the record bytes after validating common section-list framing.
    ///
    /// Checks the signature, version, and stored header address. With the `checksum` feature,
    /// verifies the checksum at the end of [`parse_range`](Self::parse_range). Bytes in the
    /// allocation after that range are ignored. Client-specific records are not decoded.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`parse_range`](Self::parse_range), [`FormatError::ChecksumMismatch`] if the checksum
    /// differs with the `checksum` feature enabled, [`FormatError::UnsupportedVersion`] if
    /// the version is not 0, or [`FormatError::InvalidFreeSpaceManager`] if the signature or
    /// header address is invalid.
    pub fn parse<'a>(
        &self,
        region: FileRegion<'a>,
    ) -> Result<FreeSpaceManagerSectionListView<'a>, FormatError> {
        let Self {
            range: _,
            parse_range,
            offsets,
            header_addr,
        } = *self;
        let bytes = region.subregion(parse_range)?.bytes();
        parse_section_list_prefix(offsets, bytes, header_addr)?;
        Ok(FreeSpaceManagerSectionListView {
            records: bytes
                .get(section_list_prefix_len(offsets)..bytes.len() - CHECKSUM_LEN)
                .ok_or_else(|| {
                    FormatError::Internal("a parsed section list has no record range".into())
                })?,
        })
    }

    /// Borrows the record bytes after validating framing and the trailing checksum.
    ///
    /// Checksum verification is required regardless of the `checksum` feature. Bytes after
    /// [`parse_range`](Self::parse_range) are excluded from the checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ChecksumMismatch`] if the stored checksum differs from the
    /// computed one. Other failures are the errors of [`parse`](Self::parse).
    pub fn parse_checked<'a>(
        &self,
        region: FileRegion<'a>,
    ) -> Result<FreeSpaceManagerSectionListView<'a>, FormatError> {
        #[cfg(not(feature = "checksum"))]
        checksum::verify_trailing_required(region.subregion(self.parse_range())?.bytes())?;
        self.parse(region)
    }

    /// Returns the complete physical allocation, including unused capacity.
    pub const fn range(&self) -> FileRange {
        self.range
    }

    /// Returns the used physical bytes, including the prefix and trailing checksum.
    pub const fn parse_range(&self) -> FileRange {
        self.parse_range
    }
}

/// Borrowed section-record bytes with validated common framing.
///
/// The prefix and checksum are excluded. The bytes may include padding and require
/// client-specific decoding before their section offsets or sizes are used.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FreeSpaceManagerSectionListView<'a> {
    records: &'a [u8],
}

impl<'a> FreeSpaceManagerSectionListView<'a> {
    /// Returns the borrowed bytes between the prefix and checksum.
    pub const fn records(&self) -> &'a [u8] {
        self.records
    }
}

/// Checks the section-list signature, version, and stored header address.
///
/// With the `checksum` feature, verifies the checksum over the used section-list bytes.
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if `list` is shorter than the prefix and checksum,
/// [`FormatError::ChecksumMismatch`] if the checksum differs with the `checksum` feature
/// enabled, [`FormatError::UnsupportedVersion`] if the version is not 0, or
/// [`FormatError::InvalidFreeSpaceManager`] if the signature or header address is invalid.
pub(super) fn parse_section_list_prefix(
    offsets: OffsetWidth,
    list: &[u8],
    header_addr: StoredAddress,
) -> Result<(), FormatError> {
    let prefix_len = section_list_prefix_len(offsets);
    if list.len() < prefix_len + CHECKSUM_LEN {
        return Err(FormatError::UnexpectedEof {
            expected: prefix_len + CHECKSUM_LEN,
            available: list.len(),
        });
    }
    let mut cursor = bytes::ByteCursor::new(list);
    let signature = cursor.array::<{ FSSE_SIGNATURE.len() }>()?;
    if &signature != FSSE_SIGNATURE {
        return Err(FormatError::InvalidFreeSpaceManager(format!(
            "the section list signature is b\"{}\", not b\"{}\"",
            signature.escape_ascii(),
            FSSE_SIGNATURE.escape_ascii()
        )));
    }
    checksum::verify_trailing(list)?;
    let version = cursor.u8()?;
    if version != FSSE_VERSION {
        return Err(FormatError::UnsupportedVersion(version));
    }
    let owner_addr = match crate::width::decode_stored_address(offsets, list, cursor.position())? {
        DecodedStoredAddress::Defined(addr) => addr,
        DecodedStoredAddress::Undefined => StoredAddress::undefined(offsets),
    };
    if owner_addr != header_addr {
        return Err(FormatError::InvalidFreeSpaceManager(format!(
            "the section list belongs to the header at {}, not to the header at {}",
            owner_addr.get(),
            header_addr.get()
        )));
    }
    Ok(())
}

/// Returns the length in bytes of the fields of a section list before its first set: the
/// signature, the version, and the header address.
pub(super) fn section_list_prefix_len(offsets: OffsetWidth) -> usize {
    FSSE_SIGNATURE.len() + 1 + usize::from(offsets.get())
}

pub(super) const FSSE_SIGNATURE: &[u8; 4] = b"FSSE";

/// The version of a section list, the one version the format specification describes.
pub(super) const FSSE_VERSION: u8 = 0;
