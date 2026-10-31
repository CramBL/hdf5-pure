//! Free-space manager headers (`FSHD`), section lists (`FSSE`), and file-client records.
//!
//! The common framing APIs parse caller-supplied bytes and retain the client ID for
//! client-specific validation. [`FreeSpaceManagerHeader`] and [`parse_section_info`] validate
//! file-client metadata. The complete allocation and used bytes required for parsing are
//! recorded separately in a section-list reference.
//!
//! Version 0 headers and section lists are defined in "Free-space Index" of the [format
//! specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex

use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::format;
use alloc::vec::Vec;
use core::num::NonZeroU64;

use hdf5_pure_core::__private::DecodedStoredAddress;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;

use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::bytes;
use crate::checksum;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::fractal_heap::FractalHeapHeader;
use crate::fractal_heap_free_space::FractalHeapFreeSpaceManager;
use crate::width;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

const FSHD_SIGNATURE: &[u8; 4] = b"FSHD";
const FSSE_SIGNATURE: &[u8; 4] = b"FSSE";

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
/// use hdf5_pure_core::PhysicalAddress;
/// use hdf5_pure_format::__private as format;
/// use hdf5_pure_format::__private::FormatWidths;
/// use hdf5_pure_format::__private::FreeSection;
/// use hdf5_pure_format::__private::FreeSpaceManagerHeaderRef;
///
/// let widths = FormatWidths::from_sizes(8, 8)?;
/// let header_addr = StoredAddress::new(64);
/// # let list_addr = StoredAddress::new(146);
/// # let (header_bytes, list_bytes) = format::serialize_free_space_manager(
/// #     widths, &[FreeSection { addr: StoredAddress::new(256), size: 32 }],
/// #     header_addr, list_addr, format::SECTION_CLASS_SIMPLE,
/// # )?;
/// let base = BaseAddress::new(512);
/// let header_ref = FreeSpaceManagerHeaderRef::new(
///     widths, PhysicalAddress::new(base.absolute(header_addr)?),
/// )?;
/// let header_region = FileRegion::new(header_ref.parse_range(), &header_bytes)?;
/// let frame = header_ref.parse(header_region)?;
/// let header = frame.file_header()?;
/// let DecodedStoredAddress::Defined(list_addr) = frame.section_list_addr() else {
///     panic!("expected a defined section list address")
/// };
/// let list_ref = frame.section_list_ref(
///     header_addr, PhysicalAddress::new(base.absolute(list_addr)?),
/// )?;
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
        let mut fields = bytes::Fields::new(block, 0);
        let signature = fields.array::<{ FSHD_SIGNATURE.len() }>()?;
        if &signature != FSHD_SIGNATURE {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the header signature is b\"{}\", not b\"{}\"",
                signature.escape_ascii(),
                FSHD_SIGNATURE.escape_ascii()
            )));
        }
        checksum::verify_trailing(block)?;
        let version = fields.u8()?;
        if version != FSHD_VERSION {
            return Err(FormatError::UnsupportedVersion(version));
        }
        let client_id = fields.u8()?;
        let total_space = fields.length(lengths)?;
        let total_sections = fields.length(lengths)?;
        let serialized_sections = fields.length(lengths)?;
        let unserialized_sections = fields.length(lengths)?;
        let section_classes = fields.u16()?;
        let (shrink_percent, expand_percent) = (fields.u16()?, fields.u16()?);
        let bits = fields.u16()?;
        let addr_space_bits = AddressSpaceBits::new(bits).ok_or_else(|| {
            FormatError::InvalidFreeSpaceManager(format!(
                "an address space of {bits} bits is not 1 to 64 bits wide"
            ))
        })?;
        let max_section_size = fields.length(lengths)?;
        let section_list_addr = crate::width::decode_stored_address(offsets, block, fields.pos())?;
        fields.address(offsets)?;
        let section_list_used = fields.length(lengths)?;
        let section_list_allocated = fields.length(lengths)?;
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

    /// Validates the header fields for the file client, client ID 1.
    ///
    /// For a nonempty manager, checks that the section list has a defined address and enough
    /// used bytes for the declared sections. Section records require [`parse_section_info`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFreeSpaceManager`] if the client ID is not 1, the section
    /// counts do not describe fully serialized sections, the class count exceeds 3, an empty
    /// manager records nonzero free bytes, or a nonempty manager has an undefined section-list
    /// address or insufficient used bytes.
    pub fn file_header(&self) -> Result<FreeSpaceManagerHeader, FormatError> {
        let Self {
            client_id,
            total_space,
            total_sections,
            serialized_sections,
            unserialized_sections,
            section_classes,
            shrink_percent: _,
            expand_percent: _,
            addr_space_bits,
            max_section_size,
            section_list_addr,
            section_list_used,
            section_list_allocated,
            widths,
        } = *self;
        let FormatWidths { offsets, lengths } = widths;
        if client_id != FILE_FSM_CLIENT_ID {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the client ID is {client_id}, not {FILE_FSM_CLIENT_ID}, the ID of the file client"
            )));
        }
        if (serialized_sections, unserialized_sections) != (total_sections, 0) {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the header's section count is {total_sections}, its serialized section count \
                 {serialized_sections} and its unserialized section count {unserialized_sections}, \
                 and the file client serializes every section"
            )));
        }
        if section_classes > FILE_FSM_NUM_CLASSES {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the header registers {section_classes} section classes, more than the \
                 {FILE_FSM_NUM_CLASSES} of the file client"
            )));
        }
        let header = FreeSpaceManagerHeader {
            total_space,
            total_sections,
            addr_space_bits,
            max_section_size,
            section_list_addr: match section_list_addr {
                DecodedStoredAddress::Defined(addr) => addr,
                DecodedStoredAddress::Undefined => StoredAddress::undefined(offsets),
            },
            section_list_used,
            section_list_allocated,
        };
        if total_sections == 0 {
            if total_space != 0 {
                return Err(FormatError::InvalidFreeSpaceManager(format!(
                    "the empty manager tracks {total_space} bytes"
                )));
            }
            return Ok(header);
        }
        if section_list_addr == DecodedStoredAddress::Undefined {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the header's section count is {total_sections}, and it stores no section list"
            )));
        }
        let max_sections = header
            .section_fields(lengths)
            .max_sections(offsets, section_list_used);
        if total_sections > max_sections {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the header's section count is {total_sections}, and a section list of {section_list_used} \
                 bytes holds at most {max_sections}"
            )));
        }
        Ok(header)
    }

    /// Validates the header fields against the fractal heap's doubling table.
    ///
    /// Checks client ID 0, the four semantic section classes, the heap address width and
    /// maximum direct-block size, and the sum of serialized and ghost section counts.
    /// The returned manager provides client-specific section-list parsing.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFreeSpaceManager`] if the client, section counts, or heap
    /// geometry disagree, a nonempty manager has no serialized proxy, or serialized records
    /// have an undefined section-list address. Returns [`FormatError::InvalidFractalHeapGeometry`]
    /// for invalid doubling-table geometry, and [`FormatError::OffsetOverflow`] if derived heap
    /// offsets overflow.
    ///
    /// # Examples
    ///
    /// The caller supplies header and used section-list bytes. The reconstructed section
    /// count includes normal rows represented by serialized first-row records. The caller
    /// reconstructs a normal row and an indirect child from a parsed first-row record.
    ///
    /// ```
    /// use hdf5_pure_core::__private::BaseAddressExt;
    /// use hdf5_pure_core::__private::DecodedStoredAddress;
    /// use hdf5_pure_core::__private::FileRegion;
    /// use hdf5_pure_core::PhysicalAddress;
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::BaseAddress;
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_format::__private::FormatWidths;
    /// use hdf5_pure_format::__private::FreeSpaceManagerHeaderRef;
    /// use hdf5_pure_format::__private::FractalHeapFreeSection;
    /// use hdf5_pure_format::__private::FractalHeapSectionClass;
    /// # use hdf5_pure_format::__private::FractalHeapHeader;
    /// # use hdf5_pure_format::FormatError;
    /// # use test_util::fractal_heap;
    /// # let widths = FormatWidths::from_sizes(8, 8)?;
    /// # let heap_bytes = fractal_heap::Header::new(256).build(test_util::widths::Widths::EIGHT);
    /// # let mut heap = FractalHeapHeader::parse_bytes(&heap_bytes, widths.offsets, widths.lengths)?;
    /// # heap.current_rows_in_root_indirect_block = 6;
    /// # heap.managed_space = 16384;
    /// # let header_bytes = [
    /// #     0x46, 0x53, 0x48, 0x44, 0x00, 0x00, 0x01, 0x0f, 0x00, 0x00, 0x00, 0x00,
    /// #     0x00, 0x00, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
    /// #     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
    /// #     0x00, 0x00, 0x04, 0x00, 0x50, 0x00, 0x78, 0x00, 0x10, 0x00, 0x00, 0x04,
    /// #     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00,
    /// #     0x00, 0x00, 0x1f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f, 0x00,
    /// #     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4e, 0x6e, 0x78, 0x88,
    /// # ];
    /// # let list_bytes = [
    /// #     0x46, 0x53, 0x53, 0x45, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    /// #     0x00, 0x01, 0x71, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
    /// #     0x00, 0x18, 0x00, 0x82, 0x46, 0x8a, 0xb2,
    /// # ];
    /// let widths = FormatWidths::from_sizes(8, 8)?;
    /// let base = BaseAddress::new(512);
    /// let header_addr = StoredAddress::new(32);
    /// let header_ref = FreeSpaceManagerHeaderRef::new(
    ///     widths, PhysicalAddress::new(base.absolute(header_addr)?),
    /// )?;
    /// let frame = header_ref.parse(FileRegion::new(header_ref.parse_range(), &header_bytes)?)?;
    /// let manager = frame.fractal_heap_header(&heap)?;
    /// let DecodedStoredAddress::Defined(list_addr) = frame.section_list_addr() else {
    ///     return Ok(());
    /// };
    /// let list_ref = manager.section_list_ref(
    ///     header_addr, PhysicalAddress::new(base.absolute(list_addr)?),
    /// )?;
    /// let space = list_ref.parse(FileRegion::new(list_ref.parse_range(), &list_bytes)?)?;
    /// assert_eq!(space.sections().len(), 1);
    /// assert_eq!(space.total_sections(), 17);
    /// let Some(FractalHeapFreeSection::FirstRow { indirect, .. }) = space.sections().first() else {
    ///     panic!("expected first-row section");
    /// };
    /// let row = manager.row_section(*indirect, 3)?;
    /// assert_eq!(row.class(), FractalHeapSectionClass::NormalRow);
    /// assert_eq!((row.offset(), row.size(), row.blocks()), (2048, 497, 4));
    /// let child = manager.child_section(*indirect, 5, 0)?;
    /// assert_eq!(child.class(), FractalHeapSectionClass::Indirect);
    /// assert_eq!((child.block_offset(), child.rows(), child.blocks()), (8192, 3, 12));
    /// let child_row = manager.row_section(child, 0)?;
    /// assert_eq!(child_row.class(), FractalHeapSectionClass::NormalRow);
    /// assert_eq!((child_row.offset(), child_row.size()), (8192, 113));
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn fractal_heap_header<'a>(
        &self,
        heap: &'a FractalHeapHeader,
    ) -> Result<FractalHeapFreeSpaceManager<'a>, FormatError> {
        FractalHeapFreeSpaceManager::new(self.widths, self.clone(), heap)
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
        if self.section_list_addr == DecodedStoredAddress::Undefined {
            return Err(FormatError::InvalidFreeSpaceManager(
                "the header stores no section list address".into(),
            ));
        }
        if header_addr.get() > StoredAddress::undefined(self.widths.offsets).get() {
            return Err(FormatError::AddressTooLarge {
                address: header_addr.get(),
                offset_size: self.widths.offsets.get(),
            });
        }
        if header_addr.is_undefined(self.widths.offsets) {
            return Err(FormatError::InvalidFreeSpaceManager(
                "the section list's expected header address is undefined".into(),
            ));
        }
        let minimum = (section_list_prefix_len(self.widths.offsets) + CHECKSUM_LEN).to_u64();
        if self.section_list_used < minimum {
            return Err(FormatError::InvalidFreeSpaceManager(format!(
                "the section list uses {} bytes, fewer than its {minimum}-byte prefix and checksum",
                self.section_list_used
            )));
        }
        Ok(FreeSpaceManagerSectionListRef {
            range: FileRange::new(address, self.section_list_allocated)?,
            parse_range: FileRange::new(address, self.section_list_used)?,
            offsets: self.widths.offsets,
            header_addr,
        })
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

/// One free region of a file: the address the file stores for it and its length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreeSection {
    /// The address of the region.
    pub addr: StoredAddress,
    /// The length of the region in bytes.
    pub size: u64,
}

/// A free-space manager header, signature `FSHD`, with the fields a reader of its section list
/// needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreeSpaceManagerHeader {
    /// The number of free bytes the manager tracks, the "Total Space Tracked" field.
    total_space: u64,
    /// The number of sections the manager tracks, the "Total Number of Sections" field.
    total_sections: u64,
    /// The number of bits an address in the section list takes, the "Size of Address Space"
    /// field.
    addr_space_bits: AddressSpaceBits,
    /// The largest section the manager tracks, the "Maximum Section Size" field.
    max_section_size: u64,
    /// The address of the section list.
    section_list_addr: StoredAddress,
    /// The length of the section list in bytes, the "Size of Serialized Section List Used" field.
    section_list_used: u64,
    /// The allocation length in bytes, the "Allocated Size of Serialized Section List" field.
    section_list_allocated: u64,
}

/// Returns the fewest bytes that hold `value`, and at least 1, as `H5VM_limit_enc_size` computes
/// it.
fn enc_size(value: u64) -> usize {
    let bits = 64 - value.leading_zeros() as usize;
    bits.div_ceil(8).max(1)
}

/// Reads `bytes` as a little-endian unsigned integer.
fn read_uint_le(bytes: &[u8]) -> u64 {
    let mut v = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        v |= (b as u64) << (8 * i);
    }
    v
}

impl FreeSpaceManagerHeader {
    /// Parses the free-space manager header at the start of `data`.
    ///
    /// Reads each length field at the length width of `widths` and the section list address at the
    /// offset width, and accepts version 0 of the header of a file client, client ID 1, alone.
    /// Checks that the used length is at most the allocated length. For a header of one or more
    /// sections, checks that the section list address is defined and the used length holds that
    /// many sections, so a reader
    /// may reserve room for [`total_sections`](Self::total_sections) sections before it reads the
    /// section list.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `data` is shorter than the header,
    /// [`FormatError::ChecksumMismatch`] if the checksum does not match and the `checksum` feature
    /// is enabled, [`FormatError::UnsupportedVersion`] if the version is not 0, and
    /// [`FormatError::InvalidFreeSpaceManager`] if the signature is not `FSHD`, the client ID
    /// is not 1, the serialized section count differs from the section count, the "Number of
    /// Un-Serialized Sections" field is not 0, the header counts more
    /// section classes than the 3 the C library registers for the file client (`H5MF.c`, HDF5
    /// 2.2.0), the address space is 0 bits or wider than 64 bits, the used length exceeds the
    /// allocated length, an empty manager records nonzero free bytes, or a header of one or
    /// more sections stores the undefined address for its section list or a used length too
    /// short for its sections.
    pub fn parse(widths: FormatWidths, data: &[u8]) -> Result<FreeSpaceManagerHeader, FormatError> {
        FreeSpaceManagerHeaderFrame::parse(widths, data)?.file_header()
    }

    /// Returns the number of free bytes the manager tracks, the "Total Space Tracked" field.
    pub fn total_space(&self) -> u64 {
        self.total_space
    }

    /// Returns the number of sections the manager tracks, the "Total Number of Sections" field.
    pub fn total_sections(&self) -> u64 {
        self.total_sections
    }

    /// Returns the number of bits an address in the section list takes, the "Size of Address
    /// Space" field.
    pub fn addr_space_bits(&self) -> u16 {
        self.addr_space_bits.get()
    }

    /// Returns the largest section size the manager tracks, the "Maximum Section Size" field.
    pub fn max_section_size(&self) -> u64 {
        self.max_section_size
    }

    /// Returns the address of the section list, the "Address of Serialized Section List" field.
    pub fn section_list_addr(&self) -> StoredAddress {
        self.section_list_addr
    }

    /// Returns the length of the section list in bytes, the "Size of Serialized Section List
    /// Used" field.
    ///
    /// The parser requires this many bytes, including the checksum.
    pub fn section_list_used(&self) -> u64 {
        self.section_list_used
    }

    /// Returns the allocation length in bytes, the "Allocated Size of Serialized Section List"
    /// field.
    ///
    /// This length is at least [`section_list_used`](Self::section_list_used). Version 0 of the
    /// free-space manager header is defined in "Free-space Index" of the [format specification,
    /// version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex
    pub fn section_list_allocated(&self) -> u64 {
        self.section_list_allocated
    }

    /// Returns the widths of the fields of the section list this header describes, in a file with
    /// lengths of `lengths`.
    fn section_fields(&self, lengths: LengthWidth) -> SectionFields {
        let offset = self.addr_space_bits.offset_width();
        SectionFields {
            count: enc_size(self.total_sections),
            size: self.section_size_width(lengths, offset),
            offset,
        }
    }

    /// Returns the width in bytes of a section size in the section list this header describes, in
    /// a file with lengths of `lengths`, where a section offset takes `offset` bytes.
    ///
    /// The width is the fewest bytes that hold the maximum section size, as "Free-space Index" of
    /// the [format specification, version 4.0][spec] defines it, and `offset` below 8-byte lengths
    /// where the maximum section size is all ones or all ones less one. libhdf5 sizes the section
    /// size fields from the maximum address of its file driver (`H5FS__sinfo_new` in
    /// `H5FSsection.c`), sets the "Size of Address Space" to the number of bits of that address
    /// (`H5MF__create_fstype` in `H5MF.c`), and writes that address into the header truncated to
    /// the length width (`H5FS__cache_hdr_serialize` in `H5FScache.c`), HDF5 1.10.11 to 2.2.0.
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex
    fn section_size_width(&self, lengths: LengthWidth, offset: usize) -> usize {
        if lengths != LengthWidth::Eight
            && self.max_section_size >= width::length_width_max(lengths) - 1
        {
            offset
        } else {
            enc_size(self.max_section_size)
        }
    }

    /// Serializes the header at `widths`, checksum included.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressTooLarge`] or [`FormatError::LengthTooLarge`] if a field does
    /// not fit its width.
    fn encode(
        &self,
        widths @ FormatWidths { offsets, lengths }: FormatWidths,
    ) -> Result<Vec<u8>, FormatError> {
        let Self {
            total_space,
            total_sections,
            addr_space_bits,
            max_section_size,
            section_list_addr,
            section_list_used,
            section_list_allocated,
        } = *self;
        let mut fshd = Vec::with_capacity(header_len(widths));
        fshd.extend_from_slice(FSHD_SIGNATURE);
        fshd.push(FSHD_VERSION);
        fshd.push(FILE_FSM_CLIENT_ID);
        bytes::try_write_length(&mut fshd, total_space, lengths)?;
        bytes::try_write_length(&mut fshd, total_sections, lengths)?;
        bytes::try_write_length(&mut fshd, total_sections, lengths)?; // serialized (all of them)
        bytes::try_write_length(&mut fshd, 0, lengths)?; // ghost (none)
        fshd.extend_from_slice(&FILE_FSM_NUM_CLASSES.to_le_bytes());
        fshd.extend_from_slice(&FILE_FSM_SHRINK_PCT.to_le_bytes());
        fshd.extend_from_slice(&FILE_FSM_EXPAND_PCT.to_le_bytes());
        fshd.extend_from_slice(&addr_space_bits.get().to_le_bytes());
        bytes::try_write_length(&mut fshd, max_section_size, lengths)?;
        bytes::try_write_offset(&mut fshd, section_list_addr.get(), offsets)?;
        bytes::try_write_length(&mut fshd, section_list_used, lengths)?; // section info used
        bytes::try_write_length(&mut fshd, section_list_allocated, lengths)?;
        let checksum = checksum::jenkins_lookup3(&fshd);
        fshd.extend_from_slice(&checksum.to_le_bytes());
        Ok(fshd)
    }
}

/// The number of section classes the header of a file free-space manager stores, the three
/// classes the C library registers for the file client (`H5MF.c`, HDF5 2.2.0).
const FILE_FSM_NUM_CLASSES: u16 = 3;
/// The shrink percent the header stores, `H5MF_FSPACE_SHRINK`.
const FILE_FSM_SHRINK_PCT: u16 = 80;
/// The expand percent the header stores, `H5MF_FSPACE_EXPAND`.
const FILE_FSM_EXPAND_PCT: u16 = 120;
/// The client ID of a file free-space manager, where 0 is a fractal heap's.
const FILE_FSM_CLIENT_ID: u8 = 1;

/// The maximum section size libhdf5 gives a file free-space manager under its default driver, sec2,
/// whatever the file's widths: the driver's maximum address, `2^63 - 1` (`H5MF__create_fstype` in
/// `H5MF.c`, HDF5 1.10.11 to 2.2.0), which HDF5 2.2.0 defines as `H5FD_MAXADDR`. The family, multi
/// and MPI-IO drivers give `HADDR_MAX`, `2^64 - 2`, and the core driver `SIZE_MAX - 1`.
const FILE_FSM_MAX_SECTION_SIZE: u64 = (1u64 << 63) - 1;

/// The section class of the free space of a file that is not paged, `H5MF_FSPACE_SECT_SIMPLE`.
///
/// A section of each of the three classes stores its offset and its class, and no data.
pub const SECTION_CLASS_SIMPLE: u8 = 0;
/// The section class of free space smaller than a page in a paged file, `H5MF_FSPACE_SECT_SMALL`.
pub const SECTION_CLASS_SMALL: u8 = 1;
/// The section class of the manager of free space of a page or more in a paged file,
/// `H5MF_FSPACE_SECT_LARGE`.
///
/// The class identifies the manager, and a section of it may be smaller than a page.
pub const SECTION_CLASS_LARGE: u8 = 2;

/// Appends the low `width` bytes of `value` to `buf`, little-endian.
fn push_uint_le(buf: &mut Vec<u8>, value: u64, width: usize) {
    // `width` is always 1..=8 (the count, size, and length widths). Take the low bytes of
    // the little-endian encoding without a narrowing cast.
    buf.extend_from_slice(&value.to_le_bytes()[..width]);
}

/// Serializes the header, for `fshd_addr`, and the section list, for `fsse_addr`, of a file
/// free-space manager that holds `sections`.
///
/// The header stores the address of the section list, and the section list the address of the
/// header. The section list groups the sections by size, in ascending order of size and then of
/// address, as the C library serializes them (`H5FScache.c`, HDF5 2.2.0), and tags every section
/// with `class_id`, since a manager holds sections of one class. Each block ends in its checksum,
/// and in a file with 8-byte addresses and lengths both match the blocks the C library writes for
/// the same sections.
///
/// Below 8-byte lengths the header stores a maximum section size of all ones less two, and the
/// section list stores each section size at the length width. libhdf5 allocates as many
/// bins as the base 2 logarithm of the maximum section size, rounded down, and files a section of
/// `size` bytes in bin `log2(size)`, so it has a bin only for a section smaller than the largest
/// power of two at or below the maximum section size.
///
/// Returns the header and the section list.
///
/// # Errors
///
/// Returns [`FormatError::InvalidFreeSpaceManager`] if a section is not smaller than that power of
/// two, [`FormatError::AddressTooLarge`] if `fshd_addr`, `fsse_addr`, or the address of a section
/// does not fit the offset width of `widths`, [`FormatError::LengthTooLarge`] if the total size
/// of the sections, their number, or the length of the section list does not fit the length width,
/// and [`FormatError::Internal`] if the sizes of `sections` sum past `u64::MAX`.
pub fn serialize_free_space_manager(
    widths: FormatWidths,
    sections: &[FreeSection],
    fshd_addr: StoredAddress,
    fsse_addr: StoredAddress,
    class_id: u8,
) -> Result<(Vec<u8>, Vec<u8>), FormatError> {
    serialize_free_space_manager_with_allocation(
        widths, sections, fshd_addr, fsse_addr, class_id, None,
    )
}

/// Serializes a free-space manager with a separately specified section-list allocation.
///
/// Passing `None` serializes the shortest section list for `sections`. Passing `Some` reserves
/// that many bytes in the returned section list, with zero padding between the section records
/// and its trailing checksum. The header stores the complete returned length in both its used
/// and allocated length fields.
///
/// The version 0 header and section list are defined in "Free-space Index" of the [format
/// specification, version 4.0][spec].
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::StoredAddress;
/// use hdf5_pure_core::__private::StoredAddressExt;
/// use hdf5_pure_format::__private as format;
/// use hdf5_pure_format::__private::FormatWidths;
/// use hdf5_pure_format::__private::FreeSection;
/// use hdf5_pure_format::__private::FreeSpaceManagerHeader;
///
/// let widths = FormatWidths::from_sizes(8, 8)?;
/// let sections = [FreeSection { addr: StoredAddress::new(256), size: 32 }];
/// let fshd_addr = StoredAddress::new(512);
/// let fsse_addr = StoredAddress::new(594);
/// let (header_bytes, section_list) = format::serialize_free_space_manager_with_allocation(
///     widths, &sections, fshd_addr, fsse_addr, format::SECTION_CLASS_SIMPLE, None,
/// )?;
/// let used = FreeSpaceManagerHeader::parse(widths, &header_bytes)?.section_list_used();
/// let allocated = used + 64;
/// let (header_bytes, allocated_section_list) =
///     format::serialize_free_space_manager_with_allocation(
///         widths, &sections, fshd_addr, fsse_addr, format::SECTION_CLASS_SIMPLE, Some(allocated),
///     )?;
/// let header = FreeSpaceManagerHeader::parse(widths, &header_bytes)?;
/// assert_eq!(allocated_section_list.len() as u64, allocated);
/// assert!(allocated_section_list.len() > section_list.len());
/// assert_eq!(header.section_list_used(), allocated);
/// assert_eq!(header.section_list_allocated(), allocated);
/// # Ok::<(), hdf5_pure_format::FormatError>(())
/// ```
///
/// # Errors
///
/// Returns [`FormatError::InvalidFreeSpaceManager`] if `section_list_allocated` is shorter than
/// the shortest serialized section list, [`FormatError::LengthTooLarge`] if it exceeds the length
/// width of `widths`, and [`FormatError::ValueTooLargeForPlatform`] if the allocation length does
/// not fit a [`usize`]. Other failures are the errors of [`serialize_free_space_manager`].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex
pub fn serialize_free_space_manager_with_allocation(
    widths: FormatWidths,
    sections: &[FreeSection],
    fshd_addr: StoredAddress,
    fsse_addr: StoredAddress,
    class_id: u8,
    section_list_allocated: Option<u64>,
) -> Result<(Vec<u8>, Vec<u8>), FormatError> {
    let max_section_size = written_max_section_size(widths.lengths);
    let limit = 1 << max_section_size.ilog2();
    if let Some(section) = sections.iter().find(|s| s.size >= limit) {
        return Err(FormatError::InvalidFreeSpaceManager(format!(
            "a free section of {} bytes is not below the {limit}-byte limit of a manager at {}-byte \
             lengths",
            section.size,
            widths.lengths.get()
        )));
    }
    let mut fsse = SectionFields::written(widths, sections.len()).encode_list(
        widths.offsets,
        fshd_addr,
        &size_sets(sections),
        class_id,
    )?;
    let section_list_used = fsse.len().to_u64();
    let section_list_allocated = section_list_allocated.unwrap_or(section_list_used);
    if section_list_allocated < section_list_used {
        return Err(FormatError::InvalidFreeSpaceManager(format!(
            "the section list uses {section_list_used} bytes, more than the \
             {section_list_allocated} bytes allocated to it"
        )));
    }
    if !width::length_width_holds(widths.lengths, section_list_allocated) {
        return Err(FormatError::LengthTooLarge {
            length: section_list_allocated,
            length_size: widths.lengths.get(),
        });
    }
    if section_list_allocated > section_list_used {
        // H5FS__sinfo_unlock in H5FSsection.c, HDF5 2.2.0, requires equal used and allocated
        // lengths. H5FS__cache_sinfo_serialize in H5FScache.c puts padding before the checksum.
        fsse.truncate(fsse.len() - CHECKSUM_LEN);
        fsse.resize(section_list_allocated.to_usize()? - CHECKSUM_LEN, 0);
        let checksum = checksum::jenkins_lookup3(&fsse);
        fsse.extend_from_slice(&checksum.to_le_bytes());
    }
    let header = FreeSpaceManagerHeader {
        total_space: sections
            .iter()
            .try_fold(0, |total: u64, section| total.checked_add(section.size))
            .ok_or_else(|| {
                FormatError::Internal(format!(
                    "the sizes of {} free sections sum past u64::MAX",
                    sections.len()
                ))
            })?,
        total_sections: sections.len().to_u64(),
        addr_space_bits: AddressSpaceBits::written(widths.offsets),
        max_section_size: max_section_size.get(),
        section_list_addr: fsse_addr,
        section_list_used: section_list_allocated,
        section_list_allocated,
    };
    Ok((header.encode(widths)?, fsse))
}

/// Returns the length in bytes of the header and the section list of a file free-space manager
/// that holds `sections`, or 0 if `sections` is empty.
///
/// A file with no free space stores the undefined address in place of a manager. The length
/// depends on `widths` and on the number and the sizes of the sections, and not on their
/// addresses, so a caller sizes the blocks before it places them.
pub fn free_space_manager_len(widths: FormatWidths, sections: &[FreeSection]) -> u64 {
    if sections.is_empty() {
        return 0;
    }
    let sets = sections
        .iter()
        .map(|section| section.size)
        .collect::<BTreeSet<u64>>()
        .len();
    free_space_manager_header_len(widths) + section_list_len(widths, sets, sections.len())
}

/// Returns the length in bytes of a header at `widths`, 82 for 8-byte addresses and lengths.
pub fn free_space_manager_header_len(widths: FormatWidths) -> u64 {
    header_len(widths).to_u64()
}

/// Returns the length in bytes of the section list of a manager that holds sections of
/// `section_sizes`.
///
/// The length depends on `widths`, the number of sections, and the number of distinct sizes, and
/// not on the addresses of the sections, and is the length of the section list
/// [`serialize_free_space_manager`] serializes for sections of these sizes.
pub fn section_info_len(widths: FormatWidths, section_sizes: &[u64]) -> u64 {
    let sets = section_sizes.iter().collect::<BTreeSet<&u64>>().len();
    section_list_len(widths, sets, section_sizes.len())
}

/// Returns a section-list capacity sufficient for at most `max_sections` sections.
///
/// Counts each section as a separate size group, so the capacity covers every grouping of
/// section sizes for `widths`. Includes the prefix and checksum even when the
/// section count is zero.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if the capacity exceeds [`u64::MAX`].
pub fn section_info_capacity(
    widths: FormatWidths,
    max_sections: usize,
) -> Result<u64, FormatError> {
    let SectionFields {
        count,
        size,
        offset,
    } = SectionFields::written(widths, max_sections);
    let per_section = (count + size + offset + SECTION_CLASS_LEN).to_u64();
    let fixed = (section_list_prefix_len(widths.offsets) + CHECKSUM_LEN).to_u64();
    max_sections
        .to_u64()
        .checked_mul(per_section)
        .and_then(|payload| payload.checked_add(fixed))
        .ok_or(FormatError::OffsetOverflow {
            offset: fixed,
            length: max_sections.to_u64(),
        })
}

/// Parses the section list of the manager `header` describes, at the start of `data`, into its free
/// sections.
///
/// The section list is the first `header.section_list_used()` bytes of `data` and ends in its
/// checksum, and `header_addr` is the address the file stores for `header`, relative to the base
/// address. Reads `header.total_sections()` sections and ignores unused bytes between the last
/// section and the checksum. With the `checksum` feature, verifies the entire used list before
/// its final four bytes, including unused bytes. libhdf5 skips the unused bytes when reading
/// (`H5FS__cache_sinfo_deserialize` in `H5FScache.c`, HDF5 1.12.3) and registers 3 section classes
/// for the file client (`H5MF.c`, HDF5 2.2.0).
///
/// # Errors
///
/// Returns [`FormatError::UnexpectedEof`] if `data` is shorter than `header.section_list_used()` or
/// the section list is shorter than its prefix and its checksum, [`FormatError::ChecksumMismatch`]
/// if the checksum does not match and the `checksum` feature is enabled,
/// [`FormatError::UnsupportedVersion`] if the version is not 0,
/// [`FormatError::InvalidFreeSpaceManager`] if the signature is not `FSSE`, the header address
/// is not `header_addr`, a field runs into the checksum, a set holds 0 sections or sections
/// of 0 bytes, the class of a section is not one of the 3 classes the C library
/// registers for the file client, the sets hold more sections than `header.total_sections()`, or
/// the sizes of the sections do not sum to `header.total_space()`, and
/// [`FormatError::ValueTooLargeForPlatform`] if `header.section_list_used()` or the section count
/// does not fit a `usize`.
pub fn parse_section_info(
    widths: FormatWidths,
    data: &[u8],
    header_addr: StoredAddress,
    header: &FreeSpaceManagerHeader,
) -> Result<Vec<FreeSection>, FormatError> {
    let used = header.section_list_used.to_usize()?;
    let list = data.get(..used).ok_or(FormatError::UnexpectedEof {
        expected: used,
        available: data.len(),
    })?;
    let prefix_len = section_list_prefix_len(widths.offsets);
    parse_section_list_prefix(widths.offsets, list, header_addr)?;
    let body = &list[..list.len() - CHECKSUM_LEN];
    let sections = header.section_fields(widths.lengths).decode_sets(
        body,
        prefix_len,
        header.total_sections.to_usize()?,
    )?;
    let space = sections
        .iter()
        .map(|section| u128::from(section.size))
        .sum::<u128>();
    if space != u128::from(header.total_space) {
        return Err(FormatError::InvalidFreeSpaceManager(format!(
            "the sections sum to {space} bytes, not the header's total of {}",
            header.total_space
        )));
    }
    Ok(sections)
}

/// The widths in bytes of the variable-size fields of a section list: the number of sections in a
/// set, the section size of a set, and the offset of a section.
#[derive(Clone, Copy)]
struct SectionFields {
    count: usize,
    size: usize,
    offset: usize,
}

impl SectionFields {
    /// Returns the widths of the fields of the section list [`serialize_free_space_manager`]
    /// writes at `widths` for a manager of `sections` sections.
    fn written(widths: FormatWidths, sections: usize) -> Self {
        Self {
            count: enc_size(sections.to_u64()),
            size: enc_size(written_max_section_size(widths.lengths).get()),
            offset: AddressSpaceBits::written(widths.offsets).offset_width(),
        }
    }

    /// Returns the most sections a section list of `list_len` bytes holds at these widths, in a
    /// file with addresses of `offsets`: the sections of one set between the prefix and the
    /// checksum.
    fn max_sections(self, offsets: OffsetWidth, list_len: u64) -> u64 {
        let Self {
            count,
            size,
            offset,
        } = self;
        let fixed = (section_list_prefix_len(offsets) + count + size + CHECKSUM_LEN).to_u64();
        list_len
            .checked_sub(fixed)
            .map_or(0, |payload| payload / (offset + SECTION_CLASS_LEN).to_u64())
    }

    /// Reads `total` sections from the sets in `body`, starting at byte `start`.
    ///
    /// The caller excludes the checksum from `body`. Ignores unused bytes after the last section.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFreeSpaceManager`] if a field runs past the end of `body`, a
    /// set holds 0 sections or sections of 0 bytes, the class of a section is not one of the 3
    /// classes the C library registers for the file client, or the sets hold more than `total`
    /// sections.
    fn decode_sets(
        self,
        body: &[u8],
        start: usize,
        total: usize,
    ) -> Result<Vec<FreeSection>, FormatError> {
        let Self {
            count: count_width,
            size: size_width,
            offset: offset_width,
        } = self;
        let field = |pos: usize, width: usize, field_name: &str| {
            body.get(pos..pos + width).map(read_uint_le).ok_or_else(|| {
                FormatError::InvalidFreeSpaceManager(format!(
                    "the {field_name} at byte {pos} of the section list runs into its checksum at \
                     byte {}",
                    body.len()
                ))
            })
        };
        // `FreeSpaceManagerHeader::parse` bounds `total` by the sections a section list of the
        // header's used length holds, and `body` is that section list less its checksum.
        let mut sections = Vec::with_capacity(total);
        let mut pos = start;
        while sections.len() < total {
            let count = field(pos, count_width, "section count of a set")?;
            if count == 0 {
                return Err(FormatError::InvalidFreeSpaceManager(format!(
                    "the set at byte {pos} of the section list holds no sections"
                )));
            }
            let size = field(pos + count_width, size_width, "section size of a set")?;
            if size == 0 {
                return Err(FormatError::InvalidFreeSpaceManager(format!(
                    "the set at byte {pos} of the section list holds sections of 0 bytes"
                )));
            }
            pos += count_width + size_width;
            for _ in 0..count {
                if sections.len() == total {
                    return Err(FormatError::InvalidFreeSpaceManager(format!(
                        "the sets hold more sections than the header's count of {total}"
                    )));
                }
                let addr = StoredAddress::new(field(pos, offset_width, "offset of a section")?);
                let class = field(pos + offset_width, SECTION_CLASS_LEN, "class of a section")?;
                if class >= u64::from(FILE_FSM_NUM_CLASSES) {
                    return Err(FormatError::InvalidFreeSpaceManager(format!(
                        "the class of the section at byte {pos} of the section list is {class}, \
                         not one of the {FILE_FSM_NUM_CLASSES} classes of the file client"
                    )));
                }
                pos += offset_width + SECTION_CLASS_LEN;
                sections.push(FreeSection { addr, size });
            }
        }
        Ok(sections)
    }

    /// Serializes the section list of the manager whose header is at `fshd_addr`, with the sets
    /// [`size_sets`] groups and every section of class `class_id`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressTooLarge`] if `fshd_addr` or the address of a section does not
    /// fit `offsets`.
    fn encode_list(
        self,
        offsets: OffsetWidth,
        fshd_addr: StoredAddress,
        sets: &BTreeMap<u64, Vec<StoredAddress>>,
        class_id: u8,
    ) -> Result<Vec<u8>, FormatError> {
        let Self { count, size, .. } = self;
        let mut fsse = Vec::new();
        fsse.extend_from_slice(FSSE_SIGNATURE);
        fsse.push(FSSE_VERSION);
        bytes::try_write_offset(&mut fsse, fshd_addr.get(), offsets)?;
        for (&section_size, addresses) in sets {
            push_uint_le(&mut fsse, addresses.len().to_u64(), count);
            push_uint_le(&mut fsse, section_size, size);
            for addr in addresses {
                bytes::try_write_offset(&mut fsse, addr.get(), offsets)?;
                fsse.push(class_id); // section class id (no class-specific data)
            }
        }
        let checksum = checksum::jenkins_lookup3(&fsse);
        fsse.extend_from_slice(&checksum.to_le_bytes());
        Ok(fsse)
    }
}

/// Returns the maximum section size the serializer writes at `lengths`.
///
/// libhdf5 checks the bin of a section against the number of bins in an `assert` alone
/// (`H5FS__sect_link_size` in `H5FSsection.c`, HDF5 1.10.11 to 2.2.0). Below 8-byte lengths the
/// maximum is all ones less two, which has the top bit of the field set, so libhdf5 allocates as
/// many bins for it as for the largest value of the field (`H5FS__sinfo_new`). It is below the
/// truncated addresses [`FreeSpaceManagerHeader::section_size_width`] reads as libhdf5's.
fn written_max_section_size(lengths: LengthWidth) -> NonZeroU64 {
    match lengths {
        LengthWidth::Two => {
            const { NonZeroU64::new(width::length_width_max(LengthWidth::Two) - 2).unwrap() }
        }
        LengthWidth::Four => {
            const { NonZeroU64::new(width::length_width_max(LengthWidth::Four) - 2).unwrap() }
        }
        LengthWidth::Eight => const { NonZeroU64::new(FILE_FSM_MAX_SECTION_SIZE).unwrap() },
    }
}

/// Groups the addresses of `sections` into one set per section size, the sets in ascending order of
/// size and the addresses of a set in ascending order.
fn size_sets(sections: &[FreeSection]) -> BTreeMap<u64, Vec<StoredAddress>> {
    let mut sets = BTreeMap::<u64, Vec<StoredAddress>>::new();
    for section in sections {
        sets.entry(section.size).or_default().push(section.addr);
    }
    for addresses in sets.values_mut() {
        addresses.sort_unstable();
    }
    sets
}

/// Returns the length in bytes of the section list [`serialize_free_space_manager`] writes at
/// `widths` for `sections` sections of `sets` distinct sizes.
fn section_list_len(widths: FormatWidths, sets: usize, sections: usize) -> u64 {
    let SectionFields {
        count,
        size,
        offset,
    } = SectionFields::written(widths, sections);
    (section_list_prefix_len(widths.offsets)
        + sets * (count + size)
        + sections * (offset + SECTION_CLASS_LEN)
        + CHECKSUM_LEN)
        .to_u64()
}

/// Returns the length in bytes of a header at `widths`, checksum included.
fn header_len(FormatWidths { offsets, lengths }: FormatWidths) -> usize {
    let (os, ls) = (usize::from(offsets.get()), usize::from(lengths.get()));
    // sig(4) ver(1) client(1) + 4*L + classes/shrink/expand/abits (2 each) +
    // max(L) + fsse_addr(O) + used(L) + alloc(L) + checksum(4)
    4 + 1 + 1 + 4 * ls + 2 * 4 + ls + os + ls + ls + CHECKSUM_LEN
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
fn parse_section_list_prefix(
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
    let mut fields = bytes::Fields::new(list, 0);
    let signature = fields.array::<{ FSSE_SIGNATURE.len() }>()?;
    if &signature != FSSE_SIGNATURE {
        return Err(FormatError::InvalidFreeSpaceManager(format!(
            "the section list signature is b\"{}\", not b\"{}\"",
            signature.escape_ascii(),
            FSSE_SIGNATURE.escape_ascii()
        )));
    }
    checksum::verify_trailing(list)?;
    let version = fields.u8()?;
    if version != FSSE_VERSION {
        return Err(FormatError::UnsupportedVersion(version));
    }
    let owner_addr = match crate::width::decode_stored_address(offsets, list, fields.pos())? {
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
fn section_list_prefix_len(offsets: OffsetWidth) -> usize {
    FSSE_SIGNATURE.len() + 1 + usize::from(offsets.get())
}

/// The "Size of Address Space" field of a free-space manager header: the number of bits of an
/// address in the space the manager tracks, 1 to 64.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AddressSpaceBits(u8);

impl AddressSpaceBits {
    /// Returns the address space of `bits` bits, or `None` if `bits` is 0 or more than 64.
    fn new(bits: u16) -> Option<Self> {
        u8::try_from(bits)
            .ok()
            .filter(|bits| (1..=64).contains(bits))
            .map(Self)
    }

    /// Returns the "Size of Address Space" the serializer writes in a file with addresses of
    /// `offsets`, one bit less than the width of an address.
    fn written(offsets: OffsetWidth) -> Self {
        Self(offsets.get() * 8 - 1)
    }

    fn get(self) -> u16 {
        u16::from(self.0)
    }

    /// Returns the width in bytes of a section offset in this address space.
    fn offset_width(self) -> usize {
        usize::from(self.0).div_ceil(8)
    }
}

/// The version of a free-space manager header, the one version the format specification describes.
const FSHD_VERSION: u8 = 0;
/// The version of a section list, the one version the format specification describes.
const FSSE_VERSION: u8 = 0;
const CHECKSUM_LEN: usize = 4;
const SECTION_CLASS_LEN: usize = 1;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::free_space;

    use super::*;

    #[rstest]
    #[case::two(2, 2, 15)]
    #[case::mixed(8, 4, 256)]
    #[case::eight(8, 8, 256)]
    fn section_capacity_bounds_distinct_sizes_and_count_width_changes(
        #[case] offsets: u8,
        #[case] lengths: u8,
        #[case] count: usize,
    ) {
        let widths = widths(offsets, lengths);
        let sizes: Vec<_> = (1..=count as u64).collect();
        let capacity = section_info_capacity(widths, count).unwrap();
        assert_eq!(section_info_len(widths, &sizes), capacity);
        let fewer: Vec<_> = sizes.into_iter().take(count - 1).collect();
        assert!(section_info_len(widths, &fewer) <= capacity);
        assert!(section_info_len(widths, &alloc::vec![1; count]) <= capacity);
    }

    #[rstest]
    #[case::header(true)]
    #[case::section_list(false)]
    fn checked_regions_require_checksums(#[case] header_corrupt: bool) {
        let widths = widths(8, 8);
        let header_address = StoredAddress::new(619);
        let header_ref = FreeSpaceManagerHeaderRef::new(widths, PhysicalAddress::new(619)).unwrap();
        let mut header_bytes = free_space::single_section_header();
        let mut list_bytes = free_space::single_section_info();
        let frame = header_ref
            .parse_checked(FileRegion::new(header_ref.parse_range(), &header_bytes).unwrap())
            .unwrap();
        let list_ref = frame
            .section_list_ref(header_address, PhysicalAddress::new(701))
            .unwrap();
        assert_eq!(
            list_ref
                .parse_checked(FileRegion::new(list_ref.parse_range(), &list_bytes).unwrap())
                .unwrap()
                .records(),
            &list_bytes[13..list_bytes.len() - 4]
        );
        let bytes = if header_corrupt {
            &mut header_bytes
        } else {
            &mut list_bytes
        };
        let last = bytes.len() - 4;
        let computed = checksum::jenkins_lookup3(&bytes[..last]);
        bytes[last] ^= 1;
        let error = if header_corrupt {
            header_ref
                .parse_checked(FileRegion::new(header_ref.parse_range(), &header_bytes).unwrap())
                .unwrap_err()
        } else {
            list_ref
                .parse_checked(FileRegion::new(list_ref.parse_range(), &list_bytes).unwrap())
                .unwrap_err()
        };
        assert_eq!(
            error,
            FormatError::ChecksumMismatch {
                expected: computed ^ 1,
                computed
            }
        );
    }

    #[rstest]
    #[case::two(
        widths(2, 2),
        free_space::two_byte_widths_header(),
        free_space::two_byte_widths_section_info(),
        0x01E5,
        0x0207
    )]
    #[case::four(
        widths(4, 4),
        free_space::four_byte_widths_header(),
        free_space::four_byte_widths_section_info(),
        0x0217,
        0x0249
    )]
    #[case::mixed(
        widths(8, 4),
        free_space::four_byte_lengths_header(),
        free_space::four_byte_lengths_section_info(),
        0x0273,
        0x02A9
    )]
    #[case::eight(
        widths(8, 8),
        free_space::single_section_header(),
        free_space::single_section_info(),
        619,
        701
    )]
    fn region_sequence_parses_c_library_manager(
        #[case] widths: FormatWidths,
        #[case] header_bytes: Vec<u8>,
        #[case] list_bytes: Vec<u8>,
        #[case] header_at: u64,
        #[case] list_at: u64,
    ) {
        use hdf5_pure_core::__private::BaseAddressExt;
        use hdf5_pure_core::BaseAddress;

        use crate::__private::FreeSpaceManagerHeaderRef;

        let base = BaseAddress::new(512);
        let header_addr = StoredAddress::new(header_at);
        let header_ref = FreeSpaceManagerHeaderRef::new(
            widths,
            PhysicalAddress::new(base.absolute(header_addr).unwrap()),
        )
        .unwrap();
        assert_eq!(header_ref.range(), header_ref.parse_range());
        let frame = header_ref
            .parse(FileRegion::new(header_ref.parse_range(), &header_bytes).unwrap())
            .unwrap();
        let max_section_size = if widths.lengths == LengthWidth::Eight {
            FILE_FSM_MAX_SECTION_SIZE
        } else {
            width::length_width_max(widths.lengths)
        };
        assert_eq!(
            frame.file_header().unwrap(),
            FreeSpaceManagerHeader {
                total_space: 1600,
                total_sections: 1,
                addr_space_bits: AddressSpaceBits(63),
                max_section_size,
                section_list_addr: StoredAddress::new(list_at),
                section_list_used: list_bytes.len() as u64,
                section_list_allocated: list_bytes.len() as u64,
            }
        );
        assert_eq!(frame.client_id(), FILE_FSM_CLIENT_ID);
        assert_eq!(frame.total_space(), 1600);
        assert_eq!(frame.total_sections(), 1);
        assert_eq!(frame.serialized_sections(), 1);
        assert_eq!(frame.unserialized_sections(), 0);
        assert_eq!(frame.section_classes(), FILE_FSM_NUM_CLASSES);
        assert_eq!(frame.shrink_percent(), FILE_FSM_SHRINK_PCT);
        assert_eq!(frame.expand_percent(), FILE_FSM_EXPAND_PCT);
        assert_eq!(frame.addr_space_bits(), 63);
        assert_eq!(frame.max_section_size(), max_section_size);
        let DecodedStoredAddress::Defined(list_addr) = frame.section_list_addr() else {
            panic!("expected a defined section list")
        };
        assert_eq!(list_addr.get(), list_at);
        assert_eq!(frame.section_list_used(), list_bytes.len() as u64);
        assert_eq!(frame.section_list_allocated(), list_bytes.len() as u64);
        let list_ref = frame
            .section_list_ref(
                header_addr,
                PhysicalAddress::new(base.absolute(list_addr).unwrap()),
            )
            .unwrap();
        let view = list_ref
            .parse(FileRegion::new(list_ref.parse_range(), &list_bytes).unwrap())
            .unwrap();
        assert_eq!(list_ref.range(), list_ref.parse_range());
        assert_eq!(
            view.records(),
            &list_bytes[section_list_prefix_len(widths.offsets)..list_bytes.len() - CHECKSUM_LEN]
        );
        assert_eq!(
            view.records().as_ptr(),
            list_bytes[section_list_prefix_len(widths.offsets)..].as_ptr()
        );
    }

    #[rstest]
    #[case::two(widths(2, 2))]
    #[case::four(widths(4, 4))]
    #[case::eight(widths(8, 8))]
    fn framing_classifies_undefined_section_list_at_its_width(#[case] widths: FormatWidths) {
        let (header_bytes, _) = serialize_free_space_manager(
            widths,
            &[],
            StoredAddress::new(32),
            StoredAddress::undefined(widths.offsets),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        let frame = FreeSpaceManagerHeaderFrame::parse(widths, &header_bytes).unwrap();
        assert_eq!(frame.section_list_addr(), DecodedStoredAddress::Undefined);
        assert_eq!(
            frame.file_header().unwrap().section_list_addr(),
            StoredAddress::undefined(widths.offsets)
        );
        assert_eq!(
            frame
                .section_list_ref(StoredAddress::new(32), PhysicalAddress::new(64))
                .unwrap_err(),
            FormatError::InvalidFreeSpaceManager(
                "the header stores no section list address".into()
            )
        );
    }

    #[rstest]
    #[case::heap(FRACTAL_HEAP_FSM_CLIENT_ID)]
    #[case::reserved(2)]
    fn file_semantics_reject_another_client_after_framing(#[case] client: u8) {
        let mut bytes = free_space::single_section_header();
        bytes[5] = client;
        let len = bytes.len();
        test_util::checksum::restamp(&mut bytes, 0, len);
        let frame = FreeSpaceManagerHeaderFrame::parse(widths(8, 8), &bytes).unwrap();
        assert_eq!(frame.client_id(), client);
        assert_eq!(
            frame.file_header().unwrap_err(),
            FormatError::InvalidFreeSpaceManager(format!(
                "the client ID is {client}, not 1, the ID of the file client"
            ))
        );
    }

    #[test]
    fn common_framing_retains_client_specific_counts() {
        let mut bytes = free_space::single_section_header();
        bytes[5] = FRACTAL_HEAP_FSM_CLIENT_ID;
        test_util::bytes::set_slice_at(&mut bytes, 14, &4u64.to_le_bytes());
        test_util::bytes::set_slice_at(&mut bytes, 30, &3u64.to_le_bytes());
        test_util::bytes::set_slice_at(&mut bytes, 38, &FRACTAL_HEAP_FSM_NUM_CLASSES.to_le_bytes());
        let len = bytes.len();
        test_util::checksum::restamp(&mut bytes, 0, len);
        let frame = FreeSpaceManagerHeaderFrame::parse(widths(8, 8), &bytes).unwrap();
        assert_eq!(frame.total_sections(), 4);
        assert_eq!(frame.serialized_sections(), 1);
        assert_eq!(frame.unserialized_sections(), 3);
        assert_eq!(frame.section_classes(), FRACTAL_HEAP_FSM_NUM_CLASSES);
    }

    #[test]
    fn common_framing_preserves_record_class_rejected_by_file_client() {
        let mut bytes = free_space::single_section_info();
        bytes[30] = 0xFF;
        let len = bytes.len();
        test_util::checksum::restamp(&mut bytes, 0, len);
        let reference = single_frame()
            .section_list_ref(StoredAddress::new(619), PhysicalAddress::new(701))
            .unwrap();
        assert_eq!(
            reference.parse(region(701, &bytes)).unwrap().records(),
            &bytes[13..31]
        );
        assert_eq!(parse_section_info(widths(8, 8), &bytes, StoredAddress::new(619), &single_frame().file_header().unwrap()).unwrap_err(),
            FormatError::InvalidFreeSpaceManager("the class of the section at byte 22 of the section list is 255, not one of the 3 classes of the file client".into()));
    }

    #[test]
    fn section_list_region_ignores_bytes_after_used_range() {
        let mut bytes = free_space::single_section_info();
        bytes.extend([0xFF; 64]);
        let frame = FreeSpaceManagerHeaderFrame {
            section_list_allocated: bytes.len() as u64,
            ..single_frame()
        };
        let reference = frame
            .section_list_ref(StoredAddress::new(619), PhysicalAddress::new(701))
            .unwrap();
        assert_eq!(
            reference.parse(region(701, &bytes)).unwrap().records(),
            &bytes[13..31]
        );
    }

    #[rstest]
    #[case::short(619, 81)]
    #[case::shifted(620, 82)]
    fn header_region_rejects_insufficient_bytes(#[case] start: u64, #[case] length: usize) {
        let reference =
            FreeSpaceManagerHeaderRef::new(widths(8, 8), PhysicalAddress::new(619)).unwrap();
        let bytes = vec![0; length];
        assert_eq!(
            reference.parse(region(start, &bytes)).unwrap_err(),
            FormatError::InsufficientRegion {
                required_start: 619,
                required_length: 82,
                provided_start: start,
                provided_length: length as u64,
            }
        );
    }

    #[test]
    fn header_region_accepts_enclosing_bytes() {
        let mut bytes = vec![0; 3];
        bytes.extend(free_space::single_section_header());
        bytes.extend([0; 7]);
        let reference =
            FreeSpaceManagerHeaderRef::new(widths(8, 8), PhysicalAddress::new(619)).unwrap();
        assert_eq!(
            reference.parse(region(616, &bytes)).unwrap(),
            FreeSpaceManagerHeaderFrame {
                client_id: FILE_FSM_CLIENT_ID,
                total_space: 1600,
                total_sections: 1,
                serialized_sections: 1,
                unserialized_sections: 0,
                section_classes: FILE_FSM_NUM_CLASSES,
                shrink_percent: FILE_FSM_SHRINK_PCT,
                expand_percent: FILE_FSM_EXPAND_PCT,
                addr_space_bits: AddressSpaceBits(63),
                max_section_size: FILE_FSM_MAX_SECTION_SIZE,
                section_list_addr: DecodedStoredAddress::Defined(StoredAddress::new(701)),
                section_list_used: 35,
                section_list_allocated: 35,
                widths: widths(8, 8),
            }
        );
    }

    #[test]
    fn header_reference_rejects_range_overflow() {
        assert_eq!(
            FreeSpaceManagerHeaderRef::new(widths(8, 8), PhysicalAddress::new(u64::MAX - 81))
                .unwrap_err(),
            FormatError::OffsetOverflow {
                offset: u64::MAX - 81,
                length: 82
            }
        );
    }

    #[rstest]
    #[case::spare(64)]
    #[case::large(u64::MAX - 35)]
    fn section_list_region_requires_only_used_bytes(#[case] spare: u64) {
        let bytes = free_space::single_section_info();
        let frame = FreeSpaceManagerHeaderFrame {
            section_list_allocated: bytes.len() as u64 + spare,
            ..single_frame()
        };
        let reference = frame
            .section_list_ref(StoredAddress::new(619), PhysicalAddress::new(0))
            .unwrap();
        assert_eq!(reference.range().length(), bytes.len() as u64 + spare);
        assert_eq!(reference.parse_range().length(), bytes.len() as u64);
        assert_eq!(
            reference.parse(region(0, &bytes)).unwrap().records(),
            &bytes[13..31]
        );
    }

    #[test]
    fn section_list_reference_checks_complete_allocation_overflow() {
        let frame = FreeSpaceManagerHeaderFrame {
            section_list_allocated: 64,
            ..single_frame()
        };
        assert_eq!(
            frame
                .section_list_ref(StoredAddress::new(619), PhysicalAddress::new(u64::MAX - 35))
                .unwrap_err(),
            FormatError::OffsetOverflow {
                offset: u64::MAX - 35,
                length: 64
            }
        );
    }

    #[test]
    fn section_list_reference_rejects_missing_prefix_geometry() {
        let frame = FreeSpaceManagerHeaderFrame {
            section_list_used: 16,
            ..single_frame()
        };
        assert_eq!(
            frame
                .section_list_ref(StoredAddress::new(619), PhysicalAddress::new(701))
                .unwrap_err(),
            FormatError::InvalidFreeSpaceManager(
                "the section list uses 16 bytes, fewer than its 17-byte prefix and checksum".into()
            )
        );
    }

    #[rstest]
    #[case::two_undefined(
        widths(2, 2),
        0xFFFF,
        "the section list's expected header address is undefined"
    )]
    #[case::four_undefined(
        widths(4, 4),
        0xFFFF_FFFF,
        "the section list's expected header address is undefined"
    )]
    #[case::eight_undefined(
        widths(8, 8),
        u64::MAX,
        "the section list's expected header address is undefined"
    )]
    fn section_list_reference_rejects_undefined_header_address(
        #[case] widths: FormatWidths,
        #[case] address: u64,
        #[case] reason: &str,
    ) {
        let frame = FreeSpaceManagerHeaderFrame {
            widths,
            ..single_frame()
        };
        assert_eq!(
            frame
                .section_list_ref(StoredAddress::new(address), PhysicalAddress::new(701))
                .unwrap_err(),
            FormatError::InvalidFreeSpaceManager(reason.into())
        );
    }

    #[rstest]
    #[case::two(widths(2, 2), 0x1_0000)]
    #[case::four(widths(4, 4), 0x1_0000_0000)]
    #[case::two_wide_sentinel(widths(2, 2), u64::MAX)]
    #[case::four_wide_sentinel(widths(4, 4), u64::MAX)]
    fn section_list_reference_rejects_header_address_outside_width(
        #[case] widths: FormatWidths,
        #[case] address: u64,
    ) {
        let frame = FreeSpaceManagerHeaderFrame {
            widths,
            ..single_frame()
        };
        assert_eq!(
            frame
                .section_list_ref(StoredAddress::new(address), PhysicalAddress::new(701))
                .unwrap_err(),
            FormatError::AddressTooLarge {
                address,
                offset_size: widths.offsets.get()
            }
        );
    }

    #[rstest]
    #[case::short(701, 34)]
    #[case::shifted(702, 35)]
    fn section_list_region_rejects_insufficient_bytes(#[case] start: u64, #[case] length: usize) {
        let reference = single_frame()
            .section_list_ref(StoredAddress::new(619), PhysicalAddress::new(701))
            .unwrap();
        assert_eq!(
            reference
                .parse(region(start, &vec![0; length]))
                .unwrap_err(),
            FormatError::InsufficientRegion {
                required_start: 701,
                required_length: 35,
                provided_start: start,
                provided_length: length as u64,
            }
        );
    }

    #[rstest]
    #[case::signature(0, vec![b'X'], "the section list signature is b\"XSSE\", not b\"FSSE\"")]
    #[case::backlink(5, 736u64.to_le_bytes().to_vec(), "the section list belongs to the header at 736, not to the header at 619")]
    #[case::undefined_backlink(5, u64::MAX.to_le_bytes().to_vec(), "the section list belongs to the header at 18446744073709551615, not to the header at 619")]
    fn section_list_region_rejects_invalid_framing(
        #[case] at: usize,
        #[case] value: Vec<u8>,
        #[case] reason: &str,
    ) {
        let mut bytes = free_space::single_section_info();
        test_util::bytes::set_slice_at(&mut bytes, at, &value);
        let len = bytes.len();
        test_util::checksum::restamp(&mut bytes, 0, len);
        let reference = single_frame()
            .section_list_ref(StoredAddress::new(619), PhysicalAddress::new(701))
            .unwrap();
        assert_eq!(
            reference.parse(region(701, &bytes)).unwrap_err(),
            FormatError::InvalidFreeSpaceManager(reason.into())
        );
    }

    #[cfg(feature = "checksum")]
    #[rstest]
    #[case::header(true)]
    #[case::section_list(false)]
    fn region_parsing_rejects_invalid_checksum(#[case] header: bool) {
        if header {
            let mut bytes = free_space::single_section_header();
            let computed = checksum::jenkins_lookup3(&bytes[..78]);
            bytes[78] ^= 0xFF;
            let reference =
                FreeSpaceManagerHeaderRef::new(widths(8, 8), PhysicalAddress::new(619)).unwrap();
            assert_eq!(
                reference.parse(region(619, &bytes)).unwrap_err(),
                FormatError::ChecksumMismatch {
                    expected: computed ^ 0xFF,
                    computed
                }
            );
        } else {
            let mut bytes = free_space::single_section_info();
            let expected = checksum::jenkins_lookup3(&bytes[..31]);
            bytes[31] ^= 0xFF;
            let reference = single_frame()
                .section_list_ref(StoredAddress::new(619), PhysicalAddress::new(701))
                .unwrap();
            assert_eq!(
                reference.parse(region(701, &bytes)).unwrap_err(),
                FormatError::ChecksumMismatch {
                    expected: expected ^ 0xFF,
                    computed: expected
                }
            );
        }
    }

    fn single_frame() -> FreeSpaceManagerHeaderFrame {
        FreeSpaceManagerHeaderFrame::parse(widths(8, 8), &free_space::single_section_header())
            .unwrap()
    }

    fn region(start: u64, bytes: &[u8]) -> FileRegion<'_> {
        FileRegion::new(
            FileRange::new(PhysicalAddress::new(start), bytes.len() as u64).unwrap(),
            bytes,
        )
        .unwrap()
    }

    /// A free section at `addr` spanning `size` bytes.
    fn section(addr: u64, size: u64) -> FreeSection {
        FreeSection {
            addr: StoredAddress::new(addr),
            size,
        }
    }

    #[test]
    fn parses_c_library_single_section() {
        // manager @619: one 1600-byte free section at offset 2848, FSSE @701.
        let fshd = free_space::single_section_header();
        let header = FreeSpaceManagerHeader::parse(widths(8, 8), &fshd).unwrap();
        assert_eq!(header.total_space, 1600);
        assert_eq!(header.total_sections, 1);
        assert_eq!(header.addr_space_bits(), 63);
        assert_eq!(header.section_list_addr, StoredAddress::new(701));
        assert_eq!(header.section_list_used, 35);

        let fsse = free_space::single_section_info();
        let sections =
            parse_section_info(widths(8, 8), &fsse, StoredAddress::new(619), &header).unwrap();
        assert_eq!(sections, vec![section(2848, 1600)]);
    }

    #[test]
    fn parses_c_library_two_sections() {
        // manager @736: 16 bytes @871 and 893 bytes @1155, FSSE @818.
        let fshd = free_space::two_section_header();
        let header = FreeSpaceManagerHeader::parse(widths(8, 8), &fshd).unwrap();
        assert_eq!(header.total_space, 909);
        assert_eq!(header.total_sections, 2);
        assert_eq!(header.section_list_addr, StoredAddress::new(818));
        assert_eq!(header.section_list_used, 53);

        let fsse = free_space::two_section_info();
        let mut sections =
            parse_section_info(widths(8, 8), &fsse, StoredAddress::new(736), &header).unwrap();
        sections.sort_by_key(|s| s.addr);
        assert_eq!(sections, vec![section(871, 16), section(1155, 893),]);
        // The section sizes sum to the header's tracked total.
        let total: u64 = sections.iter().map(|s| s.size).sum();
        assert_eq!(header.total_space, total);
    }

    #[test]
    fn serialize_matches_c_library_single_section() {
        // Byte-for-byte reproduction of the FSHD@619 / FSSE@701 fixtures,
        // including the Jenkins checksums the C library verifies on read.
        let fshd_fixture = free_space::single_section_header();
        let fsse_fixture = free_space::single_section_info();
        assert_eq!(
            free_space::raw_manager_eight(619, 701, &[(2848, 1600)]),
            (fshd_fixture.clone(), fsse_fixture.clone())
        );
        let (fshd, fsse) = serialize_free_space_manager(
            widths(8, 8),
            &[section(2848, 1600)],
            StoredAddress::new(619),
            StoredAddress::new(701),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        assert_eq!(fshd, fshd_fixture, "FSHD bytes match the C library");
        assert_eq!(fsse, fsse_fixture, "FSSE bytes match the C library");
        // The header length helper agrees with the produced bytes.
        assert_eq!(
            free_space_manager_header_len(widths(8, 8)),
            fshd.len() as u64
        );
    }

    #[test]
    fn serialize_matches_c_library_two_sections() {
        // FSHD@736 / FSSE@818: two differently-sized sections (16 @871, 893 @1155)
        // emitted as two ascending size-groups.
        let fshd_fixture = free_space::two_section_header();
        let fsse_fixture = free_space::two_section_info();
        assert_eq!(
            free_space::raw_manager_eight(736, 818, &[(1155, 893), (871, 16)]),
            (fshd_fixture.clone(), fsse_fixture.clone())
        );
        let (fshd, fsse) = serialize_free_space_manager(
            widths(8, 8),
            &[section(1155, 893), section(871, 16)],
            StoredAddress::new(736),
            StoredAddress::new(818),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        assert_eq!(fshd, fshd_fixture, "FSHD bytes match the C library");
        assert_eq!(fsse, fsse_fixture, "FSSE bytes match the C library");
    }

    #[rstest]
    #[case::two_byte_widths(
        widths(2, 2),
        free_space::two_byte_widths_header(),
        free_space::two_byte_widths_section_info(),
        0x01E5,
        0x0207,
        29
    )]
    #[case::four_byte_widths(
        widths(4, 4),
        free_space::four_byte_widths_header(),
        free_space::four_byte_widths_section_info(),
        0x0217,
        0x0249,
        31
    )]
    #[case::four_byte_lengths(
        widths(8, 4),
        free_space::four_byte_lengths_header(),
        free_space::four_byte_lengths_section_info(),
        0x0273,
        0x02A9,
        35
    )]
    fn a_c_library_manager_parses_at_the_widths_of_its_file(
        #[case] widths: FormatWidths,
        #[case] header_bytes: Vec<u8>,
        #[case] section_list: Vec<u8>,
        #[case] header_addr: u64,
        #[case] section_list_addr: u64,
        #[case] section_list_used: u64,
    ) {
        let header = FreeSpaceManagerHeader::parse(widths, &header_bytes).unwrap();

        assert_eq!(
            header,
            FreeSpaceManagerHeader {
                total_space: 1600,
                total_sections: 1,
                addr_space_bits: AddressSpaceBits(63),
                max_section_size: width::length_width_max(widths.lengths),
                section_list_addr: StoredAddress::new(section_list_addr),
                section_list_used,
                section_list_allocated: section_list_used,
            }
        );
        assert_eq!(
            free_space_manager_header_len(widths),
            header_bytes.len() as u64
        );
        assert_eq!(
            parse_section_info(
                widths,
                &section_list,
                StoredAddress::new(header_addr),
                &header
            ),
            Ok(vec![section(3648, 1600)])
        );
    }

    #[rstest]
    #[case::two_byte_widths(widths(2, 2))]
    #[case::four_byte_widths(widths(4, 4))]
    #[case::mixed_widths(widths(4, 8))]
    #[case::eight_byte_widths(widths(8, 8))]
    fn section_list_capacity_is_independent_of_its_parse_length(#[case] widths: FormatWidths) {
        let sections = vec![section(0x1000, 100)];
        let (fshd, fsse) = serialize_free_space_manager(
            widths,
            &sections,
            StoredAddress::new(0x80),
            StoredAddress::new(0x100),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        let mut header = FreeSpaceManagerHeader::parse(widths, &fshd).unwrap();
        assert_eq!(header.section_list_allocated(), fsse.len().to_u64());
        header.section_list_allocated += 16;
        let header =
            FreeSpaceManagerHeader::parse(widths, &header.encode(widths).unwrap()).unwrap();
        assert_eq!(header.section_list_used(), fsse.len().to_u64());
        assert_eq!(header.section_list_allocated(), fsse.len().to_u64() + 16);
        assert_eq!(
            parse_section_info(widths, &fsse, StoredAddress::new(0x80), &header).unwrap(),
            sections
        );
        assert_eq!(
            parse_section_info(
                widths,
                &fsse[..fsse.len() - 1],
                StoredAddress::new(0x80),
                &header
            )
            .unwrap_err(),
            FormatError::UnexpectedEof {
                expected: fsse.len(),
                available: fsse.len() - 1,
            }
        );
    }

    #[rstest]
    #[case::two_byte_widths(widths(2, 2), 34)]
    #[case::four_byte_widths(widths(4, 4), 50)]
    #[case::four_byte_lengths(widths(8, 4), 54)]
    #[case::four_byte_offsets(widths(4, 8), 78)]
    #[case::eight_byte_widths(widths(8, 8), 82)]
    fn a_header_holds_one_offset_and_seven_length_fields(
        #[case] widths: FormatWidths,
        #[case] expected: u64,
    ) {
        assert_eq!(free_space_manager_header_len(widths), expected);
    }

    #[rstest]
    fn a_manager_round_trips_at_the_widths_of_its_file(
        #[values(2, 4, 8)] offset_size: u8,
        #[values(2, 4, 8)] length_size: u8,
        #[values(&[][..], &[100], &[100, 100, 100], &[10, 20, 30], &[16384, 512, 512])]
        sizes: &[u64],
    ) {
        let widths = widths(offset_size, length_size);
        let sections: Vec<FreeSection> = sizes
            .iter()
            .zip(0u64..)
            .map(|(&size, i)| section(0x1000 + i * 0x4000, size))
            .collect();

        let (fshd, fsse) = serialize_free_space_manager(
            widths,
            &sections,
            StoredAddress::new(1000),
            StoredAddress::new(1100),
            SECTION_CLASS_LARGE,
        )
        .unwrap();

        let header = FreeSpaceManagerHeader::parse(widths, &fshd).unwrap();
        assert_eq!(
            (fshd.len() as u64, fsse.len() as u64),
            (
                free_space_manager_header_len(widths),
                section_info_len(widths, sizes)
            )
        );
        assert_eq!(header.section_list_used, fsse.len() as u64);
        let mut parsed =
            parse_section_info(widths, &fsse, StoredAddress::new(1000), &header).unwrap();
        parsed.sort_by_key(|s| s.addr);
        assert_eq!(parsed, sections);
    }

    // The manager of `four_byte_lengths_header`, with the serializer's maximum section size and
    // each section size at the length width.
    #[test]
    fn a_mixed_width_header_holds_each_field_at_its_width() {
        let (fshd, fsse) = serialize_free_space_manager(
            widths(8, 4),
            &[section(0x0E40, 1600)],
            StoredAddress::new(0x0273),
            StoredAddress::new(0x02A9),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();

        assert_eq!(
            fshd[..fshd.len() - 4],
            [
                0x46, 0x53, 0x48, 0x44, 0x00, 0x01, 0x40, 0x06, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
                0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x50, 0x00, 0x78, 0x00,
                0x3F, 0x00, 0xFD, 0xFF, 0xFF, 0xFF, 0xA9, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                0x1F, 0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00,
            ]
        );
        assert_eq!(
            fsse[..fsse.len() - 4],
            [
                0x46, 0x53, 0x53, 0x45, 0x00, 0x73, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
                0x40, 0x06, 0x00, 0x00, 0x40, 0x0E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ]
        );
    }

    #[rstest]
    #[case::total_space(
        widths(8, 2),
        vec![section(0x1000, 0x7000), section(0x9000, 0x7000), section(0x11000, 0x7000)],
        StoredAddress::new(0x100),
        FormatError::LengthTooLarge { length: 0x1_5000, length_size: 2 },
    )]
    fn a_length_wider_than_its_field_returns_an_error(
        #[case] widths: FormatWidths,
        #[case] sections: Vec<FreeSection>,
        #[case] fsse_addr: StoredAddress,
        #[case] expected: FormatError,
    ) {
        assert_eq!(
            serialize_free_space_manager(
                widths,
                &sections,
                StoredAddress::new(0x80),
                fsse_addr,
                SECTION_CLASS_SIMPLE,
            ),
            Err(expected)
        );
    }

    #[rstest]
    #[case::two_byte_lengths(widths(8, 2), 0x8000)]
    #[case::four_byte_lengths(widths(8, 4), 0x8000_0000)]
    #[case::eight_byte_lengths(widths(8, 8), 0x4000_0000_0000_0000)]
    fn a_section_without_a_bin_returns_an_error(#[case] widths: FormatWidths, #[case] limit: u64) {
        let serialize = |size| {
            serialize_free_space_manager(
                widths,
                &[section(0x100, size)],
                StoredAddress::new(0x80),
                StoredAddress::new(0xC0),
                SECTION_CLASS_SIMPLE,
            )
        };

        let (fshd, _) = serialize(limit - 1).unwrap();
        let header = FreeSpaceManagerHeader::parse(widths, &fshd).unwrap();
        assert_eq!(1 << header.max_section_size.ilog2(), limit);
        assert_eq!(
            serialize(limit),
            Err(FormatError::InvalidFreeSpaceManager(format!(
                "a free section of {limit} bytes is not below the {limit}-byte limit of a manager \
                 at {}-byte lengths",
                widths.lengths.get()
            )))
        );
    }

    #[rstest]
    #[case::two_byte_lengths(widths(8, 2), 0xFFFD)]
    #[case::four_byte_lengths(widths(8, 4), 0xFFFF_FFFD)]
    fn the_written_maximum_section_size_is_below_the_truncated_driver_addresses(
        #[case] widths: FormatWidths,
        #[case] expected: u64,
    ) {
        let (fshd, _) = serialize_free_space_manager(
            widths,
            &[section(0x100, 0x10)],
            StoredAddress::new(0x80),
            StoredAddress::new(0xC0),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();

        assert_eq!(
            FreeSpaceManagerHeader::parse(widths, &fshd)
                .unwrap()
                .max_section_size,
            expected
        );
    }

    // A 63-bit address space is sec2's, and a 32-bit one the core driver's on a 32-bit host.
    #[rstest]
    #[case::all_ones(0xFFFF_FFFF, 63, free_space::four_byte_widths_section_info())]
    #[case::all_ones_less_one(0xFFFF_FFFE, 63, free_space::four_byte_widths_section_info())]
    #[case::four_byte_address_space(
        0xFFFF_FFFE,
        32,
        section_list(widths(4, 4), 0x0217, &[0x01, 0x40, 0x06, 0x00, 0x00, 0x40, 0x0E, 0x00, 0x00, 0x00])
    )]
    fn section_sizes_under_a_truncated_driver_address_are_as_wide_as_section_offsets(
        #[case] max_section_size: u64,
        #[case] addr_space_bits: u8,
        #[case] list: Vec<u8>,
    ) {
        let widths = widths(4, 4);
        let header = FreeSpaceManagerHeader {
            max_section_size,
            addr_space_bits: AddressSpaceBits(addr_space_bits),
            section_list_used: list.len() as u64,
            section_list_allocated: list.len() as u64,
            ..FreeSpaceManagerHeader::parse(widths, &free_space::four_byte_widths_header()).unwrap()
        };

        assert_eq!(
            parse_section_info(widths, &list, StoredAddress::new(0x0217), &header),
            Ok(vec![section(3648, 1600)])
        );
    }

    #[rstest]
    #[case::zero(0)]
    #[case::nonzero(0xd0)]
    fn an_oversized_c_library_section_list_parses_its_declared_sections(#[case] unused: u8) {
        let header = FreeSpaceManagerHeader::parse(
            widths(8, 8),
            &free_space::oversized_section_list_header(),
        )
        .unwrap();

        let mut list = free_space::oversized_section_list();
        list[185] = unused;
        let len = list.len();
        test_util::checksum::restamp(&mut list, 0, len);

        assert_eq!(
            parse_section_info(widths(8, 8), &list, StoredAddress::new(48), &header),
            Ok(vec![
                section(842, 10),
                section(5375, 10),
                section(474, 24),
                section(2444, 27),
                section(1595, 47),
                section(130, 74),
                section(2618, 194),
                section(3793, 254),
                section(1122, 326),
                section(4385, 393),
            ])
        );
    }

    #[rstest]
    #[case::byte_184(184)]
    #[case::byte_185(185)]
    #[case::byte_186(186)]
    #[case::byte_187(187)]
    #[case::byte_188(188)]
    #[case::byte_189(189)]
    #[case::byte_190(190)]
    #[case::byte_191(191)]
    #[case::byte_192(192)]
    fn each_unused_section_list_byte_accepts_every_checksummed_value(
        #[case] at: usize,
        #[values(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
            68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
            90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
            109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
            126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
            143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
            160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,
            177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193,
            194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210,
            211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
            228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244,
            245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
        )]
        value: u8,
    ) {
        let widths = widths(8, 8);
        let header =
            FreeSpaceManagerHeader::parse(widths, &free_space::oversized_section_list_header())
                .unwrap();
        let mut list = free_space::oversized_section_list();
        list[at] = value;
        let len = list.len();
        test_util::checksum::restamp(&mut list, 0, len);

        assert_eq!(
            parse_section_info(widths, &list, StoredAddress::new(48), &header),
            Ok(vec![
                section(842, 10),
                section(5375, 10),
                section(474, 24),
                section(2444, 27),
                section(1595, 47),
                section(130, 74),
                section(2618, 194),
                section(3793, 254),
                section(1122, 326),
                section(4385, 393),
            ]),
        );
    }

    #[cfg(feature = "checksum")]
    #[test]
    fn unused_section_list_bytes_remain_covered_by_the_checksum() {
        let widths = widths(8, 8);
        let header =
            FreeSpaceManagerHeader::parse(widths, &free_space::oversized_section_list_header())
                .unwrap();
        let mut list = free_space::oversized_section_list();
        list[185] = 0xd0;
        let at = list.len() - 4;
        let expected = u32::from_le_bytes(list[at..].try_into().unwrap());
        let computed = checksum::jenkins_lookup3(&list[..at]);
        assert_ne!(expected, computed);

        assert_eq!(
            parse_section_info(widths, &list, StoredAddress::new(48), &header),
            Err(FormatError::ChecksumMismatch { expected, computed }),
        );
    }

    #[rstest]
    #[case::zero(&[0, 0, 0])]
    #[case::nonzero(&[0, 7, 0])]
    fn unused_bytes_between_the_sections_and_the_checksum_are_accepted(#[case] padding: &[u8]) {
        // One set of one section: a 1-byte count, an 8-byte size, an 8-byte offset and a class.
        let payload = [
            &[1][..],
            &100u64.to_le_bytes(),
            &0x1000u64.to_le_bytes(),
            &[SECTION_CLASS_SIMPLE],
            padding,
        ]
        .concat();
        let list = section_list(widths(8, 8), 0x80, &payload);

        assert_eq!(
            parse_section_info(
                widths(8, 8),
                &list,
                StoredAddress::new(0x80),
                &FreeSpaceManagerHeader {
                    total_space: 100,
                    ..header_of(&list, 1)
                }
            ),
            Ok(vec![section(0x1000, 100)])
        );
    }

    #[rstest]
    #[case::a_set(
        2,
        &[0x01, 0x64, 0x00, 0x10, 0x00, 0x00],
        "the section count of a set at byte 13 of the section list runs into its checksum at byte 13"
    )]
    #[case::the_size_of_a_set(
        1,
        &[0x01, 0x64],
        "the section size of a set at byte 8 of the section list runs into its checksum at byte 9"
    )]
    #[case::a_section(
        2,
        &[0x02, 0x64, 0x00, 0x10, 0x00, 0x00],
        "the offset of a section at byte 13 of the section list runs into its checksum at byte 13"
    )]
    #[case::the_class_of_a_section(
        1,
        &[0x01, 0x64, 0x00, 0x10, 0x00],
        "the class of a section at byte 12 of the section list runs into its checksum at byte 12"
    )]
    fn sections_past_the_used_length_fail_to_parse(
        #[case] total_sections: u64,
        #[case] payload: &[u8],
        #[case] reason: &str,
    ) {
        // A set is a 1-byte count and a 2-byte size, and a section a 2-byte offset and a class.
        let widths = widths(2, 2);
        let list = section_list(widths, 0x80, payload);
        let header = FreeSpaceManagerHeader {
            max_section_size: 0xFFFD,
            addr_space_bits: AddressSpaceBits(15),
            ..header_of(&list, total_sections)
        };

        assert_eq!(
            parse_section_info(widths, &list, StoredAddress::new(0x80), &header),
            Err(FormatError::InvalidFreeSpaceManager(reason.to_owned()))
        );
    }

    #[rstest]
    #[case::a_section_list_of_another_header(
        5,
        &736u64.to_le_bytes(),
        "the section list belongs to the header at 736, not to the header at 619"
    )]
    #[case::a_set_of_no_sections(13, &[0], "the set at byte 13 of the section list holds no sections")]
    #[case::a_set_of_empty_sections(
        14,
        &[0; 8],
        "the set at byte 13 of the section list holds sections of 0 bytes"
    )]
    #[case::a_class_the_file_client_does_not_register(
        30,
        &[3],
        "the class of the section at byte 22 of the section list is 3, not one of the 3 classes of \
         the file client"
    )]
    #[case::sections_that_sum_past_the_total_space(
        14,
        &1601u64.to_le_bytes(),
        "the sections sum to 1601 bytes, not the header's total of 1600"
    )]
    fn an_invalid_section_list_fails_to_parse(
        #[case] at: usize,
        #[case] value: &[u8],
        #[case] reason: &str,
    ) {
        let header =
            FreeSpaceManagerHeader::parse(widths(8, 8), &free_space::single_section_header())
                .unwrap();
        let mut fsse = free_space::single_section_info();
        test_util::bytes::set_slice_at(&mut fsse, at, value);
        let len = fsse.len();
        test_util::checksum::restamp(&mut fsse, 0, len);

        assert_eq!(
            parse_section_info(widths(8, 8), &fsse, StoredAddress::new(619), &header),
            Err(FormatError::InvalidFreeSpaceManager(reason.to_owned()))
        );
    }

    #[rstest]
    #[case::header(true)]
    #[case::section_list(false)]
    fn unsupported_manager_versions_report_the_version(#[case] header: bool) {
        let mut bytes = if header {
            free_space::single_section_header()
        } else {
            free_space::single_section_info()
        };
        bytes[4] = 1;
        let length = bytes.len();
        test_util::checksum::restamp(&mut bytes, 0, length);
        if header {
            let reference =
                FreeSpaceManagerHeaderRef::new(widths(8, 8), PhysicalAddress::new(619)).unwrap();
            assert_eq!(
                reference.parse_checked(region(619, &bytes)).unwrap_err(),
                FormatError::UnsupportedVersion(1)
            );
            assert_eq!(
                FreeSpaceManagerHeader::parse(widths(8, 8), &bytes).unwrap_err(),
                FormatError::UnsupportedVersion(1)
            );
        } else {
            let frame = single_frame();
            let reference = frame
                .section_list_ref(StoredAddress::new(619), PhysicalAddress::new(701))
                .unwrap();
            assert_eq!(
                reference.parse_checked(region(701, &bytes)).unwrap_err(),
                FormatError::UnsupportedVersion(1)
            );
            assert_eq!(
                parse_section_info(
                    widths(8, 8),
                    &bytes,
                    StoredAddress::new(619),
                    &frame.file_header().unwrap()
                )
                .unwrap_err(),
                FormatError::UnsupportedVersion(1)
            );
        }
    }

    #[test]
    fn a_set_with_more_sections_than_the_header_counts_fails_to_parse() {
        let widths = widths(2, 2);
        let list = section_list(
            widths,
            0x80,
            &[0x02, 0x64, 0x00, 0x10, 0x00, 0x00, 0x20, 0x00, 0x00],
        );
        let header = FreeSpaceManagerHeader {
            max_section_size: 0xFFFD,
            addr_space_bits: AddressSpaceBits(15),
            ..header_of(&list, 1)
        };

        assert_eq!(
            parse_section_info(widths, &list, StoredAddress::new(0x80), &header),
            Err(FormatError::InvalidFreeSpaceManager(
                "the sets hold more sections than the header's count of 1".to_owned()
            ))
        );
    }

    #[test]
    fn a_section_list_shorter_than_its_used_length_fails_to_parse() {
        let list = free_space::single_section_info();
        let header =
            FreeSpaceManagerHeader::parse(widths(8, 8), &free_space::single_section_header())
                .unwrap();

        assert_eq!(
            parse_section_info(
                widths(8, 8),
                &list[..list.len() - 1],
                StoredAddress::new(619),
                &header
            ),
            Err(FormatError::UnexpectedEof {
                expected: 35,
                available: 34
            })
        );
    }

    #[cfg(feature = "checksum")]
    #[test]
    fn the_checksum_at_the_end_of_the_used_length_is_verified() {
        // The error reports the flipped checksum, and not the zeros appended past the used length.
        let mut list = free_space::oversized_section_list();
        let header = FreeSpaceManagerHeader::parse(
            widths(8, 8),
            &free_space::oversized_section_list_header(),
        )
        .unwrap();
        let stored = list.len() - 4;
        list[stored] ^= 0xFF;
        list.extend_from_slice(&[0; 8]);

        assert_eq!(
            parse_section_info(widths(8, 8), &list, StoredAddress::new(48), &header),
            Err(FormatError::ChecksumMismatch {
                expected: 0xAE5C_5CC0,
                computed: 0xAE5C_5C3F,
            })
        );
    }

    #[rstest]
    #[case::fractal_heap_client(
        vec![(5, vec![0])],
        "the client ID is 0, not 1, the ID of the file client"
    )]
    #[case::unserialized_sections(
        vec![(30, vec![1, 0, 0, 0, 0, 0, 0, 0])],
        "the header's section count is 1, its serialized section count 1 and its unserialized \
         section count 1, and the file client serializes every section"
    )]
    #[case::more_section_classes_than_the_file_client(
        vec![(38, vec![4, 0])],
        "the header registers 4 section classes, more than the 3 of the file client"
    )]
    #[case::empty_address_space(vec![(44, vec![0, 0])], "an address space of 0 bits is not 1 to 64 bits wide")]
    #[case::address_space_wider_than_eight_bytes(
        vec![(44, vec![72, 0])],
        "an address space of 72 bits is not 1 to 64 bits wide"
    )]
    #[case::section_list_used_past_its_allocation(
        vec![(70, vec![34, 0, 0, 0, 0, 0, 0, 0])],
        "the section list uses 35 bytes, more than the 34 bytes allocated to it"
    )]
    #[case::empty_manager_used_past_its_allocation(
        vec![(6, vec![0; 8]), (14, vec![0; 8]), (22, vec![0; 8]), (70, vec![34, 0, 0, 0, 0, 0, 0, 0])],
        "the section list uses 35 bytes, more than the 34 bytes allocated to it"
    )]
    #[case::empty_manager_with_free_bytes(
        vec![(14, vec![0; 8]), (22, vec![0; 8]), (54, vec![0xFF; 8])],
        "the empty manager tracks 1600 bytes"
    )]
    #[case::no_section_list(
        vec![(54, vec![0xFF; 8])],
        "the header's section count is 1, and it stores no section list"
    )]
    #[case::more_sections_than_the_section_list_holds(
        vec![(14, vec![2, 0, 0, 0, 0, 0, 0, 0]), (22, vec![2, 0, 0, 0, 0, 0, 0, 0])],
        "the header's section count is 2, and a section list of 35 bytes holds at most 1"
    )]
    #[case::the_largest_section_count(
        vec![(14, vec![0xFF; 8]), (22, vec![0xFF; 8])],
        "the header's section count is 18446744073709551615, and a section list of 35 bytes holds \
         at most 0"
    )]
    fn an_invalid_header_fails_to_parse(
        #[case] fields: Vec<(usize, Vec<u8>)>,
        #[case] reason: &str,
    ) {
        let mut fshd = free_space::single_section_header();
        for (at, value) in fields {
            test_util::bytes::set_slice_at(&mut fshd, at, &value);
        }
        let len = fshd.len();
        test_util::checksum::restamp(&mut fshd, 0, len);

        assert_eq!(
            FreeSpaceManagerHeader::parse(widths(8, 8), &fshd),
            Err(FormatError::InvalidFreeSpaceManager(reason.to_owned()))
        );
    }

    #[test]
    fn an_empty_manager_header_retains_its_section_list_geometry() {
        let mut fshd = free_space::single_section_header();
        for (at, value) in [
            (6, [0; 8]),
            (14, [0; 8]),
            (22, [0; 8]),
            (54, [0xFF; 8]),
            (62, 17u64.to_le_bytes()),
            (70, 17u64.to_le_bytes()),
        ] {
            test_util::bytes::set_slice_at(&mut fshd, at, &value);
        }
        let len = fshd.len();
        test_util::checksum::restamp(&mut fshd, 0, len);

        assert_eq!(
            FreeSpaceManagerHeader::parse(widths(8, 8), &fshd),
            Ok(FreeSpaceManagerHeader {
                total_space: 0,
                total_sections: 0,
                addr_space_bits: AddressSpaceBits(63),
                max_section_size: FILE_FSM_MAX_SECTION_SIZE,
                section_list_addr: StoredAddress::new(u64::MAX),
                section_list_used: 17,
                section_list_allocated: 17,
            })
        );
    }

    #[rstest]
    #[case::default_allocation(None, 0)]
    #[case::exact_allocation(Some(0), 0)]
    #[case::spare_allocation(Some(9), 9)]
    fn serialization_records_the_requested_allocation(
        #[case] extra: Option<u64>,
        #[case] expected_extra: u64,
    ) {
        let widths = widths(8, 8);
        let sections = [FreeSection {
            addr: StoredAddress::new(4096),
            size: 512,
        }];
        let (_, prefix) = serialize_free_space_manager(
            widths,
            &sections,
            StoredAddress::new(1000),
            StoredAddress::new(1100),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        let used = prefix.len() as u64;
        let (header, list) = serialize_free_space_manager_with_allocation(
            widths,
            &sections,
            StoredAddress::new(1000),
            StoredAddress::new(1100),
            SECTION_CLASS_SIMPLE,
            extra.map(|extra| used + extra),
        )
        .unwrap();
        let header = FreeSpaceManagerHeader::parse(widths, &header).unwrap();
        assert_eq!(header.section_list_used(), used + expected_extra);
        assert_eq!(header.section_list_allocated(), used + expected_extra);
        assert_eq!(list.len() as u64, used + expected_extra);
        let records_end = prefix.len() - CHECKSUM_LEN;
        assert_eq!(&list[..records_end], &prefix[..records_end]);
        assert!(
            list[records_end..list.len() - CHECKSUM_LEN]
                .iter()
                .all(|&byte| byte == 0)
        );
        assert_eq!(
            parse_section_info(widths, &list, StoredAddress::new(1000), &header).unwrap(),
            sections
        );
    }

    #[test]
    fn serialization_rejects_an_allocation_shorter_than_the_section_list() {
        let widths = widths(8, 8);
        let sections = [FreeSection {
            addr: StoredAddress::new(4096),
            size: 512,
        }];
        let (_, list) = serialize_free_space_manager(
            widths,
            &sections,
            StoredAddress::new(1000),
            StoredAddress::new(1100),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        let used = list.len() as u64;
        let allocated = used - 1;
        assert_eq!(
            serialize_free_space_manager_with_allocation(
                widths,
                &sections,
                StoredAddress::new(1000),
                StoredAddress::new(1100),
                SECTION_CLASS_SIMPLE,
                Some(allocated),
            )
            .unwrap_err(),
            FormatError::InvalidFreeSpaceManager(format!(
                "the section list uses {used} bytes, more than the {allocated} bytes allocated to it"
            ))
        );
    }

    #[rstest]
    #[case::two_byte_lengths(2, 1 << 16)]
    #[case::four_byte_lengths(4, 1 << 32)]
    fn serialization_rejects_an_allocation_outside_the_length_width(
        #[case] length_size: u8,
        #[case] allocated: u64,
    ) {
        assert_eq!(
            serialize_free_space_manager_with_allocation(
                widths(8, length_size),
                &[FreeSection {
                    addr: StoredAddress::new(4096),
                    size: 512,
                }],
                StoredAddress::new(1000),
                StoredAddress::new(1100),
                SECTION_CLASS_SIMPLE,
                Some(allocated),
            )
            .unwrap_err(),
            FormatError::LengthTooLarge {
                length: allocated,
                length_size,
            }
        );
    }

    #[cfg(feature = "checksum")]
    #[test]
    fn a_header_whose_checksum_does_not_match_fails_to_parse() {
        let mut fshd = free_space::single_section_header();
        let stored = fshd.len() - 4;
        fshd[stored] ^= 0xFF;

        assert_eq!(
            FreeSpaceManagerHeader::parse(widths(8, 8), &fshd),
            Err(FormatError::ChecksumMismatch {
                expected: 0x1037_1315,
                computed: 0x1037_13EA,
            })
        );
    }

    #[test]
    fn enc_size_matches_reference() {
        assert_eq!(enc_size(0), 1);
        assert_eq!(enc_size(255), 1);
        assert_eq!(enc_size(256), 2);
        assert_eq!(enc_size((1 << 63) - 1), 8);
    }

    #[test]
    fn rejects_bad_signature() {
        let mut fshd = free_space::single_section_header();
        let header = FreeSpaceManagerHeader::parse(widths(8, 8), &fshd).unwrap();
        let mut fsse = free_space::single_section_info();
        fshd[0] = 0x80;
        fsse[0] = b'X';

        assert_eq!(
            FreeSpaceManagerHeader::parse(widths(8, 8), &fshd),
            Err(FormatError::InvalidFreeSpaceManager(
                r#"the header signature is b"\x80SHD", not b"FSHD""#.to_owned()
            ))
        );
        assert_eq!(
            parse_section_info(widths(8, 8), &fsse, StoredAddress::new(619), &header),
            Err(FormatError::InvalidFreeSpaceManager(
                r#"the section list signature is b"XSSE", not b"FSSE""#.to_owned()
            ))
        );
    }

    #[rstest]
    #[case::section_list_address(
        widths(4, 4),
        vec![section(0x1000, 0x100)],
        StoredAddress::new(0x1_0000_0000),
        FormatError::AddressTooLarge { address: 0x1_0000_0000, offset_size: 4 },
    )]
    #[case::section_address(
        widths(2, 4),
        vec![section(0x1_0000, 0x100)],
        StoredAddress::new(0x100),
        FormatError::AddressTooLarge { address: 0x1_0000, offset_size: 2 },
    )]
    fn an_address_wider_than_its_field_returns_an_error(
        #[case] widths: FormatWidths,
        #[case] sections: Vec<FreeSection>,
        #[case] fsse_addr: StoredAddress,
        #[case] expected: FormatError,
    ) {
        assert_eq!(
            serialize_free_space_manager(
                widths,
                &sections,
                StoredAddress::new(0x80),
                fsse_addr,
                SECTION_CLASS_SIMPLE,
            ),
            Err(expected)
        );
    }

    #[test]
    fn sections_whose_sizes_overflow_a_u64_return_an_internal_error() {
        let sections: Vec<FreeSection> = (0..5)
            .map(|i| section(0x1000 * (i + 1), (1 << 62) - 1))
            .collect();

        assert_eq!(
            serialize_free_space_manager(
                widths(8, 8),
                &sections,
                StoredAddress::new(0x80),
                StoredAddress::new(0x100),
                SECTION_CLASS_SIMPLE,
            ),
            Err(FormatError::Internal(
                "the sizes of 5 free sections sum past u64::MAX".to_owned()
            ))
        );
    }

    fn widths(offset_size: u8, length_size: u8) -> FormatWidths {
        FormatWidths::from_sizes(offset_size, length_size).unwrap()
    }

    /// Returns the section list of `payload` for the header at `fshd_addr`, with its prefix and its
    /// checksum.
    fn section_list(widths: FormatWidths, fshd_addr: u64, payload: &[u8]) -> Vec<u8> {
        let mut list = [&FSSE_SIGNATURE[..], &[FSSE_VERSION]].concat();
        bytes::try_write_offset(&mut list, fshd_addr, widths.offsets).unwrap();
        list.extend_from_slice(payload);
        let checksum = checksum::jenkins_lookup3(&list);
        list.extend_from_slice(&checksum.to_le_bytes());
        list
    }

    /// Returns a header of `total_sections` sections whose section list is `list`, with the
    /// address space and the maximum section size of the sec2 driver.
    fn header_of(list: &[u8], total_sections: u64) -> FreeSpaceManagerHeader {
        FreeSpaceManagerHeader {
            total_space: 0,
            total_sections,
            addr_space_bits: AddressSpaceBits(63),
            max_section_size: FILE_FSM_MAX_SECTION_SIZE,
            section_list_addr: StoredAddress::new(0x100),
            section_list_used: list.len() as u64,
            section_list_allocated: list.len() as u64,
        }
    }
    const FRACTAL_HEAP_FSM_CLIENT_ID: u8 = 0;
    const FRACTAL_HEAP_FSM_NUM_CLASSES: u16 = 4;
}
