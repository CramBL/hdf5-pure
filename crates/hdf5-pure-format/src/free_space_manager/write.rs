use alloc::collections::BTreeMap;
use alloc::collections::btree_map::Entry;
use alloc::vec::Vec;
use core::num::NonZeroU64;
use core::slice;

use crate::address::StoredAddress;
use crate::bytes;
use crate::bytes::UnexpectedEof;
use crate::checksum;
use crate::codec::Encode;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

use super::address_space::AddressSpaceBits;
use super::file_header::{
    FILE_FSM_CLIENT_ID, FILE_FSM_EXPAND_PCT, FILE_FSM_NUM_CLASSES, FILE_FSM_SHRINK_PCT,
};
use super::header::{FSHD_SIGNATURE, FSHD_VERSION};
use super::section_list::{FSSE_SIGNATURE, FSSE_VERSION};
use super::{CHECKSUM_LEN, FreeSection, SectionFields, SectionInfoCounts};

/// The checked sections and block lengths of a file free-space manager before placement.
///
/// Groups section addresses by size for reuse across calls to [`Self::place`]. All sections
/// belong to the same file section class.
#[derive(Debug)]
pub struct PreparedFreeSpaceManager {
    widths: FormatWidths,
    class_id: u8,
    groups: BTreeMap<NonZeroU64, SectionAddresses>,
    section_count: u64,
    fields: SectionFields,
    total_space: u64,
    max_section_size: NonZeroU64,
    section_list_len: usize,
}

impl PreparedFreeSpaceManager {
    /// Checks the section records at the file's widths and groups their addresses by size.
    ///
    /// Pass a file section class, [`SECTION_CLASS_SIMPLE`](super::SECTION_CLASS_SIMPLE),
    /// [`SECTION_CLASS_SMALL`](super::SECTION_CLASS_SMALL), or
    /// [`SECTION_CLASS_LARGE`](super::SECTION_CLASS_LARGE).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedSerializedSectionClass`] for any other class, or
    /// [`FormatError::InvalidFreeSpaceSectionSize`] if a section size is zero or exceeds the
    /// maximum for the length width.
    ///
    /// Returns [`FormatError::AddressTooLarge`] or [`FormatError::DefinedAddressIsUndefined`]
    /// if a section address is outside the offset width or equals its undefined address,
    /// respectively. Returns [`FormatError::FreeSpaceSectionSizeOverflow`] if the total size
    /// exceeds [`u64::MAX`].
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the section-list length exceeds [`u64::MAX`],
    /// [`FormatError::LengthTooLarge`] if the total size or section-list length exceeds the
    /// length width, or [`FormatError::ValueTooLargeForPlatform`] if the section list is too
    /// long for a slice on this platform.
    pub fn new(
        widths: FormatWidths,
        sections: impl IntoIterator<Item = FreeSection>,
        class_id: u8,
    ) -> Result<Self, FormatError> {
        // libhdf5 defines the simple, small, and large file classes in `H5MFpkg.h` and sets
        // their extra serialized size to zero in `H5MFsection.c` (HDF5 1.14.6).
        if u16::from(class_id) >= FILE_FSM_NUM_CLASSES {
            return Err(FormatError::UnsupportedSerializedSectionClass {
                client_id: FILE_FSM_CLIENT_ID,
                class_id,
            });
        }
        let max_section_size = super::written_max_section_size(widths.lengths);
        let maximum = (1 << max_section_size.ilog2()) - 1;
        let mut groups = BTreeMap::<NonZeroU64, SectionAddresses>::new();
        let mut total_space = 0u64;
        let mut section_count = 0u64;
        for FreeSection { addr, size } in sections {
            let size = NonZeroU64::new(size)
                .filter(|size| size.get() <= maximum)
                .ok_or(FormatError::InvalidFreeSpaceSectionSize { size, maximum })?;
            bytes::require_defined_offset(addr.get(), widths.offsets)?;
            total_space = total_space.checked_add(size.get()).ok_or(
                FormatError::FreeSpaceSectionSizeOverflow {
                    total: total_space,
                    size: size.get(),
                },
            )?;
            // Every section adds at least one byte to the checked sum.
            section_count += 1;
            match groups.entry(size) {
                Entry::Occupied(mut entry) => entry.get_mut().push(addr),
                Entry::Vacant(entry) => {
                    entry.insert(SectionAddresses::Single(addr));
                }
            }
        }
        require_length(total_space, widths.lengths)?;

        for addresses in groups.values_mut() {
            if let SectionAddresses::Multiple(addresses) = addresses {
                addresses.sort_unstable();
            }
        }
        let (fields, section_list_len) =
            SectionInfoCounts::new(section_count, groups.len().to_u64())?.layout(widths)?;
        let section_list_len = slice_len(section_list_len, widths.lengths)?;
        Ok(Self {
            widths,
            class_id,
            groups,
            section_count,
            fields,
            total_space,
            max_section_size,
            section_list_len,
        })
    }

    /// Returns the sections in ascending order of size, then of address within each size.
    pub fn sections(&self) -> impl Iterator<Item = FreeSection> + '_ {
        self.groups.iter().flat_map(|(&size, addresses)| {
            addresses.as_slice().iter().map(move |&addr| FreeSection {
                addr,
                size: size.get(),
            })
        })
    }

    /// Returns the number of size groups in the section list.
    pub fn distinct_size_count(&self) -> usize {
        self.groups.len()
    }

    /// Returns whether any section has a length of `size` bytes.
    pub fn contains_section_size(&self, size: u64) -> bool {
        NonZeroU64::new(size).is_some_and(|size| self.groups.contains_key(&size))
    }

    /// Returns the number of sections, all of which are serialized.
    pub fn section_count(&self) -> u64 {
        self.section_count
    }

    /// Returns the header length in bytes, including its checksum.
    pub fn header_len(&self) -> usize {
        super::header::header_len(self.widths)
    }

    /// Returns the section-list length in bytes, including its checksum and excluding padding.
    pub fn section_list_len(&self) -> usize {
        self.section_list_len
    }

    fn widths(&self) -> FormatWidths {
        self.widths
    }

    fn total_space(&self) -> u64 {
        self.total_space
    }

    fn max_section_size(&self) -> NonZeroU64 {
        self.max_section_size
    }

    fn fields(&self) -> SectionFields {
        self.fields
    }

    fn class_id(&self) -> u8 {
        self.class_id
    }

    fn groups(&self) -> impl Iterator<Item = (NonZeroU64, &[StoredAddress])> {
        self.groups
            .iter()
            .map(|(&size, addresses)| (size, addresses.as_slice()))
    }
}

impl PreparedFreeSpaceManager {
    /// Prepares each block to store the other block's address.
    ///
    /// Pass the two addresses relative to the file's base address. A nonempty manager requires
    /// two defined addresses. For an empty manager, either address may be the undefined address
    /// at the file's offset width or [`u64::MAX`]. The writer encodes these as the undefined
    /// address at the file's offset width.
    ///
    /// Pass `None` for the shortest section list, or `Some` with the total list length in bytes,
    /// including its checksum. The writer pads the list with zero bytes before the checksum.
    /// The header stores the total length in both the used and allocated length fields.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::__private::StoredAddressExt;
    /// use hdf5_pure_core::StoredAddress;
    /// use hdf5_pure_format::__private::{
    ///     FormatWidths, FreeSection, FreeSpaceManagerHeader, PreparedFreeSpaceManager,
    ///     SECTION_CLASS_SIMPLE, parse_section_info,
    /// };
    ///
    /// let widths = FormatWidths::from_sizes(8, 4)?;
    /// let sections = [FreeSection { addr: StoredAddress::new(256), size: 32 }];
    /// let plan = PreparedFreeSpaceManager::new(widths, sections, SECTION_CLASS_SIMPLE)?;
    /// let header_address = StoredAddress::new(512);
    /// let list_address = StoredAddress::new(1024);
    /// let allocation = plan.section_list_len() as u64 + 16;
    /// let manager = plan.place(header_address, list_address, Some(allocation))?;
    ///
    /// let header = manager.header();
    /// let mut header_bytes = vec![0; header.encoded_size()];
    /// assert_eq!(header.write_into(&mut header_bytes)?, header_bytes.len());
    ///
    /// let list = manager.section_list();
    /// let mut list_bytes = vec![0; list.encoded_size()];
    /// assert_eq!(list.write_into(&mut list_bytes)?, list_bytes.len());
    /// assert_eq!(list_bytes.len() as u64, allocation);
    ///
    /// let header = FreeSpaceManagerHeader::parse(widths, &header_bytes)?;
    /// assert_eq!(header.section_list_addr(), list_address);
    /// assert_eq!(header.section_list_used(), allocation);
    /// assert_eq!(parse_section_info(widths, &list_bytes, header_address, &header)?, sections);
    /// # Ok::<(), hdf5_pure_format::FormatError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressTooLarge`] if a block address exceeds the offset width, or
    /// [`FormatError::DefinedAddressIsUndefined`] if a block address equals the undefined address
    /// for a nonempty manager.
    ///
    /// Returns [`FormatError::FreeSpaceSectionListAllocationTooSmall`] if `allocation`
    /// is less than [`Self::section_list_len`], [`FormatError::LengthTooLarge`] if the length
    /// exceeds the file's length width, or [`FormatError::ValueTooLargeForPlatform`] if the
    /// section list is too long for a slice on this platform.
    pub fn place(
        &self,
        header_address: StoredAddress,
        section_list_address: StoredAddress,
        allocation: Option<u64>,
    ) -> Result<FreeSpaceManagerEncoding<'_>, FormatError> {
        let FormatWidths { offsets, lengths } = self.widths();
        let header_address = self.block_address(header_address, offsets)?;
        let section_list_address = self.block_address(section_list_address, offsets)?;
        let required = self.section_list_len().to_u64();
        let allocated = allocation.unwrap_or(required);
        if allocated < required {
            return Err(FormatError::FreeSpaceSectionListAllocationTooSmall {
                required,
                allocated,
            });
        }
        let allocation_len = slice_len(allocated, lengths)?;
        Ok(FreeSpaceManagerEncoding::new(Placement {
            plan: self,
            header_address,
            section_list_address,
            allocation_len,
        }))
    }

    fn block_address(
        &self,
        address: StoredAddress,
        offsets: OffsetWidth,
    ) -> Result<StoredAddress, FormatError> {
        if self.section_count() == 0 && (address.is_undefined(offsets) || address.get() == u64::MAX)
        {
            return Ok(StoredAddress::undefined(offsets));
        }
        bytes::require_defined_offset(address.get(), offsets)?;
        Ok(address)
    }
}

struct Placement<'a> {
    plan: &'a PreparedFreeSpaceManager,
    header_address: StoredAddress,
    section_list_address: StoredAddress,
    allocation_len: usize,
}

impl<'a> Placement<'a> {
    fn plan(&self) -> &'a PreparedFreeSpaceManager {
        self.plan
    }

    fn header_address(&self) -> StoredAddress {
        self.header_address
    }

    fn section_list_address(&self) -> StoredAddress {
        self.section_list_address
    }

    fn allocation_len(&self) -> usize {
        self.allocation_len
    }
}

/// A file free-space manager with checked block addresses and section-list allocation.
///
/// The version 0 header (`FSHD`) and section list (`FSSE`) can be written independently. Their
/// layouts are defined in "Free-space Index" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex
#[derive(Debug)]
pub struct FreeSpaceManagerEncoding<'a> {
    header: FreeSpaceManagerHeaderEncoding,
    section_list: FreeSpaceManagerSectionListEncoding<'a>,
}

impl<'a> FreeSpaceManagerEncoding<'a> {
    fn new(placement: Placement<'a>) -> Self {
        let plan = placement.plan();
        Self {
            header: FreeSpaceManagerHeaderEncoding {
                widths: plan.widths(),
                total_space: plan.total_space(),
                total_sections: plan.section_count(),
                address_space: AddressSpaceBits::written(plan.widths().offsets),
                max_section_size: plan.max_section_size(),
                section_list_address: placement.section_list_address(),
                section_list_len: placement.allocation_len().to_u64(),
            },
            section_list: FreeSpaceManagerSectionListEncoding {
                plan,
                header_address: placement.header_address(),
                allocation_len: placement.allocation_len(),
            },
        }
    }

    /// Returns the header with the stored address and length of the section list.
    pub fn header(&self) -> FreeSpaceManagerHeaderEncoding {
        self.header
    }

    /// Returns the section list with the stored address of the header.
    pub fn section_list(&self) -> FreeSpaceManagerSectionListEncoding<'a> {
        self.section_list
    }
}

/// A version 0 file free-space manager header prepared for encoding.
#[derive(Clone, Copy, Debug)]
pub struct FreeSpaceManagerHeaderEncoding {
    widths: FormatWidths,
    total_space: u64,
    total_sections: u64,
    address_space: AddressSpaceBits,
    max_section_size: NonZeroU64,
    section_list_address: StoredAddress,
    section_list_len: u64,
}

impl FreeSpaceManagerHeaderEncoding {
    /// Returns the number of bytes needed for the header and its checksum.
    pub fn encoded_size(&self) -> usize {
        Encode::encoded_len(self)
    }

    /// Writes the header and its checksum to the beginning of `output`.
    ///
    /// Returns the number of bytes written. Leaves any trailing bytes unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] and leaves `output` unchanged if it is shorter
    /// than [`Self::encoded_size`].
    pub fn write_into(&self, output: &mut [u8]) -> Result<usize, FormatError> {
        Ok(self.encode_into(output)?)
    }

    fn write_body(&self, body: &mut [u8]) {
        let Self {
            widths: FormatWidths { offsets, lengths },
            total_space,
            total_sections,
            address_space,
            max_section_size,
            section_list_address,
            section_list_len,
        } = *self;
        let length_width = usize::from(lengths.get());
        let mut output = BlockOutput { remaining: body };
        output.write_bytes(FSHD_SIGNATURE);
        output.write_bytes(&[FSHD_VERSION, FILE_FSM_CLIENT_ID]);
        for length in [total_space, total_sections, total_sections, 0] {
            output.write_uint_le(length, length_width);
        }
        for field in [
            FILE_FSM_NUM_CLASSES,
            FILE_FSM_SHRINK_PCT,
            FILE_FSM_EXPAND_PCT,
            address_space.get(),
        ] {
            output.write_bytes(&field.to_le_bytes());
        }
        output.write_uint_le(max_section_size.get(), length_width);
        output.write_uint_le(section_list_address.get(), usize::from(offsets.get()));
        output.write_uint_le(section_list_len, length_width);
        output.write_uint_le(section_list_len, length_width);
        debug_assert!(output.remaining.is_empty());
    }

    /// The header length in bytes at eight-byte offset and length widths, including the checksum.
    pub const MAX_ENCODED_SIZE: usize = super::header::header_len(FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    });
}

impl Encode for FreeSpaceManagerHeaderEncoding {
    type Error = UnexpectedEof;

    fn encoded_len(&self) -> usize {
        super::header::header_len(self.widths)
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        write_checksummed(output, self.encoded_len(), |body| self.write_body(body))
    }
}

/// A version 0 file free-space section list prepared for encoding.
///
/// Borrows the groups of sections from a [`PreparedFreeSpaceManager`].
#[derive(Clone, Copy, Debug)]
pub struct FreeSpaceManagerSectionListEncoding<'a> {
    plan: &'a PreparedFreeSpaceManager,
    header_address: StoredAddress,
    allocation_len: usize,
}

impl FreeSpaceManagerSectionListEncoding<'_> {
    /// Returns the number of bytes needed for the section list, padding, and checksum.
    pub fn encoded_size(&self) -> usize {
        Encode::encoded_len(self)
    }

    /// Writes the section list, padding, and checksum to the beginning of `output`.
    ///
    /// Returns the number of bytes written. Leaves any trailing bytes unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] and leaves `output` unchanged if it is shorter
    /// than [`Self::encoded_size`].
    pub fn write_into(&self, output: &mut [u8]) -> Result<usize, FormatError> {
        Ok(self.encode_into(output)?)
    }

    fn write_body(&self, body: &mut [u8]) {
        let Self {
            plan,
            header_address,
            allocation_len: _,
        } = *self;
        let fields = plan.fields();
        let mut output = BlockOutput { remaining: body };
        output.write_bytes(FSSE_SIGNATURE);
        output.write_bytes(&[FSSE_VERSION]);
        output.write_uint_le(
            header_address.get(),
            usize::from(plan.widths().offsets.get()),
        );
        for (section_size, addresses) in plan.groups() {
            output.write_uint_le(addresses.len().to_u64(), fields.count());
            output.write_uint_le(section_size.get(), fields.size());
            for address in addresses {
                output.write_uint_le(address.get(), fields.offset());
                output.write_bytes(&[plan.class_id()]);
            }
        }
        // `H5FS__cache_sinfo_serialize` writes zero padding before the checksum (`H5FScache.c`,
        // HDF5 2.2.0).
        output.remaining.fill(0);
    }
}

impl Encode for FreeSpaceManagerSectionListEncoding<'_> {
    type Error = UnexpectedEof;

    fn encoded_len(&self) -> usize {
        self.allocation_len
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        write_checksummed(output, self.encoded_len(), |body| self.write_body(body))
    }
}

/// The addresses of sections of one size, with inline storage for a single address.
#[derive(Debug)]
enum SectionAddresses {
    Multiple(Vec<StoredAddress>),
    Single(StoredAddress),
}

impl SectionAddresses {
    fn push(&mut self, address: StoredAddress) {
        match self {
            Self::Multiple(addresses) => addresses.push(address),
            Self::Single(first) => *self = Self::Multiple(Vec::from([*first, address])),
        }
    }

    fn as_slice(&self) -> &[StoredAddress] {
        match self {
            Self::Multiple(addresses) => addresses,
            Self::Single(address) => slice::from_ref(address),
        }
    }
}

/// Reserves a buffer for the block's complete encoding, then writes the block into it.
///
/// # Errors
///
/// Returns [`FormatError::AllocationFailed`] if the reservation fails. Propagates errors
/// from [`Encode::encode_into`].
pub(super) fn encode_to_vec(
    block: impl Encode<Error = UnexpectedEof>,
) -> Result<Vec<u8>, FormatError> {
    let length = block.encoded_len();
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|source| FormatError::AllocationFailed {
            values: length,
            source,
        })?;
    bytes.resize(length, 0);
    block.encode_into(&mut bytes)?;
    Ok(bytes)
}

fn require_length(value: u64, width: LengthWidth) -> Result<(), FormatError> {
    if !width::length_width_holds(width, value) {
        return Err(FormatError::LengthTooLarge {
            length: value,
            length_size: width.get(),
        });
    }
    Ok(())
}

/// Converts `value` to a byte length representable by a slice on this platform.
///
/// # Errors
///
/// Returns [`FormatError::ValueTooLargeForPlatform`] if `value` exceeds [`isize::MAX`].
fn slice_len(value: u64, width: LengthWidth) -> Result<usize, FormatError> {
    require_length(value, width)?;
    let len = value.to_usize()?;
    if len > isize::MAX.cast_unsigned() {
        return Err(FormatError::ValueTooLargeForPlatform {
            value,
            target: "slice length",
        });
    }
    Ok(len)
}

/// Writes a block of `length` bytes at the start of `output` and returns that length.
///
/// Calls `write_body` with the part of the block that precedes the checksum. Writes the
/// checksum after the callback returns and leaves any bytes after the block unchanged.
///
/// # Errors
///
/// Returns [`UnexpectedEof`] if `output` is shorter than `length`, leaving it unchanged
/// and skipping the callback.
///
/// # Panics
///
/// Panics if `length` is smaller than [`CHECKSUM_LEN`] and `output` has at least `length` bytes.
fn write_checksummed(
    output: &mut [u8],
    length: usize,
    write_body: impl FnOnce(&mut [u8]),
) -> Result<usize, UnexpectedEof> {
    let available = output.len();
    let output = output
        .get_mut(..length)
        .ok_or_else(|| UnexpectedEof::new(length, available))?;
    let (body, stored) = output.split_at_mut(length - CHECKSUM_LEN);
    write_body(body);
    stored.copy_from_slice(&checksum::jenkins_lookup3(body).to_le_bytes());
    Ok(length)
}

struct BlockOutput<'a> {
    remaining: &'a mut [u8],
}

impl BlockOutput<'_> {
    fn write_bytes(&mut self, bytes: &[u8]) {
        let (field, rest) = core::mem::take(&mut self.remaining).split_at_mut(bytes.len());
        field.copy_from_slice(bytes);
        self.remaining = rest;
    }

    fn write_uint_le(&mut self, value: u64, width: usize) {
        self.write_bytes(&value.to_le_bytes()[..width]);
    }
}
