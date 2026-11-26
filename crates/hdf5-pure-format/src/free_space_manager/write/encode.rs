use core::num::NonZeroU64;

use crate::address::StoredAddress;
use crate::bytes::UnexpectedEof;
use crate::checksum;
use crate::codec::Encode;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::free_space_manager::address_space::AddressSpaceBits;
use crate::free_space_manager::file_header::{
    FILE_FSM_CLIENT_ID, FILE_FSM_EXPAND_PCT, FILE_FSM_SHRINK_PCT,
};
use crate::free_space_manager::file_section::FILE_FSM_NUM_CLASSES;
use crate::free_space_manager::header::{self, CHECKSUM_LEN, FSHD_SIGNATURE, FSHD_VERSION};
use crate::free_space_manager::section_list::{FSSE_SIGNATURE, FSSE_VERSION};
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

use super::placement::Placement;
use super::prepared::PreparedFreeSpaceManager;

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
    pub(super) fn new(placement: Placement<'a>) -> Self {
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
    pub const MAX_ENCODED_SIZE: usize = header::header_len(FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    });
}

impl Encode for FreeSpaceManagerHeaderEncoding {
    type Error = UnexpectedEof;

    fn encoded_len(&self) -> usize {
        header::header_len(self.widths)
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
