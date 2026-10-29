//! Verifies Extensible Array metadata and allocation extents before append publication.

use core::mem;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_format::__private::ExtensibleArray;
use hdf5_pure_format::__private::ExtensibleArrayBlockFraming;
use hdf5_pure_format::__private::ExtensibleArrayHeaderLocator;
use hdf5_pure_format::__private::ExtensibleArrayIndexBlock;
use hdf5_pure_format::__private::ExtensibleArrayIndexBlockRef;
use hdf5_pure_format::__private::JenkinsLookup3;

use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::convert::Narrow;
use crate::error::Error;
use crate::error::FormatError;

use super::EA_CHECKSUM_SIZE;
use super::EA_POINTER_BATCH_BYTES;
use super::Located;
use super::Parent;
use super::Store;

/// Reads and parses the header after verifying its checksum, regardless of the `checksum` feature.
///
/// # Errors
///
/// Returns [`FormatError::ChecksumMismatch`] if the stored checksum differs from the checksum
/// computed from the header bytes.
/// Propagates allocation extent, read, and parse errors.
pub(super) fn read_header(
    file: &impl Store,
    address: StoredAddress,
) -> Result<ExtensibleArray, Error> {
    let locator = ExtensibleArrayHeaderLocator::new(file.widths(), BaseAddress::ZERO, address);
    let range = locator.parse_range()?;
    require_range(file, range)?;
    let mut bytes = [0; MAX_HEADER_BYTES];
    let bytes = bytes
        .get_mut(..range.length().to_usize()?)
        .ok_or(FormatError::InvalidExtensibleArrayGeometry)?;
    file.read_at(range.start().get(), bytes)?;
    verify_bytes(bytes)?;
    Ok(locator.parse(FileRegion::new(range, bytes)?)?)
}

/// Checks that a complete allocation fits the published EOF and the physical source.
///
/// # Errors
///
/// Returns [`Error::InvalidSpaceOwnershipExtent`] if the allocation ends past the published EOF,
/// or [`FormatError::UnexpectedSourceEof`] if it ends past the physical source.
pub(super) fn require_range(file: &impl Store, range: FileRange) -> Result<(), Error> {
    if range.end().get() > file.logical_end() {
        return Err(Error::InvalidSpaceOwnershipExtent {
            address: range.start().get(),
            length: range.length(),
            previous_end: 0,
            logical_size: file.logical_end(),
        });
    }
    if range.end().get() > file.len() {
        return Err(FormatError::UnexpectedSourceEof {
            expected: range.end().get(),
            available: file.len(),
        }
        .into());
    }
    Ok(())
}

/// An index block read into a fixed buffer after verifying its checksum and allocation extent.
pub(super) struct IndexBlockBytes {
    reference: ExtensibleArrayIndexBlockRef,
    bytes: [u8; INDEX_READ_BYTES],
    length: usize,
}

impl IndexBlockBytes {
    /// Reads the complete index block and verifies its checksum, regardless of the `checksum` feature.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] if the index block is unallocated. Propagates extent,
    /// read, and checksum errors.
    pub(super) fn read(file: &impl Store, array: &ExtensibleArray) -> Result<Self, Error> {
        let reference = array.index_block().ok_or(Error::AppendUnsupported(
            "the dataset's extensible-array index block is not allocated",
        ))?;
        let range = reference.parse_range();
        require_range(file, range)?;
        let mut bytes = [0; INDEX_READ_BYTES];
        let length = range.length().to_usize()?;
        let supplied = bytes.get_mut(..length).ok_or_else(|| {
            FormatError::Internal("the checked index block exceeds its read buffer".into())
        })?;
        file.read_at(range.start().get(), supplied)?;
        verify_bytes(supplied)?;
        Ok(Self {
            reference,
            bytes,
            length,
        })
    }

    /// Parses the buffered index block's prefix fields and borrows its elements and pointers.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArraySignature`] if the bytes do not start with
    /// `EAIB`. Propagates the other errors from [`ExtensibleArrayIndexBlockRef::parse`].
    pub(super) fn parse(&self) -> Result<ExtensibleArrayIndexBlock<'_>, FormatError> {
        let Self {
            reference,
            bytes,
            length,
        } = self;
        let bytes = bytes.get(..*length).ok_or_else(|| {
            FormatError::Internal("the checked index length exceeds its buffer".into())
        })?;
        reference.parse(FileRegion::new(reference.parse_range(), bytes)?)
    }
}

impl Located {
    /// Verifies existing data blocks and initialized pages touched by slots in `first..end`.
    ///
    /// Rechecks the header and index block. Reads larger checksummed regions in fixed windows
    /// and leaves data blocks outside the append range unread.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ChecksumMismatch`] if a stored checksum differs from the checksum
    /// computed from its append metadata bytes.
    /// Propagates metadata read, parse, extent, and geometry errors.
    pub(super) fn check_append_metadata(
        &self,
        file: &impl Store,
        mut first: u64,
        end: u64,
    ) -> Result<(), Error> {
        let array = read_header(file, self.ea_addr)?;
        let bytes = IndexBlockBytes::read(file, &array)?;
        let index = bytes.parse()?;
        first = first.max(self.idx_blk_elmts);
        while first < end {
            let position = super::locate_data_block(&self.geom, self.idx_blk_elmts, first)?;
            let framing = match position.parent {
                Parent::IndexDirect { .. } => {
                    let mut found = None;
                    for reference in index.data_blocks() {
                        let reference = reference?;
                        if reference.slots()?.contains(&first) {
                            found = Some(reference.framing()?);
                            break;
                        }
                    }
                    found
                }
                Parent::Super { sblk_j, dblk_local } => {
                    let mut found = None;
                    for reference in index.super_blocks() {
                        let reference = reference?;
                        if !reference.slots()?.contains(&first) {
                            continue;
                        }
                        let address = self.super_block_addr(file, sblk_j)?;
                        let offset = self.dblk_ptr_off(
                            Some(address),
                            &position,
                            usize::from(self.offset_width.get()),
                            self.ea_bits.block_offset_width(),
                        )?;
                        let address = file.read_addr_at(offset)?;
                        if !address.is_undefined(self.offset_width) {
                            found = Some(reference.data_block_framing(
                                dblk_local.to_u64(),
                                file.addresses().resolve(address)?,
                            )?);
                        }
                        break;
                    }
                    found
                }
            };
            if let Some(framing) = framing {
                check_block(file, framing)?;
                let page_elements = self.ea_bits.page_nelmts().get();
                if position.dblk_nelmts > page_elements {
                    let Parent::Super { sblk_j, dblk_local } = position.parent else {
                        return Err(FormatError::Internal(
                            "a paged data block has no secondary block".into(),
                        )
                        .into());
                    };
                    let secondary = self.super_block_addr(file, sblk_j)?;
                    let pages = position.dblk_nelmts / page_elements;
                    let first_page = (first - position.db_start) / page_elements;
                    let last_page = (end
                        .min(super::ea_sum([position.db_start, position.dblk_nelmts])?)
                        - position.db_start)
                        .div_ceil(page_elements);
                    let page_length = super::ea_sum([
                        super::ea_product(page_elements, self.ea_elem_size.to_u64())?,
                        EA_CHECKSUM_SIZE,
                    ])?;
                    for page in first_page..last_page {
                        let bit =
                            super::ea_sum([super::ea_product(dblk_local.to_u64(), pages)?, page])?;
                        let bitmap =
                            super::ea_sum([secondary.get(), self.block_prefix_size(), bit / 8])?;
                        let mut byte = [0];
                        file.read_at(bitmap, &mut byte)?;
                        if byte[0] & (0x80 >> (bit % 8)) != 0 {
                            let range = FileRange::new(
                                framing
                                    .parse_range()
                                    .end()
                                    .checked_add(super::ea_product(page, page_length)?)?,
                                page_length,
                            )?;
                            verify_range(file, range)?;
                        }
                    }
                }
            }
            first = super::ea_sum([position.db_start, position.dblk_nelmts])?.min(end);
        }
        Ok(())
    }
}

/// Verifies a block's complete extent, prefix fields, and checksummed region.
///
/// For a paged data block, checks the allocation extent of every page and verifies the prefix's
/// checksum. The caller verifies initialized pages separately.
///
/// # Errors
///
/// Returns [`FormatError::ChecksumMismatch`] if the stored checksum differs from the checksum
/// computed from the covered block bytes. Propagates metadata read, parse, and extent errors.
pub(super) fn check_block(
    file: &impl Store,
    framing: ExtensibleArrayBlockFraming,
) -> Result<(), Error> {
    require_range(file, framing.range())?;
    let prefix = framing.prefix_range()?;
    let mut bytes = [0; MAX_BLOCK_PREFIX_BYTES];
    let bytes = bytes
        .get_mut(..prefix.length().to_usize()?)
        .ok_or(FormatError::InvalidExtensibleArrayGeometry)?;
    file.read_at(prefix.start().get(), bytes)?;
    framing.parse_prefix(FileRegion::new(prefix, bytes)?)?;
    verify_range(file, framing.parse_range())
}

fn verify_range(file: &impl Store, range: FileRange) -> Result<(), Error> {
    require_range(file, range)?;
    let content = range
        .length()
        .checked_sub(EA_CHECKSUM_SIZE)
        .ok_or(FormatError::InvalidExtensibleArrayGeometry)?;
    let mut checksum = JenkinsLookup3::new(content);
    let mut bytes = [0u8; EA_POINTER_BATCH_BYTES];
    let mut position = range.start().get();
    let end = range.start().checked_add(content)?.get();
    while position < end {
        let take = (end - position).min(bytes.len().to_u64()).to_usize()?;
        file.read_at(position, &mut bytes[..take])?;
        checksum.update(&bytes[..take])?;
        position = super::ea_sum([position, take.to_u64()])?;
    }
    let mut stored = [0u8; CHECKSUM_BYTES];
    file.read_at(end, &mut stored)?;
    compare_checksum(u32::from_le_bytes(stored), checksum.finish()?)
}

fn verify_bytes(bytes: &[u8]) -> Result<(), Error> {
    let (content, stored) = bytes
        .split_last_chunk::<CHECKSUM_BYTES>()
        .ok_or(FormatError::InvalidExtensibleArrayGeometry)?;
    compare_checksum(
        u32::from_le_bytes(*stored),
        crate::checksum::jenkins_lookup3(content),
    )
}

fn compare_checksum(expected: u32, computed: u32) -> Result<(), Error> {
    if expected != computed {
        return Err(FormatError::ChecksumMismatch { expected, computed }.into());
    }
    Ok(())
}

const INDEX_READ_BYTES: usize = 8192;
// Section `subsec_fmt4_appendixc_extarr`, version 4.0, defines the checksum width.
const CHECKSUM_BYTES: usize = mem::size_of::<u32>();
// The header fields use at most eight-byte addresses and lengths, as section
// `subsec_fmt4_appendixc_extarr`, version 4.0, defines them.
const MAX_HEADER_BYTES: usize = 12 + 6 * 8 + 8 + CHECKSUM_BYTES;
// The block prefix has a signature, version, client, owner and block offset in section
// `subsec_fmt4_appendixc_extarr`, version 4.0.
const MAX_BLOCK_PREFIX_BYTES: usize = 6 + 8 + 8;
