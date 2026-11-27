use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;

use super::VERSION;
use super::data_block::DATA_BLOCK_SIGNATURE;
use super::geometry::{DataBlock, Layout};
use super::header::ExtensibleArrayHeader;
use super::secondary_block::{ExtensibleArraySuperBlockRef, SUPER_BLOCK_SIGNATURE};
use crate::address::{BaseAddress, BaseAddressExt, DecodedStoredAddress, StoredAddress};
use crate::bytes;
use crate::error::FormatError;

/// The expected prefix fields and physical ranges of a version 0 secondary or data block.
///
/// A caller can parse the prefix separately from reading and verifying the checksummed region.
/// For a paged data block, that region ends before the pages, while the allocation includes every
/// page. The block layouts are defined in "The Extensible Array Index" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayBlockFraming {
    owner: Owner,
    range: FileRange,
    parse_range: FileRange,
    signature: [u8; 4],
    offset: (u64, Option<u64>),
}

impl ExtensibleArrayBlockFraming {
    pub(super) fn secondary(block: &ExtensibleArraySuperBlockRef) -> Self {
        Self {
            owner: block.owner(),
            range: block.parse_range(),
            parse_range: block.parse_range(),
            signature: SUPER_BLOCK_SIGNATURE,
            offset: (block.offset(), None),
        }
    }

    pub(super) fn data(
        owner: Owner,
        address: PhysicalAddress,
        block: DataBlock,
    ) -> Result<Self, FormatError> {
        let layout = owner.layout();
        let range = FileRange::new(address, layout.data_block_len(block)?)?;
        Ok(Self {
            owner,
            range,
            parse_range: if layout.pages(block.elements()).is_some() {
                FileRange::new(address, layout.pages_offset())?
            } else {
                range
            },
            signature: DATA_BLOCK_SIGNATURE,
            offset: (block.offset(), block.libhdf5_offset()),
        })
    }

    /// Parses the signature, version, client ID, header address, and block offset in `region`.
    ///
    /// The caller verifies the checksum over [`parse_range`](Self::parse_range) separately.
    /// For a direct data block, accepts either the block's position or the alternate offset
    /// described by [`super::ExtensibleArrayDataBlockRef::parse`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`prefix_range`](Self::prefix_range), [`FormatError::InvalidExtensibleArraySignature`] if
    /// the signature differs from the expected block kind, or
    /// [`FormatError::InvalidExtensibleArrayVersion`] if the version is not 0. Returns
    /// [`FormatError::InvalidExtensibleArrayBlockClient`] if the client differs from the header's,
    /// [`FormatError::InvalidExtensibleArrayBlockOwner`] if the header address differs from the
    /// array's, or [`FormatError::InvalidExtensibleArrayBlockOffset`] if the block offset is
    /// outside the accepted offsets. Returns
    /// [`FormatError::OffsetOverflow`] if the base address plus the stored header address
    /// exceeds [`u64::MAX`].
    pub fn parse_prefix(&self, region: FileRegion<'_>) -> Result<(), FormatError> {
        let Self {
            owner,
            range: _,
            parse_range: _,
            signature,
            offset,
        } = *self;
        let prefix = region.subregion(self.prefix_range()?)?;
        let mut cursor = owner.parse_prefix(prefix.bytes(), signature)?;
        let (expected, alternate) = offset;
        owner.parse_block_offset(&mut cursor, expected, alternate)?;
        Ok(())
    }

    /// Returns the prefix range through the block offset, excluding the checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the end exceeds [`u64::MAX`].
    pub fn prefix_range(&self) -> Result<FileRange, FormatError> {
        FileRange::new(
            self.range.start(),
            self.owner.layout().prefix_len() + u64::from(self.owner.layout().block_offset_len()),
        )
    }

    /// Returns the checksummed region, including its trailing checksum.
    ///
    /// For a paged data block, the region excludes the separately checksummed pages.
    pub const fn parse_range(&self) -> FileRange {
        self.parse_range
    }

    /// Returns the complete allocation, including every page of a paged data block.
    pub const fn range(&self) -> FileRange {
        self.range
    }
}

/// The header address, layout, and slot limit used to interpret blocks of one Extensible Array.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Owner {
    base: BaseAddress,
    /// The physical address of the header, the address a block's header address resolves to.
    header: PhysicalAddress,
    layout: Layout,
    /// The header's "Max Index Set", one more than the highest slot the array has set.
    max_index_set: u64,
}

impl Owner {
    pub(super) fn from_header(
        base: BaseAddress,
        address: PhysicalAddress,
        header: &ExtensibleArrayHeader,
    ) -> Self {
        Self {
            base,
            header: address,
            layout: header.layout(),
            max_index_set: header.stored_statistics().max_idx_set,
        }
    }

    pub(super) fn layout(self) -> Layout {
        self.layout
    }
    pub(super) fn max_index_set(self) -> u64 {
        self.max_index_set
    }

    pub(super) fn resolve(self, address: StoredAddress) -> Result<PhysicalAddress, FormatError> {
        self.base.resolve(address)
    }

    pub(super) fn addresses<'a>(
        self,
        addresses: &'a [u8],
    ) -> impl Iterator<Item = Result<Option<PhysicalAddress>, FormatError>> + 'a {
        let offsets = self.layout.encoding().offsets();
        addresses
            .chunks_exact(usize::from(offsets.get()))
            .map(move |field| {
                Ok(
                    match bytes::ByteCursor::new(field).decoded_address(offsets)? {
                        DecodedStoredAddress::Defined(address) => Some(self.resolve(address)?),
                        DecodedStoredAddress::Undefined => None,
                    },
                )
            })
    }

    /// Parses the signature, the version, the client ID, and the header address of a block, and
    /// returns a cursor after those fields.
    pub(super) fn parse_prefix<'a, const N: usize>(
        self,
        bytes: &'a [u8],
        signature: [u8; N],
    ) -> Result<bytes::ByteCursor<'a>, FormatError> {
        let mut cursor = bytes::ByteCursor::new(bytes);
        if cursor.array::<N>()? != signature {
            return Err(FormatError::InvalidExtensibleArraySignature);
        }
        let version = cursor.u8()?;
        if version != VERSION {
            return Err(FormatError::InvalidExtensibleArrayVersion(version));
        }
        let client_id = cursor.u8()?;
        let expected = self.layout.encoding().client().id();
        if client_id != expected {
            return Err(FormatError::InvalidExtensibleArrayBlockClient {
                expected,
                actual: client_id,
            });
        }
        let owner = self.resolve(cursor.address(self.layout.encoding().offsets())?)?;
        if owner != self.header {
            return Err(FormatError::InvalidExtensibleArrayBlockOwner {
                expected: self.header.get(),
                actual: owner.get(),
            });
        }
        Ok(cursor)
    }

    pub(super) fn holds_set_slots(self, offset: u64) -> bool {
        self.layout
            .first_slot(offset)
            .is_ok_and(|slot| slot < self.max_index_set)
    }

    /// Parses a block offset and checks that it is `expected` or, where the caller passes one,
    /// `libhdf5`.
    pub(super) fn parse_block_offset(
        self,
        cursor: &mut bytes::ByteCursor<'_>,
        expected: u64,
        libhdf5: Option<u64>,
    ) -> Result<(), FormatError> {
        let actual = cursor.uint(usize::from(self.layout.block_offset_len()))?;
        if actual != expected && Some(actual) != libhdf5 {
            return Err(FormatError::InvalidExtensibleArrayBlockOffset { expected, actual });
        }
        Ok(())
    }
}

/// Returns the bytes of a block from `pos` to its checksum.
pub(super) fn body(bytes: &[u8], pos: usize) -> Result<&[u8], FormatError> {
    bytes
        .split_last_chunk::<{ size_of::<u32>() }>()
        .and_then(|(body, _checksum)| body.get(pos..))
        .ok_or_else(|| super::internal("block ends inside its prefix or its checksum"))
}
