//! Caller-supplied regions for version 0 fractal-heap direct and indirect blocks.
//!
//! The caller resolves each stored address to a physical position before supplying a region.
//! Block prefixes and doubling-table geometry are defined in "Fractal Heap" of the
//! [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap

use hdf5_pure_core::__private::{DecodedStoredAddress, FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;

use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum;
use crate::convert;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::filter_pipeline::FilterPipeline;
use crate::fractal_heap::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED;
use crate::fractal_heap::FractalHeapFiltering;
use crate::fractal_heap::FractalHeapHeader;
use crate::fractal_heap::FractalHeapLayout;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// Describes managed blocks using a heap header's doubling table and filter metadata.
///
/// The caller obtains each block's stored address, expected heap offset, and size, then
/// converts the address to a physical position before parsing a [`FileRegion`].
///
/// # Examples
///
/// ```
/// # use hdf5_pure_core::__private::{BaseAddressExt, FileRange, FileRegion};
/// # use hdf5_pure_core::PhysicalAddress;
/// # use hdf5_pure_core::BaseAddress;
/// # use hdf5_pure_format::__private::{FractalHeapBlock, FractalHeapBlockLayout, FractalHeapHeader, LengthWidth, OffsetWidth};
/// # use hdf5_pure_format::FormatError;
/// # use test_util::{fractal_heap, widths::Widths};
/// # let fixture = fractal_heap::Header::new(256);
/// # let encoded = fixture.build(Widths::EIGHT);
/// # let header = FractalHeapHeader::parse_bytes(&encoded, OffsetWidth::Eight, LengthWidth::Eight)?;
/// # let bytes = fixture.direct_block(96, &[7; 113], Widths::EIGHT);
/// # let image = FileRegion::new(FileRange::new(PhysicalAddress::new(768), 128)?, &bytes)?;
/// let base = BaseAddress::new(512);
/// let blocks = FractalHeapBlockLayout::new(
///     &header, OffsetWidth::Eight, LengthWidth::Eight, base.relative(608)?,
/// )?;
/// let mut pending = blocks.root()?.into_iter().collect::<Vec<_>>();
/// while let Some(block) = pending.pop() {
///     match block {
///         FractalHeapBlock::Direct(block) => {
///             let physical = PhysicalAddress::new(base.absolute(block.address())?);
///             let reference = block.at(physical)?;
///             reference.parse_prefix(image.subregion(reference.prefix_range()?)?)?;
///             let view = reference.parse(image.subregion(reference.parse_range())?)?;
///             assert_eq!(view.data(), &[7; 113]);
///         }
///         FractalHeapBlock::Indirect(block) => {
///             let physical = PhysicalAddress::new(base.absolute(block.address())?);
///             let reference = block.at(physical)?;
///             let view = reference.parse(image.subregion(reference.parse_range())?)?;
///             pending.extend(view.children().collect::<Result<Vec<_>, _>>()?);
///         }
///     }
/// }
/// # Ok::<(), FormatError>(())
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FractalHeapBlockLayout {
    layout: FractalHeapLayout,
    owner: StoredAddress,
    root: DecodedStoredAddress,
    root_rows: u16,
    starting_block_size: u64,
    heap_limit: Option<u64>,
    checksummed: bool,
    filtering: FractalHeapFiltering,
}

impl FractalHeapBlockLayout {
    /// Validates block geometry for the heap header at `owner`.
    ///
    /// `owner` is the header's base-relative address. The widths are those parsed from the
    /// superblock.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapGeometry`] if an address does not fit `offsets`,
    /// `owner` is undefined, or the doubling table, root block, or filter metadata is invalid.
    ///
    /// # Examples
    ///
    /// See the traversal example on [`FractalHeapBlockLayout`].
    pub fn new(
        header: &FractalHeapHeader,
        offsets: OffsetWidth,
        lengths: LengthWidth,
        owner: StoredAddress,
    ) -> Result<Self, FormatError> {
        if owner.get() >= StoredAddress::undefined(offsets).get()
            || header.root_block_address.get() > StoredAddress::undefined(offsets).get()
        {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        let layout = FractalHeapLayout::new(header, offsets, lengths)
            .map_err(|_error| FormatError::InvalidFractalHeapGeometry)?;
        if header.current_rows_in_root_indirect_block > layout.max_root_rows() {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        if matches!(&header.filtering, FractalHeapFiltering::Filtered { pipeline, .. }
            if pipeline.filters.len() > u32::BITS as usize)
        {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        let heap = Self {
            layout,
            owner,
            root: if header.root_block_address.is_undefined(offsets) {
                DecodedStoredAddress::Undefined
            } else {
                DecodedStoredAddress::Defined(header.root_block_address)
            },
            root_rows: header.current_rows_in_root_indirect_block,
            starting_block_size: header.starting_block_size,
            heap_limit: (header.max_heap_size < 64)
                .then(|| 1u64 << u32::from(header.max_heap_size)),
            checksummed: header.flags & FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED != 0,
            filtering: header.filtering.clone(),
        };
        heap.root()?;
        Ok(heap)
    }

    /// Returns the root managed block, or [`None`] if its stored address is undefined.
    ///
    /// Huge objects may exist in a heap without a managed root block.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapGeometry`] if the root block's geometry or
    /// filter metadata is invalid.
    pub fn root(&self) -> Result<Option<FractalHeapBlock<'_>>, FormatError> {
        let DecodedStoredAddress::Defined(root) = self.root else {
            return Ok(None);
        };
        Ok(Some(if self.root_rows == 0 {
            let (encoded_size, mask) = match self.filtering {
                FractalHeapFiltering::Unfiltered => (self.starting_block_size, None),
                FractalHeapFiltering::Filtered {
                    root_direct_block_size,
                    root_filter_mask,
                    ..
                } => (root_direct_block_size, Some(root_filter_mask)),
            };
            FractalHeapBlock::Direct(self.direct(
                root,
                0,
                self.starting_block_size,
                encoded_size,
                mask,
            )?)
        } else {
            FractalHeapBlock::Indirect(self.indirect(root, 0, self.root_rows)?)
        }))
    }

    /// Returns the doubling-table geometry used to derive block sizes and offsets.
    pub fn layout(&self) -> &FractalHeapLayout {
        &self.layout
    }

    fn direct(
        &self,
        address: StoredAddress,
        heap_offset: u64,
        decoded_size: u64,
        encoded_size: u64,
        filter_mask: Option<u32>,
    ) -> Result<FractalHeapDirectBlock<'_>, FormatError> {
        self.contains(heap_offset, decoded_size)?;
        let overhead = self.prefix_len() + if self.checksummed { CHECKSUM_LEN } else { 0 };
        if decoded_size < overhead as u64 || encoded_size == 0 {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        if let (Some(mask), FractalHeapFiltering::Filtered { pipeline, .. }) =
            (filter_mask, &self.filtering)
        {
            let count = pipeline.filters.len();
            if count < u32::BITS as usize && mask >> count != 0 {
                return Err(FormatError::InvalidFractalHeapGeometry);
            }
        }
        Ok(FractalHeapDirectBlock {
            heap: self,
            address,
            heap_offset,
            decoded_size,
            encoded_size,
            filter_mask,
        })
    }

    fn indirect(
        &self,
        address: StoredAddress,
        heap_offset: u64,
        rows: u16,
    ) -> Result<FractalHeapIndirectBlock<'_>, FormatError> {
        let span = self
            .layout
            .indirect_heap_size(rows)
            .map_err(|_error| FormatError::InvalidFractalHeapGeometry)?;
        self.contains(heap_offset, span)?;
        Ok(FractalHeapIndirectBlock {
            heap: self,
            address,
            heap_offset,
            rows,
        })
    }

    fn contains(&self, start: u64, length: u64) -> Result<(), FormatError> {
        let end = start
            .checked_add(length)
            .ok_or(FormatError::InvalidFractalHeapGeometry)?;
        if self.heap_limit.is_some_and(|limit| end > limit) {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        Ok(())
    }

    fn parse_prefix(
        &self,
        block: &[u8],
        signature: &[u8; 4],
        heap_offset: u64,
    ) -> Result<(), FormatError> {
        let mut fields = bytes::Fields::new(block, 0);
        if fields.array::<4>()? != *signature {
            return Err(FormatError::InvalidFractalHeapSignature);
        }
        let version = fields.u8()?;
        if version != VERSION {
            return Err(FormatError::InvalidFractalHeapVersion(version));
        }
        let owner = fields.address(self.layout.offset_width())?;
        if owner != self.owner {
            return Err(FormatError::InvalidFractalHeapBlockOwner {
                expected: self.owner.get(),
                actual: owner.get(),
            });
        }
        let mut offset = 0u64;
        for shift in 0..self.layout.block_offset_size() {
            offset |= u64::from(fields.u8()?) << (shift * 8);
        }
        if offset != heap_offset {
            return Err(FormatError::InvalidFractalHeapBlockOffset {
                expected: heap_offset,
                actual: offset,
            });
        }
        Ok(())
    }

    fn prefix_len(&self) -> usize {
        SIGNATURE_LEN
            + 1
            + usize::from(self.layout.offset_width().get())
            + self.layout.block_offset_size()
    }
}

/// Describes a managed block at a stored address in the heap's doubling table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FractalHeapBlock<'a> {
    /// A block containing managed-object data.
    Direct(FractalHeapDirectBlock<'a>),
    /// A block containing child block entries.
    Indirect(FractalHeapIndirectBlock<'a>),
}

/// Describes a direct block's stored address, heap offset, and encoded and decoded sizes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FractalHeapDirectBlock<'a> {
    heap: &'a FractalHeapBlockLayout,
    address: StoredAddress,
    heap_offset: u64,
    decoded_size: u64,
    encoded_size: u64,
    filter_mask: Option<u32>,
}

impl<'a> FractalHeapDirectBlock<'a> {
    /// Associates the encoded allocation with the caller-resolved physical address.
    ///
    /// The caller resolves [`Self::address`] using the opened image's effective base.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the encoded allocation's end exceeds
    /// [`u64::MAX`].
    ///
    /// # Examples
    ///
    /// See the traversal example on [`FractalHeapBlockLayout`].
    pub fn at(
        self,
        address: PhysicalAddress,
    ) -> Result<FractalHeapDirectBlockRef<'a>, FormatError> {
        Ok(FractalHeapDirectBlockRef {
            block: self,
            range: FileRange::new(address, self.encoded_size)?,
        })
    }

    /// Returns the block's base-relative file address.
    pub const fn address(&self) -> StoredAddress {
        self.address
    }
    /// Returns the block's offset in the heap's managed address space, in bytes.
    pub const fn heap_offset(&self) -> u64 {
        self.heap_offset
    }
    /// Returns the complete block size before filter encoding, in bytes.
    pub const fn decoded_size(&self) -> u64 {
        self.decoded_size
    }
    /// Returns the skipped-filter mask, or [`None`] for an unfiltered heap.
    ///
    /// A set bit identifies a filter skipped when encoding this block.
    pub const fn filter_mask(&self) -> Option<u32> {
        self.filter_mask
    }
    /// Returns the heap's filter pipeline with this block's filter mask, or [`None`] for an
    /// unfiltered heap.
    pub fn filters(&self) -> Option<FractalHeapBlockFilters<'a>> {
        let FractalHeapFiltering::Filtered { pipeline, .. } = &self.heap.filtering else {
            return None;
        };
        Some(FractalHeapBlockFilters {
            pipeline,
            mask: self.filter_mask?,
        })
    }
}

/// Describes the filter pipeline and the filter mask of one direct block of a filtered heap.
///
/// The pipeline is the heap header's "I/O Filter Information" field. The mask is the root
/// direct block's "I/O Filter Mask" in the header, or a child's "Filter Mask for Direct Block"
/// entry in its parent indirect block. Both are defined for version 0 of the heap header and of
/// the indirect block in "Fractal Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FractalHeapBlockFilters<'a> {
    /// The heap's filters, in the order the writer applies them.
    pub pipeline: &'a FilterPipeline,
    /// Bit `i` is set if the writer skipped filter `i` of [`pipeline`](Self::pipeline) for this
    /// block.
    pub mask: u32,
}

/// Describes a direct block's physical allocation and expected structural fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FractalHeapDirectBlockRef<'a> {
    block: FractalHeapDirectBlock<'a>,
    range: FileRange,
}

impl FractalHeapDirectBlockRef<'_> {
    /// Parses an unfiltered block and borrows its complete bytes and object-data area.
    ///
    /// Validates the signature, version, owner, and heap offset. With the `checksum` feature,
    /// validates the embedded checksum when the heap header requires one.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedFilteredHeapObject`] if the heap has filters,
    /// [`FormatError::InsufficientRegion`] if `region` does not contain [`Self::parse_range`],
    /// or [`FormatError::InvalidFractalHeapSignature`], [`FormatError::InvalidFractalHeapVersion`],
    /// [`FormatError::InvalidFractalHeapBlockOwner`], or [`FormatError::InvalidFractalHeapBlockOffset`]
    /// if a prefix field is invalid.
    ///
    /// Returns [`FormatError::ChecksumMismatch`] if checksum validation fails.
    ///
    /// # Examples
    ///
    /// See the traversal example on [`FractalHeapBlockLayout`].
    pub fn parse<'bytes>(
        &self,
        region: FileRegion<'bytes>,
    ) -> Result<FractalHeapDirectBlockView<'bytes>, FormatError> {
        self.parse_block(region, false)
    }

    /// Parses a complete unfiltered block with mandatory checksum validation.
    ///
    /// Validates the signature, version, owner, and heap offset. If the heap header requires a
    /// checksum, validates it regardless of the `checksum` feature. Ownership and edit callers
    /// use this operation before accepting a checksummed block.
    ///
    /// # Errors
    ///
    /// Returns the structural errors of [`Self::parse`] and [`FormatError::ChecksumMismatch`]
    /// if the complete block's checksum differs from its stored value.
    ///
    /// # Examples
    ///
    /// See the bounded object example on [`Self::parse_object`].
    pub fn parse_checked<'bytes>(
        &self,
        region: FileRegion<'bytes>,
    ) -> Result<FractalHeapDirectBlockView<'bytes>, FormatError> {
        self.parse_block(region, true)
    }

    /// Validates a managed object's bounds from an unfiltered block prefix.
    ///
    /// `heap_offset` is in the heap's managed address space. `region` must contain
    /// [`Self::object_prefix_range`], including the checksum field when present. Validates the
    /// signature, version, owner, and block offset, then checks that the object lies within the
    /// block's data area. The stored checksum is available through
    /// [`FractalHeapObjectRef::stored_checksum`]. Use [`Self::parse_checked`] with the complete
    /// region to validate that checksum.
    ///
    /// # Errors
    ///
    /// Returns the structural errors of [`Self::parse_prefix`], [`FormatError::InsufficientRegion`]
    /// if the checksum field is outside `region`, or [`FormatError::InvalidFractalHeapGeometry`]
    /// if the object bounds overflow, overlap the prefix, or exceed the block.
    ///
    /// # Examples
    ///
    /// The caller can read an object's bytes after validating its block prefix and bounds.
    /// Full-block checksum validation is a separate operation on the complete region.
    ///
    /// ```
    /// # use hdf5_pure_core::__private::BaseAddressExt;
    /// # use hdf5_pure_core::__private::FileRegion;
    /// # use hdf5_pure_core::PhysicalAddress;
    /// # use hdf5_pure_core::BaseAddress;
    /// # use hdf5_pure_format::__private::FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED;
    /// # use hdf5_pure_format::__private::FractalHeapBlock;
    /// # use hdf5_pure_format::__private::FractalHeapBlockLayout;
    /// # use hdf5_pure_format::__private::FractalHeapHeader;
    /// # use hdf5_pure_format::__private::FractalHeapObjectRef;
    /// # use hdf5_pure_format::__private::LengthWidth;
    /// # use hdf5_pure_format::__private::OffsetWidth;
    /// # use hdf5_pure_format::__private;
    /// # use hdf5_pure_format::FormatError;
    /// # use test_util::fractal_heap;
    /// # use test_util::widths::Widths;
    /// # let fixture = fractal_heap::Header::new(256);
    /// # let encoded = fixture.build(Widths::EIGHT);
    /// # let mut header = FractalHeapHeader::parse_bytes(&encoded, OffsetWidth::Eight, LengthWidth::Eight)?;
    /// # header.flags = FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED;
    /// # let base = BaseAddress::new(512);
    /// # let blocks = FractalHeapBlockLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight, base.relative(608)?)?;
    /// # let Some(FractalHeapBlock::Direct(block)) = blocks.root()? else { panic!("expected direct fixture root") };
    /// # let reference = block.at(PhysicalAddress::new(base.absolute(block.address())?))?;
    /// # let mut bytes = fixture.direct_block(96, &[0; 113], Widths::EIGHT);
    /// # let checksum_at = usize::try_from(reference.prefix_range()?.length()).unwrap();
    /// # let checksum_width = 0u32.to_le_bytes().len();
    /// # let object_at = checksum_at + checksum_width;
    /// # bytes[object_at..object_at + 3].copy_from_slice(b"abc");
    /// # let stored_checksum = __private::jenkins_lookup3(&bytes);
    /// # let encoded_checksum = stored_checksum.to_le_bytes();
    /// # bytes[checksum_at..checksum_at + encoded_checksum.len()].copy_from_slice(&encoded_checksum);
    /// # let image = FileRegion::new(reference.range(), &bytes)?;
    /// let prefix = image.subregion(reference.object_prefix_range()?)?;
    /// let heap_offset = prefix.range().length();
    /// let object: FractalHeapObjectRef = reference.parse_object(prefix, heap_offset, 3)?;
    /// assert_eq!(object.block_range(), reference.range());
    /// assert_eq!(object.stored_checksum(), Some(stored_checksum));
    /// assert_eq!(object.parse(image.subregion(object.range())?)?, b"abc");
    ///
    /// let checked = reference.parse_checked(image.subregion(reference.parse_range())?)?;
    /// assert_eq!(checked.data().get(..3), Some(&b"abc"[..]));
    /// # let mut damaged = bytes.clone();
    /// # *damaged.last_mut().unwrap() ^= 1;
    /// # let damaged = FileRegion::new(reference.range(), &damaged)?;
    /// let object = reference.parse_object(damaged.subregion(reference.object_prefix_range()?)?, heap_offset, 3)?;
    /// assert_eq!(object.stored_checksum(), Some(stored_checksum));
    /// let error = reference.parse_checked(damaged).unwrap_err();
    /// let FormatError::ChecksumMismatch { expected, computed } = error else { panic!("expected checksum mismatch, got {error:?}") };
    /// assert_eq!(expected, stored_checksum);
    /// assert_ne!(computed, stored_checksum);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn parse_object(
        &self,
        region: FileRegion<'_>,
        heap_offset: u64,
        length: u64,
    ) -> Result<FractalHeapObjectRef, FormatError> {
        self.parse_prefix(region)?;
        let prefix = region.subregion(self.object_prefix_range()?)?.bytes();
        let stored_checksum = if self.block.heap.checksummed {
            Some(bytes::Fields::new(prefix, self.block.heap.prefix_len()).u32()?)
        } else {
            None
        };
        let local = self.object_offset(heap_offset, length)?;
        Ok(FractalHeapObjectRef {
            range: FileRange::new(self.range.start().checked_add(local)?, length)?,
            block_range: self.range,
            stored_checksum,
        })
    }

    /// Parses a block from its decoded bytes and borrows its complete bytes and object-data area.
    ///
    /// `decoded` holds the block with its filters reversed, or the stored block of an unfiltered
    /// heap. Validates its length, signature, version, owner, and heap offset. If the heap header
    /// requires a checksum, validates it regardless of the `checksum` feature.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapGeometry`] if `decoded` is not
    /// [`FractalHeapDirectBlock::decoded_size`] bytes long,
    /// [`FormatError::InvalidFractalHeapSignature`], [`FormatError::InvalidFractalHeapVersion`],
    /// [`FormatError::InvalidFractalHeapBlockOwner`], or
    /// [`FormatError::InvalidFractalHeapBlockOffset`] if a prefix field is invalid, and
    /// [`FormatError::ChecksumMismatch`] if the block's checksum differs from its stored value.
    pub fn parse_decoded_checked<'bytes>(
        &self,
        decoded: &'bytes [u8],
    ) -> Result<FractalHeapDirectBlockView<'bytes>, FormatError> {
        self.check_decoded_len(decoded)?;
        self.parse_block_bytes(decoded, true)
    }

    /// Validates a managed object's bounds in a block's decoded bytes and borrows the object.
    ///
    /// `decoded` is as for [`Self::parse_decoded_checked`], and `heap_offset` is in the heap's
    /// managed address space. Validates the length, signature, version, owner, and block offset of
    /// `decoded`, then checks that the object lies within the block's data area. Checksum
    /// validation requires [`Self::parse_decoded_checked`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapGeometry`] if `decoded` is not
    /// [`FractalHeapDirectBlock::decoded_size`] bytes long or if the object bounds overflow,
    /// overlap the prefix, or exceed the block, and
    /// [`FormatError::InvalidFractalHeapSignature`], [`FormatError::InvalidFractalHeapVersion`],
    /// [`FormatError::InvalidFractalHeapBlockOwner`], or
    /// [`FormatError::InvalidFractalHeapBlockOffset`] if a prefix field is invalid.
    pub fn parse_decoded_object<'bytes>(
        &self,
        decoded: &'bytes [u8],
        heap_offset: u64,
        length: u64,
    ) -> Result<&'bytes [u8], FormatError> {
        self.check_decoded_len(decoded)?;
        self.block
            .heap
            .parse_prefix(decoded, &DIRECT_SIGNATURE, self.block.heap_offset)?;
        decoded
            .get(convert::slice_range(
                self.object_offset(heap_offset, length)?,
                length,
            )?)
            .ok_or(FormatError::InvalidFractalHeapGeometry)
    }

    /// Returns the unfiltered prefix and the checksum field when present.
    ///
    /// This range is sufficient for [`Self::parse_object`]. Complete checksum validation requires
    /// [`Self::parse_range`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedFilteredHeapObject`] if the heap has filters.
    pub fn object_prefix_range(&self) -> Result<FileRange, FormatError> {
        let prefix = self.prefix_range()?;
        FileRange::new(
            prefix.start(),
            prefix.length()
                + if self.block.heap.checksummed {
                    CHECKSUM_LEN as u64
                } else {
                    0
                },
        )
    }

    /// Validates the unfiltered block's signature, version, owner, and heap offset.
    ///
    /// A region containing [`Self::prefix_range`] is sufficient. Checksum validation requires
    /// [`Self::parse_checked`] with the complete block.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedFilteredHeapObject`] if the heap has filters,
    /// [`FormatError::InsufficientRegion`] if `region` does not contain [`Self::prefix_range`],
    /// or [`FormatError::InvalidFractalHeapSignature`], [`FormatError::InvalidFractalHeapVersion`],
    /// [`FormatError::InvalidFractalHeapBlockOwner`], or [`FormatError::InvalidFractalHeapBlockOffset`]
    /// if a prefix field is invalid.
    ///
    /// # Examples
    ///
    /// See the traversal example on [`FractalHeapBlockLayout`].
    pub fn parse_prefix(&self, region: FileRegion<'_>) -> Result<(), FormatError> {
        let prefix = region.subregion(self.prefix_range()?)?.bytes();
        self.block
            .heap
            .parse_prefix(prefix, &DIRECT_SIGNATURE, self.block.heap_offset)
    }

    /// Returns the complete encoded physical allocation.
    pub const fn range(&self) -> FileRange {
        self.range
    }
    /// Returns the physical bytes required for complete block parsing.
    ///
    /// This equals [`Self::range`]. Use [`Self::parse_decoded_checked`] for a filtered block's
    /// decoded bytes.
    pub const fn parse_range(&self) -> FileRange {
        self.range
    }

    /// Returns the unfiltered structural prefix, ending before the optional checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedFilteredHeapObject`] if the heap has filters.
    pub fn prefix_range(&self) -> Result<FileRange, FormatError> {
        if self.block.filter_mask.is_some() {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }
        FileRange::new(self.range.start(), self.block.heap.prefix_len() as u64)
    }
    fn parse_block<'bytes>(
        &self,
        region: FileRegion<'bytes>,
        require_checksum: bool,
    ) -> Result<FractalHeapDirectBlockView<'bytes>, FormatError> {
        if self.block.filter_mask.is_some() {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }
        self.parse_block_bytes(region.subregion(self.range)?.bytes(), require_checksum)
    }

    fn parse_block_bytes<'bytes>(
        &self,
        block: &'bytes [u8],
        require_checksum: bool,
    ) -> Result<FractalHeapDirectBlockView<'bytes>, FormatError> {
        self.block
            .heap
            .parse_prefix(block, &DIRECT_SIGNATURE, self.block.heap_offset)?;
        let prefix_len = self.block.heap.prefix_len();
        let data_at = if self.block.heap.checksummed {
            if require_checksum {
                checksum::verify_zeroed_required(block, prefix_len)?;
            } else {
                checksum::verify_zeroed(block, prefix_len)?;
            }
            prefix_len + CHECKSUM_LEN
        } else {
            prefix_len
        };
        let data = block
            .get(data_at..)
            .ok_or(FormatError::InvalidFractalHeapGeometry)?;
        Ok(FractalHeapDirectBlockView { bytes: block, data })
    }

    fn check_decoded_len(&self, decoded: &[u8]) -> Result<(), FormatError> {
        if decoded.len().to_u64() != self.block.decoded_size {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        Ok(())
    }

    fn object_offset(&self, heap_offset: u64, length: u64) -> Result<u64, FormatError> {
        let local = heap_offset
            .checked_sub(self.block.heap_offset)
            .ok_or(FormatError::InvalidFractalHeapGeometry)?;
        let end = local
            .checked_add(length)
            .ok_or(FormatError::InvalidFractalHeapGeometry)?;
        let data_at = self.block.heap.prefix_len()
            + if self.block.heap.checksummed {
                CHECKSUM_LEN
            } else {
                0
            };
        if local < data_at.to_u64() || end > self.block.decoded_size {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        Ok(local)
    }
}

/// Describes a managed object's physical range after prefix and bounds validation.
///
/// A stored checksum, when present, records the owning block's checksum field. Construction
/// through [`FractalHeapDirectBlockRef::parse_object`] does not validate that checksum.
///
/// # Examples
///
/// See the bounded object example on [`FractalHeapDirectBlockRef::parse_object`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FractalHeapObjectRef {
    range: FileRange,
    block_range: FileRange,
    stored_checksum: Option<u32>,
}

impl FractalHeapObjectRef {
    /// Borrows the object bytes from a region containing the object's range.
    ///
    /// This operation checks region containment. It does not validate the owning block's checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain [`Self::range`].
    ///
    /// # Examples
    ///
    /// See the bounded object example on [`FractalHeapDirectBlockRef::parse_object`].
    pub fn parse<'bytes>(&self, region: FileRegion<'bytes>) -> Result<&'bytes [u8], FormatError> {
        Ok(region.subregion(self.range)?.bytes())
    }

    /// Returns the physical bytes occupied by the object.
    pub const fn range(&self) -> FileRange {
        self.range
    }
    /// Returns the owning direct block's complete physical allocation.
    pub const fn block_range(&self) -> FileRange {
        self.block_range
    }
    /// Returns the owning block's stored checksum, or [`None`] if absent.
    ///
    /// Call [`FractalHeapDirectBlockRef::parse_checked`] for full-block checksum validation.
    pub const fn stored_checksum(&self) -> Option<u32> {
        self.stored_checksum
    }
}

/// Borrows an unfiltered direct block after structural and enabled checksum validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FractalHeapDirectBlockView<'a> {
    bytes: &'a [u8],
    data: &'a [u8],
}

impl<'a> FractalHeapDirectBlockView<'a> {
    /// Returns the complete block, including its prefix and optional checksum.
    pub const fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
    /// Returns the object-data area after the prefix and optional checksum.
    ///
    /// This area includes any free space between managed objects.
    pub const fn data(&self) -> &'a [u8] {
        self.data
    }
}

/// Describes an indirect block's stored address, heap offset, and row count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FractalHeapIndirectBlock<'a> {
    heap: &'a FractalHeapBlockLayout,
    address: StoredAddress,
    heap_offset: u64,
    rows: u16,
}

impl<'a> FractalHeapIndirectBlock<'a> {
    /// Associates the complete indirect-block allocation with a physical address.
    ///
    /// The caller resolves [`Self::address`] using the opened image's effective base.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapGeometry`] if the block size cannot be
    /// derived, or [`FormatError::OffsetOverflow`] if the allocation's end exceeds [`u64::MAX`].
    ///
    /// # Examples
    ///
    /// See the traversal example on [`FractalHeapBlockLayout`].
    pub fn at(
        self,
        address: PhysicalAddress,
    ) -> Result<FractalHeapIndirectBlockRef<'a>, FormatError> {
        let size = self
            .heap
            .layout
            .indirect_block_size(self.rows)
            .map_err(|_error| FormatError::InvalidFractalHeapGeometry)?;
        Ok(FractalHeapIndirectBlockRef {
            block: self,
            range: FileRange::new(address, size)?,
        })
    }

    /// Returns the block's base-relative file address.
    pub const fn address(&self) -> StoredAddress {
        self.address
    }
    /// Returns the block's offset in the heap's managed address space, in bytes.
    pub const fn heap_offset(&self) -> u64 {
        self.heap_offset
    }
    /// Returns the number of doubling-table rows encoded in this block.
    pub const fn rows(&self) -> u16 {
        self.rows
    }
}

/// Describes an indirect block's physical allocation and expected structural fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FractalHeapIndirectBlockRef<'a> {
    block: FractalHeapIndirectBlock<'a>,
    range: FileRange,
}

impl<'heap> FractalHeapIndirectBlockRef<'heap> {
    /// Parses the block prefix and borrows the encoded child entries.
    ///
    /// Validates the signature, version, owner, and heap offset. With the `checksum` feature,
    /// validates the trailing checksum. Child geometry is validated by
    /// [`FractalHeapIndirectBlockView::child`] or while iterating
    /// [`FractalHeapIndirectBlockView::children`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain
    /// [`Self::parse_range`], or [`FormatError::InvalidFractalHeapSignature`],
    /// [`FormatError::InvalidFractalHeapVersion`], [`FormatError::InvalidFractalHeapBlockOwner`],
    /// or [`FormatError::InvalidFractalHeapBlockOffset`] if a prefix field is invalid.
    ///
    /// Returns [`FormatError::ChecksumMismatch`] if checksum validation fails.
    ///
    /// # Examples
    ///
    /// See the traversal example on [`FractalHeapBlockLayout`].
    pub fn parse<'bytes>(
        &self,
        region: FileRegion<'bytes>,
    ) -> Result<FractalHeapIndirectBlockView<'heap, 'bytes>, FormatError> {
        let bytes = region.subregion(self.range)?.bytes();
        self.block
            .heap
            .parse_prefix(bytes, &INDIRECT_SIGNATURE, self.block.heap_offset)?;
        checksum::verify_trailing(bytes)?;
        Ok(FractalHeapIndirectBlockView {
            block: self.block,
            bytes,
        })
    }

    /// Returns the complete physical allocation, including the trailing checksum.
    pub const fn range(&self) -> FileRange {
        self.range
    }
    /// Returns the physical bytes required for parsing, equal to [`Self::range`].
    pub const fn parse_range(&self) -> FileRange {
        self.range
    }
}

/// Borrows an indirect block after prefix and enabled checksum validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FractalHeapIndirectBlockView<'heap, 'bytes> {
    block: FractalHeapIndirectBlock<'heap>,
    bytes: &'bytes [u8],
}

impl<'heap, 'bytes> FractalHeapIndirectBlockView<'heap, 'bytes> {
    /// Iterates allocated children in row order, then column order.
    ///
    /// Undefined entries are omitted. Each item is validated as in [`Self::child`].
    ///
    /// # Examples
    ///
    /// See the traversal example on [`FractalHeapBlockLayout`].
    pub fn children(
        &self,
    ) -> impl Iterator<Item = Result<FractalHeapBlock<'heap>, FormatError>> + '_ {
        (0..usize::from(self.block.rows)).flat_map(move |row| {
            (0..self.block.heap.layout.table_width())
                .filter_map(move |column| self.child(row, column).transpose())
        })
    }

    /// Derives the child at a zero-based doubling-table slot.
    ///
    /// Returns [`None`] for an undefined address with valid unused filter metadata.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapGeometry`] if the slot is out of bounds,
    /// the child exceeds the heap address space, or the child size, row count, or filter
    /// metadata is invalid.
    pub fn child(
        &self,
        row: usize,
        column: usize,
    ) -> Result<Option<FractalHeapBlock<'heap>>, FormatError> {
        let FractalHeapIndirectBlock {
            heap,
            heap_offset,
            rows,
            address: _,
        } = self.block;
        let layout = &heap.layout;
        if row >= usize::from(rows) || column >= layout.table_width() {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        let direct = row < layout.direct_rows();
        let direct_count = usize::from(rows).min(layout.direct_rows()) * layout.table_width();
        let entry = row * layout.table_width() + column;
        let offset_size = usize::from(layout.offset_width().get());
        let filter_size = if heap.filtering.is_filtered() {
            usize::from(layout.length_width().get()) + FILTER_MASK_LEN
        } else {
            0
        };
        let pos = heap.prefix_len() + entry * offset_size + entry.min(direct_count) * filter_size;
        let address = crate::width::decode_stored_address(layout.offset_width(), self.bytes, pos)?;
        let filtered = if direct && filter_size != 0 {
            let mut fields = bytes::Fields::new(self.bytes, pos + offset_size);
            Some((fields.length(layout.length_width())?, fields.u32()?))
        } else {
            None
        };
        let DecodedStoredAddress::Defined(address) = address else {
            if filtered.is_some_and(|(size, _mask)| size != 0) {
                return Err(FormatError::InvalidFractalHeapGeometry);
            }
            return Ok(None);
        };
        let offset = heap_offset
            .checked_add(
                layout
                    .slot_offset(row, column)
                    .map_err(|_error| FormatError::InvalidFractalHeapGeometry)?,
            )
            .ok_or(FormatError::InvalidFractalHeapGeometry)?;
        let size = layout
            .block_size_for_row(row)
            .map_err(|_error| FormatError::InvalidFractalHeapGeometry)?;
        Ok(Some(if direct {
            let (encoded_size, mask) = match filtered {
                Some((encoded_size, mask)) => (encoded_size, Some(mask)),
                None => (size, None),
            };
            FractalHeapBlock::Direct(heap.direct(address, offset, size, encoded_size, mask)?)
        } else {
            let child_rows = layout
                .child_indirect_rows(row)
                .map_err(|_error| FormatError::InvalidFractalHeapGeometry)?;
            if child_rows >= rows
                || layout
                    .indirect_heap_size(child_rows)
                    .map_err(|_error| FormatError::InvalidFractalHeapGeometry)?
                    != size
            {
                return Err(FormatError::InvalidFractalHeapGeometry);
            }
            FractalHeapBlock::Indirect(heap.indirect(address, offset, child_rows)?)
        }))
    }

    /// Returns the complete encoded block, including its prefix and checksum.
    pub const fn bytes(&self) -> &'bytes [u8] {
        self.bytes
    }
}

// The block signatures, version, and checksum width are defined in "Fractal Heap"
// of the format specification, version 4.0.
pub(crate) const DIRECT_SIGNATURE: [u8; 4] = *b"FHDB";
pub(crate) const INDIRECT_SIGNATURE: [u8; 4] = *b"FHIB";
pub(crate) const VERSION: u8 = 0;
const SIGNATURE_LEN: usize = 4;
pub(crate) const CHECKSUM_LEN: usize = 4;
const FILTER_MASK_LEN: usize = 4;

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use rstest::rstest;
    use test_util::fractal_heap;
    use test_util::widths::Widths;

    use crate::address::StoredAddressExt;
    use crate::filter_pipeline::FilterDescription;
    use crate::filter_pipeline::FilterPipeline;

    use super::*;

    fn heap(
        offsets: OffsetWidth,
        rows: u16,
        checksummed: bool,
        filtered: bool,
    ) -> FractalHeapBlockLayout {
        let encoded = fractal_heap::Header::new(256).build(Widths::EIGHT);
        let mut header =
            FractalHeapHeader::parse_bytes(&encoded, OffsetWidth::Eight, LengthWidth::Eight)
                .unwrap();
        header.current_rows_in_root_indirect_block = rows;
        header.flags = if checksummed {
            FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED
        } else {
            0
        };
        if filtered {
            header.filtering = FractalHeapFiltering::Filtered {
                root_direct_block_size: 47,
                root_filter_mask: 1,
                pipeline: FilterPipeline {
                    version: 2,
                    filters: vec![FilterDescription {
                        filter_id: 1,
                        name: None,
                        flags: 1,
                        client_data: vec![6],
                    }],
                },
            };
        }
        FractalHeapBlockLayout::new(&header, offsets, LengthWidth::Eight, StoredAddress::new(96))
            .unwrap()
    }

    fn push_number(bytes: &mut Vec<u8>, number: u64, width: usize) {
        bytes.extend_from_slice(&number.to_le_bytes()[..width]);
    }

    fn prefix(heap: &FractalHeapBlockLayout, signature: &[u8; 4], offset: u64) -> Vec<u8> {
        let mut bytes = signature.to_vec();
        bytes.push(VERSION);
        push_number(
            &mut bytes,
            heap.owner.get(),
            usize::from(heap.layout.offset_width().get()),
        );
        push_number(&mut bytes, offset, heap.layout.block_offset_size());
        bytes
    }

    fn direct_bytes(heap: &FractalHeapBlockLayout) -> Vec<u8> {
        let mut bytes = prefix(heap, &DIRECT_SIGNATURE, 0);
        let checksum_at = bytes.len();
        bytes.resize(128, 0);
        let data_at = checksum_at + if heap.checksummed { CHECKSUM_LEN } else { 0 };
        bytes[data_at..].fill(0x5A);
        if heap.checksummed {
            let computed = checksum::jenkins_lookup3(&bytes);
            bytes[checksum_at..checksum_at + CHECKSUM_LEN].copy_from_slice(&computed.to_le_bytes());
        }
        bytes
    }

    fn indirect_bytes(heap: &FractalHeapBlockLayout, entries: &[(usize, usize, u64)]) -> Vec<u8> {
        let mut bytes = prefix(heap, &INDIRECT_SIGNATURE, 0);
        let width = heap.layout.offset_width();
        let undefined = StoredAddress::undefined(width).get();
        for row in 0..usize::from(heap.root_rows) {
            for column in 0..heap.layout.table_width() {
                let address = entries
                    .iter()
                    .find(|&&(r, c, _)| r == row && c == column)
                    .map_or(undefined, |&(_, _, addr)| addr);
                push_number(&mut bytes, address, usize::from(width.get()));
                if row < heap.layout.direct_rows() && heap.filtering.is_filtered() {
                    push_number(
                        &mut bytes,
                        if address == undefined { 0 } else { 47 },
                        usize::from(heap.layout.length_width().get()),
                    );
                    bytes.extend_from_slice(&1u32.to_le_bytes());
                }
            }
        }
        let stored = checksum::jenkins_lookup3(&bytes);
        bytes.extend_from_slice(&stored.to_le_bytes());
        assert_eq!(
            bytes.len() as u64,
            heap.layout.indirect_block_size(heap.root_rows).unwrap()
        );
        bytes
    }

    fn region(at: u64, bytes: &[u8]) -> FileRegion<'_> {
        FileRegion::new(
            FileRange::new(PhysicalAddress::new(at), bytes.len() as u64).unwrap(),
            bytes,
        )
        .unwrap()
    }

    fn direct_root(heap: &FractalHeapBlockLayout) -> FractalHeapDirectBlock<'_> {
        let Some(FractalHeapBlock::Direct(block)) = heap.root().unwrap() else {
            panic!("expected direct root");
        };
        block
    }

    fn indirect_root(heap: &FractalHeapBlockLayout) -> FractalHeapIndirectBlock<'_> {
        let Some(FractalHeapBlock::Indirect(block)) = heap.root().unwrap() else {
            panic!("expected indirect root");
        };
        block
    }

    #[rstest]
    #[case::two(OffsetWidth::Two, false, 0)]
    #[case::four(OffsetWidth::Four, true, 512)]
    #[case::eight(OffsetWidth::Eight, true, 1u64 << 40)]
    fn direct_views_borrow_complete_blocks_at_physical_coordinates(
        #[case] offsets: OffsetWidth,
        #[case] checksummed: bool,
        #[case] base: u64,
    ) {
        let heap = heap(offsets, 0, checksummed, false);
        let block = direct_root(&heap);
        assert_eq!(block.address().get(), 256);
        assert_eq!(block.heap_offset(), 0);
        assert_eq!(block.decoded_size(), 128);
        assert_eq!(block.filter_mask(), None);
        let reference = block
            .at(PhysicalAddress::new(base + block.address().get()))
            .unwrap();
        let bytes = direct_bytes(&heap);
        let view = reference.parse(region(base + 256, &bytes)).unwrap();
        assert_eq!(view.bytes(), bytes);
        assert_eq!(view.bytes().as_ptr(), bytes.as_ptr());
        let data_at = heap.prefix_len() + if checksummed { CHECKSUM_LEN } else { 0 };
        assert_eq!(view.data(), vec![0x5A; 128 - data_at]);
        assert_eq!(reference.range(), reference.parse_range());
        assert_eq!(reference.range().start().get(), base + 256);
    }

    #[rstest]
    #[case::unchecksummed(false)]
    #[case::checksummed(true)]
    fn bounded_object_refs_retain_unvalidated_checksum_metadata(
        #[case] checksummed: bool,
        #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
    ) {
        let heap = heap(offsets, 0, checksummed, false);
        let bytes = direct_bytes(&heap);
        let reference = direct_root(&heap).at(PhysicalAddress::new(768)).unwrap();
        let prefix = reference.object_prefix_range().unwrap();
        let data_at = prefix.length();
        let object = reference
            .parse_object(region(768, &bytes[..data_at as usize]), data_at, 3)
            .unwrap();
        assert_eq!(
            object.range(),
            FileRange::new(PhysicalAddress::new(768 + data_at), 3).unwrap()
        );
        assert_eq!(object.block_range(), reference.range());
        assert_eq!(
            object.stored_checksum(),
            checksummed.then(|| {
                u32::from_le_bytes(
                    bytes[heap.prefix_len()..heap.prefix_len() + CHECKSUM_LEN]
                        .try_into()
                        .unwrap(),
                )
            })
        );
        assert_eq!(object.parse(region(768, &bytes)).unwrap(), &[0x5A; 3]);
    }

    #[rstest]
    #[case::prefix(0, 3)]
    #[case::overrun(127, 2)]
    #[case::overflow(u64::MAX, 1)]
    fn bounded_object_refs_reject_invalid_object_bounds(#[case] offset: u64, #[case] length: u64) {
        let heap = heap(OffsetWidth::Eight, 0, true, false);
        let bytes = direct_bytes(&heap);
        let reference = direct_root(&heap).at(PhysicalAddress::new(256)).unwrap();
        assert_eq!(
            reference
                .parse_object(region(256, &bytes), offset, length)
                .unwrap_err(),
            FormatError::InvalidFractalHeapGeometry
        );
    }

    #[test]
    fn required_checksum_validation_rejects_corruption_with_every_feature_set() {
        let heap = heap(OffsetWidth::Eight, 0, true, false);
        let mut bytes = direct_bytes(&heap);
        let checksum_at = heap.prefix_len();
        let stored = u32::from_le_bytes(
            bytes[checksum_at..checksum_at + CHECKSUM_LEN]
                .try_into()
                .unwrap(),
        );
        bytes[127] ^= 1;
        let reference = direct_root(&heap).at(PhysicalAddress::new(256)).unwrap();
        let object = reference.parse_object(region(256, &bytes), 19, 3).unwrap();
        assert_eq!(object.stored_checksum(), Some(stored));
        let mut zeroed = bytes.clone();
        zeroed[checksum_at..checksum_at + CHECKSUM_LEN].fill(0);
        assert_eq!(
            reference.parse_checked(region(256, &bytes)).unwrap_err(),
            FormatError::ChecksumMismatch {
                expected: stored,
                computed: checksum::jenkins_lookup3(&zeroed)
            }
        );
    }

    #[rstest]
    #[case::direct(false)]
    #[case::indirect(true)]
    fn supplied_short_regions_return_insufficient_region(#[case] indirect: bool) {
        let heap = heap(
            OffsetWidth::Eight,
            if indirect { 2 } else { 0 },
            false,
            false,
        );
        let bytes = if indirect {
            indirect_bytes(&heap, &[])
        } else {
            direct_bytes(&heap)
        };
        let required_len = bytes.len() as u64;
        let supplied = region(256, &bytes[..bytes.len() - 1]);
        let error = if indirect {
            indirect_root(&heap)
                .at(PhysicalAddress::new(256))
                .unwrap()
                .parse(supplied)
                .unwrap_err()
        } else {
            direct_root(&heap)
                .at(PhysicalAddress::new(256))
                .unwrap()
                .parse(supplied)
                .unwrap_err()
        };
        assert_eq!(
            error,
            FormatError::InsufficientRegion {
                required_start: 256,
                required_length: required_len,
                provided_start: 256,
                provided_length: required_len - 1,
            }
        );
    }

    #[rstest]
    #[case::signature(0, 0, FormatError::InvalidFractalHeapSignature)]
    #[case::version(4, 1, FormatError::InvalidFractalHeapVersion(1))]
    #[case::owner(5, 97, FormatError::InvalidFractalHeapBlockOwner { expected: 96, actual: 97 })]
    #[case::offset(13, 1, FormatError::InvalidFractalHeapBlockOffset { expected: 0, actual: 1 })]
    fn block_prefixes_reject_malformed_fields(
        #[case] position: usize,
        #[case] value: u8,
        #[case] expected: FormatError,
    ) {
        let heap = heap(OffsetWidth::Eight, 0, false, false);
        let reference = direct_root(&heap).at(PhysicalAddress::new(256)).unwrap();
        let mut bytes = direct_bytes(&heap);
        bytes[position] = value;
        assert_eq!(reference.parse(region(256, &bytes)).unwrap_err(), expected);
        assert_eq!(
            reference.parse_prefix(region(256, &bytes)).unwrap_err(),
            expected
        );
    }

    #[test]
    fn prefix_parsing_requires_only_structural_bytes() {
        let heap = heap(OffsetWidth::Eight, 0, true, false);
        let reference = direct_root(&heap).at(PhysicalAddress::new(256)).unwrap();
        let bytes = prefix(&heap, &DIRECT_SIGNATURE, 0);
        assert_eq!(
            reference.prefix_range().unwrap().length(),
            bytes.len() as u64
        );
        reference.parse_prefix(region(256, &bytes)).unwrap();
        assert_eq!(
            reference.parse(region(256, &bytes)).unwrap_err(),
            FormatError::InsufficientRegion {
                required_start: 256,
                required_length: 128,
                provided_start: 256,
                provided_length: bytes.len() as u64,
            }
        );
    }

    #[rstest]
    #[case::two(OffsetWidth::Two, false)]
    #[case::four(OffsetWidth::Four, true)]
    #[case::eight(OffsetWidth::Eight, false)]
    fn indirect_views_derive_direct_and_indirect_children(
        #[case] offsets: OffsetWidth,
        #[case] filtered: bool,
    ) {
        let heap = heap(offsets, 8, false, filtered);
        let bytes = indirect_bytes(&heap, &[(0, 0, 512), (1, 2, 640), (5, 1, 768), (7, 3, 900)]);
        let reference = indirect_root(&heap).at(PhysicalAddress::new(256)).unwrap();
        let view = reference.parse(region(256, &bytes)).unwrap();
        let children = view.children().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(children.len(), 4);
        let FractalHeapBlock::Direct(first) = children[0] else {
            panic!("expected direct child");
        };
        assert_eq!(first.address().get(), 512);
        assert_eq!(first.heap_offset(), 0);
        assert_eq!(first.decoded_size(), 128);
        assert_eq!(first.filter_mask(), filtered.then_some(1));
        assert_eq!(
            first
                .at(PhysicalAddress::new(1 << 40))
                .unwrap()
                .range()
                .length(),
            if filtered { 47 } else { 128 }
        );
        let FractalHeapBlock::Direct(second) = children[1] else {
            panic!("expected direct child");
        };
        assert_eq!(second.heap_offset(), 768);
        let FractalHeapBlock::Indirect(third) = children[2] else {
            panic!("expected indirect child");
        };
        assert_eq!(third.address().get(), 768);
        assert_eq!(third.rows(), 3);
        assert_eq!(third.heap_offset(), heap.layout.slot_offset(5, 1).unwrap());
        let FractalHeapBlock::Indirect(fourth) = children[3] else {
            panic!("expected indirect child");
        };
        assert_eq!(fourth.rows(), 5);
        assert_eq!(fourth.heap_offset(), heap.layout.slot_offset(7, 3).unwrap());
        assert_eq!(view.child(0, 1).unwrap(), None);
        assert_eq!(
            view.child(8, 0).unwrap_err(),
            FormatError::InvalidFractalHeapGeometry
        );
        assert_eq!(
            view.child(0, 4).unwrap_err(),
            FormatError::InvalidFractalHeapGeometry
        );
        assert_eq!(view.bytes().as_ptr(), bytes.as_ptr());
    }

    #[test]
    fn filtered_refs_keep_encoded_ranges_separate_from_decoded_geometry() {
        let heap = heap(OffsetWidth::Eight, 0, true, true);
        let block = direct_root(&heap);
        let reference = block.at(PhysicalAddress::new(768)).unwrap();
        assert_eq!(block.decoded_size(), 128);
        assert_eq!(block.filter_mask(), Some(1));
        assert_eq!(reference.range().length(), 47);
        assert_eq!(
            reference.prefix_range().unwrap_err(),
            FormatError::UnsupportedFilteredHeapObject
        );
        assert_eq!(
            reference.parse(region(768, &[0; 47])).unwrap_err(),
            FormatError::UnsupportedFilteredHeapObject
        );
    }

    #[rstest]
    #[case::unfiltered(false)]
    #[case::filtered(true)]
    fn direct_blocks_parse_from_their_decoded_bytes(#[case] filtered: bool) {
        let heap = heap(OffsetWidth::Eight, 0, true, filtered);
        let block = direct_root(&heap);
        let reference = block.at(PhysicalAddress::new(768)).unwrap();
        let decoded = direct_bytes(&heap);
        let data_at = heap.prefix_len() + CHECKSUM_LEN;
        let view = reference.parse_decoded_checked(&decoded).unwrap();
        assert_eq!(view.bytes(), decoded);
        assert_eq!(view.data(), vec![0x5A; 128 - data_at]);
        assert_eq!(
            reference
                .parse_decoded_object(&decoded, data_at.to_u64() + 2, 3)
                .unwrap(),
            &[0x5A; 3]
        );
        let FractalHeapFiltering::Filtered { pipeline, .. } = &heap.filtering else {
            assert_eq!(block.filters(), None);
            return;
        };
        assert_eq!(
            block.filters(),
            Some(FractalHeapBlockFilters { pipeline, mask: 1 })
        );
    }

    #[rstest]
    #[case::short(Damage::Truncate, FormatError::InvalidFractalHeapGeometry)]
    #[case::long(Damage::Extend, FormatError::InvalidFractalHeapGeometry)]
    #[case::owner(
        Damage::Byte(OWNER_AT, 97),
        FormatError::InvalidFractalHeapBlockOwner { expected: 96, actual: 97 }
    )]
    #[case::offset(
        Damage::Byte(BLOCK_OFFSET_AT, 1),
        FormatError::InvalidFractalHeapBlockOffset { expected: 0, actual: 1 }
    )]
    fn decoded_blocks_reject_a_wrong_length_or_prefix(
        #[case] damage: Damage,
        #[case] expected: FormatError,
    ) {
        let heap = heap(OffsetWidth::Eight, 0, true, true);
        let reference = direct_root(&heap).at(PhysicalAddress::new(768)).unwrap();
        let mut decoded = direct_bytes(&heap);
        match damage {
            Damage::Truncate => {
                decoded.pop();
            }
            Damage::Extend => decoded.push(0),
            Damage::Byte(at, value) => decoded[at] = value,
        }
        assert_eq!(
            reference.parse_decoded_checked(&decoded).unwrap_err(),
            expected
        );
        let data_at = (heap.prefix_len() + CHECKSUM_LEN).to_u64();
        assert_eq!(
            reference
                .parse_decoded_object(&decoded, data_at, 3)
                .unwrap_err(),
            expected
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum Damage {
        Byte(usize, u8),
        Extend,
        Truncate,
    }

    #[test]
    fn a_decoded_block_checksum_is_validated_by_the_checked_parse_alone() {
        let heap = heap(OffsetWidth::Eight, 0, true, true);
        let reference = direct_root(&heap).at(PhysicalAddress::new(768)).unwrap();
        let mut decoded = direct_bytes(&heap);
        let checksum_at = heap.prefix_len();
        let stored = u32::from_le_bytes(
            decoded[checksum_at..checksum_at + CHECKSUM_LEN]
                .try_into()
                .unwrap(),
        );
        *decoded.last_mut().unwrap() ^= 1;
        let mut zeroed = decoded.clone();
        zeroed[checksum_at..checksum_at + CHECKSUM_LEN].fill(0);
        assert_eq!(
            reference.parse_decoded_checked(&decoded).unwrap_err(),
            FormatError::ChecksumMismatch {
                expected: stored,
                computed: checksum::jenkins_lookup3(&zeroed)
            }
        );
        let data_at = (checksum_at + CHECKSUM_LEN).to_u64();
        assert_eq!(
            reference
                .parse_decoded_object(&decoded, data_at, 3)
                .unwrap(),
            &[0x5A; 3]
        );
    }

    #[rstest]
    #[case::checksum_field(ObjectBounds::ChecksumField)]
    #[case::past_the_block(ObjectBounds::PastTheBlock)]
    #[case::overflow(ObjectBounds::Overflow)]
    fn decoded_objects_reject_invalid_bounds(#[case] bounds: ObjectBounds) {
        let heap = heap(OffsetWidth::Eight, 0, true, true);
        let reference = direct_root(&heap).at(PhysicalAddress::new(768)).unwrap();
        let decoded = direct_bytes(&heap);
        let (offset, length) = match bounds {
            ObjectBounds::ChecksumField => (heap.prefix_len().to_u64(), 3),
            ObjectBounds::Overflow => (u64::MAX, 1),
            ObjectBounds::PastTheBlock => (decoded.len().to_u64() - 2, 3),
        };
        assert_eq!(
            reference
                .parse_decoded_object(&decoded, offset, length)
                .unwrap_err(),
            FormatError::InvalidFractalHeapGeometry
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum ObjectBounds {
        ChecksumField,
        Overflow,
        PastTheBlock,
    }

    #[rstest]
    #[case::size(8, 0, false)]
    #[case::mask(16, 2, false)]
    #[case::undefined_address(0, 0xFF, true)]
    fn filtered_children_reject_invalid_encoded_geometry(
        #[case] at: usize,
        #[case] value: u8,
        #[case] undefined: bool,
    ) {
        let heap = heap(OffsetWidth::Eight, 1, false, true);
        let mut bytes = indirect_bytes(&heap, &[(0, 0, 512)]);
        if undefined {
            bytes[heap.prefix_len()
                ..heap.prefix_len() + usize::from(heap.layout.offset_width().get())]
                .fill(value);
        } else {
            bytes[heap.prefix_len() + at] = value;
        }
        let checksum_at = bytes.len() - 4;
        let computed = checksum::jenkins_lookup3(&bytes[..checksum_at]);
        bytes[checksum_at..].copy_from_slice(&computed.to_le_bytes());
        let view = indirect_root(&heap)
            .at(PhysicalAddress::new(256))
            .unwrap()
            .parse(region(256, &bytes))
            .unwrap();
        assert_eq!(
            view.child(0, 0).unwrap_err(),
            FormatError::InvalidFractalHeapGeometry
        );
        assert_eq!(
            view.children().collect::<Result<Vec<_>, _>>().unwrap_err(),
            FormatError::InvalidFractalHeapGeometry
        );
    }

    #[test]
    fn absent_filtered_children_accept_unused_mask_bits() {
        let heap = heap(OffsetWidth::Eight, 1, false, true);
        let mut bytes = indirect_bytes(&heap, &[(0, 1, 512)]);
        let mask_at = heap.prefix_len()
            + usize::from(heap.layout.offset_width().get())
            + usize::from(heap.layout.length_width().get());
        bytes[mask_at..mask_at + FILTER_MASK_LEN].copy_from_slice(&u32::MAX.to_le_bytes());
        let checksum_at = bytes.len() - CHECKSUM_LEN;
        let computed = checksum::jenkins_lookup3(&bytes[..checksum_at]);
        bytes[checksum_at..].copy_from_slice(&computed.to_le_bytes());
        let view = indirect_root(&heap)
            .at(PhysicalAddress::new(256))
            .unwrap()
            .parse(region(256, &bytes))
            .unwrap();
        assert_eq!(view.child(0, 0).unwrap(), None);
        assert_eq!(
            view.children()
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn physical_range_overflow_is_distinct_from_block_geometry() {
        let heap = heap(OffsetWidth::Eight, 0, false, false);
        assert_eq!(
            direct_root(&heap)
                .at(PhysicalAddress::new(u64::MAX - 127))
                .unwrap_err(),
            FormatError::OffsetOverflow {
                offset: u64::MAX - 127,
                length: 128
            }
        );
    }

    #[cfg(feature = "checksum")]
    #[rstest]
    #[case::direct(false)]
    #[case::indirect(true)]
    fn complete_block_views_reject_bad_checksums(#[case] indirect: bool) {
        let heap = heap(
            OffsetWidth::Eight,
            if indirect { 1 } else { 0 },
            true,
            false,
        );
        let mut bytes = if indirect {
            indirect_bytes(&heap, &[])
        } else {
            direct_bytes(&heap)
        };
        let checksum_at = if indirect {
            bytes.len() - 4
        } else {
            heap.prefix_len()
        };
        let stored = u32::from_le_bytes(bytes[checksum_at..checksum_at + 4].try_into().unwrap());
        bytes[checksum_at] ^= 1;
        let expected = stored ^ 1;
        let computed = if indirect {
            checksum::jenkins_lookup3(&bytes[..checksum_at])
        } else {
            let mut zeroed = bytes.clone();
            zeroed[checksum_at..checksum_at + 4].fill(0);
            checksum::jenkins_lookup3(&zeroed)
        };
        let error = if indirect {
            indirect_root(&heap)
                .at(PhysicalAddress::new(256))
                .unwrap()
                .parse(region(256, &bytes))
                .unwrap_err()
        } else {
            direct_root(&heap)
                .at(PhysicalAddress::new(256))
                .unwrap()
                .parse(region(256, &bytes))
                .unwrap_err()
        };
        assert_eq!(error, FormatError::ChecksumMismatch { expected, computed });
    }
    #[rstest]
    #[case::two(OffsetWidth::Two)]
    #[case::four(OffsetWidth::Four)]
    #[case::eight(OffsetWidth::Eight)]
    fn empty_managed_heaps_can_still_contain_huge_objects(#[case] offsets: OffsetWidth) {
        let encoded = fractal_heap::Header::new(256).build(Widths::EIGHT);
        let mut header =
            FractalHeapHeader::parse_bytes(&encoded, OffsetWidth::Eight, LengthWidth::Eight)
                .unwrap();
        header.root_block_address = StoredAddress::undefined(offsets);
        header.huge_objects_count = 3;
        header.huge_objects_size = 300_000;
        let heap = FractalHeapBlockLayout::new(
            &header,
            offsets,
            LengthWidth::Eight,
            StoredAddress::new(96),
        )
        .unwrap();
        assert_eq!(heap.root().unwrap(), None);
    }

    #[rstest]
    #[case::undefined_owner(0, 0xFFFF)]
    #[case::unrepresentable_owner(0, 0x1_0000)]
    #[case::unrepresentable_root(1, 96)]
    #[case::invalid_root_rows(2, 96)]
    #[case::invalid_table_width(3, 96)]
    #[case::zero_encoded_root(4, 96)]
    #[case::invalid_root_mask(5, 96)]
    #[case::small_direct_root(6, 96)]
    #[case::filter_count(7, 96)]
    fn block_layout_rejects_invalid_header_geometry(#[case] mutation: u8, #[case] owner: u64) {
        let encoded = fractal_heap::Header::new(256).build(Widths::EIGHT);
        let mut header =
            FractalHeapHeader::parse_bytes(&encoded, OffsetWidth::Eight, LengthWidth::Eight)
                .unwrap();
        match mutation {
            0 => {}
            1 => header.root_block_address = StoredAddress::new(0x1_0000),
            2 => header.current_rows_in_root_indirect_block = 9,
            3 => header.table_width = 3,
            4 | 5 | 7 => {
                header.filtering = FractalHeapFiltering::Filtered {
                    root_direct_block_size: if mutation == 4 { 0 } else { 47 },
                    root_filter_mask: if mutation == 5 { 2 } else { 0 },
                    pipeline: FilterPipeline {
                        version: 2,
                        filters: vec![
                            FilterDescription {
                                filter_id: 1,
                                name: None,
                                flags: 1,
                                client_data: vec![6],
                            };
                            if mutation == 7 { 33 } else { 1 }
                        ],
                    },
                }
            }
            6 => {
                header.starting_block_size = 8;
            }
            _ => panic!("invalid test mutation"),
        }
        assert_eq!(
            FractalHeapBlockLayout::new(
                &header,
                OffsetWidth::Two,
                LengthWidth::Eight,
                StoredAddress::new(owner)
            )
            .unwrap_err(),
            FormatError::InvalidFractalHeapGeometry
        );
    }

    #[rstest]
    #[case::two(LengthWidth::Two)]
    #[case::four(LengthWidth::Four)]
    #[case::eight(LengthWidth::Eight)]
    fn filtered_entry_lengths_use_the_superblock_width(#[case] lengths: LengthWidth) {
        let mut heap = heap(OffsetWidth::Four, 2, false, true);
        let encoded = fractal_heap::Header::new(256).build(Widths::EIGHT);
        let mut header =
            FractalHeapHeader::parse_bytes(&encoded, OffsetWidth::Eight, LengthWidth::Eight)
                .unwrap();
        header.filtering = heap.filtering.clone();
        heap.layout = FractalHeapLayout::new(&header, OffsetWidth::Four, lengths).unwrap();
        let bytes = indirect_bytes(&heap, &[(1, 2, 700)]);
        let view = indirect_root(&heap)
            .at(PhysicalAddress::new(768))
            .unwrap()
            .parse(region(768, &bytes))
            .unwrap();
        let Some(FractalHeapBlock::Direct(child)) = view.child(1, 2).unwrap() else {
            panic!("expected filtered direct child");
        };
        assert_eq!(child.address().get(), 700);
        assert_eq!(
            child
                .at(PhysicalAddress::new(1212))
                .unwrap()
                .range()
                .length(),
            47
        );
        assert_eq!(child.filter_mask(), Some(1));
    }

    #[test]
    fn supplied_larger_regions_are_borrowed_at_the_required_range() {
        let heap = heap(OffsetWidth::Eight, 2, false, false);
        let bytes = indirect_bytes(&heap, &[(0, 0, 512)]);
        let mut image = vec![0xFF; 9];
        image.extend_from_slice(&bytes);
        image.extend_from_slice(&[0xFF; 7]);
        let reference = indirect_root(&heap).at(PhysicalAddress::new(521)).unwrap();
        let view = reference.parse(region(512, &image)).unwrap();
        assert_eq!(view.bytes(), bytes);
        assert_eq!(view.bytes().as_ptr(), image[9..].as_ptr());
        let error = reference.parse(region(522, &bytes)).unwrap_err();
        assert_eq!(
            error,
            FormatError::InsufficientRegion {
                required_start: 521,
                required_length: bytes.len() as u64,
                provided_start: 522,
                provided_length: bytes.len() as u64,
            }
        );
    }

    // The fixture heaps store eight-byte offsets, so the block offset follows an eight-byte owner.
    const OWNER_AT: usize = SIGNATURE_LEN + 1;
    const BLOCK_OFFSET_AT: usize = OWNER_AT + 8;
}
