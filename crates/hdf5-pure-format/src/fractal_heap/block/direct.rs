use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;

use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum;
use crate::convert::{self, Narrow};
use crate::error::FormatError;
use crate::filter_pipeline::FilterPipeline;
use crate::fractal_heap::FractalHeapFiltering;

use super::{CHECKSUM_LEN, DIRECT_SIGNATURE, FractalHeapBlockLayout};

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
        let FractalHeapFiltering::Filtered { pipeline, .. } = self.heap.filtering() else {
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
    /// # let blocks = FractalHeapBlockLayout::new(
    /// #     &header, OffsetWidth::Eight, LengthWidth::Eight,
    /// #     base.relative(PhysicalAddress::new(608))?,
    /// # )?;
    /// # let Some(FractalHeapBlock::Direct(block)) = blocks.root()? else { panic!("expected direct fixture root") };
    /// # let reference = block.at(base.resolve(block.address())?)?;
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
        let stored_checksum = if self.block.heap.checksummed() {
            Some(u32::from_le_bytes(*bytes::array_at(
                prefix,
                self.block.heap.prefix_len(),
            )?))
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
                + if self.block.heap.checksummed() {
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
        let data_at = if self.block.heap.checksummed() {
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
            + if self.block.heap.checksummed() {
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

/// The decoded bytes and object data of a structurally checked direct block.
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

impl FractalHeapBlockLayout {
    pub(super) fn direct(
        &self,
        address: StoredAddress,
        heap_offset: u64,
        decoded_size: u64,
        encoded_size: u64,
        filter_mask: Option<u32>,
    ) -> Result<FractalHeapDirectBlock<'_>, FormatError> {
        self.contains(heap_offset, decoded_size)?;
        let overhead = self.prefix_len() + if self.checksummed() { CHECKSUM_LEN } else { 0 };
        if decoded_size < overhead as u64 || encoded_size == 0 {
            return Err(FormatError::InvalidFractalHeapGeometry);
        }
        if let (Some(mask), FractalHeapFiltering::Filtered { pipeline, .. }) =
            (filter_mask, self.filtering())
        {
            let count = pipeline.filters().len();
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
}
