use alloc::vec::Vec;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;

use crate::address::StoredAddress;
use crate::bytes;
use crate::checksum;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::filter_pipeline::FilterPipeline;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// A fractal heap header, signature `FRHP`: the parameters of the heap's doubling table, the
/// address of its root block, and the count of its managed objects.
///
/// A group in dense storage keeps its links in a fractal heap, an object in dense storage keeps
/// its attributes in one, and a file that shares messages keeps the messages of each index in one.
/// [`parse`](Self::parse) accepts version 0 and retains the optional root filter fields and filter
/// pipeline. [`serialize`](Self::serialize) writes headers with
/// [`FractalHeapFiltering::Unfiltered`].
///
/// The header is defined in "Fractal Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FractalHeapHeader {
    /// The length in bytes of the heap's IDs.
    pub heap_id_length: u16,
    /// The heap status flags: bit 0 is set once the huge object IDs have wrapped around, and bit
    /// 1, [`FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED`], where the direct blocks store a checksum.
    pub flags: u8,
    /// The size limit in bytes for a managed object. The heap stores a larger object as a huge
    /// object, outside its blocks.
    pub max_managed_object_size: u32,
    /// The ID value of the next huge object the heap stores.
    pub next_huge_object_id: u64,
    /// The address of the version 2 B-tree that indexes the heap's huge objects, or the undefined
    /// address where the heap has no such tree.
    pub btree_huge_objects_address: StoredAddress,
    /// The free space in bytes in the managed direct blocks.
    pub free_space_in_managed_blocks: u64,
    /// The address of the free-space manager of the managed blocks.
    pub managed_block_free_space_manager_address: StoredAddress,
    /// The size in bytes of the managed space of the heap, the upper bound of its heap offsets.
    pub managed_space: u64,
    /// The size in bytes of the managed space allocated to direct blocks, less than
    /// [`managed_space`](Self::managed_space) where a direct block is not allocated.
    pub allocated_managed_space: u64,
    /// The heap offset of the next direct block the heap allocates.
    pub direct_block_allocation_iterator_offset: u64,
    /// The number of managed objects in the heap.
    pub managed_objects_count: u64,
    /// The total size in bytes of the huge objects of the heap.
    pub huge_objects_size: u64,
    /// The number of huge objects in the heap.
    pub huge_objects_count: u64,
    /// The total size in bytes of the tiny objects the heap IDs store.
    pub tiny_objects_size: u64,
    /// The number of tiny objects the heap IDs store.
    pub tiny_objects_count: u64,
    /// The number of blocks in each row of the doubling table.
    pub table_width: u16,
    /// The size in bytes of the blocks in the first two rows of the doubling table.
    pub starting_block_size: u64,
    /// The size in bytes of the largest direct block. A row of larger blocks holds indirect
    /// blocks.
    pub max_direct_block_size: u64,
    /// The maximum number of bits in a managed heap offset.
    pub max_heap_size: u16,
    /// The number of rows the root indirect block starts with, or 0 for the most rows the managed
    /// space of the heap needs.
    ///
    /// The boundary between the rows of direct blocks and the rows of indirect blocks follows from
    /// [`starting_block_size`](Self::starting_block_size) and
    /// [`max_direct_block_size`](Self::max_direct_block_size) alone.
    pub start_root_rows: u16,
    /// The address of the root block, or the undefined address for an empty heap.
    /// An allocated root is a direct block when
    /// [`current_rows_in_root_indirect_block`](Self::current_rows_in_root_indirect_block) is 0 and
    /// an indirect block otherwise.
    pub root_block_address: StoredAddress,
    /// The number of rows in the root indirect block, 0 where the root is a direct block.
    pub current_rows_in_root_indirect_block: u16,
    /// The optional root filter fields and decoded filter pipeline.
    pub filtering: FractalHeapFiltering,
}

impl FractalHeapHeader {
    /// Parses the fractal heap header at `offset` in `file_data`.
    ///
    /// `offset_size` and `length_size` are the superblock's "Size of Offsets" and "Size of
    /// Lengths" bytes. Parses the complete header using [`parse_bytes`](Self::parse_bytes).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if a width
    /// is not 2, 4, or 8, [`FormatError::UnexpectedEof`] if `offset` is beyond the end of
    /// `file_data`, and the errors [`parse_bytes`](Self::parse_bytes) returns.
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<FractalHeapHeader, FormatError> {
        let offsets = OffsetWidth::try_from(offset_size)?;
        let lengths = LengthWidth::try_from(length_size)?;
        let bytes = file_data.get(offset..).ok_or(FormatError::UnexpectedEof {
            expected: offset,
            available: file_data.len(),
        })?;
        Self::parse_bytes(bytes, offsets, lengths)
    }

    /// Parses the complete version 0 header at the start of `data`.
    ///
    /// `offsets` and `lengths` are the file's address and length widths. The parser acquires the
    /// complete fixed prefix before checking the signature and version. It uses the prefix to
    /// determine the complete header length and ignores bytes after the checksum.
    ///
    /// With the `checksum` feature, the parser verifies the checksum over all preceding header
    /// bytes before decoding any body fields, including the optional root filter fields and
    /// filter pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `data` is shorter than the fixed prefix or the
    /// complete header length derived from it. Propagates the other errors of
    /// [`FractalHeapHeaderFrame::parse`].
    ///
    /// With the `checksum` feature, returns [`FormatError::ChecksumMismatch`] if the stored
    /// checksum differs from the computed one.
    ///
    /// Returns [`FormatError::InvalidFilterPipelineVersion`] if the filter pipeline version is
    /// not 1 or 2, [`FormatError::InvalidFilterPipelineField`] if a pipeline field exceeds its
    /// format limit, and [`FormatError::InvalidFilterName`] if a stored filter name is malformed.
    /// Returns [`FormatError::UnexpectedEof`] if a field within the encoded pipeline is incomplete.
    pub fn parse_bytes(
        data: &[u8],
        offsets: OffsetWidth,
        lengths: LengthWidth,
    ) -> Result<Self, FormatError> {
        let prefix = FractalHeapHeaderPrefix::parse(data, offsets, lengths)?;
        let data = data
            .get(..prefix.frame.encoded_len())
            .ok_or(FormatError::UnexpectedEof {
                expected: prefix.frame.encoded_len(),
                available: data.len(),
            })?;
        Self::parse_framed(data, prefix)
    }

    /// Decodes the header body at the field widths used to frame it.
    ///
    /// The caller supplies exactly the header bytes from the signature through the checksum.
    /// With the `checksum` feature, verifies the checksum before reading the body.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ChecksumMismatch`] if the `checksum` feature is enabled and the
    /// checksum differs from the computed one. Propagates the errors of
    /// [`FractalHeapFiltering::parse`].
    fn parse_framed(data: &[u8], prefix: FractalHeapHeaderPrefix) -> Result<Self, FormatError> {
        let FractalHeapHeaderPrefix {
            heap_id_length,
            filter_encoded_len,
            frame: _,
            widths: FormatWidths { offsets, lengths },
        } = prefix;
        let (payload, _) = data.split_last_chunk::<FRACTAL_HEAP_CHECKSUM_LEN>().ok_or(
            FormatError::UnexpectedEof {
                expected: FRACTAL_HEAP_CHECKSUM_LEN,
                available: data.len(),
            },
        )?;
        checksum::verify_trailing(data)?;
        let mut cursor =
            bytes::ByteCursor::with_position(payload, FractalHeapHeaderFrame::PREFIX_LEN)?;
        let flags = cursor.u8()?;
        let max_managed_object_size = cursor.u32()?;
        let next_huge_object_id = cursor.length(lengths)?;
        let btree_huge_objects_address = cursor.address(offsets)?;
        let free_space_in_managed_blocks = cursor.length(lengths)?;
        let managed_block_free_space_manager_address = cursor.address(offsets)?;
        let managed_space = cursor.length(lengths)?;
        let allocated_managed_space = cursor.length(lengths)?;
        let direct_block_allocation_iterator_offset = cursor.length(lengths)?;
        let managed_objects_count = cursor.length(lengths)?;
        let huge_objects_size = cursor.length(lengths)?;
        let huge_objects_count = cursor.length(lengths)?;
        let tiny_objects_size = cursor.length(lengths)?;
        let tiny_objects_count = cursor.length(lengths)?;
        let table_width = cursor.u16()?;
        let starting_block_size = cursor.length(lengths)?;
        let max_direct_block_size = cursor.length(lengths)?;
        let max_heap_size = cursor.u16()?;
        let start_root_rows = cursor.u16()?;
        let root_block_address = cursor.address(offsets)?;
        let current_rows_in_root_indirect_block = cursor.u16()?;
        let filtering = FractalHeapFiltering::parse(&mut cursor, lengths, filter_encoded_len)?;

        Ok(FractalHeapHeader {
            heap_id_length,
            flags,
            max_managed_object_size,
            next_huge_object_id,
            btree_huge_objects_address,
            free_space_in_managed_blocks,
            managed_block_free_space_manager_address,
            managed_space,
            allocated_managed_space,
            direct_block_allocation_iterator_offset,
            managed_objects_count,
            huge_objects_size,
            huge_objects_count,
            tiny_objects_size,
            tiny_objects_count,
            table_width,
            starting_block_size,
            max_direct_block_size,
            max_heap_size,
            start_root_rows,
            root_block_address,
            current_rows_in_root_indirect_block,
            filtering,
        })
    }

    /// Returns the exact encoded size of an unfiltered fractal heap header in bytes.
    ///
    /// The size runs from the `FRHP` signature through the trailing checksum.
    pub const fn serialized_size(offset_width: OffsetWidth, length_width: LengthWidth) -> usize {
        let os = offset_width.get() as usize;
        let ls = length_width.get() as usize;
        FRACTAL_HEAP_SIGNATURE.len() + 1 + 2 + 2 + 1 + 4 + 12 * ls + 3 * os + 2 + 2 + 2 + 2 + 4
    }

    /// Serializes an unfiltered header from its signature through its checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedFilteredHeapObject`] if
    /// [`filtering`](Self::filtering) is [`FractalHeapFiltering::Filtered`],
    /// [`FormatError::AddressTooLarge`] if an address exceeds `offset_width` and is not
    /// [`u64::MAX`], or [`FormatError::LengthTooLarge`] if a length field exceeds `length_width`.
    pub fn serialize(
        &self,
        offset_width: OffsetWidth,
        length_width: LengthWidth,
    ) -> Result<Vec<u8>, FormatError> {
        let Self {
            heap_id_length,
            ref filtering,
            flags,
            max_managed_object_size,
            next_huge_object_id,
            btree_huge_objects_address,
            free_space_in_managed_blocks,
            managed_block_free_space_manager_address,
            managed_space,
            allocated_managed_space,
            direct_block_allocation_iterator_offset,
            managed_objects_count,
            huge_objects_size,
            huge_objects_count,
            tiny_objects_size,
            tiny_objects_count,
            table_width,
            starting_block_size,
            max_direct_block_size,
            max_heap_size,
            start_root_rows,
            root_block_address,
            current_rows_in_root_indirect_block,
        } = *self;
        if filtering.is_filtered() {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }
        let mut buf = Vec::with_capacity(Self::serialized_size(offset_width, length_width));
        buf.extend_from_slice(&FRACTAL_HEAP_SIGNATURE);
        buf.push(FRACTAL_HEAP_VERSION);
        buf.extend_from_slice(&heap_id_length.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.push(flags);
        buf.extend_from_slice(&max_managed_object_size.to_le_bytes());
        bytes::try_write_length(&mut buf, next_huge_object_id, length_width)?;
        bytes::try_write_offset(&mut buf, btree_huge_objects_address.get(), offset_width)?;
        bytes::try_write_length(&mut buf, free_space_in_managed_blocks, length_width)?;
        bytes::try_write_offset(
            &mut buf,
            managed_block_free_space_manager_address.get(),
            offset_width,
        )?;
        for length in [
            managed_space,
            allocated_managed_space,
            direct_block_allocation_iterator_offset,
            managed_objects_count,
            huge_objects_size,
            huge_objects_count,
            tiny_objects_size,
            tiny_objects_count,
        ] {
            bytes::try_write_length(&mut buf, length, length_width)?;
        }
        buf.extend_from_slice(&table_width.to_le_bytes());
        bytes::try_write_length(&mut buf, starting_block_size, length_width)?;
        bytes::try_write_length(&mut buf, max_direct_block_size, length_width)?;
        buf.extend_from_slice(&max_heap_size.to_le_bytes());
        buf.extend_from_slice(&start_root_rows.to_le_bytes());
        bytes::try_write_offset(&mut buf, root_block_address.get(), offset_width)?;
        buf.extend_from_slice(&current_rows_in_root_indirect_block.to_le_bytes());
        let checksum = crate::checksum::jenkins_lookup3(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        Ok(buf)
    }
}

/// The optional filter information of a fractal heap header.
///
/// A [`FractalHeapHeader`] stores [`Filtered`](Self::Filtered) when its encoded filter information
/// has a nonzero length, and [`Unfiltered`](Self::Unfiltered) otherwise. The fields are defined in
/// "Fractal Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FractalHeapFiltering {
    /// The header stores the root filter fields and an encoded filter pipeline.
    Filtered {
        /// The size in bytes of the root direct block after filtering.
        root_direct_block_size: u64,
        /// Filters in `pipeline` whose index bits are set are skipped for the root direct block.
        root_filter_mask: u32,
        /// The ordered filter descriptions for direct blocks and huge objects.
        pipeline: FilterPipeline,
    },
    /// The header omits the optional root filter fields and filter pipeline.
    Unfiltered,
}

impl FractalHeapFiltering {
    /// Parses optional root filter fields and `encoded_len` bytes of filter information.
    ///
    /// The parser reads the full embedded pipeline body before parsing its fields.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the root filter fields or declared pipeline
    /// body are incomplete. Converts errors from [`FilterPipeline::parse`] to [`FormatError`].
    fn parse(
        cursor: &mut bytes::ByteCursor<'_>,
        lengths: LengthWidth,
        encoded_len: u16,
    ) -> Result<Self, FormatError> {
        if encoded_len == 0 {
            return Ok(Self::Unfiltered);
        }
        let root_direct_block_size = cursor.length(lengths)?;
        let root_filter_mask = cursor.u32()?;
        let pipeline_bytes = cursor.take(usize::from(encoded_len))?;
        Ok(Self::Filtered {
            root_direct_block_size,
            root_filter_mask,
            pipeline: FilterPipeline::parse(pipeline_bytes)?,
        })
    }

    /// Returns `true` if the header contains encoded filter information.
    pub const fn is_filtered(&self) -> bool {
        matches!(self, Self::Filtered { .. })
    }
}

/// The physical position and field widths needed to frame a version 0 fractal heap header.
///
/// The caller supplies the fixed prefix in a [`FileRegion`] to [`parse`](Self::parse), then
/// supplies the complete header to the returned [`FractalHeapHeaderRef`].
///
/// The header is defined in "Fractal Heap" of the [format specification, version 4.0][spec].
///
/// # Examples
///
/// The caller reads each requested range before constructing its [`FileRegion`]. Here,
/// `bytes_in_range` borrows those bytes from an in-memory image.
///
/// ```
/// # use hdf5_pure_core::__private::{FileRange, FileRegion};
/// # use hdf5_pure_core::PhysicalAddress;
/// # use hdf5_pure_format::__private::{FractalHeapHeaderLocator, LengthWidth, OffsetWidth};
/// # use hdf5_pure_format::FormatError;
/// # use test_util::fractal_heap;
/// # use test_util::widths::Widths;
/// # let encoded = fractal_heap::Header::new(256).build(Widths::EIGHT);
/// # let image = FileRegion::new(
/// #     FileRange::new(PhysicalAddress::new(512), encoded.len() as u64)?,
/// #     &encoded,
/// # )?;
/// # let bytes_in_range = |range| image.subregion(range).map(FileRegion::bytes);
/// let locator = FractalHeapHeaderLocator::new(
///     PhysicalAddress::new(512),
///     OffsetWidth::Eight,
///     LengthWidth::Eight,
/// );
/// let prefix_range = locator.parse_range()?;
/// let prefix_bytes = bytes_in_range(prefix_range)?;
/// let header_ref = locator.parse(FileRegion::new(prefix_range, prefix_bytes)?)?;
///
/// let header_range = header_ref.parse_range();
/// let header_bytes = bytes_in_range(header_range)?;
/// let header = header_ref.parse(FileRegion::new(header_range, header_bytes)?)?;
/// assert_eq!(header.heap_id_length, 7);
/// # Ok::<(), FormatError>(())
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FractalHeapHeaderLocator {
    address: PhysicalAddress,
    offsets: OffsetWidth,
    lengths: LengthWidth,
}

impl FractalHeapHeaderLocator {
    /// Creates a locator at an address resolved against the opened image's effective base.
    ///
    /// `offsets` and `lengths` are the superblock's "Size of Offsets" and "Size of Lengths" fields.
    pub fn new(address: PhysicalAddress, offsets: OffsetWidth, lengths: LengthWidth) -> Self {
        Self {
            address,
            offsets,
            lengths,
        }
    }

    /// Parses the fixed prefix to derive the complete header range.
    ///
    /// Bytes in `region` outside [`parse_range`](Self::parse_range) are ignored.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain the prefix, and
    /// [`FormatError::OffsetOverflow`] if the prefix or complete header range exceeds [`u64::MAX`].
    /// Returns the framing errors of [`FractalHeapHeaderFrame::parse`].
    pub fn parse(&self, region: FileRegion<'_>) -> Result<FractalHeapHeaderRef, FormatError> {
        let Self {
            address,
            offsets,
            lengths,
        } = *self;
        let frame = FractalHeapHeaderFrame::parse(
            region.subregion(self.parse_range()?)?.bytes(),
            offsets,
            lengths,
        )?;
        Ok(FractalHeapHeaderRef {
            range: FileRange::new(address, frame.encoded_len() as u64)?,
            offsets,
            lengths,
        })
    }

    /// Returns the physical range of the fixed prefix needed to frame the header.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the range end exceeds [`u64::MAX`].
    pub fn parse_range(&self) -> Result<FileRange, FormatError> {
        FractalHeapHeaderFrame::prefix_range(self.address)
    }
}

/// The complete header range derived from a fractal heap's fixed prefix.
///
/// The range includes the checksum and any optional filter fields. The complete header is
/// validated by [`parse`](Self::parse), which reads its framing fields again from the supplied
/// [`FileRegion`].
///
/// See [`FractalHeapHeaderLocator`] for an example of parsing the prefix and complete header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FractalHeapHeaderRef {
    range: FileRange,
    offsets: OffsetWidth,
    lengths: LengthWidth,
}

impl FractalHeapHeaderRef {
    /// Parses the complete header in the reference's physical range.
    ///
    /// The parser acquires the fixed prefix first, checks the signature and version, and derives
    /// the header length from the prefix. It ignores bytes after the checksum. With the
    /// `checksum` feature, it verifies the checksum before decoding the body.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain the fixed prefix
    /// or [`parse_range`](Self::parse_range), or if the length derived from the prefix exceeds
    /// that range. Returns [`FormatError::OffsetOverflow`] if the derived range end exceeds
    /// [`u64::MAX`]. Propagates the framing, checksum, and body errors of
    /// [`FractalHeapHeader::parse_bytes`].
    pub fn parse(&self, region: FileRegion<'_>) -> Result<FractalHeapHeader, FormatError> {
        let Self {
            range,
            offsets,
            lengths,
        } = *self;
        let prefix_range = FractalHeapHeaderFrame::prefix_range(range.start())?;
        let prefix = FractalHeapHeaderPrefix::parse(
            region.subregion(prefix_range)?.bytes(),
            offsets,
            lengths,
        )?;
        let required = FileRange::new(range.start(), prefix.frame.encoded_len() as u64)?;
        let bytes = region.subregion(range)?.subregion(required)?.bytes();
        FractalHeapHeader::parse_framed(bytes, prefix)
    }

    /// Returns the complete physical header range derived from the prefix.
    pub const fn range(&self) -> FileRange {
        self.range
    }

    /// Returns the physical range required to parse the header, equal to [`range`](Self::range).
    pub const fn parse_range(&self) -> FileRange {
        self.range
    }
}

/// The complete encoded length of a version 0 fractal heap header.
///
/// The length includes the checksum and, for a filtered heap, the root block's filtered size
/// and filter mask, and the encoded filter pipeline. [`parse`](Self::parse) derives it from the
/// fixed prefix after checking the signature and version.
///
/// The header is defined in "Fractal Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FractalHeapHeaderFrame {
    encoded_len: usize,
}

impl FractalHeapHeaderFrame {
    /// Computes the complete header length from the fixed prefix at the start of `bytes`.
    ///
    /// `offsets` and `lengths` are the superblock's "Size of Offsets" and "Size of Lengths"
    /// fields. Bytes after [`PREFIX_LEN`](Self::PREFIX_LEN) are ignored.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `bytes` is shorter than
    /// [`PREFIX_LEN`](Self::PREFIX_LEN). After acquiring the prefix, returns
    /// [`FormatError::InvalidFractalHeapSignature`] if the signature is not `FRHP`, or
    /// [`FormatError::InvalidFractalHeapVersion`] if the version is not 0.
    ///
    /// Returns [`FormatError::Internal`] if the length arithmetic overflows, and
    /// [`FormatError::ValueTooLargeForPlatform`] if the length exceeds [`usize::MAX`].
    pub fn parse(
        bytes: &[u8],
        offsets: OffsetWidth,
        lengths: LengthWidth,
    ) -> Result<Self, FormatError> {
        Ok(FractalHeapHeaderPrefix::parse(bytes, offsets, lengths)?.frame)
    }

    /// Returns the header length in bytes, including the checksum and any filter fields.
    pub const fn encoded_len(&self) -> usize {
        self.encoded_len
    }

    fn prefix_range(address: PhysicalAddress) -> Result<FileRange, FormatError> {
        FileRange::new(address, Self::PREFIX_LEN as u64)
    }

    fn new(
        offsets: OffsetWidth,
        lengths: LengthWidth,
        filter_encoded_len: u16,
    ) -> Result<Self, FormatError> {
        let overflow = || FormatError::Internal("fractal heap header length exceeds u64".into());
        let length_fields_len = FRACTAL_HEAP_HEADER_LENGTH_FIELDS
            .checked_mul(u64::from(lengths.get()))
            .ok_or_else(overflow)?;
        let address_fields_len = FRACTAL_HEAP_HEADER_ADDRESS_FIELDS
            .checked_mul(u64::from(offsets.get()))
            .ok_or_else(overflow)?;
        let mut encoded_len = FRACTAL_HEAP_HEADER_FIXED_LEN
            .checked_add(length_fields_len)
            .and_then(|len| len.checked_add(address_fields_len))
            .ok_or_else(overflow)?;
        if filter_encoded_len > 0 {
            encoded_len = encoded_len
                .checked_add(u64::from(lengths.get()))
                .and_then(|len| len.checked_add(FRACTAL_HEAP_FILTER_MASK_LEN))
                .and_then(|len| len.checked_add(u64::from(filter_encoded_len)))
                .ok_or_else(overflow)?;
        }
        Ok(Self {
            encoded_len: encoded_len.to_usize()?,
        })
    }

    /// The prefix length in bytes: the signature (4), version (1), heap ID length (2), and
    /// encoded filter pipeline length (2).
    ///
    /// The prefix ends after the "I/O Filters' Encoded Length" field in "Fractal Heap" of the
    /// [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
    pub const PREFIX_LEN: usize = 9;
}

/// The heap ID and pipeline lengths from a checked version 0 header prefix.
///
/// The body parser uses the same field widths as the header length calculation.
struct FractalHeapHeaderPrefix {
    heap_id_length: u16,
    filter_encoded_len: u16,
    frame: FractalHeapHeaderFrame,
    widths: FormatWidths,
}

impl FractalHeapHeaderPrefix {
    fn parse(
        bytes: &[u8],
        offsets: OffsetWidth,
        lengths: LengthWidth,
    ) -> Result<Self, FormatError> {
        let prefix = bytes::ByteCursor::new(bytes).take(FractalHeapHeaderFrame::PREFIX_LEN)?;
        let mut prefix = bytes::ByteCursor::new(prefix);
        if prefix.array::<{ FRACTAL_HEAP_SIGNATURE.len() }>()? != FRACTAL_HEAP_SIGNATURE {
            return Err(FormatError::InvalidFractalHeapSignature);
        }
        let version = prefix.u8()?;
        if version != FRACTAL_HEAP_VERSION {
            return Err(FormatError::InvalidFractalHeapVersion(version));
        }
        let heap_id_length = prefix.u16()?;
        let filter_encoded_len = prefix.u16()?;
        Ok(Self {
            heap_id_length,
            filter_encoded_len,
            frame: FractalHeapHeaderFrame::new(offsets, lengths, filter_encoded_len)?,
            widths: FormatWidths { offsets, lengths },
        })
    }
}

/// The signature of a fractal heap header, from "Fractal Heap" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
const FRACTAL_HEAP_SIGNATURE: [u8; 4] = *b"FRHP";

/// The fractal heap header version, the one version the same section as
/// [`FRACTAL_HEAP_SIGNATURE`] defines.
const FRACTAL_HEAP_VERSION: u8 = 0;

/// The bit of the header's Flags field that is set where the direct blocks of the heap store a
/// checksum, bit 1.
///
/// The bit is defined in "Fractal Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
pub const FRACTAL_HEAP_DIRECT_BLOCKS_CHECKSUMMED: u8 = 0x02;
// The unfiltered version 0 header stores 26 fixed bytes, twelve length fields, and three addresses
// ("Fractal Heap", format specification version 4.0).
const FRACTAL_HEAP_HEADER_FIXED_LEN: u64 = 26;
const FRACTAL_HEAP_HEADER_LENGTH_FIELDS: u64 = 12;
const FRACTAL_HEAP_HEADER_ADDRESS_FIELDS: u64 = 3;

// The filtered-header extension stores a four-byte mask after the root size
// ("Fractal Heap", format specification version 4.0).
const FRACTAL_HEAP_FILTER_MASK_LEN: u64 = 4;

// "Fractal Heap", format specification version 4.0, defines a four-byte checksum after the
// optional filter information.
const FRACTAL_HEAP_CHECKSUM_LEN: usize = 4;

#[cfg(test)]
#[path = "tests/header.rs"]
mod tests;
