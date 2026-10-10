//! Parses and encodes fractal heap headers and heap IDs, derives doubling-table geometry, and
//! locates children of indirect blocks.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::bytes;
use crate::convert;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::filter_pipeline::FilterPipeline;
use crate::filter_pipeline::FilterPipelineError;
use crate::metadata_source::MetadataSource;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// The type of a fractal heap ID: where the heap stores the object the ID refers to.
///
/// Bits 4 and 5 of the first byte of a heap ID hold the type. The heap IDs are defined in "Fractal
/// Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FractalHeapIdType {
    /// The object is in a direct block of the heap.
    Managed,
    /// The object is in the file outside the blocks of the heap, and the heap ID holds its address
    /// and length or its key in the huge-object B-tree of the heap.
    Huge,
    /// The heap ID holds the object.
    Tiny,
}

impl FractalHeapIdType {
    /// Returns the type of the heap ID `id_bytes`, from bits 4 and 5 of its first byte.
    ///
    /// Bits 6 and 7 hold the version of the heap ID format, which the specification defines as 0,
    /// and the function reads the type whatever they hold.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `id_bytes` is empty, and
    /// [`FormatError::InvalidHeapIdType`] if the type bits hold 3, a type the format does not
    /// define.
    fn from_heap_id(id_bytes: &[u8]) -> Result<Self, FormatError> {
        let byte0 = *id_bytes.first().ok_or(FormatError::UnexpectedEof {
            expected: 1,
            available: 0,
        })?;
        match (byte0 >> HEAP_ID_TYPE_SHIFT) & HEAP_ID_TYPE_MASK {
            HEAP_ID_TYPE_MANAGED => Ok(FractalHeapIdType::Managed),
            HEAP_ID_TYPE_HUGE => Ok(FractalHeapIdType::Huge),
            HEAP_ID_TYPE_TINY => Ok(FractalHeapIdType::Tiny),
            other => Err(FormatError::InvalidHeapIdType(other)),
        }
    }

    /// Returns the first byte of a heap ID of this type, whose version bits are 0.
    const fn first_byte(self) -> u8 {
        let type_bits = match self {
            Self::Managed => HEAP_ID_TYPE_MANAGED,
            Self::Huge => HEAP_ID_TYPE_HUGE,
            Self::Tiny => HEAP_ID_TYPE_TINY,
        };
        type_bits << HEAP_ID_TYPE_SHIFT
    }
}

/// A child of a fractal heap indirect block: a direct block, which holds objects, or an indirect
/// block one level down.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FractalHeapChild {
    /// A direct block.
    Direct {
        /// The address of the block.
        addr: StoredAddress,
        /// The size of the block in bytes.
        block_size: u64,
        /// The heap offset the space of the block begins at.
        heap_offset: u64,
    },
    /// An indirect block.
    Indirect {
        /// The address of the block.
        addr: StoredAddress,
        /// The number of rows of the block.
        nrows: u16,
        /// The heap offset the space of the block begins at.
        heap_offset: u64,
    },
}

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
    /// The size in bytes of the largest managed object. The heap stores a larger object as a huge
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
    /// The number of bits in a heap offset, the base-2 logarithm of the size of the heap's managed
    /// space.
    pub max_heap_size: u16,
    /// The number of rows the root indirect block starts with, or 0 for the most rows the managed
    /// space of the heap needs.
    ///
    /// The boundary between the rows of direct blocks and the rows of indirect blocks follows from
    /// [`starting_block_size`](Self::starting_block_size) and
    /// [`max_direct_block_size`](Self::max_direct_block_size) alone.
    pub start_root_rows: u16,
    /// The address of the root block, a direct block where
    /// [`current_rows_in_root_indirect_block`](Self::current_rows_in_root_indirect_block) is 0 and
    /// an indirect block otherwise.
    pub root_block_address: StoredAddress,
    /// The number of rows in the root indirect block, 0 where the root is a direct block.
    pub current_rows_in_root_indirect_block: u16,
    /// The optional root filter fields and decoded filter pipeline.
    pub filtering: FractalHeapFiltering,
}

/// Reads the little-endian integer in the first 8 bytes of `payload`, or in all of it where it is
/// shorter. A huge heap ID holds its key in the huge-object B-tree this way.
fn read_var_le(payload: &[u8]) -> u64 {
    let mut value = 0u64;
    for (i, &b) in payload.iter().take(8).enumerate() {
        value |= (b as u64) << (i * 8);
    }
    value
}

/// The location of a huge object, as its heap ID gives it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HugeObjectReference {
    /// The heap ID holds the address and the length of the object.
    Inline {
        /// The address of the object.
        addr: StoredAddress,
        /// The length of the object in bytes.
        len: u64,
    },
    /// The heap ID holds this key, which the huge-object B-tree of the heap maps to the address
    /// and the length of the object.
    Indexed(u64),
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
    /// `offsets` and `lengths` are the file's address and length widths. The parser retains the
    /// optional root filter fields and decoded filter pipeline, and ignores bytes after the
    /// header's checksum. With the `checksum` feature, it verifies the checksum over all preceding
    /// header bytes, including the encoded filter pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the header is incomplete, and the other errors
    /// [`FractalHeapHeaderFrame::parse`] returns for its prefix.
    ///
    /// Returns [`FormatError::InvalidFilterPipelineVersion`] if the filter pipeline version is
    /// not 1 or 2, [`FormatError::InvalidFilterPipelineField`] if a pipeline field exceeds its
    /// format limit, and [`FormatError::InvalidFilterName`] if a stored filter name is malformed.
    ///
    /// With the `checksum` feature, returns [`FormatError::ChecksumMismatch`] if the stored
    /// checksum differs from the computed one.
    pub fn parse_bytes(
        data: &[u8],
        offsets: OffsetWidth,
        lengths: LengthWidth,
    ) -> Result<Self, FormatError> {
        let frame = FractalHeapHeaderFrame::parse(data, offsets, lengths)?;
        let data = data
            .get(..frame.encoded_len())
            .ok_or(FormatError::UnexpectedEof {
                expected: frame.encoded_len(),
                available: data.len(),
            })?;
        let mut fields = bytes::Fields::new(data, 0);
        fields.array::<{ FRACTAL_HEAP_SIGNATURE.len() }>()?;
        fields.u8()?;
        let heap_id_length = fields.u16()?;
        let filter_encoded_len = fields.u16()?;
        let flags = fields.u8()?;
        let max_managed_object_size = fields.u32()?;
        let next_huge_object_id = fields.length(lengths)?;
        let btree_huge_objects_address = fields.address(offsets)?;
        let free_space_in_managed_blocks = fields.length(lengths)?;
        let managed_block_free_space_manager_address = fields.address(offsets)?;
        let managed_space = fields.length(lengths)?;
        let allocated_managed_space = fields.length(lengths)?;
        let direct_block_allocation_iterator_offset = fields.length(lengths)?;
        let managed_objects_count = fields.length(lengths)?;
        let huge_objects_size = fields.length(lengths)?;
        let huge_objects_count = fields.length(lengths)?;
        let tiny_objects_size = fields.length(lengths)?;
        let tiny_objects_count = fields.length(lengths)?;
        let table_width = fields.u16()?;
        let starting_block_size = fields.length(lengths)?;
        let max_direct_block_size = fields.length(lengths)?;
        let max_heap_size = fields.u16()?;
        let start_root_rows = fields.u16()?;
        let root_block_address = fields.address(offsets)?;
        let current_rows_in_root_indirect_block = fields.u16()?;
        let filtering =
            FractalHeapFiltering::parse(data, lengths, fields.pos(), filter_encoded_len)?;
        let (payload, checksum) = data.split_last_chunk::<FRACTAL_HEAP_CHECKSUM_LEN>().ok_or(
            FormatError::UnexpectedEof {
                expected: FRACTAL_HEAP_CHECKSUM_LEN,
                available: data.len(),
            },
        )?;
        let stored = u32::from_le_bytes(*checksum);
        #[cfg(feature = "checksum")]
        {
            let computed = crate::checksum::jenkins_lookup3(payload);
            if computed != stored {
                return Err(FormatError::ChecksumMismatch {
                    expected: stored,
                    computed,
                });
            }
        }
        #[cfg(not(feature = "checksum"))]
        let _ = (payload, stored);

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
    /// [`filtering`](Self::filtering) is [`FractalHeapFiltering::Filtered`].
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
        bytes::write_length(&mut buf, next_huge_object_id, length_width);
        bytes::write_offset(&mut buf, btree_huge_objects_address.get(), offset_width);
        bytes::write_length(&mut buf, free_space_in_managed_blocks, length_width);
        bytes::write_offset(
            &mut buf,
            managed_block_free_space_manager_address.get(),
            offset_width,
        );
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
            bytes::write_length(&mut buf, length, length_width);
        }
        buf.extend_from_slice(&table_width.to_le_bytes());
        bytes::write_length(&mut buf, starting_block_size, length_width);
        bytes::write_length(&mut buf, max_direct_block_size, length_width);
        buf.extend_from_slice(&max_heap_size.to_le_bytes());
        buf.extend_from_slice(&start_root_rows.to_le_bytes());
        bytes::write_offset(&mut buf, root_block_address.get(), offset_width);
        buf.extend_from_slice(&current_rows_in_root_indirect_block.to_le_bytes());
        let checksum = crate::checksum::jenkins_lookup3(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        Ok(buf)
    }

    /// Returns the heap ID of the managed object of `length` bytes at heap offset `heap_offset`.
    ///
    /// After its first byte, the ID stores the offset in [`max_heap_size`](Self::max_heap_size)
    /// bits and then the length, as [`FractalHeapIdLayout::parse`] reads them.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `max_heap_size` is not a multiple of 8, or if
    /// `heap_offset` or `length` does not fit the ID. The C library stores the offset in whole
    /// bytes (`H5HF_MAN_ID_ENCODE` in `H5HFpkg.h`, HDF5 2.2.0), and
    /// [`FractalHeapIdLayout::parse`] reads it as `max_heap_size` packed bits.
    pub fn encode_managed_id(&self, heap_offset: u64, length: u64) -> Result<Vec<u8>, FormatError> {
        if self.max_heap_size % 8 != 0 {
            return Err(FormatError::Internal(format!(
                "a managed heap ID was encoded for a heap of {}-bit offsets, which fill no whole \
                 number of bytes",
                self.max_heap_size
            )));
        }
        let offset_bits = u32::from(self.max_heap_size);
        if !fits_bits(heap_offset, offset_bits) {
            return Err(FormatError::Internal(format!(
                "heap offset {heap_offset} does not fit the heap's {offset_bits}-bit offsets"
            )));
        }
        let packed = length
            .checked_shl(offset_bits)
            .filter(|&shifted| shifted >> offset_bits == length)
            .map(|shifted| shifted | heap_offset);
        self.encode_id(FractalHeapIdType::Managed, packed, "managed object length")
    }

    /// Returns the heap ID of the huge object whose key in the huge-object B-tree is `huge_id`.
    ///
    /// The huge IDs of a heap store a key where they are too short for an address of
    /// `offset_width` and a length of `length_width`, as [`FractalHeapIdLayout::parse`]
    /// reads them.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedFilteredHeapObject`] if the heap filters its objects,
    /// and [`FormatError::Internal`] if the huge IDs of the heap store an address and a length, or
    /// if `huge_id` does not fit the ID.
    pub fn encode_huge_id(
        &self,
        huge_id: u64,
        offset_width: OffsetWidth,
        length_width: LengthWidth,
    ) -> Result<Vec<u8>, FormatError> {
        if self.filtering.is_filtered() {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }
        if self.huge_ids_direct(offset_width.get(), length_width.get()) {
            return Err(FormatError::Internal(
                "a key was encoded into a heap ID that holds its object's address".into(),
            ));
        }
        self.encode_id(FractalHeapIdType::Huge, Some(huge_id), "huge object key")
    }

    /// Returns a heap ID of `id_type` whose bytes after the first store `payload` little-endian.
    ///
    /// A `payload` of `None` is one too wide for the ID, and the error names it as `what`.
    fn encode_id(
        &self,
        id_type: FractalHeapIdType,
        payload: Option<u64>,
        what: &str,
    ) -> Result<Vec<u8>, FormatError> {
        let mut id = vec![0; usize::from(self.heap_id_length)];
        let Some((first, rest)) = id.split_first_mut() else {
            return Err(FormatError::Internal(
                "a heap ID was encoded for a heap whose IDs are empty".into(),
            ));
        };
        let too_wide = || {
            FormatError::Internal(format!(
                "a {what} does not fit a {}-byte heap ID",
                self.heap_id_length
            ))
        };
        let payload = payload.ok_or_else(too_wide)?.to_le_bytes();
        let (low, high) = payload.split_at(rest.len().min(payload.len()));
        if high.iter().any(|&byte| byte != 0) {
            return Err(too_wide());
        }
        *first = id_type.first_byte();
        for (byte, &value) in rest.iter_mut().zip(low) {
            *byte = value;
        }
        Ok(id)
    }

    /// Returns `true` if the huge heap IDs of the heap hold the address and the length of their
    /// object, and `false` if they hold a key in the huge-object B-tree.
    ///
    /// The file does not store the choice. A reader computes it from the heap ID length and the
    /// widths of an address and a length, as `H5HF__huge_init` in `H5HFhuge.c` (HDF5 2.2.0) sets
    /// `huge_ids_direct`.
    pub fn huge_ids_direct(&self, offset_size: u8, length_size: u8) -> bool {
        let avail = (self.heap_id_length as usize).saturating_sub(1);
        if self.filtering.is_filtered() {
            avail >= offset_size as usize + length_size as usize + 4 + length_size as usize
        } else {
            avail >= offset_size as usize + length_size as usize
        }
    }

    /// Parses the fractal heap header at `address` in `source`.
    ///
    /// Reads the fixed prefix to compute the header length, then reads and parses the complete
    /// header using [`parse_bytes`](Self::parse_bytes). Both reads begin at `address`.
    ///
    /// # Errors
    ///
    /// Returns the errors [`parse`](Self::parse) returns, and the error `source` returns if a read
    /// fails.
    pub fn parse_from_source(
        source: &(impl MetadataSource + ?Sized),
        address: u64,
        offset_size: u8,
        length_size: u8,
    ) -> Result<FractalHeapHeader, FormatError> {
        let offsets = OffsetWidth::try_from(offset_size)?;
        let lengths = LengthWidth::try_from(length_size)?;
        let prefix = source.read_metadata_at(address, FractalHeapHeaderFrame::PREFIX_LEN)?;
        let frame = FractalHeapHeaderFrame::parse(&prefix, offsets, lengths)?;
        let bytes = source.read_metadata_at(address, frame.encoded_len())?;
        Self::parse_bytes(&bytes, offsets, lengths)
    }
}

/// Describes invalid fractal-heap doubling-table geometry or geometry outside representable bounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FractalHeapLayoutError {
    /// The header fields or a requested row describe invalid doubling-table geometry.
    InvalidGeometry,
}

/// Represents validated doubling-table geometry and file-width layout for one fractal heap.
///
/// The layout derives the managed-block geometry from a [`FractalHeapHeader`] once and provides
/// the same checked row, span, and encoded-size calculations to object lookup and storage
/// ownership traversal. Callers perform file I/O and traversal. The doubling table is defined in
/// "Fractal Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FractalHeapLayout {
    offset_width: OffsetWidth,
    length_width: LengthWidth,
    block_offset_size: usize,
    table_width: usize,
    direct_rows: usize,
    max_root_rows: u16,
    first_row_bits: u32,
    filtered: bool,
    row_block_sizes: Vec<u64>,
    row_offsets: Vec<u64>,
    indirect_block_sizes: Vec<u64>,
}

impl FractalHeapLayout {
    /// Derives and validates the doubling-table geometry declared by `header`.
    ///
    /// `offset_width` and `length_width` are the file's address and length field widths.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if the header fields contradict the
    /// fractal-heap doubling-table constraints or any legal row calculation overflows.
    pub fn new(
        header: &FractalHeapHeader,
        offset_width: OffsetWidth,
        length_width: LengthWidth,
    ) -> Result<Self, FractalHeapLayoutError> {
        if header.table_width == 0
            || !header.table_width.is_power_of_two()
            || header.starting_block_size == 0
            || !header.starting_block_size.is_power_of_two()
            || header.max_direct_block_size < header.starting_block_size
            || !header.max_direct_block_size.is_power_of_two()
            || header.max_heap_size == 0
            || header.max_heap_size > 64
        {
            return Err(FractalHeapLayoutError::InvalidGeometry);
        }

        let start_bits = header.starting_block_size.trailing_zeros();
        let max_direct_bits = header.max_direct_block_size.trailing_zeros();
        let first_row_bits = start_bits
            .checked_add(u64::from(header.table_width).trailing_zeros())
            .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
        let heap_bits = u32::from(header.max_heap_size);
        if heap_bits < first_row_bits {
            return Err(FractalHeapLayoutError::InvalidGeometry);
        }
        let max_root_rows = u16::try_from(heap_bits - first_row_bits + 1)
            .map_err(|_conversion_error| FractalHeapLayoutError::InvalidGeometry)?;
        if header.start_root_rows > max_root_rows {
            return Err(FractalHeapLayoutError::InvalidGeometry);
        }
        let direct_rows = usize::try_from(max_direct_bits - start_bits + 2)
            .map_err(|_conversion_error| FractalHeapLayoutError::InvalidGeometry)?;
        let block_offset_size = usize::from(header.max_heap_size).div_ceil(8);
        let table_width = usize::from(header.table_width);
        let table_width_u64 = u64::from(header.table_width);

        let max_root_rows_usize = usize::from(max_root_rows);
        let mut row_block_sizes = Vec::with_capacity(max_root_rows_usize);
        let mut row_offsets = Vec::with_capacity(max_root_rows_usize);
        let mut row_offset = 0u64;
        for row in 0..max_root_rows_usize {
            row_offsets.push(row_offset);
            let block_size = if row <= 1 {
                header.starting_block_size
            } else {
                let shift = u32::try_from(row - 1)
                    .map_err(|_conversion_error| FractalHeapLayoutError::InvalidGeometry)?;
                let multiplier = 1u64
                    .checked_shl(shift)
                    .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
                header
                    .starting_block_size
                    .checked_mul(multiplier)
                    .ok_or(FractalHeapLayoutError::InvalidGeometry)?
            };
            let row_span = block_size
                .checked_mul(table_width_u64)
                .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
            row_block_sizes.push(block_size);
            if row + 1 < max_root_rows_usize {
                row_offset = row_offset
                    .checked_add(row_span)
                    .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
            }
        }

        for (row, &block_size) in row_block_sizes
            .iter()
            .enumerate()
            .take(max_root_rows_usize)
            .skip(direct_rows)
        {
            let size_bits = block_size.trailing_zeros();
            if size_bits < first_row_bits {
                return Err(FractalHeapLayoutError::InvalidGeometry);
            }
            let child_rows = usize::try_from(size_bits - first_row_bits + 1)
                .map_err(|_conversion_error| FractalHeapLayoutError::InvalidGeometry)?;
            if child_rows == 0
                || child_rows > row
                || row_offsets.get(child_rows).copied() != Some(block_size)
            {
                return Err(FractalHeapLayoutError::InvalidGeometry);
            }
        }

        let prefix = 5u64
            .checked_add(u64::from(offset_width.get()))
            .and_then(|n| n.checked_add(u64::try_from(block_offset_size).ok()?))
            .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
        let direct_entry_size = u64::from(offset_width.get())
            .checked_add(if !header.filtering.is_filtered() {
                0
            } else {
                u64::from(length_width.get()) + 4
            })
            .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
        let indirect_entry_size = u64::from(offset_width.get());
        let mut indirect_block_sizes = Vec::with_capacity(usize::from(max_root_rows));
        for nrows in 1..=usize::from(max_root_rows) {
            let direct = nrows.min(direct_rows);
            let indirect = nrows - direct;
            let entry_row_bytes = u64::try_from(direct)
                .ok()
                .and_then(|rows| rows.checked_mul(direct_entry_size))
                .and_then(|bytes| {
                    u64::try_from(indirect)
                        .ok()
                        .and_then(|rows| rows.checked_mul(indirect_entry_size))
                        .and_then(|more| bytes.checked_add(more))
                })
                .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
            let entries = entry_row_bytes
                .checked_mul(table_width_u64)
                .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
            let size = prefix
                .checked_add(entries)
                .and_then(|n| n.checked_add(4))
                .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
            indirect_block_sizes.push(size);
        }

        Ok(Self {
            offset_width,
            length_width,
            block_offset_size,
            table_width,
            direct_rows,
            max_root_rows,
            first_row_bits,
            filtered: header.filtering.is_filtered(),
            row_block_sizes,
            row_offsets,
            indirect_block_sizes,
        })
    }

    /// Returns the file's address field width.
    pub const fn offset_width(&self) -> OffsetWidth {
        self.offset_width
    }

    /// Returns the file's length field width.
    pub const fn length_width(&self) -> LengthWidth {
        self.length_width
    }

    /// Returns the width in bytes of a managed block's "Block Offset" field.
    pub const fn block_offset_size(&self) -> usize {
        self.block_offset_size
    }

    /// Returns the number of child slots in each doubling-table row.
    pub const fn table_width(&self) -> usize {
        self.table_width
    }

    /// Returns the number of doubling-table rows whose entries address direct blocks.
    pub const fn direct_rows(&self) -> usize {
        self.direct_rows
    }

    /// Returns the maximum legal number of rows in the root indirect block.
    pub const fn max_root_rows(&self) -> u16 {
        self.max_root_rows
    }

    /// Returns the exact allocated size of a block in doubling-table `row`.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if `row` is outside the managed heap's
    /// legal doubling-table rows.
    pub fn block_size_for_row(&self, row: usize) -> Result<u64, FractalHeapLayoutError> {
        self.row_block_sizes
            .get(row)
            .copied()
            .ok_or(FractalHeapLayoutError::InvalidGeometry)
    }

    /// Returns the row count of the child indirect blocks addressed by doubling-table `row`.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if `row` is a direct row or lies outside
    /// the managed heap's legal doubling-table rows.
    pub fn child_indirect_rows(&self, row: usize) -> Result<u16, FractalHeapLayoutError> {
        if row < self.direct_rows {
            return Err(FractalHeapLayoutError::InvalidGeometry);
        }
        let size = self.block_size_for_row(row)?;
        let size_bits = size.trailing_zeros();
        if size_bits < self.first_row_bits {
            return Err(FractalHeapLayoutError::InvalidGeometry);
        }
        u16::try_from(size_bits - self.first_row_bits + 1)
            .map_err(|_conversion_error| FractalHeapLayoutError::InvalidGeometry)
    }

    /// Returns the managed heap-address space represented by an indirect block of `nrows` rows.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if `nrows` is zero or exceeds the
    /// maximum root row count.
    pub fn indirect_heap_size(&self, nrows: u16) -> Result<u64, FractalHeapLayoutError> {
        if nrows == 0 || nrows > self.max_root_rows {
            return Err(FractalHeapLayoutError::InvalidGeometry);
        }
        let last_row = usize::from(nrows - 1);
        let start = self.row_offset(last_row)?;
        let span = self
            .block_size_for_row(last_row)?
            .checked_mul(
                u64::try_from(self.table_width)
                    .map_err(|_conversion_error| FractalHeapLayoutError::InvalidGeometry)?,
            )
            .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
        start
            .checked_add(span)
            .ok_or(FractalHeapLayoutError::InvalidGeometry)
    }

    /// Returns the exact encoded allocation size of an indirect block of `nrows` rows.
    ///
    /// The size includes the `FHIB` signature, version, heap-header address, block heap offset,
    /// child entries, and trailing checksum. Filtered heaps use their wider direct-child entries.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if `nrows` is zero or exceeds the
    /// maximum root row count.
    pub fn indirect_block_size(&self, nrows: u16) -> Result<u64, FractalHeapLayoutError> {
        let index = usize::from(nrows)
            .checked_sub(1)
            .ok_or(FractalHeapLayoutError::InvalidGeometry)?;
        self.indirect_block_sizes
            .get(index)
            .copied()
            .ok_or(FractalHeapLayoutError::InvalidGeometry)
    }

    /// Returns the encoded indirect-block length through the end of its child entries.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if `nrows` is zero or exceeds the
    /// maximum root row count.
    pub fn indirect_block_entries_len(&self, nrows: u16) -> Result<u64, FractalHeapLayoutError> {
        self.indirect_block_size(nrows)?
            .checked_sub(4)
            .ok_or(FractalHeapLayoutError::InvalidGeometry)
    }

    /// Returns the heap offset of the first child slot in doubling-table `row`.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if `row` lies outside the managed
    /// heap's legal doubling-table rows.
    pub fn row_offset(&self, row: usize) -> Result<u64, FractalHeapLayoutError> {
        self.row_offsets
            .get(row)
            .copied()
            .ok_or(FractalHeapLayoutError::InvalidGeometry)
    }

    /// Returns the heap offset of child slot `column` in doubling-table `row`.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapLayoutError::InvalidGeometry`] if the row or column lies outside the
    /// doubling table.
    pub fn slot_offset(&self, row: usize, column: usize) -> Result<u64, FractalHeapLayoutError> {
        if column >= self.table_width {
            return Err(FractalHeapLayoutError::InvalidGeometry);
        }
        let row_offset = self.row_offset(row)?;
        let block_size = self.block_size_for_row(row)?;
        let column = u64::try_from(column)
            .map_err(|_conversion_error| FractalHeapLayoutError::InvalidGeometry)?;
        row_offset
            .checked_add(
                column
                    .checked_mul(block_size)
                    .ok_or(FractalHeapLayoutError::InvalidGeometry)?,
            )
            .ok_or(FractalHeapLayoutError::InvalidGeometry)
    }

    /// Returns the allocated child whose managed heap space contains `target_offset`.
    ///
    /// `block` begins at the `FHIB` signature, and `iblock_heap_offset` is the managed heap offset
    /// represented by the indirect block itself.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapSignature`] if `block` lacks the `FHIB` signature,
    /// [`FormatError::UnsupportedFilteredHeapObject`] if the heap filters managed blocks, and
    /// [`FormatError::UnexpectedEof`] if a required child address lies beyond `block`.
    pub fn find_child_for_offset(
        &self,
        block: &[u8],
        nrows: u16,
        iblock_heap_offset: u64,
        target_offset: u64,
    ) -> Result<Option<FractalHeapChild>, FormatError> {
        bytes::ensure_len(block, 0, 4)?;
        if &block[..4] != b"FHIB" {
            return Err(FormatError::InvalidFractalHeapSignature);
        }
        if self.filtered {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }
        if nrows == 0 || nrows > self.max_root_rows {
            return Err(Self::geometry_format_error());
        }

        let entries_at = 5usize
            .checked_add(usize::from(self.offset_width.get()))
            .and_then(|n| n.checked_add(self.block_offset_size))
            .ok_or_else(Self::geometry_format_error)?;
        let direct_rows = usize::from(nrows).min(self.direct_rows);
        let offset_size = usize::from(self.offset_width.get());

        for row in 0..usize::from(nrows) {
            let slot_size = self
                .block_size_for_row(row)
                .map_err(|_error| Self::geometry_format_error())?;
            let child_rows = if row < direct_rows {
                None
            } else {
                Some(
                    self.child_indirect_rows(row)
                        .map_err(|_error| Self::geometry_format_error())?,
                )
            };
            for column in 0..self.table_width {
                let entry = row
                    .checked_mul(self.table_width)
                    .and_then(|n| n.checked_add(column))
                    .ok_or_else(Self::geometry_format_error)?;
                let pos = entries_at
                    .checked_add(
                        entry
                            .checked_mul(offset_size)
                            .ok_or_else(Self::geometry_format_error)?,
                    )
                    .ok_or_else(Self::geometry_format_error)?;
                let child_addr = bytes::read_offset(block, pos, self.offset_width.get())?;
                if convert::is_undefined_addr(child_addr, self.offset_width.get()) {
                    continue;
                }
                let child_heap_offset = iblock_heap_offset
                    .checked_add(
                        self.slot_offset(row, column)
                            .map_err(|_error| Self::geometry_format_error())?,
                    )
                    .ok_or_else(Self::geometry_format_error)?;
                let span = match child_rows {
                    Some(rows) => self
                        .indirect_heap_size(rows)
                        .map_err(|_error| Self::geometry_format_error())?,
                    None => slot_size,
                };
                let child_end = child_heap_offset
                    .checked_add(span)
                    .ok_or_else(Self::geometry_format_error)?;
                if target_offset < child_heap_offset || target_offset >= child_end {
                    continue;
                }
                return Ok(Some(match child_rows {
                    Some(nrows) => FractalHeapChild::Indirect {
                        addr: StoredAddress::new(child_addr),
                        nrows,
                        heap_offset: child_heap_offset,
                    },
                    None => FractalHeapChild::Direct {
                        addr: StoredAddress::new(child_addr),
                        block_size: slot_size,
                        heap_offset: child_heap_offset,
                    },
                }));
            }
        }
        Ok(None)
    }

    fn geometry_format_error() -> FormatError {
        FormatError::ChunkedReadError("fractal heap: invalid doubling-table geometry".into())
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
    /// Parses optional root filter fields and `encoded_len` bytes of filter information at `offset`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if a field or the pipeline is incomplete,
    /// [`FormatError::InvalidFilterPipelineVersion`] if its version is not 1 or 2,
    /// [`FormatError::InvalidFilterPipelineField`] if a pipeline field exceeds its format limit,
    /// and [`FormatError::InvalidFilterName`] if a stored name is malformed.
    fn parse(
        data: &[u8],
        lengths: LengthWidth,
        offset: usize,
        encoded_len: u16,
    ) -> Result<Self, FormatError> {
        if encoded_len == 0 {
            return Ok(Self::Unfiltered);
        }
        let mut fields = bytes::Fields::new(data, offset);
        let root_direct_block_size = fields.length(lengths)?;
        let root_filter_mask = fields.u32()?;
        let end = fields.pos().checked_add(usize::from(encoded_len)).ok_or(
            FormatError::UnexpectedEof {
                expected: usize::MAX,
                available: data.len(),
            },
        )?;
        let pipeline_bytes = data
            .get(fields.pos()..end)
            .ok_or(FormatError::UnexpectedEof {
                expected: end,
                available: data.len(),
            })?;
        Ok(Self::Filtered {
            root_direct_block_size,
            root_filter_mask,
            pipeline: FilterPipeline::parse(pipeline_bytes).map_err(|error| match error {
                FilterPipelineError::Format(error) => error,
                FilterPipelineError::FieldTooLarge {
                    field,
                    value,
                    maximum,
                } => FormatError::InvalidFilterPipelineField {
                    field,
                    value,
                    maximum,
                },
                FilterPipelineError::InvalidName { filter_id, reason } => {
                    FormatError::InvalidFilterName { filter_id, reason }
                }
            })?,
        })
    }

    /// Returns `true` if the header contains encoded filter information.
    pub const fn is_filtered(&self) -> bool {
        matches!(self, Self::Filtered { .. })
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
    /// Parses the fixed prefix at the start of `bytes` to compute the complete header length.
    ///
    /// `offsets` and `lengths` are the superblock's "Size of Offsets" and "Size of Lengths"
    /// fields. Bytes after [`PREFIX_LEN`](Self::PREFIX_LEN) are ignored.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapSignature`] if the prefix does not begin with
    /// `FRHP`, [`FormatError::InvalidFractalHeapVersion`] if its version is not 0, and
    /// [`FormatError::UnexpectedEof`] if a prefix field is incomplete.
    ///
    /// Returns [`FormatError::Internal`] if the length arithmetic overflows, and
    /// [`FormatError::ValueTooLargeForPlatform`] if the length exceeds [`usize::MAX`].
    pub fn parse(
        bytes: &[u8],
        offsets: OffsetWidth,
        lengths: LengthWidth,
    ) -> Result<Self, FormatError> {
        let mut fields = bytes::Fields::new(bytes, 0);
        if fields.array::<{ FRACTAL_HEAP_SIGNATURE.len() }>()? != FRACTAL_HEAP_SIGNATURE {
            return Err(FormatError::InvalidFractalHeapSignature);
        }
        let version = fields.u8()?;
        if version != FRACTAL_HEAP_VERSION {
            return Err(FormatError::InvalidFractalHeapVersion(version));
        }
        fields.u16()?;
        let filter_encoded_len = fields.u16()?;
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

    /// Returns the header length in bytes, including the checksum and any filter fields.
    pub const fn encoded_len(&self) -> usize {
        self.encoded_len
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

/// The immutable decoding parameters for the IDs of a fractal heap.
///
/// The parameters are derived from a [`FractalHeapHeader`] and the file's address and length
/// widths. [`parse`](Self::parse) decodes supplied bytes without reading heap blocks or the
/// huge-object B-tree. The heap IDs are defined in "Fractal Heap" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
#[derive(Clone, Debug)]
pub struct FractalHeapIdLayout {
    encoded_len: usize,
    offset_bits: u32,
    offset_mask: u64,
    length_mask: u64,
    tiny_extended: bool,
    huge: HugeIdLayout,
}

impl FractalHeapIdLayout {
    /// Derives the heap ID decoding parameters from `header` and the file's field widths.
    pub fn new(header: &FractalHeapHeader, offsets: OffsetWidth, lengths: LengthWidth) -> Self {
        let encoded_len = usize::from(header.heap_id_length);
        let offset_bits = u32::from(header.max_heap_size);
        let payload_bits = u32::from(header.heap_id_length.saturating_sub(1)) * 8;
        let huge = if header.filtering.is_filtered() {
            HugeIdLayout::Filtered
        } else if header.huge_ids_direct(offsets.get(), lengths.get()) {
            HugeIdLayout::Inline { offsets, lengths }
        } else {
            HugeIdLayout::Indexed {
                has_btree: !convert::is_undefined_addr(
                    header.btree_huge_objects_address.get(),
                    offsets.get(),
                ),
            }
        };
        Self {
            encoded_len,
            offset_bits,
            offset_mask: Self::mask(offset_bits),
            length_mask: Self::mask(payload_bits.saturating_sub(offset_bits)),
            tiny_extended: header.heap_id_length.saturating_sub(1) > TINY_LEN_SHORT,
            huge,
        }
    }

    /// Parses a complete heap ID using this heap's decoding parameters.
    ///
    /// Reads the managed object's heap offset and length, the huge object's location, or the
    /// tiny object's bytes. For a tiny object, the caller receives a slice borrowed from `bytes`.
    /// The parser reads the type independently of the version bits.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapIdLength`] if `bytes` has a different length
    /// from [`encoded_len`](Self::encoded_len), [`FormatError::InvalidHeapIdType`] if its type
    /// bits hold 3, and [`FormatError::UnexpectedEof`] if the declared ID length is zero or a tiny
    /// object's declared length exceeds the available bytes.
    ///
    /// Returns [`FormatError::UnsupportedFilteredHeapObject`] for a huge ID in a filtered
    /// heap, and [`FormatError::HugeObjectNotFound`] for an indexed huge ID in a heap without
    /// a huge-object B-tree.
    pub fn parse<'a>(&self, bytes: &'a [u8]) -> Result<FractalHeapIdView<'a>, FormatError> {
        if bytes.len() != self.encoded_len {
            return Err(FormatError::InvalidFractalHeapIdLength {
                expected: self.encoded_len,
                actual: bytes.len(),
            });
        }
        let kind = FractalHeapIdType::from_heap_id(bytes)?;
        let (&first, payload) = bytes.split_first().ok_or(FormatError::UnexpectedEof {
            expected: 1,
            available: bytes.len(),
        })?;
        Ok(FractalHeapIdView {
            kind: match kind {
                FractalHeapIdType::Managed => self.parse_managed(payload),
                FractalHeapIdType::Huge => FractalHeapIdKind::Huge(self.parse_huge(payload)?),
                FractalHeapIdType::Tiny => FractalHeapIdKind::Tiny {
                    bytes: self.parse_tiny(first, bytes)?,
                },
            },
        })
    }

    /// Returns the exact ID length in bytes declared by the heap header.
    pub const fn encoded_len(&self) -> usize {
        self.encoded_len
    }

    fn parse_managed(&self, payload: &[u8]) -> FractalHeapIdKind<'static> {
        let combined = read_var_le(payload);
        FractalHeapIdKind::Managed {
            heap_offset: combined & self.offset_mask,
            object_length: combined.checked_shr(self.offset_bits).unwrap_or(0) & self.length_mask,
        }
    }

    fn parse_huge(&self, payload: &[u8]) -> Result<HugeObjectReference, FormatError> {
        Ok(match self.huge {
            HugeIdLayout::Filtered => return Err(FormatError::UnsupportedFilteredHeapObject),
            HugeIdLayout::Inline { offsets, lengths } => HugeObjectReference::Inline {
                addr: StoredAddress::new(bytes::read_offset(payload, 0, offsets.get())?),
                len: bytes::read_length(payload, usize::from(offsets.get()), lengths.get())?,
            },
            HugeIdLayout::Indexed { has_btree } => {
                let key = read_var_le(payload);
                if !has_btree {
                    return Err(FormatError::HugeObjectNotFound(key));
                }
                HugeObjectReference::Indexed(key)
            }
        })
    }

    fn parse_tiny<'a>(&self, first: u8, bytes: &'a [u8]) -> Result<&'a [u8], FormatError> {
        let (len, start) = if self.tiny_extended {
            let second = bytes.get(1).ok_or(FormatError::UnexpectedEof {
                expected: 2,
                available: bytes.len(),
            })?;
            (
                (((usize::from(first & TINY_LENGTH_MASK)) << 8) | usize::from(*second)) + 1,
                2,
            )
        } else {
            (usize::from(first & TINY_LENGTH_MASK) + 1, 1)
        };
        bytes
            .get(start..start + len)
            .ok_or(FormatError::UnexpectedEof {
                expected: start + len,
                available: bytes.len(),
            })
    }

    fn mask(bits: u32) -> u64 {
        1u64.checked_shl(bits).map_or(u64::MAX, |limit| limit - 1)
    }
}

/// A heap ID parsed against the decoding parameters of a fractal heap.
///
/// Obtained through [`FractalHeapIdLayout::parse`], which checks the encoded length and decodes
/// the type-specific fields. Only a tiny object's bytes are borrowed from the encoded ID.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FractalHeapIdView<'a> {
    kind: FractalHeapIdKind<'a>,
}

impl<'a> FractalHeapIdView<'a> {
    /// Returns the decoded object location or the borrowed bytes of a tiny object.
    pub const fn kind(self) -> FractalHeapIdKind<'a> {
        self.kind
    }
}

/// The decoded fields of a heap ID, returned by [`FractalHeapIdView::kind`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FractalHeapIdKind<'a> {
    /// The object is stored in the heap's managed blocks.
    Managed {
        /// The object's offset in the heap's address space.
        heap_offset: u64,
        /// The object's length in bytes.
        object_length: u64,
    },
    /// The object is stored outside the heap's managed blocks.
    Huge(HugeObjectReference),
    /// The encoded ID holds the object itself.
    Tiny {
        /// The object's bytes, borrowed from the encoded ID.
        bytes: &'a [u8],
    },
}

#[derive(Clone, Copy, Debug)]
enum HugeIdLayout {
    Filtered,
    Indexed {
        has_btree: bool,
    },
    Inline {
        offsets: OffsetWidth,
        lengths: LengthWidth,
    },
}

fn fits_bits(value: u64, bits: u32) -> bool {
    value.checked_shr(bits).is_none_or(|high| high == 0)
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

/// The position of the type bits, bits 4 and 5, in the first byte of a heap ID, from the same
/// section as [`FRACTAL_HEAP_SIGNATURE`].
const HEAP_ID_TYPE_SHIFT: u8 = 4;

/// The mask of the two type bits once shifted down by [`HEAP_ID_TYPE_SHIFT`].
const HEAP_ID_TYPE_MASK: u8 = 0x03;

/// The type of the heap ID of a managed object, from the same section as
/// [`FRACTAL_HEAP_SIGNATURE`].
const HEAP_ID_TYPE_MANAGED: u8 = 0;

/// The type of the heap ID of a huge object, from the same section as [`FRACTAL_HEAP_SIGNATURE`].
const HEAP_ID_TYPE_HUGE: u8 = 1;

/// The type of the heap ID of a tiny object, from the same section as [`FRACTAL_HEAP_SIGNATURE`].
const HEAP_ID_TYPE_TINY: u8 = 2;

// `H5HF__tiny_init` in `H5HFtiny.c`, HDF5 2.2.0, uses the short form for up to 16 payload bytes.
const TINY_LEN_SHORT: u16 = 16;

// The low four bits hold the length less one in the normal form and its high four bits in the
// extended form ("Fractal Heap", specification 4.0).
const TINY_LENGTH_MASK: u8 = 0x0F;

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
mod tests {
    use core::cell::RefCell;
    use core::mem;

    use rstest::rstest;
    use test_util::fractal_heap;
    use test_util::widths::Widths;

    use crate::filter_pipeline::FilterDescription;

    use super::*;

    #[rstest]
    #[case::two_two(OffsetWidth::Two, LengthWidth::Two, 56, 62)]
    #[case::two_four(OffsetWidth::Two, LengthWidth::Four, 80, 88)]
    #[case::two_eight(OffsetWidth::Two, LengthWidth::Eight, 128, 140)]
    #[case::four_two(OffsetWidth::Four, LengthWidth::Two, 62, 68)]
    #[case::four_four(OffsetWidth::Four, LengthWidth::Four, 86, 94)]
    #[case::four_eight(OffsetWidth::Four, LengthWidth::Eight, 134, 146)]
    #[case::eight_two(OffsetWidth::Eight, LengthWidth::Two, 74, 80)]
    #[case::eight_four(OffsetWidth::Eight, LengthWidth::Four, 98, 106)]
    #[case::eight_eight(OffsetWidth::Eight, LengthWidth::Eight, 146, 158)]
    fn a_header_frame_sizes_all_fields_at_their_declared_widths(
        #[case] offsets: OffsetWidth,
        #[case] lengths: LengthWidth,
        #[case] plain_len: usize,
        #[case] filtered_len: usize,
        #[values(0, 1, 512, u16::MAX)] filter_encoded_len: u16,
    ) {
        let prefix = header_prefix(filter_encoded_len);
        assert_eq!(prefix.len(), FractalHeapHeaderFrame::PREFIX_LEN);
        let frame = FractalHeapHeaderFrame::parse(&prefix, offsets, lengths).unwrap();
        assert_eq!(
            frame.encoded_len(),
            if filter_encoded_len == 0 {
                plain_len
            } else {
                filtered_len + usize::from(filter_encoded_len)
            }
        );
    }

    #[rstest]
    #[case::empty(0, FRACTAL_HEAP_SIGNATURE.len())]
    #[case::partial_signature(FRACTAL_HEAP_SIGNATURE.len() - 1, FRACTAL_HEAP_SIGNATURE.len())]
    #[case::signature(FRACTAL_HEAP_SIGNATURE.len(), HEADER_VERSION_END)]
    #[case::version(HEADER_VERSION_END, HEADER_ID_LENGTH_END)]
    #[case::partial_id_length(HEADER_VERSION_END + 1, HEADER_ID_LENGTH_END)]
    #[case::id_length(HEADER_ID_LENGTH_END, FractalHeapHeaderFrame::PREFIX_LEN)]
    #[case::partial_filter_length(FractalHeapHeaderFrame::PREFIX_LEN - 1, FractalHeapHeaderFrame::PREFIX_LEN)]
    fn an_incomplete_header_prefix_reports_the_missing_field(
        #[case] available: usize,
        #[case] expected: usize,
    ) {
        let mut prefix = header_prefix(0);
        prefix.truncate(available);
        assert_eq!(
            FractalHeapHeaderFrame::parse(&prefix, OffsetWidth::Eight, LengthWidth::Eight),
            Err(FormatError::UnexpectedEof {
                expected,
                available,
            })
        );
    }

    #[rstest]
    #[case::signature(true, FRACTAL_HEAP_VERSION, FormatError::InvalidFractalHeapSignature)]
    #[case::version(false, FRACTAL_HEAP_VERSION + 1, FormatError::InvalidFractalHeapVersion(FRACTAL_HEAP_VERSION + 1))]
    fn a_header_frame_rejects_an_invalid_signature_or_version(
        #[case] corrupt_signature: bool,
        #[case] version: u8,
        #[case] error: FormatError,
    ) {
        let mut prefix = header_prefix(0);
        if corrupt_signature {
            prefix[..FRACTAL_HEAP_SIGNATURE.len()].fill(0);
        }
        prefix[FRACTAL_HEAP_SIGNATURE.len()] = version;
        assert_eq!(
            FractalHeapHeaderFrame::parse(&prefix, OffsetWidth::Eight, LengthWidth::Eight),
            Err(error)
        );
    }

    #[rstest]
    #[case::prefix(0)]
    #[case::trailing_bytes(1024)]
    fn a_header_frame_only_interprets_the_prefix(#[case] trailing_len: usize) {
        let mut bytes = header_prefix(512);
        bytes.resize(FractalHeapHeaderFrame::PREFIX_LEN + trailing_len, 0xFF);
        assert_eq!(
            FractalHeapHeaderFrame::parse(&bytes, OffsetWidth::Eight, LengthWidth::Eight)
                .unwrap()
                .encoded_len(),
            670
        );
    }

    #[rstest]
    #[case::two_two(Widths { offset: 2, length: 2 }, OffsetWidth::Two, LengthWidth::Two)]
    #[case::two_eight(Widths { offset: 2, length: 8 }, OffsetWidth::Two, LengthWidth::Eight)]
    #[case::eight_two(Widths { offset: 8, length: 2 }, OffsetWidth::Eight, LengthWidth::Two)]
    #[case::eight_eight(Widths::EIGHT, OffsetWidth::Eight, LengthWidth::Eight)]
    fn complete_headers_retain_filter_metadata_with_two_and_eight_byte_fields(
        #[case] widths: Widths,
        #[case] offsets: OffsetWidth,
        #[case] lengths: LengthWidth,
        #[values(0, 1, 80)] client_count: usize,
        #[values(PIPELINE_VERSION_ONE, PIPELINE_VERSION_TWO)] pipeline_version: u8,
    ) {
        let pipeline = filter_pipeline(pipeline_version, client_count);
        let filtering = if client_count == 0 {
            FractalHeapFiltering::Unfiltered
        } else {
            FractalHeapFiltering::Filtered {
                root_direct_block_size: 91,
                root_filter_mask: 1,
                pipeline: pipeline.clone(),
            }
        };
        let mut builder = fractal_heap::Header::new(0x100).managed_object_count(3);
        if filtering.is_filtered() {
            builder = builder.filtering(91, 1, pipeline.serialize().unwrap());
        }
        let bytes = builder.build(widths);
        let frame = FractalHeapHeaderFrame::parse(&bytes, offsets, lengths).unwrap();
        let extension_len = if filtering.is_filtered() {
            usize::from(lengths.get())
                + FRACTAL_HEAP_FILTER_MASK_LEN.to_usize().unwrap()
                + pipeline.serialize().unwrap().len()
        } else {
            0
        };
        assert_eq!(
            frame.encoded_len(),
            FRACTAL_HEAP_HEADER_FIXED_LEN.to_usize().unwrap()
                + FRACTAL_HEAP_HEADER_LENGTH_FIELDS.to_usize().unwrap() * widths.length
                + FRACTAL_HEAP_HEADER_ADDRESS_FIELDS.to_usize().unwrap() * widths.offset
                + extension_len
        );
        assert_eq!(bytes.len(), frame.encoded_len());
        if client_count == 80 {
            assert!(bytes.len() > 256);
        }
        let parsed = FractalHeapHeader::parse_bytes(&bytes, offsets, lengths).unwrap();
        assert_eq!(parsed.filtering, filtering);
        assert_eq!(parsed.managed_objects_count, 3);
        let source = HeaderSource {
            bytes: &bytes,
            prefix: None,
            reads: RefCell::new(Vec::new()),
        };
        assert_eq!(
            FractalHeapHeader::parse_from_source(&source, 0, offsets.get(), lengths.get()),
            Ok(parsed.clone())
        );
        assert_eq!(
            *source.reads.borrow(),
            vec![(0, FractalHeapHeaderFrame::PREFIX_LEN), (0, bytes.len())]
        );
        let mut file = vec![0xFF; 13];
        file.extend_from_slice(&bytes);
        file.extend_from_slice(&[0xFF; 7]);
        assert_eq!(
            FractalHeapHeader::parse(&file, 13, offsets.get(), lengths.get()),
            Ok(parsed.clone())
        );
        assert_eq!(
            FractalHeapHeader::parse_from_source(file.as_slice(), 13, offsets.get(), lengths.get()),
            Ok(parsed)
        );
    }

    #[rstest]
    #[case::unfiltered(false, false)]
    #[case::unfiltered_after_filtered_prefix(false, true)]
    #[case::filtered_after_unfiltered_prefix(true, false)]
    #[case::filtered(true, true)]
    fn source_parsing_reads_the_prefix_fields_from_the_complete_header(
        #[case] filtered_header: bool,
        #[case] filtered_prefix: bool,
    ) {
        let pipeline = filter_pipeline(PIPELINE_VERSION_TWO, 1)
            .serialize()
            .unwrap();
        let mut builder = fractal_heap::Header::new(0x100);
        if filtered_header {
            builder = builder.filtering(91, 1, pipeline.clone());
        }
        let mut bytes = builder.build(Widths::EIGHT);
        let parsed = FractalHeapHeader::parse(&bytes, 0, 8, 8).unwrap();
        let mut prefix = header_prefix(if filtered_prefix {
            u16::try_from(pipeline.len()).unwrap()
        } else {
            0
        });
        prefix[HEADER_VERSION_END..HEADER_ID_LENGTH_END].copy_from_slice(&u16::MAX.to_le_bytes());
        let framed_len =
            FractalHeapHeaderFrame::parse(&prefix, OffsetWidth::Eight, LengthWidth::Eight)
                .unwrap()
                .encoded_len();
        bytes.resize(bytes.len().max(framed_len), 0xFF);
        let source = HeaderSource {
            bytes: &bytes,
            prefix: Some(prefix.try_into().unwrap()),
            reads: RefCell::new(Vec::new()),
        };
        let result = FractalHeapHeader::parse_from_source(&source, 0, 8, 8);
        if filtered_header && !filtered_prefix {
            assert_eq!(
                result,
                Err(FormatError::UnexpectedEof {
                    expected: bytes.len(),
                    available: framed_len,
                })
            );
        } else {
            assert_eq!(result, Ok(parsed));
        }
        assert_eq!(
            *source.reads.borrow(),
            vec![(0, FractalHeapHeaderFrame::PREFIX_LEN), (0, framed_len)]
        );
    }

    #[rstest]
    #[case::unfiltered_checksum(0, 145, 146)]
    #[case::root_size_start(1, 142, 170)]
    #[case::partial_root_size(1, 149, 170)]
    #[case::mask_start(1, 150, 170)]
    #[case::partial_mask(1, 153, 170)]
    #[case::pipeline_start(1, 154, 170)]
    #[case::partial_pipeline(1, 165, 170)]
    #[case::checksum_start(1, 166, 170)]
    #[case::partial_checksum(1, 169, 170)]
    fn an_incomplete_header_requires_its_complete_framed_extent(
        #[case] client_count: usize,
        #[case] available: usize,
        #[case] expected: usize,
    ) {
        let mut builder = fractal_heap::Header::new(0x100);
        if client_count > 0 {
            builder = builder.filtering(
                91,
                1,
                filter_pipeline(PIPELINE_VERSION_TWO, client_count)
                    .serialize()
                    .unwrap(),
            );
        }
        let mut bytes = builder.build(Widths::EIGHT);
        assert_eq!(bytes.len(), expected);
        bytes.truncate(available);
        let error = FormatError::UnexpectedEof {
            expected,
            available,
        };
        assert_eq!(
            FractalHeapHeader::parse(&bytes, 0, 8, 8),
            Err(error.clone())
        );
        assert_eq!(
            FractalHeapHeader::parse_from_source(bytes.as_slice(), 0, 8, 8),
            Err(error)
        );
    }

    #[rstest]
    #[case::truncated_client_data(vec![PIPELINE_VERSION_TWO, 1, FILTER_DEFLATE.to_le_bytes()[0], FILTER_DEFLATE.to_le_bytes()[1], 0, 0, 1, 0], FormatError::UnexpectedEof { expected: 12, available: 8 })]
    #[case::invalid_version(vec![0xFF, 0], FormatError::InvalidFilterPipelineVersion(0xFF))]
    #[case::too_many_filters(vec![PIPELINE_VERSION_TWO, FILTER_COUNT_MAXIMUM + 1], FormatError::InvalidFilterPipelineField { field: "filter count", value: usize::from(FILTER_COUNT_MAXIMUM) + 1, maximum: usize::from(FILTER_COUNT_MAXIMUM) })]
    #[case::invalid_name(vec![PIPELINE_VERSION_TWO, 1, 0x2C, 1, 1, 0, 0, 0, 0, 0, b'a'], FormatError::InvalidFilterName { filter_id: 300, reason: "missing null terminator" })]
    fn malformed_filter_information_returns_its_pipeline_error(
        #[case] pipeline: Vec<u8>,
        #[case] error: FormatError,
    ) {
        let bytes = fractal_heap::Header::new(0x100)
            .filtering(91, 1, pipeline)
            .build(Widths::EIGHT);
        assert_eq!(
            FractalHeapHeader::parse(&bytes, 0, 8, 8),
            Err(error.clone())
        );
        assert_eq!(
            FractalHeapHeader::parse_from_source(bytes.as_slice(), 0, 8, 8),
            Err(error)
        );
    }

    #[cfg(feature = "checksum")]
    #[rstest]
    #[case::unfiltered(false)]
    #[case::filtered(true)]
    fn a_header_checksum_error_reports_the_final_checksum(#[case] filtered: bool) {
        let mut builder = fractal_heap::Header::new(0x100);
        if filtered {
            builder = builder.filtering(
                91,
                1,
                filter_pipeline(PIPELINE_VERSION_TWO, 1)
                    .serialize()
                    .unwrap(),
            );
        }
        let mut bytes = builder.build(Widths::EIGHT);
        let checksum_pos = bytes.len() - FRACTAL_HEAP_CHECKSUM_LEN;
        let computed = crate::checksum::jenkins_lookup3(&bytes[..checksum_pos]);
        let expected = computed ^ 1;
        bytes[checksum_pos..].copy_from_slice(&expected.to_le_bytes());
        let error = FormatError::ChecksumMismatch { expected, computed };
        assert_eq!(
            FractalHeapHeader::parse(&bytes, 0, 8, 8),
            Err(error.clone())
        );
        assert_eq!(
            FractalHeapHeader::parse_from_source(bytes.as_slice(), 0, 8, 8),
            Err(error)
        );
    }

    #[test]
    fn a_header_parses_to_its_fields() {
        let file_data = fractal_heap::heap_with_one_object(b"Hello, World!", Widths::EIGHT);
        let expected = FractalHeapHeader {
            heap_id_length: 7,
            filtering: FractalHeapFiltering::Unfiltered,
            flags: 0,
            max_managed_object_size: 64,
            next_huge_object_id: 0,
            btree_huge_objects_address: StoredAddress::new(u64::MAX),
            free_space_in_managed_blocks: 0,
            managed_block_free_space_manager_address: StoredAddress::new(u64::MAX),
            managed_space: 0,
            allocated_managed_space: 0,
            direct_block_allocation_iterator_offset: 0,
            managed_objects_count: 1,
            huge_objects_size: 0,
            huge_objects_count: 0,
            tiny_objects_size: 0,
            tiny_objects_count: 0,
            table_width: 4,
            starting_block_size: 128,
            max_direct_block_size: 1024,
            max_heap_size: 16,
            start_root_rows: 2,
            root_block_address: StoredAddress::new(256),
            current_rows_in_root_indirect_block: 0,
        };

        assert_eq!(
            FractalHeapHeader::parse(&file_data, 0, 8, 8),
            Ok(expected.clone())
        );
        assert_eq!(
            FractalHeapHeader::parse_from_source(file_data.as_slice(), 0, 8, 8),
            Ok(expected)
        );
    }

    #[test]
    fn a_managed_id_decodes_to_its_offset_and_length() {
        let file_data = fractal_heap::heap_with_one_object(b"Hello, World!", Widths::EIGHT);
        let hdr = FractalHeapHeader::parse(&file_data, 0, 8, 8).unwrap();

        // Byte 0 holds version 0 in bits 6-7 and type 0 in bits 4-5. Bytes 1-6 hold the offset in
        // `max_heap_size` (16) bits, then the length: offset 0 and length 13 pack to 0x000D_0000.
        let offset: u64 = 0;
        let length: u64 = 13;
        let payload = offset | (length << hdr.max_heap_size);
        let mut id = vec![0u8; 7];
        id[0] = 0x00; // type=0
        for i in 0..6 {
            id[1 + i] = ((payload >> (i * 8)) & 0xFF) as u8;
        }

        assert_eq!(
            FractalHeapIdLayout::new(&hdr, OffsetWidth::Eight, LengthWidth::Eight)
                .parse(&id)
                .unwrap()
                .kind(),
            FractalHeapIdKind::Managed {
                heap_offset: 0,
                object_length: 13
            },
        );
    }

    #[rstest]
    #[case::too_short(9, vec![0; 8])]
    #[case::too_long(7, vec![0; 8])]
    #[case::empty(8, vec![])]
    fn an_id_must_have_the_declared_length(#[case] declared: u16, #[case] bytes: Vec<u8>) {
        let header = FractalHeapHeader {
            heap_id_length: declared,
            ..attribute_heap_header()
        };
        let layout = FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight);
        assert_eq!(layout.encoded_len(), usize::from(declared));
        assert_eq!(
            layout.parse(&bytes),
            Err(FormatError::InvalidFractalHeapIdLength {
                expected: usize::from(declared),
                actual: bytes.len(),
            })
        );
    }

    #[rstest]
    #[case::zero_offset_bits(0, 0, 0x1234)]
    #[case::whole_integer_offset(64, 0x1234, 0)]
    #[case::offset_above_integer(65, 0x1234, 0)]
    fn managed_fields_decode_at_integer_width_boundaries(
        #[case] bits: u16,
        #[case] offset: u64,
        #[case] length: u64,
    ) {
        let header = FractalHeapHeader {
            heap_id_length: 9,
            max_heap_size: bits,
            ..attribute_heap_header()
        };
        let bytes = [0, 0x34, 0x12, 0, 0, 0, 0, 0, 0];
        assert_eq!(
            FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
                .parse(&bytes)
                .unwrap()
                .kind(),
            FractalHeapIdKind::Managed {
                heap_offset: offset,
                object_length: length
            }
        );
    }

    #[rstest]
    #[case::reserved_type(vec![0x30, 0, 0, 0, 0, 0, 0, 0], FormatError::InvalidHeapIdType(3))]
    #[case::tiny_past_id(vec![0x2F, 0, 0, 0, 0, 0, 0, 0], FormatError::UnexpectedEof { expected: 17, available: 8 })]
    fn a_complete_id_with_invalid_fields_is_an_error(
        #[case] bytes: Vec<u8>,
        #[case] error: FormatError,
    ) {
        assert_eq!(
            FractalHeapIdLayout::new(
                &attribute_heap_header(),
                OffsetWidth::Eight,
                LengthWidth::Eight
            )
            .parse(&bytes),
            Err(error)
        );
    }

    #[rstest]
    #[case::filtered(true, FormatError::UnsupportedFilteredHeapObject)]
    #[case::missing_tree(false, FormatError::HugeObjectNotFound(5))]
    fn filtered_huge_ids_and_missing_huge_object_trees_are_errors(
        #[case] filtered: bool,
        #[case] error: FormatError,
    ) {
        let header = FractalHeapHeader {
            filtering: if filtered {
                filtered_heap()
            } else {
                FractalHeapFiltering::Unfiltered
            },
            btree_huge_objects_address: StoredAddress::new(u64::MAX),
            ..attribute_heap_header()
        };
        assert_eq!(
            FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
                .parse(&[0x10, 5, 0, 0, 0, 0, 0, 0]),
            Err(error)
        );
    }

    #[test]
    fn a_tiny_id_borrows_its_payload() {
        let bytes = [0x22, b'a', b'b', b'c', 0, 0, 0, 0];
        let FractalHeapIdKind::Tiny { bytes: object } = FractalHeapIdLayout::new(
            &attribute_heap_header(),
            OffsetWidth::Eight,
            LengthWidth::Eight,
        )
        .parse(&bytes)
        .unwrap()
        .kind() else {
            panic!("expected tiny ID")
        };
        assert_eq!(object, b"abc");
        assert_eq!(object.as_ptr(), bytes[1..].as_ptr());
    }

    #[test]
    fn a_header_without_frhp_is_an_invalid_signature() {
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(b"XXXX");
        let err = FractalHeapHeader::parse(&data, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidFractalHeapSignature);
    }

    #[test]
    fn a_header_of_version_1_is_an_invalid_version() {
        let mut data = vec![0u8; 128];
        data[0..4].copy_from_slice(b"FRHP");
        data[4] = 1;
        let err = FractalHeapHeader::parse(&data, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidFractalHeapVersion(1));
    }

    #[rstest]
    #[case::managed(0x00, FractalHeapIdKind::Managed { heap_offset: 0, object_length: 0 })]
    #[case::huge(0x10, FractalHeapIdKind::Huge(HugeObjectReference::Indexed(0)))]
    #[case::tiny(0x20, FractalHeapIdKind::Tiny { bytes: &[0] })]
    #[case::version_bits(0xD0, FractalHeapIdKind::Huge(HugeObjectReference::Indexed(0)))]
    fn the_parser_reads_type_bits_independently_of_version_bits(
        #[case] first: u8,
        #[case] expected: FractalHeapIdKind<'_>,
    ) {
        let mut bytes = [0; 8];
        bytes[0] = first;
        assert_eq!(
            FractalHeapIdLayout::new(
                &attribute_heap_header(),
                OffsetWidth::Eight,
                LengthWidth::Eight
            )
            .parse(&bytes)
            .unwrap()
            .kind(),
            expected
        );
    }

    #[test]
    fn huge_ids_direct_matches_hdf5_rule() {
        // Unfiltered: direct when `(id_len - 1) >= offset_size + length_size`.
        let mut h = dtable_header(512, 65536, 4);
        h.heap_id_length = 7; // 6 payload bytes < 8 + 8 -> indirect (B-tree)
        assert!(!h.huge_ids_direct(8, 8));
        h.heap_id_length = 17; // 16 payload bytes == 8 + 8 -> direct
        assert!(h.huge_ids_direct(8, 8));
        // A filtered heap needs room for the address, the length, the filter mask (4), and the
        // filtered length, so 17 bytes are too few.
        h.filtering = filtered_heap();
        assert!(!h.huge_ids_direct(8, 8));
    }

    #[test]
    fn the_indirect_block_walk_of_a_filtered_heap_is_unsupported() {
        let mut header = dtable_header(512, 65536, 4);
        header.filtering = filtered_heap();
        let layout =
            FractalHeapLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight).unwrap();
        let mut block = b"FHIB".to_vec();
        block.resize(256, 0);
        assert_eq!(
            layout.find_child_for_offset(&block, 2, 0, 0),
            Err(FormatError::UnsupportedFilteredHeapObject)
        );

        // In an unfiltered heap, the function returns `None` for an offset past the block's space.
        header.filtering = FractalHeapFiltering::Unfiltered;
        let layout =
            FractalHeapLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight).unwrap();
        assert_eq!(
            layout.find_child_for_offset(&block, 2, 0, u64::MAX),
            Ok(None)
        );
    }

    #[rstest]
    #[case::short(7, vec![0x23], b"abcd".as_slice())]
    #[case::extended(20, vec![0x20, 0x04], b"hello".as_slice())]
    #[case::longest_short(17, vec![0x24], b"world".as_slice())]
    fn a_tiny_id_parses_its_short_or_extended_length(
        #[case] declared: u16,
        #[case] mut bytes: Vec<u8>,
        #[case] object: &[u8],
    ) {
        let header = FractalHeapHeader {
            heap_id_length: declared,
            ..attribute_heap_header()
        };
        bytes.extend_from_slice(object);
        bytes.resize(usize::from(declared), 0);
        assert_eq!(
            FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
                .parse(&bytes)
                .unwrap()
                .kind(),
            FractalHeapIdKind::Tiny { bytes: object }
        );
    }

    #[test]
    fn a_huge_id_decodes_to_an_index_key_or_an_inline_location() {
        let indexed = FractalHeapHeader {
            btree_huge_objects_address: StoredAddress::new(0x800),
            ..dtable_header(512, 65536, 4)
        };
        let wide = FractalHeapHeader {
            heap_id_length: 17,
            ..dtable_header(512, 65536, 4)
        };
        let mut inline_huge = vec![0x10];
        inline_huge.extend_from_slice(&0x900u64.to_le_bytes());
        inline_huge.extend_from_slice(&40u64.to_le_bytes());

        assert_eq!(
            FractalHeapIdLayout::new(&indexed, OffsetWidth::Eight, LengthWidth::Eight)
                .parse(&[0x10, 5, 0, 0, 0, 0, 0])
                .map(FractalHeapIdView::kind),
            Ok(FractalHeapIdKind::Huge(HugeObjectReference::Indexed(5)))
        );
        assert_eq!(
            FractalHeapIdLayout::new(&wide, OffsetWidth::Eight, LengthWidth::Eight)
                .parse(&inline_huge)
                .map(FractalHeapIdView::kind),
            Ok(FractalHeapIdKind::Huge(HugeObjectReference::Inline {
                addr: StoredAddress::new(0x900),
                len: 40,
            }))
        );
    }

    /// Returns a header with the supplied doubling-table parameters.
    fn dtable_header(
        start_block_size: u64,
        max_direct_block_size: u64,
        table_width: u16,
    ) -> FractalHeapHeader {
        FractalHeapHeader {
            heap_id_length: 7,
            filtering: FractalHeapFiltering::Unfiltered,
            flags: 0,
            max_managed_object_size: 0,
            next_huge_object_id: 0,
            btree_huge_objects_address: StoredAddress::new(u64::MAX),
            free_space_in_managed_blocks: 0,
            managed_block_free_space_manager_address: StoredAddress::new(u64::MAX),
            managed_space: 0,
            allocated_managed_space: 0,
            direct_block_allocation_iterator_offset: 0,
            managed_objects_count: 0,
            huge_objects_size: 0,
            huge_objects_count: 0,
            tiny_objects_size: 0,
            tiny_objects_count: 0,
            table_width,
            starting_block_size: start_block_size,
            max_direct_block_size,
            max_heap_size: 40,
            start_root_rows: 1,
            root_block_address: StoredAddress::new(0),
            current_rows_in_root_indirect_block: 0,
        }
    }

    fn dtable_layout(header: &FractalHeapHeader) -> FractalHeapLayout {
        FractalHeapLayout::new(header, OffsetWidth::Eight, LengthWidth::Eight).unwrap()
    }

    #[test]
    fn layout_derives_direct_rows_from_the_doubling_table() {
        assert_eq!(
            dtable_layout(&dtable_header(512, 65536, 4)).direct_rows(),
            9
        );
        assert_eq!(
            dtable_layout(&dtable_header(4096, 65536, 4)).direct_rows(),
            6
        );
        let smallest = FractalHeapHeader {
            max_heap_size: 12,
            ..dtable_header(512, 512, 4)
        };
        assert_eq!(dtable_layout(&smallest).direct_rows(), 2);
    }

    #[test]
    fn layout_derives_child_indirect_rows_from_the_slot_size() {
        let layout = dtable_layout(&dtable_header(512, 65536, 4));
        assert_eq!(layout.block_size_for_row(9), Ok(131072));
        assert_eq!(layout.child_indirect_rows(9), Ok(7));
    }

    #[test]
    fn an_indirect_block_locates_the_child_that_holds_an_offset() {
        let header = FractalHeapHeader::parse(
            &fractal_heap::Header::new(0x100).build(Widths::EIGHT),
            0,
            8,
            8,
        )
        .unwrap();
        let mut block = fractal_heap::INDIRECT_BLOCK_SIGNATURE.to_vec();
        block.push(0);
        block.extend_from_slice(&0u64.to_le_bytes());
        block.extend_from_slice(&[0, 0]);
        for address in [0x1000u64, u64::MAX, 0x2000, u64::MAX] {
            block.extend_from_slice(&address.to_le_bytes());
        }

        let layout =
            FractalHeapLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight).unwrap();
        assert_eq!(layout.indirect_block_entries_len(1), Ok(block.len() as u64));
        assert_eq!(
            layout.find_child_for_offset(&block, 1, 0, 300),
            Ok(Some(FractalHeapChild::Direct {
                addr: StoredAddress::new(0x2000),
                block_size: 128,
                heap_offset: 256,
            }))
        );
        assert_eq!(layout.find_child_for_offset(&block, 1, 0, 200), Ok(None));
    }

    #[rstest]
    #[case::eight_byte_fields(Widths::EIGHT, OffsetWidth::Eight, LengthWidth::Eight)]
    #[case::four_byte_fields(Widths::FOUR, OffsetWidth::Four, LengthWidth::Four)]
    fn a_parsed_header_serializes_to_the_bytes_it_was_parsed_from(
        #[case] widths: Widths,
        #[case] offset_width: OffsetWidth,
        #[case] length_width: LengthWidth,
    ) {
        let bytes = fractal_heap::Header::new(0x100)
            .managed_object_count(3)
            .build(widths);
        let header =
            FractalHeapHeader::parse(&bytes, 0, offset_width.get(), length_width.get()).unwrap();

        assert_eq!(
            header.serialize(offset_width, length_width),
            Ok(bytes.clone())
        );
        assert_eq!(
            FractalHeapHeader::serialized_size(offset_width, length_width),
            bytes.len()
        );
    }

    #[test]
    fn serializing_a_filtered_header_is_an_error() {
        let header = FractalHeapHeader {
            filtering: filtered_heap(),
            ..dtable_header(512, 65536, 4)
        };
        assert_eq!(
            header.serialize(OffsetWidth::Eight, LengthWidth::Eight),
            Err(FormatError::UnsupportedFilteredHeapObject)
        );
    }

    #[rstest]
    #[case::the_first_object(0, 13, [0, 0, 0, 0, 0, 0, 13, 0])]
    #[case::an_offset_and_a_length(100, 42, [0, 100, 0, 0, 0, 0, 42, 0])]
    #[case::the_widest_offset_and_length(
        (1 << 40) - 1,
        u64::from(u16::MAX),
        [0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
    )]
    fn an_encoded_managed_id_holds_its_offset_then_its_length(
        #[case] heap_offset: u64,
        #[case] length: u64,
        #[case] expected: [u8; 8],
    ) {
        let header = attribute_heap_header();
        let id = header.encode_managed_id(heap_offset, length).unwrap();
        assert_eq!(id, expected);
        assert_eq!(
            FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
                .parse(&id)
                .unwrap()
                .kind(),
            FractalHeapIdKind::Managed {
                heap_offset,
                object_length: length
            },
        );
    }

    #[rstest]
    #[case::an_offset_past_the_offset_bits(
        1 << 40,
        1,
        "heap offset 1099511627776 does not fit the heap's 40-bit offsets"
    )]
    #[case::a_length_past_the_id(0, 1 << 16, "a managed object length does not fit a 8-byte heap ID")]
    fn a_managed_id_that_does_not_fit_is_an_error(
        #[case] heap_offset: u64,
        #[case] length: u64,
        #[case] message: &str,
    ) {
        assert_eq!(
            attribute_heap_header().encode_managed_id(heap_offset, length),
            Err(FormatError::Internal(message.into()))
        );
    }

    #[test]
    fn a_managed_id_of_a_heap_whose_offset_bits_are_not_a_multiple_of_8_is_an_error() {
        let header = FractalHeapHeader {
            max_heap_size: 33,
            ..attribute_heap_header()
        };
        assert_eq!(
            header.encode_managed_id(0, 1),
            Err(FormatError::Internal(
                "a managed heap ID was encoded for a heap of 33-bit offsets, which fill no whole \
                 number of bytes"
                    .into()
            ))
        );
    }

    #[test]
    fn an_encoded_huge_id_decodes_to_its_key() {
        let header = attribute_heap_header();
        let id = header
            .encode_huge_id(5, OffsetWidth::Eight, LengthWidth::Eight)
            .unwrap();
        assert_eq!(id, [0x10, 5, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
                .parse(&id)
                .map(FractalHeapIdView::kind),
            Ok(FractalHeapIdKind::Huge(HugeObjectReference::Indexed(5)))
        );
    }

    #[rstest]
    #[case::a_key_past_the_id(
        attribute_heap_header(),
        1 << 56,
        FormatError::Internal("a huge object key does not fit a 8-byte heap ID".into())
    )]
    #[case::an_id_that_holds_the_address(
        FractalHeapHeader { heap_id_length: 17, ..attribute_heap_header() },
        1,
        FormatError::Internal("a key was encoded into a heap ID that holds its object's address".into())
    )]
    fn a_huge_id_the_heap_cannot_hold_is_an_error(
        #[case] header: FractalHeapHeader,
        #[case] huge_id: u64,
        #[case] expected: FormatError,
    ) {
        assert_eq!(
            header.encode_huge_id(huge_id, OffsetWidth::Eight, LengthWidth::Eight),
            Err(expected)
        );
    }

    struct HeaderSource<'a> {
        bytes: &'a [u8],
        prefix: Option<[u8; FractalHeapHeaderFrame::PREFIX_LEN]>,
        reads: RefCell<Vec<(u64, usize)>>,
    }

    impl MetadataSource for HeaderSource<'_> {
        fn len(&self) -> u64 {
            self.bytes.len().to_u64()
        }
        fn read_at(&self, address: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
            self.bytes.read_at(address, bytes)
        }
        fn read_metadata_at(&self, address: u64, len: usize) -> Result<Vec<u8>, FormatError> {
            self.reads.borrow_mut().push((address, len));
            if len == FractalHeapHeaderFrame::PREFIX_LEN {
                if let Some(prefix) = self.prefix {
                    return Ok(prefix.to_vec());
                }
            }
            self.bytes.read_metadata_at(address, len)
        }
    }

    fn filter_pipeline(version: u8, client_count: usize) -> FilterPipeline {
        FilterPipeline {
            version,
            filters: vec![FilterDescription {
                filter_id: FILTER_DEFLATE,
                name: if version == PIPELINE_VERSION_ONE {
                    Some("deflate".into())
                } else {
                    None
                },
                flags: 0,
                client_data: vec![6; client_count],
            }],
        }
    }

    fn filtered_heap() -> FractalHeapFiltering {
        FractalHeapFiltering::Filtered {
            root_direct_block_size: 91,
            root_filter_mask: 1,
            pipeline: FilterPipeline {
                version: PIPELINE_VERSION_TWO,
                filters: vec![FilterDescription {
                    filter_id: FILTER_DEFLATE,
                    name: None,
                    flags: 0,
                    client_data: vec![6],
                }],
            },
        }
    }

    /// Returns a header with the heap ID layout of the attribute heaps `hdf5-pure` writes: 8-byte
    /// IDs over a 40-bit heap, with a huge-object B-tree.
    fn attribute_heap_header() -> FractalHeapHeader {
        FractalHeapHeader {
            heap_id_length: 8,
            max_heap_size: 40,
            btree_huge_objects_address: StoredAddress::new(0x800),
            ..dtable_header(1024, 65536, 4)
        }
    }

    fn header_prefix(filter_encoded_len: u16) -> Vec<u8> {
        let mut prefix = FRACTAL_HEAP_SIGNATURE.to_vec();
        prefix.push(FRACTAL_HEAP_VERSION);
        prefix.extend_from_slice(&8u16.to_le_bytes());
        prefix.extend_from_slice(&filter_encoded_len.to_le_bytes());
        prefix
    }

    const HEADER_VERSION_END: usize = FRACTAL_HEAP_SIGNATURE.len() + mem::size_of::<u8>();
    const HEADER_ID_LENGTH_END: usize = HEADER_VERSION_END + mem::size_of::<u16>();
    // "The Data Storage - Filter Pipeline Message", format specification version 4.0, defines
    // versions 1 and 2 and deflate ID 1.
    const PIPELINE_VERSION_ONE: u8 = 1;
    const PIPELINE_VERSION_TWO: u8 = 2;
    const FILTER_DEFLATE: u16 = 1;
    // "The Data Storage - Filter Pipeline Message", format specification version 4.0, limits
    // pipelines to 32 filters.
    const FILTER_COUNT_MAXIMUM: u8 = 32;
}
