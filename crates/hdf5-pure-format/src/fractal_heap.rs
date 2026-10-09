//! Parses and encodes fractal heap headers and heap IDs, derives doubling-table geometry, and
//! locates children of indirect blocks.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

#[cfg(feature = "checksum")]
use byteorder::ByteOrder;
#[cfg(feature = "checksum")]
use byteorder::LittleEndian;

use crate::address::StoredAddress;
use crate::bytes;
use crate::convert;
use crate::convert::Narrow;
use crate::error::FormatError;
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

/// Describes a failure to prove complete ownership of a fractal heap's file storage.
///
/// This type is part of the workspace-private structural API exposed through `__private`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FractalHeapStorageError {
    /// A lower-level HDF5 parser or source read failed.
    Format(FormatError),
    /// The heap's ownership graph or doubling-table geometry is inconsistent.
    InvalidStorage,
    /// The named heap feature prevents a complete ownership proof.
    UnsupportedOwnership(&'static str),
}

impl From<FormatError> for FractalHeapStorageError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}

/// A fractal heap header, signature `FRHP`: the parameters of the heap's doubling table, the
/// address of its root block, and the count of its managed objects.
///
/// A group in dense storage keeps its links in a fractal heap, an object in dense storage keeps
/// its attributes in one, and a file that shares messages keeps the messages of each index in one.
/// [`parse`](Self::parse) accepts version 0, the one version the specification defines, and keeps
/// every field of the header of a heap that does not filter its objects, which
/// [`serialize`](Self::serialize) writes back.
///
/// The header is defined in "Fractal Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FractalHeapHeader {
    /// The length in bytes of the heap's IDs.
    pub heap_id_length: u16,
    /// The size in bytes of the encoded I/O filter pipeline, 0 for a heap that does not filter its
    /// objects.
    pub io_filter_encoded_length: u16,
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
}

/// Returns the base-2 logarithm of `v` rounded down, and 0 for 0, as `H5VM_log2_gen` does.
///
/// The block sizes of a doubling table are powers of two, so the logarithm of one is exact.
fn log2_floor(v: u64) -> u32 {
    if v == 0 { 0 } else { 63 - v.leading_zeros() }
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
    /// Lengths" bytes. With the `checksum` feature, `parse` also verifies the header's checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapSignature`] if the header does not begin with
    /// `FRHP`, [`FormatError::InvalidFractalHeapVersion`] if its version is not 0,
    /// [`FormatError::UnexpectedEof`] if it runs past the end of `file_data`,
    /// [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if a width is not
    /// 2, 4, or 8, and, with the `checksum` feature, [`FormatError::ChecksumMismatch`] if the
    /// stored checksum differs from the computed one.
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<FractalHeapHeader, FormatError> {
        bytes::ensure_len(file_data, offset, 5)?;
        if file_data[offset..offset + 4] != FRACTAL_HEAP_SIGNATURE {
            return Err(FormatError::InvalidFractalHeapSignature);
        }

        let version = file_data[offset + 4];
        if version != FRACTAL_HEAP_VERSION {
            return Err(FormatError::InvalidFractalHeapVersion(version));
        }

        let offsets = OffsetWidth::try_from(offset_size)?;
        let lengths = LengthWidth::try_from(length_size)?;
        let mut fields = bytes::Fields::new(file_data, offset + 5);
        let heap_id_length = fields.u16()?;
        let io_filter_encoded_length = fields.u16()?;
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
        let mut pos = fields.pos();

        // Skip the size of the filtered root direct block and its filter mask.
        if io_filter_encoded_length > 0 {
            pos += usize::from(length_size) + 4;
        }

        #[cfg(feature = "checksum")]
        {
            bytes::ensure_len(file_data, pos, 4)?;
            let stored = LittleEndian::read_u32(&file_data[pos..pos + 4]);
            let computed = crate::checksum::jenkins_lookup3(&file_data[offset..pos]);
            if computed != stored {
                return Err(FormatError::ChecksumMismatch {
                    expected: stored,
                    computed,
                });
            }
        }
        #[cfg(not(feature = "checksum"))]
        let _ = pos;

        Ok(FractalHeapHeader {
            heap_id_length,
            io_filter_encoded_length,
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

    /// Returns the bytes of the header from its signature to its checksum, the inverse of
    /// [`parse`](Self::parse).
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedFilteredHeapObject`] if the heap filters its objects,
    /// whose header stores the filter fields this type does not hold.
    pub fn serialize(
        &self,
        offset_width: OffsetWidth,
        length_width: LengthWidth,
    ) -> Result<Vec<u8>, FormatError> {
        let Self {
            heap_id_length,
            io_filter_encoded_length,
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
        if io_filter_encoded_length > 0 {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }
        let mut buf = Vec::with_capacity(Self::serialized_size(offset_width, length_width));
        buf.extend_from_slice(&FRACTAL_HEAP_SIGNATURE);
        buf.push(FRACTAL_HEAP_VERSION);
        buf.extend_from_slice(&heap_id_length.to_le_bytes());
        buf.extend_from_slice(&io_filter_encoded_length.to_le_bytes());
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
        if self.io_filter_encoded_length > 0 {
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
        if self.io_filter_encoded_length > 0 {
            avail >= offset_size as usize + length_size as usize + 4 + length_size as usize
        } else {
            avail >= offset_size as usize + length_size as usize
        }
    }

    /// Returns the child of an indirect block whose space holds heap offset `target_offset`, or
    /// `None` if no allocated child holds it.
    ///
    /// `block` holds the indirect block from its signature on, `nrows` is its number of rows, and
    /// `iblock_heap_offset` the heap offset its space begins at. The heap offset of each child
    /// follows from the doubling table, so the function reads the child addresses alone.
    /// `offset_size` is the superblock's "Size of Offsets" byte.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFractalHeapSignature`] if `block` does not begin with `FHIB`,
    /// [`FormatError::UnsupportedFilteredHeapObject`] if the heap filters its objects,
    /// [`FormatError::UnexpectedEof`] if an entry the function reads runs past the end of `block`,
    /// and [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8.
    pub fn find_child_for_offset(
        &self,
        block: &[u8],
        nrows: u16,
        iblock_heap_offset: u64,
        target_offset: u64,
        offset_size: u8,
    ) -> Result<Option<FractalHeapChild>, FormatError> {
        bytes::ensure_len(block, 0, 4)?;
        if &block[0..4] != b"FHIB" {
            return Err(FormatError::InvalidFractalHeapSignature);
        }

        // A filtered heap follows each direct block address with the filtered size of the block
        // (`length_size` bytes) and its filter mask (4 bytes), and encodes the contents of the
        // block. At the stride of an unfiltered heap, every later child address would be read at
        // the wrong offset, so the function rejects the heap.
        if self.io_filter_encoded_length > 0 {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }

        let block_offset_bytes = (self.max_heap_size as usize).div_ceil(8);
        let iblock_header = 5 + offset_size as usize + block_offset_bytes;
        let mut pos = iblock_header;
        let tw = self.table_width as u64;
        let nrows_usize = nrows as usize;
        let direct_rows = nrows_usize.min(self.max_direct_rows());
        let mut current_heap_offset = iblock_heap_offset;

        // Direct-block rows.
        for row in 0..direct_rows {
            let block_size = self.block_size_for_row(row);
            for _col in 0..tw {
                let child_addr = bytes::read_offset(block, pos, offset_size)?;
                pos += offset_size as usize;
                if !convert::is_undefined_addr(child_addr, offset_size) {
                    let block_end = current_heap_offset.saturating_add(block_size);
                    if target_offset >= current_heap_offset && target_offset < block_end {
                        return Ok(Some(FractalHeapChild::Direct {
                            addr: StoredAddress::new(child_addr),
                            block_size,
                            heap_offset: current_heap_offset,
                        }));
                    }
                }
                current_heap_offset = current_heap_offset.saturating_add(block_size);
            }
        }

        // Indirect-block rows. A child indirect block in row `row` has the block size of that
        // row, and its own row count and the heap space it spans follow from that size.
        for row in direct_rows..nrows_usize {
            let child_nrows = self.size_to_rows(self.block_size_for_row(row));
            let total_child_space = self.indirect_block_heap_size(child_nrows);
            for _col in 0..tw {
                let child_addr = bytes::read_offset(block, pos, offset_size)?;
                pos += offset_size as usize;
                if !convert::is_undefined_addr(child_addr, offset_size) {
                    let block_end = current_heap_offset.saturating_add(total_child_space);
                    if target_offset >= current_heap_offset && target_offset < block_end {
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "fractal-heap row count is log-scale (bounded by \
                                      max_heap_size bits), so it fits u16"
                        )]
                        return Ok(Some(FractalHeapChild::Indirect {
                            addr: StoredAddress::new(child_addr),
                            nrows: child_nrows as u16,
                            heap_offset: current_heap_offset,
                        }));
                    }
                }
                current_heap_offset = current_heap_offset.saturating_add(total_child_space);
            }
        }

        Ok(None)
    }

    /// Returns the size in bytes of an indirect block of `nrows` rows up to the end of its child
    /// entries, the bytes [`find_child_for_offset`](Self::find_child_for_offset) reads.
    ///
    /// `offset_size` is the superblock's "Size of Offsets" byte.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedFilteredHeapObject`] if the heap filters its objects.
    pub fn indirect_block_entries_len(
        &self,
        nrows: u16,
        offset_size: u8,
    ) -> Result<u64, FormatError> {
        if self.io_filter_encoded_length > 0 {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }
        let block_offset_bytes = u64::from(self.max_heap_size).div_ceil(8);
        let iblock_header = 5 + u64::from(offset_size) + block_offset_bytes;
        // In an unfiltered heap every entry, direct or indirect, is one child address.
        let entries = u64::from(nrows) * u64::from(self.table_width) * u64::from(offset_size);
        Ok(iblock_header + entries)
    }

    /// Returns the width in bytes of the "Block Offset" field in a managed block header.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapStorageError::InvalidStorage`] if the doubling-table fields do not
    /// define representable geometry.
    pub fn storage_block_offset_size(&self) -> Result<usize, FractalHeapStorageError> {
        self.storage_geometry()?;
        Ok(usize::from(self.max_heap_size).div_ceil(8))
    }

    /// Returns the exact allocated size of a managed direct block in doubling-table `row`.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapStorageError::InvalidStorage`] for impossible geometry or arithmetic
    /// overflow.
    pub fn storage_block_size_for_row(&self, row: usize) -> Result<u64, FractalHeapStorageError> {
        self.storage_geometry()?;
        if row <= 1 {
            return Ok(self.starting_block_size);
        }
        let shift = u32::try_from(row - 1)
            .map_err(|_conversion_error| FractalHeapStorageError::InvalidStorage)?;
        let multiplier = 1u64
            .checked_shl(shift)
            .ok_or(FractalHeapStorageError::InvalidStorage)?;
        self.starting_block_size
            .checked_mul(multiplier)
            .ok_or(FractalHeapStorageError::InvalidStorage)
    }

    /// Returns the number of doubling-table rows whose entries address direct blocks.
    ///
    /// The configured maximum direct-block size can describe more direct rows than the maximum
    /// root can contain. A caller limits this count to the rows represented by the indirect block
    /// it traverses.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapStorageError::InvalidStorage`] for impossible doubling-table geometry.
    pub fn storage_direct_rows(&self) -> Result<usize, FractalHeapStorageError> {
        let (start_bits, max_direct_bits, _) = self.storage_geometry()?;
        usize::try_from(max_direct_bits - start_bits + 2)
            .map_err(|_conversion_error| FractalHeapStorageError::InvalidStorage)
    }

    /// Returns the maximum legal number of rows in the root indirect block.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapStorageError::InvalidStorage`] for impossible doubling-table geometry.
    pub fn storage_max_root_rows(&self) -> Result<u16, FractalHeapStorageError> {
        let (_, _, rows) = self.storage_geometry()?;
        Ok(rows)
    }

    /// Returns the row count of the child indirect blocks addressed by doubling-table `row`.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapStorageError::InvalidStorage`] if `row` does not identify an indirect
    /// row, or if the declared geometry is impossible.
    pub fn storage_child_indirect_rows(&self, row: usize) -> Result<u16, FractalHeapStorageError> {
        if row < self.storage_direct_rows()? {
            return Err(FractalHeapStorageError::InvalidStorage);
        }
        let size = self.storage_block_size_for_row(row)?;
        let start_bits = self.starting_block_size.trailing_zeros();
        let width_bits = u64::from(self.table_width).trailing_zeros();
        let first_row_bits = start_bits
            .checked_add(width_bits)
            .ok_or(FractalHeapStorageError::InvalidStorage)?;
        let size_bits = size.trailing_zeros();
        if !size.is_power_of_two() || size_bits < first_row_bits {
            return Err(FractalHeapStorageError::InvalidStorage);
        }
        u16::try_from(size_bits - first_row_bits + 1)
            .map_err(|_conversion_error| FractalHeapStorageError::InvalidStorage)
    }

    /// Returns the managed heap-address space represented by an indirect block of `nrows` rows.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapStorageError::InvalidStorage`] for an impossible row count or arithmetic
    /// overflow.
    pub fn storage_indirect_heap_size(&self, nrows: u16) -> Result<u64, FractalHeapStorageError> {
        if nrows == 0 || nrows > self.storage_max_root_rows()? {
            return Err(FractalHeapStorageError::InvalidStorage);
        }
        let width = u64::from(self.table_width);
        let mut total = 0u64;
        for row in 0..usize::from(nrows) {
            total = total
                .checked_add(
                    self.storage_block_size_for_row(row)?
                        .checked_mul(width)
                        .ok_or(FractalHeapStorageError::InvalidStorage)?,
                )
                .ok_or(FractalHeapStorageError::InvalidStorage)?;
        }
        Ok(total)
    }

    /// Returns the exact encoded allocation size of an unfiltered indirect block.
    ///
    /// The size includes the `FHIB` signature, version, heap-header address, block heap offset,
    /// every child address, and the trailing checksum.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapStorageError::UnsupportedOwnership`] for a filtered heap,
    /// [`FractalHeapStorageError::Format`] for an unsupported address width, and
    /// [`FractalHeapStorageError::InvalidStorage`] for impossible geometry or arithmetic overflow.
    pub fn storage_indirect_block_size(
        &self,
        nrows: u16,
        offset_size: u8,
    ) -> Result<u64, FractalHeapStorageError> {
        if self.io_filter_encoded_length > 0 {
            return Err(FractalHeapStorageError::UnsupportedOwnership(
                "filtered managed blocks",
            ));
        }
        OffsetWidth::try_from(offset_size)?;
        if nrows == 0 || nrows > self.storage_max_root_rows()? {
            return Err(FractalHeapStorageError::InvalidStorage);
        }
        let block_offset_size = u64::try_from(self.storage_block_offset_size()?)
            .map_err(|_conversion_error| FractalHeapStorageError::InvalidStorage)?;
        let prefix = 5u64
            .checked_add(u64::from(offset_size))
            .and_then(|n| n.checked_add(block_offset_size))
            .ok_or(FractalHeapStorageError::InvalidStorage)?;
        let entries = u64::from(nrows)
            .checked_mul(u64::from(self.table_width))
            .and_then(|n| n.checked_mul(u64::from(offset_size)))
            .ok_or(FractalHeapStorageError::InvalidStorage)?;
        prefix
            .checked_add(entries)
            .and_then(|n| n.checked_add(4))
            .ok_or(FractalHeapStorageError::InvalidStorage)
    }

    /// Returns the heap offset of the first slot in `row` of an indirect block.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapStorageError::InvalidStorage`] for impossible geometry or arithmetic
    /// overflow.
    pub fn storage_row_offset(&self, row: usize) -> Result<u64, FractalHeapStorageError> {
        self.storage_geometry()?;
        let width = u64::from(self.table_width);
        let mut offset = 0u64;
        for previous in 0..row {
            offset = offset
                .checked_add(
                    self.storage_block_size_for_row(previous)?
                        .checked_mul(width)
                        .ok_or(FractalHeapStorageError::InvalidStorage)?,
                )
                .ok_or(FractalHeapStorageError::InvalidStorage)?;
        }
        Ok(offset)
    }

    /// Returns the derived bit widths and maximum root-row count of valid doubling-table geometry.
    ///
    /// # Errors
    ///
    /// Returns [`FractalHeapStorageError::InvalidStorage`] if the table fields contradict the
    /// fractal-heap doubling-table constraints or cannot be represented without overflow.
    fn storage_geometry(&self) -> Result<(u32, u32, u16), FractalHeapStorageError> {
        if self.table_width == 0
            || !self.table_width.is_power_of_two()
            || self.starting_block_size == 0
            || !self.starting_block_size.is_power_of_two()
            || self.max_direct_block_size < self.starting_block_size
            || !self.max_direct_block_size.is_power_of_two()
            || self.max_heap_size == 0
            || self.max_heap_size > 64
        {
            return Err(FractalHeapStorageError::InvalidStorage);
        }
        let start_bits = self.starting_block_size.trailing_zeros();
        let max_direct_bits = self.max_direct_block_size.trailing_zeros();
        let first_row_bits = start_bits
            .checked_add(u64::from(self.table_width).trailing_zeros())
            .ok_or(FractalHeapStorageError::InvalidStorage)?;
        let heap_bits = u32::from(self.max_heap_size);
        if heap_bits < first_row_bits {
            return Err(FractalHeapStorageError::InvalidStorage);
        }
        let max_root_rows = u16::try_from(heap_bits - first_row_bits + 1)
            .map_err(|_conversion_error| FractalHeapStorageError::InvalidStorage)?;
        if self.start_root_rows > max_root_rows {
            return Err(FractalHeapStorageError::InvalidStorage);
        }
        Ok((start_bits, max_direct_bits, max_root_rows))
    }

    /// Parses the fractal heap header at `address` in `source`.
    ///
    /// Reads at most 256 bytes, which hold the header of an unfiltered heap at every width the
    /// format allows, and fewer where the file ends first, then parses them as
    /// [`parse`](Self::parse) does.
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
        const MAX_HEADER: u64 = 256;
        let window = MAX_HEADER
            .min(source.len().saturating_sub(address))
            .to_usize()?;
        let buf = source.read_metadata_at(address, window)?;
        Self::parse(&buf, 0, offset_size, length_size)
    }

    /// Returns the size of the blocks in row `row` of the doubling table, saturated at `u64::MAX`.
    ///
    /// A damaged header can declare any number of rows, and the function saturates where the shift
    /// or the product would overflow.
    fn block_size_for_row(&self, row: usize) -> u64 {
        let sbs = self.starting_block_size;
        if row <= 1 {
            sbs
        } else {
            match u32::try_from(row - 1)
                .ok()
                .and_then(|s| 1u64.checked_shl(s))
            {
                Some(mult) => sbs.saturating_mul(mult),
                None => u64::MAX,
            }
        }
    }

    /// Returns the number of rows of the doubling table that hold direct blocks, the rows before
    /// the first row of indirect blocks.
    ///
    /// The count is `(max_direct_bits - start_bits) + 2`, as `H5HF__dtable_init` in
    /// `H5HFdtable.c` (HDF5 2.2.0) computes it.
    fn max_direct_rows(&self) -> usize {
        let start_bits = log2_floor(self.starting_block_size);
        let max_direct_bits = log2_floor(self.max_direct_block_size);
        (max_direct_bits.saturating_sub(start_bits) + 2) as usize
    }

    /// Returns the number of rows of an indirect block that spans `size` bytes of heap space.
    ///
    /// The count is `(log2(size) - first_row_bits) + 1`, where `first_row_bits` is
    /// `log2(starting_block_size) + log2(table_width)`, as `H5HF__dtable_size_to_rows` computes it.
    fn size_to_rows(&self, size: u64) -> usize {
        let first_row_bits =
            log2_floor(self.starting_block_size) + log2_floor(self.table_width as u64);
        (log2_floor(size).saturating_sub(first_row_bits) + 1) as usize
    }

    /// Returns the heap space an indirect block of `nrows` rows spans, saturated at `u64::MAX`.
    fn indirect_block_heap_size(&self, nrows: usize) -> u64 {
        let tw = self.table_width as u64;
        let mut total = 0u64;
        for row in 0..nrows {
            total = total.saturating_add(self.block_size_for_row(row).saturating_mul(tw));
        }
        total
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
        let huge = if header.io_filter_encoded_length > 0 {
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

#[cfg(test)]
mod tests {
    use core::mem;

    use rstest::rstest;
    use test_util::fractal_heap;
    use test_util::widths::Widths;

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

    #[test]
    fn a_header_parses_to_its_fields() {
        let file_data = fractal_heap::heap_with_one_object(b"Hello, World!", Widths::EIGHT);
        let expected = FractalHeapHeader {
            heap_id_length: 7,
            io_filter_encoded_length: 0,
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
            io_filter_encoded_length: u16::from(filtered),
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
        h.io_filter_encoded_length = 4;
        assert!(!h.huge_ids_direct(8, 8));
    }

    #[test]
    fn the_indirect_block_walk_of_a_filtered_heap_is_unsupported() {
        let mut h = dtable_header(512, 65536, 4);
        h.io_filter_encoded_length = 8;
        let mut block = b"FHIB".to_vec();
        block.resize(256, 0);
        assert_eq!(
            h.indirect_block_entries_len(2, 8),
            Err(FormatError::UnsupportedFilteredHeapObject)
        );
        assert_eq!(
            h.find_child_for_offset(&block, 2, 0, 0, 8),
            Err(FormatError::UnsupportedFilteredHeapObject)
        );

        // In an unfiltered heap, the function returns `None` for an offset past the block's space.
        h.io_filter_encoded_length = 0;
        assert_eq!(h.find_child_for_offset(&block, 2, 0, u64::MAX, 8), Ok(None));
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

    #[test]
    fn log2_floor_rounds_down_and_maps_0_to_0() {
        assert_eq!(log2_floor(0), 0);
        assert_eq!(log2_floor(1), 0);
        assert_eq!(log2_floor(512), 9);
        assert_eq!(log2_floor(65536), 16);
        assert_eq!(log2_floor(131072), 17);
        // Rounded down, as `H5VM_log2_gen` rounds.
        assert_eq!(log2_floor(1023), 9);
        assert_eq!(log2_floor(1024), 10);
    }

    /// Returns a header with these doubling-table parameters, the fields `max_direct_rows` and
    /// `size_to_rows` read.
    fn dtable_header(
        start_block_size: u64,
        max_direct_block_size: u64,
        table_width: u16,
    ) -> FractalHeapHeader {
        FractalHeapHeader {
            heap_id_length: 7,
            io_filter_encoded_length: 0,
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
            max_heap_size: 64,
            start_root_rows: 1,
            root_block_address: StoredAddress::new(0),
            current_rows_in_root_indirect_block: 0,
        }
    }

    #[test]
    fn max_direct_rows_matches_hdf5_formula() {
        // `(log2(max_direct) - log2(start)) + 2`, as `H5HF__dtable_init` computes it.
        assert_eq!(dtable_header(512, 65536, 4).max_direct_rows(), 9); // (16-9)+2
        assert_eq!(dtable_header(4096, 65536, 4).max_direct_rows(), 6); // (16-12)+2
        // A largest direct block of the starting size gives the fewest direct rows, 2.
        assert_eq!(dtable_header(512, 512, 4).max_direct_rows(), 2);
    }

    #[test]
    fn size_to_rows_matches_hdf5_formula() {
        // `first_row_bits = log2(start) + log2(width)`. For start=512, width=4:
        // `first_row_bits = 9 + 2 = 11`, rows = (log2(size) - 11) + 1.
        let h = dtable_header(512, 65536, 4);
        assert_eq!(h.size_to_rows(131072), 7); // 2^17: (17-11)+1
        assert_eq!(h.size_to_rows(4096), 2); // 2^12: (12-11)+1
        // A size below `first_row_bits` saturates to one row.
        assert_eq!(h.size_to_rows(512), 1);
        assert_eq!(h.size_to_rows(1), 1);
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

        assert_eq!(
            header.indirect_block_entries_len(1, 8),
            Ok(block.len() as u64)
        );
        assert_eq!(
            header.find_child_for_offset(&block, 1, 0, 300, 8),
            Ok(Some(FractalHeapChild::Direct {
                addr: StoredAddress::new(0x2000),
                block_size: 128,
                heap_offset: 256,
            }))
        );
        assert_eq!(header.find_child_for_offset(&block, 1, 0, 200, 8), Ok(None));
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
            io_filter_encoded_length: 8,
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
}
