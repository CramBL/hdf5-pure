//! The fractal heap header and the heap IDs, each with its parser and its encoder, and the lookup
//! of a child in an indirect block.

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
pub enum FractalHeapIdType {
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
    pub fn from_heap_id(id_bytes: &[u8]) -> Result<Self, FormatError> {
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

        let mut fields = HeaderFields {
            data: file_data,
            pos: offset + 5,
            offset_size,
            length_size,
        };
        let heap_id_length = fields.u16()?;
        let io_filter_encoded_length = fields.u16()?;
        let flags = fields.u8()?;
        let max_managed_object_size = fields.u32()?;
        let next_huge_object_id = fields.length()?;
        let btree_huge_objects_address = fields.address()?;
        let free_space_in_managed_blocks = fields.length()?;
        let managed_block_free_space_manager_address = fields.address()?;
        let managed_space = fields.length()?;
        let allocated_managed_space = fields.length()?;
        let direct_block_allocation_iterator_offset = fields.length()?;
        let managed_objects_count = fields.length()?;
        let huge_objects_size = fields.length()?;
        let huge_objects_count = fields.length()?;
        let tiny_objects_size = fields.length()?;
        let tiny_objects_count = fields.length()?;
        let table_width = fields.u16()?;
        let starting_block_size = fields.length()?;
        let max_direct_block_size = fields.length()?;
        let max_heap_size = fields.u16()?;
        let start_root_rows = fields.u16()?;
        let root_block_address = fields.address()?;
        let current_rows_in_root_indirect_block = fields.u16()?;
        let mut pos = fields.pos;

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

    /// Returns the length in bytes of the header of a heap that does not filter its objects, from
    /// its signature to its checksum.
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

    /// Returns the heap offset and the length of the managed object the heap ID `id_bytes` refers
    /// to.
    ///
    /// After its first byte, a managed heap ID holds the offset of the object in
    /// [`max_heap_size`](Self::max_heap_size) bits, then its length in the bits that remain, both
    /// little-endian. The function reads 8 bytes after the first at most.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `id_bytes` is empty, and
    /// [`FormatError::InvalidHeapIdType`] if its type is not managed.
    pub fn decode_managed_id(&self, id_bytes: &[u8]) -> Result<(u64, u64), FormatError> {
        if id_bytes.is_empty() {
            return Err(FormatError::UnexpectedEof {
                expected: 1,
                available: 0,
            });
        }

        let id_type = (id_bytes[0] >> 4) & 0x03;
        if id_type != 0 {
            return Err(FormatError::InvalidHeapIdType(id_type));
        }

        let payload = &id_bytes[1..];
        let mut combined: u64 = 0;
        for (i, &b) in payload.iter().enumerate() {
            if i >= 8 {
                break;
            }
            combined |= (b as u64) << (i * 8);
        }

        let offset_bits = self.max_heap_size as u32;
        let offset_mask = if offset_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << offset_bits) - 1
        };
        let heap_offset = combined & offset_mask;

        #[expect(
            clippy::cast_possible_truncation,
            reason = "payload is a single managed-object heap entry; its bit length fits u32"
        )]
        let total_payload_bits = (payload.len() as u32) * 8;
        let length_bits = total_payload_bits.saturating_sub(offset_bits);
        let length_val = if length_bits == 0 {
            0
        } else {
            let length_mask = if length_bits >= 64 {
                u64::MAX
            } else {
                (1u64 << length_bits) - 1
            };
            (combined >> offset_bits) & length_mask
        };

        Ok((heap_offset, length_val))
    }

    /// Returns the heap ID of the managed object of `length` bytes at heap offset `heap_offset`.
    ///
    /// After its first byte, the ID stores the offset in [`max_heap_size`](Self::max_heap_size)
    /// bits and then the length, as [`decode_managed_id`](Self::decode_managed_id) reads them.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `max_heap_size` is not a multiple of 8, or if
    /// `heap_offset` or `length` does not fit the ID. The C library stores the offset in whole
    /// bytes (`H5HF_MAN_ID_ENCODE` in `H5HFpkg.h`, HDF5 2.2.0), and
    /// [`decode_managed_id`](Self::decode_managed_id) reads it as `max_heap_size` packed bits.
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

    /// Returns the tiny object the heap ID `id_bytes` holds.
    ///
    /// A tiny heap ID holds the length of the object, less one, and then the object. In a heap
    /// whose [`heap_id_length`](Self::heap_id_length) is 17 or less, the function reads the
    /// length from the low four bits of the first byte, the normal form. In a heap with longer IDs,
    /// it reads a 12-bit length from the low four bits of the first byte and the whole second
    /// byte, the extended form.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `id_bytes` ends before the length or the object.
    pub fn decode_tiny_id(&self, id_bytes: &[u8]) -> Result<Vec<u8>, FormatError> {
        const TINY_LEN_SHORT: u16 = 16;
        let Some((&first, _)) = id_bytes.split_first() else {
            return Err(FormatError::UnexpectedEof {
                expected: 1,
                available: 0,
            });
        };
        let extended = self.heap_id_length.saturating_sub(1) > TINY_LEN_SHORT;
        let (len, data_start) = if extended {
            if id_bytes.len() < 2 {
                return Err(FormatError::UnexpectedEof {
                    expected: 2,
                    available: id_bytes.len(),
                });
            }
            let len = ((((first & 0x0F) as usize) << 8) | id_bytes[1] as usize) + 1;
            (len, 2)
        } else {
            let len = (first & 0x0F) as usize + 1;
            (len, 1)
        };
        let end = data_start + len;
        if end > id_bytes.len() {
            return Err(FormatError::UnexpectedEof {
                expected: end,
                available: id_bytes.len(),
            });
        }
        Ok(id_bytes[data_start..end].to_vec())
    }

    /// Returns the location of the huge object the heap ID `id_bytes` refers to: its address and
    /// length, or its key in the huge-object B-tree of the heap.
    ///
    /// The heap ID holds the address and the length where it has room for both, and the key
    /// otherwise. A caller looks the key up in the tree at
    /// [`btree_huge_objects_address`](Self::btree_huge_objects_address). `offset_size` and
    /// `length_size` are the superblock's "Size of Offsets" and "Size of Lengths" bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedFilteredHeapObject`] if the heap filters its objects,
    /// [`FormatError::UnexpectedEof`] if `id_bytes` ends before the fields it holds,
    /// [`FormatError::InvalidOffsetSize`] or [`FormatError::InvalidLengthSize`] if a width is not
    /// 2, 4, or 8, and [`FormatError::HugeObjectNotFound`] if the heap ID holds a key and the heap
    /// has no huge-object B-tree.
    pub fn decode_huge_id(
        &self,
        id_bytes: &[u8],
        offset_size: u8,
        length_size: u8,
    ) -> Result<HugeObjectReference, FormatError> {
        if self.io_filter_encoded_length > 0 {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }
        let Some((_, payload)) = id_bytes.split_first() else {
            return Err(FormatError::UnexpectedEof {
                expected: 1,
                available: 0,
            });
        };

        if self.huge_ids_direct(offset_size, length_size) {
            let addr = StoredAddress::new(bytes::read_offset(payload, 0, offset_size)?);
            let len = bytes::read_length(payload, offset_size as usize, length_size)?;
            return Ok(HugeObjectReference::Inline { addr, len });
        }

        let huge_id = read_var_le(payload);
        if convert::is_undefined_addr(self.btree_huge_objects_address.get(), offset_size) {
            return Err(FormatError::HugeObjectNotFound(huge_id));
        }
        Ok(HugeObjectReference::Indexed(huge_id))
    }

    /// Returns the heap ID of the huge object whose key in the huge-object B-tree is `huge_id`.
    ///
    /// The huge IDs of a heap store a key where they are too short for an address of
    /// `offset_width` and a length of `length_width`, as [`decode_huge_id`](Self::decode_huge_id)
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
    fn huge_ids_direct(&self, offset_size: u8, length_size: u8) -> bool {
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
        // the wrong offset, so the function rejects the heap, as `decode_huge_id` does.
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

/// The fields of a fractal heap header from `pos` on, read in the order the header stores them.
struct HeaderFields<'a> {
    data: &'a [u8],
    pos: usize,
    offset_size: u8,
    length_size: u8,
}

impl HeaderFields<'_> {
    fn u8(&mut self) -> Result<u8, FormatError> {
        let value = *self.data.get(self.pos).ok_or(FormatError::UnexpectedEof {
            expected: self.pos.saturating_add(1),
            available: self.data.len(),
        })?;
        self.pos += 1;
        Ok(value)
    }

    fn u16(&mut self) -> Result<u16, FormatError> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32, FormatError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn length(&mut self) -> Result<u64, FormatError> {
        let value = bytes::read_length(self.data, self.pos, self.length_size)?;
        self.pos += usize::from(self.length_size);
        Ok(value)
    }

    fn address(&mut self) -> Result<StoredAddress, FormatError> {
        let value = bytes::read_offset(self.data, self.pos, self.offset_size)?;
        self.pos += usize::from(self.offset_size);
        Ok(StoredAddress::new(value))
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], FormatError> {
        let value = self
            .data
            .get(self.pos..)
            .and_then(|rest| rest.first_chunk::<N>())
            .copied()
            .ok_or(FormatError::UnexpectedEof {
                expected: self.pos.saturating_add(N),
                available: self.data.len(),
            })?;
        self.pos += N;
        Ok(value)
    }
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

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::fractal_heap;
    use test_util::widths::Widths;

    use super::*;

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

        let (off, len) = hdr.decode_managed_id(&id).unwrap();
        assert_eq!(off, 0);
        assert_eq!(len, 13);
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

    #[test]
    fn a_huge_id_is_an_invalid_type_for_the_managed_decoder() {
        let file_data = fractal_heap::heap_with_one_object(b"Hello, World!", Widths::EIGHT);
        let hdr = FractalHeapHeader::parse(&file_data, 0, 8, 8).unwrap();
        // 0x10 is type 1, huge, in bits 4-5.
        let id = vec![0x10u8, 0, 0, 0, 0, 0, 0];
        let err = hdr.decode_managed_id(&id).unwrap_err();
        assert_eq!(err, FormatError::InvalidHeapIdType(1));
    }

    #[test]
    fn heap_id_type_reads_bits_4_5() {
        assert_eq!(
            FractalHeapIdType::from_heap_id(&[0x00]).unwrap(),
            FractalHeapIdType::Managed
        );
        assert_eq!(
            FractalHeapIdType::from_heap_id(&[0x10]).unwrap(),
            FractalHeapIdType::Huge
        );
        assert_eq!(
            FractalHeapIdType::from_heap_id(&[0x20]).unwrap(),
            FractalHeapIdType::Tiny
        );
        // Reserved type 3.
        assert_eq!(
            FractalHeapIdType::from_heap_id(&[0x30]),
            Err(FormatError::InvalidHeapIdType(3))
        );
        // With the version bits (0xC0) set, the type is huge.
        assert_eq!(
            FractalHeapIdType::from_heap_id(&[0xC0 | 0x10]).unwrap(),
            FractalHeapIdType::Huge
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

    #[test]
    fn read_tiny_object_short_and_extended() {
        let header = |heap_id_length| FractalHeapHeader {
            heap_id_length,
            ..dtable_header(512, 65536, 4)
        };
        // The normal form: the low four bits of byte 0 hold the length less one.
        let id = [0x20 | 0x03, b'a', b'b', b'c', b'd', 0, 0];
        assert_eq!(header(7).decode_tiny_id(&id).unwrap(), b"abcd");
        // The extended form: a 12-bit length less one across bytes 0 and 1, 0x004 for 5 bytes.
        let mut id = vec![0x20, 0x04];
        id.extend_from_slice(b"hello");
        id.resize(20, 0);
        assert_eq!(header(20).decode_tiny_id(&id).unwrap(), b"hello");

        // `H5HF__tiny_init` in `H5HFtiny.c` (HDF5 2.2.0) gives a 17-byte heap ID the normal form.
        let mut id = vec![0x20 | 0x04]; // normal form, length 5
        id.extend_from_slice(b"world");
        id.resize(17, 0);
        assert_eq!(header(17).decode_tiny_id(&id).unwrap(), b"world");

        assert_eq!(
            header(7).decode_tiny_id(&[]),
            Err(FormatError::UnexpectedEof {
                expected: 1,
                available: 0,
            })
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
            indexed.decode_huge_id(&[0x10, 5, 0, 0, 0, 0, 0], 8, 8),
            Ok(HugeObjectReference::Indexed(5))
        );
        assert_eq!(
            wide.decode_huge_id(&inline_huge, 8, 8),
            Ok(HugeObjectReference::Inline {
                addr: StoredAddress::new(0x900),
                len: 40,
            })
        );
    }

    #[test]
    fn an_empty_huge_id_is_an_unexpected_eof() {
        let indexed = FractalHeapHeader {
            btree_huge_objects_address: StoredAddress::new(0x800),
            ..dtable_header(512, 65536, 4)
        };

        assert_eq!(
            indexed.decode_huge_id(&[], 8, 8),
            Err(FormatError::UnexpectedEof {
                expected: 1,
                available: 0,
            })
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
            FractalHeapIdType::from_heap_id(&id),
            Ok(FractalHeapIdType::Managed)
        );
        assert_eq!(header.decode_managed_id(&id), Ok((heap_offset, length)));
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
            header.decode_huge_id(&id, 8, 8),
            Ok(HugeObjectReference::Indexed(5))
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
}
