//! The fractal heap header parser, the heap ID decoders, and the lookup of a child in an indirect
//! block.

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
        match (byte0 >> 4) & 0x03 {
            0 => Ok(FractalHeapIdType::Managed),
            1 => Ok(FractalHeapIdType::Huge),
            2 => Ok(FractalHeapIdType::Tiny),
            other => Err(FormatError::InvalidHeapIdType(other)),
        }
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
/// the fields a reader needs to find an object.
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
    /// The size in bytes of the largest managed object. The heap stores a larger object as a huge
    /// object, outside its blocks.
    pub max_managed_object_size: u32,
    /// The address of the version 2 B-tree that indexes the heap's huge objects, or the undefined
    /// address where the heap has no such tree.
    pub btree_huge_objects_address: StoredAddress,
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
    /// The number of managed objects in the heap.
    pub managed_objects_count: u64,
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
        if &file_data[offset..offset + 4] != b"FRHP" {
            return Err(FormatError::InvalidFractalHeapSignature);
        }

        let version = file_data[offset + 4];
        if version != 0 {
            return Err(FormatError::InvalidFractalHeapVersion(version));
        }

        let os = offset_size as usize;
        let ls = length_size as usize;

        let mut pos = offset + 5;
        bytes::ensure_len(file_data, pos, 2)?;
        let heap_id_length = u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);
        pos += 2;

        bytes::ensure_len(file_data, pos, 2)?;
        let io_filter_encoded_length = u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);
        pos += 2;

        bytes::ensure_len(file_data, pos, 1)?;
        let _flags = file_data[pos];
        pos += 1;

        bytes::ensure_len(file_data, pos, 4)?;
        let max_managed_object_size = u32::from_le_bytes([
            file_data[pos],
            file_data[pos + 1],
            file_data[pos + 2],
            file_data[pos + 3],
        ]);
        pos += 4;

        // `next_huge_object_id` (`length_size`), skipped
        bytes::ensure_len(file_data, pos, ls)?;
        pos += ls;

        // `btree_huge_objects_address` (`offset_size`)
        let btree_huge_objects_address =
            StoredAddress::new(bytes::read_offset(file_data, pos, offset_size)?);
        pos += os;

        // Skip the remaining fixed fields: free_space_managed_blocks(ls),
        // managed_block_free_space_manager_address(os), managed_space_in_heap(ls),
        // allocated_managed_space_in_heap(ls),
        // direct_block_allocation_iterator_offset(ls)
        let skip_size = 4 * ls + os;
        bytes::ensure_len(file_data, pos, skip_size)?;
        pos += skip_size;

        // `managed_objects_count` (`length_size`)
        let managed_objects_count = bytes::read_length(file_data, pos, length_size)?;
        pos += ls;

        // `huge_objects_size` (`length_size`)
        pos += ls;
        // `huge_objects_count` (`length_size`)
        pos += ls;
        // `tiny_objects_size` (`length_size`)
        pos += ls;
        // `tiny_objects_count` (`length_size`)
        pos += ls;

        // `table_width` (2)
        bytes::ensure_len(file_data, pos, 2)?;
        let table_width = u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);
        pos += 2;

        // `starting_block_size` (`length_size`)
        let starting_block_size = bytes::read_length(file_data, pos, length_size)?;
        pos += ls;

        // `max_direct_block_size` (`length_size`)
        let max_direct_block_size = bytes::read_length(file_data, pos, length_size)?;
        pos += ls;

        // `max_heap_size` (2)
        bytes::ensure_len(file_data, pos, 2)?;
        let max_heap_size = u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);
        pos += 2;

        // `start_root_rows`: starting # of rows in the root indirect block (2)
        bytes::ensure_len(file_data, pos, 2)?;
        let start_root_rows = u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);
        pos += 2;

        // `root_block_address` (`offset_size`)
        let root_block_address =
            StoredAddress::new(bytes::read_offset(file_data, pos, offset_size)?);
        pos += os;

        // `current_rows_in_root_indirect_block` (2)
        bytes::ensure_len(file_data, pos, 2)?;
        let current_rows_in_root_indirect_block =
            u16::from_le_bytes([file_data[pos], file_data[pos + 1]]);
        #[allow(unused_variables, unused_mut, unused_assignments)]
        let mut pos = pos + 2;

        // Skip the size of the filtered root direct block and its filter mask.
        if io_filter_encoded_length > 0 {
            #[allow(unused_assignments)]
            {
                pos += ls + 4;
            }
        }

        // Validate header checksum
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

        Ok(FractalHeapHeader {
            heap_id_length,
            io_filter_encoded_length,
            max_managed_object_size,
            btree_huge_objects_address,
            table_width,
            starting_block_size,
            max_direct_block_size,
            max_heap_size,
            start_root_rows,
            root_block_address,
            current_rows_in_root_indirect_block,
            managed_objects_count,
        })
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

#[cfg(test)]
mod tests {
    use test_util::fractal_heap;
    use test_util::widths::Widths;

    use super::*;

    #[test]
    fn a_header_parses_to_its_fields() {
        let file_data = fractal_heap::heap_with_one_object(b"Hello, World!", Widths::EIGHT);
        let hdr = FractalHeapHeader::parse(&file_data, 0, 8, 8).unwrap();
        assert_eq!(hdr.heap_id_length, 7);
        assert_eq!(hdr.io_filter_encoded_length, 0);
        assert_eq!(hdr.max_managed_object_size, 64);
        assert_eq!(hdr.table_width, 4);
        assert_eq!(hdr.starting_block_size, 128);
        assert_eq!(hdr.max_heap_size, 16);
        assert_eq!(hdr.current_rows_in_root_indirect_block, 0);
        assert_eq!(hdr.managed_objects_count, 1);
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
            max_managed_object_size: 0,
            btree_huge_objects_address: StoredAddress::new(u64::MAX),
            table_width,
            starting_block_size: start_block_size,
            max_direct_block_size,
            max_heap_size: 64,
            start_root_rows: 1,
            root_block_address: StoredAddress::new(0),
            current_rows_in_root_indirect_block: 0,
            managed_objects_count: 0,
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
}
