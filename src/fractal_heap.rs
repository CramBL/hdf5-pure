//! The fractal heap object reader: [`HeapObjectReader`] reads managed, huge, and tiny objects from
//! a file image or from a [`Source`].

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use hdf5_pure_format::__private::BTREE_V2_HUGE_OBJECT;
use hdf5_pure_format::__private::FractalHeapChild;
pub use hdf5_pure_format::__private::FractalHeapHeader;
use hdf5_pure_format::__private::FractalHeapIdType;
use hdf5_pure_format::__private::HugeObjectRecord;
use hdf5_pure_format::__private::HugeObjectReference;

use crate::address::StoredAddress;
use crate::btree_v2::{
    BTreeV2Header, BTreeV2Record, collect_btree_v2_records, collect_btree_v2_records_from_source,
};
use crate::bytes;
use crate::convert::{Narrow, is_undefined_addr};
use crate::error::FormatError;
use crate::source::Source;
use crate::source::SourceMetadata;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

/// Copy `len` bytes at `addr` out of an in-memory file image, bounds-checked.
fn slice_object(file_data: &[u8], addr: u64, len: usize) -> Result<Vec<u8>, FormatError> {
    let start = addr.to_usize()?;
    let end = start.checked_add(len).ok_or(FormatError::OffsetOverflow {
        offset: addr,
        length: len as u64,
    })?;
    if end > file_data.len() {
        return Err(FormatError::UnexpectedEof {
            expected: end,
            available: file_data.len(),
        });
    }
    Ok(file_data[start..end].to_vec())
}

/// Read `len` bytes at `addr` from a streaming source.
fn read_object_at_source<S: Source + ?Sized>(
    source: &S,
    addr: u64,
    len: usize,
) -> Result<Vec<u8>, FormatError> {
    source.read_metadata_at(addr, len)
}

/// Confirm a heap's huge-objects B-tree is the one [`HugeObjectIndex::decode`]
/// knows how to read, before its records are read as that layout.
///
/// The other huge-object record types are unreachable by construction — the
/// directly accessed ones (3 and 4) resolve out of the heap ID without
/// consulting any tree, and the filtered ones (2 and 4) belong to heaps refused
/// at [`FractalHeapHeader::decode_huge_id`], but that is a fact about the heap
/// header, and this is the tree's own declaration. A file whose two disagree
/// would otherwise have its records decoded as a layout they are not, reading
/// an id out of another field's bytes.
fn check_huge_object_btree(
    header: &BTreeV2Header,
    offset_size: u8,
    length_size: u8,
) -> Result<(), FormatError> {
    let required = HugeObjectRecord::size(
        OffsetWidth::try_from(offset_size)?,
        LengthWidth::try_from(length_size)?,
    );
    if header.tree_type != BTREE_V2_HUGE_OBJECT || header.record_size < required {
        return Err(FormatError::UnexpectedHugeObjectBTree {
            tree_type: header.tree_type,
            record_size: usize::from(header.record_size),
            required: usize::from(required),
        });
    }
    Ok(())
}

/// Every huge object a heap holds, decoded from its huge-objects v2 B-tree.
///
/// Resolving even one huge object costs the whole index: that B-tree is read by
/// collecting its records rather than by descending to a key. So the records are
/// decoded here once, and kept by the [`HeapObjectReader`] that asked for them.
///
/// Only the type-1 record layout appears here, which
/// [`check_huge_object_btree`] establishes before any record reaches this.
struct HugeObjectIndex {
    /// `(id, address, length)`, ordered by id so a lookup is a binary search.
    /// The B-tree already stores them in that order; sorting again costs one
    /// pass and means correctness here does not rest on that.
    entries: Vec<(u64, StoredAddress, u64)>,
}

#[cfg(test)]
std::thread_local! {
    /// How many huge-object indexes have been decoded on this thread.
    ///
    /// Parsing one index per object rather than one per walk returns exactly the
    /// same objects while making the walk quadratic in their number, so the
    /// count of decodes is the only thing a test can hold the invariant to. The
    /// walks build their reader internally, so the count has to be reachable
    /// from outside the reader that did the decoding; it is per-thread because
    /// the test harness runs tests concurrently, and each test's walk runs on
    /// its own thread.
    static HUGE_INDEX_DECODES: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

/// Start counting huge-object index decodes on this thread from zero.
#[cfg(test)]
pub(crate) fn reset_huge_index_decodes() {
    HUGE_INDEX_DECODES.with(|count| count.set(0));
}

/// How many huge-object indexes this thread has decoded since the last reset.
#[cfg(test)]
pub(crate) fn huge_index_decodes() -> usize {
    HUGE_INDEX_DECODES.with(|count| count.get())
}

impl HugeObjectIndex {
    fn decode(
        records: &[BTreeV2Record],
        offset_size: u8,
        length_size: u8,
    ) -> Result<Self, FormatError> {
        #[cfg(test)]
        HUGE_INDEX_DECODES.with(|count| count.set(count.get() + 1));

        let offset_width = OffsetWidth::try_from(offset_size)?;
        let length_width = LengthWidth::try_from(length_size)?;
        let mut entries = Vec::with_capacity(records.len());
        for record in records {
            let HugeObjectRecord {
                address,
                length,
                id,
            } = record.huge_object(offset_width, length_width)?;
            entries.push((id, address, length));
        }
        entries.sort_unstable();
        Ok(Self { entries })
    }

    /// The `(address, length)` of the object with `huge_id`.
    fn locate(&self, huge_id: u64) -> Result<(StoredAddress, u64), FormatError> {
        match self
            .entries
            .binary_search_by_key(&huge_id, |&(id, _, _)| id)
        {
            Ok(i) => {
                let (_, addr, len) = self.entries[i];
                Ok((addr, len))
            }
            Err(_) => Err(FormatError::HugeObjectNotFound(huge_id)),
        }
    }
}

/// Reads objects out of one fractal heap, holding the indexes that resolving
/// them needs.
///
/// A managed or tiny object is resolved against the heap header alone, but a
/// huge one needs the heap's huge-objects B-tree, which costs the same to
/// consult for one object as for all of them. Every caller here walks a whole
/// heap rather than reading a single object, so parsing that index per object
/// made the walk quadratic in the number of huge objects. This type is where the
/// index lives instead: parsed on the first huge object, reused for the rest,
/// and never parsed at all by a heap that has none.
pub struct HeapObjectReader<'h> {
    header: &'h FractalHeapHeader,
    offset_size: u8,
    length_size: u8,
    /// Parsed on demand, since most heaps hold no huge object at all.
    huge: Option<HugeObjectIndex>,
    /// Which backend the cached index was decoded from, once one has been.
    huge_backend: Option<Backend>,
}

/// Which of the two ways of reading a file an object came through. Recorded
/// only to hold a reader to one of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    /// An in-memory image of the whole file.
    Buffered,
    /// A [`Source`], read from on demand.
    Streaming,
}

impl<'h> HeapObjectReader<'h> {
    /// Returns a reader of the objects of the heap `header` describes.
    ///
    /// `offset_size` and `length_size` are the superblock's "Size of Offsets" and "Size of
    /// Lengths" bytes.
    pub fn new(header: &'h FractalHeapHeader, offset_size: u8, length_size: u8) -> Self {
        HeapObjectReader {
            header,
            offset_size,
            length_size,
            huge: None,
            huge_backend: None,
        }
    }

    /// Read the object `id_bytes` names from an in-memory file image,
    /// dispatching on the heap-ID type. Managed objects live in the doubling
    /// table's blocks; huge objects are stored directly in the file (resolved
    /// through the huge-objects v2 B-tree); tiny objects are encoded in the ID.
    pub fn read(&mut self, file_data: &[u8], id_bytes: &[u8]) -> Result<Vec<u8>, FormatError> {
        match FractalHeapIdType::from_heap_id(id_bytes)? {
            FractalHeapIdType::Managed => self.read_managed_object(file_data, id_bytes),
            FractalHeapIdType::Huge => self.read_huge(file_data, id_bytes),
            FractalHeapIdType::Tiny => self.header.decode_tiny_id(id_bytes),
        }
    }

    /// Streaming counterpart to [`HeapObjectReader::read`].
    pub fn read_from_source<S: Source + ?Sized>(
        &mut self,
        source: &S,
        id_bytes: &[u8],
    ) -> Result<Vec<u8>, FormatError> {
        match FractalHeapIdType::from_heap_id(id_bytes)? {
            FractalHeapIdType::Managed => self.read_managed_object_from_source(source, id_bytes),
            FractalHeapIdType::Huge => self.read_huge_from_source(source, id_bytes),
            FractalHeapIdType::Tiny => self.header.decode_tiny_id(id_bytes),
        }
    }

    /// The `(address, length)` of huge object `huge_id`, parsing the heap's
    /// huge-object index out of `records` the first time one is asked for.
    ///
    /// What the index caches are file addresses, resolved against the image
    /// `backend` names. Every caller reads a whole heap through one backend, so
    /// a reader that saw both would be a caller that had changed which file it
    /// meant mid-walk — addresses applied to the wrong image read the wrong
    /// bytes and report no error, which is worth an assertion where the two meet.
    fn locate_huge<F>(
        &mut self,
        huge_id: u64,
        backend: Backend,
        records: F,
    ) -> Result<(StoredAddress, u64), FormatError>
    where
        F: FnOnce() -> Result<Vec<BTreeV2Record>, FormatError>,
    {
        if self.huge.is_none() {
            self.huge = Some(HugeObjectIndex::decode(
                &records()?,
                self.offset_size,
                self.length_size,
            )?);
            self.huge_backend = Some(backend);
        }
        debug_assert_eq!(
            self.huge_backend,
            Some(backend),
            "a heap's huge-object index holds addresses into the image it was decoded from",
        );
        self.huge
            .as_ref()
            .expect("filled immediately above")
            .locate(huge_id)
    }

    /// Resolve and read a "huge" object given its heap ID.
    fn read_huge(&mut self, file_data: &[u8], id_bytes: &[u8]) -> Result<Vec<u8>, FormatError> {
        let (addr, len) =
            match self
                .header
                .decode_huge_id(id_bytes, self.offset_size, self.length_size)?
            {
                HugeObjectReference::Inline { addr, len } => (addr, len),
                HugeObjectReference::Indexed(huge_id) => {
                    let (offset_size, length_size) = (self.offset_size, self.length_size);
                    let btree_addr = self.header.btree_huge_objects_address.get().to_usize()?;
                    self.locate_huge(huge_id, Backend::Buffered, || {
                        let header =
                            BTreeV2Header::parse(file_data, btree_addr, offset_size, length_size)?;
                        check_huge_object_btree(&header, offset_size, length_size)?;
                        collect_btree_v2_records(file_data, &header, offset_size, length_size)
                    })?
                }
            };
        slice_object(file_data, addr.get(), len.to_usize()?)
    }

    /// Resolve and read a "huge" object via a [`Source`].
    fn read_huge_from_source<S: Source + ?Sized>(
        &mut self,
        source: &S,
        id_bytes: &[u8],
    ) -> Result<Vec<u8>, FormatError> {
        let (addr, len) =
            match self
                .header
                .decode_huge_id(id_bytes, self.offset_size, self.length_size)?
            {
                HugeObjectReference::Inline { addr, len } => (addr, len),
                HugeObjectReference::Indexed(huge_id) => {
                    let (offset_size, length_size) = (self.offset_size, self.length_size);
                    let btree_addr = self.header.btree_huge_objects_address.get();
                    self.locate_huge(huge_id, Backend::Streaming, || {
                        let header = BTreeV2Header::parse_from_source(
                            &SourceMetadata(source),
                            btree_addr,
                            offset_size,
                            length_size,
                        )?;
                        check_huge_object_btree(&header, offset_size, length_size)?;
                        collect_btree_v2_records_from_source(
                            source,
                            &header,
                            offset_size,
                            length_size,
                        )
                    })?
                }
            };
        read_object_at_source(source, addr.get(), len.to_usize()?)
    }

    /// Reads the managed object the heap ID `id_bytes` refers to from `file_data`.
    fn read_managed_object(
        &self,
        file_data: &[u8],
        id_bytes: &[u8],
    ) -> Result<Vec<u8>, FormatError> {
        // A filtered heap stores its direct blocks filter-encoded, and the reader has no decoder
        // for them, so it returns an error.
        if self.header.io_filter_encoded_length > 0 {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }
        let (heap_offset, obj_len) = self.header.decode_managed_id(id_bytes)?;

        if is_undefined_addr(self.header.root_block_address.get(), self.offset_size) {
            return Err(FormatError::UnexpectedEof {
                expected: 1,
                available: 0,
            });
        }

        if self.header.current_rows_in_root_indirect_block == 0 {
            // Root is a direct block
            self.read_from_direct_block(
                file_data,
                self.header.root_block_address.get().to_usize()?,
                self.header.starting_block_size,
                0, // block offset in heap = 0 for root
                heap_offset,
                obj_len.to_usize()?,
            )
        } else {
            // The root is an indirect block, and the walk descends 64 levels at most.
            self.read_from_indirect_block(
                file_data,
                self.header.root_block_address.get().to_usize()?,
                self.header.current_rows_in_root_indirect_block,
                0, // block offset
                heap_offset,
                obj_len.to_usize()?,
                64, // max recursion depth
            )
        }
    }

    /// Reads `length` bytes at heap offset `target_offset` from the direct block at `block_addr`,
    /// whose space begins at heap offset `block_heap_offset`.
    ///
    /// The prefix of the block is inside its heap space, so the object is at
    /// `block_addr + target_offset - block_heap_offset`.
    #[allow(clippy::too_many_arguments)]
    fn read_from_direct_block(
        &self,
        file_data: &[u8],
        block_addr: usize,
        _block_size: u64,
        block_heap_offset: u64,
        target_offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, FormatError> {
        let local_offset = (target_offset - block_heap_offset).to_usize()?;
        let pos = block_addr
            .checked_add(local_offset)
            .ok_or(FormatError::OffsetOverflow {
                offset: block_addr as u64,
                length: target_offset - block_heap_offset,
            })?;
        bytes::ensure_len(file_data, pos, length)?;
        Ok(file_data[pos..pos + length].to_vec())
    }

    /// Reads `length` bytes at heap offset `target_offset` from a direct block below the indirect
    /// block at `iblock_addr`, descending `depth_remaining` levels at most.
    #[allow(clippy::too_many_arguments)]
    fn read_from_indirect_block(
        &self,
        file_data: &[u8],
        iblock_addr: usize,
        nrows: u16,
        iblock_heap_offset: u64,
        target_offset: u64,
        length: usize,
        depth_remaining: u16,
    ) -> Result<Vec<u8>, FormatError> {
        if depth_remaining == 0 {
            return Err(FormatError::ChunkedReadError(
                "fractal heap: maximum recursion depth exceeded".into(),
            ));
        }
        bytes::ensure_len(file_data, iblock_addr, 4)?;
        let block = &file_data[iblock_addr..];
        match self.header.find_child_for_offset(
            block,
            nrows,
            iblock_heap_offset,
            target_offset,
            self.offset_size,
        )? {
            Some(FractalHeapChild::Direct {
                addr,
                block_size,
                heap_offset,
            }) => self.read_from_direct_block(
                file_data,
                addr.get().to_usize()?,
                block_size,
                heap_offset,
                target_offset,
                length,
            ),
            Some(FractalHeapChild::Indirect {
                addr,
                nrows: child_nrows,
                heap_offset,
            }) => self.read_from_indirect_block(
                file_data,
                addr.get().to_usize()?,
                child_nrows,
                heap_offset,
                target_offset,
                length,
                depth_remaining - 1,
            ),
            None => Err(FormatError::UnexpectedEof {
                expected: target_offset.to_usize()?.saturating_add(length),
                available: file_data.len(),
            }),
        }
    }

    /// Reads the managed object the heap ID `id_bytes` refers to from `source`.
    fn read_managed_object_from_source<S: Source + ?Sized>(
        &self,
        source: &S,
        id_bytes: &[u8],
    ) -> Result<Vec<u8>, FormatError> {
        if self.header.io_filter_encoded_length > 0 {
            return Err(FormatError::UnsupportedFilteredHeapObject);
        }
        let (heap_offset, obj_len) = self.header.decode_managed_id(id_bytes)?;
        if is_undefined_addr(self.header.root_block_address.get(), self.offset_size) {
            return Err(FormatError::UnexpectedEof {
                expected: 1,
                available: 0,
            });
        }
        if self.header.current_rows_in_root_indirect_block == 0 {
            self.read_from_direct_block_from_source(
                source,
                self.header.root_block_address.get(),
                0, // root direct block starts at heap offset 0
                heap_offset,
                obj_len.to_usize()?,
            )
        } else {
            self.read_from_indirect_block_from_source(
                source,
                self.header.root_block_address.get(),
                self.header.current_rows_in_root_indirect_block,
                0,
                heap_offset,
                obj_len.to_usize()?,
                64,
            )
        }
    }

    /// Reads `length` bytes at heap offset `target_offset` from the direct block at `block_addr`
    /// in `source`, whose space begins at heap offset `block_heap_offset`.
    fn read_from_direct_block_from_source<S: Source + ?Sized>(
        &self,
        source: &S,
        block_addr: u64,
        block_heap_offset: u64,
        target_offset: u64,
        length: usize,
    ) -> Result<Vec<u8>, FormatError> {
        let local_offset = target_offset - block_heap_offset;
        let pos = block_addr
            .checked_add(local_offset)
            .ok_or(FormatError::OffsetOverflow {
                offset: block_addr,
                length: local_offset,
            })?;
        source.read_metadata_at(pos, length)
    }

    /// Reads `length` bytes at heap offset `target_offset` from a direct block below the indirect
    /// block at `iblock_addr` in `source`, descending `depth_remaining` levels at most.
    #[allow(clippy::too_many_arguments)]
    fn read_from_indirect_block_from_source<S: Source + ?Sized>(
        &self,
        source: &S,
        iblock_addr: u64,
        nrows: u16,
        iblock_heap_offset: u64,
        target_offset: u64,
        length: usize,
        depth_remaining: u16,
    ) -> Result<Vec<u8>, FormatError> {
        if depth_remaining == 0 {
            return Err(FormatError::ChunkedReadError(
                "fractal heap: maximum recursion depth exceeded".into(),
            ));
        }
        let region_len = self
            .header
            .indirect_block_entries_len(nrows, self.offset_size)?
            .min(source.len().saturating_sub(iblock_addr))
            .to_usize()?;
        let block = source.read_metadata_at(iblock_addr, region_len)?;
        match self.header.find_child_for_offset(
            &block,
            nrows,
            iblock_heap_offset,
            target_offset,
            self.offset_size,
        )? {
            Some(FractalHeapChild::Direct {
                addr, heap_offset, ..
            }) => self.read_from_direct_block_from_source(
                source,
                addr.get(),
                heap_offset,
                target_offset,
                length,
            ),
            Some(FractalHeapChild::Indirect {
                addr,
                nrows: child_nrows,
                heap_offset,
            }) => self.read_from_indirect_block_from_source(
                source,
                addr.get(),
                child_nrows,
                heap_offset,
                target_offset,
                length,
                depth_remaining - 1,
            ),
            None => Err(FormatError::UnexpectedEof {
                expected: target_offset.to_usize()?.saturating_add(length),
                available: source.len().to_usize().unwrap_or(usize::MAX),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use test_util::fractal_heap;
    use test_util::widths::Widths;

    use super::*;

    #[test]
    fn read_managed_object_from_direct_block() {
        let file_data = fractal_heap::heap_with_one_object(b"Hello, World!", Widths::EIGHT);
        let hdr = FractalHeapHeader::parse(&file_data, 0, 8, 8).unwrap();

        // Build heap ID for the test data written in build_simple_heap.
        // The test data "Hello, World!" is at the data area of the direct block.
        // The direct block header is 5 + 8 + 2 = 15 bytes (for max_heap_size=16, ceil(16/8)=2).
        // Wait, max_heap_size=16, ceil(16/8)=2. Header = sig(4)+ver(1)+addr(8)+bo(2) = 15.
        // The data was placed at data_start = block_addr + 15.
        // Since offset is from block start, the object is at offset 15 within the block.
        let dblock_header_size = 5 + 8 + (hdr.max_heap_size as usize).div_ceil(8); // 15
        let offset: u64 = dblock_header_size as u64;
        let length: u64 = 13;
        let payload = offset | (length << hdr.max_heap_size);
        let mut id = vec![0u8; 7];
        id[0] = 0x00;
        for i in 0..6 {
            id[1 + i] = ((payload >> (i * 8)) & 0xFF) as u8;
        }

        let obj = HeapObjectReader::new(&hdr, 8, 8)
            .read(&file_data, &id)
            .unwrap();
        assert_eq!(&obj, b"Hello, World!");
    }

    #[cfg(feature = "std")]
    #[test]
    fn streaming_managed_object_matches_buffered() {
        use crate::source::{BytesSource, ReadSeekSource};
        let file_data = fractal_heap::heap_with_one_object(b"Hello, World!", Widths::EIGHT);

        // Same heap ID as `read_managed_object_from_direct_block`.
        let hdr = FractalHeapHeader::parse(&file_data, 0, 8, 8).unwrap();
        let dblock_header_size = 5 + 8 + (hdr.max_heap_size as usize).div_ceil(8);
        let payload = (dblock_header_size as u64) | (13u64 << hdr.max_heap_size);
        let mut id = vec![0u8; 7];
        for i in 0..6 {
            id[1 + i] = ((payload >> (i * 8)) & 0xFF) as u8;
        }

        let buffered = HeapObjectReader::new(&hdr, 8, 8)
            .read(&file_data, &id)
            .unwrap();

        // Header parsed from a source, then the object fetched from a source.
        let mem = BytesSource::new(&file_data);
        let hdr_mem = FractalHeapHeader::parse_from_source(&SourceMetadata(&mem), 0, 8, 8).unwrap();
        let from_mem = HeapObjectReader::new(&hdr_mem, 8, 8)
            .read_from_source(&mem, &id)
            .unwrap();

        let seek = ReadSeekSource::new(std::io::Cursor::new(file_data)).unwrap();
        let hdr_seek =
            FractalHeapHeader::parse_from_source(&SourceMetadata(&seek), 0, 8, 8).unwrap();
        let from_seek = HeapObjectReader::new(&hdr_seek, 8, 8)
            .read_from_source(&seek, &id)
            .unwrap();

        assert_eq!(buffered, from_mem);
        assert_eq!(buffered, from_seek);
        assert_eq!(&from_seek, b"Hello, World!");
    }

    #[test]
    fn filtered_managed_heap_is_refused_not_misparsed() {
        // A filtered managed heap stores its direct-block contents filter-encoded
        // and inserts per-child `filtered_size`/`filter_mask` fields in indirect
        // blocks. We do not decode either, so every managed read path must refuse
        // cleanly (UnsupportedFilteredHeapObject) rather than return raw bytes or
        // mis-stride the child-pointer walk.
        let managed_id = [0x00u8, 0, 0, 0, 0, 0, 0]; // type bits 0 -> Managed

        // Root is a direct block.
        let mut h = FractalHeapHeader {
            heap_id_length: 7,
            io_filter_encoded_length: 8,
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
            table_width: 4,
            starting_block_size: 512,
            max_direct_block_size: 65536,
            max_heap_size: 64,
            start_root_rows: 1,
            root_block_address: StoredAddress::new(0x100),
            current_rows_in_root_indirect_block: 0,
        };
        let file = vec![0u8; 0x400];
        assert_eq!(
            HeapObjectReader::new(&h, 8, 8).read(&file, &managed_id),
            Err(FormatError::UnsupportedFilteredHeapObject)
        );

        // Root is an indirect block: the refusal happens before any child walk.
        h.current_rows_in_root_indirect_block = 2;
        assert_eq!(
            HeapObjectReader::new(&h, 8, 8).read(&file, &managed_id),
            Err(FormatError::UnsupportedFilteredHeapObject)
        );
    }

    /// Type-1 record: address(8) + length(8) + id(8), little-endian.
    fn huge_record(addr: u64, len: u64, id: u64) -> BTreeV2Record {
        let mut d = Vec::new();
        d.extend_from_slice(&addr.to_le_bytes());
        d.extend_from_slice(&len.to_le_bytes());
        d.extend_from_slice(&id.to_le_bytes());
        BTreeV2Record { data: d }
    }

    #[test]
    fn huge_object_index_locates_by_id() {
        let records = vec![
            huge_record(0x1000, 5000, 1),
            huge_record(0x2000, 6000, 2),
            huge_record(0x3000, 7000, 5),
        ];
        let index = HugeObjectIndex::decode(&records, 8, 8).unwrap();
        assert_eq!(index.locate(2).unwrap(), (StoredAddress::new(0x2000), 6000));
        assert_eq!(index.locate(5).unwrap(), (StoredAddress::new(0x3000), 7000));
        assert_eq!(
            index.locate(9),
            Err(FormatError::HugeObjectNotFound(9)),
            "an id no record carries is not found rather than mismatched"
        );
    }

    /// The lookup is a binary search, so the index has to impose the ordering
    /// itself rather than inherit whatever order the records arrived in.
    #[test]
    fn huge_object_index_orders_records_it_receives_out_of_order() {
        let records = vec![
            huge_record(0x3000, 7000, 5),
            huge_record(0x1000, 5000, 1),
            huge_record(0x2000, 6000, 2),
        ];
        let index = HugeObjectIndex::decode(&records, 8, 8).unwrap();
        for (id, want) in [
            (1, (StoredAddress::new(0x1000), 5000)),
            (2, (StoredAddress::new(0x2000), 6000)),
            (5, (StoredAddress::new(0x3000), 7000)),
        ] {
            assert_eq!(index.locate(id).unwrap(), want);
        }
    }

    /// A huge-objects B-tree that is not the type-1 layout is refused before its
    /// records are read as that layout.
    ///
    /// Type 2 is the filtered counterpart, whose records are longer and put the
    /// object ID somewhere else: decoding one as type 1 reads an ID out of the
    /// filter mask and returns some other object, or none, without complaint.
    #[test]
    fn a_huge_object_btree_of_another_type_is_refused() {
        let btree = |tree_type: u8, record_size: u16| BTreeV2Header {
            tree_type,
            node_size: 512,
            record_size,
            depth: 0,
            root_node_address: StoredAddress::new(0x100),
            num_records_in_root: 1,
            total_records: 1,
        };

        assert_eq!(
            check_huge_object_btree(&btree(2, 36), 8, 8),
            Err(FormatError::UnexpectedHugeObjectBTree {
                tree_type: 2,
                record_size: 36,
                required: 24,
            }),
            "a filtered huge-object tree is long enough to pass a length check alone"
        );
        assert_eq!(
            check_huge_object_btree(&btree(1, 16), 8, 8),
            Err(FormatError::UnexpectedHugeObjectBTree {
                tree_type: 1,
                record_size: 16,
                required: 24,
            }),
            "records too short for the layout the type claims"
        );
        assert_eq!(check_huge_object_btree(&btree(1, 24), 8, 8), Ok(()));
    }

    /// A record too short for the type-1 layout is a heap this reader has
    /// misread, not one object to skip past. The backstop behind
    /// [`check_huge_object_btree`], which refuses such a tree by its declared
    /// record size before any record reaches the decode.
    #[test]
    fn huge_object_index_refuses_a_truncated_record() {
        let records = vec![BTreeV2Record {
            data: vec![0u8; 20],
        }];
        assert!(matches!(
            HugeObjectIndex::decode(&records, 8, 8),
            Err(FormatError::UnexpectedEof { .. })
        ));
    }

    /// One reader parses the heap's huge-object index once, however many huge
    /// objects are read through it, on both backends.
    ///
    /// Parsing it per object returns exactly the same bytes while making a walk
    /// over the heap quadratic in the number of huge objects, so only the parse
    /// count separates the two.
    #[test]
    fn a_reader_parses_the_huge_object_index_once() {
        use crate::source::BytesSource;
        const COUNT: u64 = 5;

        // Attributes too large for a managed heap object, so the heap holds
        // nothing but huge ones.
        let mut builder = crate::FileBuilder::new();
        for i in 0..COUNT {
            builder.set_attr(
                &format!("a{i}"),
                crate::AttrValue::StringArray(vec![format!("{i:0700}"); 100]),
            );
        }
        builder.create_dataset("x").with_f64_data(&[1.0]);
        let bytes = builder.finish().unwrap();

        let frhp = bytes
            .windows(4)
            .position(|w| w == b"FRHP")
            .expect("dense storage writes a fractal heap");
        let heap = FractalHeapHeader::parse(&bytes, frhp, 8, 8).unwrap();

        // A huge heap ID: type 1 in bits 4-5 of byte 0, then the object's id,
        // little-endian, across the rest of the ID. The writer's own encoding,
        // which lets this reach the objects without walking the name index.
        let heap_id = |id: u64| {
            let mut bytes = vec![0u8; heap.heap_id_length as usize];
            bytes[0] = 0x10;
            for (i, slot) in bytes[1..].iter_mut().take(8).enumerate() {
                *slot = ((id >> (i * 8)) & 0xFF) as u8;
            }
            bytes
        };
        let ids: Vec<Vec<u8>> = (1..=COUNT).map(heap_id).collect();

        reset_huge_index_decodes();
        let mut reader = HeapObjectReader::new(&heap, 8, 8);
        let buffered: Vec<Vec<u8>> = ids
            .iter()
            .map(|id| reader.read(&bytes, id).unwrap())
            .collect();
        assert_eq!(
            huge_index_decodes(),
            1,
            "buffered reads re-parsed the index"
        );

        reset_huge_index_decodes();
        let source = BytesSource::new(bytes.clone());
        let mut reader = HeapObjectReader::new(&heap, 8, 8);
        let streamed: Vec<Vec<u8>> = ids
            .iter()
            .map(|id| reader.read_from_source(&source, id).unwrap())
            .collect();
        assert_eq!(
            huge_index_decodes(),
            1,
            "streaming reads re-parsed the index"
        );

        assert_eq!(buffered, streamed);
        assert_eq!(buffered.len(), COUNT as usize);
        for (i, object) in buffered.iter().enumerate() {
            assert!(
                object.windows(2).any(|w| w == format!("a{i}").as_bytes()),
                "object {i} is not the attribute message it should be"
            );
        }
    }
}
