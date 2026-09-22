//! Fractal heaps, which hold a dense group's links and a dense object's
//! attributes: section `subsec_fmt4_infra_fractalheap`, version 4.0.

use crate::bytes;
use crate::checksum;
use crate::widths::Widths;

/// The bytes of a fractal heap's header, checksum included.
///
/// The counts and sizes the header keeps of its own contents are what a reader
/// reports about how the heap stored an object, so each is settable and every
/// one defaults to an empty heap.
#[derive(Clone, Copy, Debug)]
pub struct Header {
    heap_id_len: u16,
    filter_encoded_len: u16,
    flags: u8,
    max_managed_object_size: u32,
    next_huge_object_id: u64,
    huge_object_btree_address: Option<u64>,
    free_space: u64,
    free_space_manager_address: Option<u64>,
    managed_space: u64,
    allocated_managed_space: u64,
    allocation_iterator: u64,
    managed_object_count: u64,
    huge_object_bytes: u64,
    huge_object_count: u64,
    tiny_object_bytes: u64,
    tiny_object_count: u64,
    table_width: u16,
    starting_block_size: u64,
    max_direct_block_size: u64,
    max_heap_size: u16,
    starting_root_rows: u16,
    root_block_address: u64,
    root_indirect_rows: u16,
}

impl Header {
    /// A heap whose root is the single direct block at `root_block_address`.
    pub fn new(root_block_address: u64) -> Self {
        Self {
            heap_id_len: 7,
            filter_encoded_len: 0,
            flags: 0,
            max_managed_object_size: 64,
            next_huge_object_id: 0,
            huge_object_btree_address: None,
            free_space: 0,
            free_space_manager_address: None,
            managed_space: 0,
            allocated_managed_space: 0,
            allocation_iterator: 0,
            managed_object_count: 0,
            huge_object_bytes: 0,
            huge_object_count: 0,
            tiny_object_bytes: 0,
            tiny_object_count: 0,
            table_width: 4,
            starting_block_size: 128,
            max_direct_block_size: 1024,
            max_heap_size: 16,
            starting_root_rows: 2,
            root_block_address,
            root_indirect_rows: 0,
        }
    }

    pub fn managed_object_count(mut self, count: u64) -> Self {
        self.managed_object_count = count;
        self
    }

    pub fn build(&self, widths: Widths) -> Vec<u8> {
        let mut header = SIGNATURE.to_vec();
        header.push(VERSION);
        header.extend_from_slice(&self.heap_id_len.to_le_bytes());
        header.extend_from_slice(&self.filter_encoded_len.to_le_bytes());
        header.push(self.flags);
        header.extend_from_slice(&self.max_managed_object_size.to_le_bytes());
        bytes::push_uint(&mut header, self.next_huge_object_id, widths.length);
        bytes::push_address(&mut header, self.huge_object_btree_address, widths.offset);
        bytes::push_uint(&mut header, self.free_space, widths.length);
        bytes::push_address(&mut header, self.free_space_manager_address, widths.offset);
        for count in [
            self.managed_space,
            self.allocated_managed_space,
            self.allocation_iterator,
            self.managed_object_count,
            self.huge_object_bytes,
            self.huge_object_count,
            self.tiny_object_bytes,
            self.tiny_object_count,
        ] {
            bytes::push_uint(&mut header, count, widths.length);
        }
        header.extend_from_slice(&self.table_width.to_le_bytes());
        bytes::push_uint(&mut header, self.starting_block_size, widths.length);
        bytes::push_uint(&mut header, self.max_direct_block_size, widths.length);
        header.extend_from_slice(&self.max_heap_size.to_le_bytes());
        header.extend_from_slice(&self.starting_root_rows.to_le_bytes());
        bytes::push_uint(&mut header, self.root_block_address, widths.offset);
        header.extend_from_slice(&self.root_indirect_rows.to_le_bytes());
        checksum::append(&mut header);
        header
    }

    /// The bytes of a managed direct block of this heap: its prefix, then
    /// `objects` laid out back to back from the block's start.
    ///
    /// The block carries a checksum only where the heap's flags request one,
    /// and these fixtures leave that bit clear.
    pub fn direct_block(&self, heap_address: u64, objects: &[u8], widths: Widths) -> Vec<u8> {
        let mut block = DIRECT_BLOCK_SIGNATURE.to_vec();
        block.push(VERSION);
        bytes::push_uint(&mut block, heap_address, widths.offset);
        let offset_width = usize::from(self.max_heap_size).div_ceil(8);
        bytes::push_uint(&mut block, 0, offset_width);
        block.extend_from_slice(objects);
        block
    }
}

/// Whether the file contains a fractal heap at all, which is the signature of
/// dense storage.
pub fn has_fractal_heap(bytes: &[u8]) -> bool {
    bytes::find_signature(bytes, SIGNATURE).is_some()
}

/// Offsets of every fractal-heap header in `bytes`, in file order.
pub fn header_offsets(bytes: &[u8]) -> Vec<usize> {
    bytes::signature_offsets(bytes, SIGNATURE)
}

/// Offset of the first fractal-heap header in `bytes`.
///
/// Panics if there is none, since a caller that reads heap storage has already
/// decided the file should have one.
#[track_caller]
fn header_at(bytes: &[u8]) -> usize {
    *header_offsets(bytes)
        .first()
        .expect("a dense attribute or link set has a fractal heap header")
}

/// Read a `u64` field from the fractal-heap header at `frhp`, `fields` 8-byte
/// fields past the fixed prefix: signature(4) + version(1) + heap ID length(2) +
/// I/O filter length(2) + flags(1) + maximum managed object size(4).
#[track_caller]
fn u64_at(bytes: &[u8], header: usize, fields: usize) -> u64 {
    bytes::u64_at(bytes, header + 4 + 1 + 2 + 2 + 1 + 4 + fields * LENGTH)
}

/// Read a `u64` field from the first fractal-heap header in `bytes`.
#[track_caller]
fn u64_field(bytes: &[u8], fields: usize) -> u64 {
    u64_at(bytes, header_at(bytes), fields)
}

/// How many objects the heap stores as fractal-heap *huge* objects, held
/// outside the managed direct blocks and indexed by the huge-objects v2
/// B-tree.
///
/// The header's field order is next huge object ID, huge B-tree address, free
/// space, free-space manager address, managed space, allocated managed space,
/// allocation iterator, managed object count, huge objects size, then this.
#[track_caller]
pub fn huge_object_count(bytes: &[u8]) -> u64 {
    u64_field(bytes, 9)
}

/// [`huge_object_count`] for every heap in the file, in file order, for
/// asserting what a *copy's* heap chose, which the first-heap reader cannot
/// see.
#[track_caller]
pub fn huge_object_counts(bytes: &[u8]) -> Vec<u64> {
    header_offsets(bytes)
        .into_iter()
        .map(|at| u64_at(bytes, at, 9))
        .collect()
}

/// How many objects the heap stores as managed objects, inside its direct blocks.
#[track_caller]
pub fn managed_object_count(bytes: &[u8]) -> u64 {
    u64_field(bytes, 7)
}

/// The total byte size the heap declares for its huge objects.
#[track_caller]
pub fn huge_object_bytes(bytes: &[u8]) -> u64 {
    u64_field(bytes, 8)
}

/// The heap's "current # of rows in root indirect block": 0 when the root is a
/// single direct block, and otherwise how many doubling-table rows the root
/// indirect block spans.
///
/// Past the twelve 8-byte fields [`u64_at`] indexes come the doubling-table
/// fields: table width(2), starting block size(8), maximum direct block
/// size(8), maximum heap size(2), starting root rows(2), then the root block
/// address(8) and this.
#[track_caller]
pub fn root_indirect_rows(bytes: &[u8]) -> u16 {
    bytes::u16_at(
        bytes,
        header_at(bytes)
            + 4
            + 1
            + 2
            + 2
            + 1
            + 4
            + 12 * LENGTH
            + 2
            + LENGTH
            + LENGTH
            + 2
            + 2
            + LENGTH,
    )
}

/// How many fractal-heap indirect blocks the file holds. More than one means the
/// root's own row of them filled up and the table nested.
pub fn indirect_block_count(bytes: &[u8]) -> usize {
    bytes::signature_offsets(bytes, INDIRECT_BLOCK_SIGNATURE).len()
}

pub const SIGNATURE: &[u8; 4] = b"FRHP";

pub const DIRECT_BLOCK_SIGNATURE: &[u8; 4] = b"FHDB";

pub const INDIRECT_BLOCK_SIGNATURE: &[u8; 4] = b"FHIB";

const VERSION: u8 = 0;

/// Byte offsets below are for the eight-byte lengths every file these readers
/// are pointed at carries.
const LENGTH: usize = 8;
