//! Fractal heaps, which hold a dense group's links and a dense object's
//! attributes: section `subsec_fmt4_infra_fractalheap`, version 4.0.

use crate::bytes;

/// Whether the file contains a fractal heap at all — the signature of dense
/// (heap) rather than compact (in-object-header) storage.
pub fn has_fractal_heap(bytes: &[u8]) -> bool {
    bytes::find_signature(bytes, SIGNATURE).is_some()
}

/// Offsets of every fractal-heap header in `bytes`, in file order.
pub fn header_offsets(bytes: &[u8]) -> Vec<usize> {
    bytes::signature_offsets(bytes, SIGNATURE)
}

/// Offset of the first fractal-heap header in `bytes`.
///
/// Panics if there is none, since a caller asking about heap storage has already
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

/// How many objects the heap stores as fractal-heap *huge* objects — held outside
/// the managed direct blocks and indexed by the huge-objects v2 B-tree.
///
/// The header's field order is next huge object ID, huge B-tree address, free
/// space, free-space manager address, managed space, allocated managed space,
/// allocation iterator, managed object count, huge objects size, then this.
#[track_caller]
pub fn huge_object_count(bytes: &[u8]) -> u64 {
    u64_field(bytes, 9)
}

/// [`huge_object_count`] for every heap in the file, in file order — for asserting
/// what a *copy's* heap chose, which the first-heap reader cannot see.
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
/// fields — table width(2), starting block size(8), maximum direct block size(8),
/// maximum heap size(2), starting root rows(2) — then the root block address(8)
/// and this.
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

pub const INDIRECT_BLOCK_SIGNATURE: &[u8; 4] = b"FHIB";

/// Byte offsets below are for the eight-byte lengths every file these readers
/// are pointed at carries.
const LENGTH: usize = 8;
