//! Planning and emitting the *managed* blocks of a dense-attribute fractal heap.
//!
//! A fractal heap's managed space is a doubling table: a row of [`TABLE_WIDTH`]
//! blocks of [`STARTING_BLOCK_SIZE`], a second row of the same size, and then a
//! row at each successive power of two up to [`MAX_DIRECT_BLOCK_SIZE`]. Rows past
//! that hold *indirect* blocks, each of which is a doubling table in turn, so the
//! heap's address space tiles exactly and grows without any block growing past
//! 64 KiB.
//!
//! This module plans that layout for a list of object sizes and then serializes
//! it. Planning and serializing are separate because the addresses of everything
//! downstream of the heap depend on how many blocks it needs, which is only known
//! once the objects have been placed.
//!
//! The geometry constants are the reference C library's own attribute-heap
//! parameters: the `H5O_FHEAP_MAN_*` macros in `H5Oprivate.h`, which
//! `H5A__dense_create` passes to `H5HF_create`. A heap emitted here therefore has
//! the shape one the C library builds. (The similarly named `H5G_FHEAP_MAN_*` in
//! `H5Gdense.c` are the *group* heap's parameters and differ in two of them; they
//! are not the ones to copy.) [`crate::fractal_heap`] reads the same geometry back
//! out of the header and derives the direct/indirect row boundary with the same
//! formula, so the two cannot disagree about where a block sits.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::address::StoredAddress;
use crate::file_writer::{write_offset, write_undef_offset};

/// Blocks per doubling-table row (`H5O_FHEAP_MAN_WIDTH`).
pub(crate) const TABLE_WIDTH: u16 = 4;

/// Size of a block in the first two doubling-table rows
/// (`H5O_FHEAP_MAN_START_BLOCK_SIZE`).
pub(crate) const STARTING_BLOCK_SIZE: u64 = 1024;

/// The largest direct block the table ever reaches
/// (`H5O_FHEAP_MAN_MAX_DIRECT_SIZE`). Rows whose blocks would be larger hold
/// indirect blocks instead, which is what makes the heap grow by levels rather
/// than by block size.
pub(crate) const MAX_DIRECT_BLOCK_SIZE: u64 = 65_536;

/// Bits of heap offset the heap declares as its "Maximum Heap Size"
/// (`H5O_FHEAP_MAN_MAX_INDEX`), and so the width of the offset packed into every
/// managed heap ID.
pub(crate) const MAX_HEAP_SIZE_BITS: u16 = 40;

/// Byte width [`MAX_HEAP_SIZE_BITS`] implies for a block offset.
pub(crate) const BLOCK_OFFSET_BYTES: usize = (MAX_HEAP_SIZE_BITS as usize).div_ceil(8);

/// Rows the root indirect block starts out with
/// (`H5O_FHEAP_MAN_START_ROOT_ROWS`). A hint for the C library's own allocator
/// rather than a bound on what this emitter writes, which sizes the root to the
/// blocks it actually placed.
pub(crate) const START_ROOT_ROWS: u16 = 1;

const WIDTH: u64 = TABLE_WIDTH as u64;

/// `log2` of the geometry constants, each asserted against the constant it
/// describes so a change to one cannot leave the other behind.
const START_BITS: u32 = 10;
const WIDTH_BITS: u32 = 2;
const MAX_DIRECT_BITS: u32 = 16;
const _: () = assert!(1u64 << START_BITS == STARTING_BLOCK_SIZE);
const _: () = assert!(1u64 << WIDTH_BITS == WIDTH);
const _: () = assert!(1u64 << MAX_DIRECT_BITS == MAX_DIRECT_BLOCK_SIZE);

/// Bits spanned by one whole row of starting-size blocks
/// (`H5HF_dtable_t::first_row_bits`).
const FIRST_ROW_BITS: u32 = START_BITS + WIDTH_BITS;

/// Rows `[0, MAX_DIRECT_ROWS)` hold direct blocks; rows at or past it hold
/// indirect blocks. `H5HF__dtable_init`'s `(max_direct_bits - start_bits) + 2`,
/// the formula [`crate::fractal_heap`] uses on the way back in.
const MAX_DIRECT_ROWS: usize = (MAX_DIRECT_BITS - START_BITS + 2) as usize;

/// Rows a root indirect block can have before its blocks run past the heap's
/// declared address space (`H5HF_dtable_t::max_root_rows`).
const MAX_ROOT_ROWS: usize = (MAX_HEAP_SIZE_BITS as u32 - FIRST_ROW_BITS + 1) as usize;

/// Heap offsets are [`MAX_HEAP_SIZE_BITS`] wide, so this is the whole address
/// space one heap can describe.
pub(crate) const MAX_HEAP_SPACE: u64 = 1u64 << MAX_HEAP_SIZE_BITS;

/// Bytes of a direct block taken by its header: signature(4) + version(1) + heap
/// header address + block offset + checksum(4). The checksum is unconditional
/// because this emitter always sets the heap header's "checksum direct blocks"
/// flag.
///
/// A `usize` rather than a heap offset: it is subtracted from in-memory buffer
/// sizes as often as from block sizes, and widening it is free where a narrowing
/// cast would not be.
pub(crate) const fn direct_block_header(offset_size: u8) -> usize {
    4 + 1 + offset_size as usize + BLOCK_OFFSET_BYTES + 4
}

/// The largest object a managed block can hold: the whole of the table's largest
/// direct block, less that block's header. Anything larger belongs in the heap's
/// *huge* storage instead.
pub(crate) const fn max_managed_object(offset_size: u8) -> usize {
    (1usize << MAX_DIRECT_BITS) - direct_block_header(offset_size)
}

/// Bytes of an indirect block with `nrows` rows: the same header, then one child
/// address per slot, then a checksum. Unfiltered heaps only, where a direct
/// child's entry is a bare address rather than an address plus a filtered size
/// and mask.
const fn indirect_block_size(nrows: u16, offset_size: u8) -> u64 {
    4 + 1
        + offset_size as u64
        + BLOCK_OFFSET_BYTES as u64
        + nrows as u64 * WIDTH * offset_size as u64
        + 4
}

/// Block size of doubling-table row `row`. Rows 0 and 1 share the starting size;
/// every row after that doubles.
fn row_block_size(row: usize) -> u64 {
    if row <= 1 {
        STARTING_BLOCK_SIZE
    } else {
        STARTING_BLOCK_SIZE << (row - 1)
    }
}

/// Heap offset, relative to its indirect block, at which row `row` begins
/// (`H5HF_dtable_t::row_block_off`). Also the space rows `[0, row)` span, which
/// is what a heap with that many rows declares as its managed size.
fn row_offset(row: usize) -> u64 {
    if row == 0 {
        0
    } else {
        (STARTING_BLOCK_SIZE * WIDTH) << (row - 1)
    }
}

/// Row and column of `offset` within an indirect block (`H5HF__dtable_lookup`).
fn lookup(offset: u64) -> (usize, u64) {
    if offset < STARTING_BLOCK_SIZE * WIDTH {
        return (0, offset / STARTING_BLOCK_SIZE);
    }
    let high_bit = 63 - offset.leading_zeros();
    let row = (high_bit - FIRST_ROW_BITS + 1) as usize;
    (row, (offset - (1u64 << high_bit)) / row_block_size(row))
}

/// Rows in an indirect block spanning `size` bytes of heap space
/// (`H5HF__dtable_size_to_rows`).
fn size_to_rows(size: u64) -> usize {
    ((63 - size.leading_zeros()) - FIRST_ROW_BITS + 1) as usize
}

/// The direct block holding heap offset `offset`: where it begins, and how big it
/// is. Descends through as many levels of indirect block as the offset's row
/// demands, which is what lets the emitter walk the direct blocks as a flat
/// sequence even though the table is a tree.
fn locate(offset: u64) -> (u64, u64) {
    let mut base = 0;
    let mut local = offset;
    loop {
        let (row, col) = lookup(local);
        let block_size = row_block_size(row);
        let within = row_offset(row) + col * block_size;
        if row < MAX_DIRECT_ROWS {
            return (base + within, block_size);
        }
        base += within;
        local -= within;
    }
}

/// Object bytes the blocks in rows `[0, nrows)` could hold between them, whether
/// or not they were allocated — HDF5's `row_tot_dblock_free` summed, which is
/// what the header's free-space field counts down from.
fn rows_capacity(nrows: usize, offset_size: u8) -> u64 {
    // An indirect row's capacity is that of the rows below it, so one pass
    // upwards fills in everything the rows above need to look back at.
    let mut per_block = [0u64; MAX_ROOT_ROWS];
    let mut total = 0;
    for row in 0..nrows {
        per_block[row] = if row < MAX_DIRECT_ROWS {
            row_block_size(row) - direct_block_header(offset_size) as u64
        } else {
            let child_rows = size_to_rows(row_block_size(row));
            WIDTH * per_block[..child_rows].iter().sum::<u64>()
        };
        total += WIDTH * per_block[row];
    }
    total
}

/// Root rows whose blocks cover `span` bytes of heap space, or `None` when the
/// table runs out of rows first.
///
/// The boundary this draws is the heap's whole address space: rows
/// `[0, MAX_ROOT_ROWS)` span exactly [`MAX_HEAP_SPACE`] between them, so any span
/// at or past that has nowhere left to go. Named separately from the placement
/// walk it serves because the walk cannot reach the boundary without something on
/// the order of a terabyte of objects, and the arithmetic deserves a test that
/// costs nothing.
fn root_rows_covering(span: u64) -> Option<usize> {
    (1..=MAX_ROOT_ROWS).find(|&n| row_offset(n) >= span)
}

/// Why a set of objects could not be laid out. Both are refusals rather than
/// mis-encodings: a heap offset too wide for its field would truncate into some
/// other object's bytes, and a region too large for `usize` cannot be built in
/// memory at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlanRefusal {
    /// The blocks would run past the heap's [`MAX_HEAP_SIZE_BITS`]-bit address
    /// space.
    HeapSpace,
    /// The blocks fit the heap, but the bytes they occupy do not fit this host's
    /// address space. Only reachable on a 32-bit target.
    Host {
        /// Bytes the blocks would occupy.
        bytes: u64,
    },
}

/// A child slot of an indirect block, once a block has been planned for it.
#[derive(Clone, Copy)]
enum Child {
    /// Index into [`ManagedPlan::directs`].
    Direct(usize),
    /// Index into [`ManagedPlan::indirects`].
    Indirect(usize),
}

/// A direct block the plan will emit, and the objects packed into it.
struct PlannedDirect {
    /// Where the block begins in the heap's address space.
    heap_offset: u64,
    /// The block's full on-disk size, header and trailing padding included.
    size: u64,
    /// Byte offset of the block within the region the plan lays out.
    region_offset: u64,
    /// Indices into the object list given to [`ManagedPlan::new`], in the order
    /// they were packed into this block.
    objects: Vec<usize>,
}

/// An indirect block the plan will emit.
struct PlannedIndirect {
    heap_offset: u64,
    nrows: u16,
    region_offset: u64,
    /// One entry per `(row, col)` slot in row-major order, `None` where no block
    /// was allocated.
    entries: Vec<Option<Child>>,
}

/// A laid-out set of managed blocks: where every object sits in the heap's
/// address space, which blocks hold them, and the statistics the heap header
/// declares alongside.
pub(crate) struct ManagedPlan {
    offset_size: u8,
    /// Heap offset of each object, in the order given to [`ManagedPlan::new`].
    offsets: Vec<u64>,
    directs: Vec<PlannedDirect>,
    /// Empty when the root is a direct block. Otherwise the root is the *last*
    /// entry, since a parent is only appended once its children are known.
    indirects: Vec<PlannedIndirect>,
    region_size: u64,
    managed_space: u64,
    allocated_space: u64,
    allocation_iterator: u64,
    free_space: u64,
}

impl ManagedPlan {
    /// Lay out blocks holding objects of `sizes`, in that order.
    ///
    /// Objects are packed into a block until the next one does not fit, and the
    /// walk then moves to the following block in heap order — the order the C
    /// library's own allocation iterator uses. A block too small for the object
    /// at hand is skipped rather than allocated, so an object larger than a
    /// starting-size block simply lands in the first row whose blocks can hold
    /// it. Every size up to `MAX_DIRECT_BLOCK_SIZE - direct_block_header`
    /// therefore has somewhere to go, and the walk always terminates.
    ///
    /// Refuses rather than lays out a set the heap or the host cannot address;
    /// see [`PlanRefusal`].
    pub(crate) fn new(sizes: &[u64], offset_size: u8) -> Result<ManagedPlan, PlanRefusal> {
        // An object past the largest direct block fits no slot at all, and the
        // walk below would look for one all the way to the top of the heap's
        // address space — a billion iterations before it gives up. Callers split
        // huge objects out before they get here, so this is a construction
        // invariant rather than an input to validate, but a walk that has no
        // cheap upper bound deserves to fail on the first line instead of
        // spinning.
        debug_assert!(
            sizes
                .iter()
                .all(|&size| size <= max_managed_object(offset_size) as u64),
            "an object too large for a managed block belongs in huge storage"
        );
        let header = direct_block_header(offset_size) as u64;
        let mut directs: Vec<PlannedDirect> = Vec::new();
        let mut offsets: Vec<u64> = Vec::with_capacity(sizes.len());
        // Heap offset of the next slot the walk has not looked at, and how much of
        // the block being filled is already spoken for.
        let mut cursor = 0;
        let mut fill = 0;

        for &size in sizes {
            loop {
                if let Some(block) = directs.last_mut() {
                    if block.size - fill >= size {
                        block.objects.push(offsets.len());
                        offsets.push(block.heap_offset + fill);
                        fill += size;
                        break;
                    }
                }
                if cursor >= MAX_HEAP_SPACE {
                    return Err(PlanRefusal::HeapSpace);
                }
                let (heap_offset, block_size) = locate(cursor);
                debug_assert_eq!(
                    heap_offset, cursor,
                    "the walk visits whole blocks, in order"
                );
                cursor = heap_offset + block_size;
                if block_size - header >= size {
                    directs.push(PlannedDirect {
                        heap_offset,
                        size: block_size,
                        region_offset: 0,
                        objects: Vec::new(),
                    });
                    fill = header;
                }
            }
        }

        // A heap with no managed objects still gets a root direct block, so the
        // header names one rather than leaving its address undefined.
        if directs.is_empty() {
            directs.push(PlannedDirect {
                heap_offset: 0,
                size: STARTING_BLOCK_SIZE,
                region_offset: 0,
                objects: Vec::new(),
            });
            cursor = STARTING_BLOCK_SIZE;
        }

        // One starting-size block at the front of the heap is the root itself,
        // exactly as the C library leaves it until a second block is needed.
        let root_is_direct = directs.len() == 1
            && directs[0].heap_offset == 0
            && directs[0].size == STARTING_BLOCK_SIZE;

        let mut indirects = Vec::new();
        let (managed_space, capacity) = if root_is_direct {
            (STARTING_BLOCK_SIZE, STARTING_BLOCK_SIZE - header)
        } else {
            // The fewest rows whose blocks cover every block placed.
            let nrows = root_rows_covering(cursor).ok_or(PlanRefusal::HeapSpace)?;
            let mut placed = 0;
            build_indirect(0, nrows, &directs, &mut placed, &mut indirects);
            debug_assert_eq!(placed, directs.len(), "every block belongs to a slot");
            (row_offset(nrows), rows_capacity(nrows, offset_size))
        };

        let mut region_size = 0;
        for block in &mut indirects {
            block.region_offset = region_size;
            region_size += indirect_block_size(block.nrows, offset_size);
        }
        for block in &mut directs {
            block.region_offset = region_size;
            region_size += block.size;
        }
        if usize::try_from(region_size).is_err() {
            return Err(PlanRefusal::Host { bytes: region_size });
        }

        let used: u64 = sizes.iter().sum();
        debug_assert!(
            used <= capacity,
            "objects cannot exceed the blocks holding them"
        );
        Ok(ManagedPlan {
            offset_size,
            offsets,
            allocated_space: directs.iter().map(|b| b.size).sum(),
            directs,
            indirects,
            region_size,
            managed_space,
            // Just past the last block the walk allocated, which is where the C
            // library's own iterator sits. A heap whose root is still a bare
            // direct block is the exception: `H5HF__man_dblock_new` creates that
            // first block without advancing the iterator, and leaves it at zero
            // until the root grows into an indirect block.
            allocation_iterator: if root_is_direct { 0 } else { cursor },
            free_space: capacity - used,
        })
    }

    /// Bytes the plan's blocks occupy, laid out back to back.
    pub(crate) fn region_size(&self) -> u64 {
        self.region_size
    }

    /// Returns the address of the block the heap header points at, given where
    /// the region begins.
    pub(crate) fn root_address(&self, region_address: StoredAddress) -> StoredAddress {
        match self.indirects.last() {
            Some(root) => region_address.offset(root.region_offset),
            None => region_address.offset(self.directs[0].region_offset),
        }
    }

    /// Rows in the root indirect block, or 0 when the root is a direct block —
    /// the heap header's "current # of rows in root indirect block", which is how
    /// a reader tells the two apart.
    pub(crate) fn root_rows(&self) -> u16 {
        self.indirects.last().map_or(0, |root| root.nrows)
    }

    /// Heap space the table's rows span, allocated or not.
    pub(crate) fn managed_space(&self) -> u64 {
        self.managed_space
    }

    /// Heap space taken by the blocks that were allocated.
    pub(crate) fn allocated_space(&self) -> u64 {
        self.allocated_space
    }

    /// Heap offset at which the next block would be allocated.
    pub(crate) fn allocation_iterator(&self) -> u64 {
        self.allocation_iterator
    }

    /// Object bytes the table's rows could still hold.
    pub(crate) fn free_space(&self) -> u64 {
        self.free_space
    }

    /// Heap offset of object `index`, which is what its managed heap ID encodes.
    pub(crate) fn heap_offset(&self, index: usize) -> u64 {
        self.offsets[index]
    }

    /// Emits every block, back to back, in the order [`ManagedPlan::new`] laid
    /// them out. `objects` supplies the bytes of each planned object, in the same
    /// order as the sizes it was planned from.
    pub(crate) fn serialize(
        &self,
        objects: &[&[u8]],
        region_address: StoredAddress,
        heap_header_address: StoredAddress,
    ) -> Vec<u8> {
        let region_size = usize::try_from(self.region_size)
            .expect("ManagedPlan::new refuses a region this host cannot address");
        let mut region = vec![0u8; region_size];

        for block in &self.indirects {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(b"FHIB");
            bytes.push(0); // version
            write_offset(&mut bytes, heap_header_address, self.offset_size);
            write_heap_offset(&mut bytes, block.heap_offset);
            for entry in block.entries.iter().copied() {
                match entry {
                    Some(Child::Direct(at)) => {
                        let address = region_address.offset(self.directs[at].region_offset);
                        write_offset(&mut bytes, address, self.offset_size);
                    }
                    Some(Child::Indirect(at)) => {
                        let address = region_address.offset(self.indirects[at].region_offset);
                        write_offset(&mut bytes, address, self.offset_size);
                    }
                    None => write_undef_offset(&mut bytes, self.offset_size),
                }
            }
            // The checksum covers everything ahead of it, unlike a direct block's,
            // which sits mid-block and is zeroed while it is computed.
            let checksum = crate::checksum::jenkins_lookup3(&bytes);
            bytes.extend_from_slice(&checksum.to_le_bytes());
            debug_assert_eq!(
                bytes.len() as u64,
                indirect_block_size(block.nrows, self.offset_size)
            );
            place(&mut region, block.region_offset, &bytes);
        }

        for block in &self.directs {
            let size = usize::try_from(block.size).expect("a direct block is at most 64 KiB");
            let mut bytes = Vec::with_capacity(size);
            bytes.extend_from_slice(b"FHDB");
            bytes.push(0); // version
            write_offset(&mut bytes, heap_header_address, self.offset_size);
            write_heap_offset(&mut bytes, block.heap_offset);
            let checksum_at = bytes.len();
            bytes.extend_from_slice(&[0u8; 4]); // checksum placeholder
            debug_assert_eq!(bytes.len(), direct_block_header(self.offset_size));
            for &object in &block.objects {
                debug_assert_eq!(
                    block.heap_offset + bytes.len() as u64,
                    self.offsets[object],
                    "an object must be emitted at the offset it was planned at"
                );
                bytes.extend_from_slice(objects[object]);
            }
            bytes.resize(size, 0);
            let checksum = crate::checksum::jenkins_lookup3(&bytes);
            bytes[checksum_at..checksum_at + 4].copy_from_slice(&checksum.to_le_bytes());
            place(&mut region, block.region_offset, &bytes);
        }

        region
    }
}

/// Copy one block's bytes into the region at the offset the plan gave it.
fn place(region: &mut [u8], at: u64, bytes: &[u8]) {
    let at = usize::try_from(at).expect("a region offset is bounded by the region size");
    region[at..at + bytes.len()].copy_from_slice(bytes);
}

/// Write a heap offset in the width [`MAX_HEAP_SIZE_BITS`] implies, little-endian
/// — HDF5's `UINT64ENCODE_VAR` against `heap_off_size`.
fn write_heap_offset(buf: &mut Vec<u8>, offset: u64) {
    debug_assert!(offset < MAX_HEAP_SPACE, "heap offset overflows its field");
    buf.extend_from_slice(&offset.to_le_bytes()[..BLOCK_OFFSET_BYTES]);
}

/// Build the indirect block covering `nrows` rows from `base`, and every indirect
/// block below it, appending each to `indirects` once its children are known.
///
/// `directs` is in heap order and `placed` counts how many of them earlier slots
/// have claimed, so a slot only has to ask whether the next unclaimed block
/// begins inside it. Returns the index of the block it appended.
fn build_indirect(
    base: u64,
    nrows: usize,
    directs: &[PlannedDirect],
    placed: &mut usize,
    indirects: &mut Vec<PlannedIndirect>,
) -> usize {
    let mut entries = vec![None; nrows * TABLE_WIDTH as usize];
    'rows: for row in 0..nrows {
        let block_size = row_block_size(row);
        for col in 0..TABLE_WIDTH as usize {
            let Some(next) = directs.get(*placed) else {
                break 'rows;
            };
            let slot_offset = base + row_offset(row) + col as u64 * block_size;
            if next.heap_offset >= slot_offset + block_size {
                continue;
            }
            entries[row * TABLE_WIDTH as usize + col] = Some(if row < MAX_DIRECT_ROWS {
                debug_assert_eq!(
                    next.heap_offset, slot_offset,
                    "a block fills its whole slot"
                );
                *placed += 1;
                Child::Direct(*placed - 1)
            } else {
                Child::Indirect(build_indirect(
                    slot_offset,
                    size_to_rows(block_size),
                    directs,
                    placed,
                    indirects,
                ))
            });
        }
    }
    indirects.push(PlannedIndirect {
        heap_offset: base,
        nrows: u16::try_from(nrows).expect("a row count is bounded by MAX_ROOT_ROWS"),
        region_offset: 0,
        entries,
    });
    indirects.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The on-disk address width this crate writes.
    const OFFSET_SIZE: u8 = 8;

    /// The largest object the table can hold in a managed block: the whole of its
    /// largest direct block, less that block's header.
    fn largest_object() -> u64 {
        MAX_DIRECT_BLOCK_SIZE - direct_block_header(OFFSET_SIZE) as u64
    }

    /// Object-size shapes worth planning, each named by what it exercises.
    fn shapes() -> Vec<(&'static str, Vec<u64>)> {
        let starting_capacity = STARTING_BLOCK_SIZE - direct_block_header(OFFSET_SIZE) as u64;
        vec![
            ("empty", Vec::new()),
            ("one tiny object", vec![1]),
            ("one starting block, exactly full", vec![starting_capacity]),
            ("one byte past a starting block", vec![starting_capacity, 1]),
            (
                "many small objects",
                (0..500).map(|i| 20 + i % 40).collect(),
            ),
            (
                "objects that skip the small rows",
                vec![largest_object(); 3],
            ),
            (
                "enough large objects to nest indirect blocks",
                vec![largest_object(); 40],
            ),
            (
                "large and small mixed",
                (0..60)
                    .map(|i| if i % 3 == 0 { largest_object() } else { 100 })
                    .collect(),
            ),
        ]
    }

    /// Every object sits inside one allocated block, past that block's header and
    /// wholly within it, and no two objects overlap.
    #[test]
    fn objects_sit_inside_the_blocks_planned_for_them() {
        for (name, sizes) in shapes() {
            let plan = ManagedPlan::new(&sizes, OFFSET_SIZE).expect("plannable");
            let header = direct_block_header(OFFSET_SIZE) as u64;
            let mut previous_end = 0;
            for (index, &size) in sizes.iter().enumerate() {
                let at = plan.heap_offset(index);
                let block = plan
                    .directs
                    .iter()
                    .find(|b| at >= b.heap_offset && at < b.heap_offset + b.size)
                    .unwrap_or_else(|| panic!("{name}: object {index} is in no allocated block"));
                assert!(
                    at >= block.heap_offset + header,
                    "{name}: object {index} overlaps its block's header"
                );
                assert!(
                    at + size <= block.heap_offset + block.size,
                    "{name}: object {index} runs past its block"
                );
                assert!(
                    at >= previous_end,
                    "{name}: object {index} overlaps its predecessor"
                );
                previous_end = at + size;
            }
        }
    }

    /// Blocks sit at doubling-table positions, in heap order, and never overlap.
    /// A block at any other offset is one the reader would compute a different
    /// size for.
    #[test]
    fn blocks_are_doubling_table_slots_in_heap_order() {
        for (name, sizes) in shapes() {
            let plan = ManagedPlan::new(&sizes, OFFSET_SIZE).expect("plannable");
            let mut previous_end = 0;
            for block in &plan.directs {
                assert_eq!(
                    locate(block.heap_offset),
                    (block.heap_offset, block.size),
                    "{name}: a block at {} is not a slot of that size",
                    block.heap_offset
                );
                assert!(block.heap_offset >= previous_end, "{name}: blocks overlap");
                previous_end = block.heap_offset + block.size;
            }
            assert!(
                plan.managed_space() >= previous_end,
                "{name}: the declared managed space does not cover the blocks"
            );
            assert!(
                plan.managed_space() <= MAX_HEAP_SPACE,
                "{name}: the heap outgrew its own address space"
            );
        }
    }

    /// The reference C library asserts that an indirect block it loads has at
    /// least one child, so an empty one aborts an assertion-enabled build rather
    /// than being rejected.
    #[test]
    fn no_indirect_block_is_childless() {
        for (name, sizes) in shapes() {
            let plan = ManagedPlan::new(&sizes, OFFSET_SIZE).expect("plannable");
            for block in &plan.indirects {
                assert!(
                    block.entries.iter().any(Option::is_some),
                    "{name}: an indirect block at {} has no children",
                    block.heap_offset
                );
            }
        }
    }

    /// Walking the tree the way a reader does — accumulating each slot's heap
    /// offset from the doubling table rather than reading it off the block — must
    /// reach every block at the offset it was planned at, and reach them all.
    #[test]
    fn the_tree_puts_every_block_at_the_slot_its_heap_offset_names() {
        for (name, sizes) in shapes() {
            let plan = ManagedPlan::new(&sizes, OFFSET_SIZE).expect("plannable");
            let Some(root) = plan.indirects.len().checked_sub(1) else {
                assert_eq!(plan.directs.len(), 1, "{name}: a direct root is one block");
                assert_eq!(plan.directs[0].heap_offset, 0);
                continue;
            };
            let mut reached = Vec::new();
            walk(&plan, root, 0, &mut reached, name);
            let planned: Vec<u64> = plan.directs.iter().map(|b| b.heap_offset).collect();
            assert_eq!(
                reached, planned,
                "{name}: the walk missed or reordered blocks"
            );
        }
    }

    fn walk(plan: &ManagedPlan, index: usize, expected: u64, reached: &mut Vec<u64>, name: &str) {
        let block = &plan.indirects[index];
        assert_eq!(
            block.heap_offset, expected,
            "{name}: block at the wrong slot"
        );
        assert_eq!(
            block.entries.len(),
            block.nrows as usize * TABLE_WIDTH as usize
        );
        for (slot, entry) in block.entries.iter().enumerate() {
            let row = slot / TABLE_WIDTH as usize;
            let col = (slot % TABLE_WIDTH as usize) as u64;
            let size = row_block_size(row);
            let at = block.heap_offset + row_offset(row) + col * size;
            match entry {
                None => {}
                Some(Child::Direct(child)) => {
                    assert!(
                        row < MAX_DIRECT_ROWS,
                        "{name}: a direct block in an indirect row"
                    );
                    let child = &plan.directs[*child];
                    assert_eq!((child.heap_offset, child.size), (at, size), "{name}");
                    reached.push(child.heap_offset);
                }
                Some(Child::Indirect(child)) => {
                    assert!(
                        row >= MAX_DIRECT_ROWS,
                        "{name}: an indirect block in a direct row"
                    );
                    walk(plan, *child, at, reached, name);
                }
            }
        }
    }

    /// Every statistic the heap header declares about its managed blocks, checked
    /// against the block sequence itself rather than against the arithmetic that
    /// produced it.
    ///
    /// The capacity here is accumulated by stepping through the managed space one
    /// slot at a time, as [`locate`] reports them, which shares nothing with
    /// [`rows_capacity`]'s closed form. That is the point: the header's
    /// free-space, allocated-space and managed-space fields are ones neither this
    /// crate's reader nor the C library validates on read, so a test that reuses
    /// the emitter's own expression for them cannot fail.
    #[test]
    fn the_header_statistics_describe_the_blocks_that_were_planned() {
        let header = direct_block_header(OFFSET_SIZE) as u64;
        for (name, sizes) in shapes() {
            let plan = ManagedPlan::new(&sizes, OFFSET_SIZE).expect("plannable");

            let mut at = 0;
            let mut capacity = 0;
            while at < plan.managed_space() {
                let (start, size) = locate(at);
                assert_eq!(start, at, "{name}: a slot does not begin where it is found");
                capacity += size - header;
                at = start + size;
            }
            assert_eq!(
                at,
                plan.managed_space(),
                "{name}: the managed space is not a whole number of blocks"
            );

            let used: u64 = sizes.iter().sum();
            assert_eq!(
                plan.free_space(),
                capacity - used,
                "{name}: free space must count every block the rows describe, \
                 allocated or not, less the object bytes"
            );

            // The allocated-space field counts direct blocks and only those, so
            // it and the indirect blocks partition the region being emitted.
            let indirect_bytes: u64 = plan
                .indirects
                .iter()
                .map(|b| indirect_block_size(b.nrows, OFFSET_SIZE))
                .sum();
            assert_eq!(
                plan.allocated_space() + indirect_bytes,
                plan.region_size(),
                "{name}: allocated space is not the direct blocks alone"
            );

            let last = plan.directs.last().expect("a heap has at least one block");
            let expected = if plan.root_rows() == 0 {
                0
            } else {
                last.heap_offset + last.size
            };
            assert_eq!(
                plan.allocation_iterator(),
                expected,
                "{name}: the iterator must sit past the last allocated block, \
                 or at zero while the root is a bare direct block"
            );
        }
    }

    /// The table's rows span the heap's address space exactly, so the last row
    /// that can hold a block is the one whose blocks end at the top of it.
    ///
    /// This is the boundary [`ManagedPlan::new`] refuses at. Reaching it through
    /// the placement walk would take on the order of a terabyte of objects, so
    /// this is the only place the arithmetic can be tested at all.
    #[test]
    fn the_root_runs_out_of_rows_exactly_at_the_heaps_address_space() {
        assert_eq!(row_offset(MAX_ROOT_ROWS), MAX_HEAP_SPACE);
        assert_eq!(root_rows_covering(MAX_HEAP_SPACE), Some(MAX_ROOT_ROWS));
        assert_eq!(root_rows_covering(MAX_HEAP_SPACE - 1), Some(MAX_ROOT_ROWS));
        assert_eq!(root_rows_covering(MAX_HEAP_SPACE + 1), None);
        // A span one byte past a row needs the next row up.
        assert_eq!(root_rows_covering(row_offset(3)), Some(3));
        assert_eq!(root_rows_covering(row_offset(3) + 1), Some(4));
    }

    /// The root stays a bare direct block exactly while one starting-size block
    /// holds everything, which is the shape the C library leaves a small heap in.
    #[test]
    fn the_root_is_direct_only_while_one_starting_block_holds_everything() {
        let capacity = STARTING_BLOCK_SIZE - direct_block_header(OFFSET_SIZE) as u64;
        for (sizes, direct) in [
            (vec![], true),
            (vec![capacity], true),
            (vec![capacity, 1], false),
            (vec![capacity - 1, 1], true),
            (vec![capacity + 1], false),
        ] {
            let plan = ManagedPlan::new(&sizes, OFFSET_SIZE).expect("plannable");
            assert_eq!(
                plan.root_rows() == 0,
                direct,
                "{sizes:?} should{} have a direct root",
                if direct { "" } else { " not" }
            );
            if direct {
                assert_eq!(plan.managed_space(), STARTING_BLOCK_SIZE);
            }
        }
    }
}
