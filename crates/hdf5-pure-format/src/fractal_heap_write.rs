//! The managed blocks of the fractal heap that holds an object's dense attributes:
//! [`AttributeHeapPlan`] lays them out for a list of object sizes and serializes them.
//!
//! The managed space of a fractal heap is a doubling table: two rows of
//! [`ATTRIBUTE_HEAP_TABLE_WIDTH`] blocks of [`ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE`], then a row at
//! each power of two up to [`ATTRIBUTE_HEAP_MAX_DIRECT_BLOCK_SIZE`]. The rows past that hold
//! indirect blocks, each a doubling table of its own, so the heap grows by levels and no block is
//! larger than 64 KiB.
//!
//! The geometry is the one the C library gives an attribute heap: `H5A__dense_create` in
//! `H5Adense.c` passes the `H5O_FHEAP_MAN_*` macros of `H5Oprivate.h` to `H5HF_create` (HDF5
//! 2.2.0). The `H5G_FHEAP_MAN_*` macros in `H5Gdense.c` are the parameters of a group's heap, and
//! two of them differ.

use alloc::vec;
use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::bytes;
use crate::width::OffsetWidth;

/// The number of blocks in a row of the doubling table, `H5O_FHEAP_MAN_WIDTH`.
pub const ATTRIBUTE_HEAP_TABLE_WIDTH: u16 = 4;

/// The size in bytes of the blocks in the first two rows of the doubling table,
/// `H5O_FHEAP_MAN_START_BLOCK_SIZE`.
pub const ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE: u64 = 1024;

/// The size in bytes of the largest direct block, `H5O_FHEAP_MAN_MAX_DIRECT_SIZE`.
///
/// A row of larger blocks holds indirect blocks.
pub const ATTRIBUTE_HEAP_MAX_DIRECT_BLOCK_SIZE: u64 = 65_536;

/// The number of bits in a heap offset, the "Maximum Heap Size" field of the header,
/// `H5O_FHEAP_MAN_MAX_INDEX`.
///
/// A managed heap ID holds the offset of its object in this many bits.
pub const ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS: u16 = 40;

/// The width in bytes of the "Block Offset" field of a direct or indirect block, the
/// [`ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS`] of a heap offset rounded up to whole bytes.
pub const ATTRIBUTE_HEAP_BLOCK_OFFSET_BYTES: usize =
    (ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS as usize).div_ceil(8);

/// The number of rows the root indirect block starts with, the "Starting # of Rows in Root
/// Indirect Block" field of the header, `H5O_FHEAP_MAN_START_ROOT_ROWS`.
///
/// [`AttributeHeapPlan`] gives the root indirect block the rows its blocks need, whatever this
/// value is.
pub const ATTRIBUTE_HEAP_START_ROOT_ROWS: u16 = 1;

const WIDTH: u64 = ATTRIBUTE_HEAP_TABLE_WIDTH as u64;

/// The base-2 logarithms of the geometry constants, each asserted against its constant.
const START_BITS: u32 = 10;
const WIDTH_BITS: u32 = 2;
const MAX_DIRECT_BITS: u32 = 16;
const _: () = assert!(1u64 << START_BITS == ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE);
const _: () = assert!(1u64 << WIDTH_BITS == WIDTH);
const _: () = assert!(1u64 << MAX_DIRECT_BITS == ATTRIBUTE_HEAP_MAX_DIRECT_BLOCK_SIZE);

/// The base-2 logarithm of the heap space a row of starting-size blocks spans,
/// `H5HF_dtable_t::first_row_bits`.
const FIRST_ROW_BITS: u32 = START_BITS + WIDTH_BITS;

/// The number of rows that hold direct blocks, the rows before the first row of indirect blocks.
///
/// The count is `(max_direct_bits - start_bits) + 2`, as `H5HF__dtable_init` and
/// [`FractalHeapHeader::find_child_for_offset`](crate::FractalHeapHeader::find_child_for_offset)
/// compute it.
const MAX_DIRECT_ROWS: usize = (MAX_DIRECT_BITS - START_BITS + 2) as usize;

/// The most rows a root indirect block has with its blocks inside the heap's space,
/// `H5HF_dtable_t::max_root_rows`.
const MAX_ROOT_ROWS: usize =
    (ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS as u32 - FIRST_ROW_BITS + 1) as usize;

/// The size in bytes of the managed space of an attribute heap, 2 to the power of
/// [`ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS`].
pub const ATTRIBUTE_HEAP_MAX_HEAP_SPACE: u64 = 1u64 << ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS;

/// Returns the size in bytes of the prefix of a direct block: the signature (4), the version (1),
/// the heap header address, the block offset, and the checksum (4).
///
/// Every block [`AttributeHeapPlan::serialize`] writes has a checksum, so a caller sets bit 1 of
/// the flags of the heap header, "direct blocks are checksummed". A caller subtracts the size from
/// the size of a buffer as often as from the size of a block, so it is a `usize`, which widens to
/// a `u64` without a check.
pub(crate) const fn attribute_heap_direct_block_header(offset_width: OffsetWidth) -> usize {
    4 + 1 + offset_width.get() as usize + ATTRIBUTE_HEAP_BLOCK_OFFSET_BYTES + 4
}

/// Returns the size in bytes of the largest object a direct block of an attribute heap holds, the
/// largest direct block less its prefix.
///
/// A heap stores a larger object as a huge object, outside its blocks.
pub const fn attribute_heap_max_managed_object(offset_width: OffsetWidth) -> usize {
    (1usize << MAX_DIRECT_BITS) - attribute_heap_direct_block_header(offset_width)
}

/// Returns the size in bytes of an indirect block of `nrows` rows in an unfiltered heap: the
/// prefix of a direct block without its checksum, a child address per slot, and the checksum.
const fn indirect_block_size(nrows: u16, offset_width: OffsetWidth) -> u64 {
    4 + 1
        + offset_width.get() as u64
        + ATTRIBUTE_HEAP_BLOCK_OFFSET_BYTES as u64
        + nrows as u64 * WIDTH * offset_width.get() as u64
        + 4
}

/// Returns the size of the blocks in row `row` of the doubling table.
///
/// Rows 0 and 1 have the starting size, and each later row twice the size of the row before.
fn row_block_size(row: usize) -> u64 {
    if row <= 1 {
        ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE
    } else {
        ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE << (row - 1)
    }
}

/// Returns the heap offset of row `row` from the start of its indirect block,
/// `H5HF_dtable_t::row_block_off`.
///
/// The offset is the space the rows before `row` span, the managed space of a heap with that many
/// rows.
fn row_offset(row: usize) -> u64 {
    if row == 0 {
        0
    } else {
        (ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE * WIDTH) << (row - 1)
    }
}

/// Returns the row and the column of heap offset `offset` in an indirect block,
/// as `H5HF__dtable_lookup` computes them.
fn lookup(offset: u64) -> (usize, u64) {
    if offset < ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE * WIDTH {
        return (0, offset / ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE);
    }
    let high_bit = 63 - offset.leading_zeros();
    let row = (high_bit - FIRST_ROW_BITS + 1) as usize;
    (row, (offset - (1u64 << high_bit)) / row_block_size(row))
}

/// Returns the number of rows of an indirect block that spans `size` bytes of heap space, as
/// `H5HF__dtable_size_to_rows` computes it.
fn size_to_rows(size: u64) -> usize {
    ((63 - size.leading_zeros()) - FIRST_ROW_BITS + 1) as usize
}

/// Returns the heap offset and the size of the direct block that holds heap offset `offset`.
///
/// The function descends through the indirect blocks above the direct block. A caller walks the
/// direct blocks in heap order by passing the end of each block as the next `offset`.
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

/// Returns the object bytes the blocks of the first `nrows` rows hold together, allocated or not:
/// the table width times the sum of `H5HF_dtable_t::row_tot_dblock_free` over the rows.
///
/// The "Amount of Free Space in Managed Blocks" field of the header is this capacity less the
/// object bytes.
fn rows_capacity(nrows: usize, offset_width: OffsetWidth) -> u64 {
    // An indirect block spans the first rows of the table, all of them before its own row, so one
    // pass in row order computes each capacity before a later row reads it.
    let mut per_block = [0u64; MAX_ROOT_ROWS];
    let mut total = 0;
    for row in 0..nrows {
        per_block[row] = if row < MAX_DIRECT_ROWS {
            row_block_size(row) - attribute_heap_direct_block_header(offset_width) as u64
        } else {
            let child_rows = size_to_rows(row_block_size(row));
            WIDTH * per_block[..child_rows].iter().sum::<u64>()
        };
        total += WIDTH * per_block[row];
    }
    total
}

/// Returns the fewest root rows whose blocks cover `span` bytes of heap space, or `None` if
/// `span` exceeds [`ATTRIBUTE_HEAP_MAX_HEAP_SPACE`].
///
/// The first [`MAX_ROOT_ROWS`] rows span [`ATTRIBUTE_HEAP_MAX_HEAP_SPACE`] together. The placement
/// walk reaches that bound only with about a terabyte of objects, and a test calls this function
/// at the bound.
fn root_rows_covering(span: u64) -> Option<usize> {
    (1..=MAX_ROOT_ROWS).find(|&n| row_offset(n) >= span)
}

/// The error [`AttributeHeapPlan::new`] returns for objects it cannot lay out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeHeapPlanError {
    /// The blocks exceed [`ATTRIBUTE_HEAP_MAX_HEAP_SPACE`], so a heap offset exceeds its
    /// [`ATTRIBUTE_HEAP_MAX_HEAP_SIZE_BITS`]-bit field.
    HeapSpace,
    /// The blocks fit the heap, and their size exceeds `usize::MAX`, which happens on a 32-bit
    /// target alone.
    Host {
        /// The size in bytes of the blocks.
        bytes: u64,
    },
}

/// A child slot of an indirect block, once a block has been planned for it.
#[derive(Clone, Copy)]
enum Child {
    /// Index into [`AttributeHeapPlan::directs`].
    Direct(usize),
    /// Index into [`AttributeHeapPlan::indirects`].
    Indirect(usize),
}

/// A direct block of the plan, and the objects it holds.
struct PlannedDirect {
    /// The heap offset the block begins at.
    heap_offset: u64,
    /// The size of the block in bytes, its prefix and its unused bytes included.
    size: u64,
    /// The offset of the block in the region the plan lays out.
    region_offset: u64,
    /// The indices of the block's objects in the sizes passed to [`AttributeHeapPlan::new`], in
    /// the order the block holds them.
    objects: Vec<usize>,
}

/// An indirect block of the plan.
struct PlannedIndirect {
    heap_offset: u64,
    nrows: u16,
    region_offset: u64,
    /// One entry per slot, by row and then by column, and `None` for a slot with no block.
    entries: Vec<Option<Child>>,
}

/// The layout of the managed blocks of an attribute heap: the heap offset of each object, the
/// blocks that hold them, and the statistics the heap header stores.
///
/// [`new`](Self::new) lays the blocks out, [`region_size`](Self::region_size) returns the space
/// they take, which a caller allocates, and [`serialize`](Self::serialize) writes them at the
/// address the caller chose. The heap is defined in "Fractal Heap" of the [format specification,
/// version 4.0][spec].
///
/// # Examples
///
/// ```
/// use hdf5_pure_format::ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE;
/// use hdf5_pure_format::AttributeHeapPlan;
/// use hdf5_pure_format::OffsetWidth;
/// use hdf5_pure_format::StoredAddress;
///
/// let objects: [&[u8]; 2] = [b"first", b"second"];
/// let sizes: Vec<u64> = objects.iter().map(|object| object.len() as u64).collect();
/// let plan = AttributeHeapPlan::new(&sizes, OffsetWidth::Eight).unwrap();
/// assert_eq!(plan.region_size(), ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE);
///
/// // The caller allocates `region_size` bytes, and writes the heap header at another address.
/// let region_address = StoredAddress::new(0x1000);
/// let region = plan.serialize(&objects, region_address, StoredAddress::new(0x800));
///
/// // The root is a direct block at the start of the region, so a heap offset is an offset in it.
/// assert_eq!(plan.root_address(region_address), region_address);
/// let at = plan.heap_offset(1) as usize;
/// assert_eq!(&region[at..at + 6], b"second");
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
pub struct AttributeHeapPlan {
    offset_width: OffsetWidth,
    /// The heap offset of each object, in the order of the sizes passed to
    /// [`AttributeHeapPlan::new`].
    offsets: Vec<u64>,
    sizes: Vec<u64>,
    directs: Vec<PlannedDirect>,
    /// Empty where the root is a direct block, and with the root last otherwise, since each
    /// parent follows its children.
    indirects: Vec<PlannedIndirect>,
    region_size: u64,
    managed_space: u64,
    allocated_space: u64,
    allocation_iterator: u64,
    free_space: u64,
}

impl AttributeHeapPlan {
    /// Lays out blocks that hold objects of `sizes`, in their order.
    ///
    /// Each block holds objects until the next does not fit, and the next object goes to the
    /// following block in heap order, the order of the allocation iterator of the C library. The
    /// walk leaves a block too small for the object unallocated, so each object up to
    /// [`attribute_heap_max_managed_object`] finds a block. A caller stores a larger object as a
    /// huge object.
    ///
    /// # Errors
    ///
    /// Returns [`AttributeHeapPlanError::HeapSpace`] if the blocks exceed the managed space of the
    /// heap, and [`AttributeHeapPlanError::Host`] if their size exceeds `usize::MAX`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if a size exceeds the size [`attribute_heap_max_managed_object`]
    /// returns for `offset_width`.
    pub fn new(
        sizes: &[u64],
        offset_width: OffsetWidth,
    ) -> Result<AttributeHeapPlan, AttributeHeapPlanError> {
        // An object larger than the largest direct block fits no slot, and the walk below would
        // visit every block of the managed space, about 10^8 of them, before it returns
        // `HeapSpace`.
        debug_assert!(
            sizes
                .iter()
                .all(|&size| size <= attribute_heap_max_managed_object(offset_width) as u64),
            "an object too large for a managed block belongs in huge storage"
        );
        let header = attribute_heap_direct_block_header(offset_width) as u64;
        let mut directs: Vec<PlannedDirect> = Vec::new();
        let mut offsets: Vec<u64> = Vec::with_capacity(sizes.len());
        // The heap offset of the next slot the walk visits, and the bytes of the current block the
        // walk has filled.
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
                if cursor >= ATTRIBUTE_HEAP_MAX_HEAP_SPACE {
                    return Err(AttributeHeapPlanError::HeapSpace);
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

        // A heap with no managed object has a root direct block, and the header holds its address.
        if directs.is_empty() {
            directs.push(PlannedDirect {
                heap_offset: 0,
                size: ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE,
                region_offset: 0,
                objects: Vec::new(),
            });
            cursor = ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE;
        }

        // A single starting-size block at heap offset 0 is the root, as the C library leaves it
        // until the heap needs a second block.
        let root_is_direct = directs.len() == 1
            && directs[0].heap_offset == 0
            && directs[0].size == ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE;

        let mut indirects = Vec::new();
        let (managed_space, capacity) = if root_is_direct {
            (
                ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE,
                ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE - header,
            )
        } else {
            // The fewest rows whose blocks cover every block placed.
            let nrows = root_rows_covering(cursor).ok_or(AttributeHeapPlanError::HeapSpace)?;
            let mut placed = 0;
            build_indirect(0, nrows, &directs, &mut placed, &mut indirects);
            debug_assert_eq!(placed, directs.len(), "every block belongs to a slot");
            (row_offset(nrows), rows_capacity(nrows, offset_width))
        };

        let mut region_size = 0;
        for block in &mut indirects {
            block.region_offset = region_size;
            region_size += indirect_block_size(block.nrows, offset_width);
        }
        for block in &mut directs {
            block.region_offset = region_size;
            region_size += block.size;
        }
        if usize::try_from(region_size).is_err() {
            return Err(AttributeHeapPlanError::Host { bytes: region_size });
        }

        let used: u64 = sizes.iter().sum();
        debug_assert!(
            used <= capacity,
            "objects cannot exceed the blocks holding them"
        );
        Ok(AttributeHeapPlan {
            offset_width,
            offsets,
            sizes: sizes.to_vec(),
            allocated_space: directs.iter().map(|b| b.size).sum(),
            directs,
            indirects,
            region_size,
            managed_space,
            // The end of the last block the walk allocated, where the iterator of the C library
            // is. A heap whose root is a direct block has the iterator at 0: `H5HF__man_dblock_new`
            // (`H5HFdblock.c`, HDF5 2.2.0) creates that block without advancing the iterator, which
            // stays at 0 until the root is an indirect block.
            allocation_iterator: if root_is_direct { 0 } else { cursor },
            free_space: capacity - used,
        })
    }

    /// Returns the size in bytes of the blocks of the plan, back to back.
    pub fn region_size(&self) -> u64 {
        self.region_size
    }

    /// Returns the address of the root block, for blocks written from `region_address` on.
    pub fn root_address(&self, region_address: StoredAddress) -> StoredAddress {
        match self.indirects.last() {
            Some(root) => region_address.offset(root.region_offset),
            None => region_address.offset(self.directs[0].region_offset),
        }
    }

    /// Returns the number of rows in the root indirect block, or 0 where the root is a direct
    /// block, the "Current # of Rows in Root Indirect Block" field of the header.
    pub fn root_rows(&self) -> u16 {
        self.indirects.last().map_or(0, |root| root.nrows)
    }

    /// Returns the heap space the rows of the doubling table span, allocated or not, the "Amount
    /// of Managed Space in Heap" field of the header.
    pub fn managed_space(&self) -> u64 {
        self.managed_space
    }

    /// Returns the heap space of the allocated direct blocks, the "Amount of Allocated Managed
    /// Space in Heap" field of the header.
    pub fn allocated_space(&self) -> u64 {
        self.allocated_space
    }

    /// Returns the heap offset of the next block to allocate, the "Offset of Direct Block
    /// Allocation Iterator in Managed Space" field of the header.
    pub fn allocation_iterator(&self) -> u64 {
        self.allocation_iterator
    }

    /// Returns the free object bytes in the rows of the doubling table, the "Amount of Free Space
    /// in Managed Blocks" field of the header.
    pub fn free_space(&self) -> u64 {
        self.free_space
    }

    /// Returns the heap offset of object `index`, which its managed heap ID holds.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not less than the number of sizes passed to [`new`](Self::new).
    pub fn heap_offset(&self, index: usize) -> u64 {
        self.offsets[index]
    }

    /// Serializes the blocks back to back, from `region_address` on, in the order
    /// [`new`](Self::new) laid them out.
    ///
    /// `objects` holds the bytes of each object, in the order of the sizes passed to `new`, and
    /// each block holds `heap_header_address`.
    ///
    /// # Panics
    ///
    /// Panics if the lengths of `objects` differ from the sizes passed to `new`, or if an address
    /// does not fit the offset width of the plan.
    pub fn serialize(
        &self,
        objects: &[&[u8]],
        region_address: StoredAddress,
        heap_header_address: StoredAddress,
    ) -> Vec<u8> {
        assert!(
            objects
                .iter()
                .map(|object| object.len() as u64)
                .eq(self.sizes.iter().copied()),
            "the objects must have the sizes the plan was made for"
        );
        let region_size = usize::try_from(self.region_size)
            .expect("`AttributeHeapPlan::new` returns `AttributeHeapPlanError::Host` for a region past `usize`");
        let mut region = vec![0u8; region_size];

        for block in &self.indirects {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(b"FHIB");
            bytes.push(0); // version
            bytes::write_offset(&mut bytes, heap_header_address.get(), self.offset_width);
            write_heap_offset(&mut bytes, block.heap_offset);
            for entry in block.entries.iter().copied() {
                match entry {
                    Some(Child::Direct(at)) => {
                        let address = region_address.offset(self.directs[at].region_offset);
                        bytes::write_offset(&mut bytes, address.get(), self.offset_width);
                    }
                    Some(Child::Indirect(at)) => {
                        let address = region_address.offset(self.indirects[at].region_offset);
                        bytes::write_offset(&mut bytes, address.get(), self.offset_width);
                    }
                    None => bytes::write_offset(
                        &mut bytes,
                        StoredAddress::undefined(self.offset_width.get()).get(),
                        self.offset_width,
                    ),
                }
            }
            // The checksum covers the bytes before it.
            let checksum = crate::checksum::jenkins_lookup3(&bytes);
            bytes.extend_from_slice(&checksum.to_le_bytes());
            debug_assert_eq!(
                bytes.len() as u64,
                indirect_block_size(block.nrows, self.offset_width)
            );
            place(&mut region, block.region_offset, &bytes);
        }

        for block in &self.directs {
            let size = usize::try_from(block.size).expect("a direct block is at most 64 KiB");
            let mut bytes = Vec::with_capacity(size);
            bytes.extend_from_slice(b"FHDB");
            bytes.push(0); // version
            bytes::write_offset(&mut bytes, heap_header_address.get(), self.offset_width);
            write_heap_offset(&mut bytes, block.heap_offset);
            let checksum_at = bytes.len();
            // The checksum of a direct block is inside the block, and holds 0 while the writer
            // computes it.
            bytes.extend_from_slice(&[0u8; 4]);
            debug_assert_eq!(
                bytes.len(),
                attribute_heap_direct_block_header(self.offset_width)
            );
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

/// Copies the bytes of a block into `region` at offset `at`.
fn place(region: &mut [u8], at: u64, bytes: &[u8]) {
    let at = usize::try_from(at).expect("a region offset is bounded by the region size");
    region[at..at + bytes.len()].copy_from_slice(bytes);
}

/// Appends heap offset `offset` to `buf` as a little-endian integer of
/// [`ATTRIBUTE_HEAP_BLOCK_OFFSET_BYTES`], as `UINT64ENCODE_VAR` with `heap_off_size` encodes it.
fn write_heap_offset(buf: &mut Vec<u8>, offset: u64) {
    debug_assert!(
        offset < ATTRIBUTE_HEAP_MAX_HEAP_SPACE,
        "heap offset overflows its field"
    );
    buf.extend_from_slice(&offset.to_le_bytes()[..ATTRIBUTE_HEAP_BLOCK_OFFSET_BYTES]);
}

/// Lays out the indirect block of `nrows` rows at heap offset `base` and every indirect block
/// below it, appends each to `indirects` after its children, and returns the index of the block
/// at `base`.
///
/// `directs` is in heap order and `placed` counts the blocks of `directs` that earlier slots
/// hold, so the function compares the heap space of each slot with the next block alone.
fn build_indirect(
    base: u64,
    nrows: usize,
    directs: &[PlannedDirect],
    placed: &mut usize,
    indirects: &mut Vec<PlannedIndirect>,
) -> usize {
    let mut entries = vec![None; nrows * ATTRIBUTE_HEAP_TABLE_WIDTH as usize];
    'rows: for row in 0..nrows {
        let block_size = row_block_size(row);
        for col in 0..ATTRIBUTE_HEAP_TABLE_WIDTH as usize {
            let Some(next) = directs.get(*placed) else {
                break 'rows;
            };
            let slot_offset = base + row_offset(row) + col as u64 * block_size;
            if next.heap_offset >= slot_offset + block_size {
                continue;
            }
            entries[row * ATTRIBUTE_HEAP_TABLE_WIDTH as usize + col] =
                Some(if row < MAX_DIRECT_ROWS {
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

    /// The offset width of the plans the tests lay out.
    const OFFSET_WIDTH: OffsetWidth = OffsetWidth::Eight;

    /// Returns the size of the largest managed object, the largest direct block less its prefix.
    fn largest_object() -> u64 {
        ATTRIBUTE_HEAP_MAX_DIRECT_BLOCK_SIZE
            - attribute_heap_direct_block_header(OFFSET_WIDTH) as u64
    }

    /// Returns lists of object sizes, each named by the layout it reaches.
    fn shapes() -> Vec<(&'static str, Vec<u64>)> {
        let starting_capacity = ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE
            - attribute_heap_direct_block_header(OFFSET_WIDTH) as u64;
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

    /// Every object is in one allocated block, after its prefix and before its end, and no two
    /// objects overlap.
    #[test]
    fn objects_sit_inside_the_blocks_planned_for_them() {
        for (name, sizes) in shapes() {
            let plan = AttributeHeapPlan::new(&sizes, OFFSET_WIDTH).expect("plannable");
            let header = attribute_heap_direct_block_header(OFFSET_WIDTH) as u64;
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

    /// Blocks are at slots of the doubling table, in heap order, and do not overlap. A reader
    /// computes the size of a block from its slot.
    #[test]
    fn blocks_are_doubling_table_slots_in_heap_order() {
        for (name, sizes) in shapes() {
            let plan = AttributeHeapPlan::new(&sizes, OFFSET_WIDTH).expect("plannable");
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
                plan.managed_space() <= ATTRIBUTE_HEAP_MAX_HEAP_SPACE,
                "{name}: the heap outgrew its own address space"
            );
        }
    }

    /// `H5HF__cache_iblock_deserialize` in `H5HFcache.c` (HDF5 2.2.0) asserts that an indirect
    /// block it loads has a child, so an assertion-enabled build aborts on an empty one.
    #[test]
    fn no_indirect_block_is_childless() {
        for (name, sizes) in shapes() {
            let plan = AttributeHeapPlan::new(&sizes, OFFSET_WIDTH).expect("plannable");
            for block in &plan.indirects {
                assert!(
                    block.entries.iter().any(Option::is_some),
                    "{name}: an indirect block at {} has no children",
                    block.heap_offset
                );
            }
        }
    }

    /// A walk that computes the heap offset of each slot from the doubling table, as a reader
    /// does, reaches every block at its planned offset.
    #[test]
    fn every_block_is_at_the_slot_of_its_heap_offset() {
        for (name, sizes) in shapes() {
            let plan = AttributeHeapPlan::new(&sizes, OFFSET_WIDTH).expect("plannable");
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

    fn walk(
        plan: &AttributeHeapPlan,
        index: usize,
        expected: u64,
        reached: &mut Vec<u64>,
        name: &str,
    ) {
        let block = &plan.indirects[index];
        assert_eq!(
            block.heap_offset, expected,
            "{name}: block at the wrong slot"
        );
        assert_eq!(
            block.entries.len(),
            block.nrows as usize * ATTRIBUTE_HEAP_TABLE_WIDTH as usize
        );
        for (slot, entry) in block.entries.iter().enumerate() {
            let row = slot / ATTRIBUTE_HEAP_TABLE_WIDTH as usize;
            let col = (slot % ATTRIBUTE_HEAP_TABLE_WIDTH as usize) as u64;
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

    /// The oracle is the sequence of slots [`locate`] returns, one slot at a time over the managed
    /// space. Neither the C library nor a reader checks the free, allocated, and managed space
    /// fields of the header.
    #[test]
    fn the_header_statistics_describe_the_blocks_that_were_planned() {
        let header = attribute_heap_direct_block_header(OFFSET_WIDTH) as u64;
        for (name, sizes) in shapes() {
            let plan = AttributeHeapPlan::new(&sizes, OFFSET_WIDTH).expect("plannable");

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

            // The allocated space counts the direct blocks alone, so it and the indirect blocks
            // add up to the region.
            let indirect_bytes: u64 = plan
                .indirects
                .iter()
                .map(|b| indirect_block_size(b.nrows, OFFSET_WIDTH))
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

    #[test]
    fn the_root_runs_out_of_rows_exactly_at_the_heaps_address_space() {
        assert_eq!(row_offset(MAX_ROOT_ROWS), ATTRIBUTE_HEAP_MAX_HEAP_SPACE);
        assert_eq!(
            root_rows_covering(ATTRIBUTE_HEAP_MAX_HEAP_SPACE),
            Some(MAX_ROOT_ROWS)
        );
        assert_eq!(
            root_rows_covering(ATTRIBUTE_HEAP_MAX_HEAP_SPACE - 1),
            Some(MAX_ROOT_ROWS)
        );
        assert_eq!(root_rows_covering(ATTRIBUTE_HEAP_MAX_HEAP_SPACE + 1), None);
        // A span one byte past a row needs the next row up.
        assert_eq!(root_rows_covering(row_offset(3)), Some(3));
        assert_eq!(root_rows_covering(row_offset(3) + 1), Some(4));
    }

    /// The C library leaves a small heap in this shape.
    #[test]
    fn the_root_is_direct_only_while_one_starting_block_holds_everything() {
        let capacity = ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE
            - attribute_heap_direct_block_header(OFFSET_WIDTH) as u64;
        for (sizes, direct) in [
            (vec![], true),
            (vec![capacity], true),
            (vec![capacity, 1], false),
            (vec![capacity - 1, 1], true),
            (vec![capacity + 1], false),
        ] {
            let plan = AttributeHeapPlan::new(&sizes, OFFSET_WIDTH).expect("plannable");
            assert_eq!(
                plan.root_rows() == 0,
                direct,
                "{sizes:?} should{} have a direct root",
                if direct { "" } else { " not" }
            );
            if direct {
                assert_eq!(plan.managed_space(), ATTRIBUTE_HEAP_STARTING_BLOCK_SIZE);
            }
        }
    }

    #[test]
    #[should_panic(expected = "the objects must have the sizes the plan was made for")]
    fn serializing_objects_of_other_sizes_panics() {
        let plan = AttributeHeapPlan::new(&[5], OFFSET_WIDTH).expect("plannable");
        plan.serialize(&[b"hello!"], StoredAddress::new(0), StoredAddress::new(0));
    }
}
