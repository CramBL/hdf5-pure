//! Fractal heaps, which hold a dense group's links and a dense object's
//! attributes: section `subsec_fmt4_infra_fractalheap`, version 4.0.

use crate::bytes;
use crate::checksum;
use crate::image::Image;
use crate::widths::Widths;

/// The bytes of a fractal heap's header, checksum included.
///
/// The counts and sizes the header keeps of its own contents are what a reader
/// reports about how the heap stored an object, so each is settable and every
/// one defaults to an empty heap.
#[derive(Clone, Debug)]
pub struct Header {
    heap_id_len: u16,
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
    filtered_root_size: u64,
    root_filter_mask: u32,
    filter_pipeline: Vec<u8>,
}

impl Header {
    /// A heap whose root is the single direct block at `root_block_address`.
    pub fn new(root_block_address: u64) -> Self {
        Self {
            heap_id_len: 7,
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
            filtered_root_size: 0,
            root_filter_mask: 0,
            filter_pipeline: Vec::new(),
        }
    }

    pub fn managed_object_count(mut self, count: u64) -> Self {
        self.managed_object_count = count;
        self
    }

    /// Sets the optional root filter fields and encoded pipeline bytes.
    ///
    /// `root_size` is the size in bytes of the root direct block after filtering, and `mask`
    /// identifies the skipped filters. An empty `pipeline` produces an unfiltered header.
    pub fn filtering(mut self, root_size: u64, mask: u32, pipeline: Vec<u8>) -> Self {
        self.filtered_root_size = root_size;
        self.root_filter_mask = mask;
        self.filter_pipeline = pipeline;
        self
    }

    /// Returns the encoded header, including its checksum.
    ///
    /// # Panics
    ///
    /// Panics if the encoded filter pipeline exceeds [`u16::MAX`] bytes, a field width is
    /// outside `1..=8`, or a value does not fit its field.
    pub fn build(&self, widths: Widths) -> Vec<u8> {
        let mut header = SIGNATURE.to_vec();
        header.push(VERSION);
        header.extend_from_slice(&self.heap_id_len.to_le_bytes());
        // "Fractal Heap", format specification version 4.0, defines a two-byte pipeline length.
        assert!(
            self.filter_pipeline.len() <= usize::from(u16::MAX),
            "fixture pipeline fits u16"
        );
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the assertion bounds the pipeline length by u16::MAX"
        )]
        let encoded_len = self.filter_pipeline.len() as u16;
        header.extend_from_slice(&encoded_len.to_le_bytes());
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
        if !self.filter_pipeline.is_empty() {
            bytes::push_uint(&mut header, self.filtered_root_size, widths.length);
            header.extend_from_slice(&self.root_filter_mask.to_le_bytes());
            header.extend_from_slice(&self.filter_pipeline);
        }
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

/// Returns the fractal-heap header from a dense-link deletion fixture.
///
/// The fixture uses 8-byte offsets and lengths. The crosscheck generates the metadata with
/// each supported HDF5 release and compares the complete bytes.
pub fn dense_link_free_space_header() -> Vec<u8> {
    decode_hex(
        "46524850000700000002001000000000000000000000ffffffffffffffffd2b80300000000007b050000000000000000040000000000006000000000000000a000000000000096000000000000000000000000000000000000000000000000000000000000000000000000000000040000020000000000000000010000000000200001001b14000000000000080099fa2d63",
    )
}

/// Returns the free-space manager header from a dense-link deletion fixture.
///
/// The fixture uses 8-byte offsets and lengths. The crosscheck generates the metadata with
/// each supported HDF5 release and compares the complete bytes.
pub fn dense_link_free_space_manager() -> Vec<u8> {
    decode_hex(
        "465348440000b52a0000000000000a000000000000000700000000000000030000000000000004005000780020000000010000000000cd130000000000004e000000000000004e00000000000000e9a83146",
    )
}

/// Returns the used free-space section-list bytes from a dense-link deletion fixture.
///
/// The fixture uses 8-byte offsets and lengths. The crosscheck generates the metadata with
/// each supported HDF5 release and compares the complete bytes.
pub fn dense_link_free_space_sections() -> Vec<u8> {
    decode_hex(
        "46535345007b05000000000000046100009f4f0000009f5f0000009f6f0000009f7f00000001eb0100000000000100000000000000001000013a08001540000000014b1100b58e000000bf37b83d",
    )
}

fn decode_hex(digits: &str) -> Vec<u8> {
    (0..digits.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&digits[at..at + 2], 16).unwrap())
        .collect()
}

/// Returns a heap header with root filter metadata and an updated checksum.
///
/// The filter metadata holds the encoded length of `pipeline`, the root direct block's filtered
/// size `root_size`, its filter mask `mask`, and the encoded `pipeline` bytes.
///
/// # Panics
///
/// Panics if `header` is too short for its checksum or filter metadata, if `pipeline` exceeds
/// [`u16::MAX`] bytes, or if `root_size` does not fit the file's length width.
pub fn with_filtering(
    header: &[u8],
    widths: Widths,
    root_size: u64,
    mask: u32,
    pipeline: &[u8],
) -> Vec<u8> {
    let mut filtered = header[..header.len() - checksum::CHECKSUM].to_vec();
    let encoded_len = u16::try_from(pipeline.len()).expect("fixture pipeline fits u16");
    filtered[FILTER_INFO_LEN_AT..FILTER_INFO_LEN_AT + 2]
        .copy_from_slice(&encoded_len.to_le_bytes());
    bytes::push_uint(&mut filtered, root_size, widths.length);
    filtered.extend_from_slice(&mask.to_le_bytes());
    filtered.extend_from_slice(pipeline);
    checksum::append(&mut filtered);
    filtered
}

/// Returns a file image with a fractal heap of one managed object: the header at offset 0, and
/// the root direct block at offset 256, with `object` after the prefix of the block.
///
/// The image has 64 bytes past the end of the block.
pub fn heap_with_one_object(object: &[u8], widths: Widths) -> Vec<u8> {
    const BLOCK_AT: usize = 256;

    let heap = Header::new(BLOCK_AT as u64).managed_object_count(1);
    let mut image = Image::starting_with(&heap.build(widths));
    image.place(BLOCK_AT, &heap.direct_block(0, object, widths));
    image.append(&[0; 64]);
    image.build()
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
/// Past the twelve 8-byte heap fields come the doubling-table
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

/// Returns the length of the heap header at `header` in `bytes`, with its filter fields when it
/// stores them.
pub fn encoded_header_len(bytes: &[u8], header: usize) -> u64 {
    let filter_info_len = usize::from(bytes::u16_at(bytes, header + FILTER_INFO_LEN_AT));
    let filter_fields_len = if filter_info_len == 0 {
        0
    } else {
        LENGTH + FILTER_MASK_LEN + filter_info_len
    };
    u64::try_from(ROOT_ROWS_AT + 2 + filter_fields_len + checksum::CHECKSUM).unwrap()
}

/// Returns the root indirect block of the filtered heap whose header is at `header` in `bytes`,
/// then the direct children of the root, each as an address and a length.
///
/// The length of a direct block is its filtered size.
///
/// # Panics
///
/// Panics if the root is a direct block or has an indirect child.
pub fn filtered_indirect_root_blocks(bytes: &[u8], header: usize) -> Vec<(u64, u64)> {
    let table_width = usize::from(bytes::u16_at(bytes, header + TABLE_WIDTH_AT));
    let starting_block_size = bytes::u64_at(bytes, header + TABLE_WIDTH_AT + 2);
    let max_direct_block_size = bytes::u64_at(bytes, header + TABLE_WIDTH_AT + 2 + LENGTH);
    let max_heap_size = usize::from(bytes::u16_at(
        bytes,
        header + TABLE_WIDTH_AT + 2 + 2 * LENGTH,
    ));
    let root = usize::try_from(bytes::u64_at(bytes, header + ROOT_ROWS_AT - LENGTH)).unwrap();
    let rows = usize::from(bytes::u16_at(bytes, header + ROOT_ROWS_AT));
    assert_ne!(rows, 0, "the heap's root is an indirect block");
    assert_eq!(
        bytes::slice_at(bytes, root, INDIRECT_BLOCK_SIGNATURE.len()),
        INDIRECT_BLOCK_SIGNATURE
    );
    let direct_rows = usize::try_from(
        max_direct_block_size.trailing_zeros() - starting_block_size.trailing_zeros() + 2,
    )
    .unwrap();
    let mut at = root + INDIRECT_BLOCK_SIGNATURE.len() + 1 + LENGTH + max_heap_size.div_ceil(8);
    let mut children = Vec::new();
    for row in 0..rows {
        for _ in 0..table_width {
            let address = bytes::u64_at(bytes, at);
            at += LENGTH;
            if row >= direct_rows {
                assert_eq!(address, u64::MAX, "the fixture nests no indirect block");
                continue;
            }
            let filtered_size = bytes::u64_at(bytes, at);
            at += LENGTH + FILTER_MASK_LEN;
            if address != u64::MAX {
                children.push((address, filtered_size));
            }
        }
    }
    let root_len = u64::try_from(at + checksum::CHECKSUM - root).unwrap();
    children.insert(0, (u64::try_from(root).unwrap(), root_len));
    children
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

/// The version of a fractal heap direct block, as defined in "Fractal Heap" of the format
/// specification, version 4.0.
pub const BLOCK_VERSION: u8 = VERSION;

/// The position of the owning heap's address in an `FHDB` direct block.
pub const DIRECT_BLOCK_OWNER_AT: usize = DIRECT_BLOCK_SIGNATURE.len() + 1;
/// The position of the checksum in fixture direct blocks with eight-byte addresses and
/// two-byte block offsets.
pub const DIRECT_BLOCK_CHECKSUM_AT: usize = DIRECT_BLOCK_OWNER_AT + 8 + 2;
/// The position of the payload in a direct block with a checksum, eight-byte addresses,
/// and two-byte block offsets.
pub const DIRECT_BLOCK_DATA_AT: usize = DIRECT_BLOCK_CHECKSUM_AT + checksum::CHECKSUM;

/// Byte offsets below are for the eight-byte lengths every file these readers
/// are pointed at carries.
const LENGTH: usize = 8;

/// The offset of the "I/O Filters' Encoded Length" field in a heap header, after the signature,
/// the version, and the heap ID length.
pub const FILTER_INFO_LEN_AT: usize = SIGNATURE.len() + 1 + 2;
// "Fractal Heap" of the format specification, version 4.0, defines the twelve eight-byte length
// and address fields of the version 0 heap header after the flags and the maximum managed object
// size. The table width, the starting and maximum direct block sizes, the maximum heap size, the
// starting root rows and the root block address precede the current root rows.
const TABLE_WIDTH_AT: usize = FILTER_INFO_LEN_AT + 2 + 1 + 4 + 12 * LENGTH;
const ROOT_ROWS_AT: usize = TABLE_WIDTH_AT + 2 + 2 * LENGTH + 2 + 2 + LENGTH;
const FILTER_MASK_LEN: usize = 4;
