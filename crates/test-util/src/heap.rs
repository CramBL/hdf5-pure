//! Reading fractal-heap facts back out of a written file, from the outside.
//!
//! Round-tripping an attribute proves it survived; it does not prove *how* it was
//! stored. These helpers decode the heap header directly so a test can assert the
//! storage class the emitter chose, which is the thing a managed/huge split is
//! actually about.
//!
//! Pure Rust on purpose: pinned here rather than in each test so the two callers
//! cannot drift on the header layout, and so 32-bit targets (which have no
//! reference C library) can still use it.

/// Byte offsets are for the 8-byte offset/length sizes this writer emits.
const SIZE: usize = 8;

/// Whether the file contains a fractal heap at all — the signature of dense
/// (heap) rather than compact (in-object-header) storage.
pub fn has_fractal_heap(bytes: &[u8]) -> bool {
    bytes.windows(4).any(|w| w == b"FRHP")
}

/// Offsets of every fractal-heap header in `bytes`, in file order.
pub fn frhp_offsets(bytes: &[u8]) -> Vec<usize> {
    bytes
        .windows(4)
        .enumerate()
        .filter(|(_, w)| *w == b"FRHP")
        .map(|(at, _)| at)
        .collect()
}

/// Offset of the first fractal-heap header in `bytes`.
///
/// Panics if there is none, since a caller asking about heap storage has already
/// decided the file should have one.
#[track_caller]
fn frhp(bytes: &[u8]) -> usize {
    *frhp_offsets(bytes)
        .first()
        .expect("a dense attribute or link set has a fractal heap header")
}

/// Read a `u64` field from the fractal-heap header at `frhp`, `fields` 8-byte
/// fields past the fixed prefix: signature(4) + version(1) + heap ID length(2) +
/// I/O filter length(2) + flags(1) + maximum managed object size(4).
#[track_caller]
fn frhp_u64_at(bytes: &[u8], frhp: usize, fields: usize) -> u64 {
    let at = frhp + 4 + 1 + 2 + 2 + 1 + 4 + fields * SIZE;
    u64::from_le_bytes(bytes[at..at + SIZE].try_into().expect("8 bytes"))
}

/// Read a `u64` field from the first fractal-heap header in `bytes`.
#[track_caller]
fn frhp_u64(bytes: &[u8], fields: usize) -> u64 {
    frhp_u64_at(bytes, frhp(bytes), fields)
}

/// How many objects the heap stores as fractal-heap *huge* objects — held outside
/// the managed direct blocks and indexed by the huge-objects v2 B-tree.
///
/// The header's field order is next huge object ID, huge B-tree address, free
/// space, free-space manager address, managed space, allocated managed space,
/// allocation iterator, managed object count, huge objects size, then this.
#[track_caller]
pub fn huge_object_count(bytes: &[u8]) -> u64 {
    frhp_u64(bytes, 9)
}

/// [`huge_object_count`] for every heap in the file, in file order — for asserting
/// what a *copy's* heap chose, which the first-heap reader cannot see.
#[track_caller]
pub fn huge_object_counts(bytes: &[u8]) -> Vec<u64> {
    frhp_offsets(bytes)
        .into_iter()
        .map(|at| frhp_u64_at(bytes, at, 9))
        .collect()
}

/// How many objects the heap stores as managed objects, inside its direct blocks.
#[track_caller]
pub fn managed_object_count(bytes: &[u8]) -> u64 {
    frhp_u64(bytes, 7)
}

/// The total byte size the heap declares for its huge objects.
#[track_caller]
pub fn huge_object_bytes(bytes: &[u8]) -> u64 {
    frhp_u64(bytes, 8)
}

/// The heap's "current # of rows in root indirect block": 0 when the root is a
/// single direct block, and otherwise how many doubling-table rows the root
/// indirect block spans.
///
/// Past the twelve 8-byte fields [`frhp_u64_at`] indexes come the doubling-table
/// fields — table width(2), starting block size(8), maximum direct block size(8),
/// maximum heap size(2), starting root rows(2) — then the root block address(8)
/// and this.
#[track_caller]
pub fn root_indirect_rows(bytes: &[u8]) -> u16 {
    let at = frhp(bytes) + 4 + 1 + 2 + 2 + 1 + 4 + 12 * SIZE + 2 + SIZE + SIZE + 2 + 2 + SIZE;
    u16::from_le_bytes(bytes[at..at + 2].try_into().expect("2 bytes"))
}

/// How many fractal-heap indirect blocks the file holds. More than one means the
/// root's own row of them filled up and the table nested.
pub fn indirect_block_count(bytes: &[u8]) -> usize {
    bytes.windows(4).filter(|w| *w == b"FHIB").count()
}

/// Depth of the file's only v2 B-tree, from its header: 0 when the root is a
/// leaf, and one more per level of internal nodes above it.
///
/// A test that means to exercise an internal node has to prove it built one,
/// since the same assertions pass vacuously on a single-leaf tree. Panics unless
/// the file holds exactly one B-tree header, so a caller cannot silently read the
/// depth of a heap's huge-objects index when it meant the attribute name index.
///
/// Header layout: signature(4) + version(1) + type(1) + node size(4) + record
/// size(2) + depth(2).
pub fn sole_btree_depth(bytes: &[u8]) -> u16 {
    let at = sole_btree_header(bytes) + 4 + 1 + 1 + 4 + 2;
    u16::from_le_bytes(bytes[at..at + 2].try_into().expect("2 bytes"))
}

/// Offset of the file's only v2 B-tree header.
///
/// Sole-ness is the point, not a convenience: a heap with a huge object carries a
/// second B-tree whose records are 24 bytes rather than 17, and decoding one at
/// the other's stride yields plausible numbers rather than an error. A caller
/// that means the attribute name index gets told when the file stopped holding
/// only that.
#[track_caller]
fn sole_btree_header(bytes: &[u8]) -> usize {
    let headers: Vec<usize> = bytes
        .windows(4)
        .enumerate()
        .filter(|(_, w)| *w == b"BTHD")
        .map(|(at, _)| at)
        .collect();
    assert_eq!(
        headers.len(),
        1,
        "expected one v2 B-tree in the file, found {}",
        headers.len()
    );
    headers[0]
}

/// The `(creation order, hash)` of `count` dense-attribute name-index records
/// laid out back to back at `first`.
///
/// Record layout, for the 8-byte heap IDs that index uses: heap ID(8) + message
/// flags(1) + creation order(4) + name hash(4). Shared by the leaf and root
/// readers so the two cannot drift on it.
fn name_index_records(bytes: &[u8], first: usize, count: usize) -> Vec<(u32, u32)> {
    const RECORD: usize = 8 + 1 + 4 + 4;
    (0..count)
        .map(|i| {
            let at = first + i * RECORD;
            let field = |off: usize| {
                u32::from_le_bytes(bytes[at + off..at + off + 4].try_into().expect("4 bytes"))
            };
            (field(9), field(13))
        })
        .collect()
}

/// The `(creation order, name hash)` of each record in the file's first v2
/// B-tree leaf node, in the order the node stores them.
///
/// That order is the one a reader binary-searches, so it is the thing a test
/// about record ordering has to look at: reading the attributes back walks the
/// node start to finish and is satisfied by any order at all. Creation order is
/// the attribute's index in the order it was set, which is what identifies
/// *which* attribute a record belongs to without decoding the heap.
///
/// Only meaningful for an index small enough to be a single leaf, and for the
/// 8-byte heap IDs a dense attribute name index uses (record layout: heap ID(8) +
/// message flags(1) + creation order(4) + name hash(4)).
pub fn name_index_leaf_records(bytes: &[u8], count: usize) -> Vec<(u32, u32)> {
    let leaf = bytes
        .windows(4)
        .position(|w| w == b"BTLF")
        .expect("a dense attribute name index has a leaf node");
    // signature(4) + version(1) + type(1), then the records.
    name_index_records(bytes, leaf + 6, count)
}

/// The `(creation order, name hash)` of each record the file's sole v2 B-tree
/// keeps in its *root* node, in the order the node stores them.
///
/// A record in an internal node is one the tree promoted out of the level below:
/// it lives only there, and a search compares against it while choosing which
/// child to descend into. That makes it the record a wrong ordering hurts most,
/// and the reason a test about ordering wants to know a colliding name reached
/// one — on a single-leaf index there is nothing to descend and the same
/// assertions pass without testing anything.
///
/// The root is the one node reachable without decoding a child pointer, whose
/// width varies by depth: the header carries both its address and its own record
/// count. That works at any depth — on a depth-0 tree the root is the single leaf
/// and this returns the whole index — but only a deeper tree makes the answer
/// mean "promoted".
///
/// Header layout past [`sole_btree_depth`]: split %(1) + merge %(1) + root
/// address(8) + records in root(2).
pub fn root_records(bytes: &[u8]) -> Vec<(u32, u32)> {
    let at = sole_btree_header(bytes) + 4 + 1 + 1 + 4 + 2 + 2 + 1 + 1;
    let root = u64::from_le_bytes(bytes[at..at + SIZE].try_into().expect("8 bytes")) as usize;
    let count =
        u16::from_le_bytes(bytes[at + SIZE..at + SIZE + 2].try_into().expect("2 bytes")) as usize;
    // signature(4) + version(1) + type(1), then the records.
    name_index_records(bytes, root + 6, count)
}
