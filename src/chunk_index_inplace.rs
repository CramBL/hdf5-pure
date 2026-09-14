//! In-place Extensible-Array chunk-index growth, driven by the edit engine
//! ([`crate::edit`]) over whichever image the session holds — the machinery
//! behind [`Dataset::append`](crate::Dataset::append) and
//! [`File::open_swmr_writer`](crate::File::open_swmr_writer).
//!
//! It grows a one-dimensional, unlimited, Extensible-Array-indexed
//! dataset *in place*: a new chunk is written at end-of-file or into freed space,
//! its record is
//! stored into an element slot of the chunk index, the index grows by appending
//! new blocks only when a block boundary is crossed (never relocating existing
//! data), and the dataspace dimension and array-header counts are patched. Writes
//! land child-before-parent so a crash (and, for SWMR, a concurrent reader) only
//! ever observes a consistent prefix, with the dataspace dimension published last
//! as the single commit point. The *order* is what a concurrent reader depends
//! on; the `fsync` barriers between the steps carry that order across power loss,
//! and a session under [`SyncPolicy::OnClose`](crate::SyncPolicy) skips them
//! without changing it, taking one barrier at teardown instead.
//!
//! This module owns the byte-level mechanics that do not depend on *why* the
//! append is happening: the writable byte seam ([`Store`]), the per-dataset
//! geometry cache ([`Located`]), and the element-slot / block / super-block
//! writes that maintain the Extensible Array. It is element-width
//! agnostic: the same code path stores a bare address for an unfiltered array
//! (client id 0) or the full `address + compressed_size + filter_mask` record for
//! a filtered array (client id 1), selected by the array header's client id. The
//! callers layer their own policy on top: SWMR sets the superblock consistency
//! flag and refuses filters; the general writer takes an exclusive lock, accepts
//! filters, and relocates a partial trailing chunk.

use core::num::NonZeroUsize;

use crate::address::StoredAddress;
use crate::checksum::jenkins_lookup3;
use crate::chunked_write::{ea_compute_stats, split_into_chunks, write_ea_addr};
use crate::convert::Narrow;
use crate::data_layout::{ChunkIndexLayout, DataLayout};
use crate::dataspace::{Dataspace, MaxExtent};
use crate::datatype::Datatype;
use crate::edit::{LOSSY_TAIL_REFUSAL, pipeline_lossless};
use crate::error::{Error, FormatError};
use crate::extensible_array::{DataBlockGeom, EaGeometry, ExtensibleArrayHeader, SuperBlockGeom};
use crate::fill_value::FillPattern;
use crate::filter_pipeline::FilterPipeline;
use crate::filters::{ChunkContext, FilterScratch, compress_chunk_with, decompress_chunk};
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::source::Source;

/// Counts the two index-block allocations, so a test can say whether its fixture
/// reached them.
///
/// Both are rare: with the C library's Extensible-Array defaults a data block is
/// allocated roughly every sixteen appends and a *super* block twice in the first
/// five hundred. A crash sweep positioned over the wrong window exercises neither
/// while looking exactly like one that exercises both, which is how the first
/// draft of [`crate::crash_replay`] came to pass with both barriers below
/// deleted.
#[cfg(test)]
pub(crate) mod alloc_probe {
    use core::cell::Cell;

    thread_local! {
        static DATA_BLOCKS: Cell<usize> = const { Cell::new(0) };
        static SUPER_BLOCKS: Cell<usize> = const { Cell::new(0) };
    }

    pub(crate) fn note_data_block() {
        DATA_BLOCKS.with(|c| c.set(c.get() + 1));
    }

    pub(crate) fn note_super_block() {
        SUPER_BLOCKS.with(|c| c.set(c.get() + 1));
    }

    /// Zero both counters and return what they held, so a caller measures one
    /// window rather than the whole process.
    pub(crate) fn take() -> (usize, usize) {
        (
            DATA_BLOCKS.with(|c| c.replace(0)),
            SUPER_BLOCKS.with(|c| c.replace(0)),
        )
    }
}

/// The undefined-address sentinel for a given offset size.
pub(crate) fn undef_addr(offset_size: u8) -> u64 {
    match offset_size {
        4 => 0xFFFF_FFFF,
        _ => u64::MAX,
    }
}

/// Push one undefined Extensible-Array element to `buf`: an offset-sized
/// all-`0xFF` address, followed (for a filtered array whose element is wider than
/// one address) by zeroed compressed-size and filter-mask fields. Mirrors
/// `chunked_write::write_undefined_element` so a freshly-allocated block matches
/// what the bulk writer and reader expect.
fn push_undef_element(buf: &mut Vec<u8>, offset_size: u8, ea_elem_size: usize) {
    write_ea_addr(buf, undef_addr(offset_size), offset_size);
    for _ in offset_size as usize..ea_elem_size {
        buf.push(0);
    }
}

/// One stored Extensible-Array element: the chunk address, plus (for filtered
/// arrays) the stored/compressed chunk size and this chunk's filter mask. The
/// size and mask are ignored for unfiltered arrays (client id 0).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ElemRecord {
    pub addr: StoredAddress,
    pub stored_size: u64,
    pub filter_mask: u32,
}

impl ElemRecord {
    /// A bare-address record (the size/mask are unused for unfiltered arrays).
    pub(crate) fn addr_only(addr: StoredAddress) -> Self {
        Self {
            addr,
            stored_size: 0,
            filter_mask: 0,
        }
    }
}

/// Writable byte-level I/O the Extensible-Array growth engine ([`Located`])
/// depends on, extending the read-only [`Source`] seam with in-place mutation.
/// Its one production owner is the read-write engine's `EditStore`, which drives
/// it against whichever [`FileImage`](crate::image::FileImage) the session
/// opened — a whole-file mirror or a handle with no mirror at all. That second
/// case is why every engine *read* goes through [`Source`] (bounded,
/// random-access) rather than a whole-file `&[u8]`. Genericizing the engine over
/// this trait is what lets one long-lived session drive an O(1) in-place append
/// against its own image and exclusive lock, rather than opening a second one
/// that would take a second lock and keep a divergent view.
pub(crate) trait Store: Source {
    /// This file's address (offset) field width in bytes.
    fn offset_size(&self) -> u8;
    /// This file's length field width in bytes.
    fn length_size(&self) -> u8;

    /// Allocate space for `bytes` as *raw* data and write them there, returning
    /// the position they occupy in the image. **Every allocation this engine
    /// makes goes through here**, chunk data and the extensible-array blocks
    /// indexing it alike. It is the trait's only allocation primitive.
    ///
    /// A store may serve the allocation out of a region an earlier commit freed
    /// rather than growing the file (issue #349), or append at end-of-file. The
    /// address is below the file's *logical* length either way once this returns,
    /// so a later in-place patch of the region addresses bytes the store already
    /// accounts for.
    ///
    /// What [`apply_ea_append`]'s phase 1 needs is stronger — a chunk an index
    /// element points at must lie inside the *recorded* end-of-file, which the
    /// superblock holds and this trait exposes no accessor for. Phase 1 gets
    /// there in two steps rather than from this contract: an appended region is
    /// covered by the patch it then issues, and a reused region is covered
    /// because a list this may draw from is populated only by a commit — a
    /// staged one, or the manager rewrite a persisting session makes before an
    /// append may reuse at all (`WriteEngine::reserve_for_immediate_append`,
    /// issue #387) — and every commit leaves the recorded end-of-file at the
    /// file's length.
    ///
    /// The bytes do **not** necessarily reach the disk here: a session gathers
    /// its writes and issues them at the next ordering barrier (issue #288).
    /// Anything that publishes an address into this region to a *lower* address —
    /// a pointer, an element count, the superblock's end-of-file — must
    /// [`sync`](Self::sync) between the two, because gathered writes go out in
    /// address order and would otherwise publish first; `ea_insert` and the two
    /// data-block allocators do exactly that. Those barriers are unchanged by
    /// which address comes back. They exist for the *appended* case, where the
    /// region sits above the pointer naming it; a reused region below its pointer
    /// is already written first by that same address order, so the barrier is
    /// then redundant rather than wrong — and which case applies is not knowable
    /// at the call site.
    ///
    /// On a paged file an append keeps pages homogeneous by never letting raw and
    /// metadata bytes share a page (issue #173), padding the tail page first when
    /// the page type changes. An index block is metadata by the format's taxonomy,
    /// but on a paged file this crate places a chunked dataset's index in raw pages
    /// beside its chunk data, and the reclaim path
    /// (`WriteEngine::chunked_storage_spans`) frees both halves as raw on that
    /// basis. Allocating an index block into a metadata page here would put a span
    /// the reclaim reports as raw inside a metadata page, so a later delete would
    /// advertise metadata-page bytes for raw reuse — the one thing the paged
    /// strategy exists to prevent, and invisible to the reference library, which
    /// reads a mixed-page file without complaint (issue #198). That is the same
    /// reason a reuse must draw from the raw list and no other.
    fn alloc_raw(&mut self, bytes: &[u8]) -> Result<u64, Error>;
    /// Overwrite `[offset, offset + bytes.len())` in place.
    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error>;
    /// Advance the superblock's recorded end-of-file to the store's current
    /// logical length and rewrite the superblock.
    fn patch_superblock_eof(&mut self) -> Result<(), Error>;
    /// An ordering barrier: everything written before this reaches the operating
    /// system before anything written after it, and under
    /// [`SyncPolicy::Always`](crate::SyncPolicy) it is forced to durable storage
    /// as well.
    ///
    /// Both halves matter, and only the second is the policy's to skip.
    /// [`SyncPolicy::OnClose`](crate::SyncPolicy) issues no `fsync` here but still
    /// releases the session's gathered writes, so this is never a no-op: writes
    /// between two barriers go out in *address* order, and every publish point in
    /// this format sits at a lower address than the content it names. What
    /// `OnClose` gives up is power-loss durability, not the order a concurrent
    /// reader observes (issue #288).
    fn sync(&mut self) -> Result<(), Error>;

    /// Read an offset-sized address at `offset`.
    fn read_addr_at(&self, offset: u64) -> Result<StoredAddress, Error> {
        let mut buf = [0u8; 8];
        let width = if self.offset_size() == 4 { 4 } else { 8 };
        self.read_at(offset, &mut buf[..width])?;
        Ok(StoredAddress::new(u64::from_le_bytes(buf)))
    }

    /// Change a checksummed structure and publish it as **one** write.
    ///
    /// Reads `[start, cks_off + 4)`, places `value` at `at` inside the body
    /// before `cks_off`, recomputes the Jenkins checksum over it, and writes back
    /// everything from `at` through the checksum field.
    ///
    /// The single write is the point, not an optimization. A value written
    /// separately from the checksum covering it is atomic only where the
    /// gatherer joins the two, which it does only when both land in the same
    /// page; a structure wider than one page publishes the new value under the
    /// *old* checksum, and a failure between the two writes leaves it unreadable
    /// (issue #307). Every other publish point in this crate — the superblock
    /// repoint above all — is already a single write, and this is what brings
    /// these into line rather than a new mechanism.
    ///
    /// One write is not crash-atomicity: a `pwrite` wide enough may still tear at
    /// a device boundary. It is the same window the superblock publish has always
    /// had, which is this crate's bar.
    ///
    /// The write starts at `at` rather than at `start`, so patching an element
    /// near the end of a large block does not rewrite the block. The *read* still
    /// covers the whole body, because the checksum does. Measured over six
    /// appends at the same number of writes: 3,066 bytes against 5,952 unpaged
    /// and 31,266 against 34,158 on a paged file with a 4 KiB header — the index
    /// block, not the object header, is where most of that sits.
    /// `crash_replay::a_publish_writes_from_the_byte_it_changed` is what holds
    /// it, since the write count is the same either way.
    fn publish_checksummed(
        &mut self,
        start: u64,
        cks_off: u64,
        at: u64,
        value: &[u8],
    ) -> Result<(), Error> {
        let span = (cks_off + 4 - start).to_usize()?;
        // A range outside the structure is refused rather than asserted: every
        // offset here is derived from geometry parsed out of the file, so a
        // malformed one is untrusted input, not a broken invariant. Checked
        // before the read, so a refusal costs nothing.
        let from = at
            .checked_sub(start)
            .and_then(|d| d.to_usize().ok())
            .filter(|&f| f.checked_add(value.len()).is_some_and(|e| e <= span - 4))
            .ok_or(Error::AppendUnsupported(
                "a checksummed structure was patched outside itself",
            ))?;
        let mut bytes = self.read_exact_at(start, span)?;
        bytes[from..from + value.len()].copy_from_slice(value);
        let (body, cks_field) = bytes.split_at_mut(span - 4);
        cks_field.copy_from_slice(&jenkins_lookup3(body).to_le_bytes());
        self.write_at(at, &bytes[from..])
    }
}

/// Widest Extensible-Array element this engine can be asked to place: an
/// 8-byte address, a stored size no wider than the `u64` that fills it, and a
/// 4-byte filter mask. `locate` refuses any file naming more.
const MAX_EA_ELEM: usize = 8 + 8 + 4;

/// Absolute file offsets of the object-header messages a caller may need to
/// parse after locating a dataset.
pub(crate) struct MessageSpans {
    /// `(data_off, size)` of the Datatype message.
    pub datatype: (u64, usize),
    /// `(data_off, size)` of the Filter Pipeline message, when present.
    pub filter: Option<(u64, usize)>,
    /// The Fill Value message, when present: its type (which distinguishes the
    /// versioned form from the legacy one) and `(data_off, size)`. An append
    /// needs it because the overhang of the chunk it completes must hold the
    /// dataset's fill value rather than zeros (issue #296).
    pub fill: Option<(MessageType, u64, usize)>,
}

/// Result of locating a dataset: its maintained geometry plus the message spans
/// and filter status the caller needs to decide policy.
pub(crate) struct LocateResult {
    pub located: Located,
    pub spans: MessageSpans,
}

/// Metadata located once per dataset, then maintained across appends.
pub(crate) struct Located {
    /// File offset of the dataspace message's first current-dimension value.
    pub dim0_off: u64,
    /// Current length along the unlimited (axis-0) dimension.
    pub current_dim: u64,
    /// Object-header chunk that contains the dataspace message: the byte range
    /// whose Jenkins checksum must be recomputed after patching the dimension.
    /// The checksum itself occupies `chunk_msg_end .. chunk_msg_end + 4`.
    pub ohdr_chunk_start: u64,
    pub ohdr_chunk_msg_end: u64,

    /// Elements per chunk along axis 0 (the only varying axis for rank 1).
    pub chunk_elems: u64,
    /// Bytes per dataset element (datatype size).
    pub elem_bytes: NonZeroUsize,
    /// Bytes per chunk (uncompressed).
    pub chunk_bytes: usize,

    /// Extensible Array client id: 0 = unfiltered (bare-address element), 1 =
    /// filtered (`address + compressed_size + filter_mask` element).
    pub client_id: u8,
    /// Extensible Array header address and derived geometry.
    pub ea_addr: StoredAddress,
    pub geom: EaGeometry,
    pub idx_blk_elmts: u64,
    /// Size of one stored EA element in bytes (offset size for unfiltered; wider
    /// for filtered).
    pub ea_elem_size: usize,
    pub page_nelmts: u64,
    /// Block-offset field width inside EA blocks (= ceil(max_nelmts_bits / 8)).
    pub blk_off_size: usize,
    /// Address of the EA index block (`EAIB`).
    pub index_block_addr: StoredAddress,
    /// Current number of chunks indexed (EA element count).
    pub num_chunks: u64,
}

impl Located {
    /// Locate the dataset whose object-header address is `oh_addr`, deriving its
    /// dataspace/layout/Extensible-Array geometry. Filter-neutral: it records
    /// whether the dataset is filtered and returns the datatype/filter message
    /// spans, leaving the accept/reject policy to the caller. `unsupported`
    /// builds the caller's "unsupported target" error so each writer reports
    /// through its own variant. Path resolution is the caller's job: an owned
    /// handle already carries its address, and the path-addressed writers
    /// resolve against their own mirrors.
    ///
    /// Every read is a bounded [`Source`] window (the object-header chunks, one
    /// message span, the EA header), so this works over a store with no
    /// whole-file mirror.
    pub(crate) fn locate_at<F: Store>(
        file: &F,
        oh_addr: u64,
        unsupported: fn(&'static str) -> Error,
    ) -> Result<LocateResult, Error> {
        let os = file.offset_size();
        let ls = file.length_size();

        let walk = walk_v2_object_header(file, oh_addr, os, ls)?;

        let dataspace_msg = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::Dataspace)
            .ok_or(Error::MissingMessage(MessageType::Dataspace))?;
        let layout_msg = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DataLayout)
            .ok_or(Error::MissingMessage(MessageType::DataLayout))?;
        let datatype_msg = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::Datatype)
            .ok_or(Error::MissingMessage(MessageType::Datatype))?;
        let filter_msg = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FilterPipeline);
        // The versioned message wins over the legacy one when a header carries
        // both, matching how the read path picks.
        let fill_msg = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FillValue)
            .or_else(|| {
                walk.messages
                    .iter()
                    .find(|m| m.msg_type == MessageType::FillValueOld)
            });

        // Each of these is parsed below as though its bytes were the message. A
        // *shared* record's body is a reference to a message stored elsewhere —
        // the shape `H5Tcommit` gives a dataset's element type — and decodes as a
        // well-formed type of the wrong kind, so an append against it would size
        // its elements from a type the dataset does not have. Refuse instead.
        for msg in [Some(dataspace_msg), Some(datatype_msg), filter_msg]
            .into_iter()
            .flatten()
        {
            if msg.flags.is_shared() {
                return Err(unsupported(
                    "dataset has a committed (shared) datatype, dataspace, or filter pipeline",
                ));
            }
        }

        // Parse the dataspace: must be rank 1 with one unlimited dimension.
        let ds_bytes = file.read_metadata_at(dataspace_msg.data_off, dataspace_msg.size)?;
        let dataspace = Dataspace::parse(&ds_bytes, ls)?;
        if dataspace.rank != 1 {
            return Err(unsupported("only rank-1 datasets are supported"));
        }
        if !matches!(
            dataspace.max_dimensions.as_deref(),
            Some([MaxExtent::Unlimited, ..])
        ) {
            return Err(unsupported("dataset has no unlimited (maxshape) dimension"));
        }
        let current_dim = dataspace.dimensions[0];
        // v2 dataspace header is 4 bytes (version, rank, flags, type); dim 0 follows.
        let dim0_off = dataspace_msg.data_off + 4;

        // Parse the layout: must be a v4 chunked Extensible Array (index type 4).
        let layout_bytes = file.read_metadata_at(layout_msg.data_off, layout_msg.size)?;
        let layout = DataLayout::parse(&layout_bytes, os, ls)?;
        let (ea_addr, chunk_dims) = match layout {
            DataLayout::Chunked {
                chunk_dimensions,
                index:
                    ChunkIndexLayout::ExtensibleArray {
                        address: Some(addr),
                    },
            } => (addr, chunk_dimensions),
            DataLayout::Chunked {
                index: ChunkIndexLayout::ExtensibleArray { address: None },
                ..
            } => {
                // An Extensible Array whose header/index block is not allocated
                // yet: an empty dataset the C library created without writing any
                // chunk (this crate's writer allocates it eagerly). In-place
                // growth needs an existing index; make the first append through
                // `Dataset::append_staged`, which materializes the index, or
                // create the dataset with initial data.
                return Err(unsupported(
                    "the dataset's extensible-array index is not allocated yet (an empty \
                     dataset with no chunks); write initial data at creation or make the \
                     first append with Dataset::append_staged",
                ));
            }
            _ => {
                return Err(unsupported(
                    "only Extensible-Array-indexed chunked datasets are supported",
                ));
            }
        };
        // chunk_dimensions for a v4 layout includes the element-size pseudo
        // dimension as its last entry; the leading entry is the axis-0 chunk size.
        if chunk_dims.len() != 2 {
            return Err(unsupported(
                "unexpected chunk dimensionality (expected a rank-1 chunked layout)",
            ));
        }
        let chunk_elems = chunk_dims[0] as u64;
        // A zero element-size pseudo-dimension is a malformed layout; refuse
        // rather than divide by it when validating an append length. The binding
        // carries that refusal forward, so no later step re-checks it.
        let Some(elem_bytes) = NonZeroUsize::new(chunk_dims[1] as usize) else {
            return Err(unsupported("dataset has a zero-sized element"));
        };
        let chunk_bytes = chunk_elems.to_usize()? * elem_bytes.get();

        let ea_header = ExtensibleArrayHeader::parse_from_source(file, ea_addr, os, ls)?;
        let has_filters = filter_msg.is_some();
        if (ea_header.client_id == 1) != has_filters {
            return Err(unsupported(
                "dataset filter metadata is inconsistent (chunk-index client id \
                 disagrees with the filter pipeline)",
            ));
        }
        // A filtered element is `address + compressed_size + filter_mask`, so its
        // stored width must leave room for a size field of one to eight bytes.
        // Reject a corrupt header outside that range before the width arithmetic
        // (`ea_elem_size - offset_size - 4`) underflows below it, or names above
        // it a size field wider than the `u64` that fills it. The whole-file read
        // path refuses both ends in `read_variable_length`.
        let elem_w = ea_header.element_size as usize;
        if ea_header.client_id == 1 && !(os as usize + 5..=os as usize + 12).contains(&elem_w) {
            return Err(unsupported(
                "malformed filtered extensible-array element width",
            ));
        }
        let geom = EaGeometry::from_header(&ea_header);
        let page_nelmts = 1u64 << ea_header.max_dblk_nelmts_bits;
        let blk_off_size = (ea_header.max_nelmts_bits as usize).div_ceil(8);
        let index_block_addr = ea_header.index_block_address;
        // The dataspace dimension is the single commit point; the EA element
        // count is published one step earlier. If a prior writer crashed between
        // the two, the on-disk EA count is ahead of the committed dimension. Seed
        // the chunk count from the *committed* dimension so the next append rolls
        // forward from the last commit -- overwriting any slots a crashed writer
        // wrote but never committed -- instead of appending past them.
        let num_chunks = if chunk_elems == 0 {
            ea_header.num_elements
        } else {
            current_dim.div_ceil(chunk_elems)
        };

        Ok(LocateResult {
            located: Located {
                dim0_off,
                current_dim,
                ohdr_chunk_start: dataspace_msg.chunk_start,
                ohdr_chunk_msg_end: dataspace_msg.chunk_msg_end,
                chunk_elems,
                elem_bytes,
                chunk_bytes,
                client_id: ea_header.client_id,
                ea_addr,
                geom,
                idx_blk_elmts: ea_header.idx_blk_elmts as u64,
                ea_elem_size: ea_header.element_size as usize,
                page_nelmts,
                blk_off_size,
                index_block_addr,
                num_chunks,
            },
            spans: MessageSpans {
                datatype: (datatype_msg.data_off, datatype_msg.size),
                filter: filter_msg.map(|m| (m.data_off, m.size)),
                fill: fill_msg.map(|m| (m.msg_type, m.data_off, m.size)),
            },
        })
    }

    /// Read the element record stored at byte offset `off`.
    fn read_element_at<F: Store>(&self, file: &F, off: u64) -> Result<ElemRecord, Error> {
        let os = file.offset_size() as usize;
        if self.client_id == 0 {
            return Ok(ElemRecord::addr_only(file.read_addr_at(off)?));
        }
        // One bounded read of the whole fixed-width element record.
        let elem = file.read_exact_at(off, self.ea_elem_size)?;
        let addr_w = if file.offset_size() == 4 { 4 } else { 8 };
        let mut a = [0u8; 8];
        a[..addr_w].copy_from_slice(&elem[..addr_w]);
        let addr = StoredAddress::new(u64::from_le_bytes(a));
        let csz = self.ea_elem_size - os - 4;
        let mut stored_size = 0u64;
        for i in 0..csz {
            stored_size |= (elem[os + i] as u64) << (8 * i);
        }
        let fm = os + csz;
        let filter_mask = u32::from_le_bytes([elem[fm], elem[fm + 1], elem[fm + 2], elem[fm + 3]]);
        Ok(ElemRecord {
            addr,
            stored_size,
            filter_mask,
        })
    }

    /// Byte offset of element `e`'s record, or `None` when the containing block
    /// (or super block) is not yet allocated, i.e. the element does not exist on
    /// disk yet.
    fn elem_slot_off<F: Store>(&self, file: &F, e: u64) -> Result<Option<u64>, Error> {
        let os = file.offset_size() as usize;
        let elem_size = self.ea_elem_size as u64;
        let idx = self.idx_blk_elmts;
        let blk_off = self.blk_off_size;
        let page_nelmts = self.page_nelmts;

        if e < idx {
            let ib_prefix = (4 + 1 + 1 + os) as u64;
            return Ok(Some(
                self.index_block_addr.get() + ib_prefix + e * elem_size,
            ));
        }

        let region = locate_data_block(&self.geom, idx, e);
        if region.ndblks == 0 {
            return Err(Error::AppendUnsupported(
                "chunk index geometry does not cover the appended element",
            ));
        }
        let is_paged = region.blocks(page_nelmts).is_paged();
        let slot = e - region.db_start;

        // Resolve the owning super block (if any) and the data-block pointer,
        // returning None if either is unallocated.
        let sblk_addr = match region.parent {
            Parent::Super { sblk_j, .. } => {
                let a = self.super_block_addr(file, sblk_j)?;
                if a.is_undefined(file.offset_size()) {
                    return Ok(None);
                }
                Some(a)
            }
            Parent::IndexDirect { .. } => None,
        };
        let dblk_ptr_off = self.dblk_ptr_off(sblk_addr, &region, os, blk_off)?;
        let dblk_addr = file.read_addr_at(dblk_ptr_off)?;
        if dblk_addr.is_undefined(file.offset_size()) {
            return Ok(None);
        }

        let off = if !is_paged {
            let db_prefix = (4 + 1 + 1 + os + blk_off) as u64;
            dblk_addr.get() + db_prefix + slot * elem_size
        } else {
            let header_size = (4 + 1 + 1 + os + blk_off + 4) as u64;
            let page = slot / page_nelmts;
            let slot_in_page = slot % page_nelmts;
            let page_bytes = page_nelmts * elem_size + 4;
            let page_off = dblk_addr.get() + header_size + page * page_bytes;
            page_off + slot_in_page * elem_size
        };
        Ok(Some(off))
    }

    /// File offset of the data-block-pointer slot for `region` (either a direct
    /// pointer in the index block or a pointer inside the resolved super block).
    fn dblk_ptr_off(
        &self,
        sblk_addr: Option<StoredAddress>,
        region: &DataBlockLoc,
        os: usize,
        blk_off: usize,
    ) -> Result<u64, Error> {
        match region.parent {
            Parent::IndexDirect { ordinal } => {
                let ib_prefix = (4 + 1 + 1 + os) as u64;
                Ok(self.index_block_addr.get()
                    + ib_prefix
                    + self.idx_blk_elmts * self.ea_elem_size as u64
                    + (ordinal * os) as u64)
            }
            Parent::Super { dblk_local, .. } => {
                let sblk_addr = sblk_addr.expect("super-block address resolved for a Super parent");
                Ok(sb_dblk_slot_off(
                    os,
                    sblk_addr,
                    dblk_local,
                    region.super_block(self.page_nelmts),
                    blk_off,
                ))
            }
        }
    }

    /// Read element `e`, or `None` if its slot is not allocated / is undefined.
    pub(crate) fn read_element<F: Store>(
        &self,
        file: &F,
        e: u64,
    ) -> Result<Option<ElemRecord>, Error> {
        match self.elem_slot_off(file, e)? {
            None => Ok(None),
            Some(off) => {
                let rec = self.read_element_at(file, off)?;
                if rec.addr.is_undefined(file.offset_size()) {
                    Ok(None)
                } else {
                    Ok(Some(rec))
                }
            }
        }
    }

    /// Store `rec` into element slot `e` of the chunk index, allocating new data
    /// blocks / super blocks as block boundaries are crossed, and
    /// re-checksumming the touched block. Works for both a fresh insert (the
    /// block is allocated on first touch) and an in-place update of an existing
    /// element (the block already exists, so it is reused rather than
    /// re-allocated). Handles non-paged and paged data blocks.
    pub(crate) fn ea_insert<F: Store>(
        &self,
        file: &mut F,
        e: u64,
        rec: ElemRecord,
    ) -> Result<(), Error> {
        let os = file.offset_size() as usize;
        let elem_size = self.ea_elem_size as u64;
        let idx = self.idx_blk_elmts;
        let blk_off = self.blk_off_size;

        // Inline element slots live directly in the index block.
        if e < idx {
            let ib_prefix = (4 + 1 + 1 + os) as u64;
            let slot_off = self.index_block_addr.get() + ib_prefix + e * elem_size;
            let mut buf = [0u8; MAX_EA_ELEM];
            let n = self.element_bytes(&mut buf, os, rec)?;
            return self.publish_index_block(file, slot_off, &buf[..n]);
        }

        let region = locate_data_block(&self.geom, idx, e);
        if region.ndblks == 0 {
            return Err(Error::AppendUnsupported(
                "chunk index geometry does not cover the appended element",
            ));
        }
        let dblk_nelmts = region.dblk_nelmts;
        let sb_geom = region.super_block(self.page_nelmts);
        let is_paged = sb_geom.blocks.is_paged();
        let slot = e - region.db_start;
        let block_offset_rel = region.db_start - idx;

        // Ensure the owning super block exists (idempotent) for a Super parent.
        let sblk_addr = match region.parent {
            Parent::Super { sblk_j, .. } => {
                Some(self.ensure_super_block(file, sblk_j, region.sb_block_offset, sb_geom)?)
            }
            Parent::IndexDirect { .. } => None,
        };

        // Resolve the data-block address, allocating a fresh block when the
        // parent pointer is undefined (first touch of the block). Keying on the
        // pointer being undefined -- rather than on `slot == 0` -- makes an
        // in-place element update reuse the existing block instead of leaking it.
        let dblk_ptr_off = self.dblk_ptr_off(sblk_addr, &region, os, blk_off)?;
        let existing = file.read_addr_at(dblk_ptr_off)?;
        let dblk_addr = if existing.is_undefined(file.offset_size()) {
            let new_addr = if is_paged {
                self.alloc_undef_paged_data_block(file, dblk_nelmts, block_offset_rel)?
            } else {
                self.alloc_undef_data_block(file, dblk_nelmts, block_offset_rel)?
            };
            #[cfg(test)]
            alloc_probe::note_data_block();
            // As in `ensure_super_block`: an appended block sits above its parent
            // pointer, so they need a barrier between them or an address-ordered
            // flush names a block that is not there yet.
            file.sync()?;
            match region.parent {
                Parent::IndexDirect { .. } => {
                    self.publish_index_block(
                        file,
                        dblk_ptr_off,
                        &new_addr.get().to_le_bytes()[..os],
                    )?;
                }
                Parent::Super { .. } => self.publish_super_block(
                    file,
                    sblk_addr.unwrap(),
                    sb_geom,
                    blk_off,
                    dblk_ptr_off,
                    &new_addr.get().to_le_bytes()[..os],
                )?,
            }
            new_addr
        } else {
            existing
        };

        if !is_paged {
            let db_prefix = (4 + 1 + 1 + os + blk_off) as u64;
            let block_start = dblk_addr.get();
            let elem_off = block_start + db_prefix + slot * elem_size;
            let mut buf = [0u8; MAX_EA_ELEM];
            let n = self.element_bytes(&mut buf, os, rec)?;
            let cks_off = block_start + db_prefix + dblk_nelmts * elem_size;
            file.publish_checksummed(block_start, cks_off, elem_off, &buf[..n])?;
        } else {
            let page_nelmts = self.page_nelmts;
            let header_size = (4 + 1 + 1 + os + blk_off + 4) as u64;
            let page = slot / page_nelmts;
            let slot_in_page = slot % page_nelmts;
            let page_bytes = page_nelmts * elem_size + 4;
            let page_off = dblk_addr.get() + header_size + page * page_bytes;
            let elem_off = page_off + slot_in_page * elem_size;
            let mut buf = [0u8; MAX_EA_ELEM];
            let n = self.element_bytes(&mut buf, os, rec)?;
            let page_cks_off = page_off + page_nelmts * elem_size;
            file.publish_checksummed(page_off, page_cks_off, elem_off, &buf[..n])?;

            if slot_in_page == 0 {
                let sblk_addr = sblk_addr.unwrap();
                let npages = dblk_nelmts / self.page_nelmts;
                if let Parent::Super { dblk_local, .. } = region.parent {
                    let global_page = dblk_local as u64 * npages + page;
                    let (byte, set) = sb_page_bit(file, sblk_addr, blk_off, global_page)?;
                    self.publish_super_block(file, sblk_addr, sb_geom, blk_off, byte, &[set])?;
                }
            }
        }
        Ok(())
    }

    /// Address of an already-allocated super block (`sblk_j`-th super-block
    /// pointer in the index block); the undefined sentinel when not yet allocated.
    fn super_block_addr<F: Store>(&self, file: &F, sblk_j: usize) -> Result<StoredAddress, Error> {
        let os = file.offset_size() as usize;
        let ib_prefix = (4 + 1 + 1 + os) as u64;
        let ndblk_addrs = self.geom.direct_dblk_nelmts.len();
        let slot_off = self.index_block_addr.get()
            + ib_prefix
            + self.idx_blk_elmts * self.ea_elem_size as u64
            + ((ndblk_addrs + sblk_j) * os) as u64;
        file.read_addr_at(slot_off)
    }

    /// Return the address of super block `sblk_j`, allocating an empty one (all
    /// data-block pointers undefined, plus a zeroed page-init bitmap when its data
    /// blocks are paged) if it does not exist yet.
    fn ensure_super_block<F: Store>(
        &self,
        file: &mut F,
        sblk_j: usize,
        sb_block_offset: u64,
        sb: SuperBlockGeom,
    ) -> Result<StoredAddress, Error> {
        let existing = self.super_block_addr(file, sblk_j)?;
        if !existing.is_undefined(file.offset_size()) {
            return Ok(existing);
        }
        let os = file.offset_size() as usize;
        let ib_prefix = (4 + 1 + 1 + os) as u64;
        let ndblk_addrs = self.geom.direct_dblk_nelmts.len();

        let bitmap = vec![0u8; sb.bitmap_size().to_usize()?];
        let undef = vec![undef_addr(file.offset_size()); sb.ndblks.to_usize()?];
        let aesb = crate::chunked_write::build_aesb(
            self.ea_addr.get(),
            sb_block_offset,
            &bitmap,
            &undef,
            file.offset_size(),
            self.blk_off_size,
            self.client_id,
        );
        // Every path into this engine rejects a file with a userblock
        // (`WriteEngine::append_prepare` and `File::open_swmr_writer`), so the base address
        // is zero and the position `alloc_raw` returns is the address the file stores. The
        // three further allocations below rest on the same invariant.
        let new_addr = StoredAddress::new(file.alloc_raw(&aesb)?);
        #[cfg(test)]
        alloc_probe::note_super_block();
        // The block exists before anything names it. When its bytes were appended
        // they sit above the slot below, which is in the index block near the
        // front, so without this the two are one barrier-free window and a
        // gathering image issues them in *address* order — the pointer first
        // (issue #288). A block drawn from freed space may land either side of
        // the slot, and which it is is not knowable here, so the barrier stands
        // unconditionally; see `Store::alloc_raw`.
        file.sync()?;

        let slot_off = self.index_block_addr.get()
            + ib_prefix
            + self.idx_blk_elmts * self.ea_elem_size as u64
            + ((ndblk_addrs + sblk_j) * os) as u64;
        self.publish_index_block(file, slot_off, &new_addr.get().to_le_bytes()[..os])?;
        Ok(new_addr)
    }

    /// Allocate a fresh non-paged data block (`EADB`) with every element
    /// slot undefined, returning its address.
    fn alloc_undef_data_block<F: Store>(
        &self,
        file: &mut F,
        dblk_nelmts: u64,
        block_offset_rel: u64,
    ) -> Result<StoredAddress, Error> {
        let os = file.offset_size();
        let mut buf = Vec::new();
        buf.extend_from_slice(b"EADB");
        buf.push(0); // version
        buf.push(self.client_id);
        write_ea_addr(&mut buf, self.ea_addr.get(), os);
        buf.extend_from_slice(&block_offset_rel.to_le_bytes()[..self.blk_off_size]);
        for _ in 0..dblk_nelmts {
            push_undef_element(&mut buf, os, self.ea_elem_size);
        }
        let cks = jenkins_lookup3(&buf);
        buf.extend_from_slice(&cks.to_le_bytes());
        Ok(StoredAddress::new(file.alloc_raw(&buf)?))
    }

    /// Allocate a fresh *paged* data block (`EADB`): a header carrying its
    /// own checksum, followed by `dblk_nelmts / page_nelmts` fully-undefined pages
    /// (each `page_nelmts` undefined elements + a checksum).
    fn alloc_undef_paged_data_block<F: Store>(
        &self,
        file: &mut F,
        dblk_nelmts: u64,
        block_offset_rel: u64,
    ) -> Result<StoredAddress, Error> {
        let os = file.offset_size();
        let page_nelmts = self.page_nelmts;
        let mut buf = Vec::new();
        buf.extend_from_slice(b"EADB");
        buf.push(0); // version
        buf.push(self.client_id);
        write_ea_addr(&mut buf, self.ea_addr.get(), os);
        buf.extend_from_slice(&block_offset_rel.to_le_bytes()[..self.blk_off_size]);
        let header_cks = jenkins_lookup3(&buf);
        buf.extend_from_slice(&header_cks.to_le_bytes());

        let npages = (dblk_nelmts / page_nelmts).to_usize()?;
        for _ in 0..npages {
            let mut page = Vec::with_capacity(page_nelmts.to_usize()? * self.ea_elem_size + 4);
            for _ in 0..page_nelmts {
                push_undef_element(&mut page, os, self.ea_elem_size);
            }
            let page_cks = jenkins_lookup3(&page);
            page.extend_from_slice(&page_cks.to_le_bytes());
            buf.extend_from_slice(&page);
        }
        Ok(StoredAddress::new(file.alloc_raw(&buf)?))
    }

    /// Width of a filtered element's stored-size field. Zero for an unfiltered
    /// array, whose element is a bare address.
    fn element_size_width(&self, os: usize) -> usize {
        if self.client_id == 0 {
            return 0;
        }
        // `element_size` is one byte out of the array header, so a file can name
        // a width the three fields do not fit in. `locate` is where that byte is
        // refused; by here it is in `os + 5 ..= os + 12`.
        debug_assert!(
            (os + 5..=os + 12).contains(&self.ea_elem_size),
            "locate admitted an element width of {} for a {os}-byte address",
            self.ea_elem_size
        );
        self.ea_elem_size - os - 4
    }

    /// Refuse a stored chunk size the element's size field cannot hold.
    ///
    /// Separate from [`element_size_width`](Self::element_size_width) because
    /// the two are needed at different times: the width when an element is laid
    /// out (phase 2 of the apply), the *check* while the plan is still being
    /// built. Re-encoding a trailing chunk is exactly the case where a stored
    /// size can grow past the width, and a refusal raised in phase 2 would land
    /// after phase 1 had already written the chunk bytes — a leaked blob and a
    /// half-applied append. `plan_ea_append` calls this for every blob it is
    /// about to hand over, so the refusal happens before anything is written.
    fn check_stored_size(&self, os: usize, stored_size: u64) -> Result<(), Error> {
        let csz = self.element_size_width(os);
        if self.client_id != 0 && csz < 8 && stored_size >= (1u64 << (8 * csz)) {
            return Err(Error::AppendUnsupported(
                "recompressed chunk size exceeds the dataset's extensible-array element width",
            ));
        }
        Ok(())
    }

    /// Lay one Extensible-Array element into `buf`, returning the bytes used.
    /// Its fields are contiguous, so this is one span rather than the three
    /// writes it replaced.
    fn element_bytes(
        &self,
        buf: &mut [u8; MAX_EA_ELEM],
        os: usize,
        rec: ElemRecord,
    ) -> Result<usize, Error> {
        buf[..os].copy_from_slice(&rec.addr.get().to_le_bytes()[..os]);
        if self.client_id == 0 {
            return Ok(os);
        }
        self.check_stored_size(os, rec.stored_size)?;
        let csz = self.element_size_width(os);
        buf[os..os + csz].copy_from_slice(&rec.stored_size.to_le_bytes()[..csz]);
        buf[os + csz..os + csz + 4].copy_from_slice(&rec.filter_mask.to_le_bytes());
        Ok(os + csz + 4)
    }

    /// Change the index block and republish it, checksum included, as one write.
    fn publish_index_block<F: Store>(
        &self,
        file: &mut F,
        at: u64,
        value: &[u8],
    ) -> Result<(), Error> {
        let os = file.offset_size() as usize;
        let ib_prefix = (4 + 1 + 1 + os) as u64;
        let ndblk_addrs = self.geom.direct_dblk_nelmts.len();
        let nsblk_addrs = self.geom.nsblk_addrs;
        let cks_off = self.index_block_addr.get()
            + ib_prefix
            + self.idx_blk_elmts * self.ea_elem_size as u64
            + ((ndblk_addrs + nsblk_addrs) * os) as u64;
        file.publish_checksummed(self.index_block_addr.get(), cks_off, at, value)
    }

    /// Change a super block and republish it, checksum included, as one write.
    fn publish_super_block<F: Store>(
        &self,
        file: &mut F,
        sblk_addr: StoredAddress,
        sb: SuperBlockGeom,
        blk_off: usize,
        at: u64,
        value: &[u8],
    ) -> Result<(), Error> {
        let os = file.offset_size() as usize;
        let prefix = (4 + 1 + 1 + os + blk_off) as u64;
        let block_start = sblk_addr.get();
        let cks_off = block_start + prefix + sb.bitmap_size() + sb.ndblks * os as u64;
        file.publish_checksummed(block_start, cks_off, at, value)
    }

    /// Patch the six EA header statistics and recompute the header checksum.
    pub(crate) fn update_ea_header<F: Store>(
        &self,
        file: &mut F,
        num_chunks: u64,
    ) -> Result<(), Error> {
        let stats = ea_compute_stats(
            &self.geom,
            self.idx_blk_elmts,
            self.ea_elem_size,
            self.page_nelmts,
            file.offset_size(),
            self.blk_off_size,
            num_chunks,
            // This engine grows a rank-1 unlimited dataset (the parse above
            // refuses any other), and a rank-1 index numbers its chunks
            // 0..n with no gap to skip, so the occupancy is dense by
            // construction and this is the predicate the walk always had.
            crate::chunked_write::SlotOccupancy::Dense(num_chunks),
        );
        let ls = file.length_size() as usize;
        let ea_addr = self.ea_addr.get();
        // The six statistics are adjacent length-sized fields, so they are one
        // span and go out as one write. Six writes of eight bytes each into the
        // same forty-eight are six syscalls and, on flash, six chances to dirty
        // the same page (issue #288).
        // On the stack: this runs on every append, and the six fields are at most
        // forty-eight bytes, so a heap allocation per append would be one the
        // six separate writes never made.
        let mut stat_block = [0u8; 6 * 8];
        let mut at = 0;
        for value in [
            stats.nsuper_blks,
            stats.super_blk_size,
            stats.ndata_blks,
            stats.data_blk_size,
            stats.max_idx_set,
            stats.nelmts,
        ] {
            stat_block[at..at + ls].copy_from_slice(&value.to_le_bytes()[..ls]);
            at += ls;
        }
        let aehd_size =
            ExtensibleArrayHeader::serialized_size(file.offset_size(), file.length_size()) as u64;
        let cks_off = ea_addr + aehd_size - 4;
        file.publish_checksummed(ea_addr, cks_off, ea_addr + 12, &stat_block[..at])
    }

    /// Publish `new_dim` as the dataspace axis-0 dimension (the commit point) and
    /// recompute the containing object-header chunk's checksum.
    pub(crate) fn patch_dimension<F: Store>(
        &self,
        file: &mut F,
        new_dim: u64,
    ) -> Result<(), Error> {
        let ls = file.length_size() as usize;
        file.publish_checksummed(
            self.ohdr_chunk_start,
            self.ohdr_chunk_msg_end,
            self.dim0_off,
            &new_dim.to_le_bytes()[..ls],
        )
    }
}

/// The mutation-free result of planning one in-place Extensible-Array append: the
/// per-chunk (possibly compressed) blobs to write plus the
/// bookkeeping the write phase publishes.
pub(crate) struct AppendPlan {
    /// Per-chunk (possibly compressed) bytes to write, in element order starting
    /// at `n_full`. The first entry is the rewritten trailing chunk when the
    /// current length was not chunk-aligned.
    pub new_chunk_bytes: Vec<Vec<u8>>,
    /// Element index the first new chunk occupies.
    pub n_full: u64,
    /// New axis-0 dimension after the append (the commit value).
    pub new_dim: u64,
    /// New number of indexed chunks.
    pub new_num_chunks: u64,
}

/// Compute, without mutating the file, the new chunk blobs to write for an append
/// of `new_elems` elements (`raw` little-endian bytes) to the Extensible-Array
/// dataset described by `loc`, together with the first element index they occupy
/// and the new dimension / chunk count. Shared by the general append writer and
/// the in-place edit engine's in-place append so the read/plan logic lives in one place.
///
/// `grow_visible_tail` says whether the caller may rewrite the trailing chunk of
/// a *filtered* dataset — a chunk a reader at the old dimension can already see.
/// A `File::open_rw` session holds an exclusive lock and passes `true`; the SWMR
/// writer, whose readers are the point, passes `false`. It has no bearing on an
/// unfiltered dataset, whose trailing chunk is raw and grows by relocation
/// either way, and none on a *lossy* pipeline, which is refused here whatever the
/// caller asks for.
pub(crate) fn plan_ea_append<F: Store>(
    file: &F,
    loc: &Located,
    datatype: &Datatype,
    spatial: &[u64],
    element_size: NonZeroUsize,
    pipeline: Option<&FilterPipeline>,
    grow_visible_tail: bool,
    raw: &[u8],
    new_elems: u64,
    fill: FillPattern<'_>,
) -> Result<AppendPlan, Error> {
    let chunk_elems = loc.chunk_elems;
    let current_dim = loc.current_dim;
    let new_dim = current_dim
        .checked_add(new_elems)
        .ok_or(Error::AppendUnsupported(
            "append would overflow the dataset dimension",
        ))?;
    let n_full = current_dim / chunk_elems;
    let has_partial = current_dim % chunk_elems != 0;

    // Growing a partial trailing chunk is copy-on-write, filtered or not. The
    // rewritten chunk is written to a *fresh* allocation in phase 1 and the old
    // slot is never touched, so the only mutation of already-visible state is
    // phase 2's repoint of element `n_full`. That element write is the same
    // write `ea_insert` makes for a brand-new element in a block that already
    // holds visible ones: since issue #307 every element goes out through
    // `Store::publish_checksummed`, one write running from the changed byte
    // through the checksum that covers it, so a filtered element's three fields
    // and their checksum are as atomic as an unfiltered element's one address.
    //
    // A crash between the phases leaves a readable prefix. After phase 1 nothing
    // points at the new chunk. After phase 2 a reader still bounded by the old
    // dimension decodes the new chunk instead of the old one, and
    // `split_into_chunks` padded it to a whole chunk with the dataset's fill
    // value, so it yields the same element count with the old live prefix
    // byte-identical — which holds because the pipelines this admits are
    // lossless, the condition the next paragraph enforces. Phases 3 and 4 are the
    // ordinary in-progress states.
    //
    // Reusing the old slot when the re-encoded bytes happen to fit is the option
    // this does *not* take: overwriting a chunk in place is not one atomic write,
    // and a reader crossing it would decode a mix of two encodings.
    //
    // What copy-on-write does *not* buy is a lossy filter. Rewriting the tail
    // decodes the committed chunk and encodes it again, and for ZFP fixed-rate or
    // float D-scale scale-offset that round trip is not the identity: the chunk's
    // new neighbours move the quantization, so values a reader has already seen
    // come back different. Measured on a chunk of 4 f64 at ZFP rate 8, growing
    // `[4.5, 5.5]` by two elements read the committed pair back as `[16.0, 0.0]`.
    // Refused for every caller, `grow_visible_tail` or not.
    //
    // The SWMR writer passes `grow_visible_tail = false`. In production that is
    // defence in depth rather than a reachable message: `append_inplace_gathered`
    // refuses every filtered SWMR append before it gets here. It is stated as a
    // parameter anyway because it is this function that knows *why* the rewrite
    // is safe, and the reason — an exclusive lock, no reader mid-sequence — is
    // the caller's property, not this function's.
    //
    // The appended *length* has never been constrained. An unaligned length only
    // makes the last chunk this append writes a partial one, and that chunk's
    // index element is a fresh insert past the old dimension — invisible until
    // phase 4 publishes the new dimension, exactly like every whole chunk beside
    // it.
    if has_partial && let Some(pl) = pipeline {
        if !grow_visible_tail {
            return Err(Error::AppendUnsupported(
                "a filtered dataset whose length is not a whole multiple of the chunk length \
                 cannot be appended in place by this writer: growing its trailing partial chunk \
                 would repoint an index element a concurrent reader can already see. Use \
                 Dataset::append_staged, or append whole chunks",
            ));
        }
        if !pipeline_lossless(pl) {
            return Err(Error::AppendUnsupported(LOSSY_TAIL_REFUSAL));
        }
    }

    // Build the raw tail region: the live prefix of any rewritten partial chunk,
    // then the appended bytes.
    let mut tail_raw: Vec<u8> = Vec::new();
    if has_partial {
        let rec = loc
            .read_element(file, n_full)?
            .ok_or(Error::AppendUnsupported(
                "trailing partial chunk is missing from the index",
            ))?;
        let stored_len = if pipeline.is_some() {
            rec.stored_size
                .narrow_or_else(|| Error::AppendUnsupported("chunk size exceeds this platform"))?
        } else {
            chunk_elems.to_usize()? * element_size.get()
        };
        rec.addr
            .get()
            .checked_add(stored_len as u64)
            .filter(|&e| e <= file.len())
            .ok_or(Error::AppendUnsupported(
                "trailing chunk extends past end-of-file",
            ))?;
        // One bounded read of the single trailing chunk.
        let stored = file.read_exact_at(rec.addr.get(), stored_len)?;
        let full = if let Some(pl) = pipeline {
            let ctx = ChunkContext::from_datatype(spatial, datatype)?;
            decompress_chunk(&stored, pl, ctx, rec.filter_mask).map_err(Error::Format)?
        } else {
            stored
        };
        let live_elems: usize = (current_dim % chunk_elems)
            .narrow_or_else(|| Error::AppendUnsupported("chunk length exceeds this platform"))?;
        let live_bytes = live_elems * element_size.get();
        if full.len() < live_bytes {
            return Err(Error::AppendUnsupported(
                "trailing chunk decoded shorter than its live element count",
            ));
        }
        tail_raw.extend_from_slice(&full[..live_bytes]);
    }
    tail_raw.extend_from_slice(raw);

    // Split the tail into full chunk buffers and compress each through the
    // pipeline when filtered. The final chunk's slots past the new dimension
    // take the dataset's fill value (issue #296).
    let tail_len_elems = new_dim - n_full * chunk_elems;
    let split = split_into_chunks(&tail_raw, &[tail_len_elems], spatial, element_size, fill)
        .map_err(Error::Format)?;
    let new_chunk_bytes: Vec<Vec<u8>> = if let Some(pl) = pipeline {
        let ctx = ChunkContext::from_datatype(spatial, datatype)?;
        let mut out = Vec::with_capacity(split.len());
        // One encoder across the appended tail; see `FilterScratch`.
        let mut scratch = FilterScratch::new();
        for buf in &split {
            out.push(compress_chunk_with(&mut scratch, buf, pl, ctx).map_err(Error::Format)?);
        }
        out
    } else {
        split
    };

    // Every blob has to fit the element it will be recorded in. Checked here,
    // while nothing has been written, rather than where the element is laid out
    // in phase 2 — by then phase 1 has already placed the chunk bytes, and the
    // refusal would leave them behind unreferenced.
    let os = file.offset_size() as usize;
    for blob in &new_chunk_bytes {
        loc.check_stored_size(os, blob.len() as u64)?;
    }

    let new_num_chunks = n_full + new_chunk_bytes.len() as u64;
    Ok(AppendPlan {
        new_chunk_bytes,
        n_full,
        new_dim,
        new_num_chunks,
    })
}

/// Apply a planned append to `file` in place, ordered child-before-parent with
/// `fsync` barriers so a crash between calls leaves either the old length or the
/// new one, never a torn or lost view. `max_phase` runs only the first N of the
/// four durability phases (production callers pass 4; the crash-consistency tests
/// stop at a boundary to simulate a crash). On a full (phase-4) apply, `loc`'s
/// cached `current_dim` / `num_chunks` are advanced to match. Shared by the
/// general append writer and the in-place edit engine's in-place append so this ordered
/// write sequence — the crash-safety heart of the engine — lives in exactly one
/// place and is never copy-pasted.
pub(crate) fn apply_ea_append<F: Store>(
    file: &mut F,
    loc: &mut Located,
    plan: &AppendPlan,
    max_phase: u8,
) -> Result<(), Error> {
    // Phase 1: new/relocated chunk bytes, then advance the superblock's recorded
    // end-of-file to cover them. This must precede the index writes: the trailing
    // partial chunk's element is *visible* at the old dimension, so once it is
    // repointed to the relocated chunk that chunk must already lie within the
    // recorded EOF.
    //
    // A chunk served from freed space (issue #349) is inside the recorded EOF
    // already, so it needs no patch, and issuing one would rewrite the whole
    // superblock and barrier for it on every batch. Both patches below are
    // therefore the same rule: rewrite the superblock only where the file grew
    // past what it records. `recorded_eof` is that value throughout — the
    // engine's own reads of `file.len()` are what maintain it, since this trait
    // exposes no accessor for the superblock field itself.
    let mut recorded_eof = file.len();
    let mut chunk_addrs = Vec::with_capacity(plan.new_chunk_bytes.len());
    for blob in &plan.new_chunk_bytes {
        chunk_addrs.push((StoredAddress::new(file.alloc_raw(blob)?), blob.len() as u64));
    }
    file.sync()?;
    if file.len() != recorded_eof {
        file.patch_superblock_eof()?;
        file.sync()?;
        recorded_eof = file.len();
    }
    if max_phase < 2 {
        return Ok(());
    }

    // Phase 2: the index element writes — a fresh insert for each new chunk, or an
    // in-place repoint of the trailing element (which only ever points at data
    // whose live prefix reproduces the old view's bytes). This may allocate new EA
    // blocks past EOF, covered by the phase-3 EOF patch (or draw them from freed
    // space, in which case there is nothing for it to cover).
    for (k, &(addr, stored_size)) in chunk_addrs.iter().enumerate() {
        let e = plan.n_full + k as u64;
        let rec = ElemRecord {
            addr,
            stored_size,
            filter_mask: 0,
        };
        loc.ea_insert(file, e, rec)?;
    }
    file.sync()?;
    if max_phase < 3 {
        return Ok(());
    }

    // Phase 3: cover any EA blocks allocated during the element writes, then
    // publish the EA header element count.
    if file.len() != recorded_eof {
        file.patch_superblock_eof()?;
    }
    loc.update_ea_header(file, plan.new_num_chunks)?;
    file.sync()?;
    if max_phase < 4 {
        return Ok(());
    }

    // Phase 4: publish the dataspace dimension — the single commit point.
    loc.patch_dimension(file, plan.new_dim)?;
    file.sync()?;

    loc.current_dim = plan.new_dim;
    loc.num_chunks = plan.new_num_chunks;
    Ok(())
}

/// The byte of a super block's page-init bitmap that holds page `global_page`'s
/// bit (MSB-first), and that byte with the bit set.
///
/// A read-modify-write, so it reads the byte and hands the *result* back for the
/// publish to place: one field of one byte is not a value the caller can name on
/// its own.
fn sb_page_bit<F: Store>(
    file: &F,
    sblk_addr: StoredAddress,
    blk_off: usize,
    global_page: u64,
) -> Result<(u64, u8), Error> {
    let os = file.offset_size() as usize;
    let bitmap_start = sblk_addr.get() + (4 + 1 + 1 + os + blk_off) as u64;
    let byte = bitmap_start + global_page / 8;
    let mut v = [0u8; 1];
    file.read_at(byte, &mut v)?;
    Ok((byte, v[0] | (0x80u8 >> (global_page % 8))))
}

/// File offset of the `dblk_local`-th data-block-address slot inside a super
/// block, accounting for the page-init bitmap when the block is paged.
fn sb_dblk_slot_off(
    os: usize,
    sblk_addr: StoredAddress,
    dblk_local: usize,
    sb: SuperBlockGeom,
    blk_off: usize,
) -> u64 {
    let prefix = (4 + 1 + 1 + os + blk_off) as u64;
    sblk_addr.get() + prefix + sb.bitmap_size() + (dblk_local * os) as u64
}

// ---------------------------------------------------------------------------
// Extensible-array element location (where element `e` lives in the structure)
// ---------------------------------------------------------------------------

enum Parent {
    /// Reached by a direct data-block pointer in the index block.
    IndexDirect { ordinal: usize },
    /// Reached via a super block (`sblk_j`-th super-block pointer), as the
    /// `dblk_local`-th data block within it.
    Super { sblk_j: usize, dblk_local: usize },
}

struct DataBlockLoc {
    db_start: u64,
    dblk_nelmts: u64,
    ndblks: u64,
    sb_block_offset: u64,
    parent: Parent,
}

impl DataBlockLoc {
    /// The shape of the super block this data block belongs to, in an array
    /// whose pages hold `page_nelmts` elements.
    ///
    /// Meaningful for a [`Parent::Super`] block; for a [`Parent::IndexDirect`]
    /// one the counts describe the single block itself and no `EASB` exists to
    /// apply them to.
    fn super_block(&self, page_nelmts: u64) -> SuperBlockGeom {
        SuperBlockGeom {
            ndblks: self.ndblks,
            blocks: self.blocks(page_nelmts),
        }
    }

    /// How this data block is paged, which holds whether or not it has a super
    /// block above it.
    fn blocks(&self, page_nelmts: u64) -> DataBlockGeom {
        DataBlockGeom {
            dblk_nelmts: self.dblk_nelmts,
            page_nelmts,
        }
    }
}

/// Locate the data block containing element `e` (which is `>= idx_blk_elmts`).
fn locate_data_block(geom: &EaGeometry, idx_blk_elmts: u64, e: u64) -> DataBlockLoc {
    let mut elem = idx_blk_elmts;
    for (ordinal, &dn) in geom.direct_dblk_nelmts.iter().enumerate() {
        if e < elem + dn {
            return DataBlockLoc {
                db_start: elem,
                dblk_nelmts: dn,
                ndblks: 1,
                sb_block_offset: 0,
                parent: Parent::IndexDirect { ordinal },
            };
        }
        elem += dn;
    }
    for j in 0..geom.nsblk_addrs {
        let (ndblks, dn) = geom.sblks[geom.first_indirect_sblk + j];
        let span = ndblks * dn;
        if e < elem + span {
            let sb_block_offset = elem - idx_blk_elmts;
            let within = e - elem;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "within/dn is a data-block index bounded by ndblks (small)"
            )]
            let dblk_local = (within / dn) as usize;
            let db_start = elem + dblk_local as u64 * dn;
            return DataBlockLoc {
                db_start,
                dblk_nelmts: dn,
                ndblks,
                sb_block_offset,
                parent: Parent::Super {
                    sblk_j: j,
                    dblk_local,
                },
            };
        }
        elem += span;
    }
    DataBlockLoc {
        db_start: e,
        dblk_nelmts: u64::MAX,
        ndblks: 0,
        sb_block_offset: 0,
        parent: Parent::IndexDirect { ordinal: 0 },
    }
}

// ---------------------------------------------------------------------------
// Object-header walk that records message file offsets + chunk checksum regions
// ---------------------------------------------------------------------------

struct WalkedMessage {
    msg_type: MessageType,
    /// The record's flags. With [`MessageFlags::SHARED`] set, the body is a reference
    /// to a shared message, which every parse below would otherwise decode as content.
    flags: MessageFlags,
    /// Absolute file offset of the message body.
    data_off: u64,
    size: usize,
    /// Containing chunk's checksum coverage: `[chunk_start, chunk_msg_end)`, with
    /// the 4-byte checksum stored at `chunk_msg_end`.
    chunk_start: u64,
    chunk_msg_end: u64,
}

struct Walk {
    messages: Vec<WalkedMessage>,
}

/// Walk a version-2 object header (chunk 0 plus any continuation chunks),
/// recording each message's absolute data offset and its containing chunk's
/// checksum region. Reads one bounded window per header chunk from `source`, so
/// the walk works over a store with no whole-file mirror.
fn walk_v2_object_header<S: Source + ?Sized>(
    source: &S,
    offset: u64,
    offset_size: u8,
    length_size: u8,
) -> Result<Walk, Error> {
    let head = match source.read_metadata_at(offset, 6) {
        Ok(head) => head,
        // A header running past end-of-file reads as a missing signature, like
        // the whole-buffer walk before it; other backend errors pass through.
        Err(FormatError::UnexpectedEof { .. }) => {
            return Err(Error::Format(FormatError::InvalidObjectHeaderSignature));
        }
        Err(e) => return Err(Error::Format(e)),
    };
    if &head[..4] != b"OHDR" {
        return Err(Error::Format(FormatError::InvalidObjectHeaderSignature));
    }
    let flags = head[5];
    let mut pos = offset + 6;
    if flags & 0x20 != 0 {
        pos += 16; // timestamps
    }
    if flags & 0x10 != 0 {
        pos += 4; // attr storage phase-change
    }
    let chunk_size_width = 1usize << (flags & 0x03);
    let size_buf = source.read_metadata_at(pos, chunk_size_width)?;
    let chunk0_size = read_uint(&size_buf, 0, chunk_size_width)?.to_usize()?;
    pos += chunk_size_width as u64;
    let chunk0_start = offset;
    let chunk0_msg_start = pos;
    let chunk0_msg_end = chunk0_msg_start + chunk0_size as u64;

    let has_creation_order = flags & 0x04 != 0;
    let mut messages = Vec::new();
    let mut continuations: Vec<(u64, usize)> = Vec::new();

    let chunk0 = source.read_metadata_at(chunk0_msg_start, chunk0_size)?;
    walk_messages(
        &chunk0,
        chunk0_msg_start,
        chunk0_start,
        chunk0_msg_end,
        has_creation_order,
        offset_size,
        length_size,
        &mut messages,
        &mut continuations,
    )?;

    let mut guard = 256;
    while let Some((cont_off, cont_len)) = continuations.pop() {
        guard -= 1;
        if guard == 0 {
            return Err(Error::Format(FormatError::NestingDepthExceeded));
        }
        // A continuation chunk is `OCHK` + messages + a trailing 4-byte
        // checksum, so anything shorter than 8 bytes is malformed.
        if cont_len < 8 {
            return Err(Error::Format(FormatError::InvalidObjectHeaderSignature));
        }
        let chunk = source.read_metadata_at(cont_off, cont_len)?;
        if &chunk[..4] != b"OCHK" {
            return Err(Error::Format(FormatError::InvalidObjectHeaderSignature));
        }
        let msg_start = cont_off + 4;
        let msg_end = cont_off + (cont_len - 4) as u64; // checksum is the last 4 bytes
        walk_messages(
            &chunk[4..cont_len - 4],
            msg_start,
            cont_off,
            msg_end,
            has_creation_order,
            offset_size,
            length_size,
            &mut messages,
            &mut continuations,
        )?;
    }

    Ok(Walk { messages })
}

/// Scan one object-header chunk's message region. `chunk` holds exactly the
/// message bytes and `base` is the absolute file offset of `chunk[0]`, so
/// recorded message offsets are absolute.
#[allow(clippy::too_many_arguments)]
fn walk_messages(
    chunk: &[u8],
    base: u64,
    chunk_start: u64,
    chunk_msg_end: u64,
    has_creation_order: bool,
    offset_size: u8,
    length_size: u8,
    messages: &mut Vec<WalkedMessage>,
    continuations: &mut Vec<(u64, usize)>,
) -> Result<(), Error> {
    let msg_header_size = if has_creation_order { 6 } else { 4 };
    let end = chunk.len();
    let mut pos = 0usize;
    while pos + msg_header_size <= end {
        let msg_type_raw = chunk[pos] as u16;
        let msg_data_size = u16::from_le_bytes([chunk[pos + 1], chunk[pos + 2]]) as usize;
        let msg_flags = MessageFlags::new(chunk[pos + 3]);
        pos += msg_header_size;
        if pos + msg_data_size > end {
            break; // padding
        }
        let msg_type = MessageType::from_u16(msg_type_raw);
        if msg_type == MessageType::ObjectHeaderContinuation {
            let cont_off = read_uint(chunk, pos, offset_size as usize)?;
            let cont_len =
                read_uint(chunk, pos + offset_size as usize, length_size as usize)?.to_usize()?;
            continuations.push((cont_off, cont_len));
        } else {
            messages.push(WalkedMessage {
                msg_type,
                flags: msg_flags,
                data_off: base + pos as u64,
                size: msg_data_size,
                chunk_start,
                chunk_msg_end,
            });
        }
        pos += msg_data_size;
    }
    Ok(())
}

fn read_uint(data: &[u8], pos: usize, size: usize) -> Result<u64, Error> {
    if pos + size > data.len() {
        return Err(Error::Format(FormatError::UnexpectedEof {
            expected: pos + size,
            available: data.len(),
        }));
    }
    let mut v = 0u64;
    for i in 0..size {
        v |= (data[pos + i] as u64) << (8 * i);
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_mode::AccessMode;
    use crate::group_v2;
    use crate::object_path::ObjectPath;
    use crate::signature;
    use crate::source::BytesSource;
    use crate::superblock::Superblock;
    use crate::writer::FileBuilder;
    use std::cell::{Cell, RefCell};

    /// An in-test [`Store`] over a `Vec<u8>` that records every read window, so
    /// the tests below can assert the engine's bounded-memory contract: the
    /// append path never reads more than a bounded window at once, no matter how
    /// large the file is. This is the seam contract issue #147's mirror-less
    /// bounded backend relies on.
    struct WindowProbeStore {
        data: Vec<u8>,
        superblock: Superblock,
        sb_sig_off: usize,
        max_read: Cell<usize>,
        total_read: Cell<usize>,
        reads: RefCell<Vec<(u64, usize)>>,
        /// How many times the append sequence rewrote the superblock to advance
        /// its recorded end-of-file. Two of the four durability phases can, and
        /// only one of them always must.
        superblock_patches: Cell<usize>,
        /// Barriers the append sequence issued. Every one is a point where the
        /// writes before it must reach the disk before the writes after it.
        syncs: Cell<usize>,
        /// Blobs appended at end-of-file: the chunk, plus one per fresh index
        /// block. Each fresh block owes a barrier before the pointer naming it.
        appends: Cell<usize>,
        /// A single below-end-of-file region nothing points at, stood up by
        /// [`with_free_tail`](Self::with_free_tail), which `alloc_raw` hands out
        /// before it will append (issue #349). Stands in for a session's free
        /// list; the engine sees only the address that comes back.
        free: Option<(u64, u64)>,
        /// Allocations `alloc_raw` served from that region rather than by
        /// extending the file.
        reuses: Cell<usize>,
    }

    impl WindowProbeStore {
        fn open(data: Vec<u8>) -> Self {
            let sb_sig_off = signature::find_signature(&data).unwrap();
            let superblock = Superblock::parse(&data, sb_sig_off).unwrap();
            Self {
                data,
                superblock,
                sb_sig_off,
                max_read: Cell::new(0),
                total_read: Cell::new(0),
                reads: RefCell::new(Vec::new()),
                superblock_patches: Cell::new(0),
                syncs: Cell::new(0),
                appends: Cell::new(0),
                free: None,
                reuses: Cell::new(0),
            }
        }

        /// Give the store a `len`-byte free region, by growing the image and
        /// recording the new end-of-file in the superblock before any counting
        /// starts. That gives the region the two properties the engine can
        /// observe — inside the recorded end-of-file, and named by nothing.
        ///
        /// It sits at the tail, where a *committed* deletion never leaves one
        /// (a free run reaching end-of-file is truncated away instead, which is
        /// why the integration tests plant a ceiling object above theirs). The
        /// engine cannot tell the difference: it reads an address and a length.
        fn with_free_tail(mut self, len: u64) -> Self {
            let addr = self.data.len() as u64;
            self.data
                .resize(self.data.len() + len.to_usize().unwrap(), 0);
            self.patch_superblock_eof().unwrap();
            self.free = Some((addr, len));
            self
        }

        /// Extend the image, the fallback `alloc_raw` takes once the free region
        /// (if any) cannot serve an allocation.
        fn append_bytes(&mut self, bytes: &[u8]) -> Result<u64, Error> {
            self.appends.set(self.appends.get() + 1);
            let addr = self.data.len() as u64;
            self.data.extend_from_slice(bytes);
            Ok(addr)
        }

        fn reset_counters(&self) {
            self.max_read.set(0);
            self.total_read.set(0);
            self.reads.borrow_mut().clear();
            self.superblock_patches.set(0);
            self.syncs.set(0);
            self.appends.set(0);
            self.reuses.set(0);
        }
    }

    impl Source for WindowProbeStore {
        fn len(&self) -> u64 {
            self.data.len() as u64
        }
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
            self.max_read.set(self.max_read.get().max(buf.len()));
            self.total_read.set(self.total_read.get() + buf.len());
            self.reads.borrow_mut().push((offset, buf.len()));
            BytesSource::new(&self.data).read_at(offset, buf)
        }
    }

    impl Store for WindowProbeStore {
        fn offset_size(&self) -> u8 {
            self.superblock.offset_size
        }
        fn length_size(&self) -> u8 {
            self.superblock.length_size
        }
        fn alloc_raw(&mut self, bytes: &[u8]) -> Result<u64, Error> {
            let len = bytes.len() as u64;
            if let Some((addr, avail)) = self.free.filter(|&(_, avail)| avail >= len) {
                self.reuses.set(self.reuses.get() + 1);
                self.free = (avail > len).then(|| (addr + len, avail - len));
                self.write_at(addr, bytes)?;
                return Ok(addr);
            }
            self.append_bytes(bytes)
        }
        fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
            let offset = offset.to_usize()?;
            self.data[offset..offset + bytes.len()].copy_from_slice(bytes);
            Ok(())
        }
        fn patch_superblock_eof(&mut self) -> Result<(), Error> {
            self.superblock_patches
                .set(self.superblock_patches.get() + 1);
            self.superblock.eof_address = self.data.len() as u64;
            let bytes = self.superblock.serialize();
            let off = self.sb_sig_off;
            self.data[off..off + bytes.len()].copy_from_slice(&bytes);
            Ok(())
        }
        fn sync(&mut self) -> Result<(), Error> {
            self.syncs.set(self.syncs.get() + 1);
            Ok(())
        }
    }

    /// Build an in-memory latest-format file with one unlimited chunked i32
    /// dataset seeded with `0..n`, returning its bytes.
    fn build_unlimited(n: i32, chunk: u64) -> Vec<u8> {
        let data: Vec<i32> = (0..n).collect();
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&data)
            .with_shape(&[n as u64])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[chunk]);
        b.finish().unwrap()
    }

    /// Drive the full located-append path (`locate_at` + `plan_ea_append` +
    /// `apply_ea_append`) through the probe store and return the located state.
    fn locate(store: &WindowProbeStore) -> (Located, crate::datatype::Datatype) {
        let oh_addr = group_v2::resolve_path_any(
            &store.data,
            AccessMode::ReadOnly,
            &store.superblock,
            &ObjectPath::parse("d"),
        )
        .unwrap();
        let result = Located::locate_at(store, oh_addr, Error::AppendUnsupported).unwrap();
        let (dt_off, dt_size) = result.spans.datatype;
        let dt_bytes = store.read_metadata_at(dt_off, dt_size).unwrap();
        let (datatype, _) = crate::datatype::Datatype::parse(&dt_bytes).unwrap();
        (result.located, datatype)
    }

    /// Every element of the located dataset, read back through the engine's own
    /// index walk: one unfiltered chunk of `chunk_elems` i32 per indexed slot.
    fn read_i32s(store: &WindowProbeStore, loc: &Located) -> Vec<i32> {
        let width = loc.chunk_elems.to_usize().unwrap() * 4;
        let mut out = Vec::new();
        for e in 0..loc.num_chunks {
            let rec = loc.read_element(store, e).unwrap().expect("indexed chunk");
            let bytes = store.read_exact_at(rec.addr.get(), width).unwrap();
            out.extend(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .copied()
                    .map(i32::from_le_bytes),
            );
        }
        out
    }

    fn append_i32s(
        store: &mut WindowProbeStore,
        loc: &mut Located,
        datatype: &crate::datatype::Datatype,
        values: std::ops::Range<i32>,
    ) {
        let raw: Vec<u8> = values.clone().flat_map(|v| v.to_le_bytes()).collect();
        let new_elems = (values.end - values.start) as u64;
        let spatial = vec![loc.chunk_elems];
        let plan = plan_ea_append(
            store,
            loc,
            datatype,
            &spatial,
            loc.elem_bytes,
            None,
            true,
            &raw,
            new_elems,
            FillPattern::ZERO,
        )
        .unwrap();
        apply_ea_append(store, loc, &plan, 4).unwrap();
    }

    /// A fresh Extensible-Array block is separated from the pointer that names
    /// it by a barrier, so the block is on the disk before anything reaches it.
    ///
    /// The block's bytes are allocated fresh; the slot that points at it lives in
    /// the index block or a super block, near the front of the file. Left in one
    /// barrier-free window, an image that gathers its writes issues them in
    /// *address* order — the pointer first — and a failure in between leaves an
    /// index block whose checksum validates and whose pointer names bytes past
    /// the end of the file. The next append then reads that pointer, believes the
    /// block exists, and writes into nothing (issue #288).
    ///
    /// Stated as a count because the barrier is what the property *is*: the
    /// appends that allocate a block must cost strictly more barriers than the
    /// ones that do not. Both directions are asserted — a run with no allocating
    /// append would satisfy "more" vacuously, and one where every append
    /// allocated would satisfy it without separating anything.
    ///
    /// 320 appends at one chunk each, because a shorter run only ever allocates
    /// *direct* data blocks: the first super block, which has the same hazard
    /// through a different call path, arrives past 300.
    #[test]
    fn allocating_an_index_block_barriers_before_the_pointer_that_names_it() {
        let mut store = WindowProbeStore::open(build_unlimited(4, 1));
        let (mut loc, datatype) = locate(&store);

        // (barriers, blobs appended at end-of-file) for each append.
        let mut rounds: Vec<(usize, usize)> = Vec::new();
        for i in 0..320i32 {
            store.reset_counters();
            append_i32s(&mut store, &mut loc, &datatype, (4 + i)..(5 + i));
            rounds.push((store.syncs.get(), store.appends.get()));
        }

        // A plain append writes only its chunk at end-of-file.
        let base = rounds
            .iter()
            .find(|&&(_, appends)| appends == 1)
            .map(|&(syncs, _)| syncs)
            .expect("some append allocates no index block");
        let blocks = |appends: usize| appends - 1;
        assert!(
            rounds.iter().any(|&(_, appends)| blocks(appends) == 1),
            "the run must allocate a data block somewhere: {rounds:?}"
        );
        assert!(
            rounds.iter().any(|&(_, appends)| blocks(appends) >= 2),
            "the run must reach a super block, which allocates two blocks in one \
             append, or the second barrier site is never exercised: {rounds:?}"
        );
        for (round, &(syncs, appends)) in rounds.iter().enumerate() {
            assert_eq!(
                syncs,
                base + blocks(appends),
                "append {round} allocated {} index block(s) and took {syncs} \
                 barriers against the {base} a plain append takes: each fresh \
                 block owes one, between its bytes and the pointer naming it",
                blocks(appends)
            );
        }
    }

    /// The append sequence rewrites the superblock **once** when its index
    /// writes allocate nothing, and twice when they allocate a block past the
    /// end-of-file phase 1 recorded.
    ///
    /// Phase 1's patch fires here because the chunk bytes extend the file; when
    /// they are served from freed space instead they do not, which
    /// [`phase_one_patches_the_superblock_only_when_the_file_grew`] covers.
    /// Phase 3's exists only to cover blocks the element writes added, and an
    /// append that adds none used to rewrite the whole superblock with the bytes
    /// it already held: one wasted write, and one wasted page dirtying, on every
    /// append (issue #288).
    ///
    /// Asserted in both directions on purpose. "Never twice" would be satisfied
    /// by dropping phase 3's patch altogether, which understates the end-of-file
    /// on exactly the appends that grow the index — so this also demands that
    /// some append in the run *does* patch twice, and the sequence below is long
    /// enough to cross a block boundary and produce one.
    #[test]
    fn phase_three_patches_the_superblock_only_when_the_index_grew() {
        let mut store = WindowProbeStore::open(build_unlimited(4, 1));
        let (mut loc, datatype) = locate(&store);

        let mut patches = Vec::new();
        for i in 0..40i32 {
            store.reset_counters();
            append_i32s(&mut store, &mut loc, &datatype, (4 + i)..(5 + i));
            patches.push(store.superblock_patches.get());
        }

        assert!(
            patches.iter().all(|&p| p >= 1),
            "every append extends the file, so every one must record the new \
             end-of-file at least once: {patches:?}"
        );
        assert!(
            patches.contains(&1),
            "an append that allocates no index block must not rewrite the \
             superblock a second time: {patches:?}"
        );
        assert!(
            patches.contains(&2),
            "an append that does allocate one must, or the file would advertise \
             an end-of-file short of its own index: {patches:?}"
        );
        assert!(
            patches.iter().all(|&p| p <= 2),
            "the sequence defines exactly two such points: {patches:?}"
        );
    }

    /// An append whose chunks come out of freed space does not extend the file,
    /// so phase 1 must not rewrite the superblock: the end-of-file it would
    /// record is the one already there (issue #349).
    ///
    /// The reuse itself is the other half of the assertion. Dropping the patch
    /// while still appending would understate the end-of-file on every append —
    /// so this demands the allocation actually came from the free region, and
    /// that the appends resume, with the patch, once that region is spent.
    #[test]
    fn phase_one_patches_the_superblock_only_when_the_file_grew() {
        // A chunk of 4 i32 is 16 bytes; 64 bytes of hole holds four of them.
        let mut store = WindowProbeStore::open(build_unlimited(4, 4)).with_free_tail(64);
        let (mut loc, datatype) = locate(&store);

        let mut rounds = Vec::new();
        for i in 0..6i32 {
            store.reset_counters();
            let base = 4 + i * 4;
            append_i32s(&mut store, &mut loc, &datatype, base..(base + 4));
            rounds.push((
                store.reuses.get(),
                store.appends.get(),
                store.superblock_patches.get(),
            ));
        }

        for &(reuses, appends, patches) in &rounds {
            assert_eq!(
                patches == 0,
                appends == 0,
                "the superblock is rewritten exactly when the file grew: {rounds:?}"
            );
            assert!(reuses + appends >= 1, "every round allocates: {rounds:?}");
        }
        assert!(
            rounds.iter().any(|&(_, appends, _)| appends == 0),
            "the first rounds fit the hole, so they must allocate nothing at \
             end-of-file: {rounds:?}"
        );
        assert!(
            rounds.iter().any(|&(_, appends, _)| appends > 0),
            "the hole is spent well before the last round: {rounds:?}"
        );

        // The reused chunks are the dataset's, not scribble.
        assert_eq!(read_i32s(&store, &loc), (0..28).collect::<Vec<i32>>());
    }

    /// **Every** allocation the append engine makes is served from freed space
    /// when a region fits — the extensible-array blocks that index the chunks as
    /// much as the chunk bytes themselves (issue #349).
    ///
    /// Both are raw by this crate's placement convention, which is what lets one
    /// pool serve them: see [`Store::alloc_raw`]. A run given a hole larger than
    /// everything it will allocate must therefore extend the file by nothing at
    /// all, across a sequence long enough to grow the index.
    #[test]
    fn a_hole_serves_the_index_blocks_as_well_as_the_chunks() {
        let mut store = WindowProbeStore::open(build_unlimited(4, 1)).with_free_tail(16 * 1024);
        let (mut loc, datatype) = locate(&store);

        let mut rounds = Vec::new();
        for i in 0..40i32 {
            store.reset_counters();
            append_i32s(&mut store, &mut loc, &datatype, (4 + i)..(5 + i));
            rounds.push((store.reuses.get(), store.appends.get()));
        }

        assert!(
            rounds.iter().all(|&(_, appends)| appends == 0),
            "the hole is larger than the whole run, so nothing may reach \
             end-of-file: {rounds:?}"
        );
        assert!(
            rounds.iter().any(|&(reuses, _)| reuses >= 2),
            "some append must allocate an index block beside its chunk, or this \
             says nothing about the blocks: {rounds:?}"
        );
        assert_eq!(
            read_i32s(&store, &loc),
            (0..44).collect::<Vec<i32>>(),
            "the reused regions must hold the dataset's elements"
        );
    }

    /// Relocating a *partial* trailing chunk into freed space is sound: the
    /// element that names it is visible at the old dimension, and phase 2
    /// repoints it to a region whose live prefix reproduces the bytes it already
    /// showed (issue #349).
    ///
    /// This is the one append shape that overwrites an element a reader can
    /// already see, so it is worth exercising against reuse specifically: the
    /// rounds below that grow a partial chunk move it to a fresh address inside
    /// the hole, and the dataset must read back whole every time.
    #[test]
    fn a_relocated_partial_tail_may_land_in_freed_space() {
        const HOLE: u64 = 1024;
        let mut store = WindowProbeStore::open(build_unlimited(4, 4)).with_free_tail(HOLE);
        let hole = (store.len() - HOLE)..store.len();
        let (mut loc, datatype) = locate(&store);

        // (chunks indexed, address of the trailing chunk) after each append.
        let mut tails = Vec::new();
        for i in 4..12i32 {
            append_i32s(&mut store, &mut loc, &datatype, i..(i + 1));
            let tail = loc.num_chunks - 1;
            let addr = loc.read_element(&store, tail).unwrap().unwrap().addr.get();
            tails.push((loc.num_chunks, addr));
            assert_eq!(
                read_i32s(&store, &loc)[..(i as usize + 1)],
                (0..=i).collect::<Vec<i32>>()[..],
                "after appending {i} the live elements must all read back"
            );
        }

        // Relocation means the *same* chunk moved. Comparing consecutive
        // addresses would not say that: the probe allocator never repeats an
        // address, so a run that only ever added fresh chunks would satisfy it.
        assert!(
            tails
                .windows(2)
                .any(|w| w[0].0 == w[1].0 && w[0].1 != w[1].1),
            "a partial trailing chunk must have been rewritten to a new address \
             at least once: {tails:?}"
        );
        assert!(
            tails.iter().all(|&(_, addr)| hole.contains(&addr)),
            "every tail chunk must have landed in the freed region, or this says \
             nothing about reuse: {tails:?}"
        );
    }

    /// The engine's bounded-read contract: appends against a file far larger
    /// than any metadata structure never read more than a bounded window at
    /// once, and never the whole file.
    #[test]
    fn append_reads_stay_bounded_windows() {
        // ~400 KiB of data: far larger than any single bounded window below.
        let n = 100_000i32;
        let mut store = WindowProbeStore::open(build_unlimited(n, 256));
        let file_len = store.data.len();
        assert!(
            file_len > 300_000,
            "test file unexpectedly small: {file_len}"
        );

        let (mut loc, datatype) = locate(&store);
        // Setup (locate + datatype) already obeys the window bound.
        const WINDOW: usize = 16 * 1024;
        assert!(
            store.max_read.get() <= WINDOW,
            "locate read a {}-byte window (> {WINDOW})",
            store.max_read.get()
        );

        // A small append against the large file: every read (trailing chunk,
        // element slots, checksum regions) stays within the window bound, and
        // the total read volume is a small constant, not O(file size).
        store.reset_counters();
        append_i32s(&mut store, &mut loc, &datatype, n..n + 10);
        assert!(
            store.max_read.get() <= WINDOW,
            "append read a {}-byte window (> {WINDOW})",
            store.max_read.get()
        );
        assert!(
            store.total_read.get() <= 64 * 1024,
            "append read {} bytes total (> 64 KiB) on a {file_len}-byte file",
            store.total_read.get()
        );
        assert!(
            store
                .reads
                .borrow()
                .iter()
                .all(|&(_, len)| len < file_len / 2),
            "an append read scaled with file size"
        );

        // Growth loop: the per-append read volume stays flat as the file grows.
        let mut worst = 0usize;
        let mut next = n + 10;
        for _ in 0..20 {
            store.reset_counters();
            append_i32s(&mut store, &mut loc, &datatype, next..next + 300);
            worst = worst.max(store.total_read.get());
            next += 300;
        }
        assert!(
            worst <= 64 * 1024,
            "per-append read volume grew to {worst} bytes"
        );

        // The grown file reads back correctly through the public reader.
        let file = crate::File::from_bytes(store.data).unwrap();
        let ds = file.dataset("d").unwrap();
        let got = ds.read_i32().unwrap();
        let expected: Vec<i32> = (0..next).collect();
        assert_eq!(got.len(), expected.len());
        assert_eq!(got, expected);
    }

    /// Unaligned appends rewrite the trailing partial chunk: still bounded — the
    /// one chunk is the largest data read the plan performs.
    #[test]
    fn partial_tail_append_reads_one_chunk_window() {
        let n = 50_000i32;
        let chunk = 256u64;
        let mut store = WindowProbeStore::open(build_unlimited(n + 7, chunk));
        let (mut loc, datatype) = locate(&store);

        store.reset_counters();
        append_i32s(&mut store, &mut loc, &datatype, n + 7..n + 7 + 13);
        let chunk_bytes = (chunk as usize) * 4;
        assert!(
            store.max_read.get() <= chunk_bytes.max(8 * 1024),
            "partial-tail append read a {}-byte window",
            store.max_read.get()
        );

        let file = crate::File::from_bytes(store.data).unwrap();
        let got = file.dataset("d").unwrap().read_i32().unwrap();
        let expected: Vec<i32> = (0..n + 7 + 13).collect();
        assert_eq!(got, expected);
    }

    /// A filtered element's stored-size field is at most the eight bytes of the
    /// `u64` that fills it, and a file naming a wider one is refused at `locate`
    /// rather than panicking when the element is laid out.
    ///
    /// The refusal is two-sided for a reason: the low side alone was what this
    /// guard checked, and the high side reached
    /// `rec.stored_size.to_le_bytes()[..csz]` with `csz` up to 243. Reproduced
    /// through the safe public API before the upper bound landed — an
    /// `element_size` of 30 on an 8-byte-address file panicked with "range end
    /// index 18 out of range for slice of length 8".
    #[test]
    fn an_element_width_wider_than_a_u64_is_refused_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wide_elem.h5");
        let mut b = crate::writer::FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..4096).collect::<Vec<_>>())
            .with_shape(&[4096])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[4])
            .with_deflate(1);
        b.write(&path).unwrap();

        // Widen the array header's element_size (byte 6 of `EAHD`) past the three
        // fields it names. A filtered element is address + stored size + filter
        // mask, so on an 8-byte-address file the format's own ceiling is 20.
        let mut bytes = std::fs::read(&path).unwrap();
        let at = bytes
            .windows(4)
            .position(|w| w == b"EAHD")
            .expect("a filtered unlimited dataset is indexed by an extensible array");
        assert_eq!(bytes[at + 6], 14, "the writer's own element width moved");
        bytes[at + 6] = 30;
        // Restamp the header. Its checksum covers this field, so leaving it
        // stale would have the reader refuse the file for being corrupt before
        // it ever reached the width guard under test.
        crate::checksum::stamp_trailing(&mut bytes, at, 12 + 6 * 8 + 8 + 4);
        std::fs::write(&path, &bytes).unwrap();

        let f = crate::reader::File::open_rw(&path).unwrap();
        let err = f
            .dataset("d")
            .and_then(|mut d| d.append(&[1i32, 2, 3, 4]))
            .expect_err("an element width wider than a u64 must be refused");
        assert!(
            std::format!("{err:?}").contains("element width"),
            "the refusal should name the element width, but said {err:?}"
        );
    }
}
