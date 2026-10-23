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

use hdf5_pure_format::__private::BlockOffsetWidth;
use hdf5_pure_format::__private::DataBlockGeometry;
use hdf5_pure_format::__private::EA_CLIENT_ID_UNFILTERED;
use hdf5_pure_format::__private::ExtensibleArrayBitGeometry;
use hdf5_pure_format::__private::ExtensibleArrayGeometry;
use hdf5_pure_format::__private::ExtensibleArrayHeader;
use hdf5_pure_format::__private::ExtensibleArrayStats;
use hdf5_pure_format::__private::IndexSlots;
use hdf5_pure_format::__private::MessageRecord;
use hdf5_pure_format::__private::MessageRecordLayout;
use hdf5_pure_format::__private::OBJECT_HEADER_PREFIX_MAX_LEN;
use hdf5_pure_format::__private::ObjectHeaderContinuation;
use hdf5_pure_format::__private::ObjectHeaderPrefix;
use hdf5_pure_format::__private::SuperBlockGeometry;
use hdf5_pure_space::__private::AllocationClass;

#[cfg(test)]
use crate::DeflateLevel;
#[cfg(test)]
use crate::address::AddressContext;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::checksum;
use crate::chunked_read::ChunkDecoding;
use crate::chunked_read::ChunkInfo;
use crate::chunked_read::StoredChunkSize;
use crate::chunked_write;
use crate::convert::Narrow;
use crate::data_layout::ChunkIndexLayout;
use crate::data_layout::ChunkedLayoutFlags;
use crate::data_layout::DataLayout;
use crate::dataspace::{Dataspace, MaxExtent};
use crate::datatype::Datatype;
use crate::edit::{self, LOSSY_TAIL_REFUSAL};
use crate::error::{Error, FormatError};
use crate::fill_value::FillPattern;
use crate::filter_pipeline::FilterPipeline;
use crate::filters::ChunkContext;
use crate::filters::FilterScratch;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::prepared_chunk::ChunkEncoding;
use crate::prepared_chunk::PreparedChunk;
use crate::source::Source;
use crate::width::{FormatWidths, OffsetWidth};

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
    /// This file's validated address and length field widths.
    fn widths(&self) -> FormatWidths;

    /// Returns the end of file recorded in the cached superblock.
    ///
    /// An interrupted append may leave image bytes beyond this position.
    fn recorded_eof(&self) -> u64;

    /// Writes `bytes` into an allocation of `class` and returns its physical position.
    ///
    /// Chunk payloads use [`AllocationClass::Raw`]. Extensible-array data and super blocks use
    /// [`AllocationClass::Metadata`]. On PAGE files, the editor places both in raw pages.
    /// During reclamation, the editor returns index storage to raw-page free space only where
    /// its PAGE placement is proven.
    ///
    /// The store may allocate from a classic aggregate, reuse committed free space, or append at
    /// end-of-file. The allocation lies
    /// within the image's logical length on return. Before publishing a chunk address,
    /// [`apply_ea_append`] records any increase in that length in the superblock.
    ///
    /// The caller must issue [`sync`](Self::sync) before publishing a reference to the allocation.
    /// A store may gather writes and flush them in address order at that barrier.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if address conversion or PAGE allocation geometry is invalid.
    /// Returns [`Error::Io`] if writing the allocation fails.
    fn allocate(&mut self, bytes: &[u8], class: AllocationClass) -> Result<u64, Error>;
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
        let width = usize::from(self.widths().offsets.get());
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
        cks_field.copy_from_slice(&checksum::jenkins_lookup3(body).to_le_bytes());
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

    /// The flags the chunked layout sets, such as [`DONT_FILTER_PARTIAL_BOUND_CHUNKS`], under which
    /// a partial edge chunk is stored without the filter pipeline.
    ///
    /// [`DONT_FILTER_PARTIAL_BOUND_CHUNKS`]: hdf5_pure_format::__private::DONT_FILTER_PARTIAL_BOUND_CHUNKS
    pub flags: ChunkedLayoutFlags,
    /// Elements per chunk along axis 0 (the only varying axis for rank 1).
    pub chunk_elems: u64,
    /// Bytes per dataset element, the element size the chunked layout stores.
    pub elem_bytes: NonZeroUsize,
    /// Bytes per chunk (uncompressed).
    pub chunk_bytes: usize,

    /// Extensible Array client id: 0 = unfiltered (bare-address element), 1 =
    /// filtered (`address + compressed_size + filter_mask` element).
    pub client_id: u8,
    /// Extensible Array header address and derived geometry.
    pub ea_addr: StoredAddress,
    pub offset_width: OffsetWidth,
    pub geom: ExtensibleArrayGeometry,
    pub idx_blk_elmts: u64,
    /// Size of one stored EA element in bytes (offset size for unfiltered; wider
    /// for filtered).
    pub ea_elem_size: usize,
    /// The parsed maximum element bit count and data block page exponent.
    pub ea_bits: ExtensibleArrayBitGeometry,
    /// Address of the EA index block (`EAIB`).
    pub index_block_addr: StoredAddress,
    /// Current number of chunks indexed (EA element count).
    pub num_chunks: u64,
}

impl Located {
    /// Locates the dataset whose object header is at `oh_addr`, and reads its dataspace, its
    /// layout with the layout flags, and the geometry of its Extensible Array.
    ///
    /// The result records whether the dataset is filtered and holds the spans of its datatype,
    /// filter pipeline, and fill value messages, and the caller decides whether it accepts the
    /// filters. `unsupported` builds the caller's error for a dataset it cannot append to, so
    /// each writer reports through its own variant. The caller resolves the path: an owned handle
    /// holds its address, and the path-addressed writers resolve against their own mirrors.
    ///
    /// Every read is a bounded [`Source`] window, so this works over a store with no whole-file
    /// mirror.
    ///
    /// # Errors
    ///
    /// Returns the error `unsupported` builds if an in-place append does not accept the dataset,
    /// and [`Error::MissingMessage`] if the object header has no dataspace, layout, or datatype
    /// message. Returns the error of a read or a parse of the object header, one of its messages,
    /// or the Extensible Array header.
    pub(crate) fn locate_at<F: Store>(
        file: &F,
        oh_addr: u64,
        unsupported: fn(&'static str) -> Error,
    ) -> Result<LocateResult, Error> {
        let widths = file.widths();
        let os = widths.offsets.get();

        let walk = walk_v2_object_header(file, oh_addr, widths)?;

        let dataspace_msg = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DATASPACE)
            .ok_or(Error::MissingMessage(MessageType::DATASPACE))?;
        let layout_msg = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DATA_LAYOUT)
            .ok_or(Error::MissingMessage(MessageType::DATA_LAYOUT))?;
        let datatype_msg = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DATATYPE)
            .ok_or(Error::MissingMessage(MessageType::DATATYPE))?;
        let filter_msg = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FILTER_PIPELINE);
        // The versioned message wins over the legacy one when a header carries
        // both, matching how the read path picks.
        let fill_msg = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FILL_VALUE)
            .or_else(|| {
                walk.messages
                    .iter()
                    .find(|m| m.msg_type == MessageType::FILL_VALUE_OLD)
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
        let dataspace = Dataspace::parse_with_length_width(&ds_bytes, widths.lengths)?;
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
        let layout = DataLayout::parse_with_widths(&layout_bytes, widths)?;
        let (ea_addr, chunk_dims, flags) = match layout {
            DataLayout::Chunked {
                chunk_dimensions,
                index:
                    ChunkIndexLayout::ExtensibleArray {
                        address: Some(addr),
                    },
                flags,
                ..
            } => (addr, chunk_dimensions, flags),
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
        let chunk_elems = chunk_dims[0];
        // A zero element-size pseudo-dimension is a malformed layout; refuse
        // rather than divide by it when validating an append length. The binding
        // carries that refusal forward, so no later step re-checks it.
        let Some(elem_bytes) = NonZeroUsize::new(chunk_dims[1].to_usize()?) else {
            return Err(unsupported("dataset has a zero-sized element"));
        };
        let chunk_bytes = chunk_elems.to_usize()? * elem_bytes.get();

        let array = crate::extensible_array::read_header(file, widths, ea_addr)?;
        let ea_header = array.header();
        let has_filters = filter_msg.is_some();
        if (ea_header.client_id == 1) != has_filters {
            return Err(unsupported(
                "dataset filter metadata is inconsistent (chunk-index client id \
                 disagrees with the filter pipeline)",
            ));
        }
        // Immediate append writes a stored-size field of one to eight bytes.
        // The header parser accepts an empty size field as well.
        let elem_w = ea_header.element_size as usize;
        if ea_header.client_id == 1 && !(os as usize + 5..=os as usize + 12).contains(&elem_w) {
            return Err(unsupported(
                "malformed filtered extensible-array element width",
            ));
        }
        if ea_header.client_id == EA_CLIENT_ID_UNFILTERED && elem_w != os as usize {
            return Err(unsupported(
                "malformed unfiltered extensible-array element width",
            ));
        }
        let geom = array.geometry()?;
        let index_block_addr = ea_header.index_block_address;
        // The dataspace dimension is the single commit point; the EA element
        // count is published one step earlier. If a prior writer crashed between
        // the two, the on-disk EA count is ahead of the committed dimension. Seed
        // the chunk count from the *committed* dimension so the next append rolls
        // forward from the last commit -- overwriting any slots a crashed writer
        // wrote but never committed -- instead of appending past them.
        let num_chunks = if chunk_elems == 0 {
            ea_header.max_idx_set
        } else {
            current_dim.div_ceil(chunk_elems)
        };

        Ok(LocateResult {
            located: Located {
                dim0_off,
                current_dim,
                ohdr_chunk_start: dataspace_msg.chunk_start,
                ohdr_chunk_msg_end: dataspace_msg.chunk_msg_end,
                flags,
                chunk_elems,
                elem_bytes,
                chunk_bytes,
                client_id: ea_header.client_id,
                ea_addr,
                offset_width: widths.offsets,
                geom,
                idx_blk_elmts: ea_header.idx_blk_elmts as u64,
                ea_elem_size: ea_header.element_size as usize,
                ea_bits: ea_header.bit_geometry(),
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

    /// Reads the trailing partial chunk and returns the bytes of its live elements.
    ///
    /// The chunk is decoded as a reader at the current extent decodes it, through the filter
    /// pipeline unless the layout flags exempt it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] if the index has no record for the chunk, and the
    /// errors of the read of the record and of [`ChunkInfo::read_live_elements`].
    fn read_live_tail<F: Store>(
        &self,
        file: &F,
        scratch: &mut FilterScratch,
        pipeline: Option<&FilterPipeline>,
        ctx: ChunkContext<'_>,
    ) -> Result<Vec<u8>, Error> {
        let rec = self
            .read_element(file, self.current_dim / self.chunk_elems)?
            .ok_or(Error::AppendUnsupported(
                "trailing partial chunk is missing from the index",
            ))?;
        let stored_size = if self.client_id == EA_CLIENT_ID_UNFILTERED {
            self.chunk_bytes.to_u64()
        } else {
            rec.stored_size
        };
        let chunk = ChunkInfo {
            chunk_size: StoredChunkSize::v4(stored_size),
            filter_mask: rec.filter_mask,
            offsets: vec![self.current_dim - self.current_dim % self.chunk_elems],
            address: rec.addr,
        };
        Ok(chunk.read_live_elements(file, scratch, self.flags, pipeline, ctx, self.current_dim)?)
    }

    /// Returns how a reader at the extent `extent` decodes the chunk at element `offset`.
    fn decoding_at(&self, offset: u64, extent: u64) -> ChunkDecoding {
        ChunkDecoding::at_extent(self.flags, &[self.chunk_elems], &[extent], &[offset])
    }

    /// Read the element record stored at byte offset `off`.
    fn read_element_at<F: Store>(&self, file: &F, off: u64) -> Result<ElemRecord, Error> {
        let os = file.widths().offsets.get() as usize;
        if self.client_id == 0 {
            return Ok(ElemRecord::addr_only(file.read_addr_at(off)?));
        }
        // One bounded read of the whole fixed-width element record.
        let elem = file.read_exact_at(off, self.ea_elem_size)?;
        let addr_w = os;
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
        let os = file.widths().offsets.get() as usize;
        let elem_size = self.ea_elem_size as u64;
        let idx = self.idx_blk_elmts;
        let blk_off = self.ea_bits.block_offset_width();
        let page_nelmts = self.ea_bits.page_nelmts().get();

        if e < idx {
            let ib_prefix = (4 + 1 + 1 + os) as u64;
            return Ok(Some(
                self.index_block_addr.get() + ib_prefix + e * elem_size,
            ));
        }

        let region = locate_data_block(&self.geom, idx, e)?;
        let is_paged = region.blocks(page_nelmts).is_paged();
        let slot = e - region.db_start;

        // Resolve the owning super block (if any) and the data-block pointer,
        // returning None if either is unallocated.
        let sblk_addr = match region.parent {
            Parent::Super { sblk_j, .. } => {
                let a = self.super_block_addr(file, sblk_j)?;
                if a.is_undefined(file.widths().offsets) {
                    return Ok(None);
                }
                Some(a)
            }
            Parent::IndexDirect { .. } => None,
        };
        let dblk_ptr_off = self.dblk_ptr_off(sblk_addr, &region, os, blk_off)?;
        let dblk_addr = file.read_addr_at(dblk_ptr_off)?;
        if dblk_addr.is_undefined(file.widths().offsets) {
            return Ok(None);
        }

        let off = if !is_paged {
            let db_prefix = (4 + 1 + 1 + os + blk_off.bytes()) as u64;
            dblk_addr.get() + db_prefix + slot * elem_size
        } else {
            let header_size = (4 + 1 + 1 + os + blk_off.bytes() + 4) as u64;
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
        blk_off: BlockOffsetWidth,
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
                    region.super_block(self.ea_bits.page_nelmts().get()),
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
                if rec.addr.is_undefined(file.widths().offsets) {
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
        let os = file.widths().offsets.get() as usize;
        let elem_size = self.ea_elem_size as u64;
        let idx = self.idx_blk_elmts;
        let blk_off = self.ea_bits.block_offset_width();

        // Inline element slots live directly in the index block.
        if e < idx {
            let ib_prefix = (4 + 1 + 1 + os) as u64;
            let slot_off = self.index_block_addr.get() + ib_prefix + e * elem_size;
            let mut buf = [0u8; MAX_EA_ELEM];
            let n = self.element_bytes(&mut buf, os, rec)?;
            return self.publish_index_block(file, slot_off, &buf[..n]);
        }

        let region = locate_data_block(&self.geom, idx, e)?;
        let dblk_nelmts = region.dblk_nelmts;
        let sb_geom = region.super_block(self.ea_bits.page_nelmts().get());
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
        let dblk_addr = if existing.is_undefined(file.widths().offsets) {
            let new_addr = self.alloc_undef_data_block(file, dblk_nelmts, block_offset_rel)?;
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
            let db_prefix = (4 + 1 + 1 + os + blk_off.bytes()) as u64;
            let block_start = dblk_addr.get();
            let elem_off = block_start + db_prefix + slot * elem_size;
            let mut buf = [0u8; MAX_EA_ELEM];
            let n = self.element_bytes(&mut buf, os, rec)?;
            let cks_off = block_start + db_prefix + dblk_nelmts * elem_size;
            file.publish_checksummed(block_start, cks_off, elem_off, &buf[..n])?;
        } else {
            let page_nelmts = self.ea_bits.page_nelmts().get();
            let header_size = (4 + 1 + 1 + os + blk_off.bytes() + 4) as u64;
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
                let npages = dblk_nelmts / self.ea_bits.page_nelmts().get();
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
        let os = file.widths().offsets.get() as usize;
        let ib_prefix = (4 + 1 + 1 + os) as u64;
        let ndblk_addrs = self.geom.direct_dblk_nelmts().len();
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
        sb: SuperBlockGeometry,
    ) -> Result<StoredAddress, Error> {
        let existing = self.super_block_addr(file, sblk_j)?;
        if !existing.is_undefined(file.widths().offsets) {
            return Ok(existing);
        }
        let os = file.widths().offsets.get() as usize;
        let ib_prefix = (4 + 1 + 1 + os) as u64;
        let ndblk_addrs = self.geom.direct_dblk_nelmts().len();

        let bitmap = vec![0u8; sb.bitmap_size().to_usize()?];
        let undef = vec![StoredAddress::undefined(file.widths().offsets); sb.ndblks.to_usize()?];
        let super_block = hdf5_pure_format::__private::encode_super_block(
            self.ea_addr,
            sb_block_offset,
            &bitmap,
            &undef,
            self.offset_width,
            self.client_id,
            self.ea_bits.block_offset_width(),
        );
        // Every path into this engine rejects a file with a userblock
        // (`WriteEngine::append_prepare` and `File::open_swmr_writer`), so the base address
        // is zero and the position `allocate` returns is the address the file stores. The
        // three further allocations below rest on the same invariant.
        let new_addr = StoredAddress::new(file.allocate(&super_block, AllocationClass::Metadata)?);
        #[cfg(test)]
        alloc_probe::note_super_block();
        // The block exists before anything names it. When its bytes were appended
        // they sit above the slot below, which is in the index block near the
        // front, so without this the two are one barrier-free window and a
        // gathering image issues them in *address* order — the pointer first
        // (issue #288). A block drawn from freed space may land either side of
        // the slot, and which it is is not knowable here, so the barrier stands
        // unconditionally. See `Store::allocate`.
        file.sync()?;

        let slot_off = self.index_block_addr.get()
            + ib_prefix
            + self.idx_blk_elmts * self.ea_elem_size as u64
            + ((ndblk_addrs + sblk_j) * os) as u64;
        self.publish_index_block(file, slot_off, &new_addr.get().to_le_bytes()[..os])?;
        Ok(new_addr)
    }

    /// Allocates a data block (`EADB`) of `dblk_nelmts` elements that each store
    /// the undefined address, and returns its address.
    ///
    /// A block of more than one page of elements is paged.
    fn alloc_undef_data_block<F: Store>(
        &self,
        file: &mut F,
        dblk_nelmts: u64,
        block_offset_rel: u64,
    ) -> Result<StoredAddress, Error> {
        let os = self.offset_width;
        let block = hdf5_pure_format::__private::encode_data_block(
            &IndexSlots::dense(&[]),
            0,
            dblk_nelmts.to_usize()?,
            block_offset_rel,
            self.ea_addr,
            os,
            self.client_id != EA_CLIENT_ID_UNFILTERED,
            self.element_size_width(usize::from(os.get())),
            self.client_id,
            self.ea_bits,
        )?;
        Ok(StoredAddress::new(
            file.allocate(&block, AllocationClass::Metadata)?,
        ))
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
        let os = file.widths().offsets.get() as usize;
        let ib_prefix = (4 + 1 + 1 + os) as u64;
        let ndblk_addrs = self.geom.direct_dblk_nelmts().len();
        let nsblk_addrs = self.geom.nsblk_addrs();
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
        sb: SuperBlockGeometry,
        blk_off: BlockOffsetWidth,
        at: u64,
        value: &[u8],
    ) -> Result<(), Error> {
        let os = file.widths().offsets.get() as usize;
        let prefix = (4 + 1 + 1 + os + blk_off.bytes()) as u64;
        let block_start = sblk_addr.get();
        let cks_off = block_start + prefix + sb.bitmap_size() + sb.ndblks * os as u64;
        file.publish_checksummed(block_start, cks_off, at, value)
    }

    /// Publishes the planned header statistics and their checksum.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] if the statistics lie outside the header, and the
    /// read, conversion, or write errors of [`Store::publish_checksummed`].
    pub(crate) fn update_ea_header<F: Store>(
        &self,
        file: &mut F,
        stats: ExtensibleArrayStats,
    ) -> Result<(), Error> {
        let ls = file.widths().lengths.get() as usize;
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
        let header_len = ExtensibleArrayHeader::serialized_size_with_widths(file.widths()) as u64;
        let cks_off = ea_addr + header_len - 4;
        file.publish_checksummed(ea_addr, cks_off, ea_addr + 12, &stat_block[..at])
    }

    /// Publish `new_dim` as the dataspace axis-0 dimension (the commit point) and
    /// recompute the containing object-header chunk's checksum.
    pub(crate) fn patch_dimension<F: Store>(
        &self,
        file: &mut F,
        new_dim: u64,
    ) -> Result<(), Error> {
        let ls = file.widths().lengths.get() as usize;
        file.publish_checksummed(
            self.ohdr_chunk_start,
            self.ohdr_chunk_msg_end,
            self.dim0_off,
            &new_dim.to_le_bytes()[..ls],
        )
    }

    /// Computes header statistics for a dense prefix of `num_chunks` element slots.
    ///
    /// Counts every data block in that prefix, including blocks whose pointers are undefined.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if an element range, block size,
    /// or statistic exceeds [`u64::MAX`], or [`FormatError::ValueTooLargeForPlatform`] if a block
    /// size or data-block index exceeds [`usize::MAX`]. Returns [`Error::AppendUnsupported`] if
    /// the prefix extends beyond the geometry.
    fn append_stats(&self, num_chunks: u64) -> Result<ExtensibleArrayStats, Error> {
        if num_chunks > self.idx_blk_elmts {
            locate_data_block(&self.geom, self.idx_blk_elmts, num_chunks - 1)?;
        }
        let mut stats = ExtensibleArrayStats {
            nsuper_blks: 0,
            super_blk_size: 0,
            ndata_blks: 0,
            data_blk_size: 0,
            max_idx_set: num_chunks,
            nelmts: self.idx_blk_elmts,
        };
        let mut elem = self.idx_blk_elmts;
        for &elements in self.geom.direct_dblk_nelmts() {
            if elem >= num_chunks {
                return Ok(stats);
            }
            let block_size = self.data_block_size(elements)?;
            stats.ndata_blks = ea_sum([stats.ndata_blks, 1])?;
            stats.data_blk_size = ea_sum([stats.data_blk_size, block_size])?;
            stats.nelmts = ea_sum([stats.nelmts, elements])?;
            elem = ea_sum([elem, elements])?;
        }
        for &(ndblks, elements) in self
            .geom
            .sblks()
            .iter()
            .skip(self.geom.first_indirect_sblk())
            .take(self.geom.nsblk_addrs())
        {
            if elem >= num_chunks {
                return Ok(stats);
            }
            let span = ea_product(ndblks, elements)?;
            let end = ea_sum([elem, span])?;
            let used = (num_chunks.min(end) - elem).div_ceil(elements);
            let blocks = DataBlockGeometry {
                dblk_nelmts: elements,
                page_nelmts: self.ea_bits.page_nelmts().get(),
            };
            let bitmap = ea_product(ndblks, blocks.npages().div_ceil(8))?;
            let super_size = ea_sum([
                self.block_prefix_size(),
                bitmap,
                ea_product(ndblks, u64::from(self.offset_width.get()))?,
                EA_CHECKSUM_SIZE,
            ])?;
            super_size.to_usize()?;
            stats.nsuper_blks = ea_sum([stats.nsuper_blks, 1])?;
            stats.super_blk_size = ea_sum([stats.super_blk_size, super_size])?;
            stats.ndata_blks = ea_sum([stats.ndata_blks, used])?;
            stats.data_blk_size = ea_sum([
                stats.data_blk_size,
                ea_product(used, self.data_block_size(elements)?)?,
            ])?;
            stats.nelmts = ea_sum([stats.nelmts, ea_product(used, elements)?])?;
            elem = end;
        }
        Ok(stats)
    }

    /// Returns the encoded data-block size, including page storage and checksums.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if the size exceeds [`u64::MAX`],
    /// or [`FormatError::ValueTooLargeForPlatform`] if it exceeds [`usize::MAX`].
    fn data_block_size(&self, elements: u64) -> Result<u64, FormatError> {
        let blocks = DataBlockGeometry {
            dblk_nelmts: elements,
            page_nelmts: self.ea_bits.page_nelmts().get(),
        };
        let size = if blocks.is_paged() {
            ea_sum([
                self.block_prefix_size(),
                EA_CHECKSUM_SIZE,
                ea_product(
                    blocks.npages(),
                    ea_sum([
                        ea_product(blocks.page_nelmts, self.ea_elem_size.to_u64())?,
                        EA_CHECKSUM_SIZE,
                    ])?,
                )?,
            ])?
        } else {
            ea_sum([
                self.block_prefix_size(),
                ea_product(elements, self.ea_elem_size.to_u64())?,
                EA_CHECKSUM_SIZE,
            ])?
        };
        size.to_usize()?;
        Ok(size)
    }

    /// Returns the prefix length through the block-offset field of a data or secondary block.
    ///
    /// Version 0 blocks are defined in "The Extensible Array Index" of the [format specification,
    /// version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
    fn block_prefix_size(&self) -> u64 {
        EA_BLOCK_PREFIX_SIZE
            + u64::from(self.offset_width.get())
            + self.ea_bits.block_offset_width().bytes().to_u64()
    }
}

/// The result of planning one in-place Extensible-Array append, computed without a write: the
/// chunks to write and the counts the write phases publish.
pub(crate) struct AppendPlan {
    /// The chunks to write, each with its stored bytes and its filter mask, in element order from
    /// `n_full`. The first is the rewritten trailing chunk when the current length is not a
    /// multiple of the chunk length.
    chunks: Vec<PreparedChunk>,
    /// Element index the first new chunk occupies.
    n_full: u64,
    /// New axis-0 dimension after the append (the commit value).
    new_dim: u64,
    /// New number of indexed chunks.
    new_num_chunks: u64,
    /// The header statistics to publish after the index writes.
    stats: ExtensibleArrayStats,
}

impl AppendPlan {
    /// Returns the stored size of the largest chunk in the plan, or 0 for a plan without chunks.
    pub(crate) fn largest_stored_size(&self) -> u64 {
        self.chunks
            .iter()
            .map(PreparedChunk::stored_size)
            .max()
            .unwrap_or(0)
    }
}

/// Plans chunk bytes, index counts, and header statistics for an in-place append.
///
/// Checks the element ranges and metadata sizes before reading a trailing chunk or encoding
/// `raw`, the little-endian bytes of `new_elems` elements. Planning leaves the file unchanged.
///
/// `grow_visible_tail` says whether the caller may rewrite the trailing chunk of
/// a *filtered* dataset — a chunk a reader at the old dimension can already see.
/// A `File::open_rw` session holds an exclusive lock and passes `true`; the SWMR
/// writer, whose readers are the point, passes `false`. It has no bearing on an
/// unfiltered dataset, whose trailing chunk is raw and grows by relocation
/// either way, and none on a *lossy* pipeline, which is refused here whatever the
/// caller asks for.
///
/// # Errors
///
/// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if an index range, metadata size, or
/// header statistic exceeds [`u64::MAX`], or [`FormatError::ValueTooLargeForPlatform`] if metadata
/// sizes or indexes exceed [`usize::MAX`].
///
/// Returns [`Error::AppendUnsupported`] if the new dimension overflows, the index geometry does
/// not cover the append, a filtered tail rewrite is unsupported, or an encoded chunk size does
/// not fit its index record. Errors from reading the trailing chunk, constructing its datatype
/// context, splitting the element bytes, or encoding the chunks are propagated.
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
    let new_num_chunks = new_dim.div_ceil(chunk_elems);
    let stats = loc.append_stats(new_num_chunks)?;
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
    // ordinary in-progress states. A chunk the layout flags exempt from the filters at either
    // dimension is stored as its data, so a reader at either dimension decodes it to the same
    // data. See `ChunkEncoding::decodable_by`.
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
        if !edit::pipeline_lossless(pl) {
            return Err(Error::AppendUnsupported(LOSSY_TAIL_REFUSAL));
        }
    }

    // Build the raw tail region: the live prefix of any rewritten partial chunk,
    // then the appended bytes.
    let ctx = ChunkContext::from_datatype(spatial, datatype)?;
    // One encoder and decoder across the appended tail. See `FilterScratch`.
    let mut scratch = FilterScratch::new();
    let mut tail_raw = if has_partial {
        loc.read_live_tail(file, &mut scratch, pipeline, ctx)?
    } else {
        Vec::new()
    };
    tail_raw.extend_from_slice(raw);

    // Split the tail into full chunk buffers and encode each for a reader at the current
    // dimension and one at the new dimension. The final chunk's slots past the new dimension
    // take the dataset's fill value (issue #296).
    let tail_len_elems = new_dim - n_full * chunk_elems;
    let split =
        chunked_write::split_into_chunks(&tail_raw, &[tail_len_elems], spatial, element_size, fill)
            .map_err(Error::Format)?;
    let chunks = (n_full..)
        .zip(split)
        .map(|(e, logical)| {
            let offset = e * chunk_elems;
            let current = (offset < current_dim).then(|| loc.decoding_at(offset, current_dim));
            let encoding = ChunkEncoding::decodable_by(current, loc.decoding_at(offset, new_dim));
            PreparedChunk::encode(&mut scratch, pipeline, ctx, encoding, logical)
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Every blob has to fit the element it will be recorded in. Checked here,
    // while nothing has been written, rather than where the element is laid out
    // in phase 2 — by then phase 1 has already placed the chunk bytes, and the
    // refusal would leave them behind unreferenced.
    let os = file.widths().offsets.get() as usize;
    for chunk in &chunks {
        loc.check_stored_size(os, chunk.stored_size())?;
    }

    Ok(AppendPlan {
        chunks,
        n_full,
        new_dim,
        new_num_chunks,
        stats,
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
    AppendPlan {
        chunks,
        n_full,
        new_dim,
        new_num_chunks,
        stats,
    }: AppendPlan,
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
    // past what it records. The initial value comes from the superblock: an
    // interrupted append can leave reserved storage beyond its recorded EOF.
    let mut recorded_eof = file.recorded_eof();
    let mut chunk_addrs = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        chunk_addrs.push(StoredAddress::new(
            file.allocate(chunk.bytes(), AllocationClass::Raw)?,
        ));
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
    for (e, (addr, chunk)) in (n_full..).zip(chunk_addrs.into_iter().zip(&chunks)) {
        let rec = ElemRecord {
            addr,
            stored_size: chunk.stored_size(),
            filter_mask: chunk.filter_mask(),
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
    loc.update_ea_header(file, stats)?;
    file.sync()?;
    if max_phase < 4 {
        return Ok(());
    }

    // Phase 4: publish the dataspace dimension — the single commit point.
    loc.patch_dimension(file, new_dim)?;
    file.sync()?;

    loc.current_dim = new_dim;
    loc.num_chunks = new_num_chunks;
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
    blk_off: BlockOffsetWidth,
    global_page: u64,
) -> Result<(u64, u8), Error> {
    let os = file.widths().offsets.get() as usize;
    let bitmap_start = sblk_addr.get() + (4 + 1 + 1 + os + blk_off.bytes()) as u64;
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
    sb: SuperBlockGeometry,
    blk_off: BlockOffsetWidth,
) -> u64 {
    let prefix = (4 + 1 + 1 + os + blk_off.bytes()) as u64;
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
    fn super_block(&self, page_nelmts: u64) -> SuperBlockGeometry {
        SuperBlockGeometry {
            ndblks: self.ndblks,
            blocks: self.blocks(page_nelmts),
        }
    }

    /// How this data block is paged, which holds whether or not it has a super
    /// block above it.
    fn blocks(&self, page_nelmts: u64) -> DataBlockGeometry {
        DataBlockGeometry {
            dblk_nelmts: self.dblk_nelmts,
            page_nelmts,
        }
    }
}

/// Locates a non-inline element in the direct or secondary data-block ranges.
///
/// # Errors
///
/// Returns [`FormatError::InvalidExtensibleArrayGeometry`] if `e` is an inline element or a
/// range calculation exceeds [`u64::MAX`], or [`FormatError::ValueTooLargeForPlatform`] if the
/// data-block index exceeds [`usize::MAX`]. Returns [`Error::AppendUnsupported`] if the geometry
/// does not cover `e`.
fn locate_data_block(
    geom: &ExtensibleArrayGeometry,
    idx_blk_elmts: u64,
    e: u64,
) -> Result<DataBlockLoc, Error> {
    if e < idx_blk_elmts {
        return Err(FormatError::InvalidExtensibleArrayGeometry.into());
    }
    let mut elem = idx_blk_elmts;
    for (ordinal, &dn) in geom.direct_dblk_nelmts().iter().enumerate() {
        let end = ea_sum([elem, dn])?;
        if e < end {
            return Ok(DataBlockLoc {
                db_start: elem,
                dblk_nelmts: dn,
                ndblks: 1,
                sb_block_offset: 0,
                parent: Parent::IndexDirect { ordinal },
            });
        }
        elem = end;
    }
    for (j, &(ndblks, dn)) in geom
        .sblks()
        .iter()
        .skip(geom.first_indirect_sblk())
        .take(geom.nsblk_addrs())
        .enumerate()
    {
        let span = ea_product(ndblks, dn)?;
        let end = ea_sum([elem, span])?;
        if e < end {
            let sb_block_offset = elem - idx_blk_elmts;
            let within = e - elem;
            let dblk_local = (within / dn).to_usize()?;
            let db_start = ea_sum([elem, ea_product(dblk_local.to_u64(), dn)?])?;
            return Ok(DataBlockLoc {
                db_start,
                dblk_nelmts: dn,
                ndblks,
                sb_block_offset,
                parent: Parent::Super {
                    sblk_j: j,
                    dblk_local,
                },
            });
        }
        elem = end;
    }
    Err(Error::AppendUnsupported(
        "chunk index geometry does not cover the appended element",
    ))
}

fn ea_sum(values: impl IntoIterator<Item = u64>) -> Result<u64, FormatError> {
    values.into_iter().try_fold(0u64, |total, value| {
        total
            .checked_add(value)
            .ok_or(FormatError::InvalidExtensibleArrayGeometry)
    })
}

fn ea_product(left: u64, right: u64) -> Result<u64, FormatError> {
    left.checked_mul(right)
        .ok_or(FormatError::InvalidExtensibleArrayGeometry)
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

/// Walks a version 2 object header, chunk 0 and every continuation block, and
/// records, for each message other than a continuation, the absolute offset of
/// its body and the checksum region of its chunk.
///
/// The walk works over a store with no whole-file mirror. It reads the first
/// [`OBJECT_HEADER_PREFIX_MAX_LEN`] bytes of the header, or fewer where the file
/// ends sooner, then the message region of chunk 0, then one window per
/// continuation block.
///
/// # Errors
///
/// Returns [`Error::Format`] if the header ends inside its prefix or does not
/// begin with its signature, if the version is not 2, if a continuation block
/// lacks its signature or is too short for it and its checksum, if a chunk runs
/// past the end of `source`, if the body of a message record runs past the end
/// of its chunk, if a continuation message ends inside its fields or stores the
/// undefined address, or if the header has more than 255 continuation blocks.
fn walk_v2_object_header<S: Source + ?Sized>(
    source: &S,
    offset: u64,
    widths: FormatWidths,
) -> Result<Walk, Error> {
    let head_len = source
        .len()
        .saturating_sub(offset)
        .min(OBJECT_HEADER_PREFIX_MAX_LEN.to_u64())
        .to_usize()?;
    let head = source.read_metadata_at(offset, head_len)?;
    let prefix = ObjectHeaderPrefix::parse(&head)?;
    let chunk0_msg_start = offset + prefix.len.to_u64();
    let chunk0 = source.read_metadata_at(chunk0_msg_start, prefix.chunk0_size.to_usize()?)?;
    let layout = prefix.prefix.layout;
    let mut messages = Vec::new();
    let mut continuations: Vec<(u64, usize)> = Vec::new();
    walk_messages(
        &chunk0,
        chunk0_msg_start,
        offset,
        chunk0_msg_start + chunk0.len().to_u64(),
        layout,
        widths,
        &mut messages,
        &mut continuations,
    )?;

    let mut guard = 256;
    while let Some((cont_off, cont_len)) = continuations.pop() {
        guard -= 1;
        if guard == 0 {
            return Err(Error::Format(FormatError::NestingDepthExceeded));
        }
        let block = source.read_metadata_at(cont_off, cont_len)?;
        let messages_range = hdf5_pure_format::__private::continuation_block_messages(&block)
            .map_err(|err| {
                Error::Format(match err {
                    FormatError::UnexpectedEof { .. } => FormatError::InvalidObjectHeaderSignature,
                    err => err,
                })
            })?;
        let region = block.get(messages_range.clone()).ok_or_else(|| {
            FormatError::Internal("a continuation block's messages lie outside the block".into())
        })?;
        walk_messages(
            region,
            cont_off + messages_range.start.to_u64(),
            cont_off,
            cont_off + messages_range.end.to_u64(),
            layout,
            widths,
            &mut messages,
            &mut continuations,
        )?;
    }

    Ok(Walk { messages })
}

/// Appends each message of one object header chunk to `messages`, except a
/// continuation message, whose block location it appends to `continuations`.
///
/// `chunk` holds the message region of the chunk, whose records are in
/// `layout`, and `base` is the absolute file offset of `chunk[0]`, so recorded
/// message offsets are absolute.
///
/// # Errors
///
/// Returns [`Error::Format`] if the body of a record runs past the end of
/// `chunk`, or if a continuation message ends inside its fields or stores the
/// undefined address.
#[allow(clippy::too_many_arguments)]
fn walk_messages(
    chunk: &[u8],
    base: u64,
    chunk_start: u64,
    chunk_msg_end: u64,
    layout: MessageRecordLayout,
    widths: FormatWidths,
    messages: &mut Vec<WalkedMessage>,
    continuations: &mut Vec<(u64, usize)>,
) -> Result<(), Error> {
    let mut msg_start = 0;
    while let Some(record) = layout.next_message(chunk, msg_start)? {
        let MessageRecord {
            msg_type,
            flags,
            body,
            body_range,
            ..
        } = record;
        if msg_type == MessageType::OBJECT_HEADER_CONTINUATION {
            let continuation = ObjectHeaderContinuation::parse_with_widths(body, widths)?;
            continuations.push((
                continuation.address().get(),
                continuation.length().to_usize()?,
            ));
        } else {
            messages.push(WalkedMessage {
                msg_type,
                flags,
                data_off: base + body_range.start.to_u64(),
                size: body.len(),
                chunk_start,
                chunk_msg_end,
            });
        }
        msg_start = body_range.end;
    }
    Ok(())
}

/// The signature, version, and client-ID width of a version 0 data or secondary block.
///
/// Defined in "The Extensible Array Index" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
const EA_BLOCK_PREFIX_SIZE: u64 = 6;
/// The checksum width of a version 0 Extensible Array block or data-block page.
///
/// Defined in "The Extensible Array Index" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
const EA_CHECKSUM_SIZE: u64 = 4;

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::cell::RefCell;

    use hdf5_pure_format::__private::LAYOUT_PROPERTIES_OFFSET;
    use hdf5_pure_format::__private::LayoutVersion;
    use rstest::rstest;
    use test_util::image::Image;
    use test_util::object_header::Message;
    use test_util::object_header::MessageType as RecordType;
    use test_util::object_header::v2 as v2_bytes;
    use test_util::widths::Widths;

    use super::*;
    use crate::access_mode::AccessMode;
    use crate::filters;
    use crate::group_v2;
    use crate::object_path::ObjectPath;
    use crate::source::BytesSource;
    use crate::superblock::LocatedSuperblock;
    use crate::superblock::Superblock;
    use crate::writer::FileBuilder;

    #[rstest]
    #[case::first_direct(4, 4, 16, 1, 0, None, 0)]
    #[case::last_in_first_direct(19, 4, 16, 1, 0, None, 0)]
    #[case::next_direct(20, 20, 32, 1, 0, None, 1)]
    #[case::first_secondary(244, 244, 64, 4, 240, Some(0), 0)]
    #[case::last_in_secondary(499, 436, 64, 4, 240, Some(0), 3)]
    #[case::next_secondary(500, 500, 128, 4, 496, Some(1), 0)]
    #[case::first_paged(131_060, 131_060, 2_048, 64, 131_056, Some(9), 0)]
    #[case::next_paged_block(133_108, 133_108, 2_048, 64, 131_056, Some(9), 1)]
    fn data_block_locations_match_expected_ranges(
        #[case] element: u64,
        #[case] start: u64,
        #[case] elements: u64,
        #[case] blocks: u64,
        #[case] offset: u64,
        #[case] secondary: Option<usize>,
        #[case] local: usize,
    ) {
        let header = range_header(32, 4, 16);
        let geom = ExtensibleArrayGeometry::from_header(&header, OffsetWidth::Eight).unwrap();
        let region = locate_data_block(&geom, 4, element).unwrap();
        assert_eq!(
            (
                region.db_start,
                region.dblk_nelmts,
                region.ndblks,
                region.sb_block_offset
            ),
            (start, elements, blocks, offset)
        );
        match (region.parent, secondary) {
            (Parent::IndexDirect { ordinal }, None) => assert_eq!(ordinal, local),
            (Parent::Super { sblk_j, dblk_local }, Some(expected)) => {
                assert_eq!((sblk_j, dblk_local), (expected, local));
            }
            _ => panic!("unexpected parent block"),
        }
    }

    #[rstest]
    #[case::span(64, 4, 16, u64::MAX - 11)]
    #[case::endpoint(63, 16, 16, u64::MAX)]
    fn data_block_range_overflow_is_rejected(
        #[case] bits: u8,
        #[case] inline: u8,
        #[case] minimum: u8,
        #[case] element: u64,
    ) {
        let header = range_header(bits, inline, minimum);
        let geom = ExtensibleArrayGeometry::from_header(&header, OffsetWidth::Eight).unwrap();
        let inline = u64::from(inline);
        let error = match locate_data_block(&geom, inline, element) {
            Ok(_) => panic!("expected a range error"),
            Err(error) => error,
        };
        assert!(
            matches!(
                error,
                Error::Format(FormatError::InvalidExtensibleArrayGeometry)
            ),
            "{error:?}"
        );
    }

    #[test]
    fn the_last_representable_secondary_range_has_the_expected_location() {
        let header = range_header(64, 4, 16);
        let geom = ExtensibleArrayGeometry::from_header(&header, OffsetWidth::Eight).unwrap();
        let end = (1u128 << 64) - 12;
        let elements = 1u64 << 34;
        let region = locate_data_block(&geom, 4, u64::try_from(end - 1).unwrap()).unwrap();
        assert_eq!(u128::from(region.db_start) + u128::from(elements), end);
        assert_eq!(region.dblk_nelmts, elements);
        assert_eq!(region.ndblks, 1 << 29);
        assert_eq!(region.sb_block_offset, (1 << 63) - 16);
        let Parent::Super { sblk_j, dblk_local } = region.parent else {
            panic!("expected a secondary block");
        };
        assert_eq!((sblk_j, dblk_local), (55, (1 << 29) - 1));
    }

    #[rstest]
    #[case::inline(4, 0, 0, 0, 4)]
    #[case::first_direct(5, 0, 0, 1, 20)]
    #[case::secondary(245, 1, 54, 7, 308)]
    fn append_statistics_match_expected_allocations(
        #[case] chunks: u64,
        #[case] secondary_count: u64,
        #[case] secondary_bytes: u64,
        #[case] data_count: u64,
        #[case] elements: u64,
        #[values(8, 14)] stride: usize,
    ) {
        let store = WindowProbeStore::open(build_unlimited(4, 1));
        let (mut loc, _) = locate(&store);
        loc.ea_elem_size = stride;
        assert_eq!(
            loc.append_stats(chunks).unwrap(),
            ExtensibleArrayStats {
                nsuper_blks: secondary_count,
                super_blk_size: secondary_bytes,
                ndata_blks: data_count,
                data_blk_size: (elements - 4) * stride.to_u64() + data_count * 22,
                max_idx_set: chunks,
                nelmts: elements,
            }
        );
    }

    #[rstest]
    #[case::span(64, u64::MAX - 11)]
    #[cfg_attr(target_pointer_width = "64", case::statistics(63, 1u64 << 61))]
    fn append_planning_rejects_unrepresentable_ranges_before_reading(
        #[case] bits: u8,
        #[case] dimension: u64,
    ) {
        let store = WindowProbeStore::open(build_unlimited(4, 1));
        let original = store.data.clone();
        let (mut loc, datatype) = locate(&store);
        let header = range_header(bits, 4, 16);
        loc.geom = ExtensibleArrayGeometry::from_header(&header, OffsetWidth::Eight).unwrap();
        loc.ea_bits = header.bit_geometry();
        loc.current_dim = dimension;
        store.reset_counters();
        let error = match plan_ea_append(
            &store,
            &loc,
            &datatype,
            &[1],
            loc.elem_bytes,
            None,
            true,
            &9i32.to_le_bytes(),
            1,
            FillPattern::ZERO,
        ) {
            Ok(_) => panic!("expected a range error"),
            Err(error) => error,
        };
        assert!(
            matches!(
                error,
                Error::Format(FormatError::InvalidExtensibleArrayGeometry)
            ),
            "{error:?}"
        );
        assert_eq!(store.total_read.get(), 0);
        assert_eq!(store.allocations.borrow().as_slice(), []);
        assert_eq!(store.data, original);
    }

    fn range_header(bits: u8, inline: u8, minimum: u8) -> ExtensibleArrayHeader {
        let mut bytes = EA_HEADER_SIGNATURE.to_vec();
        bytes.extend_from_slice(&[
            EA_HEADER_VERSION,
            EA_CLIENT_ID_UNFILTERED,
            8,
            bits,
            inline,
            minimum,
            4,
            10,
        ]);
        bytes.resize(ExtensibleArrayHeader::serialized_size(8, 8), 0);
        let len = bytes.len();
        checksum::stamp_trailing(&mut bytes, 0, len);
        ExtensibleArrayHeader::parse(&bytes, 0, 8, 8).unwrap()
    }

    #[rstest]
    #[case::inline(1, 2, false, false)]
    #[case::direct_data_block(4, 5, true, false)]
    #[case::super_block(4, 324, true, true)]
    fn append_distinguishes_chunk_and_index_allocation_intent(
        #[case] start: i32,
        #[case] end: i32,
        #[case] data_block: bool,
        #[case] super_block: bool,
    ) {
        let mut store = WindowProbeStore::open(build_unlimited(start, 1));
        let (mut loc, datatype) = locate(&store);
        store.reset_counters();
        alloc_probe::take();
        append_i32s(&mut store, &mut loc, &datatype, start..end);
        let (data_blocks, super_blocks) = alloc_probe::take();
        assert_eq!(
            (data_blocks > 0, super_blocks > 0),
            (data_block, super_block)
        );
        let mut expected = vec![AllocationClass::Raw; (end - start) as usize];
        expected.extend(core::iter::repeat_n(
            AllocationClass::Metadata,
            data_blocks + super_blocks,
        ));
        assert_eq!(*store.allocations.borrow(), expected);
        assert_eq!(read_i32s(&store, &loc), (0..end).collect::<Vec<_>>());
        store.reset_counters();
        assert_eq!(*store.allocations.borrow(), []);
    }

    /// An in-test [`Store`] over a `Vec<u8>` that records every read window, so
    /// the tests below can assert the engine's bounded-memory contract: the
    /// append path never reads more than a bounded window at once, no matter how
    /// large the file is. This is the seam contract issue #147's mirror-less
    /// bounded backend relies on.
    struct WindowProbeStore {
        data: Vec<u8>,
        superblock: Superblock,
        widths: FormatWidths,
        sb_sig_off: usize,
        max_read: Cell<usize>,
        total_read: Cell<usize>,
        reads: RefCell<Vec<(u64, usize)>>,
        allocations: RefCell<Vec<AllocationClass>>,
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
        /// A free region within the recorded file length, created by
        /// [`with_free_tail`](Self::with_free_tail). The store allocates from it before appending.
        free: Option<(u64, u64)>,
        /// The number of allocations the store served from `free`.
        reuses: Cell<usize>,
    }

    impl WindowProbeStore {
        fn open(data: Vec<u8>) -> Self {
            let LocatedSuperblock {
                address,
                superblock,
                widths,
            } = crate::superblock::read(data.as_slice()).unwrap();
            let sb_sig_off = usize::try_from(address.get()).unwrap();
            Self {
                data,
                superblock,
                widths,
                sb_sig_off,
                max_read: Cell::new(0),
                total_read: Cell::new(0),
                reads: RefCell::new(Vec::new()),
                allocations: RefCell::new(Vec::new()),
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

        /// Extends the image when the free region cannot fit an allocation.
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
            self.allocations.borrow_mut().clear();
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
        fn widths(&self) -> FormatWidths {
            self.widths
        }
        fn recorded_eof(&self) -> u64 {
            self.superblock.eof_address
        }
        fn allocate(&mut self, bytes: &[u8], class: AllocationClass) -> Result<u64, Error> {
            self.allocations.get_mut().push(class);
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
            let bytes =
                hdf5_pure_format::__private::serialize_superblock(&self.superblock).unwrap();
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
        let result =
            Located::locate_at(store, dataset_address(store), Error::AppendUnsupported).unwrap();
        let (dt_off, dt_size) = result.spans.datatype;
        let dt_bytes = store.read_metadata_at(dt_off, dt_size).unwrap();
        let (datatype, _) = hdf5_pure_format::__private::parse_datatype(&dt_bytes).unwrap();
        (result.located, datatype)
    }

    fn dataset_address(store: &WindowProbeStore) -> u64 {
        group_v2::resolve_path_any(
            &store.data,
            AccessMode::ReadOnly,
            &store.superblock,
            crate::file_writer::WIDTHS,
            AddressContext::from_effective_base(store.superblock.base_address),
            &ObjectPath::parse("d"),
        )
        .unwrap()
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
        apply_ea_append(store, loc, plan, 4).unwrap();
    }

    /// Returns the bytes of the file from a fresh data block of `dblk_nelmts`
    /// elements on, and the prefix that block opens with.
    fn fresh_data_block(dblk_nelmts: u64) -> (Vec<u8>, Vec<u8>) {
        let mut store = WindowProbeStore::open(build_unlimited(4, 1));
        let (loc, _) = locate(&store);
        let at = loc
            .alloc_undef_data_block(&mut store, dblk_nelmts, 32)
            .unwrap()
            .get()
            .to_usize()
            .unwrap();
        let mut prefix = b"EADB".to_vec();
        prefix.push(0);
        prefix.push(loc.client_id);
        prefix.extend_from_slice(&loc.ea_addr.get().to_le_bytes());
        prefix.extend_from_slice(&32u64.to_le_bytes()[..loc.ea_bits.block_offset_width().bytes()]);
        (store.data[at..].to_vec(), prefix)
    }

    #[test]
    fn the_header_walk_reads_the_prefix_window_then_the_chunk_0_messages() {
        let store = WindowProbeStore::open(build_unlimited(4, 1));
        let oh_addr = group_v2::resolve_path_any(
            &store.data,
            AccessMode::ReadOnly,
            &store.superblock,
            crate::file_writer::WIDTHS,
            AddressContext::from_effective_base(store.superblock.base_address),
            &ObjectPath::parse("d"),
        )
        .unwrap();
        store.reset_counters();

        let walk = walk_v2_object_header(&store, oh_addr, store.widths).unwrap();

        let reads = store.reads.borrow().clone();
        let [(head_at, head_len), (messages_at, messages_len)] = reads[..] else {
            panic!("expected the prefix window and the chunk 0 messages, got {reads:?}");
        };
        assert_eq!((head_at, head_len), (oh_addr, OBJECT_HEADER_PREFIX_MAX_LEN));
        let record_prefix = MessageRecordLayout::PLAIN.prefix_len().to_u64();
        assert_eq!(
            walk.messages
                .iter()
                .map(|message| (message.msg_type, message.chunk_msg_end))
                .collect::<Vec<_>>(),
            [
                MessageType::DATATYPE,
                MessageType::DATASPACE,
                MessageType::FILL_VALUE,
                MessageType::DATA_LAYOUT,
            ]
            .map(|msg_type| (msg_type, messages_at + messages_len.to_u64()))
        );
        assert_eq!(walk.messages[0].data_off, messages_at + record_prefix);
    }

    #[rstest::rstest]
    #[case::a_version_other_than_2(header_of_version_1())]
    #[case::a_prefix_cut_short(header_cut_inside_its_prefix())]
    #[case::a_chunk_0_past_the_end_of_the_file(chunk_0_past_the_end_of_the_file())]
    #[case::a_record_past_the_end_of_chunk_0(record_past_the_end_of_chunk_0())]
    #[case::a_short_continuation_message(continuation_message_shorter_than_its_fields())]
    #[case::a_continuation_to_the_undefined_address(continuation_message_to_the_undefined_address())]
    #[case::a_short_continuation_block(continuation_block_shorter_than_its_signature_and_checksum())]
    #[case::a_continuation_block_without_its_signature(continuation_block_without_its_signature())]
    fn a_damaged_header_fails_the_walk(#[case] (bytes, expected): (Vec<u8>, FormatError)) {
        let Err(err) =
            walk_v2_object_header(&BytesSource::new(&bytes), 0, crate::file_writer::WIDTHS)
        else {
            panic!("expected the walk to fail with {expected:?}");
        };
        let Error::Format(format_error) = err else {
            panic!("expected a format error, got {err:?}");
        };
        assert_eq!(format_error, expected);
    }

    fn header_of_version_1() -> (Vec<u8>, FormatError) {
        let mut header = one_message_header().build();
        header[4] = 1;
        (header, FormatError::InvalidObjectHeaderVersion(1))
    }

    fn header_cut_inside_its_prefix() -> (Vec<u8>, FormatError) {
        let header = one_message_header().build()[..5].to_vec();
        (
            header,
            FormatError::UnexpectedEof {
                expected: 6,
                available: 5,
            },
        )
    }

    fn chunk_0_past_the_end_of_the_file() -> (Vec<u8>, FormatError) {
        let mut header = one_message_header()
            .flags(v2_bytes::HeaderFlags::CHUNK_SIZE_WIDTH)
            .build();
        header[6..14].copy_from_slice(&(1u64 << 31).to_le_bytes());
        let available = header.len();
        (
            header,
            FormatError::UnexpectedSourceEof {
                expected: 14 + (1 << 31),
                available: available.to_u64(),
            },
        )
    }

    fn record_past_the_end_of_chunk_0() -> (Vec<u8>, FormatError) {
        let mut header = one_message_header().build();
        header[8] = 0xFF;
        (
            header,
            FormatError::UnexpectedEof {
                expected: 4 + 0xFF,
                available: 8,
            },
        )
    }

    fn continuation_message_shorter_than_its_fields() -> (Vec<u8>, FormatError) {
        let header = v2_bytes::Header::new()
            .message(Message::new(
                RecordType::OBJECT_HEADER_CONTINUATION,
                &[0; 4],
            ))
            .build();
        (
            header,
            FormatError::UnexpectedEof {
                expected: 8,
                available: 4,
            },
        )
    }

    fn continuation_message_to_the_undefined_address() -> (Vec<u8>, FormatError) {
        let header = v2_bytes::Header::new()
            .message(Message::undefined_continuation(64, Widths::EIGHT))
            .build();
        (header, FormatError::UndefinedContinuationAddress)
    }

    fn continuation_block_shorter_than_its_signature_and_checksum() -> (Vec<u8>, FormatError) {
        (
            image_with_continuation_block(
                &[v2_bytes::CONTINUATION_SIGNATURE.as_slice(), &[0; 3]].concat(),
            ),
            FormatError::InvalidObjectHeaderSignature,
        )
    }

    fn continuation_block_without_its_signature() -> (Vec<u8>, FormatError) {
        (
            image_with_continuation_block(&[v2_bytes::SIGNATURE.as_slice(), &[0; 4]].concat()),
            FormatError::InvalidObjectHeaderSignature,
        )
    }

    fn one_message_header() -> v2_bytes::Header {
        v2_bytes::Header::new().message(Message::new(RecordType::DATASPACE, &[1, 2, 3, 4]))
    }

    fn image_with_continuation_block(block: &[u8]) -> Vec<u8> {
        const BLOCK_AT: usize = 0x100;
        let header = v2_bytes::Header::new()
            .continuation(BLOCK_AT, block, Widths::EIGHT)
            .build();
        let mut image = Image::starting_with(&header);
        image.place(BLOCK_AT, block);
        image.build()
    }

    #[rstest]
    #[case(0)]
    #[case(7)]
    #[case(9)]
    #[case(255)]
    fn an_unfiltered_element_width_other_than_an_address_is_readable_but_append_is_unsupported(
        #[case] element_size: u8,
    ) {
        let mut data = build_unlimited(4, 1);
        let header_at = data
            .windows(4)
            .position(|window| window == b"EAHD")
            .unwrap();
        data[header_at + 6] = element_size;
        test_util::checksum::restamp(
            &mut data,
            header_at,
            ExtensibleArrayHeader::serialized_size(8, 8),
        );
        assert_eq!(
            crate::File::from_bytes(data.clone())
                .unwrap()
                .dataset("d")
                .unwrap()
                .read_i32()
                .unwrap(),
            [0, 1, 2, 3]
        );
        let store = WindowProbeStore::open(data);
        let oh_addr = group_v2::resolve_path_any(
            &store.data,
            AccessMode::ReadOnly,
            &store.superblock,
            crate::file_writer::WIDTHS,
            AddressContext::from_effective_base(store.superblock.base_address),
            &ObjectPath::parse("d"),
        )
        .unwrap();

        let Err(err) = Located::locate_at(&store, oh_addr, Error::AppendUnsupported) else {
            panic!("expected an unsupported element width");
        };
        let Error::AppendUnsupported(reason) = err else {
            panic!("expected AppendUnsupported, got {err:?}");
        };
        assert_eq!(
            reason,
            "malformed unfiltered extensible-array element width"
        );
    }

    #[rstest]
    fn locating_an_array_reads_its_complete_header_once(#[values(false, true)] filtered: bool) {
        let store = if filtered {
            EdgeFilter::Shuffle.store()
        } else {
            WindowProbeStore::open(build_unlimited(4, 1))
        };
        let address = dataset_address(&store);
        store.reset_counters();
        let located = Located::locate_at(&store, address, Error::AppendUnsupported)
            .unwrap()
            .located;
        assert_eq!(
            store
                .reads
                .borrow()
                .iter()
                .filter(|(at, _)| *at == located.ea_addr.get())
                .copied()
                .collect::<Vec<_>>(),
            [(
                located.ea_addr.get(),
                ExtensibleArrayHeader::serialized_size_with_widths(store.widths),
            )]
        );
    }

    #[test]
    fn a_filtered_header_with_an_empty_size_field_parses_and_is_unsupported_for_append() {
        let mut store = EdgeFilter::Shuffle.store();
        let address = dataset_address(&store);
        let header = Located::locate_at(&store, address, Error::AppendUnsupported)
            .unwrap()
            .located
            .ea_addr
            .get()
            .to_usize()
            .unwrap();
        store.data[header + EA_HEADER_ELEMENT_SIZE] =
            store.widths.offsets.get() + EA_FILTER_MASK_BYTES;
        test_util::checksum::restamp(
            &mut store.data,
            header,
            ExtensibleArrayHeader::serialized_size_with_widths(store.widths),
        );
        let array = crate::extensible_array::read_header(
            &store,
            store.widths,
            StoredAddress::new(header.to_u64()),
        )
        .unwrap();
        assert_eq!(
            array.header().element_size,
            store.widths.offsets.get() + EA_FILTER_MASK_BYTES
        );
        let Err(error) = Located::locate_at(&store, address, Error::AppendUnsupported) else {
            panic!("expected an unsupported element width");
        };
        let Error::AppendUnsupported(reason) = error else {
            panic!("expected AppendUnsupported, got {error:?}");
        };
        assert_eq!(reason, "malformed filtered extensible-array element width");
    }

    #[test]
    fn a_layout_element_size_other_than_the_datatype_size_is_rejected() {
        let mut store = WindowProbeStore::open(build_unlimited(6, 4));
        let oh_addr = crate::File::from_bytes(store.data.clone())
            .unwrap()
            .dataset("d")
            .unwrap()
            .header_address()
            .unwrap();
        let walk = walk_v2_object_header(&store, oh_addr, store.widths).unwrap();
        let datatype = walk
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::DATATYPE)
            .unwrap();
        let wider =
            hdf5_pure_format::__private::serialize_datatype(&crate::type_builders::make_i64_type());
        assert_eq!(wider.len(), datatype.size);
        store
            .publish_checksummed(
                datatype.chunk_start,
                datatype.chunk_msg_end,
                datatype.data_off,
                &wider,
            )
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("element_size.h5");
        std::fs::write(&path, &store.data).unwrap();

        let file = crate::File::open_rw(&path).unwrap();
        let err = file.dataset("d").unwrap().append(&[6_i64]).unwrap_err();
        let Error::Format(FormatError::ChunkElementSizeMismatch { datatype, layout }) = err else {
            panic!("expected ChunkElementSizeMismatch, got {err:?}");
        };
        assert_eq!((datatype, layout), (8, 4));
        file.close().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), store.data);
    }

    #[test]
    fn a_fresh_unpaged_data_block_holds_only_undefined_elements() {
        let (block, mut expected) = fresh_data_block(16);
        expected.extend_from_slice(&[0xFF; 16 * 8]);
        test_util::checksum::append(&mut expected);
        assert_eq!(block[..expected.len()], expected);
    }

    #[test]
    fn a_fresh_paged_data_block_checksums_its_prefix_and_each_page() {
        let (block, mut expected) = fresh_data_block(2048);
        test_util::checksum::append(&mut expected);
        let mut page = vec![0xFF; 1024 * 8];
        test_util::checksum::append(&mut page);
        expected.extend_from_slice(&page);
        expected.extend_from_slice(&page);
        assert_eq!(block[..expected.len()], expected);
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
    /// The test store uses one free region for both allocation classes. The fixture's region
    /// is large enough for the chunks and index blocks throughout the append sequence.
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

    #[test]
    fn a_filtered_element_width_beyond_the_geometry_limit_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wide_elem.h5");
        let mut b = crate::writer::FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..4096).collect::<Vec<_>>())
            .with_shape(&[4096])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[4])
            .with_deflate(DeflateLevel::One);
        b.write(&path).unwrap();

        // The filtered element limit is an 8-byte address, an 8-byte size, and a 4-byte mask.
        let mut bytes = std::fs::read(&path).unwrap();
        let at = bytes
            .windows(4)
            .position(|w| w == b"EAHD")
            .expect("a filtered unlimited dataset is indexed by an extensible array");
        assert_eq!(bytes[at + 6], 14, "the writer's own element width moved");
        bytes[at + 6] = 30;
        crate::checksum::stamp_trailing(&mut bytes, at, 12 + 6 * 8 + 8 + 4);
        std::fs::write(&path, &bytes).unwrap();

        let f = crate::reader::File::open_rw(&path).unwrap();
        let err = f
            .dataset("d")
            .and_then(|mut d| d.append(&[1i32, 2, 3, 4]))
            .unwrap_err();
        assert!(
            matches!(
                err,
                Error::Format(FormatError::InvalidExtensibleArrayGeometry)
            ),
            "{err:?}"
        );
        drop(f);
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
    }

    #[rstest]
    #[case::raw_tail_stays_partial(true, &[106], &[(Stored::Raw, [104, 105, 106, EDGE_FILL])])]
    #[case::raw_tail_becomes_full(true, &[106, 107], &[(Stored::FiltersSkipped, [104, 105, 106, 107])])]
    #[case::raw_tail_becomes_full_before_a_raw_tail(
        true,
        &[106, 107, 108],
        &[
            (Stored::FiltersSkipped, [104, 105, 106, 107]),
            (Stored::Raw, [108, EDGE_FILL, EDGE_FILL, EDGE_FILL]),
        ],
    )]
    #[case::filtered_tail_stays_partial(
        false,
        &[106],
        &[(Stored::Filtered, [104, 105, 106, EDGE_FILL])],
    )]
    #[case::filtered_tail_becomes_full(
        false,
        &[106, 107],
        &[(Stored::Filtered, [104, 105, 106, 107])],
    )]
    #[case::filtered_tail_becomes_full_before_a_filtered_tail(
        false,
        &[106, 107, 108],
        &[
            (Stored::Filtered, [104, 105, 106, 107]),
            (Stored::Filtered, [108, EDGE_FILL, EDGE_FILL, EDGE_FILL]),
        ],
    )]
    fn every_publication_state_of_an_edge_append_reads_under_its_extent(
        #[case] raw_edges: bool,
        #[case] appended: &[i32],
        #[case] written: &[(Stored, [i32; 4])],
        #[values(EdgeFilter::Deflate, EdgeFilter::Shuffle, EdgeFilter::ShuffleDeflate)]
        filter: EdgeFilter,
        #[values(1, 2, 3, 4)] max_phase: u8,
    ) {
        let mut store = edge_store(filter, raw_edges);
        let before = stored_chunks(&store.data);
        append_to_edge_store(&mut store, appended, max_phase).unwrap();

        let written = filter.stored_chunks_after_the_kept_one(written);
        let (chunks, values) = match max_phase {
            1 => (before, EDGE_INITIAL.to_vec()),
            2 | 3 => (
                [&before[..1], &written[..1]].concat(),
                EDGE_INITIAL.to_vec(),
            ),
            _ => (
                [&before[..1], &written].concat(),
                [&EDGE_INITIAL[..], appended].concat(),
            ),
        };
        assert_eq!(stored_chunks(&store.data), chunks);
        let file = crate::File::from_bytes(store.data).unwrap();
        assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), values);
    }

    #[rstest]
    #[case::short_raw_tail(true, 12)]
    #[case::long_raw_tail(true, 20)]
    #[case::short_filtered_tail(false, 12)]
    #[case::long_filtered_tail(false, 20)]
    fn an_old_tail_of_another_decoded_length_is_rejected(
        #[case] raw_edges: bool,
        #[case] decoded_len: usize,
    ) {
        let store = edge_store_decoding_to(raw_edges, decoded_len);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edge.h5");
        std::fs::write(&path, &store.data).unwrap();

        let file = crate::File::open_rw(&path).unwrap();
        let err = file.dataset("d").unwrap().append(&[106_i32]).unwrap_err();
        let Error::Format(FormatError::DataSizeMismatch { expected, actual }) = err else {
            panic!("expected DataSizeMismatch, got {err:?}");
        };
        assert_eq!((expected, actual), (16, decoded_len));
        file.close().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), store.data);
    }

    #[rstest]
    #[case::raw_tail_stays_partial(true, &[106], &[(Stored::Raw, [104, 105, 106, EDGE_FILL])])]
    #[case::raw_tail_becomes_full(true, &[106, 107], &[(Stored::Filtered, [104, 105, 106, 107])])]
    #[case::raw_tail_becomes_full_before_a_raw_tail(
        true,
        &[106, 107, 108],
        &[
            (Stored::Filtered, [104, 105, 106, 107]),
            (Stored::Raw, [108, EDGE_FILL, EDGE_FILL, EDGE_FILL]),
        ],
    )]
    #[case::filtered_tail_stays_partial(
        false,
        &[106],
        &[(Stored::Filtered, [104, 105, 106, EDGE_FILL])],
    )]
    #[case::filtered_tail_becomes_full(
        false,
        &[106, 107],
        &[(Stored::Filtered, [104, 105, 106, 107])],
    )]
    #[case::filtered_tail_becomes_full_before_a_filtered_tail(
        false,
        &[106, 107, 108],
        &[
            (Stored::Filtered, [104, 105, 106, 107]),
            (Stored::Filtered, [108, EDGE_FILL, EDGE_FILL, EDGE_FILL]),
        ],
    )]
    fn a_staged_edge_append_keeps_the_layout_flags_and_stores_each_written_chunk_for_the_new_extent(
        #[case] raw_edges: bool,
        #[case] appended: &[i32],
        #[case] written: &[(Stored, [i32; 4])],
        #[values(EdgeFilter::Deflate, EdgeFilter::Shuffle, EdgeFilter::ShuffleDeflate)]
        filter: EdgeFilter,
    ) {
        let store = edge_store(filter, raw_edges);
        let before = stored_chunks(&store.data);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edge.h5");
        std::fs::write(&path, &store.data).unwrap();

        let file = crate::File::open_rw(&path).unwrap();
        let mut dataset = file.dataset("d").unwrap();
        dataset
            .append_staged(|builder| {
                builder.append(appended);
            })
            .unwrap();
        drop(dataset);
        file.close().unwrap();

        let data = std::fs::read(&path).unwrap();
        assert_eq!(
            stored_chunks(&data),
            [
                &before[..1],
                &filter.stored_chunks_after_the_kept_one(written)
            ]
            .concat()
        );
        let flags = if raw_edges {
            crate::data_layout::DONT_FILTER_PARTIAL_BOUND_CHUNKS
        } else {
            0
        };
        assert_eq!(
            locate(&WindowProbeStore::open(data.clone())).0.flags,
            ChunkedLayoutFlags::new(flags)
        );
        let file = crate::File::from_bytes(data).unwrap();
        assert_eq!(
            file.dataset("d").unwrap().read_i32().unwrap(),
            [&EDGE_INITIAL[..], appended].concat()
        );
    }

    #[rstest]
    #[case::short_raw_tail(true, 12)]
    #[case::long_raw_tail(true, 20)]
    #[case::short_filtered_tail(false, 12)]
    #[case::long_filtered_tail(false, 20)]
    fn a_staged_append_onto_an_old_tail_of_another_decoded_length_is_rejected(
        #[case] raw_edges: bool,
        #[case] decoded_len: usize,
    ) {
        let store = edge_store_decoding_to(raw_edges, decoded_len);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("edge.h5");
        std::fs::write(&path, &store.data).unwrap();

        let file = crate::File::open_rw(&path).unwrap();
        let mut dataset = file.dataset("d").unwrap();
        dataset
            .append_staged(|builder| {
                builder.append(&[106_i32]);
            })
            .unwrap();
        let err = file.commit().unwrap_err();
        let Error::Format(FormatError::DataSizeMismatch { expected, actual }) = err else {
            panic!("expected DataSizeMismatch, got {err:?}");
        };
        assert_eq!((expected, actual), (16, decoded_len));
        drop(dataset);
        drop(file);
        assert_eq!(std::fs::read(&path).unwrap(), store.data);
    }

    #[derive(Clone, Copy, Debug)]
    enum EdgeFilter {
        Deflate,
        Shuffle,
        ShuffleDeflate,
    }

    impl EdgeFilter {
        fn pipeline(self) -> FilterPipeline {
            let store = self.store();
            edit::locate_dataset_state(&store, dataset_address(&store))
                .unwrap()
                .pipeline
                .unwrap()
        }

        fn every_filter_skipped(self) -> u32 {
            match self {
                Self::Deflate | Self::Shuffle => 0b1,
                Self::ShuffleDeflate => 0b11,
            }
        }

        fn stored_chunks_after_the_kept_one(
            self,
            written: &[(Stored, [i32; 4])],
        ) -> Vec<(Vec<u64>, u32, Vec<u8>)> {
            let pipeline = self.pipeline();
            (4..)
                .step_by(4)
                .zip(written)
                .map(|(offset, (stored, values))| {
                    let logical = i32_bytes(values);
                    let (filter_mask, bytes) = match stored {
                        Stored::Filtered => (
                            0,
                            filters::compress_chunk(
                                &logical,
                                &pipeline,
                                ChunkContext::basic(&[4], 4),
                            )
                            .unwrap(),
                        ),
                        Stored::FiltersSkipped => (self.every_filter_skipped(), logical),
                        Stored::Raw => (0, logical),
                    };
                    (vec![offset], filter_mask, bytes)
                })
                .collect()
        }

        fn store(self) -> WindowProbeStore {
            let mut builder = FileBuilder::new();
            let dataset = builder
                .create_dataset("d")
                .with_i32_data(&EDGE_INITIAL)
                .with_shape(&[6])
                .with_maxshape(&[MaxExtent::Unlimited])
                .with_chunks(&[4])
                .with_fill_value(EDGE_FILL);
            match self {
                Self::Deflate => dataset.with_deflate(DeflateLevel::Six),
                Self::Shuffle => dataset.with_shuffle(),
                Self::ShuffleDeflate => dataset.with_shuffle().with_deflate(DeflateLevel::Six),
            };
            WindowProbeStore::open(builder.finish().unwrap())
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Stored {
        Filtered,
        FiltersSkipped,
        Raw,
    }

    fn edge_store(filter: EdgeFilter, raw_edges: bool) -> WindowProbeStore {
        let mut store = filter.store();
        if raw_edges {
            let walk =
                walk_v2_object_header(&store, dataset_address(&store), store.widths).unwrap();
            let layout = walk
                .messages
                .iter()
                .find(|m| m.msg_type == MessageType::DATA_LAYOUT)
                .unwrap();
            assert_eq!(
                store.data[layout.data_off.to_usize().unwrap()],
                LayoutVersion::Four.get()
            );
            store
                .publish_checksummed(
                    layout.chunk_start,
                    layout.chunk_msg_end,
                    layout.data_off + LAYOUT_PROPERTIES_OFFSET.to_u64(),
                    &[crate::data_layout::DONT_FILTER_PARTIAL_BOUND_CHUNKS],
                )
                .unwrap();
            replace_edge_record(&mut store, &i32_bytes(&[104, 105, EDGE_FILL, EDGE_FILL]));
        }
        store
    }

    fn edge_store_decoding_to(raw_edges: bool, decoded_len: usize) -> WindowProbeStore {
        let mut store = edge_store(EdgeFilter::Shuffle, raw_edges);
        let logical = vec![0x5a; decoded_len];
        let stored = if raw_edges {
            logical
        } else {
            filters::compress_chunk(
                &logical,
                &EdgeFilter::Shuffle.pipeline(),
                ChunkContext::basic(&[4], 4),
            )
            .unwrap()
        };
        replace_edge_record(&mut store, &stored);
        store
    }

    fn replace_edge_record(store: &mut WindowProbeStore, stored: &[u8]) {
        let (loc, _) = locate(store);
        let addr = StoredAddress::new(store.allocate(stored, AllocationClass::Raw).unwrap());
        store.patch_superblock_eof().unwrap();
        let rec = ElemRecord {
            addr,
            stored_size: stored.len() as u64,
            filter_mask: 0,
        };
        loc.ea_insert(store, 1, rec).unwrap();
    }

    fn append_to_edge_store(
        store: &mut WindowProbeStore,
        values: &[i32],
        max_phase: u8,
    ) -> Result<(), Error> {
        let mut state = edit::locate_dataset_state(&*store, dataset_address(store)).unwrap();
        let plan = plan_ea_append(
            store,
            &state.loc,
            &state.datatype,
            &state.spatial,
            state.element_size,
            state.pipeline.as_ref(),
            true,
            &i32_bytes(values),
            values.len() as u64,
            state.fill.pattern(state.element_size),
        )?;
        apply_ea_append(store, &mut state.loc, plan, max_phase)
    }

    fn stored_chunks(data: &[u8]) -> Vec<(Vec<u64>, u32, Vec<u8>)> {
        crate::File::from_bytes(data.to_vec())
            .unwrap()
            .dataset("d")
            .unwrap()
            .chunks()
            .unwrap()
            .into_iter()
            .map(|chunk| {
                let at = chunk.address.to_usize().unwrap();
                let end = at + chunk.storage_size.to_usize().unwrap();
                let filter_mask = chunk.filter_mask().raw_bits();
                (chunk.offset, filter_mask, data[at..end].to_vec())
            })
            .collect()
    }

    fn i32_bytes(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    const EDGE_INITIAL: [i32; 6] = [100, 101, 102, 103, 104, 105];
    const EDGE_FILL: i32 = -7;

    // "The Extensible Array Index" in format specification, version 4.0, defines the version 0
    // header element-size field and the four-byte filter mask for version 4 chunked layouts.
    const EA_HEADER_ELEMENT_SIZE: usize = 6;
    const EA_FILTER_MASK_BYTES: u8 = 4;

    // "The Extensible Array Index" in format specification, version 4.0, defines the version 0
    // header signature and version.
    const EA_HEADER_SIGNATURE: &[u8; 4] = b"EAHD";
    const EA_HEADER_VERSION: u8 = 0;
}
