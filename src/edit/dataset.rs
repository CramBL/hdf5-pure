//! Prepared datasets and writes published by an editor commit.
//!
//! Preparation reads existing storage without mutating it. The write plans distinguish in-place
//! bytes from relocating writes and defer variable-length address resolution until allocation.

use std::borrow::Cow;

use core::num::NonZeroUsize;

use hdf5_pure_format::__private::ChunkIndexInfo;
use hdf5_pure_format::__private::ChunkRecord;
use hdf5_pure_format::__private::ChunkedGeometry;
use hdf5_pure_format::__private::IndexSlots;
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::PageType;

use crate::FillTime;
use crate::VariableLengthType;
use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunked_read;
use crate::chunked_read::ChunkDecoding;
use crate::chunked_write::{self, ChunkMeta, ChunkOptions, ChunkProvider, StorageAllocation};
use crate::convert::Narrow;
use crate::data_layout::ChunkedLayoutFlags;
use crate::data_layout::{ChunkIndexLayout, DataLayout};
use crate::dataspace::{Dataspace, DataspaceType, Extent, MaxExtent};
use crate::datatype::Datatype;
use crate::error::{Error, FormatError, OBJECT_HEADER_MESSAGE_MAX};
use crate::file_writer::{self, LENGTH_SIZE, LENGTH_WIDTH, OFFSET_WIDTH};
use crate::fill_value::FillPattern;
use crate::fill_value::PaddingFill;
use crate::filter_pipeline::FilterPipeline;
use crate::filters::ChunkContext;
use crate::filters::FilterScratch;
use crate::message_flags::MessageFlags;
use crate::message_type::MessageType;
use crate::object_path::{LinkNameBuf, ObjectPathBuf};
use crate::prepared_chunk::ChunkEncoding;
use crate::prepared_chunk::PreparedChunk;
use crate::shared_message::DatatypeLocation;
use crate::source::{BaseOffsetSource, Source};
use crate::type_builders::{self, DatasetBuilder, ObjectRefPatch, VlStringStaging};
use crate::width::FormatWidths;

use super::staging::{StagedKind, StagedOp};

use super::{
    AppendBuilder, CopyTree, DenseAttrSet, EditedAttrs, LOSSY_TAIL_REFUSAL, MAX_COPY_DEPTH,
    ObjModel, OhRegion, Placement, WriteEngine,
};

/// Element bytes staged for a value overwrite, together with the
/// variable-length staging (if any) whose global heap collections must be
/// placed  -  and whose addresses patched into the element references in `raw`  -
/// before those bytes reach the file.
///
/// The two halves travel together because resolving them is an **apply-phase**
/// step while planning is a preflight one: placing a collection allocates, and
/// [`commit`](WriteEngine::commit)'s preflight only reads. So a plan carries
/// the bytes unresolved and every consumer resolves them where it writes, with
/// [`resolve_overwrite_bytes`](WriteEngine::resolve_overwrite_bytes) (issue
/// #321).
///
/// Patching never changes `raw`'s length  -  an element reference is fixed-width
/// whatever string it refers to  -  which is what lets the plan be *chosen* from the
/// unresolved bytes. The one place that is not enough is a filtered chunked
/// dataset, whose compressed chunk sizes do depend on the addresses: see
/// [`ChunkPayload::Deferred`].
pub(super) struct OverwriteBytes {
    pub(super) raw: Vec<u8>,
    /// `None` for every overwrite that stages no variable-length data, which is
    /// all of them but a `with_vlen_strings` one.
    pub(super) vlen: Option<VlenOverwrite>,
}

/// The variable-length half of an [`OverwriteBytes`]: the staged collections to
/// place, and the path they are recorded under once placed.
pub(super) struct VlenOverwrite {
    pub(super) staging: VlStringStaging,
    /// The overwritten dataset's path, the key its collections are recorded
    /// under in [`vl_overwrite_heaps`](WriteEngine::vl_overwrite_heaps)  -  and
    /// the key the ones a previous overwrite left are read back from, to be
    /// freed once this commit lands.
    pub(super) path: ObjectPathBuf,
}

impl OverwriteBytes {
    /// Bytes that need no resolving, and so name nothing to record or free.
    pub(super) fn ready(raw: Vec<u8>) -> Self {
        Self { raw, vlen: None }
    }
}

/// How a staged value overwrite (`write_dataset`) will be applied, decided by
/// [`WriteEngine::prepare_write`] during the all-or-nothing preflight.
// `allow` in place of `expect`, which is otherwise this module's habit: the lint
// fires on a 64-bit pointer width and not on i686, where the `Vec` and `usize`
// fields inside `MovingWrite` shrink and the variant gap falls back under
// clippy's threshold. An `expect` is therefore unfulfilled on exactly the two
// 32-bit CI jobs, both of which deny `unfulfilled_lint_expectations`.
#[allow(
    clippy::large_enum_variant,
    reason = "boxing the large variant would cost more than it saves: this enum is returned by \
              `prepare_write` and destructured at its one call site, which moves the `MovingWrite` \
              straight into a `Vec<MovingWrite>` that stores it unboxed either way -- so a box \
              here adds an allocation and a free without removing a single copy"
)]
pub(super) enum WritePlan {
    /// A contiguous dataset whose new data is the same length as its existing,
    /// defined data block: overwrite the bytes straight in place at `data_addr`.
    /// No object header is rewritten.
    ///
    /// The superblock root is not flipped either  -  *unless* the bytes still
    /// carry variable-length staging, which appends a heap collection and so
    /// moves end-of-file, a figure only the superblock records. `commit`'s
    /// fast path excludes exactly that case (issue #321).
    InPlace {
        data_addr: u64,
        bytes: OverwriteBytes,
    },
    /// A chunked dataset overwritten chunk-by-chunk in place: each `(addr, bytes)`
    /// pair is written straight over an existing chunk slot. Used when every new
    /// (re-encoded) chunk is the same byte length as the slot it replaces  -  an
    /// unfiltered chunked overwrite (chunk sizes are fixed by the unchanged shape)
    /// or a filtered one whose re-encoded chunks happen to match. Like
    /// [`InPlace`](WritePlan::InPlace) it does not touch any header and no chunk index, so
    /// the superblock root is not flipped.
    InPlaceChunks { writes: Vec<(u64, Vec<u8>)> },
    /// The dataset's header relocates: a contiguous resize, a compact rewrite, or
    /// a chunked rebuild. The parent group is rebuilt and its link patched. See
    /// [`MovingWrite`].
    Moving(MovingWrite),
}

/// A value overwrite that relocates the dataset's object header  -  a contiguous
/// dataset whose data length changed (or had no data block) or a compact dataset
/// whose inline bytes are replaced. On apply the new data and a rewritten header
/// are written at end-of-file (or into reusable freed space), and the parent
/// group's link is repointed at the new header address.
pub(super) enum MovingWrite {
    /// A contiguous dataset: write `raw` elsewhere, patch the data-layout address
    /// at `addr_off` in the verbatim header `region`, rewrite the header, and free
    /// `old_extent` (the prior data block, if any) after the commit lands.
    Contiguous {
        region: OhRegion,
        addr_off: usize,
        bytes: OverwriteBytes,
        old_extent: Option<(u64, u64)>,
    },
    /// A compact dataset: rebuild the header `region` with the bytes inline.
    Compact {
        region: OhRegion,
        bytes: OverwriteBytes,
    },
    /// A relocating overwrite of a chunked dataset, which rebuilds its whole chunk storage.
    ///
    /// The commit writes a new chunk data block and index with
    /// [`write_chunked_relocatable`](WriteEngine::write_chunked_relocatable), in a freed region
    /// that fits them or at the end of the file. It keeps every message of the verbatim header
    /// `region` except the data layout message, which it replaces, so a dense attribute heap stays
    /// where the Attribute Info message addresses it. After the commit lands, it frees the old
    /// chunk storage of the dataset whose header is at `old_addr`.
    Chunked {
        region: OhRegion,
        /// The chunk geometry and encoding of the dataset, from which the new layout is planned
        /// and a [`ChunkPayload::Deferred`] is prepared.
        overwrite: ChunkedOverwrite,
        maxshape: Option<Vec<MaxExtent>>,
        pipeline_message: Option<Vec<u8>>,
        payload: ChunkPayload,
        old_addr: u64,
    },
    /// A relocating append to a chunked, unlimited dataset indexed by an Extensible Array
    /// (`Dataset::append_staged`).
    ///
    /// The kept chunks stay in place. The commit places `new_chunks`, in freed raw space where they
    /// fit, and builds an Extensible Array over `kept_chunks` and `new_chunks`. It grows the
    /// dataspace message of the verbatim header `region` to `new_dataspace_body`, repoints its
    /// data layout message at the new index, keeps every other message verbatim, and relocates
    /// the header. After the commit lands, it frees the old index and the old header of the
    /// dataset at `old_addr` and the old trailing chunk at `old_tail_extent`, and never the kept
    /// chunks, which the old and the new index share during the commit.
    AppendedChunks {
        region: OhRegion,
        /// The grown dataspace message body (v2-serialized), current axis-0
        /// dimension increased, maximum dimensions (unlimited) preserved.
        new_dataspace_body: Vec<u8>,
        /// The layout flags of the dataset, which the rebuilt layout message stores.
        flags: ChunkedLayoutFlags,
        /// Rank-only spatial chunk dimensions, for the rebuilt v4 layout message.
        chunk_dims_u32: Vec<u32>,
        element_size: NonZeroUsize,
        /// The checked uncompressed chunk byte count used to size index elements.
        raw_size: u64,
        /// Whether the dataset has a filter pipeline, and so whether the rebuilt index stores the
        /// filtered chunk element format.
        has_filters: bool,
        /// Existing complete chunks, in index order, carried by metadata alone  -
        /// their base-relative addresses, on-disk stored sizes, and filter masks
        /// preserved exactly (a nonzero mask from a C/h5py-skipped filter is kept).
        kept_chunks: Vec<ChunkRecord>,
        /// The written chunks in index order, each encoded for a reader at the grown extent: the
        /// rewritten trailing partial chunk first where the dataset has one, then the new chunks.
        new_chunks: Vec<PreparedChunk>,
        /// The dataset header address, for old-index and old-header reclaim.
        old_addr: u64,
        /// The absolute `(addr, len)` of the old trailing partial chunk's data
        /// block when it was rewritten, freed after the commit lands. `None` when
        /// the append was chunk-aligned (no partial chunk to rewrite).
        old_tail_extent: Option<(u64, u64)>,
    },
    /// A dataset-attribute edit (`set_dataset_attr` / `remove_dataset_attr`).
    /// The verbatim header `region` already carries whatever the preflight could
    /// resolve, `attrs` is what [`WriteEngine::write_moving`] still has to place  -
    /// a variable-length attribute's heap collection, or a whole dense set to
    /// rebuild. The rewritten header is relocated and the parent link repointed,
    /// exactly like the other relocating writes  -  but the data-layout message is
    /// preserved verbatim, so the dataset's chunk data and index stay in place,
    /// only the old header is freed.
    AttrEdit {
        region: OhRegion,
        attrs: EditedAttrs,
        /// Whether the old header names dense attribute indexes made dead by this edit.
        replaces_dense_storage: bool,
    },
}

/// The chunk data a relocating chunked overwrite ([`MovingWrite::Chunked`])
/// will place, either already encoded or still to be.
pub(super) enum ChunkPayload {
    /// Split and encoded by the preflight, which had to do that anyway to find
    /// out whether the chunks still fit their slots.
    Encoded(Vec<PreparedChunk>),
    /// Element bytes that still carry unresolved variable-length references
    /// ([`OverwriteBytes::vlen`]), so the split has to wait for the apply
    /// phase: a filtered chunk's compressed length depends on the heap
    /// addresses patched into it, which is why the preflight cannot split
    /// first and patch after (issue #321).
    ///
    /// Such an overwrite therefore always relocates, which is what keeps the
    /// single-hard-link rejection (the one every [`MovingWrite`] returns to) in
    /// the preflight, where it belongs: whether it could have stayed in its
    /// slots is not knowable until those lengths exist.
    Deferred { bytes: OverwriteBytes },
}

/// A staged dataset reduced to the pieces the writer needs.
pub(super) struct FlatDataset {
    pub(super) name: LinkNameBuf,
    pub(super) dt: crate::datatype::Datatype,
    pub(super) ds: Dataspace,
    pub(super) raw: Vec<u8>,
    pub(super) attrs: Vec<crate::attribute::AttributeMessage>,
    /// Chunked/filtered storage options. When [`ChunkOptions::is_chunked`] is
    /// false and `maxshape` is `None`, the dataset is written as contiguous,
    /// unfiltered storage, otherwise its chunk data and index are built by
    /// [`WriteEngine::build_chunked_dataset`].
    pub(super) chunk_options: ChunkOptions,
    /// Maximum dimensions for an extensible dataset, mirrored into
    /// `ds.max_dimensions`. `None` for a fixed-shape dataset. A maxshape with an
    /// unlimited dimension selects the extensible-array chunk index, and a
    /// finite maxshape the fixed-array or single-chunk one.
    pub(super) maxshape: Option<Vec<MaxExtent>>,
    /// Variable-length attributes still carrying a placeholder heap address:
    /// (index into `attrs`, that attribute's global heap collections).
    /// Resolved in the apply loop right before this dataset's header is built.
    pub(super) vl_attrs: Vec<(usize, Vec<Vec<u8>>)>,
    /// Whether `attrs` goes in a fractal heap in place of the object header,
    /// decided by `file_writer::needs_dense_attrs` where the set was validated
    /// against what that heap can represent.
    ///
    /// Carried in place of requested again in the apply loop, so the set that was
    /// checked and the set that is placed cannot be decided differently  -  and
    /// because the question costs a serialization of every attribute to result,
    /// which is most expensive for exactly the oversized attribute it selects
    /// for. Patching a variable-length attribute's references is
    /// length-preserving, so nothing between the two phases can change the
    /// result.
    pub(super) attrs_are_dense: bool,
    /// A staged variable-length-string dataset's element references (still
    /// carrying placeholder heap addresses in `raw`) and global heap
    /// collections. Resolved in the apply loop right before `raw` is appended.
    pub(super) vl_string_staging: Option<VlStringStaging>,
    /// An object-reference dataset's per-element targets, still unresolved.
    /// Resolved (see [`WriteEngine::resolve_reference_target`]) and patched
    /// into `raw` in the apply loop, once every object this commit places has
    /// a known address. `None` for an ordinary dataset.
    pub(super) reference_targets: Option<Vec<ObjectRefPatch>>,
    /// A user-defined fill value, encoded in the dataset's datatype, or `None`
    /// for the library default. Validated against the datatype element size in
    /// [`flatten_dataset`].
    pub(super) fill: Option<Vec<u8>>,
    /// The fill time policy for the dataset, or `None` for the library default.
    pub(super) fill_time: Option<FillTime>,
    /// The metadata attributes' own inputs, kept so they can be rebuilt when
    /// `raw` changes before the commit writes it. `None` for a dataset staged
    /// without [`DatasetBuilder::with_provenance`](crate::DatasetBuilder::with_provenance).
    #[cfg(feature = "provenance")]
    pub(super) provenance: Option<StagedProvenance>,
}

/// A staged dataset's metadata metadata, and where in
/// [`FlatDataset::attrs`] the attributes it produced sit.
///
/// The hash is over the dataset's raw bytes, so it is only correct as of the
/// bytes it was taken from, [`FlatDataset::rebuild_provenance`] is what keeps
/// it that way when [`WriteEngine::extend_staged_dataset`] grows them.
#[cfg(feature = "provenance")]
pub(super) struct StagedProvenance {
    /// What [`crate::provenance::Provenance::build_attrs`] was, and is, called
    /// with. Held in place of re-derived from the attributes it produced, which
    /// a caller may also have set by hand.
    pub(super) inputs: crate::provenance::Provenance,
    /// The index in `attrs` where its attributes begin. They are appended last
    /// and are the only ones this may replace, so everything below it  -  the
    /// caller's own attributes, and the `vl_attrs` indices into them  -  is left
    /// exactly as it was.
    pub(super) attrs_start: usize,
}

#[cfg(feature = "provenance")]
impl FlatDataset {
    /// Recompute this dataset's metadata attributes over the bytes it now
    /// holds. A no-op for a dataset staged without metadata.
    ///
    /// The rebuilt set is the same attributes in the same order and of the same
    /// serialized length  -  the digest is 64 hex characters whatever it hashes  -
    /// so `attrs_are_dense`, decided once in [`flatten_dataset`] over the
    /// original set, still describes this one.
    pub(super) fn rebuild_provenance(&mut self) {
        let Some(prov) = &self.provenance else {
            return;
        };
        self.attrs.truncate(prov.attrs_start);
        let rebuilt = prov.inputs.build_attrs(&self.raw);
        self.attrs.extend(rebuilt);
    }
}

/// Validate a staged dataset and reduce it to a [`FlatDataset`]. `name` is the link name the
/// dataset has in its parent group, taken off the staged path by the caller. Contiguous,
/// unfiltered datasets are emitted as such, chunked, filtered, or extensible
/// datasets carry their [`ChunkOptions`] and maxshape through to the commit,
/// where [`WriteEngine::build_chunked_dataset`] lays out their chunk data and
/// index. An empty (zero-element) shape is allowed under either storage,
/// mirroring the whole-file writer: a contiguous one takes the `HADDR_UNDEF`
/// data address (see the apply loop) and a chunked one an index over zero
/// chunks, which is what an extensible dataset is created as before the first
/// append fills it. The geometry validation below still requires explicit chunk
/// dimensions for it  -  auto-chunking has no shape to derive them from. A
/// `provenance` dataset has its SHA-256/creator/timestamp/source attributes
/// computed here from `raw`, exactly as the whole-file writer does. A
/// variable-length attribute's global heap collection is built here (it is
/// fully self-contained  -  no address of its own) but placed and patched later,
/// in the apply loop, once its final address is known, likewise a
/// variable-length-string dataset's staged references and collection
/// (`db.vl_string_staging`) are carried through unresolved. An object-reference
/// dataset's per-element targets (`db.reference_targets`) are likewise carried
/// through unresolved  -  resolving a path target requires knowing every other
/// object this commit places, which is only known well into the apply loop
/// (see [`WriteEngine::resolve_reference_target`]). Rejects any remaining
/// feature this engine cannot reproduce faithfully: a
/// chunked/extensible variable-length-string or object-reference dataset, or a
/// filter pipeline the build cannot construct.
pub(super) fn flatten_dataset(db: DatasetBuilder, name: LinkNameBuf) -> Result<FlatDataset, Error> {
    let dt = db
        .datatype
        .ok_or(Error::EditUnsupported("dataset has no datatype/data"))?;
    let shape = db
        .shape
        .ok_or(Error::EditUnsupported("dataset has no shape"))?;
    let is_empty = shape.contains(&0);
    let chunked = db.chunk_options.is_chunked() || db.maxshape.is_some();
    // Storage this engine allocates for every dataset it places: it appends into
    // an existing layout, and the undefined data address the whole-file writer
    // gives an unallocated dataset (issue #293) has no equivalent here. Rejected
    // by name in place of left to the "dataset has no data" arm below, which is
    // the same symptom from the opposite cause, and rejected in place of ignored,
    // since ignoring it would write out the grid of fill values the caller requested
    // not to have.
    if db.allocation == StorageAllocation::Unallocated {
        return Err(Error::EditUnsupported(
            "a dataset with unallocated storage cannot be added to an existing file in place",
        ));
    }
    // Variable-length string element references live in the global heap, whose
    // address is only known once the apply loop places the collection. For
    // chunked/filtered/resizable storage the references sit inside chunks
    // written before that address exists, so patching them in is impossible.
    //
    // The whole-file writer lifted the same restriction by placing such a
    // dataset's collections ahead of everything else (issue #109), this engine
    // appends into an existing layout, where there is no "ahead" to place them
    // in, so the equivalent fix is a separate piece of work.
    if db.vl_string_staging.is_some() && chunked {
        return Err(Error::EditUnsupported(
            "chunked or extensible variable-length-string datasets cannot be added in place yet",
        ));
    }
    // Object-reference elements are resolved (see `resolve_reference_target`)
    // and patched into `raw` right before it is appended, for chunked storage
    // that patch would need to reach inside already-built chunk data, which
    // this engine does not support (mirrors the variable-length-string
    // rejection above  -  untested and unneeded combination for v1).
    if db.reference_targets.is_some() && chunked {
        return Err(Error::EditUnsupported(
            "chunked or extensible object-reference datasets cannot be added in place yet",
        ));
    }
    let raw = if is_empty {
        db.data
            .transpose()
            .map_err(|error| *error)?
            .unwrap_or_default()
    } else {
        db.data
            .transpose()
            .map_err(|error| *error)?
            .ok_or(Error::EditUnsupported("dataset has no data"))?
    };

    // Rejected for the same reason the whole-file writer rejects it: nothing
    // occupies zero bytes per element, the writers divide by the element size,
    // and a caller-built `Datatype` never passes through `Datatype::parse`.
    // Taking it as a `NonZeroUsize` hands the proof to the staging below.
    let elem_size = crate::datatype::element_size_usize(&dt)?;

    let elem = elem_size.get() as u64;
    // Multiply with checked arithmetic: an absurd shape whose element count
    // (or byte size) overflows `u64` is rejected in place of panicking in a
    // debug build or silently wrapping in release (which could let a wrapped
    // product spuriously match `raw.len()`). For a zero-element shape this
    // expected length is always 0 (a `0` dimension makes every checked
    // multiplication `Some(0)` regardless of the other dimensions), so this
    // also catches data mistakenly supplied for a shape that holds nothing.
    let expected = shape
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))
        .and_then(|n| n.checked_mul(elem));
    match expected {
        Some(expected) if raw.len() as u64 == expected => {}
        Some(_) => {
            return Err(Error::EditUnsupported(
                "dataset data length does not match its shape",
            ));
        }
        None => {
            return Err(Error::EditUnsupported(
                "dataset shape is too large to address on this platform",
            ));
        }
    }

    if chunked {
        // Reject malformed chunk geometry up front (the same validation the
        // whole-file writer applies), so a bad request  -  chunk dimensions of the
        // wrong rank, a zero chunk dimension, an inconsistent maximum shape, or
        // chunking a scalar  -  never reaches and panics the chunk splitter, nor
        // yields a dataset the reader cannot decode.
        Extent::new(&shape, db.maxshape.as_deref())
            .and_then(|extent| db.chunk_options.validate_geometry(extent))
            .map_err(Error::EditUnsupported)?;
        // A filter this build cannot apply is rejected up front in place of
        // failing mid-apply when a chunk is compressed.
        db.chunk_options
            .refuse_unavailable_filters()
            .map_err(Error::EditUnsupported)?;
        // Validate the filter pipeline  -  before any file bytes are
        // written  -  so an unsupported filter, an incompatible datatype, or a
        // fill value the filter cannot record is rejected up front, the chunk
        // data itself is laid out in the commit's apply phase. A filter the
        // build compiled out is not among them: `build_pipeline` emits its
        // descriptor regardless, which is why the check above exists. Chunked/filtered
        // storage flows through the very builder the normal writer uses
        // ([`chunked_write::compress_chunks`] + [`assemble_chunked_at`] + [`file_writer::build_chunked_dataset_oh`]),
        // so the
        // resulting object header is byte-identical to a freshly written one.
        //
        // The fill value goes in for the same reason the apply phase passes it:
        // scale-offset records it in the filter's parameters, so a validation
        // that left it out would be checking a pipeline the commit does not
        // build  -  and would pass a fill value the encoder later rejects.
        let chunk_dims = db.chunk_options.resolve_chunk_dims(&shape);
        let ctx = ChunkContext::from_datatype(&chunk_dims, &dt)?;
        db.chunk_options.build_pipeline(
            &ctx,
            FillPattern::new(
                db.fill_time.unwrap_or_default(),
                db.fill.as_deref(),
                ctx.element_size.narrow::<NonZeroUsize>()?,
            ),
        )?;
    }

    // The link message body (whose length is independent of the address) must
    // fit the object-header message's u16 size field, a pathologically long
    // name would otherwise overflow it into silent corruption. Measured with a
    // creation index present  -  the widest form, written into a group that tracks
    // link creation order  -  since the parent group is not known here.
    let mut sized = file_writer::make_link(name.as_str(), StoredAddress::new(0));
    sized.creation_order = Some(0);
    if sized.serialize(crate::file_writer::OFFSET_WIDTH).len() > OBJECT_HEADER_MESSAGE_MAX {
        return Err(Error::EditUnsupported(
            "dataset name is too long to encode as a link message",
        ));
    }

    let ds = Dataspace {
        space_type: if shape.is_empty() {
            DataspaceType::Scalar
        } else {
            DataspaceType::Simple
        },
        #[expect(
            clippy::cast_possible_truncation,
            reason = "dataspace rank fits the 1-byte dimensionality field (HDF5 caps rank at 32)"
        )]
        rank: shape.len() as u8,
        dimensions: shape,
        // A chunked, extensible dataset records its maximum dimensions. A
        // fixed-shape dataset has none.
        max_dimensions: db.maxshape.clone(),
    };
    let mut attrs: Vec<crate::attribute::AttributeMessage> = Vec::with_capacity(db.attrs.len());
    for (n, v) in &db.attrs {
        attrs.push(v.to_message(n)?);
    }
    // The message above already carries a placeholder (heap address 0) for each
    // element of a variable-length string attribute, stage its self-contained
    // global heap collections here (no address of their own to resolve yet) and
    // record which `attrs` slot they patch once the apply loop places them.
    let mut vl_attrs: Vec<(usize, Vec<Vec<u8>>)> = Vec::new();
    for (i, (_, v)) in db.attrs.iter().enumerate() {
        if let Some(strings) = v.var_len_strings() {
            vl_attrs.push((i, type_builders::build_global_heap_collections(strings)));
        }
    }
    // Appended last, and recorded as such: `raw` can still grow before the
    // commit writes it (`WriteEngine::extend_staged_dataset`), and the digest
    // has to follow it.
    #[cfg(feature = "provenance")]
    let provenance = db.provenance.as_ref().map(|prov| {
        let inputs = crate::provenance::Provenance {
            creator: prov.creator.clone(),
            timestamp: prov.timestamp.clone(),
            source: prov.source.clone(),
        };
        let attrs_start = attrs.len();
        attrs.extend(inputs.build_attrs(&raw));
        StagedProvenance {
            inputs,
            attrs_start,
        }
    });
    // More attributes than an object header keeps compactly, or one whose message
    // overflows its 2-byte message-size field, sends the set to a fractal heap  -
    // the same disjunction, and the same heap, the whole-file writer uses. What
    // that heap cannot represent is rejected here, in the preflight, so a staged
    // dataset that cannot be written is rejected before the commit places a byte.
    let attrs_are_dense = crate::file_writer::needs_dense_attrs(&attrs).map_err(Error::Format)?;
    if attrs_are_dense {
        crate::file_writer::dense_attrs_check(&attrs).map_err(Error::Format)?;
    }

    // A committed datatype is an object of its own, which an in-place edit has no
    // way to place: it appends into an existing file in place of laying one out,
    // so there is nothing to resolve the named path against. Writing the type
    // inline instead would produce a dataset that reads correctly but no longer
    // shares the named type, so reject by name. The whole-file writer places
    // them, [`crate::repack`] is the route from an edited file to one.
    if db.committed_datatype_path.is_some()
        || attrs.iter().any(|a| a.datatype_location.is_committed())
    {
        return Err(Error::EditUnsupported(
            "a dataset or attribute naming a committed (shared) datatype cannot be added in \
             place; write the file with FileBuilder instead",
        ));
    }

    // A user-defined fill value is one element wide, so its byte length must
    // equal the datatype's element size (mirrors the whole-file writer's check).
    if let Some(fill) = &db.fill {
        let expected = elem.to_usize()?;
        if fill.len() != expected {
            return Err(Error::Format(FormatError::FillValueSizeMismatch {
                expected,
                actual: fill.len(),
            }));
        }
    }

    Ok(FlatDataset {
        name,
        dt,
        ds,
        raw,
        attrs,
        chunk_options: db.chunk_options,
        maxshape: db.maxshape,
        vl_attrs,
        attrs_are_dense,
        vl_string_staging: db.vl_string_staging,
        reference_targets: db.reference_targets,
        fill: db.fill,
        fill_time: db.fill_time,
        #[cfg(feature = "provenance")]
        provenance,
    })
}

/// Returns a copy of `region` whose first Data Layout message has `new_layout_body` as its body.
///
/// The rewritten record has no flag set and, where the region tracks creation order, a creation
/// index of 0. Every other record is copied unchanged, a later Data Layout message included.
///
/// # Errors
///
/// Returns [`Error::EditUnsupported`] if `region` lacks a Data Layout message or if the body of a
/// record it reads runs past the end of `region`. Returns [`Error::Format`] if `new_layout_body`
/// is longer than [`OBJECT_HEADER_MESSAGE_MAX`].
pub(super) fn replace_layout_message(
    region: &OhRegion,
    new_layout_body: &[u8],
) -> Result<OhRegion, Error> {
    let splice = region
        .splice_first(MessageType::DATA_LAYOUT)?
        .ok_or(Error::EditUnsupported(
            "chunked dataset header has no data-layout message to relocate",
        ))?;
    Ok(region.with_bytes(splice.replace_record(MessageFlags::NONE, 0, new_layout_body)?))
}

/// Returns a copy of `region` whose first Dataspace message has `new_dataspace_body` as its body.
///
/// The rewritten record has no flag set and, where the region tracks creation order, a creation
/// index of 0. Every other record is copied unchanged, a later Dataspace message included.
///
/// # Errors
///
/// Returns [`Error::AppendUnsupported`] if `region` lacks a Dataspace message,
/// [`Error::EditUnsupported`] if the body of a record it reads runs past the end of `region`, and
/// [`Error::Format`] if `new_dataspace_body` is longer than [`OBJECT_HEADER_MESSAGE_MAX`].
pub(super) fn replace_dataspace_message(
    region: &OhRegion,
    new_dataspace_body: &[u8],
) -> Result<OhRegion, Error> {
    let splice = region
        .splice_first(MessageType::DATASPACE)?
        .ok_or(Error::AppendUnsupported(
            "dataset header has no dataspace message to grow",
        ))?;
    Ok(region.with_bytes(splice.replace_record(MessageFlags::NONE, 0, new_dataspace_body)?))
}

/// Compares datatype structure while ignoring byte order for one-byte fixed-point types.
///
/// Compound members and base types are compared recursively. All other fields must match.
pub(super) fn overwrite_datatypes_match(disk: &Datatype, staged: &Datatype) -> bool {
    match (disk, staged) {
        (Datatype::FixedPoint(disk), Datatype::FixedPoint(staged))
            if disk.size().get() == 1 && staged.size().get() == 1 =>
        {
            disk.layout() == staged.layout()
        }
        (
            Datatype::Compound {
                size: disk_size,
                members: disk_members,
            },
            Datatype::Compound {
                size: staged_size,
                members: staged_members,
            },
        ) => {
            disk_size == staged_size
                && disk_members.len() == staged_members.len()
                && disk_members
                    .iter()
                    .zip(staged_members)
                    .all(|(disk, staged)| {
                        disk.name == staged.name
                            && disk.byte_offset == staged.byte_offset
                            && overwrite_datatypes_match(&disk.datatype, &staged.datatype)
                    })
        }
        (
            Datatype::Enumeration {
                size: disk_size,
                base_type: disk_base,
                members: disk_members,
            },
            Datatype::Enumeration {
                size: staged_size,
                base_type: staged_base,
                members: staged_members,
            },
        ) => {
            disk_size == staged_size
                && disk_members == staged_members
                && overwrite_datatypes_match(disk_base, staged_base)
        }
        (
            Datatype::VariableLength(VariableLengthType::String(disk)),
            Datatype::VariableLength(VariableLengthType::String(staged)),
        ) => {
            disk.padding() == staged.padding()
                && disk.charset() == staged.charset()
                && overwrite_datatypes_match(disk.base_type(), staged.base_type())
        }
        (
            Datatype::VariableLength(VariableLengthType::Sequence {
                element_type: disk_base,
            }),
            Datatype::VariableLength(VariableLengthType::Sequence {
                element_type: staged_base,
            }),
        ) => overwrite_datatypes_match(disk_base, staged_base),
        (
            Datatype::Array {
                base_type: disk_base,
                dimensions: disk_dimensions,
            },
            Datatype::Array {
                base_type: staged_base,
                dimensions: staged_dimensions,
            },
        ) => {
            disk_dimensions == staged_dimensions
                && overwrite_datatypes_match(disk_base, staged_base)
        }
        _ => disk == staged,
    }
}
/// The datatype, dataspace, parsed chunked data layout, and verbatim filter-
/// pipeline message bytes (if any) of a chunked dataset header, parsed by
/// [`parse_chunked_header`].
pub(super) struct ChunkedHeaderParts {
    pub(super) dt: crate::datatype::Datatype,
    pub(super) ds: Dataspace,
    pub(super) layout: DataLayout,
    pub(super) pipeline_message: Option<Vec<u8>>,
}

/// Parse the datatype, dataspace, chunked data layout, and verbatim filter-
/// pipeline message bytes (if any) from a chunked dataset header `region`. Used by
/// the chunked copy path to derive chunk geometry and the on-disk filter pipeline.
/// Errors if any required message is missing or the layout is not chunked.
pub(super) fn parse_chunked_header(
    region: &OhRegion,
    widths: FormatWidths,
) -> Result<ChunkedHeaderParts, Error> {
    let mut datatype: Option<(usize, usize)> = None;
    let mut dataspace: Option<(usize, usize)> = None;
    let mut layout: Option<(usize, usize)> = None;
    let mut pipeline: Option<(usize, usize)> = None;
    let mut p = 0;
    while let Some((msg_type, body, body_end)) = region.next_message(p)? {
        match msg_type {
            MessageType::DATATYPE => datatype = Some((body, body_end)),
            MessageType::DATASPACE => dataspace = Some((body, body_end)),
            MessageType::DATA_LAYOUT => layout = Some((body, body_end)),
            MessageType::FILTER_PIPELINE => pipeline = Some((body, body_end)),
            _ => {}
        }
        p = body_end;
    }
    let (dt_b, dt_e) = datatype.ok_or(Error::EditUnsupported("dataset header has no datatype"))?;
    let (ds_b, ds_e) =
        dataspace.ok_or(Error::EditUnsupported("dataset header has no dataspace"))?;
    let (lb, le) = layout.ok_or(Error::EditUnsupported("dataset header has no data layout"))?;
    let (dt, _) = hdf5_pure_format::__private::parse_datatype(&region[dt_b..dt_e])?;
    let ds = Dataspace::parse_with_length_width(&region[ds_b..ds_e], widths.lengths)?;
    let dl = DataLayout::parse_with_widths(&region[lb..le], widths)?;
    if !matches!(dl, DataLayout::Chunked { .. }) {
        return Err(Error::EditUnsupported("dataset is not chunked"));
    }
    let pipeline_message = pipeline.map(|(b, e)| region[b..e].to_vec());
    Ok(ChunkedHeaderParts {
        dt,
        ds,
        layout: dl,
        pipeline_message,
    })
}

/// The element bytes a staged value overwrite will write, paired with the
/// variable-length staging that has still to be resolved into them and the
/// `path` they belong to.
///
/// Cloned in place of moved because the staged set must survive a rejected
/// commit whole (issue #316), and this runs in the preflight that may reject.
pub(super) fn staged_bytes(fd: &FlatDataset, path: &ObjectPathBuf) -> OverwriteBytes {
    OverwriteBytes {
        raw: fd.raw.clone(),
        vlen: fd.vl_string_staging.clone().map(|staging| VlenOverwrite {
            staging,
            path: path.clone(),
        }),
    }
}

/// The chunk geometry and encoding of a chunked dataset that a staged overwrite rewrites.
pub(super) struct ChunkedOverwrite {
    /// The extent of the dataset, which the overwrite keeps.
    pub(super) shape: Vec<u64>,
    /// The chunk dimensions, one per dimension of the dataset.
    pub(super) chunk_dims: Vec<u64>,
    pub(super) element_size: NonZeroUsize,
    /// The layout flags of the dataset, which [`prepare_chunks`](Self::prepare_chunks) reads and
    /// the layout message of a relocating overwrite stores.
    pub(super) flags: ChunkedLayoutFlags,
    pub(super) pipeline: Option<FilterPipeline>,
    /// What a partial edge chunk holds past the extent.
    pub(super) padding: PaddingFill,
    /// The dataset's datatype, for the filter pipeline's [`ChunkContext`].
    pub(super) dt: Datatype,
}

impl ChunkedOverwrite {
    /// Splits `raw`, the dataset's new value, into whole chunks in dense row-major grid order, and
    /// encodes each for a reader at the dataset's extent.
    ///
    /// Each chunk is filtered, except a chunk that extends past the extent where
    /// [`ChunkedLayoutFlags::partial_edge_chunks_stored_raw`] is `true` for `flags`, which is stored
    /// unfiltered.
    ///
    /// The part of a partial edge chunk past the extent holds `padding`: the dataset's fill value,
    /// or zeros where the dataset defines none or never writes it. A reader returns it from those
    /// elements once the dataset is extended over them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] with [`FormatError::UnreadableFillValue`] if a chunk extends past
    /// the extent and the dataset's Fill Value message is unreadable, and with
    /// [`FormatError::ZeroSizedDatatype`] if the datatype occupies zero bytes per element. Returns
    /// the errors of [`PreparedChunk::encode`] for each chunk.
    pub(super) fn prepare_chunks(&self, raw: &[u8]) -> Result<Vec<PreparedChunk>, Error> {
        let Self {
            shape,
            chunk_dims,
            element_size,
            flags,
            pipeline,
            padding,
            dt,
        } = self;
        let split = chunked_write::split_into_chunks(
            raw,
            shape,
            chunk_dims,
            *element_size,
            padding.pattern(*element_size),
        )?;
        let ctx = ChunkContext::from_datatype(chunk_dims, dt)?;
        // One encoder across the rewrite. See `FilterScratch`.
        let mut scratch = FilterScratch::new();
        let mut offsets = vec![0; shape.len()];
        Ok(split
            .into_iter()
            .map(|logical| {
                let decoding = ChunkDecoding::at_extent(*flags, chunk_dims, shape, &offsets);
                advance_row_major(chunk_dims, shape, &mut offsets);
                PreparedChunk::encode(
                    &mut scratch,
                    pipeline.as_ref(),
                    ctx,
                    ChunkEncoding::for_decoding(decoding),
                    logical,
                )
            })
            .collect::<Result<_, _>>()?)
    }
}

/// Advances `offsets`, the element offsets of a chunk in a dataset of extent `shape`, to the next
/// chunk in row-major grid order, and back to the first chunk after the last.
pub(super) fn advance_row_major(chunk_dims: &[u64], shape: &[u64], offsets: &mut [u64]) {
    for ((&chunk, &dim), offset) in chunk_dims.iter().zip(shape).zip(offsets).rev() {
        *offset = offset.saturating_add(chunk);
        if *offset < dim {
            return;
        }
        *offset = 0;
    }
}

/// Tries to overwrite a chunked dataset's chunks in place. When the dataset's
/// on-disk chunks form a dense grid aligned with `new_chunks` (dense row-major
/// order), the index records for every slot the filter mask of its new chunk,
/// and every new chunk **fits** the slot it replaces (`new_len <= slot`),
/// returns the in-place `(address, bytes)` writes:
///
/// - When every new chunk is **exactly** its slot's size, only the chunk data is
///   written, the index is untouched (so any enumerable index type works, and a
///   crash can tear at most a chunk's value bytes, not the structure).
/// - When some new chunks are **smaller** (fit with slack), the chunk index
///   records each chunk's stored size, so the index is rebuilt in place to record
///   the new sizes (see [`try_rebuild_index_in_place`]). This is supported only
///   for a v4 fixed-array or extensible-array index occupying a single contiguous
///   on-disk region, any other case returns `None` to relocate.
///
/// Returns `None`  -  so the caller relocates the dataset instead  -  when the index
/// cannot be enumerated, the grid is sparse, a slot's filter mask differs from
/// its new chunk's, a new chunk does not fit, the index cannot be rebuilt in
/// place, or any write would be out of bounds or overlap another.
pub(super) fn try_inplace_chunk_writes<S: Source + ?Sized>(
    src: &S,
    layout: &DataLayout,
    ds: &Dataspace,
    spatial: &[u64],
    raw_size: u64,
    new_chunks: &[PreparedChunk],
) -> Option<Vec<(u64, Vec<u8>)>> {
    let infos =
        chunked_read::enumerate_chunks_from_source(src, layout, ds, crate::file_writer::WIDTHS)
            .ok()?;
    let grid = chunked_read::plan_dense_grid(infos, &ds.dimensions, spatial)?;
    if grid.grid_order.len() != new_chunks.len() {
        return None;
    }
    let mut writes = Vec::with_capacity(new_chunks.len() + 1);
    let mut spans: Vec<(u64, u64)> = Vec::with_capacity(new_chunks.len() + 1);
    let mut any_shrunk = false;
    for (ci, chunk) in grid.grid_order.iter().zip(new_chunks) {
        // An index left untouched keeps each slot's recorded mask, so the
        // overwrite relocates where a slot's mask differs from its new chunk's.
        if ci.filter_mask != chunk.filter_mask() {
            return None;
        }
        let new_len = chunk.stored_size();
        let slot = ci.chunk_size.get();
        // A chunk that no longer fits its slot must relocate.
        if new_len > slot {
            return None;
        }
        if new_len < slot {
            any_shrunk = true;
        }
        ci.address
            .get()
            .checked_add(new_len)
            .filter(|&e| e <= src.len())?;
        writes.push((ci.address.get(), chunk.bytes().to_vec()));
        spans.push((ci.address.get(), new_len));
    }

    // A shrinking overwrite changes the index-recorded chunk sizes, so the index
    // must be rebuilt in place to match, an equal-size one leaves it untouched.
    if any_shrunk {
        let (index_addr, index_bytes) = try_rebuild_index_in_place(
            src,
            layout,
            ds,
            spatial,
            raw_size,
            &grid.grid_order,
            new_chunks,
        )?;
        spans.push((index_addr, index_bytes.len() as u64));
        writes.push((index_addr, index_bytes));
    }

    // Reject to perform overlapping in-place writes (a malformed source index, or
    // an index region that overlaps a chunk slot), relocate instead so two writes
    // never clobber each other.
    if !super::spans_disjoint_in_bounds(&mut spans, src.len()) {
        return None;
    }
    Some(writes)
}

/// Rebuilds a chunked dataset's index **in place** so it records the new
/// (smaller) per-chunk stored sizes after a fits-with-slack overwrite, returning
/// the `(address, bytes)` write that replaces it. The chunks keep their existing
/// addresses (only their stored bytes shrank), so the rebuilt index points at the
/// same slots with the new sizes.
///
/// Supported only for a v4 **fixed-array** or **extensible-array** index whose
/// on-disk structure is a single contiguous region starting at the index address
/// This is the layout this crate's own writer produces. The element width derives from
/// the unchanged raw chunk size, so the rebuilt structure is byte-for-byte the
/// same length as the original, this is required to match exactly, which rejects a
/// scattered or differently-laid-out (e.g. C-written) index, leaving the caller
/// to relocate. Single-chunk (size in the layout message) and B-tree-v1 (no
/// writer) indexes are not rebuilt here.
///
/// Like any in-place value overwrite (the HDF5 `H5Dwrite` model) this is not
/// atomic: a crash mid-write can tear the index and leave the dataset needing a
/// rewrite. It is used only on the in-place path, whose linearization point is the
/// synced data write.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_rebuild_index_in_place<S: Source + ?Sized>(
    src: &S,
    layout: &DataLayout,
    ds: &Dataspace,
    spatial: &[u64],
    raw_size: u64,
    grid_order: &[crate::chunked_read::ChunkInfo],
    new_chunks: &[PreparedChunk],
) -> Option<(u64, Vec<u8>)> {
    let DataLayout::Chunked { index, .. } = layout else {
        return None;
    };
    let index_addr = index.address()?;
    let written: Vec<ChunkRecord> = grid_order
        .iter()
        .zip(new_chunks)
        .map(|(ci, chunk)| ChunkRecord {
            address: ci.address,
            stored_size: chunk.stored_size(),
            filter_mask: chunk.filter_mask(),
        })
        .collect();
    // `grid_order` is dense row-major over the *current* shape, where each of
    // those chunks sits in the index is the maximum grid's business, and the
    // rebuild has to reproduce the numbering the original index was written
    // with. Same rule, same function as the writer.
    let (_, slot_of_chunk, index_slots) = crate::chunked_write::plan_index_slots(
        &ds.dimensions,
        spatial,
        ds.max_dimensions.as_deref(),
        raw_size,
        true,
        // Rebuilding the index of a dataset whose chunks are in hand.
        crate::chunked_write::StorageAllocation::Allocated,
    )
    .ok()?;
    let slots = IndexSlots::new(&written, &slot_of_chunk, index_slots).ok()?;
    let new_index = match index {
        // `raw_size` is the whole-chunk byte size, which is what the element
        // width derives from  -  the same value the original index was built with,
        // so the rebuilt structure matches its length. (An index written by a
        // version that derived the width from the written chunks instead can
        // disagree, the length check below then rejects it and the caller
        // relocates, which is the safe direction.)
        ChunkIndexLayout::FixedArray { .. } => hdf5_pure_format::__private::build_fixed_array_at(
            &slots,
            raw_size,
            OFFSET_WIDTH,
            LENGTH_WIDTH,
            true,
            index_addr,
        )
        .ok()?,
        ChunkIndexLayout::ExtensibleArray { .. } => {
            hdf5_pure_format::__private::build_extensible_array_at(
                &slots,
                raw_size,
                OFFSET_WIDTH,
                LENGTH_WIDTH,
                true,
                index_addr,
            )
            .ok()?
        }
        // A single chunk's stored size sits in the layout message, so rebuilding
        // it would rewrite the header, an implicit index has no structure to
        // rebuild, and neither B-tree has a writer. All four relocate instead.
        ChunkIndexLayout::SingleChunk { .. }
        | ChunkIndexLayout::Implicit { .. }
        | ChunkIndexLayout::BTreeV1 { .. }
        | ChunkIndexLayout::BTreeV2 { .. } => return None,
    };

    // The on-disk index must be a single contiguous region starting at the index
    // address, and the rebuilt structure must be exactly the same length (true for
    // an index this crate wrote). A scattered or different on-disk layout fails
    // the check and the caller relocates.
    let mut spans =
        crate::chunked_read::chunk_index_spans_from_source(src, layout, crate::file_writer::WIDTHS)
            .ok()?;
    if spans.is_empty() {
        return None;
    }
    spans.sort_unstable_by_key(|&(a, _)| a);
    if spans[0].0 != index_addr.get() {
        return None;
    }
    let mut end = index_addr.get();
    for &(a, l) in &spans {
        if a != end {
            return None; // a gap means the index is not contiguous
        }
        end = a.checked_add(l)?;
    }
    if new_index.len() as u64 != end - index_addr.get() {
        return None;
    }
    index_addr
        .get()
        .checked_add(new_index.len() as u64)
        .filter(|&e| e <= src.len())?;
    Some((index_addr.get(), new_index))
}

/// A [`ChunkProvider`] over chunk bytes held in memory, in dense row-major grid
/// order, for the editor's chunked copy, whose [`CopyTree::DatasetChunked`]
/// holds the bytes of each chunk.
pub(super) struct SliceChunkProvider<'a> {
    pub(super) chunks: &'a [Vec<u8>],
}

impl ChunkProvider for SliceChunkProvider<'_> {
    fn chunk_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), FormatError> {
        let chunk = self.chunks.get(index).ok_or_else(|| {
            FormatError::ChunkedReadError("chunk index out of range for in-memory provider".into())
        })?;
        out.extend_from_slice(chunk);
        Ok(())
    }
}

/// A [`ChunkProvider`] over the chunks a relocating chunked overwrite prepared, in dense
/// row-major grid order.
pub(super) struct PreparedChunkProvider<'a> {
    pub(super) chunks: &'a [PreparedChunk],
}

impl ChunkProvider for PreparedChunkProvider<'_> {
    fn chunk_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), FormatError> {
        let chunk = self.chunks.get(index).ok_or_else(|| {
            FormatError::Internal(format!(
                "a verbatim plan asked for chunk {index} of {} prepared chunks",
                self.chunks.len()
            ))
        })?;
        out.extend_from_slice(chunk.bytes());
        Ok(())
    }
}

impl WriteEngine {
    /// Stages a prepared dataset for creation at `path` on the next [`commit`](Self::commit).
    ///
    /// The dataset name is taken from the last component of `path`. The parent group must
    /// exist in the file or be created in the same commit. [`flatten_dataset`] prepares
    /// and checks the builder before it is stored in the staged set.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if `path` is the root, collides with an existing
    /// object, or overlaps a live buffered appender. Returns the preparation error if
    /// [`flatten_dataset`] cannot prepare `builder`.
    pub(crate) fn stage_created_dataset(
        &mut self,
        path: &ObjectPathBuf,
        builder: DatasetBuilder,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        let Some((parent, leaf)) = path.split_leaf() else {
            return Err(Error::EditUnsupported("dataset path has an empty name"));
        };
        self.refuse_creation_collision(path, StagedKind::Dataset)?;
        self.staged
            .push_dataset(parent, flatten_dataset(builder, leaf)?);
        Ok(())
    }

    /// Stages replacement values for an existing dataset on the next [`commit`](Self::commit).
    ///
    /// The replacement is prepared by [`flatten_dataset`]. Shape and datatype
    /// compatibility with the committed dataset are checked during the commit.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if `path` is the root or overlaps a live buffered
    /// appender, or if `builder` specifies chunking, attributes, fill settings, or
    /// unresolved reference targets. Returns the preparation error if
    /// [`flatten_dataset`] cannot prepare `builder`.
    pub(crate) fn stage_dataset_write(
        &mut self,
        path: &ObjectPathBuf,
        builder: DatasetBuilder,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        let Some((_, leaf)) = path.split_leaf() else {
            return Err(Error::EditUnsupported("cannot overwrite the root group"));
        };
        let fd = flatten_dataset(builder, leaf)?;
        Self::refuse_unsupported_overwrite(&fd)?;
        self.staged.push(StagedOp::WriteDataset(path.clone(), fd));
        Ok(())
    }

    /// Extends a staged dataset creation with the appended elements.
    ///
    /// The caller checks that its handle addresses a pending creation. If another thread
    /// commits that creation before this call, the append is staged against the committed
    /// dataset instead.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a live buffered appender overlaps `path`, or if
    /// an atomic batch is open while a creation is staged there. Returns the error from
    /// [`extend_staged_dataset`](Self::extend_staged_dataset) for an incompatible pending
    /// append, or from [`refuse_lossy_partial_tail`](Self::refuse_lossy_partial_tail) for a
    /// committed dataset.
    pub(crate) fn stage_dataset_append_pending(
        &mut self,
        path: &ObjectPathBuf,
        builder: AppendBuilder,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        if self.staged.dataset_at(path).is_none() {
            self.refuse_lossy_partial_tail(path, &builder)?;
            self.staged
                .push(StagedOp::AppendDataset(path.clone(), builder));
            return Ok(());
        }
        self.refuse_mid_batch()?;
        self.extend_staged_dataset(path, &builder)
    }

    /// Adds elements to the prepared bytes and leading dimension of a staged creation.
    ///
    /// Chunked storage and an unlimited dimension are not required. With the `provenance`
    /// feature, digest attributes are recomputed from the extended bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] for mixed or mismatched element types, unresolved
    /// per-element references, a scalar or zero-byte row, an incomplete row, arithmetic
    /// overflow, or a leading dimension beyond its declared maximum.
    ///
    /// # Panics
    ///
    /// Panics if the caller has not established that a dataset is staged at `path`.
    pub(super) fn extend_staged_dataset(
        &mut self,
        path: &ObjectPathBuf,
        builder: &AppendBuilder,
    ) -> Result<(), Error> {
        if builder.dt_conflict() {
            return Err(Error::AppendUnsupported(
                "this append mixes element datatypes; use one element type per append",
            ));
        }
        // The caller finds this dataset before entering the append helper.
        let fd = self
            .staged
            .dataset_at_mut(path)
            .expect("caller checked a dataset is staged at this path");
        if fd.vl_string_staging.is_some() || fd.reference_targets.is_some() {
            return Err(Error::AppendUnsupported(
                "a staged variable-length-string or object-reference dataset cannot be appended \
                 to before it is committed: its per-element side table is built with the \
                 dataset. Supply every element through the builder that creates it",
            ));
        }
        if let Some(dt) = builder.elem_dt() {
            if *dt != fd.dt {
                return Err(Error::AppendUnsupported(
                    "appended element datatype does not match the staged dataset's datatype",
                ));
            }
        }
        let Some((lead, inner)) = fd.ds.dimensions.split_first() else {
            return Err(Error::AppendUnsupported(
                "a scalar dataset has no dimension to append along",
            ));
        };
        // A leading-dimension row includes every trailing extent.
        let mut row_bytes = u64::from(fd.dt.type_size());
        for &d in inner {
            row_bytes = row_bytes.checked_mul(d).ok_or(Error::AppendUnsupported(
                "staged dataset row size overflows",
            ))?;
        }
        if row_bytes == 0 {
            return Err(Error::AppendUnsupported(
                "a staged dataset whose rows hold no bytes cannot be appended to",
            ));
        }
        let bytes = builder.raw();
        if bytes.len() as u64 % row_bytes != 0 {
            return Err(Error::AppendUnsupported(
                "appended byte length is not a whole number of rows of the staged dataset",
            ));
        }
        let grown = lead
            .checked_add(bytes.len() as u64 / row_bytes)
            .ok_or(Error::AppendUnsupported("staged dataset length overflows"))?;
        if let Some(max) = fd
            .ds
            .max_dimensions
            .as_ref()
            .and_then(|m| m.first().copied())
            && matches!(max, MaxExtent::Fixed(limit) if limit < grown)
        {
            return Err(Error::AppendUnsupported(
                "appending would grow the staged dataset past its maximum shape",
            ));
        }
        fd.raw.extend_from_slice(bytes);
        fd.ds.dimensions[0] = grown;
        #[cfg(feature = "provenance")]
        fd.rebuild_provenance();
        Ok(())
    }

    /// The refusals a staged value overwrite (`write_dataset`) makes from the
    /// staged builder alone, with no on-disk header involved.
    ///
    /// [`stage_dataset_write`](Self::stage_dataset_write) applies these as the
    /// write is staged, so the call that configured the builder is the one that
    /// reports the mistake. A rejection that reads only `fd` belongs here, one that
    /// needs the target's header belongs in
    /// [`prepare_write`](Self::prepare_write).
    pub(super) fn refuse_unsupported_overwrite(fd: &FlatDataset) -> Result<(), Error> {
        // A value overwrite never introduces chunking, filters, or an extensible
        // shape: those would change the storage layout, not just the bytes.
        if fd.chunk_options.is_chunked() || fd.maxshape.is_some() {
            return Err(Error::EditUnsupported(
                "write_dataset overwrites values only; it cannot make a dataset \
                 chunked, filtered, or extensible",
            ));
        }

        // `write_dataset` overwrites element bytes only, it does not touch the
        // object header's attribute messages. Attributes staged on the returned
        // builder would otherwise be silently dropped (the in-place path rewrites
        // only the data block, and the moving path reuses the verbatim on-disk
        // header), so reject in place of degrade  -  set them in a separate edit.
        if !fd.attrs.is_empty() {
            return Err(Error::EditUnsupported(
                "write_dataset overwrites values only; it cannot set attributes \
                 (set them with a separate edit)",
            ));
        }

        // `write_dataset` overwrites element bytes only, it reuses the dataset's
        // existing Fill Value message (the in-place path rewrites only the data
        // block, and the moving path keeps every header message but the layout
        // verbatim). A fill value staged on the returned builder would otherwise
        // be silently ignored, so reject in place of degrade  -  set the fill value
        // when the dataset is first created.
        if fd.fill.is_some() {
            return Err(Error::EditUnsupported(
                "write_dataset overwrites values only; it cannot change the fill \
                 value (set it when the dataset is created)",
            ));
        }
        if fd.fill_time.is_some() {
            return Err(Error::EditUnsupported(
                "write_dataset overwrites values only; it cannot change the fill \
                 time (set it when the dataset is created)",
            ));
        }

        // `with_vlen_strings` stages placeholder element references, resolved by
        // placing their global heap collections and patching the addresses in
        // (issue #321). That happens where each plan is written, not here: see
        // `OverwriteBytes`.

        // `reference_targets` holds elements only the add path resolves, in
        // `preflight_reference_targets` and the apply loop after it, the
        // overwrite path never reads the field. Checked on the field in place of
        // on `with_path_references`, since every producer stages elements that
        // are equally unresolved. Left unrefused, a staged overwrite writes
        // address-zero placeholders over a working reference dataset and
        // `commit` reports `Ok` (issue #318). `with_reference_data` supplies
        // resolved addresses and overwrites like any other value.
        if fd.reference_targets.is_some() {
            return Err(Error::EditUnsupported(
                "write_dataset cannot overwrite an object-reference dataset's \
                 data in place yet",
            ));
        }

        Ok(())
    }

    /// Checks overwrite compatibility and plans storage for the replacement values.
    ///
    /// The staged datatype must be compatible under [`overwrite_datatypes_match`],
    /// and the shape must match the on-disk shape. A rejected write leaves the
    /// commit unapplied.
    ///
    /// Contiguous, compact, and chunked (including filtered) datasets are all
    /// supported. The chunk geometry, filter pipeline, and chunk index are read
    /// from the on-disk header.
    /// A non-persistent PAGE session relocates contiguous and chunked replacements regardless
    /// of whether they fit the existing allocation.
    ///
    /// Every rejection `fd` can decide on its own has been made by then, where the
    /// write is staged.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the dataset cannot be overwritten,
    /// the datatype is incompatible, or the shape differs.
    pub(super) fn prepare_write(
        &self,
        addr: u64,
        fd: &FlatDataset,
        path: &ObjectPathBuf,
    ) -> Result<WritePlan, Error> {
        let src = &self.image();
        let base = self.addresses;
        let widths = self.widths;
        // `stage_dataset_write` checks these options before storing a prepared write.
        // Skipping that check would discard chunk options during a contiguous overwrite.
        debug_assert!(
            Self::refuse_unsupported_overwrite(fd).is_ok(),
            "a staged write reached prepare_write without refuse_unsupported_overwrite"
        );

        let region = Self::gather_oh_messages(src, addr, base)?;

        // Locate the datatype, dataspace, and data-layout messages, and detect a
        // filter pipeline (filtered storage is always chunked, never contiguous).
        let mut datatype: Option<(usize, usize)> = None;
        let mut dataspace: Option<(usize, usize)> = None;
        let mut layout: Option<(usize, usize)> = None;
        let mut filter: Option<(usize, usize)> = None;
        // The dataset's own fill value, which the edge overhang of a partial
        // chunk must hold (issue #296). Both message forms, newest wins.
        let mut fill_msg: Option<(MessageType, usize, usize)> = None;
        let mut has_link = false;
        let mut p = 0;
        while let Some((msg_type, body, body_end)) = region.next_message(p)? {
            match msg_type {
                MessageType::DATATYPE => datatype = Some((body, body_end)),
                MessageType::DATASPACE => dataspace = Some((body, body_end)),
                // libhdf5 reads the first Data Layout message of a header
                // (`H5O_msg_read_oh` in `H5Omessage.c`, HDF5 2.2.0).
                MessageType::DATA_LAYOUT if layout.is_none() => layout = Some((body, body_end)),
                MessageType::FILTER_PIPELINE => filter = Some((body, body_end)),
                // Versioned beats legacy, and within a type the first wins  -
                // the same rule `Dataset::fill_value_message` and `Located::from_walk`
                // apply, so all three agree on a header carrying more than one.
                MessageType::FILL_VALUE
                    if !matches!(fill_msg, Some((MessageType::FILL_VALUE, ..))) =>
                {
                    fill_msg = Some((msg_type, body, body_end));
                }
                MessageType::FILL_VALUE_OLD if fill_msg.is_none() => {
                    fill_msg = Some((msg_type, body, body_end));
                }
                MessageType::LINK | MessageType::LINK_INFO | MessageType::SYMBOL_TABLE => {
                    has_link = true;
                }
                _ => {}
            }
            p = body_end;
        }

        if has_link {
            return Err(Error::EditUnsupported(
                "write_dataset target is a group, not a dataset",
            ));
        }
        let (dt_b, dt_e) =
            datatype.ok_or(Error::EditUnsupported("dataset header has no datatype"))?;
        let (ds_b, ds_e) =
            dataspace.ok_or(Error::EditUnsupported("dataset header has no dataspace"))?;
        let (lb, le) = layout.ok_or(Error::EditUnsupported("dataset header has no data layout"))?;

        // Require a compatible datatype and matching current dimensions. One-byte
        // fixed-point types may differ in byte order, including nested types.
        let (disk_dt, _) = hdf5_pure_format::__private::parse_datatype(&region[dt_b..dt_e])?;
        if !overwrite_datatypes_match(&disk_dt, &fd.dt) {
            return Err(Error::EditUnsupported(
                "write_dataset datatype does not match the on-disk dataset (overwrite, not retype)",
            ));
        }
        let disk_ds = Dataspace::parse_with_length_width(&region[ds_b..ds_e], widths.lengths)?;
        if disk_ds.space_type != fd.ds.space_type
            || disk_ds.rank != fd.ds.rank
            || disk_ds.dimensions != fd.ds.dimensions
        {
            return Err(Error::EditUnsupported(
                "write_dataset shape does not match the on-disk dataset (overwrite, not reshape)",
            ));
        }

        // Classify the layout. Version 3/4 compact (class 0), contiguous (class
        // 1), and chunked (class 2) are supported, an old-version layout or a
        // virtual layout (class 3) is rejected.
        if le - lb < 2 {
            return Err(Error::EditUnsupported("malformed data-layout message"));
        }
        let version = region[lb];
        if version != 3 && version != 4 {
            return Err(Error::EditUnsupported(
                "an unsupported data-layout version cannot be overwritten in place yet",
            ));
        }
        match region[lb + 1] {
            // Compact: the data is inline in the header. Rebuild the header with
            // the new inline bytes (relocating it), patching the parent link.
            0 => Ok(WritePlan::Moving(MovingWrite::Compact {
                region,
                bytes: staged_bytes(fd, path),
            })),
            1 => {
                if le - lb < 18 {
                    return Err(Error::EditUnsupported("malformed contiguous data layout"));
                }
                let addr_off = lb + 2;
                let data_addr = StoredAddress::new(u64::from_le_bytes(
                    region[addr_off..addr_off + 8].try_into().unwrap(),
                ));
                let data_size = u64::from_le_bytes(region[lb + 10..lb + 18].try_into().unwrap());

                // Same length and a defined, in-bounds data block: overwrite the
                // bytes straight in place. No header rewrite, no relink. The
                // in-place write targets the data block's absolute file offset.
                //
                // Never for a staged variable-length overwrite, which relocates
                // instead. Its resolution *allocates*  -  a global heap collection,
                // drawn from a freed region where one fits  -  and an in-place write
                // is a live mutation of a block the current root already reaches.
                // Together those put a reference to a just-allocated span inside
                // bytes the live tree names, so a commit failing after the write
                // and before its repoint hands the span back to the free list
                // (`restore_free`) while the image still points into it. The next
                // commit is then free to place something else there, and the
                // dataset reads that object's bytes with every checksum intact.
                //
                // A relocating write has no such window: its new data block is
                // reachable from nothing until the superblock repoint, so a
                // failed attempt leaves the region genuinely dead and re-offering
                // it is sound  -  which is the invariant `restore_free` states.
                if self.fresh_page_boundary.is_none()
                    && fd.vl_string_staging.is_none()
                    && !data_addr.is_undefined(OFFSET_WIDTH)
                    && data_size == fd.raw.len() as u64
                {
                    if let Ok(start) = base.absolute(data_addr) {
                        if start.checked_add(data_size).is_some_and(|e| e <= src.len()) {
                            return Ok(WritePlan::InPlace {
                                data_addr: start,
                                bytes: staged_bytes(fd, path),
                            });
                        }
                    }
                }

                // Length differs or the block was undefined/out of bounds: the new
                // data goes elsewhere and the old extent (if any) is freed. The
                // freed extent is recorded as an absolute file offset, which is the
                // frame the session free list uses.
                let old_extent = if !data_addr.is_undefined(OFFSET_WIDTH) && data_size > 0 {
                    Some((base.absolute(data_addr)?, data_size))
                } else {
                    None
                };
                Ok(WritePlan::Moving(MovingWrite::Contiguous {
                    region,
                    addr_off,
                    bytes: staged_bytes(fd, path),
                    old_extent,
                }))
            }
            // Chunked: overwrite each chunk in place when every new (re-encoded)
            // chunk is the same byte length as its slot, else rebuild and relocate
            // the whole chunk storage. The chunk geometry, filter pipeline, and
            // index type all come from the existing on-disk header (the staged
            // builder carries none  -  chunked/filtered/extensible builders are
            // rejected at the top of this function as "not a value overwrite").
            2 => {
                // Chunked overwrite (in-place or relocating). On a userblock file
                // every stored chunk-index and chunk address is relative to `base`:
                // the in-place path below walks the index on a base-relative view of
                // the file and shifts the resulting write offsets back by `base`,
                // and the relocating path rebuilds the chunk blob with stored
                // addresses (see `write_chunked_relocatable`).
                let dl = DataLayout::parse_with_widths(&region[lb..le], widths)?;
                let DataLayout::Chunked { index, flags, .. } = dl else {
                    return Err(Error::EditUnsupported("dataset is not chunked"));
                };
                if !index.enumerable() {
                    return Err(Error::EditUnsupported(
                        "a chunked dataset with a version-2 B-tree chunk index cannot be \
                         overwritten in place yet",
                    ));
                }

                let ChunkedGeometry {
                    spatial,
                    element_size,
                    raw_size,
                    maxshape,
                } = ChunkedGeometry::new(&fd.dt, &disk_ds, &dl)?;
                let spatial = spatial.to_vec();
                let maxshape = maxshape.map(<[MaxExtent]>::to_vec);

                let padding = fill_msg
                    .map_or(crate::fill_value::PaddingFill::Zero, |(mt, b, e)| {
                        crate::fill_value::PaddingFill::from_message(mt, &region[b..e])
                    });
                let pipeline_message: Option<Vec<u8>> =
                    filter.map(|(fb, fe)| region[fb..fe].to_vec());

                // Reject a pipeline this engine cannot re-encode before anything
                // is split or encoded, so the rejection reaches the preflight on
                // both branches below  -  including the staged one, which does its
                // splitting in the apply phase and so has no other chance to make
                // it.
                let pipeline = pipeline_message
                    .as_deref()
                    .map(crate::filter_pipeline::parse_filter_pipeline)
                    .transpose()?;
                if pipeline
                    .as_ref()
                    .is_some_and(|pipeline| !super::pipeline_reencodable(pipeline))
                {
                    return Err(Error::EditUnsupported(
                        "a chunked dataset using a filter this engine cannot re-encode \
                         cannot be overwritten in place yet",
                    ));
                }
                let overwrite = ChunkedOverwrite {
                    shape: disk_ds.dimensions.clone(),
                    chunk_dims: spatial,
                    element_size,
                    flags,
                    pipeline,
                    padding,
                    dt: fd.dt.clone(),
                };

                // Element bytes still carrying unresolved variable-length
                // references cannot be split here: a filtered chunk's compressed
                // length depends on the heap addresses patched into it, so the
                // split has to follow the patch, which follows a placement the
                // preflight may not make. See `ChunkPayload::Deferred`.
                if fd.vl_string_staging.is_some() {
                    return Ok(WritePlan::Moving(MovingWrite::Chunked {
                        region,
                        overwrite,
                        maxshape,
                        pipeline_message,
                        payload: ChunkPayload::Deferred {
                            bytes: staged_bytes(fd, path),
                        },
                        old_addr: addr,
                    }));
                }

                let new_chunks = overwrite.prepare_chunks(&fd.raw)?;

                // Fast path: overwrite each chunk straight in its slot when every
                // new chunk fits. No header rewrite and no superblock flip  -  the
                // chunk (and index) blocks are reachable from both roots. The index
                // is left untouched when chunks keep their size and rebuilt in place
                // when they shrink. The index walk runs on a base-relative view of
                // the file (so the layout's stored addresses index correctly), and
                // the returned write offsets are shifted back to absolute file
                // offsets by adding `base` (a no-op on a base-0 file).
                if self.fresh_page_boundary.is_none()
                    && let Some(writes) = try_inplace_chunk_writes(
                        &BaseOffsetSource {
                            inner: src,
                            base: base.effective_base(),
                        },
                        &dl,
                        &disk_ds,
                        &overwrite.chunk_dims,
                        raw_size,
                        &new_chunks,
                    )
                {
                    let writes = writes
                        .into_iter()
                        .map(|(off, b)| Ok((base.absolute(StoredAddress::new(off))?, b)))
                        .collect::<Result<Vec<_>, FormatError>>()?;
                    return Ok(WritePlan::InPlaceChunks { writes });
                }

                // Otherwise relocate: rebuild a fresh chunk blob + index at
                // end-of-file (carrying the re-encoded chunk bytes and the source
                // pipeline verbatim), swap the data-layout message in the verbatim
                // header, and free the old chunk storage after the commit lands.
                Ok(WritePlan::Moving(MovingWrite::Chunked {
                    region,
                    overwrite,
                    maxshape,
                    pipeline_message,
                    payload: ChunkPayload::Encoded(new_chunks),
                    old_addr: addr,
                }))
            }
            _ => Err(Error::EditUnsupported(
                "an unsupported data-layout class cannot be overwritten in place yet",
            )),
        }
    }

    /// Returns the plan of a relocating append of the elements of `ab` to the chunked dataset at
    /// `addr`, which [`write_appended_chunks`](Self::write_appended_chunks) writes.
    ///
    /// It keeps each complete chunk of the dataset by its index record, and splits the live
    /// elements of a trailing partial chunk and the appended elements into chunks. It encodes each
    /// of those for a reader at the extent the append publishes: through the filter pipeline,
    /// except a chunk that extends past that extent under
    /// [`ChunkedLayoutFlags::partial_edge_chunks_stored_raw`], which is stored as its data.
    ///
    /// It reads `src`, the file image, and writes nothing. `base` is the userblock base of `src`,
    /// and the dataset's base-relative structures are read through a view shifted by `base`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] if the object at `addr` is not a chunked dataset of
    /// rank 1, unlimited along its first dimension, with a dense grid of chunks indexed by an
    /// Extensible Array whose element format matches the filter pipeline,
    /// if that pipeline is one this crate cannot re-encode, or is lossy and the dataset ends in a
    /// partial chunk, if the elements of `ab` mix types or do not match the dataset's datatype, if
    /// `ab` holds raw bytes for a datatype that a raw append cannot write as they are, if the grown
    /// length overflows a `u64`, or if `base` is past the end of `src`. Returns the errors of the
    /// reads of the dataset's header and chunk index, of [`ChunkedGeometry::new`], of
    /// [`ChunkInfo::read_live_elements`](chunked_read::ChunkInfo::read_live_elements) for the
    /// partial chunk, and of [`PreparedChunk::encode`].
    pub(super) fn prepare_append<S: Source + ?Sized>(
        src: &S,
        addr: u64,
        ab: &AppendBuilder,
        base: AddressContext,
        widths: FormatWidths,
    ) -> Result<MovingWrite, Error> {
        if ab.dt_conflict {
            return Err(Error::AppendUnsupported(
                "append mixes element types in one builder; use one element type per \
                 append_dataset call",
            ));
        }

        let region = Self::gather_oh_messages(src, addr, base)?;

        // Locate the datatype, dataspace, data-layout, and filter-pipeline
        // messages, and detect a group (link) header.
        let mut datatype: Option<(usize, usize)> = None;
        let mut dataspace: Option<(usize, usize)> = None;
        let mut layout: Option<(usize, usize)> = None;
        let mut filter: Option<(usize, usize)> = None;
        // The dataset's own fill value, which the edge overhang of a partial
        // chunk must hold (issue #296). Both message forms, newest wins.
        let mut fill_msg: Option<(MessageType, usize, usize)> = None;
        let mut has_link = false;
        let mut p = 0;
        while let Some((msg_type, body, body_end)) = region.next_message(p)? {
            match msg_type {
                MessageType::DATATYPE => datatype = Some((body, body_end)),
                MessageType::DATASPACE => dataspace = Some((body, body_end)),
                MessageType::DATA_LAYOUT => layout = Some((body, body_end)),
                MessageType::FILTER_PIPELINE => filter = Some((body, body_end)),
                // Versioned beats legacy, and within a type the first wins  -
                // the same rule `Dataset::fill_value_message` and `Located::from_walk`
                // apply, so all three agree on a header carrying more than one.
                MessageType::FILL_VALUE
                    if !matches!(fill_msg, Some((MessageType::FILL_VALUE, ..))) =>
                {
                    fill_msg = Some((msg_type, body, body_end));
                }
                MessageType::FILL_VALUE_OLD if fill_msg.is_none() => {
                    fill_msg = Some((msg_type, body, body_end));
                }
                MessageType::LINK | MessageType::LINK_INFO | MessageType::SYMBOL_TABLE => {
                    has_link = true;
                }
                _ => {}
            }
            p = body_end;
        }
        if has_link {
            return Err(Error::AppendUnsupported(
                "append target is a group, not a dataset",
            ));
        }
        let (dt_b, dt_e) =
            datatype.ok_or(Error::AppendUnsupported("dataset header has no datatype"))?;
        let (ds_b, ds_e) =
            dataspace.ok_or(Error::AppendUnsupported("dataset header has no dataspace"))?;
        let (lb, le) = layout.ok_or(Error::AppendUnsupported(
            "dataset header has no data layout",
        ))?;

        let (disk_dt, _) = hdf5_pure_format::__private::parse_datatype(&region[dt_b..dt_e])?;
        let disk_ds = Dataspace::parse_with_length_width(&region[ds_b..ds_e], widths.lengths)?;
        let dl = DataLayout::parse_with_widths(&region[lb..le], widths)?;

        // Require a chunked dataset indexed by an Extensible Array, which only a
        // version 4 layout message stores.
        let DataLayout::Chunked { index, flags, .. } = &dl else {
            return Err(Error::AppendUnsupported(
                "append requires a chunked dataset",
            ));
        };
        let ChunkIndexLayout::ExtensibleArray { address } = index else {
            return Err(Error::AppendUnsupported(
                "append requires an Extensible-Array-indexed chunked dataset (a single \
                 unlimited dimension under the latest format)",
            ));
        };

        // Require rank 1, unlimited along axis 0.
        if disk_ds.space_type != DataspaceType::Simple || disk_ds.dimensions.len() != 1 {
            return Err(Error::AppendUnsupported(
                "append requires a rank-1 dataset in this release",
            ));
        }
        if !matches!(
            disk_ds.max_dimensions.as_deref(),
            Some([MaxExtent::Unlimited, ..])
        ) {
            return Err(Error::AppendUnsupported(
                "append requires a dataset that is unlimited along its first dimension",
            ));
        }

        let ChunkedGeometry {
            spatial,
            element_size,
            raw_size,
            ..
        } = ChunkedGeometry::new(&disk_dt, &disk_ds, &dl)?;
        let chunk_elems = spatial[0];
        // Validate the appended bytes against the on-disk element type.
        if ab.raw.len() % element_size != 0 {
            return Err(Error::AppendUnsupported(
                "appended byte length is not a whole number of elements",
            ));
        }
        match &ab.elem_dt {
            // A typed append must match the on-disk datatype exactly (class, size,
            // and byte order)  -  this is a value append, not a retype.
            Some(expected) if *expected != disk_dt => {
                return Err(Error::AppendUnsupported(
                    "append datatype does not match the on-disk dataset (wrong element \
                     type or byte order)",
                ));
            }
            Some(_) => {}
            // A raw append accepts the caller's bytes but still rejects any datatype
            // whose flat little-endian bytes cannot be written verbatim: a
            // multi-byte big-endian numeric leaf would silently misencode, and a
            // variable-length or reference leaf embeds heap/object addresses a byte
            // append cannot reproduce. A typed append is byte-order- and
            // class-checked by the datatype-equality arm above.
            None => {
                if !super::datatype_is_raw_appendable(&disk_dt) {
                    return Err(Error::AppendUnsupported(
                        "append_raw onto this dataset's datatype (non-little-endian, \
                         variable-length, or reference) could misencode the bytes; use a \
                         typed append",
                    ));
                }
            }
        }

        let new_elems = (ab.raw.len() / element_size) as u64;
        let current_dim0 = disk_ds.dimensions[0];
        let new_dim0 = current_dim0
            .checked_add(new_elems)
            .ok_or(Error::AppendUnsupported(
                "append would overflow the dataset dimension",
            ))?;

        // The filter pipeline is preserved verbatim in the rebuilt header, parse it
        // to re-encode the new chunks. An engine-unencodable filter is rejected.
        let pipeline_message: Option<Vec<u8>> = filter.map(|(fb, fe)| region[fb..fe].to_vec());
        let has_filters = pipeline_message.is_some();
        let pipeline = match &pipeline_message {
            Some(pm) => {
                let parsed = crate::filter_pipeline::parse_filter_pipeline(pm)?;
                if !super::pipeline_reencodable(&parsed) {
                    return Err(Error::AppendUnsupported(
                        "dataset uses a filter this engine cannot re-encode",
                    ));
                }
                Some(parsed)
            }
            None => None,
        };

        if base.effective_base().get() > src.len() {
            return Err(Error::AppendUnsupported(
                "userblock base address past end-of-file",
            ));
        }
        let view = BaseOffsetSource {
            inner: src,
            base: base.effective_base(),
        };

        // The rebuilt index's element format (bare address vs address+size+mask) is
        // chosen by `has_filters`, it must agree with the source index's client id,
        // or the kept chunks  -  carried by metadata into the new index  -  would be
        // re-encoded in the wrong element width.
        if let Some(idx_addr) = *address {
            let array =
                crate::extensible_array::read_header(&view, crate::file_writer::WIDTHS, idx_addr)?;
            if array.is_filtered() != has_filters {
                return Err(Error::AppendUnsupported(
                    "dataset filter metadata is inconsistent (chunk-index client id \
                     disagrees with the filter pipeline)",
                ));
            }
        }

        // Enumerate the existing chunks (base-relative addresses) and require a
        // dense grid: `plan_dense_grid` returns the chunks in index order and
        // `None` on any hole, duplicate, or count mismatch against the dimension.
        let infos = chunked_read::enumerate_chunks_from_source(
            &view,
            &dl,
            &disk_ds,
            crate::file_writer::WIDTHS,
        )?;
        let grid = chunked_read::plan_dense_grid(infos, &disk_ds.dimensions, spatial).ok_or(
            Error::AppendUnsupported(
                "dataset has a sparse or inconsistent chunk grid; cannot append",
            ),
        )?;
        let grid_order = grid.grid_order;

        // Complete chunks are kept by metadata, a trailing partial chunk (when the
        // current length is not chunk-aligned) is rewritten.
        let n_full = (current_dim0 / chunk_elems).to_usize()?;
        let has_partial = current_dim0 % chunk_elems != 0;

        // Rewriting that chunk decodes and re-encodes committed values, which a
        // *lossy* pipeline does not reproduce: a ZFP block re-quantizes to fit
        // its new contents, and float D-scale packs offsets from a per-chunk
        // minimum that a smaller appended value moves (measured: 1.45 became 1.5
        // at one decimal). Rejected here as on the immediate path (#407).
        if has_partial
            && let Some(pl) = &pipeline
            && !super::pipeline_lossless(pl)
        {
            return Err(Error::AppendUnsupported(LOSSY_TAIL_REFUSAL));
        }

        let mut kept_chunks: Vec<ChunkRecord> = Vec::with_capacity(n_full);
        for ci in grid_order.iter().take(n_full) {
            kept_chunks.push(ChunkRecord {
                address: ci.address,
                stored_size: ci.chunk_size.get(),
                // Preserve the source mask verbatim: a C/h5py file records a nonzero
                // mask for a chunk whose filter was skipped (e.g. deflate on
                // incompressible data), and forcing it to 0 would corrupt that chunk.
                filter_mask: ci.filter_mask,
            });
        }

        // Build the raw byte region for the tail (from the last chunk boundary to
        // the new end): the live prefix of any rewritten partial chunk, then the
        // appended bytes.
        let ctx = ChunkContext::from_datatype(spatial, &disk_dt)?;
        // One encoder and decoder across the appended tail. See `FilterScratch`.
        let mut scratch = FilterScratch::new();
        let (mut tail_raw, old_tail_extent) = if has_partial {
            let partial = &grid_order[n_full];
            let live = partial.read_live_elements(
                &view,
                &mut scratch,
                *flags,
                pipeline.as_ref(),
                ctx,
                current_dim0,
            )?;
            // The old partial chunk's data block is dead once the new index lands.
            let extent = (base.absolute(partial.address)?, partial.chunk_size.get());
            (live, Some(extent))
        } else {
            (Vec::new(), None)
        };
        tail_raw.extend_from_slice(&ab.raw);

        // Split the tail into full chunk buffers. The final chunk's overhang takes the
        // dataset's fill value, which is what a reader returns from those slots once a
        // later append or resize reaches them (#296).
        let tail_len_elems = new_dim0 - (n_full as u64) * chunk_elems;
        let padding = fill_msg.map_or(crate::fill_value::PaddingFill::Zero, |(mt, b, e)| {
            crate::fill_value::PaddingFill::from_message(mt, &region[b..e])
        });
        let split = chunked_write::split_into_chunks(
            &tail_raw,
            &[tail_len_elems],
            spatial,
            element_size,
            padding.pattern(element_size),
        )
        .map_err(Error::Format)?;
        let new_chunks = (n_full.to_u64()..)
            .zip(split)
            .map(|(index, logical)| {
                let offset = index * chunk_elems;
                let published = ChunkDecoding::at_extent(*flags, spatial, &[new_dim0], &[offset]);
                let encoding = ChunkEncoding::for_decoding(published);
                PreparedChunk::encode(&mut scratch, pipeline.as_ref(), ctx, encoding, logical)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Grow the dataspace along axis 0, preserving the (unlimited) max-dims.
        let mut grown = disk_ds.clone();
        grown.dimensions[0] = new_dim0;
        let new_dataspace_body = grown.serialize(LENGTH_SIZE);

        let chunk_dims_u32: Vec<u32> = spatial
            .iter()
            .map(|&dim| dim.narrow::<u32>())
            .collect::<Result<_, _>>()?;

        Ok(MovingWrite::AppendedChunks {
            region,
            new_dataspace_body,
            flags: *flags,
            chunk_dims_u32,
            element_size,
            raw_size,
            has_filters,
            kept_chunks,
            new_chunks,
            old_addr: addr,
            old_tail_extent,
        })
    }

    /// Read the object at `addr` in the source buffer `d`  -  and, for a group, its
    /// whole subtree  -  into an owned [`CopyTree`], the read half of an object copy.
    /// No bytes are written, this both validates that the subtree is copyable and
    /// captures the bytes the write half ([`write_copy_subtree`](Self::write_copy_subtree))
    /// later appends, so the source buffer need not outlive the read.
    ///
    /// `src` is the image the source object lives in: this session's own file image
    /// for an in-file [`copy`](Self::copy), or another file's image for a cross-file
    /// [`copy_from`](Self::copy_from). `base` is that image's userblock base (the
    /// session's own base for an in-file copy, always 0 for a cross-file copy, whose
    /// source is gated to base 0): the stored, base-relative addresses read out of
    /// the source headers are shifted by it to index `src`. When `cross_file` is set,
    /// every copied object header is additionally screened by
    /// [`super::reject_foreign_addresses`]  -  verbatim bytes that embed a *source-file*
    /// absolute address (variable-length or reference data, a committed datatype)
    /// would dangle in another file and are rejected, whereas an in-file copy keeps
    /// them valid by sharing the source file's heaps and objects.
    pub(super) fn read_copy_subtree<S: Source + ?Sized>(
        src: &S,
        access_mode: AccessMode,
        addr: u64,
        depth: u32,
        cross_file: bool,
        base: AddressContext,
        widths: FormatWidths,
    ) -> Result<CopyTree, Error> {
        if depth >= MAX_COPY_DEPTH {
            return Err(Error::EditUnsupported(
                "copy source nests too deeply (possible hard-link cycle)",
            ));
        }
        // `base` is the userblock base of the image `src`: this session's own base
        // for an in-file copy, and always 0 for a cross-file copy (the source is
        // gated to base 0 in `copy_from`). `addr` is an absolute offset into `src`,
        // the stored (base-relative) addresses `read_object` returns for contiguous
        // data, chunk storage, and child links are converted to absolute offsets by
        // adding `base` before `src` is read or a child is descended into.
        match Self::read_object(src, access_mode, addr, base)? {
            ObjModel::DatasetVerbatim {
                region,
                dense_attrs,
            } => {
                if cross_file {
                    super::reject_foreign_addresses(&region)?;
                    super::reject_foreign_dense_attrs(&dense_attrs.attrs)?;
                }
                Ok(CopyTree::DatasetVerbatim {
                    region,
                    dense_attrs,
                })
            }
            ObjModel::DatasetContiguous {
                region,
                addr_off,
                data_addr,
                data_size,
                dense_attrs,
            } => {
                if cross_file {
                    super::reject_foreign_addresses(&region)?;
                    super::reject_foreign_dense_attrs(&dense_attrs.attrs)?;
                }
                // Storage the source never allocated is copied as storage, not as
                // the values reading it returns with. The reference library does
                // not allocate a contiguous dataset's data until something is
                // written to it, leaving the layout message's address undefined,
                // and reading one returns the fill value for every element (#292)
                //  -  so a copy that materialized what it read would turn a
                // schema-only dataset into a fully written one of the size its
                // shape declares. `repack` carries the same shape through as
                // unallocated (#293), this is the copy path's half of it (#336).
                //
                // Only reached once external storage is out of the way: it uses
                // this same undefined address for a dataset that *does* hold
                // data, and `read_object` rejects it by name above.
                let data = if data_addr.is_undefined(OFFSET_WIDTH) {
                    None
                } else {
                    // The data block is read at its absolute offset into `src`.
                    let start = base.absolute(data_addr)?;
                    start
                        .checked_add(data_size)
                        .filter(|&e| e <= src.len())
                        .ok_or(Error::EditUnsupported("dataset data is out of bounds"))?;
                    let len = data_size.to_usize()?;
                    Some(src.read_exact_at(start, len)?)
                };
                Ok(CopyTree::DatasetContiguous {
                    region,
                    addr_off,
                    data,
                    dense_attrs,
                })
            }
            ObjModel::DatasetChunked {
                region,
                dense_attrs,
            } => {
                // Screen the verbatim header on the cross-file path. This rejects a
                // variable-length or reference datatype (whose chunk payload embeds
                // source-file global-heap / object addresses that would dangle in
                // another file) and any shared message  -  exactly the forms repack
                // also rejects for a cross-file verbatim chunk copy. An in-file copy
                // keeps them valid by sharing the source file's heaps.
                if cross_file {
                    super::reject_foreign_addresses(&region)?;
                    super::reject_foreign_dense_attrs(&dense_attrs.attrs)?;
                }
                let ChunkedHeaderParts {
                    dt,
                    ds,
                    layout,
                    pipeline_message,
                } = parse_chunked_header(&region, widths)?;
                let DataLayout::Chunked { index, flags, .. } = layout else {
                    return Err(Error::EditUnsupported("dataset is not chunked"));
                };
                if !index.enumerable() {
                    return Err(Error::EditUnsupported(
                        "a chunked dataset with a version-2 B-tree chunk index cannot be \
                         copied in place yet",
                    ));
                }
                let ChunkedGeometry {
                    spatial: chunk_dims,
                    element_size,
                    raw_size: _,
                    maxshape,
                } = ChunkedGeometry::new(&dt, &ds, &layout)?;
                let chunk_dims = chunk_dims.to_vec();
                let maxshape = maxshape.map(<[MaxExtent]>::to_vec);

                // The layout's chunk-index address and every chunk address it leads
                // to are stored base-relative, so enumerate and read on a
                // base-relative view of the source image (the identity on a base-0
                // file). The returned addresses are then offsets into `dview`.
                let dview = BaseOffsetSource {
                    inner: src,
                    base: base.effective_base(),
                };

                // Enumerate the source chunks and map them onto a dense grid, a
                // sparse (holed/unallocated) dataset cannot be reproduced by the
                // verbatim layout path, which needs every grid slot filled.
                let infos = chunked_read::enumerate_chunks_from_source(
                    &dview,
                    &layout,
                    &ds,
                    crate::file_writer::WIDTHS,
                )?;
                let grid = chunked_read::plan_dense_grid(infos, &ds.dimensions, &chunk_dims).ok_or(
                    Error::EditUnsupported(
                        "a chunked dataset with unallocated (sparse) chunks cannot be copied in place yet",
                    ),
                )?;
                if grid.grid_order.is_empty() {
                    return Err(Error::EditUnsupported(
                        "an empty chunked dataset cannot be copied in place yet",
                    ));
                }

                // Capture each chunk's already-compressed bytes (no decode) into an
                // owned buffer, in dense row-major grid order, so the copy can be
                // written after the source buffer is gone (cross-file copy reads at
                // staging time). Sizes and masks are carried verbatim.
                let mut meta = Vec::with_capacity(grid.grid_order.len());
                let mut chunk_bytes = Vec::with_capacity(grid.grid_order.len());
                for ci in &grid.grid_order {
                    let len = ci.chunk_size.to_usize()?;
                    ci.address
                        .get()
                        .checked_add(len as u64)
                        .filter(|&e| e <= dview.len())
                        .ok_or(Error::EditUnsupported("chunk data is out of bounds"))?;
                    chunk_bytes.push(dview.read_exact_at(ci.address.get(), len)?);
                    meta.push(ChunkMeta {
                        compressed_size: ci.chunk_size.get(),
                        filter_mask: ci.filter_mask,
                    });
                }

                Ok(CopyTree::DatasetChunked {
                    region,
                    shape: ds.dimensions.clone(),
                    chunk_dims,
                    element_size,
                    flags,
                    maxshape,
                    pipeline_message,
                    meta,
                    chunk_bytes,
                    dense_attrs,
                })
            }
            ObjModel::Group {
                non_link_region,
                children,
                dense_attrs,
            } => {
                if cross_file {
                    super::reject_foreign_addresses(&non_link_region)?;
                    super::reject_foreign_dense_attrs(&dense_attrs.attrs)?;
                }
                let mut kids = Vec::with_capacity(children.len());
                for (name, creation_order, child) in children {
                    // `read_copy_subtree` reads `src` at absolute offsets.
                    let child = base.absolute(child)?;
                    kids.push((
                        name,
                        creation_order,
                        Self::read_copy_subtree(
                            src,
                            access_mode,
                            child,
                            depth + 1,
                            cross_file,
                            base,
                            widths,
                        )?,
                    ));
                }
                Ok(CopyTree::Group {
                    non_link_region,
                    children: kids,
                    dense_attrs,
                })
            }
        }
    }

    /// Append the fresh copies described by `node` (data blobs and headers) into
    /// this session at end-of-file or into reusable freed regions, returning the
    /// new object-header address of the copied root. The write half of an object
    /// copy, children are written before their parent group so each parent links
    /// its children's new addresses, and a contiguous dataset's data-address field
    /// is repointed at the freshly-written copy. Every address the copy writes into
    /// a header (a contiguous data block, a child link) is stored relative to the
    /// userblock base (`- base`, a no-op on a base-0 file), the chunked storage and
    /// dense attribute heaps are laid out base-relative by their own builders.
    pub(super) fn write_copy_subtree(&mut self, node: &CopyTree) -> Result<u64, Error> {
        let base = self.addresses;
        match node {
            CopyTree::DatasetVerbatim {
                region,
                dense_attrs,
            } => {
                let mut region = region.clone();
                self.append_dense_attrs(&mut region, dense_attrs.clone())?;
                let oh = super::build_v2_object_header(&region)?;
                self.alloc_or_append_typed(&oh, AllocationClass::Metadata)
            }
            CopyTree::DatasetContiguous {
                region,
                addr_off,
                data,
                dense_attrs,
            } => {
                let mut region = region.clone();
                // Storage the source never allocated is copied as storage: there
                // is no block to place, and the region already carries the
                // undefined address the source stored, so the copy declares the
                // same empty storage (issue #336).
                if let Some(data) = data {
                    let new_data_addr = self.alloc_or_append_typed(data, AllocationClass::Raw)?;
                    // The placement is an absolute offset, the data-layout
                    // address field stores it relative to the userblock base.
                    let relative = base.relative(new_data_addr)?.get();
                    region.bytes_mut()[*addr_off..*addr_off + 8]
                        .copy_from_slice(&relative.to_le_bytes());
                }
                // The dense heap is placed independently of the data  -  it is
                // built for whatever address it gets (see `append_dense_attrs`),
                // so no ordering between the two is owed.
                self.append_dense_attrs(&mut region, dense_attrs.clone())?;
                let oh = super::build_v2_object_header(&region)?;
                self.alloc_or_append_typed(&oh, AllocationClass::Metadata)
            }
            CopyTree::DatasetChunked {
                region,
                shape,
                chunk_dims,
                element_size,
                flags,
                maxshape,
                pipeline_message,
                meta,
                chunk_bytes,
                dense_attrs,
            } => self.write_chunked_relocatable(
                region,
                shape,
                chunk_dims,
                *element_size,
                *flags,
                maxshape.as_deref(),
                pipeline_message.as_deref(),
                meta,
                &SliceChunkProvider {
                    chunks: chunk_bytes,
                },
                dense_attrs.clone(),
            ),
            CopyTree::Group {
                non_link_region,
                children,
                dense_attrs,
            } => {
                let mut region = non_link_region.clone();
                for (name, creation_order, child) in children {
                    let new_child = self.write_copy_subtree(child)?;
                    // The link target is stored relative to the userblock base.
                    // A copied link keeps the creation index the source recorded
                    // for it, so a copy of a group that tracks link creation
                    // order carries the same order as its source  -  the Link Info
                    // message recording that order is copied verbatim beside it.
                    region.push_link(name, base.relative(new_child)?, *creation_order);
                }
                // The dense heap is built for whatever address it is placed at
                // (see `append_dense_attrs`), so it needs no ordering against the
                // children's headers and data.
                self.append_dense_attrs(&mut region, dense_attrs.clone())?;
                let oh = super::build_v2_object_header(&region)?;
                self.alloc_or_append_typed(&oh, AllocationClass::Metadata)
            }
        }
    }

    /// Writes the chunk storage and the header of a chunked dataset, and returns the new header
    /// address, for a chunked copy ([`CopyTree::DatasetChunked`]) and a relocating chunked
    /// overwrite ([`MovingWrite::Chunked`]).
    ///
    /// The chunk data block and index are planned with
    /// [`chunked_write::plan_chunked_data_verbatim`] and emitted with
    /// [`chunked_write::emit_chunked_data_verbatim`], which writes the stored bytes of each of
    /// `chunks` in dense row-major grid order, with the sizes and filter masks of `meta`. No chunk
    /// is decoded or encoded again. The new layout message stores `flags`, and the index stores
    /// filtered chunk records where the caller passes a `pipeline_message`.
    ///
    /// Like [`build_chunked_dataset`](Self::build_chunked_dataset), the block is sized from its
    /// plan before it is placed, in a freed region that fits it or at the end of the file. The
    /// verbatim header `region` keeps every message except the data layout message, which the
    /// planned one replaces. Any dense attribute heap is placed, and then the header, in freed
    /// space or at the end of the file.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`chunked_write::plan_chunked_data_verbatim`] and
    /// [`chunked_write::emit_chunked_data_verbatim`], and of placing the block, a dense attribute
    /// heap, and the header.
    #[expect(
        clippy::too_many_arguments,
        reason = "the chunked rebuild needs the full geometry, \
        pipeline, and chunk payloads; bundling them into a struct would only move the list"
    )]
    pub(super) fn write_chunked_relocatable(
        &mut self,
        region: &OhRegion,
        shape: &[u64],
        chunk_dims: &[u64],
        element_size: NonZeroUsize,
        flags: ChunkedLayoutFlags,
        maxshape: Option<&[MaxExtent]>,
        pipeline_message: Option<&[u8]>,
        meta: &[ChunkMeta],
        chunks: &dyn ChunkProvider,
        dense_attrs: DenseAttrSet,
    ) -> Result<u64, Error> {
        // Plan once at a provisional base purely to size the data region: the plan
        // walks chunk *sizes* and sizes the index from its layout, so it uses stored sizes alone, and every address it embeds sits in a fixed-width
        // field, so its total length is the same wherever the blob lands. That is
        // what lets the address be chosen  -  a freed region or end-of-file  -  before
        // the bytes exist. This plan is discarded, the one built at the real base
        // below is what the emit works from.
        let sizing = chunked_write::plan_chunked_data_verbatim(
            meta,
            shape,
            chunk_dims,
            element_size,
            flags,
            pipeline_message,
            StoredAddress::new(0),
            maxshape,
        )?;
        let (_addr, layout_message) =
            self.place_relocatable(sizing.plan.total_len, AllocationClass::Raw, |stored_base| {
                // Re-plan at the address the blob really occupies, so its embedded
                // addresses resolve to their real file offsets once the reader adds
                // the userblock base back (see `build_chunked_dataset`).
                let layout = chunked_write::plan_chunked_data_verbatim(
                    meta,
                    shape,
                    chunk_dims,
                    element_size,
                    flags,
                    pipeline_message,
                    stored_base,
                    maxshape,
                )?;
                let mut buf =
                    Vec::with_capacity(usize::try_from(layout.plan.total_len).unwrap_or(0));
                chunked_write::emit_chunked_data_verbatim(&mut buf, &layout.plan, chunks)?;
                Ok((buf, layout.layout_message))
            })?;
        // Swap the data-layout message for the rebuilt one, keep every other header
        // message (datatype, dataspace, fill value, filter pipeline, attributes)
        // verbatim.
        let mut new_region = replace_layout_message(region, &layout_message)?;
        self.append_dense_attrs(&mut new_region, dense_attrs)?;
        let oh = super::build_v2_object_header(&new_region)?;
        self.alloc_or_append_typed(&oh, AllocationClass::Metadata)
    }

    /// Apply a relocating value overwrite (`write_dataset` resize / compact
    /// rewrite): write the new data and a rewritten object header at end-of-file
    /// (or into reusable freed space) and return the new header address. The
    /// caller patches the parent group's link to this address. The old data
    /// extent (for a resized contiguous dataset) is freed separately, after the
    /// commit's superblock repoint, so it is never reused mid-commit.
    pub(super) fn write_moving(&mut self, mw: &MovingWrite) -> Result<u64, Error> {
        let base = self.addresses;
        match mw {
            MovingWrite::Contiguous {
                region,
                addr_off,
                bytes,
                ..
            } => {
                let raw = self.resolve_overwrite_bytes(bytes)?;
                let new_data_addr = self.alloc_or_append_typed(&raw, AllocationClass::Raw)?;
                let mut region = region.clone();
                // The placement is an absolute file offset, the contiguous
                // data-layout field stores it relative to the userblock base (`-
                // base`, a no-op on a base-0 file).
                let relative = base.relative(new_data_addr)?.get();
                region.bytes_mut()[*addr_off..*addr_off + 8]
                    .copy_from_slice(&relative.to_le_bytes());
                // The data size field follows the 8-byte address in the contiguous
                // layout body, keep it in sync with the new length.
                let size_off = *addr_off + 8;
                let raw_len = raw.len() as u64;
                region.bytes_mut()[size_off..size_off + 8].copy_from_slice(&raw_len.to_le_bytes());
                let oh = super::build_v2_object_header(&region)?;
                self.alloc_or_append_typed(&oh, AllocationClass::Metadata)
            }
            MovingWrite::Compact { region, bytes } => {
                let raw = self.resolve_overwrite_bytes(bytes)?;
                let region = super::rebuild_compact_layout_region(region, &raw)?;
                let oh = super::build_v2_object_header(&region)?;
                self.alloc_or_append_typed(&oh, AllocationClass::Metadata)
            }
            MovingWrite::Chunked {
                region,
                overwrite,
                maxshape,
                pipeline_message,
                payload,
                ..
            } => {
                let deferred;
                let chunks = match payload {
                    ChunkPayload::Encoded(chunks) => chunks,
                    ChunkPayload::Deferred { bytes } => {
                        // Resolve before splitting: the heap collections have to
                        // be placed, and their addresses patched into the element
                        // references, before those references are cut into chunks
                        // and  -  on a filtered dataset  -  compressed over.
                        let raw = self.resolve_overwrite_bytes(bytes)?;
                        deferred = overwrite.prepare_chunks(&raw)?;
                        &deferred
                    }
                };
                let meta: Vec<ChunkMeta> = chunks
                    .iter()
                    .map(|chunk| ChunkMeta {
                        compressed_size: chunk.stored_size(),
                        filter_mask: chunk.filter_mask(),
                    })
                    .collect();
                self.write_chunked_relocatable(
                    region,
                    &overwrite.shape,
                    &overwrite.chunk_dims,
                    overwrite.element_size,
                    overwrite.flags,
                    maxshape.as_deref(),
                    pipeline_message.as_deref(),
                    &meta,
                    &PreparedChunkProvider { chunks },
                    DenseAttrSet::default(),
                )
            }
            MovingWrite::AppendedChunks {
                region,
                new_dataspace_body,
                flags,
                chunk_dims_u32,
                element_size,
                raw_size,
                has_filters,
                kept_chunks,
                new_chunks,
                ..
            } => self.write_appended_chunks(
                region,
                new_dataspace_body,
                *flags,
                chunk_dims_u32,
                *element_size,
                *raw_size,
                *has_filters,
                kept_chunks,
                new_chunks,
            ),
            MovingWrite::AttrEdit { region, attrs, .. } => {
                // `region` already carries what the commit preflight could resolve,
                // `place_edited_attrs` places the rest, exactly as the
                // group-attribute apply loop does. Then build and place the
                // relocated dataset header: the data-layout message is untouched,
                // so the dataset's chunk data and index stay in place and only the
                // header moves.
                let mut region = region.clone();
                self.place_edited_attrs(&mut region, attrs.clone())?;
                let oh = super::build_v2_object_header(&region)?;
                self.alloc_or_append_typed(&oh, AllocationClass::Metadata)
            }
        }
    }

    /// Apply a relocating append ([`MovingWrite::AppendedChunks`]): place the new
    /// (and any rewritten trailing) chunk bytes, rebuild a fresh
    /// Extensible Array over the kept plus appended chunks, grow the dataspace and
    /// repoint the data layout in the verbatim header `region`, and write the
    /// relocated header. Returns the new header address, the caller patches the
    /// parent link. The kept chunk data is untouched (referenced by both the old
    /// and new index during the commit), the old index/header/trailing chunk are
    /// freed only after the superblock repoint.
    ///
    /// The new chunks and the new index go down together where one freed region
    /// holds them and a paged file is what makes that worth requesting, and one at a
    /// time otherwise, the body says why.
    #[expect(
        clippy::too_many_arguments,
        reason = "the append rebuild needs the header region, grown dataspace, layout flags, \
        chunk geometry, and both chunk sets; bundling them into a struct would only move the list"
    )]
    pub(super) fn write_appended_chunks(
        &mut self,
        region: &OhRegion,
        new_dataspace_body: &[u8],
        flags: ChunkedLayoutFlags,
        chunk_dims_u32: &[u32],
        element_size: NonZeroUsize,
        chunk_bytes: u64,
        has_filters: bool,
        kept_chunks: &[ChunkRecord],
        new_chunks: &[PreparedChunk],
    ) -> Result<u64, Error> {
        let base = self.addresses;
        // Where the appended chunks and the rebuilt index go, and whether they go
        // together.
        //
        // Contiguity is worth something on a *paged* file. The index goes in a raw
        // page, not a metadata one, because that is where this crate puts every
        // chunk index  -  and the reclaim side can only distinguish a raw index from the
        // metadata one the reference library writes by the chunk data it abuts
        // ([`index_is_provably_raw`](Self::index_is_provably_raw)). An index that
        // abuts none of its chunks cannot be placed by the commit that supersedes
        // it, so its bytes are held as dead until the page around them empties.
        // Laying the new chunks and the new index down as one blob  -  the way the
        // from-scratch writer and `write_chunked_relocatable` lay out chunked
        // storage  -  keeps that proof available (issue #388).
        //
        // It is worth nothing on a file that is not paged, which has no page types
        // to keep apart, and it must never be bought with reuse: one blob needs one
        // freed region big enough for the lot, where the chunks placed one at a
        // time fill several smaller ones. So the blob form is taken only when a
        // single region already holds it, and otherwise each chunk is placed on its
        // own and the index after them. That fallback keeps the proof only where
        // the last chunk placed ends up immediately below the index  -  which is what
        // happens when both append at end-of-file, and not what happens when the
        // chunks find holes to sit in. An index left abutting nothing is recorded
        // as dead in place of free, which costs space and never correctness.
        //
        // Sizing runs at a provisional base, exactly as `write_chunked_relocatable`
        // sizes its blob: an Extensible Array's length follows the slot layout and
        // every address it embeds sits in a fixed-width field, so the total is the
        // same wherever the array lands, and `place` rejects any disagreement.
        //
        // The element width comes from the chunk geometry in place of from the
        // chunk list, so a rebuild declares the same width the original index did  -
        // including when the dataset it grows was created empty.
        // Rank 1 and unlimited along axis 0 (`prepare_append` rejects anything
        // else), so every chunk's index slot is its position in the grid and the
        // array is dense from zero.
        //
        // The index records each kept chunk with its own mask, which may be nonzero, and each
        // appended chunk with the mask it was prepared with.
        let chunk_total: u64 = new_chunks.iter().map(PreparedChunk::stored_size).sum();
        let placed_chunks = |blob_stored: StoredAddress| -> Vec<ChunkRecord> {
            let mut combined: Vec<ChunkRecord> = kept_chunks.to_vec();
            let mut next = blob_stored;
            for chunk in new_chunks {
                combined.push(ChunkRecord {
                    address: next,
                    stored_size: chunk.stored_size(),
                    filter_mask: chunk.filter_mask(),
                });
                next = next.offset(chunk.stored_size());
            }
            combined
        };
        let sizing = placed_chunks(StoredAddress::new(0));
        let ea_len = hdf5_pure_format::__private::extensible_array_len(
            &IndexSlots::dense(&sizing),
            chunk_bytes,
            OFFSET_WIDTH,
            LENGTH_WIDTH,
            has_filters,
        );
        let ea = |slots: &IndexSlots<'_>, at: StoredAddress| -> Result<Vec<u8>, Error> {
            hdf5_pure_format::__private::build_extensible_array_at(
                slots,
                chunk_bytes,
                OFFSET_WIDTH,
                LENGTH_WIDTH,
                has_filters,
                at,
            )
            .map_err(Error::Format)
        };
        // Reserved by hand in place of through `reserve`, which would fall back to
        // end-of-file: this requests only whether a freed region holds the whole blob,
        // and takes the per-chunk path when none does.
        let blob = if self.space.is_paged() {
            self.alloc_free(chunk_total + ea_len, AllocationClass::Raw)
        } else {
            None
        };
        let ea_stored = match blob {
            Some(blob_stored) => {
                // The region is out of the free lists from here, so every way the
                // build can fail has to put it back: a `?` straight out of this arm
                // would leave it neither free nor written for the rest of the
                // session. Collected into a `Result` and handed back on the way out.
                let placed = (|| -> Result<StoredAddress, Error> {
                    let combined = placed_chunks(blob_stored);
                    let mut buf =
                        Vec::with_capacity(usize::try_from(chunk_total + ea_len).unwrap_or(0));
                    for chunk in new_chunks {
                        buf.extend_from_slice(chunk.bytes());
                    }
                    buf.extend_from_slice(&ea(
                        &IndexSlots::dense(&combined),
                        blob_stored.offset(chunk_total),
                    )?);
                    self.place(
                        Placement::Reused {
                            addr: base.resolve(blob_stored)?.get(),
                            len: chunk_total + ea_len,
                        },
                        &buf,
                    )?;
                    Ok(blob_stored.offset(chunk_total))
                })();
                match placed {
                    Ok(ea_stored) => ea_stored,
                    Err(e) => {
                        self.release_raw_alloc(blob_stored, chunk_total + ea_len);
                        return Err(e);
                    }
                }
            }
            None => {
                // A chunk embeds no addresses of its own, so it can go anywhere and
                // the index below simply records where it went.
                let mut combined: Vec<ChunkRecord> = kept_chunks.to_vec();
                for chunk in new_chunks {
                    let abs = self.alloc_or_append_typed(chunk.bytes(), AllocationClass::Raw)?;
                    combined.push(ChunkRecord {
                        address: base.relative(abs)?,
                        stored_size: chunk.stored_size(),
                        filter_mask: chunk.filter_mask(),
                    });
                }
                let at =
                    self.reserve_with_page_type(ea_len, AllocationClass::Metadata, PageType::Raw)?;
                let bytes = ea(&IndexSlots::dense(&combined), base.relative(at.address())?)?;
                let ea_addr = self.place(at, &bytes)?;
                base.relative(ea_addr)?
            }
        };

        // Swap the dataspace (grown) and data-layout (repointed at the new index)
        // messages, every other header message is preserved verbatim.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "element size is a datatype byte width that fits u32"
        )]
        let layout_body = DataLayout::encode_chunked(
            flags,
            chunk_dims_u32,
            element_size.get() as u32,
            ChunkIndexInfo::ExtensibleArray,
            ea_stored,
            OFFSET_WIDTH,
            LENGTH_WIDTH,
        )?;
        let region = replace_dataspace_message(region, new_dataspace_body)?;
        let region = replace_layout_message(&region, &layout_body)?;
        let oh = super::build_v2_object_header(&region)?;
        self.alloc_or_append_typed(&oh, AllocationClass::Metadata)
    }

    /// Resolve a staged value overwrite's element bytes: place any global heap
    /// collections its variable-length references name, patch their addresses
    /// in, and return the bytes ready to write (issue #321). An overwrite that
    /// stages nothing  -  every one but a `with_vlen_strings` one  -  is handed
    /// back untouched.
    ///
    /// This allocates, so every caller is in the apply phase  -  the preflight
    /// that chose the plan only reads.
    ///
    /// It also carries the reclaim. The collections a *previous* overwrite of
    /// this same path placed are handed to the commit's free list, and the ones
    /// placed here are recorded in their stead, so rotating a dataset's strings
    /// reaches a steady state in place of leaking a generation per commit. Only
    /// collections this session placed are reclaimed by an overwrite, and only while
    /// nothing has been able to name them twice  -  see
    /// [`vl_overwrite_heaps`](Self::vl_overwrite_heaps) for why that metadata
    /// is the whole proof, and `repack` for the collections it cannot make.
    pub(super) fn resolve_overwrite_bytes<'b>(
        &mut self,
        bytes: &'b OverwriteBytes,
    ) -> Result<Cow<'b, [u8]>, Error> {
        let Some(vlen) = &bytes.vlen else {
            // Nothing to resolve, and nothing to copy: every overwrite but a
            // variable-length one takes this path, and the in-place loop writes
            // straight through the borrow. Cloning here would charge each of
            // them a second copy of its whole payload, which the bounded
            // backing in particular is built not to pay.
            return Ok(Cow::Borrowed(&bytes.raw));
        };
        let staging = &vlen.staging;
        // Read in place of removed: the record is replaced (not consumed) when
        // the new collections land just below.
        let superseded = self
            .vl_overwrite_heaps
            .get(&vlen.path)
            .cloned()
            .unwrap_or_default();
        let mut raw = bytes.raw.clone();
        let base = self.addresses;
        // A staging with no collections patches nothing, which is the right
        // result for it: an element with no heap object keeps the zero address
        // that reads back as null. It falls out of the general path in place of
        // being a case of its own  -  `patch_vl_refs_mrequested` has one offset per
        // object, so no offsets is no iterations  -  and it still records (an
        // empty record) and still supersedes.
        let addrs = self.place_vl_collections(&staging.collections)?;
        type_builders::patch_vl_refs_masked(&mut raw, &staging.patch_offsets, &addrs);
        // `place_vl_collections` returns base-relative, since that is what an
        // element reference stores, the free list is absolute.
        let placed = addrs
            .iter()
            .zip(&staging.collections)
            .map(|(&a, c)| Ok((base.absolute(StoredAddress::new(a))?, c.len() as u64)))
            .collect::<Result<Vec<_>, FormatError>>()?;
        self.vl_overwrite_heaps.insert(vlen.path.clone(), placed);
        self.superseded_heaps.extend(superseded);
        Ok(Cow::Owned(raw))
    }

    /// Lay out a chunked / filtered / extensible dataset and return its object
    /// header bytes (which the caller links into the parent group).
    ///
    /// The chunk data and index (fixed-array / extensible-array, with any filter
    /// pipeline applied) are produced as one relocatable blob, whose internal
    /// layout  -  and therefore total size  -  is independent of the base address it
    /// is given. That is what lets the dataset be *sized before it is placed*
    /// ([`chunked_write::chunked_data_len`]): the blob goes into a freed region big enough to
    /// hold it, or at end-of-file when there is none, and is then assembled for
    /// whichever address it got, so every absolute address it embeds (chunk
    /// addresses, index-structure addresses, the addresses in the data-layout
    /// message) lands exactly where the bytes are written. The header is then
    /// built with [`file_writer::build_chunked_dataset_oh`], which the whole-file
    /// writer also uses. The header is byte-identical to one written fresh.
    ///
    /// The dataset's single pass of the filter pipeline happens in
    /// [`chunked_write::compress_chunks`], before either decision. Sizing and assembly work from
    /// the compressed set, so choosing an address never costs a recompression
    /// (issue #261).
    pub(super) fn build_chunked_dataset(&mut self, fd: &FlatDataset) -> Result<Vec<u8>, Error> {
        let chunk_dims = fd.chunk_options.resolve_chunk_dims(&fd.ds.dimensions);
        let ctx = ChunkContext::from_datatype(&chunk_dims, &fd.dt)?;
        // The overhang of a partial edge chunk holds the staged fill value
        // (issue #296).
        let elem = ctx.element_size.narrow::<NonZeroUsize>()?;
        let fill_time = fd.fill_time.unwrap_or_default();
        let fill = FillPattern::new(fill_time, fd.fill.as_deref(), elem);
        let set = chunked_write::compress_chunks(
            &fd.raw,
            &fd.ds.dimensions,
            ctx,
            &fd.chunk_options,
            fd.maxshape.as_deref(),
            fill,
            // `flatten_dataset` rejects an unallocated dataset outright, so
            // every dataset this engine builds allocates its storage.
            StorageAllocation::Allocated,
        )?;
        // Chunk data and the index beside it are raw (see `chunked_storage_spans`,
        // which reclaims both as raw).
        let (_addr, (layout_message, pipeline_message)) = self.place_relocatable(
            chunked_write::chunked_data_len(&set)?,
            AllocationClass::Raw,
            |stored_base| {
                let result = chunked_write::assemble_chunked_at(&set, stored_base)?;
                Ok((
                    result.data_bytes,
                    (result.layout_message, result.pipeline_message),
                ))
            },
        )?;
        // As in the contiguous branch: a set too large for the header goes to a
        // fractal heap, which the header then names in place of carrying it.
        let attr_info = self.place_dense_attrs_if_needed(fd)?;
        Ok(file_writer::build_chunked_dataset_oh(
            &fd.dt,
            &DatatypeLocation::Inline,
            &fd.ds,
            &layout_message,
            pipeline_message.as_deref(),
            &fd.attrs,
            attr_info.as_deref(),
            fill_time,
            fd.fill.as_deref(),
        )?)
    }
}
