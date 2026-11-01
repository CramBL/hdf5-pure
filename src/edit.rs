//! In-place editing of an existing HDF5 file (issue #32, Group C).
//!
//! The in-place edit engine opens an existing file and adds objects, overwrites dataset
//! values, or edits compact group attributes **in place**:
//! new data and object headers are allocated according to the file-space strategy, and
//! the object headers of the touched groups and their ancestors are rewritten so the
//! superblock points at the new root header. Nothing already in the file is moved, so the cost is
//! proportional to what you add, not to the file size — unlike the
//! read-everything-then-rebuild path through [`FileBuilder`](crate::FileBuilder).
//!
//! Both new datasets, new (sub)groups, and group attribute edits are supported,
//! at any existing group path. Adding into a nested group `/a/b` rewrites `b`'s
//! header (with the new link), then `a`'s header (repointing its link to `b`'s
//! new location), then the root's — "relocation up the tree". This is always
//! safe for *additions* because no surviving object is relocated except the
//! groups on the path being edited, and those are reachable only through links
//! this same commit rewrites (the root through the superblock); absolute
//! object-reference addresses to other objects stay valid.
//!
//! Deletion ([`Group::delete`](crate::Group::delete), the HDF5 `H5Ldelete`) is the mirror image:
//! the parent group's header is rebuilt without the removed link, relocated up
//! the tree the same way, and the unlinked object (and its subtree) is freed —
//! its blocks are returned to a session-local free list (see below). A deletion
//! and an addition at the *same* path in one commit is a **replacement** (issue
//! #305): the link is removed from the rebuilt parent before the new object's is
//! appended, and the one superblock write publishes both, so a rotating store
//! expresses a rotation as one commit and the path is never momentarily absent.
//! The original storage remains live until the superblock repoint. Under a strategy with
//! free-space managers, a later commit can reuse or truncate the released storage.
//! Object copy ([`File::copy`](crate::File::copy), the HDF5 `H5Ocopy`) deep-copies
//! a source subtree. The engine allocates fresh copies of every object, repoints internal
//! links and contiguous data addresses, and links the copy like an addition. The headers
//! are reproduced from their verbatim message bytes, so
//! datatypes, dataspaces, and attributes stay byte-exact. A chunked (and filtered)
//! dataset is copied with its chunk payloads and filter pipeline preserved
//! byte-for-byte, its index rebuilt at the new location. The same machinery,
//! [`File::copy_from`](crate::File::copy_from), copies an object **across two open files** — the
//! source being a separate [`File`](crate::File) reader rather than the file being
//! edited. Because the copy is byte-for-byte, the cross-file path refuses anything
//! that embeds a source-file absolute address (variable-length or reference data,
//! a committed datatype), which an in-file copy keeps valid by sharing the source
//! file's heaps and objects.
//!
//! Value overwrite ([`Dataset::write`](crate::Dataset::write), the HDF5 `H5Dwrite`) replaces
//! an **existing** dataset's values. The replacement's datatype and shape must
//! match the on-disk dataset (an overwrite, not a reshape or retype); contiguous,
//! compact, and chunked (including filtered) datasets are all supported, the chunk
//! geometry and filter pipeline taken from the on-disk header. A same-length
//! contiguous overwrite is the cheapest edit there is — the new bytes go straight
//! into the existing data block, so no header is rewritten and the superblock root
//! is not flipped, and the synced data write is the commit's linearization point.
//! A chunked overwrite takes the same in-place path when every (re-encoded) chunk
//! still fits its slot — always for unfiltered storage (chunk sizes are fixed by
//! the unchanged shape), and for filtered storage when the re-encoded chunks match.
//! When a length differs (a resized contiguous block, a filtered chunk that no
//! longer fits, or a compact dataset) the dataset's storage is rebuilt and its
//! header relocated like an addition: the new data and a rewritten header are
//! allocated, the data-layout message is repointed, the old storage is freed, and
//! the parent group's link is patched. A relocating overwrite of a dataset
//! reachable through more than one hard link is refused, since only the one named
//! link could be repointed at the moved header.
//!
//! # Scope
//!
//! Non-persistent PAGE sessions require mirrored staged editing. They append in fresh typed pages
//! and relocate value replacements. Staged commits require supported attribute storage and a
//! complete tree proved free of object references. These sessions retain all old bytes except
//! for superblock publication, leave freed extents unavailable for reuse or truncation, and
//! preserve the File Space Info message. New content is written before publication through the
//! superblock. Immediate append, SWMR writing, and page buffering are unsupported. The in-place
//! writes and space reuse described here apply to other sessions.
//!
//! It is deliberately strict: rather than silently produce a degraded file, it
//! refuses with [`Error::EditUnsupported`] any case it cannot reproduce
//! faithfully. Requirements:
//!
//! - The file uses 8-byte offsets/lengths. A **userblock** (non-zero base address, as every MATLAB
//!   v7.3 `.mat` file has) is supported: addresses are read and written relative to the base and
//!   the userblock bytes are preserved verbatim. The engine applies every staged edit to a file
//!   with a userblock, paged or not: value overwrites, additions of contiguous and
//!   chunked/filtered datasets, in-place and relocating overwrites of every layout (with the old
//!   storage reclaimed), object deletion (with base-aware subtree reclaim), in-file copy,
//!   cross-file copy into a userblock destination, group creation, compact attributes, and
//!   free-space reuse. The engine counts a paged file's pages from the base address, so page 0
//!   holds the superblock. An open under [`MemoryStrategy::Bounded`] rejects a file with a
//!   userblock, and a cross-file copy requires a source without a userblock (see
//!   [`copy_from`](crate::File::copy_from)).
//!   Any superblock version (0–3) is accepted: a version 0/1
//!   (symbol-table) file is edited by converting each group on the edited path
//!   to the latest format and repointing the superblock's root symbol-table
//!   entry.
//! - A group on an edited path may store links compactly or densely in an unfiltered
//!   managed fractal heap. Dense links are limited to 4096-byte encoded messages and seven-byte
//!   heap IDs. Link names must be valid UTF-8. The editor selects compact or dense storage from the
//!   final link count and the group's phase-change thresholds. It preserves link targets and name
//!   encodings across transitions. Continuation chunks are collapsed when a header is rewritten.
//!   A version 1 group is converted to a version 2 header with its links and attributes. Other
//!   group messages are omitted, and attributes the conversion cannot reproduce are rejected.
//! - An object that tracks **attribute creation order** is edited like any other: its 6-byte
//!   message records are walked and re-emitted, a new attribute takes the
//!   object's next creation index, an overwrite keeps the one it had, and a
//!   deletion leaves a gap rather than renumbering. A group that tracks **link**
//!   creation order works the same way: a link added to it takes the next index
//!   from the running maximum its Link Info message records, and a deletion
//!   leaves a gap there too. Storage transitions preserve the tracking and indexing policy,
//!   per-link indexes, and counter. Dense groups have a creation-order index when their policy
//!   enables indexing.
//! - Added datasets may be contiguous *or* chunked, with any filter the
//!   whole-file writer supports (deflate, shuffle, fletcher32, scale-offset,
//!   LZF, ZFP), and may declare extensible (maximum, optionally unlimited)
//!   dimensions. A chunked dataset's data and index — and any filtered chunks —
//!   are produced by the same builder the whole-file writer uses and placed according to
//!   the file-space strategy, so its object header is byte-identical to a freshly written
//!   one. A dataset may be empty (zero-element) under either storage, which is
//!   how an extensible dataset is created before the first
//!   [`append_staged`](crate::Dataset::append_staged) fills it: a contiguous one
//!   gets the undefined data address, a chunked one an index over zero chunks.
//!   A provenance dataset (`with_provenance`) is
//!   supported, its attributes computed the same way the whole-file writer
//!   computes them. A contiguous dataset may carry a variable-length-string
//!   payload (`with_vlen_strings`) or per-element object-reference targets
//!   (`with_path_references`); chunking either is not supported. A
//!   path-resolved reference may target any object this commit is not itself
//!   still writing (an ancestor group, a same-depth sibling group ordered
//!   later in the same commit, a copy destination or its interior, a
//!   `write_dataset` target, or an object this commit deletes) — targeting
//!   one of those is refused, up front and before any byte of the commit is
//!   written, rather than resolved to a stale or wrong address; a path that
//!   resolves nowhere at all becomes an undefined reference, matching the
//!   whole-file writer. Every
//!   added dataset must have a fixed-size datatype. Group, root, and **dataset**
//!   attribute edits (`set_group_attr` / `set_dataset_attr`) may be fixed-size or
//!   variable-length. An attribute set larger than an object header holds
//!   compactly, either more than [`attributes::MAX_COMPACT_ATTRS`] of them or one whose
//!   message overflows the header's 2-byte size field, is stored in a fractal heap, on
//!   an object this engine adds and on one whose edit takes it past the
//!   threshold alike, and an object already storing its attributes densely is
//!   rebuilt the same way ([`attributes::plan_attr_ops`], issue #102). A dataset attribute
//!   edit relocates the dataset header and so requires a single hard link.
//! - A new group's parent must already exist or be created in the same session
//!   (each level created explicitly); intermediate groups are not auto-created.
//! - Rows can be appended to an existing chunked, unlimited, Extensible-Array
//!   dataset **immediately and in place** with an in-place append (amortized O(1),
//!   crash-atomic, no `commit`), interleaved with the staged edits above. A
//!   target must satisfy the requirements of [`Dataset::append`](crate::Dataset::append).
//!   The first nonempty write initializes an eligible empty dataset's unallocated index.
//!   Unsupported targets return [`Error::AppendInPlaceUnsupported`].
//!
//! # Free-space reuse (issue #21)
//!
//! The reuse described below applies to persistent PAGE and other reusable-space sessions.
//!
//! Each commit vacates space: the object headers it rewrites are superseded, and
//! a deletion abandons its target's blocks. Under a strategy with free-space
//! managers, `H5F_FSPACE_STRATEGY_FSM_AGGR` or `H5F_FSPACE_STRATEGY_PAGE`, the
//! session records those regions in a free list and writes a later object into a
//! freed region that fits it, appending only where none does. It records a region
//! shorter than the file's threshold only where the region adjoins recorded free
//! space ([`FreeList::release_all`]), and drops it otherwise. A paged session
//! merges a region shorter than a page only with free space in the same page, and
//! a region of a page or more only with whole free pages or with the space it
//! cannot place (unclassified PAGE space). It records the free tail of a page it pads
//! whatever the threshold. When freed space and dropped regions form
//! a run reaching end-of-file, the commit truncates the file to where the run
//! starts, rounded up to a page boundary in a paged file. A paged session counts
//! a dropped region in that run only where the region spans a page or more. A
//! persisting commit retains free space for future managers
//! ([`FreeList::release_trailing_run`]). A commit that appends its managers retains the whole run.
//! Under `H5F_FSPACE_STRATEGY_AGGR` and `H5F_FSPACE_STRATEGY_NONE`, which have no
//! free-space managers, the session discards the regions it vacates.
//! Opening a file with malformed File Space Info returns the parse error.
//! Under `H5F_FSPACE_STRATEGY_AGGR`, staged edits
//! and immediate appends allocate from separate metadata and raw aggregates with
//! 2048-byte refill blocks. A clean close or drop releases unused
//! aggregate space at EOA after publishing the shorter EOA with the same root. An uncertain
//! publication prevents this cleanup. An ordinary read-write session under
//! `H5F_FSPACE_STRATEGY_FSM_AGGR`, including a file without a recorded strategy, tries reusable
//! free space before these aggregates for staged edits and nonpersistent appends. Persistent
//! appends release aggregate spare through manager publication and allocate exact extents.
//! During persistent staged edits, the editor excludes active aggregate spare from the managers
//! and releases surrendered spare through a manager rewrite. SWMR sessions under
//! `FSM_AGGR` and PAGE sessions use no classic aggregates.
//! The reuse is crash-safe: it only ever overwrites space
//! freed by an *earlier*, already-durable commit (never space the current commit
//! is mid-way through freeing), and truncation happens only after the superblock
//! recording the smaller end-of-file is itself durable. A commit that fails
//! before its repoint gives back what it drew, so a failed attempt costs the
//! session nothing.
//!
//! Immediate [`Dataset::append`](crate::Dataset::append) allocates provisional chunk and index
//! storage through [`crate::chunk_index_inplace::Store::allocate`]. In a persistent session, the
//! editor publishes replacement free-space managers before linking references to those allocations.
//! The editor excludes complete owned extents and the replacement managers' storage from recorded
//! free sections, and retains unused free space under the configured threshold policy. The object
//! root and unrelated staged edits are preserved. An interruption before publication leaves the
//! preceding managers authoritative. An interruption after publication can leave actual allocations
//! unreferenced. The editor cannot recover space stranded by an earlier writer without ownership
//! records.
//!
//! During persistent appends, the editor releases unused classic aggregate spare through manager
//! publication and allocates exact extents. Without persistence, the editor can reuse session free
//! space and classic aggregates. The SWMR writer allocates append-only because its readers may be
//! using earlier storage. Manager loading and serialization depend on the number of free sections.
//!
//! Staged commits place objects through [`WriteEngine::reserve`] and [`WriteEngine::place`],
//! including chunked dataset data and dense attribute heaps. The writer sizes relocatable
//! objects before placement and serializes their addresses after reservation.
//! A paged file (`H5F_FSPACE_STRATEGY_PAGE`) draws only from free
//! space of the page type it is placing, so reuse cannot make metadata and raw
//! data share a page — except from pages that are *wholly* free, which hold
//! nothing of either type to be mixed with and so may be opened for either page
//! type ([`SessionSpace::allocate`]). The free-space rewrite a commit
//! performs is placed through the same allocator where anything fits, rather than
//! at end-of-file (which on a paged file costs a page of its own), and that is
//! what keeps a delete-and-recreate workload from growing by a tail per commit
//! (issue #286 for the paged tail, issue #358 for the flat one).
//!
//! Reclaim is best-effort and conservative. Contiguous and chunked datasets
//! (chunk index plus chunk data) and whole group subtrees are reclaimed. Dense attribute and dense
//! group version 2 B-tree indexes are reclaimed when their owning storage becomes dead. The editor
//! also reclaims their fractal heaps where complete ownership is proven, including managed-space
//! manager headers, complete allocated section lists, and filtered huge-object allocations.
//! Logical sections within the heap are excluded from file-space reclamation. The editor reclaims
//! a global heap collection that only
//! the objects a delete removes refer to, and keeps every collection when it cannot read each
//! object that can refer to one. Storage whose allocations cannot be enumerated exhaustively, such
//! as a managed heap block with an active filter this crate does not decode, is
//! left as dead bytes. This avoids freeing a region that may still be in use. Under-reclaiming only
//! wastes space. Over-reclaiming could corrupt the file.
//!
//! On a paged file, the editor reuses sections from small metadata and raw managers according to
//! their page type. From a generic-large section, it reuses complete interior pages for either
//! page type and retains partial edges as unclassified. Other manager sections remain
//! unclassified and are excluded from allocation. The editor retains these fragments in the
//! persisted managers, preserving page separation where the type of a partial page is unknown
//! ([`WriteEngine::index_is_provably_raw`]).
//!
//! Space this editor vacates and cannot place is *dead* rather than lost
//! (dead PAGE space): out of use, not yet reusable. A page whose every byte is
//! free or dead holds nothing of either type, so it is promoted whole into the
//! free lists (whole-page promotion) - the same rule that lets
//! one page type claim a wholly free page from the other. That is what bounds a
//! paged file under delete-and-recreate churn: reclaim lands at page granularity
//! where the page type cannot be shown, rather than not at all (issue #388).
//!
//! Whether the free list outlives the session depends on how the file was
//! created. For the default (non-persisting) file it is **not** persisted: it is
//! forgotten on close, so reuse and shrinkage apply to churn within a session,
//! and a single delete-then-close shrinks the file only when the freed bytes
//! reach end-of-file. A file created with
//! `H5Pset_file_space_strategy(persist = true)` instead **persists** its free
//! space: `open` seeds the list from the on-disk free-space managers (the
//! `FSHD`/`FSSE` blocks the superblock-extension File Space Info message points
//! at), and each commit rewrites those managers, so freed regions survive
//! close/reopen and are reused across sessions — by this crate and the reference
//! C library alike. Such a commit *records* the freed space it keeps rather than
//! forgetting it; the blocks holding the managers go into space an earlier commit
//! freed, or past all live data when none fits, and the superblock is repointed
//! last, so a crash before the repoint leaves the prior file wholly intact.
//!
//! Freed space that reaches the end of the file is given back there too
//! (issue #418): the commit lowers the end-of-allocation to where the run starts,
//! drops those sections from the managers it writes, and truncates the file once
//! the smaller superblock is durable — page-aligned on a paged file, so the end
//! of allocation stays a whole number of pages. The writer retains a reserve for future
//! manager tails and trims only when it can release at least as much space as it retains
//! ([`FreeList::release_trailing_run`]). A commit whose own tail had to be appended
//! *above* the run it freed takes a second, tail-only rewrite to move the blocks
//! down into it ([`WriteEngine::shrink_to_the_trailing_free_run`]) — the delete
//! case, where the freed regions only become writable once the repoint that
//! freed them is durable. Whole-file compaction that reclaims every hole at once
//! is still the separate repack path.

#![allow(clippy::self_named_module_files)]

use core::num::NonZeroU64;
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::FreeSpaceManagerHeader;
use hdf5_pure_format::__private::ObjectHeaderChunk;
use hdf5_pure_space::__private as free_space;
use hdf5_pure_space::__private::AllocationClass;
#[cfg(doc)]
use hdf5_pure_space::__private::FreeList;
use hdf5_pure_space::__private::ReservationRequest;
use hdf5_pure_space::__private::{
    Extent as FreeExtent, FreeClass, ManagerKind, PageTransition, PageType, SessionSpace,
};

use crate::SuperblockStatusFlags;
use crate::SuperblockVersion;
use crate::UserblockSize;
use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::address::BaseAddressExt;
use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::convert::Narrow;
use crate::datatype::{self, Datatype};
use crate::error::{Error, FormatError};
use crate::extension::ExtensionSummary;
use crate::extension::SharedTableSummary;
use crate::file_create_properties::FileCreateProperties;
use crate::file_lock::{self, FileLocking};
use crate::file_space_info::DEFAULT_THRESHOLD;
use crate::file_space_info::FileSpaceInfo;
use crate::file_space_info::FileSpacePageSize;
use crate::file_space_info::FileSpaceStrategy;
use crate::filter_pipeline::FilterPipeline;
use crate::free_sections::FreeSection;
use crate::free_sections::FreeSections;
use crate::free_sections::PersistedManagerSet;
use crate::free_sections::ReusableSpaceStats;
use crate::free_space_manager;
use crate::free_space_manager::PersistedSections;
use crate::image::{FileImage, HandleImage, MirrorImage, WriteBuffering};
use crate::libver::LibVer;
use crate::object_path::ObjectPathBuf;
#[cfg(test)]
use crate::page_publication_failure::{PagePublicationFailure, PublicationFailureImage};
use crate::reader::{FileAccessProperties, OpenedSource, PersistedFreeSpacePolicy};
use crate::source::MetadataCacheConfig;
use crate::source::PhysicalSource;
use crate::source::Source;
use crate::superblock;
use crate::superblock::Superblock;
#[cfg(doc)]
use crate::type_builders::ObjectRefTarget;
use crate::width::FormatWidths;

use self::aggregation::AggregateSpace;
use self::commit::PublicationState;
use self::commit::plan::InvalidatedAddresses;
#[cfg(test)]
use self::commit::plan::Node;
use self::context::FileContext;
use self::dataset::FlatDataset;
#[cfg(doc)]
use self::dataset::{MovingWrite, WritePlan};
use self::free_snapshot::FreeSpaceSnapshot;
#[cfg(test)]
use self::group::LinkCreationOrder;
pub(crate) use self::object_header::OhRegion;
use self::staging::StagedEdits;

pub(crate) use self::staging::{StagedChild, StagedKind, StagedMeta, StagedObject};

use self::block::{BlockRequest, EncodingDependency, PlannedBlock};

mod aggregation;
pub(crate) mod allocation;
mod analysis;
mod append;
pub(crate) mod attributes;
mod block;
mod commit;
mod context;
mod copy;
mod dataset;
mod free_snapshot;
mod global_heap_liveness;
mod group;
#[cfg(test)]
test_util_hdf5::hdf5_1_10_tests! {
    mod group_inspection;
    #[cfg(test)]
    mod native_crosschecks;
}
mod group_links;
pub(crate) mod object_header;
#[cfg(test)]
mod observations;
mod paged_persistence;
mod reclaim;
mod reservation;
pub(crate) mod retirement;
mod root_reclamation;
mod staging;
mod tail_recycling;
#[cfg(test)]
mod test_util;

pub use self::append::{AppendBuilder, AppendOptions, LossyTailReencoding};
pub(crate) use self::append::{AppendGeometry, AppendTarget, LocatedState};
#[cfg(test)]
pub(crate) use self::append::{datatype_is_raw_appendable, locate_dataset_state};
pub(crate) use global_heap_liveness::GlobalHeapLiveness;

/// An undefined on-disk address (all bits set), HDF5's "no address" sentinel.
const UNDEF: u64 = u64::MAX;

/// The refusal both address-side reference screens report: an object reference
/// this commit writes names an object the same commit removes.
///
/// One constant because the screens differ only in where the address came from
/// — [`WriteEngine::resolve_reference_target`] takes it from a builder's
/// [`ObjectRefTarget::Raw`], [`screen_resolved_references`] reads it out of a
/// dataset's or an attribute's element bytes — and a caller must not be able to
/// tell them apart by the message. It is the address-side twin of the by-name refusal
/// `resolve_reference_target` reports for a path, and shares its second clause.
const REFERENCE_INTO_RECLAIMED_SPACE: &str = "a reference this commit writes holds the address of an object this commit deletes, or of \
     one under it; the reference would be left pointing at storage the delete can reclaim";

/// The refusal for the other half of [`InvalidatedAddresses`]: the object is
/// still there, but not at that address any more.
///
/// It can offer a way out, which the removal refusal cannot: the target still
/// exists, so a path names it. Whether the path *works* depends on which half of
/// [`InvalidatedAddresses::moved`] the target came from, which is why the message
/// names separate commits as well rather than promising the first.
/// [`WriteEngine::resolve_reference_target`] resolves a **dirty group** once this
/// commit has placed it, and reports "still writing" until then; a **relocating
/// dataset write** it refuses outright, by exact match against `write_targets`,
/// with no later point at which that changes.
const REFERENCE_TO_A_MOVED_OBJECT: &str = "a reference this commit writes holds the pre-commit address of an object this commit \
     rewrites elsewhere; name the target by path (`with_path_references`) so it resolves to \
     where the object lands, or use separate commits";

/// A shared-message (SOHM) index record does not always name a heap entry: where
/// libhdf5 shares a message written into an object header it
/// already holds open, the record names *that header* and the message stays
/// where it is. Such a record is a stored address like any other, and it is the
/// one no rewrite reaches — the index lives outside every object a commit
/// rebuilds, and this engine writes no shared-message indexes. A commit that
/// removed or moved the named header would leave the file's own index pointing
/// at bytes that are no longer there, which the next library to share a message
/// would follow.
const SHARED_MESSAGE_INDEX_NAMES_A_MOVED_OBJECT: &str = "this file's shared-message (SOHM) index names an object header this commit \
     removes or rewrites elsewhere, and the index cannot be updated in place; \
     rewrite the file with `repack` instead";

/// Upper bound on the number of object headers walked when counting hard links
/// across the file (issue #77 / reclaim safety). Far beyond any real file; a
/// graph larger than this aborts the count, and the commit then leaves deleted
/// objects unreclaimed (a safe leak) rather than risk an unbounded walk.
const MAX_LINK_GRAPH_NODES: u32 = 1 << 24;

/// A live [`BufferedAppender`](crate::BufferedAppender)'s hold on a dataset.
///
/// The appender accepts elements into memory and is the only thing that can
/// write them, so any staged edit that would make its flush refuse turns
/// accepted data into lost data at drop time. The claim lets the engine refuse
/// that edit up front instead.
struct AppenderClaim {
    /// Identifies this claim for release; ids are never reused within a session.
    token: u64,
    /// The appender's dataset path, or `None` for a handle reached by object
    /// reference. Such a handle is named by object-header address, which *any*
    /// staged edit may move, so a path-less claim conflicts with everything —
    /// exactly the rule `append_prepare` already applies to that target.
    path: Option<ObjectPathBuf>,
}

impl AppenderClaim {
    /// Returns `true` if this claim covers the dataset `path` identifies, or if either side has
    /// no path.
    fn conflicts_with(&self, path: Option<&ObjectPathBuf>) -> bool {
        match (&self.path, path) {
            (Some(claimed), Some(candidate)) => claimed == candidate,
            _ => true,
        }
    }
}

/// The in-place write engine behind the owned read-write [`File`](crate::File)
/// (its `Backend::Edit`).
///
/// Reads and edits the file through a [`FileImage`], which owns the writable
/// handle and decides how much of the file is resident. It carries two commit
/// models: staged tree edits applied by [`commit`](Self::commit), and immediate
/// crash-atomic in-place appends ([`append_inplace_gathered`](Self::append_inplace_gathered)).
pub(crate) struct WriteEngine {
    /// The file bytes this session reads and edits, behind the [`FileImage`]
    /// abstraction: reads go through its [`Source`] impl, and the write side —
    /// the end-of-file cursor, `append`, `write_at`, `truncate`, and the
    /// durability barriers — through its own primitives.
    ///
    /// Nothing in the engine assumes the whole file is resident, so one engine
    /// serves both a whole-file mirror and a file-backed image that holds only
    /// what it is reading (issue #198). [`image_slice`](Self::image_slice)
    /// exposes the mirror's buffer where a caller can exploit it.
    image: Box<dyn FileImage>,
    context: FileContext,
    /// Every edit this session has staged and not yet applied. Held as one
    /// value so that [`commit`](WriteEngine::commit) can take the whole set out
    /// for the duration of an attempt and put it back when that attempt
    /// refuses; see [`StagedEdits`].
    staged: StagedEdits,
    /// Datasets with a live [`BufferedAppender`](crate::BufferedAppender), which
    /// holds accepted elements only it can write. A staged edit that would stop
    /// that appender from flushing is refused while the claim stands, rather
    /// than left to fail in the appender's `Drop`, where there is no caller to
    /// report it to and the buffer is simply lost. See `refuse_if_claimed`.
    appender_claims: Vec<AppenderClaim>,
    /// Monotonic id for the next claim, so releasing one is exact even when two
    /// appenders on different datasets are live at once.
    next_appender_token: u64,
    /// The reusable-space strategy established from the file's File Space Info at open.
    ///
    /// It owns the flat or PAGE free-space state and its threshold. Strategies without reusable
    /// free-space managers, and an unreadable strategy, use [`SessionSpace::Disabled`].
    ///
    /// Its extents are stored addresses, relative to the base address, as a free-space manager
    /// records them.
    space: SessionSpace,
    /// Classic aggregate windows for staged edits and nonpersistent immediate appends under
    /// [`FileSpaceStrategy::Aggr`] or [`FileSpaceStrategy::FsmAggr`].
    ///
    /// An ordinary read-write session without a recorded strategy uses FSM_AGGR with persistence
    /// disabled.
    aggregation: Option<AggregateSpace>,
    /// Allocator rollback and persisted-manager authority for immediate allocation batches.
    append_allocator: allocation::AppendAllocator,
    /// Whether this file has been *proved* to hold no object reference at all —
    /// the one thing that lets a commit skip the walk that repoints them
    /// (issue #324).
    ///
    /// Set by a walk that reaches every object and finds no reference-holding
    /// datatype; every later commit in this session then skips the walk
    /// outright, which matters because the walk is linear in the file's object
    /// count and the file that needs it is the exception.
    ///
    /// One bit rather than three states, because "not yet walked" and "walked
    /// and found one" are the same instruction to the caller: walk. Only the
    /// proof is worth carrying.
    ///
    /// Cleared by any commit that could introduce a reference the walk has not
    /// seen. Exactly one of [`StagedEdits`]' nine collections can: `datasets`,
    /// which carries a datatype and attributes the caller chose. The check
    /// covers four, and the other three are belt-and-braces rather than
    /// load-bearing — recorded here because a reason that does not hold is worse
    /// than no reason:
    ///
    /// - `writes` is a value overwrite, whose datatype must be compatible with
    ///   the on-disk datatype. A reference-typed one therefore overwrites a dataset
    ///   that already held references, which a file proved free of them does not
    ///   have. Checked anyway, because that match is a commit-time check running
    ///   after this one, so `fd.dt` here is still whatever the caller supplied.
    /// - `copies` re-emits an object *from this same file* — one the walk that
    ///   granted the proof had already visited and found reference-free.
    /// - `cross_copies` reads from another file, where `reject_foreign_addresses`
    ///   refuses a reference datatype outright.
    ///
    /// The remaining five cannot, and these reasons do hold:
    ///
    /// - `groups` stages a bare path: no datatype, no data, no attributes.
    /// - Group and dataset attribute edits contain a [`crate::type_builders::AttrValue`],
    ///   which has no reference variant.
    /// - `appends` grows an *existing* dataset along its own datatype, so it can
    ///   only add references to a dataset that already held them.
    /// - `deletes` removes objects and adds nothing.
    ///
    /// The commit fast path returns before the check and needs no reason of its
    /// own: it is taken only when every staged edit is a same-length in-place
    /// overwrite, which is the `writes` case above.
    ///
    /// A tenth collection would have to be classified the same way; the check
    /// sits beside the reference screen in `commit`, which iterates the same
    /// datatype-carrying collections.
    proved_free_of_references: bool,
    /// Free-space persistence read from the file's superblock extension on
    /// `open` (the file-creation `H5Pset_file_space_strategy(persist = true)`
    /// setting). `None` for the default non-persisting file; when `Some`, every
    /// [`commit`](Self::commit) rewrites the on-disk free-space managers so the
    /// free list survives close/reopen.
    persist: Option<PersistState>,
    /// The first stored page boundary available to a non-persistent PAGE session.
    ///
    /// Set at open at or beyond both the recorded EOA and the physical extent. Such a session
    /// appends every allocation, retains freed extents, and publishes only through the superblock.
    fresh_page_boundary: Option<StoredAddress>,
    /// Per-dataset geometry cache for the immediate O(1) in-place append
    /// ([`append_inplace_gathered`](Self::append_inplace_gathered)), keyed by the dataset's resolved
    /// **object-header address** (not its path, so two hard links to one dataset
    /// share one entry). Populated on the first append to a dataset and maintained
    /// across appends; cleared wholesale at the entry of every non-trivial
    /// [`commit`](Self::commit), since a commit can relocate a cached header or
    /// free the region it points into (see `commit`).
    located: HashMap<u64, LocatedState>,
    /// Physical ranges of global heap collections placed by this session for variable-length
    /// overwrites, keyed by dataset path.
    ///
    /// A later overwrite can reclaim these collections while their ownership remains exclusive.
    /// [`invalidate_heap_provenance`](Self::invalidate_heap_provenance) clears the recorded
    /// ownership before an operation could share them. Collections present at open remain
    /// allocated during overwrites.
    vl_overwrite_heaps: HashMap<ObjectPathBuf, Vec<FileRange>>,
    /// Heap collections superseded by a value overwrite this commit is applying,
    /// freed once the commit's superblock repoint has landed — never before, so
    /// a mid-commit crash leaves the prior root reaching bytes that are still
    /// there. Drained into the commit's `to_free` list; cleared at commit entry,
    /// so an attempt that fails partway frees nothing on the next one.
    superseded_heaps: Vec<FileRange>,
    /// The bytes each same-length in-place overwrite of the commit in flight
    /// wrote over, so a commit that fails before its linearization point can put
    /// the dataset's values back (issue #344).
    ///
    /// Everything else a commit places is written where nothing reaches it until
    /// the superblock is repointed, so an attempt that stops short of the
    /// repoint is invisible. A same-length value overwrite is the exception:
    /// [`WritePlan::InPlace`] writes straight over the dataset's existing data
    /// block, which the *current* root already reaches, so there is no repoint
    /// to withhold and the new values are live the moment they land. This
    /// journal is how they are withheld anyway.
    ///
    /// Written only by
    /// [`write_inplace_journaled`](Self::write_inplace_journaled) and drained
    /// only by [`undo_inplace_writes`](Self::undo_inplace_writes). Every
    /// [`commit`](Self::commit) that returns leaves it empty, and one that
    /// unwinds does not, which is why the next commit clears it on entry rather
    /// than trusting it.
    ///
    /// **It costs a full copy of every value being replaced, read back off the
    /// file first.** Measured on the bounded backing, whose contract is not to
    /// hold the file in memory: a same-length overwrite of a 4 MB dataset made
    /// its commit read 4,000,197 bytes where the same commit without the
    /// journal read 197, and held the 4 MB until the commit ended. That is the
    /// price of the guarantee and it is charged on the success path too, since
    /// whether the bytes will be needed is not known until the attempt ends.
    /// `a_bounded_commit_reads_the_value_it_is_replacing` states the rule so
    /// the number is in the repository rather than a surprise.
    ///
    /// The alternative — planning a same-length overwrite as a relocating write,
    /// as a variable-length one already is — would cost nothing here and was
    /// rejected for a harder reason than cost: a relocating write is refused for
    /// a dataset with more than one hard link
    /// ([`count_incoming_hard_links`](Self::count_incoming_hard_links)), so it
    /// would start refusing overwrites this engine accepts today.
    inplace_undo: Vec<(u64, Vec<u8>)>,
    /// True when this engine was opened for SWMR writing
    /// ([`open_swmr_writer`](Self::open_swmr_writer)): the append engine then
    /// enforces the SWMR subset (unfiltered, chunk-aligned) so a concurrent
    /// reader never observes a torn view. `false` for an ordinary edit session.
    swmr_mode: bool,
    /// Set by the first [`commit`](Self::commit) that does any work. A commit can
    /// relocate an object header, and nothing on disk distinguishes a relocated
    /// header from the intact bytes it vacated — the old header still parses, and
    /// its data-layout message still points at the live chunk index. An
    /// [`AppendTarget::Header`] captured before that commit would therefore append
    /// successfully *into the dead header*, growing its dataspace while the live
    /// dataset stayed put, and report `Ok`. A path is re-resolved on every append
    /// and so survives a commit; a raw address does not, so it is refused once one
    /// has run.
    committed: bool,
    /// Object-header address for each path an in-place append has resolved in
    /// this session. A single `Dataset::append` asks for the target's geometry
    /// and then appends to it, and a loop of appends repeats that; without this
    /// the path would be walked from the root on every one of those steps.
    ///
    /// An in-place append never moves an object header — that is why
    /// [`located`](Self::located) can be keyed by address and survive appends —
    /// so only a commit can stale an entry, and it clears both together.
    resolved: HashMap<ObjectPathBuf, u64>,
    /// Whether this session splits a large in-place append into batches, trading
    /// whole-call crash atomicity for a peak memory that does not scale with the
    /// call. Set by [`open_rw_with_strategy`](Self::open_rw_with_strategy); see
    /// [`batch_elems`](Self::batch_elems).
    batched_appends: bool,
    /// Whether this session reads through a handle rather than a whole-file
    /// mirror. Set by [`open_rw_with_strategy`](Self::open_rw_with_strategy), and reported by
    /// [`File::edit_backing`](crate::File::edit_backing) so a caller who
    /// asked for [`MemoryStrategy::Auto`] can tell which one it got. Distinct
    /// from [`batched_appends`](Self::batched_appends), which is a crash-atomicity
    /// trade the bounded engine happens to make, not a statement about memory.
    bounded: bool,
    /// The on-disk format this session may write, resolved from the fapl's
    /// [`FileAccessProperties::with_libver_bounds`]. `None` — the default —
    /// means unconstrained: the session adds whatever the content needs, which
    /// is what lets a file the C library wrote under its own bounds be edited at
    /// all.
    ///
    /// Set below [`LibVer::V110`] it refuses content the older format cannot
    /// carry, the way the whole-file writer does. The file's *own* superblock
    /// version cannot stand in for this: a version 2 superblock says the file is
    /// 1.8-readable today, not that the caller wants it to stay that way, and
    /// deriving the ceiling from it would refuse the C-library-file edits of
    /// issue #101.
    ///
    /// [`FileAccessProperties::with_libver_bounds`]: crate::FileAccessProperties::with_libver_bounds
    libver_ceiling: Option<LibVer>,
    /// The physical file length associated with the persisted free-space state.
    ///
    /// [`finalize_persist`](Self::finalize_persist) compares this length with the image length
    /// and checks aggregate state and [`fsm_dirty`](Self::fsm_dirty)
    /// before rewriting the managers.
    /// This field is used when [`persist`](Self::persist) is `Some`.
    fsm_len: u64,
    /// Whether confirmed immediate retirements require another manager publication.
    fsm_dirty: bool,
    /// The latest publication attempt's completion state and rollback eligibility.
    ///
    /// Non-persistent PAGE sessions retain an attempted state after an error. Before consuming
    /// another staged set, the editor completes a barrier and verifies the issued superblock.
    /// Explicit synchronization performs the same verification. Other publishers reset the
    /// state when beginning a new attempt.
    publication: PublicationState,
    /// Whether a [`stage_atomically`](Self::stage_atomically) batch is open, so
    /// [`refuse_mid_batch`](Self::refuse_mid_batch) can keep
    /// [`StagedEdits::rewind`]'s truncation an exact undo.
    staging_batch: bool,
    /// How many times this session's staged set has been *consumed* by a commit
    /// rather than handed back to it.
    ///
    /// It is what tells a handle born onto a staged creation
    /// ([`staged_generation`](Self::staged_generation)) which of two things has
    /// happened since: the object it names was published, so it addresses the
    /// file now — or the staging was withdrawn under it, and it names nothing.
    /// Both leave the staged set without an entry at its path, and reading them
    /// the same way is what let such a handle silently retarget onto whatever
    /// the file held at that path.
    ///
    /// Advanced by [`commit`](Self::commit) alone, on exactly the path where the
    /// set does not go back: a commit that succeeds, and one that fails past its
    /// publish (whose objects are in the file). A commit refused before its
    /// first write restores the set and leaves this alone, so the handles it
    /// concerns stay pending and the batch can be committed again.
    staged_generation: u64,
    /// Superblock status flags this session raised on the file and holds for its
    /// lifetime — the SWMR pair, or the page buffer's crash mark — or `0` when it
    /// holds none.
    ///
    /// A clean commit publishes *this* rather than a literal zero. The zero was
    /// there to scrub a flag the file arrived carrying (one a crashed writer left
    /// behind), and it still does: this is `0` unless the session raised
    /// something itself. What it must not do is scrub a mark that is still
    /// standing for a reason, which a page-buffered session's is for every
    /// commit it makes (issue #308).
    ///
    /// Maintained only by [`set_consistency_flags`](Self::set_consistency_flags),
    /// which is the one place this crate writes that byte, so the field cannot
    /// drift from what is on the disk.
    held_status_flags: SuperblockStatusFlags,
    /// Who owns this session's `fsync` cadence, from the fapl's
    /// [`FileAccessProperties::with_sync_policy`]. Consulted by
    /// [`barrier`](Self::barrier) and [`barrier_data`](Self::barrier_data) —
    /// every durability point the write paths define — and by nothing else, so
    /// [`force_sync`](Self::force_sync) can serve an explicit
    /// [`File::sync`](crate::File::sync) whatever it says.
    ///
    /// [`FileAccessProperties::with_sync_policy`]: crate::FileAccessProperties::with_sync_policy
    sync_policy: SyncPolicy,
}

/// How much memory a read-write open may use to hold the file being edited.
///
/// A bounded session reads through a handle and holds what a commit is building.
/// A mirrored session holds the whole file in memory. Editing a file with a pre-v2
/// superblock, a userblock, or non-persistent [`FileSpaceStrategy::Page`] allocation
/// requires the mirror.
///
/// This is what a caller says about that trade-off, on
/// [`FileAccessProperties::with_memory_strategy`](crate::FileAccessProperties::with_memory_strategy).
/// Leaving it unset lets the entry point decide: [`File::open_rw`](crate::File::open_rw)
/// prefers the bounded engine and falls back to the mirror ([`Auto`](Self::Auto)).
/// Selecting [`Bounded`](Self::Bounded) rejects a file that requires the mirror.
///
/// This is a *request*, so it is deliberately not the type a file answers with:
/// [`File::edit_backing`](crate::File::edit_backing) returns an [`EditBacking`],
/// which cannot express [`Auto`](Self::Auto).
///
/// Sealed: unlike [`FileLocking`] or [`FileSpaceStrategy`], whose variant sets
/// mirror a closed C-library enum, this is a policy this crate invented, so a
/// fourth strategy must not be a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MemoryStrategy {
    /// Never build a whole-file mirror. A file the bounded engine cannot edit is
    /// refused at open with [`Error::EditUnsupported`], before anything is
    /// staged.
    Bounded,
    /// Prefer the bounded engine, but fall back to the whole-file mirror for a
    /// file it cannot edit, rather than refusing. Memory then scales with the
    /// file, which is the cost of the file opening at all. What
    /// [`File::open_rw`](crate::File::open_rw) uses when nothing is asked for.
    ///
    /// Falls back for a pre-v2 superblock, a userblock, or [`FileSpaceStrategy::Page`]
    /// with persistence disabled. Every other file accepted for editing uses the bounded
    /// backing, regardless of size. Wherever [`Bounded`](Self::Bounded) opens a file,
    /// `Auto` selects the same backing.
    Auto,
    /// Always build the whole-file mirror, whatever the file looks like. What
    /// [`File::open_rw`](crate::File::open_rw) did before it learned to dispatch.
    Mirrored,
}

/// Which of the two read-write backends a file's editing session is actually
/// using, from [`File::edit_backing`](crate::File::edit_backing).
///
/// Deliberately a different type from [`MemoryStrategy`]: that one is what a
/// caller *asks* for and includes [`Auto`](MemoryStrategy::Auto), which is a
/// preference between these two rather than a third thing a session can be. A
/// single shared type would make `file.backing() == Auto` a comparison that
/// compiles and is false forever.
///
/// The two also evolve at different rates. A future `MemoryStrategy` may name a
/// new *policy* — a byte budget, a size threshold — without the set of backends
/// changing at all. Sealed for the rarer case that a third backend does appear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EditBacking {
    /// Reads through a file handle, holding only what a commit is building.
    /// Memory does not scale with the file.
    Bounded,
    /// Holds the whole file in memory for the life of the session.
    Mirrored,
}

impl From<EditBacking> for MemoryStrategy {
    /// Turns an outcome back into the request that pins it, so a caller can
    /// reopen a file onto the backing it got the first time:
    /// `with_memory_strategy(file.edit_backing().unwrap().into())`.
    fn from(backing: EditBacking) -> Self {
        match backing {
            EditBacking::Bounded => Self::Bounded,
            EditBacking::Mirrored => Self::Mirrored,
        }
    }
}

/// The durability policy for a read-write editing session.
///
/// With ordinary buffering, successful commits and appends have issued their writes to the
/// operating system under either policy. Explicit
/// [page buffering](crate::FileAccessProperties::with_page_buffer_size) can retain staged commit
/// writes. Immediate appends drain every buffering mode at their publication boundaries.
///
/// [`File::sync`](crate::File::sync), clean closure, and dropping the last handle request
/// synchronization under either policy. The whole-file writer and [`repack`](crate::repack())
/// do not use this editing-session policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SyncPolicy {
    /// Requests durability at each publication barrier and during session closure.
    ///
    /// Append content is synchronized before references. An EOF change is synchronized with
    /// file metadata before a reference to the allocation is published. Physical metadata writes
    /// can tear if the storage device does not write them atomically.
    #[default]
    Always,
    /// Defers durability synchronization until explicit synchronization or session closure.
    ///
    /// Append publication drains pending writes before new references, including with an
    /// explicit page buffer. Successful appends therefore preserve ordering against process
    /// interruption. Power loss can lose or reorder writes retained by the operating system.
    /// Explicit page buffering can retain staged commit writes after the commit returns.
    ///
    /// [`File::sync`](crate::File::sync) requests an intermediate durability checkpoint.
    /// [`File::close`](crate::File::close) and dropping the last handle synchronize after their
    /// final writes. Delayed storage errors may be reported during that synchronization.
    OnClose,
}

/// Returns the reason editing a file requires the whole-file mirror, if any.
fn bounded_only_limitation(session: &WriteEngine) -> Option<&'static str> {
    if session.fresh_page_boundary.is_some() {
        return Some(
            "bounded editing of PAGE files without persisted free space is unsupported; \
             use MemoryStrategy::Auto or MemoryStrategy::Mirrored for staged edits",
        );
    }

    if matches!(
        session.context.superblock().version(),
        SuperblockVersion::Zero | SuperblockVersion::One
    ) {
        return Some(
            "bounded read-write access requires a latest-format file (v2/v3 superblock); \
             leave MemoryStrategy unset, or pass MemoryStrategy::Auto, to fall back to \
             the whole-file mirror here",
        );
    }
    if !session.context.superblock().base_address.is_zero() {
        return Some(
            "bounded read-write access does not support a file with a userblock \
             (non-zero base address); leave MemoryStrategy unset, or pass \
             MemoryStrategy::Auto, to fall back to the whole-file mirror here",
        );
    }
    None
}

/// Whether a file built from `create` could not then be opened read-write under
/// `access`, and why.
///
/// [`File::create_with_options`](crate::File::create_with_options) writes a file
/// and hands back an open read-write handle, so a creation/access pair that
/// cannot survive that second half must be caught *before* the write — otherwise
/// the call leaves a file on disk and returns `Err`, which reads like a failed
/// create but is not one.
///
/// Checks the creation properties against [`bounded_only_limitation`] and the page-buffer
/// requirements in [`set_page_buffer_size`](WriteEngine::set_page_buffer_size).
pub(crate) fn create_would_refuse_reopen(
    create: &FileCreateProperties,
    access: &FileAccessProperties,
) -> Option<&'static str> {
    if create.file_space_settings().is_some_and(|settings| {
        settings.strategy() == FileSpaceStrategy::Page && !settings.persist()
    }) && access.memory_strategy() == Some(MemoryStrategy::Bounded)
    {
        return Some(
            "bounded editing of PAGE files without persisted free space is unsupported; \
             use MemoryStrategy::Auto or MemoryStrategy::Mirrored for staged edits",
        );
    }
    if create.userblock() != UserblockSize::NONE
        && access.memory_strategy() == Some(MemoryStrategy::Bounded)
    {
        return Some(
            "a userblock cannot be combined with MemoryStrategy::Bounded: the bounded engine \
             cannot edit a file with a non-zero base address, so creating one this way would \
             write the file and then refuse to open it; drop the userblock, or leave \
             MemoryStrategy unset to mirror this file",
        );
    }
    // The page-buffer refusals `set_page_buffer_size` makes at open, restated as
    // the creation pair that would walk into them. They are phrased as something
    // to *add* to the properties in hand rather than as something to recreate the
    // file with, since the file does not exist yet (issues #288 and #308).
    if access.page_buffer_size() != 0 {
        if access.sync_policy() == SyncPolicy::Always {
            return Some(
                "a page buffer does nothing under the default SyncPolicy::Always, whose every \
                 barrier is an fsync that flushes it, so creating one this way would write the \
                 file and then fail to open it; add \
                 with_sync_policy(SyncPolicy::OnClose) to the access properties, or drop the \
                 page buffer",
            );
        }
        // The version-3 counterpart. It used to need none: a page buffer required
        // a paged file, and `FileSpaceStrategy::Page` below the 1.10 format is
        // refused by the builder itself before anything is written, so that pair
        // never reached an open. Issue #357 lifted the paged requirement, and an
        // *unpaged* file at `LibVer::V18` is an ordinary, buildable file with a
        // version-2 superblock — one whose status-flags byte no library reads
        // back, so the mark could not be raised on it. Without this the pair
        // would write the file and fail the open it promised.
        //
        // Unsatisfiable bounds are left to the builder, which reports which
        // format it can write; this is not the place to restate that.
        if matches!(
            crate::libver::LibVer::resolve_writable(create.libver_bounds()),
            Ok(v) if v < crate::libver::LibVer::V110
        ) {
            return Some(
                "a page buffer marks the file in a version-3 superblock, and these bounds \
                 write the 1.8 format, so creating one this way would write the file and then \
                 fail to open it; raise with_libver_bounds to admit LibVer::V110, or drop the \
                 page buffer",
            );
        }
        // An unpaged file has no page size of its own and merges within
        // `DEFAULT_GATHER_PAGE`, which is the same figure the format defaults
        // `H5Pset_file_space_page_size` to. This arm decides on its own now that
        // a budget below the session's own gather budget is honored rather than
        // refused (issue #391): a 2 KiB buffer on an unpaged create clears every
        // other check and is caught only here. It is written out rather than
        // folded into the paged arm's `unwrap_or` because the two constants are
        // free to move apart, and because a reader comparing this function
        // against `set_page_buffer_size` should find the same rule in both.
        let page_size = match create.file_space_settings() {
            Some(settings) if settings.strategy() == FileSpaceStrategy::Page => {
                settings.page_size().get()
            }
            _ => DEFAULT_GATHER_PAGE,
        };
        if (access.page_buffer_size() as u64) < page_size {
            return Some(
                "a page buffer smaller than the file's file-space page size would be refused \
                 at open — so creating one this way would write the file and then fail to \
                 open it; raise with_page_buffer_size",
            );
        }
    }
    None
}

/// Selects storage for a persistent extension and its free-space managers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TailPlacement {
    /// Uses reusable free space where the tail fits, or appends it at EOA.
    Anywhere,
    /// Reserves the current tail for recycling while publishing a replacement elsewhere.
    Preserve,
    /// Uses the reserved extent if the tail fits, or allocates elsewhere and returns the
    /// extent through ordinary threshold admission.
    Recycle(FreeExtent),
    /// Uses existing free space, or leaves the file unchanged.
    ReuseOnly,
}

/// Holds a physical reservation for [`WriteEngine::place`].
///
/// The caller can build a relocatable allocation for its reserved address before writing it.
#[derive(Clone, Copy, Debug)]
enum Placement {
    /// A reservation from a classic aggregate, including its physical end of allocation.
    Aggregated { addr: u64, len: u64, end: u64 },
    /// The end-of-file, with the page for the requested type already open.
    Appended { addr: u64, len: u64 },
    /// A region an earlier commit freed, already removed from its free list.
    Reused { addr: u64, len: u64 },
}

impl Placement {
    /// The absolute file address the bytes will occupy.
    fn address(self) -> u64 {
        match self {
            Placement::Aggregated { addr, .. }
            | Placement::Reused { addr, .. }
            | Placement::Appended { addr, .. } => addr,
        }
    }

    /// The reserved byte count, which the placed bytes must match exactly.
    fn len(self) -> u64 {
        match self {
            Placement::Aggregated { len, .. }
            | Placement::Reused { len, .. }
            | Placement::Appended { len, .. } => len,
        }
    }
}

/// Orders pending writes and applies the data-sync policy.
///
/// [`SyncPolicy::Always`] calls [`FileImage::sync_data`]. [`SyncPolicy::OnClose`] issues an
/// ordering barrier without forcing data to durable storage.
///
/// # Errors
///
/// Returns [`Error::Io`] if the barrier fails.
fn barrier_data(image: &mut dyn FileImage, sync_policy: SyncPolicy) -> Result<(), Error> {
    match sync_policy {
        SyncPolicy::Always => image.sync_data(),
        SyncPolicy::OnClose => image.ordering_barrier(),
    }
}

/// The default byte budget for gathering writes within one operation.
///
/// An explicit [page buffer](crate::FileAccessProperties::with_page_buffer_size) replaces this
/// budget and can retain staged commit writes across operations. Append publication drains both
/// modes. A smaller budget bounds pending bytes while increasing writes within long batches.
const WRITE_GATHER_BYTES: usize = 1 << 20;

/// Page the write gatherer merges within on a file that is not paged.
///
/// A non-paged file has no page size of its own, and this is the same figure
/// HDF5 defaults `H5Pset_file_space_page_size` to, so the merge quantum matches
/// what the file *would* have used had it been paged.
const DEFAULT_GATHER_PAGE: u64 = FileSpacePageSize::DEFAULT.get();

/// Superblock consistency-flag bits raised while a SWMR writer is active: bit 0
/// (write access) | bit 2 (SWMR write access). Cleared on a clean close. Matches
/// libhdf5, h5py, and [`crate::File::open_swmr_writer`]. These
/// are the bits every open path checks — see [`file_lock::check_status_flags`].
const SWMR_WRITE_FLAGS: SuperblockStatusFlags = SuperblockStatusFlags::swmr_write_marked();

/// State for a file that persists its free space on disk. Carries the file's
/// fixed file-space parameters and the extents of the free-space-manager blocks
/// (and superblock extension) the *current* on-disk file uses, so the next
/// persisting commit can reclaim them when it writes fresh ones.
struct PersistState {
    strategy: FileSpaceStrategy,
    threshold: u64,
    page_size: FileSpacePageSize,
    /// Each block the next persisting commit frees: the superblock extension and free-space-manager
    /// blocks. A manager header or section list that does not parse is not among them.
    old_blocks: Vec<FreeExtent>,
}

/// Decoded opening settings and a separate extension-allocation result.
///
/// Allocation errors are deferred until strict manager recovery has completed. The caller uses
/// both results within the same unchanged source phase.
struct OpeningExtension {
    /// The physical extension address and first File Space Info message, if present.
    info: Option<(u64, FileSpaceInfo)>,
    /// Complete physical chunk ranges, or a deferred editor geometry error.
    ///
    /// Empty when persistence is absent or disabled.
    allocations: Result<Vec<FileRange>, Error>,
}

impl OpeningExtension {
    /// Prepares settings and allocation geometry during one extension traversal.
    ///
    /// Materializes allocation ranges only when the decoded settings enable persistence.
    ///
    /// # Errors
    ///
    /// Returns errors from [`WriteEngine::extension_summary`] and
    /// [`ExtensionSummary::into_file_space_info`]. Allocation errors remain in the result.
    fn read(source: &(impl Source + ?Sized), context: &FileContext) -> Result<Self, FormatError> {
        let mut allocations = self::object_header::ExtensionAllocations::new();
        let summary = WriteEngine::extension_summary(
            source,
            context,
            Some(&mut |chunk| {
                allocations.observe(chunk, context.addresses(), source.len());
            }),
        )?;
        let info = summary
            .map(ExtensionSummary::into_file_space_info)
            .transpose()?
            .flatten()
            .map(|(address, info)| (address.get(), info));
        let allocations = if info.as_ref().is_some_and(|(_, info)| info.persist) {
            allocations.finish()
        } else {
            Ok(Vec::new())
        };
        Ok(Self { info, allocations })
    }
}

/// Parsed persistent managers and extension allocation ranges for allocator installation.
struct PersistedRecovery {
    managers: PersistedManagerSet,
    extension: Vec<FileRange>,
}

impl PersistedRecovery {
    /// Reads the complete manager set and adopts prepared physical extension ranges.
    ///
    /// Reports manager failures before the deferred extension-allocation result. Uses
    /// [`BorrowedHandle::finish`](crate::image::BorrowedHandle::finish) to preserve captured I/O
    /// errors when discharging that result.
    ///
    /// # Errors
    ///
    /// Returns [`Error::PersistedFreeSpace`] for a manager read or parse failure. Propagates
    /// the extension error through the source adapter. Returns [`Error::Format`] for invalid
    /// address arithmetic or a source shorter than the logical end of file.
    fn read(
        source: &crate::image::BorrowedHandle<'_>,
        context: &FileContext,
        info: &FileSpaceInfo,
        extension: Result<Vec<FileRange>, Error>,
    ) -> Result<Self, Error> {
        let addresses = context.addresses();
        let stored_eof =
            addresses.relative(PhysicalAddress::new(context.superblock().eof_address))?;
        let managers = PersistedManagerSet::read(
            &source.manager_source(),
            context.widths(),
            addresses.effective_base(),
            stored_eof.get(),
            info,
        )?;
        let extension = source.finish(extension)?;
        if context.superblock().eof_address > source.len() {
            return Err(FormatError::TruncatedFile {
                expected_eof: context.superblock().eof_address,
                actual_len: source.len(),
            }
            .into());
        }
        Ok(Self {
            managers,
            extension,
        })
    }
}

/// The byte classification of file storage identified by the editor.
///
/// Extents are absolute `(offset, length)` pairs. The editor reads reachable storage with
/// the parsers used for reclamation and reports unsupported structures through
/// [`analysis_complete`](Self::analysis_complete). Unclassified extents can include live
/// storage when the analysis is incomplete.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SpaceOwnership {
    /// The physical file length in bytes.
    pub(crate) physical_size: u64,
    /// The end-of-file position stored in the superblock.
    pub(crate) logical_size: u64,
    /// The userblock, from byte 0 to the superblock, or empty for a file without one.
    pub(crate) userblock_space: Vec<(u64, u64)>,
    /// The dataset payload allocations stored outside object headers in this file, and the global
    /// heap collections that reachable objects refer to.
    pub(crate) live_raw_space: Vec<(u64, u64)>,
    /// The identified object headers, indexes, heaps, and file metadata.
    pub(crate) live_metadata_space: Vec<(u64, u64)>,
    /// The complete allocations of persisted free-space managers.
    pub(crate) free_space_manager_space: Vec<(u64, u64)>,
    /// The free extents available to the editor.
    pub(crate) reusable_free_space: Vec<(u64, u64)>,
    /// The bytes below the logical end of file outside the identified allocations and free space.
    pub(crate) unclassified_space: Vec<(u64, u64)>,
    /// Whether every reachable storage structure is supported and the traversal finished.
    ///
    /// Shared messages, unsupported headers or heaps, data the editor cannot read or that holds a
    /// file address it cannot locate, and traversal limits make the analysis incomplete, even when
    /// every byte has been classified.
    pub(crate) analysis_complete: bool,
    /// The bytes between the logical end of file and the physical file length.
    pub(crate) trailing_slack_space: Vec<(u64, u64)>,
}

impl WriteEngine {
    /// Open an existing HDF5 file for in-place editing under an explicit
    /// file-locking policy.
    ///
    /// Reads the file into memory and retains a read/write handle. Under
    /// [`FileLocking::Enabled`] it takes an exclusive OS advisory lock so the file
    /// cannot be opened concurrently by another writer or reader; the lock is
    /// released automatically when the session is dropped or the process exits
    /// (including on a crash). Fails with [`Error::FileLocked`] if the file is
    /// already locked, or [`Error::EditUnsupported`] if the file is not a
    /// supported target; its documentation enumerates the exact requirements.
    /// `HDF5_USE_FILE_LOCKING` overrides the requested policy, as in the C
    /// library.
    pub fn open_with_locking<P: AsRef<Path>>(path: P, locking: FileLocking) -> Result<Self, Error> {
        Self::open_inner(path.as_ref(), Some(locking))
    }

    /// Opens a PAGE mirror configured to inject one publication error.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] unless the file uses non-persistent PAGE allocation.
    /// Returns the opening or buffering error if the mirror cannot be configured.
    #[cfg(test)]
    pub(crate) fn open_page_publication_failure(
        path: &Path,
        failure: PagePublicationFailure,
        gathered: bool,
    ) -> Result<Self, Error> {
        let mut session = Self::open_imaged(path, Some(FileLocking::Enabled), |handle, _| {
            let inner = Self::read_mirror(handle)?;
            let publication_offset = superblock::read(&inner)?.address.get();
            Ok(Box::new(PublicationFailureImage::new(
                inner,
                publication_offset,
                failure,
            )))
        })?;
        if session.fresh_page_boundary.is_none() {
            return Err(Error::EditUnsupported(
                "publication failure crosschecks require non-persistent PAGE allocation",
            ));
        }
        if !gathered {
            session
                .image
                .set_write_buffering(WriteBuffering::Unbuffered)?;
        }
        Ok(session)
    }

    /// Opens a complete file mirror with reads through [`Source`].
    ///
    /// Requests [`FileLocking::Enabled`] and retains the complete file in memory.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if `path` cannot be opened for reading and writing. Returns the
    /// locking and format errors of [`Self::open_with_locking`].
    #[cfg(test)]
    pub(crate) fn open_source_only(path: &Path) -> Result<Self, Error> {
        Self::open_imaged(path, Some(FileLocking::Enabled), |handle, _len| {
            Ok(Box::new(crate::image::SourceOnlyImage::new(
                Self::read_mirror(handle)?,
            )))
        })
    }

    /// Open exactly as [`open_with_locking`](Self::open_with_locking) does, but
    /// behind an image whose writes into `fails` behave like a dying device:
    /// the first alters the file and then errors, the rest are refused
    /// outright. See [`crate::image::TornWriteImage`] for why it takes both to
    /// reach the state a refused commit cannot repair.
    #[cfg(test)]
    pub(crate) fn open_torn_writes(
        path: &Path,
        fails: core::ops::Range<u64>,
    ) -> Result<Self, Error> {
        Self::open_imaged(path, Some(FileLocking::Enabled), |handle, _len| {
            Ok(Box::new(crate::image::TornWriteImage::new(
                Self::read_mirror(handle)?,
                fails,
            )))
        })
    }

    /// Open a bounded session whose image counts the bytes read through it, so a
    /// test can assert that an operation touches only a small part of the file.
    /// The counter is shared with the caller.
    #[cfg(test)]
    pub(crate) fn open_bounded_counting(
        path: &Path,
        read_bytes: std::sync::Arc<std::sync::atomic::AtomicU64>,
    ) -> Result<Self, Error> {
        let mut session = Self::open_imaged(path, Some(FileLocking::Enabled), |handle, len| {
            Ok(Box::new(crate::image::CountingImage::new(
                Box::new(HandleImage::new(
                    handle,
                    len,
                    crate::source::MetadataCacheConfig::disabled(),
                )),
                read_bytes,
                std::sync::Arc::default(),
            )))
        })?;
        session.batched_appends = true;
        Ok(session)
    }

    /// Open a session under `policy` whose image counts the `fsync`s issued
    /// through it, so a test can assert what a write path actually costs. The
    /// counter is shared with the caller.
    ///
    /// It takes the bounded backing and its flags, so the session matches what
    /// [`File::open_rw`](crate::File::open_rw) builds for a latest-format file;
    /// the barrier sites live on the engine, above the choice of image, so one
    /// backing exercises them all.
    #[cfg(test)]
    pub(crate) fn open_sync_counting(
        path: &Path,
        policy: SyncPolicy,
        syncs: std::sync::Arc<std::sync::atomic::AtomicU64>,
    ) -> Result<Self, Error> {
        let mut session = Self::open_imaged(path, Some(FileLocking::Enabled), |handle, len| {
            Ok(Box::new(crate::image::CountingImage::new(
                Box::new(HandleImage::new(
                    handle,
                    len,
                    crate::source::MetadataCacheConfig::disabled(),
                )),
                std::sync::Arc::default(),
                syncs,
            )))
        })?;
        session.batched_appends = true;
        session.bounded = true;
        session.set_sync_policy(policy);
        Ok(session)
    }

    /// Opens a writable session under the selected backing and persisted-manager policies.
    ///
    /// Falls back to a mirror only for [`bounded_only_limitation`]. Metadata and acquisition
    /// errors are returned directly. Preserves the persisted-manager policy when reopening
    /// the file through a mirror under [`MemoryStrategy::Auto`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] for unsupported address widths or layouts, including
    /// a backing that cannot satisfy [`MemoryStrategy::Bounded`]. Other failures are the errors
    /// of [`open_imaged_with_policy`](Self::open_imaged_with_policy).
    pub(crate) fn open_rw_with_strategy(
        path: &Path,
        cache: MetadataCacheConfig,
        locking: FileLocking,
        strategy: MemoryStrategy,
        policy: PersistedFreeSpacePolicy,
    ) -> Result<Self, Error> {
        if strategy == MemoryStrategy::Mirrored {
            return Self::open_inner_with_policy(path, Some(locking), policy);
        }
        let mut session =
            Self::open_imaged_with_policy(path, Some(locking), policy, |handle, len| {
                Ok(Box::new(HandleImage::new(handle, len, cache)))
            })?;
        session.batched_appends = true;
        session.bounded = true;
        if let Some(reason) = bounded_only_limitation(&session) {
            if strategy == MemoryStrategy::Bounded {
                return Err(Error::EditUnsupported(reason));
            }
            // Release the handle and its exclusive lock before reopening, or the
            // mirrored open would contend with the probe we are discarding —
            // fatally so on Windows, where the OS lock is mandatory. Dropping a
            // bare `WriteEngine` writes nothing: the free-space finalize that a
            // dropped writer owes lives on `FileInner`, which this is not yet.
            // Another writer can take the lock in that window; the reopen then
            // reports `Error::FileLocked`, which is the truthful answer.
            drop(session);
            return Self::open_inner_with_policy(path, Some(locking), policy);
        }
        Ok(session)
    }

    /// Open an existing file for SWMR (single-writer/multiple-reader) writing:
    /// take **no** OS lock at all and raise the superblock's SWMR-write
    /// consistency flag. Backs [`File::open_swmr_writer`](crate::File::open_swmr_writer).
    ///
    /// The no-lock is unconditional — `lock = None` never reaches
    /// `acquire_exclusive`, so `HDF5_USE_FILE_LOCKING` cannot reintroduce a lock
    /// that would block the concurrent readers SWMR exists to permit (fatally so
    /// on Windows, where OS locks are mandatory). Requires a latest-format
    /// (version-3 superblock) file with no userblock and no persisted
    /// free-space, so the superblock can be rewritten in place.
    ///
    /// The version-3 requirement is the C library's (`H5F__super_read`: "superblock
    /// version for SWMR is less than 3"), and it is what keeps the SWMR-write flag
    /// meaningful: neither library reads the status-flags byte back on an older
    /// superblock, so a flag raised there would announce a live writer to nobody.
    /// This crate's writer emits version 3, so no file it produces is affected.
    pub(crate) fn open_swmr_writer<P: AsRef<Path>>(
        path: P,
        sync_policy: SyncPolicy,
    ) -> Result<Self, Error> {
        let mut session = Self::open_inner(path.as_ref(), None)?;
        // Before the flag write below, which is a durability point like any
        // other: a caller who asked for no `fsync` gets none, and the flag still
        // reaches every other process, which reads it from the operating system.
        session.set_sync_policy(sync_policy);
        if session.fresh_page_boundary.is_some() {
            return Err(Error::SwmrAppendUnsupported(
                "SWMR writing of PAGE files without persisted free space is unsupported; \
                 use File::open_rw with staged edits and File::commit",
            ));
        }
        if session.context.superblock().version() != SuperblockVersion::Three
            || !session.context.superblock().base_address.is_zero()
            || session.persist.is_some()
        {
            return Err(Error::SwmrAppendUnsupported(
                "SWMR writing requires a latest-format file (v3 superblock) with no userblock \
                 and no persisted free-space",
            ));
        }
        session.swmr_mode = true;
        if session.space.is_enabled() {
            session.aggregation = None;
        }
        session.set_consistency_flags(SWMR_WRITE_FLAGS)?;
        Ok(session)
    }

    /// Writes the consistency flags in the session's superblock and physical image.
    ///
    /// Adopts the flags in the file context after the write and data durability barrier succeed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the flags exceed the version's field width or the proposed
    /// superblock cannot be serialized. Returns [`Error::Io`] if writing or syncing the image
    /// fails.
    fn set_consistency_flags(&mut self, flags: SuperblockStatusFlags) -> Result<(), Error> {
        let mut superblock = self.context.superblock().clone();
        superblock.set_consistency_flags(flags)?;
        let publication = self.context.propose(superblock)?.encode()?;
        // A failed clear retains ownership of any mark the session may have written.
        if !flags.is_empty() {
            self.held_status_flags = flags;
        }
        self.write_at(self.context.signature().get(), publication.bytes())?;
        self.barrier_data()?;
        self.context.adopt(publication.into_proposed())?;
        self.held_status_flags = flags;
        Ok(())
    }

    /// Raise the page buffer's crash mark: superblock status-flag bit 0
    /// (`H5F_SUPER_WRITE_ACCESS`), the byte the C library raises for any writer
    /// and this crate otherwise raises only for a SWMR one.
    ///
    /// Why a page buffer is not offered without one is
    /// [`with_page_buffer_size`](crate::FileAccessProperties::with_page_buffer_size)'s
    /// to explain; what matters here is that two properties of *this* function
    /// are load-bearing, and neither is the [`SyncPolicy`]'s to skip:
    ///
    /// - It runs **before** the buffering mode changes, so no buffered write can
    ///   reach the disk ahead of it. Its caller states why nothing can fail in
    ///   between.
    /// - It is **fsynced**, unconditionally — the half no test here can see.
    ///   Reaching the operating system already covers a process that dies, and
    ///   this suite reproduces no power loss, so deleting the `sync_all` fails
    ///   nothing. It is there because power loss is exactly the case where the
    ///   buffer's held writes may be partly on the platter while a mark still in
    ///   the page cache is not. One `fsync` per session buys it.
    ///
    /// On failure, attempts to restore and synchronize the clear flags through the same handle.
    /// If that fails, opening the file can require
    /// [`File::clear_status_flags`](crate::File::clear_status_flags) despite the failed session.
    fn raise_crash_mark(&mut self) -> Result<(), Error> {
        let raised = self
            .set_consistency_flags(SuperblockStatusFlags::write_marked())
            .and_then(|()| self.image.sync_all());
        if raised.is_err() {
            self.set_consistency_flags(SuperblockStatusFlags::new())
                .ok();
            self.image.sync_all().ok();
        }
        raised
    }

    /// Take down whatever status flags this session raised, once everything they
    /// stood for is durable — the SWMR pair, or a page buffer's crash mark. A
    /// no-op for a session that raised none, which is every ordinary
    /// [`File::open_rw`](crate::File::open_rw) session.
    ///
    /// The whole byte goes to zero, so this cannot leave the half-set state
    /// [`check_status_flags`](crate::file_lock::check_status_flags) reports as
    /// flags in disagreement; that is why the guard asks whether anything is held
    /// rather than which bits are.
    ///
    /// Called from `File::close` and `FileInner::drop`, **after** their
    /// `force_sync`. The order is the whole point: clearing first and flushing
    /// second would leave a window in which the file's writes are still in memory
    /// and nothing on the disk says so, which is the state the flags exist to
    /// make unreachable. Nothing in this suite can tell the two orders apart —
    /// it takes a crash inside the window — so the argument is the reason, not a
    /// test.
    pub(crate) fn release_status_flags(&mut self) -> Result<(), Error> {
        if self.held_status_flags.is_empty() {
            return Ok(());
        }
        self.set_consistency_flags(SuperblockStatusFlags::new())?;
        self.image.sync_all()
    }

    /// Shared open path. `lock = Some(policy)` acquires an exclusive OS lock under
    /// that policy (the ordinary read-write session); `lock = None` takes no lock
    /// at all (the SWMR writer — see [`open_swmr_writer`](Self::open_swmr_writer)).
    fn open_inner(path: &Path, lock: Option<FileLocking>) -> Result<Self, Error> {
        Self::open_imaged(path, lock, |handle, _len| {
            Ok(Box::new(Self::read_mirror(handle)?))
        })
    }

    /// Opens a mirrored session with the selected persisted-manager policy.
    ///
    /// # Errors
    ///
    /// Returns the acquisition and parsing errors of
    /// [`open_imaged_with_policy`](Self::open_imaged_with_policy).
    fn open_inner_with_policy(
        path: &Path,
        lock: Option<FileLocking>,
        policy: PersistedFreeSpacePolicy,
    ) -> Result<Self, Error> {
        if policy == PersistedFreeSpacePolicy::Tolerant {
            return match lock {
                Some(locking) => Self::open_with_locking(path, locking),
                None => Self::open_inner(path, None),
            };
        }
        Self::open_imaged_with_policy(path, lock, policy, |handle, _len| {
            Ok(Box::new(Self::read_mirror(handle)?))
        })
    }

    /// Read `handle` whole into a [`MirrorImage`]. The one place the engine
    /// still assumes it can hold the file, kept behind a named constructor so
    /// the mirrorless opens visibly do not call it.
    ///
    /// `read_to_end` reads from the handle's *current* cursor, and the open path
    /// has already read the superblock through it, so this rewinds first. Reading
    /// from wherever the last read landed would mirror a truncated file — with no
    /// error to say so, since a short mirror is a valid `Vec<u8>`.
    fn read_mirror(mut handle: fs::File) -> Result<MirrorImage, Error> {
        handle.seek(SeekFrom::Start(0)).map_err(Error::Io)?;
        let mut data = Vec::new();
        handle.read_to_end(&mut data).map_err(Error::Io)?;
        Ok(MirrorImage::new(handle, data))
    }

    /// Opens a session with tolerant persisted free-space recovery.
    ///
    /// Calls the image constructor with the handle and physical file length after parsing
    /// the superblock and File Space Info.
    ///
    /// # Errors
    ///
    /// Returns the acquisition and parsing errors of
    /// [`open_imaged_with_policy`](Self::open_imaged_with_policy).
    fn open_imaged(
        path: &Path,
        lock: Option<FileLocking>,
        build: impl FnOnce(fs::File, u64) -> Result<Box<dyn FileImage>, Error>,
    ) -> Result<Self, Error> {
        Self::open_imaged_with_policy(path, lock, PersistedFreeSpacePolicy::Tolerant, build)
    }

    /// Parses opening metadata before constructing the editing image.
    ///
    /// Under [`PersistedFreeSpacePolicy::Strict`], reads the complete persisted manager set and
    /// extension allocation ranges before calling `build`. Strict metadata parsing failures occur
    /// before allocator installation or page buffering. Calls `build` with the handle and
    /// physical file length.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] for opening failures, and for superblock or extension read failures
    /// under the strict policy. Returns [`Error::FileLocked`] for a lock conflict,
    /// [`Error::FileMarkedInUse`] for writer status
    /// flags, or [`Error::PersistedFreeSpace`] for a strict manager read or parse failure.
    /// Returns [`Error::Format`] for invalid opening metadata or address arithmetic.
    /// Other failures are the errors of `build` and backing configuration.
    fn open_imaged_with_policy(
        path: &Path,
        lock: Option<FileLocking>,
        policy: PersistedFreeSpacePolicy,
        build: impl FnOnce(fs::File, u64) -> Result<Box<dyn FileImage>, Error>,
    ) -> Result<Self, Error> {
        let handle = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(Error::Io)?;
        // Acquire the exclusive lock before reading or mutating; the retained
        // `handle` holds it for the session's life. A `None` policy (SWMR) never
        // reaches `acquire_exclusive`, so no lock is ever taken.
        if let Some(policy) = lock {
            file_lock::acquire_exclusive(&handle, policy, path)?;
        }
        let len = handle.metadata().map_err(Error::Io)?.len();
        // Read the superblock through the handle itself, before any image owns
        // it. `probe` borrows, so it is gone by the time `build` takes the
        // handle; it leaves the handle's cursor wherever its last read ended,
        // which is why the mirror positions the handle before reading it whole.
        let probe = crate::image::BorrowedHandle::new(&handle, len);
        let located = match policy {
            PersistedFreeSpacePolicy::Strict => probe.finish(superblock::read(&probe)),
            PersistedFreeSpacePolicy::Tolerant => superblock::read(&probe).map_err(Error::from),
        }?;
        file_lock::check_status_flags(
            &located.superblock,
            file_lock::OpenIntent::Write,
            file_lock::OpenTarget::Path(path),
        )?;
        let context = FileContext::new(located)?;
        let superblock = context.superblock();
        let addresses = context.addresses();
        let sb_sig_off = context.signature().get();

        // `H5Fopen` rejects a page size outside 512 bytes to 1 GiB as well (`H5Ofsinfo.c` and
        // `H5Fsuper.c`, HDF5 2.2.0).
        let OpeningExtension {
            info: file_space_info,
            allocations,
        } = match policy {
            PersistedFreeSpacePolicy::Strict => {
                probe.finish(OpeningExtension::read(&probe, &context))
            }
            PersistedFreeSpacePolicy::Tolerant => {
                OpeningExtension::read(&probe, &context).map_err(Error::from)
            }
        }?;
        let (recovered, extension_allocations) = if policy == PersistedFreeSpacePolicy::Strict
            && let Some((_, info)) = file_space_info.as_ref().filter(|(_, info)| info.persist)
        {
            (
                Some(PersistedRecovery::read(
                    &probe,
                    &context,
                    info,
                    allocations,
                )?),
                None,
            )
        } else {
            (None, allocations.ok())
        };
        let fresh_page_boundary = if let Some((_, info)) = &file_space_info
            && info.strategy == FileSpaceStrategy::Page
            && !info.persist
        {
            let stored_eoa = addresses.relative(PhysicalAddress::new(superblock.eof_address))?;
            let superblock_len =
                hdf5_pure_format::__private::serialize_superblock(superblock)?.len() as u64;
            let superblock_end =
                sb_sig_off
                    .checked_add(superblock_len)
                    .ok_or(FormatError::OffsetOverflow {
                        offset: sb_sig_off,
                        length: superblock_len,
                    })?;
            if superblock.eof_address < superblock_end {
                return Err(Error::InvalidSpaceOwnershipExtent {
                    address: sb_sig_off,
                    length: superblock_len,
                    previous_end: sb_sig_off,
                    logical_size: superblock.eof_address,
                });
            }
            if superblock.eof_address > len {
                return Err(FormatError::TruncatedFile {
                    expected_eof: superblock.eof_address,
                    actual_len: len,
                }
                .into());
            }
            Some(Self::fresh_page_boundary(
                addresses,
                stored_eoa,
                len,
                info.page_size,
            )?)
        } else {
            None
        };

        let image = build(handle, len)?;

        let mut session = Self {
            image,
            context,
            staged: StagedEdits::default(),
            appender_claims: Vec::new(),
            next_appender_token: 0,
            // A file without a File Space Info message has the library's default strategy.
            space: SessionSpace::from_strategy(
                Some(FileSpaceStrategy::FsmAggr),
                DEFAULT_THRESHOLD,
                FileSpacePageSize::DEFAULT,
            ),
            aggregation: None,
            append_allocator: allocation::AppendAllocator::new(),
            proved_free_of_references: false,
            persist: None,
            fresh_page_boundary,
            located: HashMap::new(),
            vl_overwrite_heaps: HashMap::new(),
            superseded_heaps: Vec::new(),
            inplace_undo: Vec::new(),
            swmr_mode: false,
            committed: false,
            resolved: HashMap::new(),
            batched_appends: false,
            bounded: false,
            libver_ceiling: None,
            fsm_len: len,
            fsm_dirty: false,
            publication: PublicationState::Ready,
            staging_batch: false,
            staged_generation: 0,
            held_status_flags: SuperblockStatusFlags::new(),
            sync_policy: SyncPolicy::Always,
        };
        // If the file persists its free space, seed the free list from the
        // on-disk managers and arm persistence for future commits.
        session.load_persisted_free_space(file_space_info, recovered, extension_allocations)?;
        if session.fresh_page_boundary.is_some() {
            session.space_ownership()?;
        }
        // Gather this session's writes, now that the page size the file was laid
        // out on is known. Only an exclusively locked session: `lock = None` is
        // the SWMR writer, whose concurrent readers observe the order its ordered
        // phases become visible in, and so must see every write as it is made
        // (issue #288).
        //
        // `lock.is_some()` is a proxy for "not the SWMR writer", used because
        // `swmr_mode` is not set until after this. It is exact only while
        // `open_swmr_writer` is the sole lock-free entry point: a future one that
        // took a lock would silently *gain* gathering, which is the direction
        // that costs a reader. The other direction — a lock-free non-SWMR open
        // losing gathering — costs only writes. Whoever adds such an entry point
        // owns this condition.
        if lock.is_some() {
            let page_size = session.gather_page_size();
            session
                .image
                .set_write_buffering(WriteBuffering::Operation {
                    page_size,
                    max_bytes: WRITE_GATHER_BYTES,
                })?;
        }
        Ok(session)
    }

    /// The page this session's writes are merged within: the file's own
    /// file-space page size when it is paged, and the format's default otherwise.
    fn gather_page_size(&self) -> u64 {
        self.space
            .page_size()
            .map_or(DEFAULT_GATHER_PAGE, FileSpacePageSize::get)
    }

    /// Configures session buffering with a byte budget of `max_bytes`.
    ///
    /// A zero budget preserves the default operation buffering. Immediate append publication
    /// drains either mode. Staged commits can retain dirty bytes across general ordering barriers.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] for a budget below the merge page size, a SWMR writer,
    /// nonpersistent PAGE storage, a superblock older than version 3, or [`SyncPolicy::Always`].
    /// Returns [`Error::Io`] if publishing the session's write-access mark fails.
    pub(crate) fn set_page_buffer_size(&mut self, max_bytes: usize) -> Result<(), Error> {
        if max_bytes == 0 {
            return Ok(());
        }
        // A lock-free session's readers observe the order its writes become
        // visible in. `File::open_swmr_writer` refuses this property before it
        // raises the on-disk flag, which is the refusal a caller sees; this one is
        // here because the guarantee belongs to the engine that would break it,
        // and a SWMR session can otherwise clear every check below — an unpaged
        // version-3 file with an ample budget is exactly what SWMR writing wants.
        if self.swmr_mode {
            return Err(Error::EditUnsupported(
                "a SWMR writer cannot buffer its writes: its readers observe the order they \
                 become visible in",
            ));
        }
        // One page is the whole of the size rule, and a buffer that cannot hold
        // one drains on every page it touches.
        //
        // A budget below `WRITE_GATHER_BYTES` was refused alongside it until
        // issue #391, on the grounds that a page buffer *replaces* the byte
        // budget the session was already gathering under rather than adding to
        // it, so a smaller one is also the point at which one long run is flushed
        // and restarted. That cost is real — nothing here bypasses the gatherer
        // the way `H5PB_write` bypasses the C page buffer for I/O of a page or
        // more (issue #357) — but it is not this crate's trade to make. The
        // budget it replaces is a per-operation peak released at every barrier,
        // while this one is held across operations, so a smaller number lowers
        // both the peak and the residency, and a writer inside a tight memory cap
        // is entitled to spend the writes for that.
        let page_size = self.gather_page_size();
        if (max_bytes as u64) < page_size {
            return Err(Error::EditUnsupported(
                "a page buffer must be at least the file's file-space page size",
            ));
        }
        // A page buffer holds dirty pages across ordering barriers, and under
        // `Always` every barrier is an `fsync` that flushes it — so it would hold
        // nothing, and the caller would have paid the mark below for nothing in
        // return. It lives here with its three siblings rather than at the fapl,
        // because a refusal kept apart from the others is one
        // `create_would_refuse_reopen` forgets to restate, which is how this one
        // came to write the file before refusing it.
        if self.sync_policy == SyncPolicy::Always {
            return Err(Error::EditUnsupported(
                "a page buffer does nothing under SyncPolicy::Always, whose every barrier is \
                 an fsync that flushes it; pair with_page_buffer_size with \
                 with_sync_policy(SyncPolicy::OnClose), or drop it",
            ));
        }
        // Non-persistent PAGE commits require ordered publication without page buffering.
        //
        // Scoped to a paged file: an *unpaged* one commits and appends whether it
        // persists its free space or not, so there is nothing here to refuse. The
        // base address the mark's superblock rewrite needs is not this check's to
        // settle — `set_consistency_flags` converts the root address itself, and
        // is tested for a userblock file directly rather than through a caller.
        if self.space.is_paged() && self.persist.is_none() {
            return Err(Error::EditUnsupported(
                "a page buffer on a paged file needs its free space persisted: staged commits \
                 without persisted free space require ordered publication without page \
                 buffering; recreate the file with \
                 with_file_space_settings(FileSpaceSettings::new(\
                 FileSpaceStrategy::Page).with_persistence(true)), or drop the \
                 page buffer",
            ));
        }
        // The crash mark below is the whole reason this property is offered at
        // all, and only a version-3 superblock carries one that anybody reads:
        // `check_status_flags` is gated there, matching where the C library gates
        // it. A version-2 paged file is reachable — `LibVer::V18` writes a
        // version-2 superblock and nothing about paged file space rules it out —
        // so this is a refusal rather than an assertion (issue #308).
        if self.context.superblock().version() != SuperblockVersion::Three {
            return Err(Error::EditUnsupported(
                "a page buffer marks the file for the life of the session, so that a session \
                 that crashes leaves one every reader refuses rather than one that reads \
                 clean; only a version-3 superblock carries a status-flags byte any library \
                 reads back, and this file's is older. Rewrite it at the 1.10 format \
                 (repack, or FileBuilder's default bounds), or drop the page buffer",
            ));
        }
        // Before the mode change, and durable: from here on a write may sit in
        // memory across a barrier, and the file has to say so on the disk first.
        //
        // Nothing can fail between the two and leave the file marked with no
        // session behind it. `raise_crash_mark` ends in a `sync_all`, which
        // flushes, so the gatherer is empty when `set_write_buffering` flushes it
        // again — the one fallible step it takes.
        self.raise_crash_mark()?;
        self.image.set_write_buffering(WriteBuffering::Session {
            page_size,
            max_bytes,
        })
    }

    /// Configures reusable space and records allocations for persistent-manager rewrites.
    ///
    /// Installs checked recovery when supplied. Otherwise, reads every defined manager tolerantly.
    /// Excludes the entire manager set from reusable space if any manager read fails. After
    /// successful reads, omits invalid, overlapping, or out-of-bounds extents. Uses prepared
    /// extension ranges when available and omits unsupported extension allocations.
    /// A nonpersistent PAGE session starts with empty free lists. Files without File Space Info
    /// use the default `FSM_AGGR` strategy with separate metadata and raw aggregates.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] with [`FormatError::Internal`] if checked recovery cannot be
    /// represented as allocator extents or canonical free lists.
    fn load_persisted_free_space(
        &mut self,
        file_space_info: Option<(u64, FileSpaceInfo)>,
        recovered: Option<PersistedRecovery>,
        extension_allocations: Option<Vec<FileRange>>,
    ) -> Result<(), Error> {
        let Some((_, info)) = file_space_info else {
            if let Ok(eoa) = self.end_of_allocation() {
                self.aggregation = Some(AggregateSpace::new(eoa));
            }
            return Ok(());
        };
        let mut space =
            SessionSpace::from_strategy(Some(info.strategy), info.threshold, info.page_size);
        if (info.strategy == FileSpaceStrategy::Aggr || info.strategy == FileSpaceStrategy::FsmAggr)
            && let Ok(eoa) = self.end_of_allocation()
        {
            self.aggregation = Some(AggregateSpace::new(eoa));
        }
        let paged = space.is_paged();
        if !info.persist || !space.is_enabled() {
            self.space = space;
            return Ok(());
        }
        if let Some(recovered) = recovered {
            return self.install_persisted_free_space(info, recovered);
        }
        let widths = self.context.widths();
        let Ok(eoa) = self.end_of_allocation() else {
            self.space = space;
            return Ok(());
        };
        let managers: Vec<(
            Option<ManagerKind>,
            StoredAddress,
            Result<PersistedSections, FormatError>,
        )> = info
            .manager_addrs
            .iter()
            .enumerate()
            .filter_map(|(slot, &m)| match m {
                DecodedStoredAddress::Defined(addr) => Some((slot, addr)),
                DecodedStoredAddress::Undefined => None,
            })
            .map(|(slot, addr)| {
                let read = free_space_manager::read_persisted_sections_source(
                    &self.image(),
                    widths,
                    self.context.addresses().effective_base(),
                    &[DecodedStoredAddress::Defined(addr)],
                );
                (
                    free_space_manager::manager_kind(info.strategy, slot),
                    addr,
                    read,
                )
            })
            .collect();

        // Translate storage manager identity before the sections enter free-space policy. A parse
        // failure seeds nothing, so reusable state never describes only part of the persisted
        // manager set. Invalid, overlapping, or out-of-bounds section extents are skipped before
        // they enter the runtime strategy.
        if paged {
            if managers.iter().all(|(_, _, read)| read.is_ok()) {
                let mut tagged: Vec<(Option<ManagerKind>, FreeExtent)> = Vec::new();
                for (kind, _, read) in &managers {
                    let Ok((sections, _)) = read else {
                        continue;
                    };
                    for &section in sections {
                        let Some(extent) = FreeExtent::new(section.addr, section.size) else {
                            continue;
                        };
                        tagged.push((*kind, extent));
                    }
                }
                tagged.sort_unstable_by_key(|&(_, extent)| extent.start());
                let mut prev_end = StoredAddress::ZERO;
                let admitted = tagged.into_iter().filter(|&(_, extent)| {
                    if extent.end() > eoa || extent.start() < prev_end {
                        return false;
                    }
                    prev_end = extent.end();
                    true
                });
                space = Self::recovered_space(&info, admitted)?;
            }
        } else if let Ok(mut sections) = managers.iter().try_fold(
            Vec::new(),
            |mut all, (_, _, read)| -> Result<_, &FormatError> {
                all.extend_from_slice(&read.as_ref()?.0);
                Ok(all)
            },
        ) {
            sections.sort_unstable_by_key(|section| section.addr);
            let mut prev_end = StoredAddress::ZERO;
            let admitted = sections.into_iter().filter_map(|section| {
                let extent = FreeExtent::new(section.addr, section.size)?;
                if extent.end() > eoa || extent.start() < prev_end {
                    return None;
                }
                prev_end = extent.end();
                Some((Some(ManagerKind::Flat), extent))
            });
            space = Self::recovered_space(&info, admitted)?;
        }

        let mut old_blocks = Vec::new();
        if let Some(spans) = extension_allocations {
            old_blocks.extend(
                spans
                    .into_iter()
                    .filter_map(|range| self.stored_extent(range)),
            );
        }
        for (_, manager, read) in &managers {
            match read {
                Ok((_, blocks)) => {
                    old_blocks.extend(blocks.iter().filter_map(|range| self.stored_extent(*range)))
                }
                Err(_) => old_blocks.extend(self.manager_header_block(widths, *manager)),
            }
        }

        self.space = space;
        self.persist = Some(PersistState {
            strategy: info.strategy,
            threshold: info.threshold,
            page_size: info.page_size,
            old_blocks,
        });
        Ok(())
    }

    /// Installs eligible sections and reclamation ranges from completed recovery.
    ///
    /// Retains metadata allocations even for manager roles the allocator does not admit.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] with [`FormatError::Internal`] if checked recovery cannot be
    /// represented as allocator extents or canonical free lists.
    fn install_persisted_free_space(
        &mut self,
        info: FileSpaceInfo,
        recovered: PersistedRecovery,
    ) -> Result<(), Error> {
        let PersistedRecovery {
            managers,
            extension,
        } = recovered;
        let (report, allocations) = managers.into_parts();
        let old_blocks = extension
            .into_iter()
            .chain(allocations)
            .map(|range| {
                self.stored_extent(range).ok_or_else(|| {
                    FormatError::Internal(
                        "a checked metadata allocation has no stored extent".into(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut tagged = report
            .managers()
            .iter()
            .flat_map(|manager| {
                manager.free_sections().sections().iter().map(|section| {
                    FreeExtent::new(section.address(), section.length().get())
                        .map(|extent| {
                            (
                                free_space_manager::manager_kind(info.strategy, manager.slot()),
                                extent,
                            )
                        })
                        .ok_or_else(|| {
                            FormatError::Internal(
                                "a checked free section has no allocator extent".into(),
                            )
                        })
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        tagged.sort_unstable_by_key(|(_, extent)| extent.start());
        let space = Self::recovered_space(&info, tagged)?;
        self.space = space;
        self.persist = Some(PersistState {
            strategy: info.strategy,
            threshold: info.threshold,
            page_size: info.page_size,
            old_blocks,
        });
        Ok(())
    }

    /// Constructs session space from address-sorted sections admitted by recovery.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] with [`FormatError::Internal`] if admitted sections are unsorted
    /// or overlap.
    fn recovered_space(
        info: &FileSpaceInfo,
        sections: impl IntoIterator<Item = (Option<ManagerKind>, FreeExtent)>,
    ) -> Result<SessionSpace, Error> {
        SessionSpace::from_persisted(
            Some(info.strategy),
            info.threshold,
            info.page_size,
            sections,
        )
        .map_err(|error| {
            FormatError::Internal(format!(
                "admitted free sections cannot form a free list: {error}"
            ))
            .into()
        })
    }

    /// Returns the extent of the free-space manager header at the stored address `addr`, in stored
    /// addresses, or `None` if the read fails or the header does not parse.
    fn manager_header_block(
        &self,
        widths: FormatWidths,
        addr: StoredAddress,
    ) -> Option<FreeExtent> {
        let len = hdf5_pure_format::__private::free_space_manager_header_len(widths);
        let at = self.context.addresses().resolve(addr).ok()?;
        let fshd = self
            .image()
            .read_metadata_at(at.get(), len.to_usize().ok()?)
            .ok()?;
        FreeSpaceManagerHeader::parse(widths, &fshd)
            .ok()
            .and_then(|_| FreeExtent::new(addr, len))
    }

    /// Converts a nonempty physical range to a stored-coordinate allocator extent.
    ///
    /// Returns `None` if the range is empty or begins below the effective address base.
    fn stored_extent(&self, range: FileRange) -> Option<FreeExtent> {
        FreeExtent::new(
            self.context.addresses().relative(range.start()).ok()?,
            range.length(),
        )
    }

    /// Returns the first stored page boundary at or beyond both extents.
    ///
    /// Physical positions are converted relative to the effective base before alignment.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressBelowBase`] if `physical_len` is below the effective base.
    /// Returns [`FormatError::OffsetOverflow`] if alignment or physical resolution exceeds
    /// [`u64::MAX`].
    fn fresh_page_boundary(
        addresses: AddressContext,
        eoa: StoredAddress,
        physical_len: u64,
        page_size: FileSpacePageSize,
    ) -> Result<StoredAddress, FormatError> {
        let start = addresses
            .relative(PhysicalAddress::new(physical_len))?
            .get()
            .max(eoa.get());
        let boundary =
            start
                .checked_next_multiple_of(page_size.get())
                .ok_or(FormatError::OffsetOverflow {
                    offset: start,
                    length: page_size.get(),
                })?;
        let boundary = StoredAddress::new(boundary);
        addresses.resolve(boundary)?;
        Ok(boundary)
    }

    /// Returns the end of the image as a stored address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressBelowBase`] if the image ends below the base address.
    fn end_of_allocation(&self) -> Result<StoredAddress, FormatError> {
        self.context
            .addresses()
            .relative(PhysicalAddress::new(self.image.len()))
    }

    /// Reads the File Space Info message from the superblock extension.
    ///
    /// Returns the extension's physical position and the message, or `None` if either is absent.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the extension's absolute address exceeds `u64`,
    /// and the parser's error if the extension's object header or the message does not parse.
    fn extension_fsinfo(
        source: &(impl Source + ?Sized),
        context: &FileContext,
    ) -> Result<Option<(u64, FileSpaceInfo)>, FormatError> {
        Ok(Self::extension_summary(source, context, None)?
            .map(ExtensionSummary::into_file_space_info)
            .transpose()?
            .flatten()
            .map(|(address, info)| (address.get(), info)))
    }

    /// Resolves the extension address and reads its selected facts under a source borrow.
    ///
    /// Returns `None` when the superblock has no defined extension address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if address resolution overflows, and header or
    /// source-read errors from [`ExtensionSummary::read`].
    fn extension_summary<'a, S: Source + ?Sized>(
        source: &'a S,
        context: &FileContext,
        observer: Option<&mut dyn FnMut(ObjectHeaderChunk<'_>)>,
    ) -> Result<Option<ExtensionSummary<'a, S>>, FormatError> {
        let superblock = context.superblock();
        let widths = context.widths();
        let base = context.addresses();
        if matches!(
            superblock.version(),
            SuperblockVersion::Zero | SuperblockVersion::One
        ) {
            return Ok(None); // no superblock extension exists before v2
        }
        let Some(ext_rel) = superblock.superblock_extension_address else {
            return Ok(None);
        };
        if ext_rel.is_undefined(widths.offsets) {
            return Ok(None);
        }
        // The header is read at the extension's absolute file offset, which is the stored
        // address on a file without a userblock. The caller reads the strategy of a file with a
        // userblock as well, so the address is resolved here for both.
        let ext_addr = base.resolve(ext_rel)?.get();
        ExtensionSummary::read(
            source,
            AccessMode::ReadWrite,
            PhysicalAddress::new(ext_addr),
            widths,
            base,
            observer,
        )
        .map(Some)
    }

    /// Reserves exclusive appender access and returns its release token.
    ///
    /// `path` is [`None`] for a handle reached by object reference.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if another appender or staged edit conflicts with the
    /// claim. Returns [`Error::AllocationPublicationUncertain`] if an earlier editing outcome is
    /// uncertain, or [`FormatError::Internal`] if an allocation batch remains provisional.
    pub(crate) fn claim_for_appender(
        &mut self,
        path: Option<&ObjectPathBuf>,
    ) -> Result<u64, Error> {
        self.check_append_publication()?;
        if self
            .appender_claims
            .iter()
            .any(|claim| claim.conflicts_with(path))
        {
            return Err(Error::EditUnsupported(
                "this dataset already has a live buffered appender; two of them would interleave \
                 their buffers a chunk at a time",
            ));
        }
        let blocked = match path {
            Some(p) => self.append_conflicts_with_pending(p),
            None => self.has_staged_edits() || self.committed,
        };
        if blocked {
            return Err(Error::EditUnsupported(
                "this session holds staged edits that would stop a buffered appender from \
                 flushing; commit or discard them before opening one",
            ));
        }
        let token = self.next_appender_token;
        self.next_appender_token += 1;
        self.appender_claims.push(AppenderClaim {
            token,
            path: path.cloned(),
        });
        Ok(token)
    }

    /// Drop a claim. Called from the appender's `Drop`, after its final flush.
    pub(crate) fn release_appender_claim(&mut self, token: u64) {
        self.appender_claims.retain(|c| c.token != token);
    }

    /// Refuse a staged edit at `path` that a live appender could not survive.
    ///
    /// An appender holds elements a caller has already been told were accepted,
    /// and only its own flush can write them; that flush goes through the
    /// immediate append path, which refuses a dataset with a staged edit on it or
    /// an ancestor. Left unchecked, staging such an edit turns accepted data into
    /// data lost silently in `Drop`. Refusing here moves the failure to the call
    /// that creates the conflict, where there is someone to report it to.
    fn refuse_if_claimed(&self, path: &ObjectPathBuf) -> Result<(), Error> {
        let conflicts = self.appender_claims.iter().any(|c| match &c.path {
            Some(p) => p.overlaps(path),
            None => true,
        });
        if conflicts {
            return Err(Error::EditUnsupported(
                "this dataset has a live buffered appender holding elements only its own flush \
                 can write, and this edit would stop that flush; finish or discard the appender \
                 first",
            ));
        }
        Ok(())
    }

    /// Whether this session is the SWMR writer, whose append rules are a strict
    /// subset of the ordinary ones (see `append_inplace_gathered`).
    pub(crate) fn is_swmr(&self) -> bool {
        self.swmr_mode
    }

    /// This session's file image as one slice, when its backing holds the whole
    /// file in memory; `None` for a file-backed image.
    ///
    /// The slice reflects committed state plus immediate in-place appends, not
    /// edits still staged for `commit`. The owned read-write
    /// [`File`](crate::File) uses it to serve reads by borrowing rather than
    /// copying, and falls back to [`image`](Self::image) when it is absent.
    pub(crate) fn image_slice(&self) -> Option<&[u8]> {
        self.image.as_slice()
    }

    /// A random-access [`Source`] view of this session's file image, for the
    /// parsers the edit engine drives.
    ///
    /// Every read the engine performs against the file goes through this, so the
    /// image needs to be no more than a source of bytes — whole-file mirror or
    /// not — without the parsers knowing which (issue #198).
    pub(crate) fn image(&self) -> &dyn Source {
        self.image.as_ref()
    }

    /// Returns the session's file-image interface, including access to resident bytes.
    pub(crate) fn file_image(&self) -> &dyn FileImage {
        self.image.as_ref()
    }

    /// Borrows the session's original image with its effective address context.
    pub(crate) fn opened_source(&self) -> OpenedSource<'_> {
        OpenedSource::from_engine(self)
    }

    /// Reads issued bytes for the complete image length, leaving pending writes buffered.
    ///
    /// # Errors
    ///
    /// Returns the read or conversion error if the image cannot be read through the disk handle.
    #[cfg(test)]
    pub(crate) fn page_publication_snapshot(&self) -> Result<Vec<u8>, Error> {
        let mut bytes = vec![0; self.image.len().to_usize()?];
        self.image.read_disk_at(0, &mut bytes)?;
        Ok(bytes)
    }

    /// Returns this session's committed superblock with addresses in their encoded coordinates.
    ///
    /// A relocating commit updates the stored root address. A clone taken before that commit
    /// holds the previous root address.
    pub(crate) fn superblock(&self) -> &Superblock {
        self.context.superblock()
    }

    /// Parses the File Space Info message at the current published extension address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the extension's absolute address exceeds `u64`,
    /// and the parser's error if the extension's object header or the message does not parse.
    pub(crate) fn parse_file_space_info(&self) -> Result<Option<FileSpaceInfo>, FormatError> {
        Ok(Self::extension_fsinfo(self.image(), &self.context)?.map(|(_, info)| info))
    }

    /// Returns the address and length widths validated when this session opened the file.
    pub(crate) const fn widths(&self) -> FormatWidths {
        self.context.widths()
    }

    /// Returns the mapping between stored addresses and physical positions in this session.
    pub(crate) const fn addresses(&self) -> AddressContext {
        self.context.addresses()
    }

    /// Reads the shared-message master table if the superblock extension is readable.
    ///
    /// Returns `None` if the extension is absent, cannot be addressed or parsed, or has no
    /// Shared Message Table message.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the table message is malformed or the declared table cannot
    /// be read.
    fn shared_message_table(&self) -> Result<Option<crate::sohm::SohmTable>, Error> {
        let widths = self.context.widths();
        let rel = match self.context.superblock().superblock_extension_address {
            Some(rel) if !rel.is_undefined(widths.offsets) => rel,
            _ => return Ok(None),
        };
        let base = self.context.addresses();
        // Best-effort down to here, and strict past it. A superblock extension
        // this engine cannot parse is one nothing in this crate reads a shared
        // message through either, so it is treated as a file that shares none;
        // once the extension is readable and *says* the file shares messages,
        // failing to read the table is an error, because the screen below is
        // then the difference between a refusal and a corrupted index.
        let Ok(abs) = base.resolve(rel).map(PhysicalAddress::get) else {
            return Ok(None);
        };
        let Ok(summary) = SharedTableSummary::read(
            self.image(),
            AccessMode::ReadWrite,
            PhysicalAddress::new(abs),
            widths,
            base,
        ) else {
            return Ok(None);
        };
        let Some(message) = summary.into_message()? else {
            return Ok(None);
        };
        let opened = self.opened_source();
        Ok(Some(crate::sohm::read_table_from_source(
            &PhysicalSource::new(&opened).stored(),
            &message,
            widths.offsets,
        )?))
    }

    /// Rejects a commit that invalidates a shared-message record's object-header address.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a record refers to an invalidated header. Returns
    /// errors from [`Self::shared_message_table`] or
    /// [`read_index_records_from_source`](crate::sohm::read_index_records_from_source) if the
    /// table or index cannot be read.
    fn screen_shared_message_index(&self, invalidated: &InvalidatedAddresses) -> Result<(), Error> {
        if invalidated.is_empty() {
            return Ok(());
        }
        let Some(table) = self.shared_message_table()? else {
            return Ok(());
        };
        let opened = self.opened_source();
        let stored = PhysicalSource::new(&opened).stored();
        for index in &table.indexes {
            let records =
                crate::sohm::read_index_records_from_source(&stored, index, self.context.widths())?;
            screen_shared_message_records(&records, invalidated)?;
        }
        Ok(())
    }

    /// The on-disk format this session writes, read from the file it opened
    /// rather than chosen fresh, since a commit preserves the superblock version
    /// it found.
    ///
    /// This decides the *contiguous* data-layout message version, where the
    /// older number costs nothing: versions 3 and 4 have identical bodies there,
    /// so a dataset added to a 1.8 file stays readable by a 1.8 library.
    ///
    /// It does not gate chunked storage. A chunked dataset needs the version 4
    /// layout and a 1.10 chunk index whatever the superblock says, so adding one
    /// to an older file does make that file need 1.10 — deliberately, since the
    /// alternative is a version 1 B-tree index this crate does not write, and
    /// refusing instead would take away in-place editing of every file the C
    /// library wrote with its own default bounds. `crates/crosscheck/tests/main/edit.rs`
    /// covers exactly that case (issue #101).
    ///
    /// That is a default, not a verdict: a caller who needs the file to stay
    /// loadable by an old reader asks for it with
    /// [`FileAccessProperties::with_libver_bounds`], which sets
    /// [`libver_ceiling`](Self::libver_ceiling) and turns the addition into a
    /// refusal at commit rather than a silent format bump.
    ///
    /// [`FileAccessProperties::with_libver_bounds`]: crate::FileAccessProperties::with_libver_bounds
    pub(crate) fn libver(&self) -> LibVer {
        LibVer::from_superblock_version(self.context.superblock().version())
    }

    /// Constrain what this session may add, from the fapl's library-version
    /// bounds. See [`libver_ceiling`](Self::libver_ceiling).
    ///
    /// Resolved through [`LibVer::resolve_writable`], the same rule the
    /// whole-file writer applies, so bounds admitting no format this crate
    /// writes are refused here as they are there rather than silently ignored on
    /// the editing path.
    pub(crate) fn set_libver_bounds(
        &mut self,
        bounds: Option<(LibVer, LibVer)>,
    ) -> Result<(), Error> {
        self.libver_ceiling = match bounds {
            Some(_) => Some(LibVer::resolve_writable(bounds).map_err(Error::Format)?),
            None => None,
        };
        Ok(())
    }

    /// Refuse staged content the session's [`libver_ceiling`](Self::libver_ceiling)
    /// cannot express, before the commit writes anything.
    ///
    /// Only chunked storage is at stake: `build_chunked_dataset_oh` writes a
    /// version 4 data-layout message and a 1.10 chunk index unconditionally,
    /// while everything else this session adds is expressible in both formats
    /// (`libver` already picks the contiguous layout version). A filter or an
    /// unlimited dimension arrives as chunked storage, so they are covered here
    /// too.
    fn check_libver_admits<'a>(
        &self,
        datasets: impl IntoIterator<Item = &'a FlatDataset>,
    ) -> Result<(), Error> {
        let Some(ceiling) = self.libver_ceiling else {
            return Ok(());
        };
        if ceiling >= LibVer::V110 {
            return Ok(());
        }
        let chunked = datasets
            .into_iter()
            .any(|fd| fd.chunk_options.is_chunked() || fd.maxshape.is_some());
        if chunked {
            return Err(Error::Format(FormatError::LibverTooOldForContent {
                content: "a chunked, filtered, or resizable dataset",
                needs: LibVer::V110.name(),
                writing: ceiling.name(),
            }));
        }
        Ok(())
    }

    /// Which backend this session resolved to: [`Bounded`] when it reads through
    /// a handle, [`Mirrored`] when it holds a whole-file image.
    ///
    /// [`Bounded`]: EditBacking::Bounded
    /// [`Mirrored`]: EditBacking::Mirrored
    pub(crate) fn edit_backing(&self) -> EditBacking {
        if self.bounded {
            EditBacking::Bounded
        } else {
            EditBacking::Mirrored
        }
    }

    /// Returns the logical size and reusable bytes from cached session counters.
    ///
    /// Includes unused classic aggregate reservations. Computes the total without enumerating
    /// free sections.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the byte sum exceeds [`u64::MAX`].
    pub(crate) fn reusable_space_stats(&self) -> Result<ReusableSpaceStats, FormatError> {
        FreeSpaceSnapshot::new(self).stats()
    }

    /// Returns sorted reusable sections at absolute physical offsets.
    ///
    /// Includes allocator-held reservations, committed changes, and immediate appends. Staged
    /// edits affect the report after [`commit`](Self::commit). On paged files, combines reusable
    /// metadata and raw-data lists while retaining boundaries between page types. Dead and
    /// unclassified space is excluded.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if resolving a stored address, computing a section
    /// end, or summing section lengths overflows. Returns [`FormatError::InvalidFreeSpaceManager`]
    /// if a section length is zero, or [`Error::OverlappingFreeSections`] if sections overlap.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::File;
    /// use hdf5_pure::FileBuilder;
    ///
    /// # let directory = tempfile::tempdir()?;
    /// # let path = directory.path().join("session-space.h5");
    /// # FileBuilder::new().write(&path)?;
    /// let file = File::open_rw(&path)?;
    /// let stats = file.reusable_space_stats()?;
    /// let sections = file.reusable_free_sections()?;
    /// println!(
    ///     "{} bytes on disk, {} reusable in {} free region(s)",
    ///     stats.logical_size(),
    ///     stats.reusable_free_bytes(),
    ///     sections.section_count(),
    /// );
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    #[expect(
        rustdoc::private_doc_tests,
        reason = "rustdoc's test mode collects a private item, so this example is compiled and run by `cargo test --doc`"
    )]
    pub(crate) fn reusable_free_sections(&self) -> Result<FreeSections<PhysicalAddress>, Error> {
        FreeSections::new(
            self.reusable_ranges()
                .map(|range| {
                    let range = range?;
                    FreeSection::new(range.start(), range.length())
                })
                .collect::<Result<Vec<_>, FormatError>>()?,
        )
        .map_err(Error::from)
    }

    /// Returns physical reusable ranges, including allocator-held reservations.
    ///
    /// Reports aggregate spare when reusable allocation is disabled. With reusable allocation
    /// enabled, combines the free lists with aggregate spare, preserving
    /// page-type boundaries.
    /// Resolves stored ranges against the opened image's effective base.
    fn reusable_ranges(&self) -> impl Iterator<Item = Result<FileRange, FormatError>> + '_ {
        FreeSpaceSnapshot::new(self).ranges()
    }

    /// Returns whether the strategy permits immediate reuse outside SWMR writing.
    ///
    /// In persistent sessions, the editor publishes complete ownership before linking references.
    /// Without persistence, the editor reuses storage retired by earlier publications. The PAGE
    /// allocator preserves page classes through [`SessionSpace`]. The SWMR writer allocates
    /// append-only because concurrent readers can retain references to earlier storage.
    fn immediate_reuse_allowed(&self) -> bool {
        !self.swmr_mode && self.space.is_enabled()
    }

    /// Adopt the fapl's `fsync` cadence. Called once, on the funnel every
    /// read-write open passes through, before the session is handed out.
    pub(crate) fn set_sync_policy(&mut self, policy: SyncPolicy) {
        self.sync_policy = policy;
    }

    /// The cadence this session adopted. Nothing in the crate branches on this —
    /// [`barrier`](Self::barrier) does that — but the fapl reaching the engine is
    /// otherwise invisible from outside, and an entry point that forgot to pass
    /// it on would look exactly like one that did.
    #[cfg(test)]
    pub(crate) fn sync_policy(&self) -> SyncPolicy {
        self.sync_policy
    }

    /// Sets write buffering for a test session.
    ///
    /// Crash-prefix tests compare gathered and immediate output for the same workload.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`FileImage::set_write_buffering`].
    #[cfg(test)]
    pub(crate) fn set_write_buffering(&mut self, mode: WriteBuffering) -> Result<(), Error> {
        self.image.set_write_buffering(mode)
    }

    /// Every write this session issued, as `(offset, length)` in the order it
    /// went out. Same reasoning as [`set_write_buffering`](Self::set_write_buffering):
    /// the image is private, and a test that asks whether a publish left the
    /// engine as one write has nowhere else to look.
    #[cfg(test)]
    pub(crate) fn issued_write_order(&self) -> Vec<(u64, u64)> {
        self.image.issued_write_order()
    }

    /// The durability barrier a write path calls for, data and metadata both —
    /// issued unless this session's [`SyncPolicy`] leaves the `fsync` cadence to
    /// the application.
    ///
    /// Every such point routes through here rather than touching the image, so
    /// the policy is honored by construction at sites yet to be written.
    ///
    /// A barrier is an **ordering** point first and a durability point second,
    /// and only the second half is the policy's to skip. The writes gathered
    /// before it are issued here whatever the policy says, because the image
    /// issues gathered writes in address order: leave them gathered across a
    /// barrier and a commit's superblock — address 0 — goes to the disk *before*
    /// the content it names, which is the one order the whole tail is built to
    /// avoid. A write that then fails, or a process that dies mid-flush, would
    /// leave a superblock naming bytes that are not in the file, where before it
    /// left the previous file intact (issue #288).
    ///
    /// [`WriteBuffering::Session`] — an explicit page buffer — is the deliberate
    /// exception, and [`ordering_barrier`](crate::image::FileImage::ordering_barrier)
    /// is what encodes that: it does nothing under that mode, which is precisely
    /// the guarantee such a caller is trading away. What that caller gets back is
    /// a file that *says* so — see [`raise_crash_mark`](Self::raise_crash_mark).
    ///
    /// The teardown barrier is deliberately *not* one of these points:
    /// [`File::close`](crate::File::close) and `FileInner::drop` write after the
    /// last barrier a caller could have asked for, so they take
    /// [`force_sync`](Self::force_sync) instead.
    fn barrier(&mut self) -> Result<(), Error> {
        // Exhaustive on purpose, here and at the two sites below: `SyncPolicy` is
        // sealed to the outside but not to this crate, so a policy added later
        // fails to compile at every durability point until someone decides what
        // it means there.
        match self.sync_policy {
            // `sync_all` issues the gathered writes on its way to the disk.
            SyncPolicy::Always => self.image.sync_all(),
            SyncPolicy::OnClose => self.image.ordering_barrier(),
        }
    }

    /// The data-only counterpart to [`barrier`](Self::barrier), for a write that
    /// does not move end-of-file and so needs no metadata flush. It orders the
    /// gathered writes for the same reason.
    fn barrier_data(&mut self) -> Result<(), Error> {
        barrier_data(self.image.as_mut(), self.sync_policy)
    }

    /// Pad a paged file to a page boundary if its tail page is partially filled,
    /// recording the padding as free space of the tail page's type. A no-op on a
    /// non-paged file or an already-aligned one.
    fn pad_to_page(&mut self) -> Result<(), Error> {
        let eoa = self.end_of_allocation()?;
        if let Some(transition @ PageTransition::Pad { extent, .. }) =
            self.space.page_padding(eoa)?
        {
            let length = extent.len().to_usize()?;
            #[cfg(test)]
            observations::record_zeros(length, length);
            self.append(&vec![0u8; length])?;
            self.space.record_page_padding(transition);
        }
        Ok(())
    }

    /// Writes one block of a persisting commit's tail, paged or flat, at the stored address `addr`
    /// its layout gave it.
    ///
    /// `region` is the tail's span of stored addresses, at the end of the file when the tail is
    /// appended and in space an earlier commit left when the tail reuses it. An appended block's
    /// `addr` resolves to the current end-of-file.
    ///
    /// Checks the block against its reserved span before writing. A reused reservation may
    /// adjoin live storage, so a block outside the span could corrupt a neighboring object.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::SerializationError`] if the block extends outside `region`,
    /// [`FormatError::OffsetOverflow`] if stored or physical allocation geometry exceeds
    /// [`u64::MAX`], and the write's error if the write fails.
    fn write_tail_block(
        &mut self,
        region: FreeExtent,
        addr: StoredAddress,
        bytes: &[u8],
    ) -> Result<(), Error> {
        let size = NonZeroU64::new(region.len())
            .ok_or_else(|| FormatError::Internal("a tail reservation is empty".into()))?;
        let request = BlockRequest::new(
            ReservationRequest::new(size, AllocationClass::Metadata),
            bytes.len() as u64,
            EncodingDependency::Allocation,
        )?;
        let block = PlannedBlock::within(self.context.addresses(), region, addr, request)?;
        self.write_planned_block(&block, bytes)
    }

    fn write_planned_block(&mut self, block: &PlannedBlock, bytes: &[u8]) -> Result<(), Error> {
        let region = block.region(bytes)?;
        let at = region.range().start();
        if at == PhysicalAddress::new(self.image.len()) {
            let written = self.append(region.bytes())?;
            debug_assert_eq!(
                written,
                at.get(),
                "an appended block must land at end-of-file"
            );
            return Ok(());
        }
        self.write_at(at.get(), region.bytes())
    }

    /// Extend the file with zeros up to `target` (>= the current length), used by
    /// the paged tail to pad the final metadata page to its boundary.
    fn pad_zeros_to(&mut self, target: u64) -> Result<(), Error> {
        let len = self.image.len();
        if target > len {
            let pad = (target - len).to_usize()?;
            #[cfg(test)]
            observations::record_zeros(pad, pad);
            self.append(&vec![0u8; pad])?;
        }
        debug_assert_eq!(self.image.len(), target);
        Ok(())
    }

    /// Append `bytes` at end-of-file, returning the absolute address they were
    /// written at.
    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        self.image.append(bytes)
    }

    /// Overwrites bytes in place at `offset`. The caller guarantees the range
    /// already exists.
    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        self.image.write_at(offset, bytes)
    }

    /// Ensure the next allocation begins in a page holding page type `ty`, on a
    /// paged file. A no-op on the common non-paged file.
    ///
    /// A paged file never mixes metadata and raw data within one page, so when the
    /// tail page holds the *other* type and is only partially filled it is first
    /// padded to a page boundary, the padding being recorded as free space of the
    /// outgoing type.
    ///
    /// Call this **before** reading the image's end-of-file ([`Source::len`]) to compute an
    /// address that will be embedded in the bytes being built: several callers
    /// (the chunk blob, the extensible-array index, the dense-attribute blob)
    /// build content whose interior addresses assume it lands at the current
    /// end-of-file, and padding inserted after that read would shift the landing
    /// address out from under them.
    fn begin_page(&mut self, ty: PageType) -> Result<(), Error> {
        let eoa = self.end_of_allocation()?;
        let Self { image, space, .. } = self;
        let Some(transition) = space.plan_transition(eoa, ty)? else {
            return Ok(());
        };
        if let PageTransition::Pad { extent, .. } = transition {
            let length = extent.len().to_usize()?;
            #[cfg(test)]
            observations::record_zeros(length, length);
            image.append(&vec![0u8; length])?;
        }
        space.record_transition(ty, transition);
        Ok(())
    }

    /// Writes `bytes` for `class`, reusing a free region where one fits.
    ///
    /// Equivalent to [`reserve`](Self::reserve) followed by [`place`](Self::place),
    /// for the callers whose bytes are already built.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if placement is invalid. Returns the write error if writing fails.
    fn alloc_or_append_typed(
        &mut self,
        bytes: &[u8],
        class: AllocationClass,
    ) -> Result<u64, Error> {
        let at = self.reserve(bytes.len() as u64, class)?;
        self.place(at, bytes)
    }

    /// Reserves `len` bytes for `class` according to the active file-space strategy.
    ///
    /// The returned
    /// [`Placement`] must be handed to [`place`](Self::place) with exactly `len`
    /// bytes.
    ///
    /// Under [`FileSpaceStrategy::Aggr`], the reservation comes from the matching metadata
    /// or raw aggregate. Other strategies use reusable space or append at EOA.
    ///
    /// A relocatable object is sized before its addresses are assigned. The caller then
    /// builds its bytes for the returned address, as in
    /// [`place_relocatable`](Self::place_relocatable).
    ///
    /// Reuse draws from space freed by earlier publications or loaded from the persisted
    /// managers at open. Regions vacated by the current commit become reusable after its root
    /// publication. A failed write before publication therefore leaves the live tree intact.
    ///
    /// Persistent PAGE reservations use a matching page type or wholly free pages through
    /// [`SessionSpace::allocate`]. Reused reservations leave the tail page unchanged.
    /// Non-persistent PAGE sessions append at or beyond
    /// [`fresh_page_boundary`](Self::fresh_page_boundary).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if address resolution, aggregate planning, or PAGE transition
    /// planning fails. Returns [`Error::AllocationPublicationUncertain`] if a nonempty reservation
    /// follows uncertain publication. Returns the write error if writing PAGE padding fails.
    fn reserve(&mut self, len: u64, class: AllocationClass) -> Result<Placement, Error> {
        self.reserve_with_page_type(len, class, class.into())
    }

    /// Reserves a physical allocation with an explicit PAGE placement type.
    ///
    /// Array chunk indexes retain metadata intent while sharing raw-data pages.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if address resolution, aggregate planning, or PAGE transition
    /// planning fails. Returns [`Error::AllocationPublicationUncertain`] if a nonempty reservation
    /// follows uncertain publication. Returns the write error if writing PAGE padding fails.
    fn reserve_with_page_type(
        &mut self,
        len: u64,
        class: AllocationClass,
        page_type: PageType,
    ) -> Result<Placement, Error> {
        if let Some(size) = NonZeroU64::new(len) {
            return self
                .reserve_request(ReservationRequest::with_page_type(size, class, page_type));
        }
        self.begin_page(page_type)?;
        Ok(Placement::Appended {
            addr: self.image.len(),
            len,
        })
    }

    /// Reserves `len` reusable bytes for `class` and returns their stored address,
    /// or `None` when no single region is large enough or PAGE persistence is disabled.
    ///
    /// A non-paged file keeps one list for the whole file. A paged file keeps one
    /// per page type and uses the list matching `class`: handing a
    /// metadata hole to raw data (or the reverse) would mix the two within a page,
    /// which is the single invariant the paged strategy exists to hold. A page
    /// holding nothing at all is the exception, and [`SessionSpace::allocate`] is where it is
    /// spent. [`SessionSpace::Disabled`] draws nothing.
    fn alloc_free(&mut self, len: u64, class: AllocationClass) -> Option<StoredAddress> {
        if self.fresh_page_boundary.is_some() {
            return None;
        }
        self.space.allocate(len, class).map(|extent| extent.start())
    }

    /// Hands the stored range `[addr, addr + len)`, drawn from [`alloc_free`](Self::alloc_free)
    /// with [`AllocationClass::Raw`], back to the list it came from.
    ///
    /// A caller that rejects a placement after the draw calls this, since the region would
    /// otherwise stay out of the free lists for the rest of the session with nothing written into
    /// it.
    ///
    /// A paged file takes it back into the raw list whichever list served it: a
    /// whole page [`SessionSpace::allocate`] claimed from the metadata side for
    /// raw data is raw from the moment it is claimed, and that call already
    /// returns its alignment tail there. A page that ends up wholly free is
    /// promoted out of either list by
    /// whole-page promotion as usual.
    fn release_raw_alloc(&mut self, addr: StoredAddress, len: u64) {
        let Some(extent) = FreeExtent::new(addr, len) else {
            return;
        };
        self.space.return_allocation(extent, AllocationClass::Raw);
    }

    /// Write `bytes` at the address [`reserve`](Self::reserve) handed out,
    /// returning that address.
    ///
    /// The length is checked rather than asserted: a reused region is sized to the
    /// reservation and is followed by live bytes, so writing more than was reserved
    /// would silently destroy a neighboring object. That makes it the one internal
    /// miscount worth paying a comparison to catch in every build.
    fn place(&mut self, at: Placement, bytes: &[u8]) -> Result<u64, Error> {
        if bytes.len() as u64 != at.len() {
            return Err(Error::Format(FormatError::SerializationError(format!(
                "a placement reserved {} bytes but built {}",
                at.len(),
                bytes.len()
            ))));
        }
        match at {
            Placement::Aggregated { addr, end, .. } => {
                AggregateSpace::place(self.image.as_mut(), addr, bytes, end)?;
                Ok(addr)
            }
            Placement::Reused { addr, .. } => {
                self.write_at(addr, bytes)?;
                Ok(addr)
            }
            Placement::Appended { addr, .. } => {
                let written = self.append(bytes)?;
                debug_assert_eq!(
                    written, addr,
                    "an appended placement must land at end-of-file"
                );
                Ok(written)
            }
        }
    }

    /// Reserves `len` bytes, builds content for that address, and writes it.
    ///
    /// `build` receives a stored address relative to the file's base. The length must match the
    /// reservation exactly. Returns the physical address and the additional value produced by
    /// `build`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] for invalid address conversion or a placement-length mismatch.
    /// Returns allocation, I/O, and `build` errors.
    fn place_relocatable<T>(
        &mut self,
        len: u64,
        class: AllocationClass,
        build: impl FnOnce(StoredAddress) -> Result<(Vec<u8>, T), Error>,
    ) -> Result<(u64, T), Error> {
        let at = self.reserve(len, class)?;
        let (bytes, extra) = build(
            self.context
                .addresses()
                .relative(PhysicalAddress::new(at.address()))?,
        )?;
        let addr = self.place(at, &bytes)?;
        Ok((addr, extra))
    }

    /// Places heap collections and returns their base-relative addresses in the same order.
    ///
    /// Collections are allocated independently and may occupy noncontiguous regions.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if allocation geometry or base-relative address conversion is
    /// invalid. Returns [`Error::Io`] if writing a collection or allocation padding fails.
    // `H5MF_aggr_vfd_alloc` in `H5MFaggr.c`, HDF5 2.2.0, assigns `H5FD_MEM_GHEAP`
    // to the raw-data aggregator.
    fn place_vl_collections(&mut self, collections: &[Vec<u8>]) -> Result<Vec<u64>, Error> {
        collections
            .iter()
            .map(|collection| {
                let addr = self.alloc_or_append_typed(collection, AllocationClass::Raw)?;
                Ok(self
                    .context
                    .addresses()
                    .relative(PhysicalAddress::new(addr))?
                    .get())
            })
            .collect()
    }

    /// Forget every heap collection this session recorded as exclusively one
    /// dataset's, because `staged` contains an edit that could name one of them
    /// a second time (issue #321).
    ///
    /// The provenance in [`vl_overwrite_heaps`](Self::vl_overwrite_heaps) says a
    /// collection was named once *when it was placed*. Two staged edits can
    /// falsify that afterwards, and both do it by re-emitting element bytes
    /// somebody else's references live in:
    ///
    /// - a **copy**, in-file or cross-file, which duplicates a dataset's element
    ///   references verbatim. This is not hypothetical: an in-file copy of a
    ///   variable-length dataset is exactly how two datasets come to name one
    ///   collection.
    /// - a **raw-bytes write** over a datatype that reaches a heap address
    ///   ([`datatype::datatype_holds_file_address`]). `with_raw_data` hands the engine
    ///   element bytes it does not interpret, so a caller that read them from
    ///   one dataset and wrote them to another has aliased whatever they named —
    ///   the same unscreened door issue #317 recorded for object references.
    ///
    /// Wholesale rather than per-path: what a copy's source references is not
    /// known without decoding it, and a proof that has to be argued per edit is
    /// the kind that goes stale. Reclaim is an optimization, and giving all of
    /// it up on a rare edit costs a leak, where keeping one entry too long costs
    /// a freed collection somebody still reads.
    fn invalidate_heap_provenance(&mut self, staged: &StagedEdits) {
        if self.vl_overwrite_heaps.is_empty() {
            return;
        }
        let aliasing_edit = !staged.copies().is_empty()
            || !staged.cross_copies().is_empty()
            // A dataset staged with element bytes of its own (`with_raw_data`)
            // rather than staging its strings here, whether it is being created
            // or overwritten.
            || staged.datasets()
                .iter()
                .any(|(_, fd)| {
                    fd.storage.vl_string_staging().is_none()
                        && datatype::datatype_holds_file_address(&fd.dt)
                })
            || staged.writes().iter().any(|(_, values)| {
                values.vl_string_staging().is_none() && datatype::datatype_holds_file_address(&values.dt)
            });
        if aliasing_edit {
            self.vl_overwrite_heaps.clear();
            return;
        }
        // A deleted dataset's record would otherwise outlive it and be applied
        // to whatever later takes its path.
        for path in staged.deletes() {
            self.vl_overwrite_heaps
                .retain(|recorded, _| !recorded.overlaps(path));
        }
    }
}

/// Returns whether the physical ranges are nonempty, disjoint, and end at or before `eof`.
///
/// Sorts by start address after checking the endpoints. Adjacent ranges are allowed.
fn spans_disjoint_in_bounds(spans: &mut [FileRange], eof: u64) -> bool {
    if spans
        .iter()
        .any(|range| range.length() == 0 || range.end() > PhysicalAddress::new(eof))
    {
        return false;
    }
    spans.sort_unstable_by_key(|range| range.start());
    spans.windows(2).all(|pair| pair[0].precedes(pair[1]))
}

/// Retains nonempty, disjoint physical ranges ending at or before `eof`.
///
/// Sorts by start address and drops each range overlapping one already retained.
fn retain_disjoint_in_bounds(spans: &mut Vec<(FileRange, FreeClass)>, eof: u64) {
    spans.retain(|(range, _)| range.length() > 0 && range.end() <= PhysicalAddress::new(eof));
    spans.sort_unstable_by_key(|(range, _)| range.start());
    let mut previous: Option<FileRange> = None;
    spans.retain(|&(range, _)| {
        if previous.is_none_or(|previous| previous.precedes(range)) {
            previous = Some(range);
            true
        } else {
            false
        }
    });
}

/// Returns whether every filter in `pipeline` has a known encoder.
///
/// An in-place overwrite checks this classification before writing chunk data.
pub(crate) fn pipeline_reencodable(pipeline: &FilterPipeline) -> bool {
    hdf5_pure_filter::__private::filters_reencodable(&crate::filter_pipeline::FilterStepsRef::new(
        pipeline,
    ))
}

/// Returns whether the pipeline's filters guarantee preservation of element values.
///
/// Appending to a partial trailing chunk decodes committed values before writing
/// the expanded chunk. Lossy filters may change those values through this operation.
pub(crate) fn pipeline_lossless(pipeline: &FilterPipeline) -> bool {
    hdf5_pure_filter::__private::filters_lossless(&crate::filter_pipeline::FilterStepsRef::new(
        pipeline,
    ))
}

/// The shared rejection for re-encoding a lossy partial trailing chunk.
pub(crate) const LOSSY_TAIL_REFUSAL: &str = "this dataset's filter pipeline is lossy (ZFP, or float D-scale scale-offset), and its \
     length is not a whole multiple of the chunk length: growing that trailing chunk would \
     decode and re-encode values that are already committed, changing them. Append whole \
     chunks from a chunk-aligned length instead";

/// Reject a commit that removes or moves an object header the file's
/// shared-message (SOHM) index names.
///
/// A heap-stored record refers to no object, so it is not screened here: what a
/// commit does to it is a *reference count* that goes stale, which the engine
/// rejects at the point it would change one (see [`attributes::SHARED_ATTRIBUTE_MESSAGE`]).
/// This screens the other record shape, whose address would dangle.
fn screen_shared_message_records(
    records: &[crate::sohm::SohmRecord],
    invalidated: &InvalidatedAddresses,
) -> Result<(), Error> {
    for record in records {
        if let crate::sohm::SohmLocation::ObjectHeader { address, .. } = record.location
            && invalidated.refusal(address).is_some()
        {
            return Err(Error::EditUnsupported(
                SHARED_MESSAGE_INDEX_NAMES_A_MOVED_OBJECT,
            ));
        }
    }
    Ok(())
}

/// Reject a staged dataset whose element bytes already hold *resolved* object
/// references naming space this commit reclaims (issue #317).
///
/// This is the address-side half of the rule
/// [`WriteEngine::resolve_reference_target`] enforces on the path side. A target
/// named as a path is screened there, by name. A target supplied as an address -
/// `DatasetBuilder::with_reference_data`, or `with_raw_data` over a datatype that
/// holds a reference - never reaches that function at all, and before this screen
/// existed it was written straight through to disk.
///
/// Element bytes still carrying placeholders are unaffected: an unresolved slot
/// holds zero, which [`InvalidatedAddresses::refusal`] passes, and the target
/// that replaces it is screened by `resolve_reference_target` - by name
/// for a path, by address for a raw one. So this runs over every staged
/// dataset's `raw` without inspecting which builder filled it: the rule is about the
/// bytes, not the door they came through.
fn screen_resolved_references(
    dt: &Datatype,
    raw: &[u8],
    invalidated: &InvalidatedAddresses,
) -> Result<(), Error> {
    if !datatype::datatype_holds_object_address(dt) {
        return Ok(());
    }
    // The datatype declares an object reference, so the walker must find one.
    // Two ways it does not: slots that do not fit the element size (`None`), and
    // a reference `embedded_reference_slots` does not map - it locates the
    // 8-byte object reference only, and reports any other width as no slots at
    // all. Both mean the addresses cannot be read, and an empty list taken for
    // "nothing to check" would wave through exactly the datatype
    // `datatype_holds_object_address` had just recognised. Neither shape is
    // one this crate builds. Reject references past a screen
    // that could not see them.
    let Some(slots) = datatype::embedded_reference_slots(dt).filter(|slots| !slots.is_empty())
    else {
        // Unconditional, unlike the per-address checks below. Those reject an
        // address that lands somewhere this commit vacates. This one cannot read
        // the addresses at all, so there is nothing to compare against and no
        // amount of screening would help. Every commit that reaches here rebuilds
        // at least the root group, so `for_supplied.moved` is never empty and
        // gating this would have changed nothing anyway. A commit whose only
        // staged edit is a same-length in-place overwrite never reaches here at
        // all - it takes the fast path above, which vacates nothing. An in-file
        // copy reaches this only beside a deletion, because it is screened
        // against `for_copied`, whose `moved` is empty by construction.
        //
        // The `filter` is belt-and-braces: `embedded_reference_slots` already
        // reports an unaddressable reference as `None`, and reading an empty list as "nothing to check" is exactly the
        // hole this screen was found to have.
        return Err(Error::EditUnsupported(
            "a reference this commit writes sits in a datatype whose addresses this screen \
             cannot read — a width other than eight, a dataset-region reference, a \
             variable length of them, or a compound holding one beside a reference it can \
             read — so it cannot be checked against what this commit vacates; supply an \
             8-byte object reference (`with_reference_data`)",
        ));
    };
    // Non-empty slots mean the datatype has room for at least one 8-byte
    // address, so the element size is at least 8.
    for (_, stored) in datatype::stored_object_references(raw, dt.type_size() as usize, &slots) {
        if let Some(refusal) = invalidated.refusal(StoredAddress::new(stored)) {
            return Err(Error::EditUnsupported(refusal));
        }
    }
    Ok(())
}

/// The engine as the target a [`reference_patch::Plan`] is applied to
/// (issue #324).
///
/// Deliberately the whole engine rather than the image alone: an object header
/// is republished as one checksummed write, so applying a plan reads as well as
/// writes, and both have to go through the same image the rest of the commit
/// used — the same write-gathering, the same pending-write overlay, the same
/// end-of-file bound.
///
/// [`reference_patch::Plan`]: crate::reference_patch::Plan
impl crate::reference_patch::PatchTarget for WriteEngine {
    fn read(&self, at: u64, len: usize) -> Result<Vec<u8>, Error> {
        self.image().read_exact_at(at, len).map_err(Error::Format)
    }
    fn write(&mut self, at: u64, bytes: &[u8]) -> Result<(), Error> {
        self.image.write_at(at, bytes)
    }
}

#[cfg(test)]
mod tests;
