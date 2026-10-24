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
//! - A version 2/3 group on an edited path stores its links compactly (not in a
//!   dense fractal heap); headers split across continuation chunks (as the
//!   reference C library often writes) are collapsed into a single chunk when
//!   rewritten. A version 1 group is converted to a compact-link v2 header,
//!   carrying its links and attributes over (other group messages — symbol
//!   table, modification time — are dropped); an attribute it cannot reproduce
//!   is refused.
//! - An object that tracks **attribute creation order** (`track_order=True`,
//!   and everything netCDF-4 writes) is edited like any other: its 6-byte
//!   message records are walked and re-emitted, a new attribute takes the
//!   object's next creation index, an overwrite keeps the one it had, and a
//!   deletion leaves a gap rather than renumbering. A group that tracks **link**
//!   creation order works the same way: a link added to it takes the next index
//!   from the running maximum its Link Info message records, and a deletion
//!   leaves a gap there too. Only an addition that would send such a group's
//!   links *dense* is refused, since the creation-order B-tree that indexes them
//!   is not written here.
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
//!   target the fast path cannot handle — a userblock or pre-v2 file, an
//!   unallocated index, a non-Extensible-Array or multi-hard-link dataset, a
//!   a filtered dataset sitting on a partial trailing chunk — is refused with
//!   [`Error::AppendInPlaceUnsupported`]; use the staged `append_dataset` instead.
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
//! free space before these aggregates. Persistent sessions exclude active aggregate spare from
//! their managers and return surrendered spare through a manager rewrite. SWMR sessions under
//! `FSM_AGGR` and PAGE sessions use no classic aggregates.
//! The reuse is crash-safe: it only ever overwrites space
//! freed by an *earlier*, already-durable commit (never space the current commit
//! is mid-way through freeing), and truncation happens only after the superblock
//! recording the smaller end-of-file is itself durable. A commit that fails
//! before its repoint gives back what it drew, so a failed attempt costs the
//! session nothing.
//!
//! Both write paths draw on that list, through allocators of their own. The
//! immediate [`Dataset::append`](crate::Dataset::append) allocates its chunks and
//! index blocks through [`crate::chunk_index_inplace::Store::allocate`], which
//! [`WriteEngine::immediate_reuse_allowed`] gates: an append has no superblock
//! repoint to publish its allocations with, so it reuses only where the space it
//! overwrites is dead on the disk as it stands (issue #349). On a file that
//! **persists** its managers a hole in the list is one the disk still advertises,
//! so the append does not spend it directly: it takes a batch out of the
//! published managers first
//! ([`WriteEngine::reserve_for_immediate_append`], issue #387) and spends only
//! that, which makes the on-disk record true of every byte it writes at every
//! instant. What the session does not spend goes back before the managers are
//! next rewritten; a session that dies in between strands the remainder, which is
//! accounted to nobody and so can never be handed out twice. The SWMR writer
//! reuses nothing at all, for a reason publication does not reach: its readers
//! may still be inside the region. Everything
//! a *commit* places goes through the other allocator
//! ([`WriteEngine::reserve`] and [`WriteEngine::place`]), including a chunked
//! dataset's data region and a dense attribute heap. Those carry addresses of
//! their own, so they are sized before they are placed and then built for the
//! address they got — which is why they can land in a freed region at all
//! (issue #261). A paged file (`H5F_FSPACE_STRATEGY_PAGE`) draws only from free
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
//! manager headers and complete allocated section lists. Logical sections within the heap are
//! excluded from file-space reclamation. The editor reclaims a global heap collection that only
//! the objects a delete removes refer to, and keeps every collection when it cannot read each
//! object that can refer to one. Storage whose allocations cannot be enumerated exhaustively, such
//! as a filtered fractal heap with a filter this crate does not decode or with huge objects, is
//! left as dead bytes. This avoids freeing a region that may still be in use. Under-reclaiming only
//! wastes space. Over-reclaiming could corrupt the file.
//!
//! On a paged file that conservatism extends to anything whose *page type* is not
//! established. A file another writer produced records free space this one cannot
//! place — the reference library's generic-large manager holds metadata and raw
//! alike, and it puts a chunk index among its metadata where this crate puts one
//! beside the chunk data. Such space is kept and written back where it was found,
//! but never handed to an allocation
//! (unclassified PAGE space, [`WriteEngine::index_is_provably_raw`]), because
//! placing a byte of the wrong kind in a page is the one thing paging exists to
//! prevent and no reader would report it.
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

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::PhysicalAddress;
use hdf5_pure_format::__private::FreeSpaceManagerHeader;
use hdf5_pure_space::__private as free_space;
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::{
    Extent as FreeExtent, FreeClass, FreeList, ManagerKind, ManagerSections, PageTransition,
    PageType, SessionSpace, SessionSpaceSnapshot,
};

use crate::SuperblockStatusFlags;
use crate::SuperblockVersion;
use crate::UserblockSize;
use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::convert::Narrow;
use crate::datatype::{self, Datatype};
use crate::error::{Error, FormatError};
use crate::file_create_properties::FileCreateProperties;
use crate::file_lock::{self, FileLocking};
use crate::file_space_info;
use crate::file_space_info::DEFAULT_THRESHOLD;
use crate::file_space_info::FileSpaceInfo;
use crate::file_space_info::FileSpacePageSize;
use crate::file_space_info::FileSpaceStrategy;
use crate::file_space_info::NUM_FILE_FSM_MANAGERS;
use crate::file_writer::{LENGTH_WIDTH, OFFSET_WIDTH};
use crate::filter_pipeline::FilterPipeline;
use crate::free_space_manager;
use crate::free_space_manager::PersistedSections;
use crate::image::{FileImage, HandleImage, MirrorImage, WriteBuffering};
use crate::libver::LibVer;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::object_path::{LinkNameBuf, ObjectPathBuf};
#[cfg(any(test, feature = "__page-publication-crosscheck"))]
use crate::page_publication_failure::{PagePublicationFailure, PublicationFailureImage};
use crate::reader::FileAccessProperties;
use crate::source::SourceMetadata;
use crate::source::{BaseOffsetSource, MetadataCacheConfig, Source};
use crate::superblock;
use crate::superblock::LocatedSuperblock;
use crate::superblock::Superblock;
use crate::type_builders::ObjectRefTarget;
use crate::width::FormatWidths;

use self::aggregation::AggregateSpace;
use self::attributes::{AttrOp, EditedAttrs};
use self::copy::CopyTree;
use self::dataset::{FlatDataset, MovingWrite, WritePlan};
use self::group::LinkCreationOrder;
pub(crate) use self::object_header::OhRegion;
use self::staging::{StagedEdits, StagedScope};

pub(crate) use self::staging::{StagedChild, StagedKind, StagedMeta, StagedObject};

mod aggregation;
mod append;
pub(crate) mod attributes;
mod copy;
mod dataset;
mod global_heap_liveness;
#[cfg(test)]
mod global_heap_liveness_tests;
mod group;
pub(crate) mod object_header;
#[cfg(test)]
mod ownership_tests;
#[cfg(test)]
mod page_feasibility;
#[cfg(test)]
mod page_header_reclamation;
#[cfg(test)]
mod page_manager_tail;
mod paged_persistence;
mod reclaim;
mod root_reclamation;
mod staging;
mod tail_recycling;

pub use self::append::AppendBuilder;
pub(crate) use self::append::{AppendGeometry, AppendTarget, LocatedState};
#[cfg(test)]
pub(crate) use self::append::{datatype_is_raw_appendable, locate_dataset_state};
pub use global_heap_liveness::GlobalHeapLiveness;

/// An undefined on-disk address (all bits set), HDF5's "no address" sentinel.
const UNDEF: u64 = u64::MAX;

/// The most free space an immediate in-place append takes out of a persisting
/// file's on-disk managers in one draw, when the allocation in front of it needs
/// less (issue #387).
///
/// Every draw costs a manager rewrite and a superblock repoint — a few kilobytes
/// and two ordering barriers — so drawing per allocation would put a commit
/// between every pair of chunks and make reuse cost more than the growth it
/// avoids. A draw therefore takes as much as this of whatever holes can each hold
/// the allocation, largest first, and a megabyte covers sixteen chunks at the
/// 64 KiB scale the C library's own chunk-cache default is sized for, which puts
/// the rewrite well outside the per-chunk path.
///
/// It is a cap, not a floor. A file whose holes are all smaller than this still
/// has them drawn, one rewrite for however many of them fit under the cap: a
/// floor here made every persisting file whose deleted objects were under a
/// megabyte grow without bound, since no hole it left was ever spent
/// (issue #413). What a draw refuses is only a hole too small for the allocation
/// in hand, which nothing could be placed in. Every hole exists because a commit
/// freed it, so the rewrites spent reusing them are bounded by the commits the
/// caller already paid for.
///
/// It bounds one draw, not a session's total. A fresh draw is made whenever the
/// largest reserved region is smaller than the allocation in hand, and the
/// remainders of earlier draws are kept and still handed out, so what an abrupt
/// end strands is the sum of every remainder still held: each one smaller than
/// the allocation that forced the next draw, and one more of them possible per
/// draw. Larger would amortize further and strand more; this is the point where
/// the rewrite is already rare enough that making it rarer buys little.
const APPEND_RESERVE_BYTES: u64 = 1 << 20;

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
    /// Absolute offset of the superblock signature in the file.
    sb_sig_off: u64,
    /// The committed superblock with addresses in their encoded coordinate systems.
    superblock: Superblock,
    /// The address and length widths validated while parsing the superblock.
    widths: FormatWidths,
    /// The mapping from stored metadata addresses to physical positions in this image.
    addresses: AddressContext,
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
    /// Classic aggregate windows shared by staged edits and immediate appends under
    /// [`FileSpaceStrategy::Aggr`] or [`FileSpaceStrategy::FsmAggr`].
    ///
    /// An ordinary read-write session without a recorded strategy uses FSM_AGGR with persistence
    /// disabled.
    aggregation: Option<AggregateSpace>,
    /// Space this session has taken *out* of the on-disk free-space managers so
    /// that an immediate in-place append may spend it (issue #387).
    ///
    /// Only a file that persists its free space uses this. Such a file records
    /// its holes on disk, and an append has no superblock repoint of its own to
    /// publish an allocation with — so an append that simply drew from
    /// the active [`SessionSpace`] would leave a manager advertising bytes a live chunk
    /// now occupies, through a clean close as much as a crash. The reserve is
    /// the answer: [`reserve_for_immediate_append`](Self::reserve_for_immediate_append)
    /// moves a batch of bytes out of the session allocator and rewrites the managers
    /// *without* them, under
    /// the same crash-atomic superblock repoint every persisting commit uses.
    /// Only then may an append write there, because by then no durable record
    /// calls those bytes free.
    ///
    /// What is left unspent goes back into the session allocator before the managers are next
    /// rewritten, which is [`release_reserve`](Self::release_reserve)
    /// at the next commit and at close. A session that dies in between strands
    /// the unspent remainder: it is accounted to nobody, so it is never handed
    /// out twice, and a later `H5repack` or whole-file rewrite recovers it. That
    /// is the same trade the module header records for
    /// [`finalize_persist`](Self::finalize_persist), on a surface bounded per
    /// draw — see [`APPEND_RESERVE_BYTES`], which is the most one draw takes,
    /// and note that this list holds the unspent remainder of every draw the
    /// session has made, not just the last.
    ///
    /// Empty for a non-persisting session, which reuses from [`space`](Self::space) directly.
    /// Its allocator state is in memory alone, so nothing on the disk claims those bytes
    /// are free and nothing has to be published before they are spent.
    ///
    /// Its extents are stored addresses, as in [`space`](Self::space).
    reserved: FreeList,
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
    /// Global heap collections **this session placed** for a variable-length
    /// value overwrite, by the overwritten dataset's path, as absolute
    /// `(address, length)` extents.
    ///
    /// This is what lets the *next* overwrite of the same dataset reclaim the
    /// strings the previous one left, instead of growing the file by its whole
    /// payload on every commit forever (issue #321). Outside a delete, nothing
    /// else in the editor reclaims a collection, and the reason is sound: a
    /// collection can be shared between objects, and **nothing in the format
    /// proves otherwise**.
    /// The per-object reference count does not: an in-file
    /// [`copy`](Self::copy) of a variable-length dataset re-emits its
    /// element references verbatim, so two datasets name one collection with
    /// every object's count still 1 — and files with that shape are already in
    /// the wild.
    ///
    /// Provenance is therefore the whole proof: these collections were placed by
    /// this session, for this dataset, and were named by nothing else at the
    /// moment they were written. `invalidate_heap_provenance` is what keeps that
    /// true afterwards — every entry is dropped as soon as this session does
    /// anything that could name one of them a second time.
    ///
    /// The collections a dataset held when the session *opened* are never in
    /// here, and an overwrite does not reclaim them: whatever wrote the file
    /// placed them. `repack` is what recovers those.
    vl_overwrite_heaps: HashMap<ObjectPathBuf, Vec<(u64, u64)>>,
    /// Heap collections superseded by a value overwrite this commit is applying,
    /// freed once the commit's superblock repoint has landed — never before, so
    /// a mid-commit crash leaves the prior root reaching bytes that are still
    /// there. Drained into the commit's `to_free` list; cleared at commit entry,
    /// so an attempt that fails partway frees nothing on the next one.
    superseded_heaps: Vec<(u64, u64)>,
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
    /// and checks unspent append reserves and aggregate state before rewriting the managers.
    /// This field is used when [`persist`](Self::persist) is `Some`.
    fsm_len: u64,
    /// Whether a publisher attempted the superblock write that prevents commit rollback.
    ///
    /// The publisher sets this before writing because an I/O error can follow a partial or
    /// complete write. Once publication is attempted, restoring values or returning allocations
    /// to free lists could overwrite live data.
    ///
    /// Non-persistent PAGE sessions retain this state after a failed publication. Before taking
    /// another staged batch, the editor drains pending writes and verifies the issued superblock.
    /// Explicit synchronization performs the same verification after its durable barrier.
    /// Successful verification clears this state, including when the verified root is unchanged.
    /// Other publishers clear it on entry to their publication attempt.
    publish_attempted: bool,
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

/// Holds allocator state and heap allocation records from before a commit's apply loop.
///
/// A commit reserves bytes before publishing the new root. If it fails before attempting
/// publication, restoring this snapshot makes those bytes available again. Once publication
/// is attempted, the live tree may reference them, so [`WriteEngine::publish_attempted`]
/// prevents restoration.
struct FreeSnapshot {
    space: SessionSpaceSnapshot,
    aggregation: Option<AggregateSpace>,
    /// The append reserve, which the persisting commit tail drains back into the
    /// lists above before it rewrites the managers. A commit that then fails
    /// before publishing leaves those managers as they were — still not listing
    /// the reserve — so restoring it is what keeps the session's picture of the
    /// disk exact (issue #387).
    reserved: FreeList,
    /// The heap-collection provenance, rolled back with the free lists.
    ///
    /// [`resolve_overwrite_bytes`](WriteEngine::resolve_overwrite_bytes) records
    /// a placement in the *apply* phase, so a commit that fails after that and
    /// before its repoint hands the space back to the free list while leaving a
    /// record naming it. The next overwrite of that path would then free a
    /// region already handed out — the one piece of engine state naming file
    /// addresses that this rollback used to miss (issue #321).
    vl_overwrite_heaps: HashMap<ObjectPathBuf, Vec<(u64, u64)>>,
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

/// When a read-write session forces its writes to durable storage — who owns the
/// `fsync` cadence, this crate or the application.
///
/// This is *not* about whether a write reaches the file. Every commit and every
/// append has gone to the operating system by the time it returns, whatever this
/// policy says, so a committed edit is visible to any other process on the same
/// machine and survives this process crashing. What it governs is the `fsync` on
/// top of that, which is what makes those bytes survive the *machine* losing
/// power.
///
/// A session does gather the many small writes *inside* one such operation and
/// issue them a page at a time (issue #288), and an explicit
/// [page buffer](crate::FileAccessProperties::with_page_buffer_size) extends that
/// across operations — trading exactly the guarantee in the paragraph above,
/// which is why it is opt-in and this is not.
///
/// libhdf5 does not `fsync` on its normal path either: the
/// default `sec2` driver installs no flush callback at all, so `H5Fflush` drains
/// libhdf5's own caches with `write` and stops there, leaving power-loss
/// durability to the application. [`OnClose`](Self::OnClose) is that behavior;
/// [`Always`](Self::Always), the default here, is the stronger one.
///
/// The whole-file writer ([`FileBuilder`](crate::FileBuilder)) and
/// [`repack`](crate::repack()) are outside this: they never `fsync` under either
/// policy, since each writes a file and hands it over rather than holding an
/// editing session.
///
/// Sealed: a future policy (syncing on a timer, or once per N bytes) must not be
/// a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SyncPolicy {
    /// Force durability at every point the write paths define one: after each
    /// immediate [`Dataset::append`](crate::Dataset::append) batch, at each
    /// ordering barrier inside a [`File::commit`](crate::File::commit), and when
    /// the file is closed or dropped.
    ///
    /// This is what makes an append crash-atomic and a commit all-or-nothing
    /// against power loss: the barriers order the appended data before the
    /// superblock repoint that publishes it, so an interrupted write leaves the
    /// previous state rather than a torn one.
    #[default]
    Always,
    /// No `fsync` during the session; one when the file is finished. Nothing an
    /// append or a commit does is forced to durable storage — writes still reach
    /// the operating system immediately, so another process on the same machine
    /// sees them and a crash of *this* process loses nothing — and
    /// [`File::close`](crate::File::close), or dropping the last handle, issues
    /// a single barrier at the end. [`File::sync`](crate::File::sync) adds a
    /// checkpoint wherever the application wants one.
    ///
    /// One setting suspends the "reach the operating system immediately" half,
    /// and it is the one that requires this policy:
    /// [`with_page_buffer_size`](crate::FileAccessProperties::with_page_buffer_size)
    /// holds dirty pages in this process until the budget is spent, a
    /// [`File::sync`](crate::File::sync) is issued, or the session closes. Off by
    /// default, and it marks the file on disk for its lifetime so a session that
    /// dies holding pages leaves one every reader refuses.
    ///
    /// The terminal barrier is not an exception grafted onto "never sync": it is
    /// the point past which the application *cannot* act. `close` and `drop`
    /// both write — they apply staged edits, re-home the free-space managers of
    /// a file that persists them, and clear a SWMR writer's flag — and they
    /// destroy the handle that would have ordered those writes. A policy that
    /// skipped the barrier there would not hand the caller a cadence; it would
    /// take one away. The cost is one `fsync` per session against the five per
    /// append batch and two or three per commit that this policy removes.
    ///
    /// Three things are given up in exchange. **Power-loss ordering** during the
    /// session: with the barriers gone, a machine that loses power mid-commit
    /// can have the superblock repoint on disk without the data it points at.
    /// **Deferred write errors**: on a filesystem that allocates late, a write
    /// that will fail at writeback still returns success, and `fsync` is where
    /// the `ENOSPC`/`EIO` surfaces, so a commit the filesystem cannot complete
    /// returns `Ok` until the terminal barrier reports it. **Cross-host
    /// visibility**: "another process sees it" is the page cache, so under NFS's
    /// close-to-open semantics a reader on another host — a SWMR reader
    /// included — may not see writes a client is holding.
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
        session.superblock.version,
        SuperblockVersion::Zero | SuperblockVersion::One
    ) {
        return Some(
            "bounded read-write access requires a latest-format file (v2/v3 superblock); \
             leave MemoryStrategy unset, or pass MemoryStrategy::Auto, to fall back to \
             the whole-file mirror here",
        );
    }
    if !session.superblock.base_address.is_zero() {
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
    if let Some((FileSpaceStrategy::Page, false, _)) = create.file_space_strategy()
        && access.memory_strategy() == Some(MemoryStrategy::Bounded)
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
        let page_size = match create.file_space_strategy() {
            Some((FileSpaceStrategy::Page, _, _)) => create
                .file_space_page_size()
                .unwrap_or(FileSpacePageSize::DEFAULT)
                .get(),
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

/// Byte budget for the writes one operation may gather before they are issued
/// (see [`WriteBuffering::Operation`]): the default every exclusively locked
/// read-write session opens under.
///
/// A megabyte, the same figure as [`append::APPEND_BATCH_BYTES`] and for a related
/// reason: that is already what a bounded session spends holding one batch of
/// raw append data, so this is a familiar quantum for the write path rather than
/// a new one. The two are not derived from each other and either may be tuned
/// alone. An operation larger than this is flushed part-way, which costs it
/// little — its writes are long and contiguous by then.
///
/// # Choosing a page-buffer budget
///
/// An explicit [page buffer](crate::FileAccessProperties::with_page_buffer_size)
/// *replaces* this budget rather than adding to it, and holds what it gathers
/// across operations rather than releasing it at every ordering barrier. The two
/// figures are therefore not comparable as memory: this one is a per-operation
/// peak, a page buffer's is continuous residency. A budget below it asks for less
/// of that residency and pays in writes, wherever one long contiguous run has to
/// be flushed and restarted. Writes issued on a 4 KiB-paged file:
///
/// ```text
/// workload                          unset   4 KiB   64 KiB   1 MiB
/// 32 chunk appends into 8 + commit    188      25        4       4
/// one 4 MiB append                    131    1094       74      10
/// ```
///
/// Nothing here bypasses the gatherer the way `H5PB_write` bypasses the C page
/// buffer for any I/O of a page or more (issue #357), so the second row is what
/// a small budget actually costs; `a_smaller_budget_issues_more_writes` asserts
/// that ordering rather than these counts. Which side of it to want is the
/// caller's trade, not this crate's (issue #391).
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

/// A snapshot of a writable file's live space usage (issue #150).
///
/// This is the mutating-session counterpart of the read-only accounting on
/// [`File`](crate::File) ([`file_size`](crate::File::file_size) and
/// [`persisted_free_space`](crate::File::persisted_free_space)): it describes the
/// file *as the session currently holds it*, taken atomically at the moment of
/// the [`space_accounting`](crate::File::space_accounting) call.
///
/// It reflects the committed file plus any immediate in-place appends
/// ([`append`](crate::Dataset::append)), but **not** edits still
/// staged for the next [`commit`](crate::File::commit) — `create_group`,
/// `create_dataset`, `write_dataset`, `append_dataset`, `delete`, `copy`,
/// `copy_from`, and attribute edits change these figures only when they are
/// applied at commit. Use [`has_staged_edits`](crate::File::has_staged_edits) to
/// tell whether such pending work exists.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct SpaceAccounting {
    /// The session's current logical size in bytes: the byte length of the file
    /// as the session holds it. It equals what
    /// [`File::file_size`](crate::File::file_size) reports for the file on disk
    /// right now (the HDF5 `H5Fget_filesize` value), because the session keeps its
    /// in-memory mirror byte-for-byte identical to the file — every committed
    /// write and every immediate in-place append
    /// ([`append`](crate::Dataset::append)) updates both together.
    ///
    /// It is not monotonic: under a file-space strategy with free-space managers,
    /// [`commit`](crate::File::commit) can reclaim trailing free space and *shrink*
    /// the file. It can also exceed the superblock's
    /// recorded end-of-file address when the file was opened carrying unaccounted
    /// trailing bytes (the same slack [`File::file_size`](crate::File::file_size)
    /// surfaces), since opening does not rewrite that address.
    pub logical_size: u64,
    /// Total reusable free bytes this session can draw from before the file has to
    /// grow — the summed length of
    /// [`reusable_free_space`](Self::reusable_free_space).
    ///
    /// Counts the holes this session's earlier commits left inside
    /// [`logical_size`](Self::logical_size) and tracked (superseded object headers, the blocks of
    /// deleted objects) and, for a file created with
    /// `H5Pset_file_space_strategy(persist = true)`, the regions
    /// seeded from the on-disk free-space managers when the session was opened (so
    /// reuse spans sessions). A fresh non-persisting session reports `0` even if
    /// the file contains holes left by other tools — those are never tracked. It
    /// is neither a lower bound on the next write's growth nor a promise of
    /// shrinkage: a region counted here may be truncated away at commit — rather
    /// than reused — if adjacent space is later freed and the coalesced run
    /// reaches end-of-file. A session reports `0` under [`FileSpaceStrategy::None`]. Under
    /// [`FileSpaceStrategy::Aggr`] and [`FileSpaceStrategy::FsmAggr`],
    /// this includes unused metadata and raw aggregate reservations for staged edits and
    /// immediate appends.
    /// Held aggregate spare is counted even when shorter than the free-section threshold. Once
    /// abandoned, interior spare is admitted under the same threshold and adjacency rules as
    /// other freed extents. At persistent session teardown, unused reservations return through a
    /// manager rewrite. The aggregate windows themselves are session state. Under
    /// [`FileSpaceStrategy::FsmAggr`] a commit tracks a
    /// hole shorter than the file's [`threshold`](crate::FileSpaceInfo::threshold) only
    /// where the hole adjoins a tracked one.
    ///
    /// **Which write can spend it is not uniform on a persisting file.** Such a
    /// session may write into a hole only once it has taken that hole *out* of
    /// the on-disk managers, so an in-place [`append`](crate::Dataset::append)
    /// holds a reserve of up to a megabyte per draw that this figure counts and
    /// the next [`commit`](crate::File::commit) cannot place into: the commit
    /// gives the unspent part back in the same tail that rewrites the managers,
    /// which runs *after* it has placed everything. The commit after that one
    /// can. So on a
    /// persisting file read this as "space this session will reuse rather than
    /// grow for", not as "space the very next commit can fill".
    ///
    /// It is also not the on-disk view. That reserve is deliberately absent from
    /// the free-space managers for as long as the session holds it, so
    /// [`File::persisted_free_space`](crate::File::persisted_free_space) — which
    /// reads those managers — reports less than this by whatever is reserved,
    /// and is the figure to compare against what another tool would find.
    pub reusable_free_bytes: u64,
    /// The reusable free regions as `(offset, length)` pairs, sorted ascending by
    /// offset and fully coalesced (no two regions touch or overlap).
    ///
    /// The offsets are **absolute** file offsets (from byte 0, including any
    /// userblock prefix), matching [`logical_size`](Self::logical_size). This
    /// differs from [`File::persisted_free_space`](crate::File::persisted_free_space),
    /// whose pairs are relative to the superblock's base address. The two coincide
    /// for a file with no userblock (base address 0). Empty when nothing is reusable.
    pub reusable_free_space: Vec<(u64, u64)>,
}

/// Describes file storage identified by the editor for regression tests.
///
/// Extents are absolute `(offset, length)` pairs. The editor reads reachable storage with
/// the parsers used for reclamation and reports unsupported structures through
/// [`analysis_complete`](Self::analysis_complete). Unclassified extents can include live
/// storage when the analysis is incomplete.
///
/// This type is outside the supported API.
#[doc(hidden)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpaceOwnership {
    /// The physical file length in bytes.
    pub physical_size: u64,
    /// The end-of-file position stored in the superblock.
    pub logical_size: u64,
    /// The userblock, from byte 0 to the superblock, or empty for a file without one.
    pub userblock_space: Vec<(u64, u64)>,
    /// The dataset payload allocations stored outside object headers in this file, and the global
    /// heap collections that reachable objects refer to.
    pub live_raw_space: Vec<(u64, u64)>,
    /// The identified object headers, indexes, heaps, and file metadata.
    pub live_metadata_space: Vec<(u64, u64)>,
    /// The complete allocations of persisted free-space managers.
    pub free_space_manager_space: Vec<(u64, u64)>,
    /// The free extents available to the editor.
    pub reusable_free_space: Vec<(u64, u64)>,
    /// The bytes below the logical end of file outside the identified allocations and free space.
    pub unclassified_space: Vec<(u64, u64)>,
    /// Whether every reachable storage structure is supported and the traversal finished.
    ///
    /// Shared messages, unsupported headers or heaps, data the editor cannot read or that holds a
    /// file address it cannot locate, and traversal limits make the analysis incomplete, even when
    /// every byte has been classified.
    pub analysis_complete: bool,
    /// The bytes between the logical end of file and the physical file length.
    pub trailing_slack_space: Vec<(u64, u64)>,
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
    #[cfg(any(test, feature = "__page-publication-crosscheck"))]
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

    /// Open exactly as [`open_with_locking`](Self::open_with_locking) does, but
    /// behind an image that withholds its whole-file slice, so every read takes
    /// the [`Source`] path rather than the slice fast path.
    ///
    /// Distinct from [`open_rw_with_strategy`](Self::open_rw_with_strategy), which withholds the
    /// slice *and* the residency: this one still mirrors the file, so a test can
    /// compare the two read forms on a file the bounded open would refuse.
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

    /// Open an existing file for read-write editing under `strategy`: the one
    /// place that picks between the bounded backing (a [`HandleImage`] keeping no
    /// whole-file mirror, so resident memory is the metadata-cache budget plus
    /// whatever is being parsed) and the whole-file mirror. Backs
    /// [`File::open_rw`](crate::File::open_rw), which selects between them by
    /// the strategy it is given.
    ///
    /// The eligibility rules are checked here rather than deferred, because a
    /// caller who asked for bounded memory cannot be silently given the mirror
    /// instead. Under [`MemoryStrategy::Bounded`] a file the bounded engine
    /// cannot edit is refused up front; [`MemoryStrategy::Auto`] opts in to
    /// falling back to the mirror instead (issue #198, steps 3 and 4).
    ///
    /// Falls back only for [`bounded_only_limitation`]. [`open_imaged`](Self::open_imaged)
    /// rejects unsupported address widths and file layouts for both backings. A non-persistent
    /// PAGE file requires the mirror for staged edits in fresh pages.
    pub(crate) fn open_rw_with_strategy(
        path: &Path,
        cache: MetadataCacheConfig,
        locking: FileLocking,
        strategy: MemoryStrategy,
    ) -> Result<Self, Error> {
        if strategy == MemoryStrategy::Mirrored {
            return Self::open_with_locking(path, locking);
        }
        let mut session = Self::open_imaged(path, Some(locking), |handle, len| {
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
            return Self::open_with_locking(path, locking);
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
        if session.superblock.version != SuperblockVersion::Three
            || !session.superblock.base_address.is_zero()
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
    /// Serializes the version 2 or 3 superblock with its encoded address values, then applies
    /// the session's data durability barrier.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the superblock cannot be serialized, or [`Error::Io`] if
    /// writing or syncing the image fails.
    fn set_consistency_flags(&mut self, flags: SuperblockStatusFlags) -> Result<(), Error> {
        self.superblock.consistency_flags = flags;
        self.held_status_flags = flags;
        let bytes = hdf5_pure_format::__private::serialize_superblock(&self.superblock)?;
        self.write_at(self.sb_sig_off, &bytes)?;
        self.barrier_data()?;
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
    /// A failure rolls the byte back, best-effort. The alternative is a file
    /// marked in use by a session that never opened, which nothing can read until
    /// [`File::clear_swmr_flag`](crate::File::clear_swmr_flag) — a steep price
    /// for an open that failed. Best-effort because the rollback writes through
    /// the same handle that just failed; it is a chance, not a guarantee.
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

    /// Shared open path over any backing: acquire the handle (and, when asked,
    /// the exclusive lock), parse and validate the superblock through a borrowed
    /// view of the handle, and only then let `build` decide how the bytes are
    /// held.
    ///
    /// Every check of the superblock and of the File Space Info page size comes before `build`,
    /// because `build` may read the whole file, and a check after it would have the editor read a
    /// 20 GB flagged file into memory only to reject it. The reads of the superblock and its
    /// extension are a few bounded windows.
    ///
    /// `build` receives the file's length as well as the handle because a
    /// mirrorless image has to be told its end-of-file — it has no buffer whose
    /// length implies it.
    ///
    /// Nothing below this point reads the file as a slice, which is what lets
    /// one engine open a whole-file mirror, a mirrorless handle, and (in tests)
    /// a mirror that withholds its slice.
    fn open_imaged(
        path: &Path,
        lock: Option<FileLocking>,
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
        let LocatedSuperblock {
            address,
            superblock,
            widths,
        } = superblock::read(&probe)?;
        let sb_sig_off = address.get();

        // Refuse a file a writer already holds, before anything is mutated. This
        // is the one exclusion the OS lock above cannot make: a SWMR writer
        // takes no lock, so its file is lock-free but flagged (issue #245).
        file_lock::check_status_flags(
            &superblock,
            file_lock::OpenIntent::Write,
            file_lock::OpenTarget::Path(path),
        )?;
        if widths.offsets != OFFSET_WIDTH || widths.lengths != LENGTH_WIDTH {
            return Err(Error::EditUnsupported(
                "only 8-byte offsets and lengths are supported for in-place editing",
            ));
        }
        // A userblock shifts the whole HDF5 image forward by `base_address`: the
        // superblock sits at the base address and every stored address is relative
        // to it (the end-of-file address is the sole absolute field). The editor
        // supports this by reading at `stored + base` and writing back
        // `file_offset - base`. Only the canonical layout — superblock located
        // exactly at the base address (e.g. a MATLAB v7.3 `.mat` file's 512-byte
        // userblock) — is accepted; a base address that disagrees with the
        // superblock's location is a relocated or malformed file we will not rewrite.
        if superblock.base_address != BaseAddress::new(sb_sig_off) {
            return Err(Error::EditUnsupported(
                "a file whose superblock is not located at its base address is not editable in place",
            ));
        }
        let addresses = AddressContext::from_effective_base(superblock.base_address);

        // `H5Fopen` rejects a page size outside 512 bytes to 1 GiB as well (`H5Ofsinfo.c` and
        // `H5Fsuper.c`, HDF5 2.2.0).
        let file_space_info = Self::extension_fsinfo(&probe, &superblock, widths, addresses)?;
        let fresh_page_boundary = if let Some((_, info)) = &file_space_info
            && info.strategy == FileSpaceStrategy::Page
            && !info.persist
        {
            let stored_eoa = addresses.relative(superblock.eof_address)?;
            let superblock_len =
                hdf5_pure_format::__private::serialize_superblock(&superblock)?.len() as u64;
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
            sb_sig_off,
            addresses,
            superblock,
            widths,
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
            reserved: FreeList::new(),
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
            publish_attempted: false,
            staging_batch: false,
            staged_generation: 0,
            held_status_flags: SuperblockStatusFlags::new(),
            sync_policy: SyncPolicy::Always,
        };
        // If the file persists its free space, seed the free list from the
        // on-disk managers and arm persistence for future commits.
        session.load_persisted_free_space(file_space_info);
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

    /// Let this session's writes span operations, up to `max_bytes` of them: the
    /// `H5Pset_page_buffer_size` analogue, requested through
    /// [`FileAccessProperties::with_page_buffer_size`](crate::FileAccessProperties::with_page_buffer_size).
    ///
    /// Refused for a budget below the page this session merges within — the
    /// file's own file-space page size when it is paged and
    /// [`DEFAULT_GATHER_PAGE`] when it is not, which is what
    /// [`gather_page_size`](Self::gather_page_size) answers. A buffer that
    /// cannot hold one page flushes on every page it touches. The C library
    /// reaches the same conclusion and acts on it *silently*, rounding the budget
    /// up at `H5Fopen`, which is the one part of its behavior not worth copying.
    ///
    /// Every larger budget is honored, including one below the
    /// [`WRITE_GATHER_BYTES`] this session was already gathering under. That was
    /// refused until issue #391; see the constant for what a small one costs.
    ///
    /// It does **not** require a paged file, where `H5Pset_page_buffer_size`
    /// does. The C page buffer is a page *cache*, and `H5PB_create` rejects an
    /// unpaged file because its `min_meta_perc`/`min_raw_perc` reservations are
    /// counted in pages that the paged allocator keeps segregated by kind. This
    /// is a write gatherer instead: it merges runs within a page-sized window and
    /// flushes whole, so the window is all it needs, and `gather_page_size`
    /// supplies one on an unpaged file already — it is what every read-write
    /// session's default [`Operation`](WriteBuffering::Operation) gathering runs
    /// under. Lifting the requirement leaves the file byte-identical, which is
    /// what the unpaged arm of `a_page_buffer_holds_dirty_pages_across_operations`
    /// asserts (issue #357).
    ///
    /// A zero budget leaves the session's default gathering alone, matching the
    /// property's own "unset" value.
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
        // is entitled to spend the writes for that. See `WRITE_GATHER_BYTES` for
        // the measurement.
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
                 with_file_space_strategy(FileSpaceStrategy::Page, true, ..), or drop the \
                 page buffer",
            ));
        }
        // The crash mark below is the whole reason this property is offered at
        // all, and only a version-3 superblock carries one that anybody reads:
        // `check_status_flags` is gated there, matching where the C library gates
        // it. A version-2 paged file is reachable — `LibVer::V18` writes a
        // version-2 superblock and nothing about paged file space rules it out —
        // so this is a refusal rather than an assertion (issue #308).
        if self.superblock.version != SuperblockVersion::Three {
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

    /// Constructs the session's reusable-space strategy from the file's File Space Info, seeds it
    /// from persisted managers where supported, and records the manager and superblock-extension
    /// blocks that the next persisting commit frees.
    ///
    /// A file without a File Space Info message keeps the default `FSM_AGGR` strategy installed by
    /// [`open_imaged`](Self::open_imaged), with separate metadata and raw aggregates. Explicit
    /// [`FileSpaceStrategy::Aggr`] and [`FileSpaceStrategy::FsmAggr`] also use these aggregates.
    /// Reads persisted manager sections only when the recorded strategy has free-space managers
    /// and persistence is enabled. Adds eligible sections to the free lists at their stored
    /// addresses. A non-persistent PAGE session starts with empty free lists.
    fn load_persisted_free_space(&mut self, file_space_info: Option<(u64, FileSpaceInfo)>) {
        let Some((ext_addr, info)) = file_space_info else {
            if let Ok(eoa) = self.end_of_allocation() {
                self.aggregation = Some(AggregateSpace::new(eoa.get()));
            }
            return;
        };
        self.space =
            SessionSpace::from_strategy(Some(info.strategy), info.threshold, info.page_size);
        if (info.strategy == FileSpaceStrategy::Aggr || info.strategy == FileSpaceStrategy::FsmAggr)
            && let Ok(eoa) = self.end_of_allocation()
        {
            self.aggregation = Some(AggregateSpace::new(eoa.get()));
        }
        let paged = self.space.is_paged();
        if !info.persist || !self.space.is_enabled() {
            return;
        }
        let widths = self.widths;
        let Ok(eoa) = self.end_of_allocation() else {
            return;
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
                    self.addresses.effective_base(),
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
                        let Some(extent) = FreeExtent::new(section.addr.get(), section.size) else {
                            continue;
                        };
                        tagged.push((*kind, extent));
                    }
                }
                tagged.sort_unstable_by_key(|&(_, extent)| extent.start());
                let mut prev_end = 0u64;
                for (kind, extent) in tagged {
                    if extent.end() > eoa.get() || extent.start() < prev_end {
                        continue;
                    }
                    prev_end = extent.end();
                    self.space.seed_persisted(kind, extent);
                }
                self.space.finish_seed();
            }
        } else if let Ok(mut sections) = managers.iter().try_fold(
            Vec::new(),
            |mut all, (_, _, read)| -> Result<_, &FormatError> {
                all.extend_from_slice(&read.as_ref()?.0);
                Ok(all)
            },
        ) {
            sections.sort_unstable_by_key(|section| section.addr);
            let mut prev_end = 0u64;
            for section in sections {
                let Some(extent) = FreeExtent::new(section.addr.get(), section.size) else {
                    continue;
                };
                if extent.end() > eoa.get() || extent.start() < prev_end {
                    continue;
                }
                prev_end = extent.end();
                self.space.seed_persisted(Some(ManagerKind::Flat), extent);
            }
        }

        let mut old_blocks = Vec::new();
        if let Ok(spans) = self.oh_chunk_spans(ext_addr) {
            old_blocks.extend(
                spans
                    .into_iter()
                    .filter_map(|(at, len)| self.stored_extent(at, len)),
            );
        }
        for (_, manager, read) in &managers {
            match read {
                Ok((_, blocks)) => old_blocks.extend(
                    blocks
                        .iter()
                        .filter_map(|&(at, len)| self.stored_extent(at, len)),
                ),
                Err(_) => old_blocks.extend(self.manager_header_block(widths, *manager)),
            }
        }

        self.persist = Some(PersistState {
            strategy: info.strategy,
            threshold: info.threshold,
            page_size: info.page_size,
            old_blocks,
        });
    }

    /// Returns the extent of the free-space manager header at the stored address `addr`, in stored
    /// addresses, or `None` if the read fails or the header does not parse.
    fn manager_header_block(
        &self,
        widths: FormatWidths,
        addr: StoredAddress,
    ) -> Option<FreeExtent> {
        let len = hdf5_pure_format::__private::free_space_manager_header_len(widths);
        let at = self.addresses.resolve(addr).ok()?;
        let fshd = self
            .image()
            .read_metadata_at(at.get(), len.to_usize().ok()?)
            .ok()?;
        FreeSpaceManagerHeader::parse(widths, &fshd)
            .ok()
            .and_then(|_| FreeExtent::new(addr.get(), len))
    }

    /// Returns the extent of `len` bytes at the file position `at`, in stored addresses.
    ///
    /// Returns `None` if `at` lies below the base address, or if the extent is empty or ends past
    /// [`u64::MAX`].
    fn stored_extent(&self, at: u64, len: u64) -> Option<FreeExtent> {
        FreeExtent::new(self.addresses.relative(at).ok()?.get(), len)
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
        let start = addresses.relative(physical_len)?.get().max(eoa.get());
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
        self.addresses.relative(self.image.len())
    }

    /// Reads the superblock extension of `superblock` from `source`, and returns its absolute
    /// address and the File Space Info message in its object header, or `None` if the file has no
    /// extension or the extension has no such message.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the extension's absolute address exceeds `u64`,
    /// and the parser's error if the superblock's widths, the extension's object header, or the
    /// message does not parse.
    fn extension_fsinfo(
        source: &(impl Source + ?Sized),
        superblock: &Superblock,
        widths: FormatWidths,
        addresses: AddressContext,
    ) -> Result<Option<(u64, FileSpaceInfo)>, FormatError> {
        let base = addresses;
        if matches!(
            superblock.version,
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
        let ext_addr = base.absolute(ext_rel)?;
        let oh = ObjectHeader::parse_from_source_with_widths(
            &SourceMetadata(source),
            AccessMode::ReadWrite,
            ext_addr,
            widths,
            base.effective_base(),
        )?;
        let Some(msg) = oh
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::FILE_SPACE_INFO)
        else {
            return Ok(None);
        };
        let info = hdf5_pure_format::__private::parse_file_space_info(widths, &msg.data)?;
        Ok(Some((ext_addr, info)))
    }

    /// Register a live [`BufferedAppender`](crate::BufferedAppender) on `path`
    /// (`None` for a handle reached by object reference), returning the token
    /// that releases it.
    ///
    /// Refused when the claim could not be honored: a second appender on the
    /// same dataset would interleave the two buffers a chunk at a time, and a
    /// staged edit already pending on that path is one the appender's own flush
    /// would later refuse.
    pub(crate) fn claim_for_appender(
        &mut self,
        path: Option<&ObjectPathBuf>,
    ) -> Result<u64, Error> {
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

    /// Reads issued bytes for the complete image length, leaving pending writes buffered.
    ///
    /// # Errors
    ///
    /// Returns the read or conversion error if the image cannot be read through the disk handle.
    #[cfg(any(test, feature = "__page-publication-crosscheck"))]
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
        &self.superblock
    }

    /// Returns the address and length widths validated when this session opened the file.
    pub(crate) const fn widths(&self) -> FormatWidths {
        self.widths
    }

    /// Returns the mapping between stored addresses and physical positions in this session.
    pub(crate) const fn addresses(&self) -> AddressContext {
        self.addresses
    }

    /// The file's shared-message (SOHM) master table, or `None` for a file that
    /// shares no messages — which is every file this crate writes and nearly
    /// every file it is handed.
    ///
    /// Costs one small object-header parse per commit on a file that has a
    /// superblock extension at all, and nothing on one that does not.
    fn shared_message_table(&self) -> Result<Option<crate::sohm::SohmTable>, Error> {
        let widths = self.widths;
        let rel = match self.superblock.superblock_extension_address {
            Some(rel) if !rel.is_undefined(widths.offsets) => rel,
            _ => return Ok(None),
        };
        let base = self.addresses;
        // Best-effort down to here, and strict past it. A superblock extension
        // this engine cannot parse is one nothing in this crate reads a shared
        // message through either, so it is treated as a file that shares none;
        // once the extension is readable and *says* the file shares messages,
        // failing to read the table is an error, because the screen below is
        // then the difference between a refusal and a corrupted index.
        let Ok(abs) = base.absolute(rel) else {
            return Ok(None);
        };
        let Ok(header) = ObjectHeader::parse_from_source_with_widths(
            &SourceMetadata(&self.image()),
            AccessMode::ReadWrite,
            abs,
            widths,
            base.effective_base(),
        ) else {
            return Ok(None);
        };
        let Some(msg) = header
            .messages
            .iter()
            .find(|m| m.msg_type == MessageType::SHARED_MESSAGE_TABLE)
        else {
            return Ok(None);
        };
        let message = crate::sohm::SharedMessageTableMessage::parse_with_offset_width(
            &msg.data,
            widths.offsets,
        )?;
        let framed = BaseOffsetSource {
            inner: self.image(),
            base: base.effective_base(),
        };
        Ok(Some(
            crate::sohm::SohmTable::read_from_source_with_offset_width(
                &SourceMetadata(&framed),
                &message,
                widths.offsets,
            )?,
        ))
    }

    /// Refuse a commit that would strand a shared-message index record naming an
    /// object header. See [`SHARED_MESSAGE_INDEX_NAMES_A_MOVED_OBJECT`].
    fn screen_shared_message_index(&self, invalidated: &InvalidatedAddresses) -> Result<(), Error> {
        if invalidated.is_empty() {
            return Ok(());
        }
        let Some(table) = self.shared_message_table()? else {
            return Ok(());
        };
        let base = self.addresses;
        let framed = BaseOffsetSource {
            inner: self.image(),
            base: base.effective_base(),
        };
        for index in &table.indexes {
            let records = crate::sohm::read_index_records_from_source(&framed, index, self.widths)?;
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
        LibVer::from_superblock_version(self.superblock.version)
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

    /// Returns a snapshot of this session's live space usage, the current file size and the free
    /// space it can reuse.
    ///
    /// This is the mutating-session analogue of the read-only accounting on
    /// [`File`](crate::File): it answers "how big is the file right now, and how
    /// much space can be reused before it must grow?" from the session's own live
    /// state. The snapshot reflects the committed file plus any immediate in-place
    /// appends ([`append_inplace_gathered`](Self::append_inplace_gathered)) but excludes edits still
    /// staged for the next [`commit`](Self::commit); see [`SpaceAccounting`] for
    /// the field-by-field semantics and [`has_staged_edits`](Self::has_staged_edits)
    /// for detecting pending work.
    ///
    /// On a paged file (`H5F_FSPACE_STRATEGY_PAGE`) the reported regions are the
    /// union of the per-page-type managers, which a commit draws on under the rule
    /// paging exists for: an allocation takes a hole of the page type it is placing,
    /// or whole pages that are free of both, so reuse never re-mixes a page
    /// (issue #261). Space whose page type this file does not settle is recorded
    /// and handed back to the reference library but never spent, and is excluded
    /// from the reusable figure.
    ///
    /// The free lists hold stored addresses, and this function reports each region at its file
    /// position, the stored address plus the base address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if a region ends past [`u64::MAX`] once the base
    /// address is added.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use hdf5_pure::File;
    ///
    /// let file = File::open_rw("existing.h5")?;
    /// let acct = file.space_accounting()?;
    /// println!(
    ///     "{} bytes on disk, {} reusable in {} free region(s)",
    ///     acct.logical_size,
    ///     acct.reusable_free_bytes,
    ///     acct.reusable_free_space.len(),
    /// );
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    #[expect(
        rustdoc::private_doc_tests,
        reason = "rustdoc's test mode collects a private item, so this example is compiled and run by `cargo test --doc`"
    )]
    pub fn space_accounting(&self) -> Result<SpaceAccounting, FormatError> {
        // The append reserve is free space this session can still spend — it is
        // out of the on-disk managers, not out of the file — so leaving it out
        // would make an in-place append look like it had consumed bytes it has
        // not yet placed anything in (issue #387).
        //
        // It is folded back through `FreeList::free` rather than appended and
        // sorted, because a reserve span very often *abuts* what is left of the
        // hole it was drawn from — and the spent end of the reserve walks toward
        // that remainder until the two touch exactly. Reported as two regions
        // they would break the field's documented contract in the one way that
        // misleads: a caller sizing an allocation against the largest region
        // would be told nothing that big fits when it does.
        //
        // It goes back into the list it came from and no other. On a paged file
        // that is the raw list, which is what `take_raw_span` drew it out of;
        // the two page-type lists stay reported side by side as they always
        // were, because a caller cannot allocate across a page-type boundary and
        // merging them would claim it could. `FreeList::free`'s debug assertion
        // comes along and is welcome: the reserve is drawn out of exactly this
        // list, so an overlap here would mean that invariant had already broken.
        //
        // A paged file tracks its free space per page type; report the union of
        // the two, since the caller wants one total rather than a per-manager
        // breakdown.
        let physical = |extent: FreeExtent| {
            let range = FileRange::new(
                self.addresses.resolve(StoredAddress::new(extent.start()))?,
                extent.len(),
            )?;
            Ok((range.start().get(), range.length()))
        };
        let reusable_free_space = if let Some(aggregation) = self.aggregation
            && !self.space.is_enabled()
        {
            let mut extents = aggregation
                .unused()
                .map(physical)
                .collect::<Result<Vec<_>, _>>()?;
            extents.sort_unstable();
            extents
        } else if self.reserved.is_empty() && self.aggregation.is_none() {
            self.space
                .reusable_sections()
                .map(physical)
                .collect::<Result<Vec<_>, FormatError>>()?
        } else {
            self.space
                .reusable_sections_with_raw(
                    self.reserved.extents().iter().copied().chain(
                        self.aggregation
                            .into_iter()
                            .flat_map(AggregateSpace::unused),
                    ),
                )
                .map(physical)
                .collect::<Result<Vec<_>, FormatError>>()?
        };
        let reusable_free_bytes = reusable_free_space.iter().map(|(_, len)| len).sum();
        Ok(SpaceAccounting {
            logical_size: self.image.len(),
            reusable_free_bytes,
            reusable_free_space,
        })
    }

    /// Whether an immediate in-place append may allocate out of already-free
    /// space instead of growing the file (issue #349, issue #387).
    ///
    /// The staged [`commit`](Self::commit) always may: it publishes its
    /// allocations with the superblock repoint, so a crash before that leaves a
    /// file whose root never named them. An immediate append has no repoint, so
    /// what has to hold instead is that the bytes it overwrites are dead *on the
    /// disk as it stands*. Which list satisfies that depends on the session:
    ///
    /// - An ordinary [`File::open_rw`](crate::File::open_rw) session on a
    ///   default-strategy file holds its free space in memory alone. A region in
    ///   [`space`](Self::space) was freed by a commit this session already published, so nothing
    ///   on the disk points at it and nothing on the disk
    ///   claims it is free; a crash at any point of the append leaves it as dead
    ///   as it was. It is spent directly.
    /// - A file that **persists** its free-space managers records its holes on
    ///   disk, so a hole in the session allocator is one a durable manager still advertises and
    ///   spending it would leave that manager describing bytes a live chunk
    ///   occupies — through a clean close as much as a crash, since only a commit
    ///   or [`finalize_persist`](Self::finalize_persist) rewrites the record.
    ///   Such a session therefore spends [`reserved`](Self::reserved) instead:
    ///   space [`reserve_for_immediate_append`](Self::reserve_for_immediate_append)
    ///   has already taken *out* of the published managers.
    /// - A **paged** file keeps its free space per page type in [`SessionSpace`]. It draws its
    ///   reserve through [`SessionSpace::allocate`] with [`AllocationClass::Raw`], which keeps each page
    ///   homogeneous.
    ///   Named as its own term at the call site rather than left to the argument
    ///   that the persist term already covers it (`append_prepare` refuses a
    ///   paged file that does not persist): a page-type invariant proved from
    ///   another function's refusal is exactly the shape that has gone wrong here
    ///   before (issue #261). It costs nothing to name, because `reserved` is
    ///   empty on a session that never reserved — the term declines reuse rather
    ///   than misdirecting it.
    ///
    /// The **SWMR** writer is one absolute bar, and it is not about
    /// publication at all: its concurrent readers may still be inside a region
    /// this session freed, and the format forbids reusing one while they are,
    /// which is why the C library's own SWMR writer allocates append-only. No
    /// test can reach this term today, because a SWMR session refuses every
    /// staged edit ([`Error::SwmrStagedUnsupported`]) and so never frees
    /// anything. It is here so that lifting that refusal cannot silently enable
    /// reuse.
    ///
    /// [`SessionSpace::Disabled`] is the other bar. Its file's strategy has no reusable
    /// free-space managers or is unreadable, so an append allocates at the end of the file.
    ///
    /// One window is inherited rather than introduced. A commit whose superblock
    /// write itself fails leaves [`publish_attempted`](Self::publish_attempted)
    /// set, which withholds the free-list rollback, so the regions that commit
    /// freed stay in the list although whether it published is unknowable
    /// (issue #344). An append can now spend them where only a later commit could
    /// before. That is the same trade, on a larger surface: doing nothing is
    /// still the only answer that is never actively wrong.
    fn immediate_reuse_allowed(&self) -> bool {
        !self.swmr_mode && self.space.is_enabled()
    }

    /// Take a batch of free space out of the on-disk free-space managers so an
    /// immediate in-place append may spend it, for a session that persists them
    /// (issue #387). Afterwards [`reserved`](Self::reserved) holds whatever could
    /// be drawn; the allocator that spends it is what says whether that serves
    /// the append, since the rewrite's own tail may have taken part of it.
    ///
    /// The draw and the publication are one step and in that order: the bytes leave
    /// [`space`](Self::space) first, and the managers are then rewritten through the ordinary
    /// persisting commit tail
    /// with nothing to free and the root unchanged — the same call
    /// [`finalize_persist`](Self::finalize_persist) makes, whose superblock
    /// repoint is the crash-atomic linearization point. Before that repoint the
    /// old managers stand and still list the batch, which is correct because
    /// nothing has been written into it; after it, no durable record calls those
    /// bytes free and the append may have them.
    ///
    /// Nothing here moves a live object: `to_free` is empty and the new root is
    /// the old one, so the tail places only itself, in a hole or past the live
    /// data. The [`located`](Self::located) append-geometry cache is therefore
    /// still valid across this call, which is what lets it run between an
    /// append's plan and its apply.
    ///
    /// When the session has no qualifying persistent free region, the caller tries aggregate
    /// capacity or allocates at EOA. An error means the manager rewrite
    /// itself failed, and the lists are put back so they agree with whatever the
    /// disk now holds, on the rule [`commit`](Self::commit) applies: a rewrite that
    /// never reached its superblock write left the managers as they were, so the
    /// lists go back to exactly what they were, tail placement included; one that
    /// did may have published, so the draw is handed back — those bytes are
    /// genuinely free, nothing having been written into them, and the next commit
    /// records them again — while the span its tail took stays out, since that
    /// tail may now be the live one.
    fn reserve_for_immediate_append(&mut self, want: u64) -> Result<(), Error> {
        if want == 0 || self.persist.is_none() || self.swmr_mode || self.reserved.largest() >= want
        {
            return Ok(());
        }
        let snapshot = self.snapshot_free();
        if !self.take_raw_spans(want, APPEND_RESERVE_BYTES) {
            return Ok(());
        }
        self.publish_attempted = false;
        match self.commit_persisting(
            self.addresses
                .absolute(self.superblock.root_group_address)?,
            Vec::new(),
            TailPlacement::Anywhere,
        ) {
            Ok(()) => Ok(()),
            Err(e) => {
                if self.publish_attempted {
                    self.release_reserve();
                    if let Some(aggregation) = &mut self.aggregation {
                        aggregation.publication_confirmed = false;
                    }
                } else {
                    self.restore_free(snapshot);
                }
                Err(e)
            }
        }
    }

    /// Move raw-appendable free space into [`reserved`](Self::reserved) for one
    /// manager rewrite to publish: runs that can each hold `want` bytes, until
    /// `cap` bytes are held or no hole that large is left. Reports whether
    /// anything was drawn, which is exactly whether the first hole could hold
    /// `want`.
    ///
    /// The draw is sized on what the file has rather than on a fixed batch, and
    /// that is the whole of the fix for issue #413 — [`APPEND_RESERVE_BYTES`]
    /// says why the cap is a cap and not a floor. The rewrite that publishes a
    /// draw is the expensive part, so one draw gathers as many holes as the cap
    /// allows and the rewrite is paid once for all of them: a file fragmented
    /// into chunk-sized holes costs one rewrite per cap's worth of them, not one
    /// per chunk.
    ///
    /// Each run is sized on the largest hole and taken best fit, so it is a whole
    /// hole or a `cap`-bounded prefix of one, and the reserve never holds a
    /// fragment too small for the allocation that drew it. `want` above `cap`
    /// draws one run of exactly `want`: the cap bounds the amortization, not
    /// what a single allocation may need.
    fn take_raw_spans(&mut self, want: u64, cap: u64) -> bool {
        debug_assert!(want > 0, "a draw is sized on a real allocation");
        // The cap is the budget, except that a single allocation larger than it
        // still draws one run for itself.
        let budget = cap.max(want);
        let mut drawn = 0u64;
        while budget - drawn >= want {
            let available = self.largest_raw_run();
            if available < want {
                break;
            }
            let len = available.min(budget - drawn);
            if !self.take_raw_span(len) {
                // Sized on the allocator's own figure, so it serves; a refusal
                // here means the two disagree, and drawing nothing further is
                // the answer that leaves the file merely larger.
                debug_assert!(false, "a run of {len} bytes was reported and not served");
                break;
            }
            drawn += len;
        }
        drawn > 0
    }

    /// The longest contiguous run [`take_raw_span`](Self::take_raw_span) could
    /// move into the reserve right now, or `0` when nothing could.
    fn largest_raw_run(&self) -> u64 {
        self.space.largest(AllocationClass::Raw)
    }

    /// Move `len` bytes of raw-appendable free space into
    /// [`reserved`](Self::reserved), reporting whether a contiguous run that
    /// large was available.
    ///
    /// A paged file draws through [`SessionSpace::allocate`] with [`AllocationClass::Raw`], the same
    /// operation the staged commit uses for dataset data. A flat strategy ignores the allocation class.
    fn take_raw_span(&mut self, len: u64) -> bool {
        match self.space.allocate(len, AllocationClass::Raw) {
            Some(extent) => {
                self.reserved.free(extent);
                true
            }
            None => false,
        }
    }

    /// Give the unspent append reserve back to the lists the managers are written
    /// from, so the next rewrite records it as the free space it is.
    ///
    /// Called immediately before every rewrite of the on-disk managers — the
    /// persisting commit tail and [`finalize_persist`](Self::finalize_persist) —
    /// and nowhere else, so a commit that does *not* rewrite them (the
    /// same-length-overwrite fast path) leaves the reserve intact for the appends
    /// that follow it.
    ///
    /// The reserve is raw-typed on a paged file, so it goes back to the raw list.
    /// A run that was claimed whole from the metadata list is raw space now, and
    /// a page of it that is wholly free can be claimed back by either type
    /// through [`SessionSpace::allocate`], exactly as any other free page can.
    fn release_reserve(&mut self) {
        if self.reserved.is_empty() {
            return;
        }
        let reserved = std::mem::replace(&mut self.reserved, FreeList::new());
        for &extent in reserved.extents() {
            self.space.return_allocation(extent, AllocationClass::Raw);
        }
    }

    /// Publishes a persistent commit after returning the unspent append reserve.
    ///
    /// Reserves eligible sub-threshold manager storage during the first publication, then
    /// recycles it in a separate publication. Other commits use ordinary tail placement.
    /// Attempts to relocate a trailing manager allocation before truncating.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if aggregate planning, address resolution, or metadata
    /// serialization fails. Returns [`Error::EditUnsupported`] if the extension cannot be
    /// rewritten. Returns [`Error::Io`] if writing, synchronizing, or truncating fails.
    fn commit_persisting_releasing_reserve(
        &mut self,
        new_root: u64,
        mut to_free: Vec<(StoredAddress, u64, FreeClass)>,
    ) -> Result<(), Error> {
        self.release_aggregates_for_tail(&mut to_free)?;
        self.release_reserve();
        if let Some(target) = self.recyclable_flat_tail() {
            self.commit_persisting(new_root, Vec::new(), TailPlacement::Preserve)?;
            self.recycle_flat_tail(new_root, to_free, target)?;
        } else {
            self.commit_persisting(new_root, to_free, TailPlacement::Anywhere)?;
        }
        self.shrink_to_the_trailing_free_run(new_root)
    }

    /// Relocates persistent manager blocks into earlier free space before truncating.
    ///
    /// Runs after the root publication makes the preceding commit's freed regions reusable.
    /// The manager tail must end at EOA or, for a flat file, adjoin a tracked free run reaching
    /// EOA. The combined run below and above the tail must meet the retention threshold in
    /// [`FreeList::release_trailing_run`].
    ///
    /// [`TailPlacement::ReuseOnly`] requires the replacement to fit below the old manager tail.
    /// The writer publishes the replacement with the current root before truncating.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if metadata serialization or address resolution fails. Returns
    /// [`Error::Io`] if writing, synchronizing, or truncating fails.
    fn shrink_to_the_trailing_free_run(&mut self, new_root: u64) -> Result<(), Error> {
        let Some(persist) = self.persist.as_ref() else {
            return Ok(());
        };
        let (Some(tail_start), Some(tail_end)) = (
            persist.old_blocks.iter().map(|extent| extent.start()).min(),
            persist.old_blocks.iter().map(|extent| extent.end()).max(),
        ) else {
            return Ok(());
        };
        // A flat tail may have published reusable space above it. That space contains no live
        // allocations and can be released by the same manager rewrite.
        let eoa = self.end_of_allocation()?.get();
        if tail_end != eoa
            && (self.space.is_paged() || self.space.trailing_run_start(eoa) != tail_end)
        {
            return Ok(());
        }
        // Apply the retention threshold of `FreeList::release_trailing_run` to the combined
        // free run and old tail before attempting a rewrite.
        let run_start = self.space.trailing_run_start(tail_start);
        if eoa - run_start < 2 * free_space::TRAILING_RESERVE_TAILS * (tail_end - tail_start) {
            return Ok(());
        }
        self.commit_persisting(new_root, Vec::new(), TailPlacement::ReuseOnly)
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

    /// Override how long this session's writes may sit in memory.
    ///
    /// The image is private to this module, and the only caller outside it is
    /// [`crate::crash_replay`], which sweeps the *same* workload under gathering
    /// and without it. That comparison is the answer to the obvious worry about
    /// making writes larger, so it is worth an accessor; nothing outside a test
    /// build has one.
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

    /// Issues pending writes, synchronizes the file, and completes pending PAGE publication.
    ///
    /// The synchronization barrier is durable under every [`SyncPolicy`]. Publication recovery
    /// verifies issued bytes before adopting the superblock and leaves later staged edits pending.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if writing or synchronization fails. Returns the publication recovery
    /// error if the issued superblock cannot be read or verified.
    pub(crate) fn force_sync(&mut self) -> Result<(), Error> {
        self.image.sync_all()?;
        self.complete_page_publication()
    }

    /// Finalizes free-space metadata or classic aggregate tails at session teardown.
    ///
    /// A persisting session rewrites stale free-space managers and returns unspent immediate
    /// append reserves. A persistent session also returns unused aggregate spare through those
    /// managers. A session with classic aggregation releases unused aggregate space at EOA after
    /// publishing the shorter EOA with the current root. An uncertain publication prevents
    /// aggregate cleanup. Non-persisting sessions without classic aggregation require no
    /// finalization.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if metadata serialization, aggregate planning, or address
    /// resolution fails. Returns [`Error::Io`] if writing, syncing, or truncating fails.
    pub(crate) fn finalize_persist(&mut self) -> Result<(), Error> {
        if self.aggregation.is_some() {
            if self.persist.is_some() {
                return self.finalize_persistent_aggregates();
            }
            return self.finalize_aggregates();
        }
        if self.persist.is_none() || (self.image.len() == self.fsm_len && self.reserved.is_empty())
        {
            return Ok(());
        }
        self.commit_persisting_releasing_reserve(
            self.addresses
                .absolute(self.superblock.root_group_address)?,
            Vec::new(),
        )
    }

    /// Applies staged edits using the session's file-space strategy and synchronization policy.
    ///
    /// Tree edits rebuild affected group headers and publish the new root through the
    /// superblock. Dataset value writes may update existing storage directly. On success,
    /// the staged set is cleared.
    ///
    /// Rejection during preflight preserves the staged set for retry. Once application
    /// consumes the staged set, a failed attempt discards that work. Writes into reusable
    /// or newly allocated space may remain unreferenced after a failure.
    ///
    /// On failure before root publication is attempted, the editor restores allocator state and
    /// attempts to restore prior values from the undo journal. Rollback is disabled once
    /// publication is attempted because a failed publication write may already have changed the file.
    ///
    /// After publication, reference patching or optional root-header reclamation can fail.
    /// The committed tree retains the batch's values, including its in-place overwrites.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if an edit or an encountered file structure cannot
    /// be reproduced. Other preparation and I/O errors propagate from planning, writing,
    /// or synchronizing the file. Returns [`Error::CommitPartiallyApplied`] if restoring prior
    /// values from the undo journal fails, with both the original failure and the restore error.
    pub fn commit(&mut self) -> Result<(), Error> {
        if self.fresh_page_boundary.is_some() && self.publish_attempted {
            self.barrier()?;
            self.complete_page_publication()?;
        }
        self.synchronize_aggregates()?;
        let snapshot = self.snapshot_free();
        self.publish_attempted = false;
        // Cleared at entry, like `superseded_heaps` and for the same reason: a
        // previous attempt that unwound past this function — a panic caught by
        // the session lock, which recovers from poisoning — would otherwise
        // leave entries behind, and this commit's rollback would replay bytes
        // captured before it over whatever lives at those addresses now. The
        // clear on the way out is a separate duty: it stops a journal being
        // held across the idle time between commits.
        self.inplace_undo.clear();
        // The staged set comes out for the duration of the attempt and goes back
        // if the attempt refuses, so a refusal costs the session no staged work
        // (issue #316). What goes back is whatever the attempt did not take: its
        // preflight only reads, and its apply phase takes the whole set in one
        // move at the point of no return, so a preflight refusal restores every
        // edit and a failure past that point restores nothing.
        let mut staged = std::mem::take(&mut self.staged);
        let mut result = self.commit_inner(&mut staged);
        if result.is_err() && !self.publish_attempted {
            // The three legs of the rollback: the file's values, the free
            // lists, and the staged set. Nothing observes the order — the first
            // touches the file and the others only in-memory state, inside one
            // synchronous call — so it is written the way it reads: put the
            // file back, then re-offer what the attempt drew from (issue #344).
            let restored = self.undo_inplace_writes();
            self.restore_free(snapshot);
            self.staged = staged;
            // The set is back, so nothing this session staged has been consumed
            // and every handle onto one of those creations is still pending.
            if let Err(restore) = restored {
                // Both are worth carrying: the refusal is what the caller has to
                // fix, and the write failure is what proves the file changed.
                let refusal = result.expect_err("only a failed attempt is rolled back");
                result = Err(Error::CommitPartiallyApplied {
                    refusal: Box::new(refusal),
                    restore: Box::new(restore),
                });
            }
        } else {
            // The set is gone — published, or in the file as slack a later
            // commit must not re-issue — so a handle onto one of its creations
            // stops being pending and addresses the file from here on.
            self.staged_generation += 1;
        }
        if self.publish_attempted
            && let Some(aggregation) = &mut self.aggregation
        {
            aggregation.publication_confirmed = result.is_ok();
        }
        // For the paths that took no rollback — a commit that succeeded, whose
        // overwrites are now the file's values, and one that failed with the
        // publish already issued, where undoing anything is unsound. A rollback
        // drained the journal on its way through, so it owes nothing here.
        self.inplace_undo.clear();
        if self.fresh_page_boundary.is_some() && self.publish_attempted {
            if result.is_ok() {
                self.publish_attempted = false;
            } else {
                // A failed barrier can leave the new root on disk. Pending writes and uncertain
                // bytes remain owned until a later commit completes the barrier and recovery.
                self.recover_page_publication().ok();
            }
        }
        result
    }

    /// Verifies pending PAGE publication and clears its recovery state on success.
    ///
    /// The caller completes its ordering or synchronization barrier first. Later staged edits
    /// remain pending if verification fails.
    ///
    /// # Errors
    ///
    /// Returns the error from [`recover_page_publication`](Self::recover_page_publication).
    fn complete_page_publication(&mut self) -> Result<(), Error> {
        if self.fresh_page_boundary.is_some() && self.publish_attempted {
            self.recover_page_publication()?;
            self.publish_attempted = false;
        }
        Ok(())
    }

    /// Adopts the published PAGE root and EOA after verifying the accepted superblock bytes.
    ///
    /// Other superblock fields must match the session's framing. Before consuming staged edits
    /// in another commit, the caller drains pending writes through a barrier.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the file bytes differ from the accepted image or
    /// superblock framing. Returns the read, serialization, or parsing error if recovery fails.
    fn recover_page_publication(&mut self) -> Result<(), Error> {
        let len = hdf5_pure_format::__private::serialize_superblock(&self.superblock)?.len();
        let accepted = self.image.read_exact_at(self.sb_sig_off, len)?;
        let mut published = vec![0; len];
        self.image.read_disk_at(self.sb_sig_off, &mut published)?;
        if published != accepted {
            return Err(Error::EditUnsupported(
                "PAGE publication bytes differ from the session image after a failed write; \
                 close and reopen the file before editing again",
            ));
        }
        let recovered = superblock::read(published.as_slice())?.superblock;
        let mut expected = self.superblock.clone();
        expected.root_group_address = recovered.root_group_address;
        expected.eof_address = recovered.eof_address;
        if hdf5_pure_format::__private::serialize_superblock(&expected)? != published {
            return Err(Error::EditUnsupported(
                "PAGE superblock framing changed after a failed publication; \
                 close and reopen the file before editing again",
            ));
        }
        self.superblock = recovered;
        Ok(())
    }

    /// Captures allocator state and heap allocation records for rollback before publication.
    fn snapshot_free(&self) -> FreeSnapshot {
        FreeSnapshot {
            space: self.space.allocation_snapshot(),
            aggregation: self.aggregation,
            reserved: self.reserved.clone(),
            vl_overwrite_heaps: self.vl_overwrite_heaps.clone(),
        }
    }

    /// Restores allocator state and heap allocation records after failure before publication.
    ///
    /// The snapshot was captured by [`snapshot_free`](Self::snapshot_free).
    ///
    /// That rests on nothing the live tree reaches naming a span the attempt
    /// allocated, which takes two rules rather than one. Every object a commit
    /// *places* is unreachable until the superblock repoint; and the one write
    /// that does not wait for the repoint — a same-length value overwrite — is
    /// planned only over bytes that carry no allocated address, because
    /// [`prepare_write`](Self::prepare_write) relocates a staged
    /// variable-length overwrite instead of writing it in place, which is
    /// exactly the counterexample (issue #321).
    ///
    /// [`undo_inplace_writes`](Self::undo_inplace_writes) has nevertheless run
    /// by the time this does, so the file's values are back before its regions
    /// are offered again (issue #344).
    fn restore_free(&mut self, snapshot: FreeSnapshot) {
        self.vl_overwrite_heaps = snapshot.vl_overwrite_heaps;
        self.space.restore_allocation(snapshot.space);
        self.aggregation = snapshot.aggregation;
        self.reserved = snapshot.reserved;
    }

    /// Overwrites `raw` at `at`, keeping the bytes it replaces so a commit that
    /// fails before its repoint can put them back.
    ///
    /// The read covers exactly the range about to be written, which the write
    /// preflight sized and bounds-checked against the on-disk dataset — a
    /// same-length overwrite by construction.
    fn write_inplace_journaled(&mut self, at: u64, raw: &[u8]) -> Result<(), Error> {
        let prior = self.image().read_exact_at(at, raw.len())?;
        self.inplace_undo.push((at, prior));
        self.write_at(at, raw)
    }

    /// Put back the values the refused commit's same-length in-place overwrites
    /// replaced, and drop the journal.
    ///
    /// The third leg of a refused commit's rollback, beside
    /// [`restore_free`](Self::restore_free) and the staged set: those two give
    /// back what the attempt *consumed*, this gives back what it *changed*. It
    /// is the only one of the three that has to touch the file, because a
    /// same-length overwrite is the only edit a commit writes over bytes the
    /// live root already reaches (issue #344).
    ///
    /// # Errors
    ///
    /// Every entry is attempted even after one fails, so the file is restored as
    /// far as it can be, and the first failure met is what comes back. That is the
    /// one outcome leaving values from a batch that was refused, which is why
    /// [`commit`](Self::commit) reports it as
    /// [`Error::CommitPartiallyApplied`] rather than as the refusal.
    fn undo_inplace_writes(&mut self) -> Result<(), Error> {
        let journal = core::mem::take(&mut self.inplace_undo);
        if journal.is_empty() {
            return Ok(());
        }
        let mut failed = None;
        // Newest first, and load-bearing rather than conventional. The commit
        // refuses to overwrite the same *path* twice, which is not the same as
        // the same dataset: an object reachable through two hard links has two
        // paths and one data block, and a same-length overwrite through either
        // is allowed precisely because it rewrites the block every link sees.
        // Two such writes in one commit therefore journal the same address
        // twice, the second entry holding what the first write put there.
        // Replaying oldest-first would finish on that value — one from the very
        // batch being rolled back.
        for (at, prior) in journal.into_iter().rev() {
            if let Err(e) = self.write_at(at, &prior) {
                failed.get_or_insert(e);
            }
        }
        match failed {
            Some(e) => Err(e),
            // A restore that is not a barrier can be gathered and reordered
            // past a later write, which is the whole hazard write gathering
            // introduced (issue #288); order it here instead. The data-only
            // barrier is the one that fits, because putting bytes back into
            // blocks that already existed moves no end-of-file — it is a
            // narrower barrier than the `barrier()` the commit itself uses
            // after these writes, deliberately.
            None => self.barrier_data(),
        }
    }

    fn commit_inner(&mut self, staged: &mut StagedEdits) -> Result<(), Error> {
        if staged.is_empty() {
            return Ok(());
        }

        // An attempt that failed partway may have recorded superseded collections
        // it never freed; this one must not free them on its behalf, since it is
        // not the commit that replaced what they hold.
        self.superseded_heaps.clear();
        // Drop every heap-collection provenance record this batch could falsify,
        // before any of them is read below.
        self.invalidate_heap_provenance(staged);

        // Invalidate the in-place-append geometry cache before doing any work. A
        // commit that reaches here rewrites and relocates object headers, frees
        // vacated regions into `self.space`, and may truncate the file, any of
        // which can leave a cached `Located` pointing at a moved header or into a
        // now-free-eligible region. Clearing at *entry* (rather than the success
        // tail) means a later failure — including one after the durable root flip,
        // which leaves the session reusable — never strands a stale cache. The
        // no-op fast return above does no such work, so it keeps the cache. The
        // The next in-place append re-locates against the fresh file.
        self.located.clear();
        self.resolved.clear();
        // Past this point the commit may relocate object headers, so every address
        // a caller captured earlier is suspect; see `committed`.
        self.committed = true;

        // On a file with a userblock, stored addresses are relative to this base
        // and the editor converts at every disk boundary (read `stored + base`,
        // write `file_offset - base`). Userblock support covers value overwrites,
        // additions of contiguous and chunked/filtered datasets, in-place and
        // relocating overwrites of every layout (chunked, contiguous, compact) with
        // reclaim, object deletion (with base-aware subtree reclaim), object copy
        // (in-file, and cross-file into a userblock destination), group creation,
        // and compact group attributes. Cross-file copy still requires a base-0
        // *source* (see [`copy_from`](Self::copy_from)).
        let base = self.addresses;

        // --- Preflight value overwrites (`write_dataset`) before any write, under
        // the same all-or-nothing contract as additions. Each is resolved,
        // validated (compatible datatype and matching on-disk shape),
        // and classified: a same-length contiguous overwrite is applied straight
        // in place (no header rewrite, no superblock flip), while a resize or
        // compact rewrite relocates the header and is staged against its parent
        // group so the commit below rebuilds it and patches the link. ---
        let mut inplace_writes: Vec<(u64, Vec<u8>)> = Vec::new();
        let mut moving_writes: Vec<(ObjectPathBuf, LinkNameBuf, u64, MovingWrite)> = Vec::new();
        let mut write_targets: Vec<ObjectPathBuf> = Vec::new();
        // The file-wide hard-link count, computed lazily the first time a commit
        // relocates a header: such a write moves the object's header and patches
        // only the one parent link that names it, so an object reachable through
        // more than one hard link would have its other links left pointing at the
        // stale header. Refuse that rather than silently diverge the aliases (a
        // same-length in-place overwrite is unaffected — it rewrites the shared
        // data block, which every link sees).
        //
        // Read by four places, which is every way a header moves: a relocating
        // overwrite, a staged append, a dataset attribute edit, and — since
        // issue #327 — a rebuilt group, which had the same defect and none of
        // the guard.
        let mut incoming_links: Option<Option<HashMap<u64, u32>>> = None;
        for (full, fd) in staged.writes() {
            // A path named twice in one commit would write it twice (and double-
            // free a resized extent); require separate commits.
            if write_targets.contains(full) {
                return Err(Error::EditUnsupported(
                    "the same dataset is overwritten twice in one commit; use separate commits",
                ));
            }
            let addr = crate::group_v2::resolve_path_any_from_source(
                &self.image(),
                AccessMode::ReadWrite,
                &self.superblock,
                self.widths,
                self.addresses,
                &full.as_path(),
            )?;
            match self.prepare_write(addr, fd, full)? {
                WritePlan::InPlace { data_addr, bytes } => {
                    inplace_writes.push((data_addr, bytes));
                }
                // A chunked in-place overwrite never carries staging — a staged
                // one is a `ChunkPayload::Deferred`, which relocates.
                WritePlan::InPlaceChunks { writes } => inplace_writes.extend(writes),
                WritePlan::Moving(mw) => {
                    // A relocating overwrite rewrites the dataset's header and data
                    // address. Every variant is base-aware on a userblock file: the
                    // chunked one rebuilds the chunk blob with stored addresses and
                    // reclaims the old storage base-relative, the contiguous one
                    // stores the relocated data address base-relative (and frees the
                    // old extent at its absolute offset), and the compact one carries
                    // its data inline. The parent link to the rewritten header is
                    // patched base-relative below.
                    //
                    // A relocating overwrite is safe only when this is the
                    // dataset's sole hard link. Compute the link graph once.
                    let counts = incoming_links
                        .get_or_insert_with(|| self.count_incoming_hard_links())
                        .as_ref();
                    match counts.and_then(|c| c.get(&addr)) {
                        Some(&1) => {}
                        _ => {
                            return Err(Error::EditUnsupported(
                                "overwriting a dataset that resizes or relocates its header is \
                                 only supported when it has a single hard link",
                            ));
                        }
                    }
                    let Some((parent, leaf)) = full.split_leaf() else {
                        return Err(Error::EditUnsupported("cannot overwrite the root group"));
                    };
                    moving_writes.push((parent, leaf, addr, mw));
                }
            }
            write_targets.push(full.clone());
        }

        // --- Preflight appends (`append_dataset`) under the same all-or-nothing,
        // single-hard-link contract. Each plans a relocating append — existing
        // chunk data stays in place; the appended (and any rewritten trailing)
        // chunks and a rebuilt Extensible-Array index are staged, and the whole is
        // treated like a relocating overwrite of the dataset's header (staged
        // against its parent group so the commit patches the link). A zero-length
        // append is a no-op and is dropped here. ---
        for (full, ab) in staged.appends() {
            let Some((parent, leaf)) = full.split_leaf() else {
                return Err(Error::AppendUnsupported("cannot append to the root group"));
            };
            if ab.raw().is_empty() {
                continue; // nothing to append
            }
            // A dataset overwritten or appended earlier in this commit would be
            // planned against a stale header and its old storage double-freed;
            // require separate commits.
            if write_targets.contains(full) {
                return Err(Error::AppendUnsupported(
                    "the same dataset is edited more than once in one commit; use separate commits",
                ));
            }
            let addr = crate::group_v2::resolve_path_any_from_source(
                &self.image(),
                AccessMode::ReadWrite,
                &self.superblock,
                self.widths,
                self.addresses,
                &full.as_path(),
            )?;
            let mw = Self::prepare_append(&self.image(), addr, ab, base, self.widths)?;
            // A relocating append moves the dataset's object header and patches only
            // the one parent link that names it, so it is safe only when this is the
            // dataset's sole hard link (same rule as a relocating overwrite).
            let counts = incoming_links
                .get_or_insert_with(|| self.count_incoming_hard_links())
                .as_ref();
            match counts.and_then(|c| c.get(&addr)) {
                Some(&1) => {}
                _ => {
                    return Err(Error::AppendUnsupported(
                        "appending relocates the dataset header; only supported when it \
                         has a single hard link",
                    ));
                }
            }
            moving_writes.push((parent, leaf, addr, mw));
            write_targets.push(full.clone());
        }

        // --- Preflight dataset attribute edits (`set_dataset_attr` /
        // `remove_dataset_attr`) under the same all-or-nothing, single-hard-link
        // contract. Each gathers the dataset's verbatim object-header region,
        // applies the compact attribute ops to it, and stages a relocating
        // `AttrEdit` header rewrite against the parent group — like a value
        // overwrite, but the data-layout message (and thus the chunk data and index)
        // is preserved verbatim, so only the header moves. ---
        if !staged.dataset_attrs().is_empty() {
            // Collect the ops per dataset in first-seen path order, so multiple edits
            // to one dataset produce a single relocating header rewrite.
            let mut order: Vec<ObjectPathBuf> = Vec::new();
            let mut ops_by_path: HashMap<&ObjectPathBuf, Vec<&AttrOp>> = HashMap::new();
            for (path, op) in staged.dataset_attrs() {
                if !ops_by_path.contains_key(path) {
                    order.push(path.clone());
                }
                ops_by_path.entry(path).or_default().push(op);
            }
            for full in order {
                let ops = ops_by_path.remove(&full).unwrap();
                let Some((parent, leaf)) = full.split_leaf() else {
                    return Err(Error::EditUnsupported(
                        "cannot set a dataset attribute on the root group; use set_group_attr",
                    ));
                };
                // A dataset already overwritten or appended in this commit would be
                // planned against a stale header; require separate commits.
                if write_targets.contains(&full) {
                    return Err(Error::EditUnsupported(
                        "the same dataset is edited more than once in one commit (an attribute \
                         edit plus another edit); use separate commits",
                    ));
                }
                let addr = crate::group_v2::resolve_path_any_from_source(
                    &self.image(),
                    AccessMode::ReadWrite,
                    &self.superblock,
                    self.widths,
                    self.addresses,
                    &full.as_path(),
                )?;
                // An attribute edit relocates the dataset's object header and patches
                // only the one naming link, so it is safe only when this is the
                // dataset's sole hard link (same rule as a relocating overwrite).
                let counts = incoming_links
                    .get_or_insert_with(|| self.count_incoming_hard_links())
                    .as_ref();
                match counts.and_then(|c| c.get(&addr)) {
                    Some(&1) => {}
                    _ => {
                        return Err(Error::EditUnsupported(
                            "editing a dataset attribute relocates its header; only supported \
                             when it has a single hard link",
                        ));
                    }
                }
                let region = Self::gather_oh_messages(&self.image(), addr, base)?;
                let edits =
                    attributes::plan_attr_ops(&self.image(), base, Some(addr), &region, &ops)?;

                moving_writes.push((
                    parent,
                    leaf,
                    addr,
                    MovingWrite::AttrEdit {
                        region: edits.region,
                        attrs: edits.attrs,
                        replaces_dense_storage: edits.replaces_dense_storage,
                    },
                ));
                write_targets.push(full);
            }
        }

        // Pre-commit object-header addresses this commit rewrites elsewhere, for
        // `InvalidatedAddresses` below: a reference naming one of these keeps an
        // address the commit is vacating (issue #317). Read off the one list every
        // relocating write is staged on, so a write added later cannot be left out
        // of the screen without failing to compile.
        let mut moved_headers: Vec<u64> =
            moving_writes.iter().map(|&(_, _, addr, _)| addr).collect();

        // Fast path: when the only staged edits are same-length in-place
        // overwrites, apply them straight to their data blocks and return without
        // rebuilding any header or flipping the superblock root. The commit's
        // linearization point is the synced data write — there is no tree to
        // repoint, so each overwrite stands alone. (A persisting file takes the
        // same path: no free-space change occurs.)
        //
        // Because this path never rewrites the superblock, it deliberately leaves
        // it untouched — including a pre-existing stale consistency flag (e.g. one
        // left by a crashed SWMR writer). A lone same-length value overwrite does
        // not introduce any inconsistency, so it does not clear one either; an edit
        // that takes the full path below (any header/root change) clears the flag
        // as usual.
        //
        // A staged variable-length overwrite cannot reach here at all: it
        // relocates, so `moving_writes` is non-empty for it. That is enforced
        // where the plan is chosen (`prepare_write`) rather than re-tested here,
        // and it has to hold for a second reason besides the one that made it
        // relocate — such an overwrite places a global heap collection, and an
        // append moves end-of-file, which only the superblock this path leaves
        // untouched records (issue #321).
        if moving_writes.is_empty() && staged.scope() == StagedScope::Datasets {
            // This path's own point of no return, and it takes the staged set
            // for the same reason the main one below does (issue #316): every
            // refusal that applies to these overwrites has already run, and a
            // batch some of which has been written is not one a later `commit`
            // may re-issue. It is also what this path already did, back when the
            // write preflight drained the staged set on its way here.
            //
            // Leaving the set in place would make a half-written batch
            // retryable — these overwrites are same-length writes to fixed
            // addresses, so repeating them is idempotent — but that reads the
            // rule the other way round for one path, on an argument nothing
            // enforces. What the file holds is settled separately and the same
            // way on both paths: a `write_at` that fails partway is undone
            // through the journal below, so the batch is dropped but the
            // dataset's values are the ones it had (issue #344).
            drop(std::mem::take(staged));
            for (data_addr, bytes) in &inplace_writes {
                self.write_inplace_journaled(*data_addr, bytes)?;
            }
            self.barrier()?;
            return Ok(());
        }

        // --- Plan: build the tree of "dirty" groups (root plus every group on a
        // path to an addition or deletion), validating every target before any
        // write. `add_targets` records the full paths created this commit, used
        // to reject a deletion that overlaps an addition. ---
        let mut nodes: BTreeMap<ObjectPathBuf, Node> = BTreeMap::new();
        nodes.entry(ObjectPathBuf::root()).or_default(); // root is always dirty
        let mut add_targets: Vec<ObjectPathBuf> = Vec::new();
        // Where each in-file `copy` reads from. A copy takes its bytes from the
        // *pre-commit* file, so a source this same commit replaces would copy the
        // object being removed while the replacement lands at the same path — see
        // the delete-staging loop, which refuses that. Cross-file copies are not
        // tracked: their source is in another file, so no path here can name it.
        let mut copy_sources: Vec<ObjectPathBuf> = Vec::new();
        let mut attr_targets: Vec<ObjectPathBuf> = Vec::new();

        // Mark explicitly-created new groups, ensuring their ancestor chain.
        for path in staged.groups() {
            if path.is_empty() {
                return Err(Error::EditUnsupported("cannot create the root group"));
            }
            ensure_ancestors(&mut nodes, path);
            nodes.entry(path.clone()).or_default().is_new = true;
            add_targets.push(path.clone());
        }

        // Make each staged dataset's parent group a node (with its ancestor
        // chain). The datasets themselves stay in the staged set, which the
        // preflight reads but must not empty (issue #316); `datasets_by_group`
        // below is the grouped view both the preflight and the apply loop use.
        for (parent, fd) in staged.datasets() {
            add_targets.push(parent.join(&fd.name));
            ensure_ancestors(&mut nodes, parent);
        }

        // Attach relocating value overwrites (resized contiguous or compact) to
        // their parent group nodes: the new header is written below and the
        // parent's existing link patched to it, like an existing child group.
        for (parent, leaf, old_oh, mw) in moving_writes {
            ensure_ancestors(&mut nodes, &parent);
            nodes
                .entry(parent)
                .or_default()
                .writes
                .push((leaf, old_oh, mw));
        }

        // Stage group attribute edits against their target groups. A target may
        // be a newly-created group from this same commit, but not a copied
        // destination or a dataset being added in the same commit. The ops
        // themselves stay in the staged set and are looked up by path where they
        // are applied, so a refusal below still gives them back (issue #316).
        for (path, _) in staged.group_attrs() {
            ensure_ancestors(&mut nodes, path);
            attr_targets.push(path.clone());
        }

        // Stage copies: validate the source subtree is copyable (read-only),
        // then treat the destination like an addition to its parent group.
        for (src, dst) in staged.copies() {
            if src.is_empty() {
                return Err(Error::EditUnsupported("cannot copy the root group"));
            }
            let Some((parent, leaf)) = dst.split_leaf() else {
                return Err(Error::EditUnsupported("copy destination path is empty"));
            };
            if dst.starts_with(src) {
                return Err(Error::EditUnsupported(
                    "cannot copy an object into itself or its own subtree",
                ));
            }
            let src_addr = crate::group_v2::resolve_path_any_from_source(
                &self.image(),
                AccessMode::ReadWrite,
                &self.superblock,
                self.widths,
                self.addresses,
                &src.as_path(),
            )?;
            // Read the source subtree from this file's own mirror (`cross_file`
            // false: same address space, so verbatim addresses stay valid). On a
            // userblock file the stored addresses are base-relative, so pass this
            // session's base for the read to absolutize them.
            let tree = Self::read_copy_subtree(
                &self.image(),
                AccessMode::ReadWrite,
                src_addr,
                0,
                false,
                base,
                self.widths,
            )?;
            copy_sources.push(src.clone());
            add_targets.push(dst.clone());
            ensure_ancestors(&mut nodes, &parent);
            nodes.entry(parent).or_default().copies.push((leaf, tree));
        }

        // Stage cross-file copies: their subtrees were already read out of the
        // source file (with foreign-address screening) when `copy_from` was
        // called, so here they are simply linked into the destination parent like
        // any other addition.
        for (dst, _) in staged.cross_copies() {
            let Some((parent, leaf)) = dst.split_leaf() else {
                return Err(Error::EditUnsupported("copy destination path is empty"));
            };
            add_targets.push(dst.clone());
            ensure_ancestors(&mut nodes, &parent);
            nodes.entry(parent).or_default().cross_copies.push(leaf);
        }

        // Stage deletions: each must exist, must not overlap any other staged
        // change *unless* this commit replaces what it removes, and is recorded
        // against its parent group (which becomes dirty). `deleted_addrs` keeps
        // each removed object's header address so its owned blocks can be
        // reclaimed after the commit lands (issue #21).
        let delete_targets = staged.deletes();
        let mut deleted_addrs: Vec<u64> = Vec::new();
        for (i, d) in delete_targets.iter().enumerate() {
            let Some((parent, leaf)) = d.split_leaf() else {
                return Err(Error::EditUnsupported("cannot delete the root group"));
            };
            let del_addr = crate::group_v2::resolve_path_any_from_source(
                &self.image(),
                AccessMode::ReadWrite,
                &self.superblock,
                self.widths,
                self.addresses,
                &d.as_path(),
            )?;
            deleted_addrs.push(del_addr);
            // A deletion may overlap other staged work when this commit
            // *replaces* what it removes: an addition names exactly `d`, so the
            // removal and the new object at the same path are one rotation
            // rather than an edit of something being deleted (issue #305).
            let recreated = add_targets.iter().any(|t| t == d);
            // A replacement also requires that every group node at or below `d`
            // is one this commit builds fresh. A node that is *not* new is
            // rebuilt from the old object's on-disk header — the object being
            // replaced — and both ways that lands are wrong, in the two shapes
            // this guard was measured against:
            //
            // * At `d` itself (a group attribute set on a path this commit
            //   replaces with a *dataset*), the node and the replacement share a
            //   `path_addr` key. The replacement's address overwrites the group's,
            //   the parent's link is then patched to the address it already had,
            //   and the rebuilt group header is left orphaned — a commit that
            //   returns `Ok` having **silently discarded** the staged attribute.
            // * Strictly below `d`, the parent is a freshly built region with no
            //   existing link to patch, so `patch_link_target` reports a missing
            //   child link partway through the apply. That leaves a valid file
            //   (the superblock is never repointed) but appends dead bytes and
            //   reports the wrong thing.
            //
            // Refusing here keeps both in the preflight, where the commit is
            // all-or-nothing and the message can name the actual conflict.
            if recreated
                && !nodes
                    .iter()
                    .all(|(key, node)| !key.starts_with(d) || node.is_new)
            {
                return Err(Error::EditUnsupported(
                    "a staged edit names a group at or under a replaced path that this \
                     commit does not itself create; create it in the same commit, or use \
                     separate commits",
                ));
            }
            // A copy reading from a path this commit replaces is the same conflict
            // seen from the other side: `read_copy_subtree` already took its bytes
            // from the pre-commit file, so the commit would place the *original*
            // at the copy's destination while placing something else at `d` — two
            // different objects from one path, in one commit, with no error. A
            // source that is merely deleted and not replaced is unambiguous (it is
            // a move) and stays allowed.
            if recreated {
                for t in &copy_sources {
                    if t.starts_with(d) {
                        return Err(Error::EditUnsupported(
                            "a copy in this commit reads from a path the same commit \
                             replaces; use separate commits",
                        ));
                    }
                }
            }
            // Past that return `recreated` means the whole touched subtree is
            // fresh, so an addition at or below `d` lands in the replacement. The
            // apply loop already removes a group's deleted links before appending
            // any new one (`remove_link_from_region` runs first), so a
            // replacement needs no further sequencing here.
            for t in &add_targets {
                if recreated && t.starts_with(d) {
                    continue;
                }
                if t.starts_with(d) || d.starts_with(t) {
                    return Err(Error::EditUnsupported(
                        "a deletion overlaps an addition in the same commit; \
                         replace the path instead, or use separate commits",
                    ));
                }
            }
            for t in &attr_targets {
                if recreated && t.starts_with(d) {
                    continue;
                }
                if t.starts_with(d) {
                    return Err(Error::EditUnsupported(
                        "a deletion overlaps a group-attribute edit in the same commit; use separate commits",
                    ));
                }
            }
            for t in &write_targets {
                if t.starts_with(d) {
                    // `write_targets` holds three kinds — a value overwrite, a
                    // dataset-attribute edit, and a staged append — so the
                    // message names what they have in common rather than only
                    // the first of them.
                    return Err(Error::EditUnsupported(
                        "a deletion overlaps a staged edit to a dataset in the same \
                         commit; use separate commits",
                    ));
                }
            }
            for (j, d2) in delete_targets.iter().enumerate() {
                if i != j && d2.starts_with(d) {
                    return Err(Error::EditUnsupported(
                        "overlapping deletions in one commit; delete the common parent only",
                    ));
                }
            }
            ensure_ancestors(&mut nodes, &parent);
            nodes.entry(parent).or_default().deletes.push(leaf);
        }

        // Resolve / validate each node's base object-header region up front.
        // Every existing dirty group is rewritten to a freshly-appended header,
        // so its old header becomes dead bytes once the superblock is repointed;
        // `superseded_addrs` records those old headers for reclamation (#21),
        // paired with the path whose rebuilt header replaces each — the two
        // halves of one relocation, which is what an object reference stored
        // elsewhere in the file has to be repointed across (issue #324).
        let keys: Vec<ObjectPathBuf> = nodes.keys().cloned().collect();
        let mut superseded_addrs: Vec<(ObjectPathBuf, u64)> = Vec::new();
        for key in &keys {
            let is_new = nodes[key].is_new;
            if is_new {
                nodes.get_mut(key).unwrap().base_region = group::fresh_group_region();
            } else {
                let addr = crate::group_v2::resolve_path_any_from_source(
                    &self.image(),
                    AccessMode::ReadWrite,
                    &self.superblock,
                    self.widths,
                    self.addresses,
                    &key.as_path(),
                )?;
                // Rebuilding this group moves its header and patches only the
                // link this commit resolved it through, so every other hard link
                // to it would be left naming the old header — which this commit
                // then frees. The aliases show the pre-commit group until
                // something reuses the span, and are unreadable after (issue
                // #327). Same rule, and the same lazily-computed count, as the
                // three relocating dataset writes above.
                //
                // The root group is exempt because it is named by the superblock
                // rather than by a link, so it has no entry in the count and
                // nothing to strand. Every other group was resolved by path and
                // therefore has at least one.
                if !key.is_empty() {
                    let counts = incoming_links
                        .get_or_insert_with(|| self.count_incoming_hard_links())
                        .as_ref();
                    match counts.and_then(|c| c.get(&addr).copied()) {
                        Some(1) => {}
                        // Known, and more than one: the aliases are the problem.
                        Some(_) => {
                            return Err(Error::EditUnsupported(
                                "editing a group relocates its object header; only supported \
                                 when it has a single hard link",
                            ));
                        }
                        // Not known: the file-wide walk that counts links gave up
                        // — an object header it could not parse, a group it could
                        // not enumerate, or a link graph past its bound. A
                        // different refusal because it is a different problem, and
                        // "it has more than one hard link" would send the reader
                        // looking for a second link that may not exist.
                        None => {
                            return Err(Error::EditUnsupported(
                                "editing a group relocates its object header, and this file's \
                                 links could not be walked to establish that nothing else \
                                 names it",
                            ));
                        }
                    }
                }
                let info = self.inspect_group(addr)?;
                superseded_addrs.push((key.clone(), addr));
                let node = nodes.get_mut(key).unwrap();
                node.base_region = info.region;
                node.existing_links = info.link_names;
            }
        }

        // A rebuilt group's old header is vacated just as a relocated dataset's is,
        // so the two join one list for the screen below.
        moved_headers.extend(superseded_addrs.iter().map(|&(_, a)| a));

        // Apply and validate group attribute edits before any writes. This keeps
        // unsupported attribute edits under the same all-or-nothing preflight
        // contract as unsupported dataset additions. A variable-length attribute
        // is not fully resolved here — its global heap collection is built (it
        // is self-contained, no address needed yet) but placed and patched into
        // `base_region` only in the apply loop below, once its address is known.
        let attrs_by_group = group_by_parent(staged.group_attrs().iter().map(|(p, op)| (p, op)));
        // The pre-commit header address of each group this commit rewrites. An
        // edit that sends a group's attributes to a fractal heap reads the set it
        // rebuilds back out of the file, which needs the address; a group this
        // commit *creates* has neither an address nor any stored attribute, and
        // its `None` says so.
        let existing_group_addrs: HashMap<ObjectPathBuf, u64> =
            superseded_addrs.iter().cloned().collect();
        for key in &keys {
            if let Some(ops) = attrs_by_group.get(key) {
                let region = std::mem::take(&mut nodes.get_mut(key).unwrap().base_region);
                let edits = attributes::plan_attr_ops(
                    &self.image(),
                    base,
                    existing_group_addrs.get(key).copied(),
                    &region,
                    ops,
                )?;
                let node = nodes.get_mut(key).unwrap();
                node.base_region = edits.region;
                node.attrs = edits.attrs;
                node.replaces_dense_storage = edits.replaces_dense_storage;
            }
        }

        // Map each node to its direct child group nodes, each with the link name
        // it has in that parent (for link wiring). The root group has no name in
        // a parent, so `split_leaf` gives `None` for it and the loop skips it.
        let mut children: BTreeMap<ObjectPathBuf, Vec<(ObjectPathBuf, LinkNameBuf)>> =
            BTreeMap::new();
        for key in &keys {
            if let Some((parent, leaf)) = key.split_leaf() {
                children
                    .entry(parent)
                    .or_default()
                    .push((key.clone(), leaf));
            }
        }

        // Each group's added datasets, in the order the apply loop will place
        // them, for the guards below to read. Borrowed from the staged set:
        // every one of those guards may still refuse, and a refusal gives the
        // caller back every edit it was holding (issue #316). The apply loop
        // rebuilds the same grouping, by the same rule, once it owns the set.
        let datasets_by_group = group_by_parent(staged.datasets().iter().map(|(p, fd)| (p, fd)));

        // A group that tracks *link* creation order takes an addition — the
        // apply loop numbers each added link from the running maximum its Link
        // Info message records — up to the point the addition would send its
        // links dense, which is refused before anything is written. The count
        // this screens is the one the group would hold afterwards: the links it
        // has now, less the ones this commit deletes, plus the ones it adds.
        for key in &keys {
            let node = &nodes[key];
            let added = node.copies.len()
                + node.cross_copies.len()
                + datasets_by_group.get(key).map_or(0, Vec::len)
                + children.get(key).map_or(0, |kids| {
                    kids.iter().filter(|(child, _)| nodes[child].is_new).count()
                });
            if added > 0 {
                let kept = node
                    .existing_links
                    .iter()
                    .filter(|name| !node.deletes.iter().any(|d| d.as_str() == *name))
                    .count();
                group::reject_dense_link_creation_order(&node.base_region, kept + added)?;
            }
        }

        // Validate names: no addition may collide with an existing link or with
        // another addition under the same parent. A link this same commit
        // deletes is not one of the existing ones — the apply loop removes it
        // from the region before any addition is appended, so replacing an
        // object at its own path is a rotation, not a collision (issue #305).
        for key in &keys {
            let node = &nodes[key];
            let mut adding: Vec<&str> = Vec::new();
            for fd in datasets_by_group.get(key).into_iter().flatten() {
                adding.push(fd.name.as_str());
            }
            for (child, leaf) in children.get(key).into_iter().flatten() {
                if nodes[child].is_new {
                    adding.push(leaf.as_str());
                }
            }
            for (leaf, _) in &node.copies {
                adding.push(leaf.as_str());
            }
            for leaf in &node.cross_copies {
                adding.push(leaf.as_str());
            }
            for (i, name) in adding.iter().enumerate() {
                let survives = node.existing_links.iter().any(|n| n == name)
                    && !node.deletes.iter().any(|n| n.as_str() == *name);
                if survives || adding[..i].contains(name) {
                    return Err(Error::EditUnsupported(
                        "a link with this name already exists in the target group",
                    ));
                }
            }
        }

        // Content the caller's library-version bound cannot carry is refused
        // here, before any write, so a rejected addition leaves the commit
        // unapplied.
        self.check_libver_admits(staged.datasets().iter().map(|(_, fd)| fd))?;

        // Enumerate what this commit's deletions reclaim, from the current
        // on-disk layout and before any byte moves. It is read here, ahead of
        // every remaining refusal, because two of them screen against it: an
        // object reference this commit writes must not name space the same
        // commit frees (issue #317). The spans are carried to `to_free` below
        // rather than walked a second time, so the screen and the allocator
        // always mean the same thing by "removed".
        //
        // An object's storage is reclaimed only when the link being removed is
        // its LAST hard link: HDF5 objects can have several hard links, and one
        // reachable through a surviving link is still live (freeing it would
        // corrupt the survivor). Count every hard link in the pre-commit file
        // and reclaim a deleted object only when its count is exactly 1.
        // `deleted_addrs` is de-duplicated first so two delete paths that are
        // hard links to the same object are not visited (and freed) twice. If
        // the link graph cannot be walked in full, no deleted object is
        // reclaimed (a safe leak) — and none is screened either, which is sound
        // in the same direction: nothing is freed, so no reference dangles.
        // Superseded group headers and relocated dataset headers are vacated too,
        // and are screened separately: `moved_headers` holds them, and they are
        // addresses rather than spans.
        let mut deleted_free: Vec<(u64, u64, FreeClass)> = Vec::new();
        deleted_addrs.sort_unstable();
        deleted_addrs.dedup();
        if !deleted_addrs.is_empty() {
            if let Some(incoming) = self.count_incoming_hard_links() {
                for &a in &deleted_addrs {
                    self.collect_free_spans(a, 0, &incoming, &mut deleted_free);
                }
            }
        }
        let deleted_collections = self.deleted_collections(&deleted_addrs);
        // An in-file copy re-emits its source's element bytes verbatim, so a
        // copied object reference keeps naming whatever it named in the source —
        // including an object this same commit is removing (issue #317). The page
        // types the walk records are for the allocator, not for this.
        //
        // Removals only, and this is the one place the two halves part company. A
        // *supplied* address is a fresh claim about the file, and a commit that
        // moves the object falsifies it. A *copied* one repeats a claim the file
        // already made, and since issue #324 a move no longer falsifies either
        // one: the commit's last act walks the tree it published and repoints
        // every reachable stored address, the copy's among them
        // ([`crate::reference_patch`], and `a_copy_made_in_the_same_commit_that_
        // moves_its_target_is_repointed`). A removal is the case that stays,
        // because there is no new address to point at.
        //
        // Refusing on `moved` here would also cost what it cannot buy: every
        // commit that reaches here rebuilds its root group, so `moved` is never
        // empty, and a chunked object-reference dataset is refused on a non-empty
        // screen alone rather than on a matching address — so every such copy
        // would be refused, in every commit.
        let for_copied = InvalidatedAddresses {
            removed: deleted_free
                .iter()
                .map(|&(off, len, _)| (off, len))
                .collect(),
            moved: Vec::new(),
            base: self.addresses,
        };
        {
            // Framed at the base address: a shared-message address is stored
            // relative to it, and `SourceResolver` reads its references as
            // absolute within the view it is given.
            let image = self.image();
            let framed = BaseOffsetSource {
                inner: image,
                base: self.addresses.effective_base(),
            };
            for key in &keys {
                for (_, tree) in &nodes[key].copies {
                    copy::screen_copied_references(
                        tree,
                        self.addresses,
                        self.widths,
                        &for_copied,
                        &framed,
                    )?;
                }
            }
        }

        // The same screen the copies just got, plus the half a supplied address
        // earns and a copied one does not.
        let for_supplied = InvalidatedAddresses {
            moved: moved_headers,
            ..for_copied
        };

        // A shared-message index record can hold an object-header address too,
        // and it is the one stored address the commit's closing repoint walk
        // cannot reach. Screen it here, against the same two lists.
        self.screen_shared_message_index(&for_supplied)?;

        // References a builder already resolved to addresses never reach
        // `resolve_reference_target`, so they are screened out of the element
        // bytes instead — additions and value overwrites alike, since neither
        // form's bytes are touched again before they are written. A slot still
        // holding a placeholder screens as the zero it is, and is screened for
        // real by `resolve_reference_target` when it resolves.
        //
        // Cross-file copies need no screen for the opposite reason:
        // `reject_foreign_addresses` refuses a reference datatype outright on
        // that path.
        for (_, fd) in staged.datasets().iter().chain(staged.writes()) {
            screen_resolved_references(&fd.dt, &fd.raw, &for_supplied)?;
        }

        // Anything this commit adds could be a reference container the last walk
        // did not see, so it retires that walk's "no references in this file"
        // finding (issue #324). See `proved_free_of_references` for which of
        // these four clauses is load-bearing and which are belt-and-braces, and
        // for why the other five staged collections need no clause at all.
        //
        // A staged dataset's *attributes* are screened beside its element
        // datatype. Neither can carry a reference today — a committed one is
        // refused by `flatten_dataset`, and `AttrValue` has no reference variant
        // — but that is a refusal in another function rather than a property of
        // this collection, and the check that means what it says is the one that
        // looks.
        if staged
            .datasets()
            .iter()
            .chain(staged.writes())
            .any(|(_, fd)| {
                datatype::datatype_holds_object_address(&fd.dt)
                    || fd
                        .attrs
                        .iter()
                        .any(|a| datatype::datatype_holds_object_address(&a.datatype))
            })
            || !staged.copies().is_empty()
            || !staged.cross_copies().is_empty()
        {
            self.proved_free_of_references = false;
        }

        // Prove every object-reference target resolves before any write (see
        // `preflight_reference_targets`'s doc comment): otherwise a reference
        // resolution failure discovered mid-apply-loop would leave every
        // earlier-processed group's real writes (headers, data, copied
        // subtrees) orphaned in the file despite `commit()` returning `Err`.
        Self::preflight_reference_targets(
            &keys,
            &datasets_by_group,
            &nodes,
            &add_targets,
            &write_targets,
            delete_targets,
            &for_supplied,
            &self.image(),
            &self.superblock,
            self.widths,
            self.addresses,
        )?;

        if self.fresh_page_boundary.is_some() {
            self.preflight_fresh_pages(staged, &nodes)?;
        }

        // --- The point of no return. Every refusal is behind us, so the staged
        // set is taken here rather than at the top of this function: a refusal
        // above restores *every* edit the caller staged, and a failure below —
        // an I/O error mid-apply, which leaves the file valid because the
        // superblock is repointed last — restores none of them, so no later
        // `commit` can finish a batch this one abandoned (issue #316). Nothing
        // above this line may consume from `staged`. ---
        let mut taken = std::mem::take(staged).into_apply();
        // The same paths as the borrow above, owned now that the set has moved.
        let delete_targets = std::mem::take(&mut taken.deletes);
        // The same grouping as `datasets_by_group` above, by the same rule and
        // so in the same order — which is what keeps the apply loop's placement
        // order the one `preflight_reference_targets` proved.
        let mut flat = group_by_parent(taken.datasets.drain(..));
        // A cross-file copy's subtree can move now, so it joins its destination
        // group's in-file copies. The parent node exists: the preflight made one
        // for every destination.
        for (dst, tree) in taken.cross_copies.drain(..) {
            // A destination that identifies no link is rejected above.
            let (parent, leaf) = dst.split_leaf().expect("a copy destination names a link");
            nodes.get_mut(&parent).unwrap().copies.push((leaf, tree));
        }

        // Gather the regions this commit will vacate, read from the current
        // on-disk layout before any byte moves: every deleted object's owned
        // blocks plus every superseded group header. They stay out of the free
        // list until every append this commit makes is behind it, so none of
        // them is reused while it is still live. (On the non-persisting tail
        // that point is a few lines *before* the superblock write rather than
        // after it — the apply loop is where the appends happen, and it is long
        // done by then.) Enumeration is
        // best-effort — `collect_free_spans` simply omits anything it cannot
        // account for exhaustively, so the worst case is unreclaimed dead bytes,
        // never a freed-but-live region.
        // It starts as the deleted objects' blocks, enumerated above the preflight
        // because a refusal there screens against them.
        let mut to_free: Vec<(u64, u64, FreeClass)> = deleted_free;

        // A superseded group header is dead once the root is repointed. Its chunk
        // spans are enumerated base-aware (`oh_chunk_spans` shifts continuation
        // addresses by the userblock base and returns absolute file offsets), as is
        // the delete path (`collect_free_spans`), so all of this reclamation works
        // on userblock files too.
        for (key, a) in &superseded_addrs {
            let a = *a;
            if let Ok(spans) = self.oh_chunk_spans(a) {
                if nodes[key].replaces_dense_storage
                    && let Ok(region) = Self::gather_oh_messages(&self.image(), a, self.addresses)
                    && let Some(storage) = self.dense_attribute_storage_spans(&region, &spans)
                {
                    to_free.extend(storage);
                }
                to_free.extend(self.object_header_free_spans(&spans)?);
            }
        }

        // A relocating overwrite (`write_dataset` resize, or any compact rewrite)
        // vacates the dataset's old object header, and a resized contiguous one
        // also vacates its old data block: both become dead once the parent's
        // relinked header lands. `superseded_addrs` covers only the rebuilt group
        // headers, not the relocated dataset's own header, so record that here too.
        // The pre-commit dataset-header address rides on the write plan (`old_oh`),
        // recorded where the plan was made; its chunks and old data extent are
        // freed only after the superblock repoint.

        // The single-hard-link guard in the write preflight makes freeing the old
        // header safe (no surviving link still points at it).
        for key in &keys {
            for (_leaf, old_oh, mw) in &nodes[key].writes {
                match mw {
                    MovingWrite::Contiguous {
                        old_extent: Some(extent),
                        ..
                    } => to_free.push((extent.0, extent.1, FreeClass::Page(PageType::Raw))),
                    // A relocated chunked dataset vacates its old chunk index and
                    // chunk data blocks. `chunked_storage_spans` returns `None` for
                    // anything it cannot enumerate exhaustively (leaving dead bytes
                    // rather than freeing a region still in use); the old header
                    // chunks are freed generically below.
                    MovingWrite::Chunked { old_addr, .. } => {
                        if let Some(spans) = self.chunked_storage_spans(*old_addr) {
                            to_free.extend(spans);
                        }
                    }
                    // A relocating append keeps the existing chunk *data* in place
                    // (shared by both indexes during the commit), so only the old
                    // index structure and the relocated old trailing chunk are dead.
                    // The old header chunks are freed by the generic path below.
                    MovingWrite::AppendedChunks {
                        old_addr,
                        old_tail_extent,
                        kept_chunks,
                        ..
                    } => {
                        if let Some(spans) = self.chunked_index_spans(*old_addr) {
                            // The old index is reclaimed as raw only where it
                            // provably sits in a raw page, and recorded as dead
                            // otherwise. See `index_is_provably_raw`. The
                            // dataset's old chunk data is the kept chunks
                            // (base-relative) plus the trailing partial chunk
                            // this append relocated, both already in hand, so
                            // the proof needs no second walk of the index.
                            let data: Vec<(u64, u64)> = kept_chunks
                                .iter()
                                .filter_map(|c| {
                                    Some((base.absolute(c.address).ok()?, c.stored_size))
                                })
                                .chain(*old_tail_extent)
                                .collect();
                            let class = if self.index_is_provably_raw(&data, &spans) {
                                FreeClass::Page(PageType::Raw)
                            } else {
                                FreeClass::Dead
                            };
                            to_free.extend(spans.into_iter().map(|(a, l)| (a, l, class)));
                        }
                        if let Some(ext) = old_tail_extent {
                            // The relocated old trailing chunk is raw data.
                            to_free.push((ext.0, ext.1, FreeClass::Page(PageType::Raw)));
                        }
                    }
                    _ => {}
                }
                // The relocated dataset's old header chunks are dead too. An attribute edit of
                // an already-dense set also replaces the set's version 2 B-tree indexes and its
                // fractal heap.
                if let Ok(spans) = self.oh_chunk_spans(*old_oh) {
                    if matches!(
                        mw,
                        MovingWrite::AttrEdit {
                            replaces_dense_storage: true,
                            ..
                        }
                    ) && let Ok(region) =
                        Self::gather_oh_messages(&self.image(), *old_oh, self.addresses)
                        && let Some(storage) = self.dense_attribute_storage_spans(&region, &spans)
                    {
                        to_free.extend(storage);
                    }
                    to_free.extend(self.object_header_free_spans(&spans)?);
                }
            }
        }

        // Defense in depth: never hand the free list an out-of-bounds or
        // overlapping span. The last-link guard plus the per-object checks
        // should already make the accumulated spans disjoint; this enforces it
        // as a whole-commit invariant against the pre-commit end-of-file. Any
        // dropped span (which should not occur for a well-formed file) only
        // leaks, never corrupts.
        retain_disjoint_in_bounds(&mut to_free, self.image.len());

        // --- Apply: process deepest groups first so each parent sees its
        // children's new addresses, then repoint the superblock last.
        // `path_addr` accumulates every group's and dataset's address as it is
        // placed — read by `resolve_reference_target` to resolve a same-commit
        // object-reference target (see the dataset-placement loop below for the
        // group/dataset key convention: a group's own path, or a dataset's
        // full parent+name path). ---
        let mut path_addr: BTreeMap<ObjectPathBuf, u64> = BTreeMap::new();
        // Where each object-header address this commit vacates has been rewritten
        // to, for repointing the object references the rest of the file already
        // stores (issue #324). Filled from the two places a header moves: a
        // relocating value overwrite, recorded in the loop below as its plan is
        // applied, and a rebuilt group, joined from `superseded_addrs` once every
        // group's new address is known.
        let mut relocations: BTreeMap<u64, u64> = BTreeMap::new();
        let mut by_depth = keys.clone();
        // Stable on purpose: `keys` comes from a `BTreeMap`, so equal-depth
        // groups arrive in path order, and this order is the order they are
        // placed in — which fixes every address the commit writes. An unstable
        // sort may permute a depth group, and the file's layout with it.
        by_depth.sort_by_key(|k| std::cmp::Reverse(k.len())); // deepest first
        for key in &by_depth {
            let (mut region, deletes, copies, writes, attrs) = {
                let node = nodes.get_mut(key).unwrap();
                (
                    std::mem::take(&mut node.base_region),
                    std::mem::take(&mut node.deletes),
                    std::mem::take(&mut node.copies),
                    std::mem::take(&mut node.writes),
                    std::mem::take(&mut node.attrs),
                )
            };

            // Remove deleted links first (verbatim-preserving the rest).
            //
            // First is a correctness requirement rather than a convention, and
            // has been since a replacement became one commit (issue #305):
            // `remove_link_from_region` matches by *name*, so a removal running
            // after the additions below would take the replacement's link with
            // the original's and leave the path gone. Every link this loop could
            // collide with — a copy's, a dataset's, a new child group's — is
            // appended after it, which is what keeps that unreachable. (A
            // relocating write and an existing child group *patch* a link rather
            // than appending one, so neither can be a replacement's.)
            for name in &deletes {
                region = group::remove_link_from_region(&region, name.as_str())?;
            }

            // Every link appended below takes the next creation index from this
            // group's running maximum, on the groups that track link creation
            // order and on no others; `link_order.record` writes the bumped
            // maximum back once the last of them is placed. Read after the
            // deletions above because a deletion leaves the maximum alone — the
            // gap it opens is never handed out again.
            let mut link_order = LinkCreationOrder::for_region(&region)?;

            // Write each staged source subtree and link its root into this group.
            for (leaf, tree) in copies {
                let root = self.write_copy_subtree(&tree)?;
                region.push_link(leaf.as_str(), base.relative(root)?, link_order.take()?);
            }

            // Datasets directly under this group. Appended addresses are absolute
            // file offsets, and the contiguous data-layout address is stored
            // relative to the base address. Placed
            // non-reference datasets first (recording each into `path_addr`), then
            // reference datasets — a reference to a *non-reference* sibling added
            // in the same group's batch resolves regardless of staging
            // order (`Vec::sort_by_key` is stable, so within each of the two
            // groups the original order is preserved). Two reference datasets that
            // target each other in the same batch are still call-order-dependent —
            // whichever is placed first resolves the other, and the reverse
            // direction is safely refused as "still writing" (never corrupted),
            // caught up front by `preflight_reference_targets`.
            let mut group_datasets: Vec<FlatDataset> =
                flat.remove(key).into_iter().flatten().collect();
            // Stable on purpose: this is a partition, and the relative order
            // within each half is what the paragraph above promises.
            group_datasets.sort_by_key(|fd| fd.reference_targets.is_some());
            for mut fd in group_datasets {
                self.resolve_dataset_attributes(&mut fd)?;
                fd.resolve_references(|target| {
                    Self::resolve_reference_target(
                        target,
                        &path_addr,
                        &nodes,
                        &add_targets,
                        &write_targets,
                        &delete_targets,
                        &for_supplied,
                        &self.image(),
                        &self.superblock,
                        self.widths,
                        self.addresses,
                    )
                })?;
                let oh_addr = self.write_dataset(&mut fd)?;
                region.push_link(
                    fd.name.as_str(),
                    base.relative(oh_addr)?,
                    link_order.take()?,
                );
                path_addr.insert(key.join(&fd.name), oh_addr);
            }

            // Relocating value overwrites under this group: write the new data and
            // rewritten header, then patch this group's existing link to it. On a
            // userblock file only the chunked variant reaches here (contiguous and
            // compact resizes are refused in the write preflight).
            for (leaf, old_oh, mw) in &writes {
                let new_oh = self.write_moving(mw)?;
                group::patch_link_target(&mut region, leaf.as_str(), base.relative(new_oh)?)?;
                relocations.insert(*old_oh, new_oh);
            }

            // Wire links to dirty child groups (new → add a link; existing →
            // patch the existing link to the child's new address).
            for (child, leaf) in children.get(key).into_iter().flatten() {
                let child_name = leaf.as_str();
                let child_addr = base.relative(path_addr[child])?;
                if nodes[child].is_new {
                    region.push_link(child_name, child_addr, link_order.take()?);
                } else {
                    group::patch_link_target(&mut region, child_name, child_addr)?;
                }
            }

            // Every addition to this group is placed, so its Link Info message
            // can record the maximum creation index it has now assigned. A group
            // that gained no link, or that does not track the order, is left
            // byte-identical.
            link_order.record(&mut region)?;

            // Whatever this group's attribute edits left to place: a
            // variable-length attribute's heap collection, or a whole dense set.
            self.place_edited_attrs(&mut region, attrs)?;

            let oh = object_header::build_v2_object_header(&region)?;
            let addr = self.alloc_or_append_typed(&oh, AllocationClass::Metadata)?;
            path_addr.insert(key.clone(), addr);
        }

        // Prepared in-place bytes require no heap allocation. Journal existing data before writing
        // it.
        for (data_addr, bytes) in &inplace_writes {
            self.write_inplace_journaled(*data_addr, bytes)?;
        }

        // Every group's new header address is known now that the apply loop has
        // run, so the rebuilt groups can join the relocation map. A group whose
        // key is missing would be one the loop did not place, which cannot happen:
        // `superseded_addrs` is filled from `keys`, and the loop visits all of it.
        for (key, old) in &superseded_addrs {
            if let Some(&new) = path_addr.get(key) {
                relocations.insert(*old, new);
            }
        }

        // Repoint the superblock at the new root last: this is the commit's
        // linearization point. Until it lands, the file on disk still points at
        // the old root (the appended objects are merely unreferenced trailing
        // bytes), so a failure here leaves a valid file.
        //
        // That ordering is only crash-safe if the appended objects are durable
        // before the root pointer is flipped; otherwise a power loss could
        // persist the flip ahead of the data it references, leaving the root
        // pointing at bytes that never reached disk. `flush` on a plain `File`
        // does not force a write-back, so sync the appended bytes to disk first
        // (the barrier), then flip the pointer, then sync the flip.
        let new_root = path_addr[&ObjectPathBuf::root()];
        to_free.extend(self.unreferenced_collection_spans(new_root, deleted_collections));

        // Keep superseded collections unavailable until the new root is durable. These collections
        // were placed as raw allocations by this session, so their PAGE free-space type is known.
        to_free.extend(
            self.superseded_heaps
                .drain(..)
                .map(|(addr, len)| (addr, len, FreeClass::Page(PageType::Raw))),
        );
        // Re-run the whole-commit invariant over the set these just joined, since
        // the pass above ran before the in-place writes resolved and so never saw
        // them. This is defensive and nothing reaches it: no test fails when it
        // is deleted, and no sequence has been found that puts a superseded
        // collection out of bounds or across another freed span. It is kept
        // because a heap collection is freed on the strength of a *record*
        // rather than of a structure read back out of the file, which is the one
        // span where a stale answer would not be caught by anything else.
        retain_disjoint_in_bounds(&mut to_free, self.image.len());
        let mut to_free = self.stored_spans(to_free)?;

        // A persisting file keeps its freed space recorded on disk rather than
        // truncating it away, so its commit takes a different tail: one that
        // rewrites the on-disk managers rather than only repointing the superblock.
        if self.persist.is_some() {
            let root_reclamation = self.plan_root_reclamation(new_root, &mut to_free)?;
            self.commit_persisting_releasing_reserve(new_root, to_free)?;
            if root_reclamation.is_some() {
                self.proved_free_of_references = true;
            }
            self.repoint_stored_references(&relocations)?;
            if let Some(reclamation) = root_reclamation {
                self.reclaim_root_metadata(reclamation)?;
            }
            return Ok(());
        }

        if self.fresh_page_boundary.is_some() {
            self.pad_to_page()?;
            return self.publish_nonpersisting(new_root, None);
        }

        let mut trunc_to = self.release_freed(to_free)?;
        if let Some(aggregation) = self.aggregation {
            let end = self
                .addresses
                .resolve(StoredAddress::new(aggregation.eoa()))?;
            if end.get() < self.image.len() {
                trunc_to = Some(end);
            }
        }
        self.publish_nonpersisting(new_root, trunc_to)?;
        self.repoint_stored_references(&relocations)
    }

    /// Proves that a non-persistent PAGE commit can publish without patching stored references.
    ///
    /// Checks the existing tree and prospective attribute storage before consuming staged edits.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] for object references, unsupported attribute storage,
    /// or an incomplete reference inspection. Returns address-resolution and object-header record
    /// errors from [`prove_free_of_references`](crate::reference_patch::prove_free_of_references).
    fn preflight_fresh_pages(
        &self,
        staged: &StagedEdits,
        nodes: &BTreeMap<ObjectPathBuf, Node>,
    ) -> Result<(), Error> {
        let unsupported_dataset =
            staged
                .datasets()
                .iter()
                .chain(staged.writes())
                .any(|(_, dataset)| {
                    dataset.attrs_are_dense
                        || datatype::datatype_holds_object_address(&dataset.dt)
                        || dataset
                            .attrs
                            .iter()
                            .any(|attr| datatype::datatype_holds_object_address(&attr.datatype))
                });
        // Same-file copies come from the tree checked below. Cross-file copies reject file
        // addresses when staged, so their datatypes cannot add object references.
        if unsupported_dataset
            || staged
                .cross_copies()
                .iter()
                .any(|(_, tree)| tree.has_dense_attributes())
            || nodes.values().any(|node| {
                matches!(node.attrs, EditedAttrs::Dense(_))
                    || node.writes.iter().any(|(_, _, write)| {
                        matches!(
                            write,
                            MovingWrite::AttrEdit {
                                attrs: EditedAttrs::Dense(_),
                                ..
                            }
                        )
                    })
            })
            || !crate::reference_patch::prove_free_of_references(
                &self.image(),
                &self.superblock,
                self.widths,
                self.addresses,
                MAX_LINK_GRAPH_NODES,
            )?
        {
            return Err(Error::EditUnsupported(
                "staged edits of PAGE files without persisted free space require a complete \
                 object-reference-free tree; repack the file to edit reference-bearing or \
                 unsupported attribute storage",
            ));
        }
        Ok(())
    }

    /// Publishes the root and EOA after the content barrier, then applies any requested truncation.
    ///
    /// A non-persistent PAGE commit retains its File Space Info message and passes no truncation.
    ///
    /// # Errors
    ///
    /// Returns the address-conversion, serialization, write, synchronization, or truncation error.
    fn publish_nonpersisting(
        &mut self,
        new_root: u64,
        trunc_to: Option<PhysicalAddress>,
    ) -> Result<(), Error> {
        let base = self.addresses;
        let new_eof = trunc_to.map_or(self.image.len(), PhysicalAddress::get);
        self.barrier()?;
        if matches!(
            self.superblock.version,
            SuperblockVersion::Two | SuperblockVersion::Three
        ) {
            // Build the new superblock off a clone and adopt it only once the
            // write succeeds, so a failed write does not desync the in-memory
            // state. The v2/v3 superblock carries its own checksum.
            let mut new_sb = self.superblock.clone();
            new_sb.root_group_address = base.relative(new_root)?;
            new_sb.eof_address = new_eof;
            // Publish the flags this session *raised*, so a flag the source file
            // arrived carrying (left set by a crashed SWMR writer, say) is
            // scrubbed and this clean commit leaves the file properly closed for
            // the C library (issue #73). serialize() recomputes the v2/v3
            // checksum.
            //
            // Non-persistent PAGE sessions reject page buffering and use zero status flags.
            new_sb.consistency_flags = self.held_status_flags;
            let sb_bytes = hdf5_pure_format::__private::serialize_superblock(&new_sb)?;
            self.publish_attempted = true;
            self.write_at(self.sb_sig_off, &sb_bytes)?;
            self.barrier()?;
            self.superblock = new_sb;
        } else {
            self.publish_attempted = true;
            self.repoint_v0v1_root(base.relative(new_root)?, new_eof)?;
            self.barrier()?;
            self.superblock.root_group_address = self.addresses.relative(new_root)?;
            self.superblock.eof_address = new_eof;
        }

        // Physically shrink the file only after the superblock — now carrying the
        // smaller end-of-file — is durable. A crash between the two leaves a file
        // whose superblock end-of-file is correct and whose trailing bytes are
        // mere unreferenced slack, which the next open ignores; the reverse order
        // could advertise an end-of-file past the actual file length.
        if let Some(cut) = trunc_to {
            self.image.truncate(cut.get())?;
            self.barrier()?;
        }
        Ok(())
    }

    /// Records the regions a non-persisting commit vacated under the active reusable-space
    /// strategy, and returns the file position to truncate the file to when the flat free run
    /// reaches the end of allocation.
    ///
    /// Disabled strategies discard the regions. A non-persistent PAGE commit skips this step
    /// and retains every freed extent without making it reusable.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressBelowBase`] if the image ends below the base address, and
    /// [`FormatError::OffsetOverflow`] if the truncation point plus the base address exceeds
    /// [`u64::MAX`]. Returns [`FormatError::Internal`] if aggregate state exceeds EOA or the
    /// flat reusable-space strategy changes during release.
    fn release_freed(
        &mut self,
        to_free: Vec<(StoredAddress, u64, FreeClass)>,
    ) -> Result<Option<PhysicalAddress>, FormatError> {
        let eoa = self.end_of_allocation()?;
        if let Some(aggregation) = &mut self.aggregation {
            return aggregation
                .release_freed(
                    &mut self.space,
                    to_free
                        .into_iter()
                        .filter_map(|(addr, len, _)| FreeExtent::new(addr.get(), len)),
                )?
                .map(|cut| self.addresses.resolve(StoredAddress::new(cut)))
                .transpose();
        }
        self.space
            .release_freed(
                eoa.get(),
                to_free
                    .into_iter()
                    .filter_map(|(addr, len, _)| FreeExtent::new(addr.get(), len)),
            )
            .map(|cut| self.addresses.resolve(StoredAddress::new(cut)))
            .transpose()
    }

    /// Returns `spans` with each file position converted to a stored address.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressBelowBase`] if a span starts below the base address.
    fn stored_spans(
        &self,
        spans: Vec<(u64, u64, FreeClass)>,
    ) -> Result<Vec<(StoredAddress, u64, FreeClass)>, FormatError> {
        spans
            .into_iter()
            .map(|(at, len, class)| Ok((self.addresses.relative(at)?, len, class)))
            .collect()
    }

    /// Repoint every object reference the file already stores at a header this
    /// commit moved (issue #324).
    ///
    /// Called at the end of each commit tail, *after* the superblock repoint,
    /// which is where it belongs on both counts. The commit is atomic at the
    /// repoint and this is a fixup derived from it, so it must not run earlier:
    /// a crash before the repoint has to leave the pre-commit file, its stored
    /// references included. And the tree it walks has to be the committed one —
    /// a rebuilt group's header is a fresh copy of the pre-commit one and
    /// carries the old address of anything it referenced, so patching the
    /// superseded header instead would correct bytes that are already dead.
    ///
    /// The writes are followed by a [`barrier`](Self::barrier), so a commit that
    /// returns under [`SyncPolicy::Always`] has put its references on disk and
    /// not merely its tree. That rests on `barrier`'s own contract rather than
    /// on a crash test — nothing here sweeps the window — and it costs nothing
    /// on the files that do not need it: a plan with nothing in it returns
    /// first, and on a file holding no object reference there is never anything
    /// in it.
    ///
    /// A crash between the repoint and that barrier leaves the commit standing
    /// with some references still stale, which is exactly the state every commit
    /// left before this existed.
    ///
    /// Bounded by [`MAX_LINK_GRAPH_NODES`], the same budget
    /// [`count_incoming_hard_links`](Self::count_incoming_hard_links) walks the
    /// link graph under.
    fn repoint_stored_references(&mut self, relocations: &BTreeMap<u64, u64>) -> Result<(), Error> {
        if self.proved_free_of_references {
            return Ok(());
        }
        let plan = crate::reference_patch::plan(
            &self.image(),
            &self.superblock,
            self.widths,
            self.addresses,
            relocations,
            MAX_LINK_GRAPH_NODES,
        )?;
        // Record the walk's verdict before applying it, and only ever the
        // positive proof: a walk that found a reference, or could not read some
        // object, leaves the question open rather than answering "yes there are
        // references", because the next commit's answer could differ.
        if plan.proved_free_of_references() {
            self.proved_free_of_references = true;
        }
        if plan.is_empty() {
            return Ok(());
        }
        plan.apply(self)?;
        self.barrier()
    }

    /// Publishes `new_root` with replacement persistent free-space metadata.
    ///
    /// Writes the extension and manager blocks as one contiguous allocation according to
    /// `placement`. [`TailPlacement::Preserve`] excludes the current allocation from returned
    /// free space. [`TailPlacement::Recycle`] can reuse a reserved allocation after an earlier
    /// publication made it unreferenced. Returned extents follow ordinary threshold admission.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if address resolution, metadata serialization, or aggregate
    /// reconciliation fails. Returns [`Error::EditUnsupported`] if the extension cannot be
    /// rewritten. Returns [`Error::Io`] if writing, synchronizing, or truncating fails. For PAGE,
    /// returns the errors from [`Self::commit_persisting_paged`].
    ///
    /// # Panics
    ///
    /// Panics if persistence is disabled.
    fn commit_persisting(
        &mut self,
        new_root: u64,
        to_free: Vec<(StoredAddress, u64, FreeClass)>,
        placement: TailPlacement,
    ) -> Result<(), Error> {
        // A paged file records its free space in per-page-type managers and keeps
        // its allocation page-aligned, so it takes its own tail (issue #198).
        if self.space.is_paged() {
            return self.commit_persisting_paged(new_root, to_free, placement);
        }
        let widths = self.widths;
        let offset_width = widths.offsets;
        let (strategy, threshold, page_size, mut old_blocks) = {
            // Copy what we need so no borrow of `self.persist` is held across the
            // `&mut self` writes below; the old state stays in place so a failure
            // leaves the session reusable.
            let ps = self
                .persist
                .as_ref()
                .expect("commit_persisting is only called when persistence is armed");
            (
                ps.strategy,
                ps.threshold,
                ps.page_size,
                ps.old_blocks.clone(),
            )
        };
        if placement == TailPlacement::Preserve {
            old_blocks.clear();
        }
        let base = self.addresses;

        // The extension address is stored relative to the base address, and every
        // reader of the extension below indexes the image, so it is shifted once
        // here for all of them.
        let old_ext_addr = base.absolute(
            self.superblock
                .superblock_extension_address
                .filter(|a| !a.is_undefined(offset_width))
                .ok_or(Error::EditUnsupported(
                    "a persisting file has no superblock extension to update",
                ))?,
        )?;

        // The persist File Space Info message is fixed-size, so the rewritten
        // extension's length is independent of the addresses it will carry: size
        // it with a placeholder to place the FSM blocks that follow it.
        let placeholder = file_space_info::persistent_single_manager(
            offset_width,
            strategy,
            threshold,
            page_size,
            0,
            0,
        );
        let ext_len = object_header::build_v2_object_header(
            &self.rewrite_extension_region(old_ext_addr, &placeholder)?,
        )?
        .len() as u64;

        // Place the tail — the rewritten extension and the manager blocks — in a
        // hole an *earlier* commit freed where one fits, and at end-of-file where
        // none does. `self.space` holds only durable reusable space. This commit's own frees stay
        // in `to_free` until the repoint, so a reused span holds bytes
        // already unreachable from the on-disk root, which is the guarantee
        // [`reserve`](Self::reserve) rests on too.
        let current_eoa = self.end_of_allocation()?.get();
        let limit = (placement == TailPlacement::ReuseOnly)
            .then(|| old_blocks.iter().map(|extent| extent.start()).min())
            .flatten();
        let recycled = if let TailPlacement::Recycle(target) = placement {
            let layout =
                self.recycled_tail_layout(current_eoa, &to_free, &old_blocks, ext_len, target);
            if layout.is_none() {
                old_blocks.push(target);
            }
            layout
        } else {
            None
        };
        let (post, placed_at, tail_len, eoa) = recycled.unwrap_or_else(|| {
            self.flat_tail_layout(current_eoa, &to_free, &old_blocks, ext_len, widths, limit)
        });
        if placed_at.is_none() && placement == TailPlacement::ReuseOnly {
            // The shrink pass appends nothing: growing the file by a tail is the
            // opposite of what it was called for. Nothing has been written and the
            // layout handed its reservation back, so the session is exactly as the
            // commit before it left it.
            return Ok(());
        }
        let reused = placed_at.is_some();
        let ext_addr = placed_at.unwrap_or(current_eoa);
        let managers = ManagerSections::flat(post.extents().iter().copied());
        let plan = free_space_manager::plan_managers(
            &managers,
            StoredAddress::new(ext_addr).offset(ext_len),
            widths,
        );
        // For a reused tail, truncate to the planned EOA after publication. The post-commit
        // free list excludes dropped trailing regions and the suffix released by
        // `FreeList::release_trailing_run`. An appended tail ends the file itself.
        let final_eoa = if reused { eoa } else { ext_addr + tail_len };
        debug_assert!(
            ext_addr + tail_len <= final_eoa,
            "the tail must end at or below the end-of-allocation it was placed under"
        );
        let final_eof = self.addresses.resolve(StoredAddress::new(final_eoa))?;
        let aggregation = self
            .aggregation
            .map(|state| state.reconcile(final_eoa))
            .transpose()?;

        let flat_block = match plan.blocks.as_slice() {
            [] => None,
            [block] if block.kind == ManagerKind::Flat => Some(block),
            [block] => {
                return Err(Error::Format(FormatError::Internal(format!(
                    "flat free-space plan produced a {:?} manager",
                    block.kind
                ))));
            }
            _ => {
                return Err(Error::Format(FormatError::Internal(
                    "flat free-space plan produced more than one manager".into(),
                )));
            }
        };

        // Build the real extension and the FSM blocks. With no free space to
        // record we still refresh the extension (persist on, managers undefined).
        let ext_oh = if let Some(block) = flat_block {
            // `eoa_pre_fsm` is the end-of-allocation before the free-space-manager
            // section blocks (`FSHD`/`FSSE`) were allocated: a consumer may shrink
            // back to here and rebuild them. For an appended tail that is the FSHD,
            // not the extension — the extension sits below it and persists, so
            // shrinking leaves the superblock and its extension pointer valid (only
            // the manager blocks, which are rewritten every commit, are discarded).
            // A tail placed *inside* the file gave the allocation back nothing to
            // shrink, so it records the end-of-allocation itself. Either way it is
            // defined, which a persisting file's message must be or an
            // assertion-enabled libhdf5 aborts on open (issue #178), and is the
            // value `H5Fget_freespace` accounts for correctly (verified in the
            // crosscheck).
            let eoa_pre_fsm = if reused {
                final_eoa
            } else {
                block.fshd_addr.get()
            };
            let info = file_space_info::persistent_single_manager(
                offset_width,
                strategy,
                threshold,
                page_size,
                block.fshd_addr.get(),
                eoa_pre_fsm,
            );
            object_header::build_v2_object_header(
                &self.rewrite_extension_region(old_ext_addr, &info)?,
            )?
        } else {
            // With no managers to allocate, libhdf5 records the final end of allocation
            // (`H5MF_settle_meta_data_fsm` and `H5MF__close_aggrfs` in `H5MF.c`, HDF5 1.14.6).
            let info = file_space_info::persistent_empty(
                offset_width,
                strategy,
                threshold,
                page_size,
                final_eoa,
            );
            object_header::build_v2_object_header(
                &self.rewrite_extension_region(old_ext_addr, &info)?,
            )?
        };
        // Both forms of the message carry the same twelve manager slots, so either
        // one measures what the placeholder did. The tail was reserved for that
        // length, and the manager blocks are placed after it.
        debug_assert_eq!(
            ext_oh.len() as u64,
            ext_len,
            "extension length must be stable across the placeholder and real messages"
        );

        let new_old_blocks = vec![FreeExtent::new(ext_addr, tail_len).ok_or(Error::Format(
            FormatError::OffsetOverflow {
                offset: ext_addr,
                length: tail_len,
            },
        ))?];

        // Write the extension, then the FSM blocks. Both forms are safe against a
        // crash here: an appended tail is past everything live, a reused one sits
        // in space an earlier commit freed, and neither is referenced until the
        // repoint below.
        let region = (ext_addr, tail_len);
        self.write_tail_block(region, StoredAddress::new(ext_addr), &ext_oh)?;
        if let Some(block) = flat_block {
            let fshd_len = hdf5_pure_format::__private::free_space_manager_header_len(widths);
            let fsse_allocated = tail_len
                .checked_sub(ext_len)
                .and_then(|manager_len| manager_len.checked_sub(fshd_len))
                .ok_or(Error::Format(FormatError::Internal(
                    "flat free-space tail is shorter than its extension and manager header".into(),
                )))?;
            let (fshd, fsse) =
                hdf5_pure_format::__private::serialize_free_space_manager_with_allocation(
                    widths,
                    &block.sections,
                    block.fshd_addr,
                    block.fsse_addr,
                    block.class,
                    Some(fsse_allocated),
                )?;
            self.write_tail_block(region, block.fshd_addr, &fshd)?;
            self.write_tail_block(region, block.fsse_addr, &fsse)?;
        }
        // Barrier, then repoint the superblock (root, eof, and the new extension)
        // — the linearization point — and sync it.
        self.barrier()?;
        let mut new_sb = self.superblock.clone();
        new_sb.root_group_address = self.addresses.relative(new_root)?;
        new_sb.eof_address = final_eof.get();
        new_sb.superblock_extension_address = Some(StoredAddress::new(ext_addr));
        // As above, and zero for the same reason: this is the *unpaged* persisting
        // tail, which no page-buffered session reaches (issue #73, issue #308).
        new_sb.consistency_flags = self.held_status_flags;
        let sb_bytes = hdf5_pure_format::__private::serialize_superblock(&new_sb)?;
        self.publish_attempted = true;
        let confirmed = self
            .aggregation
            .as_ref()
            .is_none_or(|state| state.publication_confirmed);
        if let Some(state) = &mut self.aggregation {
            state.publication_confirmed = false;
        }
        self.write_at(self.sb_sig_off, &sb_bytes)?;
        self.barrier()?;
        self.superblock = new_sb;

        self.space.adopt_flat_post_free(post);
        self.persist = Some(PersistState {
            strategy,
            threshold,
            page_size,
            old_blocks: new_old_blocks,
        });
        self.aggregation = aggregation;
        if let Some(state) = &mut self.aggregation {
            state.publication_confirmed = false;
        }

        // Physically shrink the file only after the superblock — now carrying the
        // smaller end-of-file — is durable, exactly as the non-persisting tail
        // does: a crash between the two leaves trailing bytes the next open
        // ignores, where the reverse order would advertise an end-of-file past
        // the file's actual length. The managers just written describe `post`,
        // which no longer names anything above this cut.
        if final_eof < PhysicalAddress::new(self.image.len()) {
            self.image.truncate(final_eof.get())?;
            self.barrier()?;
        }

        // The repoint is durable: the prior free list plus this commit's vacated
        // regions are now genuinely free, and the freshly written blocks become
        // the ones a future commit will supersede.
        // The rewritten managers describe persistent free sections at this file length.
        self.fsm_len = self.image.len();
        if let Some(state) = &mut self.aggregation {
            state.publication_confirmed = confirmed;
        }
        Ok(())
    }

    /// Returns the free space a flat persisting commit is about to record, and
    /// `eof` less the dropped regions that end it.
    ///
    /// The list is the session's durable list plus each of the regions this commit
    /// vacated and of the superseded extension and manager blocks that is at least
    /// `threshold` bytes long or adjoins a region of the list ([`FreeList::release_all`]).
    /// Those regions and blocks are dead once the superblock is repointed.
    ///
    /// Returned as a temporary rather than folded into the session, for the reason
    /// [`SessionSpace::paged_post_free`] does the same: every region
    /// gathered here is still *live* until that repoint, and everything in between
    /// can fail, so a session that believed them free after a failed commit would
    /// hand them out over the objects still occupying them.
    ///
    /// Called once to *size* the tail and again for each round its placement takes
    /// to settle, since what remains free depends on the space the tail has drawn
    /// for itself ([`flat_tail_layout`](Self::flat_tail_layout)).
    fn flat_post_free(
        &self,
        eof: u64,
        to_free: &[(StoredAddress, u64, FreeClass)],
        old_blocks: &[FreeExtent],
    ) -> (FreeList, u64) {
        let extents = to_free
            .iter()
            .filter_map(|&(addr, len, _)| FreeExtent::new(addr.get(), len))
            .chain(old_blocks.iter().copied());
        self.space
            .flat_post_free(eof, extents)
            .expect("the flat commit tail requires a flat session strategy")
    }

    /// Reserves free space for a flat persisting commit's tail, and returns the free list the tail
    /// leaves behind, its address, its length, and the end-of-allocation the commit publishes.
    ///
    /// `current_eoa` and every address in the result are stored addresses. The address is `None`
    /// when nothing reusable fits, and the caller then appends the tail at `current_eoa`. The
    /// length is the tail's in either case. The end-of-allocation of a reused tail is `current_eoa`
    /// less the dropped regions that reach it and what [`FreeList::release_trailing_run`] releases
    /// of the free space below them, and that of an appended tail is `current_eoa`.
    ///
    /// Reserves a proposed length before measuring the managers for the remaining free sections.
    /// Accepts a plan that fits within the reservation and retries with a larger length otherwise.
    /// Spare capacity belongs to the manager allocation and is reclaimed when it is superseded.
    ///
    /// The writer records spare reservation bytes in the section list's allocated length and
    /// fills them with zeros. On reopen, the reader recovers the complete allocation for
    /// reclamation.
    ///
    /// Iteration is capped rather than trusted to converge: the length is a step
    /// function of the section set, so a proposal can in principle keep growing.
    /// Appending is always available and always correct, so giving up costs space
    /// rather than a guarantee — and the cap bounds the work, since each round
    /// clones the free list and sizes the section list for real.
    ///
    /// Draws from [`space`](Self::space) before writing. [`commit`](Self::commit) and
    /// [`finalize_persistent_aggregates`](Self::finalize_persistent_aggregates) restore their
    /// snapshots after a failure before publication.
    ///
    /// When [`space`](Self::space) has no hole for it, the tail may take one out of
    /// [`reserved`](Self::reserved) instead. Only the rewrite that publishes an
    /// append's draw ([`reserve_for_immediate_append`](Self::reserve_for_immediate_append))
    /// reaches this with anything there — every other caller releases the reserve
    /// first — and for that rewrite the fallback is what keeps the draw from
    /// growing the file: a draw takes every hole the append fits in, and on a
    /// flat file that is the one list the tail is placed from, so without it the
    /// rewrite appended its tail at end-of-file every time and the file grew by a
    /// tail per draw (issue #413). Space in the reserve is exactly what this
    /// rewrite publishes as *not* free, so a tail placed there is live in bytes
    /// the managers do not advertise, which is the same standing a tail placed
    /// from [`space`](Self::space) has once the repoint lands. Session space is tried first, so an
    /// ordinary hole is spent before the append's own space. Best fit within
    /// the reserve then takes its smallest run, leaving the largest for the
    /// append that drew it. The paged tail
    /// has no such fallback: it opens a page when nothing fits, and
    /// `paged_group_churn_with_populated_datasets_reaches_a_steady_size` holds it
    /// to not doing so once per draw.
    ///
    /// With a `limit`, restricts reservations to extents ending at or below it and uses only
    /// [`space`](Self::space). This permits relocating managers below the old tail.
    fn flat_tail_layout(
        &mut self,
        current_eoa: u64,
        to_free: &[(StoredAddress, u64, FreeClass)],
        old_blocks: &[FreeExtent],
        ext_len: u64,
        widths: FormatWidths,
        limit: Option<u64>,
    ) -> (FreeList, Option<u64>, u64, u64) {
        /// Enough rounds for the section set to settle after a reservation shrinks
        /// it, without letting a proposal that keeps growing spin.
        const ROUNDS: usize = 4;

        // The first proposal: what the tail would measure with nothing reserved.
        // A manager block's length depends only on the sections it records, never
        // on where it sits. It is also the answer for a tail that ends up appended,
        // since every round hands its reservation back before this returns.
        let (probe, _) = self.flat_post_free(current_eoa, to_free, old_blocks);
        let managers = ManagerSections::flat(probe.extents().iter().copied());
        let appended_len = ext_len + free_space_manager::manager_blocks_len(&managers, widths);
        let mut proposed = appended_len;

        for _ in 0..ROUNDS {
            let allocation = if let Some(limit) = limit {
                self.space.allocate_flat_below(proposed, limit)
            } else {
                self.space.allocate(proposed, AllocationClass::Metadata)
            };
            let (reserved, from_reserve) = match allocation {
                Some(extent) => (extent, false),
                None if limit.is_none() => match self.reserved.alloc(proposed) {
                    Some(extent) => (extent, true),
                    None => break,
                },
                None => break,
            };
            let at = reserved.start();
            let (mut post, eoa) = self.flat_post_free(current_eoa, to_free, old_blocks);
            // Exclude the released suffix before sizing the managers. The tail reservation
            // has already been removed from this list, so the trailing run starts at or
            // above the reservation's end.
            let eoa = post.release_trailing_run(eoa, proposed);
            let managers = ManagerSections::flat(post.extents().iter().copied());
            let len = ext_len + free_space_manager::manager_blocks_len(&managers, widths);
            if len <= proposed {
                debug_assert!(
                    at + proposed <= eoa,
                    "a reused tail at {at} of {proposed} bytes runs past the end-of-allocation \
                     {eoa} it was placed under"
                );
                return (post, Some(at), proposed, eoa);
            }
            if from_reserve {
                self.reserved.free(reserved);
            } else {
                self.space
                    .return_allocation(reserved, AllocationClass::Metadata);
            }
            proposed = len;
        }
        // Nothing reusable fits, so the tail is appended past end-of-file and no
        // trailing run is released: whatever tracked free space reaches end-of-file
        // was too small for the tail and stays recorded below it, and a dropped
        // extent that ended the allocation stays below it untracked.
        (probe, None, appended_len, current_eoa)
    }

    /// Publishes the root, extension, and final per-class free sections of a persistent PAGE edit.
    ///
    /// Plans the complete metadata reservation before writing. The serialized managers and the
    /// adopted session state include the same page tails and exclude the complete reservation,
    /// including owned section-list padding. Placement uses appended pages or space already free
    /// before this publication. Pending object frees and superseded managers cannot fund it.
    ///
    /// Writes the planned blocks before publishing their addresses in the superblock. Publishes
    /// a smaller EOA before truncating the file, and adopts the planned state after publication
    /// and any truncation succeed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the persistent extension cannot be updated,
    /// a format error if planning or serialization fails, or an I/O error if writing,
    /// synchronization, or truncation fails.
    fn commit_persisting_paged(
        &mut self,
        new_root: u64,
        to_free: Vec<(StoredAddress, u64, FreeClass)>,
        placement: TailPlacement,
    ) -> Result<(), Error> {
        let widths = self.widths;
        let offset_width = widths.offsets;
        let (strategy, threshold, page_size, old_blocks) = {
            let ps = self
                .persist
                .as_ref()
                .expect("commit_persisting is only called when persistence is armed");
            (
                ps.strategy,
                ps.threshold,
                ps.page_size,
                ps.old_blocks.clone(),
            )
        };

        let base = self.addresses;

        // Page-align the file before anything else, so the rewritten extension and
        // the manager blocks begin on a page boundary and stay in metadata pages.
        // The padded tail becomes free space of whatever type that page held.
        self.pad_to_page()?;

        // As on the flat path: the extension address is stored relative to the base
        // address, and every reader of the extension below indexes the image.
        let old_ext_addr = base.absolute(
            self.superblock
                .superblock_extension_address
                .filter(|a| !a.is_undefined(offset_width))
                .ok_or(Error::EditUnsupported(
                    "a persisting file has no superblock extension to update",
                ))?,
        )?;

        // The 12-slot persist message is fixed-size, so a placeholder sizes the
        // rewritten extension before its manager addresses are known — and before
        // its address is, which is what lets the tail be sized before it is placed.
        let placeholder = file_space_info::persistent_managers(
            strategy,
            threshold,
            page_size,
            [UNDEF; NUM_FILE_FSM_MANAGERS],
            0,
        );
        let ext_len = object_header::build_v2_object_header(
            &self.rewrite_extension_region(old_ext_addr, &placeholder)?,
        )?
        .len() as u64;

        let current_eoa = self.end_of_allocation()?.get();
        let freed: Vec<_> = to_free
            .iter()
            .filter_map(|&(addr, len, class)| {
                FreeExtent::new(addr.get(), len).map(|extent| (extent, class))
            })
            .collect();
        let Some(plan) = paged_persistence::PagedPersistencePlan::new(
            &self.space,
            current_eoa,
            &freed,
            &old_blocks,
            ext_len,
            widths,
            placement == TailPlacement::ReuseOnly,
        )?
        else {
            return Ok(());
        };
        let paged_persistence::PagedPersistencePlan {
            tail,
            slots,
            blocks,
        } = plan;
        let ext_addr = tail.allocation.start();
        let blocks_len = tail.allocation.len();
        let final_eoa = tail.final_eoa;
        let reused = !tail.appended;
        if reused {
            let reserved = self.space.allocate(blocks_len, AllocationClass::Metadata);
            if reserved != Some(tail.allocation) {
                return Err(FormatError::Internal(
                    "a PAGE tail reservation differs from its pure plan".into(),
                )
                .into());
            }
        } else {
            self.begin_page(PageType::Meta)?;
        }
        let final_eof = self.addresses.resolve(StoredAddress::new(final_eoa))?;

        let info = if blocks.is_empty() {
            file_space_info::persistent_empty(
                offset_width,
                strategy,
                threshold,
                page_size,
                final_eoa,
            )
        } else {
            file_space_info::persistent_managers(
                strategy,
                threshold,
                page_size,
                slots.map(StoredAddress::get),
                final_eoa,
            )
        };
        let ext_oh = object_header::build_v2_object_header(
            &self.rewrite_extension_region(old_ext_addr, &info)?,
        )?;
        debug_assert_eq!(
            ext_oh.len() as u64,
            ext_len,
            "extension length must be stable across the placeholder and real messages"
        );

        // Exactly the bytes written, contiguous from the extension. A session that reopens this
        // file records the same extents from the message and manager headers, so nothing depends
        // on remembering this across a close.
        let new_old_blocks = vec![FreeExtent::new(ext_addr, blocks_len).ok_or(Error::Format(
            FormatError::OffsetOverflow {
                offset: ext_addr,
                length: blocks_len,
            },
        ))?];

        // Write the extension, then every manager block. Both forms are safe against
        // a crash here: an appended tail is past everything live, and a reused one
        // sits in space an earlier commit freed. Neither is referenced until the
        // repoint below.
        let region = (ext_addr, blocks_len);
        self.write_tail_block(region, StoredAddress::new(ext_addr), &ext_oh)?;
        for block in &blocks {
            self.write_tail_block(region, block.header_address, &block.header)?;
            self.write_tail_block(region, block.section_list_address, &block.section_list)?;
        }
        // An appended tail ends mid-page; pad it out, so the end-of-allocation stays
        // a whole number of pages. A reused tail is already inside the file, and
        // matched its reservation exactly, so there is nothing to pad — and where
        // it released a trailing run its end-of-allocation is *below* the current
        // end-of-file, which the truncation after the repoint gives back.
        if !reused {
            self.pad_zeros_to(final_eof.get())?;
        }
        // Barrier, then repoint the superblock (root, eof, and the new extension)
        // — the linearization point — and sync it.
        self.barrier()?;
        let mut new_sb = self.superblock.clone();
        new_sb.root_group_address = self.addresses.relative(new_root)?;
        new_sb.eof_address = final_eof.get();
        new_sb.superblock_extension_address = Some(StoredAddress::new(ext_addr));
        // The one publish site a page-buffered session actually reaches, and so
        // the one where this must not be a literal zero: a crash mark has to
        // outlive every commit the session makes (issue #308).
        new_sb.consistency_flags = self.held_status_flags;
        let sb_bytes = hdf5_pure_format::__private::serialize_superblock(&new_sb)?;
        self.publish_attempted = true;
        self.write_at(self.sb_sig_off, &sb_bytes)?;
        self.barrier()?;
        self.superblock = new_sb;

        // Physically shrink the file only after the superblock — now carrying the
        // smaller end-of-file — is durable, for the reason the flat tail gives:
        // a crash between the two leaves trailing bytes the next open ignores,
        // where the reverse order would advertise an end-of-file past the file's
        // actual length.
        if final_eof < PhysicalAddress::new(self.image.len()) {
            self.image.truncate(final_eof.get())?;
            self.barrier()?;
        }

        // The repoint is durable. Only now are this commit's vacated regions
        // genuinely free, so adopt the lists built above and drop the padding tails
        // they already account for. The blocks just written become the ones the next
        // commit supersedes.
        self.space.adopt_paged_post_free(tail.post, !reused, None);
        self.persist = Some(PersistState {
            strategy,
            threshold,
            page_size,
            old_blocks: new_old_blocks,
        });
        // The rewritten managers describe persistent free sections at this file length.
        self.fsm_len = self.image.len();
        Ok(())
    }

    /// Pad a paged file to a page boundary if its tail page is partially filled,
    /// recording the padding as free space of the tail page's type. A no-op on a
    /// non-paged file or an already-aligned one.
    fn pad_to_page(&mut self) -> Result<(), Error> {
        let eoa = self.end_of_allocation()?;
        if let Some(transition @ PageTransition::Pad { extent, .. }) =
            self.space.page_padding(eoa.get())?
        {
            self.append(&vec![0u8; extent.len().to_usize()?])?;
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
    /// The region is checked rather than asserted, for the reason
    /// [`place`](Self::place) checks its own reservation: a reused span is followed
    /// by live bytes, so a block running past it would silently destroy a
    /// neighboring object, and this is the one write in a commit that lands in the
    /// middle of a live file. The region's length is derived from the same layout
    /// that placed these blocks, so the two cannot disagree today; the comparison
    /// is what keeps that true if a layout ever learns a length this sizing does
    /// not model.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::SerializationError`] if the block extends outside `region`,
    /// [`FormatError::OffsetOverflow`] if `addr` plus the base address exceeds [`u64::MAX`], and
    /// the write's error if the write fails.
    fn write_tail_block(
        &mut self,
        region: (u64, u64),
        addr: StoredAddress,
        bytes: &[u8],
    ) -> Result<(), Error> {
        let (start, len) = region;
        let stored = addr.get();
        if stored < start || stored + bytes.len() as u64 > start + len {
            return Err(Error::Format(FormatError::SerializationError(format!(
                "a commit's tail reserved [{start}, {}) but placed {} bytes at {stored}",
                start + len,
                bytes.len()
            ))));
        }
        let at = self.addresses.resolve(addr)?;
        if at == PhysicalAddress::new(self.image.len()) {
            let written = self.append(bytes)?;
            debug_assert_eq!(
                written,
                at.get(),
                "an appended block must land at end-of-file"
            );
            return Ok(());
        }
        self.write_at(at.get(), bytes)
    }

    /// Extend the file with zeros up to `target` (>= the current length), used by
    /// the paged tail to pad the final metadata page to its boundary.
    fn pad_zeros_to(&mut self, target: u64) -> Result<(), Error> {
        let len = self.image.len();
        if target > len {
            let pad = (target - len).to_usize()?;
            self.append(&vec![0u8; pad])?;
        }
        debug_assert_eq!(self.image.len(), target);
        Ok(())
    }

    /// Repoint a version 0/1 superblock at the rebuilt (now v2) root group and
    /// update its end-of-file field, patching the raw bytes in place — these
    /// superblocks carry no checksum. The root symbol-table entry is switched to
    /// cache type 0 (its scratch-pad B-tree / local-heap addresses, which
    /// describe the old symbol-table group, no longer apply). The
    /// object-header-address write is done last so it is the linearization point.
    fn repoint_v0v1_root(&mut self, new_root: StoredAddress, new_eof: u64) -> Result<(), Error> {
        let os = self.superblock.offset_size.get();
        let addr_width = u64::from(os);
        // Field layout after the fixed prefix: base / free-space / EOF / driver
        // addresses, then the root symbol-table entry (link-name offset, object
        // header address, cache type(4), reserved(4), scratch(16)). The prefix is
        // 24 bytes for v0 and 28 for v1 (the latter adds indexed-storage-K).
        let var_start = if self.superblock.version == SuperblockVersion::Zero {
            24
        } else {
            28
        };
        let base = self.sb_sig_off + var_start;
        let eof_off = base + 2 * addr_width;
        let ste = base + 4 * addr_width;
        let oh_addr_off = ste + addr_width;
        let cache_off = ste + 2 * addr_width;
        self.write_at(eof_off, &new_eof.to_le_bytes()[..usize::from(os)])?;
        self.write_at(cache_off, &[0u8; 4])?; // cache type = none
        self.write_at(cache_off + 8, &[0u8; 16])?; // clear scratch-pad
        self.write_at(
            oh_addr_off,
            &new_root.get().to_le_bytes()[..usize::from(os)],
        )?;
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
        let Some(transition) = space.plan_transition(eoa.get(), ty)? else {
            return Ok(());
        };
        if let PageTransition::Pad { extent, .. } = transition {
            image.append(&vec![0u8; extent.len().to_usize()?])?;
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
    /// Splitting the choice from the write is what lets a *relocatable* blob — one
    /// whose interior addresses are all `base + fixed offset`, like a chunked
    /// dataset's data region or a dense attribute heap — be built for a freed
    /// region rather than only for end-of-file: its size is known from its plan
    /// before its bytes exist, so the address can be settled first
    /// ([`place_relocatable`](Self::place_relocatable) is that sequence).
    ///
    /// Reuse only ever draws from free space vacated by *earlier* commits in this
    /// session (or seeded from the on-disk managers at open) — never space the
    /// current commit is about to free, which stays in its `to_free` list until
    /// after the superblock repoint. The bytes it overwrites are therefore already
    /// unreachable from the on-disk root, so a mid-commit crash cannot corrupt the
    /// live tree: the superblock still points at the prior, intact root.
    ///
    /// A persistent paged file reuses within the matching page type, or out of a page holding
    /// nothing at all, which belongs to no type ([`SessionSpace::allocate`]). Every
    /// page stays homogeneous either way. It never opens a page for a reused region
    /// — the tail page is untouched by a write into the middle of the file.
    /// A non-persistent PAGE session appends at or beyond
    /// [`fresh_page_boundary`](Self::fresh_page_boundary) without drawing from free lists.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if address resolution, aggregate planning, or PAGE transition
    /// planning fails. Returns the write error if writing PAGE padding fails.
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
    /// planning fails. Returns the write error if writing PAGE padding fails.
    fn reserve_with_page_type(
        &mut self,
        len: u64,
        class: AllocationClass,
        page_type: PageType,
    ) -> Result<Placement, Error> {
        if self.fresh_page_boundary.is_none()
            && let Some(extent) = self.space.allocate_with_page_type(len, class, page_type)
        {
            return Ok(Placement::Reused {
                addr: self
                    .addresses
                    .resolve(StoredAddress::new(extent.start()))?
                    .get(),
                len,
            });
        }
        if let Some(placement) = self.reserve_aggregated(len, class)? {
            return Ok(placement);
        }
        self.begin_page(page_type)?;
        if let Some(boundary) = self.fresh_page_boundary
            && self.end_of_allocation()? < boundary
        {
            return Err(FormatError::Internal(
                "PAGE allocation precedes its fresh-page boundary".into(),
            )
            .into());
        }
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
        self.space
            .allocate(len, class)
            .map(|extent| StoredAddress::new(extent.start()))
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
        let Some(extent) = FreeExtent::new(addr.get(), len) else {
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
        let (bytes, extra) = build(self.addresses.relative(at.address())?)?;
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
                Ok(self.addresses.relative(addr)?.get())
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
            || staged.writes()
                .iter()
                .chain(staged.datasets())
                .any(|(_, fd)| {
                    fd.vl_string_staging.is_none()
                        && datatype::datatype_holds_file_address(&fd.dt)
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

    /// Resolves one object-reference element's target to the address stored on
    /// disk. [`ObjectRefTarget::Raw`] is written back verbatim: a null or an
    /// undefined reference points at nothing, and the whole-file writer treats
    /// it the same way. [`ObjectRefTarget::Path`] resolves, in order:
    ///
    /// 1. Against `path_addr` — every group and dataset this commit has
    ///    already placed (a sibling dataset placed earlier in the same
    ///    group's batch — see the apply loop's non-reference-first ordering —
    ///    or a descendant subtree fully processed earlier in the deepest-first
    ///    walk).
    /// 2. Against the pre-commit on-disk file
    ///    ([`resolve_path_any`](crate::group_v2::resolve_path_any)), but only
    ///    when the path is untouched by this commit, so its pre-commit
    ///    address is guaranteed to still be valid post-commit. "Touched"
    ///    means: a dirty group (`nodes`, new or merely rewritten because an
    ///    addition lives under it — its own address changes either way); a
    ///    path this commit adds, or that lies under a subtree this commit
    ///    copies in (`add_targets`, checked by prefix so a copy's interior is
    ///    covered even though only its root is enumerated there); or a
    ///    `write_dataset` target (`write_targets`) — conservatively refused
    ///    even for a same-length overwrite that does not actually relocate,
    ///    since resolving that distinction here is not worth the complexity.
    /// 3. If the path resolves nowhere at all (neither this commit nor the
    ///    pre-commit file has ever heard of it), as an undefined reference
    ///    (`HADDR_UNDEF`) — mirroring [`ObjectRefTarget::Path`]'s existing
    ///    whole-file-writer resolution convention for the same builder type.
    ///
    /// A path that step 1 misses but step 2 identifies as commit-touched is
    /// refused with a clear [`Error::EditUnsupported`] rather than resolved to
    /// a stale or wrong address — the one case this engine cannot resolve
    /// without the whole-file writer's two-pass dummy/real-address scheme.
    ///
    /// A path this same commit *deletes* (`delete_targets`) is refused for a
    /// different reason, and so carries its own message: the address step 2
    /// would resolve is not stale but doomed, and the next commit to reuse the
    /// span leaves a reference that dereferenced cleanly reading whatever
    /// landed there. The test is by **prefix**, because a deletion takes the
    /// whole subtree with it (`collect_free_spans` walks it): a reference to a
    /// child of a deleted group dangles exactly as a reference to the group
    /// itself does (issue #314).
    ///
    /// It is a conservative test in the same sense `write_targets` is. A child
    /// whose object survives the delete through another hard link keeps its
    /// address — `collect_free_spans` reclaims nothing when the incoming count
    /// is not 1 — and is refused here anyway, so the message states what the
    /// commit does rather than what the allocator will conclude.
    ///
    /// The delete test runs *after* the other three so that a replacement this
    /// commit has merely not placed yet — a sibling group later in the same
    /// depth band — is reported as an ordering problem rather than as a
    /// deletion. That ordering is a better default, not a partition: because
    /// `add_targets` claims a replaced path's whole subtree by prefix, a child
    /// the replacement does *not* recreate is genuinely doomed and still
    /// reports "still writing". Distinguishing it would mean enumerating what
    /// a replacement actually rebuilds, which neither list holds. A path the
    /// commit puts back is resolved by step 1 before either test is reached.
    fn resolve_reference_target(
        target: &ObjectRefTarget,
        path_addr: &BTreeMap<ObjectPathBuf, u64>,
        nodes: &BTreeMap<ObjectPathBuf, Node>,
        add_targets: &[ObjectPathBuf],
        write_targets: &[ObjectPathBuf],
        delete_targets: &[ObjectPathBuf],
        invalidated: &InvalidatedAddresses,
        src: &(impl Source + ?Sized),
        superblock: &Superblock,
        widths: FormatWidths,
        addresses: AddressContext,
    ) -> Result<StoredAddress, Error> {
        let path = match target {
            // An address carries no name to test, so the delete check the path
            // arm makes by prefix is made here on the address itself.
            //
            // It fires on nothing today: every `Raw` this crate stages is a
            // sentinel, because `repack`'s faithful re-emit resolves a real
            // target to a `Path` and leaves only the null and undefined
            // references raw. It is here because the variant carries an
            // arbitrary address, and a producer that staged a real one would
            // otherwise write it past the screen its `Path` twin gets — which
            // is the shape of this bug in the first place (issue #317).
            ObjectRefTarget::Raw(addr) => {
                if let Some(refusal) = invalidated.refusal(*addr) {
                    return Err(Error::EditUnsupported(refusal));
                }
                return Ok(*addr);
            }
            ObjectRefTarget::Path(path) => path,
        };
        let base = addresses;
        if let Some(&addr) = path_addr.get(path) {
            return Ok(base.relative(addr)?);
        }
        if nodes.contains_key(path)
            || add_targets.iter().any(|t| path.starts_with(t))
            || write_targets.contains(path)
        {
            return Err(Error::EditUnsupported(
                "an object-reference dataset targets a path this commit is still writing; \
                 use separate commits",
            ));
        }
        // After the three above; see this function's doc for why, and for what
        // that ordering does and does not buy.
        if delete_targets.iter().any(|d| path.starts_with(d)) {
            return Err(Error::EditUnsupported(
                "an object-reference dataset targets an object this commit deletes, or one \
                 under it; the reference would be left pointing at storage the delete can \
                 reclaim",
            ));
        }
        Ok(
            match crate::group_v2::resolve_path_any_from_source(
                src,
                AccessMode::ReadWrite,
                superblock,
                widths,
                addresses,
                &path.as_path(),
            ) {
                Ok(addr) => base.relative(addr)?,
                Err(_) => StoredAddress::new(UNDEF),
            },
        )
    }

    /// Prove, before any byte of this commit is written, that every
    /// object-reference target across every staged dataset will resolve
    /// successfully — either against a pre-existing untouched object or
    /// against something this same commit places. [`resolve_reference_target`]
    /// classifies a target purely from *whether* an [`ObjectPathBuf`] has been placed
    /// yet (`path_addr.get`), never from the address *value*, so replaying the
    /// apply loop's placement order here with a placeholder address standing in
    /// for "already placed" reproduces the exact same verdict the apply loop's
    /// own calls will reach later, without writing anything.
    ///
    /// The placeholder is the file's **base address**, not zero. Any value
    /// reproduces the verdict, but it is also handed to the same `addr - base`
    /// that converts a real address into its stored form: a zero placeholder
    /// underflows that on every file with a userblock, panicking in a debug
    /// build on an otherwise legal edit. At the base address it converts to a
    /// stored `0`, which is exactly the "placed, real address not known yet"
    /// this stands for. If
    /// this preflight pass returns `Ok`, none of the apply loop's own
    /// `resolve_reference_target` calls can fail, so a reference-resolution
    /// error can no longer leave earlier-processed groups' real writes
    /// orphaned in the file (the failure surfaces here instead, before the
    /// apply loop's first `place`/`write_at`).
    ///
    /// [`resolve_reference_target`]: Self::resolve_reference_target
    fn preflight_reference_targets(
        keys: &[ObjectPathBuf],
        flat: &BTreeMap<&ObjectPathBuf, Vec<&FlatDataset>>,
        nodes: &BTreeMap<ObjectPathBuf, Node>,
        add_targets: &[ObjectPathBuf],
        write_targets: &[ObjectPathBuf],
        delete_targets: &[ObjectPathBuf],
        invalidated: &InvalidatedAddresses,
        src: &(impl Source + ?Sized),
        superblock: &Superblock,
        widths: FormatWidths,
        addresses: AddressContext,
    ) -> Result<(), Error> {
        let mut by_depth = keys.to_vec();
        // Stable on purpose, for the same reason as the apply loop's copy: this
        // simulation is only faithful if it walks the groups in that same order.
        by_depth.sort_by_key(|k| std::cmp::Reverse(k.len()));
        let mut sim_addr: BTreeMap<ObjectPathBuf, u64> = BTreeMap::new();
        for key in &by_depth {
            if let Some(datasets) = flat.get(key) {
                // Mirrors the apply loop's `group_datasets.sort_by_key(|fd|
                // fd.reference_targets.is_some())`: non-reference datasets are
                // placed (and so become resolvable) before any reference
                // dataset in the same group.
                let mut ordered: Vec<&FlatDataset> = datasets.to_vec();
                // Stable on purpose, like the loop it mirrors.
                ordered.sort_by_key(|fd| fd.reference_targets.is_some());
                for fd in ordered {
                    if let Some(patches) = &fd.reference_targets {
                        for patch in patches {
                            Self::resolve_reference_target(
                                &patch.target,
                                &sim_addr,
                                nodes,
                                add_targets,
                                write_targets,
                                delete_targets,
                                invalidated,
                                src,
                                superblock,
                                widths,
                                addresses,
                            )?;
                        }
                    }
                    sim_addr.insert(key.join(&fd.name), addresses.effective_base().get());
                }
            }
            sim_addr.insert(key.clone(), addresses.effective_base().get());
        }
        Ok(())
    }
}

/// A dirty group in the edit plan: its base object-header message region and the
/// additions targeting it.
#[derive(Default)]
struct Node {
    is_new: bool,
    /// Names of links to remove from this group (from `delete`).
    deletes: Vec<LinkNameBuf>,
    /// Copies to add to this group: (new link name, the source subtree read out
    /// for writing). Built at staging time from either this file (an in-file
    /// [`copy`](crate::File::copy)) or another open file (a cross-file
    /// [`copy_from`](crate::File::copy_from)).
    copies: Vec<(LinkNameBuf, CopyTree)>,
    /// New link names this commit adds to this group by cross-file copy. The
    /// subtrees themselves stay in the staged set — which the preflight may
    /// still refuse, and must not empty (issue #316) — and join `copies` at the
    /// point of no return.
    cross_copies: Vec<LinkNameBuf>,
    /// Value overwrites whose dataset header relocates (a resize or compact
    /// rewrite by `write_dataset`, a staged append, an attribute edit), as (child
    /// link name, the pre-commit object-header address, the relocation plan). On
    /// apply, the new data and header are written and this group's existing link
    /// to the moved header is patched to its new address — exactly like an
    /// existing child group's link. The old address is carried rather than
    /// re-resolved: the screen above and the reclaim below both need it, and one
    /// derivation cannot disagree with itself.
    writes: Vec<(LinkNameBuf, u64, MovingWrite)>,
    base_region: OhRegion,
    existing_links: Vec<String>,
    /// Attribute storage prepared by [`attributes::plan_attr_ops`] for
    /// [`WriteEngine::place_edited_attrs`]. Empty for a group without attribute edits.
    attrs: EditedAttrs,
    /// Whether this group's attribute edit replaces a pre-commit dense attribute set.
    replaces_dense_storage: bool,
}

/// Records physical addresses invalidated by a commit for object-reference screening.
///
/// Removed spans are also used for storage reclamation. Moved entries identify old object-header
/// addresses whose replacements are published by the commit.
struct InvalidatedAddresses {
    /// Absolute `(offset, length)` file spans this commit's deletions reclaim, as
    /// [`collect_free_spans`](WriteEngine::collect_free_spans) reports them.
    removed: Vec<(u64, u64)>,
    /// Superseded object-header addresses from relocating dataset writes and rebuilt groups.
    moved: Vec<u64>,
    /// The address context used to compare stored object references with physical positions.
    base: AddressContext,
}

impl InvalidatedAddresses {
    /// Returns whether the commit preserves every recorded address.
    fn is_empty(&self) -> bool {
        self.removed.is_empty() && self.moved.is_empty()
    }

    /// Returns a rejection for a reference into removed storage or a superseded header.
    ///
    /// Null, undefined, and unresolvable stored addresses are excluded from this screen.
    fn refusal(&self, stored: StoredAddress) -> Option<&'static str> {
        if stored.get() == 0 || stored.get() == UNDEF {
            return None;
        }
        let abs = self.base.absolute(stored).ok()?;
        if self
            .removed
            .iter()
            .any(|&(off, len)| abs >= off && abs - off < len)
        {
            return Some(REFERENCE_INTO_RECLAIMED_SPACE);
        }
        if self.moved.contains(&abs) {
            return Some(REFERENCE_TO_A_MOVED_OBJECT);
        }
        None
    }
}

/// Group `(parent group, item)` pairs by their parent, preserving the input
/// order within each group.
///
/// The one rule by which a commit's staged datasets become per-group batches:
/// the preflight groups borrowed ones to prove its guards, and the apply loop
/// groups the owned ones it places, so the order
/// [`preflight_reference_targets`](WriteEngine::preflight_reference_targets)
/// replays is the order the apply loop uses by construction.
fn group_by_parent<K: Ord, T>(items: impl IntoIterator<Item = (K, T)>) -> BTreeMap<K, Vec<T>> {
    let mut out: BTreeMap<K, Vec<T>> = BTreeMap::new();
    for (parent, item) in items {
        out.entry(parent).or_default().push(item);
    }
    out
}

/// Ensure a node exists for every ancestor prefix of `path` (so each is rebuilt
/// and can re-wire its child link). Does not set `is_new`.
fn ensure_ancestors(nodes: &mut BTreeMap<ObjectPathBuf, Node>, path: &ObjectPathBuf) {
    for len in 0..=path.len() {
        nodes.entry(path.prefix(len)).or_default();
    }
}

/// Validate that every reclaim span `(addr, len)` is non-empty, ends at or
/// before `eof`, and that no two overlap; sorts `spans` by address as a side
/// effect. Returns `false` on any violation so the caller can decline to
/// reclaim the object rather than feed the free list an out-of-bounds or
/// overlapping (double-free) region. Touching spans are allowed — the free list
/// coalesces them.
fn spans_disjoint_in_bounds(spans: &mut [(u64, u64)], eof: u64) -> bool {
    for &(addr, len) in spans.iter() {
        match addr.checked_add(len) {
            Some(end) if len > 0 && end <= eof => {}
            _ => return false,
        }
    }
    spans.sort_unstable_by_key(|&(addr, _)| addr);
    spans.windows(2).all(|w| w[0].0 + w[0].1 <= w[1].0)
}

/// Sanitize the accumulated free spans for a whole commit so the free list never
/// sees an out-of-bounds or overlapping (double-free) region: drop empty or
/// past-`eof` spans, sort by address, then drop any span overlapping one already
/// kept. Dropping only leaks (the bytes stay allocated); it never frees a live
/// region. With the last-hard-link guard in force nothing should be dropped for
/// a well-formed file — this is a backstop, not the primary defense.
fn retain_disjoint_in_bounds(spans: &mut Vec<(u64, u64, FreeClass)>, eof: u64) {
    spans.retain(|&(addr, len, _)| len > 0 && addr.checked_add(len).is_some_and(|e| e <= eof));
    spans.sort_unstable_by_key(|&(addr, _, _)| addr);
    let mut kept_end = 0u64;
    spans.retain(|&(addr, len, _)| {
        if addr >= kept_end {
            kept_end = addr + len;
            true
        } else {
            false // overlaps a span already kept; leak it rather than double-free
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
