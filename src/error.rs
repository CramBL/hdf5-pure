//! Errors from HDF5 file operations and their format and filter dependencies.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::string::String;

#[cfg(feature = "std")]
use std::string::String;

#[cfg(feature = "std")]
use core::fmt;

#[cfg(feature = "std")]
use hdf5_pure_core::EditStorageConflict;

#[cfg(feature = "std")]
use crate::address::StoredAddress;
#[cfg(feature = "std")]
use crate::free_sections::FreeSectionsError;
#[cfg(feature = "std")]
use crate::message_type::MessageType;
#[cfg(feature = "std")]
use crate::object_graph::TraversalError;
#[cfg(feature = "std")]
use crate::object_graph::TraversalLimit;

pub use hdf5_pure_core::FormatError;
pub use hdf5_pure_core::OBJECT_HEADER_MESSAGE_MAX;

/// Distinguishes missing components and intermediate non-group objects from format errors.
#[derive(Debug)]
pub(crate) enum ResolveError {
    /// A malformed structure, invalid target, or unsupported link.
    Format(FormatError),
    /// The root-relative path of an intermediate object that is not a group.
    NotAGroup(String),
    /// The root-relative path through the first missing component.
    NotFound(String),
}

impl From<FormatError> for ResolveError {
    fn from(e: FormatError) -> Self {
        ResolveError::Format(e)
    }
}

// ---------------------------------------------------------------------------
// High-level Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when using the high-level API.
#[cfg(feature = "std")]
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// An earlier editing operation has an unconfirmed allocation or publication outcome.
    ///
    /// The editor discards queued storage writes and rejects further appends, commits, and
    /// explicit synchronization in this session. Close and reopen the file before editing again.
    /// An interrupted file can fail to reopen.
    AllocationPublicationUncertain,
    /// The dataset or file is not a supported target for the fast, immediate
    /// in-place append ([`Dataset::append`](crate::Dataset::append)).
    ///
    /// Examples include a userblock or non-latest-format file, a dataset that is
    /// not rank 1, an unsupported chunk index kind, an append that exceeds a
    /// finite maximum dimension, a dataset reachable through more than one
    /// hard link, or a path an uncommitted staged edit in the same session will
    /// relocate or delete. Distinct from [`AppendUnsupported`](Self::AppendUnsupported)
    /// so a caller can catch this fast-path rejection and fall back to the staged
    /// [`Dataset::append_staged`](crate::Dataset::append_staged). The payload is a
    /// human-readable reason.
    AppendInPlaceUnsupported(&'static str),
    /// The dataset or appended values are unsupported by
    /// [`Dataset::append_staged`](crate::Dataset::append_staged).
    ///
    /// A committed dataset must be rank 1, chunked, and indexed by a single chunk, Fixed Array, or
    /// Extensible Array. Appended values must fit the maximum first dimension. Unsupported
    /// filters, incompatible element datatypes, sparse chunk grids, and shared hard links can
    /// also cause this error. The payload is a human-readable reason.
    AppendUnsupported(&'static str),
    /// A supported attribute value could not be decoded.
    ///
    /// The original error is available through [`source`](std::error::Error::source).
    AttributeDecode {
        /// The name of the attribute whose value could not be decoded.
        name: String,
        /// The decoding or source-read failure.
        source: FormatError,
    },
    /// Restoration of prior bytes could not be confirmed after an editing error.
    ///
    /// Before publication, the editor attempts to restore prior values overwritten by the failed
    /// commit. When this error is the source of [`Self::HeapCompactionFailed`], restoration covers
    /// only the failed collection. The published tree, its in-place values, and completed
    /// collection compactions remain committed.
    ///
    /// Both operation and restoration errors are retained. Overwritten bytes may be partially
    /// restored, or fully restored with an unconfirmed synchronization barrier.
    CommitPartiallyApplied {
        /// The original operation or completion error.
        refusal: Box<Error>,
        /// The restoration write or synchronization error, returned by
        /// [`source`](std::error::Error::source).
        restore: Box<Error>,
    },
    /// The copy screen could not read a message of an object this commit copies:
    /// a committed (shared) datatype that did not resolve, or an attribute that
    /// did not parse or resolve. Without it, the copy's elements cannot be
    /// checked against the space the same commit reclaims.
    CopyScreenUnreadable {
        /// The type of the message the screen stopped at.
        message: MessageType,
        /// The failure the resolve or the parse returned.
        source: FormatError,
    },
    /// A physical bounds, overlap, or truncation conflict reported through [`EditStorageConflict`].
    ///
    /// The conflict is also available through [`std::error::Error::source`].
    EditStorageConflict(EditStorageConflict),
    /// The file or requested operation is outside the editor's supported formats or storage
    /// structures. The payload describes the unsupported case.
    EditUnsupported(&'static str),
    /// A write was requested through a handle whose [`crate::File`] has already
    /// been sealed by [`crate::File::close`]. Immediate and staged edits are
    /// rejected. Reads through surviving handles still work. Re-open the file to
    /// modify it again.
    FileClosed,
    /// An exclusive OS advisory lock could not be acquired for editing or clearing superblock flags.
    ///
    /// The operating system releases the lock when the holding process exits. Locking can be
    /// disabled per editing open with [`crate::FileLocking::Disabled`] or globally with
    /// `HDF5_USE_FILE_LOCKING=FALSE`. The payload describes the lock conflict.
    FileLocked(String),
    /// The superblock status flags mark the file as open for writing.
    ///
    /// Ordinary and SWMR writers can leave these flags set after an interrupted session.
    /// The payload describes the flags and the applicable read or recovery options.
    /// [`crate::File::clear_status_flags`] clears them after the writer has stopped, without
    /// guaranteeing data integrity.
    FileMarkedInUse(String),
    /// A format, filter, or file-operation error reported through [`FormatError`].
    Format(FormatError),
    /// Persisted free-section inspection was requested on a writable handle.
    ///
    /// [`File::persisted_free_sections`](crate::File::persisted_free_sections) requires a
    /// read-only handle.
    FreeSectionInspectionUnsupported,
    /// An operation that needs exclusive access to the open file (e.g.
    /// [`crate::File::refresh`]) was requested while owned [`crate::Dataset`] /
    /// [`crate::Group`] handles, or a clone of the [`crate::File`], are still
    /// alive. Drop them and retry.
    HandlesOutstanding,
    /// Global heap compaction failed after protected tree publication completed.
    ///
    /// The tree edits and in-place value overwrites remain published. Completed collections remain
    /// compacted, and subsequent collection plans were not attempted. The failed collection's
    /// restoration and allocation state is recorded separately from its cause.
    HeapCompactionFailed {
        /// The number of collection plans completed before the failure.
        completed: usize,
        /// The failed collection's restoration state and the editor's allocation availability.
        state: HeapCompactionState,
        /// The operation error, including [`Self::CommitPartiallyApplied`] if restoration failed.
        ///
        /// Also returned by [`std::error::Error::source`].
        source: Box<Error>,
    },
    /// An object's declared reference count is below its known minimum.
    InconsistentReferenceCount {
        /// The absolute file offset of the object's header.
        address: u64,
        /// The original declared reference count, including an implicit count of one.
        observed: u32,
        /// The minimum count established from known original users.
        minimum: u64,
    },
    /// The allocation geometry of a reachable fractal heap is inconsistent.
    InvalidFractalHeapStorage {
        /// The stored address of the fractal-heap header.
        address: u64,
    },
    /// An identified storage extent overlaps another extent or exceeds the logical end of file.
    InvalidSpaceOwnershipExtent {
        /// The absolute file offset of the invalid extent.
        address: u64,
        /// The length of the invalid extent in bytes.
        length: u64,
        /// The end of the preceding extent in address order.
        previous_end: u64,
        /// The end-of-file position stored in the superblock.
        logical_size: u64,
    },

    /// I/O error from the filesystem.
    Io(std::io::Error),
    /// A required header message was not found.
    MissingMessage(MessageType),
    /// The object at the given path is not a dataset.
    NotADataset(String),
    /// The object at `path` is not a group.
    ///
    /// Every intermediate component must be a group: if `a/b` is a dataset, resolving `a/b/c`
    /// through [`File::group`](crate::File::group) or [`File::dataset`](crate::File::dataset)
    /// returns `Error::NotAGroup("a/b")`. A missing component returns
    /// [`FormatError::PathNotFound`] instead.
    NotAGroup(String),
    /// The child of the given name is not a committed (`H5Tcommit`) datatype. A
    /// name that resolves to nothing at all is [`FormatError::PathNotFound`]
    /// instead.
    NotANamedDatatype(String),
    /// The object named by the payload has been *staged* by this read-write
    /// session and not yet published by [`crate::File::commit`], so the
    /// operation asked for would have had to read bytes that are not in the
    /// file.
    ///
    /// A staged creation is addressable as soon as it is staged: the handle
    /// [`crate::Group::create_group`] / [`crate::Group::create_group_with`] /
    /// [`crate::Group::create_dataset`] returns - and the one a lookup for that
    /// path gives back - can stage further edits under it, and a staged dataset
    /// answers [`crate::Dataset::shape`], [`crate::Dataset::maxshape`],
    /// [`crate::Dataset::dtype`], [`crate::Dataset::datatype`],
    /// [`crate::Dataset::is_chunked`], [`crate::Dataset::filters`] and
    /// [`crate::Dataset::filter_pipeline`] from what was staged. Everything that reads the
    /// object's bytes reports this until
    /// the commit: element reads, attribute reads, and the edits that rewrite an
    /// existing object in place ([`crate::Dataset::append`],
    /// [`crate::Dataset::write`], `set_attr` on a dataset). Add the elements to
    /// the builder that stages the dataset instead - or, for a dataset,
    /// [`crate::Dataset::append_staged`], which folds them into the pending
    /// creation.
    NotCommitted(String),
    /// Two free sections overlap, including when their extents are identical.
    ///
    /// Addresses use the coordinate system of the inspected sections. Persisted sections use
    /// base-relative addresses, and session-reusable sections use absolute physical offsets.
    /// Adjacent sections with equal end and start addresses are permitted.
    OverlappingFreeSections {
        /// The start address of the overlapping section.
        address: u64,
        /// The length of the overlapping section in bytes.
        length: u64,
        /// The start address of the preceding section in address order.
        previous_address: u64,
        /// The length of the preceding section in bytes.
        previous_length: u64,
    },
    /// A persisted manager could not be read or parsed.
    ///
    /// Manager and section addresses are relative to the HDF5 base address. The cause is
    /// available through [`source`](std::error::Error::source).
    PersistedFreeSpace {
        /// The zero-based index in the File Space Info manager-address array.
        manager_slot: usize,
        /// The stored address of the manager header.
        header_address: StoredAddress,
        /// The stored section-list address, if it was decoded and is defined.
        section_list_address: Option<StoredAddress>,
        /// The stored start address and byte length of the offending section, if available.
        section: Option<(StoredAddress, u64)>,
        /// The acquisition or parsing failure.
        source: Box<Error>,
    },
    /// A write (e.g. [`crate::Dataset::append`]) was requested on a file opened
    /// read-only. Open it with [`crate::File::open_rw`] to modify it in place.
    ReadOnly,
    /// An object in the source file cannot be reproduced faithfully by
    /// [`repack`](crate::repack()). The payload identifies the object and reason.
    RepackUnsupported(String),
    /// An array shape error from the `ndarray` integration: either the flat
    /// data could not be reshaped to the dataset's dimensions, or a requested
    /// static rank (e.g. `read_array::<_, Ix2>`) did not match the dataset's
    /// runtime rank. Only constructed when the `ndarray` feature is enabled.
    Shape(String),
    /// A [`crate::Dataset`] / [`crate::Group`] handle onto a staged creation was
    /// used after that creation was **withdrawn**, so it names nothing: the path
    /// in the payload was staged when the handle was made, and this session has
    /// since dropped that staging without committing it.
    ///
    /// [`crate::Group::delete`] of an object staged in the same session is what
    /// withdraws one - including the second `delete` of a delete-then-create
    /// *replacement*, which leaves the deletion of the file's own object
    /// standing and the staged replacement gone. The handle is not retargeted at
    /// whatever the file holds at that path, because that object is precisely
    /// the one the session is removing; it reports this instead, and a fresh
    /// lookup by name is how to reach whatever the path means now.
    ///
    /// Only a handle *born* onto a staged creation can report it. One opened
    /// onto an object in the file names that object however the staged set
    /// changes around it.
    StagingWithdrawn(String),
    /// A [`crate::Dataset`] / [`crate::Group`] handle reached by object
    /// reference ([`crate::Dataset::dereference`]) was used after the file
    /// changed under it.
    ///
    /// Such a handle stores only the object-header address the reference gave
    /// it, and an edit can rewrite and relocate object headers, so there is no
    /// name left to look the object up by. Every handle opened by *path*
    /// re-resolves itself instead and never reports this. Dereference again from
    /// a fresh read to get a handle onto the current file.
    ///
    /// An immediate [`crate::Dataset::append`] rewrites a header where it stands
    /// and does not end such a handle. A commit does, as does staging one,
    /// [`crate::File::sync`], and [`crate::File::close`].
    StaleHandle,
    /// The file or dataset is not a supported target for the SWMR append writer
    /// (e.g. a userblock or non-latest-format file, or a dataset that is
    /// filtered, not rank-1 with an unlimited dimension, or not
    /// Extensible-Array indexed). The payload is a human-readable reason.
    SwmrAppendUnsupported(&'static str),
    /// A staged edit (`write` / `set_attr` / `create_*` / `delete` / `copy` /
    /// `commit`) was requested on a file opened with
    /// [`crate::File::open_swmr_writer`], which permits only immediate
    /// [`crate::Dataset::append`]. Committing a structural edit would clear the
    /// SWMR-write flag out from under a concurrent reader, so the whole staged
    /// surface is rejected in SWMR-writer mode.
    SwmrStagedUnsupported,
    /// A SWMR operation (e.g. [`crate::File::refresh`]) was requested on a file
    /// that was not opened for SWMR reading via `File::open_swmr`.
    SwmrUnsupported,
    /// An object graph holds more objects or hard links than a traversal of it reads, or nests
    /// groups deeper than a repack writes.
    ///
    /// [`crate::File::copy_from`], and the [`crate::File::commit`] that applies a
    /// [`crate::File::copy`], return this for a source that reaches more objects or hard links
    /// than a copy reads. [`crate::repack()`], [`crate::repack_into`], and
    /// [`crate::repack_to_path`] return it for a source that reaches more than a repack reads, or
    /// that nests groups deeper than a repack writes once the links the drop paths name are
    /// removed.
    TraversalLimitExceeded {
        /// The count the graph exceeds.
        limit: TraversalLimit,
        /// The most objects or hard links the traversal reads, or the most levels of nested
        /// groups a repack writes.
        maximum: usize,
    },
}

#[cfg(feature = "std")]
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::AllocationPublicationUncertain => write!(
                f,
                "allocation publication is unconfirmed; close and reopen the file before editing"
            ),
            Error::AttributeDecode { name, source } => {
                write!(f, "cannot decode attribute {name:?}: {source}")
            }
            Error::HeapCompactionFailed {
                completed,
                state,
                source,
            } => {
                let state = match state {
                    HeapCompactionState::Restored => {
                        "the failed collection is unchanged or restored"
                    }
                    HeapCompactionState::Uncertain => {
                        "restoration or allocation permission is uncertain"
                    }
                };
                write!(
                    f,
                    "global heap compaction failed after {completed} completed collections: {source}. {state}"
                )
            }
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Format(e) => write!(f, "HDF5 format error: {e}"),
            Error::InconsistentReferenceCount {
                address,
                observed,
                minimum,
            } => write!(
                f,
                "object at {address} has reference count {observed}, below the known minimum {minimum}"
            ),
            Error::InvalidFractalHeapStorage { address } => {
                write!(f, "invalid fractal heap storage at {address}")
            }
            Error::InvalidSpaceOwnershipExtent {
                address,
                length,
                previous_end,
                logical_size,
            } => {
                write!(
                    f,
                    "storage extent at {address} of length {length} overlaps the previous end {previous_end} or exceeds logical EOF {logical_size}"
                )
            }
            Error::NotADataset(path) => write!(f, "not a dataset: {path}"),
            Error::NotAGroup(path) => write!(f, "not a group: {path}"),
            Error::NotANamedDatatype(path) => write!(f, "not a named datatype: {path}"),
            Error::MissingMessage(mt) => write!(f, "missing required message: {mt}"),
            Error::Shape(msg) => write!(f, "array shape error: {msg}"),
            Error::SwmrUnsupported => write!(
                f,
                "refresh requires a file opened with File::open_swmr (live handle)"
            ),
            Error::HandlesOutstanding => write!(
                f,
                "operation needs exclusive file access: drop outstanding Dataset/Group handles and File clones first"
            ),
            Error::FreeSectionInspectionUnsupported => {
                write!(
                    f,
                    "persisted free sections cannot be inspected on a writable handle"
                )
            }
            Error::ReadOnly => write!(
                f,
                "cannot write to a read-only file; open it with File::open_rw"
            ),
            Error::StaleHandle => write!(
                f,
                "this handle was reached by object reference and the file has changed since; \
                 it has no path to re-resolve, so dereference again"
            ),
            Error::OverlappingFreeSections {
                address,
                length,
                previous_address,
                previous_length,
            } => write!(
                f,
                "the free section at {address}, {length} bytes long, overlaps the free section at {previous_address}, {previous_length} bytes long"
            ),
            Error::PersistedFreeSpace {
                manager_slot,
                header_address,
                source,
                ..
            } => write!(
                f,
                "persisted free-space manager in slot {manager_slot} at stored address {}: {source}",
                header_address.get()
            ),
            Error::NotCommitted(path) => write!(
                f,
                "\"{path}\" is staged and not written yet; File::commit publishes it"
            ),
            Error::StagingWithdrawn(path) => write!(
                f,
                "this handle was made onto the staged creation of \"{path}\", and that staging \
                 was withdrawn before any commit; look the path up again to reach what it means now"
            ),
            Error::FileClosed => write!(
                f,
                "cannot write through a handle after File::close; re-open the file to modify it"
            ),
            Error::SwmrStagedUnsupported => write!(
                f,
                "a file opened with File::open_swmr_writer allows only immediate Dataset::append, not staged edits"
            ),
            Error::SwmrAppendUnsupported(reason) => {
                write!(f, "unsupported SWMR append target: {reason}")
            }
            Error::AppendUnsupported(reason) => {
                write!(f, "unsupported append target: {reason}")
            }
            Error::AppendInPlaceUnsupported(reason) => {
                write!(f, "unsupported in-place append target: {reason}")
            }
            Error::EditStorageConflict(conflict) => write!(f, "edit storage conflict: {conflict}"),
            Error::EditUnsupported(reason) => {
                write!(f, "unsupported in-place edit target: {reason}")
            }
            Error::CopyScreenUnreadable { message, source } => write!(
                f,
                "a copy in this commit carries a {message} message that could not be read \
                 ({source}), so its elements cannot be screened against the same commit's \
                 deletions; stage the copy and the deletions in separate commits"
            ),
            Error::RepackUnsupported(reason) => {
                write!(f, "cannot repack faithfully: {reason}")
            }
            Error::FileLocked(reason) => write!(f, "file is locked: {reason}"),
            Error::FileMarkedInUse(reason) => write!(f, "file is marked in use: {reason}"),
            Error::TraversalLimitExceeded {
                limit: TraversalLimit::Depth,
                maximum,
            } => write!(
                f,
                "the object graph nests groups more than {maximum} levels deep, the most a \
                 repack writes"
            ),
            Error::TraversalLimitExceeded { limit, maximum } => write!(
                f,
                "the object graph holds more than {maximum} {limit}, the most a traversal reads"
            ),
            Error::CommitPartiallyApplied { refusal, restore } => write!(
                f,
                "an operation failed ({refusal}) and restoration could not be confirmed ({restore})"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Format(e) => Some(e),
            Error::EditStorageConflict(conflict) => Some(conflict),
            Error::AttributeDecode { source, .. } => Some(source),
            Error::CopyScreenUnreadable { source, .. } => Some(source),
            Error::PersistedFreeSpace { source, .. } => Some(&**source),
            Error::CommitPartiallyApplied { restore, .. } => Some(&**restore),
            Error::HeapCompactionFailed { source, .. } => Some(&**source),
            _ => None,
        }
    }
}

#[cfg(feature = "std")]
impl From<EditStorageConflict> for Error {
    fn from(error: EditStorageConflict) -> Self {
        Self::EditStorageConflict(error)
    }
}

#[cfg(feature = "std")]
impl From<FormatError> for Error {
    fn from(e: FormatError) -> Self {
        Error::Format(e)
    }
}

#[cfg(feature = "std")]
impl From<FreeSectionsError> for Error {
    fn from(error: FreeSectionsError) -> Self {
        match error {
            FreeSectionsError::Format(error) => Self::Format(error),
            FreeSectionsError::Overlap {
                address,
                length,
                previous_address,
                previous_length,
            } => Self::OverlappingFreeSections {
                address,
                length,
                previous_address,
                previous_length,
            },
        }
    }
}

#[cfg(feature = "std")]
impl From<ResolveError> for Error {
    fn from(e: ResolveError) -> Self {
        match e {
            ResolveError::NotAGroup(path) => Error::NotAGroup(path),
            ResolveError::Format(e) => Error::Format(e),
            ResolveError::NotFound(path) => Error::Format(FormatError::PathNotFound(path)),
        }
    }
}

#[cfg(feature = "std")]
impl From<TraversalError<Error>> for Error {
    fn from(e: TraversalError<Error>) -> Self {
        match e {
            TraversalError::Expand(e) => e,
            TraversalError::LimitExceeded { limit, maximum } => {
                Error::TraversalLimitExceeded { limit, maximum }
            }
        }
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// The restoration and allocation state after global heap compaction fails.
///
/// Reported by [`Error::HeapCompactionFailed`]. The editor preserves completed collection
/// compactions in both states.
#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeapCompactionState {
    /// The failed collection is unchanged or restored, and allocation permission is available.
    Restored,
    /// Restoration or allocation permission could not be confirmed.
    ///
    /// Further appends, commits, and explicit synchronization in this editing session return
    /// [`Error::AllocationPublicationUncertain`].
    Uncertain,
}
