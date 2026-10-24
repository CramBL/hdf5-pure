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
use crate::message_type::MessageType;

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
    /// The dataset or file is not a supported target for the fast, immediate
    /// in-place append
    /// ([`Dataset::append`](crate::Dataset::append)) - for
    /// example a userblock or non-latest-format file, a dataset whose
    /// Extensible-Array index is not yet allocated, one that is not rank-1 /
    /// unlimited / Extensible-Array indexed, one reachable through more than one
    /// hard link, or a path an uncommitted staged edit in the same session will
    /// relocate or delete. Distinct from [`AppendUnsupported`](Self::AppendUnsupported)
    /// so a caller can catch this fast-path rejection and fall back to the staged
    /// [`Dataset::append_staged`](crate::Dataset::append_staged). The
    /// payload is a human-readable reason.
    AppendInPlaceUnsupported(&'static str),
    /// The dataset is not a supported target for
    /// [`Dataset::append_staged`](crate::Dataset::append_staged) - for
    /// example a dataset that is not chunked, not extensible along its first
    /// dimension, not indexed by an Extensible Array, higher than rank 1, uses a
    /// filter this engine cannot re-encode, has a big-endian on-disk element
    /// datatype (for a raw append), or has more than one hard link. The payload
    /// is a human-readable reason.
    AppendUnsupported(&'static str),
    /// A [`commit`](crate::File::commit) failed, *and* could not put back a
    /// value it had already written over - so the file holds part of a batch
    /// that was rejected.
    ///
    /// Every other edit a commit applies lands where nothing reaches it until
    /// the superblock is repointed, so a commit that stops short of that leaves
    /// the file exactly as it found it. A same-length value overwrite is the
    /// exception: it writes straight over the dataset's existing data block,
    /// which the live root already reaches. A rejected commit therefore replays
    /// the prior bytes over each such write before returning, and this is what
    /// it returns instead when that replay itself failed.
    ///
    /// It is the one rejection after which a caller must **re-read**: the
    /// datasets the batch overwrote may hold either value.
    ///
    /// Both errors are carried because in the shape this is most likely to take
    /// they say different things: a commit rejected for a reason the caller can
    /// act on, followed by an I/O failure that prevented the restore. Reporting
    /// only the second would leave the caller retrying a batch without knowing
    /// what was wrong with it.
    CommitPartiallyApplied {
        /// Why the commit was rejected - what a caller must fix before staging
        /// the batch again.
        refusal: Box<Error>,
        /// The write failure that then prevented the prior values being put
        /// back. This is the one that proves the file changed, so it is what
        /// [`source`](std::error::Error::source) reports.
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
    /// The file could not be opened because its superblock's status-flags byte
    /// marks it as held by a writer - the durable flag
    /// [`crate::File::open_swmr_writer`] raises, and which a page-buffered
    /// session ([`crate::FileAccessProperties::with_page_buffer_size`]) raises
    /// alone. Unlike [`FileLocked`](Self::FileLocked) this outlives the process
    /// that set it, so it means either that a writer is active *or* that one
    /// exited without clearing it. The payload names
    /// [`crate::File::clear_swmr_flag`] (the `h5clear -s` equivalent) as the
    /// recovery for the latter.
    ///
    /// Which reader can get at the file depends on which mark it is, and the
    /// payload identifies the one that applies:
    ///
    /// - both bits, a live SWMR writer: follow it with
    ///   [`crate::File::open_swmr`].
    /// - the write bit alone, which no SWMR reader can follow: read a snapshot
    ///   with
    ///   [`crate::FileAccessProperties::with_write_mark_policy`], once the
    ///   writer has synced or closed.
    ///
    /// The payload is a human-readable reason.
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
    /// The object at the given path is not a group. A name that resolves to
    /// nothing at all is [`FormatError::PathNotFound`] instead.
    ///
    /// The path may be an *intermediate* component of the one that was asked
    /// for: resolving `a/b/c` opens `a` and then `a/b` to look inside them, so
    /// a dataset at `a/b` reports `NotAGroup("a/b")` rather than anything about
    /// `a/b/c`.
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
}

#[cfg(feature = "std")]
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Format(e) => write!(f, "HDF5 format error: {e}"),
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
            Error::CommitPartiallyApplied { refusal, restore } => write!(
                f,
                "a commit refused ({refusal}) could not restore a value it had overwritten \
                 ({restore}), so the file holds part of the refused batch"
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
            Error::CopyScreenUnreadable { source, .. } => Some(source),
            Error::CommitPartiallyApplied { restore, .. } => Some(&**restore),
            _ => None,
        }
    }
}

#[cfg(feature = "std")]
impl From<FormatError> for Error {
    fn from(e: FormatError) -> Self {
        Error::Format(e)
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
impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
