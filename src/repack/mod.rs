//! Whole-file repack (issue #21): copy an existing HDF5 file into a fresh,
//! compact one, optionally dropping objects.
//!
//! [`File::open_rw`](crate::File::open_rw) deletes objects in place but reclaims
//! space only within a session and cannot return a single deleted-and-closed
//! file's bytes to the OS. Repack is the complementary answer — the same one the
//! HDF5 C ecosystem ships as `h5repack`: it reads every surviving object and
//! rewrites the whole file. The writer plans destination allocations, then reconstructs each
//! object's metadata for emission under the selected file-space settings.
//!
//! # Fidelity contract
//!
//! Repack never silently degrades data. Every surviving object is reproduced
//! faithfully — datatype, shape, max-shape, chunking, filters, and byte-exact
//! element data — or the whole operation fails with [`Error::RepackUnsupported`]
//! naming the object and the reason. It refuses rather than approximate.
//! Currently reproducible:
//!
//! - Datasets with fixed-point, floating-point, time, fixed-length string,
//!   bit-field, opaque, compound, enumeration, array, and native complex datatypes,
//!   contiguous/compact or chunked.
//! - **Chunked** datasets copy their compressed chunks **verbatim** (chunk by
//!   chunk, never decoded), so *every* filter is preserved byte-exact: deflate,
//!   shuffle, fletcher32, integer **and** float scale-offset, ZFP, SZIP, and
//!   even filters this crate cannot itself apply. The destination always uses a
//!   v4 chunk index (single-chunk / fixed-array / extensible-array) regardless
//!   of the source index type.
//! - **Variable-length** datasets (1D and ND, contiguous/compact as well as
//!   chunked, filtered, or resizable): string-shaped
//!   ([`VariableLengthType::String`](crate::VariableLengthType::String) and the
//!   MATLAB VLEN-of-1-byte-ASCII-string shape) and non-string sequences over any
//!   base type that embeds no addresses. Each element's exact heap bytes are read
//!   and re-staged through a fresh global heap, preserving charset, padding, the
//!   null-vs-empty distinction, embedded NULs, and non-UTF-8 payloads, and the
//!   source's chunk geometry, filters, and resizability are carried onto the
//!   rebuilt dataset.
//! - Datatypes that *contain* an address without being one — a compound with a
//!   variable-length member, an object-reference member, or both; an array of
//!   such compounds; and nesting of either. The embedded heap references are
//!   re-staged and the embedded object addresses resolved exactly as their
//!   top-level counterparts are, in a single pass, while every other byte of the
//!   element is carried through untouched.
//! - Contiguous/compact **object-reference** datasets, and contiguous/compact
//!   datatypes containing an object-reference member: each stored address is
//!   rewritten to its target object's new location in the compacted file (null
//!   and undefined references are carried verbatim). A reference to a committed
//!   datatype object is one of those targets, since it is an object like any
//!   other.
//! - **Committed (`H5Tcommit`) datatypes**: the named datatype object is
//!   recreated at the same path, and every dataset and attribute that named it
//!   names the *same* object in the output rather than getting an inline copy of
//!   the encoding, so `h5dump` still reports `DATATYPE "/mytype"` and the
//!   object's reference count still matches its users.
//! - **Unallocated storage**: destination element storage remains unallocated. A chunked
//!   dataset's filter message is copied verbatim, including lossy and unsupported
//!   filters. A resizable destination with zero elements can retain a deferred Extensible
//!   Array index, while nonempty resizable output receives an allocated chunk index for in-place appends.
//! - Group hierarchy up to 128 levels deep, with encoded link names, soft and external targets,
//!   character sets, and link creation-order metadata preserved.
//! - Attributes, on datasets, groups, and root, carried across with the
//!   encoding the source gave them rather than rebuilt from an [`AttrValue`],
//!   which is a decoded view and cannot express a byte order, a sub-width
//!   precision, a variable-length string, or a rank above one.
//! - The source file's effective file-space settings, including persistence, unless
//!   [`RepackOptions::with_file_space_settings`] selects an output policy. The writer
//!   rebuilds persistent free-space metadata for the destination layout.
//!
//! The verbatim chunk copy never decodes, so it eliminates the
//! decompress→recompress round-trip and the per-dataset decompression blowup,
//! and a lossy filter survives byte-exact. Two paths still re-encode and so
//! require **lossless** filters: a *contiguous/compact* filtered dataset, and a
//! *partly allocated* chunked dataset (one with both stored chunks and unallocated
//! chunk-grid entries). A lossy pipeline on either of those is rejected.
//!
//! Refused (named, never dropped silently): chunked, filtered, or resizable
//! datasets whose datatype is or contains an object reference (their element
//! addresses are resolved as elements are re-staged, which a compressed chunk
//! would need rewritten in place);
//! region references and
//! non-8-byte object references; an object reference to a dropped object or to a
//! target outside the hard-link hierarchy (a dangling or region target), and
//! object references in a userblock file (non-zero base address); a
//! non-string vlen sequence whose base type embeds an address (nested vlen or
//! reference); a virtual data layout, and external data storage
//! (`H5Pset_external`), whose bytes live in files this crate does not read; a
//! lossy filter on the
//! contiguous re-encode or sparse-chunked fallback path; an attribute whose
//! datatype is or contains a reference (its stored address is not rewritten yet,
//! and no [`AttrValue`] can re-encode it); and a use of a committed datatype the
//! repack drops or that no hard link reaches. An object that cannot be
//! reproduced fails the repack by name rather than being silently dropped.
//!
//! One thing is resolved rather than reproduced: a *shared* message, which only
//! a file with a shared-message (SOHM) table has, is written back inline. Unlike
//! a committed datatype, a message in the shared-message heap has no name and no
//! HDF5 call reports one as shared, so there is nothing an inline copy loses —
//! the repacked file holds the same messages, one copy per user, and shares
//! nothing.
//!
//! # Memory
//!
//! The writer retains discovery records, allocation spans, and serialized file-space managers.
//! It reconstructs one object's metadata at a time during planning and emission. Working memory
//! also includes source caches, the current object's heap collections, and its storage buffers.
//!
//! Eligible chunked datasets use one encoded chunk buffer during verbatim data emission.
//! Re-encoded datasets and datasets with address-bearing elements can require complete dataset
//! buffers. A single large object or chunk index can therefore require substantial memory.
//!
//! Planning and emission read metadata in separate passes, which increases source I/O and
//! preparation work. The writer retains parsed discovery headers within a 16 MiB budget for
//! initial planning. The budget includes cache entry, message-vector, and message-body capacities.
//! Cached headers are consumed during planning, and remaining cache storage is released before
//! final layout checks and emission. [`File::open_streaming`] reads the source on demand.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use crate::DeflateLevel;
use crate::FileSpaceSettings;
#[cfg(test)]
use crate::VariableLengthStringType;
#[cfg(test)]
use crate::VariableLengthType;
use crate::access_mode::AccessMode;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::attribute::AttributeMessage;
use crate::chunked_read::ChunkInfo;
use crate::chunked_write::PartialEdgeFiltering;
use crate::chunked_write::{ChunkMeta, ChunkProvider, FilterKind, FilterSpec};
use crate::convert::Narrow;
use crate::data_layout::DataLayout;
use crate::dataspace::MaxExtent;
use crate::datatype::{self, Datatype, ReferenceType};
use crate::error::{Error, FormatError};
use crate::file_writer::source::Object;
use crate::file_writer::source::ObjectSource;
use crate::file_writer::source::Plan;
use crate::file_writer::source::SourceLink;
use crate::filter_pipeline::{
    FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_LZF, FILTER_SCALEOFFSET, FILTER_SHUFFLE,
    FilterPipeline,
};
use crate::libver::LibVer;
use crate::object_graph::TraversalBudget;
use crate::object_path::ObjectTarget;
use crate::object_path::{LinkNameBuf, ObjectPathBuf};
use crate::reader::Dataset;
use crate::reader::File;
use crate::reader::ObjectClass;
use crate::repack::objects::Placement;
use crate::repack::objects::SourceObjects;
use crate::scaleoffset::{self, ScaleOffset};
use crate::shared_message::DatatypeLocation;
use crate::source::Source;
use crate::type_builders::{
    AttrValue, DatasetBuilder, GroupBuilder, ObjectRefPatch, ObjectRefTarget, VlStringElement,
};
use crate::vl_data::{self, EmbeddedVlSlot, VlByteObject, VlenStringReadOptions};
use crate::width::OffsetWidth;
use crate::writer::FileBuilder;
#[cfg(test)]
use crate::{FixedPointType, TimeType};

mod headers;
mod objects;
#[cfg(test)]
test_util_hdf5::hdf5_1_10_tests! {
    #[cfg(test)]
    mod native_crosschecks;
}
#[cfg(test)]
mod test_util;

/// Options controlling a [`repack`].
///
/// Built with [`new`](Self::new) and [`drop_path`](Self::drop_path); the fields
/// are private so a future option is an additive change.
#[derive(Debug, Default, Clone)]
pub struct RepackOptions {
    /// Paths of the objects to omit from the output, each from the root group as
    /// [`drop_path`](Self::drop_path) parsed it.
    drop: Vec<String>,
    /// Library-version bounds for the output, from
    /// [`with_libver_bounds`](Self::with_libver_bounds). `None` carries the
    /// source's own format forward.
    libver_bounds: Option<(LibVer, LibVer)>,
    /// The destination allocation policy, or `None` to use the source's recorded settings.
    file_space_settings: Option<FileSpaceSettings>,
    /// Whether the source's object headers are parsed as a writer parses them, from
    /// [`reject_unknown_messages_only_a_writer_must_understand`](Self::reject_unknown_messages_only_a_writer_must_understand).
    /// See [`source_access_mode`](Self::source_access_mode) for the mode it selects.
    reject_unknown_messages_only_a_writer_must_understand: bool,
}

impl RepackOptions {
    /// Options that drop nothing — a pure compaction copy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Omits the object at `path` from the output, such as `"grp/old"`, and its whole subtree
    /// where that object is a group.
    ///
    /// `path` identifies an object whichever way it is spelled, as it does for
    /// [`Group::dataset`](crate::Group::dataset).
    ///
    /// Every path must identify an object in the source, and a path that identifies the root
    /// group is outside what a repack drops. [`repack`](fn@crate::repack) reports
    /// [`Error::RepackUnsupported`] for either.
    ///
    /// Chainable.
    pub fn drop_path(mut self, path: &str) -> Self {
        self.drop.push(ObjectPathBuf::parse(path).to_string());
        self
    }

    /// Returns the paths this repack omits, each from the root group, in the order they were
    /// added.
    pub fn drop_paths(&self) -> &[String] {
        &self.drop
    }

    /// Write the output in the format these bounds select, rather than in the
    /// source's own format. Same meaning as
    /// [`FileBuilder::with_libver_bounds`](crate::FileBuilder::with_libver_bounds):
    /// `high` picks the format, and content it cannot express is refused with
    /// [`FormatError::LibverTooOldForContent`] rather than written in a newer
    /// one.
    ///
    /// A repack carries the source's format forward by default, because
    /// compacting a file is not a request to re-target it — but it does upgrade
    /// where the content leaves no choice, since a chunked dataset needs the 1.10
    /// chunk indices and there is no older one this crate writes. Set this to make
    /// the format a guarantee instead: an `Earliest..=V18` bound turns that
    /// upgrade into an error, so a file that must stay loadable by an old reader
    /// says so rather than finding out later.
    ///
    /// ```no_run
    /// use hdf5_pure::{LibVer, RepackOptions, repack};
    ///
    /// // Fails rather than producing a file MATLAB before R2021b cannot open.
    /// let options = RepackOptions::new().with_libver_bounds(LibVer::Earliest, LibVer::V18);
    /// repack("data.mat", "compact.mat", &options)?;
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    ///
    /// Chainable.
    pub fn with_libver_bounds(mut self, low: LibVer, high: LibVer) -> Self {
        self.libver_bounds = Some((low, high));
        self
    }

    /// The library-version bounds this repack will write to, or `None` when it
    /// carries the source's format forward.
    pub fn libver_bounds(&self) -> Option<(LibVer, LibVer)> {
        self.libver_bounds
    }

    /// Selects the allocation policy for the destination file.
    ///
    /// By default, a repack preserves the source's recorded effective settings, including
    /// persistence. If the source has no recorded settings, the writer uses the
    /// [`FileBuilder`] default. The writer rebuilds free-space metadata for the destination
    /// layout under the selected policy.
    ///
    /// The writer applies the same validation as [`FileBuilder::with_file_space_settings`]
    /// before writing destination bytes. A repack returns
    /// [`FormatError::LibverTooOldForContent`] if the selected policy requires a format newer
    /// than the explicit [`with_libver_bounds`](Self::with_libver_bounds).
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::{FileSpaceSettings, FileSpaceStrategy, RepackOptions};
    ///
    /// let settings = FileSpaceSettings::new(FileSpaceStrategy::None);
    /// let options = RepackOptions::new().with_file_space_settings(settings);
    /// assert_eq!(options.file_space_settings(), Some(settings));
    /// ```
    pub fn with_file_space_settings(mut self, settings: FileSpaceSettings) -> Self {
        self.file_space_settings = Some(settings);
        self
    }

    /// Returns the destination allocation override, or `None` to preserve the source settings.
    ///
    /// With `None`, the writer uses its default when the source has no recorded settings.
    pub fn file_space_settings(&self) -> Option<FileSpaceSettings> {
        self.file_space_settings
    }

    /// Rejects a source holding an object header message of a type this crate cannot
    /// name that only a decoder with write access must understand.
    ///
    /// The flag is `H5O_MSG_FLAG_FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE`, bit 3 of the
    /// message record's flags. The source is opened read-only either way, and this
    /// option parses its object headers as a writer does, so
    /// [`repack`](fn@crate::repack) reports [`FormatError::UnsupportedMessage`] with
    /// the message's type before any byte reaches the destination. A message flagged
    /// `H5O_MSG_FLAG_FAIL_IF_UNKNOWN_ALWAYS` is rejected with this option and
    /// without it.
    ///
    /// The default is what `h5repack` does. `h5repack` also reads its input
    /// read-only, and this crate rebuilds every object without a message it cannot
    /// name. This option is this crate's own, and `h5repack` has no counterpart for
    /// it. Set it where a repack must fail on a message the source marked for a
    /// decoder with write access.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # let dir = tempfile::tempdir()?;
    /// # let src = dir.path().join("data.h5");
    /// # let dst = dir.path().join("compact.h5");
    /// # let mut builder = hdf5_pure::FileBuilder::new();
    /// # builder.create_dataset("values").with_i32_data(&[1, 2, 3]);
    /// # builder.write(&src)?;
    /// use hdf5_pure::{File, RepackOptions, repack};
    ///
    /// // Rejects a source holding a message flagged for a decoder with write access.
    /// let options = RepackOptions::new().reject_unknown_messages_only_a_writer_must_understand();
    /// repack(&src, &dst, &options)?;
    ///
    /// assert_eq!(File::open(&dst)?.dataset("values")?.read_i32()?, vec![1, 2, 3]);
    /// # Ok::<(), hdf5_pure::Error>(())
    /// ```
    pub fn reject_unknown_messages_only_a_writer_must_understand(mut self) -> Self {
        self.reject_unknown_messages_only_a_writer_must_understand = true;
        self
    }

    /// Returns `true` if this repack reads its source as a writer would, which
    /// [`reject_unknown_messages_only_a_writer_must_understand`](Self::reject_unknown_messages_only_a_writer_must_understand)
    /// sets.
    pub fn rejects_unknown_messages_only_a_writer_must_understand(&self) -> bool {
        self.reject_unknown_messages_only_a_writer_must_understand
    }

    /// The mode the source's object headers are parsed under: [`AccessMode::ReadWrite`]
    /// with [`reject_unknown_messages_only_a_writer_must_understand`](Self::reject_unknown_messages_only_a_writer_must_understand)
    /// set, which is the mode a parse rejects a message of an unknown type flagged for
    /// a decoder with write access under.
    pub(crate) fn source_access_mode(&self) -> AccessMode {
        if self.reject_unknown_messages_only_a_writer_must_understand {
            AccessMode::ReadWrite
        } else {
            AccessMode::ReadOnly
        }
    }
}

/// Repack `src` into a new file at `dst`, applying `options`.
///
/// Reads every object of `src` not excluded by [`RepackOptions::drop_path`] and
/// writes them into a fresh, compact file at `dst`. On success `dst` is a normal
/// HDF5 file holding the surviving objects, allocated under the destination file-space settings.
///
/// Preserves the source's recorded effective file-space settings by default, including
/// persistence. [`RepackOptions::with_file_space_settings`] selects a different output policy.
/// Retains each group's link thresholds and stored entry estimates when reconstructing its header.
///
/// Fidelity checks and destination layout planning run before output. An
/// [`Error::RepackUnsupported`] for an object or drop path is reported before bytes are written
/// to `dst`. The writer rereads object metadata during emission and streams eligible encoded
/// chunks from `src`. An I/O error reading the source or writing the destination can leave a
/// partial `dst`.
///
/// See [`Error::RepackUnsupported`] for the objects that cannot be reproduced
/// faithfully.
///
/// `src` is opened for streaming reads, so a file whose superblock marks it as
/// held by a writer is rejected with [`Error::FileMarkedInUse`], since repacking
/// a file a writer is still growing would capture a torn view. Clear a flag a
/// stopped writer left with [`File::clear_status_flags`](crate::File::clear_status_flags)
/// before repacking. Clearing the flags does not guarantee data integrity.
///
/// The source's object headers are parsed read-only by default, so a message of a
/// type this crate cannot name that only a decoder with write access must
/// understand is read past, and `repack` writes the object into `dst` without it,
/// wherever the message sits. `h5repack` opens its input read-only as well and
/// rebuilds a group or a named datatype, so both tools drop such a message from
/// those headers. A dataset that `h5repack` copies with `H5Ocopy` keeps the message
/// in the output (`h5repack_copy.c` and `H5Ocopy.c`, HDF5 2.2.0).
///
/// [`RepackOptions::reject_unknown_messages_only_a_writer_must_understand`] parses
/// them as a writer does, so such a message is reported as
/// [`FormatError::UnsupportedMessage`] before any byte reaches `dst`.
///
/// # Errors
///
/// Returns [`Error::RepackUnsupported`] if an object of `src` cannot be reproduced faithfully, if
/// a drop path matches no link, or if, after the drop paths, an object of `src` has more than one
/// path, through an alias or a cycle, since a repack writes each object at one path. Returns
/// [`Error::TraversalLimitExceeded`] if `src` reaches more than 16,777,216 objects or hard links,
/// or if, after the drop paths, `src` nests groups more than 128 levels deep.
/// Returns [`Error::RepackUnsupported`] for group link filters, including those configured on
/// compact groups before dense storage is allocated.
pub fn repack<P: AsRef<Path>, Q: AsRef<Path>>(
    src: P,
    dst: Q,
    options: &RepackOptions,
) -> Result<(), Error> {
    // Read source metadata and payloads on demand.
    let source = File::open_streaming_with_access_mode(src.as_ref(), options.source_access_mode())?;
    stage(&source, options, REPACK_BUDGET)?.write(dst)
}

/// Repacks the open file `source` into `writer` with `options`.
///
/// Performs what [`repack`](fn@repack) does, with the same fidelity checks and errors. Writes
/// the file in address order into `writer`, which may be an in-memory buffer, a pipe, or an
/// open file. The caller must keep source contents unchanged during planning and emission.
///
/// An [`Error::RepackUnsupported`] is reported before bytes reach `writer`. If an
/// I/O error happens partway through, it leaves what was already written there.
///
/// # Errors
///
/// Returns the errors of [`repack`](fn@repack). Returns [`Error::RepackUnsupported`] if
/// `source` has staged edits ([`File::has_staged_edits`]), or if `options` sets
/// [`RepackOptions::reject_unknown_messages_only_a_writer_must_understand`]
/// and `source` was opened read-only, because its object headers are parsed in the mode the
/// file was opened in.
///
/// # Examples
///
/// ```rust
/// # let dir = tempfile::tempdir()?;
/// # let path = dir.path().join("data.h5");
/// # let mut builder = hdf5_pure::FileBuilder::new();
/// # builder.create_dataset("keep").with_i32_data(&[1, 2, 3]);
/// # builder.create_dataset("drop").with_f64_data(&[0.0; 64]);
/// # builder.write(&path)?;
/// use hdf5_pure::{File, RepackOptions, repack_into};
///
/// // Repacks into memory, leaving out `drop`.
/// let source = File::open_streaming(&path)?;
/// let mut bytes = Vec::new();
/// repack_into(&source, &mut bytes, &RepackOptions::new().drop_path("drop"))?;
///
/// assert_eq!(File::from_bytes(bytes)?.root().datasets()?, ["keep"]);
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
pub fn repack_into(
    source: &File,
    writer: impl Write,
    options: &RepackOptions,
) -> Result<(), Error> {
    check_source_access_mode(source, options)?;
    stage(source, options, REPACK_BUDGET)?.finish_to(writer)
}

/// Repacks the open file `source` into a new file at `path`, applying `options`.
///
/// Does what [`repack`](fn@repack) does for a source the caller has already opened.
///
/// # Errors
///
/// Returns the errors of [`repack_into`].
pub fn repack_to_path(
    source: &File,
    path: impl AsRef<Path>,
    options: &RepackOptions,
) -> Result<(), Error> {
    check_source_access_mode(source, options)?;
    stage(source, options, REPACK_BUDGET)?.write(path)
}

/// Returns an error if `options` requests the source's object headers to be parsed as a
/// writer parses them and `source` was opened in another mode.
fn check_source_access_mode(source: &File, options: &RepackOptions) -> Result<(), Error> {
    if options.source_access_mode() == AccessMode::ReadWrite
        && source.access_mode() != AccessMode::ReadWrite
    {
        return Err(Error::RepackUnsupported(String::from(
            "reject_unknown_messages_only_a_writer_must_understand needs a source opened for \
             writing",
        )));
    }
    Ok(())
}

/// Discovers retained objects and checks fidelity while planning destination allocations.
///
/// The repack reads no more objects and hard links of `source` than `budget` allows.
///
/// # Errors
///
/// Returns [`Error::RepackUnsupported`] if `options` drops the root group, if the File Space Info
/// of `source` does not parse and no output override is set, or if an object of `source` cannot
/// be reproduced faithfully.
/// Returns the errors of [`SourceObjects::discover`].
fn stage(
    source: &File,
    options: &RepackOptions,
    budget: TraversalBudget,
) -> Result<Plan<RepackSource>, Error> {
    let RepackOptions {
        drop,
        libver_bounds,
        file_space_settings,
        reject_unknown_messages_only_a_writer_must_understand: _,
    } = options;
    let file = Arc::new(source.clone());
    let drop: BTreeSet<ObjectPathBuf> =
        drop.iter().map(|path| ObjectPathBuf::parse(path)).collect();
    if drop.iter().any(ObjectPathBuf::is_empty) {
        return Err(Error::RepackUnsupported(String::from(
            "a repack cannot drop the root group",
        )));
    }
    let settings = match *file_space_settings {
        Some(settings) => Some(settings),
        None => file.file_space_settings().map_err(|err| match err {
            Error::Format(err) => Error::RepackUnsupported(format!(
                "the File Space Info of the source cannot be read: {err}"
            )),
            err => err,
        })?,
    };
    let objects = SourceObjects::discover(&file, &drop, budget)?;
    let inherited = file.libver_bound().max(LibVer::WRITER_OLDEST);
    Plan::new(
        RepackSource { file, objects },
        settings,
        *libver_bounds,
        inherited,
    )
}

/// The source file and retained object identities used to reconstruct repack metadata.
struct RepackSource {
    file: Arc<File>,
    objects: SourceObjects,
}

impl ObjectSource for RepackSource {
    fn len(&self) -> usize {
        self.objects.len()
    }

    fn load(&self, index: usize) -> Result<Object, Error> {
        let Self { file, objects } = self;
        let object = objects.object(index)?;
        let path = objects.path(index)?;
        match object.class {
            ObjectClass::Dataset => {
                let dataset = file.dataset_at(
                    object.id.header_address(),
                    path.clone(),
                    objects.take_header(index)?,
                )?;
                let mut builder = DatasetBuilder::new(
                    path.components()
                        .last()
                        .ok_or_else(|| FormatError::Internal("a dataset has no link name".into()))?
                        .as_str(),
                );
                emit_dataset(&mut builder, &dataset, &path, file, objects)?;
                Ok(Object::dataset(builder)?)
            }
            ObjectClass::Group => {
                let mut builder = GroupBuilder::new("repack");
                let (attributes, group_info) = file.attr_messages_and_group_info_at(
                    object.id.header_address(),
                    objects.take_header(index)?,
                )?;
                copy_attrs(
                    &mut builder,
                    attributes,
                    || {
                        file.group_at(object.id.header_address(), path.clone())
                            .attrs()
                    },
                    &group_owner(&path),
                    objects,
                )?;
                Ok(Object::group(
                    &builder.finish().attrs,
                    group_info,
                    object.creation_order,
                )?)
            }
            ObjectClass::NamedDatatype => {
                let datatype = file
                    .named_datatype_at(object.id.header_address(), objects.take_header(index)?)?;
                check_datatype(&datatype, &format!("committed datatype {path}"))?;
                Ok(Object::committed(datatype)?)
            }
            ObjectClass::Unknown => Err(Error::RepackUnsupported(format!(
                "{}: the link {:?} refers to an object that is neither a group, a dataset nor a committed datatype",
                group_owner(&objects.parent_path(index)?),
                path.components()
                    .last()
                    .map(|name| name.as_str())
                    .unwrap_or("")
            ))),
        }
    }

    fn links(
        &self,
        index: usize,
    ) -> Result<impl ExactSizeIterator<Item = Result<SourceLink<'_>, Error>>, Error> {
        self.objects.links(index)
    }

    fn finish_planning(&self) {
        self.objects.release_headers();
    }

    fn resolve(&self, path: &ObjectPathBuf) -> Result<usize, Error> {
        self.objects.index(path)
    }
}

/// Anything a repacked attribute can be attached to: the root group, a subgroup,
/// or a dataset.
trait AttrSink {
    fn sink_set_attr(&mut self, name: &str, value: AttrValue);
    fn sink_set_attr_verbatim(
        &mut self,
        message: AttributeMessage,
        committed_datatype_target: Option<ObjectTarget>,
    );
    /// Attaches variable-length strings with the datatype and dataspace of `message`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForRepresentation`] if a string exceeds
    /// [`u32::MAX`] bytes.
    fn sink_set_attr_var_len_verbatim(
        &mut self,
        message: AttributeMessage,
        strings: Vec<String>,
        committed_datatype_target: Option<ObjectTarget>,
    ) -> Result<(), FormatError>;
}

impl AttrSink for FileBuilder {
    fn sink_set_attr(&mut self, name: &str, value: AttrValue) {
        self.set_attr(name, value);
    }
    fn sink_set_attr_verbatim(
        &mut self,
        message: AttributeMessage,
        committed_datatype_target: Option<ObjectTarget>,
    ) {
        self.set_attr_verbatim(message, committed_datatype_target);
    }
    fn sink_set_attr_var_len_verbatim(
        &mut self,
        message: AttributeMessage,
        strings: Vec<String>,
        committed_datatype_target: Option<ObjectTarget>,
    ) -> Result<(), FormatError> {
        self.set_attr_var_len_verbatim(message, strings, committed_datatype_target)?;
        Ok(())
    }
}

impl AttrSink for GroupBuilder {
    fn sink_set_attr(&mut self, name: &str, value: AttrValue) {
        self.set_attr(name, value);
    }
    fn sink_set_attr_verbatim(
        &mut self,
        message: AttributeMessage,
        committed_datatype_target: Option<ObjectTarget>,
    ) {
        self.set_attr_verbatim(message, committed_datatype_target);
    }
    fn sink_set_attr_var_len_verbatim(
        &mut self,
        message: AttributeMessage,
        strings: Vec<String>,
        committed_datatype_target: Option<ObjectTarget>,
    ) -> Result<(), FormatError> {
        self.set_attr_var_len_verbatim(message, strings, committed_datatype_target)?;
        Ok(())
    }
}

impl AttrSink for DatasetBuilder {
    fn sink_set_attr(&mut self, name: &str, value: AttrValue) {
        self.set_attr(name, value);
    }
    fn sink_set_attr_verbatim(
        &mut self,
        message: AttributeMessage,
        committed_datatype_target: Option<ObjectTarget>,
    ) {
        self.set_attr_verbatim(message, committed_datatype_target);
    }
    fn sink_set_attr_var_len_verbatim(
        &mut self,
        message: AttributeMessage,
        strings: Vec<String>,
        committed_datatype_target: Option<ObjectTarget>,
    ) -> Result<(), FormatError> {
        self.set_attr_var_len_verbatim(message, strings, committed_datatype_target)?;
        Ok(())
    }
}

/// Returns how an error message identifies the group at `path`.
fn group_owner(path: &ObjectPathBuf) -> String {
    if path.is_empty() {
        "root group".to_string()
    } else {
        format!("group {path}")
    }
}

/// Capture one dataset's full description and stage it on `db`, or fail with a
/// named [`Error::RepackUnsupported`] if any part cannot be reproduced.
fn emit_dataset(
    db: &mut DatasetBuilder,
    ds: &Dataset,
    path: &ObjectPathBuf,
    file: &Arc<File>,
    objects: &SourceObjects,
) -> Result<(), Error> {
    // A committed (`H5Tcommit`) element type lives in its own object header. The
    // dataset is reproduced naming the same type rather than inlining a copy of
    // it, which would read back correctly and still lose the name every C-library
    // reader reports.
    if let Some(address) = ds.committed_datatype_address()? {
        let type_target = committed_type_target(address, &format!("dataset {path}"), objects)?;
        db.with_committed_datatype_target(type_target);
    }

    let datatype = ds.datatype()?;
    let dataspace = ds.dataspace()?;
    let layout = ds.data_layout()?;
    let pipeline = ds.filter_pipeline_parsed()?;

    check_datatype(&datatype, &format!("dataset {path}"))?;
    check_layout(&layout, path)?;
    // An externally stored dataset carries a contiguous layout message with an
    // undefined data address, exactly as a never-written one does, and this crate
    // does not follow the external files. Refused rather than reproduced, which
    // would silently emit a schema-only dataset in place of one holding data.
    if ds.has_external_storage()? {
        return Err(Error::RepackUnsupported(format!(
            "dataset {path}: external data storage (H5Pset_external) cannot be repacked -- its \
             element bytes live in files this crate does not read"
        )));
    }

    let dims = dataspace.dimensions.clone();
    let n_elements: u64 = dims.iter().product();

    // The chunks the source holds, walked once. Both the unallocated test below
    // and the verbatim planner further down need the list, and each walk is the
    // whole chunk index. Empty for a layout that has no chunks at all, which is
    // the answer both of them want for one.
    let source_chunks = match &layout {
        DataLayout::Chunked { .. } => ds.raw_chunks()?,
        _ => Vec::new(),
    };

    // Every variable-length reference the datatype reaches, whether it *is*
    // variable-length or merely contains one through a compound member or array
    // entry. Both kinds are re-staged rather than copied, so both disqualify the
    // verbatim and fill-value paths below.
    let vlen_slots = vl_data::embedded_vlen_slots(&datatype).ok_or_else(|| {
        Error::RepackUnsupported(format!(
            "dataset {path}: datatype declares variable-length members its own element \
             size cannot hold"
        ))
    })?;
    // Likewise every object-header address it reaches. Both kinds of address are
    // rewritten rather than copied, so both disqualify the paths that move
    // element bytes unchanged.
    let reference_slots = datatype::embedded_reference_slots(&datatype).ok_or_else(|| {
        Error::RepackUnsupported(format!(
            "dataset {path}: datatype declares object references this rewrite cannot locate in \
             its element bytes -- a width other than eight, an enumeration over one, a \
             variable-length sequence of them, or offsets its own element size cannot hold"
        ))
    })?;

    // A user-defined fill value is fixed element bytes in the datatype, so it
    // reproduces exactly on the fixed-size paths below (dense-chunked verbatim,
    // and contiguous/compact/sparse re-encode). The variable-length and
    // object-reference paths that follow rebuild global-heap references and
    // object addresses from scratch, so a fill value there could carry stale heap
    // or address bytes; refuse rather than copy it, matching the crate's
    // never-degrade-on-rewrite contract.
    let fill = ds.defined_fill_bytes()?;
    let fill_time = ds.fill_time()?;
    if fill.is_some() && !(vlen_slots.is_empty() && reference_slots.is_empty()) {
        return Err(Error::RepackUnsupported(format!(
            "dataset {path}: a fill value on a variable-length or object-reference \
             dataset cannot be repacked faithfully"
        )));
    }
    db.with_fill_time(fill_time);

    // Preserve unallocated storage without materializing fill values. Copy the
    // filter message verbatim because there are no stored values to re-encode.
    if storage_is_unallocated(&layout, &source_chunks) {
        if pipeline.is_some() && !matches!(&layout, DataLayout::Chunked { .. }) {
            return Err(Error::RepackUnsupported(format!(
                "dataset {path}: filter pipeline requires chunked storage"
            )));
        }
        db.fill = fill;
        db.with_unallocated_storage(datatype, &dims);
        db.chunk_options.unallocated_pipeline = ds.filter_pipeline_message_bytes()?;
        carry_shape_and_pipeline(
            db,
            &dims,
            dataspace.extensible_max_dimensions(),
            &layout,
            &[],
        );
        copy_dataset_attrs(db, ds, path, objects)?;
        return Ok(());
    }

    // Variable-length string datasets take a dedicated path: their element
    // references point into the global heap, so they are re-emitted by reading
    // each element's exact heap bytes and re-staging them, not by copying raw
    // element bytes (whose stored heap addresses would go stale on rewrite).
    if vl_data::is_vlen_string_datatype(&datatype) {
        emit_vlen_string_dataset(db, ds, path, &datatype, &dims, &layout, &pipeline)?;
        // VL-string datasets carry attributes the same way as any other.
        copy_dataset_attrs(db, ds, path, objects)?;
        return Ok(());
    }

    // Non-string variable-length (sequence) datasets take the same global-heap
    // re-staging path as VL strings: each element's exact heap bytes are read and
    // re-emitted through a fresh global heap, so the stored heap addresses are
    // rebuilt rather than copied stale. Routed here before the verbatim chunk-copy
    // path so a chunked one is refused (not copied with stale references).
    if is_nonstring_vlen(&datatype) {
        emit_vlen_sequence_dataset(db, ds, path, &datatype, &dims, &layout, &pipeline)?;
        copy_dataset_attrs(db, ds, path, objects)?;
        return Ok(());
    }

    // Object-reference datasets store absolute object-header addresses that would
    // go stale on rewrite, so each reference is resolved to its target's *new*
    // address (via source object discovery and the writer's path resolution)
    // rather than copied. Routed here before the verbatim chunk-copy path so a
    // chunked one is refused (not copied with stale addresses).
    if is_object_reference(&datatype) {
        emit_object_reference_dataset(db, ds, path, &dims, &layout, file, objects)?;
        copy_dataset_attrs(db, ds, path, objects)?;
        return Ok(());
    }

    // A datatype that merely *contains* a variable-length member or an object
    // reference — a compound with either, or an array of them — embeds the
    // address inside each element. Those addresses go stale on rewrite exactly as
    // a top-level one does, so this dataset is re-staged rather than copied.
    // Routed before the verbatim chunk-copy path so a chunked one is rebuilt, not
    // copied with source addresses (issue #201).
    //
    // Both kinds are handled together rather than in sequence: a compound can
    // carry one of each, and rewriting only the first kind found would leave the
    // other pointing into the source file — reintroducing the very bug this path
    // exists to fix, and skipping the other kind's refusals with it.
    if !vlen_slots.is_empty() || !reference_slots.is_empty() {
        emit_embedded_address_dataset(
            db,
            ds,
            path,
            &datatype,
            &dims,
            &layout,
            &pipeline,
            &vlen_slots,
            &reference_slots,
            file,
            objects,
        )?;
        copy_dataset_attrs(db, ds, path, objects)?;
        return Ok(());
    }

    // Past the reference-bearing paths: carry the fill value through the
    // fixed-size storage paths below, both of which reproduce it exactly.
    db.fill = fill;

    // A chunked dataset with allocated chunks is copied chunk-by-chunk, verbatim:
    // each compressed chunk is laid into the output without decoding, so any
    // filter — including lossy ones (float scale-offset, ZFP) and ones this crate
    // cannot itself apply (SZIP, unknown) — is reproduced byte-exact. This avoids
    // the decompress→recompress round-trip and the whole-dataset decompression
    // blowup of the read-raw path. `check_pipeline` is intentionally skipped here:
    // never decoding makes every filter safe to carry. The datatype check above
    // still refuses time/variable-length/reference types, whose reproduction or
    // embedded addresses are unsafe even when copied verbatim.
    if let DataLayout::Chunked {
        chunk_dimensions,
        flags,
        ..
    } = &layout
        && n_elements > 0
    {
        let rank = dims.len();
        let chunk_dims: Vec<u64> = chunk_dimensions.iter().take(rank).copied().collect();

        if let Some(DenseChunkPlan { meta, grid_order }) =
            try_plan_dense_chunks(source_chunks, &dims, &chunk_dims)
        {
            let maxshape = dataspace.extensible_max_dimensions();
            let elem_size = crate::datatype::element_size_usize(&datatype)?;
            // Stream the chunks from the source at write time rather than reading
            // them all now: the provider holds an `Arc<File>` and fetches one
            // chunk at a time, so a huge dataset never sits in memory.
            let provider = DatasetChunkProvider {
                file: Arc::clone(file),
                grid_order,
            };
            db.with_raw_chunks_lazy(
                datatype,
                &dims,
                maxshape,
                &chunk_dims,
                elem_size,
                *flags,
                ds.filter_pipeline_message_bytes()?,
                meta,
                Box::new(provider),
            );

            // Carry the dataset's attributes, refusing any that cannot be
            // represented.
            copy_dataset_attrs(db, ds, path, objects)?;
            return Ok(());
        }

        // Sparse (holes) chunked dataset: the verbatim path needs a dense grid,
        // so fall through to the read-raw + re-encode path below. That path
        // re-encodes, so it is only faithful for lossless filters; a lossy
        // pipeline on a sparse dataset is refused by `check_pipeline`.
    }

    // Contiguous/compact, or a sparse chunked dataset: read the decompressed
    // bytes and re-encode. This path can only reproduce lossless filters, so
    // refuse a lossy pipeline before reading.
    let filters = check_pipeline(pipeline.as_ref(), path)?;

    if n_elements == 0 {
        // An empty dataset owns no element bytes: carry just the datatype and
        // shape so the reconstructed dataset has the same signature.
        db.with_dtype(datatype).with_shape(&dims);
    } else {
        let raw = ds.read_raw()?;
        db.with_raw_data(datatype, raw, n_elements)
            .with_shape(&dims);
    }

    carry_shape_and_pipeline(
        db,
        &dims,
        dataspace.extensible_max_dimensions(),
        &layout,
        &filters,
    );

    // Carry the dataset's attributes, refusing if any cannot be represented.
    copy_dataset_attrs(db, ds, path, objects)?;

    Ok(())
}

/// Carry `ds`'s attributes onto `db`, refusing any that cannot be represented.
///
/// Every exit from [`emit_dataset`] ends here with the same arguments, and each
/// one has to: a dataset that reaches the output without its attributes is a
/// silent loss, not a refusal.
fn copy_dataset_attrs(
    db: &mut DatasetBuilder,
    ds: &Dataset,
    path: &ObjectPathBuf,
    objects: &SourceObjects,
) -> Result<(), Error> {
    copy_attrs(
        db,
        ds.attr_messages()?,
        || ds.attrs(),
        &format!("dataset {path}"),
        objects,
    )
}

/// Sets the maximum dimensions, the chunk dimensions, the [`PartialEdgeFiltering`] policy, and
/// the filter pipeline of the source dataset on `db`.
///
/// `filters` are the ones [`check_pipeline`] decoded, which is what makes them
/// safe to re-apply: there is no way to reach this with a pipeline that was not
/// proved reproducible first.
fn carry_shape_and_pipeline(
    db: &mut DatasetBuilder,
    dims: &[u64],
    max_dimensions: Option<&[MaxExtent]>,
    layout: &DataLayout,
    filters: &[FilterSpec],
) {
    // A max-shape that differs from the current shape means a resizable dataset.
    if let Some(maxshape) = max_dimensions {
        db.with_maxshape(maxshape);
    }

    // Chunking: the v3 layout appends the element size as a trailing chunk
    // dimension, so keep only the first `rank` entries; v4 already stores `rank`.
    if let DataLayout::Chunked {
        chunk_dimensions,
        flags,
        ..
    } = layout
    {
        let rank = dims.len();
        let logical: Vec<u64> = chunk_dimensions.iter().take(rank).copied().collect();
        db.with_chunks(&logical);
        db.chunk_options.partial_edge_filtering =
            Some(PartialEdgeFiltering::from_layout_flags(*flags));
    }

    // Re-apply the source's filters in the source's own order, each carrying
    // the optional flag it was stored with (issue #333).
    //
    // Both are observable, and neither is necessarily what this crate's own
    // writer would choose. `fletcher32` between shuffle and deflate checksums
    // the shuffled bytes; moved to the end it checksums the deflated ones — a
    // different checksum over different data, and a different thing for a
    // reader to verify. A filter that loses its optional flag becomes one a
    // reader must be able to apply, where before it could skip it.
    //
    // Pushed rather than set through the `with_*` switches, which name a filter
    // to apply and place it at this crate's canonical rank: that is right for a
    // caller composing a pipeline and wrong for reproducing one that exists.
    for &spec in filters {
        db.chunk_options.push_filter(spec);
    }
}

/// Re-emit a variable-length string dataset faithfully: read each element's
/// exact heap bytes (preserving null-vs-empty, charset, padding, and the source
/// VL datatype shape) and re-stage them through the writer's VL-string path,
/// then reapply the source's chunk geometry, filters, and resizability.
///
/// The rebuilt references carry the *new* file's heap addresses. For a chunked
/// or filtered layout the writer places those collections ahead of the chunk
/// data so the addresses are known before encoding (issue #109), which is what
/// makes this path reproduce such a dataset rather than refuse it.
fn emit_vlen_string_dataset(
    db: &mut DatasetBuilder,
    ds: &Dataset,
    path: &ObjectPathBuf,
    datatype: &Datatype,
    dims: &[u64],
    layout: &DataLayout,
    pipeline: &Option<FilterPipeline>,
) -> Result<(), Error> {
    // A lossy pipeline cannot be reproduced, so refuse it before reading — the
    // same guard the fixed-size re-encode path applies.
    let filters = check_pipeline(pipeline.as_ref(), path)?;

    // Read each element's exact heap bytes, preserving the null-vs-empty
    // distinction. Reading bytes (not the lossily UTF-8-decoded `String`) keeps
    // embedded NULs and non-UTF-8 payloads byte-exact.
    let objects = ds.read_vlen_string_bytes(VlenStringReadOptions::default())?;
    let elements: Vec<VlStringElement> = objects
        .into_iter()
        .map(|o| match o {
            VlByteObject::Null => VlStringElement::Null,
            VlByteObject::Bytes(bytes) => VlStringElement::Bytes(bytes),
        })
        .collect();

    // Re-stage with the exact source datatype, then set the shape. ND datasets
    // round-trip because the element references are stored row-major, matching
    // the order `read_vlen_string_bytes` returns.
    db.with_vlen_string_elements(datatype.clone(), &elements)
        .map_err(Error::Format)?;
    db.with_shape(dims);
    carry_shape_and_pipeline(
        db,
        dims,
        ds.dataspace()?.extensible_max_dimensions(),
        layout,
        &filters,
    );
    Ok(())
}

/// Re-emit a non-string variable-length (sequence) dataset faithfully: read each
/// element's exact heap bytes and re-stage them through a fresh global heap, so
/// the rewritten file's heap addresses are rebuilt rather than copied stale.
///
/// A chunked, filtered, or resizable layout is reproduced rather than refused:
/// sequences stage through the same global-heap path as VL strings, so they
/// inherit the early collection placement that makes the heap addresses known
/// before the chunks are encoded (issue #109).
fn emit_vlen_sequence_dataset(
    db: &mut DatasetBuilder,
    ds: &Dataset,
    path: &ObjectPathBuf,
    datatype: &Datatype,
    dims: &[u64],
    layout: &DataLayout,
    pipeline: &Option<FilterPipeline>,
) -> Result<(), Error> {
    let filters = check_pipeline(pipeline.as_ref(), path)?;

    // Read each element's exact heap bytes (preserving the null-vs-empty
    // distinction and any embedded NULs), then re-stage with the source datatype.
    let (objects, _element_size) = ds.read_vlen_sequence_bytes(VlenStringReadOptions::default())?;
    let elements: Vec<VlStringElement> = objects
        .into_iter()
        .map(|o| match o {
            VlByteObject::Null => VlStringElement::Null,
            VlByteObject::Bytes(bytes) => VlStringElement::Bytes(bytes),
        })
        .collect();

    db.with_vlen_sequence_elements(datatype.clone(), &elements)
        .map_err(Error::Format)?;
    db.with_shape(dims);
    carry_shape_and_pipeline(
        db,
        dims,
        ds.dataspace()?.extensible_max_dimensions(),
        layout,
        &filters,
    );
    Ok(())
}

/// Re-emit a dataset whose datatype *contains* an address without being one — a
/// compound with a variable-length or object-reference member, an array of such
/// compounds, or nesting of either (issue #201).
///
/// The element bytes are read as they stand; each embedded variable-length
/// reference is resolved against the source's global heap and its payload
/// re-staged into the destination's own, and each embedded object address is
/// resolved to its source object index so the writer can fill in the target's destination
/// address. Copying the element bytes verbatim instead would leave every
/// address pointing into the source file, which the destination never
/// reproduces — a file that reads back plausibly here and not at all in the
/// reference C library.
///
/// Both kinds are handled in one pass because a single compound can carry one of
/// each, and they patch disjoint byte ranges of the same buffer. Everything
/// outside those ranges is carried byte-for-byte, so the fixed-size members keep
/// their exact stored bytes and byte order.
#[allow(clippy::too_many_arguments)]
fn emit_embedded_address_dataset(
    db: &mut DatasetBuilder,
    ds: &Dataset,
    path: &ObjectPathBuf,
    datatype: &Datatype,
    dims: &[u64],
    layout: &DataLayout,
    pipeline: &Option<FilterPipeline>,
    vlen_slots: &[EmbeddedVlSlot],
    reference_slots: &[usize],
    file: &Arc<File>,
    objects: &SourceObjects,
) -> Result<(), Error> {
    // This path re-encodes, so a lossy filter cannot be reproduced — the same
    // guard the other re-staging paths apply.
    let filters = check_pipeline(pipeline.as_ref(), path)?;

    // An embedded object address is resolved as the elements are re-staged, which
    // a compressed chunk would need rewritten in place, so the layout guards that
    // protect a top-level object-reference dataset apply here too. They are
    // checked before anything is read, and before the variable-length work, so a
    // compound carrying both kinds is refused rather than half-rewritten.
    if !reference_slots.is_empty() {
        check_embedded_reference_layout(ds, path, layout, file)?;
    }

    let n_elements: u64 = dims.iter().product();

    // Read the element bytes once, resolving the variable-length payloads in the
    // same pass when there are any.
    let (raw, vl_offsets, vl_elements) = if vlen_slots.is_empty() {
        let raw = if n_elements == 0 {
            Vec::new()
        } else {
            ds.read_raw()?
        };
        (raw, Vec::new(), Vec::new())
    } else {
        let data = ds.read_embedded_vlen_bytes(vlen_slots, VlenStringReadOptions::default())?;
        let elements: Vec<VlStringElement> = data
            .objects
            .into_iter()
            .map(|o| match o {
                VlByteObject::Null => VlStringElement::Null,
                VlByteObject::Bytes(bytes) => VlStringElement::Bytes(bytes),
            })
            .collect();
        (data.raw, data.offsets, elements)
    };

    // Resolve the object addresses from the bytes as read, before the
    // variable-length staging below consumes `raw`. The two kinds occupy disjoint
    // slots, so reading one is unaffected by rewriting the other.
    let reference_patches =
        resolve_embedded_references(&raw, datatype, dims, path, objects, reference_slots)?;

    if vlen_slots.is_empty() {
        db.with_embedded_object_references(datatype.clone(), raw, n_elements, reference_patches);
    } else {
        db.with_embedded_vlen_elements(
            datatype.clone(),
            raw,
            n_elements,
            &vl_offsets,
            &vl_elements,
        );
        if !reference_patches.is_empty() {
            db.reference_targets = Some(reference_patches);
        }
    }
    db.with_shape(dims);
    carry_shape_and_pipeline(
        db,
        dims,
        ds.dataspace()?.extensible_max_dimensions(),
        layout,
        &filters,
    );
    Ok(())
}

/// Turn every embedded object address in `raw` into the target the writer
/// resolves at serialization time.
fn resolve_embedded_references(
    raw: &[u8],
    datatype: &Datatype,
    dims: &[u64],
    path: &ObjectPathBuf,
    objects: &SourceObjects,
    slots: &[usize],
) -> Result<Vec<ObjectRefPatch>, Error> {
    if slots.is_empty() {
        return Ok(Vec::new());
    }
    let stride = datatype.type_size() as usize;
    let n_elements: usize = dims.iter().product::<u64>().to_usize()?;
    let needed = n_elements
        .checked_mul(stride)
        .ok_or(FormatError::OffsetOverflow {
            offset: n_elements as u64,
            length: stride as u64,
        })?;
    if raw.len() < needed {
        return Err(FormatError::UnexpectedEof {
            expected: needed,
            available: raw.len(),
        }
        .into());
    }

    let mut patches = Vec::with_capacity(n_elements * slots.len());
    for e in 0..n_elements {
        for &slot in slots {
            let at = e * stride + slot;
            let v = StoredAddress::new(u64::from_le_bytes(
                raw[at..at + 8]
                    .try_into()
                    .expect("slot offsets leave 8 bytes inside the element"),
            ));
            patches.push(ObjectRefPatch {
                byte_offset: at,
                target: resolve_reference_address(v, path, objects)?,
            });
        }
    }
    Ok(patches)
}

/// A [`ChunkProvider`] that streams a dense chunked dataset's chunks from the
/// source file one at a time during the write, so repack never holds more than a
/// single chunk's bytes. Holds an `Arc<File>` (so it owns its source with no
/// borrowed lifetime) and the source [`ChunkInfo`] for each grid slot.
struct DatasetChunkProvider {
    file: Arc<File>,
    /// Source chunk descriptors in dense row-major grid order, one per slot.
    grid_order: Vec<ChunkInfo>,
}

impl ChunkProvider for DatasetChunkProvider {
    fn chunk_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), FormatError> {
        // Read exactly the chunk's compressed bytes at its recorded address, with
        // no decode, the same slice the chunked reader consumes. The address is the
        // one the chunk index stores, and this source spans the file from byte
        // zero, so this adds the base address first. `read_at` fills the whole
        // buffer or errors, and the emitter additionally checks the length against
        // the planned size, so the layout cannot silently desync from the data.
        // Reading straight into the emitter's reused buffer keeps repack at one
        // chunk-sized allocation for the whole dataset.
        let info = &self.grid_order[index];
        let source = self.file.source();
        let len = info.chunk_size.to_usize()?;
        let at = self.file.addresses().resolve(info.address)?.get();
        // Bounds-check before growing the buffer, the way `Source::read_exact_at`
        // does and for its reason: `chunk_size` comes from the source's chunk
        // index, so a malformed file could name a 4 GiB chunk and have this zero
        // that much memory only for the read to fail EOF anyway.
        let end = at
            .checked_add(len as u64)
            .ok_or(FormatError::OffsetOverflow {
                offset: at,
                length: len as u64,
            })?;
        if end > source.len() {
            return Err(FormatError::UnexpectedSourceEof {
                expected: end,
                available: source.len(),
            });
        }
        let start = out.len();
        out.resize(start + len, 0);
        source.read_at(at, &mut out[start..])
    }
}

/// A planned dense chunked dataset: per-chunk sizes/masks (enough to lay out the
/// destination) plus the source chunk descriptors, both in dense grid order.
struct DenseChunkPlan {
    meta: Vec<ChunkMeta>,
    grid_order: Vec<ChunkInfo>,
}

/// Plan a chunked dataset's verbatim copy without reading any chunk bytes: if
/// every chunk-grid slot is present exactly once (a dense grid), return the
/// per-chunk [`ChunkMeta`] (sizes + filter masks) and the source [`ChunkInfo`]
/// for each slot, both in dense row-major grid order. Returns `None` when the
/// grid has holes (a sparse dataset), so the caller falls back to read-raw.
///
/// `chunks` is the source's chunk list, which the caller has walked already;
/// `dims` is the dataspace shape and `chunk_dims` the logical (rank-only) chunk
/// dimensions. The grid has `num_chunks_per_dim[d] = ceil(dims[d]/chunk_dims[d])`
/// slots per dimension; a chunk at N-d offset `o` maps to grid coordinate
/// `o[d]/chunk_dims[d]` and linear (row-major) index over the grid.
fn try_plan_dense_chunks(
    chunks: Vec<ChunkInfo>,
    dims: &[u64],
    chunk_dims: &[u64],
) -> Option<DenseChunkPlan> {
    // Map the source chunks onto the dense grid via the shared planner (the
    // single owner of grid-mapping logic, also used by the in-place editor); a
    // sparse grid (holes/duplicates/misalignment) returns `None`.
    let grid = crate::chunked_read::plan_dense_grid(chunks, dims, chunk_dims)?;
    let grid_order = grid.grid_order;
    let meta = grid_order
        .iter()
        .map(|info| ChunkMeta {
            compressed_size: info.chunk_size.get(),
            filter_mask: info.filter_mask,
        })
        .collect();
    Some(DenseChunkPlan { meta, grid_order })
}

/// Whether an attribute's element bytes mean the same thing in another file.
///
/// An attribute message is otherwise self-contained — name, datatype, dataspace
/// and data all travel together — so the only thing that stops a byte-for-byte
/// copy is an element that stores a *location* rather than a value: a
/// variable-length datum, whose bytes are a global-heap collection address plus
/// an index, and a reference, whose bytes are an object-header address. Both
/// point into the source file and would dangle in the destination.
///
/// Nested occurrences count: a compound with a variable-length member, or an
/// array of references, embeds the same address inside each element.
///
/// Every class is named rather than defaulted through a `_` arm, so adding one to
/// [`Datatype`] stops the build here and forces the question to be answered. A
/// wrong answer in the permissive direction writes a dangling address into the
/// copy and returns `Ok`, which is precisely the failure this function exists to
/// prevent, so the compiler is the right place to catch a new class rather than
/// a default that silently picks a side.
fn attr_bytes_are_position_independent(dt: &Datatype) -> bool {
    match dt {
        Datatype::FixedPoint(..)
        | Datatype::FloatingPoint(..)
        | Datatype::Time(..)
        | Datatype::String { .. }
        | Datatype::BitField(..)
        | Datatype::Opaque { .. }
        | Datatype::Complex(_) => true,
        Datatype::Compound { members, .. } => members
            .iter()
            .all(|m| attr_bytes_are_position_independent(&m.datatype)),
        Datatype::Enumeration { base_type, .. } | Datatype::Array { base_type, .. } => {
            attr_bytes_are_position_independent(base_type)
        }
        Datatype::VariableLength(_) | Datatype::Reference { .. } | _ => false,
    }
}

/// Copy every attribute of `owner` onto `sink`, in name order for a
/// deterministic output.
///
/// An attribute whose bytes are position-independent is copied *verbatim*: the
/// source's own datatype, dataspace and element bytes go straight into the
/// destination message. That is what keeps a rewrite faithful, because
/// [`AttrValue`] is a decoded view and cannot express a big-endian integer, a
/// sub-width precision, a variable-length string, a rank above one, or a
/// string's padding — so anything routed through it comes out re-encoded in this
/// crate's own layout, flattened to rank 1 (issue #241).
///
/// The rest — variable-length data and references — hold source-file addresses,
/// so their bytes cannot travel. Those fall back to `decode`, the decoded
/// [`AttrValue`] map, which the writer re-encodes against a heap of the
/// destination's own. An attribute that has no representation there either is
/// refused rather than dropped.
///
/// A variable-length *string* takes a third path, because the two above each get
/// half of it wrong: its element bytes address the source heap and cannot be
/// copied, but its datatype and dataspace say "variable-length UTF-8" and
/// "scalar", neither of which [`AttrValue`] can express — so a plain re-encode
/// turns `str` into `bytes` for every consumer. Keeping the source's datatype and
/// dataspace while restaging only the strings gets both halves. This is the
/// common case in practice, not an exotic one: it is what `h5py` writes for
/// every plain-string attribute.
///
/// `decode` is a closure so the decoding pass is paid for only when some
/// attribute actually needs it, which for most objects is never.
fn copy_attrs<S, F>(
    sink: &mut S,
    mut messages: Vec<AttributeMessage>,
    decode: F,
    owner: &str,
    objects: &SourceObjects,
) -> Result<(), Error>
where
    S: AttrSink + ?Sized,
    F: FnOnce() -> Result<std::collections::HashMap<String, AttrValue>, Error>,
{
    // Stable, because equal keys here are not interchangeable: a malformed source
    // header carrying one name twice carries two different payloads under it, and
    // an unstable sort would pick between them arbitrarily. Header order decides.
    messages.sort_by(|a, b| a.name.cmp(&b.name));
    // An attribute naming a committed (`H5Tcommit`) datatype is re-pointed at the
    // same type in the output, exactly as a dataset's committed element type is.
    // The source address is what says *which* committed object, so two attributes
    // sharing one type still share one in the output rather than each getting a
    // copy of the encoding.
    let messages = messages
        .into_iter()
        .map(|message| {
            let committed_datatype_target = match message.datatype_location {
                DatatypeLocation::Committed(address) => {
                    let name = &message.name;
                    Some(committed_type_target(
                        address,
                        &format!("{owner} attribute {name:?}"),
                        objects,
                    )?)
                }
                DatatypeLocation::Inline => None,
            };
            Ok((message, committed_datatype_target))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let any_needs_decoding = messages
        .iter()
        .any(|(m, _)| !attr_bytes_are_position_independent(&m.datatype));
    let decoded = if any_needs_decoding {
        decode()?
    } else {
        std::collections::HashMap::new()
    };

    for (message, committed_datatype_target) in messages {
        if attr_bytes_are_position_independent(&message.datatype) {
            sink.sink_set_attr_verbatim(message, committed_datatype_target);
        } else if let Some(value) = decoded.get(&message.name) {
            // A variable-length datatype the decode resolved into strings is
            // exactly the case that keeps its datatype and dataspace but restages
            // its payload. Gating on what the decode *produced* rather than
            // re-classifying the datatype here keeps the two from drifting apart.
            match (&message.datatype, value.as_strings()) {
                (Datatype::VariableLength(_), Some(strings)) => {
                    let strings = strings.to_vec();
                    sink.sink_set_attr_var_len_verbatim(
                        message,
                        strings,
                        committed_datatype_target,
                    )?;
                }
                _ => sink.sink_set_attr(&message.name, value.clone()),
            }
        } else {
            let name = &message.name;
            return Err(Error::RepackUnsupported(format!(
                "{owner}: attribute {name:?} has a datatype that cannot be repacked faithfully yet"
            )));
        }
    }
    Ok(())
}

/// Reject datatypes whose on-disk form this crate cannot re-emit faithfully,
/// recursing into compound members, enumeration bases, and array element types so
/// a nested occurrence is caught too. Region and non-8-byte object references are
/// the remaining refusals (their stored selections/addresses are not yet
/// rewritten); 8-byte object references are handled by the reference rewrite path.
fn check_datatype(dt: &Datatype, owner: &str) -> Result<(), Error> {
    let bad = |what: &str| {
        Err(Error::RepackUnsupported(format!(
            "{owner}: {what} datatype cannot be repacked faithfully yet"
        )))
    };
    match dt {
        // Fixed-size datatypes whose metadata the writer serializes faithfully. Their raw
        // element bytes can be copied unchanged, preserving numeric byte order and layout.
        Datatype::FixedPoint(..)
        | Datatype::FloatingPoint(..)
        | Datatype::Time(..)
        | Datatype::String { .. }
        | Datatype::BitField(..)
        | Datatype::Opaque { .. }
        | Datatype::Complex(_) => Ok(()),
        // String-shaped variable-length datatypes (`VariableLengthType::String`, or the
        // MATLAB VLEN-of-1-byte-ASCII-string shape) are reproduced by reading
        // each element's exact heap bytes and re-staging them through the
        // writer's VL-string path; the layout/filter checks gate chunked ones.
        Datatype::VariableLength(_) if vl_data::is_vlen_string_datatype(dt) => Ok(()),
        // Non-string VL (sequences of arbitrary base types) are re-staged the
        // same way, but only when the base type's bytes carry no embedded heap or
        // file addresses that a verbatim copy would leave stale.
        Datatype::VariableLength(kind) => check_vlen_base_type(kind.base_type(), owner),
        // Object references (8-byte object-header addresses) are repacked by
        // rewriting each address to its target's new location. Region references
        // (which embed a dataspace selection in the global heap) and non-8-byte
        // object references are not reproduced yet.
        Datatype::Reference {
            ref_type: ReferenceType::Object,
            size: 8,
        } => Ok(()),
        Datatype::Reference {
            ref_type: ReferenceType::Object,
            ..
        } => bad("non-8-byte object reference"),
        Datatype::Reference {
            ref_type: ReferenceType::DatasetRegion,
            ..
        } => bad("dataset-region reference"),
        Datatype::Reference { .. } => bad("unknown reference"),
        Datatype::Compound { members, .. } => {
            for m in members {
                check_datatype(&m.datatype, owner)?;
            }
            Ok(())
        }
        Datatype::Enumeration { base_type, .. } => check_datatype(base_type, owner),
        Datatype::Array { base_type, .. } => check_datatype(base_type, owner),
        _ => bad("unknown class"),
    }
}

/// Whether `dt` is a non-string variable-length (sequence) datatype — the kind
/// re-emitted by [`emit_vlen_sequence_dataset`]. Excludes the string-shaped VL
/// datatypes, which [`emit_vlen_string_dataset`] handles.
fn is_nonstring_vlen(dt: &Datatype) -> bool {
    matches!(dt, Datatype::VariableLength(_)) && !vl_data::is_vlen_string_datatype(dt)
}

/// A non-string VL sequence is repacked by re-staging each element's exact heap
/// bytes verbatim. That is faithful only when the base type's bytes embed no
/// addresses that would go stale on rewrite: a nested variable-length type (its
/// elements are themselves global-heap references) and a reference (a stale file
/// address) are refused, recursing through compound members, array elements, and
/// enumeration bases so a nested occurrence is caught too.
fn check_vlen_base_type(dt: &Datatype, owner: &str) -> Result<(), Error> {
    let bad = |what: &str| {
        Err(Error::RepackUnsupported(format!(
            "{owner}: variable-length sequence of {what} cannot be repacked faithfully yet"
        )))
    };
    match dt {
        Datatype::FixedPoint(..)
        | Datatype::FloatingPoint(..)
        | Datatype::Time(..)
        | Datatype::String { .. }
        | Datatype::BitField(..)
        | Datatype::Opaque { .. }
        | Datatype::Complex(_) => Ok(()),
        Datatype::Reference { .. } => bad("references"),
        Datatype::VariableLength(_) => bad("variable-length elements"),
        Datatype::Compound { members, .. } => {
            for m in members {
                check_vlen_base_type(&m.datatype, owner)?;
            }
            Ok(())
        }
        Datatype::Enumeration { base_type, .. } => check_vlen_base_type(base_type, owner),
        Datatype::Array { base_type, .. } => check_vlen_base_type(base_type, owner),
        _ => bad("unknown class"),
    }
}

/// Whether `dt` is an object-reference datatype handled by
/// [`emit_object_reference_dataset`].
fn is_object_reference(dt: &Datatype) -> bool {
    matches!(
        dt,
        Datatype::Reference {
            ref_type: ReferenceType::Object,
            ..
        }
    )
}

/// Returns the link name `name` spells, in the group at `parent`.
///
/// # Errors
///
/// Returns [`Error::RepackUnsupported`] if `name` is not a component of an object path: the empty
/// name, `.`, or a name holding `/`. The error reports the group and `name`. A source can hold
/// such a link, and the object under it has no path in the output.
fn link_name_in(parent: &ObjectPathBuf, name: &str) -> Result<LinkNameBuf, Error> {
    LinkNameBuf::new(name).ok_or_else(|| {
        Error::RepackUnsupported(format!(
            "{}: the link {name:?} has a name that is not a component of an object path, \
             so the object it identifies has no path in the output",
            group_owner(parent)
        ))
    })
}

/// Returns the indexed destination target of the committed datatype at source `address`.
///
/// # Errors
///
/// Returns [`Error::RepackUnsupported`] if the datatype is dropped or lies outside the retained
/// hard-link hierarchy. Propagates source address resolution errors.
fn committed_type_target(
    address: StoredAddress,
    user: &str,
    objects: &SourceObjects,
) -> Result<ObjectTarget, Error> {
    match objects.placement(address)? {
        Some(Placement::Kept(index)) => Ok(ObjectTarget::SourceIndex(index)),
        Some(Placement::Dropped(path)) => Err(Error::RepackUnsupported(format!(
            "{user}: names the committed datatype {}, which this repack drops",
            dropped_object(path.as_ref(), address)
        ))),
        None => Err(Error::RepackUnsupported(format!(
            "{user}: names a committed datatype that is not reachable by a hard link in the \
             source, so it has no place in the output"
        ))),
    }
}

/// Returns how an error message identifies a dropped object: by its path under a drop path, or
/// by `address` where the object has none.
fn dropped_object(path: Option<&ObjectPathBuf>, address: StoredAddress) -> String {
    match path {
        Some(path) => format!("{:?}", path.to_string()),
        None => format!("at address {:#x}", address.get()),
    }
}

/// Re-emit an object-reference dataset faithfully: rewrite each stored address to
/// point at its target's destination location instead of its stale source one.
///
/// Each reference is read, resolved through `objects` to its path in the output, and
/// re-staged as a path target the writer resolves once destination addresses are
/// known. Null (address 0) and undefined (`HADDR_UNDEF`) references are carried
/// verbatim. Refused by name: chunked/filtered or resizable layouts, a non-zero
/// base address, a reference to a dropped object, and a reference whose target is
/// not a hard-linked group or dataset in the source (dangling, or a named
/// datatype / region target not modelled yet).
#[allow(clippy::too_many_arguments)]
fn emit_object_reference_dataset(
    db: &mut DatasetBuilder,
    ds: &Dataset,
    path: &ObjectPathBuf,
    dims: &[u64],
    layout: &DataLayout,
    file: &Arc<File>,
    objects: &SourceObjects,
) -> Result<(), Error> {
    if matches!(layout, DataLayout::Chunked { .. }) {
        return Err(Error::RepackUnsupported(format!(
            "dataset {path}: chunked or filtered object-reference datasets cannot be repacked \
             (their addresses live inside compressed chunks and would need rewriting in place)"
        )));
    }
    if ds.dataspace()?.extensible_max_dimensions().is_some() {
        return Err(Error::RepackUnsupported(format!(
            "dataset {path}: resizable object-reference datasets cannot be repacked"
        )));
    }
    // Object references store addresses relative to the base address; the rewrite
    // path assumes a zero base (the universal case), so a userblock file is
    // refused rather than risk a mis-resolved address.
    if !file.base_address().is_zero() {
        return Err(Error::RepackUnsupported(format!(
            "dataset {path}: object references in a file with a non-zero base address (userblock) \
             cannot be repacked yet"
        )));
    }

    let n_elements: usize = dims.iter().product::<u64>().to_usize()?;
    let targets = if n_elements == 0 {
        Vec::new()
    } else {
        let raw = ds.read_raw()?;
        let needed = n_elements
            .checked_mul(8)
            .ok_or(FormatError::OffsetOverflow {
                offset: n_elements as u64,
                length: 8,
            })?;
        if raw.len() < needed {
            return Err(FormatError::UnexpectedEof {
                expected: needed,
                available: raw.len(),
            }
            .into());
        }
        let mut targets = Vec::with_capacity(n_elements);
        for chunk in raw[..needed].as_chunks::<8>().0 {
            let v = StoredAddress::new(u64::from_le_bytes(*chunk));
            targets.push(resolve_reference_address(v, path, objects)?);
        }
        targets
    };

    db.with_object_references(targets);
    db.with_shape(dims);
    Ok(())
}

/// The layout guards that protect a dataset carrying an embedded object address.
///
/// The address is resolved as the elements are re-staged, which a compressed
/// chunk would need rewritten in place, and a userblock shifts every address by a
/// base this rewrite path does not model. Each is refused by name rather than
/// risking a mis-resolved address.
fn check_embedded_reference_layout(
    ds: &Dataset,
    path: &ObjectPathBuf,
    layout: &DataLayout,
    file: &Arc<File>,
) -> Result<(), Error> {
    if matches!(layout, DataLayout::Chunked { .. }) {
        return Err(Error::RepackUnsupported(format!(
            "dataset {path}: chunked or filtered datasets with an object-reference member cannot \
             be repacked (their addresses live inside compressed chunks and would need rewriting \
             in place)"
        )));
    }
    if ds.dataspace()?.extensible_max_dimensions().is_some() {
        return Err(Error::RepackUnsupported(format!(
            "dataset {path}: resizable datasets with an object-reference member cannot be repacked"
        )));
    }
    if !file.base_address().is_zero() {
        return Err(Error::RepackUnsupported(format!(
            "dataset {path}: object references in a file with a non-zero base address (userblock) \
             cannot be repacked yet"
        )));
    }
    Ok(())
}

/// Turns one stored object-reference address into the target the writer resolves
/// at serialization time. Null (0) and undefined (`HADDR_UNDEF`) point at nothing
/// and are carried verbatim; anything else must name a hard-linked object that
/// survives the repack.
fn resolve_reference_address(
    address: StoredAddress,
    path: &ObjectPathBuf,
    objects: &SourceObjects,
) -> Result<ObjectRefTarget, Error> {
    if address.get() == 0 || address.is_undefined(OBJECT_REFERENCE_WIDTH) {
        return Ok(ObjectRefTarget::Raw(address));
    }
    match objects.placement(address)? {
        Some(Placement::Kept(index)) => Ok(ObjectRefTarget::SourceIndex(index)),
        Some(Placement::Dropped(target_path)) => Err(Error::RepackUnsupported(format!(
            "dataset {path}: object reference to dropped object {} cannot be repacked",
            dropped_object(target_path.as_ref(), address)
        ))),
        None => Err(Error::RepackUnsupported(format!(
            "dataset {path}: object reference to address {:#x} resolves to no hard-linked \
             object in the source (dangling, or a region target not supported yet)",
            address.get()
        ))),
    }
}

/// The most objects and the most hard links a repack reads from its source.
const REPACK_BUDGET: TraversalBudget = TraversalBudget::new(1 << 24, 1 << 24);

/// Byte width of the object references this repack resolves.
///
/// libhdf5 sizes an object reference at `H5R_OBJ_REF_BUF_SIZE`,
/// `sizeof(haddr_t)` (`H5Rpublic.h`, HDF5 2.2.0), so the element is eight bytes
/// whatever address width the file declares, and [`datatype::embedded_reference_slots`]
/// locates no other form: an undefined reference has all eight of those bytes
/// set.
const OBJECT_REFERENCE_WIDTH: OffsetWidth = OffsetWidth::Eight;

/// Whether the source stores nothing at all: no chunk, no contiguous data
/// region.
///
/// Read from the file's own statement of where its data is rather than from the
/// values a read answers with, which cannot tell the two apart — a dataset that
/// stores a grid of fill values reads exactly like one that stores nothing.
///
/// A contiguous dataset says so with the undefined data address — but only once
/// external storage is out of the way, since `H5Pset_external` uses that same
/// encoding for a dataset whose bytes live in other files. `emit_dataset` refuses
/// one above, which is what leaves the address meaning what this reads it as. A
/// chunked one
/// is answered from `chunks` rather than from its index address, because the two
/// can disagree: an index that exists and holds no chunk is still a dataset with
/// nothing in it — this crate builds one eagerly for every empty resizable
/// dataset — and it is the chunks that a rewrite would materialize. Only the
/// chunked arm reads `chunks`, so a layout that has none passes an empty slice.
///
/// Compact data is inline in the layout message, so it is always allocated;
/// virtual layouts are refused by [`check_layout`] before this is asked.
fn storage_is_unallocated(layout: &DataLayout, chunks: &[ChunkInfo]) -> bool {
    match layout {
        DataLayout::Contiguous { address, .. } => address.is_none(),
        DataLayout::Chunked { .. } => chunks.is_empty(),
        DataLayout::Compact { .. } | DataLayout::Virtual => false,
    }
}

/// Reject data layouts that cannot be read and re-emitted (virtual datasets;
/// contiguous/chunked with an undefined address are allowed — they store
/// nothing, and [`storage_is_unallocated`] keeps them that way).
fn check_layout(layout: &DataLayout, path: &ObjectPathBuf) -> Result<(), Error> {
    match layout {
        DataLayout::Compact { .. } | DataLayout::Contiguous { .. } | DataLayout::Chunked { .. } => {
            Ok(())
        }
        DataLayout::Virtual => Err(Error::RepackUnsupported(format!(
            "dataset {path}: virtual data layout cannot be repacked"
        ))),
    }
}

/// Reject any filter that cannot be reproduced **by the re-encoding path**, so a
/// filtered dataset is never silently rewritten without its filters.
///
/// This guards only the two paths that read each dataset's *decompressed* bytes
/// and re-apply its filters from scratch: a contiguous/compact filtered dataset,
/// and the sparse-chunked fallback. A filter is safe there only when it is
/// **lossless** — then the re-encoded chunks decompress to the exact same bytes.
/// Deflate, shuffle, fletcher32, LZF, and integer scale-offset qualify. Float D-scale
/// scale-offset and ZFP are lossy: re-encoding already-decompressed values is not
/// guaranteed idempotent, so reproducing them could silently perturb the data,
/// and they are refused. SZIP this crate cannot write at all.
///
/// The dense chunked path (the common case) copies compressed chunks verbatim
/// and never calls this — there every filter is safe because nothing is decoded.
fn check_pipeline(
    pipeline: Option<&FilterPipeline>,
    path: &ObjectPathBuf,
) -> Result<Vec<FilterSpec>, Error> {
    let Some(p) = pipeline else {
        return Ok(Vec::new());
    };
    // Re-encoding replays the filters through the builder, which refuses lzf and
    // deflate on one dataset — they fill the same byte-compressor slot — so a
    // foreign pipeline carrying both cannot be reproduced. Refused here, by
    // name, rather than left to surface as the builder's combination error.
    let has = |id| p.filters.iter().any(|f| f.filter_id == id);
    if has(FILTER_LZF) && has(FILTER_DEFLATE) {
        return Err(Error::RepackUnsupported(format!(
            "dataset {path}: an lzf + deflate pipeline cannot be re-encoded faithfully"
        )));
    }
    p.filters
        .iter()
        .map(|f| {
            let kind = match f.filter_id {
                FILTER_SHUFFLE => FilterKind::Shuffle,
                FILTER_FLETCHER32 => FilterKind::Fletcher32,
                // Client-data[0] is the deflate level; default to 6 if absent.
                FILTER_DEFLATE => FilterKind::Deflate(
                    match f.client_data.first() {
                        Some(&level) => DeflateLevel::try_from(level)?,
                        None => DeflateLevel::Six,
                    }
                ),
                FILTER_LZF => FilterKind::Lzf,
                // Only the lossless integer mode qualifies, re-applied with the
                // source's minbits parameter. Its fill availability comes along
                // too: the value the filter records is the dataset's own fill
                // value, carried separately, so only the availability needs
                // saying — and saying it keeps a source that recorded none from
                // gaining one, which would re-encode every chunk a code point
                // wider.
                FILTER_SCALEOFFSET => match scaleoffset::scale_offset_mode(&f.client_data) {
                    Some((mode @ ScaleOffset::Integer(_), fill)) => {
                        FilterKind::ScaleOffset(mode, fill)
                    }
                    _ => {
                        return Err(Error::RepackUnsupported(format!(
                            "dataset {path}: only lossless integer scale-offset can be repacked faithfully"
                        )));
                    }
                },
                other => {
                    return Err(Error::RepackUnsupported(format!(
                        "dataset {path}: filter id {other} cannot be repacked yet"
                    )));
                }
            };
            Ok(FilterSpec {
                kind,
                optional: f.is_optional(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests;
