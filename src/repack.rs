//! Whole-file repack (issue #21): copy an existing HDF5 file into a fresh,
//! compact one, optionally dropping objects.
//!
//! [`File::open_rw`](crate::File::open_rw) deletes objects in place but reclaims
//! space only within a session and cannot return a single deleted-and-closed
//! file's bytes to the OS. Repack is the complementary answer — the same one the
//! HDF5 C ecosystem ships as `h5repack`: it reads every surviving object and
//! rewrites the whole file from scratch through [`FileBuilder`], so the result
//! has no dead space and is strictly smaller when objects are dropped.
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
//!   bit-field, opaque, compound, enumeration, and array datatypes,
//!   contiguous/compact or chunked.
//! - **Chunked** datasets copy their compressed chunks **verbatim** (chunk by
//!   chunk, never decoded), so *every* filter is preserved byte-exact: deflate,
//!   shuffle, fletcher32, integer **and** float scale-offset, ZFP, SZIP, and
//!   even filters this crate cannot itself apply. The destination always uses a
//!   v4 chunk index (single-chunk / fixed-array / extensible-array) regardless
//!   of the source index type.
//! - **Variable-length** datasets (1D and ND, contiguous/compact as well as
//!   chunked, filtered, or resizable): string-shaped (`is_string: true` and the
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
//! - **Unallocated storage**: a dataset the reference library created and never
//!   wrote to holds nothing, and comes out holding nothing rather than holding
//!   the fill value a read answers with for every element it declares (issue
//!   #293). A *resizable* destination keeps the eagerly built chunk index every
//!   empty resizable dataset here gets, since an in-place append needs one to
//!   exist; it stores no chunk either way.
//! - Group hierarchy of arbitrary depth.
//! - Attributes, on datasets, groups, and root, carried across with the
//!   encoding the source gave them rather than rebuilt from an [`AttrValue`],
//!   which is a decoded view and cannot express a byte order, a sub-width
//!   precision, a variable-length string, or a rank above one.
//! - The source file's file-space management strategy (with its page size and
//!   threshold), carried into the compact output as non-persistent — a repacked
//!   file has no free space to persist.
//!
//! The verbatim chunk copy never decodes, so it eliminates the
//! decompress→recompress round-trip and the per-dataset decompression blowup,
//! and a lossy filter survives byte-exact. Two paths still re-encode and so
//! require **lossless** filters: a *contiguous/compact* filtered dataset, and a
//! *sparse* chunked dataset (one with unallocated chunk-grid holes, which the
//! dense verbatim path cannot lay out). A lossy pipeline on either of those is
//! refused.
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
//! Repack is **out-of-core** (issue [#82]): it opens the source with
//! [`File::open_streaming`], reading metadata and one working chunk on demand
//! rather than buffering the whole file, copies each chunked dataset's
//! compressed chunks verbatim one at a time, and streams the output straight to
//! the destination. Peak memory is therefore bounded by a single chunk plus the
//! file's metadata, independent of dataset (or file) size, so a file whose data
//! exceeds available RAM repacks successfully.
//!
//! [#82]: https://github.com/CramBL/hdf5-pure/issues/82

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

use crate::attribute::AttributeMessage;
use crate::chunked_read::ChunkInfo;
use crate::chunked_write::{ChunkMeta, ChunkProvider, FilterKind, FilterSpec};
use crate::convert::TryToUsize;
use crate::data_layout::DataLayout;
use crate::datatype::{Datatype, ReferenceType, embedded_reference_slots};
use crate::error::{Error, FormatError};
use crate::filter_pipeline::{
    FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_LZF, FILTER_SCALEOFFSET, FILTER_SHUFFLE,
    FilterPipeline,
};
use crate::libver::LibVer;
use crate::reader::{Dataset, File, Group};
use crate::scaleoffset::{self, ScaleOffset};
use crate::shared_message::DatatypeLocation;
use crate::source::Source;
use crate::type_builders::{
    AttrValue, DatasetBuilder, FinishedGroup, GroupBuilder, ObjectRefPatch, ObjectRefTarget,
    VlStringElement,
};
use crate::vl_data::{
    EmbeddedVlSlot, VlByteObject, VlenStringReadOptions, embedded_vlen_slots,
    is_vlen_string_datatype,
};
use crate::writer::FileBuilder;

/// Options controlling a [`repack`].
///
/// Built with [`new`](Self::new) and [`drop_path`](Self::drop_path); the fields
/// are private so a future option is an additive change.
#[derive(Debug, Default, Clone)]
pub struct RepackOptions {
    /// Full paths of objects to omit from the output. See
    /// [`drop_path`](Self::drop_path).
    drop: Vec<String>,
    /// Library-version bounds for the output, from
    /// [`with_libver_bounds`](Self::with_libver_bounds). `None` carries the
    /// source's own format forward.
    libver_bounds: Option<(LibVer, LibVer)>,
}

impl RepackOptions {
    /// Options that drop nothing — a pure compaction copy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Omit the object at `path` from the output (e.g. `"grp/old"` or
    /// `"/grp/old"`; leading and trailing slashes are ignored). Dropping a group
    /// drops its whole subtree. Every listed path must exist in the source, or
    /// the repack fails — a no-op drop is treated as a mistake rather than
    /// silently ignored. Chainable.
    pub fn drop_path(mut self, path: &str) -> Self {
        self.drop.push(path.to_string());
        self
    }

    /// The paths this repack will omit, in the order they were added.
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
}

/// Repack `src` into a new file at `dst`, applying `options`.
///
/// Reads every object of `src` not excluded by [`RepackOptions::drop_path`] and
/// writes them into a fresh, compact file at `dst`. On success `dst` is a normal
/// HDF5 file holding exactly the surviving objects with no dead space.
///
/// The fidelity checks run first: every object is validated while the output is
/// staged, so an [`Error::RepackUnsupported`] (an object that cannot be
/// reproduced faithfully, or a drop path that does not exist) is reported before
/// any byte is written to `dst`. Dataset *chunk bytes*, by contrast, are streamed
/// from `src` to `dst` during the write rather than buffered, so an I/O error
/// reading the source or writing the destination partway through can leave a
/// partial `dst` (remove it and retry).
///
/// See [`Error::RepackUnsupported`] for the objects that cannot be reproduced
/// faithfully.
///
/// `src` is opened with [`File::open_streaming`], so a file whose superblock
/// marks it as held by a writer is refused with
/// [`Error::FileMarkedInUse`] — repacking a file a writer is still growing would
/// capture a torn view. Clear a flag a crashed writer left with
/// [`File::clear_swmr_flag`](crate::File::clear_swmr_flag) first, as `h5repack`
/// needs `h5clear`.
pub fn repack<P: AsRef<Path>, Q: AsRef<Path>>(
    src: P,
    dst: Q,
    options: &RepackOptions,
) -> Result<(), Error> {
    // Open the source for on-demand streaming reads: metadata and one working
    // chunk are resident at a time, never the whole file. Shared so each streamed
    // dataset's chunk provider can pull from the same handle during the write
    // without an extra open.
    let file = Arc::new(File::open_streaming(src)?);

    // Normalize the drop set to canonical slash-free paths and remember which
    // ones actually match, so an unmatched drop can be reported as an error.
    let drop: BTreeSet<String> = options.drop.iter().map(|p| normalize(p)).collect();
    let mut matched: BTreeSet<String> = BTreeSet::new();

    let mut builder = FileBuilder::new();
    // Carry the source's file-space strategy forward. The repacked file is
    // compact with no free space, so the strategy and its page size/threshold
    // are preserved but `persist` is reset to false — there is nothing to
    // persist, and writing persistent free-space blocks is a separate feature.
    if let Some(info) = file.file_space_info() {
        builder
            .with_file_space_strategy(info.strategy, false, info.threshold)
            .with_file_space_page_size(info.page_size);
    }
    // Map every source object's (relative) header address to its path, so an
    // object-reference dataset can be rewritten to point at the same objects in
    // the compacted output rather than at their stale source addresses.
    let addr_map = build_object_address_map(&file)?;

    let root = file.root();
    populate(
        &mut builder,
        &root,
        "",
        &drop,
        &mut matched,
        &file,
        &addr_map,
    )?;

    // Choose the output format now that the content is staged.
    //
    // A repack compacts a file; it is not a request to re-target it, and the
    // format is the one property whose loss is invisible until an old reader
    // refuses the output — a `.mat` written to the 1.8 bound so MATLAB can load
    // it used to come back with a version 3 superblock MATLAB cannot open at
    // all. So the source's format is carried forward.
    //
    // Except where the content will not fit in it. A chunked dataset needs the
    // version 4 data-layout message and a 1.10 chunk index, and 1.8's version 1
    // B-tree index is one this crate reads but does not write — so a great many
    // files the C library wrote (chunked data under a version 0 or 2 superblock)
    // have no older encoding to be repacked back into. Upgrading there is forced
    // rather than gratuitous, and refusing instead would take repack away from
    // most real HDF5 files. `h5repack` makes the same trade, writing whatever
    // format the content needs unless `--low`/`--high` say otherwise.
    //
    // `RepackOptions::with_libver_bounds` is that "unless": set explicitly, the
    // bound is a guarantee, and content it cannot express is refused by the
    // writer rather than upgraded past it.
    //
    // The carried-forward ceiling is floored at `WRITER_OLDEST` because a
    // version 0/1 superblock source is older than anything this crate writes:
    // that format can be read but not produced, so the oldest writable one is as
    // close as the output gets. `Earliest` as the low bound rather than the
    // source's own version, since `high` is what selects the format and a floor
    // equal to it would leave nothing satisfiable.
    let (low, high) = match options.libver_bounds {
        Some(explicit) => explicit,
        None if builder.needs_latest_format() => (LibVer::Earliest, LibVer::WRITER_DEFAULT),
        None => (
            LibVer::Earliest,
            file.libver_bound().max(LibVer::WRITER_OLDEST),
        ),
    };
    builder.with_libver_bounds(low, high);

    // Every requested drop must have named a real object.
    if let Some(missing) = drop.iter().find(|d| !matched.contains(*d)) {
        return Err(Error::RepackUnsupported(format!(
            "drop path does not exist in the source: {missing}"
        )));
    }

    builder.write(dst)?;
    Ok(())
}

/// A destination that group contents can be added to. Implemented for both the
/// top-level [`FileBuilder`] (the root group) and [`GroupBuilder`] (subgroups)
/// so one recursive walk handles every level.
trait GroupSink: AttrSink {
    fn sink_dataset(&mut self, name: &str) -> &mut DatasetBuilder;
    fn sink_add_group(&mut self, group: FinishedGroup);
    fn sink_commit_datatype(&mut self, name: &str, datatype: Datatype);
}

/// Anything a repacked attribute can be attached to: the root group, a subgroup,
/// or a dataset. Split from [`GroupSink`] because a dataset takes attributes but
/// holds no children, so one attribute-copying routine serves all three.
trait AttrSink {
    fn sink_set_attr(&mut self, name: &str, value: AttrValue);
    fn sink_set_attr_verbatim(&mut self, message: AttributeMessage);
    fn sink_set_attr_var_len_verbatim(&mut self, message: AttributeMessage, strings: Vec<String>);
}

impl AttrSink for FileBuilder {
    fn sink_set_attr(&mut self, name: &str, value: AttrValue) {
        self.set_attr(name, value);
    }
    fn sink_set_attr_verbatim(&mut self, message: AttributeMessage) {
        self.set_attr_verbatim(message);
    }
    fn sink_set_attr_var_len_verbatim(&mut self, message: AttributeMessage, strings: Vec<String>) {
        self.set_attr_var_len_verbatim(message, strings);
    }
}

impl AttrSink for GroupBuilder {
    fn sink_set_attr(&mut self, name: &str, value: AttrValue) {
        self.set_attr(name, value);
    }
    fn sink_set_attr_verbatim(&mut self, message: AttributeMessage) {
        self.set_attr_verbatim(message);
    }
    fn sink_set_attr_var_len_verbatim(&mut self, message: AttributeMessage, strings: Vec<String>) {
        self.set_attr_var_len_verbatim(message, strings);
    }
}

impl AttrSink for DatasetBuilder {
    fn sink_set_attr(&mut self, name: &str, value: AttrValue) {
        self.set_attr(name, value);
    }
    fn sink_set_attr_verbatim(&mut self, message: AttributeMessage) {
        self.set_attr_verbatim(message);
    }
    fn sink_set_attr_var_len_verbatim(&mut self, message: AttributeMessage, strings: Vec<String>) {
        self.set_attr_var_len_verbatim(message, strings);
    }
}

impl GroupSink for FileBuilder {
    fn sink_dataset(&mut self, name: &str) -> &mut DatasetBuilder {
        self.create_dataset(name)
    }
    fn sink_add_group(&mut self, group: FinishedGroup) {
        self.add_group(group);
    }
    fn sink_commit_datatype(&mut self, name: &str, datatype: Datatype) {
        self.commit_datatype(name, datatype);
    }
}

impl GroupSink for GroupBuilder {
    fn sink_dataset(&mut self, name: &str) -> &mut DatasetBuilder {
        self.create_dataset(name)
    }
    fn sink_add_group(&mut self, group: FinishedGroup) {
        self.add_group(group);
    }
    fn sink_commit_datatype(&mut self, name: &str, datatype: Datatype) {
        self.commit_datatype(name, datatype);
    }
}

/// Copy `src`'s attributes, datasets, and subgroups (recursively) into `sink`,
/// skipping anything whose path is in `drop`. `path` is the slash-free path of
/// `src` itself (empty for the root).
fn populate<S: GroupSink>(
    sink: &mut S,
    src: &Group,
    path: &str,
    drop: &BTreeSet<String>,
    matched: &mut BTreeSet<String>,
    file: &Arc<File>,
    addr_map: &HashMap<u64, String>,
) -> Result<(), Error> {
    // Attributes, copied verbatim where their bytes travel and re-encoded where
    // they do not; refused rather than dropped if neither is possible.
    let owner = if path.is_empty() {
        "root group".to_string()
    } else {
        format!("group {path}")
    };
    copy_attrs(
        sink,
        src.attr_messages()?,
        || src.attrs(),
        &owner,
        drop,
        addr_map,
    )?;

    // Committed (`H5Tcommit`) datatype objects. Neither a group nor a dataset, so
    // the two walks below never see one: without this loop the object and its
    // link leave the output, and every dataset that named the type through it
    // stops resolving. Placed before the datasets so the type exists in the
    // output before anything references it (the writer resolves by path, so the
    // order is not load-bearing, but it matches how the file reads).
    for name in src.named_datatypes()? {
        let child_path = join(path, &name);
        if drop.contains(&child_path) {
            matched.insert(child_path);
            continue;
        }
        let (datatype, _) = src.named_datatype_at(&name)?;
        check_datatype(&datatype, &format!("committed datatype {child_path}"))?;
        sink.sink_commit_datatype(&name, datatype);
    }

    // Datasets, sorted by name.
    let mut dataset_names = src.datasets()?;
    dataset_names.sort_unstable();
    for name in dataset_names {
        let child_path = join(path, &name);
        if drop.contains(&child_path) {
            matched.insert(child_path);
            continue;
        }
        let ds = src.dataset(&name)?;
        emit_dataset(
            sink.sink_dataset(&name),
            &ds,
            &child_path,
            file,
            drop,
            addr_map,
        )?;
    }

    // Subgroups, sorted by name; built depth-first into a FinishedGroup.
    let mut group_names = src.groups()?;
    group_names.sort_unstable();
    for name in group_names {
        let child_path = join(path, &name);
        if drop.contains(&child_path) {
            matched.insert(child_path);
            continue;
        }
        let child = src.group(&name)?;
        let mut gb = GroupBuilder::new(&name);
        populate(&mut gb, &child, &child_path, drop, matched, file, addr_map)?;
        sink.sink_add_group(gb.finish());
    }
    Ok(())
}

/// Capture one dataset's full description and stage it on `db`, or fail with a
/// named [`Error::RepackUnsupported`] if any part cannot be reproduced.
fn emit_dataset(
    db: &mut DatasetBuilder,
    ds: &Dataset,
    path: &str,
    file: &Arc<File>,
    drop: &BTreeSet<String>,
    addr_map: &HashMap<u64, String>,
) -> Result<(), Error> {
    // A committed (`H5Tcommit`) element type lives in its own object header. The
    // dataset is reproduced naming the same type rather than inlining a copy of
    // it, which would read back correctly and still lose the name every C-library
    // reader reports.
    if let Some(address) = ds.committed_datatype_address()? {
        let type_path = committed_type_path(address, &format!("dataset {path}"), drop, addr_map)?;
        db.with_committed_datatype(&type_path);
    }

    let datatype = ds.datatype()?;
    let dataspace = ds.dataspace()?;
    let layout = ds.data_layout()?;
    let pipeline = ds.filter_pipeline_parsed();

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
    let vlen_slots = embedded_vlen_slots(&datatype).ok_or_else(|| {
        Error::RepackUnsupported(format!(
            "dataset {path}: datatype declares variable-length members its own element \
             size cannot hold"
        ))
    })?;
    // Likewise every object-header address it reaches. Both kinds of address are
    // rewritten rather than copied, so both disqualify the paths that move
    // element bytes unchanged.
    let reference_slots = embedded_reference_slots(&datatype).ok_or_else(|| {
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
    if fill.is_some() && !(vlen_slots.is_empty() && reference_slots.is_empty()) {
        return Err(Error::RepackUnsupported(format!(
            "dataset {path}: a fill value on a variable-length or object-reference \
             dataset cannot be repacked faithfully"
        )));
    }

    // Storage the source never allocated is reproduced as storage, not as the
    // values reading it answers with. By default the reference library does not
    // allocate a contiguous or chunked dataset's storage until something is
    // written to it, so one created and never written holds nothing at all
    // (compact data is inline and is always present); since #292 reading it
    // answers the fill value for every element, and re-writing those values
    // would turn a schema-only file into a fully materialized one of the size
    // its shape declares (issue #293).
    //
    // Placed after every refusal the datatype and layout share and before the
    // paths that move element bytes, all of which exist to rebuild an address
    // this dataset does not store: there is no data to carry, so each of them
    // would be reproducing the fill value it read rather than anything the
    // source held. `check_pipeline` still runs, so a filter this crate could not
    // re-apply is refused here exactly as it is on the re-encode path below —
    // the filters are reproduced through the same `carry_shape_and_pipeline`.
    if storage_is_unallocated(&layout, &source_chunks) {
        let filters = check_pipeline(pipeline.as_ref(), path)?;
        db.fill = fill;
        db.with_unallocated_storage(datatype, &dims);
        carry_shape_and_pipeline(
            db,
            &dims,
            dataspace.max_dimensions.as_deref(),
            &layout,
            &filters,
        );
        copy_dataset_attrs(db, ds, path, drop, addr_map)?;
        return Ok(());
    }

    // Variable-length string datasets take a dedicated path: their element
    // references point into the global heap, so they are re-emitted by reading
    // each element's exact heap bytes and re-staging them, not by copying raw
    // element bytes (whose stored heap addresses would go stale on rewrite).
    if is_vlen_string_datatype(&datatype) {
        emit_vlen_string_dataset(db, ds, path, &datatype, &dims, &layout, &pipeline)?;
        // VL-string datasets carry attributes the same way as any other.
        copy_dataset_attrs(db, ds, path, drop, addr_map)?;
        return Ok(());
    }

    // Non-string variable-length (sequence) datasets take the same global-heap
    // re-staging path as VL strings: each element's exact heap bytes are read and
    // re-emitted through a fresh global heap, so the stored heap addresses are
    // rebuilt rather than copied stale. Routed here before the verbatim chunk-copy
    // path so a chunked one is refused (not copied with stale references).
    if is_nonstring_vlen(&datatype) {
        emit_vlen_sequence_dataset(db, ds, path, &datatype, &dims, &layout, &pipeline)?;
        copy_dataset_attrs(db, ds, path, drop, addr_map)?;
        return Ok(());
    }

    // Object-reference datasets store absolute object-header addresses that would
    // go stale on rewrite, so each reference is resolved to its target's *new*
    // address (via the source address->path map and the writer's path resolution)
    // rather than copied. Routed here before the verbatim chunk-copy path so a
    // chunked one is refused (not copied with stale addresses).
    if is_object_reference(&datatype) {
        emit_object_reference_dataset(db, ds, path, &dims, &layout, file, drop, addr_map)?;
        copy_dataset_attrs(db, ds, path, drop, addr_map)?;
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
            drop,
            addr_map,
        )?;
        copy_dataset_attrs(db, ds, path, drop, addr_map)?;
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
        chunk_dimensions, ..
    } = &layout
        && n_elements > 0
    {
        let rank = dims.len();
        let chunk_dims: Vec<u64> = chunk_dimensions
            .iter()
            .take(rank)
            .map(|&c| c as u64)
            .collect();

        if let Some(DenseChunkPlan { meta, grid_order }) =
            try_plan_dense_chunks(source_chunks, &dims, &chunk_dims)
        {
            let maxshape = dataspace
                .max_dimensions
                .as_ref()
                .filter(|ms| *ms != &dims)
                .map(|ms| ms.as_slice());
            let elem_size = datatype.element_size_usize()?;
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
                ds.filter_pipeline_message_bytes()?,
                meta,
                Box::new(provider),
            );

            // Carry the dataset's attributes, refusing any that cannot be
            // represented.
            copy_dataset_attrs(db, ds, path, drop, addr_map)?;
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
        dataspace.max_dimensions.as_deref(),
        &layout,
        &filters,
    );

    // Carry the dataset's attributes, refusing if any cannot be represented.
    copy_dataset_attrs(db, ds, path, drop, addr_map)?;

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
    path: &str,
    drop: &BTreeSet<String>,
    addr_map: &HashMap<u64, String>,
) -> Result<(), Error> {
    copy_attrs(
        db,
        ds.attr_messages()?,
        || ds.attrs(),
        &format!("dataset {path}"),
        drop,
        addr_map,
    )
}

/// Carry the source dataset's resizability, chunk geometry, and filter pipeline
/// onto the rebuilt dataset, so a repack reproduces the layout it read rather
/// than flattening it. Shared by the fixed-size re-encode path and the
/// variable-length re-staging paths, which both rebuild their elements from
/// scratch and so must reapply the layout themselves.
///
/// `filters` are the ones [`check_pipeline`] decoded, which is what makes them
/// safe to re-apply: there is no way to reach this with a pipeline that was not
/// proved reproducible first.
fn carry_shape_and_pipeline(
    db: &mut DatasetBuilder,
    dims: &[u64],
    max_dimensions: Option<&[u64]>,
    layout: &DataLayout,
    filters: &[FilterSpec],
) {
    // A max-shape that differs from the current shape means a resizable dataset.
    if let Some(maxshape) = max_dimensions
        && maxshape != dims
    {
        db.with_maxshape(maxshape);
    }

    // Chunking: the v3 layout appends the element size as a trailing chunk
    // dimension, so keep only the first `rank` entries; v4 already stores `rank`.
    if let DataLayout::Chunked {
        chunk_dimensions, ..
    } = layout
    {
        let rank = dims.len();
        let logical: Vec<u64> = chunk_dimensions
            .iter()
            .take(rank)
            .map(|&c| c as u64)
            .collect();
        db.with_chunks(&logical);
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
    path: &str,
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
        ds.dataspace()?.max_dimensions.as_deref(),
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
    path: &str,
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
        ds.dataspace()?.max_dimensions.as_deref(),
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
/// resolved to the path it names so the writer can fill in the target's *new*
/// location. Copying the element bytes verbatim instead would leave every
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
    path: &str,
    datatype: &Datatype,
    dims: &[u64],
    layout: &DataLayout,
    pipeline: &Option<FilterPipeline>,
    vlen_slots: &[EmbeddedVlSlot],
    reference_slots: &[usize],
    file: &Arc<File>,
    drop: &BTreeSet<String>,
    addr_map: &HashMap<u64, String>,
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
        check_embedded_reference_layout(ds, path, dims, layout, file)?;
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
        resolve_embedded_references(&raw, datatype, dims, path, drop, addr_map, reference_slots)?;

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
        ds.dataspace()?.max_dimensions.as_deref(),
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
    path: &str,
    drop: &BTreeSet<String>,
    addr_map: &HashMap<u64, String>,
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
            let v = u64::from_le_bytes(
                raw[at..at + 8]
                    .try_into()
                    .expect("slot offsets leave 8 bytes inside the element"),
            );
            patches.push(ObjectRefPatch {
                byte_offset: at,
                target: resolve_reference_address(v, path, drop, addr_map)?,
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
        // no decode and no `addr_offset` adjustment — the same slice the chunked
        // reader consumes. `read_at` fills the whole buffer or errors, and the
        // emitter additionally checks the length against the planned size, so the
        // layout cannot silently desync from the data. Reading straight into the
        // emitter's reused buffer keeps repack at one chunk-sized allocation for
        // the whole dataset.
        let info = &self.grid_order[index];
        let source = self.file.source();
        let len = info.chunk_size as usize;
        // Bounds-check before growing the buffer, the way `Source::read_exact_at`
        // does and for its reason: `chunk_size` comes from the source's chunk
        // index, so a malformed file could name a 4 GiB chunk and have this zero
        // that much memory only for the read to fail EOF anyway.
        let end = info
            .address
            .checked_add(len as u64)
            .ok_or(FormatError::OffsetOverflow {
                offset: info.address,
                length: len as u64,
            })?;
        if end > source.len() {
            return Err(FormatError::UnexpectedEof {
                expected: end.to_usize().unwrap_or(usize::MAX),
                available: source.len().to_usize().unwrap_or(usize::MAX),
            });
        }
        let start = out.len();
        out.resize(start + len, 0);
        source.read_at(info.address, &mut out[start..])
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
            compressed_size: u64::from(info.chunk_size),
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
        Datatype::FixedPoint { .. }
        | Datatype::FloatingPoint { .. }
        | Datatype::Time { .. }
        | Datatype::String { .. }
        | Datatype::BitField { .. }
        | Datatype::Opaque { .. } => true,
        Datatype::Compound { members, .. } => members
            .iter()
            .all(|m| attr_bytes_are_position_independent(&m.datatype)),
        Datatype::Enumeration { base_type, .. } | Datatype::Array { base_type, .. } => {
            attr_bytes_are_position_independent(base_type)
        }
        Datatype::VariableLength { .. } | Datatype::Reference { .. } => false,
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
    drop: &BTreeSet<String>,
    addr_map: &HashMap<u64, String>,
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
    for message in &mut messages {
        let DatatypeLocation::Committed(address) = message.datatype_location else {
            continue;
        };
        let name = &message.name;
        let type_path = committed_type_path(
            address,
            &format!("{owner} attribute {name:?}"),
            drop,
            addr_map,
        )?;
        message.datatype_location = DatatypeLocation::CommittedPath(type_path);
    }
    let any_needs_decoding = messages
        .iter()
        .any(|m| !attr_bytes_are_position_independent(&m.datatype));
    let decoded = if any_needs_decoding {
        decode()?
    } else {
        std::collections::HashMap::new()
    };

    for message in messages {
        if attr_bytes_are_position_independent(&message.datatype) {
            sink.sink_set_attr_verbatim(message);
        } else if let Some(value) = decoded.get(&message.name) {
            // A variable-length datatype the decode resolved into strings is
            // exactly the case that keeps its datatype and dataspace but restages
            // its payload. Gating on what the decode *produced* rather than
            // re-classifying the datatype here keeps the two from drifting apart.
            match (&message.datatype, value.as_strings()) {
                (Datatype::VariableLength { .. }, Some(strings)) => {
                    let strings = strings.to_vec();
                    sink.sink_set_attr_var_len_verbatim(message, strings);
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
        // Scalar and opaque-bytes datatypes whose on-disk form `Datatype::serialize`
        // reproduces exactly (including the time type's byte order), so reading the
        // raw element bytes and re-emitting them is byte-for-byte faithful.
        Datatype::FixedPoint { .. }
        | Datatype::FloatingPoint { .. }
        | Datatype::Time { .. }
        | Datatype::String { .. }
        | Datatype::BitField { .. }
        | Datatype::Opaque { .. } => Ok(()),
        // String-shaped variable-length datatypes (`is_string: true`, or the
        // MATLAB VLEN-of-1-byte-ASCII-string shape) are reproduced by reading
        // each element's exact heap bytes and re-staging them through the
        // writer's VL-string path; the layout/filter checks gate chunked ones.
        Datatype::VariableLength { .. } if is_vlen_string_datatype(dt) => Ok(()),
        // Non-string VL (sequences of arbitrary base types) are re-staged the
        // same way, but only when the base type's bytes carry no embedded heap or
        // file addresses that a verbatim copy would leave stale.
        Datatype::VariableLength { base_type, .. } => check_vlen_base_type(base_type, owner),
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
        Datatype::Compound { members, .. } => {
            for m in members {
                check_datatype(&m.datatype, owner)?;
            }
            Ok(())
        }
        Datatype::Enumeration { base_type, .. } => check_datatype(base_type, owner),
        Datatype::Array { base_type, .. } => check_datatype(base_type, owner),
    }
}

/// Whether `dt` is a non-string variable-length (sequence) datatype — the kind
/// re-emitted by [`emit_vlen_sequence_dataset`]. Excludes the string-shaped VL
/// datatypes, which [`emit_vlen_string_dataset`] handles.
fn is_nonstring_vlen(dt: &Datatype) -> bool {
    matches!(dt, Datatype::VariableLength { .. }) && !is_vlen_string_datatype(dt)
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
        Datatype::FixedPoint { .. }
        | Datatype::FloatingPoint { .. }
        | Datatype::Time { .. }
        | Datatype::String { .. }
        | Datatype::BitField { .. }
        | Datatype::Opaque { .. } => Ok(()),
        Datatype::Reference { .. } => bad("references"),
        Datatype::VariableLength { .. } => bad("variable-length elements"),
        Datatype::Compound { members, .. } => {
            for m in members {
                check_vlen_base_type(&m.datatype, owner)?;
            }
            Ok(())
        }
        Datatype::Enumeration { base_type, .. } => check_vlen_base_type(base_type, owner),
        Datatype::Array { base_type, .. } => check_vlen_base_type(base_type, owner),
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

/// Whether `path` is dropped from the output: either listed in `drop`, or nested
/// under a dropped group (so its whole subtree is gone).
fn is_dropped(path: &str, drop: &BTreeSet<String>) -> bool {
    if drop.contains(path) {
        return true;
    }
    let mut p = path;
    while let Some(idx) = p.rfind('/') {
        p = &p[..idx];
        if drop.contains(p) {
            return true;
        }
    }
    false
}

/// Build a map from each source object's header address to its slash-free path,
/// for resolving object references. With a zero base address (the case object
/// references are repacked for) the stored reference value is exactly this
/// header address, so the lookup is direct.
fn build_object_address_map(file: &File) -> Result<HashMap<u64, String>, Error> {
    let mut map = HashMap::new();
    let root = file.root();
    // The root group can itself be referenced (the writer registers it under the
    // empty path).
    map.insert(root.header_address()?, String::new());
    collect_addresses(&root, "", &mut map)?;
    Ok(map)
}

/// Recursively record `(header address -> path)` for every dataset, committed
/// datatype, and subgroup.
fn collect_addresses(
    group: &Group,
    prefix: &str,
    map: &mut HashMap<u64, String>,
) -> Result<(), Error> {
    for (name, ds) in group.iter_datasets()? {
        map.insert(ds.header_address()?, join(prefix, &name));
    }
    // A committed datatype is an object with an address like any other: a dataset
    // or attribute naming one is resolved through this map, and an object
    // reference may point straight at it.
    for name in group.named_datatypes()? {
        let (_, address) = group.named_datatype_at(&name)?;
        map.insert(address, join(prefix, &name));
    }
    for (name, child) in group.iter_groups()? {
        let child_path = join(prefix, &name);
        map.insert(child.header_address()?, child_path.clone());
        collect_addresses(&child, &child_path, map)?;
    }
    Ok(())
}

/// The destination path of the committed datatype at source `address`.
///
/// Refused by name, rather than resolved to something else, in the two cases
/// where naming it would be wrong: the type was dropped from the output, or it
/// sits outside the hard-link hierarchy this walk covers (so nothing in the
/// output holds it). Both would otherwise surface as a dataset pointing at an
/// object that is not there.
fn committed_type_path(
    address: u64,
    user: &str,
    drop: &BTreeSet<String>,
    addr_map: &HashMap<u64, String>,
) -> Result<String, Error> {
    let path = addr_map.get(&address).ok_or_else(|| {
        Error::RepackUnsupported(format!(
            "{user}: names a committed datatype that is not reachable by a hard link in the \
             source, so it has no place in the output"
        ))
    })?;
    // `is_dropped` rather than a membership test: dropping a group drops its whole
    // subtree, so a committed type inside one leaves the output without ever being
    // named in the drop set.
    if is_dropped(path, drop) {
        return Err(Error::RepackUnsupported(format!(
            "{user}: names the committed datatype {path:?}, which this repack drops"
        )));
    }
    Ok(path.clone())
}

/// Re-emit an object-reference dataset faithfully: rewrite each stored address to
/// point at its target's destination location instead of its stale source one.
///
/// Each reference is read, resolved through `addr_map` to a source path, and
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
    path: &str,
    dims: &[u64],
    layout: &DataLayout,
    file: &Arc<File>,
    drop: &BTreeSet<String>,
    addr_map: &HashMap<u64, String>,
) -> Result<(), Error> {
    if matches!(layout, DataLayout::Chunked { .. }) {
        return Err(Error::RepackUnsupported(format!(
            "dataset {path}: chunked or filtered object-reference datasets cannot be repacked \
             (their addresses live inside compressed chunks and would need rewriting in place)"
        )));
    }
    if let Some(maxshape) = &ds.dataspace()?.max_dimensions
        && maxshape != dims
    {
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
            let v = u64::from_le_bytes(*chunk);
            targets.push(resolve_reference_address(v, path, drop, addr_map)?);
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
    path: &str,
    dims: &[u64],
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
    if let Some(maxshape) = &ds.dataspace()?.max_dimensions
        && maxshape != dims
    {
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

/// Turn one stored object-reference address into the target the writer resolves
/// at serialization time. Null (0) and undefined (`HADDR_UNDEF`) point at nothing
/// and are carried verbatim; anything else must name a hard-linked object that
/// survives the repack.
fn resolve_reference_address(
    address: u64,
    path: &str,
    drop: &BTreeSet<String>,
    addr_map: &HashMap<u64, String>,
) -> Result<ObjectRefTarget, Error> {
    if address == 0 || address == u64::MAX {
        return Ok(ObjectRefTarget::Raw(address));
    }
    match addr_map.get(&address) {
        Some(target_path) if is_dropped(target_path, drop) => {
            Err(Error::RepackUnsupported(format!(
                "dataset {path}: object reference to dropped object {target_path:?} cannot be repacked"
            )))
        }
        Some(target_path) => Ok(ObjectRefTarget::Path(target_path.clone())),
        None => Err(Error::RepackUnsupported(format!(
            "dataset {path}: object reference to address {address:#x} resolves to no hard-linked \
             object in the source (dangling, or a region target not supported yet)"
        ))),
    }
}

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
        DataLayout::Compact { .. } | DataLayout::Virtual { .. } => false,
    }
}

/// Reject data layouts that cannot be read and re-emitted (virtual datasets;
/// contiguous/chunked with an undefined address are allowed — they store
/// nothing, and [`storage_is_unallocated`] keeps them that way).
fn check_layout(layout: &DataLayout, path: &str) -> Result<(), Error> {
    match layout {
        DataLayout::Compact { .. } | DataLayout::Contiguous { .. } | DataLayout::Chunked { .. } => {
            Ok(())
        }
        DataLayout::Virtual { .. } => Err(Error::RepackUnsupported(format!(
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
fn check_pipeline(pipeline: Option<&FilterPipeline>, path: &str) -> Result<Vec<FilterSpec>, Error> {
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
                FILTER_DEFLATE => FilterKind::Deflate(f.client_data.first().copied().unwrap_or(6)),
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

/// Canonicalize a path to slash-free form: split on `/`, drop empty components,
/// rejoin. `"/a//b/"` and `"a/b"` both become `"a/b"`.
fn normalize(path: &str) -> String {
    path.split('/')
        .filter(|c| !c.is_empty())
        .collect::<Vec<_>>()
        .join("/")
}

/// Join a parent path (slash-free, possibly empty) with a child name.
fn join(parent: &str, name: &str) -> String {
    if parent.is_empty() {
        name.to_string()
    } else {
        format!("{parent}/{name}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A source's scale-offset fill availability is carried onto the rebuilt
    /// dataset, in both directions.
    ///
    /// It has to be said explicitly, because the rebuilt dataset's fill value
    /// says nothing about it: this crate records a defined fill value for every
    /// scale-offset dataset it writes, so a source that recorded none would
    /// otherwise gain one — re-encoding every chunk a code point wider, to
    /// reserve a sentinel for a value the source never treated as fill.
    ///
    /// Synthetic pipelines rather than a file-level test, for the same reason
    /// [`lzf_plus_deflate_pipeline_is_refused`] uses one: no writer left in this
    /// crate produces a filter that records no fill value, so the only files
    /// carrying one are those written before it and those a C caller set the
    /// fill value undefined on.
    #[test]
    fn a_repacked_scale_offset_filter_keeps_the_sources_fill_availability() {
        use crate::filter_pipeline::FilterDescription;
        use crate::scaleoffset::{FillAvailability, ScaleOffsetFill, build_cd_values};

        let ty = crate::scaleoffset::scale_offset_type_from_datatype(&crate::make_i32_type())
            .expect("i32 is a scale-offset type");
        let carried = |fill: ScaleOffsetFill<'_>| {
            let pipeline = Some(FilterPipeline {
                version: 2,
                filters: vec![FilterDescription {
                    filter_id: FILTER_SCALEOFFSET,
                    name: None,
                    flags: 0,
                    client_data: build_cd_values(ScaleOffset::Integer(0), ty, 4, 16, fill).unwrap(),
                }],
            });
            let mut db = DatasetBuilder::new("d");
            carry_shape_and_pipeline(
                &mut db,
                &[16],
                None,
                &DataLayout::Chunked {
                    // A v3 layout appends the element size, which is what the
                    // caller trims back off.
                    chunk_dimensions: vec![16, 4],
                    btree_address: Some(0x1000),
                    version: 3,
                    chunk_index_type: None,
                    single_chunk_filtered_size: None,
                    single_chunk_filter_mask: None,
                },
                // Routed through `check_pipeline`, as every production caller
                // is: it is the step that decodes the availability out of the
                // source's parameters.
                &check_pipeline(pipeline.as_ref(), "d").expect("integer scale-offset repacks"),
            );
            let [
                FilterSpec {
                    kind: FilterKind::ScaleOffset(_, fill),
                    ..
                },
            ] = db.chunk_options.filters[..]
            else {
                panic!(
                    "expected one scale-offset filter, got {:?}",
                    db.chunk_options.filters
                );
            };
            fill
        };

        assert_eq!(
            carried(ScaleOffsetFill::Undefined),
            FillAvailability::Undefined
        );
        assert_eq!(
            carried(ScaleOffsetFill::Defined(None)),
            FillAvailability::Defined
        );
    }

    /// A foreign pipeline carrying both lzf and deflate is refused with the
    /// repack error naming the pair, not the builder's combination error raised
    /// later from deeper down. The builder cannot produce such a file, hence a
    /// synthetic pipeline rather than a file-level test.
    #[test]
    fn lzf_plus_deflate_pipeline_is_refused() {
        use crate::filter_pipeline::FilterDescription;

        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_LZF,
                    name: Some("lzf".into()),
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE,
                    name: None,
                    flags: 0,
                    client_data: vec![6],
                },
            ],
        };
        let err = check_pipeline(Some(&pipeline), "d").unwrap_err();
        assert!(
            matches!(&err, Error::RepackUnsupported(msg) if msg.contains("lzf + deflate")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn repack_preserves_big_endian_time_dataset() {
        // The reference C library cannot create H5T_TIME, so this round-trips a
        // big-endian time dataset through our own writer and reader: repack must
        // preserve both the byte order (bf0 bit 0) and the raw element bytes.
        use crate::datatype::{Datatype, DatatypeByteOrder};
        use crate::reader::File;
        use crate::writer::FileBuilder;

        let dir = tempfile::tempdir().unwrap();
        let dir = dir.path();
        let src = dir.join("hdf5_pure_repack_time_src.h5");
        let dst = dir.join("hdf5_pure_repack_time_dst.h5");

        let dt = Datatype::Time {
            size: 4,
            byte_order: DatatypeByteOrder::BigEndian,
            bit_precision: 32,
        };
        let raw: Vec<u8> = vec![
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03,
        ];
        {
            let mut b = FileBuilder::new();
            b.create_dataset("t")
                .with_raw_data(dt.clone(), raw.clone(), 3)
                .with_shape(&[3]);
            b.write(&src).unwrap();
        }

        repack(&src, &dst, &RepackOptions::new()).unwrap();

        let f = File::open(&dst).unwrap();
        let ds = f.dataset("t").unwrap();
        assert_eq!(
            ds.datatype().unwrap(),
            dt,
            "time datatype incl. byte order must survive repack"
        );
        assert_eq!(
            ds.read_raw().unwrap(),
            raw,
            "time element bytes must be preserved"
        );

        std::fs::remove_file(&src).ok();
        std::fs::remove_file(&dst).ok();
    }

    #[test]
    fn is_dropped_matches_self_and_ancestors() {
        let drop: BTreeSet<String> = ["g/old", "lone"].iter().map(|s| s.to_string()).collect();
        // The dropped path itself.
        assert!(is_dropped("lone", &drop));
        assert!(is_dropped("g/old", &drop));
        // A descendant of a dropped group is dropped (the whole subtree goes).
        assert!(is_dropped("g/old/child", &drop));
        assert!(is_dropped("g/old/a/b", &drop));
        // Unrelated paths and partial-name collisions are not dropped.
        assert!(!is_dropped("g", &drop));
        assert!(!is_dropped("g/older", &drop));
        assert!(!is_dropped("lonely", &drop));
        assert!(!is_dropped("other/old", &drop));
    }
}

/// Repack must carry an attribute across *as it is encoded*, not as
/// [`AttrValue`] renders it.
///
/// These sit at the library level rather than in `tests/` because the encoding
/// is the thing under test and only [`Group::attr_messages`] exposes it; a
/// public read decodes, and a decode is exactly what erases the difference
/// (issue #241).
#[cfg(test)]
mod attribute_fidelity_tests {
    use super::*;
    use crate::dataspace::{Dataspace, DataspaceType};
    use crate::datatype::StringPadding;
    use crate::datatype::{CharacterSet, CompoundMember, DatatypeByteOrder, ReferenceType};
    use crate::{File, FileBuilder, RepackOptions};
    use std::collections::BTreeMap;

    fn i32_type() -> Datatype {
        Datatype::FixedPoint {
            size: 4,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        }
    }

    /// Attribute messages keyed by name, for an object reached by `path`
    /// (`""` for the root group).
    fn messages_at(path: &str, file: &Path) -> BTreeMap<String, AttributeMessage> {
        let f = File::open(file).unwrap();
        let messages = if path.is_empty() {
            f.root().attr_messages().unwrap()
        } else if let Ok(ds) = f.dataset(path) {
            ds.attr_messages().unwrap()
        } else {
            f.group(path).unwrap().attr_messages().unwrap()
        };
        messages.into_iter().map(|m| (m.name.clone(), m)).collect()
    }

    /// The core claim, stated as an identity rather than as a list of properties:
    /// every attribute whose bytes are position-independent comes out of a repack
    /// as the very message that went in.
    ///
    /// Asserting the whole message covers width, charset, string padding,
    /// dataspace kind and rank, and the element bytes together — including the
    /// ones no accessor on this crate would notice, since two messages that
    /// decode to the same value can still differ on disk. It also covers all
    /// three kinds of owner, because the root group, a subgroup and a dataset
    /// each reach the writer by a different path.
    #[test]
    fn a_position_independent_attribute_crosses_a_repack_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let (src, dst) = (dir.path().join("src.h5"), dir.path().join("dst.h5"));

        let attrs = || {
            [
                // The width cases. A decode keeps these now (#350), so what
                // they pin here is that the copy does too — the byte-order case
                // a decode still cannot express is in the test below.
                ("i32", AttrValue::I32(-7)),
                ("u32", AttrValue::U32(4_294_967_295)),
                // Charset and padding: an ASCII string and a UTF-8 one differ only
                // in the datatype's charset bit.
                ("ascii", AttrValue::AsciiString("m/s".into())),
                ("utf8", AttrValue::String("µm".into())),
                // A scalar and a one-element array are different dataspaces.
                ("one_elem", AttrValue::I64Array(vec![9])),
                ("scalar", AttrValue::I64(9)),
                ("f64s", AttrValue::F64Array(vec![1.5, -2.5])),
            ]
        };

        let mut b = FileBuilder::new();
        for (name, value) in attrs() {
            b.set_attr(name, value);
        }
        let ds = b.create_dataset("data").with_f64_data(&[1.0, 2.0]);
        for (name, value) in attrs() {
            ds.set_attr(name, value);
        }
        let mut g = b.create_group("grp");
        for (name, value) in attrs() {
            g.set_attr(name, value);
        }
        g.create_dataset("inner").with_i32_data(&[1]);
        b.add_group(g.finish());
        b.write(&src).unwrap();

        repack(&src, &dst, &RepackOptions::new()).unwrap();

        for owner in ["", "data", "grp"] {
            let before = messages_at(owner, &src);
            let after = messages_at(owner, &dst);
            assert_eq!(
                before.keys().collect::<Vec<_>>(),
                after.keys().collect::<Vec<_>>(),
                "repack changed which attributes {owner:?} has"
            );
            for (name, source_message) in &before {
                assert_eq!(
                    after.get(name),
                    Some(source_message),
                    "attribute {name:?} on {owner:?} was re-encoded rather than copied"
                );
            }
        }
    }

    /// The two rows of issue #241's table that the reference C library's own API
    /// cannot express, so no crosscheck can state them: a fixed-width string's
    /// *padding*, and a Null dataspace.
    ///
    /// The source here is built through the same verbatim seam the fix
    /// introduced, which is the only way this crate can write these encodings —
    /// but the file is written, closed and re-read, so what is compared is what
    /// two independent parses of two real files made of them, not a struct
    /// handed back to itself.
    #[test]
    fn an_encoding_this_crate_has_no_attr_value_for_still_crosses_a_repack() {
        let dir = tempfile::tempdir().unwrap();
        let (src, dst) = (dir.path().join("src.h5"), dir.path().join("dst.h5"));

        let exotic = [
            // Declared width 16 for three bytes of content, null-*terminated* —
            // the writer's own fixed-width path hardcodes null-padding, so a
            // rebuild from a value changes both.
            AttributeMessage {
                name: "units".into(),
                datatype: Datatype::String {
                    size: 16,
                    padding: StringPadding::NullTerminate,
                    charset: CharacterSet::Ascii,
                },
                dataspace: Dataspace {
                    space_type: DataspaceType::Scalar,
                    rank: 0,
                    dimensions: vec![],
                    max_dimensions: None,
                },
                raw_data: {
                    let mut v = b"m/s".to_vec();
                    v.resize(16, 0);
                    v
                },
                datatype_location: crate::shared_message::DatatypeLocation::Inline,
            },
            // Space-padded, to prove the padding field is carried rather than
            // normalized to whichever value happens to be first.
            AttributeMessage {
                name: "spaced".into(),
                datatype: Datatype::String {
                    size: 8,
                    padding: StringPadding::SpacePad,
                    charset: CharacterSet::Utf8,
                },
                dataspace: Dataspace {
                    space_type: DataspaceType::Scalar,
                    rank: 0,
                    dimensions: vec![],
                    max_dimensions: None,
                },
                raw_data: b"ab      ".to_vec(),
                datatype_location: crate::shared_message::DatatypeLocation::Inline,
            },
            // A Null dataspace holds no elements at all, which is distinct from a
            // rank-1 dataspace of length zero — the shape a decode gives it.
            AttributeMessage {
                name: "nothing".into(),
                datatype: i32_type(),
                dataspace: Dataspace {
                    space_type: DataspaceType::Null,
                    rank: 0,
                    dimensions: vec![],
                    max_dimensions: None,
                },
                raw_data: vec![],
                datatype_location: crate::shared_message::DatatypeLocation::Inline,
            },
            // Big-endian, which every `AttrValue` integer variant normalizes to
            // little-endian: the width survives a decode since #350, the layout
            // of the bytes inside it does not.
            AttributeMessage {
                name: "big_endian".into(),
                datatype: Datatype::FixedPoint {
                    size: 4,
                    byte_order: crate::datatype::DatatypeByteOrder::BigEndian,
                    signed: true,
                    bit_offset: 0,
                    bit_precision: 32,
                },
                dataspace: Dataspace {
                    space_type: DataspaceType::Scalar,
                    rank: 0,
                    dimensions: vec![],
                    max_dimensions: None,
                },
                raw_data: (-7i32).to_be_bytes().to_vec(),
                datatype_location: crate::shared_message::DatatypeLocation::Inline,
            },
            // Rank 2, which every `AttrValue` array variant flattens.
            AttributeMessage {
                name: "grid".into(),
                datatype: i32_type(),
                dataspace: Dataspace {
                    space_type: DataspaceType::Simple,
                    rank: 2,
                    dimensions: vec![2, 3],
                    max_dimensions: None,
                },
                raw_data: (1i32..=6).flat_map(i32::to_le_bytes).collect(),
                datatype_location: crate::shared_message::DatatypeLocation::Inline,
            },
        ];

        let mut b = FileBuilder::new();
        for message in &exotic {
            b.set_attr_verbatim(message.clone());
        }
        let ds = b.create_dataset("data").with_f64_data(&[1.0]);
        for message in &exotic {
            ds.set_attr_verbatim(message.clone());
        }
        b.write(&src).unwrap();

        repack(&src, &dst, &RepackOptions::new()).unwrap();

        for owner in ["", "data"] {
            let before = messages_at(owner, &src);
            let after = messages_at(owner, &dst);
            for message in &exotic {
                let name = &message.name;
                // The source file must really hold what was asked for, or the
                // comparison below would only prove repack is self-consistent.
                assert_eq!(
                    before.get(name),
                    Some(message),
                    "the source file did not record {name:?} as written"
                );
                assert_eq!(
                    after.get(name),
                    Some(message),
                    "attribute {name:?} on {owner:?} changed across the repack"
                );
            }
        }
        assert_fixture_holds(&src, &exotic);
    }

    /// `tests/data/pure/exotic_attributes.h5` is the source file of the
    /// test above, committed so that `crates/crosscheck/tests/repack.rs` can
    /// repack it through the public API and read the result with the reference
    /// C library. Without that the test proves only that this crate agrees with
    /// itself, which a message it encodes wrongly and parses back just as
    /// wrongly would satisfy. The seam that writes these encodings is
    /// crate-private, so the fixture cannot be built over there.
    ///
    /// Not pinned byte for byte, since the encodings are what matter: the
    /// committed copy has to hold them, and `HDF5_PURE_UPDATE_TEST_DATA` rewrites
    /// it.
    fn assert_fixture_holds(src: &Path, exotic: &[AttributeMessage]) {
        let fixture = crate::test_data::path("pure/exotic_attributes.h5");
        if std::env::var_os(crate::test_data::UPDATE).is_some() {
            std::fs::create_dir_all(fixture.parent().unwrap()).unwrap();
            std::fs::copy(src, &fixture).unwrap();
        }
        for owner in ["", "data"] {
            let messages = messages_at(owner, &fixture);
            for message in exotic {
                assert_eq!(
                    messages.get(&message.name),
                    Some(message),
                    "{} does not hold {:?} on {owner:?} as this test writes it; \
                     rerun with {}=1 to refresh it",
                    fixture.display(),
                    message.name,
                    crate::test_data::UPDATE
                );
            }
        }
    }

    /// The other half of the rule: an attribute whose bytes are a *location* must
    /// not be copied, because the location is in the source file.
    ///
    /// A variable-length string attribute stores a global-heap collection address,
    /// so the repacked message is required to differ in exactly that way — same
    /// datatype and dataspace, different raw bytes — while the value it resolves
    /// to stays put. Asserting the raw bytes moved is what distinguishes a correct
    /// re-encode from a verbatim copy that happens to still resolve because the
    /// destination heap landed at the same address.
    #[test]
    fn an_attribute_addressing_the_heap_is_re_encoded_not_copied() {
        let dir = tempfile::tempdir().unwrap();
        let (src, dst) = (dir.path().join("src.h5"), dir.path().join("dst.h5"));
        let fields: Vec<String> = vec!["x".into(), "y".into(), "velocity".into()];

        let mut b = FileBuilder::new();
        // Bulk that only the source carries, so the destination's heap cannot
        // land at the source's address by coincidence.
        b.create_dataset("bulk").with_f64_data(&vec![0.0; 4096]);
        b.set_attr("fields", AttrValue::VarLenAsciiCharArray(fields.clone()));
        b.write(&src).unwrap();

        repack(&src, &dst, &RepackOptions::new().drop_path("bulk")).unwrap();

        let before = &messages_at("", &src)["fields"];
        let after = &messages_at("", &dst)["fields"];
        assert_eq!(
            after.datatype, before.datatype,
            "the attribute must stay variable-length"
        );
        assert_eq!(after.dataspace, before.dataspace);
        assert_ne!(
            after.raw_data, before.raw_data,
            "a heap reference copied verbatim would point into the source file"
        );
        assert_eq!(
            File::open(&dst).unwrap().root().attrs().unwrap()["fields"],
            AttrValue::VarLenAsciiCharArray(fields),
            "and it must still resolve to its own strings"
        );
    }

    /// A datatype that merely *contains* a heap reference or an address is
    /// position-dependent too — the address sits inside each element rather than
    /// being the whole of it, and copying the element copies the address.
    #[test]
    fn a_nested_address_makes_the_whole_datatype_position_dependent() {
        let vlen = Datatype::VariableLength {
            is_string: true,
            padding: None,
            charset: Some(CharacterSet::Ascii),
            base_type: Box::new(i32_type()),
        };
        let reference = Datatype::Reference {
            size: 8,
            ref_type: ReferenceType::Object,
        };
        let compound_of = |member: Datatype| Datatype::Compound {
            size: 24,
            members: vec![
                CompoundMember {
                    name: "plain".into(),
                    byte_offset: 0,
                    datatype: i32_type(),
                },
                CompoundMember {
                    name: "nested".into(),
                    byte_offset: 8,
                    datatype: member,
                },
            ],
        };
        let array_of = |base: Datatype| Datatype::Array {
            base_type: Box::new(base),
            dimensions: vec![2, 3],
        };

        for dependent in [
            vlen.clone(),
            reference.clone(),
            compound_of(vlen.clone()),
            compound_of(reference.clone()),
            array_of(vlen.clone()),
            array_of(reference.clone()),
            // Two levels down, which a non-recursive check would let through.
            array_of(compound_of(vlen)),
            compound_of(array_of(reference)),
        ] {
            assert!(
                !attr_bytes_are_position_independent(&dependent),
                "{dependent:?} holds an address and must not be copied verbatim"
            );
        }

        for independent in [
            i32_type(),
            Datatype::String {
                size: 8,
                padding: crate::datatype::StringPadding::NullPad,
                charset: CharacterSet::Utf8,
            },
            compound_of(i32_type()),
            array_of(i32_type()),
            Datatype::Enumeration {
                size: 4,
                base_type: Box::new(i32_type()),
                members: vec![],
            },
        ] {
            assert!(
                attr_bytes_are_position_independent(&independent),
                "{independent:?} holds no address and can be copied verbatim"
            );
        }
    }
}
