//! Immediate chunk appends and staged append inputs.
//!
//! The editor checks append policy and caches dataset metadata here.
//! [`chunk_index_inplace`] writes chunk data, updates the index, and publishes the extent.

use core::num::NonZeroU64;
use core::num::NonZeroUsize;

use hdf5_pure_format::__private::ChunkedGeometry;

use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::{FreeList, PageTransition, PageType, SessionSpace};

use crate::SuperblockVersion;
use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunk_index_inplace::{self, Located, Store};
use crate::convert::Narrow;
use crate::data_layout::DataLayout;
use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::error::{Error, FormatError};
use crate::filter_pipeline::FilterPipeline;
use crate::image::FileImage;
use crate::message_type::MessageType;
use crate::object_path::ObjectPathBuf;
use crate::superblock::Superblock;
use crate::type_builders;
use crate::width::FormatWidths;

use super::LOSSY_TAIL_REFUSAL;
use super::staging::StagedOp;

use super::{AggregateSpace, Placement, SyncPolicy, WriteEngine};

/// Accumulates elements to append to an existing chunked, unlimited dataset via
/// [`Dataset::append_staged`](crate::Dataset::append_staged), in call order along the dataset's first
/// (axis-0) dimension.
///
/// It mirrors [`DatasetBuilder`](crate::DatasetBuilder)'s typed/generic vocabulary. Repeated typed or
/// [`append_raw`](Self::append_raw) calls concatenate, each typed method also
/// records the element datatype it implies, which `commit` checks against the
/// dataset's on-disk datatype (a mismatch  -  including a mix of element types in
/// one builder  -  is rejected with [`Error::AppendUnsupported`], never written as
/// garbage).
pub struct AppendBuilder {
    /// Accumulated little-endian element bytes to append, in call order.
    pub(super) raw: Vec<u8>,
    /// The element datatype implied by the typed `append_*` calls, if any were
    /// used. `None` when only [`append_raw`](Self::append_raw) was called (a raw
    /// append is checked structurally  -  element-size alignment and little-endian
    /// on-disk order  -  in place of by datatype equality).
    pub(super) elem_dt: Option<Datatype>,
    /// Set when two typed calls implied different element datatypes, `commit`
    /// rejects such a builder in place of write a mix of encodings.
    pub(super) dt_conflict: bool,
}

impl AppendBuilder {
    pub(crate) fn new() -> Self {
        Self {
            raw: Vec::new(),
            elem_dt: None,
            dt_conflict: false,
        }
    }

    /// Accumulated little-endian element bytes (for the general append writer,
    /// which reuses this builder to gather typed/generic appends).
    pub(crate) fn raw(&self) -> &[u8] {
        &self.raw
    }

    /// The element datatype implied by typed appends, if any.
    pub(crate) fn elem_dt(&self) -> Option<&Datatype> {
        self.elem_dt.as_ref()
    }

    /// Whether two typed appends implied conflicting element datatypes.
    pub(crate) fn dt_conflict(&self) -> bool {
        self.dt_conflict
    }

    /// A builder holding a copy of this one's first `byte_len` bytes, carrying
    /// the same element datatype so the prefix is type-checked exactly as the
    /// whole would have been. Used by [`BufferedAppender`](crate::BufferedAppender)
    /// to write out the chunk-aligned prefix of its buffer, it copies the prefix so a failed write leaves the buffer intact and the appender
    /// can report precisely which elements did not land.
    pub(crate) fn head(&self, byte_len: usize) -> Self {
        Self {
            raw: self.raw[..byte_len.min(self.raw.len())].to_vec(),
            elem_dt: self.elem_dt.clone(),
            dt_conflict: self.dt_conflict,
        }
    }

    /// Discard the first `byte_len` buffered bytes (a prefix that reached the
    /// file), keeping the element datatype.
    pub(crate) fn drop_front(&mut self, byte_len: usize) {
        self.raw.drain(..byte_len.min(self.raw.len()));
    }

    /// Consume the builder, yielding its accumulated element bytes. Used by
    /// [`BufferedAppender::discard`](crate::BufferedAppender::discard) to hand
    /// abandoned elements back to the caller.
    pub(crate) fn into_raw(self) -> Vec<u8> {
        self.raw
    }

    /// Cut the buffer back to `byte_len` bytes, keeping the element datatype.
    /// [`BufferedAppender`](crate::BufferedAppender) uses this to undo a call
    /// whose write was rejected before it touched the file, so the rejection leaves
    /// nothing buffered and a retry cannot append the same elements twice.
    pub(crate) fn truncate(&mut self, byte_len: usize) {
        self.raw.truncate(byte_len);
    }

    /// Record the datatype a typed append implies, flagging a conflict if an
    /// earlier typed call implied a different one.
    fn set_dt(&mut self, dt: Datatype) {
        match &self.elem_dt {
            Some(prev) if *prev != dt => self.dt_conflict = true,
            Some(_) => {}
            None => self.elem_dt = Some(dt),
        }
    }

    /// Append already-little-endian element bytes verbatim. The concatenated
    /// length must be a whole multiple of the dataset's on-disk element size, and
    /// the dataset's element datatype must be little-endian, no datatype is
    /// otherwise inferred. Prefer the typed methods when the element type is known.
    pub fn append_raw(&mut self, bytes: &[u8]) -> &mut Self {
        self.raw.extend_from_slice(bytes);
        self
    }

    /// Generic append of a flat slice of any supported scalar type  -  the
    /// counterpart of [`DatasetBuilder::with_data`](crate::DatasetBuilder::with_data).
    pub fn append<T: crate::element::H5Element>(&mut self, data: &[T]) -> &mut Self {
        T::append_into(self, data);
        self
    }
}

/// Generate the typed `append_*` methods: serialize each value little-endian and
/// record the implied element datatype.
macro_rules! append_typed {
    ($($method:ident, $ty:ty, $make:path;)*) => {
        impl AppendBuilder {
            $(
                #[doc = concat!("Append `", stringify!($ty), "` values to the dataset.")]
                pub fn $method(&mut self, data: &[$ty]) -> &mut Self {
                    self.set_dt($make());
                    // Reserved up front: serializing element by element into a
                    // growing `Vec` re-allocated it ten times per call and copied
                    // the batch twice over, which for an append loop is the whole
                    // per-call cost (issue #228).
                    self.raw.reserve(data.len() * core::mem::size_of::<$ty>());
                    for &v in data {
                        self.raw.extend_from_slice(&v.to_le_bytes());
                    }
                    self
                }
            )*
        }
    };
}

append_typed! {
    append_f64, f64, type_builders::make_f64_type;
    append_f32, f32, type_builders::make_f32_type;
    append_i8, i8, type_builders::make_i8_type;
    append_i16, i16, type_builders::make_i16_type;
    append_i32, i32, type_builders::make_i32_type;
    append_i64, i64, type_builders::make_i64_type;
    append_u8, u8, type_builders::make_u8_type;
    append_u16, u16, type_builders::make_u16_type;
    append_u32, u32, type_builders::make_u32_type;
    append_u64, u64, type_builders::make_u64_type;
}

/// What an in-place append identifies its target by.
///
/// A handle with a resolvable path uses it, which lets the session compare the
/// target against its own staged edits. A handle reached by object reference has
/// no path, so it identifies the dataset by the object-header address it was reached
/// through  -  the same key the geometry cache uses.
#[derive(Clone, Copy)]
pub(crate) enum AppendTarget<'a> {
    Path(&'a ObjectPathBuf),
    Header(u64),
}

/// Byte budget for one append batch on a session that batches: a large append is
/// split into whole-chunk batches of at most this many raw bytes (always at least
/// one chunk), each applied as its own crash-atomic fsync-barriered sequence, so
/// peak append memory never scales with the caller's slice.
pub(super) const APPEND_BATCH_BYTES: u64 = 1 << 20;

/// One dataset's append geometry, handed to the public append path so it can
/// slice a large call into aligned batches without materializing the whole
/// call's bytes first.
pub(crate) struct AppendGeometry {
    /// Elements per chunk along axis 0 (>= 1).
    pub(crate) chunk_elems: u64,
    /// Bytes per on-disk element, proven non-zero.
    pub(crate) element_size: NonZeroUsize,
    /// Current length along the unlimited dimension.
    pub(crate) current_dim: u64,
    /// Whether this dataset's filter pipeline is **lossy**  -  ZFP, or float
    /// D-scale scale-offset. Such a dataset cannot have a partial trailing chunk
    /// grown, because that decodes and re-encodes already-committed values (see
    /// [`super::pipeline_lossless`]), so an append onto one has to start chunk-aligned.
    pub(crate) lossy_filters: bool,
    /// Whole-chunk elements in one full batch (>= one chunk's worth), or
    /// [`u64::MAX`] when the session does not batch.
    pub(crate) full_batch_elems: u64,
}

/// A dataset located once for [`Dataset::append`](crate::Dataset::append) (or the bounded
/// backend's immediate append), then maintained across appends. Mirrors the
/// append writer's per-dataset state.
pub(crate) struct LocatedState {
    pub(crate) loc: Located,
    /// The dataset's on-disk element datatype (for the append type check and the
    /// filter chunk context).
    pub(crate) datatype: Datatype,
    /// Spatial (rank-length) chunk dimensions in elements: `[chunk_elems]`.
    pub(crate) spatial: Vec<u64>,
    /// Bytes per element (datatype size), proven non-zero.
    pub(crate) element_size: NonZeroUsize,
    /// The re-encodable filter pipeline, when the dataset is filtered.
    pub(crate) pipeline: Option<FilterPipeline>,
    /// What the slots past the new dimension must hold when an append completes
    /// a partial chunk (issue #296).
    pub(crate) fill: crate::fill_value::PaddingFill,
}

/// Provides the extensible-array append engine with the session's image and superblock.
///
/// The engine borrows its image through [`Store`] and maintains its dataset geometry separately
/// in [`WriteEngine::located`]. The adapter tries reusable space before classic aggregates,
/// allocating chunk payloads from the raw aggregate and index blocks from the metadata aggregate.
/// On PAGE files, both use raw pages, with [`SessionSpace::plan_transition`] before extending the
/// image.
pub(super) struct EditStore<'a> {
    pub(super) image: &'a mut dyn FileImage,
    pub(super) superblock: &'a mut Superblock,
    pub(super) widths: FormatWidths,
    pub(super) sb_sig_off: u64,
    /// The mapping between the stored addresses in `space` and the positions in `image`.
    pub(super) addresses: AddressContext,
    /// The session allocation strategy. PAGE transition state recorded here reaches the manager
    /// rewrite at the next commit or close.
    pub(super) space: &'a mut SessionSpace,
    /// Classic aggregate windows shared with staged edits under [`crate::FileSpaceStrategy::Aggr`] or
    /// [`crate::FileSpaceStrategy::FsmAggr`].
    pub(super) aggregation: Option<&'a mut AggregateSpace>,
    /// Published-manager space reserved for an immediate append on a persisting session.
    pub(super) reserve: Option<&'a mut FreeList>,
    /// Whether a non-persisting session may allocate directly from `space`.
    pub(super) reuse_session_space: bool,
    /// The session's `fsync` cadence, carried by value: the append engine's own
    /// ordered barriers ([`chunk_index_inplace::apply_ea_append`]) are durability points like the
    /// commit's, and result to the same policy.
    pub(super) sync_policy: SyncPolicy,
}

impl EditStore<'_> {
    /// Appends `bytes`, padding a metadata tail page before writing into raw PAGE space.
    ///
    /// Extensible-array index blocks use the same raw-page placement as chunk payloads.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if address conversion, PAGE geometry, or padding size is invalid.
    /// Returns [`Error::Io`] if appending the padding or bytes fails.
    fn append_into_raw_page(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        let eoa = self.addresses.relative(self.image.len())?.get();
        if let Some(transition) = self.space.plan_transition(eoa, PageType::Raw)? {
            if let PageTransition::Pad { extent, .. } = transition {
                self.image.append(&vec![0u8; extent.len().to_usize()?])?;
            }
            self.space.record_transition(PageType::Raw, transition);
        }
        self.image.append(bytes)
    }
}

impl crate::source::Source for EditStore<'_> {
    fn len(&self) -> u64 {
        self.image.len()
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), crate::error::FormatError> {
        self.image.read_at(offset, buf)
    }
    fn read_metadata_at(
        &self,
        offset: u64,
        len: usize,
    ) -> Result<Vec<u8>, crate::error::FormatError> {
        self.image.read_metadata_at(offset, len)
    }

    // Forwarded because this is a `Source` wrapper over the image and the
    // trait requires it, not because anything requests yet: `File`'s own accessor
    // reaches the image directly. A wrapper that returned `None` for a live
    // cache is the failure that rule exists to prevent.
    fn metadata_cache_stats(&self) -> Option<crate::source::MetadataCacheStats> {
        self.image.metadata_cache_stats()
    }

    fn reset_metadata_cache_stats(&self) {
        self.image.reset_metadata_cache_stats();
    }
}

impl Store for EditStore<'_> {
    fn widths(&self) -> FormatWidths {
        self.widths
    }
    fn recorded_eof(&self) -> u64 {
        self.superblock.eof_address
    }
    fn allocate(&mut self, bytes: &[u8], class: AllocationClass) -> Result<u64, Error> {
        let extent = if let Some(reserve) = self.reserve.as_deref_mut() {
            reserve.alloc(bytes.len() as u64)
        } else if self.reuse_session_space {
            self.space
                .allocate_with_page_type(bytes.len() as u64, class, PageType::Raw)
        } else {
            None
        };
        if let Some(extent) = extent {
            let at = self.addresses.resolve(StoredAddress::new(extent.start()))?;
            self.image.write_at(at.get(), bytes)?;
            return Ok(at.get());
        }
        if let (Some(aggregation), Some(size)) = (
            self.aggregation.as_deref_mut(),
            NonZeroU64::new(bytes.len() as u64),
        ) {
            let Placement::Aggregated { addr, end, .. } =
                aggregation.reserve(self.addresses, self.space, size, class)?
            else {
                return Err(FormatError::Internal(
                    "aggregate reservation is not aggregated".into(),
                )
                .into());
            };
            AggregateSpace::place(self.image, addr, bytes, end)?;
            return Ok(addr);
        }
        self.append_into_raw_page(bytes)
    }
    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        self.image.write_at(offset, bytes)
    }
    fn patch_superblock_eof(&mut self) -> Result<(), Error> {
        // Advance only the recorded end-of-file and re-serialize the superblock in
        // place. Unlike `WriteEngine::commit`, this leaves the consistency flags and stored root
        // group address unchanged.
        let mut superblock = self.superblock.clone();
        superblock.eof_address = self.image.len();
        let bytes = hdf5_pure_format::__private::serialize_superblock(&superblock)?;
        self.write_at(self.sb_sig_off, &bytes)?;
        *self.superblock = superblock;
        Ok(())
    }
    fn sync(&mut self) -> Result<(), Error> {
        super::barrier_data(self.image, self.sync_policy)
    }
}

/// Re-tags a rejection from the shared append engine (`AppendUnsupported`) as the fast-path
/// [`Error::AppendInPlaceUnsupported`], which a caller catches to fall back to the staged
/// [`stage_dataset_append`](WriteEngine::stage_dataset_append). That path covers the filtered
/// partial-trailing-chunk case and index-geometry limits. A platform-width limit applies on both
/// paths: the engine reports it as `AppendUnsupported`, and the staged path as
/// [`FormatError::ValueTooLargeForPlatform`]. I/O and format errors pass through unchanged.
pub(crate) fn as_inplace_error(e: Error) -> Error {
    match e {
        Error::AppendUnsupported(m) => Error::AppendInPlaceUnsupported(m),
        other => other,
    }
}

/// Validate a gathered append's bytes against a located dataset: the byte
/// length must be a whole number of elements, and the element datatype must
/// match the on-disk datatype (or, for a raw append, be raw-appendable).
/// Returns the appended element count (`0` = nothing to do). Shared by
/// [`WriteEngine::append_inplace_gathered`]'s path and the bounded backend's immediate
/// append so the acceptance rules stay identical.
pub(crate) fn validate_gathered_append(st: &LocatedState, b: &AppendBuilder) -> Result<u64, Error> {
    let raw = b.raw();
    if raw.len() % st.element_size != 0 {
        return Err(Error::AppendInPlaceUnsupported(
            "appended byte length is not a whole number of elements",
        ));
    }
    match b.elem_dt() {
        Some(expected) if *expected != st.datatype => {
            return Err(Error::AppendInPlaceUnsupported(
                "append datatype does not match the on-disk dataset (wrong element \
                 type or byte order)",
            ));
        }
        Some(_) => {}
        None => {
            if !super::datatype_is_raw_appendable(&st.datatype) {
                return Err(Error::AppendInPlaceUnsupported(
                    "append_raw onto this dataset's datatype (non-little-endian, \
                     variable-length, or reference) could misencode the bytes; use a \
                     typed append",
                ));
            }
        }
    }
    Ok((raw.len() / st.element_size) as u64)
}

/// Locates the dataset at `oh_addr` in `file` and builds its [`LocatedState`] for an in-place
/// append.
///
/// An in-place append accepts a dataset of rank 1 with an unlimited dimension, indexed by an
/// Extensible Array with a nonzero chunk length, whose filter pipeline the engine can encode again.
///
/// # Errors
///
/// Returns [`Error::AppendInPlaceUnsupported`] if an in-place append does not accept the dataset,
/// and [`FormatError::ChunkElementSizeMismatch`] if the layout stores an element size other than
/// the size of the datatype. Returns the error of a read or a parse of the dataset's metadata.
pub(crate) fn locate_dataset_state<F: Store>(
    file: &F,
    oh_addr: u64,
) -> Result<LocatedState, Error> {
    let result = Located::locate_at(file, oh_addr, Error::AppendInPlaceUnsupported)?;
    if result.located.chunk_elems == 0 {
        return Err(Error::AppendInPlaceUnsupported(
            "in-place append requires a nonzero chunk length",
        ));
    }
    let (dt_off, dt_size) = result.spans.datatype;
    let dt_bytes = file.read_metadata_at(dt_off, dt_size)?;
    let (datatype, _) = hdf5_pure_format::__private::parse_datatype(&dt_bytes)?;
    let element_size: NonZeroUsize = datatype.element_size()?.narrow()?;
    if element_size != result.located.elem_bytes {
        return Err(Error::Format(FormatError::ChunkElementSizeMismatch {
            datatype: element_size.get(),
            layout: result.located.elem_bytes.get(),
        }));
    }
    let pipeline = match result.spans.filter {
        Some((fb, fsize)) => {
            let fp_bytes = file.read_metadata_at(fb, fsize)?;
            let parsed = crate::filter_pipeline::parse_filter_pipeline(&fp_bytes)?;
            if !super::pipeline_reencodable(&parsed) {
                return Err(Error::AppendInPlaceUnsupported(
                    "dataset uses a filter this engine cannot re-encode",
                ));
            }
            Some(parsed)
        }
        None => None,
    };
    let spatial = vec![result.located.chunk_elems];
    // A fill message that cannot be read leaves `PaddingFill::Unknown`, which
    // fails only where a chunk actually needs padding  -  an append that lands on
    // a chunk boundary requests nothing of it.
    let fill = match result.spans.fill {
        Some((msg_type, off, size)) => match file.read_metadata_at(off, size) {
            Ok(body) => crate::fill_value::PaddingFill::from_message(msg_type, &body),
            Err(_) => crate::fill_value::PaddingFill::Unknown,
        },
        None => crate::fill_value::PaddingFill::Zero,
    };
    Ok(LocatedState {
        loc: result.located,
        datatype,
        spatial,
        element_size,
        pipeline,
        fill,
    })
}

impl WriteEngine {
    /// Stages an append to a committed dataset for the next [`commit`](Self::commit).
    ///
    /// The commit checks the dataset's append geometry and storage. A lossy partial trailing
    /// chunk is rejected at staging when its geometry can be read.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a dataset creation is staged at `path` or a live buffered
    /// appender overlaps it. Returns [`Error::AppendUnsupported`] if a nonempty append would
    /// re-encode a partial trailing chunk through a lossy pipeline.
    pub(crate) fn stage_dataset_append(
        &mut self,
        path: &ObjectPathBuf,
        builder: AppendBuilder,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        if self.staged.dataset_at(path).is_some() {
            // This handle addresses the committed object being replaced at this path.
            return Err(Error::EditUnsupported(
                "a dataset is staged at this path in the same commit, so an append through a \
                 handle onto the object the file holds there could only grow the object being \
                 replaced; append through the handle that staged the creation, or commit first",
            ));
        }
        self.refuse_lossy_partial_tail(path, &builder)?;
        self.staged
            .push(StagedOp::AppendDataset(path.clone(), builder));
        Ok(())
    }

    /// Rejects an append that would re-encode a lossy partial trailing chunk.
    ///
    /// Empty appends are accepted. If the committed geometry cannot be read, its validation
    /// is deferred to [`prepare_append`](Self::prepare_append).
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] if the committed dataset is rank one, its
    /// pipeline is lossy, and its length is not a multiple of the chunk length.
    pub(super) fn refuse_lossy_partial_tail(
        &self,
        path: &ObjectPathBuf,
        builder: &AppendBuilder,
    ) -> Result<(), Error> {
        if builder.raw().is_empty() {
            return Ok(());
        }
        if self.appends_onto_a_lossy_partial_tail(path) {
            return Err(Error::AppendUnsupported(LOSSY_TAIL_REFUSAL));
        }
        Ok(())
    }

    /// Returns whether readable rank-one geometry requires a lossy trailing-chunk rewrite.
    ///
    /// Returns `false` if any required metadata cannot be resolved or parsed.
    pub(super) fn appends_onto_a_lossy_partial_tail(&self, path: &ObjectPathBuf) -> bool {
        let Ok(addr) = crate::group_v2::resolve_path_any_from_source(
            &self.image(),
            AccessMode::ReadWrite,
            &self.superblock,
            self.widths,
            self.addresses,
            &path.as_path(),
        ) else {
            return false;
        };
        let Ok(region) = Self::gather_oh_messages(&self.image(), addr, self.addresses) else {
            return false;
        };
        let mut datatype: Option<(usize, usize)> = None;
        let mut dataspace: Option<(usize, usize)> = None;
        let mut layout: Option<(usize, usize)> = None;
        let mut filter: Option<(usize, usize)> = None;
        let mut p = 0;
        while let Ok(Some((msg_type, body, body_end))) = region.next_message(p) {
            match msg_type {
                MessageType::DATATYPE => datatype = Some((body, body_end)),
                MessageType::DATASPACE => dataspace = Some((body, body_end)),
                MessageType::DATA_LAYOUT => layout = Some((body, body_end)),
                MessageType::FILTER_PIPELINE => filter = Some((body, body_end)),
                _ => {}
            }
            p = body_end;
        }
        let (Some((fb, fe)), Some((dt_b, dt_e)), Some((ds_b, ds_e)), Some((lb, le))) =
            (filter, datatype, dataspace, layout)
        else {
            return false;
        };
        let Ok(pipeline) = crate::filter_pipeline::parse_filter_pipeline(&region[fb..fe]) else {
            return false;
        };
        if super::pipeline_lossless(&pipeline) {
            return false;
        }
        let Ok((disk_dt, _)) = hdf5_pure_format::__private::parse_datatype(&region[dt_b..dt_e])
        else {
            return false;
        };
        let Ok(disk_ds) =
            Dataspace::parse_with_length_width(&region[ds_b..ds_e], self.widths.lengths)
        else {
            return false;
        };
        let Ok(dl) = DataLayout::parse_with_widths(&region[lb..le], self.widths) else {
            return false;
        };
        if disk_ds.dimensions.len() != 1 {
            return false;
        }
        let Ok(geometry) = ChunkedGeometry::new(&disk_dt, &disk_ds, &dl) else {
            return false;
        };
        match (geometry.spatial.first(), disk_ds.dimensions.first()) {
            (Some(&chunk_elems), Some(&dim0)) if chunk_elems != 0 => dim0 % chunk_elems != 0,
            _ => false,
        }
    }

    /// The shared Extensible-Array append engine's view of this session: the
    /// image, the superblock, and the paged-file state, paired as [`EditStore`].
    ///
    /// Takes `&mut self`, so a caller that also needs [`located`](Self::located)
    /// borrowed at the same time must destructure the fields itself in place of
    /// call this.
    pub(super) fn store(&mut self) -> EditStore<'_> {
        EditStore {
            image: self.image.as_mut(),
            superblock: &mut self.superblock,
            widths: self.widths,
            sb_sig_off: self.sb_sig_off,
            addresses: self.addresses,
            space: &mut self.space,
            aggregation: self.aggregation.as_mut(),
            reserve: None,
            reuse_session_space: false,
            sync_policy: self.sync_policy,
        }
    }

    /// Resolve and locate an in-place append target, applying every rule that
    /// does not depend on the bytes being appended: the file-level eligibility
    /// guards, the staged-edit conflict check, and the geometry lookup that
    /// populates [`located`](Self::located). Returns the dataset's object-header
    /// address, which is that cache's key.
    ///
    /// Split out from the append itself so [`append_geometry`](Self::append_geometry)
    /// can report a dataset's batching geometry under exactly the same rules the
    /// append will apply  -  a caller slicing a large append into batches must be
    /// rejected before the first batch, not part-way through.
    pub(super) fn append_prepare(&mut self, target: AppendTarget<'_>) -> Result<u64, Error> {
        // The fast in-place append is only sound on a base-0 latest-format file:
        // the slot math assumes absolute addresses and the superblock is patched in
        // place per call. A userblock or pre-v2 file falls back to the staged
        // `append_dataset`, which rebuilds the index and repoints the superblock
        // last.
        if !self.superblock.base_address.is_zero() {
            return Err(Error::AppendInPlaceUnsupported(
                "in-place append does not support a file with a userblock (non-zero base \
                 address); use Dataset::append_staged",
            ));
        }
        if matches!(
            self.superblock.version,
            SuperblockVersion::Zero | SuperblockVersion::One
        ) {
            return Err(Error::AppendInPlaceUnsupported(
                "in-place append requires a latest-format file (v2/v3 superblock); use \
                 Dataset::append_staged",
            ));
        }
        // Non-persistent PAGE editing supports only staged allocation in fresh typed pages.
        // Persistent PAGE appends use `EditStore` and rewrite managers at commit or close.
        if self.space.is_paged() && self.persist.is_none() {
            return Err(Error::AppendInPlaceUnsupported(
                "in-place append is not supported on a paged file \
                 (H5F_FSPACE_STRATEGY_PAGE) without persisted free space; recreate the \
                 file with with_file_space_strategy(FileSpaceStrategy::Page, true, ..)",
            ));
        }

        // Reject an append against a dataset (or a subtree) that a still-staged edit
        // in this same session will relocate, replace, or delete  -  which would
        // strand the durably-appended rows or plan against a header the commit
        // moves. The caller must commit those edits first.
        //
        // A target named by object-header address cannot be compared against the
        // staged paths, so any staged edit at all disqualifies it. That is a
        // extension of the path check, and the remedy is the same one.
        match target {
            AppendTarget::Path(dataset) => {
                if self.append_conflicts_with_pending(dataset) {
                    return Err(Error::AppendInPlaceUnsupported(
                        "the dataset or an ancestor has a staged edit pending in this session; \
                         commit the staged edits before appending in place, or use \
                         Dataset::append_staged",
                    ));
                }
            }
            AppendTarget::Header(_) if self.has_staged_edits() || self.committed => {
                return Err(Error::AppendInPlaceUnsupported(
                    "this append target was reached by object reference, so it names a dataset \
                     by object-header address, and this session has staged or committed edits \
                     that can move that header; re-open the dataset by path to append to it",
                ));
            }
            AppendTarget::Header(_) => {}
        }

        // Resolve the dataset's object-header address  -  the geometry cache key.
        // base == 0 here, so a resolved address is absolute, two hard links to
        // one dataset share the one entry.
        let oh_addr = match target {
            AppendTarget::Path(dataset) => match self.resolved.get(dataset) {
                Some(&addr) => addr,
                None => {
                    let addr = crate::group_v2::resolve_path_any_from_source(
                        &self.image(),
                        AccessMode::ReadWrite,
                        &self.superblock,
                        self.widths,
                        self.addresses,
                        &dataset.as_path(),
                    )?;
                    self.resolved.insert(dataset.clone(), addr);
                    addr
                }
            },
            AppendTarget::Header(addr) => addr,
        };

        // Locate the dataset on the first append (cache miss) against the session's
        // own image  -  no second lock, no second view of the file, no re-read.
        if !self.located.contains_key(&oh_addr) {
            let store = self.store();
            let state = locate_dataset_state(&store, oh_addr)?;
            self.located.insert(oh_addr, state);
        }
        Ok(oh_addr)
    }

    /// The append geometry of the dataset `target` names, so a caller can slice a
    /// large append into aligned batches *before* materializing each batch's
    /// bytes  -  which is what keeps a bounded session's peak memory at one batch
    /// in place of the whole call.
    pub(crate) fn append_geometry(
        &mut self,
        target: AppendTarget<'_>,
    ) -> Result<AppendGeometry, Error> {
        let oh_addr = self.append_prepare(target)?;
        let st = &self.located[&oh_addr];
        let chunk_elems = st.loc.chunk_elems.max(1);
        Ok(AppendGeometry {
            chunk_elems,
            element_size: st.element_size,
            current_dim: st.loc.current_dim,
            lossy_filters: st
                .pipeline
                .as_ref()
                .is_some_and(|p| !super::pipeline_lossless(p)),
            full_batch_elems: self.batch_elems(st.loc.chunk_bytes, chunk_elems),
        })
    }

    /// Whole-chunk elements in one append batch.
    ///
    /// A bounded session caps a batch at [`APPEND_BATCH_BYTES`] of raw data (at
    /// least one chunk) so peak memory is independent of the call size, at the
    /// cost of splitting one crash-atomic append into several: a crash between
    /// batches leaves a valid shorter dataset, exactly as if the caller had
    /// looped. A mirror session already holds the whole file, so bounding the
    /// call buys nothing there and would trade that atomicity away for free,
    /// it takes the whole append as one batch.
    pub(super) fn batch_elems(&self, chunk_bytes: usize, chunk_elems: u64) -> u64 {
        if !self.batched_appends {
            return u64::MAX;
        }
        (APPEND_BATCH_BYTES / (chunk_bytes.max(1) as u64)).max(1) * chunk_elems
    }

    /// Apply a gathered in-place append (typed / generic / raw bytes) to the
    /// dataset `target` names, immediately and crash-atomically, driving the
    /// shared Extensible-Array engine against the session's own image through an
    /// [`EditStore`] adapter. Runs only the first `max_phase` durability phases,
    /// production callers pass 4, the crash-consistency tests stop at a boundary
    /// to simulate a crash.
    ///
    /// A bounded session splits the call into whole-chunk batches (see
    /// [`batch_elems`](Self::batch_elems)), each its own crash-atomic apply.
    /// Every predictable rejection is raised before the first batch, so a rejected
    /// append leaves the file untouched in place of partly grown.
    ///
    /// `Dataset::append` slices its own call the same way before it reaches here,
    /// so through the public API this loop runs once per call, it batches for the
    /// benefit of a caller that hands the engine one large builder directly, whose
    /// bytes are already materialized but whose plan need not be.
    pub(crate) fn append_inplace_gathered(
        &mut self,
        target: AppendTarget<'_>,
        b: &AppendBuilder,
        max_phase: u8,
    ) -> Result<(), Error> {
        if b.dt_conflict() {
            return Err(Error::AppendInPlaceUnsupported(
                "append mixes element types in one call; use one element type per append",
            ));
        }
        let oh_addr = self.append_prepare(target)?;

        // Validate the appended bytes against the on-disk datatype.
        let raw = b.raw();
        let new_elems = validate_gathered_append(&self.located[&oh_addr], b)?;
        if new_elems == 0 {
            return Ok(());
        }

        // In SWMR mode, hold to the subset a concurrent reader can follow safely:
        // unfiltered and chunk-aligned, so an append only ever inserts new,
        // not-yet-visible elements and never rewrites a visible trailing chunk
        // out from under a reader. Outside SWMR the session's exclusive lock is
        // what makes that rewrite safe, see `plan_ea_append`.
        if self.swmr_mode {
            let st = &self.located[&oh_addr];
            if st.pipeline.is_some() {
                return Err(Error::SwmrAppendUnsupported(
                    "filtered datasets are not supported for SWMR append",
                ));
            }
            let chunk_elems = st.loc.chunk_elems;
            if chunk_elems == 0
                || st.loc.current_dim % chunk_elems != 0
                || new_elems % chunk_elems != 0
            {
                return Err(Error::SwmrAppendUnsupported(
                    "SWMR append must be chunk-aligned: the current length and the appended \
                     length must both be whole multiples of the chunk length",
                ));
            }
        }

        let (chunk_elems, elem_bytes, full_batch_elems) = {
            let st = &self.located[&oh_addr];
            (
                st.loc.chunk_elems.max(1),
                st.element_size.get() as u64,
                self.batch_elems(st.loc.chunk_bytes, st.loc.chunk_elems.max(1)),
            )
        };
        // Outside SWMR the session holds an exclusive lock, so re-encoding a
        // filtered partial trailing chunk into a fresh allocation and repointing
        // its index element is a change covered by the exclusive lock, see
        // `plan_ea_append`, which owns that argument and keeps the rejection for
        // the SWMR writer.
        let grow_visible_tail = !self.swmr_mode;

        let mut done = 0u64;
        while done < new_elems {
            // Fill the trailing partial chunk first (so later batches start
            // chunk-aligned and never rewrite it again), then whole-chunk batches.
            // That first batch is the only one that can re-encode a filtered
            // trailing chunk, every batch after it starts on a boundary.
            let current_dim = self.located[&oh_addr].loc.current_dim;
            let to_boundary = (chunk_elems - current_dim % chunk_elems) % chunk_elems;
            let take = (new_elems - done).min(to_boundary.saturating_add(full_batch_elems));
            let batch =
                &raw[(done * elem_bytes).to_usize()?..((done + take) * elem_bytes).to_usize()?];

            // Read/plan phase (immutable borrows only, nothing published yet), then
            // the ordered, fsync-barriered write phase  -  both shared with
            // `Dataset::append` through the chunk-index engine. `EditStore` borrows
            // only the image-carrying fields, so `self.located` stays independently
            // borrowable.
            let plan_result = {
                let Self {
                    image,
                    superblock,
                    widths,
                    sb_sig_off,
                    addresses,
                    space,
                    aggregation,
                    located,
                    sync_policy,
                    ..
                } = self;
                let st = &located[&oh_addr];
                let store = EditStore {
                    image: image.as_mut(),
                    superblock,
                    widths: *widths,
                    sb_sig_off: *sb_sig_off,
                    addresses: *addresses,
                    space,
                    aggregation: aggregation.as_mut(),
                    reserve: None,
                    reuse_session_space: false,
                    sync_policy: *sync_policy,
                };
                chunk_index_inplace::plan_ea_append(
                    &store,
                    &st.loc,
                    &st.datatype,
                    &st.spatial,
                    st.element_size,
                    st.pipeline.as_ref(),
                    grow_visible_tail,
                    batch,
                    take,
                    st.fill.pattern(st.element_size),
                )
            };
            let plan = plan_result.map_err(as_inplace_error)?;
            // A persisting file's holes are advertised on disk, so this batch's
            // bytes have to be taken out of the published managers before any of
            // them is written into. Sized on the largest single blob, since the
            // allocator serves each one from a contiguous run: a reserve that
            // cannot hold the biggest is a reserve the biggest goes past. A
            // draw may be empty. The editor then tries aggregate capacity or allocates at EOA.
            // The manager rewrite runs between planning and application, because it publishes
            // a superblock and must precede any partial application of the append.
            self.reserve_for_immediate_append(plan.largest_stored_size())?;
            self.synchronize_aggregates()?;
            {
                let reuse = self.immediate_reuse_allowed();
                // Two terms, not one. A **paged** session must never be handed
                // the flat list: PAGE keeps reusable space separated by page type, so bypassing
                // the strategy would mix page classes. Every paged session is a persisting one
                // (`append_prepare` rejects a paged file without persistence), so
                // the first term alone would pick `reserved` for it anyway  -  and
                // that is exactly the shape issue #261 was: a page-type invariant
                // proved from a rejection in another function. Named here so the
                // safety is by construction, and so deleting that rejection cannot
                // silently turn a paged append loose on the flat list. `reserved`
                // is empty on any session that never reserved, which makes the
                // extra term cost nothing but the reuse it correctly declines.
                let from_reserve = self.persist.is_some() || self.space.is_paged();
                let Self {
                    image,
                    superblock,
                    widths,
                    sb_sig_off,
                    addresses,
                    space,
                    aggregation,
                    located,
                    reserved,
                    sync_policy,
                    ..
                } = self;
                let st = located.get_mut(&oh_addr).expect("dataset located above");
                let mut store = EditStore {
                    image: image.as_mut(),
                    superblock,
                    widths: *widths,
                    sb_sig_off: *sb_sig_off,
                    addresses: *addresses,
                    space,
                    aggregation: aggregation.as_mut(),
                    reserve: (reuse && from_reserve).then_some(reserved),
                    reuse_session_space: reuse && !from_reserve,
                    sync_policy: *sync_policy,
                };
                let confirmed = store
                    .aggregation
                    .as_ref()
                    .is_none_or(|state| state.publication_confirmed);
                if let Some(state) = store.aggregation.as_deref_mut() {
                    state.publication_confirmed = false;
                }
                let result =
                    chunk_index_inplace::apply_ea_append(&mut store, &mut st.loc, plan, max_phase);
                if result.is_ok() {
                    if let Some(state) = store.aggregation.as_deref_mut() {
                        state.publication_confirmed = confirmed;
                    }
                } else {
                    located.remove(&oh_addr);
                }
                result.map_err(as_inplace_error)?;
            }
            if max_phase < 4 {
                self.located.remove(&oh_addr);
                // Crash-consistency hook: the caller requested to stop inside the
                // first batch's durability sequence, so there is no next batch.
                // Each phase's own barrier has already issued the writes up to it
                // under either policy, which is what leaves a phase boundary a
                // real one for the tests that stop at it.
                return Ok(());
            }
            done += take;
        }
        Ok(())
    }

    /// Test-only phased in-place append (stops after `max_phase` durability phases)
    /// used by the crash-consistency tests, mirroring `Dataset::append`'s harness.
    #[cfg(test)]
    pub(super) fn append_inplace_i32_phased(
        &mut self,
        dataset: &str,
        values: &[i32],
        max_phase: u8,
    ) -> Result<(), Error> {
        let mut b = AppendBuilder::new();
        b.append_i32(values);
        self.append_inplace_gathered(
            AppendTarget::Path(&ObjectPathBuf::parse(dataset)),
            &b,
            max_phase,
        )
    }
}
