//! Immediate chunk appends and staged append inputs.
//!
//! The editor checks append policy and caches dataset metadata here.
//! [`chunk_index_inplace`] writes chunk data, updates the index, and publishes the extent.

use core::num::NonZeroU64;
use core::num::NonZeroUsize;

use hdf5_pure_format::__private::ChunkedGeometry;

use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::{FreeList, PageTransition, PageType, SessionSpace};

use hdf5_pure_core::MetadataBytes;

use crate::DatatypeByteOrder;
use crate::SuperblockVersion;
use crate::access_mode::AccessMode;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::address::{AddressContext, BaseAddressExt};
use crate::chunk_index_inplace::{self, Located, RetiredChunk, Store};
use crate::convert::Narrow;
use crate::data_layout::DataLayout;
use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::error::{Error, FormatError};
use crate::filter_pipeline::FilterPipeline;
use crate::image::FileImage;
use crate::message_type::MessageType;
use crate::object_path::ObjectPathBuf;
use crate::primitive_bytes::PrimitiveBytes;
use crate::type_builders;
use crate::width::FormatWidths;

use super::LOSSY_TAIL_REFUSAL;
use super::context::FileContext;
use super::staging::StagedOp;

use super::{AggregateSpace, Placement, SyncPolicy, WriteEngine};

/// Options for immediate, staged, or buffered appends.
///
/// The default rejects re-encoding an existing partial tail stored with lossy filters.
/// [`with_lossy_tail_reencoding`](Self::with_lossy_tail_reencoding) changes that policy for
/// the append receiving these options. A [`BufferedAppender`](crate::BufferedAppender)
/// uses the supplied policy for every write during its lifetime. The options are not stored
/// in the file and do not affect other append calls, including calls made after reopening it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AppendOptions {
    lossy_tail_reencoding: LossyTailReencoding,
}

impl AppendOptions {
    /// Creates options with [`LossyTailReencoding::Reject`].
    pub const fn new() -> Self {
        Self {
            lossy_tail_reencoding: LossyTailReencoding::Reject,
        }
    }

    /// Sets whether the append may re-encode an existing lossy partial tail.
    ///
    /// [`LossyTailReencoding::Allow`] permits changes to previously published tail values.
    pub const fn with_lossy_tail_reencoding(mut self, reencoding: LossyTailReencoding) -> Self {
        self.lossy_tail_reencoding = reencoding;
        self
    }

    /// Returns the selected policy, initially [`LossyTailReencoding::Reject`].
    pub const fn lossy_tail_reencoding(self) -> LossyTailReencoding {
        self.lossy_tail_reencoding
    }
}

/// Controls re-encoding of a partial trailing chunk stored with lossy filters.
///
/// The writer decodes the existing tail and encodes it with the appended values. Each
/// rewrite can introduce additional quantization error in values already published in that
/// tail. Complete chunks preceding the tail are retained.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LossyTailReencoding {
    /// Permits re-encoding previously published values in a lossy partial tail.
    Allow,
    /// Rejects growth of a partial tail stored with lossy filters.
    #[default]
    Reject,
}

/// Accumulates elements for a staged append along the first dimension.
///
/// Repeated calls concatenate their values in call order. Typed calls record an element datatype,
/// which the commit compares with the dataset's datatype. Mixing element types is rejected with
/// [`Error::AppendUnsupported`]. [`Dataset::append_staged`](crate::Dataset::append_staged) supplies
/// this builder to its callback.
pub struct AppendBuilder {
    /// Element bytes in call order, encoded little-endian.
    raw: Vec<u8>,
    /// The datatype of typed appends, or `None` for raw appends.
    elem_dt: Option<Datatype>,
    /// Whether typed calls used different element datatypes.
    dt_conflict: bool,
    /// Policy retained through staging and copied append batches.
    options: AppendOptions,
}

impl AppendBuilder {
    /// Creates an empty append buffer.
    pub(crate) fn new() -> Self {
        Self::with_options(AppendOptions::new())
    }

    /// Creates an empty append buffer with the supplied policy.
    pub(crate) fn with_options(options: AppendOptions) -> Self {
        Self {
            raw: Vec::new(),
            elem_dt: None,
            dt_conflict: false,
            options,
        }
    }

    /// Returns the policy retained by this append buffer.
    pub(crate) fn options(&self) -> AppendOptions {
        self.options
    }

    /// Returns the accumulated element bytes.
    pub(crate) fn raw(&self) -> &[u8] {
        &self.raw
    }

    /// Returns the datatype recorded by typed appends.
    pub(crate) fn elem_dt(&self) -> Option<&Datatype> {
        self.elem_dt.as_ref()
    }

    /// Returns whether typed appends used different datatypes.
    pub(crate) fn dt_conflict(&self) -> bool {
        self.dt_conflict
    }

    /// Copies up to `byte_len` leading bytes with their datatype information and append policy.
    ///
    /// [`BufferedAppender`](crate::BufferedAppender) retains the original buffer until the write
    /// succeeds.
    pub(crate) fn head(&self, byte_len: usize) -> Self {
        let Self {
            raw,
            elem_dt,
            dt_conflict,
            options,
        } = self;
        Self {
            raw: raw[..byte_len.min(raw.len())].to_vec(),
            elem_dt: elem_dt.clone(),
            dt_conflict: *dt_conflict,
            options: *options,
        }
    }

    /// Removes up to `byte_len` leading bytes after a successful write.
    pub(crate) fn drop_front(&mut self, byte_len: usize) {
        self.raw.drain(..byte_len.min(self.raw.len()));
    }

    /// Returns the buffered bytes to
    /// [`BufferedAppender::discard`](crate::BufferedAppender::discard).
    pub(crate) fn into_raw(self) -> Vec<u8> {
        self.raw
    }

    /// Restores the buffer length after an append rejected before writing.
    pub(crate) fn truncate(&mut self, byte_len: usize) {
        self.raw.truncate(byte_len);
    }

    /// Records a typed append's datatype and any conflict with earlier calls.
    fn set_dt(&mut self, dt: Datatype) {
        match &self.elem_dt {
            Some(prev) if *prev != dt => self.dt_conflict = true,
            Some(_) => {}
            None => self.elem_dt = Some(dt),
        }
    }

    /// Appends element bytes encoded in the dataset's on-disk datatype.
    ///
    /// The total byte length must be a multiple of the element size. Numeric fields must be
    /// little-endian, except one-byte integers, whose byte order is immaterial. Variable-length
    /// and reference datatypes are rejected when the append is prepared.
    pub fn append_raw(&mut self, bytes: &[u8]) -> &mut Self {
        self.raw.extend_from_slice(bytes);
        self
    }

    /// Appends scalar values with their element datatype.
    pub fn append<T: crate::element::H5Element>(&mut self, data: &[T]) -> &mut Self {
        T::append_into(self, data);
        self
    }
}

/// Defines typed append methods that encode values little-endian.
macro_rules! append_typed {
    ($($method:ident, $ty:ty, $make:path;)*) => {
        impl AppendBuilder {
            $(
                #[doc = concat!("Appends `", stringify!($ty), "` values to the buffer.")]
                pub fn $method(&mut self, data: &[$ty]) -> &mut Self {
                    self.set_dt($make());
                    <$ty>::append_le(data, &mut self.raw);
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

impl WriteEngine {
    /// Stages an append to a committed dataset for the next [`commit`](Self::commit).
    ///
    /// The commit checks the dataset's append geometry and storage. A lossy partial trailing
    /// chunk is rejected at staging when its geometry can be read and re-encoding is forbidden.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a dataset creation is staged at `path` or a live
    /// buffered
    /// appender overlaps it. Returns [`Error::AppendUnsupported`] if a nonempty append would
    /// re-encode a partial trailing chunk through a lossy pipeline without permission.
    pub(crate) fn stage_dataset_append(
        &mut self,
        path: &ObjectPathBuf,
        builder: AppendBuilder,
    ) -> Result<(), Error> {
        self.refuse_if_claimed(path)?;
        if self.staged.dataset_at(path).is_some() {
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

    /// Returns the geometry and batch size for an immediate append.
    ///
    /// The same target restrictions apply as in
    /// [`append_inplace_gathered`](Self::append_inplace_gathered).
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendInPlaceUnsupported`] for an unsupported file or dataset, or a
    /// conflicting staged edit. Returns metadata read and parse errors.
    pub(crate) fn append_geometry(
        &mut self,
        target: AppendTarget<'_>,
    ) -> Result<AppendGeometry, Error> {
        let oh_addr = self.append_prepare(target)?;
        let st = &self.located[&oh_addr];
        let chunk_elems = st.loc.chunk_elems;
        Ok(AppendGeometry {
            chunk_elems,
            element_size: st.element_size,
            current_dim: st.loc.current_dim,
            lossy_filters: st
                .pipeline
                .as_ref()
                .is_some_and(|p| !super::pipeline_lossless(p))
                && !st.loc.flags.partial_edge_chunks_stored_raw(),
            full_batch_elems: self.batch_elems(st.loc.chunk_bytes, chunk_elems),
        })
    }

    /// Applies an immediate append through the shared extensible-array writer.
    ///
    /// Bounded sessions limit the full-chunk portion of each batch with
    /// [`batch_elems`](Self::batch_elems).
    /// The first batch may also fill a partial trailing chunk. Intermediate batch boundaries are
    /// chunk-aligned, and the final batch may end within a chunk. Each batch publishes its extent
    /// after its data and index updates, with barriers governed by [`SyncPolicy`]. Predictable
    /// rejections are checked before writing. SWMR appends require unfiltered, chunk-aligned input.
    ///
    /// Production callers pass 4 for `max_phase`. Crash tests stop at an earlier phase boundary.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendInPlaceUnsupported`] for conflicting element types, incompatible
    /// bytes, an unsupported target, or conflicting staged edits. Returns
    /// [`Error::SwmrAppendUnsupported`] for filtered or unaligned SWMR appends. Returns metadata,
    /// allocation, encoding, and I/O errors.
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

        let raw = b.raw();
        let new_elems = validate_gathered_append(&self.located[&oh_addr], b)?;
        if new_elems == 0 {
            return Ok(());
        }

        // SWMR appends leave visible chunks unchanged.
        if self.swmr_mode {
            let st = &self.located[&oh_addr];
            if st.pipeline.is_some() {
                return Err(Error::SwmrAppendUnsupported(
                    "filtered datasets are not supported for SWMR append",
                ));
            }
            let chunk_elems = st.loc.chunk_elems;
            if st.loc.current_dim % chunk_elems != 0 || new_elems % chunk_elems != 0 {
                return Err(Error::SwmrAppendUnsupported(
                    "SWMR append must be chunk-aligned: the current length and the appended \
                     length must both be whole multiples of the chunk length",
                ));
            }
        }

        let (chunk_elems, elem_bytes, full_batch_elems) = {
            let st = &self.located[&oh_addr];
            (
                st.loc.chunk_elems,
                st.element_size.get() as u64,
                self.batch_elems(st.loc.chunk_bytes, st.loc.chunk_elems),
            )
        };
        let grow_visible_tail = !self.swmr_mode;

        let mut done = 0u64;
        while done < new_elems {
            // Add the leading partial chunk's capacity to the full-chunk budget.
            let current_dim = self.located[&oh_addr].loc.current_dim;
            let to_boundary = (chunk_elems - current_dim % chunk_elems) % chunk_elems;
            let take = (new_elems - done).min(to_boundary.saturating_add(full_batch_elems));
            let batch =
                &raw[(done * elem_bytes).to_usize()?..((done + take) * elem_bytes).to_usize()?];

            let plan_result = {
                let Self {
                    image,
                    context,
                    space,
                    aggregation,
                    located,
                    sync_policy,
                    ..
                } = self;
                let st = &located[&oh_addr];
                let store = EditStore {
                    image: image.as_mut(),
                    context,
                    space,
                    aggregation: aggregation.as_mut(),
                    reserve: None,
                    reuse_session_space: false,
                    sync_policy: *sync_policy,
                    fsm_dirty: None,
                };
                chunk_index_inplace::plan_ea_append(
                    &store,
                    &st.loc,
                    &st.datatype,
                    &st.spatial,
                    st.element_size,
                    st.pipeline.as_ref(),
                    grow_visible_tail,
                    b.options().lossy_tail_reencoding(),
                    batch,
                    take,
                    st.fill.pattern(st.element_size),
                )
            };
            let plan = plan_result.map_err(as_inplace_error)?;
            // Remove reserved space from published managers before writing into it.
            // The manager rewrite publishes a superblock and must precede append application.
            self.reserve_for_immediate_append(plan.largest_stored_size())?;
            self.synchronize_aggregates()?;
            {
                let reuse = self.immediate_reuse_allowed();
                // PAGE allocation must use page-typed reserves.
                let from_reserve = self.persist.is_some() || self.space.is_paged();
                let Self {
                    image,
                    context,
                    space,
                    aggregation,
                    located,
                    reserved,
                    fsm_dirty,
                    sync_policy,
                    ..
                } = self;
                let st = located.get_mut(&oh_addr).expect("dataset located above");
                let mut store = EditStore {
                    image: image.as_mut(),
                    context,
                    space,
                    aggregation: aggregation.as_mut(),
                    reserve: (reuse && from_reserve).then_some(reserved),
                    reuse_session_space: reuse && !from_reserve,
                    sync_policy: *sync_policy,
                    fsm_dirty: from_reserve.then_some(fsm_dirty),
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
                // Each completed phase has issued its writes under either sync policy.
                return Ok(());
            }
            done += take;
        }
        Ok(())
    }

    /// Stops an immediate append at a durability phase boundary for crash tests.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`append_inplace_gathered`](Self::append_inplace_gathered).
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

    /// Resolves an eligible append target and caches its metadata.
    ///
    /// The cache key is the object-header address, shared by every hard link to the dataset.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendInPlaceUnsupported`] for a userblock, an older superblock,
    /// non-persistent PAGE storage, conflicting staged edits, or an unsupported dataset. Returns
    /// metadata read and parse errors.
    pub(super) fn append_prepare(&mut self, target: AppendTarget<'_>) -> Result<u64, Error> {
        // Immediate append requires base-zero addresses and a version 2 or 3 superblock.
        if !self.context.superblock().base_address.is_zero() {
            return Err(Error::AppendInPlaceUnsupported(
                "in-place append does not support a file with a userblock (non-zero base \
                 address); use Dataset::append_staged",
            ));
        }
        if matches!(
            self.context.superblock().version(),
            SuperblockVersion::Zero | SuperblockVersion::One
        ) {
            return Err(Error::AppendInPlaceUnsupported(
                "in-place append requires a latest-format file (v2/v3 superblock); use \
                 Dataset::append_staged",
            ));
        }
        // Non-persistent PAGE editing supports staged allocation in fresh typed pages.
        if self.space.is_paged() && self.persist.is_none() {
            return Err(Error::AppendInPlaceUnsupported(
                "in-place append is not supported on a paged file \
                 (H5F_FSPACE_STRATEGY_PAGE) without persisted free space; recreate the \
                 file with with_file_space_settings(FileSpaceSettings::new(\
                 FileSpaceStrategy::Page).with_persistence(true))",
            ));
        }

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

        let oh_addr = match target {
            AppendTarget::Path(dataset) => match self.resolved.get(dataset) {
                Some(&addr) => addr,
                None => {
                    let addr = crate::group_v2::resolve_path_any_from_source(
                        &self.image(),
                        AccessMode::ReadWrite,
                        self.context.superblock(),
                        self.context.widths(),
                        self.context.addresses(),
                        &dataset.as_path(),
                    )?;
                    self.resolved.insert(dataset.clone(), addr);
                    addr
                }
            },
            AppendTarget::Header(addr) => addr,
        };

        if !self.located.contains_key(&oh_addr) {
            let store = self.store();
            let state = locate_dataset_state(&store, oh_addr)?;
            self.located.insert(oh_addr, state);
        }
        Ok(oh_addr)
    }

    /// Rejects an append that would re-encode a lossy partial tail without permission.
    ///
    /// Empty appends and explicit permission to re-encode the tail are accepted. If the
    /// committed geometry cannot be read, its validation is deferred to
    /// [`prepare_append`](Self::prepare_append).
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] if a nonempty append would grow a rank-one
    /// dataset's filtered lossy partial tail without permission.
    pub(super) fn refuse_lossy_partial_tail(
        &self,
        path: &ObjectPathBuf,
        builder: &AppendBuilder,
    ) -> Result<(), Error> {
        if builder.raw().is_empty()
            || builder.options().lossy_tail_reencoding() == LossyTailReencoding::Allow
        {
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
            self.context.superblock(),
            self.context.widths(),
            self.context.addresses(),
            &path.as_path(),
        ) else {
            return false;
        };
        let Ok(region) = Self::gather_oh_messages(&self.image(), addr, self.context.addresses())
        else {
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
            Dataspace::parse_with_length_width(&region[ds_b..ds_e], self.context.widths().lengths)
        else {
            return false;
        };
        let Ok(dl) = DataLayout::parse_with_widths(&region[lb..le], self.context.widths()) else {
            return false;
        };
        if matches!(&dl, DataLayout::Chunked { flags, .. } if flags.partial_edge_chunks_stored_raw())
        {
            return false;
        }
        if disk_ds.dimensions.len() != 1 {
            return false;
        }
        let Ok(geometry) = ChunkedGeometry::new(&disk_dt, &disk_ds, &dl) else {
            return false;
        };
        match (geometry.spatial.first(), disk_ds.dimensions.first()) {
            (Some(&chunk_elems), Some(&dim0)) => dim0 % chunk_elems != 0,
            _ => false,
        }
    }

    /// Returns the whole-chunk element limit for an append batch.
    ///
    /// Bounded sessions allow up to [`APPEND_BATCH_BYTES`] of full-chunk data, with at least one
    /// chunk. The first batch may also include elements added to an existing partial chunk.
    /// Sessions without batching return [`u64::MAX`].
    pub(super) fn batch_elems(&self, chunk_bytes: usize, chunk_elems: u64) -> u64 {
        if !self.batched_appends {
            return u64::MAX;
        }
        (APPEND_BATCH_BYTES / (chunk_bytes as u64)).max(1) * chunk_elems
    }

    /// Borrows the session's image and allocation state for the shared append writer.
    pub(super) fn store(&mut self) -> EditStore<'_> {
        EditStore {
            image: self.image.as_mut(),
            context: &mut self.context,
            space: &mut self.space,
            aggregation: self.aggregation.as_mut(),
            reserve: None,
            reuse_session_space: false,
            sync_policy: self.sync_policy,
            fsm_dirty: self.persist.is_some().then_some(&mut self.fsm_dirty),
        }
    }
}

/// Identifies an immediate append target by path or object-header address.
///
/// Path targets allow conflict checks against individual staged edits. Address targets require
/// a session without staged or committed edits that could relocate the header.
#[derive(Clone, Copy)]
pub(crate) enum AppendTarget<'a> {
    Path(&'a ObjectPathBuf),
    Header(u64),
}

/// Describes chunk-aligned batching without materializing append bytes.
pub(crate) struct AppendGeometry {
    /// Elements per chunk along the first dimension.
    pub(crate) chunk_elems: u64,
    /// Bytes per on-disk element.
    pub(crate) element_size: NonZeroUsize,
    /// Current length along the unlimited dimension.
    pub(crate) current_dim: u64,
    /// Whether growing a partial trailing chunk would re-encode lossy data.
    pub(crate) lossy_filters: bool,
    /// Whole-chunk element limit, or [`u64::MAX`] when batching is disabled.
    pub(crate) full_batch_elems: u64,
}

/// Caches dataset metadata maintained across immediate appends.
pub(crate) struct LocatedState {
    pub(crate) loc: Located,
    /// The datatype used for append compatibility and filter context.
    pub(crate) datatype: Datatype,
    /// The rank-one spatial chunk dimensions.
    pub(crate) spatial: Vec<u64>,
    /// Bytes per element (datatype size), proven non-zero.
    pub(crate) element_size: NonZeroUsize,
    /// The filter pipeline, restricted to filters the writer can re-encode.
    pub(crate) pipeline: Option<FilterPipeline>,
    /// Padding for elements beyond the published extent.
    pub(crate) fill: crate::fill_value::PaddingFill,
}

/// Adapts session allocation and I/O to the shared extensible-array writer.
///
/// Reusable space is tried before classic aggregates. Chunk payloads use the raw aggregate,
/// and index blocks use the metadata aggregate. PAGE allocation uses raw pages for both.
pub(super) struct EditStore<'a> {
    image: &'a mut dyn FileImage,
    context: &'a mut FileContext,
    /// Allocation state, including PAGE transitions persisted at commit or close.
    space: &'a mut SessionSpace,
    /// Classic aggregates shared with staged edits.
    aggregation: Option<&'a mut AggregateSpace>,
    /// Space removed from published free-space managers before an immediate append.
    pub(super) reserve: Option<&'a mut FreeList>,
    /// Whether allocation may use the session's free space directly.
    pub(super) reuse_session_space: bool,
    /// Durability policy for the shared writer's ordering barriers.
    sync_policy: SyncPolicy,
    /// Persistence dirtiness updated when confirmed predecessor storage is retired.
    fsm_dirty: Option<&'a mut bool>,
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
        let eoa = self.context.addresses().relative(self.image.len())?.get();
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

    fn read_metadata(&self, offset: u64, len: usize) -> Result<MetadataBytes<'_>, FormatError> {
        self.image.read_metadata(offset, len)
    }

    fn metadata_cache_stats(&self) -> Option<crate::source::MetadataCacheStats> {
        self.image.metadata_cache_stats()
    }

    fn reset_metadata_cache_stats(&self) {
        self.image.reset_metadata_cache_stats();
    }
}

impl Store for EditStore<'_> {
    fn widths(&self) -> FormatWidths {
        self.context.widths()
    }

    fn addresses(&self) -> AddressContext {
        self.context.addresses()
    }

    fn retire(&mut self, retirement: RetiredChunk) {
        let (extent, class) = retirement.into_extent();
        // Opening checks the base against the image length, and editor truncation preserves it.
        let eoa = self
            .image
            .len()
            .checked_sub(self.context.addresses().effective_base().get())
            .expect("the image includes its base address");
        self.space.record_freed(eoa, extent, class);
        if let Some(dirty) = &mut self.fsm_dirty {
            **dirty = true;
        }
    }

    fn recorded_eof(&self) -> u64 {
        self.context.superblock().eof_address
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
            let at = self
                .context
                .addresses()
                .resolve(StoredAddress::new(extent.start()))?;
            self.image.write_at(at.get(), bytes)?;
            return Ok(at.get());
        }
        if let (Some(aggregation), Some(size)) = (
            self.aggregation.as_deref_mut(),
            NonZeroU64::new(bytes.len() as u64),
        ) {
            let Placement::Aggregated { addr, end, .. } =
                aggregation.reserve(self.context.addresses(), self.space, size, class)?
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
        // Preserve the consistency flags and root address when advancing the recorded EOF.
        let mut superblock = self.context.superblock().clone();
        superblock.eof_address = self.image.len();
        let publication = self.context.propose(superblock)?.encode()?;
        self.write_at(self.context.signature().get(), publication.bytes())?;
        self.context.adopt(publication.into_proposed())?;
        Ok(())
    }

    fn sync(&mut self) -> Result<(), Error> {
        super::barrier_data(self.image, self.sync_policy)
    }
}

/// Converts a shared append rejection to the immediate-append error variant.
///
/// I/O and format errors are returned unchanged.
pub(crate) fn as_inplace_error(e: Error) -> Error {
    match e {
        Error::AppendUnsupported(m) => Error::AppendInPlaceUnsupported(m),
        other => other,
    }
}

/// Returns the element count of bytes compatible with a located dataset.
///
/// Typed appends require datatype equality. Raw appends require a datatype accepted by
/// [`datatype_is_raw_appendable`].
///
/// # Errors
///
/// Returns [`Error::AppendInPlaceUnsupported`] for an incomplete element, a datatype mismatch,
/// or an unsupported raw datatype.
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
            if !datatype_is_raw_appendable(&st.datatype) {
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

/// Reads checked geometry, the datatype, and encoding metadata for an immediate append.
///
/// The dataset must be rank one, unlimited, and indexed by an allocated Extensible Array. The
/// filter
/// pipeline must be re-encodable. Unreadable fill metadata is rejected only when padding is needed.
///
/// # Errors
///
/// Returns [`Error::AppendInPlaceUnsupported`] for unsupported geometry, filters, or storage.
/// Returns [`Error::Format`] for malformed chunk geometry or inconsistent element sizes, and
/// metadata read and parse errors.
pub(crate) fn locate_dataset_state<F: Store>(
    file: &F,
    oh_addr: u64,
) -> Result<LocatedState, Error> {
    let result = Located::locate_at(file, oh_addr, Error::AppendInPlaceUnsupported)?;
    let datatype = result.datatype;
    let element_size = result.located.elem_bytes;
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
    // Unreadable fill metadata is rejected only when a chunk needs padding.
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

/// Returns whether a datatype can be appended as flat little-endian bytes.
///
/// Numeric leaves must be little-endian, except one-byte integers. Strings and opaque values
/// are byte arrays. Aggregate types require every nested type to qualify. Variable-length and
/// reference values require address resolution and are excluded.
pub(crate) fn datatype_is_raw_appendable(dt: &Datatype) -> bool {
    match dt {
        Datatype::FixedPoint(numeric) => {
            numeric.size().get() == 1 || numeric.byte_order() == DatatypeByteOrder::LittleEndian
        }
        Datatype::FloatingPoint(numeric) => numeric.byte_order() == DatatypeByteOrder::LittleEndian,
        Datatype::Time(numeric) => numeric.byte_order() == DatatypeByteOrder::LittleEndian,
        Datatype::BitField(numeric) => numeric.byte_order() == DatatypeByteOrder::LittleEndian,
        Datatype::String { .. } | Datatype::Opaque { .. } => true,
        Datatype::Enumeration { base_type, .. } | Datatype::Array { base_type, .. } => {
            datatype_is_raw_appendable(base_type)
        }
        Datatype::Compound { members, .. } => members
            .iter()
            .all(|m| datatype_is_raw_appendable(&m.datatype)),
        Datatype::VariableLength(_) | Datatype::Reference { .. } | _ => false,
    }
}

/// The full-chunk byte budget of a bounded append batch, subject to a minimum of one chunk.
///
/// The first batch may also include the elements needed to fill an existing partial chunk.
pub(super) const APPEND_BATCH_BYTES: u64 = 1 << 20;
