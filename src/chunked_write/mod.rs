//! Chunked dataset writing: chunk splitting, compression, index building.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

/// The HDF5 "undefined address" sentinel (`HADDR_UNDEF`): all bits set, in
/// whatever width the field is. A chunked layout message carrying it declares
/// that the dataset has no storage allocated at all.
const HADDR_UNDEF: u64 = u64::MAX;
use core::convert::Infallible;
use core::num::NonZeroUsize;

use hdf5_pure_format::__private::ChunkIndexInfo;
use hdf5_pure_format::__private::ChunkRecord;
use hdf5_pure_format::__private::FilteredSingleChunk;
use hdf5_pure_format::__private::IndexSlots;
use hdf5_pure_format::__private::SlotOccupancy;

use crate::DeflateLevel;
use crate::FillTime;
#[cfg(test)]
use crate::FloatingPointType;
#[cfg(feature = "zfp")]
use crate::ZfpRate;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunk_grid::{ChunkGrid, GridOrder};
use crate::convert::Narrow;
use crate::data_layout::ChunkedLayoutFlags;
use crate::data_layout::DONT_FILTER_PARTIAL_BOUND_CHUNKS;
use crate::data_layout::DataLayout;
use crate::dataspace::{Extent, MaxExtent};
use crate::error::FormatError;
use crate::file_emission::EmissionError;
use crate::fill_value::FillPattern;
use crate::filter_pipeline::{
    FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_LZF, FILTER_SCALEOFFSET, FILTER_SHUFFLE,
    FilterDescription, FilterPipeline, H5Z_FLAG_OPTIONAL,
};
use crate::filters::ChunkContext;
use crate::filters::FilterScratch;
use crate::layout_info::FilterId;
use crate::prepared_chunk::PreparedChunk;
use crate::scaleoffset::{self, FillAvailability, ScaleOffset};
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

pub(crate) use self::gathering::ChunkTraversal;

mod gathering;
mod zfp;

#[cfg(all(test, feature = "std"))]
test_util_hdf5::hdf5_1_10_tests! {
    mod native_crosschecks;
}

/// The on-disk address and length widths every chunk index *this module* writes
/// uses. Named so the path that sizes an index and the path that emits it cannot
/// read different values.
///
/// Not the only such pair: the in-place Extensible Array rebuild in `edit`
/// reads `file_writer::OFFSET_SIZE` / `LENGTH_SIZE`, which carry the same
/// meaning and the same values. Both are what this crate's superblock declares,
/// and `edit` refuses a file whose superblock disagrees, so the two cannot drift
/// apart within one file.
const INDEX_OFFSET_SIZE: OffsetWidth = OffsetWidth::Eight;
const INDEX_LENGTH_SIZE: LengthWidth = LengthWidth::Eight;

/// Which filter, and the parameters that are the caller's to choose.
///
/// Not the whole `cd_values` word list. What a filter records about the
/// *dataset* — element size, chunk geometry, the fill-value bytes — is derived
/// at [`ChunkOptions::build_pipeline`] time from the dataset actually being
/// written, so a filter carried from one dataset onto another (what
/// [`repack`](crate::repack()) does) describes the destination without
/// restating the source's numbers. What a filter records about the *request* —
/// a deflate level, a scale-offset mode and whether it records a fill value at
/// all — is here, and is carried across unchanged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterKind {
    /// ZFP fixed-rate compression, in bits per value. A primary transform:
    /// it consumes the raw elements and displaces the byte filters.
    #[cfg(feature = "zfp")]
    Zfp(ZfpRate),
    /// Scale-offset, with whether the filter records the dataset's fill value.
    /// Also a primary transform: it displaces shuffle, but a byte compressor
    /// may follow it.
    ScaleOffset(ScaleOffset, FillAvailability),
    /// Byte shuffle.
    Shuffle,
    /// The h5py LZF filter (id 32000).
    Lzf,
    /// Deflate compression at the specified [`DeflateLevel`].
    Deflate(DeflateLevel),
    /// Fletcher32 checksum.
    Fletcher32,
}

impl FilterKind {
    /// Returns the scale-offset filter in `mode` that records the dataset's fill value.
    pub(crate) fn scale_offset(mode: ScaleOffset) -> Self {
        // libhdf5 records `Defined` for every dataset whose fill value is not
        // `H5D_FILL_VALUE_UNDEFINED` (`H5Z__set_local_scaleoffset` in `src/H5Zscaleoffset.c`,
        // release 2.2.0).
        Self::ScaleOffset(mode, FillAvailability::Defined)
    }

    /// The registered HDF5 filter id this kind is written as.
    ///
    /// Identity for every purpose that asks "is this filter in the pipeline":
    /// two `Deflate`s at different levels are one filter appearing twice, not
    /// two filters, which is what makes [`ChunkOptions::set_filter`] replace
    /// rather than accumulate.
    pub(crate) fn filter_id(self) -> FilterId {
        match self {
            #[cfg(feature = "zfp")]
            Self::Zfp(_) => FilterId::Zfp,
            Self::ScaleOffset(..) => FilterId::ScaleOffset,
            Self::Shuffle => FilterId::Shuffle,
            Self::Lzf => FilterId::Lzf,
            Self::Deflate(_) => FilterId::Deflate,
            Self::Fletcher32 => FilterId::Fletcher32,
        }
    }

    /// Whether a filter added through [`ChunkOptions::set_filter`] is recorded
    /// optional.
    ///
    /// Only LZF is. Unlike every other filter here it *can* decline a chunk:
    /// liblzf returns 0 for one it cannot shrink, and h5py's filter relies on
    /// the optional flag to store that chunk raw with its filter-mask bit set.
    /// A mandatory LZF makes that a hard error, so h5py could not write
    /// incompressible data back into a file we wrote. (Our own writer applies
    /// LZF unconditionally — a grown stream is a valid stream — so it never
    /// sets a mask bit; the flag is for the writers that come after us.)
    ///
    /// For a filter that cannot fail the flag is unobservable here, and leaving
    /// those mandatory keeps our bytes stable against existing fixtures. h5py
    /// marks every filter optional; we match it only where it means something.
    fn default_optional(self) -> bool {
        matches!(self, Self::Lzf)
    }
}

/// A filter together with whether a reader may skip it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilterSpec {
    /// Which filter, and its parameters.
    pub kind: FilterKind,
    /// `H5Z_FLAG_OPTIONAL` (flags bit 0): a reader that cannot apply this
    /// filter may skip it. A mandatory filter must be applied for the data to
    /// decode, so a reader missing it has to fail.
    pub optional: bool,
}

/// Options for chunked dataset creation.
#[derive(Debug, Clone, Default)]
pub struct ChunkOptions {
    /// Chunk dimensions (one per dataset dimension).
    pub chunk_dims: Option<Vec<u64>>,
    /// The dataset's filter pipeline, in the order it is written: each filter's
    /// output is the next one's input, and a reader reverses the list.
    ///
    /// An ordered list rather than one switch per filter, because the order is
    /// observable and is not always this crate's canonical one. `fletcher32`
    /// before deflate checksums the uncompressed bytes; after it, the
    /// compressed ones — a different checksum over different data, and a
    /// different thing to verify on read. A file written elsewhere may declare
    /// any order, and `repack` reproduces what it read (issue #333), which a
    /// set of independent switches cannot express at all.
    ///
    /// Add to it through [`set_filter`](Self::set_filter), which places a
    /// filter where this crate's own writer puts it, or
    /// [`push_filter`](Self::push_filter), which appends one exactly where the
    /// caller wants it.
    pub filters: Vec<FilterSpec>,
    /// The source filter message to copy when storage is unallocated.
    ///
    /// The writer uses this message in place of rebuilding the pipeline from [`Self::filters`].
    pub(crate) unallocated_pipeline: Option<Vec<u8>>,
    /// Whether the filter pipeline applies to a partial edge chunk, a chunk that extends past the
    /// dataset's extent, or `None` where the caller sets no policy.
    ///
    /// The writer creates a dataset under `None` as under [`PartialEdgeFiltering::Apply`], and a
    /// read-write session rejects an overwrite whose options hold either policy.
    pub(crate) partial_edge_filtering: Option<PartialEdgeFiltering>,
}

impl ChunkOptions {
    /// Whether any chunking option is enabled.
    pub fn is_chunked(&self) -> bool {
        self.chunk_dims.is_some() || !self.filters.is_empty()
    }

    /// Inserts `kind` in canonical filter order, replacing an existing filter with the same ID.
    ///
    /// A replacement keeps its original position in the pipeline.
    pub fn set_filter(&mut self, kind: FilterKind) {
        let spec = FilterSpec {
            optional: kind.default_optional(),
            kind,
        };
        // Naming a filter twice sets its parameter rather than applying it
        // twice, which is what the `Option<u32>`-per-filter fields this replaced
        // did. The whole entry is overwritten, optional flag included: this
        // setter states what the crate's own writer wants, so a flag carried in
        // by `push_filter` does not survive it. Overwritten where it sits rather
        // than removed and re-inserted, so a pushed pipeline at least keeps its
        // order. No caller mixes the two today; both halves are stated because
        // the field is `pub` and nothing enforces that.
        if let Some(slot) = self
            .filters
            .iter_mut()
            .find(|f| f.kind.filter_id() == kind.filter_id())
        {
            *slot = spec;
            return;
        }
        let at = hdf5_pure_filter::__private::canonical_filter_position(
            self.filters.iter().map(|f| f.kind.filter_id().get()),
            kind.filter_id().get(),
        );
        self.filters.insert(at, spec);
    }

    /// Append `spec` after every filter already added, keeping its optional
    /// flag as given.
    ///
    /// For reproducing a pipeline that already exists: `repack` reads a
    /// source's filters and re-applies them in the order and with the flags it
    /// found, which need not be the canonical order
    /// [`set_filter`](Self::set_filter) produces.
    pub fn push_filter(&mut self, spec: FilterSpec) {
        self.filters.push(spec);
    }

    /// Refuse a filter the pipeline asks for that this build cannot apply.
    /// Returns a static reason on the first problem; callers map it to their
    /// own error type, as they do for
    /// [`validate_geometry`](Self::validate_geometry).
    ///
    /// Deflate is behind a crate feature, but
    /// [`build_pipeline`](Self::build_pipeline) emits its descriptor either
    /// way, so nothing downstream refuses the request — the first chunk to be
    /// compressed fails instead, with `UnsupportedFilter`.
    ///
    /// Called from the in-place edit path, which has file bytes to protect: a
    /// commit that got that far would have written some. The whole-file writer
    /// does not call it and still fails at the first chunk, which costs only a
    /// discarded buffer.
    pub fn refuse_unavailable_filters(&self) -> Result<(), &'static str> {
        #[cfg(not(feature = "deflate"))]
        if self
            .filters
            .iter()
            .any(|f| f.kind.filter_id() == FilterId::Deflate)
        {
            return Err("deflate compression requires the `deflate` crate feature");
        }
        Ok(())
    }

    /// Reports a conflicting filter pair as a dataset format error.
    ///
    /// [`hdf5_pure_filter::__private::first_filter_conflict`] selects the pair independently of the
    /// order in which the caller added the filters.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::FilterError`] if the pipeline contains incompatible filters.
    fn refuse_conflicting_filters(&self) -> Result<(), FormatError> {
        let clash = |a: &str, b: &str| {
            Err(FormatError::FilterError(format!(
                "{a} and {b} cannot be combined on one dataset"
            )))
        };

        let Some((first, second)) = hdf5_pure_filter::__private::first_filter_conflict(
            self.filters.iter().map(|f| f.kind.filter_id().get()),
        ) else {
            return Ok(());
        };
        clash(first, second)
    }

    /// Builds filter descriptions in [`Self::filters`] order.
    ///
    /// The writer derives scalar and chunk parameters from `ctx`. Scale-offset records `fill`
    /// when its fill policy requires a value.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::FilterError`] if filters conflict or their parameters are invalid.
    /// Returns [`FormatError::UnreadableFillValue`] if scale-offset requires an unreadable fill
    /// value.
    /// Returns [`FormatError::UnsupportedZfp`] if ZFP cannot encode the element type or chunk
    /// dimensions.
    /// Returns [`FormatError::InvalidZfpRateForScalar`] if the ZFP rate exceeds the scalar width,
    /// or [`FormatError::InvalidZfpBitBudget`] if the encoded block budget is zero.
    /// A filter parameter encoder may return another [`FormatError`].
    pub fn build_pipeline(
        &self,
        ctx: &ChunkContext<'_>,
        fill: FillPattern<'_>,
    ) -> Result<Option<FilterPipeline>, FormatError> {
        self.encode_pipeline(None, ctx, fill)
    }

    pub(crate) fn build_pipeline_with(
        &self,
        scratch: &mut FilterScratch,
        ctx: &ChunkContext<'_>,
        fill: FillPattern<'_>,
    ) -> Result<Option<FilterPipeline>, FormatError> {
        self.encode_pipeline(Some(scratch), ctx, fill)
    }

    fn encode_pipeline(
        &self,
        mut scratch: Option<&mut FilterScratch>,
        ctx: &ChunkContext<'_>,
        fill: FillPattern<'_>,
    ) -> Result<Option<FilterPipeline>, FormatError> {
        #[cfg(not(feature = "zfp"))]
        let _ = scratch.as_mut();
        self.refuse_conflicting_filters()?;

        let element_size = ctx.element_size.get();
        let chunk_dims = ctx.chunk_dims;
        let scale_offset_type = ctx.scale_offset_type;

        let _ = ctx.element_type; // used only under the `zfp` feature below

        // In the order they are stored, which is the order they are applied:
        // `compress_chunk_with` walks the same list, and the reader reverses it.
        // `refuse_conflicting_filters` above has already established that no
        // filter here displaces another.
        let mut filters = Vec::with_capacity(self.filters.len());
        for spec in &self.filters {
            // No flag bit other than H5Z_FLAG_OPTIONAL is written.
            let flags = if spec.optional { H5Z_FLAG_OPTIONAL } else { 0 };
            filters.push(match spec.kind {
                #[cfg(feature = "zfp")]
                FilterKind::Zfp(rate) => zfp::filter_description(
                    ctx,
                    scratch.as_deref_mut(),
                    rate,
                    filters.len(),
                    flags,
                )?,
                FilterKind::ScaleOffset(mode, fill_avail) => {
                    let ty = scale_offset_type.ok_or_else(|| {
                        FormatError::FilterError(
                            "scale-offset requires an integer or floating-point scalar \
                             datatype with a definite (little/big endian) byte order"
                                .into(),
                        )
                    })?;
                    let nelmts: u32 = chunk_dims.iter().product::<u64>().narrow_or_else(|| {
                        FormatError::FilterError("scale-offset: chunk has too many elements".into())
                    })?;
                    FilterDescription {
                        filter_id: FILTER_SCALEOFFSET,
                        name: None,
                        flags,
                        client_data: scaleoffset::build_cd_values(
                            mode,
                            ty,
                            element_size,
                            nelmts,
                            crate::scaleoffset::scale_offset_fill_with_value(fill_avail, fill)?,
                        )?,
                    }
                }
                FilterKind::Shuffle => FilterDescription {
                    filter_id: FILTER_SHUFFLE,
                    name: None,
                    flags,
                    client_data: vec![element_size],
                },
                FilterKind::Lzf => FilterDescription {
                    filter_id: FILTER_LZF,
                    // Ids >= 256 serialize a name; "lzf" is h5py's registered
                    // name. Derived rather than carried from a source pipeline,
                    // since it is a property of the filter, not of the file.
                    name: Some("lzf".into()),
                    // `h5py_cd_values` records the chunk's *raw* byte size as
                    // the decompressed size h5py should expect. That is exact
                    // only where LZF reads the raw elements; behind another
                    // filter it can understate them — behind fletcher32 by the
                    // 4 checksum bytes, behind scale-offset by whatever it
                    // packed to. Neither reader is misled: ours never consults
                    // the value (`inner_output_cap` folds over the real prefix)
                    // and h5py's filter grows its buffer on `E2BIG`. Not new
                    // here — canonical rank already put LZF after scale-offset.
                    flags,
                    client_data: hdf5_pure_filter::__private::lzf_h5py_cd_values(
                        element_size,
                        chunk_dims,
                    )
                    .to_vec(),
                },
                FilterKind::Deflate(level) => FilterDescription {
                    filter_id: FILTER_DEFLATE,
                    name: None,
                    flags,
                    client_data: vec![u32::from(level.get())],
                },
                FilterKind::Fletcher32 => FilterDescription {
                    filter_id: FILTER_FLETCHER32,
                    name: None,
                    flags,
                    client_data: vec![],
                },
            });
        }

        if filters.is_empty() {
            Ok(None)
        } else {
            Ok(Some(FilterPipeline {
                version: 2,
                filters,
            }))
        }
    }

    /// Determine chunk dimensions, using user-specified or auto-computing.
    pub fn resolve_chunk_dims(&self, shape: &[u64]) -> Vec<u64> {
        if let Some(ref dims) = self.chunk_dims {
            dims.clone()
        } else {
            // Auto chunk: use the full dataset shape (single chunk)
            shape.to_vec()
        }
    }

    /// Builds layout and pipeline messages for object-header sizing from dataset metadata.
    ///
    /// The layout is for sizing, with the same encoded length as the final layout. The pipeline
    /// message contains the final filter parameters.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::FilterError`] if filter parameters are invalid or pipeline fields
    /// exceed their encoded widths. Other filter-parameter errors from [`Self::build_pipeline`]
    /// are propagated.
    ///
    /// Returns [`FormatError::ChunkedReadError`] if the chunk-index geometry is unsupported or
    /// exceeds the writer's index limits, or [`FormatError::InvalidChunkGeometry`] if the layout
    /// has too many dimensions.
    ///
    /// Returns [`FormatError::ValueTooLargeForRepresentation`] if a chunk dimension exceeds its
    /// stored width, or [`FormatError::ValueTooLargeForPlatform`] if an element size or chunk
    /// count exceeds [`usize::MAX`].
    pub(crate) fn measure_header(
        &self,
        shape: &[u64],
        ctx: ChunkContext<'_>,
        maxshape: Option<&[MaxExtent]>,
        fill: FillPattern<'_>,
        allocation: StorageAllocation,
    ) -> Result<ChunkedHeader, FormatError> {
        let pipeline = self.build_pipeline(&ctx, fill)?;
        let element_size = ctx.element_size.narrow::<NonZeroUsize>()?;
        let (kind, _, _) = plan_index_slots(
            shape,
            ctx.chunk_dims,
            maxshape,
            full_chunk_bytes(ctx.chunk_dims.iter().copied(), element_size),
            pipeline.is_some(),
            allocation,
        )?;
        let chunk_dims = ctx
            .chunk_dims
            .iter()
            .map(|&dimension| dimension.narrow())
            .collect::<Result<Vec<u32>, _>>()?;
        Ok(ChunkedHeader {
            layout_message: kind.layout_message(
                self.layout_flags(),
                &chunk_dims,
                element_size,
                &[ChunkRecord {
                    address: StoredAddress::new(0),
                    stored_size: 0,
                    filter_mask: 0,
                }],
                StoredAddress::new(0),
                pipeline.is_some(),
            )?,
            pipeline_message: pipeline
                .as_ref()
                .map(FilterPipeline::serialize)
                .transpose()
                .map_err(crate::filter_pipeline::map_filter_pipeline_error)?,
        })
    }

    /// Checks that a dataset under [`PartialEdgeFiltering::Skip`] is stored chunked.
    ///
    /// A dataset is stored chunked where [`is_chunked`](Self::is_chunked) returns `true` or
    /// `maxshape` is set.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::PartialEdgeSkipRequiresChunking`] if the policy is
    /// [`PartialEdgeFiltering::Skip`] and the dataset is stored contiguous.
    pub(crate) fn validate_partial_edge_filtering(
        &self,
        maxshape: Option<&[MaxExtent]>,
    ) -> Result<(), FormatError> {
        match self.partial_edge_filtering {
            Some(PartialEdgeFiltering::Skip) if !self.is_chunked() && maxshape.is_none() => {
                Err(FormatError::PartialEdgeSkipRequiresChunking)
            }
            None | Some(PartialEdgeFiltering::Apply | PartialEdgeFiltering::Skip) => Ok(()),
        }
    }

    /// Checks the chunk geometry of a dataset of `extent`.
    ///
    /// The check is meaningful only for a dataset stored chunked, by
    /// [`is_chunked`](Self::is_chunked) or by a maximum shape.
    /// The chunk dimensions checked are the *resolved* ones, what
    /// [`resolve_chunk_dims`](Self::resolve_chunk_dims) hands the splitter, and
    /// not only the ones the caller named. Auto-chunking derives them from the
    /// shape, so a shape that is invalid to chunk by itself is caught here,
    /// before the division one layer down. A zero-element shape, `[0]`
    /// for an empty extensible dataset, passes with explicit chunk dimensions: it
    /// is not scalar and produces zero chunks, which is well-formed.
    ///
    /// # Errors
    ///
    /// Returns a reason for the first problem found, for the caller to map to its
    /// own error type: a scalar shape, chunk dimensions of another rank than the
    /// shape, a zero chunk dimension, a zero-element shape with no explicit chunk
    /// dimensions, or more than one unlimited dimension.
    pub(crate) fn validate_geometry(&self, extent: Extent<'_>) -> Result<(), &'static str> {
        let shape = extent.dims();
        if shape.is_empty() {
            return Err("a scalar dataset cannot be chunked, filtered, or extensible");
        }
        // Explicit chunk dimensions must match the shape's rank and be non-zero;
        // a zero would divide-by-zero when counting chunks per dimension, and a
        // rank mismatch would index past the end of `chunk_dims`.
        if let Some(dims) = self.chunk_dims.as_deref() {
            if dims.len() != shape.len() {
                return Err("chunk dimensions must have the same rank as the dataset shape");
            }
            if dims.contains(&0) {
                return Err("chunk dimensions must all be non-zero");
            }
        } else if shape.contains(&0) {
            // Auto-chunking makes the whole shape one chunk, which for a
            // zero-element shape is a zero chunk dimension — the case rejected
            // just above, arrived at from the other direction. There is nothing
            // in the shape to derive a size from, so the caller has to name one.
            return Err(
                "a zero-element dataset must be given explicit chunk dimensions \
                 (its shape has none to derive)",
            );
        }
        // libhdf5 indexes a dataspace with two unlimited
        // dimensions using a version-2 B-tree, which this crate does not
        // write. An Extensible Array cannot number such a dataspace at all,
        // since it has only one dimension it can grow along. Writing one
        // anyway writes a dataset the C library fails to open ("already found
        // unlimited dimension"), so this check returns an error first
        // (issue #299).
        if extent.unlimited_dimensions() > 1 {
            return Err(
                "at most one dimension of a maxshape may be unlimited; the chunk index \
                 for more than one is a version-2 B-tree, which this crate cannot write",
            );
        }
        Ok(())
    }

    /// Returns the data layout flags of the partial edge filtering policy, or of
    /// [`PartialEdgeFiltering::Apply`] where the caller sets none.
    fn layout_flags(&self) -> ChunkedLayoutFlags {
        self.partial_edge_filtering
            .unwrap_or_default()
            .layout_flags()
    }
}

/// Whether the writer runs the partial edge chunks of a created dataset through the filter
/// pipeline.
///
/// Under [`Skip`](Self::Skip) the writer sets the `DONT_FILTER_PARTIAL_BOUND_CHUNKS` flag of the
/// version 4 data layout message, defined in "The Data Layout Message" of the [format
/// specification, version 4.0][spec]. libhdf5 sets the flag with `H5Pset_chunk_opts` and
/// `H5D_CHUNK_DONT_FILTER_PARTIAL_CHUNKS`.
///
/// [`DatasetCreateProperties::with_partial_edge_filtering`] sets the policy of a dataset.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout
/// [`DatasetCreateProperties::with_partial_edge_filtering`]: crate::DatasetCreateProperties::with_partial_edge_filtering
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PartialEdgeFiltering {
    /// The filter pipeline applies to every chunk.
    #[default]
    Apply,
    /// A partial edge chunk is stored as its data, padded to a whole chunk with the fill value.
    Skip,
}

impl PartialEdgeFiltering {
    /// Returns the policy of a dataset whose data layout message stores `flags`.
    #[cfg(feature = "std")]
    pub(crate) fn from_layout_flags(flags: ChunkedLayoutFlags) -> Self {
        if flags.partial_edge_chunks_stored_raw() {
            Self::Skip
        } else {
            Self::Apply
        }
    }

    /// Returns the data layout flags for this policy.
    pub(crate) fn layout_flags(self) -> ChunkedLayoutFlags {
        match self {
            Self::Apply => ChunkedLayoutFlags::NONE,
            Self::Skip => ChunkedLayoutFlags::new(DONT_FILTER_PARTIAL_BOUND_CHUNKS),
        }
    }
}

/// The most index bytes this writer will emit for element slots that hold no
/// chunk.
///
/// A chunk index is built as one in-memory buffer and copied into the file image,
/// measured at about 2.2x the index bytes in peak memory and 4 ms per MiB, so a
/// geometry refused at this budget was about to cost 0.13 s and 70 MB to produce
/// a file whose index is mostly empty.
///
/// Stated in bytes rather than slots because an element is 8 bytes unfiltered and
/// 15 to 20 filtered: the slot count this replaced allowed a filtered index two
/// and a half times the bytes of an unfiltered one, which nobody chose. 32 MiB is
/// the unfiltered value that bound already had (4,194,304 slots x 8 bytes).
///
/// A *Fixed* Array is dense by format — the reference library writes the same
/// 80 MB for a 10-million-slot one — so for that index this is a bound on what
/// this writer is willing to hold in memory, not a divergence from the reference.
/// An Extensible Array allocates only the blocks its chunks land in, so its
/// unused bytes are the slack inside those blocks.
const MAX_UNUSED_INDEX_BYTES: u64 = 32 << 20;

/// Result of building a chunked dataset.
pub struct ChunkedDataResult {
    /// Raw bytes containing all chunk data + index structures.
    pub data_bytes: Vec<u8>,
    /// The DataLayout v4 message bytes.
    pub layout_message: Vec<u8>,
    /// The FilterPipeline message bytes, if any.
    pub pipeline_message: Option<Vec<u8>>,
}

/// Collects full chunk buffers in dense row-major order.
///
/// Partial edge chunks are padded with `fill`. Copies available whole elements from `raw_data`
/// into their chunk slots. Missing input within a full chunk remains zero. Returns one copy
/// of `raw_data` for a scalar shape and an empty collection for a zero extent.
///
/// # Errors
///
/// Returns [`FormatError::InvalidChunkGeometry`] for a non-scalar shape if the chunk rank
/// differs, a chunk dimension is zero, or a derived count, stride, or byte length exceeds
/// [`u64::MAX`]. Returns [`FormatError::ValueTooLargeForPlatform`] if the chunk count or full
/// chunk byte length exceeds [`usize::MAX`]. Returns [`FormatError::UnreadableFillValue`] if
/// an edge chunk needs padding and `fill` is unknown.
pub fn split_into_chunks(
    raw_data: &[u8],
    shape: &[u64],
    chunk_dims: &[u64],
    element_size: NonZeroUsize,
    fill: FillPattern<'_>,
) -> Result<Vec<Vec<u8>>, FormatError> {
    if shape.is_empty() {
        return Ok(vec![raw_data.to_vec()]);
    }
    let mut traversal = gathering::ChunkTraversal::new(shape, chunk_dims, element_size)?;
    let mut buffers = Vec::with_capacity(traversal.chunk_count().narrow()?);
    traversal.visit(|chunk| {
        let mut bytes = vec![0; chunk.byte_len()];
        chunk.gather(raw_data, fill, &mut bytes)?;
        buffers.push(bytes);
        Ok(())
    })?;
    Ok(buffers)
}

/// The unfiltered byte size of one whole chunk: the product of the chunk
/// dimensions and the element size.
///
/// One derivation, shared by everything that has to size a chunk-index element
/// ([`chunk_element_encoding`](hdf5_pure_format::__private::chunk_element_encoding)): the
/// buffered builder, the verbatim planner,
/// and the in-place append's index rebuild. Each of those holds the geometry in
/// a different shape (`u32` dimensions from a set, `u64` from a plan), and a
/// second copy of this product is how one of them ends up declaring a width the
/// others do not write.
///
/// The product saturates so that a malformed geometry cannot wrap into a small, plausible
/// width.
pub(crate) fn full_chunk_bytes(
    chunk_dims: impl IntoIterator<Item = u64>,
    element_size: NonZeroUsize,
) -> u64 {
    chunk_dims
        .into_iter()
        .fold(element_size.get() as u64, |acc, d| acc.saturating_mul(d))
}

/// Whether the writer allocates element storage for the declared dataspace.
///
/// An unallocated dataset can have a nonempty shape. Reads return the fill value for elements
/// without storage. Callers select the allocation policy independently of the shape and payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StorageAllocation {
    /// Storage covers the dataspace: a chunk in every grid slot the shape
    /// implies, or a contiguous run of every element's bytes.
    Allocated,
    /// Declares the dataspace without allocating chunks or contiguous element bytes.
    ///
    /// Contiguous and fixed-grid chunked layouts store an undefined address. An unlimited
    /// chunked dataset with zero elements also has an undefined index address. For a nonempty
    /// unlimited dataspace, the writer allocates an empty Extensible Array index.
    Unallocated,
}

/// The chunk indexing type and allocation state used for layout sizing and emission.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ChunkIndexKind {
    /// An Extensible Array, for a dataset with an unlimited dimension.
    ExtensibleArray,
    /// A Fixed Array, for a finite maximum chunk grid with more than one slot.
    FixedArray,
    /// A single chunk whose address is stored in the layout message.
    SingleChunk,
    /// An index type stored with an undefined address and no allocated index structure.
    Unallocated(ChunkArrayKind),
}

/// The array indexing types that have a separate on-disk structure when allocated.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ChunkArrayKind {
    FixedArray,
    ExtensibleArray,
}

impl ChunkArrayKind {
    /// Serializes an array index using the writer's address and length widths.
    ///
    /// # Errors
    ///
    /// Propagates array-index serialization errors.
    fn build_at(
        self,
        slots: &IndexSlots<'_>,
        chunk_bytes: u64,
        has_filters: bool,
        address: StoredAddress,
    ) -> Result<Vec<u8>, FormatError> {
        match self {
            Self::ExtensibleArray => hdf5_pure_format::__private::build_extensible_array_at(
                slots,
                chunk_bytes,
                INDEX_OFFSET_SIZE,
                INDEX_LENGTH_SIZE,
                has_filters,
                address,
            ),
            Self::FixedArray => hdf5_pure_format::__private::build_fixed_array_at(
                slots,
                chunk_bytes,
                INDEX_OFFSET_SIZE,
                INDEX_LENGTH_SIZE,
                has_filters,
                address,
            ),
        }
    }
}

impl ChunkIndexKind {
    /// Returns the array type to serialize, or [`None`] for unallocated and single-chunk layouts.
    pub(crate) fn array_kind(self) -> Option<ChunkArrayKind> {
        match self {
            Self::Unallocated(_) | Self::SingleChunk => None,
            Self::FixedArray => Some(ChunkArrayKind::FixedArray),
            Self::ExtensibleArray => Some(ChunkArrayKind::ExtensibleArray),
        }
    }

    /// Encodes the version 4 Data Layout message for this index type and allocation state.
    ///
    /// A single-chunk layout stores the address of the first of `chunks`. An unallocated layout
    /// stores its array type and an undefined address, as defined in "The Data Layout Message"
    /// of the [format specification, version 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if a single-chunk layout has no chunk or `element_size`
    /// exceeds [`u32::MAX`]. Propagates layout serialization errors.
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout
    fn layout_message(
        self,
        flags: ChunkedLayoutFlags,
        chunk_dims: &[u32],
        element_size: NonZeroUsize,
        chunks: &[ChunkRecord],
        index_address: StoredAddress,
        has_filters: bool,
    ) -> Result<Vec<u8>, FormatError> {
        let (info, address) = match self {
            Self::Unallocated(kind) => (
                match kind {
                    ChunkArrayKind::FixedArray => ChunkIndexInfo::FixedArray,
                    ChunkArrayKind::ExtensibleArray => ChunkIndexInfo::ExtensibleArray,
                },
                StoredAddress::new(HADDR_UNDEF),
            ),
            Self::SingleChunk => {
                let chunk = chunks.first().ok_or_else(|| {
                    FormatError::Internal("a single-chunk layout was planned with no chunk".into())
                })?;
                let filtered = has_filters.then_some(FilteredSingleChunk {
                    filtered_size: chunk.stored_size,
                    filter_mask: chunk.filter_mask,
                });
                (ChunkIndexInfo::SingleChunk { filtered }, chunk.address)
            }
            Self::FixedArray => (ChunkIndexInfo::FixedArray, index_address),
            Self::ExtensibleArray => (ChunkIndexInfo::ExtensibleArray, index_address),
        };
        let element_size: u32 = element_size.get().narrow_or_else(|| {
            FormatError::Internal(
                "a chunked element size wider than its u32 dimension field".into(),
            )
        })?;
        DataLayout::encode_chunked(
            flags,
            chunk_dims,
            element_size,
            info,
            address,
            INDEX_OFFSET_SIZE,
            INDEX_LENGTH_SIZE,
        )
    }
}

/// Decide which index a chunk set gets, from the grid its slots are numbered
/// over and the number of chunks stored.
///
/// The grid decides everything except the chunk-less case, and it has to: the
/// reference library reads a single-chunk layout only where the *maximum* shape
/// is one chunk, and asserts (`H5D__single_idx_get_addr`) on a file that names
/// one where the dataset could hold more. A dataset of one chunk that is allowed
/// to grow into a second therefore gets a Fixed Array, not the layout its one
/// chunk would suggest.
pub(crate) fn chunk_index_kind(grid: &ChunkGrid, num_chunks: usize) -> ChunkIndexKind {
    match grid.slots() {
        // An unlimited dimension: no fixed slot count, so the extensible index.
        None => ChunkIndexKind::ExtensibleArray,
        _ if num_chunks == 0 => ChunkIndexKind::Unallocated(ChunkArrayKind::FixedArray),
        Some(1) => ChunkIndexKind::SingleChunk,
        Some(_) => ChunkIndexKind::FixedArray,
    }
}

/// The byte length the chunk index of `kind` would occupy for `chunks`, without
/// building it.
///
/// Both index structures place their length beside the builder that emits it
/// ([`extensible_array_len`](hdf5_pure_format::__private::extensible_array_len),
/// [`fixed_array_len`](hdf5_pure_format::__private::fixed_array_len)), so a caller reserving the
/// data region's span takes it from the same layout the emission works from.
pub(crate) fn chunk_index_len(
    kind: ChunkArrayKind,
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    length_size: LengthWidth,
    has_filters: bool,
) -> u64 {
    match kind {
        ChunkArrayKind::ExtensibleArray => hdf5_pure_format::__private::extensible_array_len(
            slots,
            chunk_bytes,
            offset_size,
            length_size,
            has_filters,
        ),
        ChunkArrayKind::FixedArray => hdf5_pure_format::__private::fixed_array_len(
            slots,
            chunk_bytes,
            offset_size,
            length_size,
            has_filters,
        ),
    }
}

/// A chunked dataset's encoded chunks and address-independent index properties.
///
/// The writer retains the chunks across sizing and placement. [`measure_chunked_at`] computes
/// the region length and object-header messages, and [`plan_chunked_at`] finalizes the metadata
/// for emission at a stored address. [`emit_chunked_data`] borrows the encoded bytes from this set.
pub(crate) struct CompressedChunkSet {
    /// The chunks as the writer stores them, in dense row-major grid order.
    chunks: Vec<PreparedChunk>,
    chunk_dims_u32: Vec<u32>,
    element_size: NonZeroUsize,
    /// The layout flags of [`ChunkOptions::partial_edge_filtering`], which the layout message
    /// stores with the bit of a filtered Single Chunk index added.
    flags: ChunkedLayoutFlags,
    has_filters: bool,
    /// Which index this set gets, decided once from the maximum chunk grid.
    kind: ChunkIndexKind,
    /// The element slot each chunk of `chunks` occupies, in the same order.
    /// Identical to `0..chunks.len()` unless the maximum shape is wider than
    /// the shape in some dimension past the first.
    slot_of_chunk: Vec<u64>,
    /// How many slots the index spans; see [`IndexSlots`].
    index_slots: u64,
    pipeline_message: Option<Vec<u8>>,
}

impl CompressedChunkSet {
    /// Returns scalar index geometry for an array containing no stored chunks.
    ///
    /// Returns [`None`] for unallocated and single-chunk layouts, or if the set holds chunks
    /// or chunk-slot assignments.
    pub(crate) fn empty_index(&self) -> Option<EmptyChunkIndex> {
        let Self {
            chunks,
            chunk_dims_u32,
            element_size,
            flags: _,
            has_filters,
            kind,
            slot_of_chunk,
            index_slots,
            pipeline_message: _,
        } = self;
        if !chunks.is_empty() || !slot_of_chunk.is_empty() {
            return None;
        }
        Some(EmptyChunkIndex {
            kind: kind.array_kind()?,
            chunk_bytes: full_chunk_bytes(
                chunk_dims_u32.iter().map(|&dimension| u64::from(dimension)),
                *element_size,
            ),
            slots: *index_slots,
            has_filters: *has_filters,
        })
    }

    /// Where this set's chunks sit in its chunk index, given the addresses a
    /// layout pass assigned them.
    ///
    /// Built at each use rather than stored, because the view borrows the
    /// addresses and those change from pass to pass; it costs no allocation for
    /// a set whose chunks fill their slots in order.
    fn index_slots<'a>(
        &self,
        written_chunks: &'a [ChunkRecord],
    ) -> Result<IndexSlots<'a>, FormatError> {
        IndexSlots::new(written_chunks, &self.slot_of_chunk, self.index_slots)
    }

    /// This set's whole-chunk byte size; see [`full_chunk_bytes`].
    fn full_chunk_bytes(&self) -> u64 {
        full_chunk_bytes(
            self.chunk_dims_u32.iter().map(|&d| u64::from(d)),
            self.element_size,
        )
    }
}

/// The scalar geometry of a chunk array containing no stored chunks.
#[derive(Clone, Copy)]
pub(crate) struct EmptyChunkIndex {
    kind: ChunkArrayKind,
    /// The unfiltered byte length of a full chunk.
    chunk_bytes: u64,
    /// The logical index span, including unallocated slots.
    slots: u64,
    /// Whether index records include stored sizes and filter masks.
    has_filters: bool,
}

impl EmptyChunkIndex {
    /// Serializes the empty array index at its assigned stored address.
    ///
    /// # Errors
    ///
    /// Propagates chunk-index slot and array serialization errors.
    pub(crate) fn build_at(self, address: StoredAddress) -> Result<Vec<u8>, FormatError> {
        let Self {
            kind,
            chunk_bytes,
            slots,
            has_filters,
        } = self;
        kind.build_at(
            &IndexSlots::new(&[], &[], slots)?,
            chunk_bytes,
            has_filters,
            address,
        )
    }
}

/// Splits `raw_data` into chunks and encodes each one for storage, the dataset's only pass of the
/// filter pipeline.
///
/// A partial edge chunk is stored as its data where [`ChunkOptions::partial_edge_filtering`] holds
/// [`PartialEdgeFiltering::Skip`]. [`assemble_chunked_at`] lays the set out at an address without
/// encoding the chunks again. An unallocated dataset has no chunks.
///
/// `fill` is the configured fill value used in filter parameters. The writer initializes partial
/// edge chunks with zeros under [`FillTime::Never`] and with `fill` under the other policies
/// before copying available elements.
///
/// # Errors
///
/// Returns [`FormatError::ChunkedReadError`] if the chunk index of the geometry is unsupported or
/// exceeds the writer's index limits. Returns [`FormatError::InvalidChunkGeometry`] if a chunk
/// is larger than [`u32::MAX`] bytes, including a chunk of an unallocated dataset. Returns the
/// errors of [`ChunkOptions::build_pipeline`] for the filters, and of
/// [`PreparedChunk::encode_dataset`] for the chunks.
pub(crate) fn compress_chunks(
    raw_data: &[u8],
    shape: &[u64],
    ctx: ChunkContext<'_>,
    options: &ChunkOptions,
    maxshape: Option<&[MaxExtent]>,
    fill: FillPattern<'_>,
    fill_time: FillTime,
    allocation: StorageAllocation,
) -> Result<CompressedChunkSet, FormatError> {
    let chunk_dims = ctx.chunk_dims;
    let element_size = ctx.element_size.narrow::<NonZeroUsize>()?;
    crate::chunked_read::bounded_chunk_len(chunk_dims, element_size)?;
    let mut scratch = FilterScratch::new();
    let pipeline = match allocation {
        StorageAllocation::Allocated => options.build_pipeline_with(&mut scratch, &ctx, fill)?,
        StorageAllocation::Unallocated if options.unallocated_pipeline.is_some() => None,
        StorageAllocation::Unallocated => options.build_pipeline(&ctx, fill)?,
    };
    let pipeline_message = match allocation {
        StorageAllocation::Unallocated if options.unallocated_pipeline.is_some() => {
            options.unallocated_pipeline.clone()
        }
        _ => pipeline
            .as_ref()
            .map(FilterPipeline::serialize)
            .transpose()
            .map_err(crate::filter_pipeline::map_filter_pipeline_error)?,
    };

    // Decided from the geometry alone, and decided first: it is the step that
    // refuses a maximum shape needing an index this writer will not emit, and
    // refusing after splitting and compressing the whole dataset would do all
    // that work to reach the same answer.
    let (kind, slot_of_chunk, index_slots) = plan_index_slots(
        shape,
        chunk_dims,
        maxshape,
        full_chunk_bytes(chunk_dims.iter().copied(), element_size),
        pipeline_message.is_some(),
        allocation,
    )?;

    let flags = options.layout_flags();

    // An unallocated dataset is not split. The splitter emits a chunk for every
    // slot the shape implies and this dataset supplies no bytes for them, so it
    // would write out the whole grid — and write it wrong, since a chunk that
    // lies inside the shape is zero-padded where the data runs out and only one
    // overhanging the edge takes the fill pattern. `plan_index_slots` already
    // answered for the same `allocation`, so the assertion below still pairs the
    // two.
    //
    // One encoder for every chunk of the dataset. Building one per chunk is the
    // dominant cost of a filtered write -- ~300 KiB of hash tables apiece, 615
    // MiB over an 8 MiB dataset (issue #228).
    let chunks = match allocation {
        StorageAllocation::Allocated => PreparedChunk::encode_dataset(
            &mut scratch,
            pipeline.as_ref(),
            ctx,
            flags,
            shape,
            if fill_time.is_never() {
                FillPattern::ZERO
            } else {
                fill
            },
            raw_data,
        )?,
        StorageAllocation::Unallocated => Vec::new(),
    };
    let has_filters = pipeline_message.is_some();
    debug_assert_eq!(slot_of_chunk.len(), chunks.len());

    let mut chunk_dims_u32 = Vec::with_capacity(chunk_dims.len());
    for &dimension in chunk_dims {
        chunk_dims_u32.push(dimension.narrow()?);
    }

    Ok(CompressedChunkSet {
        chunks,
        chunk_dims_u32,
        element_size,
        flags,
        has_filters,
        kind,
        slot_of_chunk,
        index_slots,
        pipeline_message,
    })
}

/// Plans the indexing type, each chunk's element slot in dense grid order, and the index span.
///
/// With [`StorageAllocation::Unallocated`], the plan contains no chunks. Fixed-grid datasets and
/// zero-element unlimited datasets have no index allocation. A nonempty unlimited dataset has
/// an empty Extensible Array index.
///
/// # Errors
///
/// Returns [`FormatError::ChunkedReadError`] if the index exceeds its addressable capacity or the
/// unused-element byte budget. Propagates chunk-grid and platform-size conversion errors.
pub(crate) fn plan_index_slots(
    shape: &[u64],
    chunk_dims: &[u64],
    maxshape: Option<&[MaxExtent]>,
    chunk_bytes: u64,
    has_filters: bool,
    allocation: StorageAllocation,
) -> Result<(ChunkIndexKind, Vec<u64>, u64), FormatError> {
    let grid = index_grid(shape, chunk_dims, maxshape)?;
    let counts: Vec<u64> = shape
        .iter()
        .zip(chunk_dims)
        .map(|(d, c)| d.div_ceil(*c))
        .collect();
    // The grid the shape implies, or none of it.
    //
    // Note what this does to the two *budget* refusals further down. Both are
    // driven by the slots the index spans, an unallocated dataset's index spans
    // none, and so neither can fire for one however wide its maximum shape --
    // measured, on a maximum shape that is refused outright once a single chunk
    // is written. That is the intent rather than a gap: the index costs nothing
    // until something is stored, and refusing to repack a file the reference
    // library wrote happily would be the worse answer. The budgets apply again
    // at the first chunk. The geometry `index_grid` itself refuses is a separate
    // matter and is unaffected -- it never sees the count.
    let num_chunks = match allocation {
        StorageAllocation::Allocated => counts.iter().product::<u64>().to_usize()?,
        StorageAllocation::Unallocated => 0,
    };
    let kind = if allocation == StorageAllocation::Unallocated
        && shape.contains(&0)
        && grid.slots().is_none()
    {
        ChunkIndexKind::Unallocated(ChunkArrayKind::ExtensibleArray)
    } else {
        chunk_index_kind(&grid, num_chunks)
    };

    let mut slot_of_chunk = Vec::with_capacity(num_chunks);
    let mut coords = vec![0u64; shape.len()];
    for dense in 0..num_chunks {
        let mut remaining = dense as u64;
        for d in (0..shape.len()).rev() {
            coords[d] = remaining % counts[d];
            remaining /= counts[d];
        }
        slot_of_chunk.push(grid.slot_of(&coords)?);
    }

    // A Fixed Array declares a slot for every chunk the maximum shape allows;
    // an Extensible Array only has to reach the last one written, and grows from
    // there as the dataset does. A layout with no positional index spans no
    // slots at all: the single-chunk layout carries its address in the layout
    // message, and an unallocated one has no address to carry.
    let index_slots = match kind {
        ChunkIndexKind::Unallocated(_) | ChunkIndexKind::SingleChunk => 0,
        ChunkIndexKind::FixedArray => grid.slots().ok_or_else(|| {
            FormatError::ChunkedReadError(
                "a Fixed Array cannot index an unlimited dimension".into(),
            )
        })?,
        _ => slot_of_chunk.iter().max().map_or(0, |s| s + 1),
    };
    // What the index will actually emit for slots that hold nothing. For a Fixed
    // Array that is every slot but the chunks'; for an Extensible Array it is the
    // slack inside the blocks the chunks land in, since the others are not
    // written at all — so this has to come from the layout rather than from the
    // span, which for a sparse array overstates it by orders of magnitude.
    // Sorted already unless the rotation scattered them, which only an
    // Extensible Array past its first dimension does — so the ordinary write
    // borrows the plan instead of copying it.
    let scratch: Vec<u64> = if slot_of_chunk.windows(2).all(|w| w[0] <= w[1]) {
        Vec::new()
    } else {
        let mut v = slot_of_chunk.clone();
        v.sort_unstable();
        v
    };
    let sorted_slots: &[u64] = if scratch.is_empty() {
        &slot_of_chunk
    } else {
        &scratch
    };
    let encoding = hdf5_pure_format::__private::chunk_element_encoding(
        chunk_bytes,
        INDEX_OFFSET_SIZE,
        has_filters,
    );
    let allocated = match kind {
        ChunkIndexKind::Unallocated(_) | ChunkIndexKind::SingleChunk => 0,
        ChunkIndexKind::FixedArray => index_slots,
        ChunkIndexKind::ExtensibleArray => {
            // An Extensible Array with this crate's creation parameters can
            // address only so many slots; a chunk numbered past the last block
            // lands in no block at all, and the writer would drop it silently
            // while the reference library fails the read ("ring type mismatch
            // occurred for cache entry"). Refused here rather than left to the
            // byte budget, which no longer covers it now that an empty block
            // costs nothing (issue #299).
            let capacity = hdf5_pure_format::__private::extensible_array_capacity();
            if index_slots > capacity {
                return Err(FormatError::ChunkedReadError(format!(
                    "this shape and maximum shape number a chunk at element {} of the chunk \
                     index, past the {capacity} an extensible array can address; the chunk would \
                     be dropped. Chunk the dimensions the dataset does not grow along more \
                     coarsely, or give them a smaller maximum",
                    index_slots - 1,
                )));
            }
            hdf5_pure_format::__private::extensible_array_layout(
                SlotOccupancy::Slots(sorted_slots),
                index_slots,
                chunk_bytes,
                INDEX_OFFSET_SIZE,
                INDEX_LENGTH_SIZE,
                has_filters,
            )
            .stats
            .nelmts
        }
    };
    let unused_bytes = allocated
        .saturating_sub(num_chunks as u64)
        .saturating_mul(encoding.elem_size as u64);
    if unused_bytes > MAX_UNUSED_INDEX_BYTES {
        return Err(FormatError::ChunkedReadError(format!(
            "this shape and maximum shape need a chunk index holding {allocated} elements for \
             {num_chunks} chunk(s), so {unused_bytes} bytes of it describe no chunk, past the \
             {MAX_UNUSED_INDEX_BYTES} this writer will emit. The unused elements come from the \
             maximum shape exceeding the shape in a dimension other than the one the dataset \
             grows along — chunk those dimensions more coarsely, or declare no maximum for them"
        )));
    }
    Ok((kind, slot_of_chunk, index_slots))
}

/// The chunk grid a written dataset's index numbers its slots over.
///
/// The read side builds the same grid from the dataspace it parsed
/// (`chunked_read::index_grid`); this is the writer's half of the same rule, and
/// the two pick the same [`GridOrder`] from the same fact — whether the maximum
/// shape names an unlimited dimension.
fn index_grid(
    shape: &[u64],
    chunk_dims: &[u64],
    maxshape: Option<&[MaxExtent]>,
) -> Result<ChunkGrid, FormatError> {
    let order = if maxshape.is_some_and(|ms| ms.contains(&MaxExtent::Unlimited)) {
        GridOrder::UnlimitedFirst
    } else {
        GridOrder::RowMajor
    };
    ChunkGrid::new(chunk_dims, shape, maxshape, order)
}

/// Assigns consecutive stored addresses to the chunks and returns the following index address.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if a chunk extends past [`u64::MAX`].
fn plan_chunk_slots(
    set: &CompressedChunkSet,
    data_address: StoredAddress,
) -> Result<(Vec<ChunkRecord>, StoredAddress), FormatError> {
    let mut cursor = data_address;
    let mut written_chunks = Vec::with_capacity(set.chunks.len());
    for chunk in &set.chunks {
        written_chunks.push(ChunkRecord {
            address: cursor,
            stored_size: chunk.stored_size(),
            filter_mask: chunk.filter_mask(),
        });
        cursor = StoredAddress::new(cursor.get().checked_add(chunk.stored_size()).ok_or(
            FormatError::OffsetOverflow {
                offset: cursor.get(),
                length: chunk.stored_size(),
            },
        )?);
    }
    Ok((written_chunks, cursor))
}

/// Build the chunk index that follows a chunk set's data at `index_address`,
/// returning the index bytes — empty for the single-chunk layout, whose chunk
/// address lives in the layout message instead — and the v4 data-layout message
/// naming it.
///
/// Every address either one embeds sits in a fixed-width field, so the returned
/// bytes have the same *length* for every `index_address` — which is what lets
/// the writer size an object header in one pass and emit it in a later one, and
/// what makes the length [`chunk_index_len`] derives valid at any address.
/// [`chunked_data_len`] takes that derived length rather than calling this, so a
/// caller sizing a dataset before its address is chosen builds nothing.
fn chunk_index_bytes(
    set: &CompressedChunkSet,
    written_chunks: &[ChunkRecord],
    index_address: StoredAddress,
) -> Result<(Vec<u8>, Vec<u8>), FormatError> {
    let index = match set.empty_index() {
        Some(index) => index.build_at(index_address)?,
        None => match set.kind.array_kind() {
            Some(kind) => kind.build_at(
                &set.index_slots(written_chunks)?,
                set.full_chunk_bytes(),
                set.has_filters,
                index_address,
            )?,
            None => Vec::new(),
        },
    };
    Ok((
        index,
        chunk_index_layout(set, written_chunks, index_address)?,
    ))
}

/// The data-layout message for `set`'s index at `index_address`.
///
/// Split out of [`chunk_index_bytes`] because sizing an object header needs the
/// message and not the index: the message names the index's address, which is
/// known before a byte of it is built. Building one to size the other is the
/// defect issues #265 and #275 removed a level up, and this is the same one a
/// level down.
fn chunk_index_layout(
    set: &CompressedChunkSet,
    written_chunks: &[ChunkRecord],
    index_address: StoredAddress,
) -> Result<Vec<u8>, FormatError> {
    set.kind.layout_message(
        set.flags,
        &set.chunk_dims_u32,
        set.element_size,
        written_chunks,
        index_address,
        set.has_filters,
    )
}

/// Returns the chunked data-region length before a placement address is chosen.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if the region length exceeds [`u64::MAX`]. Propagates
/// chunk-index slot errors.
pub(crate) fn chunked_data_len(set: &CompressedChunkSet) -> Result<u64, FormatError> {
    let data_address = StoredAddress::new(0);
    let (written_chunks, index_address) = plan_chunk_slots(set, data_address)?;
    chunked_region_len(set, &written_chunks, data_address, index_address)
}

/// A chunked data region's finalized index and object-header messages.
pub(crate) struct ChunkedPlan {
    /// The index bytes emitted after the encoded chunks.
    index: Vec<u8>,
    /// The total length of the encoded chunks and index.
    pub data_len: u64,
    /// The version 4 data-layout message for the object header.
    pub layout_message: Vec<u8>,
    /// The filter-pipeline message, if the dataset has one.
    pub pipeline_message: Option<Vec<u8>>,
}

/// Finalizes chunk addresses, index bytes, and object-header messages at `data_address`.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if a chunk or the index extends past [`u64::MAX`].
/// Propagates chunk-index and layout serialization errors.
pub(crate) fn plan_chunked_at(
    set: &CompressedChunkSet,
    data_address: StoredAddress,
) -> Result<ChunkedPlan, FormatError> {
    let (written_chunks, index_address) = plan_chunk_slots(set, data_address)?;
    let data_len = chunked_region_len(set, &written_chunks, data_address, index_address)?;
    let (index, layout_message) = chunk_index_bytes(set, &written_chunks, index_address)?;
    Ok(ChunkedPlan {
        index,
        data_len,
        layout_message,
        pipeline_message: set.pipeline_message.clone(),
    })
}

/// Writes retained encoded chunks followed by the finalized chunk index.
///
/// The caller pairs `plan` with the [`CompressedChunkSet`] used to create it and positions `sink`
/// at the region's start.
///
/// # Errors
///
/// Returns [`EmissionError::Sink`] with the error from [`ByteSink::put`] if a write fails.
pub(crate) fn emit_chunked_data<S: ByteSink>(
    sink: &mut S,
    set: &CompressedChunkSet,
    plan: &ChunkedPlan,
) -> Result<(), EmissionError<S::Error>> {
    for chunk in &set.chunks {
        sink.put(chunk.bytes()).map_err(EmissionError::Sink)?;
    }
    sink.put(&plan.index).map_err(EmissionError::Sink)
}

/// Returns the length from `data_address` through the end of the chunk index.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if the index extends past [`u64::MAX`]. Returns
/// [`FormatError::Internal`] if the region ends before `data_address`. Propagates chunk-index
/// slot errors.
fn chunked_region_len(
    set: &CompressedChunkSet,
    written_chunks: &[ChunkRecord],
    data_address: StoredAddress,
    index_address: StoredAddress,
) -> Result<u64, FormatError> {
    let index_len = match set.kind.array_kind() {
        Some(array) => chunk_index_len(
            array,
            &set.index_slots(written_chunks)?,
            set.full_chunk_bytes(),
            INDEX_OFFSET_SIZE,
            INDEX_LENGTH_SIZE,
            set.has_filters,
        ),
        None => 0,
    };
    let end = index_address
        .get()
        .checked_add(index_len)
        .ok_or(FormatError::OffsetOverflow {
            offset: index_address.get(),
            length: index_len,
        })?;
    end.checked_sub(data_address.get()).ok_or_else(|| {
        FormatError::Internal("a chunked region ends before its data address".into())
    })
}

/// Computes a chunked region's length and object-header messages without serializing its index.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if a chunk or the index extends past [`u64::MAX`].
/// Propagates chunk-index slot and layout serialization errors.
pub(crate) fn measure_chunked_at(
    set: &CompressedChunkSet,
    data_address: StoredAddress,
) -> Result<ChunkedMeasure, FormatError> {
    let (written_chunks, index_address) = plan_chunk_slots(set, data_address)?;
    let data_len = chunked_region_len(set, &written_chunks, data_address, index_address)?;
    Ok(ChunkedMeasure {
        data_len,
        layout_message: chunk_index_layout(set, &written_chunks, index_address)?,
        pipeline_message: set.pipeline_message.clone(),
    })
}

/// Everything [`ChunkedDataResult`] carries except the data region: what a caller
/// sizing an object header needs, and no more.
pub(crate) struct ChunkedMeasure {
    /// Bytes the data region will occupy.
    pub data_len: u64,
    /// The v4 data-layout message for the object header.
    pub layout_message: Vec<u8>,
    /// The filter-pipeline message, if the dataset has one.
    pub pipeline_message: Option<Vec<u8>>,
}

/// The messages needed to size a chunked dataset's object header before encoding its elements.
pub(crate) struct ChunkedHeader {
    /// The layout message used for its encoded length during object-header sizing.
    pub layout_message: Vec<u8>,
    /// The final filter-pipeline message, if the dataset has one.
    pub pipeline_message: Option<Vec<u8>>,
}

/// Materializes encoded chunks and their index in a contiguous data-region buffer.
///
/// Uses [`plan_chunked_at`] and [`emit_chunked_data`] at `data_address`. The in-place editor
/// requires an owned buffer for relocation.
///
/// # Errors
///
/// Returns [`FormatError::ValueTooLargeForPlatform`] if the region length exceeds [`usize::MAX`].
/// Propagates the errors of [`plan_chunked_at`].
pub(crate) fn assemble_chunked_at(
    set: &CompressedChunkSet,
    data_address: StoredAddress,
) -> Result<ChunkedDataResult, FormatError> {
    let plan = plan_chunked_at(set, data_address)?;
    let mut data_buf = Vec::with_capacity(plan.data_len.to_usize()?);
    emit_chunked_data(&mut data_buf, set, &plan).map_err(EmissionError::into_format)?;
    Ok(ChunkedDataResult {
        data_bytes: data_buf,
        layout_message: plan.layout_message,
        pipeline_message: plan.pipeline_message,
    })
}

/// Builds a chunked dataset's data region at `data_address`, with an optional
/// maximum shape.
///
/// Convenience composition of [`compress_chunks`] + [`assemble_chunked_at`] for
/// tests that build a single chunked dataset at a known address in one shot.
/// Production callers keep the [`CompressedChunkSet`] between passes instead, so
/// they compress each dataset only once: the file writer sizes an object header
/// before it emits the data, and the in-place editor sizes the dataset
/// ([`chunked_data_len`]) before it chooses the address to assemble it at.
///
/// `ctx` carries chunk_dims, element_size, and (for type-aware filters like
/// ZFP) the scalar element type. Build it via [`ChunkContext::from_datatype`]
/// when a `Datatype` is in scope.
#[cfg(test)]
pub fn build_chunked_data_at_ext(
    raw_data: &[u8],
    shape: &[u64],
    ctx: ChunkContext<'_>,
    options: &ChunkOptions,
    data_address: StoredAddress,
    maxshape: Option<&[MaxExtent]>,
    fill: FillPattern<'_>,
) -> Result<ChunkedDataResult, FormatError> {
    let set = compress_chunks(
        raw_data,
        shape,
        ctx,
        options,
        maxshape,
        fill,
        FillTime::IfSet,
        StorageAllocation::Allocated,
    )?;
    assemble_chunked_at(&set, data_address)
}

/// Per-chunk metadata in dense row-major grid order — enough to compute the
/// destination layout (chunk addresses and index structures) *without* the
/// chunk bytes. `compressed_size` is the exact byte count the
/// matching [`ChunkProvider::chunk_bytes`] call must return.
#[derive(Debug, Clone)]
pub(crate) struct ChunkMeta {
    /// Compressed on-disk size of this chunk, in bytes.
    pub(crate) compressed_size: u64,
    /// The chunk's filter mask from the source index, carried through verbatim.
    pub(crate) filter_mask: u32,
}

/// Yields one chunk's already-compressed bytes on demand. Called once per grid
/// slot, in ascending slot order, during the streaming assembly pass — so a
/// repacked dataset never holds more than a single chunk's bytes at a time.
///
/// `Send + Sync` is required so that a [`DatasetBuilder`](crate::type_builders::DatasetBuilder)
/// holding a boxed provider — and thus the public `FileBuilder` — keeps its
/// `Send`/`Sync` auto-traits. Real providers own an `Arc<File>`, which is both.
pub(crate) trait ChunkProvider: Send + Sync {
    /// Append grid slot `index`'s compressed bytes to `out`, which the emitter
    /// hands over empty. It is the same buffer on every call, so an
    /// implementation that appends costs one allocation for the whole dataset
    /// rather than one per chunk. The resulting length must equal the matching
    /// [`ChunkMeta::compressed_size`]; the emitter checks it.
    fn chunk_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), FormatError>;
}

/// Sequential output for buffered and streaming file assembly.
///
/// Writes and padding return the sink's associated error. Capacity reservation returns a
/// [`FormatError`] independently of the output error type. The buffered [`Vec<u8>`] implementation
/// uses [`Infallible`] for writes and padding.
pub(crate) trait ByteSink {
    /// The error from writing bytes or padding.
    type Error;
    /// Appends `bytes` to the output.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if writing the bytes fails.
    fn put(&mut self, bytes: &[u8]) -> Result<(), Self::Error>;
    /// Appends `n` zero bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if writing the padding fails.
    fn put_zeros(&mut self, n: usize) -> Result<(), Self::Error>;
    /// Returns the number of bytes written for comparison with planned layout addresses.
    fn position(&self) -> u64;
    /// Reserves capacity for additional output bytes when supported by the sink.
    ///
    /// Buffered sinks narrow the count when reserving memory. Streaming sinks may ignore it.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if a buffered sink cannot represent the
    /// capacity as [`usize`].
    ///
    /// # Panics
    ///
    /// A buffered sink may panic if its capacity would exceed [`isize::MAX`].
    fn reserve(&mut self, _additional: u64) -> Result<(), FormatError> {
        Ok(())
    }
}

impl ByteSink for Vec<u8> {
    type Error = Infallible;

    fn put(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        self.extend_from_slice(bytes);
        Ok(())
    }
    fn put_zeros(&mut self, n: usize) -> Result<(), Self::Error> {
        self.resize(self.len() + n, 0u8);
        Ok(())
    }
    fn position(&self) -> u64 {
        self.len() as u64
    }
    fn reserve(&mut self, additional: u64) -> Result<(), FormatError> {
        Vec::reserve(self, additional.to_usize()?);
        Ok(())
    }
}

/// Where the chunk index goes and how long it is, without its bytes.
///
/// The index is built once, by [`emit_chunked_data_verbatim`], at the moment it
/// is written. Planning it as a length rather than as bytes is what lets a
/// caller reserve the data region's span from a plan made at a provisional address
/// and then discard that plan: nothing was built to arrive at the number.
struct VerbatimIndexPlan {
    /// Which array to build. `ChunkIndexKind::SingleChunk` cannot appear here:
    /// the layout that writes no index is the `None` case of the field holding
    /// this, not a third variant of it.
    kind: ChunkArrayKind,
    address: StoredAddress,
    /// Whether the element records carry a compressed size and filter mask. The
    /// address and length widths are not fields: every chunk index this module
    /// writes uses `INDEX_OFFSET_SIZE` / `INDEX_LENGTH_SIZE`, and reading them
    /// at the emit is one fewer value that could be set wrong here.
    has_filters: bool,
    /// The unfiltered byte size of one whole chunk, which sizes the element's
    /// compressed-size field. Carried on the plan for the same reason
    /// `has_filters` is: the emit builds the index from this alone, and a second
    /// derivation there could disagree with the length reserved here.
    chunk_bytes: u64,
    len: u64,
}

/// The full destination layout of a verbatim chunked dataset's data region,
/// computed from chunk *sizes* alone (no chunk bytes). Feeds both the object
/// header (via the separately returned layout/pipeline messages) and the
/// streaming emit ([`emit_chunked_data_verbatim`]).
pub(crate) struct VerbatimPlan {
    /// One entry per grid slot, in ascending address order: where it goes and
    /// how many bytes it occupies. Slots are stored back to back, so a slot's own
    /// compressed byte count is its whole placement — the next begins where this
    /// one ends — and the index records the addresses that follow from that.
    pub(crate) chunks: Vec<ChunkRecord>,
    /// The chunk index emitted after the chunk bytes. `None` for the
    /// single-chunk layout, whose address rides in the layout message instead.
    index: Option<VerbatimIndexPlan>,
    /// The element slot each of `chunks` occupies, and how many slots the index
    /// spans. `chunks` stays in dense grid order because that is the order the
    /// bytes are emitted and the provider is asked in, so the index needs this
    /// alongside it.
    slot_of_chunk: Vec<u64>,
    index_slots: u64,
    /// Total byte length of the data region: the chunk bytes, then the index.
    pub(crate) total_len: u64,
}

/// The result of planning a verbatim chunked dataset: the data-region
/// [`VerbatimPlan`] (for the streaming emit) plus the object-header messages it
/// implies (the v4 layout message and the verbatim pipeline message).
pub(crate) struct VerbatimLayout {
    pub(crate) plan: VerbatimPlan,
    pub(crate) layout_message: Vec<u8>,
    pub(crate) pipeline_message: Option<Vec<u8>>,
}

/// Returns the [`VerbatimLayout`] of a dense set of chunks in row-major grid order, planned from
/// their sizes and filter masks alone.
///
/// The layout is the same whether the chunks are later buffered or streamed. The caller passes
/// each partial edge chunk unfiltered where [`ChunkedLayoutFlags::partial_edge_chunks_stored_raw`]
/// is `true` for `flags`: the layout message stores `flags` as given, and the plan records each
/// chunk as `meta` describes it.
///
/// # Errors
///
/// Returns [`FormatError::ChunkedReadError`] if `meta` is empty, holds a number of chunks other
/// than the grid over `shape`, or holds a chunk too large for the chunk size field of a filtered
/// index, and [`FormatError::OffsetOverflow`] if the stored sizes of the chunks sum past
/// [`u64::MAX`]. Returns the errors of [`plan_index_slots`] and [`IndexSlots::new`] for the
/// index, and of [`DataLayout::encode_chunked`] for the layout message.
pub(crate) fn plan_chunked_data_verbatim(
    meta: &[ChunkMeta],
    shape: &[u64],
    chunk_dims: &[u64],
    element_size: NonZeroUsize,
    flags: ChunkedLayoutFlags,
    pipeline_message: Option<&[u8]>,
    data_address: StoredAddress,
    maxshape: Option<&[MaxExtent]>,
) -> Result<VerbatimLayout, FormatError> {
    if meta.is_empty() {
        return Err(FormatError::ChunkedReadError(
            "a verbatim chunked dataset requires at least one chunk".into(),
        ));
    }
    let num_chunks = meta.len();
    let has_filters = pipeline_message.is_some();

    // Walk a running cursor instead of pushing bytes; each address is a pure
    // function of the preceding chunk sizes, mirroring the buffered builder.
    let mut cursor: u64 = 0;
    let mut written_chunks = Vec::with_capacity(num_chunks);

    for m in meta {
        let address = data_address.offset(cursor);
        let compressed_size = m.compressed_size;
        written_chunks.push(ChunkRecord {
            address,
            stored_size: compressed_size,
            filter_mask: m.filter_mask,
        });
        // The one length here a file supplies: a verbatim plan's chunk sizes come from the source
        // dataset's index, so the running sum is checked before it offsets an address.
        cursor = cursor
            .checked_add(compressed_size)
            .ok_or(FormatError::OffsetOverflow {
                offset: cursor,
                length: compressed_size,
            })?;
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "chunk dimensions written into the on-disk u32 dimension fields selected for this file"
    )]
    let chunk_dims_u32: Vec<u32> = chunk_dims.iter().map(|&d| d as u32).collect();
    let offset_size = INDEX_OFFSET_SIZE;
    let length_size = INDEX_LENGTH_SIZE;

    // The chunks arrived in dense grid order; where each one's index element
    // sits is the maximum grid's business, and the same decision the encode path
    // makes (`plan_index_slots`).
    let chunk_bytes = full_chunk_bytes(chunk_dims.iter().copied(), element_size);
    let (kind, slot_of_chunk, index_slots) = plan_index_slots(
        shape,
        chunk_dims,
        maxshape,
        chunk_bytes,
        has_filters,
        // A verbatim payload is the chunks the source held. Its own count is
        // checked against the plan's just below, so an empty one is reported
        // rather than quietly re-planned as an unallocated dataset.
        StorageAllocation::Allocated,
    )?;
    if slot_of_chunk.len() != num_chunks {
        return Err(FormatError::ChunkedReadError(format!(
            "a verbatim chunked dataset of shape {shape:?} holds {} chunks, not the \
             {num_chunks} it was given",
            slot_of_chunk.len(),
        )));
    }
    if kind.array_kind().is_some() {
        check_chunks_fit_index(meta, chunk_bytes, has_filters)?;
    }

    // The index sits immediately after the chunk bytes. Its length is taken from
    // the index's own layout rather than from a build of it, so this planner
    // touches no index bytes either — which is what lets `write_chunked_relocatable`
    // plan at a provisional address purely to size the region.
    let index_address = data_address.offset(cursor);
    let index = match kind.array_kind() {
        Some(array) => Some(VerbatimIndexPlan {
            kind: array,
            address: index_address,
            has_filters,
            chunk_bytes,
            len: chunk_index_len(
                array,
                &IndexSlots::new(&written_chunks, &slot_of_chunk, index_slots)?,
                chunk_bytes,
                offset_size,
                length_size,
                has_filters,
            ),
        }),
        None => None,
    };
    cursor += index.as_ref().map_or(0, |i| i.len);

    let layout_message = kind.layout_message(
        flags,
        &chunk_dims_u32,
        element_size,
        &written_chunks,
        index_address,
        has_filters,
    )?;

    Ok(VerbatimLayout {
        plan: VerbatimPlan {
            chunks: written_chunks,
            slot_of_chunk,
            index_slots,
            index,
            total_len: cursor,
        },
        layout_message,
        pipeline_message: pipeline_message.map(<[u8]>::to_vec),
    })
}

/// Checks that the chunk size field of a filtered chunk index holds the stored size of every chunk
/// in `meta`.
///
/// # Errors
///
/// Returns [`FormatError::ChunkedReadError`] if a chunk is too large for the field of an index
/// over chunks of `chunk_bytes` bytes.
fn check_chunks_fit_index(
    meta: &[ChunkMeta],
    chunk_bytes: u64,
    has_filters: bool,
) -> Result<(), FormatError> {
    if !has_filters {
        return Ok(());
    }
    let encoding =
        hdf5_pure_format::__private::chunk_element_encoding(chunk_bytes, INDEX_OFFSET_SIZE, true);
    match meta
        .iter()
        .enumerate()
        .find(|(_, m)| !encoding.holds_stored_size(m.compressed_size))
    {
        Some((slot, m)) => Err(FormatError::ChunkedReadError(format!(
            "chunk {slot} of a verbatim chunked dataset stores {} bytes, more than the chunk \
             index records for a filtered chunk of {chunk_bytes} bytes",
            m.compressed_size,
        ))),
        None => Ok(()),
    }
}

/// Writes a planned verbatim dataset's chunks and index in order.
///
/// Reads each chunk from `provider` into one reusable buffer. The output is byte-for-byte
/// identical to the concatenation described by [`plan_chunked_data_verbatim`]. The writer builds
/// the chunk index after writing the chunks, so a serialization failure can leave partial output.
///
/// # Errors
///
/// Returns [`EmissionError::Format`] with [`FormatError::ChunkedReadError`] if a provider returns
/// a chunk of another length, or [`FormatError::SerializationError`] if the serialized index
/// differs in length from its plan. Propagates provider and index errors as [`EmissionError::Format`]
/// and sink errors as [`EmissionError::Sink`].
pub(crate) fn emit_chunked_data_verbatim<S: ByteSink>(
    sink: &mut S,
    plan: &VerbatimPlan,
    provider: &dyn ChunkProvider,
) -> Result<(), EmissionError<S::Error>> {
    // One buffer for the whole dataset: it grows to the largest chunk and is
    // reused, so the streaming path's allocation count does not scale with the
    // chunk count.
    let mut chunk = Vec::new();
    for (i, slot) in plan.chunks.iter().enumerate() {
        chunk.clear();
        provider.chunk_bytes(i, &mut chunk)?;
        if chunk.len() as u64 != slot.stored_size {
            return Err(FormatError::ChunkedReadError(
                "verbatim chunk provider returned a chunk whose size differs from the \
                 planned size"
                    .into(),
            )
            .into());
        }
        sink.put(&chunk).map_err(EmissionError::Sink)?;
    }

    // The index is built here, once, rather than by the planner: a caller may
    // plan the same region more than once (at a provisional address to size it, then
    // at the real one), and only this call writes it.
    if let Some(index) = &plan.index {
        let slots = IndexSlots::new(&plan.chunks, &plan.slot_of_chunk, plan.index_slots)?;
        let bytes = match index.kind {
            ChunkArrayKind::ExtensibleArray => {
                hdf5_pure_format::__private::build_extensible_array_at(
                    &slots,
                    index.chunk_bytes,
                    INDEX_OFFSET_SIZE,
                    INDEX_LENGTH_SIZE,
                    index.has_filters,
                    index.address,
                )?
            }
            ChunkArrayKind::FixedArray => hdf5_pure_format::__private::build_fixed_array_at(
                &slots,
                index.chunk_bytes,
                INDEX_OFFSET_SIZE,
                INDEX_LENGTH_SIZE,
                index.has_filters,
                index.address,
            )?,
        };
        if bytes.len() as u64 != index.len {
            return Err(FormatError::SerializationError(format!(
                "a chunk index built {} bytes where its plan reserved {}; the data region's \
                 length was computed from the plan",
                bytes.len(),
                index.len,
            ))
            .into());
        }
        sink.put(&bytes).map_err(EmissionError::Sink)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
