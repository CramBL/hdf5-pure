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

#[cfg(all(
    test,
    feature = "std",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
mod allocation_tests;

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

/// Whether a dataset's storage is allocated at all.
///
/// By default the reference library does not allocate a *contiguous or chunked*
/// dataset's storage until something is written to it — compact data is inline
/// in the layout message and is always present — so one created and never
/// written holds no chunks over a non-empty dataspace. Its layout message still
/// names an index *type*; what it carries for that index is the undefined
/// address, so no index structure exists. A contiguous dataset created the same
/// way is the same story without an index in it.
///
/// This has to be said rather than derived, because the element count cannot
/// tell the two apart: a shape of 1,000 means "ten chunks of fill value" for one
/// dataset and "nothing stored" for the other, and reading the second answers the
/// fill value for every element exactly as the first does (issue #292). Deriving
/// it from the staged bytes would be worse still, since "no bytes" is also what a
/// caller who simply forgot the data looks like.
///
/// There is deliberately no `Default`: like the `fill` parameter above, every
/// caller names it, so a new write path cannot inherit "allocated" by omission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StorageAllocation {
    /// Storage covers the dataspace: a chunk in every grid slot the shape
    /// implies, or a contiguous run of every element's bytes.
    Allocated,
    /// No chunk, and no run of element bytes: the dataset declares its shape
    /// and stores none of it.
    ///
    /// What that leaves in the file depends on the layout. A contiguous or
    /// fixed-shape chunked dataset carries the undefined address and occupies
    /// nothing. A *resizable* one still gets the eagerly built Extensible Array
    /// this crate gives every empty resizable dataset — an index over no chunk,
    /// at a defined address, costing a few hundred bytes — because an in-place
    /// append needs the index to exist before the first chunk arrives.
    Unallocated,
}

/// Which chunk index a chunk set gets.
///
/// One rule, read by everything that has to agree on it: the writers that emit
/// the index, and [`chunk_index_len`], which sizes it without emitting. Two
/// copies of this `if` chain is how a length ends up describing a different
/// structure from the one written.
///
/// The variant names match [`layout_info::ChunkIndex`](crate::layout_info::ChunkIndex),
/// which classifies the same three shapes on the read side.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ChunkIndexKind {
    /// No index structure and no chunk address: a fixed-shape dataset with no
    /// chunks at all, whose layout message carries the undefined address.
    ///
    /// This is libhdf5's convention for a chunked dataset with
    /// nothing stored. For a fixed-shape dataset of *zero* slots it is the only
    /// encoding that library accepts: a Fixed Array declaring zero entries makes
    /// `H5Dget_num_chunks` fail on the dataset, where the undefined address
    /// reads back as zero chunks.
    ///
    /// Over a non-empty dataspace — the never-written dataset of issue #293 —
    /// that argument does not apply: measured, the library reads a Fixed Array
    /// whose slots are all empty perfectly well, and reports zero chunks for it.
    /// The undefined address is preferred there for the two weaker reasons, that
    /// it is what the library itself writes and that it costs no index at all. An *extensible* empty dataset is the opposite case and keeps its
    /// (eagerly built) array — the C library reads and grows that happily, and
    /// this crate's in-place append needs the index to already exist.
    Unallocated,
    /// No index structure at all: the single chunk's address rides in the
    /// layout message.
    SingleChunk,
    /// A Fixed Array, for a fixed-shape dataset of more than one chunk.
    FixedArray,
    /// An Extensible Array, for a dataset with an unlimited dimension.
    ExtensibleArray,
}

/// The two kinds that are an actual on-disk structure, and so have a length and
/// bytes. [`ChunkIndexKind::SingleChunk`] is not one of them, and this type is
/// how that is said once rather than re-checked at every use.
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
    /// The on-disk array this kind writes, or `None` for the single-chunk
    /// layout, which writes nothing after the chunk bytes.
    pub(crate) fn array_kind(self) -> Option<ChunkArrayKind> {
        match self {
            Self::Unallocated | Self::SingleChunk => None,
            Self::FixedArray => Some(ChunkArrayKind::FixedArray),
            Self::ExtensibleArray => Some(ChunkArrayKind::ExtensibleArray),
        }
    }

    /// Returns the body of the version 4 data layout message of chunks indexed
    /// by this kind, with `flags` as its layout flags and `index_address` as the
    /// address of a Fixed Array or an Extensible Array.
    ///
    /// A Single Chunk layout stores the address of the first of `chunks`, and an
    /// unallocated layout a Fixed Array index at the undefined address.
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
            Self::Unallocated => (ChunkIndexInfo::FixedArray, StoredAddress::new(HADDR_UNDEF)),
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
        _ if num_chunks == 0 => ChunkIndexKind::Unallocated,
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
    allocation: StorageAllocation,
) -> Result<CompressedChunkSet, FormatError> {
    let chunk_dims = ctx.chunk_dims;
    let element_size = ctx.element_size.narrow::<NonZeroUsize>()?;
    crate::chunked_read::bounded_chunk_len(chunk_dims, element_size)?;
    // The same pattern the unwritten slots are padded with: a filter that
    // records the fill value has to record the one this write pads with.
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
            fill,
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

/// Decide a chunked dataset's index and where each of its chunks sits in it:
/// the index kind, the element slot of each chunk in dense grid order, and how
/// many slots the index spans.
///
/// One function because the three answers are one decision. The grid the slots
/// are numbered over is the same grid whose size picks between a single-chunk
/// layout and a Fixed Array, and a caller that derived them separately could
/// number chunks for one index while declaring another.
///
/// `allocation` says whether the dataset stores anything at all. Every other
/// input describes the *dataspace*, which an unallocated dataset declares in
/// full; only this distinguishes a grid of chunks from none, and a fixed-shape
/// dataset that stores none is the [`ChunkIndexKind::Unallocated`] layout.
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
    let kind = chunk_index_kind(&grid, num_chunks);

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
        ChunkIndexKind::Unallocated | ChunkIndexKind::SingleChunk => 0,
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
        ChunkIndexKind::Unallocated | ChunkIndexKind::SingleChunk => 0,
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
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::chunk_cache::ChunkCache;

    use crate::FillTime;
    use crate::chunked_read;
    use crate::convert;
    use crate::data_layout::ChunkIndexLayout;
    use crate::data_layout::DataLayout;
    use crate::dataspace::{Dataspace, DataspaceType};
    use crate::datatype::layout::FloatingPointLayout;
    use crate::datatype::{Datatype, byte_order};
    use crate::fill_value::FillPattern;
    use crate::filters;
    use crate::read_spec::RawReadSpec;

    mod partial_edge;
    mod zfp;

    fn make_f64_type() -> Datatype {
        Datatype::FloatingPoint(
            FloatingPointType::from_layout(
                8,
                byte_order::DatatypeByteOrder::LittleEndian,
                FloatingPointLayout::IEEE754_BINARY64,
            )
            .unwrap(),
        )
    }

    /// Every chunk [`split_into_chunks`] produces is exactly one whole chunk of
    /// bytes — the edge overhang is padded, never truncated to the in-bounds
    /// part. What it is padded *with* is the dataset's fill value and does not
    /// matter here; the length is what this rests on.
    ///
    /// This is the equivalence that makes deriving the chunk-index element width
    /// from the geometry ([`full_chunk_bytes`]) the *same answer* the old
    /// derivation gave — a maximum over the written chunks' raw sizes — for every
    /// dataset that has at least one chunk, and therefore byte-identical output
    /// for every file this crate had already written. The two part company only
    /// at zero chunks, where there is no maximum to take.
    ///
    /// Swept over shapes that divide evenly and shapes that overhang in the
    /// innermost dimension, an outer one, and both at once, since a truncated
    /// edge chunk is the only way this could fail.
    /// An unreadable Fill Value message leaves what a chunk's uncovered slots
    /// must hold *undetermined*, which is not the same answer as "zeros" — and
    /// unlike a read of unallocated storage, writing the guess puts it on disk.
    /// So the split refuses.
    ///
    /// It refuses only where the answer is needed. A dataset whose chunks all
    /// fall wholly inside it has no uncovered slot, never consults the pattern,
    /// and stays writable — the same scoping the read path uses, where a
    /// fully-allocated dataset reads fine however unreadable its fill message.
    #[test]
    fn an_unknown_fill_refuses_only_the_chunks_that_need_padding() {
        let elem = convert::nz(4);
        let data = vec![0u8; 8 * 4];

        // 8 elements in chunks of 4: two whole chunks, nothing uncovered.
        assert!(
            split_into_chunks(&data, &[8], &[4], elem, FillPattern::UNKNOWN).is_ok(),
            "a write that pads nothing must not consult the fill value"
        );

        // 5 elements in chunks of 4: the second chunk has three uncovered slots.
        assert!(matches!(
            split_into_chunks(&data[..5 * 4], &[5], &[4], elem, FillPattern::UNKNOWN),
            Err(FormatError::UnreadableFillValue)
        ));

        // Rank > 1: whole along the inner dimension, short along the outer, so
        // the uncovered region is whole rows rather than a trailing run.
        assert!(matches!(
            split_into_chunks(
                &data[..3 * 2 * 4],
                &[3, 2],
                &[2, 2],
                elem,
                FillPattern::UNKNOWN
            ),
            Err(FormatError::UnreadableFillValue)
        ));

        // The known patterns are unaffected either way.
        for pattern in [
            FillPattern::ZERO,
            FillPattern::new(FillTime::IfSet, Some(&[7u8; 4]), elem),
        ] {
            assert!(split_into_chunks(&data[..5 * 4], &[5], &[4], elem, pattern).is_ok());
        }
    }

    #[test]
    fn every_split_chunk_is_a_whole_chunk() {
        let cases: &[(&[u64], &[u64])] = &[
            (&[8], &[4]),             // even
            (&[7], &[4]),             // innermost overhang
            (&[1], &[512]),           // a single, almost entirely empty chunk
            (&[4, 6], &[2, 3]),       // even, 2-D
            (&[5, 6], &[2, 3]),       // outer overhang
            (&[4, 7], &[2, 3]),       // inner overhang
            (&[5, 7], &[2, 3]),       // both
            (&[3, 5, 7], &[2, 2, 4]), // both, 3-D
        ];
        for &(shape, chunk_dims) in cases {
            for elem in [1usize, 4, 8] {
                let n: u64 = shape.iter().product();
                let raw = vec![0u8; (n as usize) * elem];
                let expected = full_chunk_bytes(chunk_dims.iter().copied(), convert::nz(elem));
                let chunks = split_into_chunks(
                    &raw,
                    shape,
                    chunk_dims,
                    convert::nz(elem),
                    FillPattern::ZERO,
                )
                .unwrap();
                assert!(
                    !chunks.is_empty(),
                    "shape {shape:?} chunk {chunk_dims:?} must produce chunks"
                );
                for (i, c) in chunks.iter().enumerate() {
                    assert_eq!(
                        c.len() as u64,
                        expected,
                        "chunk {i} of shape {shape:?} chunk {chunk_dims:?} elem {elem} \
                         is not a whole chunk"
                    );
                }
            }
        }
    }

    #[rstest]
    #[case::just_over(1 << 29)]
    #[case::unallocatable(1 << 40)]
    fn a_created_chunk_larger_than_u32_max_bytes_is_rejected(
        #[case] chunk_len: u64,
        #[values(StorageAllocation::Allocated, StorageAllocation::Unallocated)]
        allocation: StorageAllocation,
    ) {
        let chunk_dims = [chunk_len];
        let raw: &[u8] = match allocation {
            StorageAllocation::Allocated => &[0; 8],
            StorageAllocation::Unallocated => &[],
        };
        let Err(err) = compress_chunks(
            raw,
            &[1],
            ChunkContext::basic(&chunk_dims, 8),
            &ChunkOptions {
                chunk_dims: Some(chunk_dims.to_vec()),
                ..ChunkOptions::default()
            },
            None,
            FillPattern::ZERO,
            allocation,
        ) else {
            panic!("expected InvalidChunkGeometry");
        };
        assert_eq!(
            err,
            FormatError::InvalidChunkGeometry(
                "chunk logical byte size exceeds the reader's single-chunk allocation limit"
            )
        );
    }

    /// A dataset that stores nothing is planned as storing nothing, whatever
    /// its shape says (issue #293).
    ///
    /// The shape is the only thing the planner otherwise has to count chunks
    /// from, so this is the one input that can tell "ten chunks of the fill
    /// value" from "no chunks at all" — the two states a read cannot
    /// distinguish. Both answers are taken from the same geometry here, so the
    /// only difference between them is the allocation.
    #[test]
    fn an_unallocated_dataset_plans_no_chunks_over_the_shape_it_declares() {
        let shape = [1000u64];
        let chunk = [100u64];

        let (kind, slots, span) = plan_index_slots(
            &shape,
            &chunk,
            None,
            400,
            false,
            StorageAllocation::Allocated,
        )
        .expect("a dense fixed-shape plan");
        assert!(matches!(kind, ChunkIndexKind::FixedArray));
        assert_eq!(slots.len(), 10, "ten chunks cover the shape");
        assert_eq!(span, 10);

        let (kind, slots, span) = plan_index_slots(
            &shape,
            &chunk,
            None,
            400,
            false,
            StorageAllocation::Unallocated,
        )
        .expect("an unallocated plan");
        assert!(
            matches!(kind, ChunkIndexKind::Unallocated),
            "a fixed-shape dataset that stores nothing carries the undefined \
             address, not an index over an empty grid: {kind:?}"
        );
        assert!(slots.is_empty(), "no chunk has a slot");
        assert_eq!(span, 0, "and the index spans none");

        // A dataset that is allowed to grow keeps the growable index either
        // way, which is the same choice an empty resizable dataset gets: the
        // in-place append path needs the index to exist before the first chunk
        // arrives, so "stores nothing" does not mean "indexes nothing" here.
        let (kind, slots, _) = plan_index_slots(
            &shape,
            &chunk,
            Some(&[MaxExtent::Unlimited]),
            400,
            false,
            StorageAllocation::Unallocated,
        )
        .expect("an unallocated resizable plan");
        assert!(matches!(kind, ChunkIndexKind::ExtensibleArray), "{kind:?}");
        assert!(slots.is_empty());
    }

    #[rstest]
    #[case::rank_one(&[19], &[4], 38)]
    #[case::short_rank_one(&[19], &[4], 17)]
    #[case::full_width(&[8, 6], &[2, 6], 96)]
    #[case::short_full_width(&[8, 6], &[2, 6], 71)]
    #[case::gathered(&[8, 6], &[2, 3], 96)]
    #[case::edges(&[7, 5], &[2, 3], 70)]
    #[case::short_edges(&[7, 5], &[2, 3], 33)]
    #[case::slabs(&[3, 4, 5], &[1, 4, 5], 120)]
    #[case::gathered_rank_three(&[3, 4, 5], &[2, 3, 2], 120)]
    #[case::empty(&[0, 4], &[2, 2], 0)]
    fn incremental_encoding_matches_collected_chunk_bytes(
        #[case] shape: &[u64],
        #[case] chunks: &[u64],
        #[case] input_len: usize,
        #[values(
            None,
            Some(FilterKind::Shuffle),
            Some(FilterKind::Lzf),
            Some(FilterKind::Fletcher32)
        )]
        filter: Option<FilterKind>,
        #[values(PartialEdgeFiltering::Apply, PartialEdgeFiltering::Skip)]
        edges: PartialEdgeFiltering,
    ) {
        let raw = (0..input_len)
            .map(|byte| u8::try_from(byte).unwrap())
            .collect::<Vec<_>>();
        let fill = FillPattern::new(FillTime::IfSet, Some(&[241, 242]), convert::nz(2));
        let ctx = ChunkContext::basic(chunks, 2);
        let mut options = ChunkOptions {
            partial_edge_filtering: Some(edges),
            ..ChunkOptions::default()
        };
        if let Some(filter) = filter {
            options.set_filter(filter);
        }
        let pipeline = options.build_pipeline(&ctx, fill).unwrap();
        let mut scratch = FilterScratch::new();
        let expected = split_into_chunks(&raw, shape, chunks, convert::nz(2), fill)
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(index, logical)| {
                let mut remaining = index.to_u64();
                let mut partial = false;
                for (&dimension, &chunk) in shape.iter().zip(chunks).rev() {
                    let count = dimension.div_ceil(chunk);
                    let offset = (remaining % count) * chunk;
                    remaining /= count;
                    partial |= chunk > dimension - offset;
                }
                match &pipeline {
                    Some(pipeline) if edges == PartialEdgeFiltering::Apply || !partial => {
                        filters::compress_chunk_with(&mut scratch, &logical, pipeline, ctx).unwrap()
                    }
                    _ => logical,
                }
            })
            .collect::<Vec<_>>();
        let actual = compress_chunks(
            &raw,
            shape,
            ctx,
            &options,
            None,
            fill,
            StorageAllocation::Allocated,
        )
        .unwrap();
        assert_eq!(
            actual
                .chunks
                .iter()
                .map(PreparedChunk::bytes)
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(actual.has_filters, pipeline.is_some());
        assert_eq!(
            actual
                .chunks
                .iter()
                .map(PreparedChunk::filter_mask)
                .collect::<Vec<_>>(),
            vec![0; expected.len()]
        );
        assert_eq!(
            actual.slot_of_chunk,
            (0..expected.len().to_u64()).collect::<Vec<_>>()
        );
    }

    #[rstest]
    #[case::empty(&[0, 4], StorageAllocation::Allocated)]
    #[case::unallocated(&[7, 5], StorageAllocation::Unallocated)]
    fn filtered_datasets_without_chunks_skip_gathering(
        #[case] shape: &[u64],
        #[case] allocation: StorageAllocation,
    ) {
        let mut options = ChunkOptions::default();
        options.set_filter(FilterKind::Shuffle);
        let set = compress_chunks(
            &[],
            shape,
            ChunkContext::basic(&[2, 2], 2),
            &options,
            None,
            FillPattern::UNKNOWN,
            allocation,
        )
        .unwrap();
        assert!(set.chunks.is_empty());
        assert!(set.has_filters);
        assert!(set.slot_of_chunk.is_empty());
    }

    /// The encoder does not split an unallocated dataset into chunks.
    ///
    /// Compressing one is not merely wasted work. [`split_into_chunks`] emits a
    /// chunk for every slot the shape implies whatever it is given, and an
    /// unallocated dataset gives it nothing, so a set built from it would carry
    /// the whole materialized grid — the bytes issue #293 is about — with the
    /// *wrong* contents in it: a chunk lying inside the shape is zero-padded
    /// where the data runs out, and only a chunk overhanging the dataset's edge
    /// takes the fill pattern — ten chunks of zeros, measured on this geometry,
    /// under a dataset whose fill value is 7. The materialization is the reason
    /// to skip the split; that the bytes would also be wrong is what makes it
    /// worth skipping rather than merely wasteful.
    #[test]
    fn an_unallocated_dataset_encodes_no_chunk_bytes() {
        let shape = [1000u64];
        let chunk = [100u64];
        let fill = [7u8, 0, 0, 0];
        let elem = NonZeroUsize::new(4).unwrap();

        let set = compress_chunks(
            &[],
            &shape,
            ChunkContext::basic(&chunk, 4),
            &ChunkOptions::default(),
            None,
            FillPattern::new(FillTime::IfSet, Some(&fill), elem),
            StorageAllocation::Unallocated,
        )
        .expect("an unallocated set");
        assert_eq!(set.chunks.len(), 0, "no chunk was encoded");
        assert_eq!(
            chunked_data_len(&set).unwrap(),
            0,
            "and the dataset occupies no data region"
        );

        let assembled = assemble_chunked_at(&set, StoredAddress::new(0x1000)).unwrap();
        assert!(assembled.data_bytes.is_empty(), "nothing to write out");
        // The layout message names no index. `HADDR_UNDEF` is all ones, and it
        // is the field the reference library reads to decide the dataset has no
        // storage at all.
        assert!(
            assembled
                .layout_message
                .windows(8)
                .any(|w| w == u64::MAX.to_le_bytes()),
            "the layout message must carry the undefined address: {:?}",
            assembled.layout_message
        );

        // The same geometry with its storage allocated is the grid this avoids.
        let raw = vec![0u8; 1000 * 4];
        let dense = compress_chunks(
            &raw,
            &shape,
            ChunkContext::basic(&chunk, 4),
            &ChunkOptions::default(),
            None,
            FillPattern::new(FillTime::IfSet, Some(&fill), elem),
            StorageAllocation::Allocated,
        )
        .expect("a dense set");
        assert_eq!(dense.chunks.len(), 10);
        assert!(chunked_data_len(&dense).unwrap() >= 4000);
    }

    /// The index-size bound counts the bytes that describe no chunk.
    ///
    /// Two things follow, and neither did from the slot count this replaced. A
    /// dataset of four million chunks needs a four-million-element index
    /// whatever its maximum shape says, so a bound on the total refuses a
    /// perfectly ordinary dataset that names no maximum at all. And a filtered
    /// element is 15 to 20 bytes against an unfiltered 8, so the same slot count
    /// allowed a filtered index two and a half times the bytes.
    ///
    /// Asserted here rather than through a file because writing the dense case
    /// costs 50 MB and the arithmetic is the whole claim.
    #[test]
    fn the_index_bound_counts_unused_bytes_rather_than_slots() {
        // A dense index is never bounded, however large: every slot holds a
        // chunk, so none of its bytes describe nothing.
        let many = (MAX_UNUSED_INDEX_BYTES / 8 + 1) as usize;
        let (kind, slots, span) = plan_index_slots(
            &[many as u64],
            &[1],
            None,
            8,
            false,
            StorageAllocation::Allocated,
        )
        .expect("a dense index is not bounded");
        assert!(matches!(kind, ChunkIndexKind::FixedArray));
        assert_eq!(slots.len(), many);
        assert_eq!(span, many as u64);

        // The same span reached by a maximum shape instead, holding one chunk,
        // is almost entirely unused — and lands at the same *byte* figure
        // whether or not the dataset is filtered, which is the point of
        // measuring in bytes. A filtered element is wider, so that is a smaller
        // span.
        let chunk_bytes = 4096;
        for has_filters in [false, true] {
            let elem = u64::from(
                hdf5_pure_format::__private::chunk_element_encoding(
                    chunk_bytes,
                    INDEX_OFFSET_SIZE,
                    has_filters,
                )
                .elem_size as u32,
            );
            let widest = MAX_UNUSED_INDEX_BYTES / elem + 1;
            plan_index_slots(
                &[1],
                &[1],
                Some(&[MaxExtent::Fixed(widest)]),
                chunk_bytes,
                has_filters,
                StorageAllocation::Allocated,
            )
            .expect("an index exactly at the budget is written");
            let err = plan_index_slots(
                &[1],
                &[1],
                Some(&[MaxExtent::Fixed(widest + 1)]),
                chunk_bytes,
                has_filters,
                StorageAllocation::Allocated,
            )
            .unwrap_err();
            assert!(
                format!("{err}").contains("describe no chunk"),
                "filtered={has_filters}: {err}"
            );
        }

        // Room to grow along the dimension the dataset grows in costs no unused
        // slots however far it reaches, so it stays accepted.
        plan_index_slots(
            &[8, 8],
            &[4, 4],
            Some(&[MaxExtent::Unlimited, MaxExtent::Fixed(8)]),
            64,
            false,
            StorageAllocation::Allocated,
        )
        .expect("growth along the indexed dimension leaves no gaps");
    }

    /// An Extensible Array can address only as many slots as its blocks cover.
    /// A chunk numbered past that lands in no block, and the writer used to drop
    /// it — leaving a 580-byte file that this crate read as a zero and the
    /// reference library refused with "ring type mismatch occurred for cache
    /// entry" (issue #299).
    ///
    /// Unreachable while the size bound counted slots, since getting there took
    /// billions of unused ones. Counting bytes instead lets a sparse array reach
    /// it cheaply, so the capacity is now its own refusal.
    #[test]
    fn an_extensible_array_refuses_a_chunk_past_the_slots_it_can_address() {
        let capacity = hdf5_pure_format::__private::extensible_array_capacity();
        assert_eq!(
            capacity, 8_589_934_580,
            "the C library's default creation parameters address this many slots"
        );

        // Two chunks a stride apart: the second sits at slot `stride`.
        let at = |stride: u64| {
            plan_index_slots(
                &[2, 1],
                &[1, 1],
                Some(&[MaxExtent::Unlimited, MaxExtent::Fixed(stride)]),
                4,
                false,
                StorageAllocation::Allocated,
            )
        };
        at(capacity - 1).expect("the last addressable slot is written");
        let err = at(capacity).unwrap_err();
        assert!(format!("{err}").contains("can address"), "{err}");
    }

    /// Auto-chunking resolves the chunk to the whole shape, so a zero-element
    /// shape resolves to a zero chunk dimension — the same value
    /// `validate_geometry` already rejected when a caller named it, reached
    /// without one. It divided by zero in [`split_into_chunks`] until the guard
    /// checked the resolved dimensions rather than only the explicit ones.
    #[test]
    fn auto_chunking_a_zero_element_shape_is_refused() {
        let auto = ChunkOptions {
            chunk_dims: None,
            ..Default::default()
        };
        for shape in [vec![0u64], vec![4, 0], vec![0, 4]] {
            let unlimited = vec![MaxExtent::Unlimited; shape.len()];
            let extent = Extent::new(&shape, Some(&unlimited)).unwrap();
            let err = auto.validate_geometry(extent).unwrap_err();
            assert!(
                err.contains("explicit chunk dimensions"),
                "shape {shape:?}: {err}"
            );
            // The resolved dimensions are what the splitter would divide by, and
            // this is the value the guard exists to keep away from it.
            assert!(auto.resolve_chunk_dims(&shape).contains(&0));
        }

        // Named chunk dimensions make the same shape legal: it produces zero
        // chunks, which is what an extensible dataset is created as.
        let explicit = ChunkOptions {
            chunk_dims: Some(vec![512]),
            ..Default::default()
        };
        let unlimited = Extent::new(&[0], Some(&[MaxExtent::Unlimited])).unwrap();
        assert_eq!(explicit.validate_geometry(unlimited), Ok(()));
        let bounded = Extent::new(&[0], None).unwrap();
        assert_eq!(explicit.validate_geometry(bounded), Ok(()));
    }

    /// Options carrying `filters`, placed the way the `DatasetBuilder`
    /// switches place them — so a test that means "shuffle then deflate", the
    /// combination a caller asks for, does not have to restate the canonical
    /// order it comes out in. A test about a *stored* order pushes instead.
    fn filtered(filters: &[FilterKind]) -> ChunkOptions {
        let mut options = ChunkOptions::default();
        for &f in filters {
            options.set_filter(f);
        }
        options
    }

    /// [`filtered`] with chunk dimensions.
    fn chunked(chunk_dims: &[u64], filters: &[FilterKind]) -> ChunkOptions {
        ChunkOptions {
            chunk_dims: Some(chunk_dims.to_vec()),
            ..filtered(filters)
        }
    }

    fn f64_to_bytes(data: &[f64]) -> Vec<u8> {
        let mut b = Vec::with_capacity(data.len() * 8);
        for &v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }

    fn bytes_to_f64(data: &[u8]) -> Vec<f64> {
        data.chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[rstest]
    #[case(&[64], &[64], None, [16, 28], 17)]
    #[case(&[128], &[32], None, [17, 17], 17)]
    #[case(&[128], &[32], Some(&[MaxExtent::Unlimited][..]), [21, 21], 21)]
    #[case(&[2, 3], &[2, 2], None, [18, 18], 18)]
    fn header_lengths_and_filter_parameters_match_geometry(
        #[case] shape: &[u64],
        #[case] chunk_dims: &[u64],
        #[case] maxshape: Option<&[MaxExtent]>,
        #[case] allocated_lengths: [usize; 2],
        #[case] unallocated_length: usize,
        #[values(false, true)] filtered: bool,
        #[values(false, true)] varied: bool,
        #[values(StorageAllocation::Allocated, StorageAllocation::Unallocated)]
        allocation: StorageAllocation,
    ) {
        let mut options = ChunkOptions::default();
        if filtered {
            options.set_filter(FilterKind::Shuffle);
            options.set_filter(FilterKind::Deflate(DeflateLevel::Six));
        }
        let ctx = ChunkContext::basic(chunk_dims, 8);
        let header = options
            .measure_header(shape, ctx, maxshape, FillPattern::ZERO, allocation)
            .unwrap();
        let count = shape.iter().product::<u64>().to_usize().unwrap() * 8;
        let raw: Vec<_> = (0..count)
            .map(|index| {
                if varied {
                    u8::try_from(index % 251).unwrap()
                } else {
                    0
                }
            })
            .collect();
        let chunks = compress_chunks(
            &raw,
            shape,
            ctx,
            &options,
            maxshape,
            FillPattern::ZERO,
            allocation,
        )
        .unwrap();
        let measured = measure_chunked_at(&chunks, StoredAddress::new(0x1234)).unwrap();
        assert_eq!(
            header.layout_message.len(),
            match allocation {
                StorageAllocation::Allocated => allocated_lengths[usize::from(filtered)],
                StorageAllocation::Unallocated => unallocated_length,
            }
        );
        assert_eq!(
            header.pipeline_message.as_ref().map(|message| {
                FilterPipeline::parse(message)
                    .unwrap()
                    .filters
                    .into_iter()
                    .map(|filter| (filter.filter_id, filter.client_data))
                    .collect::<Vec<_>>()
            }),
            filtered.then_some(vec![(FILTER_SHUFFLE, vec![8]), (FILTER_DEFLATE, vec![6])])
        );
        assert_eq!(header.layout_message.len(), measured.layout_message.len());
        assert_eq!(header.pipeline_message, measured.pipeline_message);
    }

    /// Measuring a chunked region and assembling it must agree on its length and
    /// its layout message, at any address.
    ///
    /// This is the invariant the object-header sizing pass rests on: it asks
    /// `measure_chunked_at` how long the region will be, writes a header sized
    /// to that answer, and only then asks `assemble_chunked_at` for the bytes.
    /// A disagreement is not a failed write — it is a file whose header points
    /// somewhere the data is not, which every reader accepts and misreads.
    ///
    /// Measuring no longer builds the index to find its length (issue #228), so
    /// the two answers now come from different code. Checked across all three
    /// index kinds, since each has its own length rule, and at more than one
    /// address, since only the layout message depends on the address.
    #[test]
    fn measuring_a_chunked_region_agrees_with_assembling_it() {
        /// Dataset shape, chunk shape, and maxshape: the three inputs that
        /// decide which index kind a set gets.
        type Case = (&'static [u64], &'static [u64], Option<&'static [MaxExtent]>);

        // Chunk counts chosen for the kind each selects: one chunk is
        // `SingleChunk`, a fixed shape with many is `FixedArray`, and an
        // unlimited maxshape is `ExtensibleArray`.
        let cases: [Case; 4] = [
            (&[512], &[512], None),
            (&[4096], &[512], None),
            (&[4096], &[64], None),
            (&[4096], &[512], Some(&[MaxExtent::Unlimited])),
        ];

        for (shape, chunk_dims, maxshape) in cases {
            let elems: usize = shape.iter().product::<u64>().to_usize().unwrap();
            let raw = f64_to_bytes(&(0..elems).map(|i| i as f64).collect::<Vec<f64>>());
            let ctx = ChunkContext::basic(chunk_dims, 8);
            let set = compress_chunks(
                &raw,
                shape,
                ctx,
                &ChunkOptions::default(),
                maxshape,
                FillPattern::ZERO,
                StorageAllocation::Allocated,
            )
            .unwrap();

            for base in [0u64, 0x1000, 0x1234_5678].map(StoredAddress::new) {
                let measured = measure_chunked_at(&set, base).unwrap();
                let assembled = assemble_chunked_at(&set, base).unwrap();
                assert_eq!(
                    measured.data_len,
                    assembled.data_bytes.len() as u64,
                    "measured and assembled lengths differ for shape {shape:?} in \
                     chunks {chunk_dims:?} at {:#x}",
                    base.get()
                );
                assert_eq!(
                    measured.layout_message,
                    assembled.layout_message,
                    "measured and assembled layout messages differ for shape \
                     {shape:?} in chunks {chunk_dims:?} at {:#x}",
                    base.get()
                );
            }
        }
    }

    /// Helper: build a chunked file blob and read it back using read_chunked_data
    fn roundtrip_chunked(
        values: &[f64],
        shape: &[u64],
        chunk_dims: &[u64],
        options: &ChunkOptions,
    ) -> Vec<f64> {
        let raw = f64_to_bytes(values);
        let data_address = StoredAddress::new(0x1000);
        let ctx = ChunkContext::basic(chunk_dims, 8);
        let result = build_chunked_data_at_ext(
            &raw,
            shape,
            ctx,
            options,
            data_address,
            None,
            FillPattern::ZERO,
        )
        .unwrap();

        // Build a fake file buffer
        let file_size = data_address.get() as usize + result.data_bytes.len();
        let mut file_data = vec![0u8; file_size];
        file_data[data_address.get() as usize..].copy_from_slice(&result.data_bytes);

        // Parse layout
        let layout = DataLayout::parse(&result.layout_message, 8, 8).unwrap();
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: shape.len() as u8,
            dimensions: shape.to_vec(),
            max_dimensions: None,
        };
        let datatype = make_f64_type();

        // Parse pipeline if present
        let pipeline = result
            .pipeline_message
            .as_ref()
            .map(|pm| crate::filter_pipeline::FilterPipeline::parse(pm).unwrap());

        let output = chunked_read::read_chunked_data_cached(
            &file_data,
            RawReadSpec::parse(
                &layout,
                &dataspace,
                &datatype,
                pipeline.as_ref(),
                FillPattern::ZERO,
            )
            .unwrap(),
            crate::file_writer::WIDTHS,
            &ChunkCache::new(),
        )
        .unwrap();

        bytes_to_f64(&output)
    }

    #[test]
    fn split_1d_single_chunk() {
        let data = f64_to_bytes(&[1.0, 2.0, 3.0]);
        let result =
            split_into_chunks(&data, &[3], &[3], convert::nz(8), FillPattern::ZERO).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(bytes_to_f64(&result[0]), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn split_1d_multiple_chunks() {
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let data = f64_to_bytes(&values);
        let result =
            split_into_chunks(&data, &[10], &[4], convert::nz(8), FillPattern::ZERO).unwrap();
        assert_eq!(result.len(), 3); // ceil(10/4) = 3
        // Contents in chunk order, which is what the offsets this used to return
        // encoded: chunk `i` starts at element `4 * i`.
        assert_eq!(bytes_to_f64(&result[0]), vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(bytes_to_f64(&result[1]), vec![4.0, 5.0, 6.0, 7.0]);
        // Last chunk: 2 valid + 2 padding zeros
        assert_eq!(bytes_to_f64(&result[2]), vec![8.0, 9.0, 0.0, 0.0]);
    }

    #[test]
    fn split_2d_chunks() {
        // 4x4 dataset, 2x2 chunks -> 4 chunks
        let values: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let data = f64_to_bytes(&values);
        let result =
            split_into_chunks(&data, &[4, 4], &[2, 2], convert::nz(8), FillPattern::ZERO).unwrap();
        assert_eq!(result.len(), 4);
        // Row-major chunk order, asserted by content rather than by the offsets
        // this used to return: every chunk, so the ordering is pinned end to end
        // and not just at its head.
        // chunk (0,0): elements [0,1,4,5]
        assert_eq!(bytes_to_f64(&result[0]), vec![0.0, 1.0, 4.0, 5.0]);
        // chunk (0,2): elements [2,3,6,7]
        assert_eq!(bytes_to_f64(&result[1]), vec![2.0, 3.0, 6.0, 7.0]);
        // chunk (2,0): elements [8,9,12,13]
        assert_eq!(bytes_to_f64(&result[2]), vec![8.0, 9.0, 12.0, 13.0]);
        // chunk (2,2): elements [10,11,14,15]
        assert_eq!(bytes_to_f64(&result[3]), vec![10.0, 11.0, 14.0, 15.0]);
    }

    #[test]
    fn roundtrip_1d_single_chunk_no_compression() {
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let options = ChunkOptions {
            chunk_dims: Some(vec![10]),
            ..Default::default()
        };
        let result = roundtrip_chunked(&values, &[10], &[10], &options);
        assert_eq!(result, values);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn roundtrip_1d_single_chunk_deflate() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let options = chunked(&[100], &[FilterKind::Deflate(DeflateLevel::Six)]);
        let result = roundtrip_chunked(&values, &[100], &[100], &options);
        assert_eq!(result, values);
    }

    #[test]
    fn roundtrip_1d_multi_chunk_no_compression() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let options = ChunkOptions {
            chunk_dims: Some(vec![8]),
            ..Default::default()
        };
        let result = roundtrip_chunked(&values, &[20], &[8], &options);
        assert_eq!(result, values);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn roundtrip_1d_multi_chunk_deflate() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let options = chunked(&[20], &[FilterKind::Deflate(DeflateLevel::Six)]);
        let result = roundtrip_chunked(&values, &[100], &[20], &options);
        assert_eq!(result, values);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn roundtrip_1d_shuffle_deflate() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let options = chunked(
            &[50],
            &[FilterKind::Shuffle, FilterKind::Deflate(DeflateLevel::Six)],
        );
        let result = roundtrip_chunked(&values, &[100], &[50], &options);
        assert_eq!(result, values);
    }

    #[test]
    fn roundtrip_2d_chunks() {
        // 6x4 dataset, 3x2 chunks
        let values: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let options = ChunkOptions {
            chunk_dims: Some(vec![3, 2]),
            ..Default::default()
        };
        let result = roundtrip_chunked(&values, &[6, 4], &[3, 2], &options);
        assert_eq!(result, values);
    }

    /// Chunks are stored back to back, and the index begins where the last
    /// chunk ends. The chunk size here (7 f64 = 56 bytes) is deliberately not a
    /// multiple of any cache line, so padding at *either* site — between chunks
    /// or before the index — would move bytes this pins.
    #[test]
    fn chunks_are_stored_back_to_back() {
        let values: Vec<f64> = (0..21).map(|i| i as f64).collect();
        let raw = f64_to_bytes(&values);
        let options = ChunkOptions {
            chunk_dims: Some(vec![7]),
            ..Default::default()
        };
        let dims = [7u64];
        let ctx = ChunkContext::basic(&dims, 8);
        let result = build_chunked_data_at_ext(
            &raw,
            &[21],
            ctx,
            &options,
            StoredAddress::new(0x1000),
            None,
            FillPattern::ZERO,
        )
        .unwrap();

        // Unfiltered chunks are stored verbatim, so the data region opens with
        // the raw bytes in order.
        assert_eq!(
            &result.data_bytes[..raw.len()],
            &raw[..],
            "the three chunks must concatenate with nothing between them"
        );
        // And the Fixed Array header starts in the very next byte. Asserting the
        // signature's position, rather than only the chunk prefix, is what keeps
        // padding from creeping back in ahead of the index.
        assert_eq!(
            &result.data_bytes[raw.len()..raw.len() + 4],
            b"FAHD",
            "the chunk index must begin where the last chunk ends"
        );
    }

    /// The same rule for the streaming path, stated where it is computed: the
    /// data region is exactly the chunks plus the index, so a plan that inserted
    /// padding would make `total_len` exceed the sum.
    #[test]
    fn a_verbatim_plan_reserves_only_the_chunks_and_the_index() {
        let meta: Vec<ChunkMeta> = [37u64, 111, 5]
            .into_iter()
            .map(|compressed_size| ChunkMeta {
                compressed_size,
                filter_mask: 0,
            })
            .collect();
        let layout = plan_chunked_data_verbatim(
            &meta,
            &[21],
            &[7],
            convert::nz(8),
            ChunkedLayoutFlags::NONE,
            Some(&[]),
            StoredAddress::new(0x1000),
            None,
        )
        .unwrap();

        let planned: Vec<u64> = layout.plan.chunks.iter().map(|c| c.stored_size).collect();
        assert_eq!(planned, vec![37, 111, 5]);

        // The region's length is planned without the index existing, so pin it
        // against the bytes the emit actually writes rather than against the
        // planner's own arithmetic.
        struct SizedChunks<'a>(&'a [u64]);
        impl ChunkProvider for SizedChunks<'_> {
            fn chunk_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), FormatError> {
                out.resize(self.0[index] as usize, 0xAB);
                Ok(())
            }
        }
        let sizes: Vec<u64> = meta.iter().map(|m| m.compressed_size).collect();
        let mut emitted: Vec<u8> = Vec::new();
        emit_chunked_data_verbatim(&mut emitted, &layout.plan, &SizedChunks(&sizes)).unwrap();
        assert_eq!(emitted.len() as u64, layout.plan.total_len);
        let chunk_bytes: u64 = sizes.iter().sum();
        assert!(
            layout.plan.total_len > chunk_bytes,
            "three chunks take a fixed array, so the region is longer than its chunk bytes"
        );

        // The index is built by the emit rather than by the plan, so each index
        // kind has to be emitted to be covered. Assert the signature at the
        // planned offset, not just the total: a plan that reserved the right
        // number of bytes for the wrong structure would pass a length check.
        for (label, shape, maxshape, chunk_sizes, signature) in [
            (
                "fixed array",
                &[21u64][..],
                None,
                &[37u64, 111, 5][..],
                Some(&b"FAHD"[..]),
            ),
            (
                "extensible array",
                &[21u64][..],
                Some(&[MaxExtent::Unlimited][..]),
                &[37u64, 111, 5][..],
                Some(&b"EAHD"[..]),
            ),
            // One chunk whose maximum shape is that one chunk is the
            // single-chunk layout: its address rides in the layout message and
            // nothing follows the chunk bytes at all.
            ("single chunk", &[7u64][..], None, &[37u64][..], None),
        ] {
            let meta: Vec<ChunkMeta> = chunk_sizes
                .iter()
                .map(|&compressed_size| ChunkMeta {
                    compressed_size,
                    filter_mask: 0,
                })
                .collect();
            let layout = plan_chunked_data_verbatim(
                &meta,
                shape,
                &[7],
                convert::nz(8),
                ChunkedLayoutFlags::NONE,
                Some(&[]),
                StoredAddress::new(0x1000),
                maxshape,
            )
            .unwrap();
            let mut emitted: Vec<u8> = Vec::new();
            emit_chunked_data_verbatim(&mut emitted, &layout.plan, &SizedChunks(chunk_sizes))
                .unwrap();

            let chunk_bytes: usize = chunk_sizes.iter().sum::<u64>() as usize;
            assert_eq!(
                emitted.len() as u64,
                layout.plan.total_len,
                "{label}: the emit must fill the planned region"
            );
            match signature {
                Some(sig) => assert_eq!(
                    &emitted[chunk_bytes..chunk_bytes + 4],
                    sig,
                    "{label}: the index must begin where the last chunk ends"
                ),
                None => assert_eq!(
                    emitted.len(),
                    chunk_bytes,
                    "{label}: nothing may follow the chunk bytes"
                ),
            }
        }
    }

    /// A chunk-less plan has no first chunk to anchor the index against, so it
    /// is refused rather than planned as an empty region.
    #[test]
    fn a_verbatim_plan_with_no_chunks_is_refused() {
        let result = plan_chunked_data_verbatim(
            &[],
            &[21],
            &[7],
            convert::nz(8),
            ChunkedLayoutFlags::NONE,
            None,
            StoredAddress::new(0x1000),
            None,
        );
        assert!(
            matches!(result, Err(FormatError::ChunkedReadError(_))),
            "a chunk-less plan must be refused"
        );
    }

    #[test]
    fn a_verbatim_chunk_too_large_for_its_index_field_fails() {
        let meta = [37, 70_000, 5].map(|compressed_size| ChunkMeta {
            compressed_size,
            filter_mask: 0,
        });
        let err = plan_chunked_data_verbatim(
            &meta,
            &[6],
            &[2],
            convert::nz(1),
            ChunkedLayoutFlags::NONE,
            Some(&[]),
            StoredAddress::new(0x1000),
            None,
        )
        .err()
        .unwrap();
        let FormatError::ChunkedReadError(message) = &err else {
            panic!("expected ChunkedReadError, got {err:?}");
        };
        assert_eq!(
            message,
            "chunk 1 of a verbatim chunked dataset stores 70000 bytes, more than the chunk index \
             records for a filtered chunk of 2 bytes"
        );
    }

    #[rstest]
    #[case::fixed_array(&[21], None, 3)]
    #[case::extensible_array(&[21], Some(&[MaxExtent::Unlimited][..]), 3)]
    #[case::single_chunk(&[7], None, 1)]
    fn a_verbatim_plan_stores_the_partial_edge_chunk_flag_it_is_given(
        #[case] shape: &[u64],
        #[case] maxshape: Option<&[MaxExtent]>,
        #[case] chunks: usize,
        #[values(false, true)] stored_raw: bool,
        #[values(None, Some(&[][..]))] pipeline_message: Option<&[u8]>,
    ) {
        let flags = if stored_raw {
            ChunkedLayoutFlags::new(DONT_FILTER_PARTIAL_BOUND_CHUNKS)
        } else {
            ChunkedLayoutFlags::NONE
        };
        let meta = vec![
            ChunkMeta {
                compressed_size: 56,
                filter_mask: 0,
            };
            chunks
        ];
        let layout = plan_chunked_data_verbatim(
            &meta,
            shape,
            &[7],
            convert::nz(8),
            flags,
            pipeline_message,
            StoredAddress::new(0x1000),
            maxshape,
        )
        .unwrap();
        let DataLayout::Chunked { flags, .. } =
            DataLayout::parse(&layout.layout_message, 8, 8).unwrap()
        else {
            panic!("a verbatim plan stores a chunked layout");
        };
        assert_eq!(flags.partial_edge_chunks_stored_raw(), stored_raw);
    }

    #[test]
    fn chunk_options_auto_dims() {
        let options = filtered(&[FilterKind::Deflate(DeflateLevel::Six)]);
        let dims = options.resolve_chunk_dims(&[100, 50]);
        assert_eq!(dims, vec![100, 50]);
    }

    /// The two fill-availability settings, through the pipeline builder.
    ///
    /// The default records the dataset's fill value; the `Undefined` setting
    /// records none *and drops the value*, which is what a repack of a source
    /// that recorded none needs. A setting that carried the value
    /// through anyway would leave a filter claiming no fill value with one
    /// sitting in the parameters after it.
    #[test]
    fn the_scale_offset_fill_availability_reaches_the_filter_parameters() {
        let ctx = f64_ctx(&[8]);
        let elem = NonZeroUsize::new(8).expect("8 is non-zero");
        let fill = 2.5f64.to_le_bytes();
        let parms = |fill_availability, fill: FillPattern<'_>| {
            let options = filtered(&[FilterKind::ScaleOffset(
                ScaleOffset::FloatDScale(2),
                fill_availability,
            )]);
            let pl = options.build_pipeline(&ctx, fill).unwrap().unwrap();
            let f = pl
                .filters
                .iter()
                .find(|f| f.filter_id == FILTER_SCALEOFFSET)
                .expect("the scale-offset filter");
            // FILAVAIL, then the two entries the value can occupy.
            (f.client_data[7], f.client_data[8], f.client_data[9])
        };

        // 2.5 as an `f64` bit pattern, split across the two entries.
        let bits = 2.5f64.to_bits();
        assert_eq!(
            parms(
                FillAvailability::Defined,
                FillPattern::new(FillTime::IfSet, Some(&fill), elem)
            ),
            (1, bits as u32, (bits >> 32) as u32)
        );
        // The library default is a defined fill value of zero.
        assert_eq!(
            parms(FillAvailability::Defined, FillPattern::ZERO),
            (1, 0, 0)
        );
        // Undefined records neither, even when a fill value is available.
        assert_eq!(
            parms(
                FillAvailability::Undefined,
                FillPattern::new(FillTime::IfSet, Some(&fill), elem)
            ),
            (0, 0, 0)
        );
        // And a fill value that could not be read refuses the pipeline rather
        // than recording a zero the encoder would treat every real zero as.
        let options = filtered(&[FilterKind::ScaleOffset(
            ScaleOffset::FloatDScale(2),
            FillAvailability::Defined,
        )]);
        assert!(matches!(
            options.build_pipeline(&ctx, FillPattern::UNKNOWN),
            Err(FormatError::UnreadableFillValue)
        ));
        // But only the setting that would record it refuses: a filter that
        // records no fill value has nowhere to put one, so a value it could not
        // read is not an obstacle to saying there is none.
        let undefined = filtered(&[FilterKind::ScaleOffset(
            ScaleOffset::FloatDScale(2),
            FillAvailability::Undefined,
        )]);
        assert!(undefined.build_pipeline(&ctx, FillPattern::UNKNOWN).is_ok());

        // The default is the reference library's: every scale-offset dataset it
        // writes records a fill value, including one whose fill value was never
        // set. Asserted through the builder switch, which is the one place that
        // chooses an availability on the caller's behalf — and asserted here as
        // well as in the crosschecks because those are gated to little-endian
        // 64-bit targets, where the `cross` jobs run nothing that would notice
        // this flipping back.
        let mut db = crate::type_builders::DatasetBuilder::new("d");
        db.with_scale_offset(ScaleOffset::FloatDScale(2));
        assert_eq!(
            db.chunk_options.filters,
            vec![FilterSpec {
                kind: FilterKind::ScaleOffset(
                    ScaleOffset::FloatDScale(2),
                    FillAvailability::Defined
                ),
                optional: false,
            }]
        );
        // Every other pipeline is buildable from the same pattern: only the
        // filter that records the fill value needs to read it.
        let plain = filtered(&[FilterKind::Deflate(DeflateLevel::Six)]);
        assert!(plain.build_pipeline(&ctx, FillPattern::UNKNOWN).is_ok());
    }

    #[rstest]
    #[case(DeflateLevel::Zero)]
    #[case(DeflateLevel::Six)]
    #[case(DeflateLevel::Nine)]
    fn pipeline_preserves_deflate_level_parameters(#[case] level: DeflateLevel) {
        let options = filtered(&[FilterKind::Deflate(level)]);
        assert_eq!(options.filters[0].kind, FilterKind::Deflate(level));
        let pl = options
            .build_pipeline(&ChunkContext::basic(&[], 8), FillPattern::ZERO)
            .unwrap()
            .unwrap();
        assert_eq!(pl.filters.len(), 1);
        assert_eq!(pl.filters[0].filter_id, FILTER_DEFLATE);
        assert_eq!(pl.filters[0].client_data, [u32::from(level.get())]);
    }

    #[test]
    fn chunk_options_pipeline_shuffle_deflate_fletcher32() {
        let options = filtered(&[
            FilterKind::Shuffle,
            FilterKind::Deflate(DeflateLevel::Six),
            FilterKind::Fletcher32,
        ]);
        let pl = options
            .build_pipeline(&ChunkContext::basic(&[], 8), FillPattern::ZERO)
            .unwrap()
            .unwrap();
        assert_eq!(pl.filters.len(), 3);
        assert_eq!(pl.filters[0].filter_id, FILTER_SHUFFLE);
        assert_eq!(pl.filters[1].filter_id, FILTER_DEFLATE);
        assert_eq!(pl.filters[2].filter_id, FILTER_FLETCHER32);
    }

    // Repeated setters replace the earlier level at its original position.
    #[test]
    fn setting_filters_places_them_in_canonical_order_whatever_order_they_arrive_in() {
        let ids = |o: &ChunkOptions| -> Vec<FilterId> {
            o.filters.iter().map(|f| f.kind.filter_id()).collect()
        };
        let canonical = [FilterId::Shuffle, FilterId::Deflate, FilterId::Fletcher32];

        assert_eq!(
            ids(&filtered(&[
                FilterKind::Shuffle,
                FilterKind::Deflate(DeflateLevel::Six),
                FilterKind::Fletcher32,
            ])),
            canonical
        );
        assert_eq!(
            ids(&filtered(&[
                FilterKind::Fletcher32,
                FilterKind::Deflate(DeflateLevel::Six),
                FilterKind::Shuffle,
            ])),
            canonical
        );

        // Naming one twice sets its parameter, rather than applying it twice.
        let twice = filtered(&[
            FilterKind::Deflate(DeflateLevel::One),
            FilterKind::Shuffle,
            FilterKind::Deflate(DeflateLevel::Nine),
        ]);
        assert_eq!(ids(&twice), [FilterId::Shuffle, FilterId::Deflate]);
        assert_eq!(
            twice.filters[1].kind,
            FilterKind::Deflate(DeflateLevel::Nine)
        );
    }

    /// A pushed pipeline reaches the file in the order and with the optional
    /// flags it was pushed with, not this crate's canonical order (issue #333).
    ///
    /// `fletcher32` between shuffle and deflate is the case that matters: it
    /// checksums the shuffled bytes, where the canonical placement at the end
    /// would checksum the deflated ones. That is a different checksum over
    /// different data, so a `repack` that quietly re-ordered it produced a
    /// destination verifying something the source never did.
    #[test]
    fn a_pushed_pipeline_keeps_its_order_and_its_optional_flags() {
        let mut options = ChunkOptions::default();
        for (kind, optional) in [
            (FilterKind::Shuffle, true),
            (FilterKind::Fletcher32, false),
            (FilterKind::Deflate(DeflateLevel::Four), true),
        ] {
            options.push_filter(FilterSpec { kind, optional });
        }

        let pl = options
            .build_pipeline(&ChunkContext::basic(&[], 8), FillPattern::ZERO)
            .unwrap()
            .unwrap();
        let stored: Vec<(u16, u16)> = pl.filters.iter().map(|f| (f.filter_id, f.flags)).collect();
        assert_eq!(
            stored,
            [
                (FILTER_SHUFFLE, 1),
                (FILTER_FLETCHER32, 0),
                (FILTER_DEFLATE, 1),
            ]
        );
    }

    #[test]
    fn conflicting_filter_requests_report_both_filter_names() {
        let so = ScaleOffset::FloatDScale(2);
        let fill = FillAvailability::Defined;
        let cases: &[(&str, &str, ChunkOptions)] = &[
            (
                "lzf",
                "deflate",
                filtered(&[FilterKind::Lzf, FilterKind::Deflate(DeflateLevel::Six)]),
            ),
            (
                "shuffle",
                "scale-offset",
                filtered(&[FilterKind::Shuffle, FilterKind::ScaleOffset(so, fill)]),
            ),
        ];

        for (a, b, options) in cases {
            // Arguments a valid ZFP or scale-offset request would need. The
            // clash has to be reported whether or not they are satisfiable, so
            // pass ones that are: an error raised only because the datatype was
            // also wrong would pass an `is_err()` check while leaving the
            // combination itself unrefused.
            let Err(err) = options.build_pipeline(&f64_ctx(&[64]), FillPattern::ZERO) else {
                panic!("{a} + {b} was accepted");
            };
            let FormatError::FilterError(msg) = &err else {
                panic!("{a} + {b}: expected a filter error, got {err}");
            };
            assert!(msg.contains(a) && msg.contains(b), "{a} + {b}: {msg}");
        }
    }

    /// A context over `f64` elements carrying both type-aware filters' facts.
    ///
    /// The clash fixtures need a request that a valid ZFP or scale-offset
    /// pipeline could actually satisfy: an error raised only because the
    /// datatype was wrong would pass an `is_err()` check while leaving the
    /// combination itself unrefused.
    fn f64_ctx(chunk_dims: &[u64]) -> ChunkContext<'_> {
        ChunkContext {
            chunk_dims,
            element_size: core::num::NonZeroU32::new(8).expect("8 is non-zero"),
            element_type: crate::filters::zfp_element_type_from_datatype(&make_f64_type()),
            scale_offset_type: crate::scaleoffset::scale_offset_type_from_datatype(&make_f64_type()),
        }
    }

    /// Chaining a primary transform with a filter it does *not* displace still
    /// builds, so the refusal above is a rule about conflicts rather than a
    /// blanket ban on combining filters.
    #[test]
    fn compatible_filter_requests_still_build() {
        let so = FilterKind::ScaleOffset(ScaleOffset::FloatDScale(2), FillAvailability::Defined);
        let cases: &[(ChunkOptions, &[u16])] = &[
            (
                filtered(&[FilterKind::Shuffle, FilterKind::Deflate(DeflateLevel::Six)]),
                &[FILTER_SHUFFLE, FILTER_DEFLATE],
            ),
            (
                filtered(&[FilterKind::Shuffle, FilterKind::Lzf]),
                &[FILTER_SHUFFLE, FILTER_LZF],
            ),
            (
                filtered(&[so, FilterKind::Deflate(DeflateLevel::Six)]),
                &[FILTER_SCALEOFFSET, FILTER_DEFLATE],
            ),
            (
                filtered(&[so, FilterKind::Lzf, FilterKind::Fletcher32]),
                &[FILTER_SCALEOFFSET, FILTER_LZF, FILTER_FLETCHER32],
            ),
        ];

        for (options, expected) in cases {
            let pl = options
                .build_pipeline(&f64_ctx(&[64]), FillPattern::ZERO)
                .unwrap()
                .unwrap();
            let ids: Vec<u16> = pl.filters.iter().map(|f| f.filter_id).collect();
            assert_eq!(&ids, expected);
        }
    }

    /// Helper: roundtrip with EA (maxshape)
    fn roundtrip_ea(
        values: &[f64],
        shape: &[u64],
        chunk_dims: &[u64],
        maxshape: &[MaxExtent],
    ) -> Vec<f64> {
        let raw = f64_to_bytes(values);
        let data_address = StoredAddress::new(0x1000);
        let options = ChunkOptions {
            chunk_dims: Some(chunk_dims.to_vec()),
            ..Default::default()
        };
        let ctx = ChunkContext::basic(chunk_dims, 8);
        let result = build_chunked_data_at_ext(
            &raw,
            shape,
            ctx,
            &options,
            data_address,
            Some(maxshape),
            FillPattern::ZERO,
        )
        .unwrap();

        let file_size = data_address.get() as usize + result.data_bytes.len();
        let mut file_data = vec![0u8; file_size];
        file_data[data_address.get() as usize..].copy_from_slice(&result.data_bytes);

        let layout = DataLayout::parse(&result.layout_message, 8, 8).unwrap();
        assert!(
            matches!(
                &layout,
                DataLayout::Chunked {
                    index: ChunkIndexLayout::ExtensibleArray { .. },
                    ..
                }
            ),
            "{layout:?}"
        );

        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: shape.len() as u8,
            dimensions: shape.to_vec(),
            max_dimensions: Some(maxshape.to_vec()),
        };
        let datatype = make_f64_type();

        let output = chunked_read::read_chunked_data_cached(
            &file_data,
            RawReadSpec::plain(&layout, &dataspace, &datatype),
            crate::file_writer::WIDTHS,
            &ChunkCache::new(),
        )
        .unwrap();

        bytes_to_f64(&output)
    }

    #[test]
    fn ea_roundtrip_1d_inline_only() {
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let result = roundtrip_ea(&values, &[10], &[10], &[MaxExtent::Unlimited]);
        assert_eq!(result, values);
    }

    #[test]
    fn ea_roundtrip_1d_multi_chunks() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let result = roundtrip_ea(&values, &[20], &[5], &[MaxExtent::Unlimited]);
        assert_eq!(result, values);
    }

    #[test]
    fn ea_roundtrip_1d_many_chunks() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = roundtrip_ea(&values, &[100], &[10], &[MaxExtent::Unlimited]);
        assert_eq!(result, values);
    }

    /// One chunk per element across the inline, direct-data-block, and
    /// super-block ranges. Before the geometry fix these silently corrupted
    /// past 20 chunks (4 inline + the first 16-element direct block).
    #[test]
    fn ea_roundtrip_super_block_sizes() {
        for &n in &[245u64, 300, 2000, 50000] {
            let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let result = roundtrip_ea(&values, &[n], &[1], &[MaxExtent::Unlimited]);
            assert_eq!(result.len(), n as usize, "length mismatch at n={n}");
            assert_eq!(result, values, "data mismatch at n={n}");
        }
    }

    /// Cross the paging boundary (131060 = 4 inline + 240 direct + super blocks
    /// SB4..SB12), exercising paged data blocks in super block 13 (the first
    /// whose data blocks exceed 1024 elements) on both write and read.
    #[test]
    fn ea_roundtrip_paged_data_blocks() {
        let n: u64 = 132_000;
        let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let result = roundtrip_ea(&values, &[n], &[1], &[MaxExtent::Unlimited]);
        assert_eq!(result.len(), n as usize);
        assert_eq!(result, values);
    }

    /// `chunked_data_len` is the span the in-place editor reserves for a whole
    /// chunked data region before assembling it, so it has to equal the length
    /// `assemble_chunked_at` then produces.
    ///
    /// `WriteEngine::place` refuses a mismatch, so a wrong length here is a
    /// failed write rather than a corrupt file — but it is still a failed write,
    /// and the edit path that would hit it is not in the fast loop. Swept across
    /// the three index kinds a chunk set can take: one chunk (no index at all),
    /// several (a Fixed Array), and an unlimited dimension (an Extensible Array),
    /// filtered and not.
    #[test]
    fn chunked_data_len_matches_what_assemble_produces() {
        for &(elements, chunk) in &[
            (1u64, 8u64), // a single chunk: no index
            (21, 7),      // three chunks: a fixed array
            (8_192, 4),   // enough chunks to page the fixed array
        ] {
            for &deflate in &[false, true] {
                for &unlimited in &[false, true] {
                    let values: Vec<f64> = (0..elements).map(|i| i as f64).collect();
                    let raw = f64_to_bytes(&values);
                    let mut options = chunked(&[chunk], &[]);
                    if deflate {
                        options.set_filter(FilterKind::Deflate(DeflateLevel::Six));
                    }
                    let dims = [chunk];
                    let maxshape = unlimited.then_some([MaxExtent::Unlimited]);
                    let set = compress_chunks(
                        &raw,
                        &[elements],
                        ChunkContext::basic(&dims, 8),
                        &options,
                        maxshape.as_ref().map(<[MaxExtent; 1]>::as_slice),
                        FillPattern::ZERO,
                        StorageAllocation::Allocated,
                    )
                    .unwrap();

                    let planned = chunked_data_len(&set).unwrap();
                    let assembled =
                        assemble_chunked_at(&set, StoredAddress::new(0x10_0000)).unwrap();
                    assert_eq!(
                        planned,
                        assembled.data_bytes.len() as u64,
                        "planned region must match the assembled one at elements={elements}, \
                         chunk={chunk}, deflate={deflate}, unlimited={unlimited}"
                    );
                }
            }
        }
    }

    #[rstest]
    #[case::single(&[64], &[64], None)]
    #[case::fixed(&[256], &[32], None)]
    #[case::fixed_paged(&[8_256], &[1], None)]
    #[case::extensible(&[256], &[32], Some(&[MaxExtent::Unlimited][..]))]
    #[case::extensible_blocks(&[2_048], &[8], Some(&[MaxExtent::Unlimited][..]))]
    #[case::partial(&[3, 5], &[2, 3], None)]
    #[case::maximum_grid(&[3, 5], &[2, 3], Some(&[MaxExtent::Fixed(8), MaxExtent::Fixed(12)][..]))]
    #[case::empty(&[0], &[8], Some(&[MaxExtent::Unlimited][..]))]
    fn direct_emission_uses_retained_chunks_and_matches_assembly(
        #[case] shape: &[u64],
        #[case] chunk_dims: &[u64],
        #[case] maxshape: Option<&[MaxExtent]>,
        #[values(false, true)] filtered: bool,
        #[values(PartialEdgeFiltering::Apply, PartialEdgeFiltering::Skip)]
        edges: PartialEdgeFiltering,
        #[values(StorageAllocation::Allocated, StorageAllocation::Unallocated)]
        allocation: StorageAllocation,
        #[values(0, 0x1234_5678)] address: u64,
    ) {
        let mut options = chunked(chunk_dims, &[]);
        if filtered {
            options.set_filter(FilterKind::Shuffle);
            options.set_filter(FilterKind::Fletcher32);
        }
        options.partial_edge_filtering = Some(edges);
        let raw: Vec<_> = (0..shape.iter().product::<u64>())
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let set = compress_chunks(
            &raw,
            shape,
            ChunkContext::basic(chunk_dims, 8),
            &options,
            maxshape,
            FillPattern::ZERO,
            allocation,
        )
        .unwrap();
        let address = StoredAddress::new(address);
        let measured = measure_chunked_at(&set, address).unwrap();
        let plan = plan_chunked_at(&set, address).unwrap();
        let assembled = assemble_chunked_at(&set, address).unwrap();
        let mut sink = ChunkSlices::default();
        emit_chunked_data(&mut sink, &set, &plan).unwrap();
        assert_eq!(sink.bytes, assembled.data_bytes);
        assert_eq!(measured.data_len, plan.data_len);
        assert_eq!(plan.data_len, sink.position());
        assert_eq!(plan.layout_message, assembled.layout_message);
        assert_eq!(plan.layout_message, measured.layout_message);
        assert_eq!(plan.pipeline_message, assembled.pipeline_message);
        assert_eq!(plan.pipeline_message, measured.pipeline_message);
        let expected: Vec<_> = set
            .chunks
            .iter()
            .map(|chunk| (chunk.bytes().as_ptr(), chunk.bytes().len()))
            .chain(core::iter::once((plan.index.as_ptr(), plan.index.len())))
            .collect();
        assert_eq!(sink.slices, expected);
    }

    #[derive(Default)]
    struct ChunkSlices {
        slices: Vec<(*const u8, usize)>,
        bytes: Vec<u8>,
    }

    impl ByteSink for ChunkSlices {
        type Error = FormatError;

        fn put(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
            self.slices.push((bytes.as_ptr(), bytes.len()));
            self.bytes.extend_from_slice(bytes);
            Ok(())
        }

        fn put_zeros(&mut self, _: usize) -> Result<(), FormatError> {
            Err(FormatError::Internal("unexpected chunk padding".into()))
        }

        fn position(&self) -> u64 {
            self.bytes.len() as u64
        }
    }

    #[rstest]
    #[case::payload(false)]
    #[case::index(true)]
    fn chunk_planning_rejects_region_address_overflow(#[case] indexed: bool) {
        let chunk_dims = if indexed { &[8][..] } else { &[16][..] };
        let set = compress_chunks(
            &[0; 16],
            &[16],
            ChunkContext::basic(chunk_dims, 1),
            &ChunkOptions::default(),
            None,
            FillPattern::ZERO,
            StorageAllocation::Allocated,
        )
        .unwrap();
        let total = chunked_data_len(&set).unwrap();
        let address = StoredAddress::new(u64::MAX - total + 1);
        let expected = FormatError::OffsetOverflow {
            offset: if indexed {
                address.get() + 16
            } else {
                address.get()
            },
            length: if indexed { total - 16 } else { 16 },
        };
        assert_eq!(measure_chunked_at(&set, address).err().unwrap(), expected);
        assert_eq!(plan_chunked_at(&set, address).err().unwrap(), expected);
        assert_eq!(assemble_chunked_at(&set, address).err().unwrap(), expected);
    }

    #[rstest]
    #[case::first_chunk(0)]
    #[case::second_chunk(1)]
    #[case::index(2)]
    fn chunk_emission_stops_at_the_sink_error(
        #[case] fail_at: usize,
        #[values(false, true)] verbatim: bool,
    ) {
        let set = compress_chunks(
            &[0; 16],
            &[16],
            ChunkContext::basic(&[8], 1),
            &ChunkOptions::default(),
            None,
            FillPattern::ZERO,
            StorageAllocation::Allocated,
        )
        .unwrap();
        let plan = plan_chunked_at(&set, StoredAddress::new(64)).unwrap();
        let mut sink = FailingChunkSink {
            accepted: 0,
            fail_at,
        };
        let error = if verbatim {
            let layout = plan_chunked_data_verbatim(
                &core::array::from_fn::<_, 2, _>(|_| ChunkMeta {
                    compressed_size: 8,
                    filter_mask: 0,
                }),
                &[16],
                &[8],
                convert::nz(1),
                ChunkedLayoutFlags::NONE,
                None,
                StoredAddress::new(64),
                None,
            )
            .unwrap();
            emit_chunked_data_verbatim(&mut sink, &layout.plan, &EightByteChunks).unwrap_err()
        } else {
            emit_chunked_data(&mut sink, &set, &plan).unwrap_err()
        };
        assert_eq!(error, EmissionError::Sink(ChunkSinkFailure::Write(fail_at)));
        assert_eq!(sink.accepted, fail_at);
    }

    struct EightByteChunks;

    impl ChunkProvider for EightByteChunks {
        fn chunk_bytes(&self, _: usize, out: &mut Vec<u8>) -> Result<(), FormatError> {
            out.extend_from_slice(&[0; 8]);
            Ok(())
        }
    }

    #[derive(Debug, PartialEq)]
    enum ChunkSinkFailure {
        Padding,
        Write(usize),
    }

    struct FailingChunkSink {
        accepted: usize,
        fail_at: usize,
    }

    impl ByteSink for FailingChunkSink {
        type Error = ChunkSinkFailure;

        fn put(&mut self, _: &[u8]) -> Result<(), Self::Error> {
            if self.accepted == self.fail_at {
                return Err(ChunkSinkFailure::Write(self.fail_at));
            }
            self.accepted += 1;
            Ok(())
        }

        fn put_zeros(&mut self, _: usize) -> Result<(), Self::Error> {
            Err(ChunkSinkFailure::Padding)
        }

        fn position(&self) -> u64 {
            0
        }
    }
}
