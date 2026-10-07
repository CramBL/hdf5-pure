//! Chunked dataset writing: chunk splitting, compression, index building.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

/// The HDF5 "undefined address" sentinel (`HADDR_UNDEF`): all bits set, in
/// whatever width the field is. A chunked layout message carrying it declares
/// that the dataset has no storage allocated at all.
const HADDR_UNDEF: u64 = u64::MAX;
use core::num::NonZeroUsize;

use hdf5_pure_format::__private::ChunkIndexInfo;
use hdf5_pure_format::__private::ChunkRecord;
use hdf5_pure_format::__private::FilteredSingleChunk;
use hdf5_pure_format::__private::IndexSlots;
use hdf5_pure_format::__private::SlotOccupancy;

use crate::address::StoredAddress;
use crate::chunk_grid::{ChunkGrid, GridOrder};
use crate::convert::Narrow;
use crate::data_layout::DataLayout;
use crate::dataspace::{Extent, MaxExtent};
use crate::error::FormatError;
use crate::fill_value::FillPattern;
#[cfg(feature = "zfp")]
use crate::filter_pipeline::FILTER_ZFP;
use crate::filter_pipeline::{
    FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_LZF, FILTER_SCALEOFFSET, FILTER_SHUFFLE,
    FilterDescription, FilterPipeline, H5Z_FLAG_OPTIONAL,
};
use crate::filters::{ChunkContext, compress_chunk_with};
use crate::scaleoffset::{FillAvailability, ScaleOffset, build_cd_values};
use crate::width::LengthWidth;
use crate::width::OffsetWidth;

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
    Zfp(f64),
    /// Scale-offset, with whether the filter records the dataset's fill value.
    /// Also a primary transform: it displaces shuffle, but a byte compressor
    /// may follow it.
    ScaleOffset(ScaleOffset, FillAvailability),
    /// Byte shuffle.
    Shuffle,
    /// The h5py LZF filter (id 32000).
    Lzf,
    /// Deflate, at the given level (0-9).
    Deflate(u32),
    /// Fletcher32 checksum.
    Fletcher32,
}

impl FilterKind {
    /// The registered HDF5 filter id this kind is written as.
    ///
    /// Identity for every purpose that asks "is this filter in the pipeline":
    /// two `Deflate`s at different levels are one filter appearing twice, not
    /// two filters, which is what makes [`ChunkOptions::set_filter`] replace
    /// rather than accumulate.
    pub(crate) fn filter_id(self) -> u16 {
        match self {
            #[cfg(feature = "zfp")]
            Self::Zfp(_) => FILTER_ZFP,
            Self::ScaleOffset(..) => FILTER_SCALEOFFSET,
            Self::Shuffle => FILTER_SHUFFLE,
            Self::Lzf => FILTER_LZF,
            Self::Deflate(_) => FILTER_DEFLATE,
            Self::Fletcher32 => FILTER_FLETCHER32,
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
}

impl ChunkOptions {
    /// Whether any chunking option is enabled.
    pub fn is_chunked(&self) -> bool {
        self.chunk_dims.is_some() || !self.filters.is_empty()
    }

    /// Add `kind` at the position this crate's own writer gives it, replacing
    /// any filter with the same id already present.
    ///
    /// This is what the `DatasetBuilder` switches (`with_shuffle`,
    /// `with_deflate`, ...) call. Canonical filter order keeps
    /// `with_deflate(6).with_shuffle()` and
    /// `with_shuffle().with_deflate(6)` writing the same bytes: those setters
    /// name a filter to apply, not a position to apply it at.
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
            self.filters.iter().map(|f| f.kind.filter_id()),
            kind.filter_id(),
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
            .any(|f| f.kind.filter_id() == FILTER_DEFLATE)
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
            self.filters.iter().map(|f| f.kind.filter_id()),
        ) else {
            return Ok(());
        };
        clash(first, second)
    }

    /// Build a FilterPipeline from the options.
    ///
    /// The filters come out in [`filters`](Self::filters) order, carrying their
    /// optional flags: that is the order they are applied in and the order the
    /// message stores, and it is not always this crate's canonical one — see
    /// the field for why.
    ///
    /// The context's chunk dimensions and element type are only consulted when
    /// the ZFP filter is active — they're embedded into the ZFP cd_values so the
    /// resulting file is readable by the reference H5Z-ZFP plugin.
    ///
    /// `fill` is the dataset's fill value. Scale-offset records it in its
    /// parameters, so it is part of the pipeline rather than of the data: the
    /// same dataset written with and without a fill value carries two different
    /// filters, and the encoder diverts elements equal to that value to a
    /// reserved sentinel. It is consulted only by the filter that records it,
    /// so a pattern that could not be read
    /// ([`FormatError::UnreadableFillValue`]) refuses a scale-offset pipeline
    /// and leaves every other one buildable.
    ///
    /// Returns [`FormatError::UnsupportedZfp`] when ZFP was requested but the
    /// context's element type is `None` (e.g. the dataset's datatype isn't one
    /// of f32/f64/i32/i64), or the chunk rank is outside 1..=4, and
    /// [`FormatError::FilterError`] for a combination of filters where one
    /// would displace another — see [`refuse_conflicting_filters`] — or for a
    /// fill value whose length is not one element.
    ///
    /// [`refuse_conflicting_filters`]: Self::refuse_conflicting_filters
    pub fn build_pipeline(
        &self,
        ctx: &ChunkContext<'_>,
        fill: FillPattern<'_>,
    ) -> Result<Option<FilterPipeline>, FormatError> {
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
                FilterKind::Zfp(rate) => {
                    let elem_ty = ctx.element_type.ok_or_else(|| {
                        FormatError::UnsupportedZfp(
                            "ZFP compression requires the dataset's datatype to be one \
                             of f32, f64, i32, or i64"
                                .into(),
                        )
                    })?;
                    FilterDescription {
                        filter_id: FILTER_ZFP,
                        name: Some("zfp".into()),
                        flags,
                        client_data: hdf5_pure_filter::__private::zfp_cd_values_rate(
                            rate, elem_ty, chunk_dims,
                        )
                        .map_err(FormatError::from)?,
                    }
                }
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
                        client_data: build_cd_values(
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
                    client_data: vec![level],
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
    /// dimensions, or more than one unlimited dimension. Without these checks
    /// [`split_into_chunks`] panics on such a geometry, since it indexes
    /// `chunk_dims` by the shape's rank and divides by each chunk dimension, and
    /// a writer given more than one unlimited dimension writes a dataset the C
    /// library fails to open.
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
        // The C library indexes a dataspace with two unlimited
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

/// Split raw data into chunk-sized pieces based on shape and chunk dimensions.
///
/// Chunks are whole even where the dataset's edge falls inside one; the slots
/// past that edge take `fill`. That is not cosmetic padding: an allocated chunk
/// is expected to hold the dataset's fill value wherever nothing has been
/// written, so those slots are what a reader returns once the dataset is
/// extended into them. Passing [`FillPattern::ZERO`] where the dataset has no
/// fill value keeps this on a plain zeroed allocation (issue #296).
///
/// `fill` is a parameter rather than a field on [`ChunkContext`] on purpose:
/// every caller has to name it, so a new write path cannot inherit zeros by
/// omission.
pub fn split_into_chunks(
    raw_data: &[u8],
    shape: &[u64],
    chunk_dims: &[u64],
    element_size: NonZeroUsize,
    fill: FillPattern<'_>,
) -> Result<Vec<Vec<u8>>, FormatError> {
    let rank = shape.len();
    if rank == 0 {
        return Ok(vec![raw_data.to_vec()]);
    }

    // Compute number of chunks per dimension
    let mut num_chunks_per_dim = Vec::with_capacity(rank);
    for d in 0..rank {
        num_chunks_per_dim.push(shape[d].div_ceil(chunk_dims[d]));
    }
    let total_chunks: u64 = num_chunks_per_dim.iter().product();

    // Dataset strides (row-major)
    let mut ds_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "dataset dimension derived from the in-memory write request; bounded by addressable memory"
        )]
        let dim = shape[i + 1] as usize;
        ds_strides[i] = ds_strides[i + 1] * dim;
    }

    // Chunk strides
    let mut chunk_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "chunk dimension derived from the in-memory write request; bounded by addressable memory"
        )]
        let dim = chunk_dims[i + 1] as usize;
        chunk_strides[i] = chunk_strides[i + 1] * dim;
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "chunk/dataset dimensions derived from the in-memory write request; bounded by addressable memory"
    )]
    let (chunk_dims_us, shape_us): (Vec<usize>, Vec<usize>) = (
        chunk_dims.iter().map(|&d| d as usize).collect(),
        shape.iter().map(|&d| d as usize).collect(),
    );
    let chunk_total_elements: usize = chunk_dims_us.iter().product();

    #[expect(
        clippy::cast_possible_truncation,
        reason = "total_chunks derived from the in-memory write request; bounded by addressable memory"
    )]
    let mut buffers = Vec::with_capacity(total_chunks as usize);

    // Innermost dimension is contiguous in both the dataset (`raw_data`) and the
    // chunk buffer, so each in-bounds row is gathered with a single
    // `copy_from_slice`. Only the outer `rank - 1` dims are walked (odometer),
    // matching the read-side `copy_chunk_to_output` kernel.
    let inner = rank - 1;
    let mut coord = vec![0usize; inner];
    // Both refilled per chunk rather than allocated per chunk. Together with the
    // dataset-space offsets this loop used to return and no caller ever read,
    // that was three allocator round trips for every chunk of every dataset this
    // crate writes (issue #228). The order the offsets encoded is still a
    // correctness property -- the emitter writes chunks in it -- and the split
    // tests pin it by asserting each chunk's contents, which says the same thing.
    let mut offsets_us = vec![0usize; rank];
    let mut offsets = vec![0u64; rank];

    for linear_idx in 0..total_chunks {
        // Convert linear index to chunk grid coordinates and the chunk's
        // dataset-space offset, straight into this chunk's slice of `coords`.
        let mut remaining = linear_idx;
        for d in (0..rank).rev() {
            offsets[d] = (remaining % num_chunks_per_dim[d]) * chunk_dims[d];
            remaining /= num_chunks_per_dim[d];
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "chunk offset derived from the in-memory write request; bounded by addressable memory"
        )]
        for (slot, &o) in offsets_us.iter_mut().zip(offsets.iter()) {
            *slot = o as usize;
        }

        // A chunk wholly inside the dataset has no slot the data will not reach,
        // so it needs no fill: the row copies below overwrite every byte of it.
        // For rank > 1 an "overhang" is not only a trailing run — a partial
        // chunk has gaps between its in-bounds rows — which is why this asks
        // whether the whole chunk is in bounds rather than trying to name the
        // uncovered region.
        let whole_chunk_in_bounds =
            (0..rank).all(|d| offsets_us[d] + chunk_dims_us[d] <= shape_us[d]);

        let mut chunk_bytes = vec![0u8; chunk_total_elements * element_size.get()];
        if !whole_chunk_in_bounds {
            // Fill first, data over it: the rows copied below overwrite exactly
            // the in-bounds region, leaving every uncovered slot holding the
            // fill value.
            fill.apply(&mut chunk_bytes)?;
        }

        // In-bounds run length along the contiguous innermost dimension.
        let inner_row_len =
            chunk_dims_us[inner].min(shape_us[inner].saturating_sub(offsets_us[inner]));
        if inner_row_len > 0 {
            let row_bytes = inner_row_len * element_size.get();
            let inner_src = offsets_us[inner] * ds_strides[inner];
            let outer_total: usize = chunk_dims_us[..inner].iter().product();
            for c in coord.iter_mut() {
                *c = 0;
            }
            for _ in 0..outer_total {
                let mut dst_base = 0usize;
                let mut src_base = inner_src;
                let mut in_bounds = true;
                for d in 0..inner {
                    dst_base += coord[d] * chunk_strides[d];
                    let global = offsets_us[d] + coord[d];
                    if global >= shape_us[d] {
                        in_bounds = false;
                        break;
                    }
                    src_base += global * ds_strides[d];
                }

                if in_bounds {
                    let src = src_base * element_size.get();
                    let dst = dst_base * element_size.get();
                    let mut avail = row_bytes.min(raw_data.len().saturating_sub(src));
                    avail -= avail % element_size;
                    if avail > 0 {
                        chunk_bytes[dst..dst + avail].copy_from_slice(&raw_data[src..src + avail]);
                    }
                }

                for d in (0..inner).rev() {
                    coord[d] += 1;
                    if coord[d] < chunk_dims_us[d] {
                        break;
                    }
                    coord[d] = 0;
                }
            }
        }

        buffers.push(chunk_bytes);
    }

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
/// `ensure_chunk_bytes_representable` caps a chunk at 4 GiB before any of them
/// runs, so the product saturates here only so that a malformed geometry cannot
/// wrap into a small, plausible width.
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
    /// This is the reference C library's convention for a chunked dataset with
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
    /// by this kind, with `index_address` as the address of a Fixed Array or an
    /// Extensible Array.
    ///
    /// A Single Chunk layout stores the address of the first of `chunks`, and an
    /// unallocated layout a Fixed Array index at the undefined address.
    fn layout_message(
        self,
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

/// A chunked dataset's chunks already split and compressed — the expensive,
/// **address-independent** half of building a chunked layout. The compressed
/// bytes, the chunk-index choice, and the pipeline message do not depend on
/// where the data lands in the file. Only the addresses embedded in the chunk
/// index do. The writer sizes a dataset's object header in one pass and
/// emits its data in a later pass (it needs every prior object's size to know
/// this object's address), so it computes this set once and feeds it to
/// [`assemble_chunked_at`] twice — sizing at a dummy address, then emitting at
/// the real one — instead of recompressing the whole dataset each pass.
pub(crate) struct CompressedChunkSet {
    /// Per-chunk compressed bytes, in dense row-major grid order.
    compressed: Vec<Vec<u8>>,
    chunk_dims_u32: Vec<u32>,
    element_size: NonZeroUsize,
    has_filters: bool,
    /// Which index this set gets, decided once from the maximum chunk grid.
    kind: ChunkIndexKind,
    /// The element slot each chunk of `compressed` occupies, in the same order.
    /// Identical to `0..compressed.len()` unless the maximum shape is wider than
    /// the shape in some dimension past the first.
    slot_of_chunk: Vec<u64>,
    /// How many slots the index spans; see [`IndexSlots`].
    index_slots: u64,
    pipeline_message: Option<Vec<u8>>,
}

impl CompressedChunkSet {
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

/// Split `raw_data` into chunks and compress each one, producing the
/// address-independent [`CompressedChunkSet`]. This performs the dataset's only
/// pass of the filter pipeline (shuffle/deflate/ZFP/…); [`assemble_chunked_at`]
/// then lays the result out at a concrete address without recompressing.
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
    // The same pattern the unwritten slots are padded with: a filter that
    // records the fill value has to record the one this write pads with.
    let pipeline = options.build_pipeline(&ctx, fill)?;

    // Decided from the geometry alone, and decided first: it is the step that
    // refuses a maximum shape needing an index this writer will not emit, and
    // refusing after splitting and compressing the whole dataset would do all
    // that work to reach the same answer.
    let (kind, slot_of_chunk, index_slots) = plan_index_slots(
        shape,
        chunk_dims,
        maxshape,
        full_chunk_bytes(chunk_dims.iter().copied(), element_size),
        pipeline.is_some(),
        allocation,
    )?;

    // An unallocated dataset is not split. The splitter emits a chunk for every
    // slot the shape implies and this dataset supplies no bytes for them, so it
    // would write out the whole grid — and write it wrong, since a chunk that
    // lies inside the shape is zero-padded where the data runs out and only one
    // overhanging the edge takes the fill pattern. `plan_index_slots` already
    // answered for the same `allocation`, so the assertion below still pairs the
    // two.
    let chunks = match allocation {
        StorageAllocation::Allocated => {
            split_into_chunks(raw_data, shape, chunk_dims, element_size, fill)?
        }
        StorageAllocation::Unallocated => Vec::new(),
    };
    let num_chunks = chunks.len();
    let has_filters = pipeline.is_some();
    debug_assert_eq!(slot_of_chunk.len(), num_chunks);

    let mut compressed = Vec::with_capacity(num_chunks);
    // One encoder for every chunk of the dataset. Building one per chunk is the
    // dominant cost of a filtered write -- ~300 KiB of hash tables apiece, 615
    // MiB over an 8 MiB dataset (issue #228).
    let mut scratch = crate::filters::FilterScratch::new();
    for chunk_bytes in chunks {
        let c = if let Some(ref pl) = pipeline {
            compress_chunk_with(&mut scratch, &chunk_bytes, pl, ctx)?
        } else {
            // No pipeline: the split already produced an owned chunk buffer;
            // move it into the set instead of cloning.
            chunk_bytes
        };
        compressed.push(c);
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "chunk dimensions written into the on-disk u32 dimension fields selected for this file"
    )]
    let chunk_dims_u32: Vec<u32> = chunk_dims.iter().map(|&d| d as u32).collect();

    Ok(CompressedChunkSet {
        compressed,
        chunk_dims_u32,
        element_size,
        has_filters,
        kind,
        slot_of_chunk,
        index_slots,
        pipeline_message: pipeline
            .as_ref()
            .map(FilterPipeline::serialize)
            .transpose()
            .map_err(crate::filter_pipeline::map_filter_pipeline_error)?,
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

/// Where each of a chunk set's chunks lands when the set is laid out at
/// `data_address` — they are stored back to back from there — and the address the
/// chunk index follows them at.
fn plan_chunk_slots(
    set: &CompressedChunkSet,
    data_address: StoredAddress,
) -> (Vec<ChunkRecord>, StoredAddress) {
    let mut cursor = data_address;
    let mut written_chunks = Vec::with_capacity(set.compressed.len());
    for chunk in &set.compressed {
        written_chunks.push(ChunkRecord {
            address: cursor,
            stored_size: chunk.len() as u64,
            filter_mask: 0,
        });
        cursor = cursor.offset(chunk.len() as u64);
    }
    (written_chunks, cursor)
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
    let index = match set.kind {
        // Nothing stored, so nothing to index; the layout message below carries
        // the undefined address in place of one.
        ChunkIndexKind::Unallocated => Vec::new(),
        ChunkIndexKind::ExtensibleArray => hdf5_pure_format::__private::build_extensible_array_at(
            &set.index_slots(written_chunks)?,
            set.full_chunk_bytes(),
            INDEX_OFFSET_SIZE,
            INDEX_LENGTH_SIZE,
            set.has_filters,
            index_address,
        )?,
        ChunkIndexKind::SingleChunk => Vec::new(),
        ChunkIndexKind::FixedArray => hdf5_pure_format::__private::build_fixed_array_at(
            &set.index_slots(written_chunks)?,
            set.full_chunk_bytes(),
            INDEX_OFFSET_SIZE,
            INDEX_LENGTH_SIZE,
            set.has_filters,
            index_address,
        )?,
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
        &set.chunk_dims_u32,
        set.element_size,
        written_chunks,
        index_address,
        set.has_filters,
    )
}

/// The exact byte length [`assemble_chunked_at`] produces for `set` — the same
/// at every placement address, since the layout depends on the chunk sizes and the
/// index shape alone.
///
/// Sizing without assembling is what lets a caller pick the dataset's address
/// *first*: the in-place editor asks its free-space list for a region this long
/// and, if it gets one, assembles the set straight into it rather than growing
/// the file (issue #261).
pub(crate) fn chunked_data_len(set: &CompressedChunkSet) -> Result<u64, FormatError> {
    let (written_chunks, index_address) = plan_chunk_slots(set, StoredAddress::new(0));
    let index_len = match set.kind.array_kind() {
        Some(array) => chunk_index_len(
            array,
            &set.index_slots(&written_chunks)?,
            set.full_chunk_bytes(),
            INDEX_OFFSET_SIZE,
            INDEX_LENGTH_SIZE,
            set.has_filters,
        ),
        None => 0,
    };
    Ok(index_address.get() + index_len)
}

/// The chunk-index bytes and data-layout message for `set` at `data_address`,
/// with the total size of its chunk payload.
///
/// Everything [`assemble_chunked_at`] produces except the data region itself, so
/// that [`measure_chunked_at`] can answer "how long, and what does the layout
/// message say" without building an entire copy of the dataset.
fn plan_chunked_at(
    set: &CompressedChunkSet,
    data_address: StoredAddress,
) -> Result<(usize, Vec<u8>, Vec<u8>), FormatError> {
    let (written_chunks, index_address) = plan_chunk_slots(set, data_address);
    let (index, layout_message) = chunk_index_bytes(set, &written_chunks, index_address)?;
    let chunk_bytes_total: usize = set.compressed.iter().map(Vec::len).sum();
    Ok((chunk_bytes_total, index, layout_message))
}

/// The byte length and data-layout message [`assemble_chunked_at`] would produce
/// at `data_address`, without producing the data region.
///
/// The file writer sizes every object header before it emits a byte, and for a
/// chunked dataset that needs the layout message and the length of the region —
/// not the region. Calling `assemble_chunked_at` for it meant building a second
/// copy of every chunk in the dataset and dropping it: 8 MiB of allocation to
/// learn one integer, on every chunked write (issue #228).
///
/// Shares [`plan_chunked_at`] with the real assembly, so the length reported here
/// and the length produced there cannot drift.
pub(crate) fn measure_chunked_at(
    set: &CompressedChunkSet,
    data_address: StoredAddress,
) -> Result<ChunkedMeasure, FormatError> {
    let (written_chunks, index_address) = plan_chunk_slots(set, data_address);
    // Derived from the plan already in hand rather than by building the index —
    // and from *this* plan rather than by calling `chunked_data_len`, which
    // would lay the chunks out a second time to reach the same answer.
    let index_len = match set.kind.array_kind() {
        Some(array) => chunk_index_len(
            array,
            &set.index_slots(&written_chunks)?,
            set.full_chunk_bytes(),
            INDEX_OFFSET_SIZE,
            INDEX_LENGTH_SIZE,
            set.has_filters,
        ),
        None => 0,
    };
    let data_len = (index_address.get() - data_address.get()) + index_len;
    // Not building the index also stops it from *refusing*, and the only way it
    // can is a length that does not fit this platform's `usize`. Every such
    // check inside the build is on a part of this region, so the whole region
    // fitting means all of them do: the refusal survives the build's removal,
    // and stays where it was — before the writer has emitted a byte.
    data_len.to_usize()?;
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

/// Lays an already-[`compress`ed](compress_chunks) chunk set out at `data_address`,
/// producing the on-disk data region (chunk bytes followed by the chunk index)
/// and the v4 data-layout message. Cheap: this only concatenates and builds the
/// index, so it can be run more than once (different addresses) without
/// repeating the dataset's compression.
pub(crate) fn assemble_chunked_at(
    set: &CompressedChunkSet,
    data_address: StoredAddress,
) -> Result<ChunkedDataResult, FormatError> {
    let (chunk_bytes_total, index, layout_message) = plan_chunked_at(set, data_address)?;

    // One exact allocation for chunks plus index: the buffer is filled to its
    // capacity, never doubled and copied.
    let mut data_buf = Vec::with_capacity(chunk_bytes_total + index.len());
    for chunk in &set.compressed {
        data_buf.extend_from_slice(chunk);
    }
    data_buf.extend_from_slice(&index);

    Ok(ChunkedDataResult {
        data_bytes: data_buf,
        layout_message,
        pipeline_message: set.pipeline_message.clone(),
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

/// A minimal byte sink so the verbatim chunk emitter works against both an
/// in-memory `Vec<u8>` (the buffered / `no_std` path) and a streaming
/// `std::io::Write` (the out-of-core path), without pulling `std::io` into
/// `no_std` builds.
pub(crate) trait ByteSink {
    /// Append `bytes` to the output.
    fn put(&mut self, bytes: &[u8]) -> Result<(), FormatError>;
    /// Append `n` zero bytes.
    fn put_zeros(&mut self, n: usize) -> Result<(), FormatError>;
    /// Total bytes written so far (used to assert layout addresses on a
    /// non-seekable sink).
    fn position(&self) -> u64;
    /// Hint that `additional` more bytes are about to be written. Lets a buffered
    /// (`Vec`) sink preallocate the whole file in one shot, as the writer did
    /// before streaming. A no-op for sinks that do not benefit (e.g. a streaming
    /// `Write`).
    fn reserve(&mut self, _additional: usize) {}
}

impl ByteSink for Vec<u8> {
    fn put(&mut self, bytes: &[u8]) -> Result<(), FormatError> {
        self.extend_from_slice(bytes);
        Ok(())
    }
    fn put_zeros(&mut self, n: usize) -> Result<(), FormatError> {
        self.resize(self.len() + n, 0u8);
        Ok(())
    }
    fn position(&self) -> u64 {
        self.len() as u64
    }
    fn reserve(&mut self, additional: usize) {
        Vec::reserve(self, additional);
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

/// Computes the [`VerbatimLayout`] (data-region plan plus the v4 layout and
/// verbatim pipeline messages) for a dense, grid-ordered set of chunks, from
/// their sizes and filter masks alone — no chunk bytes. The byte layout is
/// identical whether the chunks are later buffered or streamed.
pub(crate) fn plan_chunked_data_verbatim(
    meta: &[ChunkMeta],
    shape: &[u64],
    chunk_dims: &[u64],
    element_size: NonZeroUsize,
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

/// Stream a planned verbatim dataset's data region to `sink`, pulling each
/// chunk's bytes from `provider` one at a time. The emitted bytes are identical
/// to the concatenation [`plan_chunked_data_verbatim`] describes, so a streamed
/// file and a buffered file are byte-for-byte equal.
///
/// The chunk index is built here rather than by the planner, so a failure to
/// build one now arrives mid-stream where it used to arrive at plan time. For a
/// caller that buffers (the in-place editor fills a `Vec` inside its placement
/// closure) that is invisible; for `FileBuilder::finish_to`, which writes
/// straight through, it means the chunk bytes are already on the sink. The only
/// such failure is a 32-bit `usize` overflow inside the Extensible Array
/// builder, which needs more chunks than that address space can hold.
pub(crate) fn emit_chunked_data_verbatim<S: ByteSink>(
    sink: &mut S,
    plan: &VerbatimPlan,
    provider: &dyn ChunkProvider,
) -> Result<(), FormatError> {
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
            ));
        }
        sink.put(&chunk)?;
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
            )));
        }
        sink.put(&bytes)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk_cache::ChunkCache;

    use crate::chunked_read::read_chunked_data_cached;
    use crate::convert::nz;
    use crate::data_layout::ChunkIndexLayout;
    use crate::data_layout::DataLayout;
    use crate::dataspace::{Dataspace, DataspaceType};
    use crate::datatype::layout::FloatingPointLayout;
    use crate::datatype::{Datatype, byte_order};
    use crate::fill_value::FillPattern;
    use crate::read_spec::RawReadSpec;

    fn make_f64_type() -> Datatype {
        Datatype::FloatingPoint {
            size: 8,
            byte_order: byte_order::DatatypeByteOrder::LittleEndian,
            layout: FloatingPointLayout {
                bit_offset: 0,
                bit_precision: 64,
                exponent_location: 52,
                exponent_size: 11,
                mantissa_location: 0,
                mantissa_size: 52,
                exponent_bias: 1023,
            },
        }
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
        let elem = nz(4);
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
        for pattern in [FillPattern::ZERO, FillPattern::new(Some(&[7u8; 4]), elem)] {
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
                let expected = full_chunk_bytes(chunk_dims.iter().copied(), nz(elem));
                let chunks =
                    split_into_chunks(&raw, shape, chunk_dims, nz(elem), FillPattern::ZERO)
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
            FillPattern::new(Some(&fill), elem),
            StorageAllocation::Unallocated,
        )
        .expect("an unallocated set");
        assert_eq!(set.compressed.len(), 0, "no chunk was encoded");
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
            FillPattern::new(Some(&fill), elem),
            StorageAllocation::Allocated,
        )
        .expect("a dense set");
        assert_eq!(dense.compressed.len(), 10);
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

        let output = read_chunked_data_cached(
            &file_data,
            RawReadSpec::parse(
                &layout,
                &dataspace,
                &datatype,
                pipeline.as_ref(),
                FillPattern::ZERO,
            )
            .unwrap(),
            8,
            8,
            &ChunkCache::new(),
        )
        .unwrap();

        bytes_to_f64(&output)
    }

    #[test]
    fn split_1d_single_chunk() {
        let data = f64_to_bytes(&[1.0, 2.0, 3.0]);
        let result = split_into_chunks(&data, &[3], &[3], nz(8), FillPattern::ZERO).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(bytes_to_f64(&result[0]), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn split_1d_multiple_chunks() {
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let data = f64_to_bytes(&values);
        let result = split_into_chunks(&data, &[10], &[4], nz(8), FillPattern::ZERO).unwrap();
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
        let result = split_into_chunks(&data, &[4, 4], &[2, 2], nz(8), FillPattern::ZERO).unwrap();
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
        let options = chunked(&[100], &[FilterKind::Deflate(6)]);
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
        let options = chunked(&[20], &[FilterKind::Deflate(6)]);
        let result = roundtrip_chunked(&values, &[100], &[20], &options);
        assert_eq!(result, values);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn roundtrip_1d_shuffle_deflate() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let options = chunked(&[50], &[FilterKind::Shuffle, FilterKind::Deflate(6)]);
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
            nz(8),
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
                nz(8),
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
            nz(8),
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
            nz(1),
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

    #[test]
    fn chunk_options_auto_dims() {
        let options = filtered(&[FilterKind::Deflate(6)]);
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
                FillPattern::new(Some(&fill), elem)
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
                FillPattern::new(Some(&fill), elem)
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
        let plain = filtered(&[FilterKind::Deflate(6)]);
        assert!(plain.build_pipeline(&ctx, FillPattern::UNKNOWN).is_ok());
    }

    #[test]
    fn chunk_options_pipeline_deflate() {
        let options = filtered(&[FilterKind::Deflate(6)]);
        let pl = options
            .build_pipeline(&ChunkContext::basic(&[], 8), FillPattern::ZERO)
            .unwrap()
            .unwrap();
        assert_eq!(pl.filters.len(), 1);
        assert_eq!(pl.filters[0].filter_id, FILTER_DEFLATE);
    }

    #[test]
    fn chunk_options_pipeline_shuffle_deflate_fletcher32() {
        let options = filtered(&[
            FilterKind::Shuffle,
            FilterKind::Deflate(6),
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

    /// `set_filter` places a filter at this crate's canonical rank whatever
    /// order the caller names them in, and replaces rather than repeats.
    ///
    /// This is what the `DatasetBuilder` switches promise: they name a filter to
    /// apply, not a position to apply it at, so `with_deflate(6).with_shuffle()`
    /// has to write the same bytes as `with_shuffle().with_deflate(6)`. The
    /// pipeline being an ordered list now (issue #333) is exactly what would let
    /// that drift.
    #[test]
    fn setting_filters_places_them_in_canonical_order_whatever_order_they_arrive_in() {
        let ids = |o: &ChunkOptions| -> Vec<u16> {
            o.filters.iter().map(|f| f.kind.filter_id()).collect()
        };
        let canonical = [FILTER_SHUFFLE, FILTER_DEFLATE, FILTER_FLETCHER32];

        assert_eq!(
            ids(&filtered(&[
                FilterKind::Shuffle,
                FilterKind::Deflate(6),
                FilterKind::Fletcher32,
            ])),
            canonical
        );
        assert_eq!(
            ids(&filtered(&[
                FilterKind::Fletcher32,
                FilterKind::Deflate(6),
                FilterKind::Shuffle,
            ])),
            canonical
        );

        // Naming one twice sets its parameter, rather than applying it twice.
        let twice = filtered(&[
            FilterKind::Deflate(1),
            FilterKind::Shuffle,
            FilterKind::Deflate(9),
        ]);
        assert_eq!(ids(&twice), [FILTER_SHUFFLE, FILTER_DEFLATE]);
        assert_eq!(twice.filters[1].kind, FilterKind::Deflate(9));
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
            (FilterKind::Deflate(4), true),
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

    /// Every request naming two filters where one would displace the other is
    /// refused, and the error names both (#233).
    ///
    /// The refusal is the observable part: a dropped filter leaves nothing in
    /// the file to distinguish `with_shuffle().with_zfp(16.0)` from
    /// `with_zfp(16.0)`, so a caller who wrote the first and got the second has
    /// no way to find out. Asserting on the message rather than only on
    /// `is_err()` is what keeps a refusal from being reported as some unrelated
    /// failure that happens to also be an error.
    #[test]
    fn conflicting_filter_requests_are_refused() {
        let so = ScaleOffset::FloatDScale(2);
        let fill = FillAvailability::Defined;
        let cases: &[(&str, &str, ChunkOptions)] = &[
            (
                "lzf",
                "deflate",
                filtered(&[FilterKind::Lzf, FilterKind::Deflate(6)]),
            ),
            (
                "shuffle",
                "scale-offset",
                filtered(&[FilterKind::Shuffle, FilterKind::ScaleOffset(so, fill)]),
            ),
            #[cfg(feature = "zfp")]
            (
                "scale-offset",
                "ZFP",
                filtered(&[FilterKind::ScaleOffset(so, fill), FilterKind::Zfp(16.0)]),
            ),
            #[cfg(feature = "zfp")]
            (
                "shuffle",
                "ZFP",
                filtered(&[FilterKind::Shuffle, FilterKind::Zfp(16.0)]),
            ),
            #[cfg(feature = "zfp")]
            (
                "lzf",
                "ZFP",
                filtered(&[FilterKind::Lzf, FilterKind::Zfp(16.0)]),
            ),
            #[cfg(feature = "zfp")]
            (
                "deflate",
                "ZFP",
                filtered(&[FilterKind::Deflate(6), FilterKind::Zfp(16.0)]),
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
            element_type: zfp_f64_type(),
            scale_offset_type: crate::scaleoffset::scale_offset_type_from_datatype(&make_f64_type()),
        }
    }

    /// The type a ZFP request needs, when the feature is on.
    #[cfg(feature = "zfp")]
    fn zfp_f64_type() -> Option<crate::filters::ZfpElementTypeWhenEnabled> {
        crate::filters::zfp_element_type_from_datatype(&make_f64_type())
    }

    #[cfg(not(feature = "zfp"))]
    fn zfp_f64_type() -> Option<crate::filters::ZfpElementTypeWhenEnabled> {
        None
    }

    /// Chaining a primary transform with a filter it does *not* displace still
    /// builds, so the refusal above is a rule about conflicts rather than a
    /// blanket ban on combining filters.
    #[test]
    fn compatible_filter_requests_still_build() {
        let so = FilterKind::ScaleOffset(ScaleOffset::FloatDScale(2), FillAvailability::Defined);
        let cases: &[(ChunkOptions, &[u16])] = &[
            (
                filtered(&[FilterKind::Shuffle, FilterKind::Deflate(6)]),
                &[FILTER_SHUFFLE, FILTER_DEFLATE],
            ),
            (
                filtered(&[FilterKind::Shuffle, FilterKind::Lzf]),
                &[FILTER_SHUFFLE, FILTER_LZF],
            ),
            (
                filtered(&[so, FilterKind::Deflate(6)]),
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

        let output = read_chunked_data_cached(
            &file_data,
            RawReadSpec::plain(&layout, &dataspace, &datatype),
            8,
            8,
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
                        options.set_filter(FilterKind::Deflate(6));
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

    // ---- h5py round-trip tests for chunked writes ----

    // Runs `script` under python3, passing the HDF5 file path as `sys.argv[1]`
    // so the script can open it without interpolating the path into the source.
    // Interpolating a Windows path (with backslashes) into a Python string
    // literal breaks the parser (e.g. `\U` triggers a unicode-escape error).
    #[cfg(feature = "std")]
    fn h5py_run(path: &std::path::Path, script: &str) -> Option<String> {
        let o = std::process::Command::new("python3")
            .args(["-c", script, &path.to_string_lossy()])
            .output()
            .ok()?;
        if !o.status.success() {
            let err = String::from_utf8_lossy(&o.stderr);
            // Under QEMU user emulation, a missing binary yields exit status 127
            // with empty standard error.
            if o.status.code() == Some(127) && err.is_empty()
                || err.contains("No module named")
                || err.contains("not found")
            {
                return None;
            }
            panic!("h5py: {err}");
        }
        Some(String::from_utf8(o.stdout).unwrap().trim().to_string())
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_multiple_chunked_datasets() {
        use crate::file_writer::FileWriter;
        let mut fw = FileWriter::new();
        let data1: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let data2: Vec<f64> = (0..30).map(|i| (i * 10) as f64).collect();
        fw.create_dataset("a")
            .with_f64_data(&data1)
            .with_shape(&[50])
            .with_chunks(&[25]);
        fw.create_dataset("b")
            .with_f64_data(&data2)
            .with_shape(&[30])
            .with_chunks(&[10]);
        let bytes = fw.finish().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rustyhdf5_chunked_multi.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = "import sys,h5py,json; f=h5py.File(sys.argv[1],'r'); print(json.dumps({'a':f['a'][:].tolist(),'b':f['b'][:].tolist()}))";
        let Some(out) = h5py_run(&path, script) else {
            return;
        };
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        let va: Vec<f64> = serde_json::from_value(v["a"].clone()).unwrap();
        let vb: Vec<f64> = serde_json::from_value(v["b"].clone()).unwrap();
        assert_eq!(va, data1);
        assert_eq!(vb, data2);
    }

    #[cfg(feature = "std")]
    #[test]
    fn h5py_reads_chunked_with_attrs() {
        use crate::file_writer::{AttrValue, FileWriter};
        let mut fw = FileWriter::new();
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        fw.create_dataset("data")
            .with_f64_data(&data)
            .with_shape(&[50])
            .with_chunks(&[25])
            .set_attr("units", AttrValue::String("meters".to_string()));
        let bytes = fw.finish().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rustyhdf5_chunked_attrs.h5");
        std::fs::write(&path, &bytes).unwrap();
        let script = "import sys,h5py,json; f=h5py.File(sys.argv[1],'r'); d=f['data']; print(json.dumps({'values':d[:].tolist(),'units':d.attrs['units'].decode() if isinstance(d.attrs['units'],bytes) else str(d.attrs['units'])}))";
        let Some(out) = h5py_run(&path, script) else {
            return;
        };
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        let values: Vec<f64> = serde_json::from_value(v["values"].clone()).unwrap();
        assert_eq!(values, data);
        assert_eq!(v["units"], serde_json::json!("meters"));
    }
}
