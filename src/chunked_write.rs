//! Chunked dataset writing: chunk splitting, compression, index building.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

use crate::checksum::jenkins_lookup3;
/// The HDF5 "undefined address" sentinel (`HADDR_UNDEF`): all bits set, in
/// whatever width the field is. A chunked layout message carrying it declares
/// that the dataset has no storage allocated at all.
const HADDR_UNDEF: u64 = u64::MAX;
use core::num::NonZeroUsize;

use crate::chunk_grid::{ChunkGrid, GridOrder};
use crate::convert::{TryToUsize, nonzero_usize_from};
use crate::error::FormatError;
use crate::extensible_array::{DataBlockGeom, EaGeometry, ExtensibleArrayHeader, SuperBlockGeom};
use crate::fill_value::FillPattern;
#[cfg(feature = "zfp")]
use crate::filter_pipeline::FILTER_ZFP;
use crate::filter_pipeline::{
    FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_LZF, FILTER_SCALEOFFSET, FILTER_SHUFFLE,
    FilterDescription, FilterPipeline, H5Z_FLAG_OPTIONAL,
};
use crate::filters::{ChunkContext, compress_chunk_with};
use crate::scaleoffset::{FillAvailability, ScaleOffset, build_cd_values};

/// Log2 of the Fixed Array data-block page size (`2^10 = 1024` elements).
///
/// Single source of truth for the page exponent the writer emits: it is both
/// the `max_nelmts_bits` field stored in the Fixed Array header (FAHD) and the
/// `max_dblk_page_nelmts_bits` field in the v4 chunked layout message, which the
/// HDF5 spec requires to be equal. Above this many chunks the writer switches to
/// the paged data-block layout. The value mirrors the HDF5 C library's
/// `H5D_FARRAY_MAX_DBLK_PAGE_NELMTS_BITS`. The reader does not use this constant:
/// it honors whatever page size a file declares in its FAHD.
pub(crate) const FIXED_ARRAY_PAGE_BITS: u8 = 10;

/// The on-disk address and length widths every chunk index *this module* writes
/// uses. Named so the path that sizes an index and the path that emits it cannot
/// read different values.
///
/// Not the only such pair: the in-place Extensible Array rebuild in `edit`
/// reads `file_writer::OFFSET_SIZE` / `LENGTH_SIZE`, which carry the same
/// meaning and the same values. Both are what this crate's superblock declares,
/// and `edit` refuses a file whose superblock disagrees, so the two cannot drift
/// apart within one file.
const INDEX_OFFSET_SIZE: u8 = 8;
const INDEX_LENGTH_SIZE: u8 = 8;

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

    /// Where this filter sits in the pipeline this crate's own writer builds:
    /// a primary transform first, then shuffle, then a byte compressor, then
    /// the checksum last. Lower runs earlier, so it is applied to less
    /// processed bytes.
    fn canonical_rank(self) -> u8 {
        match self {
            #[cfg(feature = "zfp")]
            Self::Zfp(_) => 0,
            Self::ScaleOffset(..) => 1,
            Self::Shuffle => 2,
            Self::Lzf => 3,
            Self::Deflate(_) => 4,
            Self::Fletcher32 => 5,
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
    /// `with_deflate`, ...) call. Placing by
    /// [`canonical_rank`](FilterKind::canonical_rank) rather than by call order
    /// is what keeps `with_deflate(6).with_shuffle()` and
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
        let at = self
            .filters
            .iter()
            .position(|f| f.kind.canonical_rank() > kind.canonical_rank())
            .unwrap_or(self.filters.len());
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

    /// Whether the pipeline includes the filter with this id.
    fn has(&self, id: u16) -> bool {
        self.filters.iter().any(|f| f.kind.filter_id() == id)
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
        if self.has(FILTER_DEFLATE) {
            return Err("deflate compression requires the `deflate` crate feature");
        }
        Ok(())
    }

    #[cfg(feature = "zfp")]
    #[inline]
    fn zfp_enabled(&self) -> bool {
        self.has(FILTER_ZFP)
    }

    #[cfg(not(feature = "zfp"))]
    #[inline]
    fn zfp_enabled(&self) -> bool {
        false
    }

    /// Refuse a combination of filters where honoring one means discarding
    /// another.
    ///
    /// Two filters here are *primary transforms* that consume the raw elements
    /// and hand on something else: ZFP, and scale-offset. Each displaces
    /// whatever it sits on top of, so a request naming a displaced filter as
    /// well is a contradiction — the caller asked for something the file cannot
    /// end up containing.
    ///
    /// Every such contradiction is an error. Dropping the loser silently is the
    /// one option a caller cannot detect: nothing in the resulting file records
    /// that a filter was requested, so `with_shuffle().with_zfp(16.0)` produced
    /// an unshuffled dataset and no way to tell that from `with_zfp(16.0)`
    /// alone. Documented precedence is not a substitute, because a precedence
    /// rule still has to be read to be obeyed and there is nothing to read it
    /// against at the call site.
    ///
    /// Checked in one place, before any filter is built, so which contradiction
    /// gets reported does not depend on the order the pipeline happens to be
    /// assembled in, and so a filter added later inherits the rule rather than
    /// having to restate it.
    fn refuse_conflicting_filters(&self) -> Result<(), FormatError> {
        let clash = |a: &str, b: &str| {
            Err(FormatError::FilterError(format!(
                "{a} and {b} cannot be combined on one dataset"
            )))
        };

        if self.zfp_enabled() {
            if self.has(FILTER_SCALEOFFSET) {
                return clash("scale-offset", "ZFP");
            }
            if self.has(FILTER_SHUFFLE) {
                return clash("shuffle", "ZFP");
            }
            if self.has(FILTER_LZF) {
                return clash("lzf", "ZFP");
            }
            if self.has(FILTER_DEFLATE) {
                return clash("deflate", "ZFP");
            }
        }
        if self.has(FILTER_SCALEOFFSET) && self.has(FILTER_SHUFFLE) {
            return clash("shuffle", "scale-offset");
        }
        // Not a primary transform, but the same shape: LZF and deflate fill one
        // byte-compressor slot, and stacking two of them is never useful.
        if self.has(FILTER_LZF) && self.has(FILTER_DEFLATE) {
            return clash("lzf", "deflate");
        }
        Ok(())
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
                        client_data: crate::zfp::zfp_cd_values_rate(rate, elem_ty, chunk_dims)?,
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
                    let nelmts =
                        u32::try_from(chunk_dims.iter().product::<u64>()).map_err(|_| {
                            FormatError::FilterError(
                                "scale-offset: chunk has too many elements".into(),
                            )
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
                            fill_avail.with_value(fill)?,
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
                    client_data: crate::lzf::h5py_cd_values(element_size, chunk_dims).to_vec(),
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

    /// Validate the chunk geometry of a dataset that will use chunked storage,
    /// against its `shape` and optional `maxshape`. Returns a static reason on
    /// the first problem; callers map it to their own error type. Only
    /// meaningful when the dataset is actually chunked
    /// ([`is_chunked`](Self::is_chunked) or a `maxshape` is set).
    ///
    /// These checks turn what would otherwise be a panic deep in the chunk
    /// splitter ([`split_into_chunks`], which indexes `chunk_dims` by the shape's
    /// rank and divides by each chunk dimension) — or a silently corrupt,
    /// unreadable dataset — into an up-front, descriptive refusal. A
    /// zero-element shape (e.g. `[0]` for an empty extensible dataset) is allowed
    /// with explicit chunk dimensions: it is not scalar and produces zero chunks,
    /// which is well-formed.
    ///
    /// The chunk dimensions checked are the *resolved* ones — what
    /// [`resolve_chunk_dims`](Self::resolve_chunk_dims) will hand the splitter —
    /// not only the ones the caller named. Auto-chunking derives them from the
    /// shape, so a shape that is invalid to chunk by itself has to be caught
    /// here rather than left to divide by zero one layer down.
    pub fn validate_geometry(
        &self,
        shape: &[u64],
        maxshape: Option<&[u64]>,
    ) -> Result<(), &'static str> {
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
        // A maximum shape must match the rank and bound the current shape in
        // every dimension (an unlimited dimension, `u64::MAX`, bounds anything).
        if let Some(ms) = maxshape {
            if ms.len() != shape.len() {
                return Err("maxshape must have the same rank as the dataset shape");
            }
            if ms.iter().zip(shape).any(|(&m, &d)| m != u64::MAX && m < d) {
                return Err("maxshape must be at least the current shape in every dimension");
            }
            // The reference library indexes a dataspace with two unlimited
            // dimensions using a version-2 B-tree, which this crate does not
            // write; an Extensible Array cannot number such a dataspace at all,
            // since it has only one dimension it can grow along. Writing one
            // anyway produced a dataset the reference library refuses to open
            // ("already found unlimited dimension"), so this is a refusal rather
            // than a file only this crate can read (issue #299).
            if ms.iter().filter(|&&m| m == u64::MAX).count() > 1 {
                return Err(
                    "at most one dimension of a maxshape may be unlimited; the chunk index \
                     for more than one is a version-2 B-tree, which this crate cannot write",
                );
            }
        }
        Ok(())
    }
}

/// A chunk that has been written to the file buffer.
#[derive(Debug, Clone)]
pub struct WrittenChunk {
    /// Address within the file where chunk data starts.
    pub address: u64,
    /// Size of the (possibly compressed) chunk data in bytes.
    ///
    /// The chunk's *uncompressed* size is deliberately not a field here. It was
    /// one, and the only thing that ever read it was the chunk-index element
    /// width — which must come from the geometry ([`full_chunk_bytes`]), since a
    /// set of zero chunks has no chunk to read it from and still has to declare
    /// the width its first chunk will need.
    pub compressed_size: u64,
    /// Filter mask (0 = all filters applied).
    pub filter_mask: u32,
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

/// A chunk index's element slots: which chunk occupies each one, and how many
/// slots the index spans.
///
/// Not the same thing as the list of chunks, and that is the whole point. A
/// Fixed Array and an Extensible Array store their elements *positionally*, and
/// the position is taken over the dataset's maximum chunk grid
/// ([`crate::chunk_grid`]), so a dataset that can grow leaves gaps between its
/// chunks and a Fixed Array declares slots it has no chunk for.
///
/// Borrows the chunks and records only where they sit, and records nothing at all
/// when they sit at slots `0..n` — which is every dataset whose maximum shape is
/// no wider than its shape past the first dimension, and so nearly every dataset
/// written. A slot table proportional to the *maximum* shape would otherwise be
/// paid for by writes that have no gaps to describe.
pub(crate) struct IndexSlots<'a> {
    chunks: &'a [WrittenChunk],
    /// `(slot, index into `chunks`)`, ascending by slot. Empty means the chunks
    /// fill slots `0..chunks.len()` in order, so the slot *is* the index.
    scattered: Vec<(usize, usize)>,
    len: usize,
}

impl<'a> IndexSlots<'a> {
    /// The slots of an index spanning `len` slots, with `chunks[i]` occupying
    /// `slot_of[i]`.
    ///
    /// `len` comes from [`plan_index_slots`], which is where the span is
    /// decided and where a span with too many unused slots is refused; what is
    /// left to refuse here is a span this platform's `usize` cannot address,
    /// which is what lets every caller below treat it as a plain `usize`.
    pub(crate) fn new(
        chunks: &'a [WrittenChunk],
        slot_of: &[u64],
        len: u64,
    ) -> Result<Self, FormatError> {
        debug_assert_eq!(
            chunks.len(),
            slot_of.len(),
            "every chunk has exactly one index slot"
        );
        let len = len.to_usize()?;
        // The dense case is not an optimization of the general one, it is the
        // overwhelming majority of writes; recognizing it here is what keeps
        // them from allocating a table describing an order they are already in.
        if len == chunks.len() && slot_of.iter().enumerate().all(|(i, &s)| s == i as u64) {
            return Ok(Self {
                chunks,
                scattered: Vec::new(),
                len,
            });
        }
        let mut scattered = Vec::with_capacity(chunks.len());
        for (i, &slot) in slot_of.iter().enumerate() {
            scattered.push((slot.to_usize()?, i));
        }
        scattered.sort_unstable();
        Ok(Self {
            chunks,
            scattered,
            len,
        })
    }

    /// The slots of an index whose chunks fill it densely from zero.
    pub(crate) fn dense(chunks: &'a [WrittenChunk]) -> Self {
        Self {
            len: chunks.len(),
            chunks,
            scattered: Vec::new(),
        }
    }

    /// How many slots the index spans, occupied or not.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Whether any of the `count` slots from `start` holds a chunk.
    ///
    /// The question an Extensible Array asks of a block before allocating it:
    /// a block none of whose slots is occupied is not written at all, so a
    /// mostly-empty index costs what its chunks cost rather than what its span
    /// does. Answered over the sparse list, so it is a binary search rather than
    /// a walk of the span.
    pub(crate) fn any_occupied(&self, start: u64, count: u64) -> bool {
        // A slot index is a `usize`, so a span starting past that cannot hold
        // one. The spans an Extensible Array walks reach 2^33 elements, which
        // is past `usize` on a 32-bit target and reachable there.
        let Ok(start) = usize::try_from(start) else {
            return false;
        };
        let end = usize::try_from(count)
            .ok()
            .and_then(|c| start.checked_add(c))
            .unwrap_or(usize::MAX);
        if self.scattered.is_empty() {
            return start < self.chunks.len().min(end);
        }
        let from = self.scattered.partition_point(|&(s, _)| s < start);
        self.scattered.get(from).is_some_and(|&(s, _)| s < end)
    }

    /// The chunk at `slot`, or `None` for a slot no chunk occupies — either past
    /// the last one or in a gap the maximum shape left.
    pub(crate) fn at(&self, slot: usize) -> Option<&WrittenChunk> {
        if self.scattered.is_empty() {
            return self.chunks.get(slot);
        }
        self.scattered
            .binary_search_by_key(&slot, |&(s, _)| s)
            .ok()
            .map(|i| &self.chunks[self.scattered[i].1])
    }
}

/// Which element slots of an Extensible Array hold a chunk.
///
/// Two walks decide the array's shape: [`ea_compute_stats`], which says which
/// blocks it allocates, and [`build_extensible_array_at`], which emits exactly
/// those. The length one predicts and the length the other writes are asserted
/// equal, so the two have to ask the same question — passing it as a value is
/// what keeps that structural rather than a pair of predicates someone has to
/// keep in step.
#[derive(Clone, Copy)]
pub(crate) enum SlotOccupancy<'a> {
    /// Slots `0..len` are occupied and nothing past them: an array whose chunks
    /// fill their slots in order, which is every rank-1 array and every array
    /// with no maximum shape wider than its shape.
    Dense(u64),
    /// The occupied slots of a possibly sparse index.
    Sparse(&'a IndexSlots<'a>),
    /// The occupied slots as bare slot numbers, ascending — what the planner
    /// holds before any chunk has an address, so that it can size the index it
    /// is about to accept or refuse without laying the chunks out first.
    Slots(&'a [u64]),
}

impl SlotOccupancy<'_> {
    /// Whether a block spanning `count` slots from `start` holds any chunk, and
    /// so needs to exist.
    ///
    /// The reference library allocates a data block when a chunk lands in it, so
    /// a dataset whose maximum shape is far wider than its shape gets an index
    /// proportional to its chunks rather than to the gap between them. Writing
    /// every block in the span instead cost 21x the file libhdf5 writes for the
    /// same 256-chunk dataset, and put figures in the array's own header
    /// statistics that disagreed with it by a factor of 40 (issue #299).
    fn any_occupied(&self, start: u64, count: u64) -> bool {
        match self {
            Self::Dense(len) => start < *len,
            Self::Sparse(slots) => slots.any_occupied(start, count),
            Self::Slots(sorted) => {
                let end = start.saturating_add(count);
                let from = sorted.partition_point(|&s| s < start);
                sorted.get(from).is_some_and(|&s| s < end)
            }
        }
    }
}

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

/// Serialize a v4 single chunk layout message.
fn serialize_v4_single_chunk(
    chunk_dims: &[u32],
    chunk_address: u64,
    filtered_size: Option<u64>,
    filter_mask: Option<u32>,
    offset_size: u8,
    element_size: u32,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(4); // version
    buf.push(2); // class = chunked

    // flags: bit 0 = unknown meaning in some files, bit 1 = filters for single chunk
    let flags: u8 = if filtered_size.is_some() { 0x02 } else { 0x00 };
    buf.push(flags);

    // dimensionality = rank + 1 (chunk dims + element size dim)
    #[expect(
        clippy::cast_possible_truncation,
        reason = "rank written into the 1-byte dimensionality field selected for this file"
    )]
    let ndims = chunk_dims.len() as u8 + 1;
    buf.push(ndims);

    // dim_size_encoded_length: how many bytes per dimension
    // We need to figure out the minimum encoding width
    let max_dim = chunk_dims
        .iter()
        .map(|&d| d as u64)
        .chain(core::iter::once(element_size as u64))
        .max()
        .unwrap_or(1);
    let dim_encoded_len: u8 = if max_dim <= 0xFF {
        1
    } else if max_dim <= 0xFFFF {
        2
    } else {
        4
    };
    buf.push(dim_encoded_len);

    // dimension sizes (chunk dims + element size)
    for &d in chunk_dims {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "dimension written into the on-disk encoding width selected for this file"
        )]
        match dim_encoded_len {
            1 => buf.push(d as u8),
            2 => buf.extend_from_slice(&(d as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&d.to_le_bytes()),
            _ => {}
        }
    }
    // Element size dimension
    #[expect(
        clippy::cast_possible_truncation,
        reason = "element size written into the on-disk encoding width selected for this file"
    )]
    match dim_encoded_len {
        1 => buf.push(element_size as u8),
        2 => buf.extend_from_slice(&(element_size as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&element_size.to_le_bytes()),
        _ => {}
    }

    // chunk index type = 1 (single chunk)
    buf.push(1);

    // Index-specific fields
    if let (Some(fs), Some(fm)) = (filtered_size, filter_mask) {
        // filtered_size (length_size bytes)
        buf.extend_from_slice(&fs.to_le_bytes()); // 8 bytes for length_size=8
        buf.extend_from_slice(&fm.to_le_bytes()); // 4 bytes
    }

    // chunk address
    #[expect(
        clippy::cast_possible_truncation,
        reason = "chunk address written into the on-disk offset width selected for this file"
    )]
    match offset_size {
        4 => buf.extend_from_slice(&(chunk_address as u32).to_le_bytes()),
        8 => buf.extend_from_slice(&chunk_address.to_le_bytes()),
        _ => {}
    }

    buf
}

/// Serialize a v4 Fixed Array layout message.
fn serialize_v4_fixed_array(
    chunk_dims: &[u32],
    fixed_array_address: u64,
    offset_size: u8,
    element_size: u32,
    max_bits: u8,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(4); // version
    buf.push(2); // class = chunked

    let flags: u8 = 0x00;
    buf.push(flags);

    #[expect(
        clippy::cast_possible_truncation,
        reason = "rank written into the 1-byte dimensionality field selected for this file"
    )]
    let ndims = chunk_dims.len() as u8 + 1;
    buf.push(ndims);

    let max_dim = chunk_dims
        .iter()
        .map(|&d| d as u64)
        .chain(core::iter::once(element_size as u64))
        .max()
        .unwrap_or(1);
    let dim_encoded_len: u8 = if max_dim <= 0xFF {
        1
    } else if max_dim <= 0xFFFF {
        2
    } else {
        4
    };
    buf.push(dim_encoded_len);

    for &d in chunk_dims {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "dimension written into the on-disk encoding width selected for this file"
        )]
        match dim_encoded_len {
            1 => buf.push(d as u8),
            2 => buf.extend_from_slice(&(d as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&d.to_le_bytes()),
            _ => {}
        }
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "element size written into the on-disk encoding width selected for this file"
    )]
    match dim_encoded_len {
        1 => buf.push(element_size as u8),
        2 => buf.extend_from_slice(&(element_size as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&element_size.to_le_bytes()),
        _ => {}
    }

    // chunk index type = 3 (Fixed Array)
    buf.push(3);

    // max_dblk_page_nelmts_bits — must match FAHD max_nelmts_bits
    buf.push(max_bits);

    // Fixed Array header address
    #[expect(
        clippy::cast_possible_truncation,
        reason = "fixed array header address written into the on-disk offset width selected for this file"
    )]
    match offset_size {
        4 => buf.extend_from_slice(&(fixed_array_address as u32).to_le_bytes()),
        8 => buf.extend_from_slice(&fixed_array_address.to_le_bytes()),
        _ => {}
    }

    buf
}

/// How a chunk index encodes one element record.
///
/// The Fixed Array and the Extensible Array agree on this to the byte — the
/// reference C library derives both from the same
/// `H5D_*ARRAY_FILT_COMPUTE_CHUNK_SIZE_LEN` rule — so they derive it here once
/// rather than each keeping its own copy of the arithmetic.
#[derive(Debug, Clone, Copy)]
struct ChunkElementEncoding {
    /// Width of the compressed-size field inside a filtered element record,
    /// sized to the largest raw chunk. Zero when the dataset is unfiltered.
    chunk_size_bytes: usize,
    /// Byte size of one element record: an address, plus the compressed size and
    /// filter mask when the dataset is filtered.
    elem_size: usize,
    /// The index's client ID: filtered (1) or not (0). The Fixed and Extensible
    /// Arrays reach that numbering through two different class tables in the
    /// reference C library (`H5FA_CLS_*` and `H5EA_CLS_*`) which happen to agree,
    /// so one field serves both only for as long as they do.
    client_id: u8,
}

/// The unfiltered byte size of one whole chunk: the product of the chunk
/// dimensions and the element size.
///
/// One derivation, shared by everything that has to size a chunk-index element
/// ([`chunk_element_encoding`]) — the buffered builder, the verbatim planner,
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

/// Derive the element encoding for a chunk set.
///
/// `chunk_size_len = 1 + ((H5VM_log2_gen(chunk.size) + 8) / 8)`, where
/// `chunk.size` is `chunk_bytes`: the *unfiltered* chunk size, the product of
/// the chunk dimensions and the element size. The field is sized to a whole raw
/// chunk rather than to any compressed one.
///
/// It is taken from the geometry rather than from the chunks that happen to have
/// been written, because the two part company exactly where it matters. Every
/// chunk [`split_into_chunks`] produces is padded to the full chunk size,
/// so for one chunk or a thousand the largest `raw_size` *is* `chunk_bytes` and
/// the two agree byte for byte. For **zero** chunks there is no `raw_size` to
/// take a maximum of, and the fallback this used to apply — treat the largest
/// chunk as 1 byte — declared a 2-byte compressed-size field for a dataset whose
/// chunks need up to 8. Nothing catches that later: the width is a header field
/// every reader honours, so an empty filtered dataset handed to the reference C
/// library came back with chunks encoded to a width our reader then decoded as
/// truncated deflate streams, and our own append refused a chunk that no longer
/// fit the width its own index had declared.
fn chunk_element_encoding(
    chunk_bytes: u64,
    offset_size: u8,
    has_filters: bool,
) -> ChunkElementEncoding {
    let os = offset_size as usize;
    let chunk_size_bytes: usize = if has_filters {
        let log2_val = if chunk_bytes <= 1 {
            0
        } else {
            63 - chunk_bytes.leading_zeros()
        };
        let len = 1 + ((log2_val + 8) / 8) as usize;
        len.min(8)
    } else {
        0
    };
    ChunkElementEncoding {
        chunk_size_bytes,
        elem_size: if has_filters {
            os + chunk_size_bytes + 4
        } else {
            os
        },
        client_id: u8::from(has_filters),
    }
}

/// Everything about a Fixed Array that does not depend on where it is placed:
/// the element encoding, the paging, the header size and the total length.
///
/// The same split as [`EaLayout`], and for the same reason: a caller that has to
/// reserve the array's span before its bytes exist takes the length from here
/// ([`fixed_array_len`]) rather than from a build it throws away.
struct FaLayout {
    encoding: ChunkElementEncoding,
    /// The page exponent the header declares, and `1 << page_bits`, the element
    /// count past which the data block is paged. Both are carried so the byte
    /// written into the header and the threshold the emitter pages at stay one
    /// value: the reader honours whatever exponent a file declares, so a writer
    /// that ever varied this must vary both together or every reader pages the
    /// block wrong.
    page_bits: u8,
    page_size: usize,
    fahd_size: usize,
    /// Header plus data block: the whole array.
    total_len: u64,
}

/// Lay out the Fixed Array that would hold `slots`, without building it.
fn fa_layout(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: u8,
    length_size: u8,
    has_filters: bool,
) -> FaLayout {
    // Both widths go into fixed-width header fields below, and the emitter
    // writes 8 bytes for anything that is not 4 — so a third width would make
    // this length disagree with the bytes. Every caller passes
    // `INDEX_OFFSET_SIZE` / `INDEX_LENGTH_SIZE`.
    debug_assert!(
        matches!(offset_size, 4 | 8) && matches!(length_size, 4 | 8),
        "a fixed array is written at a 4- or 8-byte address and length width"
    );
    let os = offset_size as usize;
    let num_elements = slots.len();
    let encoding = chunk_element_encoding(chunk_bytes, offset_size, has_filters);

    let fahd_size = 4 + 1 + 1 + 1 + 1 + length_size as usize + os + 4;

    // The data block is a prefix, then either every element inline followed by
    // one checksum, or a page-init bitmap and its checksum followed by whole
    // pages that each carry their own. Every element is written in exactly one
    // page, so the element bytes total the same either way.
    let fadb_prefix = 4 + 1 + 1 + os;
    let page_bits = FIXED_ARRAY_PAGE_BITS;
    let page_size = 1usize << page_bits;
    let elements = num_elements * encoding.elem_size;
    let fadb_size = if num_elements <= page_size {
        fadb_prefix + elements + 4
    } else {
        let npages = num_elements.div_ceil(page_size);
        fadb_prefix + npages.div_ceil(8) + 4 + elements + npages * 4
    };

    FaLayout {
        encoding,
        page_bits,
        page_size,
        fahd_size,
        total_len: (fahd_size + fadb_size) as u64,
    }
}

/// The byte length [`build_fixed_array_at`] would produce for `slots`, without
/// building it. See [`extensible_array_len`] for why this exists.
///
/// `offset_size` and `length_size` must be 4 or 8, which is what the emitter
/// writes; every caller passes [`INDEX_OFFSET_SIZE`] / [`INDEX_LENGTH_SIZE`].
pub(crate) fn fixed_array_len(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: u8,
    length_size: u8,
    has_filters: bool,
) -> u64 {
    fa_layout(slots, chunk_bytes, offset_size, length_size, has_filters).total_len
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
/// ([`extensible_array_len`], [`fixed_array_len`]), so a caller reserving the
/// data region's span takes it from the same layout the emission works from.
pub(crate) fn chunk_index_len(
    kind: ChunkArrayKind,
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: u8,
    length_size: u8,
    has_filters: bool,
) -> u64 {
    match kind {
        ChunkArrayKind::ExtensibleArray => {
            extensible_array_len(slots, chunk_bytes, offset_size, length_size, has_filters)
        }
        ChunkArrayKind::FixedArray => {
            fixed_array_len(slots, chunk_bytes, offset_size, length_size, has_filters)
        }
    }
}

/// Build a complete Fixed Array at a known absolute address.
pub fn build_fixed_array_at(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: u8,
    length_size: u8,
    has_filters: bool,
    fa_address: u64,
) -> Vec<u8> {
    let num_elements = slots.len();

    let layout = fa_layout(slots, chunk_bytes, offset_size, length_size, has_filters);
    let ChunkElementEncoding {
        chunk_size_bytes,
        elem_size,
        client_id,
    } = layout.encoding;
    let fahd_total_size = layout.fahd_size;
    let fadb_address = fa_address + fahd_total_size as u64;

    // Build FAHD
    let mut fahd = Vec::with_capacity(fahd_total_size);
    fahd.extend_from_slice(b"FAHD");
    fahd.push(0); // version
    fahd.push(client_id);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "element record size written into the 1-byte FAHD field selected for this file"
    )]
    fahd.push(elem_size as u8);

    fahd.push(layout.page_bits);

    #[expect(
        clippy::cast_possible_truncation,
        reason = "element count written into the on-disk length width selected for this file"
    )]
    match length_size {
        4 => fahd.extend_from_slice(&(num_elements as u32).to_le_bytes()),
        8 => fahd.extend_from_slice(&(num_elements as u64).to_le_bytes()),
        _ => fahd.extend_from_slice(&(num_elements as u64).to_le_bytes()),
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "FADB address written into the on-disk offset width selected for this file"
    )]
    match offset_size {
        4 => fahd.extend_from_slice(&(fadb_address as u32).to_le_bytes()),
        8 => fahd.extend_from_slice(&fadb_address.to_le_bytes()),
        _ => fahd.extend_from_slice(&fadb_address.to_le_bytes()),
    }

    // Checksum
    let checksum = jenkins_lookup3(&fahd);
    fahd.extend_from_slice(&checksum.to_le_bytes());

    debug_assert_eq!(fahd.len(), fahd_total_size);

    // Append one element record (chunk address, plus filtered size + mask), or
    // the undefined address for a slot no chunk occupies — which is how a Fixed
    // Array says "this chunk of the maximum grid has never been written", and
    // what the reader tests before it decodes anything else about the element.
    let write_element = |buf: &mut Vec<u8>, chunk: Option<&WrittenChunk>| {
        let Some(chunk) = chunk else {
            write_undefined_element(buf, offset_size, has_filters, chunk_size_bytes);
            return;
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "chunk address written into the on-disk offset width selected for this file"
        )]
        match offset_size {
            4 => buf.extend_from_slice(&(chunk.address as u32).to_le_bytes()),
            _ => buf.extend_from_slice(&chunk.address.to_le_bytes()),
        }
        if has_filters {
            // Compressed size, written using the variable chunk_size_bytes width.
            let cs_bytes = chunk.compressed_size.to_le_bytes();
            buf.extend_from_slice(&cs_bytes[..chunk_size_bytes]);
            buf.extend_from_slice(&chunk.filter_mask.to_le_bytes());
        }
    };

    // Build FADB prefix: signature + version + client_id + header address.
    let mut fadb = Vec::new();
    fadb.extend_from_slice(b"FADB");
    fadb.push(0); // version
    fadb.push(client_id);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "fixed array header address written into the on-disk offset width selected for this file"
    )]
    match offset_size {
        4 => fadb.extend_from_slice(&(fa_address as u32).to_le_bytes()),
        _ => fadb.extend_from_slice(&fa_address.to_le_bytes()),
    }

    let page_size = layout.page_size;
    if num_elements <= page_size {
        // Non-paged: elements stored directly, then a single checksum.
        for slot in 0..num_elements {
            write_element(&mut fadb, slots.at(slot));
        }
        let fadb_checksum = jenkins_lookup3(&fadb);
        fadb.extend_from_slice(&fadb_checksum.to_le_bytes());
    } else {
        // Paged: a page-init bitmap and checksum follow the prefix, then each
        // page stores its elements followed by its own checksum. Every page is
        // reserved (the reader addresses them at a fixed stride) and every page
        // is marked initialized, including one whose slots are all empty: the
        // undefined addresses it holds already say the chunks are unwritten.
        let npages = num_elements.div_ceil(page_size);
        let bitmap_size = npages.div_ceil(8);
        let mut bitmap = vec![0u8; bitmap_size];
        for page in 0..npages {
            // Most-significant-bit-first ordering, matching H5VM_bit_set.
            bitmap[page / 8] |= 1 << (7 - (page % 8));
        }
        fadb.extend_from_slice(&bitmap);
        let prefix_checksum = jenkins_lookup3(&fadb);
        fadb.extend_from_slice(&prefix_checksum.to_le_bytes());

        for page in 0..npages {
            let start = page * page_size;
            let end = core::cmp::min(start + page_size, num_elements);
            let mut page_buf = Vec::with_capacity((end - start) * elem_size);
            for slot in start..end {
                write_element(&mut page_buf, slots.at(slot));
            }
            let page_checksum = jenkins_lookup3(&page_buf);
            page_buf.extend_from_slice(&page_checksum.to_le_bytes());
            fadb.extend_from_slice(&page_buf);
        }
    }

    let mut combined = fahd;
    combined.extend_from_slice(&fadb);
    // The length `fixed_array_len` promises a caller reserving space for this
    // array, checked against the bytes actually produced.
    debug_assert_eq!(
        combined.len() as u64,
        layout.total_len,
        "a fixed array must fill the length its layout promised"
    );
    combined
}

/// Serialize a v4 Extensible Array layout message.
pub(crate) fn serialize_v4_extensible_array(
    chunk_dims: &[u32],
    ea_address: u64,
    offset_size: u8,
    element_size: u32,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(4); // version
    buf.push(2); // class = chunked
    buf.push(0x00); // flags

    #[expect(
        clippy::cast_possible_truncation,
        reason = "rank written into the 1-byte dimensionality field selected for this file"
    )]
    let ndims = chunk_dims.len() as u8 + 1;
    buf.push(ndims);

    let max_dim = chunk_dims
        .iter()
        .map(|&d| d as u64)
        .chain(core::iter::once(element_size as u64))
        .max()
        .unwrap_or(1);
    let dim_encoded_len: u8 = if max_dim <= 0xFF {
        1
    } else if max_dim <= 0xFFFF {
        2
    } else {
        4
    };
    buf.push(dim_encoded_len);

    for &d in chunk_dims {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "dimension written into the on-disk encoding width selected for this file"
        )]
        match dim_encoded_len {
            1 => buf.push(d as u8),
            2 => buf.extend_from_slice(&(d as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&d.to_le_bytes()),
            _ => {}
        }
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "element size written into the on-disk encoding width selected for this file"
    )]
    match dim_encoded_len {
        1 => buf.push(element_size as u8),
        2 => buf.extend_from_slice(&(element_size as u16).to_le_bytes()),
        4 => buf.extend_from_slice(&element_size.to_le_bytes()),
        _ => {}
    }

    // chunk index type = 4 (Extensible Array)
    buf.push(4);

    // EA creation parameters (must match AEHD and HDF5 C library defaults)
    buf.push(32); // max_nelmts_bits
    buf.push(4); // idx_blk_elmts
    buf.push(4); // super_blk_min_data_ptrs
    buf.push(16); // data_blk_min_elmts
    buf.push(10); // max_dblk_page_nelmts_bits

    // EA header address
    #[expect(
        clippy::cast_possible_truncation,
        reason = "extensible array header address written into the on-disk offset width selected for this file"
    )]
    match offset_size {
        4 => buf.extend_from_slice(&(ea_address as u32).to_le_bytes()),
        8 => buf.extend_from_slice(&ea_address.to_le_bytes()),
        _ => {}
    }

    buf
}

/// Write an offset-sized address (little-endian) to `buf`.
pub(crate) fn write_ea_addr(buf: &mut Vec<u8>, val: u64, offset_size: u8) {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "address written into the on-disk offset width selected for this file"
    )]
    match offset_size {
        4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
        _ => buf.extend_from_slice(&val.to_le_bytes()),
    }
}

/// Build a single Extensible Array Data Block (`EADB`) holding the chunk
/// elements for `[elem_start, elem_start + dblk_nelmts)`. A slot no chunk
/// occupies is written as undefined, whether it falls past the last chunk or in a
/// gap between them.
///
/// When `dblk_nelmts` exceeds the page size the block is *paged*: the header
/// carries its own checksum and the elements are split into contiguous pages of
/// `page_nelmts` slots, each followed by a checksum. Every page is emitted; which
/// of them the owning super block marks initialized is that block's business, and
/// it asks `slots` the same question this does (see the page-init bitmap in
/// [`build_extensible_array_at`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_eadb(
    slots: &IndexSlots<'_>,
    elem_start: usize,
    dblk_nelmts: usize,
    block_offset_rel: u64,
    ea_address: u64,
    offset_size: u8,
    has_filters: bool,
    chunk_size_bytes: usize,
    client_id: u8,
    page_nelmts: usize,
    blk_off_size: usize,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"EADB");
    buf.push(0); // version
    buf.push(client_id);
    write_ea_addr(&mut buf, ea_address, offset_size);
    buf.extend_from_slice(&block_offset_rel.to_le_bytes()[..blk_off_size]);

    // The paging boundary has one definition; the counts here are already
    // `usize` loop bounds, so the page arithmetic below stays in that width.
    let blocks = DataBlockGeom {
        dblk_nelmts: dblk_nelmts as u64,
        page_nelmts: page_nelmts as u64,
    };
    if !blocks.is_paged() {
        // Non-paged: elements inline, single checksum.
        for slot in 0..dblk_nelmts {
            if let Some(chunk) = slots.at(elem_start + slot) {
                write_chunk_element(&mut buf, chunk, offset_size, has_filters, chunk_size_bytes);
            } else {
                write_undefined_element(&mut buf, offset_size, has_filters, chunk_size_bytes);
            }
        }
        let cks = jenkins_lookup3(&buf);
        buf.extend_from_slice(&cks.to_le_bytes());
        buf
    } else {
        // Paged: the header has its own checksum, then full pages follow. We
        // reserve every page (matching the C library's allocation) and report
        // how many leading pages hold real data so the super block can mark
        // them initialized in its bitmap.
        let header_cks = jenkins_lookup3(&buf);
        buf.extend_from_slice(&header_cks.to_le_bytes());

        let npages = dblk_nelmts / page_nelmts;
        for page in 0..npages {
            let page_start = elem_start + page * page_nelmts;
            let mut page_buf = Vec::new();
            for slot in 0..page_nelmts {
                if let Some(chunk) = slots.at(page_start + slot) {
                    write_chunk_element(
                        &mut page_buf,
                        chunk,
                        offset_size,
                        has_filters,
                        chunk_size_bytes,
                    );
                } else {
                    write_undefined_element(
                        &mut page_buf,
                        offset_size,
                        has_filters,
                        chunk_size_bytes,
                    );
                }
            }
            let page_cks = jenkins_lookup3(&page_buf);
            page_buf.extend_from_slice(&page_cks.to_le_bytes());
            buf.extend_from_slice(&page_buf);
        }
        buf
    }
}

/// Build an Extensible Array Super (secondary) Block (`EASB`) referencing
/// `dblk_addrs`. When `page_bitmap` is non-empty the block's data blocks are
/// paged and the bitmap (already populated by the caller) is written between
/// the block offset and the data block addresses.
pub(crate) fn build_aesb(
    ea_address: u64,
    block_offset_rel: u64,
    page_bitmap: &[u8],
    dblk_addrs: &[u64],
    offset_size: u8,
    blk_off_size: usize,
    client_id: u8,
) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"EASB");
    buf.push(0); // version
    buf.push(client_id);
    write_ea_addr(&mut buf, ea_address, offset_size);
    buf.extend_from_slice(&block_offset_rel.to_le_bytes()[..blk_off_size]);
    buf.extend_from_slice(page_bitmap);
    for &addr in dblk_addrs {
        write_ea_addr(&mut buf, addr, offset_size);
    }
    let cks = jenkins_lookup3(&buf);
    buf.extend_from_slice(&cks.to_le_bytes());
    buf
}

/// On-disk byte size of an Extensible Array index block (`EAIB`): the prefix
/// (signature, version, client id, header address), the always-written inline
/// element slots, the direct data-block and super-block address pointers, and a
/// trailing checksum. The single source of truth shared by the bulk writer
/// ([`build_extensible_array_at`]) and the in-place editor's reclaim walk, so
/// the two cannot disagree on how many bytes the index block occupies.
pub(crate) fn aeib_size(
    offset_size: u8,
    inline_elmts: usize,
    elem_size: usize,
    ndblk_addrs: usize,
    nsblk_addrs: usize,
) -> usize {
    let os = offset_size as usize;
    4 + 1 + 1 + os // signature + version + client id + header address
        + inline_elmts * elem_size // inline element slots (always all written)
        + ndblk_addrs * os // direct data-block addresses
        + nsblk_addrs * os // super-block addresses
        + 4 // checksum
}

/// The six Extensible Array header statistics, in the C library's stored order.
/// Read by the incremental append writer, and by [`ea_layout`], for which two of
/// them add up to the array's body length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EaStats {
    pub nsuper_blks: u64,
    pub super_blk_size: u64,
    pub ndata_blks: u64,
    pub data_blk_size: u64,
    pub max_idx_set: u64,
    pub nelmts: u64,
}

/// On-disk byte size of one non-paged Extensible Array data block (`EADB`)
/// holding `dblk_nelmts` element slots.
pub(crate) fn eadb_size(
    blocks: DataBlockGeom,
    elem_size: usize,
    offset_size: u8,
    blk_off_size: usize,
) -> u64 {
    let prefix = (4 + 1 + 1 + offset_size as u64) + blk_off_size as u64;
    if blocks.is_paged() {
        // Paged: the header carries its own checksum, then full pages follow.
        prefix + 4 + blocks.npages() * (blocks.page_nelmts * elem_size as u64 + 4)
    } else {
        prefix + blocks.dblk_nelmts * elem_size as u64 + 4
    }
}

/// On-disk byte size of one Extensible Array super block (`EASB`): its header,
/// the page-init bitmap when its data blocks are paged, its data-block
/// addresses, and its checksum.
///
/// Computed in `u64` throughout, so the block and page counts are never narrowed
/// to a 32-bit `usize` before being multiplied.
pub(crate) fn aesb_size(geom: SuperBlockGeom, offset_size: u8, blk_off_size: usize) -> u64 {
    let os = offset_size as u64;
    let header = 4 + 1 + 1 + os + blk_off_size as u64;
    header + geom.bitmap_size() + geom.ndblks * os + 4
}

/// Compute the six Extensible Array header statistics for an array holding
/// `num_elements` densely-filled elements. Mirrors the allocation performed by
/// [`build_extensible_array_at`] so the bulk writer and the incremental append
/// writer always agree (asserted by a unit test).
///
/// The two size statistics are also what [`extensible_array_len`] reports, since
/// the array's body is its allocated blocks and nothing else — so this walk
/// decides a reservation as well as a set of header fields.
pub(crate) fn ea_compute_stats(
    geom: &EaGeometry,
    idx_blk_elmts: u64,
    elem_size: usize,
    page_nelmts: u64,
    offset_size: u8,
    blk_off_size: usize,
    num_elements: u64,
    occupancy: SlotOccupancy<'_>,
) -> EaStats {
    let mut s = EaStats {
        nsuper_blks: 0,
        super_blk_size: 0,
        ndata_blks: 0,
        data_blk_size: 0,
        max_idx_set: num_elements,
        nelmts: idx_blk_elmts,
    };
    let mut elem = idx_blk_elmts;
    for &dn in &geom.direct_dblk_nelmts {
        if occupancy.any_occupied(elem, dn) {
            let blocks = DataBlockGeom {
                dblk_nelmts: dn,
                page_nelmts,
            };
            s.ndata_blks += 1;
            s.data_blk_size += eadb_size(blocks, elem_size, offset_size, blk_off_size);
            s.nelmts += dn;
        }
        elem += dn;
    }
    for j in 0..geom.nsblk_addrs {
        let sb = geom.super_block_at(j, page_nelmts);
        let (ndblks, dn) = (sb.ndblks, sb.blocks.dblk_nelmts);
        let span = ndblks * dn;
        if occupancy.any_occupied(elem, span) {
            s.nsuper_blks += 1;
            s.super_blk_size += aesb_size(sb, offset_size, blk_off_size);
            let mut le = elem;
            for _ in 0..ndblks {
                if occupancy.any_occupied(le, dn) {
                    s.ndata_blks += 1;
                    s.data_blk_size += eadb_size(sb.blocks, elem_size, offset_size, blk_off_size);
                    s.nelmts += dn;
                }
                le += dn;
            }
        }
        elem += span;
    }
    s
}

/// Everything about an Extensible Array that does not depend on where it is
/// placed: the element encoding, the block geometry, and every byte count.
///
/// [`build_extensible_array_at`] writes an array's bytes at a chosen
/// address, but no term of this layout is that address — each block's size, and
/// so the array's total length, is the same wherever it lands. That is what lets
/// [`extensible_array_len`] answer the length without emitting the array.
///
/// The builder shares this layout's *geometry*, so no second derivation of the
/// block sizes can drift from what is written. It does not share the walk that
/// decides which of those blocks a given element count allocates: that is
/// written once here (through [`ea_compute_stats`]) and once in the builder's
/// own body. Those two are what the length assertion at the end of the builder,
/// and `extensible_array_len_matches_what_it_builds`, hold together.
struct EaLayout {
    encoding: ChunkElementEncoding,
    /// EA creation parameters — these must match the HDF5 C library defaults
    /// exactly, and are held here so the header writer and the size computation
    /// read the same values.
    max_nelmts_bits: u8,
    idx_blk_elmts: u8,
    min_dblk_nelmts: u8,
    super_blk_min_nelmts: u8,
    max_dblk_nelmts_bits: u8,
    geom: EaGeometry,
    page_nelmts: usize,
    blk_off_size: usize,
    /// Element slots held inline in the index block (`idx_blk_elmts`).
    inline: usize,
    aehd_size: usize,
    aeib_size: usize,
    /// The six header statistics, two of which (`data_blk_size` and
    /// `super_blk_size`) are exactly the body's byte length.
    stats: EaStats,
    /// Header + index block + body: the whole array.
    total_len: u64,
}

/// Lay out the Extensible Array that would hold `num_slots` slots with
/// `occupancy` filled, without building it.
fn ea_layout(
    occupancy: SlotOccupancy<'_>,
    num_slots: u64,
    chunk_bytes: u64,
    offset_size: u8,
    length_size: u8,
    has_filters: bool,
) -> EaLayout {
    let encoding = chunk_element_encoding(chunk_bytes, offset_size, has_filters);
    let ChunkElementEncoding {
        elem_size,
        client_id,
        ..
    } = encoding;

    let (
        max_nelmts_bits,
        idx_blk_elmts,
        min_dblk_nelmts,
        super_blk_min_nelmts,
        max_dblk_nelmts_bits,
    ) = (
        EA_MAX_NELMTS_BITS,
        EA_IDX_BLK_ELMTS,
        EA_MIN_DBLK_NELMTS,
        EA_SUPER_BLK_MIN_NELMTS,
        EA_MAX_DBLK_NELMTS_BITS,
    );

    // Derive the block-size geometry from the shared helper (single source of
    // truth shared with the reader).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "element record size written into the 1-byte EA header field selected for this file"
    )]
    let geom_header = ExtensibleArrayHeader {
        client_id,
        element_size: elem_size as u8,
        max_nelmts_bits,
        idx_blk_elmts,
        min_dblk_nelmts,
        super_blk_min_nelmts,
        max_dblk_nelmts_bits,
        num_elements: 0,
        index_block_address: 0,
    };
    let geom = EaGeometry::from_header(&geom_header);
    let page_nelmts = 1usize << max_dblk_nelmts_bits;
    let blk_off_size = (max_nelmts_bits as usize).div_ceil(8);
    let inline = idx_blk_elmts as usize;

    let aehd_size = ExtensibleArrayHeader::serialized_size(offset_size, length_size);
    let aeib_size = aeib_size(
        offset_size,
        inline,
        elem_size,
        geom.direct_dblk_nelmts.len(),
        geom.nsblk_addrs,
    );

    // The body is the allocated data blocks and super blocks, concatenated with
    // nothing between them, so the two size statistics are its byte length.
    let stats = ea_compute_stats(
        &geom,
        idx_blk_elmts as u64,
        elem_size,
        page_nelmts as u64,
        offset_size,
        blk_off_size,
        num_slots,
        occupancy,
    );
    let total_len = (aehd_size + aeib_size) as u64 + stats.data_blk_size + stats.super_blk_size;

    EaLayout {
        encoding,
        max_nelmts_bits,
        idx_blk_elmts,
        min_dblk_nelmts,
        super_blk_min_nelmts,
        max_dblk_nelmts_bits,
        geom,
        page_nelmts,
        blk_off_size,
        inline,
        aehd_size,
        aeib_size,
        stats,
        total_len,
    }
}

/// The byte length [`build_extensible_array_at`] would produce for `chunks`,
/// without building it.
///
/// A caller that has to reserve space for the array before it exists — the
/// in-place editor placing one into freed space — needs the length first. It
/// comes from the same [`EaLayout`] the builder emits from, so no second
/// derivation of the block geometry can drift away from what is written.
pub(crate) fn extensible_array_len(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: u8,
    length_size: u8,
    has_filters: bool,
) -> u64 {
    ea_layout(
        SlotOccupancy::Sparse(slots),
        slots.len() as u64,
        chunk_bytes,
        offset_size,
        length_size,
        has_filters,
    )
    .total_len
}

/// Build a complete Extensible Array at a known absolute address.
///
/// Lays out the header (`EAHD`), index block (`EAIB`), and — for datasets with
/// more than `idx_blk_elmts + sum(direct data blocks)` chunks — the on-disk
/// super blocks (`EASB`) and their data blocks (`EADB`, paged when large). The
/// super-block / data-block size progression comes from the shared
/// [`EaGeometry`], so the writer and reader cannot drift. Byte-for-byte
/// compatible with the reference HDF5 C library across inline, direct, super
/// block, and paged ranges (verified by crosscheck tests).
pub fn build_extensible_array_at(
    slots: &IndexSlots<'_>,
    chunk_bytes: u64,
    offset_size: u8,
    length_size: u8,
    has_filters: bool,
    ea_address: u64,
) -> Result<Vec<u8>, FormatError> {
    let num_elements = slots.len();

    let layout = ea_layout(
        SlotOccupancy::Sparse(slots),
        slots.len() as u64,
        chunk_bytes,
        offset_size,
        length_size,
        has_filters,
    );
    let ChunkElementEncoding {
        chunk_size_bytes,
        elem_size,
        client_id,
    } = layout.encoding;
    let EaLayout {
        max_nelmts_bits,
        idx_blk_elmts,
        min_dblk_nelmts,
        super_blk_min_nelmts,
        max_dblk_nelmts_bits,
        ref geom,
        page_nelmts,
        blk_off_size,
        inline,
        aehd_size,
        aeib_size,
        ..
    } = layout;

    let aeib_address = ea_address + aehd_size as u64;
    let body_base = aeib_address + aeib_size as u64;

    let undef_addr: u64 = match offset_size {
        4 => 0xFFFF_FFFF,
        _ => u64::MAX,
    };

    // ---- Build the body (direct data blocks, then super blocks) -----------
    // Addresses are absolute, computed from `body_base`, so the body can be
    // built before the index block that references it.
    let mut body: Vec<u8> =
        Vec::with_capacity((layout.stats.data_blk_size + layout.stats.super_blk_size).to_usize()?);
    let mut direct_addrs: Vec<u64> = Vec::with_capacity(geom.direct_dblk_nelmts.len());
    let mut sblk_addrs: Vec<u64> = Vec::with_capacity(geom.nsblk_addrs);

    // Stats (match the C library's EAHD fields exactly).
    let mut ndata_blks: u64 = 0;
    let mut data_blk_size: u64 = 0;
    let mut nsuper_blks: u64 = 0;
    let mut super_blk_size: u64 = 0;
    let mut alloc_slots: u64 = inline as u64; // nelmts: idx slots + every allocated data block

    // Absolute element index past the inline slots. This walks the extensible
    // array's theoretical element space (up to `2^max_nelmts_bits` slots), which
    // exceeds a 32-bit `usize`, so it and the per-block spans are tracked in
    // `u64`; only the bounded, real-data values handed to the block builders are
    // narrowed (checked) to `usize`.
    let mut elem_cursor: u64 = inline as u64;

    // Direct data blocks: addresses stored directly in the index block. A block
    // no chunk lands in is not written at all, which is the one question this
    // walk and `ea_compute_stats` must answer identically -- see
    // [`SlotOccupancy`].
    let occupancy = SlotOccupancy::Sparse(slots);
    for &dblk_nelmts in &geom.direct_dblk_nelmts {
        if !occupancy.any_occupied(elem_cursor, dblk_nelmts) {
            direct_addrs.push(undef_addr);
            elem_cursor += dblk_nelmts;
            continue;
        }
        let addr = body_base + body.len() as u64;
        let db_bytes = build_eadb(
            slots,
            elem_cursor.to_usize()?,
            dblk_nelmts.to_usize()?,
            elem_cursor - inline as u64,
            ea_address,
            offset_size,
            has_filters,
            chunk_size_bytes,
            client_id,
            page_nelmts,
            blk_off_size,
        );
        ndata_blks += 1;
        data_blk_size += db_bytes.len() as u64;
        alloc_slots += dblk_nelmts;
        body.extend_from_slice(&db_bytes);
        direct_addrs.push(addr);
        elem_cursor += dblk_nelmts;
    }

    // Super blocks: addresses stored in the index block; super-block pointer `j`
    // refers to super block `first_indirect_sblk + j`.
    for j in 0..geom.nsblk_addrs {
        let sblk_idx = geom.first_indirect_sblk + j;
        // `ndblks` and `dblk_nelmts` are u64 element counts from the EA geometry.
        // Their product (this super block's element span) and the running cursor
        // walk the array's theoretical address space and can exceed a 32-bit
        // usize, so they stay in u64; only bounded, real-data counts are narrowed.
        let (ndblks, dblk_nelmts) = geom.sblks[sblk_idx];
        let sb_span = ndblks * dblk_nelmts;
        if !occupancy.any_occupied(elem_cursor, sb_span) {
            sblk_addrs.push(undef_addr);
            elem_cursor += sb_span;
            continue;
        }

        // Past the early-out this super block holds real data, so its block
        // counts are bounded by the (usize) chunk count and narrow safely.
        let sb = geom.super_block_at(j, page_nelmts as u64);
        let is_paged = sb.blocks.is_paged();
        let npages = sb.blocks.npages();
        let sb_block_offset = elem_cursor - inline as u64;
        let mut page_bitmap = vec![0u8; sb.bitmap_size().to_usize()?];

        let mut sb_dblk_addrs: Vec<u64> = Vec::with_capacity(ndblks.to_usize()?);
        let mut local_elem = elem_cursor;
        for db_local in 0..ndblks {
            if !occupancy.any_occupied(local_elem, dblk_nelmts) {
                sb_dblk_addrs.push(undef_addr);
                local_elem += dblk_nelmts;
                continue;
            }
            let addr = body_base + body.len() as u64;
            let db_bytes = build_eadb(
                slots,
                local_elem.to_usize()?,
                dblk_nelmts.to_usize()?,
                local_elem - inline as u64,
                ea_address,
                offset_size,
                has_filters,
                chunk_size_bytes,
                client_id,
                page_nelmts,
                blk_off_size,
            );
            ndata_blks += 1;
            data_blk_size += db_bytes.len() as u64;
            alloc_slots += dblk_nelmts;
            body.extend_from_slice(&db_bytes);
            sb_dblk_addrs.push(addr);
            if is_paged {
                // A page is marked initialized when it holds a chunk — the page
                // that holds it, not the n-th page for the n-th page's worth of
                // chunks. Those are the same list only while the array has no
                // gaps, which was true of every Extensible Array this crate
                // wrote until it began numbering slots over the maximum grid: a
                // gap then left chunks in pages nothing marked, and so in pages
                // no reader visits (issue #299).
                //
                // Marking every allocated page instead reads correctly too,
                // since every page is written and an unmarked one holds only
                // undefined addresses. It is this list because it is the list the
                // reference library writes: on a 140,000-chunk unlimited dataset
                // the final super block's bitmap is `ff 80` here and in libhdf5,
                // and `ff c0` if every page is marked. **No test asserts those
                // bytes** — locating a super block's bitmap in a file needs the
                // block's own geometry, so the tests pin the behaviour (a gapped
                // array reads completely, in both libraries) and not the
                // encoding. A change here that keeps files readable will not be
                // caught.
                let base = local_elem.to_usize()?;
                for p in 0..npages.to_usize()? {
                    let page_start = base + p * page_nelmts;
                    if !(0..page_nelmts).any(|s| slots.at(page_start + s).is_some()) {
                        continue;
                    }
                    let global_page = (db_local * npages).to_usize()? + p;
                    page_bitmap[global_page / 8] |= 0x80 >> (global_page % 8);
                }
            }
            local_elem += dblk_nelmts;
        }

        let aesb_addr = body_base + body.len() as u64;
        let aesb = build_aesb(
            ea_address,
            sb_block_offset,
            &page_bitmap,
            &sb_dblk_addrs,
            offset_size,
            blk_off_size,
            client_id,
        );
        nsuper_blks += 1;
        super_blk_size += aesb.len() as u64;
        body.extend_from_slice(&aesb);
        sblk_addrs.push(aesb_addr);

        elem_cursor += sb_span;
    }

    // ---- Build the header (EAHD) ------------------------------------------
    #[expect(
        clippy::cast_possible_truncation,
        reason = "statistic written into the on-disk length width selected for this file"
    )]
    let write_length = |buf: &mut Vec<u8>, val: u64| match length_size {
        4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
        _ => buf.extend_from_slice(&val.to_le_bytes()),
    };

    let mut aehd = Vec::with_capacity(aehd_size);
    aehd.extend_from_slice(b"EAHD");
    aehd.push(0); // version
    aehd.push(client_id);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "element record size written into the 1-byte EA header field selected for this file"
    )]
    aehd.push(elem_size as u8);
    aehd.push(max_nelmts_bits);
    aehd.push(idx_blk_elmts);
    aehd.push(min_dblk_nelmts);
    aehd.push(super_blk_min_nelmts);
    aehd.push(max_dblk_nelmts_bits);

    // 6 statistics, in the C library's order:
    //   [0] nsuper_blks   [1] super_blk_size   [2] ndata_blks
    //   [3] data_blk_size [4] max_idx_set      [5] nelmts
    write_length(&mut aehd, nsuper_blks);
    write_length(&mut aehd, super_blk_size);
    write_length(&mut aehd, ndata_blks);
    write_length(&mut aehd, data_blk_size);
    write_length(&mut aehd, num_elements as u64); // max_idx_set (dense fill)
    write_length(&mut aehd, alloc_slots); // nelmts (allocated slots)

    write_ea_addr(&mut aehd, aeib_address, offset_size);

    let aehd_checksum = jenkins_lookup3(&aehd);
    aehd.extend_from_slice(&aehd_checksum.to_le_bytes());
    debug_assert_eq!(aehd.len(), aehd_size);

    // ---- Build the index block (EAIB) -------------------------------------
    let mut aeib = Vec::with_capacity(aeib_size);
    aeib.extend_from_slice(b"EAIB");
    aeib.push(0); // version
    aeib.push(client_id);
    write_ea_addr(&mut aeib, ea_address, offset_size);

    // Inline elements (always write idx_blk_elmts slots; fill unused as undefined).
    #[allow(clippy::needless_range_loop)]
    for i in 0..inline {
        if let Some(chunk) = slots.at(i) {
            write_chunk_element(&mut aeib, chunk, offset_size, has_filters, chunk_size_bytes);
        } else {
            write_undefined_element(&mut aeib, offset_size, has_filters, chunk_size_bytes);
        }
    }
    // Direct data block addresses, then super block addresses.
    for &addr in &direct_addrs {
        write_ea_addr(&mut aeib, addr, offset_size);
    }
    for &addr in &sblk_addrs {
        write_ea_addr(&mut aeib, addr, offset_size);
    }

    let aeib_checksum = jenkins_lookup3(&aeib);
    aeib.extend_from_slice(&aeib_checksum.to_le_bytes());
    debug_assert_eq!(aeib.len(), aeib_size);

    let mut combined = aehd;
    combined.extend_from_slice(&aeib);
    combined.extend_from_slice(&body);
    // The length `extensible_array_len` promises a caller reserving space for
    // this array, checked against the bytes actually produced. A reservation
    // that disagrees with the emission would place the next object on top of
    // this one, so pin it where it is emitted as well as in a test.
    debug_assert_eq!(
        combined.len() as u64,
        layout.total_len,
        "an extensible array must fill the length its layout promised"
    );
    Ok(combined)
}

fn write_chunk_element(
    buf: &mut Vec<u8>,
    chunk: &WrittenChunk,
    offset_size: u8,
    has_filters: bool,
    chunk_size_bytes: usize,
) {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "chunk address written into the on-disk offset width selected for this file"
    )]
    match offset_size {
        4 => buf.extend_from_slice(&(chunk.address as u32).to_le_bytes()),
        8 => buf.extend_from_slice(&chunk.address.to_le_bytes()),
        _ => buf.extend_from_slice(&chunk.address.to_le_bytes()),
    }
    if has_filters {
        let cs_bytes = chunk.compressed_size.to_le_bytes();
        buf.extend_from_slice(&cs_bytes[..chunk_size_bytes]);
        buf.extend_from_slice(&chunk.filter_mask.to_le_bytes());
    }
}

fn write_undefined_element(
    buf: &mut Vec<u8>,
    offset_size: u8,
    has_filters: bool,
    chunk_size_bytes: usize,
) {
    let os = offset_size as usize;
    buf.extend_from_slice(&vec![0xFF; os]);
    if has_filters {
        buf.extend_from_slice(&vec![0x00; chunk_size_bytes]);
        buf.extend_from_slice(&0u32.to_le_bytes());
    }
}

/// A chunked dataset's chunks already split and compressed — the expensive,
/// **address-independent** half of building a chunked layout. The compressed
/// bytes, the chunk-index choice, and the pipeline message do not depend on
/// where the data lands in the file; only the absolute addresses embedded in the
/// chunk index do. The writer sizes a dataset's object header in one pass and
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
        written_chunks: &'a [WrittenChunk],
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
    maxshape: Option<&[u64]>,
    fill: FillPattern<'_>,
    allocation: StorageAllocation,
) -> Result<CompressedChunkSet, FormatError> {
    let chunk_dims = ctx.chunk_dims;
    let element_size = nonzero_usize_from(ctx.element_size)?;
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
        pipeline_message: pipeline.as_ref().map(|pl| pl.serialize()),
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
    maxshape: Option<&[u64]>,
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
    let encoding = chunk_element_encoding(chunk_bytes, INDEX_OFFSET_SIZE, has_filters);
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
            let capacity = ea_addressable_slots();
            if index_slots > capacity {
                return Err(FormatError::ChunkedReadError(format!(
                    "this shape and maximum shape number a chunk at element {} of the chunk \
                     index, past the {capacity} an extensible array can address; the chunk would \
                     be dropped. Chunk the dimensions the dataset does not grow along more \
                     coarsely, or give them a smaller maximum",
                    index_slots - 1,
                )));
            }
            ea_layout(
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

/// The Extensible Array creation parameters this crate writes, which are the
/// reference C library's defaults.
///
/// Constants rather than locals because [`ea_addressable_slots`] derives the
/// array's capacity from the same five numbers the layout is built from. Two
/// copies of them is how a capacity comes to describe an array nobody writes.
const EA_MAX_NELMTS_BITS: u8 = 32;
const EA_IDX_BLK_ELMTS: u8 = 4;
const EA_MIN_DBLK_NELMTS: u8 = 16;
const EA_SUPER_BLK_MIN_NELMTS: u8 = 4;
const EA_MAX_DBLK_NELMTS_BITS: u8 = 10;

/// How many element slots an Extensible Array written with this crate's creation
/// parameters can address: the index block's inline slots, plus every direct data
/// block, plus every block of every super block the index block points at.
///
/// Derived from the geometry rather than written down, because it *is* the
/// geometry: change `max_nelmts_bits` or `min_dblk_nelmts` and this changes with
/// them. With the current parameters it comes to 8,589,934,580 — four inline
/// slots, 240 across six direct blocks, and 8,589,934,336 across 25 super blocks.
fn ea_addressable_slots() -> u64 {
    let geom_header = ExtensibleArrayHeader {
        client_id: 0,
        element_size: 8,
        max_nelmts_bits: EA_MAX_NELMTS_BITS,
        idx_blk_elmts: EA_IDX_BLK_ELMTS,
        min_dblk_nelmts: EA_MIN_DBLK_NELMTS,
        super_blk_min_nelmts: EA_SUPER_BLK_MIN_NELMTS,
        max_dblk_nelmts_bits: EA_MAX_DBLK_NELMTS_BITS,
        num_elements: 0,
        index_block_address: 0,
    };
    let geom = EaGeometry::from_header(&geom_header);
    let direct: u64 = geom.direct_dblk_nelmts.iter().sum();
    let indirect: u64 = (0..geom.nsblk_addrs)
        .map(|j| {
            let (ndblks, dn) = geom.sblks[geom.first_indirect_sblk + j];
            ndblks * dn
        })
        .sum();
    u64::from(EA_IDX_BLK_ELMTS) + direct + indirect
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
    maxshape: Option<&[u64]>,
) -> Result<ChunkGrid, FormatError> {
    let order = if maxshape.is_some_and(|ms| ms.contains(&u64::MAX)) {
        GridOrder::UnlimitedFirst
    } else {
        GridOrder::RowMajor
    };
    ChunkGrid::new(chunk_dims, shape, maxshape, order)
}

/// Where each of a chunk set's chunks lands when the set is laid out at
/// `data_address` — they are stored back to back from there — and the address the
/// chunk index follows them at.
fn plan_chunk_slots(set: &CompressedChunkSet, data_address: u64) -> (Vec<WrittenChunk>, u64) {
    let mut cursor = data_address;
    let mut written_chunks = Vec::with_capacity(set.compressed.len());
    for chunk in &set.compressed {
        written_chunks.push(WrittenChunk {
            address: cursor,
            compressed_size: chunk.len() as u64,
            filter_mask: 0,
        });
        cursor += chunk.len() as u64;
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
    written_chunks: &[WrittenChunk],
    slots: &IndexSlots<'_>,
    index_address: u64,
) -> Result<(Vec<u8>, Vec<u8>), FormatError> {
    let index = match set.kind {
        // Nothing stored, so nothing to index; the layout message below carries
        // the undefined address in place of one.
        ChunkIndexKind::Unallocated => Vec::new(),
        ChunkIndexKind::ExtensibleArray => build_extensible_array_at(
            slots,
            set.full_chunk_bytes(),
            INDEX_OFFSET_SIZE,
            INDEX_LENGTH_SIZE,
            set.has_filters,
            index_address,
        )?,
        ChunkIndexKind::SingleChunk => Vec::new(),
        ChunkIndexKind::FixedArray => build_fixed_array_at(
            slots,
            set.full_chunk_bytes(),
            INDEX_OFFSET_SIZE,
            INDEX_LENGTH_SIZE,
            set.has_filters,
            index_address,
        ),
    };
    Ok((
        index,
        chunk_index_layout(set, written_chunks, index_address),
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
    written_chunks: &[WrittenChunk],
    index_address: u64,
) -> Vec<u8> {
    let has_filters = set.has_filters;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "element size written into the on-disk u32 dimension field selected for this file"
    )]
    match set.kind {
        ChunkIndexKind::Unallocated => serialize_v4_fixed_array(
            &set.chunk_dims_u32,
            HADDR_UNDEF,
            INDEX_OFFSET_SIZE,
            set.element_size.get() as u32,
            FIXED_ARRAY_PAGE_BITS,
        ),
        ChunkIndexKind::ExtensibleArray => serialize_v4_extensible_array(
            &set.chunk_dims_u32,
            index_address,
            INDEX_OFFSET_SIZE,
            set.element_size.get() as u32,
        ),
        ChunkIndexKind::SingleChunk => {
            let chunk = &written_chunks[0];
            serialize_v4_single_chunk(
                &set.chunk_dims_u32,
                chunk.address,
                has_filters.then_some(chunk.compressed_size),
                has_filters.then_some(0u32),
                INDEX_OFFSET_SIZE,
                set.element_size.get() as u32,
            )
        }
        ChunkIndexKind::FixedArray => serialize_v4_fixed_array(
            &set.chunk_dims_u32,
            index_address,
            INDEX_OFFSET_SIZE,
            set.element_size.get() as u32,
            FIXED_ARRAY_PAGE_BITS,
        ),
    }
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
    let (written_chunks, index_address) = plan_chunk_slots(set, 0);
    let slots = set.index_slots(&written_chunks)?;
    let kind = set.kind;
    Ok(index_address
        + kind.array_kind().map_or(0, |array| {
            chunk_index_len(
                array,
                &slots,
                set.full_chunk_bytes(),
                INDEX_OFFSET_SIZE,
                INDEX_LENGTH_SIZE,
                set.has_filters,
            )
        }))
}

/// The chunk-index bytes and data-layout message for `set` at `data_address`,
/// with the total size of its chunk payload.
///
/// Everything [`assemble_chunked_at`] produces except the data region itself, so
/// that [`measure_chunked_at`] can answer "how long, and what does the layout
/// message say" without building an entire copy of the dataset.
fn plan_chunked_at(
    set: &CompressedChunkSet,
    data_address: u64,
) -> Result<(usize, Vec<u8>, Vec<u8>), FormatError> {
    let (written_chunks, index_address) = plan_chunk_slots(set, data_address);
    let slots = set.index_slots(&written_chunks)?;
    let (index, layout_message) = chunk_index_bytes(set, &written_chunks, &slots, index_address)?;
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
    data_address: u64,
) -> Result<ChunkedMeasure, FormatError> {
    let (written_chunks, index_address) = plan_chunk_slots(set, data_address);
    let slots = set.index_slots(&written_chunks)?;
    // Derived from the plan already in hand rather than by building the index —
    // and from *this* plan rather than by calling `chunked_data_len`, which
    // would lay the chunks out a second time to reach the same answer.
    let index_len = set.kind.array_kind().map_or(0, |array| {
        chunk_index_len(
            array,
            &slots,
            set.full_chunk_bytes(),
            INDEX_OFFSET_SIZE,
            INDEX_LENGTH_SIZE,
            set.has_filters,
        )
    });
    let data_len = (index_address - data_address) + index_len;
    // Not building the index also stops it from *refusing*, and the only way it
    // can is a length that does not fit this platform's `usize`. Every such
    // check inside the build is on a part of this region, so the whole region
    // fitting means all of them do: the refusal survives the build's removal,
    // and stays where it was — before the writer has emitted a byte.
    data_len.to_usize()?;
    Ok(ChunkedMeasure {
        data_len,
        layout_message: chunk_index_layout(set, &written_chunks, index_address),
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

/// Lay an already-[`compress`ed](compress_chunks) chunk set out at `data_address`,
/// producing the on-disk data region (chunk bytes followed by the chunk index)
/// and the v4 data-layout message. Cheap: this only concatenates and builds the
/// index, so it can be run more than once (different addresses) without
/// repeating the dataset's compression.
pub(crate) fn assemble_chunked_at(
    set: &CompressedChunkSet,
    data_address: u64,
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

/// Build chunked data with absolute addresses and optional maxshape.
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
    data_address: u64,
    maxshape: Option<&[u64]>,
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
    address: u64,
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
    pub(crate) chunks: Vec<WrittenChunk>,
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

/// Compute the [`VerbatimLayout`] (data-region plan plus the v4 layout and
/// verbatim pipeline messages) for a dense, grid-ordered set of chunks, from
/// their sizes and filter masks alone — no chunk bytes. The byte layout is
/// identical whether the chunks are later buffered or streamed.
pub(crate) fn plan_chunked_data_verbatim(
    meta: &[ChunkMeta],
    shape: &[u64],
    chunk_dims: &[u64],
    element_size: NonZeroUsize,
    pipeline_message: Option<&[u8]>,
    data_address: u64,
    maxshape: Option<&[u64]>,
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
        let address = data_address + cursor;
        let compressed_size = m.compressed_size;
        written_chunks.push(WrittenChunk {
            address,
            compressed_size,
            filter_mask: m.filter_mask,
        });
        cursor += compressed_size;
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
    let (kind, slot_of_chunk, index_slots) = plan_index_slots(
        shape,
        chunk_dims,
        maxshape,
        full_chunk_bytes(chunk_dims.iter().copied(), element_size),
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
    let slots = IndexSlots::new(&written_chunks, &slot_of_chunk, index_slots)?;

    // The index sits immediately after the chunk bytes. Its length is taken from
    // the index's own layout rather than from a build of it, so this planner
    // touches no index bytes either — which is what lets `write_chunked_relocatable`
    // plan at a provisional address purely to size the region.
    let index_address = data_address + cursor;
    let chunk_bytes = full_chunk_bytes(chunk_dims.iter().copied(), element_size);
    let index = kind.array_kind().map(|array| VerbatimIndexPlan {
        kind: array,
        address: index_address,
        has_filters,
        chunk_bytes,
        len: chunk_index_len(
            array,
            &slots,
            chunk_bytes,
            offset_size,
            length_size,
            has_filters,
        ),
    });
    cursor += index.as_ref().map_or(0, |i| i.len);

    #[expect(
        clippy::cast_possible_truncation,
        reason = "element size written into the on-disk u32 dimension field selected for this file"
    )]
    let layout_message = match kind {
        // Unreachable here: a verbatim plan is refused above unless it has at
        // least one chunk, and only a chunk-less fixed-shape set is unallocated.
        ChunkIndexKind::Unallocated => serialize_v4_fixed_array(
            &chunk_dims_u32,
            HADDR_UNDEF,
            offset_size,
            element_size.get() as u32,
            FIXED_ARRAY_PAGE_BITS,
        ),
        ChunkIndexKind::ExtensibleArray => serialize_v4_extensible_array(
            &chunk_dims_u32,
            index_address,
            offset_size,
            element_size.get() as u32,
        ),
        ChunkIndexKind::SingleChunk => {
            let chunk_addr = written_chunks[0].address;
            let filtered_size = if has_filters {
                Some(written_chunks[0].compressed_size)
            } else {
                None
            };
            let filter_mask = if has_filters {
                Some(written_chunks[0].filter_mask)
            } else {
                None
            };
            serialize_v4_single_chunk(
                &chunk_dims_u32,
                chunk_addr,
                filtered_size,
                filter_mask,
                offset_size,
                element_size.get() as u32,
            )
        }
        ChunkIndexKind::FixedArray => serialize_v4_fixed_array(
            &chunk_dims_u32,
            index_address,
            offset_size,
            element_size.get() as u32,
            FIXED_ARRAY_PAGE_BITS,
        ),
    };

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
        if chunk.len() as u64 != slot.compressed_size {
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
            ChunkArrayKind::ExtensibleArray => build_extensible_array_at(
                &slots,
                index.chunk_bytes,
                INDEX_OFFSET_SIZE,
                INDEX_LENGTH_SIZE,
                index.has_filters,
                index.address,
            )?,
            ChunkArrayKind::FixedArray => build_fixed_array_at(
                &slots,
                index.chunk_bytes,
                INDEX_OFFSET_SIZE,
                INDEX_LENGTH_SIZE,
                index.has_filters,
                index.address,
            ),
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
    use crate::data_layout::DataLayout;
    use crate::dataspace::{Dataspace, DataspaceType};
    use crate::datatype::{Datatype, DatatypeByteOrder};
    use crate::fill_value::FillPattern;
    use crate::read_spec::RawReadSpec;

    fn make_f64_type() -> Datatype {
        Datatype::FloatingPoint {
            size: 8,
            byte_order: DatatypeByteOrder::LittleEndian,
            bit_offset: 0,
            bit_precision: 64,
            exponent_location: 52,
            exponent_size: 11,
            mantissa_location: 0,
            mantissa_size: 52,
            exponent_bias: 1023,
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
            Some(&[u64::MAX]),
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

        let assembled = assemble_chunked_at(&set, 0x1000).unwrap();
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
                chunk_element_encoding(chunk_bytes, INDEX_OFFSET_SIZE, has_filters).elem_size
                    as u32,
            );
            let widest = MAX_UNUSED_INDEX_BYTES / elem + 1;
            plan_index_slots(
                &[1],
                &[1],
                Some(&[widest]),
                chunk_bytes,
                has_filters,
                StorageAllocation::Allocated,
            )
            .expect("an index exactly at the budget is written");
            let err = plan_index_slots(
                &[1],
                &[1],
                Some(&[widest + 1]),
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
            Some(&[u64::MAX, 8]),
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
        let capacity = ea_addressable_slots();
        assert_eq!(
            capacity, 8_589_934_580,
            "the C library's default creation parameters address this many slots"
        );

        // Two chunks a stride apart: the second sits at slot `stride`.
        let at = |stride: u64| {
            plan_index_slots(
                &[2, 1],
                &[1, 1],
                Some(&[u64::MAX, stride]),
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
            let err = auto
                .validate_geometry(&shape, Some(&vec![u64::MAX; shape.len()]))
                .unwrap_err();
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
        assert!(explicit.validate_geometry(&[0], Some(&[u64::MAX])).is_ok());
        assert!(explicit.validate_geometry(&[0], None).is_ok());
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
        type Case = (&'static [u64], &'static [u64], Option<&'static [u64]>);

        // Chunk counts chosen for the kind each selects: one chunk is
        // `SingleChunk`, a fixed shape with many is `FixedArray`, and an
        // unlimited maxshape is `ExtensibleArray`.
        let cases: [Case; 4] = [
            (&[512], &[512], None),
            (&[4096], &[512], None),
            (&[4096], &[64], None),
            (&[4096], &[512], Some(&[u64::MAX])),
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

            for base in [0u64, 0x1000, 0x1234_5678] {
                let measured = measure_chunked_at(&set, base).unwrap();
                let assembled = assemble_chunked_at(&set, base).unwrap();
                assert_eq!(
                    measured.data_len,
                    assembled.data_bytes.len() as u64,
                    "measured and assembled lengths differ for shape {shape:?} in \
                     chunks {chunk_dims:?} at {base:#x}"
                );
                assert_eq!(
                    measured.layout_message, assembled.layout_message,
                    "measured and assembled layout messages differ for shape \
                     {shape:?} in chunks {chunk_dims:?} at {base:#x}"
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
        let data_address = 0x1000u64;
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
        let file_size = data_address as usize + result.data_bytes.len();
        let mut file_data = vec![0u8; file_size];
        file_data[data_address as usize..].copy_from_slice(&result.data_bytes);

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
            RawReadSpec {
                layout: &layout,
                dataspace: &dataspace,
                datatype: &datatype,
                pipeline: pipeline.as_ref(),
                fill: FillPattern::ZERO,
            },
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
        let result =
            build_chunked_data_at_ext(&raw, &[21], ctx, &options, 0x1000, None, FillPattern::ZERO)
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
        let layout =
            plan_chunked_data_verbatim(&meta, &[21], &[7], nz(8), Some(&[]), 0x1000, None).unwrap();

        let planned: Vec<u64> = layout
            .plan
            .chunks
            .iter()
            .map(|c| c.compressed_size)
            .collect();
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
                Some(&[u64::MAX][..]),
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
            let layout =
                plan_chunked_data_verbatim(&meta, shape, &[7], nz(8), Some(&[]), 0x1000, maxshape)
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
        let result = plan_chunked_data_verbatim(&[], &[21], &[7], nz(8), None, 0x1000, None);
        assert!(
            matches!(result, Err(FormatError::ChunkedReadError(_))),
            "a chunk-less plan must be refused"
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

    #[test]
    fn serialize_v4_single_chunk_no_filters_roundtrip() {
        let msg = serialize_v4_single_chunk(&[20], 0x1000, None, None, 8, 8);
        let layout = DataLayout::parse(&msg, 8, 8).unwrap();
        match layout {
            DataLayout::Chunked {
                chunk_dimensions,
                btree_address,
                version,
                chunk_index_type,
                single_chunk_filtered_size,
                single_chunk_filter_mask,
            } => {
                assert_eq!(version, 4);
                assert_eq!(chunk_index_type, Some(1));
                assert_eq!(chunk_dimensions, vec![20, 8]);
                assert_eq!(btree_address, Some(0x1000));
                assert_eq!(single_chunk_filtered_size, None);
                assert_eq!(single_chunk_filter_mask, None);
            }
            _ => panic!("expected chunked layout"),
        }
    }

    #[test]
    fn serialize_v4_single_chunk_with_filters_roundtrip() {
        let msg = serialize_v4_single_chunk(&[100], 0x2000, Some(500), Some(0), 8, 8);
        let layout = DataLayout::parse(&msg, 8, 8).unwrap();
        match layout {
            DataLayout::Chunked {
                btree_address,
                single_chunk_filtered_size,
                single_chunk_filter_mask,
                ..
            } => {
                assert_eq!(btree_address, Some(0x2000));
                assert_eq!(single_chunk_filtered_size, Some(500));
                assert_eq!(single_chunk_filter_mask, Some(0));
            }
            _ => panic!("expected chunked layout"),
        }
    }

    #[test]
    fn serialize_v4_fixed_array_roundtrip() {
        let msg = serialize_v4_fixed_array(&[20], 0x3000, 8, 8, 4);
        let layout = DataLayout::parse(&msg, 8, 8).unwrap();
        match layout {
            DataLayout::Chunked {
                version,
                chunk_index_type,
                btree_address,
                chunk_dimensions,
                ..
            } => {
                assert_eq!(version, 4);
                assert_eq!(chunk_index_type, Some(3));
                assert_eq!(btree_address, Some(0x3000));
                assert_eq!(chunk_dimensions, vec![20, 8]);
            }
            _ => panic!("expected chunked layout"),
        }
    }

    #[test]
    fn build_fixed_array_valid_structure() {
        let chunks = vec![
            WrittenChunk {
                address: 0x1000,
                compressed_size: 160,
                filter_mask: 0,
            },
            WrittenChunk {
                address: 0x10A0,
                compressed_size: 160,
                filter_mask: 0,
            },
        ];
        let fa = build_fixed_array_at(&IndexSlots::dense(&chunks), 160, 8, 8, false, 0x2000);
        // Should start with FAHD
        assert_eq!(&fa[0..4], b"FAHD");
        // FAHD size = 4+1+1+1+1+8+8+4 = 28
        // FADB starts at offset 28
        assert_eq!(&fa[28..32], b"FADB");
    }

    // ---- Extensible Array tests ----

    #[test]
    fn serialize_v4_extensible_array_roundtrip() {
        let msg = serialize_v4_extensible_array(&[10], 0x4000, 8, 8);
        let layout = DataLayout::parse(&msg, 8, 8).unwrap();
        match layout {
            DataLayout::Chunked {
                version,
                chunk_index_type,
                btree_address,
                chunk_dimensions,
                ..
            } => {
                assert_eq!(version, 4);
                assert_eq!(chunk_index_type, Some(4));
                assert_eq!(btree_address, Some(0x4000));
                assert_eq!(chunk_dimensions, vec![10, 8]);
            }
            _ => panic!("expected chunked layout"),
        }
    }

    #[test]
    fn build_extensible_array_valid_structure() {
        let chunks = vec![
            WrittenChunk {
                address: 0x1000,
                compressed_size: 80,
                filter_mask: 0,
            },
            WrittenChunk {
                address: 0x1050,
                compressed_size: 80,
                filter_mask: 0,
            },
        ];
        let ea = build_extensible_array_at(&IndexSlots::dense(&chunks), 80, 8, 8, false, 0x2000)
            .unwrap();
        assert_eq!(&ea[0..4], b"EAHD");
        // Find EAIB after EAHD: 12 fixed + 6*8 stats + 8 addr + 4 checksum = 72
        let aehd_size = 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 6 * 8 + 8 + 4;
        assert_eq!(&ea[aehd_size..aehd_size + 4], b"EAIB");
    }

    /// Helper: roundtrip with EA (maxshape)
    fn roundtrip_ea(
        values: &[f64],
        shape: &[u64],
        chunk_dims: &[u64],
        maxshape: &[u64],
    ) -> Vec<f64> {
        let raw = f64_to_bytes(values);
        let data_address = 0x1000u64;
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

        let file_size = data_address as usize + result.data_bytes.len();
        let mut file_data = vec![0u8; file_size];
        file_data[data_address as usize..].copy_from_slice(&result.data_bytes);

        let layout = DataLayout::parse(&result.layout_message, 8, 8).unwrap();
        // Verify it uses EA index
        match &layout {
            DataLayout::Chunked {
                chunk_index_type, ..
            } => {
                assert_eq!(*chunk_index_type, Some(4), "expected EA index type");
            }
            _ => panic!("expected chunked layout"),
        }

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
        let result = roundtrip_ea(&values, &[10], &[10], &[u64::MAX]);
        assert_eq!(result, values);
    }

    #[test]
    fn ea_roundtrip_1d_multi_chunks() {
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let result = roundtrip_ea(&values, &[20], &[5], &[u64::MAX]);
        assert_eq!(result, values);
    }

    #[test]
    fn ea_roundtrip_1d_many_chunks() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = roundtrip_ea(&values, &[100], &[10], &[u64::MAX]);
        assert_eq!(result, values);
    }

    /// One chunk per element across the inline, direct-data-block, and
    /// super-block ranges. Before the geometry fix these silently corrupted
    /// past 20 chunks (4 inline + the first 16-element direct block).
    #[test]
    fn ea_roundtrip_super_block_sizes() {
        for &n in &[245u64, 300, 2000, 50000] {
            let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let result = roundtrip_ea(&values, &[n], &[1], &[u64::MAX]);
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
        let result = roundtrip_ea(&values, &[n], &[1], &[u64::MAX]);
        assert_eq!(result.len(), n as usize);
        assert_eq!(result, values);
    }

    /// `ea_compute_stats` must reproduce the EAHD statistics that
    /// `build_extensible_array_at` actually writes (these feed the in-place
    /// append writer, so any drift would corrupt appended files).
    #[cfg(feature = "std")]
    #[test]
    fn ea_compute_stats_matches_builder() {
        use crate::extensible_array::{EaGeometry, ExtensibleArrayHeader};
        let geom_header = ExtensibleArrayHeader {
            client_id: 0,
            element_size: 8,
            max_nelmts_bits: 32,
            idx_blk_elmts: 4,
            min_dblk_nelmts: 16,
            super_blk_min_nelmts: 4,
            max_dblk_nelmts_bits: 10,
            num_elements: 0,
            index_block_address: 0,
        };
        let geom = EaGeometry::from_header(&geom_header);
        for &n in &[1u64, 4, 20, 100, 244, 300, 2000, 50000, 131056, 140000] {
            let chunks: Vec<WrittenChunk> = (0..n)
                .map(|i| WrittenChunk {
                    address: 0x1000 + i * 8,
                    compressed_size: 8,
                    filter_mask: 0,
                })
                .collect();
            let ea =
                build_extensible_array_at(&IndexSlots::dense(&chunks), 8, 8, 8, false, 0x100000)
                    .unwrap();
            // Parse the 6 stats from the EAHD (12-byte fixed prefix, then 6 * ls).
            let stat =
                |k: usize| u64::from_le_bytes(ea[12 + k * 8..12 + k * 8 + 8].try_into().unwrap());
            let built = super::EaStats {
                nsuper_blks: stat(0),
                super_blk_size: stat(1),
                ndata_blks: stat(2),
                data_blk_size: stat(3),
                max_idx_set: stat(4),
                nelmts: stat(5),
            };
            let computed =
                super::ea_compute_stats(&geom, 4, 8, 1024, 8, 4, n, SlotOccupancy::Dense(n));
            assert_eq!(computed, built, "stats mismatch at n={n}");
        }
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
                    let maxshape = unlimited.then_some([u64::MAX]);
                    let set = compress_chunks(
                        &raw,
                        &[elements],
                        ChunkContext::basic(&dims, 8),
                        &options,
                        maxshape.as_ref().map(<[u64; 1]>::as_slice),
                        FillPattern::ZERO,
                        StorageAllocation::Allocated,
                    )
                    .unwrap();

                    let planned = chunked_data_len(&set).unwrap();
                    let assembled = assemble_chunked_at(&set, 0x10_0000).unwrap();
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

    /// Raw chunk sizes that select four *different* compressed-size field widths
    /// in a filtered element record, so a sweep over them varies the element,
    /// page and index-block sizes rather than repeating one shape. Shared by the
    /// Fixed and Extensible Array length tests, which since
    /// `chunk_element_encoding` was unified exercise the same derivation;
    /// `extensible_array_len_matches_what_it_builds` asserts the four widths
    /// really are distinct.
    /// Whole-chunk byte sizes that select four *different* compressed-size field
    /// widths in a filtered element record; see `CHUNK_BYTES` at its use.
    const CHUNK_BYTES: [u64; 4] = [8, 300, 100_000, 1 << 32];

    /// `fixed_array_len` is the span a caller reserves for a Fixed Array before a
    /// byte of it exists, so it has to equal the length `build_fixed_array_at`
    /// goes on to emit.
    ///
    /// Swept contiguously past the page size, so it crosses the transition from
    /// a data block holding every element inline under one checksum to a paged
    /// one carrying a page-init bitmap and a checksum per page — including the
    /// partial last page, whose element count the closed form has to get right
    /// without walking the pages.
    #[test]
    fn fixed_array_len_matches_what_it_builds() {
        fn check(n: u64, chunk_bytes: u64, offset_size: u8, length_size: u8, has_filters: bool) {
            let chunks: Vec<WrittenChunk> = (0..n)
                .map(|i| WrittenChunk {
                    address: 0x1000 + i * 8,
                    compressed_size: 8,
                    filter_mask: 0,
                })
                .collect();
            let planned = fixed_array_len(
                &IndexSlots::dense(&chunks),
                chunk_bytes,
                offset_size,
                length_size,
                has_filters,
            );
            let built = build_fixed_array_at(
                &IndexSlots::dense(&chunks),
                chunk_bytes,
                offset_size,
                length_size,
                has_filters,
                0x10_0000,
            );
            assert_eq!(
                planned,
                built.len() as u64,
                "planned length must match the emitted array at n={n}, \
                 chunk_bytes={chunk_bytes}, offset_size={offset_size}, \
                 has_filters={has_filters}"
            );
        }

        for &(offset_size, length_size) in &[(8u8, 8u8), (4u8, 4u8)] {
            for &has_filters in &[false, true] {
                // Contiguous across the page boundary: the array is paged only
                // past `1 << FIXED_ARRAY_PAGE_BITS` elements.
                for n in 0..=1_100u64 {
                    check(n, 8, offset_size, length_size, has_filters);
                }
                // Several whole pages, and a count that leaves a partial one.
                for &n in &[4_096u64, 5_000, 100_000] {
                    check(n, 8, offset_size, length_size, has_filters);
                }
            }
        }

        // The filtered element record's compressed-size field is sized to a whole
        // raw chunk, and every element and page is sized from it.
        for &chunk_bytes in &CHUNK_BYTES {
            for &n in &[1u64, 1_024, 1_025, 5_000] {
                check(n, chunk_bytes, 8, 8, true);
            }
        }
    }

    /// `extensible_array_len` is the span the in-place editor reserves for an
    /// array before a byte of it exists, so it has to equal the length
    /// `build_extensible_array_at` goes on to emit. A reservation that came out
    /// short would place the next object on top of the array.
    ///
    /// The small counts are swept *contiguously* rather than at hand-picked
    /// boundaries: which blocks an element count allocates is decided twice over
    /// — once by `ea_compute_stats`, which this length comes from, and once by
    /// the builder's own body — and a contiguous sweep crosses every transition
    /// between those two walks without anyone having to work out where the
    /// transitions are. It covers the inline slots, all six direct data blocks,
    /// and the first on-disk super block. The larger counts then reach the deeper
    /// super blocks and, at 131,061, the first *paged* data block.
    #[test]
    fn extensible_array_len_matches_what_it_builds() {
        fn check(n: u64, chunk_bytes: u64, offset_size: u8, length_size: u8, has_filters: bool) {
            let chunks: Vec<WrittenChunk> = (0..n)
                .map(|i| WrittenChunk {
                    address: 0x1000 + i * 8,
                    compressed_size: 8,
                    filter_mask: 0,
                })
                .collect();
            let planned = extensible_array_len(
                &IndexSlots::dense(&chunks),
                chunk_bytes,
                offset_size,
                length_size,
                has_filters,
            );
            let built = build_extensible_array_at(
                &IndexSlots::dense(&chunks),
                chunk_bytes,
                offset_size,
                length_size,
                has_filters,
                0x10_0000,
            )
            .unwrap();
            assert_eq!(
                planned,
                built.len() as u64,
                "planned length must match the emitted array at n={n}, \
                 chunk_bytes={chunk_bytes}, offset_size={offset_size}, \
                 has_filters={has_filters}"
            );
        }

        for &(offset_size, length_size) in &[(8u8, 8u8), (4u8, 4u8)] {
            for &has_filters in &[false, true] {
                // Contiguous across the inline, direct-block and first
                // super-block ranges.
                for n in 0..=250u64 {
                    check(n, 8, offset_size, length_size, has_filters);
                }
                // The deeper super blocks, and the paged boundary: 131,060 is the
                // last element the unpaged super block 12 holds, 131,061 the first
                // that allocates a paged data block.
                for &n in &[300u64, 2_000, 50_000, 131_060, 131_061, 140_000] {
                    check(n, 8, offset_size, length_size, has_filters);
                }
            }
        }

        // A filtered element record carries the chunk's compressed size in a field
        // sized to a whole *raw* chunk, so the record width — and with it the
        // index block and every data block — changes with that size.
        for &chunk_bytes in &CHUNK_BYTES {
            for &n in &[1u64, 5, 244, 300, 2_000] {
                check(n, chunk_bytes, 8, 8, true);
            }
        }
        // Those four chunk sizes have to select four *different* field widths, or
        // the loop above is one fixture written four times. Asserted as
        // distinctness rather than as four literals: the rule is that the width
        // tracks the chunk size, not that it takes any particular value.
        //
        // Measured with **no chunks at all**, which is both the shape that used
        // to fabricate a 1-byte chunk and the proof that the width now comes
        // from the geometry: an empty array whose width still tracks
        // `chunk_bytes` cannot be reading it off a written chunk.
        let widths: Vec<usize> = CHUNK_BYTES
            .iter()
            .map(|&chunk_bytes| {
                super::ea_layout(SlotOccupancy::Dense(0), 0, chunk_bytes, 8, 8, true)
                    .encoding
                    .chunk_size_bytes
            })
            .collect();
        let mut distinct = widths.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            CHUNK_BYTES.len(),
            "each chunk size must select a different compressed-size field width, got {widths:?}"
        );
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
