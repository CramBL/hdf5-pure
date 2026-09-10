//! HDF5 filter implementations: deflate, shuffle, fletcher32, scale-offset,
//! LZF, and ZFP (the last two behind their own modules).

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use core::num::NonZeroU32;
// `format!` is only reached by the zfp-gated code paths below.
#[cfg(all(not(feature = "std"), feature = "zfp"))]
use alloc::format;

#[cfg(feature = "zfp")]
use crate::convert::TryToUsize;
use crate::error::FormatError;
#[cfg(feature = "zfp")]
use crate::filter_pipeline::FILTER_ZFP;
use crate::filter_pipeline::{
    FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_LZF, FILTER_SCALEOFFSET, FILTER_SHUFFLE,
    FilterPipeline,
};
use crate::scaleoffset::ScaleOffsetType;
#[cfg(feature = "zfp")]
use crate::zfp::ZfpElementType;

/// Context shared with filter pipeline operations.
///
/// Most filters (deflate, shuffle, fletcher32) need only element_size; ZFP
/// also needs chunk dimensions and scalar type, carried here so future
/// type-aware filters can look them up without changing the pipeline API.
#[derive(Debug, Clone, Copy)]
pub struct ChunkContext<'a> {
    /// Chunk dimensions in elements (one per dataset rank).
    pub chunk_dims: &'a [u64],
    /// Size of one element in bytes (for shuffle's interleave width), proven
    /// non-zero: the chunk splitter clamps a row copy to an element boundary
    /// with `% element_size`, and the byte-oriented filters divide a buffer
    /// length by it. Carrying the proof here is what lets those sites skip a
    /// check of their own.
    pub element_size: NonZeroU32,
    /// Scalar type, required for type-aware filters like ZFP. `None` means
    /// the caller does not know or does not need it; type-aware filters
    /// will return an error.
    pub element_type: Option<ZfpElementTypeWhenEnabled>,
    /// Datatype facts the scale-offset encoder needs (class/sign/order).
    /// `None` for callers that don't have a `Datatype` or whose type isn't a
    /// scale-offset-compatible scalar; scale-offset writes then error.
    pub scale_offset_type: Option<ScaleOffsetType>,
}

/// Dummy wrapper so ChunkContext's type stays stable whether or not the
/// `zfp` feature is on. With `zfp` on this aliases `zfp::ZfpElementType`.
#[cfg(feature = "zfp")]
pub type ZfpElementTypeWhenEnabled = ZfpElementType;
#[cfg(not(feature = "zfp"))]
pub type ZfpElementTypeWhenEnabled = core::convert::Infallible;

impl<'a> ChunkContext<'a> {
    /// Lightweight constructor for callers that don't need ZFP support — the
    /// element_type is left `None`, so any ZFP filter in the pipeline will
    /// error out. `element_size` must still be valid.
    ///
    /// Currently only used by tests (read/write paths build the context via
    /// [`ChunkContext::from_datatype`]); gated so it is not shipped as dead code.
    ///
    /// # Panics
    ///
    /// If `element_size` is zero. Every caller passes a literal width, and this
    /// is test-only; the production constructor
    /// [`from_datatype`](Self::from_datatype) returns an error instead.
    #[cfg(test)]
    pub fn basic(chunk_dims: &'a [u64], element_size: u32) -> Self {
        Self {
            chunk_dims,
            element_size: NonZeroU32::new(element_size).expect("a test's element size is non-zero"),
            element_type: None,
            scale_offset_type: None,
        }
    }

    /// Build a full context from a dataset's `Datatype`: derives
    /// `element_size` from [`Datatype::element_size`] and `element_type` from
    /// [`zfp_element_type_from_datatype`]. This is the preferred
    /// constructor for read/write paths where a `Datatype` is in scope,
    /// so the two fields can't drift out of sync.
    ///
    /// # Errors
    ///
    /// [`FormatError::ZeroSizedDatatype`] if the type occupies zero bytes per
    /// element. Refusing here is what makes the field's guarantee true; every
    /// consumer of the context then takes the size without re-checking it.
    ///
    /// [`Datatype::element_size`]: crate::datatype::Datatype::element_size
    pub fn from_datatype(
        chunk_dims: &'a [u64],
        dt: &crate::datatype::Datatype,
    ) -> Result<Self, FormatError> {
        Ok(Self {
            chunk_dims,
            element_size: dt.element_size()?,
            element_type: zfp_element_type_from_datatype(dt),
            scale_offset_type: crate::scaleoffset::scale_offset_type_from_datatype(dt),
        })
    }
}

/// Map an HDF5 `Datatype` to the matching ZFP scalar type, if it's one of the
/// supported codec widths. Returns `None` for types outside f32/f64/i32/i64.
#[cfg(feature = "zfp")]
pub fn zfp_element_type_from_datatype(
    dt: &crate::datatype::Datatype,
) -> Option<ZfpElementTypeWhenEnabled> {
    use crate::datatype::Datatype;
    match dt {
        Datatype::FloatingPoint { size: 4, .. } => Some(ZfpElementType::F32),
        Datatype::FloatingPoint { size: 8, .. } => Some(ZfpElementType::F64),
        Datatype::FixedPoint {
            size: 4,
            signed: true,
            ..
        } => Some(ZfpElementType::I32),
        Datatype::FixedPoint {
            size: 8,
            signed: true,
            ..
        } => Some(ZfpElementType::I64),
        _ => None,
    }
}

#[cfg(not(feature = "zfp"))]
pub fn zfp_element_type_from_datatype(
    _: &crate::datatype::Datatype,
) -> Option<ZfpElementTypeWhenEnabled> {
    None
}

/// Reusable working state for the filters that carry any.
///
/// Every filter here is a pure function of its input except deflate, whose
/// compressor holds ~300 KiB of hash tables and dictionary and whose decompressor
/// holds ~42 KiB of Huffman state. All of it is *scratch*: a reset returns either
/// to exactly its initial condition, so one of each serves a whole chunk loop and
/// produces byte-identical output to a fresh one.
///
/// Building them per chunk is what a chunked write and a chunked read used to do,
/// and it dominated both: 615 MiB of allocation to write an 8 MiB deflated
/// dataset and 85 MiB to read it back, against 8 MiB of data (issue #228). So a
/// caller with a loop keeps one of these across it and calls the `*_with` entry
/// points; a caller with a single chunk uses `compress_chunk` or
/// [`decompress_chunk`], which make one for the call.
///
/// Each half is built on first use, so a pipeline with no deflate stage — or a
/// build without the feature — carries nothing.
#[derive(Default)]
pub struct FilterScratch {
    #[cfg(feature = "deflate")]
    encoder: Option<(u32, flate2::write::ZlibEncoder<Vec<u8>>)>,
    #[cfg(feature = "deflate")]
    decoder: Option<flate2::Decompress>,
}

impl FilterScratch {
    /// An empty scratch. Nothing is allocated until a filter needs it.
    pub fn new() -> Self {
        Self::default()
    }

    /// The reusable encoder, built or re-levelled as needed.
    ///
    /// A level change rebuilds rather than resets: `flate2::Compress` can change
    /// level in place, but only by flushing a stream that is mid-chunk here, and a
    /// pipeline does not change level between its own chunks anyway — this is for
    /// a scratch that outlives one dataset and meets another.
    #[cfg(feature = "deflate")]
    fn zlib_encoder(&mut self, level: u32) -> &mut flate2::write::ZlibEncoder<Vec<u8>> {
        let stale = self.encoder.as_ref().is_none_or(|(have, _)| *have != level);
        if stale {
            self.encoder = Some((
                level,
                flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(level)),
            ));
        }
        &mut self.encoder.as_mut().expect("built directly above").1
    }

    /// The reusable decoder, reset to the start of a zlib stream.
    ///
    /// Resetting on the way *in* rather than the way out is what makes a decode
    /// that failed halfway leave nothing behind for the next chunk: the failing
    /// call returns early by definition, so any cleanup it was supposed to do on
    /// the way out is the cleanup that does not happen.
    #[cfg(feature = "deflate")]
    fn zlib_decoder(&mut self) -> &mut flate2::Decompress {
        match &mut self.decoder {
            Some(d) => d.reset(true),
            slot => *slot = Some(flate2::Decompress::new(true)),
        }
        self.decoder.as_mut().expect("built directly above")
    }
}

/// Apply a filter pipeline to decompress a chunk, making the working state for
/// the call.
///
/// A caller decoding chunk after chunk should keep a [`FilterScratch`] and use
/// [`decompress_chunk_with`] instead: this one builds and drops a zlib decoder
/// per call.
pub fn decompress_chunk(
    compressed: &[u8],
    pipeline: &FilterPipeline,
    ctx: ChunkContext<'_>,
    filter_mask: u32,
) -> Result<Vec<u8>, FormatError> {
    decompress_chunk_with(
        &mut FilterScratch::new(),
        compressed,
        pipeline,
        ctx,
        filter_mask,
    )
}

/// Apply a filter pipeline to decompress a chunk.
/// Filters are applied in REVERSE order for decompression.
pub fn decompress_chunk_with(
    scratch: &mut FilterScratch,
    compressed: &[u8],
    pipeline: &FilterPipeline,
    ctx: ChunkContext<'_>,
    filter_mask: u32,
) -> Result<Vec<u8>, FormatError> {
    // Expected size of the fully decoded chunk. Every chunk, even one straddling
    // a dataset edge, is stored at full chunk size, so this is the exact decoded
    // length. Used to bound deflate output (decompression-bomb guard) and to
    // reject a chunk that decodes to the wrong size.
    let expected = expected_chunk_len(&ctx);

    let mut owned: Option<Vec<u8>> = None;
    // Filters are listed in application (forward) order; decoding reverses them.
    // `i` is the filter's forward index, which is also its bit position in
    // `filter_mask` (HDF5 H5Z pipeline numbering): bit `i` set means filter `i`
    // was skipped for THIS chunk and must NOT be reversed. Treating any non-zero
    // mask as "return raw" (the prior behaviour) corrupts chunks in a multi-filter
    // pipeline where only some filters were skipped (e.g. shuffle+gzip on an
    // incompressible chunk, which is stored shuffled but not deflated).
    for (i, filter) in pipeline.filters.iter().enumerate().rev() {
        if i < 32 && (filter_mask >> i) & 1 == 1 {
            continue;
        }
        let input: &[u8] = owned.as_deref().unwrap_or(compressed);
        let next = match filter.filter_id {
            FILTER_SHUFFLE => shuffle_decompress(input, ctx.element_size.get() as usize)?,
            FILTER_DEFLATE => deflate_decompress(
                scratch,
                input,
                inner_output_cap(expected, pipeline, filter_mask, i),
            )?,
            FILTER_LZF => {
                crate::lzf::decompress(input, inner_output_cap(expected, pipeline, filter_mask, i))?
            }
            FILTER_FLETCHER32 => fletcher32_verify(input)?,
            FILTER_SCALEOFFSET => crate::scaleoffset::decompress(
                input,
                filter,
                inner_output_cap(expected, pipeline, filter_mask, i),
            )?,
            #[cfg(feature = "zfp")]
            FILTER_ZFP => zfp_decompress(input, filter, &ctx)?,
            other => return Err(FormatError::UnsupportedFilter(other)),
        };
        owned = Some(next);
    }
    let result = owned.unwrap_or_else(|| compressed.to_vec());

    // A valid chunk always decodes to exactly the full chunk size. A mismatch
    // means a corrupt or hostile filter stream; erroring here prevents silently
    // zero-filling (when short) or dropping (when long) data during chunk
    // assembly, which copies only the in-range overlap.
    if let Some(expected) = expected {
        if result.len() != expected {
            return Err(FormatError::DataSizeMismatch {
                expected,
                actual: result.len(),
            });
        }
    }
    Ok(result)
}

/// Expected byte length of a fully decoded chunk: product of the chunk element
/// dimensions times the element size. Returns `None` when the product can't be
/// represented (treated as "unknown", so the size-dependent guards are skipped
/// rather than misfiring) or is zero.
fn expected_chunk_len(ctx: &ChunkContext<'_>) -> Option<usize> {
    let elems = ctx
        .chunk_dims
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))?;
    let bytes = elems.checked_mul(u64::from(ctx.element_size.get()))?;
    usize::try_from(bytes).ok().filter(|&n| n != 0)
}

/// Upper bound on a filter's forward (compress) output for an input of
/// `in_size` bytes. Used to bound a deflate stage's legitimate decoded output:
/// on decode, deflate is reversed BEFORE the lower-forward-index filters that
/// ran before it on the write path, so its output equals the chunk size pushed
/// forward through those inner filters. Only an upper bound is needed — the
/// exact chunk-size check after the whole pipeline still rejects wrong output —
/// so this is the decompression-bomb memory guard, not a correctness gate.
fn filter_max_forward_output(filter_id: u16, in_size: usize) -> usize {
    match filter_id {
        // Fletcher32 appends a 4-byte checksum.
        FILTER_FLETCHER32 => in_size.saturating_add(4),
        // A conforming LZF encoder may emit every byte as its own literal run
        // (control byte + literal), so a stream is at most twice its decoded
        // size; matches are denser. Efficient encoders stay near in_size/32
        // overhead, but the bound must admit any conforming stream.
        FILTER_LZF => in_size.saturating_mul(2),
        // Scale-offset prepends a fixed header and, when the data does not pack
        // smaller, stores it verbatim after that header.
        FILTER_SCALEOFFSET => in_size.saturating_add(crate::scaleoffset::HEADER_LEN),
        // Deflate can slightly expand incompressible input (zlib "stored" blocks
        // plus framing); bound it well above zlib's worst case.
        FILTER_DEFLATE => in_size.saturating_add(in_size / 16).saturating_add(64),
        // Shuffle is size-preserving; fixed-rate ZFP never exceeds the native
        // element width. An unknown filter makes the read fail when it is reached
        // after deflate regardless, so leaving the size unchanged is fine.
        _ => in_size,
    }
}

/// Largest number of bytes a conforming deflate stream can decode to per byte of
/// input. Deflate's densest encoding is a 258-byte length/distance match written
/// as two Huffman symbols, and no Huffman symbol is shorter than one bit, so a
/// match costs at least two bits: `258 / (2 / 8) = 1032`. Block headers only make
/// a real stream less dense, so the ratio bounds the stream as a whole.
const MAX_DEFLATE_EXPANSION: usize = 1032;

/// How many bytes to reserve up front for one decode stage's output.
///
/// `cap` is that stage's output bound, derived from the chunk geometry the
/// *file* declares — which an untrusted file controls. Reserving it outright
/// turns a small file claiming an enormous chunk into an allocation abort,
/// before a single byte of the stream has been looked at. The stream itself is
/// the evidence that the claim is plausible: a conforming stream of `in_size`
/// bytes decodes to at most `in_size * max_expansion`, so reserving no more than
/// that bounds a hostile file by the bytes it actually had to put on disk.
///
/// It costs a legitimate chunk nothing. The true decoded size is under `cap`
/// (the pipeline enforces that) and under the format's expansion bound, so it is
/// under the smaller of the two as well: the reservation still holds the whole
/// output without a reallocation.
///
/// This is a reservation hint, not a limit — the decoder grows past it if a
/// stream needs it to, and `cap` remains the enforced bound. `None` (chunk size
/// unknown) reserves nothing, there being no claim to be exact about.
pub(crate) fn decode_reservation(
    cap: Option<usize>,
    in_size: usize,
    max_expansion: usize,
) -> usize {
    cap.map_or(0, |cap| cap.min(in_size.saturating_mul(max_expansion)))
}

/// Upper bound for a byte-compressor stage's decoded output: the final chunk size
/// (`expected`) pushed forward through every surviving lower-forward-index
/// filter. `None` (size unknown) leaves the byte-compressor stage (deflate,
/// LZF) uncapped. A masked filter did not run on the write path, so it does
/// not change the intermediate size.
fn inner_output_cap(
    expected: Option<usize>,
    pipeline: &FilterPipeline,
    filter_mask: u32,
    filter_index: usize,
) -> Option<usize> {
    let mut size = expected?;
    for (j, f) in pipeline.filters[..filter_index].iter().enumerate() {
        if j < 32 && (filter_mask >> j) & 1 == 1 {
            continue;
        }
        size = filter_max_forward_output(f.filter_id, size);
    }
    Some(size)
}

/// Apply a filter pipeline to compress a chunk, making the working state for the
/// call.
///
/// Test-only: every place this crate compresses a chunk does so in a loop, and
/// each of those keeps one [`FilterScratch`] across it. Kept because a test that
/// wants one chunk should not have to say so in three lines.
///
/// A caller compressing chunk after chunk should keep a [`FilterScratch`] and use
/// [`compress_chunk_with`] instead: this one builds and drops a zlib encoder per
/// call, which is ~300 KiB of hash tables per chunk.
#[cfg(test)]
pub fn compress_chunk(
    data: &[u8],
    pipeline: &FilterPipeline,
    ctx: ChunkContext<'_>,
) -> Result<Vec<u8>, FormatError> {
    compress_chunk_with(&mut FilterScratch::new(), data, pipeline, ctx)
}

/// Apply a filter pipeline to compress a chunk.
/// Filters are applied in FORWARD order for compression.
pub fn compress_chunk_with(
    scratch: &mut FilterScratch,
    data: &[u8],
    pipeline: &FilterPipeline,
    ctx: ChunkContext<'_>,
) -> Result<Vec<u8>, FormatError> {
    let mut owned: Option<Vec<u8>> = None;
    for filter in &pipeline.filters {
        let input: &[u8] = owned.as_deref().unwrap_or(data);
        let next = match filter.filter_id {
            FILTER_SHUFFLE => shuffle_compress(input, ctx.element_size.get() as usize)?,
            FILTER_DEFLATE => {
                let level = filter.client_data.first().copied().unwrap_or(6);
                deflate_compress(scratch, input, level)?
            }
            FILTER_LZF => crate::lzf::compress(input),
            FILTER_FLETCHER32 => fletcher32_append(input)?,
            FILTER_SCALEOFFSET => crate::scaleoffset::compress(input, filter)?,
            #[cfg(feature = "zfp")]
            FILTER_ZFP => zfp_compress(input, filter, &ctx)?,
            other => return Err(FormatError::UnsupportedFilter(other)),
        };
        owned = Some(next);
    }
    Ok(owned.unwrap_or_else(|| data.to_vec()))
}

#[cfg(feature = "zfp")]
fn zfp_rate(filter: &crate::filter_pipeline::FilterDescription) -> Result<f64, FormatError> {
    crate::zfp::zfp_rate_from_cd_values(&filter.client_data)
        .ok_or_else(|| FormatError::FilterError("ZFP: invalid or non-rate cd_values".into()))
}

#[cfg(feature = "zfp")]
fn zfp_element_type(ctx: &ChunkContext<'_>) -> Result<ZfpElementType, FormatError> {
    ctx.element_type.ok_or_else(|| {
        FormatError::FilterError(
            "ZFP: element_type missing from ChunkContext (caller must set it)".into(),
        )
    })
}

/// Copy chunk dims into a stack buffer and return a slice of the valid
/// prefix. ZFP's rank bound is 4, so a heap Vec is unnecessary per chunk.
#[cfg(feature = "zfp")]
fn zfp_dims_on_stack(ctx: &ChunkContext<'_>) -> Result<([usize; 4], usize), FormatError> {
    let rank = ctx.chunk_dims.len();
    if rank == 0 || rank > 4 {
        return Err(FormatError::FilterError(format!(
            "ZFP: chunk rank must be 1..=4, got {rank}",
        )));
    }
    let mut buf = [0usize; 4];
    for (slot, &d) in buf.iter_mut().zip(ctx.chunk_dims.iter()) {
        *slot = d.to_usize()?;
    }
    Ok((buf, rank))
}

#[cfg(feature = "zfp")]
fn zfp_compress(
    data: &[u8],
    filter: &crate::filter_pipeline::FilterDescription,
    ctx: &ChunkContext<'_>,
) -> Result<Vec<u8>, FormatError> {
    let rate = zfp_rate(filter)?;
    let elem_ty = zfp_element_type(ctx)?;
    let (dims_buf, rank) = zfp_dims_on_stack(ctx)?;
    crate::zfp::compress(data, &dims_buf[..rank], rate, elem_ty)
}

#[cfg(feature = "zfp")]
fn zfp_decompress(
    data: &[u8],
    filter: &crate::filter_pipeline::FilterDescription,
    ctx: &ChunkContext<'_>,
) -> Result<Vec<u8>, FormatError> {
    let rate = zfp_rate(filter)?;
    let elem_ty = zfp_element_type(ctx)?;
    let (dims_buf, rank) = zfp_dims_on_stack(ctx)?;
    crate::zfp::decompress(data, &dims_buf[..rank], rate, elem_ty)
}

/// A deflate stream this decoder rejected, reported the way every other filter
/// here reports one: `FilterError`, tagged with the filter's name.
#[cfg(feature = "deflate")]
fn deflate_corrupt(reason: &str) -> FormatError {
    FormatError::FilterError(format!("deflate: {reason}"))
}

/// Decompress zlib-compressed data.
///
/// `max_output`, when known, bounds the decompressed size: a deflate stage in a
/// chunk pipeline never expands beyond the chunk's expected byte size, so a
/// stream that inflates past it signals a decompression bomb and is rejected
/// instead of being allowed to allocate unbounded memory (OOM).
///
/// A failure is a [`FormatError::FilterError`], the same variant every other
/// filter in this pipeline reports a bad stream with, so a caller can match
/// "this chunk did not decode" once rather than per compressor.
/// How much room to add when a decode fills its buffer and no decoded size was
/// declared. One page: large enough that an unbounded decode does not crawl,
/// small enough that overshooting the final block wastes little.
#[cfg(feature = "deflate")]
const DEFLATE_GROWTH_FLOOR: usize = 4096;

/// Decompress zlib data, reusing `scratch`'s decoder.
///
/// Driven through [`flate2::Decompress`] rather than a `ZlibDecoder` because
/// `Decompress::reset` is the only reset in flate2 that reuses the inflate state
/// — every `ZlibDecoder::reset` replaces it with a fresh one, and that
/// allocation per chunk is what this exists to stop paying (issue #228).
///
/// The bomb guard the `Read` path wrote as `take(limit + 1)` is an explicit
/// ceiling on the output buffer here: the buffer's *length* never passes
/// `limit + 1`, so a stream that would inflate further is refused on the same
/// evidence. Its *capacity* can reach twice that, since the growth below is
/// `Vec::reserve` and amortized — which is what `read_to_end` did too, so the
/// memory a bomb can command is unchanged.
///
/// This path is stricter than the one it replaced in one respect. `read_to_end`
/// stops at the first `Ok(0)`, and `flate2::read::ZlibDecoder` returns `Ok(0)`
/// when its input runs out mid-stream, so a chunk that carried all its data but
/// never reached `StreamEnd` used to decode and pass with its adler32 never
/// checked. Here that is a `truncated` refusal and the chunk does not decode.
/// `an_unterminated_stream_is_refused_where_the_read_path_accepted_it` pins both
/// halves.
#[cfg(feature = "deflate")]
fn deflate_decompress(
    scratch: &mut FilterScratch,
    data: &[u8],
    max_output: Option<usize>,
) -> Result<Vec<u8>, FormatError> {
    use flate2::{FlushDecompress, Status};

    let bomb = || {
        deflate_corrupt(&format!(
            "output exceeds expected chunk size of {} bytes (possible decompression bomb)",
            max_output.unwrap_or(0)
        ))
    };
    let truncated = || deflate_corrupt("stream ended before the chunk was complete");

    // The decoded size is known a priori (the chunk's expected byte size), so
    // reserve it up front instead of growing through ~log2(N) doublings — but
    // only as far as this stream could possibly justify, so a declared size no
    // stream backs cannot drive the allocation on its own.
    let reservation = decode_reservation(max_output, data.len(), MAX_DEFLATE_EXPANSION);
    // One past the limit, so an over-long stream is *detected* rather than
    // silently truncated into a plausible-looking chunk.
    let ceiling = max_output.map(|limit| limit.saturating_add(1));

    let decoder = scratch.zlib_decoder();
    let mut out = Vec::with_capacity(match ceiling {
        Some(cap) => reservation.min(cap),
        None => reservation,
    });

    loop {
        let consumed = usize::try_from(decoder.total_in()).unwrap_or(usize::MAX);
        let input = data.get(consumed..).unwrap_or(&[]);
        let before_in = decoder.total_in();
        let before_out = out.len();

        // Writes into the spare capacity between `len` and `capacity` and never
        // reallocates: growing the buffer is this loop's job.
        let status = decoder
            .decompress_vec(input, &mut out, FlushDecompress::None)
            .map_err(|e| deflate_corrupt(&e.to_string()))?;

        if status == Status::StreamEnd {
            break;
        }

        if out.len() == out.capacity() {
            // Out of room. Either the chunk is bigger than it declared, or the
            // buffer simply needs to grow.
            if ceiling.is_some_and(|cap| out.len() >= cap) {
                return Err(bomb());
            }
            let want = out.capacity().max(DEFLATE_GROWTH_FLOOR);
            out.reserve(match ceiling {
                Some(cap) => want.min(cap - out.len()),
                None => want,
            });
        } else if decoder.total_in() == before_in && out.len() == before_out {
            // Room to write, nothing written, and no input consumed: the stream
            // cannot finish. `Status::Ok` here means it ran out of input.
            return Err(truncated());
        }
    }

    if max_output.is_some_and(|limit| out.len() > limit) {
        return Err(bomb());
    }
    Ok(out)
}

#[cfg(not(feature = "deflate"))]
fn deflate_decompress(
    _scratch: &mut FilterScratch,
    _data: &[u8],
    _max_output: Option<usize>,
) -> Result<Vec<u8>, FormatError> {
    Err(FormatError::UnsupportedFilter(FILTER_DEFLATE))
}

/// Compress data with zlib, reusing `scratch`'s encoder.
///
/// The reuse is the point: a zlib encoder holds ~300 KiB of hash tables and
/// dictionary, all of it scratch that `flate2::Compress::reset` returns to its
/// initial state, and building one per chunk cost 615 MiB of allocation to write
/// an 8 MiB dataset (issue #228).
///
/// That reset is also what keeps the output identical to a fresh encoder's, byte
/// for byte — which this crate needs rather than merely likes, since the files it
/// writes are compared against the reference C library's.
#[cfg(feature = "deflate")]
fn deflate_compress(
    scratch: &mut FilterScratch,
    data: &[u8],
    level: u32,
) -> Result<Vec<u8>, FormatError> {
    use std::io::Write;
    let encoder = scratch.zlib_encoder(level);
    // Finishes the stream into the buffer the encoder is holding and hands that
    // buffer back, leaving a reset encoder wrapped around the empty one.
    let finished = encoder
        .write_all(data)
        .and_then(|()| encoder.reset(Vec::new()))
        .map_err(|e| FormatError::FilterError(format!("deflate: {e}")));
    if finished.is_err() {
        // Either step failing leaves the encoder mid-stream, and the next chunk
        // would append to this one's unfinished output. The decoder avoids the
        // same trap by resetting on acquisition; the encoder cannot, since its
        // reset is what produces the bytes, so it is discarded instead.
        //
        // Unreachable while the sink is a `Vec`, whose writes do not fail. It is
        // handled rather than asserted because what makes it unreachable is the
        // sink type, which is not this function's to guarantee.
        scratch.encoder = None;
    }
    finished
}

#[cfg(not(feature = "deflate"))]
fn deflate_compress(
    _scratch: &mut FilterScratch,
    _data: &[u8],
    _level: u32,
) -> Result<Vec<u8>, FormatError> {
    Err(FormatError::UnsupportedFilter(FILTER_DEFLATE))
}

/// Unshuffle one element width `N` (const-generic, so the inner byte loop is
/// unrolled): gather byte `j` of element `i` from plane `j` and write the `N`
/// reconstructed bytes of the element as one contiguous store.
fn unshuffle_n<const N: usize>(data: &[u8], result: &mut [u8], num_elements: usize) {
    for (i, out) in result.as_chunks_mut::<N>().0.iter_mut().enumerate() {
        let mut elem = [0u8; N];
        for (j, b) in elem.iter_mut().enumerate() {
            *b = data[j * num_elements + i];
        }
        *out = elem;
    }
}

/// Shuffle one element width `N` (const-generic): read the `N` contiguous bytes
/// of element `i` and scatter byte `j` into plane `j`.
fn shuffle_n<const N: usize>(data: &[u8], result: &mut [u8], num_elements: usize) {
    for (i, elem) in data.as_chunks::<N>().0.iter().enumerate() {
        for (j, &b) in elem.iter().enumerate() {
            result[j * num_elements + i] = b;
        }
    }
}

/// Unshuffle (decompress direction): reconstruct interleaved element bytes.
/// On disk: all byte-0s of each element together, then all byte-1s, etc.
/// Output: elements in natural order.
fn shuffle_decompress(data: &[u8], element_size: usize) -> Result<Vec<u8>, FormatError> {
    if element_size <= 1 {
        return Ok(data.to_vec());
    }
    if !data.len().is_multiple_of(element_size) {
        return Err(FormatError::FilterError(
            "shuffle: data length not a multiple of element size".into(),
        ));
    }
    let num_elements = data.len() / element_size;
    let mut result = vec![0u8; data.len()];

    // Specialize the common scalar widths so the inner loop unrolls and each
    // element is written as one contiguous store; fall back to the generic loop
    // for unusual widths (compound members, wide types).
    match element_size {
        2 => unshuffle_n::<2>(data, &mut result, num_elements),
        4 => unshuffle_n::<4>(data, &mut result, num_elements),
        8 => unshuffle_n::<8>(data, &mut result, num_elements),
        16 => unshuffle_n::<16>(data, &mut result, num_elements),
        _ => {
            for i in 0..num_elements {
                for j in 0..element_size {
                    result[i * element_size + j] = data[j * num_elements + i];
                }
            }
        }
    }

    Ok(result)
}

/// Shuffle (compress direction): group bytes by position within each element.
fn shuffle_compress(data: &[u8], element_size: usize) -> Result<Vec<u8>, FormatError> {
    if element_size <= 1 {
        return Ok(data.to_vec());
    }
    if !data.len().is_multiple_of(element_size) {
        return Err(FormatError::FilterError(
            "shuffle: data length not a multiple of element size".into(),
        ));
    }
    let num_elements = data.len() / element_size;
    let mut result = vec![0u8; data.len()];

    match element_size {
        2 => shuffle_n::<2>(data, &mut result, num_elements),
        4 => shuffle_n::<4>(data, &mut result, num_elements),
        8 => shuffle_n::<8>(data, &mut result, num_elements),
        16 => shuffle_n::<16>(data, &mut result, num_elements),
        _ => {
            for i in 0..num_elements {
                for j in 0..element_size {
                    result[j * num_elements + i] = data[i * element_size + j];
                }
            }
        }
    }

    Ok(result)
}

/// Computes the HDF5 Fletcher32 checksum of `data`.
///
/// HDF5 uses a modified Fletcher32 algorithm that operates on 16-bit words.
/// Blocks of up to 360 words are accumulated before folding each 32-bit sum into
/// 16 bits, matching `H5_checksum_fletcher32` (`H5checksum.c`, HDF5 2.2.0).
fn fletcher32_compute(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0;
    let mut sum2: u32 = 0;

    let mut words = data.len() / 2;
    let mut offset = 0;

    while words > 0 {
        let tlen = core::cmp::min(words, 360);
        words -= tlen;
        for _ in 0..tlen {
            let val = ((data[offset] as u32) << 8) | (data[offset + 1] as u32);
            offset += 2;
            sum1 += val;
            sum2 += sum1;
        }
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    if data.len() % 2 != 0 {
        sum1 += (data[offset] as u32) << 8;
        sum2 += sum1;
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);

    (sum2 << 16) | sum1
}

/// Verify Fletcher32 checksum and strip it from the data.
/// The last 4 bytes are the stored checksum.
fn fletcher32_verify(data: &[u8]) -> Result<Vec<u8>, FormatError> {
    if data.len() < 4 {
        return Err(FormatError::FilterError(
            "fletcher32: data too short for checksum".into(),
        ));
    }
    let payload = &data[..data.len() - 4];
    let stored = u32::from_le_bytes([
        data[data.len() - 4],
        data[data.len() - 3],
        data[data.len() - 2],
        data[data.len() - 1],
    ]);
    let computed = fletcher32_compute(payload);
    if stored != computed {
        return Err(FormatError::Fletcher32Mismatch {
            expected: stored,
            computed,
        });
    }
    Ok(payload.to_vec())
}

/// Append Fletcher32 checksum to data.
fn fletcher32_append(data: &[u8]) -> Result<Vec<u8>, FormatError> {
    let checksum = fletcher32_compute(data);
    let mut result = data.to_vec();
    result.extend_from_slice(&checksum.to_le_bytes());
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter_pipeline::FilterDescription;

    /// The context is the single place a `Datatype` becomes an element width for
    /// the chunk splitter and the byte-oriented filters, so it is where a
    /// degenerate type has to be turned away. Everything downstream then takes
    /// the width without a check of its own.
    #[test]
    fn a_context_cannot_be_built_from_a_zero_width_datatype() {
        let degenerate = crate::datatype::Datatype::Array {
            base_type: Box::new(crate::datatype::Datatype::FixedPoint {
                size: 4,
                byte_order: crate::datatype::DatatypeByteOrder::LittleEndian,
                signed: true,
                bit_offset: 0,
                bit_precision: 32,
            }),
            dimensions: vec![0],
        };
        assert_eq!(
            ChunkContext::from_datatype(&[4], &degenerate).unwrap_err(),
            FormatError::ZeroSizedDatatype { class: 10 }
        );

        let ordinary = crate::datatype::Datatype::FixedPoint {
            size: 4,
            byte_order: crate::datatype::DatatypeByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        };
        let ctx = ChunkContext::from_datatype(&[4], &ordinary).unwrap();
        assert_eq!(ctx.element_size.get(), 4);
    }

    // --- Deflate tests ---

    #[test]
    #[cfg(feature = "deflate")]
    fn deflate_compress_decompress_roundtrip() {
        let data: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
        let compressed = deflate_compress(&mut FilterScratch::new(), &data, 6).unwrap();
        let decompressed =
            deflate_decompress(&mut FilterScratch::new(), &compressed, None).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn deflate_decompress_python_zlib() {
        // Data compressed with Python: zlib.compress(bytes(range(10)), 6)
        // python3 -c "import zlib; print(list(zlib.compress(bytes(range(10)), 6)))"
        // = [120, 156, 99, 96, 100, 98, 102, 97, 101, 99, 231, 224, 4, 0, 1, 123, 0, 170]
        let compressed: Vec<u8> = vec![
            120, 156, 99, 96, 100, 98, 102, 97, 101, 99, 231, 224, 4, 0, 0, 175, 0, 46,
        ];
        let decompressed =
            deflate_decompress(&mut FilterScratch::new(), &compressed, None).unwrap();
        assert_eq!(decompressed, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn deflate_compress_verifiable() {
        // Compress data and verify it decompresses correctly
        let data = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let compressed = deflate_compress(&mut FilterScratch::new(), &data, 6).unwrap();
        assert!(!compressed.is_empty());
        let decompressed =
            deflate_decompress(&mut FilterScratch::new(), &compressed, None).unwrap();
        assert_eq!(decompressed, data);
    }

    // --- Shuffle tests ---

    #[test]
    fn shuffle_roundtrip_f64() {
        // 4 f64 values = 32 bytes, element_size=8
        let data: Vec<u8> = (0..32).collect();
        let shuffled = shuffle_compress(&data, 8).unwrap();
        let unshuffled = shuffle_decompress(&shuffled, 8).unwrap();
        assert_eq!(unshuffled, data);
    }

    #[test]
    fn shuffle_roundtrip_i32() {
        // 8 i32 values = 32 bytes, element_size=4
        let data: Vec<u8> = (0..32).collect();
        let shuffled = shuffle_compress(&data, 4).unwrap();
        let unshuffled = shuffle_decompress(&shuffled, 4).unwrap();
        assert_eq!(unshuffled, data);
    }

    #[test]
    fn shuffle_roundtrip_all_widths() {
        // Every specialized width (2/4/8/16) plus generic fallbacks (3, 6, 7).
        for &es in &[2usize, 3, 4, 6, 7, 8, 16] {
            let data: Vec<u8> = (0..(es * 50)).map(|i| (i * 31 % 256) as u8).collect();
            let shuffled = shuffle_compress(&data, es).unwrap();
            assert_eq!(shuffled.len(), data.len(), "es={es}");
            let back = shuffle_decompress(&shuffled, es).unwrap();
            assert_eq!(back, data, "shuffle roundtrip failed for element_size {es}");
        }
    }

    #[test]
    fn shuffle_specialized_matches_generic() {
        // The const-generic specialization must produce byte-identical output to
        // the plain transpose for the same width.
        fn generic_shuffle(data: &[u8], es: usize) -> Vec<u8> {
            let ne = data.len() / es;
            let mut out = vec![0u8; data.len()];
            for i in 0..ne {
                for j in 0..es {
                    out[j * ne + i] = data[i * es + j];
                }
            }
            out
        }
        for &es in &[2usize, 4, 8, 16] {
            let data: Vec<u8> = (0..(es * 37)).map(|i| (i * 17 + 3) as u8).collect();
            assert_eq!(
                shuffle_compress(&data, es).unwrap(),
                generic_shuffle(&data, es)
            );
        }
    }

    #[test]
    fn shuffle_known_pattern() {
        // 2 elements of size 4: [A0 A1 A2 A3 B0 B1 B2 B3]
        // After shuffle: [A0 B0 A1 B1 A2 B2 A3 B3]
        let data = vec![0xA0, 0xA1, 0xA2, 0xA3, 0xB0, 0xB1, 0xB2, 0xB3];
        let shuffled = shuffle_compress(&data, 4).unwrap();
        assert_eq!(
            shuffled,
            vec![0xA0, 0xB0, 0xA1, 0xB1, 0xA2, 0xB2, 0xA3, 0xB3]
        );
    }

    // --- Fletcher32 tests ---

    #[test]
    fn fletcher32_roundtrip() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let with_checksum = fletcher32_append(&data).unwrap();
        assert_eq!(with_checksum.len(), data.len() + 4);
        let verified = fletcher32_verify(&with_checksum).unwrap();
        assert_eq!(verified, data);
    }

    #[test]
    fn fletcher32_known_checksum() {
        // Verify checksum is deterministic
        let data = vec![0u8; 16];
        let with_checksum = fletcher32_append(&data).unwrap();
        let checksum = u32::from_le_bytes([
            with_checksum[16],
            with_checksum[17],
            with_checksum[18],
            with_checksum[19],
        ]);
        // All zeros -> sum1=0, sum2=0 -> checksum=0
        assert_eq!(checksum, 0);

        // Non-zero data
        let data2 = vec![1u8, 0, 0, 0];
        let with_checksum2 = fletcher32_append(&data2).unwrap();
        let verified = fletcher32_verify(&with_checksum2).unwrap();
        assert_eq!(verified, data2);
    }

    #[test]
    fn fletcher32_mismatch_detected() {
        let data = vec![1u8, 2, 3, 4];
        let mut with_checksum = fletcher32_append(&data).unwrap();
        // Corrupt checksum
        let last = with_checksum.len() - 1;
        with_checksum[last] ^= 0xFF;
        let result = fletcher32_verify(&with_checksum);
        assert!(matches!(
            result,
            Err(FormatError::Fletcher32Mismatch { .. })
        ));
    }

    #[test]
    fn fletcher32_all_ones_reduction_folds_to_all_ones() {
        let data = vec![0xFFu8; 4];
        let with_checksum = fletcher32_append(&data).unwrap();
        let checksum = u32::from_le_bytes([
            with_checksum[4],
            with_checksum[5],
            with_checksum[6],
            with_checksum[7],
        ]);
        assert_eq!(checksum, 0xFFFF_FFFF);
        let verified = fletcher32_verify(&with_checksum).unwrap();
        assert_eq!(verified, data);
    }

    // --- Pipeline tests ---

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_deflate_only() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![FilterDescription {
                filter_id: FILTER_DEFLATE,
                name: None,
                flags: 0,
                client_data: vec![6],
            }],
        };
        let data: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        let dims = [data.len() as u64];
        let ctx = ChunkContext::basic(&dims, 1);
        let compressed = compress_chunk(&data, &pipeline, ctx).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline, ctx, 0).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_shuffle_deflate() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE,
                    name: None,
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
        // 25 f64 values (200 bytes)
        let data: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        let dims = [(data.len() / 8) as u64];
        let ctx = ChunkContext::basic(&dims, 8);
        let compressed = compress_chunk(&data, &pipeline, ctx).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline, ctx, 0).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_compress_decompress_roundtrip() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE,
                    name: None,
                    flags: 0,
                    client_data: vec![6],
                },
                FilterDescription {
                    filter_id: FILTER_FLETCHER32,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
            ],
        };
        let data: Vec<u8> = (0..160).map(|i| (i % 256) as u8).collect();
        let dims = [(data.len() / 8) as u64];
        let ctx = ChunkContext::basic(&dims, 8);
        let compressed = compress_chunk(&data, &pipeline, ctx).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline, ctx, 0).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_shuffle_deflate_fletcher32() {
        let pipeline = FilterPipeline {
            version: 1,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE,
                    name: None,
                    flags: 0,
                    client_data: vec![9],
                },
                FilterDescription {
                    filter_id: FILTER_FLETCHER32,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
            ],
        };
        // Use realistic f64-sized data
        let data: Vec<u8> = (0..80).map(|i| (i * 3 % 256) as u8).collect();
        let dims = [(data.len() / 8) as u64];
        let ctx = ChunkContext::basic(&dims, 8);
        let compressed = compress_chunk(&data, &pipeline, ctx).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline, ctx, 0).unwrap();
        assert_eq!(decompressed, data);
    }

    /// A non-zero `filter_mask` is per-filter: only the masked filters were
    /// skipped for this chunk; the rest still apply. The common case is a
    /// shuffle+deflate pipeline where an incompressible chunk is stored shuffled
    /// but NOT deflated. Decoding must reverse shuffle while skipping deflate.
    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_partial_mask_reverses_surviving_filter() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE, // forward index 0
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE, // forward index 1
                    name: None,
                    flags: 0,
                    client_data: vec![6],
                },
            ],
        };
        let data: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        let dims = [(data.len() / 8) as u64];
        let ctx = ChunkContext::basic(&dims, 8);

        // Stored form when deflate was declined: shuffled only.
        let stored = shuffle_compress(&data, 8).unwrap();
        // Bit 1 set => deflate (index 1) was skipped for this chunk.
        let mask = 1u32 << 1;
        let decoded = decompress_chunk(&stored, &pipeline, ctx, mask).unwrap();
        assert_eq!(
            decoded, data,
            "shuffle must be reversed even when deflate is skipped"
        );

        // The previous behaviour returned raw (still-shuffled) bytes — guard it.
        assert_ne!(
            stored, data,
            "precondition: stored bytes are shuffled, not raw"
        );
    }

    /// Symmetric case: the low filter is skipped, the high one still applies.
    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_partial_mask_skips_low_filter() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE, // forward index 0
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE, // forward index 1
                    name: None,
                    flags: 0,
                    client_data: vec![6],
                },
            ],
        };
        let data: Vec<u8> = (0u32..200)
            .map(|i| (i.wrapping_mul(7) % 256) as u8)
            .collect();
        let dims = [(data.len() / 8) as u64];
        let ctx = ChunkContext::basic(&dims, 8);

        // Shuffle skipped: stored = deflate(data) directly.
        let stored = deflate_compress(&mut FilterScratch::new(), &data, 6).unwrap();
        let mask = 1u32 << 0; // bit 0 => shuffle (index 0) skipped
        let decoded = decompress_chunk(&stored, &pipeline, ctx, mask).unwrap();
        assert_eq!(decoded, data);
    }

    // --- Decompression-bomb / size guards (#5) ---

    #[test]
    #[cfg(feature = "deflate")]
    fn deflate_decompress_rejects_bomb() {
        // A few bytes that inflate to 100 KB; with a 1 KB cap this is rejected
        // rather than allowed to allocate unbounded memory.
        let huge = vec![0u8; 100_000];
        let compressed = deflate_compress(&mut FilterScratch::new(), &huge, 9).unwrap();
        assert!(compressed.len() < 1024);
        let err =
            deflate_decompress(&mut FilterScratch::new(), &compressed, Some(1024)).unwrap_err();
        assert!(matches!(err, FormatError::FilterError(_)), "{err}");
        // Without a cap it still works (used where the size is genuinely unknown).
        assert_eq!(
            deflate_decompress(&mut FilterScratch::new(), &compressed, None)
                .unwrap()
                .len(),
            100_000
        );
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn deflate_decompress_within_cap_ok() {
        let data = vec![7u8; 500];
        let compressed = deflate_compress(&mut FilterScratch::new(), &data, 6).unwrap();
        // Cap equal to the exact output length must pass.
        assert_eq!(
            deflate_decompress(&mut FilterScratch::new(), &compressed, Some(500)).unwrap(),
            data
        );
    }

    /// The property the whole scratch rests on: a reused encoder writes exactly
    /// what a fresh one writes.
    ///
    /// Not "decodes to the same thing" — *the same bytes*. This crate's output is
    /// compared against the reference C library's byte for byte, and an encoder
    /// that carried anything across a reset (a dictionary, an adaptive Huffman
    /// table, a block-splitting decision) would round-trip perfectly while
    /// quietly changing every file this crate writes.
    #[test]
    #[cfg(feature = "deflate")]
    fn a_reused_encoder_writes_the_same_bytes_as_a_fresh_one() {
        // Deliberately varied: a compressible run, incompressible noise, an empty
        // chunk, a repeat of an earlier chunk (which a stale dictionary would
        // compress *better* than a fresh encoder can), and two odd sizes.
        let chunks: Vec<Vec<u8>> = vec![
            vec![7u8; 4096],
            (0..4096u32)
                .map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8)
                .collect(),
            Vec::new(),
            vec![7u8; 4096],
            (0..1000).map(|i| (i % 251) as u8).collect(),
            vec![0u8; 1],
        ];

        for level in [1u32, 6, 9] {
            let fresh: Vec<Vec<u8>> = chunks
                .iter()
                .map(|c| deflate_compress(&mut FilterScratch::new(), c, level).unwrap())
                .collect();

            let mut scratch = FilterScratch::new();
            let reused: Vec<Vec<u8>> = chunks
                .iter()
                .map(|c| deflate_compress(&mut scratch, c, level).unwrap())
                .collect();

            assert_eq!(
                reused, fresh,
                "a reused encoder at level {level} wrote different bytes from a fresh one"
            );

            // A fixture that cannot fail proves nothing, and this one could be:
            // were every chunk the same content, *any* carry-over would still
            // produce equal vectors. Chunks 0 and 3 are equal on purpose, to give
            // a stale dictionary something to find; the rest must differ.
            assert_eq!(
                fresh[0], fresh[3],
                "chunks 0 and 3 are the same bytes, so their encodings must be too"
            );
            let distinct: std::collections::BTreeSet<_> = fresh.iter().collect();
            assert!(
                distinct.len() >= chunks.len() - 1,
                "this fixture compresses to {} distinct outputs, too few to tell a \
                 reused encoder from a fresh one",
                distinct.len()
            );
        }
    }

    /// The decoder's counterpart, covering the two states a reused one can be
    /// left in that a fresh one never is: after a stream that ended early, and
    /// after one this crate refused as a bomb. Either way the next chunk must
    /// decode exactly.
    #[test]
    #[cfg(feature = "deflate")]
    fn a_reused_decoder_recovers_from_a_stream_it_refused() {
        let good = deflate_compress(&mut FilterScratch::new(), &vec![9u8; 2048], 6).unwrap();
        let mut scratch = FilterScratch::new();

        // A clean decode first, so the decoder is known good before it is upset.
        assert_eq!(
            deflate_decompress(&mut scratch, &good, Some(2048)).unwrap(),
            vec![9u8; 2048]
        );

        // Truncated: the stream ends before the chunk does.
        assert!(deflate_decompress(&mut scratch, &good[..good.len() / 2], Some(2048)).is_err());
        assert_eq!(
            deflate_decompress(&mut scratch, &good, Some(2048)).unwrap(),
            vec![9u8; 2048]
        );

        // Refused as too large for its chunk, which stops the decode mid-stream
        // and leaves the decoder holding a partly-consumed zlib state.
        let err = deflate_decompress(&mut scratch, &good, Some(16)).unwrap_err();
        assert!(
            format!("{err}").contains("decompression bomb"),
            "expected the bomb guard, got {err}"
        );
        assert_eq!(
            deflate_decompress(&mut scratch, &good, Some(2048)).unwrap(),
            vec![9u8; 2048]
        );
    }

    /// A zlib stream that carries every byte of its chunk but never reaches
    /// `StreamEnd` is refused, where the `Read`-based decoder this replaced
    /// accepted it.
    ///
    /// `read_to_end` stops at the first `Ok(0)`, and `flate2::read::ZlibDecoder`
    /// returns `Ok(0)` when its input runs out mid-stream — so a chunk whose
    /// adler32 trailer was never written decoded to exactly the expected length
    /// and passed, with its checksum unverified. This asserts both halves, so
    /// the change of behaviour is pinned rather than incidental.
    #[test]
    #[cfg(feature = "deflate")]
    fn an_unterminated_stream_is_refused_where_the_read_path_accepted_it() {
        use std::io::Read;

        let data = vec![7u8; 500];
        let complete = deflate_compress(&mut FilterScratch::new(), &data, 6).unwrap();
        // Every byte of compressed data, minus the 4-byte adler32 trailer.
        let unterminated = &complete[..complete.len() - 4];

        // The path this replaced: all 500 bytes, no error, checksum unchecked.
        let mut accepted = Vec::new();
        flate2::read::ZlibDecoder::new(unterminated)
            .read_to_end(&mut accepted)
            .unwrap();
        assert_eq!(accepted, data, "fixture is not a full-length truncation");

        let err = deflate_decompress(&mut FilterScratch::new(), unterminated, Some(500))
            .expect_err("an unterminated stream must not decode");
        assert!(
            format!("{err}").contains("stream ended before the chunk was complete"),
            "expected the truncation refusal, got {err}"
        );

        // The complete stream still decodes, so the refusal is about termination
        // and not about the data.
        assert_eq!(
            deflate_decompress(&mut FilterScratch::new(), &complete, Some(500)).unwrap(),
            data
        );
    }

    /// A scratch that outlives one dataset and meets another at a different level
    /// must re-level, not go on emitting the first one's.
    #[test]
    #[cfg(feature = "deflate")]
    fn a_reused_encoder_follows_a_level_change() {
        let data: Vec<u8> = (0..8192).map(|i| (i % 97) as u8).collect();
        let mut scratch = FilterScratch::new();

        let at_nine = deflate_compress(&mut scratch, &data, 9).unwrap();
        let at_one = deflate_compress(&mut scratch, &data, 1).unwrap();

        assert_eq!(
            at_one,
            deflate_compress(&mut FilterScratch::new(), &data, 1).unwrap(),
            "after a level change the encoder did not write what level 1 writes"
        );
        assert_ne!(
            at_nine, at_one,
            "levels 9 and 1 produced identical bytes, so this fixture cannot see a \
             level change at all"
        );
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn decompress_chunk_rejects_wrong_decoded_size() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![FilterDescription {
                filter_id: FILTER_DEFLATE,
                name: None,
                flags: 0,
                client_data: vec![6],
            }],
        };
        // Chunk decodes to 50 bytes, but the context expects 100 (10 elems x 10).
        let data = vec![3u8; 50];
        let compressed = compress_chunk(&data, &pipeline, ChunkContext::basic(&[50], 1)).unwrap();
        let ctx = ChunkContext::basic(&[10], 10); // expected = 100 bytes
        let err = decompress_chunk(&compressed, &pipeline, ctx, 0).unwrap_err();
        assert!(matches!(
            err,
            FormatError::DataSizeMismatch {
                expected: 100,
                actual: 50
            }
        ));
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_fletcher32_inner_deflate_outer_roundtrips() {
        // Fletcher32 BEFORE deflate on the write path (forward index 0): the
        // 4-byte checksum is appended first, then deflate compresses data+4. On
        // decode, deflate is reversed first and legitimately produces
        // `expected + 4` bytes, which must NOT be mistaken for a decompression
        // bomb by the deflate output cap.
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_FLETCHER32, // forward index 0 (inner)
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE, // forward index 1 (outer)
                    name: None,
                    flags: 0,
                    client_data: vec![6],
                },
            ],
        };
        let data: Vec<u8> = (0u32..200).map(|i| (i % 256) as u8).collect();
        let ctx = ChunkContext::basic(&[200], 1); // expected = 200
        let compressed = compress_chunk(&data, &pipeline, ctx).unwrap();
        let decoded = decompress_chunk(&compressed, &pipeline, ctx, 0).unwrap();
        assert_eq!(decoded, data);
    }

    /// A file this crate did not write may declare `[lzf, deflate]`, and the
    /// read path must decode it.
    ///
    /// `build_pipeline` refuses that combination on write and `repack`'s
    /// `check_pipeline` refuses to re-encode it, but neither runs on read:
    /// `decompress_chunk` honors whatever pipeline a file declares. LZF *grows*
    /// incompressible input, so deflate here legitimately decodes to more than
    /// the chunk size, and only the `FILTER_LZF` arm of
    /// `filter_max_forward_output` raises the cap enough to admit it. Without
    /// that arm the cap stays at the chunk size and a valid foreign chunk is
    /// rejected as a decompression bomb — the arm is load-bearing, and this is
    /// the only pipeline shape that reaches it.
    #[test]
    #[cfg(feature = "deflate")]
    fn foreign_lzf_inner_deflate_outer_roundtrips() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_LZF, // forward index 0 (inner)
                    name: Some("lzf".into()),
                    flags: 1,
                    client_data: vec![4, 0x0105, 4096],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE, // forward index 1 (outer)
                    name: None,
                    flags: 0,
                    client_data: vec![6],
                },
            ],
        };

        // Incompressible, so LZF expands rather than shrinks: the whole point
        // of the case. Compressible data would leave deflate's output under
        // the chunk size and the bound untested.
        let mut x = 0x2545_F491_4F6C_DD1D_u64;
        let data: Vec<u8> = (0..4096)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x & 0xff) as u8
            })
            .collect();

        let ctx = ChunkContext::basic(&[4096], 1); // expected = 4096
        let compressed = compress_chunk(&data, &pipeline, ctx).unwrap();
        let decoded = decompress_chunk(&compressed, &pipeline, ctx, 0).unwrap();
        assert_eq!(decoded, data);
    }

    // --- Decode reservation (#233) ---

    /// A cap the file merely declares does not size the allocation on its own.
    #[test]
    fn decode_reservation_is_bounded_by_what_the_stream_could_produce() {
        // 4 GiB is what `ensure_chunk_bytes_representable` still admits, so it
        // is a size a file can genuinely claim while carrying ten bytes.
        const CLAIMED: usize = u32::MAX as usize;
        assert_eq!(decode_reservation(Some(CLAIMED), 10, 1032), 10_320);

        // Where the claim is the smaller of the two, it is exact: a legitimate
        // chunk keeps its single up-front allocation.
        assert_eq!(decode_reservation(Some(4096), 4096, 1032), 4096);

        // No claim, nothing to be exact about.
        assert_eq!(decode_reservation(None, 4096, 1032), 0);

        // A stream long enough to overflow the product still yields a bound,
        // not a panic or a wrapped-around small one.
        assert_eq!(decode_reservation(Some(CLAIMED), usize::MAX, 1032), CLAIMED);
    }

    /// The bound is wired into the deflate decoder, not merely available to it.
    ///
    /// Observed through the returned vector's capacity, which is the
    /// reservation itself whenever the decode never had to grow past it. A
    /// reservation driven by the declared size instead would be four gigabytes
    /// for the handful of bytes this stream actually contains.
    #[test]
    #[cfg(feature = "deflate")]
    fn deflate_reserves_against_the_stream_not_the_declared_chunk_size() {
        let stored = deflate_compress(&mut FilterScratch::new(), &[], 6).unwrap();
        let out = deflate_decompress(&mut FilterScratch::new(), &stored, Some(u32::MAX as usize))
            .unwrap();
        assert!(out.is_empty());
        assert!(
            out.capacity() <= stored.len() * MAX_DEFLATE_EXPANSION,
            "reserved {} bytes for a {}-byte stream",
            out.capacity(),
            stored.len()
        );
    }

    /// The same wiring for LZF. Its bound is 88:1 rather than deflate's 1032:1,
    /// so the same declared size has to reserve less again.
    #[test]
    fn lzf_reserves_against_the_stream_not_the_declared_chunk_size() {
        let stored = crate::lzf::compress(&[0u8; 64]);
        let out = crate::lzf::decompress(&stored, Some(u32::MAX as usize)).unwrap();
        assert_eq!(out, [0u8; 64]);
        assert!(
            out.capacity() <= stored.len() * crate::lzf::MAX_EXPANSION,
            "reserved {} bytes for a {}-byte stream",
            out.capacity(),
            stored.len()
        );
    }

    /// Every filter here reports a stream it could not decode with one error
    /// variant, so a caller can match "this chunk did not decode" once.
    #[test]
    fn a_failed_decode_is_a_filter_error_whichever_compressor_failed() {
        let lzf = crate::lzf::decompress(&[0x1f], None).unwrap_err();
        assert!(matches!(lzf, FormatError::FilterError(_)), "{lzf}");

        #[cfg(feature = "deflate")]
        {
            let deflate =
                deflate_decompress(&mut FilterScratch::new(), &[0xff; 8], None).unwrap_err();
            assert!(matches!(deflate, FormatError::FilterError(_)), "{deflate}");
        }
    }
}
