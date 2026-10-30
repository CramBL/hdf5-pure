#[cfg(feature = "deflate")]
use alloc::format;
#[cfg(test)]
use alloc::vec;
use alloc::vec::Vec;
#[cfg(feature = "deflate")]
use core::mem;
use core::num::NonZeroU32;

use hdf5_pure_core::__private::PipelineFilterMask;
use hdf5_pure_core::DeflateLevel;
#[cfg(any(feature = "deflate", test))]
use hdf5_pure_core::FormatError;

use crate::error::Error;
use crate::lzf;
use crate::pipeline::buffer::FilterBuffer;
use crate::scaleoffset;
use crate::scaleoffset::ScaleOffset;
use crate::scaleoffset::ScaleOffsetType;
#[cfg(feature = "zfp")]
use crate::zfp::ZfpElementType;

mod buffer;
mod zfp;

/// Supplies the identifier, flags, and parameters for one pipeline filter.
///
/// The fields correspond to a filter description in "Data Storage - Filter Pipeline Message" of
/// the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_filter
pub trait FilterStep {
    /// Returns the registered HDF5 filter identifier.
    fn id(&self) -> u16;
    /// Returns the filter flags, including [`H5Z_FLAG_OPTIONAL`].
    fn flags(&self) -> u16;
    /// Returns the filter's client data values in their stored order.
    fn client_data(&self) -> &[u32];

    /// Returns whether this filter is marked optional for output.
    fn optional(&self) -> bool {
        self.flags() & H5Z_FLAG_OPTIONAL != 0
    }
}

/// Provides indexed filter metadata to the chunk routines.
///
/// An implementation may read its entries from a slice or another representation. The caller
/// passes an index less than [`Self::len`] to each indexed method. The routines use the supplied
/// metadata directly, without collecting filter descriptions into another sequence.
pub trait FilterSteps {
    /// Returns the number of filters in the pipeline.
    fn len(&self) -> usize;

    /// Returns whether the pipeline has no filters.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the registered HDF5 filter identifier at `index`.
    ///
    /// # Panics
    ///
    /// The slice, array, and vector implementations panic if `index` is at least [`Self::len`].
    fn id(&self, index: usize) -> u16;

    /// Returns the filter flags at `index`.
    ///
    /// # Panics
    ///
    /// The slice, array, and vector implementations panic if `index` is at least [`Self::len`].
    fn flags(&self, index: usize) -> u16;

    /// Returns the stored client data values for the filter at `index`.
    ///
    /// # Panics
    ///
    /// The slice, array, and vector implementations panic if `index` is at least [`Self::len`].
    fn client_data(&self, index: usize) -> &[u32];
}

impl<T: FilterStep> FilterSteps for [T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
    fn id(&self, index: usize) -> u16 {
        self[index].id()
    }
    fn flags(&self, index: usize) -> u16 {
        self[index].flags()
    }
    fn client_data(&self, index: usize) -> &[u32] {
        self[index].client_data()
    }
}

impl<T: FilterStep, const N: usize> FilterSteps for [T; N] {
    fn len(&self) -> usize {
        N
    }
    fn id(&self, index: usize) -> u16 {
        self[index].id()
    }
    fn flags(&self, index: usize) -> u16 {
        self[index].flags()
    }
    fn client_data(&self, index: usize) -> &[u32] {
        self[index].client_data()
    }
}

impl<T: FilterStep> FilterSteps for Vec<T> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn id(&self, index: usize) -> u16 {
        self[index].id()
    }
    fn flags(&self, index: usize) -> u16 {
        self[index].flags()
    }
    fn client_data(&self, index: usize) -> &[u32] {
        self[index].client_data()
    }
}

/// Returns the insertion index for a filter in the writer's canonical order.
///
/// Known filters rank ZFP when enabled, Scale-Offset, Shuffle, LZF, Deflate, then Fletcher32.
/// Unknown identifiers share the last rank. The input identifiers must already follow this order,
/// and a filter with the same rank stays before the new filter.
pub fn canonical_filter_position(ids: impl Iterator<Item = u16>, id: u16) -> usize {
    let rank = canonical_rank(id);
    ids.take_while(|&existing| canonical_rank(existing) <= rank)
        .count()
}

/// Returns the first incompatible pair of filters, independent of input order.
///
/// The names in the pair are suitable for an error message. ZFP conflicts take precedence when
/// ZFP support is enabled, followed by scale-offset with shuffle and LZF with Deflate.
pub fn first_filter_conflict(
    ids: impl Iterator<Item = u16>,
) -> Option<(&'static str, &'static str)> {
    let ids: Vec<u16> = ids.collect();
    let has = |id| ids.contains(&id);

    #[cfg(feature = "zfp")]
    if has(FILTER_ZFP) {
        for (id, name) in [
            (FILTER_SCALEOFFSET, "scale-offset"),
            (FILTER_SHUFFLE, "shuffle"),
            (FILTER_LZF, "lzf"),
            (FILTER_DEFLATE, "deflate"),
        ] {
            if has(id) {
                return Some((name, "ZFP"));
            }
        }
    }
    if has(FILTER_SCALEOFFSET) && has(FILTER_SHUFFLE) {
        return Some(("shuffle", "scale-offset"));
    }
    if has(FILTER_LZF) && has(FILTER_DEFLATE) {
        return Some(("lzf", "deflate"));
    }
    None
}

/// Returns whether every filter identifier is supported for re-encoding.
///
/// This classification does not check whether an optional codec feature is enabled or whether
/// the filter parameters are valid. An empty pipeline qualifies.
pub fn filters_reencodable(filters: &(impl FilterSteps + ?Sized)) -> bool {
    (0..filters.len()).all(|index| match filters.id(index) {
        FILTER_DEFLATE | FILTER_SHUFFLE | FILTER_FLETCHER32 | FILTER_SCALEOFFSET | FILTER_LZF => {
            true
        }
        #[cfg(feature = "zfp")]
        FILTER_ZFP => true,
        _ => false,
    })
}

/// Returns whether the pipeline's filters guarantee preservation of element values through
/// decoding and re-encoding.
///
/// Integer scale-offset qualifies. Floating-point scale-offset, ZFP, and unknown filters do not.
/// An empty pipeline qualifies.
pub fn filters_lossless(filters: &(impl FilterSteps + ?Sized)) -> bool {
    (0..filters.len()).all(|index| match filters.id(index) {
        FILTER_DEFLATE | FILTER_SHUFFLE | FILTER_FLETCHER32 | FILTER_LZF => true,
        FILTER_SCALEOFFSET => matches!(
            crate::scaleoffset::scale_offset_mode(filters.client_data(index)),
            Some((ScaleOffset::Integer(_), _))
        ),
        _ => false,
    })
}

fn canonical_rank(id: u16) -> u8 {
    match id {
        #[cfg(feature = "zfp")]
        FILTER_ZFP => 0,
        FILTER_SCALEOFFSET => 1,
        FILTER_SHUFFLE => 2,
        FILTER_LZF => 3,
        FILTER_DEFLATE => 4,
        FILTER_FLETCHER32 => 5,
        _ => u8::MAX,
    }
}

/// The ZFP scalar type carried by [`ChunkContext`] when ZFP is enabled.
#[cfg(feature = "zfp")]
pub(crate) type ZfpElementTypeWhenEnabled = ZfpElementType;
/// An uninhabited scalar type carried by [`ChunkContext`] when ZFP is disabled.
#[cfg(not(feature = "zfp"))]
pub(crate) type ZfpElementTypeWhenEnabled = core::convert::Infallible;

/// Describes the unfiltered size and scalar type of a chunk.
///
/// The pipeline uses the dimensions and element size to check decoded lengths and bound
/// decompression. ZFP also uses the optional scalar type.
#[derive(Debug, Clone, Copy)]
pub struct ChunkContext<'a> {
    /// Chunk dimensions in elements, including the full extent of an edge chunk.
    pub chunk_dims: &'a [u64],
    /// Number of bytes per unfiltered element.
    pub element_size: NonZeroU32,
    /// Scalar type required by ZFP, if known.
    pub element_type: Option<ZfpElementTypeWhenEnabled>,
    /// Scale-Offset scalar type supplied by the dataset adapter, if known.
    pub scale_offset_type: Option<ScaleOffsetType>,
}

#[cfg(test)]
impl<'a> ChunkContext<'a> {
    fn basic(chunk_dims: &'a [u64], element_size: u32) -> Self {
        Self {
            chunk_dims,
            element_size: NonZeroU32::new(element_size).expect("a test's element size is non-zero"),
            element_type: None,
            scale_offset_type: None,
        }
    }
}

/// Retains codec state, intermediate byte capacity, and resolved ZFP settings between chunks.
///
/// The pipeline reuses one spare byte allocation for intermediate output. Returned chunk buffers
/// own their storage independently of this scratch state. Resolved ZFP settings are reused while
/// the pipeline position, client data, scalar type, and chunk dimensions match.
#[derive(Default)]
pub struct FilterScratch {
    spare: Vec<u8>,
    #[cfg(feature = "zfp")]
    zfp: Vec<zfp::PreparedZfpFilter>,
    #[cfg(all(feature = "zfp", test))]
    zfp_preparations: usize,
    #[cfg(feature = "deflate")]
    encoder: Option<(DeflateLevel, flate2::write::ZlibEncoder<Vec<u8>>)>,
    #[cfg(feature = "deflate")]
    decoder: Option<flate2::Decompress>,
}

impl FilterScratch {
    /// Creates empty filter scratch state.
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(feature = "deflate")]
    fn deflate_decompress<'a>(
        &mut self,
        current: FilterBuffer<'a>,
        max_output: Option<usize>,
    ) -> Result<FilterBuffer<'a>, Error> {
        let mut spare = mem::take(&mut self.spare);
        let decoder = self.zlib_decoder();
        let result = current.materialize(&mut spare, |input, output| {
            deflate_decompress_into(decoder, input, max_output, output)
        });
        self.spare = spare;
        result
    }

    #[cfg(not(feature = "deflate"))]
    fn deflate_decompress<'a>(
        &mut self,
        _current: FilterBuffer<'a>,
        _max_output: Option<usize>,
    ) -> Result<FilterBuffer<'a>, Error> {
        Err(Error::UnsupportedFilter(FILTER_DEFLATE))
    }

    #[cfg(feature = "deflate")]
    fn zlib_encoder(&mut self, level: DeflateLevel) -> &mut flate2::write::ZlibEncoder<Vec<u8>> {
        let stale = self.encoder.as_ref().is_none_or(|(have, _)| *have != level);
        if stale {
            self.encoder = Some((
                level,
                flate2::write::ZlibEncoder::new(
                    Vec::new(),
                    flate2::Compression::new(u32::from(level.get())),
                ),
            ));
        }
        &mut self.encoder.as_mut().expect("built directly above").1
    }

    #[cfg(feature = "deflate")]
    fn zlib_decoder(&mut self) -> &mut flate2::Decompress {
        match &mut self.decoder {
            Some(d) => d.reset(true),
            slot => *slot = Some(flate2::Decompress::new(true)),
        }
        self.decoder.as_mut().expect("built directly above")
    }
}

/// Reverses the active filters on one stored chunk.
///
/// The decoder applies filters in reverse pipeline order. Bit `i` of `filter_mask` skips the
/// filter at index `i` in forward order. The decoded length is checked when the full chunk's
/// nonzero byte count can be calculated in `u64` and fits in `usize`.
///
/// # Errors
///
/// Returns [`Error::UnsupportedFilter`] if an active filter has no decoder.
/// Returns [`Error::DataSizeMismatch`] if a representable, nonzero full chunk size differs from
/// the decoded length.
/// Returns [`Error::Pipeline`] if the pipeline contains more than 32 filters.
/// Returns [`Error::DeflateSizeExceeded`] or [`Error::LzfSizeExceeded`] if a decoder exceeds
/// the output bound derived from the chunk size and the preceding filters.
/// A decoder may return another [`Error`] for malformed input or invalid parameters.
pub fn decompress_chunk(
    compressed: &[u8],
    filters: &(impl FilterSteps + ?Sized),
    ctx: ChunkContext<'_>,
    filter_mask: u32,
) -> Result<Vec<u8>, Error> {
    decompress_chunk_with(
        &mut FilterScratch::new(),
        compressed,
        filters,
        ctx,
        filter_mask,
    )
}

/// Reverses the active filters on one stored chunk using reusable scratch state.
///
/// Bit `i` of `filter_mask` skips the filter at index `i` in forward order. Bits outside the
/// pipeline are ignored.
/// The returned vector owns its storage independently of subsequent scratch use.
///
/// # Errors
///
/// Returns [`Error::UnsupportedFilter`] if an active filter has no decoder.
/// Returns [`Error::DataSizeMismatch`] if a representable, nonzero full chunk size differs from
/// the decoded length.
/// Returns [`Error::Pipeline`] if the pipeline contains more than 32 filters.
/// Returns [`Error::DeflateSizeExceeded`] or [`Error::LzfSizeExceeded`] if a decoder exceeds
/// the output bound derived from the chunk size and the preceding filters.
/// A decoder may return another [`Error`] for malformed input or invalid parameters.
pub fn decompress_chunk_with(
    scratch: &mut FilterScratch,
    compressed: &[u8],
    filters: &(impl FilterSteps + ?Sized),
    ctx: ChunkContext<'_>,
    filter_mask: u32,
) -> Result<Vec<u8>, Error> {
    let pipeline = MaskedFilters::new(filters, filter_mask)?;
    // Even an edge chunk is stored at full chunk size. The size bounds decompression
    // and checks the decoded length when its nonzero byte count fits in usize.
    let expected = expected_chunk_len(&ctx);

    let mut current = FilterBuffer::borrowed(compressed);
    // Filters are listed in application order. Decoding reverses them.
    // `i` is the filter's forward index and its bit position in `filter_mask`.
    // A set bit means the filter was skipped for this chunk.
    for (i, skipped) in pipeline.mask.positions().rev() {
        if skipped {
            continue;
        }
        current = match filters.id(i) {
            FILTER_SHUFFLE => {
                if ctx.element_size.get() == 1 {
                    current
                } else {
                    current.materialize(&mut scratch.spare, |input, output| {
                        shuffle_decompress_into(input, ctx.element_size.get() as usize, output);
                        Ok(())
                    })?
                }
            }
            FILTER_DEFLATE => {
                let cap = pipeline.inner_output_cap(scratch, expected, i, ctx)?;
                scratch.deflate_decompress(current, cap)?
            }
            FILTER_LZF => {
                let cap = pipeline.inner_output_cap(scratch, expected, i, ctx)?;
                current.materialize(&mut scratch.spare, |input, output| {
                    lzf::decompress_into(input, cap, output)
                })?
            }
            FILTER_FLETCHER32 => {
                let len = fletcher32_verify(current.as_slice())?.len();
                current.narrow(0..len)?
            }
            FILTER_SCALEOFFSET => {
                let cap = pipeline.inner_output_cap(scratch, expected, i, ctx)?;
                let output =
                    scaleoffset::decompress(current.as_slice(), filters.client_data(i), cap)?;
                current.replace(&mut scratch.spare, output)
            }
            #[cfg(feature = "zfp")]
            FILTER_ZFP => {
                let output = scratch
                    .zfp_settings(ctx, filters.client_data(i), i)?
                    .decompress(current.as_slice())?;
                current.replace(&mut scratch.spare, output)
            }
            other => return Err(Error::UnsupportedFilter(other)),
        };
    }

    // A valid chunk always decodes to exactly the full chunk size. A mismatch
    // means a corrupt or hostile filter stream. The error prevents silently
    // zero-filling (when short) or dropping (when long) data during chunk
    // assembly, which copies only the in-range overlap.
    if let Some(expected) = expected {
        if current.as_slice().len() != expected {
            let actual = current.as_slice().len();
            current.recycle(&mut scratch.spare);
            return Err(Error::DataSizeMismatch { expected, actual });
        }
    }
    Ok(current.into_owned())
}

fn expected_chunk_len(ctx: &ChunkContext<'_>) -> Option<usize> {
    let elems = ctx
        .chunk_dims
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))?;
    let bytes = elems.checked_mul(u64::from(ctx.element_size.get()))?;
    usize::try_from(bytes).ok().filter(|&n| n != 0)
}

struct MaskedFilters<'a, F: FilterSteps + ?Sized> {
    filters: &'a F,
    mask: PipelineFilterMask,
}

impl<'a, F: FilterSteps + ?Sized> MaskedFilters<'a, F> {
    fn new(filters: &'a F, bits: u32) -> Result<Self, Error> {
        Ok(Self {
            filters,
            mask: PipelineFilterMask::from_raw_bits(bits, filters.len())
                .map_err(Error::Pipeline)?,
        })
    }

    fn inner_output_cap(
        &self,
        scratch: &mut FilterScratch,
        expected: Option<usize>,
        filter_index: usize,
        ctx: ChunkContext<'_>,
    ) -> Result<Option<usize>, Error> {
        let Some(mut size) = expected else {
            return Ok(None);
        };
        for (index, skipped) in self.mask.positions().take(filter_index) {
            if !skipped {
                size = self.max_forward_output(scratch, index, size, ctx)?;
            }
        }
        Ok(Some(size))
    }

    fn max_forward_output(
        &self,
        scratch: &mut FilterScratch,
        index: usize,
        in_size: usize,
        ctx: ChunkContext<'_>,
    ) -> Result<usize, Error> {
        #[cfg(not(feature = "zfp"))]
        let _ = (scratch, ctx);

        Ok(match self.filters.id(index) {
            // Fletcher32 appends a 4-byte checksum.
            FILTER_FLETCHER32 => in_size.saturating_add(4),
            // A conforming LZF encoder may emit every byte as its own literal run
            // (control byte + literal), so a stream is at most twice its decoded
            // size. Matches are denser. Efficient encoders stay near in_size/32
            // overhead, but the bound must admit any conforming stream.
            FILTER_LZF => in_size.saturating_mul(2),
            // Scale-offset writes a fixed header before the payload and, when the data does not pack
            // smaller, stores it verbatim after that header.
            FILTER_SCALEOFFSET => in_size.saturating_add(scaleoffset::HEADER_LEN),
            // Deflate can slightly expand incompressible input (zlib "stored" blocks
            // plus framing). The bound exceeds zlib's worst case.
            FILTER_DEFLATE => in_size.saturating_add(in_size / 16).saturating_add(64),
            #[cfg(feature = "zfp")]
            // ZFP encodes full blocks even when the chunk ends with a partial block.
            FILTER_ZFP => scratch
                .zfp_settings(ctx, self.filters.client_data(index), index)?
                .encoded_bytes(),
            _ => in_size,
        })
    }
}

#[cfg(feature = "deflate")]
const MAX_DEFLATE_EXPANSION: usize = 1032;

#[cfg(test)]
fn compress_chunk(
    data: &[u8],
    filters: &(impl FilterSteps + ?Sized),
    ctx: ChunkContext<'_>,
) -> Result<Vec<u8>, Error> {
    compress_chunk_with(&mut FilterScratch::new(), data, filters, ctx)
}

/// Applies the filters in pipeline order using reusable scratch state.
///
/// The returned vector owns its storage independently of subsequent scratch use.
///
/// # Errors
///
/// Returns [`Error::InvalidDeflateLevel`] if a Deflate parameter exceeds `9`.
/// Returns [`Error::UnsupportedFilter`] if a filter has no encoder.
/// Returns [`Error::Pipeline`] if the pipeline contains more than 32 filters.
/// An encoder may return another [`Error`] for invalid parameters or input.
pub fn compress_chunk_with(
    scratch: &mut FilterScratch,
    data: &[u8],
    filters: &(impl FilterSteps + ?Sized),
    ctx: ChunkContext<'_>,
) -> Result<Vec<u8>, Error> {
    let pipeline = MaskedFilters::new(filters, 0)?;
    let mut current = FilterBuffer::borrowed(data);
    for (i, _) in pipeline.mask.positions() {
        current = match filters.id(i) {
            FILTER_SHUFFLE => {
                if ctx.element_size.get() == 1 {
                    current
                } else {
                    current.materialize(&mut scratch.spare, |input, output| {
                        shuffle_compress_into(input, ctx.element_size.get() as usize, output);
                        Ok(())
                    })?
                }
            }
            FILTER_DEFLATE => {
                let level = match filters.client_data(i).first() {
                    Some(&level) => DeflateLevel::try_from(level)
                        .map_err(|_error| Error::InvalidDeflateLevel { level })?,
                    None => DeflateLevel::Six,
                };
                let output = deflate_compress(scratch, current.as_slice(), level)?;
                current.replace(&mut scratch.spare, output)
            }
            FILTER_LZF => current.materialize(&mut scratch.spare, |input, output| {
                lzf::compress_into(input, output);
                Ok(())
            })?,
            FILTER_FLETCHER32 => {
                let checksum = fletcher32_compute(current.as_slice());
                current.append(&mut scratch.spare, &checksum.to_le_bytes())
            }
            FILTER_SCALEOFFSET => {
                let output = scaleoffset::compress(current.as_slice(), filters.client_data(i))?;
                current.replace(&mut scratch.spare, output)
            }
            #[cfg(feature = "zfp")]
            FILTER_ZFP => {
                let output = scratch
                    .zfp_settings(ctx, filters.client_data(i), i)?
                    .compress(current.as_slice())?;
                current.replace(&mut scratch.spare, output)
            }
            other => return Err(Error::UnsupportedFilter(other)),
        };
    }
    Ok(current.into_owned())
}

#[cfg(feature = "deflate")]
fn deflate_corrupt(reason: &str) -> Error {
    Error::FilterError(format!("deflate: {reason}"))
}

#[cfg(feature = "deflate")]
const DEFLATE_GROWTH_FLOOR: usize = 4096;

/// Decodes a complete zlib stream into reusable output storage.
///
/// The caller resets `decoder` before each stream. When bounded, the writable output includes at
/// most one byte beyond `max_output` for over-limit detection, independently of retained capacity.
///
/// # Errors
///
/// Returns [`Error::DeflateSizeExceeded`] if the decoded bytes exceed `max_output`.
/// Returns [`Error::FilterError`] if the stream is malformed or incomplete.
/// Returns [`Error::Pipeline`] if the decoder's output count exceeds the platform index width or
/// writable range.
#[cfg(feature = "deflate")]
fn deflate_decompress_into(
    decoder: &mut flate2::Decompress,
    data: &[u8],
    max_output: Option<usize>,
    out: &mut Vec<u8>,
) -> Result<(), Error> {
    use flate2::{FlushDecompress, Status};

    let bomb = || Error::DeflateSizeExceeded {
        limit: max_output.unwrap_or(usize::MAX),
        required: max_output.unwrap_or(usize::MAX).saturating_add(1),
    };
    let truncated = || deflate_corrupt("stream ended before the chunk was complete");

    // The decoded size follows from the chunk dimensions. Reserve only as much
    // as the stream can produce, so a declared size alone cannot drive allocation.
    let reservation = crate::decode_reservation(max_output, data.len(), MAX_DEFLATE_EXPANSION);
    // One past the limit lets the decoder detect an overlong stream.
    let ceiling = max_output.map(|limit| limit.saturating_add(1));

    out.clear();
    out.resize(reservation, 0);
    let mut produced = 0;

    loop {
        let consumed = usize::try_from(decoder.total_in()).unwrap_or(usize::MAX);
        let input = data.get(consumed..).unwrap_or(&[]);
        let before_in = decoder.total_in();
        let before_out = produced;
        let output = out.get_mut(produced..).ok_or_else(|| {
            Error::Pipeline(FormatError::Internal(
                "deflate output count exceeds its writable range".into(),
            ))
        })?;
        let status = decoder
            .decompress(input, output, FlushDecompress::None)
            .map_err(|e| deflate_corrupt(&e.to_string()))?;
        produced = usize::try_from(decoder.total_out()).map_err(|_error| {
            Error::Pipeline(FormatError::Internal(
                "deflate output count exceeds the platform index width".into(),
            ))
        })?;

        if status == Status::StreamEnd {
            break;
        }

        if produced == out.len() {
            if ceiling.is_some_and(|cap| produced >= cap) {
                return Err(bomb());
            }
            let want = out.len().max(DEFLATE_GROWTH_FLOOR);
            let growth = match ceiling {
                Some(cap) => want.min(cap - out.len()),
                None => want,
            };
            out.resize(out.len().saturating_add(growth), 0);
        } else if decoder.total_in() == before_in && produced == before_out {
            return Err(truncated());
        }
    }

    out.truncate(produced);
    if max_output.is_some_and(|limit| out.len() > limit) {
        return Err(bomb());
    }
    Ok(())
}

#[cfg(feature = "deflate")]
fn deflate_compress(
    scratch: &mut FilterScratch,
    data: &[u8],
    level: DeflateLevel,
) -> Result<Vec<u8>, Error> {
    use std::io::Write;
    let encoder = scratch.zlib_encoder(level);
    let finished = encoder
        .write_all(data)
        .and_then(|()| encoder.reset(Vec::new()))
        .map_err(|e| Error::FilterError(format!("deflate: {e}")));
    if finished.is_err() {
        // Either step failing leaves the encoder mid-stream, and the next chunk
        // would append to this one's unfinished output. The decoder avoids the
        // same trap by resetting on acquisition. The encoder cannot, since its
        // reset is what produces the bytes, so it is discarded instead.
        //
        // Unreachable while the sink is a `Vec`, whose writes do not fail. It is
        // handled as an error because only the current sink type rules it out.
        scratch.encoder = None;
    }
    finished
}

#[cfg(not(feature = "deflate"))]
fn deflate_compress(
    _scratch: &mut FilterScratch,
    _data: &[u8],
    _level: DeflateLevel,
) -> Result<Vec<u8>, Error> {
    Err(Error::UnsupportedFilter(FILTER_DEFLATE))
}

fn unshuffle_n<const N: usize>(data: &[u8], result: &mut [u8], num_elements: usize) {
    for (i, out) in result.as_chunks_mut::<N>().0.iter_mut().enumerate() {
        let mut elem = [0u8; N];
        for (j, b) in elem.iter_mut().enumerate() {
            *b = data[j * num_elements + i];
        }
        *out = elem;
    }
}

fn shuffle_n<const N: usize>(data: &[u8], result: &mut [u8], num_elements: usize) {
    for (i, elem) in data.as_chunks::<N>().0.iter().enumerate() {
        for (j, &b) in elem.iter().enumerate() {
            result[j * num_elements + i] = b;
        }
    }
}

/// Preserves trailing bytes while reversing byte shuffling of complete elements.
///
/// `H5Z__filter_shuffle` preserves trailing bytes in both directions (`H5Zshuffle.c`, HDF5 2.2.0).
fn shuffle_decompress_into(data: &[u8], element_size: usize, result: &mut Vec<u8>) {
    result.clear();
    result.resize(data.len(), 0);
    if element_size <= 1 {
        result.copy_from_slice(data);
        return;
    }
    let num_elements = data.len() / element_size;
    let prefix_len = data.len() - data.len() % element_size;
    result[prefix_len..].copy_from_slice(&data[prefix_len..]);

    // Specialize the common scalar widths so the inner loop unrolls and each
    // element is written as one contiguous store. The generic loop handles
    // unusual widths (compound members, wide types).
    match element_size {
        2 => unshuffle_n::<2>(data, result, num_elements),
        4 => unshuffle_n::<4>(data, result, num_elements),
        8 => unshuffle_n::<8>(data, result, num_elements),
        16 => unshuffle_n::<16>(data, result, num_elements),
        _ => {
            for i in 0..num_elements {
                for j in 0..element_size {
                    result[i * element_size + j] = data[j * num_elements + i];
                }
            }
        }
    }
}

/// Preserves trailing bytes while grouping complete elements by byte position.
fn shuffle_compress_into(data: &[u8], element_size: usize, result: &mut Vec<u8>) {
    result.clear();
    result.resize(data.len(), 0);
    if element_size <= 1 {
        result.copy_from_slice(data);
        return;
    }
    let num_elements = data.len() / element_size;
    let prefix_len = data.len() - data.len() % element_size;
    result[prefix_len..].copy_from_slice(&data[prefix_len..]);

    match element_size {
        2 => shuffle_n::<2>(data, result, num_elements),
        4 => shuffle_n::<4>(data, result, num_elements),
        8 => shuffle_n::<8>(data, result, num_elements),
        16 => shuffle_n::<16>(data, result, num_elements),
        _ => {
            for i in 0..num_elements {
                for j in 0..element_size {
                    result[j * num_elements + i] = data[i * element_size + j];
                }
            }
        }
    }
}

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

fn fletcher32_verify(data: &[u8]) -> Result<&[u8], Error> {
    if data.len() < 4 {
        return Err(Error::FilterError(
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
        return Err(Error::Fletcher32Mismatch {
            expected: stored,
            computed,
        });
    }
    Ok(payload)
}

/// The reserved filter identifier.
///
/// Defined for versions 1 and 2 of the filter pipeline message in "The Data Storage - Filter
/// Pipeline Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_filter
pub const FILTER_RESERVED: u16 = 0;
/// The registered identifier for Deflate compression.
pub const FILTER_DEFLATE: u16 = 1;
/// The registered identifier for byte Shuffle.
pub const FILTER_SHUFFLE: u16 = 2;
/// The registered identifier for Fletcher32 checksums.
pub const FILTER_FLETCHER32: u16 = 3;
/// The registered identifier for SZIP compression.
///
/// Defined for versions 1 and 2 of the filter pipeline message in "The Data Storage - Filter
/// Pipeline Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_filter
pub const FILTER_SZIP: u16 = 4;
/// The registered identifier for N-bit packing.
///
/// Defined for versions 1 and 2 of the filter pipeline message in "The Data Storage - Filter
/// Pipeline Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_filter
pub const FILTER_NBIT: u16 = 5;
/// The registered identifier for Scale-Offset compression.
pub const FILTER_SCALEOFFSET: u16 = 6;
/// The registered identifier for LZF compression.
pub const FILTER_LZF: u16 = 32000;
/// The registered identifier for ZFP compression.
pub const FILTER_ZFP: u16 = 32013;
/// Permits a filter to be omitted if it fails during output.
pub const H5Z_FLAG_OPTIONAL: u16 = 1;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::scaleoffset::ScaleOffsetByteOrder;
    use crate::scaleoffset::ScaleOffsetFill;

    mod zfp;

    #[rstest]
    #[case::first_unsupported_level(10)]
    #[case::maximum_byte(255)]
    #[case::first_value_above_byte(256)]
    #[case::wrapping_to_level_six(262)]
    #[case::maximum_u32(u32::MAX)]
    fn rejects_invalid_deflate_metadata_before_encoding(#[case] level: u32) {
        let filters = vec![FilterDescription {
            filter_id: FILTER_DEFLATE,
            flags: 0,
            client_data: vec![level],
        }];
        let error = compress_chunk_with(
            &mut FilterScratch::new(),
            &[1, 2, 3, 4],
            &filters,
            ChunkContext::basic(&[1], 4),
        )
        .unwrap_err();
        assert_eq!(error, Error::InvalidDeflateLevel { level });
    }

    #[rstest]
    #[case::one_byte(1, &[1, 2, 3, 4], &[1, 2, 3, 4, 6, 4, 8, 5])]
    #[case::two_bytes(2, &[1, 2, 3, 4], &[1, 3, 2, 4, 7, 3, 10, 4])]
    fn shuffle_checksum_encoding_returns_the_reused_allocation(
        #[case] width: u32,
        #[case] data: &[u8],
        #[case] expected: &[u8],
    ) {
        let filters = [
            FilterDescription {
                filter_id: FILTER_SHUFFLE,
                flags: 0,
                client_data: vec![],
            },
            FilterDescription {
                filter_id: FILTER_FLETCHER32,
                flags: 0,
                client_data: vec![],
            },
        ];
        let mut scratch = FilterScratch::new();
        scratch.spare = Vec::with_capacity(64);
        let pointer = scratch.spare.as_ptr();
        let context = ChunkContext::basic(&[2], width);
        let output = compress_chunk_with(&mut scratch, data, &filters, context).unwrap();
        assert_eq!(output.as_ptr(), pointer);
        assert_eq!(output, expected);
        assert_eq!(scratch.spare.capacity(), 0);
        assert_eq!(data, [1, 2, 3, 4]);
    }

    #[rstest]
    #[case::lzf_only(&[FILTER_LZF], 1, &[3, 1, 2, 3, 4])]
    #[case::shuffle_lzf(&[FILTER_SHUFFLE, FILTER_LZF], 2, &[3, 1, 3, 2, 4])]
    #[case::shuffle_lzf_checksum(&[FILTER_SHUFFLE, FILTER_LZF, FILTER_FLETCHER32], 2, &[3, 1, 3, 2, 4, 3, 10, 7, 19])]
    fn lzf_pipeline_reuses_intermediate_storage_and_preserves_returned_chunks(
        #[case] ids: &[u16],
        #[case] width: u32,
        #[case] expected: &[u8],
    ) {
        let filters: Vec<_> = ids
            .iter()
            .map(|&filter_id| FilterDescription {
                filter_id,
                flags: 0,
                client_data: vec![],
            })
            .collect();
        let dims = [u64::from(4 / width)];
        let context = ChunkContext::basic(&dims, width);
        let mut encoder = FilterScratch::new();
        encoder.spare = Vec::with_capacity(64);
        let encoded_pointer = encoder.spare.as_ptr();
        let encoded = compress_chunk_with(&mut encoder, &[1, 2, 3, 4], &filters, context).unwrap();
        assert_eq!(encoded, expected);
        let mut decoder = FilterScratch::new();
        decoder.spare = Vec::with_capacity(64);
        let decoded_pointer = decoder.spare.as_ptr();
        let decoded = decompress_chunk_with(&mut decoder, &encoded, &filters, context, 0).unwrap();
        assert_eq!(decoded, [1, 2, 3, 4]);
        if ids.contains(&FILTER_SHUFFLE) {
            assert_eq!(encoder.spare.as_ptr(), encoded_pointer);
            assert_eq!(decoder.spare.as_ptr(), decoded_pointer);
            assert_ne!(encoded.as_ptr(), encoded_pointer);
            assert_ne!(decoded.as_ptr(), decoded_pointer);
        } else {
            assert_eq!(encoded.as_ptr(), encoded_pointer);
            assert_eq!(decoded.as_ptr(), decoded_pointer);
            assert_eq!(encoder.spare.capacity(), 0);
            assert_eq!(decoder.spare.capacity(), 0);
        }
        let other = compress_chunk_with(&mut encoder, &[5, 6, 7, 8], &filters, context).unwrap();
        assert_eq!(
            decompress_chunk_with(&mut decoder, &other, &filters, context, 0).unwrap(),
            [5, 6, 7, 8]
        );
        assert_eq!(encoded, expected);
        assert_eq!(decoded, [1, 2, 3, 4]);
        assert!(encoder.spare.is_empty());
        assert!(decoder.spare.is_empty());
    }

    #[test]
    fn lzf_pipeline_clears_failed_output_and_reuses_size_rejections() {
        let filters = [FilterDescription {
            filter_id: FILTER_LZF,
            flags: 0,
            client_data: vec![],
        }];
        let context = ChunkContext::basic(&[4], 1);
        let mut scratch = FilterScratch::new();
        scratch.spare = Vec::with_capacity(64);
        let pointer = scratch.spare.as_ptr();
        assert_eq!(
            decompress_chunk_with(&mut scratch, &[0, b'a', 0x20], &filters, context, 0),
            Err(Error::InvalidLzfStream("truncated match offset"))
        );
        assert!(scratch.spare.is_empty());
        assert_eq!(scratch.spare.as_ptr(), pointer);
        let retained =
            decompress_chunk_with(&mut scratch, &[3, 1, 2, 3, 4], &filters, context, 0).unwrap();
        assert_eq!(retained.as_ptr(), pointer);
        assert_eq!(retained, [1, 2, 3, 4]);
        assert_eq!(
            decompress_chunk_with(&mut scratch, &[1, 5, 6], &filters, context, 0),
            Err(Error::DataSizeMismatch {
                expected: 4,
                actual: 2
            })
        );
        assert!(scratch.spare.is_empty());
        let pointer = scratch.spare.as_ptr();
        let decoded = decompress_chunk_with(
            &mut scratch,
            &[1, 7, 8],
            &filters,
            ChunkContext::basic(&[2], 1),
            0,
        )
        .unwrap();
        assert_eq!(decoded.as_ptr(), pointer);
        assert_eq!(decoded, [7, 8]);
        assert_eq!(retained, [1, 2, 3, 4]);
    }

    #[rstest]
    #[case::empty_pipeline(&[], 0)]
    #[case::one_byte_shuffle(&[FILTER_SHUFFLE], 0)]
    #[case::skipped_unknown(&[40000], u32::MAX)]
    fn borrowed_final_output_is_copied_only_at_the_owned_boundary(
        #[case] ids: &[u16],
        #[case] mask: u32,
    ) {
        let filters: Vec<_> = ids
            .iter()
            .map(|&filter_id| FilterDescription {
                filter_id,
                flags: 0,
                client_data: vec![],
            })
            .collect();
        let mut scratch = FilterScratch::new();
        scratch.spare = Vec::with_capacity(64);
        let pointer = scratch.spare.as_ptr();
        let input = [1, 2, 3, 4];
        let output = decompress_chunk_with(
            &mut scratch,
            &input,
            &filters,
            ChunkContext::basic(&[4], 1),
            mask,
        )
        .unwrap();
        assert_eq!(output, input);
        assert_ne!(output.as_ptr(), input.as_ptr());
        assert_eq!(scratch.spare.as_ptr(), pointer);
        assert_eq!(scratch.spare.capacity(), 64);
        assert!(scratch.spare.is_empty());
    }

    #[rstest]
    #[case::none_skipped(0)]
    #[case::first_shuffle_skipped(1)]
    #[case::middle_shuffle_skipped(4)]
    #[case::last_shuffle_skipped(16)]
    #[case::middle_checksum_skipped(8)]
    #[case::high_bits_ignored(0xffff_ffe0)]
    fn masked_transitions_follow_owned_storage(#[case] mask: u32) {
        let filters: Vec<_> = [
            FILTER_SHUFFLE,
            FILTER_FLETCHER32,
            FILTER_SHUFFLE,
            FILTER_FLETCHER32,
            FILTER_SHUFFLE,
        ]
        .into_iter()
        .map(|filter_id| FilterDescription {
            filter_id,
            flags: 0,
            client_data: vec![],
        })
        .collect();
        let data = [1, 2, 3, 4];
        let mut stored = data.to_vec();
        for (i, filter) in filters.iter().enumerate() {
            if mask & (1 << i) != 0 {
                continue;
            }
            stored = if filter.filter_id == FILTER_SHUFFLE {
                shuffle_compress(&stored, 2)
            } else {
                fletcher32_append(&stored).unwrap()
            };
        }
        let mut scratch = FilterScratch::new();
        scratch.spare = Vec::with_capacity(128);
        let result = decompress_chunk_with(
            &mut scratch,
            &stored,
            &filters,
            ChunkContext::basic(&[2], 2),
            mask,
        )
        .unwrap();
        assert_eq!(result, data);
        assert!(scratch.spare.is_empty());
        assert!(scratch.spare.capacity() <= 128);
    }

    #[rstest]
    #[case::short_checksum(&[1, 2, 3], Error::FilterError("fletcher32: data too short for checksum".into()))]
    #[case::wrong_checksum(&[1, 2, 3, 4, 0, 0, 0, 0], Error::Fletcher32Mismatch { expected: 0, computed: 84_411_398 })]
    fn scratch_recovers_after_checksum_errors(
        #[case] input: &[u8],
        #[case] expected: Error,
        #[values(false, true)] outer_shuffle: bool,
    ) {
        let mut filters = vec![FilterDescription {
            filter_id: FILTER_FLETCHER32,
            flags: 0,
            client_data: vec![],
        }];
        let stored = if outer_shuffle {
            filters.push(FilterDescription {
                filter_id: FILTER_SHUFFLE,
                flags: 0,
                client_data: vec![],
            });
            shuffle_compress(input, 2)
        } else {
            input.to_vec()
        };
        let mut scratch = FilterScratch::new();
        scratch.spare = Vec::with_capacity(64);
        let context = ChunkContext::basic(&[2], 2);
        assert_eq!(
            decompress_chunk_with(&mut scratch, &stored, &filters, context, 0),
            Err(expected)
        );
        assert!(scratch.spare.is_empty());
        let encoded =
            compress_chunk_with(&mut scratch, &[9, 10, 11, 12], &filters, context).unwrap();
        assert_eq!(
            decompress_chunk_with(&mut scratch, &encoded, &filters, context, 0).unwrap(),
            [9, 10, 11, 12]
        );
        assert_eq!(
            encoded,
            if outer_shuffle {
                [9, 11, 22, 32, 10, 12, 20, 29]
            } else {
                [9, 10, 11, 12, 22, 20, 32, 29]
            }
        );
    }

    #[test]
    fn scratch_recovers_after_size_rejection_without_changing_retained_results() {
        let filters = [
            FilterDescription {
                filter_id: FILTER_SHUFFLE,
                flags: 0,
                client_data: vec![],
            },
            FilterDescription {
                filter_id: FILTER_FLETCHER32,
                flags: 0,
                client_data: vec![],
            },
        ];
        let context = ChunkContext::basic(&[2], 2);
        let mut scratch = FilterScratch::new();
        let encoded = compress_chunk_with(&mut scratch, &[1, 2, 3, 4], &filters, context).unwrap();
        let decoded = decompress_chunk_with(&mut scratch, &encoded, &filters, context, 0).unwrap();
        assert_eq!(
            decompress_chunk_with(
                &mut scratch,
                &encoded,
                &filters,
                ChunkContext::basic(&[3], 2),
                0
            ),
            Err(Error::DataSizeMismatch {
                expected: 6,
                actual: 4
            })
        );
        assert!(scratch.spare.is_empty());
        assert!(scratch.spare.capacity() >= 4);
        let other_encoded = compress_chunk_with(&mut scratch, &[5, 6], &filters, context).unwrap();
        let other_decoded = decompress_chunk_with(
            &mut scratch,
            &other_encoded,
            &filters,
            ChunkContext::basic(&[1], 2),
            0,
        )
        .unwrap();
        assert_eq!(other_decoded, [5, 6]);
        assert_eq!(decoded, [1, 2, 3, 4]);
        assert_eq!(encoded, [1, 3, 2, 4, 7, 3, 10, 4]);
    }

    #[cfg(feature = "deflate")]
    fn deflate_decompress(
        scratch: &mut FilterScratch,
        data: &[u8],
        max_output: Option<usize>,
    ) -> Result<Vec<u8>, Error> {
        scratch
            .deflate_decompress(FilterBuffer::borrowed(data), max_output)
            .map(FilterBuffer::into_owned)
    }

    fn shuffle_compress(data: &[u8], element_size: usize) -> Vec<u8> {
        let mut output = Vec::new();
        shuffle_compress_into(data, element_size, &mut output);
        output
    }

    fn shuffle_decompress(data: &[u8], element_size: usize) -> Vec<u8> {
        let mut output = Vec::new();
        shuffle_decompress_into(data, element_size, &mut output);
        output
    }

    fn fletcher32_append(data: &[u8]) -> Result<Vec<u8>, Error> {
        Ok(FilterBuffer::borrowed(data)
            .append(&mut Vec::new(), &fletcher32_compute(data).to_le_bytes())
            .into_owned())
    }

    struct FilterDescription {
        filter_id: u16,
        flags: u16,
        client_data: Vec<u32>,
    }

    impl FilterStep for FilterDescription {
        fn id(&self) -> u16 {
            self.filter_id
        }
        fn flags(&self) -> u16 {
            self.flags
        }
        fn client_data(&self) -> &[u32] {
            &self.client_data
        }
    }

    struct FilterPipeline {
        filters: Vec<FilterDescription>,
    }

    #[rstest::rstest]
    #[cfg(feature = "deflate")]
    #[case::all_byte_values((0..256).map(|i| (i % 256) as u8).collect())]
    #[case::ten_byte_ramp((0..10).collect())]
    fn deflate_compress_decompress_roundtrip(#[case] data: Vec<u8>) {
        let compressed =
            deflate_compress(&mut FilterScratch::new(), &data, DeflateLevel::Six).unwrap();
        assert!(!compressed.is_empty());
        let decompressed =
            deflate_decompress(&mut FilterScratch::new(), &compressed, None).unwrap();
        assert_eq!(decompressed, data);
    }

    #[rstest::rstest]
    #[case::empty(0, 0)]
    #[case::one(1, 1)]
    #[case::two(2, 3)]
    #[case::full(32, u32::MAX)]
    fn every_skipped_filter_is_omitted_without_shifting_past_the_mask(
        #[case] count: usize,
        #[case] bits: u32,
    ) {
        let filters: Vec<_> = (0..count)
            .map(|_| FilterDescription {
                filter_id: 40000,
                flags: H5Z_FLAG_OPTIONAL,
                client_data: vec![7, 8],
            })
            .collect();
        assert_eq!(
            decompress_chunk(&[1, 2, 3, 4], &filters, ChunkContext::basic(&[4], 1), bits).unwrap(),
            [1, 2, 3, 4]
        );
    }

    #[test]
    fn a_pipeline_larger_than_the_mask_is_rejected_by_both_directions() {
        let filters: Vec<_> = (0..33)
            .map(|_| FilterDescription {
                filter_id: FILTER_SHUFFLE,
                flags: 0,
                client_data: vec![1],
            })
            .collect();
        let expected = Error::Pipeline(FormatError::InvalidFilterPipelineField {
            field: "filter count",
            value: 33,
            maximum: 32,
        });
        let ctx = ChunkContext::basic(&[4], 1);
        assert_eq!(
            decompress_chunk(&[1, 2, 3, 4], &filters, ctx, 0),
            Err(expected.clone())
        );
        assert_eq!(compress_chunk(&[1, 2, 3, 4], &filters, ctx), Err(expected));
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn deflate_decompress_python_zlib() {
        let compressed: Vec<u8> = vec![
            120, 156, 99, 96, 100, 98, 102, 97, 101, 99, 231, 224, 4, 0, 0, 175, 0, 46,
        ];
        let decompressed =
            deflate_decompress(&mut FilterScratch::new(), &compressed, None).unwrap();
        assert_eq!(decompressed, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[rstest::rstest]
    #[case::one_byte(1)]
    #[case::two_bytes(2)]
    #[case::three_bytes(3)]
    #[case::four_bytes(4)]
    #[case::six_bytes(6)]
    #[case::seven_bytes(7)]
    #[case::eight_bytes(8)]
    #[case::sixteen_bytes(16)]
    fn shuffle_roundtrip(
        #[case] width: usize,
        #[values(0, 1, 50)] elements: usize,
        #[values(0, 1)] remainder: usize,
    ) {
        let data: Vec<u8> = (0..(width * elements + remainder))
            .map(|i| (i * 31 % 256) as u8)
            .collect();
        let shuffled = shuffle_compress(&data, width);
        assert_eq!(shuffled.len(), data.len());
        assert_eq!(shuffle_decompress(&shuffled, width), data);
    }

    #[rstest::rstest]
    #[case::two_bytes(2)]
    #[case::four_bytes(4)]
    #[case::eight_bytes(8)]
    #[case::sixteen_bytes(16)]
    fn shuffle_specialized_matches_generic(#[case] width: usize) {
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
        let data: Vec<u8> = (0..(width * 37)).map(|i| (i * 17 + 3) as u8).collect();
        assert_eq!(
            shuffle_compress(&data, width),
            generic_shuffle(&data, width)
        );
    }

    #[test]
    fn shuffle_known_pattern() {
        let data = vec![0xA0, 0xA1, 0xA2, 0xA3, 0xB0, 0xB1, 0xB2, 0xB3];
        let shuffled = shuffle_compress(&data, 4);
        assert_eq!(
            shuffled,
            vec![0xA0, 0xB0, 0xA1, 0xB1, 0xA2, 0xB2, 0xA3, 0xB3]
        );
    }

    #[rstest]
    #[case::two_byte_elements_one_tail_byte(2, &[1, 2, 3, 4, 91], &[1, 3, 2, 4, 91])]
    #[case::three_byte_elements_two_tail_bytes(3, &[1, 2, 3, 4, 5, 6, 91, 92], &[1, 4, 2, 5, 3, 6, 91, 92])]
    #[case::four_byte_elements_two_tail_bytes(4, &[1, 2, 3, 4, 5, 6, 7, 8, 91, 92], &[1, 5, 2, 6, 3, 7, 4, 8, 91, 92])]
    #[case::eight_byte_elements_two_tail_bytes(
        8,
        &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 91, 92],
        &[1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 8, 16, 91, 92]
    )]
    #[case::sixteen_byte_elements_two_tail_bytes(
        16,
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 91, 92
        ],
        &[
            1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24,
            9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 16, 32, 91, 92
        ]
    )]
    #[case::empty(4, &[], &[])]
    #[case::shorter_than_element(4, &[91, 92], &[91, 92])]
    #[case::one_element_with_tail(4, &[1, 2, 3, 4, 91, 92], &[1, 2, 3, 4, 91, 92])]
    fn shuffle_preserves_remainder(
        #[case] width: usize,
        #[case] data: &[u8],
        #[case] shuffled: &[u8],
        #[values(false, true)] reverse: bool,
    ) {
        if reverse {
            assert_eq!(shuffle_decompress(shuffled, width), data);
        } else {
            assert_eq!(shuffle_compress(data, width), shuffled);
        }
    }

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
        let data = vec![0u8; 16];
        let with_checksum = fletcher32_append(&data).unwrap();
        let checksum = u32::from_le_bytes([
            with_checksum[16],
            with_checksum[17],
            with_checksum[18],
            with_checksum[19],
        ]);
        assert_eq!(checksum, 0);

        let data2 = vec![1u8, 0, 0, 0];
        let with_checksum2 = fletcher32_append(&data2).unwrap();
        let verified = fletcher32_verify(&with_checksum2).unwrap();
        assert_eq!(verified, data2);
    }

    #[test]
    fn fletcher32_mismatch_detected() {
        let data = vec![1u8, 2, 3, 4];
        let mut with_checksum = fletcher32_append(&data).unwrap();
        let last = with_checksum.len() - 1;
        with_checksum[last] ^= 0xFF;
        let expected = u32::from_le_bytes(with_checksum[data.len()..].try_into().unwrap());
        let err = fletcher32_verify(&with_checksum).unwrap_err();
        assert_eq!(
            err,
            Error::Fletcher32Mismatch {
                expected,
                computed: fletcher32_compute(&data),
            }
        );
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

    #[test]
    fn pipeline_fletcher32_then_shuffle_preserves_remainder() {
        let pipeline = FilterPipeline {
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_FLETCHER32,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_SHUFFLE,
                    flags: 0,
                    client_data: vec![8],
                },
            ],
        };
        let data: Vec<u8> = (0..512).map(|i| (i % 251) as u8).collect();
        let ctx = ChunkContext::basic(&[64], 8);
        let encoded = compress_chunk(&data, &pipeline.filters, ctx).unwrap();
        assert_eq!(encoded.len(), 516);
        assert_eq!(&encoded[512..], &fletcher32_append(&data).unwrap()[512..]);
        assert_eq!(
            decompress_chunk(&encoded, &pipeline.filters, ctx, 0).unwrap(),
            data
        );
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_deflate_only() {
        let pipeline = FilterPipeline {
            filters: vec![FilterDescription {
                filter_id: FILTER_DEFLATE,
                flags: 0,
                client_data: vec![6],
            }],
        };
        let data: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        let dims = [data.len() as u64];
        let ctx = ChunkContext::basic(&dims, 1);
        let compressed = compress_chunk(&data, &pipeline.filters, ctx).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline.filters, ctx, 0).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_shuffle_deflate() {
        let pipeline = FilterPipeline {
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE,
                    flags: 0,
                    client_data: vec![6],
                },
            ],
        };
        let data: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        let dims = [(data.len() / 8) as u64];
        let ctx = ChunkContext::basic(&dims, 8);
        let compressed = compress_chunk(&data, &pipeline.filters, ctx).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline.filters, ctx, 0).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_compress_decompress_roundtrip() {
        let pipeline = FilterPipeline {
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE,
                    flags: 0,
                    client_data: vec![6],
                },
                FilterDescription {
                    filter_id: FILTER_FLETCHER32,
                    flags: 0,
                    client_data: vec![],
                },
            ],
        };
        let data: Vec<u8> = (0..160).map(|i| (i % 256) as u8).collect();
        let dims = [(data.len() / 8) as u64];
        let ctx = ChunkContext::basic(&dims, 8);
        let compressed = compress_chunk(&data, &pipeline.filters, ctx).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline.filters, ctx, 0).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_shuffle_deflate_fletcher32() {
        let pipeline = FilterPipeline {
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE,
                    flags: 0,
                    client_data: vec![9],
                },
                FilterDescription {
                    filter_id: FILTER_FLETCHER32,
                    flags: 0,
                    client_data: vec![],
                },
            ],
        };
        let data: Vec<u8> = (0..80).map(|i| (i * 3 % 256) as u8).collect();
        let dims = [(data.len() / 8) as u64];
        let ctx = ChunkContext::basic(&dims, 8);
        let compressed = compress_chunk(&data, &pipeline.filters, ctx).unwrap();
        let decompressed = decompress_chunk(&compressed, &pipeline.filters, ctx, 0).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_partial_mask_reverses_surviving_filter() {
        let pipeline = FilterPipeline {
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE, // forward index 0
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE, // forward index 1
                    flags: 0,
                    client_data: vec![6],
                },
            ],
        };
        let data: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        let dims = [(data.len() / 8) as u64];
        let ctx = ChunkContext::basic(&dims, 8);

        let stored = shuffle_compress(&data, 8);
        let mask = 1u32 << 1;
        let decoded = decompress_chunk(&stored, &pipeline.filters, ctx, mask).unwrap();
        assert_eq!(
            decoded, data,
            "shuffle must be reversed even when deflate is skipped"
        );

        assert_ne!(
            stored, data,
            "precondition: stored bytes are shuffled, not raw"
        );
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_partial_mask_skips_low_filter() {
        let pipeline = FilterPipeline {
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE, // forward index 0
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE, // forward index 1
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

        let stored = deflate_compress(&mut FilterScratch::new(), &data, DeflateLevel::Six).unwrap();
        let mask = 1u32 << 0; // bit 0 => shuffle (index 0) skipped
        let decoded = decompress_chunk(&stored, &pipeline.filters, ctx, mask).unwrap();
        assert_eq!(decoded, data);
    }

    #[rstest]
    #[cfg(feature = "deflate")]
    #[case::zero_limit(0)]
    #[case::one_byte_limit(1)]
    #[case::small_limit(16)]
    #[case::above_growth_floor(4097)]
    fn deflate_writes_only_through_the_operation_limit(
        #[case] limit: usize,
        #[values(false, true)] retained: bool,
    ) {
        let compressed = deflate_compress(
            &mut FilterScratch::new(),
            &vec![7; 100_000],
            DeflateLevel::Six,
        )
        .unwrap();
        let mut scratch = FilterScratch::new();
        if retained {
            scratch.spare = Vec::with_capacity(128 * 1024);
        }
        let capacity = scratch.spare.capacity();
        let pointer = scratch.spare.as_ptr();
        assert_eq!(
            deflate_decompress(&mut scratch, &compressed, Some(limit)),
            Err(Error::DeflateSizeExceeded {
                limit,
                required: limit + 1
            })
        );
        assert_eq!(
            scratch.decoder.as_ref().unwrap().total_out(),
            u64::try_from(limit + 1).unwrap()
        );
        assert!(scratch.spare.is_empty());
        if retained {
            assert_eq!(scratch.spare.capacity(), capacity);
            assert_eq!(scratch.spare.as_ptr(), pointer);
        }
        let valid =
            deflate_compress(&mut FilterScratch::new(), &[9; 32], DeflateLevel::Six).unwrap();
        assert_eq!(
            deflate_decompress(&mut scratch, &valid, Some(32)).unwrap(),
            [9; 32]
        );
    }

    #[rstest]
    #[cfg(feature = "deflate")]
    #[case::missing_checksum(4)]
    #[case::partial_checksum(2)]
    #[case::missing_last_byte(1)]
    fn deflate_rejects_a_missing_trailer_after_producing_the_full_payload(
        #[case] removed: usize,
        #[values(false, true)] retained: bool,
    ) {
        let stored = [
            120, 156, 99, 96, 100, 98, 102, 97, 101, 99, 231, 224, 4, 0, 0, 175, 0, 46,
        ];
        let mut scratch = FilterScratch::new();
        if retained {
            scratch.spare = Vec::with_capacity(64 * 1024);
        }
        assert_eq!(
            deflate_decompress(&mut scratch, &stored[..stored.len() - removed], Some(10)),
            Err(Error::FilterError(
                "deflate: stream ended before the chunk was complete".into()
            ))
        );
        assert_eq!(scratch.decoder.as_ref().unwrap().total_out(), 10);
        assert!(scratch.spare.is_empty());
        let pointer = scratch.spare.as_ptr();
        let decoded = deflate_decompress(&mut scratch, &stored, Some(10)).unwrap();
        assert_eq!(decoded, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(decoded.as_ptr(), pointer);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn deflate_reuses_rejected_output_without_changing_retained_chunks() {
        let filters = [FilterDescription {
            filter_id: FILTER_DEFLATE,
            flags: 0,
            client_data: vec![],
        }];
        let first =
            deflate_compress(&mut FilterScratch::new(), &[1; 256], DeflateLevel::Six).unwrap();
        let second =
            deflate_compress(&mut FilterScratch::new(), &[2; 64], DeflateLevel::Six).unwrap();
        let third =
            deflate_compress(&mut FilterScratch::new(), &[3; 32], DeflateLevel::Six).unwrap();
        let mut scratch = FilterScratch::new();
        let retained = decompress_chunk_with(
            &mut scratch,
            &first,
            &filters,
            ChunkContext::basic(&[256], 1),
            0,
        )
        .unwrap();
        assert_eq!(
            decompress_chunk_with(
                &mut scratch,
                &second,
                &filters,
                ChunkContext::basic(&[128], 1),
                0
            ),
            Err(Error::DataSizeMismatch {
                expected: 128,
                actual: 64
            })
        );
        assert!(scratch.spare.is_empty());
        let pointer = scratch.spare.as_ptr();
        let last = decompress_chunk_with(
            &mut scratch,
            &third,
            &filters,
            ChunkContext::basic(&[32], 1),
            0,
        )
        .unwrap();
        assert_eq!(last.as_ptr(), pointer);
        assert_eq!(last, [3; 32]);
        assert_eq!(retained, [1; 256]);
        assert_eq!(scratch.spare.capacity(), 0);
    }

    #[rstest]
    #[cfg(feature = "deflate")]
    #[case::checksum_active(0)]
    #[case::checksum_skipped(1)]
    #[case::outside_bits_ignored(0xffff_fffc)]
    fn deflate_uses_the_limit_from_active_inner_stages(#[case] mask: u32) {
        let filters = [
            FilterDescription {
                filter_id: FILTER_FLETCHER32,
                flags: 0,
                client_data: vec![],
            },
            FilterDescription {
                filter_id: FILTER_DEFLATE,
                flags: 0,
                client_data: vec![],
            },
        ];
        let context = ChunkContext::basic(&[16], 1);
        let encoded = compress_chunk(&[1; 16], &filters, context).unwrap();
        let mut scratch = FilterScratch::new();
        scratch.spare = Vec::with_capacity(64 * 1024);
        let decoded = decompress_chunk_with(&mut scratch, &encoded, &filters, context, mask);
        if mask & 1 != 0 {
            assert_eq!(
                decoded,
                Err(Error::DeflateSizeExceeded {
                    limit: 16,
                    required: 17
                })
            );
            assert_eq!(scratch.decoder.as_ref().unwrap().total_out(), 17);
            assert!(scratch.spare.is_empty());
        } else {
            assert_eq!(decoded.unwrap(), [1; 16]);
        }
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn deflate_decompress_rejects_bomb() {
        let huge = vec![0u8; 100_000];
        let compressed =
            deflate_compress(&mut FilterScratch::new(), &huge, DeflateLevel::Nine).unwrap();
        assert!(compressed.len() < 1024);
        let err =
            deflate_decompress(&mut FilterScratch::new(), &compressed, Some(1024)).unwrap_err();
        assert_eq!(
            err,
            Error::DeflateSizeExceeded {
                limit: 1024,
                required: 1025
            }
        );
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
        let compressed =
            deflate_compress(&mut FilterScratch::new(), &data, DeflateLevel::Six).unwrap();
        assert_eq!(
            deflate_decompress(&mut FilterScratch::new(), &compressed, Some(500)).unwrap(),
            data
        );
    }

    #[rstest::rstest]
    #[cfg(feature = "deflate")]
    #[case::level_one(DeflateLevel::One)]
    #[case::level_six(DeflateLevel::Six)]
    #[case::level_nine(DeflateLevel::Nine)]
    fn a_reused_encoder_writes_the_same_bytes_as_a_fresh_one(#[case] level: DeflateLevel) {
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
            "a reused encoder at level {level:?} wrote different bytes from a fresh one"
        );

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

    #[test]
    #[cfg(feature = "deflate")]
    fn a_reused_decoder_recovers_from_a_stream_it_refused() {
        let good = deflate_compress(
            &mut FilterScratch::new(),
            &vec![9u8; 2048],
            DeflateLevel::Six,
        )
        .unwrap();
        let mut scratch = FilterScratch::new();

        assert_eq!(
            deflate_decompress(&mut scratch, &good, Some(2048)).unwrap(),
            vec![9u8; 2048]
        );

        let err =
            deflate_decompress(&mut scratch, &good[..good.len() / 2], Some(2048)).unwrap_err();
        assert_eq!(
            err,
            Error::FilterError("deflate: stream ended before the chunk was complete".into())
        );
        assert_eq!(
            deflate_decompress(&mut scratch, &good, Some(2048)).unwrap(),
            vec![9u8; 2048]
        );

        let err = deflate_decompress(&mut scratch, &good, Some(16)).unwrap_err();
        assert_eq!(
            err,
            Error::DeflateSizeExceeded {
                limit: 16,
                required: 17
            }
        );
        assert_eq!(
            deflate_decompress(&mut scratch, &good, Some(2048)).unwrap(),
            vec![9u8; 2048]
        );
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn unterminated_stream_is_rejected_after_read_decoder_returns_data() {
        use std::io::Read;

        let data = vec![7u8; 500];
        let complete =
            deflate_compress(&mut FilterScratch::new(), &data, DeflateLevel::Six).unwrap();
        let unterminated = &complete[..complete.len() - 4];

        let mut accepted = Vec::new();
        flate2::read::ZlibDecoder::new(unterminated)
            .read_to_end(&mut accepted)
            .unwrap();
        assert_eq!(accepted, data, "fixture is not a full-length truncation");

        let err = deflate_decompress(&mut FilterScratch::new(), unterminated, Some(500))
            .expect_err("an unterminated stream must not decode");
        assert_eq!(
            err,
            Error::FilterError("deflate: stream ended before the chunk was complete".into())
        );

        assert_eq!(
            deflate_decompress(&mut FilterScratch::new(), &complete, Some(500)).unwrap(),
            data
        );
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn a_reused_encoder_follows_a_level_change() {
        let data: Vec<u8> = (0..8192).map(|i| (i % 97) as u8).collect();
        let mut scratch = FilterScratch::new();

        let at_nine = deflate_compress(&mut scratch, &data, DeflateLevel::Nine).unwrap();
        let at_one = deflate_compress(&mut scratch, &data, DeflateLevel::One).unwrap();

        assert_eq!(
            at_one,
            deflate_compress(&mut FilterScratch::new(), &data, DeflateLevel::One).unwrap(),
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
            filters: vec![FilterDescription {
                filter_id: FILTER_DEFLATE,
                flags: 0,
                client_data: vec![6],
            }],
        };
        let data = vec![3u8; 50];
        let compressed =
            compress_chunk(&data, &pipeline.filters, ChunkContext::basic(&[50], 1)).unwrap();
        let ctx = ChunkContext::basic(&[10], 10); // expected = 100 bytes
        let err = decompress_chunk(&compressed, &pipeline.filters, ctx, 0).unwrap_err();
        assert!(matches!(
            err,
            Error::DataSizeMismatch {
                expected: 100,
                actual: 50
            }
        ));
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn pipeline_fletcher32_inner_deflate_outer_roundtrips() {
        let pipeline = FilterPipeline {
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_FLETCHER32, // forward index 0 (inner)
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE, // forward index 1 (outer)
                    flags: 0,
                    client_data: vec![6],
                },
            ],
        };
        let data: Vec<u8> = (0u32..200).map(|i| (i % 256) as u8).collect();
        let ctx = ChunkContext::basic(&[200], 1); // expected = 200
        let compressed = compress_chunk(&data, &pipeline.filters, ctx).unwrap();
        let decoded = decompress_chunk(&compressed, &pipeline.filters, ctx, 0).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn lzf_inner_deflate_outer_roundtrips() {
        let pipeline = FilterPipeline {
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_LZF, // forward index 0 (inner)
                    flags: 1,
                    client_data: vec![4, 0x0105, 4096],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE, // forward index 1 (outer)
                    flags: 0,
                    client_data: vec![6],
                },
            ],
        };

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
        let compressed = compress_chunk(&data, &pipeline.filters, ctx).unwrap();
        let decoded = decompress_chunk(&compressed, &pipeline.filters, ctx, 0).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn decode_reservation_is_bounded_by_what_the_stream_could_produce() {
        const CLAIMED: usize = u32::MAX as usize;
        assert_eq!(crate::decode_reservation(Some(CLAIMED), 10, 1032), 10_320);

        assert_eq!(crate::decode_reservation(Some(4096), 4096, 1032), 4096);

        assert_eq!(crate::decode_reservation(None, 4096, 1032), 0);

        assert_eq!(
            crate::decode_reservation(Some(CLAIMED), usize::MAX, 1032),
            CLAIMED
        );
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn deflate_reserves_against_the_stream_not_the_declared_chunk_size() {
        let stored = deflate_compress(&mut FilterScratch::new(), &[], DeflateLevel::Six).unwrap();
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

    #[test]
    fn lzf_reserves_against_the_stream_not_the_declared_chunk_size() {
        let stored = lzf::compress(&[0u8; 64]);
        let out = lzf::decompress(&stored, Some(u32::MAX as usize)).unwrap();
        assert_eq!(out, [0u8; 64]);
        assert!(
            out.capacity() <= stored.len() * lzf::MAX_EXPANSION,
            "reserved {} bytes for a {}-byte stream",
            out.capacity(),
            stored.len()
        );
    }

    #[test]
    fn a_failed_decode_reports_which_compressor_failed() {
        let lzf = lzf::decompress(&[0x1f], None).unwrap_err();
        assert_eq!(lzf, Error::InvalidLzfStream("truncated literal run"));

        #[cfg(feature = "deflate")]
        {
            let deflate =
                deflate_decompress(&mut FilterScratch::new(), &[0xff; 8], None).unwrap_err();
            let Error::FilterError(reason) = deflate else {
                panic!("expected FilterError, got {deflate:?}");
            };
            assert!(
                matches!(
                    reason.as_str(),
                    "deflate: deflate decompression error"
                        | "deflate: deflate decompression error: incorrect header check"
                ),
                "{reason:?}"
            );
        }
    }

    struct Step {
        id: u16,
        flags: u16,
        data: Vec<u32>,
    }

    impl FilterStep for Step {
        fn id(&self) -> u16 {
            self.id
        }

        fn flags(&self) -> u16 {
            self.flags
        }

        fn client_data(&self) -> &[u32] {
            &self.data
        }
    }

    #[test]
    fn shuffle_and_fletcher32_encode_and_a_chunk_that_skips_fletcher32_decodes() {
        let steps = [
            Step {
                id: FILTER_SHUFFLE,
                flags: 0,
                data: Vec::new(),
            },
            Step {
                id: FILTER_FLETCHER32,
                flags: H5Z_FLAG_OPTIONAL,
                data: Vec::new(),
            },
        ];
        assert!(steps[1].optional());
        let ctx = ChunkContext {
            chunk_dims: &[2],
            element_size: NonZeroU32::new(2).unwrap(),
            element_type: None,
            scale_offset_type: None,
        };
        let data = [1, 2, 3, 4];
        let encoded = compress_chunk_with(&mut FilterScratch::new(), &data, &steps, ctx).unwrap();
        assert_eq!(encoded, [1, 3, 2, 4, 7, 3, 10, 4]);
        let sliced =
            compress_chunk_with(&mut FilterScratch::new(), &data, &steps[..1], ctx).unwrap();
        assert_eq!(sliced, [1, 3, 2, 4]);
        assert_eq!(decompress_chunk(&encoded, &steps, ctx, 0).unwrap(), data);
        assert_eq!(
            decompress_chunk(&encoded[..4], &steps, ctx, 1 << 1).unwrap(),
            data
        );
    }

    #[rstest]
    #[case::shuffle_before_deflate_and_checksum(&[FILTER_DEFLATE, FILTER_FLETCHER32], FILTER_SHUFFLE, 0)]
    #[case::deflate_between_shuffle_and_checksum(&[FILTER_SHUFFLE, FILTER_FLETCHER32], FILTER_DEFLATE, 1)]
    #[case::checksum_after_shuffle_and_deflate(&[FILTER_SHUFFLE, FILTER_DEFLATE], FILTER_FLETCHER32, 2)]
    #[case::shuffle_before_reversed_pipeline(&[FILTER_FLETCHER32, FILTER_DEFLATE], FILTER_SHUFFLE, 0)]
    fn canonical_position_keeps_existing_pipeline_order(
        #[case] existing: &[u16],
        #[case] added: u16,
        #[case] position: usize,
    ) {
        assert_eq!(
            canonical_filter_position(existing.iter().copied(), added),
            position
        );
    }

    #[rstest]
    #[case::lzf_then_deflate(&[FILTER_LZF, FILTER_DEFLATE], Some(("lzf", "deflate")))]
    #[case::deflate_then_lzf(&[FILTER_DEFLATE, FILTER_LZF], Some(("lzf", "deflate")))]
    #[case::scale_offset_then_shuffle(&[FILTER_SCALEOFFSET, FILTER_SHUFFLE], Some(("shuffle", "scale-offset")))]
    #[case::shuffle_then_scale_offset(&[FILTER_SHUFFLE, FILTER_SCALEOFFSET], Some(("shuffle", "scale-offset")))]
    #[case::shuffle_then_checksum(&[FILTER_SHUFFLE, FILTER_FLETCHER32], None)]
    fn conflicts_are_independent_of_filter_order(
        #[case] filters: &[u16],
        #[case] expected: Option<(&str, &str)>,
    ) {
        assert_eq!(first_filter_conflict(filters.iter().copied()), expected);
    }

    #[test]
    fn edit_classification_distinguishes_lossless_scale_offset() {
        let scalar = ScaleOffsetType::integer(false, ScaleOffsetByteOrder::LittleEndian);
        let integer = crate::scaleoffset::build_cd_values(
            ScaleOffset::Integer(0),
            scalar,
            1,
            4,
            ScaleOffsetFill::Undefined,
        )
        .unwrap();
        let integer_pipeline = [Step {
            id: FILTER_SCALEOFFSET,
            flags: 0,
            data: integer,
        }];
        assert!(filters_reencodable(&integer_pipeline));
        assert!(filters_lossless(&integer_pipeline));

        let scalar = ScaleOffsetType::floating(ScaleOffsetByteOrder::LittleEndian);
        let float = crate::scaleoffset::build_cd_values(
            ScaleOffset::FloatDScale(2),
            scalar,
            4,
            4,
            ScaleOffsetFill::Undefined,
        )
        .unwrap();
        let float_pipeline = [Step {
            id: FILTER_SCALEOFFSET,
            flags: 0,
            data: float,
        }];
        assert!(filters_reencodable(&float_pipeline));
        assert!(!filters_lossless(&float_pipeline));

        let unknown_pipeline = [Step {
            id: u16::MAX,
            flags: 0,
            data: Vec::new(),
        }];
        assert!(!filters_reencodable(&unknown_pipeline));
        assert!(!filters_lossless(&unknown_pipeline));
    }
}
