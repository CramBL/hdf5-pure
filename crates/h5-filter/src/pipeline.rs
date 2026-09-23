#[cfg(feature = "deflate")]
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroU32;

use crate::Error;
use crate::ScaleOffsetType;
#[cfg(feature = "zfp")]
use crate::ZfpElementType;

/// Supplies the identifier, flags, and parameters for one pipeline filter.
///
/// The fields correspond to a filter description in "Data Storage - Filter Pipeline Message" of
/// the [format specification, version 4.0][spec]. See [`decompress_chunk`] for an example.
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

/// The ZFP scalar type carried by [`ChunkContext`] when ZFP is enabled.
#[cfg(feature = "zfp")]
pub type ZfpElementTypeWhenEnabled = ZfpElementType;
/// An uninhabited scalar type carried by [`ChunkContext`] when ZFP is disabled.
#[cfg(not(feature = "zfp"))]
pub type ZfpElementTypeWhenEnabled = core::convert::Infallible;

/// Describes the unfiltered size and scalar type of a chunk.
///
/// The pipeline uses the dimensions and element size to check decoded lengths and bound
/// decompression. ZFP also uses the optional scalar type. See [`decompress_chunk`] for an example.
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

#[cfg(all(test, feature = "deflate"))]
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

/// Holds reusable Deflate encoder and decoder state between chunks.
///
/// The state is allocated on first use. Pipelines without Deflate need no Deflate state.
#[derive(Default)]
pub struct FilterScratch {
    #[cfg(feature = "deflate")]
    encoder: Option<(u32, flate2::write::ZlibEncoder<Vec<u8>>)>,
    #[cfg(feature = "deflate")]
    decoder: Option<flate2::Decompress>,
}

impl FilterScratch {
    /// Creates empty filter scratch state.
    pub fn new() -> Self {
        Self::default()
    }

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
/// Filters are reversed in pipeline order. Bit `i` of `filter_mask` skips the filter at
/// index `i` in forward order. The decoded length is checked when the full chunk's nonzero
/// byte count can be calculated in `u64` and fits in `usize`.
///
/// # Errors
///
/// Returns [`Error::UnsupportedFilter`] if an active filter has no decoder.
/// Returns [`Error::DataSizeMismatch`] if a representable, nonzero full chunk size differs from
/// the decoded length.
/// A decoder may return another [`Error`] for malformed input or invalid parameters.
///
/// # Examples
///
/// ```
/// use core::num::NonZeroU32;
/// use h5_filter::{
///     ChunkContext, FILTER_FLETCHER32, FILTER_SHUFFLE, FilterScratch, FilterStep,
///     H5Z_FLAG_OPTIONAL, compress_chunk_with, decompress_chunk, decompress_chunk_with,
/// };
///
/// struct Step {
///     id: u16,
///     flags: u16,
/// }
///
/// impl FilterStep for Step {
///     fn id(&self) -> u16 { self.id }
///     fn flags(&self) -> u16 { self.flags }
///     fn client_data(&self) -> &[u32] { &[] }
/// }
///
/// # fn main() -> Result<(), h5_filter::Error> {
/// let filters = [
///     Step { id: FILTER_SHUFFLE, flags: 0 },
///     Step { id: FILTER_FLETCHER32, flags: H5Z_FLAG_OPTIONAL },
/// ];
/// let context = ChunkContext {
///     chunk_dims: &[4],
///     element_size: NonZeroU32::new(2).unwrap(),
///     element_type: None,
///     scale_offset_type: None,
/// };
/// let data = [1_u8, 2, 3, 4, 5, 6, 7, 8];
/// let mut scratch = FilterScratch::new();
/// let stored = compress_chunk_with(&mut scratch, &data, &filters, context)?;
/// assert_eq!(decompress_chunk_with(&mut scratch, &stored, &filters, context, 0)?, data);
///
/// let stored_without_checksum = compress_chunk_with(&mut scratch, &data, &filters[..1], context)?;
/// assert_eq!(decompress_chunk(&stored_without_checksum, &filters, context, 0b10)?, data);
/// # Ok(())
/// # }
/// ```
pub fn decompress_chunk(
    compressed: &[u8],
    filters: &[impl FilterStep],
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
/// Bit `i` of `filter_mask` skips the filter at index `i` in forward order.
/// See [`decompress_chunk`] for an example using reusable scratch state.
///
/// # Errors
///
/// Returns [`Error::UnsupportedFilter`] if an active filter has no decoder.
/// Returns [`Error::DataSizeMismatch`] if a representable, nonzero full chunk size differs from
/// the decoded length.
/// A decoder may return another [`Error`] for malformed input or invalid parameters.
pub fn decompress_chunk_with(
    scratch: &mut FilterScratch,
    compressed: &[u8],
    filters: &[impl FilterStep],
    ctx: ChunkContext<'_>,
    filter_mask: u32,
) -> Result<Vec<u8>, Error> {
    // Even an edge chunk is stored at full chunk size. The size bounds decompression
    // and checks the decoded length when its nonzero byte count fits in usize.
    let expected = expected_chunk_len(&ctx);

    let mut owned: Option<Vec<u8>> = None;
    // Filters are listed in application order. Decoding reverses them.
    // `i` is the filter's forward index and its bit position in `filter_mask`.
    // A set bit means the filter was skipped for this chunk.
    for (i, filter) in filters.iter().enumerate().rev() {
        if i < 32 && (filter_mask >> i) & 1 == 1 {
            continue;
        }
        let input: &[u8] = owned.as_deref().unwrap_or(compressed);
        let next = match filter.id() {
            FILTER_SHUFFLE => shuffle_decompress(input, ctx.element_size.get() as usize)?,
            FILTER_DEFLATE => deflate_decompress(
                scratch,
                input,
                inner_output_cap(expected, filters, filter_mask, i, ctx)?,
            )?,
            FILTER_LZF => crate::decompress_lzf(
                input,
                inner_output_cap(expected, filters, filter_mask, i, ctx)?,
            )?,
            FILTER_FLETCHER32 => fletcher32_verify(input)?,
            FILTER_SCALEOFFSET => crate::decompress_scale_offset(
                input,
                filter.client_data(),
                inner_output_cap(expected, filters, filter_mask, i, ctx)?,
            )?,
            #[cfg(feature = "zfp")]
            FILTER_ZFP => crate::decompress_zfp_filter(
                input,
                filter.client_data(),
                ctx.chunk_dims,
                ctx.element_type,
            )?,
            other => return Err(Error::UnsupportedFilter(other)),
        };
        owned = Some(next);
    }
    let result = owned.unwrap_or_else(|| compressed.to_vec());

    // A valid chunk always decodes to exactly the full chunk size. A mismatch
    // means a corrupt or hostile filter stream. The error prevents silently
    // zero-filling (when short) or dropping (when long) data during chunk
    // assembly, which copies only the in-range overlap.
    if let Some(expected) = expected {
        if result.len() != expected {
            return Err(Error::DataSizeMismatch {
                expected,
                actual: result.len(),
            });
        }
    }
    Ok(result)
}

fn expected_chunk_len(ctx: &ChunkContext<'_>) -> Option<usize> {
    let elems = ctx
        .chunk_dims
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))?;
    let bytes = elems.checked_mul(u64::from(ctx.element_size.get()))?;
    usize::try_from(bytes).ok().filter(|&n| n != 0)
}

fn filter_max_forward_output(
    filter: &impl FilterStep,
    in_size: usize,
    ctx: ChunkContext<'_>,
) -> Result<usize, Error> {
    #[cfg(not(feature = "zfp"))]
    let _ = ctx;

    Ok(match filter.id() {
        // Fletcher32 appends a 4-byte checksum.
        FILTER_FLETCHER32 => in_size.saturating_add(4),
        // A conforming LZF encoder may emit every byte as its own literal run
        // (control byte + literal), so a stream is at most twice its decoded
        // size. Matches are denser. Efficient encoders stay near in_size/32
        // overhead, but the bound must admit any conforming stream.
        FILTER_LZF => in_size.saturating_mul(2),
        // Scale-offset writes a fixed header before the payload and, when the data does not pack
        // smaller, stores it verbatim after that header.
        FILTER_SCALEOFFSET => in_size.saturating_add(crate::SCALE_OFFSET_HEADER_LEN),
        // Deflate can slightly expand incompressible input (zlib "stored" blocks
        // plus framing). The bound exceeds zlib's worst case.
        FILTER_DEFLATE => in_size.saturating_add(in_size / 16).saturating_add(64),
        #[cfg(feature = "zfp")]
        // ZFP encodes full blocks even when the chunk ends with a partial block.
        FILTER_ZFP => {
            crate::zfp::filter_encoded_len(filter.client_data(), ctx.chunk_dims, ctx.element_type)?
        }
        _ => in_size,
    })
}

#[cfg(feature = "deflate")]
const MAX_DEFLATE_EXPANSION: usize = 1032;

fn inner_output_cap(
    expected: Option<usize>,
    filters: &[impl FilterStep],
    filter_mask: u32,
    filter_index: usize,
    ctx: ChunkContext<'_>,
) -> Result<Option<usize>, Error> {
    let Some(mut size) = expected else {
        return Ok(None);
    };
    for (j, f) in filters[..filter_index].iter().enumerate() {
        if j < 32 && (filter_mask >> j) & 1 == 1 {
            continue;
        }
        size = filter_max_forward_output(f, size, ctx)?;
    }
    Ok(Some(size))
}

#[cfg(all(test, feature = "deflate"))]
fn compress_chunk(
    data: &[u8],
    filters: &[impl FilterStep],
    ctx: ChunkContext<'_>,
) -> Result<Vec<u8>, Error> {
    compress_chunk_with(&mut FilterScratch::new(), data, filters, ctx)
}

/// Applies the filters in pipeline order using reusable scratch state.
///
/// See [`decompress_chunk`] for an example of the filter order and chunk context.
///
/// # Errors
///
/// Returns [`Error::UnsupportedFilter`] if a filter has no encoder.
/// An encoder may return another [`Error`] for invalid parameters or input.
pub fn compress_chunk_with(
    scratch: &mut FilterScratch,
    data: &[u8],
    filters: &[impl FilterStep],
    ctx: ChunkContext<'_>,
) -> Result<Vec<u8>, Error> {
    let mut owned: Option<Vec<u8>> = None;
    for filter in filters {
        let input: &[u8] = owned.as_deref().unwrap_or(data);
        let next = match filter.id() {
            FILTER_SHUFFLE => shuffle_compress(input, ctx.element_size.get() as usize)?,
            FILTER_DEFLATE => {
                let level = filter.client_data().first().copied().unwrap_or(6);
                deflate_compress(scratch, input, level)?
            }
            FILTER_LZF => crate::compress_lzf(input),
            FILTER_FLETCHER32 => fletcher32_append(input)?,
            FILTER_SCALEOFFSET => crate::compress_scale_offset(input, filter.client_data())?,
            #[cfg(feature = "zfp")]
            FILTER_ZFP => crate::compress_zfp_filter(
                input,
                filter.client_data(),
                ctx.chunk_dims,
                ctx.element_type,
            )?,
            other => return Err(Error::UnsupportedFilter(other)),
        };
        owned = Some(next);
    }
    Ok(owned.unwrap_or_else(|| data.to_vec()))
}

#[cfg(feature = "deflate")]
fn deflate_corrupt(reason: &str) -> Error {
    Error::FilterError(format!("deflate: {reason}"))
}

#[cfg(feature = "deflate")]
const DEFLATE_GROWTH_FLOOR: usize = 4096;

#[cfg(feature = "deflate")]
fn deflate_decompress(
    scratch: &mut FilterScratch,
    data: &[u8],
    max_output: Option<usize>,
) -> Result<Vec<u8>, Error> {
    use flate2::{FlushDecompress, Status};

    let bomb = || {
        deflate_corrupt(&format!(
            "output exceeds expected chunk size of {} bytes (possible decompression bomb)",
            max_output.unwrap_or(0)
        ))
    };
    let truncated = || deflate_corrupt("stream ended before the chunk was complete");

    // The decoded size follows from the chunk dimensions. Reserve only as much
    // as the stream can produce, so a declared size alone cannot drive allocation.
    let reservation = crate::decode_reservation(max_output, data.len(), MAX_DEFLATE_EXPANSION);
    // One past the limit lets the decoder detect an overlong stream.
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

        let status = decoder
            .decompress_vec(input, &mut out, FlushDecompress::None)
            .map_err(|e| deflate_corrupt(&e.to_string()))?;

        if status == Status::StreamEnd {
            break;
        }

        if out.len() == out.capacity() {
            if ceiling.is_some_and(|cap| out.len() >= cap) {
                return Err(bomb());
            }
            let want = out.capacity().max(DEFLATE_GROWTH_FLOOR);
            out.reserve(match ceiling {
                Some(cap) => want.min(cap - out.len()),
                None => want,
            });
        } else if decoder.total_in() == before_in && out.len() == before_out {
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
) -> Result<Vec<u8>, Error> {
    Err(Error::UnsupportedFilter(FILTER_DEFLATE))
}

#[cfg(feature = "deflate")]
fn deflate_compress(
    scratch: &mut FilterScratch,
    data: &[u8],
    level: u32,
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
    _level: u32,
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

fn shuffle_decompress(data: &[u8], element_size: usize) -> Result<Vec<u8>, Error> {
    if element_size <= 1 {
        return Ok(data.to_vec());
    }
    if !data.len().is_multiple_of(element_size) {
        return Err(Error::FilterError(
            "shuffle: data length not a multiple of element size".into(),
        ));
    }
    let num_elements = data.len() / element_size;
    let mut result = vec![0u8; data.len()];

    // Specialize the common scalar widths so the inner loop unrolls and each
    // element is written as one contiguous store. The generic loop handles
    // unusual widths (compound members, wide types).
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

fn shuffle_compress(data: &[u8], element_size: usize) -> Result<Vec<u8>, Error> {
    if element_size <= 1 {
        return Ok(data.to_vec());
    }
    if !data.len().is_multiple_of(element_size) {
        return Err(Error::FilterError(
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

fn fletcher32_verify(data: &[u8]) -> Result<Vec<u8>, Error> {
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
    Ok(payload.to_vec())
}

fn fletcher32_append(data: &[u8]) -> Result<Vec<u8>, Error> {
    let checksum = fletcher32_compute(data);
    let mut result = data.to_vec();
    result.extend_from_slice(&checksum.to_le_bytes());
    Ok(result)
}

/// The registered identifier for Deflate compression.
pub const FILTER_DEFLATE: u16 = 1;
/// The registered identifier for byte Shuffle.
pub const FILTER_SHUFFLE: u16 = 2;
/// The registered identifier for Fletcher32 checksums.
pub const FILTER_FLETCHER32: u16 = 3;
/// The registered identifier for Scale-Offset compression.
pub const FILTER_SCALEOFFSET: u16 = 6;
/// The registered identifier for LZF compression.
pub const FILTER_LZF: u16 = 32000;
#[cfg(feature = "zfp")]
/// The registered identifier for ZFP compression.
pub const FILTER_ZFP: u16 = 32013;
/// Permits a filter to be omitted if it fails during output.
pub const H5Z_FLAG_OPTIONAL: u16 = 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "deflate")]
    struct FilterDescription {
        filter_id: u16,
        flags: u16,
        client_data: Vec<u32>,
    }

    #[cfg(feature = "deflate")]
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

    #[cfg(feature = "deflate")]
    struct FilterPipeline {
        filters: Vec<FilterDescription>,
    }

    #[rstest::rstest]
    #[cfg(feature = "deflate")]
    #[case((0..256).map(|i| (i % 256) as u8).collect())]
    #[case((0..10).collect())]
    fn deflate_compress_decompress_roundtrip(#[case] data: Vec<u8>) {
        let compressed = deflate_compress(&mut FilterScratch::new(), &data, 6).unwrap();
        assert!(!compressed.is_empty());
        let decompressed =
            deflate_decompress(&mut FilterScratch::new(), &compressed, None).unwrap();
        assert_eq!(decompressed, data);
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
    #[case(2)]
    #[case(3)]
    #[case(4)]
    #[case(6)]
    #[case(7)]
    #[case(8)]
    #[case(16)]
    fn shuffle_roundtrip(#[case] width: usize) {
        let data: Vec<u8> = (0..(width * 50)).map(|i| (i * 31 % 256) as u8).collect();
        let shuffled = shuffle_compress(&data, width).unwrap();
        assert_eq!(shuffled.len(), data.len());
        assert_eq!(shuffle_decompress(&shuffled, width).unwrap(), data);
    }

    #[rstest::rstest]
    #[case(2)]
    #[case(4)]
    #[case(8)]
    #[case(16)]
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
            shuffle_compress(&data, width).unwrap(),
            generic_shuffle(&data, width)
        );
    }

    #[test]
    fn shuffle_known_pattern() {
        let data = vec![0xA0, 0xA1, 0xA2, 0xA3, 0xB0, 0xB1, 0xB2, 0xB3];
        let shuffled = shuffle_compress(&data, 4).unwrap();
        assert_eq!(
            shuffled,
            vec![0xA0, 0xB0, 0xA1, 0xB1, 0xA2, 0xB2, 0xA3, 0xB3]
        );
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

        let stored = shuffle_compress(&data, 8).unwrap();
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

        let stored = deflate_compress(&mut FilterScratch::new(), &data, 6).unwrap();
        let mask = 1u32 << 0; // bit 0 => shuffle (index 0) skipped
        let decoded = decompress_chunk(&stored, &pipeline.filters, ctx, mask).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    #[cfg(feature = "deflate")]
    fn deflate_decompress_rejects_bomb() {
        let huge = vec![0u8; 100_000];
        let compressed = deflate_compress(&mut FilterScratch::new(), &huge, 9).unwrap();
        assert!(compressed.len() < 1024);
        let err =
            deflate_decompress(&mut FilterScratch::new(), &compressed, Some(1024)).unwrap_err();
        assert_eq!(
            err,
            Error::FilterError(
                "deflate: output exceeds expected chunk size of 1024 bytes (possible decompression bomb)".into()
            )
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
        let compressed = deflate_compress(&mut FilterScratch::new(), &data, 6).unwrap();
        assert_eq!(
            deflate_decompress(&mut FilterScratch::new(), &compressed, Some(500)).unwrap(),
            data
        );
    }

    #[rstest::rstest]
    #[cfg(feature = "deflate")]
    #[case(1)]
    #[case(6)]
    #[case(9)]
    fn a_reused_encoder_writes_the_same_bytes_as_a_fresh_one(#[case] level: u32) {
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
            "a reused encoder at level {level} wrote different bytes from a fresh one"
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
        let good = deflate_compress(&mut FilterScratch::new(), &vec![9u8; 2048], 6).unwrap();
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
            Error::FilterError(
                "deflate: output exceeds expected chunk size of 16 bytes (possible decompression bomb)".into()
            )
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
        let complete = deflate_compress(&mut FilterScratch::new(), &data, 6).unwrap();
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

    #[rstest::rstest]
    #[cfg(all(feature = "deflate", feature = "zfp"))]
    #[case::rank1(&[1], 16)]
    #[case::rank2(&[1, 1], 64)]
    #[case::rank3(&[1, 1, 1], 256)]
    #[case::rank4(&[1, 1, 1, 1], 1024)]
    fn zfp_partial_block_inner_deflate_outer_roundtrips(
        #[case] dims: &[u64],
        #[case] encoded_len: usize,
    ) {
        let cd_values = crate::zfp_cd_values_rate(32.0, ZfpElementType::F32, dims).unwrap();
        let filters = [
            FilterDescription {
                filter_id: FILTER_ZFP,
                flags: 0,
                client_data: cd_values,
            },
            FilterDescription {
                filter_id: FILTER_DEFLATE,
                flags: 0,
                client_data: vec![6],
            },
        ];
        let ctx = ChunkContext {
            chunk_dims: dims,
            element_size: NonZeroU32::new(4).unwrap(),
            element_type: Some(ZfpElementType::F32),
            scale_offset_type: None,
        };
        let data = 1.0f32.to_le_bytes();
        let zfp_bytes = crate::compress_zfp_filter(
            &data,
            filters[0].client_data(),
            ctx.chunk_dims,
            ctx.element_type,
        )
        .unwrap();
        assert_eq!(zfp_bytes.len(), encoded_len);
        let stored = compress_chunk_with(&mut FilterScratch::new(), &data, &filters, ctx).unwrap();
        assert_eq!(decompress_chunk(&stored, &filters, ctx, 0).unwrap(), data);
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

    #[test]
    fn lzf_reserves_against_the_stream_not_the_declared_chunk_size() {
        let stored = crate::compress_lzf(&[0u8; 64]);
        let out = crate::decompress_lzf(&stored, Some(u32::MAX as usize)).unwrap();
        assert_eq!(out, [0u8; 64]);
        assert!(
            out.capacity() <= stored.len() * crate::LZF_MAX_EXPANSION,
            "reserved {} bytes for a {}-byte stream",
            out.capacity(),
            stored.len()
        );
    }

    #[test]
    fn a_failed_decode_reports_which_compressor_failed() {
        let lzf = crate::decompress_lzf(&[0x1f], None).unwrap_err();
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
}
