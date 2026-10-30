//! Encodes and decodes the LZF filter registered by h5py as filter 32000.
//!
//! The LZF filter encodes its input as a raw stream without a container header.
//! A control byte below 32 introduces a literal run; other control bytes
//! introduce a back-reference.
//!
//! The compressor produces streams compatible with liblzf's decoder. The
//! surrounding pipeline handles optional filters and masked chunks.

use alloc::vec;
use alloc::vec::Vec;

use crate::error::Error;

/// Longest literal run one control byte can introduce.
const MAX_LITERAL_RUN: usize = 32;

/// Longest back-reference: `7 + 255` from the length encoding, plus the
/// implicit `+ 2`.
const MAX_MATCH_LEN: usize = 264;

/// Largest encodable back-reference distance (13 offset bits, plus one).
const MAX_MATCH_DISTANCE: usize = 1 << 13;

/// Is the number of slots in the compressor's match-hash table.
///
/// The hash and allocation use the same count. [`MAX_MATCH_DISTANCE`] limits
/// encoded matches independently of the table size.
const HASH_TABLE_SLOTS: usize = 1 << 13;

/// Top bits of the 32-bit multiplicative hash product that select a slot.
const HASH_TABLE_BITS: u32 = HASH_TABLE_SLOTS.trailing_zeros();

/// Is the maximum number of decoded bytes per encoded byte in a conforming LZF
/// stream.
///
/// A three-byte extended back-reference can emit 264 bytes.
pub(crate) const MAX_EXPANSION: usize = MAX_MATCH_LEN / 3;

/// h5py's `H5PY_FILTER_LZF_VERSION` (lzf/lzf_filter.h), `cd_values[0]`.
const H5PY_FILTER_LZF_VERSION: u32 = 4;

/// liblzf's `LZF_VERSION` (0x0105), `cd_values[1]` per h5py's `lzf_set_local`.
const LIBLZF_API_VERSION: u32 = 0x0105;

/// Returns h5py's LZF client data for a chunk's element size and dimensions.
///
/// The entries are the filter version, liblzf version, and chunk byte size.
/// The last entry is zero when the byte size exceeds `u32::MAX`.
pub fn h5py_cd_values(element_size: u32, chunk_dims: &[u64]) -> [u32; 3] {
    let chunk_bytes = chunk_dims
        .iter()
        .try_fold(u64::from(element_size), |acc, &d| acc.checked_mul(d))
        .and_then(|b| u32::try_from(b).ok())
        .unwrap_or(0);
    [H5PY_FILTER_LZF_VERSION, LIBLZF_API_VERSION, chunk_bytes]
}

fn corrupt(reason: &'static str) -> Error {
    Error::InvalidLzfStream(reason)
}

/// Decodes a raw LZF stream into chunk bytes.
///
/// `max_output` bounds the decoded size. Passing `None` leaves it unbounded.
///
/// The initial allocation is at most `max_output` and at most 88 times the length of `input`,
/// and is empty when `max_output` is `None`.
///
/// # Errors
///
/// Returns [`Error::InvalidLzfStream`] if a token is truncated or a match refers to bytes before
/// the output. Returns [`Error::LzfSizeExceeded`] if decoding exceeds `max_output`.
pub fn decompress(input: &[u8], max_output: Option<usize>) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    decompress_into(input, max_output, &mut out)?;
    Ok(out)
}

/// Replaces `out` with decoded bytes from a raw LZF stream.
///
/// Retains the output allocation when it has sufficient capacity. `max_output` bounds the decoded
/// length independently of retained capacity, and `None` leaves it unbounded. An error may leave a
/// decoded prefix in `out`.
///
/// # Errors
///
/// Returns [`Error::InvalidLzfStream`] if a token is truncated or a match refers to bytes before
/// the output. Returns [`Error::LzfSizeExceeded`] if decoding exceeds `max_output`.
pub(crate) fn decompress_into(
    input: &[u8],
    max_output: Option<usize>,
    out: &mut Vec<u8>,
) -> Result<(), Error> {
    let cap = max_output.unwrap_or(usize::MAX);
    out.clear();
    out.reserve_exact(crate::decode_reservation(
        max_output,
        input.len(),
        MAX_EXPANSION,
    ));
    let mut ip = 0;

    while ip < input.len() {
        let ctrl = usize::from(input[ip]);
        ip += 1;

        if ctrl < MAX_LITERAL_RUN {
            let len = ctrl + 1;
            let literals = input
                .get(ip..ip + len)
                .ok_or_else(|| corrupt("truncated literal run"))?;
            if out.len() + len > cap {
                return Err(Error::LzfSizeExceeded {
                    limit: cap,
                    required: out.len() + len,
                });
            }
            out.extend_from_slice(literals);
            ip += len;
        } else {
            let mut len = ctrl >> 5;
            if len == 7 {
                len += usize::from(
                    *input
                        .get(ip)
                        .ok_or_else(|| corrupt("truncated match length"))?,
                );
                ip += 1;
            }
            len += 2;

            let low = usize::from(
                *input
                    .get(ip)
                    .ok_or_else(|| corrupt("truncated match offset"))?,
            );
            ip += 1;
            let distance = (((ctrl & 0x1f) << 8) | low) + 1;

            if distance > out.len() {
                return Err(corrupt("match reaches before start of output"));
            }
            if out.len() + len > cap {
                return Err(Error::LzfSizeExceeded {
                    limit: cap,
                    required: out.len() + len,
                });
            }
            let start = out.len() - distance;
            if distance >= len {
                // Disjoint source and destination: one reservation + memcpy.
                out.extend_from_within(start..start + len);
            } else {
                // Overlapping (RLE-style) match must materialize byte by byte.
                for i in start..start + len {
                    let byte = out[i];
                    out.push(byte);
                }
            }
        }
    }

    Ok(())
}

/// Encodes chunk bytes as a raw LZF stream.
///
/// The compressor uses greedy single-probe matching. It may produce more
/// bytes than it receives for incompressible input.
pub fn compress(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    compress_into(input, &mut out);
    out
}

/// Replaces `out` with an encoded raw LZF stream.
///
/// Retains the output allocation when it has sufficient capacity. Allocates a match table for each
/// call. The encoded output may be larger than `input` for incompressible bytes.
pub(crate) fn compress_into(input: &[u8], out: &mut Vec<u8>) {
    /// Hash of a 3-byte window → slot in the table of `position + 1`.
    fn hash(a: u8, b: u8, c: u8) -> usize {
        let v = (usize::from(a) << 16) | (usize::from(b) << 8) | usize::from(c);
        (v.wrapping_mul(0x9E37_79B1) >> (32 - HASH_TABLE_BITS)) & (HASH_TABLE_SLOTS - 1)
    }

    /// Emit `input[from..to]` as literal runs.
    fn flush_literals(out: &mut Vec<u8>, input: &[u8], from: usize, to: usize) {
        let mut i = from;
        while i < to {
            let n = (to - i).min(MAX_LITERAL_RUN);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "n is in 1..=32, so n - 1 fits in u8"
            )]
            out.push((n - 1) as u8);
            out.extend_from_slice(&input[i..i + n]);
            i += n;
        }
    }

    out.clear();
    out.reserve_exact(input.len() + input.len() / MAX_LITERAL_RUN + 2);
    // Slots hold `position + 1`; 0 marks an empty slot, which is why the
    // candidate check below is `> 0`.
    //
    // A heap allocation keeps the table off the stack for small-stack targets.
    // Slots use `usize` because each holds an input position, which can exceed
    // the maximum match distance and `u16::MAX`.
    let mut table = vec![0_usize; HASH_TABLE_SLOTS];
    let mut ip = 0;
    let mut literal_start = 0;

    while ip + 2 < input.len() {
        let slot = hash(input[ip], input[ip + 1], input[ip + 2]);
        let candidate = table[slot];
        table[slot] = ip + 1;

        if candidate > 0 {
            let match_pos = candidate - 1;
            let distance = ip - match_pos;
            if (1..=MAX_MATCH_DISTANCE).contains(&distance)
                && input[match_pos..match_pos + 3] == input[ip..ip + 3]
            {
                let max_len = (input.len() - ip).min(MAX_MATCH_LEN);
                let mut len = 3;
                while len < max_len && input[match_pos + len] == input[ip + len] {
                    len += 1;
                }

                flush_literals(out, input, literal_start, ip);
                let off = distance - 1;
                let encoded_len = len - 2;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "encoded_len <= 262 and off < 8192, so token bytes fit in u8"
                )]
                if encoded_len < 7 {
                    out.push(((encoded_len << 5) | (off >> 8)) as u8);
                } else {
                    out.push(((7 << 5) | (off >> 8)) as u8);
                    out.push((encoded_len - 7) as u8);
                }
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the mask limits the offset byte to 0..=255"
                )]
                out.push((off & 0xff) as u8);

                ip += len;
                literal_start = ip;
                continue;
            }
        }
        ip += 1;
    }

    flush_literals(out, input, literal_start, input.len());
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::empty(&[], b"")]
    #[case::literal(&[2, b'a', b'b', b'c'], b"abc")]
    #[case::disjoint_match(&[4, b'a', b'b', b'c', b'd', b'e', 0x60, 4], b"abcdeabcde")]
    #[case::overlapping_match(&[0, b'a', 0x20, 0], b"aaaa")]
    #[case::maximum_match(&[0, b'a', 0xe0, 255, 0], &[b'a'; 265])]
    fn decoding_reuses_storage_and_preserves_token_values(
        #[case] stored: &[u8],
        #[case] expected: &[u8],
        #[values(None, Some(1024))] limit: Option<usize>,
    ) {
        let mut output = Vec::with_capacity(4096);
        output.extend_from_slice(&[99; 32]);
        let capacity = output.capacity();
        let pointer = output.as_ptr();
        decompress_into(stored, limit, &mut output).unwrap();
        assert_eq!(output, expected);
        assert_eq!(output.as_ptr(), pointer);
        assert_eq!(output.capacity(), capacity);
    }

    #[rstest]
    #[case::empty(b"", &[])]
    #[case::one_literal(b"a", &[0, b'a'])]
    #[case::three_literals(b"abc", &[2, b'a', b'b', b'c'])]
    #[case::overlapping_match(b"aaaa", &[0, b'a', 0x20, 0])]
    #[case::disjoint_match(b"abcdeabcde", &[4, b'a', b'b', b'c', b'd', b'e', 0x60, 4])]
    #[case::maximum_match(&[b'a'; 265], &[0, b'a', 0xe0, 255, 0])]
    #[case::match_then_literal(&[b'a'; 266], &[0, b'a', 0xe0, 255, 0, 0, b'a'])]
    fn encoding_reuses_storage_and_preserves_encoded_bytes(
        #[case] input: &[u8],
        #[case] expected: &[u8],
    ) {
        let mut output = Vec::with_capacity(4096);
        output.extend_from_slice(&[99; 32]);
        let pointer = output.as_ptr();
        let capacity = output.capacity();
        compress_into(input, &mut output);
        assert_eq!(output, expected);
        assert_eq!(output.as_ptr(), pointer);
        assert_eq!(output.capacity(), capacity);
        assert_eq!(compress(input), expected);
    }

    #[rstest]
    #[case::literal_truncated(&[0], None, Error::InvalidLzfStream("truncated literal run"))]
    #[case::literal_truncated_before_limit(&[31, b'a'], Some(0), Error::InvalidLzfStream("truncated literal run"))]
    #[case::match_length_truncated(&[0xe0], None, Error::InvalidLzfStream("truncated match length"))]
    #[case::match_offset_truncated(&[0x20], None, Error::InvalidLzfStream("truncated match offset"))]
    #[case::extended_offset_truncated(&[0xe0, 0], None, Error::InvalidLzfStream("truncated match offset"))]
    #[case::match_before_output(&[0x20, 0], None, Error::InvalidLzfStream("match reaches before start of output"))]
    #[case::match_before_output_before_limit(&[0x20, 0], Some(0), Error::InvalidLzfStream("match reaches before start of output"))]
    #[case::literal_limit(&[2, b'a', b'b', b'c'], Some(2), Error::LzfSizeExceeded { limit: 2, required: 3 })]
    #[case::match_limit(&[0, b'a', 0x20, 0], Some(3), Error::LzfSizeExceeded { limit: 3, required: 4 })]
    #[case::extended_match_limit(&[0, b'a', 0xe0, 255, 0], Some(264), Error::LzfSizeExceeded { limit: 264, required: 265 })]
    #[case::failure_after_literal(&[0, b'a', 0x20], None, Error::InvalidLzfStream("truncated match offset"))]
    fn reused_storage_preserves_decode_errors_and_resets_on_the_next_operation(
        #[case] input: &[u8],
        #[case] limit: Option<usize>,
        #[case] expected: Error,
    ) {
        let mut output = Vec::with_capacity(4096);
        output.extend_from_slice(&[99; 32]);
        let pointer = output.as_ptr();
        assert_eq!(decompress_into(input, limit, &mut output), Err(expected));
        decompress_into(&[1, b'x', b'y'], Some(2), &mut output).unwrap();
        assert_eq!(output, b"xy");
        assert_eq!(output.as_ptr(), pointer);
        compress_into(b"z", &mut output);
        assert_eq!(output, [0, b'z']);
        assert_eq!(output.as_ptr(), pointer);
    }

    fn round_trip(data: &[u8]) {
        let compressed = compress(data);
        let decompressed = decompress(&compressed, Some(data.len())).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn round_trips() {
        round_trip(b"");
        round_trip(b"a");
        round_trip(b"hello world hello world hello world");
        // Long RLE run (overlapping matches).
        round_trip(&[0_u8; 10_000]);
        round_trip(&(0..=255).cycle().take(70_000).collect::<Vec<u8>>());
        // Incompressible pseudo-random bytes (xorshift).
        let mut x = 0x2545_F491_4F6C_DD1D_u64;
        let noise: Vec<u8> = (0..50_000)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x & 0xff) as u8
            })
            .collect();
        round_trip(&noise);
    }

    #[test]
    fn known_stream_decodes() {
        // 5 literals, then a distance-5 length-5 match ("abcdeabcde"): the
        // control byte carries length 3+2 and offset high bits 0, then the
        // offset low byte 4 (+1 = distance 5).
        let stream = [4, b'a', b'b', b'c', b'd', b'e', 3 << 5, 4];
        assert_eq!(decompress(&stream, None).unwrap(), b"abcdeabcde");
    }

    #[test]
    fn worst_case_expansion_stream_decodes() {
        // One literal token per byte makes the encoded stream twice the decoded size.
        let stream: Vec<u8> = (0..=255u8).flat_map(|b| [0, b]).collect();
        let expected: Vec<u8> = (0..=255).collect();
        assert_eq!(stream.len(), 2 * expected.len());
        assert_eq!(decompress(&stream, Some(expected.len())).unwrap(), expected);
    }

    /// Compression succeeds with a stack smaller than the match table.
    ///
    /// The test runs in a separate thread because a stack overflow aborts the
    /// process instead of returning an error.
    #[cfg(feature = "std")]
    #[test]
    fn compresses_on_a_stack_smaller_than_the_table() {
        let stack = HASH_TABLE_SLOTS * size_of::<usize>() * 3 / 4;
        let data: Vec<u8> = (0..=255).cycle().take(70_000).collect();
        let expected = data.clone();
        let out = std::thread::Builder::new()
            .stack_size(stack)
            .spawn(move || compress(&data))
            .expect("spawn small-stack thread")
            .join()
            .expect("compress must not overflow a stack smaller than the table");
        assert_eq!(decompress(&out, Some(expected.len())).unwrap(), expected);
    }

    #[test]
    fn corrupt_streams_error() {
        // Literal run past end of input.
        assert_eq!(
            decompress(&[10, b'x'], None).unwrap_err(),
            Error::InvalidLzfStream("truncated literal run")
        );
        // Match before start of output.
        assert_eq!(
            decompress(&[(3 << 5), 200], None).unwrap_err(),
            Error::InvalidLzfStream("match reaches before start of output")
        );
        // Output larger than cap.
        assert_eq!(
            decompress(&[4, b'a', b'b', b'c', b'd', b'e'], Some(3)).unwrap_err(),
            Error::LzfSizeExceeded {
                limit: 3,
                required: 5
            }
        );
    }

    #[test]
    fn h5py_cd_values_multiply_multidimensional_chunk_extents() {
        assert_eq!(h5py_cd_values(2, &[3, 4]), [4, 0x0105, 24]);
    }
}
