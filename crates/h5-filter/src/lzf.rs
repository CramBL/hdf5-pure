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

use crate::Error;

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
pub const MAX_EXPANSION: usize = MAX_MATCH_LEN / 3;

/// h5py's `H5PY_FILTER_LZF_VERSION` (lzf/lzf_filter.h), `cd_values[0]`.
const H5PY_FILTER_LZF_VERSION: u32 = 4;

/// liblzf's `LZF_VERSION` (0x0105), `cd_values[1]` per h5py's `lzf_set_local`.
const LIBLZF_API_VERSION: u32 = 0x0105;

/// Returns h5py's LZF client data for a chunk's element size and dimensions.
///
/// The entries are the filter version, liblzf version, and chunk byte size.
/// The last entry is zero when the byte size exceeds `u32::MAX`.
///
/// # Examples
///
/// ```
/// use h5_filter::lzf_h5py_cd_values;
///
/// assert_eq!(lzf_h5py_cd_values(2, &[3, 4]), [4, 0x0105, 24]);
/// ```
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
/// The decoder uses [`decode_reservation`](crate::decode_reservation) to size
/// its initial allocation from the encoded input and the cap.
///
/// # Errors
///
/// Returns [`Error::InvalidLzfStream`] if a token is truncated, a match refers
/// to bytes before the output, or decoding exceeds `max_output`.
///
/// # Examples
///
/// ```
/// use h5_filter::decompress_lzf;
///
/// let stream = [4, b'a', b'b', b'c', b'd', b'e', 3 << 5, 4];
/// assert_eq!(decompress_lzf(&stream, Some(10)).unwrap(), b"abcdeabcde");
/// ```
pub fn decompress(input: &[u8], max_output: Option<usize>) -> Result<Vec<u8>, Error> {
    let cap = max_output.unwrap_or(usize::MAX);
    let mut out = Vec::with_capacity(crate::decode_reservation(
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
                return Err(corrupt("output exceeds expected chunk size"));
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
                return Err(corrupt("output exceeds expected chunk size"));
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

    Ok(out)
}

/// Encodes chunk bytes as a raw LZF stream.
///
/// The compressor uses greedy single-probe matching. It may produce more
/// bytes than it receives for incompressible input.
///
/// # Examples
///
/// ```
/// use h5_filter::{compress_lzf, decompress_lzf};
///
/// let input = b"abcdeabcde";
/// let encoded = compress_lzf(input);
/// assert_eq!(decompress_lzf(&encoded, Some(input.len())).unwrap(), input);
/// ```
pub fn compress(input: &[u8]) -> Vec<u8> {
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
            #[expect(clippy::cast_possible_truncation)]
            out.push((n - 1) as u8);
            out.extend_from_slice(&input[i..i + n]);
            i += n;
        }
    }

    let mut out = Vec::with_capacity(input.len() + input.len() / MAX_LITERAL_RUN + 2);
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

                flush_literals(&mut out, input, literal_start, ip);
                let off = distance - 1;
                let encoded_len = len - 2;
                #[expect(clippy::cast_possible_truncation)]
                if encoded_len < 7 {
                    out.push(((encoded_len << 5) | (off >> 8)) as u8);
                } else {
                    out.push(((7 << 5) | (off >> 8)) as u8);
                    out.push((encoded_len - 7) as u8);
                }
                #[expect(clippy::cast_possible_truncation)]
                out.push((off & 0xff) as u8);

                ip += len;
                literal_start = ip;
                continue;
            }
        }
        ip += 1;
    }

    flush_literals(&mut out, input, literal_start, input.len());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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
            Error::InvalidLzfStream("output exceeds expected chunk size")
        );
    }
}
