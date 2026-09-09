//! LZF filter (H5Z filter id 32000), as registered by h5py.
//!
//! LZF is an LZ77-family byte codec (liblzf, Marc Lehmann): a stream of control
//! bytes where `ctrl < 32` introduces a literal run of `ctrl + 1` bytes, and
//! anything else a back-reference of `(ctrl >> 5) + 2` bytes (`7` adds an
//! extension byte) at distance `(((ctrl & 0x1f) << 8) | low) + 1`. There is no
//! container or per-block header: h5py stores the raw codec stream per chunk.
//!
//! The compressor is format-compatible with liblzf's decoder but makes no
//! attempt to reproduce liblzf's exact byte output — any conforming stream is
//! valid. h5py registers the filter as *optional*, so on read a chunk h5py
//! stored raw (its filter-mask bit set) must be tolerated. This crate's writer
//! records the same optional flag — a later writer, h5py included, needs it to
//! store an incompressible chunk raw rather than fail — but never exercises it
//! itself: it applies LZF to every chunk and accepts the grown stream that
//! incompressible input produces.

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

use crate::error::FormatError;

/// Longest literal run one control byte can introduce.
const MAX_LITERAL_RUN: usize = 32;

/// Longest back-reference: `7 + 255` from the length encoding, plus the
/// implicit `+ 2`.
const MAX_MATCH_LEN: usize = 264;

/// Largest encodable back-reference distance (13 offset bits, plus one).
const MAX_MATCH_DISTANCE: usize = 1 << 13;

/// Slots in the compressor's match-hash table. Independent of
/// [`MAX_MATCH_DISTANCE`], which the stream format fixes; this is only a
/// speed/ratio choice, but the hash and the allocation must agree on it or a
/// hashed slot indexes past the table.
const HASH_TABLE_SLOTS: usize = 1 << 13;

/// Top bits of the 32-bit multiplicative hash product that select a slot.
const HASH_TABLE_BITS: u32 = HASH_TABLE_SLOTS.trailing_zeros();

/// Largest number of bytes a conforming LZF stream can decode to per byte of
/// input. The densest token is a three-byte extended back-reference emitting
/// [`MAX_MATCH_LEN`]; every other token is denser in the input, so the ratio
/// bounds the stream as a whole.
pub(crate) const MAX_EXPANSION: usize = MAX_MATCH_LEN / 3;

/// h5py's `H5PY_FILTER_LZF_VERSION` (lzf/lzf_filter.h), `cd_values[0]`.
const H5PY_FILTER_LZF_VERSION: u32 = 4;

/// liblzf's `LZF_VERSION` (0x0105), `cd_values[1]` per h5py's `lzf_set_local`.
const LIBLZF_API_VERSION: u32 = 0x0105;

/// cd_values as h5py's `lzf_set_local` records them: `[filter version, liblzf
/// version, chunk bytes]`. h5py's decoder reads only `[2]`, as a buffer-size
/// hint, and treats 0 as "no hint" — so an overflowing chunk size degrades to
/// no hint rather than an error.
pub(crate) fn h5py_cd_values(element_size: u32, chunk_dims: &[u64]) -> [u32; 3] {
    let chunk_bytes = chunk_dims
        .iter()
        .try_fold(u64::from(element_size), |acc, &d| acc.checked_mul(d))
        .and_then(|b| u32::try_from(b).ok())
        .unwrap_or(0);
    [H5PY_FILTER_LZF_VERSION, LIBLZF_API_VERSION, chunk_bytes]
}

fn corrupt(reason: &str) -> FormatError {
    FormatError::FilterError(format!("lzf: {reason}"))
}

/// Decompress an LZF stream.
///
/// `max_output` is the decoded size cap (the expected chunk size pushed through
/// the surviving inner filters); exceeding it means a corrupt or hostile
/// stream, and `None` (unknown chunk size) leaves the output uncapped.
///
/// The cap bounds the output but does not by itself size the allocation: see
/// [`decode_reservation`](crate::filters::decode_reservation) for why a size the
/// file merely claims is not enough to reserve on.
pub(crate) fn decompress(input: &[u8], max_output: Option<usize>) -> Result<Vec<u8>, FormatError> {
    let cap = max_output.unwrap_or(usize::MAX);
    let mut out = Vec::with_capacity(crate::filters::decode_reservation(
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

/// Compress `input` into an LZF stream (greedy, single-probe hash matching).
pub(crate) fn compress(input: &[u8]) -> Vec<u8> {
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
    // The table lives on the heap, not in a `[usize; HASH_TABLE_SLOTS]` local.
    // As a local it is a 64 KiB stack frame (32 KiB where pointers are 32-bit),
    // zeroed behind a stack probe on every call, and was the only frame over
    // 8 KiB in the whole crate — more than a bare-metal `no_std` target, which
    // this crate supports, is likely to have in total. `alloc` is unconditional
    // here, so this costs no feature split. Measured with both forms in one
    // process: within noise from 64 KiB chunks up, +4% at 8 KiB, +15% (an
    // absolute 0.35 us) at 1 KiB ([#234]).
    //
    // The slot type must index the whole input, not the match distance. A
    // narrower table has been proposed and measured: `u16` slots wrap on any
    // chunk over 64 KiB and silently destroy compression there (1 MiB of RLE
    // data went from 14,356 bytes out to 1,013,048, with no error), and the
    // speed case does not hold either — `u16` wins 26% at 1 KiB, an absolute
    // 0.6 us, and loses 6-78% across the 6-16 KiB band on incompressible
    // input. Keep `usize`.
    //
    // [#234]: https://github.com/CramBL/hdf5-pure/issues/234
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

// Byte-level crosscheck of this codec against h5py-produced fixtures. Lives
// in-crate (rather than tests/) because it exercises the internal `compress`/
// `decompress` entry points; std-gated (unlike zfp's) because this module also
// compiles under no_std, where the fixture-reading `std::fs` would not resolve.
#[cfg(all(test, feature = "std"))]
#[path = "lzf_crosscheck.rs"]
mod lzf_crosscheck;

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
        // A conforming encoder may emit every byte as its own literal run,
        // doubling the stream relative to its decoded size. This checks only
        // that the decoder accepts such a stream; the matching 2x cap in
        // `filters::filter_max_forward_output`, which this test does not call,
        // is pinned by `filters::tests::foreign_lzf_inner_deflate_outer_roundtrips`.
        let stream: Vec<u8> = (0..=255u8).flat_map(|b| [0, b]).collect();
        let expected: Vec<u8> = (0..=255).collect();
        assert_eq!(stream.len(), 2 * expected.len());
        assert_eq!(decompress(&stream, Some(expected.len())).unwrap(), expected);
    }

    /// The match-hash table is heap-allocated so that a caller on a small stack
    /// — the bare-metal `no_std` target this crate advertises — can compress at
    /// all. Stated as the rule rather than a fixed number: the thread gets less
    /// stack than the table itself would occupy at this pointer width, so a
    /// table that moved back into a local could not fit however wide `usize` is.
    ///
    /// A regression here overflows the thread's stack, which aborts the whole
    /// test binary rather than failing this one test. That is the loudest form
    /// the invariant has; there is no portable way to assert a frame size.
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
        assert!(decompress(&[10, b'x'], None).is_err());
        // Match before start of output.
        assert!(decompress(&[(3 << 5), 200], None).is_err());
        // Output larger than cap.
        assert!(decompress(&[4, b'a', b'b', b'c', b'd', b'e'], Some(3)).is_err());
    }
}
