//! Fixed-rate ZFP encoding and decoding for floating-point and signed integer chunks.
//!
//! The codec supports `f32`, `f64`, `i32`, and `i64` chunks with one to four dimensions.
//! Each block occupies a fixed number of bits set by the rate. Floating-point values
//! should be finite when their decoded values matter.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};

use crate::Error;

/// The scalar type of a ZFP chunk.
///
/// H5Z-ZFP records this type in the filter's client data.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZfpElementType {
    /// A 32-bit floating-point scalar.
    F32,
    /// A 64-bit floating-point scalar.
    F64,
    /// A 32-bit signed integer scalar.
    I32,
    /// A 64-bit signed integer scalar.
    I64,
}

impl ZfpElementType {
    fn byte_width(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
        }
    }
}

/// Number of values per ZFP block (1D).
const BLOCK_SIZE: usize = 4;

/// Exponent bits in the block-float header, per scalar type.
const EBITS_F32: u32 = 8;
const EBITS_F64: u32 = 11;

/// Negabinary masks (per integer width).
const NBMASK_U32: u32 = 0xAAAA_AAAA;
const NBMASK_U64: u64 = 0xAAAA_AAAA_AAAA_AAAA;

// ---------------------------------------------------------------------------
// Bit-stream I/O
// ---------------------------------------------------------------------------

/// A bit-level writer that packs bits LSB-first into 64-bit little-endian words.
struct BitWriter {
    /// Completed 64-bit words (stored as raw little-endian bytes).
    buf: Vec<u8>,
    /// Current partial word being filled.
    word: u64,
    /// Number of valid bits in `word` (0..64).
    bits: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            word: 0,
            bits: 0,
        }
    }

    /// Like `new`, but sizes the output buffer for a known-upper-bound
    /// output length. Each `compress_Nd` knows `total_bits` up front, so
    /// passing it here avoids log₂(N) amortized-growth reallocations
    /// inside the encoder's hot loop.
    fn with_capacity(bytes: usize) -> Self {
        Self {
            buf: Vec::with_capacity(bytes),
            word: 0,
            bits: 0,
        }
    }

    /// Write the `n` least-significant bits of `value`. Bits above `n` in
    /// `value` are ignored (masked off) so the caller may pass a packed
    /// bit-plane register without pre-masking.
    #[inline]
    fn write(&mut self, n: u32, value: u64) {
        debug_assert!(n <= 64);
        if n == 0 {
            return;
        }
        let masked = if n == 64 {
            value
        } else {
            value & ((1u64 << n) - 1)
        };
        self.word |= masked << self.bits;
        self.bits += n;
        if self.bits >= 64 {
            self.buf.extend_from_slice(&self.word.to_le_bytes());
            self.bits -= 64;
            // Carry: the high bits of `masked` that didn't fit in the
            // flushed word. When bits was 0 and n == 64 we've already
            // flushed everything, so a fresh word starts at 0.
            self.word = if self.bits > 0 {
                masked >> (n - self.bits)
            } else {
                0
            };
        }
    }

    /// Write a single bit. Specialized body (not routed through `write(1, …)`)
    /// because the unary run-length encoder in `encode_few_ints` /
    /// `encode_many_ints` calls this per bit on dense planes; dispatching
    /// through the `write(n, _)` path adds a mask/shift setup per call that
    /// the unary path doesn't need.
    #[inline]
    fn write_bit(&mut self, bit: bool) {
        self.word |= (bit as u64) << self.bits;
        self.bits += 1;
        if self.bits == 64 {
            self.buf.extend_from_slice(&self.word.to_le_bytes());
            self.word = 0;
            self.bits = 0;
        }
    }

    /// Flush any partial word (zero-padded to 64 bits) and return the buffer.
    fn finish(mut self) -> Vec<u8> {
        if self.bits > 0 {
            self.buf.extend_from_slice(&self.word.to_le_bytes());
        }
        self.buf
    }

    /// Current position in bits.
    fn position(&self) -> usize {
        self.buf.len() * 8 + self.bits as usize
    }
}

/// A bit-level reader that unpacks bits LSB-first from 64-bit LE words.
struct BitReader<'a> {
    data: &'a [u8],
    /// Byte offset of the next word to load.
    byte_pos: usize,
    /// Current word.
    word: u64,
    /// Number of unconsumed bits remaining in `word`.
    bits: u32,
    consumed_bits: usize,
    exhausted: bool,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut r = Self {
            data,
            byte_pos: 0,
            word: 0,
            bits: 0,
            consumed_bits: 0,
            exhausted: false,
        };
        r.refill();
        r
    }

    /// Load the next 64-bit word from `data`.
    fn refill(&mut self) {
        if self.byte_pos + 8 <= self.data.len() {
            let bytes: [u8; 8] = self.data[self.byte_pos..self.byte_pos + 8]
                .try_into()
                .unwrap();
            self.word = u64::from_le_bytes(bytes);
            self.byte_pos += 8;
            self.bits = 64;
        } else {
            // Partial or exhausted -- zero-fill the remaining bytes. Clamp the
            // start index to the buffer length: a prior partial refill bumps
            // `byte_pos` past the end (the `+= 8` below is unconditional), so on
            // the next call `byte_pos > len`, and slicing `data[byte_pos..byte_pos]`
            // would panic on the `start <= len` bounds check even though the range
            // is empty. Clamping keeps the slice in bounds and yields a zero word.
            let mut bytes = [0u8; 8];
            let start = self.byte_pos.min(self.data.len());
            let avail = self.data.len() - start;
            bytes[..avail].copy_from_slice(&self.data[start..start + avail]);
            self.word = u64::from_le_bytes(bytes);
            self.byte_pos += 8;
            self.bits = 64;
        }
    }

    /// Read `n` bits and return them right-justified.
    #[inline]
    fn read(&mut self, n: u32) -> u64 {
        debug_assert!(n <= 64);
        if n == 0 {
            return 0;
        }
        self.consumed_bits = self.consumed_bits.saturating_add(n as usize);
        self.exhausted |= self.consumed_bits > self.data.len().saturating_mul(8);
        if n <= self.bits {
            let mask = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
            let val = self.word & mask;
            // Shifting a u64 right by 64 overflows (panics in debug). When the
            // whole word is consumed, `bits` reaches 0 and we refill, so the
            // residual word is never observed — just zero it.
            self.word = if n == 64 { 0 } else { self.word >> n };
            self.bits -= n;
            if self.bits == 0 {
                self.refill();
            }
            val
        } else {
            // Need bits from next word
            let lo_bits = self.bits;
            let lo = self.word; // all remaining bits
            self.refill();
            let hi_bits = n - lo_bits;
            let hi_mask = if hi_bits == 64 {
                u64::MAX
            } else {
                (1u64 << hi_bits) - 1
            };
            let hi = self.word & hi_mask;
            self.word >>= hi_bits;
            self.bits -= hi_bits;
            if self.bits == 0 {
                self.refill();
            }
            lo | (hi << lo_bits)
        }
    }

    /// Read a single bit.
    #[inline]
    fn read_bit(&mut self) -> bool {
        self.read(1) != 0
    }

    fn finish_decode(&self) -> Result<(), Error> {
        if self.exhausted {
            return Err(Error::TruncatedZfpStream {
                expected: self.consumed_bits.div_ceil(8),
                actual: self.data.len(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Block-floating-point cast (f32 <-> i32)
// ---------------------------------------------------------------------------

/// IEEE 754 f32 exponent bias.
const EBIAS_F32: i32 = 127;

/// Reference `pad_block` for a partial 1D block of width `n` (0 ≤ n ≤ 4).
///
/// For `n = 0` the block becomes all-zero. Otherwise the tail is filled by
/// the fall-through cascade: `p[1] = p[0]; p[2] = p[1]; p[3] = p[0]`
/// starting from `case n`. This matches the symmetric reflection used by
/// LLNL/zfp so that partial blocks encode identically under crosscheck.
macro_rules! impl_pad_partial {
    ($name:ident, $scalar:ty, $zero:expr) => {
        fn $name(block: &mut [$scalar; BLOCK_SIZE], n: usize) {
            match n {
                0 => {
                    block[0] = $zero;
                    block[1] = $zero;
                    block[2] = $zero;
                    block[3] = $zero;
                }
                1 => {
                    block[1] = block[0];
                    block[2] = block[1];
                    block[3] = block[0];
                }
                2 => {
                    block[2] = block[1];
                    block[3] = block[0];
                }
                3 => {
                    block[3] = block[0];
                }
                _ => {}
            }
        }
    };
}

impl_pad_partial!(pad_partial_block_f32, f32, 0.0f32);
impl_pad_partial!(pad_partial_block_f64, f64, 0.0f64);
impl_pad_partial!(pad_partial_block_i32, i32, 0i32);
impl_pad_partial!(pad_partial_block_i64, i64, 0i64);

/// Strided variant of `pad_block` used to fill partial N-D blocks. For a
/// 2D block we pad along x within each row (stride 1) and then along y
/// across columns (stride 4); 3D / 4D are analogous.
macro_rules! impl_pad_strided {
    ($name:ident, $scalar:ty, $zero:expr) => {
        fn $name(block: &mut [$scalar], base: usize, n: usize, stride: usize) {
            match n {
                0 => {
                    block[base] = $zero;
                    block[base + stride] = $zero;
                    block[base + 2 * stride] = $zero;
                    block[base + 3 * stride] = $zero;
                }
                1 => {
                    block[base + stride] = block[base];
                    block[base + 2 * stride] = block[base + stride];
                    block[base + 3 * stride] = block[base];
                }
                2 => {
                    block[base + 2 * stride] = block[base + stride];
                    block[base + 3 * stride] = block[base];
                }
                3 => {
                    block[base + 3 * stride] = block[base];
                }
                _ => {}
            }
        }
    };
}
impl_pad_strided!(pad_strided_f32, f32, 0.0f32);
impl_pad_strided!(pad_strided_f64, f64, 0.0f64);
impl_pad_strided!(pad_strided_i32, i32, 0i32);
impl_pad_strided!(pad_strided_i64, i64, 0i64);

/// Convert a block of 4 f32 values to block-floating-point i32 representation,
/// matching LLNL/zfp's `fwd_cast`.
///
/// The reference defines:
///
///   emax = max frexp-exponent across the block (0.5 ≤ |m| < 1 convention)
///   `iblock[i] = (Int)(fblock[i] * 2^(30 - emax))`
///
/// Using the IEEE biased exponent `b` of the largest-magnitude value,
/// frexp's exponent is `b - (EBIAS - 1) = b - 126`, so the scale becomes
/// `2^(156 - b)`. The returned header value is `emax_biased = frexp_emax +
/// EBIAS = b + 1`, which is what the reference stream writes (as the high
/// bits of `2*e + 1`).
///
/// If all values are zero or subnormal the reference's `exponent()` returns
/// `-EBIAS`, which we signal here as `emax_biased = 0` — the encoder then
/// emits a single zero-bit empty-block marker.
fn fwd_cast_f32(vals: &[f32; BLOCK_SIZE]) -> (u32, [i32; BLOCK_SIZE]) {
    let mut out = [0i32; BLOCK_SIZE];
    let e = fwd_cast_f32_slice(vals, &mut out);
    (e, out)
}

/// N-D variant: fills `coeffs` in-place and returns `emax_biased`.
fn fwd_cast_f32_slice(vals: &[f32], coeffs: &mut [i32]) -> u32 {
    debug_assert_eq!(vals.len(), coeffs.len());
    let mut ieee_max: i32 = 0;
    for &v in vals {
        #[expect(
            clippy::cast_possible_wrap,
            reason = "8-bit IEEE-754 f32 exponent field (0..=255) cast to i32 never wraps"
        )]
        let e = ((v.to_bits() >> 23) & 0xFF) as i32;
        if e > ieee_max {
            ieee_max = e;
        }
    }
    if ieee_max == 0 {
        for c in coeffs.iter_mut() {
            *c = 0;
        }
        return 0;
    }
    let emax_biased = (ieee_max + 1) as u32;
    let scale_exp = 30 + EBIAS_F32 - 1 - ieee_max;
    let scale = pow2_f64(scale_exp);
    for i in 0..vals.len() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "scaled block coefficient is bounded to |x| <= 2^30 by construction, fits i32"
        )]
        {
            coeffs[i] = (vals[i] as f64 * scale) as i32;
        }
    }
    emax_biased
}

/// Inverse block-floating-point cast matching LLNL/zfp's `inv_cast`:
///
///   value = coefficient * 2^(emax - 30)   (frexp-exponent convention)
///
/// Given the stored header value `emax_biased = frexp_emax + EBIAS`, the
/// dequantization scale is `2^(emax_biased - EBIAS - 30)`.
fn inv_cast_f32(emax_biased: u32, coeffs: &[i32; BLOCK_SIZE]) -> [f32; BLOCK_SIZE] {
    let mut result = [0.0f32; BLOCK_SIZE];
    inv_cast_f32_slice(emax_biased, coeffs, &mut result);
    result
}

fn inv_cast_f32_slice(emax_biased: u32, coeffs: &[i32], out: &mut [f32]) {
    debug_assert_eq!(coeffs.len(), out.len());
    for v in out.iter_mut() {
        *v = 0.0;
    }
    if emax_biased == 0 {
        return;
    }
    #[expect(
        clippy::cast_possible_wrap,
        reason = "emax_biased is a stored 8-bit f32 exponent (0..=255); cast to i32 never wraps"
    )]
    let exp = emax_biased as i32 - EBIAS_F32 - 30;
    let scale = pow2_f64(exp);
    for (i, &c) in coeffs.iter().enumerate() {
        if c == 0 {
            continue;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "dequantized value is the original f32 magnitude, in range by construction"
        )]
        {
            out[i] = ((c as f64) * scale) as f32;
        }
    }
}

/// IEEE 754 f64 exponent bias.
const EBIAS_F64: i32 = 1023;

/// f64 analog of `fwd_cast_f32` — produces i64 coefficients scaled such that
/// `|coef| ≤ 2^62`, and the header value `emax_biased = frexp_emax + EBIAS`
/// (i.e., the raw IEEE biased exponent of the largest-magnitude input +1).
fn fwd_cast_f64(vals: &[f64; BLOCK_SIZE]) -> (u64, [i64; BLOCK_SIZE]) {
    let mut out = [0i64; BLOCK_SIZE];
    let e = fwd_cast_f64_slice(vals, &mut out);
    (e, out)
}

fn fwd_cast_f64_slice(vals: &[f64], coeffs: &mut [i64]) -> u64 {
    debug_assert_eq!(vals.len(), coeffs.len());
    let mut ieee_max: i32 = 0;
    for &v in vals {
        let e = ((v.to_bits() >> 52) & 0x7FF) as i32;
        if e > ieee_max {
            ieee_max = e;
        }
    }
    if ieee_max == 0 {
        for c in coeffs.iter_mut() {
            *c = 0;
        }
        return 0;
    }
    let emax_biased = (ieee_max + 1) as u64;
    let scale = pow2_f64_wide(1084 - ieee_max);
    for i in 0..vals.len() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "scaled block coefficient is bounded to |x| <= 2^62 by construction, fits i64"
        )]
        {
            coeffs[i] = (vals[i] * scale) as i64;
        }
    }
    emax_biased
}

/// f64 inverse cast — `value = coef * 2^(emax_biased - EBIAS - 62)`.
fn inv_cast_f64(emax_biased: u64, coeffs: &[i64; BLOCK_SIZE]) -> [f64; BLOCK_SIZE] {
    let mut result = [0.0f64; BLOCK_SIZE];
    inv_cast_f64_slice(emax_biased, coeffs, &mut result);
    result
}

fn inv_cast_f64_slice(emax_biased: u64, coeffs: &[i64], out: &mut [f64]) {
    debug_assert_eq!(coeffs.len(), out.len());
    for v in out.iter_mut() {
        *v = 0.0;
    }
    if emax_biased == 0 {
        return;
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "emax_biased is a stored 11-bit f64 exponent (0..=2047), so it fits i32"
    )]
    let exp = emax_biased as i32 - EBIAS_F64 - 62;
    let scale = pow2_f64_wide(exp);
    for (i, &c) in coeffs.iter().enumerate() {
        if c == 0 {
            continue;
        }
        out[i] = (c as f64) * scale;
    }
}

/// Like `pow2_f64` but handles `exp` outside `[-1022, 1023]` by splitting
/// into multiple steps. Needed for f64 quantization where the scale can
/// exceed the normal-range upper bound (up to ~2^1083).
#[inline]
fn pow2_f64_wide(exp: i32) -> f64 {
    if (-1022..=1023).contains(&exp) {
        return pow2_f64(exp);
    }
    let mut remaining = exp;
    let mut acc = 1.0f64;
    while remaining > 1023 {
        acc *= pow2_f64(1023);
        remaining -= 1023;
    }
    while remaining < -1022 {
        acc *= pow2_f64(-1022);
        remaining += 1022;
    }
    acc * pow2_f64(remaining)
}

/// `2.0_f64.powi(exp)` without `std` / `libm`.
///
/// Builds the f64 directly from its IEEE 754 biased-exponent field. The ZFP
/// f32 codec uses `exp = emax - 127 - 30` where `emax ∈ [0, 255]`, giving
/// `exp ∈ [-157, 98]` — comfortably inside the normal f64 range — so we do
/// not need a subnormal/overflow fallback path.
#[inline]
fn pow2_f64(exp: i32) -> f64 {
    debug_assert!(
        (-1022..=1023).contains(&exp),
        "pow2_f64 out of normal range"
    );
    let biased = (exp + 1023) as u64;
    f64::from_bits(biased << 52)
}

// ---------------------------------------------------------------------------
// Lifting transform (1D, 4 values). Matches reference `fwd_lift` / `inv_lift`
// (`src/template/encode.c`, `decode.c`) — identical instruction sequence
// except for the integer width. Shifts are arithmetic (sign-preserving), so
// i32/i64 share the source.
// ---------------------------------------------------------------------------

macro_rules! impl_lift {
    ($fwd:ident, $inv:ident, $int:ty) => {
        #[allow(clippy::many_single_char_names)]
        fn $fwd(p: &mut [$int; BLOCK_SIZE]) {
            let mut x = p[0];
            let mut y = p[1];
            let mut z = p[2];
            let mut w = p[3];
            x = x.wrapping_add(w);
            x >>= 1;
            w = w.wrapping_sub(x);
            z = z.wrapping_add(y);
            z >>= 1;
            y = y.wrapping_sub(z);
            x = x.wrapping_add(z);
            x >>= 1;
            z = z.wrapping_sub(x);
            w = w.wrapping_add(y);
            w >>= 1;
            y = y.wrapping_sub(w);
            w = w.wrapping_add(y >> 1);
            y = y.wrapping_sub(w >> 1);
            p[0] = x;
            p[1] = y;
            p[2] = z;
            p[3] = w;
        }
        #[allow(clippy::many_single_char_names)]
        fn $inv(p: &mut [$int; BLOCK_SIZE]) {
            let mut x = p[0];
            let mut y = p[1];
            let mut z = p[2];
            let mut w = p[3];
            y = y.wrapping_add(w >> 1);
            w = w.wrapping_sub(y >> 1);
            y = y.wrapping_add(w);
            w <<= 1;
            w = w.wrapping_sub(y);
            z = z.wrapping_add(x);
            x <<= 1;
            x = x.wrapping_sub(z);
            y = y.wrapping_add(z);
            z <<= 1;
            z = z.wrapping_sub(y);
            w = w.wrapping_add(x);
            x <<= 1;
            x = x.wrapping_sub(w);
            p[0] = x;
            p[1] = y;
            p[2] = z;
            p[3] = w;
        }
    };
}

impl_lift!(fwd_lift_i32, inv_lift_i32, i32);
impl_lift!(fwd_lift_i64, inv_lift_i64, i64);

// ---------------------------------------------------------------------------
// N-D multi-axis lift. For 2D/3D/4D the block is 4^N elements arranged
// row-major with strides (1, 4, 16, 64, ...). `fwd_xform` applies `fwd_lift`
// along each axis in turn; `inv_xform` reverses them.
// ---------------------------------------------------------------------------

#[inline]
fn fwd_lift_axis_i32(block: &mut [i32], base: usize, stride: usize) {
    let mut v = [
        block[base],
        block[base + stride],
        block[base + 2 * stride],
        block[base + 3 * stride],
    ];
    fwd_lift_i32(&mut v);
    block[base] = v[0];
    block[base + stride] = v[1];
    block[base + 2 * stride] = v[2];
    block[base + 3 * stride] = v[3];
}
#[inline]
fn inv_lift_axis_i32(block: &mut [i32], base: usize, stride: usize) {
    let mut v = [
        block[base],
        block[base + stride],
        block[base + 2 * stride],
        block[base + 3 * stride],
    ];
    inv_lift_i32(&mut v);
    block[base] = v[0];
    block[base + stride] = v[1];
    block[base + 2 * stride] = v[2];
    block[base + 3 * stride] = v[3];
}
#[inline]
fn fwd_lift_axis_i64(block: &mut [i64], base: usize, stride: usize) {
    let mut v = [
        block[base],
        block[base + stride],
        block[base + 2 * stride],
        block[base + 3 * stride],
    ];
    fwd_lift_i64(&mut v);
    block[base] = v[0];
    block[base + stride] = v[1];
    block[base + 2 * stride] = v[2];
    block[base + 3 * stride] = v[3];
}
#[inline]
fn inv_lift_axis_i64(block: &mut [i64], base: usize, stride: usize) {
    let mut v = [
        block[base],
        block[base + stride],
        block[base + 2 * stride],
        block[base + 3 * stride],
    ];
    inv_lift_i64(&mut v);
    block[base] = v[0];
    block[base + stride] = v[1];
    block[base + 2 * stride] = v[2];
    block[base + 3 * stride] = v[3];
}

macro_rules! impl_nd_xform {
    ($fwd2:ident, $inv2:ident, $fwd3:ident, $inv3:ident, $fwd4:ident, $inv4:ident,
     $fwd_axis:ident, $inv_axis:ident, $int:ty) => {
        fn $fwd2(block: &mut [$int; 16]) {
            for y in 0..4 {
                $fwd_axis(block.as_mut_slice(), 4 * y, 1);
            }
            for x in 0..4 {
                $fwd_axis(block.as_mut_slice(), x, 4);
            }
        }
        fn $inv2(block: &mut [$int; 16]) {
            for x in 0..4 {
                $inv_axis(block.as_mut_slice(), x, 4);
            }
            for y in 0..4 {
                $inv_axis(block.as_mut_slice(), 4 * y, 1);
            }
        }

        fn $fwd3(block: &mut [$int; 64]) {
            for z in 0..4 {
                for y in 0..4 {
                    $fwd_axis(block.as_mut_slice(), 16 * z + 4 * y, 1);
                }
            }
            for z in 0..4 {
                for x in 0..4 {
                    $fwd_axis(block.as_mut_slice(), 16 * z + x, 4);
                }
            }
            for y in 0..4 {
                for x in 0..4 {
                    $fwd_axis(block.as_mut_slice(), 4 * y + x, 16);
                }
            }
        }
        fn $inv3(block: &mut [$int; 64]) {
            for y in 0..4 {
                for x in 0..4 {
                    $inv_axis(block.as_mut_slice(), 4 * y + x, 16);
                }
            }
            for z in 0..4 {
                for x in 0..4 {
                    $inv_axis(block.as_mut_slice(), 16 * z + x, 4);
                }
            }
            for z in 0..4 {
                for y in 0..4 {
                    $inv_axis(block.as_mut_slice(), 16 * z + 4 * y, 1);
                }
            }
        }

        fn $fwd4(block: &mut [$int; 256]) {
            for w in 0..4 {
                for z in 0..4 {
                    for y in 0..4 {
                        $fwd_axis(block.as_mut_slice(), 64 * w + 16 * z + 4 * y, 1);
                    }
                }
            }
            for w in 0..4 {
                for z in 0..4 {
                    for x in 0..4 {
                        $fwd_axis(block.as_mut_slice(), 64 * w + 16 * z + x, 4);
                    }
                }
            }
            for w in 0..4 {
                for y in 0..4 {
                    for x in 0..4 {
                        $fwd_axis(block.as_mut_slice(), 64 * w + 4 * y + x, 16);
                    }
                }
            }
            for z in 0..4 {
                for y in 0..4 {
                    for x in 0..4 {
                        $fwd_axis(block.as_mut_slice(), 16 * z + 4 * y + x, 64);
                    }
                }
            }
        }
        fn $inv4(block: &mut [$int; 256]) {
            for z in 0..4 {
                for y in 0..4 {
                    for x in 0..4 {
                        $inv_axis(block.as_mut_slice(), 16 * z + 4 * y + x, 64);
                    }
                }
            }
            for w in 0..4 {
                for y in 0..4 {
                    for x in 0..4 {
                        $inv_axis(block.as_mut_slice(), 64 * w + 4 * y + x, 16);
                    }
                }
            }
            for w in 0..4 {
                for z in 0..4 {
                    for x in 0..4 {
                        $inv_axis(block.as_mut_slice(), 64 * w + 16 * z + x, 4);
                    }
                }
            }
            for w in 0..4 {
                for z in 0..4 {
                    for y in 0..4 {
                        $inv_axis(block.as_mut_slice(), 64 * w + 16 * z + 4 * y, 1);
                    }
                }
            }
        }
    };
}

impl_nd_xform!(
    fwd_xform_i32_2d,
    inv_xform_i32_2d,
    fwd_xform_i32_3d,
    inv_xform_i32_3d,
    fwd_xform_i32_4d,
    inv_xform_i32_4d,
    fwd_lift_axis_i32,
    inv_lift_axis_i32,
    i32
);
impl_nd_xform!(
    fwd_xform_i64_2d,
    inv_xform_i64_2d,
    fwd_xform_i64_3d,
    inv_xform_i64_3d,
    fwd_xform_i64_4d,
    inv_xform_i64_4d,
    fwd_lift_axis_i64,
    inv_lift_axis_i64,
    i64
);

// ---------------------------------------------------------------------------
// Coefficient permutation tables (from LLNL/zfp's `codec{2,3,4}.c`).
// `PERM_D[i]` is the flat index of the i-th coefficient in encoding order.
// Ordered by polynomial degree / frequency (low-frequency first).
// ---------------------------------------------------------------------------

const PERM_2: [u8; 16] = [0, 1, 4, 5, 2, 8, 6, 9, 3, 12, 10, 7, 13, 11, 14, 15];

const PERM_3: [u8; 64] = [
    0, 1, 4, 16, 20, 17, 5, 2, 8, 32, 21, 6, 18, 24, 9, 33, 36, 3, 12, 48, 22, 25, 37, 40, 34, 10,
    7, 19, 28, 13, 49, 52, 41, 38, 26, 23, 29, 53, 11, 35, 44, 14, 50, 56, 42, 27, 39, 45, 30, 54,
    57, 60, 51, 15, 43, 46, 58, 61, 55, 31, 62, 59, 47, 63,
];

const PERM_4: [u8; 256] = [
    0, 1, 4, 16, 64, 5, 80, 17, 68, 65, 20, 2, 8, 32, 128, 84, 81, 69, 21, 6, 18, 66, 24, 72, 9,
    96, 33, 36, 129, 132, 144, 3, 12, 48, 192, 85, 82, 70, 22, 73, 25, 88, 37, 100, 97, 148, 145,
    133, 10, 160, 34, 136, 130, 40, 7, 19, 67, 28, 76, 13, 112, 49, 52, 193, 196, 208, 86, 89, 101,
    149, 161, 137, 41, 134, 38, 164, 26, 152, 146, 104, 98, 74, 83, 71, 23, 77, 29, 92, 53, 116,
    113, 212, 209, 197, 11, 35, 131, 44, 140, 14, 176, 50, 56, 194, 200, 224, 90, 165, 102, 153,
    150, 105, 168, 162, 138, 42, 87, 93, 117, 213, 27, 75, 99, 39, 135, 147, 108, 45, 141, 156, 30,
    78, 177, 180, 54, 114, 120, 57, 198, 210, 216, 201, 225, 228, 15, 240, 51, 204, 195, 60, 169,
    166, 154, 106, 91, 103, 151, 109, 157, 94, 181, 118, 121, 214, 217, 229, 163, 139, 43, 142, 46,
    172, 58, 184, 178, 232, 226, 202, 241, 205, 61, 199, 55, 244, 31, 220, 211, 124, 115, 79, 170,
    167, 155, 107, 158, 110, 173, 122, 185, 182, 233, 230, 218, 95, 245, 119, 221, 215, 125, 242,
    206, 62, 203, 59, 248, 47, 236, 227, 188, 179, 143, 171, 174, 186, 234, 246, 222, 126, 219,
    123, 249, 111, 237, 231, 189, 183, 159, 252, 243, 207, 63, 175, 250, 187, 238, 235, 190, 253,
    247, 223, 127, 254, 251, 239, 191, 255,
];

// ---------------------------------------------------------------------------
// Negabinary conversion (signed ↔ unsigned).
// ---------------------------------------------------------------------------

#[inline]
fn int2uint_i32(x: i32) -> u32 {
    ((x as u32).wrapping_add(NBMASK_U32)) ^ NBMASK_U32
}
#[inline]
#[expect(
    clippy::cast_possible_wrap,
    reason = "negabinary-to-signed conversion: the u32 bit pattern is reinterpreted as i32 by design"
)]
fn uint2int_i32(x: u32) -> i32 {
    ((x ^ NBMASK_U32).wrapping_sub(NBMASK_U32)) as i32
}
#[inline]
fn int2uint_i64(x: i64) -> u64 {
    ((x as u64).wrapping_add(NBMASK_U64)) ^ NBMASK_U64
}
#[inline]
#[expect(
    clippy::cast_possible_wrap,
    reason = "negabinary-to-signed conversion: the u64 bit pattern is reinterpreted as i64 by design"
)]
fn uint2int_i64(x: u64) -> i64 {
    ((x ^ NBMASK_U64).wrapping_sub(NBMASK_U64)) as i64
}

// ---------------------------------------------------------------------------
// Embedded bit-plane encoder / decoder
//
// Direct port of LLNL/zfp's `encode_few_ints` / `decode_few_ints` from
// `src/template/encode.c` and `src/template/decode.c`, specialized for
// `size <= 64`. For the 4-coefficient 1D case the generic u32 form covers
// f32 and i32 blocks; the u64 form (added for 2D/3D) covers f64 and i64.
//
// Algorithm: from most-significant bit plane down to `kmin`, each iteration:
//   * Write the first `n` bits of the plane verbatim — one refinement bit
//     per already-significant coefficient (they've had their group-test
//     "1" emitted in a previous plane and are now in fixed-precision mode).
//   * Run a unary run-length scan over the remaining `size - n`
//     coefficients. Repeatedly: emit a group-test bit. If 0, done with
//     this plane. If 1, scan one-by-one; emit each coefficient's bit at
//     this plane until a 1-bit is found (that coefficient becomes newly
//     significant). If only one coefficient remains after a positive
//     group test, its significance is implicit.
// ---------------------------------------------------------------------------

/// `encode_few_ints` / `decode_few_ints` ports. Generated per integer width.
///
/// `size` must be ≤ 64 — the bit plane is packed into a single u64 register.
/// For N-D blocks where size > 64 (4D has 256) the codec dispatches to
/// `encode_many_ints` / `decode_many_ints` instead.
macro_rules! impl_few_ints {
    ($enc:ident, $dec:ident, $uint:ty, $intprec:expr) => {
        fn $enc(
            w: &mut BitWriter,
            maxbits: usize,
            maxprec: u32,
            data: &[$uint],
            size: usize,
        ) -> usize {
            let kmin: u32 = $intprec.saturating_sub(maxprec);
            let mut bits = maxbits;
            let mut n: usize = 0;
            let mut k: u32 = $intprec;
            while bits > 0 && k > kmin {
                k -= 1;
                // step 1: extract bit plane k into x (bit i = data[i] >> k & 1)
                let mut x: u64 = 0;
                for i in 0..size {
                    x |= ((data[i] >> k) as u64 & 1) << i;
                }
                // step 2: emit first n bits (refinement for already-sig coefs)
                let m = n.min(bits);
                bits -= m;
                if m > 0 {
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "m is a bit count <= n <= size <= 64, fits u32"
                    )]
                    w.write(m as u32, x);
                    x >>= m;
                }
                // step 3: unary run-length encode remainder
                while bits > 0 && n < size {
                    bits -= 1;
                    let group = x != 0;
                    w.write_bit(group);
                    if !group {
                        break;
                    }
                    while bits > 0 && n < size - 1 {
                        bits -= 1;
                        let bit = (x & 1) != 0;
                        w.write_bit(bit);
                        if bit {
                            break;
                        }
                        x >>= 1;
                        n += 1;
                    }
                    x >>= 1;
                    n += 1;
                }
            }
            maxbits - bits
        }

        fn $dec(
            r: &mut BitReader<'_>,
            maxbits: usize,
            maxprec: u32,
            data: &mut [$uint],
            size: usize,
        ) -> usize {
            let kmin: u32 = $intprec.saturating_sub(maxprec);
            let mut bits = maxbits;
            let mut n: usize = 0;
            for v in data.iter_mut().take(size) {
                *v = 0;
            }
            let mut k: u32 = $intprec;
            while bits > 0 && k > kmin {
                k -= 1;
                let m = n.min(bits);
                bits -= m;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "m is a bit count <= n <= size <= 64, fits u32"
                )]
                let mut x: u64 = if m > 0 { r.read(m as u32) } else { 0 };
                while bits > 0 && n < size {
                    bits -= 1;
                    if r.read_bit() {
                        while bits > 0 && n < size - 1 {
                            bits -= 1;
                            if r.read_bit() {
                                break;
                            }
                            n += 1;
                        }
                        x |= 1u64 << n;
                    } else {
                        break;
                    }
                    n += 1;
                }
                let mut xx = x;
                let mut i = 0;
                while xx != 0 {
                    data[i] |= ((xx & 1) as $uint) << k;
                    xx >>= 1;
                    i += 1;
                }
            }
            maxbits - bits
        }
    };
}

impl_few_ints!(encode_few_ints_u32, decode_few_ints_u32, u32, 32u32);
impl_few_ints!(encode_few_ints_u64, decode_few_ints_u64, u64, 64u32);

/// `encode_many_ints` / `decode_many_ints` ports for `size > 64` (used by 4D
/// blocks, which have 256 coefficients). Same algorithm as the `few` variant
/// but the bit plane is not packed into a single u64 — the encoder iterates
/// `data[i] >> k` directly and tracks the remaining-1-bit count in `c`.
macro_rules! impl_many_ints {
    ($enc:ident, $dec:ident, $uint:ty, $intprec:expr) => {
        fn $enc(
            w: &mut BitWriter,
            maxbits: usize,
            maxprec: u32,
            data: &[$uint],
            size: usize,
        ) -> usize {
            let kmin: u32 = $intprec.saturating_sub(maxprec);
            let mut bits = maxbits;
            let mut n: usize = 0;
            let mut k: u32 = $intprec;
            while bits > 0 && k > kmin {
                k -= 1;
                // step 1: emit first n bits (refinement for already-sig coefs)
                let m = n.min(bits);
                bits -= m;
                for i in 0..m {
                    w.write_bit((data[i] >> k) & 1 == 1);
                }
                // step 2: count remaining 1-bits in this plane
                let mut c: usize = 0;
                for i in m..size {
                    if (data[i] >> k) & 1 == 1 {
                        c += 1;
                    }
                }
                // step 3: unary run-length encode remainder
                while bits > 0 && n < size {
                    bits -= 1;
                    let group = c > 0;
                    w.write_bit(group);
                    if !group {
                        break;
                    }
                    // found at least one; scan for the next 1-bit
                    c -= 1;
                    while bits > 0 && n < size - 1 {
                        bits -= 1;
                        let bit = (data[n] >> k) & 1 == 1;
                        w.write_bit(bit);
                        if bit {
                            break;
                        }
                        n += 1;
                    }
                    n += 1;
                }
            }
            maxbits - bits
        }

        fn $dec(
            r: &mut BitReader<'_>,
            maxbits: usize,
            maxprec: u32,
            data: &mut [$uint],
            size: usize,
        ) -> usize {
            let kmin: u32 = $intprec.saturating_sub(maxprec);
            let mut bits = maxbits;
            let mut n: usize = 0;
            for v in data.iter_mut().take(size) {
                *v = 0;
            }
            let mut k: u32 = $intprec;
            while bits > 0 && k > kmin {
                k -= 1;
                // step 1: decode first n bits
                let m = n.min(bits);
                bits -= m;
                for i in 0..m {
                    if r.read_bit() {
                        data[i] |= (1 as $uint) << k;
                    }
                }
                // step 2: unary run-length decode remainder
                while bits > 0 && n < size {
                    bits -= 1;
                    if r.read_bit() {
                        while bits > 0 && n < size - 1 {
                            bits -= 1;
                            if r.read_bit() {
                                break;
                            }
                            n += 1;
                        }
                        data[n] |= (1 as $uint) << k;
                    } else {
                        break;
                    }
                    n += 1;
                }
            }
            maxbits - bits
        }
    };
}

impl_many_ints!(encode_many_ints_u32, decode_many_ints_u32, u32, 32u32);
impl_many_ints!(encode_many_ints_u64, decode_many_ints_u64, u64, 64u32);

// ---------------------------------------------------------------------------
// Block-level encode / decode
// ---------------------------------------------------------------------------

/// Encode a single block of 4 f32 values into the bit stream.
///
/// Each block gets exactly `maxbits` bits of output (fixed-rate mode).
fn encode_block_f32(w: &mut BitWriter, vals: &[f32; BLOCK_SIZE], maxbits: usize) {
    let start = w.position();

    // Step 1: block-floating-point cast
    let (emax_biased, mut icoeffs) = fwd_cast_f32(vals);

    if emax_biased == 0 {
        // Empty block -- write a 0-bit header and pad
        w.write_bit(false);
        let remaining = maxbits.saturating_sub(1);
        pad_bits(w, remaining);
        return;
    }

    // Non-empty block
    w.write_bit(true);

    // Write exponent (EBITS = 8 bits).
    // `emax_biased` holds `emax + 1` where `emax` is the raw IEEE biased
    // exponent of the block's largest-magnitude value. ZFP stores this
    // `emax + 1` value directly (matches reference LLNL/zfp output).
    w.write(EBITS_F32, u64::from(emax_biased));

    // Step 2: forward lifting transform
    fwd_lift_i32(&mut icoeffs);

    // Step 3: negabinary conversion
    let mut ucoeffs = [0u32; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        ucoeffs[i] = int2uint_i32(icoeffs[i]);
    }

    // Step 4: embedded bit-plane encoding with budget
    let header_bits = 1 + EBITS_F32 as usize;
    let coeff_bits = maxbits.saturating_sub(header_bits);
    // maxprec for 1D fixed-rate: full precision of the coefficient type.
    encode_few_ints_u32(w, coeff_bits, 32, &ucoeffs, BLOCK_SIZE);

    // Pad to exactly maxbits
    let used = w.position() - start;
    let remaining = maxbits.saturating_sub(used);
    pad_bits(w, remaining);
}

/// Decode a single block of 4 f32 values from the bit stream.
fn decode_block_f32(r: &mut BitReader<'_>, maxbits: usize) -> Result<[f32; BLOCK_SIZE], Error> {
    // Read empty-block flag
    let nonempty = r.read_bit();
    if !nonempty {
        // Skip remaining bits
        let remaining = maxbits.saturating_sub(1);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "min(64) caps the bit count at 64, well within u32"
        )]
        r.read(remaining.min(64) as u32);
        if remaining > 64 {
            skip_bits(r, remaining - 64);
        }
        return Ok([0.0f32; BLOCK_SIZE]);
    }

    // Read exponent — the stored value is `emax + 1` (see encode_block_f32).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "r.read(EBITS_F32) yields an 8-bit value (returned as u64); fits u32"
    )]
    let emax_biased = r.read(EBITS_F32) as u32;

    // Decode coefficients
    let header_bits = 1 + EBITS_F32 as usize;
    let coeff_bits = maxbits.saturating_sub(header_bits);
    let mut ucoeffs = [0u32; BLOCK_SIZE];
    let bits_consumed = decode_few_ints_u32(r, coeff_bits, 32, &mut ucoeffs, BLOCK_SIZE);

    // Skip remaining padding bits to maintain block alignment.
    let remaining = coeff_bits.saturating_sub(bits_consumed);
    skip_bits(r, remaining);

    // Negabinary -> signed
    let mut icoeffs = [0i32; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        icoeffs[i] = uint2int_i32(ucoeffs[i]);
    }

    // Inverse lifting transform
    inv_lift_i32(&mut icoeffs);

    // Inverse block-floating-point cast
    Ok(inv_cast_f32(emax_biased, &icoeffs))
}

fn encode_block_f64(w: &mut BitWriter, vals: &[f64; BLOCK_SIZE], maxbits: usize) {
    let start = w.position();
    let (emax_biased, mut icoeffs) = fwd_cast_f64(vals);
    if emax_biased == 0 {
        w.write_bit(false);
        pad_bits(w, maxbits.saturating_sub(1));
        return;
    }
    w.write_bit(true);
    w.write(EBITS_F64, emax_biased);
    fwd_lift_i64(&mut icoeffs);
    let mut ucoeffs = [0u64; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        ucoeffs[i] = int2uint_i64(icoeffs[i]);
    }
    let header_bits = 1 + EBITS_F64 as usize;
    let coeff_bits = maxbits.saturating_sub(header_bits);
    encode_few_ints_u64(w, coeff_bits, 64, &ucoeffs, BLOCK_SIZE);
    let used = w.position() - start;
    pad_bits(w, maxbits.saturating_sub(used));
}

fn decode_block_f64(r: &mut BitReader<'_>, maxbits: usize) -> Result<[f64; BLOCK_SIZE], Error> {
    let nonempty = r.read_bit();
    if !nonempty {
        let remaining = maxbits.saturating_sub(1);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "min(64) caps the bit count at 64, well within u32"
        )]
        r.read(remaining.min(64) as u32);
        if remaining > 64 {
            skip_bits(r, remaining - 64);
        }
        return Ok([0.0f64; BLOCK_SIZE]);
    }
    let emax_biased = r.read(EBITS_F64);
    let header_bits = 1 + EBITS_F64 as usize;
    let coeff_bits = maxbits.saturating_sub(header_bits);
    let mut ucoeffs = [0u64; BLOCK_SIZE];
    let bits_consumed = decode_few_ints_u64(r, coeff_bits, 64, &mut ucoeffs, BLOCK_SIZE);
    skip_bits(r, coeff_bits.saturating_sub(bits_consumed));
    let mut icoeffs = [0i64; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        icoeffs[i] = uint2int_i64(ucoeffs[i]);
    }
    inv_lift_i64(&mut icoeffs);
    Ok(inv_cast_f64(emax_biased, &icoeffs))
}

/// Integer-path block encoder — no block-float header, coefficients are the
/// input values cast directly. Covers i32 (u32 coeff) and i64 (u64 coeff).
macro_rules! impl_int_block_codec {
    ($enc:ident, $dec:ident, $int:ty, $uint:ty, $fwd_lift:ident, $inv_lift:ident,
     $to_uint:ident, $to_int:ident, $enc_ints:ident, $dec_ints:ident, $intprec:expr) => {
        fn $enc(w: &mut BitWriter, vals: &[$int; BLOCK_SIZE], maxbits: usize) {
            let start = w.position();
            let mut icoeffs = *vals;
            $fwd_lift(&mut icoeffs);
            let mut ucoeffs = [0 as $uint; BLOCK_SIZE];
            for i in 0..BLOCK_SIZE {
                ucoeffs[i] = $to_uint(icoeffs[i]);
            }
            $enc_ints(w, maxbits, $intprec, &ucoeffs, BLOCK_SIZE);
            let used = w.position() - start;
            pad_bits(w, maxbits.saturating_sub(used));
        }
        fn $dec(r: &mut BitReader<'_>, maxbits: usize) -> Result<[$int; BLOCK_SIZE], Error> {
            let mut ucoeffs = [0 as $uint; BLOCK_SIZE];
            let bits_consumed = $dec_ints(r, maxbits, $intprec, &mut ucoeffs, BLOCK_SIZE);
            skip_bits(r, maxbits.saturating_sub(bits_consumed));
            let mut icoeffs = [0 as $int; BLOCK_SIZE];
            for i in 0..BLOCK_SIZE {
                icoeffs[i] = $to_int(ucoeffs[i]);
            }
            $inv_lift(&mut icoeffs);
            Ok(icoeffs)
        }
    };
}

impl_int_block_codec!(
    encode_block_i32,
    decode_block_i32,
    i32,
    u32,
    fwd_lift_i32,
    inv_lift_i32,
    int2uint_i32,
    uint2int_i32,
    encode_few_ints_u32,
    decode_few_ints_u32,
    32u32
);
impl_int_block_codec!(
    encode_block_i64,
    decode_block_i64,
    i64,
    u64,
    fwd_lift_i64,
    inv_lift_i64,
    int2uint_i64,
    uint2int_i64,
    encode_few_ints_u64,
    decode_few_ints_u64,
    64u32
);

// ---------------------------------------------------------------------------
// N-D block codec (2D / 3D / 4D). Structure identical to the 1D code above
// but with larger blocks and a coefficient permutation (`PERM_D`). 4D blocks
// have 256 coefficients, beyond the 64-coefficient `encode_few_ints` limit,
// so they use `encode_many_ints` instead.
// ---------------------------------------------------------------------------

macro_rules! impl_float_block_nd {
    ($enc:ident, $dec:ident, $bs:expr, $scalar:ty, $int:ty, $uint:ty, $zero:expr,
     $fwd_cast_slice:ident, $inv_cast_slice:ident,
     $fwd_xform:ident, $inv_xform:ident,
     $int2uint:ident, $uint2int:ident, $perm:ident,
     $enc_ints:ident, $dec_ints:ident, $intprec:expr,
     $ebits:expr, $emax_ty:ty) => {
        fn $enc(w: &mut BitWriter, block: &[$scalar; $bs], maxbits: usize) {
            let start = w.position();
            let mut icoeffs = [0 as $int; $bs];
            let emax_biased = $fwd_cast_slice(block, &mut icoeffs);
            if emax_biased == 0 {
                w.write_bit(false);
                pad_bits(w, maxbits.saturating_sub(1));
                return;
            }
            w.write_bit(true);
            w.write($ebits, emax_biased as u64);
            $fwd_xform(&mut icoeffs);
            let mut ucoeffs = [0 as $uint; $bs];
            for i in 0..$bs {
                ucoeffs[i] = $int2uint(icoeffs[$perm[i] as usize]);
            }
            let header_bits = 1 + ($ebits as usize);
            let coeff_bits = maxbits.saturating_sub(header_bits);
            $enc_ints(w, coeff_bits, $intprec, &ucoeffs, $bs);
            let used = w.position() - start;
            pad_bits(w, maxbits.saturating_sub(used));
        }

        fn $dec(r: &mut BitReader<'_>, maxbits: usize) -> Result<[$scalar; $bs], Error> {
            let nonempty = r.read_bit();
            if !nonempty {
                let remaining = maxbits.saturating_sub(1);
                // skip_bits handles > 64 bits
                skip_bits(r, remaining);
                return Ok([$zero; $bs]);
            }
            // `#[allow]`, not `#[expect]`: `$emax_ty` is `u64` for f64 blocks
            // (the cast is identity, so the lint never fires) and `u32` for f32
            // blocks (a safe narrowing of an 8-bit exponent). An `#[expect]`
            // would be unfulfilled in the u64 expansions.
            #[allow(
                clippy::cast_possible_truncation,
                reason = "exponent occupies $ebits (8 or 11) bits, so it fits the narrower $emax_ty"
            )]
            let emax_biased: $emax_ty = r.read($ebits) as $emax_ty;
            let header_bits = 1 + ($ebits as usize);
            let coeff_bits = maxbits.saturating_sub(header_bits);
            let mut ucoeffs = [0 as $uint; $bs];
            let bits_consumed = $dec_ints(r, coeff_bits, $intprec, &mut ucoeffs, $bs);
            skip_bits(r, coeff_bits.saturating_sub(bits_consumed));
            let mut icoeffs = [0 as $int; $bs];
            for i in 0..$bs {
                icoeffs[$perm[i] as usize] = $uint2int(ucoeffs[i]);
            }
            $inv_xform(&mut icoeffs);
            let mut out = [$zero; $bs];
            $inv_cast_slice(emax_biased, &icoeffs, &mut out);
            Ok(out)
        }
    };
}

macro_rules! impl_int_block_nd {
    ($enc:ident, $dec:ident, $bs:expr, $int:ty, $uint:ty, $zero:expr,
     $fwd_xform:ident, $inv_xform:ident,
     $int2uint:ident, $uint2int:ident, $perm:ident,
     $enc_ints:ident, $dec_ints:ident, $intprec:expr) => {
        fn $enc(w: &mut BitWriter, block: &[$int; $bs], maxbits: usize) {
            let start = w.position();
            let mut icoeffs = *block;
            $fwd_xform(&mut icoeffs);
            let mut ucoeffs = [0 as $uint; $bs];
            for i in 0..$bs {
                ucoeffs[i] = $int2uint(icoeffs[$perm[i] as usize]);
            }
            $enc_ints(w, maxbits, $intprec, &ucoeffs, $bs);
            let used = w.position() - start;
            pad_bits(w, maxbits.saturating_sub(used));
        }
        fn $dec(r: &mut BitReader<'_>, maxbits: usize) -> Result<[$int; $bs], Error> {
            let mut ucoeffs = [0 as $uint; $bs];
            let bits_consumed = $dec_ints(r, maxbits, $intprec, &mut ucoeffs, $bs);
            skip_bits(r, maxbits.saturating_sub(bits_consumed));
            let mut icoeffs = [$zero; $bs];
            for i in 0..$bs {
                icoeffs[$perm[i] as usize] = $uint2int(ucoeffs[i]);
            }
            $inv_xform(&mut icoeffs);
            Ok(icoeffs)
        }
    };
}

// 2D block codec (16 coefs).
impl_float_block_nd!(
    encode_block_f32_2d,
    decode_block_f32_2d,
    16,
    f32,
    i32,
    u32,
    0.0f32,
    fwd_cast_f32_slice,
    inv_cast_f32_slice,
    fwd_xform_i32_2d,
    inv_xform_i32_2d,
    int2uint_i32,
    uint2int_i32,
    PERM_2,
    encode_few_ints_u32,
    decode_few_ints_u32,
    32u32,
    EBITS_F32,
    u32
);
impl_float_block_nd!(
    encode_block_f64_2d,
    decode_block_f64_2d,
    16,
    f64,
    i64,
    u64,
    0.0f64,
    fwd_cast_f64_slice,
    inv_cast_f64_slice,
    fwd_xform_i64_2d,
    inv_xform_i64_2d,
    int2uint_i64,
    uint2int_i64,
    PERM_2,
    encode_few_ints_u64,
    decode_few_ints_u64,
    64u32,
    EBITS_F64,
    u64
);
impl_int_block_nd!(
    encode_block_i32_2d,
    decode_block_i32_2d,
    16,
    i32,
    u32,
    0i32,
    fwd_xform_i32_2d,
    inv_xform_i32_2d,
    int2uint_i32,
    uint2int_i32,
    PERM_2,
    encode_few_ints_u32,
    decode_few_ints_u32,
    32u32
);
impl_int_block_nd!(
    encode_block_i64_2d,
    decode_block_i64_2d,
    16,
    i64,
    u64,
    0i64,
    fwd_xform_i64_2d,
    inv_xform_i64_2d,
    int2uint_i64,
    uint2int_i64,
    PERM_2,
    encode_few_ints_u64,
    decode_few_ints_u64,
    64u32
);

// 3D block codec (64 coefs).
impl_float_block_nd!(
    encode_block_f32_3d,
    decode_block_f32_3d,
    64,
    f32,
    i32,
    u32,
    0.0f32,
    fwd_cast_f32_slice,
    inv_cast_f32_slice,
    fwd_xform_i32_3d,
    inv_xform_i32_3d,
    int2uint_i32,
    uint2int_i32,
    PERM_3,
    encode_few_ints_u32,
    decode_few_ints_u32,
    32u32,
    EBITS_F32,
    u32
);
impl_float_block_nd!(
    encode_block_f64_3d,
    decode_block_f64_3d,
    64,
    f64,
    i64,
    u64,
    0.0f64,
    fwd_cast_f64_slice,
    inv_cast_f64_slice,
    fwd_xform_i64_3d,
    inv_xform_i64_3d,
    int2uint_i64,
    uint2int_i64,
    PERM_3,
    encode_few_ints_u64,
    decode_few_ints_u64,
    64u32,
    EBITS_F64,
    u64
);
impl_int_block_nd!(
    encode_block_i32_3d,
    decode_block_i32_3d,
    64,
    i32,
    u32,
    0i32,
    fwd_xform_i32_3d,
    inv_xform_i32_3d,
    int2uint_i32,
    uint2int_i32,
    PERM_3,
    encode_few_ints_u32,
    decode_few_ints_u32,
    32u32
);
impl_int_block_nd!(
    encode_block_i64_3d,
    decode_block_i64_3d,
    64,
    i64,
    u64,
    0i64,
    fwd_xform_i64_3d,
    inv_xform_i64_3d,
    int2uint_i64,
    uint2int_i64,
    PERM_3,
    encode_few_ints_u64,
    decode_few_ints_u64,
    64u32
);

// 4D block codec (256 coefs) — uses `encode_many_ints` / `decode_many_ints`.
impl_float_block_nd!(
    encode_block_f32_4d,
    decode_block_f32_4d,
    256,
    f32,
    i32,
    u32,
    0.0f32,
    fwd_cast_f32_slice,
    inv_cast_f32_slice,
    fwd_xform_i32_4d,
    inv_xform_i32_4d,
    int2uint_i32,
    uint2int_i32,
    PERM_4,
    encode_many_ints_u32,
    decode_many_ints_u32,
    32u32,
    EBITS_F32,
    u32
);
impl_float_block_nd!(
    encode_block_f64_4d,
    decode_block_f64_4d,
    256,
    f64,
    i64,
    u64,
    0.0f64,
    fwd_cast_f64_slice,
    inv_cast_f64_slice,
    fwd_xform_i64_4d,
    inv_xform_i64_4d,
    int2uint_i64,
    uint2int_i64,
    PERM_4,
    encode_many_ints_u64,
    decode_many_ints_u64,
    64u32,
    EBITS_F64,
    u64
);
impl_int_block_nd!(
    encode_block_i32_4d,
    decode_block_i32_4d,
    256,
    i32,
    u32,
    0i32,
    fwd_xform_i32_4d,
    inv_xform_i32_4d,
    int2uint_i32,
    uint2int_i32,
    PERM_4,
    encode_many_ints_u32,
    decode_many_ints_u32,
    32u32
);
impl_int_block_nd!(
    encode_block_i64_4d,
    decode_block_i64_4d,
    256,
    i64,
    u64,
    0i64,
    fwd_xform_i64_4d,
    inv_xform_i64_4d,
    int2uint_i64,
    uint2int_i64,
    PERM_4,
    encode_many_ints_u64,
    decode_many_ints_u64,
    64u32
);

/// Write `n` zero bits.
fn pad_bits(w: &mut BitWriter, n: usize) {
    // Write in chunks of 64
    let mut remaining = n;
    while remaining >= 64 {
        w.write(64, 0);
        remaining -= 64;
    }
    if remaining > 0 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "loop above drains multiples of 64, so remaining is < 64 here; fits u32"
        )]
        w.write(remaining as u32, 0);
    }
}

/// Skip `n` bits in the reader.
fn skip_bits(r: &mut BitReader<'_>, n: usize) {
    let mut remaining = n;
    while remaining >= 64 {
        r.read(64);
        remaining -= 64;
    }
    if remaining > 0 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "loop above drains multiples of 64, so remaining is < 64 here; fits u32"
        )]
        r.read(remaining as u32);
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

#[inline]
fn f32_from_le(b: [u8; 4]) -> f32 {
    f32::from_le_bytes(b)
}
#[inline]
fn f64_from_le(b: [u8; 8]) -> f64 {
    f64::from_le_bytes(b)
}
#[inline]
fn i32_from_le(b: [u8; 4]) -> i32 {
    i32::from_le_bytes(b)
}
#[inline]
fn i64_from_le(b: [u8; 8]) -> i64 {
    i64::from_le_bytes(b)
}

/// Generates a per-scalar codec module with 1D/2D/3D/4D compress and
/// decompress functions, plus public dispatchers `compress` / `decompress`
/// that route by `dims.len()`. Dims use HDF5 row-major convention (outermost
/// first; the last entry is the fastest-varying axis).
macro_rules! impl_codec {
    (
        $mod_name:ident, $scalar:ty, $zero_s:expr, $esz:expr, $from_le:path,
        $pad1:ident, $pad_s:ident,
        $enc1:ident, $dec1:ident,
        $enc2:ident, $dec2:ident,
        $enc3:ident, $dec3:ident,
        $enc4:ident, $dec4:ident
    ) => {
        pub(crate) mod $mod_name {
            use super::*;

            pub fn compress(
                data: &[u8],
                dims: ZfpChunkDims,
                rate: f64,
            ) -> Result<Vec<u8>, Error> {
                let expected = dims
                    .checked_element_count()?
                    .checked_mul($esz)
                    .ok_or(Error::ZfpSizeOverflow)?;
                if data.len() != expected {
                    return Err(Error::ZfpFilter(format!(
                        "ZFP: data length {} does not match dims product × element size ({})",
                        data.len(),
                        expected,
                    )));
                }
                match dims {
                    ZfpChunkDims::One([nx]) => compress_1d(data, nx, rate),
                    ZfpChunkDims::Two([ny, nx]) => compress_2d(data, ny, nx, rate),
                    ZfpChunkDims::Three([nz, ny, nx]) => compress_3d(data, nz, ny, nx, rate),
                    ZfpChunkDims::Four([nw, nz, ny, nx]) => {
                        compress_4d(data, nw, nz, ny, nx, rate)
                    }
                }
            }

            pub fn decompress(
                compressed: &[u8],
                dims: ZfpChunkDims,
                rate: f64,
            ) -> Result<Vec<u8>, Error> {
                match dims {
                    ZfpChunkDims::One([nx]) => decompress_1d(compressed, nx, rate),
                    ZfpChunkDims::Two([ny, nx]) => decompress_2d(compressed, ny, nx, rate),
                    ZfpChunkDims::Three([nz, ny, nx]) => {
                        decompress_3d(compressed, nz, ny, nx, rate)
                    }
                    ZfpChunkDims::Four([nw, nz, ny, nx]) => {
                        decompress_4d(compressed, nw, nz, ny, nx, rate)
                    }
                }
            }

            fn compress_1d(data: &[u8], n: usize, rate: f64) -> Result<Vec<u8>, Error> {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "rate is bits-per-value (format-bounded); rate * 4 (1D block) is a small bit budget that fits usize"
                )]
                let maxbits = (rate * 4.0) as usize;
                let total_bits = n.div_ceil(4) * maxbits;
                let mut w = BitWriter::with_capacity(total_bits.div_ceil(8) + 8);
                let mut i = 0;
                while i < n {
                    let mut block = [$zero_s; 4];
                    let nr = (n - i).min(4);
                    for j in 0..nr {
                        let off = (i + j) * $esz;
                        let mut buf = [0u8; $esz];
                        buf.copy_from_slice(&data[off..off + $esz]);
                        block[j] = $from_le(buf);
                    }
                    $pad1(&mut block, nr);
                    $enc1(&mut w, &block, maxbits);
                    i += 4;
                }
                let mut out = w.finish();
                out.truncate(total_bits.div_ceil(8));
                Ok(out)
            }

            fn decompress_1d(
                compressed: &[u8],
                n: usize,
                rate: f64,
            ) -> Result<Vec<u8>, Error> {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "rate is bits-per-value (format-bounded); rate * 4 (1D block) is a small bit budget that fits usize"
                )]
                let maxbits = (rate * 4.0) as usize;
                let mut r = BitReader::new(compressed);
                let mut output = Vec::with_capacity(n * $esz);
                let mut i = 0;
                while i < n {
                    let block = $dec1(&mut r, maxbits)?;
                    let c = (n - i).min(4);
                    for j in 0..c {
                        output.extend_from_slice(&block[j].to_le_bytes());
                    }
                    i += 4;
                }
                r.finish_decode()?;
                Ok(output)
            }

            fn compress_2d(
                data: &[u8],
                n1: usize,
                n0: usize,
                rate: f64,
            ) -> Result<Vec<u8>, Error> {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "rate is bits-per-value (format-bounded); rate * 16 (2D block) is a small bit budget that fits usize"
                )]
                let maxbits = (rate * 16.0) as usize;
                let nb1 = n1.div_ceil(4);
                let nb0 = n0.div_ceil(4);
                let total_bits = nb1 * nb0 * maxbits;
                let mut w = BitWriter::with_capacity(total_bits.div_ceil(8) + 8);
                for b1 in 0..nb1 {
                    for b0 in 0..nb0 {
                        let y0 = b1 * 4;
                        let x0 = b0 * 4;
                        let ry = (n1 - y0).min(4);
                        let rx = (n0 - x0).min(4);
                        let mut block = [$zero_s; 16];
                        for y in 0..ry {
                            for x in 0..rx {
                                let src = (y0 + y) * n0 + (x0 + x);
                                let off = src * $esz;
                                let mut buf = [0u8; $esz];
                                buf.copy_from_slice(&data[off..off + $esz]);
                                block[4 * y + x] = $from_le(buf);
                            }
                            $pad_s(block.as_mut_slice(), 4 * y, rx, 1);
                        }
                        for x in 0..4 {
                            $pad_s(block.as_mut_slice(), x, ry, 4);
                        }
                        $enc2(&mut w, &block, maxbits);
                    }
                }
                let mut out = w.finish();
                out.truncate(total_bits.div_ceil(8));
                Ok(out)
            }

            fn decompress_2d(
                compressed: &[u8],
                n1: usize,
                n0: usize,
                rate: f64,
            ) -> Result<Vec<u8>, Error> {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "rate is bits-per-value (format-bounded); rate * 16 (2D block) is a small bit budget that fits usize"
                )]
                let maxbits = (rate * 16.0) as usize;
                let nb1 = n1.div_ceil(4);
                let nb0 = n0.div_ceil(4);
                let mut r = BitReader::new(compressed);
                let mut output = vec![0u8; n1 * n0 * $esz];
                for b1 in 0..nb1 {
                    for b0 in 0..nb0 {
                        let y0 = b1 * 4;
                        let x0 = b0 * 4;
                        let block = $dec2(&mut r, maxbits)?;
                        let ry = (n1 - y0).min(4);
                        let rx = (n0 - x0).min(4);
                        for y in 0..ry {
                            for x in 0..rx {
                                let dst = (y0 + y) * n0 + (x0 + x);
                                let off = dst * $esz;
                                output[off..off + $esz]
                                    .copy_from_slice(&block[4 * y + x].to_le_bytes());
                            }
                        }
                    }
                }
                r.finish_decode()?;
                Ok(output)
            }

            fn compress_3d(
                data: &[u8],
                n2: usize,
                n1: usize,
                n0: usize,
                rate: f64,
            ) -> Result<Vec<u8>, Error> {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "rate is bits-per-value (format-bounded); rate * 64 (3D block) is a small bit budget that fits usize"
                )]
                let maxbits = (rate * 64.0) as usize;
                let nb2 = n2.div_ceil(4);
                let nb1 = n1.div_ceil(4);
                let nb0 = n0.div_ceil(4);
                let total_bits = nb2 * nb1 * nb0 * maxbits;
                let mut w = BitWriter::with_capacity(total_bits.div_ceil(8) + 8);
                for b2 in 0..nb2 {
                    for b1 in 0..nb1 {
                        for b0 in 0..nb0 {
                            let z0 = b2 * 4;
                            let y0 = b1 * 4;
                            let x0 = b0 * 4;
                            let rz = (n2 - z0).min(4);
                            let ry = (n1 - y0).min(4);
                            let rx = (n0 - x0).min(4);
                            let mut block = [$zero_s; 64];
                            if rz == 4 && ry == 4 && rx == 4 {
                                // Fast path: interior block, no padding.
                                // The fixed 4×4×4 loop lets LLVM fully unroll
                                // and skip the ~48 no-op `$pad_s` dispatches
                                // that the partial path emits per block.
                                for z in 0..4 {
                                    for y in 0..4 {
                                        let row_off = (((z0 + z) * n1 + (y0 + y)) * n0 + x0) * $esz;
                                        let row = &data[row_off..row_off + 4 * $esz];
                                        for x in 0..4 {
                                            let buf: [u8; $esz] =
                                                row[x * $esz..(x + 1) * $esz].try_into().unwrap();
                                            block[16 * z + 4 * y + x] = $from_le(buf);
                                        }
                                    }
                                }
                            } else {
                                // Partial block (edge): extract what's there and
                                // pad the rest along each axis.
                                for z in 0..rz {
                                    for y in 0..ry {
                                        let row_off = (((z0 + z) * n1 + (y0 + y)) * n0 + x0) * $esz;
                                        let row = &data[row_off..row_off + rx * $esz];
                                        for x in 0..rx {
                                            let buf: [u8; $esz] =
                                                row[x * $esz..(x + 1) * $esz].try_into().unwrap();
                                            block[16 * z + 4 * y + x] = $from_le(buf);
                                        }
                                        $pad_s(block.as_mut_slice(), 16 * z + 4 * y, rx, 1);
                                    }
                                    for x in 0..4 {
                                        $pad_s(block.as_mut_slice(), 16 * z + x, ry, 4);
                                    }
                                }
                                for y in 0..4 {
                                    for x in 0..4 {
                                        $pad_s(block.as_mut_slice(), 4 * y + x, rz, 16);
                                    }
                                }
                            }
                            $enc3(&mut w, &block, maxbits);
                        }
                    }
                }
                let mut out = w.finish();
                out.truncate(total_bits.div_ceil(8));
                Ok(out)
            }

            fn decompress_3d(
                compressed: &[u8],
                n2: usize,
                n1: usize,
                n0: usize,
                rate: f64,
            ) -> Result<Vec<u8>, Error> {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "rate is bits-per-value (format-bounded); rate * 64 (3D block) is a small bit budget that fits usize"
                )]
                let maxbits = (rate * 64.0) as usize;
                let nb2 = n2.div_ceil(4);
                let nb1 = n1.div_ceil(4);
                let nb0 = n0.div_ceil(4);
                let mut r = BitReader::new(compressed);
                let mut output = vec![0u8; n2 * n1 * n0 * $esz];
                for b2 in 0..nb2 {
                    for b1 in 0..nb1 {
                        for b0 in 0..nb0 {
                            let z0 = b2 * 4;
                            let y0 = b1 * 4;
                            let x0 = b0 * 4;
                            let block = $dec3(&mut r, maxbits)?;
                            let rz = (n2 - z0).min(4);
                            let ry = (n1 - y0).min(4);
                            let rx = (n0 - x0).min(4);
                            if rz == 4 && ry == 4 && rx == 4 {
                                // Fast path: interior block. Compile-time 4×4×4
                                // lets LLVM unroll and vectorize the stores.
                                for z in 0..4 {
                                    for y in 0..4 {
                                        let row_off = (((z0 + z) * n1 + (y0 + y)) * n0 + x0) * $esz;
                                        let row = &mut output[row_off..row_off + 4 * $esz];
                                        for x in 0..4 {
                                            row[x * $esz..(x + 1) * $esz].copy_from_slice(
                                                &block[16 * z + 4 * y + x].to_le_bytes(),
                                            );
                                        }
                                    }
                                }
                            } else {
                                // Partial block (edge): write only what fits.
                                for z in 0..rz {
                                    for y in 0..ry {
                                        let row_off = (((z0 + z) * n1 + (y0 + y)) * n0 + x0) * $esz;
                                        let row = &mut output[row_off..row_off + rx * $esz];
                                        for x in 0..rx {
                                            row[x * $esz..(x + 1) * $esz].copy_from_slice(
                                                &block[16 * z + 4 * y + x].to_le_bytes(),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                r.finish_decode()?;
                Ok(output)
            }

            fn compress_4d(
                data: &[u8],
                n3: usize,
                n2: usize,
                n1: usize,
                n0: usize,
                rate: f64,
            ) -> Result<Vec<u8>, Error> {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "rate is bits-per-value (format-bounded); rate * 256 (4D block) is a small bit budget that fits usize"
                )]
                let maxbits = (rate * 256.0) as usize;
                let nb3 = n3.div_ceil(4);
                let nb2 = n2.div_ceil(4);
                let nb1 = n1.div_ceil(4);
                let nb0 = n0.div_ceil(4);
                let total_bits = nb3 * nb2 * nb1 * nb0 * maxbits;
                let mut w = BitWriter::with_capacity(total_bits.div_ceil(8) + 8);
                for b3 in 0..nb3 {
                    for b2 in 0..nb2 {
                        for b1 in 0..nb1 {
                            for b0 in 0..nb0 {
                                let w0 = b3 * 4;
                                let z0 = b2 * 4;
                                let y0 = b1 * 4;
                                let x0 = b0 * 4;
                                let rw = (n3 - w0).min(4);
                                let rz = (n2 - z0).min(4);
                                let ry = (n1 - y0).min(4);
                                let rx = (n0 - x0).min(4);
                                let mut block = [$zero_s; 256];
                                if rw == 4 && rz == 4 && ry == 4 && rx == 4 {
                                    // Fast path: interior block, no padding.
                                    // Fixed 4×4×4×4 lets LLVM unroll, and the
                                    // row-slice hoist gives the inner x loop
                                    // a bounded mutable slice.
                                    for wi in 0..4 {
                                        for z in 0..4 {
                                            for y in 0..4 {
                                                let row_off = ((((w0 + wi) * n2 + (z0 + z)) * n1
                                                    + (y0 + y))
                                                    * n0
                                                    + x0)
                                                    * $esz;
                                                let row = &data[row_off..row_off + 4 * $esz];
                                                for x in 0..4 {
                                                    let buf: [u8; $esz] = row
                                                        [x * $esz..(x + 1) * $esz]
                                                        .try_into()
                                                        .unwrap();
                                                    block[64 * wi + 16 * z + 4 * y + x] =
                                                        $from_le(buf);
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    for wi in 0..rw {
                                        for z in 0..rz {
                                            for y in 0..ry {
                                                for x in 0..rx {
                                                    let src = (((w0 + wi) * n2 + (z0 + z)) * n1
                                                        + (y0 + y))
                                                        * n0
                                                        + (x0 + x);
                                                    let off = src * $esz;
                                                    let mut buf = [0u8; $esz];
                                                    buf.copy_from_slice(&data[off..off + $esz]);
                                                    block[64 * wi + 16 * z + 4 * y + x] =
                                                        $from_le(buf);
                                                }
                                                $pad_s(
                                                    block.as_mut_slice(),
                                                    64 * wi + 16 * z + 4 * y,
                                                    rx,
                                                    1,
                                                );
                                            }
                                            for x in 0..4 {
                                                $pad_s(
                                                    block.as_mut_slice(),
                                                    64 * wi + 16 * z + x,
                                                    ry,
                                                    4,
                                                );
                                            }
                                        }
                                        for y in 0..4 {
                                            for x in 0..4 {
                                                $pad_s(
                                                    block.as_mut_slice(),
                                                    64 * wi + 4 * y + x,
                                                    rz,
                                                    16,
                                                );
                                            }
                                        }
                                    }
                                    for z in 0..4 {
                                        for y in 0..4 {
                                            for x in 0..4 {
                                                $pad_s(
                                                    block.as_mut_slice(),
                                                    16 * z + 4 * y + x,
                                                    rw,
                                                    64,
                                                );
                                            }
                                        }
                                    }
                                }
                                $enc4(&mut w, &block, maxbits);
                            }
                        }
                    }
                }
                let mut out = w.finish();
                out.truncate(total_bits.div_ceil(8));
                Ok(out)
            }

            fn decompress_4d(
                compressed: &[u8],
                n3: usize,
                n2: usize,
                n1: usize,
                n0: usize,
                rate: f64,
            ) -> Result<Vec<u8>, Error> {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "rate is bits-per-value (format-bounded); rate * 256 (4D block) is a small bit budget that fits usize"
                )]
                let maxbits = (rate * 256.0) as usize;
                let nb3 = n3.div_ceil(4);
                let nb2 = n2.div_ceil(4);
                let nb1 = n1.div_ceil(4);
                let nb0 = n0.div_ceil(4);
                let mut r = BitReader::new(compressed);
                let mut output = vec![0u8; n3 * n2 * n1 * n0 * $esz];
                for b3 in 0..nb3 {
                    for b2 in 0..nb2 {
                        for b1 in 0..nb1 {
                            for b0 in 0..nb0 {
                                let w0 = b3 * 4;
                                let z0 = b2 * 4;
                                let y0 = b1 * 4;
                                let x0 = b0 * 4;
                                let block = $dec4(&mut r, maxbits)?;
                                let rw = (n3 - w0).min(4);
                                let rz = (n2 - z0).min(4);
                                let ry = (n1 - y0).min(4);
                                let rx = (n0 - x0).min(4);
                                if rw == 4 && rz == 4 && ry == 4 && rx == 4 {
                                    // Fast path: interior block. 4×4×4×4 literal
                                    // bounds plus a hoisted row slice lets LLVM
                                    // unroll and vectorize.
                                    for wi in 0..4 {
                                        for z in 0..4 {
                                            for y in 0..4 {
                                                let row_off = ((((w0 + wi) * n2 + (z0 + z)) * n1
                                                    + (y0 + y))
                                                    * n0
                                                    + x0)
                                                    * $esz;
                                                let row = &mut output[row_off..row_off + 4 * $esz];
                                                for x in 0..4 {
                                                    row[x * $esz..(x + 1) * $esz].copy_from_slice(
                                                        &block[64 * wi + 16 * z + 4 * y + x]
                                                            .to_le_bytes(),
                                                    );
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    for wi in 0..rw {
                                        for z in 0..rz {
                                            for y in 0..ry {
                                                for x in 0..rx {
                                                    let dst = (((w0 + wi) * n2 + (z0 + z)) * n1
                                                        + (y0 + y))
                                                        * n0
                                                        + (x0 + x);
                                                    let off = dst * $esz;
                                                    output[off..off + $esz].copy_from_slice(
                                                        &block[64 * wi + 16 * z + 4 * y + x]
                                                            .to_le_bytes(),
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                r.finish_decode()?;
                Ok(output)
            }
        }
    };
}

impl_codec!(
    codec_f32,
    f32,
    0.0f32,
    4,
    f32_from_le,
    pad_partial_block_f32,
    pad_strided_f32,
    encode_block_f32,
    decode_block_f32,
    encode_block_f32_2d,
    decode_block_f32_2d,
    encode_block_f32_3d,
    decode_block_f32_3d,
    encode_block_f32_4d,
    decode_block_f32_4d
);
impl_codec!(
    codec_f64,
    f64,
    0.0f64,
    8,
    f64_from_le,
    pad_partial_block_f64,
    pad_strided_f64,
    encode_block_f64,
    decode_block_f64,
    encode_block_f64_2d,
    decode_block_f64_2d,
    encode_block_f64_3d,
    decode_block_f64_3d,
    encode_block_f64_4d,
    decode_block_f64_4d
);
impl_codec!(
    codec_i32,
    i32,
    0i32,
    4,
    i32_from_le,
    pad_partial_block_i32,
    pad_strided_i32,
    encode_block_i32,
    decode_block_i32,
    encode_block_i32_2d,
    decode_block_i32_2d,
    encode_block_i32_3d,
    decode_block_i32_3d,
    encode_block_i32_4d,
    decode_block_i32_4d
);
impl_codec!(
    codec_i64,
    i64,
    0i64,
    8,
    i64_from_le,
    pad_partial_block_i64,
    pad_strided_i64,
    encode_block_i64,
    decode_block_i64,
    encode_block_i64_2d,
    decode_block_i64_2d,
    encode_block_i64_3d,
    decode_block_i64_3d,
    encode_block_i64_4d,
    decode_block_i64_4d
);

/// Compresses a chunk using fixed-rate settings from H5Z-ZFP client data.
///
/// The caller passes chunk dimensions in row-major order and the dataset's scalar type.
///
/// # Errors
///
/// Returns [`Error::ZfpFilter`] if the client data, rank, or scalar type is invalid.
/// Returns the errors from [`compress`] for invalid dimensions, rate, or input bytes.
pub fn compress_filter(
    data: &[u8],
    cd_values: &[u32],
    chunk_dims: &[u64],
    element_type: Option<ZfpElementType>,
) -> Result<Vec<u8>, Error> {
    let (rate, dims, element_type) = filter_arguments(cd_values, chunk_dims, element_type)?;
    compress(data, &dims[..chunk_dims.len()], rate, element_type)
}

/// Decodes a chunk using fixed-rate settings from H5Z-ZFP client data.
///
/// The caller passes chunk dimensions in row-major order and the dataset's scalar type.
///
/// # Errors
///
/// Returns [`Error::ZfpFilter`] if the client data, rank, or scalar type is invalid.
/// Returns the errors from [`decompress`] for invalid dimensions, rate, or encoded bytes.
pub fn decompress_filter(
    data: &[u8],
    cd_values: &[u32],
    chunk_dims: &[u64],
    element_type: Option<ZfpElementType>,
) -> Result<Vec<u8>, Error> {
    let (rate, dims, element_type) = filter_arguments(cd_values, chunk_dims, element_type)?;
    decompress(data, &dims[..chunk_dims.len()], rate, element_type)
}

fn filter_arguments(
    cd_values: &[u32],
    chunk_dims: &[u64],
    element_type: Option<ZfpElementType>,
) -> Result<(f64, [usize; 4], ZfpElementType), Error> {
    let rate = zfp_rate_from_cd_values(cd_values)
        .ok_or_else(|| Error::ZfpFilter("ZFP: invalid or non-rate cd_values".into()))?;
    let element_type = element_type.ok_or_else(|| {
        Error::ZfpFilter("ZFP: element_type missing from ChunkContext (caller must set it)".into())
    })?;
    let rank = chunk_dims.len();
    if rank == 0 || rank > 4 {
        return Err(Error::ZfpFilter(format!(
            "ZFP: chunk rank must be 1..=4, got {rank}",
        )));
    }
    validate_metadata_dimensions(chunk_dims)?;
    let mut dims = [0usize; 4];
    for (slot, &dimension) in dims.iter_mut().zip(chunk_dims) {
        *slot = dimension_to_usize(dimension)?;
    }
    Ok((rate, dims, element_type))
}

/// Compresses a chunk at a fixed number of bits per scalar.
///
/// `data` holds `dims.iter().product()` little-endian scalars of `element_type`
/// in row-major order (outer-most dimension first). `rate` is bits-per-scalar
/// and must be finite and in `(0, 8 * sizeof(element_type)]`.
///
/// # Errors
///
/// Returns [`Error::UnsupportedZfp`] if the rank is outside one to four dimensions
/// or any dimension is zero.
/// Returns [`Error::ZfpFilter`] if the rate or input length is invalid.
/// Returns [`Error::ZfpHeaderTooLarge`] if a nonzero float block needs more header bits
/// than the rate permits. Returns [`Error::ZfpSizeOverflow`] if a chunk size overflows `usize`.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "zfp")] {
/// use h5_filter::ZfpElementType;
///
/// let raw = [0u8; 16];
/// let encoded = h5_filter::compress_zfp(&raw, &[4], 16.0, ZfpElementType::F32)?;
/// assert_eq!(encoded.len(), 8);
/// assert_eq!(
///     h5_filter::decompress_zfp(&encoded, &[4], 16.0, ZfpElementType::F32)?,
///     raw,
/// );
/// # }
/// # Ok::<(), h5_filter::Error>(())
/// ```
pub fn compress(
    data: &[u8],
    dims: &[usize],
    rate: f64,
    element_type: ZfpElementType,
) -> Result<Vec<u8>, Error> {
    let dims = ZfpChunkDims::try_from(dims)?;
    validate_rate(rate, element_type)?;
    checked_codec_sizes(dims, rate, element_type)?;
    reject_short_float_header(data, dims, rate, element_type)?;
    match element_type {
        ZfpElementType::F32 => codec_f32::compress(data, dims, rate),
        ZfpElementType::F64 => codec_f64::compress(data, dims, rate),
        ZfpElementType::I32 => codec_i32::compress(data, dims, rate),
        ZfpElementType::I64 => codec_i64::compress(data, dims, rate),
    }
}

/// Decodes a fixed-rate chunk into row-major, little-endian scalars.
///
/// # Errors
///
/// Returns [`Error::UnsupportedZfp`] if the rank is outside one to four dimensions
/// or any dimension is zero.
/// Returns [`Error::ZfpFilter`] if the rate is invalid, [`Error::ZfpSizeOverflow`] if a
/// chunk size overflows `usize`, or [`Error::TruncatedZfpStream`] if the input is too short.
pub fn decompress(
    compressed: &[u8],
    dims: &[usize],
    rate: f64,
    element_type: ZfpElementType,
) -> Result<Vec<u8>, Error> {
    let dims = ZfpChunkDims::try_from(dims)?;
    validate_rate(rate, element_type)?;
    let encoded_bytes = checked_codec_sizes(dims, rate, element_type)?;
    if compressed.len() < encoded_bytes {
        return Err(Error::TruncatedZfpStream {
            expected: encoded_bytes,
            actual: compressed.len(),
        });
    }
    match element_type {
        ZfpElementType::F32 => codec_f32::decompress(compressed, dims, rate),
        ZfpElementType::F64 => codec_f64::decompress(compressed, dims, rate),
        ZfpElementType::I32 => codec_i32::decompress(compressed, dims, rate),
        ZfpElementType::I64 => codec_i64::decompress(compressed, dims, rate),
    }
}

fn validate_rate(rate: f64, element_type: ZfpElementType) -> Result<(), Error> {
    let scalar_bits = element_type.byte_width() * 8;
    if !rate.is_finite() || rate <= 0.0 || rate > scalar_bits as f64 {
        return Err(Error::ZfpFilter(format!(
            "ZFP: rate must be in (0, {scalar_bits}]; got {rate}"
        )));
    }
    Ok(())
}

fn checked_codec_sizes(
    dims: ZfpChunkDims,
    rate: f64,
    element_type: ZfpElementType,
) -> Result<usize, Error> {
    dims.checked_element_count()?
        .checked_mul(element_type.byte_width())
        .ok_or(Error::ZfpSizeOverflow)?;
    let blocks = dims.checked_block_count()?;
    let block_values = 4usize.pow(dims.rank());
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the rate is bounded by a 64-bit scalar and a block has at most 256 values"
    )]
    let maxbits = (rate * block_values as f64) as usize;
    if maxbits == 0 {
        return Err(Error::ZfpFilter(format!(
            "ZFP: rate {rate} gives no bits per block"
        )));
    }
    let bits = blocks.checked_mul(maxbits).ok_or(Error::ZfpSizeOverflow)?;
    Ok(bits.div_ceil(8))
}

fn reject_short_float_header(
    data: &[u8],
    dims: ZfpChunkDims,
    rate: f64,
    element_type: ZfpElementType,
) -> Result<(), Error> {
    let required = match element_type {
        ZfpElementType::F32 => 1 + EBITS_F32 as usize,
        ZfpElementType::F64 => 1 + EBITS_F64 as usize,
        ZfpElementType::I32 | ZfpElementType::I64 => return Ok(()),
    };
    #[expect(
        clippy::cast_possible_truncation,
        reason = "validate_rate bounds the rate by 64 bits, and a block has at most 256 values"
    )]
    let budget = (rate * 4usize.pow(dims.rank()) as f64) as usize;
    if budget >= required {
        return Ok(());
    }

    let expected = dims
        .checked_element_count()?
        .checked_mul(element_type.byte_width())
        .ok_or(Error::ZfpSizeOverflow)?;
    if data.len() != expected {
        return Ok(());
    }
    let has_header = match element_type {
        ZfpElementType::F32 => data
            .as_chunks::<4>()
            .0
            .iter()
            .any(|bytes| bytes[2] & 0x80 != 0 || bytes[3] & 0x7f != 0),
        ZfpElementType::F64 => data
            .as_chunks::<8>()
            .0
            .iter()
            .any(|bytes| bytes[6] & 0xf0 != 0 || bytes[7] & 0x7f != 0),
        ZfpElementType::I32 | ZfpElementType::I64 => false,
    };
    if has_header {
        return Err(Error::ZfpHeaderTooLarge { budget, required });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// HDF5 filter cd_values encoding, matching the H5Z-ZFP plugin layout so that
// files we write are readable by the reference plugin (and vice versa).
//
// Layout as produced by `H5Z_zfp_set_local` in `H5Zzfp.c`:
//   cd_values[0] : version word
//                  (ZFP_VERSION_NO<<16) | (ZFP_CODEC<<12) | H5Z_FILTER_ZFP_VERSION_NO
//   cd_values[1..] : a ZFP native header, written bit-by-bit into the u32
//                    slots as a little-endian bit stream. The header is:
//                      * 32-bit magic: 'z', 'f', 'p', ZFP_CODEC  (LSB first)
//                      * 52-bit meta : type + rank + per-axis sizes (see
//                        `zfp_field_metadata` in src/zfp.c). The low 2 bits
//                        are `type - 1`, next 2 are `rank - 1`, the rest
//                        hold size-1 per axis (48/24/16/12 bits each for
//                        rank 1/2/3/4).
//                      * 12-bit mode : for rate-mode with maxbits ≤ 2048,
//                        `mode = maxbits - 1` (short form).
// ---------------------------------------------------------------------------

const ZFP_VERSION_NO: u32 = 0x1010; // ZFP 1.0.1.0
const ZFP_CODEC: u32 = 5;
const H5Z_FILTER_ZFP_VERSION_NO: u32 = 0x111; // H5Z-ZFP 1.1.1

/// The dimensions of a ZFP chunk, one variant per rank the codec supports.
///
/// ZFP encodes blocks of one to four dimensions, so a chunk of any other rank has no
/// encoding. Each variant holds its dimension sizes row-major, outermost first.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ZfpChunkDims {
    One([usize; 1]),
    Two([usize; 2]),
    Three([usize; 3]),
    Four([usize; 4]),
}

impl ZfpChunkDims {
    /// Returns the number of dimensions, 1 to 4.
    fn rank(self) -> u32 {
        match self {
            Self::One(_) => 1,
            Self::Two(_) => 2,
            Self::Three(_) => 3,
            Self::Four(_) => 4,
        }
    }

    fn checked_element_count(self) -> Result<usize, Error> {
        self.checked_product(false)
    }

    fn checked_block_count(self) -> Result<usize, Error> {
        self.checked_product(true)
    }

    fn checked_product(self, blocks: bool) -> Result<usize, Error> {
        match self {
            Self::One(sizes) => checked_product(&sizes, blocks),
            Self::Two(sizes) => checked_product(&sizes, blocks),
            Self::Three(sizes) => checked_product(&sizes, blocks),
            Self::Four(sizes) => checked_product(&sizes, blocks),
        }
    }
}

fn checked_product(sizes: &[usize], blocks: bool) -> Result<usize, Error> {
    sizes.iter().try_fold(1usize, |product, &size| {
        product
            .checked_mul(if blocks { size.div_ceil(4) } else { size })
            .ok_or(Error::ZfpSizeOverflow)
    })
}

fn dimension_to_usize(value: u64) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_error| Error::ValueTooLargeForPlatform {
        value,
        target: "usize",
    })
}

impl TryFrom<&[u64]> for ZfpChunkDims {
    type Error = Error;

    /// Converts the chunk dimensions a caller passes to [`zfp_cd_values_rate`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedZfp`] if the rank is outside one to four dimensions
    /// or a dimension does not fit its metadata field. Returns
    /// [`Error::ValueTooLargeForPlatform`] if a dimension exceeds `usize`.
    fn try_from(dims: &[u64]) -> Result<Self, Error> {
        validate_metadata_dimensions(dims)?;
        Ok(match dims {
            [nx] => Self::One([dimension_to_usize(*nx)?]),
            [ny, nx] => Self::Two([dimension_to_usize(*ny)?, dimension_to_usize(*nx)?]),
            [nz, ny, nx] => Self::Three([
                dimension_to_usize(*nz)?,
                dimension_to_usize(*ny)?,
                dimension_to_usize(*nx)?,
            ]),
            [nw, nz, ny, nx] => Self::Four([
                dimension_to_usize(*nw)?,
                dimension_to_usize(*nz)?,
                dimension_to_usize(*ny)?,
                dimension_to_usize(*nx)?,
            ]),
            _ => return Err(unsupported_rank(dims.len())),
        })
    }
}

fn validate_metadata_dimensions(dims: &[u64]) -> Result<(), Error> {
    let rank = dims.len();
    if !(1..=4).contains(&rank) {
        return Err(unsupported_rank(rank));
    }
    let max = 1u64 << (48 / rank);
    for &dimension in dims {
        if dimension == 0 || dimension > max {
            return Err(Error::UnsupportedZfp(format!(
                "chunk dimension {dimension} does not fit the {rank}D ZFP metadata field (1..={max})"
            )));
        }
    }
    Ok(())
}

impl TryFrom<&[usize]> for ZfpChunkDims {
    type Error = Error;

    /// Converts the chunk dimensions a caller passes to [`compress`] or [`decompress`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedZfp`] if the rank is outside one to four dimensions
    /// or any dimension is zero.
    fn try_from(dims: &[usize]) -> Result<Self, Error> {
        if dims.contains(&0) {
            return Err(Error::UnsupportedZfp(
                "chunk dimensions must be non-zero".into(),
            ));
        }
        Ok(match *dims {
            [nx] => Self::One([nx]),
            [ny, nx] => Self::Two([ny, nx]),
            [nz, ny, nx] => Self::Three([nz, ny, nx]),
            [nw, nz, ny, nx] => Self::Four([nw, nz, ny, nx]),
            _ => return Err(unsupported_rank(dims.len())),
        })
    }
}

/// Returns the error that reports a chunk rank outside 1 to 4.
fn unsupported_rank(rank: usize) -> Error {
    Error::UnsupportedZfp(format!("only 1D-4D chunks are supported, got rank {rank}"))
}

/// Encode meta (52 bits) per `zfp_field_metadata`. `dims` is row-major
/// (outer→inner). Returns a u64 whose low 52 bits are the meta value.
fn zfp_meta_for(elem: ZfpElementType, dims: ZfpChunkDims) -> u64 {
    // zfp_type: int32=1, int64=2, float=3, double=4
    let zt: u64 = match elem {
        ZfpElementType::I32 => 1,
        ZfpElementType::I64 => 2,
        ZfpElementType::F32 => 3,
        ZfpElementType::F64 => 4,
    };
    let mut meta: u64 = 0;
    // Build via shift-and-add in the same order the reference does:
    // first the sizes in reverse row-major (nx last -> nw/nz/ny/nx), then
    // rank-1, then type-1. Because we shift left before adding, the final
    // low bits are type-1, then rank-1, then nx-1, etc. — matching what
    // the reference stream reads LSB-first.
    match dims {
        ZfpChunkDims::One([nx]) => {
            meta = (meta << 48) + (nx as u64 - 1);
        }
        ZfpChunkDims::Two([ny, nx]) => {
            // C does: <<24 ny-1, <<24 nx-1. That yields nx-1 at lower bits,
            // ny-1 at upper. Row-major `dims = [ny, nx]` so dims[0]=ny.
            meta = (meta << 24) + (ny as u64 - 1);
            meta = (meta << 24) + (nx as u64 - 1);
        }
        ZfpChunkDims::Three([nz, ny, nx]) => {
            // C: <<16 nz, <<16 ny, <<16 nx
            meta = (meta << 16) + (nz as u64 - 1);
            meta = (meta << 16) + (ny as u64 - 1);
            meta = (meta << 16) + (nx as u64 - 1);
        }
        ZfpChunkDims::Four([nw, nz, ny, nx]) => {
            // C: <<12 nw, <<12 nz, <<12 ny, <<12 nx
            meta = (meta << 12) + (nw as u64 - 1);
            meta = (meta << 12) + (nz as u64 - 1);
            meta = (meta << 12) + (ny as u64 - 1);
            meta = (meta << 12) + (nx as u64 - 1);
        }
    }
    meta = (meta << 2) + (u64::from(dims.rank()) - 1);
    meta = (meta << 2) + zt - 1;
    meta
}

/// Builds H5Z-ZFP client data for a fixed-rate chunk.
///
/// The caller passes dimensions in row-major order. H5Z-ZFP's `set_local`
/// callback writes the same client-data layout.
///
/// # Errors
///
/// Returns [`Error::UnsupportedZfp`] if a dimension does not fit its metadata field
/// or the rank is outside one to four dimensions. Returns [`Error::ZfpFilter`] if the
/// rate is invalid. Returns [`Error::ZfpSizeOverflow`] if a chunk size overflows `usize`.
/// Returns [`Error::ValueTooLargeForPlatform`] if a dimension does not fit `usize`.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "zfp")] {
/// use h5_filter::ZfpElementType;
///
/// let client_data = h5_filter::zfp_cd_values_rate(16.0, ZfpElementType::F32, &[4])?;
/// assert_eq!(h5_filter::zfp_rate_from_cd_values(&client_data), Some(16.0));
///
/// let raw = [0u8; 16];
/// let encoded = h5_filter::compress_zfp_filter(
///     &raw, &client_data, &[4], Some(ZfpElementType::F32),
/// )?;
/// assert_eq!(
///     h5_filter::decompress_zfp_filter(
///         &encoded, &client_data, &[4], Some(ZfpElementType::F32),
///     )?,
///     raw,
/// );
/// # }
/// # Ok::<(), h5_filter::Error>(())
/// ```
pub fn zfp_cd_values_rate(
    rate: f64,
    element_type: ZfpElementType,
    chunk_dims: &[u64],
) -> Result<Vec<u32>, Error> {
    let dims = ZfpChunkDims::try_from(chunk_dims)?;
    validate_rate(rate, element_type)?;
    checked_codec_sizes(dims, rate, element_type)?;
    let block_values: usize = 4usize.pow(dims.rank());
    #[expect(
        clippy::cast_possible_truncation,
        reason = "validate_rate bounds the rate by 64 bits, and a block has at most 256 values"
    )]
    let maxbits = (rate * block_values as f64) as u64;
    // Encode header bits into a buffer.
    let mut w = BitWriter::new();
    w.write(8, u64::from(b'z'));
    w.write(8, u64::from(b'f'));
    w.write(8, u64::from(b'p'));
    #[expect(
        clippy::cast_possible_truncation,
        reason = "ZFP_CODEC is the constant 5; the codec id is a single magic byte and fits u8"
    )]
    w.write(8, u64::from(ZFP_CODEC as u8));
    w.write(52, zfp_meta_for(element_type, dims));
    // Rate-mode short form: mode = maxbits - 1 for maxbits ≤ 2048.
    // Our rate × 4^rank is well inside that range for supported inputs.
    // Mode (12 bits short / 64 bits long). `zfp_stream_mode` short form for
    // fixed-rate is `maxbits - 1` when maxbits ≤ 2048; otherwise encode each
    // stream parameter separately in 64 bits (low 12 = 0xFFF sentinel).
    let header_bits: usize = if maxbits <= 2048 {
        let mode = maxbits.saturating_sub(1);
        w.write(12, mode);
        32 + 52 + 12
    } else {
        // Long form matches zfp::zfp_stream_mode for the fixed-rate
        // configuration produced by `zfp_stream_set_rate`:
        //   minbits = maxbits (+ `bits` floor for floats)
        //   maxprec = ZFP_MAX_PREC = 64
        //   minexp  = ZFP_MIN_EXP  = -1074
        let minbits = maxbits;
        let maxprec: u64 = 63; // ZFP_MAX_PREC - 1
        let minexp_enc: u64 = (-1074i64 + 16495) as u64; // 15421, fits in 15 bits
        let mut mode: u64 = 0;
        mode = (mode << 15) + minexp_enc;
        mode = (mode << 7) + maxprec;
        mode = (mode << 15) + (maxbits - 1);
        mode = (mode << 15) + (minbits - 1);
        mode = (mode << 12) + 0xFFF;
        // Write 64 bits of mode, LSB-first (write() masks to n bits).
        w.write(64, mode);
        32 + 52 + 64
    };
    let mut bytes = w.finish();
    // The BitWriter flushes 64-bit words; trim to the exact bit budget so
    // the resulting cd_values match what H5Z-ZFP writes.
    bytes.truncate(header_bits.div_ceil(8));
    // Pack the byte stream into u32 little-endian slots.
    let mut cd: Vec<u32> = Vec::with_capacity(1 + bytes.len().div_ceil(4));
    let v0 = (ZFP_VERSION_NO << 16) | (ZFP_CODEC << 12) | H5Z_FILTER_ZFP_VERSION_NO;
    cd.push(v0);
    for chunk in bytes.chunks(4) {
        let mut buf = [0u8; 4];
        buf[..chunk.len()].copy_from_slice(chunk);
        cd.push(u32::from_le_bytes(buf));
    }
    Ok(cd)
}

/// Metadata parsed from H5Z-ZFP client data in tests.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub struct ZfpFilterMeta {
    /// The scalar type stored in the metadata.
    pub element_type: ZfpElementType,
    /// The chunk dimensions in row-major order.
    pub dims: Vec<u64>,
    /// The bit budget divided by the number of values per block.
    pub rate: f64,
}

/// Parses the scalar type, dimensions, and rate from H5Z-ZFP client data in tests.
///
/// It checks the magic bytes and reads the scalar type, dimensions, and bit budget from
/// the short or long mode field. The version word and other mode parameters are ignored.
/// Returns `None` if the required bytes are missing or the magic bytes differ.
#[cfg(test)]
pub fn zfp_filter_meta_from_cd_values(cd_values: &[u32]) -> Option<ZfpFilterMeta> {
    if cd_values.len() < 4 {
        return None;
    }
    // cd_values[0] is the version word; we trust it shape-wise.
    // Treat cd_values[1..] as a little-endian byte stream.
    let mut bytes: Vec<u8> = Vec::with_capacity((cd_values.len() - 1) * 4);
    for &w in &cd_values[1..] {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    let mut r = BitReader::new(&bytes);
    // Magic (32 bits): 'z', 'f', 'p', codec.
    if r.read(8) != u64::from(b'z') || r.read(8) != u64::from(b'f') || r.read(8) != u64::from(b'p')
    {
        return None;
    }
    let _codec = r.read(8); // accept any codec version
    // Meta (52 bits).
    let meta = r.read(52);
    let zt = (meta & 0x3) + 1; // 1..=4
    let rank = ((meta >> 2) & 0x3) + 1; // 1..=4
    let element_type = match zt {
        1 => ZfpElementType::I32,
        2 => ZfpElementType::I64,
        3 => ZfpElementType::F32,
        4 => ZfpElementType::F64,
        _ => return None,
    };
    // Extract dim sizes per rank (reverse of the encoder).
    let mut dims: Vec<u64> = match rank {
        1 => {
            let n = (meta >> 4) & ((1u64 << 48) - 1);
            vec![n + 1]
        }
        2 => {
            let nx = (meta >> 4) & ((1u64 << 24) - 1);
            let ny = (meta >> 28) & ((1u64 << 24) - 1);
            vec![ny + 1, nx + 1]
        }
        3 => {
            let nx = (meta >> 4) & ((1u64 << 16) - 1);
            let ny = (meta >> 20) & ((1u64 << 16) - 1);
            let nz = (meta >> 36) & ((1u64 << 16) - 1);
            vec![nz + 1, ny + 1, nx + 1]
        }
        4 => {
            let nx = (meta >> 4) & ((1u64 << 12) - 1);
            let ny = (meta >> 16) & ((1u64 << 12) - 1);
            let nz = (meta >> 28) & ((1u64 << 12) - 1);
            let nw = (meta >> 40) & ((1u64 << 12) - 1);
            vec![nw + 1, nz + 1, ny + 1, nx + 1]
        }
        _ => return None,
    };
    // Mode: 12 bits short, extended to 64 if the 12-bit short is 0xFFF.
    let mode_short = r.read(12);
    let maxbits = if mode_short < 0xFFF {
        mode_short + 1
    } else {
        // Long mode: read 52 more bits → full 64-bit mode code that packs
        // (minbits, maxbits, maxprec, minexp). We only care about maxbits.
        let rest = r.read(52);
        let mode = mode_short | (rest << 12);
        // From zfp.c: minbits field occupies bits 12..=26 (15 bits, +1).
        //             maxbits field occupies bits 27..=41 (15 bits, +1).
        let maxbits_enc = (mode >> 27) & 0x7FFF;
        maxbits_enc + 1
    };
    let block_values = 4u64.pow(rank as u32);
    let rate = maxbits as f64 / block_values as f64;
    // Drop any dims that are zero (guard against malformed inputs).
    dims.retain(|&d| d != 0);
    if dims.len() as u64 != rank {
        return None;
    }
    r.finish_decode().ok()?;
    Some(ZfpFilterMeta {
        element_type,
        dims,
        rate,
    })
}

/// Reads a rate from the bit budget in H5Z-ZFP client data.
///
/// The parser checks the magic bytes and reads `maxbits` from a short or long mode field.
/// It uses no heap allocation. Returns `None` if the required words are missing or the
/// magic bytes differ.
pub fn zfp_rate_from_cd_values(cd_values: &[u32]) -> Option<f64> {
    if cd_values.len() < 4 {
        return None;
    }
    // cd_values[0] is the version word; the bit stream starts at cd_values[1],
    // packed LSB-first within each u32 word.
    let words = &cd_values[1..];
    // Magic: 'z','f','p', codec_version — 32 bits starting at bit 0.
    if read_bits_u32(words, 0, 8)? != u64::from(b'z')
        || read_bits_u32(words, 8, 8)? != u64::from(b'f')
        || read_bits_u32(words, 16, 8)? != u64::from(b'p')
    {
        return None;
    }
    // Meta (52 bits at bit 32). Layout per `zfp_field_metadata`:
    //   bits 0..=1 : type-1    (we ignore — caller knows the scalar type)
    //   bits 2..=3 : rank-1    (1..=4)
    //   the rest   : per-axis sizes (not needed for rate)
    let meta = read_bits_u32(words, 32, 52)?;
    let rank = ((meta >> 2) & 0x3) + 1;
    // Mode: 12 bits short at bit 84. Short form: `maxbits = mode + 1`.
    // Long form (`mode_short == 0xFFF`): extended by 52 more bits,
    // maxbits lives in bits 27..=41 of the full 64-bit mode word (+1).
    let mode_short = read_bits_u32(words, 84, 12)?;
    let maxbits = if mode_short < 0xFFF {
        mode_short + 1
    } else {
        let rest = read_bits_u32(words, 96, 52)?;
        let mode = mode_short | (rest << 12);
        let maxbits_enc = (mode >> 27) & 0x7FFF;
        maxbits_enc + 1
    };
    #[expect(
        clippy::cast_possible_truncation,
        reason = "rank is a validated 1..=4 chunk rank; 4^rank <= 256, fits u64"
    )]
    let block_values = 4u64.pow(rank as u32);
    Some(maxbits as f64 / block_values as f64)
}

/// Read `n_bits` (≤ 64) starting at `bit_pos` from the LSB-first u32
/// bit stream. Returns `None` if the request runs past the end of
/// `words`. Used by the alloc-free rate parser above.
fn read_bits_u32(words: &[u32], bit_pos: u64, n_bits: u32) -> Option<u64> {
    debug_assert!(n_bits <= 64);
    let mut out: u64 = 0;
    let mut remaining = n_bits;
    let mut pos = bit_pos;
    let mut written: u32 = 0;
    while remaining > 0 {
        let word_idx = (pos / 32) as usize;
        if word_idx >= words.len() {
            return None;
        }
        let bit_off = (pos % 32) as u32;
        let avail = 32 - bit_off;
        let take = remaining.min(avail);
        let mask = if take == 32 {
            u32::MAX as u64
        } else {
            (1u64 << take) - 1
        };
        let slice = ((words[word_idx] as u64) >> bit_off) & mask;
        out |= slice << written;
        written += take;
        remaining -= take;
        pos += take as u64;
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    // -- BitWriter / BitReader round-trip --

    #[test]
    fn bitstream_roundtrip() {
        let mut w = BitWriter::new();
        w.write(3, 0b101);
        w.write(8, 0xAB);
        w.write(1, 1);
        w.write_bit(false);
        w.write(64, 0xDEAD_BEEF_CAFE_BABE);
        let buf = w.finish();

        let mut r = BitReader::new(&buf);
        assert_eq!(r.read(3), 0b101);
        assert_eq!(r.read(8), 0xAB);
        assert_eq!(r.read(1), 1);
        assert!(!r.read_bit());
        assert_eq!(r.read(64), 0xDEAD_BEEF_CAFE_BABE);
    }

    #[test]
    fn bitreader_reports_reads_past_eof() {
        // A partial word is zero-filled for bit operations, but decoding
        // reports an attempt to consume bits beyond the supplied bytes.
        let mut r = BitReader::new(&[0xAB, 0xCD, 0xEF]);
        assert_eq!(r.read(24), 0xEF_CD_AB);
        assert_eq!(r.finish_decode(), Ok(()));
        r.read(1);
        assert_eq!(
            r.finish_decode(),
            Err(Error::TruncatedZfpStream {
                expected: 4,
                actual: 3,
            })
        );
    }

    #[test]
    fn decompress_truncated_chunk_reports_expected_length() {
        assert_eq!(
            decompress(&[0x12, 0x34], &[8usize], 4.0, ZfpElementType::F32),
            Err(Error::TruncatedZfpStream {
                expected: 4,
                actual: 2,
            })
        );
        assert_eq!(
            decompress(&[], &[8usize], 4.0, ZfpElementType::F64),
            Err(Error::TruncatedZfpStream {
                expected: 4,
                actual: 0,
            })
        );
    }

    // -- Negabinary --

    #[test]
    fn negabinary_roundtrip() {
        for x in [-1000, -1, 0, 1, 42, i32::MIN, i32::MAX] {
            assert_eq!(uint2int_i32(int2uint_i32(x)), x);
        }
    }

    // -- Lifting transform --

    #[test]
    fn lift_roundtrip() {
        let original = [100, -200, 300, -400];
        let mut p = original;
        fwd_lift_i32(&mut p);
        inv_lift_i32(&mut p);
        assert_eq!(p, original);
    }

    #[test]
    fn lift_zeros() {
        let mut p = [0, 0, 0, 0];
        fwd_lift_i32(&mut p);
        assert_eq!(p, [0, 0, 0, 0]);
        inv_lift_i32(&mut p);
        assert_eq!(p, [0, 0, 0, 0]);
    }

    // -- Block-floating-point cast --

    #[test]
    fn cast_zeros() {
        let vals = [0.0f32; 4];
        let (e, c) = fwd_cast_f32(&vals);
        assert_eq!(e, 0);
        assert_eq!(c, [0, 0, 0, 0]);
    }

    #[test]
    fn cast_roundtrip_ones() {
        let vals = [1.0f32, -1.0, 2.0, -0.5];
        let (e, c) = fwd_cast_f32(&vals);
        assert!(e > 0);
        let reconstructed = inv_cast_f32(e, &c);
        for i in 0..4 {
            let err = (vals[i] - reconstructed[i]).abs();
            assert!(
                err < 1e-6,
                "value {i}: expected {}, got {}, err={err}",
                vals[i],
                reconstructed[i]
            );
        }
    }

    // -- Full codec round-trip --

    #[test]
    fn compress_decompress_zeros() {
        let data = vec![0u8; 16]; // 4 zero f32 values
        let compressed = compress(&data, &[4], 16.0, ZfpElementType::F32).unwrap();
        let decompressed = decompress(&compressed, &[4], 16.0, ZfpElementType::F32).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn compress_decompress_ones() {
        let vals: Vec<f32> = vec![1.0; 8];
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let compressed = compress(&data, &[8], 16.0, ZfpElementType::F32).unwrap();
        let decompressed = decompress(&compressed, &[8], 16.0, ZfpElementType::F32).unwrap();
        // Lossy -- check within tolerance
        let recon: Vec<f32> = decompressed
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for (i, (&orig, &rec)) in vals.iter().zip(recon.iter()).enumerate() {
            let err = (orig - rec).abs();
            assert!(
                err < 0.01,
                "value {i}: expected {orig}, got {rec}, err={err}"
            );
        }
    }

    #[test]
    fn compress_decompress_varied() {
        // Values within similar magnitude ranges per block of 4 (ZFP uses
        // block-floating-point, so values spanning many orders of magnitude
        // within one block lose precision on the smaller ones).
        let vals: Vec<f32> = vec![
            1.0, 2.0, -1.5, 3.0, // block 0: similar magnitudes
            100.0, -50.5, 42.0, 80.0, // block 1: similar magnitudes
            0.001, 0.002, -0.003, 0.004, // block 2: similar magnitudes
        ];
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        // High rate should give good accuracy for same-magnitude blocks
        let dims = [vals.len()];
        let compressed = compress(&data, &dims, 24.0, ZfpElementType::F32).unwrap();
        let decompressed = decompress(&compressed, &dims, 24.0, ZfpElementType::F32).unwrap();
        let recon: Vec<f32> = decompressed
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for (i, (&orig, &rec)) in vals.iter().zip(recon.iter()).enumerate() {
            let rel_err = if orig.abs() > 1e-10 {
                ((orig - rec) / orig).abs()
            } else {
                (orig - rec).abs()
            };
            assert!(
                rel_err < 0.01,
                "value {i}: expected {orig}, got {rec}, rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn compress_decompress_high_rate_lossless_ish() {
        // At rate=32 (maximum), should be nearly lossless for normal f32 values
        let vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let compressed = compress(&data, &[4], 32.0, ZfpElementType::F32).unwrap();
        let decompressed = decompress(&compressed, &[4], 32.0, ZfpElementType::F32).unwrap();
        let recon: Vec<f32> = decompressed
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for (i, (&orig, &rec)) in vals.iter().zip(recon.iter()).enumerate() {
            let err = (orig - rec).abs();
            assert!(
                err < 1e-6,
                "value {i}: expected {orig}, got {rec}, err={err}"
            );
        }
    }

    #[test]
    fn cd_values_roundtrip() {
        // Integer rate at 1D f32 — short-form mode, no loss in the round-trip.
        let cd = zfp_cd_values_rate(16.0, ZfpElementType::F32, &[16]).unwrap();
        let meta = zfp_filter_meta_from_cd_values(&cd).unwrap();
        assert_eq!(meta.rate, 16.0);
        assert_eq!(meta.element_type, ZfpElementType::F32);
        assert_eq!(meta.dims, vec![16]);
    }

    #[test]
    fn cd_values_matches_reference_fixture() {
        // Probed from h5py + hdf5plugin for an f32 1D 16-value ramp at rate 16:
        // `[0x10105111, 0x0570667a, 0x000000f2, 0x03f00000]`.
        let cd = zfp_cd_values_rate(16.0, ZfpElementType::F32, &[16]).unwrap();
        assert_eq!(cd, vec![0x10105111, 0x0570667a, 0x000000f2, 0x03f00000],);
    }

    #[rstest]
    #[case(&[], "only 1D-4D chunks are supported, got rank 0")]
    #[case(&[2, 2, 2, 2, 2], "only 1D-4D chunks are supported, got rank 5")]
    fn a_chunk_rank_outside_one_to_four_is_rejected_by_the_codec(
        #[case] dims: &[usize],
        #[case] expected: &str,
    ) {
        let data = vec![0u8; 128];

        let err = compress(&data, dims, 16.0, ZfpElementType::F32).unwrap_err();
        let Error::UnsupportedZfp(reason) = &err else {
            panic!("expected UnsupportedZfp, got {err:?}");
        };
        assert_eq!(reason, expected);

        let err = decompress(&data, dims, 16.0, ZfpElementType::F32).unwrap_err();
        let Error::UnsupportedZfp(reason) = &err else {
            panic!("expected UnsupportedZfp, got {err:?}");
        };
        assert_eq!(reason, expected);
    }

    #[test]
    fn cd_values_rejects_5d_chunks() {
        assert_eq!(
            zfp_cd_values_rate(16.0, ZfpElementType::F32, &[2, 2, 2, 2, 2]),
            Err(Error::UnsupportedZfp(
                "only 1D-4D chunks are supported, got rank 5".into()
            ))
        );
    }

    #[test]
    fn cd_values_rejects_zero_rank() {
        assert_eq!(
            zfp_cd_values_rate(16.0, ZfpElementType::F32, &[]),
            Err(Error::UnsupportedZfp(
                "only 1D-4D chunks are supported, got rank 0".into()
            ))
        );
    }

    #[test]
    fn rate_from_cd_values_matches_full_parse() {
        // The alloc-free rate parser must agree with the full-meta parser
        // across every rank the codec supports. Regression for the direct
        // `&[u32]` bit reader — a sign flip or mask error would surface here.
        for rank in 1..=4 {
            for &elem in &[
                ZfpElementType::F32,
                ZfpElementType::F64,
                ZfpElementType::I32,
                ZfpElementType::I64,
            ] {
                let dims: Vec<u64> = (0..rank).map(|i| 4 + i as u64).collect();
                let rate = 12.0 + rank as f64;
                let cd = zfp_cd_values_rate(rate, elem, &dims).unwrap();
                let fast = zfp_rate_from_cd_values(&cd)
                    .unwrap_or_else(|| panic!("fast parse failed for {elem:?} rank {rank}"));
                let slow = zfp_filter_meta_from_cd_values(&cd).unwrap().rate;
                assert!(
                    (fast - slow).abs() < 1e-9,
                    "fast={fast} slow={slow} for {elem:?} rank {rank}",
                );
                assert!(
                    (fast - rate).abs() < 1e-9,
                    "fast={fast} expected={rate} for {elem:?} rank {rank}",
                );
            }
        }
    }

    #[test]
    fn rate_from_cd_values_matches_reference_fixture() {
        // Same fixture as `cd_values_matches_reference_fixture`, but exercises
        // the allocation-free parser directly.
        let cd = vec![0x10105111, 0x0570667a, 0x000000f2, 0x03f00000];
        assert_eq!(zfp_rate_from_cd_values(&cd), Some(16.0));
    }

    #[test]
    fn rate_from_cd_values_rejects_truncated_input() {
        // The parser must refuse to run off the end of a short buffer.
        let cd = vec![0x10105111, 0x0570667a]; // missing meta/mode words
        assert_eq!(zfp_rate_from_cd_values(&cd), None);
        // Fewer than 4 u32s: the version-word guard trips first.
        assert_eq!(zfp_rate_from_cd_values(&[]), None);
        assert_eq!(zfp_rate_from_cd_values(&[0; 3]), None);
    }

    #[test]
    fn rate_from_cd_values_rejects_wrong_magic() {
        // Flip one byte of the 'zfp' magic and we should reject the stream.
        let mut cd = vec![0x10105111, 0x0570667a, 0x000000f2, 0x03f00000];
        cd[1] ^= 0x1; // corrupt the LSB of 'z'
        assert_eq!(zfp_rate_from_cd_values(&cd), None);
    }

    #[rstest]
    #[case(
        12,
        "ZFP: data length 12 does not match dims product × element size (16)"
    )]
    #[case(
        20,
        "ZFP: data length 20 does not match dims product × element size (16)"
    )]
    fn compress_rejects_a_wrong_buffer_size(#[case] size: usize, #[case] reason: &str) {
        let data = vec![0u8; size];
        assert_eq!(
            compress(&data, &[4], 16.0, ZfpElementType::F32),
            Err(Error::ZfpFilter(reason.into()))
        );
    }

    #[rstest]
    #[case(f64::NAN, "ZFP: rate must be in (0, 32]; got NaN")]
    #[case(f64::INFINITY, "ZFP: rate must be in (0, 32]; got inf")]
    #[case(0.0, "ZFP: rate must be in (0, 32]; got 0")]
    #[case(-1.0, "ZFP: rate must be in (0, 32]; got -1")]
    #[case(33.0, "ZFP: rate must be in (0, 32]; got 33")]
    fn compress_rejects_a_bad_rate(#[case] rate: f64, #[case] reason: &str) {
        let data = vec![0u8; 16 * 4];
        assert_eq!(
            compress(&data, &[16], rate, ZfpElementType::F32),
            Err(Error::ZfpFilter(reason.into()))
        );
    }

    #[rstest]
    #[case(32.0, ZfpElementType::F32, 16)]
    #[case(64.0, ZfpElementType::F64, 32)]
    fn compress_rate_at_scalar_width_is_accepted(
        #[case] rate: f64,
        #[case] element_type: ZfpElementType,
        #[case] encoded_len: usize,
    ) {
        let data = vec![0u8; 4 * element_type.byte_width()];
        assert_eq!(
            compress(&data, &[4], rate, element_type).unwrap(),
            vec![0u8; encoded_len]
        );
    }

    #[rstest]
    #[case(f64::NAN, "ZFP: rate must be in (0, 32]; got NaN")]
    #[case(0.0, "ZFP: rate must be in (0, 32]; got 0")]
    #[case(33.0, "ZFP: rate must be in (0, 32]; got 33")]
    fn decompress_rejects_a_bad_rate(#[case] rate: f64, #[case] reason: &str) {
        let compressed = vec![0u8; 64];
        assert_eq!(
            decompress(&compressed, &[4], rate, ZfpElementType::F32),
            Err(Error::ZfpFilter(reason.into()))
        );
    }

    #[test]
    fn partial_block() {
        // 6 values = 1 full block + 1 partial block (2 values + 2 zeros)
        let vals: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let compressed = compress(&data, &[6], 16.0, ZfpElementType::F32).unwrap();
        let decompressed = decompress(&compressed, &[6], 16.0, ZfpElementType::F32).unwrap();
        let recon: Vec<f32> = decompressed
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(recon.len(), 6);
        for (i, (&orig, &rec)) in vals.iter().zip(recon.iter()).enumerate() {
            let err = (orig - rec).abs();
            assert!(
                err < 1.0,
                "value {i}: expected {orig}, got {rec}, err={err}"
            );
        }
    }
}
