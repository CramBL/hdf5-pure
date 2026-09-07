//! HDF5 Scale-Offset filter (filter id 6).
//!
//! Two modes, matching the only two the reference HDF5 library implements:
//!
//! * **Integer** — lossless. Each chunk's minimum is subtracted and the
//!   residuals are packed into the fewest bits that cover the chunk's range.
//! * **Float D-scale** — lossy. Values are multiplied by `10^decimals`,
//!   rounded to integers, then compressed like the integer mode.
//!
//! Ported faithfully from the reference `H5Zscaleoffset.c` so files we write
//! are read by the C library and vice versa. The on-disk compressed chunk is
//! a fixed 21-byte header followed by an MSB-first bitstream of per-element
//! offsets:
//!
//! | bytes   | meaning                                  |
//! |---------|------------------------------------------|
//! | `0..4`  | `minbits` (u32, little-endian)           |
//! | `4`     | size of the `minval` field, always `8`   |
//! | `5..13` | `minval` (u64, little-endian)            |
//! | `13..21`| zero padding                             |
//! | `21..`  | payload: each offset in `minbits` bits   |
//!
//! The payload bitstream is endianness-independent (it encodes the integer
//! offset MSB-first); the dataset byte order only governs how reconstructed
//! values are serialized.

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{format, string::ToString, vec, vec::Vec};

use crate::convert::TryToUsize;
use crate::datatype::{Datatype, DatatypeByteOrder};
use crate::error::FormatError;
use crate::fill_value::FillPattern;
use crate::filter_pipeline::FilterDescription;

// cd_values indices (H5Z_SCALEOFFSET_PARM_*).
const PARM_SCALETYPE: usize = 0;
const PARM_SCALEFACTOR: usize = 1;
const PARM_NELMTS: usize = 2;
const PARM_CLASS: usize = 3;
const PARM_SIZE: usize = 4;
const PARM_SIGN: usize = 5;
const PARM_ORDER: usize = 6;
const PARM_FILAVAIL: usize = 7;
const PARM_FILVAL: usize = 8;
/// Total number of filter parameters (`H5Z_SCALEOFFSET_TOTAL_NPARMS`).
const TOTAL_NPARMS: usize = 20;
/// The first `PARM_FILVAL` entries are the always-present "core" parameters.
const CORE_NPARMS: usize = PARM_FILVAL;

// Scale types (`H5Z_SO_scale_type_t`).
const SO_FLOAT_DSCALE: u32 = 0;
const SO_FLOAT_ESCALE: u32 = 1;
const SO_INT: u32 = 2;

// Datatype classes.
const CLS_INTEGER: u32 = 0;
const CLS_FLOAT: u32 = 1;

// Integer sign.
const SGN_NONE: u32 = 0;
const SGN_2: u32 = 1;

// Byte order.
const ORDER_LE: u32 = 0;
const ORDER_BE: u32 = 1;

// Fill-value availability.
const FILL_UNDEFINED: u32 = 0;
const FILL_DEFINED: u32 = 1;

/// Length of the fixed parameter header that precedes the bit-packed payload
/// (`buf_offset` in the reference filter). This is also the most this filter can
/// expand a chunk on the forward path: when the data does not pack smaller it
/// falls back to storing the input verbatim after the header.
pub(crate) const HEADER_LEN: usize = 21;

/// Scale-offset compression mode requested by the writer.
///
/// Mirrors the two variants the reference HDF5 library exposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleOffset {
    /// Integer scale-offset (lossless). `0` lets the encoder auto-compute the
    /// minimum bit width from each chunk's value range, which is the usual
    /// choice and the only one this encoder acts on.
    ///
    /// A value equal to the datatype's bit width selects the reference
    /// library's pass-through mode, where the filter stores the chunk
    /// unchanged. Anything in between is *recorded* in the filter's parameters
    /// and read back by [`Dataset::filter_pipeline`](crate::Dataset::filter_pipeline),
    /// but the encoder still picks each chunk's width from its own range — it
    /// never packs narrower than the data needs, where the reference would and
    /// would truncate values that did not fit.
    Integer(u32),
    /// Floating-point decimal scaling (lossy). The value is the number of
    /// decimal digits of precision retained (the "D" scale factor).
    FloatDScale(i32),
}

/// Datatype facts the writer needs to assemble scale-offset `cd_values`.
///
/// Derived from a dataset's [`Datatype`] via
/// [`scale_offset_type_from_datatype`]. Carried on
/// [`ChunkContext`](crate::filters::ChunkContext) the same way ZFP's scalar
/// type is, so the write path can build the filter parameters without
/// re-deriving them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScaleOffsetType {
    /// `CLS_INTEGER` or `CLS_FLOAT`.
    class: u32,
    /// `SGN_NONE` or `SGN_2` (only meaningful for integers).
    sign: u32,
    /// `ORDER_LE` or `ORDER_BE`.
    order: u32,
}

/// Map an HDF5 `Datatype` to the facts scale-offset needs, if the type is a
/// fixed-point or floating-point scalar with a definite byte order. Returns
/// `None` for any other class (compound, string, …) or an indeterminate order.
pub fn scale_offset_type_from_datatype(dt: &Datatype) -> Option<ScaleOffsetType> {
    match dt {
        Datatype::FixedPoint {
            size,
            byte_order,
            signed,
            ..
        } if matches!(*size, 1 | 2 | 4 | 8) => Some(ScaleOffsetType {
            class: CLS_INTEGER,
            sign: if *signed { SGN_2 } else { SGN_NONE },
            order: order_code(byte_order)?,
        }),
        Datatype::FloatingPoint {
            size, byte_order, ..
        } if matches!(*size, 4 | 8) => Some(ScaleOffsetType {
            class: CLS_FLOAT,
            sign: SGN_NONE,
            order: order_code(byte_order)?,
        }),
        _ => None,
    }
}

fn order_code(order: &DatatypeByteOrder) -> Option<u32> {
    match order {
        DatatypeByteOrder::LittleEndian => Some(ORDER_LE),
        DatatypeByteOrder::BigEndian => Some(ORDER_BE),
        // Scale-offset only supports definite little/big endian, matching the
        // reference filter's `can_apply` check.
        DatatypeByteOrder::Vax => None,
    }
}

/// Whether a scale-offset filter records a fill value
/// (`H5Z_SCALEOFFSET_PARM_FILAVAIL`), without the value itself — the form that
/// can be stored in a dataset's write options, which own no borrowed bytes.
///
/// The reference library derives this from the dataset's fill value and answers
/// [`Defined`](Self::Defined) unless the caller set that value to *undefined*
/// explicitly, so a dataset that never mentions a fill value still gets a
/// defined one, of zero. This crate's writer cannot express an undefined fill
/// value at all — `DatasetBuilder`'s fill is "a value or the library default" —
/// so [`Defined`](Self::Defined) is what it writes and the other variant exists
/// for one caller: [`repack`](crate::repack), reproducing a source file whose
/// filter recorded `FILL_UNDEFINED`. The day the writer models an undefined
/// fill value, this collapses back into the fill value itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillAvailability {
    /// `FILL_DEFINED`: the filter records the dataset's fill value, and encodes
    /// elements equal to it as the reserved sentinel.
    Defined,
    /// `FILL_UNDEFINED`: the filter records no fill value, and every element is
    /// encoded as an ordinary offset.
    Undefined,
}

impl FillAvailability {
    /// The filter's fill parameters for a dataset whose fill value is `fill`.
    ///
    /// The pattern is read only by [`Defined`](Self::Defined), which is the
    /// variant that records it: [`Undefined`](Self::Undefined) has nowhere to
    /// put a value, so one it could not read
    /// ([`FormatError::UnreadableFillValue`]) is not an obstacle to recording
    /// that there is none.
    pub fn with_value(self, fill: FillPattern<'_>) -> Result<ScaleOffsetFill<'_>, FormatError> {
        match self {
            Self::Defined => Ok(ScaleOffsetFill::Defined(fill.element()?)),
            Self::Undefined => Ok(ScaleOffsetFill::Undefined),
        }
    }
}

/// The fill-value parameters a scale-offset filter records: availability
/// (`H5Z_SCALEOFFSET_PARM_FILAVAIL`) together with the value
/// (`H5Z_SCALEOFFSET_PARM_FILVAL`) when there is one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleOffsetFill<'a> {
    /// Record `FILL_DEFINED` over one element of fill bytes in the dataset's
    /// byte order, or over the library-default fill value — an all-zero element
    /// — for `None`. The reference library records a defined *zero* for a
    /// dataset whose fill value was never set, so the two are the same
    /// parameters, not two cases.
    Defined(Option<&'a [u8]>),
    /// Record `FILL_UNDEFINED`, leaving the value parameters zero.
    Undefined,
}

/// Build the 20-entry `cd_values` array for a scale-offset filter.
///
/// `nelmts` is the number of elements in one chunk. Validates that the
/// requested [`ScaleOffset`] mode matches the datatype class (integer mode on
/// integer data, float D-scale on float data).
///
/// The fill value is stored the way `H5Z__scaleoffset_set_parms_fillval` stores
/// it: as the datatype's bit pattern in *value* order rather than in the
/// dataset's byte order, packed four bytes to a `cd_values` entry from
/// [`PARM_FILVAL`]. [`read_fill_bits`] is the inverse, and a big-endian dataset
/// is converted in both directions.
///
/// Storing the *value* rather than the dataset's bytes is what makes this
/// independent of the host: the reference reaches the same entries by two
/// different routes, walking the fill value forwards on a little-endian host
/// and backwards from its last four bytes on a big-endian one, and its
/// `need_convert` step brings a foreign-order fill value into that form first.
pub fn build_cd_values(
    mode: ScaleOffset,
    ty: ScaleOffsetType,
    size: u32,
    nelmts: u32,
    fill: ScaleOffsetFill<'_>,
) -> Result<Vec<u32>, FormatError> {
    let (scale_type, scale_factor) = match (mode, ty.class) {
        (ScaleOffset::Integer(minbits), CLS_INTEGER) => (SO_INT, minbits),
        (ScaleOffset::FloatDScale(decimals), CLS_FLOAT) => (SO_FLOAT_DSCALE, decimals as u32),
        (ScaleOffset::Integer(_), _) => {
            return Err(FormatError::FilterError(
                "scaleoffset: integer mode requires an integer dataset".to_string(),
            ));
        }
        (ScaleOffset::FloatDScale(_), _) => {
            return Err(FormatError::FilterError(
                "scaleoffset: float D-scale mode requires a floating-point dataset".to_string(),
            ));
        }
    };

    // The same 1..=8-byte scalars [`Parms::parse`] accepts on the way back in.
    // Stated here rather than left to the datatype that produced `ty`, because
    // the two arrive as separate arguments: a wider one would run the fill value
    // off the end of the parameter array, which is what makes
    // [`write_fill_bits`] safe to write without a bound of its own.
    if size == 0 || size > 8 {
        return Err(FormatError::FilterError(format!(
            "scaleoffset: unsupported datatype size {size}"
        )));
    }

    let mut cd = vec![0u32; TOTAL_NPARMS];
    cd[PARM_SCALETYPE] = scale_type;
    cd[PARM_SCALEFACTOR] = scale_factor;
    cd[PARM_NELMTS] = nelmts;
    cd[PARM_CLASS] = ty.class;
    cd[PARM_SIZE] = size;
    cd[PARM_SIGN] = ty.sign;
    cd[PARM_ORDER] = ty.order;
    match fill {
        ScaleOffsetFill::Undefined => cd[PARM_FILAVAIL] = FILL_UNDEFINED,
        ScaleOffsetFill::Defined(bytes) => {
            let width = size.to_usize()?;
            let bits = match bytes {
                // The library default: a defined fill value of zero.
                None => 0,
                Some(b) if b.len() == width => read_value(b, width, ty.order),
                Some(b) => {
                    return Err(FormatError::FilterError(format!(
                        "scaleoffset: fill value is {} bytes for a {width}-byte datatype",
                        b.len()
                    )));
                }
            };
            cd[PARM_FILAVAIL] = FILL_DEFINED;
            write_fill_bits(&mut cd, bits, width);
        }
    }
    Ok(cd)
}

/// Recover the [`ScaleOffset`] mode a parsed filter encodes from its
/// `cd_values`, along with whether it records a fill value, so a tool like
/// [`repack`](crate::repack) can re-apply both.
///
/// Returns `None` if the parameter array is too short, names a scale type this
/// crate never writes (the reference library's float *E*-scale), or records
/// neither fill availability — a value the reference never writes and whose
/// meaning is undefined.
///
/// The fill *value* is not returned with the availability, because re-applying
/// it is not this parameter's job: the rebuilt filter takes its value from the
/// dataset's own fill value, which the writer records the way the reference
/// does. A source whose filter disagreed with its Fill Value message is
/// therefore re-emitted agreeing with it — the two encodings hold the same
/// elements, since each decodes its sentinel back to the value its own
/// parameters name.
pub(crate) fn scale_offset_mode(cd_values: &[u32]) -> Option<(ScaleOffset, FillAvailability)> {
    let scale_type = *cd_values.get(PARM_SCALETYPE)?;
    let scale_factor = *cd_values.get(PARM_SCALEFACTOR)?;
    let fill = match *cd_values.get(PARM_FILAVAIL)? {
        FILL_UNDEFINED => FillAvailability::Undefined,
        FILL_DEFINED => FillAvailability::Defined,
        _ => return None,
    };
    let mode = match scale_type {
        SO_INT => ScaleOffset::Integer(scale_factor),
        #[expect(
            clippy::cast_possible_wrap,
            reason = "scale_factor is a small decimal scale factor (D-scale parameter); the reference treats it as a signed int"
        )]
        SO_FLOAT_DSCALE => ScaleOffset::FloatDScale(scale_factor as i32),
        // SO_FLOAT_ESCALE (and anything unrecognized) is never written by this
        // crate and cannot be reproduced.
        _ => return None,
    };
    Some((mode, fill))
}

/// Decoded scale-offset parameters shared by the compress and decompress paths.
struct Parms {
    scale_type: u32,
    scale_factor: i32,
    nelmts: usize,
    class: u32,
    size: usize,
    order: u32,
    filavail: u32,
}

impl Parms {
    fn parse(cd: &[u32]) -> Result<Parms, FormatError> {
        if cd.len() < CORE_NPARMS {
            return Err(FormatError::FilterError(
                "scaleoffset: too few cd_values".to_string(),
            ));
        }
        let class = cd[PARM_CLASS];
        if class != CLS_INTEGER && class != CLS_FLOAT {
            return Err(FormatError::FilterError(
                "scaleoffset: unsupported datatype class".to_string(),
            ));
        }
        let size = cd[PARM_SIZE] as usize;
        if size == 0 || size > 8 {
            return Err(FormatError::FilterError(
                "scaleoffset: unsupported datatype size".to_string(),
            ));
        }
        if class == CLS_FLOAT && size != 4 && size != 8 {
            return Err(FormatError::FilterError(
                "scaleoffset: float size must be 4 or 8".to_string(),
            ));
        }
        let order = cd[PARM_ORDER];
        if order != ORDER_LE && order != ORDER_BE {
            return Err(FormatError::FilterError(
                "scaleoffset: bad byte order".to_string(),
            ));
        }
        #[expect(
            clippy::cast_possible_wrap,
            reason = "scale_factor is a small filter parameter (decimal scale factor or bit width); the reference stores it in a signed int"
        )]
        Ok(Parms {
            scale_type: cd[PARM_SCALETYPE],
            scale_factor: cd[PARM_SCALEFACTOR] as i32,
            nelmts: cd[PARM_NELMTS].to_usize()?,
            class,
            size,
            order,
            filavail: cd[PARM_FILAVAIL],
        })
    }

    /// Bit mask covering the datatype's full width.
    fn width_mask(&self) -> u64 {
        if self.size >= 8 {
            u64::MAX
        } else {
            (1u64 << (self.size * 8)) - 1
        }
    }
}

/// The two `scale_factor` decisions the reference makes **before** it splits on
/// compress versus decompress (`H5Zscaleoffset.c`, the `H5Z_FLAG_REVERSE` test),
/// so they hold identically in both directions: a requested width wider than the
/// datatype is an error, and a width *equal* to it makes the filter a
/// pass-through that stores the chunk unchanged, with no header of its own.
///
/// Float D-scale is exempt, because there `scale_factor` is a decimal exponent
/// rather than a bit width.
///
/// Returns `Ok(true)` when the caller should hand the chunk through untouched.
///
/// Both callers share this rather than testing it themselves: implementing one
/// direction and not the other is exactly the defect the pass-through fixes —
/// a chunk packed by a writer that ignores the mode and handed back verbatim by
/// a reader that honours it decodes as garbage in both libraries.
fn pass_through_or_refuse(p: &Parms) -> Result<bool, FormatError> {
    if p.scale_type == SO_FLOAT_DSCALE {
        return Ok(false);
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "p.size is validated to 1..=8, so p.size * 8 is 8..=64 and fits in u32"
    )]
    let type_bits = (p.size * 8) as u32;
    #[expect(
        clippy::cast_possible_wrap,
        reason = "type_bits is 8..=64, so it fits in a non-negative i32"
    )]
    let type_bits_i = type_bits as i32;
    if p.scale_factor > type_bits_i {
        return Err(FormatError::FilterError(format!(
            "scaleoffset: requested minimum bit width {} exceeds the {type_bits}-bit datatype",
            p.scale_factor
        )));
    }
    Ok(p.scale_factor == type_bits_i)
}

/// Decompress one scale-offset chunk into raw element bytes (in the dataset's
/// stored byte order).
pub fn decompress(
    input: &[u8],
    filter: &FilterDescription,
    max_output: Option<usize>,
) -> Result<Vec<u8>, FormatError> {
    let cd = &filter.client_data;
    let p = Parms::parse(cd)?;

    if p.scale_type == SO_FLOAT_ESCALE {
        return Err(FormatError::FilterError(
            "scaleoffset: float E-scale method is not supported".to_string(),
        ));
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "p.size is validated to 1..=8, so p.size * 8 is 8..=64 and fits in u32"
    )]
    let full_bits = (p.size * 8) as u32;
    // `nelmts` is file-derived; `nelmts * size` is an allocation size and a
    // slice bound. The product is computed in `u64` (it cannot overflow there:
    // `nelmts` originates from a `u32` and `size` is 1..=8) and narrowed to
    // `usize`, which errors instead of truncating on a 32-bit host.
    let size_out = (p.nelmts as u64 * p.size as u64).to_usize()?;

    // `size_out` sizes an allocation, and `nelmts` comes from the filter's own
    // cd_values rather than from the chunk geometry, so nothing has yet tied it
    // to a plausible size. Unlike the byte compressors this decoder cannot bound
    // itself by the input length — the `minbits == 0` path below expands a
    // header into `size_out` bytes with no payload at all — so the check is
    // against the size the pipeline expects of this stage. Scale-offset decodes
    // last, so that is the chunk size itself; a chunk claiming more than the
    // whole chunk is refused here rather than after the memory is committed.
    if let Some(cap) = max_output
        && size_out > cap
    {
        return Err(FormatError::FilterError(format!(
            "scaleoffset: chunk declares {size_out} decoded bytes, more than the \
             {cap} the chunk can hold"
        )));
    }

    // Pass-through mode, and the width refusal beside it; see
    // [`pass_through_or_refuse`].
    if pass_through_or_refuse(&p)? {
        return Ok(input.to_vec());
    }

    // Header: minbits + minval.
    if input.len() < 5 {
        return Err(FormatError::FilterError(
            "scaleoffset: chunk shorter than header".to_string(),
        ));
    }
    let minbits = u32::from_le_bytes([input[0], input[1], input[2], input[3]]);
    if minbits > full_bits {
        return Err(FormatError::FilterError(
            "scaleoffset: minbits exceeds datatype size".to_string(),
        ));
    }
    let minval_size = (input[4] as usize).min(8);
    if input.len() < 5 + minval_size {
        return Err(FormatError::FilterError(
            "scaleoffset: chunk too short for minval".to_string(),
        ));
    }
    let mut minval_bytes = [0u8; 8];
    minval_bytes[..minval_size].copy_from_slice(&input[5..5 + minval_size]);
    let minval = u64::from_le_bytes(minval_bytes);

    // Raw payload (no per-element packing): minbits at full precision.
    if minbits == full_bits {
        let start = HEADER_LEN;
        if input.len() < start + size_out {
            return Err(FormatError::FilterError(
                "scaleoffset: chunk too short for raw payload".to_string(),
            ));
        }
        return Ok(input[start..start + size_out].to_vec());
    }

    // minbits == 0: there is no payload, so every element takes the same value.
    // Which value is not the same in both fill modes, and the reference reaches
    // the answer by not special-casing this at all — `H5Z__filter_scaleoffset`
    // zeroes the output buffer and still runs its postdecompress, where the
    // sentinel `(1 << 0) - 1` is zero and therefore matches every zeroed
    // element. So a fill-*defined* chunk decodes entirely to the fill value,
    // and only a fill-undefined one decodes to `minval`.
    //
    // No conforming encoder emits this pair — reserving the sentinel keeps
    // `minbits` at 1 or more whenever a fill value is defined — so it is only
    // reachable from a hand-built or damaged chunk. Measured against the C
    // library rather than reasoned about, in
    // `minbits_zero_with_fill_defined_reads_as_the_fill_value`.
    let mut out = Vec::with_capacity(size_out);
    if minbits == 0 {
        let mask = p.width_mask();
        // `read_fill_bits` writes only `size` bytes into a zeroed buffer, so the
        // recovered value never carries bits above the datatype width and needs
        // no mask of its own — unlike `minval`, which comes off the chunk header
        // as a full 8 bytes whatever the datatype is.
        let bits = if p.filavail == FILL_DEFINED {
            read_fill_bits(cd, p.size)?
        } else {
            minval & mask
        };
        for _ in 0..p.nelmts {
            write_value(&mut out, bits, p.size, p.order);
        }
        return Ok(out);
    }

    if input.len() < HEADER_LEN {
        return Err(FormatError::FilterError(
            "scaleoffset: chunk too short for packed payload".to_string(),
        ));
    }
    let payload = &input[HEADER_LEN..];
    // Validate payload length once, up front; the BitReader trusts the
    // caller to not read past the end. Compare in `u64` so the bit counts
    // (`nelmts` is file-derived, `minbits` up to 64) cannot overflow a 32-bit
    // `usize` and falsely pass the check.
    if (payload.len() as u64) * 8 < p.nelmts as u64 * minbits as u64 {
        return Err(FormatError::FilterError(
            "scaleoffset: payload too short for packed data".to_string(),
        ));
    }
    let mut reader = BitReader::new(payload);

    if p.class == CLS_INTEGER {
        reconstruct_integer(&mut out, &mut reader, &p, minbits, minval, cd)?;
    } else {
        reconstruct_float(&mut out, &mut reader, &p, minbits, minval, cd)?;
    }
    Ok(out)
}

/// Compress one full chunk of raw element bytes with scale-offset.
///
/// `input` must be exactly `nelmts * size` bytes (the chunk writer always pads
/// edge chunks to full size).
///
/// Both fill-value modes are produced. When the filter records `FILL_DEFINED`
/// — which is what the reference library writes for any dataset carrying a fill
/// value, and so what h5py files routinely carry — elements equal to the fill
/// value are excluded from the chunk's range and stored as the all-ones offset
/// instead of as `value - min`. `minbits` is widened by one code point so that
/// sentinel cannot also be a legitimate offset. See [`precompress_integer`].
pub fn compress(input: &[u8], filter: &FilterDescription) -> Result<Vec<u8>, FormatError> {
    let cd = &filter.client_data;
    let p = Parms::parse(cd)?;

    // The class/scale-type agreement is checked first, matching the reference's
    // order — the pass-through below is an early return, so anything it must not
    // skip has to come above it.
    if p.class == CLS_INTEGER && p.scale_type != SO_INT {
        return Err(FormatError::FilterError(
            "scaleoffset: integer class requires integer scale type".to_string(),
        ));
    }
    if p.class == CLS_FLOAT && p.scale_type != SO_FLOAT_DSCALE {
        return Err(FormatError::FilterError(
            "scaleoffset: float class requires D-scale scale type".to_string(),
        ));
    }

    if pass_through_or_refuse(&p)? {
        return Ok(input.to_vec());
    }

    // `nelmts * size` in `u64` (cannot overflow there) narrowed to `usize`,
    // so a file-derived `nelmts` cannot wrap a 32-bit `usize` and spuriously
    // equal `input.len()`.
    let expected = (p.nelmts as u64 * p.size as u64).to_usize()?;
    if input.len() != expected {
        return Err(FormatError::FilterError(
            "scaleoffset: chunk size does not match nelmts * datatype size".to_string(),
        ));
    }
    if p.nelmts == 0 {
        return Ok(emit(0, 0, &[]));
    }

    let signed = cd[PARM_SIGN] == SGN_2;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "p.size is validated to 1..=8, so p.size * 8 is 8..=64 and fits in u32"
    )]
    let full_bits = (p.size * 8) as u32;

    // The fill value the filter parameters carry, or `None` when they record
    // `FILL_UNDEFINED`. Read once here rather than per chunk-scan pass.
    let filval = if p.filavail == FILL_DEFINED {
        Some(read_fill_bits(cd, p.size)?)
    } else {
        None
    };

    let (minbits, minval, offsets) = if p.class == CLS_INTEGER {
        precompress_integer(input, &p, signed, filval)
    } else {
        precompress_float(input, &p, filval)
    };

    // Raw path: store the original element bytes after the header.
    if minbits >= full_bits {
        return Ok(emit_raw(full_bits, minval, input));
    }
    let payload = pack_offsets(&offsets, minbits, p.nelmts)?;
    Ok(emit(minbits, minval, &payload))
}

// --- integer reconstruction / pre-compression ----------------------------

fn reconstruct_integer(
    out: &mut Vec<u8>,
    reader: &mut BitReader<'_>,
    p: &Parms,
    minbits: u32,
    minval: u64,
    cd: &[u32],
) -> Result<(), FormatError> {
    let mask = p.width_mask();
    let sentinel = sentinel(minbits);
    // Hoist `fv & mask` out of the per-element branch.
    let filval = if p.filavail == FILL_DEFINED {
        Some(read_fill_bits(cd, p.size)? & mask)
    } else {
        None
    };
    for _ in 0..p.nelmts {
        let d = reader.read(minbits);
        let bits = match filval {
            Some(fv) if d == sentinel => fv,
            _ => d.wrapping_add(minval) & mask,
        };
        write_value(out, bits, p.size, p.order);
    }
    Ok(())
}

/// `filval` is the raw bit pattern of the dataset's fill value when the filter
/// records `FILL_DEFINED`, and `None` otherwise. A defined fill value changes
/// the encoding in three places, all mirroring `H5Z_scaleoffset_precompress_1`
/// and `_2`:
///
/// * the chunk's range is taken over the **non-fill** elements only (a chunk
///   that is nothing but fill values leaves `min == max == 0`, the reference's
///   initializers);
/// * `minbits` covers `span + 1` rather than `span`, reserving one code point
///   so the all-ones sentinel can never also be a legitimate offset;
/// * each fill-valued element is stored as that sentinel instead of as
///   `value - min`.
fn precompress_integer(
    input: &[u8],
    p: &Parms,
    signed: bool,
    filval: Option<u64>,
) -> (u32, u64, Vec<u64>) {
    // Stream the input twice (min/max, then offsets) instead of materializing
    // a Vec<i128>. The intermediate dominated cache pressure on large chunks
    // (16 B/element vs the source's 1-8 B/element); two cache-warm passes
    // come out ahead.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "p.size is validated to 1..=8, so p.size * 8 is 8..=64 and fits in u32"
    )]
    let full_bits = (p.size * 8) as u32;
    let read_as_i128 = |i: usize| -> i128 {
        let bits = read_value(&input[i * p.size..(i + 1) * p.size], p.size, p.order);
        if signed {
            sign_extend(bits, p.size) as i128
        } else {
            bits as i128
        }
    };
    // Compare the fill value the way the elements are read, so the equality
    // test is on values rather than on bit patterns of different widths.
    let filval = filval.map(|bits| {
        if signed {
            sign_extend(bits, p.size) as i128
        } else {
            (bits & p.width_mask()) as i128
        }
    });

    let (min, max) = match filval {
        None => {
            let first = read_as_i128(0);
            let (mut min, mut max) = (first, first);
            for i in 1..p.nelmts {
                let v = read_as_i128(i);
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
            (min, max)
        }
        Some(fv) => {
            // `H5Z_scaleoffset_max_min_1`: skip fill values, and leave both
            // bounds at zero when every element is one.
            let (mut min, mut max) = (0i128, 0i128);
            let mut seen = false;
            for i in 0..p.nelmts {
                let v = read_as_i128(i);
                if v == fv {
                    continue;
                }
                if !seen {
                    min = v;
                    max = v;
                    seen = true;
                    continue;
                }
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
            (min, max)
        }
    };

    #[expect(
        clippy::cast_possible_truncation,
        reason = "min is an element value read from at most 8 bytes (p.size 1..=8), so it fits in i64/u64 despite the i128 accumulator type"
    )]
    let minval = if signed {
        (min as i64) as u64
    } else {
        min as u64
    };

    // Overflow guard mirrors `H5Z_scaleoffset_check_{1,2}`: a span within two
    // of the full range can't gain from packing, so store at full precision.
    //
    // The reference reaches this by an early `return` from the precompress
    // function, which skips the `*minval = min` its packed paths end with — so
    // the header carries the zero the caller initialized, not the chunk's
    // minimum.
    //
    // With one exception, and it is an accident of the reference's own source
    // rather than a rule: `signed char` is the one type `H5Z__scaleoffset_
    // precompress_i` hand-expands instead of routing through the
    // `H5Z_scaleoffset_check_2` macro, and the expansion assigns `*minval` in
    // its fill-*undefined* early return and not in its fill-defined one. So a
    // 1-byte signed chunk with no fill value carries the minimum here where
    // every other combination carries zero.
    //
    // A decoder ignores `minval` once `minbits` is the full width, so none of
    // this changes what any reader returns; it is what writing the same bytes
    // as the reference costs.
    let width_max: u128 = p.width_mask() as u128;
    let spread = (max - min) as u128;
    if spread > width_max.saturating_sub(2) {
        let schar_quirk = signed && p.size == 1 && filval.is_none();
        return (full_bits, if schar_quirk { minval } else { 0 }, Vec::new());
    }

    // `span` is the reference's `max - min + 1`; a defined fill value costs one
    // more code point for the sentinel. The guard above leaves room for both:
    // `spread <= width_max - 2` makes `spread + 2 <= width_max`.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the guard above returns early unless spread <= width_max - 2, and width_max <= u64::MAX, so spread + 2 fits in u64"
    )]
    let span = (spread as u64) + 1;
    let minbits = ceil_log2(if filval.is_some() { span + 1 } else { span });
    // Unlike the spread guard above, this return is reached with `min` computed,
    // and the reference's trailing `*minval = min` does run here — the two
    // full-precision returns carry different headers.
    //
    // The comparison is `>=` rather than `>` only to skip building offsets that
    // `compress` would discard: it applies the same `minbits >= full_bits` test
    // to the value returned here and emits the chunk raw either way, so
    // loosening this one changes nothing but wasted work.
    if minbits >= full_bits {
        return (full_bits, minval, Vec::new());
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "each offset (value - min) is at most spread <= width_max <= u64::MAX, so it fits in u64"
    )]
    let offsets = match filval {
        None => (0..p.nelmts)
            .map(|i| (read_as_i128(i) - min) as u64)
            .collect(),
        Some(fv) => {
            let sentinel = sentinel(minbits);
            (0..p.nelmts)
                .map(|i| {
                    let v = read_as_i128(i);
                    if v == fv { sentinel } else { (v - min) as u64 }
                })
                .collect()
        }
    };
    (minbits, minval, offsets)
}

// --- float (D-scale) reconstruction / pre-compression ---------------------

fn reconstruct_float(
    out: &mut Vec<u8>,
    reader: &mut BitReader<'_>,
    p: &Parms,
    minbits: u32,
    minval: u64,
    cd: &[u32],
) -> Result<(), FormatError> {
    let sentinel = sentinel(minbits);
    let decimals = p.scale_factor;
    let filval = if p.filavail == FILL_DEFINED {
        Some(read_fill_bits(cd, p.size)?)
    } else {
        None
    };
    // Dispatch on width once: pow, min, and the size/order args are all
    // loop-invariant. The compiler won't hoist the inner `match p.size`
    // across both branches by itself.
    if p.size == 4 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "size == 4 branch: minval holds a 4-byte f32 bit pattern in its low 32 bits, so narrowing to u32 reconstructs that pattern"
        )]
        let min = f32::from_bits(minval as u32);
        let pow = pow10_f32(decimals);
        for _ in 0..p.nelmts {
            let d = reader.read(minbits);
            #[expect(
                clippy::cast_possible_wrap,
                reason = "d is a minbits-wide offset (minbits < full_bits <= 64); reinterpreting it as i64 matches the reference's signed reconstruction"
            )]
            let bits = if let Some(fv) = filval.filter(|_| d == sentinel) {
                fv
            } else {
                ((d as i64 as f32) / pow + min).to_bits() as u64
            };
            write_value(out, bits, 4, p.order);
        }
    } else {
        let min = f64::from_bits(minval);
        let pow = pow10_f64(decimals);
        for _ in 0..p.nelmts {
            let d = reader.read(minbits);
            #[expect(
                clippy::cast_possible_wrap,
                reason = "d is a minbits-wide offset (minbits < full_bits <= 64); reinterpreting it as i64 matches the reference's signed reconstruction"
            )]
            let bits = if let Some(fv) = filval.filter(|_| d == sentinel) {
                fv
            } else {
                ((d as i64 as f64) / pow + min).to_bits()
            };
            write_value(out, bits, 8, p.order);
        }
    }
    Ok(())
}

/// The float counterpart of [`precompress_integer`]. A defined fill value
/// widens `minbits` and diverts matching elements to the sentinel the same way,
/// with one difference worth stating: the reference does **not** match a float
/// element against the fill value by equality but by *proximity*, treating any
/// element within one decimal quantum (`|v - fill| < 10^-D`) as a fill value.
///
/// It also applies that test at two different precisions for `f32` data —
/// `H5Z_scaleoffset_max_min_3` hardcodes the `double` `fabs`/`pow` while
/// `H5Z_scaleoffset_modify_1` uses the type's own `fabsf`/`powf` — so an `f32`
/// element can sit inside one tolerance and outside the other. Both are
/// mirrored as written rather than reconciled, since agreeing with the C
/// encoder is the whole point.
fn precompress_float(input: &[u8], p: &Parms, filval: Option<u64>) -> (u32, u64, Vec<u64>) {
    // Two streaming passes over the input bytes (min/max, then offsets),
    // no intermediate Vec<f32>/Vec<f64>. `min_scaled = min * pow` is hoisted
    // out of the per-element loop so we don't recompute the same product N
    // times — bit-identical to the previous `v * pow - min * pow`.
    let decimals = p.scale_factor;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "p.size is validated to 1..=8, so p.size * 8 is 8..=64 and fits in u32"
    )]
    let full_bits = (p.size * 8) as u32;
    if p.size == 4 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "read_value with size 4 returns a u64 whose meaningful bits are the low 32, so narrowing to u32 reconstructs the f32 bit pattern"
        )]
        let read_f32 = |i: usize| -> f32 {
            f32::from_bits(read_value(&input[i * 4..i * 4 + 4], 4, p.order) as u32)
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the fill value of a 4-byte float occupies the low 32 bits of the recovered pattern"
        )]
        let fill = filval.map(|bits| f32::from_bits(bits as u32));
        // `max_min_3`'s tolerance: the difference is taken in the element type,
        // then widened to double and compared against a double `pow`.
        let scan_tol = pow10_f64(-decimals);
        let is_fill_in_scan = |v: f32| fill.is_some_and(|fv| f64::from(v - fv).abs() < scan_tol);

        let (min, max) = match fill {
            None => {
                let first = read_f32(0);
                let (mut min, mut max) = (first, first);
                for i in 1..p.nelmts {
                    let v = read_f32(i);
                    if v < min {
                        min = v;
                    }
                    if v > max {
                        max = v;
                    }
                }
                (min, max)
            }
            Some(_) => {
                let (mut min, mut max) = (0f32, 0f32);
                let mut seen = false;
                for i in 0..p.nelmts {
                    let v = read_f32(i);
                    if is_fill_in_scan(v) {
                        continue;
                    }
                    if !seen {
                        min = v;
                        max = v;
                        seen = true;
                        continue;
                    }
                    if v < min {
                        min = v;
                    }
                    if v > max {
                        max = v;
                    }
                }
                (min, max)
            }
        };

        let pow = pow10_f32(decimals);
        let min_scaled = min * pow;
        // check_3: residual span beyond signed 32-bit range → store raw.
        let residual = max * pow - min_scaled;
        let minval = min.to_bits() as u64;
        // `check_3` jumps past the `save_min` at the end of the reference's
        // precompress, so the header keeps the zero set on entry. Same as the
        // integer guard above.
        if residual > (1u64 << 31) as f32 {
            return (full_bits, 0, Vec::new());
        }
        let span = (round_half_away_f32(residual) as u64) + 1;
        let minbits = ceil_log2(if fill.is_some() { span + 1 } else { span });
        if minbits >= full_bits {
            return (full_bits, minval, Vec::new());
        }
        // `modify_1`'s tolerance, unlike the scan's: computed entirely in the
        // element type.
        let modify_tol = pow10_f32(-decimals);
        let sentinel = sentinel(minbits);
        let offsets = (0..p.nelmts)
            .map(|i| {
                let v = read_f32(i);
                match fill {
                    Some(fv) if (v - fv).abs() < modify_tol => sentinel,
                    _ => round_half_away_f32(v * pow - min_scaled) as u64,
                }
            })
            .collect();
        (minbits, minval, offsets)
    } else {
        let read_f64 =
            |i: usize| -> f64 { f64::from_bits(read_value(&input[i * 8..i * 8 + 8], 8, p.order)) };
        let fill = filval.map(f64::from_bits);
        let tol = pow10_f64(-decimals);

        let (min, max) = match fill {
            None => {
                let first = read_f64(0);
                let (mut min, mut max) = (first, first);
                for i in 1..p.nelmts {
                    let v = read_f64(i);
                    if v < min {
                        min = v;
                    }
                    if v > max {
                        max = v;
                    }
                }
                (min, max)
            }
            Some(fv) => {
                let (mut min, mut max) = (0f64, 0f64);
                let mut seen = false;
                for i in 0..p.nelmts {
                    let v = read_f64(i);
                    if (v - fv).abs() < tol {
                        continue;
                    }
                    if !seen {
                        min = v;
                        max = v;
                        seen = true;
                        continue;
                    }
                    if v < min {
                        min = v;
                    }
                    if v > max {
                        max = v;
                    }
                }
                (min, max)
            }
        };

        let pow = pow10_f64(decimals);
        let min_scaled = min * pow;
        let residual = max * pow - min_scaled;
        let minval = min.to_bits();
        if residual > (1u64 << 63) as f64 {
            return (full_bits, 0, Vec::new());
        }
        let span = (round_half_away_f64(residual) as u64) + 1;
        let minbits = ceil_log2(if fill.is_some() { span + 1 } else { span });
        if minbits >= full_bits {
            return (full_bits, minval, Vec::new());
        }
        let sentinel = sentinel(minbits);
        let offsets = (0..p.nelmts)
            .map(|i| {
                let v = read_f64(i);
                match fill {
                    Some(fv) if (v - fv).abs() < tol => sentinel,
                    _ => round_half_away_f64(v * pow - min_scaled) as u64,
                }
            })
            .collect();
        (minbits, minval, offsets)
    }
}

// --- header emit helpers --------------------------------------------------

/// Assemble a compressed chunk: 21-byte header + bit-packed `payload`.
/// The trailing safety byte built into `payload` matches the reference layout.
fn emit(minbits: u32, minval: u64, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + payload.len().max(1));
    write_header(&mut out, minbits, minval);
    if payload.is_empty() {
        // minbits == 0: the reference still reserves one trailing byte.
        out.push(0);
    } else {
        out.extend_from_slice(payload);
    }
    out
}

/// Assemble a full-precision (raw) chunk: header + verbatim element bytes.
fn emit_raw(full_bits: u32, minval: u64, input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + input.len());
    write_header(&mut out, full_bits, minval);
    out.extend_from_slice(input);
    out
}

fn write_header(out: &mut Vec<u8>, minbits: u32, minval: u64) {
    out.extend_from_slice(&minbits.to_le_bytes()); // bytes 0..4
    out.push(8); // byte 4: sizeof(minval)
    out.extend_from_slice(&minval.to_le_bytes()); // bytes 5..13
    out.extend_from_slice(&[0u8; HEADER_LEN - 13]); // bytes 13..21: padding
}

// --- bit packing ----------------------------------------------------------

/// Pack `nelmts` offsets, `minbits` bits each, MSB-first. The buffer length
/// matches the reference `nelmts * minbits / 8 + 1`: enough for the bit
/// stream plus, when the stream is a multiple of 8 bits, one trailing zero
/// the reference always reserves.
///
/// Implementation maintains a `u64` accumulator with the most recently
/// written bits at its bottom, flushing whole bytes from the top as the
/// accumulator fills. This processes a chunk of `nelmts * minbits` bits
/// with O(nelmts) iterations rather than O(nelmts * minbits) — roughly an
/// order of magnitude fewer instructions in the hot path for typical
/// minbits.
fn pack_offsets(offsets: &[u64], minbits: u32, nelmts: usize) -> Result<Vec<u8>, FormatError> {
    // `nelmts * minbits` is a bit count that sizes the payload buffer. Compute
    // it in `u64` (cannot overflow there: `nelmts` originates from a `u32` and
    // `minbits` is <= 64) and narrow the resulting byte length to `usize`,
    // erroring instead of truncating on a 32-bit host.
    let payload_len = (nelmts as u64 * minbits as u64 / 8 + 1).to_usize()?;
    if minbits == 0 {
        // All offsets are zero; the reference layout is just `payload_len`
        // zero bytes.
        return Ok(vec![0u8; payload_len]);
    }
    let mut buf = Vec::with_capacity(payload_len);
    let mask: u64 = if minbits == 64 {
        u64::MAX
    } else {
        (1u64 << minbits) - 1
    };
    let mut acc: u64 = 0;
    let mut nbits: u32 = 0;
    for &v in offsets {
        // Drain whole bytes already buffered so nbits ∈ 0..=7 going in.
        while nbits >= 8 {
            nbits -= 8;
            buf.push(((acc >> nbits) & 0xFF) as u8);
        }
        // Merge in `minbits` bits. Work in u128 because `acc << 64` would be
        // UB in u64 (we'd hit it when minbits == 64 and nbits == 0).
        let combined = ((acc as u128) << minbits) | (v & mask) as u128;
        let total = nbits + minbits;
        if total <= 64 {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "this branch is guarded by total <= 64, and combined occupies exactly `total` bits, so it fits in u64"
            )]
            {
                acc = combined as u64;
            }
            nbits = total;
        } else {
            // The merged value spans more than a u64; emit 8 bytes from the
            // top and keep the spillover (always 1..=7 bits) for next round.
            let drop = total - 64;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "combined >> drop keeps the high 64 bits of a (64 + drop)-bit value, which fits in u64"
            )]
            let top = (combined >> drop) as u64;
            buf.extend_from_slice(&top.to_be_bytes());
            #[expect(
                clippy::cast_possible_truncation,
                reason = "combined & ((1 << drop) - 1) masks to the low `drop` bits (drop is 1..=7 here), which fits in u64"
            )]
            {
                acc = (combined & ((1u128 << drop) - 1)) as u64;
            }
            nbits = drop;
        }
    }
    while nbits >= 8 {
        nbits -= 8;
        buf.push(((acc >> nbits) & 0xFF) as u8);
    }
    if nbits > 0 {
        // Left-align the remaining bits into a final byte; the low bits are
        // zero padding within the partial byte.
        buf.push(((acc << (8 - nbits)) & 0xFF) as u8);
    }
    // For bit streams that end on a byte boundary, the reference still
    // reserves one trailing zero. The formula above sized the buffer for
    // either case; pad here.
    while buf.len() < payload_len {
        buf.push(0);
    }
    Ok(buf)
}

/// Streaming MSB-first bit reader over a packed payload. Fused with the
/// reconstruction loop so we don't have to materialize a `Vec<u64>` of
/// intermediate offsets.
struct BitReader<'a> {
    payload: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(payload: &'a [u8]) -> Self {
        Self {
            payload,
            bit_pos: 0,
        }
    }

    /// Read the next `minbits` bits (1..=64) MSB-first. Caller validates
    /// that the payload holds enough bits up front.
    #[inline]
    fn read(&mut self, minbits: u32) -> u64 {
        debug_assert!((1..=64).contains(&minbits));
        let byte = self.bit_pos >> 3;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "bit_pos & 7 masks to 0..=7, which trivially fits in u32"
        )]
        let off = (self.bit_pos & 7) as u32;

        // Load a 72-bit window at `byte`. Hot path: payload has ≥ 9 bytes
        // remaining and we can read directly. Tail: zero-pad a stack buffer.
        let (hi, lo) = if byte + 9 <= self.payload.len() {
            let hi = u64::from_be_bytes(self.payload[byte..byte + 8].try_into().unwrap());
            let lo = self.payload[byte + 8] as u64;
            (hi, lo)
        } else {
            let mut window = [0u8; 9];
            let take = self.payload.len().saturating_sub(byte).min(9);
            window[..take].copy_from_slice(&self.payload[byte..byte + take]);
            let hi = u64::from_be_bytes(window[..8].try_into().unwrap());
            let lo = window[8] as u64;
            (hi, lo)
        };

        // Align the window so the desired `minbits` bits sit at the top of
        // a 64-bit value, then shift them down. The off == 0 branch avoids
        // the otherwise-suspicious `lo >> 8` on a value with only 8 set bits.
        let combined = if off == 0 {
            hi
        } else {
            (hi << off) | (lo >> (8 - off))
        };
        self.bit_pos += minbits as usize;
        combined >> (64 - minbits)
    }
}

/// Test-only helper: collect a full bitstream into a `Vec<u64>`. The main
/// decompress path drives the reconstructor directly off [`BitReader`].
#[cfg(test)]
fn unpack_bits(payload: &[u8], nelmts: usize, minbits: u32) -> Result<Vec<u64>, FormatError> {
    let total_bits = nelmts * minbits as usize;
    if payload.len() * 8 < total_bits {
        return Err(FormatError::FilterError(
            "scaleoffset: payload too short for packed data".to_string(),
        ));
    }
    let mut reader = BitReader::new(payload);
    Ok((0..nelmts).map(|_| reader.read(minbits)).collect())
}

// --- value (de)serialization ----------------------------------------------

/// Read a `size`-byte element as a u64 (low `size` bytes meaningful),
/// normalizing to little-endian regardless of stored `order`.
fn read_value(chunk: &[u8], size: usize, order: u32) -> u64 {
    let mut bytes = [0u8; 8];
    if order == ORDER_LE {
        bytes[..size].copy_from_slice(&chunk[..size]);
    } else {
        for (k, &b) in chunk[..size].iter().enumerate() {
            bytes[size - 1 - k] = b;
        }
    }
    u64::from_le_bytes(bytes)
}

/// Write the low `size` bytes of `bits` in the dataset's byte order.
fn write_value(out: &mut Vec<u8>, bits: u64, size: usize, order: u32) {
    let le = bits.to_le_bytes();
    if order == ORDER_LE {
        out.extend_from_slice(&le[..size]);
    } else {
        for k in (0..size).rev() {
            out.push(le[k]);
        }
    }
}

#[expect(
    clippy::cast_possible_wrap,
    reason = "size >= 8: reinterpreting all 64 stored bits as i64 is the intentional sign reinterpretation of a full-width value; size < 8: ((bits << shift) as i64) >> shift sign-extends the `size`-byte value, an intentional wrap"
)]
fn sign_extend(bits: u64, size: usize) -> i64 {
    if size >= 8 {
        bits as i64
    } else {
        let shift = 64 - size * 8;
        ((bits << shift) as i64) >> shift
    }
}

/// `1 << minbits - 1`, the all-ones offset that flags a fill value.
fn sentinel(minbits: u32) -> u64 {
    (1u64 << minbits).wrapping_sub(1)
}

/// Store a `size`-byte fill value in `cd_values[8..]`, least-significant 4
/// bytes per entry: the inverse of [`read_fill_bits`], and the layout
/// `H5Z__scaleoffset_set_parms_fillval` writes.
///
/// `size` is at most 8 and `cd` is always [`TOTAL_NPARMS`] long, so the two
/// entries this can reach are always present.
fn write_fill_bits(cd: &mut [u32], bits: u64, size: usize) {
    let le = bits.to_le_bytes();
    let mut off = 0;
    let mut idx = PARM_FILVAL;
    while off < size {
        let take = (size - off).min(4);
        let mut entry = [0u8; 4];
        entry[..take].copy_from_slice(&le[off..off + take]);
        cd[idx] = u32::from_le_bytes(entry);
        off += take;
        idx += 1;
    }
}

/// Reassemble a `size`-byte fill value from `cd_values[8..]` (stored
/// least-significant 4 bytes per entry).
fn read_fill_bits(cd: &[u32], size: usize) -> Result<u64, FormatError> {
    let entries = size.div_ceil(4);
    if cd.len() < PARM_FILVAL + entries {
        return Err(FormatError::FilterError(
            "scaleoffset: cd_values missing fill value".to_string(),
        ));
    }
    let mut bytes = [0u8; 8];
    let mut off = 0;
    let mut idx = PARM_FILVAL;
    while off < size {
        let take = (size - off).min(4);
        bytes[off..off + take].copy_from_slice(&cd[idx].to_le_bytes()[..take]);
        off += take;
        idx += 1;
    }
    Ok(u64::from_le_bytes(bytes))
}

/// `10^exp` as `f64`, computed without `std` (the float `powf`/`powi` methods
/// require `std`). `exp` is a small decimal scale factor; exponentiation by
/// squaring keeps this both cheap and accurate.
fn pow10_f64(exp: i32) -> f64 {
    let mut result = 1.0f64;
    let mut base = 10.0f64;
    let mut n = exp.unsigned_abs();
    while n > 0 {
        if n & 1 == 1 {
            result *= base;
        }
        base *= base;
        n >>= 1;
    }
    if exp < 0 { 1.0 / result } else { result }
}

/// `10^exp` as `f32` (computed in `f64` for accuracy, then narrowed).
///
/// Narrowing an `f64` result rounds twice where the reference's `powf` rounds
/// once, so the two can land one ulp apart — measured, at `exp = -22` and
/// nowhere else in `-45..=38`. This form is the more accurate one, and `powf`
/// is platform libm rather than a correctly-rounded function, so agreeing with
/// it bit-for-bit is not something a portable implementation can promise.
#[expect(
    clippy::cast_possible_truncation,
    reason = "narrowing 10^exp from f64 to f32 is the intended value conversion; out-of-range magnitudes saturate to +/-inf, matching the C reference"
)]
fn pow10_f32(exp: i32) -> f32 {
    pow10_f64(exp) as f32
}

/// Round half away from zero to the nearest integer, matching C `llround`.
/// Float-to-int `as` casts saturate in Rust, so out-of-range inputs are safe.
///
/// Rounding the *sum* `x + 0.5` is not the same function as rounding `x`, and
/// the difference is observable: at `x = 0.49999999999999994` — the largest
/// double below one half — the exact sum `1 - 2^-54` sits halfway between two
/// doubles, so it rounds to `1.0` and the cast yields 1 where `llround(x)`
/// yields 0. The fractional part is therefore compared against one half
/// directly. `x - trunc(x)` is exact for every finite `x`, which is what makes
/// that comparison exact rather than merely closer.
///
/// Every double of magnitude `2^52` or more is already an integer, so above that
/// there is nothing to round and the cast — which saturates where C's `llround`
/// is undefined — answers on its own. That also catches NaN, which fails both
/// comparisons.
#[expect(
    clippy::cast_possible_truncation,
    reason = "float-to-int `as` saturates in Rust, so rounding to i64 is safe even for out-of-range inputs (matches C llround)"
)]
fn round_half_away_f64(x: f64) -> i64 {
    /// The first `f64` magnitude at which consecutive doubles are 1 apart.
    const INTEGRAL: f64 = (1u64 << 52) as f64;
    if !(x > -INTEGRAL && x < INTEGRAL) {
        return x as i64;
    }
    // Exact: |x| < 2^52 so the truncation round-trips through i64, and the
    // subtraction of two nearby doubles is representable.
    let trunc = x as i64;
    let frac = x - trunc as f64;
    if frac >= 0.5 {
        trunc + 1
    } else if frac <= -0.5 {
        trunc - 1
    } else {
        trunc
    }
}

/// `f32` counterpart of [`round_half_away_f64`] (matches C `lroundf`), rounding
/// `x` itself for the same reason — the `f32` sum rounds to `1.0` at
/// `x = 0.499_999_97`, the largest float below one half.
#[expect(
    clippy::cast_possible_truncation,
    reason = "float-to-int `as` saturates in Rust, so rounding to i64 is safe even for out-of-range inputs (matches C lroundf)"
)]
fn round_half_away_f32(x: f32) -> i64 {
    /// The first `f32` magnitude at which consecutive floats are 1 apart.
    const INTEGRAL: f32 = (1u32 << 23) as f32;
    if !(x > -INTEGRAL && x < INTEGRAL) {
        return x as i64;
    }
    let trunc = x as i64;
    let frac = x - trunc as f32;
    if frac >= 0.5 {
        trunc + 1
    } else if frac <= -0.5 {
        trunc - 1
    } else {
        trunc
    }
}

/// Ceiling of log2, matching the reference `H5Z__scaleoffset_log2`
/// (`log2(0) == 1`, `log2(1) == 0`).
fn ceil_log2(num: u64) -> u32 {
    let mut v = 0u32;
    let mut lower_bound = 1u64;
    let mut val = num;
    loop {
        val >>= 1;
        if val == 0 {
            break;
        }
        v += 1;
        lower_bound <<= 1;
    }
    if num == lower_bound { v } else { v + 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_filter(size: u32, signed: bool, order: u32, nelmts: u32) -> FilterDescription {
        let ty = ScaleOffsetType {
            class: CLS_INTEGER,
            sign: if signed { SGN_2 } else { SGN_NONE },
            order,
        };
        FilterDescription {
            filter_id: crate::filter_pipeline::FILTER_SCALEOFFSET,
            name: None,
            flags: 0,
            client_data: build_cd_values(
                ScaleOffset::Integer(0),
                ty,
                size,
                nelmts,
                ScaleOffsetFill::Undefined,
            )
            .unwrap(),
        }
    }

    fn float_filter(size: u32, decimals: i32, order: u32, nelmts: u32) -> FilterDescription {
        let ty = ScaleOffsetType {
            class: CLS_FLOAT,
            sign: SGN_NONE,
            order,
        };
        FilterDescription {
            filter_id: crate::filter_pipeline::FILTER_SCALEOFFSET,
            name: None,
            flags: 0,
            client_data: build_cd_values(
                ScaleOffset::FloatDScale(decimals),
                ty,
                size,
                nelmts,
                ScaleOffsetFill::Undefined,
            )
            .unwrap(),
        }
    }

    #[test]
    fn scale_offset_mode_recovers_and_refuses() {
        // Lossless integer is recovered (the only mode repack will re-emit).
        let f = int_filter(4, true, ORDER_LE, 16);
        assert_eq!(
            scale_offset_mode(&f.client_data),
            Some((ScaleOffset::Integer(0), FillAvailability::Undefined))
        );

        // Lossy float D-scale is recognized as such; repack refuses it upstream.
        let f = float_filter(8, 3, ORDER_LE, 16);
        assert_eq!(
            scale_offset_mode(&f.client_data),
            Some((ScaleOffset::FloatDScale(3), FillAvailability::Undefined))
        );

        // A defined fill value is recovered as one, so repack re-applies the
        // availability the source recorded rather than refusing the dataset.
        let mut fill_defined = int_filter(4, true, ORDER_LE, 16);
        fill_defined.client_data[PARM_FILAVAIL] = FILL_DEFINED;
        assert_eq!(
            scale_offset_mode(&fill_defined.client_data),
            Some((ScaleOffset::Integer(0), FillAvailability::Defined))
        );

        // Neither availability: a value the reference never writes, so what it
        // would mean is undefined and re-applying it would be a guess.
        let mut bad = int_filter(4, true, ORDER_LE, 16);
        bad.client_data[PARM_FILAVAIL] = 2;
        assert_eq!(scale_offset_mode(&bad.client_data), None);

        // Too-short parameter arrays -> None, never a panic.
        assert_eq!(scale_offset_mode(&[]), None);
        assert_eq!(scale_offset_mode(&[SO_INT, 0]), None);
    }

    /// `signed char` is the one type `H5Z__scaleoffset_precompress_i` hand-
    /// expands rather than routing through the `H5Z_scaleoffset_check_2` macro,
    /// and the expansion assigns `*minval` in its fill-**undefined** early
    /// return where the macro assigns nothing. So a 1-byte signed chunk that
    /// falls back to full precision carries the chunk's minimum in its header,
    /// and every other width, signedness, and fill mode carries zero.
    ///
    /// Measured against the C library when this encoder was written. It stays a
    /// unit test rather than a crosscheck because nothing this crate *writes*
    /// records an undefined fill value any longer — the branch is reached by
    /// re-encoding a chunk of a file that already carries one, which is what an
    /// append to a dataset written before 0.39, or a repack of one, does.
    ///
    /// Every fixture's minimum is deliberately non-zero: a chunk whose minimum
    /// is zero writes the same header under either rule.
    #[test]
    fn the_fill_undefined_fallback_carries_the_signed_char_minimum() {
        let minval_of = |raw: &[u8], size: u32, signed: bool| {
            let nelmts = raw.len() as u32 / size;
            let f = int_filter(size, signed, ORDER_LE, nelmts);
            let out = compress(raw, &f).unwrap();
            // The fixtures all spread past the point where packing can pay, so
            // each must have taken the full-precision fallback.
            assert_eq!(
                u32::from_le_bytes(out[..4].try_into().unwrap()),
                size * 8,
                "the fixture must reach the full-precision fallback"
            );
            u64::from_le_bytes(out[5..13].try_into().unwrap())
        };

        let i8_raw: Vec<u8> = [-128i8, 126, -128, 126, -1, -2, -3, -4]
            .iter()
            .map(|&v| v as u8)
            .collect();
        assert_eq!(minval_of(&i8_raw, 1, true), (-128i64) as u64);

        // One byte wide but unsigned: the macro-driven path, so zero. Its own
        // fixture, because the signed one read as unsigned spans only 126..255
        // — inside the threshold, so it would reach the *other* full-precision
        // return, the one that does carry the minimum in every mode.
        let u8_raw: Vec<u8> = vec![1, 255, 1, 255, 2, 3, 4, 5];
        assert_eq!(minval_of(&u8_raw, 1, false), 0);

        let i16_raw: Vec<u8> = [-32768i16, 32766, -32768, 32766, -1, -2, -3, -4]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            minval_of(&i16_raw, 2, true),
            0,
            "signed, but two bytes wide"
        );

        // And the fill-defined half of the same 1-byte signed chunk, which the
        // crosscheck pins against the C library: the hand-expanded branch
        // assigns nothing there either.
        let mut defined = int_filter(1, true, ORDER_LE, i8_raw.len() as u32);
        defined.client_data[PARM_FILAVAIL] = FILL_DEFINED;
        let out = compress(&i8_raw, &defined).unwrap();
        assert_eq!(u64::from_le_bytes(out[5..13].try_into().unwrap()), 0);
    }

    /// The fill value's round trip through `cd_values`: written the way
    /// `H5Z__scaleoffset_set_parms_fillval` writes it and read back by the same
    /// [`read_fill_bits`] the decoder uses, at every width and in both byte
    /// orders. A `u64` fill spans two entries, which is the case a single-entry
    /// writer would silently truncate.
    #[test]
    fn a_fill_value_round_trips_through_the_filter_parameters() {
        for (size, bits) in [
            (1u32, 0xA5u64),
            (2, 0xBEEF),
            (4, 0xDEAD_BEEF),
            (8, 0x0123_4567_89AB_CDEF),
        ] {
            for order in [ORDER_LE, ORDER_BE] {
                let ty = ScaleOffsetType {
                    class: CLS_INTEGER,
                    sign: SGN_NONE,
                    order,
                };
                // The fill value as the dataset stores it: `size` bytes in the
                // dataset's own byte order.
                let mut bytes = Vec::new();
                write_value(&mut bytes, bits, size as usize, order);
                let cd = build_cd_values(
                    ScaleOffset::Integer(0),
                    ty,
                    size,
                    4,
                    ScaleOffsetFill::Defined(Some(&bytes)),
                )
                .unwrap();

                assert_eq!(cd[PARM_FILAVAIL], FILL_DEFINED);
                assert_eq!(
                    read_fill_bits(&cd, size as usize).unwrap(),
                    bits,
                    "size {size}, order {order}"
                );
                // Entries past the value stay zero, as the reference leaves them.
                let used = (size as usize).div_ceil(4);
                assert!(cd[PARM_FILVAL + used..].iter().all(|&e| e == 0));
            }
        }

        // The library default is a defined fill value of zero, not an absent one.
        let ty = ScaleOffsetType {
            class: CLS_INTEGER,
            sign: SGN_NONE,
            order: ORDER_LE,
        };
        let cd = build_cd_values(
            ScaleOffset::Integer(0),
            ty,
            4,
            4,
            ScaleOffsetFill::Defined(None),
        )
        .unwrap();
        assert_eq!(cd[PARM_FILAVAIL], FILL_DEFINED);
        assert_eq!(read_fill_bits(&cd, 4).unwrap(), 0);

        // A datatype wider than the filter's own limit is refused rather than
        // running the fill value off the end of the parameter array.
        for width in [0u32, 9, 16] {
            assert!(
                build_cd_values(
                    ScaleOffset::Integer(0),
                    ty,
                    width,
                    4,
                    ScaleOffsetFill::Defined(Some(&vec![0u8; width as usize]))
                )
                .is_err(),
                "a {width}-byte datatype"
            );
        }

        // A fill value that is not one element wide is refused rather than
        // truncated or padded into a different value.
        for bad in [vec![1u8], vec![1u8; 8]] {
            assert!(
                build_cd_values(
                    ScaleOffset::Integer(0),
                    ty,
                    4,
                    4,
                    ScaleOffsetFill::Defined(Some(&bad))
                )
                .is_err(),
                "a {}-byte fill value for a 4-byte datatype",
                bad.len()
            );
        }
    }

    /// The rounding these helpers do is C `llround`/`lroundf` — round `x` half
    /// away from zero — and not "round `x + 0.5`", which is a different
    /// function. The two disagree at exactly one input per sign and precision:
    /// the largest float below one half, where the sum lands *on* the half and
    /// rounds up.
    ///
    /// That input reaches the encoder as `value * 10^D - min_scaled`, so it is
    /// data-dependent rather than exotic, and it changes a decoded value rather
    /// than only the bytes it is stored as.
    #[test]
    fn rounding_is_llround_and_not_the_rounded_sum() {
        // The half-way cases themselves round away from zero, in both signs and
        // at every magnitude where a half is representable.
        for (x, want) in [
            (0.5f64, 1i64),
            (-0.5, -1),
            (1.5, 2),
            (-1.5, -2),
            (2.5, 3),
            (-2.5, -3),
            (0.0, 0),
            (0.4, 0),
            (-0.4, 0),
            (7.0, 7),
        ] {
            assert_eq!(round_half_away_f64(x), want, "f64 round({x})");
        }

        // The largest double below one half: the sum `x + 0.5` rounds up to
        // exactly 1.0, where rounding `x` itself gives 0.
        let below_half = 0.499_999_999_999_999_94f64;
        assert!(below_half < 0.5 && below_half + 0.5 == 1.0);
        assert_eq!(round_half_away_f64(below_half), 0);
        assert_eq!(round_half_away_f64(-below_half), 0);

        // The `f32` counterpart, where the sum rounds to exactly 1.0.
        let below_half_f32 = 0.499_999_97f32;
        assert!(below_half_f32 < 0.5 && below_half_f32 + 0.5 == 1.0);
        assert_eq!(round_half_away_f32(below_half_f32), 0);
        assert_eq!(round_half_away_f32(-below_half_f32), 0);
        assert_eq!(round_half_away_f32(0.5), 1);
        assert_eq!(round_half_away_f32(-2.5), -3);

        // Past the point where consecutive floats are more than 1 apart every
        // value is already an integer, and beyond the integer range the cast
        // saturates rather than wrapping (C leaves it undefined).
        assert_eq!(round_half_away_f64(1e300), i64::MAX);
        assert_eq!(round_half_away_f64(-1e300), i64::MIN);
        assert_eq!(round_half_away_f32(1e30), i64::MAX);
        assert_eq!(round_half_away_f64(f64::NAN), 0);
        assert_eq!(round_half_away_f32(f32::NAN), 0);
        // Either side of the constant that separates the two branches.
        let integral = (1u64 << 52) as f64;
        assert_eq!(round_half_away_f64(integral), 1 << 52);
        assert_eq!(round_half_away_f64(integral - 1.0), (1 << 52) - 1);
        assert_eq!(round_half_away_f64(-integral), -(1 << 52));
    }

    #[test]
    fn ceil_log2_matches_reference() {
        assert_eq!(ceil_log2(0), 1);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(255), 8);
        assert_eq!(ceil_log2(256), 8);
        assert_eq!(ceil_log2(257), 9);
    }

    fn roundtrip_u32(vals: &[u32], order: u32) {
        let mut raw = Vec::new();
        for &v in vals {
            if order == ORDER_LE {
                raw.extend_from_slice(&v.to_le_bytes());
            } else {
                raw.extend_from_slice(&v.to_be_bytes());
            }
        }
        let f = int_filter(4, false, order, vals.len() as u32);
        let comp = compress(&raw, &f).unwrap();
        let dec = decompress(&comp, &f, None).unwrap();
        assert_eq!(dec, raw);
    }

    #[test]
    fn integer_unsigned_roundtrip_le_and_be() {
        let vals = [100u32, 105, 101, 110, 100, 128];
        roundtrip_u32(&vals, ORDER_LE);
        roundtrip_u32(&vals, ORDER_BE);
    }

    #[test]
    fn integer_signed_roundtrip_with_negatives() {
        let vals: [i16; 6] = [-100, -50, -100, 0, 27, -99];
        for &order in &[ORDER_LE, ORDER_BE] {
            let mut raw = Vec::new();
            for &v in &vals {
                if order == ORDER_LE {
                    raw.extend_from_slice(&v.to_le_bytes());
                } else {
                    raw.extend_from_slice(&v.to_be_bytes());
                }
            }
            let f = int_filter(2, true, order, vals.len() as u32);
            let comp = compress(&raw, &f).unwrap();
            let dec = decompress(&comp, &f, None).unwrap();
            assert_eq!(dec, raw, "order {order}");
        }
    }

    #[test]
    fn integer_all_equal_uses_minbits_zero() {
        let vals = [7u32; 5];
        let mut raw = Vec::new();
        for &v in &vals {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let f = int_filter(4, false, ORDER_LE, vals.len() as u32);
        let comp = compress(&raw, &f).unwrap();
        // minbits == 0 -> 21-byte header + 1 trailing byte.
        assert_eq!(comp.len(), HEADER_LEN + 1);
        assert_eq!(u32::from_le_bytes([comp[0], comp[1], comp[2], comp[3]]), 0);
        let dec = decompress(&comp, &f, None).unwrap();
        assert_eq!(dec, raw);
    }

    #[test]
    fn integer_full_range_uses_raw_path() {
        // 0 and u32::MAX force the full-precision (raw) path.
        let vals = [0u32, u32::MAX, 123];
        let mut raw = Vec::new();
        for &v in &vals {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let f = int_filter(4, false, ORDER_LE, vals.len() as u32);
        let comp = compress(&raw, &f).unwrap();
        assert_eq!(comp.len(), HEADER_LEN + raw.len());
        assert_eq!(u32::from_le_bytes([comp[0], comp[1], comp[2], comp[3]]), 32);
        let dec = decompress(&comp, &f, None).unwrap();
        assert_eq!(dec, raw);
    }

    #[test]
    fn integer_u8_roundtrip() {
        let raw = vec![10u8, 11, 12, 250, 10, 200];
        let f = int_filter(1, false, ORDER_LE, raw.len() as u32);
        let comp = compress(&raw, &f).unwrap();
        let dec = decompress(&comp, &f, None).unwrap();
        assert_eq!(dec, raw);
    }

    #[test]
    fn integer_i64_roundtrip() {
        let vals: [i64; 4] = [-1_000_000, 5, -999_999, 42];
        let mut raw = Vec::new();
        for &v in &vals {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let f = int_filter(8, true, ORDER_LE, vals.len() as u32);
        let comp = compress(&raw, &f).unwrap();
        let dec = decompress(&comp, &f, None).unwrap();
        assert_eq!(dec, raw);
    }

    #[test]
    fn float_dscale_roundtrip_within_tolerance() {
        let vals = [1.234f64, 1.235, 1.250, 1.111, 1.234, 1.999];
        let decimals = 3;
        let mut raw = Vec::new();
        for &v in &vals {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let f = float_filter(8, decimals, ORDER_LE, vals.len() as u32);
        let comp = compress(&raw, &f).unwrap();
        let dec = decompress(&comp, &f, None).unwrap();
        let got: Vec<f64> = dec
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect();
        let tol = 0.5 * 10f64.powi(-decimals);
        for (g, v) in got.iter().zip(vals.iter()) {
            assert!((g - v).abs() <= tol, "got {g}, want {v}");
        }
    }

    #[test]
    fn float32_dscale_roundtrip_be() {
        let vals = [10.25f32, 10.50, 10.75, 10.00, 10.25];
        let decimals = 2;
        let mut raw = Vec::new();
        for &v in &vals {
            raw.extend_from_slice(&v.to_be_bytes());
        }
        let f = float_filter(4, decimals, ORDER_BE, vals.len() as u32);
        let comp = compress(&raw, &f).unwrap();
        let dec = decompress(&comp, &f, None).unwrap();
        let got: Vec<f32> = dec
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_be_bytes(*c))
            .collect();
        let tol = 0.5 * 10f32.powi(-decimals);
        for (g, v) in got.iter().zip(vals.iter()) {
            assert!((g - v).abs() <= tol, "got {g}, want {v}");
        }
    }

    #[test]
    fn truncated_chunk_errors_not_panics() {
        // A chunk whose header claims minbits=3 but is shorter than the 21-byte
        // header region must error rather than panic when slicing the payload.
        let f = int_filter(4, false, ORDER_LE, 4);
        let mut bad = Vec::new();
        bad.extend_from_slice(&3u32.to_le_bytes()); // minbits = 3
        bad.push(8); // minval size
        bad.extend_from_slice(&0u64.to_le_bytes()); // minval (bytes 5..13)
        // Only 13 bytes total: shorter than the 21-byte header.
        assert!(matches!(
            decompress(&bad, &f, None),
            Err(FormatError::FilterError(_))
        ));
    }

    #[test]
    fn header_byte_layout() {
        // Two u32 values {min=5, max=9} -> span 5 -> minbits 3.
        let vals = [5u32, 9, 6, 5];
        let mut raw = Vec::new();
        for &v in &vals {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let f = int_filter(4, false, ORDER_LE, vals.len() as u32);
        let comp = compress(&raw, &f).unwrap();
        assert_eq!(u32::from_le_bytes([comp[0], comp[1], comp[2], comp[3]]), 3);
        assert_eq!(comp[4], 8); // sizeof(minval)
        let minval = u64::from_le_bytes(comp[5..13].try_into().unwrap());
        assert_eq!(minval, 5);
        assert_eq!(&comp[13..21], &[0u8; 8]); // padding
    }

    /// A chunk that claims more decoded bytes than the chunk can hold is
    /// refused before the allocation, not after it (#233).
    ///
    /// Scale-offset is the one filter here that cannot bound itself by the
    /// length of its input: `minbits == 0` means every value equals the
    /// minimum, so a thirteen-byte header expands into the whole chunk with no
    /// payload behind it. The size it claims comes from its own cd_values,
    /// which nothing has yet tied to the chunk geometry, so the bound has to be
    /// the size the pipeline expects of this stage.
    #[test]
    fn decompress_refuses_a_claim_larger_than_the_chunk() {
        // Thirteen bytes claiming two gigabytes: 2^28 elements of 8 bytes.
        let mut chunk = vec![0u8; HEADER_LEN];
        chunk[4] = 8; // sizeof(minval); minbits stays 0

        let huge = int_filter(8, true, ORDER_LE, 1 << 28);
        let err = decompress(&chunk, &huge, Some(4096)).unwrap_err();
        let FormatError::FilterError(msg) = &err else {
            panic!("expected a filter error, got {err}");
        };
        assert!(msg.contains("scaleoffset"), "{msg}");

        // The guard is the cap, not a new refusal of chunks in general: a claim
        // the chunk can hold still decodes, and to exactly that many bytes.
        let fits = int_filter(8, true, ORDER_LE, 512);
        assert_eq!(decompress(&chunk, &fits, Some(4096)).unwrap().len(), 4096);
        // And with no cap to check against, behavior is what it always was.
        assert_eq!(decompress(&chunk, &fits, None).unwrap().len(), 4096);
    }

    #[test]
    fn build_cd_values_rejects_mismatched_mode() {
        let int_ty = ScaleOffsetType {
            class: CLS_INTEGER,
            sign: SGN_2,
            order: ORDER_LE,
        };
        assert!(
            build_cd_values(
                ScaleOffset::FloatDScale(2),
                int_ty,
                4,
                10,
                ScaleOffsetFill::Undefined
            )
            .is_err()
        );
        let float_ty = ScaleOffsetType {
            class: CLS_FLOAT,
            sign: SGN_NONE,
            order: ORDER_LE,
        };
        assert!(
            build_cd_values(
                ScaleOffset::Integer(0),
                float_ty,
                8,
                10,
                ScaleOffsetFill::Undefined
            )
            .is_err()
        );
    }

    /// Deterministic xorshift64 used to drive the property-style tests below.
    /// Keeps the dev-dependency footprint minimal while still covering a wide
    /// input space; seeds are fixed so failures are reproducible.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn range(&mut self, hi: u64) -> u64 {
            // Modulo bias is fine for our coverage purposes.
            self.next() % hi
        }
    }

    #[test]
    fn pack_unpack_equivalence_random() {
        // For every minbits in 1..=64 and across many random widths/lengths,
        // pack(offsets) followed by unpack must return the original offsets.
        let mut rng = Rng::new(0x00C0_FFEE_F00D_1234);
        for _ in 0..400 {
            let minbits = (rng.range(64) + 1) as u32; // 1..=64
            let nelmts = (rng.range(257) + 1) as usize; // 1..=257
            let mask = if minbits == 64 {
                u64::MAX
            } else {
                (1u64 << minbits) - 1
            };
            let offsets: Vec<u64> = (0..nelmts).map(|_| rng.next() & mask).collect();
            let packed = pack_offsets(&offsets, minbits, nelmts).unwrap();
            // Reference layout reserves at least one trailing byte.
            assert_eq!(packed.len(), nelmts * minbits as usize / 8 + 1);
            let unpacked = unpack_bits(&packed, nelmts, minbits).unwrap();
            assert_eq!(
                unpacked, offsets,
                "minbits={minbits}, nelmts={nelmts}, seed=0xC0FFEEF00D1234"
            );
        }
    }

    fn roundtrip_random<T: Copy>(
        seed: u64,
        size: u32,
        signed: bool,
        order: u32,
        encode: impl Fn(&mut Rng) -> T,
        to_bytes: impl Fn(T, u32) -> Vec<u8>,
    ) {
        let mut rng = Rng::new(seed);
        for trial in 0..40 {
            let nelmts = (rng.range(199) + 1) as usize;
            let mut raw = Vec::with_capacity(nelmts * size as usize);
            for _ in 0..nelmts {
                raw.extend_from_slice(&to_bytes(encode(&mut rng), order));
            }
            let f = int_filter(size, signed, order, nelmts as u32);
            let comp = compress(&raw, &f).unwrap();
            let dec = decompress(&comp, &f, None).unwrap();
            assert_eq!(
                dec, raw,
                "trial {trial}: size={size}, signed={signed}, order={order}"
            );
        }
    }

    #[test]
    fn integer_roundtrip_random_u8() {
        roundtrip_random::<u8>(0x11, 1, false, ORDER_LE, |r| r.next() as u8, |v, _| vec![v]);
    }

    #[test]
    fn integer_roundtrip_random_u16_le_and_be() {
        for &order in &[ORDER_LE, ORDER_BE] {
            roundtrip_random::<u16>(
                0x22 ^ order as u64,
                2,
                false,
                order,
                |r| r.next() as u16,
                |v, o| {
                    if o == ORDER_LE {
                        v.to_le_bytes().to_vec()
                    } else {
                        v.to_be_bytes().to_vec()
                    }
                },
            );
        }
    }

    #[test]
    fn integer_roundtrip_random_i32_narrow_and_wide() {
        // Mix narrow (small span -> small minbits) and wide (forces raw path).
        let mut rng = Rng::new(0x33);
        for trial in 0..40 {
            let nelmts = (rng.range(199) + 1) as usize;
            let narrow = rng.next() & 1 == 0;
            let mut raw = Vec::with_capacity(nelmts * 4);
            for _ in 0..nelmts {
                let v: i32 = if narrow {
                    (rng.range(1024) as i32) - 512 // span ~= 1024
                } else {
                    rng.next() as i32 // full range
                };
                raw.extend_from_slice(&v.to_le_bytes());
            }
            let f = int_filter(4, true, ORDER_LE, nelmts as u32);
            let comp = compress(&raw, &f).unwrap();
            let dec = decompress(&comp, &f, None).unwrap();
            assert_eq!(dec, raw, "trial {trial}: narrow={narrow}");
        }
    }

    #[test]
    fn integer_roundtrip_random_i64() {
        roundtrip_random::<i64>(
            0x44,
            8,
            true,
            ORDER_LE,
            |r| {
                // Bias toward narrow spans so we exercise the packing path,
                // but include the full-range tail too.
                let span = 1u64 << (r.range(48) as u32 + 1);
                let v = (r.next() % span) as i64;
                let off = (r.next() as i64) >> 1;
                v.wrapping_add(off)
            },
            |v, _| v.to_le_bytes().to_vec(),
        );
    }

    #[test]
    fn float_dscale_roundtrip_random_within_tolerance() {
        let mut rng = Rng::new(0x55);
        for trial in 0..30 {
            let nelmts = (rng.range(99) + 1) as usize;
            let decimals = (rng.range(5) as i32) + 1; // 1..=5
            // Keep magnitudes modest so f64 ULP stays well below the D-scale
            // tolerance. Avoid generating values pre-quantized to the same
            // precision (decimals) we then round to, otherwise the input
            // sits exactly on a rounding boundary.
            let mut raw = Vec::with_capacity(nelmts * 8);
            let mut vals = Vec::with_capacity(nelmts);
            for _ in 0..nelmts {
                let v = (rng.next() as i32 as f64) * 1e-9; // |v| <= ~2.1
                vals.push(v);
                raw.extend_from_slice(&v.to_le_bytes());
            }
            let f = float_filter(8, decimals, ORDER_LE, nelmts as u32);
            let comp = compress(&raw, &f).unwrap();
            let dec = decompress(&comp, &f, None).unwrap();
            let got: Vec<f64> = dec
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c))
                .collect();
            // 0.5 ULP rounding + a few ULP of float arithmetic noise.
            let tol = 0.501 * 10f64.powi(-decimals);
            for (g, v) in got.iter().zip(vals.iter()) {
                assert!(
                    (g - v).abs() <= tol,
                    "trial {trial}: got {g}, want {v} (tol {tol})"
                );
            }
        }
    }

    /// Build a synthetic chunk where `minbits == 0`, the compressed header
    /// declares `minval = 7`, but the filter's `cd_values` say
    /// `filavail = FILL_DEFINED` with `filval = 99`.
    ///
    /// This test previously asserted the opposite, on the stated grounds that
    /// the reference short-circuits `minbits == 0` to emit `minval`. It does
    /// not short-circuit at all: it zeroes the buffer and runs its
    /// postdecompress anyway, where the sentinel `(1 << 0) - 1` is zero and
    /// matches every element. Measured — libhdf5 reads this exact chunk as
    /// `[99; 5]` — rather than re-derived from the source a second time.
    ///
    /// No encoder emits this pair, so it is reachable only from a damaged or
    /// hand-built chunk; the point is that a damaged chunk reads the same here
    /// as it does through the C library.
    #[test]
    fn minbits_zero_with_fill_defined_reads_as_the_fill_value() {
        let nelmts = 5u32;
        let size = 4u32;
        let mut cd = vec![0u32; TOTAL_NPARMS];
        cd[PARM_SCALETYPE] = SO_INT;
        cd[PARM_SCALEFACTOR] = 0;
        cd[PARM_NELMTS] = nelmts;
        cd[PARM_CLASS] = CLS_INTEGER;
        cd[PARM_SIZE] = size;
        cd[PARM_SIGN] = SGN_NONE;
        cd[PARM_ORDER] = ORDER_LE;
        cd[PARM_FILAVAIL] = FILL_DEFINED;
        cd[PARM_FILVAL] = 99;
        let f = FilterDescription {
            filter_id: crate::filter_pipeline::FILTER_SCALEOFFSET,
            name: None,
            flags: 0,
            client_data: cd,
        };
        // 21-byte header (minbits=0, minval=7) + 1-byte trailing pad.
        let mut chunk = Vec::with_capacity(HEADER_LEN + 1);
        chunk.extend_from_slice(&0u32.to_le_bytes());
        chunk.push(8);
        chunk.extend_from_slice(&7u64.to_le_bytes());
        chunk.extend_from_slice(&[0u8; HEADER_LEN - 13]);
        chunk.push(0);
        let out = decompress(&chunk, &f, None).unwrap();
        let got: Vec<u32> = out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        assert_eq!(
            got,
            vec![99u32; nelmts as usize],
            "minbits==0 with FILL_DEFINED reads as the fill value, not minval"
        );
    }

    /// A chunk with `minbits > 0` and `FILL_DEFINED`: the offsets at the
    /// all-ones sentinel must reconstruct to `filval`, while every other
    /// offset reconstructs to `minval + offset`. Companion to the
    /// `minbits == 0` regression above — together they cover both sides of
    /// the sentinel branch.
    #[test]
    fn fill_defined_sentinel_emits_filval_not_minval() {
        let nelmts = 4u32;
        let size = 4u32;
        let mut cd = vec![0u32; TOTAL_NPARMS];
        cd[PARM_SCALETYPE] = SO_INT;
        cd[PARM_SCALEFACTOR] = 0;
        cd[PARM_NELMTS] = nelmts;
        cd[PARM_CLASS] = CLS_INTEGER;
        cd[PARM_SIZE] = size;
        cd[PARM_SIGN] = SGN_NONE;
        cd[PARM_ORDER] = ORDER_LE;
        cd[PARM_FILAVAIL] = FILL_DEFINED;
        cd[PARM_FILVAL] = 999;
        let f = FilterDescription {
            filter_id: crate::filter_pipeline::FILTER_SCALEOFFSET,
            name: None,
            flags: 0,
            client_data: cd,
        };
        // minbits = 3 → sentinel = 7. Offsets [0, 1, 7, 2] expect
        // [minval+0, minval+1, filval, minval+2] = [10, 11, 999, 12].
        let mut chunk = Vec::new();
        chunk.extend_from_slice(&3u32.to_le_bytes());
        chunk.push(8);
        chunk.extend_from_slice(&10u64.to_le_bytes());
        chunk.extend_from_slice(&[0u8; HEADER_LEN - 13]);
        chunk.extend_from_slice(&pack_offsets(&[0, 1, 7, 2], 3, 4).unwrap());
        let out = decompress(&chunk, &f, None).unwrap();
        let got: Vec<u32> = out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        assert_eq!(got, vec![10u32, 11, 999, 12]);
    }

    /// An integer filter carrying a defined fill value, the shape the
    /// reference library writes for any dataset with one.
    fn int_filter_with_fill(
        size: u32,
        signed: bool,
        order: u32,
        nelmts: u32,
        fill: u64,
    ) -> FilterDescription {
        // Through the writer rather than by packing the parameters here: a test
        // helper that laid them out itself would agree with a writer that laid
        // them out wrongly.
        let mut bytes = Vec::new();
        write_value(&mut bytes, fill, size as usize, order);
        let mut f = int_filter(size, signed, order, nelmts);
        f.client_data = build_cd_values(
            ScaleOffset::Integer(0),
            ScaleOffsetType {
                class: CLS_INTEGER,
                sign: if signed { SGN_2 } else { SGN_NONE },
                order,
            },
            size,
            nelmts,
            ScaleOffsetFill::Defined(Some(&bytes)),
        )
        .unwrap();
        f
    }

    /// A big-endian dataset's fill value is carried through `cd_values` as a
    /// *value*, not as the dataset's bytes: the writer converts on the way in
    /// and the decoder converts on the way out, matching what the reference does
    /// on a little-endian host. Convert on only one side and a fill element
    /// matches nothing — or, worse, something else does.
    ///
    /// `minbits` is what shows it. The non-fill values here span three code
    /// points, so recognizing the fill value packs them into two bits; missing
    /// it stretches the chunk's range across the whole distance to `0x0BAD`.
    #[test]
    fn a_big_endian_fill_value_is_recognized() {
        let fill = 0x0BADu64;
        let values: Vec<u16> = vec![0x0BAD, 3, 4, 0x0BAD, 5];
        for order in [ORDER_LE, ORDER_BE] {
            let raw: Vec<u8> = values
                .iter()
                .flat_map(|v| {
                    if order == ORDER_LE {
                        v.to_le_bytes()
                    } else {
                        v.to_be_bytes()
                    }
                })
                .collect();
            let f = int_filter_with_fill(2, false, order, values.len() as u32, fill);
            let packed = compress(&raw, &f).unwrap();

            assert_eq!(
                u32::from_le_bytes(packed[..4].try_into().unwrap()),
                2,
                "order {order}: the fill value must be recognized, not packed"
            );
            assert_eq!(
                decompress(&packed, &f, None).unwrap(),
                raw,
                "order {order}: round trip"
            );
        }
    }

    /// Encode with a defined fill value and decode it back (issue #287).
    ///
    /// `crates/crosscheck/tests/scaleoffset_fill.rs` proves the *bytes* match what the
    /// reference encoder produces, but it links the C library and so is gated
    /// to 64-bit little-endian. This states the round-trip property on every
    /// target, and covers the two chunk shapes that have no interior range:
    /// nothing but fill values, and no fill value at all.
    #[test]
    fn compress_round_trips_with_a_defined_fill_value() {
        let fill = 0xDEAD_u32;
        for (label, values) in [
            ("mixed", vec![10u32, fill, 11, 12, fill, 10]),
            ("all fill", vec![fill; 6]),
            ("no fill", vec![10u32, 11, 12, 13, 14, 15]),
            // min = 10, max = 13: a span of 4 needs 2 bits, whose all-ones
            // code is the offset 13 would otherwise take.
            ("sentinel collision", vec![10u32, 11, 12, 13, fill, 13]),
        ] {
            for order in [ORDER_LE, ORDER_BE] {
                let f = int_filter_with_fill(4, false, order, values.len() as u32, u64::from(fill));
                let raw: Vec<u8> = values
                    .iter()
                    .flat_map(|v| {
                        let le = v.to_le_bytes();
                        if order == ORDER_LE {
                            le.to_vec()
                        } else {
                            le.iter().rev().copied().collect()
                        }
                    })
                    .collect();
                let packed = compress(&raw, &f).unwrap();
                let out = decompress(&packed, &f, None).unwrap();
                assert_eq!(out, raw, "{label}, order {order}");
            }
        }
    }

    /// A float filter carrying a defined fill value.
    fn float_filter_with_fill(
        size: u32,
        decimals: i32,
        order: u32,
        nelmts: u32,
        fill: u64,
    ) -> FilterDescription {
        let mut f = float_filter(size, decimals, order, nelmts);
        f.client_data[PARM_FILAVAIL] = FILL_DEFINED;
        for (k, entry) in fill
            .to_le_bytes()
            .chunks(4)
            .take((size as usize).div_ceil(4))
            .enumerate()
        {
            let mut w = [0u8; 4];
            w[..entry.len()].copy_from_slice(entry);
            f.client_data[PARM_FILVAL + k] = u32::from_le_bytes(w);
        }
        f
    }

    /// The float counterpart of [`compress_round_trips_with_a_defined_fill_value`].
    ///
    /// Worth its own test rather than folding into the integer one: the
    /// crosschecks that pin the float encoding link the C library and so are
    /// gated to 64-bit little-endian, which leaves the `cross` i686 and s390x
    /// jobs with no float fill-defined coverage at all — and big-endian is
    /// where byte-order handling is most likely to be wrong.
    #[test]
    fn float_compress_round_trips_with_a_defined_fill_value() {
        let decimals = 3;
        for size in [4u32, 8] {
            for order in [ORDER_LE, ORDER_BE] {
                for (label, values) in [
                    ("mixed", vec![1.5f64, -999.0, 2.25, -999.0, 3.125]),
                    ("all fill", vec![-999.0f64; 5]),
                    ("no fill", vec![1.5f64, 2.25, 3.125, 4.0, 5.5]),
                    // Scaled by 10^3 this spans exactly 1024 values, so the
                    // largest offset is the all-ones code of an unwidened
                    // `minbits` and would decode as the fill value. The spans
                    // above cannot distinguish a reserved code point from an
                    // unreserved one; only a power-of-two span can.
                    (
                        "sentinel collision",
                        vec![0.0f64, 1.023, 0.5, -999.0, 1.023],
                    ),
                ] {
                    let fill_bits = if size == 4 {
                        u64::from((-999.0f32).to_bits())
                    } else {
                        (-999.0f64).to_bits()
                    };
                    let n = values.len() as u32;
                    let f = float_filter_with_fill(size, decimals, order, n, fill_bits);
                    let raw: Vec<u8> = values
                        .iter()
                        .flat_map(|v| {
                            let le = if size == 4 {
                                (*v as f32).to_bits().to_le_bytes().to_vec()
                            } else {
                                v.to_bits().to_le_bytes().to_vec()
                            };
                            if order == ORDER_LE {
                                le
                            } else {
                                le.iter().rev().copied().collect()
                            }
                        })
                        .collect();
                    let packed = compress(&raw, &f).unwrap();
                    let out = decompress(&packed, &f, None).unwrap();

                    let got: Vec<f64> = out
                        .chunks_exact(size as usize)
                        .map(|c| {
                            let mut b = c.to_vec();
                            if order == ORDER_BE {
                                b.reverse();
                            }
                            if size == 4 {
                                f64::from(f32::from_bits(u32::from_le_bytes(b.try_into().unwrap())))
                            } else {
                                f64::from_bits(u64::from_le_bytes(b.try_into().unwrap()))
                            }
                        })
                        .collect();
                    let tol = 0.501 * 10f64.powi(-decimals);
                    assert_eq!(
                        got.len(),
                        values.len(),
                        "{label}, size {size}, order {order}"
                    );
                    for (g, v) in got.iter().zip(values.iter()) {
                        assert!(
                            (g - v).abs() <= tol,
                            "{label}, size {size}, order {order}: got {g}, want {v}"
                        );
                    }
                }
            }
        }
    }

    /// The pass-through mode and the width refusal beside it, on **both**
    /// directions. The reference takes both before it splits on compress versus
    /// decompress, so a reader that honours one and a writer that does not
    /// disagree about the same file.
    ///
    /// This is the unit half of
    /// `a_full_width_minbits_stores_the_chunk_unfiltered`, which needs the C
    /// library and so does not run on the 32-bit or big-endian jobs.
    #[test]
    fn a_full_width_scale_factor_passes_the_chunk_through_both_ways() {
        let raw: Vec<u8> = (0..40u8).collect();

        for size in [1u32, 2, 4, 8] {
            let bits = size * 8;
            let nelmts = raw.len() as u32 / size;

            let mut f = int_filter(size, false, ORDER_LE, nelmts);
            f.client_data[PARM_SCALEFACTOR] = bits;
            assert_eq!(compress(&raw, &f).unwrap(), raw, "size {size}: compress");
            assert_eq!(
                decompress(&raw, &f, None).unwrap(),
                raw,
                "size {size}: decompress"
            );

            // One bit past the datatype is not a width at all.
            f.client_data[PARM_SCALEFACTOR] = bits + 1;
            assert!(
                matches!(compress(&raw, &f), Err(FormatError::FilterError(_))),
                "size {size}: compress must refuse a width past the datatype"
            );
            assert!(
                matches!(decompress(&raw, &f, None), Err(FormatError::FilterError(_))),
                "size {size}: decompress must refuse it too"
            );
        }
    }

    /// The pass-through is an early return, so everything that must not be
    /// skipped has to sit above it. A `cd_values` whose class and scale type
    /// disagree is refused whatever the requested width — the reference
    /// validates that first.
    #[test]
    fn a_pass_through_width_does_not_skip_the_class_check() {
        let raw = vec![0u8; 32];
        // Float class carrying the *integer* scale type, at the pass-through
        // width. Reachable only from a malformed parameter array.
        let mut f = float_filter(4, 0, ORDER_LE, 8);
        f.client_data[PARM_SCALETYPE] = SO_INT;
        f.client_data[PARM_SCALEFACTOR] = 32;
        assert!(matches!(
            compress(&raw, &f),
            Err(FormatError::FilterError(_))
        ));

        // Integer class carrying float E-scale, likewise.
        let mut f = int_filter(4, false, ORDER_LE, 8);
        f.client_data[PARM_SCALETYPE] = SO_FLOAT_ESCALE;
        f.client_data[PARM_SCALEFACTOR] = 32;
        assert!(matches!(
            compress(&raw, &f),
            Err(FormatError::FilterError(_))
        ));
    }

    /// `precompress_integer` has *two* full-precision returns and they carry
    /// different `minval`s. The spread guard fires before the chunk's minimum is
    /// known to be usable and leaves zero; the `minbits >= full_bits` return
    /// below it is reached with the minimum computed, and the reference's
    /// trailing `*minval = min` does run there.
    ///
    /// `the_full_precision_fallback_header_matches_the_c_library` reaches only
    /// the first. This reaches the second: a spread wide enough that
    /// `ceil_log2` saturates the datatype without exceeding `width_max - 2`.
    #[test]
    fn the_second_full_precision_return_carries_the_chunk_minimum() {
        // u8, min = 1, max = 128: spread 127, inside the guard (<= 253), and
        // with a fill value `ceil_log2(127 + 2) = 8` — the full width.
        let values: Vec<u8> = vec![9, 1, 128, 9, 1, 128, 2, 3];
        let f = int_filter_with_fill(1, false, ORDER_LE, values.len() as u32, 9);
        let packed = compress(&values, &f).unwrap();
        assert_eq!(
            u32::from_le_bytes(packed[..4].try_into().unwrap()),
            8,
            "the fixture must reach the full width"
        );
        assert_eq!(
            u64::from_le_bytes(packed[5..13].try_into().unwrap()),
            1,
            "the second full-precision return carries the chunk minimum, not zero"
        );
        // Stored raw after the header, never packed at the full width.
        assert_eq!(&packed[HEADER_LEN..], &values[..]);
    }

    /// Reserving a code point for the sentinel is what makes the round trip
    /// above possible, and it is visible in the encoded header: a chunk with a
    /// defined fill value covers `span + 1` values where one without covers
    /// `span`.
    ///
    /// Swept over every span rather than checked at a hand-picked one. The two
    /// formulas agree at most spans — they differ only where `span + 1` crosses
    /// a power of two — so a single fixture is as likely to be blind to the
    /// reservation as to catch it, and equally blind to a *double* reservation.
    #[test]
    fn a_defined_fill_value_reserves_exactly_one_code_point() {
        let fill = 0xDEAD_u32;
        for span in 1..=40u32 {
            // `span` distinct values, 0..span, plus one fill element.
            let mut values: Vec<u32> = (0..span).collect();
            values.push(fill);
            let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let n = values.len() as u32;

            let minbits_of = |f: &FilterDescription| {
                let packed = compress(&raw, f).unwrap();
                u32::from_le_bytes(packed[..4].try_into().unwrap())
            };

            // Without a fill value the lone `fill` element is just another
            // value, so the span is the whole 0..=fill range; use a separate
            // buffer for that half.
            let plain: Vec<u8> = (0..span).flat_map(|v| v.to_le_bytes()).collect();
            let packed = compress(&plain, &int_filter(4, false, ORDER_LE, span)).unwrap();
            let plain_minbits = u32::from_le_bytes(packed[..4].try_into().unwrap());
            assert_eq!(plain_minbits, ceil_log2(u64::from(span)), "span {span}");

            assert_eq!(
                minbits_of(&int_filter_with_fill(
                    4,
                    false,
                    ORDER_LE,
                    n,
                    u64::from(fill)
                )),
                ceil_log2(u64::from(span) + 1),
                "span {span}: a defined fill value must reserve one code point, \
                 no more and no less"
            );
        }
    }

    /// Float E-scale is forbidden by the reference library too; we error
    /// before touching the payload.
    #[test]
    fn decompress_rejects_escale() {
        let mut cd = vec![0u32; TOTAL_NPARMS];
        cd[PARM_SCALETYPE] = SO_FLOAT_ESCALE;
        cd[PARM_NELMTS] = 4;
        cd[PARM_CLASS] = CLS_FLOAT;
        cd[PARM_SIZE] = 4;
        cd[PARM_ORDER] = ORDER_LE;
        let f = FilterDescription {
            filter_id: crate::filter_pipeline::FILTER_SCALEOFFSET,
            name: None,
            flags: 0,
            client_data: cd,
        };
        let chunk = vec![0u8; HEADER_LEN + 4];
        assert!(matches!(
            decompress(&chunk, &f, None),
            Err(FormatError::FilterError(_))
        ));
    }

    /// A corrupt header that claims `minbits > size * 8` must error rather
    /// than reach the bit reader and panic.
    #[test]
    fn decompress_rejects_oversized_minbits_header() {
        let f = int_filter(4, false, ORDER_LE, 4);
        // size=4 → full_bits=32. Set minbits=33.
        let mut bad = Vec::new();
        bad.extend_from_slice(&33u32.to_le_bytes());
        bad.push(8);
        bad.extend_from_slice(&0u64.to_le_bytes());
        bad.extend_from_slice(&[0u8; HEADER_LEN - 13]);
        bad.extend_from_slice(&[0u8; 16]); // dummy payload
        assert!(matches!(
            decompress(&bad, &f, None),
            Err(FormatError::FilterError(_))
        ));
    }

    /// Signed 1-byte ints: covers the `size == 1` arm of `sign_extend`,
    /// which the unsigned `u8` round-trip skips.
    #[test]
    fn integer_i8_roundtrip_with_negatives() {
        let vals: [i8; 6] = [-100, -50, 0, 27, -99, 100];
        let raw: Vec<u8> = vals.iter().map(|&v| v as u8).collect();
        let f = int_filter(1, true, ORDER_LE, vals.len() as u32);
        let comp = compress(&raw, &f).unwrap();
        let dec = decompress(&comp, &f, None).unwrap();
        assert_eq!(dec, raw);
    }

    /// `nelmts == 1` exercises the streaming min/max loops in
    /// `precompress_{integer,float}` with an empty body (the `1..nelmts`
    /// range yields nothing). Span is 0, minbits is 0 — round-trips both
    /// integer and float through the all-equal short-circuit.
    #[test]
    fn single_element_chunk_roundtrip() {
        let raw = 42u32.to_le_bytes().to_vec();
        let f = int_filter(4, false, ORDER_LE, 1);
        let comp = compress(&raw, &f).unwrap();
        let dec = decompress(&comp, &f, None).unwrap();
        assert_eq!(dec, raw);

        let raw = 3.14f64.to_le_bytes().to_vec();
        let f = float_filter(8, 3, ORDER_LE, 1);
        let comp = compress(&raw, &f).unwrap();
        let dec = decompress(&comp, &f, None).unwrap();
        let got = f64::from_le_bytes(dec.as_slice().try_into().unwrap());
        assert!((got - 3.14).abs() <= 0.5e-3);
    }

    #[test]
    fn scale_offset_type_from_datatype_classes() {
        let i32_ty = Datatype::FixedPoint {
            size: 4,
            byte_order: DatatypeByteOrder::LittleEndian,
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        };
        let so = scale_offset_type_from_datatype(&i32_ty).unwrap();
        assert_eq!(so.class, CLS_INTEGER);
        assert_eq!(so.sign, SGN_2);
        assert_eq!(so.order, ORDER_LE);

        let f64_ty = Datatype::FloatingPoint {
            size: 8,
            byte_order: DatatypeByteOrder::BigEndian,
            bit_offset: 0,
            bit_precision: 64,
            exponent_location: 52,
            exponent_size: 11,
            mantissa_location: 0,
            mantissa_size: 52,
            exponent_bias: 1023,
        };
        let so = scale_offset_type_from_datatype(&f64_ty).unwrap();
        assert_eq!(so.class, CLS_FLOAT);
        assert_eq!(so.order, ORDER_BE);
    }
}
