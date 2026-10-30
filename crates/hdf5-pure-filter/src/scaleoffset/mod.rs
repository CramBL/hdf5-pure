//! Encodes and decodes chunks with the HDF5 Scale-Offset filter (filter ID 6).
//!
//! Integer mode packs offsets from each chunk's minimum without loss. Float D-scale mode
//! rounds values to a decimal scale before packing. An encoded chunk has a 21-byte header
//! followed by offsets packed most significant bit first. A full-width integer scale factor
//! passes the input through without a header. The dataset's byte order applies to element bytes,
//! while the packed offsets have the same bit order for every dataset.

use alloc::format;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use hdf5_pure_core::FormatError;

use crate::error::Error;
use crate::pipeline::FILTER_SCALEOFFSET;
use crate::pipeline::buffer::{FilterBuffer, InitializedBuffer};

// `cd_values` indices (H5Z_SCALEOFFSET_PARM_*).
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

/// The number of bytes before a chunk's packed offsets.
///
/// A chunk that cannot be packed more tightly stores its input after this header, so encoding
/// expands the input by at most this many bytes.
pub(crate) const HEADER_LEN: usize = 21;

/// Selects integer packing or decimal scaling for a chunk.
#[allow(clippy::exhaustive_enums)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleOffset {
    /// Packs integer offsets without loss.
    ///
    /// Zero selects the bit width from each chunk's range. A value equal to the datatype's bit
    /// width stores the chunk unchanged. Other values are recorded in the filter parameters.
    /// The encoder selects a width that preserves every value in the chunk.
    Integer(u32),
    /// Rounds floating-point values to `decimals` decimal places before packing.
    FloatDScale(i32),
}

/// Selects the byte order of scalar elements passed to the filter.
#[allow(clippy::exhaustive_enums)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScaleOffsetByteOrder {
    /// Stores each element's most significant byte first.
    BigEndian,
    /// Stores each element's least significant byte first.
    LittleEndian,
}

impl ScaleOffsetByteOrder {
    fn code(self) -> u32 {
        match self {
            Self::LittleEndian => ORDER_LE,
            Self::BigEndian => ORDER_BE,
        }
    }
}

/// Holds the scalar class, sign, and byte order used to build filter parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScaleOffsetType {
    /// `CLS_INTEGER` or `CLS_FLOAT`.
    class: u32,
    /// `SGN_NONE` or `SGN_2` (only meaningful for integers).
    sign: u32,
    /// `ORDER_LE` or `ORDER_BE`.
    order: u32,
}

impl ScaleOffsetType {
    /// Describes an integer scalar by its sign and byte order.
    pub fn integer(signed: bool, order: ScaleOffsetByteOrder) -> Self {
        Self {
            class: CLS_INTEGER,
            sign: if signed { SGN_2 } else { SGN_NONE },
            order: order.code(),
        }
    }

    /// Describes a floating-point scalar by its byte order.
    pub fn floating(order: ScaleOffsetByteOrder) -> Self {
        Self {
            class: CLS_FLOAT,
            sign: SGN_NONE,
            order: order.code(),
        }
    }
}

/// Indicates whether filter parameters record a fill value.
///
/// This value represents `H5Z_SCALEOFFSET_PARM_FILAVAIL` without borrowing the fill bytes.
/// [`Defined`](Self::Defined) also applies when the dataset uses the default fill value of zero.
#[allow(clippy::exhaustive_enums)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillAvailability {
    /// Records the fill value and encodes matching elements with the reserved fill code.
    Defined,
    /// Encodes every element as an ordinary offset.
    Undefined,
}

/// Supplies fill availability and element bytes for filter parameters.
#[allow(clippy::exhaustive_enums)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleOffsetFill<'a> {
    /// Records one element in dataset byte order, or zero when the value is [`None`].
    Defined(Option<&'a [u8]>),
    /// Records undefined availability and zero value parameters.
    Undefined,
}

/// Builds the filter parameters for one chunk of scalar elements.
///
/// `nelmts` is the number of elements in the chunk. Fill bytes use the dataset's byte order.
/// The filter parameters store their bit pattern in value order.
///
/// # Errors
///
/// Returns [`Error::ScaleOffset`] if the mode does not match the scalar class, the element size
/// is outside `1..=8`, or the supplied fill value has the wrong length.
pub fn build_cd_values(
    mode: ScaleOffset,
    ty: ScaleOffsetType,
    size: u32,
    nelmts: u32,
    fill: ScaleOffsetFill<'_>,
) -> Result<Vec<u32>, Error> {
    let (scale_type, scale_factor) = match (mode, ty.class) {
        (ScaleOffset::Integer(minbits), CLS_INTEGER) => (SO_INT, minbits),
        (ScaleOffset::FloatDScale(decimals), CLS_FLOAT) => (SO_FLOAT_DSCALE, decimals as u32),
        (ScaleOffset::Integer(_), _) => {
            return Err(Error::ScaleOffset(
                "scaleoffset: integer mode requires an integer dataset".to_string(),
            ));
        }
        (ScaleOffset::FloatDScale(_), _) => {
            return Err(Error::ScaleOffset(
                "scaleoffset: float D-scale mode requires a floating-point dataset".to_string(),
            ));
        }
    };

    // The same 1..=8-byte scalars [`Parms::parse`] accepts on the way back in.
    // Stated here because
    // the two arrive as separate arguments: a wider one would run the fill value
    // off the end of the parameter array, which is what makes
    // [`write_fill_bits`] safe to write without a bound of its own.
    if size == 0 || size > 8 {
        return Err(Error::ScaleOffset(format!(
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
            let width = to_usize(u64::from(size))?;
            let bits = match bytes {
                // The library default: a defined fill value of zero.
                None => 0,
                Some(b) if b.len() == width => read_value(b, width, ty.order),
                Some(b) => {
                    return Err(Error::ScaleOffset(format!(
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

/// Reads a mode and fill availability from filter parameters.
///
/// Returns [`None`] if the parameters are too short, specify float E-scale or an unknown mode,
/// or have invalid fill availability. A caller that rebuilds the parameters supplies the fill
/// value separately from the returned [`FillAvailability`].
pub fn scale_offset_mode(cd_values: &[u32]) -> Option<(ScaleOffset, FillAvailability)> {
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
            reason = "the filter stores signed D-scale factors as u32 bits, including negative values"
        )]
        SO_FLOAT_DSCALE => ScaleOffset::FloatDScale(scale_factor as i32),
        // SO_FLOAT_ESCALE (and anything unrecognized) is never written by this
        // crate and cannot be reproduced.
        _ => return None,
    };
    Some((mode, fill))
}

/// Parses the scalar metadata and mode of a stored scale-offset filter.
///
/// Returns [`None`] for floating-point E-scale mode. `H5Z__filter_scaleoffset` in
/// `H5Zscaleoffset.c`, HDF5 2.2.0, requires `H5Z_SCALEOFFSET_TOTAL_NPARMS` values and interprets
/// their scalar class and mode together.
///
/// # Errors
///
/// Returns [`FormatError::InvalidFilterParameterCount`] if `values` does not contain 20 entries.
/// Returns a scale-offset variant of [`FormatError`] for invalid scalar metadata or a mode that
/// does not match the scalar class.
pub fn parse_scale_offset_parameters(values: &[u32]) -> Result<Option<ScaleOffset>, FormatError> {
    if values.len() != TOTAL_NPARMS {
        return Err(FormatError::InvalidFilterParameterCount {
            filter_id: FILTER_SCALEOFFSET,
            expected: TOTAL_NPARMS,
            actual: values.len(),
        });
    }
    if values[PARM_NELMTS] == 0 {
        return Err(FormatError::InvalidScaleOffsetElementCount {
            count: values[PARM_NELMTS],
        });
    }
    let size = values[PARM_SIZE];
    if !(1..=8).contains(&size) {
        return Err(FormatError::InvalidScaleOffsetElementSize { size });
    }
    if !matches!(values[PARM_ORDER], ORDER_LE | ORDER_BE) {
        return Err(FormatError::InvalidScaleOffsetByteOrder {
            order: values[PARM_ORDER],
        });
    }
    if !matches!(values[PARM_FILAVAIL], FILL_UNDEFINED | FILL_DEFINED) {
        return Err(FormatError::InvalidScaleOffsetFillAvailability {
            value: values[PARM_FILAVAIL],
        });
    }
    match (values[PARM_SCALETYPE], values[PARM_CLASS]) {
        (SO_INT, CLS_INTEGER) => {
            if !matches!(values[PARM_SIGN], SGN_NONE | SGN_2) {
                return Err(FormatError::InvalidScaleOffsetIntegerSign {
                    sign: values[PARM_SIGN],
                });
            }
            if values[PARM_SCALEFACTOR] > size * 8 {
                return Err(FormatError::InvalidScaleOffsetIntegerPrecision {
                    bits: values[PARM_SCALEFACTOR],
                    element_size: size,
                });
            }
            Ok(Some(ScaleOffset::Integer(values[PARM_SCALEFACTOR])))
        }
        (SO_FLOAT_DSCALE | SO_FLOAT_ESCALE, CLS_FLOAT) => {
            if !matches!(size, 4 | 8) {
                return Err(FormatError::InvalidScaleOffsetFloatSize { size });
            }
            if values[PARM_SCALETYPE] == SO_FLOAT_ESCALE {
                return Ok(None);
            }
            Ok(Some(ScaleOffset::FloatDScale(i32::from_ne_bytes(
                values[PARM_SCALEFACTOR].to_ne_bytes(),
            ))))
        }
        _ => Err(FormatError::InvalidScaleOffsetMode {
            mode: values[PARM_SCALETYPE],
            class: values[PARM_CLASS],
        }),
    }
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
    fn parse(cd: &[u32]) -> Result<Parms, Error> {
        if cd.len() < CORE_NPARMS {
            return Err(Error::ScaleOffset(
                "scaleoffset: too few cd_values".to_string(),
            ));
        }
        let class = cd[PARM_CLASS];
        if class != CLS_INTEGER && class != CLS_FLOAT {
            return Err(Error::ScaleOffset(
                "scaleoffset: unsupported datatype class".to_string(),
            ));
        }
        let size = cd[PARM_SIZE] as usize;
        if size == 0 || size > 8 {
            return Err(Error::ScaleOffset(
                "scaleoffset: unsupported datatype size".to_string(),
            ));
        }
        if class == CLS_FLOAT && size != 4 && size != 8 {
            return Err(Error::ScaleOffset(
                "scaleoffset: float size must be 4 or 8".to_string(),
            ));
        }
        let order = cd[PARM_ORDER];
        if order != ORDER_LE && order != ORDER_BE {
            return Err(Error::ScaleOffset(
                "scaleoffset: bad byte order".to_string(),
            ));
        }
        #[expect(
            clippy::cast_possible_wrap,
            reason = "the filter reinterprets the u32 parameter as a signed 32-bit scale factor"
        )]
        Ok(Parms {
            scale_type: cd[PARM_SCALETYPE],
            scale_factor: cd[PARM_SCALEFACTOR] as i32,
            nelmts: to_usize(u64::from(cd[PARM_NELMTS]))?,
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
/// compress versus decompress (`H5Zscaleoffset.c`, HDF5 2.2.0, the `H5Z_FLAG_REVERSE` test),
/// so they hold identically in both directions: a requested width wider than the
/// datatype is an error, and a width *equal* to it makes the filter a
/// pass-through that stores the chunk unchanged, with no header of its own.
///
/// Float D-scale is exempt, because there `scale_factor` is a decimal exponent
/// which is a decimal exponent.
///
/// Returns `Ok(true)` when the caller should hand the chunk through untouched.
///
/// Both directions use the same width decision, so the decoder recognizes a
/// chunk the encoder stored without a header.
fn integer_scale_factor_is_full_width(p: &Parms) -> Result<bool, Error> {
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
        return Err(Error::ScaleOffset(format!(
            "scaleoffset: requested minimum bit width {} exceeds the {type_bits}-bit datatype",
            p.scale_factor
        )));
    }
    Ok(p.scale_factor == type_bits_i)
}

/// Decodes a chunk into element bytes in the dataset's byte order.
///
/// `max_output` bounds the declared decoded size before allocation. Pass the maximum number of bytes
/// the chunk can hold when its parameters come from a file.
///
/// Full-width integer mode preserves the input view. Raw full-precision payloads return a view of
/// the element bytes alone. The declared-size bound is checked before either view is returned.
/// Other modes replace the input with owned output and recycle displaced owned storage into `spare`.
///
/// # Errors
///
/// Returns [`Error::ScaleOffset`] if the parameters or encoded bytes are invalid, or the
/// decoded size exceeds `max_output`. Returns
/// [`Error::ScaleOffsetValueTooLargeForPlatform`] if an element count or output size does not
/// fit `usize`.
pub(crate) fn decompress<'a>(
    current: FilterBuffer<'a>,
    cd: &[u32],
    max_output: Option<usize>,
    spare: &mut InitializedBuffer,
) -> Result<FilterBuffer<'a>, Error> {
    let input = current.as_slice();
    let p = Parms::parse(cd)?;

    if p.scale_type == SO_FLOAT_ESCALE {
        return Err(Error::ScaleOffset(
            "scaleoffset: float E-scale method is not supported".to_string(),
        ));
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "p.size is validated to 1..=8, so p.size * 8 is 8..=64 and fits in u32"
    )]
    let full_bits = (p.size * 8) as u32;
    // `nelmts` is file-derived. `nelmts * size` is an allocation size and a
    // slice bound. The product is computed in `u64` (it cannot overflow there:
    // `nelmts` originates from a `u32` and `size` is 1..=8) and narrowed to
    // `usize`, which reports an error for a value too large for a 32-bit host.
    let size_out = to_usize(p.nelmts as u64 * p.size as u64)?;

    // `size_out` sizes an allocation, and `nelmts` comes from the filter's own
    // `cd_values`. The chunk geometry does not constrain this value, so nothing has yet tied it
    // to a plausible size. Unlike the byte compressors this decoder cannot bound
    // itself by the input length  -  the `minbits == 0` path below expands a
    // header into `size_out` bytes with no payload at all  -  so the check is
    // against the size the pipeline allows for this filter stage. A chunk
    // claiming more than that bound is rejected before allocation.
    if let Some(cap) = max_output
        && size_out > cap
    {
        return Err(Error::ScaleOffset(format!(
            "scaleoffset: chunk declares {size_out} decoded bytes, more than the \
             {cap} the chunk can hold"
        )));
    }

    // Pass-through mode, and the width rejection beside it. see
    // [`integer_scale_factor_is_full_width`].
    if integer_scale_factor_is_full_width(&p)? {
        return Ok(current);
    }

    // Header: minbits + minval.
    if input.len() < 5 {
        return Err(Error::ScaleOffset(
            "scaleoffset: chunk shorter than header".to_string(),
        ));
    }
    let minbits = u32::from_le_bytes([input[0], input[1], input[2], input[3]]);
    if minbits > full_bits {
        return Err(Error::ScaleOffset(
            "scaleoffset: minbits exceeds datatype size".to_string(),
        ));
    }
    let minval_size = (input[4] as usize).min(8);
    if input.len() < 5 + minval_size {
        return Err(Error::ScaleOffset(
            "scaleoffset: chunk too short for minval".to_string(),
        ));
    }
    let mut minval_bytes = [0u8; 8];
    minval_bytes[..minval_size].copy_from_slice(&input[5..5 + minval_size]);
    let minval = u64::from_le_bytes(minval_bytes);

    // Raw payload (no per-element packing): minbits at full precision.
    if minbits == full_bits {
        let start = HEADER_LEN;
        let end = start
            .checked_add(size_out)
            .filter(|&end| end <= input.len())
            .ok_or_else(|| {
                Error::ScaleOffset("scaleoffset: chunk too short for raw payload".to_string())
            })?;
        return current.narrow(start..end);
    }

    // minbits == 0: there is no payload, so every element takes the same value.
    // Which value is not the same in both fill modes, and the reference reaches
    // the result by not special-casing this at all  -  `H5Z__filter_scaleoffset`
    // in `H5Zscaleoffset.c`, HDF5 2.2.0,
    // zeroes the output buffer and still runs its postdecompress, where the
    // reserved fill code `(1 << 0) - 1` is zero and therefore matches every zeroed
    // element. So a fill-*defined* chunk decodes entirely to the fill value,
    // and only a fill-undefined one decodes to `minval`.
    //
    // No conforming encoder emits this pair  -  reserving the reserved fill code keeps
    // `minbits` at 1 or more whenever a fill value is defined  -  so it is only
    // reachable from a hand-built or damaged chunk. Measured against the C
    // library in
    // `minbits_zero_with_fill_defined_reads_as_the_fill_value`.
    let mut out = Vec::with_capacity(size_out);
    if minbits == 0 {
        let mask = p.width_mask();
        // `read_fill_bits` writes only `size` bytes into a zeroed buffer, so the
        // recovered value never carries bits above the datatype width and needs
        // no mask of its own  -  unlike `minval`, which comes off the chunk header
        // as a full 8 bytes whatever the datatype is.
        let bits = if p.filavail == FILL_DEFINED {
            read_fill_bits(cd, p.size)?
        } else {
            minval & mask
        };
        for _ in 0..p.nelmts {
            write_value(&mut out, bits, p.size, p.order);
        }
        return Ok(current.replace(spare, out));
    }

    if input.len() < HEADER_LEN {
        return Err(Error::ScaleOffset(
            "scaleoffset: chunk too short for packed payload".to_string(),
        ));
    }
    let payload = &input[HEADER_LEN..];
    // Validate payload length once, up front. the BitReader requires the
    // caller to not read past the end. Compare in `u64` so the bit counts
    // (`nelmts` is file-derived, `minbits` up to 64) cannot overflow a 32-bit
    // `usize` and falsely pass the check.
    if (payload.len() as u64) * 8 < p.nelmts as u64 * minbits as u64 {
        return Err(Error::ScaleOffset(
            "scaleoffset: payload too short for packed data".to_string(),
        ));
    }
    let mut reader = BitReader::new(payload);

    if p.class == CLS_INTEGER {
        reconstruct_integer(&mut out, &mut reader, &p, minbits, minval, cd)?;
    } else {
        reconstruct_float(&mut out, &mut reader, &p, minbits, minval, cd)?;
    }
    Ok(current.replace(spare, out))
}

/// Encodes one full chunk of scalar element bytes.
///
/// Encoded modes require an input length equal to the element count times the size recorded in
/// `cd`. A full-width integer scale factor preserves the input view. Other modes replace the input
/// with encoded bytes and recycle displaced owned storage into `spare`. The encoder uses a
/// reserved offset code for defined fill values, so matching elements do not contribute to the
/// chunk's value range.
///
/// # Errors
///
/// Returns [`Error::ScaleOffset`] if the parameters and scalar class disagree, the element
/// width is invalid, or the input length differs from the recorded chunk size in an encoded mode.
/// Returns
/// [`Error::ScaleOffsetValueTooLargeForPlatform`] if an element count or encoded size does not
/// fit `usize`.
pub(crate) fn compress<'a>(
    current: FilterBuffer<'a>,
    cd: &[u32],
    spare: &mut InitializedBuffer,
) -> Result<FilterBuffer<'a>, Error> {
    let input = current.as_slice();
    let p = Parms::parse(cd)?;

    // The class/scale-type agreement is checked first, matching the reference's
    // order  -  the pass-through below is an early return, so anything it must not
    // skip has to come above it.
    if p.class == CLS_INTEGER && p.scale_type != SO_INT {
        return Err(Error::ScaleOffset(
            "scaleoffset: integer class requires integer scale type".to_string(),
        ));
    }
    if p.class == CLS_FLOAT && p.scale_type != SO_FLOAT_DSCALE {
        return Err(Error::ScaleOffset(
            "scaleoffset: float class requires D-scale scale type".to_string(),
        ));
    }

    if integer_scale_factor_is_full_width(&p)? {
        return Ok(current);
    }

    // `nelmts * size` in `u64` (cannot overflow there) narrowed to `usize`,
    // so a file-derived `nelmts` cannot wrap a 32-bit `usize` and spuriously
    // equal `input.len()`.
    let expected = to_usize(p.nelmts as u64 * p.size as u64)?;
    if input.len() != expected {
        return Err(Error::ScaleOffset(
            "scaleoffset: chunk size does not match nelmts * datatype size".to_string(),
        ));
    }
    if p.nelmts == 0 {
        return Ok(current.replace(spare, emit(0, 0, &[])));
    }

    let signed = cd[PARM_SIGN] == SGN_2;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "p.size is validated to 1..=8, so p.size * 8 is 8..=64 and fits in u32"
    )]
    let full_bits = (p.size * 8) as u32;

    // The fill value the filter parameters carry, or `None` when they record
    // `FILL_UNDEFINED`. Read once for this chunk.
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
        let output = emit_raw(full_bits, minval, input);
        return Ok(current.replace(spare, output));
    }
    let payload = pack_offsets(&offsets, minbits, p.nelmts)?;
    Ok(current.replace(spare, emit(minbits, minval, &payload)))
}

// --- integer reconstruction / pre-compression ----------------------------

fn reconstruct_integer(
    out: &mut Vec<u8>,
    reader: &mut BitReader<'_>,
    p: &Parms,
    minbits: u32,
    minval: u64,
    cd: &[u32],
) -> Result<(), Error> {
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
/// and `_2` in `H5Zscaleoffset.c`, HDF5 2.2.0:
///
/// * the chunk's range is taken over the **non-fill** elements only (a chunk
///   that is nothing but fill values leaves `min == max == 0`, the reference's
///   input values).
/// * `minbits` covers `span + 1`, reserving one code point
///   so the all-ones reserved fill code can never also be a legitimate offset.
/// * each fill-valued element is stored as that reserved fill code with no
///   `value - min`.
fn precompress_integer(
    input: &[u8],
    p: &Parms,
    signed: bool,
    filval: Option<u64>,
) -> (u32, u64, Vec<u64>) {
    // Stream the input twice (min/max, then offsets) to avoid allocating
    // a `Vec<i128>`. The intermediate dominated cache pressure on large chunks
    // (16 B/element vs the source's 1-8 B/element). two cache-warm passes
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
    // test compares values across widths.
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
            // `H5Z_scaleoffset_max_min_1` in `H5Zscaleoffset.c`, HDF5 2.2.0:
            // skip fill values, and leave both
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

    // Overflow guard mirrors `H5Z_scaleoffset_check_{1,2}` in
    // `H5Zscaleoffset.c`, HDF5 2.2.0: a span within two
    // of the full range can't gain from packing, so store at full precision.
    //
    // The reference reaches this by an early `return` from the precompress
    // function, which skips the `*minval = min` its packed paths end with  -  so
    // the header carries the zero the caller initialized, not the chunk's
    // minimum.
    //
    // With one exception, and it is an accident of the reference's own source
    // because of a reference implementation difference: `signed char` is the one type `H5Z__scaleoffset_
    // precompress_i` hand-expands without the
    // `H5Z_scaleoffset_check_2` macro, and the expansion assigns `*minval` in
    // its fill-*undefined* early return and not in its fill-defined one. So a
    // 1-byte signed chunk with no fill value carries the minimum here where
    // every other combination carries zero.
    //
    // A decoder ignores `minval` once `minbits` is the full width, so none of
    // this changes what any reader returns. it is what writing the same bytes
    // as the reference costs.
    let width_max: u128 = p.width_mask() as u128;
    let spread = (max - min) as u128;
    if spread > width_max.saturating_sub(2) {
        let schar_quirk = signed && p.size == 1 && filval.is_none();
        return (full_bits, if schar_quirk { minval } else { 0 }, Vec::new());
    }

    // `span` is the reference's `max - min + 1`. a defined fill value costs one
    // more code point for the reserved fill code. The guard above leaves room for both:
    // `spread <= width_max - 2` makes `spread + 2 <= width_max`.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the guard above returns early unless spread <= width_max - 2, and width_max <= u64::MAX, so spread + 2 fits in u64"
    )]
    let span = (spread as u64) + 1;
    let minbits = ceil_log2(if filval.is_some() { span + 1 } else { span });
    // Unlike the spread guard above, this return is reached with `min` computed,
    // and the reference's trailing `*minval = min` does run here  -  the two
    // full-precision returns carry different headers.
    //
    // The `>=` comparison skips building offsets that
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
) -> Result<(), Error> {
    let sentinel = sentinel(minbits);
    let decimals = p.scale_factor;
    let filval = if p.filavail == FILL_DEFINED {
        Some(read_fill_bits(cd, p.size)?)
    } else {
        None
    };
    // Dispatch on width once: pow, min, and the size/order arguments are all
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
/// widens `minbits` and diverts matching elements to the reserved fill code the same way,
/// with one difference worth stating: the reference does **not** match a float
/// element against the fill value by equality but by *proximity*, treating any
/// element within one decimal quantum (`|v - fill| < 10^-D`) as a fill value.
///
/// It also applies that test at two different precisions for `f32` data  -
/// `H5Z_scaleoffset_max_min_3` hardcodes the `double` `fabs`/`pow` while
/// `H5Z_scaleoffset_modify_1` uses the type's own `fabsf`/`powf`
/// (`H5Zscaleoffset.c`, HDF5 2.2.0), so an `f32`
/// element can sit inside one tolerance and outside the other. Both are
/// mirrored as written, since agreeing with the C
/// encoder is the whole point.
fn precompress_float(input: &[u8], p: &Parms, filval: Option<u64>) -> (u32, u64, Vec<u64>) {
    // Two streaming passes over the input bytes (min/max, then offsets),
    // no intermediate `Vec<f32>` or `Vec<f64>`. `min_scaled = min * pow` is hoisted
    // out of the per-element loop so we don't recompute the same product N
    // times  -  bit-identical to the previous `v * pow - min * pow`.
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
/// with O(nelmts) iterations  -  roughly an
/// order of magnitude fewer instructions in the hot path for typical
/// minbits.
fn pack_offsets(offsets: &[u64], minbits: u32, nelmts: usize) -> Result<Vec<u8>, Error> {
    // `nelmts * minbits` is a bit count that sizes the payload buffer. Compute
    // it in `u64` (cannot overflow there: `nelmts` originates from a `u32` and
    // `minbits` is <= 64) and narrow the resulting byte length to `usize`,
    // reporting an error when the value exceeds `usize` on a 32-bit host.
    let payload_len = to_usize(nelmts as u64 * minbits as u64 / 8 + 1)?;
    if minbits == 0 {
        // All offsets are zero. the reference layout is just `payload_len`
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
            // The merged value spans more than a u64. emit 8 bytes from the
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
        // Left-align the remaining bits into a final byte. the low bits are
        // zero padding within the partial byte.
        buf.push(((acc << (8 - nbits)) & 0xFF) as u8);
    }
    // For bit streams that end on a byte boundary, the reference still
    // reserves one trailing zero. The formula above sized the buffer for
    // either case. pad here.
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
/// decompress path drives reconstruction directly off [`BitReader`].
#[cfg(test)]
fn unpack_bits(payload: &[u8], nelmts: usize, minbits: u32) -> Result<Vec<u64>, Error> {
    let total_bits = nelmts * minbits as usize;
    if payload.len() * 8 < total_bits {
        return Err(Error::ScaleOffset(
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
/// `H5Z__scaleoffset_set_parms_fillval` writes (`H5Zscaleoffset.c`, HDF5 2.2.0).
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
fn read_fill_bits(cd: &[u32], size: usize) -> Result<u64, Error> {
    let entries = size.div_ceil(4);
    if cd.len() < PARM_FILVAL + entries {
        return Err(Error::ScaleOffset(
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
/// require `std`). `exp` is a small decimal scale factor. exponentiation by
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
/// once, so the two can land one unit in last place apart  -  measured, at `exp = -22` and
/// nowhere else in `-45..=38`. This form is the more accurate one, and `powf`
/// uses platform libm, so agreeing with
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
/// the difference is observable: at `x = 0.49999999999999994`  -  the largest
/// double below one half  -  the exact sum `1 - 2^-54` sits halfway between two
/// doubles, so it rounds to `1.0` and the cast yields 1 where `llround(x)`
/// yields 0. The fractional part is therefore compared against one half
/// directly. `x - trunc(x)` is exact for every finite `x`, which is what makes
/// that comparison exact.
///
/// Every double of magnitude `2^52` or more is already an integer, so above that
/// there is nothing to round and the cast  -  which saturates where C's `llround`
/// is undefined  -  returns false on its own. That also catches NaN, which fails both
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
/// `x` itself for the same reason  -  the `f32` sum rounds to `1.0` at
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

fn to_usize(value: u64) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_conversion| Error::ScaleOffsetValueTooLargeForPlatform {
        value,
        target: "usize",
    })
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
mod tests;
