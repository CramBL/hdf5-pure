#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

pub(crate) use hdf5_pure_filter::FillAvailability;
use hdf5_pure_filter::ScaleOffsetByteOrder;
pub(crate) use hdf5_pure_filter::ScaleOffsetFill;
pub(crate) use hdf5_pure_filter::ScaleOffsetType;

use crate::datatype::Datatype;
use crate::datatype::byte_order::DatatypeByteOrder;
use crate::datatype::layout::FixedPointLayout;
use crate::error::FormatError;
use crate::fill_value::FillPattern;

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

impl From<ScaleOffset> for hdf5_pure_filter::ScaleOffset {
    fn from(mode: ScaleOffset) -> Self {
        match mode {
            ScaleOffset::Integer(bits) => Self::Integer(bits),
            ScaleOffset::FloatDScale(decimals) => Self::FloatDScale(decimals),
        }
    }
}

impl From<hdf5_pure_filter::ScaleOffset> for ScaleOffset {
    fn from(mode: hdf5_pure_filter::ScaleOffset) -> Self {
        match mode {
            hdf5_pure_filter::ScaleOffset::Integer(bits) => Self::Integer(bits),
            hdf5_pure_filter::ScaleOffset::FloatDScale(decimals) => Self::FloatDScale(decimals),
        }
    }
}

pub(crate) fn build_cd_values(
    mode: ScaleOffset,
    ty: ScaleOffsetType,
    size: u32,
    nelmts: u32,
    fill: ScaleOffsetFill<'_>,
) -> Result<Vec<u32>, FormatError> {
    hdf5_pure_filter::build_scale_offset_cd_values(mode.into(), ty, size, nelmts, fill)
        .map_err(FormatError::from)
}

pub(crate) fn scale_offset_mode(cd_values: &[u32]) -> Option<(ScaleOffset, FillAvailability)> {
    hdf5_pure_filter::scale_offset_mode(cd_values).map(|(mode, fill)| (mode.into(), fill))
}

/// Extracts scalar facts from a datatype supported by the Scale-Offset filter.
///
/// Returns [`None`] for byte orders other than little or big endian, integer sizes outside
/// 1, 2, 4, or 8 bytes, floating-point sizes other than 4 or 8 bytes, and other datatype classes.
pub fn scale_offset_type_from_datatype(dt: &Datatype) -> Option<ScaleOffsetType> {
    let order = match dt {
        Datatype::FixedPoint { byte_order, .. } | Datatype::FloatingPoint { byte_order, .. } => {
            match byte_order {
                DatatypeByteOrder::LittleEndian => ScaleOffsetByteOrder::LittleEndian,
                DatatypeByteOrder::BigEndian => ScaleOffsetByteOrder::BigEndian,
                DatatypeByteOrder::Vax => return None,
            }
        }
        _ => return None,
    };
    match dt {
        Datatype::FixedPoint {
            size,
            layout: FixedPointLayout { signed, .. },
            ..
        } if matches!(*size, 1 | 2 | 4 | 8) => Some(ScaleOffsetType::integer(*signed, order)),
        Datatype::FloatingPoint { size, .. } if matches!(*size, 4 | 8) => {
            Some(ScaleOffsetType::floating(order))
        }
        _ => None,
    }
}

pub(crate) fn scale_offset_fill_with_value(
    availability: FillAvailability,
    fill: FillPattern<'_>,
) -> Result<ScaleOffsetFill<'_>, FormatError> {
    match availability {
        FillAvailability::Defined => Ok(ScaleOffsetFill::Defined(fill.element()?)),
        FillAvailability::Undefined => Ok(ScaleOffsetFill::Undefined),
    }
}

#[cfg(test)]
mod tests {
    use crate::datatype::layout::FloatingPointLayout;

    use super::*;

    #[test]
    fn scale_offset_type_from_datatype_classes() {
        let i32_ty = Datatype::FixedPoint {
            size: 4,
            byte_order: DatatypeByteOrder::LittleEndian,
            layout: FixedPointLayout {
                signed: true,
                bit_offset: 0,
                bit_precision: 32,
            },
        };
        let so = scale_offset_type_from_datatype(&i32_ty).unwrap();
        assert_eq!(
            so,
            ScaleOffsetType::integer(true, ScaleOffsetByteOrder::LittleEndian)
        );

        let f64_ty = Datatype::FloatingPoint {
            size: 8,
            byte_order: DatatypeByteOrder::BigEndian,
            layout: FloatingPointLayout {
                bit_offset: 0,
                bit_precision: 64,
                exponent_location: 52,
                exponent_size: 11,
                mantissa_location: 0,
                mantissa_size: 52,
                exponent_bias: 1023,
            },
        };
        let so = scale_offset_type_from_datatype(&f64_ty).unwrap();
        assert_eq!(
            so,
            ScaleOffsetType::floating(ScaleOffsetByteOrder::BigEndian)
        );
    }
}
