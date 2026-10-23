//! ZFP scalar types for dataset filter contexts.

#![cfg(feature = "zfp")]

use hdf5_pure_filter::__private::ZfpElementType;

/// The ZFP scalar type stored in a chunk context.
///
/// Without the feature, the alias is [`core::convert::Infallible`].
pub type ZfpElementTypeWhenEnabled = ZfpElementType;
/// Returns the ZFP scalar type for a supported fixed-point or floating-point datatype.
///
/// Returns `None` for datatypes other than 32-bit and 64-bit floats or signed integers.
pub fn zfp_element_type_from_datatype(
    dt: &crate::datatype::Datatype,
) -> Option<ZfpElementTypeWhenEnabled> {
    use crate::datatype::Datatype;
    match dt {
        Datatype::FloatingPoint(numeric) if numeric.size().get() == 4 => Some(ZfpElementType::F32),
        Datatype::FloatingPoint(numeric) if numeric.size().get() == 8 => Some(ZfpElementType::F64),
        Datatype::FixedPoint(numeric) if numeric.size().get() == 4 && numeric.layout().signed() => {
            Some(ZfpElementType::I32)
        }
        Datatype::FixedPoint(numeric) if numeric.size().get() == 8 && numeric.layout().signed() => {
            Some(ZfpElementType::I64)
        }
        _ => None,
    }
}
