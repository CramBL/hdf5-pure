use crate::{
    FormatError,
    datatype::{
        byte_order::FixedWidthByteOrder,
        layout::{StandardNumericLayout, StandardWidth},
    },
};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Decode standard fixed-width scalars and append their converted values.
///
/// `S` describes the scalar representation reconstructed from the bytes.
/// `T` is the caller's requested scalar type.
/// `C` describes the selected HDF5 conversion path.
///
/// TODO: A soft conversion can be represented by a different concrete `C`
/// which carries whatever datatype metadata its algorithm needs.
pub(crate) fn decode_fixed_width_into<const W: usize, S, T, C>(
    src: &[u8],
    order: FixedWidthByteOrder,
    conversion: C,
    dst: &mut Vec<T>,
) -> Result<(), FormatError>
where
    S: FromEndianBytes<W>,
    C: H5Conversion<S, T>,
{
    let (chunks, remainder) = src.as_chunks::<W>();

    if !remainder.is_empty() {
        return Err(FormatError::DataSizeMismatch {
            expected: src.len() - remainder.len(),
            actual: src.len(),
        });
    }

    dst.reserve(chunks.len());

    match order {
        FixedWidthByteOrder::LittleEndian => {
            for chunk in chunks {
                dst.push(conversion.convert(S::from_le_slice(chunk)));
            }
        }
        FixedWidthByteOrder::BigEndian => {
            for chunk in chunks {
                dst.push(conversion.convert(S::from_be_slice(chunk)));
            }
        }
    }

    Ok(())
}

pub(crate) fn decode_standard_fixed_point_into<T>(
    src: &[u8],
    signed: bool,
    StandardNumericLayout { width, order }: StandardNumericLayout,
    dst: &mut Vec<T>,
) -> Result<(), FormatError>
where
    T: StandardFixedPointReadTarget,
{
    match (signed, width) {
        (true, StandardWidth::OneByte) => {
            decode_fixed_width_into::<1, i8, T, T::FromI8>(src, order, T::FromI8::default(), dst)
        }
        (true, StandardWidth::TwoBytes) => {
            decode_fixed_width_into::<2, i16, T, T::FromI16>(src, order, T::FromI16::default(), dst)
        }
        (true, StandardWidth::FourBytes) => {
            decode_fixed_width_into::<4, i32, T, T::FromI32>(src, order, T::FromI32::default(), dst)
        }
        (true, StandardWidth::EightBytes) => {
            decode_fixed_width_into::<8, i64, T, T::FromI64>(src, order, T::FromI64::default(), dst)
        }

        (false, StandardWidth::OneByte) => {
            decode_fixed_width_into::<1, u8, T, T::FromU8>(src, order, T::FromU8::default(), dst)
        }
        (false, StandardWidth::TwoBytes) => {
            decode_fixed_width_into::<2, u16, T, T::FromU16>(src, order, T::FromU16::default(), dst)
        }
        (false, StandardWidth::FourBytes) => {
            decode_fixed_width_into::<4, u32, T, T::FromU32>(src, order, T::FromU32::default(), dst)
        }
        (false, StandardWidth::EightBytes) => {
            decode_fixed_width_into::<8, u64, T, T::FromU64>(src, order, T::FromU64::default(), dst)
        }
    }
}

pub(crate) trait SignedIntegerReadTarget: Sized {
    type FromI8: H5Conversion<i8, Self> + Default;
    type FromI16: H5Conversion<i16, Self> + Default;
    type FromI32: H5Conversion<i32, Self> + Default;
    type FromI64: H5Conversion<i64, Self> + Default;

    fn from_i64(value: i64) -> Self;
}

pub(crate) trait UnsignedIntegerReadTarget: Sized {
    type FromU8: H5Conversion<u8, Self> + Default;
    type FromU16: H5Conversion<u16, Self> + Default;
    type FromU32: H5Conversion<u32, Self> + Default;
    type FromU64: H5Conversion<u64, Self> + Default;

    fn from_u64(value: u64) -> Self;
}

pub(crate) trait StandardFixedPointReadTarget: Sized {
    type FromI8: H5Conversion<i8, Self> + Default;
    type FromI16: H5Conversion<i16, Self> + Default;
    type FromI32: H5Conversion<i32, Self> + Default;
    type FromI64: H5Conversion<i64, Self> + Default;

    type FromU8: H5Conversion<u8, Self> + Default;
    type FromU16: H5Conversion<u16, Self> + Default;
    type FromU32: H5Conversion<u32, Self> + Default;
    type FromU64: H5Conversion<u64, Self> + Default;
}

impl StandardFixedPointReadTarget for f32 {
    type FromI8 = HardConversion;
    type FromI16 = HardConversion;
    type FromI32 = HardConversion;
    type FromI64 = HardConversion;

    type FromU8 = HardConversion;
    type FromU16 = HardConversion;
    type FromU32 = HardConversion;
    type FromU64 = HardConversion;
}

impl StandardFixedPointReadTarget for f64 {
    type FromI8 = HardConversion;
    type FromI16 = HardConversion;
    type FromI32 = HardConversion;
    type FromI64 = HardConversion;

    type FromU8 = HardConversion;
    type FromU16 = HardConversion;
    type FromU32 = HardConversion;
    type FromU64 = HardConversion;
}

impl UnsignedIntegerReadTarget for u8 {
    type FromU8 = NoOpConversion;
    type FromU16 = HardConversion;
    type FromU32 = HardConversion;
    type FromU64 = HardConversion;

    fn from_u64(value: u64) -> Self {
        value as Self
    }
}
impl UnsignedIntegerReadTarget for u16 {
    type FromU8 = HardConversion;
    type FromU16 = NoOpConversion;
    type FromU32 = HardConversion;
    type FromU64 = HardConversion;

    fn from_u64(value: u64) -> Self {
        value as Self
    }
}
impl UnsignedIntegerReadTarget for u32 {
    type FromU8 = HardConversion;
    type FromU16 = HardConversion;
    type FromU32 = NoOpConversion;
    type FromU64 = HardConversion;

    fn from_u64(value: u64) -> Self {
        value as Self
    }
}
impl UnsignedIntegerReadTarget for u64 {
    type FromU8 = HardConversion;
    type FromU16 = HardConversion;
    type FromU32 = HardConversion;
    type FromU64 = NoOpConversion;

    fn from_u64(value: u64) -> Self {
        value as Self
    }
}

impl SignedIntegerReadTarget for i8 {
    type FromI8 = NoOpConversion;
    type FromI16 = HardConversion;
    type FromI32 = HardConversion;
    type FromI64 = HardConversion;

    fn from_i64(value: i64) -> Self {
        value as Self
    }
}

impl SignedIntegerReadTarget for i16 {
    type FromI8 = HardConversion;
    type FromI16 = NoOpConversion;
    type FromI32 = HardConversion;
    type FromI64 = HardConversion;

    fn from_i64(value: i64) -> Self {
        value as Self
    }
}

impl SignedIntegerReadTarget for i32 {
    type FromI8 = HardConversion;
    type FromI16 = HardConversion;
    type FromI32 = NoOpConversion;
    type FromI64 = HardConversion;

    fn from_i64(value: i64) -> Self {
        value as Self
    }
}

impl SignedIntegerReadTarget for i64 {
    type FromI8 = HardConversion;
    type FromI16 = HardConversion;
    type FromI32 = HardConversion;
    type FromI64 = NoOpConversion;

    fn from_i64(value: i64) -> Self {
        value
    }
}

/// A primitive scalar that can be reconstructed from exactly `W` bytes.
pub(crate) trait FromEndianBytes<const W: usize>: Sized {
    fn from_le_slice(slice: &[u8; W]) -> Self;
    fn from_be_slice(slice: &[u8; W]) -> Self;
}

macro_rules! impl_from_endian_bytes {
    ($ty:ty, $w:literal) => {
        impl FromEndianBytes<$w> for $ty {
            #[inline]
            fn from_le_slice(slice: &[u8; $w]) -> Self {
                <$ty>::from_le_bytes(*slice)
            }

            #[inline]
            fn from_be_slice(slice: &[u8; $w]) -> Self {
                <$ty>::from_be_bytes(*slice)
            }
        }
    };
}

impl_from_endian_bytes!(u8, 1);
impl_from_endian_bytes!(i8, 1);
impl_from_endian_bytes!(u16, 2);
impl_from_endian_bytes!(i16, 2);
impl_from_endian_bytes!(u32, 4);
impl_from_endian_bytes!(i32, 4);
impl_from_endian_bytes!(u64, 8);
impl_from_endian_bytes!(i64, 8);
impl_from_endian_bytes!(f32, 4);
impl_from_endian_bytes!(f64, 8);

/// HDF5's classification of a datatype conversion path.
#[allow(dead_code, reason = "for when we support soft conversions")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ConversionKind {
    /// Source and destination require no value conversion.
    NoOp,

    /// A native/compiler conversion.
    Hard,

    /// A conversion implemented by library logic.
    Soft,
}

/// Converts one already-decoded scalar into another.
pub(crate) trait H5Conversion<S, T> {
    #[allow(dead_code, reason = "for when we support soft conversions")]
    const KIND: ConversionKind;

    fn convert(&self, value: S) -> T;
}

/// Identity/no-op conversion.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NoOpConversion;

impl<T> H5Conversion<T, T> for NoOpConversion {
    const KIND: ConversionKind = ConversionKind::NoOp;

    #[inline]
    fn convert(&self, value: T) -> T {
        value
    }
}

/// Native/compiler conversion.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct HardConversion;

/// Implement hard conversions for which the standard library provides an
/// infallible `From` conversion.
macro_rules! impl_hard_from {
    ($src:ty => $($dst:ty),+ $(,)?) => {
        $(
            impl H5Conversion<$src, $dst> for HardConversion {
                const KIND: ConversionKind = ConversionKind::Hard;

                #[inline]
                fn convert(&self, value: $src) -> $dst {
                    value.into()
                }
            }
        )+
    };
}

// Signed integer widening and exact integer -> float conversions.
impl_hard_from!(i8 => i16, i32, i64, f32, f64);
impl_hard_from!(i16 => i32, i64, f32, f64);
impl_hard_from!(i32 => i64, f64);

// Unsigned integer widening and exact integer -> float conversions.
impl_hard_from!(u8 => u16, u32, u64, f32, f64);
impl_hard_from!(u16 => u32, u64, f32, f64);
impl_hard_from!(u32 => u64, f64);

// Exact floating-point widening.
impl_hard_from!(f32 => f64);

/// Implements a hard conversion as a native Rust `as` cast.
///
/// Each `$src`/`$dst` pair appears here because the HDF5 conversion rule for
/// that pair is the native `as` cast.
///
/// A `TryFrom` implementation would change the behavior of these narrowing
/// conversions. `TryFrom` rejects out-of-range values, while the HDF5 rule for
/// these pairs produces the value the native `as` cast yields.
///
/// # Warning
///
/// Adding a `$src`/`$dst` pair requires confirming that the HDF5 conversion
/// rule for that pair is the native `as` cast. The `as` cast truncates, wraps,
/// or changes sign for values outside `$dst`'s range, so a pair whose HDF5 rule
/// is a fallible conversion or a library algorithm must not use this macro.
macro_rules! impl_hard_cast {
    ($src:ty => $($dst:ty),+ $(,)?) => {
        $(
            impl H5Conversion<$src, $dst> for HardConversion {
                const KIND: ConversionKind = ConversionKind::Hard;

                #[inline]
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_possible_wrap,
                    clippy::cast_sign_loss,
                    clippy::cast_precision_loss,
                    reason = "explicit HDF5 hard conversion using native primitive cast semantics"
                )]
                fn convert(&self, value: $src) -> $dst {
                    value as $dst
                }
            }
        )+
    };
}

// Signed narrowing.
impl_hard_cast!(i16 => i8);
impl_hard_cast!(i32 => i8, i16);
impl_hard_cast!(i64 => i8, i16, i32);

// Unsigned narrowing.
impl_hard_cast!(u16 => u8);
impl_hard_cast!(u32 => u8, u16);
impl_hard_cast!(u64 => u8, u16, u32);

// Integer -> float conversions which are not lossless `From` conversions.
impl_hard_cast!(i32 => f32);
impl_hard_cast!(i64 => f32, f64);
impl_hard_cast!(u32 => f32);
impl_hard_cast!(u64 => f32, f64);

// Floating-point narrowing.
impl_hard_cast!(f64 => f32);
