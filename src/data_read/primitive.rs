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
    T: FixedPointReadTarget,
{
    match (signed, width) {
        (true, StandardWidth::OneByte) => decode_fixed_width_into::<1, i8, T, T::I8Conversion>(
            src,
            order,
            T::I8Conversion::default(),
            dst,
        ),
        (true, StandardWidth::TwoBytes) => decode_fixed_width_into::<2, i16, T, T::I16Conversion>(
            src,
            order,
            T::I16Conversion::default(),
            dst,
        ),
        (true, StandardWidth::FourBytes) => decode_fixed_width_into::<4, i32, T, T::I32Conversion>(
            src,
            order,
            T::I32Conversion::default(),
            dst,
        ),
        (true, StandardWidth::EightBytes) => {
            decode_fixed_width_into::<8, i64, T, T::I64Conversion>(
                src,
                order,
                T::I64Conversion::default(),
                dst,
            )
        }

        (false, StandardWidth::OneByte) => decode_fixed_width_into::<1, u8, T, T::U8Conversion>(
            src,
            order,
            T::U8Conversion::default(),
            dst,
        ),
        (false, StandardWidth::TwoBytes) => decode_fixed_width_into::<2, u16, T, T::U16Conversion>(
            src,
            order,
            T::U16Conversion::default(),
            dst,
        ),
        (false, StandardWidth::FourBytes) => {
            decode_fixed_width_into::<4, u32, T, T::U32Conversion>(
                src,
                order,
                T::U32Conversion::default(),
                dst,
            )
        }
        (false, StandardWidth::EightBytes) => {
            decode_fixed_width_into::<8, u64, T, T::U64Conversion>(
                src,
                order,
                T::U64Conversion::default(),
                dst,
            )
        }
    }
}

pub(crate) trait FixedPointReadTarget: Sized {
    type I8Conversion: H5Conversion<i8, Self> + Default;
    type I16Conversion: H5Conversion<i16, Self> + Default;
    type I32Conversion: H5Conversion<i32, Self> + Default;
    type I64Conversion: H5Conversion<i64, Self> + Default;

    type U8Conversion: H5Conversion<u8, Self> + Default;
    type U16Conversion: H5Conversion<u16, Self> + Default;
    type U32Conversion: H5Conversion<u32, Self> + Default;
    type U64Conversion: H5Conversion<u64, Self> + Default;
}

impl FixedPointReadTarget for i8 {
    type I8Conversion = NoOpConversion;
    type I16Conversion = HardConversion;
    type I32Conversion = HardConversion;
    type I64Conversion = HardConversion;

    type U8Conversion = HardConversion;
    type U16Conversion = HardConversion;
    type U32Conversion = HardConversion;
    type U64Conversion = HardConversion;
}

impl FixedPointReadTarget for i16 {
    type I8Conversion = HardConversion;
    type I16Conversion = NoOpConversion;
    type I32Conversion = HardConversion;
    type I64Conversion = HardConversion;

    type U8Conversion = HardConversion;
    type U16Conversion = HardConversion;
    type U32Conversion = HardConversion;
    type U64Conversion = HardConversion;
}

impl FixedPointReadTarget for i32 {
    type I8Conversion = HardConversion;
    type I16Conversion = HardConversion;
    type I32Conversion = NoOpConversion;
    type I64Conversion = HardConversion;

    type U8Conversion = HardConversion;
    type U16Conversion = HardConversion;
    type U32Conversion = HardConversion;
    type U64Conversion = HardConversion;
}

impl FixedPointReadTarget for i64 {
    type I8Conversion = HardConversion;
    type I16Conversion = HardConversion;
    type I32Conversion = HardConversion;
    type I64Conversion = NoOpConversion;

    type U8Conversion = HardConversion;
    type U16Conversion = HardConversion;
    type U32Conversion = HardConversion;
    type U64Conversion = HardConversion;
}

impl FixedPointReadTarget for u8 {
    type I8Conversion = HardConversion;
    type I16Conversion = HardConversion;
    type I32Conversion = HardConversion;
    type I64Conversion = HardConversion;

    type U8Conversion = NoOpConversion;
    type U16Conversion = HardConversion;
    type U32Conversion = HardConversion;
    type U64Conversion = HardConversion;
}

impl FixedPointReadTarget for u16 {
    type I8Conversion = HardConversion;
    type I16Conversion = HardConversion;
    type I32Conversion = HardConversion;
    type I64Conversion = HardConversion;

    type U8Conversion = HardConversion;
    type U16Conversion = NoOpConversion;
    type U32Conversion = HardConversion;
    type U64Conversion = HardConversion;
}

impl FixedPointReadTarget for u32 {
    type I8Conversion = HardConversion;
    type I16Conversion = HardConversion;
    type I32Conversion = HardConversion;
    type I64Conversion = HardConversion;

    type U8Conversion = HardConversion;
    type U16Conversion = HardConversion;
    type U32Conversion = NoOpConversion;
    type U64Conversion = HardConversion;
}

impl FixedPointReadTarget for u64 {
    type I8Conversion = HardConversion;
    type I16Conversion = HardConversion;
    type I32Conversion = HardConversion;
    type I64Conversion = HardConversion;

    type U8Conversion = HardConversion;
    type U16Conversion = HardConversion;
    type U32Conversion = HardConversion;
    type U64Conversion = NoOpConversion;
}

impl FixedPointReadTarget for f32 {
    type I8Conversion = HardConversion;
    type I16Conversion = HardConversion;
    type I32Conversion = HardConversion;
    type I64Conversion = HardConversion;

    type U8Conversion = HardConversion;
    type U16Conversion = HardConversion;
    type U32Conversion = HardConversion;
    type U64Conversion = HardConversion;
}

impl FixedPointReadTarget for f64 {
    type I8Conversion = HardConversion;
    type I16Conversion = HardConversion;
    type I32Conversion = HardConversion;
    type I64Conversion = HardConversion;

    type U8Conversion = HardConversion;
    type U16Conversion = HardConversion;
    type U32Conversion = HardConversion;
    type U64Conversion = HardConversion;
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

// Signed integer -> unsigned integer.
impl_hard_cast!(i8 => u8, u16, u32, u64);
impl_hard_cast!(i16 => u8, u16, u32, u64);
impl_hard_cast!(i32 => u8, u16, u32, u64);
impl_hard_cast!(i64 => u8, u16, u32, u64);

// Unsigned integer -> signed integer.
impl_hard_cast!(u8 => i8, i16, i32, i64);
impl_hard_cast!(u16 => i8, i16, i32, i64);
impl_hard_cast!(u32 => i8, i16, i32, i64);
impl_hard_cast!(u64 => i8, i16, i32, i64);
