#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::primitive_bytes::PrimitiveBytes;
use crate::{
    FormatError,
    datatype::{
        byte_order::FixedWidthByteOrder,
        layout::{StandardNumericLayout, StandardWidth},
    },
};

/// Appends standard fixed-width numeric values after conversion.
///
/// Copies native-endian bytes for [`NoOpConversion`] and decodes each element for other
/// conversions or byte orders. `S` is the stored primitive type, and `T` is the destination type.
///
/// # Errors
///
/// Returns [`FormatError::DataSizeMismatch`] if `src` contains an incomplete element.
/// On error, `dst` is unchanged.
///
/// # Panics
///
/// Panics if the new capacity exceeds [`isize::MAX`] bytes.
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

    let native_order = if cfg!(target_endian = "little") {
        FixedWidthByteOrder::LittleEndian
    } else {
        FixedWidthByteOrder::BigEndian
    };
    if order == native_order && conversion.append_native(src, dst)? {
        return Ok(());
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
    T: NumericReadTarget,
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

pub(crate) trait NumericReadTarget: Sized {
    type I8Conversion: H5Conversion<i8, Self> + Default;
    type I16Conversion: H5Conversion<i16, Self> + Default;
    type I32Conversion: H5Conversion<i32, Self> + Default;
    type I64Conversion: H5Conversion<i64, Self> + Default;

    type U8Conversion: H5Conversion<u8, Self> + Default;
    type U16Conversion: H5Conversion<u16, Self> + Default;
    type U32Conversion: H5Conversion<u32, Self> + Default;
    type U64Conversion: H5Conversion<u64, Self> + Default;

    type F32Conversion: H5Conversion<f32, Self> + Default;
    type F64Conversion: H5Conversion<f64, Self> + Default;
}

impl NumericReadTarget for i8 {
    type I8Conversion = NoOpConversion;
    type I16Conversion = SoftIntegerConversion;
    type I32Conversion = SoftIntegerConversion;
    type I64Conversion = SoftIntegerConversion;

    type U8Conversion = SoftIntegerConversion;
    type U16Conversion = SoftIntegerConversion;
    type U32Conversion = SoftIntegerConversion;
    type U64Conversion = SoftIntegerConversion;

    type F32Conversion = HardConversion;
    type F64Conversion = HardConversion;
}

impl NumericReadTarget for i16 {
    type I8Conversion = HardConversion;
    type I16Conversion = NoOpConversion;
    type I32Conversion = SoftIntegerConversion;
    type I64Conversion = SoftIntegerConversion;

    type U8Conversion = HardConversion;
    type U16Conversion = SoftIntegerConversion;
    type U32Conversion = SoftIntegerConversion;
    type U64Conversion = SoftIntegerConversion;

    type F32Conversion = HardConversion;
    type F64Conversion = HardConversion;
}

impl NumericReadTarget for i32 {
    type I8Conversion = HardConversion;
    type I16Conversion = HardConversion;
    type I32Conversion = NoOpConversion;
    type I64Conversion = SoftIntegerConversion;

    type U8Conversion = HardConversion;
    type U16Conversion = HardConversion;
    type U32Conversion = SoftIntegerConversion;
    type U64Conversion = SoftIntegerConversion;

    type F32Conversion = HardConversion;
    type F64Conversion = HardConversion;
}

impl NumericReadTarget for i64 {
    type I8Conversion = HardConversion;
    type I16Conversion = HardConversion;
    type I32Conversion = HardConversion;
    type I64Conversion = NoOpConversion;

    type U8Conversion = HardConversion;
    type U16Conversion = HardConversion;
    type U32Conversion = HardConversion;
    type U64Conversion = SoftIntegerConversion;

    type F32Conversion = HardConversion;
    type F64Conversion = HardConversion;
}

impl NumericReadTarget for u8 {
    type I8Conversion = SoftIntegerConversion;
    type I16Conversion = SoftIntegerConversion;
    type I32Conversion = SoftIntegerConversion;
    type I64Conversion = SoftIntegerConversion;

    type U8Conversion = NoOpConversion;
    type U16Conversion = SoftIntegerConversion;
    type U32Conversion = SoftIntegerConversion;
    type U64Conversion = SoftIntegerConversion;

    type F32Conversion = HardConversion;
    type F64Conversion = HardConversion;
}

impl NumericReadTarget for u16 {
    type I8Conversion = SoftIntegerConversion;
    type I16Conversion = SoftIntegerConversion;
    type I32Conversion = SoftIntegerConversion;
    type I64Conversion = SoftIntegerConversion;

    type U8Conversion = HardConversion;
    type U16Conversion = NoOpConversion;
    type U32Conversion = SoftIntegerConversion;
    type U64Conversion = SoftIntegerConversion;

    type F32Conversion = HardConversion;
    type F64Conversion = HardConversion;
}

impl NumericReadTarget for u32 {
    type I8Conversion = SoftIntegerConversion;
    type I16Conversion = SoftIntegerConversion;
    type I32Conversion = SoftIntegerConversion;
    type I64Conversion = SoftIntegerConversion;

    type U8Conversion = HardConversion;
    type U16Conversion = HardConversion;
    type U32Conversion = NoOpConversion;
    type U64Conversion = SoftIntegerConversion;

    type F32Conversion = HardConversion;
    type F64Conversion = HardConversion;
}

impl NumericReadTarget for u64 {
    type I8Conversion = SoftIntegerConversion;
    type I16Conversion = SoftIntegerConversion;
    type I32Conversion = SoftIntegerConversion;
    type I64Conversion = SoftIntegerConversion;

    type U8Conversion = HardConversion;
    type U16Conversion = HardConversion;
    type U32Conversion = HardConversion;
    type U64Conversion = NoOpConversion;

    type F32Conversion = HardConversion;
    type F64Conversion = HardConversion;
}

impl NumericReadTarget for f32 {
    type I8Conversion = HardConversion;
    type I16Conversion = HardConversion;
    type I32Conversion = HardConversion;
    type I64Conversion = HardConversion;

    type U8Conversion = HardConversion;
    type U16Conversion = HardConversion;
    type U32Conversion = HardConversion;
    type U64Conversion = HardConversion;

    type F32Conversion = NoOpConversion;
    type F64Conversion = HardConversion;
}

impl NumericReadTarget for f64 {
    type I8Conversion = HardConversion;
    type I16Conversion = HardConversion;
    type I32Conversion = HardConversion;
    type I64Conversion = HardConversion;

    type U8Conversion = HardConversion;
    type U16Conversion = HardConversion;
    type U32Conversion = HardConversion;
    type U64Conversion = HardConversion;

    type F32Conversion = HardConversion;
    type F64Conversion = NoOpConversion;
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

/// Converts numeric values from `S` to `T`.
pub(crate) trait H5Conversion<S, T> {
    #[allow(dead_code, reason = "for when we support soft conversions")]
    const KIND: ConversionKind;

    fn convert(&self, value: S) -> T;

    /// Appends native-endian values when this conversion supports a byte copy.
    ///
    /// Returns `Ok(true)` after appending the values, or `Ok(false)` with `dst` unchanged
    /// when scalar conversion is required.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DataSizeMismatch`] if the byte-copy implementation receives
    /// an incomplete element.
    ///
    /// # Panics
    ///
    /// Panics if a byte copy requires a new capacity exceeding [`isize::MAX`] bytes.
    fn append_native(&self, _src: &[u8], _dst: &mut Vec<T>) -> Result<bool, FormatError> {
        Ok(false)
    }
}

/// Preserves each primitive value and its native byte representation.
///
/// Implements [`H5Conversion`] only when the source and destination types are identical.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NoOpConversion;

impl<T: PrimitiveBytes> H5Conversion<T, T> for NoOpConversion {
    const KIND: ConversionKind = ConversionKind::NoOp;

    #[inline]
    fn convert(&self, value: T) -> T {
        value
    }

    fn append_native(&self, src: &[u8], dst: &mut Vec<T>) -> Result<bool, FormatError> {
        T::append_native(src, dst)?;
        Ok(true)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SoftIntegerConversion;

macro_rules! impl_soft_from_signed {
    ($src:ty => $($dst:ty),+ $(,)?) => {
        $(
            impl H5Conversion<$src, $dst> for SoftIntegerConversion {
                const KIND: ConversionKind = ConversionKind::Soft;

                #[inline]
                fn convert(&self, value: $src) -> $dst {
                    match <$dst>::try_from(value) {
                        Ok(value) => value,
                        Err(_) if value < 0 => <$dst>::MIN,
                        Err(_) => <$dst>::MAX,
                    }
                }
            }
        )+
    };
}

macro_rules! impl_soft_from_unsigned {
    ($src:ty => $($dst:ty),+ $(,)?) => {
        $(
            impl H5Conversion<$src, $dst> for SoftIntegerConversion {
                const KIND: ConversionKind = ConversionKind::Soft;

                #[inline]
                fn convert(&self, value: $src) -> $dst {
                    <$dst>::try_from(value).unwrap_or(<$dst>::MAX)
                }
            }
        )+
    };
}

impl_soft_from_signed!(i8  => u8, u16, u32, u64);
impl_soft_from_signed!(i16 => i8, u8, u16, u32, u64);
impl_soft_from_signed!(i32 => i8, i16, u8, u16, u32, u64);
impl_soft_from_signed!(i64 => i8, i16, i32, u8, u16, u32, u64);

impl_soft_from_unsigned!(u8  => i8);
impl_soft_from_unsigned!(u16 => u8, i8, i16);
impl_soft_from_unsigned!(u32 => u8, u16, i8, i16, i32);
impl_soft_from_unsigned!(u64 => u8, u16, u32, i8, i16, i32, i64);

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
impl_hard_from!(i8  => i16, i32, i64, f32, f64);
impl_hard_from!(i16 => i32, i64, f32, f64);
impl_hard_from!(i32 => i64, f64);

// Unsigned integer widening and exact integer -> float conversions.
impl_hard_from!(u8  => u16, u32, u64, f32, f64);
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

// Floating-point -> signed integer.
impl_hard_cast!(f32 => i8, i16, i32, i64);
impl_hard_cast!(f64 => i8, i16, i32, i64);

// Floating-point -> unsigned integer.
impl_hard_cast!(f32 => u8, u16, u32, u64);
impl_hard_cast!(f64 => u8, u16, u32, u64);

// Floating-point narrowing.
impl_hard_cast!(f64 => f32);

// Unsigned integer -> signed integer.
impl_hard_cast!(u8  => i16, i32, i64);
impl_hard_cast!(u16 => i32, i64);
impl_hard_cast!(u32 => i64);

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;
    use core::mem;

    use rstest::rstest;

    use super::{ConversionKind, FromEndianBytes, H5Conversion, NoOpConversion};
    use crate::FormatError;
    use crate::datatype::byte_order::FixedWidthByteOrder;
    use crate::primitive_bytes::PrimitiveBytes;

    macro_rules! exact_decode_test {
        ($name:ident, $ty:ty, $samples:expr) => {
            #[rstest]
            #[case::empty(0)]
            #[case::single(1)]
            #[case::large(257)]
            fn $name(
                #[case] count: usize,
                #[values(false, true)] big_endian: bool,
                #[values(false, true)] spare_capacity: bool,
            ) {
                let samples: &[$ty] = &$samples;
                let values: Vec<_> = samples.iter().copied().cycle().take(count).collect();
                let raw: Vec<_> = values
                    .iter()
                    .flat_map(|value| {
                        if big_endian {
                            value.to_be_bytes()
                        } else {
                            value.to_le_bytes()
                        }
                    })
                    .collect();
                let mut dst = vec![samples[0]].into_boxed_slice().into_vec();
                if spare_capacity {
                    dst.reserve(count);
                }
                let order = if big_endian {
                    FixedWidthByteOrder::BigEndian
                } else {
                    FixedWidthByteOrder::LittleEndian
                };
                super::decode_fixed_width_into::<{ mem::size_of::<$ty>() }, $ty, $ty, _>(
                    &raw,
                    order,
                    NoOpConversion,
                    &mut dst,
                )
                .unwrap();
                let expected: Vec<_> = core::iter::once(samples[0])
                    .chain(values)
                    .flat_map(<$ty>::to_ne_bytes)
                    .collect();
                assert_eq!(
                    dst.into_iter()
                        .flat_map(<$ty>::to_ne_bytes)
                        .collect::<Vec<_>>(),
                    expected,
                );
            }
        };
    }

    exact_decode_test!(exact_i8, i8, [i8::MIN, 0, i8::MAX]);
    exact_decode_test!(exact_i16, i16, [i16::MIN, 0, i16::MAX]);
    exact_decode_test!(exact_i32, i32, [i32::MIN, 0, i32::MAX]);
    exact_decode_test!(exact_i64, i64, [i64::MIN, 0, i64::MAX]);
    exact_decode_test!(exact_u8, u8, [0, 1, u8::MAX]);
    exact_decode_test!(exact_u16, u16, [0, 1, u16::MAX]);
    exact_decode_test!(exact_u32, u32, [0, 1, u32::MAX]);
    exact_decode_test!(exact_u64, u64, [0, 1, u64::MAX]);
    exact_decode_test!(
        exact_f32,
        f32,
        [
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(1),
            f32::from_bits(0x7FC0_1234),
            f32::from_bits(0x7F80_1234),
            f32::from_bits(0xFFC0_5678),
        ]
    );
    exact_decode_test!(
        exact_f64,
        f64,
        [
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(1),
            f64::from_bits(0x7FF8_0000_0000_1234),
            f64::from_bits(0x7FF0_0000_0000_1234),
            f64::from_bits(0xFFF8_0000_0000_5678),
        ]
    );

    #[rstest]
    #[case::i16(incomplete::<2, i16>)]
    #[case::i32(incomplete::<4, i32>)]
    #[case::i64(incomplete::<8, i64>)]
    #[case::u16(incomplete::<2, u16>)]
    #[case::u32(incomplete::<4, u32>)]
    #[case::u64(incomplete::<8, u64>)]
    #[case::f32(incomplete::<4, f32>)]
    #[case::f64(incomplete::<8, f64>)]
    fn rejects_incomplete_elements(
        #[case] decode: fn(FixedWidthByteOrder),
        #[values(FixedWidthByteOrder::LittleEndian, FixedWidthByteOrder::BigEndian)]
        order: FixedWidthByteOrder,
    ) {
        decode(order);
    }

    fn incomplete<const W: usize, T>(order: FixedWidthByteOrder)
    where
        T: FromEndianBytes<W> + PrimitiveBytes + core::fmt::Debug + PartialEq,
    {
        let raw = vec![0; W + 1];
        let initial = T::from_le_slice(&[0; W]);
        let mut dst = vec![initial];
        let error =
            super::decode_fixed_width_into::<W, T, T, _>(&raw, order, NoOpConversion, &mut dst)
                .unwrap_err();
        assert_eq!(
            error,
            FormatError::DataSizeMismatch {
                expected: W,
                actual: W + 1
            }
        );
        assert_eq!(dst, [initial]);
        assert_eq!(T::append_native(&raw, &mut dst).unwrap_err(), error);
        assert_eq!(dst, [initial]);
    }

    #[test]
    fn converts_distinct_types_with_noop_discriminant() {
        struct DifferentTypes;

        impl H5Conversion<i32, u32> for DifferentTypes {
            const KIND: ConversionKind = ConversionKind::NoOp;

            fn convert(&self, _value: i32) -> u32 {
                7
            }
        }

        let mut dst = vec![5];
        super::decode_fixed_width_into::<4, i32, u32, _>(
            &(-1i32).to_ne_bytes(),
            if cfg!(target_endian = "little") {
                FixedWidthByteOrder::LittleEndian
            } else {
                FixedWidthByteOrder::BigEndian
            },
            DifferentTypes,
            &mut dst,
        )
        .unwrap();
        assert_eq!(dst, [5, 7]);
    }
}
