//! Byte copies for native and little-endian numeric representations.

use alloc::vec::Vec;
#[cfg(target_endian = "little")]
use core::slice;
use core::{mem, ptr};

use crate::FormatError;

/// A fixed-width numeric primitive with a padding-free representation.
///
/// This trait is sealed to [`i8`], [`i16`], [`i32`], [`i64`], [`u8`], [`u16`], [`u32`], [`u64`],
/// [`f32`], and [`f64`]. Every bit pattern is valid for each of these types.
pub(crate) trait PrimitiveBytes: Copy + sealed::Sealed {
    /// Appends the little-endian bytes of `src`, preserving the existing bytes in `dst`.
    ///
    /// Copies the native representation on little-endian targets and encodes each element on
    /// big-endian targets.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds [`isize::MAX`] bytes.
    #[cfg(target_endian = "little")]
    fn append_le(src: &[Self], dst: &mut Vec<u8>) {
        // SAFETY: Sealing restricts `Self` to numeric primitives with initialized, padding-free
        // representations, and `size_of_val` covers exactly the live slice within one allocation.
        let bytes =
            unsafe { slice::from_raw_parts(src.as_ptr().cast::<u8>(), mem::size_of_val(src)) };
        dst.extend_from_slice(bytes);
    }

    /// Appends the little-endian bytes of `src`, preserving the existing bytes in `dst`.
    ///
    /// Copies the native representation on little-endian targets and encodes each element on
    /// big-endian targets.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds [`isize::MAX`] bytes.
    #[cfg(target_endian = "big")]
    fn append_le(src: &[Self], dst: &mut Vec<u8>);

    /// Appends `src` as native-endian values, preserving the existing elements of `dst`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DataSizeMismatch`] if `src` contains an incomplete element.
    /// On error, `dst` is unchanged.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds [`isize::MAX`] bytes.
    fn append_native(src: &[u8], dst: &mut Vec<Self>) -> Result<(), FormatError> {
        let remainder = src.len() % mem::size_of::<Self>();
        if remainder != 0 {
            return Err(FormatError::DataSizeMismatch {
                expected: src.len() - remainder,
                actual: src.len(),
            });
        }

        let count = src.len() / mem::size_of::<Self>();
        let len = dst.len();
        dst.reserve(count);
        // SAFETY: Sealing restricts `Self` to padding-free numeric primitives for which every bit
        // pattern is valid. The whole-element check and `reserve` provide `count` aligned slots after
        // the existing elements, and the disjoint source initializes those slots before `set_len`.
        unsafe {
            ptr::copy_nonoverlapping(
                src.as_ptr(),
                dst.spare_capacity_mut().as_mut_ptr().cast::<u8>(),
                src.len(),
            );
            dst.set_len(len + count);
        }
        Ok(())
    }
}

macro_rules! impl_primitive_bytes {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl sealed::Sealed for $ty {}
            impl PrimitiveBytes for $ty {
                #[cfg(target_endian = "big")]
                fn append_le(src: &[Self], dst: &mut Vec<u8>) {
                    dst.reserve(mem::size_of_val(src));
                    for &value in src {
                        dst.extend_from_slice(&value.to_le_bytes());
                    }
                }
            }
        )+
    };
}

impl_primitive_bytes!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

mod sealed {
    pub trait Sealed {}
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use rstest::rstest;

    use crate::primitive_bytes::PrimitiveBytes;
    use crate::type_builders::{self, AttrValue, DatasetBuilder};
    use crate::{AppendBuilder, EnumTypeBuilder};

    macro_rules! encode_test {
        ($name:ident, $ty:ty, $setter:ident, $append:ident, $attr:ident, $samples:expr) => {
            #[rstest]
            #[case::empty(0)]
            #[case::single(1)]
            #[case::multiple(17)]
            #[cfg_attr(not(miri), case::large(65_536))]
            fn $name(#[case] count: usize) {
                let samples: &[$ty] = &$samples;
                let values: Vec<_> = samples.iter().copied().cycle().take(count).collect();
                let expected: Vec<_> = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect();
                let mut bytes = vec![0xA5];
                <$ty>::append_le(&values, &mut bytes);
                assert_eq!(&bytes[1..], expected);
                assert_eq!(bytes[0], 0xA5);

                let mut dataset = DatasetBuilder::new("values");
                dataset.$setter(&values);
                let datatype = dataset.datatype.clone().unwrap();
                assert_eq!(dataset.data.as_ref().unwrap().as_ref().unwrap(), &expected);
                assert_eq!(dataset.shape, Some(vec![count as u64]));
                dataset.with_vlen_strings(&["previous"]);
                dataset.$setter(&values);
                assert!(dataset.vl_string_staging.is_none());
                assert_eq!(dataset.data.as_ref().unwrap().as_ref().unwrap(), &expected);
                dataset.with_path_references(&["target"]);
                dataset.$setter(&values);
                assert!(dataset.reference_targets.is_none());
                assert_eq!(dataset.data.as_ref().unwrap().as_ref().unwrap(), &expected);
                assert_eq!(dataset.datatype.as_ref(), Some(&datatype));
                assert_eq!(dataset.shape, Some(vec![count as u64]));

                let mut append = AppendBuilder::new();
                append.$append(&[]);
                values.chunks(7).for_each(|chunk| {
                    append.$append(chunk);
                });
                assert_eq!(append.raw(), expected);
                assert_eq!(append.elem_dt(), Some(&datatype));
                assert!(!append.dt_conflict());

                let attr =
                    type_builders::build_attr_message("values", &AttrValue::$attr(values)).unwrap();
                assert_eq!(attr.raw_data, expected);
                assert_eq!(attr.datatype, datatype);
                assert_eq!(attr.dataspace, type_builders::simple_1d(count as u64));
            }
        };
    }

    encode_test!(
        encode_i8,
        i8,
        with_i8_data,
        append_i8,
        I8Array,
        [i8::MIN, 0, i8::MAX]
    );
    encode_test!(
        encode_i16,
        i16,
        with_i16_data,
        append_i16,
        I16Array,
        [i16::MIN, 0, i16::MAX]
    );
    encode_test!(
        encode_i32,
        i32,
        with_i32_data,
        append_i32,
        I32Array,
        [i32::MIN, 0, i32::MAX]
    );
    encode_test!(
        encode_i64,
        i64,
        with_i64_data,
        append_i64,
        I64Array,
        [i64::MIN, 0, i64::MAX]
    );
    encode_test!(
        encode_u8,
        u8,
        with_u8_data,
        append_u8,
        U8Array,
        [0, 1, u8::MAX]
    );
    encode_test!(
        encode_u16,
        u16,
        with_u16_data,
        append_u16,
        U16Array,
        [0, 1, u16::MAX]
    );
    encode_test!(
        encode_u32,
        u32,
        with_u32_data,
        append_u32,
        U32Array,
        [0, 1, u32::MAX]
    );
    encode_test!(
        encode_u64,
        u64,
        with_u64_data,
        append_u64,
        U64Array,
        [0, 1, u64::MAX]
    );
    encode_test!(
        encode_f32,
        f32,
        with_f32_data,
        append_f32,
        F32Array,
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
    encode_test!(
        encode_f64,
        f64,
        with_f64_data,
        append_f64,
        F64Array,
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
    #[case::empty(vec![])]
    #[case::single(vec![i32::MIN])]
    #[case::extrema(vec![i32::MIN, 0, i32::MAX])]
    fn encode_i32_enum(#[case] values: Vec<i32>) {
        let datatype = EnumTypeBuilder::i32_based()
            .value("min", i32::MIN)
            .value("zero", 0)
            .value("max", i32::MAX)
            .build()
            .unwrap();
        let mut dataset = DatasetBuilder::new("enum");
        dataset.with_shape(&[values.len() as u64, 1]);
        dataset.with_vlen_strings(&["previous"]);
        dataset.with_enum_i32_data(datatype.clone(), &values);
        assert!(dataset.vl_string_staging.is_none());
        assert_eq!(dataset.datatype, Some(datatype));
        assert_eq!(dataset.shape, Some(vec![values.len() as u64, 1]));
        assert_eq!(
            dataset.data.unwrap().unwrap(),
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn typed_appends_preserve_datatype_conflicts() {
        let mut append = AppendBuilder::new();
        append.append_i32(&[i32::MIN]).append_u32(&[u32::MAX]);
        assert_eq!(append.elem_dt(), Some(&type_builders::make_i32_type()));
        assert!(append.dt_conflict());
        assert_eq!(
            append.raw(),
            [i32::MIN.to_le_bytes(), u32::MAX.to_le_bytes()].concat()
        );
    }
}
