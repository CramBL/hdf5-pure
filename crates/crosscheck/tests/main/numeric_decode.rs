#![cfg(feature = "hdf5")]

use hdf5::Datatype as CDatatype;
use hdf5_pure::{
    Datatype, DatatypeByteOrder, Error, File, FloatingPointLayout, FormatError, H5Element,
    MantissaNormalization,
};
use hdf5_pure::{EnumTypeBuilder, FileBuilder, LibVer};
use hdf5_pure::{FixedPointType, FloatingPointType};
use hdf5_sys::h5t::{self, H5T_norm_t, H5T_order_t};
use rstest::rstest;
use test_util_hdf5::numeric;
use test_util_hdf5::numeric::ByteOrder;

#[rstest]
#[case::exact(42, 42, 42)]
#[case::narrowing(300, u8::MAX, i8::MAX)]
fn reads_c_scalar_fill(
    #[values(false, true)] big_endian: bool,
    #[values(1, 600_003)] count: usize,
    #[case] value: u16,
    #[case] unsigned: u8,
    #[case] signed: i8,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("scalar_fill.h5");
    {
        let datatype = CDatatype::from_type::<u16>().unwrap();
        let status = hdf5::sync::sync(|| {
            // SAFETY: The library lock and live transient integer datatype cover the call.
            unsafe { h5t::H5Tset_order(datatype.id(), numeric::byte_order(big_endian)) }
        });
        assert_eq!(status, 0);
        let reference = hdf5::File::create(&path).unwrap();
        let dataset = reference
            .new_dataset_builder()
            .empty_as(&datatype)
            .fill_value(value)
            .shape([count])
            .create("values")
            .unwrap();
        assert_eq!(
            dataset.dcpl().unwrap().get_fill_value_as::<u16>().unwrap(),
            Some(value)
        );
        assert_eq!(dataset.read_raw::<u16>().unwrap(), vec![value; count]);
    }
    let file = File::open(&path).unwrap();
    let dataset = file.dataset("values").unwrap();
    assert_eq!(dataset.fill_value::<u8>().unwrap(), Some(unsigned));
    assert_eq!(dataset.fill_value::<i8>().unwrap(), Some(signed));
    assert_eq!(dataset.read_u8().unwrap(), vec![unsigned; count]);
    assert_eq!(dataset.read_i8().unwrap(), vec![signed; count]);
    assert_eq!(dataset.read_u8_rows(0, 1).unwrap(), [unsigned]);
    assert_eq!(dataset.read_i8_region(&[0], &[1]).unwrap(), [signed]);
}

#[test]
fn c_reads_i32_enum_extrema() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("enum.h5");
    let datatype = EnumTypeBuilder::i32_based()
        .value("min", i32::MIN)
        .value("zero", 0)
        .value("max", i32::MAX)
        .build()
        .unwrap();
    let values = [i32::MIN, 0, i32::MAX];
    let mut builder = FileBuilder::new();
    builder.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    builder
        .create_dataset("values")
        .with_enum_i32_data(datatype, &values);
    builder.write(&path).unwrap();
    let reference = hdf5::File::open(&path).unwrap();
    assert_eq!(
        reference
            .dataset("values")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        values
    );
}

#[rstest]
fn reads_c_standard_i32(#[values(false, true)] big_endian: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("standard.h5");
    let order = if big_endian {
        ByteOrder::BigEndian
    } else {
        ByteOrder::LittleEndian
    };
    let datatype = order.i32_datatype();
    let values = [i32::MIN, 0, i32::MAX];
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
    numeric::write_stored(&path, &datatype, &raw, values.len());
    let reference = hdf5::File::open(&path).unwrap().dataset("values").unwrap();
    assert_eq!(reference.read_raw::<i32>().unwrap(), values);
    drop(reference);
    let dataset = File::open(&path).unwrap().dataset("values").unwrap();
    assert_eq!(dataset.read_i32().unwrap(), values);
    assert_eq!(dataset.read_f64().unwrap(), values.map(f64::from));
    let stored = dataset.datatype().unwrap();
    assert_eq!(u8::convert_raw(&raw, &stored).unwrap(), [0, 0, u8::MAX]);
    assert_eq!(
        i8::convert_raw(&raw, &stored).unwrap(),
        [i8::MIN, 0, i8::MAX]
    );
    assert_eq!(dataset.read_u8().unwrap(), [0, 0, u8::MAX]);
    assert_eq!(dataset.read_i8().unwrap(), [i8::MIN, 0, i8::MAX]);
}

#[rstest]
fn reads_c_offset_integers(
    #[values(false, true)] enumeration: bool,
    #[values(false, true)] big_endian: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("offset.h5");
    let datatype = CDatatype::from_type::<u8>().unwrap();
    hdf5::sync::sync(|| {
        // SAFETY: The lock serializes access and `datatype` owns a live transient integer type.
        unsafe {
            assert_eq!(h5t::H5Tset_precision(datatype.id(), 4), 0);
            assert_eq!(h5t::H5Tset_offset(datatype.id(), 4), 0);
            assert_eq!(
                h5t::H5Tset_order(datatype.id(), numeric::byte_order(big_endian)),
                0
            );
        }
    });
    let datatype = if enumeration {
        hdf5::sync::sync(|| {
            // SAFETY: The lock serializes access and the live integer type is a valid enum base.
            let id = unsafe { h5t::H5Tenum_create(datatype.id()) };
            assert!(id >= 0);
            // SAFETY: `id` is the newly created datatype ID and ownership transfers to its handle.
            let enumeration = unsafe { hdf5::from_id::<CDatatype>(id) }.unwrap();
            // SAFETY: Both names are NUL-terminated and both values have the base type's one-byte
            // width. The live enum handle and lock cover both calls.
            unsafe {
                assert_eq!(
                    h5t::H5Tenum_insert(
                        enumeration.id(),
                        c"eleven".as_ptr(),
                        [0xB0u8].as_ptr().cast()
                    ),
                    0
                );
                assert_eq!(
                    h5t::H5Tenum_insert(
                        enumeration.id(),
                        c"three".as_ptr(),
                        [0x30u8].as_ptr().cast()
                    ),
                    0
                );
            }
            enumeration
        })
    } else {
        datatype
    };
    numeric::write_stored(&path, &datatype, &[0xB0, 0x30], 2);
    let reference = hdf5::File::open(&path).unwrap().dataset("values").unwrap();
    assert_eq!(reference.read_raw::<u64>().unwrap(), vec![11, 3]);
    drop(reference);
    let dataset = File::open(&path).unwrap().dataset("values").unwrap();
    assert_eq!(dataset.read_raw().unwrap(), vec![0xB0, 0x30]);
    assert_eq!(dataset.read_f64().unwrap(), vec![11.0, 3.0]);
    assert_eq!(dataset.read_f32().unwrap(), vec![11.0, 3.0]);
    assert_eq!(dataset.read_i64().unwrap(), vec![11, 3]);
    assert_eq!(dataset.read_u64().unwrap(), vec![11, 3]);
    let stored = dataset.datatype().unwrap();
    assert_eq!(u8::convert_raw(&[0xB0, 0x30], &stored).unwrap(), [11, 3]);
    assert_eq!(i8::convert_raw(&[0xB0, 0x30], &stored).unwrap(), [11, 3]);
    let base = match &stored {
        Datatype::Enumeration { base_type, .. } => base_type.as_ref(),
        other => other,
    };
    assert_eq!(
        *base,
        Datatype::FixedPoint(FixedPointType::new(1, pure_order(big_endian), false, 4, 4).unwrap())
    );
}

#[rstest]
#[case(1, 4, 4)]
#[case(3, 7, 13)]
#[case(8, 60, 4)]
fn reads_c_signed_bit_ranges(
    #[case] size: u32,
    #[case] offset: u16,
    #[case] precision: u16,
    #[values(false, true)] big_endian: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("signed-range.h5");
    let datatype = CDatatype::from_type::<i64>().unwrap();
    hdf5::sync::sync(|| {
        // SAFETY: The lock serializes access and `datatype` owns a live transient integer type.
        unsafe {
            assert_eq!(h5t::H5Tset_precision(datatype.id(), precision.into()), 0);
            assert_eq!(h5t::H5Tset_offset(datatype.id(), offset.into()), 0);
            assert_eq!(h5t::H5Tset_size(datatype.id(), size as usize), 0);
            assert_eq!(
                h5t::H5Tset_order(datatype.id(), numeric::byte_order(big_endian)),
                0
            );
        }
    });
    let raw = [-5i64, 7]
        .into_iter()
        .flat_map(|value| {
            let word = (u64::from_ne_bytes(value.to_ne_bytes()) & (u64::MAX >> (64 - precision)))
                << offset;
            if big_endian {
                word.to_be_bytes()[8 - size as usize..].to_vec()
            } else {
                word.to_le_bytes()[..size as usize].to_vec()
            }
        })
        .collect::<Vec<_>>();
    numeric::write_stored(&path, &datatype, &raw, 2);
    let reference = hdf5::File::open(&path).unwrap().dataset("values").unwrap();
    assert_eq!(reference.read_raw::<i64>().unwrap(), vec![-5, 7]);
    drop(reference);
    let dataset = File::open(&path).unwrap().dataset("values").unwrap();
    let stored = Datatype::FixedPoint(
        FixedPointType::new(size, pure_order(big_endian), true, offset, precision).unwrap(),
    );
    assert_eq!(dataset.datatype().unwrap(), stored);
    assert_eq!(dataset.read_raw().unwrap(), raw);
    assert_eq!(i8::convert_raw(&raw, &stored).unwrap(), [-5, 7]);
    assert_eq!(u8::convert_raw(&raw, &stored).unwrap(), [0, 7]);
    assert_eq!(dataset.read_i64().unwrap(), vec![-5, 7]);
    assert_eq!(dataset.read_f64().unwrap(), vec![-5.0, 7.0]);
    let pure_path = dir.path().join("pure-signed-range.h5");
    let mut builder = FileBuilder::new();
    builder.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    builder
        .create_dataset("values")
        .with_raw_data(stored, raw, 2)
        .with_shape(&[2]);
    builder.write(&pure_path).unwrap();
    let reference = hdf5::File::open(&pure_path)
        .unwrap()
        .dataset("values")
        .unwrap();
    assert_eq!(reference.dtype().unwrap(), datatype);
    assert_eq!(reference.read_raw::<i64>().unwrap(), vec![-5, 7]);
}

#[rstest]
#[case(FloatLayout::Bias)]
#[case(FloatLayout::LeadingOne)]
#[case(FloatLayout::Sign)]
#[case(FloatLayout::Standard)]
#[case(FloatLayout::Unnormalized)]
#[case(FloatLayout::Vax)]
fn reads_only_supported_c_floats(
    #[case] encoding: FloatLayout,
    #[values(false, true)] wide: bool,
    #[values(false, true)] big_endian: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("float.h5");
    let datatype = if wide {
        CDatatype::from_type::<f64>()
    } else {
        CDatatype::from_type::<f32>()
    }
    .unwrap();
    let layout = if wide {
        FloatingPointLayout::IEEE754_BINARY64
    } else {
        FloatingPointLayout::IEEE754_BINARY32
    };
    let mut exponent_bias = layout.exponent_bias();
    let mut normalization = layout.mantissa_normalization();
    let mut sign_location = layout.sign_location();
    let mut mantissa_location = layout.mantissa_location();
    let mut mantissa_size = layout.mantissa_size();
    let order = if matches!(encoding, FloatLayout::Vax) {
        H5T_order_t::H5T_ORDER_VAX
    } else {
        numeric::byte_order(big_endian)
    };
    hdf5::sync::sync(|| {
        // SAFETY: The lock and live transient floating-point handle cover each setter. Each
        // field position and size fits within the type. The fields do not overlap.
        unsafe {
            assert_eq!(h5t::H5Tset_order(datatype.id(), order), 0);
            match encoding {
                FloatLayout::Bias => {
                    exponent_bias += 1;
                    assert_eq!(h5t::H5Tset_ebias(datatype.id(), exponent_bias as usize), 0);
                }
                FloatLayout::LeadingOne => {
                    normalization = MantissaNormalization::LeadingOne;
                    assert_eq!(
                        h5t::H5Tset_norm(datatype.id(), H5T_norm_t::H5T_NORM_MSBSET),
                        0
                    );
                }
                FloatLayout::Sign => {
                    sign_location = 0;
                    mantissa_location = 1;
                    mantissa_size -= 1;
                    assert_eq!(
                        h5t::H5Tset_fields(
                            datatype.id(),
                            0,
                            layout.exponent_location().into(),
                            layout.exponent_size().into(),
                            1,
                            mantissa_size.into()
                        ),
                        0
                    );
                }
                FloatLayout::Unnormalized => {
                    normalization = MantissaNormalization::Unnormalized;
                    assert_eq!(
                        h5t::H5Tset_norm(datatype.id(), H5T_norm_t::H5T_NORM_NONE),
                        0
                    );
                }
                FloatLayout::Standard | FloatLayout::Vax => {}
            }
        }
    });
    let raw = if wide {
        [11.0f64, 3.0]
            .into_iter()
            .flat_map(|value| {
                if big_endian {
                    value.to_be_bytes()
                } else {
                    value.to_le_bytes()
                }
            })
            .collect::<Vec<_>>()
    } else {
        [11.0f32, 3.0]
            .into_iter()
            .flat_map(|value| {
                if big_endian {
                    value.to_be_bytes()
                } else {
                    value.to_le_bytes()
                }
            })
            .collect::<Vec<_>>()
    };
    numeric::write_stored(&path, &datatype, &raw, 2);
    let dataset = File::open(&path).unwrap().dataset("values").unwrap();
    let byte_order = if matches!(encoding, FloatLayout::Vax) {
        DatatypeByteOrder::Vax
    } else {
        pure_order(big_endian)
    };
    let size = if wide { 8 } else { 4 };
    let numeric = FloatingPointType::new(
        size,
        byte_order,
        normalization,
        sign_location,
        layout.bit_offset(),
        layout.bit_precision(),
        layout.exponent_location(),
        layout.exponent_size(),
        mantissa_location,
        mantissa_size,
        exponent_bias,
    )
    .unwrap();
    let layout = numeric.layout();
    let stored = Datatype::FloatingPoint(numeric);
    assert_eq!(dataset.datatype().unwrap(), stored);
    assert_eq!(dataset.read_raw().unwrap(), raw);
    let pure_path = dir.path().join("pure-float.h5");
    let mut builder = FileBuilder::new();
    builder.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    builder
        .create_dataset("values")
        .with_raw_data(stored, raw, 2)
        .with_shape(&[2]);
    builder.write(&pure_path).unwrap();
    let reference = hdf5::File::open(&pure_path)
        .unwrap()
        .dataset("values")
        .unwrap();
    assert_eq!(reference.dtype().unwrap(), datatype);
    drop(reference);
    if matches!(encoding, FloatLayout::Standard) {
        let reference = hdf5::File::open(&path).unwrap().dataset("values").unwrap();
        assert_eq!(reference.read_raw::<f64>().unwrap(), vec![11.0, 3.0]);
        assert_eq!(dataset.read_f64().unwrap(), vec![11.0, 3.0]);
        assert_eq!(dataset.read_f32().unwrap(), vec![11.0, 3.0]);
        assert_eq!(dataset.read_i64().unwrap(), vec![11, 3]);
        let raw = dataset.read_raw().unwrap();
        let stored = dataset.datatype().unwrap();
        assert_eq!(u8::convert_raw(&raw, &stored).unwrap(), [11, 3]);
        assert_eq!(i8::convert_raw(&raw, &stored).unwrap(), [11, 3]);
        return;
    }
    let expected = if matches!(encoding, FloatLayout::Vax) {
        FormatError::UnsupportedVaxByteOrder
    } else {
        FormatError::UnsupportedFloatingPointLayout {
            size: size as usize,
            layout,
        }
    };
    assert_format(dataset.read_f64().unwrap_err(), &expected);
    assert_format(dataset.read_f32().unwrap_err(), &expected);
    assert_format(dataset.read_i64().unwrap_err(), &expected);
}

#[test]
#[should_panic(expected = "fixture datatype must be fixed-point, floating-point, or enumeration")]
fn rejects_pointer_bearing_fixture_type() {
    let dir = tempfile::tempdir().unwrap();
    let datatype = CDatatype::from_type::<hdf5::types::VarLenArray<u8>>().unwrap();
    let bytes = vec![0; datatype.size()];
    numeric::write_stored(&dir.path().join("invalid.h5"), &datatype, &bytes, 1);
}

#[derive(Clone, Copy, Debug)]
enum FloatLayout {
    Bias,
    LeadingOne,
    Sign,
    Standard,
    Unnormalized,
    Vax,
}

fn assert_format(error: Error, expected: &FormatError) {
    let Error::Format(actual) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(&actual, expected);
}

fn pure_order(big_endian: bool) -> DatatypeByteOrder {
    if big_endian {
        DatatypeByteOrder::BigEndian
    } else {
        DatatypeByteOrder::LittleEndian
    }
}
