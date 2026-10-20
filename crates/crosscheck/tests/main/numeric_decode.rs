#![cfg(feature = "hdf5")]

use hdf5::Datatype as CDatatype;
use hdf5_pure::{
    Datatype, DatatypeByteOrder, Error, File, FixedPointLayout, FloatingPointLayout, FormatError,
    MantissaNormalization,
};
use hdf5_pure::{FileBuilder, LibVer};
use hdf5_sys::h5t::{self, H5T_norm_t, H5T_order_t};
use rstest::rstest;
use test_util_hdf5::numeric;

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
    let base = match &stored {
        Datatype::Enumeration { base_type, .. } => base_type.as_ref(),
        other => other,
    };
    assert_eq!(
        *base,
        Datatype::FixedPoint {
            size: 1,
            byte_order: pure_order(big_endian),
            layout: FixedPointLayout {
                signed: false,
                bit_offset: 4,
                bit_precision: 4
            }
        }
    );
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
    let mut layout = if wide {
        FloatingPointLayout::IEEE754_BINARY64
    } else {
        FloatingPointLayout::IEEE754_BINARY32
    };
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
                    layout.exponent_bias += 1;
                    assert_eq!(
                        h5t::H5Tset_ebias(datatype.id(), layout.exponent_bias as usize),
                        0
                    );
                }
                FloatLayout::LeadingOne => {
                    layout.mantissa_normalization = MantissaNormalization::LeadingOne;
                    assert_eq!(
                        h5t::H5Tset_norm(datatype.id(), H5T_norm_t::H5T_NORM_MSBSET),
                        0
                    );
                }
                FloatLayout::Sign => {
                    layout.sign_location = 0;
                    layout.mantissa_location = 1;
                    layout.mantissa_size -= 1;
                    assert_eq!(
                        h5t::H5Tset_fields(
                            datatype.id(),
                            0,
                            layout.exponent_location.into(),
                            layout.exponent_size.into(),
                            1,
                            layout.mantissa_size.into()
                        ),
                        0
                    );
                }
                FloatLayout::Unnormalized => {
                    layout.mantissa_normalization = MantissaNormalization::Unnormalized;
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
    let stored = Datatype::FloatingPoint {
        size,
        byte_order,
        layout,
    };
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
