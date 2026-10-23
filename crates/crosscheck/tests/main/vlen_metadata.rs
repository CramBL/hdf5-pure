#![cfg(feature = "hdf5")]

use hdf5::Datatype as CDatatype;
use hdf5::file::LibraryVersion;
use hdf5::types::IntSize;
use hdf5::types::TypeDescriptor;
use hdf5::types::VarLenAscii;
use hdf5::types::VarLenUnicode;
use hdf5_pure::CharacterSet;
use hdf5_pure::Datatype;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FixedPointType;
use hdf5_pure::FormatError;
use hdf5_pure::StringPadding;
use hdf5_sys::h5t;
use hdf5_sys::h5t::H5T_order_t;
use hdf5_sys::h5t::H5T_str_t;
use rstest::rstest;
use test_util_hdf5::numeric;

#[rstest]
#[case(StringPadding::NullTerminate, H5T_str_t::H5T_STR_NULLTERM)]
#[case(StringPadding::NullPad, H5T_str_t::H5T_STR_NULLPAD)]
#[case(StringPadding::SpacePad, H5T_str_t::H5T_STR_SPACEPAD)]
fn reads_c_variable_length_string_metadata(
    #[case] padding: StringPadding,
    #[case] c_padding: H5T_str_t,
    #[values(false, true)] utf8: bool,
    #[values(false, true)] big_endian: bool,
    #[values(false, true)] low_padding: bool,
    #[values(false, true)] high_padding: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("strings.h5");
    let descriptor = if utf8 {
        TypeDescriptor::VarLenUnicode
    } else {
        TypeDescriptor::VarLenAscii
    };
    let datatype = CDatatype::from_descriptor(&descriptor).unwrap();
    let order = if big_endian {
        H5T_order_t::H5T_ORDER_BE
    } else {
        H5T_order_t::H5T_ORDER_LE
    };
    hdf5::sync::sync(|| {
        // SAFETY: The lock serializes access and `datatype` owns a live transient string type.
        // Both padding and byte order are defined values accepted by these setters.
        unsafe {
            assert_eq!(h5t::H5Tset_strpad(datatype.id(), c_padding), 0);
            assert_eq!(h5t::H5Tset_order(datatype.id(), order), 0);
            assert_eq!(
                h5t::H5Tset_pad(
                    datatype.id(),
                    numeric::bit_padding(low_padding),
                    numeric::bit_padding(high_padding)
                ),
                0
            );
        }
    });
    {
        let file = hdf5::File::create(&path).unwrap();
        let dataset = file
            .new_dataset_builder()
            .empty_as(&datatype)
            .shape(2)
            .create("strings")
            .unwrap();
        if utf8 {
            dataset
                .write_raw(&[
                    "alpha".parse::<VarLenUnicode>().unwrap(),
                    "mètre".parse::<VarLenUnicode>().unwrap(),
                ])
                .unwrap();
        } else {
            dataset
                .write_raw(&[
                    VarLenAscii::from_ascii("alpha").unwrap(),
                    VarLenAscii::from_ascii("beta").unwrap(),
                ])
                .unwrap();
        }
    }
    let file = File::open(&path).unwrap();
    let dataset = file.dataset("strings").unwrap();
    let datatype = dataset.datatype().unwrap();
    assert_eq!(
        datatype,
        Datatype::VariableLength {
            is_string: true,
            padding: Some(padding),
            charset: Some(if utf8 {
                CharacterSet::Utf8
            } else {
                CharacterSet::Ascii
            }),
            base_type: Box::new(Datatype::FixedPoint(
                FixedPointType::new(
                    1,
                    if big_endian {
                        DatatypeByteOrder::BigEndian
                    } else {
                        DatatypeByteOrder::LittleEndian
                    },
                    false,
                    0,
                    8
                )
                .unwrap()
                .with_padding(low_padding, high_padding)
            )),
        }
    );
    assert_eq!(
        dataset.read_string().unwrap(),
        ["alpha", if utf8 { "mètre" } else { "beta" }]
    );
    let encoded = hdf5_pure_format::__private::serialize_datatype(&datatype);
    assert_eq!(
        hdf5_pure_format::__private::parse_datatype(&encoded).unwrap(),
        (datatype, encoded.len())
    );
}

#[rstest]
#[case([2, 0, 0], FormatError::InvalidVariableLengthSubtype(2))]
#[case([0, 0x10, 0], FormatError::InvalidVariableLengthReservedBits(0x00_1000))]
#[case([0x10, 1, 0], FormatError::InvalidVariableLengthSequenceMetadata { padding: 1, charset: 1 })]
fn reports_variable_length_metadata_errors_through_the_public_api(
    #[case] flags: [u8; 3],
    #[case] expected: FormatError,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("invalid.h5");
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_bounds(LibraryVersion::Earliest, LibraryVersion::V18))
            .create(&path)
            .unwrap();
        file.new_dataset::<VarLenAscii>()
            .shape(0)
            .create("strings")
            .unwrap();
    }
    let datatype = File::open(&path)
        .unwrap()
        .dataset("strings")
        .unwrap()
        .datatype()
        .unwrap();
    let encoded = hdf5_pure_format::__private::serialize_datatype(&datatype);
    let mut bytes = std::fs::read(&path).unwrap();
    let offset = bytes
        .windows(encoded.len())
        .position(|part| part == encoded)
        .unwrap();
    bytes[offset + 1..offset + 4].copy_from_slice(&flags);
    std::fs::write(&path, bytes).unwrap();
    let error = File::open(&path)
        .unwrap()
        .dataset("strings")
        .and_then(|dataset| dataset.datatype())
        .unwrap_err();
    let Error::Format(actual) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(actual, expected);
}

#[test]
fn reads_c_nested_variable_length_sequence_datatype() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nested.h5");
    {
        let file = hdf5::File::create(&path).unwrap();
        file.new_dataset_builder()
            .empty_as(TypeDescriptor::VarLenArray(Box::new(
                TypeDescriptor::VarLenArray(Box::new(TypeDescriptor::Unsigned(IntSize::U4))),
            )))
            .shape(0)
            .create("nested")
            .unwrap();
    }
    let file = File::open(&path).unwrap();
    let datatype = file.dataset("nested").unwrap().datatype().unwrap();
    assert_eq!(
        datatype,
        Datatype::VariableLength {
            is_string: false,
            padding: None,
            charset: None,
            base_type: Box::new(Datatype::VariableLength {
                is_string: false,
                padding: None,
                charset: None,
                base_type: Box::new(Datatype::FixedPoint(
                    FixedPointType::new(
                        4,
                        if cfg!(target_endian = "big") {
                            DatatypeByteOrder::BigEndian
                        } else {
                            DatatypeByteOrder::LittleEndian
                        },
                        false,
                        0,
                        32
                    )
                    .unwrap()
                )),
            }),
        }
    );
    let encoded = hdf5_pure_format::__private::serialize_datatype(&datatype);
    assert_eq!(
        hdf5_pure_format::__private::parse_datatype(&encoded).unwrap(),
        (datatype, encoded.len())
    );
}
