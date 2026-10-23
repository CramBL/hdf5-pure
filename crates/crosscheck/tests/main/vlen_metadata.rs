#![cfg(feature = "hdf5")]

use hdf5::Datatype as CDatatype;
use hdf5::file::LibraryVersion;
use hdf5::types::FloatSize;
use hdf5::types::IntSize;
use hdf5::types::Reference;
use hdf5::types::TypeDescriptor;
use hdf5::types::VarLenAscii;
use hdf5::types::VarLenUnicode;
use hdf5_pure::CharacterSet;
use hdf5_pure::Datatype;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FixedPointType;
use hdf5_pure::FormatError;
use hdf5_pure::LibVer;
use hdf5_pure::StringPadding;
use hdf5_pure::VariableLengthStringType;
use hdf5_pure::VariableLengthType;
use hdf5_sys::h5p::H5P_DEFAULT;
use hdf5_sys::h5t;
use hdf5_sys::h5t::H5T_class_t;
use hdf5_sys::h5t::H5T_order_t;
use hdf5_sys::h5t::H5T_sign_t;
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
        Datatype::VariableLength(VariableLengthType::String(
            VariableLengthStringType::with_base_type(
                padding,
                if utf8 {
                    CharacterSet::Utf8
                } else {
                    CharacterSet::Ascii
                },
                Datatype::FixedPoint(
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
                )
            )
            .unwrap()
        ))
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
        Datatype::VariableLength(VariableLengthType::sequence(Datatype::VariableLength(
            VariableLengthType::sequence(Datatype::FixedPoint(
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
            ))
        )))
    );
    let encoded = hdf5_pure_format::__private::serialize_datatype(&datatype);
    assert_eq!(
        hdf5_pure_format::__private::parse_datatype(&encoded).unwrap(),
        (datatype, encoded.len())
    );
}

#[rstest]
#[case::fixed_string(Some(TypeDescriptor::FixedAscii(1)))]
#[case::floating_point(Some(TypeDescriptor::Float(FloatSize::U4)))]
#[case::object_reference(Some(TypeDescriptor::Reference(Reference::Object)))]
#[case::opaque(None)]
fn preserves_c_variable_length_string_bases(#[case] descriptor: Option<TypeDescriptor>) {
    let base = match descriptor {
        Some(descriptor) => CDatatype::from_descriptor(&descriptor).unwrap(),
        None => hdf5::sync::sync(|| {
            // SAFETY: The lock serializes access, and three bytes is a positive opaque size.
            let id = unsafe { h5t::H5Tcreate(H5T_class_t::H5T_OPAQUE, 3) };
            assert!(id >= 0);
            // SAFETY: The returned handle owns the newly created datatype ID.
            unsafe { hdf5::from_id::<CDatatype>(id) }.unwrap()
        }),
    };
    let datatype = numeric::variable_length_string(&base);
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.h5");
    let destination = dir.path().join("destination.h5");
    {
        let file = hdf5::File::create(&source).unwrap();
        file.new_dataset_builder()
            .empty_as(&base)
            .shape(0)
            .create("base")
            .unwrap();
        let dataset = file
            .new_dataset_builder()
            .empty_as(&datatype)
            .shape(1)
            .create("strings")
            .unwrap();
        numeric::write_variable_length_strings(&dataset, &[c"alpha"]);
    }
    let file = File::open(&source).unwrap();
    let expected_base = file.dataset("base").unwrap().datatype().unwrap();
    let strings = file.dataset("strings").unwrap();
    assert_eq!(strings.read_string().unwrap(), ["alpha"]);
    let actual = strings.datatype().unwrap();
    let Datatype::VariableLength(VariableLengthType::String(string)) = &actual else {
        panic!("expected variable-length string, got {actual:?}");
    };
    assert_eq!(string.base_type(), &expected_base);
    let mut builder = FileBuilder::new();
    builder.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    builder.commit_datatype("strings", actual);
    builder.write(&destination).unwrap();
    let file = hdf5::File::open(&destination).unwrap();
    hdf5::sync::sync(|| {
        // SAFETY: The lock and live file cover the call. The name is NUL-terminated.
        let id = unsafe { h5t::H5Topen2(file.id(), c"strings".as_ptr(), H5P_DEFAULT) };
        assert!(id >= 0);
        // SAFETY: The returned handle owns the newly opened datatype ID.
        let written = unsafe { hdf5::from_id::<CDatatype>(id) }.unwrap();
        // SAFETY: Both handles remain live and the lock serializes access.
        assert_eq!(unsafe { h5t::H5Tequal(written.id(), datatype.id()) }, 1);
    });
}

#[rstest]
#[case(false, 1, 8, 0, false, false)]
#[case(true, 1, 8, 0, false, false)]
#[case(false, 1, 7, 1, true, true)]
#[case(true, 7, 33, 17, true, false)]
#[case(false, 1, 1, 0, false, true)]
fn preserves_c_variable_length_integer_string_bases(
    #[case] signed: bool,
    #[case] size: u32,
    #[case] precision: u16,
    #[case] offset: u16,
    #[case] low_padding: bool,
    #[case] high_padding: bool,
    #[values(false, true)] big_endian: bool,
) {
    let datatype = CDatatype::from_descriptor(&TypeDescriptor::VarLenAscii).unwrap();
    hdf5::sync::sync(|| {
        // SAFETY: The lock and live datatype cover the setters, whose values describe a valid integer parent.
        unsafe {
            assert_eq!(
                h5t::H5Tset_precision(datatype.id(), usize::from(precision)),
                0
            );
            assert_eq!(h5t::H5Tset_offset(datatype.id(), usize::from(offset)), 0);
            assert_eq!(
                h5t::H5Tset_sign(
                    datatype.id(),
                    if signed {
                        H5T_sign_t::H5T_SGN_2
                    } else {
                        H5T_sign_t::H5T_SGN_NONE
                    }
                ),
                0
            );
            assert_eq!(
                h5t::H5Tset_pad(
                    datatype.id(),
                    numeric::bit_padding(low_padding),
                    numeric::bit_padding(high_padding)
                ),
                0
            );
            assert_eq!(
                h5t::H5Tset_order(datatype.id(), numeric::byte_order(big_endian)),
                0
            );
        }
    });
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("strings.h5");
    {
        let file = hdf5::File::create(&path).unwrap();
        let dataset = file
            .new_dataset_builder()
            .empty_as(&datatype)
            .shape(1)
            .create("strings")
            .unwrap();
        numeric::write_variable_length_strings(&dataset, &[c"alpha"]);
    }
    let file = File::open(&path).unwrap();
    let dataset = file.dataset("strings").unwrap();
    assert_eq!(dataset.read_string().unwrap(), ["alpha"]);
    let actual = dataset.datatype().unwrap();
    let expected = Datatype::FixedPoint(
        FixedPointType::new(
            size,
            if big_endian {
                DatatypeByteOrder::BigEndian
            } else {
                DatatypeByteOrder::LittleEndian
            },
            signed,
            offset,
            precision,
        )
        .unwrap()
        .with_padding(low_padding, high_padding),
    );
    let Datatype::VariableLength(VariableLengthType::String(string)) = &actual else {
        panic!("expected variable-length string, got {actual:?}");
    };
    assert_eq!(string.base_type(), &expected);
    let bytes = hdf5_pure_format::__private::serialize_datatype(&actual);
    assert_eq!(
        hdf5_pure_format::__private::parse_datatype(&bytes).unwrap(),
        (actual, bytes.len())
    );
}
