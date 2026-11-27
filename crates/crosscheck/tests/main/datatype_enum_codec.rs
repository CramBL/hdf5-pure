#![cfg(feature = "__libhdf5-1.10")]

use std::ffi::CString;

use hdf5_pure::{File, FileBuilder};
use hdf5_pure_core::{
    Datatype, DatatypeByteOrder, DatatypeMemberName, EnumMember, EnumerationType, FixedPointType,
    MessageType, VariableLengthType,
};
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::DecodedDatatype;
use hdf5_pure_format::__private::{
    AccessMode, DatatypeMessage, DatatypeVersion, FormatWidths, ObjectHeader,
};
use libhdf5::Datatype as NativeDatatype;
use libhdf5::datatype::{ByteOrder, EncodedEnumVariant};
use libhdf5::file::LibraryVersion;
use libhdf5::globals;
use libhdf5_sys::h5i::hid_t;
use libhdf5_sys::h5t;
use rstest::rstest;
use test_util::datatype;
use test_util_hdf5::__private::c_object;
use test_util_hdf5::lock;

#[rstest]
#[case::earliest_direct(LibraryVersion::Earliest, Shape::Direct)]
#[case::earliest_sequence(LibraryVersion::Earliest, Shape::Sequence)]
#[case::v110_direct(LibraryVersion::V110, Shape::Direct)]
#[case::v110_array(LibraryVersion::V110, Shape::Array)]
#[case::v110_sequence(LibraryVersion::V110, Shape::Sequence)]
#[case::latest_direct(LibraryVersion::latest(), Shape::Direct)]
#[case::latest_array(LibraryVersion::latest(), Shape::Array)]
#[case::latest_sequence(LibraryVersion::latest(), Shape::Sequence)]
fn datatype_enum_codecs_preserve_native_descriptions(
    #[case] version: LibraryVersion,
    #[case] shape: Shape,
    #[values(false, true)] big_endian: bool,
) {
    let _guard = lock::libhdf5_guard();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("enum.h5");
    let upper_bound = if version == LibraryVersion::Earliest {
        LibraryVersion::V18
    } else {
        version
    };
    let name = b"\xff";
    let datatype = shape.native(big_endian, name);
    let file = libhdf5::File::with_options()
        .with_fapl(|properties| properties.libver_bounds(version, upper_bound))
        .create(&path)
        .unwrap();
    let dataset = file
        .new_dataset_builder()
        .empty_as(&datatype)
        .shape(0)
        .create("values")
        .unwrap();
    let address = c_object::header_address(&file, "values");
    drop(dataset);
    drop(datatype);
    file.close().unwrap();
    let image = std::fs::read(&path).unwrap();
    let pure = File::from_bytes(image.clone()).unwrap();
    let superblock = pure.superblock();
    let header = ObjectHeader::parse_with_base_and_widths(
        &image,
        AccessMode::ReadOnly,
        usize::try_from(address).unwrap(),
        FormatWidths {
            offsets: superblock.offset_size,
            lengths: superblock.length_size,
        },
        superblock.base_address,
    )
    .unwrap();
    let bytes = &header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::DATATYPE)
        .unwrap()
        .data;
    let (message, consumed) = DatatypeMessage::parse_prefix(bytes).unwrap();
    let (decoded, bytes_read) = DecodedDatatype::parse_prefix(bytes).unwrap();
    assert_eq!(
        (decoded.into_datatype(), bytes_read),
        (message.to_datatype().unwrap(), consumed)
    );
    let semantic = shape.semantic(big_endian, name);
    assert_eq!(message.to_datatype().unwrap(), semantic);
    assert_eq!(
        pure.dataset("values").unwrap().datatype().unwrap(),
        semantic
    );
    assert_eq!(
        message.serialize(superblock.offset_size).unwrap(),
        bytes[..consumed]
    );
    assert_eq!(&bytes[consumed..], vec![0; bytes.len() - consumed]);
    let canonical =
        format::serialize_datatype_with_offset_width(&semantic, superblock.offset_size).unwrap();
    let (parsed, consumed) = DatatypeMessage::parse_prefix(&canonical).unwrap();
    assert_eq!(consumed, canonical.len());
    assert_eq!(parsed.to_datatype().unwrap(), semantic);
    assert_eq!(parsed.version(), DatatypeVersion::Three);
    match version {
        LibraryVersion::Earliest => assert_eq!(message.version(), DatatypeVersion::One),
        LibraryVersion::V110 => assert_eq!(message.version(), DatatypeVersion::Three),
        #[cfg(feature = "__libhdf5-2")]
        version if version == LibraryVersion::latest() => {
            assert_eq!(message.version(), DatatypeVersion::Five);
        }
        _ => {}
    }
    drop(pure);
}

#[test]
fn native_reader_preserves_a_decoded_floating_point_enumeration() {
    let _guard = lock::libhdf5_guard();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("decoded_enum.h5");
    std::fs::write(&path, datatype::float_enum_dataset(4)).unwrap();

    let file = libhdf5::File::open(&path).unwrap();
    let dataset = file.dataset("values").unwrap();
    let enumeration = dataset.dtype().unwrap();
    let base = NativeDatatype::from_type::<f32>()
        .unwrap()
        .with_byte_order(ByteOrder::LittleEndian)
        .unwrap();
    assert_eq!(enumeration.base().unwrap(), base);
    assert_eq!(enumeration.size(), 4);
    assert_eq!(
        enumeration.encoded_enum_variants().unwrap(),
        [EncodedEnumVariant {
            name: "one".into(),
            value: 1.0f32.to_le_bytes().into()
        }]
    );
    drop(enumeration);
    drop(dataset);
    file.close().unwrap();
}

#[rstest]
fn native_reader_preserves_canonical_integer_enumeration_bytes(
    #[values(false, true)] big_endian: bool,
) {
    let _guard = lock::libhdf5_guard();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("canonical_enum.h5");
    let mut builder = FileBuilder::new();
    let data = value(-3, big_endian);
    builder.create_dataset("values").with_raw_data(
        Shape::Direct.semantic(big_endian, "é".as_bytes()),
        data.to_vec(),
        1,
    );
    builder.write(&path).unwrap();

    let file = libhdf5::File::open(&path).unwrap();
    let dataset = file.dataset("values").unwrap();
    let enumeration = dataset.dtype().unwrap();
    let base = NativeDatatype::from_type::<i16>()
        .unwrap()
        .with_byte_order(if big_endian {
            ByteOrder::BigEndian
        } else {
            ByteOrder::LittleEndian
        })
        .unwrap();
    assert_eq!(enumeration.base().unwrap(), base);
    assert_eq!(enumeration.size(), 2);
    assert_eq!(
        enumeration.encoded_enum_variants().unwrap(),
        [
            EncodedEnumVariant {
                name: "negative".into(),
                value: data.into()
            },
            EncodedEnumVariant {
                name: "é".into(),
                value: value(7, big_endian).into()
            },
        ],
    );
    drop(enumeration);
    drop(dataset);
    file.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    Array,
    Direct,
    Sequence,
}

impl Shape {
    fn native(self, big_endian: bool, name: &[u8]) -> NativeDatatype {
        libhdf5::sync::sync(|| {
            let base = if big_endian {
                *globals::H5T_STD_I16BE
            } else {
                *globals::H5T_STD_I16LE
            };
            // SAFETY: The predefined integer parent is live and the call creates a new ID.
            let enumeration = owned(unsafe { h5t::H5Tenum_create(base) });
            let negative = CString::new("negative").unwrap();
            let positive = CString::new(name).unwrap();
            let negative_value = value(-3, big_endian);
            let positive_value = value(7, big_endian);
            assert_eq!(
                // SAFETY: The ID, terminated name and two-byte value are live for the call.
                unsafe {
                    h5t::H5Tenum_insert(
                        enumeration.id(),
                        negative.as_ptr(),
                        negative_value.as_ptr().cast(),
                    )
                },
                0,
            );
            assert_eq!(
                // SAFETY: The ID, terminated name and two-byte value are live for the call.
                unsafe {
                    h5t::H5Tenum_insert(
                        enumeration.id(),
                        positive.as_ptr(),
                        positive_value.as_ptr().cast(),
                    )
                },
                0,
            );
            match self {
                Self::Array => {
                    let dimensions = [2];
                    // SAFETY: The parent is live. The dimension array has one entry for the
                    // declared rank.
                    owned(unsafe {
                        h5t::H5Tarray_create2(enumeration.id(), 1, dimensions.as_ptr())
                    })
                }
                Self::Direct => enumeration,
                // SAFETY: The parent is live and the call creates a new ID.
                Self::Sequence => owned(unsafe { h5t::H5Tvlen_create(enumeration.id()) }),
            }
        })
    }

    fn semantic(self, big_endian: bool, name: &[u8]) -> Datatype {
        let base = FixedPointType::new(
            2,
            if big_endian {
                DatatypeByteOrder::BigEndian
            } else {
                DatatypeByteOrder::LittleEndian
            },
            true,
            0,
            16,
        )
        .unwrap();
        let members = vec![
            EnumMember::new(
                DatatypeMemberName::new("negative").unwrap(),
                value(-3, big_endian).to_vec(),
            ),
            EnumMember::new(
                DatatypeMemberName::new(name).unwrap(),
                value(7, big_endian).to_vec(),
            ),
        ];
        let enumeration = Datatype::Enumeration(EnumerationType::new(base, members).unwrap());
        match self {
            Self::Array => Datatype::Array {
                base_type: Box::new(enumeration),
                dimensions: vec![2],
            },
            Self::Direct => enumeration,
            Self::Sequence => Datatype::VariableLength(VariableLengthType::sequence(enumeration)),
        }
    }
}

fn value(value: i16, big_endian: bool) -> [u8; 2] {
    if big_endian {
        value.to_be_bytes()
    } else {
        value.to_le_bytes()
    }
}

fn owned(id: hid_t) -> NativeDatatype {
    assert!(id >= 0);
    // SAFETY: Every caller transfers a newly created datatype ID exactly once.
    unsafe { libhdf5::from_id::<NativeDatatype>(id) }.unwrap()
}
