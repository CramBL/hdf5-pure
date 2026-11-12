#![cfg(feature = "__hdf5-1.10")]

use std::ffi::CString;

use hdf5::Datatype as NativeDatatype;
use hdf5::file::LibraryVersion;
use hdf5::globals;
use hdf5_pure::File;
use hdf5_pure_core::__private;
use hdf5_pure_core::{
    Datatype, DatatypeByteOrder, FixedPointType, MessageType, VariableLengthType,
};
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::{
    AccessMode, DatatypeMessage, DatatypeVersion, FormatWidths, ObjectHeader,
};
use hdf5_sys::h5i::hid_t;
use hdf5_sys::h5t;
use rstest::rstest;
use test_util_hdf5::{lock, storage};

#[rstest]
fn datatype_enum_codecs_preserve_native_descriptions(
    #[values(false, true)] big_endian: bool,
    #[values(false, true)] latest: bool,
    #[values(Shape::Direct, Shape::Array, Shape::Sequence)] shape: Shape,
) {
    let _guard = lock::libhdf5_guard();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("enum.h5");
    let version = if latest {
        LibraryVersion::latest()
    } else {
        LibraryVersion::V110
    };
    let datatype = shape.native(big_endian);
    let file = hdf5::File::with_options()
        .with_fapl(|properties| properties.libver_bounds(version, version))
        .create(&path)
        .unwrap();
    let dataset = file
        .new_dataset_builder()
        .empty_as(&datatype)
        .shape(0)
        .create("values")
        .unwrap();
    drop(dataset);
    drop(datatype);
    file.close().unwrap();
    let image = std::fs::read(&path).unwrap();
    let pure = File::from_bytes(image.clone()).unwrap();
    let superblock = pure.superblock();
    let header = ObjectHeader::parse_with_base_and_widths(
        &image,
        AccessMode::ReadOnly,
        usize::try_from(storage::root_object_address(&pure, &image, "values").get()).unwrap(),
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
    let (message, consumed) = DatatypeMessage::parse(bytes).unwrap();
    assert_eq!(message.to_datatype().unwrap(), shape.semantic(big_endian));
    assert_eq!(
        message.serialize(superblock.offset_size).unwrap(),
        bytes[..consumed]
    );
    assert_eq!(&bytes[consumed..], vec![0; bytes.len() - consumed]);
    let semantic = shape.semantic(big_endian);
    let canonical =
        format::serialize_datatype_with_offset_width(&semantic, superblock.offset_size).unwrap();
    let (parsed, consumed) = DatatypeMessage::parse(&canonical).unwrap();
    assert_eq!(consumed, canonical.len());
    assert_eq!(parsed.to_datatype().unwrap(), semantic);
    assert_eq!(parsed.version(), DatatypeVersion::Three);
    if !latest {
        assert_eq!(message.version(), DatatypeVersion::Three);
    }
    #[cfg(feature = "__hdf5-2")]
    if latest {
        assert_eq!(message.version(), DatatypeVersion::Five);
    }
    drop(pure);
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    Array,
    Direct,
    Sequence,
}

impl Shape {
    fn native(self, big_endian: bool) -> NativeDatatype {
        hdf5::sync::sync(|| {
            let base = if big_endian {
                *globals::H5T_STD_I16BE
            } else {
                *globals::H5T_STD_I16LE
            };
            // SAFETY: The predefined integer parent is live and the call creates a new ID.
            let enumeration = owned(unsafe { h5t::H5Tenum_create(base) });
            let negative = CString::new("negative").unwrap();
            let positive = CString::new("é").unwrap();
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

    fn semantic(self, big_endian: bool) -> Datatype {
        let enumeration = Datatype::Enumeration {
            size: 2,
            base_type: Box::new(Datatype::FixedPoint(
                FixedPointType::new(
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
                .unwrap(),
            )),
            members: vec![
                __private::enum_member("negative".into(), value(-3, big_endian).to_vec()),
                __private::enum_member("é".into(), value(7, big_endian).to_vec()),
            ],
        };
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
    unsafe { hdf5::from_id::<NativeDatatype>(id) }.unwrap()
}
