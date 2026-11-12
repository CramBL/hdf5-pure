#![cfg(feature = "__hdf5-1.10")]

use hdf5::Datatype as NativeDatatype;
use hdf5::file::LibraryVersion;
use hdf5::globals;
use hdf5::plist::file_create::{Sizeof, SizeofInfo};
use hdf5_pure::File;
#[cfg(feature = "__hdf5-2")]
use hdf5_pure_core::ComplexType;
use hdf5_pure_core::{
    Datatype, DatatypeByteOrder, FloatingPointType, MessageType, VariableLengthType,
};
use hdf5_pure_format::__private::{
    AccessMode, DatatypeMessage, DatatypeVersion, FormatWidths, ObjectHeader, OffsetWidth,
};
use hdf5_sys::h5i::hid_t;
use hdf5_sys::h5t;
use rstest::rstest;
use test_util_hdf5::{lock, storage};

#[rstest]
fn datatype_array_complex_codecs_preserve_native_arrays(
    #[values(false, true)] big_endian: bool,
    #[values(false, true)] sequence: bool,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
    #[values(false, true)] latest: bool,
) {
    let _guard = lock::libhdf5_guard();
    let parent = native_float(big_endian);
    let mut semantic = Datatype::FloatingPoint(component(big_endian));
    let parent = if sequence {
        semantic = Datatype::VariableLength(VariableLengthType::sequence(semantic));
        hdf5::sync::sync(|| {
            // SAFETY: The floating-point parent is live and the call returns a new ID.
            owned(unsafe { h5t::H5Tvlen_create(parent.id()) })
        })
    } else {
        parent
    };
    let datatype = hdf5::sync::sync(|| {
        let dimensions = [2, 3];
        // SAFETY: The parent is live and the dimension array holds the declared two extents.
        owned(unsafe { h5t::H5Tarray_create2(parent.id(), 2, dimensions.as_ptr()) })
    });
    drop(parent);
    assert_native(
        width,
        if latest {
            LibraryVersion::latest()
        } else {
            LibraryVersion::V110
        },
        datatype,
        Datatype::Array {
            base_type: Box::new(semantic),
            dimensions: vec![2, 3],
        },
    );
}

#[cfg(feature = "__hdf5-2")]
#[rstest]
fn datatype_array_complex_codecs_preserve_native_complex(#[values(false, true)] big_endian: bool) {
    let _guard = lock::libhdf5_guard();
    let parent = native_float(big_endian);
    let datatype = hdf5::sync::sync(|| {
        // SAFETY: The parent is a live floating-point type and the call returns a new ID.
        owned(unsafe { h5t::H5Tcomplex_create(parent.id()) })
    });
    drop(parent);
    assert_native(
        OffsetWidth::Eight,
        LibraryVersion::latest(),
        datatype,
        Datatype::Complex(ComplexType::new(component(big_endian)).unwrap()),
    );
}

fn assert_native(
    width: OffsetWidth,
    version: LibraryVersion,
    datatype: NativeDatatype,
    semantic: Datatype,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("datatype.h5");
    let file = hdf5::File::with_options()
        .with_fapl(|properties| properties.libver_bounds(version, version))
        .with_fcpl(|properties| {
            properties.sizes(SizeofInfo {
                sizeof_addr: match width {
                    OffsetWidth::Two => Sizeof::Bytes2,
                    OffsetWidth::Four => Sizeof::Bytes4,
                    OffsetWidth::Eight => Sizeof::Bytes8,
                },
                sizeof_size: Sizeof::Bytes8,
            })
        })
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
    assert_eq!(superblock.offset_size, width);
    let header = ObjectHeader::parse_with_base_and_widths(
        &image,
        AccessMode::ReadOnly,
        usize::try_from(storage::root_object_address(&pure, &image, "values").get()).unwrap(),
        FormatWidths {
            offsets: width,
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
    assert_eq!(message.to_datatype().unwrap(), semantic);
    assert_eq!(message.serialize(width).unwrap(), bytes[..consumed]);
    assert_eq!(&bytes[consumed..], vec![0; bytes.len() - consumed]);
    if matches!(semantic, Datatype::Complex(_)) {
        assert_eq!(message.version(), DatatypeVersion::Five);
    } else if version == LibraryVersion::V110 {
        assert_eq!(message.version(), DatatypeVersion::Three);
    }
    drop(pure);
}

fn native_float(big_endian: bool) -> NativeDatatype {
    hdf5::sync::sync(|| {
        let source = if big_endian {
            *globals::H5T_IEEE_F64BE
        } else {
            *globals::H5T_IEEE_F64LE
        };
        // SAFETY: The predefined floating-point ID is live and the copy returns a new ID.
        owned(unsafe { h5t::H5Tcopy(source) })
    })
}

fn component(big_endian: bool) -> FloatingPointType {
    FloatingPointType::ieee754_binary64(if big_endian {
        DatatypeByteOrder::BigEndian
    } else {
        DatatypeByteOrder::LittleEndian
    })
}

fn owned(id: hid_t) -> NativeDatatype {
    assert!(id >= 0);
    // SAFETY: Every caller transfers a newly created datatype ID exactly once.
    unsafe { hdf5::from_id::<NativeDatatype>(id) }.unwrap()
}
