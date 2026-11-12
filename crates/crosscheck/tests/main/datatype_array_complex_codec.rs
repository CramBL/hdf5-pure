#![cfg(feature = "__hdf5-1.10")]

use hdf5::Datatype as NativeDatatype;
use hdf5::file::LibraryVersion;
use hdf5::globals;
use hdf5::plist::file_create::{Sizeof, SizeofInfo};
use hdf5_pure::File;
#[cfg(feature = "__hdf5-2")]
use hdf5_pure_core::{ComplexType, MantissaNormalization};
use hdf5_pure_core::{
    Datatype, DatatypeByteOrder, FloatingPointType, MessageType, VariableLengthType,
};
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::{
    AccessMode, DatatypeMessage, DatatypeProperties, DatatypeVersion, FormatWidths, ObjectHeader,
    OffsetWidth,
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
#[case::little(ComplexParent::Little)]
#[case::big(ComplexParent::Big)]
#[case::vax(ComplexParent::Vax)]
fn datatype_array_complex_codecs_preserve_native_complex(#[case] parent: ComplexParent) {
    let _guard = lock::libhdf5_guard();
    let component = parent.semantic();
    let parent = parent.native();
    let datatype = hdf5::sync::sync(|| {
        // SAFETY: The parent is a live floating-point type and the call returns a new ID.
        owned(unsafe { h5t::H5Tcomplex_create(parent.id()) })
    });
    drop(parent);
    assert_native(
        OffsetWidth::Eight,
        LibraryVersion::latest(),
        datatype,
        Datatype::Complex(ComplexType::new(component).unwrap()),
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
    assert_eq!(
        format::parse_datatype(bytes).unwrap(),
        (message.to_datatype().unwrap(), consumed)
    );
    assert_eq!(message.to_datatype().unwrap(), semantic);
    assert_eq!(message.serialize(width).unwrap(), bytes[..consumed]);
    assert_eq!(&bytes[consumed..], vec![0; bytes.len() - consumed]);
    let canonical = format::serialize_datatype_with_offset_width(&semantic, width).unwrap();
    let (parsed, consumed) = DatatypeMessage::parse(&canonical).unwrap();
    assert_eq!(consumed, canonical.len());
    assert_eq!(parsed.to_datatype().unwrap(), semantic);
    match parsed.properties() {
        DatatypeProperties::Array {
            base_type,
            dimensions: _,
        } => {
            assert_eq!(parsed.version(), DatatypeVersion::Three);
            assert_eq!(base_type.version(), DatatypeVersion::One);
        }
        DatatypeProperties::Complex { component } => {
            assert_eq!(parsed.version(), DatatypeVersion::Five);
            let DatatypeProperties::FloatingPoint(properties) = component.properties() else {
                panic!("{component:?}")
            };
            assert_eq!(
                component.version(),
                if properties.byte_order() == DatatypeByteOrder::Vax {
                    DatatypeVersion::Three
                } else {
                    DatatypeVersion::One
                }
            );
        }
        _ => panic!("{parsed:?}"),
    }
    if matches!(semantic, Datatype::Complex(_)) {
        assert_eq!(message.version(), DatatypeVersion::Five);
    } else if version == LibraryVersion::V110 {
        assert_eq!(message.version(), DatatypeVersion::Three);
    }
    drop(pure);
}

#[cfg(feature = "__hdf5-2")]
#[derive(Clone, Copy, Debug)]
enum ComplexParent {
    Big,
    Little,
    Vax,
}

#[cfg(feature = "__hdf5-2")]
impl ComplexParent {
    fn native(self) -> NativeDatatype {
        match self {
            Self::Big => native_float(true),
            Self::Little => native_float(false),
            Self::Vax => hdf5::sync::sync(|| {
                // SAFETY: The predefined VAX datatype is live and the call returns a new ID.
                owned(unsafe { h5t::H5Tcopy(*globals::H5T_VAX_F64) })
            }),
        }
    }

    fn semantic(self) -> FloatingPointType {
        match self {
            Self::Big => component(true),
            Self::Little => component(false),
            Self::Vax => FloatingPointType::new(
                8,
                DatatypeByteOrder::Vax,
                MantissaNormalization::ImpliedLeadingOne,
                63,
                0,
                64,
                52,
                11,
                0,
                52,
                VAX_BIAS,
            )
            .unwrap(),
        }
    }
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

/// The VAX exponent bias assigned by `H5T_INIT_TYPE_DOUBLEVAX_CORE` (`H5T.c`, HDF5 2.2.0).
#[cfg(feature = "__hdf5-2")]
const VAX_BIAS: u32 = 0x0401;
