#![cfg(feature = "__hdf5-1.10")]

use std::ffi::CString;

use hdf5::Datatype as NativeDatatype;
use hdf5::file::LibraryVersion;
use hdf5::globals;
use hdf5::plist::file_create::{Sizeof, SizeofInfo};
use hdf5_pure::File;
use hdf5_pure_core::__private;
#[cfg(feature = "__hdf5-2")]
use hdf5_pure_core::ComplexType;
use hdf5_pure_core::{
    Datatype, DatatypeByteOrder, FloatingPointType, MessageType, VariableLengthType,
};
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::{
    AccessMode, DatatypeMessage, DatatypeVersion, FormatWidths, ObjectHeader, OffsetWidth,
};
use hdf5_sys::h5i::hid_t;
use hdf5_sys::h5t::{self, H5T_class_t};
use rstest::rstest;
use test_util_hdf5::{lock, storage};

#[rstest]
fn datatype_compound_codecs_preserve_native_sequences(
    #[values(false, true)] big_endian: bool,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
    #[values(false, true)] latest: bool,
    #[values(Placement::Adjacent, Placement::Disjoint)] placement: Placement,
) {
    assert_native(
        width,
        if latest {
            LibraryVersion::latest()
        } else {
            LibraryVersion::V110
        },
        placement,
        Shape::Sequence,
        big_endian,
    );
}

#[cfg(feature = "__hdf5-2")]
#[rstest]
fn datatype_compound_codecs_preserve_native_complex(
    #[values(false, true)] big_endian: bool,
    #[values(Placement::Adjacent, Placement::Disjoint)] placement: Placement,
) {
    assert_native(
        OffsetWidth::Eight,
        LibraryVersion::latest(),
        placement,
        Shape::Complex,
        big_endian,
    );
}

fn assert_native(
    width: OffsetWidth,
    version: LibraryVersion,
    placement: Placement,
    shape: Shape,
    big_endian: bool,
) {
    let _guard = lock::libhdf5_guard();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("compound.h5");
    let datatype = shape.native(placement, big_endian);
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
    assert_eq!(
        message.to_datatype().unwrap(),
        shape.semantic(width, placement, big_endian)
    );
    assert_eq!(message.serialize(width).unwrap(), bytes[..consumed]);
    assert_eq!(&bytes[consumed..], vec![0; bytes.len() - consumed]);
    let semantic = shape.semantic(width, placement, big_endian);
    let canonical = format::serialize_datatype_with_offset_width(&semantic, width).unwrap();
    let (parsed, consumed) = DatatypeMessage::parse(&canonical).unwrap();
    assert_eq!(consumed, canonical.len());
    assert_eq!(parsed.to_datatype().unwrap(), semantic);
    assert_eq!(
        parsed.version(),
        match shape {
            #[cfg(feature = "__hdf5-2")]
            Shape::Complex => DatatypeVersion::Five,
            Shape::Sequence => DatatypeVersion::Three,
        }
    );
    if version == LibraryVersion::V110 {
        assert_eq!(message.version(), DatatypeVersion::Three);
    }
    #[cfg(feature = "__hdf5-2")]
    if matches!(shape, Shape::Complex) {
        assert_eq!(message.version(), DatatypeVersion::Five);
    }
    drop(pure);
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    #[cfg(feature = "__hdf5-2")]
    Complex,
    Sequence,
}

impl Shape {
    fn native(self, placement: Placement, big_endian: bool) -> NativeDatatype {
        hdf5::sync::sync(|| {
            let base = if big_endian {
                *globals::H5T_IEEE_F64BE
            } else {
                *globals::H5T_IEEE_F64LE
            };
            let child = match self {
                #[cfg(feature = "__hdf5-2")]
                // SAFETY: The predefined floating parent is live and the call creates a new ID.
                Self::Complex => owned(unsafe { h5t::H5Tcomplex_create(base) }),
                // SAFETY: The predefined parent is live and the call creates a new ID.
                Self::Sequence => owned(unsafe { h5t::H5Tvlen_create(base) }),
            };
            // SAFETY: Compound is a defined class. Both fields fit within the element.
            let datatype =
                owned(unsafe { h5t::H5Tcreate(H5T_class_t::H5T_COMPOUND, placement.size()) });
            let first = CString::new("first").unwrap();
            let second = CString::new("é").unwrap();
            assert_eq!(
                // SAFETY: Both IDs and the terminated name are live. The eight-byte field fits.
                unsafe { h5t::H5Tinsert(datatype.id(), first.as_ptr(), 0, base) },
                0,
            );
            assert_eq!(
                // SAFETY: Both IDs and the terminated name are live. The child fits the remaining
                // sixteen bytes.
                unsafe {
                    h5t::H5Tinsert(
                        datatype.id(),
                        second.as_ptr(),
                        placement.offset(),
                        child.id(),
                    )
                },
                0,
            );
            datatype
        })
    }

    // H5T_set_loc in H5T.c, libhdf5 1.10.11 and 2.2.0, adjusts a compound's size by
    // the size changes of its members when their storage location changes.
    fn semantic(self, width: OffsetWidth, placement: Placement, big_endian: bool) -> Datatype {
        let component = FloatingPointType::ieee754_binary64(if big_endian {
            DatatypeByteOrder::BigEndian
        } else {
            DatatypeByteOrder::LittleEndian
        });
        let (size, child) = match self {
            #[cfg(feature = "__hdf5-2")]
            Self::Complex => (
                u32::try_from(placement.size()).unwrap(),
                Datatype::Complex(ComplexType::new(component).unwrap()),
            ),
            Self::Sequence => (
                u32::try_from(placement.size()).unwrap()
                    - u32::try_from(core::mem::size_of::<h5t::hvl_t>()).unwrap()
                    + 8
                    + u32::from(width.get()),
                Datatype::VariableLength(VariableLengthType::sequence(Datatype::FloatingPoint(
                    component,
                ))),
            ),
        };
        Datatype::Compound {
            size,
            members: vec![
                __private::compound_member("first".into(), 0, Datatype::FloatingPoint(component)),
                __private::compound_member(
                    "é".into(),
                    u64::try_from(placement.offset()).unwrap(),
                    child,
                ),
            ],
        }
    }
}

/// The position of the second member after the first eight-byte field.
#[derive(Clone, Copy, Debug)]
enum Placement {
    /// Starts immediately after the first field.
    Adjacent,
    /// Starts after an eight-byte gap.
    Disjoint,
}

impl Placement {
    fn size(self) -> usize {
        self.offset() + 16
    }
    fn offset(self) -> usize {
        match self {
            Self::Adjacent => 8,
            Self::Disjoint => 16,
        }
    }
}

fn owned(id: hid_t) -> NativeDatatype {
    assert!(id >= 0);
    // SAFETY: Every caller transfers a newly created datatype ID exactly once.
    unsafe { hdf5::from_id::<NativeDatatype>(id) }.unwrap()
}
