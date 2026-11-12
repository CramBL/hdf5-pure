#![cfg(feature = "hdf5")]

use hdf5::Datatype as NativeDatatype;
use hdf5::file::LibraryVersion;
use hdf5::globals;
use hdf5::plist::file_create::{Sizeof, SizeofInfo};
use hdf5_pure::File;
use hdf5_pure_core::{
    CharacterSet, Datatype, DatatypeByteOrder, FixedPointType, FloatingPointType,
    MantissaNormalization, MessageType, StringPadding, VariableLengthStringType,
    VariableLengthType,
};
use hdf5_pure_format::__private::{
    AccessMode, DatatypeMessage, FormatWidths, ObjectHeader, OffsetWidth,
};
use hdf5_sys::h5t;
use hdf5_sys::h5t::{H5T_cset_t, H5T_str_t};
use rstest::rstest;
use test_util_hdf5::{lock, storage};

#[rstest]
#[case::integer(Kind::Integer)]
#[case::float(Kind::Float)]
#[case::vax(Kind::Vax)]
#[case::nested(Kind::Nested)]
#[case::ascii(Kind::Ascii)]
#[case::utf8(Kind::Utf8)]
fn datatype_variable_length_codecs_preserve_native_file_metadata(
    #[case] kind: Kind,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
    #[values(false, true)] latest: bool,
) {
    let _guard = lock::libhdf5_guard();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("variable-length.h5");
    let file = hdf5::File::with_options()
        .with_fapl(|properties| {
            let version = if latest {
                LibraryVersion::latest()
            } else {
                LibraryVersion::V18
            };
            properties.libver_bounds(version, version)
        })
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
    let datatype = kind.native();
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
    let body = &header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::DATATYPE)
        .unwrap()
        .data;
    assert_eq!(body[0] & 0x0f, 9);
    assert_eq!(
        u32::from_le_bytes(body[4..8].try_into().unwrap()),
        8 + u32::from(width.get())
    );
    let (message, consumed) = DatatypeMessage::parse(body).unwrap();
    assert_eq!(message.to_datatype().unwrap(), kind.semantic());
    assert_eq!(message.version().get(), body[0] >> 4);
    if !latest {
        assert_eq!(
            message.version().get(),
            if matches!(kind, Kind::Vax) { 3 } else { 1 }
        );
    }
    assert_eq!(message.serialize(width).unwrap(), body[..consumed]);
    assert_eq!(&body[consumed..], vec![0; body.len() - consumed]);
    drop(pure);
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Ascii,
    Float,
    Integer,
    Nested,
    Utf8,
    Vax,
}

impl Kind {
    fn native(self) -> NativeDatatype {
        hdf5::sync::sync(|| {
            let source = match self {
                Self::Ascii | Self::Utf8 => *globals::H5T_C_S1,
                Self::Float => *globals::H5T_IEEE_F64BE,
                Self::Integer | Self::Nested => *globals::H5T_STD_I16BE,
                Self::Vax => *globals::H5T_VAX_F64,
            };
            let id = match self {
                // SAFETY: The predefined string type is live and the copy returns a new ID.
                Self::Ascii | Self::Utf8 => unsafe { h5t::H5Tcopy(source) },
                // SAFETY: The predefined base is live and H5Tvlen_create returns a new ID.
                _ => unsafe { h5t::H5Tvlen_create(source) },
            };
            assert!(id >= 0);
            // SAFETY: The creation call transfers a newly created datatype ID exactly once.
            let datatype = unsafe { hdf5::from_id::<NativeDatatype>(id) }.unwrap();
            match self {
                Self::Ascii | Self::Utf8 => {
                    assert_eq!(
                        // SAFETY: The owned string ID is live and H5T_VARIABLE selects a VL string.
                        unsafe { h5t::H5Tset_size(datatype.id(), h5t::H5T_VARIABLE) },
                        0,
                    );
                    if matches!(self, Self::Utf8) {
                        assert_eq!(
                            // SAFETY: The owned string ID is live and UTF-8 is a defined charset.
                            unsafe { h5t::H5Tset_cset(datatype.id(), H5T_cset_t::H5T_CSET_UTF8) },
                            0,
                        );
                        assert_eq!(
                            // SAFETY: The owned string ID is live and space padding is defined.
                            unsafe {
                                h5t::H5Tset_strpad(datatype.id(), H5T_str_t::H5T_STR_SPACEPAD)
                            },
                            0,
                        );
                    }
                    datatype
                }
                Self::Nested => {
                    // SAFETY: The owned base is live and H5Tvlen_create returns a new ID.
                    let id = unsafe { h5t::H5Tvlen_create(datatype.id()) };
                    assert!(id >= 0);
                    // SAFETY: The creation call transfers a newly created datatype ID once.
                    unsafe { hdf5::from_id::<NativeDatatype>(id) }.unwrap()
                }
                _ => datatype,
            }
        })
    }

    // `H5T__set_size` in `H5T.c`, HDF5 1.10.11 and 2.2.0, copies `H5T_NATIVE_UCHAR`
    // as the parent of a variable-length string.
    fn semantic(self) -> Datatype {
        let integer = Datatype::FixedPoint(
            FixedPointType::new(2, DatatypeByteOrder::BigEndian, true, 0, 16).unwrap(),
        );
        let base = match self {
            Self::Ascii | Self::Utf8 => {
                return Datatype::VariableLength(VariableLengthType::String(
                    VariableLengthStringType::with_base_type(
                        if matches!(self, Self::Utf8) {
                            StringPadding::SpacePad
                        } else {
                            StringPadding::NullTerminate
                        },
                        if matches!(self, Self::Utf8) {
                            CharacterSet::Utf8
                        } else {
                            CharacterSet::Ascii
                        },
                        Datatype::FixedPoint(
                            FixedPointType::new(
                                1,
                                if cfg!(target_endian = "big") {
                                    DatatypeByteOrder::BigEndian
                                } else {
                                    DatatypeByteOrder::LittleEndian
                                },
                                false,
                                0,
                                8,
                            )
                            .unwrap(),
                        ),
                    )
                    .unwrap(),
                ));
            }
            Self::Float => Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(
                DatatypeByteOrder::BigEndian,
            )),
            Self::Integer => integer,
            Self::Nested => Datatype::VariableLength(VariableLengthType::sequence(integer)),
            Self::Vax => Datatype::FloatingPoint(
                FloatingPointType::new(
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
            ),
        };
        Datatype::VariableLength(VariableLengthType::sequence(base))
    }
}

/// The VAX exponent bias assigned by `H5T_INIT_TYPE_DOUBLEVAX_CORE` (`H5T.c`, HDF5 2.2.0).
const VAX_BIAS: u32 = 0x0401;
