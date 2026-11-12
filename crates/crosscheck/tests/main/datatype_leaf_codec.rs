#![cfg(feature = "hdf5")]

use std::ffi::CString;

use hdf5::Datatype as NativeDatatype;
use hdf5::file::LibraryVersion;
use hdf5::globals;
use hdf5_pure_core::{
    BitFieldType, CharacterSet, DatatypeByteOrder, FixedPointType, FloatingPointType,
    MantissaNormalization, StringPadding, TimeType,
};
use hdf5_pure_format::__private::{DatatypeMessage, DatatypeProperties, OffsetWidth};
use hdf5_sys::h5i::hid_t;
use hdf5_sys::h5t;
use hdf5_sys::h5t::{H5T_class_t, H5T_cset_t, H5T_str_t};
use rstest::rstest;
use test_util_hdf5::lock;

#[rstest]
#[case::integer(Leaf::Integer)]
#[case::float(Leaf::Float)]
#[case::vax(Leaf::Vax)]
#[case::time(Leaf::Time)]
#[case::string(Leaf::String)]
#[case::bitfield(Leaf::Bitfield)]
#[case::opaque(Leaf::Opaque(OpaqueTag::Short))]
#[case::opaque_aligned(Leaf::Opaque(OpaqueTag::Aligned))]
#[case::opaque_maximum(Leaf::Opaque(OpaqueTag::Maximum))]
fn datatype_leaf_codecs_preserve_native_file_versions_and_properties(
    #[case] leaf: Leaf,
    #[values(false, true)] latest: bool,
) {
    let _guard = lock::libhdf5_guard();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("datatype.h5");
    let file = hdf5::File::with_options()
        .with_fapl(|properties| {
            let version = if latest {
                LibraryVersion::latest()
            } else {
                LibraryVersion::V18
            };
            properties.libver_bounds(version, version)
        })
        .create(&path)
        .unwrap();
    let datatype = leaf.native();
    let dataset = file
        .new_dataset_builder()
        .empty_as(&datatype)
        .shape(0)
        .create("values")
        .unwrap();
    drop(dataset);
    drop(datatype);
    file.close().unwrap();

    let file = hdf5::File::open(&path).unwrap();
    let dataset = file.dataset("values").unwrap();
    let datatype = dataset.dtype().unwrap();
    let encoded = encode(&datatype);
    drop(datatype);
    drop(dataset);
    file.close().unwrap();
    let body = &encoded[NATIVE_PREFIX_LEN..];
    assert_eq!(body[0] & 0x0f, leaf.class());
    let version = body[0] >> 4;
    if !latest {
        assert_eq!(version, if matches!(leaf, Leaf::Vax) { 3 } else { 1 });
    }
    let (message, consumed) = DatatypeMessage::parse(body).unwrap();
    assert_eq!(message.version().get(), version);
    assert_eq!(message.properties(), &leaf.properties());
    let canonical = message.serialize(OffsetWidth::Eight).unwrap();
    if let Leaf::Opaque(tag) = leaf {
        let mut expected = body[..8].to_vec();
        expected[1] = tag.encoded_len();
        expected.extend(tag.bytes());
        expected.resize(8 + usize::from(tag.encoded_len()), 0);
        assert_eq!(canonical, expected);
        let mut wrapped = encoded[..NATIVE_PREFIX_LEN].to_vec();
        wrapped.extend(canonical);
        let reference = hdf5::sync::sync(|| {
            // SAFETY: The asserted opaque frame has a complete header and terminated property.
            let id = unsafe { h5t::H5Tdecode1(wrapped.as_ptr().cast()) };
            owned(id)
        });
        assert_eq!(encode(&reference), encoded);
    } else {
        assert_eq!(canonical, &body[..consumed]);
    }
    assert_eq!(&body[consumed..], vec![0; body.len() - consumed]);
    let image = std::fs::read(&path).unwrap();
    assert!(
        image
            .windows(consumed)
            .any(|window| window == &body[..consumed])
    );
}

#[derive(Clone, Copy, Debug)]
enum Leaf {
    Integer,
    Float,
    Vax,
    Time,
    String,
    Bitfield,
    Opaque(OpaqueTag),
}

impl Leaf {
    fn native(self) -> NativeDatatype {
        hdf5::sync::sync(|| {
            let id = match self {
                // SAFETY: Each predefined datatype ID is live, and copying transfers a new ID.
                Self::Integer => unsafe { h5t::H5Tcopy(*globals::H5T_STD_I16BE) },
                // SAFETY: The predefined datatype is live and H5Tcopy returns a new owned ID.
                Self::Float => unsafe { h5t::H5Tcopy(*globals::H5T_IEEE_F64BE) },
                // SAFETY: The predefined VAX datatype is live and H5Tcopy returns a new owned ID.
                Self::Vax => unsafe { h5t::H5Tcopy(*globals::H5T_VAX_F64) },
                // SAFETY: The predefined datatype is live and H5Tcopy returns a new owned ID.
                Self::Time => unsafe { h5t::H5Tcopy(*globals::H5T_UNIX_D32BE) },
                // SAFETY: The predefined datatype is live and H5Tcopy returns a new owned ID.
                Self::String => unsafe { h5t::H5Tcopy(*globals::H5T_C_S1) },
                // SAFETY: The predefined datatype is live and H5Tcopy returns a new owned ID.
                Self::Bitfield => unsafe { h5t::H5Tcopy(*globals::H5T_STD_B32BE) },
                // SAFETY: Opaque is a defined pointer-free class and three is a positive size.
                Self::Opaque(_) => unsafe { h5t::H5Tcreate(H5T_class_t::H5T_OPAQUE, 3) },
            };
            let datatype = owned(id);
            match self {
                Self::String => {
                    // SAFETY: The copied datatype is live and mutable. All properties are valid.
                    unsafe {
                        assert_eq!(h5t::H5Tset_size(datatype.id(), 7), 0);
                        assert_eq!(
                            h5t::H5Tset_cset(datatype.id(), H5T_cset_t::H5T_CSET_UTF8),
                            0
                        );
                        assert_eq!(
                            h5t::H5Tset_strpad(datatype.id(), H5T_str_t::H5T_STR_SPACEPAD),
                            0
                        );
                    }
                }
                Self::Opaque(tag) => {
                    let tag = CString::new(tag.bytes()).unwrap();
                    assert_eq!(
                        // SAFETY: The opaque type is live. The tag owns terminated bytes
                        // for this synchronous call.
                        unsafe { h5t::H5Tset_tag(datatype.id(), tag.as_ptr()) },
                        0
                    );
                }
                _ => {}
            }
            datatype
        })
    }

    fn class(self) -> u8 {
        match self {
            Self::Integer => 0,
            Self::Float | Self::Vax => 1,
            Self::Time => 2,
            Self::String => 3,
            Self::Bitfield => 4,
            Self::Opaque(_) => 5,
        }
    }

    fn properties(self) -> DatatypeProperties {
        match self {
            Self::Integer => DatatypeProperties::FixedPoint(
                FixedPointType::new(2, DatatypeByteOrder::BigEndian, true, 0, 16).unwrap(),
            ),
            Self::Float => DatatypeProperties::FloatingPoint(FloatingPointType::ieee754_binary64(
                DatatypeByteOrder::BigEndian,
            )),
            Self::Vax => DatatypeProperties::FloatingPoint(
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
            Self::Time => DatatypeProperties::Time(
                TimeType::new(4, DatatypeByteOrder::BigEndian, 32).unwrap(),
            ),
            Self::String => DatatypeProperties::String {
                size: 7,
                padding: StringPadding::SpacePad,
                charset: CharacterSet::Utf8,
            },
            Self::Bitfield => DatatypeProperties::BitField(
                BitFieldType::new(4, DatatypeByteOrder::BigEndian, 0, 32).unwrap(),
            ),
            Self::Opaque(tag) => DatatypeProperties::Opaque {
                size: 3,
                tag: tag.bytes(),
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum OpaqueTag {
    Aligned,
    Maximum,
    Short,
}

impl OpaqueTag {
    fn bytes(self) -> Vec<u8> {
        match self {
            Self::Short => b"tag".to_vec(),
            Self::Aligned => vec![b't'; 8],
            Self::Maximum => vec![b't'; 247],
        }
    }

    fn encoded_len(self) -> u8 {
        match self {
            Self::Short => 8,
            Self::Aligned => 16,
            Self::Maximum => 248,
        }
    }
}

fn owned(id: hid_t) -> NativeDatatype {
    assert!(id >= 0);
    // SAFETY: Every caller transfers a newly created datatype ID exactly once.
    unsafe { hdf5::from_id::<NativeDatatype>(id) }.unwrap()
}

fn encode(datatype: &NativeDatatype) -> Vec<u8> {
    hdf5::sync::sync(|| {
        let mut length = 0;
        assert_eq!(
            // SAFETY: The datatype is live. A null output requests the required byte count.
            unsafe { h5t::H5Tencode(datatype.id(), core::ptr::null_mut(), &mut length) },
            0
        );
        let mut bytes = vec![0; length];
        assert_eq!(
            // SAFETY: The initialized buffer covers the length reported by the preceding call.
            unsafe { h5t::H5Tencode(datatype.id(), bytes.as_mut_ptr().cast(), &mut length) },
            0
        );
        assert_eq!(length, bytes.len());
        bytes
    })
}

/// The two-byte message kind and encoding version written by `H5T_encode` (`H5T.c`, HDF5 2.2.0).
const NATIVE_PREFIX_LEN: usize = 2;

/// The VAX exponent bias assigned by `H5T_INIT_TYPE_DOUBLEVAX_CORE` (`H5T.c`, HDF5 2.2.0).
const VAX_BIAS: u32 = 0x0401;
