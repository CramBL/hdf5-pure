//! Native observations of opaque tags, element bytes, and datatype sharing.

#![cfg(feature = "hdf5")]

use std::ffi::CStr;
use std::ffi::CString;
use std::path::Path;

use hdf5::Datatype as NativeDatatype;
use hdf5_pure::CompoundTypeBuilder;
use hdf5_pure::Datatype;
use hdf5_pure::Error;
use hdf5_pure::FileBuilder;
use hdf5_pure::FormatError;
use hdf5_pure::LibVer;
use hdf5_pure::RepackOptions;
use hdf5_sys::h5;
use hdf5_sys::h5a;
use hdf5_sys::h5d;
use hdf5_sys::h5i::hid_t;
use hdf5_sys::h5p::H5P_DEFAULT;
use hdf5_sys::h5s::H5S_ALL;
use hdf5_sys::h5t;
use hdf5_sys::h5t::H5T_class_t;
use rstest::rstest;
use test_util_hdf5::file;
use test_util_hdf5::lock;
use test_util_hdf5::reader::ReadBackend;

#[rstest]
#[case::pure_inline(Producer::Pure, false)]
#[case::pure_committed(Producer::Pure, true)]
#[case::native_inline(Producer::Native, false)]
#[case::native_committed(Producer::Native, true)]
fn opaque_files_preserve_tags_values_and_shared_types_through_repack(
    #[case] producer: Producer,
    #[case] committed: bool,
    #[values(Shape::Opaque, Shape::Array, Shape::Compound)] shape: Shape,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let _guard = lock::libhdf5_guard();
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.h5");
    let destination = directory.path().join("repacked.h5");
    producer.write(&source, shape, b"BLOB", committed);
    observe_pure(
        &source,
        shape,
        b"BLOB",
        committed,
        producer.has_attribute(),
        backend,
    );
    observe_native(&source, shape, b"BLOB", committed, producer.has_attribute());
    hdf5_pure::repack(&source, &destination, &RepackOptions::new()).unwrap();
    observe_pure(
        &destination,
        shape,
        b"BLOB",
        committed,
        producer.has_attribute(),
        backend,
    );
    observe_native(
        &destination,
        shape,
        b"BLOB",
        committed,
        producer.has_attribute(),
    );
}

#[rstest]
#[case::absent(Vec::new(), 0)]
#[case::blob(b"BLOB".to_vec(), 8)]
#[case::full_property(b"12345678".to_vec(), 8)]
#[case::non_ascii(vec![0xff], 8)]
#[case::largest_canonical(vec![b'x'; 247], 248)]
fn native_opaque_tags_expose_logical_bytes_and_reencode_canonically(
    #[case] tag: Vec<u8>,
    #[case] native_property_length: u8,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let _guard = lock::libhdf5_guard();
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("native.h5");
    let destination = directory.path().join("canonical.h5");
    let native = native_opaque(&tag, 4);
    let encoded = encode_native(&native);
    assert_eq!(encoded[ENCODED_DATATYPE_PREFIX_LEN] & 0x0f, 5);
    let body = &encoded[ENCODED_DATATYPE_PREFIX_LEN..];
    assert_eq!(&body[1..4], &[native_property_length, 0, 0]);
    assert_eq!(body.len(), 8 + usize::from(native_property_length));
    assert_eq!(&body[8..8 + tag.len()], tag);
    assert_eq!(&body[8 + tag.len()..], vec![0; body.len() - 8 - tag.len()]);
    drop(native);
    Producer::Native.write(&source, Shape::Opaque, &tag, false);
    observe_pure(&source, Shape::Opaque, &tag, false, true, backend);
    hdf5_pure::repack(&source, &destination, &RepackOptions::new()).unwrap();
    observe_pure(&destination, Shape::Opaque, &tag, false, true, backend);
    observe_native(&destination, Shape::Opaque, &tag, false, true);
    let bytes = std::fs::read(&destination).unwrap();
    let expected = if tag.is_empty() {
        vec![0x15, 0, 0, 0, 4, 0, 0, 0]
    } else {
        let length = (tag.len() + 1).next_multiple_of(8);
        let mut expected = vec![0x15, u8::try_from(length).unwrap(), 0, 0, 4, 0, 0, 0];
        expected.extend_from_slice(&tag);
        expected.resize(8 + length, 0);
        expected
    };
    assert!(
        bytes
            .windows(expected.len())
            .any(|window| window == expected)
    );
}

#[rstest]
#[case::dataset(OversizedLocation::Dataset)]
#[case::attribute(OversizedLocation::Attribute)]
#[case::committed(OversizedLocation::Committed)]
fn native_full_248_byte_tags_read_losslessly_and_repack_returns_an_error(
    #[case] location: OversizedLocation,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let _guard = lock::libhdf5_guard();
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("native.h5");
    let destination = directory.path().join("repacked.h5");
    let tag = vec![b'x'; 248];
    location.write(&source, &tag);
    {
        let file = backend.open(&source).unwrap();
        let dataset = file.dataset("values").unwrap();
        let expected = Shape::Opaque.pure_type(&tag);
        match location {
            OversizedLocation::Dataset | OversizedLocation::Committed => {
                assert_eq!(dataset.datatype().unwrap(), expected);
            }
            OversizedLocation::Attribute => {
                assert_eq!(
                    dataset.attr_datatypes().unwrap(),
                    [("tagged".into(), expected)].into()
                );
            }
        }
        assert_eq!(dataset.read_raw().unwrap(), Shape::Opaque.values());
        drop(dataset);
        file.close().unwrap();
    }
    let before = std::fs::read(&source).unwrap();
    let error = hdf5_pure::repack(&source, &destination, &RepackOptions::new()).unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected an opaque encoding error, got {error:?}");
    };
    assert_eq!(error, FormatError::OpaqueTagTooLong { length: 248 });
    assert!(!destination.exists());
    assert_eq!(std::fs::read(&source).unwrap(), before);
}

#[derive(Clone, Copy)]
enum Producer {
    Native,
    Pure,
}

impl Producer {
    fn write(self, path: &Path, shape: Shape, tag: &[u8], committed: bool) {
        match self {
            Self::Native => {
                let file = file::libhdf5_create_v18(path);
                let datatype = shape.native_type(tag);
                if committed {
                    file.commit_datatype("opaque", &datatype).unwrap();
                }
                let dataset = file
                    .new_dataset_builder()
                    .empty_as(&datatype)
                    .shape(shape.count())
                    .create("values")
                    .unwrap();
                write_dataset(&dataset, &datatype, &shape.values());
                let attribute = dataset
                    .new_attr_builder()
                    .empty_as(&datatype)
                    .shape(shape.count())
                    .create("tagged")
                    .unwrap();
                write_attribute(&attribute, &datatype, &shape.values());
                drop(attribute);
                drop(dataset);
                drop(datatype);
                file.close().unwrap();
            }
            Self::Pure => {
                let datatype = shape.pure_type(tag);
                let mut builder = FileBuilder::new();
                builder.with_libver_bounds(LibVer::Earliest, LibVer::V18);
                if committed {
                    builder.commit_datatype("opaque", datatype.clone());
                }
                let dataset = builder.create_dataset("values");
                dataset
                    .with_raw_data(
                        datatype,
                        shape.values(),
                        u64::try_from(shape.count()).unwrap(),
                    )
                    .with_shape(&[u64::try_from(shape.count()).unwrap()]);
                if committed {
                    dataset.with_committed_datatype("opaque");
                }
                builder.write(path).unwrap();
            }
        }
    }

    fn has_attribute(self) -> bool {
        matches!(self, Self::Native)
    }
}

#[derive(Clone, Copy)]
enum Shape {
    Array,
    Compound,
    Opaque,
}

impl Shape {
    fn pure_type(self, tag: &[u8]) -> Datatype {
        let base = Datatype::Opaque {
            size: 4,
            tag: tag.to_vec(),
        };
        match self {
            Self::Array => Datatype::Array {
                base_type: Box::new(base),
                dimensions: vec![2],
            },
            Self::Compound => CompoundTypeBuilder::with_size(5)
                .field("blob", 0, base)
                .field(
                    "next",
                    4,
                    Datatype::Opaque {
                        size: 1,
                        tag: b"NEXT".to_vec(),
                    },
                )
                .build()
                .unwrap(),
            Self::Opaque => base,
        }
    }

    fn native_type(self, tag: &[u8]) -> NativeDatatype {
        let base = native_opaque(tag, 4);
        hdf5::sync::sync(|| match self {
            Self::Array => {
                // SAFETY: The live base type and one initialized dimension cover the call.
                owned_type(unsafe { h5t::H5Tarray_create2(base.id(), 1, [2].as_ptr()) })
            }
            Self::Compound => {
                // SAFETY: The library lock covers creating a five-byte compound datatype.
                let datatype = owned_type(unsafe { h5t::H5Tcreate(H5T_class_t::H5T_COMPOUND, 5) });
                let next = native_opaque(b"NEXT", 1);
                // SAFETY: Live handles and terminated names cover the calls. Both complete fields
                // occupy disjoint ranges within the five-byte compound.
                unsafe {
                    assert_eq!(
                        h5t::H5Tinsert(datatype.id(), c"blob".as_ptr(), 0, base.id()),
                        0
                    );
                    assert_eq!(
                        h5t::H5Tinsert(datatype.id(), c"next".as_ptr(), 4, next.id()),
                        0
                    );
                }
                datatype
            }
            Self::Opaque => base,
        })
    }

    fn count(self) -> usize {
        match self {
            Self::Array => 1,
            Self::Compound | Self::Opaque => 2,
        }
    }

    fn values(self) -> Vec<u8> {
        match self {
            Self::Array | Self::Opaque => vec![0, 1, 2, 255, 8, 7, 6, 5],
            Self::Compound => vec![0, 1, 2, 255, 42, 8, 7, 6, 5, 99],
        }
    }

    fn assert_native_type(self, datatype: &NativeDatatype, tag: &[u8]) {
        match self {
            Self::Opaque => assert_eq!(native_tag(datatype), tag),
            Self::Array => hdf5::sync::sync(|| {
                // SAFETY: The live datatype is an array, and the returned type receives ownership.
                let base = owned_type(unsafe { h5t::H5Tget_super(datatype.id()) });
                assert_eq!(base.size(), 4);
                assert_eq!(native_tag(&base), tag);
            }),
            Self::Compound => hdf5::sync::sync(|| {
                // SAFETY: The two member indexes exist in the live compound datatype.
                let blob = owned_type(unsafe { h5t::H5Tget_member_type(datatype.id(), 0) });
                // SAFETY: The second member exists and the returned handle receives ownership.
                let next = owned_type(unsafe { h5t::H5Tget_member_type(datatype.id(), 1) });
                assert_eq!(blob.size(), 4);
                assert_eq!(next.size(), 1);
                assert_eq!(native_tag(&blob), tag);
                assert_eq!(native_tag(&next), b"NEXT");
                // SAFETY: Both member indexes exist in the live compound datatype.
                unsafe {
                    assert_eq!(h5t::H5Tget_member_offset(datatype.id(), 0), 0);
                    assert_eq!(h5t::H5Tget_member_offset(datatype.id(), 1), 4);
                }
            }),
        }
    }
}

#[derive(Clone, Copy)]
enum OversizedLocation {
    Attribute,
    Committed,
    Dataset,
}

impl OversizedLocation {
    fn write(self, path: &Path, tag: &[u8]) {
        if matches!(self, Self::Dataset | Self::Committed) {
            Producer::Native.write(path, Shape::Opaque, tag, matches!(self, Self::Committed));
        } else {
            Producer::Pure.write(path, Shape::Opaque, b"BLOB", false);
            let file = hdf5::File::open_rw(path).unwrap();
            let dataset = file.dataset("values").unwrap();
            let datatype = native_opaque(tag, 4);
            let attribute = dataset
                .new_attr_builder()
                .empty_as(&datatype)
                .shape(2)
                .create("tagged")
                .unwrap();
            write_attribute(&attribute, &datatype, &Shape::Opaque.values());
            drop(attribute);
            drop(datatype);
            drop(dataset);
            file.close().unwrap();
        }
    }
}

fn observe_pure(
    path: &Path,
    shape: Shape,
    tag: &[u8],
    committed: bool,
    attribute: bool,
    backend: ReadBackend,
) {
    let file = backend.open(path).unwrap();
    let dataset = file.dataset("values").unwrap();
    let expected = shape.pure_type(tag);
    assert_eq!(dataset.datatype().unwrap(), expected);
    assert_eq!(
        dataset.shape().unwrap(),
        [u64::try_from(shape.count()).unwrap()]
    );
    assert_eq!(
        dataset.element_size().unwrap(),
        u64::from(expected.type_size())
    );
    assert_eq!(dataset.read_raw().unwrap(), shape.values());
    if attribute {
        assert_eq!(
            dataset.attr_datatypes().unwrap(),
            [("tagged".into(), expected.clone())].into()
        );
    }
    if committed {
        assert_eq!(file.root().named_datatype("opaque").unwrap(), expected);
    }
    drop(dataset);
    file.close().unwrap();
}

fn observe_native(path: &Path, shape: Shape, tag: &[u8], committed: bool, attribute: bool) {
    let file = hdf5::File::open(path).unwrap();
    let dataset = file.dataset("values").unwrap();
    let datatype = dataset.dtype().unwrap();
    assert_eq!(datatype.is_committed(), committed);
    shape.assert_native_type(&datatype, tag);
    assert_eq!(dataset.shape(), [shape.count()]);
    let mut values = vec![0; datatype.size() * shape.count()];
    hdf5::sync::sync(|| {
        assert_eq!(
            // SAFETY: The output covers the complete dataset extent in its own pointer-free datatype.
            unsafe {
                h5d::H5Dread(
                    dataset.id(),
                    datatype.id(),
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    values.as_mut_ptr().cast(),
                )
            },
            0
        );
    });
    assert_eq!(values, shape.values());
    if attribute {
        let attribute = dataset.attr("tagged").unwrap();
        let attribute_type = attribute.dtype().unwrap();
        assert_eq!(attribute_type.is_committed(), committed);
        assert_eq!(attribute_type, datatype);
        shape.assert_native_type(&attribute_type, tag);
        assert_eq!(attribute.shape(), [shape.count()]);
        values.fill(0);
        hdf5::sync::sync(|| {
            assert_eq!(
                // SAFETY: The matching attribute extent and datatype cover all output bytes.
                unsafe {
                    h5a::H5Aread(
                        attribute.id(),
                        attribute_type.id(),
                        values.as_mut_ptr().cast(),
                    )
                },
                0
            );
        });
        assert_eq!(values, shape.values());
    }
    drop(datatype);
    drop(dataset);
    file.close().unwrap();
}

fn native_opaque(tag: &[u8], size: usize) -> NativeDatatype {
    let tag = CString::new(tag).unwrap();
    hdf5::sync::sync(|| {
        // SAFETY: The library lock covers creating a pointer-free opaque datatype.
        let datatype = owned_type(unsafe { h5t::H5Tcreate(H5T_class_t::H5T_OPAQUE, size) });
        // SAFETY: The datatype is live and `tag` is a terminated string for the synchronous call.
        assert_eq!(unsafe { h5t::H5Tset_tag(datatype.id(), tag.as_ptr()) }, 0);
        datatype
    })
}

fn owned_type(id: hid_t) -> NativeDatatype {
    assert!(id >= 0);
    // SAFETY: Callers transfer ownership of a newly returned datatype ID exactly once.
    unsafe { hdf5::from_id::<NativeDatatype>(id) }.unwrap()
}

fn native_tag(datatype: &NativeDatatype) -> Vec<u8> {
    hdf5::sync::sync(|| {
        // SAFETY: The live opaque datatype and lock cover obtaining a terminated HDF5 allocation.
        let tag = unsafe { h5t::H5Tget_tag(datatype.id()) };
        assert!(!tag.is_null());
        // SAFETY: H5Tget_tag returns a NUL-terminated allocation live until H5free_memory below.
        let bytes = unsafe { CStr::from_ptr(tag) }.to_bytes().to_vec();
        // SAFETY: This allocation came from H5Tget_tag and is released exactly once.
        assert_eq!(unsafe { h5::H5free_memory(tag.cast()) }, 0);
        bytes
    })
}

fn write_dataset(dataset: &hdf5::Dataset, datatype: &NativeDatatype, values: &[u8]) {
    assert_eq!(dataset.size() * datatype.size(), values.len());
    hdf5::sync::sync(|| {
        assert_eq!(
            // SAFETY: The initialized bytes cover the entire dataset in its pointer-free stored type.
            unsafe {
                h5d::H5Dwrite(
                    dataset.id(),
                    datatype.id(),
                    H5S_ALL,
                    H5S_ALL,
                    H5P_DEFAULT,
                    values.as_ptr().cast(),
                )
            },
            0
        );
    });
}

fn write_attribute(attribute: &hdf5::Attribute, datatype: &NativeDatatype, values: &[u8]) {
    assert_eq!(attribute.size() * datatype.size(), values.len());
    hdf5::sync::sync(|| {
        assert_eq!(
            // SAFETY: The initialized bytes cover the entire attribute in its pointer-free stored type.
            unsafe { h5a::H5Awrite(attribute.id(), datatype.id(), values.as_ptr().cast()) },
            0
        );
    });
}

fn encode_native(datatype: &NativeDatatype) -> Vec<u8> {
    hdf5::sync::sync(|| {
        let mut length = 0;
        assert_eq!(
            // SAFETY: The live datatype and null output pointer request only the encoded length.
            unsafe { h5t::H5Tencode(datatype.id(), core::ptr::null_mut(), &mut length) },
            0
        );
        let mut bytes = vec![0; length];
        assert_eq!(
            // SAFETY: The initialized buffer covers the complete length just reported for this type.
            unsafe { h5t::H5Tencode(datatype.id(), bytes.as_mut_ptr().cast(), &mut length) },
            0
        );
        assert_eq!(length, bytes.len());
        bytes
    })
}

/// The number of bytes before the datatype message in a native encoded datatype.
///
/// `H5T_encode` writes the message type and encoding version before the message
/// (H5T.c, HDF5 2.2.0).
const ENCODED_DATATYPE_PREFIX_LEN: usize = 2;
