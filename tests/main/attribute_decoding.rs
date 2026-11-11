use std::collections::HashMap;
use std::error::Error as StdError;
use std::io::Cursor;

use hdf5_pure::{AttrValue, EditBacking, Error, File, FileAccessProperties, FormatError};
use hdf5_pure::{CharacterSet, Datatype, DatatypeByteOrder, FloatingPointType, StringPadding};
use hdf5_pure::{MemoryStrategy, ReadSeekSource, Source};
use hdf5_pure::{VariableLengthStringType, VariableLengthType};
use rstest::rstest;
use test_util::attribute_values::{self, Damage, Value};
use test_util_hdf5::reader::ReadBackend;

#[rstest]
#[case::duplicate(Damage::DuplicateIndex)]
#[case::explicit_free_extent(Damage::FreeExtent)]
#[case::allocation_extent(Damage::HeapExtent)]
#[case::signature(Damage::HeapSignature)]
#[case::missing_object(Damage::MissingObject)]
#[case::object_extent(Damage::ObjectExtent)]
#[case::payload_extent(Damage::PayloadExtent)]
#[case::undefined_collection(Damage::UndefinedHeap)]
#[case::wide_index(Damage::WideIndex)]
#[case::zero_index(Damage::ZeroIndex)]
fn attrs_preserves_heap_errors_with_attribute_context(
    #[case] damage: Damage,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
    #[values(false, true)] dataset: bool,
) {
    let image = attribute_values::damaged_file(damage);
    let expected = heap_error(damage, image.len());
    let file = backend.open_image(image).unwrap();
    assert_decode_error(attrs(&file, dataset).unwrap_err(), &expected);
    file.close().unwrap();
}

#[rstest]
#[case::implied_tail(Value::ImpliedTail)]
#[case::null(Value::Null)]
#[case::unsupported(Value::Opaque)]
fn attrs_preserves_values_and_omits_only_unsupported_representations(
    #[case] value: Value,
    #[values(ReadBackend::Bytes, ReadBackend::Source)] backend: ReadBackend,
    #[values(false, true)] dataset: bool,
) {
    let file = backend.open_image(attribute_values::file(value)).unwrap();
    assert_eq!(attrs(&file, dataset).unwrap(), expected_values(value));
    let datatypes = if dataset {
        file.dataset("data").unwrap().attr_datatypes().unwrap()
    } else {
        file.root().attr_datatypes().unwrap()
    };
    let numeric = Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(
        DatatypeByteOrder::LittleEndian,
    ));
    let datatype = match value {
        Value::Opaque => Datatype::Opaque {
            size: 1,
            tag: vec![],
        },
        Value::ImpliedTail | Value::Null => Datatype::VariableLength(VariableLengthType::String(
            VariableLengthStringType::new(StringPadding::NullTerminate, CharacterSet::Utf8),
        )),
        Value::VaxFloat => panic!("a VAX float has no decoded value"),
    };
    assert_eq!(
        datatypes,
        HashMap::from([
            ("before".into(), numeric.clone()),
            ("after".into(), numeric),
            (attribute_values::ATTRIBUTE_NAME.into(), datatype),
        ])
    );
    file.close().unwrap();
}

#[rstest]
fn attrs_preserves_numeric_decoding_errors(
    #[values(ReadBackend::Bytes, ReadBackend::Source)] backend: ReadBackend,
    #[values(false, true)] dataset: bool,
) {
    let file = backend
        .open_image(attribute_values::file(Value::VaxFloat))
        .unwrap();
    assert_decode_error(
        attrs(&file, dataset).unwrap_err(),
        &FormatError::UnsupportedVaxByteOrder,
    );
    file.close().unwrap();
}

#[rstest]
#[case::bounded(MemoryStrategy::Bounded, EditBacking::Bounded)]
#[case::mirrored(MemoryStrategy::Mirrored, EditBacking::Mirrored)]
fn attrs_propagates_failures_and_values_from_edit_backends(
    #[case] strategy: MemoryStrategy,
    #[case] backing: EditBacking,
    #[values(false, true)] dataset: bool,
    #[values(false, true)] damaged: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("attrs.h5");
    let image = if damaged {
        attribute_values::damaged_file(Damage::PayloadExtent)
    } else {
        attribute_values::file(Value::ImpliedTail)
    };
    std::fs::write(&path, image).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(strategy),
    )
    .unwrap();
    assert_eq!(file.edit_backing(), Some(backing));
    if damaged {
        assert_decode_error(
            attrs(&file, dataset).unwrap_err(),
            &heap_error(Damage::PayloadExtent, 0),
        );
    } else {
        assert_eq!(
            attrs(&file, dataset).unwrap(),
            expected_values(Value::ImpliedTail)
        );
    }
    file.close().unwrap();
}

#[rstest]
#[case::collection_header(attribute_values::HEAP as u64)]
#[case::object_header(attribute_values::HEAP as u64 + 16)]
#[case::tail(attribute_values::HEAP as u64 + 4088)]
#[case::payload(attribute_values::PAYLOAD as u64)]
fn attrs_preserves_source_failures(#[case] fail_at: u64, #[values(false, true)] dataset: bool) {
    let image = attribute_values::file(Value::ImpliedTail);
    let file = File::from_source(FailingSource {
        source: ReadSeekSource::new(Cursor::new(image)).unwrap(),
        fail_at,
    })
    .unwrap();
    assert_decode_error(
        attrs(&file, dataset).unwrap_err(),
        &FormatError::Source(format!("backend read failed at {fail_at}")),
    );
    file.close().unwrap();
}

fn attrs(file: &File, dataset: bool) -> Result<HashMap<String, AttrValue>, Error> {
    if dataset {
        file.dataset("data").unwrap().attrs()
    } else {
        file.root().attrs()
    }
}

fn expected_values(value: Value) -> HashMap<String, AttrValue> {
    let mut expected = HashMap::from([
        ("before".into(), AttrValue::F64(7.0)),
        ("after".into(), AttrValue::F64(9.0)),
    ]);
    match value {
        Value::ImpliedTail => {
            expected.insert(
                attribute_values::ATTRIBUTE_NAME.into(),
                AttrValue::VarLenString("x".repeat(attribute_values::PAYLOAD_LEN)),
            );
        }
        Value::Null => {
            expected.insert(
                attribute_values::ATTRIBUTE_NAME.into(),
                AttrValue::VarLenString(String::new()),
            );
        }
        Value::Opaque => {}
        Value::VaxFloat => panic!("a VAX float has no decoded value"),
    }
    expected
}

fn heap_error(damage: Damage, image_len: usize) -> FormatError {
    match damage {
        Damage::DuplicateIndex => FormatError::DuplicateGlobalHeapObjectIndex {
            collection_address: 64,
            index: 1,
        },
        Damage::FreeExtent => FormatError::MisplacedGlobalHeapFreeSpace {
            address: 104,
            size: 4048,
            collection_end: 4160,
        },
        Damage::HeapExtent => FormatError::UnexpectedSourceEof {
            expected: 8256,
            available: image_len as u64,
        },
        Damage::HeapSignature => FormatError::InvalidGlobalHeapSignature,
        Damage::MissingObject => FormatError::GlobalHeapObjectNotFound {
            collection_address: 64,
            index: 2,
        },
        Damage::ObjectExtent => FormatError::GlobalHeapObjectOutsideCollection {
            address: 80,
            index: 1,
            collection_end: 4160,
        },
        Damage::PayloadExtent => {
            FormatError::VlDataError("a 4057-byte read exceeds global heap object size 4056".into())
        }
        Damage::UndefinedHeap => {
            FormatError::VlDataError("non-empty VL element has an undefined heap address".into())
        }
        Damage::WideIndex => {
            FormatError::VlDataError("global heap object index 65536 does not fit u16".into())
        }
        Damage::ZeroIndex => FormatError::GlobalHeapObjectNotFound {
            collection_address: 64,
            index: 0,
        },
    }
}

fn assert_decode_error(error: Error, expected: &FormatError) {
    let Error::AttributeDecode { name, source } = &error else {
        panic!("expected AttributeDecode, got {error:?}");
    };
    assert_eq!(name, attribute_values::ATTRIBUTE_NAME);
    assert_eq!(source, expected);
    assert_eq!(
        error.to_string(),
        format!(
            "cannot decode attribute {:?}: {expected}",
            attribute_values::ATTRIBUTE_NAME
        )
    );
    assert_eq!(
        error
            .source()
            .unwrap()
            .downcast_ref::<FormatError>()
            .unwrap(),
        expected
    );
}

struct FailingSource {
    source: ReadSeekSource<Cursor<Vec<u8>>>,
    fail_at: u64,
}

impl Source for FailingSource {
    fn len(&self) -> u64 {
        self.source.len()
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
        if offset == self.fail_at {
            Err(FormatError::Source(format!(
                "backend read failed at {}",
                self.fail_at
            )))
        } else {
            self.source.read_at(offset, buffer)
        }
    }
}
