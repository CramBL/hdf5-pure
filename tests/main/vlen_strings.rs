use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::{Error, File, FormatError, VlenStringReadOptions};
use test_util::global_heap;
use test_util::widths::Widths;

use rstest::rstest;
use test_util_hdf5::reader::ReadBackend;

fn expected_names() -> Vec<String> {
    ["Alice", "Bob", "Charlie"]
        .into_iter()
        .map(str::to_owned)
        .collect()
}

#[rstest]
fn vlen_dataset_read_and_size(
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let file = backend
        .open_image(test_util::legacy_attributes::vl_strings())
        .unwrap();
    let dataset = file.dataset("names").unwrap();

    assert_eq!(dataset.vlen_string_payload_size().unwrap(), 15);
    assert_eq!(dataset.read_string().unwrap(), expected_names());
    assert_eq!(
        dataset
            .read_vlen_strings(
                VlenStringReadOptions::new()
                    .with_max_elements(3)
                    .with_max_payload_bytes(15),
            )
            .unwrap(),
        expected_names()
    );
}

#[rstest]
fn vlen_element_limit_is_checked_before_payload_read(
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let file = backend
        .open_image(test_util::legacy_attributes::vl_strings())
        .unwrap();
    let dataset = file.dataset("names").unwrap();
    let error = dataset
        .read_vlen_strings(VlenStringReadOptions::new().with_max_elements(2))
        .unwrap_err();

    assert!(matches!(
        error,
        Error::Format(FormatError::VariableLengthElementLimitExceeded {
            limit: 2,
            actual: 3,
        })
    ));
}

#[rstest]
fn vlen_payload_limit_reports_required_bytes(
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let file = backend
        .open_image(test_util::legacy_attributes::vl_strings())
        .unwrap();
    let dataset = file.dataset("names").unwrap();
    let error = dataset
        .read_vlen_strings(VlenStringReadOptions::new().with_max_payload_bytes(14))
        .unwrap_err();

    assert!(matches!(
        error,
        Error::Format(FormatError::VariableLengthByteLimitExceeded {
            limit: 14,
            required: 15,
        })
    ));
}

#[rstest]
fn vlen_visitor_delivers_strings_in_order(
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
) {
    let file = backend
        .open_image(test_util::legacy_attributes::vl_strings())
        .unwrap();
    let dataset = file.dataset("names").unwrap();
    let mut names = Vec::new();

    dataset
        .visit_vlen_strings(
            VlenStringReadOptions::new().with_max_payload_bytes(15),
            |name| names.push(name.to_owned()),
        )
        .unwrap();

    assert_eq!(names, expected_names());
}

#[test]
fn vlen_specific_apis_reject_non_vlen_datasets() {
    let numeric = File::from_bytes(test_util::legacy::simple_dataset()).unwrap();
    let numeric_dataset = numeric.dataset("data").unwrap();
    assert!(matches!(
        numeric_dataset.vlen_string_payload_size(),
        Err(Error::Format(FormatError::TypeMismatch {
            expected: "VariableLength string",
            actual: "non-VariableLength string",
        }))
    ));
}

// --- Multiple global heap collections (one collection indexes 65,535 objects) ---

/// One collection's object index is a `u16` with 0 reserved for the free-space
/// marker, so the writer splits past 65,535 objects into a second collection
/// whose indices restart at 1. Elements on both sides of that boundary, and the
/// boundary elements themselves, must read back.
fn assert_split_labels_roundtrip(read: &[String], count: usize) {
    assert_eq!(read.len(), count);
    for i in [0, 1, 65_534, 65_535, 65_536, count - 1] {
        assert_eq!(read[i], format!("s{i}"), "element {i} did not round-trip");
    }
}

fn labels(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("s{i}")).collect()
}

/// Just past two collections' worth, so the split is exercised twice and the
/// third collection is only partly filled.
const SPLIT_COUNT: usize = 2 * (u16::MAX as usize) + 7;

#[test]
fn vlen_dataset_spans_multiple_heap_collections() {
    let values = labels(SPLIT_COUNT);
    let refs: Vec<&str> = values.iter().map(String::as_str).collect();
    let mut builder = hdf5_pure::FileBuilder::new();
    builder.create_dataset("labels").with_vlen_strings(&refs);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let read = file.dataset("labels").unwrap().read_string().unwrap();
    assert_split_labels_roundtrip(&read, SPLIT_COUNT);
}

/// The paged writer lays the collections out in its own metadata region and
/// patches the references in a separate pass, so it needs its own coverage.
#[test]
fn paged_vlen_dataset_spans_multiple_heap_collections() {
    let path = test_util::temp::temp_path("paged.h5");
    let values = labels(SPLIT_COUNT);
    let refs: Vec<&str> = values.iter().map(String::as_str).collect();
    let mut builder = hdf5_pure::FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true),
    );
    builder.create_dataset("labels").with_vlen_strings(&refs);
    builder.write(&path).unwrap();

    let file = File::open(&path).unwrap();
    let read = file.dataset("labels").unwrap().read_string().unwrap();
    assert_split_labels_roundtrip(&read, SPLIT_COUNT);
}

/// The edit session stages a dataset through a different pipeline than
/// `FileBuilder` (`flatten_dataset`), and places each collection separately
/// (they need not land contiguously), so it needs its own coverage.
#[test]
fn edit_session_vlen_dataset_spans_multiple_heap_collections() {
    let path = test_util::temp::temp_path("edit.h5");
    let mut builder = hdf5_pure::FileBuilder::new();
    builder.create_dataset("d").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();

    let values = labels(SPLIT_COUNT);
    let refs: Vec<&str> = values.iter().map(String::as_str).collect();
    let file = File::open_rw(&path).unwrap();
    file.root()
        .create_dataset("labels", |b| {
            b.with_vlen_strings(&refs);
        })
        .unwrap();
    file.commit().unwrap();
    drop(file);

    let file = File::open(&path).unwrap();
    let read = file.dataset("labels").unwrap().read_string().unwrap();
    assert_split_labels_roundtrip(&read, SPLIT_COUNT);
    assert_eq!(file.dataset("d").unwrap().read_f64().unwrap(), vec![1.0]);
}

/// Repack re-stages variable-length data through fresh collections, so a file
/// whose data spans several of them must survive the round trip.
#[test]
fn repack_preserves_multi_collection_vlen_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src.h5");
    let dst = dir.path().join("dst.h5");
    let values = labels(SPLIT_COUNT);
    let refs: Vec<&str> = values.iter().map(String::as_str).collect();
    let mut builder = hdf5_pure::FileBuilder::new();
    builder.create_dataset("labels").with_vlen_strings(&refs);
    builder.write(&src).unwrap();

    hdf5_pure::repack(&src, &dst, &hdf5_pure::RepackOptions::default()).unwrap();

    let file = File::open(&dst).unwrap();
    let read = file.dataset("labels").unwrap().read_string().unwrap();
    assert_split_labels_roundtrip(&read, SPLIT_COUNT);
}

/// Sets `per_object` variable-length string attributes of `strings_each` strings
/// on the root, on group `g` and on dataset `g/d`, and returns every value set,
/// keyed by object path and attribute name.
fn set_string_attrs(
    builder: &mut hdf5_pure::FileBuilder,
    per_object: usize,
    strings_each: usize,
) -> Vec<(&'static str, String, Vec<String>)> {
    let mut expected = Vec::new();
    let mut next = 0;
    let mut values = || {
        let values: Vec<String> = (next..next + strings_each)
            .map(|i| format!("s{i}"))
            .collect();
        next += strings_each;
        values
    };
    for i in 0..per_object {
        let value = values();
        builder.set_attr(
            &format!("a{i}"),
            hdf5_pure::AttrValue::VarLenStringArray(value.clone()),
        );
        expected.push(("/", format!("a{i}"), value));
    }
    let mut group = builder.create_group("g");
    for i in 0..per_object {
        let value = values();
        group.set_attr(
            &format!("a{i}"),
            hdf5_pure::AttrValue::VarLenStringArray(value.clone()),
        );
        expected.push(("g", format!("a{i}"), value));
    }
    let dataset = group.create_dataset("d");
    dataset.with_f64_data(&[1.0]);
    for i in 0..per_object {
        let value = values();
        dataset.set_attr(
            &format!("a{i}"),
            hdf5_pure::AttrValue::VarLenStringArray(value.clone()),
        );
        expected.push(("g/d", format!("a{i}"), value));
    }
    builder.add_group(group.finish());
    expected
}

fn assert_string_attrs(file: &File, expected: &[(&str, String, Vec<String>)]) {
    for (path, name, value) in expected {
        let attrs = match *path {
            "/" => file.root().attrs().unwrap(),
            "g" => file.group("g").unwrap().attrs().unwrap(),
            _ => file.dataset(path).unwrap().attrs().unwrap(),
        };
        match attrs.get(name) {
            Some(hdf5_pure::AttrValue::VarLenStringArray(read)) => {
                assert_eq!(read, value, "{path} {name} did not round-trip");
            }
            other => panic!("expected {path} {name} as strings, got {other:?}"),
        }
    }
}

/// Every variable-length string attribute of the file shares one global heap
/// collection, as libhdf5 shares one across objects. A collection is at least
/// 4,096 bytes, so a collection per attribute makes a file of many short string
/// attributes mostly padding.
#[test]
fn string_attributes_share_one_heap_collection() {
    let mut builder = hdf5_pure::FileBuilder::new();
    let expected = set_string_attrs(&mut builder, 2, 3);
    let bytes = builder.finish().unwrap();

    assert_eq!(global_heap::collection_starts(&bytes).len(), 1);
    assert_string_attrs(&File::from_bytes(bytes).unwrap(), &expected);
}

/// The paged writer places the collections in its own pass, so it needs its own
/// coverage.
#[test]
fn paged_string_attributes_share_one_heap_collection() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paged.h5");
    let mut builder = hdf5_pure::FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(1),
    );
    let expected = set_string_attrs(&mut builder, 2, 3);
    builder.write(&path).unwrap();

    assert_eq!(
        global_heap::collection_starts(&std::fs::read(&path).unwrap()).len(),
        1
    );
    assert_string_attrs(&File::open(&path).unwrap(), &expected);
}

/// Repack writes one object at a time, so the strings of each object's
/// attributes share the object's collections.
#[test]
fn repacked_string_attributes_share_a_heap_collection_per_object() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src.h5");
    let dst = dir.path().join("dst.h5");
    let mut builder = hdf5_pure::FileBuilder::new();
    let expected = set_string_attrs(&mut builder, 2, 3);
    builder.write(&src).unwrap();

    hdf5_pure::repack(&src, &dst, &hdf5_pure::RepackOptions::default()).unwrap();

    // The root, `g` and `g/d`.
    assert_eq!(
        global_heap::collection_starts(&std::fs::read(&dst).unwrap()).len(),
        3
    );
    assert_string_attrs(&File::open(&dst).unwrap(), &expected);
}

/// Attribute strings past one collection's 4,096 bytes, the size libhdf5 writes
/// collections at, continue in the next collection, which can start within an
/// attribute: its strings on both sides of that boundary must point into the
/// right collection at the right index. The size bounds what a reader walks to
/// find one string.
#[test]
fn string_attributes_past_one_collection_continue_in_the_next() {
    let mut builder = hdf5_pure::FileBuilder::new();
    let expected = set_string_attrs(&mut builder, 2, 100);
    let bytes = builder.finish().unwrap();

    let collections = global_heap::collection_ranges(&bytes, Widths::EIGHT);
    // 600 strings of 24 bytes each, header included, fill 169 to a collection.
    assert_eq!(collections.len(), 4);
    for collection in &collections {
        assert_eq!(collection.end - collection.start, 4096, "{collection:?}");
    }
    assert_string_attrs(&File::from_bytes(bytes).unwrap(), &expected);
}

/// A variable-length *attribute* outgrows the compact attribute message's
/// 65,535-byte size field long before it reaches any heap-object boundary (16
/// bytes of reference per element binds at ~4,090 elements). Past that field it
/// is stored in a fractal heap rather than refused (issue #102), and its
/// elements still point into as many global-heap collections as they need — the
/// two heaps are independent, which is what this pins.
#[test]
fn a_vlen_attribute_past_the_compact_message_size_is_stored_in_a_heap() {
    let path = test_util::temp::temp_path("ga.h5");
    let mut builder = hdf5_pure::FileBuilder::new();
    builder.create_dataset("d").with_f64_data(&[1.0]);
    builder.write(&path).unwrap();

    let values = labels(u16::MAX as usize + 1);
    {
        let file = File::open_rw(&path).unwrap();
        file.root()
            .set_attr(
                "labels",
                hdf5_pure::AttrValue::VarLenAsciiCharArray(values.clone()),
            )
            .unwrap();
        file.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    match file.root().attrs().unwrap().get("labels") {
        Some(hdf5_pure::AttrValue::VarLenAsciiCharArray(read)) => assert_eq!(*read, values),
        other => panic!("expected the variable-length attribute back, got {other:?}"),
    }
}
