use hdf5_pure::{Error, File, FormatError, VlenStringReadOptions};
use tempfile::tempdir;

const FIXTURE: &str = "tests/data/unattributed/vl_strings.h5";

fn expected_names() -> Vec<String> {
    ["Alice", "Bob", "Charlie"]
        .into_iter()
        .map(str::to_owned)
        .collect()
}

#[test]
fn buffered_vlen_dataset_read_and_size() {
    let file = File::open(FIXTURE).unwrap();
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

#[test]
fn vlen_element_limit_is_checked_before_payload_read() {
    let file = File::open(FIXTURE).unwrap();
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

#[test]
fn vlen_payload_limit_reports_required_bytes() {
    let file = File::open(FIXTURE).unwrap();
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

#[test]
fn vlen_visitor_delivers_strings_in_order() {
    let file = File::open(FIXTURE).unwrap();
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
    let file = File::open(FIXTURE).unwrap();
    let dataset = file.dataset("names").unwrap();
    let datatype = dataset.datatype().unwrap();
    assert!(matches!(
        datatype,
        hdf5_pure::Datatype::VariableLength { .. }
    ));

    let numeric = File::open("tests/data/unattributed/simple_dataset.h5").unwrap();
    let numeric_dataset = numeric.dataset("data").unwrap();
    assert!(matches!(
        numeric_dataset.vlen_string_payload_size(),
        Err(Error::Format(FormatError::TypeMismatch { .. }))
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
    let dir = tempdir().unwrap();
    let path = dir.path().join("paged.h5");
    let values = labels(SPLIT_COUNT);
    let refs: Vec<&str> = values.iter().map(String::as_str).collect();
    let mut builder = hdf5_pure::FileBuilder::new();
    builder.with_file_space_strategy(hdf5_pure::FileSpaceStrategy::Page, true, 1);
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
    let dir = tempdir().unwrap();
    let path = dir.path().join("edit.h5");
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
    let dir = tempdir().unwrap();
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

/// A variable-length *attribute* outgrows the compact attribute message's
/// 65,535-byte size field long before it reaches any heap-object boundary (16
/// bytes of reference per element binds at ~4,090 elements). Past that field it
/// is stored in a fractal heap rather than refused (issue #102), and its
/// elements still point into as many global-heap collections as they need — the
/// two heaps are independent, which is what this pins.
#[test]
fn a_vlen_attribute_past_the_compact_message_size_is_stored_in_a_heap() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ga.h5");
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
