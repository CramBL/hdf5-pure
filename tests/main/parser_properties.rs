//! Parser no-panic properties for arbitrary bytes and mutations of a generated file.
//!
//! Dataset reads and group opens are limited to immediate root children. Proptest records
//! regression seeds in `tests/proptest-regressions/parser_properties.txt`.

use hdf5_pure::DeflateLevel;
use hdf5_pure::{File, FileBuilder, Group};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;

// Independent signature bytes for parsing beyond signature detection.
const HDF5_SIGNATURE: [u8; 8] = [0x89, b'H', b'D', b'F', b'\r', b'\n', 0x1A, b'\n'];

/// Exercises root children without recursive traversal, allowing parser errors.
fn exercise(file: &File) {
    let root: Group = file.root();
    if let Ok(names) = root.datasets() {
        for name in names.iter().take(16) {
            if let Ok(ds) = root.dataset(name) {
                ds.shape().ok();
                ds.dtype().ok();
                ds.read_f64().ok();
                ds.read_i64().ok();
                ds.read_u8().ok();
                ds.read_string().ok();
            }
        }
    }
    if let Ok(groups) = root.groups() {
        for name in groups.iter().take(16) {
            root.group(name).ok();
        }
    }
}

/// Builds a root attribute, a contiguous dataset, and a grouped deflate-compressed dataset.
fn valid_sample_file() -> Vec<u8> {
    let mut b = FileBuilder::new();
    b.set_attr("version", hdf5_pure::AttrValue::I64(2));
    b.create_dataset("contig")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0])
        .with_shape(&[2, 2]);
    let mut grp = b.create_group("g");
    grp.create_dataset("chunked")
        .with_i32_data(&[1, 2, 3, 4, 5, 6])
        .with_shape(&[6])
        .with_chunks(&[3])
        .with_deflate(DeflateLevel::Four);
    let finished = grp.finish();
    b.add_group(finished);
    b.finish().expect("sample file must build")
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(FileFailurePersistence::Direct(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/proptest-regressions/parser_properties.txt"
        )))),
        ..ProptestConfig::default()
    })]

    #[test]
    fn open_arbitrary_bytes_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..4096)) {
        if let Ok(file) = File::from_bytes(bytes) {
            exercise(&file);
        }
    }

    #[test]
    fn open_signature_prefixed_bytes_never_panics(tail in prop::collection::vec(any::<u8>(), 0..4096)) {
        let mut bytes = HDF5_SIGNATURE.to_vec();
        bytes.extend_from_slice(&tail);
        if let Ok(file) = File::from_bytes(bytes) {
            exercise(&file);
        }
    }

    #[test]
    fn corrupted_valid_file_never_panics(
        muts in prop::collection::vec((any::<usize>(), any::<u8>()), 0..48),
        truncate in 0usize..=96,
    ) {
        let mut bytes = valid_sample_file();
        let len = bytes.len();
        for (idx, val) in muts {
            bytes[idx % len] = val;
        }
        bytes.truncate(len.saturating_sub(truncate));
        if let Ok(file) = File::from_bytes(bytes) {
            exercise(&file);
        }
    }
}
