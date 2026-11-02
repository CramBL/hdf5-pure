use std::collections::HashMap;

use hdf5_pure::{AttrValue, Error, File, FormatError, RepackOptions};
use rstest::rstest;
use test_util::attribute_policy::{self, Damage};
use test_util_hdf5::reader::ReadBackend;

#[test]
fn repack_accepts_the_independent_attribute_metadata_control() {
    let source = File::from_bytes(attribute_policy::file()).unwrap();
    assert_eq!(source.root().groups().unwrap(), Vec::<String>::new());
    let expected = HashMap::from([
        ("a".into(), AttrValue::F64(71.0)),
        ("b".into(), AttrValue::F64(73.0)),
    ]);
    assert_eq!(source.root().attrs().unwrap(), expected);
    let mut output = Vec::new();
    hdf5_pure::repack_into(&source, &mut output, &RepackOptions::new()).unwrap();
    source.close().unwrap();
    let file = File::from_bytes(output).unwrap();
    assert_eq!(file.root().attrs().unwrap(), expected);
}

#[rstest]
#[case::policy(Damage::Policy)]
#[case::counter(Damage::Counter)]
#[case::index(Damage::DuplicateIndex)]
#[case::name(Damage::DuplicateName)]
#[case::coherence(Damage::Coherence)]
#[case::index_without_tracking(Damage::IndexWithoutTracking)]
fn repack_rejects_malformed_attribute_metadata_before_output(
    #[case] damage: Damage,
    #[values(
        ReadBackend::Buffered,
        ReadBackend::Bytes,
        ReadBackend::Source,
        ReadBackend::Streaming
    )]
    backend: ReadBackend,
    #[values(false, true)] path_output: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.h5");
    let destination = directory.path().join("output.h5");
    let bytes = attribute_policy::damaged_file(damage);
    std::fs::write(&source, &bytes).unwrap();
    let file = backend.open(&source).unwrap();
    let error = if path_output {
        hdf5_pure::repack_to_path(&file, &destination, &RepackOptions::new()).unwrap_err()
    } else {
        let before = vec![0xCC; 13];
        let mut output = before.clone();
        let error = hdf5_pure::repack_into(&file, &mut output, &RepackOptions::new()).unwrap_err();
        assert_eq!(output, before);
        error
    };
    let expected = match damage {
        Damage::Policy => Some(FormatError::InvalidAttributePhaseChange {
            max_compact: 3,
            min_dense: 4,
        }),
        Damage::Counter => Some(FormatError::InvalidAttributeCreationCounter { index: 5, next: 5 }),
        Damage::DuplicateIndex => Some(FormatError::DuplicateAttributeCreationIndex(2)),
        Damage::DuplicateName => Some(FormatError::DuplicateAttributeName("a".into())),
        Damage::Coherence | Damage::IndexWithoutTracking => None,
    };
    match (error, expected) {
        (Error::Format(actual), Some(expected)) => assert_eq!(actual, expected),
        (Error::RepackUnsupported(reason), None) => assert_eq!(
            reason,
            "an object's attribute creation-order metadata does not match its header prefix"
        ),
        (actual, expected) => panic!("expected metadata error {expected:?}, got {actual:?}"),
    }
    file.close().unwrap();
    assert!(!destination.exists());
    assert_eq!(std::fs::read(source).unwrap(), bytes);
}
