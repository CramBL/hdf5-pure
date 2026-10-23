//! Public-API tests for provenance verification (`Dataset::verify_provenance`).
#![cfg(feature = "provenance")]

#[cfg(any(feature = "deflate", feature = "fast-deflate"))]
use hdf5_pure::DeflateLevel;
use hdf5_pure::Error;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FormatError;
use hdf5_pure::Object;
use hdf5_pure::VlenStringReadOptions;
use hdf5_pure::{AttrValue, File, FileBuilder, VerifyResult};
use rstest::rstest;
use sha2::Digest;
use sha2::Sha256;

// The attribute name is an internal constant; mirror it here so the test
// exercises the crate strictly through its public surface.
const ATTR_SHA256: &str = "_provenance_sha256";

#[rstest]
#[case(AddressData::ContiguousStrings)]
#[case(AddressData::ChunkedStrings)]
#[cfg_attr(
    any(feature = "deflate", feature = "fast-deflate"),
    case(AddressData::DeflatedStrings)
)]
#[case(AddressData::PathReferences)]
#[case(AddressData::ChunkedPathReferences)]
#[case(AddressData::ResolvedReferences)]
#[cfg_attr(
    any(feature = "deflate", feature = "fast-deflate"),
    case(AddressData::ShuffledDeflatedStrings)
)]
fn provenance_hashes_final_unfiltered_element_bytes(
    #[case] data: AddressData,
    #[values(false, true)] page: bool,
    #[values(false, true)] dense: bool,
    #[values(0, 4096)] userblock: u64,
    #[values(Output::Finish, Output::Sink, Output::Write)] output: Output,
) {
    let build = |address| address_data_builder(data, page, dense, userblock, address);
    let builder = if matches!(data, AddressData::ResolvedReferences) {
        let probe = File::from_bytes(build(0).finish().unwrap()).unwrap();
        let raw = probe.dataset("probe").unwrap().read_raw().unwrap();
        build(u64::from_le_bytes(raw.try_into().unwrap()))
    } else {
        build(0)
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("provenance.h5");
    let bytes = match output {
        Output::Finish => builder.finish().unwrap(),
        Output::Sink => {
            let mut bytes = Vec::new();
            builder.finish_to(&mut bytes).unwrap();
            bytes
        }
        Output::Write => {
            builder.write(&path).unwrap();
            std::fs::read(&path).unwrap()
        }
    };
    if userblock != 0 {
        assert_eq!(&bytes[..10], b"provenance");
    }
    assert_eq!(test_util::fractal_heap::has_fractal_heap(&bytes), dense);
    let file = match output {
        Output::Write => File::open(&path).unwrap(),
        Output::Finish | Output::Sink => File::from_bytes(bytes).unwrap(),
    };
    let dataset = file.dataset("data").unwrap();
    if matches!(
        data,
        AddressData::PathReferences
            | AddressData::ChunkedPathReferences
            | AddressData::ResolvedReferences
    ) {
        let values: Vec<_> = dataset
            .dereference()
            .unwrap()
            .into_iter()
            .map(|object| {
                let Object::Dataset(target) = object else {
                    panic!("expected a dataset target, got {object:?}");
                };
                target.read_i32().unwrap()
            })
            .collect();
        assert_eq!(values, [vec![7], vec![7]]);
    } else {
        assert_eq!(
            dataset
                .read_vlen_strings(VlenStringReadOptions::default())
                .unwrap(),
            ["alpha", "beta", "gamma"]
        );
    }
    let attrs = dataset.attrs().unwrap();
    assert_eq!(attrs.len(), if dense { 13 } else { 5 });
    assert_eq!(attrs["note"], AttrValue::VarLenString("preserved".into()));
    assert_eq!(
        attrs["_provenance_creator"],
        AttrValue::String("test-suite".into())
    );
    assert_eq!(
        attrs["_provenance_timestamp"],
        AttrValue::String("2026-10-26T00:00:00Z".into())
    );
    assert_eq!(
        attrs["_provenance_source"],
        AttrValue::String("fixture".into())
    );
    assert_eq!(
        attrs[ATTR_SHA256],
        AttrValue::String(
            Sha256::digest(dataset.read_raw().unwrap())
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect()
        )
    );
    assert_eq!(dataset.verify_provenance().unwrap(), VerifyResult::Ok);
}

fn address_data_builder(
    data: AddressData,
    page: bool,
    dense: bool,
    userblock: u64,
    address: u64,
) -> FileBuilder {
    let mut builder = FileBuilder::new();
    if page {
        builder.with_file_space_strategy(FileSpaceStrategy::Page, true, 1);
    }
    builder.with_userblock(userblock);
    if userblock != 0 {
        builder.with_userblock_content(b"provenance");
    }
    builder.create_dataset("target").with_i32_data(&[7]);
    builder
        .create_dataset("probe")
        .with_path_references(&["target"]);
    let dataset = builder.create_dataset("data");
    match data {
        AddressData::ChunkedPathReferences => dataset
            .with_path_references(&["target", "target"])
            .with_chunks(&[1]),
        AddressData::ChunkedStrings => dataset
            .with_vlen_strings(&["alpha", "beta", "gamma"])
            .with_chunks(&[2]),
        AddressData::ContiguousStrings => dataset.with_vlen_strings(&["alpha", "beta", "gamma"]),
        #[cfg(any(feature = "deflate", feature = "fast-deflate"))]
        AddressData::DeflatedStrings => dataset
            .with_vlen_strings(&["alpha", "beta", "gamma"])
            .with_chunks(&[2])
            .with_deflate(DeflateLevel::Six),
        AddressData::PathReferences => dataset.with_path_references(&["target", "target"]),
        AddressData::ResolvedReferences => dataset
            .with_reference_data(&[address, address])
            .with_chunks(&[1]),
        #[cfg(any(feature = "deflate", feature = "fast-deflate"))]
        AddressData::ShuffledDeflatedStrings => dataset
            .with_vlen_strings(&["alpha", "beta", "gamma"])
            .with_chunks(&[2])
            .with_shuffle()
            .with_deflate(DeflateLevel::Six),
    }
    .set_attr("note", AttrValue::VarLenString("preserved".into()))
    .with_provenance("test-suite", "2026-10-26T00:00:00Z", Some("fixture"));
    if dense {
        for index in 0..8 {
            dataset.set_attr(&format!("extra{index}"), AttrValue::I32(index));
        }
    }
    builder
}

#[derive(Clone, Copy, Debug)]
enum AddressData {
    ChunkedPathReferences,
    ChunkedStrings,
    ContiguousStrings,
    #[cfg(any(feature = "deflate", feature = "fast-deflate"))]
    DeflatedStrings,
    PathReferences,
    ResolvedReferences,
    #[cfg(any(feature = "deflate", feature = "fast-deflate"))]
    ShuffledDeflatedStrings,
}

#[derive(Clone, Copy, Debug)]
enum Output {
    Finish,
    Sink,
    Write,
}

#[rstest]
#[case(&[])]
#[case(&[""])]
#[case(&["", "", ""])]
fn empty_variable_length_values_verify(
    #[case] values: &[&str],
    #[values(false, true)] page: bool,
    #[values(false, true)] chunked: bool,
) {
    let mut builder = FileBuilder::new();
    if page {
        builder.with_file_space_strategy(FileSpaceStrategy::Page, true, 1);
    }
    let dataset = builder.create_dataset("data").with_vlen_strings(values);
    if chunked {
        dataset.with_chunks(&[2]);
    }
    dataset.with_provenance("test-suite", "2026-10-26T00:00:00Z", None);
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let dataset = file.dataset("data").unwrap();
    assert_eq!(
        dataset
            .read_vlen_strings(VlenStringReadOptions::default())
            .unwrap(),
        values
    );
    assert_eq!(dataset.verify_provenance().unwrap(), VerifyResult::Ok);
}

#[rstest]
#[case(true, u64::MAX)]
#[case(false, 0)]
#[case(false, u64::MAX)]
fn provenance_preserves_missing_and_undefined_reference_values(
    #[case] path: bool,
    #[case] expected: u64,
    #[values(false, true)] page: bool,
) {
    let mut builder = FileBuilder::new();
    if page {
        builder.with_file_space_strategy(FileSpaceStrategy::Page, true, 1);
    }
    let dataset = builder.create_dataset("data");
    if path {
        dataset.with_path_references(&["missing"]);
    } else {
        dataset.with_reference_data(&[expected]);
    }
    dataset.with_provenance("test-suite", "2026-10-26T00:00:00Z", None);
    let file = File::from_bytes(builder.finish().unwrap()).unwrap();
    let dataset = file.dataset("data").unwrap();
    assert_eq!(dataset.read_raw().unwrap(), expected.to_le_bytes());
    assert_eq!(dataset.verify_provenance().unwrap(), VerifyResult::Ok);
    let error = dataset.dereference().unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, FormatError::InvalidObjectReference(expected));
}

#[test]
fn verify_ok_on_roundtrip() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("sensor")
        .with_f64_data(&(0..24).map(|v| v as f64).collect::<Vec<_>>())
        .with_provenance("test-suite", "2026-02-19T12:00:00Z", None);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("sensor").unwrap();
    assert_eq!(ds.verify_provenance().unwrap(), VerifyResult::Ok);
}

#[test]
fn verify_no_hash_without_provenance() {
    let mut builder = FileBuilder::new();
    builder.create_dataset("plain").with_f64_data(&[1.0, 2.0]);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("plain").unwrap();
    assert_eq!(ds.verify_provenance().unwrap(), VerifyResult::NoHash);
}

#[test]
fn verify_mismatch_on_wrong_stored_hash() {
    // Store a deliberately bogus hash directly (rather than via
    // `with_provenance`, which would compute the correct one) so the recomputed
    // digest is guaranteed to differ.
    let bogus = "0".repeat(64);
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&[1.0, 2.0, 3.0])
        .set_attr(ATTR_SHA256, AttrValue::String(bogus.clone()));
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let ds = file.dataset("data").unwrap();
    match ds.verify_provenance().unwrap() {
        VerifyResult::Mismatch { stored, computed } => {
            assert_eq!(stored, bogus);
            assert_ne!(computed, bogus);
            assert_eq!(computed.len(), 64);
        }
        other => panic!("expected Mismatch, got {other:?}"),
    }
}
