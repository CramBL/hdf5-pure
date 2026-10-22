#![cfg(all(feature = "hdf5", feature = "provenance"))]

use std::str::FromStr;

use hdf5::ObjectReference;
use hdf5::ObjectReference1;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::ReferencedObject;
#[cfg(feature = "__hdf5-1.10")]
use hdf5::types::FixedUnicode;
use hdf5::types::VarLenUnicode;
#[cfg(feature = "__hdf5-1.10")]
use hdf5_pure::AttrValue;
#[cfg(feature = "__hdf5-1.10")]
use hdf5_pure::DeflateLevel;
use hdf5_pure::File;
#[cfg(feature = "__hdf5-1.10")]
use hdf5_pure::FileBuilder;
#[cfg(feature = "__hdf5-1.10")]
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::Object;
use hdf5_pure::VerifyResult;
use hdf5_pure::VlenStringReadOptions;
#[cfg(feature = "__hdf5-1.10")]
use rstest::rstest;

#[cfg(feature = "__hdf5-1.10")]
#[rstest]
#[case(Data::ChunkedStrings)]
#[case(Data::ContiguousStrings)]
#[case(Data::DeflatedStrings)]
#[case(Data::PathReferences)]
#[case(Data::ResolvedReferences)]
#[case(Data::ShuffledDeflatedStrings)]
fn c_reads_values_and_final_provenance(
    #[case] data: Data,
    #[values(false, true)] page: bool,
    #[values(false, true)] dense: bool,
    #[values(0, 4096)] userblock: u64,
    #[values(false, true)] write: bool,
) {
    let build = |address| builder(data, page, dense, userblock, address);
    let builder = if matches!(data, Data::ResolvedReferences) {
        let probe = File::from_bytes(build(0).finish().unwrap()).unwrap();
        let raw = probe.dataset("probe").unwrap().read_raw().unwrap();
        build(u64::from_le_bytes(raw.try_into().unwrap()))
    } else {
        build(0)
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("provenance.h5");
    if write {
        builder.write(&path).unwrap();
    } else {
        std::fs::write(&path, builder.finish().unwrap()).unwrap();
    }
    assert_eq!(
        test_util::fractal_heap::has_fractal_heap(&std::fs::read(&path).unwrap()),
        dense
    );
    let rust = File::open(&path).unwrap();
    let dataset = rust.dataset("data").unwrap();
    assert_eq!(dataset.verify_provenance().unwrap(), VerifyResult::Ok);
    let attrs = dataset.attrs().unwrap();
    let AttrValue::String(hash) = &attrs["_provenance_sha256"] else {
        panic!(
            "expected a string digest, got {:?}",
            attrs["_provenance_sha256"]
        );
    };
    let c = hdf5::File::open(&path).unwrap();
    let dataset = c.dataset("data").unwrap();
    assert_eq!(
        dataset.loc_info().unwrap().num_attrs,
        if dense { 12 } else { 4 }
    );
    assert_eq!(
        dataset
            .attr("_provenance_sha256")
            .unwrap()
            .read_scalar::<FixedUnicode<64>>()
            .unwrap()
            .as_str(),
        hash
    );
    assert_eq!(
        dataset
            .attr("_provenance_creator")
            .unwrap()
            .read_scalar::<FixedUnicode<10>>()
            .unwrap()
            .as_str(),
        "test-suite"
    );
    assert_eq!(
        dataset
            .attr("_provenance_timestamp")
            .unwrap()
            .read_scalar::<FixedUnicode<20>>()
            .unwrap()
            .as_str(),
        "2026-10-26T00:00:00Z"
    );
    assert_eq!(
        dataset
            .attr("_provenance_source")
            .unwrap()
            .read_scalar::<FixedUnicode<7>>()
            .unwrap()
            .as_str(),
        "fixture"
    );
    if matches!(data, Data::PathReferences | Data::ResolvedReferences) {
        let values: Vec<_> = dataset
            .read_raw::<ObjectReference1>()
            .unwrap()
            .into_iter()
            .map(|reference| {
                let object = reference.dereference(&c).unwrap();
                let ReferencedObject::Dataset(target) = object else {
                    panic!("expected a dataset target, got {object:?}");
                };
                target.read_raw::<i32>().unwrap()
            })
            .collect();
        assert_eq!(values, [vec![7], vec![7]]);
    } else {
        let values: Vec<_> = dataset
            .read_raw::<VarLenUnicode>()
            .unwrap()
            .into_iter()
            .map(|value| value.as_str().to_owned())
            .collect();
        assert_eq!(values, ["alpha", "beta", "gamma"]);
    }
    c.close().unwrap();
}

#[cfg(feature = "__hdf5-1.10")]
fn builder(data: Data, page: bool, dense: bool, userblock: u64, address: u64) -> FileBuilder {
    let mut builder = FileBuilder::new();
    builder.with_userblock(userblock);
    if page {
        builder.with_file_space_strategy(FileSpaceStrategy::Page, true, 1);
    }
    builder.create_dataset("target").with_i32_data(&[7]);
    builder
        .create_dataset("probe")
        .with_path_references(&["target"]);
    let dataset = builder.create_dataset("data");
    match data {
        Data::ChunkedStrings => dataset
            .with_vlen_strings(&["alpha", "beta", "gamma"])
            .with_chunks(&[2]),
        Data::ContiguousStrings => dataset.with_vlen_strings(&["alpha", "beta", "gamma"]),
        Data::DeflatedStrings => dataset
            .with_vlen_strings(&["alpha", "beta", "gamma"])
            .with_chunks(&[2])
            .with_deflate(DeflateLevel::Six),
        Data::PathReferences => dataset.with_path_references(&["target", "target"]),
        Data::ResolvedReferences => dataset
            .with_reference_data(&[address, address])
            .with_chunks(&[1])
            .with_shuffle()
            .with_deflate(DeflateLevel::Six),
        Data::ShuffledDeflatedStrings => dataset
            .with_vlen_strings(&["alpha", "beta", "gamma"])
            .with_chunks(&[2])
            .with_shuffle()
            .with_deflate(DeflateLevel::Six),
    }
    .with_provenance("test-suite", "2026-10-26T00:00:00Z", Some("fixture"));
    if dense {
        for index in 0..8 {
            dataset.set_attr(&format!("extra{index}"), AttrValue::I32(index));
        }
    }
    builder
}

#[cfg(feature = "__hdf5-1.10")]
#[derive(Clone, Copy, Debug)]
enum Data {
    ChunkedStrings,
    ContiguousStrings,
    DeflatedStrings,
    PathReferences,
    ResolvedReferences,
    ShuffledDeflatedStrings,
}

#[test]
fn c_created_strings_and_references_return_values_and_no_hash() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reference.h5");
    let c = hdf5::File::create(&path).unwrap();
    c.new_dataset::<i32>()
        .shape([1])
        .create("target")
        .unwrap()
        .write(&[7])
        .unwrap();
    c.new_dataset::<ObjectReference1>()
        .shape([1])
        .create("references")
        .unwrap()
        .write(&[ObjectReference1::create(&c, "target").unwrap()])
        .unwrap();
    c.new_dataset::<VarLenUnicode>()
        .shape([2])
        .create("strings")
        .unwrap()
        .write(&[
            VarLenUnicode::from_str("alpha").unwrap(),
            VarLenUnicode::from_str("beta").unwrap(),
        ])
        .unwrap();
    c.close().unwrap();
    let rust = File::open(&path).unwrap();
    let references = rust.dataset("references").unwrap();
    let objects = references.dereference().unwrap();
    let Object::Dataset(target) = &objects[0] else {
        panic!("expected a dataset target, got {:?}", objects[0]);
    };
    assert_eq!(target.read_i32().unwrap(), [7]);
    assert_eq!(
        references.verify_provenance().unwrap(),
        VerifyResult::NoHash
    );
    let strings = rust.dataset("strings").unwrap();
    assert_eq!(
        strings
            .read_vlen_strings(VlenStringReadOptions::default())
            .unwrap(),
        ["alpha", "beta"]
    );
    assert_eq!(strings.verify_provenance().unwrap(), VerifyResult::NoHash);
}
