#![cfg(feature = "__hdf5-1.10")]

use std::path::Path;

use hdf5::Datatype;
use hdf5::datatype::ByteOrder as NativeByteOrder;
use hdf5::filters::Filter;
use hdf5::types::VarLenUnicode;
use hdf5_pure::LengthWidth;
use hdf5_pure::OffsetWidth;
use rstest::rstest;
use test_util_hdf5::__private::width_workload::{Case, Index, Phase};
use test_util_hdf5::dataset;
use test_util_hdf5::file;
use test_util_hdf5::file_space::persisted::PersistedStrategy;
use test_util_hdf5::file_space_oracle;
use test_util_hdf5::numeric::ByteOrder;

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two)]
#[case::two_four(OffsetWidth::Two, LengthWidth::Four)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::four_two(OffsetWidth::Four, LengthWidth::Two)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::eight_four(OffsetWidth::Eight, LengthWidth::Four)]
#[case::eight_eight(OffsetWidth::Eight, LengthWidth::Eight)]
fn finite_benchmark_workloads_preserve_native_contents_and_manager_observations(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(PersistedStrategy::FsmAggr, PersistedStrategy::Page)] strategy: PersistedStrategy,
    #[values(false, true)] filtered: bool,
) {
    run(Case::new(offsets, lengths, Index::Finite, strategy, filtered).unwrap());
}

#[rstest]
#[case::two(OffsetWidth::Two)]
#[case::four(OffsetWidth::Four)]
#[case::eight(OffsetWidth::Eight)]
fn extensible_benchmark_workloads_preserve_native_contents_and_manager_observations(
    #[case] offsets: OffsetWidth,
    #[values(PersistedStrategy::FsmAggr, PersistedStrategy::Page)] strategy: PersistedStrategy,
    #[values(false, true)] filtered: bool,
) {
    run(Case::new(
        offsets,
        LengthWidth::Eight,
        Index::Extensible,
        strategy,
        filtered,
    )
    .unwrap());
}

fn run(case: Case) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("benchmark.h5");
    case.create(&path).unwrap();
    verify_native(&path, case, Phase::Create);
    let pure = case.open(&path, 65_536).unwrap();
    case.append(&pure).unwrap();
    pure.commit().unwrap();
    pure.close().unwrap();
    verify_native(&path, case, Phase::Append);
    let pure = case.open(&path, 65_536).unwrap();
    case.stage_reclamation(&pure).unwrap();
    pure.commit().unwrap();
    pure.close().unwrap();
    verify_native(&path, case, Phase::Reclaim);
    let pure = case.open(&path, 65_536).unwrap();
    case.stage_reuse(&pure).unwrap();
    pure.commit().unwrap();
    pure.close().unwrap();
    verify_native(&path, case, Phase::Reuse);
    directory.close().unwrap();
}

fn verify_native(path: &Path, case: Case, phase: Phase) {
    let pure = case.open_read(path, 65_536).unwrap();
    case.verify(&pure, phase);
    let observed = case.observe(&pure).unwrap();
    pure.close().unwrap();
    let native = file::libhdf5_open(path);
    let properties = native.create_plist().unwrap();
    let sizes = properties.get_sizes().unwrap();
    assert_eq!(
        usize::from(sizes.sizeof_addr),
        usize::from(case.offsets().get())
    );
    assert_eq!(
        usize::from(sizes.sizeof_size),
        usize::from(case.lengths().get())
    );
    assert_eq!(native.size(), observed.file_bytes);
    let dataset_bytes = ["metadata", "data", "holes"]
        .into_iter()
        .map(|group_name| {
            let group = native.group(group_name).unwrap();
            group
                .member_names()
                .unwrap()
                .into_iter()
                .map(|link_name| group.dataset(&link_name).unwrap().storage_size())
                .sum::<u64>()
        })
        .sum::<u64>();
    assert_eq!(dataset_bytes, observed.dataset_bytes);
    let mut members = native.member_names().unwrap();
    members.sort();
    assert_eq!(members, ["data", "holes", "metadata"]);
    assert!(native.attr_names().unwrap().is_empty());
    let metadata = native.group("metadata").unwrap();
    let mut attributes = metadata.attr_names().unwrap();
    attributes.sort();
    assert_eq!(
        attributes,
        (0..9)
            .map(|value| format!("attribute_{value}"))
            .collect::<Vec<_>>()
    );
    let mut members = metadata.member_names().unwrap();
    members.sort();
    assert_eq!(
        members,
        (0..9)
            .map(|value| format!("member_{value}"))
            .collect::<Vec<_>>()
    );
    for value in 0_u32..9 {
        assert_eq!(
            metadata
                .dataset(&format!("member_{value}"))
                .unwrap()
                .dtype()
                .unwrap(),
            ByteOrder::LittleEndian.u32_datatype()
        );
        assert_eq!(
            metadata
                .dataset(&format!("member_{value}"))
                .unwrap()
                .shape(),
            [3]
        );
        let expected = if value == 0 && matches!(phase, Phase::Reclaim | Phase::Reuse) {
            37
        } else {
            value
        };
        assert_eq!(
            metadata
                .attr(&format!("attribute_{value}"))
                .unwrap()
                .read_scalar::<u32>()
                .unwrap(),
            expected
        );
        assert_eq!(
            metadata
                .dataset(&format!("member_{value}"))
                .unwrap()
                .read_raw::<u32>()
                .unwrap(),
            [value; 3]
        );
        assert!(
            metadata
                .dataset(&format!("member_{value}"))
                .unwrap()
                .attr_names()
                .unwrap()
                .is_empty()
        );
    }
    let data = native.group("data").unwrap();
    let mut names = data.member_names().unwrap();
    names.sort();
    assert_eq!(names, ["contiguous", "series", "single", "strings"]);
    assert!(data.attr_names().unwrap().is_empty());
    for link_name in names {
        let dataset = data.dataset(&link_name).unwrap();
        let length = match link_name.as_str() {
            "series" => {
                if phase == Phase::Create {
                    8
                } else {
                    12
                }
            }
            "single" => {
                if phase == Phase::Create {
                    1
                } else {
                    2
                }
            }
            "strings" => 3,
            _ => 32,
        };
        assert_eq!(dataset.shape(), [length]);
        if link_name == "strings" {
            assert_eq!(
                dataset.dtype().unwrap(),
                Datatype::from_type::<VarLenUnicode>()
                    .unwrap()
                    .with_byte_order(NativeByteOrder::LittleEndian)
                    .unwrap()
            );
        } else {
            assert_eq!(
                dataset.dtype().unwrap(),
                ByteOrder::LittleEndian.i32_datatype()
            );
        }
        assert!(dataset.attr_names().unwrap().is_empty());
    }
    let series = native.dataset("data/series").unwrap();
    assert_eq!(
        series.read_raw::<i32>().unwrap(),
        (0..if phase == Phase::Create { 8 } else { 12 }).collect::<Vec<_>>()
    );
    assert_eq!(series.chunk(), Some(vec![4]));
    assert_eq!(
        series.space().unwrap().maxdims(),
        [match case.index() {
            Index::Extensible => None,
            Index::Finite => Some(24),
        }]
    );
    assert_eq!(
        series.filters(),
        if case.filtered() {
            vec![Filter::Deflate(6)]
        } else {
            Vec::new()
        }
    );
    assert_eq!(
        native
            .dataset("data/single")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        if phase == Phase::Create {
            vec![1]
        } else {
            vec![1, 2]
        }
    );
    assert_eq!(
        native
            .dataset("data/single")
            .unwrap()
            .space()
            .unwrap()
            .maxdims(),
        [Some(4)]
    );
    assert_eq!(
        native
            .dataset("data/contiguous")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [13; 32]
    );
    let holes = native.group("holes").unwrap();
    assert!(holes.attr_names().unwrap().is_empty());
    let mut members = holes.member_names().unwrap();
    members.sort();
    let mut expected = (0..4)
        .map(|value| format!("spacer_{value}"))
        .collect::<Vec<_>>();
    if matches!(phase, Phase::Create | Phase::Append) {
        expected.extend((0..4).map(|value| format!("scratch_{value}")));
    }
    if phase == Phase::Reuse {
        expected.push("replacement".into());
        assert_eq!(
            holes
                .dataset("replacement")
                .unwrap()
                .read_raw::<u8>()
                .unwrap(),
            [43; 128]
        );
    }
    expected.sort();
    assert_eq!(members, expected);
    for link_name in members {
        let dataset = holes.dataset(&link_name).unwrap();
        assert_eq!(
            dataset.dtype().unwrap(),
            ByteOrder::LittleEndian.u8_datatype()
        );
        assert_eq!(
            dataset.shape(),
            [if link_name.starts_with("spacer_") {
                64
            } else {
                128
            }]
        );
        assert!(dataset.attr_names().unwrap().is_empty());
    }
    for value in 0..4 {
        assert_eq!(
            holes
                .dataset(&format!("spacer_{value}"))
                .unwrap()
                .read_raw::<u8>()
                .unwrap(),
            [31; 64]
        );
        if matches!(phase, Phase::Create | Phase::Append) {
            assert_eq!(
                holes
                    .dataset(&format!("scratch_{value}"))
                    .unwrap()
                    .read_raw::<u8>()
                    .unwrap(),
                [17; 128]
            );
        }
    }
    drop(holes);
    drop(data);
    drop(series);
    drop(metadata);
    drop(properties);
    native.close().unwrap();
    assert_eq!(
        dataset::read_libhdf5_strings(path, "data/strings"),
        ["alpha", "", "ζ"]
    );
    file_space_oracle::assert_persisted_sections(path);
    file_space_oracle::assert_manager_metadata_bytes(path);
}
