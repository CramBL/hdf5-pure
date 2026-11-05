use std::path::Path;

use hdf5_pure::Datatype;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FixedPointType;
use hdf5_pure::Layout;
use hdf5_pure::LengthWidth;
use hdf5_pure::MessageType;
use hdf5_pure::OffsetWidth;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::AttributeInfoMessage;
use hdf5_pure_format::__private::LinkInfoMessage;
use hdf5_pure_format::__private::ObjectHeader;
use rstest::rstest;
use test_util::range;

use crate::__private::width_workload::{Case, Index, Phase, Unsupported};
use crate::file_space::persisted::PersistedStrategy;
use crate::storage;

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
fn finite_workloads_preserve_contents_reuse_and_cache_observations(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(PersistedStrategy::FsmAggr, PersistedStrategy::Page)] strategy: PersistedStrategy,
    #[values(false, true)] filtered: bool,
) {
    lifecycle(Case::new(offsets, lengths, Index::Finite, strategy, filtered).unwrap());
}

#[rstest]
#[case::two(OffsetWidth::Two)]
#[case::four(OffsetWidth::Four)]
#[case::eight(OffsetWidth::Eight)]
fn extensible_workloads_preserve_contents_reuse_and_cache_observations(
    #[case] offsets: OffsetWidth,
    #[values(PersistedStrategy::FsmAggr, PersistedStrategy::Page)] strategy: PersistedStrategy,
    #[values(false, true)] filtered: bool,
) {
    lifecycle(
        Case::new(
            offsets,
            LengthWidth::Eight,
            Index::Extensible,
            strategy,
            filtered,
        )
        .unwrap(),
    );
}

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two)]
#[case::two_four(OffsetWidth::Two, LengthWidth::Four)]
#[case::four_two(OffsetWidth::Four, LengthWidth::Two)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::eight_four(OffsetWidth::Eight, LengthWidth::Four)]
fn unlimited_workloads_reject_narrow_lengths(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
) {
    assert_eq!(
        Case::new(
            offsets,
            lengths,
            Index::Extensible,
            PersistedStrategy::FsmAggr,
            false
        )
        .unwrap_err(),
        Unsupported { lengths }
    );
}

#[rstest]
#[case::disabled(0)]
#[case::oversize(1)]
#[case::retained(65_536)]
fn cache_observations_distinguish_disabled_oversize_and_retained_reads(#[case] cache_bytes: usize) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cache.h5");
    let case = case();
    case.create(&path).unwrap();
    let file = case.open_read(&path, cache_bytes).unwrap();
    case.verify(&file, Phase::Create);
    let first = case.observe(&file).unwrap();
    case.verify(&file, Phase::Create);
    let second = case.observe(&file).unwrap();
    match (cache_bytes, first.metadata_cache, second.metadata_cache) {
        (0, None, None) => {}
        (1, Some(first), Some(second)) => {
            assert_eq!((first.entries(), first.bytes()), (0, 0));
            assert_eq!((second.entries(), second.bytes()), (0, 0));
            assert!(first.oversize_reads() > 0);
            assert!(second.oversize_reads() > first.oversize_reads());
        }
        (65_536, Some(first), Some(second)) => {
            assert!(first.entries() > 0 && first.bytes() > 0);
            assert!(second.hits() > first.hits());
            assert!(second.bytes() <= cache_bytes);
        }
        observed => panic!("unexpected cache observations: {observed:?}"),
    }
    file.close().unwrap();
    directory.close().unwrap();
}

#[test]
fn space_observations_reject_files_without_persistent_managers() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("nonpersistent.h5");
    FileBuilder::new().write(&path).unwrap();
    let file = File::open_streaming(&path).unwrap();
    let error = case().observe(&file).unwrap_err();
    assert!(
        matches!(
            error,
            Error::EditUnsupported("width workload requires persisted free-space managers")
        ),
        "{error:?}"
    );
    file.close().unwrap();
    directory.close().unwrap();
}

#[test]
#[should_panic(expected = "width workload contiguous values differ")]
fn content_validation_detects_a_changed_dataset() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("different.h5");
    let case = case();
    case.create(&path).unwrap();
    let file = case.open(&path, 65_536).unwrap();
    file.dataset("data/contiguous")
        .unwrap()
        .write_staged(|builder| {
            builder.with_i32_data(&[29; 32]);
        })
        .unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    let file = case.open(&path, 65_536).unwrap();
    case.verify(&file, Phase::Create);
}

#[test]
#[should_panic(expected = "width workload datatype differs")]
fn schema_validation_detects_equal_values_with_a_different_datatype() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("datatype.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("different").with_u32_data(&[13; 32]);
    builder.write(&path).unwrap();
    let file = File::open(&path).unwrap();
    let dataset = file.dataset("different").unwrap();
    assert_eq!(dataset.read_i32().unwrap(), [13; 32]);
    super::assert_schema(&dataset, Datatype::FixedPoint(FixedPointType::i32()), 32);
}

#[test]
#[should_panic(expected = "width workload shape differs")]
fn schema_validation_detects_equal_values_with_a_different_shape() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("shape.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("different")
        .with_i32_data(&[13; 32])
        .with_shape(&[4, 8]);
    builder.write(&path).unwrap();
    let file = File::open(&path).unwrap();
    let dataset = file.dataset("different").unwrap();
    assert_eq!(dataset.read_i32().unwrap(), [13; 32]);
    super::assert_schema(&dataset, Datatype::FixedPoint(FixedPointType::i32()), 32);
}

fn lifecycle(case: Case) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("workload.h5");
    case.create(&path).unwrap();
    assert_dense_storage(&path);
    let file = case.open_read(&path, 65_536).unwrap();
    case.verify(&file, Phase::Create);
    let initial = case.observe(&file).unwrap();
    assert_eq!(initial.file_bytes, std::fs::metadata(&path).unwrap().len());
    assert_eq!(
        initial.file_bytes,
        initial.reusable_bytes + initial.non_reusable_bytes
    );
    assert!(initial.metadata_cache.unwrap().bytes() > 0);
    assert!(initial.dataset_bytes > 0 && initial.dataset_bytes < initial.non_reusable_bytes);
    file.close().unwrap();
    let file = case.open(&path, 65_536).unwrap();
    case.append(&file).unwrap();
    file.commit().unwrap();
    case.verify(&file, Phase::Append);
    file.close().unwrap();
    let file = case.open(&path, 65_536).unwrap();
    case.verify(&file, Phase::Append);
    let retired = (0..4)
        .map(|index| {
            match file
                .dataset(&format!("holes/scratch_{index}"))
                .unwrap()
                .layout()
                .unwrap()
            {
                Layout::Contiguous {
                    address: Some(address),
                    size,
                } => address..address + size,
                layout => panic!("unexpected scratch layout: {layout:?}"),
            }
        })
        .collect::<Vec<_>>();
    case.stage_reclamation(&file).unwrap();
    file.commit().unwrap();
    case.verify(&file, Phase::Reclaim);
    file.close().unwrap();
    let file = case.open_read(&path, 65_536).unwrap();
    case.verify(&file, Phase::Reclaim);
    let reclaimed = case.observe(&file).unwrap();
    assert!(reclaimed.reusable_sections > 0 && reclaimed.reusable_bytes > 0);
    assert!(reclaimed.manager_bytes > 0);
    let sections = file
        .persisted_free_sections()
        .unwrap()
        .unwrap()
        .free_sections()
        .clone();
    let free = sections
        .sections()
        .iter()
        .map(|section| (section.address().get(), section.length().get()))
        .collect::<Vec<_>>();
    for extent in retired {
        assert_eq!(
            range::covered_len(&free, &extent),
            extent.end - extent.start
        );
    }
    file.close().unwrap();
    let file = case.open(&path, 65_536).unwrap();
    case.stage_reuse(&file).unwrap();
    file.commit().unwrap();
    case.verify(&file, Phase::Reuse);
    let Layout::Contiguous {
        address: Some(address),
        size,
    } = file.dataset("holes/replacement").unwrap().layout().unwrap()
    else {
        panic!("replacement is not allocated contiguous storage")
    };
    assert_eq!(size, 128);
    assert!(sections.sections().iter().any(|section| {
        section.address().get() <= address && address + size <= section.end_address().get()
    }));
    file.close().unwrap();
    let file = case.open_read(&path, 65_536).unwrap();
    case.verify(&file, Phase::Reuse);
    let final_space = case.observe(&file).unwrap();
    assert_eq!(
        final_space.reusable_bytes,
        file.persisted_free_sections()
            .unwrap()
            .unwrap()
            .free_sections()
            .free_bytes()
    );
    file.close().unwrap();
    directory.close().unwrap();
}

fn assert_dense_storage(path: &Path) {
    let bytes = std::fs::read(path).unwrap();
    let file = File::open_streaming(path).unwrap();
    let address = storage::root_object_address(&file, &bytes, "metadata");
    let header = ObjectHeader::parse(
        &bytes,
        AccessMode::ReadOnly,
        usize::try_from(address.get()).unwrap(),
        file.superblock().offset_size.get(),
        file.superblock().length_size.get(),
    )
    .unwrap();
    let links = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::LINK_INFO)
        .unwrap();
    let links =
        LinkInfoMessage::parse_with_offset_width(&links.data, file.superblock().offset_size)
            .unwrap();
    assert!(links.fractal_heap_address.is_some());
    assert!(links.btree_name_index_address.is_some());
    let attributes = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::ATTRIBUTE_INFO)
        .unwrap();
    let attributes = AttributeInfoMessage::parse_with_offset_width(
        &attributes.data,
        file.superblock().offset_size,
    )
    .unwrap();
    assert!(attributes.fractal_heap_address.is_some());
    assert!(attributes.btree_name_index_address.is_some());
    assert!(
        header
            .messages
            .iter()
            .all(|message| message.msg_type != MessageType::LINK
                && message.msg_type != MessageType::ATTRIBUTE)
    );
    file.close().unwrap();
}

fn case() -> Case {
    Case::new(
        OffsetWidth::Eight,
        LengthWidth::Eight,
        Index::Finite,
        PersistedStrategy::FsmAggr,
        false,
    )
    .unwrap()
}
