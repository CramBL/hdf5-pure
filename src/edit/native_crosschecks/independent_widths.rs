use std::collections::BTreeMap;
use std::io::ErrorKind;
use std::path::Path;

use rstest::rstest;
use test_util::disk_log;
use test_util::disk_log::{IoEvent, IoOutcome};
use test_util_hdf5::attr;
use test_util_hdf5::dataset;
use test_util_hdf5::dataset::FiniteChunkIndex;
use test_util_hdf5::file;
use test_util_hdf5::file_space_oracle;
use test_util_hdf5::numeric::ByteOrder;
use test_util_hdf5::object_copy::{self, ReferenceSite};

use crate::AttrValue;
use crate::ChunkIndex;
use crate::DeflateLevel;
use crate::EditBacking;
use crate::Error;
use crate::File;
use crate::FileAccessProperties;
use crate::FileBuilder;
use crate::FileSpacePageSize;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::FilterId;
use crate::FormatError;
use crate::Layout;
use crate::LibVer;
use crate::MaxExtent;
use crate::MemoryStrategy;
use crate::PersistedFreeSpacePolicy;
use crate::SuperblockVersion;
use crate::UserblockSize;
use crate::address::AddressContext;
use crate::attribute_info::AttributeInfoMessage;
use crate::edit::WriteEngine;
use crate::link_info::LinkInfoMessage;
use crate::message_type::MessageType;
use crate::source::BytesSource;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

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
fn finite_mutations_preserve_native_values_widths_and_space(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(
        FileSpaceStrategy::None,
        FileSpaceStrategy::Aggr,
        FileSpaceStrategy::FsmAggr,
        FileSpaceStrategy::Page
    )]
    strategy: FileSpaceStrategy,
    #[values(Producer::Native, Producer::Rust)] producer: Producer,
    #[values(FiniteChunkIndex::Single, FiniteChunkIndex::Fixed)] index: FiniteChunkIndex,
    #[values(false, true)] filtered: bool,
) {
    numeric_lifecycle(
        FormatWidths { offsets, lengths },
        backing,
        strategy,
        producer,
        Index::Finite(index),
        filtered,
    );
}

#[rstest]
#[case::two(OffsetWidth::Two)]
#[case::four(OffsetWidth::Four)]
#[case::eight(OffsetWidth::Eight)]
fn extensible_mutations_preserve_native_values_widths_and_space(
    #[case] offsets: OffsetWidth,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(
        FileSpaceStrategy::None,
        FileSpaceStrategy::Aggr,
        FileSpaceStrategy::FsmAggr,
        FileSpaceStrategy::Page
    )]
    strategy: FileSpaceStrategy,
    #[values(Producer::Native, Producer::Rust)] producer: Producer,
    #[values(false, true)] filtered: bool,
) {
    numeric_lifecycle(
        FormatWidths {
            offsets,
            lengths: LengthWidth::Eight,
        },
        backing,
        strategy,
        producer,
        Index::Extensible,
        filtered,
    );
}

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
fn dense_metadata_copies_and_heaps_preserve_native_values_and_widths(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(FileSpaceStrategy::FsmAggr, FileSpaceStrategy::Page)] strategy: FileSpaceStrategy,
    #[values(Producer::Native, Producer::Rust)] producer: Producer,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("dense.h5");
    let source_path = directory.path().join("source.h5");
    let widths = FormatWidths { offsets, lengths };
    create(
        &path,
        widths,
        settings(strategy),
        producer,
        Index::Finite(FiniteChunkIndex::Fixed),
        false,
    );
    let mut source = FileBuilder::new();
    source.with_offset_width(offsets).with_length_width(lengths);
    source.create_dataset("source").with_i32_data(&[19, 23, 29]);
    source.write(&source_path).unwrap();
    let source = File::open(&source_path).unwrap();
    let access = FileAccessProperties::new()
        .with_memory_strategy(backing)
        .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict);
    let file = File::open_rw_with_options(&path, access).unwrap();
    let group = file.group("g").unwrap();
    for value in 0_u32..9 {
        group
            .create_dataset(&format!("child_{value}"), |dataset| {
                dataset.with_u32_data(&[value; 3]);
            })
            .unwrap();
        group
            .set_attr(&format!("number_{value}"), AttrValue::U32(value))
            .unwrap();
    }
    group
        .set_attr(
            "text",
            AttrValue::VarLenStringArray(vec!["alpha".into(), String::new(), "ζ".into()]),
        )
        .unwrap();
    file.root()
        .create_dataset("strings", |dataset| {
            dataset.with_vlen_strings(&["alpha", "", "ζ"]);
        })
        .unwrap();
    file.copy_from(&source, "source", "copied_from").unwrap();
    file.commit().unwrap();
    assert_dense(&file, true);
    let expected = (0_u32..9)
        .map(|value| (format!("number_{value}"), AttrValue::U32(value)))
        .chain([(
            "text".into(),
            AttrValue::VarLenStringArray(vec!["alpha".into(), String::new(), "ζ".into()]),
        )])
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        group
            .attrs()
            .unwrap()
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
        expected
    );
    drop(group);
    file.close().unwrap();
    source.close().unwrap();
    {
        let native = file::libhdf5_open(&path);
        let group = native.group("g").unwrap();
        let mut members = group.member_names().unwrap();
        members.sort();
        assert_eq!(
            members,
            (0..9)
                .map(|value| format!("child_{value}"))
                .collect::<Vec<_>>()
        );
        assert_eq!(attr::unicode_array(&group, "text"), ["alpha", "", "ζ"]);
        for value in 0_u32..9 {
            assert_eq!(
                group
                    .attr(&format!("number_{value}"))
                    .unwrap()
                    .read_scalar::<u32>()
                    .unwrap(),
                value
            );
            assert_eq!(
                group
                    .dataset(&format!("child_{value}"))
                    .unwrap()
                    .read_raw::<u32>()
                    .unwrap(),
                [value; 3]
            );
        }
        assert_eq!(
            native
                .dataset("copied_from")
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            [19, 23, 29]
        );
    }
    assert_eq!(
        dataset::read_libhdf5_strings(&path, "strings"),
        ["alpha", "", "ζ"]
    );
    file_space_oracle::assert_persisted_sections(&path);
    file_space_oracle::assert_manager_metadata_bytes(&path);
    let file = File::open_rw_with_options(&path, access).unwrap();
    file.copy("g/child_0", "copied").unwrap();
    let group = file.group("g").unwrap();
    for value in 1_u32..9 {
        group.delete(&format!("child_{value}")).unwrap();
        group.remove_attr(&format!("number_{value}")).unwrap();
    }
    group.remove_attr("text").unwrap();
    file.dataset("strings")
        .unwrap()
        .write_staged(|dataset| {
            dataset.with_vlen_strings(&["delta", "ε", ""]);
        })
        .unwrap();
    file.commit().unwrap();
    assert_dense(&file, false);
    assert_eq!(group.datasets().unwrap(), ["child_0"]);
    assert_eq!(
        group
            .attrs()
            .unwrap()
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
        BTreeMap::from([("number_0".into(), AttrValue::U32(0))])
    );
    assert_eq!(file.dataset("copied").unwrap().read_u32().unwrap(), [0; 3]);
    assert_eq!(
        file.dataset("strings").unwrap().read_string().unwrap(),
        ["delta", "ε", ""]
    );
    assert_eq!(file.superblock().offset_size, offsets);
    assert_eq!(file.superblock().length_size, lengths);
    drop(group);
    file.close().unwrap();
    {
        let native = file::libhdf5_open(&path);
        let group = native.group("g").unwrap();
        assert_eq!(group.member_names().unwrap(), ["child_0"]);
        assert_eq!(group.attr_names().unwrap(), ["number_0"]);
        assert_eq!(
            native.dataset("copied").unwrap().read_raw::<u32>().unwrap(),
            [0; 3]
        );
        assert_eq!(
            native
                .dataset("copied_from")
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            [19, 23, 29]
        );
    }
    assert_eq!(
        dataset::read_libhdf5_strings(&path, "strings"),
        ["delta", "ε", ""]
    );
    file_space_oracle::assert_persisted_sections(&path);
    file_space_oracle::assert_manager_metadata_bytes(&path);
    let file = File::open_rw_with_options(&path, access).unwrap();
    let reclaimed = file.reusable_free_sections().unwrap();
    assert!(reclaimed.free_bytes() > 0);
    file.root()
        .create_dataset("replacement", |dataset| {
            dataset.with_u32_data(&[31; 3]);
        })
        .unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    {
        let native = file::libhdf5_open(&path);
        let dataset = native.dataset("replacement").unwrap();
        assert_eq!(dataset.read_raw::<u32>().unwrap(), [31; 3]);
        let address = dataset.offset().unwrap();
        assert!(
            reclaimed.sections().iter().any(|section| {
                section.address().get() <= address && address + 12 <= section.end_address().get()
            }),
            "replacement at {address} did not reuse {reclaimed:?}"
        );
    }
    file_space_oracle::assert_persisted_sections(&path);
    file_space_oracle::assert_manager_metadata_bytes(&path);
    let stable = std::fs::read(&path).unwrap();
    for _ in 0..3 {
        let file = File::open_rw_with_options(&path, access).unwrap();
        file.commit().unwrap();
        file.close().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), stable);
    }
    directory.close().unwrap();
}

#[rstest]
#[case::two(LengthWidth::Two, u64::from(u16::MAX))]
#[case::four(LengthWidth::Four, u64::from(u32::MAX))]
fn finite_all_ones_maxima_remain_finite_in_both_libraries(
    #[case] lengths: LengthWidth,
    #[case] maximum: u64,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
    #[values(Producer::Native, Producer::Rust)] producer: Producer,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("finite-maximum.h5");
    match producer {
        Producer::Native => {
            let native = file::libhdf5_create_with_widths_and_space(
                &path,
                offsets,
                lengths,
                0,
                settings(FileSpaceStrategy::None),
            );
            native
                .new_dataset::<u8>()
                .shape((0..=usize::try_from(maximum).unwrap(),))
                .chunk((4,))
                .create("schema")
                .unwrap();
            native.close().unwrap();
        }
        Producer::Rust => {
            let mut builder = FileBuilder::new();
            builder
                .with_offset_width(offsets)
                .with_length_width(lengths);
            builder
                .create_dataset("schema")
                .with_unallocated_storage(crate::make_u8_type(), &[0])
                .with_maxshape(&[MaxExtent::Fixed(maximum)])
                .with_chunks(&[4]);
            builder.write(&path).unwrap();
        }
    }
    {
        let native = file::libhdf5_open(&path);
        let dataset = native.dataset("schema").unwrap();
        assert_eq!(dataset.shape(), [0]);
        assert_eq!(
            dataset.space().unwrap().maxdims(),
            [Some(usize::try_from(maximum).unwrap())]
        );
        assert_eq!(dataset.storage_size(), 0);
    }
    let file = File::open_rw(&path).unwrap();
    let mut dataset = file.dataset("schema").unwrap();
    assert_eq!(
        dataset.maxshape().unwrap(),
        Some(vec![MaxExtent::Fixed(maximum)])
    );
    dataset.set_attr("tag", AttrValue::U32(17)).unwrap();
    file.commit().unwrap();
    assert_eq!(
        dataset.maxshape().unwrap(),
        Some(vec![MaxExtent::Fixed(maximum)])
    );
    drop(dataset);
    file.close().unwrap();
    let native = file::libhdf5_open(&path);
    let dataset = native.dataset("schema").unwrap();
    assert_eq!(
        dataset.space().unwrap().maxdims(),
        [Some(usize::try_from(maximum).unwrap())]
    );
    assert_eq!(
        dataset.attr("tag").unwrap().read_scalar::<u32>().unwrap(),
        17
    );
    drop(dataset);
    native.close().unwrap();
    directory.close().unwrap();
}

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
fn mirrored_variants_preserve_narrow_native_values_and_domains(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(MirroredCase::NonpersistentPage, MirroredCase::Userblock)] layout: MirroredCase,
    #[values(MemoryStrategy::Auto, MemoryStrategy::Mirrored)] backing: MemoryStrategy,
    #[values(Producer::Native, Producer::Rust)] producer: Producer,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("mirrored.h5");
    let (settings, base) = match layout {
        MirroredCase::NonpersistentPage => {
            (settings(FileSpaceStrategy::Page).with_persistence(false), 0)
        }
        MirroredCase::Userblock => (settings(FileSpaceStrategy::FsmAggr), 512),
    };
    create_with_base(
        &path,
        FormatWidths { offsets, lengths },
        settings,
        producer,
        Index::Finite(FiniteChunkIndex::Fixed),
        false,
        UserblockSize::try_from(base).unwrap(),
    );
    let mut end = 1;
    for cycle in 0_u32..2 {
        let file = File::open_rw_with_options(
            &path,
            FileAccessProperties::new().with_memory_strategy(backing),
        )
        .unwrap();
        assert_eq!(file.edit_backing(), Some(EditBacking::Mirrored));
        let mut target = file.dataset("series").unwrap();
        target
            .append_staged(|append| {
                append.append(&(end..end + 3).collect::<Vec<i32>>());
            })
            .unwrap();
        end += 3;
        file.group("g")
            .unwrap()
            .set_attr("counter", AttrValue::U32(cycle))
            .unwrap();
        file.commit().unwrap();
        assert_eq!(target.read_i32().unwrap(), (0..end).collect::<Vec<_>>());
        assert_eq!(file.superblock().base_address.get(), base);
        assert_eq!(file.superblock().offset_size, offsets);
        assert_eq!(file.superblock().length_size, lengths);
        drop(target);
        file.close().unwrap();
        let native = file::libhdf5_open(&path);
        assert_eq!(
            native.dataset("series").unwrap().read_raw::<i32>().unwrap(),
            (0..end).collect::<Vec<_>>()
        );
        assert_eq!(native.create_plist().unwrap().userblock(), base);
        native.close().unwrap();
        if settings.persist() {
            file_space_oracle::assert_persisted_sections(&path);
            file_space_oracle::assert_manager_metadata_bytes(&path);
        }
    }
    directory.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum MirroredCase {
    NonpersistentPage,
    Userblock,
}

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
fn legacy_reference_copy_policies_preserve_the_native_source(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(ReferenceSite::Dataset, ReferenceSite::Attribute)] site: ReferenceSite,
) {
    legacy_reference_copy(FormatWidths { offsets, lengths }, backing, site);
}

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
fn failed_narrow_commits_keep_the_published_root_and_live_values(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(FileSpaceStrategy::FsmAggr, FileSpaceStrategy::Page)] strategy: FileSpaceStrategy,
    #[values(Producer::Native, Producer::Rust)] producer: Producer,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failed-commit.h5");
    create(
        &path,
        FormatWidths { offsets, lengths },
        settings(strategy),
        producer,
        Index::Finite(FiniteChunkIndex::Fixed),
        false,
    );
    let access = FileAccessProperties::new()
        .with_memory_strategy(backing)
        .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict);
    let edited = File::open_rw_with_options(&path, access).unwrap();
    edited
        .group("g")
        .unwrap()
        .set_attr("counter", AttrValue::U32(0))
        .unwrap();
    edited.commit().unwrap();
    edited.close().unwrap();
    let edited = File::open_rw_with_options(&path, access).unwrap();
    let root = edited.superblock().root_group_address;
    edited
        .root()
        .create_dataset("pending", |dataset| {
            dataset.with_u32_data(&[13, 17]);
        })
        .unwrap();
    edited
        .group("g")
        .unwrap()
        .set_attr("counter", AttrValue::U32(1))
        .unwrap();
    disk_log::start();
    disk_log::fail_write_after(0, 3);
    let error = edited.commit().unwrap_err();
    let recording = disk_log::finish();
    let Error::Io(error) = error else {
        panic!("expected injected I/O failure, got {error:?}")
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    let failed = recording
        .events
        .iter()
        .filter_map(|event| match event {
            IoEvent::Write {
                requested,
                written,
                outcome: IoOutcome::Failed(ErrorKind::Other),
                ..
            } => Some((*requested, *written)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].1, 3);
    assert!(failed[0].0 > 3);
    assert_eq!(edited.superblock().root_group_address, root);
    assert_eq!(edited.dataset("series").unwrap().read_i32().unwrap(), [0]);
    assert_eq!(
        edited.dataset("shared").unwrap().read_i32().unwrap(),
        [7, 11]
    );
    drop(edited);
    let reopened = File::open_rw_with_options(&path, access).unwrap();
    assert_eq!(reopened.superblock().root_group_address, root);
    let mut datasets = reopened.root().datasets().unwrap();
    datasets.sort();
    assert_eq!(datasets, ["series", "shared", "shared_alias"]);
    assert_eq!(
        reopened
            .group("g")
            .unwrap()
            .attrs()
            .unwrap()
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
        BTreeMap::from([("counter".into(), AttrValue::U32(0))])
    );
    reopened.close().unwrap();
    let native = file::libhdf5_open(&path);
    assert_eq!(
        native.dataset("series").unwrap().read_raw::<i32>().unwrap(),
        [0]
    );
    assert_eq!(
        native.dataset("shared").unwrap().read_raw::<i32>().unwrap(),
        [7, 11]
    );
    assert_eq!(
        native
            .dataset("shared_alias")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [7, 11]
    );
    assert_eq!(
        native
            .group("g")
            .unwrap()
            .attr("counter")
            .unwrap()
            .read_scalar::<u32>()
            .unwrap(),
        0
    );
    native.close().unwrap();
    file_space_oracle::assert_persisted_sections(&path);
    file_space_oracle::assert_manager_metadata_bytes(&path);
    directory.close().unwrap();
}

#[rstest]
#[case::two(OffsetWidth::Two)]
#[case::four(OffsetWidth::Four)]
#[case::eight(OffsetWidth::Eight)]
fn dense_legacy_reference_copy_policies_preserve_the_native_source(
    #[case] offsets: OffsetWidth,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
) {
    legacy_reference_copy(
        FormatWidths {
            offsets,
            lengths: LengthWidth::Eight,
        },
        backing,
        ReferenceSite::DenseAttribute,
    );
}

fn legacy_reference_copy(widths: FormatWidths, backing: MemoryStrategy, site: ReferenceSite) {
    let FormatWidths { offsets, lengths } = widths;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("references.h5");
    {
        let native = file::libhdf5_create_v110_with_widths(&path, offsets, lengths, 0);
        native
            .new_dataset_builder()
            .with_data(&[7_u32])
            .create("target")
            .unwrap();
        object_copy::write_legacy_reference(&native, site);
        native.close().unwrap();
    }
    object_copy::assert_legacy_reference(&path, site, 7);
    let before = std::fs::read(&path).unwrap();
    let edited = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(backing),
    )
    .unwrap();
    assert_reference_attribute_storage(&edited, &before, site);
    edited.copy("target", "copied").unwrap();
    if offsets == OffsetWidth::Eight {
        edited.commit().unwrap();
        assert_eq!(edited.dataset("copied").unwrap().read_u32().unwrap(), [7]);
        edited.close().unwrap();
    } else {
        let error = edited.commit().unwrap_err();
        assert!(
            matches!(error, Error::Format(FormatError::ReferenceWidthUnsupported { offset_size }) if offset_size == offsets.get()),
            "{error:?}"
        );
        assert_eq!(edited.disk_bytes().unwrap(), before);
        drop(edited);
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }
    object_copy::assert_legacy_reference(&path, site, 7);
    directory.close().unwrap();
}

fn assert_reference_attribute_storage(file: &File, bytes: &[u8], site: ReferenceSite) {
    if site == ReferenceSite::Dataset {
        return;
    }
    let widths = FormatWidths {
        offsets: file.superblock().offset_size,
        lengths: file.superblock().length_size,
    };
    let region = WriteEngine::gather_oh_messages(
        &BytesSource::new(bytes),
        file.root().header_address().unwrap(),
        AddressContext::from_effective_base(file.superblock().base_address),
        widths,
    )
    .unwrap();
    let mut position = 0;
    let mut compact = 0;
    let mut info = None;
    while let Some(record) = region.layout().next_message(&region, position).unwrap() {
        match record.msg_type {
            MessageType::ATTRIBUTE => compact += 1,
            MessageType::ATTRIBUTE_INFO => {
                info = Some(
                    AttributeInfoMessage::parse_with_offset_width(record.body, widths.offsets)
                        .unwrap(),
                );
            }
            _ => {}
        }
        position = record.body_range.end;
    }
    if site == ReferenceSite::DenseAttribute {
        assert_eq!(compact, 0);
        let info = info.unwrap();
        assert!(info.fractal_heap_address.is_some());
        assert!(info.btree_name_index_address.is_some());
    } else {
        assert_eq!(compact, 1);
        assert!(info.is_none_or(|info| {
            info.fractal_heap_address.is_none() && info.btree_name_index_address.is_none()
        }));
    }
}

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
fn version_two_files_preserve_independent_widths_during_public_edits(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(Producer::Native, Producer::Rust)] producer: Producer,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("version-two.h5");
    match producer {
        Producer::Native => {
            let native = file::libhdf5_create_with_widths(&path, offsets, lengths, 0);
            native
                .new_dataset_builder()
                .with_data(&[7_i32, 11])
                .create("kept")
                .unwrap();
            native.close().unwrap();
        }
        Producer::Rust => {
            let mut builder = FileBuilder::new();
            builder
                .with_offset_width(offsets)
                .with_length_width(lengths)
                .with_libver_bounds(LibVer::V18, LibVer::V18);
            builder.create_dataset("kept").with_i32_data(&[7, 11]);
            builder.write(&path).unwrap();
        }
    }
    let edited = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(backing),
    )
    .unwrap();
    assert_eq!(edited.superblock().version(), SuperblockVersion::Two);
    edited
        .root()
        .create_dataset("added", |dataset| {
            dataset.with_i32_data(&[13, 17]);
        })
        .unwrap();
    edited
        .root()
        .set_attr("counter", AttrValue::U32(19))
        .unwrap();
    edited.commit().unwrap();
    assert_eq!(edited.dataset("kept").unwrap().read_i32().unwrap(), [7, 11]);
    assert_eq!(
        edited.dataset("added").unwrap().read_i32().unwrap(),
        [13, 17]
    );
    edited.close().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(
        &bytes[8..11],
        &[
            test_util::superblock::VERSION_2,
            offsets.get(),
            lengths.get()
        ]
    );
    let reopened = File::open(&path).unwrap();
    assert_eq!(reopened.superblock().version(), SuperblockVersion::Two);
    reopened.close().unwrap();
    let native = file::libhdf5_open(&path);
    assert_eq!(
        native.dataset("kept").unwrap().read_raw::<i32>().unwrap(),
        [7, 11]
    );
    assert_eq!(
        native.dataset("added").unwrap().read_raw::<i32>().unwrap(),
        [13, 17]
    );
    assert_eq!(
        native
            .attr("counter")
            .unwrap()
            .read_scalar::<u32>()
            .unwrap(),
        19
    );
    native.close().unwrap();
    directory.close().unwrap();
}

fn assert_dense(file: &File, dense: bool) {
    let bytes = file.disk_bytes().unwrap();
    let source = BytesSource::new(bytes.as_slice());
    let widths = FormatWidths {
        offsets: file.superblock().offset_size,
        lengths: file.superblock().length_size,
    };
    let address = file.group("g").unwrap().header_address().unwrap();
    let region = WriteEngine::gather_oh_messages(
        &source,
        address,
        AddressContext::from_effective_base(file.superblock().base_address),
        widths,
    )
    .unwrap();
    let mut position = 0;
    let mut links = None;
    let mut attributes = None;
    while let Some(record) = region.layout().next_message(&region, position).unwrap() {
        match record.msg_type {
            MessageType::LINK_INFO => {
                links = Some(
                    LinkInfoMessage::parse_with_offset_width(record.body, widths.offsets).unwrap(),
                )
            }
            MessageType::ATTRIBUTE_INFO => {
                attributes = Some(
                    AttributeInfoMessage::parse_with_offset_width(record.body, widths.offsets)
                        .unwrap(),
                )
            }
            _ => {}
        }
        position = record.body_range.end;
    }
    let links = links.unwrap();
    assert_eq!(links.fractal_heap_address.is_some(), dense);
    assert_eq!(links.btree_name_index_address.is_some(), dense);
    let attributes = attributes.unwrap();
    assert_eq!(attributes.fractal_heap_address.is_some(), dense);
    assert_eq!(attributes.btree_name_index_address.is_some(), dense);
}

fn numeric_lifecycle(
    widths: FormatWidths,
    backing: MemoryStrategy,
    strategy: FileSpaceStrategy,
    producer: Producer,
    index: Index,
    filtered: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("lifecycle.h5");
    let FormatWidths { offsets, lengths } = widths;
    create(&path, widths, settings(strategy), producer, index, filtered);
    let ends = match index {
        Index::Finite(FiniteChunkIndex::Single) => [2, 3, 4],
        _ => [3, 6, 9],
    };
    let mut current = 1;
    for (cycle, end) in ends.into_iter().enumerate() {
        let edited = File::open_rw_with_options(
            &path,
            FileAccessProperties::new()
                .with_memory_strategy(backing)
                .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict),
        )
        .unwrap();
        let mut series = edited.dataset("series").unwrap();
        assert_eq!(
            series.layout().unwrap(),
            Layout::Chunked {
                chunk_shape: vec![4],
                index: match index {
                    Index::Extensible => ChunkIndex::ExtensibleArray,
                    Index::Finite(FiniteChunkIndex::Single) => ChunkIndex::SingleChunk,
                    Index::Finite(FiniteChunkIndex::Fixed) => ChunkIndex::FixedArray,
                },
            }
        );
        assert_eq!(series.maxshape().unwrap(), Some(vec![index.maximum()]));
        assert_eq!(
            series.filters().unwrap(),
            if filtered {
                vec![FilterId::Deflate]
            } else {
                vec![]
            }
        );
        let values = (current..end).collect::<Vec<i32>>();
        match cycle {
            0 => series.append(&values).unwrap(),
            1 => series
                .append_staged(|append| {
                    append.append(&values);
                })
                .unwrap(),
            _ => match index {
                Index::Finite(FiniteChunkIndex::Single) => {
                    let mut buffered = series.buffered_appender().unwrap();
                    buffered.append(&values).unwrap();
                    buffered.finish().unwrap();
                }
                _ => {
                    let mut buffered = series.clone().into_buffered_appender().unwrap();
                    buffered.append(&values).unwrap();
                    buffered.finish().unwrap();
                }
            },
        }
        let group = edited.group("g").unwrap();
        group
            .set_attr("counter", AttrValue::U32(u32::try_from(cycle).unwrap()))
            .unwrap();
        edited.commit().unwrap();
        let expected = (0..end).collect::<Vec<i32>>();
        assert_eq!(series.read_i32().unwrap(), expected);
        assert_eq!(
            edited.dataset("shared").unwrap().read_i32().unwrap(),
            [7, 11]
        );
        assert_eq!(
            edited.dataset("shared_alias").unwrap().read_i32().unwrap(),
            [7, 11]
        );
        let stats = edited.reusable_space_stats().unwrap();
        let sections = edited.reusable_free_sections().unwrap();
        assert_eq!(stats.reusable_free_bytes(), sections.free_bytes());
        assert_eq!(stats.logical_size(), edited.file_size());
        assert_eq!(edited.superblock().offset_size, offsets);
        assert_eq!(edited.superblock().length_size, lengths);
        drop(series);
        drop(group);
        edited.close().unwrap();
        assert_eq!(
            &std::fs::read(&path).unwrap()[9..11],
            &[offsets.get(), lengths.get()]
        );
        {
            let native = file::libhdf5_open(&path);
            let sizes = native.create_plist().unwrap().get_sizes().unwrap();
            assert_eq!(usize::from(sizes.sizeof_addr), usize::from(offsets.get()));
            assert_eq!(usize::from(sizes.sizeof_size), usize::from(lengths.get()));
            assert_eq!(
                native.dataset("series").unwrap().read_raw::<i32>().unwrap(),
                expected
            );
            assert_eq!(
                native.dataset("shared").unwrap().read_raw::<i32>().unwrap(),
                [7, 11]
            );
            assert_eq!(
                native
                    .dataset("shared_alias")
                    .unwrap()
                    .read_raw::<i32>()
                    .unwrap(),
                [7, 11]
            );
            assert_eq!(
                native
                    .group("g")
                    .unwrap()
                    .attr("counter")
                    .unwrap()
                    .read_scalar::<u32>()
                    .unwrap(),
                u32::try_from(cycle).unwrap()
            );
        }
        if matches!(
            strategy,
            FileSpaceStrategy::FsmAggr | FileSpaceStrategy::Page
        ) {
            file_space_oracle::assert_persisted_sections(&path);
            file_space_oracle::assert_manager_metadata_bytes(&path);
        }
        current = end;
    }
    directory.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum Producer {
    Native,
    Rust,
}

#[derive(Clone, Copy, Debug)]
enum Index {
    Extensible,
    Finite(FiniteChunkIndex),
}

impl Index {
    fn maximum(self) -> MaxExtent {
        match self {
            Self::Extensible => MaxExtent::Unlimited,
            Self::Finite(index) => MaxExtent::Fixed(index.maximum()),
        }
    }
}

fn settings(strategy: FileSpaceStrategy) -> FileSpaceSettings {
    FileSpaceSettings::new(strategy)
        .with_persistence(matches!(
            strategy,
            FileSpaceStrategy::FsmAggr | FileSpaceStrategy::Page
        ))
        .with_threshold(1)
        .with_page_size(FileSpacePageSize::try_from(512).unwrap())
}

fn create(
    path: &Path,
    widths: FormatWidths,
    settings: FileSpaceSettings,
    producer: Producer,
    index: Index,
    filtered: bool,
) {
    create_with_base(
        path,
        widths,
        settings,
        producer,
        index,
        filtered,
        UserblockSize::NONE,
    );
}

fn create_with_base(
    path: &Path,
    widths: FormatWidths,
    settings: FileSpaceSettings,
    producer: Producer,
    index: Index,
    filtered: bool,
    userblock: UserblockSize,
) {
    match producer {
        Producer::Native => {
            let native = file::libhdf5_create_with_widths_and_space(
                path,
                widths.offsets,
                widths.lengths,
                userblock.get(),
                settings,
            );
            let mut builder = native
                .new_dataset_builder()
                .empty_as(ByteOrder::LittleEndian.i32_datatype())
                .chunk((4,));
            if filtered {
                builder = builder.deflate(1);
            }
            let series = match index.maximum() {
                MaxExtent::Fixed(maximum) => builder
                    .shape((4..=usize::try_from(maximum).unwrap(),))
                    .create("series")
                    .unwrap(),
                MaxExtent::Unlimited => builder.shape((1..,)).create("series").unwrap(),
            };
            series.resize((1,)).unwrap();
            series.write_raw(&[0_i32]).unwrap();
            drop(series);
            native
                .new_dataset_builder()
                .with_data(&[7_i32, 11])
                .create("shared")
                .unwrap();
            native.create_group("g").unwrap();
            native.link_hard("shared", "shared_alias").unwrap();
            native.close().unwrap();
        }
        Producer::Rust => {
            let mut builder = FileBuilder::new();
            builder
                .with_offset_width(widths.offsets)
                .with_length_width(widths.lengths)
                .with_file_space_settings(settings)
                .with_userblock(userblock);
            let series = builder.create_dataset("series");
            series
                .with_i32_data(&[0])
                .with_maxshape(&[index.maximum()])
                .with_chunks(&[4]);
            if filtered {
                series.with_deflate(DeflateLevel::One);
            }
            builder.create_dataset("shared").with_i32_data(&[7, 11]);
            builder.create_group_with("g", |_| Ok(())).unwrap();
            builder.write(path).unwrap();
        }
    }
    if matches!(producer, Producer::Rust) {
        let native = file::libhdf5_open_rw(path);
        native.link_hard("shared", "shared_alias").unwrap();
        native.close().unwrap();
    }
}
