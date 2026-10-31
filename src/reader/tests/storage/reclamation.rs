use std::path::Path;

use rstest::rstest;
use test_util::range;
use test_util::temp;
use test_util_hdf5::dataset::{Filter, Unlimited};

use crate::FileSpaceSettings;
use crate::PhysicalAddress;
use crate::edit::SpaceOwnership;
use crate::reader::test_util as reader_test_util;
use crate::{
    AttrValue, DeflateLevel, Error, File, FileAccessProperties, FileBuilder, FileSpacePageSize,
    FileSpaceStrategy, FormatError, MaxExtent, SyncPolicy,
};

#[rstest]
#[case::fsm(FileSpaceStrategy::FsmAggr)]
#[case::page(FileSpaceStrategy::Page)]
fn edited_storage_partitions_every_byte_across_close_and_reopen(
    #[case] strategy: FileSpaceStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("partition.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(true));
    builder.create_dataset("first").with_u8_data(&[7; 8_192]);
    builder.create_dataset("hole").with_u8_data(&[9; 8_192]);
    builder.create_dataset("last").with_u8_data(&[11; 8_192]);
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    reader_test_util::assert_exact_space_ownership(&file);
    file.root().delete("hole").unwrap();
    file.commit().unwrap();
    let owned = reader_test_util::assert_exact_space_ownership(&file);
    assert!(!owned.reusable_free_space.is_empty());
    assert!(!owned.free_space_manager_space.is_empty());
    for payload in [
        reader_test_util::contiguous_extent(&file, "first"),
        reader_test_util::contiguous_extent(&file, "last"),
    ] {
        assert_eq!(
            range::covered_len(&owned.live_raw_space, &payload),
            payload.end - payload.start
        );
        assert_eq!(range::covered_len(&owned.reusable_free_space, &payload), 0);
    }
    for &(address, length) in &owned.free_space_manager_space {
        let allocation = address..address.checked_add(length).unwrap();
        assert_eq!(
            range::covered_len(&owned.reusable_free_space, &allocation),
            0
        );
    }
    let survivor = file.clone();
    file.close().unwrap();
    reader_test_util::ownership(&survivor)
        .unwrap()
        .assert_exact();
    drop(survivor);
    let reopened = File::open_rw(&path).unwrap();
    reader_test_util::assert_exact_space_ownership(&reopened);
    assert_eq!(
        reopened.dataset("first").unwrap().read_u8().unwrap(),
        [7; 8_192]
    );
    assert_eq!(
        reopened.dataset("last").unwrap().read_u8().unwrap(),
        [11; 8_192]
    );
}

#[rstest]
#[case::below_threshold(1_025)]
#[case::at_threshold(1_024)]
#[case::above_threshold(1_023)]
fn isolated_deletion_classifies_exactly_the_bytes_admitted_by_the_threshold(
    #[case] threshold: u64,
    #[values(false, true)] persist: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("threshold.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(persist)
            .with_threshold(threshold),
    );
    builder.create_dataset("first").with_i32_data(&[7; 256]);
    builder.create_dataset("hole").with_i32_data(&[9; 256]);
    builder.create_dataset("last").with_i32_data(&[11; 256]);
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let deleted = reader_test_util::contiguous_extent(&file, "hole");
    let length = deleted.end - deleted.start;
    file.root().delete("hole").unwrap();
    file.commit().unwrap();
    let owned = reader_test_util::ownership(&file).unwrap();
    assert!(owned.analysis_complete);
    let accounted = [
        &owned.reusable_free_space,
        &owned.live_metadata_space,
        &owned.free_space_manager_space,
    ]
    .into_iter()
    .map(|sections| range::covered_len(sections, &deleted))
    .sum::<u64>();
    assert_eq!(accounted, if length >= threshold { length } else { 0 });
    if length < threshold {
        assert_eq!(
            range::covered_len(&owned.unclassified_space, &deleted),
            length
        );
        assert_eq!(range::covered_len(&owned.reusable_free_space, &deleted), 0);
    }
    assert_eq!(file.dataset("first").unwrap().read_i32().unwrap(), [7; 256]);
    assert_eq!(file.dataset("last").unwrap().read_i32().unwrap(), [11; 256]);
}

#[test]
fn a_paged_commit_tail_is_placed_in_free_space() {
    const PAGE: u64 = 16_384;
    let path = temp::temp_path("hdf5_pure_fs_paged_tail_reuse.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("seed").with_i32_data(&[0i32; 4]);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_page_size(FileSpacePageSize::try_from(PAGE).unwrap()),
    );
    b.write(&path).unwrap();
    let start_len = std::fs::metadata(&path).unwrap().len();

    for i in 0..3 {
        let (before, before_capacity) = {
            let file = File::open_rw(&path).unwrap();
            let capacity: u64 = reader_test_util::ownership(&file)
                .unwrap()
                .free_space_manager_space
                .iter()
                .map(|&(_, len)| len)
                .sum();
            (file.reusable_free_sections().unwrap(), capacity)
        };
        let f = File::open_rw(&path).unwrap();
        f.root().set_attr("n", AttrValue::I64(i)).unwrap();
        f.commit().unwrap();
        drop(f);

        let f = File::open_rw(&path).unwrap();
        let managers: Vec<u64> = f
            .file_space_info()
            .unwrap()
            .expect("a persisting file records its managers")
            .manager_addrs
            .iter()
            .filter_map(|&a| match a {
                crate::DecodedStoredAddress::Defined(addr) => Some(addr.get()),
                crate::DecodedStoredAddress::Undefined => None,
            })
            .collect();
        assert!(
            !managers.is_empty(),
            "commit {i}: the file has free space, so it has managers to place"
        );
        for addr in managers {
            assert!(
                before
                    .sections()
                    .iter()
                    .any(|section| addr >= section.address().get()
                        && addr < section.end_address().get()),
                "commit {i}: a manager block landed at {addr}, which was not free \
                 before the commit ({before:?}) — the tail appended instead of \
                 reusing"
            );
        }
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            start_len,
            "commit {i}: rewriting the tail must not grow the file"
        );
        // Free bytes plus complete manager capacity stay constant when the file does not grow.
        if i > 0 {
            let before_total = before.free_bytes();
            let after_total: u64 = f.reusable_space_stats().unwrap().reusable_free_bytes();
            assert_eq!(
                after_total
                    + reader_test_util::ownership(&f)
                        .unwrap()
                        .free_space_manager_space
                        .iter()
                        .map(|&(_, len)| len)
                        .sum::<u64>(),
                before_total + before_capacity,
                "commit {i}: rewriting the tail must conserve free space"
            );
        }
    }
    reader_test_util::assert_eof_matches_physical_size(&path);
}

#[test]
fn a_reused_flat_manager_tail_preserves_its_spare_allocation_across_reopen() {
    const FILLER: usize = 7;

    let path = temp::temp_path("hdf5_pure_fs_flat_tail_allocated_suffix.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("seed")
        .with_i32_data(&(0..100 + FILLER as i32).collect::<Vec<i32>>());
    for (index, len) in [256usize, 37, 91].into_iter().enumerate() {
        builder
            .create_dataset(&format!("scratch{index}"))
            .with_i32_data(&vec![7i32; len + FILLER]);
        builder
            .create_dataset(&format!("keep{index}"))
            .with_i32_data(&[9i32; 4]);
    }
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    builder.write(&path).unwrap();

    let open = || {
        File::open_rw_with_options(
            &path,
            FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
        )
        .unwrap()
    };

    for index in 0..3 {
        let file = open();
        file.root().delete(&format!("scratch{index}")).unwrap();
        file.commit().unwrap();
        file.close().unwrap();
    }

    let mut spare_tail = None;
    for generation in 0..6 {
        let file = open();
        file.root()
            .set_attr("n", AttrValue::I64(generation))
            .unwrap();
        file.commit().unwrap();
        let ownership = reader_test_util::assert_exact_space_ownership(&file);
        let physical_size = std::fs::metadata(&path).unwrap().len();
        file.close().unwrap();
        let spare = reader_test_util::serialized_manager_padding(&path);
        if spare > 0 {
            assert!(
                ownership
                    .free_space_manager_space
                    .iter()
                    .map(|&(_, len)| len)
                    .sum::<u64>()
                    > spare,
                "the padding must be part of a larger manager allocation"
            );
            spare_tail = Some((spare, physical_size));
            break;
        }
    }
    let (_, size_with_spare_tail) = spare_tail
        .expect("filler 7 must produce a reused flat tail with serialized section-list padding");

    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        size_with_spare_tail,
        "closing the writer must preserve the complete tail allocation"
    );
    let reader = File::open(&path).unwrap();
    let persisted_free: u64 = reader
        .persisted_free_sections()
        .unwrap()
        .unwrap()
        .free_sections()
        .free_bytes();
    drop(reader);

    let reopened = open();
    let recovered = reopened.reusable_space_stats().unwrap();
    assert_eq!(
        recovered.reusable_free_bytes(),
        persisted_free,
        "reopen must recover the same reusable free-space total"
    );
    let ownership = reader_test_util::assert_exact_space_ownership(&reopened);
    assert_eq!(ownership.unclassified_space, []);

    reopened
        .root()
        .set_attr("n", AttrValue::I64(10_000))
        .unwrap();
    reopened.commit().unwrap();
    let ownership = reader_test_util::assert_exact_space_ownership(&reopened);
    assert_eq!(ownership.unclassified_space, []);
    assert!(
        std::fs::metadata(&path).unwrap().len() <= size_with_spare_tail,
        "rewriting the recovered manager tail should reuse its complete previous allocation"
    );
    reopened.close().unwrap();

    reader_test_util::assert_eof_matches_physical_size(&path);
    let final_reader = File::open(&path).unwrap();
    assert_eq!(
        final_reader.dataset("seed").unwrap().read_i32().unwrap(),
        (0..100 + FILLER as i32).collect::<Vec<i32>>()
    );
    for index in 0..3 {
        assert_eq!(
            final_reader
                .dataset(&format!("keep{index}"))
                .unwrap()
                .read_i32()
                .unwrap(),
            vec![9, 9, 9, 9]
        );
        let path = format!("scratch{index}");
        let err = final_reader.dataset(&path).unwrap_err();
        let Error::Format(FormatError::PathNotFound(missing)) = &err else {
            panic!("expected PathNotFound, got {err:?}");
        };
        assert_eq!(missing, &path);
    }
}

#[test]
fn empty_extensible_arrays_retain_exact_storage_coverage_through_deletion() {
    let (with_index, index_bytes) =
        scratch_delete_releases(FileSpaceStrategy::FsmAggr, true, "empty_index_owned.h5");
    with_index.close().unwrap();
    let (without_index, header_bytes) =
        scratch_delete_releases(FileSpaceStrategy::FsmAggr, false, "empty_fixed_owned.h5");
    without_index.close().unwrap();
    assert!(index_bytes - header_bytes >= 256);
}

#[test]
fn a_relocating_overwrite_reclaims_the_old_header_with_its_storage() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("relocate.h5");

    // The compressed zero-filled chunks have insufficient capacity for the replacement values.
    Unlimited::new("d", &[0i32; 64], 16)
        .filters(&[Filter::Deflate(DeflateLevel::Six)])
        .pure_create(&path);
    let original_end = std::fs::metadata(&path).unwrap().len();

    let session = File::open_rw(&path).unwrap();
    let original = reader_test_util::ownership(&session).unwrap();
    assert!(original.analysis_complete);
    let incompressible: Vec<i32> = (0..64)
        .map(|i: i32| i.wrapping_mul(0x9E37_79B1u32 as i32))
        .collect();
    session
        .dataset("d")
        .unwrap()
        .write_staged(|b| {
            b.with_i32_data(&incompressible);
        })
        .unwrap();
    session.commit().unwrap();

    let stats = session.reusable_space_stats().unwrap();
    let sections = session.reusable_free_sections().unwrap();
    test_util_hdf5::assert_free_sections_consistent!(&stats, &sections, PhysicalAddress);
    for (start, len) in original
        .live_metadata_space
        .into_iter()
        .chain(original.live_raw_space)
        .filter(|&(start, _)| start != 0)
    {
        assert_eq!(
            test_util_hdf5::covered_free_bytes!(&sections, &(start..start + len), PhysicalAddress),
            len,
            "old allocation {start}..{} is not fully reclaimed: {:?}",
            start + len,
            sections,
        );
    }
    assert_eq!(
        sections
            .sections()
            .iter()
            .filter(|section| section.address().get() < original_end)
            .count(),
        2,
    );

    drop(session);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        incompressible
    );
}

fn churn_fixture(path: &Path, strategy: FileSpaceStrategy) -> File {
    let mut b = FileBuilder::new();
    let settings = FileSpaceSettings::new(strategy)
        .with_persistence(true)
        .with_threshold(0);
    b.with_file_space_settings(if strategy == FileSpaceStrategy::Page {
        settings.with_page_size(FileSpacePageSize::try_from(RECLAIM_PAGE).unwrap())
    } else {
        settings
    });
    // The surviving group retains metadata allocations between scratch deletions.
    let mut keep = b.create_group("keep");
    keep.set_attr("kind", AttrValue::I64(1));
    b.add_group(keep.finish());
    b.write(path).unwrap();
    open_rw_on_close(path)
}

fn scratch_delete_releases(
    strategy: FileSpaceStrategy,
    resizable: bool,
    name: &str,
) -> (File, u64) {
    let path = temp::temp_path(name);
    let f = churn_fixture(&path, strategy);
    f.root()
        .create_group_with("scratch", |g| {
            g.create_dataset("log", |b| {
                b.with_i32_data(&[]).with_shape(&[0]).with_chunks(&[64]);
                if resizable {
                    b.with_maxshape(&[MaxExtent::Unlimited]);
                }
            });
        })
        .unwrap();
    f.commit().unwrap();
    let before = reader_test_util::assert_exact_space_ownership(&f);
    f.root().delete("scratch").unwrap();
    f.commit().unwrap();
    let after = reader_test_util::assert_exact_space_ownership(&f);
    (f, live_bytes(&before) - live_bytes(&after))
}

fn open_rw_on_close(path: &Path) -> File {
    File::open_rw_with_options(
        path,
        FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap()
}

fn live_bytes(owned: &SpaceOwnership) -> u64 {
    owned
        .live_raw_space
        .iter()
        .chain(&owned.live_metadata_space)
        .map(|&(_, len)| len)
        .sum()
}

const RECLAIM_PAGE: u64 = 4_096;
