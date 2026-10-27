#![cfg(feature = "__hdf5-1.10")]
//! Cross-validation of the file-space strategy (`H5Pset_file_space_strategy`)
//! against the reference HDF5 C library: a strategy hdf5-pure writes is read back
//! by the C library, and a strategy the C library writes is read back by
//! hdf5-pure — including a persisted (persist=true) file whose File Space Info
//! message carries free-space-manager addresses.

use std::ops::Range;
use std::path::Path;

use hdf5::dataset::Layout as CLayout;
use hdf5::plist::file_create::FileSpaceStrategy as CStrategy;
use hdf5::plist::file_create::Sizeof;
use hdf5::plist::file_create::SizeofInfo;
use hdf5_pure::AttrValue;
use hdf5_pure::ChunkIndex;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileCreateProperties;
use hdf5_pure::FileSpaceInfo;
use hdf5_pure::FileSpacePageSize;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FormatError;
use hdf5_pure::FreeSectionCoordinates;
use hdf5_pure::FreeSections;
use hdf5_pure::FreeSpaceManagerRole;
use hdf5_pure::Layout;
use hdf5_pure::MaxExtent;
use hdf5_pure::MemoryStrategy;
use hdf5_pure::MessageType;
use hdf5_pure::PersistedFreeSections;
use hdf5_pure::PersistedFreeSpacePolicy;
use hdf5_pure::SyncPolicy;
use hdf5_pure::UserblockSize;
use hdf5_pure_core::__private::BaseAddressExt;
use hdf5_pure_core::__private::FileSpaceInfoFields;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::OffsetWidth;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::DEFAULT_THRESHOLD;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::FreeSpaceManagerHeader;
use hdf5_pure_format::__private::LinkMessage;
use hdf5_pure_format::__private::LinkTarget;
use hdf5_pure_format::__private::NUM_FILE_FSM_MANAGERS;
use hdf5_pure_format::__private::ObjectHeader;
use hdf5_pure_format::__private::ObjectHeaderPrefix;
use hdf5_pure_format::__private::SECTION_CLASS_SIMPLE;
use rstest::rstest;

use test_util::file_space_info;
use test_util::free_space;
use test_util::object_header::v2 as header_bytes;
use test_util::range;
use test_util::userblock::Userblock;
use test_util::widths::Widths;
use test_util_hdf5::absence;
use test_util_hdf5::dataset::Unlimited;
use test_util_hdf5::file_space;
use test_util_hdf5::file_space_oracle::FreeSpaceOracleCheckpoint;
use test_util_hdf5::page_publication;
use test_util_hdf5::paged;
use test_util_hdf5::session;
use test_util_hdf5::storage;
use test_util_hdf5::userblock;

#[rstest]
#[case::large_threshold(4_096, false, false, false)]
#[case::maximum_threshold(u64::MAX, false, false, false)]
#[case::cyclic_layout(4_096, true, false, false)]
#[case::aliased_root(4_096, false, false, true)]
#[case::aliased_root_maximum_threshold(u64::MAX, false, false, true)]
#[case::aliased_root_cyclic_layout(4_096, true, false, true)]
#[case::ordinary_threshold(1, false, false, false)]
#[case::tracked_space(4_096, false, true, false)]
#[case::intermediate_threshold(200, false, true, false)]
fn c_written_fixed_width_root_updates_reach_stable_physical_length(
    #[case] threshold: u64,
    #[case] cycle: bool,
    #[case] tracked: bool,
    #[case] alias: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("root-reclamation.h5");
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(|fcpl| fcpl.file_space_strategy(fsm_aggr(true, threshold)))
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape([3])
            .create("keep")
            .unwrap()
            .write_raw(&[1, 2, 3])
            .unwrap();
        file.new_attr::<i64>()
            .create("generation")
            .unwrap()
            .write_scalar(&-1)
            .unwrap();
        if tracked {
            let data = vec![7u8; 8_192];
            file.new_dataset::<u8>()
                .shape([8_192])
                .create("hole")
                .unwrap()
                .write_raw(&data)
                .unwrap();
            file.new_dataset::<u8>()
                .shape([8_192])
                .create("guard")
                .unwrap()
                .write_raw(&data)
                .unwrap();
        }
        if alias {
            file.link_hard("/", "root_alias").unwrap();
        }
        if cycle {
            file.create_group("cycle")
                .unwrap()
                .link_hard(".", "again")
                .unwrap();
        }
        file.close().unwrap();
    }
    if tracked {
        let file = File::open_rw(&path).unwrap();
        file.root().delete("hole").unwrap();
        file.commit().unwrap();
        file.close().unwrap();
    }
    let mut roots = Vec::new();
    let mut lengths = Vec::new();
    let mut closed_lengths = Vec::new();
    let mut managers = Vec::new();
    for generation in 0..12 {
        let file = File::open_rw(&path).unwrap();
        file.root()
            .set_attr("generation", AttrValue::I64(generation))
            .unwrap();
        file.commit().unwrap();
        roots.push(file.superblock().root_group_address);
        lengths.push(std::fs::metadata(&path).unwrap().len());
        let sections = file.reusable_free_sections().unwrap();
        file.close().unwrap();
        closed_lengths.push(std::fs::metadata(&path).unwrap().len());
        let reopened = File::open(&path).unwrap();
        let persisted = reopened.persisted_free_sections().unwrap().unwrap();
        assert_eq!(
            persisted.free_sections().section_count(),
            sections.section_count()
        );
        assert_eq!(
            persisted.free_sections().free_bytes(),
            sections.free_bytes()
        );
        let mut previous_end = 0;
        for (stored, physical) in persisted
            .free_sections()
            .sections()
            .iter()
            .zip(sections.sections())
        {
            assert_eq!(stored.coordinates(), FreeSectionCoordinates::Stored);
            assert_eq!(physical.coordinates(), FreeSectionCoordinates::Absolute);
            assert_eq!(
                physical.address(),
                stored.address() + reopened.superblock().base_address.get()
            );
            assert_eq!(physical.length(), stored.length());
            assert!(stored.address() >= previous_end);
            previous_end = stored.end_address();
            assert!(
                previous_end <= reopened.file_size() - reopened.superblock().base_address.get()
            );
        }
        managers.push(
            reopened
                .file_space_info()
                .unwrap()
                .unwrap()
                .manager_addrs
                .clone(),
        );
        drop(reopened);
        FreeSpaceOracleCheckpoint::observe(&path, "root-reclamation", "updated")
            .unwrap()
            .assert_equal();
        let file = hdf5::File::open(&path).unwrap();
        assert_eq!(
            file.attr("generation")
                .unwrap()
                .read_scalar::<i64>()
                .unwrap(),
            generation
        );
        assert_eq!(
            file.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
            [1, 2, 3]
        );
        if alias {
            assert_eq!(
                file.group("root_alias")
                    .unwrap()
                    .attr("generation")
                    .unwrap()
                    .read_scalar::<i64>()
                    .unwrap(),
                -1
            );
        }
        if cycle {
            assert!(file.link_exists("cycle/again/again"));
        }
        file.close().unwrap();
    }
    if threshold > 1 {
        assert!(roots[4..].iter().all(|&root| root == roots[4]), "{roots:?}");
        assert!(
            managers[4..].iter().all(|set| set == &managers[4]),
            "{managers:?}"
        );
    }
    assert!(
        lengths[4..].iter().all(|&len| len == lengths[4]),
        "{lengths:?}"
    );
    assert!(
        closed_lengths[4..]
            .iter()
            .all(|&len| len == closed_lengths[4]),
        "{closed_lengths:?}"
    );
}

#[rstest]
#[case::before_exact_fill(false, 512, 0, 17, 512)]
#[case::exact_fill(false, 512, 0, 18, 512)]
#[case::after_exact_fill(false, 512, 0, 19, 512)]
#[case::metadata_only(false, 512, 0, 18, 0)]
#[case::raw_manager(false, 512, 0, 18, 511)]
#[case::large_manager(false, 512, 0, 18, 513)]
#[case::raw_and_metadata_managers(false, 512, 0, 412, 511)]
#[case::large_and_metadata_managers(false, 512, 0, 412, 513)]
#[case::userblock(false, 512, 4096, 18, 512)]
#[case::default_page(false, 4096, 4096, 3601, 4096)]
#[case::larger_page(false, 8192, 8192, 7697, 8192)]
#[case::c_created(true, 512, 0, 18, 512)]
fn created_page_tail_survives_reference_library_mutation(
    #[case] c_created: bool,
    #[case] page: u64,
    #[case] base: u64,
    #[case] attribute_len: usize,
    #[case] raw_len: usize,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("creation_tail.h5");
    if c_created {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(|fcpl| {
                fcpl.file_space_strategy(CStrategy::FreeSpaceManager {
                    paged: true,
                    persist: true,
                    threshold: 1,
                })
                .file_space_page_size(page)
                .userblock(base)
            })
            .create(&path)
            .unwrap();
        file.new_attr::<u8>()
            .shape([attribute_len])
            .create("fill")
            .unwrap()
            .write_raw(&vec![0u8; attribute_len])
            .unwrap();
        file.new_dataset::<u8>()
            .shape([raw_len])
            .create("d")
            .unwrap()
            .write_raw(&vec![7u8; raw_len])
            .unwrap();
        file.close().unwrap();
    } else {
        paged::write_metadata_tail(&path, page, base, 1, attribute_len, raw_len);
    }
    let userblock = Userblock::read(&path, usize::try_from(base).unwrap());
    let file = File::open(&path).unwrap();
    let free: u64 = file
        .persisted_free_sections()
        .unwrap()
        .unwrap()
        .free_sections()
        .free_bytes();
    drop(file);
    let bytes = std::fs::read(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    storage::assert_unavailable(&file, &storage::file_metadata_extents(&file, &bytes));
    file.close().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    if !c_created
        && ((attribute_len == 18 && raw_len == 512)
            || (attribute_len == 412 && (raw_len == 511 || raw_len == 513)))
    {
        assert_eq!(file_space::serialized_manager_padding(&path), 1);
    }
    let file = hdf5::File::open_rw(&path).unwrap();
    assert_eq!(file.free_space(), free);
    assert_eq!(
        file.create_plist()
            .unwrap()
            .get_file_space_strategy()
            .unwrap(),
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 1
        }
    );
    assert_eq!(
        file.dataset("d").unwrap().read_raw::<u8>().unwrap(),
        vec![7; raw_len]
    );
    assert_eq!(
        file.attr("fill").unwrap().read_raw::<u8>().unwrap(),
        vec![0; attribute_len]
    );
    file.dataset("d")
        .unwrap()
        .write_raw(&vec![11u8; raw_len])
        .unwrap();
    file.attr("fill")
        .unwrap()
        .write_raw(&vec![13u8; attribute_len])
        .unwrap();
    file.new_dataset::<u8>()
        .shape([32])
        .create("from_c")
        .unwrap()
        .write_raw(&[17u8; 32])
        .unwrap();
    file.close().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_u8().unwrap(),
        vec![11; raw_len]
    );
    assert_eq!(
        file.dataset("from_c").unwrap().read_u8().unwrap(),
        vec![17; 32]
    );
    assert_eq!(
        file.root().attrs().unwrap()["fill"],
        AttrValue::U8Array(vec![13; attribute_len])
    );
    storage::assert_unavailable(&file, &storage::file_metadata_extents(&file, &bytes));
    file.root()
        .create_dataset("from_rust", |builder| {
            builder.with_u8_data(&[19; 32]);
        })
        .unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_u8().unwrap(),
        vec![11; raw_len]
    );
    assert_eq!(
        file.dataset("from_c").unwrap().read_u8().unwrap(),
        vec![17; 32]
    );
    assert_eq!(
        file.dataset("from_rust").unwrap().read_u8().unwrap(),
        vec![19; 32]
    );
    userblock.assert_unchanged(&path);
}

#[rstest]
#[case::rust_small(false, 512, 0, 1)]
#[case::c_small(true, 512, 0, 1)]
#[case::rust_default(false, 4_096, 0, 1)]
#[case::c_threshold(true, 4_096, 0, 8_193)]
#[case::rust_userblock(false, 8_192, 8_192, 1)]
#[case::c_userblock(true, 8_192, 8_192, 1)]
fn paged_manager_capacity_survives_reference_library_mutation(
    #[case] c_created: bool,
    #[case] page: u64,
    #[case] base: u64,
    #[case] threshold: u64,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("manager_capacity.h5");
    let raw = vec![3u8; usize::try_from(2 * page).unwrap()];
    if c_created {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(|fcpl| {
                fcpl.file_space_strategy(CStrategy::FreeSpaceManager {
                    paged: true,
                    persist: true,
                    threshold,
                })
                .file_space_page_size(page)
                .userblock(base)
            })
            .create(&path)
            .unwrap();
        file.new_dataset::<u8>()
            .shape([100])
            .create("keep")
            .unwrap()
            .write_raw(&[7u8; 100])
            .unwrap();
        file.new_dataset::<u8>()
            .shape([raw.len()])
            .create("victim")
            .unwrap()
            .write_raw(&raw)
            .unwrap();
        file.new_dataset::<u8>()
            .shape([100])
            .create("ceiling")
            .unwrap()
            .write_raw(&[5u8; 100])
            .unwrap();
        file.close().unwrap();
    } else {
        let mut builder = FileBuilder::new();
        builder
            .with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::Page)
                    .with_persistence(true)
                    .with_threshold(threshold)
                    .with_page_size(FileSpacePageSize::try_from(page).unwrap()),
            )
            .with_userblock(UserblockSize::try_from(base).unwrap());
        builder.create_dataset("keep").with_u8_data(&[7; 100]);
        builder.create_dataset("victim").with_u8_data(&raw);
        builder.create_dataset("ceiling").with_u8_data(&[5; 100]);
        builder.write(&path).unwrap();
    }
    let userblock = Userblock::read(&path, usize::try_from(base).unwrap());
    let mut file = File::open_rw(&path).unwrap();
    file.root().delete("victim").unwrap();
    file.root()
        .create_dataset("added", |builder| {
            builder.with_u8_data(&[9; 100]);
        })
        .unwrap();
    file.commit().unwrap();
    for generation in 0..3 {
        file.root()
            .set_attr("generation", AttrValue::I64(generation))
            .unwrap();
        file.commit().unwrap();
        file.close().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        file = File::open_rw(&path).unwrap();
        let mut allocated = storage::file_metadata_extents(&file, &bytes);
        for name in ["keep", "ceiling", "added"] {
            allocated.extend(storage::header_extents(
                file.superblock(),
                &bytes,
                storage::root_object_address(&file, &bytes, name),
            ));
            allocated.extend(
                file_space::raw_extents(&file, name)
                    .iter()
                    .map(|extent| (extent.start, extent.end - extent.start)),
            );
        }
        storage::assert_unavailable(&file, &allocated);
    }
    let reusable = file.reusable_free_sections().unwrap();
    file.close().unwrap();
    assert!(file_space::serialized_manager_padding(&path) > 0);
    let total = reusable.free_bytes();
    let file = hdf5::File::open_rw(&path).unwrap();
    assert_eq!(file.free_space(), total);
    assert_eq!(
        file.dataset("keep").unwrap().read_raw::<u8>().unwrap(),
        [7; 100]
    );
    assert_eq!(
        file.dataset("added").unwrap().read_raw::<u8>().unwrap(),
        [9; 100]
    );
    assert_eq!(
        file.dataset("ceiling").unwrap().read_raw::<u8>().unwrap(),
        [5; 100]
    );
    absence::assert_libhdf5_absent(&file.dataset("victim").unwrap_err(), "victim");
    file.dataset("keep")
        .unwrap()
        .write_raw(&[11u8; 100])
        .unwrap();
    let added = file
        .new_dataset::<u8>()
        .shape([100])
        .create("c_added")
        .unwrap();
    added.write_raw(&[13u8; 100]).unwrap();
    let allocation = file_space::libhdf5_contiguous_extent(&added);
    assert_eq!(
        test_util_hdf5::covered_free_bytes!(
            &reusable,
            &allocation,
            FreeSectionCoordinates::Absolute
        ),
        100
    );
    drop(added);
    file.close().unwrap();
    userblock.assert_unchanged(&path);
    let file = File::open_rw(&path).unwrap();
    assert_eq!(file.dataset("keep").unwrap().read_u8().unwrap(), [11; 100]);
    assert_eq!(file.dataset("added").unwrap().read_u8().unwrap(), [9; 100]);
    assert_eq!(
        file.dataset("ceiling").unwrap().read_u8().unwrap(),
        [5; 100]
    );
    assert_eq!(
        file.dataset("c_added").unwrap().read_u8().unwrap(),
        [13; 100]
    );
    assert_eq!(
        file.file_space_info().unwrap().unwrap().strategy,
        FileSpaceStrategy::Page
    );
    assert!(file.file_space_info().unwrap().unwrap().persist);
    file.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum HeaderReclamation {
    DeleteDataset,
    DeleteDenseGroup,
    DeleteGroup,
    EditDatasetAttribute,
    RebuildGroup,
    ReplaceDataset,
    UnlinkDataset,
}

#[rstest]
#[case::delete_512(512, 0, HeaderReclamation::DeleteDataset)]
#[case::replace_512(512, 0, HeaderReclamation::ReplaceDataset)]
#[case::attribute_512(512, 0, HeaderReclamation::EditDatasetAttribute)]
#[case::delete_group_512(512, 0, HeaderReclamation::DeleteGroup)]
#[case::rebuild_group_512(512, 0, HeaderReclamation::RebuildGroup)]
#[case::dense_group_512(512, 0, HeaderReclamation::DeleteDenseGroup)]
#[case::hard_link_512(512, 0, HeaderReclamation::UnlinkDataset)]
#[case::delete_4096(4_096, 0, HeaderReclamation::DeleteDataset)]
#[case::replace_4096(4_096, 0, HeaderReclamation::ReplaceDataset)]
#[case::attribute_userblock(4_096, 512, HeaderReclamation::EditDatasetAttribute)]
#[case::delete_group_4096(4_096, 0, HeaderReclamation::DeleteGroup)]
#[case::rebuild_group_userblock(4_096, 512, HeaderReclamation::RebuildGroup)]
#[case::dense_group_4096(4_096, 0, HeaderReclamation::DeleteDenseGroup)]
#[case::delete_8192(8_192, 0, HeaderReclamation::DeleteDataset)]
#[case::replace_8192(8_192, 8_192, HeaderReclamation::ReplaceDataset)]
fn reclaimed_page_header_continuations_remain_compatible_with_c_mutation(
    #[case] page: u64,
    #[case] userblock: u64,
    #[case] action: HeaderReclamation,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("header_reclamation.h5");
    write_c_large_header(&path, page, action);
    if userblock != 0 {
        userblock::prepend(&path, &vec![0xA5; usize::try_from(userblock).unwrap()]);
    }
    let original = std::fs::read(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let owner = match action {
        HeaderReclamation::DeleteDenseGroup
        | HeaderReclamation::DeleteGroup
        | HeaderReclamation::RebuildGroup => "group",
        _ => "target",
    };
    let address = storage::root_object_address(&file, &original, owner);
    let continuations = storage::header_extents(file.superblock(), &original, address)
        .into_iter()
        .filter(|&(at, len)| {
            len >= page
                && original[usize::try_from(at).unwrap()..]
                    .starts_with(header_bytes::CONTINUATION_SIGNATURE)
        })
        .collect::<Vec<_>>();
    assert_eq!(continuations.len(), 1, "{continuations:?}");
    let (address, length) = continuations[0];
    assert_eq!((address - userblock) % page, 0);
    assert_ne!(length % page, 0);
    if userblock == 512 && page == 4_096 {
        assert_ne!(address % page, 0);
    }
    let partial = address + length / page * page..address + length;
    let generic_fragment =
        partial.end..userblock + (partial.end - userblock).next_multiple_of(page);
    assert!(
        File::from_bytes(original.clone())
            .unwrap()
            .persisted_free_sections()
            .unwrap()
            .unwrap()
            .free_sections()
            .sections()
            .iter()
            .any(|section| {
                section.coordinates() == FreeSectionCoordinates::Stored
                    && section.address() + userblock == generic_fragment.start
                    && section.end_address() + userblock == generic_fragment.end
            }),
        "continuations={continuations:?}, fragment={generic_fragment:?}, free={:?}",
        File::from_bytes(original.clone())
            .unwrap()
            .persisted_free_sections()
            .unwrap()
            .unwrap()
    );
    match action {
        HeaderReclamation::DeleteDataset | HeaderReclamation::UnlinkDataset => {
            file.root().delete("target").unwrap()
        }
        HeaderReclamation::DeleteDenseGroup | HeaderReclamation::DeleteGroup => {
            file.root().delete("group").unwrap()
        }
        HeaderReclamation::EditDatasetAttribute => file
            .dataset("target")
            .unwrap()
            .set_attr("changed", AttrValue::I32(17))
            .unwrap(),
        HeaderReclamation::RebuildGroup => file
            .group("group")
            .unwrap()
            .set_attr("changed", AttrValue::I32(17))
            .unwrap(),
        HeaderReclamation::ReplaceDataset => file
            .dataset("target")
            .unwrap()
            .write_staged(|builder| {
                builder.with_u8_data(&[17; HEADER_TARGET_ELEMENTS]);
            })
            .unwrap(),
    }
    file.commit().unwrap();
    assert_header_reclamation_state(&file, &partial, &generic_fragment, action);
    if matches!(action, HeaderReclamation::UnlinkDataset) {
        assert_eq!(
            test_util_hdf5::covered_free_bytes!(
                &file.reusable_free_sections().unwrap(),
                &(address..address + length),
                FreeSectionCoordinates::Absolute
            ),
            0
        );
        assert_eq!(
            file.dataset("alias").unwrap().read_u8().unwrap(),
            vec![3; HEADER_TARGET_ELEMENTS]
        );
    }
    file.close().unwrap();
    let file = File::open_rw(&path).unwrap();
    assert_header_reclamation_state(&file, &partial, &generic_fragment, action);
    file.close().unwrap();
    let persisted_bytes: u64 = File::from_bytes(std::fs::read(&path).unwrap())
        .unwrap()
        .persisted_free_sections()
        .unwrap()
        .unwrap()
        .free_sections()
        .free_bytes();

    let file = hdf5::File::open_rw(&path).unwrap();
    assert_eq!(file.free_space(), persisted_bytes);
    assert_c_header_survivors(&file, action);
    file.new_dataset_builder()
        .with_data(&[23i32; 64])
        .create("from_c")
        .unwrap();
    file.dataset("keep")
        .unwrap()
        .write_raw(&[11i32; 64])
        .unwrap();
    file.close().unwrap();
    let file = File::open_rw(&path).unwrap();
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [11; 64]);
    assert_eq!(
        file.dataset("from_c").unwrap().read_i32().unwrap(),
        [23; 64]
    );
    assert_rust_header_survivors(&file, action, 11);
    file.close().unwrap();
    if userblock != 0 {
        assert_eq!(
            &std::fs::read(&path).unwrap()[..usize::try_from(userblock).unwrap()],
            &original[..usize::try_from(userblock).unwrap()]
        );
    }
}

fn write_c_large_header(path: &Path, page: u64, action: HeaderReclamation) {
    let file = hdf5::FileBuilder::new()
        .with_fapl(|fapl| fapl.libver_v110())
        .with_fcpl(|fcpl| {
            fcpl.file_space_strategy(CStrategy::FreeSpaceManager {
                paged: true,
                persist: true,
                threshold: 1,
            })
            .file_space_page_size(page)
        })
        .create(path)
        .unwrap();
    let group = matches!(
        action,
        HeaderReclamation::DeleteGroup
            | HeaderReclamation::DeleteDenseGroup
            | HeaderReclamation::RebuildGroup
    );
    let owner: hdf5::Location = if group {
        file.create_group("group").unwrap().as_location().unwrap()
    } else {
        let dataset = file
            .new_dataset::<u8>()
            .shape([HEADER_TARGET_ELEMENTS])
            .layout(if matches!(action, HeaderReclamation::ReplaceDataset) {
                CLayout::Compact
            } else {
                CLayout::Contiguous
            })
            .create("target")
            .unwrap();
        dataset.write_raw(&[3u8; HEADER_TARGET_ELEMENTS]).unwrap();
        dataset.as_location().unwrap()
    };
    let elements = usize::try_from((page / 4).max(512)).unwrap();
    owner
        .new_attr::<i64>()
        .shape([elements])
        .create("big_attr")
        .unwrap()
        .write_raw(&vec![7i64; elements])
        .unwrap();
    if group {
        let group = file.group("group").unwrap();
        let count = if matches!(action, HeaderReclamation::DeleteDenseGroup) {
            12
        } else {
            1
        };
        for index in 0..count {
            group
                .new_dataset_builder()
                .with_data(&[index; 4])
                .create(format!("child{index}").as_str())
                .unwrap();
        }
    }
    if matches!(action, HeaderReclamation::DeleteDenseGroup) {
        assert_eq!(
            file.group("group").unwrap().info().unwrap().storage_type,
            hdf5::GroupStorageType::Dense
        );
    }
    if matches!(action, HeaderReclamation::UnlinkDataset) {
        file.link_hard("target", "alias").unwrap();
    }
    file.new_dataset_builder()
        .with_data(&[7i32; 64])
        .create("keep")
        .unwrap();
    file.new_dataset_builder()
        .with_data(&vec![31i32; usize::try_from(page / 4 + 1).unwrap()])
        .create("anchor")
        .unwrap();
    file.link_soft("/keep", "soft").unwrap();
    let survivor = file.create_group("survivor").unwrap();
    survivor
        .new_attr::<i32>()
        .create("attribute")
        .unwrap()
        .write_scalar(&19)
        .unwrap();
    survivor
        .new_dataset_builder()
        .with_data(&[29i32; 4])
        .create("nested")
        .unwrap();
    drop(survivor);
    drop(owner);
    file.close().unwrap();
}

fn assert_header_reclamation_state(
    file: &File,
    partial: &Range<u64>,
    generic_fragment: &Range<u64>,
    action: HeaderReclamation,
) {
    let free = file.reusable_free_sections().unwrap();
    assert_rust_header_survivors(file, action, 7);
    assert!(
        free.sections()
            .iter()
            .all(|section| !range::overlaps(&(section.address()..section.end_address()), partial)),
        "{free:?}"
    );
    assert_eq!(
        file.dataset("anchor").unwrap().read_i32().unwrap(),
        vec![
            31;
            usize::try_from(file.file_space_info().unwrap().unwrap().page_size.get() / 4 + 1)
                .unwrap()
        ]
    );
    if !matches!(action, HeaderReclamation::UnlinkDataset) {
        assert_eq!(
            test_util_hdf5::covered_free_bytes!(
                &free,
                generic_fragment,
                FreeSectionCoordinates::Absolute
            ),
            0
        );
        assert_eq!(
            test_util_hdf5::covered_free_bytes!(&free, partial, FreeSectionCoordinates::Absolute),
            0
        );
    }
}

fn assert_rust_header_survivors(file: &File, action: HeaderReclamation, keep: i32) {
    let page = file.file_space_info().unwrap().unwrap().page_size.get();
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        [keep; 64]
    );
    assert_eq!(
        file.dataset("survivor/nested").unwrap().read_i32().unwrap(),
        [29; 4]
    );
    assert_eq!(
        file.group("survivor").unwrap().attrs().unwrap()["attribute"],
        AttrValue::I32(19)
    );
    assert_eq!(
        file.dataset("anchor").unwrap().read_i32().unwrap(),
        vec![31; usize::try_from(page / 4 + 1).unwrap()]
    );
    let attribute = AttrValue::I64Array(vec![7; usize::try_from((page / 4).max(512)).unwrap()]);
    match action {
        HeaderReclamation::UnlinkDataset => {
            let dataset = file.dataset("alias").unwrap();
            assert_eq!(dataset.read_u8().unwrap(), [3; HEADER_TARGET_ELEMENTS]);
            assert_eq!(dataset.attrs().unwrap()["big_attr"], attribute);
        }
        HeaderReclamation::ReplaceDataset | HeaderReclamation::EditDatasetAttribute => {
            let dataset = file.dataset("target").unwrap();
            let value = if matches!(action, HeaderReclamation::ReplaceDataset) {
                17
            } else {
                3
            };
            assert_eq!(dataset.read_u8().unwrap(), [value; HEADER_TARGET_ELEMENTS]);
            assert_eq!(dataset.attrs().unwrap()["big_attr"], attribute);
            if matches!(action, HeaderReclamation::EditDatasetAttribute) {
                assert_eq!(dataset.attrs().unwrap()["changed"], AttrValue::I32(17));
            }
        }
        HeaderReclamation::RebuildGroup => {
            assert_eq!(
                file.group("group").unwrap().attrs().unwrap()["big_attr"],
                attribute
            );
            assert_eq!(
                file.group("group").unwrap().attrs().unwrap()["changed"],
                AttrValue::I32(17)
            );
            assert_eq!(
                file.dataset("group/child0").unwrap().read_i32().unwrap(),
                [0; 4]
            );
        }
        HeaderReclamation::DeleteDataset
        | HeaderReclamation::DeleteDenseGroup
        | HeaderReclamation::DeleteGroup => {}
    }
}

fn assert_c_header_survivors(file: &hdf5::File, action: HeaderReclamation) {
    assert_eq!(
        file.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
        [7; 64]
    );
    assert_eq!(
        file.dataset("soft").unwrap().read_raw::<i32>().unwrap(),
        [7; 64]
    );
    assert_eq!(
        file.dataset("survivor/nested")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [29; 4]
    );
    assert_eq!(
        file.group("survivor")
            .unwrap()
            .attr("attribute")
            .unwrap()
            .read_scalar::<i32>()
            .unwrap(),
        19
    );
    assert_eq!(
        file.dataset("anchor").unwrap().read_raw::<i32>().unwrap(),
        vec![31; usize::try_from(file.fcpl().unwrap().file_space_page_size() / 4 + 1).unwrap()]
    );
    let owner = match action {
        HeaderReclamation::UnlinkDataset => {
            Some(file.dataset("alias").unwrap().as_location().unwrap())
        }
        HeaderReclamation::ReplaceDataset | HeaderReclamation::EditDatasetAttribute => {
            Some(file.dataset("target").unwrap().as_location().unwrap())
        }
        HeaderReclamation::RebuildGroup => {
            Some(file.group("group").unwrap().as_location().unwrap())
        }
        HeaderReclamation::DeleteDataset
        | HeaderReclamation::DeleteDenseGroup
        | HeaderReclamation::DeleteGroup => None,
    };
    if let Some(owner) = owner {
        assert_eq!(
            owner.attr("big_attr").unwrap().read_raw::<i64>().unwrap(),
            vec![
                7;
                usize::try_from((file.fcpl().unwrap().file_space_page_size() / 4).max(512))
                    .unwrap()
            ]
        );
    }
    match action {
        HeaderReclamation::DeleteDataset => {
            absence::assert_libhdf5_absent(&file.dataset("target").unwrap_err(), "target")
        }
        HeaderReclamation::DeleteGroup | HeaderReclamation::DeleteDenseGroup => {
            absence::assert_libhdf5_absent(&file.group("group").unwrap_err(), "group")
        }
        HeaderReclamation::UnlinkDataset => assert_eq!(
            file.dataset("alias").unwrap().read_raw::<u8>().unwrap(),
            [3; HEADER_TARGET_ELEMENTS]
        ),
        HeaderReclamation::ReplaceDataset => assert_eq!(
            file.dataset("target").unwrap().read_raw::<u8>().unwrap(),
            [17; HEADER_TARGET_ELEMENTS]
        ),
        HeaderReclamation::EditDatasetAttribute => assert_eq!(
            file.dataset("target")
                .unwrap()
                .attr("changed")
                .unwrap()
                .read_scalar::<i32>()
                .unwrap(),
            17
        ),
        HeaderReclamation::RebuildGroup => {
            assert_eq!(
                file.group("group")
                    .unwrap()
                    .attr("changed")
                    .unwrap()
                    .read_scalar::<i32>()
                    .unwrap(),
                17
            );
            assert_eq!(
                file.dataset("group/child0")
                    .unwrap()
                    .read_raw::<i32>()
                    .unwrap(),
                [0; 4]
            );
        }
    }
}

const HEADER_TARGET_ELEMENTS: usize = 256;

#[rstest]
#[case::plain(4_096, 0)]
#[case::userblock(4_096, 4_096)]
#[case::larger_pages(8_192, 8_192)]
fn nonpersistent_page_staged_edits_alternate_with_c_edits(
    #[case] page_size: u64,
    #[case] userblock: u64,
    #[values(false, true)] c_origin: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nonpersistent_page.h5");
    write_nonpersistent_page_origin(&path, page_size, userblock, c_origin);
    let original = std::fs::read(&path).unwrap();
    add_pure_page_dataset(&path);
    edit_page_with_libhdf5(&path);

    let before = std::fs::read(&path).unwrap();
    let file = File::open(&path).unwrap();
    let old_eoa = file.superblock().eof_address;
    let old_raw = file_space::raw_extents(&file, "keep");
    assert_eq!(file.dataset("from_c").unwrap().read_u8().unwrap(), [13; 64]);
    assert_eq!(file.dataset("alias").unwrap().read_u8().unwrap(), [7; 64]);
    drop(file);
    replace_and_delete_page_datasets(&path);
    assert_pure_page_edit(
        &path, page_size, userblock, &original, &before, old_eoa, &old_raw,
    );
    assert_c_page_edit(&path, page_size);
}

/// Writes matching datasets and a group attribute through either writer.
fn write_nonpersistent_page_origin(path: &Path, page_size: u64, userblock: u64, c_origin: bool) {
    if c_origin {
        page_publication::write_origin(path, page_size, userblock);
    } else {
        let mut builder = FileBuilder::new();
        builder
            .with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::Page)
                    .with_page_size(FileSpacePageSize::try_from(page_size).unwrap()),
            )
            .with_userblock(UserblockSize::try_from(userblock).unwrap());
        for (name, value) in [("keep", 7), ("replace", 3), ("remove", 5)] {
            builder.create_dataset(name).with_u8_data(&[value; 64]);
        }
        let mut group = builder.create_group("group");
        group.set_attr("attribute", AttrValue::I32(9));
        builder.add_group(group.finish());
        builder.write(path).unwrap();
    }
    if userblock != 0 {
        let mut bytes = std::fs::read(path).unwrap();
        bytes[..userblock as usize].fill(0xA5);
        std::fs::write(path, bytes).unwrap();
    }
}

/// Commits `added` with hdf5-pure for the libhdf5 editing session to read.
fn add_pure_page_dataset(path: &Path) {
    let file = File::open_rw(path).unwrap();
    file.root()
        .create_dataset("added", |dataset| {
            dataset.with_u8_data(&[9; 64]);
        })
        .unwrap();
    file.commit().unwrap();
    drop(file);
}

/// Replaces `added` with `from_c` and creates hard and soft links through libhdf5.
fn edit_page_with_libhdf5(path: &Path) {
    let c_file = hdf5::File::open_rw(path).unwrap();
    assert_eq!(
        c_file.dataset("added").unwrap().read_raw::<u8>().unwrap(),
        [9; 64]
    );
    assert_eq!(
        c_file.dataset("keep").unwrap().read_raw::<u8>().unwrap(),
        [7; 64]
    );
    c_file.unlink("added").unwrap();
    c_file
        .new_dataset::<u8>()
        .shape(64)
        .create("from_c")
        .unwrap()
        .write_raw(&[13; 64])
        .unwrap();
    c_file.link_hard("keep", "alias").unwrap();
    c_file.link_soft("/keep", "soft").unwrap();
    c_file.close().unwrap();
}

/// Commits dataset replacement, deletion, and creation with hdf5-pure.
fn replace_and_delete_page_datasets(path: &Path) {
    let file = File::open_rw(path).unwrap();
    file.root().delete("remove").unwrap();
    file.dataset("replace")
        .unwrap()
        .write_staged(|dataset| {
            dataset.with_u8_data(&[17; 64]);
        })
        .unwrap();
    file.root()
        .create_dataset("second", |dataset| {
            dataset.with_u8_data(&[19; 64]);
        })
        .unwrap();
    file.commit().unwrap();
    drop(file);
}

/// Checks appended placement, preserved bytes, and file-space settings after a staged commit.
fn assert_pure_page_edit(
    path: &Path,
    page_size: u64,
    userblock: u64,
    original: &[u8],
    before: &[u8],
    old_eoa: u64,
    old_raw: &[Range<u64>],
) {
    let after = std::fs::read(path).unwrap();
    let file = File::open(path).unwrap();
    assert_eq!(file.superblock().base_address.get(), userblock);
    assert_eq!((file.superblock().eof_address - userblock) % page_size, 0);
    let info = file.file_space_info().unwrap().unwrap();
    assert_eq!(info.strategy, FileSpaceStrategy::Page);
    assert_eq!(info.page_size.get(), page_size);
    assert!(!info.persist);
    assert!(info.manager_addrs.is_empty());
    assert_eq!(file.persisted_free_sections().unwrap(), None);
    assert_eq!(file.dataset("keep").unwrap().read_u8().unwrap(), [7; 64]);
    assert_eq!(file.dataset("alias").unwrap().read_u8().unwrap(), [7; 64]);
    assert_eq!(file.dataset("from_c").unwrap().read_u8().unwrap(), [13; 64]);
    assert_eq!(
        file.dataset("replace").unwrap().read_u8().unwrap(),
        [17; 64]
    );
    assert_eq!(file.dataset("second").unwrap().read_u8().unwrap(), [19; 64]);
    assert_eq!(
        file.group("group")
            .unwrap()
            .attrs()
            .unwrap()
            .get("attribute"),
        Some(&AttrValue::I32(9))
    );
    assert_eq!(file_space::raw_extents(&file, "keep"), old_raw);
    let boundary = (old_eoa - userblock).next_multiple_of(page_size) + userblock;
    assert!(
        file_space::raw_extents(&file, "replace")
            .iter()
            .chain(&file_space::raw_extents(&file, "second"))
            .all(|extent| extent.start >= boundary)
    );
    let sb_len = hdf5_pure_format::__private::serialize_superblock(file.superblock())
        .unwrap()
        .len();
    drop(file);
    assert_eq!(
        &after[..userblock as usize],
        &original[..userblock as usize]
    );
    assert_eq!(
        &after[userblock as usize + sb_len..before.len()],
        &before[userblock as usize + sb_len..]
    );
    assert!(
        !after
            .windows(4)
            .any(|window| window == free_space::SIGNATURE
                || window == free_space::SECTIONS_SIGNATURE)
    );
    paged::assert_pages_homogeneous(path, page_size, &["replace", "second"]);
}

/// Checks datasets, links, attributes, and file-space settings through libhdf5.
fn assert_c_page_edit(path: &Path, page_size: u64) {
    let c_file = hdf5::File::open(path).unwrap();
    assert!(!c_file.link_exists("remove"));
    assert!(!c_file.link_exists("added"));
    assert_eq!(
        c_file.dataset("keep").unwrap().read_raw::<u8>().unwrap(),
        [7; 64]
    );
    assert_eq!(
        c_file.dataset("alias").unwrap().read_raw::<u8>().unwrap(),
        [7; 64]
    );
    assert_eq!(
        c_file.dataset("soft").unwrap().read_raw::<u8>().unwrap(),
        [7; 64]
    );
    assert_eq!(
        c_file.dataset("from_c").unwrap().read_raw::<u8>().unwrap(),
        [13; 64]
    );
    assert_eq!(
        c_file.dataset("replace").unwrap().read_raw::<u8>().unwrap(),
        [17; 64]
    );
    assert_eq!(
        c_file.dataset("second").unwrap().read_raw::<u8>().unwrap(),
        [19; 64]
    );
    assert_eq!(
        c_file
            .group("group")
            .unwrap()
            .attr("attribute")
            .unwrap()
            .read_scalar::<i32>()
            .unwrap(),
        9
    );
    assert_eq!(
        c_file.fcpl().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: false,
            threshold: 1
        }
    );
    assert_eq!(
        c_file.fcpl().unwrap().get_file_space_page_size().unwrap(),
        page_size
    );
}

#[rstest]
#[case::fsm_aggr(FileSpaceStrategy::FsmAggr, false, 0)]
#[case::persistent_fsm_aggr(FileSpaceStrategy::FsmAggr, true, 128)]
#[case::page(FileSpaceStrategy::Page, false, 0)]
#[case::persistent_page(FileSpaceStrategy::Page, true, 128)]
#[case::aggr(FileSpaceStrategy::Aggr, true, 128)]
#[case::none(FileSpaceStrategy::None, true, 128)]
fn recorded_settings_are_equivalent_for_rust_and_libhdf5_files(
    #[case] strategy: FileSpaceStrategy,
    #[case] persist: bool,
    #[case] threshold: u64,
    #[values(false, true)] c_origin: bool,
    #[values(false, true)] streaming: bool,
    #[values(false, true)] from_properties: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("recorded-settings.h5");
    let page_size = FileSpacePageSize::try_from(8192).unwrap();
    let expected = FileSpaceSettings::new(strategy)
        .with_persistence(persist)
        .with_threshold(threshold)
        .with_page_size(page_size);
    if c_origin {
        let c_strategy = match strategy {
            FileSpaceStrategy::FsmAggr | FileSpaceStrategy::Page => CStrategy::FreeSpaceManager {
                paged: strategy == FileSpaceStrategy::Page,
                persist,
                threshold,
            },
            FileSpaceStrategy::Aggr => CStrategy::PageAggregation,
            FileSpaceStrategy::None => CStrategy::None,
        };
        let file = hdf5::FileBuilder::new()
            .with_fapl(|properties| properties.libver_v110())
            .with_fcpl(|properties| {
                properties
                    .file_space_strategy(c_strategy)
                    .file_space_page_size(page_size.get())
                    .userblock(8192)
            })
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape(3)
            .create("data")
            .unwrap()
            .write(&[1, 2, 3])
            .unwrap();
        file.close().unwrap();
    } else {
        let mut builder = FileBuilder::new();
        if from_properties {
            builder.with_create_properties(
                FileCreateProperties::new().with_file_space_settings(expected),
            );
        } else {
            builder.with_file_space_settings(expected);
        }
        builder.with_userblock(UserblockSize::try_from(8192).unwrap());
        builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
        builder.write(&path).unwrap();
        let file = hdf5::File::open(&path).unwrap();
        let properties = file.create_plist().unwrap();
        let c_strategy = match strategy {
            FileSpaceStrategy::FsmAggr | FileSpaceStrategy::Page => CStrategy::FreeSpaceManager {
                paged: strategy == FileSpaceStrategy::Page,
                persist: expected.persist(),
                threshold: expected.threshold(),
            },
            FileSpaceStrategy::Aggr => CStrategy::PageAggregation,
            FileSpaceStrategy::None => CStrategy::None,
        };
        assert_eq!(properties.get_file_space_strategy().unwrap(), c_strategy);
        assert_eq!(
            properties.get_file_space_page_size().unwrap(),
            page_size.get()
        );
        assert_eq!(
            file.dataset("data").unwrap().read_raw::<i32>().unwrap(),
            [1, 2, 3]
        );
        file.close().unwrap();
    }
    let file = if streaming {
        File::open_streaming(&path).unwrap()
    } else {
        File::open(&path).unwrap()
    };
    assert_eq!(file.file_space_settings().unwrap(), Some(expected));
    let info = file.file_space_info().unwrap().unwrap();
    assert_eq!(
        (info.strategy, info.persist, info.threshold, info.page_size),
        (
            expected.strategy(),
            expected.persist(),
            expected.threshold(),
            expected.page_size()
        )
    );
    assert_eq!(file.dataset("data").unwrap().read_i32().unwrap(), [1, 2, 3]);
}

#[test]
fn c_library_reads_our_strategy() {
    let dir = tempfile::tempdir().unwrap();
    for (i, (ours, expected)) in [
        (
            FileSpaceStrategy::FsmAggr,
            CStrategy::FreeSpaceManager {
                paged: false,
                persist: false,
                threshold: 1,
            },
        ),
        (
            FileSpaceStrategy::Page,
            CStrategy::FreeSpaceManager {
                paged: true,
                persist: false,
                threshold: 1,
            },
        ),
        // libhdf5 maps the AGGR strategy onto its `PageAggregation` name.
        (FileSpaceStrategy::Aggr, CStrategy::PageAggregation),
        (FileSpaceStrategy::None, CStrategy::None),
    ]
    .into_iter()
    .enumerate()
    {
        let path = dir.path().join(format!("ours_{i}.h5"));
        let mut b = FileBuilder::new();
        b.create_dataset("d").with_i32_data(&[10, 20, 30]);
        b.with_file_space_settings(
            FileSpaceSettings::new(ours).with_page_size(FileSpacePageSize::DEFAULT),
        );
        b.write(&path).unwrap();

        // libhdf5 opens our file, reads the data, and recovers the strategy
        // from the superblock-extension File Space Info message.
        let f = hdf5::File::open(&path).unwrap();
        assert_eq!(
            f.dataset("d").unwrap().read_raw::<i32>().unwrap(),
            vec![10, 20, 30]
        );
        let strategy = f.create_plist().unwrap().get_file_space_strategy().unwrap();
        assert_eq!(
            strategy, expected,
            "strategy {ours:?} read back by the C library"
        );
    }
}

#[rstest]
#[case::fsm_aggr_at_the_minimum(FileSpaceStrategy::FsmAggr, FileSpacePageSize::MIN)]
#[case::fsm_aggr_at_the_maximum(FileSpaceStrategy::FsmAggr, FileSpacePageSize::MAX)]
#[case::fsm_aggr_at_an_irregular_size(FileSpaceStrategy::FsmAggr, FileSpacePageSize::try_from(3000).unwrap())]
#[case::page_at_an_irregular_size(FileSpaceStrategy::Page, FileSpacePageSize::try_from(3000).unwrap())]
#[case::page_at_the_minimum(FileSpaceStrategy::Page, FileSpacePageSize::MIN)]
#[case::aggr_at_the_maximum(FileSpaceStrategy::Aggr, FileSpacePageSize::MAX)]
#[case::none_at_the_maximum(FileSpaceStrategy::None, FileSpacePageSize::MAX)]
fn libhdf5_reads_the_requested_page_size(
    #[case] strategy: FileSpaceStrategy,
    #[case] page_size: FileSpacePageSize,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bound.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[10, 20, 30]);
    b.with_file_space_settings(FileSpaceSettings::new(strategy).with_page_size(page_size));
    b.write(&path).unwrap();

    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        (
            f.dataset("d").unwrap().read_raw::<i32>().unwrap(),
            f.create_plist()
                .unwrap()
                .get_file_space_page_size()
                .unwrap(),
        ),
        (vec![10, 20, 30], page_size.get())
    );
}

#[rstest]
#[case::fsm_aggr_at_the_minimum(false, FileSpacePageSize::MIN)]
#[case::fsm_aggr_at_the_maximum(false, FileSpacePageSize::MAX)]
#[case::fsm_aggr_at_an_irregular_size(false, FileSpacePageSize::try_from(3000).unwrap())]
#[case::page_at_an_irregular_size(true, FileSpacePageSize::try_from(3000).unwrap())]
#[case::page_at_the_minimum(true, FileSpacePageSize::MIN)]
fn reads_the_page_size_libhdf5_writes(#[case] paged: bool, #[case] page_size: FileSpacePageSize) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bound.h5");
    hdf5::FileBuilder::new()
        .with_fapl(|fapl| fapl.libver_v110())
        .with_fcpl(|fcpl| {
            fcpl.file_space_strategy(CStrategy::FreeSpaceManager {
                paged,
                persist: false,
                threshold: 1,
            })
            .file_space_page_size(page_size.get())
        })
        .create(&path)
        .unwrap()
        .close()
        .unwrap();

    assert_eq!(
        File::open(&path)
            .unwrap()
            .file_space_info()
            .unwrap()
            .map(|info| info.page_size),
        Some(page_size)
    );
}

#[test]
fn we_read_c_library_persisted_free_space() {
    // libhdf5 writes a persisted FSM file with real free space (a deleted
    // dataset). hdf5-pure must follow the File Space Info manager addresses to the
    // on-disk FSHD/FSSE blocks and recover the freed sections.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_persisted_fsm.h5");
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(|fcpl| {
                fcpl.file_space_strategy(CStrategy::FreeSpaceManager {
                    paged: false,
                    persist: true,
                    threshold: 1,
                })
            })
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape((100,))
            .create("a")
            .unwrap()
            .write(&vec![1i32; 100])
            .unwrap();
        file.new_dataset::<i32>()
            .shape((400,))
            .create("b")
            .unwrap()
            .write(&vec![2i32; 400])
            .unwrap();
        file.new_dataset::<i32>()
            .shape((100,))
            .create("c")
            .unwrap()
            .write(&vec![3i32; 100])
            .unwrap();
        file.unlink("b").unwrap(); // frees b's storage into the persisted FSM
        file.close().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_eq!(
        f.file_space_settings()
            .unwrap()
            .map(|settings| settings.strategy()),
        Some(FileSpaceStrategy::FsmAggr)
    );
    assert!(f.file_space_info().unwrap().unwrap().persist);

    let free = f.persisted_free_sections().unwrap().unwrap();
    assert!(
        free.free_sections().section_count() > 0,
        "expected persisted free sections from the deleted dataset"
    );
    // The freed regions are non-overlapping and account for real space (b held
    // 400 * 4 = 1600 bytes of raw data, freed as at least one large section).
    let total = free.free_sections().free_bytes();
    assert!(
        total >= 1600,
        "freed space {total} should cover the dataset"
    );
    assert!(
        free.free_sections()
            .sections()
            .iter()
            .any(|section| section.length().get() >= 1600)
    );

    // Surviving datasets still read correctly alongside the persisted managers.
    assert_eq!(f.dataset("a").unwrap().read_i32().unwrap(), vec![1; 100]);
    assert_eq!(f.dataset("c").unwrap().read_i32().unwrap(), vec![3; 100]);
}

#[test]
fn c_library_reads_our_persisted_free_space() {
    // The mirror of `we_read_c_library_persisted_free_space`: hdf5-pure writes a
    // persisted file with real free space (a deleted dataset), and the reference
    // C library opens it, recovers the strategy, reads the survivors, and loads
    // the on-disk free-space managers we wrote.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ours_persisted.h5");

    // Create a persisted file, then free the interior dataset's storage in place.
    file_space::write_persisted_interior_hole(&path);

    // hdf5-pure's own reader recovers the persisted sections (covering "big").
    let ours = File::open(&path).unwrap();
    let total_ours: u64 = ours
        .persisted_free_sections()
        .unwrap()
        .unwrap()
        .free_sections()
        .free_bytes();
    assert!(
        total_ours >= 1600,
        "we persist the freed storage: {total_ours}"
    );
    drop(ours);

    // libhdf5 opens the same file: strategy and persist flag round-trip,
    // the survivors read byte-exact, and `H5Fget_freespace` parses our managers.
    let f = hdf5::File::open(&path).unwrap();
    let strategy = f.create_plist().unwrap().get_file_space_strategy().unwrap();
    assert_eq!(
        strategy,
        CStrategy::FreeSpaceManager {
            paged: false,
            persist: true,
            threshold: 1,
        },
        "C library recovers our persisted FSM strategy"
    );
    assert_eq!(
        f.dataset("a").unwrap().read_raw::<i32>().unwrap(),
        vec![1; 100]
    );
    assert_eq!(
        f.dataset("c").unwrap().read_raw::<i32>().unwrap(),
        vec![3; 100]
    );
    absence::assert_libhdf5_absent(&f.dataset("big").unwrap_err(), "big");

    // Loading the managers requires parsing our FSHD/FSSE blocks; the C library
    // reports at least the freed dataset's storage as free space.
    let free_c = f.free_space() as i64;
    assert!(
        free_c >= 1600,
        "C library loads our free-space managers and reports the freed space (got {free_c})"
    );
}

#[test]
fn c_library_accepts_reclaimed_v2_btree_chunk_storage_and_allocates_afterward() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v2_btree_reclaim.h5");
    let raw_values = (0..40 * 40).collect::<Vec<i32>>();
    let filtered_values = vec![7i32; 40 * 40];
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(|fcpl| {
                fcpl.file_space_strategy(CStrategy::FreeSpaceManager {
                    paged: false,
                    persist: true,
                    threshold: 1,
                })
            })
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape((hdf5::Extent::resizable(40), hdf5::Extent::resizable(40)))
            .chunk((4, 4))
            .create("raw")
            .unwrap()
            .write_raw(&raw_values)
            .unwrap();
        file.new_dataset::<i32>()
            .shape((hdf5::Extent::resizable(40), hdf5::Extent::resizable(40)))
            .chunk((4, 4))
            .deflate(6)
            .create("filtered")
            .unwrap()
            .write_raw(&filtered_values)
            .unwrap();
        file.new_dataset::<i32>()
            .shape((4,))
            .create("keep")
            .unwrap()
            .write(&[1, 2, 3, 4])
            .unwrap();
        file.close().unwrap();
    }

    let (raw_bytes, filtered_bytes) = {
        let before = File::open(&path).unwrap();
        let raw = before.dataset("raw").unwrap();
        let filtered = before.dataset("filtered").unwrap();
        assert_eq!(raw.chunk_index().unwrap(), Some(ChunkIndex::BTreeV2));
        assert_eq!(filtered.chunk_index().unwrap(), Some(ChunkIndex::BTreeV2));
        let raw_bytes: u64 = raw
            .chunks()
            .unwrap()
            .iter()
            .map(|chunk| chunk.storage_size)
            .sum();
        let filtered_bytes: u64 = filtered
            .chunks()
            .unwrap()
            .iter()
            .map(|chunk| chunk.storage_size)
            .sum();
        (raw_bytes, filtered_bytes)
    };
    assert!(filtered_bytes < raw_bytes);

    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("raw").unwrap();
        session.root().delete("filtered").unwrap();
        session.commit().unwrap();
    }

    let ours = File::open(&path).unwrap();
    for name in ["raw", "filtered"] {
        let err = ours.dataset(name).unwrap_err();
        let Error::Format(FormatError::PathNotFound(missing)) = &err else {
            panic!("expected PathNotFound for {name}, got {err:?}");
        };
        assert_eq!(missing, name);
    }
    assert_eq!(
        ours.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3, 4]
    );
    let free_after_delete = ours.persisted_free_sections().unwrap().unwrap();
    let free_ours = free_after_delete.free_sections().free_bytes();
    assert!(
        free_ours >= raw_bytes + filtered_bytes,
        "persisted free space {free_ours} must include every raw v2 B-tree chunk"
    );
    drop(ours);

    let c = hdf5::File::open(&path).unwrap();
    absence::assert_libhdf5_absent(&c.dataset("raw").unwrap_err(), "raw");
    absence::assert_libhdf5_absent(&c.dataset("filtered").unwrap_err(), "filtered");
    assert_eq!(
        c.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
        vec![1, 2, 3, 4]
    );
    assert!(
        c.free_space() as u64 >= raw_bytes + filtered_bytes,
        "the C library must accept the persisted managers containing the reclaimed chunks"
    );
    drop(c);

    // libhdf5 allocates after it has loaded hdf5-pure's persisted managers.
    // The allocator may choose a reusable section or newly extended file space. This
    // crosscheck verifies that both libraries accept the resulting file.
    let reused = vec![11i32; 40 * 40];
    {
        let c = hdf5::File::open_rw(&path).unwrap();
        c.new_dataset::<i32>()
            .shape((hdf5::Extent::resizable(40), hdf5::Extent::resizable(40)))
            .chunk((4, 4))
            .create("reused")
            .unwrap()
            .write_raw(&reused)
            .unwrap();
        c.close().unwrap();
    }

    let after_reuse = File::open(&path).unwrap();
    let reused_dataset = after_reuse.dataset("reused").unwrap();
    assert_eq!(reused_dataset.chunks().unwrap().len(), 100);
    assert_eq!(reused_dataset.read_i32().unwrap(), reused);
}

#[test]
fn c_library_reads_managers_we_placed_mid_file() {
    // A file's first persisted free-space managers have no earlier free region to occupy,
    // which places them at end-of-file. Once a file has free space, later commits can place
    // managers in it (issue #358). libhdf5 then has to find the extension and managers
    // below live data and accept an `eoa_fsm_fsalloc` that spans the whole file. Several commits
    // ensure the final managers moved into space an earlier set vacated.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ours_persisted_mid_file.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[1; 100]);
    // Freed below, and large enough that every tail these commits write fits in
    // what it leaves however the free list fragments.
    b.create_dataset("big").with_i32_data(&[7; 4000]);
    b.create_dataset("c").with_i32_data(&[3; 100]);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    b.write(&path).unwrap();
    let mut sizes = Vec::new();
    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("big").unwrap();
        s.commit().unwrap();
        for i in 0..4 {
            s.root().set_attr("n", AttrValue::I64(i)).unwrap();
            s.commit().unwrap();
            sizes.push(std::fs::metadata(&path).unwrap().len());
        }
        // A dataset larger than every hole, so it is written past everything and
        // the managers this commit rewrites — drawn from holes below it — end up
        // with live data above them whichever hole they land in.
        s.root()
            .create_dataset("above", |b| {
                b.with_i32_data(&vec![5; 8000]);
            })
            .unwrap();
        s.commit().unwrap();
    }
    assert_eq!(
        sizes[1],
        *sizes.last().unwrap(),
        "each rewrite of the managers should land in space an earlier one left, \
         leaving the file the size it was (sizes {sizes:?})"
    );

    let ours = File::open(&path).unwrap();
    let managers: Vec<u64> = ours
        .file_space_info()
        .unwrap()
        .expect("a persisting file records its managers")
        .manager_addrs
        .iter()
        .filter_map(|&a| match a {
            hdf5_pure::DecodedStoredAddress::Defined(stored) => Some(stored.get()),
            hdf5_pure::DecodedStoredAddress::Undefined => None,
        })
        .collect();
    assert!(!managers.is_empty(), "the file has free space to record");
    let Layout::Contiguous {
        address: Some(above),
        ..
    } = ours.dataset("above").unwrap().layout().unwrap()
    else {
        panic!("`above` is a contiguous dataset with allocated storage");
    };
    for m in &managers {
        assert!(
            *m < above,
            "a manager at {m} should sit in a hole below the data appended after \
             it, at {above}"
        );
    }
    let total_ours: u64 = ours
        .persisted_free_sections()
        .unwrap()
        .unwrap()
        .free_sections()
        .free_bytes();
    drop(ours);

    // libhdf5 reads the same file: survivors byte-exact, and its own
    // free-space total — which it can only get by parsing the managers we placed
    // mid-file — agrees with ours to the byte.
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.dataset("a").unwrap().read_raw::<i32>().unwrap(),
        vec![1; 100]
    );
    assert_eq!(
        f.dataset("c").unwrap().read_raw::<i32>().unwrap(),
        vec![3; 100]
    );
    let free_c = f.free_space() as i64;
    assert_eq!(
        free_c as u64, total_ours,
        "the C library's free-space total must match the managers we wrote"
    );
    drop(f);

    // And *writes* to it. This is the path that consumes `eoa_fsm_fsalloc`: a
    // read-write open loads the free-space managers, frees the blocks holding
    // them, and takes the file's end-of-allocation back to that recorded value —
    // which a reused tail records as the whole file rather than as its own start.
    // Reading proves the C library parses what we wrote; only this proves it acts
    // on it without losing the file.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        f.new_dataset::<i32>()
            .shape((30,))
            .create("added")
            .unwrap()
            .write(&(0..30).collect::<Vec<i32>>())
            .unwrap();
        f.close().unwrap();
    }
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.dataset("c").unwrap().read_raw::<i32>().unwrap(),
        vec![3; 100]
    );
    assert_eq!(
        f.dataset("added").unwrap().read_raw::<i32>().unwrap(),
        (0..30).collect::<Vec<i32>>()
    );
    drop(f);
    // Our own reader survives the C library's rewrite of the same file.
    let ours = File::open(&path).unwrap();
    assert_eq!(ours.dataset("c").unwrap().read_i32().unwrap(), vec![3; 100]);
    assert_eq!(
        ours.dataset("above").unwrap().read_i32().unwrap(),
        vec![5; 8000]
    );
}

#[test]
fn libhdf5_reads_and_edits_a_reused_manager_with_spare_allocation() {
    const FILLER: usize = 7;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reused_manager_spare_allocation.h5");
    let seed = (0..100 + FILLER as i32).collect::<Vec<i32>>();
    let mut builder = FileBuilder::new();
    builder.create_dataset("seed").with_i32_data(&seed);
    for (index, len) in [256usize, 37, 91].into_iter().enumerate() {
        builder
            .create_dataset(&format!("scratch{index}"))
            .with_i32_data(&vec![7; len + FILLER]);
        builder
            .create_dataset(&format!("keep{index}"))
            .with_i32_data(&[9; 4]);
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

    let mut spare = 0;
    for generation in 0..6 {
        let file = open();
        file.root()
            .set_attr("n", AttrValue::I64(generation))
            .unwrap();
        file.commit().unwrap();
        file.close().unwrap();
        spare = file_space::serialized_manager_padding(&path);
        if spare > 0 {
            break;
        }
    }
    assert!(spare > 0, "the fixture must write an oversized allocation");
    let reader = File::open(&path).unwrap();
    let persisted_free = reader
        .persisted_free_sections()
        .unwrap()
        .unwrap()
        .free_sections()
        .free_bytes();
    drop(reader);

    let native = hdf5::File::open(&path).unwrap();
    assert_eq!(native.free_space(), persisted_free);
    assert_eq!(
        native.dataset("seed").unwrap().read_raw::<i32>().unwrap(),
        seed
    );
    for index in 0..3 {
        assert_eq!(
            native
                .dataset(&format!("keep{index}"))
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            vec![9; 4]
        );
        absence::assert_libhdf5_absent(
            &native.dataset(&format!("scratch{index}")).unwrap_err(),
            &format!("scratch{index}"),
        );
    }
    native.close().unwrap();

    let added = (0..30).collect::<Vec<i32>>();
    let native = hdf5::File::open_rw(&path).unwrap();
    native
        .new_dataset::<i32>()
        .shape((30,))
        .create("added")
        .unwrap()
        .write(&added)
        .unwrap();
    native.close().unwrap();

    let reader = File::open(&path).unwrap();
    let native = hdf5::File::open(&path).unwrap();
    assert_eq!(reader.dataset("seed").unwrap().read_i32().unwrap(), seed);
    assert_eq!(
        native.dataset("seed").unwrap().read_raw::<i32>().unwrap(),
        seed
    );
    assert_eq!(reader.dataset("added").unwrap().read_i32().unwrap(), added);
    assert_eq!(
        native.dataset("added").unwrap().read_raw::<i32>().unwrap(),
        added
    );
    for index in 0..3 {
        assert_eq!(
            reader
                .dataset(&format!("keep{index}"))
                .unwrap()
                .read_i32()
                .unwrap(),
            vec![9; 4]
        );
        assert_eq!(
            native
                .dataset(&format!("keep{index}"))
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            vec![9; 4]
        );
    }
    assert_eq!(
        native.free_space(),
        reader
            .persisted_free_sections()
            .unwrap()
            .unwrap()
            .free_sections()
            .free_bytes()
    );
}

#[rstest]
fn the_serializer_writes_the_message_libhdf5_writes_at_the_widths_of_its_file(
    #[values(Sizeof::Bytes2, Sizeof::Bytes4, Sizeof::Bytes8)] sizeof_addr: Sizeof,
    #[values(Sizeof::Bytes2, Sizeof::Bytes4, Sizeof::Bytes8)] sizeof_size: Sizeof,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_persisted_widths.h5");
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(|fcpl| {
                fcpl.sizes(SizeofInfo {
                    sizeof_addr,
                    sizeof_size,
                })
                .file_space_strategy(CStrategy::FreeSpaceManager {
                    paged: false,
                    persist: true,
                    threshold: 1,
                })
            })
            .create(&path)
            .unwrap();
        for name in ["a", "b"] {
            file.new_dataset::<i32>()
                .shape((400,))
                .create(name)
                .unwrap()
                .write(&vec![1i32; 400])
                .unwrap();
        }
        file.unlink("a").unwrap();
        file.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    let superblock = file.superblock();
    assert_eq!(
        usize::from(superblock.offset_size.get()),
        usize::from(sizeof_addr)
    );
    assert_eq!(
        usize::from(superblock.length_size.get()),
        usize::from(sizeof_size)
    );
    let widths = FormatWidths {
        offsets: superblock.offset_size,
        lengths: superblock.length_size,
    };
    let info = file.file_space_info().unwrap().unwrap();
    assert_eq!(
        (
            info.strategy,
            info.persist,
            info.threshold,
            info.page_size,
            info.page_end_meta_threshold,
            info.manager_addrs.len(),
        ),
        (
            FileSpaceStrategy::FsmAggr,
            true,
            1,
            FileSpacePageSize::DEFAULT,
            0,
            12
        )
    );
    assert!(
        info.manager_addrs
            .iter()
            .any(|&addr| matches!(addr, hdf5_pure::DecodedStoredAddress::Defined(_))),
        "libhdf5 records a manager for the deleted dataset's space"
    );
    let encoded = hdf5_pure_format::__private::serialize_file_space_info_v1(widths, &info).unwrap();
    // The version, strategy and persist bytes, the 2-byte page-end threshold, two lengths, the end
    // of allocation, and twelve manager addresses.
    assert_eq!(
        encoded.len(),
        5 + 2 * usize::from(superblock.length_size.get())
            + 13 * usize::from(superblock.offset_size.get())
    );

    let written = std::fs::read(&path).unwrap();
    assert!(
        written
            .windows(encoded.len())
            .any(|window| window == encoded),
        "libhdf5 writes the message the serializer produces, {encoded:02x?}"
    );
}

// TODO: Report to HDF Group
// Below 8-byte lengths, libhdf5 writes the maximum address of its file driver into the "Maximum
// Section Size" of a free-space manager header truncated to the length width
// (`H5FS__cache_hdr_serialize` in `H5FScache.c`), and writes each section size in the bytes the
// whole address takes (`H5FS__sinfo_new` in `H5FSsection.c`), HDF5 1.10.11 to 2.2.0. A section
// offset takes as many bytes, since libhdf5 sets the "Size of Address Space" to the number of bits
// of the whole address (`H5MF__create_fstype` in `H5MF.c`): 8 under the sec2 driver, and under the
// core driver, whose maximum address is `SIZE_MAX - 1`, the width of a `size_t` on the host.
// "Free-space Index" of the format specification, version 4.0, defines a section size field as the
// fewest bytes that hold the stored maximum. The truncated address is all ones under the sec2
// driver and all ones less one under the core driver. On reopen libhdf5 sizes the section size
// fields from the truncated value, and misreads its own section list where the truncated value
// takes fewer bytes than the whole address. When it reopens such a file read-write, 1.10.11 calls a
// null function pointer and 1.14.6 ends the process with SIGSEGV. At 2-byte offsets 1.14.6 loads
// the misread list without an error.
#[rstest]
#[case::default_driver(Driver::Sec2, 0, 63)]
#[case::core_driver(Driver::Core, 1, usize::BITS)]
fn libhdf5_truncates_the_maximum_section_size_and_writes_section_sizes_as_wide_as_its_address_space(
    #[case] driver: Driver,
    #[case] below_all_ones: u64,
    #[case] addr_space_bits: u32,
    #[values(Sizeof::Bytes2, Sizeof::Bytes4, Sizeof::Bytes8)] sizeof_addr: Sizeof,
    #[values(Sizeof::Bytes2, Sizeof::Bytes4)] sizeof_size: Sizeof,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_truncated_maximum.h5");
    let freed = write_c_persisted_with_a_hole(&path, driver, sizeof_addr, sizeof_size, 0);
    let file = File::open(&path).unwrap();
    let (offset_size, length_size) = (file.superblock().offset_size, file.superblock().length_size);
    let manager_addrs = file
        .file_space_info()
        .unwrap()
        .unwrap()
        .manager_addrs
        .clone();
    drop(file);
    let widths = FormatWidths {
        offsets: offset_size,
        lengths: length_size,
    };
    let bytes = std::fs::read(&path).unwrap();
    let headers: Vec<FreeSpaceManagerHeader> = manager_addrs
        .into_iter()
        .filter_map(|addr| match addr {
            hdf5_pure::DecodedStoredAddress::Defined(stored) => Some(stored),
            hdf5_pure::DecodedStoredAddress::Undefined => None,
        })
        .map(|stored| {
            FreeSpaceManagerHeader::parse(widths, &bytes[usize::try_from(stored.get()).unwrap()..])
                .unwrap()
        })
        .collect();

    let truncated = u64::MAX >> (64 - 8 * u32::from(length_size.get()));
    for header in &headers {
        assert_eq!(
            (
                header.max_section_size(),
                u32::from(header.addr_space_bits())
            ),
            (truncated - below_all_ones, addr_space_bits)
        );
    }
    let width = usize::try_from(addr_space_bits.div_ceil(8)).unwrap();
    let header = headers
        .iter()
        .find(|header| (header.total_space(), header.total_sections()) == (1600, 1))
        .expect("libhdf5 records the deleted dataset's storage in a manager of its own");
    let fsse_at = usize::try_from(header.section_list_addr().get()).unwrap();
    let fsse = &bytes[fsse_at..fsse_at + usize::try_from(header.section_list_used()).unwrap()];
    assert_eq!(
        fsse[4 + 1 + usize::from(offset_size.get())..fsse.len() - 4],
        [
            &[1][..],
            &1600u64.to_le_bytes()[..width],
            &freed.to_le_bytes()[..width],
            &[SECTION_CLASS_SIMPLE],
        ]
        .concat()
    );
}

#[rstest]
#[case::bounded(MemoryStrategy::Bounded, 0)]
#[case::mirrored(MemoryStrategy::Mirrored, 0)]
#[case::auto(MemoryStrategy::Auto, 0)]
#[case::mirrored_userblock(MemoryStrategy::Mirrored, 4096)]
#[case::auto_userblock(MemoryStrategy::Auto, 4096)]
fn strict_persisted_open_recovers_libhdf5_managers(
    #[case] memory: MemoryStrategy,
    #[case] userblock: u64,
    #[values(false, true)] paged: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("strict.h5");
    let tolerant_path = directory.path().join("tolerant.h5");
    let native = hdf5::FileBuilder::new()
        .with_fapl(|properties| properties.libver_v110())
        .with_fcpl(|properties| {
            properties
                .userblock(userblock)
                .file_space_strategy(CStrategy::FreeSpaceManager {
                    paged,
                    persist: true,
                    threshold: 1,
                })
        })
        .create(&path)
        .unwrap();
    native
        .new_dataset_builder()
        .with_data(&[1i32; 100])
        .create("a")
        .unwrap();
    native
        .new_dataset_builder()
        .with_data(&[2i32; 400])
        .create("b")
        .unwrap();
    native
        .new_dataset_builder()
        .with_data(&[3i32; 100])
        .create("c")
        .unwrap();
    let hole = native.dataset("b").unwrap().offset().unwrap();
    native.unlink("b").unwrap();
    native.close().unwrap();
    std::fs::copy(&path, &tolerant_path).unwrap();
    let properties = FileAccessProperties::new().with_memory_strategy(memory);
    let tolerant = File::open_rw_with_options(&tolerant_path, properties).unwrap();
    let strict = File::open_rw_with_options(
        &path,
        properties.with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict),
    )
    .unwrap();
    let reusable = strict.reusable_free_sections().unwrap();
    assert_eq!(reusable, tolerant.reusable_free_sections().unwrap());
    assert_eq!(
        test_util_hdf5::covered_free_bytes!(
            &reusable,
            &(hole..hole + 1600),
            FreeSectionCoordinates::Absolute
        ),
        1600
    );
    strict
        .root()
        .create_dataset("d", |builder| {
            builder.with_i32_data(&[9; 300]);
        })
        .unwrap();
    strict.commit().unwrap();
    strict.close().unwrap();
    tolerant.close().unwrap();
    let native = hdf5::File::open(&path).unwrap();
    assert_eq!(
        native.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        [9; 300]
    );
    assert_eq!(
        native.dataset("a").unwrap().read_raw::<i32>().unwrap(),
        [1; 100]
    );
}

#[rstest]
fn supported_free_sections_match_libhdf5(
    #[values(false, true)] paged: bool,
    #[values(0, 4096)] userblock_size: u64,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("supported-free-sections.h5");
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|properties| properties.libver_v110())
            .with_fcpl(|properties| {
                properties.userblock(userblock_size).file_space_strategy(
                    CStrategy::FreeSpaceManager {
                        paged,
                        persist: true,
                        threshold: 1,
                    },
                )
            })
            .create(&path)
            .unwrap();
        file.new_dataset_builder()
            .with_data(&[1i32; 100])
            .create("a")
            .unwrap();
        file.new_dataset_builder()
            .with_data(&[2i32; 400])
            .create("b")
            .unwrap();
        file.new_dataset_builder()
            .with_data(&[3i32; 100])
            .create("c")
            .unwrap();
        file.unlink("b").unwrap();
        file.close().unwrap();
    }
    let native = hdf5::File::open(&path).unwrap();
    let mut expected = file_space::libhdf5_free_sections(&native).unwrap();
    expected.sort_unstable();
    let free_bytes = native.free_space();
    let boundary = native.size() - userblock_size;
    drop(native);
    let report = assert_supported_free_sections(&path);
    assert_section_map(report.free_sections(), &expected, boundary);
    assert_eq!(report.free_sections().free_bytes(), free_bytes);
    if paged {
        assert!(
            report
                .managers()
                .iter()
                .all(|manager| match manager.slot() {
                    0 => manager.role() == FreeSpaceManagerRole::PageSmallMetadata,
                    2 => manager.role() == FreeSpaceManagerRole::PageSmallRawData,
                    6 => manager.role() == FreeSpaceManagerRole::PageLargeGeneric,
                    _ => manager.role() == FreeSpaceManagerRole::Unknown,
                })
        );
    } else {
        assert!(
            report
                .managers()
                .iter()
                .all(|manager| manager.role() == FreeSpaceManagerRole::FsmAggr)
        );
    }
}

#[rstest]
fn the_free_sections_libhdf5_persists_are_read_at_the_widths_of_its_file(
    #[values(Driver::Sec2, Driver::Core)] driver: Driver,
    #[values(Sizeof::Bytes2, Sizeof::Bytes4, Sizeof::Bytes8)] sizeof_addr: Sizeof,
    #[values(Sizeof::Bytes2, Sizeof::Bytes4, Sizeof::Bytes8)] sizeof_size: Sizeof,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_persisted_sections.h5");
    let freed = write_c_persisted_with_a_hole(&path, driver, sizeof_addr, sizeof_size, 0);
    let free_c = hdf5::File::open(&path).unwrap().free_space();

    let free = assert_supported_free_sections(&path);

    assert!(
        free.free_sections()
            .sections()
            .iter()
            .any(|section| { section.address() == freed && section.length().get() == 1600 }),
        "the deleted dataset's storage at {freed} is a free section, {free:?}"
    );
    assert_eq!(free.free_sections().free_bytes(), free_c);
    // `H5FS__cache_sinfo_deserialize` in H5FScache.c, HDF5 2.2.0, asserts on these fixtures
    // when their length width is smaller than eight bytes.
    if sizeof_size == Sizeof::Bytes8 {
        // libhdf5 sizes Core section addresses and lengths from the host `size_t` width
        // (`CORE_MAXADDR` in H5FDcore.c and `H5FS__sinfo_new` in H5FSsection.c, HDF5 2.2.0).
        let width_suffix = if matches!(driver, Driver::Core) && cfg!(target_pointer_width = "32") {
            "-32bit"
        } else {
            ""
        };
        super::file_space_oracle::checkpoint(
            &path,
            &format!("widths-{driver:?}-{sizeof_addr:?}-{sizeof_size:?}{width_suffix}"),
            "persisted",
        );
    }
}

// libhdf5 writes one 197-byte section list with 9 unused bytes before its checksum.
#[cfg(feature = "__hdf5-1.12")]
#[test]
fn the_sections_libhdf5_persists_in_an_oversized_section_list_are_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_oversized_section_list.h5");
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(|fcpl| {
                fcpl.file_space_strategy(CStrategy::FreeSpaceManager {
                    paged: false,
                    persist: true,
                    threshold: 1,
                })
            })
            .create(&path)
            .unwrap();
        for i in 0..16 {
            let group = file.create_group(&format!("g{i}")).unwrap();
            for j in 0..i % 7 {
                let len = 1 + (i + j) % 5;
                group
                    .new_attr::<i32>()
                    .shape(len)
                    .create(format!("attr{j}").as_str())
                    .unwrap()
                    .write(&vec![0; len])
                    .unwrap();
            }
        }
        for i in (0..16).step_by(2) {
            file.unlink(&format!("g{i}")).unwrap();
        }
        file.close().unwrap();
    }
    let free_c = hdf5::File::open(&path).unwrap().free_space();

    let file = File::open(&path).unwrap();
    let superblock = file.superblock();
    let widths = FormatWidths {
        offsets: superblock.offset_size,
        lengths: superblock.length_size,
    };
    let bytes = std::fs::read(&path).unwrap();
    let padded: Vec<(u64, u64)> = file
        .file_space_info()
        .unwrap()
        .unwrap()
        .manager_addrs
        .iter()
        .filter_map(|&addr| match addr {
            hdf5_pure::DecodedStoredAddress::Defined(stored) => Some(stored),
            hdf5_pure::DecodedStoredAddress::Undefined => None,
        })
        .map(|stored| {
            let header = FreeSpaceManagerHeader::parse(
                widths,
                &bytes[usize::try_from(stored.get()).unwrap()..],
            )
            .unwrap();
            let fsse_at = usize::try_from(header.section_list_addr().get()).unwrap();
            let sizes: Vec<u64> = hdf5_pure_format::__private::parse_section_info(
                widths,
                &bytes[fsse_at..],
                stored,
                &header,
            )
            .unwrap()
            .iter()
            .map(|section| section.size)
            .collect();
            (
                header.section_list_used(),
                hdf5_pure_format::__private::section_info_len(widths, &sizes),
            )
        })
        .filter(|&(used, len)| used > len)
        .collect();

    assert_eq!(padded, vec![(197, 188)]);
    assert_eq!(
        file.persisted_free_sections()
            .unwrap()
            .unwrap()
            .free_sections()
            .free_bytes(),
        free_c
    );
}

// The serializer rewrites each manager libhdf5 wrote in place, and libhdf5 then edits the file.
#[rstest]
fn libhdf5_loads_the_managers_the_serializer_writes_at_the_widths_of_its_file(
    #[values(Sizeof::Bytes2, Sizeof::Bytes4, Sizeof::Bytes8)] sizeof_addr: Sizeof,
    #[values(Sizeof::Bytes2, Sizeof::Bytes4, Sizeof::Bytes8)] sizeof_size: Sizeof,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ours_persisted_sections.h5");
    let freed = write_c_persisted_with_a_hole(&path, Driver::Sec2, sizeof_addr, sizeof_size, 0);
    let file = File::open(&path).unwrap();
    let superblock = file.superblock();
    let widths = FormatWidths {
        offsets: superblock.offset_size,
        lengths: superblock.length_size,
    };
    let managers: Vec<u64> = file
        .file_space_info()
        .unwrap()
        .unwrap()
        .manager_addrs
        .iter()
        .filter_map(|&addr| match addr {
            hdf5_pure::DecodedStoredAddress::Defined(stored) => Some(stored.get()),
            hdf5_pure::DecodedStoredAddress::Undefined => None,
        })
        .collect();
    drop(file);

    let mut bytes = std::fs::read(&path).unwrap();
    let mut sections = Vec::new();
    for addr in managers {
        let fshd_at = usize::try_from(addr).unwrap();
        let header = FreeSpaceManagerHeader::parse(widths, &bytes[fshd_at..]).unwrap();
        let fsse_at = usize::try_from(header.section_list_addr().get()).unwrap();
        let fsse_used = usize::try_from(header.section_list_used()).unwrap();
        let manager_sections = hdf5_pure_format::__private::parse_section_info(
            widths,
            &bytes[fsse_at..fsse_at + fsse_used],
            StoredAddress::new(addr),
            &header,
        )
        .unwrap();
        let (fshd, fsse) = hdf5_pure_format::__private::serialize_free_space_manager(
            widths,
            &manager_sections,
            StoredAddress::new(addr),
            header.section_list_addr(),
            SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        assert!(fsse.len() <= fsse_used, "{} > {fsse_used}", fsse.len());
        bytes[fshd_at..fshd_at + fshd.len()].copy_from_slice(&fshd);
        bytes[fsse_at..fsse_at + fsse.len()].copy_from_slice(&fsse);
        sections.extend(manager_sections);
    }
    std::fs::write(&path, &bytes).unwrap();
    assert!(
        sections
            .iter()
            .any(|section| (section.addr.get(), section.size) == (freed, 1600)),
        "the deleted dataset's storage at {freed} is a free section, {sections:?}"
    );

    let file = hdf5::File::open(&path).unwrap();
    assert_eq!(
        file.free_space(),
        sections.iter().map(|section| section.size).sum::<u64>()
    );
    drop(file);
    {
        let file = hdf5::File::open_rw(&path).unwrap();
        file.new_dataset::<i32>()
            .shape((30,))
            .create("added")
            .unwrap()
            .write(&(0..30).collect::<Vec<i32>>())
            .unwrap();
        file.unlink("big").unwrap();
        file.close().unwrap();
    }
    let file = hdf5::File::open(&path).unwrap();
    for (name, value) in [("a", 1), ("c", 3)] {
        assert_eq!(
            file.dataset(name).unwrap().read_raw::<i32>().unwrap(),
            vec![value; 400]
        );
    }
    assert_eq!(
        file.dataset("added").unwrap().read_raw::<i32>().unwrap(),
        (0..30).collect::<Vec<i32>>()
    );
    let free_c = file.free_space();
    drop(file);
    let free = File::open(&path)
        .unwrap()
        .persisted_free_sections()
        .unwrap()
        .unwrap();
    assert_eq!(free.free_sections().free_bytes(), free_c);
}

#[rstest]
#[case::plain(0)]
#[case::userblock(512)]
fn libhdf5_reads_a_section_list_whose_allocation_the_test_enlarges(#[case] userblock: u64) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("enlarged_section_list.h5");
    let (free, _) = enlarged_section_list_file(&path, userblock);
    let manager = free
        .managers()
        .iter()
        .find(|manager| {
            manager.free_sections().free_bytes() == 1600
                && manager.free_sections().section_count() == 1
        })
        .unwrap();
    assert_eq!(manager.metadata_bytes(), 82 + 35);
    // Enumeration of this enlarged list reaches `H5FS__sinfo_unlock`'s used/allocated-length
    // assertion (`H5FSsection.c`, HDF5 2.2.0). A fresh `H5Fget_info2` query uses allocated capacity.
    assert_eq!(
        super::file_space_oracle::assert_manager_metadata_bytes(&path),
        free.manager_metadata_bytes() + 16,
    );
    let file = hdf5::File::open(&path).unwrap();
    assert_eq!(file.free_space(), free.free_sections().free_bytes());
    assert_eq!(
        file.dataset("a").unwrap().read_raw::<i32>().unwrap(),
        vec![1; 400]
    );
    file.close().unwrap();
    assert_eq!(
        File::open(&path)
            .unwrap()
            .persisted_free_sections()
            .unwrap()
            .unwrap()
            .free_sections(),
        free.free_sections()
    );
}

#[test]
fn an_editor_reclaims_the_complete_section_list_allocation_the_test_enlarges() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reclaimed_section_list.h5");
    let (_, allocation) = enlarged_section_list_file(&path, 0);
    let session = File::open_rw(&path).unwrap();
    session
        .root()
        .set_attr("changed", AttrValue::I64(7))
        .unwrap();
    session.commit().unwrap();
    let free = session.reusable_free_sections().unwrap();
    session.close().unwrap();
    let file_len = std::fs::metadata(&path).unwrap().len();
    assert_eq!(
        test_util_hdf5::covered_free_bytes!(&free, &allocation, FreeSectionCoordinates::Absolute)
            + range::covered_len(&[(file_len, u64::MAX - file_len)], &allocation),
        allocation.end - allocation.start
    );
    let file = hdf5::File::open(&path).unwrap();
    assert_eq!(
        file.dataset("a").unwrap().read_raw::<i32>().unwrap(),
        vec![1; 400]
    );
    assert_eq!(
        file.attr("changed").unwrap().read_scalar::<i64>().unwrap(),
        7
    );
}

fn enlarged_section_list_file(path: &Path, userblock: u64) -> (PersistedFreeSections, Range<u64>) {
    write_c_persisted_with_a_hole(
        path,
        Driver::Sec2,
        Sizeof::Bytes8,
        Sizeof::Bytes8,
        userblock,
    );
    let file = File::open(path).unwrap();
    let mut superblock = file.superblock().clone();
    let free = file.persisted_free_sections().unwrap().unwrap();
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let base = superblock.base_address;
    let mut bytes = std::fs::read(path).unwrap();
    let (header_at, header) = file
        .file_space_info()
        .unwrap()
        .unwrap()
        .manager_addrs
        .iter()
        .filter_map(|&addr| match addr {
            hdf5_pure::DecodedStoredAddress::Defined(stored) => Some(stored),
            hdf5_pure::DecodedStoredAddress::Undefined => None,
        })
        .map(|stored| {
            let at = usize::try_from(base.absolute(stored).unwrap()).unwrap();
            (
                at,
                FreeSpaceManagerHeader::parse(widths, &bytes[at..]).unwrap(),
            )
        })
        .find(|(_, header)| (header.total_space(), header.total_sections()) == (1600, 1))
        .unwrap();
    drop(file);
    let used = usize::try_from(header.section_list_used()).unwrap();
    let list_at = usize::try_from(base.absolute(header.section_list_addr()).unwrap()).unwrap();
    let list = bytes[list_at..list_at + used].to_vec();
    let new_addr = base.relative(u64::try_from(bytes.len()).unwrap()).unwrap();
    bytes.extend_from_slice(&list);
    bytes.extend_from_slice(&[0xA5; 16]);
    let allocated = header.section_list_used() + 16;
    test_util::bytes::set_slice_at(
        &mut bytes,
        header_at + SECTION_LIST_ADDR_AT,
        &new_addr.get().to_le_bytes(),
    );
    test_util::bytes::set_slice_at(
        &mut bytes,
        header_at + SECTION_LIST_ALLOCATED_AT,
        &allocated.to_le_bytes(),
    );
    let header_len = usize::try_from(hdf5_pure_format::__private::free_space_manager_header_len(
        widths,
    ))
    .unwrap();
    test_util::checksum::restamp(&mut bytes, header_at, header_len);
    superblock.eof_address = u64::try_from(bytes.len()).unwrap();
    let encoded = hdf5_pure_format::__private::serialize_superblock(&superblock).unwrap();
    let at = usize::try_from(userblock).unwrap();
    bytes[at..at + encoded.len()].copy_from_slice(&encoded);
    std::fs::write(path, &bytes).unwrap();

    let header = FreeSpaceManagerHeader::parse(widths, &bytes[header_at..]).unwrap();
    assert_eq!(
        (header.section_list_used(), header.section_list_allocated()),
        (u64::try_from(list.len()).unwrap(), allocated)
    );
    let start = base.absolute(new_addr).unwrap();
    (free, start..start + allocated)
}

#[rstest]
#[case::header_version(ManagerBlock::Header, 4, vec![1], "the header version is 1, not 0")]
#[case::header_section_count(
    ManagerBlock::Header,
    14,
    vec![0xFF; 8],
    "the header's section count is 18446744073709551615, its serialized section count 1 and its \
     unserialized section count 0, and the file client serializes every section"
)]
#[case::section_list_version(
    ManagerBlock::SectionList,
    4,
    vec![1],
    "the section list version is 1, not 0"
)]
fn a_libhdf5_manager_with_an_invalid_field_is_reported_and_seeds_no_free_space(
    #[case] block: ManagerBlock,
    #[case] at: usize,
    #[case] value: Vec<u8>,
    #[case] reason: &str,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_invalid_manager.h5");
    let freed =
        write_c_persisted_with_a_hole(&path, Driver::Sec2, Sizeof::Bytes8, Sizeof::Bytes8, 0);
    assert!(
        reusable_free_space(&path)
            .iter()
            .any(|&(addr, len)| addr <= freed && freed + 1600 <= addr + len),
        "the premise: an editable open seeds the deleted dataset's storage at {freed}"
    );
    let range = deleted_dataset_manager_block(&path, block);
    let mut bytes = std::fs::read(&path).unwrap();
    test_util::bytes::set_slice_at(&mut bytes, range.start + at, &value);
    test_util::checksum::restamp(&mut bytes, range.start, range.len());
    std::fs::write(&path, &bytes).unwrap();

    let err = File::open(&path)
        .unwrap()
        .persisted_free_sections()
        .unwrap_err();
    if at == 4 {
        assert!(
            matches!(err, Error::Format(FormatError::UnsupportedVersion(1))),
            "{err:?}"
        );
    } else {
        let Error::Format(FormatError::InvalidFreeSpaceManager(message)) = &err else {
            panic!("expected InvalidFreeSpaceManager, got {err:?}");
        };
        assert_eq!(message, reason);
    }
    assert_eq!(reusable_free_space(&path), Vec::new());
}

#[test]
fn a_libhdf5_section_list_whose_checksum_does_not_match_is_reported_and_seeds_no_free_space() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_section_list_checksum.h5");
    write_c_persisted_with_a_hole(&path, Driver::Sec2, Sizeof::Bytes8, Sizeof::Bytes8, 0);
    let range = deleted_dataset_manager_block(&path, ManagerBlock::SectionList);
    let mut bytes = std::fs::read(&path).unwrap();
    let at = range.end - 4;
    let stored = u32::from_le_bytes(bytes[at..range.end].try_into().unwrap());
    test_util::bytes::set_slice_at(&mut bytes, at, &(!stored).to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

    let err = File::open(&path)
        .unwrap()
        .persisted_free_sections()
        .unwrap_err();
    let Error::Format(FormatError::ChecksumMismatch { expected, computed }) = err else {
        panic!("expected ChecksumMismatch, got {err:?}");
    };
    assert_eq!((expected, computed), (!stored, stored));
    assert_eq!(reusable_free_space(&path), Vec::new());
}

#[test]
fn an_editor_never_frees_the_live_bytes_a_libhdf5_header_points_to_as_its_section_list() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_header_into_live_data.h5");
    write_c_persisted_with_a_hole(&path, Driver::Sec2, Sizeof::Bytes8, Sizeof::Bytes8, 0);
    let live = hdf5::File::open(&path)
        .unwrap()
        .dataset("a")
        .unwrap()
        .offset()
        .unwrap();
    let pointed_to = live + 16..live + 80;
    let header = deleted_dataset_manager_block(&path, ManagerBlock::Header);
    let mut bytes = std::fs::read(&path).unwrap();
    for (at, value) in [
        (SECTION_LIST_ADDR_AT, pointed_to.start),
        (SECTION_LIST_USED_AT, pointed_to.end - pointed_to.start),
        (SECTION_LIST_ALLOCATED_AT, pointed_to.end - pointed_to.start),
    ] {
        test_util::bytes::set_slice_at(&mut bytes, header.start + at, &value.to_le_bytes());
    }
    test_util::checksum::restamp(&mut bytes, header.start, header.len());
    std::fs::write(&path, &bytes).unwrap();

    let err = File::open(&path)
        .unwrap()
        .persisted_free_sections()
        .unwrap_err();
    let Error::Format(FormatError::InvalidFreeSpaceManager(message)) = &err else {
        panic!("expected InvalidFreeSpaceManager, got {err:?}");
    };
    assert_eq!(
        *message,
        format!(
            "the section list signature is b\"{}\", not b\"FSSE\"",
            1i32.to_ne_bytes().escape_ascii()
        )
    );
    let session = File::open_rw(&path).unwrap();
    session.root().delete("big").unwrap();
    for round in 0..3 {
        session
            .root()
            .create_dataset(&format!("added{round}"), |b| {
                b.with_i32_data(&[round; 16]);
            })
            .unwrap();
        session.commit().unwrap();
        let free = session.reusable_free_sections().unwrap();
        assert!(
            !free
                .sections()
                .iter()
                .any(|section| section.address() < pointed_to.end
                    && pointed_to.start < section.end_address()),
            "after commit {round}, the free regions {free:?} reach the {pointed_to:?} the header points to"
        );
    }
    session.close().unwrap();
    let file = hdf5::File::open(&path).unwrap();
    for (name, value) in [("a", 1), ("c", 3)] {
        assert_eq!(
            file.dataset(name).unwrap().read_raw::<i32>().unwrap(),
            vec![value; 400]
        );
    }
}

#[test]
fn a_libhdf5_section_list_of_another_header_is_reported_and_neither_seeded_nor_reclaimed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_section_list_of_another_header.h5");
    write_c_persisted_with_a_hole(&path, Driver::Sec2, Sizeof::Bytes8, Sizeof::Bytes8, 0);
    let header = deleted_dataset_manager_block(&path, ManagerBlock::Header);
    let list = deleted_dataset_manager_block(&path, ManagerBlock::SectionList);
    let mut bytes = std::fs::read(&path).unwrap();
    test_util::bytes::set_slice_at(
        &mut bytes,
        list.start + SECTION_LIST_HEADER_ADDR_AT,
        &0u64.to_le_bytes(),
    );
    test_util::checksum::restamp(&mut bytes, list.start, list.len());
    std::fs::write(&path, &bytes).unwrap();

    let err = File::open(&path)
        .unwrap()
        .persisted_free_sections()
        .unwrap_err();
    let Error::Format(FormatError::InvalidFreeSpaceManager(message)) = &err else {
        panic!("expected InvalidFreeSpaceManager, got {err:?}");
    };
    assert_eq!(
        *message,
        format!(
            "the section list belongs to the header at 0, not to the header at {}",
            header.start
        )
    );
    assert_eq!(reusable_free_space(&path), Vec::new());
    let session = File::open_rw(&path).unwrap();
    session
        .root()
        .create_dataset("added", |b| {
            b.with_i32_data(&[7; 16]);
        })
        .unwrap();
    session.commit().unwrap();
    let free = session.reusable_free_sections().unwrap();
    session.close().unwrap();
    let list_addr = u64::try_from(list.start).unwrap();
    let list_end = u64::try_from(list.end).unwrap();
    assert!(
        !free
            .sections()
            .iter()
            .any(|section| section.address() < list_end && list_addr < section.end_address()),
        "the free regions {free:?} reach the section list at {list:?}"
    );
    assert_eq!(std::fs::read(&path).unwrap()[list.clone()], bytes[list]);
}

/// A block of a free-space manager.
#[derive(Clone, Copy, Debug)]
enum ManagerBlock {
    Header,
    SectionList,
}

/// Returns the range of the file at `path` that `block` takes, of the manager libhdf5 wrote at
/// 8-byte widths for the 1600 bytes of the deleted dataset.
fn deleted_dataset_manager_block(path: &std::path::Path, block: ManagerBlock) -> Range<usize> {
    let manager_addrs = File::open(path)
        .unwrap()
        .file_space_info()
        .unwrap()
        .unwrap()
        .manager_addrs
        .clone();
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let bytes = std::fs::read(path).unwrap();
    let (fshd_at, header) = manager_addrs
        .into_iter()
        .filter_map(|addr| match addr {
            hdf5_pure::DecodedStoredAddress::Defined(stored) => Some(stored),
            hdf5_pure::DecodedStoredAddress::Undefined => None,
        })
        .map(|stored| {
            let at = usize::try_from(stored.get()).unwrap();
            (
                at,
                FreeSpaceManagerHeader::parse(widths, &bytes[at..]).unwrap(),
            )
        })
        .find(|(_, header)| (header.total_space(), header.total_sections()) == (1600, 1))
        .expect("libhdf5 records the deleted dataset's storage in a manager of its own");
    let (at, len) = match block {
        ManagerBlock::Header => (
            fshd_at,
            hdf5_pure_format::__private::free_space_manager_header_len(widths),
        ),
        ManagerBlock::SectionList => (
            usize::try_from(header.section_list_addr().get()).unwrap(),
            header.section_list_used(),
        ),
    };
    at..at + usize::try_from(len).unwrap()
}

/// Returns the free regions an editable open of the file at `path` seeds its session with.
fn reusable_free_space(path: &std::path::Path) -> Vec<(u64, u64)> {
    File::open_rw(path)
        .unwrap()
        .reusable_free_sections()
        .unwrap()
        .sections()
        .iter()
        .map(|section| {
            assert_eq!(section.coordinates(), FreeSectionCoordinates::Absolute);
            (section.address(), section.length().get())
        })
        .collect()
}

/// The file driver libhdf5 writes a file through.
#[derive(Clone, Copy, Debug)]
enum Driver {
    /// The core driver, with the file as its backing store.
    Core,
    /// The default driver.
    Sec2,
}

/// Creates a file at `path` that persists its free-space managers, with libhdf5 through `driver`,
/// at the widths `sizeof_addr` and `sizeof_size` and after a userblock of `userblock` bytes, and
/// returns the file offset of the 1600 bytes the deleted dataset `b` held.
fn write_c_persisted_with_a_hole(
    path: &std::path::Path,
    driver: Driver,
    sizeof_addr: Sizeof,
    sizeof_size: Sizeof,
    userblock: u64,
) -> u64 {
    let file = hdf5::FileBuilder::new()
        .with_fapl(|fapl| match driver {
            Driver::Core => fapl.libver_v110().core_filebacked(true),
            Driver::Sec2 => fapl.libver_v110(),
        })
        .with_fcpl(|fcpl| {
            fcpl.sizes(SizeofInfo {
                sizeof_addr,
                sizeof_size,
            })
            .userblock(userblock)
            .file_space_strategy(CStrategy::FreeSpaceManager {
                paged: false,
                persist: true,
                threshold: 1,
            })
        })
        .create(path)
        .unwrap();
    for (name, value, len) in [
        ("a", 1, 400),
        ("b", 2, 400),
        ("c", 3, 400),
        ("big", 4, 5000),
    ] {
        file.new_dataset::<i32>()
            .shape((len,))
            .create(name)
            .unwrap()
            .write(&vec![value; len])
            .unwrap();
    }
    let freed = file.dataset("b").unwrap().offset().unwrap();
    file.unlink("b").unwrap();
    file.close().unwrap();
    freed
}

#[test]
fn we_read_c_library_strategy() {
    let dir = tempfile::tempdir().unwrap();

    // Each case: a C-written strategy and what hdf5-pure should report. The C
    // library only writes a File Space Info message for non-default settings, so
    // these all differ from the default (FSM_AGGR, non-persistent).
    let write_c = |path: &std::path::Path, strategy: CStrategy| {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(move |fcpl| fcpl.file_space_strategy(strategy))
            .create(path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape((3,))
            .create("d")
            .unwrap()
            .write(&[1i32, 2, 3])
            .unwrap();
        file.close().unwrap();
    };

    // NONE and AGGR: non-persistent, no manager addresses.
    let p_none = dir.path().join("c_none.h5");
    write_c(&p_none, CStrategy::None);
    let f = File::open(&p_none).unwrap();
    assert_eq!(
        f.file_space_info().unwrap(),
        Some(unmanaged_file_space_info(FileSpaceStrategy::None))
    );
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap(), vec![1, 2, 3]);

    let p_aggr = dir.path().join("c_aggr.h5");
    write_c(&p_aggr, CStrategy::PageAggregation);
    assert_eq!(
        File::open(&p_aggr).unwrap().file_space_info().unwrap(),
        Some(unmanaged_file_space_info(FileSpaceStrategy::Aggr))
    );

    // FSM_AGGR with persist=true: the message carries free-space-manager
    // addresses, which we parse (without following them to their on-disk blocks).
    let p_persist = dir.path().join("c_persist.h5");
    write_c(
        &p_persist,
        CStrategy::FreeSpaceManager {
            paged: false,
            persist: true,
            threshold: 5,
        },
    );
    let f = File::open(&p_persist).unwrap();
    let info = f
        .file_space_info()
        .unwrap()
        .expect("persisted file records a strategy");
    assert_eq!(info.strategy, FileSpaceStrategy::FsmAggr);
    assert!(info.persist, "persist flag is read");
    assert_eq!(info.threshold, 5);
    assert!(
        !info.manager_addrs.is_empty(),
        "a persisted file carries free-space-manager addresses"
    );
    // The data still reads correctly.
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap(), vec![1, 2, 3]);
}

/// The settings of a file whose `strategy` has no free-space managers: no persistence, the default
/// threshold and page size, and no manager addresses.
fn unmanaged_file_space_info(strategy: FileSpaceStrategy) -> FileSpaceInfo {
    FileSpaceInfoFields {
        strategy,
        persist: false,
        threshold: DEFAULT_THRESHOLD,
        page_size: FileSpacePageSize::DEFAULT,
        page_end_meta_threshold: 0,
        eoa_pre_fsm: Some(hdf5_pure::DecodedStoredAddress::Undefined),
        manager_addrs: Vec::new(),
    }
    .build()
}

// Only HDF5 1.10.0 writes a version 0 message, so the version 0 tests replace the message libhdf5
// writes.
#[rstest]
#[case::all(
    file_space_info::ALL,
    CStrategy::FreeSpaceManager {
        paged: false,
        persist: false,
        threshold: 64,
    },
    FileSpaceStrategy::FsmAggr,
    64
)]
#[case::aggr_vfd(
    file_space_info::AGGR_VFD,
    CStrategy::PageAggregation,
    FileSpaceStrategy::Aggr,
    1
)]
#[case::vfd(file_space_info::VFD, CStrategy::None, FileSpaceStrategy::None, 1)]
fn a_version_0_strategy_reads_as_libhdf5_maps_it(
    #[case] version_0_strategy: u8,
    #[case] c_strategy: CStrategy,
    #[case] strategy: FileSpaceStrategy,
    #[case] threshold: u64,
    #[values(false, true)] streaming: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_version_0.h5");
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(|fcpl| {
                fcpl.file_space_strategy(CStrategy::FreeSpaceManager {
                    paged: false,
                    persist: false,
                    threshold: 64,
                })
            })
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape((3,))
            .create("d")
            .unwrap()
            .write(&[1i32, 2, 3])
            .unwrap();
        file.close().unwrap();
    }
    file_space_info::replace_message(
        &path,
        extension_addr(&path),
        &file_space_info::version_0(Widths::EIGHT, version_0_strategy, 64, &[]),
    );

    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        c_strategy
    );
    drop(f);
    let f = if streaming {
        File::open_streaming(&path).unwrap()
    } else {
        File::open(&path).unwrap()
    };
    assert_eq!(
        f.file_space_info().unwrap(),
        Some(version_0_mapped(strategy, false, threshold, Vec::new()))
    );
    assert_eq!(
        f.file_space_settings().unwrap(),
        Some(FileSpaceSettings::new(strategy).with_threshold(threshold))
    );
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap(), vec![1, 2, 3]);
}

#[rstest]
fn a_version_0_message_s_persisted_free_space_reads_as_libhdf5_reads_it(
    #[values(Sizeof::Bytes2, Sizeof::Bytes4, Sizeof::Bytes8)] sizeof_size: Sizeof,
    #[values(0, 512)] userblock: u64,
) {
    assert_version_0_free_space_reads_as_libhdf5_reads_it(Sizeof::Bytes8, sizeof_size, userblock);
}

// libhdf5 1.10.11 and 1.12.3 bound each address of a version 0 message by `H5_SIZEOF_HADDR_T`, so
// they reject a message whose addresses are narrower (`H5O_fsinfo_decode` in `H5Ofsinfo.c`, HDF5
// 1.10.11, and `H5O__fsinfo_decode`, HDF5 1.12.3).
#[cfg(feature = "__hdf5-1.14")]
#[rstest]
fn a_version_0_message_s_persisted_free_space_reads_as_libhdf5_reads_it_at_a_narrow_address_width(
    #[values(Sizeof::Bytes2, Sizeof::Bytes4)] sizeof_addr: Sizeof,
    #[values(Sizeof::Bytes2, Sizeof::Bytes4, Sizeof::Bytes8)] sizeof_size: Sizeof,
) {
    assert_version_0_free_space_reads_as_libhdf5_reads_it(sizeof_addr, sizeof_size, 0);
}

fn assert_version_0_free_space_reads_as_libhdf5_reads_it(
    sizeof_addr: Sizeof,
    sizeof_size: Sizeof,
    userblock: u64,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_persisted_version_0.h5");
    let freed =
        write_c_persisted_with_a_hole(&path, Driver::Sec2, sizeof_addr, sizeof_size, userblock);
    let undefined =
        StoredAddress::undefined(OffsetWidth::try_from(sizeof_addr as u8).unwrap()).get();
    let version_1 = rewrite_as_version_0(
        &path,
        Widths::new(sizeof_addr.into(), sizeof_size.into()),
        undefined,
    );

    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: false,
            persist: true,
            threshold: 1,
        }
    );
    let free_c = f.free_space();
    drop(f);
    let f = File::open(&path).unwrap();
    let (version_0_managers, _) = version_1
        .manager_addrs
        .split_at(file_space_info::VERSION_0_MANAGERS);
    let settings = Some(FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true));
    assert_eq!(f.file_space_settings().unwrap(), settings);
    let streamed = File::open_streaming(&path).unwrap();
    assert_eq!(streamed.file_space_settings().unwrap(), settings);
    assert_eq!(
        streamed.file_space_info().unwrap(),
        f.file_space_info().unwrap()
    );
    assert_eq!(
        f.file_space_info().unwrap(),
        Some(version_0_mapped(
            FileSpaceStrategy::FsmAggr,
            true,
            1,
            version_0_managers.to_vec(),
        ))
    );
    let free = assert_supported_free_sections(&path);
    assert!(
        free.free_sections().sections().iter().any(|section| {
            section.address() == freed - userblock && section.length().get() == 1600
        }),
        "{free:?}"
    );
    assert_eq!(free.free_sections().free_bytes(), free_c);
    // `H5FS__cache_sinfo_deserialize` in H5FScache.c, HDF5 2.2.0, asserts on these fixtures
    // when their length width is smaller than eight bytes.
    if sizeof_size == Sizeof::Bytes8 {
        drop(f);
        super::file_space_oracle::checkpoint(
            &path,
            &format!("version-0-{sizeof_addr:?}-{sizeof_size:?}-base-{userblock}"),
            "persisted",
        );
    }
}

#[test]
fn libhdf5_reads_a_version_0_file_after_an_edit_reuses_its_persisted_free_space() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_persisted_version_0_edited.h5");
    let freed =
        write_c_persisted_with_a_hole(&path, Driver::Sec2, Sizeof::Bytes8, Sizeof::Bytes8, 0);
    rewrite_as_version_0(&path, Widths::EIGHT, u64::MAX);

    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("d", |b| {
                b.with_i32_data(&[9; 300]);
            })
            .unwrap();
        s.commit().unwrap();
    }
    let total_ours: u64 = File::open(&path)
        .unwrap()
        .persisted_free_sections()
        .unwrap()
        .unwrap()
        .free_sections()
        .free_bytes();

    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: false,
            persist: true,
            threshold: 1,
        }
    );
    let d = f.dataset("d").unwrap();
    assert!(
        (freed..freed + 1600).contains(&d.offset().unwrap()),
        "the new dataset's data is in the hole at {freed}, at {:?}",
        d.offset()
    );
    assert_eq!(d.read_raw::<i32>().unwrap(), vec![9; 300]);
    assert_eq!(
        f.dataset("a").unwrap().read_raw::<i32>().unwrap(),
        vec![1; 400]
    );
    assert_eq!(
        f.dataset("c").unwrap().read_raw::<i32>().unwrap(),
        vec![3; 400]
    );
    assert_eq!(
        f.dataset("big").unwrap().read_raw::<i32>().unwrap(),
        vec![4; 5000]
    );
    assert_eq!(f.free_space(), total_ours);
}

#[test]
fn libhdf5_reads_a_non_persisting_version_0_file_after_an_edit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_non_persisted_version_0_edited.h5");
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(|fcpl| {
                fcpl.file_space_strategy(CStrategy::FreeSpaceManager {
                    paged: false,
                    persist: false,
                    threshold: 64,
                })
            })
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape((3,))
            .create("d")
            .unwrap()
            .write(&[1i32, 2, 3])
            .unwrap();
        file.close().unwrap();
    }
    file_space_info::replace_message(
        &path,
        extension_addr(&path),
        &file_space_info::version_0(Widths::EIGHT, file_space_info::ALL, 64, &[]),
    );

    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("extra", |b| {
                b.with_i32_data(&[10, 20, 30]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: false,
            persist: false,
            threshold: 64,
        }
    );
    assert_eq!(
        f.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(
        f.dataset("extra").unwrap().read_raw::<i32>().unwrap(),
        vec![10, 20, 30]
    );
}

#[rstest]
fn libhdf5_reads_its_file_after_an_edit_reuses_the_free_space_it_persisted(
    #[values(0, 512)] userblock: u64,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_persisted_edited.h5");
    let freed = write_c_persisted_with_a_hole(
        &path,
        Driver::Sec2,
        Sizeof::Bytes8,
        Sizeof::Bytes8,
        userblock,
    );
    let stamped = Userblock::read(&path, usize::try_from(userblock).unwrap());
    let persisted = File::open(&path)
        .unwrap()
        .persisted_free_sections()
        .unwrap()
        .unwrap();
    assert!(
        persisted.free_sections().sections().iter().any(|section| {
            section.coordinates() == FreeSectionCoordinates::Stored
                && section.address() == freed - userblock
                && section.length().get() == 1600
        }),
        "{persisted:?}"
    );

    let session = File::open_rw(&path).unwrap();
    let reusable = session.reusable_free_sections().unwrap();
    assert_eq!(
        reusable.section_count(),
        persisted.free_sections().section_count()
    );
    for (physical, stored) in reusable
        .sections()
        .iter()
        .zip(persisted.free_sections().sections())
    {
        assert_eq!(physical.coordinates(), FreeSectionCoordinates::Absolute);
        assert_eq!(stored.coordinates(), FreeSectionCoordinates::Stored);
        assert_eq!(physical.address(), stored.address() + userblock);
        assert_eq!(physical.length(), stored.length());
    }
    session
        .root()
        .create_dataset("d", |builder| {
            builder.with_i32_data(&[9; 300]);
        })
        .unwrap();
    session.commit().unwrap();
    session.close().unwrap();
    let ours = File::open(&path)
        .unwrap()
        .persisted_free_sections()
        .unwrap()
        .unwrap();

    let f = hdf5::File::open(&path).unwrap();
    let d = f.dataset("d").unwrap();
    assert_eq!(
        file_space::libhdf5_contiguous_extent(&d),
        freed..freed + 1200
    );
    assert_eq!(d.read_raw::<i32>().unwrap(), vec![9; 300]);
    assert_eq!(
        f.dataset("a").unwrap().read_raw::<i32>().unwrap(),
        vec![1; 400]
    );
    assert_eq!(
        f.dataset("big").unwrap().read_raw::<i32>().unwrap(),
        vec![4; 5000]
    );
    assert_eq!(f.free_space(), ours.free_sections().free_bytes());
    assert_section_map(
        ours.free_sections(),
        &file_space::libhdf5_free_sections(&f).unwrap(),
        f.size() - userblock,
    );
    drop(d);
    f.close().unwrap();
    stamped.assert_unchanged(&path);
}

#[rstest]
fn libhdf5_reads_relocated_persisted_free_space_and_writes_to_the_file(
    #[values(0, 512)] userblock: u64,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ours_persisted_relocated.h5");
    let mut builder = FileBuilder::new();
    if userblock > 0 {
        builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
        builder.with_userblock_content(b"relocated managers");
    }
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    builder.create_dataset("b").with_i32_data(&[2; 400]);
    builder.create_dataset("c").with_i32_data(&[3; 100]);
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    builder.write(&path).unwrap();
    let stamped = Userblock::read(&path, usize::try_from(userblock).unwrap());

    let first = File::open_rw(&path).unwrap();
    first.root().delete("b").unwrap();
    first.commit().unwrap();
    first.close().unwrap();
    let second = File::open_rw(&path).unwrap();
    second
        .root()
        .create_dataset("d", |builder| {
            builder.with_i32_data(&[9; 300]);
        })
        .unwrap();
    second.commit().unwrap();
    second.root().delete("c").unwrap();
    second.commit().unwrap();
    second.close().unwrap();
    let ours = File::open(&path)
        .unwrap()
        .persisted_free_sections()
        .unwrap()
        .unwrap();
    assert!(ours.free_sections().section_count() > 0);

    let f = hdf5::File::open_rw(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: false,
            persist: true,
            threshold: 1,
        }
    );
    assert_eq!(
        f.dataset("a").unwrap().read_raw::<i32>().unwrap(),
        vec![1; 100]
    );
    assert_eq!(
        f.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        vec![9; 300]
    );
    absence::assert_libhdf5_absent(&f.dataset("c").unwrap_err(), "c");
    assert_eq!(f.free_space(), ours.free_sections().free_bytes());
    assert_section_map(
        ours.free_sections(),
        &file_space::libhdf5_free_sections(&f).unwrap(),
        f.size() - userblock,
    );
    f.new_dataset::<i32>()
        .shape((50,))
        .create("e")
        .unwrap()
        .write(&[5; 50])
        .unwrap();
    f.close().unwrap();

    let reader = File::open(&path).unwrap();
    assert_eq!(
        reader.dataset("e").unwrap().read_i32().unwrap(),
        vec![5; 50]
    );
    assert_eq!(
        reader.dataset("d").unwrap().read_i32().unwrap(),
        vec![9; 300]
    );
    stamped.assert_unchanged(&path);
}

#[rstest]
fn libhdf5_reads_its_paged_file_after_an_edit_reuses_the_free_space_it_persisted(
    #[values(0, PAGE_SIZE)] userblock: u64,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_paged_persisted_edited.h5");
    write_c_paged(&path, userblock, 1, C_PAGED_DATASETS);
    let file = hdf5::File::open_rw(&path).unwrap();
    let freed = file.dataset("b").unwrap().offset().unwrap();
    file.unlink("b").unwrap();
    file.close().unwrap();
    let stamped = Userblock::read(&path, usize::try_from(userblock).unwrap());
    let file = File::open(&path).unwrap();
    let persisted = file.persisted_free_sections().unwrap().unwrap();
    let big = file_space::contiguous_extent(&file, "big");
    drop(file);
    let last_page_of_big = (
        big.end - userblock,
        PAGE_SIZE - (big.end - userblock) % PAGE_SIZE,
    );
    assert_eq!(
        persisted.free_sections().sections().iter().any(|section| {
            section.address() == last_page_of_big.0 && section.length().get() == last_page_of_big.1
        }),
        hdf5::library_version() < (1, 12, 0),
        "{persisted:?}"
    );

    let session = File::open_rw(&path).unwrap();
    let reusable = session.reusable_free_sections().unwrap();
    let recovered = persisted
        .free_sections()
        .sections()
        .iter()
        .filter(|section| {
            section.address() != last_page_of_big.0 || section.length().get() != last_page_of_big.1
        });
    assert_eq!(reusable.section_count(), recovered.clone().count());
    for (physical, stored) in reusable.sections().iter().zip(recovered) {
        assert_eq!(physical.coordinates(), FreeSectionCoordinates::Absolute);
        assert_eq!(stored.coordinates(), FreeSectionCoordinates::Stored);
        assert_eq!(physical.address(), stored.address() + userblock);
        assert_eq!(physical.length(), stored.length());
    }
    session
        .root()
        .create_dataset("d", |builder| {
            builder.with_i32_data(&[9; 300]);
        })
        .unwrap();
    session.commit().unwrap();
    session.close().unwrap();
    let ours = File::open(&path)
        .unwrap()
        .persisted_free_sections()
        .unwrap()
        .unwrap();

    let f = hdf5::File::open(&path).unwrap();
    let d = f.dataset("d").unwrap();
    assert_eq!(
        file_space::libhdf5_contiguous_extent(&d),
        freed..freed + 1200
    );
    assert_eq!(d.read_raw::<i32>().unwrap(), vec![9; 300]);
    for (value, &(name, len, _)) in (1..)
        .zip(C_PAGED_DATASETS)
        .filter(|&(_, &(name, ..))| name != "b")
    {
        assert_eq!(
            f.dataset(name).unwrap().read_raw::<i32>().unwrap(),
            vec![value; len],
            "{name}"
        );
    }
    assert_eq!(f.free_space(), ours.free_sections().free_bytes());
    assert_section_map(
        ours.free_sections(),
        &file_space::libhdf5_free_sections(&f).unwrap(),
        f.size() - userblock,
    );
    drop(d);
    f.close().unwrap();

    let f = hdf5::File::open_rw(&path).unwrap();
    f.new_dataset::<i32>()
        .shape((50,))
        .create("e")
        .unwrap()
        .write(&[5; 50])
        .unwrap();
    f.close().unwrap();
    let reader = File::open(&path).unwrap();
    assert_eq!(
        reader.dataset("e").unwrap().read_i32().unwrap(),
        vec![5; 50]
    );
    assert_eq!(
        reader.dataset("d").unwrap().read_i32().unwrap(),
        vec![9; 300]
    );
    stamped.assert_unchanged(&path);
}

#[rstest]
fn libhdf5_reads_and_writes_our_edited_paged_file(#[values(0, PAGE_SIZE)] userblock: u64) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ours_paged_edited.h5");
    let mut builder = FileBuilder::new();
    if userblock > 0 {
        builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
        builder.with_userblock_content(b"paged managers");
    }
    builder.create_dataset("a").with_i32_data(&[1; 100]);
    builder.create_dataset("b").with_i32_data(&[2; 400]);
    builder.create_dataset("c").with_i32_data(&[3; 100]);
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_page_size(FileSpacePageSize::try_from(PAGE_SIZE).unwrap()),
    );
    builder.write(&path).unwrap();
    let stamped = Userblock::read(&path, usize::try_from(userblock).unwrap());

    let first = File::open_rw(&path).unwrap();
    first.root().delete("b").unwrap();
    first.commit().unwrap();
    first.close().unwrap();
    let second = File::open_rw(&path).unwrap();
    second
        .root()
        .create_dataset("d", |builder| {
            builder.with_i32_data(&[9; 300]);
        })
        .unwrap();
    second.commit().unwrap();
    second.root().delete("c").unwrap();
    second.commit().unwrap();
    second.close().unwrap();
    paged::assert_consistent(&path, PAGE_SIZE);
    let ours = File::open(&path)
        .unwrap()
        .persisted_free_sections()
        .unwrap()
        .unwrap();
    assert!(ours.free_sections().section_count() > 0);

    let f = hdf5::File::open_rw(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        page(1)
    );
    assert_eq!(f.create_plist().unwrap().userblock(), userblock);
    assert_eq!(
        f.dataset("a").unwrap().read_raw::<i32>().unwrap(),
        vec![1; 100]
    );
    assert_eq!(
        f.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        vec![9; 300]
    );
    absence::assert_libhdf5_absent(&f.dataset("c").unwrap_err(), "c");
    assert_eq!(f.free_space(), ours.free_sections().free_bytes());
    assert_section_map(
        ours.free_sections(),
        &file_space::libhdf5_free_sections(&f).unwrap(),
        f.size() - userblock,
    );
    f.new_dataset::<i32>()
        .shape((50,))
        .create("e")
        .unwrap()
        .write(&[5; 50])
        .unwrap();
    f.close().unwrap();

    let reader = File::open(&path).unwrap();
    assert_eq!(
        reader.dataset("e").unwrap().read_i32().unwrap(),
        vec![5; 50]
    );
    assert_eq!(
        reader.dataset("d").unwrap().read_i32().unwrap(),
        vec![9; 300]
    );
    stamped.assert_unchanged(&path);
}

const C_PAGED_DATASETS: &[(&str, usize, CLayout)] = &[
    ("a", 400, CLayout::Contiguous),
    ("b", 400, CLayout::Contiguous),
    ("c", 400, CLayout::Contiguous),
    ("big", 5000, CLayout::Contiguous),
];

/// Rewrites the version 1 File Space Info message of the file at `path` as a version 0 message with
/// its first six manager addresses, and returns the version 1 message.
///
/// # Panics
///
/// Panics if any of the six other manager addresses is not `undefined`.
fn rewrite_as_version_0(path: &Path, widths: Widths, undefined: u64) -> FileSpaceInfo {
    let version_1 = File::open(path)
        .unwrap()
        .file_space_info()
        .unwrap()
        .unwrap();
    let (version_0_managers, unused) = version_1
        .manager_addrs
        .split_at(file_space_info::VERSION_0_MANAGERS);
    assert_eq!(
        unused,
        [hdf5_pure::DecodedStoredAddress::Undefined;
            NUM_FILE_FSM_MANAGERS - file_space_info::VERSION_0_MANAGERS]
    );
    let v0_raw_managers: Vec<u64> = version_0_managers
        .iter()
        .map(|&m| match m {
            hdf5_pure::DecodedStoredAddress::Defined(addr) => addr.get(),
            hdf5_pure::DecodedStoredAddress::Undefined => undefined,
        })
        .collect();
    file_space_info::replace_message(
        path,
        extension_addr(path),
        &file_space_info::version_0(
            widths,
            file_space_info::ALL_PERSIST,
            version_1.threshold,
            &v0_raw_managers,
        ),
    );
    version_1
}

/// Returns the absolute file position of the superblock extension of the file at `path`.
fn extension_addr(path: &Path) -> u64 {
    let file = File::open(path).unwrap();
    let superblock = file.superblock();
    superblock
        .base_address
        .absolute(superblock.superblock_extension_address.unwrap())
        .unwrap()
}

/// Returns the fields of version 1 that a version 0 message maps to.
fn version_0_mapped(
    strategy: FileSpaceStrategy,
    persist: bool,
    threshold: u64,
    manager_addrs: Vec<hdf5_pure::DecodedStoredAddress>,
) -> FileSpaceInfo {
    FileSpaceInfoFields {
        strategy,
        persist,
        threshold,
        page_size: FileSpacePageSize::DEFAULT,
        page_end_meta_threshold: 0,
        eoa_pre_fsm: None,
        manager_addrs,
    }
    .build()
}

#[rstest]
#[case::settled_spare(1, true)]
#[case::no_sections(4_096, false)]
fn c_library_reads_our_fresh_persisting_file(#[case] threshold: u64, #[case] has_sections: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ours_fresh_persist.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3]);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(threshold),
    );
    b.write(&path).unwrap();

    let total_ours: u64 = File::open(&path)
        .unwrap()
        .persisted_free_sections()
        .unwrap()
        .unwrap()
        .free_sections()
        .free_bytes();
    assert_eq!(total_ours != 0, has_sections);

    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: false,
            persist: true,
            threshold,
        }
    );
    assert_eq!(
        f.dataset("d").unwrap().read_raw::<i32>().unwrap(),
        vec![1, 2, 3]
    );
    let free_c = f.free_space() as i64;
    assert_eq!(free_c as u64, total_ours);
}

#[rstest]
#[case::a_power_of_two(16384)]
#[case::not_a_power_of_two(3000)]
fn c_library_reads_our_paged_file(#[case] page_size: u64) {
    // hdf5-pure writes a genuine paged (H5F_FSPACE_STRATEGY_PAGE) persisting file
    // with small and large datasets. libhdf5 must recover the
    // paged strategy, read every dataset, load the per-page-type free-space
    // managers (H5Fget_freespace parses our FSHD/FSSE blocks), and reopen the
    // file read-write to mutate it (proving the paged layout is consistent).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ours_paged.h5");

    paged::write_small_large_i32(&path, page_size);
    let small_a: Vec<i32> = (0..100).collect();
    let small_b: Vec<i32> = (0..400).collect();
    let big: Vec<i32> = (0..5000).collect();
    assert_eq!(std::fs::metadata(&path).unwrap().len() % page_size, 0);

    // hdf5-pure's own view of the tracked free space (SUPER + DRAW + LARGE tails).
    let ours = File::open(&path).unwrap();
    let total_ours: u64 = ours
        .persisted_free_sections()
        .unwrap()
        .unwrap()
        .free_sections()
        .free_bytes();
    assert!(total_ours > 0, "paged persist tracks page-tail free space");
    drop(ours);

    // libhdf5 recovers the paged strategy and reads every dataset.
    let f = hdf5::File::open(&path).unwrap();
    let strategy = f.create_plist().unwrap().get_file_space_strategy().unwrap();
    assert_eq!(
        strategy,
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 0,
        },
        "C library recovers our paged strategy"
    );
    assert_eq!(
        f.create_plist()
            .unwrap()
            .get_file_space_page_size()
            .unwrap(),
        page_size
    );
    assert_eq!(f.dataset("a").unwrap().read_raw::<i32>().unwrap(), small_a);
    assert_eq!(f.dataset("b").unwrap().read_raw::<i32>().unwrap(), small_b);
    assert_eq!(f.dataset("big").unwrap().read_raw::<i32>().unwrap(), big);

    // Loading the managers requires parsing our FSHD/FSSE blocks; the C library's
    // free-space total equals the sum of the sections we wrote.
    let free_c = f.free_space() as i64;
    assert_eq!(
        free_c as u64, total_ours,
        "C library free-space total matches our paged managers"
    );
    drop(f);

    // libhdf5 reopens the paged file read-write and adds a dataset, then
    // reads everything back: the paged layout survives a C round-trip.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        f.new_dataset::<i32>()
            .shape((30,))
            .create("added")
            .unwrap()
            .write(&(0..30).collect::<Vec<i32>>())
            .unwrap();
        f.close().unwrap();
    }
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(f.dataset("a").unwrap().read_raw::<i32>().unwrap(), small_a);
    assert_eq!(f.dataset("big").unwrap().read_raw::<i32>().unwrap(), big);
    assert_eq!(
        f.dataset("added").unwrap().read_raw::<i32>().unwrap(),
        (0..30).collect::<Vec<i32>>()
    );
}

#[test]
fn c_library_reads_our_paged_chunked_file() {
    // Paged interop for chunked datasets: hdf5-pure writes a paged persisting
    // file with a small chunked dataset and a large compressed (shuffle+deflate)
    // chunked dataset whose data begins a page-aligned run. libhdf5 must
    // read both, parse the managers, and reopen read-write.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ours_paged_chunked.h5");

    paged::write_chunked_f64(&path, 16384);
    let small: Vec<f64> = (0..64).map(|i| i as f64).collect();
    let big: Vec<f64> = (0..8000).map(|i| i as f64 * 0.5).collect();

    let ours = File::open(&path).unwrap();
    let total_ours: u64 = ours
        .persisted_free_sections()
        .unwrap()
        .unwrap()
        .free_sections()
        .free_bytes();
    drop(ours);

    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 0,
        },
    );
    assert_eq!(f.dataset("s").unwrap().read_raw::<f64>().unwrap(), small);
    assert_eq!(f.dataset("big").unwrap().read_raw::<f64>().unwrap(), big);
    let free_c = f.free_space() as i64;
    assert_eq!(
        free_c as u64, total_ours,
        "C free-space matches our managers"
    );
    drop(f);

    // C reopens read-write and appends a dataset without corrupting the file.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        f.new_dataset::<f64>()
            .shape((4,))
            .create("extra")
            .unwrap()
            .write(&[1.0f64, 2.0, 3.0, 4.0])
            .unwrap();
        f.close().unwrap();
    }
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(f.dataset("big").unwrap().read_raw::<f64>().unwrap(), big);
    assert_eq!(
        f.dataset("extra").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn c_library_reads_our_bounded_mutated_paged_file() {
    // hdf5-pure creates a genuine paged persisting file, then grows it through the
    // bounded backend, appending enough rows to force
    // extensible-array index growth so the append allocates metadata as well as
    // raw chunks (exercising page segregation). libhdf5 must then
    // recover the paged strategy, read every row, load the per-page-type managers
    // (H5Fget_freespace parses our FSHD/FSSE blocks), and reopen it read-write.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ours_paged_mutated.h5");

    // Create with one chunk, then bounded-append to 5000 rows.
    paged::write_unlimited_i32(&path, 4096, true, 64, 64);
    {
        let file = session::open_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(64..5000).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    // hdf5-pure's own view of the tracked free space after the mutation.
    let want: Vec<i32> = (0..5000).collect();
    let ours = File::open(&path).unwrap();
    assert_eq!(ours.dataset("d").unwrap().read_i32().unwrap(), want);
    let total_ours: u64 = ours
        .persisted_free_sections()
        .unwrap()
        .unwrap()
        .free_sections()
        .free_bytes();
    drop(ours);

    // libhdf5 recovers the paged strategy and reads every row.
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 0,
        },
        "C library recovers our paged strategy after a bounded mutation"
    );
    assert_eq!(f.dataset("d").unwrap().read_raw::<i32>().unwrap(), want);
    // Loading the managers parses our rewritten FSHD/FSSE blocks; the C library's
    // free-space total equals the sum of the sections we wrote.
    let free_c = f.free_space() as i64;
    assert_eq!(
        free_c as u64, total_ours,
        "C free-space total matches our rewritten paged managers"
    );
    drop(f);

    // libhdf5 reopens the mutated paged file read-write and appends more
    // rows, then reads everything back: the paged layout survives a C round-trip.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        let ds = f.dataset("d").unwrap();
        ds.resize((5100,)).unwrap();
        ds.write_slice(&(5000..5100).collect::<Vec<i32>>(), 5000..5100)
            .unwrap();
        f.close().unwrap();
    }
    let f = hdf5::File::open(&path).unwrap();
    let after = f.dataset("d").unwrap().read_raw::<i32>().unwrap();
    assert_eq!(after.len(), 5100);
    assert_eq!(after[..5000], want[..]);
    assert_eq!(after[5099], 5099);
}

#[test]
fn c_library_reads_our_staged_mutated_paged_file() {
    // The whole-file editor's counterpart of the bounded crosscheck above (issue
    // #198): hdf5-pure creates a genuine paged persisting file, then mutates it
    // through `File::open_rw` + staged edits + `commit`, which allocates both raw
    // data (a new dataset's values, appended chunks) and metadata (object headers,
    // the rebuilt chunk index, the rewritten managers). libhdf5
    // must then recover the paged strategy, read every dataset, load the
    // per-page-type managers, and reopen the file read-write.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ours_paged_staged.h5");

    paged::write_unlimited_i32(&path, 4096, true, 64, 64);
    {
        let file = File::open_rw(&path).unwrap();
        // A staged append (raw chunks + a rebuilt extensible-array index) and a
        // fresh contiguous dataset (raw data + a new object header), so the commit
        // allocates both page types and has to switch between them.
        let mut ds = file.dataset("d").unwrap();
        ds.append_staged(|b| {
            b.append_i32(&(64..2000).collect::<Vec<i32>>());
        })
        .unwrap();
        file.root()
            .create_dataset("added", |b| {
                b.with_f64_data(&vec![2.5f64; 512]);
            })
            .unwrap();
        file.commit().unwrap();
    }

    let want: Vec<i32> = (0..2000).collect();
    let ours = File::open(&path).unwrap();
    assert_eq!(ours.dataset("d").unwrap().read_i32().unwrap(), want);
    assert_eq!(
        ours.dataset("added").unwrap().read_f64().unwrap(),
        vec![2.5f64; 512]
    );
    let total_ours: u64 = ours
        .persisted_free_sections()
        .unwrap()
        .unwrap()
        .free_sections()
        .free_bytes();
    drop(ours);

    // libhdf5 recovers the paged strategy and reads both datasets.
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 0,
        },
        "C library recovers our paged strategy after a staged commit"
    );
    assert_eq!(f.dataset("d").unwrap().read_raw::<i32>().unwrap(), want);
    assert_eq!(
        f.dataset("added").unwrap().read_raw::<f64>().unwrap(),
        vec![2.5f64; 512]
    );
    // Loading the managers parses the per-page-type FSHD/FSSE blocks the commit
    // wrote; the C library's free-space total equals the sum of our sections.
    let free_c = f.free_space() as i64;
    assert_eq!(
        free_c as u64, total_ours,
        "C free-space total matches our rewritten paged managers"
    );
    drop(f);

    // libhdf5 reopens the file read-write and appends: the paged layout the
    // staged commit produced survives a C round-trip.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        let ds = f.dataset("d").unwrap();
        ds.resize((2100,)).unwrap();
        ds.write_slice(&(2000..2100).collect::<Vec<i32>>(), 2000..2100)
            .unwrap();
        f.close().unwrap();
    }
    let f = hdf5::File::open(&path).unwrap();
    let after = f.dataset("d").unwrap().read_raw::<i32>().unwrap();
    assert_eq!(after.len(), 2100);
    assert_eq!(after[..2000], want[..]);
    assert_eq!(after[2099], 2099);
}

#[test]
fn c_library_reads_our_paged_file_after_free_space_reuse() {
    // Issue #261: a paged file now draws on its own free space, so a delete
    // followed by a write of the same shape reuses the pages the delete released
    // instead of appending past them. Reuse is confined to the page type being
    // placed, which keeps every page holding one kind of byte — an invariant the
    // C library depends on for page buffering and cannot report on directly. What
    // it *can* do is prove the result is a well-formed paged file: recover the
    // strategy, read every dataset back, and re-parse the rewritten per-page-type
    // managers into the same free-space total we record.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ours_paged_reused.h5");

    let kept: Vec<f64> = (0..4096).map(|i| (i % 17) as f64).collect();
    {
        let mut b = FileBuilder::new();
        b.create_dataset("victim")
            .with_f64_data(&vec![1.0; 8192])
            .with_chunks(&[1024]);
        b.create_dataset("ceiling").with_f64_data(&kept);
        b.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page)
                .with_persistence(true)
                .with_threshold(0)
                .with_page_size(FileSpacePageSize::DEFAULT),
        );
        b.write(&path).unwrap();
    }
    let size_before = std::fs::metadata(&path).unwrap().len();

    let replacement: Vec<f64> = (0..8192).map(|i| (i % 29) as f64).collect();
    {
        let file = File::open_rw(&path).unwrap();
        file.root().delete("victim").unwrap();
        file.commit().unwrap();
        file.root()
            .create_dataset("replacement", |b| {
                b.with_f64_data(&replacement).with_chunks(&[1024]);
            })
            .unwrap();
        file.commit().unwrap();
    }

    let size_after = std::fs::metadata(&path).unwrap().len();
    assert!(
        size_after < size_before + 8192 * 8 / 4,
        "the replacement should reuse the freed raw pages (was {size_before}, \
         now {size_after})"
    );

    let ours = File::open(&path).unwrap();
    assert_eq!(
        ours.dataset("replacement").unwrap().read_f64().unwrap(),
        replacement
    );
    let total_ours: u64 = ours
        .persisted_free_sections()
        .unwrap()
        .unwrap()
        .free_sections()
        .free_bytes();
    drop(ours);

    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 0,
        },
        "C library recovers our paged strategy after a commit that reused free space"
    );
    assert_eq!(
        f.dataset("replacement").unwrap().read_raw::<f64>().unwrap(),
        replacement,
        "the C library resolves a chunk index placed in reused pages"
    );
    assert_eq!(
        f.dataset("ceiling").unwrap().read_raw::<f64>().unwrap(),
        kept
    );
    let free_c = f.free_space() as i64;
    assert_eq!(
        free_c as u64, total_ours,
        "C free-space total matches the managers the reusing commit rewrote"
    );
    drop(f);

    // And the file is still writable by the C library afterwards.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        f.new_dataset::<i32>()
            .shape([4])
            .create("c_added")
            .unwrap()
            .write_raw(&[1i32, 2, 3, 4])
            .unwrap();
        f.close().unwrap();
    }
    let ours = File::open(&path).unwrap();
    assert_eq!(
        ours.dataset("c_added").unwrap().read_i32().unwrap(),
        vec![1, 2, 3, 4]
    );
    assert_eq!(
        ours.dataset("replacement").unwrap().read_f64().unwrap(),
        replacement
    );
}

#[test]
fn c_library_reads_our_paged_file_after_a_cross_page_type_claim() {
    // Issue #286: a page holding nothing at all belongs to no page type, so either
    // kind may open one — which is what lets metadata reuse the pages a deleted
    // dataset's chunk data released, and how free space survives a close at all
    // (whole free pages are recorded in the manager that names no type, so every
    // one of them comes back as raw). It is also the mechanism most able to break
    // page homogeneity, the invariant the C library depends on for page buffering
    // and cannot report on directly. What it can do is prove the result is a
    // well-formed paged file it reads, re-parses, and writes to.
    //
    // The shape below is what forces the claim: delete the chunked dataset to free
    // whole raw pages, then commit metadata too large for any metadata hole that
    // remains, so the allocator has to open one of those raw pages for it.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ours_paged_crosstype.h5");

    let kept: Vec<f64> = (0..512).map(|i| (i % 17) as f64).collect();
    {
        let mut b = FileBuilder::new();
        b.create_dataset("victim")
            .with_f64_data(&vec![1.0; 8192])
            .with_chunks(&[1024]);
        b.create_dataset("ceiling").with_f64_data(&kept);
        b.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page)
                .with_persistence(true)
                .with_threshold(0)
                .with_page_size(FileSpacePageSize::DEFAULT),
        );
        b.write(&path).unwrap();
    }

    let label = "x".repeat(3000);
    {
        let file = File::open_rw(&path).unwrap();
        file.root().delete("victim").unwrap();
        file.commit().unwrap();
        for i in 0..8 {
            file.root()
                .create_group_with(&format!("g{i}"), |g| {
                    g.set_attr("label", AttrValue::String(label.clone()));
                })
                .unwrap();
        }
        file.commit().unwrap();
    }

    let ours = File::open(&path).unwrap();
    assert_eq!(ours.dataset("ceiling").unwrap().read_f64().unwrap(), kept);
    let total_ours: u64 = ours
        .persisted_free_sections()
        .unwrap()
        .unwrap()
        .free_sections()
        .free_bytes();
    drop(ours);

    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 0,
        },
        "C library recovers our paged strategy after metadata opened a freed raw page"
    );
    for i in 0..8 {
        assert!(
            f.group(&format!("g{i}")).is_ok(),
            "the C library reads a group header placed in a page a delete released"
        );
    }
    assert_eq!(
        f.dataset("ceiling").unwrap().read_raw::<f64>().unwrap(),
        kept
    );
    let free_c = f.free_space() as i64;
    assert_eq!(
        free_c as u64, total_ours,
        "C free-space total matches the managers the claiming commit rewrote"
    );
    drop(f);

    // And the file is still writable by the C library afterwards.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        f.new_dataset::<i32>()
            .shape([4])
            .create("c_added")
            .unwrap()
            .write_raw(&[5i32, 6, 7, 8])
            .unwrap();
        f.close().unwrap();
    }
    let ours = File::open(&path).unwrap();
    assert_eq!(
        ours.dataset("c_added").unwrap().read_i32().unwrap(),
        vec![5, 6, 7, 8]
    );
    assert_eq!(ours.dataset("ceiling").unwrap().read_f64().unwrap(), kept);
}

#[test]
#[cfg(target_endian = "little")]
fn pure_bounded_mutates_c_created_paged_file() {
    // The reverse direction: libhdf5 creates a genuine paged
    // persisting file with an unlimited chunked dataset; hdf5-pure grows it through
    // the bounded backend; then both the C library and hdf5-pure read every row
    // back and the C library re-parses the rewritten managers.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c_paged_pure_mutated.h5");

    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(|fcpl| {
                fcpl.file_space_strategy(CStrategy::FreeSpaceManager {
                    paged: true,
                    persist: true,
                    threshold: 1,
                })
                .file_space_page_size(4096)
            })
            .create(&path)
            .unwrap();
        let ds = file
            .new_dataset::<i32>()
            .chunk((64,))
            .shape((hdf5::Extent::resizable(64),))
            .create("d")
            .unwrap();
        ds.write(&(0..64).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    // hdf5-pure grows the C-created paged file.
    {
        let file = session::open_bounded(&path).unwrap();
        let mut ds = file.dataset("d").unwrap();
        ds.append(&(64..4000).collect::<Vec<i32>>()).unwrap();
        file.close().unwrap();
    }

    let want: Vec<i32> = (0..4000).collect();
    // hdf5-pure reads its own mutation back.
    let ours = File::open(&path).unwrap();
    assert_eq!(ours.dataset("d").unwrap().read_i32().unwrap(), want);
    let total_ours: u64 = ours
        .persisted_free_sections()
        .unwrap()
        .unwrap()
        .free_sections()
        .free_bytes();
    drop(ours);

    // libhdf5 reads it back and re-parses the managers.
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 1,
        },
    );
    assert_eq!(f.dataset("d").unwrap().read_raw::<i32>().unwrap(), want);
    let free_c = f.free_space() as i64;
    assert_eq!(
        free_c as u64, total_ours,
        "C free-space matches our managers"
    );
}

#[test]
fn c_library_reads_our_paged_file_after_group_churn() {
    // The reclaim a paged file performs under delete-and-recreate churn is a claim
    // about which bytes are free, and the reference library is the reader that has
    // to agree with it (issue #388). hdf5-pure creates a paged persisting file,
    // then repeatedly creates and deletes a scratch group of empty resizable
    // chunked datasets — the shape whose chunk index the reclaim used to drop,
    // leaving the page it sat in unaccountable. libhdf5 must recover the
    // strategy, read the surviving objects, and report exactly the free space our
    // managers record, which it can only do by parsing the FSHD/FSSE blocks the
    // churn rewrote.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ours_paged_churn.h5");

    let mut b = FileBuilder::new();
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(0)
            .with_page_size(FileSpacePageSize::try_from(16384).unwrap()),
    );
    b.create_dataset("keep").with_i32_data(&[1, 2, 3, 4]);
    b.write(&path).unwrap();

    {
        let file = File::open_rw_with_options(
            &path,
            FileAccessProperties::new().with_sync_policy(hdf5_pure::SyncPolicy::OnClose),
        )
        .unwrap();
        for cycle in 0..20 {
            let name = format!("scratch_{cycle}");
            file.root()
                .create_group_with(&name, |g| {
                    for i in 0..3 {
                        g.create_dataset(&format!("log{i}"), |b| {
                            b.with_i32_data(&[])
                                .with_shape(&[0])
                                .with_chunks(&[64])
                                .with_maxshape(&[MaxExtent::Unlimited]);
                        });
                    }
                })
                .unwrap();
            file.commit().unwrap();
            file.root().delete(&name).unwrap();
            file.commit().unwrap();
        }
        // One scratch group survives, so the file is not merely back to its
        // original shape when the C library reads it.
        file.root()
            .create_group_with("live", |g| {
                g.create_dataset("log", |b| {
                    b.with_i32_data(&[7i32, 8, 9])
                        .with_shape(&[3])
                        .with_chunks(&[2])
                        .with_maxshape(&[MaxExtent::Unlimited]);
                });
            })
            .unwrap();
        file.commit().unwrap();
        file.close().unwrap();
    }

    let ours = File::open(&path).unwrap();
    let total_ours: u64 = ours
        .persisted_free_sections()
        .unwrap()
        .unwrap()
        .free_sections()
        .free_bytes();
    assert!(total_ours > 0, "the churn left free space to account for");
    drop(ours);

    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.create_plist().unwrap().get_file_space_strategy().unwrap(),
        CStrategy::FreeSpaceManager {
            paged: true,
            persist: true,
            threshold: 0,
        },
        "C library recovers our paged strategy after the churn"
    );
    assert_eq!(
        f.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
        vec![1, 2, 3, 4]
    );
    assert_eq!(
        f.dataset("live/log").unwrap().read_raw::<i32>().unwrap(),
        vec![7, 8, 9]
    );
    let free_c = f.free_space() as i64;
    assert_eq!(
        free_c as u64, total_ours,
        "C free-space total matches the managers the churn rewrote"
    );
    drop(f);

    // And the C library reopens the churned file read-write and writes into it.
    {
        let f = hdf5::File::open_rw(&path).unwrap();
        f.new_dataset::<f64>()
            .shape((4,))
            .create("extra")
            .unwrap()
            .write(&[1.0f64, 2.0, 3.0, 4.0])
            .unwrap();
        f.close().unwrap();
    }
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.dataset("extra").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        f.dataset("live/log").unwrap().read_raw::<i32>().unwrap(),
        vec![7, 8, 9]
    );
}

/// A file the commit shortened by releasing the free run at its end is read back
/// by the reference library — data, strategy, and free-space total alike
/// (issue #418).
///
/// The release publishes a *smaller* end-of-file and then truncates the file to
/// it, so the two figures the C library cross-checks on open — the superblock's
/// end-of-file and the length of the file it is opening — move together, and the
/// free-space managers it parses must name no section above either. `h5stat`
/// asks the same question through `H5Fget_freespace`, which can only answer it
/// by loading the FSHD/FSSE blocks the release rewrote; a section pointing past
/// the shortened file would be one it hands to the next writer.
///
/// Both strategies, because the run is released in whole pages on a paged file
/// and in bytes on a flat one, and then the C library writes into the shortened
/// file and hdf5-pure reads the result.
#[test]
fn c_library_reads_our_file_after_a_release_shortened_it() {
    let dir = tempfile::tempdir().unwrap();
    const ROWS: usize = 65536;
    let payload: Vec<i32> = (0..ROWS as i32).collect();

    for (label, strategy, paged) in [
        ("flat", FileSpaceStrategy::FsmAggr, false),
        ("paged", FileSpaceStrategy::Page, true),
    ] {
        let path = dir.path().join(format!("ours_released_{label}.h5"));
        let mut b = FileBuilder::new();
        let settings = FileSpaceSettings::new(strategy)
            .with_persistence(true)
            .with_threshold(0);
        b.with_file_space_settings(if paged {
            settings.with_page_size(FileSpacePageSize::try_from(16384).unwrap())
        } else {
            settings
        });
        b.create_dataset("keep").with_i32_data(&[1, 2, 3, 4]);
        b.write(&path).unwrap();

        let peak = {
            let file = File::open_rw_with_options(
                &path,
                FileAccessProperties::new().with_sync_policy(hdf5_pure::SyncPolicy::OnClose),
            )
            .unwrap();
            file.root()
                .create_group_with("scratch", |g| {
                    for c in 0..2 {
                        g.create_dataset(&format!("c{c}"), |b| {
                            b.with_i32_data(&payload)
                                .with_shape(&[ROWS as u64])
                                .with_chunks(&[512])
                                .with_maxshape(&[MaxExtent::Unlimited]);
                        });
                    }
                })
                .unwrap();
            file.commit().unwrap();
            let peak = file.file_size();
            file.root().delete("scratch").unwrap();
            file.commit().unwrap();
            file.close().unwrap();
            peak
        };

        let len = std::fs::metadata(&path).unwrap().len();
        assert!(
            len < peak / 4,
            "{label}: the delete should have released the end of the file (peak \
             {peak}, now {len})"
        );

        super::file_space_oracle::checkpoint(
            &path,
            &format!("tail-truncation-{label}"),
            "rust-close",
        );
        let ours = File::open(&path).unwrap();
        assert_eq!(ours.file_size(), len, "{label}: superblock EOF matches");
        let free_ours = ours.persisted_free_sections().unwrap().unwrap();
        for section in free_ours.free_sections().sections() {
            assert_eq!(section.coordinates(), FreeSectionCoordinates::Stored);
            let (addr, size) = (section.address(), section.length().get());
            assert!(
                addr + size <= len,
                "{label}: a persisted section [{addr}, {}) runs past the {len}-byte \
                 file",
                addr + size
            );
        }
        let total_ours = free_ours.free_sections().free_bytes();
        drop(ours);

        let f = hdf5::File::open(&path).unwrap();
        assert_eq!(
            f.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
            vec![1, 2, 3, 4],
            "{label}: the C library reads the survivor out of the shortened file"
        );
        let free_c = f.free_space() as i64;
        assert_eq!(
            free_c as u64, total_ours,
            "{label}: the C free-space total matches the managers the release rewrote"
        );
        drop(f);

        // And the C library writes into the shortened file, which hdf5-pure reads.
        {
            let f = hdf5::File::open_rw(&path).unwrap();
            f.new_dataset::<f64>()
                .shape((4,))
                .create("extra")
                .unwrap()
                .write(&[1.0f64, 2.0, 3.0, 4.0])
                .unwrap();
            f.close().unwrap();
        }
        let ours = File::open(&path).unwrap();
        assert_eq!(
            ours.dataset("extra").unwrap().read_f64().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0],
            "{label}: hdf5-pure reads what the C library added"
        );
        assert_eq!(
            ours.dataset("keep").unwrap().read_i32().unwrap(),
            vec![1, 2, 3, 4]
        );
    }
}

/// An in-place `Dataset::append` on a persisting file spends holes an earlier
/// commit left, and it does so by taking them out of the on-disk managers first
/// (issue #387). The reference library is the reader that has to agree with the
/// result: it re-parses those rewritten managers, reads every appended value
/// back, and reports exactly the free space we record.
///
/// Both strategies, because the reserve is drawn from one list on a flat file and
/// from the raw-typed one on a paged file, and only the C library's own accounting
/// says the paged managers still segregate what they should.
#[test]
fn c_library_reads_our_file_after_an_append_reused_free_space() {
    let dir = tempfile::tempdir().unwrap();

    // Comfortably more than one reserve batch, so the append draws from the hole
    // rather than declining it.
    let payload: Vec<i32> = (0..768 * 1024).collect();

    for (label, strategy, paged) in [
        ("flat", FileSpaceStrategy::FsmAggr, false),
        ("paged", FileSpaceStrategy::Page, true),
    ] {
        let path = dir.path().join(format!("ours_append_reuse_{label}.h5"));
        let mut b = FileBuilder::new();
        let settings = FileSpaceSettings::new(strategy).with_persistence(true);
        b.with_file_space_settings(if paged {
            settings.with_page_size(FileSpacePageSize::try_from(16384).unwrap())
        } else {
            settings
        });
        b.create_dataset("keep").with_i32_data(&[1, 2, 3, 4]);
        b.write(&path).unwrap();

        let make_log = |file: &File, name: &str| {
            file.root()
                .create_dataset(name, |b| {
                    b.with_i32_data(&[])
                        .with_shape(&[0])
                        .with_chunks(&[16384])
                        .with_maxshape(&[MaxExtent::Unlimited]);
                })
                .unwrap();
            file.commit().unwrap();
        };

        {
            let file = File::open_rw_with_options(
                &path,
                FileAccessProperties::new().with_sync_policy(hdf5_pure::SyncPolicy::OnClose),
            )
            .unwrap();
            make_log(&file, "log");
            file.dataset("log").unwrap().append(&payload).unwrap();
            // A live object above the churn, so the delete leaves an interior hole
            // rather than a run the commit simply truncates away.
            file.root()
                .create_dataset("ceiling", |b| {
                    b.with_i32_data(&[9, 9, 9]);
                })
                .unwrap();
            file.commit().unwrap();
            let after_first = std::fs::metadata(&path).unwrap().len();

            file.root().delete("log").unwrap();
            file.commit().unwrap();

            make_log(&file, "log2");
            file.dataset("log2").unwrap().append(&payload).unwrap();
            file.close().unwrap();

            let after_second = std::fs::metadata(&path).unwrap().len();
            // The delete left a hole one payload wide. An append that reuses it
            // grows the file barely at all; one that goes past the end grows it by
            // the whole payload. Half of that, in bytes, is the line — the same
            // figure the elements-times-two form above it worked out to, said in
            // the unit the file size is in.
            let payload_bytes = std::mem::size_of_val(&payload[..]) as u64;
            assert!(
                after_second < after_first + payload_bytes / 2,
                "{label}: the second append should have reused the hole (was \
                 {after_first}, now {after_second}, payload {payload_bytes} bytes)"
            );
        }

        super::file_space_oracle::checkpoint(&path, &format!("append-reuse-{label}"), "rust-close");
        let ours = File::open(&path).unwrap();
        assert_eq!(ours.dataset("log2").unwrap().read_i32().unwrap(), payload);
        let total_ours: u64 = ours
            .persisted_free_sections()
            .unwrap()
            .unwrap()
            .free_sections()
            .free_bytes();
        drop(ours);

        let f = hdf5::File::open(&path).unwrap();
        assert_eq!(
            f.create_plist().unwrap().get_file_space_strategy().unwrap(),
            CStrategy::FreeSpaceManager {
                paged,
                persist: true,
                threshold: 1,
            },
            "{label}: C library recovers our strategy after an append reused free space"
        );
        assert_eq!(
            f.dataset("log2").unwrap().read_raw::<i32>().unwrap(),
            payload,
            "{label}: the C library reads every value the reusing append wrote"
        );
        assert_eq!(
            f.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            f.dataset("ceiling").unwrap().read_raw::<i32>().unwrap(),
            vec![9, 9, 9]
        );
        let free_c = f.free_space() as i64;
        assert_eq!(
            free_c as u64, total_ours,
            "{label}: C free-space total matches the managers the reservation rewrote"
        );
        drop(f);

        // And the C library reopens the file read-write and allocates out of those
        // managers itself — the session that would have written over live data had
        // the append spent a hole the managers still advertised.
        {
            let f = hdf5::File::open_rw(&path).unwrap();
            f.new_dataset::<f64>()
                .shape((4096,))
                .create("c_added")
                .unwrap()
                .write(&vec![2.5f64; 4096])
                .unwrap();
            f.close().unwrap();
        }
        let ours = File::open(&path).unwrap();
        assert_eq!(
            ours.dataset("log2").unwrap().read_i32().unwrap(),
            payload,
            "{label}: the C library's own allocation must not have landed on live chunks"
        );
        assert_eq!(
            ours.dataset("c_added").unwrap().read_f64().unwrap(),
            vec![2.5f64; 4096]
        );
    }
}

#[rstest]
#[case::fsm_aggr(fsm_aggr(false, 1), true)]
#[case::fsm_aggr_below_the_extent(fsm_aggr(false, DELETED_LEN - 1), true)]
#[case::fsm_aggr_at_the_extent(fsm_aggr(false, DELETED_LEN), true)]
#[case::fsm_aggr_above_the_extent(fsm_aggr(false, DELETED_LEN + 1), false)]
#[case::fsm_aggr_persisting_below_the_extent(fsm_aggr(true, DELETED_LEN - 1), true)]
#[case::fsm_aggr_persisting_at_the_extent(fsm_aggr(true, DELETED_LEN), true)]
#[case::fsm_aggr_persisting_above_the_extent(fsm_aggr(true, DELETED_LEN + 1), false)]
#[case::page_below_the_extent(page(DELETED_LEN - 1), true)]
#[case::page_at_the_extent(page(DELETED_LEN), true)]
#[case::page_above_the_extent(page(DELETED_LEN + 1), false)]
#[case::aggr(CStrategy::PageAggregation, false)]
#[case::none(CStrategy::None, false)]
fn a_replacement_reuses_a_deleted_extent_where_libhdf5_does(
    #[case] strategy: CStrategy,
    #[case] reuses: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let ours = dir.path().join("ours.h5");
    let theirs = dir.path().join("theirs.h5");
    file_space::libhdf5_write_strategy_triplet(&ours, strategy);
    std::fs::copy(&ours, &theirs).unwrap();

    let ours_reuse = {
        let file = File::open_rw(&ours).unwrap();
        let deleted = file_space::contiguous_extent(&file, "b");
        file.root().delete("b").unwrap();
        file.commit().unwrap();
        file.root()
            .create_dataset("d", |b| {
                b.with_i32_data(&[4; 300]);
            })
            .unwrap();
        file.commit().unwrap();
        range::overlaps(&deleted, &file_space::contiguous_extent(&file, "d"))
    };
    let theirs_reuse = {
        let file = hdf5::File::open_rw(&theirs).unwrap();
        let deleted = file_space::libhdf5_contiguous_extent(&file.dataset("b").unwrap());
        file.unlink("b").unwrap();
        let replacement = file.new_dataset::<i32>().shape((300,)).create("d").unwrap();
        replacement.write(&[4; 300]).unwrap();
        range::overlaps(
            &deleted,
            &file_space::libhdf5_contiguous_extent(&replacement),
        )
    };

    assert_eq!((ours_reuse, theirs_reuse), (reuses, reuses));
}

#[rstest]
#[case::fsm_aggr_alone(false, &["c"], 0)]
#[case::fsm_aggr_beside_a_tracked_extent(false, &["b", "c"], SUB_THRESHOLD_LEN)]
#[case::fsm_aggr_persisting_alone(true, &["c"], 0)]
#[case::fsm_aggr_persisting_beside_a_tracked_extent(true, &["b", "c"], SUB_THRESHOLD_LEN)]
fn a_sub_threshold_extent_is_tracked_only_beside_a_tracked_extent_as_libhdf5_tracks_it(
    #[case] persist: bool,
    #[case] deletions: &[&str],
    #[case] tracked: u64,
) {
    let datasets = [
        ("a", 1, 100),
        ("b", 2, THRESHOLD_ELEMENTS),
        ("c", 3, SUB_THRESHOLD_ELEMENTS),
        ("e", 4, 100),
    ];
    let dir = tempfile::tempdir().unwrap();
    let ours = dir.path().join("ours.h5");
    let theirs = dir.path().join("theirs.h5");
    {
        let file = hdf5::FileBuilder::new()
            .with_fapl(|fapl| fapl.libver_v110())
            .with_fcpl(|fcpl| fcpl.file_space_strategy(fsm_aggr(persist, THRESHOLD)))
            .create(&ours)
            .unwrap();
        for (name, value, len) in datasets {
            file.new_dataset::<i32>()
                .shape((len,))
                .create(name)
                .unwrap()
                .write(&vec![value; len])
                .unwrap();
        }
        file.close().unwrap();
    }
    std::fs::copy(&ours, &theirs).unwrap();

    let (extent, ours_session) = {
        let file = File::open_rw(&ours).unwrap();
        let extent = file_space::contiguous_extent(&file, "c");
        assert_eq!(file_space::contiguous_extent(&file, "b").end, extent.start);
        for &path in deletions {
            file.root().delete(path).unwrap();
            file.commit().unwrap();
        }
        let tracked = test_util_hdf5::covered_free_bytes!(
            &file.reusable_free_sections().unwrap(),
            &extent,
            FreeSectionCoordinates::Absolute
        );
        (extent, tracked)
    };
    let theirs_session = {
        let file = hdf5::File::open_rw(&theirs).unwrap();
        let (last, earlier) = deletions.split_last().unwrap();
        for &path in earlier {
            file.unlink(path).unwrap();
        }
        let before = file.free_space();
        file.unlink(last).unwrap();
        file.free_space() - before
    };
    let (ours_reopened, ours_total) = {
        let file = File::open(&ours).unwrap();
        let reference = hdf5::File::open(&ours).unwrap();
        for (path, value, len) in datasets {
            if deletions.contains(&path) {
                continue;
            }
            let expected = vec![value; len];
            assert_eq!(file.dataset(path).unwrap().read_i32().unwrap(), expected);
            assert_eq!(
                reference.dataset(path).unwrap().read_raw::<i32>().unwrap(),
                expected
            );
        }
        let report = file.persisted_free_sections().unwrap();
        assert_eq!(report.is_some(), persist);
        match report {
            Some(report) => (
                covered_free_bytes(report.free_sections(), &extent),
                report.free_sections().free_bytes(),
            ),
            None => (0, 0),
        }
    };
    let c_total = hdf5::File::open(&ours).unwrap().free_space();

    assert_eq!(
        (ours_session, theirs_session, ours_reopened, c_total),
        (
            tracked,
            tracked,
            if persist { tracked } else { 0 },
            ours_total
        )
    );
    super::file_space_oracle::checkpoint(
        &ours,
        &format!("fsm-subthreshold-{persist}-{}", deletions.join("-")),
        "rust-close",
    );
    super::file_space_oracle::checkpoint(
        &theirs,
        &format!("fsm-subthreshold-{persist}-{}", deletions.join("-")),
        "c-close",
    );
}

#[rstest]
#[case::a_deletion(delete_c)]
#[case::a_creation(create_d)]
#[case::an_attribute(set_root_attribute)]
#[case::an_append_at_close(append_at_close)]
fn a_persisting_edit_that_tracks_no_free_space_records_the_end_of_allocation_for_libhdf5(
    #[case] edit: fn(&Path),
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ours.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[1; 100]);
    b.create_dataset("c").with_i32_data(&[3; 100]);
    Unlimited::new("u", &[4; 64], 64).add_to(&mut b);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(THRESHOLD_ABOVE_EVERY_EXTENT),
    );
    b.write(&path).unwrap();

    edit(&path);

    let (eoa_pre_fsm, manager_addrs, free, eof) = {
        let file = File::open(&path).unwrap();
        let info = file.file_space_info().unwrap().unwrap();
        (
            info.eoa_pre_fsm,
            info.manager_addrs.clone(),
            file.persisted_free_sections().unwrap().unwrap(),
            file.superblock().eof_address,
        )
    };
    let len = std::fs::metadata(&path).unwrap().len();
    let c_file = hdf5::File::open(&path).unwrap();
    assert_eq!(
        (
            eoa_pre_fsm,
            eof,
            manager_addrs,
            free.free_sections().section_count(),
            c_file.dataset("a").unwrap().read_raw::<i32>().unwrap(),
            c_file.free_space(),
        ),
        (
            Some(hdf5_pure::DecodedStoredAddress::Defined(
                hdf5_pure::StoredAddress::new(len)
            )),
            len,
            vec![hdf5_pure::DecodedStoredAddress::Undefined; NUM_FILE_FSM_MANAGERS],
            0,
            vec![1; 100],
            0
        )
    );
}

fn delete_c(path: &Path) {
    let file = File::open_rw(path).unwrap();
    file.root().delete("c").unwrap();
    file.commit().unwrap();
}

fn create_d(path: &Path) {
    let file = File::open_rw(path).unwrap();
    file.root()
        .create_dataset("d", |b| {
            b.with_i32_data(&[5; 100]);
        })
        .unwrap();
    file.commit().unwrap();
}

fn set_root_attribute(path: &Path) {
    let file = File::open_rw(path).unwrap();
    file.root().set_attr("n", AttrValue::I64(1)).unwrap();
    file.commit().unwrap();
}

fn append_at_close(path: &Path) {
    let file = File::open_rw(path).unwrap();
    let mut dataset = file.dataset("u").unwrap();
    dataset.append(&[6; 64]).unwrap();
    drop(dataset);
    file.close().unwrap();
}

#[rstest]
#[case::raw_short_of_the_threshold(EXTENT_LEN + 1, FOUR_EXTENTS, &["b"], "b", false)]
#[case::raw_at_the_threshold(EXTENT_LEN, FOUR_EXTENTS, &["b"], "b", true)]
#[case::raw_past_the_threshold(EXTENT_LEN - 1, FOUR_EXTENTS, &["b"], "b", true)]
#[case::raw_beside_a_tracked_extent(THRESHOLD_IN_A_PAGE, ABOVE_A_TRACKED_EXTENT, &["b", "c"], "c", true)]
#[case::raw_beside_the_page_tail(THRESHOLD_IN_A_PAGE, FOUR_EXTENTS, &["e"], "e", true)]
#[case::raw_pages_alone(THRESHOLD_OF_THREE_PAGES, PAGE_SPANS, &["d"], "d", false)]
#[case::raw_completing_a_page(THRESHOLD_OF_THREE_PAGES, PAGE_SPANS, &["a"], "a", false)]
#[case::raw_pages_beside_tracked_pages(THRESHOLD_OF_THREE_PAGES, PAGE_SPANS, &["e", "d"], "d", true)]
#[case::metadata_short_of_the_threshold(THRESHOLD_IN_A_PAGE, TWO_HEADERS, &["k2"], "k2", false)]
#[case::metadata_past_the_threshold(THRESHOLD_IN_A_PAGE, TWO_HEADERS, &["k1"], "k1", true)]
#[case::metadata_beside_a_tracked_extent(THRESHOLD_IN_A_PAGE, TWO_HEADERS, &["k1", "k2"], "k2", true)]
fn a_paged_returned_extent_is_tracked_as_libhdf5_tracks_it(
    #[case] threshold: u64,
    #[case] datasets: &[(&str, usize, CLayout)],
    #[case] deletions: &[&str],
    #[case] measured: &str,
    #[case] tracked: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let ours = dir.path().join("ours.h5");
    let theirs = dir.path().join("theirs.h5");
    write_c_paged(&ours, 0, threshold, datasets);
    std::fs::copy(&ours, &theirs).unwrap();
    let extent = object_extent(&ours, measured);

    let (ours_session, ours_reopened, theirs_reopened) =
        paged_tracked_after_deletions(&ours, &theirs, deletions, &extent);

    let expected = if tracked {
        extent.end - extent.start
    } else {
        0
    };
    assert_eq!(
        (ours_session, ours_reopened, theirs_reopened),
        (expected, expected, expected)
    );
    let case = format!(
        "page-threshold-{threshold}-{measured}-{}",
        deletions.join("-")
    );
    super::file_space_oracle::checkpoint(&ours, &case, "rust-close");
    super::file_space_oracle::checkpoint(&theirs, &case, "c-close");
}

#[test]
fn a_dropped_extent_of_whole_pages_that_ends_the_file_is_given_back_for_libhdf5() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ours.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[1; EXTENT_ELEMENTS]);
    b.create_dataset("l").with_i32_data(&[2; 2 * PAGE_ELEMENTS]);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(THRESHOLD_ABOVE_EVERY_EXTENT)
            .with_page_size(FileSpacePageSize::try_from(PAGE_SIZE).unwrap()),
    );
    b.write(&path).unwrap();
    let l = object_extent(&path, "l");
    assert_eq!(
        (l.start % PAGE_SIZE, l.end),
        (0, std::fs::metadata(&path).unwrap().len())
    );

    {
        let file = File::open_rw(&path).unwrap();
        file.root().delete("l").unwrap();
        file.commit().unwrap();
    }

    let c_file = hdf5::File::open(&path).unwrap();
    assert_eq!(
        (
            File::open(&path).unwrap().superblock().eof_address,
            std::fs::metadata(&path).unwrap().len(),
            c_file.dataset("a").unwrap().read_raw::<i32>().unwrap(),
        ),
        (l.start, l.start, vec![1; EXTENT_ELEMENTS])
    );
}

#[test]
fn a_dropped_extent_of_whole_pages_below_an_appended_tail_stays_tracked_for_libhdf5() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ours.h5");
    let page_size = FileSpacePageSize::MIN.get();
    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[1; EXTENT_ELEMENTS]);
    b.create_dataset("l")
        .with_i32_data(&vec![2; 2 * page_size as usize / size_of::<i32>()]);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(THRESHOLD_ABOVE_EVERY_EXTENT)
            .with_page_size(FileSpacePageSize::try_from(page_size).unwrap()),
    );
    b.write(&path).unwrap();
    let l = object_extent(&path, "l");
    assert_eq!(
        (l.start % page_size, l.end),
        (0, std::fs::metadata(&path).unwrap().len())
    );

    {
        let file = File::open_rw(&path).unwrap();
        file.root().delete("l").unwrap();
        file.commit().unwrap();
    }

    let c_file = hdf5::File::open(&path).unwrap();
    assert_eq!(
        (
            covered_free_bytes(
                File::open(&path)
                    .unwrap()
                    .persisted_free_sections()
                    .unwrap()
                    .unwrap()
                    .free_sections(),
                &l
            ),
            std::fs::metadata(&path).unwrap().len() > l.end,
            c_file.dataset("a").unwrap().read_raw::<i32>().unwrap(),
        ),
        (l.end - l.start, true, vec![1; EXTENT_ELEMENTS])
    );
}

#[test]
fn a_paged_returned_extent_beside_free_space_across_a_page_boundary_is_dropped_as_in_libhdf5() {
    let dir = tempfile::tempdir().unwrap();
    let ours = dir.path().join("ours.h5");
    let theirs = dir.path().join("theirs.h5");
    write_c_paged(
        &ours,
        0,
        THRESHOLD_ACROSS_PAGES,
        &[
            ("f0", 768, CLayout::Contiguous),
            ("f1", 256, CLayout::Contiguous),
            ("g", 300, CLayout::Contiguous),
            ("l", 1500, CLayout::Contiguous),
        ],
    );
    std::fs::copy(&ours, &theirs).unwrap();
    let (f1, g) = (object_extent(&ours, "f1"), object_extent(&ours, "g"));
    assert_eq!((f1.end, f1.end % PAGE_SIZE), (g.start, 0));

    let tracked = [&f1, &g].map(|extent| {
        let dir = tempfile::tempdir().unwrap();
        let (ours_copy, theirs_copy) = (dir.path().join("ours.h5"), dir.path().join("theirs.h5"));
        std::fs::copy(&ours, &ours_copy).unwrap();
        std::fs::copy(&theirs, &theirs_copy).unwrap();
        paged_tracked_after_deletions(&ours_copy, &theirs_copy, &["g", "f1"], extent)
    });

    let g_len = g.end - g.start;
    assert_eq!(tracked, [(0, 0, 0), (g_len, g_len, g_len)]);
}

#[rstest]
#[case::a_libhdf5_tail_after_a_deletion(
    &[("a", 900, CLayout::Contiguous), ("k", 50, CLayout::Compact)],
    delete_k,
    unlink_k,
    "a"
)]
#[case::an_editor_tail_after_a_creation(&[("a", 900, CLayout::Contiguous)], create_n, c_create_n, "n")]
fn a_page_tail_short_of_the_threshold_stays_tracked_as_libhdf5_tracks_it(
    #[case] datasets: &[(&str, usize, CLayout)],
    #[case] ours_edit: fn(&File),
    #[case] theirs_edit: fn(&hdf5::File),
    #[case] tail_after: &str,
) {
    let dir = tempfile::tempdir().unwrap();
    let ours = dir.path().join("ours.h5");
    let theirs = dir.path().join("theirs.h5");
    write_c_paged(&ours, 0, THRESHOLD_IN_A_PAGE, datasets);
    std::fs::copy(&ours, &theirs).unwrap();

    let (ours_tail, ours_session) = {
        let file = File::open_rw(&ours).unwrap();
        ours_edit(&file);
        file.commit().unwrap();
        let tail = page_tail(&file_space::contiguous_extent(&file, tail_after));
        let session = test_util_hdf5::covered_free_bytes!(
            &file.reusable_free_sections().unwrap(),
            &tail,
            FreeSectionCoordinates::Absolute
        );
        (tail, session)
    };
    let ours_reopened = covered_free_bytes(
        File::open(&ours)
            .unwrap()
            .persisted_free_sections()
            .unwrap()
            .unwrap()
            .free_sections(),
        &ours_tail,
    );
    let theirs_tail = {
        let file = hdf5::File::open_rw(&theirs).unwrap();
        theirs_edit(&file);
        let tail = page_tail(&file_space::libhdf5_contiguous_extent(
            &file.dataset(tail_after).unwrap(),
        ));
        file.close().unwrap();
        tail
    };
    let theirs_reopened = covered_free_bytes(
        File::open(&theirs)
            .unwrap()
            .persisted_free_sections()
            .unwrap()
            .unwrap()
            .free_sections(),
        &theirs_tail,
    );

    let len = ours_tail.end - ours_tail.start;
    assert!(len < THRESHOLD_IN_A_PAGE, "{ours_tail:?}");
    assert_eq!(
        (
            ours_session,
            ours_reopened,
            theirs_tail.end - theirs_tail.start,
            theirs_reopened
        ),
        (len, len, len, len)
    );
}

#[test]
fn a_paged_persisting_edit_that_tracks_no_free_space_records_the_end_of_allocation_for_libhdf5() {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base.h5");
    let path = dir.path().join("ours.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("a").with_i32_data(&[1; PAGE_ELEMENTS]);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(THRESHOLD_ABOVE_EVERY_EXTENT),
    );
    b.write(&base).unwrap();

    let filled = (0..PAGE_SIZE).find(|&len| {
        std::fs::copy(&base, &path).unwrap();
        {
            let file = File::open_rw(&path).unwrap();
            file.root()
                .set_attr("fill", AttrValue::U8Array(vec![0; len as usize]))
                .unwrap();
            file.commit().unwrap();
        }
        File::open(&path)
            .unwrap()
            .persisted_free_sections()
            .unwrap()
            .unwrap()
            .free_sections()
            .section_count()
            == 0
    });

    let (eoa_pre_fsm, manager_addrs, free, eof) = {
        let file = File::open(&path).unwrap();
        let info = file.file_space_info().unwrap().unwrap();
        (
            info.eoa_pre_fsm,
            info.manager_addrs.clone(),
            file.persisted_free_sections().unwrap().unwrap(),
            file.superblock().eof_address,
        )
    };
    let len = std::fs::metadata(&path).unwrap().len();
    let c_file = hdf5::File::open(&path).unwrap();
    assert!(filled.is_some(), "no attribute fills the free space");
    assert_eq!(
        (
            eoa_pre_fsm,
            eof,
            manager_addrs,
            free.free_sections().section_count(),
            c_file.dataset("a").unwrap().read_raw::<i32>().unwrap(),
            c_file.free_space(),
        ),
        (
            Some(hdf5_pure::DecodedStoredAddress::Defined(
                hdf5_pure::StoredAddress::new(len)
            )),
            len,
            vec![hdf5_pure::DecodedStoredAddress::Undefined; NUM_FILE_FSM_MANAGERS],
            0,
            vec![1; PAGE_ELEMENTS],
            0
        )
    );
}

fn paged_tracked_after_deletions(
    ours: &Path,
    theirs: &Path,
    deletions: &[&str],
    extent: &Range<u64>,
) -> (u64, u64, u64) {
    let ours_session = {
        let file = File::open_rw(ours).unwrap();
        for &object_path in deletions {
            file.root().delete(object_path).unwrap();
        }
        file.commit().unwrap();
        test_util_hdf5::covered_free_bytes!(
            &file.reusable_free_sections().unwrap(),
            extent,
            FreeSectionCoordinates::Absolute
        )
    };
    {
        let file = hdf5::File::open_rw(theirs).unwrap();
        for &object_path in deletions {
            file.unlink(object_path).unwrap();
        }
        file.close().unwrap();
    }
    let reopened = |path: &Path| {
        covered_free_bytes(
            File::open(path)
                .unwrap()
                .persisted_free_sections()
                .unwrap()
                .unwrap()
                .free_sections(),
            extent,
        )
    };
    (ours_session, reopened(ours), reopened(theirs))
}

fn write_c_paged(path: &Path, userblock: u64, threshold: u64, datasets: &[(&str, usize, CLayout)]) {
    let file = hdf5::FileBuilder::new()
        .with_fapl(|fapl| fapl.libver_v110())
        .with_fcpl(|fcpl| {
            fcpl.userblock(userblock)
                .file_space_strategy(page(threshold))
                .file_space_page_size(PAGE_SIZE)
        })
        .create(path)
        .unwrap();
    for (value, &(name, len, layout)) in (1..).zip(datasets) {
        file.new_dataset::<i32>()
            .layout(layout)
            .shape((len,))
            .create(name)
            .unwrap()
            .write(&vec![value; len])
            .unwrap();
    }
    file.close().unwrap();
}

fn object_extent(file_path: &Path, object_path: &str) -> Range<u64> {
    let file = File::open(file_path).unwrap();
    match file.dataset(object_path).unwrap().layout().unwrap() {
        Layout::Contiguous {
            address: Some(address),
            size,
        } => address..address + size,
        Layout::Compact { .. } => header_extent(file_path, &file, object_path),
        layout => panic!("expected contiguous or compact storage, got {layout:?}"),
    }
}

fn header_extent(path: &Path, file: &File, link_name: &str) -> Range<u64> {
    let bytes = std::fs::read(path).unwrap();
    let superblock = file.superblock();
    let root = ObjectHeader::parse_with_widths(
        &bytes,
        AccessMode::ReadOnly,
        usize::try_from(
            superblock
                .base_address
                .absolute(superblock.root_group_address)
                .unwrap(),
        )
        .unwrap(),
        FormatWidths {
            offsets: superblock.offset_size,
            lengths: superblock.length_size,
        },
    )
    .unwrap();
    let address = root
        .messages
        .iter()
        .filter(|message| message.msg_type == MessageType::LINK)
        .map(|message| {
            LinkMessage::parse_with_offset_width(&message.data, superblock.offset_size).unwrap()
        })
        .find(|link| link.name == link_name)
        .map(|link| match link.link_target {
            LinkTarget::Hard {
                object_header_address,
            } => object_header_address.get(),
            target => panic!("expected a hard link, got {target:?}"),
        })
        .unwrap();
    let prefix = ObjectHeaderPrefix::parse(&bytes[usize::try_from(address).unwrap()..]).unwrap();
    address..address + prefix.len as u64 + prefix.chunk0_size + CHECKSUM_LEN
}

fn page_tail(extent: &Range<u64>) -> Range<u64> {
    extent.end..extent.end.next_multiple_of(PAGE_SIZE)
}

fn delete_k(file: &File) {
    file.root().delete("k").unwrap();
}

fn unlink_k(file: &hdf5::File) {
    file.unlink("k").unwrap();
}

fn create_n(file: &File) {
    file.root()
        .create_dataset("n", |b| {
            b.with_i32_data(&[9; TAIL_ELEMENTS]);
        })
        .unwrap();
}

fn c_create_n(file: &hdf5::File) {
    file.new_dataset::<i32>()
        .shape((TAIL_ELEMENTS,))
        .create("n")
        .unwrap()
        .write(&[9; TAIL_ELEMENTS])
        .unwrap();
}

fn assert_section_map(sections: &FreeSections, expected: &[(u64, u64)], boundary: u64) {
    let mut expected = expected.to_vec();
    expected.sort_unstable();
    assert_eq!(sections.section_count(), expected.len());
    assert_eq!(
        sections.free_bytes(),
        expected.iter().map(|&(_, length)| length).sum::<u64>()
    );
    let mut previous_end = 0;
    for (section, (address, length)) in sections.sections().iter().zip(expected) {
        assert!(length > 0);
        assert!(address >= previous_end);
        previous_end = address.checked_add(length).unwrap();
        assert!(previous_end <= boundary);
        assert_eq!(section.coordinates(), FreeSectionCoordinates::Stored);
        assert_eq!(section.address(), address);
        assert_eq!(section.length().get(), length);
    }
}

fn covered_free_bytes(sections: &FreeSections, extent: &Range<u64>) -> u64 {
    sections
        .sections()
        .iter()
        .map(|section| {
            assert_eq!(section.coordinates(), FreeSectionCoordinates::Stored);
            section
                .end_address()
                .min(extent.end)
                .saturating_sub(section.address().max(extent.start))
        })
        .sum()
}

fn assert_supported_free_sections(path: &Path) -> PersistedFreeSections {
    let buffered = File::open(path)
        .unwrap()
        .persisted_free_sections()
        .unwrap()
        .unwrap();
    let streamed = File::open_streaming(path)
        .unwrap()
        .persisted_free_sections()
        .unwrap()
        .unwrap();
    assert_eq!(buffered, streamed);
    let all = buffered.free_sections();
    assert!(
        all.sections()
            .iter()
            .all(|section| section.coordinates() == FreeSectionCoordinates::Stored)
    );
    assert_eq!(
        buffered
            .managers()
            .iter()
            .map(|manager| manager.free_sections().section_count())
            .sum::<usize>(),
        all.section_count()
    );
    assert_eq!(
        buffered
            .managers()
            .iter()
            .map(|manager| manager.free_sections().free_bytes())
            .sum::<u64>(),
        all.free_bytes()
    );
    buffered
}

fn page(threshold: u64) -> CStrategy {
    CStrategy::FreeSpaceManager {
        paged: true,
        persist: true,
        threshold,
    }
}

fn fsm_aggr(persist: bool, threshold: u64) -> CStrategy {
    CStrategy::FreeSpaceManager {
        paged: false,
        persist,
        threshold,
    }
}

/// The offset of the "Address of Serialized Section List" field in a manager header with 8-byte
/// addresses and lengths.
const SECTION_LIST_ADDR_AT: usize =
    test_util::free_space::SIGNATURE.len() + 1 + 1 + 4 * 8 + 4 * 2 + 8;
/// The offset of the "Size of Serialized Section List Used" field in a manager header with 8-byte
/// addresses and lengths.
const SECTION_LIST_USED_AT: usize = SECTION_LIST_ADDR_AT + 8;
/// The offset of the "Allocated Size of Serialized Section List" field in a manager header with
/// 8-byte addresses and lengths.
const SECTION_LIST_ALLOCATED_AT: usize = SECTION_LIST_USED_AT + 8;
/// The offset of the "Free-space Manager Header Address" field in a section list.
const SECTION_LIST_HEADER_ADDR_AT: usize = test_util::free_space::SECTIONS_SIGNATURE.len() + 1;

const DELETED_LEN: u64 = file_space::STRATEGY_DELETED_LEN;
const THRESHOLD_ELEMENTS: usize = 1024;
const THRESHOLD: u64 = (THRESHOLD_ELEMENTS * size_of::<i32>()) as u64;
const SUB_THRESHOLD_ELEMENTS: usize = 256;
const SUB_THRESHOLD_LEN: u64 = (SUB_THRESHOLD_ELEMENTS * size_of::<i32>()) as u64;
const THRESHOLD_ABOVE_EVERY_EXTENT: u64 = 1 << 20;
const PAGE_SIZE: u64 = 4096;
const PAGE_ELEMENTS: usize = PAGE_SIZE as usize / size_of::<i32>();
const CHECKSUM_LEN: u64 = 4;
const EXTENT_ELEMENTS: usize = 100;
const EXTENT_LEN: u64 = (EXTENT_ELEMENTS * size_of::<i32>()) as u64;
const THRESHOLD_IN_A_PAGE: u64 = 1000;
const THRESHOLD_ACROSS_PAGES: u64 = 1100;
const TAIL_ELEMENTS: usize = 950;
const FOUR_EXTENTS: &[(&str, usize, CLayout)] = &[
    ("a", EXTENT_ELEMENTS, CLayout::Contiguous),
    ("b", EXTENT_ELEMENTS, CLayout::Contiguous),
    ("c", EXTENT_ELEMENTS, CLayout::Contiguous),
    ("e", EXTENT_ELEMENTS, CLayout::Contiguous),
];
const ABOVE_A_TRACKED_EXTENT: &[(&str, usize, CLayout)] = &[
    ("a", EXTENT_ELEMENTS, CLayout::Contiguous),
    ("b", 300, CLayout::Contiguous),
    ("c", EXTENT_ELEMENTS, CLayout::Contiguous),
    ("e", EXTENT_ELEMENTS, CLayout::Contiguous),
];
const THRESHOLD_OF_THREE_PAGES: u64 = 3 * PAGE_SIZE;
const PAGE_SPANS: &[(&str, usize, CLayout)] = &[
    ("a", EXTENT_ELEMENTS, CLayout::Contiguous),
    ("d", 2 * PAGE_ELEMENTS, CLayout::Contiguous),
    ("e", 3 * PAGE_ELEMENTS, CLayout::Contiguous),
    ("f", PAGE_ELEMENTS, CLayout::Contiguous),
];
const TWO_HEADERS: &[(&str, usize, CLayout)] = &[
    ("a", EXTENT_ELEMENTS, CLayout::Contiguous),
    ("k1", 300, CLayout::Compact),
    ("k2", 50, CLayout::Compact),
    ("e", EXTENT_ELEMENTS, CLayout::Contiguous),
];
