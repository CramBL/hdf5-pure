//! Commit rollback, append publication, buffering, and synchronization.

use std::io::ErrorKind;

use hdf5_pure_core::__private::PhysicalAddress;
use rstest::rstest;

use super::common;
use crate::DeflateLevel;
use crate::FileSpaceSettings;
use crate::Layout;
use crate::PersistedFreeSpacePolicy;
use crate::SuperblockStatusFlags;
use crate::SuperblockVersion;
use crate::UserblockSize;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::dataspace::MaxExtent;
use crate::edit::AppendBuilder;
use crate::edit::AppendTarget;
use crate::edit::MemoryStrategy;
use crate::edit::SWMR_WRITE_FLAGS;
use crate::edit::SyncPolicy;
use crate::edit::WRITE_GATHER_BYTES;
use crate::edit::WriteEngine;
use crate::error::{Error, FormatError};
use crate::file_lock::FileLocking;
use crate::file_space_info::FileSpacePageSize;
use crate::file_space_info::FileSpaceStrategy;
use crate::image::HandleImage;
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;
use crate::reader::File;
use crate::source::MetadataCacheConfig;
use crate::superblock::LocatedSuperblock;
use crate::type_builders::DatasetBuilder;
use crate::writer::FileBuilder;

#[rstest]
#[case::before_publication(false)]
#[case::after_publication(true)]
fn failed_mixed_overwrites_restore_old_values_before_publication_and_keep_new_values_after(
    #[case] published: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("mixed.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("fixed").with_i32_data(&[1, 2, 3]);
    builder
        .create_dataset("strings")
        .with_vlen_strings(&["old", "values"]);
    builder.create_dataset("refs").with_path_references(&[""]);
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(0),
    );
    builder.write(&path).unwrap();
    let references = {
        let file = File::open(&path).unwrap();
        let Layout::Contiguous {
            address: Some(address),
            size,
        } = file.dataset("refs").unwrap().layout().unwrap()
        else {
            panic!("expected allocated contiguous reference storage");
        };
        address..address + size
    };
    let mut engine = if published {
        WriteEngine::open_torn_writes(&path, references).unwrap()
    } else {
        WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap()
    };
    let original_heaps = engine.vl_overwrite_heaps.clone();
    if !published {
        let mut superblock = engine.context.superblock().clone();
        superblock.superblock_extension_address = Some(StoredAddress::ZERO);
        let proposed = engine.context.propose(superblock).unwrap();
        engine.context.adopt(proposed).unwrap();
    }
    let mut strings = DatasetBuilder::new("");
    strings.with_vlen_strings(&["replacement", "strings"]);
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("strings"), strings)
        .unwrap();
    let mut fixed = DatasetBuilder::new("");
    fixed.with_i32_data(&[11, 22, 33]);
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("fixed"), fixed)
        .unwrap();
    let error = engine.commit().unwrap_err();
    if published {
        let Error::Io(error) = error else {
            panic!("expected the injected reference write error, got {error:?}");
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(error.to_string(), "simulated device write failure");
        assert!(!engine.vl_overwrite_heaps.is_empty());
    } else {
        let Error::EditUnsupported(reason) = error else {
            panic!("expected the invalid extension error, got {error:?}");
        };
        assert_eq!(reason, "an object does not use a version 2 object header");
        assert_eq!(engine.vl_overwrite_heaps, original_heaps);
    }
    assert_eq!(engine.publication.attempted(), published);
    assert!(!engine.has_staged_edits());
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("fixed").unwrap().read_i32().unwrap(),
        if published { [11, 22, 33] } else { [1, 2, 3] }
    );
    assert_eq!(
        file.dataset("strings").unwrap().read_string().unwrap(),
        if published {
            ["replacement", "strings"]
        } else {
            ["old", "values"]
        }
    );
}

#[rstest]
#[case::raise_before_write(0, false)]
#[case::raise_partial_write(17, false)]
#[case::raise_completed_write(usize::MAX, false)]
#[case::clear_before_write(0, true)]
#[case::clear_partial_write(17, true)]
#[case::clear_completed_write(usize::MAX, true)]
fn failed_flag_publication_preserves_the_confirmed_context(
    #[case] prefix: usize,
    #[case] clearing: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(false, true)] gathered: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("flags.h5");
    common::build_appendable(&path, 4, 4);
    let original = std::fs::read(&path).unwrap();
    let mut engine = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        backing,
        PersistedFreeSpacePolicy::Tolerant,
    )
    .unwrap();
    if !gathered {
        engine
            .set_write_buffering(WriteBuffering::Unbuffered)
            .unwrap();
    }
    if clearing {
        engine
            .set_consistency_flags(SuperblockStatusFlags::write_marked())
            .unwrap();
    }
    let before = engine.context.superblock().clone();
    test_util::disk_log::fail_write_after(0, prefix);
    let error = if clearing {
        engine.release_status_flags()
    } else {
        engine.set_consistency_flags(SuperblockStatusFlags::write_marked())
    }
    .unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected injected disk write failure, got {error:?}");
    };
    assert_eq!(error.kind(), std::io::ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    assert_eq!(engine.context.superblock(), &before);
    if clearing {
        engine
            .append_inplace_i32_phased("d", &[5, 6, 7, 8], 4)
            .unwrap();
        assert!(engine.context.superblock().eof_address > before.eof_address);
        assert_eq!(
            engine.context.superblock().consistency_flags(),
            SuperblockStatusFlags::write_marked()
        );
        engine.force_sync().unwrap();
        engine.release_status_flags().unwrap();
        assert_eq!(
            engine.context.superblock().consistency_flags(),
            SuperblockStatusFlags::new()
        );
        drop(engine);
        let file = crate::reader::File::open(&path).unwrap();
        assert_eq!(
            file.superblock().consistency_flags(),
            SuperblockStatusFlags::new()
        );
        assert_eq!(
            file.dataset("d").unwrap().read_i32().unwrap(),
            vec![0, 1, 2, 3, 5, 6, 7, 8]
        );
    } else {
        assert_eq!(
            engine.held_status_flags,
            SuperblockStatusFlags::write_marked()
        );
        engine
            .set_consistency_flags(SuperblockStatusFlags::new())
            .unwrap();
        assert_eq!(engine.context.superblock(), &before);
        engine.force_sync().unwrap();
        drop(engine);
        assert_eq!(std::fs::read(&path).unwrap(), original);
    }
}

#[rstest]
#[case::legacy(
    "c/1.8/v1_superblock.h5",
    SuperblockVersion::One,
    MemoryStrategy::Mirrored,
    0
)]
#[case::modern(
    "c/1.8/v2_superblock.h5",
    SuperblockVersion::Two,
    MemoryStrategy::Mirrored,
    0
)]
#[case::modern_bounded(
    "c/1.8/v2_superblock.h5",
    SuperblockVersion::Two,
    MemoryStrategy::Bounded,
    0
)]
#[case::modern_userblock(
    "c/1.8/v2_superblock.h5",
    SuperblockVersion::Two,
    MemoryStrategy::Mirrored,
    512
)]
fn publication_encodes_the_physical_root_once(
    #[case] fixture: &str,
    #[case] version: SuperblockVersion,
    #[case] backing: MemoryStrategy,
    #[case] userblock: usize,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = test_util::copy_data(fixture, directory.path());
    if userblock != 0 {
        test_util_hdf5::userblock::prepend(&path, &vec![0xA5; userblock]);
    }
    let mut engine = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        backing,
        PersistedFreeSpacePolicy::Tolerant,
    )
    .unwrap();
    assert_eq!(engine.context.superblock().version(), version);
    let stored_root = engine.context.superblock().root_group_address;
    let root = engine.context.addresses().resolve(stored_root).unwrap();
    assert_eq!(root.get(), userblock as u64 + stored_root.get());
    let eof = engine.image.len();
    engine.publish_nonpersisting(root, None).unwrap();
    assert_eq!(engine.context.superblock().root_group_address, stored_root);
    assert_eq!(engine.context.superblock().eof_address, eof);
    drop(engine);
    let file = crate::reader::File::open(&path).unwrap();
    assert_eq!(file.superblock().root_group_address, stored_root);
    assert_eq!(file.superblock().eof_address, eof);
    assert_eq!(
        file.dataset("values").unwrap().read_f64().unwrap(),
        vec![1.5, 2.5, 3.5, 4.5]
    );
    drop(file);
    assert_eq!(
        &std::fs::read(&path).unwrap()[..userblock],
        &vec![0xA5; userblock]
    );
}

#[rstest]
#[case::before_base(0)]
#[case::last_userblock_byte(511)]
fn publication_rejects_a_physical_root_below_base(#[case] root: u64) {
    let directory = tempfile::tempdir().unwrap();
    let path = test_util::copy_data("c/1.8/v2_superblock.h5", directory.path());
    test_util_hdf5::userblock::prepend(&path, &[0xA5; 512]);
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let before = engine.context.superblock().clone();
    let bytes = engine.page_publication_snapshot().unwrap();
    let error = engine
        .publish_nonpersisting(PhysicalAddress::new(root), None)
        .unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a root below the address base");
    };
    assert_eq!(
        error,
        FormatError::AddressBelowBase {
            address: root,
            base: 512
        }
    );
    assert_eq!(engine.context.superblock(), &before);
    assert_eq!(engine.page_publication_snapshot().unwrap(), bytes);
}

#[rstest]
fn context_tracks_root_publication_and_ordered_append(
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] sync: SyncPolicy,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("published_context.h5");
    common::build_appendable(&path, 4, 4);
    let mut engine = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        backing,
        PersistedFreeSpacePolicy::Tolerant,
    )
    .unwrap();
    engine.set_sync_policy(sync);
    let old_root = engine.context.superblock().root_group_address;
    let mut added = DatasetBuilder::new("added");
    added.with_u8_data(&[9]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("added"), added)
        .unwrap();
    engine.commit().unwrap();
    let root = engine.context.superblock().root_group_address;
    assert_ne!(root, old_root);
    let bytes = engine.page_publication_snapshot().unwrap();
    let published = crate::superblock::read(bytes.as_slice())
        .unwrap()
        .superblock;
    assert_eq!(
        engine.context.superblock().root_group_address,
        published.root_group_address
    );
    assert_eq!(
        engine.context.superblock().eof_address,
        published.eof_address
    );
    engine
        .append_inplace_i32_phased("d", &[5, 6, 7, 8], 4)
        .unwrap();
    assert_eq!(engine.context.superblock().root_group_address, root);
    assert_eq!(engine.context.superblock().eof_address, engine.image.len());
    let bytes = engine.page_publication_snapshot().unwrap();
    let published = crate::superblock::read(bytes.as_slice())
        .unwrap()
        .superblock;
    assert_eq!(
        engine.context.superblock().root_group_address,
        published.root_group_address
    );
    assert_eq!(
        engine.context.superblock().eof_address,
        published.eof_address
    );
    drop(engine);
    let file = crate::reader::File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        vec![0, 1, 2, 3, 5, 6, 7, 8]
    );
}

#[test]
fn append_inplace_crash_consistency_partial_tail_prefix() {
    use crate::reader::File as PureFile;
    use crate::writer::FileBuilder;

    let build = |path: &std::path::Path, n: i32, chunk: u64, deflate: bool| {
        let data: Vec<i32> = (0..n).collect();
        let mut b = FileBuilder::new();
        let d = b
            .create_dataset("d")
            .with_i32_data(&data)
            .with_shape(&[n as u64])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[chunk]);
        if deflate {
            d.with_deflate(DeflateLevel::Six);
        }
        b.write(path).unwrap();
    };

    for (n, chunk, add, deflate) in [
        (6i32, 4u64, 5i32, false),
        (9, 2, 6, false),
        (6, 4, 5, true),
        (9, 4, 3, true),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.h5");
        build(&base, n, chunk, deflate);

        for max_phase in 1u8..=4 {
            let p = dir
                .path()
                .join(format!("crash_{n}_{chunk}_{deflate}_{max_phase}.h5"));
            std::fs::copy(&base, &p).unwrap();
            {
                let mut s = WriteEngine::open_with_locking(&p, FileLocking::Enabled).unwrap();
                s.append_inplace_i32_phased("d", &(n..n + add).collect::<Vec<_>>(), max_phase)
                    .unwrap();
            }
            let expected_len = if max_phase == 4 { n + add } else { n };
            let f = PureFile::from_bytes(std::fs::read(&p).unwrap()).unwrap();
            assert_eq!(
                f.dataset("d").unwrap().read_i32().unwrap(),
                (0..expected_len).collect::<Vec<_>>(),
                "inconsistent view after crash at phase {max_phase} (n={n}, chunk={chunk}, \
                 deflate={deflate})"
            );
        }
    }
}

#[test]
fn append_inplace_crash_consistency_filtered_partial_last_chunk() {
    use crate::reader::File as PureFile;
    use crate::writer::FileBuilder;

    let build = |path: &std::path::Path, n: i32, chunk: u64| {
        let data: Vec<i32> = (0..n).collect();
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&data)
            .with_shape(&[n as u64])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[chunk])
            .with_shuffle()
            .with_deflate(DeflateLevel::Four);
        b.write(path).unwrap();
    };
    for (n, chunk, add) in [(8i32, 4u64, 2i32), (8, 4, 9), (0, 4, 3)] {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.h5");
        build(&base, n, chunk);

        for max_phase in 1u8..=4 {
            let p = dir
                .path()
                .join(format!("crash_f_{n}_{chunk}_{max_phase}.h5"));
            std::fs::copy(&base, &p).unwrap();
            {
                let mut s = WriteEngine::open_with_locking(&p, FileLocking::Enabled).unwrap();
                s.append_inplace_i32_phased("d", &(n..n + add).collect::<Vec<_>>(), max_phase)
                    .unwrap();
            }
            let expected_len = if max_phase == 4 { n + add } else { n };
            let f = PureFile::from_bytes(std::fs::read(&p).unwrap()).unwrap();
            assert_eq!(
                f.dataset("d").unwrap().read_i32().unwrap(),
                (0..expected_len).collect::<Vec<_>>(),
                "inconsistent view after crash at phase {max_phase} (n={n}, chunk={chunk}, \
                 add={add})"
            );
        }
    }
}

#[test]
fn append_inplace_crash_consistency_across_ea_boundaries() {
    use crate::reader::File as PureFile;

    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base.h5");
    let (n, target) = (50i32, 250i32);
    build_unit_chunked(&base, n);

    for max_phase in 1u8..=4 {
        let p = dir.path().join(format!("crash_ea_{max_phase}.h5"));
        append_stopped_at(&base, &p, n..target, max_phase);
        let expected_len = if max_phase == 4 { target } else { n };
        let f = PureFile::from_bytes(std::fs::read(&p).unwrap()).unwrap();
        assert_eq!(
            f.dataset("d").unwrap().read_i32().unwrap(),
            (0..expected_len).collect::<Vec<_>>(),
            "inconsistent view after crash at phase {max_phase}"
        );
    }
}

#[test]
fn failed_commit_restores_heap_collection_ownership_records() {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("failed_commit_vl_provenance.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&["seed-one", "seed-two"]);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(0),
    );
    b.write(&path).unwrap();

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    s.stage_dataset_write(&ObjectPathBuf::parse("/labels"), {
        let mut db = crate::type_builders::DatasetBuilder::new("");
        db.with_vlen_strings(&["round-one-aaaa", "round-one-bbbb"]);
        db
    })
    .unwrap();
    s.commit().unwrap();
    let recorded = s.vl_overwrite_heaps.clone();
    assert!(!recorded.is_empty(), "the good round recorded nothing");
    let good_ext = s.context.superblock().superblock_extension_address;
    let mut superblock = s.context.superblock().clone();
    superblock.superblock_extension_address = Some(StoredAddress::ZERO);
    let proposed = s.context.propose(superblock).unwrap();
    s.context.adopt(proposed).unwrap();
    s.stage_dataset_write(&ObjectPathBuf::parse("/labels"), {
        let mut db = crate::type_builders::DatasetBuilder::new("");
        db.with_vlen_strings(&["round-two-aaaa", "round-two-bbbb"]);
        db
    })
    .unwrap();
    assert!(
        s.commit().is_err(),
        "a commit with an unreadable extension must fail"
    );

    assert_eq!(
        s.vl_overwrite_heaps, recorded,
        "a failed commit must not leave a record naming space it gave back"
    );
    let mut superblock = s.context.superblock().clone();
    superblock.superblock_extension_address = good_ext;
    let proposed = s.context.propose(superblock).unwrap();
    s.context.adopt(proposed).unwrap();
    s.stage_dataset_write(&ObjectPathBuf::parse("/labels"), {
        let mut db = crate::type_builders::DatasetBuilder::new("");
        db.with_vlen_strings(&["round-three-a", "round-three-b"]);
        db
    })
    .unwrap();
    s.commit()
        .expect("the session is usable after a failed commit");
    drop(s);

    let f = crate::reader::File::open(&path).unwrap();
    assert_eq!(
        f.dataset("labels").unwrap().read_string().unwrap(),
        vec!["round-three-a".to_string(), "round-three-b".to_string()]
    );
}

#[test]
fn a_failed_commit_that_reused_free_space_leaves_the_file_intact() {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reuse_failed_commit.h5");
    let victim: Vec<f64> = (0..4096).map(|i| (i % 13) as f64).collect();
    let ceiling: Vec<i32> = (0..500).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.create_dataset("victim")
        .with_f64_data(&victim)
        .with_shape(&[4096])
        .with_chunks(&[512]);
    b.create_dataset("ceiling")
        .with_i32_data(&ceiling)
        .with_shape(&[500]);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    b.write(&path).unwrap();

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    s.delete(&ObjectPathBuf::parse("/victim")).unwrap();
    s.commit().unwrap();
    let free_before = s.reusable_free_sections().unwrap();
    let len_before = std::fs::metadata(&path).unwrap().len();
    assert!(
        free_before
            .sections()
            .iter()
            .any(|section| section.length().get() > 4096 * 8 / 2),
        "the deleted chunked dataset left a hole worth reusing: {free_before:?}"
    );
    let mut superblock = s.context.superblock().clone();
    superblock.superblock_extension_address = Some(StoredAddress::ZERO);
    let proposed = s.context.propose(superblock).unwrap();
    s.context.adopt(proposed).unwrap();
    let mut db = crate::type_builders::DatasetBuilder::new("fresh");
    db.with_f64_data(&vec![7.5f64; 4096])
        .with_shape(&[4096])
        .with_chunks(&[512]);
    s.stage_created_dataset(&ObjectPathBuf::parse("/fresh"), db)
        .unwrap();
    assert!(
        s.commit().is_err(),
        "a commit with an unreadable extension must fail"
    );
    assert_eq!(
        s.reusable_free_sections().unwrap(),
        free_before,
        "the failed commit gives back the region it drew from"
    );
    drop(s);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), len_before);
    let f = crate::reader::File::open(&path).unwrap();
    assert_eq!(f.file_size(), len_before);
    assert_eq!(
        f.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert_eq!(f.dataset("ceiling").unwrap().read_i32().unwrap(), ceiling);
    assert!(f.dataset("victim").is_err());
    assert!(f.dataset("fresh").is_err());
}

#[test]
fn a_failed_commit_puts_back_the_value_it_overwrote() {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("inplace_partial.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("nums").with_i32_data(&[1, 2, 3]);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(0),
    );
    b.write(&path).unwrap();

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();

    let good_ext = s.context.superblock().superblock_extension_address;
    let mut superblock = s.context.superblock().clone();
    superblock.superblock_extension_address = Some(StoredAddress::ZERO);
    let proposed = s.context.propose(superblock).unwrap();
    s.context.adopt(proposed).unwrap();
    s.stage_dataset_write(&ObjectPathBuf::parse("/nums"), {
        let mut db = crate::type_builders::DatasetBuilder::new("");
        db.with_i32_data(&[9, 9, 9]);
        db
    })
    .unwrap();
    s.stage_created_dataset(&ObjectPathBuf::parse("/extra"), {
        let mut db = crate::type_builders::DatasetBuilder::new("");
        db.with_i32_data(&[42]);
        db
    })
    .unwrap();
    let refused = s.commit();
    assert!(
        matches!(refused, Err(Error::EditUnsupported(_))),
        "the refusal, not a rollback failure: {refused:?}"
    );
    let mut superblock = s.context.superblock().clone();
    superblock.superblock_extension_address = good_ext;
    let proposed = s.context.propose(superblock).unwrap();
    s.context.adopt(proposed).unwrap();
    s.stage_created_dataset(&ObjectPathBuf::parse("/later"), {
        let mut db = crate::type_builders::DatasetBuilder::new("");
        db.with_i32_data(&[7]);
        db
    })
    .unwrap();
    s.commit()
        .expect("the session is usable after a failed commit");
    drop(s);

    let f = crate::reader::File::open(&path).unwrap();
    assert_eq!(
        f.dataset("nums").unwrap().read_i32().unwrap(),
        vec![1, 2, 3],
        "the refused batch's value overwrite is in the file"
    );
    assert!(
        f.dataset("extra").is_err(),
        "/extra was correctly discarded"
    );
    assert_eq!(f.dataset("later").unwrap().read_i32().unwrap(), vec![7]);
}

#[test]
fn failed_fast_path_commit_reports_partial_rollback() {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("inplace_fast_path_torn.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("kept").with_i32_data(&[1, 2, 3]);
    b.create_dataset("victim").with_i32_data(&[4, 5, 6]);
    b.write(&path).unwrap();
    let victim_block = {
        let f = crate::reader::File::open(&path).unwrap();
        match f.dataset("victim").unwrap().layout().unwrap() {
            crate::Layout::Contiguous {
                address: Some(a),
                size,
            } => a..a + size,
            other => panic!("expected a contiguous victim: {other:?}"),
        }
    };

    let mut s = WriteEngine::open_torn_writes(&path, victim_block).unwrap();
    for (name, data) in [("/kept", [9, 9, 9]), ("/victim", [8, 8, 8])] {
        s.stage_dataset_write(&ObjectPathBuf::parse(name), {
            let mut db = crate::type_builders::DatasetBuilder::new("");
            db.with_i32_data(&data);
            db
        })
        .unwrap();
    }
    let refused = s.commit();
    assert!(
        matches!(refused, Err(Error::CommitPartiallyApplied { .. })),
        "a rollback that could not run is not an ordinary refusal: {refused:?}"
    );
    drop(s);

    let f = crate::reader::File::open(&path).unwrap();
    assert_eq!(
        f.dataset("kept").unwrap().read_i32().unwrap(),
        vec![1, 2, 3],
        "the overwrite the rollback could reach is put back"
    );
    assert_eq!(
        f.dataset("victim").unwrap().read_i32().unwrap(),
        vec![8, 8, 8],
        "the overwrite it could not reach is the value the error is about"
    );
}

#[test]
fn a_commit_that_fails_past_its_repoint_keeps_the_value_it_published() {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("inplace_after_repoint.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("nums").with_i32_data(&[1, 2, 3]);
    b.create_dataset("refs").with_path_references(&[""]);
    b.write(&path).unwrap();

    let refs_block = {
        let f = crate::reader::File::open(&path).unwrap();
        match f.dataset("refs").unwrap().layout().unwrap() {
            crate::Layout::Contiguous {
                address: Some(a),
                size,
            } => a..a + size,
            other => panic!("expected a contiguous reference dataset: {other:?}"),
        }
    };

    let mut s = WriteEngine::open_torn_writes(&path, refs_block).unwrap();
    s.stage_dataset_write(&ObjectPathBuf::parse("/nums"), {
        let mut db = crate::type_builders::DatasetBuilder::new("");
        db.with_i32_data(&[9, 9, 9]);
        db
    })
    .unwrap();
    s.stage_created_dataset(&ObjectPathBuf::parse("/extra"), {
        let mut db = crate::type_builders::DatasetBuilder::new("");
        db.with_i32_data(&[42]);
        db
    })
    .unwrap();
    let failed = s.commit();
    assert!(
        matches!(failed, Err(Error::Io(_))),
        "the reference write is what failed, after the commit published: {failed:?}"
    );
    assert!(
        s.publication.attempted(),
        "the failure has to be past the publish for this test to mean anything"
    );
    drop(s);

    let f = crate::reader::File::open(&path).unwrap();
    assert_eq!(
        f.dataset("nums").unwrap().read_i32().unwrap(),
        vec![9, 9, 9],
        "the published overwrite must not be rolled back under it"
    );
    assert_eq!(f.dataset("extra").unwrap().read_i32().unwrap(), vec![42]);
}

#[test]
fn append_inplace_crash_consistency_paged_prefix() {
    use crate::reader::File as PureFile;

    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base.h5");
    let (start, target) = (131_000i32, 132_000i32);
    build_unit_chunked(&base, start);

    for max_phase in 1u8..=4 {
        let p = dir.path().join(format!("crash_paged_{max_phase}.h5"));
        append_stopped_at(&base, &p, start..target, max_phase);
        let expected_len = if max_phase == 4 { target } else { start };
        let f = PureFile::from_bytes(std::fs::read(&p).unwrap()).unwrap();
        assert_eq!(
            f.dataset("d").unwrap().read_i32().unwrap(),
            (0..expected_len).collect::<Vec<_>>(),
            "inconsistent paged view after crash at phase {max_phase}"
        );
    }
}

#[test]
fn append_inplace_crash_consistency_leaves_the_committed_prefix() {
    for (n, chunk, add) in [(6i32, 4u64, 5i32), (8, 2, 6), (50, 1, 200)] {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.h5");
        {
            use crate::writer::FileBuilder;
            let mut b = FileBuilder::new();
            b.create_dataset("d")
                .with_i32_data(&(0..n).collect::<Vec<i32>>())
                .with_shape(&[n as u64])
                .with_maxshape(&[MaxExtent::Unlimited])
                .with_chunks(&[chunk]);
            b.write(&base).unwrap();
        }

        for max_phase in 1u8..=4 {
            let p = dir
                .path()
                .join(format!("crash_c_{n}_{chunk}_{max_phase}.h5"));
            append_stopped_at(&base, &p, n..n + add, max_phase);
            let expected_len = if max_phase == 4 { n + add } else { n };
            let pf = crate::reader::File::from_bytes(std::fs::read(&p).unwrap()).unwrap();
            assert_eq!(
                pf.dataset("d").unwrap().read_i32().unwrap(),
                (0..expected_len).collect::<Vec<_>>(),
                "inconsistent view after crash at phase {max_phase} (n={n}, chunk={chunk})"
            );
        }
    }
}

#[test]
fn append_inplace_recover_and_reappend_after_phase3_crash() {
    use crate::reader::File as PureFile;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("phase3_recover.h5");
    let n = 50i32;
    build_unit_chunked(&path, n);
    {
        let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        // The interrupted append uses values outside the committed sequence.
        s.append_inplace_i32_phased("d", &(1000..1200).collect::<Vec<_>>(), 3)
            .unwrap();
    }
    let committed: Vec<i32> = (0..n).collect();
    let pf = PureFile::from_bytes(std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(
        pf.dataset("d").unwrap().read_i32().unwrap(),
        committed,
        "phase-3 crash exposed uncommitted data to the pure reader"
    );
    {
        let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        s.append_inplace_i32_phased("d", &(n..150).collect::<Vec<_>>(), 4)
            .unwrap();
    }

    let expected: Vec<i32> = (0..150).collect();
    let pf = PureFile::from_bytes(std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(
        pf.dataset("d").unwrap().read_i32().unwrap(),
        expected,
        "recovery did not roll forward correctly (pure reader)"
    );
}

#[test]
fn open_rejection_precedes_image_construction() {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let flagged = dir.path().join("flagged.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3]);
    b.write(&flagged).unwrap();
    let ancient = dir.path().join("ancient.h5");
    std::fs::copy(&flagged, &ancient).unwrap();
    for (path, version, flags) in [
        (&flagged, 3, SWMR_WRITE_FLAGS),
        (&ancient, 9, SuperblockStatusFlags::new()),
    ] {
        let mut data = std::fs::read(path).unwrap();
        let LocatedSuperblock {
            address,
            superblock: mut sb,
            ..
        } = crate::superblock::read(data.as_slice()).unwrap();
        let off = usize::try_from(address.get()).unwrap();
        sb.set_consistency_flags(flags).unwrap();
        let mut bytes = hdf5_pure_format::__private::serialize_superblock(&sb).unwrap();
        bytes[hdf5_pure_format::__private::HDF5_SIGNATURE.len()] = version;
        data[off..off + bytes.len()].copy_from_slice(&bytes);
        std::fs::write(path, &data).unwrap();
    }

    for path in [&flagged, &ancient] {
        let built = std::cell::Cell::new(false);
        let err = match WriteEngine::open_imaged(path, Some(FileLocking::Enabled), |h, len| {
            built.set(true);
            Ok(Box::new(HandleImage::new(
                h,
                len,
                MetadataCacheConfig::disabled(),
            )))
        }) {
            Err(e) => e,
            Ok(_) => panic!("{} must be refused", path.display()),
        };
        assert!(
            !built.get(),
            "{} was refused with {err:?}, but the image was built first",
            path.display()
        );
    }
}

#[test]
fn a_stale_consistency_flag_is_rejected_then_cleared_by_a_commit() {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stale_flag.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("d").with_i32_data(&[1, 2, 3]);
    b.write(&path).unwrap();
    {
        let mut data = std::fs::read(&path).unwrap();
        let LocatedSuperblock {
            address,
            superblock: mut sb,
            ..
        } = crate::superblock::read(data.as_slice()).unwrap();
        let off = usize::try_from(address.get()).unwrap();
        assert!(
            matches!(
                sb.version(),
                SuperblockVersion::Two | SuperblockVersion::Three
            ),
            "FileBuilder should emit a v2/v3 superblock"
        );
        sb.set_consistency_flags(SuperblockStatusFlags::swmr_write_marked())
            .unwrap();
        let bytes = hdf5_pure_format::__private::serialize_superblock(&sb).unwrap();
        data[off..off + bytes.len()].copy_from_slice(&bytes);
        std::fs::write(&path, &data).unwrap();
        assert_eq!(
            crate::superblock::read(data.as_slice())
                .unwrap()
                .superblock
                .consistency_flags(),
            SuperblockStatusFlags::swmr_write_marked()
        );
    }
    match WriteEngine::open_with_locking(&path, FileLocking::Enabled) {
        Err(Error::FileMarkedInUse(_)) => {}
        Err(e) => panic!("expected the flag refusal, got {e:?}"),
        Ok(_) => panic!("a flagged file must not be edited in place"),
    }
    {
        let mut data = std::fs::read(&path).unwrap();
        let LocatedSuperblock {
            address,
            superblock: mut sb,
            ..
        } = crate::superblock::read(data.as_slice()).unwrap();
        let off = usize::try_from(address.get()).unwrap();
        sb.set_version(SuperblockVersion::Two).unwrap();
        sb.set_consistency_flags(SuperblockStatusFlags::write_marked())
            .unwrap();
        let bytes = hdf5_pure_format::__private::serialize_superblock(&sb).unwrap();
        data[off..off + bytes.len()].copy_from_slice(&bytes);
        std::fs::write(&path, &data).unwrap();
    }
    {
        let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled)
            .expect("the gate skips a v2 superblock, so this opens");
        let mut b = DatasetBuilder::new("e");
        b.with_i32_data(&[4, 5]);
        s.stage_created_dataset(&ObjectPathBuf::parse("e"), b)
            .unwrap();
        s.commit().unwrap();
    }

    let data = std::fs::read(&path).unwrap();
    assert_eq!(
        crate::superblock::read(data.as_slice())
            .unwrap()
            .superblock
            .consistency_flags(),
        SuperblockStatusFlags::new(),
        "commit must clear the stale consistency flag"
    );
}

#[test]
fn bounded_append_crash_consistency_partial_tail_prefix() {
    let dir = tempfile::tempdir().unwrap();
    for (case, (n, chunk, add)) in [(0usize, (6i32, 4u64, 5i32)), (1, (8, 2, 6))] {
        let base = dir.path().join(std::format!("base_{case}.h5"));
        common::build_appendable(&base, n, chunk);
        for max_phase in 1u8..=4 {
            let p = dir.path().join(std::format!("crash_{case}_{max_phase}.h5"));
            std::fs::copy(&base, &p).unwrap();
            {
                let mut engine = common::open_bounded_session(&p);
                let addr = common::dataset_addr(&engine);
                let mut b = AppendBuilder::new();
                b.append_i32(&(n..n + add).collect::<Vec<_>>());
                engine
                    .append_inplace_gathered(AppendTarget::Header(addr), &b, max_phase)
                    .unwrap();
            }
            let expected_len = if max_phase == 4 { n + add } else { n };
            let got = crate::File::open(&p)
                .unwrap()
                .dataset("d")
                .unwrap()
                .read_i32()
                .unwrap();
            assert_eq!(
                got,
                (0..expected_len).collect::<Vec<_>>(),
                "case {case} phase {max_phase}"
            );
        }
    }
}

#[test]
fn bounded_persist_append_without_finalize_is_readable() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("persist_crash.h5");
    let mut b = crate::writer::FileBuilder::new();
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    b.create_dataset("d")
        .with_i32_data(&(0..6).collect::<Vec<i32>>())
        .with_shape(&[6])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[4]);
    b.write(&p).unwrap();
    {
        let mut engine = common::open_bounded_session(&p);
        assert!(engine.persist.is_some(), "persist state is armed at open");
        let addr = common::dataset_addr(&engine);
        let mut ab = AppendBuilder::new();
        ab.append_i32(&(6..20).collect::<Vec<_>>());
        engine
            .append_inplace_gathered(AppendTarget::Header(addr), &ab, 4)
            .unwrap();
    }
    let got = crate::File::open(&p)
        .unwrap()
        .dataset("d")
        .unwrap()
        .read_i32()
        .unwrap();
    assert_eq!(got, (0..20).collect::<Vec<_>>());
}

#[test]
fn bounded_paged_reopen_after_crash_realigns_and_stays_readable() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("paged_crash.h5");
    let mut b = crate::writer::FileBuilder::new();
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(0)
            .with_page_size(FileSpacePageSize::DEFAULT),
    );
    b.create_dataset("d")
        .with_i32_data(&(0..64).collect::<Vec<i32>>())
        .with_shape(&[64])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[64]);
    b.write(&p).unwrap();
    {
        let mut engine = common::open_bounded_session(&p);
        let addr = common::dataset_addr(&engine);
        let mut ab = AppendBuilder::new();
        ab.append_i32(&(64..2000).collect::<Vec<_>>());
        engine
            .append_inplace_gathered(AppendTarget::Header(addr), &ab, 4)
            .unwrap();
    }
    assert_ne!(
        std::fs::metadata(&p).unwrap().len() % 4096,
        0,
        "a crashed (un-finalized) paged session leaves the file non-page-aligned"
    );
    {
        let mut engine = common::open_bounded_session(&p);
        let addr = common::dataset_addr(&engine);
        let mut ab = AppendBuilder::new();
        ab.append_i32(&(2000..2500).collect::<Vec<_>>());
        engine
            .append_inplace_gathered(AppendTarget::Header(addr), &ab, 4)
            .unwrap();
        engine.finalize_persist().unwrap();
        engine.barrier().unwrap();
    }
    assert_eq!(
        std::fs::metadata(&p).unwrap().len() % 4096,
        0,
        "reopen + append + finalize re-aligns the paged file"
    );
    let got = crate::File::open(&p)
        .unwrap()
        .dataset("d")
        .unwrap()
        .read_i32()
        .unwrap();
    assert_eq!(got, (0..2500).collect::<Vec<_>>());
}

#[test]
fn a_bounded_commit_reads_far_less_than_the_file() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("bulk.h5");
    let rows = 2_000_000i32;
    common::build_appendable(&p, rows, 8192);
    let file_len = std::fs::metadata(&p).unwrap().len();
    assert!(file_len > 4 << 20, "file is only {file_len} bytes");

    let read_bytes = Arc::new(AtomicU64::new(0));
    {
        let mut engine = WriteEngine::open_bounded_counting(&p, Arc::clone(&read_bytes)).unwrap();
        engine.create_group(&ObjectPathBuf::parse("g")).unwrap();
        engine.commit().unwrap();
    }
    let read = read_bytes.load(Ordering::Relaxed);

    assert!(
        read > 0,
        "the commit read nothing, so the test proves nothing"
    );
    assert!(
        read < 64 << 10,
        "a bounded commit read {read} bytes of a {file_len}-byte file"
    );
}

#[test]
fn gathering_writes_costs_fewer_of_them_and_changes_no_byte() {
    let dir = tempfile::tempdir().unwrap();
    for paged in [false, true] {
        let straight = dir.path().join(std::format!("straight_{paged}.h5"));
        let gathered = dir.path().join(std::format!("gathered_{paged}.h5"));
        common::gather_fixture(&straight, 4, paged);
        common::gather_fixture(&gathered, 4, paged);

        let mut a = WriteEngine::open_with_locking(&straight, FileLocking::Enabled).unwrap();
        a.set_sync_policy(SyncPolicy::OnClose);
        a.image
            .set_write_buffering(WriteBuffering::Unbuffered)
            .unwrap();
        let (straight_appends, straight_commit) = gather_workload(&mut a);
        a.force_sync().unwrap();
        drop(a);

        let mut b = WriteEngine::open_with_locking(&gathered, FileLocking::Enabled).unwrap();
        b.set_sync_policy(SyncPolicy::OnClose);
        let (gathered_appends, gathered_commit) = gather_workload(&mut b);
        b.force_sync().unwrap();
        drop(b);
        assert!(
            gathered_commit * 3 < straight_commit,
            "paged={paged}: gathering must cost meaningfully fewer writes for a \
             commit, but cost {gathered_commit} against {straight_commit}"
        );
        assert!(
            gathered_appends <= straight_appends,
            "paged={paged}: gathering must not cost more writes for the appends, \
             but cost {gathered_appends} against {straight_appends}"
        );
        assert_eq!(
            std::fs::read(&straight).unwrap(),
            std::fs::read(&gathered).unwrap(),
            "paged={paged}: gathering changed the file it produced"
        );
    }
}

#[test]
fn a_status_flag_write_leaves_a_userblock_files_root_alone() {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("userblock.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UserblockSize::try_from(4096).unwrap());
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3, 4])
        .with_shape(&[4]);
    b.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    assert_eq!(
        s.context.superblock().base_address,
        BaseAddress::new(4096),
        "the fixture must have a userblock, or this test holds for a file \
         whose absolute and relative roots are the same number"
    );
    s.set_consistency_flags(SuperblockStatusFlags::write_marked())
        .unwrap();
    s.set_consistency_flags(SuperblockStatusFlags::new())
        .unwrap();
    s.force_sync().unwrap();
    drop(s);

    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "raising and clearing the flags must leave the file as it was"
    );
    assert_eq!(
        crate::reader::File::open(&path)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_i32()
            .unwrap(),
        vec![1, 2, 3, 4],
        "and the file must still read"
    );
}

#[test]
fn a_page_buffered_session_flushes_when_it_exceeds_its_budget() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("outrun.h5");
    common::gather_fixture(&path, 1, true);

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    s.set_sync_policy(SyncPolicy::OnClose);
    s.set_page_buffer_size(WRITE_GATHER_BYTES).unwrap();
    let before = s.image.issued_writes();
    s.append_inplace_i32_phased("t0", &vec![7i32; 1 << 20], 4)
        .unwrap();
    let during = s.image.issued_writes() - before;
    s.force_sync().unwrap();
    s.release_status_flags().unwrap();
    drop(s);

    assert!(
        during > 0,
        "a session that wrote four times its budget issued nothing until it \
         closed, so the budget it was given is not the one it asked for"
    );
    assert_eq!(
        crate::reader::File::open(&path)
            .unwrap()
            .dataset("t0")
            .unwrap()
            .read_i32()
            .unwrap()
            .len(),
        256 + (1 << 20),
        "and the flushed append must still read back whole"
    );
}

#[test]
fn a_smaller_budget_issues_more_writes() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = |path: &std::path::Path, tables: usize| {
        use crate::writer::FileBuilder;
        let mut b = FileBuilder::new();
        b.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page)
                .with_persistence(true)
                .with_page_size(FileSpacePageSize::DEFAULT),
        );
        for t in 0..tables {
            b.create_dataset(&std::format!("t{t}"))
                .with_i32_data(&(0..256).collect::<Vec<_>>())
                .with_shape(&[256])
                .with_maxshape(&[MaxExtent::Unlimited])
                .with_chunks(&[64]);
        }
        b.write(path).unwrap();
    };
    let run = |path: &std::path::Path, budget: usize, workload: &dyn Fn(&mut WriteEngine)| {
        let mut s = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        s.set_page_buffer_size(budget).unwrap();
        let before = s.image.issued_writes();
        workload(&mut s);
        s.force_sync().unwrap();
        s.release_status_flags().unwrap();
        let issued = s.image.issued_writes() - before;
        drop(s);
        issued
    };
    let workload = |scattered: bool| {
        move |s: &mut WriteEngine| {
            if scattered {
                for round in 0..4 {
                    for t in 0..8 {
                        s.append_inplace_i32_phased(&std::format!("t{t}"), &[round; 64], 4)
                            .unwrap();
                    }
                }
            } else {
                s.append_inplace_i32_phased("t0", &vec![7i32; 1 << 20], 4)
                    .unwrap();
            }
        }
    };

    for (label, tables, scattered) in [("scattered", 8, true), ("one long run", 1, false)] {
        let workload = workload(scattered);
        let at = |budget: usize| {
            let path = dir
                .path()
                .join(std::format!("{}_{budget}.h5", label.replace(' ', "_")));
            fixture(&path, tables);
            run(&path, budget, &workload)
        };
        let ample = at(WRITE_GATHER_BYTES);
        let smallest = at(4096);
        for smaller in [4096, 64 * 1024] {
            let below = at(smaller);
            assert!(
                ample <= below,
                "{label}: a {smaller}-byte budget issued {below} writes against \
                 {ample} at {WRITE_GATHER_BYTES}, so a smaller budget is not the \
                 trade the property documents"
            );
        }
        assert!(
            smallest > ample,
            "{label}: the budget changed nothing — {smallest} writes at 4 KiB \
             against {ample} at 1 MiB — so this workload cannot say which \
             budget is better"
        );
    }
}

#[rstest]
fn a_page_buffer_marks_a_userblock_file_without_repointing_its_root(
    #[values(false, true)] paged: bool,
) {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ub_buffered.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UserblockSize::try_from(4096).unwrap());
    if paged {
        b.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page)
                .with_persistence(true)
                .with_page_size(FileSpacePageSize::DEFAULT),
        );
    }
    b.create_dataset("t0")
        .with_i32_data(&(0..256).collect::<Vec<_>>())
        .with_shape(&[256]);
    b.write(&path).unwrap();
    let userblock_before = std::fs::read(&path).unwrap()[..4096].to_vec();

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    assert_eq!(
        s.context.superblock().base_address,
        BaseAddress::new(4096),
        "the fixture must have a userblock, or this test says nothing about \
         the conversion it exists for"
    );
    s.set_sync_policy(SyncPolicy::OnClose);
    s.set_page_buffer_size(1 << 20)
        .expect("a userblock file must accept a page buffer");
    assert_eq!(
        s.held_status_flags,
        SuperblockStatusFlags::write_marked(),
        "a page-buffered session must hold the crash mark"
    );
    for t in 0..3 {
        let mut db = crate::type_builders::DatasetBuilder::new(&std::format!("n{t}"));
        db.with_f64_data(&[2.5f64; 32]).with_shape(&[32]);
        s.stage_created_dataset(&ObjectPathBuf::parse(&std::format!("/n{t}")), db)
            .unwrap();
    }
    s.commit().unwrap();
    s.force_sync().unwrap();
    s.release_status_flags().unwrap();
    drop(s);

    assert_eq!(
        &std::fs::read(&path).unwrap()[..4096],
        &userblock_before[..],
        "the page-buffered session rewrote the userblock"
    );
    let f = crate::reader::File::open(&path).unwrap();
    assert_eq!(
        f.dataset("t0").unwrap().read_i32().unwrap(),
        (0..256).collect::<Vec<_>>(),
        "the commit must have left the original dataset alone"
    );
    for t in 0..3 {
        assert_eq!(
            f.dataset(&std::format!("n{t}"))
                .unwrap()
                .read_f64()
                .unwrap(),
            vec![2.5f64; 32],
            "n{t}: the committed dataset must read back"
        );
    }
}

#[test]
fn a_page_buffer_holds_dirty_pages_across_operations() {
    let dir = tempfile::tempdir().unwrap();
    let run = |path: &std::path::Path, page_buffer: usize| {
        let mut s = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        if page_buffer != 0 {
            s.set_page_buffer_size(page_buffer).unwrap();
        }
        let (appends, commit) = gather_workload(&mut s);
        s.force_sync().unwrap();
        s.release_status_flags().unwrap();
        drop(s);
        appends + commit
    };

    for paged in [true, false] {
        let label = if paged { "paged" } else { "unpaged" };
        let per_op = dir.path().join(std::format!("{label}_per_op.h5"));
        let buffered = dir.path().join(std::format!("{label}_buffered.h5"));
        common::gather_fixture(&per_op, 4, paged);
        common::gather_fixture(&buffered, 4, paged);

        let per_op_writes = run(&per_op, 0);
        let buffered_writes = run(&buffered, 1 << 20);

        assert!(
            buffered_writes * 4 < per_op_writes,
            "{label}: a page buffer must cost meaningfully fewer writes than the \
             per-operation default, but cost {buffered_writes} against {per_op_writes}"
        );
        assert_eq!(
            std::fs::read(&per_op).unwrap(),
            std::fs::read(&buffered).unwrap(),
            "{label}: a page buffer changed the file it produced"
        );
    }
}

#[test]
fn a_page_buffer_below_the_gather_budget_writes_the_same_file() {
    let dir = tempfile::tempdir().unwrap();
    let run = |path: &std::path::Path, budget: usize| -> Vec<i32> {
        let file = crate::reader::File::create_with_options(
            path,
            crate::FileCreateProperties::new().with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::Page)
                    .with_persistence(true)
                    .with_page_size(FileSpacePageSize::try_from(16 * 1024).unwrap()),
            ),
            crate::FileAccessProperties::new()
                .with_sync_policy(SyncPolicy::OnClose)
                .with_page_buffer_size(budget),
        )
        .unwrap();
        file.root()
            .create_dataset("d", |b| {
                b.with_i32_data(&(0..64).collect::<Vec<i32>>())
                    .with_shape(&[64])
                    .with_maxshape(&[MaxExtent::Unlimited])
                    .with_chunks(&[64]);
            })
            .unwrap();
        file.commit().unwrap();
        for round in 0..8i32 {
            let mut ds = file.dataset("d").unwrap();
            ds.append(&vec![round; 64]).unwrap();
        }
        file.root()
            .create_dataset("added", |b| {
                b.with_f64_data(&[2.5f64; 32]).with_shape(&[32]);
            })
            .unwrap();
        file.commit().unwrap();
        file.close().unwrap();

        let reopened = crate::reader::File::open(path).unwrap();
        assert_eq!(
            reopened.dataset("added").unwrap().read_f64().unwrap(),
            vec![2.5f64; 32],
            "the dataset committed through the buffer must read back"
        );
        reopened.dataset("d").unwrap().read_i32().unwrap()
    };

    let small = dir.path().join("small_budget.h5");
    let ample = dir.path().join("ample_budget.h5");
    let from_small = run(&small, 256 * 1024);
    let from_ample = run(&ample, 1 << 20);

    let mut expected: Vec<i32> = (0..64).collect();
    for round in 0..8i32 {
        expected.extend(std::iter::repeat_n(round, 64));
    }
    assert_eq!(
        from_small, expected,
        "a 256 KiB page buffer did not append what it was given"
    );
    assert_eq!(
        from_ample, expected,
        "and neither did the 1 MiB one, so the file comparison below would be \
         two arms agreeing on the wrong answer"
    );
    assert_eq!(
        std::fs::read(&small).unwrap(),
        std::fs::read(&ample).unwrap(),
        "a 256 KiB page buffer produced a different file from a 1 MiB one"
    );
}

#[test]
fn a_barrier_orders_the_publish_point_last_under_every_policy() {
    let dir = tempfile::tempdir().unwrap();
    for policy in [SyncPolicy::Always, SyncPolicy::OnClose] {
        let path = dir.path().join(std::format!("order_{policy:?}.h5"));
        common::gather_fixture(&path, 1, false);

        let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        s.set_sync_policy(policy);

        let mut db = crate::type_builders::DatasetBuilder::new("added");
        db.with_f64_data(&[2.5f64; 64]).with_shape(&[64]);
        s.stage_created_dataset(&ObjectPathBuf::parse("/added"), db)
            .unwrap();
        let before = s.image.issued_write_order().len();
        s.commit().unwrap();

        let order = s.image.issued_write_order()[before..].to_vec();
        let superblock = order
            .iter()
            .position(|&(at, _)| at == s.context.signature().get())
            .unwrap_or_else(|| panic!("{policy:?}: the commit never wrote the superblock"));
        let content = order
            .iter()
            .rposition(|&(at, _)| at != s.context.signature().get())
            .unwrap_or_else(|| panic!("{policy:?}: the commit wrote nothing but a superblock"));
        assert!(
            superblock > content,
            "{policy:?}: the superblock was issued at position {superblock} of \
             {order:?}, ahead of content at {content} — a failure in that window \
             leaves a root pointing at bytes that are not in the file"
        );
        let before = s.image.issued_write_order().len();
        s.append_inplace_i32_phased("t0", &[7; 64], 4).unwrap();
        let order = s.image.issued_write_order()[before..].to_vec();
        let highest = order
            .iter()
            .map(|&(at, _)| at)
            .max()
            .expect("the append wrote");
        assert!(
            order.last().expect("the append wrote").0 < highest,
            "{policy:?}: the append's last write is its highest-addressed one, so \
             the whole append went out in address order and the dimension that \
             publishes the new rows preceded the chunk bytes: {order:?}"
        );
    }
}

#[test]
fn a_page_buffer_changes_no_byte_through_the_bounded_backing() {
    use crate::reader::File;

    let dir = tempfile::tempdir().unwrap();
    let run = |name: &str, page_buffer: usize| {
        let path = dir.path().join(name);
        common::gather_fixture(&path, 4, true);
        {
            let props = crate::FileAccessProperties::new()
                .with_sync_policy(SyncPolicy::OnClose)
                .with_memory_strategy(MemoryStrategy::Bounded)
                .with_page_buffer_size(page_buffer);
            let file = File::open_rw_with_options(&path, props).unwrap();
            assert_eq!(
                file.edit_backing(),
                Some(crate::EditBacking::Bounded),
                "{name}: this must exercise the mirrorless backing"
            );
            let root = file.root();
            for round in 0..4u8 {
                for t in 0..4 {
                    let mut ds = file.dataset(&std::format!("t{t}")).unwrap();
                    ds.append(&[i32::from(round); 64]).unwrap();
                }
            }
            for t in 0..4 {
                root.create_dataset(&std::format!("n{t}"), |b| {
                    b.with_f64_data(&[2.5f64; 32]).with_shape(&[32]);
                })
                .unwrap();
            }
            file.commit().unwrap();
            root.set_attr("tag", crate::AttrValue::I32(1)).unwrap();
            file.commit().unwrap();
            file.close().unwrap();
        }
        std::fs::read(&path).unwrap()
    };

    assert_eq!(
        run("bounded_plain.h5", 0),
        run("bounded_buffered.h5", 1 << 20),
        "a page buffer changed the file the bounded backing produced"
    );
}

#[test]
fn a_finished_operation_has_nothing_left_to_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("finished_op.h5");
    common::gather_fixture(&path, 1, false);

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    s.set_sync_policy(SyncPolicy::OnClose);

    let before_append = s.image.issued_writes();
    s.append_inplace_i32_phased("t0", &[7; 64], 4).unwrap();
    let after_append = s.image.issued_writes();
    assert!(
        after_append > before_append,
        "the append issued nothing at all, so the equality below holds for a \
         session that did no work"
    );
    s.force_sync().unwrap();
    assert_eq!(
        s.image.issued_writes(),
        after_append,
        "a finished append left writes in this process's memory"
    );

    let mut db = crate::type_builders::DatasetBuilder::new("added");
    db.with_f64_data(&[1.5f64; 8]).with_shape(&[8]);
    s.stage_created_dataset(&ObjectPathBuf::parse("/added"), db)
        .unwrap();
    s.commit().unwrap();
    let after_commit = s.image.issued_writes();
    assert!(
        after_commit > after_append,
        "the commit issued nothing at all"
    );
    s.force_sync().unwrap();
    assert_eq!(
        s.image.issued_writes(),
        after_commit,
        "a finished commit left writes in this process's memory"
    );
    drop(s);
    let f = crate::reader::File::open(&path).unwrap();
    assert_eq!(f.dataset("t0").unwrap().read_i32().unwrap().len(), 320);
    assert_eq!(
        f.dataset("added").unwrap().read_f64().unwrap(),
        vec![1.5f64; 8]
    );
}

#[test]
fn finalize_persist_has_nothing_left_to_write() {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("persisting.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    b.create_dataset("t0")
        .with_i32_data(&(0..256).collect::<Vec<_>>())
        .with_shape(&[256])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[64]);
    b.write(&path).unwrap();

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    s.set_sync_policy(SyncPolicy::OnClose);
    s.append_inplace_i32_phased("t0", &[7; 64], 4).unwrap();

    s.finalize_persist().unwrap();
    let after = s.image.issued_writes();
    s.force_sync().unwrap();
    assert_eq!(
        s.image.issued_writes(),
        after,
        "finalize_persist returned with writes still gathered"
    );
}

#[test]
fn an_unbuffered_append_costs_a_small_constant_number_of_writes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("swmr_cost.h5");
    common::gather_fixture(&path, 1, false);

    let mut s = WriteEngine::open_swmr_writer(&path, SyncPolicy::OnClose).unwrap();
    let before = s.image.issued_writes();
    s.append_inplace_i32_phased("t0", &[7; 64], 4).unwrap();
    let cost = s.image.issued_writes() - before;

    assert!(
        cost > 0,
        "the append issued nothing, so the ceiling below proves nothing"
    );
    assert!(
        cost <= 10,
        "an unbuffered append costs {cost} writes; the array header's six \
         statistics belong in one write, not six, and each checksum belongs in \
         the write that changed what it covers (measured at 8)"
    );
}

#[test]
fn the_swmr_writer_holds_no_write_back() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("swmr.h5");
    common::gather_fixture(&path, 1, false);

    let mut s = WriteEngine::open_swmr_writer(&path, SyncPolicy::OnClose).unwrap();
    let before = std::fs::read(&path).unwrap().len();
    s.append_inplace_i32_phased("t0", &[7; 64], 1).unwrap();

    assert!(
        std::fs::read(&path).unwrap().len() > before,
        "a SWMR reader must see the phase-1 chunk bytes as soon as they are written"
    );
}

#[test]
fn sync_policy_governs_every_barrier() {
    use crate::writer::FileBuilder;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    let dir = tempfile::tempdir().unwrap();
    let run = |name: &str, policy: SyncPolicy| -> [u64; 5] {
        let path = dir.path().join(name);
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..8).collect::<Vec<_>>())
            .with_shape(&[8])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[4]);
        b.write(&path).unwrap();

        let syncs = Arc::new(AtomicU64::new(0));
        let mut s = WriteEngine::open_sync_counting(&path, policy, Arc::clone(&syncs)).unwrap();
        s.append_inplace_i32_phased("d", &[8, 9, 10, 11], 4)
            .unwrap();
        let after_append = syncs.load(Ordering::Relaxed);
        let mut db = crate::type_builders::DatasetBuilder::new("added");
        db.with_f64_data(&[2.5f64; 8]).with_shape(&[8]);
        s.stage_created_dataset(&ObjectPathBuf::parse("/added"), db)
            .unwrap();
        s.commit().unwrap();
        let after_commit = syncs.load(Ordering::Relaxed);
        let mut ow = crate::type_builders::DatasetBuilder::new("added");
        ow.with_f64_data(&[4.5f64; 8]).with_shape(&[8]);
        s.stage_dataset_write(&ObjectPathBuf::parse("/added"), ow)
            .unwrap();
        s.commit().unwrap();
        let after_overwrite = syncs.load(Ordering::Relaxed);
        s.delete(&ObjectPathBuf::parse("/added")).unwrap();
        s.commit().unwrap();
        let after_truncate = syncs.load(Ordering::Relaxed);

        s.force_sync().unwrap();
        let after_forced = syncs.load(Ordering::Relaxed);
        drop(s);
        let f = crate::reader::File::open(&path).unwrap();
        assert_eq!(
            f.dataset("d").unwrap().read_i32().unwrap(),
            (0..12).collect::<Vec<_>>(),
            "the append must land under {policy:?}"
        );
        assert!(
            f.dataset("added").is_err(),
            "the deleting commit must land under {policy:?}"
        );
        [
            after_append,
            after_commit,
            after_overwrite,
            after_truncate,
            after_forced,
        ]
    };

    let always = run("always.h5", SyncPolicy::Always);
    assert!(always[0] > 0, "an immediate append syncs under Always");
    assert!(always[1] > always[0], "so does a commit");
    assert!(
        always[2] > always[1],
        "so does the same-length-overwrite fast path"
    );
    assert!(
        always[3] > always[2],
        "so does a commit that truncates the file"
    );
    assert_eq!(
        always[4],
        always[3] + 1,
        "and a forced sync is exactly one more"
    );

    let deferred = run("on_close.h5", SyncPolicy::OnClose);
    assert_eq!(
        &deferred[..4],
        &[0, 0, 0, 0],
        "OnClose must leave every one of those in-session barriers unissued"
    );
    assert_eq!(
        deferred[4], 1,
        "a forced sync is issued whatever the policy says — it is what the \
         teardown path and File::sync both take"
    );
}

#[test]
fn sync_policy_governs_the_persisting_and_flag_barriers() {
    use crate::writer::FileBuilder;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    let dir = tempfile::tempdir().unwrap();
    let run = |name: &str, strategy: FileSpaceStrategy, policy: SyncPolicy| -> [u64; 3] {
        let path = dir.path().join(name);
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..8).collect::<Vec<_>>())
            .with_shape(&[8])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[4]);
        b.create_dataset("victim")
            .with_f64_data(&[1.5f64; 64])
            .with_shape(&[64]);
        b.with_file_space_settings(
            FileSpaceSettings::new(strategy)
                .with_persistence(true)
                .with_threshold(if strategy == FileSpaceStrategy::Page {
                    0
                } else {
                    1
                }),
        );
        b.write(&path).unwrap();

        let syncs = Arc::new(AtomicU64::new(0));
        let mut s = WriteEngine::open_sync_counting(&path, policy, Arc::clone(&syncs)).unwrap();
        assert!(
            s.persist.is_some(),
            "the fixture must persist its free space, or this tests the wrong tail"
        );
        s.delete(&ObjectPathBuf::parse("/victim")).unwrap();
        s.commit().unwrap();
        let after_commit = syncs.load(Ordering::Relaxed);

        s.set_consistency_flags(SuperblockStatusFlags::new())
            .unwrap();
        let after_flags = syncs.load(Ordering::Relaxed);
        let issued = s.image.issued_writes();
        s.force_sync().unwrap();
        assert_eq!(
            s.image.issued_writes(),
            issued,
            "{policy:?}: the consistency-flag write was left gathered"
        );
        s.append_inplace_i32_phased("d", &[8, 9, 10, 11], 4)
            .unwrap();
        let before_close = syncs.load(Ordering::Relaxed);
        s.finalize_persist().unwrap();
        let after_close = syncs.load(Ordering::Relaxed) - before_close;

        drop(s);
        let f = crate::reader::File::open(&path).unwrap();
        assert!(
            f.dataset("victim").is_err(),
            "the delete must land under {policy:?}"
        );
        assert_eq!(
            f.dataset("d").unwrap().read_i32().unwrap(),
            (0..12).collect::<Vec<_>>(),
            "and so must the append under {policy:?}"
        );
        [after_commit, after_flags, after_close]
    };
    for (label, strategy) in [
        ("fsmaggr", FileSpaceStrategy::FsmAggr),
        ("paged", FileSpaceStrategy::Page),
    ] {
        let always = run(&format!("{label}_always.h5"), strategy, SyncPolicy::Always);
        assert!(
            always[0] > 0,
            "the {label} persisting commit tail syncs under Always"
        );
        assert!(
            always[1] > always[0],
            "so does the consistency-flag write a SWMR session makes ({label})"
        );
        assert!(
            always[2] > 0,
            "so does the manager re-homing close owes ({label})"
        );

        assert_eq!(
            run(
                &format!("{label}_on_close.h5"),
                strategy,
                SyncPolicy::OnClose
            ),
            [0, 0, 0],
            "OnClose must leave the {label} persisting tail, the flag write, and the \
             close-time manager re-homing with no fsync at all"
        );
    }
}

#[test]
fn a_commit_whose_publish_write_fails_rolls_nothing_back() {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("torn_publish.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("nums").with_i32_data(&[1, 2, 3]);
    b.write(&path).unwrap();
    let mut s = WriteEngine::open_torn_writes(&path, 0..48).unwrap();
    s.stage_dataset_write(&ObjectPathBuf::parse("/nums"), {
        let mut db = crate::type_builders::DatasetBuilder::new("");
        db.with_i32_data(&[9, 9, 9]);
        db
    })
    .unwrap();
    s.stage_created_dataset(&ObjectPathBuf::parse("/extra"), {
        let mut db = crate::type_builders::DatasetBuilder::new("");
        db.with_i32_data(&[42]);
        db
    })
    .unwrap();
    let failed = s.commit();
    assert!(
        matches!(failed, Err(Error::Io(_))),
        "the publish write is what failed: {failed:?}"
    );
    drop(s);

    let f = crate::reader::File::open(&path).unwrap();
    assert_eq!(
        f.dataset("extra").unwrap().read_i32().unwrap(),
        vec![42],
        "the publish landed despite reporting failure, so this commit is live"
    );
    assert_eq!(
        f.dataset("nums").unwrap().read_i32().unwrap(),
        vec![9, 9, 9],
        "a rollback under a commit that did publish would tear it"
    );
}

#[test]
fn an_undo_replays_two_hard_links_to_one_block_newest_first() {
    let dir = tempfile::tempdir().unwrap();
    let path = test_util::copy_data("c/hard_link_undo.h5", dir.path());

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let mut superblock = s.context.superblock().clone();
    superblock.superblock_extension_address = Some(StoredAddress::ZERO);
    let proposed = s.context.propose(superblock).unwrap();
    s.context.adopt(proposed).unwrap();
    for (name, data) in [("/aa", [9, 9, 9]), ("/bb", [8, 8, 8])] {
        s.stage_dataset_write(&ObjectPathBuf::parse(name), {
            let mut db = crate::type_builders::DatasetBuilder::new("");
            db.with_i32_data(&data);
            db
        })
        .unwrap();
    }
    s.stage_created_dataset(&ObjectPathBuf::parse("/extra"), {
        let mut db = crate::type_builders::DatasetBuilder::new("");
        db.with_i32_data(&[42]);
        db
    })
    .unwrap();
    assert!(s.commit().is_err());
    drop(s);

    let f = crate::reader::File::open(&path).unwrap();
    assert_eq!(
        f.dataset("aa").unwrap().read_i32().unwrap(),
        vec![1, 2, 3],
        "the block both links share must hold the value it held before"
    );
    assert_eq!(f.dataset("bb").unwrap().read_i32().unwrap(), vec![1, 2, 3]);
}

#[test]
fn a_failed_commit_puts_back_every_chunk_it_overwrote() {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chunked_undo.h5");
    let original: Vec<i32> = (0..4096i32)
        .map(|i| i.wrapping_mul(2_654_435_761u32 as i32) ^ i)
        .collect();
    let mut b = FileBuilder::new();
    b.create_dataset("grid")
        .with_i32_data(&original)
        .with_shape(&[4096])
        .with_chunks(&[512])
        .with_deflate(DeflateLevel::Six);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(0),
    );
    b.write(&path).unwrap();

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let mut superblock = s.context.superblock().clone();
    superblock.superblock_extension_address = Some(StoredAddress::ZERO);
    let proposed = s.context.propose(superblock).unwrap();
    s.context.adopt(proposed).unwrap();
    s.stage_dataset_write(&ObjectPathBuf::parse("/grid"), {
        let mut db = crate::type_builders::DatasetBuilder::new("");
        db.with_i32_data(&vec![0i32; 4096]);
        db
    })
    .unwrap();
    s.stage_created_dataset(&ObjectPathBuf::parse("/extra"), {
        let mut db = crate::type_builders::DatasetBuilder::new("");
        db.with_i32_data(&[42]);
        db
    })
    .unwrap();
    assert!(s.commit().is_err());
    drop(s);

    let f = crate::reader::File::open(&path).unwrap();
    assert_eq!(
        f.dataset("grid").unwrap().read_i32().unwrap(),
        original,
        "every chunk, and the index sizing them, must read as it did before"
    );
}

#[test]
fn a_commit_does_not_replay_a_journal_it_did_not_write() {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stale_journal.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("nums").with_i32_data(&[1, 2, 3]);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(0),
    );
    b.write(&path).unwrap();

    let nums_block = {
        let f = crate::reader::File::open(&path).unwrap();
        match f.dataset("nums").unwrap().layout().unwrap() {
            crate::Layout::Contiguous {
                address: Some(a), ..
            } => a,
            other => panic!("expected a contiguous dataset: {other:?}"),
        }
    };

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let stale: Vec<u8> = [7i32, 7, 7].iter().flat_map(|v| v.to_le_bytes()).collect();
    s.inplace_undo.push((nums_block, stale));
    let mut superblock = s.context.superblock().clone();
    superblock.superblock_extension_address = Some(StoredAddress::ZERO);
    let proposed = s.context.propose(superblock).unwrap();
    s.context.adopt(proposed).unwrap();
    s.stage_created_dataset(&ObjectPathBuf::parse("/extra"), {
        let mut db = crate::type_builders::DatasetBuilder::new("");
        db.with_i32_data(&[42]);
        db
    })
    .unwrap();
    assert!(s.commit().is_err());
    drop(s);

    let f = crate::reader::File::open(&path).unwrap();
    assert_eq!(
        f.dataset("nums").unwrap().read_i32().unwrap(),
        vec![1, 2, 3],
        "the commit replayed a journal entry that was not its own"
    );
}

#[test]
fn a_rollback_orders_the_values_it_puts_back() {
    use crate::writer::FileBuilder;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    let mut counted = Vec::new();
    for with_overwrite in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rollback_syncs.h5");
        let mut b = FileBuilder::new();
        b.create_dataset("nums").with_i32_data(&[1, 2, 3]);
        b.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
                .with_persistence(true)
                .with_threshold(0),
        );
        b.write(&path).unwrap();

        let syncs = Arc::new(AtomicU64::new(0));
        let mut s =
            WriteEngine::open_sync_counting(&path, SyncPolicy::Always, Arc::clone(&syncs)).unwrap();
        let mut superblock = s.context.superblock().clone();
        superblock.superblock_extension_address = Some(StoredAddress::ZERO);
        let proposed = s.context.propose(superblock).unwrap();
        s.context.adopt(proposed).unwrap();
        if with_overwrite {
            s.stage_dataset_write(&ObjectPathBuf::parse("/nums"), {
                let mut db = crate::type_builders::DatasetBuilder::new("");
                db.with_i32_data(&[9, 9, 9]);
                db
            })
            .unwrap();
        }
        s.stage_created_dataset(&ObjectPathBuf::parse("/extra"), {
            let mut db = crate::type_builders::DatasetBuilder::new("");
            db.with_i32_data(&[42]);
            db
        })
        .unwrap();
        let before = syncs.load(Ordering::Relaxed);
        assert!(s.commit().is_err());
        counted.push(syncs.load(Ordering::Relaxed) - before);
    }

    assert_eq!(
        counted[0], 0,
        "a refusal with nothing to put back reaches no barrier"
    );
    assert_eq!(
        counted[1],
        counted[0] + 1,
        "a refusal that put values back must order them: {counted:?}"
    );
}

#[test]
fn a_bounded_commit_reads_the_value_it_is_replacing() {
    use crate::writer::FileBuilder;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("journal_reads.h5");
    let data: Vec<i32> = (0..250_000).collect();
    let payload = (data.len() * 4) as u64;
    let mut b = FileBuilder::new();
    b.create_dataset("nums").with_i32_data(&data);
    b.write(&p).unwrap();

    let read_bytes = Arc::new(AtomicU64::new(0));
    let mut engine = WriteEngine::open_bounded_counting(&p, Arc::clone(&read_bytes)).unwrap();
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("/nums"), {
            let mut db = crate::type_builders::DatasetBuilder::new("");
            db.with_i32_data(&data);
            db
        })
        .unwrap();
    let before = read_bytes.load(Ordering::Relaxed);
    engine.commit().unwrap();
    let read = read_bytes.load(Ordering::Relaxed) - before;

    assert!(
        read >= payload,
        "the journal must read the whole value it replaces: {read} of {payload}"
    );
    assert!(
        read < payload + (64 << 10),
        "and not much more than it: {read} against {payload}"
    );
}

/// Writes dataset `d` with one element per chunk and initial values `0..n`.
fn build_unit_chunked(path: &std::path::Path, n: i32) {
    use crate::writer::FileBuilder;
    let data: Vec<i32> = (0..n).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&data)
        .with_shape(&[n as u64])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[1]);
    b.write(path).unwrap();
}

/// Copies the fixture to `out` and stops an append after `max_phase` durability phases.
fn append_stopped_at(
    base: &std::path::Path,
    out: &std::path::Path,
    values: std::ops::Range<i32>,
    max_phase: u8,
) {
    std::fs::copy(base, out).unwrap();
    let mut s = WriteEngine::open_with_locking(out, FileLocking::Enabled).unwrap();
    s.append_inplace_i32_phased("d", &values.collect::<Vec<_>>(), max_phase)
        .unwrap();
}

/// Runs immediate appends and a staged commit, returning their separate write counts.
fn gather_workload(session: &mut WriteEngine) -> (u64, u64) {
    let before = session.image.issued_writes();
    for round in 0..4 {
        for t in 0..4 {
            session
                .append_inplace_i32_phased(&std::format!("t{t}"), &[round; 64], 4)
                .unwrap();
        }
    }
    for t in 0..4 {
        let mut db = crate::type_builders::DatasetBuilder::new(&std::format!("n{t}"));
        db.with_f64_data(&[2.5f64; 32]).with_shape(&[32]);
        session
            .stage_created_dataset(&ObjectPathBuf::parse(&std::format!("/n{t}")), db)
            .unwrap();
    }
    let after_appends = session.image.issued_writes();
    session.commit().unwrap();
    (
        after_appends - before,
        session.image.issued_writes() - after_appends,
    )
}
