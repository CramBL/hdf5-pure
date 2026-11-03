use std::fs;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use hdf5_pure_format::__private::HDF5_SIGNATURE;
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::FreeList;
use hdf5_pure_space::__private::ReturnedExtents;
use hdf5_pure_space::__private::{Extent, ManagerKind, ManagerSections};
use rstest::rstest;
use test_util::disk_log::{self, DiskOp};
use test_util::free_space;
use test_util::object_header::v2 as object_header;
use test_util::snapshot::text::{Report, Value};
use test_util_hdf5::file_space::persisted::{ManagerSlot, PersistedImage, PersistedStrategy};

use crate::FileSpacePageSize;
#[cfg(test)]
use crate::FileSpaceSettings;
use crate::FormatError;
use crate::UserblockSize;
use crate::address::{DecodedStoredAddress, StoredAddress, StoredAddressExt};
use crate::chunk_index_inplace::Store;
use crate::edit::commit::PublicationPurpose;
use crate::edit::object_header as edit_header;
use crate::edit::test_util as edit_test_util;
use crate::edit::{TailPlacement, WriteEngine};
use crate::image::{CountingImage, FileImage, HandleImage, WriteBuffering};
use crate::object_path::ObjectPathBuf;
use crate::source::MetadataCacheConfig;
use crate::type_builders::DatasetBuilder;
use crate::{AttrValue, Error, File, FileBuilder, FileLocking, FileSpaceStrategy, SyncPolicy};
use crate::{MemoryStrategy, PersistedFreeSpacePolicy};

#[rstest]
#[case::small(512, 0, 1)]
#[case::default(4_096, 0, 1)]
#[case::userblock(4_096, 4_096, 1)]
#[case::threshold(4_096, 0, 8_193)]
#[case::large(8_192, 8_192, 1)]
fn final_typed_manager_sections_survive_commit_finalization_and_reopen(
    #[case] page: u64,
    #[case] base: u64,
    #[case] threshold: u64,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("typed.h5");
    let mut builder = FileBuilder::new();
    builder
        .with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page)
                .with_persistence(true)
                .with_threshold(threshold)
                .with_page_size(FileSpacePageSize::try_from(page).unwrap()),
        )
        .with_userblock(UserblockSize::try_from(base).unwrap());
    if base != 0 {
        builder.with_userblock_content(b"typed manager tail");
    }
    builder.create_dataset("keep").with_u8_data(&[7; 100]);
    builder.write(&path).unwrap();
    let original = fs::read(&path).unwrap();
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    for index in 0..6 {
        let mut dataset = DatasetBuilder::new("");
        dataset.with_u8_data(&[9; 100]);
        engine
            .stage_created_dataset(&ObjectPathBuf::parse(&format!("added{index}")), dataset)
            .unwrap();
        engine.commit().unwrap();
        let expected = typed_sections(&engine);
        assert_manager_allocations_owned(&engine);
        engine.force_sync().unwrap();
        drop(engine);
        engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        assert_eq!(typed_sections(&engine), expected);
        engine.finalize_persist().unwrap();
        let settled = issued_bytes(&engine);
        engine.finalize_persist().unwrap();
        assert_eq!(issued_bytes(&engine), settled);
        assert_eq!(typed_sections(&engine), expected);
    }
    drop(engine);
    assert_eq!(
        &fs::read(&path).unwrap()[..base as usize],
        &original[..base as usize]
    );
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("keep").unwrap().read_u8().unwrap(), [7; 100]);
    for index in 0..6 {
        assert_eq!(
            file.dataset(&format!("added{index}"))
                .unwrap()
                .read_u8()
                .unwrap(),
            [9; 100]
        );
    }
}

#[test]
fn append_ownership_publication_and_finalization_persist_their_final_sections() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reserve.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true),
    );
    builder.create_dataset("keep").with_u8_data(&[7; 100]);
    builder.create_dataset("hole").with_u8_data(&[3; 16_384]);
    builder.write(&path).unwrap();
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    engine.delete(&ObjectPathBuf::parse("hole")).unwrap();
    engine.commit().unwrap();
    let mut store = engine.store();
    let allocation = store.allocate(&[1; 4_096], AllocationClass::Raw).unwrap();
    let publication = store.publish_allocations().unwrap();
    publication.confirm(allocation).unwrap();
    let published = typed_sections(&engine);
    assert_manager_allocations_owned(&engine);
    engine.force_sync().unwrap();
    let snapshot = dir.path().join("published_snapshot.h5");
    fs::write(&snapshot, issued_bytes(&engine)).unwrap();
    let reopened = WriteEngine::open_with_locking(&snapshot, FileLocking::Enabled).unwrap();
    assert_eq!(typed_sections(&reopened), published);
    drop(reopened);
    engine.finalize_persist().unwrap();
    let finalized = typed_sections(&engine);
    let bytes = issued_bytes(&engine);
    engine.finalize_persist().unwrap();
    assert_eq!(issued_bytes(&engine), bytes);
    drop(engine);
    let engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    assert_eq!(typed_sections(&engine), finalized);
    drop(engine);
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("keep")
            .unwrap()
            .read_u8()
            .unwrap(),
        [7; 100]
    );
}

#[test]
fn recovered_metadata_tail_can_hold_a_rebuilt_header() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reuse.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true),
    );
    builder.create_dataset("keep").with_u8_data(&[7; 100]);
    builder.write(&path).unwrap();
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    engine.space = hdf5_pure_space::__private::SessionSpace::from_strategy(
        Some(FileSpaceStrategy::Page),
        1,
        hdf5_pure_core::FileSpacePageSize::DEFAULT,
    );
    let root = engine
        .context
        .addresses()
        .resolve(engine.context.superblock().root_group_address)
        .unwrap();
    let before = issued_bytes(&engine);
    disk_log::start();
    engine
        .commit_persisting(root, ReturnedExtents::default(), TailPlacement::ReuseOnly)
        .unwrap();
    assert!(disk_log::take().is_empty());
    assert_eq!(issued_bytes(&engine), before);
    engine
        .commit_persisting(root, ReturnedExtents::default(), TailPlacement::Anywhere)
        .unwrap();
    let end = engine.persist.as_ref().unwrap().old_blocks[0].end();
    let tail = engine
        .space
        .paged_sections()
        .unwrap()
        .metadata()
        .iter()
        .copied()
        .find(|extent| extent.start() == end)
        .unwrap();
    engine.force_sync().unwrap();
    drop(engine);
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    engine
        .set_group_attr(
            &ObjectPathBuf::parse("/"),
            "large",
            AttrValue::U8Array(vec![1; 700]),
        )
        .unwrap();
    engine.commit().unwrap();
    let root = engine.context.superblock().root_group_address;
    assert!((tail.start()..tail.end()).contains(&root));
    assert_manager_allocations_owned(&engine);
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.root().attrs().unwrap()["large"],
        AttrValue::U8Array(vec![1; 700])
    );
    assert_eq!(file.dataset("keep").unwrap().read_u8().unwrap(), [7; 100]);
}

fn typed_sections(engine: &WriteEngine) -> Vec<(ManagerKind, Vec<Extent>)> {
    ManagerSections::paged(
        &engine.space.paged_sections().unwrap(),
        engine.space.page_size().unwrap(),
    )
    .into_iter()
    .map(|manager| (manager.kind(), manager.sections().to_vec()))
    .collect()
}

fn assert_manager_allocations_owned(engine: &WriteEngine) {
    let ownership = engine.space_ownership().unwrap();
    assert!(ownership.analysis_complete);
    let free = engine.space.reusable_sections().collect::<Vec<_>>();
    for block in &engine.persist.as_ref().unwrap().old_blocks {
        assert!(
            free.iter()
                .all(|extent| extent.end() <= block.start() || extent.start() >= block.end())
        );
    }
}

#[rstest]
#[case::metadata(false, false)]
#[case::section_list(false, true)]
#[case::publication(true, false)]
fn manager_metadata_failures_preserve_ownership_and_reopen_the_issued_state(
    #[case] publication: bool,
    #[case] section_list: bool,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] sync: SyncPolicy,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("failure.h5");
    let control_path = dir.path().join("control.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true),
    );
    builder.create_dataset("keep").with_u8_data(&[7; 100]);
    builder.create_dataset("replace").with_u8_data(&[3; 100]);
    builder.create_dataset("remove").with_u8_data(&[5; 100]);
    builder.write(&path).unwrap();
    fs::copy(&path, &control_path).unwrap();
    let mut control = WriteEngine::open_with_locking(&control_path, FileLocking::Enabled).unwrap();
    control.set_sync_policy(sync);
    control
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    stage_batch(&mut control);
    disk_log::start();
    control.commit().unwrap();
    let writes: Vec<_> = disk_log::take()
        .into_iter()
        .filter_map(|operation| {
            if let DiskOp::Write { offset, bytes } = operation {
                Some((offset, bytes))
            } else {
                None
            }
        })
        .collect();
    let target = if publication {
        control.context.signature().get()
    } else if section_list {
        writes
            .iter()
            .find(|(_, bytes)| bytes.starts_with(free_space::SECTIONS_SIGNATURE))
            .unwrap()
            .0
    } else {
        control
            .context
            .addresses()
            .resolve(
                control
                    .context
                    .superblock()
                    .superblock_extension_address
                    .unwrap(),
            )
            .unwrap()
            .get()
    };
    let index = writes
        .iter()
        .position(|(offset, _)| *offset == target)
        .unwrap();
    assert!(writes[index].1.starts_with(if publication {
        &HDF5_SIGNATURE
    } else if section_list {
        free_space::SECTIONS_SIGNATURE
    } else {
        object_header::SIGNATURE
    }));
    drop(control);
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    engine.set_sync_policy(sync);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let before = typed_sections(&engine);
    stage_batch(&mut engine);
    disk_log::start();
    disk_log::fail_write_after(
        index,
        if publication {
            usize::MAX
        } else if section_list {
            17
        } else {
            0
        },
    );
    let error = engine.commit().unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected an injected write error, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    let issued: Vec<_> = disk_log::take()
        .into_iter()
        .filter_map(|operation| {
            if let DiskOp::Write { offset, bytes } = operation {
                Some((offset, bytes))
            } else {
                None
            }
        })
        .collect();
    if publication {
        assert_eq!(issued.last().unwrap().0, target);
        assert_eq!(issued.last().unwrap().1, writes[index].1);
    } else {
        assert_eq!(typed_sections(&engine), before);
        assert!(
            issued
                .iter()
                .all(|(offset, _)| *offset != engine.context.signature().get())
        );
        let previous = File::from_bytes(issued_bytes(&engine)).unwrap();
        assert_eq!(
            previous.dataset("keep").unwrap().read_u8().unwrap(),
            [7; 100]
        );
        assert_eq!(
            previous.dataset("replace").unwrap().read_u8().unwrap(),
            [3; 100]
        );
        assert_eq!(
            previous.dataset("remove").unwrap().read_u8().unwrap(),
            [5; 100]
        );
        let error = previous.dataset("added").unwrap_err();
        assert!(
            matches!(error, Error::Format(FormatError::PathNotFound(ref path)) if path == "added"),
            "{error:?}"
        );
        stage_batch(&mut engine);
        engine.commit().unwrap();
    }
    drop(engine);
    let engine = edit_test_util::open_engine(&path);
    assert!(engine.space_ownership().unwrap().analysis_complete);
    drop(engine);
    let file = File::open_rw(&path).unwrap();
    assert_eq!(file.dataset("keep").unwrap().read_u8().unwrap(), [7; 100]);
    assert_eq!(
        file.dataset("replace").unwrap().read_u8().unwrap(),
        [17; 100]
    );
    assert_eq!(file.dataset("added").unwrap().read_u8().unwrap(), [9; 100]);
    let error = file.dataset("remove").unwrap_err();
    assert!(
        matches!(error, Error::Format(FormatError::PathNotFound(ref path)) if path == "remove"),
        "{error:?}"
    );
    let expected = file.reusable_free_sections().unwrap();
    file.close().unwrap();
    let file = File::open_rw(&path).unwrap();
    assert_eq!(file.reusable_free_sections().unwrap(), expected);
}

fn stage_batch(engine: &mut WriteEngine) {
    let mut dataset = DatasetBuilder::new("");
    dataset.with_u8_data(&[9; 100]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("added"), dataset)
        .unwrap();
    let mut dataset = DatasetBuilder::new("");
    dataset.with_u8_data(&[17; 100]);
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("replace"), dataset)
        .unwrap();
    engine.delete(&ObjectPathBuf::parse("remove")).unwrap();
}

fn issued_bytes(engine: &WriteEngine) -> Vec<u8> {
    let mut bytes = vec![0; usize::try_from(engine.image.len()).unwrap()];
    engine.image.read_disk_at(0, &mut bytes).unwrap();
    bytes
}

#[test]
fn large_measured_manager_lists_do_not_require_page_proportional_header_padding() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("large-manager.h5");
    fragmented_page_file(&path, FragmentedSections::Large);
    let mut engine = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Bounded,
        PersistedFreeSpacePolicy::Strict,
    )
    .unwrap();
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    assert_eq!(
        engine.space.admitted_snapshot().reusable_bytes().unwrap(),
        1_600_000
    );
    let root = engine
        .context
        .addresses()
        .resolve(engine.context.superblock().root_group_address)
        .unwrap();
    engine
        .commit_persisting(root, ReturnedExtents::default(), TailPlacement::Anywhere)
        .unwrap();
    let (chunks, owned) = persistence_allocations(&engine);
    assert!(chunks.len() > 1);
    assert!(chunks.iter().map(|chunk| chunk.len()).sum::<u64>() < 4_096);
    assert!(owned.iter().any(|block| block.len() > 14_000_000));
    assert_manager_allocations_owned(&engine);
    engine.force_sync().unwrap();
    drop(engine);
    let engine = edit_test_util::open_engine(&path);
    let (reopened_chunks, reopened_owned) = persistence_allocations(&engine);
    assert_eq!(reopened_chunks, chunks);
    assert_eq!(normalized_owned(reopened_owned), normalized_owned(owned));
}

#[rstest]
#[case::bounded(MemoryStrategy::Bounded)]
#[case::mirrored(MemoryStrategy::Mirrored)]
fn extension_capacity_is_recovered_and_reclaimed_across_repeated_commits(
    #[case] memory: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("owned-extension.h5");
    fragmented_page_file(&path, FragmentedSections::Continued);
    let open = || {
        WriteEngine::open_rw_with_strategy(
            &path,
            MetadataCacheConfig::disabled(),
            FileLocking::Enabled,
            memory,
            PersistedFreeSpacePolicy::Strict,
        )
        .unwrap()
    };
    let mut engine = open();
    let initial_extension = engine
        .context
        .addresses()
        .resolve(
            engine
                .context
                .superblock()
                .superblock_extension_address
                .unwrap(),
        )
        .unwrap();
    let initial_region = WriteEngine::gather_oh_messages(
        &engine.image(),
        initial_extension.get(),
        engine.context.addresses(),
    )
    .unwrap();
    let minimum = edit_header::build_v2_object_header(&initial_region)
        .unwrap()
        .len();
    let mut rows = Vec::new();
    for round in 0..6 {
        let previous = engine.persist.as_ref().unwrap().old_blocks.clone();
        let root = engine
            .context
            .addresses()
            .resolve(engine.context.superblock().root_group_address)
            .unwrap();
        engine
            .commit_persisting(root, ReturnedExtents::default(), TailPlacement::Anywhere)
            .unwrap();
        let (actual_chunks, owned) = persistence_allocations(&engine);
        let owned_bytes = owned.iter().map(|block| block.len()).sum::<u64>();
        let address = engine
            .context
            .addresses()
            .resolve(
                engine
                    .context
                    .superblock()
                    .superblock_extension_address
                    .unwrap(),
            )
            .unwrap();
        let chunks =
            edit_header::read_oh_chunks(&engine.image(), address.get(), engine.context.addresses())
                .unwrap();
        let extension = chunks.iter().map(|chunk| chunk.span.length()).sum::<u64>();
        assert!(chunks.len() > 1);
        assert!(extension < 131_072);
        assert_eq!(chunks.len(), actual_chunks.len());
        assert_eq!(
            normalized_owned(engine.persist.as_ref().unwrap().old_blocks.clone()),
            normalized_owned(owned.clone())
        );
        let region = WriteEngine::gather_oh_messages(
            &engine.image(),
            address.get(),
            engine.context.addresses(),
        )
        .unwrap()
        .without_nil()
        .unwrap();
        assert_eq!(
            edit_header::build_v2_object_header(&region).unwrap().len(),
            minimum
        );
        let eoa = engine.end_of_allocation().unwrap();
        let mut free = engine.space.reusable_sections().collect::<Vec<_>>();
        free.sort_by_key(|extent| extent.start());
        for old in previous {
            let mut start = old.start();
            let end = old.end().min(eoa);
            for available in &free {
                if available.start() <= start && available.end() > start {
                    start = available.end().min(end);
                }
            }
            assert!(start >= end);
        }
        rows.push([
            Value::Number(round),
            Value::Number(u64::try_from(chunks.len()).unwrap()),
            Value::Number(extension),
            Value::Number(owned_bytes),
            Value::Number(eoa.get()),
        ]);
        engine.force_sync().unwrap();
        drop(engine);
        engine = open();
        assert_eq!(
            normalized_owned(engine.persist.as_ref().unwrap().old_blocks.clone()),
            normalized_owned(owned)
        );
        assert_manager_allocations_owned(&engine);
    }
    drop(engine);
    let mut report = Report::new();
    report.table(
        "owned PAGE metadata",
        [
            "commit",
            "extension chunks",
            "extension bytes",
            "owned metadata bytes",
            "EOA",
        ],
        rows,
    );
    insta::assert_snapshot!("owned_page_extension", report.finish());
}

#[rstest]
#[case::bounded(MemoryStrategy::Bounded)]
#[case::mirrored(MemoryStrategy::Mirrored)]
fn a_page_publication_reads_each_protected_extension_chunk_once(#[case] memory: MemoryStrategy) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("extension-read-cost.h5");
    fragmented_page_file(&path, FragmentedSections::Continued);
    let reads = Arc::new(AtomicU64::new(0));
    let calls = Arc::new(AtomicU64::new(0));
    let mut engine = WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, len| {
        let image: Box<dyn FileImage> = match memory {
            MemoryStrategy::Bounded => Box::new(HandleImage::new(
                handle,
                len,
                MetadataCacheConfig::disabled(),
            )),
            MemoryStrategy::Mirrored => Box::new(WriteEngine::read_mirror(handle)?),
            MemoryStrategy::Auto => panic!("read accounting requires an explicit backing"),
        };
        Ok(Box::new(
            CountingImage::new(image, reads.clone(), Arc::default()).with_read_calls(calls.clone()),
        ))
    })
    .unwrap();
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let root = engine
        .context
        .addresses()
        .resolve(engine.context.superblock().root_group_address)
        .unwrap();
    engine
        .commit_persisting_paged(
            root,
            ReturnedExtents::default(),
            TailPlacement::Anywhere,
            PublicationPurpose::Commit,
        )
        .unwrap();
    let extension = engine
        .context
        .addresses()
        .resolve(
            engine
                .context
                .superblock()
                .superblock_extension_address
                .unwrap(),
        )
        .unwrap();
    let chunks =
        edit_header::read_oh_chunks(engine.image(), extension.get(), engine.context.addresses())
            .unwrap();
    assert!(chunks.len() > 1);
    let source_bytes = chunks.iter().map(|chunk| chunk.span.length()).sum::<u64>()
        + u64::try_from(hdf5_pure_format::__private::OBJECT_HEADER_PREFIX_MAX_LEN).unwrap();
    reads.store(0, Ordering::Relaxed);
    calls.store(0, Ordering::Relaxed);
    engine
        .commit_persisting_paged(
            root,
            ReturnedExtents::default(),
            TailPlacement::Anywhere,
            PublicationPurpose::Commit,
        )
        .unwrap();
    assert_eq!(
        calls.load(Ordering::Relaxed),
        u64::try_from(chunks.len() + 1).unwrap()
    );
    assert_eq!(reads.load(Ordering::Relaxed), source_bytes);
    eprintln!(
        "{memory:?}: predecessor {} chunks, {} source calls, {} source bytes",
        chunks.len(),
        calls.load(Ordering::Relaxed),
        source_bytes
    );
    assert_manager_allocations_owned(&engine);
    engine.force_sync().unwrap();
    drop(engine);
    let reopened = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        memory,
        PersistedFreeSpacePolicy::Strict,
    )
    .unwrap();
    assert!(reopened.persist.as_ref().unwrap().old_blocks.len() > chunks.len());
    assert_manager_allocations_owned(&reopened);
    drop(reopened);
}

#[rstest]
#[case::operation(false)]
#[case::session(true)]
fn page_manager_emission_orders_large_lists_before_later_continuations(
    #[case] session: bool,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("manager-emission.h5");
    let page = 131_072;
    fragmented_page_file_with_page(&path, FragmentedSections::Continued, page);
    let mut engine = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        memory,
        PersistedFreeSpacePolicy::Strict,
    )
    .unwrap();
    engine.set_sync_policy(SyncPolicy::OnClose);
    engine
        .set_write_buffering(if session {
            WriteBuffering::Session {
                page_size: page,
                max_bytes: 65_536,
            }
        } else {
            WriteBuffering::Operation {
                page_size: page,
                max_bytes: 65_536,
            }
        })
        .unwrap();
    let before = engine.image.buffering_work();
    let initial_len = engine.image.len();
    crate::edit::observations::reset_zeros();
    let root = engine
        .context
        .addresses()
        .resolve(engine.context.superblock().root_group_address)
        .unwrap();
    engine
        .commit_persisting(root, ReturnedExtents::default(), TailPlacement::Anywhere)
        .unwrap();
    let pending = engine.image.buffering_work();
    assert!(pending.pending_bytes <= 65_536);
    engine.image.flush().unwrap();
    let after = engine.image.buffering_work();
    let zeros = crate::edit::observations::zero_work();
    let (chunks, owned) = persistence_allocations(&engine);
    assert!(chunks.len() > 1);
    let list = owned.iter().find(|block| block.len() > 65_536).unwrap();
    assert!(chunks.iter().any(|chunk| chunk.start() >= list.end()));
    assert!(zeros.bytes > 0);
    assert!(zeros.scratch_peak > 0);
    assert!(zeros.scratch_peak <= 65_536);
    assert_eq!(after.clean_gap_calls - before.clean_gap_calls, 0);
    assert_eq!(after.clean_gap_bytes - before.clean_gap_bytes, 0);
    assert!(after.flush_peak_bytes <= 65_536);
    let appended = engine.image.len() - initial_len;
    let owned_bytes = owned.iter().map(|block| block.len()).sum::<u64>();
    assert_eq!(appended, owned_bytes + zeros.bytes);
    assert!(after.write_bytes - before.write_bytes >= appended);
    assert!(after.write_bytes - before.write_bytes <= appended + 4_096);
    assert!(after.copied_bytes - before.copied_bytes <= 2 * appended);
    assert!(after.copied_bytes > before.copied_bytes);
    match memory {
        MemoryStrategy::Bounded => assert_eq!(after.mirror_bytes, 0),
        MemoryStrategy::Mirrored => assert_eq!(after.mirror_bytes, engine.image.len()),
        MemoryStrategy::Auto => panic!("the case must select an explicit backing"),
    }
    eprintln!(
        "{memory:?} session={session}: appended={appended}, owned={owned_bytes}, zeros={zeros:?}, work={after:?}, before={before:?}"
    );
    assert_manager_allocations_owned(&engine);
}

#[derive(Clone, Copy)]
enum GappedFailure {
    Continuation,
    Gap,
    Publication,
}

#[rstest]
#[case::gap(GappedFailure::Gap)]
#[case::continuation(GappedFailure::Continuation)]
#[case::publication(GappedFailure::Publication)]
fn gapped_page_publication_failures_preserve_predecessor_and_reserved_ownership(
    #[case] failure: GappedFailure,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] sync: SyncPolicy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("gapped-failure.h5");
    let control_path = directory.path().join("gapped-control.h5");
    fragmented_page_file(&path, FragmentedSections::Continued);
    fs::copy(&path, &control_path).unwrap();
    let prepare = |path: &Path| {
        let mut engine = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
        engine.set_sync_policy(sync);
        engine
            .set_write_buffering(WriteBuffering::Unbuffered)
            .unwrap();
        engine
            .set_group_attr(
                &ObjectPathBuf::root(),
                "audit",
                AttrValue::U8Array(vec![19; 800]),
            )
            .unwrap();
        engine
    };
    let mut control = prepare(&control_path);
    disk_log::start();
    control.commit().unwrap();
    let writes = disk_log::take()
        .into_iter()
        .filter_map(|operation| match operation {
            DiskOp::Write { offset, bytes } => Some((offset, bytes)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let (chunks, owned) = persistence_allocations(&control);
    let target = match failure {
        GappedFailure::Gap => {
            let ranges = normalized_owned(owned);
            let gap = ranges
                .windows(2)
                .find(|pair| pair[0].end() < pair[1].start())
                .unwrap();
            gap[0].end().get()
        }
        GappedFailure::Continuation => chunks[1].start().get(),
        GappedFailure::Publication => control.context.signature().get(),
    };
    let index = writes
        .iter()
        .position(|(offset, _)| *offset == target)
        .unwrap();
    match failure {
        GappedFailure::Gap => assert!(writes[index].1.iter().all(|byte| *byte == 0)),
        GappedFailure::Continuation => assert!(
            writes[index]
                .1
                .starts_with(object_header::CONTINUATION_SIGNATURE)
        ),
        GappedFailure::Publication => assert!(writes[index].1.starts_with(&HDF5_SIGNATURE)),
    }
    drop(control);
    let mut engine = prepare(&path);
    let previous = typed_sections(&engine);
    let old_blocks = engine.persist.as_ref().unwrap().old_blocks.clone();
    disk_log::start();
    disk_log::fail_write_after(
        index,
        if matches!(failure, GappedFailure::Publication) {
            usize::MAX
        } else {
            0
        },
    );
    let error = engine.commit().unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected injected I/O failure, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    disk_log::take();
    if !matches!(failure, GappedFailure::Publication) {
        assert_eq!(typed_sections(&engine), previous);
        assert_eq!(engine.persist.as_ref().unwrap().old_blocks, old_blocks);
        let file = File::from_bytes(issued_bytes(&engine)).unwrap();
        assert_eq!(file.root().attrs().unwrap().get("audit"), None);
        engine
            .set_group_attr(
                &ObjectPathBuf::root(),
                "audit",
                AttrValue::U8Array(vec![19; 800]),
            )
            .unwrap();
        engine.commit().unwrap();
    }
    assert_manager_allocations_owned(&engine);
    if matches!(failure, GappedFailure::Publication) {
        let error = engine.force_sync().unwrap_err();
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{error:?}"
        );
    } else {
        engine.force_sync().unwrap();
    }
    drop(engine);
    let engine = edit_test_util::open_engine(&path);
    assert_manager_allocations_owned(&engine);
    persistence_allocations(&engine);
    drop(engine);
    assert_eq!(
        File::open(&path).unwrap().root().attrs().unwrap()["audit"],
        AttrValue::U8Array(vec![19; 800])
    );
}

fn persistence_allocations(engine: &WriteEngine) -> (Vec<Extent>, Vec<Extent>) {
    let address = engine
        .context
        .addresses()
        .resolve(
            engine
                .context
                .superblock()
                .superblock_extension_address
                .unwrap(),
        )
        .unwrap();
    let chunks =
        edit_header::read_oh_chunks(engine.image(), address.get(), engine.context.addresses())
            .unwrap()
            .into_iter()
            .map(|chunk| {
                Extent::new(
                    engine
                        .context
                        .addresses()
                        .relative(chunk.span.start())
                        .unwrap(),
                    chunk.span.length(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
    let file = File::from_bytes(issued_bytes(engine)).unwrap();
    let info = file.file_space_info().unwrap().unwrap();
    let (_, managers) = crate::free_space_manager::read_persisted_sections_source(
        &engine.image(),
        engine.context.widths(),
        engine.context.addresses().effective_base(),
        &info.manager_addrs,
    )
    .unwrap();
    let mut owned = chunks.clone();
    owned.extend(managers.into_iter().map(|range| {
        Extent::new(
            engine.context.addresses().relative(range.start()).unwrap(),
            range.length(),
        )
        .unwrap()
    }));
    let page = info.page_size.get();
    for block in &owned {
        if block.len() < page {
            assert_eq!(
                block.start().get() / page,
                (block.end().get() - 1) / page,
                "{block:?}"
            );
        } else {
            assert_eq!(block.start().get() % page, 0, "{block:?}");
        }
    }
    (chunks, owned)
}

fn normalized_owned(mut owned: Vec<Extent>) -> Vec<Extent> {
    owned.sort_unstable();
    FreeList::from_sorted(owned).unwrap().extents().to_vec()
}

fn fragmented_page_file(path: &Path, scale: FragmentedSections) {
    fragmented_page_file_with_page(path, scale, 4_096);
}

fn fragmented_page_file_with_page(path: &Path, scale: FragmentedSections, page: u64) {
    let count: u64 = match scale {
        FragmentedSections::Continued => 8_000,
        FragmentedSections::Large => 1_600_000,
    };
    let mut fixture = PersistedImage::new(path, PersistedStrategy::Page, 1);
    let file = File::open_streaming(path).unwrap();
    let mut superblock = file.superblock().clone();
    file.close().unwrap();
    let sections = (0..count)
        .map(|index| ((1 << 24) + index * 2, 1))
        .collect::<Vec<_>>();
    let eof = ((1 << 24) + count * 2).div_ceil(page) * page;
    fixture.info.page_size = FileSpacePageSize::try_from(page).unwrap();
    fixture
        .bytes
        .resize(usize::try_from(count * 10 + 16_384).unwrap(), 0);
    fixture.manager(ManagerSlot::Metadata, &sections);
    fixture.info.eoa_pre_fsm = Some(DecodedStoredAddress::Defined(StoredAddress::new(eof)));
    let mut image = fixture.write(path);
    superblock.eof_address = eof;
    let encoded = hdf5_pure_format::__private::serialize_superblock(&superblock).unwrap();
    image[..encoded.len()].copy_from_slice(&encoded);
    fs::write(path, image).unwrap();
    drop(sections);
    fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_len(eof)
        .unwrap();
}

enum FragmentedSections {
    Continued,
    Large,
}
