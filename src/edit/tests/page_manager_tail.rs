use std::fs;
use std::io::ErrorKind;

use hdf5_pure_format::__private::HDF5_SIGNATURE;
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::{Extent, ManagerKind, ManagerSections};
use rstest::rstest;
use test_util::disk_log::{self, DiskOp};
use test_util::free_space;
use test_util::object_header::v2 as object_header;

use crate::FileSpacePageSize;
#[cfg(test)]
use crate::FileSpaceSettings;
use crate::FormatError;
use crate::UserblockSize;
use crate::chunk_index_inplace::Store;
use crate::edit::test_util as edit_test_util;
use crate::edit::{TailPlacement, WriteEngine};
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;
use crate::type_builders::DatasetBuilder;
use crate::{AttrValue, Error, File, FileBuilder, FileLocking, FileSpaceStrategy, SyncPolicy};

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
        .commit_persisting(root, Vec::new(), TailPlacement::ReuseOnly)
        .unwrap();
    assert!(disk_log::take().is_empty());
    assert_eq!(issued_bytes(&engine), before);
    engine
        .commit_persisting(root, Vec::new(), TailPlacement::Anywhere)
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
            .absolute(
                control
                    .context
                    .superblock()
                    .superblock_extension_address
                    .unwrap(),
            )
            .unwrap()
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
