//! Tests non-persistent PAGE allocation and root publication through the write engine.

use std::collections::BTreeSet;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::PageType;
use rstest::rstest;
use test_util::checksum::CHECKSUM;
use test_util::disk_log::{self, DiskOp};
use test_util::object_header::v2 as object_header_bytes;
use test_util::publication;
use test_util_hdf5::userblock;

use crate::FileSpaceSettings;
use crate::PersistedFreeSpacePolicy;
use crate::UserblockSize;
use crate::access_mode::AccessMode;
use crate::address::{
    AddressContext, BaseAddress, BaseAddressExt, StoredAddress, StoredAddressExt,
};
use crate::edit;
use crate::edit::WriteEngine;
use crate::error::{Error, FormatError};
use crate::file_space_info::FileSpacePageSize;
use crate::file_writer;
use crate::image::WriteBuffering;
use crate::object_path::{LinkNameBuf, ObjectPathBuf};
use crate::page_publication_failure::{
    BarrierRecord, PagePublicationFailure as Failure, PublicationFailureImage,
};
use crate::reader::File;
use crate::shared_message::DatatypeLocation;
use crate::source::MetadataCacheConfig;
use crate::superblock;
use crate::type_builders::{AttrValue, DatasetBuilder};
use crate::writer::FileBuilder;
use crate::{FileLocking, FileSpaceStrategy, Layout, MemoryStrategy, SyncPolicy};

#[rstest]
#[case::plain(4096, 0, 0, SyncPolicy::Always)]
#[case::userblock(4096, 4096, 0, SyncPolicy::Always)]
#[case::larger_pages(8192, 8192, 0, SyncPolicy::OnClose)]
#[case::physical_slack(4096, 4096, 37, SyncPolicy::Always)]
fn fresh_page_placement_and_root_publication_preserve_old_allocations(
    #[case] page_size: u64,
    #[case] userblock: u64,
    #[case] slack: usize,
    #[case] sync: SyncPolicy,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fresh_pages.h5");
    let mut builder = FileBuilder::new();
    builder
        .with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page)
                .with_page_size(FileSpacePageSize::try_from(page_size).unwrap()),
        )
        .with_userblock(UserblockSize::try_from(userblock).unwrap());
    if userblock > 0 {
        builder.with_userblock_content(b"preserve this userblock");
    }
    builder.create_dataset("keep").with_u8_data(&[7; 64]);
    builder.write(&path).unwrap();
    let mut original = fs::read(&path).unwrap();
    let stored = superblock::read(original.as_slice()).unwrap();
    let old_stored_eoa = stored
        .superblock
        .eof_address
        .checked_sub(userblock)
        .unwrap();
    assert_eq!(old_stored_eoa % page_size, 0);
    original.extend(vec![0xA5; slack]);
    fs::write(&path, &original).unwrap();
    let file = File::from_bytes(original.clone()).unwrap();
    let Layout::Contiguous {
        address: Some(old_raw),
        size: old_raw_len,
    } = file.dataset("keep").unwrap().layout().unwrap()
    else {
        panic!("expected allocated contiguous storage");
    };
    drop(file);

    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    engine.sync_policy = sync;
    assert_eq!(engine.context.signature().get(), userblock);
    assert_eq!(engine.context.addresses().effective_base().get(), userblock);
    assert_eq!(
        engine.context.superblock().eof_address,
        userblock + old_stored_eoa
    );
    assert_eq!(engine.image.len(), original.len() as u64);
    assert!(engine.persist.is_none());
    assert!(engine.aggregation.is_none());
    assert!(engine.space.reusable_sections().next().is_none());
    let (_, info) = WriteEngine::extension_fsinfo(&engine.image(), &engine.context)
        .unwrap()
        .unwrap();
    assert_eq!(info.strategy, FileSpaceStrategy::Page);
    assert_eq!(info.page_size.get(), page_size);
    assert!(!info.persist);
    assert!(info.manager_addrs.is_empty());

    let boundary = engine
        .end_of_allocation()
        .unwrap()
        .get()
        .checked_next_multiple_of(page_size)
        .unwrap();
    assert!(boundary >= old_stored_eoa);
    let old_root = engine
        .context
        .addresses()
        .resolve(engine.context.superblock().root_group_address)
        .unwrap()
        .get();
    let root_spans = engine.oh_chunk_spans(old_root).unwrap();
    disk_log::start();
    let (raw, header, root) = engine.place_dataset_and_root();
    assert!(raw >= boundary);
    assert_eq!(raw % page_size, 0);
    assert_ne!(raw / page_size, header / page_size);
    assert_eq!(header / page_size, root / page_size);
    engine.pad_to_page().unwrap();
    assert_eq!(engine.end_of_allocation().unwrap().get() % page_size, 0);
    let freed = engine.stored_spans(
        root_spans
            .into_iter()
            .map(|range| (range, FreeClass::Page(PageType::Meta)))
            .chain([(
                FileRange::new(PhysicalAddress::new(old_raw), old_raw_len).unwrap(),
                FreeClass::Page(PageType::Raw),
            )])
            .collect(),
    );
    assert_eq!(engine.release_freed(freed.unwrap()).unwrap(), None);
    assert!(engine.space.reusable_sections().next().is_none());
    engine
        .publish_nonpersisting(PhysicalAddress::new(userblock + root), None)
        .unwrap();
    engine.finalize_persist().unwrap();
    let operations = disk_log::take();
    let superblock_len =
        hdf5_pure_format::__private::serialize_superblock(engine.context.superblock())
            .unwrap()
            .len();
    assert_publication_prefixes(&original, &operations, userblock, superblock_len);
    drop(engine);

    let after = fs::read(&path).unwrap();
    let publication = usize::try_from(userblock).unwrap();
    assert_eq!(&after[..publication], &original[..publication]);
    assert_eq!(
        &after[publication + superblock_len..original.len()],
        &original[publication + superblock_len..]
    );
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("keep").unwrap().read_u8().unwrap(), [7; 64]);
    assert_eq!(file.dataset("added").unwrap().read_u8().unwrap(), [9; 64]);
    assert_eq!(file.file_space_info().unwrap().unwrap(), info);
    assert_eq!(file.persisted_free_sections().unwrap(), None);
    drop(file);
    let mut reopened = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    assert!(reopened.space.reusable_sections().next().is_none());
    assert!(reopened.persist.is_none());
    let placement = reopened.reserve(64, AllocationClass::Raw).unwrap();
    assert!(placement.address() >= after.len() as u64);
    assert_eq!((placement.address() - userblock) % page_size, 0);
    assert!(reopened.space.reusable_sections().next().is_none());
}

impl WriteEngine {
    /// Writes a dataset and replacement root.
    ///
    /// Returns the stored addresses of the raw data, dataset header, and root header, in that order.
    fn place_dataset_and_root(&mut self) -> (u64, u64, u64) {
        let mut builder = DatasetBuilder::new("added");
        builder.with_u8_data(&[9; 64]);
        let dataset =
            edit::dataset::flatten_dataset(builder, LinkNameBuf::new("added").unwrap()).unwrap();
        let raw = self
            .alloc_or_append_typed(dataset.storage.raw(), AllocationClass::Raw)
            .unwrap();
        let header = file_writer::build_dataset_oh(
            &dataset.dt,
            &DatatypeLocation::Inline,
            &dataset.ds,
            self.context
                .addresses()
                .relative(PhysicalAddress::new(raw))
                .unwrap(),
            dataset.storage.raw().len() as u64,
            &dataset.attrs,
            None,
            dataset.fill_time,
            dataset.fill.as_deref(),
            self.libver(),
        )
        .unwrap();
        let header = self
            .alloc_or_append_typed(&header, AllocationClass::Metadata)
            .unwrap();
        let old_root = self
            .context
            .addresses()
            .resolve(self.context.superblock().root_group_address)
            .unwrap()
            .get();
        let mut region =
            WriteEngine::gather_oh_messages(&self.image(), old_root, self.context.addresses())
                .unwrap();
        region.push_link(
            "added",
            self.context
                .addresses()
                .relative(PhysicalAddress::new(header))
                .unwrap(),
            None,
        );
        let root = self
            .alloc_or_append_typed(
                &edit::object_header::build_v2_object_header(&region).unwrap(),
                AllocationClass::Metadata,
            )
            .unwrap();
        (
            self.context
                .addresses()
                .relative(PhysicalAddress::new(raw))
                .unwrap()
                .get(),
            self.context
                .addresses()
                .relative(PhysicalAddress::new(header))
                .unwrap()
                .get(),
            self.context
                .addresses()
                .relative(PhysicalAddress::new(root))
                .unwrap()
                .get(),
        )
    }
}

fn assert_publication_prefixes(
    original: &[u8],
    operations: &[DiskOp],
    base: u64,
    superblock_len: usize,
) {
    let mut replay = original.to_vec();
    let mut published = false;
    let mut publications = 0;
    assert!(!operations.is_empty());
    for operation in operations {
        match operation {
            DiskOp::Write { offset, bytes } => {
                if *offset < original.len() as u64 {
                    assert_eq!(*offset, base);
                    assert_eq!(bytes.len(), superblock_len);
                    published = true;
                    publications += 1;
                } else {
                    assert!(!published, "content follows publication");
                }
            }
            DiskOp::SetLen(len) => {
                assert!(*len >= original.len() as u64);
            }
        }
        operation.apply(&mut replay);
        let file = File::from_bytes(replay.clone()).unwrap();
        assert_eq!(file.dataset("keep").unwrap().read_u8().unwrap(), [7; 64]);
        assert_eq!(
            file.root().datasets().unwrap(),
            if published {
                vec!["keep", "added"]
            } else {
                vec!["keep"]
            }
        );
        if published {
            assert_eq!(file.dataset("added").unwrap().read_u8().unwrap(), [9; 64]);
        }
    }
    assert_eq!(publications, 1);
}

#[rstest]
#[case::plain(4_096, 0, SyncPolicy::Always)]
#[case::userblock(4_096, 4_096, SyncPolicy::Always)]
#[case::subpage_userblock(4_096, 512, SyncPolicy::Always)]
#[case::larger_pages(8_192, 0, SyncPolicy::OnClose)]
fn staged_commits_preserve_preexisting_bytes_outside_the_superblock(
    #[case] page_size: u64,
    #[case] userblock: u64,
    #[case] sync: SyncPolicy,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("staged.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_page_size(FileSpacePageSize::try_from(page_size).unwrap()),
    );
    builder.create_dataset("keep").with_u8_data(&[7; 64]);
    builder.create_dataset("replace").with_u8_data(&[3; 64]);
    builder
        .create_dataset("chunks")
        .with_u8_data(&[4; 64])
        .with_chunks(&[16]);
    builder.create_dataset("remove").with_u8_data(&[5; 64]);
    let mut group = builder.create_group("group");
    group.set_attr("attribute", AttrValue::I32(9));
    builder.add_group(group.finish());
    builder.write(&path).unwrap();
    if userblock != 0 {
        userblock::prepend(&path, &vec![0xA5; usize::try_from(userblock).unwrap()]);
    }
    let original = fs::read(&path).unwrap();
    let mut engine = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::default(),
        FileLocking::Enabled,
        MemoryStrategy::Auto,
        PersistedFreeSpacePolicy::Tolerant,
    )
    .unwrap();
    engine.set_sync_policy(sync);
    let mut replacement = DatasetBuilder::new("replace");
    replacement.with_u8_data(&[8; 64]);
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("replace"), replacement)
        .unwrap();
    let mut chunks = DatasetBuilder::new("chunks");
    chunks.with_u8_data(&[9; 64]);
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("chunks"), chunks)
        .unwrap();
    engine.delete(&ObjectPathBuf::parse("remove")).unwrap();
    stage_byte_dataset(&mut engine, "added", 6);
    engine
        .copy(
            &ObjectPathBuf::parse("keep"),
            &ObjectPathBuf::parse("copied"),
        )
        .unwrap();
    disk_log::start();
    engine.commit().unwrap();
    engine.finalize_persist().unwrap();
    let operations = disk_log::take();
    let superblock_len =
        hdf5_pure_format::__private::serialize_superblock(engine.context.superblock())
            .unwrap()
            .len();
    let owned = engine.space_ownership().unwrap();
    let boundary = original.len() as u64 - userblock;
    let raw_pages = owned
        .live_raw_space
        .iter()
        .filter(|&&(at, _)| at >= original.len() as u64)
        .flat_map(|&(at, len)| {
            (at - userblock) / page_size..=(at + len - 1 - userblock) / page_size
        })
        .collect::<BTreeSet<_>>();
    let chunks_addr = crate::group_v2::resolve_path_any_from_source(
        &engine.opened_source(),
        AccessMode::ReadWrite,
        engine.context.superblock(),
        engine.context.widths(),
        &ObjectPathBuf::parse("chunks").as_path(),
    )
    .unwrap();
    let index_spans = engine.chunked_index_spans(chunks_addr).unwrap();
    let metadata_pages = owned
        .live_metadata_space
        .iter()
        .filter(|&&(at, len)| {
            !index_spans.contains(&FileRange::new(PhysicalAddress::new(at), len).unwrap())
        })
        .filter(|&&(at, _)| at >= original.len() as u64)
        .flat_map(|&(at, len)| {
            (at - userblock) / page_size..=(at + len - 1 - userblock) / page_size
        })
        .collect::<BTreeSet<_>>();
    assert!(raw_pages.is_disjoint(&metadata_pages));
    assert!(
        index_spans
            .iter()
            .all(|range| raw_pages.contains(&((range.start().get() - userblock) / page_size)))
    );
    assert!(
        raw_pages
            .iter()
            .chain(&metadata_pages)
            .all(|&page| page * page_size >= boundary)
    );
    let mut replay = original.clone();
    let mut published = false;
    for operation in &operations {
        if let DiskOp::Write { offset, bytes } = operation {
            if *offset < original.len() as u64 {
                assert_eq!(*offset, userblock);
                assert_eq!(bytes.len(), superblock_len);
                published = true;
            }
        }
        operation.apply(&mut replay);
        let file = File::from_bytes(replay.clone()).unwrap();
        assert_eq!(file.dataset("keep").unwrap().read_u8().unwrap(), [7; 64]);
        assert_eq!(
            file.dataset("replace").unwrap().read_u8().unwrap(),
            [if published { 8 } else { 3 }; 64]
        );
        assert_eq!(
            file.dataset("chunks").unwrap().read_u8().unwrap(),
            [if published { 9 } else { 4 }; 64]
        );
    }
    drop(engine);
    let after = fs::read(&path).unwrap();
    assert_eq!(
        &after[..userblock as usize],
        &original[..userblock as usize]
    );
    assert_eq!(
        &after[userblock as usize + superblock_len..original.len()],
        &original[userblock as usize + superblock_len..]
    );
    let file = File::open_rw(&path).unwrap();
    file.root()
        .create_dataset("second", |dataset| {
            dataset.with_u8_data(&[11; 64]);
        })
        .unwrap();
    file.commit().unwrap();
    drop(file);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("second").unwrap().read_u8().unwrap(), [11; 64]);
    assert!(!file.file_space_info().unwrap().unwrap().persist);
    assert_eq!(file.persisted_free_sections().unwrap(), None);
    assert_eq!((file.superblock().eof_address - userblock) % page_size, 0);
    assert_eq!(
        file.root().datasets().unwrap(),
        ["keep", "replace", "chunks", "copied", "added", "second"]
    );
    assert_eq!(file.root().groups().unwrap(), ["group"]);
    assert_eq!(
        file.group("group")
            .unwrap()
            .attrs()
            .unwrap()
            .get("attribute"),
        Some(&AttrValue::I32(9))
    );
}

#[test]
fn reference_updates_are_rejected_before_publication() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("references.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Page));
    let group = builder.create_group("group");
    builder.add_group(group.finish());
    builder
        .create_dataset("references")
        .with_path_references(&["group"]);
    builder.write(&path).unwrap();
    let original = fs::read(&path).unwrap();
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    stage_byte_dataset(&mut engine, "added", 9);
    assert_preflight_refusal(&mut engine);
    assert_preflight_refusal(&mut engine);
    drop(engine);
    assert_eq!(fs::read(&path).unwrap(), original);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), ["references"]);
}

#[rstest]
#[case::reference_dataset(PendingPageEdit::Reference)]
#[case::dense_dataset(PendingPageEdit::DenseDataset)]
#[case::dense_group_edit(PendingPageEdit::DenseGroupEdit)]
#[case::dense_dataset_edit(PendingPageEdit::DenseDatasetEdit)]
#[case::dense_cross_copy(PendingPageEdit::DenseCopy)]
fn prospective_reference_storage_is_rejected_before_mutation(#[case] operation: PendingPageEdit) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prospective.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Page));
    builder.create_dataset("keep").with_u8_data(&[7; 64]);
    builder.write(&path).unwrap();
    let original = fs::read(&path).unwrap();
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    stage_byte_dataset(&mut engine, "added", 9);
    let mut dataset = DatasetBuilder::new("unsupported");
    match operation {
        PendingPageEdit::Reference => {
            dataset.with_path_references(&["keep"]);
            engine
                .stage_created_dataset(&ObjectPathBuf::parse("unsupported"), dataset)
                .unwrap();
        }
        PendingPageEdit::DenseDataset | PendingPageEdit::DenseCopy => {
            dataset.with_u8_data(&[7; 64]);
            for index in 0..9 {
                dataset.set_attr(&format!("attribute{index}"), AttrValue::I32(index));
            }
            if matches!(operation, PendingPageEdit::DenseDataset) {
                engine
                    .stage_created_dataset(&ObjectPathBuf::parse("unsupported"), dataset)
                    .unwrap();
            } else {
                let source_path = dir.path().join("source.h5");
                let mut source = FileBuilder::new();
                *source.create_dataset("unsupported") = dataset;
                source.write(&source_path).unwrap();
                engine
                    .copy_from(
                        &File::open(&source_path).unwrap(),
                        &ObjectPathBuf::parse("unsupported"),
                        &ObjectPathBuf::parse("copied"),
                    )
                    .unwrap();
            }
        }
        PendingPageEdit::DenseGroupEdit | PendingPageEdit::DenseDatasetEdit => {
            for index in 0..9 {
                let attribute = format!("attribute{index}");
                if matches!(operation, PendingPageEdit::DenseGroupEdit) {
                    engine
                        .set_group_attr(&ObjectPathBuf::root(), &attribute, AttrValue::I32(index))
                        .unwrap();
                } else {
                    engine
                        .set_dataset_attr(
                            &ObjectPathBuf::parse("keep"),
                            &attribute,
                            AttrValue::I32(index),
                        )
                        .unwrap();
                }
            }
        }
    }
    assert_preflight_refusal(&mut engine);
    assert_preflight_refusal(&mut engine);
    drop(engine);
    assert_eq!(fs::read(&path).unwrap(), original);
}

#[derive(Clone, Copy)]
enum PendingPageEdit {
    DenseCopy,
    DenseDataset,
    DenseDatasetEdit,
    DenseGroupEdit,
    Reference,
}

fn assert_preflight_refusal(engine: &mut WriteEngine) {
    let mark = engine.staged.snapshot();
    disk_log::start();
    let error = engine.commit().unwrap_err();
    assert!(
        matches!(
            error,
            Error::EditUnsupported(
                "staged edits of PAGE files without persisted free space require a complete \
             object-reference-free tree; repack the file to edit reference-bearing or \
             unsupported attribute storage"
            )
        ),
        "{error:?}"
    );
    assert!(engine.has_staged_edits());
    assert!(engine.staged.snapshot() == mark);
    assert!(!engine.publication.attempted());
    assert!(engine.space.reusable_sections().next().is_none());
    assert!(disk_log::take().is_empty());
}

fn stage_byte_dataset(engine: &mut WriteEngine, path: &str, value: u8) {
    let mut dataset = DatasetBuilder::new(path);
    dataset.with_u8_data(&[value; 64]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse(path), dataset)
        .unwrap();
}

#[rstest]
#[case::allocation(Failure::Append, 0)]
#[case::first_disk_write(Failure::DiskWrite(0, 0), 0)]
#[case::partial_disk_write(Failure::DiskWrite(0, 17), 0)]
#[case::completed_content_write(Failure::DiskWrite(0, usize::MAX), 0)]
#[case::publication(Failure::BeforePublish, 0)]
#[case::published(Failure::AfterPublish, 0)]
#[case::before_content_drain(Failure::BeforeDrain, 0)]
#[case::content_sync(Failure::ContentSync, 0)]
#[case::publication_sync(Failure::PublishedSync, 0)]
#[case::completed_publication(Failure::AfterDrain, 0)]
#[case::userblock_publication_sync(Failure::PublishedSync, 4_096)]
fn failed_staged_commits_keep_old_pages_unavailable(
    #[case] failure: Failure,
    #[case] userblock: u64,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] sync: SyncPolicy,
    #[values(false, true)] gathered: bool,
    #[values(false, true)] reopen: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("failed.h5");
    let original = write_page_failure_fixture(&path, userblock);
    let barriers = Arc::new(Mutex::new(Vec::new()));
    let mut engine = WriteEngine::open_with_page_failure(&path, failure, sync, gathered, &barriers);
    engine.stage_page_edits();
    let mut replay = engine.assert_page_commit_failure(&original, failure);
    failure.assert_barriers(sync, &barriers.lock().unwrap());
    assert!(engine.space.reusable_sections().next().is_none());
    if reopen {
        let interrupted = dir.path().join("interrupted.h5");
        fs::write(&interrupted, &replay).unwrap();
        assert_interrupted_reopen(&interrupted, &replay, sync, failure);
        disk_log::start();
        drop(engine);
        for operation in disk_log::take() {
            operation.apply(&mut replay);
        }
        assert_eq!(fs::read(&path).unwrap(), replay);
        engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        engine.set_sync_policy(sync);
        assert!(engine.space.reusable_sections().next().is_none());
    }
    let write_boundary = if reopen {
        replay.len() as u64
    } else {
        original.len() as u64
    };
    let failed_barriers = barriers.lock().unwrap().len();
    engine.assert_page_retry(&path, &original, replay, write_boundary, failure);
    if !reopen && failure.publishes() {
        assert_eq!(
            barriers.lock().unwrap().get(failed_barriers),
            Some(&BarrierRecord {
                durable: matches!(sync, SyncPolicy::Always),
                published: true,
                completed: true,
            })
        );
    }
}

#[rstest]
#[case::unbuffered_completed(false, usize::MAX)]
#[case::buffered_completed(true, usize::MAX)]
#[case::buffered_partial(true, 17)]
#[case::unbuffered_zero(false, 0)]
#[case::buffered_zero(true, 0)]
#[case::unbuffered_unchanged(false, 17)]
#[case::unbuffered_eoa(false, SUPERBLOCK_EOF_END)]
#[case::buffered_eoa(true, SUPERBLOCK_EOF_END)]
#[case::unbuffered_root(false, SUPERBLOCK_ROOT_END)]
#[case::buffered_root(true, SUPERBLOCK_ROOT_END)]
fn publication_write_errors_require_verified_session_bytes(
    #[case] gathered: bool,
    #[case] prefix: usize,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] sync: SyncPolicy,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("publication_write.h5");
    let original = write_page_failure_fixture(&path, 0);
    let control_path = dir.path().join("control.h5");
    fs::write(&control_path, &original).unwrap();
    let mut control = WriteEngine::open_with_locking(&control_path, FileLocking::Enabled).unwrap();
    control.stage_page_edits();
    control.commit().unwrap();
    let expected =
        hdf5_pure_format::__private::serialize_superblock(control.context.superblock()).unwrap();
    drop(control);
    let barriers = Arc::new(Mutex::new(Vec::new()));
    let mut engine = WriteEngine::open_with_page_failure(
        &path,
        Failure::PublicationWrite(prefix),
        sync,
        gathered,
        &barriers,
    );
    engine.stage_page_edits();
    disk_log::start();
    let error = engine.commit().unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected injected disk write failure, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    let mut replay = original.clone();
    for operation in disk_log::take() {
        operation.apply(&mut replay);
    }
    publication::assert_preserved_bytes(&original, &replay, 0, expected.len());
    let mut issued = vec![0; expected.len()];
    engine.image.read_disk_at(0, &mut issued).unwrap();
    assert_eq!(issued, replay[..expected.len()]);
    let original_superblock = &original[..expected.len()];
    let mixed = &replay[..expected.len()];
    let changes_addresses = prefix >= SUPERBLOCK_EOF_END;
    if matches!(prefix, SUPERBLOCK_EOF_END | SUPERBLOCK_ROOT_END) {
        assert_ne!(mixed, original_superblock);
        assert_ne!(mixed, expected);
        assert_ne!(
            &mixed[SUPERBLOCK_EOF_START..SUPERBLOCK_EOF_END],
            &original_superblock[SUPERBLOCK_EOF_START..SUPERBLOCK_EOF_END]
        );
        assert_eq!(
            &mixed[SUPERBLOCK_EOF_START..SUPERBLOCK_EOF_END],
            &expected[SUPERBLOCK_EOF_START..SUPERBLOCK_EOF_END]
        );
        assert_ne!(
            &expected[SUPERBLOCK_ROOT_START..SUPERBLOCK_ROOT_END],
            &original_superblock[SUPERBLOCK_ROOT_START..SUPERBLOCK_ROOT_END]
        );
        assert_eq!(
            &mixed[SUPERBLOCK_ROOT_START..SUPERBLOCK_ROOT_END],
            if prefix == SUPERBLOCK_EOF_END {
                &original_superblock[SUPERBLOCK_ROOT_START..SUPERBLOCK_ROOT_END]
            } else {
                &expected[SUPERBLOCK_ROOT_START..SUPERBLOCK_ROOT_END]
            },
        );
    }
    if !gathered && changes_addresses {
        engine.assert_page_publication_rejected();
        drop(engine);
        assert_eq!(fs::read(&path).unwrap(), replay);
        if matches!(prefix, SUPERBLOCK_EOF_END | SUPERBLOCK_ROOT_END) {
            #[cfg(feature = "checksum")]
            {
                let error = File::open(&path).unwrap_err();
                let Error::Format(FormatError::ChecksumMismatch {
                    expected: stored,
                    computed,
                }) = error
                else {
                    panic!("expected mixed-superblock checksum error, got {error:?}");
                };
                let end = expected.len() - CHECKSUM;
                assert_eq!(stored, u32::from_le_bytes(mixed[end..].try_into().unwrap()));
                assert_eq!(computed, crate::checksum::jenkins_lookup3(&mixed[..end]));
                assert_ne!(stored, computed);
            }
            return;
        }
        let file = File::open(&path).unwrap();
        Failure::AfterPublish.assert_page_datasets(&file, None);
        drop(file);
        engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        engine.set_sync_policy(sync);
        let boundary = replay.len() as u64;
        engine.assert_page_retry(&path, &original, replay, boundary, Failure::AfterPublish);
        return;
    }
    engine.assert_page_retry(
        &path,
        &original,
        replay,
        original.len() as u64,
        if gathered {
            Failure::AfterPublish
        } else {
            Failure::BeforePublish
        },
    );
}

/// The start of the end-of-file address in a version 2 or 3 superblock with eight-byte addresses.
///
/// The EOF and root fields follow the base and extension addresses, as defined in
/// "Format Signature and Superblock" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super
const SUPERBLOCK_EOF_START: usize = 28;
const SUPERBLOCK_EOF_END: usize = 36;
const SUPERBLOCK_ROOT_START: usize = SUPERBLOCK_EOF_END;
const SUPERBLOCK_ROOT_END: usize = 44;

#[rstest]
#[case::barrier(Failure::RepeatedPublishedSync)]
#[case::read(Failure::PublishedRead)]
fn failed_publication_recovery_preserves_staged_edits(
    #[case] failure: Failure,
    #[values(false, true)] gathered: bool,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] sync: SyncPolicy,
    #[values(false, true)] explicit_sync: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("recovery_barrier.h5");
    let original = write_page_failure_fixture(&path, 0);
    let barriers = Arc::new(Mutex::new(Vec::new()));
    let mut engine = WriteEngine::open_with_page_failure(&path, failure, sync, gathered, &barriers);
    engine.stage_page_edits();
    let published = engine.assert_page_commit_failure(&original, failure);
    if matches!(failure, Failure::RepeatedPublishedSync) {
        assert_eq!(
            engine.context.superblock().root_group_address,
            superblock::read(published.as_slice())
                .unwrap()
                .superblock
                .root_group_address
        );
    }
    stage_byte_dataset(&mut engine, "second", 11);
    assert_recovery_preserves_staging(&mut engine, failure, explicit_sync);
    assert_recovery_preserves_staging(&mut engine, failure, explicit_sync);
    assert_eq!(
        barriers.lock().unwrap().last().unwrap().durable,
        explicit_sync || matches!(sync, SyncPolicy::Always),
    );
    drop(engine);
    assert_eq!(fs::read(&path).unwrap(), published);
    let file = File::open(&path).unwrap();
    Failure::AfterPublish.assert_page_datasets(&file, None);
}

#[rstest]
fn explicit_sync_clears_pending_page_publication_and_preserves_staged_edits(
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] sync: SyncPolicy,
    #[values(false, true)] gathered: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("explicit_sync.h5");
    let original = write_page_failure_fixture(&path, 0);
    let barriers = Arc::new(Mutex::new(Vec::new()));
    let mut engine =
        WriteEngine::open_with_page_failure(&path, Failure::AfterDrain, sync, gathered, &barriers);
    engine.stage_page_edits();
    let published = engine.assert_page_commit_failure(&original, Failure::AfterDrain);
    stage_byte_dataset(&mut engine, "second", 11);
    let mark = engine.staged.snapshot();
    assert!(engine.publication.attempted());
    disk_log::start();
    engine.force_sync().unwrap();
    assert_eq!(
        *engine.context.superblock(),
        superblock::read(published.as_slice()).unwrap().superblock
    );
    assert!(!engine.publication.attempted());
    assert!(engine.staged.snapshot() == mark);
    assert!(disk_log::take().is_empty());
    let records = barriers.lock().unwrap();
    assert_eq!(
        records.last(),
        Some(&BarrierRecord {
            durable: true,
            published: true,
            completed: true,
        })
    );
    drop(records);
    drop(engine);
    assert_eq!(fs::read(&path).unwrap(), published);
    Failure::AfterDrain.assert_page_datasets(&File::open(&path).unwrap(), None);
}

fn assert_recovery_preserves_staging(
    engine: &mut WriteEngine,
    failure: Failure,
    explicit_sync: bool,
) {
    let mark = engine.staged.snapshot();
    disk_log::start();
    let error = if explicit_sync {
        engine.force_sync()
    } else {
        engine.commit()
    }
    .unwrap_err();
    match (failure, error) {
        (Failure::RepeatedPublishedSync, Error::Io(error)) => {
            assert_eq!(error.kind(), ErrorKind::Other);
            assert_eq!(error.to_string(), "injected PAGE publication failure");
        }
        (Failure::PublishedRead, Error::Format(FormatError::Source(reason))) => {
            assert_eq!(reason, "injected PAGE publication read failure");
        }
        (_, error) => panic!("expected injected PAGE recovery failure, got {error:?}"),
    }
    assert!(engine.publication.attempted());
    assert!(engine.staged.snapshot() == mark);
    assert!(engine.space.reusable_sections().next().is_none());
    assert!(disk_log::take().is_empty());
}

/// Creates datasets for preservation, replacement, and deletion, and returns the file bytes.
fn write_page_failure_fixture(path: &Path, userblock: u64) -> Vec<u8> {
    let mut builder = FileBuilder::new();
    builder
        .with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Page))
        .with_userblock(UserblockSize::try_from(userblock).unwrap());
    if userblock > 0 {
        builder.with_userblock_content(b"preserve the userblock across publication recovery");
    }
    builder.create_dataset("keep").with_u8_data(&[7; 64]);
    builder.create_dataset("replace").with_u8_data(&[3; 64]);
    builder.create_dataset("remove").with_u8_data(&[5; 64]);
    builder.write(path).unwrap();
    fs::read(path).unwrap()
}

impl WriteEngine {
    /// Opens a mirror with injected I/O failures and records its barrier attempts.
    fn open_with_page_failure(
        path: &Path,
        failure: Failure,
        sync: SyncPolicy,
        gathered: bool,
        barriers: &Arc<Mutex<Vec<BarrierRecord>>>,
    ) -> Self {
        let mut engine = Self::open_imaged(path, Some(FileLocking::Enabled), |handle, _| {
            let inner = WriteEngine::read_mirror(handle)?;
            let publication_offset = superblock::read(&inner)?.address.get();
            Ok(Box::new(PublicationFailureImage::with_barriers(
                inner,
                publication_offset,
                failure,
                Arc::clone(barriers),
            )))
        })
        .unwrap();
        engine.set_sync_policy(sync);
        if !gathered {
            engine
                .set_write_buffering(WriteBuffering::Unbuffered)
                .unwrap();
        }
        engine
    }

    /// Stages creation, replacement, and deletion against the failure fixture.
    fn stage_page_edits(&mut self) {
        stage_byte_dataset(self, "added", 9);
        let mut replacement = DatasetBuilder::new("replace");
        replacement.with_u8_data(&[17; 64]);
        self.stage_dataset_write(&ObjectPathBuf::parse("replace"), replacement)
            .unwrap();
        self.delete(&ObjectPathBuf::parse("remove")).unwrap();
    }

    /// Checks the injected error and replays recorded writes to read the published state.
    fn assert_page_commit_failure(&mut self, original: &[u8], failure: Failure) -> Vec<u8> {
        disk_log::start();
        if let Failure::DiskWrite(completed, prefix) = failure {
            disk_log::fail_write_after(completed, prefix);
        }
        let before = self.context.superblock().clone();
        let error = self.commit().unwrap_err();
        let Error::Io(error) = error else {
            panic!("expected injected I/O failure, got {error:?}");
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(
            error.to_string(),
            if matches!(failure, Failure::DiskWrite(..)) {
                "injected disk write failure"
            } else {
                "injected PAGE publication failure"
            }
        );
        let mut replay = original.to_vec();
        let mut published = false;
        for operation in disk_log::take() {
            if let DiskOp::Write { offset, .. } = &operation {
                published |= *offset == self.context.signature().get();
            }
            operation.apply(&mut replay);
        }
        assert_eq!(published, failure.publishes());
        if failure.publishes() && !matches!(failure, Failure::PublishedRead) {
            assert_eq!(
                *self.context.superblock(),
                superblock::read(replay.as_slice()).unwrap().superblock
            );
        } else {
            assert_eq!(self.context.superblock(), &before);
        }
        let file = File::from_bytes(replay.clone()).unwrap();
        failure.assert_page_datasets(&file, None);
        drop(file);
        replay
    }

    /// Checks that a publication mismatch preserves staged edits and issues no writes.
    fn assert_page_publication_rejected(&mut self) {
        stage_byte_dataset(self, "second", 11);
        self.assert_page_recovery_rejected(false);
        self.assert_page_recovery_rejected(true);
    }

    fn assert_page_recovery_rejected(&mut self, explicit_sync: bool) {
        let before = self.context.superblock().clone();
        let mark = self.staged.snapshot();
        disk_log::start();
        let error = if explicit_sync {
            self.force_sync()
        } else {
            self.commit()
        }
        .unwrap_err();
        assert!(
            matches!(error, Error::EditUnsupported(reason)
                if reason == "PAGE publication bytes differ from the session image after a failed write; close and reopen the file before editing again"),
            "{error:?}"
        );
        assert_eq!(self.context.superblock(), &before);
        assert!(self.staged.snapshot() == mark);
        assert!(self.space.reusable_sections().next().is_none());
        assert!(disk_log::take().is_empty());
    }

    /// Commits another dataset and checks that old bytes outside the superblock are preserved.
    fn assert_page_retry(
        mut self,
        path: &Path,
        original: &[u8],
        mut replay: Vec<u8>,
        write_boundary: u64,
        failure: Failure,
    ) {
        let sb_len = hdf5_pure_format::__private::serialize_superblock(self.context.superblock())
            .unwrap()
            .len();
        let accepted = self
            .image
            .read_exact_at(self.context.signature().get(), sb_len)
            .unwrap();
        let mut issued = vec![0; sb_len];
        self.image
            .read_disk_at(self.context.signature().get(), &mut issued)
            .unwrap();
        let pending = self.publication.attempted() && issued != accepted;
        stage_byte_dataset(&mut self, "second", 11);
        disk_log::start();
        self.commit().unwrap();
        let operations = disk_log::take();
        if pending {
            let Some(DiskOp::Write { offset, bytes }) = operations.first() else {
                panic!("expected pending superblock write before later content");
            };
            assert_eq!(*offset, self.context.signature().get());
            assert_eq!(*bytes, accepted);
        }
        for operation in operations {
            if let DiskOp::Write { offset, .. } = &operation {
                assert!(*offset == self.context.signature().get() || *offset >= write_boundary);
            }
            operation.apply(&mut replay);
        }
        assert!(self.space.reusable_sections().next().is_none());
        let sb_start = usize::try_from(self.context.signature().get()).unwrap();
        drop(self);
        let bytes = fs::read(path).unwrap();
        assert_eq!(bytes, replay);
        publication::assert_preserved_bytes(original, &bytes, sb_start, sb_len);
        let file = File::open(path).unwrap();
        failure.assert_page_datasets(&file, Some(("second", 11)));
        assert!(!file.file_space_info().unwrap().unwrap().persist);
        assert!(
            file.file_space_info()
                .unwrap()
                .unwrap()
                .manager_addrs
                .is_empty()
        );
        assert_eq!(file.persisted_free_sections().unwrap(), None);
    }
}

/// Opens the interrupted file and checks that another commit preserves its existing allocations.
fn assert_interrupted_reopen(path: &Path, interrupted: &[u8], sync: SyncPolicy, failure: Failure) {
    let mut engine = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
    engine.set_sync_policy(sync);
    assert!(engine.space.reusable_sections().next().is_none());
    let boundary = engine
        .end_of_allocation()
        .unwrap()
        .get()
        .max(interrupted.len() as u64)
        .next_multiple_of(4_096);
    stage_byte_dataset(&mut engine, "reopened", 23);
    disk_log::start();
    engine.commit().unwrap();
    let sb_len = hdf5_pure_format::__private::serialize_superblock(engine.context.superblock())
        .unwrap()
        .len();
    for operation in disk_log::take() {
        match operation {
            DiskOp::Write { offset, .. } => {
                assert!(
                    offset == engine.context.signature().get()
                        || offset >= interrupted.len() as u64
                )
            }
            DiskOp::SetLen(len) => assert!(len >= interrupted.len() as u64),
        }
    }
    assert!(engine.space.reusable_sections().next().is_none());
    let sb_start = usize::try_from(engine.context.signature().get()).unwrap();
    drop(engine);
    let after = fs::read(path).unwrap();
    publication::assert_preserved_bytes(interrupted, &after, sb_start, sb_len);
    let file = File::open(path).unwrap();
    failure.assert_page_datasets(&file, Some(("reopened", 23)));
    let Layout::Contiguous {
        address: Some(raw), ..
    } = file.dataset("reopened").unwrap().layout().unwrap()
    else {
        panic!("expected contiguous storage")
    };
    assert!(raw >= boundary);
    assert!(!file.file_space_info().unwrap().unwrap().persist);
    assert!(
        file.file_space_info()
            .unwrap()
            .unwrap()
            .manager_addrs
            .is_empty()
    );
    assert_eq!(file.persisted_free_sections().unwrap(), None);
}

impl Failure {
    /// Checks dataset membership and values against the expected publication outcome.
    fn assert_page_datasets(self, file: &File, additional: Option<(&str, u8)>) {
        let mut expected = if self.publishes() {
            vec![("keep", 7), ("replace", 17), ("added", 9)]
        } else {
            vec![("keep", 7), ("replace", 3), ("remove", 5)]
        };
        expected.extend(additional);
        assert_eq!(
            file.root().datasets().unwrap(),
            expected.iter().map(|(path, _)| *path).collect::<Vec<_>>()
        );
        for (path, value) in expected {
            assert_eq!(file.dataset(path).unwrap().read_u8().unwrap(), [value; 64]);
        }
    }

    fn publishes(self) -> bool {
        matches!(
            self,
            Self::AfterPublish
                | Self::PublishedSync
                | Self::PublishedRead
                | Self::AfterDrain
                | Self::RepeatedPublishedSync
        )
    }

    /// Checks barrier durability and publication order, including completion at the failure phase.
    fn assert_barriers(self, sync: SyncPolicy, records: &[BarrierRecord]) {
        assert!(
            records
                .iter()
                .all(|record| record.durable == matches!(sync, SyncPolicy::Always))
        );
        if matches!(
            self,
            Failure::BeforeDrain
                | Failure::ContentSync
                | Failure::PublishedSync
                | Failure::AfterDrain
        ) {
            assert!(!records.is_empty());
        }
        if matches!(self, Failure::PublishedSync | Failure::AfterDrain) {
            assert_eq!(
                records.first(),
                Some(&BarrierRecord {
                    durable: matches!(sync, SyncPolicy::Always),
                    published: false,
                    completed: true
                })
            );
            assert_eq!(
                records.last(),
                Some(&BarrierRecord {
                    durable: matches!(sync, SyncPolicy::Always),
                    published: true,
                    completed: matches!(self, Failure::AfterDrain)
                })
            );
        }
    }
}

#[rstest]
#[case::before_superblock(1)]
#[case::beyond_physical_extent(16_384)]
#[case::before_live_data(4_096)]
fn invalid_page_eoa_is_rejected_before_mutation(#[case] eoa: u64) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("invalid_eoa.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Page));
    builder.create_dataset("keep").with_u8_data(&[7; 64]);
    builder.write(&path).unwrap();
    let mut bytes = fs::read(&path).unwrap();
    let header_at = bytes
        .windows(object_header_bytes::SIGNATURE.len())
        .rposition(|window| window == object_header_bytes::SIGNATURE)
        .unwrap();
    let prefix = object_header_bytes::Header::new().build();
    let width = object_header_bytes::HeaderFlags::default().chunk_size_width();
    let size_at = prefix.len() - CHECKSUM - width;
    assert_eq!(&bytes[header_at..header_at + size_at], &prefix[..size_at]);
    let message_size = usize::try_from(test_util::bytes::uint_at(
        &bytes,
        header_at + size_at,
        width,
    ))
    .unwrap();
    let metadata_end =
        u64::try_from(header_at + size_at + width + message_size + CHECKSUM).unwrap();
    let mut sb = superblock::read(bytes.as_slice()).unwrap().superblock;
    sb.eof_address = eoa;
    let encoded = hdf5_pure_format::__private::serialize_superblock(&sb).unwrap();
    bytes[..encoded.len()].copy_from_slice(&encoded);
    fs::write(&path, &bytes).unwrap();
    let error = WriteEngine::open_with_locking(&path, FileLocking::Enabled)
        .err()
        .unwrap();
    if eoa == 1 {
        assert!(
            matches!(
                error,
                Error::InvalidSpaceOwnershipExtent {
                    address: 0,
                    length: 48,
                    previous_end: 0,
                    logical_size: 1
                }
            ),
            "{error:?}"
        );
    } else if eoa == 4_096 {
        assert!(
            matches!(
                error,
                Error::InvalidSpaceOwnershipExtent {
                    address: 4_096,
                    length: 64,
                    logical_size: 4_096,
                    previous_end,
                } if previous_end == metadata_end
            ),
            "{error:?}"
        );
    } else {
        assert!(
            matches!(
                error,
                Error::Format(FormatError::TruncatedFile {
                    expected_eof: 16_384,
                    actual_len: 8_192
                })
            ),
            "{error:?}"
        );
    }
    assert_eq!(fs::read(&path).unwrap(), bytes);
}

#[rstest]
#[case::stored_alignment(0, u64::MAX, u64::MAX, u64::MAX, 4_096)]
#[case::physical_resolution(4_096, u64::MAX - 4_096, u64::MAX, u64::MAX - 4_095, 4_096)]
fn overflowing_page_boundary_is_rejected(
    #[case] base: u64,
    #[case] eoa: u64,
    #[case] physical_len: u64,
    #[case] expected_offset: u64,
    #[case] expected_length: u64,
) {
    let error = WriteEngine::fresh_page_boundary(
        AddressContext::from_effective_base(BaseAddress::new(base)),
        StoredAddress::new(eoa),
        physical_len,
        FileSpacePageSize::DEFAULT,
    )
    .unwrap_err();
    let FormatError::OffsetOverflow { offset, length } = error else {
        panic!("expected OffsetOverflow, got {error:?}");
    };
    assert_eq!((offset, length), (expected_offset, expected_length));
}

#[test]
fn nonpersistent_page_swmr_is_rejected_before_mutation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("swmr.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Page));
    builder.write(&path).unwrap();
    let original = fs::read(&path).unwrap();
    let error = File::open_swmr_writer(&path).unwrap_err();
    assert!(
        matches!(
            error,
            Error::SwmrAppendUnsupported(
                "SWMR writing of PAGE files without persisted free space is unsupported; \
             use File::open_rw with staged edits and File::commit"
            )
        ),
        "{error:?}"
    );
    assert_eq!(fs::read(&path).unwrap(), original);
}
