//! Tests non-persistent PAGE allocation and root publication through the write engine.

use std::collections::BTreeSet;
use std::fs;
use std::io::Error as IoError;
use std::io::ErrorKind;

use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::PageType;
use rstest::rstest;
use test_util::disk_log::{self, DiskOp};

use crate::access_mode::AccessMode;
use crate::address::{
    AddressContext, BaseAddress, BaseAddressExt, StoredAddress, StoredAddressExt,
};
use crate::edit;
use crate::edit::WriteEngine;
use crate::error::{Error, FormatError};
use crate::file_space_info::FileSpacePageSize;
use crate::file_writer;
use crate::image::{FileImage, MirrorImage, WriteBuffering};
use crate::object_path::{LinkNameBuf, ObjectPathBuf};
use crate::reader::File;
use crate::shared_message::DatatypeLocation;
use crate::source::{MetadataCacheConfig, Source};
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
        .with_file_space_strategy(FileSpaceStrategy::Page, false, 1)
        .with_file_space_page_size(page_size)
        .with_userblock(userblock);
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
    assert_eq!(engine.sb_sig_off, userblock);
    assert_eq!(engine.addresses.effective_base().get(), userblock);
    assert_eq!(engine.superblock.eof_address, userblock + old_stored_eoa);
    assert_eq!(engine.image.len(), original.len() as u64);
    assert!(engine.persist.is_none());
    assert!(engine.aggregation.is_none());
    assert!(engine.space.reusable_sections().next().is_none());
    let (_, info) = WriteEngine::extension_fsinfo(
        &engine.image(),
        &engine.superblock,
        engine.widths,
        engine.addresses,
    )
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
        .addresses
        .absolute(engine.superblock.root_group_address)
        .unwrap();
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
            .map(|(addr, len)| (addr, len, FreeClass::Page(PageType::Meta)))
            .chain([(old_raw, old_raw_len, FreeClass::Page(PageType::Raw))])
            .collect(),
    );
    assert_eq!(engine.release_freed(freed.unwrap()).unwrap(), None);
    assert!(engine.space.reusable_sections().next().is_none());
    engine
        .publish_nonpersisting(userblock + root, None)
        .unwrap();
    engine.finalize_persist().unwrap();
    let operations = disk_log::take();
    let superblock_len = hdf5_pure_format::__private::serialize_superblock(&engine.superblock)
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
    assert_eq!(file.file_space_info().unwrap(), &info);
    assert_eq!(file.persisted_free_space().unwrap(), []);
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
        let dataset = edit::flatten_dataset(builder, LinkNameBuf::new("added").unwrap()).unwrap();
        let raw = self
            .alloc_or_append_typed(&dataset.raw, AllocationClass::Raw)
            .unwrap();
        let header = file_writer::build_dataset_oh(
            &dataset.dt,
            &DatatypeLocation::Inline,
            &dataset.ds,
            self.addresses.relative(raw).unwrap(),
            dataset.raw.len() as u64,
            &dataset.attrs,
            None,
            dataset.fill_time.unwrap_or_default(),
            dataset.fill.as_deref(),
            self.libver(),
        )
        .unwrap();
        let header = self
            .alloc_or_append_typed(&header, AllocationClass::Metadata)
            .unwrap();
        let old_root = self
            .addresses
            .absolute(self.superblock.root_group_address)
            .unwrap();
        let mut region =
            WriteEngine::gather_oh_messages(&self.image(), old_root, self.addresses).unwrap();
        region.push_link("added", self.addresses.relative(header).unwrap(), None);
        let root = self
            .alloc_or_append_typed(
                &edit::build_v2_object_header(&region).unwrap(),
                AllocationClass::Metadata,
            )
            .unwrap();
        (
            self.addresses.relative(raw).unwrap().get(),
            self.addresses.relative(header).unwrap().get(),
            self.addresses.relative(root).unwrap().get(),
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
    builder
        .with_file_space_strategy(FileSpaceStrategy::Page, false, 1)
        .with_file_space_page_size(page_size);
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
        let original = fs::read(&path).unwrap();
        let mut sb = superblock::read(original.as_slice()).unwrap().superblock;
        sb.base_address = BaseAddress::new(userblock);
        sb.eof_address += userblock;
        let mut bytes = vec![0xA5; userblock as usize];
        bytes.extend_from_slice(&original);
        let encoded = hdf5_pure_format::__private::serialize_superblock(&sb).unwrap();
        bytes[userblock as usize..userblock as usize + encoded.len()].copy_from_slice(&encoded);
        fs::write(&path, bytes).unwrap();
    }
    let original = fs::read(&path).unwrap();
    let mut engine = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::default(),
        FileLocking::Enabled,
        MemoryStrategy::Auto,
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
    let superblock_len = hdf5_pure_format::__private::serialize_superblock(&engine.superblock)
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
        &engine.image(),
        AccessMode::ReadWrite,
        &engine.superblock,
        engine.widths,
        engine.addresses,
        &ObjectPathBuf::parse("chunks").as_path(),
    )
    .unwrap();
    let index_spans = engine.chunked_index_spans(chunks_addr).unwrap();
    let metadata_pages = owned
        .live_metadata_space
        .iter()
        .filter(|span| !index_spans.contains(span))
        .filter(|&&(at, _)| at >= original.len() as u64)
        .flat_map(|&(at, len)| {
            (at - userblock) / page_size..=(at + len - 1 - userblock) / page_size
        })
        .collect::<BTreeSet<_>>();
    assert!(raw_pages.is_disjoint(&metadata_pages));
    assert!(
        index_spans
            .iter()
            .all(|&(at, _)| raw_pages.contains(&((at - userblock) / page_size)))
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
    assert!(!file.file_space_info().unwrap().persist);
    assert_eq!(file.persisted_free_space().unwrap(), []);
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
    builder.with_file_space_strategy(FileSpaceStrategy::Page, false, 1);
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
    builder.with_file_space_strategy(FileSpaceStrategy::Page, false, 1);
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
    let mark = engine.staged.mark();
    disk_log::start();
    let error = engine.commit().unwrap_err();
    assert!(
        matches!(error, Error::EditUnsupported(reason) if reason.contains("object-reference-free tree")),
        "{error:?}"
    );
    assert!(engine.has_staged_edits());
    assert!(engine.staged.mark() == mark);
    assert!(!engine.publish_attempted);
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
#[case::allocation(Failure::Append)]
#[case::publication(Failure::BeforePublish)]
#[case::published(Failure::AfterPublish)]
#[case::content_sync(Failure::FirstSync)]
#[case::publication_sync(Failure::PublishedSync)]
fn failed_staged_commits_keep_old_pages_unavailable(#[case] failure: Failure) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("failed.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_strategy(FileSpaceStrategy::Page, false, 1);
    builder.create_dataset("keep").with_u8_data(&[7; 64]);
    builder.write(&path).unwrap();
    let original = fs::read(&path).unwrap();
    let mut engine = WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, _| {
        Ok(Box::new(FailingImage {
            inner: WriteEngine::read_mirror(handle)?,
            failure,
            failed: false,
            published: false,
        }))
    })
    .unwrap();
    stage_byte_dataset(&mut engine, "added", 9);
    disk_log::start();
    let error = engine.commit().unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected injected I/O failure, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected PAGE commit failure");
    let mut replay = original.clone();
    for operation in disk_log::take() {
        operation.apply(&mut replay);
    }
    let file = File::from_bytes(replay.clone()).unwrap();
    assert_eq!(file.dataset("keep").unwrap().read_u8().unwrap(), [7; 64]);
    assert_eq!(
        file.root().datasets().unwrap(),
        if matches!(failure, Failure::AfterPublish | Failure::PublishedSync) {
            vec!["keep", "added"]
        } else {
            vec!["keep"]
        }
    );
    stage_byte_dataset(&mut engine, "second", 11);
    disk_log::start();
    engine.commit().unwrap();
    for operation in disk_log::take() {
        operation.apply(&mut replay);
    }
    assert!(engine.space.reusable_sections().next().is_none());
    let sb_len = hdf5_pure_format::__private::serialize_superblock(&engine.superblock)
        .unwrap()
        .len();
    drop(engine);
    let bytes = fs::read(&path).unwrap();
    assert_eq!(bytes, replay);
    assert_eq!(&bytes[sb_len..original.len()], &original[sb_len..]);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("keep").unwrap().read_u8().unwrap(), [7; 64]);
    assert_eq!(file.dataset("second").unwrap().read_u8().unwrap(), [11; 64]);
    assert_eq!(file.persisted_free_space().unwrap(), []);
}

#[derive(Clone, Copy)]
enum Failure {
    AfterPublish,
    Append,
    BeforePublish,
    FirstSync,
    PublishedSync,
}

struct FailingImage {
    inner: MirrorImage,
    failure: Failure,
    failed: bool,
    published: bool,
}

impl FailingImage {
    fn fail(&mut self) -> Result<(), Error> {
        self.failed = true;
        Err(IoError::other("injected PAGE commit failure").into())
    }
}

impl FileImage for FailingImage {
    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        if !self.failed && matches!(self.failure, Failure::Append) {
            self.inner.append(&bytes[..bytes.len() / 2])?;
            self.fail()?;
        }
        self.inner.append(bytes)
    }
    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        if !self.failed && offset == 0 && matches!(self.failure, Failure::BeforePublish) {
            self.fail()?;
        }
        self.inner.write_at(offset, bytes)?;
        if offset == 0 {
            self.published = true;
            if !self.failed && matches!(self.failure, Failure::AfterPublish) {
                self.fail()?;
            }
        }
        Ok(())
    }
    fn truncate(&mut self, _len: u64) -> Result<(), Error> {
        Err(IoError::other("unexpected PAGE truncation").into())
    }
    fn sync_data(&mut self) -> Result<(), Error> {
        self.inner.sync_data()
    }
    fn sync_all(&mut self) -> Result<(), Error> {
        if !self.failed
            && (matches!(self.failure, Failure::FirstSync)
                || (self.published && matches!(self.failure, Failure::PublishedSync)))
        {
            self.fail()?;
        }
        self.inner.sync_all()
    }
    fn ordering_barrier(&mut self) -> Result<(), Error> {
        self.inner.ordering_barrier()
    }
    fn set_write_buffering(&mut self, _mode: WriteBuffering) -> Result<(), Error> {
        self.inner.set_write_buffering(WriteBuffering::Unbuffered)
    }
    fn issued_write_order(&self) -> Vec<(u64, u64)> {
        self.inner.issued_write_order()
    }
}

impl Source for FailingImage {
    fn len(&self) -> u64 {
        self.inner.len()
    }
    fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        self.inner.read_at(offset, bytes)
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
    builder.with_file_space_strategy(FileSpaceStrategy::Page, false, 1);
    builder.create_dataset("keep").with_u8_data(&[7; 64]);
    builder.write(&path).unwrap();
    let mut bytes = fs::read(&path).unwrap();
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
                    ..
                }
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
    builder.with_file_space_strategy(FileSpaceStrategy::Page, false, 1);
    builder.write(&path).unwrap();
    let original = fs::read(&path).unwrap();
    let error = File::open_swmr_writer(&path).unwrap_err();
    assert!(
        matches!(error, Error::SwmrAppendUnsupported(reason) if reason.contains("File::commit")),
        "{error:?}"
    );
    assert_eq!(fs::read(&path).unwrap(), original);
}
