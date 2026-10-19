//! Tests non-persistent PAGE allocation and root publication through the write engine.

use std::fs;

use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::PageType;
use rstest::rstest;
use test_util::disk_log::{self, DiskOp};

use crate::FileLocking;
use crate::FileSpaceStrategy;
use crate::Layout;
use crate::SyncPolicy;
use crate::edit;
use crate::edit::WriteEngine;
use crate::file_writer;
use crate::object_path::LinkNameBuf;
use crate::reader::File;
use crate::shared_message::DatatypeLocation;
use crate::superblock;
use crate::type_builders::DatasetBuilder;
use crate::writer::FileBuilder;

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
