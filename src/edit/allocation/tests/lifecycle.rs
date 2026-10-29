use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use hdf5_pure_space::__private::AllocationClass;
use rstest::rstest;
use test_util::disk_log::{self, DiskOp};

use super::persistent::{self, PersistentStrategy};
use crate::chunk_index_inplace::Store;
use crate::edit::WriteEngine;
use crate::image::WriteBuffering;
use crate::{Error, File, FormatError, MemoryStrategy, SyncPolicy};

#[rstest]
#[case::flat_mirrored(PersistentStrategy::Flat, MemoryStrategy::Mirrored, 8192)]
#[case::flat_bounded(PersistentStrategy::Flat, MemoryStrategy::Bounded, 8192)]
#[case::page_mirrored(PersistentStrategy::Page, MemoryStrategy::Mirrored, 8192)]
#[case::page_bounded(PersistentStrategy::Page, MemoryStrategy::Bounded, 8192)]
#[case::flat_mirrored_eoa(PersistentStrategy::Flat, MemoryStrategy::Mirrored, 131_072)]
#[case::flat_bounded_eoa(PersistentStrategy::Flat, MemoryStrategy::Bounded, 131_072)]
#[case::page_mirrored_eoa(PersistentStrategy::Page, MemoryStrategy::Mirrored, 131_072)]
#[case::page_bounded_eoa(PersistentStrategy::Page, MemoryStrategy::Bounded, 131_072)]
fn unpublished_manager_writes_are_discarded_before_allocation_reuse(
    #[case] strategy: PersistentStrategy,
    #[case] backing: MemoryStrategy,
    #[case] size: usize,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("canceled.h5");
    persistent::fixture(&path, strategy);
    let before = fs::read(&path).unwrap();
    let mut engine = open(&path, backing, SyncPolicy::OnClose);
    let prior_length = engine.image.len();
    let original = engine
        .store()
        .allocate(&vec![47; size], AllocationClass::Raw)
        .unwrap();
    let range = original.range();
    disk_log::fail_write_after(0, 0);
    assert_injected(engine.store().publish_allocations().err().unwrap());
    assert_eq!(engine.disk_bytes().unwrap(), before);
    assert_eq!(engine.image.len(), prior_length);
    if size > 65_536 {
        assert!(range.start().get() >= prior_length);
    }
    let replacement = engine
        .store()
        .allocate(&vec![83; size], AllocationClass::Raw)
        .unwrap();
    assert_eq!(replacement.range(), range);
    let publication = engine.store().publish_allocations().unwrap();
    let confirmed = publication.confirm(replacement).unwrap();
    assert_eq!(
        engine.context.allocations.require(&confirmed).unwrap(),
        range
    );
    let error = publication.confirm(original).err().unwrap();
    let Error::Format(FormatError::Internal(reason)) = error else {
        panic!("expected canceled generation rejection, got {error:?}");
    };
    assert_eq!(
        reason,
        "allocation belongs to another session or publication"
    );
    let mut payload = vec![0; usize::try_from(range.length()).unwrap()];
    engine
        .image
        .read_disk_at(range.start().get(), &mut payload)
        .unwrap();
    assert_eq!(payload, vec![83; size]);
    let issued = engine.disk_bytes().unwrap();
    disk_log::start();
    drop(engine);
    assert!(disk_log::take().is_empty());
    assert_eq!(fs::read(&path).unwrap(), issued);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("values").unwrap().read_i32().unwrap(),
        [0, 1, 2, 3]
    );
    let sections = file.persisted_free_sections().unwrap().unwrap();
    assert!(sections.free_sections().sections().iter().all(|section| {
        section.end_address() <= range.start().get() || range.end().get() <= section.address()
    }));
}

#[rstest]
#[case::before_publication(0)]
#[case::partial_publication(17)]
#[case::completed_publication(usize::MAX)]
fn uncertain_manager_publication_excludes_teardown_writes(
    #[case] prefix: usize,
    #[values(PersistentStrategy::Flat, PersistentStrategy::Page)] strategy: PersistentStrategy,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] policy: SyncPolicy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("uncertain.h5");
    persistent::fixture(&path, strategy);
    let control = directory.path().join("control.h5");
    fs::copy(&path, &control).unwrap();
    let mut reference = open(&control, backing, policy);
    let allocation = reference
        .store()
        .allocate(&[47; 8192], AllocationClass::Raw)
        .unwrap();
    disk_log::start();
    reference
        .store()
        .publish_allocations()
        .unwrap()
        .confirm(allocation)
        .unwrap();
    let operations = disk_log::take();
    assert!(!operations.is_empty());
    let writes = operations
        .iter()
        .filter_map(|operation| match operation {
            DiskOp::Write { offset, .. } => Some(*offset),
            DiskOp::SetLen(_) => None,
        })
        .collect::<Vec<_>>();
    let target = writes
        .iter()
        .position(|&offset| offset == reference.context.signature().get())
        .unwrap();
    assert!(target > 0);
    drop(reference);
    let mut engine = open(&path, backing, policy);
    engine
        .store()
        .allocate(&[47; 8192], AllocationClass::Raw)
        .unwrap();
    disk_log::fail_write_after(target, prefix);
    assert_injected(engine.store().publish_allocations().err().unwrap());
    let issued = engine.disk_bytes().unwrap();
    disk_log::start();
    assert_uncertain(
        engine
            .store()
            .allocate(&[83; 8192], AllocationClass::Raw)
            .err()
            .unwrap(),
    );
    assert_uncertain(engine.commit().unwrap_err());
    assert_uncertain(engine.finalize_persist().unwrap_err());
    assert_uncertain(engine.force_sync().unwrap_err());
    drop(engine);
    assert!(disk_log::take().is_empty());
    assert_eq!(fs::read(&path).unwrap(), issued);
    if prefix == 0 || prefix == usize::MAX {
        let file = File::open(&path).unwrap();
        assert_eq!(
            file.dataset("values").unwrap().read_i32().unwrap(),
            [0, 1, 2, 3]
        );
        file.persisted_free_sections().unwrap().unwrap();
    }
}

#[rstest]
#[case::reference_write(false)]
#[case::reference_barrier(true)]
fn uncertain_reference_io_discards_pending_writes(#[case] barrier: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("reference.h5");
    persistent::fixture(&path, PersistentStrategy::Flat);
    let mut engine = open(&path, MemoryStrategy::Bounded, SyncPolicy::OnClose);
    let allocation = engine
        .store()
        .allocate(&[47; 8192], AllocationClass::Raw)
        .unwrap();
    engine
        .store()
        .publish_allocations()
        .unwrap()
        .confirm(allocation)
        .unwrap();
    let signature = engine.context.signature().get();
    let original = engine.disk_bytes().unwrap();
    let length = hdf5_pure_format::__private::serialize_superblock(engine.superblock())
        .unwrap()
        .len();
    let bytes = original[..length].to_vec();
    if barrier {
        engine.store().write_at(signature, &bytes).unwrap();
        disk_log::fail_write_after(0, 0);
        assert_injected(engine.store().sync().unwrap_err());
    } else {
        engine
            .set_write_buffering(WriteBuffering::Operation {
                page_size: 4096,
                max_bytes: bytes.len(),
            })
            .unwrap();
        engine.store().write_at(signature, &bytes[..16]).unwrap();
        disk_log::fail_write_after(0, 0);
        assert_injected(engine.store().write_at(signature, &bytes).unwrap_err());
    }
    let issued = engine.disk_bytes().unwrap();
    disk_log::start();
    assert_uncertain(engine.force_sync().unwrap_err());
    drop(engine);
    assert!(disk_log::take().is_empty());
    assert_eq!(fs::read(&path).unwrap(), issued);
}

fn open(path: &Path, backing: MemoryStrategy, policy: SyncPolicy) -> WriteEngine {
    let mut engine = persistent::open(path, backing, policy);
    if policy == SyncPolicy::OnClose {
        engine
            .set_write_buffering(WriteBuffering::Session {
                page_size: 4096,
                max_bytes: 1 << 20,
            })
            .unwrap();
    }
    engine
}

fn assert_injected(error: Error) {
    let Error::Io(error) = error else {
        panic!("expected injected I/O failure, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
}

fn assert_uncertain(error: Error) {
    assert!(
        matches!(error, Error::AllocationPublicationUncertain),
        "{error:?}"
    );
}
