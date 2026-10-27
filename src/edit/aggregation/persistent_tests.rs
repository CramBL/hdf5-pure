use core::sync::atomic::AtomicBool;
use core::sync::atomic::Ordering;

use std::io::Error as IoError;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::Arc;

use rstest::rstest;

use crate::AttrValue;
use crate::Error;
use crate::File;
use crate::FileBuilder;
use crate::FileLocking;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::FormatError;
use crate::MaxExtent;
use crate::MemoryStrategy;
use crate::PersistedFreeSpacePolicy;
use crate::UserblockSize;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunk_index_inplace::Store;
use crate::edit::AppendBuilder;
use crate::edit::AppendTarget;
use crate::edit::WriteEngine;
use crate::image::FileImage;
use crate::image::MirrorImage;
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;
use crate::source::MetadataCacheConfig;
use crate::source::Source;
use crate::type_builders::DatasetBuilder;

use super::AllocationClass;
use super::ClassicAggregators;
use super::Extent;

#[rstest]
#[case::mirror(MemoryStrategy::Mirrored, 0)]
#[case::bounded(MemoryStrategy::Bounded, 0)]
#[case::userblock(MemoryStrategy::Mirrored, 512)]
fn persistent_reuse_precedes_aggregates(
    #[case] backing: MemoryStrategy,
    #[case] userblock: u64,
    #[values(false, true)] immediate: bool,
    #[values(AllocationClass::Metadata, AllocationClass::Raw)] class: AllocationClass,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("precedence.h5");
    write_file(&path, userblock);
    let mut engine = open_engine(&path, backing);
    engine.delete(&ObjectPathBuf::parse("scratch")).unwrap();
    engine.commit().unwrap();
    if immediate {
        engine.reserve_for_immediate_append(64).unwrap();
    }
    let expected = if immediate {
        let extent = engine.reserved.alloc(64).unwrap();
        engine.reserved.free(extent);
        extent
    } else {
        let extent = engine.space.allocate(64, class).unwrap();
        engine.space.return_allocation(extent, class);
        extent
    };
    let policy = engine.aggregation.unwrap();
    let address = if immediate {
        let mut reserved = std::mem::take(&mut engine.reserved);
        let mut store = engine.store();
        store.reserve = Some(&mut reserved);
        let address = store.allocate(&[19; 64], class).unwrap();
        engine.reserved = reserved;
        address
    } else {
        let placement = engine.reserve(64, class).unwrap();
        engine.place(placement, &[19; 64]).unwrap()
    };
    assert_eq!(address, expected.start() + userblock);
    assert_eq!(engine.aggregation, Some(policy));
    engine.finalize_persist().unwrap();
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("keep").unwrap().read_u8().unwrap(), [11; 4096]);
    assert!(
        !file
            .root()
            .datasets()
            .unwrap()
            .iter()
            .any(|path| path == "scratch")
    );
}

#[rstest]
#[case::mirror(MemoryStrategy::Mirrored, 0)]
#[case::bounded(MemoryStrategy::Bounded, 0)]
#[case::userblock(MemoryStrategy::Mirrored, 512)]
fn active_spare_is_excluded_until_persistent_settlement(
    #[case] backing: MemoryStrategy,
    #[case] userblock: u64,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("settlement.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
    builder.write(&path).unwrap();
    let mut engine = open_engine(&path, backing);
    stage_dataset(&mut engine, "first", 37);
    engine.commit().unwrap();
    let spare = engine.aggregation.unwrap().unused().collect::<Vec<_>>();
    assert_eq!(spare.len(), 2);
    assert_eq!(engine.image.len(), engine.fsm_len);
    assert_spare_excluded(&engine, &spare);
    let before = engine.aggregation.unwrap();
    let placement = engine
        .reserve_aggregated(32, AllocationClass::Raw)
        .unwrap()
        .unwrap();
    engine.place(placement, &[43; 32]).unwrap();
    assert_eq!(engine.image.len(), engine.fsm_len);
    assert_ne!(engine.aggregation, Some(before));
    let mut anchor = DatasetBuilder::new("");
    anchor.with_u8_data(&[]);
    anchor.set_attr("padding", AttrValue::AsciiString("p".repeat(4_096)));
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("anchor"), anchor)
        .unwrap();
    engine.commit().unwrap();
    let interior = engine
        .aggregation
        .unwrap()
        .policy
        .unused(AllocationClass::Raw)
        .unwrap();
    engine.finalize_persist().unwrap();
    assert_eq!(
        engine.aggregation.unwrap().policy,
        ClassicAggregators::new()
    );
    assert!(
        engine
            .space
            .reusable_sections()
            .any(|extent| { extent.start() <= interior.start() && extent.end() >= interior.end() })
    );
    let length = engine.image.len();
    engine.finalize_persist().unwrap();
    assert_eq!(engine.image.len(), length);
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("first").unwrap().read_u8().unwrap(), [37; 64]);
    assert!(
        file.persisted_free_sections()
            .unwrap()
            .unwrap()
            .free_sections()
            .sections()
            .iter()
            .any(|section| {
                section.address() <= interior.start() && section.end_address() >= interior.end()
            })
    );
}

#[rstest]
#[case::small_chunks(4, 600)]
#[case::large_chunks(4096, 8192)]
fn persistent_staged_edits_and_immediate_appends_survive_reopen(
    #[case] chunk: usize,
    #[case] end: usize,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("lifecycle.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    builder
        .create_dataset("values")
        .with_u8_data(&vec![7; chunk])
        .with_shape(&[chunk as u64])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[chunk as u64]);
    builder.write(&path).unwrap();
    let mut engine = open_engine(&path, backing);
    stage_dataset(&mut engine, "first", 37);
    engine.commit().unwrap();
    let mut batch = AppendBuilder::new();
    batch.append_u8(&vec![11; end - chunk]);
    engine
        .append_inplace_gathered(
            AppendTarget::Path(&ObjectPathBuf::parse("values")),
            &batch,
            4,
        )
        .unwrap();
    stage_dataset(&mut engine, "last", 43);
    engine.commit().unwrap();
    let ownership = engine.space_ownership().unwrap();
    assert!(ownership.analysis_complete);
    assert_eq!(ownership.logical_size, ownership.physical_size);
    assert_eq!(ownership.unclassified_space, []);
    assert!(!ownership.free_space_manager_space.is_empty());
    assert_spare_excluded(
        &engine,
        &engine.aggregation.unwrap().unused().collect::<Vec<_>>(),
    );
    engine.finalize_persist().unwrap();
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("first").unwrap().read_u8().unwrap(), [37; 64]);
    assert_eq!(file.dataset("last").unwrap().read_u8().unwrap(), [43; 64]);
    let mut expected = vec![7; chunk];
    expected.extend(vec![11; end - chunk]);
    assert_eq!(file.dataset("values").unwrap().read_u8().unwrap(), expected);
    drop(file);
    let mut engine = open_engine(&path, backing);
    engine.delete(&ObjectPathBuf::parse("first")).unwrap();
    engine.commit().unwrap();
    engine.finalize_persist().unwrap();
    drop(engine);
    let file = File::open(&path).unwrap();
    let mut datasets = file.root().datasets().unwrap();
    datasets.sort_unstable();
    assert_eq!(datasets, ["last", "values"]);
    assert_eq!(file.dataset("values").unwrap().read_u8().unwrap(), expected);
}

#[rstest]
#[case::tail_write(Failure::TailWrite, false)]
#[case::before_publish(Failure::BeforePublish, true)]
#[case::after_publish(Failure::AfterPublish, true)]
#[case::barrier(Failure::Barrier, true)]
#[case::before_truncate(Failure::BeforeTruncate, true)]
#[case::after_truncate(Failure::AfterTruncate, true)]
fn persistent_settlement_failure_preserves_publication_order(
    #[case] failure: Failure,
    #[case] attempted: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failure.h5");
    write_file(&path, 0);
    let armed = Arc::new(AtomicBool::new(false));
    let failed = Arc::new(AtomicBool::new(false));
    let mut engine = WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, _| {
        Ok(Box::new(FailingImage {
            inner: WriteEngine::read_mirror(handle)?,
            failure,
            armed: armed.clone(),
            failed: failed.clone(),
            published: false,
        }))
    })
    .unwrap();
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    stage_dataset(&mut engine, "first", 37);
    engine.commit().unwrap();
    engine.delete(&ObjectPathBuf::parse("scratch")).unwrap();
    engine.commit().unwrap();
    let vacant = engine.image.append(&[0; 16_384]).unwrap();
    engine.space.return_allocation(
        super::Extent::new(vacant, 16_384).unwrap(),
        AllocationClass::Raw,
    );
    engine.synchronize_aggregates().unwrap();
    let before = engine.aggregation;
    let prior = engine.image.len();
    armed.store(true, Ordering::Relaxed);
    let error = engine.finalize_persist().unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected settlement I/O failure, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert!(failed.load(Ordering::Relaxed));
    assert_eq!(engine.publication.attempted(), attempted);
    if !attempted {
        assert_eq!(engine.aggregation, before);
        engine.finalize_persist().unwrap();
        assert!(engine.aggregation.unwrap().publication_confirmed);
    } else {
        assert!(!engine.aggregation.unwrap().publication_confirmed);
        let length = engine.image.len();
        engine.finalize_persist().unwrap();
        assert_eq!(engine.image.len(), length);
    }
    assert!(engine.image.len() <= prior);
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("keep").unwrap().read_u8().unwrap(), [11; 4096]);
    assert_eq!(file.dataset("first").unwrap().read_u8().unwrap(), [37; 64]);
    file.persisted_free_sections().unwrap().unwrap();
}

#[rstest]
#[case::below(49, false)]
#[case::equal(48, true)]
#[case::above(47, true)]
fn abandoned_persistent_spare_uses_threshold(#[case] threshold: u64, #[case] tracked: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("threshold.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(threshold),
    );
    builder.write(&path).unwrap();
    let mut engine = open_engine(&path, MemoryStrategy::Mirrored);
    let start = engine.image.len();
    for (class, bytes) in [
        (AllocationClass::Metadata, vec![19; 2000]),
        (AllocationClass::Raw, vec![23; 64]),
        (AllocationClass::Metadata, vec![29; 64]),
    ] {
        let placement = engine
            .reserve_aggregated(bytes.len() as u64, class)
            .unwrap()
            .unwrap();
        engine.place(placement, &bytes).unwrap();
    }
    let extent = super::Extent::new(start + 2000, 48).unwrap();
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        if tracked { vec![extent] } else { vec![] }
    );
    assert!(engine.aggregation.unwrap().manager_dirty);
    engine.finalize_persist().unwrap();
    drop(engine);
    let file = File::open(&path).unwrap();
    let free = file.persisted_free_sections().unwrap().unwrap();
    assert_eq!(
        free.free_sections()
            .sections()
            .iter()
            .any(|section| section.address() <= extent.start()
                && section.end_address() >= extent.end()),
        tracked
    );
}

#[rstest]
#[case::tail_write(Failure::TailWrite, false)]
#[case::before_publish(Failure::BeforePublish, true)]
#[case::after_publish(Failure::AfterPublish, true)]
#[case::barrier(Failure::Barrier, true)]
fn reserve_publication_failure_prevents_reuse(#[case] failure: Failure, #[case] attempted: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("reserve_failure.h5");
    write_file(&path, 0);
    let armed = Arc::new(AtomicBool::new(false));
    let failed = Arc::new(AtomicBool::new(false));
    let mut engine = WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, _| {
        Ok(Box::new(FailingImage {
            inner: WriteEngine::read_mirror(handle)?,
            failure,
            armed: armed.clone(),
            failed: failed.clone(),
            published: false,
        }))
    })
    .unwrap();
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine.delete(&ObjectPathBuf::parse("scratch")).unwrap();
    engine.commit().unwrap();
    let before = engine.aggregation;
    let free = engine.space.reusable_sections().collect::<Vec<_>>();
    armed.store(true, Ordering::Relaxed);
    let error = engine.reserve_for_immediate_append(64).unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected reserve publication failure, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(engine.publication.attempted(), attempted);
    assert!(engine.reserved.is_empty());
    if !attempted {
        assert_eq!(engine.aggregation, before);
        assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), free);
        engine.reserve_for_immediate_append(64).unwrap();
        assert!(!engine.reserved.is_empty());
    } else {
        assert!(!engine.aggregation.unwrap().publication_confirmed);
    }
    engine.finalize_persist().unwrap();
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("keep").unwrap().read_u8().unwrap(), [11; 4096]);
    file.persisted_free_sections().unwrap().unwrap();
}

#[rstest]
#[case::manager_write(Failure::ManagerWrite, false, false)]
#[case::before_publish(Failure::BeforePublish, true, false)]
#[case::after_publish(Failure::AfterPublish, true, true)]
#[case::barrier(Failure::Barrier, true, true)]
fn releasing_spare_for_a_deletion_preserves_ownership_on_failure(
    #[case] failure: Failure,
    #[case] attempted: bool,
    #[case] deleted: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("deletion_failure.h5");
    write_file(&path, 0);
    let armed = Arc::new(AtomicBool::new(false));
    let failed = Arc::new(AtomicBool::new(false));
    let mut engine = WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, _| {
        Ok(Box::new(FailingImage {
            inner: WriteEngine::read_mirror(handle)?,
            failure,
            armed: armed.clone(),
            failed: failed.clone(),
            published: false,
        }))
    })
    .unwrap();
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let mut first = DatasetBuilder::new("");
    first.with_u8_data(&[37; 2_000]);
    first.set_attr("padding", AttrValue::AsciiString("p".repeat(1_750)));
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("first"), first)
        .unwrap();
    engine.commit().unwrap();
    let before = engine.aggregation;
    assert_eq!(before.unwrap().unused().count(), 2);
    assert!(engine.space_ownership().unwrap().analysis_complete);
    engine.delete(&ObjectPathBuf::parse("first")).unwrap();
    armed.store(true, Ordering::Relaxed);
    let error = engine.commit().unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected deletion publication failure, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(engine.publication.attempted(), attempted);
    if !attempted {
        assert_eq!(engine.aggregation, before);
    } else {
        assert_eq!(engine.aggregation.unwrap().unused().count(), 0);
        assert!(!engine.aggregation.unwrap().publication_confirmed);
    }
    drop(engine);
    let file = File::open(&path).unwrap();
    let mut datasets = file.root().datasets().unwrap();
    datasets.sort_unstable();
    assert_eq!(
        datasets,
        if deleted {
            vec!["keep", "scratch"]
        } else {
            vec!["first", "keep", "scratch"]
        }
    );
    assert_eq!(file.dataset("keep").unwrap().read_u8().unwrap(), [11; 4096]);
    assert_eq!(
        file.dataset("scratch").unwrap().read_u8().unwrap(),
        [7; 4096]
    );
    if !deleted {
        assert_eq!(
            file.dataset("first").unwrap().read_u8().unwrap(),
            [37; 2_000]
        );
    }
    file.persisted_free_sections().unwrap().unwrap();
}

#[derive(Clone, Copy)]
enum Failure {
    AfterPublish,
    AfterTruncate,
    Barrier,
    BeforePublish,
    BeforeTruncate,
    ManagerWrite,
    TailWrite,
}

struct FailingImage {
    inner: MirrorImage,
    failure: Failure,
    armed: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
    published: bool,
}

impl FailingImage {
    fn fail(&self) -> Result<(), Error> {
        self.failed.store(true, Ordering::Relaxed);
        Err(IoError::other("persistent aggregate settlement failure").into())
    }

    fn enabled(&self) -> bool {
        self.armed.load(Ordering::Relaxed) && !self.failed.load(Ordering::Relaxed)
    }
}

impl FileImage for FailingImage {
    fn read_disk_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        self.inner.read_disk_at(offset, bytes)
    }

    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        if self.enabled()
            && (matches!(self.failure, Failure::TailWrite)
                || (matches!(self.failure, Failure::ManagerWrite)
                    && bytes.starts_with(FSHD_SIGNATURE)))
        {
            self.fail()?;
        }
        self.inner.append(bytes)
    }

    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        if self.enabled() {
            if offset == 0 && matches!(self.failure, Failure::BeforePublish) {
                return self.fail();
            }
            if offset != 0 && matches!(self.failure, Failure::TailWrite) {
                return self.fail();
            }
            if matches!(self.failure, Failure::ManagerWrite) && bytes.starts_with(FSHD_SIGNATURE) {
                return self.fail();
            }
        }
        self.inner.write_at(offset, bytes)?;
        if offset == 0 && self.armed.load(Ordering::Relaxed) {
            self.published = true;
            if self.enabled() && matches!(self.failure, Failure::AfterPublish) {
                return self.fail();
            }
        }
        Ok(())
    }

    fn truncate(&mut self, length: u64) -> Result<(), Error> {
        if self.enabled() && matches!(self.failure, Failure::BeforeTruncate) {
            return self.fail();
        }
        self.inner.truncate(length)?;
        if self.enabled() && matches!(self.failure, Failure::AfterTruncate) {
            return self.fail();
        }
        Ok(())
    }

    fn sync_data(&mut self) -> Result<(), Error> {
        self.inner.sync_data()
    }

    fn sync_all(&mut self) -> Result<(), Error> {
        self.inner.sync_all()?;
        if self.enabled() && self.published && matches!(self.failure, Failure::Barrier) {
            return self.fail();
        }
        Ok(())
    }

    fn ordering_barrier(&mut self) -> Result<(), Error> {
        self.inner.ordering_barrier()
    }

    fn set_write_buffering(&mut self, mode: WriteBuffering) -> Result<(), Error> {
        self.inner.set_write_buffering(mode)
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

fn assert_spare_excluded(engine: &WriteEngine, spare: &[Extent]) {
    let owned = engine.space_ownership().unwrap();
    for extent in spare {
        let start = engine
            .context
            .addresses()
            .resolve(StoredAddress::new(extent.start()))
            .unwrap()
            .get();
        assert!(
            owned
                .free_space_manager_space
                .iter()
                .all(|&(at, len)| { start + extent.len() <= at || at + len <= start })
        );
        let file = File::from_bytes(
            engine
                .image
                .read_exact_at(0, usize::try_from(engine.image.len()).unwrap())
                .unwrap(),
        )
        .unwrap();
        assert!(
            file.persisted_free_sections()
                .unwrap()
                .unwrap()
                .free_sections()
                .sections()
                .iter()
                .all(|section| {
                    extent.end() <= section.address() || section.end_address() <= extent.start()
                })
        );
    }
}

fn stage_dataset(engine: &mut WriteEngine, path: &str, value: u8) {
    let mut dataset = DatasetBuilder::new("");
    dataset.with_u8_data(&[value; 64]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse(path), dataset)
        .unwrap();
}

fn write_file(path: &Path, userblock: u64) {
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
    builder.create_dataset("scratch").with_u8_data(&[7; 4096]);
    builder.create_dataset("keep").with_u8_data(&[11; 4096]);
    builder.write(path).unwrap();
}

fn open_engine(path: &Path, backing: MemoryStrategy) -> WriteEngine {
    WriteEngine::open_rw_with_strategy(
        path,
        MetadataCacheConfig::default(),
        FileLocking::Enabled,
        backing,
        PersistedFreeSpacePolicy::Tolerant,
    )
    .unwrap()
}

// The "Free-space Index" section of format specification version 4.0 defines this signature.
const FSHD_SIGNATURE: &[u8; 4] = b"FSHD";
