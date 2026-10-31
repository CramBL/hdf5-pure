use std::io::Error as IoError;
use std::io::ErrorKind;

use rstest::rstest;

use crate::FileLocking;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::MaxExtent;
use crate::MemoryStrategy;
use crate::PersistedFreeSpacePolicy;
use crate::UserblockSize;
use crate::address::{StoredAddress, StoredAddressExt};
use crate::error::Error;
use crate::error::FormatError;
use crate::image::FileImage;
use crate::image::MirrorImage;
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;
use crate::reader::File;
use crate::source::MetadataCacheConfig;
use crate::source::Source;
use crate::type_builders::DatasetBuilder;
use crate::writer::FileBuilder;

use super::AllocationClass;
use super::WriteEngine;

mod fsm;
mod immediate;
mod persistent;

#[rstest]
#[case::segregation(
    &[(AllocationClass::Metadata, 64), (AllocationClass::Raw, 64), (AllocationClass::Metadata, 64)],
    &[0, 2048, 64], 4096
)]
#[case::refill(
    &[(AllocationClass::Raw, 1024), (AllocationClass::Raw, 1024), (AllocationClass::Raw, 64)],
    &[0, 1024, 2048], 4096
)]
#[case::large_direct(&[(AllocationClass::Raw, 4096)], &[0], 4096)]
#[case::large_extension(
    &[(AllocationClass::Raw, 64), (AllocationClass::Raw, 4096)], &[0, 64], 6144
)]
fn staged_reservations_use_classic_windows(
    #[case] requests: &[(AllocationClass, usize)],
    #[case] offsets: &[u64],
    #[case] reserved: u64,
    #[values(0, 512, 4096)] userblock: u64,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("windows.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Aggr));
    builder.write(&path).unwrap();
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let start = engine.image.len();
    for ((class, length), offset) in requests.iter().zip(offsets) {
        let bytes = vec![19; *length];
        let placement = engine.reserve(*length as u64, *class).unwrap();
        assert_eq!(placement.address(), start + offset);
        let address = engine.place(placement, &bytes).unwrap();
        assert_eq!(
            engine.image().read_exact_at(address, *length).unwrap(),
            bytes
        );
    }
    assert_eq!(engine.image.len(), start + reserved);
    let state = engine.aggregation.unwrap();
    assert_eq!(
        state.eoa(),
        StoredAddress::new(start + reserved - userblock)
    );
    assert_eq!(
        engine.reusable_space_stats().unwrap().reusable_free_bytes(),
        state.unused().map(|extent| extent.len()).sum::<u64>()
    );
}

#[rstest]
#[case::mirror(MemoryStrategy::Mirrored)]
#[case::bounded(MemoryStrategy::Bounded)]
fn aggregate_tail_release_preserves_committed_data(#[case] backing: MemoryStrategy) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("close.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Aggr));
    builder.write(&path).unwrap();
    let mut engine = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::default(),
        FileLocking::Enabled,
        backing,
        PersistedFreeSpacePolicy::Tolerant,
    )
    .unwrap();
    let mut dataset = DatasetBuilder::new("");
    dataset.with_i32_data(&[11; 16]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("/values"), dataset)
        .unwrap();
    engine.commit().unwrap();
    let before = engine.image.len();
    let tail = engine
        .aggregation
        .unwrap()
        .unused()
        .find(|extent| extent.end() == StoredAddress::new(before))
        .unwrap();
    engine.finalize_persist().unwrap();
    assert_eq!(
        engine.image.len(),
        engine.addresses().absolute(tail.start()).unwrap()
    );
    assert_eq!(
        engine.context.superblock().eof_address,
        engine.addresses().absolute(tail.start()).unwrap()
    );
    assert!(engine.aggregation.unwrap().unused().next().is_none());
    let order = engine.image.issued_write_order();
    assert_eq!(order.last().unwrap().0, engine.context.signature().get());
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("values").unwrap().read_i32().unwrap(),
        [11; 16]
    );
    assert_eq!(file.persisted_free_sections().unwrap(), None);
}

#[rstest]
fn overflowing_aggregate_reservation_keeps_state(
    #[values(AllocationClass::Metadata, AllocationClass::Raw)] class: AllocationClass,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("overflow.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Aggr));
    builder.write(&path).unwrap();
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let state = engine.aggregation;
    let end = engine.image.len();
    let error = engine.reserve(u64::MAX, class).unwrap_err();
    assert!(
        matches!(error, Error::Format(FormatError::OffsetOverflow { offset, length }) if offset == end && length == u64::MAX),
        "{error:?}"
    );
    assert_eq!(engine.aggregation, state);
    assert_eq!(engine.image.len(), end);
}

#[rstest]
fn failed_staged_write_restores_aggregate_state(
    #[values(FileSpaceStrategy::Aggr, FileSpaceStrategy::FsmAggr)] strategy: FileSpaceStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failed.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(strategy));
    builder.create_dataset("keep").with_i32_data(&[7]);
    builder.write(&path).unwrap();
    let end = std::fs::metadata(&path).unwrap().len();
    let mut engine = WriteEngine::open_torn_writes(&path, end..u64::MAX).unwrap();
    let original = engine.aggregation;
    let mut dataset = DatasetBuilder::new("");
    dataset.with_i32_data(&[11; 16]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("/failed"), dataset)
        .unwrap();
    let error = engine.commit().unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected device write failure, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(engine.aggregation, original);
    assert_eq!(engine.context.superblock().eof_address, end);
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), ["keep"]);
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [7]);
}

#[rstest]
fn failed_staged_write_restores_existing_aggregate_state(
    #[values(FileSpaceStrategy::Aggr, FileSpaceStrategy::FsmAggr)] strategy: FileSpaceStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("rollback_reserve.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(strategy));
    builder.write(&path).unwrap();
    let start = std::fs::metadata(&path).unwrap().len();
    let mut engine = WriteEngine::open_torn_writes(&path, start + 64..start + 128).unwrap();
    let mut first = DatasetBuilder::new("");
    first.with_i32_data(&[43; 16]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("/first"), first)
        .unwrap();
    engine.commit().unwrap();
    let state = engine.aggregation;
    let mut second = DatasetBuilder::new("");
    second.with_i32_data(&[47; 16]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("/second"), second)
        .unwrap();
    let error = engine.commit().unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected reserved-storage write failure, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert!(!engine.publication.attempted());
    assert_eq!(engine.aggregation, state);
    engine.finalize_persist().unwrap();
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), ["first"]);
    assert_eq!(file.dataset("first").unwrap().read_i32().unwrap(), [43; 16]);
}

#[rstest]
fn failed_publish_keeps_consumed_aggregate_space(
    #[values(FileSpaceStrategy::Aggr, FileSpaceStrategy::FsmAggr)] strategy: FileSpaceStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("publish.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(strategy));
    builder.write(&path).unwrap();
    let mut engine = WriteEngine::open_torn_writes(&path, 0..48).unwrap();
    let original = engine.aggregation;
    let mut dataset = DatasetBuilder::new("");
    dataset.with_i32_data(&[13; 16]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("/values"), dataset)
        .unwrap();
    let error = engine.commit().unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected device write failure, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert!(engine.publication.attempted());
    assert_ne!(engine.aggregation, original);
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("values").unwrap().read_i32().unwrap(),
        [13; 16]
    );
}

#[rstest]
#[case::write_before(PublicationFailure::WriteBefore, false)]
#[case::write_after(PublicationFailure::WriteAfter, true)]
#[case::barrier(PublicationFailure::Barrier, true)]
fn aggregate_teardown_preserves_a_publication_after_transient_failure(
    #[case] failure: PublicationFailure,
    #[case] applied: bool,
    #[values(FileSpaceStrategy::Aggr, FileSpaceStrategy::FsmAggr)] strategy: FileSpaceStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("transient.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(strategy));
    builder.create_dataset("keep").with_i32_data(&[41]);
    builder.write(&path).unwrap();
    let mut engine =
        WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, _length| {
            Ok(Box::new(TransientPublicationImage {
                inner: WriteEngine::read_mirror(handle)?,
                failure,
                publication_written: false,
                failed: false,
            }))
        })
        .unwrap();
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let mut dataset = DatasetBuilder::new("");
    dataset.with_i32_data(&[37; 16]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("/values"), dataset)
        .unwrap();
    let error = engine.commit().unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected transient publication failure, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert!(engine.publication.attempted());
    let published = File::from_bytes(engine.image.as_slice().unwrap().to_vec()).unwrap();
    let expected = if applied {
        vec!["keep", "values"]
    } else {
        vec!["keep"]
    };
    assert_eq!(published.root().datasets().unwrap(), expected);
    if applied {
        assert_eq!(
            published.dataset("values").unwrap().read_i32().unwrap(),
            [37; 16]
        );
    }
    assert!(!engine.aggregation.unwrap().publication_confirmed);
    let physical_size = engine.image.len();
    let issued = engine.disk_bytes().unwrap();
    assert!(matches!(
        engine.commit(),
        Err(Error::AllocationPublicationUncertain)
    ));
    assert!(!engine.aggregation.unwrap().publication_confirmed);
    assert!(matches!(
        engine.finalize_persist(),
        Err(Error::AllocationPublicationUncertain)
    ));
    assert!(matches!(
        engine.force_sync(),
        Err(Error::AllocationPublicationUncertain)
    ));
    assert_eq!(engine.disk_bytes().unwrap(), issued);
    let after_cleanup = engine.image.len();
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), expected);
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [41]);
    if applied {
        assert_eq!(
            file.dataset("values").unwrap().read_i32().unwrap(),
            [37; 16]
        );
    }
    assert_eq!(after_cleanup, physical_size);
}

#[test]
fn immediate_growth_keeps_previous_reserves_disjoint() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("interleaved.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Aggr));
    builder
        .create_dataset("append")
        .with_i32_data(&[1, 2, 3, 4])
        .with_shape(&[4])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[4]);
    builder.write(&path).unwrap();
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let mut staged = DatasetBuilder::new("");
    staged.with_i32_data(&[41; 16]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("/first"), staged)
        .unwrap();
    engine.commit().unwrap();
    let before = engine.image.len();
    let policy = engine.aggregation.unwrap();
    let raw = policy.policy.unused(AllocationClass::Raw).unwrap();
    engine
        .append_inplace_i32_phased("append", &[5, 6, 7, 8], 4)
        .unwrap();
    assert_eq!(engine.image.len(), before);
    let after = engine.aggregation.unwrap();
    assert_eq!(
        after.policy.unused(AllocationClass::Metadata),
        policy.policy.unused(AllocationClass::Metadata)
    );
    assert_eq!(
        after.policy.unused(AllocationClass::Raw).unwrap().start(),
        raw.start().checked_add(16).unwrap()
    );
    engine
        .create_group(&ObjectPathBuf::parse("/second"))
        .unwrap();
    engine.commit().unwrap();
    let ownership = engine.space_ownership().unwrap();
    assert!(ownership.analysis_complete);
    assert_eq!(ownership.physical_size, ownership.logical_size);
    engine.finalize_persist().unwrap();
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("append").unwrap().read_i32().unwrap(),
        (1..=8).collect::<Vec<_>>()
    );
    assert_eq!(file.root().groups().unwrap(), ["second"]);
}

struct TransientPublicationImage {
    inner: MirrorImage,
    failure: PublicationFailure,
    publication_written: bool,
    failed: bool,
}

impl FileImage for TransientPublicationImage {
    fn disk_bytes(&self) -> Result<Vec<u8>, Error> {
        self.inner.disk_bytes()
    }

    fn read_disk_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        self.inner.read_disk_at(offset, bytes)
    }

    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        self.inner.append(bytes)
    }

    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        if offset == 0 && matches!(self.failure, PublicationFailure::WriteBefore) && !self.failed {
            self.failed = true;
            return Err(Error::Io(IoError::other(
                "transient publication write failure",
            )));
        }
        self.inner.write_at(offset, bytes)?;
        if offset == 0 {
            self.publication_written = true;
            if matches!(self.failure, PublicationFailure::WriteAfter) && !self.failed {
                self.failed = true;
                return Err(Error::Io(IoError::other(
                    "transient publication write failure",
                )));
            }
        }
        Ok(())
    }

    fn truncate(&mut self, length: u64) -> Result<(), Error> {
        self.inner.truncate(length)
    }

    fn sync_data(&mut self) -> Result<(), Error> {
        self.inner.sync_data()
    }

    fn sync_all(&mut self) -> Result<(), Error> {
        self.inner.sync_all()?;
        if self.publication_written
            && matches!(self.failure, PublicationFailure::Barrier)
            && !self.failed
        {
            self.failed = true;
            return Err(Error::Io(IoError::other(
                "transient publication barrier failure",
            )));
        }
        Ok(())
    }

    fn ordering_barrier(&mut self) -> Result<(), Error> {
        self.inner.ordering_barrier()
    }

    fn flush(&mut self) -> Result<(), Error> {
        self.inner.flush()
    }

    fn discard_pending_writes(&mut self) {
        self.inner.discard_pending_writes();
    }

    fn issued_write_order(&self) -> Vec<(u64, u64)> {
        self.inner.issued_write_order()
    }

    fn set_write_buffering(&mut self, mode: WriteBuffering) -> Result<(), Error> {
        self.inner.set_write_buffering(mode)
    }

    fn as_slice(&self) -> Option<&[u8]> {
        self.inner.as_slice()
    }
}

impl Source for TransientPublicationImage {
    fn len(&self) -> u64 {
        self.inner.len()
    }

    fn read_at(&self, offset: u64, buffer: &mut [u8]) -> Result<(), FormatError> {
        self.inner.read_at(offset, buffer)
    }
}
#[derive(Clone, Copy)]
enum PublicationFailure {
    Barrier,
    WriteAfter,
    WriteBefore,
}
