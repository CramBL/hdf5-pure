use std::io::Error as IoError;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::{Arc, Mutex};

use hdf5_pure_space::__private::AllocationClass;
use rstest::rstest;

use crate::FileBuilder;
use crate::FileLocking;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::MaxExtent;
use crate::SyncPolicy;
use crate::chunk_index_inplace::Store;
use crate::edit::WriteEngine;
use crate::error::{Error, FormatError};
use crate::image::{FileImage, ImageWork, MirrorImage, WriteBuffering};
use crate::object_path::ObjectPathBuf;
use crate::source::Source;

#[rstest]
#[case::durable(
    SyncPolicy::Always,
    Synchronization::Data,
    Synchronization::Metadata,
    false
)]
#[case::ordered(
    SyncPolicy::OnClose,
    Synchronization::Ordering,
    Synchronization::Ordering,
    false
)]
#[case::explicit_page_buffer(
    SyncPolicy::OnClose,
    Synchronization::Ordering,
    Synchronization::Ordering,
    true
)]
fn ownership_confirmation_follows_content_and_complete_eof_publication(
    #[case] policy: SyncPolicy,
    #[case] content: Synchronization,
    #[case] publication: Synchronization,
    #[case] page_buffer: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("publication.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("values")
        .with_i32_data(&[0, 1, 2, 3])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[4]);
    builder.write(&path).unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut engine = open(&path, policy, Failure::None, &events);
    if page_buffer {
        engine
            .set_write_buffering(WriteBuffering::Session {
                page_size: 4096,
                max_bytes: 65_536,
            })
            .unwrap();
    }
    let mut store = engine.store();
    let allocation = store.allocate(&[7; 128], AllocationClass::Raw).unwrap();
    let range = allocation.range();
    let before = events.lock().unwrap().clone();
    let error = store.write_at(0, &[0]).unwrap_err();
    let Error::Format(FormatError::Internal(reason)) = error else {
        panic!("expected provisional ownership to prohibit a reference, got {error:?}");
    };
    assert_eq!(
        reason,
        "reference publication precedes allocation ownership publication"
    );
    assert_eq!(*events.lock().unwrap(), before);
    let confirmed = store.publish_allocations().unwrap();
    let confirmed = confirmed.confirm(allocation).unwrap();
    assert_eq!(
        store.allocation_publication().require(&confirmed).unwrap(),
        range
    );
    let mut bytes = vec![0; range.end().get() as usize];
    engine.image.read_disk_at(0, &mut bytes).unwrap();
    let superblock = crate::superblock::read(bytes.as_slice())
        .unwrap()
        .superblock;
    assert!(superblock.eof_address >= range.end().get());
    assert_eq!(
        &bytes[range.start().get() as usize..range.end().get() as usize],
        &[7; 128]
    );
    let events = events.lock().unwrap();
    let payload = events
        .iter()
        .position(|event| *event == Event::Content)
        .unwrap();
    let content_barrier = events
        .iter()
        .enumerate()
        .find(|(index, event)| *index > payload && **event == Event::Synchronize(content, true))
        .map(|(index, _)| index)
        .unwrap();
    let superblock = events
        .iter()
        .position(|event| *event == Event::Superblock)
        .unwrap();
    let publication_barrier = events
        .iter()
        .enumerate()
        .find(|(index, event)| {
            *index > superblock && **event == Event::Synchronize(publication, true)
        })
        .map(|(index, _)| index)
        .unwrap();
    assert!(
        payload < content_barrier
            && content_barrier < superblock
            && superblock < publication_barrier,
        "{events:?}"
    );
    drop(events);
    drop(engine);
}

#[rstest]
#[case::content_sync(Failure::ContentSync)]
#[case::before_publication(Failure::BeforeWrite)]
#[case::partial_publication(Failure::PartialWrite)]
#[case::completed_publication_error(Failure::WrittenError)]
#[case::publication_sync(Failure::PublicationSync)]
fn uncertain_ownership_prevents_later_allocation_and_reference_writes(#[case] failure: Failure) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("uncertain.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("values")
        .with_i32_data(&[0, 1, 2, 3]);
    builder.write(&path).unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut engine = open(&path, SyncPolicy::Always, failure, &events);
    let mut store = engine.store();
    store.allocate(&[7; 128], AllocationClass::Raw).unwrap();
    let error = match store.publish_allocations() {
        Ok(_) => panic!("expected an injected publication failure: {failure:?}"),
        Err(error) => error,
    };
    let Error::Io(error) = error else {
        panic!("expected the injected storage error for {failure:?}, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected ownership publication failure");
    let before = events.lock().unwrap().clone();
    assert!(!before.is_empty());
    assert!(matches!(
        store.allocate(&[9; 16], AllocationClass::Raw),
        Err(Error::AllocationPublicationUncertain)
    ));
    assert!(matches!(
        store.write_at(0, &[0]),
        Err(Error::AllocationPublicationUncertain)
    ));
    assert!(matches!(
        store.publish_allocations(),
        Err(Error::AllocationPublicationUncertain)
    ));
    assert_eq!(*events.lock().unwrap(), before);
    drop(engine);
}

#[test]
fn an_uncertain_preallocation_drain_prevents_further_writes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("allocation-drain.h5");
    FileBuilder::new().write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut engine = open(&path, SyncPolicy::Always, Failure::AllocationDrain, &events);
    let mut store = engine.store();
    let error = match store.allocate(&[7; 128], AllocationClass::Raw) {
        Ok(_) => panic!("allocation proceeded after a failed prior-publication drain"),
        Err(error) => error,
    };
    let Error::Io(error) = error else {
        panic!("expected injected drain failure, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected ownership publication failure");
    let failed = events.lock().unwrap().clone();
    assert!(!failed.is_empty());
    assert!(matches!(
        store.allocate(&[9; 16], AllocationClass::Raw),
        Err(Error::AllocationPublicationUncertain)
    ));
    assert!(matches!(
        store.write_at(0, &[0]),
        Err(Error::AllocationPublicationUncertain)
    ));
    assert_eq!(*events.lock().unwrap(), failed);
    assert_eq!(engine.disk_bytes().unwrap(), before);
}

#[test]
fn failed_provisional_tail_truncation_blocks_later_publication() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failed-tail-restoration.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    builder
        .create_dataset("values")
        .with_i32_data(&[0, 1, 2, 3]);
    builder.write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut engine = open(
        &path,
        SyncPolicy::Always,
        Failure::RollbackTruncate,
        &events,
    );
    engine
        .set_write_buffering(WriteBuffering::Operation {
            page_size: 4096,
            max_bytes: 1 << 20,
        })
        .unwrap();
    let allocation = engine
        .store()
        .allocate(&[7; 16_384], AllocationClass::Raw)
        .unwrap();
    assert!(allocation.range().start().get() >= before.len() as u64);
    let error = engine.store().publish_allocations().err().unwrap();
    let Error::Io(error) = error else {
        panic!("expected rollback truncation failure, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected ownership publication failure");
    let failed = events.lock().unwrap().clone();
    assert!(matches!(
        engine.commit(),
        Err(Error::AllocationPublicationUncertain)
    ));
    assert!(matches!(
        engine.finalize_persist(),
        Err(Error::AllocationPublicationUncertain)
    ));
    assert!(matches!(
        engine.force_sync(),
        Err(Error::AllocationPublicationUncertain)
    ));
    assert_eq!(*events.lock().unwrap(), failed);
    assert_eq!(engine.disk_bytes().unwrap(), before);
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn failed_explicit_sync_discards_pending_staged_writes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("uncertain-sync.h5");
    FileBuilder::new().write(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut engine = open(
        &path,
        SyncPolicy::OnClose,
        Failure::PublicationSync,
        &events,
    );
    engine
        .set_write_buffering(WriteBuffering::Session {
            page_size: 4096,
            max_bytes: 1 << 20,
        })
        .unwrap();
    engine
        .create_group(&ObjectPathBuf::parse("pending"))
        .unwrap();
    engine.commit().unwrap();
    assert_eq!(engine.disk_bytes().unwrap(), before);
    let error = engine.force_sync().unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected explicit synchronization failure, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected ownership publication failure");
    let failed = events.lock().unwrap().clone();
    assert!(matches!(
        engine.commit(),
        Err(Error::AllocationPublicationUncertain)
    ));
    assert!(matches!(
        engine.force_sync(),
        Err(Error::AllocationPublicationUncertain)
    ));
    assert_eq!(*events.lock().unwrap(), failed);
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

fn open(
    path: &Path,
    policy: SyncPolicy,
    failure: Failure,
    events: &Arc<Mutex<Vec<Event>>>,
) -> WriteEngine {
    let mut engine = WriteEngine::open_imaged(path, Some(FileLocking::Enabled), |handle, _| {
        let inner = WriteEngine::read_mirror(handle)?;
        Ok(Box::new(BoundaryImage {
            inner,
            failure,
            published: false,
            events: Arc::clone(events),
        }))
    })
    .unwrap();
    engine.aggregation = None;
    engine.set_sync_policy(policy);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    events.lock().unwrap().clear();
    engine
}

struct BoundaryImage {
    inner: MirrorImage,
    failure: Failure,
    published: bool,
    events: Arc<Mutex<Vec<Event>>>,
}

impl BoundaryImage {
    fn synchronize(&mut self, synchronization: Synchronization) -> Result<(), Error> {
        let fail = matches!(
            (self.failure, self.published, synchronization),
            (Failure::AllocationDrain, false, Synchronization::Ordering)
                | (Failure::ContentSync, false, Synchronization::Data)
                | (Failure::RollbackTruncate, false, Synchronization::Metadata)
                | (Failure::PublicationSync, true, _)
        );
        if fail {
            if !matches!(self.failure, Failure::RollbackTruncate) {
                self.failure = Failure::None;
            }
            self.events
                .lock()
                .unwrap()
                .push(Event::Synchronize(synchronization, false));
            return Err(injected());
        }
        match synchronization {
            Synchronization::Data => self.inner.sync_data()?,
            Synchronization::Metadata => self.inner.sync_all()?,
            Synchronization::Ordering => self.inner.flush()?,
        }
        self.events
            .lock()
            .unwrap()
            .push(Event::Synchronize(synchronization, true));
        Ok(())
    }
}

impl FileImage for BoundaryImage {
    fn buffering_work(&self) -> ImageWork {
        self.inner.buffering_work()
    }

    fn disk_bytes(&self) -> Result<Vec<u8>, Error> {
        self.inner.disk_bytes()
    }

    fn read_disk_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        self.inner.read_disk_at(offset, bytes)
    }

    fn append(&mut self, bytes: &[u8]) -> Result<u64, Error> {
        let at = self.inner.append(bytes)?;
        self.events.lock().unwrap().push(Event::Content);
        Ok(at)
    }

    fn write_at(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        if offset != 0 {
            self.events.lock().unwrap().push(Event::Content);
            return self.inner.write_at(offset, bytes);
        }
        self.events.lock().unwrap().push(Event::Superblock);
        let failure = std::mem::replace(&mut self.failure, Failure::None);
        match failure {
            Failure::BeforeWrite => return Err(injected()),
            Failure::PartialWrite => {
                self.inner.write_at(offset, &bytes[..bytes.len() / 2])?;
                return Err(injected());
            }
            Failure::WrittenError => {
                self.inner.write_at(offset, bytes)?;
                return Err(injected());
            }
            other => self.failure = other,
        }
        self.inner.write_at(offset, bytes)?;
        self.published = true;
        Ok(())
    }

    fn truncate(&mut self, len: u64) -> Result<(), Error> {
        if matches!(self.failure, Failure::RollbackTruncate) {
            return Err(injected());
        }
        self.inner.truncate(len)
    }
    fn sync_data(&mut self) -> Result<(), Error> {
        self.synchronize(Synchronization::Data)
    }
    fn sync_all(&mut self) -> Result<(), Error> {
        self.synchronize(Synchronization::Metadata)
    }
    fn ordering_barrier(&mut self) -> Result<(), Error> {
        self.inner.ordering_barrier()
    }
    fn flush(&mut self) -> Result<(), Error> {
        self.synchronize(Synchronization::Ordering)
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
}

impl Source for BoundaryImage {
    fn len(&self) -> u64 {
        self.inner.len()
    }
    fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        self.inner.read_at(offset, bytes)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Event {
    Content,
    Superblock,
    Synchronize(Synchronization, bool),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Synchronization {
    Data,
    Metadata,
    Ordering,
}

#[derive(Clone, Copy, Debug)]
enum Failure {
    AllocationDrain,
    BeforeWrite,
    ContentSync,
    None,
    PartialWrite,
    PublicationSync,
    RollbackTruncate,
    WrittenError,
}

fn injected() -> Error {
    IoError::other("injected ownership publication failure").into()
}
