use std::fs;
use std::io::ErrorKind;
use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, Mutex};

use hdf5_pure_core::__private::FileRange;
use rstest::rstest;
use test_util::disk_log::{self, DiskOp, FailurePoint, IoEvent, IoOutcome, Recording};
use test_util::global_heap::{self, Object};
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use crate::edit::WriteEngine;
use crate::image::{CountingImage, FileImage, WriteBuffering};
use crate::{
    AttrValue, Error, File, FileAccessProperties, FileLocking, FormatError, HeapCompactionState,
};
use crate::{MemoryStrategy, SyncPolicy};

#[path = "heap_compaction/buffering.rs"]
mod buffering;
#[path = "heap_compaction/progress.rs"]
mod progress;
#[path = "heap_compaction/restoration.rs"]
mod restoration;

test_util_hdf5::hdf5_1_10_tests! {
    #[path = "heap_compaction/native_crosschecks.rs"]
    mod native_crosschecks;
}

const HEADER: usize = 16;
const RECORD: usize = 24;
const PARTIAL: usize = 2;

#[derive(Clone, Copy)]
enum Failure {
    FirstParent(HeapWrite, Prefix),
    Restoration,
    LaterParent,
}

impl Failure {
    fn parent(self) -> usize {
        match self {
            Self::FirstParent(..) | Self::Restoration => 0,
            Self::LaterParent => 1,
        }
    }

    fn write(self) -> (HeapWrite, Prefix) {
        match self {
            Self::FirstParent(write, prefix) => (write, prefix),
            Self::Restoration | Self::LaterParent => (HeapWrite::Gathered, Prefix::Partial),
        }
    }

    fn assert_error(self, error: Error) {
        let Error::HeapCompactionFailed {
            completed,
            state,
            source,
        } = error
        else {
            panic!("expected a heap compaction failure, got {error:?}");
        };
        assert_eq!(completed, self.parent());
        if matches!(self, Self::Restoration) {
            assert_eq!(state, HeapCompactionState::Uncertain);
            let Error::CommitPartiallyApplied { refusal, restore } = *source else {
                panic!("expected a restoration failure, got {source:?}");
            };
            assert_io(*restore, "injected disk synchronization failure");
            assert_io(*refusal, "injected disk write failure");
        } else {
            assert_eq!(state, HeapCompactionState::Restored);
            assert_io(*source, "injected disk write failure");
        }
    }
}

#[derive(Clone, Copy)]
enum HeapWrite {
    Move,
    FreeHeader,
    Gathered,
}

impl HeapWrite {
    fn offset(self, parent: u64) -> u64 {
        parent
            + match self {
                Self::Move | Self::Gathered => HEADER as u64,
                Self::FreeHeader => (HEADER + RECORD) as u64,
            }
    }

    fn length(self) -> usize {
        match self {
            Self::Move => RECORD,
            Self::FreeHeader => HEADER,
            Self::Gathered => RECORD + HEADER,
        }
    }
}

#[derive(Clone, Copy)]
enum Prefix {
    Zero,
    Partial,
    Full,
}

impl Prefix {
    fn bytes(self, length: usize) -> usize {
        match self {
            Self::Zero => 0,
            Self::Partial => PARTIAL,
            Self::Full => length,
        }
    }
}

#[rstest]
#[case::move_unwritten(Failure::FirstParent(HeapWrite::Move, Prefix::Zero))]
#[case::move_partial(Failure::FirstParent(HeapWrite::Move, Prefix::Partial))]
#[case::move_complete(Failure::FirstParent(HeapWrite::Move, Prefix::Full))]
#[case::free_header_unwritten(Failure::FirstParent(HeapWrite::FreeHeader, Prefix::Zero))]
#[case::free_header_partial(Failure::FirstParent(HeapWrite::FreeHeader, Prefix::Partial))]
#[case::free_header_complete(Failure::FirstParent(HeapWrite::FreeHeader, Prefix::Full))]
#[case::gathered_move_partial(Failure::FirstParent(HeapWrite::Gathered, Prefix::Partial))]
#[case::unconfirmed_restoration(Failure::Restoration)]
#[case::later_parent(Failure::LaterParent)]
fn public_heap_compaction_failure_keeps_the_published_edit(#[case] failure: Failure) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failure.h5");
    let control_path = directory.path().join("control.h5");
    let parents = write_source(&path);
    fs::copy(&path, &control_path).unwrap();

    let (target, prefix) = failure.write();
    let (original, published, recording) = successful_commit(&control_path, &parents, target);
    let parent = parents[failure.parent()];
    let event = heap_write(&recording, parent, target);
    let written = prefix.bytes(target.length());

    let file = open_editor(&path, target);
    let before = fs::read(&path).unwrap();
    assert_eq!(before, original);
    assert_original_values(&file);
    stage(&file);
    disk_log::start();
    let prefix = &recording.events[..event];
    disk_log::fail_write_after(
        prefix
            .iter()
            .filter(|event| matches!(event, IoEvent::Write { .. }))
            .count(),
        written,
    );
    if matches!(failure, Failure::Restoration) {
        disk_log::fail_sync_after(
            prefix
                .iter()
                .filter(|event| matches!(event, IoEvent::Synchronize { .. }))
                .count(),
            FailurePoint::Before,
        );
    }
    let error = file.commit().unwrap_err();
    disk_log::clear_failures();
    let failed = disk_log::finish();
    failure.assert_error(error);
    assert_failed_write(
        &recording,
        &failed,
        event,
        written,
        &before,
        parent_range(parent),
    );
    if matches!(failure, Failure::Restoration) {
        assert!(matches!(
            failed.events.last(),
            Some(IoEvent::Synchronize {
                outcome: IoOutcome::Failed(ErrorKind::Other),
                ..
            })
        ));
    }

    let after = fs::read(&path).unwrap();
    assert_eq!(
        after[..Fixture::ROOT as usize],
        published[..Fixture::ROOT as usize]
    );
    assert_parents(&parents, &original, &after, failure.parent());
    assert_published_values(&file);
    let stored = File::from_bytes(after.clone()).unwrap();
    assert_published_values(&stored);
    stored.close().unwrap();
    assert!(!file.has_staged_edits());
    if matches!(failure, Failure::Restoration) {
        file.root().set_attr("after", AttrValue::I64(31)).unwrap();
        disk_log::start();
        for result in [file.commit(), file.sync()] {
            assert!(matches!(
                result.unwrap_err(),
                Error::AllocationPublicationUncertain
            ));
        }
        assert!(disk_log::finish().events.is_empty());
        assert_eq!(fs::read(&path).unwrap(), after);
        drop(file);
        File::clear_status_flags(&path).unwrap();
    } else {
        assert_allocated(&file, &parents);
        file.root().set_attr("after", AttrValue::I64(31)).unwrap();
        file.commit().unwrap();
        assert_parents(
            &parents,
            &original,
            &fs::read(&path).unwrap(),
            failure.parent(),
        );
        file.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_published_values(&file);
    let attrs = file.root().attrs().unwrap();
    let fresh = if matches!(failure, Failure::Restoration) {
        None
    } else {
        Some(AttrValue::I64(31))
    };
    assert_eq!(attrs.get("after"), fresh.as_ref());
    file.close().unwrap();
}

fn assert_failed_write(
    control: &Recording,
    failed: &Recording,
    event: usize,
    written: usize,
    original: &[u8],
    parent: Range<usize>,
) {
    let DiskOp::Write { offset, bytes } =
        &control.operations[control.events[event].operation().unwrap()]
    else {
        panic!("expected the control write");
    };
    assert_issued_write(
        failed,
        event,
        *offset,
        bytes,
        written,
        IoOutcome::Failed(ErrorKind::Other),
    );

    let mut expected = replay_prefix(control, original, event);
    let start = usize::try_from(*offset).unwrap();
    let range = start..start + written;
    if written != 0 {
        assert_ne!(expected[range.clone()], bytes[..written]);
    }
    expected[range].copy_from_slice(&bytes[..written]);
    assert_eq!(
        replay_prefix(failed, original, event + 1)[parent.clone()],
        expected[parent]
    );
}

fn assert_issued_write(
    recording: &Recording,
    event: usize,
    at: u64,
    bytes: &[u8],
    count: usize,
    expected: IoOutcome,
) {
    let IoEvent::Write {
        offset,
        requested,
        written: completed,
        outcome,
        operation,
    } = recording.events[event]
    else {
        panic!("expected a write, got {:?}", recording.events[event]);
    };
    assert_eq!(offset, at);
    assert_eq!(requested, bytes.len());
    assert_eq!(completed, count);
    assert_eq!(outcome, expected);
    if count == 0 {
        assert_eq!(operation, None);
    } else {
        assert_eq!(
            recording.operations[operation.unwrap()],
            DiskOp::Write {
                offset: at,
                bytes: bytes[..count].to_vec(),
            }
        );
    }
}

fn replay_prefix(recording: &Recording, original: &[u8], end: usize) -> Vec<u8> {
    let mut image = original.to_vec();
    for operation in recording.events[..end]
        .iter()
        .filter_map(IoEvent::operation)
    {
        recording.operations[operation].apply(&mut image);
    }
    image
}

#[derive(Clone, Copy)]
enum ReadPoint {
    Preflight,
    MoveSource,
    FreeHeaderUndo,
}

impl ReadPoint {
    fn matches(self, read: FileRange, parent: u64) -> bool {
        let published = disk_log::has_written_at(0);
        let free = HeapWrite::FreeHeader;
        let moved = || disk_log::has_written_at(HeapWrite::Move.offset(parent));
        match self {
            Self::Preflight => !published && read.start().get() == Fixture::ROOT,
            Self::MoveSource => {
                published
                    && read.start().get() == free.offset(parent)
                    && read.length() == RECORD as u64
                    && !moved()
            }
            Self::FreeHeaderUndo => {
                published
                    && read.start().get() == free.offset(parent)
                    && read.length() == HEADER as u64
                    && moved()
            }
        }
    }

    fn assert_error(self, error: Error) {
        let source = match self {
            Self::Preflight => error,
            Self::MoveSource | Self::FreeHeaderUndo => {
                let Error::HeapCompactionFailed {
                    completed,
                    state,
                    source,
                } = error
                else {
                    panic!("expected a heap compaction failure, got {error:?}");
                };
                assert_eq!(completed, 0);
                assert_eq!(state, HeapCompactionState::Restored);
                *source
            }
        };
        let Error::Format(error) = source else {
            panic!("expected a source read failure, got {source:?}");
        };
        assert_eq!(error, FormatError::Source(READ_FAILURE.into()));
    }
}

const READ_FAILURE: &str = "injected commit source read failure";

#[rstest]
#[case::preflight(ReadPoint::Preflight)]
#[case::move_source(ReadPoint::MoveSource)]
#[case::free_header_undo(ReadPoint::FreeHeaderUndo)]
fn public_heap_read_failure_preserves_the_commit_boundary(#[case] point: ReadPoint) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failure.h5");
    let control_path = directory.path().join("control.h5");
    let parents = write_source(&path);
    fs::copy(&path, &control_path).unwrap();
    let (original, published, recording) =
        successful_commit(&control_path, &parents, HeapWrite::Move);
    let armed = Arc::new(Mutex::new(None));
    let fault = Arc::clone(&armed);
    let first = parents[0];
    let file = open_read_fault(&path, move |read| {
        let mut fault = fault.lock().unwrap();
        if fault.is_some_and(|point: ReadPoint| point.matches(read, first)) {
            *fault = None;
            return Err(FormatError::Source(READ_FAILURE.into()));
        }
        Ok(())
    });
    let before = fs::read(&path).unwrap();
    assert_eq!(before, original);
    assert_original_values(&file);
    let original_free = file.reusable_free_sections().unwrap();
    stage(&file);
    assert!(file.has_staged_edits());
    *armed.lock().unwrap() = Some(point);
    disk_log::start();
    let error = file.commit().unwrap_err();
    let failed = disk_log::finish();
    assert!(armed.lock().unwrap().is_none());
    point.assert_error(error);

    let after = fs::read(&path).unwrap();
    let completed = if matches!(point, ReadPoint::Preflight) {
        assert!(failed.events.is_empty());
        assert_eq!(after, before);
        assert_eq!(file.reusable_free_sections().unwrap(), original_free);
        assert!(file.has_staged_edits());
        assert_original_values(&file);
        file.commit().unwrap();
        parents.len()
    } else {
        let write = if matches!(point, ReadPoint::MoveSource) {
            HeapWrite::Move
        } else {
            HeapWrite::FreeHeader
        };
        let next = heap_write(&recording, first, write);
        assert_eq!(failed.events[..next], recording.events[..next]);
        if matches!(point, ReadPoint::MoveSource) {
            assert_eq!(failed.events.len(), next);
        } else {
            let record = &original
                [parent_range(first).start + HEADER..parent_range(first).start + HEADER + RECORD];
            let operation = failed.events[next].operation().unwrap();
            assert_eq!(
                failed.operations[operation],
                DiskOp::Write {
                    offset: HeapWrite::Move.offset(first),
                    bytes: record.to_vec(),
                }
            );
        }
        assert_eq!(
            after[..Fixture::ROOT as usize],
            published[..Fixture::ROOT as usize]
        );
        assert_parents(&parents, &original, &after, 0);
        0
    };
    assert!(!file.has_staged_edits());
    assert_published_values(&file);
    assert_allocated(&file, &parents);
    file.close().unwrap();
    let after = fs::read(&path).unwrap();
    assert_parents(&parents, &original, &after, completed);
    let reopened = File::open(&path).unwrap();
    assert_published_values(&reopened);
    reopened.close().unwrap();
}

fn open_read_fault(
    path: &Path,
    fault: impl Fn(FileRange) -> Result<(), FormatError> + Send + Sync + 'static,
) -> File {
    open_unbuffered(path, |handle| {
        let image = WriteEngine::read_mirror(handle)?;
        Ok(Box::new(
            CountingImage::new(Box::new(image), Arc::default(), Arc::default())
                .with_read_fault(fault),
        ))
    })
}

fn successful_commit(
    path: &Path,
    parents: &[u64],
    target: HeapWrite,
) -> (Vec<u8>, Vec<u8>, Recording) {
    let file = open_editor(path, target);
    let original = fs::read(path).unwrap();
    stage(&file);
    disk_log::start();
    file.commit().unwrap();
    let recording = disk_log::finish();
    let published = fs::read(path).unwrap();
    assert_published_values(&file);
    assert_parents(parents, &original, &published, parents.len());
    assert_allocated(&file, parents);
    file.close().unwrap();
    (original, published, recording)
}

fn write_source(path: &Path) -> [u64; 3] {
    write_source_with_gap(path, 0)
}

fn write_source_with_gap(path: &Path, first_parent: usize) -> [u64; 3] {
    let mut fixture = Fixture::new();
    if first_parent != 0 {
        fixture.image.place(first_parent - 1, &[0]);
    }
    let parents = std::array::from_fn(|_| {
        fixture.collection(&[Object::new(7, b"removed"), Object::new(91, b"retained")])
    });
    let removed = parents
        .iter()
        .flat_map(|&parent| global_heap_fixture::reference(7, parent, 7))
        .collect::<Vec<_>>();
    let retained = parents
        .iter()
        .flat_map(|&parent| global_heap_fixture::reference(8, parent, 91))
        .collect::<Vec<_>>();
    let mut messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &crate::make_i32_type(),
        &17_i32.to_le_bytes(),
    );
    messages.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &removed,
    ));
    fixture.object("value", messages);
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &crate::make_i32_type(),
        &17_i32.to_le_bytes(),
    );
    fixture.object("numbers", messages);
    let messages =
        Form::CompactDataset.messages(&mut fixture, &global_heap_fixture::vlen_string(), &retained);
    fixture.object("keep", messages);
    fixture.write(path);
    parents
}

fn open_editor(path: &Path, target: HeapWrite) -> File {
    if matches!(target, HeapWrite::Gathered) {
        File::open_rw_with_options(path, editor_options()).unwrap()
    } else {
        open_unbuffered(path, |handle| {
            Ok(Box::new(WriteEngine::read_mirror(handle)?))
        })
    }
}

fn open_unbuffered(
    path: &Path,
    build: impl FnOnce(fs::File) -> Result<Box<dyn FileImage>, Error>,
) -> File {
    let mut engine =
        WriteEngine::open_imaged(path, Some(FileLocking::Disabled), |handle, _| build(handle))
            .unwrap();
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    File::from_test_edit_session(engine, editor_options()).unwrap()
}

fn editor_options() -> FileAccessProperties {
    FileAccessProperties::new()
        .with_locking(FileLocking::Disabled)
        .with_memory_strategy(MemoryStrategy::Mirrored)
        .with_page_buffer_size(0)
        .with_sync_policy(SyncPolicy::Always)
}

fn stage(file: &File) {
    file.dataset("value")
        .unwrap()
        .set_attr("a", AttrValue::I64(9))
        .unwrap();
    file.dataset("numbers")
        .unwrap()
        .write_staged(|builder| {
            builder.with_i32_data(&[23]);
        })
        .unwrap();
}

fn assert_published_values(file: &File) {
    assert_published_scalars(file);
    assert_eq!(
        file.dataset("keep").unwrap().read_string().unwrap(),
        ["retained"; 3]
    );
}

fn assert_published_scalars(file: &File) {
    assert_eq!(file.dataset("numbers").unwrap().read_i32().unwrap(), [23]);
    let value = file.dataset("value").unwrap();
    assert_eq!(value.read_i32().unwrap(), [17]);
    assert_eq!(value.attrs().unwrap().get("a"), Some(&AttrValue::I64(9)));
}

fn assert_original_values(file: &File) {
    assert_eq!(file.dataset("numbers").unwrap().read_i32().unwrap(), [17]);
    assert_eq!(
        file.dataset("keep").unwrap().read_string().unwrap(),
        ["retained"; 3]
    );
    let value = file.dataset("value").unwrap();
    assert_eq!(value.read_i32().unwrap(), [17]);
    assert_eq!(
        value.attrs().unwrap().get("a"),
        Some(&AttrValue::VarLenAsciiStringArray(vec![
            "removed".into();
            3
        ]))
    );
}

fn heap_write(recording: &Recording, parent: u64, target: HeapWrite) -> usize {
    let publication = recording
        .events
        .iter()
        .position(|event| matches!(event, IoEvent::Write { offset: 0, .. }))
        .unwrap();
    let write = recording.events.iter().position(|event| matches!(
        event,
        IoEvent::Write { offset, requested, written, outcome: IoOutcome::Succeeded, .. }
            if *offset == target.offset(parent) && *requested == target.length() && written == requested
    )).unwrap_or_else(|| panic!("no compaction write for heap at {parent}: {:?}", recording.events));
    assert!(publication < write);
    write
}

fn parent_range(address: u64) -> Range<usize> {
    let start = usize::try_from(address).unwrap();
    start..start + global_heap::MIN_COLLECTION_SIZE
}

fn assert_parents(parents: &[u64], original: &[u8], observed: &[u8], completed: usize) {
    let packed = global_heap::collection(&[Object::new(91, b"retained")], Widths::EIGHT);
    for (index, &address) in parents.iter().enumerate() {
        let range = parent_range(address);
        let mut expected = original[range.clone()].to_vec();
        if index < completed {
            let written = HEADER + RECORD + HEADER;
            expected[..written].copy_from_slice(&packed[..written]);
        }
        assert_eq!(observed[range], expected, "heap parent {index}");
    }
}

fn assert_allocated(file: &File, parents: &[u64]) {
    let free = file.reusable_free_sections().unwrap();
    for &address in parents {
        let end = address + global_heap::MIN_COLLECTION_SIZE as u64;
        assert!(free.sections().iter().all(|section| {
            section.end_address().get() <= address || section.address().get() >= end
        }));
    }
}

fn assert_io(error: Error, message: &str) {
    let Error::Io(error) = error else {
        panic!("expected an I/O error, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), message);
}
