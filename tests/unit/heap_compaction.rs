use std::fs;
use std::io::ErrorKind;
use std::ops::Range;
use std::path::{Path, PathBuf};
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

#[path = "heap_compaction/progress.rs"]
mod progress;
#[path = "heap_compaction/restoration.rs"]
mod restoration;

const HEADER: usize = 16;
const RECORD: usize = 24;
const PARTIAL: usize = 2;

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
#[case::move_unwritten(0, HeapWrite::Move, Prefix::Zero)]
#[case::move_partial(0, HeapWrite::Move, Prefix::Partial)]
#[case::move_complete(0, HeapWrite::Move, Prefix::Full)]
#[case::free_header_unwritten(0, HeapWrite::FreeHeader, Prefix::Zero)]
#[case::free_header_partial(0, HeapWrite::FreeHeader, Prefix::Partial)]
#[case::free_header_complete(0, HeapWrite::FreeHeader, Prefix::Full)]
#[case::gathered_move_partial(0, HeapWrite::Gathered, Prefix::Partial)]
#[case::later_parent(1, HeapWrite::Gathered, Prefix::Partial)]
fn restored_heap_compaction_failure_allows_another_edit(
    #[case] parent: usize,
    #[case] write: HeapWrite,
    #[case] prefix: Prefix,
) {
    let directory = tempfile::tempdir().unwrap();
    let commit = FaultCommit::prepare(directory.path(), parent, write);
    let written = prefix.bytes(write.length());
    commit.fail_write(written);
    let error = commit.file.commit().unwrap_err();
    disk_log::clear_failures();
    let failed = disk_log::finish();
    let (state, source) = compaction_failure(error, parent);
    assert_eq!(state, HeapCompactionState::Restored);
    assert_io(source, "injected disk write failure");

    commit.assert_published(&failed, written);
    assert_allocated(&commit.file, &commit.parents);
    commit
        .file
        .root()
        .set_attr("after", AttrValue::I64(31))
        .unwrap();
    commit.file.commit().unwrap();
    assert_parents(
        &commit.parents,
        &commit.original,
        &fs::read(&commit.path).unwrap(),
        parent,
    );
    commit.file.close().unwrap();

    let file = File::open(&commit.path).unwrap();
    assert_published_values(&file);
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.get("after"), Some(&AttrValue::I64(31)));
    file.close().unwrap();
}

#[test]
fn unconfirmed_heap_compaction_restoration_quarantines_the_editor() {
    let directory = tempfile::tempdir().unwrap();
    let commit = FaultCommit::prepare(directory.path(), 0, HeapWrite::Gathered);
    commit.fail_write(PARTIAL);
    disk_log::fail_sync_after(
        commit.recording.events[..commit.event]
            .iter()
            .filter(|event| matches!(event, IoEvent::Synchronize { .. }))
            .count(),
        FailurePoint::Before,
    );
    let error = commit.file.commit().unwrap_err();
    disk_log::clear_failures();
    let failed = disk_log::finish();
    let (state, source) = compaction_failure(error, 0);
    assert_eq!(state, HeapCompactionState::Uncertain);
    let Error::CommitPartiallyApplied { refusal, restore } = source else {
        panic!("expected a restoration failure, got {source:?}");
    };
    assert_io(*restore, "injected disk synchronization failure");
    assert_io(*refusal, "injected disk write failure");
    assert!(matches!(
        failed.events.last(),
        Some(IoEvent::Synchronize {
            outcome: IoOutcome::Failed(ErrorKind::Other),
            ..
        })
    ));

    let after = commit.assert_published(&failed, PARTIAL);
    commit
        .file
        .root()
        .set_attr("after", AttrValue::I64(31))
        .unwrap();
    disk_log::start();
    for result in [commit.file.commit(), commit.file.sync()] {
        assert!(matches!(
            result.unwrap_err(),
            Error::AllocationPublicationUncertain
        ));
    }
    assert!(disk_log::finish().events.is_empty());
    assert_eq!(fs::read(&commit.path).unwrap(), after);
    drop(commit.file);
    File::clear_status_flags(&commit.path).unwrap();

    let file = File::open(&commit.path).unwrap();
    assert_published_values(&file);
    assert_eq!(file.root().attrs().unwrap().get("after"), None);
    file.close().unwrap();
}

struct FaultCommit {
    path: PathBuf,
    file: File,
    parents: [u64; 3],
    original: Vec<u8>,
    published: Vec<u8>,
    recording: Recording,
    event: usize,
    parent: usize,
}

impl FaultCommit {
    fn prepare(directory: &Path, parent: usize, write: HeapWrite) -> Self {
        let path = directory.join("failure.h5");
        let control_path = directory.join("control.h5");
        let parents = write_source(&path);
        fs::copy(&path, &control_path).unwrap();

        let (original, published, recording) = successful_commit(&control_path, &parents, write);
        let event = heap_write(&recording, parents[parent], write);

        let file = open_editor(&path, write);
        let before = fs::read(&path).unwrap();
        assert_eq!(before, original);
        assert_original_values(&file);
        stage(&file);
        Self {
            path,
            file,
            parents,
            original,
            published,
            recording,
            event,
            parent,
        }
    }

    fn fail_write(&self, written: usize) {
        disk_log::start();
        let prefix = &self.recording.events[..self.event];
        disk_log::fail_write_after(
            prefix
                .iter()
                .filter(|event| matches!(event, IoEvent::Write { .. }))
                .count(),
            written,
        );
    }

    fn assert_published(&self, failed: &Recording, written: usize) -> Vec<u8> {
        assert_failed_write(
            &self.recording,
            failed,
            self.event,
            written,
            &self.original,
            parent_range(self.parents[self.parent]),
        );

        let after = fs::read(&self.path).unwrap();
        assert_eq!(
            after[..Fixture::ROOT as usize],
            self.published[..Fixture::ROOT as usize]
        );
        assert_parents(&self.parents, &self.original, &after, self.parent);
        assert_published_values(&self.file);
        let stored = File::from_bytes(after.clone()).unwrap();
        assert_published_values(&stored);
        stored.close().unwrap();
        assert!(!self.file.has_staged_edits());
        after
    }
}

fn compaction_failure(error: Error, parent: usize) -> (HeapCompactionState, Error) {
    let Error::HeapCompactionFailed {
        completed,
        state,
        source,
    } = error
    else {
        panic!("expected a heap compaction failure, got {error:?}");
    };
    assert_eq!(completed, parent);
    (state, *source)
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

const READ_FAILURE: &str = "injected commit source read failure";

#[test]
fn preflight_heap_read_failure_preserves_staging_for_retry() {
    let directory = tempfile::tempdir().unwrap();
    let source = HeapReadSource::prepare(directory.path());
    let (file, armed) = source.open_with_fault(|read, _| {
        !disk_log::has_written_at(0) && read.start().get() == Fixture::ROOT
    });
    let original_free = file.reusable_free_sections().unwrap();

    let (error, failed) = fail_commit_read(&file, &armed);
    assert_read_error(error);
    assert!(failed.events.is_empty());
    assert_eq!(fs::read(&source.path).unwrap(), source.original);
    assert_eq!(file.reusable_free_sections().unwrap(), original_free);
    assert!(file.has_staged_edits());
    assert_original_values(&file);

    file.commit().unwrap();
    source.assert_closed_and_reopened(file, source.parents.len());
}

#[rstest]
#[case::move_source(HeapWrite::Move)]
#[case::free_header_undo(HeapWrite::FreeHeader)]
fn postpublication_heap_read_failure_restores_the_parent(#[case] write: HeapWrite) {
    let directory = tempfile::tempdir().unwrap();
    let source = HeapReadSource::prepare(directory.path());
    let (file, armed) = source.open_with_fault(move |read, parent| {
        disk_log::has_written_at(0)
            && read.start().get() == HeapWrite::FreeHeader.offset(parent)
            && read.length() == write.length() as u64
            && disk_log::has_written_at(HeapWrite::Move.offset(parent))
                == matches!(write, HeapWrite::FreeHeader)
    });
    let first = source.parents[0];

    let (error, failed) = fail_commit_read(&file, &armed);
    let (state, error) = compaction_failure(error, 0);
    assert_eq!(state, HeapCompactionState::Restored);
    assert_read_error(error);

    let next = heap_write(&source.recording, first, write);
    assert_eq!(failed.events[..next], source.recording.events[..next]);
    if matches!(write, HeapWrite::Move) {
        assert_eq!(failed.events.len(), next);
    } else {
        let record_start = parent_range(first).start + HEADER;
        let record = &source.original[record_start..record_start + RECORD];
        let operation = failed.events[next].operation().unwrap();
        assert_eq!(
            failed.operations[operation],
            DiskOp::Write {
                offset: HeapWrite::Move.offset(first),
                bytes: record.to_vec(),
            }
        );
    }
    let after = fs::read(&source.path).unwrap();
    assert_eq!(
        after[..Fixture::ROOT as usize],
        source.published[..Fixture::ROOT as usize]
    );
    assert_parents(&source.parents, &source.original, &after, 0);
    source.assert_closed_and_reopened(file, 0);
}

struct HeapReadSource {
    path: PathBuf,
    parents: [u64; 3],
    original: Vec<u8>,
    published: Vec<u8>,
    recording: Recording,
}

impl HeapReadSource {
    fn prepare(directory: &Path) -> Self {
        let path = directory.join("failure.h5");
        let control_path = directory.join("control.h5");
        let parents = write_source(&path);
        fs::copy(&path, &control_path).unwrap();
        let (original, published, recording) =
            successful_commit(&control_path, &parents, HeapWrite::Move);
        Self {
            path,
            parents,
            original,
            published,
            recording,
        }
    }

    fn open_with_fault(
        &self,
        matches: impl Fn(FileRange, u64) -> bool + Send + Sync + 'static,
    ) -> (File, Arc<Mutex<bool>>) {
        let armed = Arc::new(Mutex::new(false));
        let fault = Arc::clone(&armed);
        let first = self.parents[0];
        let file = open_read_fault(&self.path, move |read| {
            let mut fault = fault.lock().unwrap();
            if *fault && matches(read, first) {
                *fault = false;
                return Err(FormatError::Source(READ_FAILURE.into()));
            }
            Ok(())
        });
        assert_eq!(fs::read(&self.path).unwrap(), self.original);
        assert_original_values(&file);
        (file, armed)
    }

    fn assert_closed_and_reopened(&self, file: File, completed: usize) {
        assert!(!file.has_staged_edits());
        assert_published_values(&file);
        assert_allocated(&file, &self.parents);
        file.close().unwrap();
        let after = fs::read(&self.path).unwrap();
        assert_parents(&self.parents, &self.original, &after, completed);
        let reopened = File::open(&self.path).unwrap();
        assert_published_values(&reopened);
        reopened.close().unwrap();
    }
}

fn fail_commit_read(file: &File, armed: &Mutex<bool>) -> (Error, Recording) {
    stage(file);
    assert!(file.has_staged_edits());
    *armed.lock().unwrap() = true;
    disk_log::start();
    let error = file.commit().unwrap_err();
    let failed = disk_log::finish();
    assert!(!*armed.lock().unwrap());
    (error, failed)
}

fn assert_read_error(error: Error) {
    let Error::Format(error) = error else {
        panic!("expected a source read failure, got {error:?}");
    };
    assert_eq!(error, FormatError::Source(READ_FAILURE.into()));
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
    let mut fixture = Fixture::new();
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
