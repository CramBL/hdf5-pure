//! Checks stored ownership and further I/O after injected write, synchronization, and resize errors.

use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use rstest::rstest;
use test_util::allocation_oracle::AllocationKind;
use test_util::allocation_oracle::Coverage;
use test_util::disk_log;
use test_util::disk_log::DiskOp;
use test_util::disk_log::FailurePoint;
use test_util::disk_log::IoEvent;
use test_util::disk_log::IoOutcome;
use test_util::disk_log::Recording;
use test_util::publication_order::EofChange;
use test_util::publication_order::Policy;
use test_util::publication_order::Publication;

use crate::BufferedAppender;
use crate::DatasetBuilder;
use crate::Error;
use crate::File;
use crate::FileAccessProperties;
use crate::MemoryStrategy;
use crate::OwnedBufferedAppender;
use crate::SyncPolicy;
use crate::edit::WriteEngine;
use crate::edit::test_util as edit_test_util;
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;

use super::CHUNK;
use super::Fact;
use super::Observation;
use super::Workload;

#[rstest]
#[case::immediate(Action::Immediate)]
#[case::borrowed_automatic(Action::Borrowed(Flush::Automatic))]
#[case::borrowed_manual(Action::Borrowed(Flush::Manual))]
#[case::borrowed_finish(Action::Borrowed(Flush::Finish))]
#[case::borrowed_drop(Action::Borrowed(Flush::Drop))]
#[case::owned_automatic(Action::Owned(Flush::Automatic))]
#[case::owned_manual(Action::Owned(Flush::Manual))]
#[case::owned_finish(Action::Owned(Flush::Finish))]
#[case::owned_drop(Action::Owned(Flush::Drop))]
fn actual_appender_failures_preserve_ownership_and_stop_teardown(
    #[case] action: Action,
    #[values(false, true)] deferred: bool,
    #[values(false, true)] filtered: bool,
    #[values(
        Fault::Content(WriteFault::Before),
        Fault::Content(WriteFault::Partial),
        Fault::Content(WriteFault::After),
        Fault::Ownership(FailurePoint::Before),
        Fault::Ownership(FailurePoint::After)
    )]
    fault: Fault,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] policy: SyncPolicy,
) {
    appender_failure(action, deferred, filtered, policy, fault);
}

#[rstest]
#[case::borrowed_automatic(Action::Borrowed(Flush::Automatic))]
#[case::borrowed_manual(Action::Borrowed(Flush::Manual))]
#[case::borrowed_finish(Action::Borrowed(Flush::Finish))]
#[case::borrowed_drop(Action::Borrowed(Flush::Drop))]
#[case::owned_automatic(Action::Owned(Flush::Automatic))]
#[case::owned_manual(Action::Owned(Flush::Manual))]
#[case::owned_finish(Action::Owned(Flush::Finish))]
#[case::owned_drop(Action::Owned(Flush::Drop))]
fn actual_appender_barrier_failures_preserve_ownership_and_stop_teardown(
    #[case] action: Action,
    #[values(false, true)] deferred: bool,
    #[values(
        Fault::ContentSync(FailurePoint::Before),
        Fault::ContentSync(FailurePoint::After),
        Fault::OwnershipSync(FailurePoint::Before),
        Fault::OwnershipSync(FailurePoint::After)
    )]
    fault: Fault,
) {
    appender_failure(action, deferred, true, SyncPolicy::Always, fault);
}

fn appender_failure(
    action: Action,
    deferred: bool,
    filtered: bool,
    policy: SyncPolicy,
    fault: Fault,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("lifecycle.h5");
    let hole = super::pure_fixture(&path, false, deferred, filtered);
    failure_on(&path, action, policy, fault, &hole);
}

/// Checks a successful append's prefixes, an injected error, and subsequent retry or rejection.
///
/// Uses the fixture at `path` and checks exact unused coverage of `hole` after each interruption.
pub(super) fn failure_on(
    path: &Path,
    action: Action,
    policy: SyncPolicy,
    fault: Fault,
    hole: &Coverage,
) {
    let elements = action.elements();
    let workload = Workload::record_elements(path, elements, |values| {
        let file = open(path, policy);
        action.run(&file, values, true).unwrap();
        file.close().unwrap();
    });
    let replay = tempfile::tempdir().unwrap();
    workload.replay(&replay.path().join("prefix.h5"), hole, |_, _, _| {});
    let boundary = Boundary::read(&workload, policy);
    let event = fault.event(&workload, &boundary);
    fs::write(path, &workload.base).unwrap();
    let file = open(path, policy);
    let values = (workload.initial_values..workload.initial_values + elements)
        .map(|value| i32::try_from(value).unwrap())
        .collect::<Vec<_>>();
    disk_log::start();
    fault.install(&workload.events, event);
    let result = action.run(&file, &values, fault.known_unpublished());
    if action.drops() {
        result.unwrap();
    } else {
        assert_io(result.unwrap_err(), fault.synchronizes());
    }
    disk_log::clear_failures();
    let recording = disk_log::finish();
    assert_failed(
        &workload.events,
        &recording,
        event,
        fault,
        u64::try_from(workload.base.len()).unwrap(),
    );
    let interrupted = fs::read(path).unwrap();
    reconstruct(&workload.base, &recording, &interrupted);
    let snapshot = Observation::read(&interrupted);
    assert_eq!(snapshot.values.len(), workload.initial_values);
    snapshot.inspect(
        "appender actual I/O failure",
        event,
        hole,
        &workload.published_history(workload.completed_prefix(&recording)),
    );
    disk_log::start();
    if fault.known_unpublished() {
        let mut dataset = file.dataset("d").unwrap();
        dataset.append(&values).unwrap();
        drop(dataset);
        file.close().unwrap();
        let retry = disk_log::finish();
        assert!(!retry.events.is_empty());
        let completed = fs::read(path).unwrap();
        reconstruct(&interrupted, &retry, &completed);
        let final_state = Observation::read(&completed);
        assert_eq!(final_state.values.len(), workload.initial_values + elements);
        final_state.inspect("appender retry", event, hole, &[]);
    } else {
        assert_uncertain(file.dataset("d").unwrap().append(&values).unwrap_err());
        assert_uncertain(file.commit().unwrap_err());
        assert_uncertain(file.sync().unwrap_err());
        assert_uncertain(file.close().unwrap_err());
        assert!(disk_log::finish().events.is_empty());
        assert_eq!(fs::read(path).unwrap(), interrupted);
    }
}

/// The immediate or buffered append operation exercised by a failure case.
#[derive(Clone, Copy, Debug)]
pub(super) enum Action {
    Borrowed(Flush),
    Immediate,
    Owned(Flush),
}

impl Action {
    fn elements(self) -> usize {
        match self {
            Self::Borrowed(Flush::Automatic) | Self::Immediate | Self::Owned(Flush::Automatic) => {
                CHUNK
            }
            _ => 17,
        }
    }

    fn drops(self) -> bool {
        matches!(self, Self::Borrowed(Flush::Drop) | Self::Owned(Flush::Drop))
    }

    fn run(self, file: &File, values: &[i32], known_unpublished: bool) -> Result<(), Error> {
        let mut dataset = file.dataset("d").unwrap();
        match self {
            Self::Immediate => dataset.append(values),
            Self::Borrowed(flush) => Buffer::Borrowed(dataset.buffered_appender().unwrap()).run(
                values,
                flush,
                known_unpublished,
            ),
            Self::Owned(flush) => Buffer::Owned(dataset.into_buffered_appender().unwrap()).run(
                values,
                flush,
                known_unpublished,
            ),
        }
    }
}

/// The buffered flush mode exercised by a failure case.
#[derive(Clone, Copy, Debug)]
pub(super) enum Flush {
    Automatic,
    Drop,
    Finish,
    Manual,
}

enum Buffer<'a> {
    Borrowed(BufferedAppender<'a>),
    Owned(OwnedBufferedAppender),
}

impl Buffer<'_> {
    fn run(mut self, values: &[i32], flush: Flush, known_unpublished: bool) -> Result<(), Error> {
        let result = match &mut self {
            Self::Borrowed(buffer) => buffer.append(values),
            Self::Owned(buffer) => buffer.append(values),
        };
        if let Err(error) = result {
            self.assert_stopped(values, known_unpublished);
            return Err(error);
        }
        match flush {
            Flush::Automatic | Flush::Finish => match self {
                Self::Borrowed(buffer) => buffer.finish(),
                Self::Owned(buffer) => buffer.finish(),
            },
            Flush::Drop => {
                drop(self);
                Ok(())
            }
            Flush::Manual => {
                let result = match &mut self {
                    Self::Borrowed(buffer) => buffer.flush(),
                    Self::Owned(buffer) => buffer.flush(),
                };
                if result.is_err() {
                    self.assert_stopped(values, known_unpublished);
                }
                result
            }
        }
    }

    fn finish(self) -> Result<(), Error> {
        match self {
            Self::Borrowed(buffer) => buffer.finish(),
            Self::Owned(buffer) => buffer.finish(),
        }
    }

    fn assert_stopped(&mut self, values: &[i32], known_unpublished: bool) {
        let expected = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let (unwritten, error) = match self {
            Self::Borrowed(buffer) => (buffer.unwritten().to_vec(), buffer.flush().unwrap_err()),
            Self::Owned(buffer) => (buffer.unwritten().to_vec(), buffer.flush().unwrap_err()),
        };
        assert_eq!(unwritten, expected);
        if !known_unpublished {
            assert_uncertain(error);
            return;
        }
        let Error::AppendInPlaceUnsupported(reason) = error else {
            panic!("expected a poisoned buffer, got {error:?}")
        };
        assert_eq!(
            reason,
            "this buffered appender failed a write and holds unwritten elements; read them back with BufferedAppender::unwritten and start a new appender"
        );
    }
}

/// Event indices for preparing content, publishing ownership, and storing a reference.
struct Boundary {
    /// The first write before ownership publication.
    content: usize,
    /// The superblock write for publishing ownership metadata.
    ownership: usize,
    /// The first stored allocation reference, or the event count for commit and close cases.
    reference: usize,
}

impl Boundary {
    fn read(workload: &Workload, policy: SyncPolicy) -> Self {
        let ownership = workload
            .events
            .iter()
            .position(|event| matches!(event, IoEvent::Write { offset: 0, .. }))
            .unwrap();
        let content = workload.events[..ownership]
            .iter()
            .position(|event| matches!(event, IoEvent::Write { .. }))
            .unwrap();
        let published = workload.events[ownership].operation().unwrap() + 1;
        let owned = &workload.snapshots[published];
        let allocation = workload
            .candidates
            .iter()
            .find(|fact| {
                fact.kind
                    == if workload.snapshots[0].index_defined {
                        AllocationKind::Payload
                    } else {
                        AllocationKind::Index
                    }
            })
            .unwrap();
        let complete = Coverage::new([allocation.range]);
        assert!(complete.intersection(&owned.free).is_empty());
        assert!(allocation.range.end() <= owned.eof);
        assert!(
            !complete
                .intersection(&workload.snapshots[0].free)
                .is_empty()
        );
        let reference_point = workload
            .snapshots
            .iter()
            .position(|snapshot| {
                if workload.snapshots[0].index_defined {
                    snapshot.live.contains(allocation)
                } else {
                    snapshot.index_defined
                }
            })
            .unwrap();
        let reference = workload
            .events
            .iter()
            .position(|event| event.operation() == Some(reference_point - 1))
            .unwrap();
        let eof = if owned.eof > workload.snapshots[published - 1].eof {
            EofChange::Extend
        } else {
            EofChange::Unchanged
        };
        let prepared = (content..ownership)
            .filter(|&event| matches!(workload.events[event], IoEvent::Write { .. }))
            .collect::<Vec<_>>();
        let recording = Recording {
            operations: workload.operations.clone(),
            events: workload.events.clone(),
        };
        Publication::parse(&recording, &prepared, ownership, reference, eof)
            .unwrap()
            .require(match policy {
                SyncPolicy::Always => Policy::Always,
                SyncPolicy::OnClose => Policy::OnClose,
            })
            .unwrap();
        assert!(
            recording
                .events
                .iter()
                .all(|event| event.outcome() == IoOutcome::Succeeded)
        );
        Self {
            content,
            ownership,
            reference,
        }
    }
}

/// The written prefix before reporting an injected write error.
#[derive(Clone, Copy, Debug)]
pub(super) enum WriteFault {
    /// All bytes passed to the write.
    After,
    /// Zero bytes.
    Before,
    /// Half the bytes passed to the write, rounded down.
    Partial,
}

/// The write or synchronization call selected for an injected error.
#[derive(Clone, Copy, Debug)]
pub(super) enum Fault {
    /// The first content write before ownership publication.
    Content(WriteFault),
    /// The synchronization between content writes and ownership publication.
    ContentSync(FailurePoint),
    /// The last synchronization during close.
    FinalSync(FailurePoint),
    /// The superblock write for publishing ownership metadata.
    Ownership(FailurePoint),
    /// The synchronization after ownership publication.
    OwnershipSync(FailurePoint),
}

impl Fault {
    fn event(self, workload: &Workload, boundary: &Boundary) -> usize {
        match self {
            Self::Content(_) => boundary.content,
            Self::FinalSync(_) => workload
                .events
                .iter()
                .rposition(|event| matches!(event, IoEvent::Synchronize { .. }))
                .unwrap(),
            Self::Ownership(_) => boundary.ownership,
            Self::ContentSync(_) => (boundary.content + 1..boundary.ownership)
                .find(|&event| matches!(workload.events[event], IoEvent::Synchronize { .. }))
                .unwrap(),
            Self::OwnershipSync(_) => (boundary.ownership + 1..boundary.reference)
                .find(|&event| matches!(workload.events[event], IoEvent::Synchronize { .. }))
                .unwrap(),
        }
    }

    fn synchronizes(self) -> bool {
        matches!(
            self,
            Self::ContentSync(_) | Self::OwnershipSync(_) | Self::FinalSync(_)
        )
    }

    fn known_unpublished(self) -> bool {
        matches!(self, Self::Content(_) | Self::ContentSync(_))
    }

    fn install(self, events: &[IoEvent], event: usize) {
        match self {
            Self::ContentSync(point) | Self::OwnershipSync(point) | Self::FinalSync(point) => {
                disk_log::fail_sync_after(
                    events[..event]
                        .iter()
                        .filter(|event| matches!(event, IoEvent::Synchronize { .. }))
                        .count(),
                    point,
                )
            }
            Self::Content(_) | Self::Ownership(_) => disk_log::fail_write_after(
                events[..event]
                    .iter()
                    .filter(|event| matches!(event, IoEvent::Write { .. }))
                    .count(),
                self.prefix(&events[event]),
            ),
        }
    }

    fn prefix(self, event: &IoEvent) -> usize {
        let IoEvent::Write { requested, .. } = event else {
            panic!("expected write event")
        };
        match self {
            Self::Content(WriteFault::After) | Self::Ownership(FailurePoint::After) => *requested,
            Self::Content(WriteFault::Before) | Self::Ownership(FailurePoint::Before) => 0,
            Self::Content(WriteFault::Partial) => requested / 2,
            _ => panic!("a synchronization has no write prefix"),
        }
    }
}

fn open(path: &Path, policy: SyncPolicy) -> File {
    File::open_rw_with_options(
        path,
        FileAccessProperties::new()
            .with_sync_policy(policy)
            .with_memory_strategy(MemoryStrategy::Bounded),
    )
    .unwrap()
}

fn assert_io(error: Error, synchronization: bool) {
    let Error::Io(error) = error else {
        panic!("expected injected I/O, got {error:?}")
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(
        error.to_string(),
        if synchronization {
            "injected disk synchronization failure"
        } else {
            "injected disk write failure"
        }
    );
}

fn assert_uncertain(error: Error) {
    assert!(
        matches!(error, Error::AllocationPublicationUncertain),
        "{error:?}"
    );
}

fn reconstruct(base: &[u8], recording: &Recording, expected: &[u8]) {
    let mut image = base.to_vec();
    for operation in &recording.operations {
        operation.apply(&mut image);
    }
    assert_eq!(image, expected);
}

fn assert_failed(
    events: &[IoEvent],
    failed: &Recording,
    event: usize,
    fault: Fault,
    original_length: u64,
) {
    assert_eq!(&failed.events[..event], &events[..event]);
    assert_eq!(
        failed.events[event].outcome(),
        IoOutcome::Failed(ErrorKind::Other)
    );
    if fault.known_unpublished() {
        assert!(failed.events[event + 1..].iter().all(|event| matches!(event, IoEvent::Resize { length, outcome: IoOutcome::Succeeded, .. } if *length == original_length)));
    } else {
        assert_eq!(failed.events.len(), event + 1);
    }
    match (&failed.events[event], &events[event]) {
        (
            IoEvent::Write {
                offset,
                requested,
                written,
                ..
            },
            IoEvent::Write {
                offset: expected_offset,
                requested: expected_length,
                ..
            },
        ) => {
            assert_eq!((offset, requested), (expected_offset, expected_length));
            assert_eq!(*written, fault.prefix(&events[event]));
        }
        (
            IoEvent::Synchronize { kind, .. },
            IoEvent::Synchronize {
                kind: expected_kind,
                ..
            },
        ) => assert_eq!(kind, expected_kind),
        _ => panic!("control and failed operations differ"),
    }
}

impl Workload {
    /// Counts consecutive completed operations with matching write offsets and operation lengths.
    ///
    /// Compares write locations and lengths because object-header timestamps can differ between
    /// the control and failure executions.
    fn completed_prefix(&self, recording: &Recording) -> usize {
        self.operations
            .iter()
            .zip(&recording.operations)
            .take_while(|(control, actual)| match (control, actual) {
                (
                    DiskOp::Write { offset, bytes },
                    DiskOp::Write {
                        offset: at,
                        bytes: written,
                    },
                ) => offset == at && bytes.len() == written.len(),
                (DiskOp::SetLen(length), DiskOp::SetLen(actual)) => length == actual,
                _ => false,
            })
            .count()
    }
}

#[rstest]
#[case::content_before(Fault::Content(WriteFault::Before))]
#[case::content_partial(Fault::Content(WriteFault::Partial))]
#[case::content_after(Fault::Content(WriteFault::After))]
#[case::root_before(Fault::Ownership(FailurePoint::Before))]
#[case::root_after(Fault::Ownership(FailurePoint::After))]
fn actual_commit_and_close_write_failures_preserve_allocations(
    #[case] fault: Fault,
    #[values(Terminal::Commit, Terminal::Close)] operation: Terminal,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] policy: SyncPolicy,
    #[values(false, true)] cleanup: bool,
) {
    terminal_failure(operation, policy, fault, cleanup);
}

#[rstest]
#[case::content_before(Fault::ContentSync(FailurePoint::Before))]
#[case::content_after(Fault::ContentSync(FailurePoint::After))]
#[case::root_before(Fault::OwnershipSync(FailurePoint::Before))]
#[case::root_after(Fault::OwnershipSync(FailurePoint::After))]
fn actual_commit_and_close_barrier_failures_preserve_allocations(
    #[case] fault: Fault,
    #[values(Terminal::Commit, Terminal::Close)] operation: Terminal,
    #[values(false, true)] cleanup: bool,
) {
    terminal_failure(operation, SyncPolicy::Always, fault, cleanup);
}

#[rstest]
#[case::before(FailurePoint::Before)]
#[case::after(FailurePoint::After)]
fn actual_close_synchronization_failure_blocks_later_publication(
    #[case] point: FailurePoint,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] policy: SyncPolicy,
    #[values(false, true)] cleanup: bool,
) {
    terminal_failure(Terminal::Close, policy, Fault::FinalSync(point), cleanup);
}

fn terminal_failure(operation: Terminal, policy: SyncPolicy, fault: Fault, cleanup: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("terminal.h5");
    let hole = super::pure_fixture(&path, false, false, true);
    let fixture = fs::read(&path).unwrap();
    let file = open(&path, policy);
    let retired = prepare_terminal(&file, cleanup);
    let mut workload = Workload::record_elements(&path, 0, |_| {
        operation.run(&file).unwrap();
        if !cleanup {
            assert_eq!(file.root().groups().unwrap(), ["pending"]);
        }
        if matches!(operation, Terminal::Commit) {
            file.clone().close().unwrap();
        }
    });
    drop(file);
    if let Some(retired) = retired {
        assert!(!workload.snapshots[0].live.contains(&retired));
        assert!(
            Coverage::new([retired.range])
                .intersection(&workload.snapshots[0].free)
                .is_empty()
        );
        assert!(retired.range.end() <= workload.snapshots[0].eof);
        let completed = workload.snapshots.last().unwrap();
        assert!(
            Coverage::new([retired.range])
                .difference(&Coverage::new(
                    completed
                        .free
                        .intervals()
                        .iter()
                        .copied()
                        .chain(completed.live.iter().map(|fact| fact.range))
                ))
                .is_empty()
        );
        workload.history.push((0, retired));
    }
    workload.replay(&directory.path().join("prefix.h5"), &hole, |_, _, _| {});
    let ownership = workload
        .events
        .iter()
        .position(|event| matches!(event, IoEvent::Write { offset: 0, .. }))
        .unwrap();
    let content = workload
        .events
        .iter()
        .position(|event| matches!(event, IoEvent::Write { .. }))
        .unwrap();
    let boundary = Boundary {
        content,
        ownership,
        reference: workload.events.len(),
    };
    let event = fault.event(&workload, &boundary);
    fs::write(&path, &fixture).unwrap();
    let file = open(&path, policy);
    assert_eq!(prepare_terminal(&file, cleanup), retired);
    assert_eq!(fs::read(&path).unwrap(), workload.base);
    let dataset = file.dataset("d").unwrap();
    let layout = dataset.data_layout().unwrap();
    disk_log::start();
    fault.install(&workload.events, event);
    assert_io(operation.run(&file).unwrap_err(), fault.synchronizes());
    disk_log::clear_failures();
    let recording = disk_log::finish();
    assert_failed(
        &workload.events,
        &recording,
        event,
        fault,
        u64::try_from(workload.base.len()).unwrap(),
    );
    let interrupted = fs::read(&path).unwrap();
    reconstruct(&workload.base, &recording, &interrupted);
    let snapshot = Observation::read(&interrupted);
    snapshot.inspect(
        "terminal actual I/O failure",
        event,
        &hole,
        &workload.published_history(workload.completed_prefix(&recording)),
    );
    assert_eq!(snapshot.values, workload.snapshots[0].values);
    let groups = File::from_bytes(interrupted.clone())
        .unwrap()
        .root()
        .groups()
        .unwrap();
    assert_eq!(
        groups,
        if !cleanup
            && matches!(
                fault,
                Fault::Ownership(FailurePoint::After)
                    | Fault::OwnershipSync(_)
                    | Fault::FinalSync(_)
            )
        {
            vec!["pending"]
        } else {
            Vec::<&str>::new()
        }
    );
    disk_log::start();
    if fault.known_unpublished() {
        file.root().create_group("retry").unwrap();
        file.commit().unwrap();
        assert_eq!(dataset.data_layout().unwrap(), layout);
        assert_eq!(dataset.read_i32().unwrap(), snapshot.values);
        drop(dataset);
        file.close().unwrap();
        let retry = disk_log::finish();
        assert!(!retry.events.is_empty());
        let completed = fs::read(&path).unwrap();
        reconstruct(&interrupted, &retry, &completed);
        Observation::read(&completed).inspect("terminal retry", event, &hole, &[]);
        assert_eq!(
            File::from_bytes(completed)
                .unwrap()
                .root()
                .groups()
                .unwrap(),
            ["retry"]
        );
    } else {
        assert_uncertain(file.commit().unwrap_err());
        assert_uncertain(file.sync().unwrap_err());
        let mut writer = file.dataset("d").unwrap();
        assert_uncertain(writer.append(&[i32::try_from(CHUNK).unwrap()]).unwrap_err());
        drop(writer);
        assert_uncertain(file.close().unwrap_err());
        drop(dataset);
        assert!(disk_log::finish().events.is_empty());
        assert_eq!(fs::read(&path).unwrap(), interrupted);
    }
}

fn prepare_terminal(file: &File, cleanup: bool) -> Option<Fact> {
    if !cleanup {
        file.root().create_group("pending").unwrap();
        return None;
    }
    let mut dataset = file.dataset("d").unwrap();
    dataset
        .append(
            &(CHUNK..CHUNK + 17)
                .map(|value| i32::try_from(value).unwrap())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let chunk = dataset.chunks().unwrap().pop().unwrap();
    let retired = Fact::new(AllocationKind::Payload, chunk.address, chunk.storage_size);
    dataset
        .append(&[i32::try_from(CHUNK + 17).unwrap()])
        .unwrap();
    assert!(!file.has_staged_edits());
    Some(retired)
}

#[derive(Clone, Copy, Debug)]
enum Terminal {
    Close,
    Commit,
}

impl Terminal {
    fn run(self, file: &File) -> Result<(), Error> {
        match self {
            Self::Close => file.clone().close(),
            Self::Commit => file.commit(),
        }
    }
}

#[rstest]
#[case::before(FailurePoint::Before)]
#[case::after(FailurePoint::After)]
fn actual_truncation_failure_preserves_published_source_bounds(
    #[case] point: FailurePoint,
    #[values(false, true)] paged: bool,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] policy: SyncPolicy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("truncation.h5");
    super::pure_fixture(&path, paged, false, false);
    let mut engine = open_engine(&path, policy);
    let mut tail = DatasetBuilder::new("tail");
    tail.with_u8_data(&vec![17; 262_144]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("tail"), tail)
        .unwrap();
    engine.commit().unwrap();
    engine.finalize_persist().unwrap();
    drop(engine);
    let base = fs::read(&path).unwrap();
    let mut writer = open_engine(&path, policy);
    writer.delete(&ObjectPathBuf::parse("tail")).unwrap();
    disk_log::start();
    writer.commit().unwrap();
    writer.finalize_persist().unwrap();
    drop(writer);
    let control = disk_log::finish();
    let completed = fs::read(&path).unwrap();
    reconstruct(&base, &control, &completed);
    assert!(completed.len() < base.len());
    assert_eq!(
        Observation::read(&completed).eof,
        u64::try_from(completed.len()).unwrap()
    );
    let mut image = base.clone();
    let mut shrink = None;
    for (operation, entry) in control.operations.iter().enumerate() {
        if let DiskOp::SetLen(length) = entry
            && *length < u64::try_from(image.len()).unwrap()
            && shrink.is_none()
        {
            shrink = Some((operation, *length, image.clone()));
        }
        entry.apply(&mut image);
        let observation = Observation::read(&image);
        assert_source(&observation, &image);
        let reader = File::from_bytes(image.clone()).unwrap();
        let mut datasets = reader.root().datasets().unwrap();
        datasets.sort();
        if datasets.iter().any(|path| path == "tail") {
            assert_eq!(datasets, ["ceiling", "d", "tail"]);
            assert_eq!(
                reader.dataset("tail").unwrap().read_u8().unwrap(),
                vec![17; 262_144]
            );
        } else {
            assert_eq!(datasets, ["ceiling", "d"]);
        }
        assert_eq!(
            observation.values,
            (0..i32::try_from(CHUNK).unwrap()).collect::<Vec<_>>()
        );
    }
    let (operation, length, before) = shrink.unwrap();
    let event = control
        .events
        .iter()
        .position(|event| event.operation() == Some(operation))
        .unwrap();
    assert!(length < u64::try_from(before.len()).unwrap());
    assert_eq!(Observation::read(&before).eof, length);
    fs::write(&path, &base).unwrap();
    let mut writer = open_engine(&path, policy);
    writer.delete(&ObjectPathBuf::parse("tail")).unwrap();
    disk_log::start();
    disk_log::fail_resize_after(
        control.events[..event]
            .iter()
            .filter(|event| matches!(event, IoEvent::Resize { .. }))
            .count(),
        point,
    );
    let error = writer
        .commit()
        .and_then(|()| writer.finalize_persist())
        .unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected resize I/O error, got {error:?}")
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk resize failure");
    disk_log::clear_failures();
    let failed = disk_log::finish();
    assert_eq!(&failed.events[..event], &control.events[..event]);
    assert_eq!(failed.events.len(), event + 1);
    assert_eq!(
        failed.events[event],
        IoEvent::Resize {
            length,
            outcome: IoOutcome::Failed(ErrorKind::Other),
            operation: (point == FailurePoint::After).then_some(operation),
        }
    );
    let interrupted = fs::read(&path).unwrap();
    reconstruct(&base, &failed, &interrupted);
    assert_eq!(
        interrupted,
        if point == FailurePoint::After {
            before[..usize::try_from(length).unwrap()].to_vec()
        } else {
            before
        }
    );
    let observation = Observation::read(&interrupted);
    assert_eq!(observation.eof, length);
    assert!(observation.eof <= u64::try_from(interrupted.len()).unwrap());
    assert_source(&observation, &interrupted);
    disk_log::start();
    assert_uncertain(writer.commit().unwrap_err());
    assert_uncertain(writer.finalize_persist().unwrap_err());
    assert_uncertain(writer.force_sync().unwrap_err());
    drop(writer);
    assert!(disk_log::finish().events.is_empty());
    assert_eq!(fs::read(&path).unwrap(), interrupted);
    for cycle in 0..3 {
        let file = open(&path, policy);
        let mut dataset = file.dataset("d").unwrap();
        let start = dataset.read_i32().unwrap().len();
        dataset.append(&[i32::try_from(start).unwrap()]).unwrap();
        drop(dataset);
        file.close().unwrap();
        let image = fs::read(&path).unwrap();
        let observation = Observation::read(&image);
        assert_eq!(observation.values.len(), CHUNK + cycle + 1);
        assert_source(&observation, &image);
    }
}

fn open_engine(path: &Path, policy: SyncPolicy) -> WriteEngine {
    let mut engine = edit_test_util::open_engine(path);
    engine.set_sync_policy(policy);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine
}

fn assert_source(observation: &Observation, image: &[u8]) {
    assert!(
        observation
            .free
            .intersection(&observation.reachable())
            .is_empty()
    );
    assert!(
        observation
            .free
            .intervals()
            .iter()
            .all(|interval| interval.end() <= observation.eof)
    );
    assert!(observation.eof <= u64::try_from(image.len()).unwrap());
}

#[rstest]
#[case::borrowed(false)]
#[case::owned(true)]
fn unrelated_commits_preserve_active_appender_and_object_identities(
    #[case] owned: bool,
    #[values(false, true)] deferred: bool,
    #[values(false, true)] filtered: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("active.h5");
    let hole = super::pure_fixture(&path, false, deferred, filtered);
    let file = open(&path, SyncPolicy::Always);
    let mut dataset = file.dataset("d").unwrap();
    let identity = dataset.header_address().unwrap();
    let ceiling = file.dataset("ceiling").unwrap().header_address().unwrap();
    let initial = dataset.read_i32().unwrap().len();
    let mut buffer = if owned {
        Buffer::Owned(file.dataset("d").unwrap().into_buffered_appender().unwrap())
    } else {
        Buffer::Borrowed(dataset.buffered_appender().unwrap())
    };
    let values = (initial..initial + 17)
        .map(|value| i32::try_from(value).unwrap())
        .collect::<Vec<_>>();
    match &mut buffer {
        Buffer::Borrowed(buffer) => buffer.append(&values).unwrap(),
        Buffer::Owned(buffer) => buffer.append(&values).unwrap(),
    }
    file.root().create_group("first").unwrap();
    file.commit().unwrap();
    assert_eq!(
        file.dataset("d").unwrap().header_address().unwrap(),
        identity
    );
    assert_eq!(
        file.dataset("ceiling").unwrap().header_address().unwrap(),
        ceiling
    );
    file.root().create_group("second").unwrap();
    match &mut buffer {
        Buffer::Borrowed(buffer) => buffer.flush().unwrap(),
        Buffer::Owned(buffer) => buffer.flush().unwrap(),
    }
    assert!(file.has_staged_edits());
    assert_eq!(
        File::from_bytes(fs::read(&path).unwrap())
            .unwrap()
            .root()
            .groups()
            .unwrap(),
        ["first"]
    );
    file.commit().unwrap();
    buffer.finish().unwrap();
    assert_eq!(
        file.dataset("d").unwrap().header_address().unwrap(),
        identity
    );
    assert_eq!(
        file.dataset("ceiling").unwrap().header_address().unwrap(),
        ceiling
    );
    assert_eq!(file.root().groups().unwrap(), ["first", "second"]);
    drop(dataset);
    file.close().unwrap();
    let image = fs::read(&path).unwrap();
    let observation = Observation::read(&image);
    assert_eq!(observation.values.len(), initial + 17);
    observation.inspect("active appender after unrelated commits", 0, &hole, &[]);
}
