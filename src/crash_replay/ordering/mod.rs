//! Checks storage call ordering, synchronization failures, and readable negative controls.

use std::fmt;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use hdf5_pure_format::__private::FreeSpaceManagerHeader;
use rstest::rstest;
use test_util::disk_log;
use test_util::disk_log::DiskOp;
use test_util::disk_log::FailurePoint;
use test_util::disk_log::IoEvent;
use test_util::disk_log::IoOutcome;
use test_util::disk_log::Recording;
use test_util::disk_log::Synchronization;
use test_util::publication_order::EofChange;
use test_util::publication_order::OrderError;
use test_util::publication_order::Policy;
use test_util::publication_order::Publication;

use crate::File;
use crate::FileBuilder;
use crate::FileLocking;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::MaxExtent;
use crate::MemoryStrategy;
use crate::PersistedFreeSpacePolicy;
use crate::SyncPolicy;
use crate::edit::AppendBuilder;
use crate::edit::AppendTarget;
use crate::edit::WriteEngine;
use crate::error::Error;
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;
use crate::source::MetadataCacheConfig;
use crate::width::FormatWidths;

use super::ordering_facts::PublicationFacts;

mod pages;

#[rstest]
#[case::flat_durable(
    false,
    false,
    false,
    MemoryStrategy::Mirrored,
    SyncPolicy::Always,
    Buffering::Operation,
    2
)]
#[case::page_durable(
    true,
    false,
    false,
    MemoryStrategy::Bounded,
    SyncPolicy::Always,
    Buffering::Unbuffered,
    24
)]
#[case::flat_ordered(
    false,
    false,
    false,
    MemoryStrategy::Bounded,
    SyncPolicy::OnClose,
    Buffering::Operation,
    24
)]
#[case::page_ordered(
    true,
    false,
    false,
    MemoryStrategy::Mirrored,
    SyncPolicy::OnClose,
    Buffering::Unbuffered,
    2
)]
#[case::flat_explicit(
    false,
    false,
    true,
    MemoryStrategy::Bounded,
    SyncPolicy::Always,
    Buffering::Session,
    2
)]
#[case::page_explicit(
    true,
    false,
    true,
    MemoryStrategy::Mirrored,
    SyncPolicy::OnClose,
    Buffering::Session,
    24
)]
#[case::flat_initialization(
    false,
    true,
    false,
    MemoryStrategy::Bounded,
    SyncPolicy::Always,
    Buffering::Operation,
    2
)]
#[case::page_initialization(
    true,
    true,
    true,
    MemoryStrategy::Mirrored,
    SyncPolicy::Always,
    Buffering::Unbuffered,
    2
)]
#[case::flat_initialization_explicit(
    false,
    true,
    true,
    MemoryStrategy::Mirrored,
    SyncPolicy::OnClose,
    Buffering::Session,
    2
)]
#[case::page_initialization_explicit(
    true,
    true,
    false,
    MemoryStrategy::Bounded,
    SyncPolicy::Always,
    Buffering::Session,
    2
)]
fn allocation_publication_orders_actual_storage_io(
    #[case] paged: bool,
    #[case] deferred: bool,
    #[case] filtered: bool,
    #[case] backing: MemoryStrategy,
    #[case] policy: SyncPolicy,
    #[case] buffering: Buffering,
    #[case] rounds: i32,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("ordered.h5");
    let observed = record(
        &path,
        Fixture::Persistent {
            paged,
            deferred,
            filtered,
        },
        backing,
        policy,
        buffering,
        rounds,
    );
    let gates = observed.gates();
    assert!(!gates.is_empty(), "no ownership publications: {observed:?}");
    for gate in &gates {
        gate.parse(&observed.trace).unwrap().require(oracle_policy(policy)).unwrap_or_else(|error| {
            panic!("{paged:?}/{deferred:?}/{filtered:?}/{backing:?}/{policy:?}/{buffering:?}, publication {gate:?}: {error:?}; events {:?}", observed.trace.events)
        });
    }
    let mut managers = 0;
    for operation in &observed.trace.operations {
        if let DiskOp::Write { bytes, .. } = operation {
            for offset in
                test_util::bytes::signature_offsets(bytes, test_util::free_space::SIGNATURE)
            {
                FreeSpaceManagerHeader::parse(
                    FormatWidths::from_sizes(8, 8).unwrap(),
                    &bytes[offset..],
                )
                .unwrap();
                managers += 1;
            }
        }
    }
    assert!(managers > 0, "no replacement manager writes: {observed:?}");
    if policy == SyncPolicy::OnClose {
        assert!(
            !observed
                .trace
                .events
                .iter()
                .any(|event| matches!(event, IoEvent::Synchronize { .. })),
            "OnClose requests a durability barrier: {observed:?}"
        );
    }
}

#[rstest]
#[case::operation(Buffering::Operation)]
#[case::session(Buffering::Session)]
#[case::unbuffered(Buffering::Unbuffered)]
fn growing_eoa_requires_full_metadata_synchronization(#[case] buffering: Buffering) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("growing.h5");
    let observed = record(
        &path,
        Fixture::AppendOnly,
        MemoryStrategy::Bounded,
        SyncPolicy::Always,
        buffering,
        2,
    );
    let gates = observed.gates();
    assert!(
        gates.iter().any(|gate| gate.eof == EofChange::Extend),
        "no EOF change: {observed:?}"
    );
    for gate in gates {
        gate.parse(&observed.trace)
            .unwrap()
            .require(Policy::Always)
            .unwrap_or_else(|error| {
                panic!("{buffering:?}, publication {gate:?}: {error:?}; {observed:?}")
            });
    }
}

#[rstest]
#[case::allocated_content_before(false, Failure::Content(FailurePoint::Before), SyncPolicy::Always)]
#[case::allocated_content_after(false, Failure::Content(FailurePoint::After), SyncPolicy::Always)]
#[case::initial_content_before(true, Failure::Content(FailurePoint::Before), SyncPolicy::Always)]
#[case::initial_content_after(true, Failure::Content(FailurePoint::After), SyncPolicy::Always)]
#[case::allocated_publication_before(
    false,
    Failure::Publication(FailurePoint::Before),
    SyncPolicy::Always
)]
#[case::allocated_publication_after(
    false,
    Failure::Publication(FailurePoint::After),
    SyncPolicy::Always
)]
#[case::initial_publication_before(
    true,
    Failure::Publication(FailurePoint::Before),
    SyncPolicy::Always
)]
#[case::initial_publication_after(
    true,
    Failure::Publication(FailurePoint::After),
    SyncPolicy::Always
)]
#[case::allocated_write_before(false, Failure::Write(WriteFailure::Before), SyncPolicy::Always)]
#[case::allocated_write_partial(false, Failure::Write(WriteFailure::Partial), SyncPolicy::Always)]
#[case::allocated_write_after(false, Failure::Write(WriteFailure::After), SyncPolicy::Always)]
#[case::initial_write_before(true, Failure::Write(WriteFailure::Before), SyncPolicy::Always)]
#[case::initial_write_partial(true, Failure::Write(WriteFailure::Partial), SyncPolicy::Always)]
#[case::initial_write_after(true, Failure::Write(WriteFailure::After), SyncPolicy::Always)]
#[case::ordered_write_before(false, Failure::Write(WriteFailure::Before), SyncPolicy::OnClose)]
#[case::ordered_write_partial(true, Failure::Write(WriteFailure::Partial), SyncPolicy::OnClose)]
#[case::ordered_write_after(true, Failure::Write(WriteFailure::After), SyncPolicy::OnClose)]
fn actual_io_failure_prevents_dependent_reference_publication(
    #[case] deferred: bool,
    #[case] failure: Failure,
    #[case] policy: SyncPolicy,
) {
    let directory = tempfile::tempdir().unwrap();
    let control = directory.path().join("successful.h5");
    let observed = record(
        &control,
        Fixture::Persistent {
            paged: false,
            deferred,
            filtered: false,
        },
        MemoryStrategy::Bounded,
        policy,
        Buffering::Session,
        1,
    );
    let gate = observed.gates().into_iter().next().unwrap();
    gate.parse(&observed.trace)
        .unwrap()
        .require(oracle_policy(policy))
        .unwrap();
    let target = match failure {
        Failure::Content(_) => (0..gate.publication)
            .rev()
            .find(|&index| matches!(observed.trace.events[index], IoEvent::Synchronize { .. }))
            .unwrap(),
        Failure::Publication(_) => (gate.publication + 1..gate.reference)
            .find(|&index| matches!(observed.trace.events[index], IoEvent::Synchronize { .. }))
            .unwrap(),
        Failure::Write(_) => gate.publication,
    };
    let path = directory.path().join("failed.h5");
    fs::write(&path, &observed.base).unwrap();
    let mut engine = open(&path, MemoryStrategy::Bounded, policy, Buffering::Session);
    disk_log::start();
    match failure {
        Failure::Content(point) | Failure::Publication(point) => {
            let completed = observed.trace.events[..target]
                .iter()
                .filter(|event| matches!(event, IoEvent::Synchronize { .. }))
                .count();
            disk_log::fail_sync_after(completed, point);
        }
        Failure::Write(prefix) => {
            let completed = observed.trace.events[..target]
                .iter()
                .filter(|event| matches!(event, IoEvent::Write { .. }))
                .count();
            disk_log::fail_write_after(
                completed,
                match prefix {
                    WriteFailure::After => usize::MAX,
                    WriteFailure::Before => 0,
                    WriteFailure::Partial => {
                        let IoEvent::Write { requested, .. } = observed.trace.events[target] else {
                            panic!("the publication target is a write")
                        };
                        assert!(requested > 1);
                        requested / 2
                    }
                },
            );
        }
    }
    let values: Vec<_> = if deferred {
        (0..4).collect()
    } else {
        (4..8).collect()
    };
    let mut append = AppendBuilder::new();
    append.append_i32(&values);
    let error = engine
        .append_inplace_gathered(AppendTarget::Path(&ObjectPathBuf::parse("d")), &append, 4)
        .unwrap_err();
    let Error::Io(error) = error else {
        panic!(
            "{deferred:?}/{policy:?}/{failure:?}, event {target}: expected storage failure, got {error:?}"
        )
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(
        error.to_string(),
        if matches!(failure, Failure::Write(_)) {
            "injected disk write failure"
        } else {
            "injected disk synchronization failure"
        }
    );
    let trace = disk_log::finish();
    disk_log::clear_failures();
    assert_eq!(
        trace.events.get(target).map(IoEvent::outcome),
        Some(IoOutcome::Failed(ErrorKind::Other)),
        "{deferred:?}/{policy:?}/{failure:?}, publication {gate:?}: {:?}",
        trace.events
    );
    assert!(
        !trace.events[target + 1..]
            .iter()
            .any(|event| matches!(event, IoEvent::Write { .. })),
        "{deferred:?}/{policy:?}/{failure:?}, event {target}: issued writes after failure: {:?}",
        trace.events
    );
    let interrupted = fs::read(&path).unwrap();
    let mut replay = observed.base.clone();
    for operation in trace.operations {
        operation.apply(&mut replay);
    }
    assert_eq!(
        replay, interrupted,
        "{failure:?}: failure trace does not reconstruct storage"
    );
    if !matches!(failure, Failure::Write(WriteFailure::Partial)) {
        let file = File::from_bytes(interrupted.clone()).unwrap();
        assert_eq!(
            file.dataset("d").unwrap().read_i32().unwrap(),
            if deferred {
                Vec::new()
            } else {
                vec![0, 1, 2, 3]
            },
            "{failure:?}: a completed I/O boundary changes the committed values"
        );
    }
    disk_log::start();
    if matches!(failure, Failure::Content(_)) {
        engine
            .append_inplace_gathered(AppendTarget::Path(&ObjectPathBuf::parse("d")), &append, 4)
            .unwrap();
        let retry = disk_log::finish();
        let final_bytes = fs::read(&path).unwrap();
        let retry = Observation::new(
            interrupted,
            final_bytes,
            retry,
            if deferred {
                (0..4).collect()
            } else {
                (0..8).collect()
            },
        );
        for gate in retry.gates() {
            gate.parse(&retry.trace)
                .unwrap()
                .require(Policy::Always)
                .unwrap();
        }
        drop(engine);
        retry.assert_complete(&path);
    } else {
        let error = engine
            .append_inplace_gathered(AppendTarget::Path(&ObjectPathBuf::parse("d")), &append, 4)
            .unwrap_err();
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{failure:?}: {error:?}"
        );
        let error = engine.commit().unwrap_err();
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{failure:?}: {error:?}"
        );
        let error = engine.finalize_persist().unwrap_err();
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{failure:?}: {error:?}"
        );
        let error = engine.force_sync().unwrap_err();
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{failure:?}: {error:?}"
        );
        drop(engine);
        assert!(
            disk_log::finish().events.is_empty(),
            "{failure:?}: an uncertain session issues more I/O"
        );
        assert_eq!(
            fs::read(&path).unwrap(),
            interrupted,
            "{failure:?}: Drop changes uncertain storage"
        );
    }
}

#[rstest]
#[case::missing_content(Control::Content, false, SyncPolicy::Always)]
#[case::missing_publication(Control::Publication, false, SyncPolicy::Always)]
#[case::weakened_metadata(Control::Metadata, false, SyncPolicy::Always)]
#[case::failed_sync(Control::Failure, false, SyncPolicy::Always)]
#[case::early_reference(Control::Reference, false, SyncPolicy::Always)]
#[case::coalesced_reference(Control::Coalesced, false, SyncPolicy::Always)]
#[case::initial_content(Control::Content, true, SyncPolicy::Always)]
#[case::initial_publication(Control::Publication, true, SyncPolicy::Always)]
#[case::initial_failed_sync(Control::Failure, true, SyncPolicy::Always)]
#[case::initial_early_reference(Control::Reference, true, SyncPolicy::Always)]
#[case::initial_coalesced_reference(Control::Coalesced, true, SyncPolicy::Always)]
#[case::ordered_early_reference(Control::Reference, false, SyncPolicy::OnClose)]
#[case::ordered_initial_reference(Control::Reference, true, SyncPolicy::OnClose)]
#[case::late_content(Control::ContentAfter, false, SyncPolicy::Always)]
#[case::late_initial_content(Control::ContentAfter, true, SyncPolicy::Always)]
#[case::ordered_late_content(Control::ContentAfter, false, SyncPolicy::OnClose)]
#[case::ordered_initial_content(Control::ContentAfter, true, SyncPolicy::OnClose)]
fn synchronization_controls_reject_readable_completed_files(
    #[case] control: Control,
    #[case] deferred: bool,
    #[case] policy: SyncPolicy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("control.h5");
    let mut observed = record(
        &path,
        if deferred {
            Fixture::Persistent {
                paged: false,
                deferred: true,
                filtered: false,
            }
        } else {
            Fixture::AppendOnly
        },
        MemoryStrategy::Bounded,
        policy,
        Buffering::Operation,
        2,
    );
    let ordinal = observed
        .gates()
        .iter()
        .position(|gate| deferred || gate.eof == EofChange::Extend)
        .unwrap();
    let gate = observed.gates().remove(ordinal);
    let before = observed.trace.operations.clone();
    match control {
        Control::Coalesced => observed.coalesce_reference(&gate),
        Control::ContentAfter => {
            let prepared = observed.facts[ordinal].allocation_content[0];
            let position = observed
                .trace
                .events
                .iter()
                .position(|event| event.operation() == Some(prepared))
                .unwrap();
            assert!(position < gate.publication);
            let event = observed.trace.events.remove(position);
            observed.trace.events.insert(gate.publication, event);
        }
        Control::Reference => observed.trace.events.swap(gate.publication, gate.reference),
        Control::Content | Control::Failure | Control::Metadata | Control::Publication => {
            let range = if control == Control::Content {
                gate.content.last().copied().unwrap() + 1..gate.publication
            } else {
                gate.publication + 1..gate.reference
            };
            let mut affected = 0;
            for index in range.rev() {
                if let IoEvent::Synchronize { kind, outcome } = &mut observed.trace.events[index] {
                    affected += 1;
                    match control {
                        Control::Content | Control::Publication => {
                            observed.trace.events.remove(index);
                        }
                        Control::Failure => *outcome = IoOutcome::Failed(ErrorKind::Other),
                        Control::Metadata => *kind = Synchronization::Data,
                        Control::Coalesced | Control::ContentAfter | Control::Reference => {
                            panic!("this control does not change a synchronization event")
                        }
                    }
                }
            }
            assert!(
                affected > 0,
                "{deferred:?}/{policy:?}/{control:?}: the control changes no synchronization"
            );
        }
    }
    let gate = observed.gates().remove(ordinal);
    let result = gate
        .parse(&observed.trace)
        .and_then(|publication| publication.require(oracle_policy(policy)));
    match (control, result.unwrap_err()) {
        (
            Control::Content,
            OrderError::ContentBarrier {
                content,
                publication,
            },
        ) => {
            assert_eq!(content, *gate.content.last().unwrap());
            assert_eq!(publication, gate.publication);
        }
        (
            Control::Metadata | Control::Publication,
            OrderError::PublicationBarrier {
                publication,
                reference,
                required,
            },
        ) => {
            assert_eq!(
                (publication, reference, required),
                (
                    gate.publication,
                    gate.reference,
                    match gate.eof {
                        EofChange::Extend => Synchronization::All,
                        EofChange::Unchanged => Synchronization::Data,
                    }
                )
            );
        }
        (Control::Failure, OrderError::FailedIo { event, outcome }) => {
            assert!(event > gate.publication && event < gate.reference);
            assert_eq!(outcome, IoOutcome::Failed(ErrorKind::Other));
        }
        (
            Control::Coalesced | Control::Reference,
            OrderError::ReferenceOrder {
                publication,
                reference,
                ..
            },
        ) => {
            assert_eq!((publication, reference), (gate.publication, gate.reference));
            assert!(reference <= publication);
        }
        (
            Control::ContentAfter,
            OrderError::ReferenceOrder {
                content,
                publication,
                ..
            },
        ) => {
            assert_eq!(
                (content, publication),
                (*gate.content.last().unwrap(), gate.publication)
            );
            assert!(content >= publication);
        }
        (_, error) => panic!(
            "{deferred:?}/{policy:?}/{control:?}, publication {gate:?}: unexpected {error:?}"
        ),
    }
    if control != Control::Coalesced {
        assert_eq!(observed.trace.operations, before);
    }
    observed.assert_complete(&path);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Buffering {
    Operation,
    Session,
    Unbuffered,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Control {
    Coalesced,
    Content,
    ContentAfter,
    Failure,
    Metadata,
    Publication,
    Reference,
}

#[derive(Clone, Copy, Debug)]
enum Failure {
    Content(FailurePoint),
    Publication(FailurePoint),
    Write(WriteFailure),
}

#[derive(Clone, Copy, Debug)]
enum WriteFailure {
    After,
    Before,
    Partial,
}

struct Observation {
    base: Vec<u8>,
    final_bytes: Vec<u8>,
    trace: Recording,
    expected: Vec<i32>,
    facts: Vec<PublicationFacts>,
}

impl Observation {
    fn new(base: Vec<u8>, final_bytes: Vec<u8>, trace: Recording, expected: Vec<i32>) -> Self {
        let facts = PublicationFacts::read(&base, &final_bytes, &trace);
        Self {
            base,
            final_bytes,
            trace,
            expected,
            facts,
        }
    }

    fn gates(&self) -> Vec<Gate> {
        let locate = |operation| {
            self.trace
                .events
                .iter()
                .position(|event| event.operation() == Some(operation))
                .unwrap()
        };
        let mut previous = 0;
        self.facts
            .iter()
            .map(|fact| {
                let publication = locate(fact.operation);
                let reference = fact
                    .references
                    .iter()
                    .map(|&operation| locate(operation))
                    .min()
                    .unwrap();
                let mut content: Vec<_> = fact
                    .prepared
                    .iter()
                    .map(|&operation| locate(operation))
                    .collect();
                content.extend(
                    self.trace.events[previous..publication]
                        .iter()
                        .enumerate()
                        .filter_map(|(index, event)| {
                            matches!(event, IoEvent::Write { .. }).then_some(previous + index)
                        }),
                );
                content.sort_unstable();
                content.dedup();
                previous = publication + 1;
                Gate {
                    content,
                    publication,
                    reference,
                    eof: fact.eof,
                }
            })
            .collect()
    }

    fn assert_complete(&self, path: &Path) {
        assert!(!self.trace.events.is_empty());
        assert!(!self.trace.operations.is_empty());
        let mut replay = self.base.clone();
        let mut used = std::collections::BTreeSet::new();
        for event in &self.trace.events {
            if let Some(operation) = event.operation() {
                assert!(
                    used.insert(operation),
                    "operation {operation} is issued twice"
                );
                self.trace.operations[operation].apply(&mut replay);
            }
        }
        assert_eq!(
            used.len(),
            self.trace.operations.len(),
            "the event trace omits a byte operation"
        );
        assert_eq!(replay, self.final_bytes, "full I/O reconstruction differs");
        let before = fs::read(path).unwrap();
        let file = File::open(path).unwrap();
        assert_eq!(
            file.dataset("d").unwrap().read_i32().unwrap(),
            self.expected
        );
        file.persisted_free_sections().unwrap();
        drop(file);
        assert_eq!(
            fs::read(path).unwrap(),
            before,
            "inspection changes fixture bytes"
        );
    }

    fn coalesce_reference(&mut self, gate: &Gate) {
        let publication = self.trace.events[gate.publication].operation().unwrap();
        let reference = self.trace.events[gate.reference].operation().unwrap();
        assert!(publication < reference);
        let mut image = self.base.clone();
        for operation in &self.trace.operations[..publication] {
            operation.apply(&mut image);
        }
        self.trace.operations[publication].apply(&mut image);
        self.trace.operations[reference].apply(&mut image);
        let DiskOp::Write { offset, bytes } = &self.trace.operations[reference] else {
            panic!("the reference is a write")
        };
        let end = usize::try_from(*offset)
            .unwrap()
            .checked_add(bytes.len())
            .unwrap();
        let bytes = image[..end].to_vec();
        let length = bytes.len();
        self.trace.operations[publication] = DiskOp::Write { offset: 0, bytes };
        self.trace.operations.remove(reference);
        for fact in &mut self.facts {
            fact.fold_operation(reference, publication);
        }
        self.trace.events[gate.publication] = IoEvent::Write {
            offset: 0,
            requested: length,
            written: length,
            outcome: IoOutcome::Succeeded,
            operation: Some(publication),
        };
        self.trace.events.remove(gate.reference);
        for event in &mut self.trace.events {
            if let IoEvent::Write {
                operation: Some(operation),
                ..
            }
            | IoEvent::Resize {
                operation: Some(operation),
                ..
            } = event
                && *operation > reference
            {
                *operation -= 1;
            }
        }
    }
}

impl fmt::Debug for Observation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Observation")
            .field("base_length", &self.base.len())
            .field("final_length", &self.final_bytes.len())
            .field("events", &self.trace.events)
            .field("facts", &self.facts)
            .finish()
    }
}

#[derive(Clone, Copy, Debug)]
enum Fixture {
    AppendOnly,
    Persistent {
        paged: bool,
        deferred: bool,
        filtered: bool,
    },
}

#[derive(Debug)]
struct Gate {
    content: Vec<usize>,
    publication: usize,
    reference: usize,
    eof: EofChange,
}

impl Gate {
    fn parse<'a>(&self, recording: &'a Recording) -> Result<Publication<'a>, OrderError> {
        Publication::parse(
            recording,
            &self.content,
            self.publication,
            self.reference,
            self.eof,
        )
    }
}

fn record(
    path: &Path,
    fixture: Fixture,
    backing: MemoryStrategy,
    policy: SyncPolicy,
    buffering: Buffering,
    rounds: i32,
) -> Observation {
    let deferred = match fixture {
        Fixture::AppendOnly => {
            let mut builder = FileBuilder::new();
            builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::None));
            builder
                .create_dataset("d")
                .with_i32_data(&[0, 1, 2, 3])
                .with_chunks(&[4])
                .with_maxshape(&[MaxExtent::Unlimited]);
            builder.write(path).unwrap();
            false
        }
        Fixture::Persistent {
            paged,
            deferred,
            filtered,
        } => {
            super::deferred_base(path, filtered, paged);
            if !deferred {
                let mut engine = open(path, backing, policy, buffering);
                super::append(&mut engine, 0, 4);
                drop(engine);
            }
            deferred
        }
    };
    let base = fs::read(path).unwrap();
    let mut engine = open(path, backing, policy, buffering);
    disk_log::start();
    let start = if deferred { 0 } else { 4 };
    for round in 0..rounds {
        super::append(&mut engine, start + round * 4, 4);
    }
    let trace = disk_log::finish();
    let final_bytes = fs::read(path).unwrap();
    drop(engine);
    assert_eq!(
        fs::read(path).unwrap(),
        final_bytes,
        "Drop modifies completed append bytes"
    );
    let observed = Observation::new(base, final_bytes, trace, (0..start + rounds * 4).collect());
    observed.assert_complete(path);
    observed
}

fn open(
    path: &Path,
    backing: MemoryStrategy,
    policy: SyncPolicy,
    buffering: Buffering,
) -> WriteEngine {
    let mut engine = WriteEngine::open_rw_with_strategy(
        path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        backing,
        PersistedFreeSpacePolicy::Strict,
    )
    .unwrap();
    engine.set_sync_policy(policy);
    match buffering {
        Buffering::Operation => {}
        Buffering::Session => engine
            .set_write_buffering(WriteBuffering::Session {
                page_size: 4096,
                max_bytes: 65_536,
            })
            .unwrap(),
        Buffering::Unbuffered => engine
            .set_write_buffering(WriteBuffering::Unbuffered)
            .unwrap(),
    }
    engine
}

fn oracle_policy(policy: SyncPolicy) -> Policy {
    match policy {
        SyncPolicy::Always => Policy::Always,
        SyncPolicy::OnClose => Policy::OnClose,
    }
}
