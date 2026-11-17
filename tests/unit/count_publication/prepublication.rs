use std::error::Error as StdError;
use std::io::ErrorKind;

use rstest::rstest;
use test_util::disk_log::{self, FailurePoint, IoEvent, IoOutcome, Recording};
use test_util::superblock;

use super::{CountFile, assert_values, stage, type_header};
use crate::{Error, File, FileAccessProperties, FileLocking, MemoryStrategy};

#[rstest]
#[case::count_zero(Failure::Count(Part::Count, Prefix::Zero))]
#[case::count_partial(Failure::Count(Part::Count, Prefix::Half))]
#[case::count_complete(Failure::Count(Part::Count, Prefix::Complete))]
#[case::checksum_zero(Failure::Count(Part::Checksum, Prefix::Zero))]
#[case::checksum_partial(Failure::Count(Part::Checksum, Prefix::Half))]
#[case::checksum_complete(Failure::Count(Part::Checksum, Prefix::Complete))]
#[case::increase_barrier_before(Failure::Barrier(Direction::Increase, FailurePoint::Before))]
#[case::increase_barrier_after(Failure::Barrier(Direction::Increase, FailurePoint::After))]
#[case::decrease_barrier_before(Failure::Barrier(Direction::Decrease, FailurePoint::Before))]
#[case::decrease_barrier_after(Failure::Barrier(Direction::Decrease, FailurePoint::After))]
#[case::increase_root(Failure::Publication(Direction::Increase))]
#[case::decrease_root(Failure::Publication(Direction::Decrease))]
fn count_failures_preserve_the_publication_and_rollback_boundary(
    #[case] failure: Failure,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let direction = failure.direction();
    let original = if direction.extra_alias() { 3 } else { 2 };
    let copies = if direction.extra_alias() { 1 } else { 2 };
    let path = test_util::temp::temp_path("public-count-failure.h5");
    let fixture = CountFile::write(&path, original, direction.extra_alias());
    let input = std::fs::read(&path).unwrap();
    let control = test_util::temp::temp_path("public-count-control.h5");
    std::fs::write(&control, &input).unwrap();
    let properties = FileAccessProperties::new()
        .with_locking(FileLocking::Disabled)
        .with_memory_strategy(memory)
        .with_page_buffer_size(0);
    let reference = File::open_rw_with_options(&control, properties).unwrap();
    let before_control = std::fs::read(&control).unwrap();
    stage(&reference, copies, direction.extra_alias(), false);
    disk_log::start();
    reference.commit().unwrap();
    let recording = disk_log::finish();
    fixture.assert_publication(
        &before_control,
        &recording.operations,
        &std::fs::read(&control).unwrap(),
        if direction.extra_alias() { 2 } else { 3 },
    );
    let schedule = Schedule::new(&recording, &fixture, failure);
    reference.close().unwrap();

    let file = File::open_rw_with_options(&path, properties).unwrap();
    let before = std::fs::read(&path).unwrap();
    let free = file.reusable_free_sections().unwrap();
    stage(&file, copies, direction.extra_alias(), false);
    disk_log::start();
    schedule.inject(&recording);
    let error = file.commit().unwrap_err();
    disk_log::clear_failures();
    let failed = disk_log::finish();
    schedule.assert_failed(&recording, &failed);
    let mut issued = before.clone();
    for event in &failed.events[..=schedule.event()] {
        if let Some(operation) = event.operation() {
            failed.operations[operation].apply(&mut issued);
        }
    }
    let mut expected = type_header(original);
    match failure {
        Failure::Count(part, prefix) => {
            let updated = type_header(3);
            let field = match part {
                Part::Checksum => {
                    expected[12..16].copy_from_slice(&updated[12..16]);
                    expected.len() - 4
                }
                Part::Count => 12,
            };
            expected[field..field + prefix.bytes()]
                .copy_from_slice(&updated[field..field + prefix.bytes()]);
        }
        Failure::Barrier(_, _) | Failure::Publication(_) => expected = type_header(3),
    }
    let start = usize::try_from(fixture.target).unwrap();
    assert_eq!(&issued[start..start + expected.len()], expected);
    assert_eq!(error.source().unwrap().to_string(), schedule.message());
    let Error::Io(error) = error else {
        panic!("expected the scheduled I/O failure: {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), schedule.message());
    assert!(!file.has_staged_edits());
    let sections = file.reusable_free_sections().unwrap();
    if matches!(failure, Failure::Publication(_)) {
        assert!(
            sections
                .sections()
                .iter()
                .all(|section| { section.address().get() >= before.len() as u64 })
        );
    } else {
        assert_eq!(sections, free);
    }
    let after = std::fs::read(&path).unwrap();
    let mut replay = before.clone();
    for operation in &failed.operations {
        operation.apply(&mut replay);
    }
    assert_eq!(replay, after);
    assert_eq!(
        superblock::root_header_address(&after, 0),
        superblock::root_header_address(&before, 0)
    );
    fixture.assert_header(
        &after,
        if matches!(failure, Failure::Publication(_)) {
            3
        } else {
            original
        },
    );
    if matches!(failure, Failure::Publication(_)) {
        disk_log::start();
        file.root()
            .create_dataset("unavailable", |builder| {
                builder.with_i32_data(&[1]);
            })
            .unwrap();
        assert!(file.has_staged_edits());
        let error = file.commit().unwrap_err();
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{error:?}"
        );
        assert!(file.has_staged_edits());
        assert!(disk_log::finish().events.is_empty());
        assert_eq!(std::fs::read(&path).unwrap(), after);
        drop(file);
        File::clear_status_flags(&path).unwrap();
        let file = File::open(&path).unwrap();
        let mut names = file.root().datasets().unwrap();
        names.sort();
        assert_eq!(names, ["dead", "source", "value"]);
        let mut names = file.root().named_datatypes().unwrap();
        names.sort();
        assert_eq!(
            names,
            if direction.extra_alias() {
                vec!["alias", "type"]
            } else {
                vec!["type"]
            }
        );
        assert_eq!(file.root().named_datatype_references("type").unwrap(), 3);
        assert_eq!(
            file.dataset("dead").unwrap().read_i32().unwrap(),
            [7; 1_024]
        );
        assert_eq!(file.dataset("source").unwrap().read_f64().unwrap(), [17.0]);
        assert_eq!(file.dataset("value").unwrap().read_f64().unwrap(), [31.0]);
        file.close().unwrap();
    } else {
        assert_eq!(&after[..before.len()], before);
        file.commit().unwrap();
        stage(&file, copies, direction.extra_alias(), false);
        file.commit().unwrap();
        assert_values(&file, copies, false);
        file.close().unwrap();
        fixture.assert_header(
            &std::fs::read(&path).unwrap(),
            if direction.extra_alias() { 2 } else { 3 },
        );
        let file = File::open(&path).unwrap();
        assert_values(&file, copies, false);
        file.close().unwrap();
    }
}

#[derive(Clone, Copy)]
enum Failure {
    Barrier(Direction, FailurePoint),
    Count(Part, Prefix),
    Publication(Direction),
}

impl Failure {
    fn direction(self) -> Direction {
        match self {
            Self::Barrier(direction, _) | Self::Publication(direction) => direction,
            Self::Count(_, _) => Direction::Increase,
        }
    }
}

#[derive(Clone, Copy)]
enum Direction {
    Decrease,
    Increase,
}

impl Direction {
    fn extra_alias(self) -> bool {
        matches!(self, Self::Decrease)
    }
}

#[derive(Clone, Copy)]
pub(super) enum Part {
    Checksum,
    Count,
}

#[derive(Clone, Copy)]
pub(super) enum Prefix {
    Complete,
    Half,
    Zero,
}

impl Prefix {
    pub(super) fn bytes(self) -> usize {
        match self {
            Self::Complete => 4,
            Self::Half => 2,
            Self::Zero => 0,
        }
    }
}

#[derive(Clone, Copy)]
enum Schedule {
    Synchronize { event: usize, point: FailurePoint },
    Write { event: usize, prefix: usize },
}

impl Schedule {
    fn new(recording: &Recording, fixture: &CountFile, failure: Failure) -> Self {
        let publication = recording
            .events
            .iter()
            .position(|event| matches!(event, IoEvent::Write { offset: 0, .. }))
            .unwrap();
        match failure {
            Failure::Barrier(_, point) => {
                let event = recording.events[..publication]
                    .iter()
                    .rposition(|event| matches!(event, IoEvent::Synchronize { .. }))
                    .unwrap();
                Self::Synchronize { event, point }
            }
            Failure::Count(part, prefix) => {
                let field = fixture.target
                    + match part {
                        Part::Checksum => type_header(fixture.original).len() as u64 - 4,
                        Part::Count => 12,
                    };
                let (event, relative) = recording.events[..publication]
                    .iter()
                    .enumerate()
                    .find_map(|(index, event)| {
                        let IoEvent::Write {
                            offset, requested, ..
                        } = event
                        else {
                            return None;
                        };
                        let relative = usize::try_from(field.checked_sub(*offset)?).ok()?;
                        (relative + 4 <= *requested).then_some((index, relative))
                    })
                    .unwrap();
                let prefix = relative + prefix.bytes();
                Self::Write { event, prefix }
            }
            Failure::Publication(_) => Self::Write {
                event: publication,
                prefix: 0,
            },
        }
    }

    fn inject(self, recording: &Recording) {
        let event = self.event();
        match self {
            Self::Synchronize { point, .. } => disk_log::fail_sync_after(
                recording.events[..event]
                    .iter()
                    .filter(|event| matches!(event, IoEvent::Synchronize { .. }))
                    .count(),
                point,
            ),
            Self::Write { prefix, .. } => disk_log::fail_write_after(
                recording.events[..event]
                    .iter()
                    .filter(|event| matches!(event, IoEvent::Write { .. }))
                    .count(),
                prefix,
            ),
        }
    }

    fn assert_failed(self, control: &Recording, failed: &Recording) {
        let event = self.event();
        assert_eq!(
            failed
                .events
                .iter()
                .filter(|event| matches!(event.outcome(), IoOutcome::Failed(_)))
                .count(),
            1
        );
        assert_eq!(
            failed.events[event].outcome(),
            IoOutcome::Failed(ErrorKind::Other)
        );
        match (self, &control.events[event], &failed.events[event]) {
            (
                Self::Write { prefix, .. },
                IoEvent::Write {
                    offset, requested, ..
                },
                IoEvent::Write {
                    offset: at,
                    requested: length,
                    written,
                    ..
                },
            ) => {
                assert_eq!(
                    (at, length, *written),
                    (offset, requested, prefix.min(*requested))
                );
            }
            (
                Self::Synchronize { .. },
                IoEvent::Synchronize { kind, .. },
                IoEvent::Synchronize { kind: actual, .. },
            ) => assert_eq!(actual, kind),
            _ => panic!("failure did not reach its recorded I/O operation"),
        }
    }

    fn event(self) -> usize {
        match self {
            Self::Synchronize { event, .. } | Self::Write { event, .. } => event,
        }
    }

    fn message(self) -> &'static str {
        match self {
            Self::Synchronize { .. } => "injected disk synchronization failure",
            Self::Write { .. } => "injected disk write failure",
        }
    }
}
