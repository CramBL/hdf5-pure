use std::error::Error as StdError;
use std::fs;
use std::iter;

use hdf5_pure_format::__private::ObjectHeaderPrefix;
use rstest::rstest;
use test_util::__private::durability::FaultSchedule;
use test_util::datatype;
use test_util::disk_log::{self, DiskOp, FailurePoint, IoEvent, Recording, Synchronization};
use test_util::file_space_info;
use test_util::global_heap;
use test_util::object_header::{Message, MessageType, v1, v2};
use test_util::superblock::v2::Superblock;
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::members::ParentImage;

use crate::edit::test_util::durability;
use crate::{
    AttrValue, Error, File, FileAccessProperties, FileLocking, MemoryStrategy, SyncPolicy,
};

#[rstest]
#[case::count_before(Fault::Count(0))]
#[case::count_partial(Fault::Count(2))]
#[case::count_complete(Fault::Count(4))]
#[case::barrier_before(Fault::Barrier(FailurePoint::Before))]
#[case::barrier_after(Fault::Barrier(FailurePoint::After))]
#[case::restore_write(Fault::RestoreWrite)]
#[case::restore_barrier_before(Fault::RestoreBarrier(FailurePoint::Before))]
#[case::restore_barrier_after(Fault::RestoreBarrier(FailurePoint::After))]
fn postpublication_count_failures_preserve_the_tree_and_quarantine_allocations(
    #[case] fault: Fault,
    #[values(Version::One, Version::Two)] version: Version,
    #[values(1, 2)] targets: usize,
) {
    run(version, targets, fault, false);
}

#[rstest]
#[case::before(0)]
#[case::partial(2)]
#[case::complete(4)]
fn postpublication_checksum_failures_restore_complete_count_headers(
    #[case] prefix: usize,
    #[values(1, 2)] targets: usize,
) {
    run(Version::Two, targets, Fault::Checksum(prefix), false);
}

#[rstest]
#[case::restored_count(Fault::Count(2))]
#[case::restored_barrier(Fault::Barrier(FailurePoint::Before))]
#[case::failed_restoration_write(Fault::RestoreWrite)]
#[case::failed_restoration_barrier(Fault::RestoreBarrier(FailurePoint::Before))]
fn postpublication_page_count_failures_distinguish_recovery_from_quarantine(#[case] fault: Fault) {
    run(Version::Two, 2, fault, true);
}

fn run(version: Version, targets: usize, fault: Fault, page: bool) {
    let path = test_util::temp::temp_path("published-count-journal.h5");
    let addresses = super::fixture_counts(&path, true, version, targets, 3);
    if page {
        let mut bytes = fs::read(&path).unwrap();
        let extension = bytes.len() as u64;
        bytes.extend(
            v2::Header::new()
                .message(Message::new(
                    MessageType::FILE_SPACE_INFO,
                    &file_space_info::non_persistent_version_1(
                        Widths::EIGHT,
                        file_space_info::PAGE,
                        1,
                        4096,
                    ),
                ))
                .build(),
        );
        bytes.resize(bytes.len().next_multiple_of(4096), 0);
        let superblock = Superblock::new(Widths::EIGHT)
            .version(3)
            .root_header_address(test_util_hdf5::global_heap_fixture::Fixture::ROOT)
            .extension_address(extension)
            .eof_address(bytes.len() as u64)
            .build();
        bytes[..superblock.len()].copy_from_slice(&superblock);
        fs::write(&path, bytes).unwrap();
    }
    let original = fs::read(&path).unwrap();
    let parent = global_heap::collection_ranges(&original, Widths::EIGHT)[0].start;
    let control_path = test_util::temp::temp_path("published-count-control.h5");
    fs::write(&control_path, &original).unwrap();
    let control = open(&control_path);
    stage(&control, targets);
    disk_log::start();
    control.commit().unwrap();
    let recording = disk_log::finish();
    let published = fs::read(&control_path).unwrap();
    let publication = recording
        .events
        .iter()
        .position(|event| matches!(event, IoEvent::Write { offset: 0, .. }))
        .unwrap();
    let writes: Vec<_> = addresses
        .iter()
        .map(|address| {
            let count = version.field(*address);
            let event = write_at(&recording, publication, count);
            assert_eq!(
                &written(&recording, event.event)[event.before..event.before + 4],
                2_u32.to_le_bytes()
            );
            let checksum = matches!(version, Version::Two).then(|| {
                write_at(
                    &recording,
                    publication,
                    *address + version.header(2).len() as u64 - 4,
                )
            });
            CountWrites {
                count: event,
                checksum,
            }
        })
        .collect();
    if targets == 2 {
        assert!(writes[0].count.event < writes[1].count.event);
    }
    for &address in &addresses {
        assert_eq!(header(&published, address, version), version.header(2));
    }
    assert_values(&control, targets);
    control.close().unwrap();
    let CountWrites { count, checksum } = *writes.last().unwrap();
    let last = checksum.unwrap_or(count).event;
    let barrier = recording.events[last + 1..]
        .iter()
        .position(|event| matches!(event, IoEvent::Synchronize { .. }))
        .map(|index| index + last + 1)
        .unwrap();
    let schedule = fault.schedule(&recording, count, checksum, barrier);
    let file = open(&path);
    stage(&file, targets);
    disk_log::start();
    schedule.inject();
    let error = file.commit().unwrap_err();
    disk_log::clear_failures();
    let failed = disk_log::finish();
    schedule.assert_failed(&failed);
    fault.assert_error(error);
    assert!(!file.has_staged_edits());
    let after = fs::read(&path).unwrap();
    assert_eq!(&after[..48], &published[..48]);
    ParentImage::read(&original, parent).assert_unchanged(&after);
    for &address in &addresses {
        let expected = version.header(if matches!(fault, Fault::RestoreWrite) {
            2
        } else {
            3
        });
        assert_eq!(header(&after, address, version), expected);
    }
    assert_count_restoration(&failed, &writes, fault, version);
    let free = file.reusable_free_sections().unwrap();
    let stored = File::from_bytes(after.clone()).unwrap();
    let root = stored.superblock().root_group_address.get();
    assert_ne!(root, test_util_hdf5::global_heap_fixture::Fixture::ROOT);
    stored.close().unwrap();
    let prefix = ObjectHeaderPrefix::parse(&after[root as usize..]).unwrap();
    let root_end = root + prefix.len as u64 + prefix.chunk0_size + 4;
    assert_eq!(
        &after[root as usize..root_end as usize],
        &published[root as usize..root_end as usize]
    );
    for extent in addresses
        .iter()
        .map(|&address| address..address + version.header(3).len() as u64)
        .chain(iter::once(
            parent..parent + global_heap::MIN_COLLECTION_SIZE as u64,
        ))
        .chain(iter::once(root..root_end))
    {
        assert!(free.sections().iter().all(|section| {
            section.end_address().get() <= extent.start || section.address().get() >= extent.end
        }));
    }
    assert_values(&file, targets);
    file.root().set_attr("after", AttrValue::I64(41)).unwrap();
    assert!(file.has_staged_edits());
    let partial = matches!(fault, Fault::RestoreWrite | Fault::RestoreBarrier(_));
    if page && !partial {
        file.commit().unwrap();
        assert!(!file.has_staged_edits());
        assert_eq!(file.root().attrs().unwrap()["after"], AttrValue::I64(41));
        assert_values(&file, targets);
        file.close().unwrap();
    } else {
        disk_log::start();
        durability::assert_uncertain(file.commit().unwrap_err());
        durability::assert_uncertain(file.sync().unwrap_err());
        assert!(disk_log::finish().operations.is_empty());
        assert!(file.has_staged_edits());
        assert_eq!(fs::read(&path).unwrap(), after);
        drop(file);
        File::clear_status_flags(&path).unwrap();
    }
    let reopened = File::open(&path).unwrap();
    assert_values(&reopened, targets);
    reopened.close().unwrap();
}

fn assert_count_restoration(
    failed: &Recording,
    writes: &[CountWrites],
    fault: Fault,
    version: Version,
) {
    let first = writes[0].count.event;
    let counts: Vec<_> = failed.events[first..]
        .iter()
        .filter_map(|event| match event {
            IoEvent::Write {
                offset, requested, ..
            } => Some((*offset, *requested)),
            _ => None,
        })
        .collect();
    let prefix: Vec<_> = failed.events[first..]
        .iter()
        .take_while(|event| event.outcome() == disk_log::IoOutcome::Succeeded)
        .filter_map(|event| match event {
            IoEvent::Write {
                offset, requested, ..
            } => Some((*offset, *requested)),
            _ => None,
        })
        .collect();
    let attempted = if matches!(
        fault,
        Fault::Count(_) | Fault::Checksum(_) | Fault::RestoreBarrier(_)
    ) {
        prefix.len() + 1
    } else {
        prefix.len()
    };
    let (updates, restores) = counts.split_at(attempted);
    assert_eq!(
        restores,
        if matches!(fault, Fault::RestoreWrite) {
            &updates[..1]
        } else {
            updates
        }
    );
    let syncs: Vec<_> = failed.events[first..]
        .iter()
        .filter_map(|event| match event {
            IoEvent::Synchronize { kind, outcome } => Some((*kind, *outcome)),
            _ => None,
        })
        .collect();
    let success = disk_log::IoOutcome::Succeeded;
    let failure = disk_log::IoOutcome::Failed(std::io::ErrorKind::Other);
    assert_eq!(
        syncs,
        match fault {
            Fault::Count(_) | Fault::Checksum(_) => vec![(Synchronization::Data, success)],
            Fault::Barrier(_) => vec![
                (Synchronization::All, failure),
                (Synchronization::Data, success)
            ],
            Fault::RestoreWrite => vec![(Synchronization::All, failure)],
            Fault::RestoreBarrier(_) => vec![(Synchronization::Data, failure)],
        }
    );
    assert_eq!(
        writes[0].checksum.is_some(),
        matches!(version, Version::Two)
    );
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Version {
    One,
    Two,
}

impl Version {
    pub(super) fn header(self, count: u32) -> Vec<u8> {
        match self {
            Self::One => {
                let mut datatype = datatype::f64_le();
                datatype.resize(datatype.len().next_multiple_of(v1::MESSAGE_ALIGNMENT), 0);
                v1::Header::new()
                    .reference_count(count)
                    .message(Message::new(MessageType::DATATYPE, &datatype))
                    .build()
            }
            Self::Two => super::type_header(count),
        }
    }

    fn field(self, address: u64) -> u64 {
        address
            + match self {
                Self::One => 4,
                Self::Two => 12,
            }
    }
}

#[derive(Clone, Copy, Debug)]
enum Fault {
    Barrier(FailurePoint),
    Checksum(usize),
    Count(usize),
    RestoreBarrier(FailurePoint),
    RestoreWrite,
}

impl Fault {
    fn schedule<'a>(
        self,
        control: &'a Recording,
        count: FieldWrite,
        checksum: Option<FieldWrite>,
        barrier: usize,
    ) -> FaultSchedule<'a> {
        match self {
            Self::Count(prefix) => {
                FaultSchedule::write(control, count.event, count.before + prefix)
            }
            Self::Checksum(prefix) => {
                let FieldWrite { event, before } = checksum.unwrap();
                FaultSchedule::write(control, event, before + prefix)
            }
            Self::Barrier(point) => FaultSchedule::synchronize(control, barrier, point),
            Self::RestoreWrite => {
                FaultSchedule::synchronize(control, barrier, FailurePoint::Before)
                    .with_recovery_write(0)
            }
            Self::RestoreBarrier(point) => {
                FaultSchedule::write(control, count.event, count.before + 2)
                    .with_recovery_synchronization(point)
            }
        }
    }

    fn assert_error(self, error: Error) {
        match self {
            Self::RestoreWrite | Self::RestoreBarrier(_) => {
                let cause = StdError::source(&error)
                    .and_then(StdError::source)
                    .unwrap()
                    .to_string();
                let Error::CommitPartiallyApplied { refusal, restore } = error else {
                    panic!("{error:?}")
                };
                let (original, recovery) = if matches!(self, Self::RestoreWrite) {
                    (
                        "injected disk synchronization failure",
                        "injected disk write failure",
                    )
                } else {
                    (
                        "injected disk write failure",
                        "injected disk synchronization failure",
                    )
                };
                assert_eq!(cause, recovery);
                durability::assert_io(*refusal, original);
                durability::assert_io(*restore, recovery);
            }
            _ => durability::assert_io(
                error,
                if matches!(self, Self::Barrier(_)) {
                    "injected disk synchronization failure"
                } else {
                    "injected disk write failure"
                },
            ),
        }
    }
}

fn open(path: &std::path::Path) -> File {
    File::open_rw_with_options(
        path,
        FileAccessProperties::new()
            .with_locking(FileLocking::Disabled)
            .with_memory_strategy(MemoryStrategy::Mirrored)
            .with_page_buffer_size(0)
            .with_sync_policy(SyncPolicy::Always),
    )
    .unwrap()
}

fn stage(file: &File, targets: usize) {
    super::stage(file, 1, false, true);
    for index in 1..targets {
        file.copy(&format!("source{index}"), &format!("copy{index}"))
            .unwrap();
        file.root().delete(&format!("type{index}")).unwrap();
        file.root().delete(&format!("alias{index}")).unwrap();
    }
}

#[derive(Clone, Copy)]
struct CountWrites {
    count: FieldWrite,
    checksum: Option<FieldWrite>,
}

#[derive(Clone, Copy)]
struct FieldWrite {
    event: usize,
    before: usize,
}

fn write_at(recording: &Recording, after: usize, address: u64) -> FieldWrite {
    recording.events[after + 1..]
        .iter()
        .enumerate()
        .find_map(|(index, event)| {
            if let IoEvent::Write {
                offset, requested, ..
            } = event
                && *offset <= address
                && address + 4 <= *offset + *requested as u64
            {
                Some(FieldWrite {
                    event: index + after + 1,
                    before: usize::try_from(address - offset).unwrap(),
                })
            } else {
                None
            }
        })
        .unwrap_or_else(|| panic!("missing count write at {address}: {:?}", recording.events))
}

fn written(recording: &Recording, index: usize) -> &[u8] {
    let Some(DiskOp::Write { bytes, .. }) = recording.events[index]
        .operation()
        .and_then(|operation| recording.operations.get(operation))
    else {
        panic!("missing completed write");
    };
    bytes
}

fn header(image: &[u8], address: u64, version: Version) -> &[u8] {
    let start = usize::try_from(address).unwrap();
    &image[start..start + version.header(3).len()]
}

fn assert_values(file: &File, targets: usize) {
    assert_eq!(file.dataset("value").unwrap().read_f64().unwrap(), [31.0]);
    assert_eq!(
        file.dataset("keep").unwrap().read_string().unwrap(),
        ["retained"]
    );
    assert_eq!(
        file.dataset("heap").unwrap().attrs().unwrap()["a"],
        AttrValue::I64(9)
    );
    for index in 0..targets {
        let source = if index == 0 {
            "source".to_owned()
        } else {
            format!("source{index}")
        };
        assert_eq!(file.dataset(&source).unwrap().read_f64().unwrap(), [17.0]);
        assert_eq!(
            file.dataset(&format!("copy{index}"))
                .unwrap()
                .read_f64()
                .unwrap(),
            [17.0]
        );
    }
    let mut names = file.root().datasets().unwrap();
    names.sort();
    let mut expected = vec![
        "copy0".to_owned(),
        "heap".into(),
        "keep".into(),
        "source".into(),
        "value".into(),
    ];
    for index in 1..targets {
        expected.extend([format!("copy{index}"), format!("source{index}")]);
    }
    expected.sort();
    assert_eq!(names, expected);
    assert!(file.root().named_datatypes().unwrap().is_empty());
}
