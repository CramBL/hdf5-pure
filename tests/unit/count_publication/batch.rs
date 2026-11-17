use std::error::Error as StdError;

use rstest::rstest;
use test_util::datatype;
use test_util::disk_log::{self, DiskOp, FailurePoint, IoEvent, IoOutcome};
use test_util::object_header::{Message, MessageType, v1};

use super::postpublication::{FieldWrite, assert_quarantined, root_range};
use super::prepublication::{Prefix, Schedule};
use super::{stage, type_header, write_count_users};
use crate::{Error, File, FileAccessProperties, FileLocking, MemoryStrategy};

#[rstest]
#[case::v1_single_zero(Case::Count(Version::One, Targets::One, Prefix::Zero))]
#[case::v1_single_partial(Case::Count(Version::One, Targets::One, Prefix::Half))]
#[case::v1_single_complete(Case::Count(Version::One, Targets::One, Prefix::Complete))]
#[case::v1_batch_zero(Case::Count(Version::One, Targets::Two, Prefix::Zero))]
#[case::v1_batch_partial(Case::Count(Version::One, Targets::Two, Prefix::Half))]
#[case::v1_batch_complete(Case::Count(Version::One, Targets::Two, Prefix::Complete))]
#[case::v2_batch_zero(Case::Count(Version::Two, Targets::Two, Prefix::Zero))]
#[case::v2_batch_partial(Case::Count(Version::Two, Targets::Two, Prefix::Half))]
#[case::v2_batch_complete(Case::Count(Version::Two, Targets::Two, Prefix::Complete))]
#[case::v2_batch_checksum_zero(Case::Checksum(Prefix::Zero))]
#[case::v2_batch_checksum_partial(Case::Checksum(Prefix::Half))]
#[case::v2_batch_checksum_complete(Case::Checksum(Prefix::Complete))]
#[case::v1_batch_barrier_before(Case::Barrier(Version::One, FailurePoint::Before))]
#[case::v1_batch_barrier_after(Case::Barrier(Version::One, FailurePoint::After))]
#[case::v2_batch_barrier_before(Case::Barrier(Version::Two, FailurePoint::Before))]
#[case::v2_batch_barrier_after(Case::Barrier(Version::Two, FailurePoint::After))]
fn published_count_failures_restore_complete_v1_v2_headers_and_earlier_batch_entries(
    #[case] case: Case,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let version = case.version();
    let targets = case.targets();
    let path = test_util::temp::temp_path("count-batch.h5");
    let headers = (0..targets).map(|_| version.header(3)).collect::<Vec<_>>();
    let (addresses, _) = write_count_users(&path, &headers, true);
    let control_path = test_util::temp::temp_path("count-batch-control.h5");
    std::fs::copy(&path, &control_path).unwrap();
    let properties = FileAccessProperties::new()
        .with_locking(FileLocking::Disabled)
        .with_memory_strategy(memory)
        .with_page_buffer_size(0);
    let reference = File::open_rw_with_options(&control_path, properties).unwrap();
    let before_control = std::fs::read(&control_path).unwrap();
    for &address in &addresses {
        version.assert_header(&before_control, address, &version.header(3));
    }
    stage_batch(&reference, targets);
    disk_log::start();
    reference.commit().unwrap();
    let recording = disk_log::finish();
    let published = std::fs::read(&control_path).unwrap();
    let publication = recording
        .events
        .iter()
        .position(|event| matches!(event, IoEvent::Write { offset: 0, .. }))
        .unwrap();
    let writes = addresses
        .iter()
        .map(|&address| {
            let field = FieldWrite::find(&recording, publication, address + version.count_offset());
            let Some(DiskOp::Write { bytes, .. }) = recording.events[field.event]
                .operation()
                .and_then(|operation| recording.operations.get(operation))
            else {
                panic!("missing issued count update");
            };
            assert_eq!(&bytes[field.before..field.before + 4], 2_u32.to_le_bytes());
            version.assert_header(&published, address, &version.header(2));
            field
        })
        .collect::<Vec<_>>();
    if targets == 2 {
        assert!(writes[0].event < writes[1].event);
    }
    assert_batch_values(&reference, targets);
    reference.close().unwrap();
    let last = *addresses.last().unwrap();
    let count = *writes.last().unwrap();
    let schedule = match case {
        Case::Barrier(_, point) => {
            let event = recording.events[count.event + 1..]
                .iter()
                .position(|event| matches!(event, IoEvent::Synchronize { .. }))
                .unwrap()
                + count.event
                + 1;
            Schedule::Synchronize { event, point }
        }
        Case::Checksum(prefix) => {
            let checksum = FieldWrite::find(
                &recording,
                publication,
                last + version.header(3).len() as u64 - 4,
            );
            Schedule::Write {
                event: checksum.event,
                prefix: checksum.before + prefix.bytes(),
            }
        }
        Case::Count(_, _, prefix) => Schedule::Write {
            event: count.event,
            prefix: count.before + prefix.bytes(),
        },
    };
    assert!(schedule.event() > publication);
    let file = File::open_rw_with_options(&path, properties).unwrap();
    let before = std::fs::read(&path).unwrap();
    stage_batch(&file, targets);
    disk_log::start();
    schedule.inject(&recording);
    let error = file.commit().unwrap_err();
    disk_log::clear_failures();
    let failed = disk_log::finish();
    schedule.assert_failed(&recording, &failed);
    assert_eq!(
        &failed.events[..schedule.event()],
        &recording.events[..schedule.event()]
    );
    assert_eq!(error.source().unwrap().to_string(), schedule.message());
    let Error::Io(error) = error else {
        panic!("expected the selected I/O failure: {error:?}");
    };
    assert_eq!(error.kind(), std::io::ErrorKind::Other);
    assert_eq!(error.to_string(), schedule.message());
    assert!(!file.has_staged_edits());
    let mut issued = before.clone();
    for operation in failed.events[..=schedule.event()]
        .iter()
        .filter_map(IoEvent::operation)
    {
        failed.operations[operation].apply(&mut issued);
    }
    for &address in &addresses[..addresses.len() - 1] {
        version.assert_header(&issued, address, &version.header(2));
    }
    let mut expected = version.header(3);
    let updated = version.header(2);
    match case {
        Case::Barrier(_, _) => expected = updated,
        Case::Checksum(prefix) => {
            let field = usize::try_from(version.count_offset()).unwrap();
            expected[field..field + 4].copy_from_slice(&updated[field..field + 4]);
            let checksum = expected.len() - 4;
            expected[checksum..checksum + prefix.bytes()]
                .copy_from_slice(&updated[checksum..checksum + prefix.bytes()]);
        }
        Case::Count(_, _, prefix) => {
            let field = usize::try_from(version.count_offset()).unwrap();
            expected[field..field + prefix.bytes()]
                .copy_from_slice(&updated[field..field + prefix.bytes()]);
        }
    }
    version.assert_header(&issued, last, &expected);
    let after = std::fs::read(&path).unwrap();
    let mut replay = before;
    for operation in &failed.operations {
        operation.apply(&mut replay);
    }
    assert_eq!(replay, after);
    let IoEvent::Write { requested, .. } = recording.events[publication] else {
        panic!("missing publication");
    };
    assert_eq!(&after[..requested], &published[..requested]);
    let root = root_range(&published);
    assert_eq!(&after[root.clone()], &published[root.clone()]);
    let sections = file.reusable_free_sections().unwrap();
    for range in addresses
        .iter()
        .map(|&address| {
            let start = usize::try_from(address).unwrap();
            start..start + version.header(3).len()
        })
        .chain(std::iter::once(root))
    {
        assert!(sections.sections().iter().all(|section| {
            section.end_address().get() <= range.start as u64
                || section.address().get() >= range.end as u64
        }));
    }
    for &address in &addresses {
        version.assert_header(&after, address, &version.header(3));
    }
    let syncs = failed.events[schedule.event()..]
        .iter()
        .filter_map(|event| {
            let IoEvent::Synchronize { kind, outcome } = event else {
                return None;
            };
            Some((*kind, *outcome))
        })
        .collect::<Vec<_>>();
    let restored = (disk_log::Synchronization::Data, IoOutcome::Succeeded);
    assert_eq!(
        syncs,
        if matches!(case, Case::Barrier(_, _)) {
            vec![
                (
                    disk_log::Synchronization::All,
                    IoOutcome::Failed(std::io::ErrorKind::Other),
                ),
                restored,
            ]
        } else {
            vec![restored]
        }
    );
    assert_batch_values(&file, targets);
    assert_quarantined(&file, &path, &after);
    drop(file);
    File::clear_status_flags(&path).unwrap();
    let reopened = File::open(&path).unwrap();
    assert_batch_values(&reopened, targets);
    reopened.close().unwrap();
}

#[derive(Clone, Copy)]
enum Case {
    Barrier(Version, FailurePoint),
    Checksum(Prefix),
    Count(Version, Targets, Prefix),
}

impl Case {
    fn version(self) -> Version {
        match self {
            Self::Barrier(version, _) | Self::Count(version, _, _) => version,
            Self::Checksum(_) => Version::Two,
        }
    }

    fn targets(self) -> usize {
        match self {
            Self::Count(_, Targets::One, _) => 1,
            Self::Barrier(_, _) | Self::Checksum(_) | Self::Count(_, Targets::Two, _) => 2,
        }
    }
}

#[derive(Clone, Copy)]
enum Targets {
    One,
    Two,
}

#[derive(Clone, Copy)]
enum Version {
    One,
    Two,
}

impl Version {
    fn header(self, count: u32) -> Vec<u8> {
        match self {
            Self::One => {
                let mut datatype = datatype::f64_le();
                datatype.resize(datatype.len().next_multiple_of(v1::MESSAGE_ALIGNMENT), 0);
                v1::Header::new()
                    .reference_count(count)
                    .message(Message::new(MessageType::DATATYPE, &datatype))
                    .build()
            }
            Self::Two => type_header(count),
        }
    }

    fn count_offset(self) -> u64 {
        match self {
            Self::One => 4,
            Self::Two => 12,
        }
    }

    fn assert_header(self, bytes: &[u8], address: u64, expected: &[u8]) {
        let start = usize::try_from(address).unwrap();
        assert_eq!(expected.len(), self.header(3).len());
        assert_eq!(&bytes[start..start + expected.len()], expected);
    }
}

fn stage_batch(file: &File, targets: usize) {
    stage(file, 1, true, false);
    for index in 1..targets {
        file.copy(&format!("source{index}"), &format!("copy{index}"))
            .unwrap();
        file.root().delete(&format!("type{index}")).unwrap();
        file.root().delete(&format!("alias{index}")).unwrap();
    }
}

fn assert_batch_values(file: &File, targets: usize) {
    let mut expected = vec!["source".to_owned(), "copy0".to_owned(), "value".to_owned()];
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
        if index > 0 {
            expected.extend([source, format!("copy{index}")]);
        }
    }
    expected.sort();
    let mut names = file.root().datasets().unwrap();
    names.sort();
    assert_eq!(names, expected);
    assert!(file.root().named_datatypes().unwrap().is_empty());
    assert_eq!(file.dataset("value").unwrap().read_f64().unwrap(), [31.0]);
}
