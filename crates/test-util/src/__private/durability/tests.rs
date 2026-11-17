use core::ptr;
use std::fs::File;
use std::io;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;

use rstest::rstest;

use super::FaultSchedule;
use crate::disk_log;
use crate::disk_log::FailurePoint;
use crate::disk_log::Synchronization;

#[rstest]
#[case::write_before(Fault::Write(Prefix::Before))]
#[case::write_partial(Fault::Write(Prefix::Partial))]
#[case::write_complete(Fault::Write(Prefix::Complete))]
#[case::slice_write_before(Fault::SliceWrite(Prefix::Before))]
#[case::slice_write_partial(Fault::SliceWrite(Prefix::Partial))]
#[case::slice_write_complete(Fault::SliceWrite(Prefix::Complete))]
#[case::slice_synchronize_before(Fault::SliceSynchronize(FailurePoint::Before))]
#[case::slice_synchronize_after(Fault::SliceSynchronize(FailurePoint::After))]
#[case::merged_write(Fault::InsideWrite)]
#[case::synchronize_before(Fault::Synchronize(FailurePoint::Before))]
#[case::synchronize_after(Fault::Synchronize(FailurePoint::After))]
#[case::recovery_write(Fault::RecoveryWrite)]
#[case::recovery_synchronization(Fault::RecoverySynchronization)]
fn scheduled_faults_preserve_attempt_ordinals_and_completed_prefixes(#[case] fault: Fault) {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    disk_log::clear_failures();
    disk_log::start();
    write(file.as_file_mut()).unwrap();
    let control = disk_log::finish();
    assert_eq!(control.events.len(), 5);
    assert_eq!(control.operations.len(), 3);
    let schedule = match fault {
        Fault::Write(prefix) => FaultSchedule::write(&control, 3, prefix.written()),
        Fault::SliceWrite(prefix) => {
            FaultSchedule::write_events(&control.events, 3, prefix.written())
        }
        Fault::SliceSynchronize(point) => {
            FaultSchedule::synchronize_events(&control.events, 4, point)
        }
        Fault::InsideWrite => FaultSchedule::write_containing(&control, 14, 1),
        Fault::Synchronize(point) => FaultSchedule::synchronize_after(&control, 3, point),
        Fault::RecoveryWrite => {
            FaultSchedule::synchronize(&control, 4, FailurePoint::Before).with_recovery_write(0)
        }
        Fault::RecoverySynchronization => {
            FaultSchedule::write(&control, 3, 2).with_recovery_synchronization(FailurePoint::Before)
        }
    };
    assert!(ptr::eq(schedule.events, control.events.as_slice()));
    assert_eq!(
        schedule.event(),
        if fault.synchronization() { 4 } else { 3 }
    );
    file.as_file_mut().set_len(0).unwrap();
    disk_log::start();
    schedule.inject();
    let error = write(file.as_file_mut()).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::Other);
    assert_eq!(
        error.to_string(),
        if fault.synchronization() {
            "injected disk synchronization failure"
        } else {
            "injected disk write failure"
        }
    );
    if matches!(fault, Fault::RecoveryWrite | Fault::RecoverySynchronization) {
        file.as_file_mut().seek(SeekFrom::Start(20)).unwrap();
        let restore = disk_log::write_all(file.as_file_mut(), 20, b"undo");
        if matches!(fault, Fault::RecoveryWrite) {
            let error = restore.unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::Other);
            assert_eq!(error.to_string(), "injected disk write failure");
        } else {
            restore.unwrap();
            let error = disk_log::synchronize(file.as_file(), Synchronization::All).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::Other);
            assert_eq!(error.to_string(), "injected disk synchronization failure");
        }
    }
    let failed = disk_log::finish();
    disk_log::clear_failures();
    schedule.assert_failed(&failed);
    let mut expected = vec![0; 24];
    expected[4..8].copy_from_slice(b"base");
    let written = match fault {
        Fault::Write(prefix) | Fault::SliceWrite(prefix) => prefix.written().min(4),
        Fault::InsideWrite => 3,
        Fault::RecoverySynchronization => 2,
        Fault::RecoveryWrite | Fault::SliceSynchronize(_) | Fault::Synchronize(_) => 4,
    };
    expected[12..12 + written].copy_from_slice(&b"tail"[..written]);
    if matches!(fault, Fault::RecoverySynchronization) {
        expected[20..24].copy_from_slice(b"undo");
    }
    file.as_file_mut().seek(SeekFrom::Start(0)).unwrap();
    let mut actual = Vec::new();
    file.as_file_mut().read_to_end(&mut actual).unwrap();
    assert_eq!(actual, expected);
    file.close().unwrap();
}

#[rstest]
#[case::recording_before(Trace::Recording, FailurePoint::Before)]
#[case::recording_after(Trace::Recording, FailurePoint::After)]
#[case::events_before(Trace::Events, FailurePoint::Before)]
#[case::events_after(Trace::Events, FailurePoint::After)]
fn scheduled_resizes_preserve_completed_operations(
    #[case] trace: Trace,
    #[case] point: FailurePoint,
) {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    disk_log::clear_failures();
    disk_log::start();
    resize(file.as_file_mut()).unwrap();
    let control = disk_log::finish();
    assert_eq!(control.events.len(), 5);
    assert_eq!(control.operations.len(), 4);
    let schedule = match trace {
        Trace::Recording => FaultSchedule::resize(&control, 4, point),
        Trace::Events => FaultSchedule::resize_events(&control.events, 4, point),
    };
    assert!(ptr::eq(schedule.events, control.events.as_slice()));
    assert_eq!(schedule.event(), 4);
    file.as_file_mut().set_len(0).unwrap();
    disk_log::start();
    schedule.inject();
    let error = resize(file.as_file_mut()).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk resize failure");
    let failed = disk_log::finish();
    disk_log::clear_failures();
    schedule.assert_failed(&failed);
    let length = if point == FailurePoint::Before {
        32
    } else {
        20
    };
    assert_eq!(file.as_file().metadata().unwrap().len(), length);
    let mut expected = vec![0; usize::try_from(length).unwrap()];
    expected[4..8].copy_from_slice(b"base");
    expected[12..16].copy_from_slice(b"tail");
    file.as_file_mut().seek(SeekFrom::Start(0)).unwrap();
    let mut actual = Vec::new();
    file.as_file_mut().read_to_end(&mut actual).unwrap();
    assert_eq!(actual, expected);
    file.close().unwrap();
}

#[derive(Clone, Copy)]
enum Trace {
    Events,
    Recording,
}

#[derive(Clone, Copy)]
enum Fault {
    InsideWrite,
    RecoverySynchronization,
    RecoveryWrite,
    SliceSynchronize(FailurePoint),
    SliceWrite(Prefix),
    Synchronize(FailurePoint),
    Write(Prefix),
}

impl Fault {
    fn synchronization(self) -> bool {
        matches!(
            self,
            Self::RecoveryWrite | Self::SliceSynchronize(_) | Self::Synchronize(_)
        )
    }
}

#[derive(Clone, Copy)]
enum Prefix {
    Before,
    Complete,
    Partial,
}

impl Prefix {
    fn written(self) -> usize {
        match self {
            Self::Before => 0,
            Self::Complete => usize::MAX,
            Self::Partial => 2,
        }
    }
}

fn write(file: &mut File) -> io::Result<()> {
    file.seek(SeekFrom::Start(4))?;
    disk_log::write_all(file, 4, b"base")?;
    disk_log::synchronize(file, Synchronization::Data)?;
    disk_log::set_len(file, 24)?;
    file.seek(SeekFrom::Start(12))?;
    disk_log::write_all(file, 12, b"tail")?;
    disk_log::synchronize(file, Synchronization::All)
}

fn resize(file: &mut File) -> io::Result<()> {
    file.seek(SeekFrom::Start(4))?;
    disk_log::write_all(file, 4, b"base")?;
    disk_log::set_len(file, 32)?;
    disk_log::synchronize(file, Synchronization::Data)?;
    file.seek(SeekFrom::Start(12))?;
    disk_log::write_all(file, 12, b"tail")?;
    disk_log::set_len(file, 20)
}
