use std::io::ErrorKind;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::sync::atomic::Ordering;

use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::disk_log;
use test_util::disk_log::FailurePoint;
use test_util::disk_log::IoEvent;
use test_util::disk_log::IoOutcome;
use test_util::disk_log::Synchronization;
use test_util::temp::TempPath;

use crate::File;
use crate::Layout;
use crate::edit::SyncPolicy;
use crate::edit::WriteEngine;
use crate::error::Error;
use crate::error::FormatError;
use crate::image::WriteBuffering;
use crate::writer::FileBuilder;

use super::JournalScope;

#[rstest]
#[case::success(Outcome::Success)]
#[case::refusal(Outcome::Refusal)]
#[case::write_before(Outcome::Write(0))]
#[case::write_partial(Outcome::Write(2))]
#[case::write_complete(Outcome::Write(usize::MAX))]
#[case::barrier(Outcome::Barrier)]
#[case::restore_write(Outcome::RestoreWrite)]
#[case::restore_barrier(Outcome::RestoreBarrier)]
fn journal_scope_preserves_prior_entries_and_restores_its_overlapping_writes(
    #[case] outcome: Outcome,
    #[values(false, true)] prefix: bool,
) {
    let (path, mut engine, address) = fixture(Arc::default(), SyncPolicy::Always);
    let first = if prefix { 11 } else { 1 };
    if prefix {
        engine
            .write_inplace_journaled(address.get(), &11_i32.to_le_bytes())
            .unwrap();
    }
    let prefix_pointer = engine.inplace_undo.first().map(|(_, bytes)| bytes.as_ptr());
    disk_log::start();
    let result = JournalScope::run(&mut engine, |scope| {
        scope.write(address, &bytes([first, 21, 31]))?;
        if let Outcome::Write(written) = outcome {
            disk_log::fail_write_after(0, written);
        }
        scope.write(address, &bytes([first, 41, 51]))?;
        match outcome {
            Outcome::Barrier | Outcome::RestoreBarrier => {
                disk_log::fail_sync_after(0, FailurePoint::Before)
            }
            Outcome::RestoreWrite => disk_log::fail_write_after(0, 0),
            _ => {}
        }
        if matches!(
            outcome,
            Outcome::Refusal | Outcome::RestoreWrite | Outcome::RestoreBarrier
        ) {
            Err(refusal())
        } else {
            Ok(())
        }
    });
    let recording = disk_log::finish();
    match outcome {
        Outcome::Success => result.unwrap(),
        Outcome::Refusal => assert_refusal(result.unwrap_err()),
        Outcome::Write(_) => assert_io(result.unwrap_err(), "injected disk write failure"),
        Outcome::Barrier => assert_io(result.unwrap_err(), "injected disk synchronization failure"),
        Outcome::RestoreWrite | Outcome::RestoreBarrier => {
            let error = result.unwrap_err();
            let Error::CommitPartiallyApplied { refusal, restore } = error else {
                panic!("expected scoped restoration failure, got {error:?}")
            };
            assert_refusal(*refusal);
            assert_io(
                *restore,
                if matches!(outcome, Outcome::RestoreWrite) {
                    "injected disk write failure"
                } else {
                    "injected disk synchronization failure"
                },
            );
        }
    }
    let expected = if matches!(outcome, Outcome::Success) {
        [first, 41, 51]
    } else {
        [first, 2, 3]
    };
    assert_eq!(
        engine.image().read_exact_at(address.get(), 12).unwrap(),
        bytes(expected)
    );
    assert_eq!(engine.inplace_undo.len(), usize::from(prefix));
    assert_eq!(
        engine.inplace_undo.first().map(|(_, bytes)| bytes.as_ptr()),
        prefix_pointer
    );
    if prefix {
        assert_eq!(
            engine.inplace_undo[0],
            (address.get(), 1_i32.to_le_bytes().to_vec())
        );
    }
    let writes = recording
        .events
        .iter()
        .filter(|event| matches!(event, IoEvent::Write { .. }))
        .count();
    assert_eq!(
        writes,
        if matches!(outcome, Outcome::Success) {
            2
        } else {
            4
        }
    );
    let synchronizations = recording
        .events
        .iter()
        .filter_map(|event| match event {
            IoEvent::Synchronize { kind, outcome } => Some((*kind, *outcome)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let failure = IoOutcome::Failed(ErrorKind::Other);
    let success = IoOutcome::Succeeded;
    assert_eq!(
        synchronizations,
        match outcome {
            Outcome::Success => vec![(Synchronization::All, success)],
            Outcome::Barrier => vec![
                (Synchronization::All, failure),
                (Synchronization::Data, success)
            ],
            Outcome::RestoreWrite => vec![],
            Outcome::RestoreBarrier => vec![(Synchronization::Data, failure)],
            _ => vec![(Synchronization::Data, success)],
        }
    );
    engine.undo_inplace_writes().unwrap();
    assert!(engine.inplace_undo.is_empty());
    assert_eq!(engine.inplace_undo.capacity(), 0);
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("data").unwrap().read_i32().unwrap(),
        if matches!(outcome, Outcome::Success) {
            [1, 41, 51]
        } else {
            [1, 2, 3]
        }
    );
    drop(file);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::empty(Work::Empty)]
#[case::failure(Work::Failure)]
#[case::writing(Work::Write)]
fn empty_journal_scopes_skip_allocations_and_barriers(
    #[case] work: Work,
    #[values(false, true)] prefix: bool,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] policy: SyncPolicy,
) {
    let writing = matches!(work, Work::Write);
    let syncs = Arc::new(AtomicU64::new(0));
    let (_path, mut engine, address) = fixture(syncs.clone(), policy);
    if prefix {
        engine
            .write_inplace_journaled(address.get(), &11_i32.to_le_bytes())
            .unwrap();
    }
    syncs.store(0, Ordering::Relaxed);
    let (result, measured) = allocation::measure("scoped_journal_work", || {
        JournalScope::run(&mut engine, |scope| {
            if writing {
                scope.write(address.checked_add(4).unwrap(), &19_i32.to_le_bytes())?;
            }
            if matches!(work, Work::Failure) {
                Err(refusal())
            } else {
                Ok(())
            }
        })
    });
    if matches!(work, Work::Failure) {
        assert_refusal(result.unwrap_err());
    } else {
        result.unwrap();
    }
    assert_eq!(
        syncs.load(Ordering::Relaxed),
        u64::from(writing && matches!(policy, SyncPolicy::Always))
    );
    assert_eq!(engine.inplace_undo.len(), usize::from(prefix));
    assert_eq!(
        engine.image().read_exact_at(address.get(), 12).unwrap(),
        bytes([if prefix { 11 } else { 1 }, if writing { 19 } else { 2 }, 3])
    );
    if writing {
        assert!(measured.blocks > 0 && measured.bytes >= 4, "{measured}");
    } else {
        assert_eq!(measured.blocks, 0, "{measured}");
        assert_eq!(measured.bytes, 0, "{measured}");
    }
    drop(engine);
}

#[test]
fn an_invalid_undo_checkpoint_preserves_the_existing_journal() {
    let (_path, mut engine, address) = fixture(Arc::default(), SyncPolicy::Always);
    engine
        .write_inplace_journaled(address.get(), &11_i32.to_le_bytes())
        .unwrap();
    let pointer = engine.inplace_undo[0].1.as_ptr();
    let error = engine.undo_inplace_since(2).unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected internal checkpoint error, got {error:?}")
    };
    assert_eq!(
        error,
        FormatError::Internal("an undo checkpoint exceeds its journal".into())
    );
    assert_eq!(
        engine.inplace_undo,
        [(address.get(), 1_i32.to_le_bytes().to_vec())]
    );
    assert_eq!(engine.inplace_undo[0].1.as_ptr(), pointer);
    assert_eq!(
        engine.image().read_exact_at(address.get(), 4).unwrap(),
        11_i32.to_le_bytes()
    );
    engine.undo_inplace_writes().unwrap();
    assert_eq!(
        engine.image().read_exact_at(address.get(), 4).unwrap(),
        1_i32.to_le_bytes()
    );
    drop(engine);
}

fn fixture(syncs: Arc<AtomicU64>, policy: SyncPolicy) -> (TempPath, WriteEngine, PhysicalAddress) {
    let path = test_util::temp::temp_path("journal-scope.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
    builder.write(&path).unwrap();
    let address = {
        let file = File::open(&path).unwrap();
        let Layout::Contiguous {
            address: Some(address),
            size: 12,
        } = file.dataset("data").unwrap().layout().unwrap()
        else {
            panic!("expected twelve contiguous value bytes")
        };
        PhysicalAddress::new(address)
    };
    let mut engine = WriteEngine::open_sync_counting(&path, policy, syncs).unwrap();
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    (path, engine, address)
}

fn bytes(values: [i32; 3]) -> Vec<u8> {
    values.into_iter().flat_map(i32::to_le_bytes).collect()
}
fn refusal() -> Error {
    Error::EditUnsupported("planned scoped failure")
}
fn assert_refusal(error: Error) {
    let Error::EditUnsupported(reason) = error else {
        panic!("expected scoped failure, got {error:?}")
    };
    assert_eq!(reason, "planned scoped failure");
}
fn assert_io(error: Error, expected: &str) {
    let Error::Io(error) = error else {
        panic!("expected I/O failure, got {error:?}")
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), expected);
}

#[derive(Clone, Copy)]
enum Outcome {
    Barrier,
    Refusal,
    RestoreBarrier,
    RestoreWrite,
    Success,
    Write(usize),
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
enum Work {
    Empty,
    Failure,
    Write,
}
