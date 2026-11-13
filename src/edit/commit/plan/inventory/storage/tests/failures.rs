use std::cmp::Ordering;
use std::fs;
use std::io;
use std::path::Path;

use rstest::rstest;
use test_util::disk_log;
use test_util::disk_log::IoEvent;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use super::super::StorageIntegrityError;
use super::super::StorageUnavailable;
use crate::AttrValue;
use crate::File;
use crate::edit::WriteEngine;
use crate::edit::commit::CommitOutcome;
use crate::edit::commit::PublicationState;
use crate::edit::commit::plan::inventory::EditInputError;
use crate::edit::commit::plan::inventory::InventoryPreparation;
use crate::edit::commit::plan::inventory::OriginalEdits;
use crate::edit::commit::plan::inventory::PreparedEditInventory;
use crate::edit::commit::plan::inventory::tests as inventory_tests;
use crate::edit::commit::plan::inventory::tests::Target;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;

#[test]
fn inventory_outcome_preserves_rejected_staging() {
    let path = test_util::temp::temp_path("inventory-rejected-outcome.h5");
    let target = inventory_tests::fixture(&path, Target::Named(0));
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("type")).unwrap();
    let before = engine.disk_bytes().unwrap();
    let snapshot = engine.snapshot_free();
    let preparation = prepare(&mut engine);
    let outcome = preparation.apply(&mut engine);
    let CommitOutcome::Rejected { staged, error } = outcome else {
        panic!("expected rejected preparation");
    };
    let Error::InconsistentReferenceCount {
        address,
        observed,
        minimum,
    } = &error
    else {
        panic!("unexpected preparation error: {error:?}");
    };
    assert_eq!(
        (*address, *observed, *minimum),
        (target.header_address().get(), 0, 1)
    );
    let error = engine
        .finish_commit(snapshot, CommitOutcome::Rejected { staged, error })
        .unwrap_err();
    assert!(
        matches!(error, Error::InconsistentReferenceCount { .. }),
        "{error:?}"
    );
    assert_eq!(engine.staged.deletes(), &[ObjectPathBuf::parse("type")]);
    assert_eq!(engine.disk_bytes().unwrap(), before);
    engine.context.allocations.available().unwrap();
}

#[rstest]
#[case::claims(Capture::Claims)]
#[case::coverage(Capture::Coverage)]
#[case::parents(Capture::Parents)]
#[case::original_source(Capture::Source)]
#[case::original_domain(Capture::Domain)]
#[case::original_count_input(Capture::Counts)]
fn inventory_outcome_returns_captured_source_errors(#[case] capture: Capture) {
    let path = test_util::temp::temp_path("inventory-capture-error.h5");
    let mut fixture = Fixture::new();
    fixture.object("owner", vec![]);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    let InventoryPreparation::Tree(mut prepared) = prepare(&mut engine) else {
        panic!("expected tree preparation");
    };
    let error = Error::Io(io::Error::new(
        io::ErrorKind::UnexpectedEof,
        "source read failed",
    ));
    prepared.storage.integrity = Err(StorageIntegrityError::Unavailable);
    let source = prepared.original.source().unwrap().clone();
    prepared.storage.unavailable = Some(match capture {
        Capture::Claims => StorageUnavailable::Claims(error),
        Capture::Coverage => StorageUnavailable::Coverage(error),
        Capture::Parents => {
            prepared.storage.parents = Err(error);
            StorageUnavailable::Parents
        }
        Capture::Source | Capture::Domain | Capture::Counts => {
            prepared.original = if matches!(capture, Capture::Source) {
                OriginalEdits::Unavailable {
                    counts: Err(EditInputError::IncompleteCounts),
                    source: Err(error),
                    domain: None,
                }
            } else if matches!(capture, Capture::Domain) {
                OriginalEdits::Unavailable {
                    counts: Err(EditInputError::IncompleteCounts),
                    source: Ok(source),
                    domain: Some(Err(error)),
                }
            } else {
                OriginalEdits::Unavailable {
                    counts: Err(EditInputError::Copy(error)),
                    source: Ok(source),
                    domain: None,
                }
            };
            StorageUnavailable::Source
        }
    });
    let error = applied_error(InventoryPreparation::Tree(prepared).apply(&mut engine));
    let Error::Io(error) = error else {
        panic!("unexpected source error: {error:?}");
    };
    assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
    assert_eq!(error.to_string(), "source read failed");
    assert_eq!(engine.publication, PublicationState::Ready);
}

#[rstest]
#[case::placement(Failure::Placement)]
#[case::root_publication(Failure::Root)]
#[case::source_count_decrease(Failure::Count)]
fn inventory_outcome_preserves_consumed_failure_phase(#[case] failure: Failure) {
    let path = test_util::temp::temp_path("inventory-consumed-outcome.h5");
    let count = matches!(failure, Failure::Count);
    if count {
        inventory_tests::fixture(&path, Target::Named(2));
    } else {
        let mut fixture = Fixture::new();
        let records = Form::CompactDataset.messages(
            &mut fixture,
            &crate::make_i32_type(),
            &17_i32.to_le_bytes(),
        );
        fixture.object("value", records);
        fixture.write(&path);
    }
    let original = fs::read(&path).unwrap();
    let mut engine = open(&path, failure);
    let prepared = prepare(&mut engine);
    let count_range = match &prepared {
        InventoryPreparation::Tree(prepared) if count => Some(
            prepared.original().facts().unwrap().counts().changes()[0]
                .field()
                .unwrap()
                .chunk(),
        ),
        _ => None,
    };
    disk_log::start();
    let snapshot = engine.snapshot_free();
    let outcome = prepared.apply(&mut engine);
    assert!(matches!(outcome, CommitOutcome::Published(Ok(()))));
    engine.finish_commit(snapshot, outcome).unwrap();
    let recording = disk_log::finish();
    let publication = recording
        .events
        .iter()
        .position(|event| matches!(event, IoEvent::Write { offset: 0, .. }))
        .unwrap();
    let failed_event = match failure {
        Failure::Placement => recording
            .events
            .iter()
            .position(|event| matches!(event, IoEvent::Write { .. }))
            .unwrap(),
        Failure::Root => publication,
        Failure::Count => recording
            .events
            .iter()
            .enumerate()
            .find_map(|(index, event)| {
                matches!(event, IoEvent::Write { offset, .. } if count_range.is_some_and(|range| *offset >= range.start().get() && *offset < range.end().get()))
                    .then_some(index)
            })
            .unwrap(),
    };
    assert_eq!(
        failed_event.cmp(&publication),
        match failure {
            Failure::Placement => Ordering::Less,
            Failure::Root => Ordering::Equal,
            Failure::Count => Ordering::Greater,
        }
    );
    let failed_write = recording.events[..failed_event]
        .iter()
        .filter(|event| matches!(event, IoEvent::Write { .. }))
        .count();
    drop(engine);
    fs::write(&path, &original).unwrap();
    let mut engine = open(&path, failure);
    let snapshot = engine.snapshot_free();
    let prepared = prepare(&mut engine);
    disk_log::start();
    disk_log::fail_write_after(failed_write, 0);
    let outcome = prepared.apply(&mut engine);
    let error = applied_error(outcome);
    assert_eq!(
        engine.publication,
        match failure {
            Failure::Placement => PublicationState::Ready,
            Failure::Root => PublicationState::Uncertain,
            Failure::Count => PublicationState::Published,
        }
    );
    let error = engine
        .finish_commit(snapshot, CommitOutcome::Applied(Err(error)))
        .unwrap_err();
    disk_log::finish();
    let Error::Io(error) = error else {
        panic!("unexpected consumed failure: {error:?}");
    };
    assert_eq!(error.kind(), io::ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    assert!(engine.staged.is_empty());
    assert!(engine.inplace_undo.is_empty());
    if matches!(failure, Failure::Placement) {
        engine.context.allocations.available().unwrap();
        assert_eq!(engine.disk_bytes().unwrap(), original);
    } else {
        let error = engine.context.allocations.available().unwrap_err();
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{error:?}"
        );
        assert!(engine.disk_bytes().unwrap().len() > original.len());
        let error = engine.commit().unwrap_err();
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{error:?}"
        );
    }
    if count {
        drop(engine);
        let file = File::open(&path).unwrap();
        assert_eq!(file.root().datasets().unwrap(), ["source"]);
        assert!(file.root().named_datatypes().unwrap().is_empty());
        assert_eq!(file.dataset("source").unwrap().read_f64().unwrap(), [17.0]);
    }
}

fn prepare(engine: &mut WriteEngine) -> InventoryPreparation {
    let staged = std::mem::take(&mut engine.staged);
    PreparedEditInventory::prepare_observed(engine, staged)
}

fn open(path: &Path, failure: Failure) -> WriteEngine {
    let mut engine = edit_test_util::open_engine(path);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    if matches!(failure, Failure::Count) {
        engine.delete(&ObjectPathBuf::parse("type")).unwrap();
    } else {
        engine
            .set_dataset_attr(&ObjectPathBuf::parse("value"), "a", AttrValue::I64(9))
            .unwrap();
    }
    engine
}

fn applied_error(outcome: CommitOutcome) -> Error {
    let CommitOutcome::Applied(Err(error)) = outcome else {
        panic!("expected a consumed failure");
    };
    error
}

#[derive(Clone, Copy, Debug)]
enum Capture {
    Claims,
    Counts,
    Coverage,
    Domain,
    Parents,
    Source,
}

#[derive(Clone, Copy, Debug)]
enum Failure {
    Count,
    Placement,
    Root,
}
