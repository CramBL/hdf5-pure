use std::fs;
use std::io;
use std::path::Path;

use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::disk_log;
use test_util::disk_log::FailurePoint;
use test_util::disk_log::IoEvent;
use test_util::global_heap::MIN_COLLECTION_SIZE;
use test_util::global_heap::Object;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use super::super::PendingEditInventory;
use crate::AttrValue;
use crate::File;
use crate::FileBuilder;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::HeapCompactionState;
use crate::Layout;
use crate::edit::SyncPolicy;
use crate::edit::WriteEngine;
use crate::edit::commit::CommitOutcome;
use crate::edit::commit::PublicationState;
use crate::edit::commit::heap::HeapMutationFailure;
use crate::edit::commit::heap::HeapMutationState;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::rejected(Outcome::Rejected, false)]
#[case::before_publication(Outcome::BeforePublication, false)]
#[case::uncertain_publication(Outcome::UncertainPublication, false)]
#[case::generic_after_publication(Outcome::GenericPublished, false)]
#[case::restored(Outcome::Restored, false)]
#[case::restored_page(Outcome::Restored, true)]
#[case::uncertain_collection(Outcome::UncertainCollection, false)]
#[case::uncertain_collection_page(Outcome::UncertainCollection, true)]
#[case::lost_allocation_permission(Outcome::LostPermission, false)]
#[case::lost_page_allocation_permission(Outcome::LostPermission, true)]
#[case::untrusted_error(Outcome::UntrustedError, false)]
fn commit_finalization_preserves_publication_and_allocation_evidence(
    #[case] outcome: Outcome,
    #[case] page: bool,
) {
    let path = test_util::temp::temp_path("commit-finalization-outcome.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(if page {
        FileSpaceStrategy::Page
    } else {
        FileSpaceStrategy::FsmAggr
    }));
    builder.create_dataset("data").with_i32_data(&[7]);
    builder.write(&path).unwrap();
    let address = {
        let file = File::open(&path).unwrap();
        let Layout::Contiguous {
            address: Some(address),
            ..
        } = file.dataset("data").unwrap().layout().unwrap()
        else {
            panic!("expected contiguous values");
        };
        PhysicalAddress::new(address)
    };
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    assert_eq!(engine.fresh_page_boundary.is_some(), page);
    let generation = engine.staged_generation;
    let snapshot = engine.snapshot_free();
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    let staged = std::mem::take(&mut engine.staged);
    engine
        .write_inplace_journaled(address, &11_i32.to_le_bytes())
        .unwrap();
    engine.publication = match outcome {
        Outcome::Rejected | Outcome::BeforePublication => PublicationState::Ready,
        Outcome::UncertainPublication => PublicationState::Uncertain,
        _ => PublicationState::Published,
    };
    let published = engine.publication.attempted();
    let operation = Error::Io(io::Error::other("collection failed"));
    let typed = match outcome {
        Outcome::Rejected => CommitOutcome::Rejected {
            staged,
            error: operation,
        },
        Outcome::BeforePublication | Outcome::UncertainPublication | Outcome::GenericPublished => {
            CommitOutcome::Applied(Err(operation))
        }
        Outcome::UntrustedError => CommitOutcome::Applied(Err(Error::HeapCompactionFailed {
            completed: 2,
            state: HeapCompactionState::Restored,
            source: Box::new(operation),
        })),
        Outcome::Restored | Outcome::UncertainCollection | Outcome::LostPermission => {
            if matches!(outcome, Outcome::LostPermission) {
                engine.make_allocation_uncertain();
            }
            CommitOutcome::Published(Err(HeapMutationFailure {
                completed: 2,
                state: if matches!(outcome, Outcome::UncertainCollection) {
                    HeapMutationState::Uncertain
                } else {
                    HeapMutationState::Restored
                },
                error: operation,
            }))
        }
    };
    let error = engine.finish_commit(snapshot, typed).unwrap_err();
    let confirmed = matches!(outcome, Outcome::Restored);
    if matches!(
        outcome,
        Outcome::Restored
            | Outcome::UncertainCollection
            | Outcome::LostPermission
            | Outcome::UntrustedError
    ) {
        let Error::HeapCompactionFailed {
            completed,
            state,
            source,
        } = error
        else {
            panic!("unexpected compaction error: {error:?}");
        };
        assert_eq!(completed, 2);
        assert_eq!(
            state,
            if matches!(
                outcome,
                Outcome::UncertainCollection | Outcome::LostPermission
            ) {
                HeapCompactionState::Uncertain
            } else {
                HeapCompactionState::Restored
            }
        );
        assert_operation(*source);
    } else {
        assert_operation(error);
    }
    assert_eq!(
        engine.image().read_exact_at(address.get(), 4).unwrap(),
        if published { 11_i32 } else { 7_i32 }.to_le_bytes()
    );
    assert!(engine.inplace_undo.is_empty());
    assert_eq!(
        engine.staged.groups(),
        if matches!(outcome, Outcome::Rejected) {
            vec![ObjectPathBuf::parse("new")]
        } else {
            vec![]
        }
    );
    assert_eq!(engine.staged_generation, generation + u64::from(published));
    if published && !confirmed {
        let error = engine.context.allocations.available().unwrap_err();
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{error:?}"
        );
    } else {
        engine.context.allocations.available().unwrap();
    }
    if let Some(aggregation) = &engine.aggregation {
        assert_eq!(aggregation.publication_confirmed, !published || confirmed);
    }
    if page && confirmed {
        assert_eq!(engine.publication, PublicationState::Ready);
    }
}

#[rstest]
#[case::unbuffered(Profile::Unbuffered)]
#[case::session(Profile::Session)]
#[case::mirrored(Profile::Mirrored)]
#[case::restoration_barrier(Profile::RestorationBarrier)]
fn later_compaction_failure_keeps_published_values_and_completed_collections(
    #[case] profile: Profile,
) {
    let path = test_util::temp::temp_path("commit-later-compaction-failure.h5");
    let mut fixture = Fixture::new();
    let parents = [
        fixture.collection(&[Object::new(7, b"removed"), Object::new(91, b"retained")]),
        fixture.collection(&[Object::new(7, b"removed"), Object::new(91, b"retained")]),
        fixture.collection(&[Object::new(7, b"removed"), Object::new(91, b"retained")]),
    ];
    let values = parents
        .iter()
        .flat_map(|&parent| global_heap_fixture::reference(7, parent, 7))
        .collect::<Vec<_>>();
    let raw = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    let mut messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &crate::make_i32_type(),
        &17_i32.to_le_bytes(),
    );
    messages.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &values,
    ));
    fixture.object("value", messages);
    fixture.write(&path);
    let original = fs::read(&path).unwrap();
    let mut engine = open(&path, profile);
    engine
        .set_dataset_attr(&ObjectPathBuf::parse("value"), "a", AttrValue::I64(9))
        .unwrap();
    let prepared = super::prepare(&mut engine);
    let snapshot = engine.snapshot_free();
    let applied = prepared.apply(&mut engine).unwrap();
    engine
        .write_inplace_journaled(PhysicalAddress::new(raw), &23_i32.to_le_bytes())
        .unwrap();
    let pending = PendingEditInventory::new(&mut engine, applied);
    assert_eq!(pending.member_decisions().plans().count(), 3);
    let failed_start = pending.member_decisions().plans().nth(1).unwrap().moves()[0]
        .destination()
        .start()
        .get();
    disk_log::start();
    let completion = pending.publish_compacting_members().ok().unwrap();
    engine
        .finish_commit(snapshot, completion.into_outcome())
        .unwrap();
    let recording = disk_log::finish();
    let failed_event = recording
        .events
        .iter()
        .position(|event| matches!(event, IoEvent::Write { offset, .. } if *offset == failed_start))
        .unwrap();
    let failed_at = recording.events[..failed_event]
        .iter()
        .filter(|event| matches!(event, IoEvent::Write { .. }))
        .count();
    let restore_barrier = recording.events[..failed_event]
        .iter()
        .filter(|event| matches!(event, IoEvent::Synchronize { .. }))
        .count();
    drop(engine);
    fs::write(&path, &original).unwrap();
    let mut engine = open(&path, profile);
    engine
        .set_dataset_attr(&ObjectPathBuf::parse("value"), "a", AttrValue::I64(9))
        .unwrap();
    let snapshot = engine.snapshot_free();
    let applied = super::prepare(&mut engine).apply(&mut engine).unwrap();
    engine
        .write_inplace_journaled(PhysicalAddress::new(raw), &23_i32.to_le_bytes())
        .unwrap();
    let pending = PendingEditInventory::new(&mut engine, applied);
    disk_log::start();
    disk_log::fail_write_after(failed_at, 2);
    let uncertain = matches!(profile, Profile::RestorationBarrier);
    if uncertain {
        disk_log::fail_sync_after(restore_barrier, FailurePoint::Before);
    }
    let completion = pending.publish_compacting_members().ok().unwrap();
    let error = engine
        .finish_commit(snapshot, completion.into_outcome())
        .unwrap_err();
    disk_log::finish();
    let Error::HeapCompactionFailed {
        completed,
        state,
        source,
    } = error
    else {
        panic!("unexpected error: {error:?}");
    };
    assert_eq!(completed, 1);
    assert_eq!(
        state,
        if uncertain {
            HeapCompactionState::Uncertain
        } else {
            HeapCompactionState::Restored
        }
    );
    let source = if uncertain {
        assert_eq!(
            source.to_string(),
            "an operation failed (I/O error: injected disk write failure) and restoration could not be confirmed (I/O error: injected disk synchronization failure)"
        );
        let Error::CommitPartiallyApplied { refusal, restore } = *source else {
            panic!("expected both restoration errors: {source:?}");
        };
        let Error::Io(restore) = *restore else {
            panic!("unexpected restore error: {restore:?}");
        };
        assert_eq!(restore.kind(), io::ErrorKind::Other);
        assert_eq!(restore.to_string(), "injected disk synchronization failure");
        refusal
    } else {
        source
    };
    let Error::Io(error) = *source else {
        panic!("unexpected source: {source:?}");
    };
    assert_eq!(error.kind(), io::ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    if uncertain {
        let error = engine.context.allocations.available().unwrap_err();
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{error:?}"
        );
    } else {
        engine.context.allocations.available().unwrap();
    }
    assert!(engine.inplace_undo.is_empty());
    assert!(engine.staged.is_empty());
    assert_eq!(
        engine.image().read_exact_at(raw, 4).unwrap(),
        23_i32.to_le_bytes()
    );
    let first = crate::global_heap::collect_members(
        engine.image(),
        engine.widths().lengths,
        PhysicalAddress::new(parents[0]),
    )
    .unwrap();
    assert_eq!(
        first
            .members()
            .iter()
            .map(|member| member.index().get())
            .collect::<Vec<_>>(),
        [91]
    );
    let parent_length = u64::try_from(MIN_COLLECTION_SIZE).unwrap();
    assert_eq!(
        engine
            .image()
            .read_exact_at(parents[1], MIN_COLLECTION_SIZE)
            .unwrap(),
        original[usize::try_from(parents[1]).unwrap()
            ..usize::try_from(parents[1].checked_add(parent_length).unwrap()).unwrap()]
    );
    assert_eq!(
        engine
            .image()
            .read_exact_at(parents[2], MIN_COLLECTION_SIZE)
            .unwrap(),
        original[usize::try_from(parents[2]).unwrap()
            ..usize::try_from(parents[2].checked_add(parent_length).unwrap()).unwrap()]
    );
    if uncertain {
        let error = engine.commit().unwrap_err();
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{error:?}"
        );
    } else {
        let retained_length = MIN_COLLECTION_SIZE.checked_mul(2).unwrap();
        let retained = engine
            .image()
            .read_exact_at(parents[1], retained_length)
            .unwrap()
            .to_vec();
        engine.commit().unwrap();
        assert_eq!(
            engine
                .image()
                .read_exact_at(parents[1], retained_length)
                .unwrap(),
            retained
        );
        engine.force_sync().unwrap();
    }
    drop(engine);
    let file = File::open(&path).unwrap();
    let dataset = file.dataset("value").unwrap();
    assert_eq!(dataset.read_i32().unwrap(), [23]);
    assert_eq!(
        dataset.attrs().unwrap().remove("a"),
        Some(AttrValue::I64(9))
    );
}

fn open(path: &Path, profile: Profile) -> WriteEngine {
    let mut engine = if matches!(profile, Profile::Mirrored) {
        WriteEngine::open_source_only(path).unwrap()
    } else {
        edit_test_util::open_engine(path)
    };
    engine.set_sync_policy(if matches!(profile, Profile::RestorationBarrier) {
        SyncPolicy::Always
    } else {
        SyncPolicy::OnClose
    });
    engine
        .set_write_buffering(if matches!(profile, Profile::Session) {
            WriteBuffering::Session {
                page_size: 4096,
                max_bytes: 1_000_000,
            }
        } else {
            WriteBuffering::Unbuffered
        })
        .unwrap();
    engine
}

fn assert_operation(error: Error) {
    let Error::Io(error) = error else {
        panic!("unexpected operation error: {error:?}");
    };
    assert_eq!(error.kind(), io::ErrorKind::Other);
    assert_eq!(error.to_string(), "collection failed");
}

#[derive(Clone, Copy, Debug)]
enum Outcome {
    BeforePublication,
    GenericPublished,
    LostPermission,
    Rejected,
    Restored,
    UncertainCollection,
    UncertainPublication,
    UntrustedError,
}

#[derive(Clone, Copy, Debug)]
enum Profile {
    Mirrored,
    RestorationBarrier,
    Session,
    Unbuffered,
}
