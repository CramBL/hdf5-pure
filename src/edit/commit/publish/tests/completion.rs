use core::cell::Cell;
use std::collections::BTreeMap;
use std::io::ErrorKind;
use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::FreeClass;
use rstest::rstest;
use test_util::disk_log;
use test_util::disk_log::FailurePoint;
use test_util::disk_log::IoEvent;
use test_util::disk_log::IoOutcome;
use test_util::disk_log::Synchronization;
use test_util_hdf5::global_heap_fixture::Fixture;

use super::super::AppliedTree;
use crate::AttrValue;
use crate::File;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::Layout;
use crate::UserblockSize;
use crate::convert;
use crate::edit::SyncPolicy;
use crate::edit::WriteEngine;
use crate::edit::commit::PublicationState;
use crate::edit::commit::plan::PreparedCountWork;
use crate::edit::retirement::RootRetirement;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::image::WriteBuffering;
use crate::writer::FileBuilder;

#[rstest]
#[case::flat(Strategy::Flat)]
#[case::page(Strategy::Page)]
#[case::persistent(Strategy::Persistent)]
#[case::reclamation(Strategy::Reclaim)]
fn completion_borrows_the_actual_published_tree_after_its_barrier(
    #[case] strategy: Strategy,
    #[values(0, 4_096)] base: u64,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] policy: SyncPolicy,
) {
    let path = test_util::temp::temp_path("tree-completion.h5");
    let mut engine = fixture(&path, strategy, base);
    assert_eq!(
        engine.persist.is_some(),
        matches!(strategy, Strategy::Persistent | Strategy::Reclaim)
    );
    assert_eq!(
        engine.fresh_page_boundary.is_some(),
        matches!(strategy, Strategy::Page)
    );
    engine.set_sync_policy(policy);
    let root = engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap();
    engine.publication = PublicationState::Published;
    let TreeInput {
        tree,
        requested,
        predecessor,
    } = strategy.tree(&mut engine);
    disk_log::start();
    let observed = engine
        .publish_tree_with_completion(tree, |published| {
            assert_eq!(published.root(), root);
            assert_eq!(published.requested_root(), requested);
            let end = published.end();
            let engine = published.into_engine();
            assert_eq!(engine.publication, PublicationState::Published);
            assert_eq!(
                root,
                engine
                    .addresses()
                    .resolve(engine.superblock().root_group_address)
                    .unwrap()
            );
            if predecessor.is_some() {
                assert_ne!(root, requested);
            }
            assert_eq!(end.get(), engine.image().len());
            assert_eq!(engine.superblock().eof_address, end.get());
            Ok((end, disk_log::finish()))
        })
        .unwrap();
    assert!(observed.1.events.iter().any(|event| matches!(
        event,
        IoEvent::Write {
            outcome: IoOutcome::Succeeded,
            ..
        }
    )));
    if let Some(predecessor) = predecessor {
        assert_eq!(predecessor.start(), root);
        assert!(observed.1.events.iter().any(|event| matches!(event, IoEvent::Write { offset, requested, outcome: IoOutcome::Succeeded, .. } if *offset == predecessor.start().get() && *requested == usize::try_from(predecessor.length()).unwrap())));
    }
    if matches!(policy, SyncPolicy::Always) {
        assert!(observed.1.events.iter().any(|event| matches!(
            event,
            IoEvent::Synchronize {
                kind: Synchronization::All,
                outcome: IoOutcome::Succeeded
            }
        )));
    }
    assert_eq!(engine.publication, PublicationState::Published);
    assert_eq!(engine.image().len(), observed.0.get());
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("data").unwrap().read_i32().unwrap(), [1, 2, 3]);
    if matches!(strategy, Strategy::Reclaim) {
        assert_eq!(
            file.root().attrs().unwrap()["generation"],
            AttrValue::I64(-1)
        );
    }
    drop(file);
}

#[rstest]
#[case::content(Failure::Content)]
#[case::root_before(Failure::Root(0))]
#[case::root_partial(Failure::Root(2))]
#[case::root_complete(Failure::Root(usize::MAX))]
#[case::confirmation(Failure::Confirmation)]
fn completion_requires_current_publication_success_after_earlier_confirmation(
    #[case] failure: Failure,
) {
    let path = test_util::temp::temp_path("tree-completion-failure.h5");
    let mut engine = fixture(&path, Strategy::Flat, 0);
    let root = engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap();
    engine.publication = PublicationState::Published;
    let called = Cell::new(false);
    disk_log::start();
    match failure {
        Failure::Content => disk_log::fail_sync_after(0, FailurePoint::Before),
        Failure::Root(prefix) => disk_log::fail_write_after(0, prefix),
        Failure::Confirmation => disk_log::fail_sync_after(1, FailurePoint::Before),
    }
    let error = engine
        .publish_tree_with_completion(tree(root), |_| {
            called.set(true);
            Ok(())
        })
        .unwrap_err();
    let recording = disk_log::finish();
    let Error::Io(error) = error else {
        panic!("expected publication I/O failure, got {error:?}")
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(
        error.to_string(),
        if matches!(failure, Failure::Root(_)) {
            "injected disk write failure"
        } else {
            "injected disk synchronization failure"
        }
    );
    assert!(!called.get());
    assert_eq!(
        engine.publication,
        if matches!(failure, Failure::Content) {
            PublicationState::Published
        } else {
            PublicationState::Uncertain
        }
    );
    assert!(recording.events.iter().any(|event| matches!(
        event,
        IoEvent::Write {
            outcome: IoOutcome::Failed(ErrorKind::Other),
            ..
        } | IoEvent::Synchronize {
            outcome: IoOutcome::Failed(ErrorKind::Other),
            ..
        }
    )));
    drop(engine);
}

#[test]
fn completion_failure_preserves_root_confirmation_and_earlier_journal_entries() {
    let path = test_util::temp::temp_path("tree-completion-refusal.h5");
    let mut engine = fixture(&path, Strategy::Flat, 0);
    let root = engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap();
    let original = engine.image().read_exact_at(root.get(), 1).unwrap();
    engine
        .write_inplace_journaled(root.get(), &original)
        .unwrap();
    let journal_pointer = engine.inplace_undo[0].1.as_ptr();
    let error = engine
        .publish_tree_with_completion(tree(root), |published| {
            assert_eq!(published.root(), root);
            Err::<(), _>(Error::EditUnsupported("completion failure"))
        })
        .unwrap_err();
    let Error::EditUnsupported(reason) = error else {
        panic!("expected completion failure, got {error:?}")
    };
    assert_eq!(reason, "completion failure");
    assert_eq!(engine.publication, PublicationState::Published);
    assert_eq!(engine.inplace_undo.len(), 1);
    assert_eq!(engine.inplace_undo[0].1.as_ptr(), journal_pointer);
    assert_eq!(engine.inplace_undo[0], (root.get(), original));
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("data").unwrap().read_i32().unwrap(), [1, 2, 3]);
    drop(file);
}

#[rstest]
#[case::success(None)]
#[case::reference_write(Some(ReferenceFailure::Write))]
#[case::reference_barrier(Some(ReferenceFailure::Barrier))]
fn completion_runs_after_stored_references_are_updated(#[case] failure: Option<ReferenceFailure>) {
    let path = test_util::temp::temp_path("tree-completion-references.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
    builder
        .create_dataset("refs")
        .with_path_references(&["data"]);
    builder.write(&path).unwrap();
    let file = File::open(&path).unwrap();
    let Layout::Contiguous {
        address: Some(address),
        size,
    } = file.dataset("refs").unwrap().layout().unwrap()
    else {
        panic!("expected contiguous reference storage")
    };
    assert_eq!(size, u64::try_from(core::mem::size_of::<u64>()).unwrap());
    drop(file);
    let bytes = std::fs::read(&path).unwrap();
    let original = u64::from_le_bytes(
        bytes[convert::slice_range(address, size).unwrap()]
            .try_into()
            .unwrap(),
    );
    let mut engine = edit_test_util::open_engine(&path);
    engine.set_sync_policy(SyncPolicy::Always);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let root = engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap();
    let mut tree = tree(root);
    tree.relocations.insert(original, root.get());
    let called = Cell::new(false);
    disk_log::start();
    match failure {
        Some(ReferenceFailure::Write) => disk_log::fail_write_after(1, 0),
        Some(ReferenceFailure::Barrier) => disk_log::fail_sync_after(2, FailurePoint::Before),
        None => {}
    }
    let result = engine.publish_tree_with_completion(tree, |published| {
        called.set(true);
        let engine = published.into_engine();
        assert_eq!(
            engine
                .image()
                .read_exact_at(address, usize::try_from(size).unwrap())
                .unwrap(),
            root.get().to_le_bytes()
        );
        Ok(())
    });
    let recording = disk_log::finish();
    assert_eq!(engine.publication, PublicationState::Published);
    if let Some(failure) = failure {
        let Error::Io(error) = result.unwrap_err() else {
            panic!("expected reference-update I/O failure")
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(
            error.to_string(),
            if matches!(failure, ReferenceFailure::Write) {
                "injected disk write failure"
            } else {
                "injected disk synchronization failure"
            }
        );
        assert!(!called.get());
    } else {
        result.unwrap();
        assert!(called.get());
        let writes = recording
            .events
            .iter()
            .filter_map(|event| match event {
                IoEvent::Write { offset, .. } => Some(*offset),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(writes, [engine.context.signature().get(), address]);
        drop(engine);
        let file = File::open(&path).unwrap();
        let references = file.dataset("refs").unwrap().dereference().unwrap();
        assert!(matches!(&references[0], crate::Object::Group(_)));
        drop(references);
        drop(file);
    }
}

#[rstest]
#[case::success(None)]
#[case::before(Some(FailurePoint::Before))]
#[case::after(Some(FailurePoint::After))]
fn completion_requires_the_requested_truncation_to_finish(#[case] failure: Option<FailurePoint>) {
    let path = test_util::temp::temp_path("tree-completion-truncation.h5");
    let mut source = Fixture::new();
    let tail = vec![7; 4_096];
    let end = u64::try_from(source.image.append(&tail)).unwrap();
    source.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine.set_sync_policy(SyncPolicy::Always);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let root = engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap();
    let mut tree = tree(root);
    tree.freed.push((
        FileRange::new(
            PhysicalAddress::new(end),
            u64::try_from(tail.len()).unwrap(),
        )
        .unwrap(),
        FreeClass::Dead,
    ));
    let called = Cell::new(false);
    disk_log::start();
    if let Some(point) = failure {
        disk_log::fail_resize_after(0, point);
    }
    let result = engine.publish_tree_with_completion(tree, |published| {
        called.set(true);
        assert_eq!(published.end().get(), end);
        assert_eq!(published.into_engine().image().len(), end);
        Ok(())
    });
    let recording = disk_log::finish();
    assert_eq!(engine.publication, PublicationState::Published);
    assert_eq!(engine.superblock().eof_address, end);
    assert!(
        recording
            .events
            .iter()
            .any(|event| matches!(event, IoEvent::Resize { .. }))
    );
    if failure.is_some() {
        let Error::Io(error) = result.unwrap_err() else {
            panic!("expected truncation I/O failure")
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(error.to_string(), "injected disk resize failure");
        assert!(!called.get());
    } else {
        result.unwrap();
        assert!(called.get());
    }
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), Vec::<String>::new());
    assert_eq!(file.root().groups().unwrap(), Vec::<String>::new());
    drop(file);
}

fn fixture(path: &Path, strategy: Strategy, base: u64) -> WriteEngine {
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(base).unwrap());
    builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
    match strategy {
        Strategy::Flat => {}
        Strategy::Page | Strategy::Persistent => {
            builder.with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::Page)
                    .with_persistence(matches!(strategy, Strategy::Persistent)),
            );
        }
        Strategy::Reclaim => {
            builder.with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
                    .with_persistence(true)
                    .with_threshold(u64::MAX),
            );
            builder.set_attr("generation", AttrValue::I64(-1));
        }
    }
    builder.write(path).unwrap();
    let mut engine = edit_test_util::open_engine(path);
    engine.set_sync_policy(SyncPolicy::Always);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine
}

fn tree(root: PhysicalAddress) -> AppliedTree {
    AppliedTree {
        counts: PreparedCountWork::Empty,
        retirement: RootRetirement::new(root, vec![]),
        freed: vec![],
        relocations: BTreeMap::new(),
    }
}

#[derive(Clone, Copy)]
enum Strategy {
    Flat,
    Page,
    Persistent,
    Reclaim,
}

impl Strategy {
    fn tree(self, engine: &mut WriteEngine) -> TreeInput {
        let root = engine
            .addresses()
            .resolve(engine.superblock().root_group_address)
            .unwrap();
        if !matches!(self, Self::Reclaim) {
            return TreeInput {
                tree: tree(root),
                requested: root,
                predecessor: None,
            };
        }
        let spans = engine.oh_chunk_spans(root.get()).unwrap();
        let [predecessor] = spans.as_slice() else {
            panic!("expected one root allocation")
        };
        assert!(predecessor.length() < engine.persist.as_ref().unwrap().threshold);
        let current = engine
            .image()
            .read_exact_at(root.get(), usize::try_from(predecessor.length()).unwrap())
            .unwrap();
        let requested = PhysicalAddress::new(engine.place_object_header(&current).unwrap());
        assert!(requested > root);
        assert_eq!(
            engine.oh_chunk_spans(requested.get()).unwrap(),
            [FileRange::new(requested, predecessor.length()).unwrap()]
        );
        let mut tree = tree(requested);
        tree.freed = engine.object_header_free_spans(&spans).unwrap();
        assert_eq!(tree.freed.len(), 1);
        assert_eq!(tree.freed[0].0, *predecessor);
        TreeInput {
            tree,
            requested,
            predecessor: Some(*predecessor),
        }
    }
}

struct TreeInput {
    tree: AppliedTree,
    requested: PhysicalAddress,
    predecessor: Option<FileRange>,
}

#[derive(Clone, Copy)]
enum Failure {
    Confirmation,
    Content,
    Root(usize),
}

#[derive(Clone, Copy)]
enum ReferenceFailure {
    Barrier,
    Write,
}
