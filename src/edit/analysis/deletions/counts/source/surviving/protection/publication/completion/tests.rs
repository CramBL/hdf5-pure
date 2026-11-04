use core::cell::Cell;
use std::collections::BTreeMap;
use std::io::ErrorKind;
use std::path::Path;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::sync::atomic::Ordering;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::SessionSpace;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::disk_log;
use test_util::disk_log::FailurePoint;
use test_util::file_space_info;
use test_util::identity;
use test_util::widths::Widths;

use super::super::super::tests as protection_tests;
use super::super::super::tests::CountStorage;
use super::super::CountPublicationBounds;
use super::super::CountPublicationPhase;
use super::CountCompletion;
use super::CountCompletionError;
use super::CountProtectionError;
use super::CountPublicationError;
use crate::File;
use crate::UserblockSize;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::edit::WriteEngine;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::edit::analysis::DISCOVERY_WORK;
use crate::edit::commit::PublicationState;
use crate::edit::group;
use crate::edit::object_header;
use crate::edit::retirement::PendingRootFrees;
use crate::edit::retirement::RootRetirement;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::error::FormatError;
use crate::file_space_info::FileSpacePageSize;
use crate::file_space_info::FileSpaceStrategy;
use crate::image::WriteBuffering;
use crate::object_path::LinkNameBuf;

#[rstest]
#[case::header(CountStorage::Header)]
#[case::continuation(CountStorage::Continuation)]
fn confirmed_scope_keeps_original_facts_and_consumes_the_original_retirement_carrier(
    #[case] storage: CountStorage,
    #[values(0, 512)] base: u64,
) {
    let path = test_util::temp::temp_path("count-write-permission.h5");
    let fixture = protection_tests::fixture_with_retirements(
        &path,
        UserblockSize::try_from(base).unwrap(),
        storage,
    );
    let mut engine = edit_test_util::open_engine(&path);
    let bounds = bounds(&mut engine, fixture.retired.to_vec(), &["a"]);
    let original_root = bounds.original_root();
    let root = bounds.root();
    let end = bounds.end();
    assert_eq!(bounds.phase.retirement.batch.entries(), fixture.retired);
    let object = bounds.updates()[0].object();
    let allocation = bounds.updates()[0].allocation().range();
    let field = bounds.updates()[0].field();
    let before = bounds
        .phase
        .engine
        .image()
        .read_exact_at(
            field.range().start().get(),
            usize::try_from(field.range().length()).unwrap(),
        )
        .unwrap();
    let result = bounds
        .with_publication(BTreeMap::new(), |scope| {
            let scope = scope.unwrap();
            assert_eq!(scope.counts.root(), original_root);
            assert_eq!(scope.root, root);
            assert_eq!(scope.end, end);
            assert_eq!(scope.proposed_end, end);
            assert_eq!(scope.counts.updates().len(), 1);
            assert_eq!(scope.counts.updates()[0].object(), object);
            assert_eq!(scope.counts.updates()[0].allocation().range(), allocation);
            assert_eq!(scope.counts.updates()[0].field(), field);
            assert_eq!(scope.engine.publication, PublicationState::Published);
            assert_eq!(
                scope
                    .engine
                    .image()
                    .read_exact_at(
                        field.range().start().get(),
                        usize::try_from(field.range().length()).unwrap(),
                    )
                    .unwrap(),
                before.as_slice()
            );
            let retired = scope.engine.space.reusable_sections().collect::<Vec<_>>();
            assert_eq!(
                retired,
                vec![
                    Extent::new(
                        scope
                            .engine
                            .addresses()
                            .relative(fixture.retired[0].0.start())
                            .unwrap(),
                        fixture
                            .retired
                            .iter()
                            .map(|(range, _)| range.length())
                            .sum(),
                    )
                    .unwrap()
                ]
            );
            Ok((scope.root, scope.end))
        })
        .unwrap();
    assert_eq!(result, (root, end));
    assert_eq!(
        engine
            .addresses()
            .resolve(engine.superblock().root_group_address)
            .unwrap(),
        root
    );
    assert_eq!(engine.superblock().eof_address, end.get());
}

#[rstest]
#[case::trailing_header(PublicationChange::Truncate(CountStorage::Header))]
#[case::trailing_continuation(PublicationChange::Truncate(CountStorage::Continuation))]
#[case::reclaimed_root(PublicationChange::Reclaim)]
fn confirmed_scope_uses_the_adopted_root_and_eoa(#[case] change: PublicationChange) {
    let path = test_util::temp::temp_path("count-confirmed-geometry.h5");
    let removed = match change {
        PublicationChange::Truncate(storage) => {
            protection_tests::fixture_with_storage(&path, UserblockSize::NONE, storage);
            "a"
        }
        PublicationChange::Reclaim => {
            reclamation_fixture(&path);
            "aa"
        }
    };
    let mut engine = edit_test_util::open_engine(&path);
    let original_root = engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap();
    let (counts, _) = protection_tests::counts(&engine, &[removed]);
    let counts = counts.protect(&engine, []).unwrap();
    assert_eq!(counts.updates().len(), 1);
    let object = counts.updates()[0].object();
    let allocation = counts.updates()[0].allocation().range();
    let field = counts.updates()[0].field();
    let original = engine
        .image()
        .read_exact_at(
            allocation.start().get(),
            usize::try_from(allocation.length()).unwrap(),
        )
        .unwrap();
    let mut retired = Vec::new();
    let requested_root = match change {
        PublicationChange::Truncate(_) => replacement_root(&mut engine, &[removed]),
        PublicationChange::Reclaim => {
            let spans = engine.oh_chunk_spans(original_root).unwrap();
            let [predecessor] = spans.as_slice() else {
                panic!("expected one predecessor root allocation")
            };
            let persist = engine.persist.as_ref().unwrap();
            assert_eq!(persist.strategy, FileSpaceStrategy::FsmAggr);
            assert!(predecessor.length() < persist.threshold);
            let region = engine.inspect_group(original_root.get()).unwrap().region;
            let region =
                group::remove_links_from_region(&region, &[LinkNameBuf::new(removed).unwrap()])
                    .unwrap();
            let encoded = object_header::build_v2_object_header_allocation(
                &region,
                Extent::new(
                    engine
                        .addresses()
                        .relative(PhysicalAddress::new(engine.image().len()))
                        .unwrap(),
                    predecessor.length(),
                )
                .unwrap(),
            )
            .unwrap();
            let requested = PhysicalAddress::new(engine.place_object_header(&encoded).unwrap());
            assert!(requested > original_root);
            assert_eq!(
                engine.oh_chunk_spans(requested).unwrap(),
                [FileRange::new(requested, predecessor.length()).unwrap()],
            );
            retired = engine.object_header_free_spans(&spans).unwrap();
            assert_eq!(retired.len(), 1);
            assert_eq!(retired[0].0, *predecessor);
            requested
        }
    };
    let expected_end = match change {
        PublicationChange::Truncate(_) => {
            let bytes = vec![7; 4_096];
            let end = PhysicalAddress::new(engine.append(&bytes).unwrap());
            retired.push((
                FileRange::new(end, u64::try_from(bytes.len()).unwrap()).unwrap(),
                FreeClass::Dead,
            ));
            Some(end)
        }
        PublicationChange::Reclaim => None,
    };
    assert!(retired.iter().all(|(range, _)| !range.overlaps(allocation)));
    let proposed_end = PhysicalAddress::new(engine.image().len());
    let pending = RootRetirement::new(requested_root, vec![])
        .into_pending_frees(retired.clone(), proposed_end.get());
    assert_eq!(pending.batch.entries(), retired);
    let bounds = counts
        .before_publication(&mut engine, pending)
        .unwrap()
        .at_end(proposed_end)
        .unwrap_or_else(|failure| panic!("{:?}", failure.error));
    let (final_root, confirmed_end) = bounds
        .with_publication(BTreeMap::new(), |scope| {
            let scope = scope.unwrap();
            let confirmed_root = scope
                .engine
                .addresses()
                .resolve(scope.engine.superblock().root_group_address)
                .unwrap();
            assert_eq!(scope.root, confirmed_root);
            assert_eq!(scope.end.get(), scope.engine.superblock().eof_address);
            assert_eq!(scope.end.get(), scope.engine.image().len());
            assert_eq!(scope.proposed_end, proposed_end);
            assert_eq!(scope.counts.root().unwrap().header_address(), original_root);
            assert_eq!(scope.counts.updates().len(), 1);
            assert_eq!(scope.counts.updates()[0].object(), object);
            assert_eq!(scope.counts.updates()[0].allocation().range(), allocation);
            assert_eq!(scope.counts.updates()[0].field(), field);
            assert!(allocation.end() <= scope.end);
            assert_eq!(
                scope
                    .engine
                    .image()
                    .read_exact_at(
                        allocation.start().get(),
                        usize::try_from(allocation.length()).unwrap(),
                    )
                    .unwrap(),
                original.as_slice()
            );
            match change {
                PublicationChange::Truncate(_) => {
                    assert_eq!(scope.root, requested_root);
                    assert_eq!(Some(scope.end), expected_end);
                    assert!(scope.end < scope.proposed_end);
                    assert_eq!(
                        scope
                            .engine
                            .inspect_group(scope.root.get())
                            .unwrap()
                            .links
                            .links()
                            .iter()
                            .map(|link| link.name().to_vec())
                            .collect::<Vec<_>>(),
                        [b"b".to_vec(), b"c".to_vec(), b"d".to_vec()],
                    );
                }
                PublicationChange::Reclaim => {
                    assert_ne!(scope.root, requested_root);
                    assert_eq!(scope.root, original_root);
                }
            }
            Ok((scope.root, scope.end))
        })
        .unwrap();
    drop(engine);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), confirmed_end.get());
    let file = File::open(&path).unwrap();
    assert_eq!(file.superblock().root_group_address.get(), final_root.get());
    if matches!(change, PublicationChange::Reclaim) {
        assert_eq!(file.dataset("bb").unwrap().read_i32().unwrap(), [1, 2, 3]);
        assert_eq!(file.root().datasets().unwrap(), ["bb"]);
    }
    drop(file);
}

#[test]
fn a_different_requested_root_does_not_grant_a_count_write_scope() {
    let path = test_util::temp::temp_path("count-wrong-publication.h5");
    protection_tests::fixture_with_base(&path, UserblockSize::NONE);
    let mut engine = edit_test_util::open_engine(&path);
    let original = engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap();
    let bounds = bounds(&mut engine, vec![], &["a"]);
    let expected = bounds.root();
    assert_ne!(expected, original);
    let result = with_unchecked_carrier(bounds, |engine, mut pending, complete| {
        pending.root = original;
        engine.publish_pending_tree_with_completion(pending, BTreeMap::new(), |published| {
            let Err(error) = complete.into_write_scope(published) else {
                panic!("expected another publication root")
            };
            Ok(error)
        })
    })
    .unwrap();
    let CountCompletionError::Root {
        expected: actual_expected,
        actual,
    } = result
    else {
        panic!("expected root mismatch")
    };
    assert_eq!(actual_expected, expected);
    assert_eq!(actual, original);
}

#[test]
fn a_failed_attempt_cannot_use_an_earlier_same_root_confirmation() {
    let path = test_util::temp::temp_path("count-old-confirmation.h5");
    protection_tests::fixture_with_base(&path, UserblockSize::NONE);
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let root = replacement_root(&mut engine, &["a"]);
    engine.publish_nonpersisting(root, None).unwrap();
    assert_eq!(engine.publication, PublicationState::Published);
    let (counts, _) = protection_tests::counts(&engine, &["b"]);
    let counts = counts.protect(&engine, []).unwrap();
    let end = PhysicalAddress::new(engine.image().len());
    let pending = RootRetirement::new(root, vec![]).into_pending_frees(vec![], end.get());
    let bounds = counts
        .before_publication(&mut engine, pending)
        .unwrap()
        .at_end(end)
        .unwrap_or_else(|failure| panic!("{:?}", failure.error));
    let called = Cell::new(false);
    disk_log::start();
    disk_log::fail_sync_after(0, FailurePoint::Before);
    let result = bounds.with_publication(BTreeMap::new(), |_| {
        called.set(true);
        Ok(())
    });
    disk_log::finish();
    let error = result.unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected publication synchronization failure")
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk synchronization failure");
    assert!(!called.get());
    assert_eq!(engine.publication, PublicationState::Published);
}

#[rstest]
#[case::reusable(Claim::Reusable)]
#[case::provisional(Claim::Provisional)]
fn ownership_changes_after_the_original_bounds_cannot_grant_a_write_scope(#[case] claim: Claim) {
    let path = test_util::temp::temp_path("count-changed-claims.h5");
    protection_tests::fixture_with_base(&path, UserblockSize::NONE);
    let mut engine = edit_test_util::open_engine(&path);
    let bounds = bounds(&mut engine, vec![], &["a"]);
    let object = bounds.updates()[0].object();
    let allocation = bounds.updates()[0].allocation().range();
    let reusable = FileRange::new(allocation.start(), 1).unwrap();
    let error = with_unchecked_carrier(bounds, |engine, pending, complete| {
        match claim {
            Claim::Reusable => {
                engine.space = SessionSpace::from_persisted(
                    Some(FileSpaceStrategy::FsmAggr),
                    1,
                    FileSpacePageSize::DEFAULT,
                    [(
                        None,
                        Extent::new(
                            StoredAddress::new(reusable.start().get()),
                            reusable.length(),
                        )
                        .unwrap(),
                    )],
                )
                .unwrap();
            }
            Claim::Provisional => {
                engine.context.allocations.provision(reusable).unwrap();
            }
        }
        engine.publish_pending_tree_with_completion(pending, BTreeMap::new(), |published| {
            let Err(error) = complete.into_write_scope(published) else {
                panic!("expected a changed ownership claim")
            };
            Ok(error)
        })
    })
    .unwrap();
    match (claim, error) {
        (
            Claim::Reusable,
            CountCompletionError::Protection(CountPublicationError::Protection(
                CountProtectionError::Reusable {
                    object: actual_object,
                    allocation: actual_allocation,
                    reusable: actual_reusable,
                },
            )),
        ) => {
            assert_eq!(actual_object, object);
            assert_eq!(actual_allocation, allocation);
            assert_eq!(actual_reusable, reusable);
        }
        (
            Claim::Provisional,
            CountCompletionError::Protection(CountPublicationError::Protection(
                CountProtectionError::Publication(Error::Format(FormatError::Internal(reason))),
            )),
        ) => {
            assert_eq!(
                reason,
                "reference publication precedes allocation ownership publication"
            );
        }
        (_, other) => panic!("unexpected ownership error: {other:?}"),
    }
}

#[test]
fn identical_addresses_in_a_different_session_cannot_grant_a_write_scope() {
    let path = test_util::temp::temp_path("count-publisher-session.h5");
    let other_path = test_util::temp::temp_path("count-publisher-other-session.h5");
    protection_tests::fixture_with_base(&path, UserblockSize::NONE);
    std::fs::copy(&path, &other_path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    let mut other = edit_test_util::open_engine(&other_path);
    let other_root = replacement_root(&mut other, &["a"]);
    let bounds = bounds(&mut engine, vec![], &["a"]);
    assert_eq!(bounds.root(), other_root);
    let error = with_unchecked_carrier(bounds, |_, pending, complete| {
        other.publish_pending_tree_with_completion(pending, BTreeMap::new(), |published| {
            let Err(error) = complete.into_write_scope(published) else {
                panic!("expected another count session")
            };
            Ok(error)
        })
    })
    .unwrap();
    assert!(
        matches!(
            error,
            CountCompletionError::Protection(CountPublicationError::Protection(
                CountProtectionError::Session
            ))
        ),
        "{error:?}"
    );
}

#[rstest]
#[case::empty(&["a", "b"], 0)]
#[case::surviving(&["a"], 1)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn completion_keeps_borrowed_facts_without_reading_source_bytes(
    #[case] names: &[&str],
    #[case] updates: usize,
) {
    let path = test_util::temp::temp_path("count-completion-cost.h5");
    protection_tests::fixture_with_base(&path, UserblockSize::NONE);
    let (mut engine, calls) = protection_tests::engine_with_read_call_counter(&path);
    let mut control = bounds(&mut engine, vec![], &["a"]);
    calls.store(0, Ordering::Relaxed);
    let (prepared, measured) = allocation::measure("count-completion-read-control", || {
        control.prepare_counts().unwrap()
    });
    assert_eq!(prepared.updates().len(), 1);
    assert!(measured.blocks > 0 && measured.live_bytes > 0, "{measured}");
    assert!(calls.load(Ordering::Relaxed) > 0);
    drop(prepared);
    drop(control);
    let bounds = bounds(&mut engine, vec![], names);
    assert_eq!(bounds.updates().len(), updates);
    with_unchecked_carrier(bounds, |engine, pending, complete| {
        engine.publish_pending_tree_with_completion(pending, BTreeMap::new(), |published| {
            calls.store(0, Ordering::Relaxed);
            DISCOVERY_WORK.with(|work| work.set((0, 0)));
            let (scope, measured) = allocation::measure("count-completion-scope", || {
                complete.into_write_scope(published).unwrap()
            });
            assert_eq!(scope.counts.updates().len(), updates);
            assert_eq!(measured.blocks, 0, "{measured}");
            assert_eq!(measured.live_bytes, 0, "{measured}");
            assert_eq!(calls.load(Ordering::Relaxed), 0);
            DISCOVERY_WORK.with(|work| assert_eq!(work.get(), (0, 0)));
            Ok(())
        })
    })
    .unwrap();
}

#[derive(Clone, Copy)]
enum Claim {
    Provisional,
    Reusable,
}

#[derive(Clone, Copy)]
enum PublicationChange {
    Reclaim,
    Truncate(CountStorage),
}

fn reclamation_fixture(path: &Path) {
    std::fs::write(path, identity::hard_link_undo()).unwrap();
    let engine = edit_test_util::open_engine(path);
    let extension = engine
        .addresses()
        .resolve(engine.superblock().superblock_extension_address.unwrap())
        .unwrap();
    drop(engine);
    file_space_info::replace_message(
        path,
        extension.get(),
        &file_space_info::version_0(
            Widths::EIGHT,
            file_space_info::ALL_PERSIST,
            u64::MAX,
            &[u64::MAX; file_space_info::VERSION_0_MANAGERS],
        ),
    );
}

fn with_unchecked_carrier<T>(
    bounds: CountPublicationBounds<'_>,
    apply: impl for<'scope> FnOnce(
        &'scope mut WriteEngine,
        PendingRootFrees,
        CountCompletion<'scope>,
    ) -> T,
) -> T {
    let CountPublicationBounds { phase, end } = bounds;
    let CountPublicationPhase {
        engine,
        counts,
        retirement,
    } = phase;
    let request = CountCompletion {
        counts: &counts,
        root: retirement.root,
        proposed_end: end,
    };
    apply(engine, retirement, request)
}

fn bounds<'engine>(
    engine: &'engine mut WriteEngine,
    retired: Vec<(FileRange, FreeClass)>,
    names: &[&str],
) -> CountPublicationBounds<'engine> {
    let (counts, _) = protection_tests::counts(engine, names);
    let counts = counts.protect(engine, []).unwrap();
    let root = replacement_root(engine, names);
    let end = PhysicalAddress::new(engine.image().len());
    let pending =
        RootRetirement::new(root, vec![]).into_pending_frees(retired, engine.image().len());
    counts
        .before_publication(engine, pending)
        .unwrap()
        .at_end(end)
        .unwrap_or_else(|failure| panic!("{:?}", failure.error))
}

fn replacement_root(engine: &mut WriteEngine, names: &[&str]) -> PhysicalAddress {
    let root = engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap();
    let region = engine.inspect_group(root.get()).unwrap().region;
    let names = names
        .iter()
        .map(|name| LinkNameBuf::new(name).unwrap())
        .collect::<Vec<_>>();
    let region = group::remove_links_from_region(&region, &names).unwrap();
    let encoded = object_header::build_v2_object_header(&region).unwrap();
    PhysicalAddress::new(engine.place_object_header(&encoded).unwrap())
}
