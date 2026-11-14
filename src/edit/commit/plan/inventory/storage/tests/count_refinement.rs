use rstest::rstest;

use super::super::OriginalStorage;
use super::super::SpaceClaim;
use super::super::StorageAllocation;
use super::super::StorageConflict;
use super::super::StorageOrigin;
use crate::PhysicalAddress;
use crate::edit::allocation::AllocationPublication;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::CountProtectionError;
use crate::edit::analysis::ownership::AllocationKind;
use crate::edit::commit::plan::CommitPlan;
use crate::edit::commit::plan::PreparedTreeCounts;
use crate::edit::commit::plan::inventory::tests as inventory_tests;
use crate::edit::commit::plan::inventory::tests::Target;
use crate::edit::test_util as edit_test_util;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::exact(Coverage::Exact)]
#[case::prefix(Coverage::Prefix)]
#[case::suffix(Coverage::Suffix)]
#[case::interior(Coverage::Interior)]
#[case::split(Coverage::Split)]
#[case::disjoint(Coverage::Disjoint)]
fn checked_count_headers_refine_only_intersecting_unobserved_intervals(#[case] coverage: Coverage) {
    let path = test_util::temp::temp_path("count-refinement.h5");
    let object = inventory_tests::fixture(&path, Target::Named(2));
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("type")).unwrap();
    let prepared = inventory_tests::prepare(&mut engine);
    let CommitPlan::Tree(tree) = &prepared.prepared.plan else {
        panic!("expected a tree");
    };
    let Some(PreparedTreeCounts::Complete { protected, .. }) = &tree.counts else {
        panic!("expected checked counts");
    };
    let header = protected.updates()[0].allocation().range();
    let start = header.start().get();
    let end = header.end().get();
    let middle = start + header.length() / 2;
    let (original, prefix, suffix) = match coverage {
        Coverage::Exact => (vec![(start, end)], None, None),
        Coverage::Prefix => (vec![(start - 3, middle)], Some((start - 3, start)), None),
        Coverage::Suffix => (vec![(middle, end + 5)], None, Some((end, end + 5))),
        Coverage::Interior => (
            vec![(start - 3, end + 5)],
            Some((start - 3, start)),
            Some((end, end + 5)),
        ),
        Coverage::Split => (
            vec![(start - 3, middle - 1), (middle + 1, end + 5)],
            Some((start - 3, start)),
            Some((end, end + 5)),
        ),
        Coverage::Disjoint => (
            vec![(start - 3, start), (end, end + 5)],
            Some((start - 3, start)),
            Some((end, end + 5)),
        ),
    };
    let mut storage = OriginalStorage::from_test_ranges(&[]);
    storage.source = Some(prepared.dispositions().source().as_ref().unwrap().clone());
    storage.allocations = original.into_iter().map(unobserved).collect();
    storage.include_counts(&engine, protected).unwrap();
    let expected = prefix
        .into_iter()
        .map(unobserved)
        .chain([StorageAllocation {
            range: header,
            origin: StorageOrigin::Owner {
                object,
                kind: AllocationKind::Header,
            },
        }])
        .chain(suffix.into_iter().map(unobserved))
        .collect::<Vec<_>>();
    assert_eq!(storage.allocations(), expected);
    let pointer = storage.allocations().as_ptr();
    storage.include_counts(&engine, protected).unwrap();
    assert_eq!(storage.allocations(), expected);
    assert_eq!(storage.allocations().as_ptr(), pointer);
    let error = storage.exclude(header, SpaceClaim::RootCopy).unwrap_err();
    assert!(
        matches!(error, StorageConflict::Overlap { allocation, range, claim: SpaceClaim::RootCopy } if allocation.range == header && range == header),
        "{error:?}"
    );
}

#[rstest]
#[case::owner(false)]
#[case::parent(true)]
fn checked_count_headers_reject_known_storage_between_unobserved_intervals(#[case] parent: bool) {
    let path = test_util::temp::temp_path("count-refinement-conflict.h5");
    let object = inventory_tests::fixture(&path, Target::Named(2));
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("type")).unwrap();
    let prepared = inventory_tests::prepare(&mut engine);
    let CommitPlan::Tree(tree) = &prepared.prepared.plan else {
        panic!("expected a tree");
    };
    let Some(PreparedTreeCounts::Complete { protected, .. }) = &tree.counts else {
        panic!("expected checked counts");
    };
    let header = protected.updates()[0].allocation().range();
    let middle = header.start().get() + header.length() / 2;
    let known = StorageAllocation {
        range: super::range(middle, header.end().get() - middle),
        origin: if parent {
            StorageOrigin::Parent {
                address: PhysicalAddress::new(middle),
                class: None,
            }
        } else {
            StorageOrigin::Owner {
                object,
                kind: AllocationKind::Data,
            }
        },
    };
    let original = vec![
        unobserved((header.start().get() - 3, middle)),
        known,
        unobserved((header.end().get(), header.end().get() + 5)),
    ];
    let mut storage = OriginalStorage::from_test_ranges(&[]);
    storage.allocations = original.clone();
    let error = storage.include_counts(&engine, protected).unwrap_err();
    let StorageConflict::Overlap {
        allocation,
        range,
        claim,
    } = error
    else {
        panic!("unexpected refinement error: {error:?}");
    };
    assert_eq!(allocation, known);
    assert_eq!(range, header);
    assert_eq!(claim, SpaceClaim::CountHeader);
    assert_eq!(storage.allocations(), original);
}

#[rstest]
#[case::phase(Association::Phase)]
#[case::root(Association::Root)]
#[case::session(Association::Session)]
fn count_refinement_rejects_incompatible_source_associations(#[case] association: Association) {
    let path = test_util::temp::temp_path("count-refinement-association.h5");
    let object = inventory_tests::fixture(&path, Target::Named(2));
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("type")).unwrap();
    let prepared = inventory_tests::prepare(&mut engine);
    let CommitPlan::Tree(tree) = &prepared.prepared.plan else {
        panic!("expected a tree");
    };
    let Some(PreparedTreeCounts::Complete { protected, .. }) = &tree.counts else {
        panic!("expected checked counts");
    };
    let root = prepared.dispositions().source().as_ref().unwrap();
    let mut storage = OriginalStorage::from_counts(protected);
    storage.source = Some(root.clone());
    match association {
        Association::Phase | Association::Root => {
            storage.source = Some(
                AnalysisPhase::new(&engine)
                    .reconstruction_source(
                        &engine,
                        if matches!(association, Association::Root) {
                            object
                        } else {
                            root.object()
                        },
                    )
                    .unwrap(),
            );
        }
        Association::Session => engine.context.allocations = AllocationPublication::new(),
    }
    let before = storage.allocations().to_vec();
    let error = storage.include_counts(&engine, protected).unwrap_err();
    match (association, &error) {
        (
            Association::Phase,
            StorageConflict::CountAssociation(CountProtectionError::Phase { object }),
        ) => {
            assert_eq!(*object, root.object());
        }
        (
            Association::Root,
            StorageConflict::CountAssociation(CountProtectionError::Root { expected, actual }),
        ) => {
            assert_eq!(*expected, root.object());
            assert_eq!(*actual, object);
        }
        (
            Association::Session,
            StorageConflict::CountAssociation(CountProtectionError::Session),
        ) => {}
        _ => panic!("unexpected association error: {error:?}"),
    }
    assert_eq!(storage.allocations(), before);
}

fn unobserved((start, end): (u64, u64)) -> StorageAllocation {
    StorageAllocation {
        range: super::range(start, end - start),
        origin: StorageOrigin::Unobserved,
    }
}

#[derive(Clone, Copy, Debug)]
enum Association {
    Phase,
    Root,
    Session,
}

#[derive(Clone, Copy, Debug)]
enum Coverage {
    Disjoint,
    Exact,
    Interior,
    Prefix,
    Split,
    Suffix,
}
