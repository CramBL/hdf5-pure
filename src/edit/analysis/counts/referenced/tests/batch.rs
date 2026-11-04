#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use core::num::NonZeroU64;
use std::sync::atomic::Ordering;

use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::object_header::Message;
use test_util_hdf5::global_heap_fixture::Fixture;

use crate::PhysicalAddress;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::counts::CountContradiction;
use crate::edit::analysis::counts::CountEvidenceError;
use crate::edit::analysis::counts::CountObservation;
use crate::edit::analysis::counts::added_users::CopyUsersUnion;
use crate::error::FormatError;
use crate::object_graph::ObjectId;

use super::{CopyUsersError, ReferencedCountError};

#[rstest]
#[case::one(1)]
#[case::many(16)]
fn referenced_batches_retain_ordered_original_counts_and_added_multiplicity(#[case] size: u32) {
    let path = test_util::temp::temp_path("referenced-count-batch.h5");
    let mut fixture = Fixture::new();
    let objects = (0..size)
        .map(|index| {
            ObjectId::new(PhysicalAddress::new(
                fixture.append_header(vec![Message::reference_count(index + 7)]),
            ))
        })
        .collect::<Vec<_>>();
    fixture.write(&path);
    let (engine, calls, _) = super::observed(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let mut union = CopyUsersUnion::new();
    for &object in objects.iter().rev() {
        let tree = super::tree(object);
        let copy = phase.copy_users(&tree).unwrap().unwrap();
        union = union.add(&phase, Some(&copy)).unwrap().unwrap();
        union = union.add(&phase, Some(&copy)).unwrap().unwrap();
    }
    let added = union.finish();
    let batch = phase.referenced_counts(&added).unwrap().unwrap();
    assert!(batch.matches(&phase));
    assert!(!batch.matches(&AnalysisPhase::new(&engine)));
    assert_eq!(
        batch.root(),
        Some(ObjectId::new(PhysicalAddress::new(Fixture::ROOT)))
    );
    assert_eq!(
        batch
            .targets()
            .map(|(object, _)| object)
            .collect::<Vec<_>>(),
        objects
    );
    for (index, (object, target)) in batch.targets().enumerate() {
        assert_eq!(target.count().object(), object);
        assert_eq!(target.count().original(), u32::try_from(index).unwrap() + 7);
        assert_eq!(target.count().incoming(), None);
        assert_eq!(target.count().minimum(), None);
        let field = target.count().field().unwrap();
        assert_eq!(field.value(), target.count().original());
        assert_eq!(
            target.count().header_allocation().unwrap().range(),
            field.chunk()
        );
        assert_eq!(target.added().get(), 2);
        assert_eq!(batch.get(object).unwrap().added(), target.added());
    }
    assert!(
        batch
            .get(ObjectId::new(PhysicalAddress::new(u64::MAX)))
            .is_none()
    );
    assert!(phase.topology.is_none());
    assert_eq!(phase.counts.len(), objects.len());
    let reads = calls.load(Ordering::Relaxed);
    assert!(reads > 0);
    let repeated = phase.referenced_counts(&added).unwrap().unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), reads);
    assert_eq!(
        repeated
            .into_targets()
            .map(|(object, target)| (object, target.added().get()))
            .collect::<Vec<_>>(),
        objects
            .iter()
            .map(|&object| (object, 2))
            .collect::<Vec<_>>()
    );
}

#[rstest]
#[case::ambiguous(false)]
#[case::source_failure(true)]
fn a_missing_later_observation_drops_the_batch_prefix(#[case] source_failure: bool) {
    let path = test_util::temp::temp_path("incomplete-reference-batch.h5");
    let mut fixture = Fixture::new();
    let first = fixture.append_header(vec![Message::reference_count(7)]);
    let second = fixture.append_header(vec![
        Message::reference_count(8),
        Message::reference_count(9),
    ]);
    let third = fixture.append_header(vec![Message::reference_count(10)]);
    fixture.write(&path);
    let (engine, _, _) = super::observed(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let objects =
        [first, second, third].map(|address| ObjectId::new(PhysicalAddress::new(address)));
    let mut union = CopyUsersUnion::new();
    for object in objects {
        let tree = super::tree(object);
        let copy = phase.copy_users(&tree).unwrap().unwrap();
        union = union.add(&phase, Some(&copy)).unwrap().unwrap();
    }
    let added = union.finish();
    let failure = FormatError::UnexpectedSourceEof {
        expected: second + 1,
        available: second,
    };
    if source_failure {
        phase.counts.insert(objects[1], Err(failure.clone()));
    }
    let result = phase.referenced_counts(&added);
    if source_failure {
        let Err(error) = result else {
            panic!("expected the cached source error")
        };
        assert_eq!(
            error,
            ReferencedCountError::Evidence(CountEvidenceError::Source(failure))
        );
    } else {
        assert!(result.unwrap().is_none());
    }
    assert_eq!(
        phase.counts.keys().copied().collect::<Vec<_>>(),
        objects[..2]
    );
    assert!(phase.topology.is_none());
}

#[test]
fn batch_enrichment_reuses_cached_counts_and_preserves_original_observations() {
    let path = test_util::temp::temp_path("enriched-reference-batch.h5");
    let mut fixture = Fixture::new();
    let object = ObjectId::new(PhysicalAddress::new(
        fixture.append_header(vec![Message::reference_count(1)]),
    ));
    fixture.link("first", object.header_address().get());
    fixture.link("second", object.header_address().get());
    fixture.write(&path);
    let (engine, calls, _) = super::observed(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let tree = super::tree(object);
    let added = super::added(&mut phase, &tree);
    let before = phase.referenced_counts(&added).unwrap().unwrap();
    assert_eq!(before.get(object).unwrap().count().minimum(), None);
    assert!(phase.topology().is_some());
    calls.store(0, Ordering::Relaxed);
    let Err(error) = phase.referenced_counts(&added) else {
        panic!("expected the demonstrated original undercount")
    };
    assert_eq!(
        error,
        ReferencedCountError::Evidence(CountEvidenceError::Contradiction(CountContradiction {
            object,
            observed: 1,
            minimum: 2
        }))
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(before.get(object).unwrap().count().original(), 1);
}

#[rstest]
#[case::complete_allocation(false)]
#[case::unknown_allocation(true)]
fn batch_enrichment_preserves_counts_fields_and_optional_allocations(#[case] partial: bool) {
    let path = test_util::temp::temp_path("reference-batch-facts.h5");
    let mut fixture = Fixture::new();
    let object = ObjectId::new(PhysicalAddress::new(
        fixture.append_header(vec![Message::reference_count(7)]),
    ));
    fixture.write(&path);
    let (engine, calls, _) = super::observed(&path);
    let mut phase = AnalysisPhase::new(&engine);
    if partial {
        let count = phase.object_reference_count(object).unwrap();
        phase.counts.insert(
            object,
            Ok(CountObservation {
                count,
                allocation: None,
            }),
        );
    }
    let tree = super::tree(object);
    let added = super::added(&mut phase, &tree);
    let before = phase.referenced_counts(&added).unwrap().unwrap();
    assert_eq!(before.get(object).unwrap().count().incoming(), None);
    assert!(phase.topology().is_some());
    calls.store(0, Ordering::Relaxed);
    let after = phase.referenced_counts(&added).unwrap().unwrap();
    let fact = after.get(object).unwrap().count();
    assert_eq!(fact.original(), 7);
    assert_eq!(fact.incoming(), Some(0));
    assert_eq!(fact.minimum(), None);
    assert_eq!(fact.field(), before.get(object).unwrap().count().field());
    assert_eq!(fact.header_allocation().is_none(), partial);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[rstest]
#[case::engine(Association::Engine)]
#[case::phase(Association::Phase)]
#[case::root(Association::Root)]
fn batch_association_failures_do_not_read_source_counts(#[case] context: Association) {
    let path = test_util::temp::temp_path("reference-batch-association.h5");
    let other_path = test_util::temp::temp_path("other-reference-batch-association.h5");
    Fixture::new().write(&path);
    Fixture::new().write(&other_path);
    let (engine, calls, _) = super::observed(&path);
    let (other_engine, other_calls, _) = super::observed(&other_path);
    let object = ObjectId::new(PhysicalAddress::new(Fixture::ROOT));
    let tree = super::tree(object);
    let mut first = AnalysisPhase::new(&engine);
    let added = super::added(&mut first, &tree);
    let mut other = if matches!(context, Association::Engine) {
        AnalysisPhase::new(&other_engine)
    } else {
        AnalysisPhase::new(&engine)
    };
    if matches!(context, Association::Root) {
        other.root = Some(ObjectId::new(PhysicalAddress::new(Fixture::ROOT + 1)));
    }
    let Err(error) = other.referenced_counts(&added) else {
        panic!("expected the incompatible batch association")
    };
    assert_eq!(
        error,
        ReferencedCountError::Association(match context {
            Association::Engine => CopyUsersError::Engine,
            Association::Phase => CopyUsersError::Phase,
            Association::Root => CopyUsersError::Root {
                expected: ObjectId::new(PhysicalAddress::new(Fixture::ROOT + 1)),
                actual: object
            },
        })
    );
    assert!(other.counts.is_empty() && other.topology.is_none());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(other_calls.load(Ordering::Relaxed), 0);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn empty_batches_allocate_and_read_nothing() {
    let path = test_util::temp::temp_path("empty-reference-batch.h5");
    Fixture::new().write(&path);
    let (engine, calls, _) = super::observed(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let added = phase.added_copy_users([]).unwrap().unwrap();
    let (batch, measured) = allocation::measure("empty_referenced_counts", || {
        phase.referenced_counts(&added)
    });
    let batch = batch.unwrap().unwrap();
    assert_eq!(batch.targets().count(), 0);
    assert_eq!(batch.root(), None);
    assert!(batch.matches(&phase));
    assert!(phase.topology.is_none() && phase.identity.is_none() && phase.counts.is_empty());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let object = ObjectId::new(PhysicalAddress::new(Fixture::ROOT));
    let tree = super::tree(object);
    let added = super::added(&mut phase, &tree);
    let (positive, control) = allocation::measure("nonempty_referenced_counts", || {
        phase.referenced_counts(&added)
    });
    let positive = positive.unwrap().unwrap();
    assert_eq!(positive.get(object).unwrap().added(), NonZeroU64::MIN);
    assert!(control.blocks > 0 && control.bytes > 0, "{control}");
    assert!(calls.load(Ordering::Relaxed) > 0);
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(measured.bytes, 0, "{measured}");
}

#[derive(Clone, Copy)]
enum Association {
    Engine,
    Phase,
    Root,
}
