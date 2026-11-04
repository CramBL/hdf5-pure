use core::num::NonZeroU64;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use hdf5_pure_format::__private::ObjectHeaderPrefix;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::datatype;
use test_util::object_header::{Message, MessageFlags, MessageType};
use test_util::object_header::{v1, v2};
use test_util_hdf5::global_heap_fixture::Fixture;

use crate::edit::WriteEngine;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::counts::{CountContradiction, CountEvidenceError, CountObservation};
use crate::edit::attributes::DenseAttrSet;
use crate::edit::copy::{CopyNode, CopyTree, CopyTreeBuilder};
use crate::edit::object_header::OhRegion;
use crate::error::FormatError;
use crate::image::{CountingImage, HandleImage};
use crate::object_graph::ObjectId;
use crate::source::MetadataCacheConfig;
use crate::{FileLocking, PhysicalAddress};

use super::{
    super::added_users::{AddedCopyUsers, CopyUsersError},
    ReferencedCountError,
};

mod batch;

#[rstest]
#[case::implicit(None, false)]
#[case::unproven_zero(Some(0), false)]
#[case::excess(Some(u32::MAX), false)]
#[case::outside_complete(Some(7), true)]
fn referenced_observations_preserve_counts_without_original_user_minima(
    #[case] original: Option<u32>,
    #[case] topology: bool,
) {
    let path = test_util::temp::temp_path("referenced-count.h5");
    let mut fixture = Fixture::new();
    let object = ObjectId::new(PhysicalAddress::new(
        fixture.append_header(original.map(Message::reference_count).into_iter().collect()),
    ));
    fixture.write(&path);
    let (engine, calls, bytes) = observed(&path);
    let tree = tree(object);
    let mut phase = AnalysisPhase::new(&engine);
    if topology {
        assert!(phase.topology().is_some());
    }
    let added = added(&mut phase, &tree);
    calls.store(0, Ordering::Relaxed);
    let fact = phase.referenced_count(&added, object).unwrap().unwrap();
    assert!(fact.matches(&phase));
    assert_eq!(fact.object(), object);
    assert_eq!(
        fact.root(),
        ObjectId::new(PhysicalAddress::new(Fixture::ROOT))
    );
    assert_eq!(fact.original(), original.unwrap_or(1));
    assert_eq!(fact.incoming(), topology.then_some(0));
    assert_eq!(fact.minimum(), None);
    assert_eq!(fact.field().map(|field| field.value()), original);
    assert_eq!(fact.header_allocation().is_some(), original.is_some());
    assert!(calls.load(Ordering::Relaxed) > 0 && bytes.load(Ordering::Relaxed) > 0);
    let reads = calls.load(Ordering::Relaxed);
    let again = phase.referenced_count(&added, object).unwrap().unwrap();
    assert_eq!(again.field(), fact.field());
    assert_eq!(calls.load(Ordering::Relaxed), reads);
    assert_eq!(
        phase.object_reference_count(object).unwrap(),
        phase.counts[&object].as_ref().unwrap().into_parts().0
    );
    assert_eq!(calls.load(Ordering::Relaxed), reads);
    assert_eq!(phase.topology.is_some(), topology);
}

#[test]
fn discovering_original_links_refines_consistency_without_rereading_the_count() {
    let path = test_util::temp::temp_path("referenced-minimum.h5");
    let mut fixture = Fixture::new();
    let object = ObjectId::new(PhysicalAddress::new(
        fixture.append_header(vec![Message::reference_count(1)]),
    ));
    fixture.link("a", object.header_address().get());
    fixture.link("b", object.header_address().get());
    fixture.write(&path);
    let (engine, calls, _) = observed(&path);
    let tree = tree(object);
    let mut phase = AnalysisPhase::new(&engine);
    let added = added(&mut phase, &tree);
    let before = phase.referenced_count(&added, object).unwrap().unwrap();
    assert_eq!(before.incoming(), None);
    assert_eq!(before.minimum(), None);
    assert!(phase.topology().is_some());
    calls.store(0, Ordering::Relaxed);
    let Err(error) = phase.referenced_count(&added, object) else {
        panic!("expected the demonstrated undercount");
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
    assert_eq!(before.original(), 1);
}

#[rstest]
#[case::without_topology(false)]
#[case::known_self_link(true)]
fn referenced_root_counts_include_only_proven_original_references(#[case] topology: bool) {
    let path = test_util::temp::temp_path("referenced-root.h5");
    let mut fixture = Fixture::new();
    fixture.link("self", Fixture::ROOT);
    fixture.write(&path);
    let (engine, _, _) = observed(&path);
    let root = ObjectId::new(PhysicalAddress::new(Fixture::ROOT));
    let tree = tree(root);
    let mut phase = AnalysisPhase::new(&engine);
    if topology {
        assert!(phase.topology().is_some());
    }
    let added = added(&mut phase, &tree);
    let result = phase.referenced_count(&added, root);
    if topology {
        let Err(error) = result else {
            panic!("expected a root undercount");
        };
        assert_eq!(
            error,
            ReferencedCountError::Evidence(CountEvidenceError::Contradiction(CountContradiction {
                object: root,
                observed: 1,
                minimum: 2
            }))
        );
    } else {
        let fact = result.unwrap().unwrap();
        assert_eq!(fact.original(), 1);
        assert_eq!(fact.incoming(), None);
        assert_eq!(fact.minimum(), Some(NonZeroU64::MIN));
        assert!(phase.topology.is_none());
    }
}

#[rstest]
#[case::ambiguous(vec![Message::reference_count(2), Message::reference_count(3)])]
#[case::unsupported(vec![Message::new(MessageType::OBJECT_REFERENCE_COUNT, &[1, 2, 0, 0, 0])])]
fn ambiguous_or_unsupported_referenced_counts_are_cached(#[case] records: Vec<Message>) {
    let path = test_util::temp::temp_path("unsupported-referenced-count.h5");
    let mut fixture = Fixture::new();
    let object = ObjectId::new(PhysicalAddress::new(fixture.append_header(records)));
    fixture.write(&path);
    let (engine, calls, _) = observed(&path);
    let tree = tree(object);
    let mut phase = AnalysisPhase::new(&engine);
    let added = added(&mut phase, &tree);
    assert!(phase.referenced_count(&added, object).unwrap().is_none());
    let reads = calls.load(Ordering::Relaxed);
    assert!(reads > 0);
    assert!(phase.referenced_count(&added, object).unwrap().is_none());
    assert_eq!(calls.load(Ordering::Relaxed), reads);
}

#[test]
fn referenced_counts_reuse_source_failures_and_partial_allocation_facts() {
    let path = test_util::temp::temp_path("cached-referenced-count.h5");
    let mut fixture = Fixture::new();
    let object = ObjectId::new(PhysicalAddress::new(
        fixture.append_header(vec![Message::reference_count(7)]),
    ));
    fixture.write(&path);
    let (engine, calls, _) = observed(&path);
    let tree = tree(object);
    let mut phase = AnalysisPhase::new(&engine);
    let added = added(&mut phase, &tree);
    let count = phase.object_reference_count(object).unwrap();
    phase.counts.insert(
        object,
        Ok(CountObservation {
            count,
            allocation: None,
        }),
    );
    calls.store(0, Ordering::Relaxed);
    let fact = phase.referenced_count(&added, object).unwrap().unwrap();
    assert_eq!(fact.original(), 7);
    assert_eq!(fact.header_allocation(), None);
    assert_eq!(fact.field().unwrap().value(), 7);
    let failure = FormatError::UnexpectedSourceEof {
        expected: object.header_address().get() + 1,
        available: object.header_address().get(),
    };
    phase.counts.insert(object, Err(failure.clone()));
    let Err(error) = phase.referenced_count(&added, object) else {
        panic!("expected the cached source failure");
    };
    assert_eq!(
        error,
        ReferencedCountError::Evidence(CountEvidenceError::Source(failure))
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[rstest]
#[case::engine(Context::Engine)]
#[case::phase(Context::Phase)]
#[case::root(Context::Root)]
#[case::prospective(Context::Prospective)]
fn original_count_context_mismatches_fail_before_reading(#[case] context: Context) {
    let path = test_util::temp::temp_path("wrong-count-context.h5");
    let other_path = test_util::temp::temp_path("other-count-context.h5");
    let mut fixture = Fixture::new();
    let object = ObjectId::new(PhysicalAddress::new(
        fixture.append_header(vec![Message::reference_count(7)]),
    ));
    fixture.link("target", object.header_address().get());
    fixture.write(&path);
    std::fs::copy(&path, &other_path).unwrap();
    let (engine, calls, bytes) = observed(&path);
    let (other, _, _) = observed(&other_path);
    let tree = tree(object);
    let mut phase = if matches!(context, Context::Prospective) {
        AnalysisPhase::prospective(&engine, object.header_address())
    } else {
        AnalysisPhase::new(&engine)
    };
    let added = added(&mut phase, &tree);
    if !matches!(context, Context::Prospective) {
        let mut foreign = match context {
            Context::Engine => AnalysisPhase::new(&other),
            Context::Root => AnalysisPhase::prospective(&engine, object.header_address()),
            Context::Phase | Context::Prospective => AnalysisPhase::new(&engine),
        };
        assert!(foreign.topology().is_some());
        if matches!(context, Context::Root) {
            foreign
                .topology
                .as_mut()
                .unwrap()
                .as_mut()
                .unwrap()
                .identity = Arc::clone(phase.identity.as_ref().unwrap());
        }
        phase.topology = foreign.topology.take();
    }
    calls.store(0, Ordering::Relaxed);
    bytes.store(0, Ordering::Relaxed);
    let Err(error) = phase.referenced_count(&added, object) else {
        panic!("expected a mismatched source-count context");
    };
    assert_eq!(
        error,
        ReferencedCountError::Association(match context {
            Context::Engine => CopyUsersError::Engine,
            Context::Phase => CopyUsersError::Phase,
            Context::Root | Context::Prospective => CopyUsersError::Root {
                expected: ObjectId::new(PhysicalAddress::new(Fixture::ROOT)),
                actual: object
            },
        })
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(bytes.load(Ordering::Relaxed), 0);
    assert!(phase.counts.is_empty());
}

#[test]
fn a_matching_numeric_root_does_not_transfer_added_facts_between_phases() {
    let path = test_util::temp::temp_path("foreign-added-count.h5");
    let mut fixture = Fixture::new();
    let object = ObjectId::new(PhysicalAddress::new(
        fixture.append_header(vec![Message::reference_count(7)]),
    ));
    fixture.write(&path);
    let (engine, calls, _) = observed(&path);
    let tree = tree(object);
    let mut original = AnalysisPhase::new(&engine);
    let added = added(&mut original, &tree);
    let mut other = AnalysisPhase::new(&engine);
    let Err(error) = other.referenced_count(&added, object) else {
        panic!("expected another phase's added facts to be rejected");
    };
    assert_eq!(
        error,
        ReferencedCountError::Association(CopyUsersError::Phase)
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(other.counts.is_empty());
}

#[test]
fn legacy_count_fields_keep_their_complete_allocation() {
    let path = test_util::temp::temp_path("legacy-referenced-count.h5");
    let mut fixture = Fixture::new();
    let legacy = v1::Header::new().reference_count(3).build();
    let address = fixture.collection(&[]);
    let object = ObjectId::new(PhysicalAddress::new(address));
    fixture.write(&path);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.resize(usize::try_from(address).unwrap() + legacy.len(), 0);
    bytes[usize::try_from(address).unwrap()..].copy_from_slice(&legacy);
    std::fs::write(&path, bytes).unwrap();
    let (engine, _, _) = observed(&path);
    let tree = tree(object);
    let mut phase = AnalysisPhase::new(&engine);
    let added = added(&mut phase, &tree);
    let fact = phase.referenced_count(&added, object).unwrap().unwrap();
    assert_eq!(fact.original(), 3);
    assert_eq!(
        fact.field().unwrap().chunk(),
        fact.header_allocation().unwrap().range()
    );
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn empty_references_do_no_count_or_heap_work() {
    let path = test_util::temp::temp_path("empty-referenced-count.h5");
    Fixture::new().write(&path);
    let (engine, calls, bytes) = observed(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let empty = phase.added_copy_users([]).unwrap().unwrap();
    let object = ObjectId::new(PhysicalAddress::new(Fixture::ROOT));
    let (result, measured) = allocation::measure("empty_referenced_count", || {
        phase.referenced_count(&empty, object)
    });
    assert!(result.unwrap().is_none());
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(measured.bytes, 0, "{measured}");
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(bytes.load(Ordering::Relaxed), 0);
    assert!(phase.identity.is_none() && phase.topology.is_none() && phase.counts.is_empty());
    let tree = tree(object);
    let added = added(&mut phase, &tree);
    let absent = ObjectId::new(PhysicalAddress::new(u64::MAX));
    assert!(phase.referenced_count(&added, absent).unwrap().is_none());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let (control, positive) = allocation::measure("nonempty_referenced_count", || {
        phase.referenced_count(&added, object)
    });
    assert_eq!(control.unwrap().unwrap().original(), 1);
    assert!(positive.blocks > 0 && positive.bytes > 0, "{positive}");
    assert!(calls.load(Ordering::Relaxed) > 0);
}

#[derive(Clone, Copy)]
enum Context {
    Engine,
    Phase,
    Prospective,
    Root,
}

fn tree(object: ObjectId) -> CopyTree {
    let mut message = Message::new(
        MessageType::DATATYPE,
        &datatype::committed_reference(object.header_address().get()),
    );
    message.flags = MessageFlags::SHARED;
    CopyTreeBuilder::default().finish(CopyNode::DatasetVerbatim {
        region: OhRegion::new(
            crate::file_writer::WIDTHS,
            v2::message_records(&[message], v2::HeaderFlags::default()),
            ObjectHeaderPrefix::PLAIN,
        ),
        dense_attrs: DenseAttrSet::default(),
    })
}

fn added<'source>(phase: &mut AnalysisPhase<'source>, tree: &CopyTree) -> AddedCopyUsers<'source> {
    let users = phase.copy_users(tree).unwrap().unwrap();
    phase.added_copy_users([Some(&users)]).unwrap().unwrap()
}

fn observed(path: &Path) -> (WriteEngine, Arc<AtomicU64>, Arc<AtomicU64>) {
    let calls = Arc::new(AtomicU64::new(0));
    let bytes = Arc::new(AtomicU64::new(0));
    let engine = WriteEngine::open_imaged(path, Some(FileLocking::Enabled), |handle, len| {
        Ok(Box::new(
            CountingImage::new(
                Box::new(HandleImage::new(
                    handle,
                    len,
                    MetadataCacheConfig::disabled(),
                )),
                Arc::clone(&bytes),
                Arc::default(),
            )
            .with_read_calls(Arc::clone(&calls)),
        ))
    })
    .unwrap();
    calls.store(0, Ordering::Relaxed);
    bytes.store(0, Ordering::Relaxed);
    (engine, calls, bytes)
}
