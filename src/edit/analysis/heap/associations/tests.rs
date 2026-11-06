use core::cell::Cell;
use core::num::NonZeroU16;
use std::sync::atomic::Ordering;

use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::global_heap::Object;
use test_util::object_header::{Message, MessageType};
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use super::MemberPresence;
use crate::edit::analysis::{
    AnalysisPhase, COLLECTION_ALLOCATIONS, COLLECTION_WORK, DISCOVERY_WORK,
};
use crate::edit::test_util as edit_test_util;
use crate::error::{Error, FormatError};
use crate::object_graph::ObjectId;

#[rstest]
#[case::cold(false)]
#[case::enriched(true)]
fn member_associations_preserve_referrers_and_distinct_member_outcomes(#[case] enriched: bool) {
    let path = test_util::temp::temp_path("member-associations.h5");
    let mut fixture = Fixture::new();
    let address = fixture.collection(&[Object::new(7, b"shared07"), Object::new(1, b"absent01")]);
    let first = object(&mut fixture, "first", address, 7);
    fixture.link("alias", first.header_address().get());
    let second = object(&mut fixture, "second", address, 7);
    let missing = object(&mut fixture, "missing", address, 9);
    let outside_collection = fixture.collection(&[Object::new(1, b"outside1")]);
    let records = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(7, outside_collection, 1),
    );
    let outside = fixture.append_header(records);
    fixture.write(&path);
    let (engine, calls) = edit_test_util::open_counted_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase
        .reconstruction_source(&engine, id(Fixture::ROOT))
        .unwrap();
    let outside_source = phase.reconstruction_source(&engine, id(outside)).unwrap();
    phase.heap_reference_facts(&outside_source).unwrap();
    phase
        .heap_members(&outside_source, PhysicalAddress::new(outside_collection))
        .unwrap();
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    COLLECTION_ALLOCATIONS.with(|work| work.set(0));
    if enriched {
        phase.physical_claims(&source).unwrap();
    }
    let before = calls.load(Ordering::Relaxed);
    assert!(phase.associations.is_none());
    let facts = phase.member_associations(&source).unwrap();
    assert_eq!(facts.root(), Some(id(Fixture::ROOT)));
    assert!(facts.matches_phase(&source));
    assert!(facts.topology_available());
    assert!(facts.incomplete_references().is_empty());
    assert_eq!(facts.collections().len(), 1);
    let collection = &facts.collections()[&PhysicalAddress::new(address)];
    assert!(collection.error().is_none());
    assert_eq!(collection.unmatched(), [index(1)]);
    assert_eq!(collection.observed().len(), 2);
    assert_eq!(
        collection.observed()[&index(7)].referrers(),
        [first, second]
    );
    assert_eq!(
        collection.observed()[&index(7)].presence(),
        MemberPresence::Allocated
    );
    assert_eq!(collection.observed()[&index(9)].referrers(), [missing]);
    assert_eq!(
        collection.observed()[&index(9)].presence(),
        MemberPresence::Missing
    );
    let pointer = core::ptr::from_ref(facts);
    if enriched {
        assert_eq!(calls.load(Ordering::Relaxed), before);
    }
    let after = calls.load(Ordering::Relaxed);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (1, 4));
    assert_eq!(COLLECTION_WORK.with(Cell::get).1, 4);
    assert_eq!(COLLECTION_ALLOCATIONS.with(Cell::get), 1);
    assert_eq!(
        core::ptr::from_ref(phase.member_associations(&source).unwrap()),
        pointer
    );
    assert_eq!(calls.load(Ordering::Relaxed), after);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (1, 4));
    assert_eq!(phase.physical.is_some(), enriched);
    assert_eq!(phase.ownership.is_empty(), !enriched);
    let mut foreign = AnalysisPhase::new(&engine);
    let other = foreign
        .reconstruction_source(&engine, id(Fixture::ROOT))
        .unwrap();
    let Error::Format(error) = phase.member_associations(&other).err().unwrap() else {
        panic!("expected a phase association error");
    };
    assert_eq!(
        error,
        FormatError::Internal("heap query belongs to another source phase".into())
    );
    assert_eq!(calls.load(Ordering::Relaxed), after);
}

#[rstest]
#[case::references(Partial::References)]
#[case::framing(Partial::Framing)]
#[case::topology(Partial::Topology)]
fn partial_associations_keep_reference_and_framing_outcomes_independent(#[case] partial: Partial) {
    let path = test_util::temp::temp_path("member-associations-partial.h5");
    let mut fixture = Fixture::new();
    let address = fixture.collection(&[Object::new(1, b"observed"), Object::new(7, b"unmatched")]);
    let mut records = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(8, address, 1),
    );
    if matches!(partial, Partial::References) {
        records.push(Message::new(MessageType(0x7f), &[]));
    }
    let object = id(fixture.append_header(records));
    fixture.link("data", object.header_address().get());
    fixture.write(&path);
    let (engine, calls) = edit_test_util::open_counted_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase
        .reconstruction_source(&engine, id(Fixture::ROOT))
        .unwrap();
    if matches!(partial, Partial::Topology) {
        phase.topology = Some(None);
    }
    if matches!(partial, Partial::Framing) {
        phase.allocations.insert(
            PhysicalAddress::new(address),
            Err(FormatError::UnexpectedEof {
                expected: 16,
                available: 8,
            }),
        );
    }
    let facts = phase.member_associations(&source).unwrap();
    assert_eq!(
        facts.topology_available(),
        !matches!(partial, Partial::Topology)
    );
    assert_eq!(
        facts.incomplete_references(),
        if matches!(partial, Partial::References) {
            vec![object]
        } else {
            vec![]
        }
    );
    if matches!(partial, Partial::Topology) {
        assert!(facts.collections().is_empty());
    } else {
        let collection = &facts.collections()[&PhysicalAddress::new(address)];
        assert_eq!(collection.observed()[&index(1)].referrers(), [object]);
        if matches!(partial, Partial::Framing) {
            assert_eq!(
                collection.error(),
                Some(&FormatError::UnexpectedEof {
                    expected: 16,
                    available: 8
                })
            );
            assert_eq!(
                collection.observed()[&index(1)].presence(),
                MemberPresence::Unknown
            );
            assert!(collection.unmatched().is_empty());
        } else {
            assert!(collection.error().is_none());
            assert_eq!(
                collection.observed()[&index(1)].presence(),
                MemberPresence::Allocated
            );
            assert_eq!(collection.unmatched(), [index(7)]);
        }
    }
    let pointer = core::ptr::from_ref(facts);
    let before = calls.load(Ordering::Relaxed);
    assert_eq!(
        core::ptr::from_ref(phase.member_associations(&source).unwrap()),
        pointer
    );
    assert_eq!(calls.load(Ordering::Relaxed), before);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::numeric(false)]
#[case::variable(true)]
fn association_queries_bound_retained_state_and_reuse_cached_payload_facts(#[case] variable: bool) {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;

    use test_util::allocation;

    let path = test_util::temp::temp_path("member-associations-cost.h5");
    let mut fixture = Fixture::new();
    if variable {
        let collection = fixture.collection(&[Object::new(1, b"value001")]);
        object(&mut fixture, "data", collection, 1);
    } else {
        let records = Form::ContiguousDataset.messages(
            &mut fixture,
            &crate::make_i32_type(),
            &vec![0; 1_048_576],
        );
        fixture.object("data", records);
    }
    fixture.write(&path);
    let read_bytes = Arc::new(AtomicU64::new(0));
    let (engine, calls) =
        edit_test_util::open_counted_engine_with_bytes(&path, Arc::clone(&read_bytes));
    calls.store(0, Ordering::Relaxed);
    let (mut phase, empty) =
        allocation::measure("member_associations_empty", || AnalysisPhase::new(&engine));
    assert_eq!(
        (empty.blocks, empty.bytes, calls.load(Ordering::Relaxed)),
        (0, 0, 0)
    );
    let source = phase
        .reconstruction_source(&engine, id(Fixture::ROOT))
        .unwrap();
    let ((), first) = allocation::measure("member_associations_first", || {
        let facts = phase.member_associations(&source).unwrap();
        assert!(facts.topology_available());
        assert!(facts.incomplete_references().is_empty());
        assert_eq!(facts.collections().len(), usize::from(variable));
    });
    assert!(
        first.blocks > 0
            && first.bytes > 0
            && first.bytes < 32_768
            && first.peak_bytes < 32_768
            && first.live_bytes < 16_384,
        "{first}"
    );
    let before = (
        calls.load(Ordering::Relaxed),
        read_bytes.load(Ordering::Relaxed),
    );
    assert!(
        before.0 > 0 && before.0 < 32 && before.1 > 0 && before.1 < 16_384,
        "{before:?}"
    );
    let ((), cached) = allocation::measure("member_associations_cached", || {
        phase.member_associations(&source).unwrap();
    });
    assert_eq!((cached.blocks, cached.bytes), (0, 0));
    assert_eq!(
        (
            calls.load(Ordering::Relaxed),
            read_bytes.load(Ordering::Relaxed)
        ),
        before
    );
    assert!(phase.physical.is_none());
    assert!(phase.claims.is_none());
    assert!(phase.ownership.is_empty());
    let (control, measured) =
        allocation::measure("member_associations_control", || vec![0_u8; 1_048_576]);
    assert_eq!(control.len(), 1_048_576);
    assert!(measured.bytes >= 1_048_576);
}

fn object(fixture: &mut Fixture, name: &str, collection: u64, member: u32) -> ObjectId {
    let records = Form::CompactDataset.messages(
        fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(8, collection, member),
    );
    let object = fixture.append_header(records);
    fixture.link(name, object);
    id(object)
}

fn id(address: u64) -> ObjectId {
    ObjectId::new(PhysicalAddress::new(address))
}
fn index(value: u16) -> NonZeroU16 {
    NonZeroU16::new(value).unwrap()
}

#[derive(Clone, Copy)]
enum Partial {
    Framing,
    References,
    Topology,
}
