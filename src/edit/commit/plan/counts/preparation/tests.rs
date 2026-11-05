use core::num::NonZeroU32;
use std::collections::BTreeMap;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::sync::atomic::Ordering;

use hdf5_pure_core::__private::FileRange;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use hdf5_pure_format::__private::ObjectReferenceCount;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::PageType;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::datatype;
use test_util::global_heap::Object;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags;
use test_util::object_header::MessageType;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use super::PreparedTreeCounts;
use crate::Datatype;
use crate::DatatypeByteOrder;
use crate::FloatingPointType;
use crate::PhysicalAddress;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::access_mode::AccessMode;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::edit::WriteEngine;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::commit::plan::CommitPlan;
use crate::edit::commit::plan::CommitPreparation;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::edit::commit::plan::Node;
use crate::edit::commit::plan::PreparedCommit;
use crate::edit::test_util as edit_test_util;
use crate::object_graph::ObjectId;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::object_path::LinkNameBuf;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::one_alias(2, false, 1)]
#[case::final_aliases(2, true, 0)]
#[case::manual_pin(3, true, 1)]
fn commit_preparation_retains_exact_count_storage_and_source_protection(
    #[case] original: u32,
    #[case] all: bool,
    #[case] remaining: u32,
) {
    let path = test_util::temp::temp_path("prepared-count-storage.h5");
    let mut fixture = Fixture::new();
    let before_raw = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    let mut messages =
        Form::ContiguousDataset.messages(&mut fixture, &number(), &17_f64.to_le_bytes());
    let raw = range(before_raw, 8);
    messages.push(Message::reference_count(original));
    let target = fixture.append_header(messages);
    let header = range(
        target,
        u64::try_from(fixture.image.as_bytes().len()).unwrap() - target,
    );
    fixture.link("a", target);
    fixture.link("b", target);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let before = engine.disk_bytes().unwrap();
    engine.delete(&ObjectPathBuf::parse("a")).unwrap();
    if all {
        engine.delete(&ObjectPathBuf::parse("b")).unwrap();
    }
    let staged = std::mem::take(&mut engine.staged);
    engine.begin_commit(&staged);
    let CommitPreparation::Prepared(prepared) = PreparedCommit::prepare(&mut engine, staged) else {
        panic!("expected count preparation");
    };
    let CommitPlan::Tree(tree) = prepared.plan else {
        panic!("expected tree preparation")
    };
    let counts = tree.counts.into_checked().unwrap();
    let entries = counts.preflight.changes();
    assert_eq!(entries.len(), 1);
    assert_eq!(
        (
            entries[0].object(),
            entries[0].original().value(),
            entries[0].remaining()
        ),
        (id(target), original, remaining)
    );
    assert_eq!(entries[0].header_allocation().unwrap().range(), header);
    assert!(
        counts
            .projection
            .matches(&counts.preflight.projection_identity())
    );
    assert!(counts.storage.same_session(&engine));
    assert!(
        AnalysisPhase::new(&engine)
            .unreferenced_collection_spans(counts.collections)
            .is_empty()
    );
    assert_eq!(counts.storage.root(), Some(id(Fixture::ROOT)));
    if remaining == 0 {
        let mut allocations = counts.storage.allocations().to_vec();
        allocations.sort_unstable_by_key(|(range, _)| range.start());
        assert_eq!(
            allocations,
            [
                (raw, FreeClass::Page(PageType::Raw)),
                (header, FreeClass::Page(PageType::Meta))
            ]
        );
        assert!(counts.storage.reference_roots().is_empty());
        assert!(counts.preflight.protected().updates().is_empty());
    } else {
        assert!(counts.storage.allocations().is_empty());
        assert_eq!(counts.storage.reference_roots(), [id(target)]);
        let updates = counts.preflight.protected().updates();
        assert_eq!(updates.len(), 1);
        assert_eq!(
            (
                updates[0].object(),
                updates[0].count().get(),
                updates[0].allocation().range()
            ),
            (id(target), remaining, header)
        );
    }
    assert_eq!(engine.disk_bytes().unwrap(), before);
}

#[test]
fn commit_preparation_retains_collections_from_count_eligible_objects() {
    let path = test_util::temp::temp_path("prepared-count-collections.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"value")]);
    let mut messages = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, collection, 1),
    );
    messages.push(Message::reference_count(2));
    let owner = fixture.append_header(messages);
    fixture.link("a", owner);
    fixture.link("b", owner);
    let replacement = fixture.append_header(vec![]);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let before = engine.disk_bytes().unwrap();
    let allocation = engine.collection_span(collection).unwrap();
    engine.delete(&ObjectPathBuf::parse("a")).unwrap();
    engine.delete(&ObjectPathBuf::parse("b")).unwrap();
    let staged = std::mem::take(&mut engine.staged);
    engine.begin_commit(&staged);
    let CommitPreparation::Prepared(prepared) = PreparedCommit::prepare(&mut engine, staged) else {
        panic!("expected count preparation");
    };
    let CommitPlan::Tree(tree) = prepared.plan else {
        panic!("expected tree preparation")
    };
    let counts = tree.counts.into_checked().unwrap();
    assert_eq!(counts.storage.allocations().len(), 1);
    assert_eq!(
        counts
            .storage
            .reference_phase(&engine, PhysicalAddress::new(replacement), &BTreeMap::new())
            .unwrap()
            .unreferenced_collection_spans(counts.collections),
        [(allocation, FreeClass::Page(PageType::Raw))]
    );
    assert_eq!(engine.disk_bytes().unwrap(), before);
}

#[rstest]
#[case::one_copy(&["copy"], false, 3)]
#[case::two_copies(&["copy", "again"], false, 4)]
#[case::copy_and_remove_name(&["copy"], true, 2)]
fn actual_copy_emissions_share_the_commit_count_plan(
    #[case] copies: &[&str],
    #[case] remove: bool,
    #[case] remaining: u32,
) {
    let path = test_util::temp::temp_path("prepared-copy-counts.h5");
    let mut fixture = Fixture::new();
    let target = fixture.append_header(vec![
        Message::new(MessageType::DATATYPE, &datatype::f64_le()),
        Message::reference_count(2),
    ]);
    fixture.link("type", target);
    let mut messages =
        Form::CompactDataset.messages(&mut fixture, &number(), &17_f64.to_le_bytes());
    *messages
        .iter_mut()
        .find(|message| message.msg_type == MessageType::DATATYPE)
        .unwrap() = Message::new(
        MessageType::DATATYPE,
        &datatype::committed_reference(target),
    )
    .with_flags(MessageFlags::SHARED);
    fixture.object("source", messages);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let before = engine.disk_bytes().unwrap();
    for destination in copies {
        engine
            .copy(
                &ObjectPathBuf::parse("source"),
                &ObjectPathBuf::parse(destination),
            )
            .unwrap();
    }
    if remove {
        engine.delete(&ObjectPathBuf::parse("type")).unwrap();
    }
    let staged = std::mem::take(&mut engine.staged);
    engine.begin_commit(&staged);
    let CommitPreparation::Prepared(prepared) = PreparedCommit::prepare(&mut engine, staged) else {
        panic!("expected copy count preparation")
    };
    let CommitPlan::Tree(tree) = prepared.plan else {
        panic!("expected tree preparation")
    };
    let counts = tree.counts.into_checked().unwrap();
    assert_eq!(counts.preflight.changes().len(), 1);
    let entry = &counts.preflight.changes()[0];
    assert_eq!(
        (entry.object(), entry.original().value(), entry.remaining()),
        (id(target), 2, remaining)
    );
    assert_eq!(
        counts.preflight.protected().updates().len(),
        usize::from(remaining != 2)
    );
    assert_eq!(counts.storage.reference_roots(), [id(target)]);
    assert!(counts.storage.allocations().is_empty());
    assert!(
        counts
            .projection
            .matches(&counts.preflight.projection_identity())
    );
    assert_eq!(engine.disk_bytes().unwrap(), before);
}

#[test]
fn reconstructed_root_counts_are_projected_before_application() {
    let path = test_util::temp::temp_path("prepared-root-count.h5");
    let mut fixture = Fixture::new().with_root_reference_count(NonZeroU32::new(2).unwrap());
    fixture.link("self", Fixture::ROOT);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let before = engine.disk_bytes().unwrap();
    engine.delete(&ObjectPathBuf::parse("self")).unwrap();
    let staged = std::mem::take(&mut engine.staged);
    engine.begin_commit(&staged);
    let CommitPreparation::Prepared(prepared) = PreparedCommit::prepare(&mut engine, staged) else {
        panic!("expected root count preparation")
    };
    let CommitPlan::Tree(mut tree) = prepared.plan else {
        panic!("expected tree preparation")
    };
    let counts = tree.counts.into_checked().unwrap();
    let root = tree.groups.get_mut(&ObjectPathBuf::root()).unwrap();
    assert_eq!(
        root.prepared.region.reference_count_mut().unwrap().count(),
        1
    );
    assert_eq!(counts.preflight.changes()[0].remaining(), 1);
    assert!(counts.preflight.protected().updates().is_empty());
    assert!(counts.storage.allocations().is_empty());
    assert_eq!(counts.storage.reference_roots(), [id(Fixture::ROOT)]);
    assert_eq!(engine.disk_bytes().unwrap(), before);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::empty(Idle::Empty)]
#[case::new_group(Idle::Group)]
#[case::numeric_copy(Idle::Copy)]
fn count_preparation_without_users_requires_no_reads_or_allocations(#[case] work: Idle) {
    let path = test_util::temp::temp_path("idle-count-preparation.h5");
    let mut fixture = Fixture::new();
    let mut records = Form::CompactDataset.messages(&mut fixture, &number(), &17_f64.to_le_bytes());
    records.push(Message::reference_count(2));
    let target = fixture.append_header(records);
    fixture.link("source", target);
    fixture.write(&path);
    let (engine, reads) = edit_test_util::open_counted_engine(&path);
    let mut positive = AnalysisPhase::new(&engine);
    let ObjectReferenceCount::Version2(field) =
        positive.object_reference_count(id(target)).unwrap()
    else {
        panic!("expected the independent explicit count")
    };
    assert_eq!(field.value(), 2);
    assert!(reads.load(Ordering::Relaxed) > 0);
    drop(positive);
    let (control, measured) = allocation::measure("idle_count_control", || {
        std::hint::black_box(vec![1_u8; 64])
    });
    assert_eq!(control, vec![1; 64]);
    assert!(measured.blocks > 0 && measured.bytes >= 64, "{measured}");
    drop(control);
    let mut nodes = BTreeMap::new();
    if !matches!(work, Idle::Empty) {
        let mut node = Node::default();
        if matches!(work, Idle::Copy) {
            let tree = WriteEngine::read_copy_subtree(
                &engine.opened_source(),
                AccessMode::ReadWrite,
                target,
                false,
                engine.widths(),
            )
            .unwrap();
            node.prepared
                .copies
                .push((LinkNameBuf::new("copy").unwrap(), tree));
        }
        nodes.insert(ObjectPathBuf::root(), node);
    }
    let mut analysis = AnalysisPhase::new(&engine);
    reads.store(0, Ordering::Relaxed);
    let (prepared, measured) = allocation::measure("idle_count_preparation", || {
        PreparedTreeCounts::prepare(&engine, &mut nodes, &mut analysis).unwrap()
    });
    assert_eq!((measured.blocks, measured.bytes), (0, 0), "{measured}");
    assert_eq!(reads.load(Ordering::Relaxed), 0);
    assert!(matches!(prepared, super::PreparedCountWork::Empty));
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[derive(Clone, Copy)]
enum Idle {
    Copy,
    Empty,
    Group,
}

fn number() -> Datatype {
    Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(
        DatatypeByteOrder::LittleEndian,
    ))
}

fn id(address: u64) -> ObjectId {
    ObjectId::new(PhysicalAddress::new(address))
}

fn range(address: u64, length: u64) -> FileRange {
    FileRange::new(PhysicalAddress::new(address), length).unwrap()
}
