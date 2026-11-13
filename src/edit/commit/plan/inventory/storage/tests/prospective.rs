use core::cell::Cell;
use core::num::NonZeroU16;
use std::collections::BTreeSet;
use std::fs;
use std::io::ErrorKind;

use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::disk_log;
use test_util::global_heap::Object;
use test_util::group_info;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::object_header::v2::Header;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use super::super::PendingEditInventory;
use super::super::StorageApplyFailure;
use crate::AttrValue;
use crate::DatasetBuilder;
use crate::edit::allocation::AllocationPublication;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::COLLECTION_WORK;
use crate::edit::analysis::OriginalOwnerError;
use crate::edit::analysis::heap::qualification::OriginalQualification;
use crate::edit::commit::plan::CommitPlan;
use crate::edit::commit::plan::inventory::OriginalEdits;
use crate::edit::commit::plan::inventory::dispositions::PlacementError;
use crate::edit::commit::plan::inventory::dispositions::TransitionWithheld;
use crate::edit::commit::plan::inventory::prospective::ProspectiveUnavailable;
use crate::edit::commit::plan::inventory::prospective::ReferencePresence;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::error::FormatError;
use crate::global_heap::GlobalHeapId;
use crate::image::WriteBuffering;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::root_attribute(Edit::Root)]
#[case::rebuilt_ancestors(Edit::Ancestors)]
#[case::mixed_value_write(Edit::Values)]
fn checked_application_collects_one_fresh_union(#[case] edit: Edit) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fresh.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"prior"), Object::new(91, b"value")]);
    let mut records = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, parent, 91),
    );
    records.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, parent, 7),
    ));
    let owner = fixture.append_header(records);
    let b = fixture.append_header(group_info::compact_messages(&[
        global_heap_fixture::hard_link("value", owner),
    ]));
    let a = fixture.append_header(group_info::compact_messages(&[
        global_heap_fixture::hard_link("b", b),
    ]));
    let mut root_records = group_info::compact_messages(&[global_heap_fixture::hard_link("a", a)]);
    root_records.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, parent, 7),
    ));
    let root_header = Header::new().messages(root_records).build();
    fixture.write(&path);
    let mut bytes = fs::read(&path).unwrap();
    let root_start = usize::try_from(Fixture::ROOT).unwrap();
    bytes[root_start..root_start + root_header.len()].copy_from_slice(&root_header);
    fs::write(&path, bytes).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    if matches!(edit, Edit::Root | Edit::Values) {
        engine
            .set_group_attr(&ObjectPathBuf::root(), "a", AttrValue::I64(2))
            .unwrap();
    } else {
        engine
            .set_dataset_attr(&ObjectPathBuf::parse("a/b/value"), "a", AttrValue::I64(2))
            .unwrap();
    }
    if !matches!(edit, Edit::Root) {
        engine
            .create_group(&ObjectPathBuf::parse("a/b/new"))
            .unwrap();
    }
    if matches!(edit, Edit::Values) {
        let mut values = DatasetBuilder::new("");
        values.with_raw_data(
            global_heap_fixture::vlen_string(),
            global_heap_fixture::reference(5, parent, 7),
            1,
        );
        engine
            .stage_dataset_write(&ObjectPathBuf::parse("a/b/value"), values)
            .unwrap();
    }
    let prepared = super::prepare(&mut engine);
    assert!(
        prepared
            .original()
            .facts()
            .unwrap()
            .counts()
            .changes()
            .is_empty()
    );
    assert_eq!(
        prepared.dispositions().predecessors().len(),
        match edit {
            Edit::Root => 1,
            Edit::Ancestors => 4,
            Edit::Values => 3,
        }
    );
    let CommitPlan::Tree(tree) = &prepared.prepared.plan else {
        panic!("expected a prepared tree");
    };
    assert_eq!(
        tree.inplace_writes.len(),
        usize::from(matches!(edit, Edit::Values))
    );
    assert!(matches!(
        prepared.collections().qualification(),
        OriginalQualification::Qualified(_)
    ));
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    let pending = prepared.apply_excluding(&mut engine).ok().unwrap();
    let prospective = pending.prospective();
    assert!(prospective.is_complete(), "{:?}", prospective.unavailable());
    assert_eq!(
        prospective.roots(),
        &BTreeSet::from([ObjectId::new(pending.pending.root)])
    );
    assert_eq!(prospective.root(), Some(pending.pending.root));
    let references = prospective.references().unwrap();
    assert_eq!(references.collections(), &BTreeSet::from([parent]));
    let mut expected = BTreeSet::from([id(parent, 7)]);
    if !matches!(edit, Edit::Values) {
        expected.insert(id(parent, 91));
    }
    assert_eq!(references.objects(), &expected);
    let objects = if matches!(edit, Edit::Root) { 4 } else { 5 };
    assert_eq!(COLLECTION_WORK.with(Cell::get), (objects, objects));
    let objects = references.objects().clone();
    pending
        .publish_with_completion(|_, _, _, _, _, prospective| {
            assert!(prospective.is_complete());
            assert_eq!(prospective.references().unwrap().objects(), &objects);
            Ok(())
        })
        .ok()
        .unwrap();
}

#[rstest]
#[case::retained_outside_root(false)]
#[case::retained_alias(true)]
fn each_object_is_decoded_once_across_actual_and_retained_roots(#[case] aliased: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("retained.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"value")]);
    let mut records = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, parent, 7),
    );
    records.push(Message::reference_count(2));
    let owner = fixture.append_header(records);
    fixture.link("remove", owner);
    if aliased {
        fixture.link("alias", owner);
    }
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("remove")).unwrap();
    let prepared = super::prepare(&mut engine);
    assert_eq!(
        prepared
            .original()
            .facts()
            .unwrap()
            .storage()
            .reference_roots(),
        [ObjectId::new(PhysicalAddress::new(owner))]
    );
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    let pending = prepared.apply_excluding(&mut engine).ok().unwrap();
    let prospective = pending.prospective();
    assert!(prospective.is_complete(), "{:?}", prospective.unavailable());
    assert_eq!(
        prospective.roots(),
        &BTreeSet::from([
            ObjectId::new(pending.pending.root),
            ObjectId::new(PhysicalAddress::new(owner))
        ])
    );
    assert_eq!(
        prospective.references().unwrap().objects(),
        &BTreeSet::from([id(parent, 7)])
    );
    assert_eq!(COLLECTION_WORK.with(Cell::get), (2, 2));
}

#[rstest]
#[case::original_qualification(Unavailable::Original)]
#[case::recorded_placement(Unavailable::Placement)]
#[case::different_source_phase(Unavailable::Source)]
#[case::different_session(Unavailable::Session)]
#[case::retained_predecessor(Unavailable::Predecessor)]
#[case::collection_qualification(Unavailable::Collection)]
#[case::incomplete_content(Unavailable::Content)]
#[case::malformed_descriptor(Unavailable::Descriptor)]
fn unavailable_roots_and_reference_outcomes_remain_independent(#[case] failure: Unavailable) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("unavailable.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"value")]);
    let mut values = global_heap_fixture::reference(5, parent, 7);
    if matches!(failure, Unavailable::Descriptor) {
        values.extend(global_heap_fixture::reference(1, parent, 0));
    }
    let mut records =
        Form::CompactDataset.messages(&mut fixture, &global_heap_fixture::vlen_string(), &values);
    records.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, parent, 7),
    ));
    if matches!(failure, Unavailable::Content) {
        records.push(Message::new(MessageType(0x7f), &[]));
    }
    if matches!(failure, Unavailable::Predecessor) {
        records.push(Message::reference_count(2));
    }
    let owner = fixture.append_header(records);
    fixture.link("owner", owner);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_dataset_attr(&ObjectPathBuf::parse("owner"), "a", AttrValue::I64(2))
        .unwrap();
    let other_original = if matches!(failure, Unavailable::Source) {
        let original = super::prepare(&mut engine).original;
        engine
            .set_dataset_attr(&ObjectPathBuf::parse("owner"), "a", AttrValue::I64(2))
            .unwrap();
        Some(original)
    } else {
        None
    };
    let mut prepared = super::prepare(&mut engine);
    match failure {
        Unavailable::Original => {
            prepared.original = OriginalEdits::Association(OriginalOwnerError::Phase)
        }
        Unavailable::Placement => prepared.dispositions.record(
            &ObjectPathBuf::root(),
            None,
            ObjectId::new(PhysicalAddress::new(0)),
        ),
        Unavailable::Source => prepared.original = other_original.unwrap(),
        Unavailable::Collection => {
            let mut phase = AnalysisPhase::new(&engine);
            let candidates =
                phase.deleted_collections(&[ObjectId::new(PhysicalAddress::new(owner))]);
            prepared.collections = phase.capture_collections(None, candidates);
        }
        _ => {}
    }
    let applied = prepared.apply(&mut engine).unwrap();
    if matches!(failure, Unavailable::Session) {
        engine.context.allocations = AllocationPublication::new();
    }
    let pending = PendingEditInventory::new(&mut engine, applied);
    let prospective = pending.prospective();
    let references = prospective.references().unwrap();
    assert_eq!(references.objects(), &BTreeSet::from([id(parent, 7)]));
    match failure {
        Unavailable::Original => assert_eq!(
            prospective.unavailable(),
            Some(&ProspectiveUnavailable::Original)
        ),
        Unavailable::Placement => assert_eq!(
            prospective.unavailable(),
            Some(&ProspectiveUnavailable::Dispositions(
                TransitionWithheld::Placement(PlacementError::Source(ObjectPathBuf::root()))
            ))
        ),
        Unavailable::Source => assert_eq!(
            prospective.unavailable(),
            Some(&ProspectiveUnavailable::Source)
        ),
        Unavailable::Session => assert_eq!(
            prospective.unavailable(),
            Some(&ProspectiveUnavailable::Session)
        ),
        Unavailable::Predecessor => assert_eq!(
            prospective.unavailable(),
            Some(&ProspectiveUnavailable::Dispositions(
                TransitionWithheld::Predecessors
            ))
        ),
        Unavailable::Collection => {
            assert!(prospective.is_complete());
            assert!(matches!(
                pending.collections().qualification(),
                OriginalQualification::Unavailable { .. }
            ));
        }
        Unavailable::Content => {
            assert_eq!(prospective.unavailable(), None);
            assert!(!references.is_complete());
            assert!(references.error().is_none());
        }
        Unavailable::Descriptor => {
            let Some(Error::Format(error)) = references.error() else {
                panic!("expected descriptor error");
            };
            assert_eq!(
                error,
                &FormatError::GlobalHeapObjectNotFound {
                    collection_address: parent,
                    index: 0
                }
            );
        }
    }
    assert_eq!(
        prospective.member_presence(id(parent, 7)),
        ReferencePresence::ObservedLive
    );
    assert_eq!(
        prospective.parent_presence(PhysicalAddress::new(parent)),
        ReferencePresence::ObservedLive
    );
    let absent = if matches!(failure, Unavailable::Collection) {
        ReferencePresence::Unreferenced
    } else {
        ReferencePresence::Unavailable
    };
    assert_eq!(prospective.member_presence(id(parent, 91)), absent);
    assert_eq!(
        prospective.parent_presence(PhysicalAddress::new(parent + 1)),
        absent
    );
    if !matches!(failure, Unavailable::Collection) {
        assert!(!prospective.is_complete());
    }
    if matches!(
        failure,
        Unavailable::Original
            | Unavailable::Placement
            | Unavailable::Predecessor
            | Unavailable::Source
            | Unavailable::Session
    ) {
        assert!(references.is_complete());
    }
}

#[rstest]
#[case::empty_candidates(false)]
#[case::application_failure(true)]
fn prospective_decode_work_is_zero_for_empty_input_and_application_failure(#[case] fail: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("lazy.h5");
    let mut fixture = Fixture::new();
    let mut records =
        Form::CompactDataset.messages(&mut fixture, &crate::make_i32_type(), &3_i32.to_le_bytes());
    if fail {
        let parent = fixture.collection(&[Object::new(7, b"value")]);
        records.extend(Form::CompactAttribute.messages(
            &mut fixture,
            &global_heap_fixture::vlen_string(),
            &global_heap_fixture::reference(5, parent, 7),
        ));
    }
    fixture.object("value", records);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    if fail {
        engine
            .set_dataset_attr(&ObjectPathBuf::parse("value"), "a", AttrValue::I64(2))
            .unwrap();
    } else {
        engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    }
    let prepared = super::prepare(&mut engine);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    if fail {
        disk_log::start();
        disk_log::fail_write_after(0, 0);
        let failure = prepared.apply_excluding(&mut engine).err().unwrap();
        let recording = disk_log::finish();
        let StorageApplyFailure::Application(Error::Io(error)) = failure else {
            panic!("expected application I/O failure");
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(error.to_string(), "injected disk write failure");
        assert!(!recording.events.is_empty());
    } else {
        let pending = prepared.apply_excluding(&mut engine).ok().unwrap();
        assert!(pending.prospective().root().is_none());
        assert!(pending.prospective().references().is_none());
        assert!(pending.prospective().roots().is_empty());
        assert!(pending.prospective().unavailable().is_none());
        assert!(!pending.prospective().is_complete());
        assert_eq!(
            pending.prospective().member_presence(id(Fixture::ROOT, 7)),
            ReferencePresence::Unavailable
        );
        assert_eq!(
            pending
                .prospective()
                .parent_presence(PhysicalAddress::new(Fixture::ROOT)),
            ReferencePresence::Unavailable
        );
        let decisions = pending.member_decisions();
        assert_eq!(decisions.physical_members(), 0);
        assert!(decisions.members().is_empty());
        assert!(decisions.parents().is_empty());
        assert!(decisions.plans().is_empty());
    }
    assert_eq!(COLLECTION_WORK.with(Cell::get), (0, 0));
}

fn id(parent: u64, index: u16) -> GlobalHeapId {
    GlobalHeapId {
        collection: PhysicalAddress::new(parent),
        index: NonZeroU16::new(index).unwrap(),
    }
}

#[derive(Clone, Copy)]
enum Edit {
    Ancestors,
    Root,
    Values,
}

#[derive(Clone, Copy)]
enum Unavailable {
    Collection,
    Content,
    Descriptor,
    Original,
    Placement,
    Predecessor,
    Session,
    Source,
}
