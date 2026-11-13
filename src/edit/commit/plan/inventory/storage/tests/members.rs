use core::cell::Cell;
use core::num::NonZeroU16;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::{FreeClass, PageType};
use rstest::rstest;
use test_util::disk_log;
use test_util::disk_log::IoEvent;
use test_util::global_heap::Object;
use test_util::object_header::Message;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use super::super::{
    PendingEditInventory, SpaceClaim, StorageConflict, StorageOrigin, StoragePublicationFailure,
    StoragePublicationRejection,
};
use crate::AttrValue;
use crate::edit::WriteEngine;
use crate::edit::allocation::AllocationPublication;
use crate::edit::analysis::heap::qualification::{
    MemberUnavailable, OriginalQualification, ParentUnavailable,
};
use crate::edit::analysis::ownership::AllocationKind;
use crate::edit::analysis::{COLLECTION_WORK, CollectionContent, CollectionProvenance};
use crate::edit::commit::PublicationError;
use crate::edit::commit::heap::HeapMutationState;
use crate::edit::commit::plan::inventory::members::MemberDecision;
use crate::edit::test_util as edit_test_util;
use crate::error::{Error, FormatError};
use crate::global_heap::GlobalHeapId;
use crate::image::WriteBuffering;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::unmatched_member(Form::CompactDataset)]
#[case::fill_reference(Form::FillValue)]
fn checked_completion_preserves_live_and_unobserved_members(#[case] form: Form) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("members.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[
        Object::new(7, b"removed"),
        Object::new(91, b"live"),
        Object::new(42, b"unobserved"),
    ]);
    let mut messages = form.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(4, parent, 91),
    );
    messages.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(7, parent, 7),
    ));
    let owner = ObjectId::new(PhysicalAddress::new(fixture.append_header(messages)));
    fixture.link("value", owner.header_address().get());
    let mut alias =
        Form::CompactDataset.messages(&mut fixture, &crate::make_i32_type(), &1_i32.to_le_bytes());
    alias.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(7, parent, 7),
    ));
    fixture.object("alias", alias);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_dataset_attr(&ObjectPathBuf::parse("value"), "a", AttrValue::I64(9))
        .unwrap();
    engine
        .set_dataset_attr(&ObjectPathBuf::parse("alias"), "a", AttrValue::I64(9))
        .unwrap();
    let prepared = super::prepare(&mut engine);
    let pending = prepared.apply_excluding(&mut engine).ok().unwrap();
    let before = COLLECTION_WORK.with(Cell::get);
    assert!(before.0 > 0 && before.1 > 0);
    let decisions = pending.member_decisions();
    assert_eq!(
        decisions.members().values().copied().collect::<Vec<_>>(),
        [Ok(MemberDecision::Remove), Ok(MemberDecision::Remove)]
    );
    assert_eq!(
        decisions.parents()[&PhysicalAddress::new(parent)],
        Err(ParentUnavailable::Unmatched)
    );
    assert_eq!(decisions.plans().len(), 1);
    assert_eq!(decisions.plans()[0].retained(), 2);
    let range = decisions.plans()[0].range();
    let completion = pending.publish_compacting_members().ok().unwrap();
    completion.mutation.unwrap();
    let OriginalQualification::Qualified(qualification) = completion.collections.qualification()
    else {
        panic!("expected retained original qualification");
    };
    assert_eq!(
        qualification.whole_parent(PhysicalAddress::new(parent)),
        Err(ParentUnavailable::Unmatched)
    );
    assert!(completion.prospective.is_complete());
    assert_eq!(completion.decisions.planning().len(), 1);
    completion.decisions.planning()[&PhysicalAddress::new(parent)]
        .as_ref()
        .unwrap();
    assert!(completion.end >= range.end());
    assert_eq!(completion.end.get(), engine.superblock().eof_address);
    assert_eq!(
        completion.root,
        engine
            .addresses()
            .resolve(engine.superblock().root_group_address)
            .unwrap()
    );
    assert_eq!(COLLECTION_WORK.with(Cell::get), before);
    let contents = crate::global_heap::collect_members(
        engine.image(),
        engine.widths().lengths,
        PhysicalAddress::new(parent),
    )
    .unwrap();
    assert_eq!(contents.range(), range);
    assert_eq!(
        contents
            .members()
            .iter()
            .map(|member| (member.index().get(), member.reference_count()))
            .collect::<Vec<_>>(),
        [(91, 1), (42, 1)]
    );
    let payloads = contents
        .members()
        .iter()
        .map(|member| {
            engine
                .image()
                .read_exact_at(
                    member.payload().range().start().get(),
                    usize::try_from(member.payload().range().length()).unwrap(),
                )
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(payloads, [b"live".as_slice(), b"unobserved".as_slice()]);
}

#[rstest]
#[case::new_copy(false)]
#[case::fill_reference(true)]
fn prospective_references_preserve_observed_members(#[case] retained: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("live.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"live")]);
    let mut messages =
        Form::CompactDataset.messages(&mut fixture, &crate::make_i32_type(), &1_i32.to_le_bytes());
    messages.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(4, parent, 7),
    ));
    fixture.object("value", messages);
    if retained {
        let messages = Form::FillValue.messages(
            &mut fixture,
            &global_heap_fixture::vlen_string(),
            &global_heap_fixture::reference(4, parent, 7),
        );
        fixture.object("keep", messages);
    }
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    if !retained {
        engine
            .copy(
                &ObjectPathBuf::parse("value"),
                &ObjectPathBuf::parse("copy"),
            )
            .unwrap();
    }
    engine
        .set_dataset_attr(&ObjectPathBuf::parse("value"), "a", AttrValue::I64(9))
        .unwrap();
    let pending = super::prepare(&mut engine)
        .apply_excluding(&mut engine)
        .ok()
        .unwrap();
    let decisions = pending.member_decisions();
    assert_eq!(
        decisions.members().values().copied().collect::<Vec<_>>(),
        [Ok(MemberDecision::Live)]
    );
    assert!(decisions.plans().is_empty());
    let completion = match pending.publish_compacting_members() {
        Ok(completion) => completion,
        Err(failure) => match *failure {
            StoragePublicationFailure::BeforePublication { reason, .. } => {
                panic!("unexpected publication rejection: {reason:?}")
            }
            StoragePublicationFailure::Publication { error, .. } => {
                panic!("unexpected publication failure: {error:?}")
            }
        },
    };
    completion.mutation.unwrap();
}

#[rstest]
#[case::outside_actual_root(false)]
#[case::aliased_into_actual_root(true)]
fn translated_retained_roots_preserve_their_observed_members(#[case] aliased: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("retained.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"retained")]);
    let mut messages = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(8, parent, 7),
    );
    messages.push(Message::reference_count(2));
    let owner = fixture.append_header(messages);
    let owner_end = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    fixture.link("remove", owner);
    if aliased {
        fixture.link("keep", owner);
    }
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("remove")).unwrap();
    let original = engine
        .image()
        .read_exact_at(parent, usize::try_from(owner - parent).unwrap())
        .unwrap();
    let mut applied = super::prepare(&mut engine).apply(&mut engine).unwrap();
    if !aliased {
        applied.tree.freed.push((
            super::range(owner, owner_end - owner),
            FreeClass::Page(PageType::Meta),
        ));
    }
    let pending = PendingEditInventory::new(&mut engine, applied);
    assert!(pending.prospective().is_complete());
    assert!(
        pending
            .prospective()
            .roots()
            .contains(&ObjectId::new(PhysicalAddress::new(owner)))
    );
    let decisions = pending.member_decisions();
    assert_eq!(
        decisions.members().values().copied().collect::<Vec<_>>(),
        [Ok(MemberDecision::Live)]
    );
    assert!(decisions.plans().is_empty());
    disk_log::start();
    let result = pending.publish_compacting_members();
    let recording = disk_log::finish();
    if aliased {
        result.ok().unwrap().mutation.unwrap();
    } else {
        let StoragePublicationFailure::BeforePublication {
            pending,
            reason: StoragePublicationRejection::Exclusion,
        } = *result.err().unwrap()
        else {
            panic!("expected protected retained-owner rejection");
        };
        let Err(StorageConflict::Overlap {
            allocation,
            range,
            claim,
        }) = pending.exclusion()
        else {
            panic!("expected protected header overlap");
        };
        assert_eq!(allocation.range, super::range(owner, owner_end - owner));
        assert_eq!(*range, allocation.range);
        assert_eq!(
            allocation.origin,
            StorageOrigin::Owner {
                object: ObjectId::new(PhysicalAddress::new(owner)),
                kind: AllocationKind::Header,
            }
        );
        assert_eq!(
            *claim,
            SpaceClaim::Retirement(FreeClass::Page(PageType::Meta))
        );
        assert!(recording.events.is_empty());
        assert_eq!(
            pending
                .engine
                .image()
                .read_exact_at(parent, original.len())
                .unwrap(),
            original
        );
    }
}

#[rstest]
#[case::missing_member(true)]
#[case::different_session(false)]
fn original_and_prospective_restrictions_withhold_compaction(#[case] missing: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("unavailable.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"live")]);
    let mut messages =
        Form::CompactDataset.messages(&mut fixture, &crate::make_i32_type(), &1_i32.to_le_bytes());
    messages.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(4, parent, if missing { 91 } else { 7 }),
    ));
    let owner = fixture.append_header(messages);
    fixture.link("value", owner);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_dataset_attr(&ObjectPathBuf::parse("value"), "a", AttrValue::I64(9))
        .unwrap();
    let applied = super::prepare(&mut engine).apply(&mut engine).unwrap();
    if !missing {
        engine.context.allocations = AllocationPublication::new();
    }
    let pending = PendingEditInventory::new(&mut engine, applied);
    let decisions = pending.member_decisions();
    let outcome = decisions.members()[&(
        CollectionProvenance {
            object: ObjectId::new(PhysicalAddress::new(owner)),
            content: CollectionContent::Attribute("a".into()),
        },
        id(parent, if missing { 91 } else { 7 }),
    )];
    assert_eq!(
        outcome,
        if missing {
            Err(MemberUnavailable::Missing)
        } else {
            Ok(MemberDecision::Prospective)
        }
    );
    assert!(decisions.plans().is_empty());
}

#[test]
fn incomplete_prospective_interpretation_preserves_a_qualified_parent() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("incomplete.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"removed")]);
    let end = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    let mut messages =
        Form::CompactDataset.messages(&mut fixture, &crate::make_i32_type(), &1_i32.to_le_bytes());
    messages.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(7, parent, 7),
    ));
    fixture.object("value", messages);
    let data = PhysicalAddress::new(u64::try_from(fixture.image.as_bytes().len()).unwrap());
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(0, 0, 0),
    );
    fixture.object("peer", messages);
    fixture.write(&path);
    let mut engine = stage(&path);
    let original = engine
        .image()
        .read_exact_at(parent, usize::try_from(end - parent).unwrap())
        .unwrap();
    let applied = super::prepare(&mut engine).apply(&mut engine).unwrap();
    engine
        .write_inplace_journaled(data, &global_heap_fixture::reference(1, parent, 0))
        .unwrap();
    let pending = PendingEditInventory::new(&mut engine, applied);
    assert!(!pending.prospective().is_complete());
    let Some(Error::Format(error)) = pending.prospective().references().unwrap().error() else {
        panic!("expected a prospective descriptor error");
    };
    assert_eq!(
        error,
        &FormatError::GlobalHeapObjectNotFound {
            collection_address: parent,
            index: 0,
        }
    );
    let decisions = pending.member_decisions();
    assert_eq!(
        decisions.members().values().copied().collect::<Vec<_>>(),
        [Ok(MemberDecision::Prospective)]
    );
    assert!(decisions.plans().is_empty());
    let completion = pending.publish_compacting_members().ok().unwrap();
    completion.mutation.unwrap();
    assert!(!completion.prospective.is_complete());
    assert_eq!(
        engine
            .image()
            .read_exact_at(parent, original.len())
            .unwrap(),
        original
    );
}

#[rstest]
#[case::success(Completion::Success)]
#[case::publication_failure(Completion::PublicationFailure)]
#[case::second_parent_failure(Completion::CompactionFailure)]
fn multiple_parent_plans_begin_after_publication(#[case] outcome: Completion) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("batch.h5");
    let mut fixture = Fixture::new();
    let first = fixture.collection(&[Object::new(7, b"first")]);
    let second = fixture.collection(&[Object::new(7, b"second")]);
    let mut messages =
        Form::CompactDataset.messages(&mut fixture, &crate::make_i32_type(), &1_i32.to_le_bytes());
    let mut descriptors = global_heap_fixture::reference(5, first, 7);
    descriptors.extend(global_heap_fixture::reference(6, second, 7));
    messages.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &descriptors,
    ));
    fixture.object("value", messages);
    fixture.write(&path);
    let initial = fs::read(&path).unwrap();
    let mut engine = stage(&path);
    let pending = super::prepare(&mut engine)
        .apply_excluding(&mut engine)
        .ok()
        .unwrap();
    let decisions = pending.member_decisions();
    assert_eq!(decisions.plans().len(), 2);
    let second_range = decisions.plans()[1].range();
    let second_header = decisions.plans()[1].free_space().unwrap().header_range();
    disk_log::start();
    let completion = pending.publish_compacting_members().ok().unwrap();
    let recording = disk_log::finish();
    completion.mutation.unwrap();
    assert_eq!(
        completion
            .decisions
            .parents()
            .values()
            .copied()
            .collect::<Vec<_>>(),
        [Ok(MemberDecision::Remove), Ok(MemberDecision::Remove)]
    );
    assert!(
        crate::global_heap::collect_members(
            engine.image(),
            engine.widths().lengths,
            PhysicalAddress::new(first)
        )
        .unwrap()
        .members()
        .is_empty()
    );
    assert!(
        crate::global_heap::collect_members(
            engine.image(),
            engine.widths().lengths,
            PhysicalAddress::new(second)
        )
        .unwrap()
        .members()
        .is_empty()
    );
    if matches!(outcome, Completion::Success) {
        return;
    }
    let write = if matches!(outcome, Completion::PublicationFailure) {
        0
    } else {
        recording
            .events
            .iter()
            .filter_map(|event| match event {
                IoEvent::Write { offset, .. } => Some(*offset),
                _ => None,
            })
            .position(|offset| offset == second_header.start().get())
            .unwrap()
    };
    drop(engine);
    fs::write(&path, initial).unwrap();
    let mut engine = stage(&path);
    let first_original = engine
        .image()
        .read_exact_at(first, usize::try_from(second - first).unwrap())
        .unwrap();
    let original = engine
        .image()
        .read_exact_at(
            second_range.start().get(),
            usize::try_from(second_range.length()).unwrap(),
        )
        .unwrap();
    let pending = super::prepare(&mut engine)
        .apply_excluding(&mut engine)
        .ok()
        .unwrap();
    disk_log::start();
    disk_log::fail_write_after(write, 2);
    let result = pending.publish_compacting_members();
    let recording = disk_log::finish();
    if matches!(outcome, Completion::PublicationFailure) {
        let StoragePublicationFailure::Publication {
            error: PublicationError::Write(Error::Io(error)),
            ..
        } = *result.err().unwrap()
        else {
            panic!("expected publication I/O failure");
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(error.to_string(), "injected disk write failure");
        assert_eq!(recording.operations.len(), 1);
        assert_eq!(
            engine
                .image()
                .read_exact_at(first, first_original.len())
                .unwrap(),
            first_original
        );
        assert_eq!(
            engine
                .image()
                .read_exact_at(second, original.len())
                .unwrap(),
            original
        );
    } else {
        let completion = result.ok().unwrap();
        let failure = completion.mutation.unwrap_err();
        assert_eq!(failure.completed, 1);
        assert_eq!(failure.state, HeapMutationState::Restored);
        let Error::Io(error) = failure.error else {
            panic!("expected a compaction I/O failure");
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(error.to_string(), "injected disk write failure");
        assert!(
            crate::global_heap::collect_members(
                engine.image(),
                engine.widths().lengths,
                PhysicalAddress::new(first)
            )
            .unwrap()
            .members()
            .is_empty()
        );
        assert_eq!(
            engine
                .image()
                .read_exact_at(second, original.len())
                .unwrap(),
            original
        );
    }
}

fn stage(path: &Path) -> WriteEngine {
    let mut engine = edit_test_util::open_engine(path);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine
        .set_dataset_attr(&ObjectPathBuf::parse("value"), "a", AttrValue::I64(9))
        .unwrap();
    engine
}

fn id(parent: u64, index: u16) -> GlobalHeapId {
    GlobalHeapId {
        collection: PhysicalAddress::new(parent),
        index: NonZeroU16::new(index).unwrap(),
    }
}

#[derive(Clone, Copy)]
enum Completion {
    CompactionFailure,
    PublicationFailure,
    Success,
}
