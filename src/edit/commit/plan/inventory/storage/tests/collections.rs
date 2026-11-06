use core::num::NonZeroU16;
use std::io::ErrorKind;

use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::disk_log;
use test_util::global_heap::Object;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use super::super::{
    PendingEditInventory, StorageApplyError, StorageApplyFailure, StorageConflict,
    StoragePublicationFailure, StoragePublicationRejection,
};
use crate::AttrValue;
use crate::edit::allocation::AllocationPublication;
use crate::edit::analysis::heap::qualification::{OriginalCollections, OriginalQualification};
use crate::edit::analysis::{CollectionContent, CollectionProvenance};
use crate::edit::commit::PublicationError;
use crate::edit::test_util as edit_test_util;
use crate::error::{Error, FormatError};
use crate::global_heap::GlobalHeapId;
use crate::image::WriteBuffering;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::completion(Outcome::Complete)]
#[case::publication_failure(Outcome::PublicationFailure)]
#[case::completion_failure(Outcome::CompletionFailure)]
#[case::rejected_publication(Outcome::Rejected)]
#[case::preplacement_failure(Outcome::PreplacementFailure)]
#[case::application_failure(Outcome::ApplicationFailure)]
fn checked_tree_handoffs_retain_original_members_and_framing(#[case] outcome: Outcome) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("members.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"before"), Object::new(91, b"other")]);
    let mut records =
        Form::CompactDataset.messages(&mut fixture, &crate::make_i32_type(), &1_i32.to_le_bytes());
    records.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(6, parent, 7),
    ));
    let owner = ObjectId::new(PhysicalAddress::new(fixture.append_header(records)));
    fixture.link("owner", owner.header_address().get());
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_dataset_attr(&ObjectPathBuf::parse("owner"), "a", AttrValue::I64(9))
        .unwrap();
    let prepared = super::prepare(&mut engine);
    let id = GlobalHeapId {
        collection: PhysicalAddress::new(parent),
        index: NonZeroU16::new(7).unwrap(),
    };
    let members = check(prepared.collections(), owner, id);
    if matches!(outcome, Outcome::PreplacementFailure) {
        engine.context.allocations = AllocationPublication::new();
        let StorageApplyFailure::BeforePlacement { prepared, error } =
            prepared.apply_excluding(&mut engine).err().unwrap()
        else {
            panic!("expected preplacement failure");
        };
        assert!(matches!(error, StorageApplyError::Session), "{error:?}");
        assert_eq!(check(prepared.collections(), owner, id), members);
        return;
    }
    if matches!(outcome, Outcome::ApplicationFailure) {
        engine
            .set_write_buffering(WriteBuffering::Unbuffered)
            .unwrap();
        disk_log::start();
        disk_log::fail_write_after(1, 0);
        let failure = prepared.apply_excluding(&mut engine).err().unwrap();
        let recording = disk_log::finish();
        let StorageApplyFailure::Application(Error::Io(error)) = failure else {
            panic!("expected application I/O failure");
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(error.to_string(), "injected disk write failure");
        assert_eq!(
            recording
                .events
                .iter()
                .filter(|event| matches!(
                    event,
                    disk_log::IoEvent::Write {
                        outcome: disk_log::IoOutcome::Succeeded,
                        ..
                    }
                ))
                .count(),
            1
        );
        return;
    }
    let applied = prepared.apply(&mut engine).unwrap();
    assert_eq!(check(applied.collections(), owner, id), members);
    assert!(
        applied
            .tree
            .freed
            .iter()
            .all(|(range, _)| range.start() != id.collection)
    );
    let mut pending = PendingEditInventory::new(&mut engine, applied);
    pending.exclusion().unwrap();
    assert_eq!(check(pending.collections(), owner, id), members);
    let original_bytes = pending
        .engine
        .image()
        .read_exact_at(
            id.collection.get(),
            usize::try_from(
                pending.collections().contents()[&id.collection]
                    .as_ref()
                    .unwrap()
                    .range()
                    .length(),
            )
            .unwrap(),
        )
        .unwrap();
    if matches!(outcome, Outcome::Rejected) {
        pending.exclusion = Err(StorageConflict::End {
            proposed: PhysicalAddress::new(0),
            required: pending.maximum_end(),
        });
    }
    pending
        .engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    disk_log::start();
    if matches!(outcome, Outcome::PublicationFailure) {
        disk_log::fail_write_after(0, 0);
    }
    let result = pending.publish_with_completion(|published, _, _, _, collections, prospective| {
        assert!(prospective.is_complete());
        assert!(prospective.references().unwrap().objects().is_empty());
        assert_eq!(check(collections, owner, id), members);
        let engine = published.into_engine();
        assert_eq!(
            engine
                .image()
                .read_exact_at(id.collection.get(), original_bytes.len())
                .unwrap(),
            original_bytes
        );
        if matches!(outcome, Outcome::CompletionFailure) {
            return Err(FormatError::Internal("completion failure".into()).into());
        }
        Ok(())
    });
    let recording = disk_log::finish();
    match outcome {
        Outcome::ApplicationFailure | Outcome::PreplacementFailure => {
            panic!("expected the application failure to return")
        }
        Outcome::Complete => {
            let Ok(()) = result else {
                panic!("expected successful protected publication");
            };
        }
        Outcome::Rejected => {
            let StoragePublicationFailure::BeforePublication {
                pending,
                reason: StoragePublicationRejection::Exclusion,
            } = *result.err().unwrap()
            else {
                panic!("expected rejected publication");
            };
            assert_eq!(check(pending.collections(), owner, id), members);
            assert!(recording.events.is_empty());
        }
        Outcome::PublicationFailure | Outcome::CompletionFailure => {
            let StoragePublicationFailure::Publication {
                collections,
                prospective,
                error,
                ..
            } = *result.err().unwrap()
            else {
                panic!("expected publication or completion failure");
            };
            assert_eq!(check(&collections, owner, id), members);
            assert!(prospective.is_complete());
            assert!(prospective.references().unwrap().objects().is_empty());
            match error {
                PublicationError::Write(Error::Io(error))
                    if matches!(outcome, Outcome::PublicationFailure) =>
                {
                    assert_eq!(error.kind(), ErrorKind::Other);
                    assert_eq!(error.to_string(), "injected disk write failure");
                }
                PublicationError::Write(Error::Format(error))
                    if matches!(outcome, Outcome::CompletionFailure) =>
                {
                    assert_eq!(error, FormatError::Internal("completion failure".into()));
                }
                other => panic!("unexpected publication error: {other:?}"),
            }
        }
    }
}

fn check(collections: &OriginalCollections, owner: ObjectId, id: GlobalHeapId) -> Vec<(u16, u16)> {
    let OriginalQualification::Qualified(qualification) = collections.qualification() else {
        panic!("expected original collection qualification");
    };
    qualification
        .member(
            &CollectionProvenance {
                object: owner,
                content: CollectionContent::Attribute("a".into()),
            },
            id,
        )
        .unwrap();
    let contents = collections.contents()[&id.collection].as_ref().unwrap();
    assert_eq!(contents.range().start(), id.collection);
    assert_eq!(
        qualification.original_allocation(id.collection).unwrap().0,
        contents.range()
    );
    let members = contents
        .members()
        .iter()
        .map(|member| (member.index().get(), member.reference_count()))
        .collect::<Vec<_>>();
    assert_eq!(members, [(7, 1), (91, 1)]);
    members
}

#[derive(Clone, Copy)]
enum Outcome {
    ApplicationFailure,
    Complete,
    CompletionFailure,
    PreplacementFailure,
    PublicationFailure,
    Rejected,
}
