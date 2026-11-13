use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::FreeClass;
use rstest::rstest;
use test_util::global_heap::Object;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use super::super::OriginalStorage;
use super::super::SpaceClaim;
use super::super::StorageApplyError;
use super::super::StorageApplyFailure;
use super::super::StorageConflict;
use super::super::StorageOrigin;
use super::super::StoragePublicationFailure;
use super::super::StorageUnavailable;
use super::super::capture::StorageIntegrityError;
use crate::AttrValue;
use crate::File;
use crate::edit::allocation::AllocationPublication;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::heap::qualification::ParentUnavailable;
use crate::edit::commit::plan::CommitPlan;
use crate::edit::commit::plan::OverwritePayloads;
use crate::edit::commit::plan::PreparedCommit;
use crate::edit::commit::plan::ReplacedHeader;
use crate::edit::commit::plan::inventory::InventoryCapture;
use crate::edit::commit::plan::inventory::OriginalEdits;
use crate::edit::commit::plan::inventory::PreparedEditInventory;
use crate::edit::commit::plan::inventory::dispositions::ReconstructionDispositions;
use crate::edit::commit::plan::inventory::members::ParentDisposition;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::error::FormatError;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::unknown_message(true)]
#[case::unobserved_bytes(false)]
fn optional_owner_evidence_preserves_source_bytes_while_supported_edits_publish(
    #[case] unknown_message: bool,
) {
    let path = test_util::temp::temp_path("optional-owner-admission.h5");
    let mut fixture = Fixture::new();
    fixture.object(
        "owner",
        if unknown_message {
            vec![Message::new(MessageType(0x7f), &[])]
        } else {
            vec![]
        },
    );
    if !unknown_message {
        fixture.image.append(b"unobserved allocation");
    }
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    let prepared = super::prepare(&mut engine);
    assert!(matches!(
        prepared.storage().unavailable(),
        Some(StorageUnavailable::Owners)
    ));
    assert_eq!(prepared.storage().integrity(), Ok(()));
    assert!(prepared.collections().addresses().is_empty());
    let preserved = prepared
        .storage()
        .allocations()
        .iter()
        .map(|allocation| {
            (
                allocation.range,
                engine
                    .image()
                    .read_exact_at(
                        allocation.range.start().get(),
                        usize::try_from(allocation.range.length()).unwrap(),
                    )
                    .unwrap()
                    .to_vec(),
            )
        })
        .collect::<Vec<_>>();
    assert!(!preserved.is_empty());
    let pending = match prepared.apply_excluding(&mut engine) {
        Ok(pending) => pending,
        Err(StorageApplyFailure::BeforePlacement { error, .. }) => {
            panic!("unexpected application rejection: {error:?}")
        }
        Err(StorageApplyFailure::Application(error)) => {
            panic!("unexpected application failure: {error:?}")
        }
    };
    assert!(pending.prospective().roots().is_empty());
    let result = pending.publish_with_completion(|published, _, _, storage, _, _| {
        let engine = published.into_engine();
        for (range, bytes) in &preserved {
            assert_eq!(
                engine
                    .image()
                    .read_exact_at(range.start().get(), bytes.len())
                    .unwrap(),
                *bytes
            );
            assert!(
                storage
                    .allocations()
                    .iter()
                    .any(|allocation| allocation.range.contains(*range))
            );
        }
        Ok(())
    });
    finish(result);
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.group("new").unwrap().datasets().unwrap(),
        Vec::<String>::new()
    );
}

#[test]
fn optional_domain_evidence_withholds_generated_deletion_retirement() {
    let path = test_util::temp::temp_path("optional-deletion-retirement.h5");
    let mut fixture = Fixture::new();
    let raw = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &crate::make_i32_type(),
        &7_i32.to_le_bytes(),
    );
    let object = fixture.append_header(messages);
    let end = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    fixture.link("removed", object);
    fixture.object("unknown", vec![Message::new(MessageType(0x7f), &[])]);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let before = engine
        .image()
        .read_exact_at(raw, usize::try_from(end - raw).unwrap())
        .unwrap()
        .to_vec();
    engine.delete(&ObjectPathBuf::parse("removed")).unwrap();
    let prepared = super::prepare(&mut engine);
    assert!(prepared.original().owners().is_none());
    assert_eq!(
        prepared.original().facts().unwrap().eligible_owners(),
        [ObjectId::new(PhysicalAddress::new(object))]
    );
    assert!(
        prepared
            .reclamation
            .as_ref()
            .unwrap()
            .iter()
            .all(|(range, _)| !range.overlaps(super::range(raw, end - raw)))
    );
    let pending = match prepared.apply_excluding(&mut engine) {
        Ok(pending) => pending,
        Err(StorageApplyFailure::BeforePlacement { error, .. }) => {
            panic!("unexpected application rejection: {error:?}")
        }
        Err(StorageApplyFailure::Application(error)) => {
            panic!("unexpected application failure: {error:?}")
        }
    };
    pending.exclusion().unwrap();
    finish(pending.publish_with_completion(|published, _, _, _, _, _| {
        assert_eq!(
            published
                .into_engine()
                .image()
                .read_exact_at(raw, before.len())
                .unwrap(),
            before
        );
        Ok(())
    }));
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), Vec::<String>::new());
}

#[rstest]
#[case::retained_owner(Protection::Owner, (100, 40), false)]
#[case::partial_owner(Protection::Owner, (110, 10), true)]
#[case::enclosing_owner(Protection::Owner, (90, 60), true)]
#[case::unobserved(Protection::Unobserved, (110, 10), false)]
#[case::crossing_unobserved(Protection::Unobserved, (90, 20), true)]
#[case::parent(Protection::Parent, (100, 40), true)]
fn generated_retirement_withholds_exact_owner_or_contained_unobserved_ranges(
    #[case] protection: Protection,
    #[case] request: (u64, u64),
    #[case] kept: bool,
) {
    let path = test_util::temp::temp_path("optional-retirement-qualification.h5");
    let mut fixture = Fixture::new();
    fixture.object("unknown", vec![Message::new(MessageType(0x7f), &[])]);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    let prepared = super::prepare(&mut engine);
    assert!(prepared.original().owners().is_none());
    let mut storage = OriginalStorage::from_test_ranges(&[super::range(100, 40)]);
    storage.allocations[0].origin = match protection {
        Protection::Owner => storage.allocations[0].origin,
        Protection::Parent => StorageOrigin::Parent {
            address: PhysicalAddress::new(100),
            class: None,
        },
        Protection::Unobserved => StorageOrigin::Unobserved,
    };
    let entry = (super::range(request.0, request.1), FreeClass::Dead);
    let mut reclamation = Ok(vec![entry]);
    storage.withhold_unqualified_reclamation(prepared.original(), &mut reclamation);
    assert_eq!(
        reclamation.unwrap(),
        if kept { vec![entry] } else { vec![] }
    );
    if kept {
        let error = storage
            .exclude(entry.0, SpaceClaim::Retirement(entry.1))
            .unwrap_err();
        assert!(
            matches!(error, StorageConflict::Overlap { allocation, range, claim: SpaceClaim::Retirement(FreeClass::Dead) } if allocation == storage.allocations[0] && range == entry.0),
            "{error:?}"
        );
    }
}

#[rstest]
#[case::optional(true)]
#[case::source_failure(false)]
fn application_classifies_reclamation_failures(#[case] optional: bool) {
    let path = test_util::temp::temp_path("optional-reclamation-result.h5");
    Fixture::new().write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    let staged = std::mem::take(&mut engine.staged);
    let (CommitPlan::Tree(mut tree), _) = engine
        .plan_commit_with_inventory(&staged, InventoryCapture::Omit)
        .unwrap()
    else {
        panic!("expected tree preparation");
    };
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase
        .reconstruction_source(&engine, ObjectId::new(PhysicalAddress::new(Fixture::ROOT)))
        .unwrap();
    tree.groups
        .get_mut(&ObjectPathBuf::root())
        .unwrap()
        .prepared
        .replaces = Some(ReplacedHeader::Header(source));
    let original = OriginalEdits::collect(&engine, &mut phase, &tree, &staged);
    let dispositions = ReconstructionDispositions::collect(&engine, &mut phase, &tree);
    let reclamation = Err(if optional {
        Error::EditUnsupported("reclamation geometry is unavailable")
    } else {
        FormatError::OffsetOverflow {
            offset: u64::MAX,
            length: 7,
        }
        .into()
    });
    let storage = OriginalStorage::collect(
        &engine,
        &mut phase,
        &tree,
        &original,
        &dispositions,
        &reclamation,
    );
    assert!(matches!(
        storage.unavailable(),
        Some(StorageUnavailable::Reclamation)
    ));
    assert!(!storage.allocations().is_empty());
    let preserved = storage
        .allocations()
        .iter()
        .map(|allocation| {
            (
                allocation.range.start(),
                engine
                    .image()
                    .read_exact_at(
                        allocation.range.start().get(),
                        usize::try_from(allocation.range.length()).unwrap(),
                    )
                    .unwrap()
                    .to_vec(),
            )
        })
        .collect::<Vec<_>>();
    let collections = phase.capture_collections(
        dispositions.source().as_ref().ok(),
        std::mem::take(&mut tree.collection_candidates),
    );
    drop(phase);
    let (staged, writes) = staged.into_apply();
    let prepared = PreparedEditInventory {
        prepared: PreparedCommit {
            plan: CommitPlan::Tree(tree),
            staged,
            overwrites: OverwritePayloads { writes },
        },
        original,
        dispositions,
        storage,
        collections,
        reclamation,
        session: engine.context.allocations.session(),
    };
    let before = engine.disk_bytes().unwrap();
    let result = prepared.apply_excluding(&mut engine);
    match result {
        Ok(pending) => {
            assert!(optional);
            assert!(pending.pending.batch.entries().is_empty());
            finish(pending.publish_with_completion(|published, _, _, _, _, _| {
                let engine = published.into_engine();
                for (start, bytes) in &preserved {
                    assert_eq!(
                        engine
                            .image()
                            .read_exact_at(start.get(), bytes.len())
                            .unwrap(),
                        *bytes
                    );
                }
                Ok(())
            }));
        }
        Err(StorageApplyFailure::Application(Error::Format(FormatError::OffsetOverflow {
            offset,
            length,
        }))) => {
            assert!(!optional);
            assert_eq!(offset, u64::MAX);
            assert_eq!(length, 7);
        }
        Err(StorageApplyFailure::BeforePlacement { error, .. }) => {
            panic!("unexpected rejection: {error:?}")
        }
        Err(StorageApplyFailure::Application(error)) => {
            panic!("unexpected application error: {error:?}")
        }
    }
    if !optional {
        assert_eq!(engine.disk_bytes().unwrap(), before);
    }
}

#[rstest]
#[case::smaller_original(ParentEvidence::SmallerOriginal)]
#[case::smaller_framing(ParentEvidence::SmallerFraming)]
#[case::unavailable_framing(ParentEvidence::Framing)]
fn retained_parents_accept_enclosing_protection_without_removal_permission(
    #[case] evidence: ParentEvidence,
) {
    let path = test_util::temp::temp_path("optional-parent-admission.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"before")]);
    let mut messages =
        Form::CompactDataset.messages(&mut fixture, &crate::make_i32_type(), &7_i32.to_le_bytes());
    messages.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(6, parent, 7),
    ));
    let owner = fixture.append_header(messages);
    fixture.link("owner", owner);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_dataset_attr(&ObjectPathBuf::parse("owner"), "a", AttrValue::I64(3))
        .unwrap();
    let mut prepared = super::prepare(&mut engine);
    let parent = PhysicalAddress::new(parent);
    let expected = super::range(parent.get(), 4096);
    let before = engine
        .image()
        .read_exact_at(parent.get(), 4096)
        .unwrap()
        .to_vec();
    let storage = prepared
        .storage
        .parents
        .as_mut()
        .unwrap()
        .get_mut(&parent)
        .unwrap();
    match evidence {
        ParentEvidence::SmallerOriginal => {
            storage.original.as_mut().unwrap().0 = super::range(parent.get() + 8, 4088)
        }
        ParentEvidence::SmallerFraming => {
            storage.framing = Ok(super::range(parent.get() + 8, 4088))
        }
        ParentEvidence::Framing => storage.framing = Err(FormatError::InvalidGlobalHeapSignature),
    }
    storage.allocation = Err(ParentUnavailable::Qualification);
    let mut phase = AnalysisPhase::new(&engine);
    let candidates = phase.deleted_collections(&[ObjectId::new(PhysicalAddress::new(owner))]);
    prepared.collections = phase.capture_collections(None, candidates);
    let pending = match prepared.apply_excluding(&mut engine) {
        Ok(pending) => pending,
        Err(StorageApplyFailure::BeforePlacement { error, .. }) => {
            panic!("unexpected application rejection: {error:?}")
        }
        Err(StorageApplyFailure::Application(error)) => {
            panic!("unexpected application failure: {error:?}")
        }
    };
    assert!(matches!(
        pending.member_decisions().dispositions()[&parent],
        ParentDisposition::Retain
    ));
    finish(
        pending.publish_with_completion(|published, _, _, storage, _, _| {
            assert!(
                storage
                    .allocations()
                    .iter()
                    .any(|allocation| allocation.range == expected)
            );
            assert_eq!(
                published
                    .into_engine()
                    .image()
                    .read_exact_at(parent.get(), 4096)
                    .unwrap(),
                before
            );
            Ok(())
        }),
    );
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("owner").unwrap().read_i32().unwrap(), [7]);
}

#[rstest]
#[case::phase(false)]
#[case::session(true)]
fn application_checks_count_associations_without_owner_qualification(#[case] session: bool) {
    let path = test_util::temp::temp_path("optional-count-association.h5");
    let mut fixture = Fixture::new();
    let mut messages =
        Form::CompactDataset.messages(&mut fixture, &crate::make_i32_type(), &7_i32.to_le_bytes());
    messages.push(Message::reference_count(2));
    let object = fixture.append_header(messages);
    fixture.link("a", object);
    fixture.link("b", object);
    fixture.object("unknown", vec![Message::new(MessageType(0x7f), &[])]);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("a")).unwrap();
    let mut prepared = super::prepare(&mut engine);
    assert!(prepared.original().owners().is_none());
    assert_eq!(
        prepared
            .original()
            .facts()
            .unwrap()
            .counts()
            .changes()
            .len(),
        1
    );
    let original_session = session.then(|| {
        std::mem::replace(
            &mut engine.context.allocations,
            AllocationPublication::new(),
        )
    });
    engine.delete(&ObjectPathBuf::parse("a")).unwrap();
    let second = super::prepare(&mut engine);
    if let Some(original) = original_session {
        engine.context.allocations = original;
    }
    let OriginalEdits::Unavailable { counts, .. } = second.original else {
        panic!("expected unqualified owners");
    };
    let OriginalEdits::Unavailable { counts: first, .. } = &mut prepared.original else {
        panic!("expected unqualified owners");
    };
    *first = counts;
    let before = engine.disk_bytes().unwrap();
    let failure = match prepared.apply_excluding(&mut engine) {
        Ok(_) => panic!("expected a count association rejection"),
        Err(failure) => failure,
    };
    let StorageApplyFailure::BeforePlacement { error, .. } = failure else {
        panic!("expected rejection before placement");
    };
    assert!(
        matches!(
            (&error, session),
            (StorageApplyError::Session, true) | (StorageApplyError::Phase, false)
        ),
        "{error:?}"
    );
    assert_eq!(engine.disk_bytes().unwrap(), before);
}

#[test]
fn application_rejects_a_missing_source_tail() {
    let path = test_util::temp::temp_path("optional-source-tail.h5");
    Fixture::new().write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let end = engine.image.len();
    let mut superblock = engine.superblock().clone();
    superblock.eof_address = engine
        .addresses()
        .relative(PhysicalAddress::new(end + 8))
        .unwrap()
        .get();
    engine
        .context
        .adopt(engine.context.propose(superblock).unwrap())
        .unwrap();
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    let prepared = super::prepare(&mut engine);
    assert_eq!(
        prepared.storage().integrity(),
        Err(StorageIntegrityError::TruncatedSource {
            end: PhysicalAddress::new(end),
            required: PhysicalAddress::new(end + 8)
        })
    );
    let before = engine.disk_bytes().unwrap();
    let failure = match prepared.apply_excluding(&mut engine) {
        Ok(_) => panic!("expected a source integrity rejection"),
        Err(failure) => failure,
    };
    let StorageApplyFailure::BeforePlacement { error, .. } = failure else {
        panic!("expected rejection before placement");
    };
    assert!(
        matches!(error, StorageApplyError::Conflict(StorageConflict::Original(StorageIntegrityError::TruncatedSource { end: actual, required })) if actual.get() == end && required.get() == end + 8),
        "{error:?}"
    );
    assert_eq!(engine.disk_bytes().unwrap(), before);
}

fn finish<T>(result: Result<T, Box<StoragePublicationFailure<'_>>>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => match *error {
            StoragePublicationFailure::BeforePublication { reason, .. } => {
                panic!("unexpected publication rejection: {reason:?}")
            }
            StoragePublicationFailure::Publication { error, .. } => {
                panic!("unexpected publication failure: {error:?}")
            }
        },
    }
}

#[derive(Clone, Copy, Debug)]
enum ParentEvidence {
    Framing,
    SmallerFraming,
    SmallerOriginal,
}

#[derive(Clone, Copy, Debug)]
enum Protection {
    Owner,
    Parent,
    Unobserved,
}
