use core::cell::Cell;
use core::num::NonZeroU16;
use std::sync::atomic::Ordering;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::global_heap::Object;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use super::{MemberUnavailable, ParentUnavailable};
use crate::edit::analysis::heap::HeapReferenceError;
use crate::edit::analysis::heap::associations::MemberPresence;
use crate::edit::analysis::physical::claims::domain::{DomainUnavailable, OwnerDomain};
use crate::edit::analysis::{
    AnalysisPhase, COLLECTION_ALLOCATIONS, COLLECTION_WORK, CollectionCandidates,
    CollectionContent, CollectionProvenance,
};
use crate::edit::test_util as edit_test_util;
use crate::error::{Error, FormatError};
use crate::global_heap::GlobalHeapId;
use crate::object_graph::ObjectId;

#[rstest]
#[case::complete(Restriction::None)]
#[case::unmatched(Restriction::Unmatched)]
#[case::missing(Restriction::Missing)]
#[case::framing(Restriction::Framing)]
#[case::class(Restriction::Class)]
#[case::range(Restriction::Range)]
#[case::common_domain(Restriction::Domain)]
#[case::other_parent_framing(Restriction::PeerFraming)]
#[case::allocation_only(Restriction::Synthetic)]
fn qualification_preserves_independent_member_and_parent_outcomes(
    #[case] restriction: Restriction,
) {
    let path = test_util::temp::temp_path("collection-qualification.h5");
    let mut fixture = Fixture::new();
    let address = fixture.collection(&[Object::new(7, b"before"), Object::new(91, b"other")]);
    let first = object(
        &mut fixture,
        "first",
        address,
        if matches!(restriction, Restriction::Missing) {
            12
        } else {
            7
        },
    );
    let alias = object(&mut fixture, "alias", address, 7);
    if !matches!(restriction, Restriction::Unmatched) {
        object(&mut fixture, "second", address, 91);
    }
    let peer = matches!(restriction, Restriction::PeerFraming).then(|| {
        let address = fixture.collection(&[Object::new(3, b"peer")]);
        (
            PhysicalAddress::new(address),
            object(&mut fixture, "peer", address, 3),
        )
    });
    fixture.write(&path);
    let (engine, calls) = edit_test_util::open_counted_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase.reconstruction_source(&engine, first).unwrap();
    let mut candidates = phase.deleted_collections(&[first]);
    let address = PhysicalAddress::new(address);
    let original = candidates.collections[&address].range;
    if let Some((address, owner)) = peer {
        candidates
            .extend(phase.deleted_collections(&[owner]))
            .unwrap();
        phase.allocations.insert(
            address,
            Err(FormatError::Source("peer framing unavailable".into())),
        );
    }
    if matches!(restriction, Restriction::Synthetic) {
        candidates = CollectionCandidates::default();
        candidates
            .insert(
                phase.collection_allocation(address).unwrap(),
                [CollectionProvenance {
                    object: first,
                    content: CollectionContent::Object,
                }],
            )
            .unwrap();
    }
    match restriction {
        Restriction::Class => candidates.collections.get_mut(&address).unwrap().class = None,
        Restriction::Range => {
            candidates.collections.get_mut(&address).unwrap().range =
                FileRange::new(address, original.length() + 8).unwrap()
        }
        Restriction::Framing => {
            phase.allocations.insert(
                address,
                Err(FormatError::Source("framing unavailable".into())),
            );
        }
        Restriction::Domain => {
            phase.topology = Some(None);
        }
        _ => {}
    }
    let facts = phase.qualify_collections(&source, candidates).unwrap();
    let parent = &facts.parents()[&address];
    let origin = CollectionProvenance {
        object: first,
        content: CollectionContent::Object,
    };
    let id = GlobalHeapId {
        collection: address,
        index: index(if matches!(restriction, Restriction::Missing) {
            12
        } else {
            7
        }),
    };
    let expected = match restriction {
        Restriction::Missing => Err(MemberUnavailable::Missing),
        Restriction::Framing => Err(MemberUnavailable::Unknown),
        Restriction::Class => Err(MemberUnavailable::Parent(ParentUnavailable::Class)),
        Restriction::Range => Err(MemberUnavailable::Parent(ParentUnavailable::Range)),
        Restriction::Domain | Restriction::PeerFraming => Err(MemberUnavailable::Domain),
        Restriction::Synthetic => Err(MemberUnavailable::Observation),
        _ => Ok(()),
    };
    assert_eq!(facts.member(&origin, id), expected);
    let whole = match restriction {
        Restriction::None | Restriction::Synthetic => Ok(()),
        Restriction::Unmatched => Err(ParentUnavailable::Unmatched),
        Restriction::Missing => Err(ParentUnavailable::MissingMember),
        Restriction::Framing => Err(ParentUnavailable::Framing),
        Restriction::Class => Err(ParentUnavailable::Class),
        Restriction::Range => Err(ParentUnavailable::Range),
        Restriction::Domain => Err(ParentUnavailable::References),
        Restriction::PeerFraming => Err(ParentUnavailable::Domain),
    };
    assert_eq!(facts.whole_parent(address), whole);
    assert_eq!(
        parent.whole(),
        if matches!(restriction, Restriction::PeerFraming) {
            Ok(())
        } else {
            whole
        }
    );
    assert_eq!(
        parent.allocation(),
        match restriction {
            Restriction::Framing => Err(ParentUnavailable::Framing),
            Restriction::Class => Err(ParentUnavailable::Class),
            Restriction::Range => Err(ParentUnavailable::Range),
            _ => Ok(()),
        }
    );
    if matches!(restriction, Restriction::Synthetic) {
        assert!(parent.members().is_empty());
    } else {
        assert_eq!(
            parent.members()[&id.index],
            match restriction {
                Restriction::Missing => MemberPresence::Missing,
                Restriction::Framing => MemberPresence::Unknown,
                _ => MemberPresence::Allocated,
            }
        );
    }
    if let Some((peer, _)) = peer {
        assert_eq!(facts.parents().len(), 2);
        assert_eq!(
            facts.parents()[&peer].framing().unwrap_err(),
            &FormatError::Source("peer framing unavailable".into())
        );
        assert!(
            matches!(facts.domain(), Some(OwnerDomain::Unavailable(DomainUnavailable::Collection(found))) if *found == peer)
        );
    }
    if matches!(restriction, Restriction::Framing) {
        assert_eq!(
            parent.framing().unwrap_err(),
            &FormatError::Source("framing unavailable".into())
        );
        assert!(
            matches!(facts.domain(), Some(OwnerDomain::Unavailable(DomainUnavailable::Collection(found))) if *found == address)
        );
    } else {
        assert_eq!(parent.framing().unwrap(), original);
        assert_eq!(
            facts.contents(&mut phase, address).unwrap().members().len(),
            2
        );
    }
    assert_eq!(
        facts.original_allocation(address).unwrap().0.length(),
        original.length()
            + if matches!(restriction, Restriction::Range) {
                8
            } else {
                0
            }
    );
    assert_eq!(
        facts
            .allocation_origins(address)
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        [&origin]
    );
    assert!(facts.predecessors_complete());
    if !matches!(
        restriction,
        Restriction::Framing | Restriction::Domain | Restriction::PeerFraming
    ) {
        let associations = facts.associations(&mut phase).unwrap();
        let referrers = associations.collections()[&address].observed()[&index(7)].referrers();
        assert_eq!(
            referrers,
            if matches!(restriction, Restriction::Missing) {
                vec![alias]
            } else {
                vec![first, alias]
            }
        );
        assert!(matches!(facts.domain(), Some(OwnerDomain::Available(_))));
    }
    let reads = calls.load(Ordering::Relaxed);
    let repeated = phase.deleted_collections(&[first]);
    phase.qualify_collections(&source, repeated).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), reads);
}

#[rstest]
#[case::partial(Interpretation::Partial)]
#[case::error(Interpretation::Error)]
#[case::unsupported(Interpretation::Unsupported)]
#[case::deletion(Interpretation::Deletion)]
fn qualification_retains_scoped_predecessor_outcomes(#[case] interpretation: Interpretation) {
    let path = test_util::temp::temp_path("collection-qualification-scopes.h5");
    let mut fixture = Fixture::new();
    let address = fixture.collection(&[Object::new(7, b"before")]);
    let owner = object(&mut fixture, "data", address, 7);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase.reconstruction_source(&engine, owner).unwrap();
    let mut candidates = phase.deleted_collections(&[owner]);
    let original = candidates
        .observations
        .as_ref()
        .unwrap()
        .predecessors
        .values()
        .next()
        .unwrap()
        .clone();
    let attribute = CollectionProvenance {
        object: owner,
        content: CollectionContent::Attribute("label".into()),
    };
    let values = CollectionProvenance {
        object: owner,
        content: CollectionContent::DatasetValues,
    };
    candidates
        .observe(&mut phase, attribute.clone(), original.clone())
        .unwrap();
    let mut references = original;
    if !matches!(interpretation, Interpretation::Deletion) {
        references.complete = false;
    }
    if matches!(interpretation, Interpretation::Unsupported) {
        references.collections.clear();
        references.objects.clear();
    }
    if matches!(interpretation, Interpretation::Error) {
        references.error = Some(HeapReferenceError::Format(FormatError::Source(
            "predecessor unavailable".into(),
        )));
    }
    candidates
        .observe(&mut phase, values.clone(), references)
        .unwrap();
    if matches!(interpretation, Interpretation::Deletion) {
        candidates
            .require_phase(&mut phase)
            .unwrap()
            .incomplete_deletions
            .insert(owner);
    }
    let facts = phase.qualify_collections(&source, candidates).unwrap();
    let id = GlobalHeapId {
        collection: PhysicalAddress::new(address),
        index: index(7),
    };
    assert_eq!(facts.predecessors().unwrap().len(), 3);
    assert_eq!(facts.member(&attribute, id), Ok(()));
    assert_eq!(
        facts.member(&values, id),
        match interpretation {
            Interpretation::Unsupported => Err(MemberUnavailable::Observation),
            Interpretation::Deletion => Ok(()),
            _ => Err(MemberUnavailable::Interpretation),
        }
    );
    assert!(!facts.predecessors_complete());
    assert_eq!(
        facts.incomplete_deletions().unwrap().contains(&owner),
        matches!(interpretation, Interpretation::Deletion)
    );
    let outcome = &facts.predecessors().unwrap()[&values];
    assert_eq!(
        outcome.is_complete(),
        matches!(interpretation, Interpretation::Deletion)
    );
    if matches!(interpretation, Interpretation::Error) {
        let Some(HeapReferenceError::Format(error)) = outcome.error() else {
            panic!("expected predecessor source error");
        };
        assert_eq!(
            error,
            &FormatError::Source("predecessor unavailable".into())
        );
    } else {
        assert!(outcome.error().is_none());
    }
}

#[rstest]
#[case::phase(Association::Phase)]
#[case::session(Association::Session)]
#[case::same_root(Association::SameRoot)]
#[case::other_root(Association::OtherRoot)]
fn qualification_rejects_candidates_from_another_source(#[case] association: Association) {
    let path = test_util::temp::temp_path("collection-qualification-phase.h5");
    let mut fixture = Fixture::new();
    let address = fixture.collection(&[Object::new(7, b"before")]);
    let owner = object(&mut fixture, "data", address, 7);
    fixture.write(&path);
    let other_path = test_util::temp::temp_path("collection-qualification-session.h5");
    std::fs::copy(&path, &other_path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let other_engine = edit_test_util::open_engine(&other_path);
    let mut original = AnalysisPhase::new(&engine);
    let candidates = original.deleted_collections(&[owner]);
    let mut other = match association {
        Association::Session => AnalysisPhase::new(&other_engine),
        Association::Phase => AnalysisPhase::new(&engine),
        Association::SameRoot => {
            AnalysisPhase::prospective(&engine, PhysicalAddress::new(Fixture::ROOT))
        }
        Association::OtherRoot => AnalysisPhase::prospective(&engine, owner.header_address()),
    };
    let source = other
        .reconstruction_source(
            if matches!(association, Association::Session) {
                &other_engine
            } else {
                &engine
            },
            owner,
        )
        .unwrap();
    let error = other
        .qualify_collections(&source, candidates)
        .err()
        .unwrap();
    let Error::Format(error) = error else {
        panic!("expected phase association error");
    };
    assert_eq!(
        error,
        FormatError::Internal(
            if matches!(association, Association::Session) {
                "collection qualification belongs to another allocation session"
            } else {
                "collection qualification belongs to another source phase"
            }
            .into()
        )
    );
    assert!(other.claims.is_none());
    assert!(other.associations.is_none());
    let candidates = original.deleted_collections(&[owner]);
    let source = original.reconstruction_source(&engine, owner).unwrap();
    let facts = original.qualify_collections(&source, candidates).unwrap();
    assert_eq!(
        facts
            .contents(&mut other, PhysicalAddress::new(address))
            .err()
            .unwrap(),
        FormatError::Internal("heap query belongs to another source phase".into())
    );
    let Error::Format(error) = facts.associations(&mut other).err().unwrap() else {
        panic!("expected cache association error");
    };
    assert_eq!(
        error,
        FormatError::Internal("heap query belongs to another source phase".into())
    );
}

#[test]
fn empty_qualification_avoids_domain_and_numeric_payload_reads() {
    let path = test_util::temp::temp_path("collection-qualification-empty.h5");
    let mut fixture = Fixture::new();
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &crate::make_i32_type(),
        &vec![0x5a; 1_048_576],
    );
    let owner = ObjectId::new(PhysicalAddress::new(fixture.append_header(messages)));
    fixture.link("numeric", owner.header_address().get());
    fixture.write(&path);
    let (engine, calls) = edit_test_util::open_counted_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase.reconstruction_source(&engine, owner).unwrap();
    let before = calls.load(Ordering::Relaxed);
    let facts = phase
        .qualify_collections(&source, CollectionCandidates::default())
        .unwrap();
    assert!(facts.parents().is_empty());
    assert!(facts.domain().is_none());
    assert!(phase.claims.is_none());
    assert_eq!(calls.load(Ordering::Relaxed), before);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn nonempty_qualification_reuses_metadata_without_reading_or_retaining_payloads() {
    use std::sync::Arc;
    use std::sync::Mutex;

    use test_util::allocation;

    let path = test_util::temp::temp_path("collection-qualification-cost.h5");
    let mut fixture = Fixture::new();
    let payload = vec![0x5a; 1_048_576];
    let collection = fixture.collection(&[Object::new(7, &payload)]);
    let owner = object(&mut fixture, "variable", collection, 7);
    let numeric_start = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    let messages =
        Form::ContiguousDataset.messages(&mut fixture, &crate::make_i32_type(), &payload);
    fixture.object("numeric", messages);
    fixture.write(&path);
    let numeric = FileRange::new(
        PhysicalAddress::new(numeric_start),
        u64::try_from(payload.len()).unwrap(),
    )
    .unwrap();
    let reads = Arc::new(Mutex::new(Vec::new()));
    let engine = edit_test_util::open_observed_engine(&path, Arc::clone(&reads));
    reads.lock().unwrap().clear();
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase.reconstruction_source(&engine, owner).unwrap();
    let candidates = phase.deleted_collections(&[owner]);
    let (facts, first) = allocation::measure("collection_qualification_first", || {
        phase.qualify_collections(&source, candidates).unwrap()
    });
    assert!(
        first.blocks > 0
            && first.bytes > 0
            && first.bytes < 65_536
            && first.peak_bytes < 65_536
            && first.live_bytes > 0
            && first.live_bytes < 32_768,
        "{first}"
    );
    let collection = PhysicalAddress::new(collection);
    assert_eq!(facts.whole_parent(collection), Ok(()));
    assert!(facts.predecessors_complete());
    let origin = CollectionProvenance {
        object: owner,
        content: CollectionContent::Object,
    };
    let id = GlobalHeapId {
        collection,
        index: index(7),
    };
    assert_eq!(facts.member(&origin, id), Ok(()));
    let contents = facts.contents(&mut phase, collection).unwrap();
    assert_eq!(contents.members().len(), 1);
    let heap_payload = contents.members()[0].payload().range();
    assert_eq!(heap_payload.length(), numeric.length());
    let before = reads.lock().unwrap().len();
    assert!(before > 0);
    assert!(
        reads
            .lock()
            .unwrap()
            .iter()
            .all(|range| !range.overlaps(numeric) && !range.overlaps(heap_payload))
    );
    let framing = core::ptr::from_ref(contents);
    let associations = core::ptr::from_ref(facts.associations(&mut phase).unwrap());
    let repeated = phase.deleted_collections(&[owner]);
    let (cached, measured) = allocation::measure("collection_qualification_cached", || {
        phase.qualify_collections(&source, repeated).unwrap()
    });
    assert!(
        measured.blocks > 0
            && measured.bytes > 0
            && measured.bytes < 8_192
            && measured.peak_bytes < 8_192
            && measured.live_bytes > 0
            && measured.live_bytes < 8_192,
        "{measured}"
    );
    assert_eq!(cached.member(&origin, id), Ok(()));
    assert_eq!(cached.whole_parent(collection), Ok(()));
    let ((), access) = allocation::measure("collection_qualification_cache_access", || {
        assert_eq!(
            core::ptr::from_ref(cached.contents(&mut phase, collection).unwrap()),
            framing
        );
        assert_eq!(
            core::ptr::from_ref(cached.associations(&mut phase).unwrap()),
            associations
        );
    });
    let (control, floor) = allocation::measure("collection_qualification_control", || {
        vec![0_u8; payload.len()]
    });
    assert_eq!(control.len(), payload.len());
    assert!(
        floor.bytes >= u64::try_from(control.len()).unwrap(),
        "{floor}"
    );
    assert_eq!((access.blocks, access.bytes), (0, 0));
    assert_eq!(reads.lock().unwrap().len(), before);
}

#[rstest]
#[case::complete(StorageRestriction::None)]
#[case::unknown_class(StorageRestriction::Class)]
#[case::different_range(StorageRestriction::Range)]
#[case::framing_failure(StorageRestriction::Framing)]
#[case::absent_allocation(StorageRestriction::Absent)]
#[case::unavailable_domain(StorageRestriction::Domain)]
fn collection_storage_retains_independent_framing_and_allocation_outcomes(
    #[case] restriction: StorageRestriction,
) {
    let path = test_util::temp::temp_path("parent-storage-outcomes.h5");
    let mut fixture = Fixture::new();
    let address = fixture.collection(&[Object::new(7, b"before")]);
    let owner = object(&mut fixture, "owner", address, 7);
    fixture.write(&path);
    let (engine, calls) = edit_test_util::open_counted_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase.reconstruction_source(&engine, owner).unwrap();
    let mut candidates = phase.deleted_collections(&[owner]);
    let address = PhysicalAddress::new(address);
    let original = candidates.collections[&address].range;
    match restriction {
        StorageRestriction::Class => candidates.collections.get_mut(&address).unwrap().class = None,
        StorageRestriction::Range => {
            candidates.collections.get_mut(&address).unwrap().range =
                FileRange::new(address, original.length() + 8).unwrap()
        }
        StorageRestriction::Framing => {
            phase.allocations.insert(
                address,
                Err(FormatError::Source("unavailable collection framing".into())),
            );
        }
        StorageRestriction::Absent => {
            candidates.collections.remove(&address);
        }
        StorageRestriction::Domain => phase.topology = Some(None),
        StorageRestriction::None => {}
    }
    let before = (
        calls.load(Ordering::Relaxed),
        COLLECTION_ALLOCATIONS.with(Cell::get),
        COLLECTION_WORK.with(Cell::get),
    );
    let parents = phase.collection_storage(&source, &candidates).unwrap();
    assert_eq!(parents.len(), 1);
    let parent = &parents[&address];
    assert_eq!(
        parent.allocation,
        match restriction {
            StorageRestriction::Class => Err(ParentUnavailable::Class),
            StorageRestriction::Range => Err(ParentUnavailable::Range),
            StorageRestriction::Framing => Err(ParentUnavailable::Framing),
            StorageRestriction::Absent => Err(ParentUnavailable::Allocation),
            StorageRestriction::None | StorageRestriction::Domain => Ok(()),
        }
    );
    if matches!(restriction, StorageRestriction::Framing) {
        assert_eq!(
            parent.framing,
            Err(FormatError::Source("unavailable collection framing".into()))
        );
    } else {
        assert_eq!(parent.framing, Ok(original));
    }
    assert_eq!(
        parent.original,
        candidates
            .collections
            .get(&address)
            .map(|allocation| (allocation.range, allocation.class))
    );
    assert_eq!(
        (
            calls.load(Ordering::Relaxed),
            COLLECTION_ALLOCATIONS.with(Cell::get),
            COLLECTION_WORK.with(Cell::get)
        ),
        before
    );
}

#[derive(Clone, Copy, Debug)]
enum StorageRestriction {
    Absent,
    Class,
    Domain,
    Framing,
    None,
    Range,
}

fn object(fixture: &mut Fixture, name: &str, collection: u64, member: u32) -> ObjectId {
    let messages = Form::CompactDataset.messages(
        fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(6, collection, member),
    );
    let address = fixture.append_header(messages);
    fixture.link(name, address);
    ObjectId::new(PhysicalAddress::new(address))
}

fn index(value: u16) -> NonZeroU16 {
    NonZeroU16::new(value).unwrap()
}

#[derive(Clone, Copy)]
enum Restriction {
    Class,
    Domain,
    Framing,
    Missing,
    None,
    PeerFraming,
    Range,
    Synthetic,
    Unmatched,
}
#[derive(Clone, Copy)]
enum Interpretation {
    Deletion,
    Error,
    Partial,
    Unsupported,
}
#[derive(Clone, Copy)]
enum Association {
    OtherRoot,
    Phase,
    SameRoot,
    Session,
}
