use std::collections::BTreeMap;
use std::fs;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::PageType;
use rstest::rstest;
use test_util::data_layout;
use test_util::dataspace;
use test_util::global_heap::Object;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use super::super::OriginalStorage;
use super::super::SpaceClaim;
use super::super::StorageAllocation;
use super::super::StorageConflict;
use super::super::StorageOrigin;
use super::super::StorageUnavailable;
use super::super::tests as storage_tests;
use super::StorageIntegrityError;
use crate::FileBuilder;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::heap::qualification::ParentStorage;
use crate::edit::analysis::heap::qualification::ParentUnavailable;
use crate::edit::analysis::ownership::AllocationKind;
use crate::edit::analysis::physical::claims::ClaimOrigin;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::error::FormatError;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::framed(Some((100, 40)), None, Some((100, 40)))]
#[case::original(None, Some((100, 40)), Some((100, 40)))]
#[case::equal(Some((100, 40)), Some((100, 40)), Some((100, 40)))]
#[case::larger_original(Some((100, 40)), Some((100, 60)), Some((100, 60)))]
#[case::larger_framing(Some((100, 60)), Some((100, 40)), Some((100, 60)))]
#[case::different_starts(Some((110, 30)), Some((100, 40)), Some((100, 40)))]
#[case::unavailable(None, None, None)]
fn parent_capture_preserves_known_extents_without_removal_evidence(
    #[case] framing: Option<(u64, u64)>,
    #[case] original: Option<(u64, u64)>,
    #[case] expected: Option<(u64, u64)>,
) {
    let address = PhysicalAddress::new(100);
    let mut storage = OriginalStorage::from_test_ranges(&[]);
    storage.parents = Ok(BTreeMap::from([(
        address,
        ParentStorage {
            framing: framing
                .map(|(start, len)| range(start, len))
                .ok_or(FormatError::InvalidGlobalHeapSignature),
            original: original
                .map(|(start, len)| (range(start, len), Some(FreeClass::Page(PageType::Raw)))),
            allocation: Err(ParentUnavailable::Qualification),
        },
    )]));
    storage.capture_parents();
    assert_eq!(
        storage.allocations(),
        expected
            .map(|(start, len)| StorageAllocation {
                range: range(start, len),
                origin: StorageOrigin::Parent {
                    address,
                    class: original.map(|_| FreeClass::Page(PageType::Raw))
                },
            })
            .into_iter()
            .collect::<Vec<_>>()
    );
    assert!(
        matches!(storage.unavailable(), Some(StorageUnavailable::Parent { address: actual, reason: ParentUnavailable::Qualification }) if *actual == address)
    );
    assert_eq!(storage.integrity(), Ok(()));
}

#[rstest]
#[case::empty(&[], &[(100, 100)])]
#[case::inside(&[(120, 20), (160, 20)], &[(100, 20), (140, 20), (180, 20)])]
#[case::adjacent(&[(80, 20), (200, 20)], &[(100, 100)])]
#[case::covering(&[(80, 140)], &[])]
#[case::crossing_left(&[(80, 40)], &[(120, 80)])]
#[case::crossing_right(&[(180, 40)], &[(100, 80)])]
fn unobserved_intervals_exclude_complete_recorded_allocations(
    #[case] known: &[(u64, u64)],
    #[case] expected: &[(u64, u64)],
) {
    let known = known
        .iter()
        .map(|&(start, len)| range(start, len))
        .collect::<Vec<_>>();
    let mut storage = OriginalStorage::from_test_ranges(&known);
    storage.capture_unobserved(&[range(100, 100)]).unwrap();
    assert_eq!(
        storage
            .allocations()
            .iter()
            .filter(|allocation| allocation.origin == StorageOrigin::Unobserved)
            .map(|allocation| (allocation.range.start().get(), allocation.range.length()))
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(
        storage
            .allocations()
            .iter()
            .filter(|allocation| allocation.origin != StorageOrigin::Unobserved)
            .map(|allocation| allocation.range)
            .collect::<Vec<_>>(),
        known
    );
    assert!(
        storage
            .allocations()
            .windows(2)
            .all(|pair| !pair[0].range.overlaps(pair[1].range))
    );
}

#[rstest]
#[case::empty(&[], None)]
#[case::adjacent(&[(100, 20), (120, 20)], None)]
#[case::disjoint(&[(100, 20), (180, 20)], None)]
#[case::overlap(&[(100, 40), (120, 20)], Some(Geometry::Overlap))]
#[case::duplicate(&[(100, 40), (100, 40)], Some(Geometry::Overlap))]
#[case::bounds(&[(180, 24)], Some(Geometry::Bounds))]
fn recorded_protection_requires_disjoint_complete_in_bounds_ranges(
    #[case] input: &[(u64, u64)],
    #[case] expected: Option<Geometry>,
) {
    let ranges = input
        .iter()
        .map(|&(start, len)| range(start, len))
        .collect::<Vec<_>>();
    let storage = OriginalStorage::from_test_ranges(&ranges);
    let result = storage.protected_geometry(PhysicalAddress::new(200));
    match (result, expected) {
        (Ok(()), None) => {}
        (
            Err(StorageIntegrityError::ProtectionBounds {
                allocation: actual,
                end,
            }),
            Some(Geometry::Bounds),
        ) => {
            assert_eq!(actual, allocation(180, 24));
            assert_eq!(end.get(), 200);
        }
        (
            Err(StorageIntegrityError::ConflictingProtection { first, second }),
            Some(Geometry::Overlap),
        ) => {
            assert_eq!(first, allocation(100, 40));
            assert_eq!(
                second,
                if input[1].0 == 100 {
                    allocation(100, 40)
                } else {
                    allocation(120, 20)
                }
            );
        }
        (actual, _) => panic!("unexpected protected geometry: {actual:?}"),
    }
}

#[rstest]
#[case::overlap(Integrity::Overlap)]
#[case::outside_source(Integrity::Source)]
#[case::overflow(Integrity::Overflow)]
#[case::local_overlap(Integrity::LocalOverlap)]
#[case::outside_allocation(Integrity::Allocation)]
fn original_integrity_failures_remain_distinct_from_optional_qualification(
    #[case] mode: Integrity,
) {
    let path = test_util::temp::temp_path("original-storage-integrity.h5");
    let mut fixture = Fixture::new();
    let raw = match mode {
        Integrity::Overlap => u64::try_from(fixture.image.append(&11_i32.to_le_bytes())).unwrap(),
        Integrity::LocalOverlap => u64::try_from(fixture.image.as_bytes().len()).unwrap(),
        Integrity::Overflow => u64::MAX - 2,
        Integrity::Allocation | Integrity::Source => 10_000,
    };
    let mut messages =
        global_heap_fixture::dataset_messages(&crate::make_i32_type(), &dataspace::scalar());
    messages.push(Message::new(
        MessageType::DATA_LAYOUT,
        &data_layout::contiguous(raw, 4),
    ));
    let first = fixture.append_header(messages.clone());
    fixture.link("first", first);
    let second = matches!(mode, Integrity::Overlap).then(|| {
        let second = fixture.append_header(messages);
        fixture.link("second", second);
        second
    });
    fixture.write(&path);
    let original_end = fs::metadata(&path).unwrap().len();
    if matches!(mode, Integrity::Allocation) {
        fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(20_000)
            .unwrap();
    }
    let mut engine = edit_test_util::open_engine(&path);
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    let prepared = storage_tests::prepare(&mut engine);
    let error = prepared.storage().integrity().unwrap_err();
    match (mode, error.clone()) {
        (
            Integrity::Overlap,
            StorageIntegrityError::ConflictingClaims {
                first: left,
                second: right,
            },
        ) => {
            assert_eq!(left.range(), range(raw, 4));
            assert_eq!(right.range(), range(raw, 4));
            assert_eq!(
                left.origin(),
                ClaimOrigin::Object {
                    object: ObjectId::new(PhysicalAddress::new(first)),
                    kind: AllocationKind::Data
                }
            );
            assert_eq!(
                right.origin(),
                ClaimOrigin::Object {
                    object: ObjectId::new(PhysicalAddress::new(second.unwrap())),
                    kind: AllocationKind::Data
                }
            );
        }
        (Integrity::Source, StorageIntegrityError::ObjectFormat { object, error }) => {
            assert_eq!(object, ObjectId::new(PhysicalAddress::new(first)));
            assert_eq!(
                *error,
                FormatError::UnexpectedSourceEof {
                    expected: raw + 4,
                    available: original_end
                }
            );
        }
        (Integrity::Overflow, StorageIntegrityError::ObjectFormat { object, error }) => {
            assert_eq!(object, ObjectId::new(PhysicalAddress::new(first)));
            assert_eq!(
                *error,
                FormatError::OffsetOverflow {
                    offset: raw,
                    length: 4
                }
            );
        }
        (
            Integrity::LocalOverlap,
            StorageIntegrityError::ObjectExtent {
                object,
                address,
                length,
                previous_end,
                logical_size,
            },
        ) => {
            assert_eq!(object, ObjectId::new(PhysicalAddress::new(first)));
            assert_eq!(
                (address, length, previous_end, logical_size),
                (raw, 4, original_end, original_end)
            );
        }
        (Integrity::Allocation, StorageIntegrityError::BeyondAllocation { claim, end }) => {
            assert_eq!(claim.range(), range(raw, 4));
            assert_eq!(
                claim.origin(),
                ClaimOrigin::Object {
                    object: ObjectId::new(PhysicalAddress::new(first)),
                    kind: AllocationKind::Data
                }
            );
            assert_eq!(end.get(), original_end);
        }
        (_, error) => panic!("unexpected original integrity failure: {error:?}"),
    }
    assert!(matches!(
        prepared.storage().unavailable(),
        Some(StorageUnavailable::Owners)
    ));
    let actual = prepared.storage().exclude_space(&engine).unwrap_err();
    assert!(
        matches!(&actual, StorageConflict::Original(actual) if *actual == error),
        "{actual:?}"
    );
}

#[rstest]
#[case::reusable(SpaceClaim::Reusable)]
#[case::recycle(SpaceClaim::Recycle)]
#[case::retirement(SpaceClaim::Retirement(FreeClass::Page(PageType::Raw)))]
#[case::aggregate(SpaceClaim::Aggregate)]
#[case::root_copy(SpaceClaim::RootCopy)]
#[case::manager(SpaceClaim::OldManager)]
fn unknown_source_bytes_remain_protected_without_heap_candidates(#[case] claim: SpaceClaim) {
    let path = test_util::temp::temp_path("unobserved-source-storage.h5");
    let mut fixture = Fixture::new();
    let opaque = u64::try_from(fixture.image.append(b"unobserved source")).unwrap();
    fixture.object("owner", vec![Message::new(MessageType(0x7f), &[])]);
    fixture.write(&path);
    let end = fs::metadata(&path).unwrap().len();
    let mut engine = edit_test_util::open_engine(&path);
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    let prepared = storage_tests::prepare(&mut engine);
    let storage = prepared.storage();
    assert_eq!(storage.integrity(), Ok(()));
    assert!(prepared.collections().addresses().is_empty());
    let checked = range(opaque + 1, 1);
    let allocation = *storage
        .allocations()
        .iter()
        .find(|allocation| allocation.range.contains(checked))
        .unwrap();
    assert_eq!(allocation.origin, StorageOrigin::Unobserved);
    let error = storage.exclude(checked, claim).unwrap_err();
    assert!(
        matches!(error, StorageConflict::Overlap { allocation: actual, range, claim: actual_claim } if actual == allocation && range == checked && actual_claim == claim),
        "{error:?}"
    );
    let error = storage
        .check_end(PhysicalAddress::new(end - 1))
        .unwrap_err();
    assert!(
        matches!(error, StorageConflict::End { proposed, required } if proposed.get() == end - 1 && required.get() == end),
        "{error:?}"
    );
}

#[test]
fn unchanged_heap_parent_is_protected_without_reclamation_candidates() {
    let path = test_util::temp::temp_path("unchanged-parent-storage.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"retained")]);
    let messages = Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(8, parent, 7),
    );
    fixture.object("owner", messages);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    let prepared = storage_tests::prepare(&mut engine);
    assert!(prepared.collections().addresses().is_empty());
    assert!(prepared.storage().parents().unwrap().is_empty());
    assert_eq!(
        prepared
            .storage()
            .allocations()
            .iter()
            .filter(|allocation| matches!(allocation.origin, StorageOrigin::Parent { .. }))
            .copied()
            .collect::<Vec<_>>(),
        [StorageAllocation {
            range: range(parent, 4096),
            origin: StorageOrigin::Parent {
                address: PhysicalAddress::new(parent),
                class: None
            }
        }]
    );
    assert_eq!(prepared.storage().integrity(), Ok(()));
}

#[test]
fn failed_source_observation_preserves_its_error_over_optional_unavailability() {
    let path = test_util::temp::temp_path("capture-source-association.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("data").with_i32_data(&[7]);
    builder.write(&path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let mut first = AnalysisPhase::new(&engine);
    let root =
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap();
    let source = first.reconstruction_source(&engine, root).unwrap();
    let mut second = AnalysisPhase::new(&engine);
    let mut storage = OriginalStorage::from_test_ranges(&[]);
    storage.unavailable = Some(StorageUnavailable::Owners);
    storage.capture_integrity(&engine, &mut second, &source);
    assert_eq!(storage.integrity(), Err(StorageIntegrityError::Unavailable));
    let Some(StorageUnavailable::Coverage(Error::Format(FormatError::Internal(message)))) =
        storage.unavailable()
    else {
        panic!(
            "expected the source association failure, got {:?}",
            storage.unavailable()
        );
    };
    assert_eq!(message, "physical claims belong to another source phase");
    assert_eq!(storage.allocations(), []);
}

fn allocation(start: u64, length: u64) -> StorageAllocation {
    StorageAllocation {
        range: range(start, length),
        origin: StorageOrigin::Owner {
            object: ObjectId::new(PhysicalAddress::new(start)),
            kind: AllocationKind::Data,
        },
    }
}

#[derive(Clone, Copy)]
enum Geometry {
    Bounds,
    Overlap,
}

fn range(start: u64, length: u64) -> FileRange {
    FileRange::new(PhysicalAddress::new(start), length).unwrap()
}

#[derive(Clone, Copy)]
enum Integrity {
    Allocation,
    LocalOverlap,
    Overflow,
    Overlap,
    Source,
}
