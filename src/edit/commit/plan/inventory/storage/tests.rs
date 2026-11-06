use core::num::NonZeroU32;
use std::io::ErrorKind;

use hdf5_pure_core::__private::{ByteLength, FileRange};
use hdf5_pure_space::__private::{Extent, FreeClass, PageType, SessionSpace};
use rstest::rstest;
use test_util::disk_log;
use test_util::global_heap::Object as GlobalHeapObject;
use test_util::object_header::{Message, MessageType};
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use super::{
    PendingEditInventory, SpaceClaim, StorageAllocation, StorageApplyError, StorageApplyFailure,
    StorageConflict, StorageOrigin, StorageUnavailable,
};
use crate::chunked_read::{ChunkInfo, StoredChunkSize};
use crate::edit::WriteEngine;
use crate::edit::allocation::AllocationPublication;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::ownership::AllocationKind;
use crate::edit::commit::plan::inventory::{InventoryPreparation, PreparedEditInventory};
use crate::edit::retirement::{ChunkAllocation, RootRetirement};
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::image::WriteBuffering;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;
use crate::{AttrValue, FileBuilder, FileSpacePageSize, FileSpaceStrategy, PhysicalAddress};

mod acceptance;

#[rstest]
#[case::root_only(false)]
#[case::dataset_attribute(true)]
fn empty_counts_protect_original_payloads_even_for_reconstructed_owners(#[case] reconstruct: bool) {
    let path = test_util::temp::temp_path("retained-storage.h5");
    let mut fixture = Fixture::new();
    let raw = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    let records = Form::ContiguousDataset.messages(
        &mut fixture,
        &crate::type_builders::make_i32_type(),
        &19_i32.to_le_bytes(),
    );
    let header = fixture.append_header(records);
    let end = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    fixture.link("value", header);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    if reconstruct {
        engine
            .set_dataset_attr(&ObjectPathBuf::parse("value"), "a", AttrValue::I64(2))
            .unwrap();
    } else {
        engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    }
    let prepared = prepare(&mut engine);
    assert!(
        prepared
            .original()
            .facts()
            .unwrap()
            .counts()
            .changes()
            .is_empty()
    );
    assert!(prepared.storage().unavailable().is_none());
    let object = ObjectId::new(PhysicalAddress::new(header));
    let raw_allocation = StorageAllocation {
        range: range(raw, 4),
        origin: StorageOrigin::Owner {
            object,
            kind: AllocationKind::Data,
        },
    };
    let mut expected = vec![raw_allocation];
    if !reconstruct {
        expected.push(StorageAllocation {
            range: range(header, end - header),
            origin: StorageOrigin::Owner {
                object,
                kind: AllocationKind::Header,
            },
        });
    }
    assert_eq!(prepared.storage().allocations(), expected);
    let pending = match prepared.apply_excluding(&mut engine) {
        Ok(pending) => pending,
        Err(StorageApplyFailure::BeforePlacement { error, .. }) => {
            panic!("unexpected preplacement failure: {error:?}")
        }
        Err(StorageApplyFailure::Application(error)) => {
            panic!("unexpected application error: {error:?}")
        }
    };
    pending.exclusion().unwrap();
    assert_eq!(
        pending.maximum_end(),
        PhysicalAddress::new(if reconstruct { raw + 4 } else { end })
    );
    pending.check_end(pending.maximum_end()).unwrap();
    let required = pending.maximum_end();
    let proposed = PhysicalAddress::new(required.get() - 1);
    assert!(
        matches!(pending.check_end(proposed), Err(StorageConflict::End {
        proposed: actual, required: maximum
    }) if actual == proposed && maximum == required)
    );
    assert_eq!(
        pending.dispositions.replacements().unwrap(),
        &pending.relocations
    );
    assert!(pending.original.owners().is_some());
    assert_eq!(
        pending.engine.superblock().root_group_address.get(),
        Fixture::ROOT
    );
}

#[rstest]
#[case::conflict(true)]
#[case::adjacent(false)]
fn allocator_exclusion_precedes_actual_placement(#[case] conflict: bool) {
    let path = test_util::temp::temp_path("retained-reuse.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("value").with_i32_data(&[3]);
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    let prepared = prepare(&mut engine);
    let allocation = prepared.storage().allocations()[0];
    let start = if conflict {
        allocation.range.start()
    } else {
        PhysicalAddress::new(allocation.range.start().get() - 64)
    };
    let free = range(start.get(), 64);
    let extent = Extent::new(
        engine.addresses().relative(start).unwrap(),
        ByteLength::new(64),
    )
    .unwrap();
    engine.aggregation = None;
    engine.space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::FsmAggr),
        ByteLength::new(1),
        FileSpacePageSize::MIN,
        [(None, extent)],
    )
    .unwrap();
    let before = engine.disk_bytes().unwrap();
    let result = prepared.apply_excluding(&mut engine);
    if conflict {
        let StorageApplyFailure::BeforePlacement { prepared, error } = result.err().unwrap() else {
            panic!("expected a preplacement failure");
        };
        assert!(
            matches!(error, StorageApplyError::Conflict(StorageConflict::Overlap {
            allocation: actual, range, claim: SpaceClaim::Reusable,
        }) if actual == allocation && range == free)
        );
        assert_eq!(prepared.storage().allocations()[0], allocation);
        assert_eq!(engine.disk_bytes().unwrap(), before);
    } else {
        let pending = match result {
            Ok(pending) => pending,
            Err(StorageApplyFailure::BeforePlacement { error, .. }) => {
                panic!("unexpected preplacement failure: {error:?}")
            }
            Err(StorageApplyFailure::Application(error)) => {
                panic!("unexpected application error: {error:?}")
            }
        };
        pending.exclusion().unwrap();
        let (_, placed) = AnalysisPhase::prospective(pending.engine, pending.pending.root)
            .topology()
            .unwrap()
            .find_link(&ObjectPathBuf::parse("new"))
            .unwrap();
        assert!(placed.header_address() >= start);
        assert!(placed.header_address() < free.end());
    }
}

#[rstest]
#[case::ordinary(false)]
#[case::deferred_chunk(true)]
fn actual_pending_retirement_checks_complete_boundaries_and_classes(#[case] deferred: bool) {
    let path = test_util::temp::temp_path("retained-retirement.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("value").with_i32_data(&[3, 5]);
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    let prepared = prepare(&mut engine);
    let allocation = prepared.storage().allocations()[0];
    let mut applied = prepared.apply(&mut engine).unwrap();
    let class = FreeClass::Dead;
    if deferred {
        let chunk = ChunkInfo {
            address: engine
                .addresses()
                .relative(allocation.range.start())
                .unwrap(),
            chunk_size: StoredChunkSize::v4(allocation.range.length()),
            filter_mask: 0,
            offsets: vec![0],
        };
        let chunk = ChunkAllocation::capture(engine.addresses(), &chunk).unwrap();
        applied.tree.retirement = RootRetirement::new(applied.tree.retirement.root(), vec![chunk]);
    } else {
        applied.tree.freed.push((allocation.range, class));
    }
    let pending = PendingEditInventory::new(&mut engine, applied);
    let expected_class = if deferred {
        FreeClass::Page(PageType::Raw)
    } else {
        class
    };
    assert!(matches!(pending.exclusion(), Err(StorageConflict::Overlap {
        allocation: actual, range, claim: SpaceClaim::Retirement(actual_class),
    }) if *actual == allocation && *range == allocation.range && *actual_class == expected_class));
    assert!(
        pending
            .pending
            .batch
            .entries()
            .contains(&(allocation.range, expected_class))
    );
    assert_eq!(
        pending.dispositions.replacements().unwrap(),
        &pending.relocations
    );
}

#[rstest]
#[case::surviving_member(ParentRetirement::Retain)]
#[case::superseded_heap(ParentRetirement::Superseded)]
#[case::candidate_free(ParentRetirement::Candidate)]
fn candidate_parent_capacity_is_checked_against_actual_retirement(
    #[case] retire: ParentRetirement,
) {
    let path = test_util::temp::temp_path("retained-parent.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[GlobalHeapObject::new(7, b"before")]);
    let parent_end = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    let datatype = global_heap_fixture::vlen_string();
    let descriptor = global_heap_fixture::reference(6, parent, 7);
    let mut records = Form::CompactDataset.messages(
        &mut fixture,
        &crate::type_builders::make_i32_type(),
        &1_i32.to_le_bytes(),
    );
    records.extend(Form::CompactAttribute.messages(&mut fixture, &datatype, &descriptor));
    fixture.object("owner", records);
    if !matches!(retire, ParentRetirement::Candidate) {
        let records = Form::CompactAttribute.messages(&mut fixture, &datatype, &descriptor);
        fixture.object("survivor", records);
    }
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_dataset_attr(&ObjectPathBuf::parse("owner"), "a", AttrValue::I64(9))
        .unwrap();
    let prepared = prepare(&mut engine);
    assert!(prepared.storage().unavailable().is_none());
    let parent = PhysicalAddress::new(parent);
    let storage = &prepared.storage().parents().unwrap()[&parent];
    assert_eq!(
        storage.framing.as_ref().copied().unwrap(),
        range(parent.get(), parent_end - parent.get())
    );
    assert_eq!(storage.allocation, Ok(()));
    let parent_range = range(parent.get(), parent_end - parent.get());
    assert_eq!(
        storage.original,
        Some((parent_range, Some(FreeClass::Page(PageType::Raw))))
    );
    assert!(parent_range.length() > u64::try_from(descriptor.len() + 6).unwrap());
    assert!(prepared.storage().allocations().iter().any(|allocation| {
        allocation.range == range(parent.get(), parent_end-parent.get())
            && matches!(allocation.origin, StorageOrigin::Parent { address, .. } if address == parent)
    }));
    if matches!(retire, ParentRetirement::Superseded) {
        engine
            .superseded_heaps
            .push(range(parent.get(), parent_end - parent.get()));
    }
    let pending = prepared.apply_excluding(&mut engine).ok().unwrap();
    if matches!(retire, ParentRetirement::Retain) {
        pending.exclusion().unwrap();
        return;
    }
    assert!(matches!(pending.exclusion(), Err(StorageConflict::Overlap {
        claim: SpaceClaim::Retirement(FreeClass::Page(PageType::Raw)), allocation, range,
    }) if *range == parent_range && matches!(allocation.origin, StorageOrigin::Parent { address, .. } if address == parent)));
    assert!(
        pending
            .pending
            .batch
            .entries()
            .contains(&(parent_range, FreeClass::Page(PageType::Raw)))
    );
}

#[test]
fn failure_after_placement_preserves_application_error_ownership() {
    let path = test_util::temp::temp_path("retained-failure.h5");
    let mut builder = FileBuilder::new();
    builder.create_group_with("g", |_| Ok(())).unwrap();
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_group_attr(&ObjectPathBuf::parse("g"), "a", AttrValue::I64(1))
        .unwrap();
    let prepared = prepare(&mut engine);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    disk_log::start();
    disk_log::fail_write_after(1, 0);
    let failure = prepared.apply_excluding(&mut engine).err().unwrap();
    let recording = disk_log::finish();
    let StorageApplyFailure::Application(Error::Io(error)) = failure else {
        panic!("expected the application I/O failure");
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
}

#[rstest]
#[case::different_phase(Restriction::Phase)]
#[case::different_session(Restriction::Session)]
#[case::different_root(Restriction::Root)]
#[case::incomplete_geometry(Restriction::Geometry)]
#[case::excess_root_count(Restriction::Excess)]
fn unavailable_or_mismatched_original_storage_withholds_placement(#[case] mode: Restriction) {
    let path = test_util::temp::temp_path("retained-unavailable.h5");
    let mut fixture = if matches!(mode, Restriction::Excess) {
        Fixture::new().with_root_reference_count(NonZeroU32::new(2).unwrap())
    } else {
        Fixture::new()
    };
    fixture.object(
        "owner",
        if matches!(mode, Restriction::Geometry) {
            vec![Message::new(MessageType(0x7f), &[])]
        } else {
            vec![]
        },
    );
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine.create_group(&ObjectPathBuf::parse("first")).unwrap();
    let mut prepared = prepare(&mut engine);
    if matches!(mode, Restriction::Phase) {
        engine
            .create_group(&ObjectPathBuf::parse("second"))
            .unwrap();
        prepared.storage = prepare(&mut engine).storage;
    }
    match mode {
        Restriction::Session => engine.context.allocations = AllocationPublication::new(),
        Restriction::Root => {
            let StorageOrigin::Owner { object, .. } = prepared.storage().allocations()[0].origin
            else {
                panic!("expected retained owner storage");
            };
            let mut root = engine.superblock().clone();
            root.root_group_address = engine
                .addresses()
                .relative(object.header_address())
                .unwrap();
            engine
                .context
                .adopt(engine.context.propose(root).unwrap())
                .unwrap();
        }
        _ => {}
    }
    let before = engine.disk_bytes().unwrap();
    let StorageApplyFailure::BeforePlacement { prepared, error } =
        prepared.apply_excluding(&mut engine).err().unwrap()
    else {
        panic!("expected a preplacement failure");
    };
    match mode {
        Restriction::Phase => assert!(matches!(error, StorageApplyError::Phase), "{:?}", error),
        Restriction::Session => assert!(matches!(error, StorageApplyError::Session), "{error:?}"),
        Restriction::Root => assert!(matches!(error, StorageApplyError::Root), "{error:?}"),
        Restriction::Geometry | Restriction::Excess => {
            assert!(
                matches!(error, StorageApplyError::Unavailable),
                "{:?}",
                error
            );
            let original = prepared;
            assert!(!original.storage().allocations().is_empty());
            assert!(
                matches!(
                    original.storage().unavailable(),
                    Some(StorageUnavailable::Owners)
                ) && matches!(mode, Restriction::Geometry)
                    || matches!(
                        original.storage().unavailable(),
                        Some(StorageUnavailable::Dispositions)
                    ) && matches!(mode, Restriction::Excess)
            );
        }
    }
    assert_eq!(engine.disk_bytes().unwrap(), before);
}

#[derive(Clone, Copy, Debug)]
enum ParentRetirement {
    Candidate,
    Retain,
    Superseded,
}

#[derive(Clone, Copy, Debug)]
enum Restriction {
    Excess,
    Geometry,
    Phase,
    Root,
    Session,
}

fn prepare(engine: &mut WriteEngine) -> PreparedEditInventory {
    let staged = std::mem::take(&mut engine.staged);
    match PreparedEditInventory::prepare(engine, staged) {
        InventoryPreparation::Tree(prepared) => prepared,
        InventoryPreparation::Rejected { error, .. } => {
            panic!("unexpected preparation error: {error:?}")
        }
        InventoryPreparation::InPlace(_) => panic!("expected a tree preparation"),
    }
}

fn range(start: u64, length: u64) -> FileRange {
    FileRange::new(PhysicalAddress::new(start), length).unwrap()
}
