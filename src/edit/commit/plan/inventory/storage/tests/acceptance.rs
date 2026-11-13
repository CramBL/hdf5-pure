use std::fs;

use hdf5_pure_core::__private::{ByteLength, NonZeroByteLength};
use hdf5_pure_space::__private::{
    AllocationClass, Extent, FreeClass, ManagerKind, PageTransition, PageType, RetainedKind,
    SessionSpace,
};
use rstest::rstest;
use test_util::btree::v2 as btree_v2;
use test_util::bytes;
use test_util::dense_attributes;
use test_util::group_info;
use test_util::object_header::Message;
use test_util::object_header::v2::{self, Header, HeaderFlags};
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use crate::UserblockSize;
use crate::address::{StoredAddress, StoredAddressExt};
use crate::edit::aggregation::AggregateSpace;
use crate::edit::analysis::ownership::AllocationKind;
use crate::edit::commit::plan::inventory::storage::{
    SpaceClaim, StorageApplyError, StorageApplyFailure, StorageConflict, StorageOrigin,
};
use crate::edit::test_util as edit_test_util;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;
use crate::{FileBuilder, FileSpacePageSize, FileSpaceStrategy, PhysicalAddress};

#[test]
fn retained_group_descendants_preserve_complete_continuation_and_payload_allocations() {
    let path = test_util::temp::temp_path("retained-descendants.h5");
    let mut fixture = Fixture::new();
    let raw = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    let records = Form::ContiguousDataset.messages(
        &mut fixture,
        &crate::type_builders::make_i32_type(),
        &31_i32.to_le_bytes(),
    );
    let child = fixture.append_header(records);
    let child_end = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    let continuation = v2::continuation_chunk(
        &[
            global_heap_fixture::hard_link("child", child),
            Message::nil(80),
        ],
        HeaderFlags::default(),
    );
    let continued = fixture.image.append(&continuation);
    let header = Header::new()
        .messages(group_info::compact_messages(&[]))
        .continuation(continued, &continuation, Widths::EIGHT)
        .build();
    let group = u64::try_from(fixture.image.append(&header)).unwrap();
    fixture.link("retained", group);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    let prepared = super::prepare(&mut engine);
    assert!(prepared.storage().unavailable().is_none());
    assert_eq!(
        prepared
            .storage()
            .allocations()
            .iter()
            .map(|allocation| { (allocation.range, allocation.origin) })
            .collect::<Vec<_>>(),
        [
            (
                super::range(raw, 4),
                StorageOrigin::Owner {
                    object: ObjectId::new(PhysicalAddress::new(child)),
                    kind: AllocationKind::Data
                }
            ),
            (
                super::range(child, child_end - child),
                StorageOrigin::Owner {
                    object: ObjectId::new(PhysicalAddress::new(child)),
                    kind: AllocationKind::Header
                }
            ),
            (
                super::range(
                    u64::try_from(continued).unwrap(),
                    u64::try_from(continuation.len()).unwrap()
                ),
                StorageOrigin::Owner {
                    object: ObjectId::new(PhysicalAddress::new(group)),
                    kind: AllocationKind::Header
                }
            ),
            (
                super::range(group, u64::try_from(header.len()).unwrap()),
                StorageOrigin::Owner {
                    object: ObjectId::new(PhysicalAddress::new(group)),
                    kind: AllocationKind::Header
                }
            ),
        ]
    );
    prepared
        .apply_excluding(&mut engine)
        .ok()
        .unwrap()
        .exclusion()
        .unwrap();
}

#[test]
fn retained_dense_storage_includes_full_heap_and_index_capacity() {
    let path = test_util::temp::temp_path("retained-dense.h5");
    let image = dense_attributes::dense_attrs();
    let header = bytes::signature_offsets(&image, v2::SIGNATURE);
    let heap = bytes::sole_signature(&image, test_util::fractal_heap::SIGNATURE);
    let block = bytes::sole_signature(&image, test_util::fractal_heap::DIRECT_BLOCK_SIGNATURE);
    let leaf = bytes::sole_signature(&image, btree_v2::LEAF_SIGNATURE);
    let index = bytes::sole_signature(&image, btree_v2::SIGNATURE);
    assert_eq!(header.len(), 2);
    fs::write(&path, &image).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    let prepared = super::prepare(&mut engine);
    assert!(prepared.storage().unavailable().is_none());
    let raw = header[0] - 24;
    let expected = [
        (heap, block - heap, AllocationKind::Metadata),
        (block, leaf - block, AllocationKind::Metadata),
        (leaf, index - leaf, AllocationKind::Metadata),
        (index, raw - index, AllocationKind::Metadata),
        (raw, 24, AllocationKind::Data),
        (header[0], header[1] - header[0], AllocationKind::Header),
    ];
    assert_eq!(
        prepared
            .storage()
            .allocations()
            .iter()
            .map(|allocation| { (allocation.range, allocation.origin) })
            .collect::<Vec<_>>(),
        expected.map(|(start, len, kind)| (
            super::range(u64::try_from(start).unwrap(), u64::try_from(len).unwrap()),
            StorageOrigin::Owner {
                object: ObjectId::new(PhysicalAddress::new(u64::try_from(header[0]).unwrap())),
                kind
            },
        ))
    );
    prepared
        .apply_excluding(&mut engine)
        .ok()
        .unwrap()
        .exclusion()
        .unwrap();
}

#[rstest]
#[case::typed_metadata(Claim::Typed(PageType::Meta))]
#[case::typed_raw(Claim::Typed(PageType::Raw))]
#[case::unclassified(Claim::Retained(RetainedKind::Unclassified))]
#[case::dead(Claim::Retained(RetainedKind::Dead))]
#[case::page_padding(Claim::Retained(RetainedKind::Padding(PageType::Meta)))]
#[case::metadata_aggregate(Claim::Aggregate(AllocationClass::Metadata))]
#[case::raw_aggregate(Claim::Aggregate(AllocationClass::Raw))]
fn allocator_claims_use_physical_coordinates_before_placement(
    #[case] claim: Claim,
    #[values(false, true)] conflict: bool,
) {
    let path = test_util::temp::temp_path("retained-allocator-claims.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::MIN);
    builder.create_dataset("value").with_i32_data(&[7]);
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    let prepared = super::prepare(&mut engine);
    let allocation = prepared.storage().allocations()[0];
    let start = if conflict {
        allocation.range.start()
    } else {
        prepared.storage().allocations().last().unwrap().range.end()
    };
    let extent = Extent::new(
        engine.addresses().relative(start).unwrap(),
        ByteLength::new(1),
    )
    .unwrap();
    assert_eq!(start.get() - extent.start().get(), UserblockSize::MIN.get());
    engine.aggregation = None;
    engine.space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        ByteLength::new(1),
        FileSpacePageSize::MIN,
        match claim {
            Claim::Typed(PageType::Meta) => Some((Some(ManagerKind::PagedSmallMetadata), extent)),
            Claim::Typed(PageType::Raw) => Some((Some(ManagerKind::PagedSmallRaw), extent)),
            Claim::Retained(RetainedKind::Unclassified) => Some((None, extent)),
            _ => None,
        },
    )
    .unwrap();
    match claim {
        Claim::Retained(RetainedKind::Dead) => {
            engine
                .space
                .record_freed(extent.end(), extent, FreeClass::Dead);
        }
        Claim::Retained(RetainedKind::Padding(page_type)) => {
            engine.space.record_page_padding(PageTransition::Pad {
                extent,
                track_as: Some(page_type),
            })
        }
        Claim::Aggregate(class) => {
            let mut aggregate = AggregateSpace::new(StoredAddress::new(extent.start().get() - 1));
            aggregate
                .reserve_with(
                    engine.widths().offsets,
                    engine.addresses(),
                    NonZeroByteLength::new(1).unwrap(),
                    class,
                    |_, _| panic!("a new aggregate has no abandoned capacity"),
                )
                .unwrap();
            engine.aggregation = Some(aggregate);
        }
        _ => {}
    }
    let expected_extent = match claim {
        Claim::Aggregate(_) => engine.aggregation.unwrap().unused().next().unwrap(),
        Claim::Typed(_) | Claim::Retained(_) => extent,
    };
    let expected_range = super::range(
        engine
            .addresses()
            .resolve(expected_extent.start())
            .unwrap()
            .get(),
        expected_extent.len().get(),
    );
    let expected = match claim {
        Claim::Typed(_) => SpaceClaim::Reusable,
        Claim::Retained(kind) => SpaceClaim::Retained(kind),
        Claim::Aggregate(_) => SpaceClaim::Aggregate,
    };
    let bytes = engine.disk_bytes().unwrap();
    if conflict {
        let StorageApplyFailure::BeforePlacement { error, .. } =
            prepared.apply_excluding(&mut engine).err().unwrap()
        else {
            panic!("expected a preplacement conflict");
        };
        assert!(
            matches!(error, StorageApplyError::Conflict(StorageConflict::Overlap {
            allocation: actual, range, claim: actual_claim,
        }) if actual == allocation && actual_claim == expected && range == expected_range)
        );
        assert_eq!(engine.disk_bytes().unwrap(), bytes);
    } else {
        prepared.storage.exclude_space(&engine).unwrap();
        if let Claim::Typed(page_type) = claim {
            let class = match page_type {
                PageType::Meta => AllocationClass::Metadata,
                PageType::Raw => AllocationClass::Raw,
            };
            assert_eq!(
                engine
                    .space
                    .allocate_with_page_type(ByteLength::new(1), class, page_type),
                Some(extent)
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Claim {
    Aggregate(AllocationClass),
    Retained(RetainedKind),
    Typed(PageType),
}

#[test]
fn storage_capture_reuses_original_observations() {
    use core::cell::Cell;
    use std::sync::atomic::Ordering;

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    use test_util::allocation;
    use test_util::global_heap::Object as GlobalHeapObject;

    use crate::edit::analysis::AnalysisPhase;
    use crate::edit::analysis::ownership::OWNERSHIP_READS;
    use crate::edit::analysis::{COLLECTION_ALLOCATIONS, COLLECTION_WORK, DISCOVERY_WORK};
    use crate::edit::commit::plan::inventory::OriginalEdits;
    use crate::edit::commit::plan::inventory::dispositions::ReconstructionDispositions;
    use crate::edit::commit::plan::inventory::storage::OriginalStorage;
    use crate::edit::commit::plan::{CommitPlan, ReplacedHeader};

    let path = test_util::temp::temp_path("retained-storage-cached.h5");
    let payload = vec![0x2a; 1_048_576];
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[GlobalHeapObject::new(7, b"before")]);
    let mut records = Form::ContiguousDataset.messages(
        &mut fixture,
        &crate::type_builders::make_i32_type(),
        &payload,
    );
    records.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(6, parent, 7),
    ));
    fixture.object("owner", records);
    fixture.write(&path);
    let (mut engine, calls) = edit_test_util::open_counted_engine(&path);
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    let staged = std::mem::take(&mut engine.staged);
    let (CommitPlan::Tree(mut tree), _) =
        engine.plan_commit_with_inventory(&staged, false).unwrap()
    else {
        panic!("expected prepared tree edits");
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
    let reclamation =
        engine.prepare_reclamation(&tree.groups, std::mem::take(&mut tree.deleted_free));
    let before = (
        calls.load(Ordering::Relaxed),
        DISCOVERY_WORK.with(Cell::get),
        OWNERSHIP_READS.with(Cell::get),
        COLLECTION_WORK.with(Cell::get),
        COLLECTION_ALLOCATIONS.with(Cell::get),
    );
    assert!(before.0 > 0 && before.1.0 > 0 && before.2 > 0 && before.4 > 0);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let (storage, measured) = allocation::measure("prepared_retained_storage", || {
        OriginalStorage::collect(
            &engine,
            &mut phase,
            &tree,
            &original,
            &dispositions,
            &reclamation,
        )
    });
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let storage = OriginalStorage::collect(
        &engine,
        &mut phase,
        &tree,
        &original,
        &dispositions,
        &reclamation,
    );
    assert!(
        storage.unavailable().is_none(),
        "{:?}",
        storage.unavailable()
    );
    assert_eq!(storage.allocations().len(), 3);
    assert!(storage.allocations().iter().any(|allocation| allocation.range == super::range(parent, 4096) && matches!(allocation.origin, StorageOrigin::Parent { address, class: None } if address.get() == parent)));
    assert!(
        storage
            .allocations()
            .iter()
            .any(|allocation| allocation.range.length() == u64::try_from(payload.len()).unwrap())
    );
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    assert!(
        measured.blocks > 0 && measured.bytes > 0 && measured.bytes < 32_768,
        "{measured}"
    );
    assert_eq!(
        (
            calls.load(Ordering::Relaxed),
            DISCOVERY_WORK.with(Cell::get),
            OWNERSHIP_READS.with(Cell::get),
            COLLECTION_WORK.with(Cell::get),
            COLLECTION_ALLOCATIONS.with(Cell::get)
        ),
        before
    );
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let (copy, floor) =
            allocation::measure("retained_storage_payload_control", || payload.clone());
        assert_eq!(copy, payload);
        assert!(
            floor.bytes >= u64::try_from(payload.len()).unwrap(),
            "{floor}"
        );
    }
}

#[rstest]
#[case::deletion(Retirement::Deletion)]
#[case::dense_dataset(Retirement::DenseDataset)]
#[case::dense_root(Retirement::DenseRoot)]
#[case::dense_links(Retirement::DenseLinks)]
#[case::chunked_overwrite(Retirement::Chunked)]
#[case::appended_tail(Retirement::Append)]
fn actual_retirement_producers_preserve_original_boundaries_and_classes(
    #[case] retirement: Retirement,
) {
    use crate::edit::commit::plan::CommitPlan;
    use crate::edit::dataset::MovingWrite;
    use crate::{
        AppendBuilder, AttrValue, DatasetBuilder, DeflateLevel, LinkPhaseChange, MaxExtent,
    };

    let path = test_util::temp::temp_path("retained-actual-retirement.h5");
    match retirement {
        Retirement::DenseDataset => fs::write(&path, dense_attributes::dense_attrs()).unwrap(),
        Retirement::DenseRoot => fs::write(&path, dense_attributes::dense_attrs_root()).unwrap(),
        Retirement::DenseLinks => {
            let mut builder = FileBuilder::new();
            builder
                .create_group_with("g", |group| {
                    group.with_link_phase_change(LinkPhaseChange::AlwaysDense);
                    group.create_dataset("child").with_i32_data(&[7]);
                    Ok(())
                })
                .unwrap();
            builder.write(&path).unwrap();
        }
        _ => {
            let mut builder = FileBuilder::new();
            let dataset = builder.create_dataset("value");
            dataset.with_i32_data(&[0; 6]);
            if matches!(retirement, Retirement::Append | Retirement::Chunked) {
                dataset
                    .with_chunks(&[4])
                    .with_maxshape(&[MaxExtent::Unlimited]);
            }
            if matches!(retirement, Retirement::Chunked) {
                dataset.with_deflate(DeflateLevel::Six);
            }
            builder.write(&path).unwrap();
        }
    }
    let dense_signatures = matches!(retirement, Retirement::DenseLinks).then(|| {
        let source = fs::read(&path).unwrap();
        (
            bytes::sole_signature(&source, test_util::fractal_heap::SIGNATURE),
            bytes::sole_signature(&source, btree_v2::SIGNATURE),
        )
    });
    let mut engine = edit_test_util::open_engine(&path);
    match retirement {
        Retirement::Deletion => {
            engine.delete(&ObjectPathBuf::parse("value")).unwrap();
        }
        Retirement::DenseDataset => {
            engine
                .set_dataset_attr(&ObjectPathBuf::parse("data"), "attr_000", AttrValue::I64(9))
                .unwrap();
        }
        Retirement::DenseLinks => {
            engine
                .set_group_attr(&ObjectPathBuf::parse("g"), "a", AttrValue::I64(9))
                .unwrap();
        }
        Retirement::DenseRoot => {
            engine
                .set_group_attr(&ObjectPathBuf::root(), "root_attr_00", AttrValue::I64(9))
                .unwrap();
        }
        Retirement::Append => {
            let mut append = AppendBuilder::new();
            append.append_i32(&[7]);
            engine
                .stage_dataset_append_pending(&ObjectPathBuf::parse("value"), append)
                .unwrap();
        }
        Retirement::Chunked => {
            let mut dataset = DatasetBuilder::new("");
            dataset.with_i32_data(&[1, 2, 3, 4, 5, 6]);
            engine
                .stage_dataset_write(&ObjectPathBuf::parse("value"), dataset)
                .unwrap();
            engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
        }
    }
    let prepared = super::prepare(&mut engine);
    assert!(
        prepared.storage().unavailable().is_none(),
        "{:?}",
        prepared.storage().unavailable()
    );
    let mut expected = prepared.reclamation.as_ref().unwrap().clone();
    assert!(!expected.is_empty());
    let CommitPlan::Tree(tree) = &prepared.prepared.plan else {
        panic!("expected tree edits")
    };
    match retirement {
        Retirement::Append => {
            let write = &tree.groups[&ObjectPathBuf::root()].prepared.writes[0].write;
            let MovingWrite::AppendedChunks {
                old_tail_allocation: Some(tail),
                kept_chunks,
                ..
            } = write
            else {
                panic!("expected a deferred tail replacement");
            };
            assert_eq!(kept_chunks.len(), 1);
            let kept = super::range(
                engine
                    .addresses()
                    .resolve(kept_chunks[0].address)
                    .unwrap()
                    .get(),
                kept_chunks[0].stored_size,
            );
            assert!(
                prepared
                    .storage()
                    .allocations()
                    .iter()
                    .any(|allocation| allocation.range == kept)
            );
            assert!(
                !expected
                    .iter()
                    .any(|(range, _)| range.overlaps(tail.range()))
            );
            expected.push((tail.range(), FreeClass::Page(PageType::Raw)));
        }
        Retirement::Chunked => assert!(
            tree.groups
                .values()
                .flat_map(|node| &node.prepared.writes)
                .any(|write| matches!(write.write, MovingWrite::Chunked { .. }))
        ),
        Retirement::DenseDataset | Retirement::DenseRoot => assert!(expected.len() >= 5),
        Retirement::DenseLinks => {
            let (heap, index) = dense_signatures.unwrap();
            assert!(
                expected
                    .iter()
                    .any(|(range, _)| range.start().get() == u64::try_from(heap).unwrap())
            );
            assert!(
                expected
                    .iter()
                    .any(|(range, _)| range.start().get() == u64::try_from(index).unwrap())
            );
        }
        Retirement::Deletion => assert!(
            expected
                .iter()
                .any(|(_, class)| *class == FreeClass::Page(PageType::Raw))
        ),
    }
    expected.sort_unstable_by_key(|(range, _)| range.start());
    let pending = prepared.apply_excluding(&mut engine).ok().unwrap();
    pending.exclusion().unwrap();
    assert_eq!(pending.pending.batch.entries(), expected);
}

#[derive(Clone, Copy, Debug)]
enum Retirement {
    Append,
    Chunked,
    Deletion,
    DenseDataset,
    DenseLinks,
    DenseRoot,
}
