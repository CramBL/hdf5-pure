//! Storage reclamation, page classification, free lists, and append reserves.

use core::num::NonZeroU64;
use std::path::Path;

use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::{
    BTreeV2Layout, FractalHeapHeader, FreeSpaceManagerHeader, HugeObjectRecordContext,
};
use hdf5_pure_space::__private::AGGREGATE_BLOCK_SIZE;
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::Extent as FreeExtent;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::FreeList;
use hdf5_pure_space::__private::ManagerKind;
use hdf5_pure_space::__private::ManagerSections;
use hdf5_pure_space::__private::PageType;
use hdf5_pure_space::__private::PagedSections;
use hdf5_pure_space::__private::SessionSpace;
use rstest::rstest;
use test_util::btree_v2;
use test_util::fractal_heap;
use test_util::widths::Widths;

use super::common;
use crate::FileSpaceSettings;
use crate::FreeSectionCoordinates;
use crate::MemoryStrategy;
use crate::PersistedFreeSpacePolicy;
use crate::UserblockSize;
use crate::access_mode::AccessMode;
use crate::address::AddressContext;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::DecodedStoredAddress;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunk_index_inplace::Store;
use crate::convert::Narrow;
use crate::data_layout::ChunkIndexLayout;
use crate::dataspace::MaxExtent;
use crate::edit;
use crate::edit::APPEND_RESERVE_BYTES;
use crate::edit::AppendBuilder;
use crate::edit::AppendTarget;
use crate::edit::DEFAULT_GATHER_PAGE;
use crate::edit::OhRegion;
use crate::edit::Placement;
use crate::edit::SyncPolicy;
use crate::edit::TailPlacement;
use crate::edit::WriteEngine;
use crate::edit::aggregation::AggregateSpace;
use crate::edit::reclaim;
use crate::error::Error;
use crate::error::FormatError;
use crate::file_lock::FileLocking;
use crate::file_space_info::FileSpacePageSize;
use crate::file_space_info::FileSpaceStrategy;
use crate::file_writer::LENGTH_SIZE;
use crate::file_writer::OFFSET_SIZE;
use crate::file_writer::OFFSET_WIDTH;
use crate::fractal_heap_storage::FractalHeapAllocation;
use crate::fractal_heap_storage::FractalHeapStorage;
use crate::fractal_heap_storage::FractalHeapStorageError;
use crate::free_space_manager;
use crate::link_info::LinkInfoMessage;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPath;
use crate::object_path::ObjectPathBuf;
use crate::source::BytesSource;
use crate::source::MetadataCacheConfig;
use crate::source::SourceMetadata;
use crate::type_builders::AttrValue;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;
use crate::writer::FileBuilder;

#[test]
fn strict_persisted_recovery_retains_complete_metadata_allocations() {
    let path = test_util::temp::temp_path("manager-capacity.h5");
    test_util_hdf5::file_space::write_persisted_interior_hole(&path);
    let file = crate::File::open(&path).unwrap();
    let report = file.persisted_free_sections().unwrap().unwrap();
    let address = report.managers()[0].address();
    let mut superblock = file.superblock().clone();
    let widths = FormatWidths::from_sizes(8, 8).unwrap();
    let header_length = usize::try_from(format::free_space_manager_header_len(widths)).unwrap();
    let header_at = usize::try_from(address.get()).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let header = FreeSpaceManagerHeader::parse(widths, &bytes[header_at..]).unwrap();
    let list_at = header.section_list_addr().get();
    let capacity = header.section_list_used() + 128;
    let extension = superblock.superblock_extension_address.unwrap().get();
    superblock.eof_address = superblock.eof_address.max(list_at + capacity);
    drop(file);
    bytes.resize(usize::try_from(superblock.eof_address).unwrap(), 0);
    bytes[header_at + header_length - 12..header_at + header_length - 4]
        .copy_from_slice(&capacity.to_le_bytes());
    test_util::checksum::restamp(&mut bytes, header_at, header_length);
    let encoded = format::serialize_superblock(&superblock).unwrap();
    bytes[..encoded.len()].copy_from_slice(&encoded);
    std::fs::write(&path, &bytes).unwrap();
    let mut expected: Vec<_> = edit::object_header::read_oh_chunks(
        &BytesSource::new(&bytes),
        extension,
        AddressContext::from_effective_base(BaseAddress::ZERO),
    )
    .unwrap()
    .into_iter()
    .map(|chunk| chunk.span)
    .collect();
    expected.extend([
        (address.get(), u64::try_from(header_length).unwrap()),
        (list_at, capacity),
    ]);
    expected.sort_unstable();
    let engine = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Bounded,
        PersistedFreeSpacePolicy::Strict,
    )
    .unwrap();
    let mut actual: Vec<_> = engine
        .persist
        .as_ref()
        .unwrap()
        .old_blocks
        .iter()
        .map(|extent| (extent.start(), extent.len()))
        .collect();
    actual.sort_unstable();
    assert_eq!(actual, expected);
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
}

#[test]
fn strict_persisted_failure_precedes_image_construction() {
    let path = test_util::temp::temp_path("manager-failure.h5");
    test_util_hdf5::file_space::write_persisted_interior_hole(&path);
    let mut bytes = std::fs::read(&path).unwrap();
    let at = test_util::bytes::find_signature(&bytes, test_util::free_space::SIGNATURE).unwrap();
    let length = usize::try_from(format::free_space_manager_header_len(
        FormatWidths::from_sizes(8, 8).unwrap(),
    ))
    .unwrap();
    let checksum = u32::from_le_bytes(bytes[at + length - 4..at + length].try_into().unwrap());
    bytes[at + length - 4] ^= 1;
    std::fs::write(&path, &bytes).unwrap();
    let built = std::cell::Cell::new(false);
    let Err(error) = WriteEngine::open_imaged_with_policy(
        &path,
        Some(FileLocking::Enabled),
        PersistedFreeSpacePolicy::Strict,
        |handle, _| {
            built.set(true);
            Ok(Box::new(WriteEngine::read_mirror(handle)?))
        },
    ) else {
        panic!("expected a manager checksum error");
    };
    let Error::PersistedFreeSpace { source, .. } = error else {
        panic!("{error:?}");
    };
    let Error::Format(crate::FormatError::ChecksumMismatch { expected, computed }) = *source else {
        panic!("{source:?}");
    };
    assert_eq!((expected, computed), (checksum ^ 1, checksum));
    assert!(!built.get());
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
}

#[rstest]
#[case::flat(FileSpaceStrategy::FsmAggr)]
#[case::page(FileSpaceStrategy::Page)]
fn reusable_space_stats_include_and_release_append_reserves(#[case] strategy: FileSpaceStrategy) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("reserve-summary.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(true));
    builder
        .create_dataset("scratch")
        .with_i32_data(&vec![1; 32_768]);
    builder
        .create_dataset("keep")
        .with_i32_data(&vec![2; 4_096]);
    builder.write(&path).unwrap();
    let mut session = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    session.delete(&ObjectPathBuf::parse("/scratch")).unwrap();
    session.commit().unwrap();
    session.reserve_for_immediate_append(16_384).unwrap();
    assert!(session.reserved.total_bytes() >= 16_384);
    let reserved = session.reusable_space_stats().unwrap();
    assert_eq!(reserved.logical_size(), session.image.len());
    let sections = session.reusable_free_sections().unwrap();
    test_util_hdf5::assert_free_sections_consistent!(
        &reserved,
        &sections,
        FreeSectionCoordinates::Absolute
    );
    session.release_reserve();
    assert!(session.reserved.is_empty());
    assert_eq!(session.reusable_space_stats().unwrap(), reserved);
}

#[rstest]
#[case::reserve(true, 1)]
#[case::aggregate(false, AGGREGATE_BLOCK_SIZE - 1)]
fn reusable_space_stats_check_additional_byte_sums(#[case] reserve: bool, #[case] expected: u64) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("summary-overflow.h5");
    FileBuilder::new().write(&path).unwrap();
    let mut session = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    session.space = SessionSpace::from_strategy(
        Some(FileSpaceStrategy::FsmAggr),
        1,
        FileSpacePageSize::DEFAULT,
    );
    session.aggregation = None;
    session
        .space
        .seed_persisted(None, FreeExtent::new(0, u64::MAX).unwrap());
    if reserve {
        session.reserved.free(FreeExtent::new(0, 1).unwrap());
    } else {
        let mut aggregation = AggregateSpace::new(session.end_of_allocation().unwrap().get());
        aggregation
            .reserve(
                session.context.addresses(),
                &mut session.space,
                NonZeroU64::new(1).unwrap(),
                AllocationClass::Raw,
            )
            .unwrap();
        session.aggregation = Some(aggregation);
    }
    assert_eq!(
        session.reusable_space_stats().unwrap_err(),
        FormatError::OffsetOverflow {
            offset: u64::MAX,
            length: expected
        }
    );
    session.aggregation = None;
    session.reserved = FreeList::new();
    session.space = SessionSpace::Disabled;
}

#[rstest]
#[case::index(AllocationClass::Metadata)]
#[case::chunk(AllocationClass::Raw)]
fn array_append_preserves_raw_page_reuse(#[case] class: AllocationClass) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("page_reuse.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("keep").with_i32_data(&[1, 2, 3, 4]);
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page).with_persistence(true),
    );
    builder.write(&path).unwrap();
    let mut engine = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let before = engine.space.paged_sections().unwrap();
    let raw = *before.raw().first().unwrap();
    assert!(before.metadata().iter().any(|extent| extent.len() >= 8));
    let at = {
        let mut store = engine.store();
        store.reuse_session_space = true;
        store.allocate(&[0; 8], class).unwrap()
    };
    assert_eq!(at, raw.start());
    let after = engine.space.paged_sections().unwrap();
    assert_eq!(after.metadata(), before.metadata());
    assert_eq!(
        after.raw(),
        [FreeExtent::new(raw.start() + 8, raw.len() - 8).unwrap()]
    );
}

#[rstest]
#[case::managed_objects(0)]
#[case::huge_objects(2)]
fn deleting_a_dense_attribute_owner_reclaims_its_heap_as_metadata(#[case] huge_attrs: usize) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dense_attribute_reclaim.h5");
    write_dense_attribute_owner(&path, huge_attrs, FileSpaceStrategy::Page);
    assert_eq!(
        test_util::fractal_heap::huge_object_count(&std::fs::read(&path).unwrap()),
        u64::try_from(huge_attrs).unwrap()
    );

    let object = ObjectPathBuf::parse("/g");
    let storage = {
        let mut session = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        let DenseAttributeOwner { addr, storage, .. } = dense_attribute_owner(&session, &object);
        assert_eq!(
            storage.first().map(|&(storage_addr, _, _)| storage_addr),
            Some(attribute_heap_address(&session, addr))
        );
        assert!(
            storage
                .windows(2)
                .all(|pair| pair[0].0 + pair[0].1 == pair[1].0),
            "{storage:?}"
        );
        assert!(
            storage
                .iter()
                .all(|&(_, _, class)| class == FreeClass::Page(PageType::Meta)),
            "{storage:?}"
        );

        assert_ownership_covers_storage(&session, &storage, false);

        let incoming = session.count_incoming_hard_links().unwrap();
        let mut planned = Vec::new();
        session.collect_free_spans(addr, 0, &incoming, &mut planned);
        for span in &storage {
            assert!(planned.contains(span), "{span:?} is not in {planned:?}");
        }

        session.delete(&object).unwrap();
        session.commit().unwrap();
        storage
    };

    let reopened = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let sections = reopened
        .space
        .paged_sections()
        .expect("the PAGE strategy installs per-type free lists");
    for &(addr, len, _) in &storage {
        assert!(
            freed_as_metadata(&sections, addr, len),
            "dense attribute storage ({addr}, {len}) must persist as metadata free space"
        );
    }
}

#[rstest]
fn dense_attribute_storage_is_classified_by_its_stored_pages(#[values(0, 512)] userblock: u64) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dense_attribute_pages.h5");
    let mut builder = FileBuilder::new();
    let mut group = builder.create_group("g");
    for i in 0..12 {
        group.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    group.set_attr("huge0", AttrValue::AsciiString("y".repeat(71_267)));
    group.set_attr("huge1", AttrValue::AsciiString("y".repeat(70_000)));
    builder.add_group(group.finish());
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(0)
            .with_page_size(FileSpacePageSize::try_from(DENSE_ATTRIBUTE_PAGE_SIZE).unwrap()),
    );
    builder.write(&path).unwrap();
    prepend_userblock(&path, userblock);

    let session = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let DenseAttributeOwner { storage, .. } =
        dense_attribute_owner(&session, &ObjectPathBuf::parse("/g"));
    let meta = FreeClass::Page(PageType::Meta);
    assert_eq!(
        storage,
        [
            (164, 146, meta),
            (310, 1024, meta),
            (1334, 38, meta),
            (1372, 512, meta),
            (1884, 38, meta),
            (1922, 512, meta),
            (2434, 71_294, meta),
            (18 * DENSE_ATTRIBUTE_PAGE_SIZE, 70_027, FreeClass::Dead),
        ]
        .map(|(addr, len, class)| (addr + userblock, len, class))
    );
}

#[test]
fn rebuilding_dense_group_attributes_reclaims_the_old_storage_in_flat_space() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dense_attribute_replace.h5");
    write_dense_attribute_owner(&path, 1, FileSpaceStrategy::FsmAggr);

    let object = ObjectPathBuf::parse("/g");
    {
        let mut session = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        let DenseAttributeOwner { storage, .. } = dense_attribute_owner(&session, &object);
        assert!(storage.len() > 2, "{storage:?}");

        session
            .set_group_attr(&object, "a0", AttrValue::I64(99))
            .unwrap();
        for &(addr, len, _) in &storage {
            assert!(
                !reusable_overlaps(&session, addr, len),
                "the old dense storage ({addr}, {len}) stays live until the replacement commits"
            );
        }
        session.commit().unwrap();
        for &(addr, len, _) in &storage {
            assert!(
                reusable_covers(&session, addr, len),
                "the replaced dense storage ({addr}, {len}) must be reusable"
            );
        }
    }

    let attrs = crate::reader::File::open(&path)
        .unwrap()
        .group("g")
        .unwrap()
        .attrs()
        .unwrap();
    assert_eq!(attrs.get("a0"), Some(&AttrValue::I64(99)));
    assert_eq!(
        attrs.get("huge0"),
        Some(&AttrValue::AsciiString("y".repeat(70_000)))
    );
}

#[rstest]
#[case::wrong_block_owner(HeapDamage::DirectBlockOwner, false)]
#[case::managed_free_space_manager(HeapDamage::FreeSpaceManager, true)]
#[case::unencoded_filtered_block(HeapDamage::UnencodedFilteredBlock, false)]
fn reclamation_retains_storage_after_heap_validation_fails(
    #[case] damage: HeapDamage,
    #[case] reclaim_other: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dense_attribute_rejected_heap.h5");
    let object = ObjectPathBuf::parse("/g");
    let RejectedHeapOwner {
        header_spans,
        index,
        heap,
    } = write_rejected_heap_owner(&path, &object, damage);

    let mut session = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    session.delete(&object).unwrap();
    session.commit().unwrap();

    for &(addr, len) in &header_spans {
        if reclaim_other {
            assert!(
                reusable_covers(&session, addr, len),
                "header ({addr}, {len}) is reusable"
            );
        } else {
            assert!(
                !reusable_overlaps(&session, addr, len),
                "header ({addr}, {len}) stays unavailable"
            );
        }
    }
    for &(addr, len, _) in &index {
        if reclaim_other {
            assert!(
                reusable_covers(&session, addr, len),
                "index ({addr}, {len}) is reusable"
            );
        } else {
            assert!(
                !reusable_overlaps(&session, addr, len),
                "index ({addr}, {len}) stays unavailable"
            );
        }
    }
    for (extent, _) in heap.allocations() {
        assert!(
            !reusable_overlaps(&session, extent.start(), extent.len()),
            "the rejected heap allocation {extent:?} must stay allocated"
        );
    }
}

#[rstest]
#[case::flat(FileSpaceStrategy::FsmAggr, 0)]
#[case::userblock(FileSpaceStrategy::FsmAggr, 512)]
#[case::paged(FileSpaceStrategy::Page, 0)]
fn deleting_the_owner_of_a_filtered_attribute_heap_reclaims_its_storage(
    #[case] strategy: FileSpaceStrategy,
    #[case] userblock: u64,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dense_attribute_filtered_heap.h5");
    write_dense_attribute_owner(&path, 0, strategy);
    prepend_userblock(&path, userblock);
    let object = ObjectPathBuf::parse("/g");
    let heap_address = {
        let session = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        let DenseAttributeOwner { addr, .. } = dense_attribute_owner(&session, &object);
        attribute_heap_address(&session, addr)
    };
    let mut bytes = std::fs::read(&path).unwrap();
    let heap = filter_attribute_heap(&mut bytes, userblock, heap_address, true);
    std::fs::write(&path, bytes).unwrap();
    let attrs = crate::reader::File::open(&path)
        .unwrap()
        .group("g")
        .unwrap()
        .attrs()
        .unwrap();
    assert_eq!(attrs.len(), 12);
    assert_eq!(attrs.get("a3"), Some(&AttrValue::I64(3)));

    let mut session = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let DenseAttributeOwner {
        header_spans,
        storage,
        ..
    } = dense_attribute_owner(&session, &object);
    for &(addr, len) in &heap {
        assert!(
            storage.contains(&(addr, len, FreeClass::Page(PageType::Meta))),
            "the filtered heap allocation ({addr}, {len}) is in {storage:?}"
        );
    }
    assert_ownership_covers_storage(&session, &storage, true);
    session.delete(&object).unwrap();
    session.commit().unwrap();
    assert!(session.space_ownership().unwrap().analysis_complete);

    for (addr, len) in header_spans
        .into_iter()
        .chain(storage.into_iter().map(|(addr, len, _)| (addr, len)))
    {
        assert!(
            reusable_covers(&session, addr - userblock, len),
            "({addr}, {len}) of the deleted object is reusable"
        );
    }
    drop(session);
    let reader = crate::reader::File::open(&path).unwrap();
    let err = reader.group("g").unwrap_err();
    let Error::Format(FormatError::PathNotFound(missing)) = &err else {
        panic!("expected PathNotFound for g, got {err:?}");
    };
    assert_eq!(missing, "g");
    assert_eq!(
        reader.dataset("keep").unwrap().read_i32().unwrap(),
        (0..1024).collect::<Vec<_>>()
    );
}

#[cfg(feature = "deflate")]
#[rstest]
fn deleting_a_filtered_heap_with_huge_objects_reclaims_its_encoded_allocations(
    #[values(0, 512)] userblock: u64,
) {
    let path = test_util::temp::temp_path("filtered_huge_reclaim.h5");
    write_dense_attribute_owner(&path, 2, FileSpaceStrategy::FsmAggr);
    prepend_userblock(&path, userblock);
    let object = ObjectPathBuf::parse("/g");
    let heap_address = {
        let session = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        let DenseAttributeOwner { addr, .. } = dense_attribute_owner(&session, &object);
        attribute_heap_address(&session, addr)
    };
    let mut bytes = std::fs::read(&path).unwrap();
    filter_attribute_heap(&mut bytes, userblock, heap_address, true);
    let header = FractalHeapHeader::parse(
        &bytes,
        heap_address.to_usize().unwrap(),
        OFFSET_SIZE,
        LENGTH_SIZE,
    )
    .unwrap();
    let widths = crate::file_writer::WIDTHS;
    let tree_at = (header.btree_huge_objects_address.get() + userblock)
        .to_usize()
        .unwrap();
    let tree = crate::btree_v2::BTreeV2Header::parse_with_widths(&bytes, tree_at, widths).unwrap();
    assert_eq!(tree.depth, 0);
    let layout = BTreeV2Layout::new(&tree, widths.offsets).unwrap();
    let root = layout.root().unwrap();
    let node_at = (root.address().get() + userblock).to_usize().unwrap();
    let node = layout.parse(&bytes[node_at..], root).unwrap();
    let records: Vec<_> = (0..node.record_count())
        .map(|index| {
            let record = crate::btree_v2::BTreeV2Record {
                data: node.record(index).unwrap().to_vec(),
            };
            record.huge_object(widths.offsets, widths.lengths).unwrap()
        })
        .collect();
    let pipeline = crate::test_data::deflate_pipeline();
    let records: Vec<_> = records
        .into_iter()
        .map(|record| {
            let at = (record.address.get() + userblock).to_usize().unwrap();
            let decoded = &bytes[at..at + record.length.to_usize().unwrap()];
            let encoded = crate::test_data::encode_block(decoded, &pipeline);
            assert!(encoded.len().to_u64() < record.length);
            bytes[at..at + encoded.len()].copy_from_slice(&encoded);
            let mut filtered = record.address.get().to_le_bytes().to_vec();
            filtered.extend_from_slice(&encoded.len().to_u64().to_le_bytes());
            filtered.extend_from_slice(&0u32.to_le_bytes());
            filtered.extend_from_slice(&record.length.to_le_bytes());
            filtered.extend_from_slice(&record.id.to_le_bytes());
            filtered
        })
        .collect();
    let context = HugeObjectRecordContext::new(&header, widths).unwrap();
    let encoded_tree = btree_v2::Header::new(
        context.tree_type(),
        context.record_size(),
        tree.root_node_address.get(),
        tree.num_records_in_root,
    )
    .node_size(tree.node_size)
    .build(Widths::EIGHT);
    bytes[tree_at..tree_at + encoded_tree.len()].copy_from_slice(&encoded_tree);
    let encoded_node = btree_v2::leaf(context.tree_type(), &records);
    assert!(encoded_node.len() <= tree.node_size as usize);
    bytes[node_at..node_at + encoded_node.len()].copy_from_slice(&encoded_node);
    let heap = crate::fractal_heap_storage::collect_fractal_heap_storage_from_source(
        &crate::source::BaseOffsetSource {
            inner: &BytesSource::new(&bytes),
            base: BaseAddress::new(userblock),
        },
        AddressContext::from_effective_base(BaseAddress::new(userblock)),
        StoredAddress::new(heap_address - userblock),
        widths,
    )
    .unwrap();
    assert_eq!(
        heap.allocations()
            .iter()
            .filter(|(_, kind)| *kind == FractalHeapAllocation::HugeObject)
            .count(),
        2
    );
    std::fs::write(&path, bytes).unwrap();
    let reader = crate::reader::File::open(&path).unwrap();
    let attrs = reader.group("g").unwrap().attrs().unwrap();
    assert_eq!(attrs.len(), 14);
    assert_eq!(
        attrs.get("huge0"),
        Some(&AttrValue::AsciiString("y".repeat(70_000)))
    );
    drop(reader);
    let mut session = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let DenseAttributeOwner { storage, .. } = dense_attribute_owner(&session, &object);
    assert_ownership_covers_storage(&session, &storage, false);
    session.delete(&object).unwrap();
    session.commit().unwrap();
    assert!(session.space_ownership().unwrap().analysis_complete);
    for (extent, _) in heap.allocations() {
        assert!(
            reusable_covers(&session, extent.start(), extent.len()),
            "{extent:?}"
        );
    }
    drop(session);
    let reader = crate::reader::File::open(&path).unwrap();
    assert_eq!(
        reader.dataset("keep").unwrap().read_i32().unwrap(),
        (0..1024).collect::<Vec<_>>()
    );
    let error = reader.group("g").unwrap_err();
    let Error::Format(FormatError::PathNotFound(missing)) = error else {
        panic!("expected PathNotFound, got {error:?}")
    };
    assert_eq!(missing, "g");
}

#[test]
fn deleting_a_c_written_dense_link_group_reclaims_its_whole_subtree() {
    let dir = tempfile::tempdir().unwrap();
    let path = test_util::copy_data("c/huge_links_12.h5", dir.path());
    let reader = crate::reader::File::open(&path).unwrap();
    let link_names = reader.group("g").unwrap().datasets().unwrap();
    assert_eq!(link_names.len(), 12);
    let data_blocks: Vec<(u64, u64, FreeClass)> = link_names
        .iter()
        .map(|link_name| {
            let crate::Layout::Contiguous {
                address: Some(data_addr),
                size,
            } = reader
                .dataset(&format!("g/{link_name}"))
                .unwrap()
                .layout()
                .unwrap()
            else {
                panic!("the child {link_name:?} stores its data contiguously");
            };
            (data_addr, size, FreeClass::Page(PageType::Raw))
        })
        .collect();
    drop(reader);

    let object = ObjectPathBuf::parse("/g");
    let mut session = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let resolve = |object_path: &ObjectPathBuf| {
        crate::group_v2::resolve_path_any_from_source(
            &session.image(),
            AccessMode::ReadWrite,
            session.superblock(),
            session.widths(),
            session.addresses(),
            &object_path.as_path(),
        )
        .unwrap()
    };
    let addr = resolve(&object);
    let header_spans = session.oh_chunk_spans(addr).unwrap();
    let header = ObjectHeader::parse_from_source_with_widths(
        &SourceMetadata(&session.image()),
        AccessMode::ReadWrite,
        addr,
        session.widths(),
        BaseAddress::ZERO,
    )
    .unwrap();
    let link_info = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::LINK_INFO)
        .map(|message| {
            LinkInfoMessage::parse_with_offset_width(&message.data, OFFSET_WIDTH).unwrap()
        })
        .expect("a dense group has a Link Info message");
    let heap_address = link_info
        .fractal_heap_address
        .expect("a dense group's Link Info message stores its heap");
    let link_storage = session
        .dense_group_storage_spans(&header_spans, None, &link_info)
        .expect("the dense link storage is structurally complete");
    assert_ownership_covers_storage(&session, &link_storage, false);
    let heap = crate::fractal_heap_storage::collect_fractal_heap_storage_from_source(
        &session.image(),
        AddressContext::from_effective_base(BaseAddress::ZERO),
        heap_address,
        crate::file_writer::WIDTHS,
    )
    .unwrap();
    assert_eq!(
        heap.allocations()
            .iter()
            .filter(|(_, kind)| *kind == FractalHeapAllocation::HugeObject)
            .count(),
        12
    );
    for &(extent, kind) in heap.allocations() {
        let span = (
            extent.start(),
            extent.len(),
            FreeClass::Page(kind.allocation_class().into()),
        );
        assert!(
            link_storage.contains(&span),
            "{span:?} is not in {link_storage:?}"
        );
    }

    let mut expected: Vec<(u64, u64, FreeClass)> = reclaim::meta_spans(header_spans)
        .chain(link_storage)
        .chain(data_blocks)
        .chain(link_names.iter().flat_map(|link_name| {
            let child = resolve(&ObjectPathBuf::parse(&format!("/g/{link_name}")));
            reclaim::meta_spans(session.oh_chunk_spans(child).unwrap())
        }))
        .collect();
    expected.sort_unstable_by_key(|&(span_addr, _, _)| span_addr);
    let incoming = session.count_incoming_hard_links().unwrap();
    let mut planned = Vec::new();
    session.collect_free_spans(addr, 0, &incoming, &mut planned);
    planned.sort_unstable_by_key(|&(span_addr, _, _)| span_addr);
    assert_eq!(planned, expected);

    session.delete(&object).unwrap();
    session.commit().unwrap();
    drop(session);
    let err = crate::reader::File::open(&path)
        .unwrap()
        .group("g")
        .unwrap_err();
    let Error::Format(FormatError::PathNotFound(missing)) = &err else {
        panic!("expected PathNotFound for g, got {err:?}");
    };
    assert_eq!(missing, "g");
}

#[test]
fn a_v2_btree_chunk_index_is_metadata_when_reclaimed() {
    let btree = ChunkIndexLayout::BTreeV2 {
        address: Some(StoredAddress::new(4096)),
    };
    assert_eq!(
        reclaim::chunk_index_free_class(btree, false),
        FreeClass::Page(PageType::Meta)
    );
    assert_eq!(
        reclaim::chunk_index_free_class(btree, true),
        FreeClass::Page(PageType::Meta)
    );

    let fixed = ChunkIndexLayout::FixedArray {
        address: Some(StoredAddress::new(4096)),
    };
    assert_eq!(
        reclaim::chunk_index_free_class(fixed, true),
        FreeClass::Page(PageType::Raw)
    );
    assert_eq!(
        reclaim::chunk_index_free_class(fixed, false),
        FreeClass::Dead
    );
}

#[test]
fn a_chunk_index_is_placed_by_the_chunk_data_that_abuts_it() {
    assert!(reclaim::index_abuts_chunk_data(&[(100, 60)], &[(160, 40)]));
    assert!(reclaim::index_abuts_chunk_data(&[(200, 60)], &[(160, 40)]));
    assert!(reclaim::index_abuts_chunk_data(
        &[(100, 60), (900, 60)],
        &[(160, 40)]
    ));
    assert!(reclaim::index_abuts_chunk_data(
        &[(100, 60)],
        &[(180, 20), (160, 20)]
    ));
    assert!(!reclaim::index_abuts_chunk_data(
        &[(9000, 60)],
        &[(160, 40)]
    ));
    assert!(!reclaim::index_abuts_chunk_data(&[(100, 59)], &[(160, 40)]));
    assert!(!reclaim::index_abuts_chunk_data(&[], &[(160, 40)]));
    assert!(reclaim::index_abuts_chunk_data(&[], &[]));
    assert!(!reclaim::index_abuts_chunk_data(&[(160, 0)], &[(160, 40)]));
}

#[test]
fn page_zero_detection_checks_each_index_span() {
    const PAGE: FileSpacePageSize = FileSpacePageSize::MIN;
    let c_written = [(48u64, 80u64), (512, 512)];
    assert!(reclaim::index_abuts_chunk_data(&[(1024, 512)], &c_written));
    assert!(reclaim::touches_page_zero(PAGE, c_written.iter().copied()));
    assert!(!reclaim::touches_page_zero(PAGE, [(1024, 200)]));
    assert!(reclaim::touches_page_zero(PAGE, [(511, 200)]));
    assert!(!reclaim::touches_page_zero(PAGE, [(512, 200)]));
    assert!(!reclaim::touches_page_zero(PAGE, [(0, 0)]));
    assert!(!reclaim::touches_page_zero(PAGE, []));
}

#[rstest]
fn page_zero_classification_is_relative_to_the_base_address(#[values(0, 512)] userblock: u64) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paged_index.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UserblockSize::try_from(userblock).unwrap());
    b.create_dataset("d").with_i32_data(&[1; 100]);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_page_size(FileSpacePageSize::MIN),
    );
    b.write(&path).unwrap();
    let s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();

    let data = [(userblock + 1024, 512)];
    assert!(!s.index_is_provably_raw(&data, &[(userblock + 48, 80), (userblock + 512, 512)]));
    assert!(s.index_is_provably_raw(&data, &[(userblock + 512, 512)]));
}

#[rstest]
fn paged_open_seeds_each_manager_by_slot(
    #[values(PersistedFreeSpacePolicy::Tolerant, PersistedFreeSpacePolicy::Strict)]
    policy: PersistedFreeSpacePolicy,
) {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paged_seed.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&(0..1000).collect::<Vec<i32>>())
        .with_shape(&[1000]);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(0)
            .with_page_size(FileSpacePageSize::DEFAULT),
    );
    b.write(&path).unwrap();
    let on_disk: u64 = crate::reader::File::open(&path)
        .unwrap()
        .persisted_free_sections()
        .unwrap()
        .unwrap()
        .free_sections()
        .free_bytes();

    let s = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Bounded,
        policy,
    )
    .unwrap();
    assert_eq!(
        s.space.page_size().expect("a paged file has a page size"),
        FileSpacePageSize::DEFAULT
    );
    let sections = s
        .space
        .paged_sections()
        .expect("a paged file installs paged state");
    assert!(
        !sections.metadata().is_empty(),
        "SUPER (slot 0) sections seed the metadata list"
    );
    assert!(
        !sections.raw().is_empty(),
        "DRAW (slot 2) sections seed the raw list, not the metadata list"
    );
    let mut all: Vec<_> = s.space.reusable_sections().collect();
    all.extend_from_slice(sections.unclassified());
    assert!(
        sections.unclassified().is_empty(),
        "our own paged writer files nothing whose page type is unknown"
    );
    let flat: u64 = all.iter().map(|extent| extent.len()).sum();
    assert_eq!(
        flat, on_disk,
        "the split lists hold exactly the file's free space"
    );
    all.sort_by_key(|extent| extent.start());
    let mut prev_end = 0u64;
    for extent in all {
        assert!(
            extent.start() >= prev_end,
            "the per-type lists do not overlap"
        );
        prev_end = extent.end();
    }
}

#[test]
fn a_c_written_chunk_index_is_not_reclaimed_as_raw() {
    const PAGE: u64 = 4096;
    let dir = tempfile::tempdir().unwrap();
    let path = test_util::copy_data("c/paged_index.h5", dir.path());

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let page_size = s.space.page_size().expect("a paged file has a page size");
    assert_eq!(page_size.get(), PAGE);
    let victim_addr = crate::group_v2::resolve_path_any(
        s.image.as_slice().unwrap(),
        AccessMode::ReadWrite,
        s.context.superblock(),
        s.context.widths(),
        AddressContext::from_effective_base(s.context.superblock().base_address),
        &ObjectPath::parse("victim"),
    )
    .unwrap();
    let index_spans = s
        .chunked_index_spans(victim_addr)
        .expect("the C library's extensible-array index is enumerable");
    assert!(!index_spans.is_empty());
    let mut meta_pages: Vec<u64> = vec![0];
    for (addr, len) in &index_spans {
        for p in (addr / PAGE)..=((addr + len - 1) / PAGE) {
            meta_pages.push(p);
        }
    }
    meta_pages.sort_unstable();
    meta_pages.dedup();
    s.delete(&ObjectPathBuf::parse("/victim")).unwrap();
    s.commit().unwrap();
    let mut db = crate::type_builders::DatasetBuilder::new("added");
    db.with_f64_data(&[2.5f64; 8]).with_shape(&[8]);
    s.stage_created_dataset(&ObjectPathBuf::parse("/added"), db)
        .unwrap();
    s.commit().unwrap();
    drop(s);
    let f = crate::reader::File::open(&path).unwrap();
    let added = f.dataset("added").unwrap();
    let crate::Layout::Contiguous {
        address: Some(addr),
        size,
    } = added.layout().unwrap()
    else {
        panic!("a small f64 dataset is stored contiguously");
    };
    for p in (addr / PAGE)..=((addr + size - 1) / PAGE) {
        assert!(
            !meta_pages.contains(&p),
            "the new dataset's raw data landed at ({addr}, {size}), in page {p}, \
             which held live metadata before the edit: {meta_pages:?}"
        );
    }

    assert_eq!(added.read_f64().unwrap(), vec![2.5f64; 8]);
    assert_eq!(
        f.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3, 4]
    );
}

#[test]
fn padding_a_tail_page_of_unknown_type_records_nothing() {
    use crate::writer::FileBuilder;

    const PAGE: u64 = 4096;
    let dir = tempfile::tempdir().unwrap();

    let build = |path: &std::path::Path| {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..1000).collect::<Vec<i32>>())
            .with_shape(&[1000]);
        b.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page)
                .with_persistence(true)
                .with_threshold(0)
                .with_page_size(FileSpacePageSize::try_from(PAGE).unwrap()),
        );
        b.write(path).unwrap();
    };
    let unknown = dir.path().join("unknown_tail.h5");
    build(&unknown);
    let mut s = WriteEngine::open_with_locking(&unknown, FileLocking::Enabled).unwrap();
    s.append(&[0u8; 100]).unwrap(); // leaves the image non-page-aligned
    assert_eq!(s.space.paged_tail_type(), Some(None));
    s.pad_to_page().unwrap();
    assert_eq!(s.image.len() % PAGE, 0, "the file is padded to a page");
    assert!(
        s.space
            .pending_padding(PageType::Meta)
            .expect("the file is paged")
            .is_empty()
            && s.space
                .pending_padding(PageType::Raw)
                .expect("the file is paged")
                .is_empty(),
        "padding a tail page of unknown type must claim no page type"
    );
    drop(s);
    let known = dir.path().join("known_tail.h5");
    build(&known);
    let mut s = WriteEngine::open_with_locking(&known, FileLocking::Enabled).unwrap();
    s.begin_page(PageType::Meta).unwrap();
    s.append(&[0u8; 100]).unwrap();
    s.pad_to_page().unwrap();
    assert_eq!(
        s.space
            .pending_padding(PageType::Meta)
            .expect("the file is paged")
            .len(),
        1,
        "a known metadata tail records its padding as metadata free space"
    );
    assert!(
        s.space
            .pending_padding(PageType::Raw)
            .expect("the file is paged")
            .is_empty()
    );
}

#[rstest]
fn a_generic_large_section_is_only_reusable_as_whole_pages(
    #[values(PersistedFreeSpacePolicy::Tolerant, PersistedFreeSpacePolicy::Strict)]
    policy: PersistedFreeSpacePolicy,
) {
    const PAGE: u64 = 512;
    let dir = tempfile::tempdir().unwrap();
    let path = test_util::copy_data("c/generic_large.h5", dir.path());
    let opened = crate::reader::File::open(&path).unwrap();
    let info = opened
        .file_space_info()
        .unwrap()
        .expect("a persisting file");
    drop(opened);
    assert_eq!(info.page_size.get(), PAGE);
    let bytes = std::fs::read(&path).unwrap();
    let src = crate::source::BytesSource::new(bytes.as_slice());
    let slot6 = info.manager_addrs[6];
    assert_ne!(
        slot6,
        DecodedStoredAddress::Undefined,
        "the C library populated the large manager"
    );
    let (sections, _) = free_space_manager::read_persisted_sections_source(
        &src,
        crate::file_writer::WIDTHS,
        BaseAddress::ZERO,
        &[slot6],
    )
    .unwrap();
    let fragments: Vec<(u64, u64)> = sections
        .iter()
        .map(|s| (s.addr.get(), s.size))
        .filter(|&(addr, size)| addr % PAGE != 0 || size % PAGE != 0)
        .collect();
    assert!(
        !fragments.is_empty(),
        "the premise of this test: the C library files sub-page fragments in \
         its generic-large manager, but this file has none ({sections:?})"
    );
    let s = WriteEngine::open_rw_with_strategy(
        &path,
        MetadataCacheConfig::disabled(),
        FileLocking::Enabled,
        MemoryStrategy::Bounded,
        policy,
    )
    .unwrap();
    let unclassified = s
        .space
        .paged_sections()
        .expect("a paged file installs paged state")
        .unclassified()
        .to_vec();
    let reusable: Vec<_> = s.space.reusable_sections().collect();
    for &(addr, size) in &fragments {
        let fragment = FreeExtent::new(addr, size).expect("wire sections are non-empty");
        assert!(
            unclassified.contains(&fragment),
            "fragment ({addr}, {size}) must be recorded as unclassified, not lost"
        );
        assert!(
            !reusable.iter().any(|extent| {
                extent.start() < fragment.end() && fragment.start() < extent.end()
            }),
            "fragment ({addr}, {size}) must not be offered to any allocation: {reusable:?}"
        );
    }
}

#[test]
fn a_paged_allocation_only_crosses_page_types_over_whole_free_pages() {
    use crate::writer::FileBuilder;

    const PAGE: u64 = 4096;
    fn session(
        path: &std::path::Path,
        meta: Option<(u64, u64)>,
        raw: Option<(u64, u64)>,
    ) -> WriteEngine {
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..4000).collect::<Vec<i32>>())
            .with_shape(&[4000]);
        b.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page)
                .with_persistence(true)
                .with_threshold(0)
                .with_page_size(FileSpacePageSize::try_from(PAGE).unwrap()),
        );
        b.write(path).unwrap();
        let mut s = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
        s.space = SessionSpace::from_strategy(
            Some(FileSpaceStrategy::Page),
            0,
            FileSpacePageSize::DEFAULT,
        );
        if let Some((addr, len)) = meta {
            s.space.seed_persisted(
                Some(ManagerKind::PagedSmallMetadata),
                FreeExtent::new(addr, len).unwrap(),
            );
        }
        if let Some((addr, len)) = raw {
            s.space.seed_persisted(
                Some(ManagerKind::PagedSmallRaw),
                FreeExtent::new(addr, len).unwrap(),
            );
        }
        s
    }

    let dir = tempfile::tempdir().unwrap();
    let hole_a = (PAGE + 512, 2048);
    let hole_b = (2 * PAGE + 512, 2048);
    let mut s = session(&dir.path().join("a.h5"), Some(hole_a), None);
    assert!(
        matches!(
            s.reserve(1024, AllocationClass::Raw).unwrap(),
            Placement::Appended { .. }
        ),
        "a raw allocation must not be served out of a live metadata page"
    );
    drop(s);
    let mut s = session(&dir.path().join("b.h5"), None, Some(hole_b));
    assert!(
        matches!(
            s.reserve(1024, AllocationClass::Metadata).unwrap(),
            Placement::Appended { .. }
        ),
        "a metadata allocation must not be served out of a live raw page"
    );
    drop(s);
    let mut s = session(&dir.path().join("c.h5"), Some(hole_a), Some(hole_b));
    assert!(
        matches!(
            s.reserve(1024, AllocationClass::Raw).unwrap(),
            Placement::Reused { addr, .. } if addr == hole_b.0
        ),
        "a raw allocation takes the raw hole"
    );
    assert!(
        matches!(
            s.reserve(1024, AllocationClass::Metadata).unwrap(),
            Placement::Reused { addr, .. } if addr == hole_a.0
        ),
        "a metadata allocation takes the metadata hole"
    );
    drop(s);
    let mut s = session(&dir.path().join("d.h5"), Some((PAGE, PAGE)), None);
    assert!(
        matches!(
            s.reserve(1024, AllocationClass::Raw).unwrap(),
            Placement::Reused { addr, .. } if addr == PAGE
        ),
        "a raw allocation may open an empty metadata page"
    );
    let sections = s.space.paged_sections().expect("still paged");
    assert_eq!(
        sections.raw(),
        [FreeExtent::new(PAGE + 1024, PAGE - 1024).unwrap()],
        "the rest of the claimed page is free space of the claiming type"
    );
    assert!(
        sections.metadata().is_empty(),
        "the page left the list it was claimed from"
    );
    drop(s);
    let mut s = session(
        &dir.path().join("e.h5"),
        None,
        Some((PAGE + 512, 3 * PAGE - 512)),
    );
    assert!(
        matches!(
            s.reserve(1024, AllocationClass::Metadata).unwrap(),
            Placement::Reused { addr, .. } if addr == 2 * PAGE
        ),
        "the claim must begin at the empty page, not at the run's start"
    );
    let sections = s.space.paged_sections().expect("still paged");
    assert_eq!(
        sections.raw(),
        [
            FreeExtent::new(PAGE + 512, PAGE - 512).unwrap(),
            FreeExtent::new(3 * PAGE, PAGE).unwrap(),
        ],
        "the fragment below the claimed page and the page above it both stay free"
    );
    assert_eq!(
        sections.metadata(),
        [FreeExtent::new(2 * PAGE + 1024, PAGE - 1024).unwrap()],
        "the rest of the claimed page is free space of the claiming type"
    );
}

#[test]
fn a_reuse_only_tail_rewrite_that_cannot_place_itself_writes_nothing() {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reuse_only.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&(0..400).collect::<Vec<i32>>())
        .with_shape(&[400]);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(0),
    );
    b.write(&path).unwrap();
    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    s.space = SessionSpace::from_strategy(
        Some(FileSpaceStrategy::FsmAggr),
        0,
        FileSpacePageSize::DEFAULT,
    );
    let root = s
        .context
        .addresses()
        .resolve(s.context.superblock().root_group_address)
        .unwrap();
    let len_before = s.image.len();
    let ext_before = s.context.superblock().superblock_extension_address;

    s.commit_persisting(root, Vec::new(), TailPlacement::ReuseOnly)
        .unwrap();
    assert_eq!(
        s.image.len(),
        len_before,
        "a reuse-only rewrite with nothing to reuse must leave the file alone"
    );
    assert_eq!(
        s.context.superblock().superblock_extension_address,
        ext_before,
        "and must not have published a new extension either"
    );

    s.commit_persisting(root, Vec::new(), TailPlacement::Anywhere)
        .unwrap();
    assert!(
        s.image.len() > len_before,
        "the same rewrite that may append does grow the file, so the case \
         above was a tail declined rather than a tail of no size"
    );
}

#[test]
fn a_flat_tail_takes_exactly_the_extent_it_records() {
    use crate::writer::FileBuilder;
    const EXT_LEN: u64 = 100;
    const HOLE_AT: u64 = 512;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("flat_tail_exact.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&(0..4000).collect::<Vec<i32>>())
        .with_shape(&[4000]);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(0),
    );
    b.write(&path).unwrap();
    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let widths = FormatWidths {
        offsets: s.context.superblock().offset_size,
        lengths: s.context.superblock().length_size,
    };

    let free_total =
        |s: &WriteEngine| -> u64 { s.space.reusable_sections().map(|extent| extent.len()).sum() };

    let (mut placed, mut declined, mut with_slack) = (0usize, 0usize, 0usize);
    for hole in 120..420u64 {
        s.space = SessionSpace::from_strategy(
            Some(FileSpaceStrategy::FsmAggr),
            0,
            FileSpacePageSize::DEFAULT,
        );
        s.space.seed_persisted(
            Some(ManagerKind::Flat),
            FreeExtent::new(HOLE_AT, hole).unwrap(),
        );
        let free_before = free_total(&s);
        let eoa = s.end_of_allocation().unwrap().get();
        let (post, at, tail_len, _) = s.flat_tail_layout(eoa, &[], &[], EXT_LEN, widths, None);
        let free_after = free_total(&s);
        let managers = ManagerSections::flat(post.extents().iter().copied());
        let written = EXT_LEN + free_space_manager::manager_blocks_len(&managers, widths);
        match at {
            Some(at) => {
                placed += 1;
                assert_eq!(
                    free_before - free_after,
                    tail_len,
                    "hole {hole}: the tail took {} bytes at {at} for an extent of \
                     {tail_len}",
                    free_before - free_after
                );
                if written < tail_len {
                    with_slack += 1;
                }
            }
            None => {
                declined += 1;
                assert_eq!(
                    free_after, free_before,
                    "hole {hole}: a tail that declines to place itself must hand \
                     back everything it tried"
                );
                assert_eq!(
                    tail_len, written,
                    "hole {hole}: a tail with nowhere to go is appended, and the \
                     length it reports is what it will write there"
                );
            }
        }
    }
    assert!(
        placed > 0,
        "the sweep must reach holes the tail can actually use, or it asserts \
         nothing about placement"
    );
    assert!(
        declined > 0,
        "the sweep must reach holes too small for the tail, or the rule about \
         handing back a failed attempt is never exercised"
    );
    assert!(
        with_slack > 0,
        "the sweep must reach a hole the tail does not fill exactly, or the \
         rule about spare bytes is never exercised"
    );
}

#[test]
fn paged_tail_allocation_releases_unused_page_bytes() {
    use crate::writer::FileBuilder;

    const PAGE: u64 = 4096;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tail_appends.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(&(0..4000).collect::<Vec<i32>>())
        .with_shape(&[4000]);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(0)
            .with_page_size(FileSpacePageSize::try_from(PAGE).unwrap()),
    );
    b.write(&path).unwrap();
    let before = std::fs::metadata(&path).unwrap().len();

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    s.space =
        SessionSpace::from_strategy(Some(FileSpaceStrategy::Page), 0, FileSpacePageSize::DEFAULT);
    s.create_group(&ObjectPathBuf::parse("g")).unwrap();
    s.commit().unwrap();

    let after = std::fs::metadata(&path).unwrap().len();
    assert!(
        after > before && (after - before) % PAGE == 0,
        "the commit opened whole pages ({before} -> {after})"
    );
    let sections = s.space.paged_sections().expect("still paged");
    let extent = sections
        .metadata()
        .iter()
        .copied()
        .find(|extent| extent.end() == after)
        .expect("the page the tail opened leaves a free remainder at end-of-file");
    assert!(
        extent.len() < PAGE && extent.start() >= before,
        "the tail's blocks take the front of the page it opened and the rest is free ({} of \
         {PAGE} at {}, file {before} -> {after})",
        extent.len(),
        extent.start()
    );
}

#[test]
fn deleted_chunk_index_is_freed_into_a_raw_manager() {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paged_chunk_index_free.h5");
    let page = 4096u64;
    let mut b = FileBuilder::new();
    for name in ["drop", "keep"] {
        b.create_dataset(name)
            .with_i32_data(&(0..200).collect::<Vec<i32>>())
            .with_shape(&[200])
            .with_chunks(&[50]);
    }
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(0)
            .with_page_size(FileSpacePageSize::try_from(page).unwrap()),
    );
    b.write(&path).unwrap();

    {
        let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        s.delete(&ObjectPathBuf::parse("/drop")).unwrap();
        s.commit().unwrap();
    }
    let live_raw_pages: Vec<u64> = {
        let f = crate::reader::File::open(&path).unwrap();
        let ds = f.dataset("keep").unwrap();
        let mut pages: Vec<u64> = ds
            .chunks()
            .unwrap()
            .iter()
            .filter(|c| c.storage_size > 0)
            .flat_map(|c| (c.address / page)..=((c.address + c.storage_size - 1) / page))
            .collect();
        pages.sort_unstable();
        pages.dedup();
        pages
    };
    assert!(!live_raw_pages.is_empty(), "expected live raw pages");

    let s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let sections = s
        .space
        .paged_sections()
        .expect("a paged file installs paged state");
    for &extent in sections.metadata() {
        for p in (extent.start() / page)..=((extent.end() - 1) / page) {
            assert!(
                !live_raw_pages.contains(&p),
                "metadata free section ({}, {}) sits in page {p}, which still holds live raw \
                 chunk data",
                extent.start(),
                extent.len()
            );
        }
    }
    let reclaimed: u64 = s.space.reusable_sections().map(|extent| extent.len()).sum();
    assert!(reclaimed > 0, "the delete reclaimed nothing");
}

#[test]
fn failed_paged_commit_leaves_the_free_lists_untouched() {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("paged_failed_commit.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep")
        .with_i32_data(&(0..200).collect::<Vec<i32>>())
        .with_shape(&[200]);
    b.create_dataset("drop")
        .with_i32_data(&(0..200).collect::<Vec<i32>>())
        .with_shape(&[200]);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(0)
            .with_page_size(FileSpacePageSize::DEFAULT),
    );
    b.write(&path).unwrap();

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    let before = s.reusable_free_sections().unwrap();
    let good_ext = s.context.superblock().superblock_extension_address;
    let mut superblock = s.context.superblock().clone();
    superblock.superblock_extension_address = Some(StoredAddress::ZERO);
    let proposed = s.context.propose(superblock).unwrap();
    s.context.adopt(proposed).unwrap();
    s.delete(&ObjectPathBuf::parse("/drop")).unwrap();
    assert!(
        s.commit().is_err(),
        "a commit with an unreadable extension must fail"
    );

    assert_eq!(
        s.reusable_free_sections().unwrap(),
        before,
        "a failed commit must not record still-live regions as free"
    );
    let mut superblock = s.context.superblock().clone();
    superblock.superblock_extension_address = good_ext;
    let proposed = s.context.propose(superblock).unwrap();
    s.context.adopt(proposed).unwrap();
    s.delete(&ObjectPathBuf::parse("/drop")).unwrap();
    s.commit()
        .expect("the session is usable after a failed commit");
    drop(s);

    let f = crate::reader::File::open(&path).unwrap();
    let kept = f.dataset("keep").unwrap().read_i32().unwrap();
    assert_eq!(kept, (0..200).collect::<Vec<i32>>(), "keep survives intact");
    let freed: u64 = f
        .persisted_free_sections()
        .unwrap()
        .unwrap()
        .free_sections()
        .free_bytes();
    let live_end = f.file_size();
    assert!(
        freed < live_end,
        "the recorded free space cannot cover the whole file"
    );
}

#[rstest]
#[case::fsm_aggr(FileSpaceStrategy::FsmAggr, 0)]
#[case::fsm_aggr_userblock(FileSpaceStrategy::FsmAggr, 512)]
#[case::page(FileSpaceStrategy::Page, 0)]
#[case::page_userblock(FileSpaceStrategy::Page, 4096)]
fn failed_persistent_commit_preserves_live_bytes_and_reusable_space(
    #[case] strategy: FileSpaceStrategy,
    #[case] userblock: u64,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("failed_persisting_commit.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UserblockSize::try_from(userblock).unwrap());
    b.create_dataset("a").with_i32_data(&[1; 100]);
    b.create_dataset("b").with_i32_data(&[2; 400]);
    b.create_dataset("c").with_i32_data(&[3; 100]);
    b.with_file_space_settings(
        FileSpaceSettings::new(strategy)
            .with_persistence(true)
            .with_page_size(FileSpacePageSize::DEFAULT),
    );
    b.write(&path).unwrap();

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    s.delete(&ObjectPathBuf::parse("/b")).unwrap();
    s.commit().unwrap();
    let eof = s.context.superblock().eof_address;
    let free = s.reusable_free_sections().unwrap();
    let live_bytes = |s: &WriteEngine| {
        let mut bytes = s
            .image()
            .read_exact_at(0, usize::try_from(eof).unwrap())
            .unwrap();
        for section in free.sections() {
            assert_eq!(section.coordinates(), FreeSectionCoordinates::Absolute);
            let at = usize::try_from(section.address()).unwrap();
            bytes[at..at + usize::try_from(section.length().get()).unwrap()].fill(0);
        }
        bytes
    };
    let before = live_bytes(&s);
    let summary_before = s.reusable_space_stats().unwrap();

    let good_ext = s.context.superblock().superblock_extension_address;
    let mut superblock = s.context.superblock().clone();
    superblock.superblock_extension_address = Some(StoredAddress::ZERO);
    let proposed = s.context.propose(superblock).unwrap();
    s.context.adopt(proposed).unwrap();
    s.delete(&ObjectPathBuf::parse("/c")).unwrap();
    let err = s.commit().unwrap_err();
    assert!(
        matches!(
            err,
            Error::EditUnsupported("an object does not use a version 2 object header")
        ),
        "{err:?}"
    );

    assert_eq!(s.context.superblock().eof_address, eof);
    assert_eq!(s.reusable_free_sections().unwrap(), free);
    let summary_after = s.reusable_space_stats().unwrap();
    assert_eq!(summary_after.logical_size(), s.image.len());
    let sections = s.reusable_free_sections().unwrap();
    assert_eq!(
        summary_after.reusable_free_bytes(),
        summary_before.reusable_free_bytes()
    );
    test_util_hdf5::assert_free_sections_consistent!(
        &summary_after,
        &sections,
        FreeSectionCoordinates::Absolute
    );
    assert!(
        live_bytes(&s) == before,
        "the failed commit changed live bytes"
    );

    let mut superblock = s.context.superblock().clone();

    superblock.superblock_extension_address = good_ext;

    let proposed = s.context.propose(superblock).unwrap();
    s.context.adopt(proposed).unwrap();
    s.delete(&ObjectPathBuf::parse("/c")).unwrap();
    s.commit().unwrap();
    drop(s);
    let f = crate::reader::File::open(&path).unwrap();
    assert_eq!(f.dataset("a").unwrap().read_i32().unwrap(), vec![1; 100]);
    assert_eq!(f.file_size(), std::fs::metadata(&path).unwrap().len());
}

#[test]
fn a_failed_commit_leaves_no_reusable_span_the_image_names() {
    use crate::writer::FileBuilder;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("failed_commit_live_span.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("labels")
        .with_vlen_strings(&["seed-one", "seed-two"]);
    b.create_dataset("big").with_u8_data(&vec![0x5A; 40960]);
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(0),
    );
    b.write(&path).unwrap();

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    s.delete(&ObjectPathBuf::parse("/big")).unwrap();
    s.commit().unwrap();

    let round_one = ["round-one-aaaa", "round-one-bbbb"];
    s.stage_dataset_write(&ObjectPathBuf::parse("/labels"), {
        let mut db = crate::type_builders::DatasetBuilder::new("");
        db.with_vlen_strings(&round_one);
        db
    })
    .unwrap();
    s.commit().unwrap();
    let good_ext = s.context.superblock().superblock_extension_address;
    let mut superblock = s.context.superblock().clone();
    superblock.superblock_extension_address = Some(StoredAddress::ZERO);
    let proposed = s.context.propose(superblock).unwrap();
    s.context.adopt(proposed).unwrap();
    s.stage_dataset_write(&ObjectPathBuf::parse("/labels"), {
        let mut db = crate::type_builders::DatasetBuilder::new("");
        db.with_vlen_strings(&["round-two-aaaa", "round-two-bbbb"]);
        db
    })
    .unwrap();
    assert!(
        s.commit().is_err(),
        "a commit with an unreadable extension must fail"
    );
    let mut superblock = s.context.superblock().clone();
    superblock.superblock_extension_address = good_ext;
    let proposed = s.context.propose(superblock).unwrap();
    s.context.adopt(proposed).unwrap();
    for i in 0..4 {
        s.stage_created_dataset(&ObjectPathBuf::parse(&format!("/filler{i}")), {
            let mut db = crate::type_builders::DatasetBuilder::new("");
            db.with_vlen_strings(&["XXXXXXXXXXXXXXXXXXXX", "YYYYYYYYYYYYYYYYYYYY"]);
            db
        })
        .unwrap();
        s.commit().unwrap();
    }
    drop(s);

    let f = crate::reader::File::open(&path).unwrap();
    let got = f.dataset("labels").unwrap().read_string();
    assert!(
        got.as_ref()
            .is_ok_and(|v| v.iter().map(String::as_str).eq(round_one)),
        "/labels reads another dataset's heap objects: {got:?}"
    );
}

#[test]
fn a_commit_after_an_append_pads_the_raw_page_the_append_left() {
    const PAGE: u64 = 4096;
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("paged_interleave.h5");
    let mut b = crate::writer::FileBuilder::new();
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(0)
            .with_page_size(FileSpacePageSize::try_from(PAGE).unwrap()),
    );
    b.create_dataset("d")
        .with_i32_data(&(0..64).collect::<Vec<i32>>())
        .with_shape(&[64])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[64]);
    b.write(&p).unwrap();

    let mut engine = WriteEngine::open_with_locking(&p, FileLocking::Enabled).unwrap();
    let mut ab = AppendBuilder::new();
    ab.append_i32(&(64..2000).collect::<Vec<_>>());
    engine
        .append_inplace_gathered(AppendTarget::Path(&ObjectPathBuf::parse("d")), &ab, 4)
        .unwrap();

    assert_eq!(
        engine.space.paged_tail_type(),
        Some(Some(PageType::Raw)),
        "the append must record that the tail page now holds raw data"
    );
    assert_ne!(
        engine.image.len() % PAGE,
        0,
        "the append must leave a partially-filled page for the commit to pad"
    );

    engine.create_group(&ObjectPathBuf::parse("g")).unwrap();
    engine.commit().unwrap();
    let raw_free = engine
        .space
        .paged_sections()
        .expect("the file is paged")
        .raw()
        .to_vec();
    assert!(
        !raw_free.is_empty(),
        "the commit packed metadata into the raw page the append left open"
    );
    for extent in raw_free {
        assert_eq!(
            extent.end() % PAGE,
            0,
            "padding {}+{} does not reach a page boundary",
            extent.start(),
            extent.len()
        );
    }

    drop(engine);
    assert_eq!(
        crate::File::open(&p)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_i32()
            .unwrap(),
        (0..2000).collect::<Vec<_>>()
    );
}

#[test]
fn an_inplace_append_to_a_paged_file_allocates_only_raw_pages() {
    const PAGE: u64 = 4096;
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("paged_raw.h5");
    let mut b = crate::writer::FileBuilder::new();
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_threshold(0)
            .with_page_size(FileSpacePageSize::try_from(PAGE).unwrap()),
    );
    b.create_dataset("d")
        .with_i32_data(&(0..64).collect::<Vec<i32>>())
        .with_shape(&[64])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[64]);
    b.write(&p).unwrap();

    let mut engine = WriteEngine::open_with_locking(&p, FileLocking::Enabled).unwrap();
    let before = engine.image().len();
    for range in [64..2000, 2000..4000] {
        let mut ab = AppendBuilder::new();
        ab.append_i32(&range.collect::<Vec<_>>());
        engine
            .append_inplace_gathered(AppendTarget::Path(&ObjectPathBuf::parse("d")), &ab, 4)
            .unwrap();
    }

    assert_eq!(
        engine.space.paged_tail_type(),
        Some(Some(PageType::Raw)),
        "the append left the tail page holding something other than raw data"
    );
    let meta_padding = engine
        .space
        .pending_padding(PageType::Meta)
        .expect("the file is paged");
    let raw_padding = engine
        .space
        .pending_padding(PageType::Raw)
        .expect("the file is paged");
    assert!(
        meta_padding.is_empty() && raw_padding.is_empty(),
        "an in-place append switched page type: meta_pad={meta_padding:?} raw_pad={raw_padding:?}"
    );
    let addr = crate::group_v2::resolve_path_any_from_source(
        &engine.image(),
        AccessMode::ReadWrite,
        engine.superblock(),
        engine.widths(),
        engine.addresses(),
        &ObjectPath::parse("d"),
    )
    .unwrap();
    let spans = engine
        .chunked_storage_spans(addr)
        .expect("a chunked dataset has reclaimable spans");
    let fresh = spans.iter().filter(|&&(a, _, _)| a >= before).count();
    assert!(
        fresh > 0,
        "the append allocated nothing above {before}, so the assertion above proves nothing"
    );
    assert!(
        spans
            .iter()
            .all(|&(_, _, class)| class == FreeClass::Page(PageType::Raw)),
        "the reclaim tags every chunked span raw; a metadata or dead tag here would \
         need the placement rule above to change with it"
    );
}

#[test]
fn gather_page_size_matches_file_space_properties() {
    let dir = tempfile::tempdir().unwrap();
    let paged = dir.path().join("paged.h5");
    let plain = dir.path().join("plain.h5");
    common::gather_fixture(&paged, 1, true);
    common::gather_fixture(&plain, 1, false);

    let s = WriteEngine::open_with_locking(&paged, FileLocking::Enabled).unwrap();
    assert_eq!(
        s.gather_page_size(),
        16 * 1024,
        "a paged file merges within its own file-space page"
    );
    assert_ne!(
        16 * 1024,
        DEFAULT_GATHER_PAGE,
        "the fixture must not be built at the default, or the assertion above \
         holds for a session that ignores the file entirely"
    );
    drop(s);

    let s = WriteEngine::open_with_locking(&plain, FileLocking::Enabled).unwrap();
    assert_eq!(
        s.gather_page_size(),
        DEFAULT_GATHER_PAGE,
        "an unpaged file has no page size of its own"
    );
}

#[test]
fn the_append_reserve_is_reported_as_one_coalesced_free_list() {
    use crate::writer::FileBuilder;
    const VICTIM: usize = (APPEND_RESERVE_BYTES as usize * 3) / 4;
    const CHUNK: u64 = 16384;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reserve_coalesced.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    b.create_dataset("t0")
        .with_i32_data(&[0i32])
        .with_shape(&[1])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[CHUNK]);
    b.create_dataset("victim")
        .with_i32_data(&vec![7i32; VICTIM]);
    b.create_dataset("ceiling").with_i32_data(&[9i32, 9, 9]);
    b.write(&path).unwrap();

    let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
    s.set_sync_policy(SyncPolicy::OnClose);
    s.delete(&ObjectPathBuf::parse("victim")).unwrap();
    s.commit().unwrap();

    let batch: Vec<i32> = (0..CHUNK as i32).collect();
    let mut reserved_ever = false;
    for round in 0..16 {
        s.append_inplace_i32_phased("t0", &batch, 4).unwrap();
        reserved_ever |= !s.reserved.is_empty();
        let sections = s.reusable_free_sections().unwrap();
        for pair in sections.sections().windows(2) {
            assert!(
                pair[0].end_address() < pair[1].address(),
                "round {round}: reusable sections touch or overlap: {sections:?}"
            );
        }
        let stats = s.reusable_space_stats().unwrap();
        test_util_hdf5::assert_free_sections_consistent!(
            &stats,
            &sections,
            FreeSectionCoordinates::Absolute
        );
    }
    assert!(
        reserved_ever,
        "no reserve was ever held, so this measured the plain free list"
    );
}

#[test]
fn an_unspent_append_reserve_goes_back_to_the_managers() {
    use crate::writer::FileBuilder;
    const VICTIM: usize = (APPEND_RESERVE_BYTES as usize * 2) / 4;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reserve_return.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    b.create_dataset("t0")
        .with_i32_data(&(0..256).collect::<Vec<_>>())
        .with_shape(&[256])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[64]);
    b.create_dataset("victim")
        .with_i32_data(&vec![7i32; VICTIM]);
    b.create_dataset("ceiling").with_i32_data(&[9i32, 9, 9]);
    b.write(&path).unwrap();

    let persisted = |p: &std::path::Path| -> u64 {
        crate::reader::File::open(p)
            .unwrap()
            .persisted_free_sections()
            .unwrap()
            .unwrap()
            .free_sections()
            .free_bytes()
    };

    {
        let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        s.delete(&ObjectPathBuf::parse("victim")).unwrap();
        s.commit().unwrap();
    }
    let before = persisted(&path);
    assert!(
        before > APPEND_RESERVE_BYTES,
        "the fixture must leave more than one reserve batch on disk, not {before} bytes"
    );
    let appended: Vec<i32> = (0..1024).collect();
    {
        let mut s = WriteEngine::open_with_locking(&path, FileLocking::Enabled).unwrap();
        s.set_sync_policy(SyncPolicy::OnClose);
        s.append_inplace_i32_phased("t0", &appended, 4).unwrap();
        s.finalize_persist().unwrap();
    }

    let after = persisted(&path);
    let spent = 64 * 1024;
    assert!(
        after + spent >= before,
        "the session reserved {APPEND_RESERVE_BYTES} bytes and spent a few of them, \
         so the managers should still describe nearly all of the {before} they did \
         before — they describe {after}"
    );
    assert!(
        after < before,
        "the append placed bytes inside the hole, so the managers must describe \
         less than the {before} they did (they describe {after})"
    );

    let f = crate::reader::File::open(&path).unwrap();
    let mut want = (0..256).collect::<Vec<i32>>();
    want.extend_from_slice(&appended);
    assert_eq!(f.dataset("t0").unwrap().read_i32().unwrap(), want);
    assert_eq!(f.dataset("ceiling").unwrap().read_i32().unwrap(), [9, 9, 9]);
}

#[test]
fn a_draw_gathers_every_hole_the_append_fits_in() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("draw_gathers.h5");
    let mut s = fragmented_persisting_fixture(&path, 3, 4 * 64);
    let before: Vec<_> = s.space.reusable_sections().collect();
    assert!(
        before.len() >= 3,
        "the fixture must leave three separate holes, not {before:?}"
    );

    s.append_inplace_i32_phased("t0", &[1i32; 64], 4).unwrap();

    let held: Vec<_> = s
        .reserved
        .extents()
        .iter()
        .map(|extent| (extent.start(), extent.len()))
        .collect();
    assert!(
        held.len() >= 2,
        "one draw should have gathered runs from several of the holes {before:?}, \
         not only {held:?}"
    );
    let drawn: u64 = held.iter().map(|&(_, len)| len).sum();
    assert!(
        drawn > before.iter().map(|extent| extent.len()).max().unwrap(),
        "the reserve holds {drawn} bytes, no more than the largest hole alone"
    );
}

#[test]
fn append_reservation_rewrite_uses_reserved_free_space() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("draw_tail.h5");
    let mut s = fragmented_persisting_fixture(&path, 1, 8 * 64);
    let len_before = s.image.len();

    s.append_inplace_i32_phased("t0", &[1i32; 64], 4).unwrap();

    assert!(
        !s.reserved.is_empty(),
        "no reserve was drawn, so this measured an ordinary end-of-file append"
    );
    let ext = s
        .context
        .addresses()
        .absolute(s.context.superblock().superblock_extension_address.unwrap())
        .unwrap();
    assert!(
        ext < len_before,
        "the rewrite appended its tail at {ext}, the pre-append end-of-file \
         {len_before}, instead of placing it in the hole it was publishing as taken"
    );
    assert!(
        s.image.len() <= len_before,
        "a rewrite that placed its tail inside the file must not have grown it \
         (was {len_before}, now {})",
        s.image.len()
    );
}

fn section_covers(sections: &[FreeExtent], addr: u64, len: u64) -> bool {
    let Some(end) = addr.checked_add(len) else {
        return false;
    };
    sections
        .iter()
        .any(|extent| extent.start() <= addr && extent.end() >= end)
}

fn write_dense_attribute_owner(path: &Path, huge_attrs: usize, strategy: FileSpaceStrategy) {
    let mut builder = FileBuilder::new();
    let mut group = builder.create_group("g");
    for i in 0..12 {
        group.set_attr(&format!("a{i}"), AttrValue::I64(i));
    }
    for i in 0..huge_attrs {
        group.set_attr(
            &format!("huge{i}"),
            AttrValue::AsciiString("y".repeat(70_000)),
        );
    }
    builder.add_group(group.finish());
    builder
        .create_dataset("keep")
        .with_i32_data(&(0..1024).collect::<Vec<_>>());
    let settings = FileSpaceSettings::new(strategy)
        .with_persistence(true)
        .with_threshold(0);
    builder.with_file_space_settings(if strategy == FileSpaceStrategy::Page {
        settings.with_page_size(FileSpacePageSize::try_from(DENSE_ATTRIBUTE_PAGE_SIZE).unwrap())
    } else {
        settings
    });
    builder.write(path).unwrap();
}

struct DenseAttributeOwner {
    addr: u64,
    header_spans: Vec<(u64, u64)>,
    storage: Vec<(u64, u64, FreeClass)>,
}

fn dense_attribute_owner(session: &WriteEngine, object: &ObjectPathBuf) -> DenseAttributeOwner {
    let addr = crate::group_v2::resolve_path_any_from_source(
        &session.image(),
        AccessMode::ReadWrite,
        session.superblock(),
        session.widths(),
        session.addresses(),
        &object.as_path(),
    )
    .unwrap();
    let header_spans = session.oh_chunk_spans(addr).unwrap();
    let mut storage = session
        .dense_attribute_storage_spans(&header_region(session, addr), &header_spans)
        .expect("the dense attribute storage is structurally complete");
    storage.sort_unstable_by_key(|&(storage_addr, _, _)| storage_addr);
    DenseAttributeOwner {
        addr,
        header_spans,
        storage,
    }
}

fn header_region(session: &WriteEngine, addr: u64) -> OhRegion {
    WriteEngine::gather_oh_messages(&session.image(), addr, session.addresses()).unwrap()
}

fn attribute_heap_address(session: &WriteEngine, addr: u64) -> u64 {
    let info = edit::attributes::find_attribute_info(&header_region(session, addr))
        .unwrap()
        .expect("a dense object has Attribute Info")
        .2;
    session
        .addresses()
        .absolute(
            info.fractal_heap_address
                .expect("the heap address is defined"),
        )
        .unwrap()
}

fn reusable_covers(session: &WriteEngine, addr: u64, len: u64) -> bool {
    session
        .space
        .reusable_sections()
        .any(|extent| extent.start() <= addr && extent.end() >= addr + len)
}

fn reusable_overlaps(session: &WriteEngine, addr: u64, len: u64) -> bool {
    session
        .space
        .reusable_sections()
        .any(|extent| extent.start() < addr + len && addr < extent.end())
}

/// Returns whether each page intersection is free metadata or part of a whole free raw page.
fn freed_as_metadata(sections: &PagedSections, addr: u64, len: u64) -> bool {
    let page_size = DENSE_ATTRIBUTE_PAGE_SIZE;
    let end = addr + len;
    (addr / page_size..=(end - 1) / page_size).all(|page| {
        let page_start = page * page_size;
        let start = addr.max(page_start);
        let stop = end.min(page_start + page_size);
        section_covers(sections.metadata(), start, stop - start)
            || section_covers(sections.raw(), page_start, page_size)
    })
}

const DENSE_ATTRIBUTE_PAGE_SIZE: u64 = 4096;

/// Writes a zero-filled userblock before the file contents and updates its base and EOF addresses.
fn prepend_userblock(path: &Path, len: u64) {
    let original = std::fs::read(path).unwrap();
    let mut superblock = crate::superblock::read(original.as_slice())
        .unwrap()
        .superblock;
    superblock.base_address = BaseAddress::new(len);
    superblock.eof_address += len;
    let rewritten = format::serialize_superblock(&superblock).unwrap();
    let prefix = usize::try_from(len).unwrap();
    let mut bytes = vec![0; prefix];
    bytes.extend_from_slice(&original);
    bytes[prefix..prefix + rewritten.len()].copy_from_slice(&rewritten);
    std::fs::write(path, bytes).unwrap();
}

/// Damage applied to a dense attribute heap before reclamation.
#[derive(Clone, Copy, Debug)]
enum HeapDamage {
    /// A direct block whose recorded owner differs from its heap header.
    DirectBlockOwner,
    /// A free-space manager address at the end of the file.
    FreeSpaceManager,
    /// A declared deflate pipeline with raw bytes in the root direct block.
    UnencodedFilteredBlock,
}

struct RejectedHeapOwner {
    header_spans: Vec<(u64, u64)>,
    index: Vec<(u64, u64, FreeClass)>,
    heap: FractalHeapStorage,
}

/// Writes and damages a dense attribute owner, returning its original storage allocations.
fn write_rejected_heap_owner(
    path: &Path,
    object: &ObjectPathBuf,
    damage: HeapDamage,
) -> RejectedHeapOwner {
    write_dense_attribute_owner(path, 0, FileSpaceStrategy::FsmAggr);
    let (addr, header_spans, storage, heap_address, heap) = {
        let session = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
        let DenseAttributeOwner {
            addr,
            header_spans,
            storage,
        } = dense_attribute_owner(&session, object);
        let heap_address = attribute_heap_address(&session, addr);
        let heap = crate::fractal_heap_storage::collect_fractal_heap_storage_from_source(
            &session.image(),
            AddressContext::from_effective_base(BaseAddress::ZERO),
            StoredAddress::new(heap_address),
            crate::file_writer::WIDTHS,
        )
        .unwrap();
        (addr, header_spans, storage, heap_address, heap)
    };

    let mut bytes = std::fs::read(path).unwrap();
    let heap_at = usize::try_from(heap_address).unwrap();
    let mut header = FractalHeapHeader::parse(&bytes, heap_at, OFFSET_SIZE, LENGTH_SIZE).unwrap();
    let rejections = match damage {
        HeapDamage::DirectBlockOwner => {
            assert_eq!(header.current_rows_in_root_indirect_block, 0);
            let owner_at = usize::try_from(header.root_block_address.get()).unwrap()
                + DIRECT_BLOCK_HEAP_ADDRESS_AT;
            bytes[owner_at..owner_at + usize::from(OFFSET_SIZE)]
                .copy_from_slice(&(heap_address + 1).to_le_bytes());
            vec![FractalHeapStorageError::Format(
                FormatError::InvalidFractalHeapBlockOwner {
                    expected: heap_address,
                    actual: heap_address + 1,
                },
            )]
        }
        HeapDamage::UnencodedFilteredBlock => {
            filter_attribute_heap(&mut bytes, 0, heap_address, false);
            crate::test_data::DEFLATE_HEADER_ERRORS
                .map(|reason| {
                    FractalHeapStorageError::Format(FormatError::FilterError(reason.into()))
                })
                .to_vec()
        }
        HeapDamage::FreeSpaceManager => {
            header.managed_block_free_space_manager_address =
                StoredAddress::new(u64::try_from(bytes.len()).unwrap());
            let encoded = header
                .serialize(OffsetWidth::Eight, LengthWidth::Eight)
                .unwrap();
            bytes[heap_at..heap_at + encoded.len()].copy_from_slice(&encoded);
            vec![FractalHeapStorageError::Format(
                FormatError::UnexpectedSourceEof {
                    expected: bytes.len().to_u64()
                        + format::free_space_manager_header_len(
                            FormatWidths::from_sizes(OFFSET_SIZE, LENGTH_SIZE).unwrap(),
                        ),
                    available: bytes.len().to_u64(),
                },
            )]
        }
    };
    let err = crate::fractal_heap_storage::collect_fractal_heap_storage_from_source(
        &BytesSource::new(bytes.as_slice()),
        AddressContext::from_effective_base(BaseAddress::ZERO),
        StoredAddress::new(heap_address),
        crate::file_writer::WIDTHS,
    )
    .unwrap_err();
    assert!(rejections.contains(&err), "{err:?}");
    std::fs::write(path, bytes).unwrap();

    let index: Vec<(u64, u64, FreeClass)> = storage
        .into_iter()
        .filter(|&(addr, len, _)| {
            heap.allocations()
                .iter()
                .all(|(extent, _)| (extent.start(), extent.len()) != (addr, len))
        })
        .collect();
    let session = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
    assert_eq!(
        session.dense_attribute_storage_spans(&header_region(&session, addr), &header_spans),
        Some(index.clone())
    );
    RejectedHeapOwner {
        header_spans,
        index,
        heap,
    }
}

/// The offset of the Heap Header Address in a version 0 fractal heap direct block.
///
/// The address follows the signature and version in "Fractal Heap" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_fractalheap
const DIRECT_BLOCK_HEAP_ADDRESS_AT: usize =
    test_util::fractal_heap::DIRECT_BLOCK_SIGNATURE.len() + 1;

/// Adds a deflate pipeline and moves the root direct block after the expanded heap header.
///
/// With `encode`, the block is compressed. Otherwise, its first [`UNENCODED_BLOCK_LEN`]
/// bytes are copied without encoding. `base` is the userblock length. Returns the absolute
/// positions and lengths of the rewritten heap header and root block.
fn filter_attribute_heap(
    bytes: &mut [u8],
    base: u64,
    heap_address: u64,
    encode: bool,
) -> Vec<(u64, u64)> {
    let heap_at = usize::try_from(heap_address).unwrap();
    let header = FractalHeapHeader::parse(bytes, heap_at, OFFSET_SIZE, LENGTH_SIZE).unwrap();
    assert_eq!(header.current_rows_in_root_indirect_block, 0);
    let root_at = usize::try_from(header.root_block_address.get() + base).unwrap();
    let block_len = usize::try_from(header.starting_block_size).unwrap();
    let block = bytes[root_at..root_at + block_len].to_vec();
    let pipeline = crate::test_data::deflate_pipeline();
    let encoded = if encode {
        crate::test_data::encode_block(&block, &pipeline)
    } else {
        block[..UNENCODED_BLOCK_LEN].to_vec()
    };
    let pipeline = pipeline.serialize().unwrap();
    let filtered_header = |root: u64| {
        let unfiltered = FractalHeapHeader {
            root_block_address: StoredAddress::new(root),
            ..header.clone()
        }
        .serialize(OffsetWidth::Eight, LengthWidth::Eight)
        .unwrap();
        fractal_heap::with_filtering(
            &unfiltered,
            Widths::EIGHT,
            u64::try_from(encoded.len()).unwrap(),
            0,
            &pipeline,
        )
    };
    let header_len = filtered_header(header.root_block_address.get()).len();
    let filtered_root = heap_at + header_len;
    assert!(filtered_root + encoded.len() <= root_at + block_len);
    let filtered = filtered_header(u64::try_from(filtered_root).unwrap() - base);
    bytes[heap_at..heap_at + header_len].copy_from_slice(&filtered);
    bytes[filtered_root..filtered_root + encoded.len()].copy_from_slice(&encoded);
    vec![
        (heap_address, u64::try_from(header_len).unwrap()),
        (
            u64::try_from(filtered_root).unwrap(),
            u64::try_from(encoded.len()).unwrap(),
        ),
    ]
}

const UNENCODED_BLOCK_LEN: usize = 64;

/// Asserts complete ownership analysis and coverage of every selected allocation.
///
/// With `metadata`, coverage must come entirely from live metadata. Otherwise, both live
/// metadata and raw storage count toward coverage.
///
/// # Panics
///
/// Panics if analysis fails, is incomplete, or the required storage classes do not cover
/// every selected byte.
fn assert_ownership_covers_storage(
    session: &WriteEngine,
    storage: &[(u64, u64, FreeClass)],
    metadata: bool,
) {
    let owned = session.space_ownership().unwrap();
    assert!(owned.analysis_complete);
    for &(at, len, _) in storage {
        let covered = test_util::range::covered_len(&owned.live_metadata_space, &(at..at + len));
        let raw = if metadata {
            0
        } else {
            test_util::range::covered_len(&owned.live_raw_space, &(at..at + len))
        };
        assert_eq!(
            covered + raw,
            len,
            "heap storage {at}..{} must be classified",
            at + len
        );
    }
}

/// Opens a persistent session with an append target and separate interior free regions.
///
/// Each deleted dataset is followed by a retained dataset to prevent coalescing. A final
/// retained dataset prevents the free regions from being truncated.
fn fragmented_persisting_fixture(
    path: &std::path::Path,
    victims: usize,
    victim_elems: usize,
) -> WriteEngine {
    use crate::writer::FileBuilder;

    let mut b = FileBuilder::new();
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    b.create_dataset("t0")
        .with_i32_data(&[0i32])
        .with_shape(&[1])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[64]);
    for v in 0..victims {
        b.create_dataset(&std::format!("victim{v}"))
            .with_i32_data(&vec![7i32; victim_elems]);
        b.create_dataset(&std::format!("keeper{v}"))
            .with_i32_data(&[v as i32]);
    }
    b.create_dataset("ceiling").with_i32_data(&[9i32, 9, 9]);
    b.write(path).unwrap();

    let mut s = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
    s.set_sync_policy(SyncPolicy::OnClose);
    for v in 0..victims {
        s.delete(&ObjectPathBuf::parse(&std::format!("victim{v}")))
            .unwrap();
    }
    s.commit().unwrap();
    s
}
