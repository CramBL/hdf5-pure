use core::cell::Cell;
use std::fs;
use std::sync::atomic::Ordering;

use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::btree::v2 as btree_v2;
use test_util::bytes;
use test_util::data_layout;
use test_util::dense_attributes;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::object_header::v1;
use test_util::object_header::v2;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use super::AllocationKind;
use super::OWNERSHIP_READS;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::DISCOVERY_WORK;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::error::FormatError;
use crate::object_graph::ObjectId;
use crate::type_builders;

#[rstest]
#[case::compact(Form::CompactDataset)]
#[case::contiguous(Form::ContiguousDataset)]
fn local_ownership_is_lazy_cached_and_preserves_complete_allocations(#[case] form: Form) {
    let path = test_util::temp::temp_path("local-ownership.h5");
    let mut fixture = Fixture::new();
    let raw = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    let records = form.messages(
        &mut fixture,
        &type_builders::make_i32_type(),
        &[7_i32.to_le_bytes(), 11_i32.to_le_bytes()].concat(),
    );
    let target = fixture.append_header(records);
    let end = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    fixture.link("data", target);
    fixture.write(&path);
    let (engine, calls) = edit_test_util::open_counted_engine(&path);
    calls.store(0, Ordering::Relaxed);
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    OWNERSHIP_READS.with(|reads| reads.set(0));
    let free = engine.reusable_free_sections().unwrap();
    let mut phase = AnalysisPhase::new(&engine);
    let origin = phase
        .reconstruction_source(&engine, ObjectId::new(PhysicalAddress::new(target)))
        .unwrap();
    assert!(phase.ownership.is_empty());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let observed = phase.object_ownership(&origin).unwrap();
    assert!(observed.is_complete());
    assert!(observed.error().is_none());
    let ranges = observed
        .allocations()
        .iter()
        .map(|allocation| {
            (
                allocation.kind(),
                allocation.range().start().get(),
                allocation.range().length(),
            )
        })
        .collect::<Vec<_>>();
    let mut expected = vec![(AllocationKind::Header, target, end - target)];
    if matches!(form, Form::ContiguousDataset) {
        expected.push((AllocationKind::Data, raw, 8));
    }
    assert_eq!(ranges, expected);
    let reads = calls.load(Ordering::Relaxed);
    assert!(reads > 0);
    assert!(phase.object_ownership(&origin).unwrap().is_complete());
    assert_eq!(calls.load(Ordering::Relaxed), reads);
    assert_eq!(OWNERSHIP_READS.with(Cell::get), 1);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (0, 0));
    assert_eq!(engine.reusable_free_sections().unwrap(), free);
    assert!(phase.topology().is_some());
    let discovered = DISCOVERY_WORK.with(Cell::get);
    assert_eq!(discovered, (1, 2));
    assert!(phase.object_ownership(&origin).unwrap().is_complete());
    assert_eq!(DISCOVERY_WORK.with(Cell::get), discovered);
    assert_eq!(OWNERSHIP_READS.with(Cell::get), 1);
}

#[rstest]
#[case::unknown_message(Completeness::Unknown)]
#[case::object_reference(Completeness::Reference)]
#[case::bad_layout(Completeness::BadLayout)]
fn local_ownership_keeps_partial_allocations_separate_from_other_facts(#[case] mode: Completeness) {
    let path = test_util::temp::temp_path("partial-local-ownership.h5");
    let mut fixture = Fixture::new();
    let datatype = if matches!(mode, Completeness::Reference) {
        global_heap_fixture::object_reference()
    } else {
        type_builders::make_i32_type()
    };
    let mut records = Form::CompactDataset.messages(&mut fixture, &datatype, &[0; 8]);
    match mode {
        Completeness::BadLayout => {
            records
                .iter_mut()
                .find(|record| record.msg_type == MessageType::DATA_LAYOUT)
                .unwrap()
                .data = vec![0xff, 0];
        }
        Completeness::Unknown => records.push(Message::new(MessageType(0x7f), &[])),
        Completeness::Reference => {}
    }
    let target = fixture.append_header(records);
    fixture.link("data", target);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let origin = phase
        .reconstruction_source(&engine, ObjectId::new(PhysicalAddress::new(target)))
        .unwrap();
    OWNERSHIP_READS.with(|reads| reads.set(0));
    let observed = phase.object_ownership(&origin).unwrap();
    assert_eq!(
        observed.is_complete(),
        matches!(mode, Completeness::Reference)
    );
    assert_eq!(observed.allocations().len(), 1);
    assert_eq!(observed.allocations()[0].kind(), AllocationKind::Header);
    match mode {
        Completeness::BadLayout => assert!(matches!(
            observed.error().unwrap(),
            Error::Format(FormatError::InvalidLayoutVersion(0xff))
        )),
        Completeness::Reference | Completeness::Unknown => assert!(observed.error().is_none()),
    }
    phase.object_ownership(&origin).unwrap();
    assert_eq!(OWNERSHIP_READS.with(Cell::get), 1);
    assert!(phase.topology().is_some());
    if !matches!(mode, Completeness::BadLayout) {
        assert!(
            !phase
                .collections_reachable_from([ObjectId::new(PhysicalAddress::new(Fixture::ROOT))])
                .unwrap()
                .analysis_complete
        );
    }
}

#[rstest]
#[case::same_session(false)]
#[case::other_session(true)]
fn local_ownership_rejects_other_source_associations_before_reads(#[case] other_session: bool) {
    let path = test_util::temp::temp_path("ownership-association.h5");
    let mut fixture = Fixture::new();
    let target = fixture.append_header(vec![]);
    fixture.link("data", target);
    fixture.write(&path);
    let other_path = test_util::temp::temp_path("ownership-other-session.h5");
    fs::copy(&path, &other_path).unwrap();
    let (engine, original_calls) = edit_test_util::open_counted_engine(&path);
    let other = other_session.then(|| edit_test_util::open_counted_engine(&other_path));
    let mut first = AnalysisPhase::new(&engine);
    let origin = first
        .reconstruction_source(&engine, ObjectId::new(PhysicalAddress::new(target)))
        .unwrap();
    let (query_engine, calls) = other
        .as_ref()
        .map(|(engine, calls)| (engine, calls))
        .unwrap_or((&engine, &original_calls));
    let mut second = AnalysisPhase::new(query_engine);
    let own = second
        .reconstruction_source(query_engine, ObjectId::new(PhysicalAddress::new(target)))
        .unwrap();
    calls.store(0, Ordering::Relaxed);
    let Err(error) = second.object_ownership(&origin) else {
        panic!("expected an ownership source-phase mismatch")
    };
    assert_eq!(
        error,
        FormatError::Internal("ownership query belongs to another source phase".into())
    );
    assert!(second.ownership.is_empty());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(second.object_ownership(&own).unwrap().is_complete());
}

#[test]
fn legacy_header_ownership_includes_prefix_and_alignment() {
    let path = test_util::temp::temp_path("legacy-local-ownership.h5");
    let mut fixture = Fixture::new();
    let mut records = Form::CompactDataset.messages(
        &mut fixture,
        &type_builders::make_i32_type(),
        &17_i32.to_le_bytes(),
    );
    for record in &mut records {
        record
            .data
            .resize(record.data.len().next_multiple_of(v1::MESSAGE_ALIGNMENT), 0);
    }
    let header = v1::Header::new().messages(records).build();
    let target = u64::try_from(fixture.image.append(&header)).unwrap();
    fixture.link("data", target);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let origin = phase
        .reconstruction_source(&engine, ObjectId::new(PhysicalAddress::new(target)))
        .unwrap();
    let observed = phase.object_ownership(&origin).unwrap();
    assert!(observed.is_complete(), "{:?}", observed.error());
    assert_eq!(observed.allocations().len(), 1);
    assert_eq!(observed.allocations()[0].range().start().get(), target);
    assert_eq!(
        observed.allocations()[0].range().length(),
        u64::try_from(header.len()).unwrap()
    );
}

#[test]
fn failed_header_geometry_is_cached_without_an_allocation_proof() {
    let path = test_util::temp::temp_path("failed-local-ownership.h5");
    let mut fixture = Fixture::new();
    let target = fixture.append_header(vec![]);
    fixture
        .image
        .place(usize::try_from(target).unwrap(), &[0xee]);
    fixture.image.append(&[0; 64]);
    fixture.link("data", target);
    fixture.write(&path);
    let (engine, calls) = edit_test_util::open_counted_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let origin = phase
        .reconstruction_source(&engine, ObjectId::new(PhysicalAddress::new(target)))
        .unwrap();
    OWNERSHIP_READS.with(|reads| reads.set(0));
    let observed = phase.object_ownership(&origin).unwrap();
    assert!(!observed.is_complete());
    assert!(observed.allocations().is_empty());
    assert!(
        matches!(
            observed.error().unwrap(),
            Error::Format(FormatError::InvalidObjectHeaderVersion(0xee))
        ),
        "{:?}",
        observed.error()
    );
    let reads = calls.load(Ordering::Relaxed);
    phase.object_ownership(&origin).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), reads);
    assert_eq!(OWNERSHIP_READS.with(Cell::get), 1);
}

#[rstest]
#[case::undefined_target(global_heap_fixture::hard_link("undefined", u64::MAX))]
#[case::malformed_link(Message::new(MessageType::LINK, &[0xff]))]
fn incomplete_topology_does_not_discard_complete_local_header_ownership(#[case] link: Message) {
    let path = test_util::temp::temp_path("ownership-incomplete-topology.h5");
    let mut fixture = Fixture::new();
    let target = fixture.append_header(vec![link]);
    fixture.link("owner", target);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let origin = phase
        .reconstruction_source(&engine, ObjectId::new(PhysicalAddress::new(target)))
        .unwrap();
    assert!(phase.object_ownership(&origin).unwrap().is_complete());
    assert!(phase.topology().is_none());
    let ownership = phase.object_ownership(&origin).unwrap();
    assert!(ownership.is_complete());
    assert_eq!(ownership.allocations().len(), 1);
    assert_eq!(ownership.allocations()[0].kind(), AllocationKind::Header);
}

#[derive(Clone, Copy)]
enum Completeness {
    BadLayout,
    Reference,
    Unknown,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn ownership_fact_queries_bound_source_reads_and_allocations() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;

    use test_util::allocation;

    use crate::FileLocking;
    use crate::edit::WriteEngine;
    use crate::image::CountingImage;
    use crate::image::HandleImage;
    use crate::source::MetadataCacheConfig;

    let path = test_util::temp::temp_path("ownership-metadata-cost.h5");
    let payload = vec![0x5a; 1_048_576];
    let mut fixture = Fixture::new();
    let records =
        Form::ContiguousDataset.messages(&mut fixture, &type_builders::make_i32_type(), &payload);
    let target = fixture.append_header(records);
    fixture.link("data", target);
    fixture.write(&path);
    let calls = Arc::new(AtomicU64::new(0));
    let bytes = Arc::new(AtomicU64::new(0));
    let engine = WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, length| {
        Ok(Box::new(
            CountingImage::new(
                Box::new(HandleImage::new(
                    handle,
                    length,
                    MetadataCacheConfig::disabled(),
                )),
                Arc::clone(&bytes),
                Arc::default(),
            )
            .with_read_calls(Arc::clone(&calls)),
        ))
    })
    .unwrap();
    calls.store(0, Ordering::Relaxed);
    bytes.store(0, Ordering::Relaxed);
    let (mut phase, empty) =
        allocation::measure("ownership_empty_phase", || AnalysisPhase::new(&engine));
    assert_eq!((empty.blocks, empty.bytes), (0, 0));
    assert_eq!(
        (calls.load(Ordering::Relaxed), bytes.load(Ordering::Relaxed)),
        (0, 0)
    );
    let origin = phase
        .reconstruction_source(&engine, ObjectId::new(PhysicalAddress::new(target)))
        .unwrap();
    let (count, first) = allocation::measure("ownership_first_fact", || {
        let ownership = phase.object_ownership(&origin).unwrap();
        assert!(ownership.is_complete());
        ownership.allocations().len()
    });
    assert_eq!(count, 2);
    assert!(
        first.blocks > 0 && first.bytes > 0 && first.bytes < 16_384,
        "{first}"
    );
    let source_reads = (calls.load(Ordering::Relaxed), bytes.load(Ordering::Relaxed));
    assert!(
        source_reads.0 > 0 && source_reads.1 > 0 && source_reads.1 < 16_384,
        "{source_reads:?}"
    );
    let (count, repeated) = allocation::measure("ownership_cached_fact", || {
        phase.object_ownership(&origin).unwrap().allocations().len()
    });
    assert_eq!(count, 2);
    assert_eq!((repeated.blocks, repeated.bytes), (0, 0));
    assert_eq!(
        (calls.load(Ordering::Relaxed), bytes.load(Ordering::Relaxed)),
        source_reads
    );
    let (copy, positive) =
        allocation::measure("ownership_payload_copy_control", || payload.clone());
    assert_eq!(copy, payload);
    assert!(
        positive.blocks > 0 && positive.bytes >= u64::try_from(payload.len()).unwrap(),
        "{positive}"
    );
}

#[rstest]
#[case::complete(false)]
#[case::unavailable_heap(true)]
fn failed_heap_queries_preserve_independent_dense_index_allocations(#[case] damaged: bool) {
    let path = test_util::temp::temp_path("local-dense-ownership.h5");
    let image = if damaged {
        dense_attributes::dense_attrs_with_unavailable_heap_manager()
    } else {
        dense_attributes::dense_attrs()
    };
    let index = bytes::sole_signature(&image, btree_v2::SIGNATURE);
    let leaf = bytes::sole_signature(&image, btree_v2::LEAF_SIGNATURE);
    let headers = bytes::signature_offsets(&image, v2::SIGNATURE);
    let heap = bytes::sole_signature(&image, test_util::fractal_heap::SIGNATURE);
    let block = bytes::sole_signature(&image, test_util::fractal_heap::DIRECT_BLOCK_SIGNATURE);
    assert_eq!(headers.len(), 2);
    let raw = headers[0] - 24;
    fs::write(&path, &image).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let object = phase
        .topology()
        .unwrap()
        .find_link(&crate::object_path::ObjectPathBuf::parse("data"))
        .unwrap()
        .1;
    let origin = phase.reconstruction_source(&engine, object).unwrap();
    let ownership = phase.object_ownership(&origin).unwrap();
    assert_eq!(ownership.is_complete(), !damaged);
    let mut actual = ownership
        .allocations()
        .iter()
        .map(|allocation| {
            (
                allocation.range().start().get(),
                allocation.range().length(),
                allocation.kind(),
            )
        })
        .collect::<Vec<_>>();
    actual.sort_unstable_by_key(|&(address, _, _)| address);
    let mut expected = vec![
        (
            u64::try_from(headers[0]).unwrap(),
            u64::try_from(headers[1] - headers[0]).unwrap(),
            AllocationKind::Header,
        ),
        (
            u64::try_from(leaf).unwrap(),
            u64::try_from(index - leaf).unwrap(),
            AllocationKind::Metadata,
        ),
        (
            u64::try_from(index).unwrap(),
            u64::try_from(raw - index).unwrap(),
            AllocationKind::Metadata,
        ),
        (u64::try_from(raw).unwrap(), 24, AllocationKind::Data),
    ];
    if !damaged {
        expected.extend([
            (
                u64::try_from(heap).unwrap(),
                u64::try_from(block - heap).unwrap(),
                AllocationKind::Metadata,
            ),
            (
                u64::try_from(block).unwrap(),
                u64::try_from(leaf - block).unwrap(),
                AllocationKind::Metadata,
            ),
        ]);
    }
    expected.sort_unstable_by_key(|&(address, _, _)| address);
    assert_eq!(actual, expected);
    if damaged {
        let available = u64::try_from(image.len()).unwrap();
        let expected =
            available + hdf5_pure_format::__private::free_space_manager_header_len(engine.widths());
        let Some(Error::Format(FormatError::UnexpectedSourceEof {
            expected: actual_expected,
            available: actual_available,
        })) = ownership.error()
        else {
            panic!(
                "expected unavailable heap-manager bytes, got {:?}",
                ownership.error()
            );
        };
        assert_eq!((*actual_expected, *actual_available), (expected, available));
    } else {
        assert!(ownership.error().is_none());
    }
}

#[rstest]
#[case::outside_source(InvalidGeometry::Outside)]
#[case::overlapping_header(InvalidGeometry::Overlap)]
fn invalid_storage_geometry_keeps_only_independently_proven_header_ranges(
    #[case] geometry: InvalidGeometry,
) {
    let path = test_util::temp::temp_path("invalid-local-ownership.h5");
    let mut fixture = Fixture::new();
    let mut records =
        Form::ContiguousDataset.messages(&mut fixture, &type_builders::make_i32_type(), &[0; 8]);
    let target = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    let address = match geometry {
        InvalidGeometry::Outside => 1_000_000,
        InvalidGeometry::Overlap => target + 1,
    };
    records
        .iter_mut()
        .find(|record| record.msg_type == MessageType::DATA_LAYOUT)
        .unwrap()
        .data = data_layout::contiguous(address, 8);
    assert_eq!(fixture.append_header(records), target);
    let end = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    fixture.link("data", target);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let origin = phase
        .reconstruction_source(&engine, ObjectId::new(PhysicalAddress::new(target)))
        .unwrap();
    let ownership = phase.object_ownership(&origin).unwrap();
    assert!(!ownership.is_complete());
    assert_eq!(ownership.allocations().len(), 1);
    assert_eq!(ownership.allocations()[0].range().start().get(), target);
    assert_eq!(ownership.allocations()[0].range().length(), end - target);
    match (geometry, ownership.error().unwrap()) {
        (
            InvalidGeometry::Overlap,
            Error::InvalidSpaceOwnershipExtent {
                address: actual_address,
                length,
                previous_end,
                logical_size,
            },
        ) => {
            assert_eq!(
                (*actual_address, *length, *previous_end, *logical_size),
                (address, 8, end, end)
            );
        }
        (
            InvalidGeometry::Outside,
            Error::Format(FormatError::UnexpectedSourceEof {
                expected,
                available,
            }),
        ) => {
            assert_eq!((*expected, *available), (address + 8, end));
        }
        (_, error) => panic!("expected invalid allocation geometry, got {error:?}"),
    }
}

#[derive(Clone, Copy)]
enum InvalidGeometry {
    Outside,
    Overlap,
}
