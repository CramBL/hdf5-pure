use core::cell::Cell;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use core::mem;
use core::num::NonZeroU32;
use std::fs;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::sync::atomic::Ordering;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private as format;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::PageType;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::btree::v2 as btree_v2;
use test_util::bytes;
use test_util::datatype;
use test_util::dense_attributes;
use test_util::fractal_heap;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::object_header::v2;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;
use test_util_hdf5::userblock;

use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::DISCOVERY_WORK;
use crate::edit::group;
use crate::edit::test_util as edit_test_util;
use crate::error::FormatError;
use crate::file_writer;
use crate::fractal_heap_storage;
use crate::fractal_heap_storage::FractalHeapStorageError;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;
use crate::source::PhysicalSource;
use crate::type_builders;

#[rstest]
#[case::aliases(Scenario::Aliases)]
#[case::duplicate_edge(Scenario::Duplicate)]
#[case::outside_child(Scenario::Outside)]
#[case::excess_owner(Scenario::Excess)]
#[case::positive_cycle(Scenario::Cycle)]
#[case::broken_cycle(Scenario::BrokenCycle)]
#[case::root_reference(Scenario::Root)]
#[case::named_type(Scenario::NamedType)]
fn storage_uses_only_count_eligible_objects_and_keeps_retained_reference_roots(
    #[case] scenario: Scenario,
    #[values(0, 512)] base: usize,
) {
    let path = test_util::temp::temp_path("count-storage.h5");
    let mut fixture = if scenario == Scenario::Root {
        Fixture::new().with_root_reference_count(NonZeroU32::new(2).unwrap())
    } else {
        Fixture::new()
    };
    let raw = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    let mut records = Form::ContiguousDataset.messages(
        &mut fixture,
        &type_builders::make_i32_type(),
        &[17_i32.to_le_bytes(), 23_i32.to_le_bytes()].concat(),
    );
    records.push(Message::reference_count(if scenario == Scenario::Outside {
        3
    } else {
        2
    }));
    let (child, child_header) = append(&mut fixture, records);
    let owner_at = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    let records = if scenario == Scenario::NamedType {
        vec![
            Message::new(MessageType::DATATYPE, &datatype::f64_le()),
            Message::reference_count(1),
        ]
    } else {
        let mut records = vec![
            Message::new(
                MessageType::LINK_INFO,
                &file_writer::COMPACT_LINK_INFO
                    .serialize(file_writer::OFFSET_WIDTH)
                    .unwrap(),
            ),
            Message::new(MessageType::GROUP_INFO, &group::GROUP_INFO_BODY),
            global_heap_fixture::hard_link("left", child),
            global_heap_fixture::hard_link("right", child),
            Message::reference_count(
                if matches!(
                    scenario,
                    Scenario::Excess | Scenario::Cycle | Scenario::BrokenCycle
                ) {
                    2
                } else {
                    1
                },
            ),
        ];
        if matches!(
            scenario,
            Scenario::Cycle | Scenario::BrokenCycle | Scenario::Root
        ) {
            records.push(global_heap_fixture::hard_link(
                "back",
                if scenario == Scenario::Root {
                    Fixture::ROOT
                } else {
                    owner_at
                },
            ));
        }
        records
    };
    let (owner, owner_header) = append(&mut fixture, records);
    assert_eq!(owner, owner_at);
    fixture.link("owner", owner);
    if scenario == Scenario::Outside {
        fixture.link("keep", child);
    }
    fixture.write(&path);
    if base != 0 {
        userblock::prepend(&path, &vec![0xA5; base]);
    }
    let original = fs::read(&path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let before = engine.space.reusable_sections().collect::<Vec<_>>();
    let mut phase = AnalysisPhase::new(&engine);
    let added = phase.added_copy_users([]).unwrap().unwrap();
    let mut paths = vec!["owner"];
    match scenario {
        Scenario::Duplicate => paths.extend(["owner/left", "owner/left"]),
        Scenario::BrokenCycle => paths.push("owner/back"),
        _ => {}
    }
    let deletions = paths
        .into_iter()
        .map(|path| {
            phase
                .topology()
                .unwrap()
                .deletion(&ObjectPathBuf::parse(path))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let plan = phase.net_count_plan(&deletions, &added).unwrap().unwrap();
    let work = DISCOVERY_WORK.with(Cell::get);
    let storage = plan.collect_storage();
    assert_eq!(DISCOVERY_WORK.with(Cell::get), work);
    assert!(storage.same_session(&engine));
    assert!(storage.shares_phase(plan.source().unwrap()));
    assert_eq!(
        storage.root(),
        Some(id(Fixture::ROOT + u64::try_from(base).unwrap()))
    );
    let offset = u64::try_from(base).unwrap();
    let expected = if matches!(
        scenario,
        Scenario::Excess | Scenario::Cycle | Scenario::NamedType
    ) {
        vec![]
    } else {
        let mut expected = vec![(shift(owner_header, offset), FreeClass::Page(PageType::Meta))];
        if scenario != Scenario::Outside {
            expected.extend([
                (shift(child_header, offset), FreeClass::Page(PageType::Meta)),
                (range(raw + offset, 8), FreeClass::Page(PageType::Raw)),
            ]);
        }
        expected
    };
    assert_eq!(storage.allocations(), expected);
    let roots = match scenario {
        Scenario::Excess | Scenario::Cycle | Scenario::NamedType => vec![id(owner + offset)],
        Scenario::Outside => vec![id(child + offset)],
        Scenario::Root => vec![id(Fixture::ROOT + offset)],
        _ => vec![],
    };
    assert_eq!(storage.reference_roots(), roots);
    drop(plan);
    drop(phase);
    assert_eq!(storage.allocations(), expected);
    assert_eq!(storage.reference_roots(), roots);
    let mut other_phase = AnalysisPhase::new(&engine);
    let other_source = other_phase
        .reconstruction_source(&engine, storage.root().unwrap())
        .unwrap();
    assert!(!storage.shares_phase(&other_source));
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), before);
    drop(other_phase);
    drop(engine);
    assert_eq!(fs::read(&path).unwrap(), original);
    let reopened = edit_test_util::open_engine(&path);
    assert!(!storage.same_session(&reopened));
}

#[rstest]
#[case::ambiguous(true)]
#[case::unknown(false)]
fn unknown_original_counts_keep_storage_and_reference_roots(#[case] ambiguous: bool) {
    let path = test_util::temp::temp_path("unknown-count-storage.h5");
    let mut fixture = Fixture::new();
    let mut records = vec![Message::new(
        MessageType::OBJECT_REFERENCE_COUNT,
        &[1, 1, 0, 0, 0],
    )];
    if ambiguous {
        records = vec![Message::reference_count(1), Message::reference_count(1)];
    }
    let (owner, _) = append(&mut fixture, records);
    fixture.link("owner", owner);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("owner"))
        .unwrap();
    let plan = phase.source_count_plan([&deletion]).unwrap().unwrap();
    let mut storage = Vec::new();
    plan.collect_storage(&mut storage);
    assert!(storage.is_empty());
    assert_eq!(plan.retained_objects().collect::<Vec<_>>(), [id(owner)]);
}

#[rstest]
#[case::complete(false)]
#[case::unproven_heap(true)]
fn storage_preserves_dense_indexes_when_heap_ownership_is_unproven(#[case] damaged: bool) {
    let path = test_util::temp::temp_path("count-storage-dense.h5");
    let bytes = if damaged {
        dense_attributes::dense_attrs_with_unavailable_heap_manager()
    } else {
        dense_attributes::dense_attrs()
    };
    let heap = bytes::sole_signature(&bytes, fractal_heap::SIGNATURE);
    let block = bytes::sole_signature(&bytes, fractal_heap::DIRECT_BLOCK_SIGNATURE);
    let leaf = bytes::sole_signature(&bytes, btree_v2::LEAF_SIGNATURE);
    let index = bytes::sole_signature(&bytes, btree_v2::SIGNATURE);
    let headers = bytes::signature_offsets(&bytes, v2::SIGNATURE);
    assert_eq!(headers.len(), 2);
    let raw = headers[0] - 24;
    let mut expected = vec![
        (
            range(
                u64::try_from(headers[0]).unwrap(),
                u64::try_from(headers[1] - headers[0]).unwrap(),
            ),
            FreeClass::Page(PageType::Meta),
        ),
        (
            range(
                u64::try_from(leaf).unwrap(),
                u64::try_from(index - leaf).unwrap(),
            ),
            FreeClass::Page(PageType::Meta),
        ),
        (
            range(
                u64::try_from(index).unwrap(),
                u64::try_from(raw - index).unwrap(),
            ),
            FreeClass::Page(PageType::Meta),
        ),
    ];
    if !damaged {
        expected.extend([
            (
                range(
                    u64::try_from(heap).unwrap(),
                    u64::try_from(block - heap).unwrap(),
                ),
                FreeClass::Page(PageType::Meta),
            ),
            (
                range(
                    u64::try_from(block).unwrap(),
                    u64::try_from(leaf - block).unwrap(),
                ),
                FreeClass::Page(PageType::Meta),
            ),
        ]);
    }
    expected.push((
        range(u64::try_from(raw).unwrap(), 24),
        FreeClass::Page(PageType::Raw),
    ));
    fs::write(&path, &bytes).unwrap();
    let engine = edit_test_util::open_engine(&path);
    if damaged {
        let source = engine.opened_source();
        let source = PhysicalSource::new(&source);
        let error = fractal_heap_storage::collect_fractal_heap_storage_from_source(
            &source.stored(),
            StoredAddress::new(u64::try_from(heap).unwrap()),
            engine.widths(),
        )
        .unwrap_err();
        let available = u64::try_from(bytes.len()).unwrap();
        assert_eq!(
            error,
            FractalHeapStorageError::Format(FormatError::UnexpectedSourceEof {
                expected: available + format::free_space_manager_header_len(engine.widths()),
                available,
            })
        );
    }
    let mut phase = AnalysisPhase::new(&engine);
    let added = phase.added_copy_users([]).unwrap().unwrap();
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("data"))
        .unwrap();
    let plan = phase.net_count_plan([&deletion], &added).unwrap().unwrap();
    let storage = plan.collect_storage();
    assert_eq!(storage.allocations(), expected);
    assert!(storage.reference_roots().is_empty());
    assert_eq!(plan.removals().unwrap().counted_zero_objects().count(), 1);
    assert!(engine.space.reusable_sections().next().is_none());
    drop(plan);
    drop(phase);
    drop(engine);
    assert_eq!(fs::read(&path).unwrap(), bytes);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::empty(0)]
#[case::one(1)]
#[case::wide(64)]
fn retained_storage_uses_bounded_metadata_without_source_reads(#[case] count: usize) {
    let path = test_util::temp::temp_path("count-storage-retained.h5");
    let mut fixture = Fixture::new();
    let mut links = Vec::new();
    let targets = (0..count)
        .map(|index| {
            let (object, _) = append(&mut fixture, vec![Message::reference_count(2)]);
            links.extend([
                global_heap_fixture::hard_link(&format!("a{index}"), object),
                global_heap_fixture::hard_link(&format!("b{index}"), object),
            ]);
            id(object)
        })
        .collect::<Vec<_>>();
    let (owner, _) = append(&mut fixture, links);
    fixture.link("owner", owner);
    fixture.write(&path);
    let (engine, calls) = super::open_counted_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let added = phase.added_copy_users([]).unwrap().unwrap();
    let deletions = (0..count)
        .map(|index| {
            phase
                .topology()
                .unwrap()
                .deletion(&ObjectPathBuf::parse(&format!("owner/a{index}")))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let plan = phase.net_count_plan(&deletions, &added).unwrap().unwrap();
    let work = DISCOVERY_WORK.with(Cell::get);
    calls.store(0, Ordering::Relaxed);
    let (storage, measured) =
        allocation::measure(&format!("count-storage-retained-{count}"), || {
            plan.collect_storage()
        });
    assert!(storage.allocations().is_empty());
    assert_eq!(storage.reference_roots(), targets);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), work);
    assert!(storage.same_session(&engine));
    if count == 0 {
        assert_eq!(
            (measured.blocks, measured.bytes, measured.live_bytes),
            (0, 0, 0)
        );
        assert_eq!(storage.root(), None);
        assert!(phase.topology.is_none());
        let origin = phase
            .reconstruction_source(&engine, id(Fixture::ROOT))
            .unwrap();
        assert!(!storage.shares_phase(&origin));
    } else {
        let floor = u64::try_from(count * mem::size_of::<ObjectId>()).unwrap();
        assert!(measured.blocks > 0 && measured.bytes >= floor, "{measured}");
        assert!(
            measured.bytes <= 8 * floor && measured.live_bytes <= 4 * floor,
            "{measured}"
        );
        assert_eq!(storage.root(), Some(id(Fixture::ROOT)));
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Scenario {
    Aliases,
    BrokenCycle,
    Cycle,
    Duplicate,
    Excess,
    NamedType,
    Outside,
    Root,
}

fn append(fixture: &mut Fixture, messages: Vec<Message>) -> (u64, FileRange) {
    let address = u64::try_from(
        fixture.image.append(
            &v2::Header::new()
                .flags(v2::HeaderFlags::CHUNK_SIZE_WIDTH)
                .messages(messages)
                .build(),
        ),
    )
    .unwrap();
    (
        address,
        range(
            address,
            u64::try_from(fixture.image.as_bytes().len()).unwrap() - address,
        ),
    )
}

fn id(address: u64) -> ObjectId {
    ObjectId::new(PhysicalAddress::new(address))
}

fn range(address: u64, length: u64) -> FileRange {
    FileRange::new(PhysicalAddress::new(address), length).unwrap()
}

fn shift(original: FileRange, offset: u64) -> FileRange {
    range(original.start().get() + offset, original.length())
}
