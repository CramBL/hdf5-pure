use core::num::NonZeroU16;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_format::__private::GlobalHeapCollectionLocator;
use hdf5_pure_space::__private::SessionSpace;
use rstest::rstest;
use test_util::bytes;
use test_util::global_heap;
use test_util::snapshot::text::Report;
use test_util::snapshot::text::Value;
use test_util::widths::Widths;
use test_util_hdf5::__private::heap_members;
use test_util_hdf5::file;
use test_util_hdf5::file_space_oracle;
use test_util_hdf5::global_heap::attributes as native_attributes;

use crate::AttrValue;
use crate::DatasetBuilder;
use crate::File;
use crate::FileLocking;
use crate::FileSpaceStrategy;
use crate::FreeSections;
use crate::Layout;
use crate::LengthWidth;
use crate::MemoryStrategy;
use crate::PersistedFreeSpacePolicy;
use crate::PhysicalAddress;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::edit::WriteEngine;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::heap::associations::MemberPresence;
use crate::edit::analysis::physical::claims::ClaimOrigin;
use crate::edit::analysis::physical::claims::coverage::content::ResidualOutcome;
use crate::edit::test_util as edit_test_util;
use crate::global_heap::GlobalHeapId;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;
use crate::source::MetadataCacheConfig;

#[test]
fn native_replacement_removes_a_manually_aliased_heap_member() {
    let path = test_util::temp::temp_path("native-heap-member-lifetime.h5");
    let fixture = heap_members::MemberLifetime::write(&path);
    let (collection, original_index) = token_id(fixture.original());
    let (retained_collection, retained_index) = token_id(fixture.retained());
    assert_eq!(retained_collection, collection);
    assert_ne!(retained_index, original_index);
    let mut image = std::fs::read(&path).unwrap();
    let mut original_members = collection_members(&image, collection);
    assert_eq!(original_members.len(), 3);
    let original_payload = original_members[&original_index].1.clone();
    assert_eq!(original_payload, original_members[&retained_index].1);
    let descriptor = global_heap::reference(
        u32::try_from(original_payload.len()).unwrap(),
        collection.get(),
        u32::from(original_index.get()),
        Widths::EIGHT,
    );
    let sequence_at = usize::try_from(fixture.sequence_address()).unwrap();
    assert_eq!(
        bytes::slice_at(&image, sequence_at, descriptor.len()),
        vec![0; descriptor.len()]
    );
    bytes::set_slice_at(&mut image, sequence_at, &descriptor);
    std::fs::write(&path, &image).unwrap();
    fixture.assert_aliased_values(&path, &original_payload);
    assert_eq!(std::fs::read(&path).unwrap(), image);
    let retained = fixture.replace_with_empty(&path);
    let replaced = std::fs::read(&path).unwrap();
    assert_eq!(
        bytes::slice_at(&replaced, sequence_at, descriptor.len()),
        vec![0; descriptor.len()]
    );
    assert_eq!(
        original_members.remove(&original_index).unwrap().1,
        original_payload
    );
    assert_eq!(collection_members(&replaced, collection), original_members);
    heap_members::assert_lifetime_replacement(&path, &retained);
    assert_eq!(std::fs::read(&path).unwrap(), replaced);
}

#[rstest]
#[case::ordinary(false)]
#[case::external_region_token(true)]
fn native_heap_members_are_distinct_from_root_reference_completeness(#[case] external_token: bool) {
    let path = test_util::temp::temp_path("native-heap-member-facts.h5");
    let token = heap_members::write(&path, external_token);
    let original = std::fs::read(&path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let root =
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap();
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase.reconstruction_source(&engine, root).unwrap();
    let live = phase.collections_reachable_from([root]).unwrap();
    assert!(live.analysis_complete);
    let references = phase.heap_reference_facts(&source).unwrap();
    assert!(references.is_complete());
    assert!(references.error().is_none());
    assert_eq!(references.objects().len(), 2);
    assert_eq!(references.collections(), &live.referenced_collections);
    assert_eq!(live.referenced_collections.len(), 1);
    let referenced = references.objects().clone();
    let collection = PhysicalAddress::new(*live.referenced_collections.first().unwrap());
    phase.collection_allocation(collection).unwrap();
    let contents = phase.heap_members(&source, collection).unwrap();
    assert_eq!(
        contents.range().length(),
        bytes::uint_at(
            &original,
            usize::try_from(collection.get()).unwrap() + global_heap::SIZE_OFFSET,
            Widths::EIGHT.length
        )
    );
    assert_eq!(contents.members().len(), if external_token { 3 } else { 2 });
    let allocated = contents
        .members()
        .iter()
        .map(|member| GlobalHeapId {
            collection,
            index: member.index(),
        })
        .collect::<BTreeSet<_>>();
    assert!(referenced.is_subset(&allocated));
    assert!(
        contents
            .members()
            .iter()
            .all(|member| member.reference_count() == 0)
    );
    let missing = allocated
        .difference(&referenced)
        .copied()
        .collect::<Vec<_>>();
    if let Some(token) = &token {
        let offsets = usize::from(engine.widths().offsets.get());
        let address = engine
            .addresses()
            .resolve(StoredAddress::new(bytes::uint_at(
                token.bytes(),
                0,
                offsets,
            )))
            .unwrap();
        let index = u16::try_from(bytes::u32_at(token.bytes(), offsets)).unwrap();
        assert_eq!(
            missing,
            [GlobalHeapId {
                collection: address,
                index: NonZeroU16::new(index).unwrap()
            }]
        );
        assert_eq!(address, collection);
        let member = contents
            .members()
            .iter()
            .find(|member| member.index().get() == index)
            .unwrap();
        assert!(member.payload().range().length() > 0);
        assert!(member.range().length() > member.payload().range().length());
    } else {
        assert_eq!(missing, []);
    }
    let expected_length = contents.range().length();
    let claims = phase.physical_claims(&source).unwrap();
    assert_eq!(claims.root(), Some(root));
    assert_eq!(
        claims
            .claims()
            .iter()
            .filter(|claim| claim.origin() == ClaimOrigin::Heap(collection))
            .map(|claim| (claim.range().start(), claim.range().length()))
            .collect::<Vec<_>>(),
        [(collection, expected_length)]
    );
    let coverage = phase.physical_coverage(&source).unwrap();
    assert_eq!(coverage.conflict(), None);
    assert!(coverage.beyond_source().is_empty());
    assert!(
        coverage
            .covered()
            .iter()
            .any(|range| range.start() <= collection
                && range.end().get() >= collection.get() + expected_length)
    );
    let associations = phase.member_associations(&source).unwrap();
    assert!(associations.topology_available());
    assert!(associations.incomplete_references().is_empty());
    let association = &associations.collections()[&collection];
    assert!(association.error().is_none());
    assert_eq!(
        association.unmatched(),
        missing.iter().map(|id| id.index).collect::<Vec<_>>()
    );
    assert_eq!(association.observed().len(), 2);
    for observed in association.observed().values() {
        assert_eq!(observed.presence(), MemberPresence::Allocated);
        assert_eq!(observed.referrers(), [root]);
    }
    drop(phase);
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), original);
    heap_members::assert_values(&path, token.as_ref());
}

#[rstest]
#[case::removed_bounded(None, MemoryStrategy::Bounded)]
#[case::replaced_mirrored(Some("after"), MemoryStrategy::Mirrored)]
fn distinct_heap_members_remain_reclaimable_after_partial_edit_and_reopen(
    #[case] replacement: Option<&str>,
    #[case] memory: MemoryStrategy,
) {
    let path = test_util::temp::temp_path("partial-heap-member-lifecycle.h5");
    heap_members::write_persisted(&path);
    assert_lifecycle_values(&path, &[("first", "left"), ("second", "right")], None);
    let original = std::fs::read(&path).unwrap();
    let ranges = global_heap::collection_ranges(&original, Widths::EIGHT);
    let [parent] = ranges.as_slice() else {
        panic!("expected one original collection: {ranges:?}");
    };
    let parent = FileRange::new(
        PhysicalAddress::new(parent.start),
        parent.end - parent.start,
    )
    .unwrap();
    assert!(parent.end().get() <= u64::try_from(original.len()).unwrap());
    let members = collection_members(&original, parent.start());
    assert_eq!(members.len(), 2);
    assert_eq!(
        members
            .values()
            .map(|(_, payload)| payload.clone())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([b"left".to_vec(), b"right".to_vec()])
    );
    let open = || {
        WriteEngine::open_rw_with_strategy(
            &path,
            MetadataCacheConfig::disabled(),
            FileLocking::Enabled,
            memory,
            PersistedFreeSpacePolicy::Tolerant,
        )
        .unwrap()
    };
    let mut engine = open();
    let persistence = engine.persist.as_ref().unwrap();
    assert_eq!(persistence.strategy, FileSpaceStrategy::FsmAggr);
    assert_eq!(persistence.threshold.get(), 1);
    assert!(matches!(&engine.space, SessionSpace::Flat(_)));
    let mut phase = AnalysisPhase::new(&engine);
    let root =
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap();
    let source = phase.reconstruction_source(&engine, root).unwrap();
    let references = phase.heap_reference_facts(&source).unwrap();
    assert!(references.is_complete());
    assert!(references.error().is_none());
    assert_eq!(
        references.objects(),
        &members
            .keys()
            .map(|&index| GlobalHeapId {
                collection: parent.start(),
                index
            })
            .collect()
    );
    assert_eq!(
        phase.heap_members(&source, parent.start()).unwrap().range(),
        parent
    );
    drop(phase);
    let root = ObjectPathBuf::root();
    match replacement {
        Some(value) => engine
            .set_group_attr(&root, "first", AttrValue::VarLenAsciiString(value.into()))
            .unwrap(),
        None => engine.remove_group_attr(&root, "first").unwrap(),
    }
    engine.commit().unwrap();
    observe_retained_parent(&engine, parent, "First edit");
    let mut expected = replacement
        .map(|value| vec![("first", value)])
        .unwrap_or_default();
    expected.push(("second", "right"));
    drop(engine);
    assert_lifecycle_values(&path, &expected, None);
    let mut engine = open();
    observe_retained_parent(&engine, parent, "Reopened");
    engine.remove_group_attr(&root, "second").unwrap();
    engine.commit().unwrap();
    let free = engine.reusable_free_sections().unwrap();
    let end = PhysicalAddress::new(engine.context.superblock().eof_address);
    drop(engine);
    expected.pop();
    assert_lifecycle_values(&path, &expected, None);
    let capacity = assert_retired_parent(&path, parent, &free, end);
    let payload = vec![0xa5; usize::try_from(capacity).unwrap()];
    let mut engine = open();
    let mut builder = DatasetBuilder::new("reuse");
    builder.with_u8_data(&payload);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("reuse"), builder)
        .unwrap();
    engine.commit().unwrap();
    let free_after_reuse = engine.reusable_free_sections().unwrap();
    drop(engine);
    assert_lifecycle_values(&path, &expected, Some(&payload));
    let file = File::open(&path).unwrap();
    let Layout::Contiguous {
        address: Some(address),
        size,
    } = file.dataset("reuse").unwrap().layout().unwrap()
    else {
        panic!("reused payload is not allocated contiguous storage");
    };
    assert_eq!(size, capacity);
    let reused = FileRange::new(PhysicalAddress::new(address), size).unwrap();
    assert!(
        reused.start() < parent.end() && parent.start() < reused.end(),
        "{reused:?} did not reuse {parent:?}"
    );
    drop(file);
    assert!(
        free_after_reuse
            .sections()
            .iter()
            .all(|section| section.end_address() <= reused.start()
                || reused.end() <= section.address())
    );
}

/// Checks exclusion from reusable space and reports phase observations for the parent.
fn observe_retained_parent(engine: &WriteEngine, parent: FileRange, stage: &str) {
    assert!(
        engine
            .reusable_free_sections()
            .unwrap()
            .sections()
            .iter()
            .all(|section| section.end_address() <= parent.start()
                || parent.end() <= section.address())
    );
    let mut phase = AnalysisPhase::new(engine);
    let root =
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap();
    let source = phase.reconstruction_source(engine, root).unwrap();
    let allocated = phase.heap_members(&source, parent.start()).unwrap();
    assert_eq!(allocated.range(), parent);
    let count = u64::try_from(allocated.members().len()).unwrap();
    let associations = phase.member_associations(&source).unwrap();
    assert!(associations.topology_available());
    assert!(associations.incomplete_references().is_empty());
    let association = &associations.collections()[&parent.start()];
    assert!(association.error().is_none());
    assert_eq!(association.observed().len(), 1);
    let mut report = Report::new();
    report.metrics(
        stage,
        "Count",
        [
            ("Allocated members", count),
            (
                "Observed IDs",
                u64::try_from(association.observed().len()).unwrap(),
            ),
            (
                "Unmatched IDs",
                u64::try_from(association.unmatched().len()).unwrap(),
            ),
            (
                "Claim problems",
                u64::try_from(phase.physical_claims(&source).unwrap().problems().len()).unwrap(),
            ),
            (
                "Conflicts",
                u64::from(
                    phase
                        .physical_coverage(&source)
                        .unwrap()
                        .conflict()
                        .is_some(),
                ),
            ),
        ],
    );
    let content = phase.residual_content(&source).unwrap();
    report.table(
        "Physical residuals",
        ["Start", "Length", "Outcome"],
        content.intervals().iter().map(|entry| {
            [
                Value::Number(entry.range().start().get()),
                Value::Number(entry.range().length()),
                Value::text(match entry.outcome() {
                    ResidualOutcome::AllZero => "all zero",
                    ResidualOutcome::NonZero { .. } => "nonzero",
                    ResidualOutcome::ReadFailed(_) => "read failed",
                }),
            ]
        }),
    );
    eprintln!("{}", report.finish());
}

/// Checks complete parent coverage and returns its largest reusable intersection.
///
/// Accounts separately for free bytes, manager and extension allocations, and truncation.
fn assert_retired_parent(
    path: &Path,
    parent: FileRange,
    free: &FreeSections<PhysicalAddress>,
    end: PhysicalAddress,
) -> u64 {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(u64::try_from(bytes.len()).unwrap(), end.get());
    let file = File::open(path).unwrap();
    assert_eq!(file.superblock().eof_address, end.get());
    drop(file);
    let metadata = heap_members::persistence_allocations(path);
    let mut pieces = free
        .sections()
        .iter()
        .map(|section| (section.address(), section.end_address(), "Reusable"))
        .chain(
            metadata
                .iter()
                .map(|range| (range.start(), range.end(), "Manager or extension")),
        )
        .filter_map(|(start, stop, kind)| {
            let start = start.max(parent.start());
            let stop = stop.min(parent.end());
            (start < stop).then_some((start, stop, kind))
        })
        .collect::<Vec<_>>();
    let capacity = free
        .sections()
        .iter()
        .filter_map(|section| {
            let start = section.address().max(parent.start());
            let stop = section.end_address().min(parent.end());
            (start < stop).then(|| stop.get() - start.get())
        })
        .max()
        .expect("fixture retains reusable capacity in the original parent");
    if end < parent.end() {
        pieces.push((end.max(parent.start()), parent.end(), "Truncated"));
    }
    pieces.sort_unstable();
    let mut cursor = parent.start();
    for &(start, stop, _) in &pieces {
        assert_eq!(
            start, cursor,
            "original parent {parent:?} has overlapping or unaccounted pieces: {pieces:?}"
        );
        cursor = stop;
    }
    assert_eq!(cursor, parent.end());
    let mut report = Report::new();
    report.metrics(
        "Final original-parent retirement",
        "Value",
        [
            ("Parent start", parent.start().get()),
            ("Parent length", parent.length()),
            ("Confirmed EOA", end.get()),
            ("Fitting reuse capacity", capacity),
        ],
    );
    report.table(
        "Original-parent disposition",
        ["Start", "Length", "Disposition"],
        pieces.iter().map(|&(start, stop, kind)| {
            [
                Value::Number(start.get()),
                Value::Number(stop.get() - start.get()),
                Value::text(kind),
            ]
        }),
    );
    report.table(
        "Complete persistence allocations",
        ["Start", "Length"],
        metadata.iter().map(|range| {
            [
                Value::Number(range.start().get()),
                Value::Number(range.length()),
            ]
        }),
    );
    eprintln!("{}", report.finish());
    capacity
}

/// Checks fixture values and persistent free-space metadata through both readers.
fn assert_lifecycle_values(path: &Path, expected: &[(&str, &str)], reused: Option<&[u8]>) {
    let pure = File::open(path).unwrap();
    let settings = pure.file_space_settings().unwrap().unwrap();
    assert_eq!(settings.strategy(), FileSpaceStrategy::FsmAggr);
    assert!(settings.persist());
    assert_eq!(settings.threshold(), 1);
    assert_eq!(
        pure.dataset("data").unwrap().read_i32().unwrap(),
        [7, 11, 13, 17]
    );
    assert_eq!(
        pure.root()
            .attrs()
            .unwrap()
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
        expected
            .iter()
            .map(|&(name, value)| (name.to_owned(), AttrValue::VarLenAsciiString(value.into())))
            .collect()
    );
    if let Some(payload) = reused {
        assert_eq!(pure.dataset("reuse").unwrap().read_u8().unwrap(), payload);
    }
    drop(pure);
    let native = file::libhdf5_open(path);
    assert_eq!(
        native.dataset("data").unwrap().read_raw::<i32>().unwrap(),
        [7, 11, 13, 17]
    );
    let mut names = native.attr_names().unwrap();
    names.sort();
    assert_eq!(
        names,
        expected.iter().map(|&(name, _)| name).collect::<Vec<_>>()
    );
    for &(name, value) in expected {
        assert_eq!(
            native_attributes::ascii_values(&native.attr(name).unwrap()),
            [value]
        );
    }
    if let Some(payload) = reused {
        assert_eq!(
            native.dataset("reuse").unwrap().read_raw::<u8>().unwrap(),
            payload
        );
    }
    native.close().unwrap();
    file_space_oracle::assert_persisted_sections(path);
}

/// Decodes the fixture's collection address and member index from a native token.
fn token_id(token: &heap_members::RegionToken) -> (PhysicalAddress, NonZeroU16) {
    (
        PhysicalAddress::new(bytes::uint_at(token.bytes(), 0, Widths::EIGHT.offset)),
        NonZeroU16::new(u16::try_from(bytes::u32_at(token.bytes(), Widths::EIGHT.offset)).unwrap())
            .unwrap(),
    )
}

/// Parses allocated member indices, stored counts, and payloads from the image.
fn collection_members(
    bytes: &[u8],
    address: PhysicalAddress,
) -> BTreeMap<NonZeroU16, (u16, Vec<u8>)> {
    let image = FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), u64::try_from(bytes.len()).unwrap()).unwrap(),
        bytes,
    )
    .unwrap();
    let locator = GlobalHeapCollectionLocator::new(address, LengthWidth::Eight);
    let collection = locator
        .parse(image.subregion(locator.parse_range().unwrap()).unwrap())
        .unwrap();
    let mut walk = collection.walk();
    let mut members = BTreeMap::new();
    while let Some(range) = walk.parse_range() {
        if let Some(member) = walk.parse(image.subregion(range).unwrap()).unwrap() {
            assert_eq!(
                members.insert(
                    member.index(),
                    (
                        member.reference_count(),
                        member.payload().parse(image).unwrap().to_vec()
                    )
                ),
                None
            );
        }
    }
    members
}
