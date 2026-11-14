//! Candidate, reference, and buffering observations for attribute edits.
//!
//! Image reads are counted as requests, including those served from a mirror. The owner
//! encoding is sampled separately from commit buffering and resident mirror storage.

use core::cell::Cell;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private as format;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::PageType;
use rstest::rstest;
use test_util::disk_log;
use test_util::disk_log::DiskOp;
use test_util::global_heap;
use test_util::global_heap::Object;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::object_header::v2::Header;
use test_util::object_header::v2::HeaderFlags;
use test_util::snapshot::text::Report;
use test_util::snapshot::text::Value;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use super::super::AnalysisPhase;
use super::super::COLLECTION_ALLOCATIONS;
use super::super::COLLECTION_WORK;
use super::super::CollectionCandidates;
use super::super::CollectionProvenance;
use crate::FileLocking;
use crate::MemoryStrategy;
use crate::edit::WriteEngine;
use crate::edit::analysis::CollectionContent;
use crate::edit::attributes;
use crate::edit::attributes::AttrOp;
use crate::edit::object_header;
use crate::image::CountingImage;
use crate::image::FileImage;
use crate::image::HandleImage;
use crate::image::WriteBuffering;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;
use crate::source::MetadataCacheConfig;

#[rstest]
#[case::bounded(MemoryStrategy::Bounded, "attribute_work_bounded")]
#[case::mirrored(MemoryStrategy::Mirrored, "attribute_work_mirrored")]
fn attribute_candidates_share_liveness_and_keep_resource_costs_separate(
    #[case] memory: MemoryStrategy,
    #[case] snapshot: &str,
    #[values(false, true)] references: bool,
) {
    let path = test_util::temp::temp_path("attribute-work.h5");
    let (dead, owner, root) = write_fixture(&path, references);
    let bytes = Arc::new(AtomicU64::new(0));
    let calls = Arc::new(AtomicU64::new(0));
    let peak = Arc::new(AtomicU64::new(0));
    let mut engine = WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, len| {
        let image: Box<dyn FileImage> = match memory {
            MemoryStrategy::Bounded => Box::new(HandleImage::new(
                handle,
                len,
                MetadataCacheConfig::disabled(),
            )),
            MemoryStrategy::Mirrored => Box::new(WriteEngine::read_mirror(handle)?),
            MemoryStrategy::Auto => panic!("resource observations require an explicit backing"),
        };
        Ok(Box::new(
            CountingImage::new(image, bytes.clone(), Arc::default())
                .with_read_calls(calls.clone())
                .with_read_peak(peak.clone()),
        ))
    })
    .unwrap();
    engine
        .set_write_buffering(WriteBuffering::Session {
            page_size: 4096,
            max_bytes: 16_384,
        })
        .unwrap();
    let region = WriteEngine::gather_oh_messages(
        engine.image(),
        PhysicalAddress::new(owner),
        engine.context.addresses(),
        engine.context.widths(),
    )
    .unwrap();
    let descriptor_bytes = region.len();
    let operations: Vec<_> = (0..ATTRIBUTES)
        .map(|index| AttrOp::Remove {
            name: format!("a{index}"),
        })
        .collect();
    peak.store(0, Ordering::Relaxed);
    calls.store(0, Ordering::Relaxed);
    bytes.store(0, Ordering::Relaxed);
    let edited = attributes::plan_attr_ops(
        &engine.opened_source(),
        &region,
        &operations.iter().collect::<Vec<_>>(),
    )
    .unwrap();
    let encoded = object_header::build_v2_object_header(&edited.region).unwrap();
    let encoded_bytes = u64::try_from(encoded.len()).unwrap();
    let encoded_capacity = u64::try_from(encoded.capacity()).unwrap();
    assert!(encoded_bytes > 0 && encoded_capacity >= encoded_bytes);
    drop(encoded);
    COLLECTION_ALLOCATIONS.with(|count| count.set(0));
    let mut predecessor = AnalysisPhase::new(&engine);
    let candidates = predecessor
        .attribute_collections(
            ObjectId::new(PhysicalAddress::new(owner)),
            edited.predecessors,
        )
        .unwrap();
    assert_eq!(candidates.collections.len(), 1);
    let candidate = &candidates.collections[&PhysicalAddress::new(dead)];
    assert_eq!(candidate.origins.len(), ATTRIBUTES);
    let range = candidate.range;
    assert_eq!(
        range.length(),
        u64::try_from(global_heap::MIN_COLLECTION_SIZE).unwrap()
    );
    assert_eq!(predecessor.work().allocation_results, 1);
    let allocation_parses = COLLECTION_ALLOCATIONS.with(Cell::get);
    assert_eq!(allocation_parses, 1);
    let capture_bytes = bytes.load(Ordering::Relaxed);
    let capture_calls = calls.load(Ordering::Relaxed);
    assert!(capture_bytes > 0 && capture_bytes < range.length());
    peak.store(0, Ordering::Relaxed);
    calls.store(0, Ordering::Relaxed);
    bytes.store(0, Ordering::Relaxed);
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    let mut phase = AnalysisPhase::prospective(&engine, PhysicalAddress::new(root));
    assert!(
        phase
            .unreferenced_collection_spans(CollectionCandidates::default())
            .is_empty()
    );
    assert_eq!(phase.work().liveness_results, 0);
    assert_eq!(bytes.load(Ordering::Relaxed), 0);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        phase.unreferenced_collection_spans(candidates),
        [(range, FreeClass::Page(PageType::Raw))]
    );
    let first_bytes = bytes.load(Ordering::Relaxed);
    let first_calls = calls.load(Ordering::Relaxed);
    let first_peak = peak.load(Ordering::Relaxed);
    let parsed = COLLECTION_WORK.with(Cell::get);
    assert_eq!(parsed, if references { (3, 3) } else { (2, 2) });
    let mut later = CollectionCandidates::default();
    later
        .insert(
            predecessor.collection_allocation(range.start()).unwrap(),
            [CollectionProvenance {
                object: ObjectId::new(PhysicalAddress::new(owner)),
                content: CollectionContent::Attribute("a0".into()),
            }],
        )
        .unwrap();
    assert_eq!(
        phase.unreferenced_collection_spans(later),
        [(range, FreeClass::Page(PageType::Raw))]
    );
    assert_eq!(bytes.load(Ordering::Relaxed), first_bytes);
    assert_eq!(calls.load(Ordering::Relaxed), first_calls);
    assert_eq!(COLLECTION_WORK.with(Cell::get), parsed);
    let work = phase.work();
    assert_eq!(work.liveness_results, 1);
    assert_eq!(work.candidate_lookups, 2);
    assert_eq!(work.reference_objects, if references { 3 } else { 2 });
    assert_eq!(work.collection_references, u64::from(references));
    let payload_bytes = if references {
        REFERENCE_ELEMENTS * global_heap_fixture::REFERENCE_SIZE
    } else {
        0
    };
    assert!(first_calls > 0);
    assert!(first_bytes > 0);
    assert!(first_bytes >= u64::try_from(payload_bytes).unwrap());
    assert!(first_bytes < u64::try_from(payload_bytes + 4096).unwrap());
    assert!(first_peak <= SCAN_READ_LIMIT);
    if references {
        assert_eq!(first_peak, SCAN_READ_LIMIT);
    }
    drop(phase);
    drop(predecessor);
    assert!(
        engine
            .reusable_ranges()
            .all(|free| !free.unwrap().overlaps(range))
    );
    for operation in &operations {
        let AttrOp::Remove { name } = operation else {
            panic!("expected a removal");
        };
        engine
            .remove_group_attr(&ObjectPathBuf::parse("owner"), name)
            .unwrap();
    }
    let before = engine.image.buffering_work();
    peak.store(0, Ordering::Relaxed);
    calls.store(0, Ordering::Relaxed);
    bytes.store(0, Ordering::Relaxed);
    disk_log::start();
    engine.commit().unwrap();
    let operations = disk_log::take();
    let after = engine.image.buffering_work();
    let writes: Vec<_> = operations
        .iter()
        .filter_map(|operation| match operation {
            DiskOp::Write { bytes, .. } => Some(bytes.len()),
            DiskOp::SetLen(_) => None,
        })
        .collect();
    let written: usize = writes.iter().sum();
    assert!(!writes.is_empty());
    assert_eq!(
        after.write_calls - before.write_calls,
        u64::try_from(writes.len()).unwrap()
    );
    assert_eq!(
        after.write_bytes - before.write_bytes,
        u64::try_from(written).unwrap()
    );
    assert_eq!(after.pending_bytes, 0);
    assert_eq!(
        after.mirror_bytes > 0,
        matches!(memory, MemoryStrategy::Mirrored)
    );
    assert!(engine.reusable_ranges().any(|free| {
        let free = free.unwrap();
        free.start() <= range.start() && range.end() <= free.end()
    }));
    let mut report = Report::new();
    report.table(
        "Configuration",
        ["Property", "Value"],
        [[Value::text("Backing"), Value::text(format!("{memory:?}"))]],
    );
    report.metrics(
        "Predecessor descriptors and candidates",
        "Value",
        [
            (
                "Touched original descriptors",
                u64::try_from(ATTRIBUTES).unwrap(),
            ),
            (
                "Encoded message-region input bytes",
                u64::try_from(descriptor_bytes).unwrap(),
            ),
            ("Prepared owner encoding bytes", encoded_bytes),
            ("Prepared owner encoding capacity", encoded_capacity),
            ("Unique collection allocations", 1),
            (
                "Collection allocation parses",
                u64::try_from(allocation_parses).unwrap(),
            ),
            ("Original allocation bytes", range.length()),
            (
                "Recorded attribute origins",
                u64::try_from(ATTRIBUTES).unwrap(),
            ),
            ("Capture source read attempts", capture_calls),
            ("Capture source requested bytes", capture_bytes),
        ],
    );
    report.metrics(
        "One prospective source phase",
        "Value",
        [
            ("Objects parsed for links", u64::try_from(parsed.0).unwrap()),
            (
                "Objects parsed for references",
                u64::try_from(parsed.1).unwrap(),
            ),
            ("Reference summary objects", work.reference_objects),
            ("Retained collection references", work.collection_references),
            ("Cached root liveness results", work.liveness_results),
            ("Candidate membership lookups", work.candidate_lookups),
            (
                "Reference payload bytes",
                u64::try_from(payload_bytes).unwrap(),
            ),
            ("Source read attempts", first_calls),
            ("Source requested bytes", first_bytes),
            ("Largest source read request", first_peak),
            ("Additional reads for later candidates", 0),
        ],
    );
    report.metrics(
        "Complete commit I/O and buffering",
        "Value",
        [
            ("Source read attempts", calls.load(Ordering::Relaxed)),
            ("Source requested bytes", bytes.load(Ordering::Relaxed)),
            ("Largest source read request", peak.load(Ordering::Relaxed)),
            ("Issued write calls", after.write_calls - before.write_calls),
            ("Issued write bytes", after.write_bytes - before.write_bytes),
            ("Pending byte runs", after.pending_runs),
            ("Pending byte capacity", after.pending_capacity),
            ("Flush buffer peak bytes", after.flush_peak_bytes),
            ("Flush buffer peak capacity", after.flush_peak_capacity),
            (
                "Clean-gap read calls",
                after.clean_gap_calls - before.clean_gap_calls,
            ),
            (
                "Clean-gap read bytes",
                after.clean_gap_bytes - before.clean_gap_bytes,
            ),
            (
                "Explicit buffering copies",
                after.copied_bytes - before.copied_bytes,
            ),
            ("Resident mirror bytes", after.mirror_bytes),
            ("Resident mirror capacity", after.mirror_capacity),
        ],
    );
    drop(engine);
    if references {
        insta::assert_snapshot!(snapshot, report.finish());
    }
}

/// Writes the fixture and returns physical addresses for its original collection, owner, and
/// prospective root.
///
/// # Panics
///
/// Panics if the fixture cannot be written.
fn write_fixture(path: &Path, references: bool) -> (u64, u64, u64) {
    let mut fixture = Fixture::new();
    let dead = fixture.collection(&[Object::new(1, b"before")]);
    let mut messages = vec![link_info()];
    messages.extend(
        (0..ATTRIBUTES).map(|index| super::attributes::attribute(&format!("a{index}"), dead)),
    );
    let owner = fixture.image.append(
        &Header::new()
            .flags(HeaderFlags::CHUNK_SIZE_WIDTH)
            .phase_change(u16::try_from(ATTRIBUTES).unwrap(), 6)
            .messages(messages)
            .build(),
    );
    let owner = u64::try_from(owner).unwrap();
    fixture.link("owner", owner);
    let numeric = Form::ContiguousDataset.messages(
        &mut fixture,
        &crate::make_i32_type(),
        &vec![0; NUMERIC_BYTES],
    );
    let numeric = fixture.append_header(numeric);
    fixture.link("numeric", numeric);
    let mut prospective = vec![
        link_info(),
        global_heap_fixture::hard_link("numeric", numeric),
    ];
    if references {
        let live = fixture.collection(&[Object::new(1, b"live")]);
        let values = global_heap_fixture::reference(4, live, 1).repeat(REFERENCE_ELEMENTS);
        let messages = Form::ContiguousDataset.messages(
            &mut fixture,
            &global_heap_fixture::vlen_string(),
            &values,
        );
        let address = fixture.append_header(messages);
        fixture.link("references", address);
        prospective.push(global_heap_fixture::hard_link("references", address));
    }
    let root = fixture.append_header(prospective);
    fixture.link("prospective", root);
    fixture.write(path);
    (dead, owner, root)
}

fn link_info() -> Message {
    Message::new(
        MessageType::LINK_INFO,
        &format::LinkInfoMessage {
            max_creation_order: None,
            fractal_heap_address: None,
            btree_name_index_address: None,
            btree_creation_order_address: None,
        }
        .serialize(format::OffsetWidth::Eight)
        .unwrap(),
    )
}

const ATTRIBUTES: usize = 16;
const NUMERIC_BYTES: usize = 131_072;
const REFERENCE_ELEMENTS: usize = 8192;

const SCAN_READ_LIMIT: u64 = 65_536;
