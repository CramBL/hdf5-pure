//! Allocation bounds for reads, writes, editing sessions, and repacks.
//!
//! Every bound here is a statement about how the work *scales* — a windowed read
//! allocates on the order of its window, a chunked read allocates a fixed number
//! of blocks per chunk — so it holds on every platform, where an exact count does
//! not. Those are pinned separately, on one platform, in
//! `tests/allocation_baseline.rs`.
//!
//! Scaling is not the same as unbreakable, and the margins differ: the byte and
//! peak bounds have several times' headroom and will not notice a `Vec` growth
//! step anywhere, while `BLOCKS_PER_CHUNK` sits about 3% above a measured
//! 3.00-per-chunk rate on purpose — one more allocation *inside the per-chunk
//! loop* is exactly what it exists to catch, and a growth step there would fail
//! it too. That is the trade it is meant to make.
//!
//! Measurements use [`allocation::measure`]. Read bounds include an output-size floor, and
//! statistics polls use a separate allocating control before asserting zero allocations.

// heapscope unwinds with the frame-pointer convention, which it implements for
// x86_64 and aarch64 only, and on any other target the profiler errors at
// startup.
#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]

use std::fs::File as StdFile;
use std::path::Path;

use hdf5_pure::AttrValue;
use hdf5_pure::ChunkCacheConfig;
use hdf5_pure::ChunkIndex;
use hdf5_pure::DeflateLevel;
use hdf5_pure::FileSpacePageSize;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::MessageType as ParsedMessageType;
use hdf5_pure::OffsetWidth;
use hdf5_pure::RepackOptions;
use hdf5_pure::StoredAddress;
use hdf5_pure::UserblockSize;
use hdf5_pure::{
    ChunkCacheStats, EditBacking, File, FileAccessProperties, FileBuilder, FileSpaceStrategy,
    MaxExtent, MemoryStrategy, SyncPolicy,
};
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::ChunkIndexLayout;
use hdf5_pure_format::__private::DataLayout;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::GroupLinkStorage;
use hdf5_pure_format::__private::LinkInfoMessage;
use hdf5_pure_format::__private::ObjectHeader;
use hdf5_pure_format::__private::OwnedLink;
use hdf5_pure_format::__private::OwnedLinkTarget;
use rstest::rstest;
use test_util::allocation;
use test_util::data_layout;
use test_util::dataspace;
use test_util::datatype;
use test_util::image::Image;
use test_util::link_message::HardLink;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::object_header::v2;
use test_util::object_header::v2::Header;
use test_util::superblock::v2::Superblock;
use test_util::temp::TempPath;
use test_util::widths::Widths;
use test_util_hdf5::file_space;
use test_util_hdf5::reader::ReadBackend;

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

#[test]
fn parsing_ten_thousand_hard_link_messages_has_bounded_allocations() {
    const LINK_COUNT: usize = 10_000;
    let (probe, positive) = allocation::measure("hard_link_messages_control", || {
        std::hint::black_box(vec![0u8; 1_024])
    });
    assert!(positive.blocks > 0, "{positive}");
    assert!(positive.bytes >= 1_024, "{positive}");
    assert!(positive.live_bytes >= 1_024, "{positive}");
    assert!(positive.peak_bytes >= 1_024, "{positive}");
    drop(probe);

    let mut encoded_links = Vec::with_capacity(LINK_COUNT);
    let mut expected = Vec::with_capacity(LINK_COUNT);
    for index in 0..LINK_COUNT {
        let name = format!("link_{index:05}");
        let address = (u64::try_from(index).unwrap() + 1) * 64;
        encoded_links.push(HardLink::new(&name, address).build(Widths::EIGHT));
        expected.push((name.into_bytes(), StoredAddress::new(address)));
    }

    let info_bytes = LinkInfoMessage {
        max_creation_order: None,
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: None,
    }
    .serialize(OffsetWidth::Eight);

    let (storage, measured) = allocation::measure("hard_link_messages_parse_and_assemble", || {
        let mut links = Vec::with_capacity(LINK_COUNT);
        for data in &encoded_links {
            links.push(OwnedLink::parse(data, OffsetWidth::Eight).unwrap());
        }
        GroupLinkStorage::parse(&info_bytes, links, OffsetWidth::Eight).unwrap()
    });

    assert_eq!(storage.links().len(), LINK_COUNT);
    for (link, (name, address)) in storage.links().iter().zip(&expected) {
        assert_eq!(link.name(), name);
        assert_eq!(
            link.target(),
            &OwnedLinkTarget::Hard {
                object_header_address: *address
            }
        );
    }

    let retained_floor = u64::try_from(
        LINK_COUNT * std::mem::size_of::<OwnedLink>()
            + expected.iter().map(|(name, _)| name.len()).sum::<usize>(),
    )
    .unwrap();
    assert!(measured.blocks > LINK_COUNT as u64, "{measured}");
    assert!(measured.blocks <= 2 * LINK_COUNT as u64, "{measured}");
    assert!(measured.live_bytes >= retained_floor, "{measured}");
    assert!(measured.live_bytes <= 76 * LINK_COUNT as u64, "{measured}");
    assert!(measured.peak_bytes >= measured.live_bytes, "{measured}");
    assert!(measured.peak_bytes <= 120 * LINK_COUNT as u64, "{measured}");
    assert!(measured.bytes >= measured.peak_bytes, "{measured}");
    assert!(measured.bytes <= 120 * LINK_COUNT as u64, "{measured}");
}

#[test]
fn repeated_partial_replacements_retain_bounded_session_memory() {
    let directory = tempfile::tempdir().unwrap();
    let short = measure_partial_replacements(
        &directory.path().join("short-retirement.h5"),
        64,
        "short_retirement_memory",
    );
    let long = measure_partial_replacements(
        &directory.path().join("long-retirement.h5"),
        1024,
        "long_retirement_memory",
    );
    assert!(short.blocks > 64, "{short}");
    assert!(long.blocks > short.blocks, "{long}");
    assert!(long.bytes >= 1024 * 4096 * 8, "{long}");
    assert!(
        long.live_bytes <= short.live_bytes + 4096,
        "retained append memory grew with replacements: {short}; {long}"
    );
    assert!(
        long.peak_bytes <= short.peak_bytes + 16_384,
        "peak append memory grew with replacements: {short}; {long}"
    );
}

fn measure_partial_replacements(
    path: &Path,
    count: usize,
    region: &'static str,
) -> allocation::Measured {
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::FsmAggr));
    builder
        .create_dataset("d")
        .with_i32_data(&[1])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[4096]);
    builder.write(path).unwrap();
    let file = File::open_rw_with_options(
        path,
        FileAccessProperties::new()
            .with_memory_strategy(MemoryStrategy::Bounded)
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    let mut dataset = file.dataset("d").unwrap();
    dataset.append(&[1_i32]).unwrap();
    let ((), measured) = allocation::measure(region, || {
        for _ in 0..count {
            dataset.append(&[1_i32]).unwrap();
        }
    });
    assert_eq!(dataset.read_i32().unwrap(), vec![1; count + 2]);
    drop(dataset);
    file.close().unwrap();
    measured
}

#[rstest]
#[case::bounded_flat(MemoryStrategy::Bounded, FileSpaceStrategy::FsmAggr, "bounded_flat")]
#[case::bounded_page(MemoryStrategy::Bounded, FileSpaceStrategy::Page, "bounded_page")]
#[case::mirrored_flat(MemoryStrategy::Mirrored, FileSpaceStrategy::FsmAggr, "mirrored_flat")]
#[case::mirrored_page(MemoryStrategy::Mirrored, FileSpaceStrategy::Page, "mirrored_page")]
fn persistent_append_memory_is_bounded_as_data_and_session_duration_grow(
    #[case] backing: MemoryStrategy,
    #[case] strategy: FileSpaceStrategy,
    #[case] region: &str,
    #[values(false, true)] active: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let PersistentAppendMemory {
        first: small_first,
        warm: small,
    } = measure_persistent_appends(
        &directory.path().join("small.h5"),
        backing,
        strategy,
        active,
        32,
        16,
        &format!("{region}_{active}_small"),
    );
    let PersistentAppendMemory {
        first: large_first,
        warm: large,
    } = measure_persistent_appends(
        &directory.path().join("large.h5"),
        backing,
        strategy,
        active,
        8192,
        512,
        &format!("{region}_{active}_large"),
    );
    let PersistentAppendMemory {
        first: deferred, ..
    } = measure_persistent_appends(
        &directory.path().join("deferred.h5"),
        backing,
        strategy,
        active,
        0,
        16,
        &format!("{region}_{active}_deferred"),
    );
    assert!(
        large_first.live_bytes <= small_first.live_bytes + 8192,
        "{small_first}; {large_first}"
    );
    assert!(
        large_first.peak_bytes <= small_first.peak_bytes + 32 * 1024,
        "{small_first}; {large_first}"
    );
    assert!(
        large_first.bytes <= small_first.bytes + 64 * 1024,
        "{small_first}; {large_first}"
    );
    assert!(
        deferred.live_bytes <= small_first.live_bytes + 8192,
        "{small_first}; {deferred}"
    );
    assert!(
        deferred.peak_bytes <= small_first.peak_bytes + 32 * 1024,
        "{small_first}; {deferred}"
    );
    let (control, positive) =
        allocation::measure(&format!("{region}_{active}_dataset_copy_control"), || {
            vec![7_i32; 8192 * 64]
        });
    assert!(positive.blocks > 0, "{positive}");
    assert!(positive.live_bytes >= 8192 * 64 * 4, "{positive}");
    assert!(
        positive.peak_bytes > small.peak_bytes + 32 * 1024,
        "{positive}; {small}"
    );
    assert!(
        positive.peak_bytes > small_first.peak_bytes + 32 * 1024,
        "{positive}; {small_first}"
    );
    assert!(
        positive.live_bytes > small_first.live_bytes + 8192,
        "{positive}; {small_first}"
    );
    drop(control);
    assert!(large.blocks > small.blocks * 8, "{small}; {large}");
    assert!(large.bytes >= 512 * 64 * 4, "{large}");
    assert!(large.live_bytes < 32 * 1024, "{large}");
    assert!(
        large.live_bytes <= small.live_bytes + 8192,
        "{small}; {large}"
    );
    assert!(
        large.peak_bytes <= small.peak_bytes + 32 * 1024,
        "{small}; {large}"
    );
    assert!(
        large.bytes / 512 <= small.bytes / 16 + 16 * 1024,
        "{small}; {large}"
    );
}

/// Allocations during the first append and a subsequent append loop.
struct PersistentAppendMemory {
    /// Includes dataset lookup, appender construction, and retained handles after the first append.
    first: allocation::Measured,
    /// Includes repeated fixed batches and unrelated attribute commits with an active appender.
    warm: allocation::Measured,
}

/// Measures first and repeated 256-byte appends into an interior free hole.
///
/// Excludes fixture creation, file opening, mirror residency, free-section queries, and value
/// checks from both regions. The persistence threshold is 4096 bytes, and the free-section count
/// is bounded before and after the loop. The separate copy control allocates the large dataset's
/// payload and exceeds the permitted peak and retained-memory growth.
fn measure_persistent_appends(
    path: &Path,
    backing: MemoryStrategy,
    strategy: FileSpaceStrategy,
    active: bool,
    chunks: usize,
    appends: usize,
    region: &str,
) -> PersistentAppendMemory {
    const CHUNK: usize = 64;
    let initial = chunks * CHUNK;
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(strategy)
            .with_persistence(true)
            .with_threshold(4096),
    );
    let target = builder.create_dataset("d");
    if initial == 0 {
        target.with_unallocated_storage(hdf5_pure::make_i32_type(), &[0]);
    } else {
        target.with_i32_data(&vec![7; initial]);
    }
    target
        .with_chunks(&[CHUNK as u64])
        .with_maxshape(&[MaxExtent::Unlimited]);
    builder
        .create_dataset("scratch")
        .with_u8_data(&vec![3; 2 * 1024 * 1024]);
    builder.create_dataset("ceiling").with_u8_data(&[9; 8192]);
    builder.write(path).unwrap();
    let options = FileAccessProperties::new()
        .with_memory_strategy(backing)
        .with_chunk_cache(ChunkCacheConfig::disabled())
        .with_sync_policy(SyncPolicy::OnClose);
    let file = File::open_rw_with_options(path, options).unwrap();
    file.root().delete("scratch").unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    if initial == 0 {
        assert_resource_index_unallocated(path);
    }
    let file = File::open_rw_with_options(path, options).unwrap();
    assert_eq!(
        file.edit_backing().unwrap(),
        match backing {
            MemoryStrategy::Bounded => EditBacking::Bounded,
            MemoryStrategy::Mirrored => EditBacking::Mirrored,
            _ => panic!("resource test requires an explicit backing"),
        }
    );
    let ((mut dataset, mut buffer), first) =
        allocation::measure(&format!("{region}_first"), || {
            let mut dataset = file.dataset("d").unwrap();
            let mut buffer =
                active.then(|| file.dataset("d").unwrap().into_buffered_appender().unwrap());
            if let Some(buffer) = &mut buffer {
                buffer.append(&[7; CHUNK]).unwrap();
                buffer.flush().unwrap();
            } else {
                dataset.append(&[7; CHUNK]).unwrap();
            }
            (dataset, buffer)
        });
    assert!(first.blocks > 0, "{first}");
    assert!(first.bytes >= (CHUNK * 4) as u64, "{first}");
    assert!(first.live_bytes > 0, "{first}");
    let initialized = file.dataset("d").unwrap();
    assert_eq!(initialized.chunks().unwrap().len(), chunks + 1);
    assert_eq!(initialized.read_i32().unwrap(), vec![7; initial + CHUNK]);
    drop(initialized);
    let sections = file.reusable_free_sections().unwrap().section_count();
    assert!(sections <= 16, "sections before appends: {sections}");
    let length = std::fs::metadata(path).unwrap().len();
    let ((), measured) = allocation::measure(region, || {
        for round in 0..appends {
            if let Some(buffer) = &mut buffer {
                buffer.append(&[7; CHUNK]).unwrap();
                buffer.flush().unwrap();
                if (round + 1) % 16 == 0 {
                    file.dataset("ceiling")
                        .unwrap()
                        .set_attr("cycle", AttrValue::I32(i32::try_from(round + 1).unwrap()))
                        .unwrap();
                    file.commit().unwrap();
                }
            } else {
                dataset.append(&[7; CHUNK]).unwrap();
            }
        }
    });
    assert!(measured.blocks >= appends as u64, "{measured}");
    assert!(measured.bytes >= (appends * CHUNK * 4) as u64, "{measured}");
    assert!(
        file.reusable_free_sections().unwrap().section_count() <= 16,
        "sections: {sections} -> {}",
        file.reusable_free_sections().unwrap().section_count()
    );
    assert!(std::fs::metadata(path).unwrap().len() <= length + 64 * 1024);
    assert_eq!(
        dataset.read_i32().unwrap(),
        vec![7; initial + (appends + 1) * CHUNK]
    );
    assert_eq!(
        file.dataset("ceiling").unwrap().read_u8().unwrap(),
        [9; 8192]
    );
    if let Some(buffer) = buffer {
        buffer.finish().unwrap();
        assert_eq!(
            file.dataset("ceiling").unwrap().attrs().unwrap()["cycle"],
            AttrValue::I32(i32::try_from(appends).unwrap())
        );
    }
    drop(dataset);
    file.close().unwrap();
    let file = File::open(path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        vec![7; initial + (appends + 1) * CHUNK]
    );
    PersistentAppendMemory {
        first,
        warm: measured,
    }
}

fn assert_resource_index_unallocated(path: &Path) {
    let file = File::open(path).unwrap();
    let dataset = file.dataset("d").unwrap();
    assert_eq!(dataset.shape().unwrap(), [0]);
    let widths = FormatWidths::from_sizes(
        file.superblock().offset_size.get(),
        file.superblock().length_size.get(),
    )
    .unwrap();
    drop(dataset);
    drop(file);
    let bytes = std::fs::read(path).unwrap();
    let mut candidates = test_util::bytes::signature_offsets(&bytes, v2::SIGNATURE)
        .into_iter()
        .filter_map(|header_at| {
            ObjectHeader::parse(
                &bytes,
                AccessMode::ReadOnly,
                header_at,
                widths.offsets.get(),
                widths.lengths.get(),
            )
            .unwrap()
            .messages
            .into_iter()
            .find(|message| {
                message.msg_type == ParsedMessageType::DATA_LAYOUT
                    && matches!(
                        DataLayout::parse_with_widths(&message.data, widths).unwrap(),
                        DataLayout::Chunked {
                            index: ChunkIndexLayout::ExtensibleArray { .. },
                            ..
                        }
                    )
            })
            .map(|message| (header_at, message.data))
        })
        .collect::<Vec<_>>();
    assert_eq!(candidates.len(), 1);
    let (_, layout) = candidates.pop().unwrap();
    let (original, pointer) = DataLayout::parse_with_ea_address(&layout, widths).unwrap();
    assert!(matches!(
        original,
        DataLayout::Chunked {
            index: ChunkIndexLayout::ExtensibleArray { address: None, .. },
            ..
        }
    ));
    assert_eq!(&layout[pointer.unwrap()], &[u8::MAX; 8]);
}

#[rstest]
#[case::flat(
    FileSpaceStrategy::FsmAggr,
    false,
    "flat_stats_poll",
    "flat_stats_control"
)]
#[case::page(
    FileSpaceStrategy::Page,
    false,
    "page_stats_poll",
    "page_stats_control"
)]
#[case::flat_reserve(
    FileSpaceStrategy::FsmAggr,
    true,
    "flat_reserve_poll",
    "flat_reserve_control"
)]
#[case::page_reserve(
    FileSpaceStrategy::Page,
    true,
    "page_reserve_poll",
    "page_reserve_control"
)]
#[case::aggregates(
    FileSpaceStrategy::Aggr,
    false,
    "aggregate_stats_poll",
    "aggregate_stats_control"
)]
fn reusable_space_stats_poll_without_allocating(
    #[case] strategy: FileSpaceStrategy,
    #[case] reserve: bool,
    #[case] region: &'static str,
    #[case] control: &'static str,
) {
    let directory = tempfile::tempdir().unwrap();
    let file =
        file_space::fragmented_session(&directory.path().join("fragmented.h5"), strategy, 32);
    if reserve {
        file.root()
            .create_dataset("log", |builder| {
                builder
                    .with_i32_data(&[])
                    .with_shape(&[0])
                    .with_chunks(&[64])
                    .with_maxshape(&[MaxExtent::Unlimited]);
            })
            .unwrap();
        file.commit().unwrap();
        file.dataset("log").unwrap().append(&[1; 64]).unwrap();
    } else if strategy == FileSpaceStrategy::Aggr {
        file.root()
            .create_dataset("spare", |builder| {
                builder.with_i32_data(&[3; 4]);
            })
            .unwrap();
        file.commit().unwrap();
    }
    let sections = file.reusable_free_sections().unwrap();
    assert!(
        sections.section_count()
            >= if strategy == FileSpaceStrategy::Aggr {
                1
            } else {
                16
            }
    );
    let expected = file.reusable_space_stats().unwrap();
    assert_eq!(expected.reusable_free_bytes(), sections.free_bytes());
    let (probe, positive) = allocation::measure(control, || std::hint::black_box(vec![0u8; 1_024]));
    assert!(positive.blocks > 0, "{positive}");
    assert!(positive.bytes >= 1_024, "{positive}");
    drop(probe);
    let (observed, measured) = allocation::measure(region, || {
        let mut observed = expected;
        for _ in 0..10_000 {
            observed = std::hint::black_box(&file).reusable_space_stats().unwrap();
        }
        observed
    });
    assert_eq!(observed, expected);
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(measured.bytes, 0, "{measured}");
}

#[rstest]
#[case::compact_whole(true, 0, u64::MAX, "compact_whole_baseline", "compact_whole_rows")]
#[case::compact_partial(true, 2, 2, "compact_partial_baseline", "compact_partial_rows")]
#[case::contiguous_whole(
    false,
    0,
    u64::MAX,
    "contiguous_whole_baseline",
    "contiguous_whole_rows"
)]
#[case::contiguous_partial(false, 2, 2, "contiguous_partial_baseline", "contiguous_partial_rows")]
fn row_windows_preserve_the_whole_reads_allocation_count(
    #[case] compact: bool,
    #[case] start: u64,
    #[case] count: u64,
    #[case] baseline_region: &'static str,
    #[case] rows_region: &'static str,
) {
    let values: Vec<_> = (0..16).map(f64::from).collect();
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let mut image = Image::starting_with(&[0; 64]);
    let layout = if compact {
        data_layout::compact(&bytes)
    } else {
        let address = image.append(&bytes);
        data_layout::contiguous(address as u64, bytes.len() as u64)
    };
    let dataset = image.append(
        &Header::new()
            .messages([
                Message::new(
                    MessageType::DATASPACE,
                    &dataspace::v2(
                        2,
                        dataspace::Flags::NONE,
                        dataspace::Kind::SIMPLE,
                        &[8, 2],
                        None,
                    ),
                ),
                Message::new(MessageType::DATATYPE, &datatype::f64_le()),
                Message::new(MessageType::DATA_LAYOUT, &layout),
            ])
            .build(),
    );
    let root = image.append(
        &Header::new()
            .message(Message::new(
                MessageType::LINK,
                &HardLink::new("data", dataset as u64).build(Widths::EIGHT),
            ))
            .build(),
    );
    image.place(
        0,
        &Superblock::new(Widths::EIGHT)
            .root_header_address(root as u64)
            .eof_address(image.as_bytes().len() as u64)
            .build(),
    );
    let file = File::from_bytes(image.build()).unwrap();
    let dataset = file.dataset("data").unwrap();
    assert_eq!(dataset.read_raw().unwrap(), bytes);
    let (whole, baseline) = allocation::measure(baseline_region, || dataset.read_raw().unwrap());
    let (window, measured) =
        allocation::measure(rows_region, || dataset.read_raw_rows(start, count).unwrap());
    let start = usize::try_from(start).unwrap() * 2 * 8;
    let end = if count == u64::MAX {
        whole.len()
    } else {
        start + usize::try_from(count).unwrap() * 2 * 8
    };
    assert_eq!(whole, bytes);
    assert_eq!(window, whole[start..end]);
    assert!(measured.live_bytes >= window.len() as u64, "{measured}");
    assert!(
        measured.blocks <= baseline.blocks,
        "rows: {measured}; whole: {baseline}"
    );
}

#[test]
fn full_inner_chunk_bands_preserve_the_whole_reads_allocation_count() {
    let values: Vec<_> = (0..2048).map(f64::from).collect();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bands.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&values)
        .with_shape(&[128, 16])
        .with_chunks(&[64, 16]);
    builder.write(&path).unwrap();
    let file = File::open_streaming(&path).unwrap();
    let dataset = file.dataset("data").unwrap();
    assert_eq!(dataset.read_f64().unwrap(), values);
    let (whole, baseline) =
        allocation::measure("chunk_bands_whole", || dataset.read_raw().unwrap());
    let (rows, measured) = allocation::measure("chunk_bands_rows", || {
        dataset.read_raw_rows(1, 126).unwrap()
    });
    assert_f64_bytes(&whole, values.iter().copied());
    assert_eq!(rows, whole[128..whole.len() - 128]);
    assert!(measured.live_bytes >= rows.len() as u64, "{measured}");
    assert!(
        measured.blocks <= baseline.blocks,
        "rows: {measured}; whole: {baseline}"
    );
}

#[test]
fn typed_contiguous_sweeps_allocate_the_output_and_windows() {
    const WINDOWS: u64 = 8;
    let values: Vec<_> = (0..1024 * 1024).map(f64::from).collect();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("sweep.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&values)
        .with_shape(&[1024, 1024]);
    builder.write(&path).unwrap();
    let file = File::open_streaming(&path).unwrap();
    let dataset = file.dataset("data").unwrap();
    assert_eq!(dataset.read_f64_rows(0, 1).unwrap(), values[..1024]);
    assert_eq!(dataset.chunk_cache_stats(), ChunkCacheStats::default());
    let (whole, baseline) =
        allocation::measure("contiguous_sweep_whole", || dataset.read_raw().unwrap());
    let (typed, measured) =
        allocation::measure("contiguous_sweep_typed", || dataset.read_f64().unwrap());
    assert_eq!(typed, values);
    assert_f64_bytes(&whole, values.iter().copied());
    assert!(measured.live_bytes >= whole.len() as u64, "{measured}");
    assert!(
        measured.blocks <= baseline.blocks + WINDOWS,
        "typed: {measured}; whole: {baseline}"
    );
}

#[test]
fn typed_byte_arrays_keep_only_one_read_window_beside_output() {
    const COUNT: usize = 1_048_576;
    let values = vec![[0, 127, 128, 255, 42, 19, 5]; COUNT];
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("byte_arrays.h5");
    let mut builder = FileBuilder::new();
    let ((), staged) = allocation::measure("typed_byte_array_staging", || {
        builder.create_dataset("values").with_data(&values);
    });
    let payload = (COUNT * 7) as u64;
    assert!(staged.live_bytes >= payload, "{staged}");
    assert!(staged.peak_bytes < payload + payload / 2, "{staged}");
    builder.write(&path).unwrap();
    let file = File::open_streaming(&path).unwrap();
    let dataset = file.dataset("values").unwrap();
    let (output, measured) = allocation::measure("typed_byte_array_read", || {
        dataset.read::<[u8; 7]>().unwrap()
    });
    assert_eq!(output, values);
    assert!(measured.live_bytes >= payload, "{measured}");
    assert!(measured.peak_bytes < payload + payload / 2, "{measured}");
    let ((raw, copied), control) = allocation::measure("typed_byte_array_copy_control", || {
        let raw = dataset.read_raw().unwrap();
        let copied = raw.clone();
        (raw, copied)
    });
    assert_eq!(raw, values.as_flattened());
    assert_eq!(copied, raw);
    assert!(control.peak_bytes >= 2 * payload, "{control}");
}

#[test]
fn typed_ascii_keeps_only_one_read_window_beside_output() {
    const COUNT: usize = 1_048_576;
    let value = hdf5_pure::FixedAscii::<7>::new("abcdefg").unwrap();
    let values = vec![value; COUNT];
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("ascii.h5");
    let mut builder = FileBuilder::new();
    let ((), staged) = allocation::measure("typed_ascii_staging", || {
        builder.create_dataset("values").with_data(&values);
    });
    let payload = (COUNT * 7) as u64;
    assert!(staged.live_bytes >= payload, "{staged}");
    assert!(staged.peak_bytes < payload + payload / 2, "{staged}");
    builder.write(&path).unwrap();
    let file = File::open_streaming(&path).unwrap();
    let dataset = file.dataset("values").unwrap();
    let (output, measured) = allocation::measure("typed_ascii_read", || {
        dataset.read::<hdf5_pure::FixedAscii<7>>().unwrap()
    });
    assert_eq!(output, values);
    assert!(measured.live_bytes >= payload, "{measured}");
    assert!(measured.peak_bytes < payload + payload / 2, "{measured}");
    let ((raw, copied), control) = allocation::measure("typed_ascii_copy_control", || {
        let raw = dataset.read_raw().unwrap();
        let copied = raw.clone();
        (raw, copied)
    });
    assert_eq!(raw, b"abcdefg".repeat(COUNT));
    assert_eq!(copied, raw);
    assert!(control.peak_bytes >= 2 * payload, "{control}");
}

#[test]
fn windowed_row_read_allocates_on_the_order_of_the_window() {
    // 4 MiB of f64 rows with inner-split storage chunks: [2048, 32, 8] in
    // [64, 16, 4] chunks — a 2x2 inner chunk grid per 64-row band, deflated so
    // chunks really decode (32 KiB decompressed each).
    const N0: usize = 2048;
    const ROW_ELEMS: usize = 32 * 8;
    const DATASET_BYTES: usize = N0 * ROW_ELEMS * 8;

    let data: Vec<f64> = (0..N0 * ROW_ELEMS).map(|i| i as f64).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("t")
        .with_f64_data(&data)
        .with_shape(&[N0 as u64, 32, 8])
        .with_chunks(&[64, 16, 4])
        .with_deflate(DeflateLevel::Three);
    builder.write(&path).unwrap();
    drop(data);

    // The streaming backend reads from disk on demand; the buffered backend
    // would hold the whole file in memory by design and drown the signal.
    let file = File::open_streaming(&path).unwrap();
    let ds = file.dataset("t").unwrap();

    // A 64-row window straddling a chunk-band boundary mid-file, so it decodes
    // chunks from two leading bands.
    let (window, measured) =
        allocation::measure("windowed_row_read", || ds.read_f64_rows(992, 64).unwrap());

    // Window + raw/typed conversion + a few decompressed 32 KiB chunks lands
    // well under 1 MiB; a whole-read fallback peaks above the full dataset
    // (the assembled whole read plus cached chunks and the sliced window).
    assert!(
        measured.peak_bytes < (DATASET_BYTES / 4) as u64,
        "peak allocation during the windowed read must be bounded by the window, \
         not the {DATASET_BYTES}-byte dataset; measured {measured}"
    );

    // The read's own output is in this measurement, so its size is a floor a
    // measurement of nothing cannot reach. Without it the ceiling above passes
    // on a read whose work moved to a thread this region does not see.
    const WINDOW_BYTES: usize = 64 * ROW_ELEMS * 8;
    assert!(
        measured.live_bytes >= WINDOW_BYTES as u64,
        "the window this read returned is not in its own measurement, so the \
         ceiling above is bounding something other than the read: {measured}"
    );

    // The window must still be the right bytes.
    let expected: Vec<f64> = (992 * ROW_ELEMS..(992 + 64) * ROW_ELEMS)
        .map(|i| i as f64)
        .collect();
    assert_eq!(window, expected);
}

#[test]
fn windowed_vlen_string_read_allocates_on_the_order_of_the_window() {
    // ~4 MiB of variable-length string payload: 32k rows of 128-byte strings,
    // plus ~512 KiB of heap references in the dataset itself. The writer packs
    // the strings into one giant heap collection — the degenerate case for a
    // windowed read, whose directory alone rivals the window if parsed whole.
    const N0: usize = 32 * 1024;
    const STR_LEN: usize = 128;
    const PAYLOAD_BYTES: usize = N0 * STR_LEN;

    let strings: Vec<String> = (0..N0)
        .map(|i| format!("{i:0>width$}", width = STR_LEN))
        .collect();
    let refs: Vec<&str> = strings.iter().map(String::as_str).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("labels").with_vlen_strings(&refs);
    builder.write(&path).unwrap();
    drop(refs);
    drop(strings);

    let file = File::open_streaming(&path).unwrap();
    let ds = file.dataset("labels").unwrap();

    // 256 mid-file rows — ~32 KiB of text plus ~4 KiB of references, resolved
    // against a window-filtered slice of the collection's directory.
    let (window, measured) = allocation::measure("windowed_vlen_read", || {
        ds.read_string_rows(15_000, 256).unwrap()
    });

    // Window references + text + touched heap-collection directories land far
    // under 1 MiB; resolving every reference first peaks above the full payload
    // (all references plus a Vec<String> of every row). This also catches a
    // collection parse that starts buffering whole collections instead of
    // walking their metadata.
    assert!(
        measured.peak_bytes < (PAYLOAD_BYTES / 4) as u64,
        "peak allocation during the windowed vlen read must be bounded by the window, \
         not the {PAYLOAD_BYTES}-byte payload; measured {measured}"
    );

    // The other half of the same story, which the peak cannot see. A global heap
    // collection chains each object's position in the previous object's header,
    // so reaching the window's objects means passing every object before them —
    // the traversal is the format's, not this crate's, and no bound here can
    // remove it. What it must not cost is an *allocation* apiece: reading this
    // window once took 65,536 of them, two per object of a collection it kept 256
    // entries from (issue #228). One per eight objects is far above what a
    // windowed walk needs and far below what a per-object one takes, which is the
    // gap this bound is placed in.
    assert!(
        measured.blocks < (N0 / 8) as u64,
        "a windowed vlen read must not allocate per object of the collection it \
         walks: measured {measured} over a {N0}-object collection"
    );

    // The third axis, and the one this test was blind to: bytes *moved*. The
    // peak cannot see it, because each directory window replaces the last rather
    // than accumulating, and the block count *falls* when refills get larger —
    // so a change that reads a megabyte per refill instead of a page passes both
    // bounds above while reading thirty times the file. What holds is that the
    // walk crosses the collection about once: it must pass every object before
    // the window, since the format chains each object's position in the previous
    // one, but it has no reason to pass any of them twice.
    assert!(
        measured.bytes < (PAYLOAD_BYTES * 2) as u64,
        "a windowed vlen read must cross its collection about once, not \
         repeatedly: measured {measured} against a {PAYLOAD_BYTES}-byte payload"
    );

    // The strings this read returned are in the measurement, so their size is a
    // floor a measurement of nothing cannot reach.
    const WINDOW_TEXT_BYTES: usize = 256 * STR_LEN;
    assert!(
        measured.live_bytes >= WINDOW_TEXT_BYTES as u64,
        "the strings this read returned are not in its own measurement, so the \
         bounds above are bounding something other than the read: {measured}"
    );

    // The window must still be the right strings.
    assert_eq!(window.len(), 256);
    let expected: Vec<String> = (15_000..15_000 + 256)
        .map(|i| format!("{i:0>width$}", width = STR_LEN))
        .collect();
    assert_eq!(window, expected);
}

/// A dataset whose chunks lie end to end for its whole length is what the
/// coalescing reader (`src/chunk_span.rs`) exists for, and the worst case for
/// its memory bound: a coalescer with no budget would merge all 2,048 chunks
/// into a single 8 MiB span and hold the dataset a second time to serve them.
/// The 256 KiB budget is what keeps the span a constant beside the read rather
/// than a second copy of it.
///
/// The read here is [`Dataset::read_raw`] rather than a typed one on purpose:
/// a typed read allocates its own copy of the dataset, and a term that large
/// hides the one being measured.
///
/// The three bounds are one story told three ways — peak, bytes, blocks — and
/// each catches a defect the others do not: a second buffer of dataset size, a
/// throwaway copy of every chunk, and a per-chunk `Vec` in the scatter loop.
#[test]
fn whole_read_of_adjacent_chunks_costs_a_constant_per_chunk() {
    // 8 MiB of f64 in 4 KiB chunks: 2,048 of them, unfiltered so a chunk's
    // stored bytes are its data and a span holds no less than the file does.
    const N0: usize = 1024 * 1024;
    const DATASET_BYTES: usize = N0 * 8;
    const CHUNK_ELEMS: usize = 512;
    const CHUNKS: u64 = (N0 / CHUNK_ELEMS) as u64;

    let data: Vec<f64> = (0..N0).map(|i| i as f64).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("t")
        .with_f64_data(&data)
        .with_shape(&[N0 as u64])
        .with_chunks(&[CHUNK_ELEMS as u64]);
    builder.write(&path).unwrap();
    drop(data);

    let file = File::open_streaming(&path).unwrap();
    let ds = file.dataset("t").unwrap();

    let (all, measured) = allocation::measure("whole_read", || ds.read_raw().unwrap());

    // The read's own output is the dataset, and the parsed chunk index is most
    // of what is left; an unbudgeted span would add the dataset again.
    assert!(
        measured.peak_bytes < (DATASET_BYTES + DATASET_BYTES / 2) as u64,
        "peak allocation during the whole read must not grow with the run of \
         adjacent chunks; measured {measured} against a {DATASET_BYTES}-byte dataset"
    );

    // Bytes *ever* allocated, which the peak cannot see: a copy of each chunk
    // that is made and dropped costs the dataset over again without moving the
    // high-water mark at all. The read's own output is one dataset; the chunk
    // cache is entitled to its configured budget on top, and nothing else here
    // scales with the data.
    assert!(
        measured.bytes < (DATASET_BYTES + DATASET_BYTES / 4) as u64,
        "a whole read must allocate the dataset about once — its output — not a \
         second time in throwaway per-chunk copies; measured {measured} against a \
         {DATASET_BYTES}-byte dataset"
    );

    // Per-chunk *count*, which neither byte figure can see: a `Vec` of the
    // chunk's coordinates costs 32 bytes and is invisible next to 8 MiB, but it
    // is an allocator round trip per chunk of every dataset this crate reads.
    const BLOCKS_PER_CHUNK: u64 = 3;
    assert!(
        measured.blocks <= BLOCKS_PER_CHUNK * CHUNKS + 256,
        "a whole read must cost a small constant number of allocations per chunk, \
         not a growing one: measured {measured} over {CHUNKS} chunks, above the \
         {BLOCKS_PER_CHUNK} per chunk this bound allows"
    );

    // The read's own output is the dataset, so its size is a floor a measurement
    // of nothing cannot reach — which is what all three ceilings above become if
    // the per-chunk work moves to a thread this region does not see.
    assert!(
        measured.live_bytes >= DATASET_BYTES as u64,
        "the dataset this read returned is not in its own measurement, so the \
         three bounds above are bounding something other than the read: {measured}"
    );

    assert_f64_bytes(&all, (0..N0).map(|index| index as f64));
}

/// With no decompressed chunk retained, the warm read reuses only the cold read's index.
#[rstest]
#[case::buffered(false, "warm_whole_read_buffered")]
#[case::streaming(true, "warm_whole_read_streaming")]
fn a_warm_whole_read_lists_its_chunks_without_an_allocation_per_chunk(
    #[case] streaming: bool,
    #[case] region: &'static str,
) {
    const CHUNKS: u64 = 2048;
    const CHUNK_ELEMS: u64 = 8;

    let data: Vec<f64> = (0..CHUNKS * CHUNK_ELEMS).map(|i| i as f64).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("warm.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("t")
        .with_f64_data(&data)
        .with_shape(&[CHUNKS * CHUNK_ELEMS])
        .with_chunks(&[CHUNK_ELEMS]);
    builder.write(&path).unwrap();
    let access = FileAccessProperties::new()
        .with_chunk_cache(ChunkCacheConfig::new().with_max_bytes(0).with_max_slots(0));
    let file = if streaming {
        File::open_streaming_with_options(&path, access)
    } else {
        File::open_with_options(&path, access)
    }
    .unwrap();
    let ds = file.dataset("t").unwrap();
    let cold = ds.read_raw().unwrap();
    assert!(ds.chunk_cache_stats().index_loaded());

    let (warm, measured) = allocation::measure(region, || ds.read_raw().unwrap());

    assert_f64_bytes(&cold, data.iter().copied());
    assert_eq!(warm, cold);
    assert!(
        measured.live_bytes >= warm.len() as u64,
        "the dataset this read returned is not in its own measurement: {measured}"
    );
    assert!(
        measured.blocks < CHUNKS / 16,
        "a warm whole read must take its {CHUNKS} chunks from the retained index without an \
         allocation per chunk: {measured}"
    );
}

#[rstest]
#[case::buffered_zero(UserblockSize::NONE, false, "large_chunk_buffered_zero")]
#[case::buffered_userblock(UserblockSize::MIN, false, "large_chunk_buffered_userblock")]
#[case::mirrored_userblock(UserblockSize::MIN, true, "large_chunk_mirrored_userblock")]
fn resident_whole_reads_borrow_large_unfiltered_chunks(
    #[case] userblock: UserblockSize,
    #[case] writable: bool,
    #[case] region: &str,
) {
    const CHUNK_ELEMENTS: u64 = 1_048_576;
    let values: Vec<i32> = (0..i32::try_from(CHUNK_ELEMENTS * 2).unwrap()).collect();
    let expected: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let path = test_util::temp::temp_path("large-chunks.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(userblock);
    builder
        .create_dataset("data")
        .with_i32_data(&values)
        .with_shape(&[CHUNK_ELEMENTS * 2])
        .with_chunks(&[CHUNK_ELEMENTS]);
    builder.write(&path).unwrap();
    let access = FileAccessProperties::new()
        .with_chunk_cache(ChunkCacheConfig::new().with_max_bytes(0).with_max_slots(0))
        .with_memory_strategy(MemoryStrategy::Mirrored);
    let file = if writable {
        File::open_rw_with_options(&path, access)
    } else {
        File::open_with_options(&path, access)
    }
    .unwrap();
    if writable {
        assert_eq!(file.edit_backing(), Some(EditBacking::Mirrored));
    }
    let dataset = file.dataset("data").unwrap();
    assert_eq!(dataset.chunk_shape().unwrap(), Some(vec![CHUNK_ELEMENTS]));
    let chunks = dataset.chunks().unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.storage_size)
            .collect::<Vec<_>>(),
        vec![CHUNK_ELEMENTS * 4; 2]
    );
    assert_eq!(dataset.read_raw().unwrap(), expected);
    assert!(dataset.chunk_cache_stats().index_loaded());
    let (actual, allocations) = allocation::measure(region, || dataset.read_raw().unwrap());
    assert_eq!(actual, expected);
    assert!(
        allocations.live_bytes >= expected.len() as u64,
        "{allocations}"
    );
    assert!(
        allocations.peak_bytes < expected.len() as u64 + 256 * 1024,
        "{allocations}"
    );
    assert!(
        allocations.bytes < expected.len() as u64 + 512 * 1024,
        "{allocations}"
    );
    drop(dataset);
    file.close().unwrap();
}

#[rstest]
#[case::rank_two(&[256, 256], &[8, 8], "rank_two_scatter")]
#[case::rank_two_bands(&[1024, 64], &[1, 64], "rank_two_bands")]
#[case::rank_two_edges(&[255, 253], &[8, 8], "rank_two_edges")]
#[case::rank_three(&[32, 32, 32], &[2, 4, 4], "rank_three_scatter")]
#[case::rank_three_bands(&[256, 8, 8], &[1, 8, 8], "rank_three_bands")]
fn warm_chunk_scatter_uses_bounded_allocations(
    #[case] shape: &[u64],
    #[case] chunks: &[u64],
    #[case] region: &str,
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
) {
    let values: Vec<_> = (0..shape.iter().product::<u64>())
        .map(|value| value as f64)
        .collect();
    let expected: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let path = test_util::temp::temp_path("scatter.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("data")
        .with_f64_data(&values)
        .with_shape(shape)
        .with_chunks(chunks);
    builder.write(&path).unwrap();
    let access = FileAccessProperties::new()
        .with_chunk_cache(ChunkCacheConfig::new().with_max_bytes(0).with_max_slots(0));
    let file = backend.open_with_options(&path, access).unwrap();
    let dataset = file.dataset("data").unwrap();
    assert_eq!(dataset.read_raw().unwrap(), expected);
    assert!(dataset.chunk_cache_stats().index_loaded());
    let whole_region = format!("{region}_{backend:?}_whole");
    let rows_region = format!("{region}_{backend:?}_rows");
    let (whole, whole_allocations) =
        allocation::measure(&whole_region, || dataset.read_raw().unwrap());
    let (rows, rows_allocations) = allocation::measure(&rows_region, || {
        dataset.read_raw_rows(1, shape[0] - 2).unwrap()
    });
    let row_bytes = usize::try_from(shape[1..].iter().product::<u64>()).unwrap() * 8;
    assert_eq!(whole, expected);
    assert_eq!(rows, expected[row_bytes..expected.len() - row_bytes]);
    assert!(
        whole_allocations.live_bytes >= whole.len() as u64,
        "{whole_allocations}"
    );
    assert!(
        rows_allocations.live_bytes >= rows.len() as u64,
        "{rows_allocations}"
    );
    assert!(whole_allocations.blocks < 64, "{whole_allocations}");
    assert!(rows_allocations.blocks < 64, "{rows_allocations}");
}

#[rstest]
#[case::buffered(false, "cold_implicit_read_buffered", "warm_implicit_read_buffered")]
#[case::streaming(true, "cold_implicit_read_streaming", "warm_implicit_read_streaming")]
fn a_cold_implicit_read_bounds_extra_allocations_beyond_a_warm_read(
    #[case] streaming: bool,
    #[case] cold_region: &'static str,
    #[case] warm_region: &'static str,
) {
    const CHUNKS: u64 = 131_072;

    let data: Vec<u8> = (0..CHUNKS).flat_map(|i| (i as f64).to_le_bytes()).collect();
    let mut image = Image::starting_with(&[0; 64]);
    let data_address = image.append(&data) as u64;
    let image = image.build_with_dataset(
        "data",
        [
            Message::new(
                MessageType::DATASPACE,
                &dataspace::v2(
                    1,
                    dataspace::Flags::NONE,
                    dataspace::Kind::SIMPLE,
                    &[CHUNKS],
                    None,
                ),
            ),
            Message::new(MessageType::DATATYPE, &datatype::f64_le()),
            Message::new(
                MessageType::DATA_LAYOUT,
                &data_layout::implicit(data_address, &[1], 8),
            ),
        ],
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("implicit.h5");
    std::fs::write(&path, image).unwrap();
    let access = FileAccessProperties::new()
        .with_chunk_cache(ChunkCacheConfig::new().with_max_bytes(0).with_max_slots(0));
    let file = if streaming {
        File::open_streaming_with_options(&path, access)
    } else {
        File::open_with_options(&path, access)
    }
    .unwrap();
    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.chunk_index().unwrap(), Some(ChunkIndex::Implicit));

    let (cold_bytes, cold) = allocation::measure(cold_region, || ds.read_raw().unwrap());
    let (warm_bytes, warm) = allocation::measure(warm_region, || ds.read_raw().unwrap());

    assert_eq!(cold_bytes, data);
    assert_eq!(warm_bytes, data);
    assert!(ds.chunk_cache_stats().index_loaded());
    assert!(
        cold.live_bytes >= warm.live_bytes + CHUNKS * 8,
        "the index the cold read retained is not in its own measurement: cold: {cold}; warm: \
         {warm}"
    );
    assert!(
        warm.live_bytes >= warm_bytes.len() as u64,
        "the dataset the warm read returned is not in its own measurement: {warm}"
    );
    let index_bytes = cold.live_bytes - warm.live_bytes;
    assert!(
        cold.bytes - warm.bytes < index_bytes + CHUNKS * 8,
        "a cold read through an implicit index of {CHUNKS} chunks must allocate the \
         {index_bytes} bytes of the index it retains and little else beside the warm read: \
         cold: {cold}; warm: {warm}"
    );
}

/// A typed whole-dataset read costs its output, not its output *and* a whole
/// copy of the stored bytes beside it.
///
/// Decoding used to be a `read_raw` followed by a conversion of the entire
/// buffer, so both were live at once and the peak was twice the dataset: a
/// caller reading a 4 GiB array needed 8 GiB (issue #289). What replaced it
/// sweeps row windows, and both figures below say so — the peak because only one
/// window of stored bytes stands beside the output, and the byte total because
/// each window is decoded straight into that output rather than into a buffer
/// that is then copied into it.
///
/// The fixture is the one [`whole_read_of_adjacent_chunks_costs_a_constant_per_chunk`]
/// reads raw, at the same stored width as the requested type, so "the dataset"
/// is one figure for both the stored bytes and the decoded values and the two
/// tests are directly comparable. The margin is a quarter of the dataset against
/// a window of one mebibyte — an eighth of it — so a window budget that grew
/// with the dataset instead of staying constant would fail this, and so would a
/// return to reading whole.
#[test]
fn typed_whole_read_costs_its_output_and_a_window() {
    // 8 MiB of f64 in 4 KiB chunks, read as `f64`: output and stored bytes are
    // the same size, so one constant names both.
    const N0: usize = 1024 * 1024;
    const DATASET_BYTES: usize = N0 * 8;
    const CHUNK_ELEMS: usize = 512;

    let data: Vec<f64> = (0..N0).map(|i| i as f64).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("t")
        .with_f64_data(&data)
        .with_shape(&[N0 as u64])
        .with_chunks(&[CHUNK_ELEMS as u64]);
    builder.write(&path).unwrap();
    drop(data);

    let file = File::open_streaming(&path).unwrap();
    let ds = file.dataset("t").unwrap();

    let (values, measured) = allocation::measure("typed_whole_read", || ds.read_f64().unwrap());

    assert!(
        measured.peak_bytes < (DATASET_BYTES + DATASET_BYTES / 4) as u64,
        "peak allocation during a typed whole read must be its output plus a \
         window, not a second copy of the {DATASET_BYTES}-byte dataset; \
         measured {measured}"
    );

    // Bytes *ever* allocated, which the peak cannot see: a window decoded into a
    // fresh `Vec` and then appended to the output, or a chunk copied into a
    // cache that has no room for it, costs the whole dataset over again without
    // moving the high-water mark at all. Two copies are what this read is
    // entitled to — its output, and the stored bytes it sweeps past one window
    // at a time — plus the span buffer each window refills, so the bound is
    // placed where a *third* would put it.
    assert!(
        measured.bytes < (3 * DATASET_BYTES) as u64,
        "a typed whole read must allocate its output and one sweep of the \
         stored bytes, not a third copy of either; measured {measured} against \
         a {DATASET_BYTES}-byte dataset"
    );

    // Per-chunk *count*, which neither byte figure can see, and the axis a sweep
    // is most easily made quadratic on: a window that takes the whole chunk
    // index out of the cache to find the chunks it overlaps allocates per chunk
    // of the *dataset* per window, and this read visits each chunk once through
    // eight windows.
    const BLOCKS_PER_CHUNK: u64 = 5;
    const CHUNKS: u64 = (N0 / CHUNK_ELEMS) as u64;
    assert!(
        measured.blocks <= BLOCKS_PER_CHUNK * CHUNKS + 256,
        "a typed whole read must cost a small constant number of allocations \
         per chunk, not one per chunk per window: measured {measured} over \
         {CHUNKS} chunks, above the {BLOCKS_PER_CHUNK} per chunk this bound allows"
    );

    // The values this read returned are in the measurement, so their size is a
    // floor a measurement of nothing cannot reach.
    assert!(
        measured.live_bytes >= DATASET_BYTES as u64,
        "the values this read returned are not in its own measurement, so the \
         bounds above are bounding something other than the read: {measured}"
    );

    // The values must still be the right ones, in the right order — a sweep that
    // lost or repeated a window would pass every bound above.
    assert_eq!(
        values,
        (0..N0).map(|index| index as f64).collect::<Vec<_>>()
    );
}

/// Writing a chunked dataset costs a constant per chunk, and about one copy of
/// the data beyond the one the caller handed over.
///
/// The two bounds catch different things. The count catches a scratch `Vec` in
/// the per-chunk loop — the splitter kept three of them, one of which was a
/// coordinate vector no caller ever read. The byte bound catches a whole second
/// copy of the dataset, which is what the object-header sizing pass was making:
/// it built the entire data region to learn how long it would be, then dropped it
/// and built it again at the real address (issue #228).
#[test]
fn chunked_write_costs_a_constant_per_chunk() {
    const N0: usize = 1024 * 1024;
    const DATASET_BYTES: usize = N0 * 8;
    const CHUNK_ELEMS: u64 = 512;
    const CHUNKS: u64 = (N0 as u64) / CHUNK_ELEMS;

    let data: Vec<f64> = (0..N0).map(|i| i as f64).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("write.h5");

    let ((), measured) = allocation::measure("chunked_write", || {
        let mut builder = FileBuilder::new();
        builder
            .create_dataset("t")
            .with_f64_data(&data)
            .with_shape(&[N0 as u64])
            .with_chunks(&[CHUNK_ELEMS]);
        builder.write(&path).unwrap();
    });

    // The writer's own copy of the caller's data is in this measurement, so the
    // dataset's size is a floor a measurement of nothing cannot reach.
    assert!(
        measured.bytes >= DATASET_BYTES as u64,
        "the write's own copy of the data is not in this measurement, so the \
         bounds below are measuring something other than the write: {measured}"
    );

    // The bound includes the builder's staged elements, retained encoded chunks, and metadata.
    assert!(
        measured.bytes < (DATASET_BYTES * 7 / 2) as u64,
        "a chunked write may hold the dataset a small number of times over, not \
         one more: measured {measured} against a {DATASET_BYTES}-byte dataset"
    );

    // Measured at 1.06 allocations per chunk, and allowed half again. Two per
    // chunk — what this allowed first — is exactly what *one* restored scratch
    // `Vec` in the splitter loop costs, so a bound there absorbs the defect it
    // names: the splitter kept three, and only restoring all three failed it.
    const BLOCKS_PER_TWO_CHUNKS: u64 = 3;
    assert!(
        measured.blocks <= BLOCKS_PER_TWO_CHUNKS * CHUNKS / 2 + 256,
        "a chunked write must cost a small constant number of allocations per \
         chunk: measured {measured} over {CHUNKS} chunks, above the \
         {BLOCKS_PER_TWO_CHUNKS} per two chunks this bound allows"
    );

    // The file must still be the right file.
    let file = File::open(&path).unwrap();
    let back = file.dataset("t").unwrap().read_f64().unwrap();
    assert_eq!(back, data);
}

/// A filtered write builds one compressor, not one per chunk.
///
/// A zlib compressor holds ~300 KiB of hash tables, so building one per chunk
/// made writing this 8 MiB dataset allocate 743 MB — ninety times the data, and
/// by a wide margin the most expensive thing this crate did (issue #228). The
/// bound is on bytes because that is the axis a per-chunk codec moves; a count
/// bound alone would miss it, since the codec is a handful of allocations that
/// happen to be enormous.
#[test]
fn filtered_write_does_not_build_a_compressor_per_chunk() {
    const N0: usize = 1024 * 1024;
    const DATASET_BYTES: usize = N0 * 8;
    const CHUNK_ELEMS: u64 = 512;
    const CHUNKS: u64 = (N0 as u64) / CHUNK_ELEMS;

    let data: Vec<f64> = (0..N0).map(|i| i as f64).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("deflate.h5");

    let ((), measured) = allocation::measure("filtered_write", || {
        let mut builder = FileBuilder::new();
        builder
            .create_dataset("t")
            .with_f64_data(&data)
            .with_shape(&[N0 as u64])
            .with_chunks(&[CHUNK_ELEMS])
            .with_deflate(DeflateLevel::Three);
        builder.write(&path).unwrap();
    });

    assert!(
        measured.bytes >= DATASET_BYTES as u64,
        "the write's own copy of the data is not in this measurement: {measured}"
    );

    // One compressor per chunk would be ~615 MiB here, seventy times this bound.
    assert!(
        measured.bytes < (DATASET_BYTES * 4) as u64,
        "a filtered write must build its compressor once, not per chunk: measured \
         {measured} against a {DATASET_BYTES}-byte dataset over {CHUNKS} chunks"
    );

    // The block limit also catches one allocation for each filter description per chunk.
    assert!(
        measured.blocks <= 7 * CHUNKS / 2 + 256,
        "filtered write allocated an extra block per chunk: {measured} over {CHUNKS} chunks"
    );

    let file = File::open(&path).unwrap();
    let back = file.dataset("t").unwrap().read_f64().unwrap();
    assert_eq!(back, data);
}

/// The same for the read side, where a decoder per chunk cost 174 MB to read the
/// 8 MiB back (issue #228).
#[test]
fn filtered_read_does_not_build_a_decompressor_per_chunk() {
    const N0: usize = 1024 * 1024;
    const DATASET_BYTES: usize = N0 * 8;
    const CHUNKS: u64 = (N0 as u64) / 512;

    let data: Vec<f64> = (0..N0).map(|i| i as f64).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("deflate_read.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("t")
        .with_f64_data(&data)
        .with_shape(&[N0 as u64])
        .with_chunks(&[512])
        .with_deflate(DeflateLevel::Three);
    builder.write(&path).unwrap();
    drop(data);

    let file = File::open_streaming(&path).unwrap();
    let ds = file.dataset("t").unwrap();
    let (all, measured) = allocation::measure("filtered_read", || ds.read_raw().unwrap());

    assert!(
        measured.bytes >= DATASET_BYTES as u64,
        "the read's own output is not in this measurement: {measured}"
    );

    // One decoder per chunk would be ~85 MiB of Huffman state on top of the
    // decoded data, ten times this bound.
    assert!(
        measured.bytes < (DATASET_BYTES * 4) as u64,
        "a filtered read must build its decoder once, not per chunk: measured \
         {measured} against a {DATASET_BYTES}-byte dataset"
    );

    // The block limit also catches one allocation for each filter description per chunk.
    assert!(
        measured.blocks <= 9 * CHUNKS / 2 + 256,
        "filtered read allocated an extra block per chunk: {measured} over {CHUNKS} chunks"
    );

    assert_f64_bytes(&all, (0..N0).map(|index| index as f64));
}

/// Opening one child of a group costs the group's header, not one allocation per
/// child in it.
///
/// A group stores its children as one Link message each, and a lookup used to
/// build an entry with an owned name for every one of them before returning the
/// one it was asked for: 2,084 allocations and 310 KiB to open a single dataset
/// out of 1,024, paid again on the next open. That is what makes walking a group
/// quadratic in its size, and this is the rule that keeps the walk linear
/// (issue #228).
///
/// Three measurements, because the lookup is reached by three routes that are
/// separate code: a path from the file and a name within an opened group, and
/// each of those over a streaming file and a buffered one — `File::open`, the
/// default entry point, resolves through a different function than
/// `File::open_streaming` does, and a bound on one says nothing about the other.
#[test]
fn opening_one_child_does_not_allocate_per_child_of_the_group() {
    const OBJECTS: usize = 1024;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("many.h5");
    let mut builder = FileBuilder::new();
    for i in 0..OBJECTS {
        builder
            .create_dataset(&format!("d{i:05}"))
            .with_i32_data(&[i as i32]);
    }
    builder.write(&path).unwrap();

    // The last child written, so the scan passes every link before it: the worst
    // case for a lookup that stops at its match, and the one where a per-child
    // cost cannot hide behind an early hit.
    let last = format!("d{:05}", OBJECTS - 1);
    let file = File::open_streaming(&path).unwrap();
    // Opened before the measurements: a buffered open reads the whole file, and
    // what is being measured is the lookup inside it.
    let buffered = File::open(&path).unwrap();

    let (by_path, path_measured) =
        allocation::measure("child_by_path", || file.dataset(&last).unwrap());
    let (by_name, name_measured) =
        allocation::measure("child_by_name", || file.root().dataset(&last).unwrap());
    let (buffered_by_path, buffered_measured) =
        allocation::measure("child_by_path_buffered", || {
            buffered.dataset(&last).unwrap()
        });

    for (what, measured) in [
        ("by path", path_measured),
        ("in the group", name_measured),
        ("by path in a buffered file", buffered_measured),
    ] {
        // Measured at 18 allocations: the group's header, the opened dataset's
        // header, and its handful of messages. One per link is 1,026, and the
        // bound sits between the two rather than near either.
        assert!(
            measured.blocks < (OBJECTS / 8) as u64,
            "opening one child {what} must not allocate per child of the group: \
             measured {measured} in a {OBJECTS}-child group"
        );

        // The other axis, which the count cannot see: the group's header is read
        // whole (about 30 stored bytes per child, so ~30 KiB here) because the
        // links must be scanned, but nothing may allocate several times over it.
        // The per-child entries this replaced cost ten times the header.
        assert!(
            measured.bytes < (OBJECTS * 128) as u64,
            "opening one child {what} must allocate about its group's header, not \
             a multiple of it: measured {measured} in a {OBJECTS}-child group"
        );

        // The handle each call returned — its parsed object header among it — is
        // in this measurement, so its size is a floor a measurement of nothing
        // cannot reach.
        assert!(
            measured.live_bytes >= 128,
            "the handle this open returned is not in its own measurement, so the \
             bounds above are bounding something other than the open: {measured}"
        );
    }

    // Every handle must still be the right dataset.
    assert_eq!(by_path.read_i32().unwrap(), vec![(OBJECTS - 1) as i32]);
    assert_eq!(by_name.read_i32().unwrap(), vec![(OBJECTS - 1) as i32]);
    assert_eq!(
        buffered_by_path.read_i32().unwrap(),
        vec![(OBJECTS - 1) as i32]
    );
}

/// Writing variable-length strings copies the text into the heap collections it
/// has to end up in, and not into a per-element buffer on the way.
///
/// The writer is handed `&[&str]` and used to own each one before staging it: a
/// second copy of the whole payload, in one allocation per string — 4 MiB in
/// 32,768 blocks to write 4 MiB of text (issue #228). The count is the axis that
/// names that defect, since a copy made one string at a time is one allocation
/// per string whatever it costs in bytes; the byte bound beside it is the one
/// that would still catch the same copy made in a single buffer.
#[test]
fn vlen_string_write_does_not_own_each_string_before_staging_it() {
    const N0: usize = 32 * 1024;
    const STR_LEN: usize = 128;
    const PAYLOAD_BYTES: usize = N0 * STR_LEN;

    // Built before the measurement: these are the caller's strings, and the
    // question is what the writer adds to them.
    let strings: Vec<String> = (0..N0)
        .map(|i| format!("{i:0>width$}", width = STR_LEN))
        .collect();
    let refs: Vec<&str> = strings.iter().map(String::as_str).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vlen.h5");

    let ((), measured) = allocation::measure("vlen_string_write", || {
        let mut builder = FileBuilder::new();
        builder.create_dataset("labels").with_vlen_strings(&refs);
        builder.write(&path).unwrap();
    });

    // The collections this write builds hold the payload, so its size is a floor
    // a measurement of nothing cannot reach.
    assert!(
        measured.bytes >= PAYLOAD_BYTES as u64,
        "the heap collections this write built are not in its own measurement, so \
         the bounds below are measuring something other than the write: {measured}"
    );

    // Measured at 104 allocations for 32,768 strings: the collections, the
    // reference vector, the file image and the object headers. One per string is
    // the defect.
    assert!(
        measured.blocks < (N0 / 8) as u64,
        "a variable-length string write must not allocate per string: measured \
         {measured} over {N0} strings"
    );

    // Measured at 1.57 copies of the payload: the heap collections it is written
    // into, and the file image assembled around them, with the 16-byte reference
    // per element beside those. One more whole copy — the caller's strings owned
    // before staging, however few allocations that took — is 2.8, and this bound
    // sits below it.
    assert!(
        measured.bytes < (PAYLOAD_BYTES * 2) as u64,
        "a variable-length string write may hold the payload about once, not one \
         time more: measured {measured} against a {PAYLOAD_BYTES}-byte payload"
    );

    // The file must still be the right file.
    let file = File::open(&path).unwrap();
    let back = file.dataset("labels").unwrap().read_string().unwrap();
    assert_eq!(back, strings);
}

/// One in-place append allocates on the order of its own batch, not of the
/// dataset it grows.
///
/// This is the write path that runs many times over one file, so a per-call cost
/// that scales with what is already there is what turns a long append loop
/// quadratic. The measurement is of the *last* append, onto a dataset already
/// several hundred times the batch, so a term in the dataset's size cannot hide
/// in a term in the batch's.
#[test]
fn one_append_costs_its_batch_not_the_dataset() {
    const CHUNK_ELEMS: u64 = 512;
    const BATCH_BYTES: usize = CHUNK_ELEMS as usize * 8;
    const WARMUP_APPENDS: usize = 512;
    const DATASET_BYTES: usize = (WARMUP_APPENDS + 1) * BATCH_BYTES;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("growing.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("t")
        .with_f64_data(&[0.0f64; CHUNK_ELEMS as usize])
        .with_shape(&[CHUNK_ELEMS])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[CHUNK_ELEMS]);
    builder.write(&path).unwrap();

    let batch: Vec<f64> = (0..CHUNK_ELEMS).map(|i| i as f64).collect();
    {
        // One barrier at close rather than one per append: an `fsync` apiece
        // would make this test's runtime, not its allocations, the thing to look
        // at (see `.config/nextest.toml`, which fails the run on either).
        let file = hdf5_pure::File::open_rw_with_options(
            &path,
            hdf5_pure::FileAccessProperties::new().with_sync_policy(hdf5_pure::SyncPolicy::OnClose),
        )
        .unwrap();
        let mut ds = file.dataset("t").unwrap();
        for _ in 0..WARMUP_APPENDS {
            ds.append(&batch).unwrap();
        }

        let ((), measured) = allocation::measure("append_in_place", || ds.append(&batch).unwrap());

        // The batch's own bytes are staged inside this measurement, so their size
        // is a floor a measurement of nothing cannot reach.
        assert!(
            measured.bytes >= BATCH_BYTES as u64,
            "the appended batch is not in this append's own measurement, so the \
             bounds below are measuring something other than the append: {measured}"
        );

        // Measured at 14 KiB and 28 allocations for a 4 KiB batch onto a 2 MiB
        // dataset: the batch staged and written, and the chunk index and object
        // header around it. The same append measured against datasets of 17,
        // 513 and 4,097 chunks costs 13.4, 14.3 and 15.3 KiB and 28 allocations
        // throughout — the growth is in the index's *depth*, not its size, so
        // eight batches' worth is a ceiling with room in it, and anything
        // proportional to the dataset (512 batches here) is far above it.
        assert!(
            measured.bytes < (BATCH_BYTES * 8) as u64,
            "one append must allocate on the order of its {BATCH_BYTES}-byte batch, \
             not of the {DATASET_BYTES}-byte dataset it grows: measured {measured}"
        );
        assert!(
            measured.blocks < 128,
            "one append must cost a bounded number of allocations, whatever the \
             dataset's size: measured {measured}"
        );
    }

    // The session is closed before reading: an open read-write file holds a lock
    // that a second open refuses on Windows.
    let file = File::open(&path).unwrap();
    let back = file.dataset("t").unwrap().read_f64().unwrap();
    // The dataset it was written with, plus the warmup appends and the measured
    // one.
    let mut expected = vec![0.0; CHUNK_ELEMS as usize];
    expected.extend(batch.repeat(WARMUP_APPENDS + 1));
    assert_eq!(back, expected);
}

/// Gathering writes does not copy what it has already gathered on every new one.
///
/// Chunks are placed one after another, so each new write starts exactly where
/// the last one ended. Merging that by building a fresh buffer and copying the
/// old one into it makes the gathering quadratic in the number of writes it
/// holds: filling the budget that way copies hundreds of megabytes to write one
/// (issue #288). Extending the pending run in place makes it linear.
///
/// Measured through one large append rather than many small ones, because that
/// is what grows a single run to the whole gather budget: an append of this size
/// is split into one-megabyte batches, and within a batch the chunk writes are
/// contiguous and merge into one run that reaches the budget before it drains.
/// Many separate appends would not show it — each drains at its own barriers
/// while its run is still a few kilobytes — and a commit shows the same defect
/// but buries it under the header, link and index work that dominates at any
/// size worth measuring.
#[test]
fn gathering_writes_does_not_recopy_what_it_holds() {
    const CHUNK: usize = 512;
    const CHUNKS: usize = 1024;
    const RAW_BYTES: u64 = (CHUNK * CHUNKS * 8) as u64;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("buffered.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_page_size(FileSpacePageSize::DEFAULT),
    );
    builder
        .create_dataset("growing")
        .with_f64_data(&[1.0f64; CHUNK])
        .with_shape(&[CHUNK as u64])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[CHUNK as u64]);
    builder.write(&path).unwrap();

    // Built outside the region: this is the caller's buffer, not the library's.
    let batch: Vec<f64> = (0..CHUNK * CHUNKS).map(|i| (i % CHUNK) as f64).collect();
    // Scoped: a `Dataset` owns a handle on the session, so the session outlives
    // `file` itself and the append is only certainly on disk once both are gone.
    let measured = {
        let file = File::open_rw_with_options(
            &path,
            hdf5_pure::FileAccessProperties::new().with_sync_policy(hdf5_pure::SyncPolicy::OnClose),
        )
        .unwrap();
        let mut ds = file.dataset("growing").unwrap();
        allocation::measure("gathered_append", || {
            ds.append(&batch).unwrap();
        })
        .1
    };

    // The appended data is in this measurement, so a measurement of nothing
    // cannot pass the ceiling below by default.
    assert!(
        measured.bytes >= RAW_BYTES,
        "the appended data is not in this measurement, so the bound below is \
         measuring something other than the append: {measured}"
    );

    // Measured at 5.2x the raw bytes when a pending run is extended in place, and
    // at 132x when each write rebuilds the run it lands in. The ceiling sits
    // between them with room on both sides, since the point is the shape of the
    // curve rather than one host's constant.
    assert!(
        measured.bytes < 32 * RAW_BYTES,
        "gathering must not recopy what it holds: measured {measured} against \
         {RAW_BYTES} bytes of appended data"
    );

    let file = File::open(&path).unwrap();
    let back = file.dataset("growing").unwrap().read_f64().unwrap();
    let mut expected = vec![1.0; CHUNK];
    expected.extend_from_slice(&batch);
    assert_eq!(back, expected);
}

/// A page buffer does not recopy what it holds when a write lands *inside* a run
/// it is already holding.
///
/// The companion to the bound above, through the configuration that reaches the
/// other fast path. `absorb` has two: extending a pending run at its end, which
/// one large append reaches by itself, and patching a write that falls wholly
/// inside a run already held, which needs a run grown to the byte budget and
/// then written into again. Only a buffer held *across* operations gets there —
/// every append here re-patches the object header and chunk index that the runs
/// before it left in the buffer — so the default gathering cannot bound this
/// path and this is the vehicle that can (issues #288 and #308).
#[test]
fn a_page_buffer_does_not_recopy_what_it_holds_across_operations() {
    const CHUNK: usize = 512;
    const APPENDS: usize = 256;
    const RAW_BYTES: u64 = (CHUNK * APPENDS * 8) as u64;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("page_buffered.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_page_size(FileSpacePageSize::DEFAULT),
    );
    builder
        .create_dataset("growing")
        .with_f64_data(&[1.0f64; CHUNK])
        .with_shape(&[CHUNK as u64])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[CHUNK as u64]);
    builder.write(&path).unwrap();

    // Built outside the region: this is the caller's buffer, not the library's.
    let batch: Vec<f64> = (0..CHUNK).map(|i| i as f64).collect();
    // Scoped: a `Dataset` owns a handle on the session, so the session outlives
    // `file` itself, and with a page buffer that is the difference between a
    // file with every append in it and one still holding them.
    let measured = {
        let file = File::open_rw_with_options(
            &path,
            hdf5_pure::FileAccessProperties::new()
                .with_sync_policy(hdf5_pure::SyncPolicy::OnClose)
                .with_page_buffer_size(1 << 20),
        )
        .unwrap();
        let mut ds = file.dataset("growing").unwrap();
        allocation::measure("page_buffered_appends", || {
            for _ in 0..APPENDS {
                ds.append(&batch).unwrap();
            }
        })
        .1
    };

    // The appended data is in this measurement, so a measurement of nothing
    // cannot pass the ceiling below by default.
    assert!(
        measured.bytes >= RAW_BYTES,
        "the appended data is not in this measurement, so the bound below is \
         measuring something other than the appends: {measured}"
    );

    // Measured at 7.3x the raw bytes when a write inside a held run patches it in
    // place, and at 388x when each one rebuilds the run it lands in (7.7 MB
    // against 407 MB, to write one megabyte) — nearly two orders of magnitude
    // apart, so this bound need not be tight to separate them.
    assert!(
        measured.bytes < 64 * RAW_BYTES,
        "a page buffer must not recopy what it holds: measured {measured} against \
         {RAW_BYTES} bytes of appended data"
    );

    let file = File::open(&path).unwrap();
    let back = file.dataset("growing").unwrap().read_f64().unwrap();
    let mut expected = vec![1.0; CHUNK];
    expected.extend(batch.repeat(APPENDS));
    assert_eq!(back, expected);
}

/// A write too large for the gather budget is issued rather than copied into it.
///
/// Gathering holds a write so it can be merged with its neighbours, and a write
/// already larger than the whole budget has no neighbours it could be merged
/// with — it is flushed on the very next line. Absorbing it first therefore buys
/// nothing and costs a full copy of it, which for a staged commit is a second
/// copy of the dataset (issue #288). The bound is on bytes because that is the
/// only axis this moves: the allocation *count* is identical either way.
#[test]
fn a_write_larger_than_the_gather_budget_is_not_copied_into_it() {
    const ELEMS: usize = 2 * 1024 * 1024;
    const DATASET_BYTES: u64 = (ELEMS * 8) as u64;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big_commit.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("seed")
        .with_f64_data(&[1.0f64; 8])
        .with_shape(&[8]);
    builder.write(&path).unwrap();

    let data: Vec<f64> = (0..ELEMS).map(|i| i as f64).collect();
    let measured = {
        let file = File::open_rw(&path).unwrap();
        let root = file.root();
        let ((), m) = allocation::measure("big_staged_commit", || {
            root.create_dataset("big", |b| {
                b.with_f64_data(&data).with_shape(&[ELEMS as u64]);
            })
            .unwrap();
            file.commit().unwrap();
        });
        m
    };

    // The staged dataset is in this measurement, so a commit that did nothing
    // cannot reach this floor.
    assert!(
        measured.bytes >= DATASET_BYTES,
        "the staged dataset is not in this measurement: {measured}"
    );

    // Measured at 1.00x the dataset with the bypass and 2.00x without it: the
    // difference is exactly one more copy. Half again is ample headroom for the
    // headers and the link the commit also writes, and still nowhere near two.
    assert!(
        measured.bytes < DATASET_BYTES * 3 / 2,
        "a staged commit must not copy its data into the gather buffer as well: \
         measured {measured} against a {DATASET_BYTES}-byte dataset"
    );
    let file = File::open(&path).unwrap();
    let dataset = file.dataset("big").unwrap();
    assert_eq!(dataset.shape().unwrap(), &[ELEMS as u64]);
    assert_eq!(dataset.read_f64().unwrap(), data);
    assert_eq!(
        file.dataset("seed").unwrap().read_f64().unwrap(),
        vec![1.0; 8]
    );
}

/// A bounded read-write open holds a metadata-sized amount of the file, not a
/// copy of it — and the mirror beside it is what makes that a measurement rather
/// than a hope (issue #389).
///
/// This is the rule behind `MemoryStrategy::Auto` preferring the bounded engine:
/// the choice is only worth making if the two backings really differ by the size
/// of the file. The contrast is measured on the same file through the same
/// appends, so nothing but the strategy separates the two figures.
#[test]
fn open_rw_of_a_persisting_fsm_file_does_not_allocate_the_file() {
    // 16 MiB in 32 KiB chunks, persisting its free-space managers under the
    // default FsmAggr strategy: the ordinary growing file of issue #389.
    const CHUNK_ELEMS: usize = 4096;
    const CHUNKS: usize = 512;
    const APPENDS: usize = 4;
    const BATCH_BYTES: u64 = (CHUNK_ELEMS * 8) as u64;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fsm.h5");
    let data: Vec<u64> = (0..(CHUNK_ELEMS * CHUNKS) as u64).collect();
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(0),
    );
    builder
        .create_dataset("samples")
        .with_u64_data(&data)
        .with_shape(&[data.len() as u64])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[CHUNK_ELEMS as u64]);
    builder.write(&path).unwrap();
    drop(data);

    let file_len = std::fs::metadata(&path).unwrap().len();
    let batch: Vec<u64> = (0..CHUNK_ELEMS as u64).collect();

    // One barrier at close rather than one per append; this test measures
    // allocations, not `fsync` cadence.
    let access = |strategy| {
        FileAccessProperties::new()
            .with_sync_policy(SyncPolicy::OnClose)
            .with_memory_strategy(strategy)
    };

    let open_and_append = |name: &'static str, strategy, batches| {
        let (file, measured) = allocation::measure(name, || {
            let file = File::open_rw_with_options(&path, access(strategy)).unwrap();
            let mut ds = file.dataset("samples").unwrap();
            for _ in 0..APPENDS {
                ds.append(&batch).unwrap();
            }
            drop(ds);
            file
        });
        let backing = file.edit_backing();
        file.close().unwrap();
        let file = File::open(&path).unwrap();
        let dataset = file.dataset("samples").unwrap();
        let mut expected: Vec<_> = (0..(CHUNK_ELEMS * CHUNKS) as u64).collect();
        expected.extend(batch.repeat(batches));
        assert_eq!(dataset.shape().unwrap(), &[expected.len() as u64]);
        assert_eq!(dataset.read_u64().unwrap(), expected);
        (backing, measured)
    };

    let (backing, bounded) = open_and_append(
        "bounded_open_rw_of_a_large_file",
        MemoryStrategy::Auto,
        APPENDS,
    );
    assert_eq!(
        backing,
        Some(EditBacking::Bounded),
        "Auto must reach the bounded engine on this file, or the bounds below are \
         measuring the mirror"
    );

    // The appended batches are staged inside this measurement, so their bytes are
    // a floor a measurement of nothing cannot reach — and the session itself is
    // returned from the region, so its retained allocations are in `live_bytes`.
    assert!(
        bounded.bytes >= BATCH_BYTES * APPENDS as u64,
        "the appended batches are not in this measurement, so the ceilings below \
         are bounding something other than the session: {bounded}"
    );
    assert!(
        bounded.blocks > 0,
        "no allocation was recorded for the open at all, so this region saw none \
         of the work it names: {bounded}"
    );

    // What a bounded session keeps is the metadata it is holding plus whatever an
    // append is building — a figure with no term in the file's length. Measured
    // at 3.9 KiB live and 101 KiB at peak against a 16 MiB file, where the mirror
    // below holds 32 MiB live, so a sixteenth of the file is orders above the one
    // and orders below the other.
    assert!(
        bounded.live_bytes < file_len / 16,
        "a bounded session must not retain a copy of the {file_len}-byte file: \
         {bounded}"
    );
    assert!(
        bounded.peak_bytes < file_len / 8,
        "a bounded session must not build a copy of the {file_len}-byte file even \
         transiently: {bounded}"
    );

    // The contrast that makes the ceilings above mean something: the same file,
    // the same appends, one strategy apart.
    let (backing, mirrored) = open_and_append(
        "mirrored_open_rw_of_a_large_file",
        MemoryStrategy::Mirrored,
        APPENDS * 2,
    );
    assert_eq!(backing, Some(EditBacking::Mirrored));
    assert!(
        mirrored.live_bytes > file_len,
        "the mirror is supposed to hold the whole {file_len}-byte file, so a \
         measurement below that is not measuring the mirror and the bounded \
         bounds above are not a contrast with anything: {mirrored}"
    );
}

/// Repeated staged appends do not retain memory per commit (issue #412).
///
/// A staged append relocates: the commit writes the new chunks, rebuilds the
/// dataset's chunk index and moves its object header. The report was that a
/// process doing this in a loop grew without bound, which would mean each commit
/// left something behind — a vacated index the engine still held, a batch it did
/// not release. On the bounded engine a commit holds only what it is building,
/// so what is live after a run of commits must not scale with their number, and
/// the peak of that run must be the scratch of *one* commit rather than the sum
/// of them.
///
/// The mirror is the other backing and has its own rule below: it holds the file,
/// so its retained memory scales with the file and must not scale with anything
/// else.
#[test]
fn repeated_staged_append_commits_retain_nothing_per_commit() {
    const DATASETS: usize = 4;
    const CHUNK_ELEMS: u64 = 512;
    const BATCH_BYTES: usize = CHUNK_ELEMS as usize * 8;
    const WARMUP_COMMITS: usize = 128;
    const MEASURED_COMMITS: usize = 64;
    const APPENDED_BYTES: usize = MEASURED_COMMITS * DATASETS * BATCH_BYTES;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("growing.h5");
    let batch = write_growing_fixture(&path, DATASETS, CHUNK_ELEMS);
    {
        let file = File::open_rw_with_options(
            &path,
            FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
        )
        .unwrap();
        assert_eq!(file.edit_backing(), Some(EditBacking::Bounded));
        for _ in 0..WARMUP_COMMITS {
            commit_one_batch_per_dataset(&file, DATASETS, &batch);
        }

        let ((), measured) = allocation::measure("staged_append_commits", || {
            for _ in 0..MEASURED_COMMITS {
                commit_one_batch_per_dataset(&file, DATASETS, &batch);
            }
        });

        // Every batch is staged inside this region, so their sum is a floor a
        // measurement of nothing cannot reach.
        assert!(
            measured.bytes >= APPENDED_BYTES as u64,
            "the appended batches are not in the commits' own measurement, so the \
             bounds below are measuring something other than the commits: {measured}"
        );

        // Measured at 0 to 8 KiB live after 64 commits. A commit that kept even
        // one batch per dataset would leave 64 times this ceiling behind; one
        // that kept its rebuilt index would leave far more.
        assert!(
            measured.live_bytes < (2 * DATASETS * BATCH_BYTES) as u64,
            "{MEASURED_COMMITS} staged-append commits must not retain memory in \
             proportion to their number: measured {measured}"
        );

        // Measured at 115 KiB against 1 MiB appended. The peak is one commit's
        // scratch — four batches, the four rebuilt indexes and headers — and it
        // grows with the datasets' chunk count (218 KiB and 353 KiB after 512 and
        // 1024 warm-up commits), never with the number of commits measured. A run
        // that carried anything from one commit into the next would peak at the
        // sum of what it appended, above this ceiling four times over.
        assert!(
            measured.peak_bytes < (APPENDED_BYTES / 4) as u64,
            "the peak over {MEASURED_COMMITS} staged-append commits must be one \
             commit's scratch, not the {APPENDED_BYTES} bytes they appended \
             together: measured {measured}"
        );
    }

    // The session is closed before reading: an open read-write file holds a lock
    // that a second open refuses on Windows.
    assert_growing_fixture_contents(
        &path,
        DATASETS,
        CHUNK_ELEMS,
        WARMUP_COMMITS + MEASURED_COMMITS,
    );
}

/// The whole-file mirror holds the file and nothing that grows with the number
/// of commits made through it (issue #412).
///
/// Its memory scales with the file by design — that is the cost
/// [`MemoryStrategy::Mirrored`] states — so the rule is stated against the file
/// the run ends with: the image, with the headroom its growth leaves, and the
/// scratch of the last commit. An image that kept its superseded bytes, or a
/// commit that held on to a copy of it, would put the run's retained and peak
/// memory at a multiple of the file these ceilings do not allow.
#[test]
fn the_mirror_retains_the_file_not_the_commits_made_through_it() {
    const DATASETS: usize = 4;
    const CHUNK_ELEMS: u64 = 512;
    const COMMITS: usize = 192;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("growing.h5");
    let batch = write_growing_fixture(&path, DATASETS, CHUNK_ELEMS);

    // The open is inside the region so the image itself is in the measurement,
    // and the file is handed back out of it so that image is still live when the
    // region closes.
    let (file, measured) = allocation::measure("mirrored_staged_append_commits", || {
        let file = File::open_rw_with_options(
            &path,
            FileAccessProperties::new()
                .with_sync_policy(SyncPolicy::OnClose)
                .with_memory_strategy(MemoryStrategy::Mirrored),
        )
        .unwrap();
        assert_eq!(file.edit_backing(), Some(EditBacking::Mirrored));
        for _ in 0..COMMITS {
            commit_one_batch_per_dataset(&file, DATASETS, &batch);
        }
        file
    });
    let file_size = file.file_size();

    // The image is the floor: a measurement that does not contain the file is
    // measuring something other than the mirror.
    assert!(
        measured.live_bytes >= file_size,
        "the mirror's image of the {file_size}-byte file is not in its own \
         measurement, so the ceilings below are bounding something else: {measured}"
    );

    // Measured at 1.5 file sizes live and at peak for a 3.2 MiB file. The image
    // grows in steps, so up to two file sizes is the image alone, and a growth
    // step holds the old and new image together; a mirror that kept one
    // superseded image per commit would hold hundreds. The peak ceiling is not
    // tight enough to see one transient copy of the image inside a commit — that
    // fits under the growth step's headroom — only copies that accumulate.
    assert!(
        measured.live_bytes < 3 * file_size,
        "the mirror must retain its image of the {file_size}-byte file and the \
         headroom of that image's growth, not what {COMMITS} commits made through \
         it: measured {measured}"
    );
    assert!(
        measured.peak_bytes < 4 * file_size,
        "the peak over {COMMITS} mirrored commits must be a small multiple of the \
         {file_size}-byte image, not the images those commits built together: \
         measured {measured}"
    );

    drop(file);
    assert_growing_fixture_contents(&path, DATASETS, CHUNK_ELEMS, COMMITS);
}

/// `datasets` rank-1 unlimited f64 datasets of one `chunk_elems` chunk each, and
/// the batch the tests above append to them.
fn write_growing_fixture(path: &std::path::Path, datasets: usize, chunk_elems: u64) -> Vec<f64> {
    let mut builder = FileBuilder::new();
    let initial = vec![0.0f64; chunk_elems as usize];
    for i in 0..datasets {
        builder
            .create_dataset(&format!("d{i}"))
            .with_f64_data(&initial)
            .with_shape(&[chunk_elems])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[chunk_elems]);
    }
    builder.write(path).unwrap();
    (0..chunk_elems).map(|i| i as f64).collect()
}

/// One staged append of `batch` onto each dataset, then one commit — the loop
/// issue #412 describes.
fn commit_one_batch_per_dataset(file: &File, datasets: usize, batch: &[f64]) {
    for i in 0..datasets {
        file.dataset(&format!("d{i}"))
            .unwrap()
            .append_staged(|b| {
                b.append_f64(batch);
            })
            .unwrap();
    }
    file.commit().unwrap();
}

/// Checks every dataset's shape and complete initial and appended values.
fn assert_growing_fixture_contents(
    path: &std::path::Path,
    datasets: usize,
    chunk_elems: u64,
    commits: usize,
) {
    let file = File::open(path).unwrap();
    for i in 0..datasets {
        let dataset = file.dataset(&format!("d{i}")).unwrap();
        let mut expected = vec![0.0; chunk_elems as usize];
        let batch: Vec<_> = (0..chunk_elems).map(|index| index as f64).collect();
        expected.extend(batch.repeat(commits));
        assert_eq!(dataset.shape().unwrap(), &[expected.len() as u64]);
        assert_eq!(dataset.read_f64().unwrap(), expected);
    }
}

#[rstest]
#[case::streamed_flat(false, false, FileSpaceStrategy::None, "encoded_streamed_flat")]
#[case::buffered_flat(true, false, FileSpaceStrategy::None, "encoded_buffered_flat")]
#[case::streamed_page(false, false, FileSpaceStrategy::Page, "encoded_streamed_page")]
#[case::buffered_page(true, false, FileSpaceStrategy::Page, "encoded_buffered_page")]
#[case::streamed_deflate(false, true, FileSpaceStrategy::None, "encoded_streamed_deflate")]
#[case::buffered_deflate(true, true, FileSpaceStrategy::None, "encoded_buffered_deflate")]
#[case::streamed_page_deflate(
    false,
    true,
    FileSpaceStrategy::Page,
    "encoded_streamed_page_deflate"
)]
#[case::buffered_page_deflate(true, true, FileSpaceStrategy::Page, "encoded_buffered_page_deflate")]
fn encoded_chunk_finish_avoids_a_payload_sized_assembly_allocation(
    #[case] buffered: bool,
    #[case] filtered: bool,
    #[case] strategy: FileSpaceStrategy,
    #[case] region: &'static str,
) {
    const PAYLOAD: usize = 8 * 1024 * 1024;
    let mut state = 0x9e37_79b9_u32;
    let data: Vec<_> = (0..PAYLOAD)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state.to_le_bytes()[0]
        })
        .collect();
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(strategy));
    let dataset = builder.create_dataset("encoded");
    dataset.with_u8_data(&data).with_chunks(&[65_536]);
    if filtered {
        dataset.with_deflate(DeflateLevel::Three);
    }
    let destination = if buffered {
        OutputDestination::Buffered
    } else {
        let path = test_util::temp::temp_path("encoded");
        let file = StdFile::create(&path).unwrap();
        OutputDestination::Streamed { file, path }
    };
    let (output, measured) = allocation::measure(region, || destination.finish(builder));
    assert!(measured.bytes >= PAYLOAD as u64, "{measured}");
    let byte_budget = PAYLOAD * (if filtered { 21 } else { 6 } + 4 * usize::from(buffered)) / 4;
    let peak_budget = PAYLOAD * (if filtered { 10 } else { 6 } + 4 * usize::from(buffered)) / 4;
    assert!(
        measured.bytes < byte_budget as u64,
        "{measured} against {byte_budget} bytes"
    );
    assert!(
        measured.peak_bytes < peak_budget as u64,
        "{measured} against {peak_budget} bytes at peak"
    );
    let file = output.open();
    let dataset = file.dataset("encoded").unwrap();
    assert_eq!(dataset.shape().unwrap(), &[PAYLOAD as u64]);
    assert_eq!(dataset.chunk_shape().unwrap(), Some(vec![65_536]));
    assert_eq!(dataset.read_raw().unwrap(), data);
}

enum OutputDestination {
    Buffered,
    Streamed { file: StdFile, path: TempPath },
}

impl OutputDestination {
    fn finish(self, builder: FileBuilder) -> WrittenOutput {
        match self {
            Self::Buffered => WrittenOutput::Buffered(builder.finish().unwrap()),
            Self::Streamed { file, path } => {
                builder.finish_to(file).unwrap();
                WrittenOutput::Streamed(path)
            }
        }
    }
}

enum WrittenOutput {
    Buffered(Vec<u8>),
    Streamed(TempPath),
}

impl WrittenOutput {
    fn open(&self) -> File {
        match self {
            Self::Buffered(bytes) => File::from_bytes(bytes.clone()).unwrap(),
            Self::Streamed(path) => File::open(path).unwrap(),
        }
    }
}

fn assert_f64_bytes(bytes: &[u8], values: impl Iterator<Item = f64>) {
    let expected: Vec<_> = values.flat_map(f64::to_le_bytes).collect();
    assert_eq!(bytes, expected);
}

#[rstest]
#[case::small(32, "repack_tree_small", "repack_tree_small_control")]
#[case::large(256, "repack_tree_large", "repack_tree_large_control")]
fn repack_bounds_metadata_retention_and_preparation(
    #[case] branches: usize,
    #[case] region: &str,
    #[case] control: &str,
) {
    let directory = tempfile::tempdir().unwrap();
    let source_path = directory.path().join("source.h5");
    let output_path = directory.path().join("output.h5");
    write_repack_resource_tree(&source_path, branches);
    let (probe, positive) = allocation::measure(control, || vec![0_u8; 4_096]);
    assert!(positive.blocks > 0, "{positive}");
    assert_eq!(positive.peak_bytes, 4_096, "{positive}");
    drop(probe);
    let (result, measured) = allocation::measure(region, || {
        hdf5_pure::repack(&source_path, &output_path, &RepackOptions::new())
    });
    result.unwrap();
    let objects = u64::try_from(branches * 49 + 1).unwrap();
    assert!(measured.blocks >= objects, "{measured}");
    assert!(measured.bytes >= objects * 256, "{measured}");
    assert!(measured.peak_bytes >= objects * 16, "{measured}");
    assert!(measured.blocks < objects * 250, "{measured}");
    assert!(measured.bytes < objects * 24_000 + 65_536, "{measured}");
    assert!(measured.peak_bytes < objects * 1_400 + 65_536, "{measured}");
    assert_eq!(measured.live_bytes, 0, "{measured}");
    let source = File::open_streaming(&source_path).unwrap();
    let output = File::open_streaming(&output_path).unwrap();
    let mut groups = output.root().groups().unwrap();
    groups.sort_unstable();
    assert_eq!(
        groups,
        (0..branches)
            .map(|index| format!("tree{index:04}"))
            .collect::<Vec<_>>()
    );
    for branch in 0..branches {
        let root = format!("tree{branch:04}");
        assert_eq!(output.group(&root).unwrap().groups().unwrap().len(), 16);
        for index in 0..32 {
            let path = format!("{root}/group{:02}/data{index:02}", index % 16);
            let before = source.dataset(&path).unwrap();
            let after = output.dataset(&path).unwrap();
            let values = repack_resource_values(index);
            assert_eq!(after.read_i32().unwrap(), values);
            assert_eq!(
                after.shape().unwrap(),
                [u64::try_from(values.len()).unwrap()]
            );
            assert_eq!(after.maxshape().unwrap(), Some(vec![MaxExtent::Unlimited]));
            assert_eq!(after.chunk_shape().unwrap(), Some(vec![16]));
            assert_eq!(
                after.filter_pipeline().unwrap(),
                before.filter_pipeline().unwrap()
            );
            assert_eq!(after.attrs().unwrap(), before.attrs().unwrap());
            assert_eq!(after.chunks().unwrap().len(), values.len() / 16);
        }
    }
}

/// Writes repeated group trees with mostly empty resizable chunked datasets.
fn write_repack_resource_tree(path: &Path, branches: usize) {
    let mut builder = FileBuilder::new();
    for branch in 0..branches {
        let mut root = builder.create_group(&format!("tree{branch:04}"));
        for group in 0..16 {
            let mut child = root.create_group(&format!("group{group:02}"));
            for slot in 0..2 {
                let index = group + slot * 16;
                let dataset = child.create_dataset(&format!("data{index:02}"));
                dataset
                    .with_i32_data(&repack_resource_values(index))
                    .with_chunks(&[16])
                    .with_maxshape(&[MaxExtent::Unlimited])
                    .set_attr("index", AttrValue::I32(i32::try_from(index).unwrap()));
                if index.is_multiple_of(3) {
                    dataset.with_deflate(DeflateLevel::Six);
                }
            }
            root.add_group(child.finish());
        }
        builder.add_group(root.finish());
    }
    builder.write(path).unwrap();
}

/// Returns values for every eighth dataset and an empty vector for the others.
fn repack_resource_values(index: usize) -> Vec<i32> {
    if index.is_multiple_of(8) {
        (0..64).collect()
    } else {
        Vec::new()
    }
}
