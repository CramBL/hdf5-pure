use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use rstest::rstest;
use tempfile::TempDir;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::disk_log;
use test_util::disk_log::{IoEvent, IoOutcome};
use test_util::fixed_array;
use test_util::widths::Widths;

use crate::AppendBuilder;
use crate::ChunkIndex;
use crate::DatasetBuilder;
use crate::File;
use crate::FileBuilder;
use crate::FileLocking;
use crate::FileSpacePageSize;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::Layout;
use crate::MaxExtent;
use crate::PersistedFreeSpacePolicy;
use crate::SyncPolicy;
use crate::edit::WriteEngine;
use crate::edit::observations;
use crate::image::{CountingImage, HandleImage, WriteBuffering};
use crate::object_path::ObjectPathBuf;
use crate::source::{MetadataCacheConfig, Source};
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

#[cfg(test)]
test_util_hdf5::hdf5_1_10_tests! {
    mod native_crosschecks;
}

#[rstest]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::default(OffsetWidth::Eight, LengthWidth::Eight)]
fn fragmented_commits_bound_width_work_io_scratch_and_growth(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(FileSpaceStrategy::FsmAggr, FileSpaceStrategy::Page)] strategy: FileSpaceStrategy,
    #[values(false, true)] mixed: bool,
    #[values(4, 12)] holes: usize,
) {
    let fixture = run(FormatWidths { offsets, lengths }, strategy, mixed, holes);
    assert!(fixture.path.is_file());
    fixture.directory.close().unwrap();
}

/// Runs fragmented commit/reopen cycles and returns a closed fixture for native observations.
fn run(widths: FormatWidths, strategy: FileSpaceStrategy, mixed: bool, holes: usize) -> Fixture {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fragmented.h5");
    let settings = FileSpaceSettings::new(strategy)
        .with_persistence(true)
        .with_threshold(1)
        .with_page_size(FileSpacePageSize::try_from(512).unwrap());
    let mut builder = FileBuilder::new();
    builder
        .with_offset_width(widths.offsets)
        .with_length_width(widths.lengths)
        .with_file_space_settings(settings);
    builder
        .create_dataset("d")
        .with_i32_data(&[0; 16])
        .with_chunks(&[16])
        .with_maxshape(&[MaxExtent::Fixed(16 + 8 * u64::try_from(holes).unwrap() * 4)]);
    builder
        .create_group_with("g", |group| {
            for index in 0..holes {
                group
                    .create_dataset(&format!("hole_{index:02}"))
                    .with_i32_data(&vec![-1; hole_elements(index, mixed)]);
                group
                    .create_dataset(&format!("keep_{index:02}"))
                    .with_i32_data(&[i32::try_from(index).unwrap(); 16]);
            }
            Ok(())
        })
        .unwrap();
    builder.write(&path).unwrap();
    let reader = File::open(&path).unwrap();
    assert_eq!(
        reader.dataset("d").unwrap().layout().unwrap(),
        Layout::Chunked {
            chunk_shape: vec![16],
            index: ChunkIndex::FixedArray
        }
    );
    let dead = (0..holes)
        .map(|index| {
            match reader
                .dataset(&format!("g/hole_{index:02}"))
                .unwrap()
                .layout()
                .unwrap()
            {
                Layout::Contiguous {
                    address: Some(address),
                    size,
                } => (address, size),
                layout => panic!("expected contiguous hole storage, got {layout:?}"),
            }
        })
        .collect::<Vec<_>>();
    reader.close().unwrap();
    let reads = Arc::new(AtomicU64::new(0));
    let calls = Arc::new(AtomicU64::new(0));
    let peak = Arc::new(AtomicU64::new(0));
    let mut engine = open(&path, &reads, &calls, &peak);
    engine.space_ownership().unwrap().assert_exact();
    for index in 0..holes {
        engine
            .delete(&ObjectPathBuf::parse(&format!("g/hole_{index:02}")))
            .unwrap();
    }
    engine.commit().unwrap();
    engine.finalize_persist().unwrap();
    engine.force_sync().unwrap();
    let free = engine.reusable_free_sections().unwrap();
    assert!(free.section_count() >= holes);
    for &(start, length) in &dead {
        assert!(
            free.sections()
                .iter()
                .any(|section| section.address().get() <= start
                    && section.address().get() + section.length().get() >= start + length)
        );
    }
    engine.space_ownership().unwrap().assert_exact();
    drop(engine);

    for cycle in 0..8 {
        let mut engine = open(&path, &reads, &calls, &peak);
        let fragments = engine.space.reusable_sections().count();
        let batch = holes * 4;
        let values = (cycle * batch..(cycle + 1) * batch)
            .map(|value| i32::try_from(value).unwrap())
            .collect::<Vec<_>>();
        reads.store(0, Ordering::Relaxed);
        calls.store(0, Ordering::Relaxed);
        peak.store(0, Ordering::Relaxed);
        observations::reset_width_work();
        observations::reset_zeros();
        let before_eoa = engine.image.len();
        let io_before = engine.image.buffering_work();
        let region = format!(
            "width_{}_{}_{strategy:?}_{mixed}_{holes}_{cycle}_{}",
            widths.offsets.get(),
            widths.lengths.get(),
            directory.path().file_name().unwrap().to_str().unwrap()
        );
        commit(&mut engine, &values, &region, holes, cycle);
        let io = engine.image.buffering_work();
        let write_bytes = io.write_bytes - io_before.write_bytes;
        let write_calls = io.write_calls - io_before.write_calls;
        assert!(write_bytes > 0 && write_calls > 0);
        let work = observations::width_work();
        let zeros = observations::zero_work();
        // On a flat preferred hit, the writer plans the final index after independent preflight.
        let plan_bound = if strategy == FileSpaceStrategy::Page {
            2
        } else {
            3
        };
        assert!((2..=plan_bound).contains(&work.array_plans), "{work:?}");
        assert_eq!(
            work.array_encodings,
            if strategy == FileSpaceStrategy::Page {
                4
            } else {
                2
            },
            "{work:?}"
        );
        assert_eq!(work.array_bytes, index_bytes(widths, holes), "{work:?}");
        assert!(work.array_bytes > 0 && work.manager_bytes > 0, "{work:?}");
        assert!(
            work.manager_projections > 0 && work.manager_projections <= 8,
            "{work:?}"
        );
        assert!(
            work.manager_encodings > 0 && work.manager_encodings <= 8,
            "{work:?}"
        );
        assert!(
            work.manager_encodings <= work.projected_managers,
            "{work:?}"
        );
        assert!(
            work.projected_sections <= 8 * u64::try_from(fragments + 32).unwrap(),
            "{work:?}"
        );
        assert!(
            work.reservation_batches > 0 && work.reservation_batches <= 8,
            "{work:?}"
        );
        assert!(
            work.indexed_extents
                <= 8 * u64::try_from(fragments + usize::try_from(work.reservations).unwrap() + 8)
                    .unwrap(),
            "{work:?}"
        );
        assert!(work.lookups <= 3 * work.reservations, "{work:?}");
        assert!(
            work.reused_blocks > 0 && work.reused_block_bytes > 0,
            "{work:?}"
        );
        assert!(
            work.allocation_probes > 0
                && work.allocation_probes <= work.reservations + 4 * work.manager_projections,
            "{work:?}"
        );
        assert!(zeros.scratch_peak <= 65_536, "{zeros:?}");
        assert!(reads.load(Ordering::Relaxed) > 0);
        assert!(calls.load(Ordering::Relaxed) > 0);
        assert!(peak.load(Ordering::Relaxed) <= 4_096);
        let operation_reads = reads.load(Ordering::Relaxed);
        let operation_calls = calls.load(Ordering::Relaxed);
        let read_bound = u64::try_from(256 * (holes * 4 + cycle * holes + 32)).unwrap();
        assert!(reads.load(Ordering::Relaxed) < read_bound);
        assert!(
            calls.load(Ordering::Relaxed)
                < u64::try_from(16 * (holes * 4 + cycle * holes + 32)).unwrap()
        );
        assert!(write_bytes < read_bound && write_calls < work.reservations * 16 + 64);
        let eoa = engine.image.len();
        let growth = eoa.saturating_sub(before_eoa);
        let growth_budget = work.appended_capacity + zeros.bytes;
        assert!(growth <= growth_budget, "growth={growth}, {work:?}");
        disk_log::start();
        engine.commit().unwrap();
        engine.finalize_persist().unwrap();
        engine.force_sync().unwrap();
        let noop = disk_log::finish();
        assert!(
            noop.events
                .iter()
                .all(|event| !matches!(event, IoEvent::Write { written, .. } if *written != 0))
        );
        assert_eq!(engine.image.len(), eoa);
        let control_before = reads.load(Ordering::Relaxed);
        let maximum_read = peak.load(Ordering::Relaxed);
        let mut full = vec![0; usize::try_from(eoa).unwrap()];
        let control_chunks = 2 * u64::try_from(holes + 1 + (cycle + 1) * holes / 4).unwrap();
        for _ in 0..control_chunks {
            engine.image.read_at(0, &mut full).unwrap();
        }
        assert_eq!(
            reads.load(Ordering::Relaxed) - control_before,
            control_chunks * eoa
        );
        assert!(control_chunks * eoa > read_bound);
        assert_eq!(peak.load(Ordering::Relaxed), eoa.max(maximum_read));
        let before_control = engine.image.buffering_work();
        disk_log::start();
        engine.image.write_at(0, &full).unwrap();
        engine.image.flush().unwrap();
        let recording = disk_log::finish();
        let recorded = recording
            .events
            .iter()
            .filter_map(|event| match event {
                IoEvent::Write {
                    written,
                    outcome: IoOutcome::Succeeded,
                    ..
                } => Some(u64::try_from(*written).unwrap()),
                _ => None,
            })
            .sum::<u64>();
        assert_eq!(recorded, eoa);
        assert_eq!(
            engine.image.buffering_work().write_bytes - before_control.write_bytes,
            eoa
        );
        let extra = growth_budget - growth + 1;
        engine
            .image
            .append(&vec![0; usize::try_from(extra).unwrap()])
            .unwrap();
        assert!(engine.image.len().saturating_sub(before_eoa) > growth_budget);
        engine.image.truncate(eoa).unwrap();
        engine.force_sync().unwrap();
        engine.space_ownership().unwrap().assert_exact();
        let stats = engine.reusable_space_stats().unwrap();
        assert_eq!(
            stats.reusable_free_bytes(),
            engine.reusable_free_sections().unwrap().free_bytes()
        );
        assert_eq!(stats.logical_size(), eoa);
        drop(engine);
        let reader = File::open(&path).unwrap();
        assert_eq!(reader.superblock().offset_size, widths.offsets);
        assert_eq!(reader.superblock().length_size, widths.lengths);
        let replacement = reader.dataset(&format!("batch_{cycle}")).unwrap();
        assert_eq!(
            replacement.layout().unwrap(),
            Layout::Chunked {
                chunk_shape: vec![4],
                index: ChunkIndex::FixedArray
            }
        );
        assert_eq!(replacement.read_i32().unwrap(), values);
        drop(replacement);
        assert_eq!(
            reader.dataset("d").unwrap().read_i32().unwrap(),
            [0; 16]
                .into_iter()
                .chain(0..i32::try_from((cycle + 1) * batch).unwrap())
                .collect::<Vec<_>>()
        );
        for index in 0..holes {
            assert_eq!(
                reader
                    .dataset(&format!("g/keep_{index:02}"))
                    .unwrap()
                    .read_i32()
                    .unwrap(),
                [i32::try_from(index).unwrap(); 16]
            );
        }
        let mut names = reader.group("g").unwrap().datasets().unwrap();
        names.sort();
        assert_eq!(
            names,
            (0..holes)
                .map(|index| format!("keep_{index:02}"))
                .collect::<Vec<_>>()
        );
        reader.close().unwrap();
        eprintln!(
            "{widths:?} {strategy:?} mixed={mixed} holes={holes} cycle={cycle}: {work:?}, reads={}, calls={}, zero={zeros:?}, eoa={eoa}, writes={write_bytes}/{write_calls}",
            operation_reads, operation_calls
        );
    }
    Fixture { directory, path }
}

/// Measures staging, commit, persistence, and synchronization together.
///
/// Heap profiling is enabled on x86_64 and aarch64. Allocation controls are measured separately.
fn commit(engine: &mut WriteEngine, values: &[i32], region: &str, holes: usize, cycle: usize) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let control_region = format!("{region}_control");
        let ((), measured) =
            allocation::measure(region, || append_and_replace(engine, values, cycle));
        let bound = u64::try_from(512 * (holes * 4 + cycle * holes + 32)).unwrap();
        assert!(measured.peak_bytes < bound, "{measured}");
        assert!(
            measured.live_bytes <= u64::try_from(2_048 + 256 * holes).unwrap(),
            "{measured}"
        );
        let (control, positive) = allocation::measure(&control_region, || {
            std::hint::black_box(vec![0_u8; usize::try_from(bound).unwrap()])
        });
        assert!(
            positive.peak_bytes >= bound && positive.blocks > 0,
            "{positive}"
        );
        drop(control);
        eprintln!("{region}: {measured}");
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (region, holes);
        append_and_replace(engine, values, cycle);
    }
}

fn append_and_replace(engine: &mut WriteEngine, values: &[i32], cycle: usize) {
    if cycle != 0 {
        engine
            .delete(&ObjectPathBuf::parse(&format!("batch_{}", cycle - 1)))
            .unwrap();
    }
    let name = format!("batch_{cycle}");
    let mut builder = DatasetBuilder::new(&name);
    builder
        .with_i32_data(values)
        .with_chunks(&[4])
        .with_maxshape(&[MaxExtent::Fixed(u64::try_from(values.len() + 4).unwrap())]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse(&name), builder)
        .unwrap();
    let mut append = AppendBuilder::new();
    append.append_i32(values);
    engine
        .stage_dataset_append(&ObjectPathBuf::parse("d"), append)
        .unwrap();
    engine.commit().unwrap();
    engine.finalize_persist().unwrap();
    engine.force_sync().unwrap();
}

/// Opens a strict bounded test session with counted reads and disabled image metadata caching.
///
/// Writes are unbuffered, and the caller synchronizes explicitly.
fn open(
    path: &Path,
    reads: &Arc<AtomicU64>,
    calls: &Arc<AtomicU64>,
    peak: &Arc<AtomicU64>,
) -> WriteEngine {
    let mut engine = WriteEngine::open_imaged_with_policy(
        path,
        Some(FileLocking::Enabled),
        PersistedFreeSpacePolicy::Strict,
        |handle, len| {
            Ok(Box::new(
                CountingImage::new(
                    Box::new(HandleImage::new(
                        handle,
                        len,
                        MetadataCacheConfig::disabled(),
                    )),
                    reads.clone(),
                    Arc::default(),
                )
                .with_read_calls(calls.clone())
                .with_read_peak(peak.clone()),
            ))
        },
    )
    .unwrap();
    engine.bounded = true;
    engine.batched_appends = true;
    engine.set_sync_policy(SyncPolicy::OnClose);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine
}

/// Returns both arrays' combined encoded length from independent header and data-block fixtures.
fn index_bytes(widths: FormatWidths, holes: usize) -> u64 {
    let widths = Widths::new(
        usize::from(widths.offsets.get()),
        usize::from(widths.lengths.get()),
    );
    [holes + 1, 2 * holes + 1]
        .into_iter()
        .map(|entries| {
            let header = fixed_array::header(
                widths,
                fixed_array::Client::Plain,
                8,
                u64::try_from(entries).unwrap(),
                Some(64),
            );
            let data =
                fixed_array::empty_data_block(widths, fixed_array::Client::Plain, 8, entries, 32);
            u64::try_from(header.len() + data.len()).unwrap()
        })
        .sum()
}

fn hole_elements(index: usize, mixed: bool) -> usize {
    if mixed && index % 2 == 1 { 32 } else { 16 }
}

/// A closed file and the temporary directory owned by a workload.
struct Fixture {
    directory: TempDir,
    path: PathBuf,
}
