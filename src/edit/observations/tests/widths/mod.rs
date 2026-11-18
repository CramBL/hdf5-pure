use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use rstest::rstest;
use tempfile::TempDir;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::disk_log;
use test_util::disk_log::IoEvent;

use crate::ChunkIndex;
use crate::File;
use crate::FileAccessProperties;
use crate::FileBuilder;
use crate::FileLocking;
use crate::FileSpacePageSize;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::Layout;
use crate::MaxExtent;
use crate::MemoryStrategy;
use crate::PersistedFreeSpacePolicy;
use crate::SyncPolicy;
use crate::edit::WriteEngine;
use crate::image::{CountingImage, HandleImage, WriteBuffering};
use crate::source::MetadataCacheConfig;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

#[cfg(test)]
test_util_hdf5::hdf5_1_10_tests! {
    mod native_crosschecks;
}

#[rstest]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::default(OffsetWidth::Eight, LengthWidth::Eight)]
fn fragmented_commits_preserve_widths_and_bound_resource_use(
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

/// Runs fragmented commit and reopen cycles and returns a closed fixture for native checks.
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
    let file = File::open_rw_with_options(&path, properties()).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().layout().unwrap(),
        Layout::Chunked {
            chunk_shape: vec![16],
            index: ChunkIndex::FixedArray
        }
    );
    let dead = (0..holes)
        .map(|index| {
            match file
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
    let reads = Arc::new(AtomicU64::new(0));
    let calls = Arc::new(AtomicU64::new(0));
    let peak = Arc::new(AtomicU64::new(0));
    for index in 0..holes {
        file.group("g")
            .unwrap()
            .delete(&format!("hole_{index:02}"))
            .unwrap();
    }
    file.commit().unwrap();
    file.sync().unwrap();
    let free = file.reusable_free_sections().unwrap();
    assert!(free.section_count() >= holes);
    for &(start, length) in &dead {
        assert!(
            free.sections()
                .iter()
                .any(|section| section.address().get() <= start
                    && section.address().get() + section.length().get() >= start + length)
        );
    }
    file.close().unwrap();

    for cycle in 0..8 {
        let file = open_counted(&path, &reads, &calls, &peak);
        let before_eoa = file.file_size();
        let free = file.reusable_free_sections().unwrap();
        let batch = holes * 4;
        let values = (cycle * batch..(cycle + 1) * batch)
            .map(|value| i32::try_from(value).unwrap())
            .collect::<Vec<_>>();
        reads.store(0, Ordering::Relaxed);
        calls.store(0, Ordering::Relaxed);
        peak.store(0, Ordering::Relaxed);
        let region = format!(
            "width_{}_{}_{strategy:?}_{mixed}_{holes}_{cycle}_{}",
            widths.offsets.get(),
            widths.lengths.get(),
            directory.path().file_name().unwrap().to_str().unwrap()
        );
        let scale = u64::try_from(holes * 4 + cycle * holes + 32).unwrap();
        let (write_calls, write_bytes, reused) = measure_commit(&region, scale, holes, || {
            disk_log::start();
            append_and_replace(&file, &values, cycle);
            let mut write_calls = 0;
            let mut write_bytes = 0;
            let mut reused = false;
            for (offset, length) in completed_writes(disk_log::finish().events) {
                write_calls += 1;
                write_bytes += length;
                reused |= free.sections().iter().any(|section| {
                    section.address().get() <= offset
                        && offset + length <= section.end_address().get()
                });
            }
            (write_calls, write_bytes, reused)
        });
        let read_bytes = reads.load(Ordering::Relaxed);
        let read_calls = calls.load(Ordering::Relaxed);
        assert!(read_bytes > 0);
        assert!(read_calls > 0);
        assert!(peak.load(Ordering::Relaxed) <= 4_096);
        assert!(write_bytes > 0);
        assert!(write_calls > 0);
        assert!(reused, "no write to reusable space in cycle {cycle}");
        let eoa = file.file_size();
        assert!(eoa.saturating_sub(before_eoa) <= write_bytes);

        let control = file.dataset("d").unwrap().read_raw_rows(0, 16).unwrap();
        assert_eq!(control, [0; 64]);
        assert!(reads.load(Ordering::Relaxed) - read_bytes >= 64);

        disk_log::start();
        file.commit().unwrap();
        file.sync().unwrap();
        assert!(disk_log::take().is_empty());
        assert_eq!(file.file_size(), eoa);
        let stats = file.reusable_space_stats().unwrap();
        assert_eq!(
            stats.reusable_free_bytes(),
            file.reusable_free_sections().unwrap().free_bytes()
        );
        assert_eq!(stats.logical_size(), eoa);
        let reads_before_close = reads.load(Ordering::Relaxed);
        let calls_before_close = calls.load(Ordering::Relaxed);
        peak.store(0, Ordering::Relaxed);
        let (close_write_calls, close_write_bytes) = measure_close(&region, || {
            disk_log::start();
            file.close().unwrap();
            completed_writes(disk_log::finish().events)
                .fold((0, 0), |(calls, bytes), (_, length)| {
                    (calls + 1, bytes + length)
                })
        });
        let close_read_bytes = reads.load(Ordering::Relaxed) - reads_before_close;
        let close_read_calls = calls.load(Ordering::Relaxed) - calls_before_close;
        assert!(peak.load(Ordering::Relaxed) <= 4_096);
        assert!(
            read_bytes + close_read_bytes < 256 * scale,
            "read bytes in commit: {read_bytes}, close: {close_read_bytes}"
        );
        assert!(
            read_calls + close_read_calls < 16 * scale,
            "read calls in commit: {read_calls}, close: {close_read_calls}"
        );
        assert!(
            write_bytes + close_write_bytes < 256 * scale,
            "write bytes in commit: {write_bytes}, close: {close_write_bytes}"
        );
        assert!(
            write_calls + close_write_calls < 16 * scale,
            "write calls in commit: {write_calls}, close: {close_write_calls}"
        );
        let reader = File::open(&path).unwrap();
        assert_eq!(reader.file_size(), eoa);
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
    }
    Fixture { directory, path }
}

fn measure_commit<T>(region: &str, scale: u64, holes: usize, commit: impl FnOnce() -> T) -> T {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        // The recorder keeps copies of writes until the operation is checked.
        let bound = 512 * scale;
        let (result, measured) = allocation::measure(region, commit);
        assert!(measured.peak_bytes < bound, "{measured}");
        assert!(
            measured.live_bytes <= u64::try_from(2_048 + 256 * holes).unwrap(),
            "{measured}"
        );
        let (control, positive) = allocation::measure(&format!("{region}_control"), || {
            std::hint::black_box(vec![0_u8; usize::try_from(bound).unwrap()])
        });
        assert!(
            positive.peak_bytes >= bound && positive.blocks > 0,
            "{positive}"
        );
        drop(control);
        result
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = (region, scale, holes);
        commit()
    }
}

fn completed_writes(events: Vec<IoEvent>) -> impl Iterator<Item = (u64, u64)> {
    events.into_iter().filter_map(|event| match event {
        IoEvent::Write {
            offset, written, ..
        } if written > 0 => Some((offset, u64::try_from(written).unwrap())),
        _ => None,
    })
}

fn measure_close<T>(region: &str, close: impl FnOnce() -> T) -> T {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let (result, measured) = allocation::measure(&format!("{region}_close"), close);
        // The close peak includes the I/O recorder's event and byte buffers.
        assert!(measured.peak_bytes < 4_096, "{measured}");
        result
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = region;
        close()
    }
}

fn append_and_replace(file: &File, values: &[i32], cycle: usize) {
    if cycle != 0 {
        file.root().delete(&format!("batch_{}", cycle - 1)).unwrap();
    }
    file.root()
        .create_dataset(&format!("batch_{cycle}"), |builder| {
            builder
                .with_i32_data(values)
                .with_chunks(&[4])
                .with_maxshape(&[MaxExtent::Fixed(u64::try_from(values.len() + 4).unwrap())]);
        })
        .unwrap();
    file.dataset("d")
        .unwrap()
        .append_staged(|append| {
            append.append_i32(values);
        })
        .unwrap();
    file.commit().unwrap();
    file.sync().unwrap();
}

fn properties() -> FileAccessProperties {
    FileAccessProperties::new()
        .with_locking(FileLocking::Enabled)
        .with_memory_strategy(MemoryStrategy::Bounded)
        .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict)
        .with_sync_policy(SyncPolicy::OnClose)
}

fn open_counted(
    path: &Path,
    reads: &Arc<AtomicU64>,
    calls: &Arc<AtomicU64>,
    peak: &Arc<AtomicU64>,
) -> File {
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
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    File::from_test_edit_session(engine, properties()).unwrap()
}

fn hole_elements(index: usize, mixed: bool) -> usize {
    if mixed && index % 2 == 1 { 32 } else { 16 }
}

/// A closed file and the temporary directory owned by a workload.
struct Fixture {
    directory: TempDir,
    path: PathBuf,
}
