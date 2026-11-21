//! Allocation and I/O bounds for complete creation operations and repeated edits.

use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use rstest::rstest;
use test_util::allocation;
use test_util::disk_log::{self, IoEvent};
use test_util_hdf5::file_space::chunk_holes;
use test_util_hdf5::file_space::persisted::PersistedStrategy;

use crate::dataspace::MaxExtent;
use crate::edit::WriteEngine;
use crate::image::{CountingImage, FileImage, HandleImage, MirrorImage, WriteBuffering};
use crate::{
    ChunkIndex, File, FileAccessProperties, FileBuilder, FileLocking, FileSpaceSettings,
    FileSpaceStrategy, MemoryStrategy, MetadataCacheConfig, PersistedFreeSpacePolicy, SyncPolicy,
};

#[rstest]
#[case::small_unbuffered(128, 0)]
#[case::large_unbuffered(512, 0)]
#[case::small_gathered(128, 8_192)]
#[case::large_gathered(512, 8_192)]
#[case::small_default(128, crate::edit::WRITE_GATHER_BYTES)]
#[case::large_default(512, crate::edit::WRITE_GATHER_BYTES)]
fn fragmented_creation_bounds_complete_operation_storage_and_io(
    #[case] count: usize,
    #[case] budget: usize,
    #[values(false, true)] paged: bool,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] backing: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("resources.h5");
    let trace_path = directory.path().join("trace.h5");
    let payload = count * 16_384;
    let hole_size = payload / 8;
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(if paged {
            FileSpaceStrategy::Page
        } else {
            FileSpaceStrategy::FsmAggr
        })
        .with_persistence(true),
    );
    for index in 0..16 {
        builder
            .create_dataset(&format!("scratch{index}"))
            .with_u8_data(&vec![13; hole_size]);
        builder
            .create_dataset(&format!("spacer{index}"))
            .with_u8_data(&[31; 4_096]);
    }
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    for index in 0..16 {
        file.root().delete(&format!("scratch{index}")).unwrap();
    }
    file.commit().unwrap();
    file.close().unwrap();
    std::fs::copy(&path, &trace_path).unwrap();

    let expected = (0..payload)
        .map(|index| (index % 251) as u8)
        .collect::<Vec<_>>();
    let (file, reads) = open_counted(&path, budget, backing);
    let available = file.reusable_free_sections().unwrap();
    assert!(
        available
            .sections()
            .iter()
            .all(|section| section.length().get() < payload as u64)
    );
    assert!(available.free_bytes() >= payload as u64);
    let initial_size = file.file_size();
    let available_sections = available.section_count() as u64;
    let metadata_budget = count as u64 * 1_088 + available_sections * 2_048 + 262_144;
    reads.bytes.store(0, Ordering::Relaxed);
    reads.calls.store(0, Ordering::Relaxed);
    let region = format!("fragmented_operation_{count}_{budget}_{paged}_{backing:?}");
    let ((), measured) = allocation::measure(&region, || create_replacement(&file, &expected));
    let read_bytes = reads.bytes.load(Ordering::Relaxed);
    let read_calls = reads.calls.load(Ordering::Relaxed);
    assert!((1..=metadata_budget).contains(&read_bytes), "{read_bytes}");
    assert!(
        (1..=available_sections * 16 + 128).contains(&read_calls),
        "{read_calls}"
    );

    let bound = payload as u64 * 2 + metadata_budget + (2 * budget) as u64;
    assert!(measured.peak_bytes <= bound, "{measured}; bound={bound}");
    let (control, assembled) =
        allocation::measure(&format!("{region}_assembly_control"), || expected.clone());
    assert_eq!(control, expected);
    assert_eq!(assembled.live_bytes, payload as u64);
    if budget <= 8_192 || payload >= 8 * 1_024 * 1_024 {
        assert!(metadata_budget + ((2 * budget) as u64) < assembled.live_bytes / 2);
    }

    let growth = file.file_size().saturating_sub(initial_size);
    assert!(growth < metadata_budget + 65_536);
    file.close().unwrap();

    // Recording keeps copies of the writes. Use the same fixture in a separate run, outside
    // the allocation interval.
    let (file, _) = open_counted(&trace_path, budget, backing);
    assert_eq!(available, file.reusable_free_sections().unwrap());
    disk_log::start();
    create_replacement(&file, &expected);
    let recording = disk_log::finish();
    let writes = completed_writes(&recording.events);
    assert_eq!(writes.iter().filter(|&&(start, _)| start == 0).count(), 1);
    assert!((2..=count + 64).contains(&writes.len()), "{}", writes.len());
    if budget >= 16_384 {
        assert!(
            writes.len() <= payload.div_ceil(budget.min(hole_size)) + 32,
            "{}",
            writes.len()
        );
    } else {
        assert!(writes.len() >= count);
    }
    let written = writes.iter().map(|&(_, length)| length).sum::<u64>();
    assert!(
        (payload as u64..=payload as u64 + metadata_budget).contains(&written),
        "{written}"
    );
    assert!(
        recording
            .events
            .iter()
            .filter(|event| matches!(event, IoEvent::Synchronize { .. }))
            .count()
            <= 2
    );
    file.close().unwrap();

    let file = File::open(&path).unwrap();
    let persisted = file.persisted_free_sections().unwrap().unwrap();
    assert!(persisted.free_sections().section_count() as u64 <= available_sections + 64);
    assert!(persisted.manager_metadata_bytes() <= 4_096 + (available_sections + 64) * 128);
    let created = file.dataset("replacement").unwrap();
    assert_eq!(created.read_u8().unwrap(), expected);
    assert_eq!(
        created.chunk_index().unwrap(),
        Some(ChunkIndex::ExtensibleArray)
    );
    let chunks = created.chunks().unwrap();
    assert_eq!(chunks.len(), count);
    for chunk in chunks {
        assert_eq!(chunk.storage_size, 16_384);
        assert!(available.sections().iter().any(|section| {
            section.address().get() <= chunk.address
                && chunk.address + chunk.storage_size <= section.end_address().get()
        }));
    }
    for index in 0..16 {
        assert_eq!(
            file.dataset(&format!("spacer{index}"))
                .unwrap()
                .read_u8()
                .unwrap(),
            [31; 4_096]
        );
    }
    drop(created);
    file.close().unwrap();
}

fn create_replacement(file: &File, contents: &[u8]) {
    file.root()
        .create_dataset("replacement", |dataset| {
            dataset
                .with_u8_data(contents)
                .with_chunks(&[16_384])
                .with_maxshape(&[MaxExtent::Unlimited]);
        })
        .unwrap();
    file.commit().unwrap();
    file.sync().unwrap();
}

fn completed_writes(events: &[IoEvent]) -> Vec<(u64, u64)> {
    events
        .iter()
        .filter_map(|event| match event {
            IoEvent::Write {
                offset, written, ..
            } if *written != 0 => Some((*offset, u64::try_from(*written).unwrap())),
            _ => None,
        })
        .collect()
}

#[derive(Default)]
struct Reads {
    bytes: Arc<AtomicU64>,
    calls: Arc<AtomicU64>,
}

fn open_counted(path: &Path, budget: usize, backing: MemoryStrategy) -> (File, Reads) {
    let reads = Reads::default();
    let mut engine = WriteEngine::open_prepared(
        path,
        Some(FileLocking::Enabled),
        PersistedFreeSpacePolicy::Tolerant,
        |mut handle, length| {
            let image: Box<dyn FileImage> = if backing == MemoryStrategy::Mirrored {
                use std::io::{Seek, SeekFrom};

                handle.seek(SeekFrom::Start(0)).map_err(crate::Error::Io)?;
                // Supply mirror capacity at setup so the allocation interval measures creation
                // and buffering, independent of the mirror's growth policy.
                let mut data = Vec::with_capacity(usize::try_from(length).unwrap() + 1_048_576);
                handle.read_to_end(&mut data).map_err(crate::Error::Io)?;
                Box::new(MirrorImage::new(handle, data))
            } else {
                Box::new(HandleImage::new(
                    handle,
                    length,
                    MetadataCacheConfig::disabled(),
                ))
            };
            Ok(Box::new(
                CountingImage::new(image, reads.bytes.clone(), Arc::default())
                    .with_read_calls(reads.calls.clone()),
            ))
        },
    )
    .unwrap()
    .into_engine();
    engine.batched_appends = true;
    engine.bounded = backing == MemoryStrategy::Bounded;
    engine.sync_policy = SyncPolicy::OnClose;
    engine
        .set_write_buffering(if budget == 0 {
            WriteBuffering::Unbuffered
        } else {
            WriteBuffering::Operation {
                page_size: 4_096,
                max_bytes: budget,
            }
        })
        .unwrap();
    let file = File::from_test_edit_session(
        engine,
        FileAccessProperties::new()
            .with_locking(FileLocking::Enabled)
            .with_memory_strategy(backing)
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    (file, reads)
}

#[rstest]
#[case::empty(0, true)]
#[case::single(1, true)]
#[case::contiguous(8, true)]
#[case::append_only(8, false)]
fn small_creation_bounds_allocation_and_places_complete_chunks(
    #[case] count: usize,
    #[case] reuse: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cheap.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(if reuse {
            FileSpaceStrategy::FsmAggr
        } else {
            FileSpaceStrategy::None
        })
        .with_persistence(true),
    );
    builder
        .create_dataset("scratch")
        .with_u8_data(&vec![13; 65_536]);
    builder.create_dataset("kept").with_u8_data(&[31; 1_024]);
    builder.write(&path).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Bounded),
    )
    .unwrap();
    if reuse {
        file.root().delete("scratch").unwrap();
        file.commit().unwrap();
    }
    let free = file.reusable_free_sections().unwrap();
    let original_end = file.file_size();
    let expected = vec![17; count * 1_024];
    let region = format!("cheap_creation_{count}_{reuse}");
    let ((), measured) = allocation::measure(&region, || {
        file.root()
            .create_dataset("d", |dataset| {
                dataset.with_u8_data(&expected).with_chunks(&[1_024]);
            })
            .unwrap();
        file.commit().unwrap();
    });
    assert!(
        (1..=32_768 + expected.len() as u64 * 2).contains(&measured.peak_bytes),
        "{measured}"
    );
    file.close().unwrap();

    let file = File::open(&path).unwrap();
    let dataset = file.dataset("d").unwrap();
    assert_eq!(dataset.read_u8().unwrap(), expected);
    assert_eq!(
        dataset.chunk_index().unwrap(),
        Some(if count != 1 {
            ChunkIndex::FixedArray
        } else {
            ChunkIndex::SingleChunk
        })
    );
    let chunks = dataset.chunks().unwrap();
    assert_eq!(chunks.len(), count);
    if reuse {
        for adjacent in chunks.windows(2) {
            assert_eq!(
                adjacent[0].address + adjacent[0].storage_size,
                adjacent[1].address
            );
        }
    }
    for chunk in chunks {
        assert_eq!(chunk.storage_size, 1_024);
        if reuse {
            assert!(free.sections().iter().any(|section| {
                section.address().get() <= chunk.address
                    && chunk.address + chunk.storage_size <= section.end_address().get()
            }));
        } else {
            assert!(chunk.address >= original_end);
        }
    }
    assert_eq!(
        file.dataset("kept").unwrap().read_u8().unwrap(),
        [31; 1_024]
    );
    drop(dataset);
    file.close().unwrap();
}

#[rstest]
fn fragmented_creation_cycles_bound_growth_and_publish_once_per_commit(
    #[values(PersistedStrategy::FsmAggr, PersistedStrategy::Page)] strategy: PersistedStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cycles.h5");
    let (file, _) = chunk_holes::prepare(&path, strategy);
    file.close().unwrap();
    let mut expected = (0..64).collect::<Vec<i32>>();
    for cycle in 0..3 {
        let file = File::open_rw(&path).unwrap();
        let initial_size = file.file_size();
        let values = (0..1_024)
            .map(|value| value + cycle * 2_048)
            .collect::<Vec<i32>>();
        let added = vec![cycle + 17; 64];
        file.root()
            .create_dataset("replacement", |created| {
                created.with_i32_data(&values).with_chunks(&[64]);
            })
            .unwrap();
        file.dataset("d")
            .unwrap()
            .append_staged(|append| {
                append.append_i32(&added);
            })
            .unwrap();

        disk_log::start();
        file.commit().unwrap();
        let writes = completed_writes(&disk_log::finish().events);
        assert_eq!(writes.iter().filter(|&&(start, _)| start == 0).count(), 1);
        assert!(file.file_size().saturating_sub(initial_size) <= 32_768);
        assert_eq!(
            file.reusable_space_stats().unwrap().reusable_free_bytes(),
            file.reusable_free_sections().unwrap().free_bytes(),
        );
        let dataset = file.dataset("replacement").unwrap();
        assert_eq!(dataset.read_i32().unwrap(), values);
        assert_eq!(dataset.chunks().unwrap().len(), 16);
        drop(dataset);
        expected.extend_from_slice(&added);
        assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), expected);

        file.root().delete("replacement").unwrap();
        disk_log::start();
        file.commit().unwrap();
        let writes = completed_writes(&disk_log::finish().events);
        assert_eq!(writes.iter().filter(|&&(start, _)| start == 0).count(), 1);
        file.close().unwrap();

        let reopened = File::open(&path).unwrap();
        assert_eq!(reopened.dataset("d").unwrap().read_i32().unwrap(), expected);
        assert!(
            !reopened
                .root()
                .datasets()
                .unwrap()
                .iter()
                .any(|name| name == "replacement")
        );
        for index in 0..16 {
            assert_eq!(
                reopened
                    .dataset(&format!("spacer{index}"))
                    .unwrap()
                    .read_i32()
                    .unwrap(),
                (0..72).collect::<Vec<i32>>()
            );
        }
        reopened.close().unwrap();
    }
}
