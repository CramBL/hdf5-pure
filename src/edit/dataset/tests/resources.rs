//! Allocation and I/O bounds for complete creation operations and repeated edits.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_space::__private::ReservationWork;
use rstest::rstest;
use test_util::allocation;
use test_util_hdf5::file_space::chunk_holes;
use test_util_hdf5::file_space::persisted::PersistedStrategy;

use crate::File;
use crate::FileBuilder;
use crate::FileLocking;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::MemoryStrategy;
use crate::PersistedFreeSpacePolicy;
use crate::SyncPolicy;
use crate::dataspace::MaxExtent;
use crate::edit::WriteEngine;
use crate::edit::observations;
use crate::edit::observations::CreationWork;
use crate::free_space_manager::ManagerProjection;
use crate::image::CountingImage;
use crate::image::FileImage;
use crate::image::HandleImage;
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;
use crate::source::MetadataCacheConfig;
use crate::type_builders::DatasetBuilder;

#[rstest]
#[case::small_unbuffered(128, 0)]
#[case::large_unbuffered(512, 0)]
#[case::small_gathered(128, 8_192)]
#[case::large_gathered(512, 8_192)]
#[case::small_default(128, crate::edit::WRITE_GATHER_BYTES)]
#[case::large_default(512, crate::edit::WRITE_GATHER_BYTES)]
fn fragmented_creation_bounds_complete_operation_storage_and_work(
    #[case] count: usize,
    #[case] budget: usize,
    #[values(false, true)] paged: bool,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] backing: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("resources.h5");
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
    let read_bytes = Arc::new(AtomicU64::new(0));
    let read_calls = Arc::new(AtomicU64::new(0));
    let syncs = Arc::new(AtomicU64::new(0));
    let mut engine = WriteEngine::open_prepared(
        &path,
        Some(FileLocking::Enabled),
        PersistedFreeSpacePolicy::Tolerant,
        |handle, length| {
            let image: Box<dyn FileImage> = if backing == MemoryStrategy::Mirrored {
                Box::new(WriteEngine::read_mirror(handle)?)
            } else {
                Box::new(HandleImage::new(
                    handle,
                    length,
                    MetadataCacheConfig::disabled(),
                ))
            };
            Ok(Box::new(
                CountingImage::new(image, read_bytes.clone(), syncs.clone())
                    .with_read_calls(read_calls.clone()),
            ))
        },
    )
    .unwrap()
    .into_engine();
    engine.batched_appends = true;
    engine.bounded = backing == MemoryStrategy::Bounded;
    engine.sync_policy = SyncPolicy::OnClose;
    for index in 0..16 {
        engine
            .delete(&ObjectPathBuf::parse(&format!("scratch{index}")))
            .unwrap();
    }
    engine.commit().unwrap();
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
    let available = engine
        .space
        .reusable_sections()
        .map(|extent| (extent.start().get(), extent.len().get()))
        .collect::<Vec<_>>();
    assert!(available.iter().all(|&(_, length)| length < payload as u64));
    assert!(available.iter().map(|&(_, length)| length).sum::<u64>() >= payload as u64);
    let initial_size = engine.image.len();
    let before = engine.image.buffering_work();
    read_bytes.store(0, Ordering::Relaxed);
    read_calls.store(0, Ordering::Relaxed);
    syncs.store(0, Ordering::Relaxed);
    let expected = (0..payload)
        .map(|index| (index % 251) as u8)
        .collect::<Vec<_>>();
    observations::reset_creation();
    let ((), measured) = allocation::measure(
        &format!("fragmented_operation_{count}_{budget}_{paged}_{backing:?}"),
        || {
            let mut dataset = DatasetBuilder::new("replacement");
            dataset
                .with_u8_data(&expected)
                .with_chunks(&[16_384])
                .with_maxshape(&[MaxExtent::Unlimited]);
            engine
                .stage_created_dataset(&ObjectPathBuf::parse("replacement"), dataset)
                .unwrap();
            engine.commit().unwrap();
            engine.image.ordering_barrier().unwrap();
        },
    );
    let observed = observations::creation_work();
    let after = engine.image.buffering_work();
    let CreationWork {
        chunks,
        filtered_chunks,
        prepared_bytes,
        prepared_capacity,
        index_encodings,
        index_bytes,
        index_capacity,
        preferred_probes,
        raw_capacity_scans,
        scalar_requests,
        batch_builds,
        requests,
        reservations,
    } = observed;
    let ReservationWork {
        indexed_extents,
        lookups,
        removed_extents,
        inserted_extents,
        restored_extents,
    } = reservations;
    assert_eq!(chunks, count as u64);
    assert_eq!(filtered_chunks, 0);
    assert_eq!(prepared_bytes, payload as u64);
    assert_eq!(prepared_capacity, payload as u64);
    assert_eq!(index_encodings, 1);
    assert!(index_bytes > count as u64 * 8);
    assert!(index_capacity >= index_bytes);
    assert_eq!(preferred_probes, 1);
    assert_eq!(raw_capacity_scans, u64::from(paged));
    eprintln!("complete operation {observed:?}");
    assert_eq!(batch_builds, if paged { 2 } else { 1 });
    assert!(indexed_extents <= available.len() as u64 * if paged { 2 } else { 1 } + 32);
    assert!(lookups >= count as u64);
    assert!(lookups <= requests * if paged { 2 } else { 1 });
    assert!(removed_extents >= count as u64);
    assert!(removed_extents <= requests * 2);
    assert!(inserted_extents <= indexed_extents + requests * 2);
    assert!(restored_extents <= indexed_extents + requests);
    assert!(requests <= count as u64 + 32);
    assert!(scalar_requests <= 16);
    let metadata_budget =
        (count * 1_024 + available.len() * 2_048 + 262_144) as u64 + index_capacity * 4;
    let read_bytes = read_bytes.load(Ordering::Relaxed);
    let read_calls = read_calls.load(Ordering::Relaxed);
    let syncs = syncs.load(Ordering::Relaxed);
    assert!(read_bytes <= metadata_budget);
    assert!(read_calls <= available.len() as u64 * 16 + 128);
    assert!(syncs <= 2);
    let mirror_growth = after.mirror_capacity.saturating_sub(before.mirror_capacity);
    let bound =
        payload as u64 + prepared_capacity + metadata_budget + (2 * budget) as u64 + mirror_growth;
    assert!(
        measured.peak_bytes <= bound,
        "{measured}; bound={bound}, {observed:?}"
    );
    let (control, assembled) = allocation::measure(
        &format!("fragmented_assembly_control_{count}_{budget}_{paged}_{backing:?}"),
        || expected.clone(),
    );
    assert_eq!(control, expected);
    assert_eq!(assembled.live_bytes, payload as u64);
    if budget <= 8_192 || payload >= 8 * 1_024 * 1_024 {
        assert!(metadata_budget + (2 * budget) as u64 + mirror_growth < assembled.live_bytes / 2);
    }
    assert_eq!(after.pending_bytes, 0);
    assert!(after.flush_peak_capacity <= (budget * 2 + 65_536) as u64);
    assert!(after.copied_bytes - before.copied_bytes <= payload as u64 * 3 + metadata_budget);
    let writes = &engine.image.issued_write_order()[before.write_calls as usize..];
    assert_eq!(writes.iter().filter(|&&(start, _)| start == 0).count(), 1);
    assert!(writes.len() <= count + 64);
    if budget >= 16_384 {
        assert!(writes.len() <= payload.div_ceil(budget.min(hole_size)) + 32);
        assert!(after.copied_bytes - before.copied_bytes >= payload as u64);
    } else {
        assert!(writes.len() >= count);
        if budget == 0 {
            assert_eq!(after.copied_bytes, before.copied_bytes);
        }
    }
    assert!(writes.len() > 1);
    let manager = ManagerProjection::new(engine.space.admitted_snapshot(), engine.widths())
        .unwrap()
        .work();
    assert!(manager.sections <= available.len() as u64 + 64);
    assert!(manager.section_capacity <= (available.len() as u64 + 64) * 4);
    assert!(engine.space_ownership().unwrap().analysis_complete);
    assert!(engine.image.len().saturating_sub(initial_size) < metadata_budget + 65_536);
    eprintln!(
        "count={count} budget={budget} paged={paged} backend={backing:?} raw_input={payload} image_read_bytes={read_bytes} image_read_calls={read_calls} syncs={syncs} initial_mirror={before:?} final_mirror_and_io={after:?} observation={observed:?} manager={manager:?} growth={} heap={measured} positive_control={assembled}",
        engine.image.len().saturating_sub(initial_size)
    );
    drop(engine);
    let file = File::open(&path).unwrap();
    let created = file.dataset("replacement").unwrap();
    assert_eq!(created.read_u8().unwrap(), expected);
    let chunks = created.chunks().unwrap();
    assert_eq!(chunks.len(), count);
    for chunk in chunks {
        assert!(
            available
                .iter()
                .any(|&(start, length)| start <= chunk.address
                    && chunk.address + chunk.storage_size <= start + length)
        );
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

#[rstest]
#[case::empty(0, true)]
#[case::single(1, true)]
#[case::contiguous(8, true)]
#[case::append_only(8, false)]
fn creation_uses_scalar_reservations_for_small_and_contiguous_storage(
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
    let mut engine = super::edit_test_util::open_engine(&path);
    if reuse {
        engine.delete(&ObjectPathBuf::parse("scratch")).unwrap();
        engine.commit().unwrap();
    }
    let expected = vec![17; count * 1_024];
    observations::reset_creation();
    let ((), measured) = allocation::measure(&format!("cheap_creation_{count}_{reuse}"), || {
        let mut dataset = DatasetBuilder::new("d");
        dataset.with_u8_data(&expected).with_chunks(&[1_024]);
        engine
            .stage_created_dataset(&ObjectPathBuf::parse("d"), dataset)
            .unwrap();
        engine.commit().unwrap();
    });
    let work = observations::creation_work();
    let CreationWork {
        chunks,
        filtered_chunks: _,
        prepared_bytes,
        prepared_capacity,
        index_encodings,
        index_bytes: _,
        index_capacity: _,
        preferred_probes,
        raw_capacity_scans,
        scalar_requests: _,
        batch_builds,
        requests: _,
        reservations,
    } = work;
    assert_eq!(chunks, count as u64);
    assert_eq!(prepared_bytes, expected.len() as u64);
    assert_eq!(prepared_capacity, expected.len() as u64);
    assert_eq!(index_encodings, u64::from(count > 1));
    assert_eq!(preferred_probes, u64::from(reuse && count > 1));
    assert_eq!(raw_capacity_scans, 0);
    assert_eq!(batch_builds, 0);
    assert_eq!(reservations, ReservationWork::default());
    eprintln!("count={count} reuse={reuse} operation={work:?} heap={measured}");
    drop(engine);
    let reopened = File::open(&path).unwrap();
    assert_eq!(reopened.dataset("d").unwrap().read_u8().unwrap(), expected);
    assert_eq!(
        reopened.dataset("kept").unwrap().read_u8().unwrap(),
        [31; 1_024]
    );
    reopened.close().unwrap();
}

#[rstest]
fn fragmented_creation_cycles_bound_work_and_publish_once_per_commit(
    #[values(PersistedStrategy::FsmAggr, PersistedStrategy::Page)] strategy: PersistedStrategy,
) {
    use crate::AppendBuilder;
    use crate::edit::test_util;

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cycles.h5");
    let (file, _) = chunk_holes::prepare(&path, strategy);
    file.close().unwrap();
    let mut expected = (0..64).collect::<Vec<i32>>();
    for cycle in 0..3 {
        let mut engine = test_util::open_engine(&path);
        let available = engine.space.reusable_sections().count() as u64;
        let initial_size = engine.image.len();
        let before = engine.image.buffering_work();
        let mut created = DatasetBuilder::new("replacement");
        let values = (0..1_024)
            .map(|value| value + cycle * 2_048)
            .collect::<Vec<i32>>();
        created.with_i32_data(&values).with_chunks(&[64]);
        let mut append = AppendBuilder::new();
        let added = vec![cycle + 17; 64];
        append.append_i32(&added);
        observations::reset_creation();
        engine
            .stage_created_dataset(&ObjectPathBuf::parse("replacement"), created)
            .unwrap();
        engine
            .stage_dataset_append_pending(&ObjectPathBuf::parse("d"), append)
            .unwrap();
        engine.commit().unwrap();
        let work = observations::creation_work();
        assert_eq!(work.chunks, 17);
        assert_eq!(work.index_encodings, 2);
        assert!(work.batch_builds <= 3);
        assert!(work.reservations.indexed_extents <= available * 3 + 64);
        assert!(work.reservations.lookups <= work.requests * 2);
        assert!(
            work.reservations.restored_extents <= work.reservations.indexed_extents + work.requests
        );
        assert_eq!(
            engine.image.issued_write_order()[before.write_calls as usize..]
                .iter()
                .filter(|&&(start, _)| start == 0)
                .count(),
            1
        );
        assert!(engine.image.len().saturating_sub(initial_size) <= 32_768);
        assert!(engine.space_ownership().unwrap().analysis_complete);
        assert_eq!(
            engine.space.reusable_bytes().unwrap(),
            ByteLength::new(
                engine
                    .space
                    .reusable_sections()
                    .map(|extent| extent.len().get())
                    .sum::<u64>()
            )
        );
        eprintln!(
            "cycle={cycle} {strategy:?} work={work:?} buffering={:?} reusable={} growth={}",
            engine.image.buffering_work(),
            engine.space.reusable_bytes().unwrap().get(),
            engine.image.len().saturating_sub(initial_size)
        );
        let observed = File::from_bytes(engine.image.disk_bytes().unwrap()).unwrap();
        assert_eq!(
            observed.dataset("replacement").unwrap().read_i32().unwrap(),
            values
        );
        let mut grown = expected.clone();
        grown.extend_from_slice(&added);
        assert_eq!(observed.dataset("d").unwrap().read_i32().unwrap(), grown);
        observed.close().unwrap();
        engine.delete(&ObjectPathBuf::parse("replacement")).unwrap();
        let before_delete = engine.image.issued_write_order().len();
        engine.commit().unwrap();
        assert_eq!(
            engine.image.issued_write_order()[before_delete..]
                .iter()
                .filter(|&&(start, _)| start == 0)
                .count(),
            1
        );
        assert!(engine.space_ownership().unwrap().analysis_complete);
        drop(engine);
        expected.extend_from_slice(&added);
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
