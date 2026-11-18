use std::path::Path;

use ::test_util::range;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_space::__private::{AllocationClass, FreeClass, PageType};
use test_util_hdf5::file_space::chunk_holes;
use test_util_hdf5::file_space::persisted::PersistedStrategy;

use crate::dataspace::MaxExtent;
use crate::edit::SpaceOwnership;
use crate::edit::reclaim::ObjectStorageKind;
use crate::edit::test_util;
use crate::object_graph::ObjectId;
use crate::{File, FileAccessProperties, PhysicalAddress, SyncPolicy};

/// Creates `replacement` in existing free extents and returns its element values.
///
/// Checks that the sixteen chunks occupy original free extents, each smaller than the complete
/// payload. Uses an unlimited dimension when `extensible` is `true`. Closes the file before
/// returning, retaining the seed dataset and spacers for survivor checks.
///
/// # Panics
///
/// Panics if fixture creation or I/O fails, or placement and value checks fail.
pub(in crate::edit) fn created_fixture(
    path: &Path,
    strategy: PersistedStrategy,
    extensible: bool,
) -> Vec<i32> {
    let (file, _) = chunk_holes::prepare(path, strategy);
    let free = file
        .reusable_free_sections()
        .unwrap()
        .sections()
        .iter()
        .map(|section| (section.address().get(), section.length().get()))
        .collect::<Vec<_>>();
    let expected = values();
    let dataset = file
        .root()
        .create_dataset("replacement", |builder| {
            builder.with_i32_data(&expected).with_chunks(&[64]);
            if extensible {
                builder.with_maxshape(&[MaxExtent::Unlimited]);
            }
        })
        .unwrap();
    file.commit().unwrap();
    let chunks = dataset.chunks().unwrap();
    assert_eq!(chunks.len(), 16);
    assert!(free.iter().all(|&(_, length)| length < 4_096));
    assert!(chunks.iter().all(
        |chunk| free.iter().any(|&(start, length)| start <= chunk.address
            && chunk.address + chunk.storage_size <= start + length)
    ));
    assert_eq!(dataset.read_i32().unwrap(), expected);
    drop(dataset);
    file.close().unwrap();
    expected
}

/// Checks allocation coverage after deletion and reuses retired payload.
///
/// Requires a fixture from [`created_fixture`], with `name` as the target dataset's last hard
/// link and `expected` as its values. Checks complete payload and index allocations below the
/// confirmed end of file against live allocations or admitted free space of the applicable class.
///
/// Creates a successor whose payload exceeds the largest eligible contiguous raw extent, checks
/// reuse of retired payload and the surviving datasets, and returns the successor's values.
///
/// # Panics
///
/// Panics if I/O fails or the fixture, ownership, placement, or value checks fail.
pub(in crate::edit) fn retire_and_reuse(path: &Path, name: &str, expected: &[i32]) -> Vec<i32> {
    let file = File::open(path).unwrap();
    let dataset = file.dataset(name).unwrap();
    assert_eq!(dataset.read_i32().unwrap(), expected);
    let header = dataset.header_address().unwrap();
    let payload = dataset
        .chunks()
        .unwrap()
        .into_iter()
        .map(|chunk| chunk.address..chunk.address + chunk.storage_size)
        .collect::<Vec<_>>();
    drop(dataset);
    file.close().unwrap();
    let engine = test_util::open_engine(path);
    let storage = engine
        .chunked_storage_spans(ObjectId::new(PhysicalAddress::new(header)))
        .unwrap();
    let index = engine
        .chunked_index_spans(ObjectId::new(PhysicalAddress::new(header)))
        .unwrap();
    assert!(index.len() >= 2);
    assert_eq!(storage.len(), payload.len() + index.len());
    for (allocation, class) in &storage {
        let raw = payload.iter().any(|range| {
            range.start == allocation.start().get() && range.end == allocation.end().get()
        });
        assert_eq!(
            *class,
            FreeClass::Page(if engine.space.is_paged() && !raw {
                PageType::Meta
            } else {
                PageType::Raw
            })
        );
    }
    drop(engine);
    let file = File::open_rw_with_options(
        path,
        FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    file.root().delete(name).unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    let engine = test_util::open_engine(path);
    let SpaceOwnership {
        physical_size,
        logical_size,
        userblock_space: _,
        live_raw_space,
        live_metadata_space,
        free_space_manager_space,
        reusable_free_space: _,
        unclassified_space: _,
        analysis_complete,
        trailing_slack_space: _,
    } = engine.space_ownership().unwrap();
    assert!(analysis_complete);
    assert_eq!(logical_size, physical_size);
    let free = engine
        .space
        .reusable_sections()
        .map(|extent| (extent.start().get(), extent.len().get()))
        .collect::<Vec<_>>();
    assert!(
        payload
            .iter()
            .any(|extent| range::covered_len(&free, extent) == extent.end - extent.start)
    );
    let count =
        usize::try_from(engine.space.largest(AllocationClass::Raw).get() / 256 + 1).unwrap() * 64;
    let raw_capacity = engine
        .space
        .admitted_snapshot()
        .sections()
        .filter(|(_, class)| !engine.space.is_paged() || *class == FreeClass::Page(PageType::Raw))
        .map(|(extent, _)| extent.len().get())
        .sum::<u64>();
    assert!(raw_capacity >= u64::try_from(count * 4).unwrap());
    let successor = expected
        .iter()
        .copied()
        .cycle()
        .take(count)
        .collect::<Vec<_>>();
    let live = live_raw_space
        .iter()
        .chain(&live_metadata_space)
        .chain(&free_space_manager_space)
        .copied()
        .collect::<Vec<_>>();
    let classified = engine
        .space
        .admitted_snapshot()
        .sections()
        .collect::<Vec<_>>();
    for (allocation, class) in storage {
        let below =
            allocation.start().get().min(physical_size)..allocation.end().get().min(physical_size);
        let reusable = classified
            .iter()
            .filter(|(_, current)| !engine.space.is_paged() || *current == class)
            .map(|(extent, _)| (extent.start().get(), extent.len().get()))
            .collect::<Vec<_>>();
        assert_eq!(
            range::covered_len(&live, &below) + range::covered_len(&reusable, &below),
            below.end - below.start,
            "{allocation:?}, {class:?}, {classified:?}, {live:?}"
        );
    }
    drop(engine);
    let file = File::open_rw_with_options(
        path,
        FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    let available = file
        .reusable_free_sections()
        .unwrap()
        .sections()
        .iter()
        .map(|section| section.address().get()..section.end_address().get())
        .collect::<Vec<_>>();
    assert!(
        available
            .iter()
            .map(|extent| extent.end - extent.start)
            .sum::<u64>()
            >= u64::try_from(successor.len() * 4).unwrap()
    );
    let next = file
        .root()
        .create_dataset("successor", |builder| {
            builder.with_i32_data(&successor).with_chunks(&[64]);
        })
        .unwrap();
    file.commit().unwrap();
    assert_eq!(next.read_i32().unwrap(), successor);
    let chunks = next.chunks().unwrap();
    assert!(chunks.iter().all(|chunk| {
        available.iter().any(|extent| {
            extent.start <= chunk.address && chunk.address + chunk.storage_size <= extent.end
        })
    }));
    assert!(
        chunks
            .iter()
            .any(|chunk| payload.iter().any(|extent| range::overlaps(
                extent,
                &(chunk.address..chunk.address + chunk.storage_size)
            )))
    );
    drop(next);
    file.close().unwrap();
    let file = File::open(path).unwrap();
    assert_eq!(
        file.dataset("successor").unwrap().read_i32().unwrap(),
        successor
    );
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        (0..64).collect::<Vec<i32>>()
    );
    for index in 0..16 {
        assert_eq!(
            file.dataset(&format!("spacer{index}"))
                .unwrap()
                .read_i32()
                .unwrap(),
            (0..72).collect::<Vec<i32>>()
        );
    }
    file.close().unwrap();
    successor
}

/// Returns supported complete physical allocations owned by the named dataset.
///
/// Uses the count-independent storage collector. Incoming-link counts and reference completeness
/// require separate proofs before retirement.
///
/// # Panics
///
/// Panics if the file or dataset cannot be read, or the storage collector cannot identify it as
/// a supported dataset.
pub(in crate::edit) fn owned_ranges(path: &Path, name: &str) -> Vec<FileRange> {
    let file = File::open(path).unwrap();
    let header = file.dataset(name).unwrap().header_address().unwrap();
    file.close().unwrap();
    let engine = test_util::open_engine(path);
    let mut storage = Vec::new();
    assert_eq!(
        engine.collect_object_storage(ObjectId::new(PhysicalAddress::new(header)), &mut storage),
        Some(ObjectStorageKind::Dataset)
    );
    storage.into_iter().map(|(range, _)| range).collect()
}

pub(in crate::edit::dataset) fn values() -> Vec<i32> {
    (0..1_024).collect()
}
