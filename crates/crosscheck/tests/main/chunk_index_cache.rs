#![cfg(feature = "__hdf5-1.10")]

use std::path::Path;

use hdf5::Extent;
use hdf5::dataset::AllocTime;
use hdf5_pure::ChunkCacheConfig;
use hdf5_pure::ChunkIndex;
use hdf5_pure::DatasetAccessProperties;
use ndarray::IxDyn;
use rstest::rstest;
use test_util_hdf5::file;
use test_util_hdf5::reader::ReadBackend;
use test_util_hdf5::region;

#[rstest]
fn reads_with_and_without_a_retained_index_match_libhdf5(
    #[values(
        ChunkIndex::BTreeV1,
        ChunkIndex::BTreeV2,
        ChunkIndex::ExtensibleArray,
        ChunkIndex::FixedArray,
        ChunkIndex::Implicit,
        ChunkIndex::SingleChunk
    )]
    index: ChunkIndex,
    #[values(ReadBackend::Buffered, ReadBackend::Streaming)] backend: ReadBackend,
    #[values(false, true)] index_cache: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("indexed.h5");
    libhdf5_create(&path, index);
    let whole = values();
    let window = hdf5::File::open(&path)
        .unwrap()
        .dataset("d")
        .unwrap()
        .read_slice::<u32, _, IxDyn>(region::selection(&WINDOW_START, &WINDOW_COUNT))
        .unwrap()
        .into_raw_vec_and_offset()
        .0;
    let file = backend.open(&path).unwrap();
    let dataset = file
        .dataset_with_options(
            "d",
            DatasetAccessProperties::new()
                .with_chunk_cache(ChunkCacheConfig::disabled().with_index_cache(index_cache)),
        )
        .unwrap();
    assert_eq!(dataset.chunk_index().unwrap(), Some(index));
    let start = WINDOW_START.map(|start| start as u64);
    let count = WINDOW_COUNT.map(|count| count as u64);
    let window_bytes: Vec<_> = window
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let whole_bytes: Vec<_> = whole.iter().flat_map(|value| value.to_ne_bytes()).collect();
    let read = || {
        assert_eq!(dataset.read_raw().unwrap(), whole_bytes);
        assert_eq!(dataset.read_u32().unwrap(), whole);
        assert_eq!(
            dataset.read_raw_region(&start, &count).unwrap(),
            window_bytes
        );
        assert_eq!(dataset.read_u32_region(&start, &count).unwrap(), window);
    };

    read();
    read();

    assert_eq!(dataset.chunk_cache_stats().index_loaded(), index_cache);
}

fn libhdf5_create(path: &Path, index: ChunkIndex) {
    let file = match index {
        ChunkIndex::BTreeV1 => file::libhdf5_create_earliest(path),
        _ => file::libhdf5_create_v110(path),
    };
    let builder = file.new_dataset::<u32>();
    let builder = match index {
        ChunkIndex::BTreeV2 => builder.chunk(CHUNK).shape(vec![
            Extent::new(SHAPE[0], None),
            Extent::new(SHAPE[1], None),
        ]),
        ChunkIndex::ExtensibleArray => builder.chunk(CHUNK).shape(vec![
            Extent::new(SHAPE[0], None),
            Extent::new(SHAPE[1], Some(SHAPE[1])),
        ]),
        ChunkIndex::Implicit => builder
            .chunk(CHUNK)
            .alloc_time(Some(AllocTime::Early))
            .shape(SHAPE),
        ChunkIndex::SingleChunk => builder.chunk(SHAPE).shape(SHAPE),
        _ => builder.chunk(CHUNK).shape(SHAPE),
    };
    let dataset = builder.create("d").unwrap();
    dataset.write_raw(&values()).unwrap();
    drop(dataset);
    file.close().unwrap();
}

fn values() -> Vec<u32> {
    (0..u32::try_from(SHAPE[0] * SHAPE[1]).unwrap()).collect()
}

const SHAPE: [usize; 2] = [10, 9];

const CHUNK: [usize; 2] = [3, 4];

const WINDOW_START: [usize; 2] = [2, 3];

const WINDOW_COUNT: [usize; 2] = [6, 6];
