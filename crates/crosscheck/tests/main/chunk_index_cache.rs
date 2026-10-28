#![cfg(feature = "__hdf5-1.10")]

use std::path::Path;

use hdf5::Extent;
use hdf5::dataset::AllocTime;
use hdf5::file::LibraryVersion;
use hdf5_pure::ChunkCacheConfig;
use hdf5_pure::ChunkIndex;
use hdf5_pure::DatasetAccessProperties;
use hdf5_pure::{AttrValue, File, FileAccessProperties, MetadataCacheConfig, ReadSeekSource};
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

#[rstest]
#[case::streaming_zero(0, false)]
#[case::streaming_userblock(512, false)]
#[case::source_zero(0, true)]
#[case::source_userblock(512, true)]
fn shared_metadata_reads_preserve_native_values_and_userblock_coordinates(
    #[case] userblock: u64,
    #[case] custom_source: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("shared-metadata.h5");
    let expected: Vec<i32> = (0..96).collect();
    {
        let file = hdf5::File::with_options()
            .with_fapl(|properties| {
                properties.libver_bounds(LibraryVersion::V110, LibraryVersion::latest())
            })
            .with_fcpl(|properties| properties.userblock(userblock))
            .create(&path)
            .unwrap();
        let group = file.create_group("measurements").unwrap();
        let dataset = group
            .new_dataset::<i32>()
            .shape([96])
            .chunk([8])
            .create("values")
            .unwrap();
        dataset.write_raw(&expected).unwrap();
        dataset
            .new_attr::<i32>()
            .shape([2])
            .create("bounds")
            .unwrap()
            .write_raw(&[0, 95])
            .unwrap();
        assert_eq!(dataset.read_raw::<i32>().unwrap(), expected);
        drop(dataset);
        drop(group);
        file.close().unwrap();
    }
    let properties =
        FileAccessProperties::new().with_metadata_cache(MetadataCacheConfig::new(1_048_576));
    let file = if custom_source {
        File::from_source_with_options(
            ReadSeekSource::new(std::fs::File::open(&path).unwrap()).unwrap(),
            properties,
        )
    } else {
        File::open_streaming_with_options(&path, properties)
    }
    .unwrap();
    assert_eq!(file.superblock().base_address.get(), userblock);
    let dataset = file
        .dataset_with_options(
            "measurements/values",
            DatasetAccessProperties::new()
                .with_chunk_cache(ChunkCacheConfig::disabled().with_index_cache(false)),
        )
        .unwrap();
    assert_eq!(dataset.chunk_index().unwrap(), Some(ChunkIndex::FixedArray));
    assert_eq!(dataset.chunk_shape().unwrap(), Some(vec![8]));
    assert_eq!(dataset.shape().unwrap(), vec![96]);
    assert_eq!(
        dataset.attrs().unwrap().get("bounds"),
        Some(&AttrValue::I32Array(vec![0, 95]))
    );
    assert_eq!(dataset.read_i32().unwrap(), expected);
    assert_eq!(
        dataset.read_i32_region(&[11], &[17]).unwrap(),
        expected[11..28]
    );
    let populated = file.metadata_cache_stats().unwrap();
    assert!(populated.entries() > 0);
    assert!(populated.misses() > 0);
    assert_eq!(populated.evictions(), 0);
    file.reset_metadata_cache_stats();
    assert_eq!(dataset.read_i32().unwrap(), expected);
    assert_eq!(
        dataset.read_i32_region(&[11], &[17]).unwrap(),
        expected[11..28]
    );
    let warm = file.metadata_cache_stats().unwrap();
    assert!(warm.hits() > 0);
    assert_eq!(
        (warm.misses(), warm.evictions(), warm.invalidations()),
        (0, 0, 0)
    );
    assert_eq!(warm.bytes(), populated.bytes());
    assert!(!dataset.chunk_cache_stats().index_loaded());
    drop(dataset);
    file.close().unwrap();
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
