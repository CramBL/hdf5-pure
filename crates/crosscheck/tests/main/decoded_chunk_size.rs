#![cfg(feature = "__hdf5-1.10")]

use hdf5::dataset::ChunkOpts;
use hdf5_pure::File;
use rstest::rstest;
use test_util_hdf5::file;

#[rstest]
#[case::unfiltered(false, false)]
#[case::filtered_edges(true, false)]
#[case::raw_edges(true, true)]
fn complete_c_chunks_and_absent_storage_read_with_nonzero_fill(
    #[case] filtered: bool,
    #[case] raw_edges: bool,
    #[values(false, true)] streaming: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edges.h5");
    let values: Vec<i32> = (1..=10).collect();
    {
        let file = file::libhdf5_create_v110(&path);
        let builder = file
            .new_dataset::<i32>()
            .shape([10])
            .chunk([4])
            .fill_value(17);
        let builder = if filtered {
            builder.shuffle().deflate(6)
        } else {
            builder
        };
        let builder = if raw_edges {
            builder.chunk_opts(ChunkOpts::DONT_FILTER_PARTIAL_CHUNKS)
        } else {
            builder
        };
        let dataset = builder.create("d").unwrap();
        dataset.write_raw(&values).unwrap();
        assert_eq!(dataset.read_raw::<i32>().unwrap(), values);
        assert_eq!(dataset.num_chunks().unwrap(), 3);
        if raw_edges || !filtered {
            assert_eq!(dataset.chunk_info(2).unwrap().size, 16);
        }
        let absent = file
            .new_dataset::<i32>()
            .shape([10])
            .chunk([4])
            .fill_value(17)
            .create("absent")
            .unwrap();
        assert_eq!(absent.read_raw::<i32>().unwrap(), [17; 10]);
    }
    let file = if streaming {
        File::open_streaming(&path)
    } else {
        File::open(&path)
    }
    .unwrap();
    let dataset = file.dataset("d").unwrap();
    assert_eq!(dataset.read_i32().unwrap(), values);
    let before = dataset.chunk_cache_stats();
    assert_eq!(dataset.read_i32_rows(8, 2).unwrap(), values[8..]);
    assert_eq!(dataset.read_i32_region(&[8], &[2]).unwrap(), values[8..]);
    assert_eq!(dataset.read_i32().unwrap(), values);
    assert!(dataset.chunk_cache_stats().hits() > before.hits());
    if raw_edges || !filtered {
        assert_eq!(dataset.chunks().unwrap().last().unwrap().storage_size, 16);
    }
    let absent = file.dataset("absent").unwrap();
    assert_eq!(absent.read_i32().unwrap(), [17; 10]);
    assert_eq!(absent.read_i32_rows(8, 2).unwrap(), [17; 2]);
    assert_eq!(absent.chunks().unwrap(), []);
    assert_eq!(absent.chunk_cache_stats().cached_chunks(), 0);
}
