#![cfg(feature = "__hdf5-1.10")]

use std::path::Path;

use hdf5::dataset::ChunkOpts;
use hdf5::filters::Filter as CFilter;
use hdf5_pure::{File, FilterDisposition, FilterId, RepackOptions};
use rstest::rstest;
use test_util_hdf5::chunk_filter;
use test_util_hdf5::dataset;
use test_util_hdf5::file;

#[rstest]
#[case::second_skipped(0b10, 0)]
#[case::uninterpreted_high_bit(0x8000_0002, 0x8000_0000)]
fn c_written_masks_keep_their_pipeline_positions_and_uninterpreted_bits(
    #[case] mask: u32,
    #[case] uninterpreted: u32,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("masked.h5");
    let chunks = [[1, 2, 3, 4], [5, 6, 7, 8]];
    let values = chunks.concat();
    chunk_filter::write_masked_chunks(&path, mask, &chunks);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), values);
    assert_masks(&path, mask, uninterpreted);

    dataset::pure_append_staged(&path, "d", &[9, 10, 11, 12]);
    assert_masks(&path, mask, uninterpreted);
    let expected: Vec<i32> = (1..=12).collect();
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), expected);

    let repacked = dir.path().join("repacked.h5");
    hdf5_pure::repack(&path, &repacked, &RepackOptions::new()).unwrap();
    assert_masks(&repacked, mask, uninterpreted);
    assert_eq!(dataset::read_libhdf5::<i32>(&repacked, "d"), expected);
}

#[test]
fn a_c_written_partial_edge_chunk_reports_raw_bypass_with_a_zero_mask() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("raw_edge.h5");
    let values: Vec<i32> = (1..=6).collect();
    {
        let file = file::libhdf5_create_v110(&path);
        let dataset = file
            .new_dataset::<i32>()
            .shape([6])
            .chunk([4])
            .set_filters(&[CFilter::Shuffle, CFilter::Deflate(6)])
            .chunk_opts(ChunkOpts::DONT_FILTER_PARTIAL_CHUNKS)
            .create("d")
            .unwrap();
        dataset.write_raw(&values).unwrap();
    }
    let file = File::open(&path).unwrap();
    let dataset = file.dataset("d").unwrap();
    let mut chunks = dataset.chunks().unwrap();
    chunks.sort_by_key(|chunk| chunk.offset.clone());
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].filter_mask().raw_bits(), 0);
    assert_eq!(chunks[1].filter_mask().raw_bits(), 0);
    assert!(!chunks[0].filters().bypasses_pipeline());
    assert!(chunks[1].filters().bypasses_pipeline());
    assert_eq!(
        chunks[0]
            .filters()
            .dispositions()
            .map(|(_, state)| state)
            .collect::<Vec<_>>(),
        [FilterDisposition::Applied; 2]
    );
    assert_eq!(
        chunks[1]
            .filters()
            .dispositions()
            .map(|(_, state)| state)
            .collect::<Vec<_>>(),
        [FilterDisposition::BypassedRaw; 2]
    );
    assert_eq!(dataset.read_i32().unwrap(), values);
}

fn assert_masks(path: &Path, mask: u32, uninterpreted: u32) {
    let file = File::open(path).unwrap();
    let dataset = file.dataset("d").unwrap();
    let mut chunks = dataset.chunks().unwrap();
    chunks.sort_by_key(|chunk| chunk.offset.clone());
    assert!(chunks.len() >= 2);
    for chunk in &chunks[..2] {
        let view = chunk.filters();
        assert_eq!(chunk.filter_mask().raw_bits(), mask);
        assert_eq!(view.uninterpreted_bits(), uninterpreted);
        assert!(!view.bypasses_pipeline());
        assert_eq!(
            view.dispositions()
                .map(|(filter, state)| (filter.id, state))
                .collect::<Vec<_>>(),
            [
                (FilterId::Shuffle, FilterDisposition::Applied),
                (FilterId::Deflate, FilterDisposition::Skipped)
            ]
        );
        assert_eq!(view.filter(2), None);
        assert_eq!(view.filter(32), None);
    }
}
