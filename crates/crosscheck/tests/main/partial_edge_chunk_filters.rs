#![cfg(feature = "__hdf5-1.10")]
//! Reference-C-library interop for a chunked dataset whose partial edge chunks
//! are stored with the filter pipeline skipped.
//!
//! `ChunkOpts::DONT_FILTER_PARTIAL_CHUNKS` sets the layout flag through
//! `H5Pset_chunk_opts`, so the C library shuffles or deflates the whole chunks
//! of the dataset and stores each chunk that extends past the extent as it
//! stands. Both libraries read the values back.
//!
//! An append through hdf5-pure, to a dataset with the flag or without it,
//! stores each chunk it writes filtered or raw. An in-place append also stores
//! a chunk with every filter skipped where a reader at the extent before the
//! append and a reader at the extent after it decode the chunk differently.
//! Both libraries read the values and report the same chunk records, before and
//! after libhdf5 writes past the appended values.

#[cfg(target_endian = "little")]
use std::ops::Range;
use std::path::Path;

use hdf5::Extent;
use hdf5::dataset::ChunkOpts;
use hdf5::filters::Filter as CFilter;
use hdf5_pure::DeflateLevel;
use hdf5_pure::File;
use rstest::rstest;
use test_util_hdf5::dataset;
use test_util_hdf5::dataset::Unlimited;
use test_util_hdf5::file;
use test_util_hdf5::numeric::ByteOrder;

use crate::array_index;

#[derive(Clone, Copy, Debug)]
enum Filter {
    Deflate,
    Shuffle,
    ShuffleDeflate,
}

impl Filter {
    fn pipeline(self) -> Vec<CFilter> {
        match self {
            Self::Deflate => vec![CFilter::Deflate(DEFLATE_LEVEL)],
            Self::Shuffle => vec![CFilter::Shuffle],
            Self::ShuffleDeflate => vec![CFilter::Shuffle, CFilter::Deflate(DEFLATE_LEVEL)],
        }
    }

    fn test_util_pipeline(self) -> Vec<dataset::Filter> {
        let deflate = dataset::Filter::Deflate(DeflateLevel::try_from(DEFLATE_LEVEL).unwrap());
        match self {
            Self::Deflate => vec![deflate],
            Self::Shuffle => vec![dataset::Filter::Shuffle],
            Self::ShuffleDeflate => vec![dataset::Filter::Shuffle, deflate],
        }
    }

    #[cfg(target_endian = "little")]
    fn every_filter_skipped(self) -> u32 {
        match self {
            Self::Deflate | Self::Shuffle => 0b1,
            Self::ShuffleDeflate => 0b11,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Stored {
    Filtered,
    #[cfg(target_endian = "little")]
    FiltersSkipped,
    Raw,
}

fn c_create(path: &Path, filter: Filter, shape: &[usize], chunk: &[usize], data: &[i32]) {
    let file = file::libhdf5_create_v110(path);
    let builder = file.new_dataset::<i32>();
    let builder = match filter {
        Filter::Deflate => builder.deflate(DEFLATE_LEVEL),
        Filter::Shuffle => builder.shuffle(),
        Filter::ShuffleDeflate => builder.shuffle().deflate(DEFLATE_LEVEL),
    };
    let ds = builder
        .chunk(chunk.to_vec())
        .chunk_opts(ChunkOpts::DONT_FILTER_PARTIAL_CHUNKS)
        .shape(extents_the_c_library_accepts(shape, chunk))
        .create("d")
        .unwrap();
    ds.write_raw(data).unwrap();
    file.close().unwrap();
}

// H5D__chunk_construct (H5Dchunk.c, 2.2.0) rejects a chunk edge past a fixed dimension's maximum.
fn extents_the_c_library_accepts(shape: &[usize], chunk: &[usize]) -> Vec<Extent> {
    shape
        .iter()
        .zip(chunk)
        .map(|(&len, &edge)| {
            if edge > len {
                Extent::resizable(len)
            } else {
                Extent::fixed(len)
            }
        })
        .collect()
}

fn read_c_creation_properties(path: &Path) -> (Vec<CFilter>, Option<ChunkOpts>) {
    let ds = hdf5::File::open(path).unwrap().dataset("d").unwrap();
    (ds.filters(), ds.dcpl().unwrap().chunk_opts())
}

#[rstest]
#[case(Filter::Shuffle, &[10], &[4])]
#[case(Filter::Deflate, &[10], &[4])]
#[case(Filter::Shuffle, &[5, 5], &[2, 3])]
#[case(Filter::Deflate, &[5, 5], &[2, 3])]
#[case(Filter::Shuffle, &[3], &[4])]
#[case(Filter::Deflate, &[2, 2], &[4, 2])]
#[case(Filter::Shuffle, &[12], &[4])]
#[case(Filter::Deflate, &[6, 9], &[2, 3])]
#[case(Filter::Deflate, &[4, 5], &[2, 3])]
#[case(Filter::Shuffle, &[5, 6], &[2, 3])]
#[case(Filter::ShuffleDeflate, &[10], &[4])]
#[case(Filter::ShuffleDeflate, &[5, 5], &[2, 3])]
fn a_dataset_exempting_its_partial_chunks_from_the_filters_reads_back(
    #[case] filter: Filter,
    #[case] shape: &[usize],
    #[case] chunk: &[usize],
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dont_filter_partial_chunks.h5");
    let data: Vec<i32> = (0..shape.iter().product::<usize>())
        .map(|i| i as i32 * 0x0101_0101)
        .collect();
    c_create(&path, filter, shape, chunk, &data);

    assert_eq!(
        read_c_creation_properties(&path),
        (
            filter.pipeline(),
            Some(ChunkOpts::DONT_FILTER_PARTIAL_CHUNKS)
        )
    );
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), data);
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), data);
}

#[rstest]
#[cfg(target_endian = "little")]
#[case::raw_tail_stays_partial(true, Some(APPEND_FILL), 100..106, &[106], &[Stored::Raw])]
#[case::raw_tail_becomes_full(
    true,
    Some(APPEND_FILL),
    100..106,
    &[106, 107],
    &[Stored::FiltersSkipped],
)]
#[case::raw_tail_becomes_full_before_a_raw_tail(
    true,
    Some(APPEND_FILL),
    100..106,
    &[106, 107, 108],
    &[Stored::FiltersSkipped, Stored::Raw],
)]
#[case::filtered_tail_stays_partial(
    false,
    Some(APPEND_FILL),
    100..106,
    &[106],
    &[Stored::Filtered],
)]
#[case::filtered_tail_becomes_full(
    false,
    Some(APPEND_FILL),
    100..106,
    &[106, 107],
    &[Stored::Filtered],
)]
#[case::filtered_tail_becomes_full_before_a_filtered_tail(
    false,
    Some(APPEND_FILL),
    100..106,
    &[106, 107, 108],
    &[Stored::Filtered, Stored::Filtered],
)]
#[case::raw_tail_without_a_fill_value_stays_partial(true, None, 0..10, &[10], &[Stored::Raw])]
fn an_in_place_append_stores_each_written_chunk_as_both_libraries_read_it(
    #[case] raw_edges: bool,
    #[case] fill: Option<i32>,
    #[case] initial: Range<i32>,
    #[case] appended: &[i32],
    #[case] written: &[Stored],
    #[values(Filter::Deflate, Filter::Shuffle, Filter::ShuffleDeflate)] filter: Filter,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("partial_edge_append.h5");
    let initial: Vec<i32> = initial.collect();
    let unlimited =
        Unlimited::new("d", &initial, APPEND_CHUNK).filters(&filter.test_util_pipeline());
    let unlimited = match fill {
        Some(fill) => unlimited.fill_value(fill),
        None => unlimited,
    };
    let unlimited = if raw_edges {
        unlimited.partial_edge_chunks_unfiltered()
    } else {
        unlimited
    };
    unlimited.libhdf5_create(&path);
    let before = array_index::libhdf5_chunks(&path, 0);

    dataset::pure_append(&path, "d", appended);

    let mut values = [&initial[..], appended].concat();
    let chunk_opts = if raw_edges {
        ChunkOpts::DONT_FILTER_PARTIAL_CHUNKS
    } else {
        ChunkOpts::empty()
    };
    assert_eq!(
        read_c_creation_properties(&path),
        (filter.pipeline(), Some(chunk_opts))
    );
    assert_eq!(
        hdf5::File::open(&path)
            .unwrap()
            .dataset("d")
            .unwrap()
            .shape(),
        [values.len()]
    );
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), values);
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), values);
    let chunks = array_index::libhdf5_chunks(&path, 0);
    let kept = before.len() - 1;
    assert_eq!(chunks[..kept], before[..kept]);
    let written_records: Vec<_> = chunks[kept..]
        .iter()
        .zip(written)
        .map(|(chunk, stored)| match stored {
            Stored::Filtered => (chunk.offset().to_vec(), None, chunk.filter_mask()),
            Stored::FiltersSkipped | Stored::Raw => (
                chunk.offset().to_vec(),
                Some(chunk.size()),
                chunk.filter_mask(),
            ),
        })
        .collect();
    let expected: Vec<_> = written
        .iter()
        .zip(kept as u64..)
        .map(|(stored, index)| {
            let offset = vec![index * APPEND_CHUNK];
            match stored {
                Stored::Filtered => (offset, None, 0),
                Stored::FiltersSkipped => (offset, Some(16), filter.every_filter_skipped()),
                Stored::Raw => (offset, Some(16), 0),
            }
        })
        .collect();
    assert_eq!(
        (chunks.len(), written_records),
        (kept + written.len(), expected)
    );
    assert_eq!(
        array_index::pure_chunks(&File::open(&path).unwrap()),
        chunks
    );

    let file = hdf5::File::open_rw(&path).unwrap();
    let ds = file.dataset("d").unwrap();
    ds.resize([12]).unwrap();
    ds.write_slice(&[111], 11..12).unwrap();
    drop(ds);
    file.close().unwrap();

    values.resize(11, fill.unwrap_or(0));
    values.push(111);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), values);
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), values);
    assert_eq!(
        array_index::pure_chunks(&File::open(&path).unwrap()),
        array_index::libhdf5_chunks(&path, 0)
    );
}

#[rstest]
#[case::raw_tail_stays_partial(true, &[106], &[Stored::Raw])]
#[case::raw_tail_becomes_full(true, &[106, 107], &[Stored::Filtered])]
#[case::raw_tail_becomes_full_before_a_raw_tail(
    true,
    &[106, 107, 108],
    &[Stored::Filtered, Stored::Raw],
)]
#[case::filtered_tail_stays_partial(false, &[106], &[Stored::Filtered])]
#[case::filtered_tail_becomes_full(false, &[106, 107], &[Stored::Filtered])]
#[case::filtered_tail_becomes_full_before_a_filtered_tail(
    false,
    &[106, 107, 108],
    &[Stored::Filtered, Stored::Filtered],
)]
fn a_staged_append_stores_each_written_chunk_as_both_libraries_read_it(
    #[case] raw_edges: bool,
    #[case] appended: &[i32],
    #[case] written: &[Stored],
    #[values(Filter::Deflate, Filter::Shuffle, Filter::ShuffleDeflate)] filter: Filter,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("partial_edge_staged_append.h5");
    let initial: Vec<i32> = (100..106).collect();
    let unlimited = Unlimited::new("d", &initial, APPEND_CHUNK)
        .filters(&filter.test_util_pipeline())
        .fill_value(APPEND_FILL)
        .stored_as(ByteOrder::LittleEndian.i32_datatype());
    let (unlimited, chunk_opts) = if raw_edges {
        (
            unlimited.partial_edge_chunks_unfiltered(),
            ChunkOpts::DONT_FILTER_PARTIAL_CHUNKS,
        )
    } else {
        (unlimited, ChunkOpts::empty())
    };
    unlimited.libhdf5_create(&path);
    let before = array_index::libhdf5_chunks(&path, 0);

    dataset::pure_append_staged(&path, "d", appended);

    let mut values = [&initial[..], appended].concat();
    assert_eq!(
        read_c_creation_properties(&path),
        (filter.pipeline(), Some(chunk_opts))
    );
    assert_eq!(
        hdf5::File::open(&path)
            .unwrap()
            .dataset("d")
            .unwrap()
            .shape(),
        [values.len()]
    );
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), values);
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), values);
    let chunks = array_index::libhdf5_chunks(&path, 0);
    let written_records: Vec<_> = chunks[1..]
        .iter()
        .zip(written)
        .map(|(chunk, stored)| {
            let size = matches!(stored, Stored::Raw).then(|| chunk.size());
            (chunk.offset().to_vec(), size, chunk.filter_mask())
        })
        .collect();
    let expected: Vec<_> = written
        .iter()
        .zip(1..)
        .map(|(stored, index)| {
            let size = matches!(stored, Stored::Raw).then_some(16);
            (vec![index * APPEND_CHUNK], size, 0)
        })
        .collect();
    assert_eq!(
        (&chunks[..1], chunks.len(), written_records),
        (&before[..1], 1 + written.len(), expected)
    );
    assert_eq!(
        array_index::pure_chunks(&File::open(&path).unwrap()),
        chunks
    );

    let file = hdf5::File::open_rw(&path).unwrap();
    let ds = file.dataset("d").unwrap();
    ds.resize([12]).unwrap();
    ds.write_slice(&[111], 11..12).unwrap();
    drop(ds);
    file.close().unwrap();

    values.resize(11, APPEND_FILL);
    values.push(111);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), values);
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), values);
    assert_eq!(
        array_index::pure_chunks(&File::open(&path).unwrap()),
        array_index::libhdf5_chunks(&path, 0)
    );
}

const APPEND_CHUNK: u64 = 4;
const APPEND_FILL: i32 = 17;
const DEFLATE_LEVEL: u8 = 6;
