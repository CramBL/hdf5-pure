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
//!
//! A staged overwrite through hdf5-pure, to a dataset with the flag or without
//! it, stores each partial edge chunk unfiltered under the flag and every other
//! chunk filtered, and so does an overwrite of variable-length strings. A second
//! overwrite, whose chunks fit their slots, keeps every chunk at its address. A
//! copy from another file and a repack keep each chunk as the source stores it.
//! Both libraries read the values and report the same chunk records, before and
//! after libhdf5 writes the last element.

#[cfg(target_endian = "little")]
use std::ops::Range;
use std::path::Path;

use hdf5::Extent;
use hdf5::Hyperslab;
use hdf5::SliceOrIndex;
use hdf5::dataset::ChunkOpts;
use hdf5::filters::Filter as CFilter;
#[cfg(target_endian = "little")]
use hdf5::types::VarLenUnicode;
use hdf5_pure::DeflateLevel;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::RepackOptions;
use ndarray::ArrayD;
use ndarray::IxDyn;
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
    ChunkedFixture {
        shape,
        chunk,
        filter,
        raw_edges: true,
        fill: None,
    }
    .libhdf5_create(&path, &data);

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

#[rstest]
#[case::rank_1_partial_tail(&[10], &[4], Some(OVERWRITE_FILL))]
#[case::rank_1_partial_tail_without_a_fill_value(&[10], &[4], None)]
#[case::chunk_past_the_whole_extent(&[3], &[4], Some(OVERWRITE_FILL))]
#[case::partial_along_both_axes(&[5, 5], &[2, 3], Some(OVERWRITE_FILL))]
#[case::partial_along_the_inner_axis(&[4, 5], &[2, 3], Some(OVERWRITE_FILL))]
#[case::partial_along_the_outer_axis(&[5, 6], &[2, 3], Some(OVERWRITE_FILL))]
#[case::partial_along_the_first_and_last_of_three_axes(
    &[3, 4, 5],
    &[2, 4, 3],
    Some(OVERWRITE_FILL),
)]
fn a_staged_overwrite_stores_each_chunk_as_both_libraries_read_it(
    #[case] shape: &[usize],
    #[case] chunk: &[usize],
    #[case] fill: Option<i32>,
    #[values(false, true)] raw_edges: bool,
    #[values(Filter::Deflate, Filter::Shuffle, Filter::ShuffleDeflate)] filter: Filter,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("partial_edge_staged_overwrite.h5");
    let fixture = ChunkedFixture {
        shape,
        chunk,
        filter,
        raw_edges,
        fill,
    };
    let len = shape.iter().product();
    fixture.libhdf5_create(&path, &(0..len).map(|i| i as i32).collect::<Vec<_>>());

    let mut values = dataset::incompressible(7, len);
    fixture.pure_overwrite(&path, &values);
    fixture.assert_stored(&path, &values);

    let placed = chunk_addresses(&path);
    values = (0..len).map(|i| 1_000 + i as i32 / 8).collect();
    fixture.pure_overwrite(&path, &values);
    fixture.assert_stored(&path, &values);
    assert_eq!(chunk_addresses(&path), placed);

    fixture.libhdf5_write_last(&path, 999);

    values[len - 1] = 999;
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), values);
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), values);
    assert_eq!(
        array_index::pure_chunks(&File::open(&path).unwrap()),
        array_index::libhdf5_chunks(&path, 0)
    );
}

#[rstest]
#[case::rank_1_partial_tail(&[10], &[4])]
#[case::partial_along_both_axes(&[5, 5], &[2, 3])]
fn a_copy_keeps_each_chunk_as_both_libraries_read_it(
    #[case] shape: &[usize],
    #[case] chunk: &[usize],
    #[values(false, true)] raw_edges: bool,
    #[values(Filter::Deflate, Filter::Shuffle, Filter::ShuffleDeflate)] filter: Filter,
    #[values(Duplicate::CrossFileCopy, Duplicate::Repack)] duplicate: Duplicate,
) {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("partial_edge_source.h5");
    let copied = dir.path().join("partial_edge_copy.h5");
    let fixture = ChunkedFixture {
        shape,
        chunk,
        filter,
        raw_edges,
        fill: Some(OVERWRITE_FILL),
    };
    let len = shape.iter().product();
    let mut values: Vec<i32> = (0..len).map(|i| i as i32).collect();
    fixture.libhdf5_create(&source, &values);

    duplicate.write(&source, &copied);

    assert_eq!(
        chunk_records(&fixture.assert_read_back(&copied, &values)),
        chunk_records(&array_index::libhdf5_chunks(&source, 0))
    );

    fixture.libhdf5_write_last(&copied, 999);

    values[len - 1] = 999;
    fixture.assert_read_back(&copied, &values);
}

#[derive(Clone, Copy, Debug)]
enum Duplicate {
    CrossFileCopy,
    Repack,
}

impl Duplicate {
    fn write(self, source: &Path, copied: &Path) {
        match self {
            Self::CrossFileCopy => {
                FileBuilder::new().write(copied).unwrap();
                let file = File::open_rw(copied).unwrap();
                file.copy_from(&File::open(source).unwrap(), "d", "d")
                    .unwrap();
                file.commit().unwrap();
            }
            Self::Repack => hdf5_pure::repack(source, copied, &RepackOptions::new()).unwrap(),
        }
    }
}

fn chunk_records(chunks: &[array_index::Chunk]) -> Vec<(Vec<u64>, u64, u32)> {
    chunks
        .iter()
        .map(|chunk| (chunk.offset().to_vec(), chunk.size(), chunk.filter_mask()))
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct ChunkedFixture<'a> {
    shape: &'a [usize],
    chunk: &'a [usize],
    filter: Filter,
    /// Whether the dataset sets `ChunkOpts::DONT_FILTER_PARTIAL_CHUNKS`.
    raw_edges: bool,
    fill: Option<i32>,
}

impl ChunkedFixture<'_> {
    fn libhdf5_create(self, path: &Path, data: &[i32]) {
        let file = file::libhdf5_create_v110(path);
        let builder = file
            .new_dataset_builder()
            .empty_as(ByteOrder::LittleEndian.i32_datatype());
        let builder = match self.filter {
            Filter::Deflate => builder.deflate(DEFLATE_LEVEL),
            Filter::Shuffle => builder.shuffle(),
            Filter::ShuffleDeflate => builder.shuffle().deflate(DEFLATE_LEVEL),
        };
        let builder = match self.fill {
            Some(fill) => builder.fill_value(fill),
            None => builder,
        };
        let builder = if self.raw_edges {
            builder.chunk_opts(ChunkOpts::DONT_FILTER_PARTIAL_CHUNKS)
        } else {
            builder
        };
        builder
            .chunk(self.chunk.to_vec())
            .shape(extents_the_c_library_accepts(self.shape, self.chunk))
            .create("d")
            .unwrap()
            .write_raw(data)
            .unwrap();
        file.close().unwrap();
    }

    fn pure_overwrite(self, path: &Path, values: &[i32]) {
        let shape: Vec<u64> = self.shape.iter().map(|&len| len as u64).collect();
        let file = File::open_rw(path).unwrap();
        file.dataset("d")
            .unwrap()
            .write_staged(|builder| {
                builder.with_i32_data(values).with_shape(&shape);
            })
            .unwrap();
        file.commit().unwrap();
    }

    fn libhdf5_write_last(self, path: &Path, value: i32) {
        let file = hdf5::File::open_rw(path).unwrap();
        file.dataset("d")
            .unwrap()
            .write_slice(
                &ArrayD::from_elem(IxDyn(&vec![1; self.shape.len()]), value),
                Hyperslab::from(
                    self.shape
                        .iter()
                        .map(|&len| SliceOrIndex::from(len - 1..len))
                        .collect::<Vec<_>>(),
                ),
            )
            .unwrap();
        file.close().unwrap();
    }

    /// Asserts what [`assert_read_back`](Self::assert_read_back) asserts, a filter mask of 0 for
    /// every chunk, and under `raw_edges` that each partial edge chunk stores its whole logical
    /// chunk unfiltered.
    fn assert_stored(self, path: &Path, values: &[i32]) {
        let chunks = self.assert_read_back(path, values);
        let image = std::fs::read(path).unwrap();
        let chunk_bytes = (self.chunk.iter().product::<usize>() * size_of::<i32>()) as u64;
        let stored: Vec<_> = chunks
            .iter()
            .map(|chunk| {
                let raw = self.raw_edges && self.is_partial(chunk.offset());
                let elements = raw.then(|| {
                    let start = usize::try_from(chunk.address()).unwrap();
                    let end = start + usize::try_from(chunk.size()).unwrap();
                    image[start..end]
                        .as_chunks()
                        .0
                        .iter()
                        .map(|&bytes| i32::from_le_bytes(bytes))
                        .collect::<Vec<_>>()
                });
                let size = raw.then(|| chunk.size());
                (chunk.offset().to_vec(), size, chunk.filter_mask(), elements)
            })
            .collect();
        let expected: Vec<_> = chunks
            .iter()
            .map(|chunk| {
                let raw = self.raw_edges && self.is_partial(chunk.offset());
                (
                    chunk.offset().to_vec(),
                    raw.then_some(chunk_bytes),
                    0,
                    raw.then(|| self.logical_chunk(chunk.offset(), values)),
                )
            })
            .collect();
        assert_eq!(stored, expected);
    }

    /// Asserts that libhdf5 reports the fixture's filters, chunk options and shape, and that both
    /// libraries read `values` and the same chunk records, which it returns.
    fn assert_read_back(self, path: &Path, values: &[i32]) -> Vec<array_index::Chunk> {
        let chunk_opts = if self.raw_edges {
            ChunkOpts::DONT_FILTER_PARTIAL_CHUNKS
        } else {
            ChunkOpts::empty()
        };
        assert_eq!(
            read_c_creation_properties(path),
            (self.filter.pipeline(), Some(chunk_opts))
        );
        assert_eq!(
            hdf5::File::open(path)
                .unwrap()
                .dataset("d")
                .unwrap()
                .shape(),
            self.shape
        );
        assert_eq!(dataset::read_libhdf5::<i32>(path, "d"), values);
        assert_eq!(dataset::read_pure::<i32>(path, "d"), values);
        let chunks = array_index::libhdf5_chunks(path, 0);
        assert_eq!(array_index::pure_chunks(&File::open(path).unwrap()), chunks);
        chunks
    }

    fn is_partial(self, offset: &[u64]) -> bool {
        offset
            .iter()
            .zip(self.chunk)
            .zip(self.shape)
            .any(|((&offset, &edge), &len)| offset as usize + edge > len)
    }

    /// Returns the elements of the whole chunk at `offset`, with the fill value, or 0 without one,
    /// past the extent.
    fn logical_chunk(self, offset: &[u64], values: &[i32]) -> Vec<i32> {
        (0..self.chunk.iter().product::<usize>())
            .map(|local| {
                let mut remaining = local;
                let mut element = Some(0);
                let mut stride = 1;
                for ((&offset, &edge), &len) in offset.iter().zip(self.chunk).zip(self.shape).rev()
                {
                    let coord = offset as usize + remaining % edge;
                    remaining /= edge;
                    element = element
                        .filter(|_| coord < len)
                        .map(|element| element + coord * stride);
                    stride *= len;
                }
                element.map_or(self.fill.unwrap_or(0), |element| values[element])
            })
            .collect()
    }
}

fn chunk_addresses(path: &Path) -> Vec<(Vec<u64>, u64)> {
    array_index::libhdf5_chunks(path, 0)
        .iter()
        .map(|chunk| (chunk.offset().to_vec(), chunk.address()))
        .collect()
}

#[rstest]
#[cfg(target_endian = "little")]
fn a_staged_overwrite_of_strings_stores_each_chunk_as_both_libraries_read_it(
    #[values(false, true)] raw_edges: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("partial_edge_staged_string_overwrite.h5");
    let chunk_opts = if raw_edges {
        ChunkOpts::DONT_FILTER_PARTIAL_CHUNKS
    } else {
        ChunkOpts::empty()
    };
    let file = file::libhdf5_create_v110(&path);
    file.new_dataset::<VarLenUnicode>()
        .deflate(DEFLATE_LEVEL)
        .chunk([4])
        .chunk_opts(chunk_opts)
        .shape([10])
        .create("d")
        .unwrap()
        .write(&var_len_strings((0..10).map(|i| format!("initial-{i}"))))
        .unwrap();
    file.close().unwrap();

    let mut strings: Vec<String> = (0..10).map(|i| format!("overwritten-{i}")).collect();
    {
        let file = File::open_rw(&path).unwrap();
        file.dataset("d")
            .unwrap()
            .write_staged(|builder| {
                builder.with_vlen_strings(&strings.iter().map(String::as_str).collect::<Vec<_>>());
            })
            .unwrap();
        file.commit().unwrap();
    }

    assert_eq!(
        read_c_creation_properties(&path),
        (Filter::Deflate.pipeline(), Some(chunk_opts))
    );
    assert_eq!(dataset::read_libhdf5_strings(&path, "d"), strings);
    assert_eq!(dataset::read_pure_strings(&path, "d"), strings);
    let chunks = array_index::libhdf5_chunks(&path, 0);
    assert_eq!(
        array_index::pure_chunks(&File::open(&path).unwrap()),
        chunks
    );
    let stored: Vec<_> = chunks
        .iter()
        .map(|chunk| {
            let raw = raw_edges && chunk.offset() == [8];
            (
                chunk.offset().to_vec(),
                raw.then(|| chunk.size()),
                chunk.filter_mask(),
            )
        })
        .collect();
    let mut expected = vec![(vec![0], None, 0), (vec![4], None, 0), (vec![8], None, 0)];
    if raw_edges {
        expected[2].1 = Some(4 * VLEN_REFERENCE_SIZE);
    }
    assert_eq!(stored, expected);

    let file = hdf5::File::open_rw(&path).unwrap();
    file.dataset("d")
        .unwrap()
        .write_slice(&var_len_strings(["rewritten-9".to_owned()]), 9..10)
        .unwrap();
    file.close().unwrap();

    strings[9] = "rewritten-9".to_owned();
    assert_eq!(dataset::read_libhdf5_strings(&path, "d"), strings);
    assert_eq!(dataset::read_pure_strings(&path, "d"), strings);
    assert_eq!(
        array_index::pure_chunks(&File::open(&path).unwrap()),
        array_index::libhdf5_chunks(&path, 0)
    );
}

#[cfg(target_endian = "little")]
fn var_len_strings(strings: impl IntoIterator<Item = String>) -> Vec<VarLenUnicode> {
    strings
        .into_iter()
        .map(|string| string.parse().unwrap())
        .collect()
}

const APPEND_CHUNK: u64 = 4;
const APPEND_FILL: i32 = 17;
const DEFLATE_LEVEL: u8 = 6;
const OVERWRITE_FILL: i32 = -3;
// A variable-length string element is stored as a sequence length, a global heap collection
// address and an object index.
#[cfg(target_endian = "little")]
const VLEN_REFERENCE_SIZE: u64 = 16;
