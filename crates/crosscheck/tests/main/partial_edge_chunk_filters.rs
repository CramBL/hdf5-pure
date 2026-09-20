#![cfg(feature = "__hdf5-1.10")]
//! Reference-C-library interop for a chunked dataset whose partial edge chunks
//! are stored with the filter pipeline skipped.
//!
//! `ChunkOpts::DONT_FILTER_PARTIAL_CHUNKS` sets the layout flag through
//! `H5Pset_chunk_opts`, so the C library shuffles or deflates the whole chunks
//! of the dataset and stores each chunk that extends past the extent as it
//! stands. Both libraries read the values back.

use std::path::Path;

use hdf5::Extent;
use hdf5::dataset::ChunkOpts;
use hdf5::file::LibraryVersion;
use hdf5::filters::Filter as CFilter;
use rstest::rstest;

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
}

fn c_create(path: &Path, filter: Filter, shape: &[usize], chunk: &[usize], data: &[i32]) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
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

fn read_c(path: &Path) -> Vec<i32> {
    hdf5::File::open(path)
        .unwrap()
        .dataset("d")
        .unwrap()
        .read_raw::<i32>()
        .unwrap()
}

fn read_pure(path: &Path) -> Vec<i32> {
    hdf5_pure::File::open(path)
        .unwrap()
        .dataset("d")
        .unwrap()
        .read_i32()
        .unwrap()
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
    assert_eq!(read_c(&path), data);
    assert_eq!(read_pure(&path), data);
}

const DEFLATE_LEVEL: u8 = 6;
