#![cfg(feature = "__hdf5-1.10")]
//! Reference-C-library interop for a chunked dataset whose partial edge chunks
//! are stored with the filter pipeline skipped.
//!
//! `ChunkOpts::DONT_FILTER_PARTIAL_CHUNKS` sets the layout flag through
//! `H5Pset_chunk_opts`, so the C library shuffles or deflates the whole chunks
//! of the dataset and stores each chunk that extends past the extent as it
//! stands. Both libraries read the values back.
//!
//! A dataset that hdf5-pure creates, buffered, staged, or streamed, with the flag
//! or without it, stores the layout flags, the chunk offsets and the filter
//! masks of the dataset libhdf5 creates with the same properties, and each
//! partial edge chunk under the flag as its elements padded with the fill value.
//! Under the flag, a dataset inside one partial chunk gets a Single Chunk index
//! and stores that chunk raw. Both libraries read the values, before and after
//! libhdf5 writes the last element.
//! An append through libhdf5 filters a raw edge chunk it fills, whichever
//! library created the dataset.
//!
//! An append through hdf5-pure, to a dataset either library created with the
//! flag or without it, stores each chunk it writes filtered or raw. An in-place
//! append with a lossless pipeline stores a chunk with every filter skipped where readers
//! at the old and new extents decode it differently. Under a lossy pipeline, an exempt
//! partial tail stays raw until complete, then is filtered after extent publication.
//! Both libraries read the values and report the same chunk
//! records, before and after libhdf5 writes past the appended values.
//!
//! With explicit permission to re-encode a filtered lossy tail, both libraries read the
//! values produced by repeated immediate, staged, and buffered appends, including after
//! reopening either library's file. Buffered writes cover partial flushes, automatic writes,
//! finish, and drop. Both libraries report the same chunk records.
//!
//! A staged overwrite through hdf5-pure, to a dataset either library created
//! with the flag or without it, stores each partial edge chunk unfiltered under
//! the flag and every other chunk filtered, and so does an overwrite of the
//! variable-length strings of a dataset libhdf5 created. A second overwrite,
//! whose chunks fit their slots, keeps every chunk at its address. A copy from
//! another file and a repack keep each chunk as the source stores it. Both
//! libraries read the values and report the same chunk records, before and
//! after libhdf5 writes the last element.
//!
//! A repack that re-encodes a sparse or an unallocated dataset, or restages the
//! variable-length elements of a dataset, stores the layout flags of the source
//! and each partial edge chunk under the flag unfiltered. libhdf5 reads the
//! values, and where the repack restages the elements, it stores the chunk
//! offsets, the filter masks, and the raw chunk sizes of the dataset libhdf5
//! creates with the same elements.

use std::cmp::Ordering;
use std::ops::Range;
use std::path::Path;

use hdf5::Datatype as CDatatype;
use hdf5::Extent;
use hdf5::Hyperslab;
use hdf5::SliceOrIndex;
use hdf5::dataset::ChunkOpts;
use hdf5::filters::Filter as CFilter;
#[cfg(target_endian = "little")]
use hdf5::types::VarLenArray;
#[cfg(target_endian = "little")]
use hdf5::types::VarLenUnicode;
use hdf5_pure::AppendOptions;
use hdf5_pure::BufferedAppender;
use hdf5_pure::ChunkIndex;
use hdf5_pure::DatasetCreateProperties;
use hdf5_pure::DeflateLevel;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::Layout;
use hdf5_pure::LossyTailReencoding;
use hdf5_pure::MaxExtent;
use hdf5_pure::MessageType;
use hdf5_pure::PartialEdgeFiltering;
use hdf5_pure::RepackOptions;
use hdf5_pure::ScaleOffset;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::ChunkedLayoutFlags;
use hdf5_pure_format::__private::DONT_FILTER_PARTIAL_BOUND_CHUNKS;
use hdf5_pure_format::__private::DataLayout;
use hdf5_pure_format::__private::ObjectHeader;
use hdf5_pure_format::__private::SINGLE_INDEX_WITH_FILTER;
use hdf5_sys::h5t;
use hdf5_sys::h5t::H5T_order_t;
use ndarray::ArrayD;
use ndarray::IxDyn;
use ndarray::Slice;
use rstest::rstest;
use test_util_hdf5::dataset;
use test_util_hdf5::dataset::Unlimited;
use test_util_hdf5::dataset_writer::DatasetWriter;
use test_util_hdf5::file;
use test_util_hdf5::numeric::ByteOrder;

use crate::array_index;
use crate::c_object;

#[derive(Clone, Copy, Debug)]
enum Filter {
    Deflate,
    Shuffle,
    ShuffleDeflate,
    Unfiltered,
}

impl Filter {
    fn pipeline(self) -> Vec<CFilter> {
        match self {
            Self::Deflate => vec![CFilter::Deflate(DEFLATE_LEVEL)],
            Self::Shuffle => vec![CFilter::Shuffle],
            Self::ShuffleDeflate => vec![CFilter::Shuffle, CFilter::Deflate(DEFLATE_LEVEL)],
            Self::Unfiltered => vec![],
        }
    }

    fn test_util_pipeline(self) -> Vec<dataset::Filter> {
        let deflate = dataset::Filter::Deflate(DeflateLevel::try_from(DEFLATE_LEVEL).unwrap());
        match self {
            Self::Deflate => vec![deflate],
            Self::Shuffle => vec![dataset::Filter::Shuffle],
            Self::ShuffleDeflate => vec![dataset::Filter::Shuffle, deflate],
            Self::Unfiltered => vec![],
        }
    }

    fn create_properties(self, properties: DatasetCreateProperties) -> DatasetCreateProperties {
        let deflate = DeflateLevel::try_from(DEFLATE_LEVEL).unwrap();
        match self {
            Self::Deflate => properties.with_deflate(deflate),
            Self::Shuffle => properties.with_shuffle(),
            Self::ShuffleDeflate => properties.with_shuffle().with_deflate(deflate),
            Self::Unfiltered => properties,
        }
    }

    #[cfg(target_endian = "little")]
    fn every_filter_skipped(self) -> u32 {
        match self {
            Self::Deflate | Self::Shuffle => 0b1,
            Self::ShuffleDeflate => 0b11,
            Self::Unfiltered => 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Creator {
    Libhdf5,
    Pure,
}

impl Creator {
    fn create_unlimited(self, path: &Path, unlimited: Unlimited<'_, i32>) {
        match self {
            Self::Libhdf5 => unlimited
                .stored_as(ByteOrder::LittleEndian.i32_datatype())
                .libhdf5_create(path),
            Self::Pure => unlimited.pure_create(path),
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
#[case::rank_1_partial_tail(&[10], &[4], Some(OVERWRITE_FILL))]
#[case::rank_1_partial_tail_without_a_fill_value(&[10], &[4], None)]
#[case::rank_1_whole_chunks(&[12], &[4], Some(OVERWRITE_FILL))]
#[case::chunk_past_the_whole_extent(&[3], &[4], Some(OVERWRITE_FILL))]
#[case::partial_along_both_axes(&[5, 5], &[2, 3], Some(OVERWRITE_FILL))]
#[case::partial_along_the_inner_axis(&[4, 5], &[2, 3], Some(OVERWRITE_FILL))]
#[case::partial_along_the_outer_axis(&[5, 6], &[2, 3], Some(OVERWRITE_FILL))]
#[case::partial_along_the_first_and_last_of_three_axes(
    &[3, 4, 5],
    &[2, 4, 3],
    Some(OVERWRITE_FILL),
)]
fn a_created_dataset_stores_each_chunk_as_libhdf5_creates_it(
    #[case] shape: &[usize],
    #[case] chunk: &[usize],
    #[case] fill: Option<i32>,
    #[values(false, true)] raw_edges: bool,
    #[values(
        Filter::Deflate,
        Filter::Shuffle,
        Filter::ShuffleDeflate,
        Filter::Unfiltered
    )]
    filter: Filter,
    #[values(
        DatasetWriter::Buffered,
        DatasetWriter::Staged,
        DatasetWriter::Streaming
    )]
    writer: DatasetWriter,
) {
    let dir = tempfile::tempdir().unwrap();
    let created = dir.path().join("partial_edge_created.h5");
    let reference = dir.path().join("partial_edge_reference.h5");
    let fixture = ChunkedFixture {
        shape,
        chunk,
        filter,
        raw_edges,
        fill,
    };
    let len = shape.iter().product();
    let mut values: Vec<i32> = (0..len).map(|i| i as i32).collect();
    fixture.pure_create(writer, &created, &values);
    fixture.libhdf5_create(&reference, &values);

    fixture.assert_stored(&created, &values);
    let flags = layout_flags(&created);
    assert_eq!(
        (flags.partial_edge_chunks_stored_raw(), flags),
        (raw_edges, layout_flags(&reference))
    );
    assert_eq!(
        chunk_offsets_and_masks(&created),
        chunk_offsets_and_masks(&reference)
    );

    fixture.libhdf5_write_last(&created, 999);

    values[len - 1] = 999;
    fixture.assert_stored(&created, &values);
}

#[rstest]
fn a_dataset_inside_one_partial_chunk_stores_it_raw_under_the_flag_and_a_single_chunk_index(
    #[values(Filter::Deflate, Filter::Shuffle, Filter::ShuffleDeflate)] filter: Filter,
    #[values(
        DatasetWriter::Buffered,
        DatasetWriter::Staged,
        DatasetWriter::Streaming
    )]
    writer: DatasetWriter,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("partial_edge_single_chunk.h5");
    let fixture = ChunkedFixture {
        shape: &[3],
        chunk: &[4],
        filter,
        raw_edges: true,
        fill: Some(OVERWRITE_FILL),
    };
    let mut values = vec![1, 2, 3];
    writer.create(&path, "d", |builder| {
        builder
            .with_i32_data(&values)
            .with_create_properties(fixture.create_properties());
    });

    fixture.assert_stored(&path, &values);
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("d")
            .unwrap()
            .layout()
            .unwrap(),
        Layout::Chunked {
            chunk_shape: vec![4],
            index: ChunkIndex::SingleChunk,
        }
    );
    assert_eq!(
        layout_flags(&path),
        ChunkedLayoutFlags::new(DONT_FILTER_PARTIAL_BOUND_CHUNKS | SINGLE_INDEX_WITH_FILTER)
    );

    fixture.libhdf5_write_last(&path, 999);

    values[2] = 999;
    fixture.assert_stored(&path, &values);
}

#[rstest]
#[cfg(target_endian = "little")]
fn a_libhdf5_append_filters_the_raw_edge_chunk_it_fills(
    #[values(Filter::Deflate, Filter::Shuffle, Filter::ShuffleDeflate)] filter: Filter,
    #[values(Creator::Libhdf5, Creator::Pure)] creator: Creator,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("partial_edge_filled.h5");
    let initial: Vec<i32> = (100..106).collect();
    creator.create_unlimited(
        &path,
        Unlimited::new("d", &initial, APPEND_CHUNK)
            .filters(&filter.test_util_pipeline())
            .fill_value(APPEND_FILL)
            .partial_edge_chunks_unfiltered(),
    );

    let file = hdf5::File::open_rw(&path).unwrap();
    let ds = file.dataset("d").unwrap();
    ds.resize([8]).unwrap();
    ds.write_slice(&[106, 107], 6..8).unwrap();
    drop(ds);
    file.close().unwrap();

    let values: Vec<i32> = (100..108).collect();
    assert_eq!(
        read_c_creation_properties(&path),
        (
            filter.pipeline(),
            Some(ChunkOpts::DONT_FILTER_PARTIAL_CHUNKS)
        )
    );
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), values);
    assert_eq!(dataset::read_pure::<i32>(&path, "d"), values);
    let chunks = array_index::libhdf5_chunks(&path, 0);
    assert_eq!(
        array_index::pure_chunks(&File::open(&path).unwrap()),
        chunks
    );
    let image = std::fs::read(&path).unwrap();
    let stored: Vec<_> = chunks
        .iter()
        .zip(values.chunks(4))
        .map(|(chunk, elements)| {
            let start = usize::try_from(chunk.address()).unwrap();
            let end = start + usize::try_from(chunk.size()).unwrap();
            let raw: Vec<u8> = elements.iter().flat_map(|e| e.to_le_bytes()).collect();
            (
                chunk.offset().to_vec(),
                image[start..end] == raw[..],
                chunk.filter_mask(),
            )
        })
        .collect();
    assert_eq!(
        stored,
        [(vec![0], false, 0), (vec![APPEND_CHUNK], false, 0)]
    );
}

fn chunk_offsets_and_masks(path: &Path) -> Vec<(Vec<u64>, u32)> {
    array_index::libhdf5_chunks(path, 0)
        .iter()
        .map(|chunk| (chunk.offset().to_vec(), chunk.filter_mask()))
        .collect()
}

#[rstest]
#[case::empty(&[])]
#[case::partial(&[1.46])]
fn lossy_unfiltered_tails_append_and_filter_on_completion(
    #[case] values: &[f64],
    #[values(Creator::Libhdf5, Creator::Pure)] creator: Creator,
    #[values(false, true)] staged: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lossy_raw_tail.h5");
    let reference = dir.path().join("lossy_raw_tail_reference.h5");
    let filters = [dataset::Filter::ScaleOffset(ScaleOffset::FloatDScale(1))];
    let datatype = little_endian_f64();
    let initial = Unlimited::new("d", values, 4)
        .filters(&filters)
        .partial_edge_chunks_unfiltered();
    match creator {
        Creator::Libhdf5 => initial
            .clone()
            .stored_as(datatype.clone())
            .libhdf5_create(&path),
        Creator::Pure => initial.pure_create(&path),
    }
    initial.stored_as(datatype).libhdf5_create(&reference);

    if values.is_empty() {
        append_lossy_tail(
            &path,
            &[1.46],
            staged || matches!(creator, Creator::Libhdf5),
        );
    }
    append_lossy_tail(&path, &[1.56], staged);
    let raw = [1.46, 1.56];
    assert_eq!(dataset::read_libhdf5::<f64>(&path, "d"), raw);
    assert_eq!(dataset::read_pure::<f64>(&path, "d"), raw);
    let chunks = array_index::libhdf5_chunks(&path, 0);
    assert_eq!(chunks.len(), 1);
    assert_eq!((chunks[0].size(), chunks[0].filter_mask()), (32, 0));

    append_lossy_tail(&path, &[1.0, 2.0, 3.46], staged);
    {
        let file = hdf5::File::open_rw(&reference).unwrap();
        let ds = file.dataset("d").unwrap();
        ds.resize([5]).unwrap();
        ds.write(&[1.46, 1.56, 1.0, 2.0, 3.46]).unwrap();
        drop(ds);
        file.close().unwrap();
    }
    let expected = dataset::read_libhdf5::<f64>(&reference, "d");
    assert_eq!(expected, [1.5, 1.6, 1.0, 2.0, 3.46]);
    assert_eq!(dataset::read_libhdf5::<f64>(&path, "d"), expected);
    assert_eq!(dataset::read_pure::<f64>(&path, "d"), expected);
    let chunks = array_index::libhdf5_chunks(&path, 0);
    assert_eq!(
        array_index::pure_chunks(&File::open(&path).unwrap()),
        chunks
    );
    assert_eq!(chunks.len(), 2);
    assert_eq!((chunks[0].filter_mask(), chunks[1].filter_mask()), (0, 0));
    assert_ne!(chunks[0].size(), 32);
    assert_eq!(chunks[1].size(), 32);

    append_lossy_tail(&path, &[3.56], staged);
    assert_eq!(
        dataset::read_libhdf5::<f64>(&path, "d"),
        [1.5, 1.6, 1.0, 2.0, 3.46, 3.56]
    );
    assert_eq!(
        dataset::read_pure::<f64>(&path, "d"),
        [1.5, 1.6, 1.0, 2.0, 3.46, 3.56]
    );
}

#[rstest]
fn filtered_lossy_tails_reencode_with_explicit_permission(
    #[values(Creator::Libhdf5, Creator::Pure)] creator: Creator,
    #[values(false, true)] staged: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reencoded_lossy_tail.h5");
    let reference = dir.path().join("reencoded_lossy_tail_reference.h5");
    let filters = [dataset::Filter::ScaleOffset(ScaleOffset::FloatDScale(1))];
    let initial = Unlimited::new("d", &[1.46, 1.56], 4).filters(&filters);
    match creator {
        Creator::Libhdf5 => initial
            .clone()
            .stored_as(little_endian_f64())
            .libhdf5_create(&path),
        Creator::Pure => initial.pure_create(&path),
    }
    initial
        .stored_as(little_endian_f64())
        .libhdf5_create(&reference);
    reencode_lossy_tail(&path, &reference, &[-0.26], staged);
    reencode_lossy_tail(&path, &reference, &[2.0, 3.46], staged);
    reencode_lossy_tail(&path, &reference, &[3.56], staged);
    let expected = dataset::read_libhdf5::<f64>(&reference, "d");
    assert_eq!(dataset::read_libhdf5::<f64>(&path, "d"), expected);
    assert_eq!(dataset::read_pure::<f64>(&path, "d"), expected);
    assert_eq!(expected, [1.44, 1.54, -0.26, 2.04, 3.46, 3.56]);
    let chunks = array_index::libhdf5_chunks(&path, 0);
    assert_eq!(
        array_index::pure_chunks(&File::open(&path).unwrap()),
        chunks
    );
    assert_eq!(chunks.len(), 2);
    assert_eq!((chunks[0].filter_mask(), chunks[1].filter_mask()), (0, 0));
}

#[rstest]
fn buffered_filtered_lossy_tails_reencode_with_explicit_permission(
    #[values(Creator::Libhdf5, Creator::Pure)] creator: Creator,
    #[values(false, true)] raw: bool,
    #[values(false, true)] finish: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("buffered_reencoded_lossy_tail.h5");
    let reference = dir
        .path()
        .join("buffered_reencoded_lossy_tail_reference.h5");
    let filters = [dataset::Filter::ScaleOffset(ScaleOffset::FloatDScale(1))];
    let initial = Unlimited::new("d", &[1.46, 1.56], 4).filters(&filters);
    match creator {
        Creator::Libhdf5 => initial
            .clone()
            .stored_as(little_endian_f64())
            .libhdf5_create(&path),
        Creator::Pure => initial.pure_create(&path),
    }
    initial
        .stored_as(little_endian_f64())
        .libhdf5_create(&reference);
    {
        let file = File::open_rw(&path).unwrap();
        let mut dataset = file.dataset("d").unwrap();
        let mut appender = dataset
            .buffered_appender_with_options(
                AppendOptions::new().with_lossy_tail_reencoding(LossyTailReencoding::Allow),
            )
            .unwrap();
        append_buffered_lossy_tail(&mut appender, &[-0.26], raw);
        appender.flush().unwrap();
        append_c_lossy_tail(&reference, &[-0.26]);
        append_buffered_lossy_tail(&mut appender, &[2.0, 3.46], raw);
        appender.flush().unwrap();
        append_c_lossy_tail(&reference, &[2.0, 3.46]);
        append_buffered_lossy_tail(&mut appender, &[3.56], raw);
        if finish {
            appender.finish().unwrap();
        } else {
            drop(appender);
        }
        append_c_lossy_tail(&reference, &[3.56]);
    }
    let expected = dataset::read_libhdf5::<f64>(&reference, "d");
    assert_eq!(expected, [1.44, 1.54, -0.26, 2.04, 3.46, 3.56]);
    assert_eq!(dataset::read_libhdf5::<f64>(&path, "d"), expected);
    assert_eq!(dataset::read_pure::<f64>(&path, "d"), expected);
    let chunks = array_index::libhdf5_chunks(&path, 0);
    assert_eq!(
        array_index::pure_chunks(&File::open(&path).unwrap()),
        chunks
    );
    assert_eq!(chunks.len(), 2);
    assert_eq!((chunks[0].filter_mask(), chunks[1].filter_mask()), (0, 0));
}

fn append_buffered_lossy_tail(appender: &mut BufferedAppender<'_>, values: &[f64], raw: bool) {
    if raw {
        appender
            .append_raw(
                &values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    } else {
        appender.append(values).unwrap();
    }
}

fn reencode_lossy_tail(path: &Path, reference: &Path, values: &[f64], staged: bool) {
    {
        let file = File::open_rw(path).unwrap();
        let mut dataset = file.dataset("d").unwrap();
        let options = AppendOptions::new().with_lossy_tail_reencoding(LossyTailReencoding::Allow);
        if staged {
            dataset
                .append_staged_with_options(options, |builder| {
                    builder.append_f64(values);
                })
                .unwrap();
            file.commit().unwrap();
        } else {
            dataset.append_with_options(options, values).unwrap();
        }
    }
    append_c_lossy_tail(reference, values);
    let expected = dataset::read_libhdf5::<f64>(reference, "d");
    assert_eq!(dataset::read_libhdf5::<f64>(path, "d"), expected);
    assert_eq!(dataset::read_pure::<f64>(path, "d"), expected);
}

fn append_c_lossy_tail(path: &Path, values: &[f64]) {
    let file = hdf5::File::open_rw(path).unwrap();
    let dataset = file.dataset("d").unwrap();
    let old_len = dataset.size();
    dataset.resize([old_len + values.len()]).unwrap();
    dataset
        .write_slice(values, old_len..old_len + values.len())
        .unwrap();
}

fn little_endian_f64() -> CDatatype {
    let datatype = CDatatype::from_type::<f64>().unwrap();
    hdf5::sync::sync(|| {
        // SAFETY: The lock serializes access and `datatype` owns a live floating-point type.
        let status = unsafe { h5t::H5Tset_order(datatype.id(), H5T_order_t::H5T_ORDER_LE) };
        assert_eq!(status, 0);
    });
    datatype
}

fn append_lossy_tail(path: &Path, values: &[f64], staged: bool) {
    let file = File::open_rw(path).unwrap();
    let mut ds = file.dataset("d").unwrap();
    if staged {
        ds.append_staged(|builder| {
            builder.append_f64(values);
        })
        .unwrap();
        file.commit().unwrap();
    } else {
        ds.append(values).unwrap();
    }
}

fn layout_flags(path: &Path) -> ChunkedLayoutFlags {
    let header = c_object::header_address(&hdf5::File::open(path).unwrap(), "d");
    let image = std::fs::read(path).unwrap();
    let parsed = ObjectHeader::parse(
        &image,
        AccessMode::ReadOnly,
        usize::try_from(header).unwrap(),
        OFFSET_SIZE,
        LENGTH_SIZE,
    )
    .unwrap();
    let message = parsed
        .messages
        .into_iter()
        .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
        .unwrap();
    let layout = DataLayout::parse(&message.data, OFFSET_SIZE, LENGTH_SIZE).unwrap();
    let DataLayout::Chunked { flags, .. } = layout else {
        panic!("expected a chunked layout, got {layout:?}");
    };
    flags
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
    #[values(Creator::Libhdf5, Creator::Pure)] creator: Creator,
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
    creator.create_unlimited(&path, unlimited);
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
    #[values(Creator::Libhdf5, Creator::Pure)] creator: Creator,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("partial_edge_staged_append.h5");
    let initial: Vec<i32> = (100..106).collect();
    let unlimited = Unlimited::new("d", &initial, APPEND_CHUNK)
        .filters(&filter.test_util_pipeline())
        .fill_value(APPEND_FILL);
    let (unlimited, chunk_opts) = if raw_edges {
        (
            unlimited.partial_edge_chunks_unfiltered(),
            ChunkOpts::DONT_FILTER_PARTIAL_CHUNKS,
        )
    } else {
        (unlimited, ChunkOpts::empty())
    };
    creator.create_unlimited(&path, unlimited);
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
    #[values(Creator::Libhdf5, Creator::Pure)] creator: Creator,
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
    fixture.create(
        creator,
        &path,
        &(0..len).map(|i| i as i32).collect::<Vec<_>>(),
    );

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

#[rstest]
#[case::sparse_rank_1(ReEncoded::SparseFromLibhdf5, &[10], &[4])]
#[case::sparse_partial_along_both_axes(ReEncoded::SparseFromLibhdf5, &[5, 5], &[2, 3])]
#[case::unallocated_rank_1_from_libhdf5(ReEncoded::UnallocatedFromLibhdf5, &[10], &[4])]
#[case::unallocated_partial_along_both_axes_from_libhdf5(
    ReEncoded::UnallocatedFromLibhdf5,
    &[5, 5],
    &[2, 3],
)]
#[case::unallocated_rank_1_from_hdf5_pure(ReEncoded::UnallocatedFromPure, &[10], &[4])]
#[case::unallocated_partial_along_both_axes_from_hdf5_pure(
    ReEncoded::UnallocatedFromPure,
    &[5, 5],
    &[2, 3],
)]
fn a_repack_that_re_encodes_a_dataset_keeps_its_partial_edge_filtering(
    #[case] source: ReEncoded,
    #[case] shape: &[usize],
    #[case] chunk: &[usize],
    #[values(false, true)] raw_edges: bool,
    #[values(Filter::Deflate, Filter::Shuffle, Filter::ShuffleDeflate)] filter: Filter,
) {
    let dir = tempfile::tempdir().unwrap();
    let original = dir.path().join("partial_edge_re_encoded_source.h5");
    let repacked = dir.path().join("partial_edge_re_encoded.h5");
    let fixture = ChunkedFixture {
        shape,
        chunk,
        filter,
        raw_edges,
        fill: Some(OVERWRITE_FILL),
    };
    let len = shape.iter().product();
    let written: Vec<i32> = (0..len).map(|i| i as i32).collect();
    source.create(fixture, &original, &written);

    hdf5_pure::repack(&original, &repacked, &RepackOptions::new()).unwrap();

    let mut values = source.values(fixture, &written);
    fixture.assert_stored(&repacked, &values);
    assert_eq!(
        (chunk_offsets(&original), chunk_offsets(&repacked)),
        source.chunk_offsets(fixture)
    );
    let flags = layout_flags(&repacked);
    assert_eq!(
        (flags.partial_edge_chunks_stored_raw(), flags),
        (raw_edges, layout_flags(&original))
    );

    fixture.libhdf5_write_last(&repacked, 999);

    values[len - 1] = 999;
    fixture.assert_stored(&repacked, &values);
}

#[derive(Clone, Copy, Debug)]
enum ReEncoded {
    SparseFromLibhdf5,
    UnallocatedFromLibhdf5,
    UnallocatedFromPure,
}

impl ReEncoded {
    fn create(self, fixture: ChunkedFixture<'_>, path: &Path, written: &[i32]) {
        match self {
            Self::SparseFromLibhdf5 => {
                fixture.libhdf5_create_without_the_first_chunk(path, written)
            }
            Self::UnallocatedFromLibhdf5 => fixture.libhdf5_create_unallocated(path),
            Self::UnallocatedFromPure => fixture.pure_create_unallocated(path),
        }
    }

    /// Returns the values of the dataset [`create`](Self::create) creates from `written`: the fill
    /// value, or 0 without one, in place of each element it leaves unwritten.
    fn values(self, fixture: ChunkedFixture<'_>, written: &[i32]) -> Vec<i32> {
        let fill = fixture.fill.unwrap_or(0);
        written
            .iter()
            .enumerate()
            .map(|(index, &value)| match self {
                Self::SparseFromLibhdf5 if !fixture.in_first_chunk(index) => value,
                Self::SparseFromLibhdf5
                | Self::UnallocatedFromLibhdf5
                | Self::UnallocatedFromPure => fill,
            })
            .collect()
    }

    /// Returns the offsets of the chunks the source stores and of the chunks its repack stores.
    fn chunk_offsets(self, fixture: ChunkedFixture<'_>) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
        match self {
            Self::SparseFromLibhdf5 => {
                let grid = fixture.grid_offsets();
                (grid[1..].to_vec(), grid)
            }
            Self::UnallocatedFromLibhdf5 | Self::UnallocatedFromPure => (Vec::new(), Vec::new()),
        }
    }
}

fn chunk_offsets(path: &Path) -> Vec<Vec<u64>> {
    array_index::libhdf5_chunks(path, 0)
        .iter()
        .map(|chunk| chunk.offset().to_vec())
        .collect()
}

#[rstest]
#[cfg(target_endian = "little")]
fn a_repack_that_restages_variable_length_elements_keeps_the_partial_edge_filtering(
    #[values(
        Restaged::CompoundsFromLibhdf5,
        Restaged::SequencesFromLibhdf5,
        Restaged::StringsFromLibhdf5,
        Restaged::StringsFromPure
    )]
    source: Restaged,
    #[values(false, true)] raw_edges: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let original = dir.path().join("partial_edge_restaged_source.h5");
    let repacked = dir.path().join("partial_edge_restaged.h5");
    let repacked_again = dir.path().join("partial_edge_restaged_again.h5");
    let reference = dir.path().join("partial_edge_restaged_reference.h5");
    let chunk_opts = if raw_edges {
        ChunkOpts::DONT_FILTER_PARTIAL_CHUNKS
    } else {
        ChunkOpts::empty()
    };
    source.create(&original, chunk_opts);
    source.libhdf5_reference().create(&reference, chunk_opts);

    hdf5_pure::repack(&original, &repacked, &RepackOptions::new()).unwrap();

    let values = source.values();
    assert_eq!(
        read_c_creation_properties(&repacked),
        (Filter::Deflate.pipeline(), Some(chunk_opts))
    );
    let flags = layout_flags(&repacked);
    assert_eq!(
        (flags.partial_edge_chunks_stored_raw(), flags),
        (raw_edges, layout_flags(&original))
    );
    assert_eq!(source.read_libhdf5(&repacked), values);
    assert_eq!(
        array_index::pure_chunks(&File::open(&repacked).unwrap()),
        array_index::libhdf5_chunks(&repacked, 0)
    );
    assert_eq!(
        restaged_chunks(&repacked, raw_edges),
        restaged_chunks(&reference, raw_edges)
    );

    hdf5_pure::repack(&repacked, &repacked_again, &RepackOptions::new()).unwrap();

    assert_eq!(source.read_libhdf5(&repacked_again), values);
}

#[cfg(target_endian = "little")]
#[derive(Clone, Copy, Debug)]
enum Restaged {
    CompoundsFromLibhdf5,
    SequencesFromLibhdf5,
    StringsFromLibhdf5,
    StringsFromPure,
}

#[cfg(target_endian = "little")]
impl Restaged {
    fn create(self, path: &Path, chunk_opts: ChunkOpts) {
        let indices = 0..RESTAGED_LEN;
        match self {
            Self::CompoundsFromLibhdf5 => libhdf5_create_restaged(
                path,
                chunk_opts,
                &indices
                    .map(|i| Labeled {
                        id: i as i32,
                        label: format!("element-{i}").parse().unwrap(),
                    })
                    .collect::<Vec<_>>(),
            ),
            Self::SequencesFromLibhdf5 => libhdf5_create_restaged(
                path,
                chunk_opts,
                &indices
                    .map(|i| VarLenArray::from_slice(&(0..i as i32).collect::<Vec<_>>()))
                    .collect::<Vec<_>>(),
            ),
            Self::StringsFromLibhdf5 => libhdf5_create_restaged(
                path,
                chunk_opts,
                &var_len_strings(indices.map(|i| format!("element-{i}"))),
            ),
            Self::StringsFromPure => {
                let strings: Vec<String> = indices.map(|i| format!("element-{i}")).collect();
                let filtering = if chunk_opts.is_empty() {
                    PartialEdgeFiltering::Apply
                } else {
                    PartialEdgeFiltering::Skip
                };
                DatasetWriter::Buffered.create(path, "d", |builder| {
                    builder
                        .with_vlen_strings(&strings.iter().map(String::as_str).collect::<Vec<_>>())
                        .with_create_properties(
                            Filter::Deflate
                                .create_properties(
                                    DatasetCreateProperties::new()
                                        .with_chunks(&[RESTAGED_CHUNK as u64]),
                                )
                                .with_partial_edge_filtering(filtering),
                        );
                });
            }
        }
    }

    /// Returns the source libhdf5 creates with the elements of `self`, the oracle for the
    /// [`restaged_chunks`] of a repack of `self`.
    fn libhdf5_reference(self) -> Self {
        match self {
            Self::StringsFromPure => Self::StringsFromLibhdf5,
            Self::CompoundsFromLibhdf5 | Self::SequencesFromLibhdf5 | Self::StringsFromLibhdf5 => {
                self
            }
        }
    }

    /// Returns the elements [`create`](Self::create) writes, formatted as
    /// [`read_libhdf5`](Self::read_libhdf5) formats them.
    fn values(self) -> Vec<String> {
        (0..RESTAGED_LEN)
            .map(|i| match self {
                Self::CompoundsFromLibhdf5 => format!("{i}:element-{i}"),
                Self::SequencesFromLibhdf5 => format!("{:?}", (0..i as i32).collect::<Vec<_>>()),
                Self::StringsFromLibhdf5 | Self::StringsFromPure => format!("element-{i}"),
            })
            .collect()
    }

    fn read_libhdf5(self, path: &Path) -> Vec<String> {
        let ds = hdf5::File::open(path).unwrap().dataset("d").unwrap();
        match self {
            Self::CompoundsFromLibhdf5 => ds
                .read_raw::<Labeled>()
                .unwrap()
                .iter()
                .map(|element| format!("{}:{}", element.id, element.label))
                .collect(),
            Self::SequencesFromLibhdf5 => ds
                .read_raw::<VarLenArray<i32>>()
                .unwrap()
                .iter()
                .map(|sequence| format!("{:?}", sequence.as_slice()))
                .collect(),
            Self::StringsFromLibhdf5 | Self::StringsFromPure => {
                dataset::read_libhdf5_strings(path, "d")
            }
        }
    }
}

/// Returns the offset and the filter mask of each chunk of the dataset at `path`, and the size of
/// each partial edge chunk stored raw under `raw_edges`.
///
/// The size of a filtered chunk is left out: the chunk holds global heap addresses, and its
/// compressed size depends on where each file places its heap.
#[cfg(target_endian = "little")]
fn restaged_chunks(path: &Path, raw_edges: bool) -> Vec<(Vec<u64>, Option<u64>, u32)> {
    array_index::libhdf5_chunks(path, 0)
        .iter()
        .map(|chunk| {
            let raw = raw_edges && chunk.offset()[0] as usize + RESTAGED_CHUNK > RESTAGED_LEN;
            (
                chunk.offset().to_vec(),
                raw.then(|| chunk.size()),
                chunk.filter_mask(),
            )
        })
        .collect()
}

#[cfg(target_endian = "little")]
#[derive(hdf5::H5Type, Clone, Debug)]
#[repr(C)]
struct Labeled {
    id: i32,
    label: VarLenUnicode,
}

#[cfg(target_endian = "little")]
fn libhdf5_create_restaged<T: hdf5::H5Type>(path: &Path, chunk_opts: ChunkOpts, values: &[T]) {
    let file = file::libhdf5_create_v110(path);
    file.new_dataset::<T>()
        .deflate(DEFLATE_LEVEL)
        .chunk([RESTAGED_CHUNK])
        .chunk_opts(chunk_opts)
        .shape([values.len()])
        .create("d")
        .unwrap()
        .write(values)
        .unwrap();
    file.close().unwrap();
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
    fn create(self, creator: Creator, path: &Path, data: &[i32]) {
        match creator {
            Creator::Libhdf5 => self.libhdf5_create(path, data),
            Creator::Pure => self.pure_create(DatasetWriter::Streaming, path, data),
        }
    }

    fn pure_create(self, writer: DatasetWriter, path: &Path, data: &[i32]) {
        writer.create(path, "d", |builder| {
            builder
                .with_i32_data(data)
                .with_shape(&self.dims())
                .with_maxshape(&self.maxshape())
                .with_create_properties(self.create_properties());
        });
    }

    fn pure_create_unallocated(self, path: &Path) {
        DatasetWriter::Buffered.create(path, "d", |builder| {
            builder
                .with_unallocated_storage(hdf5_pure::make_i32_type(), &self.dims())
                .with_maxshape(&self.maxshape())
                .with_create_properties(self.create_properties());
        });
    }

    fn maxshape(self) -> Vec<MaxExtent> {
        extents_the_c_library_accepts(self.shape, self.chunk)
            .iter()
            .map(|extent| match extent.max {
                Some(max) => MaxExtent::Fixed(max as u64),
                None => MaxExtent::Unlimited,
            })
            .collect()
    }

    fn create_properties(self) -> DatasetCreateProperties {
        let chunk: Vec<u64> = self.chunk.iter().map(|&edge| edge as u64).collect();
        let properties = self
            .filter
            .create_properties(DatasetCreateProperties::new().with_chunks(&chunk));
        let properties = match self.fill {
            Some(fill) => properties.with_fill_value(fill),
            None => properties,
        };
        properties.with_partial_edge_filtering(if self.raw_edges {
            PartialEdgeFiltering::Skip
        } else {
            PartialEdgeFiltering::Apply
        })
    }

    fn dims(self) -> Vec<u64> {
        self.shape.iter().map(|&len| len as u64).collect()
    }

    fn libhdf5_create(self, path: &Path, data: &[i32]) {
        let file = file::libhdf5_create_v110(path);
        self.libhdf5_create_dataset(&file).write_raw(data).unwrap();
        file.close().unwrap();
    }

    fn libhdf5_create_without_the_first_chunk(self, path: &Path, data: &[i32]) {
        let file = file::libhdf5_create_v110(path);
        let ds = self.libhdf5_create_dataset(&file);
        let data = ArrayD::from_shape_vec(IxDyn(self.shape), data.to_vec()).unwrap();
        for axis in 0..self.shape.len() {
            let region: Vec<Range<usize>> = self
                .shape
                .iter()
                .zip(self.chunk)
                .enumerate()
                .map(|(other, (&len, &edge))| match other.cmp(&axis) {
                    Ordering::Less => 0..edge.min(len),
                    Ordering::Equal => edge.min(len)..len,
                    Ordering::Greater => 0..len,
                })
                .collect();
            if region[axis].is_empty() {
                continue;
            }
            ds.write_slice(
                &data
                    .slice_each_axis(|ax| Slice::from(region[ax.axis.index()].clone()))
                    .to_owned(),
                Hyperslab::from(
                    region
                        .into_iter()
                        .map(SliceOrIndex::from)
                        .collect::<Vec<_>>(),
                ),
            )
            .unwrap();
        }
        drop(ds);
        file.close().unwrap();
    }

    fn libhdf5_create_unallocated(self, path: &Path) {
        let file = file::libhdf5_create_v110(path);
        drop(self.libhdf5_create_dataset(&file));
        file.close().unwrap();
    }

    fn libhdf5_create_dataset(self, file: &hdf5::File) -> hdf5::Dataset {
        let builder = file
            .new_dataset_builder()
            .empty_as(ByteOrder::LittleEndian.i32_datatype());
        let builder = match self.filter {
            Filter::Deflate => builder.deflate(DEFLATE_LEVEL),
            Filter::Shuffle => builder.shuffle(),
            Filter::ShuffleDeflate => builder.shuffle().deflate(DEFLATE_LEVEL),
            Filter::Unfiltered => builder,
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
    }

    fn pure_overwrite(self, path: &Path, values: &[i32]) {
        let file = File::open_rw(path).unwrap();
        file.dataset("d")
            .unwrap()
            .write_staged(|builder| {
                builder.with_i32_data(values).with_shape(&self.dims());
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

    /// Returns `true` if the element at the row-major `index` lies in the chunk at the origin.
    fn in_first_chunk(self, index: usize) -> bool {
        self.shape
            .iter()
            .rev()
            .scan(index, |remaining, &len| {
                let coord = *remaining % len;
                *remaining /= len;
                Some(coord)
            })
            .zip(self.chunk.iter().rev())
            .all(|(coord, &edge)| coord < edge)
    }

    /// Returns the offset of every chunk of the grid, in row-major order.
    fn grid_offsets(self) -> Vec<Vec<u64>> {
        self.shape
            .iter()
            .zip(self.chunk)
            .fold(vec![Vec::new()], |offsets, (&len, &edge)| {
                offsets
                    .iter()
                    .flat_map(|offset| {
                        (0..len).step_by(edge).map(move |start| {
                            let mut offset = offset.clone();
                            offset.push(start as u64);
                            offset
                        })
                    })
                    .collect()
            })
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
const LENGTH_SIZE: u8 = 8;
const OFFSET_SIZE: u8 = 8;
const OVERWRITE_FILL: i32 = -3;
#[cfg(target_endian = "little")]
const RESTAGED_CHUNK: usize = 4;
#[cfg(target_endian = "little")]
const RESTAGED_LEN: usize = 10;
// A variable-length string element is stored as a sequence length, a global heap collection
// address and an object index.
#[cfg(target_endian = "little")]
const VLEN_REFERENCE_SIZE: u64 = 16;
