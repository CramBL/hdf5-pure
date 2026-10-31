#![cfg(feature = "deflate")]

#[cfg(feature = "std")]
use std::path::Path;

use hdf5_pure_format::__private::SINGLE_INDEX_WITH_FILTER;

use super::*;
use crate::DatatypeByteOrder;
#[cfg(feature = "std")]
use crate::File;
#[cfg(feature = "std")]
use crate::FileBuilder;
use crate::FixedPointType;
use crate::filters;
#[cfg(feature = "std")]
use crate::reader::Dataset;
#[cfg(feature = "std")]
use crate::reader::Object;
use crate::source;
#[cfg(feature = "std")]
use crate::type_builders::DatasetBuilder;

#[rstest]
#[case::one_dimension(&[10], &[4], &[false, false, true])]
#[case::two_dimensions(
    &[5, 7],
    &[2, 3],
    &[false, false, true, false, false, true, true, true, true],
)]
#[case::three_dimensions(&[3, 4, 5], &[2, 4, 3], &[false, true, true, true])]
#[case::single_chunk(&[5], &[8], &[true])]
#[case::no_partial_chunk(&[8], &[4], &[false, false])]
fn a_chunk_set_skipping_partial_edge_filtering_stores_only_its_partial_edge_chunks_raw(
    #[case] shape: &[u64],
    #[case] chunk_dims: &[u64],
    #[case] partial: &[bool],
    #[values(
        EdgeFilters::Unfiltered,
        EdgeFilters::Deflate,
        EdgeFilters::ShuffleDeflate
    )]
    edge_filters: EdgeFilters,
    #[values(None, Some(-7))] fill: Option<i32>,
) {
    let options = edge_filters.options(chunk_dims, Some(PartialEdgeFiltering::Skip));
    let fill_bytes = fill.map(i32::to_le_bytes);
    let fill = i32_fill(fill_bytes.as_ref());
    let raw = i32_element_bytes(shape);
    let ctx = ChunkContext::basic(chunk_dims, 4);

    let set = compress_chunks(
        &raw,
        shape,
        ctx,
        &options,
        None,
        fill,
        FillTime::IfSet,
        StorageAllocation::Allocated,
    )
    .unwrap();

    assert_eq!(
        stored_chunks(&set),
        expected_chunks(&raw, shape, ctx, &options, fill, partial)
    );
    let assembled = assemble_chunked_at(&set, StoredAddress::new(BASE)).unwrap();
    assert!(layout_flags(&assembled.layout_message).partial_edge_chunks_stored_raw());
    assert_eq!(read_back(&assembled, shape, fill), raw);
}

#[rstest]
#[case::default_fill(None, [4, 5, 0, 0])]
#[case::explicit_fill(Some(-7), [4, 5, -7, -7])]
fn a_raw_partial_edge_chunk_holds_a_whole_chunk_padded_with_the_fill_value(
    #[case] fill: Option<i32>,
    #[case] edge: [i32; 4],
) {
    let fill_bytes = fill.map(i32::to_le_bytes);
    let set = compress_chunks(
        &i32_element_bytes(&[6]),
        &[6],
        ChunkContext::basic(&[4], 4),
        &EdgeFilters::Deflate.options(&[4], Some(PartialEdgeFiltering::Skip)),
        None,
        i32_fill(fill_bytes.as_ref()),
        FillTime::IfSet,
        StorageAllocation::Allocated,
    )
    .unwrap();

    assert_eq!(set.chunks[1].bytes(), i32_bytes(&edge));
    assert_eq!(set.chunks[1].filter_mask(), 0);
}

#[test]
fn the_default_policy_filters_every_chunk_and_stores_no_layout_flag() {
    let options = EdgeFilters::ShuffleDeflate.options(&[4], None);
    let raw = i32_element_bytes(&[10]);
    let ctx = ChunkContext::basic(&[4], 4);

    let set = compress_chunks(
        &raw,
        &[10],
        ctx,
        &options,
        None,
        FillPattern::ZERO,
        FillTime::IfSet,
        StorageAllocation::Allocated,
    )
    .unwrap();

    assert_eq!(
        stored_chunks(&set),
        expected_chunks(
            &raw,
            &[10],
            ctx,
            &options,
            FillPattern::ZERO,
            &[false, false, false]
        )
    );
    let assembled = assemble_chunked_at(&set, StoredAddress::new(BASE)).unwrap();
    assert_eq!(
        layout_flags(&assembled.layout_message),
        ChunkedLayoutFlags::NONE
    );
}

#[rstest]
#[case::fixed_array(&[10], &[4], None)]
#[case::extensible_array(&[10], &[4], Some(&[MaxExtent::Unlimited][..]))]
#[case::single_chunk(&[5], &[8], None)]
fn the_sizing_and_assembly_passes_store_the_same_layout_flags(
    #[case] shape: &[u64],
    #[case] chunk_dims: &[u64],
    #[case] maxshape: Option<&[MaxExtent]>,
    #[values(
        None,
        Some(PartialEdgeFiltering::Apply),
        Some(PartialEdgeFiltering::Skip)
    )]
    policy: Option<PartialEdgeFiltering>,
    #[values(EdgeFilters::Unfiltered, EdgeFilters::Deflate)] edge_filters: EdgeFilters,
    #[values(StorageAllocation::Allocated, StorageAllocation::Unallocated)]
    allocation: StorageAllocation,
) {
    let options = edge_filters.options(chunk_dims, policy);
    let ctx = ChunkContext::basic(chunk_dims, 4);
    let set = compress_chunks(
        &i32_element_bytes(shape),
        shape,
        ctx,
        &options,
        maxshape,
        FillPattern::ZERO,
        FillTime::IfSet,
        allocation,
    )
    .unwrap();
    let header = options
        .measure_header(shape, ctx, maxshape, FillPattern::ZERO, allocation)
        .unwrap();

    let measured = measure_chunked_at(&set, StoredAddress::new(BASE)).unwrap();
    let assembled = assemble_chunked_at(&set, StoredAddress::new(BASE)).unwrap();

    assert_eq!(measured.layout_message, assembled.layout_message);
    assert_eq!(header.layout_message.len(), assembled.layout_message.len());
    let flags = layout_flags(&assembled.layout_message);
    assert_eq!(layout_flags(&header.layout_message), flags);
    assert_eq!(
        flags.partial_edge_chunks_stored_raw(),
        policy == Some(PartialEdgeFiltering::Skip)
    );
}

#[test]
fn a_raw_single_chunk_stores_its_size_in_the_filtered_single_chunk_index() {
    let set = compress_chunks(
        &i32_element_bytes(&[5]),
        &[5],
        ChunkContext::basic(&[8], 4),
        &EdgeFilters::Deflate.options(&[8], Some(PartialEdgeFiltering::Skip)),
        None,
        FillPattern::ZERO,
        FillTime::IfSet,
        StorageAllocation::Allocated,
    )
    .unwrap();

    let assembled = assemble_chunked_at(&set, StoredAddress::new(BASE)).unwrap();

    assert_eq!(
        DataLayout::parse(&assembled.layout_message, 8, 8).unwrap(),
        DataLayout::Chunked {
            flags: ChunkedLayoutFlags::new(
                DONT_FILTER_PARTIAL_BOUND_CHUNKS | SINGLE_INDEX_WITH_FILTER
            ),
            chunk_dimensions: vec![8, 4],
            index: ChunkIndexLayout::SingleChunk {
                filtered: Some(FilteredSingleChunk {
                    filtered_size: 32,
                    filter_mask: 0,
                }),
                address: Some(StoredAddress::new(BASE)),
            },
        }
    );
}

#[cfg(feature = "std")]
#[rstest]
#[case::one_dimension(&[10], &[4], &[false, false, true])]
#[case::two_dimensions(
    &[5, 7],
    &[2, 3],
    &[false, false, true, false, false, true, true, true, true],
)]
#[case::single_chunk(&[5], &[8], &[true])]
fn a_dataset_written_skipping_partial_edge_filtering_stores_only_its_partial_edge_chunks_raw(
    #[case] shape: &[u64],
    #[case] chunk_dims: &[u64],
    #[case] partial: &[bool],
    #[values(EdgeFilters::Unfiltered, EdgeFilters::ShuffleDeflate)] edge_filters: EdgeFilters,
    #[values(Writer::Buffered, Writer::Streamed, Writer::Staged)] writer: Writer,
) {
    let values = i32_elements(shape);
    let bytes = writer.write(|dataset| {
        dataset
            .with_i32_data(&values)
            .with_shape(shape)
            .with_fill_value(EDGE_FILL);
        edge_filters.configure(dataset, chunk_dims);
        dataset.chunk_options.partial_edge_filtering = Some(PartialEdgeFiltering::Skip);
    });

    let file = File::from_bytes(bytes.clone()).unwrap();
    let dataset = file.dataset("d").unwrap();
    assert!(dataset_flags(&dataset).partial_edge_chunks_stored_raw());
    let fill_bytes = EDGE_FILL.to_le_bytes();
    let fill = i32_fill(Some(&fill_bytes));
    let options = edge_filters.options(chunk_dims, Some(PartialEdgeFiltering::Skip));
    let expected: Vec<(bool, u32, Vec<u8>)> = expected_chunks(
        &i32_bytes(&values),
        shape,
        ChunkContext::basic(chunk_dims, 4),
        &options,
        fill,
        partial,
    )
    .into_iter()
    .zip(partial)
    .map(|((stored, filter_mask), &partial)| (partial, filter_mask, stored))
    .collect();
    assert_eq!(file_chunks(&dataset, &bytes), expected);
    assert_eq!(dataset.read_i32().unwrap(), values);
}

#[cfg(feature = "std")]
#[test]
fn streaming_a_dataset_skipping_partial_edge_filtering_writes_the_bytes_of_finish() {
    let configure = |dataset: &mut DatasetBuilder| {
        dataset
            .with_i32_data(&i32_elements(&[10]))
            .with_shape(&[10]);
        EdgeFilters::ShuffleDeflate.configure(dataset, &[4]);
        dataset.chunk_options.partial_edge_filtering = Some(PartialEdgeFiltering::Skip);
    };

    assert_eq!(
        Writer::Streamed.write(configure),
        Writer::Buffered.write(configure)
    );
}

#[cfg(feature = "std")]
#[test]
fn an_object_reference_dataset_skipping_partial_edge_filtering_stores_its_partial_edge_chunk_raw() {
    let mut builder = FileBuilder::new();
    builder.create_dataset("target").with_i32_data(&[1, 2]);
    let dataset = builder
        .create_dataset("references")
        .with_path_references(&["target", "target", "target"]);
    EdgeFilters::ShuffleDeflate.configure(dataset, &[2]);
    dataset.chunk_options.partial_edge_filtering = Some(PartialEdgeFiltering::Skip);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes.clone()).unwrap();
    let dataset = file.dataset("references").unwrap();
    assert!(dataset_flags(&dataset).partial_edge_chunks_stored_raw());
    let expected: Vec<(bool, u32, Vec<u8>)> = expected_chunks(
        &dataset.read_raw().unwrap(),
        &[3],
        ChunkContext::basic(&[2], 8),
        &EdgeFilters::ShuffleDeflate.options(&[2], Some(PartialEdgeFiltering::Skip)),
        FillPattern::ZERO,
        &[false, true],
    )
    .into_iter()
    .zip([false, true])
    .map(|((stored, filter_mask), partial)| (partial, filter_mask, stored))
    .collect();
    assert_eq!(file_chunks(&dataset, &bytes), expected);
    let targets: Vec<Vec<i32>> = dataset
        .dereference()
        .unwrap()
        .into_iter()
        .map(|object| {
            let Object::Dataset(target) = object else {
                panic!("a reference refers to a group");
            };
            target.read_i32().unwrap()
        })
        .collect();
    assert_eq!(targets, vec![vec![1, 2]; 3]);
}

#[cfg(feature = "std")]
#[rstest]
#[case::in_place_tail_stays_partial(
    Append::InPlace,
    &[106],
    &[(Stored::Raw, [104, 105, 106, EDGE_FILL])],
)]
#[case::in_place_tail_becomes_whole(
    Append::InPlace,
    &[106, 107],
    &[(Stored::FiltersSkipped, [104, 105, 106, 107])],
)]
#[case::in_place_tail_becomes_whole_before_a_partial_tail(
    Append::InPlace,
    &[106, 107, 108],
    &[
        (Stored::FiltersSkipped, [104, 105, 106, 107]),
        (Stored::Raw, [108, EDGE_FILL, EDGE_FILL, EDGE_FILL]),
    ],
)]
#[case::staged_tail_stays_partial(
    Append::Staged,
    &[106],
    &[(Stored::Raw, [104, 105, 106, EDGE_FILL])],
)]
#[case::staged_tail_becomes_whole(
    Append::Staged,
    &[106, 107],
    &[(Stored::Filtered, [104, 105, 106, 107])],
)]
#[case::staged_tail_becomes_whole_before_a_partial_tail(
    Append::Staged,
    &[106, 107, 108],
    &[
        (Stored::Filtered, [104, 105, 106, 107]),
        (Stored::Raw, [108, EDGE_FILL, EDGE_FILL, EDGE_FILL]),
    ],
)]
fn an_append_to_a_dataset_written_skipping_partial_edge_filtering_stores_each_chunk_for_its_extent(
    #[case] append: Append,
    #[case] appended: &[i32],
    #[case] tail: &[(Stored, [i32; 4])],
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edge.h5");
    std::fs::write(&path, edge_file()).unwrap();

    append.append(&path, appended);

    let bytes = std::fs::read(&path).unwrap();
    let file = File::from_bytes(bytes.clone()).unwrap();
    let dataset = file.dataset("d").unwrap();
    assert!(dataset_flags(&dataset).partial_edge_chunks_stored_raw());
    let expected: Vec<(bool, u32, Vec<u8>)> =
        core::iter::once(&(Stored::Filtered, [100, 101, 102, 103]))
            .chain(tail)
            .map(|&(stored, values)| stored.chunk(values))
            .collect();
    assert_eq!(file_chunks(&dataset, &bytes), expected);
    assert_eq!(
        dataset.read_i32().unwrap(),
        [&EDGE_INITIAL[..], appended].concat()
    );
}

#[cfg(feature = "std")]
#[test]
fn an_overwrite_of_a_dataset_written_skipping_partial_edge_filtering_stores_its_edge_chunk_raw() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("edge.h5");
    std::fs::write(&path, edge_file()).unwrap();
    let replacement = [200, 201, 202, 203, 204, 205];

    {
        let file = File::open_rw(&path).unwrap();
        file.dataset("d")
            .unwrap()
            .write_staged(|dataset| {
                dataset.with_i32_data(&replacement);
            })
            .unwrap();
        file.commit().unwrap();
    }

    let bytes = std::fs::read(&path).unwrap();
    let file = File::from_bytes(bytes.clone()).unwrap();
    let dataset = file.dataset("d").unwrap();
    assert!(dataset_flags(&dataset).partial_edge_chunks_stored_raw());
    assert_eq!(
        file_chunks(&dataset, &bytes),
        vec![
            Stored::Filtered.chunk([200, 201, 202, 203]),
            Stored::Raw.chunk([204, 205, EDGE_FILL, EDGE_FILL]),
        ]
    );
    assert_eq!(dataset.read_i32().unwrap(), replacement);
}

/// The filter pipelines the tests write a dataset with.
#[derive(Clone, Copy, Debug)]
enum EdgeFilters {
    Unfiltered,
    Deflate,
    ShuffleDeflate,
}

impl EdgeFilters {
    fn options(self, chunk_dims: &[u64], policy: Option<PartialEdgeFiltering>) -> ChunkOptions {
        ChunkOptions {
            partial_edge_filtering: policy,
            ..chunked(chunk_dims, self.kinds())
        }
    }

    #[cfg(feature = "std")]
    fn configure(self, dataset: &mut DatasetBuilder, chunk_dims: &[u64]) {
        dataset.with_chunks(chunk_dims);
        for &kind in self.kinds() {
            dataset.chunk_options.set_filter(kind);
        }
    }

    fn kinds(self) -> &'static [FilterKind] {
        match self {
            Self::Unfiltered => &[],
            Self::Deflate => &[FilterKind::Deflate(DeflateLevel::Six)],
            Self::ShuffleDeflate => &[FilterKind::Shuffle, FilterKind::Deflate(DeflateLevel::Six)],
        }
    }
}

/// The writers that create a dataset: [`FileBuilder::finish`], [`FileBuilder::finish_to`], and a
/// staged creation in a file opened for writing.
#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug)]
enum Writer {
    Buffered,
    Streamed,
    Staged,
}

#[cfg(feature = "std")]
impl Writer {
    fn write(self, configure: impl FnOnce(&mut DatasetBuilder)) -> Vec<u8> {
        match self {
            Self::Buffered => {
                let mut builder = FileBuilder::new();
                configure(builder.create_dataset("d"));
                builder.finish().unwrap()
            }
            Self::Streamed => {
                let mut builder = FileBuilder::new();
                configure(builder.create_dataset("d"));
                let mut bytes = Vec::new();
                builder.finish_to(&mut bytes).unwrap();
                bytes
            }
            Self::Staged => {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("staged.h5");
                FileBuilder::new().write(&path).unwrap();
                {
                    let file = File::open_rw(&path).unwrap();
                    let dataset = file.root().create_dataset("d", configure).unwrap();
                    file.commit().unwrap();
                    drop(dataset);
                }
                std::fs::read(&path).unwrap()
            }
        }
    }
}

#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug)]
enum Append {
    InPlace,
    Staged,
}

#[cfg(feature = "std")]
impl Append {
    fn append(self, path: &Path, values: &[i32]) {
        let file = File::open_rw(path).unwrap();
        let mut dataset = file.dataset("d").unwrap();
        match self {
            Self::InPlace => dataset.append(values).unwrap(),
            Self::Staged => dataset
                .append_staged(|builder| {
                    builder.append(values);
                })
                .unwrap(),
        }
        drop(dataset);
        file.close().unwrap();
    }
}

/// Returns a file whose dataset `d` holds [`EDGE_INITIAL`] in chunks of 4 with an unlimited
/// maximum extent, filtered with shuffle and deflate except at the partial edge.
#[cfg(feature = "std")]
fn edge_file() -> Vec<u8> {
    Writer::Buffered.write(|dataset| {
        dataset
            .with_i32_data(&EDGE_INITIAL)
            .with_shape(&[6])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_fill_value(EDGE_FILL);
        EdgeFilters::ShuffleDeflate.configure(dataset, &[4]);
        dataset.chunk_options.partial_edge_filtering = Some(PartialEdgeFiltering::Skip);
    })
}

/// The form a chunk of the [`edge_file`] dataset is stored in.
#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug)]
enum Stored {
    Filtered,
    FiltersSkipped,
    Raw,
}

#[cfg(feature = "std")]
impl Stored {
    /// Returns the chunk of `values` stored in this form: whether it bypasses the pipeline, its
    /// filter mask, and its stored bytes.
    fn chunk(self, values: [i32; 4]) -> (bool, u32, Vec<u8>) {
        let logical = i32_bytes(&values);
        match self {
            Self::Filtered => {
                let fill_bytes = EDGE_FILL.to_le_bytes();
                let ctx = ChunkContext::basic(&[4], 4);
                let pipeline = EdgeFilters::ShuffleDeflate
                    .options(&[4], Some(PartialEdgeFiltering::Skip))
                    .build_pipeline(&ctx, i32_fill(Some(&fill_bytes)))
                    .unwrap()
                    .unwrap();
                (
                    false,
                    0,
                    filters::compress_chunk(&logical, &pipeline, ctx).unwrap(),
                )
            }
            Self::FiltersSkipped => (false, 0b11, logical),
            Self::Raw => (true, 0, logical),
        }
    }
}

#[cfg(feature = "std")]
fn dataset_flags(dataset: &Dataset) -> ChunkedLayoutFlags {
    let DataLayout::Chunked { flags, .. } = dataset.data_layout().unwrap() else {
        panic!("the dataset is stored chunked");
    };
    flags
}

/// Returns each chunk of `dataset` in `file`: whether it bypasses the pipeline, its filter mask,
/// and its stored bytes.
#[cfg(feature = "std")]
fn file_chunks(dataset: &Dataset, file: &[u8]) -> Vec<(bool, u32, Vec<u8>)> {
    dataset
        .chunks()
        .unwrap()
        .into_iter()
        .map(|chunk| {
            let at = chunk.address.to_usize().unwrap();
            let end = at + chunk.storage_size.to_usize().unwrap();
            (
                chunk.filters().bypasses_pipeline(),
                chunk.filter_mask().raw_bits(),
                file[at..end].to_vec(),
            )
        })
        .collect()
}

/// Returns the stored bytes and the filter mask of each chunk of `set`.
fn stored_chunks(set: &CompressedChunkSet) -> Vec<(Vec<u8>, u32)> {
    set.chunks
        .iter()
        .map(|chunk| (chunk.bytes().to_vec(), chunk.filter_mask()))
        .collect()
}

/// Returns the stored bytes and the filter mask of each chunk of `raw`: the data of a chunk
/// `partial` marks, and for every other chunk the output of the pipeline of `options`, if it has
/// one.
fn expected_chunks(
    raw: &[u8],
    shape: &[u64],
    ctx: ChunkContext<'_>,
    options: &ChunkOptions,
    fill: FillPattern<'_>,
    partial: &[bool],
) -> Vec<(Vec<u8>, u32)> {
    let pipeline = options.build_pipeline(&ctx, fill).unwrap();
    let logical = split_into_chunks(
        raw,
        shape,
        ctx.chunk_dims,
        ctx.element_size.narrow().unwrap(),
        fill,
    )
    .unwrap();
    assert_eq!(logical.len(), partial.len());
    logical
        .into_iter()
        .zip(partial)
        .map(|(logical, &partial)| match (&pipeline, partial) {
            (Some(pipeline), false) => {
                (filters::compress_chunk(&logical, pipeline, ctx).unwrap(), 0)
            }
            _ => (logical, 0),
        })
        .collect()
}

fn layout_flags(layout_message: &[u8]) -> ChunkedLayoutFlags {
    let DataLayout::Chunked { flags, .. } = DataLayout::parse(layout_message, 8, 8).unwrap() else {
        panic!("a chunk set stores a chunked layout");
    };
    flags
}

/// Returns the data of `assembled` as the chunked reader reads it from a file that holds the set at
/// [`BASE`].
fn read_back(assembled: &ChunkedDataResult, shape: &[u64], fill: FillPattern<'_>) -> Vec<u8> {
    let mut file = vec![0; usize::try_from(BASE).unwrap()];
    file.extend_from_slice(&assembled.data_bytes);
    let layout = DataLayout::parse(&assembled.layout_message, 8, 8).unwrap();
    let dataspace = Dataspace {
        space_type: DataspaceType::Simple,
        rank: shape.len().try_into().unwrap(),
        dimensions: shape.to_vec(),
        max_dimensions: None,
    };
    let datatype = Datatype::FixedPoint(
        FixedPointType::new(4, DatatypeByteOrder::LittleEndian, true, 0, 32).unwrap(),
    );
    let pipeline = assembled
        .pipeline_message
        .as_ref()
        .map(|message| FilterPipeline::parse(message).unwrap());
    chunked_read::read_chunked_data_cached(
        &file,
        &source::zero_base_source(&source::BytesSource::new(&file)),
        RawReadSpec::parse(&layout, &dataspace, &datatype, pipeline.as_ref(), fill).unwrap(),
        crate::file_writer::WIDTHS,
        &ChunkCache::new(),
    )
    .unwrap()
}

fn i32_fill(bytes: Option<&[u8; 4]>) -> FillPattern<'_> {
    FillPattern::new(
        FillTime::IfSet,
        bytes.map(<[u8; 4]>::as_slice),
        convert::nz(4),
    )
}

/// Returns the values 0, 1, 2, and so on, one per element of a dataset of extent `shape`.
fn i32_elements(shape: &[u64]) -> Vec<i32> {
    (0..shape.iter().product::<u64>())
        .map(|value| i32::try_from(value).unwrap())
        .collect()
}

fn i32_element_bytes(shape: &[u64]) -> Vec<u8> {
    i32_bytes(&i32_elements(shape))
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// The address the tests lay a chunk set out at.
const BASE: u64 = 0x1000;
const EDGE_INITIAL: [i32; 6] = [100, 101, 102, 103, 104, 105];
const EDGE_FILL: i32 = -7;
