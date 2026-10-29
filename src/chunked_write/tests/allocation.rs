//! Allocation comparisons for raw staging, chunk encoding, and completed file output.
//!
//! Raw staging excludes traversal setup. Encoding excludes pipeline setup and includes traversal,
//! filter scratch, and retained chunks. File-output measurements start after builder setup.
//! Input buffers are allocated before measurement. Each phase is measured separately.

use core::mem;

use rstest::rstest;
use test_util::allocation;
use test_util::filtered_chunks::WORKLOADS;
use test_util::filtered_chunks::Workload;

use super::super::ChunkOptions;
use super::super::FilterKind;
use super::super::gathering::ChunkTraversal;
use crate::File;
use crate::FileBuilder;
use crate::FillTime;
use crate::chunked_write;
use crate::convert;
use crate::data_layout::ChunkedLayoutFlags;
use crate::fill_value::FillPattern;
use crate::filters;
use crate::filters::ChunkContext;
use crate::filters::FilterScratch;
use crate::prepared_chunk::ChunkEncoding;
use crate::prepared_chunk::PreparedChunk;

#[rstest]
#[case::small_shuffle(0, ["small_shuffle_raw_collected", "small_shuffle_raw_incremental", "small_shuffle_encode_collected", "small_shuffle_encode_incremental", "small_shuffle_finish"])]
#[case::small_lzf(1, ["small_lzf_raw_collected", "small_lzf_raw_incremental", "small_lzf_encode_collected", "small_lzf_encode_incremental", "small_lzf_finish"])]
#[case::incompressible_lzf(2, ["incompressible_lzf_raw_collected", "incompressible_lzf_raw_incremental", "incompressible_lzf_encode_collected", "incompressible_lzf_encode_incremental", "incompressible_lzf_finish"])]
#[case::edges_custom_fill(3, ["edges_custom_fill_raw_collected", "edges_custom_fill_raw_incremental", "edges_custom_fill_encode_collected", "edges_custom_fill_encode_incremental", "edges_custom_fill_finish"])]
#[case::full_width_shuffle(4, ["full_width_shuffle_raw_collected", "full_width_shuffle_raw_incremental", "full_width_shuffle_encode_collected", "full_width_shuffle_encode_incremental", "full_width_shuffle_finish"])]
fn filtered_creation_bounds_raw_staging_and_retains_encoded_output(
    #[case] index: usize,
    #[case] names: [&'static str; 5],
) {
    let workload = WORKLOADS[index];
    let data = workload.data();
    let Workload {
        name,
        shape,
        chunks,
        fill,
        lzf,
        random: _,
    } = workload;
    let raw = data
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    measure_staging(workload, &raw, [names[0], names[1]]);
    measure_encoding(workload, &raw, [names[2], names[3]]);
    let mut builder = FileBuilder::new();
    let dataset = builder
        .create_dataset("values")
        .with_u32_data(&data)
        .with_shape(&shape)
        .with_chunks(&chunks)
        .with_fill_value(fill)
        .with_shuffle();
    if lzf {
        dataset.with_lzf();
    }
    let (image, final_output) = allocation::measure(names[4], || builder.finish().unwrap());
    eprintln!("{}: final output {final_output}", name);
    assert_eq!(
        final_output.live_bytes,
        u64::try_from(image.capacity()).unwrap()
    );
    assert_eq!(
        File::from_bytes(image)
            .unwrap()
            .dataset("values")
            .unwrap()
            .read_u32()
            .unwrap(),
        data
    );
}

fn measure_staging(workload: Workload, raw: &[u8], names: [&'static str; 2]) {
    let Workload {
        name,
        shape,
        chunks,
        fill,
        lzf: _,
        random: _,
    } = workload;
    let fill_bytes = fill.to_le_bytes();
    let fill = FillPattern::new(FillTime::IfSet, Some(&fill_bytes), convert::nz(4));
    let mut traversal = ChunkTraversal::new(&shape, &chunks, convert::nz(4)).unwrap();
    let (collected, collected_raw) = allocation::measure(names[0], || {
        let mut collected = Vec::with_capacity(usize::try_from(traversal.chunk_count()).unwrap());
        traversal
            .visit(|chunk| {
                let mut bytes = vec![0; chunk.byte_len()];
                chunk.gather(raw, fill, &mut bytes)?;
                collected.push(bytes);
                Ok(())
            })
            .unwrap();
        collected
    });
    let staged = collected.iter().map(Vec::capacity).sum::<usize>();
    assert_eq!(
        collected_raw.live_bytes,
        u64::try_from(staged + collected.capacity() * mem::size_of::<Vec<u8>>()).unwrap()
    );
    assert_eq!(
        collected_raw.blocks,
        u64::try_from(collected.len() + 1).unwrap()
    );
    assert_eq!(collected.len(), chunk_count(workload));
    for (index, bytes) in collected.iter().enumerate() {
        assert_eq!(*bytes, expected_chunk(workload, raw, index));
    }
    drop(collected);

    let (gathered, incremental_raw) = allocation::measure(names[1], || {
        let mut gathered = Vec::new();
        traversal
            .visit(|chunk| {
                if chunk.contiguous(raw)?.is_none() {
                    gathered.resize(chunk.byte_len(), 0);
                    chunk.gather(raw, fill, &mut gathered)?;
                }
                Ok(())
            })
            .unwrap();
        gathered
    });
    let scratch = gathered.capacity();
    assert_eq!(incremental_raw.live_bytes, u64::try_from(scratch).unwrap());
    assert_eq!(incremental_raw.peak_bytes, incremental_raw.live_bytes);
    assert_eq!(incremental_raw.blocks, u64::from(scratch != 0));
    assert!(scratch <= traversal.byte_len());
    if chunks[1] == shape[1] {
        assert_eq!(scratch, 0);
    } else {
        assert_eq!(scratch, traversal.byte_len());
        assert_eq!(
            gathered,
            expected_chunk(workload, raw, chunk_count(workload) - 1)
        );
    }
    drop(gathered);

    eprintln!(
        "{}: raw payload {staged} -> {scratch} bytes; collected raw phase {collected_raw}; incremental raw phase {incremental_raw}",
        name
    );
}

fn measure_encoding(workload: Workload, raw: &[u8], names: [&'static str; 2]) {
    let Workload {
        name,
        shape,
        chunks,
        fill,
        lzf,
        random: _,
    } = workload;
    let fill_bytes = fill.to_le_bytes();
    let fill = FillPattern::new(FillTime::IfSet, Some(&fill_bytes), convert::nz(4));
    let context = ChunkContext::basic(&chunks, 4);
    let mut options = ChunkOptions::default();
    options.set_filter(FilterKind::Shuffle);
    if lzf {
        options.set_filter(FilterKind::Lzf);
    }
    let pipeline = options.build_pipeline(&context, fill).unwrap().unwrap();

    let (collected_encoded, collected_encoding) = allocation::measure(names[0], || {
        let raw_chunks =
            chunked_write::split_into_chunks(raw, &shape, &chunks, convert::nz(4), fill).unwrap();
        let mut encoded = Vec::with_capacity(raw_chunks.len());
        let mut scratch = FilterScratch::new();
        for chunk in raw_chunks {
            encoded.push(
                PreparedChunk::encode(
                    &mut scratch,
                    Some(&pipeline),
                    context,
                    ChunkEncoding::Filtered,
                    chunk,
                )
                .unwrap(),
            );
        }
        encoded
    });
    let (encoded, encoding) = allocation::measure(names[1], || {
        PreparedChunk::encode_dataset(
            &mut FilterScratch::new(),
            Some(&pipeline),
            context,
            ChunkedLayoutFlags::NONE,
            &shape,
            fill,
            raw,
        )
        .unwrap()
    });
    assert_eq!(
        encoded.iter().map(PreparedChunk::bytes).collect::<Vec<_>>(),
        collected_encoded
            .iter()
            .map(PreparedChunk::bytes)
            .collect::<Vec<_>>()
    );
    assert_eq!(encoded.len(), chunk_count(workload));
    assert_eq!(collected_encoded.len(), chunk_count(workload));
    for (index, (incremental, collected)) in encoded.iter().zip(&collected_encoded).enumerate() {
        let expected = expected_chunk(workload, raw, index);
        assert_eq!(
            filters::decompress_chunk(
                incremental.bytes(),
                &pipeline,
                context,
                incremental.filter_mask()
            )
            .unwrap(),
            expected
        );
        assert_eq!(
            filters::decompress_chunk(
                collected.bytes(),
                &pipeline,
                context,
                collected.filter_mask()
            )
            .unwrap(),
            expected
        );
    }
    let chunks = encoded.len();
    let retained = encoded
        .iter()
        .map(PreparedChunk::storage_capacity)
        .sum::<usize>()
        + encoded.capacity() * mem::size_of::<PreparedChunk>();
    assert_eq!(encoding.live_bytes, u64::try_from(retained).unwrap());
    assert!(
        encoding.blocks + u64::try_from(chunks - 1).unwrap() <= collected_encoding.blocks,
        "{encoding} against {collected_encoding}"
    );
    assert!(
        encoding.peak_bytes < collected_encoding.peak_bytes,
        "{encoding} against {collected_encoding}"
    );
    assert!(encoding.peak_bytes >= encoding.live_bytes);
    drop(collected_encoded);
    drop(encoded);

    eprintln!(
        "{}: collected encoding {collected_encoding}; incremental encoding {encoding}; retained encoded {retained} bytes",
        name
    );
}

fn chunk_count(workload: Workload) -> usize {
    usize::try_from(
        workload.shape[0].div_ceil(workload.chunks[0])
            * workload.shape[1].div_ceil(workload.chunks[1]),
    )
    .unwrap()
}

fn expected_chunk(workload: Workload, raw: &[u8], index: usize) -> Vec<u8> {
    let rows = usize::try_from(workload.shape[0]).unwrap();
    let columns = usize::try_from(workload.shape[1]).unwrap();
    let chunk_rows = usize::try_from(workload.chunks[0]).unwrap();
    let chunk_columns = usize::try_from(workload.chunks[1]).unwrap();
    let grid_columns = columns.div_ceil(chunk_columns);
    let first_row = index / grid_columns * chunk_rows;
    let first_column = index % grid_columns * chunk_columns;
    let mut expected = Vec::with_capacity(chunk_rows * chunk_columns * 4);
    for row in first_row..first_row + chunk_rows {
        for column in first_column..first_column + chunk_columns {
            if row < rows && column < columns {
                let offset = (row * columns + column) * 4;
                expected.extend_from_slice(&raw[offset..offset + 4]);
            } else {
                expected.extend_from_slice(&workload.fill.to_le_bytes());
            }
        }
    }
    expected
}
