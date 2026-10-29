use rstest::rstest;

use super::*;
use crate::chunk_cache::ChunkCache;

use crate::FillTime;
use crate::chunked_read;
use crate::convert;
use crate::data_layout::ChunkIndexLayout;
use crate::data_layout::DataLayout;
use crate::dataspace::{Dataspace, DataspaceType};
use crate::datatype::layout::FloatingPointLayout;
use crate::datatype::{Datatype, byte_order};
use crate::fill_value::FillPattern;
use crate::filters;
use crate::read_spec::RawReadSpec;

#[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
mod allocation;

mod partial_edge;
mod zfp;

fn make_f64_type() -> Datatype {
    Datatype::FloatingPoint(
        FloatingPointType::from_layout(
            8,
            byte_order::DatatypeByteOrder::LittleEndian,
            FloatingPointLayout::IEEE754_BINARY64,
        )
        .unwrap(),
    )
}

#[test]
fn an_unknown_fill_refuses_only_the_chunks_that_need_padding() {
    let elem = convert::nz(4);
    let data = vec![0u8; 8 * 4];

    // 8 elements in chunks of 4: two whole chunks, nothing uncovered.
    assert!(
        split_into_chunks(&data, &[8], &[4], elem, FillPattern::UNKNOWN).is_ok(),
        "a write that pads nothing must not consult the fill value"
    );

    // 5 elements in chunks of 4: the second chunk has three uncovered slots.
    assert!(matches!(
        split_into_chunks(&data[..5 * 4], &[5], &[4], elem, FillPattern::UNKNOWN),
        Err(FormatError::UnreadableFillValue)
    ));

    // The outer overhang requires padding whole rows.
    assert!(matches!(
        split_into_chunks(
            &data[..3 * 2 * 4],
            &[3, 2],
            &[2, 2],
            elem,
            FillPattern::UNKNOWN
        ),
        Err(FormatError::UnreadableFillValue)
    ));

    // The known patterns are unaffected either way.
    for pattern in [
        FillPattern::ZERO,
        FillPattern::new(FillTime::IfSet, Some(&[7u8; 4]), elem),
    ] {
        assert!(split_into_chunks(&data[..5 * 4], &[5], &[4], elem, pattern).is_ok());
    }
}

#[test]
fn every_split_chunk_is_a_whole_chunk() {
    let cases: &[(&[u64], &[u64])] = &[
        (&[8], &[4]),             // even
        (&[7], &[4]),             // innermost overhang
        (&[1], &[512]),           // a single, almost entirely empty chunk
        (&[4, 6], &[2, 3]),       // even, 2-D
        (&[5, 6], &[2, 3]),       // outer overhang
        (&[4, 7], &[2, 3]),       // inner overhang
        (&[5, 7], &[2, 3]),       // both
        (&[3, 5, 7], &[2, 2, 4]), // both, 3-D
    ];
    for &(shape, chunk_dims) in cases {
        for elem in [1usize, 4, 8] {
            let n: u64 = shape.iter().product();
            let raw = vec![0u8; (n as usize) * elem];
            let expected = full_chunk_bytes(chunk_dims.iter().copied(), convert::nz(elem));
            let chunks = split_into_chunks(
                &raw,
                shape,
                chunk_dims,
                convert::nz(elem),
                FillPattern::ZERO,
            )
            .unwrap();
            assert!(
                !chunks.is_empty(),
                "shape {shape:?} chunk {chunk_dims:?} must produce chunks"
            );
            for (i, c) in chunks.iter().enumerate() {
                assert_eq!(
                    c.len() as u64,
                    expected,
                    "chunk {i} of shape {shape:?} chunk {chunk_dims:?} elem {elem} \
                     is not a whole chunk"
                );
            }
        }
    }
}

#[rstest]
#[case::just_over(1 << 29)]
#[case::unallocatable(1 << 40)]
fn a_created_chunk_larger_than_u32_max_bytes_is_rejected(
    #[case] chunk_len: u64,
    #[values(StorageAllocation::Allocated, StorageAllocation::Unallocated)]
    allocation: StorageAllocation,
) {
    let chunk_dims = [chunk_len];
    let raw: &[u8] = match allocation {
        StorageAllocation::Allocated => &[0; 8],
        StorageAllocation::Unallocated => &[],
    };
    let Err(err) = compress_chunks(
        raw,
        &[1],
        ChunkContext::basic(&chunk_dims, 8),
        &ChunkOptions {
            chunk_dims: Some(chunk_dims.to_vec()),
            ..ChunkOptions::default()
        },
        None,
        FillPattern::ZERO,
        allocation,
    ) else {
        panic!("expected InvalidChunkGeometry");
    };
    assert_eq!(
        err,
        FormatError::InvalidChunkGeometry(
            "chunk logical byte size exceeds the reader's single-chunk allocation limit"
        )
    );
}

// The same geometry distinguishes allocated fill chunks from absent storage.
#[test]
fn an_unallocated_dataset_plans_no_chunks_over_the_shape_it_declares() {
    let shape = [1000u64];
    let chunk = [100u64];

    let (kind, slots, span) = plan_index_slots(
        &shape,
        &chunk,
        None,
        400,
        false,
        StorageAllocation::Allocated,
    )
    .expect("a dense fixed-shape plan");
    assert!(matches!(kind, ChunkIndexKind::FixedArray));
    assert_eq!(slots.len(), 10, "ten chunks cover the shape");
    assert_eq!(span, 10);

    let (kind, slots, span) = plan_index_slots(
        &shape,
        &chunk,
        None,
        400,
        false,
        StorageAllocation::Unallocated,
    )
    .expect("an unallocated plan");
    assert!(
        matches!(kind, ChunkIndexKind::Unallocated),
        "a fixed-shape dataset that stores nothing carries the undefined \
         address, not an index over an empty grid: {kind:?}"
    );
    assert!(slots.is_empty(), "no chunk has a slot");
    assert_eq!(span, 0, "and the index spans none");

    // A dataset that is allowed to grow keeps the growable index either
    // way, which is the same choice an empty resizable dataset gets: the
    // in-place append path needs the index to exist before the first chunk
    // arrives, so "stores nothing" does not mean "indexes nothing" here.
    let (kind, slots, _) = plan_index_slots(
        &shape,
        &chunk,
        Some(&[MaxExtent::Unlimited]),
        400,
        false,
        StorageAllocation::Unallocated,
    )
    .expect("an unallocated resizable plan");
    assert!(matches!(kind, ChunkIndexKind::ExtensibleArray), "{kind:?}");
    assert!(slots.is_empty());
}

#[rstest]
#[case::rank_one(&[19], &[4], 38)]
#[case::short_rank_one(&[19], &[4], 17)]
#[case::full_width(&[8, 6], &[2, 6], 96)]
#[case::short_full_width(&[8, 6], &[2, 6], 71)]
#[case::gathered(&[8, 6], &[2, 3], 96)]
#[case::edges(&[7, 5], &[2, 3], 70)]
#[case::short_edges(&[7, 5], &[2, 3], 33)]
#[case::slabs(&[3, 4, 5], &[1, 4, 5], 120)]
#[case::gathered_rank_three(&[3, 4, 5], &[2, 3, 2], 120)]
#[case::empty(&[0, 4], &[2, 2], 0)]
fn incremental_encoding_matches_collected_chunk_bytes(
    #[case] shape: &[u64],
    #[case] chunks: &[u64],
    #[case] input_len: usize,
    #[values(
        None,
        Some(FilterKind::Shuffle),
        Some(FilterKind::Lzf),
        Some(FilterKind::Fletcher32)
    )]
    filter: Option<FilterKind>,
    #[values(PartialEdgeFiltering::Apply, PartialEdgeFiltering::Skip)] edges: PartialEdgeFiltering,
) {
    let raw = (0..input_len)
        .map(|byte| u8::try_from(byte).unwrap())
        .collect::<Vec<_>>();
    let fill = FillPattern::new(FillTime::IfSet, Some(&[241, 242]), convert::nz(2));
    let ctx = ChunkContext::basic(chunks, 2);
    let mut options = ChunkOptions {
        partial_edge_filtering: Some(edges),
        ..ChunkOptions::default()
    };
    if let Some(filter) = filter {
        options.set_filter(filter);
    }
    let pipeline = options.build_pipeline(&ctx, fill).unwrap();
    let mut scratch = FilterScratch::new();
    let expected = split_into_chunks(&raw, shape, chunks, convert::nz(2), fill)
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(index, logical)| {
            let mut remaining = index.to_u64();
            let mut partial = false;
            for (&dimension, &chunk) in shape.iter().zip(chunks).rev() {
                let count = dimension.div_ceil(chunk);
                let offset = (remaining % count) * chunk;
                remaining /= count;
                partial |= chunk > dimension - offset;
            }
            match &pipeline {
                Some(pipeline) if edges == PartialEdgeFiltering::Apply || !partial => {
                    filters::compress_chunk_with(&mut scratch, &logical, pipeline, ctx).unwrap()
                }
                _ => logical,
            }
        })
        .collect::<Vec<_>>();
    let actual = compress_chunks(
        &raw,
        shape,
        ctx,
        &options,
        None,
        fill,
        StorageAllocation::Allocated,
    )
    .unwrap();
    assert_eq!(
        actual
            .chunks
            .iter()
            .map(PreparedChunk::bytes)
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(actual.has_filters, pipeline.is_some());
    assert_eq!(
        actual
            .chunks
            .iter()
            .map(PreparedChunk::filter_mask)
            .collect::<Vec<_>>(),
        vec![0; expected.len()]
    );
    assert_eq!(
        actual.slot_of_chunk,
        (0..expected.len().to_u64()).collect::<Vec<_>>()
    );
}

#[rstest]
#[case::empty(&[0, 4], StorageAllocation::Allocated)]
#[case::unallocated(&[7, 5], StorageAllocation::Unallocated)]
fn filtered_datasets_without_chunks_skip_gathering(
    #[case] shape: &[u64],
    #[case] allocation: StorageAllocation,
) {
    let mut options = ChunkOptions::default();
    options.set_filter(FilterKind::Shuffle);
    let set = compress_chunks(
        &[],
        shape,
        ChunkContext::basic(&[2, 2], 2),
        &options,
        None,
        FillPattern::UNKNOWN,
        allocation,
    )
    .unwrap();
    assert!(set.chunks.is_empty());
    assert!(set.has_filters);
    assert!(set.slot_of_chunk.is_empty());
}

// The fill value is 7, while splitting an empty input would materialize zero-filled chunks.
#[test]
fn an_unallocated_dataset_encodes_no_chunk_bytes() {
    let shape = [1000u64];
    let chunk = [100u64];
    let fill = [7u8, 0, 0, 0];
    let elem = NonZeroUsize::new(4).unwrap();

    let set = compress_chunks(
        &[],
        &shape,
        ChunkContext::basic(&chunk, 4),
        &ChunkOptions::default(),
        None,
        FillPattern::new(FillTime::IfSet, Some(&fill), elem),
        StorageAllocation::Unallocated,
    )
    .expect("an unallocated set");
    assert_eq!(set.chunks.len(), 0, "no chunk was encoded");
    assert_eq!(
        chunked_data_len(&set).unwrap(),
        0,
        "and the dataset occupies no data region"
    );

    let assembled = assemble_chunked_at(&set, StoredAddress::new(0x1000)).unwrap();
    assert!(assembled.data_bytes.is_empty(), "nothing to write out");
    // The undefined address represents unallocated storage in the layout message.
    assert!(
        assembled
            .layout_message
            .windows(8)
            .any(|w| w == u64::MAX.to_le_bytes()),
        "the layout message must carry the undefined address: {:?}",
        assembled.layout_message
    );

    // The same geometry with its storage allocated is the grid this avoids.
    let raw = vec![0u8; 1000 * 4];
    let dense = compress_chunks(
        &raw,
        &shape,
        ChunkContext::basic(&chunk, 4),
        &ChunkOptions::default(),
        None,
        FillPattern::new(FillTime::IfSet, Some(&fill), elem),
        StorageAllocation::Allocated,
    )
    .expect("a dense set");
    assert_eq!(dense.chunks.len(), 10);
    assert!(chunked_data_len(&dense).unwrap() >= 4000);
}

// Exercise the byte bound through index planning without serializing a file.
#[test]
fn the_index_bound_counts_unused_bytes_rather_than_slots() {
    // A dense index is never bounded, however large: every slot holds a
    // chunk, so none of its bytes describe nothing.
    let many = (MAX_UNUSED_INDEX_BYTES / 8 + 1) as usize;
    let (kind, slots, span) = plan_index_slots(
        &[many as u64],
        &[1],
        None,
        8,
        false,
        StorageAllocation::Allocated,
    )
    .expect("a dense index is not bounded");
    assert!(matches!(kind, ChunkIndexKind::FixedArray));
    assert_eq!(slots.len(), many);
    assert_eq!(span, many as u64);

    // Filtered entries are wider, so the same unused-byte bound permits fewer slots.
    let chunk_bytes = 4096;
    for has_filters in [false, true] {
        let elem = u64::from(
            hdf5_pure_format::__private::chunk_element_encoding(
                chunk_bytes,
                INDEX_OFFSET_SIZE,
                has_filters,
            )
            .elem_size as u32,
        );
        let widest = MAX_UNUSED_INDEX_BYTES / elem + 1;
        plan_index_slots(
            &[1],
            &[1],
            Some(&[MaxExtent::Fixed(widest)]),
            chunk_bytes,
            has_filters,
            StorageAllocation::Allocated,
        )
        .expect("an index exactly at the budget is written");
        let err = plan_index_slots(
            &[1],
            &[1],
            Some(&[MaxExtent::Fixed(widest + 1)]),
            chunk_bytes,
            has_filters,
            StorageAllocation::Allocated,
        )
        .unwrap_err();
        assert!(
            format!("{err}").contains("describe no chunk"),
            "filtered={has_filters}: {err}"
        );
    }

    // Room to grow along the dimension the dataset grows in costs no unused
    // slots however far it reaches, so it stays accepted.
    plan_index_slots(
        &[8, 8],
        &[4, 4],
        Some(&[MaxExtent::Unlimited, MaxExtent::Fixed(8)]),
        64,
        false,
        StorageAllocation::Allocated,
    )
    .expect("growth along the indexed dimension leaves no gaps");
}

// Two sparse chunks exercise the last addressable slot and the first slot beyond capacity.
#[test]
fn an_extensible_array_refuses_a_chunk_past_the_slots_it_can_address() {
    let capacity = hdf5_pure_format::__private::extensible_array_capacity();
    assert_eq!(
        capacity, 8_589_934_580,
        "the C library's default creation parameters address this many slots"
    );

    // Two chunks a stride apart: the second sits at slot `stride`.
    let at = |stride: u64| {
        plan_index_slots(
            &[2, 1],
            &[1, 1],
            Some(&[MaxExtent::Unlimited, MaxExtent::Fixed(stride)]),
            4,
            false,
            StorageAllocation::Allocated,
        )
    };
    at(capacity - 1).expect("the last addressable slot is written");
    let err = at(capacity).unwrap_err();
    assert!(format!("{err}").contains("can address"), "{err}");
}

#[test]
fn auto_chunking_a_zero_element_shape_is_refused() {
    let auto = ChunkOptions {
        chunk_dims: None,
        ..Default::default()
    };
    for shape in [vec![0u64], vec![4, 0], vec![0, 4]] {
        let unlimited = vec![MaxExtent::Unlimited; shape.len()];
        let extent = Extent::new(&shape, Some(&unlimited)).unwrap();
        let err = auto.validate_geometry(extent).unwrap_err();
        assert!(
            err.contains("explicit chunk dimensions"),
            "shape {shape:?}: {err}"
        );
        // The resolved dimensions are what the splitter would divide by, and
        // this is the value the guard exists to keep away from it.
        assert!(auto.resolve_chunk_dims(&shape).contains(&0));
    }

    // Named chunk dimensions make the same shape legal: it produces zero
    // chunks, which is what an extensible dataset is created as.
    let explicit = ChunkOptions {
        chunk_dims: Some(vec![512]),
        ..Default::default()
    };
    let unlimited = Extent::new(&[0], Some(&[MaxExtent::Unlimited])).unwrap();
    assert_eq!(explicit.validate_geometry(unlimited), Ok(()));
    let bounded = Extent::new(&[0], None).unwrap();
    assert_eq!(explicit.validate_geometry(bounded), Ok(()));
}

/// Sets filters through the builder switches in their canonical order.
fn filtered(filters: &[FilterKind]) -> ChunkOptions {
    let mut options = ChunkOptions::default();
    for &f in filters {
        options.set_filter(f);
    }
    options
}

/// Returns canonical filter options with explicit chunk dimensions.
fn chunked(chunk_dims: &[u64], filters: &[FilterKind]) -> ChunkOptions {
    ChunkOptions {
        chunk_dims: Some(chunk_dims.to_vec()),
        ..filtered(filters)
    }
}

fn f64_to_bytes(data: &[f64]) -> Vec<u8> {
    let mut b = Vec::with_capacity(data.len() * 8);
    for &v in data {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

fn bytes_to_f64(data: &[u8]) -> Vec<f64> {
    data.chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[rstest]
#[case(&[64], &[64], None, [16, 28], 17)]
#[case(&[128], &[32], None, [17, 17], 17)]
#[case(&[128], &[32], Some(&[MaxExtent::Unlimited][..]), [21, 21], 21)]
#[case(&[2, 3], &[2, 2], None, [18, 18], 18)]
fn header_lengths_and_filter_parameters_match_geometry(
    #[case] shape: &[u64],
    #[case] chunk_dims: &[u64],
    #[case] maxshape: Option<&[MaxExtent]>,
    #[case] allocated_lengths: [usize; 2],
    #[case] unallocated_length: usize,
    #[values(false, true)] filtered: bool,
    #[values(false, true)] varied: bool,
    #[values(StorageAllocation::Allocated, StorageAllocation::Unallocated)]
    allocation: StorageAllocation,
) {
    let mut options = ChunkOptions::default();
    if filtered {
        options.set_filter(FilterKind::Shuffle);
        options.set_filter(FilterKind::Deflate(DeflateLevel::Six));
    }
    let ctx = ChunkContext::basic(chunk_dims, 8);
    let header = options
        .measure_header(shape, ctx, maxshape, FillPattern::ZERO, allocation)
        .unwrap();
    let count = shape.iter().product::<u64>().to_usize().unwrap() * 8;
    let raw: Vec<_> = (0..count)
        .map(|index| {
            if varied {
                u8::try_from(index % 251).unwrap()
            } else {
                0
            }
        })
        .collect();
    let chunks = compress_chunks(
        &raw,
        shape,
        ctx,
        &options,
        maxshape,
        FillPattern::ZERO,
        allocation,
    )
    .unwrap();
    let measured = measure_chunked_at(&chunks, StoredAddress::new(0x1234)).unwrap();
    assert_eq!(
        header.layout_message.len(),
        match allocation {
            StorageAllocation::Allocated => allocated_lengths[usize::from(filtered)],
            StorageAllocation::Unallocated => unallocated_length,
        }
    );
    assert_eq!(
        header.pipeline_message.as_ref().map(|message| {
            FilterPipeline::parse(message)
                .unwrap()
                .filters
                .into_iter()
                .map(|filter| (filter.filter_id, filter.client_data))
                .collect::<Vec<_>>()
        }),
        filtered.then_some(vec![(FILTER_SHUFFLE, vec![8]), (FILTER_DEFLATE, vec![6])])
    );
    assert_eq!(header.layout_message.len(), measured.layout_message.len());
    assert_eq!(header.pipeline_message, measured.pipeline_message);
}

// Compare measured lengths with assembled bytes for each index kind and address.
#[test]
fn measuring_a_chunked_region_agrees_with_assembling_it() {
    /// Dataset shape, chunk shape, and maximum dimensions.
    type Case = (&'static [u64], &'static [u64], Option<&'static [MaxExtent]>);

    // Chunk counts chosen for the kind each selects: one chunk is
    // `SingleChunk`, a fixed shape with many is `FixedArray`, and an
    // unlimited maxshape is `ExtensibleArray`.
    let cases: [Case; 4] = [
        (&[512], &[512], None),
        (&[4096], &[512], None),
        (&[4096], &[64], None),
        (&[4096], &[512], Some(&[MaxExtent::Unlimited])),
    ];

    for (shape, chunk_dims, maxshape) in cases {
        let elems: usize = shape.iter().product::<u64>().to_usize().unwrap();
        let raw = f64_to_bytes(&(0..elems).map(|i| i as f64).collect::<Vec<f64>>());
        let ctx = ChunkContext::basic(chunk_dims, 8);
        let set = compress_chunks(
            &raw,
            shape,
            ctx,
            &ChunkOptions::default(),
            maxshape,
            FillPattern::ZERO,
            StorageAllocation::Allocated,
        )
        .unwrap();

        for base in [0u64, 0x1000, 0x1234_5678].map(StoredAddress::new) {
            let measured = measure_chunked_at(&set, base).unwrap();
            let assembled = assemble_chunked_at(&set, base).unwrap();
            assert_eq!(
                measured.data_len,
                assembled.data_bytes.len() as u64,
                "measured and assembled lengths differ for shape {shape:?} in \
                 chunks {chunk_dims:?} at {:#x}",
                base.get()
            );
            assert_eq!(
                measured.layout_message,
                assembled.layout_message,
                "measured and assembled layout messages differ for shape \
                 {shape:?} in chunks {chunk_dims:?} at {:#x}",
                base.get()
            );
        }
    }
}

/// Assembles encoded chunks and decodes them with [`chunked_read::read_chunked_data_cached`].
fn roundtrip_chunked(
    values: &[f64],
    shape: &[u64],
    chunk_dims: &[u64],
    options: &ChunkOptions,
) -> Vec<f64> {
    let raw = f64_to_bytes(values);
    let data_address = StoredAddress::new(0x1000);
    let ctx = ChunkContext::basic(chunk_dims, 8);
    let result = build_chunked_data_at_ext(
        &raw,
        shape,
        ctx,
        options,
        data_address,
        None,
        FillPattern::ZERO,
    )
    .unwrap();

    let file_size = data_address.get() as usize + result.data_bytes.len();
    let mut file_data = vec![0u8; file_size];
    file_data[data_address.get() as usize..].copy_from_slice(&result.data_bytes);

    let layout = DataLayout::parse(&result.layout_message, 8, 8).unwrap();
    let dataspace = Dataspace {
        space_type: DataspaceType::Simple,
        rank: shape.len() as u8,
        dimensions: shape.to_vec(),
        max_dimensions: None,
    };
    let datatype = make_f64_type();

    let pipeline = result
        .pipeline_message
        .as_ref()
        .map(|pm| crate::filter_pipeline::FilterPipeline::parse(pm).unwrap());

    let output = chunked_read::read_chunked_data_cached(
        &file_data,
        RawReadSpec::parse(
            &layout,
            &dataspace,
            &datatype,
            pipeline.as_ref(),
            FillPattern::ZERO,
        )
        .unwrap(),
        crate::file_writer::WIDTHS,
        &ChunkCache::new(),
    )
    .unwrap();

    bytes_to_f64(&output)
}

#[test]
fn split_1d_single_chunk() {
    let data = f64_to_bytes(&[1.0, 2.0, 3.0]);
    let result = split_into_chunks(&data, &[3], &[3], convert::nz(8), FillPattern::ZERO).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(bytes_to_f64(&result[0]), vec![1.0, 2.0, 3.0]);
}

#[test]
fn split_1d_multiple_chunks() {
    let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let data = f64_to_bytes(&values);
    let result = split_into_chunks(&data, &[10], &[4], convert::nz(8), FillPattern::ZERO).unwrap();
    assert_eq!(result.len(), 3); // ceil(10/4) = 3
    // Chunk `i` starts at element `4 * i`.
    assert_eq!(bytes_to_f64(&result[0]), vec![0.0, 1.0, 2.0, 3.0]);
    assert_eq!(bytes_to_f64(&result[1]), vec![4.0, 5.0, 6.0, 7.0]);
    // Last chunk: 2 valid + 2 padding zeros
    assert_eq!(bytes_to_f64(&result[2]), vec![8.0, 9.0, 0.0, 0.0]);
}

#[test]
fn split_2d_chunks() {
    // 4x4 dataset, 2x2 chunks -> 4 chunks
    let values: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let data = f64_to_bytes(&values);
    let result =
        split_into_chunks(&data, &[4, 4], &[2, 2], convert::nz(8), FillPattern::ZERO).unwrap();
    assert_eq!(result.len(), 4);
    // Row-major chunk (0, 0): elements [0, 1, 4, 5].
    assert_eq!(bytes_to_f64(&result[0]), vec![0.0, 1.0, 4.0, 5.0]);
    // chunk (0,2): elements [2,3,6,7]
    assert_eq!(bytes_to_f64(&result[1]), vec![2.0, 3.0, 6.0, 7.0]);
    // chunk (2,0): elements [8,9,12,13]
    assert_eq!(bytes_to_f64(&result[2]), vec![8.0, 9.0, 12.0, 13.0]);
    // chunk (2,2): elements [10,11,14,15]
    assert_eq!(bytes_to_f64(&result[3]), vec![10.0, 11.0, 14.0, 15.0]);
}

#[test]
fn roundtrip_1d_single_chunk_no_compression() {
    let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let options = ChunkOptions {
        chunk_dims: Some(vec![10]),
        ..Default::default()
    };
    let result = roundtrip_chunked(&values, &[10], &[10], &options);
    assert_eq!(result, values);
}

#[cfg(feature = "deflate")]
#[test]
fn roundtrip_1d_single_chunk_deflate() {
    let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let options = chunked(&[100], &[FilterKind::Deflate(DeflateLevel::Six)]);
    let result = roundtrip_chunked(&values, &[100], &[100], &options);
    assert_eq!(result, values);
}

#[test]
fn roundtrip_1d_multi_chunk_no_compression() {
    let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let options = ChunkOptions {
        chunk_dims: Some(vec![8]),
        ..Default::default()
    };
    let result = roundtrip_chunked(&values, &[20], &[8], &options);
    assert_eq!(result, values);
}

#[cfg(feature = "deflate")]
#[test]
fn roundtrip_1d_multi_chunk_deflate() {
    let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let options = chunked(&[20], &[FilterKind::Deflate(DeflateLevel::Six)]);
    let result = roundtrip_chunked(&values, &[100], &[20], &options);
    assert_eq!(result, values);
}

#[cfg(feature = "deflate")]
#[test]
fn roundtrip_1d_shuffle_deflate() {
    let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let options = chunked(
        &[50],
        &[FilterKind::Shuffle, FilterKind::Deflate(DeflateLevel::Six)],
    );
    let result = roundtrip_chunked(&values, &[100], &[50], &options);
    assert_eq!(result, values);
}

#[test]
fn roundtrip_2d_chunks() {
    // 6x4 dataset, 3x2 chunks
    let values: Vec<f64> = (0..24).map(|i| i as f64).collect();
    let options = ChunkOptions {
        chunk_dims: Some(vec![3, 2]),
        ..Default::default()
    };
    let result = roundtrip_chunked(&values, &[6, 4], &[3, 2], &options);
    assert_eq!(result, values);
}

// Seven f64 elements occupy 56 bytes, exposing padding between chunks or before the index.
#[test]
fn chunks_are_stored_back_to_back() {
    let values: Vec<f64> = (0..21).map(|i| i as f64).collect();
    let raw = f64_to_bytes(&values);
    let options = ChunkOptions {
        chunk_dims: Some(vec![7]),
        ..Default::default()
    };
    let dims = [7u64];
    let ctx = ChunkContext::basic(&dims, 8);
    let result = build_chunked_data_at_ext(
        &raw,
        &[21],
        ctx,
        &options,
        StoredAddress::new(0x1000),
        None,
        FillPattern::ZERO,
    )
    .unwrap();

    // Unfiltered chunks are stored verbatim, so the data region opens with
    // the raw bytes in order.
    assert_eq!(
        &result.data_bytes[..raw.len()],
        &raw[..],
        "the three chunks must concatenate with nothing between them"
    );
    // Check the index signature immediately after the last chunk.
    assert_eq!(
        &result.data_bytes[raw.len()..raw.len() + 4],
        b"FAHD",
        "the chunk index must begin where the last chunk ends"
    );
}

/// The same rule for the streaming path, stated where it is computed: the
/// data region is exactly the chunks plus the index, so a plan that inserted
/// padding would make `total_len` exceed the sum.
#[test]
fn a_verbatim_plan_reserves_only_the_chunks_and_the_index() {
    let meta: Vec<ChunkMeta> = [37u64, 111, 5]
        .into_iter()
        .map(|compressed_size| ChunkMeta {
            compressed_size,
            filter_mask: 0,
        })
        .collect();
    let layout = plan_chunked_data_verbatim(
        &meta,
        &[21],
        &[7],
        convert::nz(8),
        ChunkedLayoutFlags::NONE,
        Some(&[]),
        StoredAddress::new(0x1000),
        None,
    )
    .unwrap();

    let planned: Vec<u64> = layout.plan.chunks.iter().map(|c| c.stored_size).collect();
    assert_eq!(planned, vec![37, 111, 5]);

    // Compare the planned length with bytes written by an independent chunk provider.
    struct SizedChunks<'a>(&'a [u64]);
    impl ChunkProvider for SizedChunks<'_> {
        fn chunk_bytes(&self, index: usize, out: &mut Vec<u8>) -> Result<(), FormatError> {
            out.resize(self.0[index] as usize, 0xAB);
            Ok(())
        }
    }
    let sizes: Vec<u64> = meta.iter().map(|m| m.compressed_size).collect();
    let mut emitted: Vec<u8> = Vec::new();
    emit_chunked_data_verbatim(&mut emitted, &layout.plan, &SizedChunks(&sizes)).unwrap();
    assert_eq!(emitted.len() as u64, layout.plan.total_len);
    let chunk_bytes: u64 = sizes.iter().sum();
    assert!(
        layout.plan.total_len > chunk_bytes,
        "three chunks take a fixed array, so the region is longer than its chunk bytes"
    );

    // Check each index signature at its planned offset as well as the total length.
    for (label, shape, maxshape, chunk_sizes, signature) in [
        (
            "fixed array",
            &[21u64][..],
            None,
            &[37u64, 111, 5][..],
            Some(&b"FAHD"[..]),
        ),
        (
            "extensible array",
            &[21u64][..],
            Some(&[MaxExtent::Unlimited][..]),
            &[37u64, 111, 5][..],
            Some(&b"EAHD"[..]),
        ),
        // One chunk whose maximum shape is that one chunk is the
        // single-chunk layout: its address rides in the layout message and
        // nothing follows the chunk bytes at all.
        ("single chunk", &[7u64][..], None, &[37u64][..], None),
    ] {
        let meta: Vec<ChunkMeta> = chunk_sizes
            .iter()
            .map(|&compressed_size| ChunkMeta {
                compressed_size,
                filter_mask: 0,
            })
            .collect();
        let layout = plan_chunked_data_verbatim(
            &meta,
            shape,
            &[7],
            convert::nz(8),
            ChunkedLayoutFlags::NONE,
            Some(&[]),
            StoredAddress::new(0x1000),
            maxshape,
        )
        .unwrap();
        let mut emitted: Vec<u8> = Vec::new();
        emit_chunked_data_verbatim(&mut emitted, &layout.plan, &SizedChunks(chunk_sizes)).unwrap();

        let chunk_bytes: usize = chunk_sizes.iter().sum::<u64>() as usize;
        assert_eq!(
            emitted.len() as u64,
            layout.plan.total_len,
            "{label}: the emit must fill the planned region"
        );
        match signature {
            Some(sig) => assert_eq!(
                &emitted[chunk_bytes..chunk_bytes + 4],
                sig,
                "{label}: the index must begin where the last chunk ends"
            ),
            None => assert_eq!(
                emitted.len(),
                chunk_bytes,
                "{label}: nothing may follow the chunk bytes"
            ),
        }
    }
}

#[test]
fn a_verbatim_plan_with_no_chunks_is_refused() {
    let result = plan_chunked_data_verbatim(
        &[],
        &[21],
        &[7],
        convert::nz(8),
        ChunkedLayoutFlags::NONE,
        None,
        StoredAddress::new(0x1000),
        None,
    );
    assert!(
        matches!(result, Err(FormatError::ChunkedReadError(_))),
        "a chunk-less plan must be refused"
    );
}

#[test]
fn a_verbatim_chunk_too_large_for_its_index_field_fails() {
    let meta = [37, 70_000, 5].map(|compressed_size| ChunkMeta {
        compressed_size,
        filter_mask: 0,
    });
    let err = plan_chunked_data_verbatim(
        &meta,
        &[6],
        &[2],
        convert::nz(1),
        ChunkedLayoutFlags::NONE,
        Some(&[]),
        StoredAddress::new(0x1000),
        None,
    )
    .err()
    .unwrap();
    let FormatError::ChunkedReadError(message) = &err else {
        panic!("expected ChunkedReadError, got {err:?}");
    };
    assert_eq!(
        message,
        "chunk 1 of a verbatim chunked dataset stores 70000 bytes, more than the chunk index \
         records for a filtered chunk of 2 bytes"
    );
}

#[rstest]
#[case::fixed_array(&[21], None, 3)]
#[case::extensible_array(&[21], Some(&[MaxExtent::Unlimited][..]), 3)]
#[case::single_chunk(&[7], None, 1)]
fn a_verbatim_plan_stores_the_partial_edge_chunk_flag_it_is_given(
    #[case] shape: &[u64],
    #[case] maxshape: Option<&[MaxExtent]>,
    #[case] chunks: usize,
    #[values(false, true)] stored_raw: bool,
    #[values(None, Some(&[][..]))] pipeline_message: Option<&[u8]>,
) {
    let flags = if stored_raw {
        ChunkedLayoutFlags::new(DONT_FILTER_PARTIAL_BOUND_CHUNKS)
    } else {
        ChunkedLayoutFlags::NONE
    };
    let meta = vec![
        ChunkMeta {
            compressed_size: 56,
            filter_mask: 0,
        };
        chunks
    ];
    let layout = plan_chunked_data_verbatim(
        &meta,
        shape,
        &[7],
        convert::nz(8),
        flags,
        pipeline_message,
        StoredAddress::new(0x1000),
        maxshape,
    )
    .unwrap();
    let DataLayout::Chunked { flags, .. } =
        DataLayout::parse(&layout.layout_message, 8, 8).unwrap()
    else {
        panic!("a verbatim plan stores a chunked layout");
    };
    assert_eq!(flags.partial_edge_chunks_stored_raw(), stored_raw);
}

#[test]
fn chunk_options_auto_dims() {
    let options = filtered(&[FilterKind::Deflate(DeflateLevel::Six)]);
    let dims = options.resolve_chunk_dims(&[100, 50]);
    assert_eq!(dims, vec![100, 50]);
}

// Undefined fill availability requires omitting the fill value from the filter parameters.
#[test]
fn the_scale_offset_fill_availability_reaches_the_filter_parameters() {
    let ctx = f64_ctx(&[8]);
    let elem = NonZeroUsize::new(8).expect("8 is non-zero");
    let fill = 2.5f64.to_le_bytes();
    let parms = |fill_availability, fill: FillPattern<'_>| {
        let options = filtered(&[FilterKind::ScaleOffset(
            ScaleOffset::FloatDScale(2),
            fill_availability,
        )]);
        let pl = options.build_pipeline(&ctx, fill).unwrap().unwrap();
        let f = pl
            .filters
            .iter()
            .find(|f| f.filter_id == FILTER_SCALEOFFSET)
            .expect("the scale-offset filter");
        // FILAVAIL, then the two entries the value can occupy.
        (f.client_data[7], f.client_data[8], f.client_data[9])
    };

    // 2.5 as an `f64` bit pattern, split across the two entries.
    let bits = 2.5f64.to_bits();
    assert_eq!(
        parms(
            FillAvailability::Defined,
            FillPattern::new(FillTime::IfSet, Some(&fill), elem)
        ),
        (1, bits as u32, (bits >> 32) as u32)
    );
    // The library default is a defined fill value of zero.
    assert_eq!(
        parms(FillAvailability::Defined, FillPattern::ZERO),
        (1, 0, 0)
    );
    // Undefined records neither, even when a fill value is available.
    assert_eq!(
        parms(
            FillAvailability::Undefined,
            FillPattern::new(FillTime::IfSet, Some(&fill), elem)
        ),
        (0, 0, 0)
    );
    // Defined availability requires a readable fill value.
    let options = filtered(&[FilterKind::ScaleOffset(
        ScaleOffset::FloatDScale(2),
        FillAvailability::Defined,
    )]);
    assert!(matches!(
        options.build_pipeline(&ctx, FillPattern::UNKNOWN),
        Err(FormatError::UnreadableFillValue)
    ));
    // Undefined availability permits an unreadable fill value.
    let undefined = filtered(&[FilterKind::ScaleOffset(
        ScaleOffset::FloatDScale(2),
        FillAvailability::Undefined,
    )]);
    assert!(undefined.build_pipeline(&ctx, FillPattern::UNKNOWN).is_ok());

    // Check the builder default independently of native feature and target availability.
    let mut db = crate::type_builders::DatasetBuilder::new("d");
    db.with_scale_offset(ScaleOffset::FloatDScale(2));
    assert_eq!(
        db.chunk_options.filters,
        vec![FilterSpec {
            kind: FilterKind::ScaleOffset(ScaleOffset::FloatDScale(2), FillAvailability::Defined),
            optional: false,
        }]
    );
    // Deflate permits an unreadable fill value.
    let plain = filtered(&[FilterKind::Deflate(DeflateLevel::Six)]);
    assert!(plain.build_pipeline(&ctx, FillPattern::UNKNOWN).is_ok());
}

#[rstest]
#[case(DeflateLevel::Zero)]
#[case(DeflateLevel::Six)]
#[case(DeflateLevel::Nine)]
fn pipeline_preserves_deflate_level_parameters(#[case] level: DeflateLevel) {
    let options = filtered(&[FilterKind::Deflate(level)]);
    assert_eq!(options.filters[0].kind, FilterKind::Deflate(level));
    let pl = options
        .build_pipeline(&ChunkContext::basic(&[], 8), FillPattern::ZERO)
        .unwrap()
        .unwrap();
    assert_eq!(pl.filters.len(), 1);
    assert_eq!(pl.filters[0].filter_id, FILTER_DEFLATE);
    assert_eq!(pl.filters[0].client_data, [u32::from(level.get())]);
}

#[test]
fn chunk_options_pipeline_shuffle_deflate_fletcher32() {
    let options = filtered(&[
        FilterKind::Shuffle,
        FilterKind::Deflate(DeflateLevel::Six),
        FilterKind::Fletcher32,
    ]);
    let pl = options
        .build_pipeline(&ChunkContext::basic(&[], 8), FillPattern::ZERO)
        .unwrap()
        .unwrap();
    assert_eq!(pl.filters.len(), 3);
    assert_eq!(pl.filters[0].filter_id, FILTER_SHUFFLE);
    assert_eq!(pl.filters[1].filter_id, FILTER_DEFLATE);
    assert_eq!(pl.filters[2].filter_id, FILTER_FLETCHER32);
}

// Repeated setters replace the earlier level at its original position.
#[test]
fn setting_filters_places_them_in_canonical_order_whatever_order_they_arrive_in() {
    let ids = |o: &ChunkOptions| -> Vec<FilterId> {
        o.filters.iter().map(|f| f.kind.filter_id()).collect()
    };
    let canonical = [FilterId::Shuffle, FilterId::Deflate, FilterId::Fletcher32];

    assert_eq!(
        ids(&filtered(&[
            FilterKind::Shuffle,
            FilterKind::Deflate(DeflateLevel::Six),
            FilterKind::Fletcher32,
        ])),
        canonical
    );
    assert_eq!(
        ids(&filtered(&[
            FilterKind::Fletcher32,
            FilterKind::Deflate(DeflateLevel::Six),
            FilterKind::Shuffle,
        ])),
        canonical
    );

    // Repeating a filter updates its parameters and preserves a single pipeline entry.
    let twice = filtered(&[
        FilterKind::Deflate(DeflateLevel::One),
        FilterKind::Shuffle,
        FilterKind::Deflate(DeflateLevel::Nine),
    ]);
    assert_eq!(ids(&twice), [FilterId::Shuffle, FilterId::Deflate]);
    assert_eq!(
        twice.filters[1].kind,
        FilterKind::Deflate(DeflateLevel::Nine)
    );
}

// Place Fletcher32 between shuffle and deflate to make pipeline ordering observable.
#[test]
fn a_pushed_pipeline_keeps_its_order_and_its_optional_flags() {
    let mut options = ChunkOptions::default();
    for (kind, optional) in [
        (FilterKind::Shuffle, true),
        (FilterKind::Fletcher32, false),
        (FilterKind::Deflate(DeflateLevel::Four), true),
    ] {
        options.push_filter(FilterSpec { kind, optional });
    }

    let pl = options
        .build_pipeline(&ChunkContext::basic(&[], 8), FillPattern::ZERO)
        .unwrap()
        .unwrap();
    let stored: Vec<(u16, u16)> = pl.filters.iter().map(|f| (f.filter_id, f.flags)).collect();
    assert_eq!(
        stored,
        [
            (FILTER_SHUFFLE, 1),
            (FILTER_FLETCHER32, 0),
            (FILTER_DEFLATE, 1),
        ]
    );
}

#[test]
fn conflicting_filter_requests_report_both_filter_names() {
    let so = ScaleOffset::FloatDScale(2);
    let fill = FillAvailability::Defined;
    let cases: &[(&str, &str, ChunkOptions)] = &[
        (
            "lzf",
            "deflate",
            filtered(&[FilterKind::Lzf, FilterKind::Deflate(DeflateLevel::Six)]),
        ),
        (
            "shuffle",
            "scale-offset",
            filtered(&[FilterKind::Shuffle, FilterKind::ScaleOffset(so, fill)]),
        ),
    ];

    for (a, b, options) in cases {
        // Use a valid datatype so the error discriminates filter conflicts.
        let Err(err) = options.build_pipeline(&f64_ctx(&[64]), FillPattern::ZERO) else {
            panic!("{a} + {b} was accepted");
        };
        let FormatError::FilterError(msg) = &err else {
            panic!("{a} + {b}: expected a filter error, got {err}");
        };
        assert!(msg.contains(a) && msg.contains(b), "{a} + {b}: {msg}");
    }
}

/// Supplies valid floating-point metadata for ZFP and scale-offset conflict cases.
fn f64_ctx(chunk_dims: &[u64]) -> ChunkContext<'_> {
    ChunkContext {
        chunk_dims,
        element_size: core::num::NonZeroU32::new(8).expect("8 is non-zero"),
        element_type: crate::filters::zfp_element_type_from_datatype(&make_f64_type()),
        scale_offset_type: crate::scaleoffset::scale_offset_type_from_datatype(&make_f64_type()),
    }
}

#[test]
fn compatible_filter_requests_still_build() {
    let so = FilterKind::ScaleOffset(ScaleOffset::FloatDScale(2), FillAvailability::Defined);
    let cases: &[(ChunkOptions, &[u16])] = &[
        (
            filtered(&[FilterKind::Shuffle, FilterKind::Deflate(DeflateLevel::Six)]),
            &[FILTER_SHUFFLE, FILTER_DEFLATE],
        ),
        (
            filtered(&[FilterKind::Shuffle, FilterKind::Lzf]),
            &[FILTER_SHUFFLE, FILTER_LZF],
        ),
        (
            filtered(&[so, FilterKind::Deflate(DeflateLevel::Six)]),
            &[FILTER_SCALEOFFSET, FILTER_DEFLATE],
        ),
        (
            filtered(&[so, FilterKind::Lzf, FilterKind::Fletcher32]),
            &[FILTER_SCALEOFFSET, FILTER_LZF, FILTER_FLETCHER32],
        ),
    ];

    for (options, expected) in cases {
        let pl = options
            .build_pipeline(&f64_ctx(&[64]), FillPattern::ZERO)
            .unwrap()
            .unwrap();
        let ids: Vec<u16> = pl.filters.iter().map(|f| f.filter_id).collect();
        assert_eq!(&ids, expected);
    }
}

/// Assembles and decodes an extensible-array chunk index.
fn roundtrip_ea(
    values: &[f64],
    shape: &[u64],
    chunk_dims: &[u64],
    maxshape: &[MaxExtent],
) -> Vec<f64> {
    let raw = f64_to_bytes(values);
    let data_address = StoredAddress::new(0x1000);
    let options = ChunkOptions {
        chunk_dims: Some(chunk_dims.to_vec()),
        ..Default::default()
    };
    let ctx = ChunkContext::basic(chunk_dims, 8);
    let result = build_chunked_data_at_ext(
        &raw,
        shape,
        ctx,
        &options,
        data_address,
        Some(maxshape),
        FillPattern::ZERO,
    )
    .unwrap();

    let file_size = data_address.get() as usize + result.data_bytes.len();
    let mut file_data = vec![0u8; file_size];
    file_data[data_address.get() as usize..].copy_from_slice(&result.data_bytes);

    let layout = DataLayout::parse(&result.layout_message, 8, 8).unwrap();
    assert!(
        matches!(
            &layout,
            DataLayout::Chunked {
                index: ChunkIndexLayout::ExtensibleArray { .. },
                ..
            }
        ),
        "{layout:?}"
    );

    let dataspace = Dataspace {
        space_type: DataspaceType::Simple,
        rank: shape.len() as u8,
        dimensions: shape.to_vec(),
        max_dimensions: Some(maxshape.to_vec()),
    };
    let datatype = make_f64_type();

    let output = chunked_read::read_chunked_data_cached(
        &file_data,
        RawReadSpec::plain(&layout, &dataspace, &datatype),
        crate::file_writer::WIDTHS,
        &ChunkCache::new(),
    )
    .unwrap();

    bytes_to_f64(&output)
}

#[test]
fn ea_roundtrip_1d_inline_only() {
    let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let result = roundtrip_ea(&values, &[10], &[10], &[MaxExtent::Unlimited]);
    assert_eq!(result, values);
}

#[test]
fn ea_roundtrip_1d_multi_chunks() {
    let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let result = roundtrip_ea(&values, &[20], &[5], &[MaxExtent::Unlimited]);
    assert_eq!(result, values);
}

#[test]
fn ea_roundtrip_1d_many_chunks() {
    let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let result = roundtrip_ea(&values, &[100], &[10], &[MaxExtent::Unlimited]);
    assert_eq!(result, values);
}

// One element per chunk exercises inline, direct-block, and super-block storage.
#[test]
fn ea_roundtrip_super_block_sizes() {
    for &n in &[245u64, 300, 2000, 50000] {
        let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let result = roundtrip_ea(&values, &[n], &[1], &[MaxExtent::Unlimited]);
        assert_eq!(result.len(), n as usize, "length mismatch at n={n}");
        assert_eq!(result, values, "data mismatch at n={n}");
    }
}

// Paging starts at slot 131060: 4 inline, 240 direct, and super blocks 4 through 12.
// Super block 13 uses data blocks larger than 1024 elements.
#[test]
fn ea_roundtrip_paged_data_blocks() {
    let n: u64 = 132_000;
    let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let result = roundtrip_ea(&values, &[n], &[1], &[MaxExtent::Unlimited]);
    assert_eq!(result.len(), n as usize);
    assert_eq!(result, values);
}

// Compare reserved and assembled lengths across index kinds, with and without filtering.
#[test]
fn chunked_data_len_matches_what_assemble_produces() {
    for &(elements, chunk) in &[
        (1u64, 8u64), // a single chunk: no index
        (21, 7),      // three chunks: a fixed array
        (8_192, 4),   // enough chunks to page the fixed array
    ] {
        for &deflate in &[false, true] {
            for &unlimited in &[false, true] {
                let values: Vec<f64> = (0..elements).map(|i| i as f64).collect();
                let raw = f64_to_bytes(&values);
                let mut options = chunked(&[chunk], &[]);
                if deflate {
                    options.set_filter(FilterKind::Deflate(DeflateLevel::Six));
                }
                let dims = [chunk];
                let maxshape = unlimited.then_some([MaxExtent::Unlimited]);
                let set = compress_chunks(
                    &raw,
                    &[elements],
                    ChunkContext::basic(&dims, 8),
                    &options,
                    maxshape.as_ref().map(<[MaxExtent; 1]>::as_slice),
                    FillPattern::ZERO,
                    StorageAllocation::Allocated,
                )
                .unwrap();

                let planned = chunked_data_len(&set).unwrap();
                let assembled = assemble_chunked_at(&set, StoredAddress::new(0x10_0000)).unwrap();
                assert_eq!(
                    planned,
                    assembled.data_bytes.len() as u64,
                    "planned region must match the assembled one at elements={elements}, \
                     chunk={chunk}, deflate={deflate}, unlimited={unlimited}"
                );
            }
        }
    }
}

#[rstest]
#[case::single(&[64], &[64], None)]
#[case::fixed(&[256], &[32], None)]
#[case::fixed_paged(&[8_256], &[1], None)]
#[case::extensible(&[256], &[32], Some(&[MaxExtent::Unlimited][..]))]
#[case::extensible_blocks(&[2_048], &[8], Some(&[MaxExtent::Unlimited][..]))]
#[case::partial(&[3, 5], &[2, 3], None)]
#[case::maximum_grid(&[3, 5], &[2, 3], Some(&[MaxExtent::Fixed(8), MaxExtent::Fixed(12)][..]))]
#[case::empty(&[0], &[8], Some(&[MaxExtent::Unlimited][..]))]
fn direct_emission_uses_retained_chunks_and_matches_assembly(
    #[case] shape: &[u64],
    #[case] chunk_dims: &[u64],
    #[case] maxshape: Option<&[MaxExtent]>,
    #[values(false, true)] filtered: bool,
    #[values(PartialEdgeFiltering::Apply, PartialEdgeFiltering::Skip)] edges: PartialEdgeFiltering,
    #[values(StorageAllocation::Allocated, StorageAllocation::Unallocated)]
    allocation: StorageAllocation,
    #[values(0, 0x1234_5678)] address: u64,
) {
    let mut options = chunked(chunk_dims, &[]);
    if filtered {
        options.set_filter(FilterKind::Shuffle);
        options.set_filter(FilterKind::Fletcher32);
    }
    options.partial_edge_filtering = Some(edges);
    let raw: Vec<_> = (0..shape.iter().product::<u64>())
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let set = compress_chunks(
        &raw,
        shape,
        ChunkContext::basic(chunk_dims, 8),
        &options,
        maxshape,
        FillPattern::ZERO,
        allocation,
    )
    .unwrap();
    let address = StoredAddress::new(address);
    let measured = measure_chunked_at(&set, address).unwrap();
    let plan = plan_chunked_at(&set, address).unwrap();
    let assembled = assemble_chunked_at(&set, address).unwrap();
    let mut sink = ChunkSlices::default();
    emit_chunked_data(&mut sink, &set, &plan).unwrap();
    assert_eq!(sink.bytes, assembled.data_bytes);
    assert_eq!(measured.data_len, plan.data_len);
    assert_eq!(plan.data_len, sink.position());
    assert_eq!(plan.layout_message, assembled.layout_message);
    assert_eq!(plan.layout_message, measured.layout_message);
    assert_eq!(plan.pipeline_message, assembled.pipeline_message);
    assert_eq!(plan.pipeline_message, measured.pipeline_message);
    let expected: Vec<_> = set
        .chunks
        .iter()
        .map(|chunk| (chunk.bytes().as_ptr(), chunk.bytes().len()))
        .chain(core::iter::once((plan.index.as_ptr(), plan.index.len())))
        .collect();
    assert_eq!(sink.slices, expected);
}

#[derive(Default)]
struct ChunkSlices {
    slices: Vec<(*const u8, usize)>,
    bytes: Vec<u8>,
}

impl ByteSink for ChunkSlices {
    type Error = FormatError;

    fn put(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        self.slices.push((bytes.as_ptr(), bytes.len()));
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    fn put_zeros(&mut self, _: usize) -> Result<(), FormatError> {
        Err(FormatError::Internal("unexpected chunk padding".into()))
    }

    fn position(&self) -> u64 {
        self.bytes.len() as u64
    }
}

#[rstest]
#[case::payload(false)]
#[case::index(true)]
fn chunk_planning_rejects_region_address_overflow(#[case] indexed: bool) {
    let chunk_dims = if indexed { &[8][..] } else { &[16][..] };
    let set = compress_chunks(
        &[0; 16],
        &[16],
        ChunkContext::basic(chunk_dims, 1),
        &ChunkOptions::default(),
        None,
        FillPattern::ZERO,
        StorageAllocation::Allocated,
    )
    .unwrap();
    let total = chunked_data_len(&set).unwrap();
    let address = StoredAddress::new(u64::MAX - total + 1);
    let expected = FormatError::OffsetOverflow {
        offset: if indexed {
            address.get() + 16
        } else {
            address.get()
        },
        length: if indexed { total - 16 } else { 16 },
    };
    assert_eq!(measure_chunked_at(&set, address).err().unwrap(), expected);
    assert_eq!(plan_chunked_at(&set, address).err().unwrap(), expected);
    assert_eq!(assemble_chunked_at(&set, address).err().unwrap(), expected);
}

#[rstest]
#[case::first_chunk(0)]
#[case::second_chunk(1)]
#[case::index(2)]
fn chunk_emission_stops_at_the_sink_error(
    #[case] fail_at: usize,
    #[values(false, true)] verbatim: bool,
) {
    let set = compress_chunks(
        &[0; 16],
        &[16],
        ChunkContext::basic(&[8], 1),
        &ChunkOptions::default(),
        None,
        FillPattern::ZERO,
        StorageAllocation::Allocated,
    )
    .unwrap();
    let plan = plan_chunked_at(&set, StoredAddress::new(64)).unwrap();
    let mut sink = FailingChunkSink {
        accepted: 0,
        fail_at,
    };
    let error = if verbatim {
        let layout = plan_chunked_data_verbatim(
            &core::array::from_fn::<_, 2, _>(|_| ChunkMeta {
                compressed_size: 8,
                filter_mask: 0,
            }),
            &[16],
            &[8],
            convert::nz(1),
            ChunkedLayoutFlags::NONE,
            None,
            StoredAddress::new(64),
            None,
        )
        .unwrap();
        emit_chunked_data_verbatim(&mut sink, &layout.plan, &EightByteChunks).unwrap_err()
    } else {
        emit_chunked_data(&mut sink, &set, &plan).unwrap_err()
    };
    assert_eq!(error, EmissionError::Sink(ChunkSinkFailure::Write(fail_at)));
    assert_eq!(sink.accepted, fail_at);
}

struct EightByteChunks;

impl ChunkProvider for EightByteChunks {
    fn chunk_bytes(&self, _: usize, out: &mut Vec<u8>) -> Result<(), FormatError> {
        out.extend_from_slice(&[0; 8]);
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
enum ChunkSinkFailure {
    Padding,
    Write(usize),
}

struct FailingChunkSink {
    accepted: usize,
    fail_at: usize,
}

impl ByteSink for FailingChunkSink {
    type Error = ChunkSinkFailure;

    fn put(&mut self, _: &[u8]) -> Result<(), Self::Error> {
        if self.accepted == self.fail_at {
            return Err(ChunkSinkFailure::Write(self.fail_at));
        }
        self.accepted += 1;
        Ok(())
    }

    fn put_zeros(&mut self, _: usize) -> Result<(), Self::Error> {
        Err(ChunkSinkFailure::Padding)
    }

    fn position(&self) -> u64 {
        0
    }
}
