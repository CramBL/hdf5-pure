#![cfg(feature = "__hdf5-1.10")]

use hdf5::Extent;
use hdf5::dataset::ChunkOpts;
use hdf5_pure::ChunkIndex;
use hdf5_pure::DatasetCreateProperties;
use hdf5_pure::DeflateLevel;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::Layout;
use hdf5_pure::MaxExtent;
use hdf5_pure::PartialEdgeFiltering;
use rstest::rstest;
use test_util_hdf5::file;

#[rstest]
#[case::rank_one(&[12], &[4])]
#[case::single_chunk(&[4], &[4])]
#[case::single_edge(&[3], &[4])]
#[case::full_width(&[4, 6], &[2, 6])]
#[case::gathered(&[4, 6], &[2, 3])]
#[case::edges(&[5, 7], &[2, 3])]
#[case::slabs(&[2, 3, 4], &[1, 3, 4])]
#[case::higher_rank_edges(&[3, 4, 5], &[2, 3, 2])]
#[case::many_chunks(&[4096], &[2])]
fn filtered_creation_matches_c_written_chunks_and_streaming_output(
    #[case] shape: &[u64],
    #[case] chunk_shape: &[u64],
    #[values(false, true)] unlimited: bool,
    #[values(false, true)] deflate: bool,
    #[values(PartialEdgeFiltering::Apply, PartialEdgeFiltering::Skip)] edges: PartialEdgeFiltering,
) {
    let values = (0..shape.iter().product::<u64>())
        .map(|value| i32::try_from(value).unwrap().wrapping_mul(0x0101_2345))
        .collect::<Vec<_>>();
    let fill = -37i32;
    let maximum = shape
        .iter()
        .zip(chunk_shape)
        .map(|(&dimension, &chunk)| dimension.max(chunk))
        .collect::<Vec<_>>();
    let build = || {
        let mut builder = FileBuilder::new();
        let dataset = builder
            .create_dataset("d")
            .with_create_properties(
                DatasetCreateProperties::new().with_partial_edge_filtering(edges),
            )
            .with_i32_data(&values)
            .with_shape(shape)
            .with_chunks(chunk_shape)
            .with_fill_value(fill)
            .with_shuffle();
        if deflate {
            dataset.with_deflate(DeflateLevel::Six);
        }
        dataset.with_fletcher32();
        if unlimited || maximum != shape {
            let mut maxshape = maximum
                .iter()
                .copied()
                .map(MaxExtent::Fixed)
                .collect::<Vec<_>>();
            if unlimited {
                maxshape[0] = MaxExtent::Unlimited;
            }
            dataset.with_maxshape(&maxshape);
        }
        builder
    };
    let buffered = build().finish().unwrap();
    let mut streaming = Vec::new();
    build().finish_to(&mut streaming).unwrap();
    assert_eq!(buffered, streaming);

    let dir = tempfile::tempdir().unwrap();
    let pure_path = dir.path().join("pure.h5");
    let c_path = dir.path().join("c.h5");
    std::fs::write(&pure_path, &buffered).unwrap();
    let dimensions = shape
        .iter()
        .map(|value| usize::try_from(*value).unwrap())
        .collect::<Vec<_>>();
    let chunks = chunk_shape
        .iter()
        .map(|value| usize::try_from(*value).unwrap())
        .collect::<Vec<_>>();
    {
        let c_file = file::libhdf5_create_v110(&c_path);
        let extents = dimensions
            .iter()
            .enumerate()
            .map(|(axis, dimension)| {
                if axis == 0 && unlimited {
                    Extent::resizable((*dimension).max(chunks[axis]))
                } else {
                    Extent::fixed((*dimension).max(chunks[axis]))
                }
            })
            .collect::<Vec<_>>();
        let mut dataset = c_file
            .new_dataset::<i32>()
            .shape(extents)
            .chunk(chunks.clone())
            .fill_value(fill)
            .shuffle();
        if deflate {
            dataset = dataset.deflate(DeflateLevel::Six.get());
        }
        if edges == PartialEdgeFiltering::Skip {
            dataset = dataset.chunk_opts(ChunkOpts::DONT_FILTER_PARTIAL_CHUNKS);
        }
        let dataset = dataset.fletcher32().create("d").unwrap();
        if maximum != shape {
            dataset.resize(&dimensions).unwrap();
        }
        dataset.write_raw(&values).unwrap();
        assert_eq!(dataset.read_raw::<i32>().unwrap(), values);
    }
    {
        let c_file = hdf5::File::open(&pure_path).unwrap();
        let dataset = c_file.dataset("d").unwrap();
        assert_eq!(dataset.shape(), dimensions);
        assert_eq!(dataset.chunk(), Some(chunks));
        assert_eq!(dataset.read_raw::<i32>().unwrap(), values);
        assert_eq!(
            dataset.dcpl().unwrap().get_fill_value_as::<i32>().unwrap(),
            Some(fill)
        );
    }
    let pure_file = File::from_bytes(buffered.clone()).unwrap();
    let dataset = pure_file.dataset("d").unwrap();
    assert_eq!(dataset.read_i32().unwrap(), values);
    let count = shape
        .iter()
        .zip(chunk_shape)
        .map(|(dimension, chunk)| dimension.div_ceil(*chunk))
        .product::<u64>();
    let index = if unlimited {
        ChunkIndex::ExtensibleArray
    } else if count == 1 {
        ChunkIndex::SingleChunk
    } else {
        ChunkIndex::FixedArray
    };
    assert_eq!(
        dataset.layout().unwrap(),
        Layout::Chunked {
            chunk_shape: chunk_shape.to_vec(),
            index
        }
    );
    let records = dataset.chunks().unwrap();
    assert_eq!(records.len(), usize::try_from(count).unwrap());
    let c_file = File::open(&c_path).unwrap();
    let c_dataset = c_file.dataset("d").unwrap();
    assert_eq!(c_dataset.read_i32().unwrap(), values);
    assert_eq!(c_dataset.layout().unwrap(), dataset.layout().unwrap());
    let c_records = c_dataset.chunks().unwrap();
    assert_eq!(records.len(), c_records.len());
    #[cfg(target_endian = "little")]
    let c_bytes = std::fs::read(&c_path).unwrap();
    for (actual, reference) in records.iter().zip(c_records) {
        assert_eq!(actual.offset, reference.offset);
        let raw_edge = edges == PartialEdgeFiltering::Skip
            && actual
                .offset
                .iter()
                .zip(shape)
                .zip(chunk_shape)
                .any(|((&offset, &dimension), &chunk)| chunk > dimension - offset);
        assert_eq!(actual.filters().bypasses_pipeline(), raw_edge);
        assert_eq!(reference.filters().bypasses_pipeline(), raw_edge);
        if raw_edge {
            assert_eq!(actual.storage_size, chunk_shape.iter().product::<u64>() * 4);
        }
        assert_eq!(actual.filter_mask().raw_bits(), 0);
        assert_eq!(reference.filter_mask().raw_bits(), 0);
        #[cfg(target_endian = "little")]
        if !deflate || raw_edge {
            let start = usize::try_from(actual.address).unwrap();
            let end = start + usize::try_from(actual.storage_size).unwrap();
            let reference_start = usize::try_from(reference.address).unwrap();
            let reference_end = reference_start + usize::try_from(reference.storage_size).unwrap();
            assert_eq!(
                &buffered[start..end],
                &c_bytes[reference_start..reference_end]
            );
        }
    }
}
