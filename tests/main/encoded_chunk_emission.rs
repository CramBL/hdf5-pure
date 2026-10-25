use std::io;
use std::io::Write;

use hdf5_pure::DatasetCreateProperties;
use hdf5_pure::DeflateLevel;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FileSpacePageSize;
use hdf5_pure::FileSpaceSettings;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::FormatError;
use hdf5_pure::MaxExtent;
use hdf5_pure::PartialEdgeFiltering;
use hdf5_pure::UserblockSize;
use rstest::rstest;

#[rstest]
#[case::single(&[32], &[32], None)]
#[case::fixed(&[257], &[16], None)]
#[case::fixed_large(&[4_097], &[128], None)]
#[case::extensible(&[257], &[16], Some(&[MaxExtent::Unlimited][..]))]
#[case::extensible_large(&[4_097], &[128], Some(&[MaxExtent::Unlimited][..]))]
#[case::partial(&[3, 5], &[2, 3], None)]
#[case::maximum_grid(&[3, 5], &[2, 3], Some(&[MaxExtent::Fixed(8), MaxExtent::Fixed(12)][..]))]
#[case::empty(&[0], &[16], Some(&[MaxExtent::Unlimited][..]))]
fn buffered_and_streaming_encoded_chunks_have_identical_bytes(
    #[case] shape: &[u64],
    #[case] chunks: &[u64],
    #[case] maxshape: Option<&[MaxExtent]>,
    #[values(
        FileSpaceStrategy::None,
        FileSpaceStrategy::Aggr,
        FileSpaceStrategy::FsmAggr,
        FileSpaceStrategy::Page
    )]
    strategy: FileSpaceStrategy,
    #[values(false, true)] filtered: bool,
    #[values(false, true)] userblock: bool,
) {
    let values: Vec<_> = (0..shape.iter().product::<u64>())
        .map(|index| i32::try_from(index).unwrap() - 19)
        .collect();
    let build = || {
        let mut builder = FileBuilder::new();
        builder.with_file_space_settings(
            FileSpaceSettings::new(strategy)
                .with_persistence(true)
                .with_page_size(FileSpacePageSize::try_from(512).unwrap()),
        );
        if userblock {
            builder.with_userblock(UserblockSize::MIN);
        }
        builder.create_dataset("prefix").with_i32_data(&[7, 9]);
        let dataset = builder.create_dataset("encoded");
        dataset
            .with_i32_data(&values)
            .with_shape(shape)
            .with_create_properties(
                DatasetCreateProperties::new()
                    .with_partial_edge_filtering(PartialEdgeFiltering::Skip),
            )
            .with_chunks(chunks)
            .with_fill_value(-37_i32);
        if let Some(maxshape) = maxshape {
            dataset.with_maxshape(maxshape);
        }
        if filtered {
            dataset
                .with_shuffle()
                .with_deflate(DeflateLevel::Three)
                .with_fletcher32();
        }
        builder
            .create_dataset("unallocated")
            .with_unallocated_storage(hdf5_pure::make_i32_type(), &[33])
            .with_chunks(&[8])
            .with_fill_value(-37_i32);
        builder.create_dataset("suffix").with_i32_data(&[11, 13]);
        builder
    };
    let buffered = build().finish().unwrap();
    let mut streamed = Vec::new();
    build().finish_to(&mut streamed).unwrap();
    assert_eq!(buffered, streamed);
    let file = File::from_bytes(buffered).unwrap();
    assert_eq!(file.dataset("encoded").unwrap().read_i32().unwrap(), values);
    assert_eq!(file.dataset("prefix").unwrap().read_i32().unwrap(), [7, 9]);
    assert_eq!(
        file.dataset("suffix").unwrap().read_i32().unwrap(),
        [11, 13]
    );
    assert_eq!(
        file.dataset("unallocated").unwrap().read_i32().unwrap(),
        [-37; 33]
    );
}

#[test]
fn encoded_chunk_geometry_errors_precede_streaming_output() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("invalid")
        .with_i32_data(&[1])
        .with_chunks(&[1 << 30]);
    let mut output = Vec::new();
    let error = builder.finish_to(&mut output).unwrap_err();
    let Error::Format(FormatError::InvalidChunkGeometry(reason)) = error else {
        panic!("expected InvalidChunkGeometry, got {error:?}");
    };
    assert_eq!(
        reason,
        "chunk logical byte size exceeds the reader's single-chunk allocation limit"
    );
    assert!(output.is_empty());
}

#[test]
fn encoded_chunk_streaming_returns_the_storage_error() {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("encoded")
        .with_i32_data(&vec![17; 65_536])
        .with_chunks(&[16_384]);
    let mut output = InterruptedOutput { accepted: 0 };
    let error = builder.finish_to(&mut output).unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected Io, got {error:?}");
    };
    assert_eq!(error.kind(), io::ErrorKind::StorageFull);
    assert!(output.accepted > 0);
    assert!(output.accepted < 65_536 * 4);
}

struct InterruptedOutput {
    accepted: usize,
}

impl Write for InterruptedOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.accepted >= 8_192 {
            return Err(io::Error::from(io::ErrorKind::StorageFull));
        }
        let count = bytes.len().min(8_192 - self.accepted);
        self.accepted += count;
        Ok(count)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
