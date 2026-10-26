use std::path::Path;

use hdf5_pure::Dataset;
use hdf5_pure::DatasetCreateProperties;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::MaxExtent;
use hdf5_pure::PartialEdgeFiltering;
use hdf5_pure::ScaleOffset;
#[cfg(feature = "zfp")]
use hdf5_pure::ZfpRate;
use rstest::rstest;

#[rstest]
#[case::scale_offset(Filter::ScaleOffset, &[1.5, 1.6, 1.0, 2.0, 3.46])]
#[cfg_attr(feature = "zfp", case::zfp(Filter::Zfp, &[1.4599609375, 1.5595703125, 1.0, 2.0, 3.46]))]
fn unfiltered_lossy_tails_grow_and_filter_when_complete(
    #[case] filter: Filter,
    #[case] expected: &[f64],
    #[values(Append::Buffered, Append::Immediate, Append::Staged)] append: Append,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("partial_lossy.h5");
    filter.create(&path, PartialEdgeFiltering::Skip, &[]);
    {
        let file = File::open_rw(&path).unwrap();
        append.write(&file, &[1.46]).unwrap();
        assert_raw_tail(&file.dataset("d").unwrap(), &[1.46]);
        append.write(&file, &[1.56]).unwrap();
        assert_raw_tail(&file.dataset("d").unwrap(), &[1.46, 1.56]);
    }
    let committed;
    {
        let file = File::open_rw(&path).unwrap();
        append.write(&file, &[1.0, 2.0, 3.46]).unwrap();
        let dataset = file.dataset("d").unwrap();
        let chunks = dataset.chunks().unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].filter_mask().raw_bits(), 0);
        assert_ne!(chunks[0].storage_size, 32);
        assert_eq!(
            (chunks[1].storage_size, chunks[1].filter_mask().raw_bits()),
            (32, 0)
        );
        committed = dataset.read_f64().unwrap();
        assert_eq!(committed, expected);
    }
    {
        let file = File::open_rw(&path).unwrap();
        append.write(&file, &[3.56]).unwrap();
        let actual = file.dataset("d").unwrap().read_f64().unwrap();
        assert_eq!(&actual[..4], &committed[..4]);
        assert_eq!(&actual[4..], &[3.46, 3.56]);
    }
}

#[rstest]
#[case::scale_offset(Filter::ScaleOffset)]
#[cfg_attr(feature = "zfp", case::zfp(Filter::Zfp))]
fn filtered_lossy_tails_reject_growth(
    #[case] filter: Filter,
    #[values(Append::Buffered, Append::Immediate, Append::Staged)] append: Append,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("filtered_lossy.h5");
    filter.create(&path, PartialEdgeFiltering::Apply, &[1.46, 1.56]);
    let before = std::fs::read(&path).unwrap();
    {
        let file = File::open_rw(&path).unwrap();
        let err = append.write(&file, &[1.0]).unwrap_err();
        let reason = match (&err, append) {
            (Error::AppendUnsupported(reason), Append::Staged)
            | (Error::AppendInPlaceUnsupported(reason), Append::Buffered | Append::Immediate) => {
                reason
            }
            _ => panic!("expected the lossy tail rejection, got {err:?}"),
        };
        let expected_reason = match append {
            Append::Buffered => {
                "this dataset's filter pipeline is lossy and its length is not a whole multiple \
                 of the chunk length, so a buffered appender's first write would have to \
                 re-encode already-committed values; append whole chunks from a chunk-aligned \
                 length instead"
            }
            Append::Immediate | Append::Staged => {
                "this dataset's filter pipeline is lossy (ZFP, or float D-scale scale-offset), and its \
                 length is not a whole multiple of the chunk length: growing that trailing chunk would \
                 decode and re-encode values that are already committed, changing them. Append whole \
                 chunks from a chunk-aligned length instead"
            }
        };
        assert_eq!(*reason, expected_reason);
    }
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn a_buffered_lossy_writer_continues_after_partial_flushes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("buffered_lossy.h5");
    Filter::ScaleOffset.create(&path, PartialEdgeFiltering::Skip, &[]);
    let file = File::open_rw(&path).unwrap();
    let mut dataset = file.dataset("d").unwrap();
    let mut appender = dataset.buffered_appender().unwrap();
    appender.append(&[1.46]).unwrap();
    appender.flush().unwrap();
    appender.append(&[1.56]).unwrap();
    appender.flush().unwrap();
    assert_raw_tail(&file.dataset("d").unwrap(), &[1.46, 1.56]);
    appender.append(&[1.0, 2.0, 3.46]).unwrap();
    appender.flush().unwrap();
    appender.append(&[3.56]).unwrap();
    appender.flush().unwrap();
    appender.finish().unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_f64().unwrap(),
        [1.5, 1.6, 1.0, 2.0, 3.46, 3.56]
    );
}

#[derive(Clone, Copy, Debug)]
enum Append {
    Buffered,
    Immediate,
    Staged,
}

impl Append {
    fn write(self, file: &File, values: &[f64]) -> Result<(), Error> {
        let mut dataset = file.dataset("d")?;
        match self {
            Self::Buffered => {
                let mut appender = dataset.buffered_appender()?;
                appender.append(values)?;
                appender.flush()?;
                appender.finish()
            }
            Self::Immediate => dataset.append(values),
            Self::Staged => {
                dataset.append_staged(|builder| {
                    builder.append_f64(values);
                })?;
                file.commit()
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Filter {
    ScaleOffset,
    #[cfg(feature = "zfp")]
    Zfp,
}

impl Filter {
    fn create(self, path: &Path, policy: PartialEdgeFiltering, values: &[f64]) {
        let mut builder = FileBuilder::new();
        let dataset = builder
            .create_dataset("d")
            .with_f64_data(values)
            .with_shape(&[values.len() as u64])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_create_properties(
                DatasetCreateProperties::new()
                    .with_chunks(&[4])
                    .with_partial_edge_filtering(policy),
            );
        match self {
            Self::ScaleOffset => dataset.with_scale_offset(ScaleOffset::FloatDScale(1)),
            #[cfg(feature = "zfp")]
            Self::Zfp => dataset.with_zfp(ZfpRate::try_from(16.0).unwrap()),
        };
        builder.write(path).unwrap();
    }
}

fn assert_raw_tail(dataset: &Dataset, expected: &[f64]) {
    assert_eq!(dataset.read_f64().unwrap(), expected);
    let chunks = dataset.chunks().unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(
        (chunks[0].storage_size, chunks[0].filter_mask().raw_bits()),
        (32, 0)
    );
}
