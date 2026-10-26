use std::path::Path;

use hdf5_pure::AppendOptions;
use hdf5_pure::Dataset;
use hdf5_pure::DatasetCreateProperties;
use hdf5_pure::Error;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileBuilder;
use hdf5_pure::LossyTailReencoding;
use hdf5_pure::MaxExtent;
use hdf5_pure::MemoryStrategy;
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

#[rstest]
#[case::scale_offset(Filter::ScaleOffset)]
#[cfg_attr(feature = "zfp", case::zfp(Filter::Zfp))]
fn filtered_lossy_tails_reencode_after_reopening_with_explicit_options(
    #[case] filter: Filter,
    #[values(
        AppendWithOptions::Immediate,
        AppendWithOptions::Raw,
        AppendWithOptions::Staged,
        AppendWithOptions::StagedRaw
    )]
    append: AppendWithOptions,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reencoded_tail.h5");
    filter.create(&path, PartialEdgeFiltering::Apply, &[]);
    let options = AppendOptions::new().with_lossy_tail_reencoding(LossyTailReencoding::Allow);
    append.reopen(&path, memory, AppendOptions::new(), &[1.5]);
    append.reopen(&path, memory, options, &[2.5]);
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_f64()
            .unwrap(),
        [1.5, 2.5]
    );
    append.reopen(&path, memory, options, &[1.0, 2.0, 3.5]);
    let completed = File::open(&path)
        .unwrap()
        .dataset("d")
        .unwrap()
        .chunks()
        .unwrap()[0]
        .clone();
    append.reopen(&path, memory, options, &[4.5]);
    let file = File::open(&path).unwrap();
    let dataset = file.dataset("d").unwrap();
    assert_eq!(dataset.read_f64().unwrap(), [1.5, 2.5, 1.0, 2.0, 3.5, 4.5]);
    let chunks = dataset.chunks().unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0], completed);
    assert_eq!(chunks[1].filter_mask().raw_bits(), 0);
    assert_ne!(chunks[1].storage_size, 32);
}

#[rstest]
#[case::scale_offset(Filter::ScaleOffset)]
#[cfg_attr(feature = "zfp", case::zfp(Filter::Zfp))]
fn lossy_tail_permission_applies_only_to_one_append(
    #[case] filter: Filter,
    #[values(
        AppendWithOptions::Immediate,
        AppendWithOptions::Raw,
        AppendWithOptions::Staged,
        AppendWithOptions::StagedRaw
    )]
    append: AppendWithOptions,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("scoped_permission.h5");
    filter.create(&path, PartialEdgeFiltering::Apply, &[1.5]);
    let options = AppendOptions::new().with_lossy_tail_reencoding(LossyTailReencoding::Allow);
    let expected_path = dir.path().join("permitted_append.h5");
    std::fs::copy(&path, &expected_path).unwrap();
    {
        let file = File::open_rw(&expected_path).unwrap();
        append.write(&file, options, &[2.5]).unwrap();
    }
    let before = std::fs::read(&expected_path).unwrap();
    let file = File::open_rw(&path).unwrap();
    append.write(&file, options, &[2.5]).unwrap();
    assert_lossy_rejection(
        append,
        append
            .write(&file, AppendOptions::new(), &[2.0])
            .unwrap_err(),
    );
    assert_eq!(file.dataset("d").unwrap().read_f64().unwrap(), [1.5, 2.5]);
    drop(file);
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let file = File::open_rw(&path).unwrap();
    assert_lossy_rejection(
        append,
        append
            .write(&file, AppendOptions::new(), &[2.0])
            .unwrap_err(),
    );
    file.commit().unwrap();
    assert_eq!(file.dataset("d").unwrap().read_f64().unwrap(), [1.5, 2.5]);
    drop(file);
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn staged_appends_with_lossy_tail_permission_accept_pending_and_committed_datasets() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pending_lossy.h5");
    FileBuilder::new().write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let mut dataset = file
        .root()
        .create_dataset("d", |builder| {
            builder
                .with_f64_data(&[1.5])
                .with_maxshape(&[MaxExtent::Unlimited])
                .with_chunks(&[4])
                .with_scale_offset(ScaleOffset::FloatDScale(1));
        })
        .unwrap();
    let options = AppendOptions::new().with_lossy_tail_reencoding(LossyTailReencoding::Allow);
    dataset
        .append_staged_with_options(options, |builder| {
            builder.append_f64(&[2.5]);
        })
        .unwrap();
    file.commit().unwrap();
    dataset
        .append_staged_with_options(options, |builder| {
            builder.append_f64(&[2.0]);
        })
        .unwrap();
    file.commit().unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_f64().unwrap(),
        [1.5, 2.5, 2.0]
    );
}

#[test]
fn filtered_swmr_append_rejects_lossy_tail_reencoding() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("swmr_lossy.h5");
    Filter::ScaleOffset.create(&path, PartialEdgeFiltering::Apply, &[1.5]);
    let file = File::open_swmr_writer(&path).unwrap();
    let error = file
        .dataset("d")
        .unwrap()
        .append_with_options(
            AppendOptions::new().with_lossy_tail_reencoding(LossyTailReencoding::Allow),
            &[2.5],
        )
        .unwrap_err();
    let Error::SwmrAppendUnsupported(reason) = error else {
        panic!("expected the filtered SWMR rejection, got {error:?}");
    };
    assert_eq!(
        reason,
        "filtered datasets are not supported for SWMR append"
    );
}

#[rstest]
#[case::raw_lossy_tail(true)]
#[case::lossless_tail(false)]
fn lossy_tail_options_preserve_raw_and_lossless_tail_support(#[case] raw_lossy: bool) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("other_tail.h5");
    if raw_lossy {
        Filter::ScaleOffset.create(&path, PartialEdgeFiltering::Skip, &[1.46]);
    } else {
        let mut builder = FileBuilder::new();
        builder
            .create_dataset("d")
            .with_f64_data(&[1.46])
            .with_chunks(&[4])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_fletcher32();
        builder.write(&path).unwrap();
    }
    let file = File::open_rw(&path).unwrap();
    file.dataset("d")
        .unwrap()
        .append_with_options(
            AppendOptions::new().with_lossy_tail_reencoding(LossyTailReencoding::Allow),
            &[2.56],
        )
        .unwrap();
    assert_eq!(file.dataset("d").unwrap().read_f64().unwrap(), [1.46, 2.56]);
}

#[derive(Clone, Copy, Debug)]
enum AppendWithOptions {
    Immediate,
    Raw,
    Staged,
    StagedRaw,
}

impl AppendWithOptions {
    fn reopen(self, path: &Path, memory: MemoryStrategy, options: AppendOptions, values: &[f64]) {
        let file = File::open_rw_with_options(
            path,
            FileAccessProperties::new().with_memory_strategy(memory),
        )
        .unwrap();
        self.write(&file, options, values).unwrap();
    }

    fn write(self, file: &File, options: AppendOptions, values: &[f64]) -> Result<(), Error> {
        let mut dataset = file.dataset("d")?;
        match self {
            Self::Immediate => dataset.append_with_options(options, values),
            Self::Raw => dataset.append_raw_with_options(
                options,
                &values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            Self::Staged | Self::StagedRaw => {
                dataset.append_staged_with_options(options, |builder| {
                    if matches!(self, Self::StagedRaw) {
                        builder.append_raw(
                            &values
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<_>>(),
                        );
                    } else {
                        builder.append_f64(values);
                    }
                })?;
                file.commit()
            }
        }
    }
}

fn assert_lossy_rejection(append: AppendWithOptions, error: Error) {
    let reason = match (append, &error) {
        (
            AppendWithOptions::Immediate | AppendWithOptions::Raw,
            Error::AppendInPlaceUnsupported(reason),
        )
        | (
            AppendWithOptions::Staged | AppendWithOptions::StagedRaw,
            Error::AppendUnsupported(reason),
        ) => *reason,
        _ => panic!("expected a lossy-tail rejection, got {error:?}"),
    };
    assert_eq!(
        reason,
        "this dataset's filter pipeline is lossy (ZFP, or float D-scale scale-offset), and its \
                 length is not a whole multiple of the chunk length: growing that trailing chunk would \
                 decode and re-encode values that are already committed, changing them. Append whole \
                 chunks from a chunk-aligned length instead"
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
