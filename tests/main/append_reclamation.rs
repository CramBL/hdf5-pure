use std::path::Path;

use hdf5_pure::{
    AppendOptions, DatasetCreateProperties, DeflateLevel, File, FileAccessProperties, FileBuilder,
    FileSpaceSettings, FileSpaceStrategy, LossyTailReencoding, MaxExtent, MemoryStrategy,
    PartialEdgeFiltering, ScaleOffset, SyncPolicy,
};
use rstest::rstest;

#[rstest]
#[case::typed(EntryPoint::Typed)]
#[case::raw(EntryPoint::Raw)]
#[case::borrowed(EntryPoint::Borrowed)]
#[case::owned(EntryPoint::Owned)]
fn partial_tail_retirement_bounds_unreclaimed_storage_after_deletion(#[case] entry: EntryPoint) {
    let directory = tempfile::tempdir().unwrap();
    let short = retained_after_deletion(&directory.path().join("short.h5"), 16, entry);
    let long = retained_after_deletion(&directory.path().join("long.h5"), 128, entry);
    assert!(
        long <= short + 4 * 4096,
        "unreclaimed storage grew from {short} to {long}"
    );
}

#[rstest]
#[case::unfiltered(Encoding::Unfiltered)]
#[case::deflate(Encoding::Deflate)]
#[case::lossy_apply(Encoding::LossyApply)]
#[case::lossy_skip(Encoding::LossySkip)]
fn replacements_preserve_reusable_storage_across_commit_and_reopen(
    #[case] encoding: Encoding,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] sync: SyncPolicy,
    #[values(FileSpaceStrategy::FsmAggr, FileSpaceStrategy::Page)] strategy: FileSpaceStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("retirement.h5");
    create(&path, strategy, encoding, &[1.5, 2.5, 3.5]);
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(memory)
            .with_sync_policy(sync),
    )
    .unwrap();
    let options = AppendOptions::new().with_lossy_tail_reencoding(LossyTailReencoding::Allow);
    for value in 0..37 {
        file.dataset("d")
            .unwrap()
            .append_with_options(options, &[f64::from(value)])
            .unwrap();
    }
    let expected = [vec![1.5, 2.5, 3.5], (0..37).map(f64::from).collect()].concat();
    assert_eq!(file.dataset("d").unwrap().read_f64().unwrap(), expected);
    let stats = file.reusable_space_stats().unwrap();
    assert!(stats.reusable_free_bytes() > 0);
    file.commit().unwrap();
    file.close().unwrap();
    let reopened = File::open_rw(&path).unwrap();
    assert_eq!(reopened.dataset("d").unwrap().read_f64().unwrap(), expected);
    let sections = reopened.reusable_free_sections().unwrap();
    assert_eq!(
        reopened
            .reusable_space_stats()
            .unwrap()
            .reusable_free_bytes(),
        sections.free_bytes()
    );
    reopened.root().delete("d").unwrap();
    reopened.commit().unwrap();
    reopened.close().unwrap();
}

fn retained_after_deletion(path: &Path, flushes: usize, entry: EntryPoint) -> u64 {
    create(path, FileSpaceStrategy::FsmAggr, Encoding::Unfiltered, &[]);
    let file = File::open_rw_with_options(
        path,
        FileAccessProperties::new()
            .with_memory_strategy(MemoryStrategy::Bounded)
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    let values = [0.0_f64; 64];
    let mut dataset = file.dataset("d").unwrap();
    match entry {
        EntryPoint::Borrowed => {
            let mut appender = dataset.buffered_appender().unwrap();
            for _ in 0..flushes {
                appender.append(&values).unwrap();
                appender.flush().unwrap();
            }
            appender.finish().unwrap();
        }
        EntryPoint::Owned => {
            let mut appender = dataset.into_buffered_appender().unwrap();
            for _ in 0..flushes {
                appender.append(&values).unwrap();
                appender.flush().unwrap();
            }
            appender.finish().unwrap();
            dataset = file.dataset("d").unwrap();
        }
        EntryPoint::Raw => {
            let bytes = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            for _ in 0..flushes {
                dataset.append_raw(&bytes).unwrap();
            }
        }
        EntryPoint::Typed => {
            for _ in 0..flushes {
                dataset.append(&values).unwrap();
            }
        }
    }
    assert_eq!(dataset.read_f64().unwrap(), vec![0.0; flushes * 64]);
    drop(dataset);
    file.root().delete("d").unwrap();
    file.commit().unwrap();
    let stats = file.reusable_space_stats().unwrap();
    let retained = stats.logical_size() - stats.reusable_free_bytes();
    file.close().unwrap();
    let file = File::open_rw(path).unwrap();
    let stats = file.reusable_space_stats().unwrap();
    assert!(stats.logical_size() - stats.reusable_free_bytes() <= retained + 4096);
    file.close().unwrap();
    retained
}

#[derive(Clone, Copy)]
enum EntryPoint {
    Borrowed,
    Owned,
    Raw,
    Typed,
}

#[derive(Clone, Copy)]
enum Encoding {
    Deflate,
    LossyApply,
    LossySkip,
    Unfiltered,
}

fn create(path: &Path, strategy: FileSpaceStrategy, encoding: Encoding, values: &[f64]) {
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(true));
    let dataset = builder.create_dataset("d");
    dataset
        .with_f64_data(values)
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_create_properties(
            DatasetCreateProperties::new()
                .with_chunks(&[512])
                .with_partial_edge_filtering(if matches!(encoding, Encoding::LossySkip) {
                    PartialEdgeFiltering::Skip
                } else {
                    PartialEdgeFiltering::Apply
                }),
        );
    match encoding {
        Encoding::Deflate => {
            dataset.with_deflate(DeflateLevel::Six);
        }
        Encoding::LossyApply | Encoding::LossySkip => {
            dataset.with_scale_offset(ScaleOffset::FloatDScale(1));
        }
        Encoding::Unfiltered => {}
    }
    builder.write(path).unwrap();
}
