use std::path::Path;

use hdf5_pure::{
    AppendOptions, Chunk, DatasetCreateProperties, DeflateLevel, File, FileAccessProperties,
    FileBuilder, FileSpaceSettings, FileSpaceStrategy, LossyTailReencoding, MaxExtent,
    MemoryStrategy, PartialEdgeFiltering, ScaleOffset, SyncPolicy,
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
    let path = test_util::temp::temp_path("retirement.h5");
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

#[rstest]
#[case::borrowed(false)]
#[case::owned(true)]
fn buffered_flush_modes_retire_replaced_tails(#[case] owned: bool) {
    let path = test_util::temp::temp_path("buffered-modes.h5");
    create(
        &path,
        FileSpaceStrategy::FsmAggr,
        Encoding::Unfiltered,
        &[1.0],
    );
    let file = File::open_rw(&path).unwrap();
    let mut dataset = file.dataset("d").unwrap();
    let predecessor = current_tail(&file);
    if owned {
        let mut appender = dataset.into_buffered_appender().unwrap();
        appender.append(&[2.0_f64; 10]).unwrap();
        appender.flush().unwrap();
        assert_retired_chunk_is_reusable(&file, &predecessor);
        let predecessor = current_tail(&file);
        appender.append(&[3.0_f64; 520]).unwrap();
        assert_retired_chunk_is_reusable(&file, &predecessor);
        assert_eq!(appender.buffered_elements(), 19);
        appender.flush().unwrap();
        let predecessor = current_tail(&file);
        assert_eq!(predecessor.offset, [512]);
        assert_eq!(file.dataset("d").unwrap().shape().unwrap(), [531]);
        appender.append(&[4.0_f64; 7]).unwrap();
        appender.finish().unwrap();
        assert_retired_chunk_is_reusable(&file, &predecessor);
        dataset = file.dataset("d").unwrap();
    } else {
        let mut appender = dataset.buffered_appender().unwrap();
        appender.append(&[2.0_f64; 10]).unwrap();
        appender.flush().unwrap();
        assert_retired_chunk_is_reusable(&file, &predecessor);
        let predecessor = current_tail(&file);
        appender.append(&[3.0_f64; 520]).unwrap();
        assert_retired_chunk_is_reusable(&file, &predecessor);
        assert_eq!(appender.buffered_elements(), 19);
        appender.flush().unwrap();
        let predecessor = current_tail(&file);
        assert_eq!(predecessor.offset, [512]);
        assert_eq!(file.dataset("d").unwrap().shape().unwrap(), [531]);
        appender.append(&[4.0_f64; 7]).unwrap();
        appender.finish().unwrap();
        assert_retired_chunk_is_reusable(&file, &predecessor);
    }
    let expected = [vec![1.0], vec![2.0; 10], vec![3.0; 520], vec![4.0; 7]].concat();
    assert_eq!(dataset.read_f64().unwrap(), expected);
    drop(dataset);
    file.commit().unwrap();
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("d").unwrap().read_f64().unwrap(), expected);
}

#[rstest]
fn page_buffered_replacements_survive_sync_and_reopen(
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let path = test_util::temp::temp_path("page-buffered-retirement.h5");
    create(&path, FileSpaceStrategy::Page, Encoding::Unfiltered, &[1.0]);
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(memory)
            .with_sync_policy(SyncPolicy::OnClose)
            .with_page_buffer_size(1 << 20),
    )
    .unwrap();
    let mut dataset = file.dataset("d").unwrap();
    for value in 2..130 {
        dataset.append(&[f64::from(value)]).unwrap();
    }
    let expected = (1..130).map(f64::from).collect::<Vec<_>>();
    assert_eq!(dataset.read_f64().unwrap(), expected);
    let eof = file.reusable_space_stats().unwrap().logical_size();
    dataset.append(&[130.0]).unwrap();
    assert_eq!(file.reusable_space_stats().unwrap().logical_size(), eof);
    drop(dataset);
    file.commit().unwrap();
    file.sync().unwrap();
    file.close().unwrap();
    let file = File::open_rw(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_f64().unwrap(),
        (1..131).map(f64::from).collect::<Vec<_>>()
    );
    assert!(file.reusable_space_stats().unwrap().reusable_free_bytes() > 0);
    let chunks = file.dataset("d").unwrap().chunks().unwrap();
    let sections = file.reusable_free_sections().unwrap();
    for chunk in chunks {
        for section in sections.sections() {
            assert!(
                chunk.address + chunk.storage_size <= section.address()
                    || section.end_address() <= chunk.address
            );
        }
    }
    file.close().unwrap();
}

fn current_tail(file: &File) -> Chunk {
    file.dataset("d").unwrap().chunks().unwrap().pop().unwrap()
}

fn assert_retired_chunk_is_reusable(file: &File, predecessor: &Chunk) {
    assert!(
        file.reusable_free_sections()
            .unwrap()
            .sections()
            .iter()
            .any(|section| section.address() <= predecessor.address
                && predecessor.address + predecessor.storage_size <= section.end_address()),
        "predecessor {}..{} is unavailable",
        predecessor.address,
        predecessor.address + predecessor.storage_size
    );
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
