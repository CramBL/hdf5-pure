use std::path::PathBuf;

use rstest::rstest;
use tempfile::TempDir;
use test_util::disk_log;
use test_util::disk_log::IoEvent;

use crate::ChunkIndex;
use crate::File;
use crate::FileAccessProperties;
use crate::FileBuilder;
use crate::FileSpacePageSize;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::Layout;
use crate::MaxExtent;
use crate::MemoryStrategy;
use crate::PersistedFreeSpacePolicy;
use crate::SyncPolicy;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

#[cfg(test)]
test_util_hdf5::hdf5_1_10_tests! {
    mod native_crosschecks;
}

#[rstest]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::default(OffsetWidth::Eight, LengthWidth::Eight)]
fn fragmented_public_commits_preserve_widths_values_and_reusable_space(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(FileSpaceStrategy::FsmAggr, FileSpaceStrategy::Page)] strategy: FileSpaceStrategy,
    #[values(false, true)] mixed: bool,
    #[values(4, 12)] holes: usize,
) {
    let fixture = run(FormatWidths { offsets, lengths }, strategy, mixed, holes);
    assert!(fixture.path.is_file());
    fixture.directory.close().unwrap();
}

/// Runs public fragmented commit/reopen cycles and returns a closed file for native checks.
fn run(widths: FormatWidths, strategy: FileSpaceStrategy, mixed: bool, holes: usize) -> Fixture {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("fragmented.h5");
    let settings = FileSpaceSettings::new(strategy)
        .with_persistence(true)
        .with_threshold(1)
        .with_page_size(FileSpacePageSize::try_from(512).unwrap());
    let mut builder = FileBuilder::new();
    builder
        .with_offset_width(widths.offsets)
        .with_length_width(widths.lengths)
        .with_file_space_settings(settings);
    builder
        .create_dataset("d")
        .with_i32_data(&[0; 16])
        .with_chunks(&[16])
        .with_maxshape(&[MaxExtent::Fixed(16 + 8 * u64::try_from(holes).unwrap() * 4)]);
    builder
        .create_group_with("g", |group| {
            for index in 0..holes {
                group
                    .create_dataset(&format!("hole_{index:02}"))
                    .with_i32_data(&vec![-1; hole_elements(index, mixed)]);
                group
                    .create_dataset(&format!("keep_{index:02}"))
                    .with_i32_data(&[i32::try_from(index).unwrap(); 16]);
            }
            Ok(())
        })
        .unwrap();
    builder.write(&path).unwrap();
    let properties = FileAccessProperties::new()
        .with_memory_strategy(MemoryStrategy::Bounded)
        .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict)
        .with_sync_policy(SyncPolicy::OnClose);
    let file = File::open_rw_with_options(&path, properties).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().layout().unwrap(),
        Layout::Chunked {
            chunk_shape: vec![16],
            index: ChunkIndex::FixedArray
        }
    );
    let dead = (0..holes)
        .map(|index| {
            match file
                .dataset(&format!("g/hole_{index:02}"))
                .unwrap()
                .layout()
                .unwrap()
            {
                Layout::Contiguous {
                    address: Some(address),
                    size,
                } => (address, size),
                layout => panic!("expected contiguous hole storage, got {layout:?}"),
            }
        })
        .collect::<Vec<_>>();
    for index in 0..holes {
        file.group("g")
            .unwrap()
            .delete(&format!("hole_{index:02}"))
            .unwrap();
    }
    file.commit().unwrap();
    file.sync().unwrap();
    let free = file.reusable_free_sections().unwrap();
    assert!(free.section_count() >= holes);
    for &(start, length) in &dead {
        assert!(free.sections().iter().any(|section| {
            section.address().get() <= start
                && section.address().get() + section.length().get() >= start + length
        }));
    }
    file.close().unwrap();

    for cycle in 0..8 {
        let file = File::open_rw_with_options(&path, properties).unwrap();
        let batch = holes * 4;
        let values = (cycle * batch..(cycle + 1) * batch)
            .map(|value| i32::try_from(value).unwrap())
            .collect::<Vec<_>>();
        if cycle != 0 {
            file.root().delete(&format!("batch_{}", cycle - 1)).unwrap();
        }
        file.root()
            .create_dataset(&format!("batch_{cycle}"), |builder| {
                builder
                    .with_i32_data(&values)
                    .with_chunks(&[4])
                    .with_maxshape(&[MaxExtent::Fixed(u64::try_from(values.len() + 4).unwrap())]);
            })
            .unwrap();
        file.dataset("d")
            .unwrap()
            .append_staged(|append| {
                append.append_i32(&values);
            })
            .unwrap();
        file.commit().unwrap();
        file.sync().unwrap();
        let eoa = file.file_size();
        disk_log::start();
        file.commit().unwrap();
        file.sync().unwrap();
        let noop = disk_log::finish();
        assert!(
            noop.events
                .iter()
                .all(|event| !matches!(event, IoEvent::Write { written, .. } if *written != 0))
        );
        assert_eq!(file.file_size(), eoa);
        let stats = file.reusable_space_stats().unwrap();
        assert_eq!(
            stats.reusable_free_bytes(),
            file.reusable_free_sections().unwrap().free_bytes()
        );
        assert_eq!(stats.logical_size(), eoa);
        file.close().unwrap();
        let reader = File::open(&path).unwrap();
        assert_eq!(reader.superblock().offset_size, widths.offsets);
        assert_eq!(reader.superblock().length_size, widths.lengths);
        let replacement = reader.dataset(&format!("batch_{cycle}")).unwrap();
        assert_eq!(
            replacement.layout().unwrap(),
            Layout::Chunked {
                chunk_shape: vec![4],
                index: ChunkIndex::FixedArray
            }
        );
        assert_eq!(replacement.read_i32().unwrap(), values);
        drop(replacement);
        assert_eq!(
            reader.dataset("d").unwrap().read_i32().unwrap(),
            [0; 16]
                .into_iter()
                .chain(0..i32::try_from((cycle + 1) * batch).unwrap())
                .collect::<Vec<_>>()
        );
        for index in 0..holes {
            assert_eq!(
                reader
                    .dataset(&format!("g/keep_{index:02}"))
                    .unwrap()
                    .read_i32()
                    .unwrap(),
                [i32::try_from(index).unwrap(); 16]
            );
        }
        let mut names = reader.group("g").unwrap().datasets().unwrap();
        names.sort();
        assert_eq!(
            names,
            (0..holes)
                .map(|index| format!("keep_{index:02}"))
                .collect::<Vec<_>>()
        );
        reader.close().unwrap();
    }
    Fixture { directory, path }
}

fn hole_elements(index: usize, mixed: bool) -> usize {
    if mixed && index % 2 == 1 { 32 } else { 16 }
}

/// A closed file and its temporary directory.
struct Fixture {
    directory: TempDir,
    path: PathBuf,
}
