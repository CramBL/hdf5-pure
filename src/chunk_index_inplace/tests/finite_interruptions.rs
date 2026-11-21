use std::fs;
use std::io::ErrorKind;

use test_util::disk_log::{self, FailurePoint, IoEvent, IoOutcome};

use crate::width::FormatWidths;
use crate::{
    AddressRange, Chunk, ChunkIndex, DeflateLevel, Error, File, FileBuilder, FileSpaceSettings,
    FileSpaceStrategy, MaxExtent, PhysicalAddress,
};

use super::append_records::open;

pub(super) fn allocated(widths: FormatWidths, maximum: u64, filtered: bool) -> Vec<u8> {
    let mut builder = FileBuilder::new();
    builder
        .with_offset_width(widths.offsets)
        .with_length_width(widths.lengths)
        .with_file_space_settings(storage_settings());
    let dataset = builder
        .create_dataset("d")
        .with_i32_data(&[0])
        .with_maxshape(&[MaxExtent::Fixed(maximum)])
        .with_chunks(&[4]);
    if filtered {
        dataset.with_deflate(DeflateLevel::One);
    }
    builder.finish().unwrap()
}

pub(super) fn storage_settings() -> FileSpaceSettings {
    FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
        .with_persistence(true)
        .with_threshold(1)
}

pub(super) fn chunk_storage(chunk: &Chunk) -> AddressRange<PhysicalAddress> {
    AddressRange::new(PhysicalAddress::new(chunk.address), chunk.storage_size).unwrap()
}

pub(super) fn chunk_bytes<'a>(bytes: &'a [u8], chunk: &Chunk) -> &'a [u8] {
    let range = chunk_storage(chunk);
    let start = usize::try_from(range.start().get()).unwrap();
    let end = usize::try_from(range.end().get()).unwrap();
    &bytes[start..end]
}

pub(super) fn check(
    fixture: &[u8],
    widths: FormatWidths,
    index: ChunkIndex,
    initial: &[i32],
    end: i32,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("finite.h5");
    fs::write(&path, fixture).unwrap();
    let file = open(&path);
    let base = fs::read(&path).unwrap();
    let mut dataset = file.dataset("d").unwrap();
    assert_eq!(file.superblock().offset_size, widths.offsets);
    assert_eq!(file.superblock().length_size, widths.lengths);
    assert_eq!(dataset.chunk_index().unwrap(), Some(index));
    assert_eq!(dataset.read_i32().unwrap(), initial);
    let original = dataset.chunks().unwrap().into_iter().next();
    let values = (initial.len() as i32..end).collect::<Vec<_>>();
    let complete = [initial, &values].concat();
    disk_log::start();
    dataset.append(&values).unwrap();
    let trace = disk_log::finish();
    drop(dataset);
    let completed_bytes = fs::read(&path).unwrap();
    if let Some(original) = &original {
        let free = file.reusable_free_sections().unwrap();
        assert!(
            free.sections()
                .iter()
                .any(|section| section.range().contains(chunk_storage(original)))
        );
    }
    file.close().unwrap();
    let reopened = File::open(&path).unwrap();
    assert_eq!(reopened.dataset("d").unwrap().read_i32().unwrap(), complete);
    reopened.close().unwrap();

    let mut image = base.clone();
    let mut published = false;
    let mut synchronization = 0;
    let mut prior_chunk = original.as_ref().map(chunk_storage);
    let mut tested_replacement = false;
    for event in trace.events {
        if let Some(operation) = event.operation() {
            trace.operations[operation].apply(&mut image);
            let file = File::from_bytes(image.clone()).unwrap();
            let dataset = file.dataset("d").unwrap();
            let visible = dataset.read_i32().unwrap();
            if visible == complete {
                published = true;
            } else {
                assert!(!published);
                assert_eq!(visible, initial);
            }
            assert_eq!(dataset.shape().unwrap(), [visible.len() as u64]);
            if let Some(original) = &original {
                assert_eq!(chunk_bytes(&image, original), chunk_bytes(&base, original));
            }
            fs::write(&path, &image).unwrap();
            let retry = open(&path);
            retry.dataset("d").unwrap().append(&[99_i32]).unwrap();
            retry.close().unwrap();
            let reopened = File::open(&path).unwrap();
            assert_eq!(
                reopened.dataset("d").unwrap().read_i32().unwrap(),
                [visible.as_slice(), &[99]].concat()
            );
            reopened.close().unwrap();
        }
        if let IoEvent::Synchronize { .. } = event
            && let Some(original) = &original
        {
            assert_eq!(event.outcome(), IoOutcome::Succeeded);
            let visible = File::from_bytes(image.clone()).unwrap();
            let chunks = visible.dataset("d").unwrap().chunks().unwrap();
            let current = chunk_storage(&chunks[0]);
            let previous = prior_chunk.unwrap();
            let predecessor = chunk_storage(original);
            tested_replacement |= previous == predecessor && current != predecessor && !published;

            fs::write(&path, fixture).unwrap();
            let failed = open(&path);
            assert_eq!(fs::read(&path).unwrap(), base);
            disk_log::start();
            disk_log::fail_sync_after(synchronization, FailurePoint::Before);
            let error = failed.dataset("d").unwrap().append(&values).unwrap_err();
            let failed_trace = disk_log::finish();
            disk_log::clear_failures();
            let Error::Io(error) = error else {
                panic!("expected a synchronization error, got {error:?}");
            };
            assert_eq!(error.kind(), ErrorKind::Other);
            assert_eq!(error.to_string(), "injected disk synchronization failure");
            assert_eq!(
                failed_trace.events.last().unwrap().outcome(),
                IoOutcome::Failed(ErrorKind::Other)
            );
            assert_eq!(fs::read(&path).unwrap(), image);
            let free = failed.reusable_free_sections().unwrap();
            assert!(free.sections().iter().all(|section| {
                let range = section.range();
                !range.overlaps(previous)
                    && chunks
                        .iter()
                        .all(|chunk| !range.overlaps(chunk_storage(chunk)))
            }));
            assert_eq!(
                free.sections()
                    .iter()
                    .any(|section| section.range().contains(predecessor)),
                previous != predecessor,
            );
            drop(failed);
            assert_eq!(fs::read(&path).unwrap(), image);

            let reopened = File::open(&path).unwrap();
            assert_eq!(
                reopened.dataset("d").unwrap().read_i32().unwrap(),
                if published { &complete } else { initial }
            );
            reopened.close().unwrap();
            prior_chunk = Some(current);
            synchronization += 1;
        }
    }
    assert!(published);
    assert_eq!(tested_replacement, original.is_some());
    assert_eq!(image, completed_bytes);
}
