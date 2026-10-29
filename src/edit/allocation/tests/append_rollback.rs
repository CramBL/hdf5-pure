use std::fs;
use std::io::ErrorKind;

use rstest::rstest;
use test_util::disk_log;

use crate::{
    Error, File, FileAccessProperties, FileBuilder, FileSpacePageSize, FileSpaceSettings,
    FileSpaceStrategy, FreeSectionCoordinates, MaxExtent, MemoryStrategy, SyncPolicy,
};

#[rstest]
#[case::page(FileSpaceStrategy::Page)]
#[case::flat(FileSpaceStrategy::FsmAggr)]
fn unpublished_append_restores_space_before_a_larger_retry(
    #[case] strategy: FileSpaceStrategy,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] policy: SyncPolicy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("append-rollback.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(strategy)
            .with_page_size(FileSpacePageSize::try_from(PAGE_BYTES).unwrap())
            .with_persistence(true)
            .with_threshold(1),
    );
    builder
        .create_dataset("values")
        .with_i32_data(&vec![0; CHUNK_ELEMENTS])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[CHUNK_ELEMENTS as u64]);
    builder.write(&path).unwrap();
    let properties = FileAccessProperties::new()
        .with_memory_strategy(backing)
        .with_sync_policy(policy);
    let file = File::open_rw_with_options(&path, properties).unwrap();
    let mut dataset = file.dataset("values").unwrap();
    let before = fs::read(&path).unwrap();
    let observation = File::from_bytes(before.clone()).unwrap();
    let free = observation.persisted_free_sections().unwrap().unwrap();
    assert!(
        free.free_sections()
            .sections()
            .iter()
            .all(|section| section.length().get() < CHUNK_BYTES)
    );
    let old_eof = observation.file_size();
    assert_eq!(old_eof, before.len() as u64);
    if strategy == FileSpaceStrategy::Page {
        assert_eq!(old_eof % PAGE_BYTES, 0);
        assert_ne!(CHUNK_BYTES % PAGE_BYTES, 0);
    }
    drop(observation);
    disk_log::fail_write_after(0, 0);
    let error = dataset.append(&vec![1; CHUNK_ELEMENTS]).unwrap_err();
    disk_log::clear_failures();
    let Error::Io(error) = error else {
        panic!("expected the injected write error, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(dataset.shape().unwrap(), [CHUNK_ELEMENTS as u64]);
    assert_eq!(dataset.read_i32().unwrap(), vec![0; CHUNK_ELEMENTS]);

    dataset.append(&vec![2; 2 * CHUNK_ELEMENTS]).unwrap();
    let mut expected = vec![0; CHUNK_ELEMENTS];
    expected.extend(vec![2; 2 * CHUNK_ELEMENTS]);
    let retry_image = fs::read(&path).unwrap();
    let chunks = assert_image(&retry_image, &expected);
    assert!(chunks[1].0 >= old_eof);
    assert_eq!(chunks[2].0, chunks[1].1);
    if strategy == FileSpaceStrategy::Page {
        let failed_end = old_eof + CHUNK_BYTES;
        let failed_padding_end = failed_end.next_multiple_of(PAGE_BYTES);
        assert!(failed_end < failed_padding_end);
        assert!(chunks[2].0 < failed_padding_end && failed_end < chunks[2].1);
    }
    drop(dataset);
    file.close().unwrap();
    assert_image(&fs::read(&path).unwrap(), &expected);
    let file = File::open_rw_with_options(&path, properties).unwrap();
    let mut dataset = file.dataset("values").unwrap();
    assert_eq!(dataset.read_i32().unwrap(), expected);
    dataset.append(&vec![3; CHUNK_ELEMENTS]).unwrap();
    expected.extend(vec![3; CHUNK_ELEMENTS]);
    assert_image(&fs::read(&path).unwrap(), &expected);
    drop(dataset);
    file.close().unwrap();
    assert_image(&fs::read(&path).unwrap(), &expected);
}

fn assert_image(bytes: &[u8], expected: &[i32]) -> Vec<(u64, u64)> {
    let file = File::from_bytes(bytes.to_vec()).unwrap();
    let dataset = file.dataset("values").unwrap();
    assert_eq!(dataset.shape().unwrap(), [expected.len() as u64]);
    assert_eq!(dataset.read_i32().unwrap(), expected);
    let chunks = dataset.chunks().unwrap();
    assert_eq!(chunks.len(), expected.len() / CHUNK_ELEMENTS);
    let free = file.persisted_free_sections().unwrap().unwrap();
    let mut extents = Vec::new();
    for chunk in chunks {
        assert_eq!(chunk.storage_size, CHUNK_BYTES);
        let start = chunk.address;
        let end = start + chunk.storage_size;
        assert!(end <= bytes.len() as u64);
        for &(other_start, other_end) in &extents {
            assert!(start >= other_end || other_start >= end);
        }
        for section in free.free_sections().sections() {
            assert_eq!(section.coordinates(), FreeSectionCoordinates::Stored);
            assert!(section.end_address() <= bytes.len() as u64);
            assert!(
                section.address() >= end || start >= section.end_address(),
                "free section {section:?} overlaps chunk {chunk:?}"
            );
        }
        extents.push((start, end));
    }
    extents
}

const CHUNK_ELEMENTS: usize = 32_800;
const CHUNK_BYTES: u64 = CHUNK_ELEMENTS as u64 * 4;
const PAGE_BYTES: u64 = 4096;
