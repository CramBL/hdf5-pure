use rstest::rstest;
use test_util_hdf5::__private::ownership_append;
use test_util_hdf5::__private::unallocated_append;
use test_util_hdf5::dataset;

use crate::File;
use crate::FileAccessProperties;
use crate::MemoryStrategy;
use crate::SyncPolicy;
use crate::edit::WriteEngine;
use crate::edit::test_util;

#[rstest]
#[case::plain(false, false)]
#[case::filtered(true, false)]
#[case::incompressible(true, true)]
fn staged_appends_keep_native_array_blocks_in_metadata_pages(
    #[case] filtered: bool,
    #[case] incompressible: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("staged-array-blocks.h5");
    let fixture = ownership_append::Fixture::write(&path, true, filtered, incompressible);
    for cycle in 0..5 {
        let file = File::open_rw_with_options(
            &path,
            FileAccessProperties::new()
                .with_memory_strategy(backing)
                .with_sync_policy(SyncPolicy::OnClose),
        )
        .unwrap();
        file.dataset("d")
            .unwrap()
            .append_staged(|append| {
                append.append_u8(&ownership_append::payload(0x3cb6_7710 + cycle));
            })
            .unwrap();
        file.commit().unwrap();
        let dataset = file.dataset("d").unwrap().header_address().unwrap();
        file.close().unwrap();
        let engine = test_util::open_engine(&path);
        engine.assert_array_metadata_pages(dataset);
        drop(engine);
        fixture.assert_contents(&path, cycle);
    }
}

#[rstest]
fn staged_first_writes_preserve_native_deferred_array_values(
    #[values(false, true)] filtered: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values([32, 4, 4, 16, 10], [24, 8, 2, 8, 8])] parameters: [u8; 5],
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("staged-deferred-array.h5");
    unallocated_append::fixture(&path, filtered, true, parameters);
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(backing)
            .with_sync_policy(SyncPolicy::OnClose),
    )
    .unwrap();
    let expected: Vec<i32> = (0..40).collect();
    file.dataset("d")
        .unwrap()
        .append_staged(|append| {
            append.append_i32(&expected);
        })
        .unwrap();
    file.commit().unwrap();
    let header = file.dataset("d").unwrap().header_address().unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), expected);
    file.close().unwrap();
    let engine = test_util::open_engine(&path);
    engine.assert_array_metadata_pages(header);
    drop(engine);
    assert_eq!(dataset::read_libhdf5::<i32>(&path, "d"), expected);
}

impl WriteEngine {
    fn assert_array_metadata_pages(&self, dataset: u64) {
        let index = self.chunked_index_spans(dataset).unwrap();
        let spans = self.chunked_storage_spans(dataset).unwrap();
        let page = self.space.page_size().unwrap().get();
        let addresses = self.context.addresses();
        assert!(index.len() >= 2);
        for allocation in &index {
            let start = addresses.relative(allocation.start()).unwrap().get();
            let last = start + allocation.length() - 1;
            if allocation.length() < page {
                assert_eq!(start / page, last / page);
            } else {
                assert_eq!(start % page, 0);
            }
            for &(raw, _) in spans.iter().filter(|(range, _)| !index.contains(range)) {
                let raw_start = addresses.relative(raw.start()).unwrap().get();
                let raw_last = raw_start + raw.length() - 1;
                assert!(last / page < raw_start / page || raw_last / page < start / page);
            }
        }
        let ownership = self.space_ownership().unwrap();
        assert!(ownership.analysis_complete);
        for allocation in index {
            assert!(
                ownership
                    .live_metadata_space
                    .iter()
                    .any(|&(start, length)| {
                        start <= allocation.start().get()
                            && allocation.end().get() <= start + length
                    })
            );
        }
    }
}
