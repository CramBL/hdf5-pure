use proptest::prelude::*;
use test_util::range;

use super::file_space;
use crate as api;

#[expect(
    dead_code,
    reason = "external crosschecks inspect the returned observations"
)]
#[path = "../../crates/test-util-hdf5/src/free_space_threshold.rs"]
mod workload;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn interior_deletions_classify_the_extents_admitted_by_the_threshold(
        persist in any::<bool>(),
        threshold in 1u64..=8192,
        datasets in prop::collection::vec((1usize..=2048usize, any::<bool>()), 3..=6),
    ) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("threshold.h5");
        workload::run(&path, persist, threshold, &datasets, |file, extent, isolated| {
            let ownership = super::ownership(file).unwrap();
            assert!(ownership.analysis_complete);
            let accounted = [
                &ownership.reusable_free_space,
                &ownership.live_metadata_space,
                &ownership.free_space_manager_space,
            ]
            .into_iter()
            .map(|sections| range::covered_len(sections, extent))
            .sum::<u64>();
            if isolated {
                let len = extent.end - extent.start;
                assert_eq!(accounted, if len >= threshold { len } else { 0 });
            }
        });
    }
}
