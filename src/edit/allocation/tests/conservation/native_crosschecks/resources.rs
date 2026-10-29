use std::fs;

use rstest::rstest;
use test_util::allocation_oracle::Coverage;
use test_util::allocation_oracle::Interval;
use test_util_hdf5::allocation_publication;
use test_util_hdf5::file_space;

use super::super::Observation;
use super::super::resources;

#[rstest]
#[case::separate(1)]
#[case::batched(250)]
fn native_reads_threshold_one_appends_with_bounded_manager_growth(#[case] batch: usize) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("threshold-one.h5");
    resources::threshold_one_workload(&path, 2000, batch);
    assert!(file_space::serialized_manager_padding(&path) > 0);
    let pure = Observation::read(&fs::read(&path).unwrap());
    let native = allocation_publication::Observation::read(&path);
    assert_eq!(native.values, pure.values);
    assert_eq!(native.chunks, pure.chunks);
    assert_eq!(
        Coverage::new(
            native
                .free
                .into_iter()
                .map(|(address, length)| { Interval::new(address, length).unwrap() })
        ),
        pure.free
    );
}
