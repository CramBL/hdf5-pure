//! Compares values, free ranges, and chunk records with libhdf5 across complete EA allocations.
//!
//! Section-list capacity cases use the HDF5 1.10 release tier.

use std::fs;

use rstest::rstest;
use test_util::allocation_oracle::Coverage;
use test_util::allocation_oracle::Interval;
use test_util_hdf5::allocation_publication;

use crate::File;

use super::super::Observation;
use super::super::Workload;
use super::super::WritePath;
use super::super::extents;

#[rstest]
#[case::immediate(WritePath::Immediate)]
#[case::owned(WritePath::Owned)]
fn native_observations_cover_complete_secondary_and_paged_storage(
    #[case] writer: WritePath,
    #[values(false, true)] paged: bool,
    #[values(false, true)] deferred: bool,
    #[values(false, true)] filtered: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native-extents.h5");
    let (address, length) = allocation_publication::write_fixture(&path, paged, true, filtered);
    assert!(!Observation::read(&fs::read(&path).unwrap()).index_defined);
    extents::compact_geometry(&path);
    if !deferred {
        allocation_publication::append(&path);
    }
    let original = Observation::read(&fs::read(&path).unwrap());
    let hole =
        Coverage::new([Interval::new(address, length).unwrap()]).intersection(&original.free);
    assert!(!hole.is_empty());
    let workload = Workload::record(&path, writer, extents::ROUNDS);
    let replay = directory.path().join("prefix.h5");
    workload.replay(&replay, &hole, |point, observation, _| {
        let native = allocation_publication::Observation::read(&replay);
        assert_eq!(
            observation.values, native.values,
            "{writer:?}/{paged}/{deferred}/{filtered}, prefix {point}"
        );
        assert_eq!(
            observation.free,
            Coverage::new(
                native
                    .free
                    .iter()
                    .map(|&(start, length)| Interval::new(start, length).unwrap())
            ),
            "{writer:?}/{paged}/{deferred}/{filtered}, prefix {point}"
        );
        assert!(
            native
                .chunks
                .iter()
                .all(|chunk| observation.chunks.contains(chunk)),
            "{writer:?}/{paged}/{deferred}/{filtered}, prefix {point}: native {:?}, stored {:?}",
            native.chunks,
            observation.chunks
        );
        if point == workload.operations.len() {
            assert_eq!(native.chunks, observation.chunks);
        }
    });
    extents::IndexStorage::read(&workload.final_bytes).assert_paths();
    let before = fs::read(&replay).unwrap();
    let reader = File::open(&replay).unwrap();
    assert_eq!(
        reader.dataset("d").unwrap().read_i32().unwrap(),
        workload.snapshots.last().unwrap().values
    );
    drop(reader);
    assert_eq!(fs::read(&replay).unwrap(), before);
}

test_util_hdf5::hdf5_1_10_series_tests! {
    #[rstest]
    #[case::existing(false)]
    #[case::deferred(true)]
    fn native_observations_preserve_unused_section_list_capacity(#[case] deferred: bool) {
        use test_util::allocation_oracle::AllocationKind;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("padded-native.h5");
        let (start, length) = allocation_publication::write_fixture(&path, false, true, false);
        extents::compact_geometry(&path);
        if !deferred {
            allocation_publication::append(&path);
        }
        let padded = extents::padded_manager(&fs::read(&path).unwrap());
        fs::write(&path, &padded).unwrap();
        let initial = Observation::read(&padded);
        assert!(initial.live.iter().any(|fact| fact.kind
            == AllocationKind::Manager
            && fact.range.length() > fact.parsed.length()));
        let native = allocation_publication::Observation::read(&path);
        assert_eq!(native.values, initial.values);
        assert_eq!(
            Coverage::new(
                native
                    .free
                    .into_iter()
                    .map(|(start, length)| Interval::new(start, length).unwrap())
            ),
            initial.free
        );
        let hole = Coverage::new([Interval::new(start, length).unwrap()]).intersection(&initial.free);
        let workload = Workload::record(&path, WritePath::Immediate, extents::ROUNDS);
        workload.replay(
            &directory.path().join("prefix.h5"),
            &hole,
            |_, observation, _| {
                let native =
                    allocation_publication::Observation::read(&directory.path().join("prefix.h5"));
                assert_eq!(observation.values, native.values);
                assert_eq!(
                    observation.free,
                    Coverage::new(
                        native
                            .free
                            .into_iter()
                            .map(|(start, length)| Interval::new(start, length).unwrap())
                    )
                );
            },
        );
    }

}
