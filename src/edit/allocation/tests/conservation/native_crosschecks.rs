//! Crosschecks allocation conservation and chunk facts against libhdf5.

use std::fs;

use rstest::rstest;
use test_util::allocation_oracle::AllocationKind;
use test_util::allocation_oracle::AllocationState;
use test_util::allocation_oracle::Coverage;
use test_util::allocation_oracle::Interval;
use test_util_hdf5::allocation_publication;

use crate::File;

use super::Observation;
use super::Workload;
use super::WritePath;

#[rstest]
#[case::immediate(WritePath::Immediate)]
#[case::borrowed(WritePath::Borrowed)]
#[case::owned(WritePath::Owned)]
fn native_append_prefixes_conserve_holes_and_agree_on_allocations(
    #[case] writer: WritePath,
    #[values(false, true)] paged: bool,
    #[values(false, true)] deferred: bool,
    #[values(false, true)] filtered: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native.h5");
    let (address, length) = allocation_publication::write_fixture(&path, paged, deferred, filtered);
    let hole = Coverage::new([Interval::new(address, length).unwrap()]);
    let original = Observation::read(&fs::read(&path).unwrap());
    assert_eq!(original.index_defined, !deferred);
    assert!(hole.difference(&original.free).is_empty());
    let workload = Workload::record(&path, writer, 5);
    let replay = directory.path().join("prefix.h5");
    let mut saw_provisional = false;
    let mut saw_orphan = false;
    workload.replay(&replay, &hole, |point, pure, allocations| {
        saw_provisional |= allocations.iter().any(|fact| {
            fact.state() == AllocationState::Provisional
                && !Coverage::new([fact.complete()])
                    .intersection(&hole)
                    .is_empty()
        });
        saw_orphan |= allocations
            .iter()
            .any(|fact| fact.state() == AllocationState::Orphaned);
        let native = allocation_publication::Observation::read(&replay);
        assert_eq!(
            pure.values, native.values,
            "{writer:?}/{paged}/{deferred}/{filtered}, prefix {point}"
        );
        assert!(
            native
                .chunks
                .iter()
                .all(|chunk| pure.chunks.contains(chunk)),
            "{writer:?}/{paged}/{deferred}/{filtered}, prefix {point}: native {:?}, reachable {:?}",
            native.chunks,
            pure.chunks
        );
        if point == workload.operations.len() {
            assert_eq!(pure.chunks, native.chunks);
        }
        let mut raw_free = native
            .free
            .into_iter()
            .map(|(address, length)| Interval::new(address, length).unwrap())
            .collect::<Vec<_>>();
        super::assert_disjoint(&mut raw_free, "native raw free sections");
        let native_free = Coverage::new(raw_free);
        assert_eq!(
            pure.free, native_free,
            "{writer:?}/{paged}/{deferred}/{filtered}, prefix {point}"
        );
    });
    assert!(
        saw_provisional,
        "{writer:?}/{paged}/{deferred}/{filtered}: no provisional hole allocation"
    );
    assert!(
        saw_orphan,
        "{writer:?}/{paged}/{deferred}/{filtered}: no owned unlinked allocation"
    );
    let final_observation = Observation::read(&workload.final_bytes);
    assert!(
        final_observation
            .live
            .iter()
            .filter(|fact| fact.kind == AllocationKind::Index)
            .count()
            > original
                .live
                .iter()
                .filter(|fact| fact.kind == AllocationKind::Index)
                .count()
    );
}

#[rstest]
#[case::existing(false)]
#[case::deferred(true)]
fn native_allocation_after_interruption_preserves_values_and_ownership(
    #[case] deferred: bool,
    #[values(false, true)] paged: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native-reuse.h5");
    let (address, length) = allocation_publication::write_fixture(&path, paged, deferred, false);
    let hole = Coverage::new([Interval::new(address, length).unwrap()]);
    let control_path = directory.path().join("native-control.h5");
    fs::copy(&path, &control_path).unwrap();
    allocation_publication::append(&control_path);
    let control = Observation::read(&fs::read(&control_path).unwrap());
    control.inspect("native-only append control", 0, &hole, &[]);
    let workload = Workload::record(&path, WritePath::Immediate, 3);
    let point = workload
        .snapshots
        .iter()
        .enumerate()
        .find_map(|(point, snapshot)| {
            (point > 0
                && snapshot.values.len() == workload.initial_values
                && snapshot.index_defined
                && workload.facts(point, snapshot).iter().any(|fact| {
                    fact.kind() == AllocationKind::Payload
                        && fact.state() == AllocationState::Orphaned
                }))
            .then_some(point)
        })
        .unwrap();
    fs::write(&path, workload.prefix(point)).unwrap();
    let observation = Observation::read(&fs::read(&path).unwrap());
    let length = 8192;
    let address = allocation_publication::allocate_contiguous(&path, length);
    assert!(
        address + u64::try_from(length).unwrap()
            <= u64::try_from(fs::read(&path).unwrap().len()).unwrap()
    );
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        observation.values
    );
    assert_eq!(
        file.dataset("native_reuse").unwrap().read_u8().unwrap(),
        vec![19; usize::try_from(length).unwrap()]
    );
    drop(file);
    let after = Observation::read(&fs::read(&path).unwrap());
    after.inspect(
        "native allocation after interruption",
        point,
        &hole,
        &workload.published_history(point),
    );
}
