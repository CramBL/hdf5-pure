//! Checks ownership lifetimes across manager release and address reuse.

use std::fs;

use rstest::rstest;
use test_util::allocation_oracle;
use test_util::allocation_oracle::AllocationKind;
use test_util::allocation_oracle::AllocationState;
use test_util::allocation_oracle::Context;
use test_util::allocation_oracle::Coverage;
use test_util::allocation_oracle::Violation;

use super::Fact;
use super::Observation;
use super::Workload;
use super::WritePath;

#[test]
fn readable_control_rejects_missing_previously_released_manager_storage() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("released-manager.h5");
    super::pure_fixture(&path, false, true, false);
    let hole = Observation::read(&fs::read(&path).unwrap()).free;
    let workload = Workload::record_with(&path, 12, |values| {
        for chunk in values.chunks(super::CHUNK) {
            WritePath::Immediate.append(&path, chunk);
        }
    });
    let before = Observation::read(&workload.final_bytes);
    let released = workload
        .history
        .iter()
        .find_map(|(published, fact)| {
            let coverage = Coverage::new([fact.range]);
            let freed = (*published + 1..workload.snapshots.len()).find(|point| {
                coverage
                    .difference(&workload.snapshots[*point].free)
                    .is_empty()
            });
            (freed.is_some()
                && coverage.difference(&hole).is_empty()
                && coverage.difference(&before.free).is_empty())
            .then_some(*fact)
        })
        .unwrap_or_else(|| {
            panic!(
                "hole {hole:?}, free {:?}, history {:?}",
                before.free, workload.history
            )
        });
    let removed = Coverage::new([released.range]);
    let damaged =
        super::replace_free_sections(&workload.final_bytes, &before.free.difference(&removed));
    fs::write(&path, &damaged).unwrap();
    let observation = Observation::read(&damaged);
    assert_eq!(observation.values, before.values);
    let point = workload.operations.len();
    let mut history = workload.published_history(point);
    assert!(!history.contains(&released));
    let direct = workload.facts(point, &observation);
    let inspected = observation.allocation_facts(&mut history);
    assert_eq!(direct, inspected);
    let failure = allocation_oracle::inspect(
        Context {
            case: "missing previously released manager",
            interruption: point,
            publication: observation.state(),
        },
        &hole,
        &observation.free,
        &observation.reachable(),
        &direct,
    )
    .unwrap_err();
    let Violation::UnusedSpaceMissing { missing } = failure.violation else {
        panic!("expected unused manager storage, got {failure:?}");
    };
    assert_eq!(missing, removed);
    let file = crate::File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        before.values
    );
    drop(file);
    assert_eq!(fs::read(&path).unwrap(), damaged);
}

#[rstest]
#[case::released(false)]
#[case::reallocated(true)]
fn manager_orphan_exemption_requires_an_active_allocation_lifetime(#[case] reallocated: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("lifetime.h5");
    super::pure_fixture(&path, false, true, false);
    let bytes = fs::read(&path).unwrap();
    let mut workload = Workload::record(&path, WritePath::Immediate, 2);
    let fact = Fact::new(AllocationKind::Manager, 1_000_000, 80);
    let free = Coverage::new([fact.range]);
    let mut history = Vec::new();
    workload.snapshots.clear();
    workload.history.clear();
    workload.candidates.clear();
    for point in 0..64 {
        let mut snapshot = Observation::read(&bytes);
        snapshot.free = if point % 4 == 1 {
            free.clone()
        } else {
            Coverage::default()
        };
        snapshot.live.clear();
        if point % 4 == 0 {
            snapshot.live.push(fact);
            workload.history.push((point, fact));
            history.push(fact);
        }
        if point == 63 && reallocated {
            snapshot.live.push(fact);
            workload.history.push((point, fact));
            history.push(fact);
        }
        snapshot.allocation_facts(&mut history);
        workload.snapshots.push(snapshot);
        let retained = workload.published_history(point);
        assert_eq!(
            retained.contains(&fact),
            point % 4 == 0 || (point == 63 && reallocated)
        );
        assert_eq!(history.contains(&fact), retained.contains(&fact));
        assert_eq!(workload.retained_history(point, &[fact]), retained);
        if point % 4 == 2 {
            assert!(
                workload.snapshots[point]
                    .allocation_facts(&mut history)
                    .is_empty()
            );
        }
    }
    let mut orphan = Observation::read(&bytes);
    orphan.free = Coverage::default();
    orphan.live.clear();
    let allocations = orphan.allocation_facts(&mut history);
    assert_eq!(
        allocations,
        if reallocated {
            vec![fact.allocation(AllocationState::Orphaned)]
        } else {
            Vec::new()
        }
    );
}
