use std::ops::Range;
use std::panic::{self, AssertUnwindSafe};

use rstest::rstest;
use test_util_hdf5::dataset;

use self::workload::{Base, Lifecycle, Strategy};
use super::file_space;
use crate as api;
use crate::File;
use crate::edit::SpaceOwnership;

#[expect(
    dead_code,
    reason = "the shared workload has entry points used only by external tests"
)]
#[path = "../../crates/test-util-hdf5/src/free_space_churn.rs"]
mod workload;

#[rstest]
#[case::continuous_fsm(Lifecycle::ContinuousOpen, Strategy::FsmAggr)]
#[case::continuous_page(Lifecycle::ContinuousOpen, Strategy::Page)]
#[case::final_fsm(Lifecycle::ReopenBeforeFinalCheck, Strategy::FsmAggr)]
#[case::final_page(Lifecycle::ReopenBeforeFinalCheck, Strategy::Page)]
#[case::frequent_fsm(Lifecycle::ReopenFrequently, Strategy::FsmAggr)]
fn churn_partitions_committed_closed_and_reopened_storage(
    #[case] lifecycle: Lifecycle,
    #[case] strategy: Strategy,
    #[values(Base::Zero, Base::Userblock)] base: Base,
) {
    let mut steps = Vec::new();
    let mut observed = Vec::new();
    let mut sessions = String::new();
    let mut observations = String::new();
    workload::run_churn_with_checkpoints(
        lifecycle,
        strategy,
        base,
        |path, step| {
            panic::catch_unwind(AssertUnwindSafe(|| {
                super::with_reopened_file(path, |file| {
                    assert_ownership(file, lifecycle, strategy, base, step)
                        .write_checkpoint(&mut observations, step);
                });
            }))
            .unwrap_or_else(|_| {
                panic!(
                    "closed-file ownership assertion failed for {lifecycle:?}/{strategy:?}/{base:?}/{step}"
                )
            });
            observed.push(step.to_owned());
        },
        |file, step| {
            panic::catch_unwind(AssertUnwindSafe(|| {
                assert_ownership(file, lifecycle, strategy, base, step)
                    .write_checkpoint(&mut sessions, step);
            }))
            .unwrap_or_else(|_| {
                panic!("ownership assertion failed for {lifecycle:?}/{strategy:?}/{base:?}/{step}")
            });
            steps.push(step.to_owned());
        },
    );
    if cfg!(not(feature = "fast-deflate")) {
        insta::assert_snapshot!(
            format!("churn_{lifecycle:?}_{strategy:?}_{base:?}"),
            format!(
                "Editing checkpoints (category values are bytes/extents):\n{sessions}\nClosed-file observations:\n{observations}"
            )
        );
    }
    assert_eq!(steps.first().unwrap(), "initial-commit");
    assert_eq!(&steps[steps.len() - 2..], ["final-commit", "final-close"]);
    assert!(
        steps
            .iter()
            .filter(|step| step.starts_with("deletion-wave-"))
            .count()
            > 1
    );
    let closed = steps
        .iter()
        .filter_map(|step| step.strip_prefix("closed-"))
        .collect::<Vec<_>>();
    let reopened = steps
        .iter()
        .filter_map(|step| step.strip_prefix("reopened-"))
        .collect::<Vec<_>>();
    assert_eq!(closed, reopened);
    assert_eq!(observed.len(), reopened.len() + 1);
    assert_eq!(observed.last().unwrap(), "final-close");
    match lifecycle {
        Lifecycle::ContinuousOpen => assert!(reopened.is_empty()),
        Lifecycle::ReopenBeforeFinalCheck => {
            assert_eq!(reopened.len(), 1);
            assert!(reopened[0].starts_with("final-check-"));
        }
        Lifecycle::ReopenFrequently => {
            assert!(reopened.len() > 2);
            assert!(reopened[0].starts_with("initial-fill-"));
            assert!(reopened.last().unwrap().starts_with("deletion-wave-"));
        }
    }
}

fn assert_ownership(
    file: &File,
    lifecycle: Lifecycle,
    strategy: Strategy,
    base: Base,
    step: &str,
) -> SpaceOwnership {
    let owned = file_space::assert_exact_space_ownership(file);
    assert_eq!(
        owned.userblock_space,
        match base {
            Base::Zero => vec![],
            Base::Userblock => vec![(0, 4_096)],
        }
    );
    assert_eq!(
        owned.reusable_free_space,
        file.space_accounting().unwrap().reusable_free_space
    );
    assert!(!owned.live_metadata_space.is_empty());
    if step != "initial-commit" {
        assert!(!owned.free_space_manager_space.is_empty());
    }
    let raw = raw_extents(file);
    assert_eq!(
        raw.iter().map(|range| range.end - range.start).sum::<u64>(),
        owned
            .live_raw_space
            .iter()
            .map(|&(_, len)| len)
            .sum::<u64>()
    );
    for range in raw {
        assert!(
            owned
                .live_raw_space
                .iter()
                .any(|&(at, len)| at <= range.start && range.end <= at + len)
        );
    }
    if cfg!(not(feature = "fast-deflate")) && matches!(step, "final-commit" | "final-close") {
        let expected = match (strategy, lifecycle, step) {
            (Strategy::Page, _, _) => [
                (27_980_619, 8_461),
                (205_815, 569),
                (12_457, 6),
                (2_242_581, 100),
            ],
            (Strategy::FsmAggr, Lifecycle::ReopenFrequently, _) => [
                (27_980_619, 8_461),
                (205_815, 569),
                (1_143, 2),
                (1_872_957, 71),
            ],
            (Strategy::FsmAggr, Lifecycle::ContinuousOpen, "final-commit") => [
                (27_980_619, 8_461),
                (205_815, 569),
                (1_287, 2),
                (2_131_918, 79),
            ],
            (Strategy::FsmAggr, _, _) => [
                (27_980_619, 8_461),
                (205_815, 569),
                (1_296, 2),
                (2_131_909, 78),
            ],
        };
        assert_eq!(
            [
                &owned.live_raw_space,
                &owned.live_metadata_space,
                &owned.free_space_manager_space,
                &owned.reusable_free_space,
            ]
            .map(|extents| (
                extents.iter().map(|&(_, len)| len).sum::<u64>(),
                extents.len()
            )),
            expected
        );
    }
    owned
}

fn raw_extents(file: &File) -> Vec<Range<u64>> {
    let mut groups = vec![String::new()];
    let mut raw = Vec::new();
    while let Some(path) = groups.pop() {
        let group = if path.is_empty() {
            file.root()
        } else {
            file.group(&path).unwrap()
        };
        for dataset in group.datasets().unwrap() {
            raw.extend(file_space::raw_extents(file, &format!("{path}/{dataset}")));
        }
        groups.extend(
            group
                .groups()
                .unwrap()
                .iter()
                .map(|child| format!("{path}/{child}")),
        );
    }
    raw
}
