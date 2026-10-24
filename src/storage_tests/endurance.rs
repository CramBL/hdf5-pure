use std::panic::{self, AssertUnwindSafe};

use rstest::rstest;

use super::file_space;
use crate as api;
use crate::edit::SpaceOwnership;
use crate::{File, FileSpaceStrategy};

#[expect(
    dead_code,
    reason = "the shared workload has entry points used only by external tests"
)]
#[path = "../../crates/test-util-hdf5/src/free_space_endurance.rs"]
mod workload;

#[rstest]
#[case::fsm_aggr(FileSpaceStrategy::FsmAggr)]
#[case::page(FileSpaceStrategy::Page)]
#[case::none(FileSpaceStrategy::None)]
fn endurance_partitions_committed_and_reopened_storage(#[case] strategy: FileSpaceStrategy) {
    let mut steps = Vec::new();
    let mut closed = Vec::new();
    let mut sessions = String::new();
    let mut observations = String::new();
    workload::run_endurance_with_checkpoints(
        strategy,
        |path, step| {
            panic::catch_unwind(AssertUnwindSafe(|| {
                super::with_reopened_file(path, |file| {
                    assert_ownership(file, strategy, step)
                        .write_checkpoint(&mut observations, step);
                });
            }))
            .unwrap_or_else(|_| {
                panic!("closed-file ownership assertion failed for {strategy:?}/{step}")
            });
            closed.push(step.to_owned());
        },
        |file, step| {
            panic::catch_unwind(AssertUnwindSafe(|| {
                assert_ownership(file, strategy, step).write_checkpoint(&mut sessions, step);
            }))
            .unwrap_or_else(|_| panic!("ownership assertion failed for {strategy:?}/{step}"));
            steps.push(step.to_owned());
        },
    );
    insta::assert_snapshot!(
        format!("endurance_{strategy:?}"),
        format!(
            "Editing checkpoints (category values are bytes/extents):\n{sessions}\nClosed-file observations:\n{observations}"
        )
    );
    assert_eq!(
        steps,
        std::iter::once("initial-commit".to_owned())
            .chain(
                (1..=16).flat_map(|i| [format!("commit-{}", i * 64), format!("reopen-{}", i * 64)])
            )
            .collect::<Vec<_>>()
    );
    assert_eq!(
        closed,
        (1..=16)
            .map(|i| format!("reopen-{}", i * 64))
            .chain(std::iter::once("final-close".to_owned()))
            .collect::<Vec<_>>()
    );
}

fn assert_ownership(file: &File, strategy: FileSpaceStrategy, step: &str) -> SpaceOwnership {
    let owned = if strategy == FileSpaceStrategy::None {
        let owned = super::ownership(file).unwrap();
        owned.assert_classified(&owned.unclassified_space);
        assert!(owned.free_space_manager_space.is_empty());
        assert!(owned.reusable_free_space.is_empty());
        if step == "reopen-1024" {
            assert_eq!(
                [
                    &owned.live_raw_space,
                    &owned.live_metadata_space,
                    &owned.unclassified_space,
                ]
                .map(|extents| (
                    extents.iter().map(|&(_, len)| len).sum::<u64>(),
                    extents.len()
                )),
                [(44_800, 191), (22_941, 146), (4_767_335, 17)]
            );
        }
        owned
    } else {
        file_space::assert_exact_space_ownership(file)
    };
    assert!(!owned.live_metadata_space.is_empty());
    assert!(owned.userblock_space.is_empty());
    if step.starts_with("reopen-") || step == "final-close" {
        if strategy != FileSpaceStrategy::None {
            assert!(!owned.free_space_manager_space.is_empty());
        }
    }
    assert_eq!(
        owned.reusable_free_space,
        file.space_accounting().unwrap().reusable_free_space
    );
    let mut raw = Vec::new();
    for group_path in file.root().groups().unwrap() {
        raw.extend(file_space::raw_extents(
            file,
            &format!("{group_path}/contiguous"),
        ));
        raw.extend(file_space::raw_extents(
            file,
            &format!("{group_path}/chunked"),
        ));
        let nested_path = format!("{group_path}/nested");
        for dataset in file.group(&nested_path).unwrap().datasets().unwrap() {
            raw.extend(file_space::raw_extents(
                file,
                &format!("{nested_path}/{dataset}"),
            ));
        }
    }
    assert_eq!(
        raw.iter()
            .map(|extent| extent.end - extent.start)
            .sum::<u64>(),
        owned
            .live_raw_space
            .iter()
            .map(|&(_, len)| len)
            .sum::<u64>()
    );
    for extent in raw {
        assert!(
            owned
                .live_raw_space
                .iter()
                .any(|&(at, len)| at <= extent.start && extent.end <= at + len)
        );
    }
    owned
}
