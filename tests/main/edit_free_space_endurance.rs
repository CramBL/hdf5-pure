use hdf5_pure::FileSpaceStrategy;
use rstest::rstest;
use test_util_hdf5::free_space_endurance;

#[rstest]
#[case::fsm_aggr(FileSpaceStrategy::FsmAggr)]
#[case::page(FileSpaceStrategy::Page)]
#[case::none(FileSpaceStrategy::None)]
fn small_file_churn_plateaus_only_with_persistent_free_space(#[case] strategy: FileSpaceStrategy) {
    insta::assert_snapshot!(
        format!("free_space_endurance_{strategy:?}").to_ascii_lowercase(),
        free_space_endurance::run_endurance(strategy),
    );
}
