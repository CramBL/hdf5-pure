//! Persistent free-space churn across editor lifecycles.
//!
//! The tests allocate mixed datasets and metadata, delete the oldest groups, and reuse their
//! storage. They use the same workload for each lifecycle and validate surviving and deleted
//! objects after an explicit close and a fresh reader open.

use rstest::rstest;
use test_util_hdf5::free_space_churn::{self, Base, Lifecycle, Strategy};

#[rstest]
fn free_space_churn_continuous_open(
    #[values(Strategy::FsmAggr, Strategy::Page)] strategy: Strategy,
    #[values(Base::Zero, Base::Userblock)] base: Base,
) {
    let report = free_space_churn::run_churn(Lifecycle::ContinuousOpen, strategy, base);
    insta::assert_snapshot!(
        strategy.snapshot("free_space_churn_continuous_open"),
        report.render()
    );
}

#[rstest]
fn free_space_churn_reopen_before_final_check(
    #[values(Strategy::FsmAggr, Strategy::Page)] strategy: Strategy,
    #[values(Base::Zero, Base::Userblock)] base: Base,
) {
    let report = free_space_churn::run_churn(Lifecycle::ReopenBeforeFinalCheck, strategy, base);
    insta::assert_snapshot!(
        strategy.snapshot("free_space_churn_reopen_before_final_check"),
        report.render()
    );
}

#[rstest]
fn free_space_churn_reopen_frequently(#[values(Base::Zero, Base::Userblock)] base: Base) {
    let report = free_space_churn::run_churn(Lifecycle::ReopenFrequently, Strategy::FsmAggr, base);
    insta::assert_snapshot!("free_space_churn_reopen_frequently", report.render());
}
