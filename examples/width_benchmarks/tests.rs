use std::cell::Cell;
use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::thread;
use std::time::Duration;

use hdf5_pure::{Error, LengthWidth, OffsetWidth};
use rstest::rstest;
use test_util_hdf5::__private::width_workload::{Case, Index, Phase};
use test_util_hdf5::file_space::persisted::PersistedStrategy;

use super::report::{Snapshot, Summary};
use super::{Action, Arguments, Options};

#[rstest]
#[case::debug_default(true, vec![])]
#[case::debug_explicit(true, vec!["--check"])]
#[case::release_explicit(false, vec!["--check"])]
fn cli_selects_validation_without_a_timing_report(#[case] debug: bool, #[case] values: Vec<&str>) {
    assert_eq!(
        Action::parse(values.into_iter().map(String::from).collect(), debug).unwrap(),
        Action::Check
    );
}

#[rstest]
#[case::default(vec![], 21, 65_536)]
#[case::configured(vec!["3", "0"], 3, 0)]
fn release_cli_selects_measurement(
    #[case] values: Vec<&str>,
    #[case] samples: usize,
    #[case] cache_bytes: usize,
) {
    assert_eq!(
        Action::parse(values.into_iter().map(String::from).collect(), false).unwrap(),
        Action::Measure(Options {
            samples: NonZeroUsize::new(samples).unwrap(),
            cache_bytes
        })
    );
}

#[test]
fn debug_cli_rejects_measurement_arguments() {
    assert_eq!(
        Action::parse(vec!["1".into()], true).unwrap_err(),
        Arguments::DebugBuild
    );
}

#[rstest]
#[case::zero(vec!["0"], Arguments::Samples {value:0})]
#[case::too_many_samples(vec!["1001"], Arguments::Samples {value:1_001})]
#[case::invalid_integer(vec!["word"], Arguments::Number {value:"word".into()})]
#[case::invalid_cache(vec!["3","word"], Arguments::Number {value:"word".into()})]
#[case::extra_arguments(vec!["3","4","5"], Arguments::Count {actual:3})]
fn arguments_reject_invalid_configuration(#[case] values: Vec<&str>, #[case] expected: Arguments) {
    assert_eq!(
        Options::parse(values.into_iter().map(String::from).collect()).unwrap_err(),
        expected
    );
}

#[test]
fn default_configuration_has_repeatable_sample_and_cache_budgets() {
    let options = Options::parse(Vec::new()).unwrap();
    assert_eq!(options.samples.get(), 21);
    assert_eq!(options.cache_bytes, 65_536);
}

#[test]
fn matrix_lists_supported_and_explicitly_unsupported_configurations() {
    let (cases, unsupported) = super::cases();
    assert_eq!(cases.len(), 48);
    assert_eq!(unsupported.len(), 24);
    let labels = cases
        .iter()
        .map(|(_, label)| serde_json::to_string(label).unwrap())
        .chain(
            unsupported
                .iter()
                .map(|label| serde_json::to_string(label).unwrap()),
        )
        .collect::<BTreeSet<_>>();
    assert_eq!(labels.len(), 72);
    assert_eq!(
        cases
            .iter()
            .filter(|(case, _)| case.index() == Index::Finite)
            .count(),
        36
    );
    assert_eq!(
        cases
            .iter()
            .filter(|(case, _)| case.index() == Index::Extensible)
            .count(),
        12
    );
    let excluded = serde_json::to_value(unsupported).unwrap();
    assert!(
        excluded
            .as_array()
            .unwrap()
            .iter()
            .all(|label| label["index"] == "extensible" && label["length_bytes"] != 8)
    );
}

#[rstest]
#[case::first(0, [0, 1, 2])]
#[case::reverse(1, [1, 0, 47])]
#[case::rotated(17, [17, 16, 15])]
#[case::wrapped(49, [1, 0, 47])]
fn sample_order_visits_every_case_once(#[case] round: usize, #[case] prefix: [usize; 3]) {
    let mut order = super::sample_order(NonZeroUsize::new(48).unwrap(), round);
    assert_eq!(order[..3], prefix);
    order.sort_unstable();
    assert_eq!(order, (0..48).collect::<Vec<_>>());
}

#[rstest]
#[case::odd(&[9,1,7],1,7,9)]
#[case::even(&[4,1,3,2],1,3,4)]
#[case::single(&[5],5,5,5)]
fn summaries_retain_extremes_and_the_upper_middle_sample(
    #[case] values: &[u128],
    #[case] minimum: u128,
    #[case] median: u128,
    #[case] maximum: u128,
) {
    assert_eq!(
        serde_json::to_value(Summary::new(values)).unwrap(),
        serde_json::json!({"minimum_ns":minimum,"median_ns":median,"maximum_ns":maximum})
    );
}

#[test]
fn timing_includes_the_operation_and_executes_it_once() {
    let calls = Cell::new(0);
    let (value, elapsed) = super::timed(|| {
        calls.set(calls.get() + 1);
        thread::sleep(Duration::from_millis(1));
        Ok(17)
    })
    .unwrap();
    assert_eq!(value, 17);
    assert_eq!(calls.get(), 1);
    assert!(elapsed >= Duration::from_millis(1).as_nanos());
}

#[test]
fn timing_propagates_an_operation_error() {
    let error = super::timed::<()>(|| Err(Error::ReadOnly)).unwrap_err();
    assert!(matches!(error, Error::ReadOnly), "{error:?}");
}

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two, 24, 130, 30)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight, 24, 268, 30)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two, 48, 130, 48)]
#[case::default(OffsetWidth::Eight, LengthWidth::Eight, 48, 268, 48)]
fn selected_wire_model_separates_address_and_length_fields(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[case] superblock: u64,
    #[case] dataspaces: u64,
    #[case] descriptors: u64,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.h5");
    let case = Case::new(
        offsets,
        lengths,
        Index::Finite,
        PersistedStrategy::FsmAggr,
        false,
    )
    .unwrap();
    case.create(&path).unwrap();
    let value =
        serde_json::to_value(Snapshot::capture(case, &path, 65_536, Phase::Create).unwrap())
            .unwrap();
    assert_eq!(
        value["wire_model"],
        serde_json::json!({"superblock_bytes":superblock,"dataset_dataspace_bytes":dataspaces,"vl_descriptor_bytes":descriptors})
    );
    directory.close().unwrap();
}

#[test]
fn one_complete_round_records_all_operation_samples_and_default_differences() {
    let report = super::run(Options {
        samples: NonZeroUsize::new(1).unwrap(),
        cache_bytes: 65_536,
    })
    .unwrap();
    let value = serde_json::to_value(report).unwrap();
    let rows = value["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 48);
    assert_eq!(
        value["unsupported_unlimited_maximum"]
            .as_array()
            .unwrap()
            .len(),
        24
    );
    for row in rows {
        for name in ["create_ns", "append_ns", "commit_ns", "reopen_ns"] {
            let values = row[name].as_array().unwrap();
            assert_eq!(values.len(), 1);
            assert!(values[0].as_u64().unwrap() > 0);
        }
        assert_eq!(row["snapshots"].as_array().unwrap().len(), 4);
        if row["case"]["offset_bytes"] == 8 && row["case"]["length_bytes"] == 8 {
            for difference in row["differences_from_eight_eight"].as_array().unwrap() {
                assert!(
                    difference
                        .as_object()
                        .unwrap()
                        .values()
                        .all(|value| value == 0)
                );
            }
        }
    }
}
