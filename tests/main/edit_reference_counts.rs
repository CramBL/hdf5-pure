use hdf5_pure::Error;
use rstest::rstest;

#[rstest]
#[case::zero(0, 1)]
#[case::undercount(2, 3)]
fn reference_count_contradictions_report_the_physical_object_and_counts(
    #[case] observed: u32,
    #[case] minimum: u64,
) {
    let error = Error::InconsistentReferenceCount {
        address: 2048,
        observed,
        minimum,
    };
    assert_eq!(
        error.to_string(),
        format!("object at 2048 has reference count {observed}, below the known minimum {minimum}")
    );
}
