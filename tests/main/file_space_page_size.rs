use hdf5_pure::FileSpacePageSize;
use hdf5_pure::FormatError;
use rstest::rstest;

#[rstest]
#[case::minimum(512)]
#[case::default(4096)]
#[case::not_a_power_of_two(3000)]
#[case::maximum(1 << 30)]
fn a_page_size_from_512_bytes_to_1_gib_parses(#[case] page_size: u64) {
    assert_eq!(
        FileSpacePageSize::try_from(page_size).map(FileSpacePageSize::get),
        Ok(page_size)
    );
}

#[rstest]
#[case::zero(0)]
#[case::below_the_minimum(511)]
#[case::above_the_maximum((1 << 30) + 1)]
#[case::two_gib(2 << 30)]
#[case::u64_max(u64::MAX)]
fn a_page_size_outside_512_bytes_to_1_gib_is_rejected(#[case] page_size: u64) {
    assert_eq!(
        FileSpacePageSize::try_from(page_size),
        Err(FormatError::InvalidFileSpacePageSize(page_size))
    );
}

#[test]
fn the_default_page_size_is_4096_bytes() {
    assert_eq!(FileSpacePageSize::default().get(), 4096);
}
