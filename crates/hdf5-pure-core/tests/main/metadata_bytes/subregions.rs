use hdf5_pure_core::__private::{FileRange, MetadataBytesExt};
use hdf5_pure_core::{FormatError, PhysicalAddress};
use rstest::rstest;

use super::support::{self, Backing};

#[rstest]
#[case::borrowed_start(Backing::Borrowed, 512, 0, &[])]
#[case::owned_start(Backing::Owned, 512, 0, &[])]
#[case::borrowed_whole(Backing::Borrowed, 512, 4, &[1, 2, 3, 4])]
#[case::owned_whole(Backing::Owned, 512, 4, &[1, 2, 3, 4])]
#[case::borrowed_end(Backing::Borrowed, 516, 0, &[])]
#[case::owned_end(Backing::Owned, 516, 0, &[])]
fn workspace_subregions_preserve_contained_ranges(
    #[case] backing: Backing,
    #[case] start: u64,
    #[case] length: u64,
    #[case] expected: &[u8],
) {
    let data = [1, 2, 3, 4];
    let owner = support::bytes(backing, 512, &data);
    let range = FileRange::new(PhysicalAddress::new(start), length).unwrap();
    let selected = owner.region().subregion(range).unwrap();
    assert_eq!(selected.range(), range);
    assert_eq!(selected.bytes(), expected);
}

#[rstest]
#[case::borrowed_before(Backing::Borrowed, 511, 0)]
#[case::owned_before(Backing::Owned, 511, 0)]
#[case::borrowed_after(Backing::Borrowed, 517, 0)]
#[case::owned_after(Backing::Owned, 517, 0)]
#[case::borrowed_partial(Backing::Borrowed, 515, 2)]
#[case::owned_partial(Backing::Owned, 515, 2)]
fn workspace_subregions_reject_uncontained_ranges(
    #[case] backing: Backing,
    #[case] start: u64,
    #[case] length: u64,
) {
    let data = [1, 2, 3, 4];
    let owner = support::bytes(backing, 512, &data);
    let range = FileRange::new(PhysicalAddress::new(start), length).unwrap();
    assert_eq!(
        owner.region().subregion(range).unwrap_err(),
        FormatError::InsufficientRegion {
            required_start: start,
            required_length: length,
            provided_start: 512,
            provided_length: 4,
        }
    );
}
