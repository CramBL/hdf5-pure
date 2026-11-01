use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::{FileSpacePageSize, FileSpaceStrategy, StoredAddress};
use hdf5_pure_space::__private::{Extent, FreeClass, ReturnedExtents, SessionSpace};
use rstest::rstest;

#[rstest]
#[case::seedless(false, Vec::new())]
#[case::with_seed(true, vec![Extent::new(StoredAddress::new(100), 128).unwrap()])]
fn returned_allocations_reach_admission_with_their_original_boundaries(
    #[case] seed: bool,
    #[case] expected: Vec<Extent>,
) {
    let mut space = SessionSpace::from_strategy(
        Some(FileSpaceStrategy::FsmAggr),
        64,
        FileSpacePageSize::DEFAULT,
    );
    if seed {
        space.seed_persisted(None, Extent::new(StoredAddress::new(100), 64).unwrap());
    }
    let returned = ReturnedExtents::from_sorted(vec![
        (
            Extent::new(StoredAddress::new(164), 32).unwrap(),
            FreeClass::Dead,
        ),
        (
            Extent::new(StoredAddress::new(196), 32).unwrap(),
            FreeClass::Dead,
        ),
    ])
    .unwrap();
    let (post, eoa) = space
        .flat_post_free(
            StoredAddress::new(300),
            returned.into_iter().map(|(extent, _)| extent),
        )
        .unwrap();
    assert_eq!(post.extents(), expected);
    assert_eq!(eoa, StoredAddress::new(300));
}
