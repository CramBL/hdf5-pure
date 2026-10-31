use core::fmt::Debug;

use hdf5_pure_core::__private::{Address, MetadataFrame, StoredAddressExt};
use hdf5_pure_core::{AddressRange, FormatError, PhysicalAddress, StoredAddress};
use rstest::rstest;

#[rstest]
#[case::zero(0, 0)]
#[case::empty_at_maximum(u64::MAX, 0)]
#[case::ordinary(512, 96)]
#[case::maximum_end(u64::MAX - 96, 96)]
#[case::maximum_length(0, u64::MAX)]
fn public_ranges_preserve_typed_endpoints(#[case] start: u64, #[case] length: u64) {
    let physical = AddressRange::new(PhysicalAddress::new(start), length).unwrap();
    let physical_start: PhysicalAddress = physical.start();
    let physical_end: PhysicalAddress = physical.end();
    assert_eq!(physical_start, PhysicalAddress::new(start));
    assert_eq!(physical_end, PhysicalAddress::new(start + length));
    assert_eq!(physical.length(), length);
    assert_eq!(physical.is_empty(), length == 0);
    let stored = AddressRange::new(StoredAddress::new(start), length).unwrap();
    let stored_start: StoredAddress = stored.start();
    let stored_end: StoredAddress = stored.end();
    assert_eq!(stored_start, StoredAddress::new(start));
    assert_eq!(stored_end, StoredAddress::new(start + length));
    assert_eq!(stored.length(), length);
    assert_eq!(stored.is_empty(), length == 0);
}

#[rstest]
#[case::one(u64::MAX, 1)]
#[case::many(u64::MAX - 1, 512)]
fn public_ranges_reject_end_overflow(#[case] start: u64, #[case] length: u64) {
    let error = FormatError::OffsetOverflow {
        offset: start,
        length,
    };
    assert_eq!(
        AddressRange::new(PhysicalAddress::new(start), length).unwrap_err(),
        error
    );
    assert_eq!(
        AddressRange::new(StoredAddress::new(start), length).unwrap_err(),
        error
    );
}

#[rstest]
#[case::equal((10, 20), (10, 20), true)]
#[case::interior((10, 20), (15, 10), true)]
#[case::enclosing((10, 20), (5, 30), false)]
#[case::overlap_before((10, 20), (5, 10), false)]
#[case::overlap_after((10, 20), (25, 10), false)]
#[case::before((10, 20), (0, 5), false)]
#[case::after((10, 20), (35, 5), false)]
#[case::empty_start((10, 20), (10, 0), true)]
#[case::empty_interior((10, 20), (20, 0), true)]
#[case::empty_end((10, 20), (30, 0), true)]
#[case::empty_before((10, 20), (9, 0), false)]
#[case::empty_after((10, 20), (31, 0), false)]
#[case::equal_empty((u64::MAX, 0), (u64::MAX, 0), true)]
#[case::empty_contains_nonempty((10, 0), (10, 1), false)]
#[case::maximum((0, u64::MAX), (u64::MAX, 0), true)]
fn public_ranges_contain_ordered_endpoints(
    #[case] outer: (u64, u64),
    #[case] inner: (u64, u64),
    #[case] expected: bool,
) {
    assert_eq!(physical(outer).contains(physical(inner)), expected);
    assert_eq!(stored(outer).contains(stored(inner)), expected);
}

#[rstest]
#[case::equal((10, 20), (10, 20), Some((10, 20)))]
#[case::interior((10, 20), (15, 10), Some((15, 10)))]
#[case::overlap_before((10, 20), (5, 10), Some((10, 5)))]
#[case::overlap_after((10, 20), (25, 10), Some((25, 5)))]
#[case::adjacent_before((10, 20), (0, 10), None)]
#[case::adjacent_after((10, 20), (30, 10), None)]
#[case::disjoint((10, 20), (40, 10), None)]
#[case::empty_start((10, 20), (10, 0), None)]
#[case::empty_interior((10, 20), (20, 0), None)]
#[case::empty_end((10, 20), (30, 0), None)]
#[case::equal_empty((u64::MAX, 0), (u64::MAX, 0), None)]
#[case::maximum_end((u64::MAX - 10, 10), (u64::MAX - 5, 5), Some((u64::MAX - 5, 5)))]
fn public_range_intersections_agree_with_overlap(
    #[case] left: (u64, u64),
    #[case] right: (u64, u64),
    #[case] expected: Option<(u64, u64)>,
) {
    assert_intersection(physical(left), physical(right), expected.map(physical));
    assert_intersection(stored(left), stored(right), expected.map(stored));
}

#[rstest]
#[case::equal((10, 20), (10, 20), false)]
#[case::overlapping((10, 20), (20, 20), false)]
#[case::adjacent_before((10, 20), (0, 10), true)]
#[case::adjacent_after((10, 20), (30, 10), true)]
#[case::gap((10, 20), (31, 10), false)]
#[case::empty_start((10, 20), (10, 0), true)]
#[case::empty_interior((10, 20), (20, 0), false)]
#[case::empty_end((10, 20), (30, 0), true)]
#[case::equal_empty((u64::MAX, 0), (u64::MAX, 0), true)]
#[case::maximum_end((u64::MAX - 10, 5), (u64::MAX - 5, 5), true)]
fn public_range_adjacency_requires_a_shared_endpoint(
    #[case] left: (u64, u64),
    #[case] right: (u64, u64),
    #[case] expected: bool,
) {
    assert_eq!(physical(left).adjoins(physical(right)), expected);
    assert_eq!(physical(right).adjoins(physical(left)), expected);
    assert_eq!(stored(left).adjoins(stored(right)), expected);
    assert_eq!(stored(right).adjoins(stored(left)), expected);
}

#[rstest]
#[case::gap((10, 20), (40, 10), true)]
#[case::adjacent((10, 20), (30, 10), true)]
#[case::reversed((40, 10), (10, 20), false)]
#[case::overlapping((10, 20), (20, 20), false)]
#[case::equal((10, 20), (10, 20), false)]
#[case::empty_at_start((10, 0), (10, 20), true)]
#[case::empty_inside((20, 0), (10, 20), false)]
#[case::empty_at_end((10, 20), (30, 0), true)]
#[case::equal_empty((u64::MAX, 0), (u64::MAX, 0), true)]
fn public_range_ordering_compares_exclusive_ends(
    #[case] left: (u64, u64),
    #[case] right: (u64, u64),
    #[case] expected: bool,
) {
    assert_eq!(physical(left).precedes(physical(right)), expected);
    assert_eq!(stored(left).precedes(stored(right)), expected);
    assert_eq!(physical(left).cmp(&physical(right)), left.cmp(&right));
    assert_eq!(stored(left).cmp(&stored(right)), left.cmp(&right));
}

fn physical((start, length): (u64, u64)) -> AddressRange<PhysicalAddress> {
    AddressRange::new(PhysicalAddress::new(start), length).unwrap()
}

fn stored((start, length): (u64, u64)) -> AddressRange<StoredAddress> {
    AddressRange::new(StoredAddress::new(start), length).unwrap()
}

fn assert_intersection<A: Address + Debug>(
    left: AddressRange<A>,
    right: AddressRange<A>,
    expected: Option<AddressRange<A>>,
) {
    assert_eq!(left.intersection(right), expected);
    assert_eq!(right.intersection(left), expected);
    assert_eq!(left.overlaps(right), expected.is_some());
    assert_eq!(right.overlaps(left), expected.is_some());
}

#[rstest]
#[case::zero(0)]
#[case::userblock(512)]
#[case::maximum_end(u64::MAX - 3)]
fn metadata_frame_range_bounds_the_last_byte(#[case] origin: u64) {
    let frame = MetadataFrame::new(PhysicalAddress::new(origin), &[1, 2, 3]).unwrap();
    let range = frame.range();
    assert_eq!(range.start(), PhysicalAddress::new(origin));
    assert_eq!(range.end(), PhysicalAddress::new(origin + 3));
    let offset = range.end().checked_distance_from(frame.origin()).unwrap() - 1;
    let last = frame.read_at(offset, 1).unwrap();
    assert_eq!(last.as_ref(), &[3]);
    assert_eq!(last.offset(), origin + 2);
    assert_eq!(last.range().end(), range.end());
}
