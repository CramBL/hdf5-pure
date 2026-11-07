use hdf5_pure_core::__private::{FileRange, MetadataBytesExt};
use hdf5_pure_core::{FormatError, MetadataBytes, PhysicalAddress};

use rstest::rstest;

mod boundary;

#[derive(Clone, Copy, Debug)]
enum Backing {
    Borrowed,
    Owned,
}

fn bytes(backing: Backing, offset: u64, data: &[u8]) -> MetadataBytes<'_> {
    match backing {
        Backing::Borrowed => MetadataBytes::from_slice(offset, data),
        Backing::Owned => MetadataBytes::from_vec(offset, data.to_vec()),
    }
    .unwrap()
}

#[rstest]
#[case::borrowed_empty(Backing::Borrowed, u64::MAX, &[])]
#[case::owned_empty(Backing::Owned, u64::MAX, &[])]
#[case::borrowed_end(Backing::Borrowed, u64::MAX - 3, &[1, 2, 3])]
#[case::owned_end(Backing::Owned, u64::MAX - 3, &[1, 2, 3])]
fn stable_constructors_preserve_extent(
    #[case] backing: Backing,
    #[case] offset: u64,
    #[case] data: &[u8],
) {
    let owner = bytes(backing, offset, data);
    assert_eq!(owner.offset(), offset);
    assert_eq!(owner.as_ref(), data);
    assert_eq!(&*owner, data);
}

#[rstest]
#[case::borrowed_one(Backing::Borrowed, u64::MAX, &[1])]
#[case::owned_one(Backing::Owned, u64::MAX, &[1])]
#[case::borrowed_many(Backing::Borrowed, u64::MAX - 1, &[1, 2, 3])]
#[case::owned_many(Backing::Owned, u64::MAX - 1, &[1, 2, 3])]
fn stable_constructors_reject_overflow(
    #[case] backing: Backing,
    #[case] offset: u64,
    #[case] data: &[u8],
) {
    let error = match backing {
        Backing::Borrowed => MetadataBytes::from_slice(offset, data),
        Backing::Owned => MetadataBytes::from_vec(offset, data.to_vec()),
    }
    .unwrap_err();
    assert_eq!(
        error,
        FormatError::OffsetOverflow {
            offset,
            length: data.len() as u64,
        }
    );
}

#[rstest]
#[case::borrowed_short(Backing::Borrowed, 5)]
#[case::borrowed_long(Backing::Borrowed, 3)]
#[case::owned_short(Backing::Owned, 5)]
#[case::owned_long(Backing::Owned, 3)]
fn workspace_constructors_reject_count_mismatch(#[case] backing: Backing, #[case] length: u64) {
    let range = FileRange::new(PhysicalAddress::new(512), length).unwrap();
    let data = [1, 2, 3, 4];
    let error = match backing {
        Backing::Borrowed => MetadataBytes::borrowed(range, &data),
        Backing::Owned => MetadataBytes::owned(range, data.to_vec()),
    }
    .unwrap_err();
    assert_eq!(
        error,
        FormatError::FileRegionLengthMismatch {
            range_length: length,
            bytes_length: 4,
        }
    );
}

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
    let owner = bytes(backing, 512, &data);
    let range = FileRange::new(PhysicalAddress::new(start), length).unwrap();
    let selected = owner.subregion(range).unwrap();
    assert_eq!(selected.range(), range);
    assert_eq!(selected.region().range(), range);
    assert_eq!(selected.region().bytes(), expected);
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
    let owner = bytes(backing, 512, &data);
    let range = FileRange::new(PhysicalAddress::new(start), length).unwrap();
    assert_eq!(
        owner.subregion(range).unwrap_err(),
        FormatError::InsufficientRegion {
            required_start: start,
            required_length: length,
            provided_start: 512,
            provided_length: 4,
        }
    );
}

#[test]
fn workspace_subregions_share_nested_offsets_and_retained_capacity() {
    let mut data = Vec::with_capacity(64);
    data.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
    let pointer = data.as_ptr();
    let capacity = data.capacity();
    let range = FileRange::new(PhysicalAddress::new(u64::MAX - 6), 6).unwrap();
    let owner = MetadataBytes::owned(range, data).unwrap();
    assert_eq!(owner.as_ptr(), pointer);
    let middle = owner
        .subregion(FileRange::new(PhysicalAddress::new(u64::MAX - 5), 4).unwrap())
        .unwrap();
    let nested = middle
        .subregion(FileRange::new(PhysicalAddress::new(u64::MAX - 4), 2).unwrap())
        .unwrap();
    let expected_pointer = owner.get(2..).unwrap().as_ptr();
    let retained: MetadataBytes<'static> = nested.shared().unwrap();
    drop(owner);
    drop(middle);
    drop(nested);
    assert_eq!(retained.offset(), u64::MAX - 4);
    assert_eq!(retained.as_ref(), &[3, 4]);
    assert_eq!(retained.as_ptr(), expected_pointer);
    assert_eq!(retained.retained_bytes(), capacity);
    assert_eq!(retained.into_vec(), vec![3, 4]);
}

#[test]
fn stable_ownership_accessors_preserve_the_payload_allocation() {
    let mut data = Vec::with_capacity(64);
    data.extend_from_slice(&[1, 2, 3]);
    let pointer = data.as_ptr();
    let capacity = data.capacity();
    let owner = MetadataBytes::from_vec(512, data).unwrap();
    let retained = owner.shared().unwrap();
    let cloned = owner.clone();
    drop(owner);
    drop(cloned);
    assert_eq!(retained.offset(), 512);
    assert_eq!(retained.retained_bytes(), capacity);
    let data = retained.into_vec();
    assert_eq!(data, vec![1, 2, 3]);
    assert_eq!(data.as_ptr(), pointer);
    assert_eq!(data.capacity(), capacity);
}

#[test]
fn borrowed_ownership_accessors_keep_the_borrow() {
    let data = [1, 2, 3];
    let owner = MetadataBytes::from_slice(512, &data).unwrap();
    assert_eq!(owner.as_ptr(), data.as_ptr());
    assert!(owner.shared().is_none());
    assert_eq!(owner.retained_bytes(), 0);
    assert_eq!(owner.into_vec(), data);
}
