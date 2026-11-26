use hdf5_pure_core::__private::{FileRange, MetadataBytesExt};
use hdf5_pure_core::{FormatError, MetadataBytes, PhysicalAddress};
use rstest::rstest;

use super::support::{self, Backing};

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
    let owner = support::bytes(backing, offset, data);
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
