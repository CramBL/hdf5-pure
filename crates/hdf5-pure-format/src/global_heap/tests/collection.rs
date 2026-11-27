use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;

use super::fixture;
use crate::error::FormatError;
use crate::global_heap::collection::{
    self, GCOL_SIGNATURE, GCOL_VERSION, GlobalHeapCollectionLocator,
};
use crate::global_heap::write;
use crate::width::LengthWidth;

#[rstest]
#[case::exact(u64::MAX - 7, Ok(u64::MAX - 7))]
#[case::first_overflow(u64::MAX - 6, Err(FormatError::OffsetOverflow { offset: u64::MAX - 6, length: 7 }))]
#[case::maximum(u64::MAX, Err(FormatError::OffsetOverflow { offset: u64::MAX, length: 1 }))]
fn collection_alignment_checks_u64_geometry(
    #[case] size: u64,
    #[case] expected: Result<u64, FormatError>,
) {
    assert_eq!(collection::aligned_collection_size(size), expected);
}

#[rstest]
#[case::below_the_minimum(
    4095,
    Err(FormatError::GlobalHeapCollectionTooSmall { collection_address: 64, size: 4095 })
)]
#[case::the_minimum(4096, Ok(4096))]
#[case::one_alignment_past_the_minimum(4104, Ok(4104))]
#[case::one_byte_past_the_minimum(
    4097,
    Err(FormatError::UnalignedGlobalHeapCollection { collection_address: 64, size: 4097 })
)]
#[case::half_an_alignment_past_the_minimum(
    4100,
    Err(FormatError::UnalignedGlobalHeapCollection { collection_address: 64, size: 4100 })
)]
fn a_collection_size_is_at_least_the_minimum_and_aligned(
    #[case] size: u64,
    #[case] expected: Result<u64, FormatError>,
) {
    let mut collection_header = GCOL_SIGNATURE.to_vec();
    collection_header.extend_from_slice(&[GCOL_VERSION, 0, 0, 0]);
    collection_header.extend_from_slice(&size.to_le_bytes());
    assert_eq!(
        GlobalHeapCollectionLocator::new(PhysicalAddress::new(64), LengthWidth::Eight)
            .parse(fixture::region(64, &collection_header))
            .map(|collection| collection.range().length()),
        expected
    );
}

#[rstest]
#[case::signature(0, b'X', FormatError::InvalidGlobalHeapSignature)]
#[case::version(4, 2, FormatError::InvalidGlobalHeapVersion(2))]
fn a_locator_rejects_a_header_that_is_not_a_collection(
    #[case] at: usize,
    #[case] value: u8,
    #[case] expected: FormatError,
) {
    let mut encoded = write::encode_global_heap_collection(LengthWidth::Eight, &[]).unwrap();
    encoded[at] = value;
    assert_eq!(
        GlobalHeapCollectionLocator::new(PhysicalAddress::new(0), LengthWidth::Eight)
            .parse(fixture::region(0, &encoded))
            .unwrap_err(),
        expected
    );
}

#[rstest]
#[case::two_byte_lengths(LengthWidth::Two)]
#[case::four_byte_lengths(LengthWidth::Four)]
#[case::eight_byte_lengths(LengthWidth::Eight)]
fn a_locator_rejects_an_otherwise_valid_collection_of_unaligned_size(#[case] lengths: LengthWidth) {
    let bytes = fixture::padded_collection(lengths, 4100, &[(1, b"first")]);
    assert_eq!(
        GlobalHeapCollectionLocator::new(PhysicalAddress::new(0), lengths)
            .parse(fixture::region(0, &bytes))
            .unwrap_err(),
        FormatError::UnalignedGlobalHeapCollection {
            collection_address: 0,
            size: 4100,
        }
    );
}
