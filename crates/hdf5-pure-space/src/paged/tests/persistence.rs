use alloc::vec;

use hdf5_pure_core::__private::NonZeroByteLength;
use rstest::rstest;

use super::super::*;
use super::PAGE;

#[rstest]
#[case::empty(&[], &[], &[])]
#[case::partial(&[(PAGE + 10, 1)], &[(PAGE + 10, 1)], &[])]
#[case::adjacent(&[(PAGE + 10, 1), (PAGE + 11, 2)], &[(PAGE + 10, 3)], &[])]
#[case::whole_interior(&[(2 * PAGE - 3, PAGE + 6)], &[(2 * PAGE - 3, 3), (3 * PAGE, 3)], &[(2 * PAGE, PAGE)])]
fn allocation_padding_retains_small_tails_and_promotes_complete_interiors(
    #[case] padding: &[(u64, u64)],
    #[case] metadata: &[(u64, u64)],
    #[case] raw: &[(u64, u64)],
) {
    let tail = PagedTail {
        allocation: super::extent(PAGE, 3 * PAGE),
        post: super::paged_free_space(&[vec![], vec![], vec![]]),
        final_eoa: StoredAddress::new(4 * PAGE),
        appended: true,
        request: ReservationRequest::new(
            NonZeroByteLength::new(3 * PAGE).unwrap(),
            AllocationClass::Metadata,
        ),
    }
    .with_metadata_padding(
        &padding
            .iter()
            .map(|&(start, len)| super::extent(start, len))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(
        tail.post.meta.extents(),
        metadata
            .iter()
            .map(|&(start, len)| super::extent(start, len))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        tail.post.raw.extents(),
        raw.iter()
            .map(|&(start, len)| super::extent(start, len))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        tail.post.admitted_snapshot().reusable_bytes().unwrap(),
        ByteLength::new(padding.iter().map(|&(_, len)| len).sum::<u64>())
    );
}

#[rstest]
#[case::before(PAGE - 1, "PAGE allocation padding is outside its reservation")]
#[case::after(4 * PAGE, "PAGE allocation padding is outside its reservation")]
#[case::already_free(PAGE + 1, "PAGE allocation padding overlaps free space")]
fn allocation_padding_rejects_unowned_or_already_free_bytes(
    #[case] start: u64,
    #[case] message: &str,
) {
    let tail = PagedTail {
        allocation: super::extent(PAGE, 3 * PAGE),
        post: super::paged_free_space(&[vec![(PAGE + 1, 1)], vec![], vec![]]),
        final_eoa: StoredAddress::new(4 * PAGE),
        appended: true,
        request: ReservationRequest::new(
            NonZeroByteLength::new(3 * PAGE).unwrap(),
            AllocationClass::Metadata,
        ),
    };
    let error = tail
        .with_metadata_padding(&[super::extent(start, 1)])
        .err()
        .unwrap();
    assert_eq!(error, FormatError::Internal(message.into()));
}

#[rstest]
#[case::overlap(&[(PAGE + 10, 2), (PAGE + 11, 1)], "overlapping free extents [4106, 4108) and [4107, 4108)")]
#[case::unordered(&[(PAGE + 11, 1), (PAGE + 10, 1)], "out-of-order free extents [4107, 4108) and [4106, 4107)")]
fn allocation_padding_rejects_invalid_batches(
    #[case] padding: &[(u64, u64)],
    #[case] message: &str,
) {
    let tail = PagedTail {
        allocation: super::extent(PAGE, 3 * PAGE),
        post: super::paged_free_space(&[vec![], vec![], vec![]]),
        final_eoa: StoredAddress::new(4 * PAGE),
        appended: true,
        request: ReservationRequest::new(
            NonZeroByteLength::new(3 * PAGE).unwrap(),
            AllocationClass::Metadata,
        ),
    };
    let error = tail
        .with_metadata_padding(
            &padding
                .iter()
                .map(|&(start, len)| super::extent(start, len))
                .collect::<Vec<_>>(),
        )
        .err()
        .unwrap();
    assert_eq!(
        error,
        FormatError::Internal(format!("invalid PAGE allocation padding: {message}"))
    );
}
