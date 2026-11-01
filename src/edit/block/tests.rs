use core::num::NonZeroU64;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::{AllocationClass, Extent, ReservationRequest};
use rstest::rstest;

use crate::address::{
    AddressContext, BaseAddress, BaseAddressExt, StoredAddress, StoredAddressExt,
};
use crate::error::FormatError;

use super::{BlockRequest, EncodingDependency, PlannedBlock};

#[rstest]
#[case::empty(104, 0)]
#[case::partial(104, 16)]
#[case::complete(100, 64)]
fn block_ranges_preserve_complete_owner_and_encoded_subrange(
    #[case] address: u64,
    #[case] encoded: u64,
    #[values(0, 512, u64::MAX - 1_000)] base: u64,
) {
    let addresses = AddressContext::from_effective_base(BaseAddress::new(base));
    let request = BlockRequest::new(
        ReservationRequest::new(NonZeroU64::new(64).unwrap(), AllocationClass::Metadata),
        encoded,
        EncodingDependency::Allocation,
    )
    .unwrap();
    let block = PlannedBlock::within(
        addresses,
        Extent::new(StoredAddress::new(100), 64).unwrap(),
        StoredAddress::new(address),
        request,
    )
    .unwrap();
    assert_eq!(
        block.allocation(),
        FileRange::new(PhysicalAddress::new(base + 100), 64).unwrap()
    );
    assert_eq!(
        block.encoding(),
        FileRange::new(PhysicalAddress::new(base + address), encoded).unwrap()
    );
    assert_eq!(block.request(), request);
    assert_eq!(block.request().dependency(), EncodingDependency::Allocation);
    let bytes = vec![7; usize::try_from(encoded).unwrap()];
    let region = block.region(&bytes).unwrap();
    assert_eq!(
        region.range(),
        FileRange::new(PhysicalAddress::new(base + address), encoded).unwrap()
    );
    assert_eq!(region.bytes(), bytes);
}

#[rstest]
#[case::before(99, 1)]
#[case::after(164, 1)]
#[case::past_end(160, 8)]
fn block_encoding_outside_owned_allocation_is_rejected(#[case] address: u64, #[case] encoded: u64) {
    let request = BlockRequest::new(
        ReservationRequest::new(NonZeroU64::new(64).unwrap(), AllocationClass::Metadata),
        encoded,
        EncodingDependency::Allocation,
    )
    .unwrap();
    let Err(error) = PlannedBlock::within(
        AddressContext::from_effective_base(BaseAddress::ZERO),
        Extent::new(StoredAddress::new(100), 64).unwrap(),
        StoredAddress::new(address),
        request,
    ) else {
        panic!("expected a block outside its allocation");
    };
    assert_eq!(
        error,
        FormatError::SerializationError(format!(
            "a commit's tail reserved [100, 164) but placed {encoded} bytes at {address}",
        ))
    );
}

#[test]
fn block_request_rejects_encoding_larger_than_allocation() {
    assert_eq!(
        BlockRequest::new(
            ReservationRequest::new(NonZeroU64::new(16).unwrap(), AllocationClass::Raw),
            17,
            EncodingDependency::Independent,
        )
        .unwrap_err(),
        FormatError::SerializationError("a block reserves 16 bytes but encodes 17".into()),
    );
}
