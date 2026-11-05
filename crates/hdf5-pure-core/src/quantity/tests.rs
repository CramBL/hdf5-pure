use rstest::rstest;

use super::{
    ByteLength, ChunkSlot, ElementCount, ElementSize, NonZeroByteLength, NonZeroElementCount,
    SlotCount,
};
use crate::FormatError;

#[rstest]
#[case::zero(0)]
#[case::one(1)]
#[case::maximum(u64::MAX)]
fn quantities_preserve_their_values_and_positive_forms(#[case] value: u64) {
    let bytes = ByteLength::new(value);
    let elements = ElementCount::new(value);
    let slots = SlotCount::new(value);
    assert_eq!(bytes.get(), value);
    assert_eq!(elements.get(), value);
    assert_eq!(slots.get(), value);
    assert_eq!(ChunkSlot::new(value).get(), value);
    assert_eq!(slots.end(), ChunkSlot::new(value));
    assert_eq!(bytes.is_zero(), value == 0);
    assert_eq!(elements.is_zero(), value == 0);
    assert_eq!(slots.is_zero(), value == 0);
    if value == 0 {
        assert_eq!(bytes.nonzero(), None);
        assert_eq!(elements.nonzero(), None);
        assert_eq!(NonZeroByteLength::new(value), None);
        assert_eq!(NonZeroElementCount::new(value), None);
        assert_eq!(ElementSize::new(value), None);
    } else {
        let positive_bytes = bytes.nonzero().unwrap();
        let positive_elements = elements.nonzero().unwrap();
        let size = ElementSize::new(value).unwrap();
        assert_eq!(positive_bytes.get(), value);
        assert_eq!(positive_bytes.length(), bytes);
        assert_eq!(positive_elements.get(), value);
        assert_eq!(positive_elements.count(), elements);
        assert_eq!(size.get(), value);
        assert_eq!(size.bytes(), positive_bytes);
    }
}

#[rstest]
#[case::zero(0, 0, Some(0))]
#[case::ordinary(13, 29, Some(42))]
#[case::exact_maximum(u64::MAX - 1, 1, Some(u64::MAX))]
#[case::one_beyond(u64::MAX, 1, None)]
#[case::maximum_with_zero(u64::MAX, 0, Some(u64::MAX))]
fn additions_reject_overflow_in_each_unit(
    #[case] left: u64,
    #[case] right: u64,
    #[case] expected: Option<u64>,
) {
    assert_eq!(
        ByteLength::new(left)
            .checked_add(ByteLength::new(right))
            .map(ByteLength::get),
        expected
    );
    assert_eq!(
        ElementCount::new(left)
            .checked_add(ElementCount::new(right))
            .map(ElementCount::get),
        expected
    );
    assert_eq!(
        SlotCount::new(left)
            .checked_add(SlotCount::new(right))
            .map(SlotCount::get),
        expected
    );
    assert_eq!(
        ChunkSlot::new(left)
            .checked_add(SlotCount::new(right))
            .map(ChunkSlot::get),
        expected
    );
}

#[rstest]
#[case::zero(0, 0, Some(0))]
#[case::ordinary(42, 29, Some(13))]
#[case::maximum(u64::MAX, 0, Some(u64::MAX))]
#[case::equal(u64::MAX, u64::MAX, Some(0))]
#[case::underflow(0, 1, None)]
fn differences_reject_underflow_and_keep_slot_distances_as_counts(
    #[case] left: u64,
    #[case] right: u64,
    #[case] expected: Option<u64>,
) {
    assert_eq!(
        ByteLength::new(left)
            .checked_sub(ByteLength::new(right))
            .map(ByteLength::get),
        expected
    );
    assert_eq!(
        ElementCount::new(left)
            .checked_sub(ElementCount::new(right))
            .map(ElementCount::get),
        expected
    );
    assert_eq!(
        SlotCount::new(left)
            .checked_sub(SlotCount::new(right))
            .map(SlotCount::get),
        expected
    );
    assert_eq!(
        ChunkSlot::new(left)
            .checked_distance_from(ChunkSlot::new(right))
            .map(SlotCount::get),
        expected
    );
}

#[rstest]
#[case::empty(0, u64::MAX, Some(0))]
#[case::ordinary(7, 6, Some(42))]
#[case::maximum(u64::MAX, 1, Some(u64::MAX))]
#[case::largest_even(u64::MAX / 2, 2, Some(u64::MAX - 1))]
#[case::one_beyond(u64::MAX / 2 + 1, 2, None)]
fn element_products_return_counts_and_stride_products_return_bytes(
    #[case] count: u64,
    #[case] factor: u64,
    #[case] expected: Option<u64>,
) {
    assert_eq!(
        ElementCount::new(count)
            .checked_mul(ElementCount::new(factor))
            .map(ElementCount::get),
        expected
    );
    assert_eq!(
        ElementSize::new(factor)
            .unwrap()
            .byte_length(ElementCount::new(count))
            .map(ByteLength::get),
        expected
    );
}

#[rstest]
#[case::empty(0, 4, Some(0))]
#[case::one(4, 4, Some(1))]
#[case::many(24, 4, Some(6))]
#[case::partial(25, 4, None)]
#[case::maximum(u64::MAX, 1, Some(u64::MAX))]
fn byte_lengths_require_whole_elements(
    #[case] bytes: u64,
    #[case] stride: u64,
    #[case] expected: Option<u64>,
) {
    assert_eq!(
        ByteLength::new(bytes)
            .elements(ElementSize::new(stride).unwrap())
            .map(ElementCount::get),
        expected
    );
}

#[rstest]
#[case::empty(0, 4, 0)]
#[case::partial(1, 4, 1)]
#[case::complete(4, 4, 1)]
#[case::next(5, 4, 2)]
#[case::maximum_single_element(u64::MAX, 1, u64::MAX)]
#[case::maximum_pairs(u64::MAX, 2, 1_u64 << 63)]
#[case::maximum_chunk(u64::MAX, u64::MAX, 1)]
fn chunk_counts_round_up_without_overflow(
    #[case] elements: u64,
    #[case] chunk: u64,
    #[case] expected: u64,
) {
    assert_eq!(
        ElementCount::new(elements).chunks(NonZeroElementCount::new(chunk).unwrap()),
        SlotCount::new(expected)
    );
}

#[rstest]
#[case::zero(0, 0)]
#[case::one(1, 1)]
#[case::maximum_32bit(u64::from(u32::MAX), 32)]
#[case::beyond_32bit(u64::from(u32::MAX) + 1, 33)]
#[case::maximum_64bit(u64::MAX, 64)]
fn materialized_counts_and_offsets_reject_values_beyond_usize(
    #[case] value: u64,
    #[case] required_bits: u32,
) {
    let expected = if usize::BITS >= required_bits {
        Ok(value as usize)
    } else {
        Err(FormatError::ValueTooLargeForPlatform {
            value,
            target: "usize",
        })
    };
    assert_eq!(ByteLength::new(value).to_usize(), expected);
    assert_eq!(ElementCount::new(value).to_usize(), expected);
    assert_eq!(SlotCount::new(value).to_usize(), expected);
    assert_eq!(ChunkSlot::new(value).to_usize(), expected);
}
