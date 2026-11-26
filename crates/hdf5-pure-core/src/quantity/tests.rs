use rstest::rstest;

use super::{
    ByteLength, ChunkSlot, ElementCount, ElementSize, NonZeroByteLength, NonZeroElementCount,
    SlotCount,
};

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
