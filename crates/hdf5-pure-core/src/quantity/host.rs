use crate::error::FormatError;

pub(super) fn to_usize(value: u64) -> Result<usize, FormatError> {
    usize::try_from(value).map_err(|_error| FormatError::ValueTooLargeForPlatform {
        value,
        target: "usize",
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::error::FormatError;
    use crate::quantity::{ByteLength, ChunkSlot, ElementCount, SlotCount};

    #[rstest]
    #[case::zero(0, 0)]
    #[case::one(1, 1)]
    #[case::maximum_32bit(u64::from(u32::MAX), 32)]
    #[case::beyond_32bit(u64::from(u32::MAX) + 1, 33)]
    #[case::maximum_64bit(u64::MAX, 64)]
    fn quantities_exceeding_usize_return_a_platform_size_error(
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
}
