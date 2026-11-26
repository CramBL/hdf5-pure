use core::num::NonZeroU64;

use super::byte_length::{ByteLength, NonZeroByteLength};
use super::chunk_slot::SlotCount;
use super::host;
use crate::error::FormatError;

/// A number of elements, including zero.
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::__private::{ElementCount, NonZeroElementCount, SlotCount};
///
/// let chunk = NonZeroElementCount::new(4).unwrap();
/// assert_eq!(ElementCount::new(5).chunks(chunk), SlotCount::new(2));
/// assert_eq!(ElementCount::new(0).chunks(chunk), SlotCount::new(0));
/// assert_eq!(chunk.count(), ElementCount::new(4));
/// ```
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ElementCount(u64);

impl ElementCount {
    /// Creates an element count, including zero.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the element count as a [`u64`].
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Returns `true` if the element count is zero.
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Returns a positive element count, or [`None`] if this count is zero.
    pub fn nonzero(self) -> Option<NonZeroElementCount> {
        NonZeroElementCount::new(self.0)
    }

    /// Adds `other`, returning [`None`] on [`u64`] overflow.
    pub fn checked_add(self, other: Self) -> Option<Self> {
        self.0.checked_add(other.0).map(Self)
    }

    /// Subtracts `other`, returning [`None`] if it exceeds this count.
    pub fn checked_sub(self, other: Self) -> Option<Self> {
        self.0.checked_sub(other.0).map(Self)
    }

    /// Multiplies element counts, returning [`None`] on [`u64`] overflow.
    pub fn checked_mul(self, other: Self) -> Option<Self> {
        self.0.checked_mul(other.0).map(Self)
    }

    /// Returns the number of chunks needed for this count, rounding up partial chunks.
    ///
    /// An empty element count requires zero chunks.
    pub fn chunks(self, chunk: NonZeroElementCount) -> SlotCount {
        SlotCount::new(self.0.div_ceil(chunk.0.get()))
    }

    /// Converts the element count to [`usize`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the value exceeds [`usize::MAX`].
    pub fn to_usize(self) -> Result<usize, FormatError> {
        host::to_usize(self.0)
    }
}

/// A positive element count.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct NonZeroElementCount(NonZeroU64);

impl NonZeroElementCount {
    /// Creates a positive element count, or returns [`None`] for zero.
    pub const fn new(value: u64) -> Option<Self> {
        match NonZeroU64::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    /// Returns the positive element count as a [`u64`].
    pub const fn get(self) -> u64 {
        self.0.get()
    }

    /// Returns this count as an [`ElementCount`].
    pub const fn count(self) -> ElementCount {
        ElementCount(self.0.get())
    }
}

/// A positive byte size for one element.
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::__private::{ByteLength, ElementCount, ElementSize};
///
/// let size = ElementSize::new(6).unwrap();
/// assert_eq!(size.byte_length(ElementCount::new(7)), Some(ByteLength::new(42)));
/// assert_eq!(size.byte_length(ElementCount::new(u64::MAX)), None);
/// ```
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ElementSize(NonZeroU64);

impl ElementSize {
    /// Creates a positive element size, or returns [`None`] for zero.
    pub const fn new(value: u64) -> Option<Self> {
        match NonZeroU64::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    /// Returns the bytes per element as a [`u64`].
    pub const fn get(self) -> u64 {
        self.0.get()
    }

    /// Returns the byte length of one element.
    pub const fn bytes(self) -> NonZeroByteLength {
        NonZeroByteLength::from_nonzero(self.0)
    }

    /// Returns the byte length of `count` elements, or [`None`] on [`u64`] overflow.
    pub fn byte_length(self, count: ElementCount) -> Option<ByteLength> {
        self.0.get().checked_mul(count.0).map(ByteLength::new)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{ElementCount, ElementSize, NonZeroElementCount};
    use crate::quantity::{ByteLength, SlotCount};

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
}
