use core::num::NonZeroU64;

use super::element::{ElementCount, ElementSize};
use super::host;
use crate::error::FormatError;

/// A length in bytes, including zero.
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::__private::{ByteLength, ElementCount, ElementSize};
///
/// let length = ByteLength::new(24);
/// let size = ElementSize::new(4).unwrap();
/// assert_eq!(length.elements(size), Some(ElementCount::new(6)));
/// assert_eq!(ByteLength::new(25).elements(size), None);
/// assert_eq!(length.nonzero().unwrap().length(), length);
/// assert_eq!(length.to_usize().unwrap(), 24);
/// ```
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ByteLength(u64);

impl ByteLength {
    /// Creates a byte length, including zero.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the byte length as a [`u64`].
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Returns `true` if the byte length is zero.
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Returns a positive byte length, or [`None`] if this length is zero.
    pub fn nonzero(self) -> Option<NonZeroByteLength> {
        NonZeroByteLength::new(self.0)
    }

    /// Adds `other`, returning [`None`] on [`u64`] overflow.
    pub fn checked_add(self, other: Self) -> Option<Self> {
        self.0.checked_add(other.0).map(Self)
    }

    /// Subtracts `other`, returning [`None`] if it exceeds this length.
    pub fn checked_sub(self, other: Self) -> Option<Self> {
        self.0.checked_sub(other.0).map(Self)
    }

    /// Returns the element count, or [`None`] if this length is not a multiple of `size`.
    pub fn elements(self, size: ElementSize) -> Option<ElementCount> {
        self.0
            .is_multiple_of(size.get())
            .then(|| ElementCount::new(self.0 / size.get()))
    }

    /// Converts the byte length to [`usize`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the value exceeds [`usize::MAX`].
    pub fn to_usize(self) -> Result<usize, FormatError> {
        host::to_usize(self.0)
    }
}

/// A positive byte length.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct NonZeroByteLength(NonZeroU64);

impl NonZeroByteLength {
    /// Creates a positive byte length, or returns [`None`] for zero.
    pub const fn new(value: u64) -> Option<Self> {
        match NonZeroU64::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    pub(super) const fn from_nonzero(value: NonZeroU64) -> Self {
        Self(value)
    }

    /// Returns the positive byte count as a [`u64`].
    pub const fn get(self) -> u64 {
        self.0.get()
    }

    /// Returns this length as a [`ByteLength`].
    pub const fn length(self) -> ByteLength {
        ByteLength(self.0.get())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::ByteLength;
    use crate::quantity::{ElementCount, ElementSize};

    #[rstest]
    #[case::empty(0, 4, Some(0))]
    #[case::one(4, 4, Some(1))]
    #[case::many(24, 4, Some(6))]
    #[case::partial(25, 4, None)]
    #[case::maximum(u64::MAX, 1, Some(u64::MAX))]
    fn element_counts_require_whole_elements(
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
}
