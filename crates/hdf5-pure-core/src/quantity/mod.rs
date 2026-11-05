//! Byte lengths, element counts, and positions in a chunk index.
//!
//! Arithmetic uses [`u64`]. Host indexing uses explicit checked conversions to [`usize`]. Callers
//! establish encoded-field limits, allocation ownership, and resource bounds separately.

use core::num::NonZeroU64;

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
            .is_multiple_of(size.0.get())
            .then(|| ElementCount(self.0 / size.0.get()))
    }

    /// Converts the byte length to [`usize`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the value exceeds [`usize::MAX`].
    pub fn to_usize(self) -> Result<usize, FormatError> {
        to_usize(self.0)
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

    /// Returns the positive byte count as a [`u64`].
    pub const fn get(self) -> u64 {
        self.0.get()
    }

    /// Returns this length as a [`ByteLength`].
    pub const fn length(self) -> ByteLength {
        ByteLength(self.0.get())
    }
}

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
        SlotCount(self.0.div_ceil(chunk.0.get()))
    }

    /// Converts the element count to [`usize`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the value exceeds [`usize::MAX`].
    pub fn to_usize(self) -> Result<usize, FormatError> {
        to_usize(self.0)
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
        NonZeroByteLength(self.0)
    }

    /// Returns the byte length of `count` elements, or [`None`] on [`u64`] overflow.
    pub fn byte_length(self, count: ElementCount) -> Option<ByteLength> {
        self.0.get().checked_mul(count.0).map(ByteLength)
    }
}

/// A zero-based position in a chunk index.
///
/// A position can designate an exclusive end. Callers check bounds against the index being used.
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::__private::{ChunkSlot, SlotCount};
///
/// let count = SlotCount::new(4);
/// assert_eq!(count.end(), ChunkSlot::new(4));
/// let start = ChunkSlot::new(3);
/// let end = start.checked_add(count).unwrap();
/// assert_eq!(end, ChunkSlot::new(7));
/// assert_eq!(end.checked_distance_from(start), Some(count));
/// ```
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ChunkSlot(u64);

impl ChunkSlot {
    /// Creates a zero-based chunk-index position.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the chunk-index position as a [`u64`].
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Advances by `count` slots, returning [`None`] on [`u64`] overflow.
    pub fn checked_add(self, count: SlotCount) -> Option<Self> {
        self.0.checked_add(count.0).map(Self)
    }

    /// Returns the distance from `earlier`, or [`None`] if it follows this position.
    pub fn checked_distance_from(self, earlier: Self) -> Option<SlotCount> {
        self.0.checked_sub(earlier.0).map(SlotCount)
    }

    /// Converts the chunk-index position to [`usize`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the value exceeds [`usize::MAX`].
    pub fn to_usize(self) -> Result<usize, FormatError> {
        to_usize(self.0)
    }
}

/// A number of chunk-index slots, including zero.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct SlotCount(u64);

impl SlotCount {
    /// Creates a slot count, including zero.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the slot count as a [`u64`].
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Returns `true` if the slot count is zero.
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Returns the exclusive-end position for a zero-based index with this many slots.
    pub const fn end(self) -> ChunkSlot {
        ChunkSlot(self.0)
    }

    /// Adds `other`, returning [`None`] on [`u64`] overflow.
    pub fn checked_add(self, other: Self) -> Option<Self> {
        self.0.checked_add(other.0).map(Self)
    }

    /// Subtracts `other`, returning [`None`] if it exceeds this count.
    pub fn checked_sub(self, other: Self) -> Option<Self> {
        self.0.checked_sub(other.0).map(Self)
    }

    /// Converts the slot count to [`usize`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the value exceeds [`usize::MAX`].
    pub fn to_usize(self) -> Result<usize, FormatError> {
        to_usize(self.0)
    }
}

fn to_usize(value: u64) -> Result<usize, FormatError> {
    usize::try_from(value).map_err(|_error| FormatError::ValueTooLargeForPlatform {
        value,
        target: "usize",
    })
}

#[cfg(test)]
mod tests;
