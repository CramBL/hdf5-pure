use super::host;
use crate::error::FormatError;

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
        host::to_usize(self.0)
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
        host::to_usize(self.0)
    }
}
