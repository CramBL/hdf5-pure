//! Physical file ranges and borrowed bytes associated with them.

use crate::address::PhysicalAddress;
use crate::error::FormatError;

/// Borrowed bytes associated with a physical file range.
///
/// The byte count equals the range length. The caller asserts that the bytes represent the
/// range, without verification of their source or its revision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FileRegion<'a> {
    range: FileRange,
    bytes: &'a [u8],
}

impl<'a> FileRegion<'a> {
    /// Associates `bytes` with their physical range.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::FileRegionLengthMismatch`] if the byte count differs from the
    /// range length.
    pub fn new(range: FileRange, bytes: &'a [u8]) -> Result<Self, FormatError> {
        if range.length() != bytes.len() as u64 {
            return Err(FormatError::FileRegionLengthMismatch {
                range_length: range.length(),
                bytes_length: bytes.len() as u64,
            });
        }
        Ok(Self { range, bytes })
    }

    /// Returns the physical range associated with these bytes.
    pub const fn range(self) -> FileRange {
        self.range
    }

    /// Returns the borrowed bytes for the entire range.
    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// Borrows the bytes for a contained physical range.
    ///
    /// An empty range at either endpoint is contained.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `required` is outside this region's range.
    pub fn subregion(self, required: FileRange) -> Result<Self, FormatError> {
        let Self { range, bytes } = self;
        let error = FormatError::InsufficientRegion {
            required_start: required.start().get(),
            required_length: required.length(),
            provided_start: range.start().get(),
            provided_length: range.length(),
        };
        if !range.contains(required) {
            return Err(error);
        }
        let start = required.start().get() - range.start().get();
        let end = required.end().get() - range.start().get();
        let bytes = usize::try_from(start)
            .ok()
            .zip(usize::try_from(end).ok())
            .and_then(|(start, end)| bytes.get(start..end))
            .ok_or(error)?;
        Ok(Self {
            range: required,
            bytes,
        })
    }
}

/// A checked half-open interval of physical byte positions.
///
/// The interval includes its start and excludes its end. Its length may be zero, and both
/// endpoints are represented as [`u64`] on every platform.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FileRange {
    start: PhysicalAddress,
    end: PhysicalAddress,
}

impl FileRange {
    /// Creates an interval beginning at `start` and spanning `length` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the end exceeds [`u64::MAX`].
    pub fn new(start: PhysicalAddress, length: u64) -> Result<Self, FormatError> {
        Ok(Self {
            start,
            end: start.checked_add(length)?,
        })
    }

    /// Returns the inclusive start position.
    pub const fn start(self) -> PhysicalAddress {
        self.start
    }

    /// Returns the number of bytes between the endpoints.
    pub const fn length(self) -> u64 {
        let Self { start, end } = self;
        end.get() - start.get()
    }

    /// Returns the exclusive end position.
    pub const fn end(self) -> PhysicalAddress {
        self.end
    }

    /// Returns `true` if both endpoints of `required` lie within this interval's endpoints.
    ///
    /// An empty range at either endpoint is contained.
    pub fn contains(
        self,
        Self {
            start: required_start,
            end: required_end,
        }: Self,
    ) -> bool {
        let Self { start, end } = self;
        start <= required_start && required_end <= end
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::FileRange;
    use super::FileRegion;
    use super::FormatError;
    use super::PhysicalAddress;

    #[rstest]
    #[case::zero(0, 0)]
    #[case::empty_at_end(u64::MAX, 0)]
    #[case::ordinary(512, 96)]
    #[case::end(u64::MAX - 96, 96)]
    fn range_preserves_its_start_length_and_end(#[case] start: u64, #[case] length: u64) {
        let range = FileRange::new(PhysicalAddress::new(start), length).unwrap();
        assert_eq!(range.start().get(), start);
        assert_eq!(range.length(), length);
        assert_eq!(range.end().get(), start + length);
    }

    #[rstest]
    #[case::one(u64::MAX, 1)]
    #[case::many(u64::MAX - 1, 512)]
    fn range_rejects_end_overflow(#[case] start: u64, #[case] length: u64) {
        assert_eq!(
            FileRange::new(PhysicalAddress::new(start), length).unwrap_err(),
            FormatError::OffsetOverflow {
                offset: start,
                length
            }
        );
    }

    #[rstest]
    #[case::empty(0, &[])]
    #[case::ordinary(512, &[1, 2, 3])]
    #[case::large(u64::MAX - 3, &[1, 2, 3])]
    fn region_preserves_range_and_borrows_bytes(#[case] start: u64, #[case] bytes: &[u8]) {
        let range = FileRange::new(PhysicalAddress::new(start), bytes.len() as u64).unwrap();
        let region = FileRegion::new(range, bytes).unwrap();
        assert_eq!(region.range(), range);
        assert_eq!(region.bytes(), bytes);
        assert_eq!(region.bytes().as_ptr(), bytes.as_ptr());
    }

    #[rstest]
    #[case::short(4, &[1, 2, 3])]
    #[case::long(2, &[1, 2, 3])]
    #[case::empty_buffer(1, &[])]
    #[case::empty_range(0, &[1])]
    fn region_rejects_byte_length_mismatch(#[case] length: u64, #[case] bytes: &[u8]) {
        let range = FileRange::new(PhysicalAddress::new(512), length).unwrap();
        assert_eq!(
            FileRegion::new(range, bytes).unwrap_err(),
            FormatError::FileRegionLengthMismatch {
                range_length: length,
                bytes_length: bytes.len() as u64,
            }
        );
    }

    #[rstest]
    #[case::whole(512, 4, &[1, 2, 3, 4])]
    #[case::prefix(512, 2, &[1, 2])]
    #[case::middle(513, 2, &[2, 3])]
    #[case::suffix(514, 2, &[3, 4])]
    #[case::empty_start(512, 0, &[])]
    #[case::empty_middle(514, 0, &[])]
    #[case::empty_end(516, 0, &[])]
    fn subregion_borrows_the_contained_range(
        #[case] start: u64,
        #[case] length: u64,
        #[case] expected: &[u8],
    ) {
        let region = FileRegion::new(
            FileRange::new(PhysicalAddress::new(512), 4).unwrap(),
            &[1, 2, 3, 4],
        )
        .unwrap();
        let required = FileRange::new(PhysicalAddress::new(start), length).unwrap();
        assert!(region.range().contains(required));
        let subregion = region.subregion(required).unwrap();
        assert_eq!(subregion.range(), required);
        assert_eq!(subregion.bytes(), expected);
        assert_eq!(
            subregion.bytes().as_ptr(),
            region
                .bytes()
                .get(usize::try_from(start - 512).unwrap()..)
                .unwrap()
                .as_ptr()
        );
    }

    #[rstest]
    #[case::before(511, 1)]
    #[case::after(516, 1)]
    #[case::partial_before(511, 2)]
    #[case::partial_after(515, 2)]
    #[case::enclosing(511, 6)]
    #[case::empty_before(511, 0)]
    #[case::empty_after(517, 0)]
    fn subregion_rejects_an_uncontained_range(#[case] start: u64, #[case] length: u64) {
        let provided = FileRange::new(PhysicalAddress::new(512), 4).unwrap();
        let region = FileRegion::new(provided, &[1, 2, 3, 4]).unwrap();
        let required = FileRange::new(PhysicalAddress::new(start), length).unwrap();
        assert!(!provided.contains(required));
        assert_eq!(
            region.subregion(required).unwrap_err(),
            FormatError::InsufficientRegion {
                required_start: start,
                required_length: length,
                provided_start: 512,
                provided_length: 4,
            }
        );
    }

    #[test]
    fn an_empty_region_contains_only_its_position() {
        let range = FileRange::new(PhysicalAddress::new(u64::MAX), 0).unwrap();
        let region = FileRegion::new(range, &[]).unwrap();
        assert_eq!(region.subregion(range).unwrap(), region);
        let required = FileRange::new(PhysicalAddress::new(u64::MAX - 1), 0).unwrap();
        assert_eq!(
            region.subregion(required).unwrap_err(),
            FormatError::InsufficientRegion {
                required_start: u64::MAX - 1,
                required_length: 0,
                provided_start: u64::MAX,
                provided_length: 0,
            }
        );
    }

    #[test]
    fn subregion_indexes_local_bytes_at_a_large_physical_address() {
        let start = u64::MAX - 4;
        let region = FileRegion::new(
            FileRange::new(PhysicalAddress::new(start), 4).unwrap(),
            &[1, 2, 3, 4],
        )
        .unwrap();
        let required = FileRange::new(PhysicalAddress::new(start + 1), 2).unwrap();
        assert_eq!(region.subregion(required).unwrap().bytes(), &[2, 3]);
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn region_accepts_a_physical_address_above_usize_max() {
        let start = PhysicalAddress::new(usize::MAX as u64 + 1);
        let range = FileRange::new(start, 2).unwrap();
        let region = FileRegion::new(range, &[1, 2]).unwrap();
        assert_eq!(region.range().start(), start);
        assert_eq!(region.subregion(range).unwrap().bytes(), &[1, 2]);
    }
}
