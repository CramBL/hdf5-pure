//! Immutable metadata bytes associated with physical file ranges.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::{Deref, Range};

use crate::address::PhysicalAddress;
use crate::error::FormatError;
use crate::region::{FileRange, FileRegion};

/// Immutable bytes associated with a checked physical file range.
///
/// The byte count equals the range length. Cloning and selecting a subregion share the
/// backing bytes. An owned allocation remains valid until its last owner is dropped.
/// The caller supplies the bytes' physical position. Source identity and revision are unchecked.
/// Separate reads of a mutable source may observe different revisions.
///
/// # Examples
///
/// Borrow a slice and select a contained physical range:
///
/// ```
/// use hdf5_pure_core::__private::{FileRange, MetadataBytes};
/// use hdf5_pure_core::PhysicalAddress;
///
/// let data = [1, 2, 3, 4];
/// let range = FileRange::new(PhysicalAddress::new(512), 4)?;
/// let bytes = MetadataBytes::borrowed(range, &data)?;
/// let selected = bytes.subregion(FileRange::new(PhysicalAddress::new(513), 2)?)?;
/// assert_eq!(selected.as_ref(), &[2, 3]);
/// assert_eq!(selected.region().range().start().get(), 513);
/// # Ok::<(), hdf5_pure_core::FormatError>(())
/// ```
///
/// Transfer a vector and retain its bytes through another owner:
///
/// ```
/// use hdf5_pure_core::__private::{FileRange, MetadataBytes};
/// use hdf5_pure_core::PhysicalAddress;
///
/// let range = FileRange::new(PhysicalAddress::new(512), 3)?;
/// let bytes = MetadataBytes::owned(range, vec![1, 2, 3])?;
/// let retained = bytes.clone();
/// drop(bytes);
/// assert_eq!(retained.as_ref(), &[1, 2, 3]);
/// # Ok::<(), hdf5_pure_core::FormatError>(())
/// ```
#[derive(Clone, Debug)]
pub struct MetadataBytes<'a> {
    backing: MetadataBacking<'a>,
    local: Range<usize>,
    range: FileRange,
}

impl<'a> MetadataBytes<'a> {
    /// Borrows bytes whose first byte is at physical file position `offset`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if `offset + bytes.len()` exceeds [`u64::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::MetadataBytes;
    ///
    /// let data = [1, 2, 3];
    /// let bytes = MetadataBytes::from_slice(512, &data)?;
    /// assert_eq!(bytes.offset(), 512);
    /// assert_eq!(bytes.as_ref(), &[1, 2, 3]);
    /// # Ok::<(), hdf5_pure_core::FormatError>(())
    /// ```
    pub fn from_slice(offset: u64, bytes: &'a [u8]) -> Result<Self, FormatError> {
        Self::borrowed(
            FileRange::new(PhysicalAddress::new(offset), bytes.len() as u64)?,
            bytes,
        )
    }

    /// Takes ownership of a vector whose first byte is at physical file position `offset`.
    ///
    /// Retains the vector's payload allocation. Clones share that allocation.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if `offset + bytes.len()` exceeds [`u64::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::MetadataBytes;
    ///
    /// let bytes = MetadataBytes::from_vec(512, vec![1, 2, 3])?;
    /// let retained = bytes.clone();
    /// drop(bytes);
    /// assert_eq!(retained.offset(), 512);
    /// assert_eq!(retained.as_ref(), &[1, 2, 3]);
    /// # Ok::<(), hdf5_pure_core::FormatError>(())
    /// ```
    pub fn from_vec(offset: u64, bytes: Vec<u8>) -> Result<Self, FormatError> {
        Self::owned(
            FileRange::new(PhysicalAddress::new(offset), bytes.len() as u64)?,
            bytes,
        )
    }

    /// Associates borrowed bytes with their physical range.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::FileRegionLengthMismatch`] if the byte count differs from the
    /// range length.
    pub fn borrowed(range: FileRange, bytes: &'a [u8]) -> Result<Self, FormatError> {
        Ok(Self::from_borrowed(FileRegion::new(range, bytes)?))
    }

    /// Associates a vector with its physical range, retaining its allocation.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::FileRegionLengthMismatch`] if the byte count differs from the
    /// range length.
    pub fn owned(range: FileRange, bytes: Vec<u8>) -> Result<Self, FormatError> {
        FileRegion::new(range, &bytes)?;
        let local = 0..bytes.len();
        Ok(Self {
            backing: MetadataBacking::Shared(Arc::new(bytes)),
            local,
            range,
        })
    }

    /// Returns the physical file position of the first byte.
    pub fn offset(&self) -> u64 {
        self.range.start().get()
    }

    /// Shares owned backing bytes independently of this owner's lifetime.
    ///
    /// Returns [`None`] for borrowed bytes. The returned owner has the same physical range
    /// and shares the entire backing allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::MetadataBytes;
    ///
    /// let bytes = MetadataBytes::from_vec(512, vec![1, 2, 3])?;
    /// let retained: MetadataBytes<'static> = bytes.shared().unwrap();
    /// drop(bytes);
    /// assert_eq!(retained.as_ref(), &[1, 2, 3]);
    /// assert!(MetadataBytes::from_slice(0, &[4, 5])?.shared().is_none());
    /// # Ok::<(), hdf5_pure_core::FormatError>(())
    /// ```
    pub fn shared(&self) -> Option<MetadataBytes<'static>> {
        match &self.backing {
            MetadataBacking::Borrowed(_) => None,
            MetadataBacking::Shared(bytes) => Some(MetadataBytes {
                backing: MetadataBacking::Shared(bytes.clone()),
                local: self.local.clone(),
                range: self.range,
            }),
        }
    }

    /// Returns the capacity of the owned backing allocation, or zero for borrowed bytes.
    ///
    /// Includes capacity outside the selected subregion and excludes ownership bookkeeping.
    pub fn retained_bytes(&self) -> usize {
        match &self.backing {
            MetadataBacking::Borrowed(_) => 0,
            MetadataBacking::Shared(bytes) => bytes.capacity(),
        }
    }

    /// Returns the selected bytes in a vector.
    ///
    /// Reuses the backing vector when this is its sole owner and selects the entire vector.
    /// Copies the selected bytes otherwise.
    pub fn into_vec(self) -> Vec<u8> {
        let Self {
            backing,
            local,
            range,
        } = self;
        match backing {
            MetadataBacking::Shared(bytes) if local == (0..bytes.len()) => {
                Arc::try_unwrap(bytes).unwrap_or_else(|bytes| bytes.as_ref().clone())
            }
            backing => {
                let bytes = MetadataBytes {
                    backing,
                    local,
                    range,
                };
                bytes.as_ref().to_vec()
            }
        }
    }

    /// Returns the physical range associated with these bytes.
    pub fn range(&self) -> FileRange {
        self.range
    }

    /// Borrows the bytes and their checked physical range.
    pub fn region(&self) -> FileRegion<'_> {
        FileRegion::from(self)
    }

    /// Shares the backing bytes for a contained physical range.
    ///
    /// An empty range at either endpoint is contained. A subregion retains the entire backing
    /// allocation when the bytes are owned.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `required` is outside this owner's range.
    pub fn subregion(&self, required: FileRange) -> Result<Self, FormatError> {
        let region = self.region().subregion(required)?;
        let error = FormatError::InsufficientRegion {
            required_start: required.start().get(),
            required_length: required.length(),
            provided_start: self.range.start().get(),
            provided_length: self.range.length(),
        };
        let start = usize::try_from(required.start().get() - self.range.start().get())
            .ok()
            .and_then(|offset| self.local.start.checked_add(offset))
            .ok_or(error.clone())?;
        let end = start.checked_add(region.bytes().len()).ok_or(error)?;
        Ok(Self {
            backing: self.backing.clone(),
            local: start..end,
            range: region.range(),
        })
    }

    fn from_borrowed(region: FileRegion<'a>) -> Self {
        Self {
            backing: MetadataBacking::Borrowed(region.bytes()),
            local: 0..region.bytes().len(),
            range: region.range(),
        }
    }
}

impl AsRef<[u8]> for MetadataBytes<'_> {
    fn as_ref(&self) -> &[u8] {
        let bytes = match &self.backing {
            MetadataBacking::Borrowed(bytes) => bytes,
            MetadataBacking::Shared(bytes) => bytes.as_slice(),
        };
        // Constructors cover the entire backing, and subregion accepts only contained ranges.
        bytes
            .get(self.local.clone())
            .expect("contained metadata bytes")
    }
}

impl Deref for MetadataBytes<'_> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

#[derive(Clone, Debug)]
enum MetadataBacking<'a> {
    Borrowed(&'a [u8]),
    Shared(Arc<Vec<u8>>),
}

/// A borrowed slice whose first byte has a known physical position.
///
/// Read offsets are relative to the slice's origin. Returned byte owners use physical ranges.
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::__private::{MetadataFrame};
/// use hdf5_pure_core::PhysicalAddress;
///
/// let data = [1, 2, 3, 4];
/// let frame = MetadataFrame::new(PhysicalAddress::new(512), &data)?;
/// let bytes = frame.read_at(1, 2)?;
/// assert_eq!(bytes.as_ref(), &[2, 3]);
/// assert_eq!(bytes.range().start().get(), 513);
/// # Ok::<(), hdf5_pure_core::FormatError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct MetadataFrame<'a> {
    region: FileRegion<'a>,
}

impl<'a> MetadataFrame<'a> {
    /// Associates the slice's first byte with `origin`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the slice's physical end exceeds [`u64::MAX`].
    pub fn new(origin: PhysicalAddress, bytes: &'a [u8]) -> Result<Self, FormatError> {
        Ok(Self {
            region: FileRegion::new(FileRange::new(origin, bytes.len() as u64)?, bytes)?,
        })
    }

    /// Returns the complete slice beginning at the frame's origin.
    pub fn bytes(self) -> &'a [u8] {
        self.region.bytes()
    }

    /// Returns the physical position of the slice's first byte.
    pub fn origin(self) -> PhysicalAddress {
        self.region.range().start()
    }

    /// Borrows bytes at an offset relative to the slice's origin.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the physical start or end exceeds [`u64::MAX`],
    /// and [`FormatError::InsufficientRegion`] if the range extends outside the slice.
    pub fn read_at(self, offset: u64, length: u64) -> Result<MetadataBytes<'a>, FormatError> {
        let range = FileRange::new(self.origin().checked_add(offset)?, length)?;
        Ok(MetadataBytes::from_borrowed(self.region.subregion(range)?))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{MetadataBytes, MetadataFrame};
    use crate::address::PhysicalAddress;
    use crate::error::FormatError;
    use crate::region::{FileRange, FileRegion};

    #[derive(Clone, Copy, Debug)]
    enum Backing {
        Borrowed,
        Shared,
    }

    fn bytes(backing: Backing, start: u64, data: &[u8]) -> MetadataBytes<'_> {
        let range = FileRange::new(PhysicalAddress::new(start), data.len() as u64).unwrap();
        match backing {
            Backing::Borrowed => MetadataBytes::borrowed(range, data).unwrap(),
            Backing::Shared => MetadataBytes::owned(range, data.to_vec()).unwrap(),
        }
    }

    #[rstest]
    #[case::borrowed(Backing::Borrowed)]
    #[case::shared(Backing::Shared)]
    fn clone_reuses_backing_and_preserves_physical_range(#[case] backing: Backing) {
        let data = [1, 2, 3, 4];
        let original = bytes(backing, u64::MAX - 4, &data);
        let cloned = original.clone();
        assert_eq!(cloned.as_ref(), data);
        assert_eq!(cloned.range(), original.range());
        assert_eq!(cloned.as_ref().as_ptr(), original.as_ref().as_ptr());
        assert_eq!(FileRegion::from(&cloned), original.region());
        if matches!(backing, Backing::Borrowed) {
            assert_eq!(original.as_ref().as_ptr(), data.as_ptr());
        }
    }

    #[rstest]
    #[case::borrowed(Backing::Borrowed)]
    #[case::shared(Backing::Shared)]
    fn empty_owner_preserves_its_physical_position(#[case] backing: Backing) {
        let owner = bytes(backing, u64::MAX, &[]);
        let cloned = owner.clone();
        assert_eq!(cloned.as_ref(), &[]);
        assert_eq!(cloned.range().start().get(), u64::MAX);
        assert_eq!(cloned.range().length(), 0);
        assert_eq!(
            cloned.subregion(cloned.range()).unwrap().region(),
            owner.region()
        );
    }

    #[test]
    fn owned_bytes_retain_the_vector_allocation() {
        let mut data = Vec::with_capacity(64);
        data.extend_from_slice(&[1, 2, 3, 4]);
        let ptr = data.as_ptr();
        let range = FileRange::new(PhysicalAddress::new(512), 4).unwrap();
        let bytes = MetadataBytes::owned(range, data).unwrap();
        assert_eq!(bytes.as_ref(), &[1, 2, 3, 4]);
        assert_eq!(bytes.as_ref().as_ptr(), ptr);
    }

    #[rstest]
    #[case::borrowed_short(Backing::Borrowed, 5)]
    #[case::borrowed_long(Backing::Borrowed, 3)]
    #[case::shared_short(Backing::Shared, 5)]
    #[case::shared_long(Backing::Shared, 3)]
    fn construction_rejects_extent_disagreement(#[case] backing: Backing, #[case] length: u64) {
        let range = FileRange::new(PhysicalAddress::new(512), length).unwrap();
        let data = [1, 2, 3, 4];
        let err = match backing {
            Backing::Borrowed => MetadataBytes::borrowed(range, &data),
            Backing::Shared => MetadataBytes::owned(range, data.to_vec()),
        }
        .unwrap_err();
        assert_eq!(
            err,
            FormatError::FileRegionLengthMismatch {
                range_length: length,
                bytes_length: 4,
            }
        );
    }

    #[rstest]
    #[case::borrowed(Backing::Borrowed)]
    #[case::shared(Backing::Shared)]
    fn nested_subregions_keep_local_offsets_and_survive_parent_drop(#[case] backing: Backing) {
        let data = [1, 2, 3, 4, 5, 6];
        let original = bytes(backing, u64::MAX - 6, &data);
        let middle = original
            .subregion(FileRange::new(PhysicalAddress::new(u64::MAX - 5), 4).unwrap())
            .unwrap();
        let nested = middle
            .subregion(FileRange::new(PhysicalAddress::new(u64::MAX - 4), 2).unwrap())
            .unwrap();
        assert_eq!(
            nested.as_ref().as_ptr(),
            original.as_ref().get(2..).unwrap().as_ptr()
        );
        drop(original);
        drop(middle);
        assert_eq!(nested.as_ref(), &[3, 4]);
        assert_eq!(nested.region().range().start().get(), u64::MAX - 4);
        assert_eq!(nested.region().range().length(), 2);
    }

    #[rstest]
    #[case::empty_start(512, 0, &[])]
    #[case::whole(512, 4, &[1, 2, 3, 4])]
    #[case::empty_end(516, 0, &[])]
    fn subregions_include_both_empty_endpoints(
        #[case] start: u64,
        #[case] length: u64,
        #[case] expected: &[u8],
    ) {
        let original = bytes(Backing::Shared, 512, &[1, 2, 3, 4]);
        let required = FileRange::new(PhysicalAddress::new(start), length).unwrap();
        let selected = original.subregion(required).unwrap();
        assert_eq!(selected.as_ref(), expected);
        assert_eq!(selected.range(), required);
    }

    #[rstest]
    #[case::before(511, 1)]
    #[case::after(516, 1)]
    #[case::partial(515, 2)]
    fn subregions_reject_ranges_outside_the_owner(#[case] start: u64, #[case] length: u64) {
        let original = bytes(Backing::Shared, 512, &[1, 2, 3, 4]);
        let required = FileRange::new(PhysicalAddress::new(start), length).unwrap();
        assert_eq!(
            original.subregion(required).unwrap_err(),
            FormatError::InsufficientRegion {
                required_start: start,
                required_length: length,
                provided_start: 512,
                provided_length: 4,
            }
        );
    }

    #[rstest]
    #[case::zero(0)]
    #[case::userblock(512)]
    #[case::above_32_bits(1_u64 << 40)]
    fn frame_resolves_its_origin_once_and_borrows_local_bytes(#[case] origin: u64) {
        let data = [1, 2, 3, 4];
        let frame = MetadataFrame::new(PhysicalAddress::new(origin), &data).unwrap();
        let selected = frame.read_at(1, 2).unwrap();
        assert_eq!(frame.origin().get(), origin);
        assert_eq!(selected.range().start().get(), origin + 1);
        assert_eq!(selected.as_ref(), &[2, 3]);
        assert_eq!(selected.as_ref().as_ptr(), data.get(1..).unwrap().as_ptr());
    }

    #[test]
    fn frame_rejects_a_physical_extent_overflow() {
        assert_eq!(
            MetadataFrame::new(PhysicalAddress::new(u64::MAX), &[1]).unwrap_err(),
            FormatError::OffsetOverflow {
                offset: u64::MAX,
                length: 1
            }
        );
    }

    #[test]
    fn frame_rejects_a_read_outside_its_extent() {
        let frame = MetadataFrame::new(PhysicalAddress::new(512), &[1, 2]).unwrap();
        assert_eq!(
            frame.read_at(1, 2).unwrap_err(),
            FormatError::InsufficientRegion {
                required_start: 513,
                required_length: 2,
                provided_start: 512,
                provided_length: 2,
            }
        );
    }

    #[test]
    fn frame_rejects_origin_resolution_overflow() {
        let frame = MetadataFrame::new(PhysicalAddress::new(u64::MAX), &[]).unwrap();
        assert_eq!(
            frame.read_at(1, 0).unwrap_err(),
            FormatError::OffsetOverflow {
                offset: u64::MAX,
                length: 1
            }
        );
    }
}
