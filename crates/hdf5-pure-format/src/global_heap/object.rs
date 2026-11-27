use core::num::NonZeroU16;

use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;

use super::collection::{self, ALIGNMENT, GlobalHeapCollection};
use crate::bytes;
use crate::error::FormatError;
use crate::width::LengthWidth;

/// The next parsing position within a collection's declared extent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GlobalHeapSpan {
    End,
    Header(GlobalHeapObjectHeader),
    ImpliedFree(FileRange),
}

impl GlobalHeapSpan {
    pub(super) const fn range(self) -> Option<FileRange> {
        match self {
            Self::End => None,
            Self::Header(header) => Some(header.range),
            Self::ImpliedFree(range) => Some(range),
        }
    }
}

/// The range of a complete object header within its collection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct GlobalHeapObjectHeader {
    range: FileRange,
}

impl GlobalHeapObjectHeader {
    /// Computes the padded object header range at `start`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the header ends past [`u64::MAX`].
    pub(super) fn at(start: PhysicalAddress, lengths: LengthWidth) -> Result<Self, FormatError> {
        Ok(Self {
            range: FileRange::new(start, collection::header_len(lengths))?,
        })
    }

    pub(super) const fn range(&self) -> FileRange {
        self.range
    }

    /// Parses the complete header from `region`, returning `None` for an explicit free-space record.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain the header range,
    /// [`FormatError::GlobalHeapObjectOutsideCollection`] if the padded member extent overflows or
    /// exceeds the collection boundary, and [`FormatError::MisplacedGlobalHeapFreeSpace`] if an
    /// explicit free-space record does not span the rest of the collection.
    pub(super) fn parse(
        &self,
        collection: GlobalHeapCollection,
        region: FileRegion<'_>,
    ) -> Result<Option<GlobalHeapObject>, FormatError> {
        let header = region.subregion(self.range)?;
        let range = self.range;
        let collection_end = collection.range().end().get();
        let outside = |index| FormatError::GlobalHeapObjectOutsideCollection {
            address: range.start().get(),
            index,
            collection_end,
        };
        let header_len = collection::header_len(collection.lengths());
        let mut cursor = bytes::ByteCursor::new(header.bytes());
        let index = cursor.u16()?;
        let reference_count = cursor.u16()?;
        cursor.array::<OBJECT_RESERVED_LEN>()?;
        let size = cursor.length(collection.lengths())?;
        let remaining = collection.remaining(range.start())?;
        let Some(index) = NonZeroU16::new(index) else {
            if size != remaining {
                return Err(FormatError::MisplacedGlobalHeapFreeSpace {
                    address: range.start().get(),
                    size,
                    collection_end,
                });
            }
            return Ok(None);
        };
        let extent = size
            .checked_next_multiple_of(ALIGNMENT as u64)
            .and_then(|padded| padded.checked_add(header_len))
            .filter(|&extent| extent <= remaining)
            .ok_or_else(|| outside(index.get()))?;
        Ok(Some(GlobalHeapObject {
            index,
            reference_count,
            range: FileRange::new(range.start(), extent)?,
            payload: GlobalHeapPayloadRef {
                range: FileRange::new(range.end(), size)?,
            },
        }))
    }
}

/// An object of a global heap collection, parsed from its object header.
///
/// The object is defined in "Global Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlobalHeapObject {
    index: NonZeroU16,
    reference_count: u16,
    range: FileRange,
    payload: GlobalHeapPayloadRef,
}

impl GlobalHeapObject {
    /// Returns the index of the object in its collection, the index a global heap ID stores.
    pub const fn index(&self) -> NonZeroU16 {
        self.index
    }

    /// Returns the "Reference Count" field of the object header.
    pub const fn reference_count(&self) -> u16 {
        self.reference_count
    }

    /// Returns the range of the object: its header and its data, padded to a multiple of 8 bytes.
    pub const fn range(&self) -> FileRange {
        self.range
    }

    /// Returns the reference to the object's data, which [`GlobalHeapPayloadRef::parse`] borrows
    /// from a region the caller reads.
    pub const fn payload(&self) -> GlobalHeapPayloadRef {
        self.payload
    }
}

/// The range of the data of a global heap object, the "Object Size" bytes after its object header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlobalHeapPayloadRef {
    range: FileRange,
}

impl GlobalHeapPayloadRef {
    /// Borrows the object's data from `region`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain the range
    /// [`range`](Self::range) returns.
    pub fn parse<'a>(&self, region: FileRegion<'a>) -> Result<&'a [u8], FormatError> {
        Ok(region.subregion(self.range)?.bytes())
    }

    /// Returns the range of the object's data, without its padding.
    pub const fn range(&self) -> FileRange {
        self.range
    }
}

/// The number of reserved bytes between the reference count and the object size of an object
/// header.
pub(super) const OBJECT_RESERVED_LEN: usize = 4;
