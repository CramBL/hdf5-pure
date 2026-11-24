use alloc::collections::BTreeMap;
use core::num::NonZeroU32;

use hdf5_pure_core::FormatError;

/// Disjoint member byte ranges within one compound element.
///
/// Member indices refer to declaration order. Ranges are indexed by their starting byte
/// offsets. `H5O__dtype_decode_helper` in `H5Odtype.c`, HDF5 2.2.0, rejects overlapping
/// compound members.
pub(in crate::datatype) struct CompoundLayout {
    /// The compound element size in bytes.
    size: u32,
    /// The accepted ranges, keyed by their starting byte offsets.
    extents: BTreeMap<u64, MemberExtent>,
}

impl CompoundLayout {
    /// Starts an empty layout bounded by `size` bytes.
    pub(in crate::datatype) fn new(size: u32) -> Self {
        Self {
            size,
            extents: BTreeMap::new(),
        }
    }

    /// Checks and records one member's byte range.
    ///
    /// Pass the stored member size when reading, the encoded member size when writing,
    /// or the size at the minimum address width when constructing a description.
    /// Use declaration indices throughout, including in `member_name`, which the checker calls
    /// only when reporting an error. Adjacent ranges are allowed.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::CompoundFieldOutOfBounds`] if the range exceeds the element,
    /// and [`FormatError::CompoundFieldOverlap`] if it intersects an accepted range.
    /// Reports conflicting names in byte-offset order, with encounter order for equal offsets.
    /// Returns [`FormatError::Internal`] if `member_name` returns `None` for an index needed
    /// in an error.
    pub(in crate::datatype) fn insert<'a>(
        &mut self,
        offset: u64,
        field_size: NonZeroU32,
        index: usize,
        member_name: impl Fn(usize) -> Option<&'a str>,
    ) -> Result<(), FormatError> {
        let name = |index| {
            member_name(index).ok_or_else(|| {
                FormatError::Internal("a compound layout refers to a missing member".into())
            })
        };
        let end = self.member_end(offset, field_size.get(), || name(index))?;
        let extents = &mut self.extents;
        let conflict = if let Some((_, previous)) = extents.range(..=offset).next_back()
            && previous.end > offset
        {
            Some((previous.index, index))
        } else if let Some((&next_start, next)) = extents.range(offset..).next()
            && next_start < end
        {
            Some((index, next.index))
        } else {
            None
        };
        if let Some((first, second)) = conflict {
            return Err(FormatError::CompoundFieldOverlap {
                first: name(first)?.into(),
                second: name(second)?.into(),
            });
        }
        extents.insert(offset, MemberExtent { end, index });
        Ok(())
    }

    /// Returns the exclusive end of a member within the compound element.
    ///
    /// For a zero-sized member, returns `offset` if it is at or before the end of the element.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::CompoundFieldOutOfBounds`] if the member exceeds the element.
    /// Calls `name` only to report this error and propagates any error from that call.
    pub(super) fn member_end<'a>(
        &self,
        offset: u64,
        field_size: u32,
        name: impl FnOnce() -> Result<&'a str, FormatError>,
    ) -> Result<u64, FormatError> {
        // `H5O__dtype_decode_helper` in `H5Odtype.c`, HDF5 1.14.6, checks the member bounds.
        let Some(end) = offset
            .checked_add(u64::from(field_size))
            .filter(|&end| end <= u64::from(self.size))
        else {
            return Err(FormatError::CompoundFieldOutOfBounds {
                name: name()?.into(),
                offset,
                field_size,
                compound_size: self.size,
            });
        };
        Ok(end)
    }
}

/// The exclusive end and declaration index of an accepted member range.
struct MemberExtent {
    /// The first byte after the member.
    end: u64,
    /// The member's index in the declaration slice.
    index: usize,
}
