use alloc::collections::BTreeMap;
use core::num::NonZeroU32;

use hdf5_pure_core::{DatatypeMemberName, FormatError};

use super::DatatypeVersion;
use super::description::NameLayout;
use crate::datatype;

/// The layout of a compound member's name, offset, and optional dimension block.
///
/// The layouts are defined in "The Datatype Message" of format specification version 4.0.
#[derive(Clone, Copy)]
pub(super) enum CompoundFields {
    /// A padded name and a four-byte offset in version 2.
    Legacy,
    /// A padded name, a four-byte offset, and a dimension block in version 1.
    LegacyWithShape,
    /// An unpadded name and a size-dependent offset width in versions 3 through 5.
    Packed(usize),
}

impl CompoundFields {
    pub(super) fn for_version(version: DatatypeVersion, size: u32) -> Self {
        match version {
            DatatypeVersion::One => Self::LegacyWithShape,
            DatatypeVersion::Two => Self::Legacy,
            DatatypeVersion::Three | DatatypeVersion::Four | DatatypeVersion::Five => {
                Self::Packed(datatype::offset_bytes_for_size(size))
            }
        }
    }

    pub(super) fn offset_width(self) -> usize {
        match self {
            Self::Legacy | Self::LegacyWithShape => size_of::<u32>(),
            Self::Packed(width) => width,
        }
    }

    pub(super) fn name_layout(self) -> NameLayout {
        match self {
            Self::Legacy | Self::LegacyWithShape => NameLayout::Padded,
            Self::Packed(_) => NameLayout::Unpadded,
        }
    }
}

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
    pub(in crate::datatype) fn insert(
        &mut self,
        offset: u64,
        field_size: NonZeroU32,
        index: usize,
        member_name: impl Fn(usize) -> Option<DatatypeMemberName>,
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
                first: name(first)?,
                second: name(second)?,
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
    pub(super) fn member_end(
        &self,
        offset: u64,
        field_size: u32,
        name: impl FnOnce() -> Result<DatatypeMemberName, FormatError>,
    ) -> Result<u64, FormatError> {
        // `H5O__dtype_decode_helper` in `H5Odtype.c`, HDF5 1.14.6, checks the member bounds.
        let Some(end) = offset
            .checked_add(u64::from(field_size))
            .filter(|&end| end <= u64::from(self.size))
        else {
            return Err(FormatError::CompoundFieldOutOfBounds {
                name: name()?,
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
