use alloc::collections::BTreeMap;
use core::num::NonZeroU32;

use hdf5_pure_core::FormatError;

use super::DatatypeMessageMember;

/// Disjoint member byte ranges within one compound element.
///
/// Member indices refer to declaration order. Ranges are indexed by their starting byte
/// offsets. `H5O__dtype_decode_helper` in `H5Odtype.c`, HDF5 2.2.0, rejects overlapping
/// compound members.
pub(super) struct CompoundLayout {
    /// The compound element size in bytes.
    size: u32,
    /// The accepted ranges, keyed by their starting byte offsets.
    extents: BTreeMap<u64, MemberExtent>,
}

impl CompoundLayout {
    /// Starts an empty layout bounded by `size` bytes.
    pub(super) fn new(size: u32) -> Self {
        Self {
            size,
            extents: BTreeMap::new(),
        }
    }

    /// Checks and records one member's byte range.
    ///
    /// Each index must refer to the same member throughout insertion. `field_size` is the
    /// stored size at the source or output address width, or the minimum width during
    /// construction. Adjacent ranges are allowed.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::CompoundFieldOutOfBounds`] if the range exceeds the element,
    /// and [`FormatError::CompoundFieldOverlap`] if it intersects an accepted range.
    /// Reports conflicting names in byte-offset order, with encounter order for equal offsets.
    /// Returns [`FormatError::Internal`] if a member index is missing or `field_size` is zero.
    pub(super) fn insert(
        &mut self,
        members: &[DatatypeMessageMember],
        index: usize,
        field_size: u32,
    ) -> Result<(), FormatError> {
        let Self { size, extents } = self;
        let DatatypeMessageMember {
            name,
            byte_offset,
            datatype: _,
        } = members.get(index).ok_or_else(|| {
            FormatError::Internal("a compound layout refers to a missing member".into())
        })?;
        let field_size = NonZeroU32::new(field_size)
            .ok_or_else(|| {
                FormatError::Internal("a checked compound member has zero stored size".into())
            })?
            .get();
        let end = byte_offset
            .checked_add(u64::from(field_size))
            .filter(|&end| end <= u64::from(*size))
            .ok_or_else(|| FormatError::CompoundFieldOutOfBounds {
                name: name.clone(),
                offset: *byte_offset,
                field_size,
                compound_size: *size,
            })?;
        let previous = extents.range(..=*byte_offset).next_back();
        let next = extents.range(*byte_offset..).next();
        let conflict = match (previous, next) {
            (
                Some((
                    _,
                    MemberExtent {
                        end: previous_end,
                        index: previous_index,
                    },
                )),
                _,
            ) if *previous_end > *byte_offset => Some((*previous_index, index)),
            (
                _,
                Some((
                    next_start,
                    MemberExtent {
                        end: _,
                        index: next_index,
                    },
                )),
            ) if *next_start < end => Some((index, *next_index)),
            _ => None,
        };
        if let Some((first, second)) = conflict {
            let first = members.get(first).ok_or_else(|| {
                FormatError::Internal("a compound overlap refers to a missing member".into())
            })?;
            let second = members.get(second).ok_or_else(|| {
                FormatError::Internal("a compound overlap refers to a missing member".into())
            })?;
            return Err(FormatError::CompoundFieldOverlap {
                first: first.name.clone(),
                second: second.name.clone(),
            });
        }
        extents.insert(*byte_offset, MemberExtent { end, index });
        Ok(())
    }
}

/// The exclusive end and declaration index of an accepted member range.
struct MemberExtent {
    /// The first byte after the member.
    end: u64,
    /// The member's index in the declaration slice.
    index: usize,
}
