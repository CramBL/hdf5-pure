use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;

use super::MessageRecordLayout;

/// A borrowed message region and its complete physical chunk range.
///
/// [`ObjectHeader::parse_from_source_with_chunk_observer`](super::ObjectHeader::parse_from_source_with_chunk_observer)
/// calls its observer after reading the chunk's local framing. Message compatibility and later
/// chunks may remain unchecked at that point. The caller must establish allocation ownership and
/// source freshness separately.
#[derive(Clone, Copy, Debug)]
pub struct ObjectHeaderChunk<'a> {
    allocation: FileRange,
    records: FileRegion<'a>,
    v2_layout: Option<MessageRecordLayout>,
}

impl<'a> ObjectHeaderChunk<'a> {
    /// Associates a borrowed message region with its checked chunk allocation.
    pub(super) fn new(
        allocation: FileRange,
        records: FileRegion<'a>,
        v2_layout: Option<MessageRecordLayout>,
    ) -> Self {
        debug_assert!(allocation.contains(records.range()));
        Self {
            allocation,
            records,
            v2_layout,
        }
    }

    /// Returns the complete physical range, including the chunk's framing bytes.
    ///
    /// Chunk 0 includes its header prefix and, for version 1, alignment padding. Version 2 chunks
    /// include their trailing checksum. This range contains [`records`](Self::records).
    pub const fn allocation(self) -> FileRange {
        self.allocation
    }

    /// Returns the borrowed message region with its physical range.
    ///
    /// Excludes header prefixes, continuation signatures, and checksums. The region may include
    /// NIL and continuation records and a trailing gap. Individual record boundaries may remain
    /// unchecked when the observer is called.
    pub const fn records(self) -> FileRegion<'a> {
        self.records
    }

    /// Returns the inherited version 2 record layout, or `None` for a version 1 header.
    pub const fn v2_layout(self) -> Option<MessageRecordLayout> {
        self.v2_layout
    }
}

#[cfg(test)]
mod tests;
