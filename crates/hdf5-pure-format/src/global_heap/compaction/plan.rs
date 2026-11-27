use alloc::vec::Vec;

use hdf5_pure_core::__private::FileRange;

use super::collector::GlobalHeapMembers;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::global_heap::GlobalHeapObject;
use crate::global_heap::collection::FIXED_HEADER_LEN;

/// The record moves and trailing free space header for one collection.
///
/// Moves are ordered by original physical position. The caller copies each complete record
/// toward the beginning of the collection in that order, allowing overlap within a move, then
/// writes the free space header. Indices, reference counts, payloads, and padding are copied
/// unchanged. The collection retains its original address and capacity.
/// Source access, mutation protection, and file-level reclamation are caller responsibilities.
pub struct GlobalHeapCompaction {
    collection: FileRange,
    moves: Vec<GlobalHeapMove>,
    free_space: Option<GlobalHeapFreeSpace>,
    retained: usize,
}

impl GlobalHeapCompaction {
    pub(super) fn new(
        members: &GlobalHeapMembers,
        mut remove: impl FnMut(GlobalHeapObject) -> bool,
    ) -> Result<Self, FormatError> {
        let collection = members.collection();
        let mut destination = collection.members_start();
        let mut moves = Vec::new();
        let mut removed = 0;
        for &object in members.members() {
            if remove(object) {
                removed += 1;
                continue;
            }
            let range = FileRange::new(destination, object.range().length())?;
            if destination != object.range().start() {
                moves.push(GlobalHeapMove {
                    object,
                    destination: range,
                });
            }
            destination = range.end();
        }
        let free_space = if removed == 0 {
            None
        } else {
            // Packing retained records never extends past their original checked ranges.
            let length = collection.range().end().get() - destination.get();
            let range = FileRange::new(destination, length)?;
            let header = FileRange::new(destination, FREE_HEADER_LEN as u64)?;
            let mut bytes = [0; FREE_HEADER_LEN];
            bytes::write_length_at(
                &mut bytes,
                FIXED_HEADER_LEN.to_usize()?,
                length,
                collection.lengths(),
            )?;
            Some(GlobalHeapFreeSpace {
                range,
                header,
                bytes,
            })
        };
        Ok(Self {
            collection: collection.range(),
            moves,
            free_space,
            retained: members.members().len() - removed,
        })
    }

    /// Returns the complete original collection allocation.
    pub const fn range(&self) -> FileRange {
        self.collection
    }

    /// Returns the records whose packed position differs from their original position.
    pub fn moves(&self) -> &[GlobalHeapMove] {
        &self.moves
    }

    /// Returns the trailing free space header if at least one member was removed.
    pub const fn free_space(&self) -> Option<&GlobalHeapFreeSpace> {
        self.free_space.as_ref()
    }

    /// Returns the number of retained members, including those already at their packed position.
    pub const fn retained(&self) -> usize {
        self.retained
    }
}

/// The original member metadata and packed destination for one record.
///
/// The destination has the same length as the original record and starts at a lower address.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlobalHeapMove {
    object: GlobalHeapObject,
    destination: FileRange,
}

impl GlobalHeapMove {
    /// Returns the member metadata with its original record and payload ranges.
    pub const fn object(&self) -> GlobalHeapObject {
        self.object
    }

    /// Returns the packed range for the complete record, including its header and padding.
    pub const fn destination(&self) -> FileRange {
        self.destination
    }
}

/// The trailing free space range and its serialized object header.
///
/// The free space remains part of a version 1 global heap collection's allocation. The header
/// stores index 0, a zero reference count, zero reserved bytes, and the free space length
/// including the header, as defined in "Global Heap" of the [format specification, version
/// 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
pub struct GlobalHeapFreeSpace {
    range: FileRange,
    header: FileRange,
    bytes: [u8; FREE_HEADER_LEN],
}

impl GlobalHeapFreeSpace {
    /// Returns the free space through the original collection's end, including its header.
    pub const fn range(&self) -> FileRange {
        self.range
    }

    /// Returns the range where the writer writes [`bytes`](Self::bytes).
    pub const fn header_range(&self) -> FileRange {
        self.header
    }

    /// Returns the serialized header, with the original collection's length-field width.
    pub const fn bytes(&self) -> &[u8; FREE_HEADER_LEN] {
        &self.bytes
    }
}

// H5HG_SIZEOF_OBJHDR in H5HGpkg.h, HDF5 2.2.0, pads every supported length width to 16 bytes.
const FREE_HEADER_LEN: usize = 16;
