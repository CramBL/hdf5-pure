//! Plans packed global heap records from complete object-header framing.

use alloc::vec::Vec;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;

use super::GlobalHeapCollection;
use super::GlobalHeapObject;
use super::GlobalHeapWalk;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;

/// A header-by-header collection of global heap member metadata.
///
/// The caller reads the range [`parse_range`](Self::parse_range) returns and passes it to
/// [`parse`](Self::parse). After parsing the complete collection, [`finish`](Self::finish)
/// returns the members in physical order. Only object headers are needed.
///
/// See [`GlobalHeapMembers::compact`] for an example.
pub struct GlobalHeapMemberCollector {
    walk: GlobalHeapWalk,
    members: Vec<GlobalHeapObject>,
}

impl GlobalHeapMemberCollector {
    pub(super) fn new(collection: GlobalHeapCollection) -> Self {
        Self {
            walk: collection.walk(),
            members: Vec::new(),
        }
    }

    /// Parses the next object header and retains the member's metadata.
    ///
    /// After an error, [`parse_range`](Self::parse_range) returns the same range. Once the
    /// collection is complete, further calls succeed without adding members.
    ///
    /// # Errors
    ///
    /// Returns the errors from [`GlobalHeapWalk::parse`].
    pub fn parse(&mut self, region: FileRegion<'_>) -> Result<(), FormatError> {
        if let Some(object) = self.walk.parse(region)? {
            self.members.push(object);
        }
        Ok(())
    }

    /// Returns the next object-header range, or `None` after complete framing.
    pub const fn parse_range(&self) -> Option<FileRange> {
        self.walk.parse_range()
    }

    /// Returns the collected members, or `None` if an object header remains to be parsed.
    pub fn finish(self) -> Option<GlobalHeapMembers> {
        let Self { walk, members } = self;
        if walk.parse_range().is_some() {
            return None;
        }
        Some(GlobalHeapMembers {
            collection: walk.collection,
            members,
        })
    }
}

/// The member metadata from a complete global heap collection walk.
///
/// Each member refers to its original record and payload ranges. The collection address and
/// allocation length are preserved. The caller keeps the source bytes current while reusing
/// this metadata.
pub struct GlobalHeapMembers {
    collection: GlobalHeapCollection,
    members: Vec<GlobalHeapObject>,
}

impl GlobalHeapMembers {
    /// Plans packed records after removing the members selected by `remove`.
    ///
    /// Calls `remove` once per member in physical order. Retained records keep that order and
    /// occupy consecutive ranges after the collection header. Each move spans the original
    /// header, payload, and padding. Records already at their destination are omitted from
    /// [`GlobalHeapCompaction::moves`].
    ///
    /// If any member is removed, the plan includes a trailing free space object header. If no
    /// member is removed, both the moves and the free space header are absent. Planning uses
    /// the collected metadata without reading record bytes.
    ///
    /// Before applying the plan, the caller checks which members can be removed, source
    /// freshness, and permission to mutate the collection, including when every member is
    /// removed.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if a planned range ends past [`u64::MAX`], or
    /// [`FormatError::LengthTooLarge`] if the free space length exceeds the collection's length
    /// field width.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hdf5_pure_core::__private::{FileRange, FileRegion};
    /// # use hdf5_pure_core::PhysicalAddress;
    /// # use hdf5_pure_format::__private::{GlobalHeapCollectionLocator, LengthWidth};
    /// # use hdf5_pure_format::__private::encode_global_heap_collection;
    /// # use hdf5_pure_format::FormatError;
    /// # let bytes = encode_global_heap_collection(LengthWidth::Eight, &[b"alpha".as_slice(), b"beta"])?;
    /// # let range = FileRange::new(PhysicalAddress::new(512), bytes.len() as u64)?;
    /// # let file = FileRegion::new(range, &bytes)?;
    /// # let locator = GlobalHeapCollectionLocator::new(range.start(), LengthWidth::Eight);
    /// # let collection = locator.parse(file.subregion(locator.parse_range()?)?)?;
    /// let mut collector = collection.collect_members();
    /// while let Some(range) = collector.parse_range() {
    ///     collector.parse(file.subregion(range)?)?;
    /// }
    /// let members = collector.finish().unwrap();
    /// let plan = members.compact(|object| object.index().get() == 1)?;
    /// assert_eq!(plan.range(), collection.range());
    /// assert_eq!(plan.retained(), 1);
    /// assert_eq!(plan.moves().len(), 1);
    /// let movement = &plan.moves()[0];
    /// assert_eq!(movement.object().index().get(), 2);
    /// assert_eq!(movement.object().range().start().get(), 552);
    /// assert_eq!(movement.destination().start().get(), 528);
    /// assert_eq!(movement.destination().length(), 24);
    /// let free = plan.free_space().unwrap();
    /// assert_eq!(free.range().start().get(), 552);
    /// assert_eq!(free.range().end(), collection.range().end());
    /// assert_eq!(free.bytes().len(), 16);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn compact(
        &self,
        mut remove: impl FnMut(GlobalHeapObject) -> bool,
    ) -> Result<GlobalHeapCompaction, FormatError> {
        let Self {
            collection,
            members,
        } = self;
        let mut destination = collection.first.start();
        let mut moves = Vec::new();
        let mut removed = 0;
        for &object in members {
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
                super::FIXED_HEADER_LEN.to_usize()?,
                length,
                collection.lengths,
            )?;
            Some(GlobalHeapFreeSpace {
                range,
                header,
                bytes,
            })
        };
        Ok(GlobalHeapCompaction {
            collection: collection.range(),
            moves,
            free_space,
            retained: members.len() - removed,
        })
    }

    /// Returns the complete original collection allocation, including its header and free space.
    pub const fn range(&self) -> FileRange {
        self.collection.range()
    }

    /// Returns the allocated members in their original physical order.
    pub fn members(&self) -> &[GlobalHeapObject] {
        &self.members
    }
}

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

#[cfg(test)]
mod tests;
