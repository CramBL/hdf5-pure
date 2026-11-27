use alloc::vec::Vec;

use hdf5_pure_core::__private::{FileRange, FileRegion};

use super::plan::GlobalHeapCompaction;
use crate::error::FormatError;
use crate::global_heap::{GlobalHeapCollection, GlobalHeapObject, GlobalHeapWalk};

/// A header-by-header collection of global heap member metadata.
///
/// The caller reads the range [`parse_range`](Self::parse_range) returns and passes it to
/// [`parse`](Self::parse). After parsing the complete collection, [`finish`](Self::finish)
/// returns the members in physical order. The caller supplies each object header and any implied
/// free tail. Member payloads can be skipped.
///
/// See [`GlobalHeapMembers::compact`] for an example.
pub struct GlobalHeapMemberCollector {
    walk: GlobalHeapWalk,
    members: Vec<GlobalHeapObject>,
}

impl GlobalHeapMemberCollector {
    pub(in crate::global_heap) fn new(collection: GlobalHeapCollection) -> Self {
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

    /// Returns the range of the next object header or implied free tail, or `None` at the end.
    pub const fn parse_range(&self) -> Option<FileRange> {
        self.walk.parse_range()
    }

    /// Returns the collected members, or `None` if the walk has not reached the collection end.
    pub fn finish(self) -> Option<GlobalHeapMembers> {
        let Self { walk, members } = self;
        Some(GlobalHeapMembers {
            collection: walk.into_complete_collection()?,
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
        remove: impl FnMut(GlobalHeapObject) -> bool,
    ) -> Result<GlobalHeapCompaction, FormatError> {
        GlobalHeapCompaction::new(self, remove)
    }

    /// Returns the complete original collection allocation, including its header and free space.
    pub const fn range(&self) -> FileRange {
        self.collection.range()
    }

    /// Returns the allocated members in their original physical order.
    pub fn members(&self) -> &[GlobalHeapObject] {
        &self.members
    }

    pub(super) const fn collection(&self) -> &GlobalHeapCollection {
        &self.collection
    }
}
