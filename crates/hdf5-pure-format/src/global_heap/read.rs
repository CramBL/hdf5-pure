use core::num::NonZeroU16;

use hdf5_pure_core::__private::{FileRange, FileRegion};

use super::collection::GlobalHeapCollection;
use super::index::IndexSet;
use super::object::{GlobalHeapObject, GlobalHeapSpan};
use crate::error::FormatError;

/// A lookup of the object at one index of a global heap collection.
///
/// The lookup checks every object header for duplicate indices, including those after the
/// matching object.
///
/// # Examples
///
/// ```
/// # use core::num::NonZeroU16;
/// # use hdf5_pure_core::__private::{FileRange, FileRegion};
/// # use hdf5_pure_core::PhysicalAddress;
/// # use hdf5_pure_format::__private::GlobalHeapCollectionLocator;
/// # use hdf5_pure_format::__private::LengthWidth;
/// # use hdf5_pure_format::__private::encode_global_heap_collection;
/// # use hdf5_pure_format::FormatError;
/// # let objects = [b"alpha".as_slice(), b"beta"];
/// # let bytes = encode_global_heap_collection(LengthWidth::Eight, &objects).unwrap();
/// # let range = FileRange::new(PhysicalAddress::new(512), bytes.len() as u64)?;
/// # let file = FileRegion::new(range, &bytes)?;
/// # let locator = GlobalHeapCollectionLocator::new(PhysicalAddress::new(512), LengthWidth::Eight);
/// # let collection = locator.parse(file.subregion(locator.parse_range()?)?)?;
/// let mut lookup = collection.lookup(NonZeroU16::new(2).unwrap());
/// let mut found = None;
/// while let Some(range) = lookup.parse_range() {
///     found = lookup.parse(file.subregion(range)?)?;
/// }
/// assert_eq!(found.unwrap().payload().parse(file)?, b"beta");
/// # Ok::<(), FormatError>(())
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlobalHeapLookup {
    walk: GlobalHeapWalk,
    index: NonZeroU16,
    found: Option<GlobalHeapObject>,
}

impl GlobalHeapLookup {
    pub(super) fn new(collection: GlobalHeapCollection, index: NonZeroU16) -> Self {
        Self {
            walk: collection.walk(),
            index,
            found: None,
        }
    }

    /// Parses the next object header or implied free tail from `region`.
    ///
    /// Returns the object at the index once the walk ends, and `None` at earlier steps.
    ///
    /// After an error from [`GlobalHeapWalk::parse`], [`parse_range`](Self::parse_range) returns
    /// the same range.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::GlobalHeapObjectNotFound`] if the collection does not hold an object
    /// at the index, and the errors [`GlobalHeapWalk::parse`] returns.
    pub fn parse(
        &mut self,
        region: FileRegion<'_>,
    ) -> Result<Option<GlobalHeapObject>, FormatError> {
        if let Some(object) = self.walk.parse(region)?
            && object.index() == self.index
        {
            self.found = Some(object);
        }
        if self.walk.parse_range().is_some() {
            return Ok(None);
        }
        self.found
            .map(Some)
            .ok_or(FormatError::GlobalHeapObjectNotFound {
                collection_address: self.walk.collection.range().start().get(),
                index: self.index.get(),
            })
    }

    /// Returns the range of the next object header or implied free tail, or `None` once the
    /// walk ends.
    pub const fn parse_range(&self) -> Option<FileRange> {
        self.walk.parse_range()
    }
}

/// A walk over the objects of a global heap collection.
///
/// The walk ends after the free space object, at the end of the collection, or after a tail too
/// short for an object header, which "Global Heap" of the [format specification, version
/// 4.0][spec] defines as free space with an implied header.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlobalHeapWalk {
    collection: GlobalHeapCollection,
    next: GlobalHeapSpan,
    seen: IndexSet,
}

impl GlobalHeapWalk {
    pub(super) fn new(collection: GlobalHeapCollection) -> Self {
        Self {
            collection,
            next: GlobalHeapSpan::Header(collection.first_header()),
            seen: IndexSet::new(),
        }
    }

    pub(super) fn into_complete_collection(self) -> Option<GlobalHeapCollection> {
        self.parse_range().is_none().then_some(self.collection)
    }

    /// Parses the next object header or implied free tail from `region`.
    ///
    /// Returns the object for a member header, or `None` for free space. The parser ignores the
    /// bytes of an implied free tail. Once the walk ends, every call returns `None`. After an error,
    /// [`parse_range`](Self::parse_range) returns the same range.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain the range
    /// [`parse_range`](Self::parse_range) returns,
    /// [`FormatError::GlobalHeapObjectOutsideCollection`] if the object, with its data padded to a
    /// multiple of 8 bytes, runs past the end of the collection,
    /// [`FormatError::MisplacedGlobalHeapFreeSpace`] if the free space object does not span the
    /// rest of the collection, and [`FormatError::DuplicateGlobalHeapObjectIndex`] if an earlier
    /// step returned an object at the same index.
    pub fn parse(
        &mut self,
        region: FileRegion<'_>,
    ) -> Result<Option<GlobalHeapObject>, FormatError> {
        let object = match self.next {
            GlobalHeapSpan::End => return Ok(None),
            GlobalHeapSpan::Header(header) => header.parse(self.collection, region)?,
            GlobalHeapSpan::ImpliedFree(range) => {
                region.subregion(range)?;
                None
            }
        };
        let next = object.map_or(Ok(GlobalHeapSpan::End), |object| {
            self.collection.next_span(object.range().end())
        })?;
        if let Some(object) = object
            && !self.seen.insert(object.index())
        {
            return Err(FormatError::DuplicateGlobalHeapObjectIndex {
                collection_address: self.collection.range().start().get(),
                index: object.index().get(),
            });
        }
        self.next = next;
        Ok(object)
    }

    /// Returns the range of the next object header or implied free tail, or `None` once the
    /// walk ends.
    pub const fn parse_range(&self) -> Option<FileRange> {
        self.next.range()
    }
}
