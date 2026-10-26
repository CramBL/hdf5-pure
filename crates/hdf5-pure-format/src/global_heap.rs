//! The global heap collection parser and encoder.
//!
//! A collection holds global heap objects, such as the bytes of variable-length values, each at a
//! 1-based index. A global heap ID in the file holds the address of a collection and the index of
//! an object in it.
//!
//! [`GlobalHeapCollectionLocator`] parses one from the regions of the file the caller reads, one
//! header at a time.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::num::NonZeroU16;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::__private::PhysicalAddress;

use crate::bytes;
use crate::error::FormatError;
use crate::width;
use crate::width::LengthWidth;

/// The position of a global heap collection and the width of its length fields.
///
/// The caller reads the bytes of the range [`parse_range`](Self::parse_range) returns and passes
/// them to [`parse`](Self::parse), which parses the collection header. The
/// [`GlobalHeapCollection`] it returns starts a [`GlobalHeapWalk`] or a [`GlobalHeapLookup`],
/// which returns the range of each object header in turn. The collection is defined in
/// "Global Heap" of the [format specification, version 4.0][spec].
///
/// # Examples
///
/// ```
/// # use hdf5_pure_core::__private::{FileRange, FileRegion, PhysicalAddress};
/// # use hdf5_pure_format::__private::GlobalHeapCollectionLocator;
/// # use hdf5_pure_format::__private::LengthWidth;
/// # use hdf5_pure_format::__private::encode_global_heap_collection;
/// # use hdf5_pure_format::FormatError;
/// # let objects = [b"alpha".as_slice(), b"beta"];
/// # let bytes = encode_global_heap_collection(LengthWidth::Eight, &objects).unwrap();
/// # let range = FileRange::new(PhysicalAddress::new(512), bytes.len() as u64)?;
/// # let file = FileRegion::new(range, &bytes)?;
/// let locator = GlobalHeapCollectionLocator::new(PhysicalAddress::new(512), LengthWidth::Eight);
/// let collection = locator.parse(file.subregion(locator.parse_range()?)?)?;
///
/// let mut walk = collection.walk();
/// let mut objects = Vec::new();
/// while let Some(range) = walk.parse_range() {
///     if let Some(object) = walk.parse(file.subregion(range)?)? {
///         objects.push(object.payload().parse(file)?);
///     }
/// }
/// assert_eq!(objects, [b"alpha".as_slice(), b"beta"]);
/// # Ok::<(), FormatError>(())
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlobalHeapCollectionLocator {
    address: PhysicalAddress,
    lengths: LengthWidth,
}

impl GlobalHeapCollectionLocator {
    /// Creates a locator of the collection at `address`.
    ///
    /// `lengths` is the width the superblock's "Size of Lengths" field stores.
    pub fn new(address: PhysicalAddress, lengths: LengthWidth) -> Self {
        Self { address, lengths }
    }

    /// Parses the collection header from `region`.
    ///
    /// The parser ignores the reserved bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain the range
    /// [`parse_range`](Self::parse_range) returns, [`FormatError::InvalidGlobalHeapSignature`] if
    /// the collection does not begin with `GCOL`, [`FormatError::InvalidGlobalHeapVersion`] if its
    /// version is not 1, [`FormatError::GlobalHeapCollectionTooSmall`] if its size is below 4096
    /// bytes, [`FormatError::UnalignedGlobalHeapCollection`] if its size is not a multiple of 8,
    /// and [`FormatError::OffsetOverflow`] if the collection header or the collection ends past
    /// [`u64::MAX`].
    pub fn parse(&self, region: FileRegion<'_>) -> Result<GlobalHeapCollection, FormatError> {
        let Self { address, lengths } = *self;
        let mut fields = bytes::Fields::new(region.subregion(self.parse_range()?)?.bytes(), 0);
        if fields.array::<{ GCOL_SIGNATURE.len() }>()? != GCOL_SIGNATURE {
            return Err(FormatError::InvalidGlobalHeapSignature);
        }
        let version = fields.u8()?;
        if version != GCOL_VERSION {
            return Err(FormatError::InvalidGlobalHeapVersion(version));
        }
        fields.array::<GCOL_RESERVED_LEN>()?;
        let size = fields.length(lengths)?;
        if size < MIN_COLLECTION_SIZE as u64 {
            return Err(FormatError::GlobalHeapCollectionTooSmall {
                collection_address: address.get(),
                size,
            });
        }
        if !size.is_multiple_of(ALIGNMENT as u64) {
            return Err(FormatError::UnalignedGlobalHeapCollection {
                collection_address: address.get(),
                size,
            });
        }
        let header_len = header_len(lengths);
        Ok(GlobalHeapCollection {
            range: FileRange::new(address, size)?,
            lengths,
            first: FileRange::new(address.checked_add(header_len)?, header_len)?,
        })
    }

    /// Returns the range of the collection header: the signature, the version and the collection
    /// size, padded to a multiple of 8 bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the collection header ends past [`u64::MAX`].
    pub fn parse_range(&self) -> Result<FileRange, FormatError> {
        FileRange::new(self.address, header_len(self.lengths))
    }
}

/// A global heap collection, parsed from its header.
///
/// A [`GlobalHeapWalk`] parses every object of the collection in turn, and a [`GlobalHeapLookup`]
/// finds the object at one index. Both parse one object header per step, from a region the caller
/// reads. The collection is defined in "Global Heap" of the [format specification, version
/// 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlobalHeapCollection {
    range: FileRange,
    lengths: LengthWidth,
    first: FileRange,
}

impl GlobalHeapCollection {
    /// Starts a lookup of the object at `index`, at the first object header.
    pub fn lookup(&self, index: NonZeroU16) -> GlobalHeapLookup {
        GlobalHeapLookup {
            walk: self.walk(),
            index,
            found: None,
        }
    }

    /// Starts a walk over the objects of the collection, at the first object header.
    pub fn walk(&self) -> GlobalHeapWalk {
        GlobalHeapWalk {
            collection: *self,
            next: Some(self.first),
            seen: IndexSet::new(),
        }
    }

    /// Returns the range of the collection, from its address over the size its header stores.
    pub const fn range(&self) -> FileRange {
        self.range
    }

    /// Parses the object header in `header`, or returns `None` for the free space.
    ///
    /// `header` spans the range [`header_range`](Self::header_range) returns. A `header` shorter
    /// than an object header is a tail of the collection, and free space where its index is 0.
    fn parse_object(
        &self,
        header: FileRegion<'_>,
    ) -> Result<Option<GlobalHeapObject>, FormatError> {
        let range = header.range();
        let collection_end = self.range.end().get();
        let outside = |index| FormatError::GlobalHeapObjectOutsideCollection {
            address: range.start().get(),
            index,
            collection_end,
        };
        let header_len = header_len(self.lengths);
        let mut fields = bytes::Fields::new(header.bytes(), 0);
        let index = fields.u16()?;
        if range.length() < header_len {
            return match index {
                0 => Ok(None),
                index => Err(outside(index)),
            };
        }
        let reference_count = fields.u16()?;
        fields.array::<OBJECT_RESERVED_LEN>()?;
        let size = fields.length(self.lengths)?;
        let remaining = self.remaining(range.start())?;
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

    /// Returns the range of the object header at `start`, cut short at the end of the collection,
    /// or `None` at the end of the collection.
    fn header_range(&self, start: PhysicalAddress) -> Result<Option<FileRange>, FormatError> {
        Ok(match self.remaining(start)? {
            0 => None,
            remaining => Some(FileRange::new(
                start,
                remaining.min(header_len(self.lengths)),
            )?),
        })
    }

    /// Returns the number of bytes from `start` to the end of the collection.
    fn remaining(&self, start: PhysicalAddress) -> Result<u64, FormatError> {
        self.range
            .end()
            .get()
            .checked_sub(start.get())
            .ok_or_else(|| {
                FormatError::Internal(
                    "a global heap object header starts past its collection".into(),
                )
            })
    }
}

/// A lookup of the object at one index of a global heap collection, one object header per step.
///
/// The lookup parses every object header of the collection, the ones after the object included,
/// so it rejects a collection that holds an index twice wherever the second one sits.
///
/// # Examples
///
/// ```
/// # use core::num::NonZeroU16;
/// # use hdf5_pure_core::__private::{FileRange, FileRegion, PhysicalAddress};
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
    /// Parses the next object header from `region`, and returns `None` until the step that
    /// parses the last header, which returns the object at the index.
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
            && object.index == self.index
        {
            self.found = Some(object);
        }
        if self.walk.parse_range().is_some() {
            return Ok(None);
        }
        self.found
            .map(Some)
            .ok_or(FormatError::GlobalHeapObjectNotFound {
                collection_address: self.walk.collection.range.start().get(),
                index: self.index.get(),
            })
    }

    /// Returns the range of the object header the next step parses, or `None` once the last header
    /// is parsed.
    pub const fn parse_range(&self) -> Option<FileRange> {
        self.walk.parse_range()
    }
}

/// A walk over the objects of a global heap collection, one object header per step.
///
/// The walk ends after the free space object, at the end of the collection, or after a tail too
/// short for an object header, which "Global Heap" of the [format specification, version
/// 4.0][spec] defines as free space with an implied header.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlobalHeapWalk {
    collection: GlobalHeapCollection,
    next: Option<FileRange>,
    seen: IndexSet,
}

impl GlobalHeapWalk {
    /// Parses the next object header from `region`, and returns the object, or `None` for the free
    /// space.
    ///
    /// Once the walk has parsed the last header, every call returns `None`. After an error,
    /// [`parse_range`](Self::parse_range) returns the same range.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InsufficientRegion`] if `region` does not contain the range
    /// [`parse_range`](Self::parse_range) returns,
    /// [`FormatError::GlobalHeapObjectOutsideCollection`] if the object, with its data padded to a
    /// multiple of 8 bytes, runs past the end of the collection or a tail too short for an object
    /// header has a nonzero index,
    /// [`FormatError::MisplacedGlobalHeapFreeSpace`] if the free space object does not span the
    /// rest of the collection, and [`FormatError::DuplicateGlobalHeapObjectIndex`] if an earlier
    /// step returned an object at the same index.
    pub fn parse(
        &mut self,
        region: FileRegion<'_>,
    ) -> Result<Option<GlobalHeapObject>, FormatError> {
        let Some(range) = self.next else {
            return Ok(None);
        };
        let object = self.collection.parse_object(region.subregion(range)?)?;
        let next = object.map_or(Ok(None), |object| {
            self.collection.header_range(object.range.end())
        })?;
        if let Some(object) = object
            && !self.seen.insert(object.index)
        {
            return Err(FormatError::DuplicateGlobalHeapObjectIndex {
                collection_address: self.collection.range.start().get(),
                index: object.index.get(),
            });
        }
        self.next = next;
        Ok(object)
    }

    /// Returns the range of the object header the next step parses, or `None` once the last header
    /// is parsed.
    pub const fn parse_range(&self) -> Option<FileRange> {
        self.next
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

/// A set of global heap object indices, one bit for each `u16` value.
#[derive(Clone, Debug, Eq, PartialEq)]
struct IndexSet(Box<[u64; INDEX_WORDS]>);

impl IndexSet {
    fn new() -> Self {
        Self(Box::new([0; INDEX_WORDS]))
    }

    /// Adds `index` to the set, and returns whether it was newly inserted.
    fn insert(&mut self, index: NonZeroU16) -> bool {
        let index = index.get();
        // `index / 64` is at most `u16::MAX / 64`, the last of the `INDEX_WORDS` words.
        let word = &mut self.0[usize::from(index / 64)];
        let bit = 1 << (index % 64);
        let inserted = *word & bit == 0;
        *word |= bit;
        inserted
    }
}

// libhdf5 pads the collection header and each object header to a multiple of 8 bytes at
// every length width, in `H5HG_SIZEOF_HDR` and `H5HG_SIZEOF_OBJHDR` (`H5HGpkg.h`, HDF5 1.8.23 to
// 2.2.0).
fn header_len(lengths: LengthWidth) -> u64 {
    (FIXED_HEADER_LEN + u64::from(lengths.get())).next_multiple_of(ALIGNMENT as u64)
}

/// Encodes a global heap collection that holds `objects` at the indices 1, 2, and so on, in order.
///
/// Every length field is `length_width` bytes wide. The collection header, each object header and
/// each object's data are padded to a multiple of 8 bytes, and a free space object at index 0
/// spans the rest of the collection. A collection smaller than 4096 bytes, the minimum collection
/// size the specification defines, is padded to 4096. Each object stores a reference count of 1.
/// The collection is defined in "Global Heap" of the [format specification, version 4.0][spec].
///
/// libhdf5 pads a header to a multiple of 8 bytes at every length width (`H5HG_SIZEOF_HDR`
/// and `H5HG_SIZEOF_OBJHDR` in `H5HGpkg.h`, HDF5 2.2.0).
///
/// # Errors
///
/// Returns [`GlobalHeapCollectionError::TooManyObjects`] if `objects` has more than
/// [`GLOBAL_HEAP_MAX_OBJECTS`] objects, and [`GlobalHeapCollectionError::TooLarge`] if the size of
/// the collection does not fit a length field of `length_width` bytes.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
pub fn encode_global_heap_collection(
    length_width: LengthWidth,
    objects: &[&[u8]],
) -> Result<Vec<u8>, GlobalHeapCollectionError> {
    if objects.len() > GLOBAL_HEAP_MAX_OBJECTS {
        return Err(GlobalHeapCollectionError::TooManyObjects {
            count: objects.len(),
        });
    }
    // The length of the collection header, sig(4) + ver(1) + reserved(3) + `collection_size`, and
    // of each object header, index(2) + reference count(2) + reserved(4) + size, padded to
    // `ALIGNMENT`.
    let header_size = (8 + usize::from(length_width.get())).next_multiple_of(ALIGNMENT);
    let collection_size = header_size
        + objects
            .iter()
            .map(|obj| header_size + obj.len().next_multiple_of(ALIGNMENT))
            .sum::<usize>()
        + header_size; // free space marker (full object header size)
    let padded_collection = collection_size
        .max(MIN_COLLECTION_SIZE)
        .next_multiple_of(ALIGNMENT);
    // Every length the collection stores is at most its own size.
    if !width::length_width_holds(length_width, padded_collection as u64) {
        return Err(GlobalHeapCollectionError::TooLarge {
            size: padded_collection,
            length_width,
        });
    }

    let mut buf = Vec::with_capacity(padded_collection);
    // Header
    buf.extend_from_slice(&GCOL_SIGNATURE);
    buf.push(GCOL_VERSION);
    buf.extend_from_slice(&[0u8; 3]); // reserved
    bytes::write_length(&mut buf, padded_collection as u64, length_width);
    buf.resize(header_size, 0);

    // Objects (1-based indices)
    for (i, obj) in objects.iter().enumerate() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the check above bounds the object count by `u16::MAX`, so the 1-based \
                      index `i + 1` fits the 2-byte heap object index field"
        )]
        let index = (i + 1) as u16;
        let start = buf.len();
        buf.extend_from_slice(&index.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // reference count
        buf.extend_from_slice(&[0u8; 4]); // reserved
        bytes::write_length(&mut buf, obj.len() as u64, length_width);
        buf.resize(start + header_size, 0);
        buf.extend_from_slice(obj);
        // Pad to 8-byte boundary
        buf.resize(buf.len().next_multiple_of(ALIGNMENT), 0);
    }

    // Free space marker (index 0): the C library uses this size as the total
    // skip distance from the start of the object (including its header), so
    // it must equal the remaining bytes in the collection from this point.
    let free_total_size = padded_collection - buf.len();
    buf.extend_from_slice(&0u16.to_le_bytes()); // index 0
    buf.extend_from_slice(&0u16.to_le_bytes()); // reference count
    buf.extend_from_slice(&[0u8; 4]); // reserved
    bytes::write_length(&mut buf, free_total_size as u64, length_width); // size

    // Pad collection to full size
    buf.resize(padded_collection, 0);

    Ok(buf)
}

/// The error [`encode_global_heap_collection`] returns for objects one collection cannot hold.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlobalHeapCollectionError {
    /// The size of the collection does not fit a length field of `length_width` bytes.
    TooLarge {
        /// The size in bytes of the collection, padding included.
        size: usize,
        /// The width of the collection's length fields.
        length_width: LengthWidth,
    },
    /// The caller passed more objects than [`GLOBAL_HEAP_MAX_OBJECTS`].
    TooManyObjects {
        /// The number of objects the caller passed.
        count: usize,
    },
}

/// The most objects one global heap collection holds, 65,535.
///
/// The "Heap Object Index" field is 2 bytes wide, and index 0 is the free space. A writer with more
/// objects writes them to several collections, each with its objects at indices from 1. The field
/// is defined in "Global Heap" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
pub const GLOBAL_HEAP_MAX_OBJECTS: usize = u16::MAX as usize;

/// The signature a global heap collection begins with.
const GCOL_SIGNATURE: [u8; 4] = *b"GCOL";

/// The version of a global heap collection, the one version the specification defines.
const GCOL_VERSION: u8 = 1;

/// The number of reserved bytes between the version and the collection size of a collection
/// header.
const GCOL_RESERVED_LEN: usize = 3;

/// The number of reserved bytes between the reference count and the object size of an object
/// header.
const OBJECT_RESERVED_LEN: usize = 4;

/// The number of bytes before the length field of a collection header and of an object header.
const FIXED_HEADER_LEN: u64 = 8;

/// The number of 64-bit words in an [`IndexSet`].
const INDEX_WORDS: usize = (u16::MAX as usize + 1) / 64;

// The collection header, each object header and each object's data are padded to a multiple of
// this many bytes, `H5HG_ALIGNMENT` in the C library (`H5HGpkg.h`, HDF5 2.2.0).
const ALIGNMENT: usize = 8;

// The minimum collection size "Global Heap" of the format specification, version 4.0, defines,
// and `H5HG_MINSIZE` in the C library (`H5HGpkg.h`, HDF5 2.2.0).
const MIN_COLLECTION_SIZE: usize = 4096;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::global_heap;
    use test_util::widths::Widths;

    use super::*;

    #[rstest]
    #[case::padded_to_the_minimum(&[b"alpha".as_slice(), b"".as_slice()], MIN_COLLECTION_SIZE)]
    #[case::past_the_minimum(&[[7u8; 5000].as_slice()], 5048)]
    fn an_encoded_collection_parses_back_to_its_objects(
        #[case] objects: &[&[u8]],
        #[case] size: usize,
    ) {
        let encoded = encode_global_heap_collection(LengthWidth::Eight, objects).unwrap();

        assert_eq!(encoded.len(), size);
        assert_eq!(bytes::read_length(&encoded, 8, 8), Ok(size as u64));
        let expected: Vec<(u16, &[u8])> = (1..).zip(objects.iter().copied()).collect();
        assert_eq!(
            walk_objects(region(0, &encoded), LengthWidth::Eight).unwrap(),
            expected
        );

        let free_space = objects
            .iter()
            .map(|object| 16 + object.len().next_multiple_of(8))
            .sum::<usize>()
            + 16;
        assert_eq!(encoded[free_space..free_space + 8], [0; 8]);
        assert_eq!(
            bytes::read_length(&encoded, free_space + 8, 8),
            Ok((size - free_space) as u64)
        );
    }

    #[rstest]
    #[case::two_byte_lengths(LengthWidth::Two)]
    #[case::four_byte_lengths(LengthWidth::Four)]
    fn the_encoder_pads_each_narrow_length_header_to_eight_bytes(#[case] width: LengthWidth) {
        let bytes = encode_global_heap_collection(width, &[b"abc"]).unwrap();
        let length = usize::from(width.get());

        let mut expected = GCOL_SIGNATURE.to_vec();
        expected.extend_from_slice(&[GCOL_VERSION, 0, 0, 0]);
        expected.extend_from_slice(&4096u64.to_le_bytes()[..length]);
        expected.resize(16, 0);
        expected.extend_from_slice(&[1, 0, 1, 0, 0, 0, 0, 0]);
        expected.extend_from_slice(&3u64.to_le_bytes()[..length]);
        expected.resize(32, 0);
        expected.extend_from_slice(b"abc\0\0\0\0\0");
        // The free space object at byte 40 spans the rest of the 4096 bytes.
        expected.extend_from_slice(&[0; 8]);
        expected.extend_from_slice(&(4096u64 - 40).to_le_bytes()[..length]);
        expected.resize(4096, 0);
        assert_eq!(bytes, expected);
    }

    // 16 + 16 + 65,480 + 16 bytes fit the 65,535 of a 2-byte length, and 65,481 bytes of data pad
    // to 65,488 and a collection of 65,536.
    #[rstest]
    #[case::fits(65_480, Ok(65_528))]
    #[case::overflows(
        65_481,
        Err(GlobalHeapCollectionError::TooLarge {
            size: 65_536,
            length_width: LengthWidth::Two,
        })
    )]
    fn a_two_byte_length_collection_holds_at_most_its_field_maximum(
        #[case] object_len: usize,
        #[case] expected: Result<usize, GlobalHeapCollectionError>,
    ) {
        let object = vec![0u8; object_len];

        assert_eq!(
            encode_global_heap_collection(LengthWidth::Two, &[&object]).map(|bytes| bytes.len()),
            expected
        );
    }

    #[test]
    fn the_encoder_rejects_more_than_global_heap_max_objects() {
        let objects = vec![b"".as_slice(); GLOBAL_HEAP_MAX_OBJECTS + 1];

        assert_eq!(
            encode_global_heap_collection(LengthWidth::Eight, &objects),
            Err(GlobalHeapCollectionError::TooManyObjects { count: 65_536 })
        );
    }

    fn region(start: u64, bytes: &[u8]) -> FileRegion<'_> {
        FileRegion::new(
            FileRange::new(PhysicalAddress::new(start), bytes.len() as u64).unwrap(),
            bytes,
        )
        .unwrap()
    }

    /// Parses the header of the collection at the start of `image`.
    fn collection_at(image: FileRegion<'_>, lengths: LengthWidth) -> GlobalHeapCollection {
        let locator = GlobalHeapCollectionLocator::new(image.range().start(), lengths);
        locator
            .parse(image.subregion(locator.parse_range().unwrap()).unwrap())
            .unwrap()
    }

    /// Walks the collection at the start of `image`, and returns the index and the data of each
    /// object.
    fn walk_objects(
        image: FileRegion<'_>,
        lengths: LengthWidth,
    ) -> Result<Vec<(u16, &[u8])>, FormatError> {
        let mut walk = collection_at(image, lengths).walk();
        let mut objects = Vec::new();
        while let Some(range) = walk.parse_range() {
            if let Some(object) = walk.parse(image.subregion(range)?)? {
                objects.push((object.index().get(), object.payload().parse(image)?));
            }
        }
        Ok(objects)
    }

    /// Returns a collection of `size` bytes that holds `objects`, each at its index with a
    /// reference count of 1, and a free space object where the tail has room for its header.
    fn padded_collection(lengths: LengthWidth, size: usize, objects: &[(u16, &[u8])]) -> Vec<u8> {
        let objects: Vec<_> = objects
            .iter()
            .map(|&(index, data)| global_heap::Object::new(index, data))
            .collect();
        global_heap::collection_of_size(&objects, Widths::new(8, usize::from(lengths.get())), size)
    }

    #[rstest]
    #[case::empty(LengthWidth::Eight, 0, &[])]
    #[case::two_byte_lengths(LengthWidth::Two, 0, &[b"alpha".as_slice(), b"".as_slice(), b"gamma!!!".as_slice()])]
    #[case::four_byte_lengths(LengthWidth::Four, 0, &[b"alpha".as_slice(), b"".as_slice(), b"gamma!!!".as_slice()])]
    #[case::eight_byte_lengths(LengthWidth::Eight, 0, &[b"alpha".as_slice(), b"".as_slice(), b"gamma!!!".as_slice()])]
    #[case::large_object(LengthWidth::Four, 0, &[LARGE_OBJECT.as_slice()])]
    #[case::above_four_gib(LengthWidth::Eight, (1 << 32) + 4096, &[b"alpha".as_slice(), b"beta".as_slice()])]
    fn a_walk_reads_back_the_objects_the_encoder_writes(
        #[case] lengths: LengthWidth,
        #[case] address: u64,
        #[case] objects: &[&[u8]],
    ) {
        let encoded = encode_global_heap_collection(lengths, objects).unwrap();
        let expected: Vec<(u16, &[u8])> = (1..).zip(objects.iter().copied()).collect();
        assert_eq!(
            walk_objects(region(address, &encoded), lengths).unwrap(),
            expected
        );
    }

    #[test]
    fn a_walk_reads_many_objects() {
        let payloads: Vec<[u8; 2]> = (0..2000u16).map(u16::to_le_bytes).collect();
        let objects: Vec<&[u8]> = payloads.iter().map(<[u8; 2]>::as_slice).collect();
        let encoded = encode_global_heap_collection(LengthWidth::Two, &objects).unwrap();
        let expected: Vec<(u16, &[u8])> = (1..).zip(objects.iter().copied()).collect();
        assert_eq!(
            walk_objects(region(0, &encoded), LengthWidth::Two).unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::empty(&[], &[])]
    #[case::two(&[b"alpha".as_slice(), b"beta".as_slice()], &[(1, 32, 5), (2, 56, 4)])]
    #[case::large(
        &[b"head".as_slice(), [7u8; 5000].as_slice(), b"tail".as_slice()],
        &[(1, 32, 4), (2, 56, 5000), (3, 5072, 4)]
    )]
    fn a_walk_frames_each_payload_after_its_object_header(
        #[case] objects: &[&[u8]],
        #[case] expected: &[(u16, u64, u64)],
    ) {
        let encoded = encode_global_heap_collection(LengthWidth::Eight, objects).unwrap();
        let image = region(0, &encoded);
        let mut walk = collection_at(image, LengthWidth::Eight).walk();
        let mut walked = Vec::new();
        while let Some(range) = walk.parse_range() {
            walked.extend(walk.parse(image.subregion(range).unwrap()).unwrap());
        }
        let framed: Vec<(u16, u64, u64)> = walked
            .iter()
            .map(|object| {
                let payload = object.payload().range();
                (
                    object.index().get(),
                    payload.start().get(),
                    payload.length(),
                )
            })
            .collect();
        assert_eq!(framed, expected);
    }

    #[test]
    fn a_lookup_reads_the_object_headers_and_its_own_payload_alone() {
        let objects: [&[u8]; 4] = [b"one", &[2; 100], b"three", &[4; 200]];
        let encoded = encode_global_heap_collection(LengthWidth::Eight, &objects).unwrap();
        let image = region(512, &encoded);
        let mut lookup =
            collection_at(image, LengthWidth::Eight).lookup(NonZeroU16::new(3).unwrap());
        let mut requested = Vec::new();
        let object = loop {
            let range = lookup.parse_range().unwrap();
            requested.push(range.start().get()..range.end().get());
            if let Some(object) = lookup.parse(image.subregion(range).unwrap()).unwrap() {
                break object;
            }
        };

        // The headers of the 3-, 100-, 5-, and 200-byte objects start at 528, 552, 672, and 696,
        // and the free space header at 912.
        assert_eq!(
            requested,
            [528..544, 552..568, 672..688, 696..712, 912..928]
        );
        assert_eq!(object.index().get(), 3);
        assert_eq!(object.reference_count(), 1);
        assert_eq!(
            object.range().start().get()..object.range().end().get(),
            672..696
        );
        assert_eq!(
            object.payload().parse(region(688, b"three")).unwrap(),
            b"three"
        );
        assert_eq!(lookup.parse_range(), None);
    }

    #[test]
    fn a_walk_frames_an_8_gib_object_from_the_object_headers_alone() {
        // An 8 GiB collection at 1 TiB, filled by one object up to a 16-byte free space object.
        let address = 1 << 40;
        let size = 1u64 << 33;
        let mut collection_header = GCOL_SIGNATURE.to_vec();
        collection_header.extend_from_slice(&[GCOL_VERSION, 0, 0, 0]);
        collection_header.extend_from_slice(&size.to_le_bytes());
        let mut object = vec![1, 0, 1, 0, 0, 0, 0, 0];
        object.extend_from_slice(&(size - 48).to_le_bytes());
        let mut free_space = vec![0; 8];
        free_space.extend_from_slice(&16u64.to_le_bytes());
        let headers = [
            region(address + 16, &object),
            region(address + size - 16, &free_space),
        ];

        let mut walk =
            collection_at(region(address, &collection_header), LengthWidth::Eight).walk();
        let objects: Vec<GlobalHeapObject> = headers
            .iter()
            .filter_map(|&header| {
                assert_eq!(walk.parse_range(), Some(header.range()));
                walk.parse(header).unwrap()
            })
            .collect();

        assert_eq!(walk.parse_range(), None);
        let [object] = objects.as_slice() else {
            panic!("expected one object, got {objects:?}");
        };
        let payload = object.payload().range();
        assert_eq!(payload.start().get(), address + 32);
        assert_eq!(payload.length(), size - 48);
    }

    #[rstest]
    #[case::below_the_minimum(
        4095,
        Err(FormatError::GlobalHeapCollectionTooSmall { collection_address: 64, size: 4095 })
    )]
    #[case::the_minimum(4096, Ok(4096))]
    #[case::one_alignment_past_the_minimum(4104, Ok(4104))]
    #[case::one_byte_past_the_minimum(
        4097,
        Err(FormatError::UnalignedGlobalHeapCollection { collection_address: 64, size: 4097 })
    )]
    #[case::half_an_alignment_past_the_minimum(
        4100,
        Err(FormatError::UnalignedGlobalHeapCollection { collection_address: 64, size: 4100 })
    )]
    fn a_collection_size_is_at_least_the_minimum_and_aligned(
        #[case] size: u64,
        #[case] expected: Result<u64, FormatError>,
    ) {
        let mut collection_header = GCOL_SIGNATURE.to_vec();
        collection_header.extend_from_slice(&[GCOL_VERSION, 0, 0, 0]);
        collection_header.extend_from_slice(&size.to_le_bytes());
        assert_eq!(
            GlobalHeapCollectionLocator::new(PhysicalAddress::new(64), LengthWidth::Eight)
                .parse(region(64, &collection_header))
                .map(|collection| collection.range().length()),
            expected
        );
    }

    #[rstest]
    #[case::signature(0, b'X', FormatError::InvalidGlobalHeapSignature)]
    #[case::version(4, 2, FormatError::InvalidGlobalHeapVersion(2))]
    fn a_locator_rejects_a_header_that_is_not_a_collection(
        #[case] at: usize,
        #[case] value: u8,
        #[case] expected: FormatError,
    ) {
        let mut encoded = encode_global_heap_collection(LengthWidth::Eight, &[]).unwrap();
        encoded[at] = value;
        assert_eq!(
            GlobalHeapCollectionLocator::new(PhysicalAddress::new(0), LengthWidth::Eight)
                .parse(region(0, &encoded))
                .unwrap_err(),
            expected
        );
    }

    #[rstest]
    #[case::requested_first(1)]
    #[case::requested_between(2)]
    fn a_lookup_rejects_a_duplicate_index_after_the_requested_object(#[case] index: u16) {
        let bytes = padded_collection(
            LengthWidth::Eight,
            4096,
            &[(1, b"first"), (2, b"second"), (1, b"again")],
        );
        let image = region(0, &bytes);
        let mut lookup =
            collection_at(image, LengthWidth::Eight).lookup(NonZeroU16::new(index).unwrap());
        let err = loop {
            let range = lookup.parse_range().unwrap();
            match lookup.parse(image.subregion(range).unwrap()) {
                Ok(found) => assert_eq!(found, None),
                Err(err) => break err,
            }
        };
        assert_eq!(
            err,
            FormatError::DuplicateGlobalHeapObjectIndex {
                collection_address: 0,
                index: 1,
            }
        );
    }

    #[test]
    fn a_lookup_of_an_absent_index_walks_to_the_end_and_reports_it() {
        let bytes = padded_collection(LengthWidth::Eight, 4096, &[(1, b"first"), (3, b"third")]);
        let image = region(0, &bytes);
        let mut lookup =
            collection_at(image, LengthWidth::Eight).lookup(NonZeroU16::new(2).unwrap());
        let mut steps = 0;
        let err = loop {
            steps += 1;
            let range = lookup.parse_range().unwrap();
            if let Err(err) = lookup.parse(image.subregion(range).unwrap()) {
                break err;
            }
        };
        assert_eq!(steps, 3);
        assert_eq!(
            err,
            FormatError::GlobalHeapObjectNotFound {
                collection_address: 0,
                index: 2,
            }
        );
    }

    #[rstest]
    #[case::free_space_ends_at_the_collection_end(4096, 64, 0, 4032, Ok(vec![1, 2]))]
    #[case::free_space_followed_by_bytes(
        4096,
        64,
        0,
        4024,
        Err(FormatError::MisplacedGlobalHeapFreeSpace { address: 64, size: 4024, collection_end: 4096 })
    )]
    #[case::free_space_past_the_collection_end(
        4096,
        64,
        0,
        4040,
        Err(FormatError::MisplacedGlobalHeapFreeSpace { address: 64, size: 4040, collection_end: 4096 })
    )]
    #[case::free_space_followed_by_an_object(
        4096,
        16,
        0,
        24,
        Err(FormatError::MisplacedGlobalHeapFreeSpace { address: 16, size: 24, collection_end: 4096 })
    )]
    #[case::object_past_the_collection_end(
        4096,
        40,
        2,
        4041,
        Err(FormatError::GlobalHeapObjectOutsideCollection {
            address: 40,
            index: 2,
            collection_end: 4096,
        })
    )]
    #[case::object_size_overflows(
        4096,
        40,
        2,
        u64::MAX,
        Err(FormatError::GlobalHeapObjectOutsideCollection {
            address: 40,
            index: 2,
            collection_end: 4096,
        })
    )]
    #[case::object_fills_the_collection(4104, 40, 2, 4048, Ok(vec![1, 2]))]
    #[case::object_one_byte_past_the_collection_end(
        4104,
        40,
        2,
        4049,
        Err(FormatError::GlobalHeapObjectOutsideCollection {
            address: 40,
            index: 2,
            collection_end: 4104,
        })
    )]
    fn a_walk_checks_each_extent_against_the_collection_end(
        #[case] collection_size: usize,
        #[case] at: usize,
        #[case] index: u16,
        #[case] size: u64,
        #[case] expected: Result<Vec<u16>, FormatError>,
    ) {
        let mut bytes = padded_collection(
            LengthWidth::Eight,
            collection_size,
            &[(1, b"first"), (2, b"second")],
        );
        bytes[at..at + 2].copy_from_slice(&index.to_le_bytes());
        bytes[at + 8..at + 16].copy_from_slice(&size.to_le_bytes());
        assert_eq!(
            walk_objects(region(0, &bytes), LengthWidth::Eight)
                .map(|objects| objects.iter().map(|&(index, _)| index).collect()),
            expected
        );
    }

    #[rstest]
    #[case::implicit_free_space(4096, 4056, 0, Ok(vec![1]))]
    #[case::implicit_free_space_in_a_larger_collection(4104, 4064, 0, Ok(vec![1]))]
    #[case::no_free_space(4096, 4064, 0, Ok(vec![1]))]
    #[case::nonzero_index_in_the_tail(
        4096,
        4056,
        7,
        Err(FormatError::GlobalHeapObjectOutsideCollection {
            address: 4088,
            index: 7,
            collection_end: 4096,
        })
    )]
    fn a_tail_shorter_than_an_object_header_is_free_space(
        #[case] collection_size: usize,
        #[case] object_len: usize,
        #[case] tail_index: u16,
        #[case] expected: Result<Vec<u16>, FormatError>,
    ) {
        let object = vec![9; object_len];
        let mut bytes = padded_collection(LengthWidth::Eight, collection_size, &[(1, &object)]);
        if tail_index != 0 {
            bytes[collection_size - 8..collection_size - 6]
                .copy_from_slice(&tail_index.to_le_bytes());
        }
        assert_eq!(
            walk_objects(region(0, &bytes), LengthWidth::Eight)
                .map(|objects| objects.iter().map(|&(index, _)| index).collect()),
            expected
        );
    }

    #[rstest]
    #[case::two_byte_lengths(LengthWidth::Two)]
    #[case::four_byte_lengths(LengthWidth::Four)]
    #[case::eight_byte_lengths(LengthWidth::Eight)]
    fn a_locator_rejects_an_otherwise_valid_collection_of_unaligned_size(
        #[case] lengths: LengthWidth,
    ) {
        let bytes = padded_collection(lengths, 4100, &[(1, b"first")]);
        assert_eq!(
            GlobalHeapCollectionLocator::new(PhysicalAddress::new(0), lengths)
                .parse(region(0, &bytes))
                .unwrap_err(),
            FormatError::UnalignedGlobalHeapCollection {
                collection_address: 0,
                size: 4100,
            }
        );
    }

    #[rstest]
    #[case::two_byte_lengths(LengthWidth::Two)]
    #[case::four_byte_lengths(LengthWidth::Four)]
    #[case::eight_byte_lengths(LengthWidth::Eight)]
    fn a_walk_ignores_the_reserved_bytes_of_every_header(#[case] lengths: LengthWidth) {
        let objects: [&[u8]; 2] = [b"alpha", b"beta"];
        let mut bytes = encode_global_heap_collection(lengths, &objects).unwrap();
        bytes[5..8].fill(0xff);
        for object_header in [16, 40, 64] {
            bytes[object_header + 4..object_header + 8].fill(0xff);
        }
        assert_eq!(
            walk_objects(region(0, &bytes), lengths).unwrap(),
            vec![(1, objects[0]), (2, objects[1])]
        );
    }

    #[test]
    fn each_step_rejects_a_region_short_of_its_range_and_accepts_the_range_after() {
        let bytes = padded_collection(LengthWidth::Eight, 4096, &[(1, b"first")]);
        let image = region(256, &bytes);
        let short = |range: FileRange| {
            image
                .subregion(FileRange::new(range.start(), range.length() - 1).unwrap())
                .unwrap()
        };
        let insufficient = |range: FileRange| FormatError::InsufficientRegion {
            required_start: range.start().get(),
            required_length: range.length(),
            provided_start: range.start().get(),
            provided_length: range.length() - 1,
        };

        let locator =
            GlobalHeapCollectionLocator::new(PhysicalAddress::new(256), LengthWidth::Eight);
        let collection_header = locator.parse_range().unwrap();
        assert_eq!(
            locator.parse(short(collection_header)).unwrap_err(),
            insufficient(collection_header)
        );
        let mut walk = locator
            .parse(image.subregion(collection_header).unwrap())
            .unwrap()
            .walk();

        let object_header = walk.parse_range().unwrap();
        assert_eq!(
            walk.parse(short(object_header)).unwrap_err(),
            insufficient(object_header)
        );
        assert_eq!(walk.parse_range(), Some(object_header));
        let object = walk
            .parse(image.subregion(object_header).unwrap())
            .unwrap()
            .unwrap();

        let free_space_header = walk.parse_range().unwrap();
        assert_eq!(
            walk.parse(short(free_space_header)).unwrap_err(),
            insufficient(free_space_header)
        );
        assert_eq!(
            walk.parse(image.subregion(free_space_header).unwrap()),
            Ok(None)
        );
        assert_eq!(walk.parse_range(), None);

        let payload = object.payload().range();
        assert_eq!(
            object.payload().parse(short(payload)).unwrap_err(),
            insufficient(payload)
        );
        assert_eq!(object.payload().parse(image).unwrap(), b"first");
    }

    static LARGE_OBJECT: [u8; 70_000] = [7; 70_000];
}
