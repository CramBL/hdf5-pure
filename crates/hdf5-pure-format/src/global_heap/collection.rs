use core::num::NonZeroU16;

use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;

use super::compaction::GlobalHeapMemberCollector;
use super::object::{GlobalHeapObjectHeader, GlobalHeapSpan};
use super::read::{GlobalHeapLookup, GlobalHeapWalk};
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::LengthWidth;

/// The position of a global heap collection and the width of its length fields.
///
/// The caller reads the bytes of the range [`parse_range`](Self::parse_range) returns and passes
/// them to [`parse`](Self::parse), which parses the collection header. The
/// [`GlobalHeapCollection`] it returns starts a [`GlobalHeapWalk`] or a [`GlobalHeapLookup`],
/// which returns the range of each object header and any implied free tail in turn. The
/// collection is defined in "Global Heap" of the [format specification, version 4.0][spec].
///
/// # Examples
///
/// ```
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
        let mut cursor = bytes::ByteCursor::new(region.subregion(self.parse_range()?)?.bytes());
        if cursor.array::<{ GCOL_SIGNATURE.len() }>()? != GCOL_SIGNATURE {
            return Err(FormatError::InvalidGlobalHeapSignature);
        }
        let version = cursor.u8()?;
        if version != GCOL_VERSION {
            return Err(FormatError::InvalidGlobalHeapVersion(version));
        }
        cursor.array::<GCOL_RESERVED_LEN>()?;
        let size = cursor.length(lengths)?;
        if size < GLOBAL_HEAP_MIN_COLLECTION_SIZE as u64 {
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
            first: GlobalHeapObjectHeader::at(address.checked_add(header_len)?, lengths)?,
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
/// finds the object at one index. For each step, the caller supplies either a complete object
/// header or an implied free tail. The collection is defined in "Global Heap" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlobalHeapCollection {
    range: FileRange,
    lengths: LengthWidth,
    first: GlobalHeapObjectHeader,
}

impl GlobalHeapCollection {
    /// Starts collecting member metadata from object headers in physical order.
    ///
    /// See [`GlobalHeapMembers::compact`](super::GlobalHeapMembers::compact) for an example.
    pub fn collect_members(&self) -> GlobalHeapMemberCollector {
        GlobalHeapMemberCollector::new(*self)
    }

    /// Starts a lookup of the object at `index`, at the first object header.
    pub fn lookup(&self, index: NonZeroU16) -> GlobalHeapLookup {
        GlobalHeapLookup::new(*self, index)
    }

    /// Starts a walk over the objects of the collection, at the first object header.
    pub fn walk(&self) -> GlobalHeapWalk {
        GlobalHeapWalk::new(*self)
    }

    /// Returns the range of the collection, from its address over the size its header stores.
    pub const fn range(&self) -> FileRange {
        self.range
    }

    pub(super) const fn first_header(&self) -> GlobalHeapObjectHeader {
        self.first
    }

    pub(super) const fn members_start(&self) -> PhysicalAddress {
        self.first.range().start()
    }

    pub(super) const fn lengths(&self) -> LengthWidth {
        self.lengths
    }

    /// Classifies the span at `start` from the declared collection boundary.
    ///
    /// For version 1 collections, "Global Heap" of the [format specification, version 4.0][spec]
    /// defines a terminal span too short for an object header as free space with an implied header.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `start` is past the collection end.
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
    pub(super) fn next_span(&self, start: PhysicalAddress) -> Result<GlobalHeapSpan, FormatError> {
        let remaining = self.remaining(start)?;
        Ok(if remaining == 0 {
            GlobalHeapSpan::End
        } else if remaining < header_len(self.lengths) {
            GlobalHeapSpan::ImpliedFree(FileRange::new(start, remaining)?)
        } else {
            GlobalHeapSpan::Header(GlobalHeapObjectHeader::at(start, self.lengths)?)
        })
    }

    /// Returns the number of bytes from `start` to the end of the collection.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `start` is past the collection end.
    pub(super) fn remaining(&self, start: PhysicalAddress) -> Result<u64, FormatError> {
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

// libhdf5 pads the collection header and each object header to a multiple of 8 bytes at
// every length width, in `H5HG_SIZEOF_HDR` and `H5HG_SIZEOF_OBJHDR` (`H5HGpkg.h`, HDF5 1.8.23 to
// 2.2.0).
pub(super) fn header_len(lengths: LengthWidth) -> u64 {
    (FIXED_HEADER_LEN + u64::from(lengths.get())).next_multiple_of(ALIGNMENT as u64)
}

/// Rounds a collection size up to the global heap's alignment.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if the rounded size exceeds [`u64::MAX`].
pub(super) fn aligned_collection_size(size: u64) -> Result<u64, FormatError> {
    let alignment = ALIGNMENT.to_u64();
    let padding = (alignment - size % alignment) % alignment;
    add_collection_size(size, padding)
}

/// Adds a header, payload, or padding length to a collection size.
///
/// # Errors
///
/// Returns [`FormatError::OffsetOverflow`] if the sum exceeds [`u64::MAX`].
pub(super) fn add_collection_size(size: u64, additional: u64) -> Result<u64, FormatError> {
    size.checked_add(additional)
        .ok_or(FormatError::OffsetOverflow {
            offset: size,
            length: additional,
        })
}

/// The signature a global heap collection begins with.
pub(super) const GCOL_SIGNATURE: [u8; 4] = *b"GCOL";

/// The version of a global heap collection, the one version the specification defines.
pub(super) const GCOL_VERSION: u8 = 1;

/// The number of reserved bytes between the version and the collection size of a collection
/// header.
pub(super) const GCOL_RESERVED_LEN: usize = 3;

/// The number of bytes before the length field of a collection header and of an object header.
pub(super) const FIXED_HEADER_LEN: u64 = 8;

// The collection header, each object header and each object's data are padded to a multiple of
// this many bytes, `H5HG_ALIGNMENT` in the C library (`H5HGpkg.h`, HDF5 2.2.0).
pub(super) const ALIGNMENT: usize = 8;

/// The minimum size in bytes of a global heap collection, 4096.
///
/// "Global Heap" of the [format specification, version 4.0][spec] defines the minimum, and
/// libhdf5 calls it `H5HG_MINSIZE` (`H5HGpkg.h`, HDF5 2.2.0). libhdf5 allocates a new collection of
/// this size unless the first object needs more (`H5HG__create` in `H5HG.c`, HDF5 2.2.0).
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
pub const GLOBAL_HEAP_MIN_COLLECTION_SIZE: usize = 4096;
