use alloc::vec::Vec;
use core::num::NonZeroU16;

use super::collection;
use super::error::GlobalHeapCollectionError;
use super::write;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::LengthWidth;

/// A global heap collection that a writer adds objects to one at a time, up to a size limit.
///
/// The size limit applies from the second object onward and includes room for a free space
/// header. [`finish`](Self::finish) encodes the objects in insertion order, starting at index 1.
///
/// libhdf5 uses the object header and padded payload size to find a collection with enough free
/// space. It can extend an existing collection before allocating another (`H5HG_insert` in
/// `H5HG.c` and `H5F_cwfs_find_free_heap` in `H5Fcwfs.c`, HDF5 2.2.0).
///
/// # Examples
///
/// ```
/// # use hdf5_pure_format::__private::GlobalHeapCollectionError;
/// # use hdf5_pure_format::__private::GlobalHeapCollectionWriter;
/// # use hdf5_pure_format::__private::LengthWidth;
/// let mut collection = GlobalHeapCollectionWriter::new(LengthWidth::Eight, 64);
/// assert_eq!(collection.push(b"alpha")?.map(|index| index.get()), Some(1));
/// // A second object would take the collection past 64 bytes.
/// assert_eq!(collection.push(b"beta")?, None);
/// assert_eq!(collection.finish()?.len(), 4096);
/// # Ok::<(), GlobalHeapCollectionError>(())
/// ```
#[derive(Clone, Debug)]
pub struct GlobalHeapCollectionWriter {
    length_width: LengthWidth,
    /// The size in bytes, the free space object included, past which [`push`](Self::push)
    /// rejects a second or later object.
    max_size: u64,
    /// The size in bytes of the collection header and the objects added so far.
    size: u64,
    /// The bytes of the objects added so far, one after another.
    payloads: Vec<u8>,
    /// The end of each object added so far in `payloads`.
    ends: Vec<usize>,
}

impl GlobalHeapCollectionWriter {
    /// Creates an empty collection with length fields `length_width` bytes wide.
    ///
    /// The first object may exceed `max_size`. See [`push`](Self::push).
    #[must_use]
    pub fn new(length_width: LengthWidth, max_size: u64) -> Self {
        Self {
            length_width,
            max_size,
            size: collection::header_len(length_width),
            payloads: Vec::new(),
            ends: Vec::new(),
        }
    }

    /// Adds `object` at the next index, from 1, and returns that index.
    ///
    /// An empty collection takes any one object, so an object larger than the size limit gets a
    /// collection of its own. Each object stores a reference count of 1.
    ///
    /// Returns `Ok(None)` if the collection already holds
    /// [`GLOBAL_HEAP_MAX_OBJECTS`](super::GLOBAL_HEAP_MAX_OBJECTS) objects or if adding another
    /// object to a nonempty collection would exceed its size limit.
    ///
    /// # Errors
    ///
    /// Returns [`GlobalHeapCollectionError::Format`] if the size of the collection exceeds
    /// [`u64::MAX`], or if growing the object buffers fails.
    pub fn push(&mut self, object: &[u8]) -> Result<Option<NonZeroU16>, GlobalHeapCollectionError> {
        let Some(index) = u16::try_from(self.ends.len() + 1)
            .ok()
            .and_then(NonZeroU16::new)
        else {
            return Ok(None);
        };
        let header_len = collection::header_len(self.length_width);
        let size = collection::add_collection_size(
            collection::add_collection_size(self.size, header_len)?,
            collection::aligned_collection_size(object.len().to_u64())?,
        )?;
        // The free space object, which `finish` appends, takes at least a header.
        if !self.ends.is_empty()
            && collection::add_collection_size(size, header_len)? > self.max_size
        {
            return Ok(None);
        }
        self.payloads.try_reserve(object.len()).map_err(|source| {
            FormatError::AllocationFailed {
                values: object.len(),
                source,
            }
        })?;
        self.ends
            .try_reserve(1)
            .map_err(|source| FormatError::AllocationFailed { values: 1, source })?;
        self.payloads.extend_from_slice(object);
        self.ends.push(self.payloads.len());
        self.size = size;
        Ok(Some(index))
    }

    /// Returns the encoded collection, with a free space object at index 0 that spans the rest of
    /// it.
    ///
    /// Pads the collection to at least [`GLOBAL_HEAP_MIN_COLLECTION_SIZE`](super::GLOBAL_HEAP_MIN_COLLECTION_SIZE)
    /// bytes.
    ///
    /// # Errors
    ///
    /// Returns [`GlobalHeapCollectionError::Format`] if reserving the list of objects fails, or
    /// any error from [`encode_global_heap_collection`](write::encode_global_heap_collection).
    pub fn finish(self) -> Result<Vec<u8>, GlobalHeapCollectionError> {
        let mut objects = Vec::new();
        objects
            .try_reserve_exact(self.ends.len())
            .map_err(|source| FormatError::AllocationFailed {
                values: self.ends.len(),
                source,
            })?;
        let mut start = 0;
        for &end in &self.ends {
            // An empty object starts and ends at the same offset.
            objects.push(&self.payloads[start..end]);
            start = end;
        }
        write::encode_global_heap_collection(self.length_width, &objects)
    }
}
