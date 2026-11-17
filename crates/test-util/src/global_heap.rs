//! Global heap collections, which hold variable-length data.
//!
//! A collection is defined in "Global Heap" of the [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap

use std::ops::Range;

use crate::bytes;
use crate::widths::Widths;

/// Returns a collection that holds `objects` in the order given, and a free space object after
/// them.
///
/// The collection is 4096 bytes, the minimum collection size, or the size its objects and the free
/// space object header need where that is larger.
pub fn collection(objects: &[Object<'_>], widths: Widths) -> Vec<u8> {
    let used = HEADER_SIZE + objects.iter().map(Object::len).sum::<usize>() + HEADER_SIZE;
    collection_of_size(objects, widths, used.max(MIN_COLLECTION_SIZE))
}

/// Returns a collection of `size` bytes that holds `objects` in the order given.
///
/// A free space object spans the rest of the collection where the rest has room for an object
/// header, and a shorter rest is zero bytes.
///
/// # Panics
///
/// Panics if the objects do not fit in `size` bytes, or if `size` or the size of an object does
/// not fit a field `widths.length` bytes wide.
pub fn collection_of_size(objects: &[Object<'_>], widths: Widths, size: usize) -> Vec<u8> {
    collection_of_size_with_padding(objects, widths, size, 0)
}

/// Returns a collection with `padding` between each object's payload and its next boundary.
///
/// Writes `objects` in slice order. The collection has `size` bytes, with a free-space
/// object in the remaining space when its header fits. Other unused bytes are zero.
///
/// # Panics
///
/// Panics if the objects do not fit in `size` bytes, or a size does not fit a
/// `widths.length`-byte field or `widths.length` is outside 1 through 8.
pub fn collection_of_size_with_padding(
    objects: &[Object<'_>],
    widths: Widths,
    size: usize,
    padding: u8,
) -> Vec<u8> {
    let mut heap = SIGNATURE.to_vec();
    heap.push(VERSION);
    heap.extend_from_slice(&[0; COLLECTION_RESERVED]);
    bytes::push_uint(&mut heap, size as u64, widths.length);
    heap.resize(HEADER_SIZE, 0);
    for object in objects {
        push_object_header(
            &mut heap,
            object.index,
            object.reference_count,
            object.data.len() as u64,
            widths,
        );
        heap.extend_from_slice(object.data);
        heap.resize(heap.len().next_multiple_of(ALIGNMENT), padding);
    }
    assert!(
        heap.len() <= size,
        "{} bytes of objects do not fit a {size}-byte collection",
        heap.len()
    );
    let free_space = size - heap.len();
    if free_space >= HEADER_SIZE {
        push_object_header(&mut heap, FREE_SPACE_INDEX, 0, free_space as u64, widths);
    }
    heap.resize(size, 0);
    heap
}

/// Returns a free-space object header with index zero and total size `size`.
///
/// The returned bytes contain the header and its padding, without the remaining free bytes.
///
/// # Panics
///
/// Panics if `size` does not fit a `widths.length`-byte field or `widths.length` is outside
/// 1 through 8.
pub fn free_space_header(size: u64, widths: Widths) -> Vec<u8> {
    let mut header = Vec::new();
    push_object_header(&mut header, FREE_SPACE_INDEX, 0, size, widths);
    header
}

/// Appends an object header to `heap`, padded to 16 bytes, with a size field `widths.length` bytes
/// wide.
fn push_object_header(
    heap: &mut Vec<u8>,
    index: u16,
    reference_count: u16,
    size: u64,
    widths: Widths,
) {
    let start = heap.len();
    heap.extend_from_slice(&index.to_le_bytes());
    heap.extend_from_slice(&reference_count.to_le_bytes());
    heap.extend_from_slice(&[0; OBJECT_RESERVED]);
    bytes::push_uint(heap, size, widths.length);
    heap.resize(start + HEADER_SIZE, 0);
}

/// One object in a collection, addressed by the collection's address and this
/// index together.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Object<'d> {
    pub index: u16,
    pub reference_count: u16,
    pub data: &'d [u8],
}

impl<'d> Object<'d> {
    /// An object nothing yet refers to but the element being written.
    pub fn new(index: u16, data: &'d [u8]) -> Self {
        Self {
            index,
            reference_count: 1,
            data,
        }
    }

    /// Returns the number of bytes the object spans, its header and the padding of its data
    /// included.
    fn len(&self) -> usize {
        HEADER_SIZE + self.data.len().next_multiple_of(ALIGNMENT)
    }
}

/// A variable-length element's raw value: how many elements it holds, and
/// which global heap object holds them.
pub fn reference(len: u32, collection_address: u64, index: u32, widths: Widths) -> Vec<u8> {
    let mut reference = len.to_le_bytes().to_vec();
    bytes::push_uint(&mut reference, collection_address, widths.offset);
    reference.extend_from_slice(&index.to_le_bytes());
    reference
}

/// Returns the range of each collection that [`collection_starts`] finds in `bytes`, with the
/// length read from its `widths.length`-byte collection size field.
pub fn collection_ranges(bytes: &[u8], widths: Widths) -> Vec<Range<u64>> {
    collection_starts(bytes)
        .into_iter()
        .map(|start| {
            let size = bytes::uint_at(bytes, start as usize + SIZE_OFFSET, widths.length);
            start..start + size
        })
        .collect()
}

/// Returns the offset of each [`SIGNATURE`] in `bytes`, where a collection starts.
///
/// An object whose data holds the signature bytes matches as well.
pub fn collection_starts(bytes: &[u8]) -> Vec<u64> {
    bytes
        .windows(SIGNATURE.len())
        .enumerate()
        .filter(|(_, window)| window == SIGNATURE)
        .map(|(start, _)| start as u64)
        .collect()
}

pub const SIGNATURE: &[u8; 4] = b"GCOL";

/// The offset of the version in a collection header.
pub const VERSION_OFFSET: usize = SIGNATURE.len();

/// The offset of the collection size in a collection header.
pub const SIZE_OFFSET: usize = VERSION_OFFSET + 1 + COLLECTION_RESERVED;

/// The size in bytes of each collection header and object header in these fixtures.
// libhdf5 pads the collection header and each object header to 16 bytes at every length
// width, in `H5HG_SIZEOF_HDR` and `H5HG_SIZEOF_OBJHDR` (`H5HGpkg.h`, HDF5 1.8.23 to 2.2.0).
pub const HEADER_SIZE: usize = 16;

/// Every object begins on an eight-byte boundary within its collection.
pub const ALIGNMENT: usize = 8;

/// The number of reserved bytes between the version and the collection size of a collection
/// header.
const COLLECTION_RESERVED: usize = 3;

/// The number of reserved bytes between the reference count and the object size of an object
/// header.
const OBJECT_RESERVED: usize = 4;

/// The index of the free space object.
const FREE_SPACE_INDEX: u16 = 0;

/// The minimum collection size in bytes, defined in "Global Heap" of the [format specification,
/// version 4.0][spec].
///
/// libhdf5 calls it `H5HG_MINSIZE` (`H5HGpkg.h`, HDF5 2.2.0).
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
pub const MIN_COLLECTION_SIZE: usize = 4096;

const VERSION: u8 = 1;
