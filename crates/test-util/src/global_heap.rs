//! Global heap collections, which hold variable-length data: section
//! `subsec_fmt4_infra_globalheap`, version 4.0.

use crate::bytes;
use crate::widths::Widths;

/// The bytes of one collection holding `objects`, in the order given.
///
/// The collection declares its own total size, so a caller that means to give
/// it free space past its objects grows that declaration and leaves the bytes
/// alone.
pub fn collection(objects: &[Object<'_>], widths: Widths) -> Vec<u8> {
    let declared = PREFIX
        + widths.length
        + objects
            .iter()
            .map(|object| object.len(widths))
            .sum::<usize>()
        + FREE_SPACE_MARKER;

    let mut heap = SIGNATURE.to_vec();
    heap.push(VERSION);
    heap.extend_from_slice(&[0; 3]);
    bytes::push_uint(&mut heap, declared as u64, widths.length);
    for object in objects {
        heap.extend_from_slice(&object.index.to_le_bytes());
        heap.extend_from_slice(&object.reference_count.to_le_bytes());
        heap.extend_from_slice(&[0; 4]);
        bytes::push_uint(&mut heap, object.data.len() as u64, widths.length);
        heap.extend_from_slice(object.data);
        heap.resize(heap.len().next_multiple_of(ALIGNMENT), 0);
    }
    // Object zero's own size field holds the free space left, and is zero once
    // the collection is full.
    heap.extend_from_slice(&0u16.to_le_bytes());
    heap
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

    /// The bytes this object occupies, its prefix and its padding included.
    pub fn len(&self, widths: Widths) -> usize {
        PREFIX + widths.length + self.data.len().next_multiple_of(ALIGNMENT)
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

pub const SIGNATURE: &[u8; 4] = b"GCOL";

/// signature(4) + version(1) + reserved(3) for the collection, and
/// index(2) + reference count(2) + reserved(4) for an object.
const PREFIX: usize = 8;

/// Every object begins on an eight-byte boundary within its collection.
const ALIGNMENT: usize = 8;

/// The two bytes a full collection ends with, which are object zero's size.
const FREE_SPACE_MARKER: usize = 2;

const VERSION: u8 = 1;
