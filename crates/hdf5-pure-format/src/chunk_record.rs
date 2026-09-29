use alloc::vec;
use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::convert::Narrow;
use crate::error::FormatError;

/// A chunk that has been written to the file buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkRecord {
    /// The address of the chunk in the file.
    pub address: StoredAddress,
    /// The size of the chunk in the file in bytes, after its filters.
    ///
    /// The element of an unfiltered index omits the size, and a reader of one sets the size of a
    /// whole chunk.
    pub stored_size: u64,
    /// The filters skipped for the chunk: bit `i` is set where filter `i` of the pipeline was not
    /// applied.
    pub filter_mask: u32,
}

/// The slots of a chunk index, and the chunk at each occupied one.
///
/// A Fixed Array or an Extensible Array stores its elements by position, and the caller numbers
/// the positions over the chunk grid of the dataset's maximum shape. A dataset that can grow
/// leaves slots between its chunks unoccupied, and a Fixed Array spans slots it has no chunk for.
///
/// The slots borrow the chunks. A table of scattered slots is built only where the chunks do
/// not fill slots `0..n` in order.
pub struct IndexSlots<'a> {
    chunks: &'a [ChunkRecord],
    /// The `(slot, index into chunks)` pairs, ascending by slot, or empty where chunk `i`
    /// occupies slot `i`.
    scattered: Vec<(usize, usize)>,
    len: usize,
}

impl<'a> IndexSlots<'a> {
    /// Creates the slots of an index that spans `len` slots, with `chunks[i]` at slot
    /// `slot_of[i]`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if `len` or a slot does not fit a
    /// `usize`.
    pub fn new(chunks: &'a [ChunkRecord], slot_of: &[u64], len: u64) -> Result<Self, FormatError> {
        debug_assert_eq!(
            chunks.len(),
            slot_of.len(),
            "every chunk has exactly one index slot"
        );
        let len = len.to_usize()?;
        // The dense case is the overwhelming majority of writes. Recognizing it
        // here keeps them from allocating a table describing an order they are
        // already in.
        if len == chunks.len() && slot_of.iter().enumerate().all(|(i, &s)| s == i as u64) {
            return Ok(Self {
                chunks,
                scattered: Vec::new(),
                len,
            });
        }
        let mut scattered = Vec::with_capacity(chunks.len());
        for (i, &slot) in slot_of.iter().enumerate() {
            scattered.push((slot.to_usize()?, i));
        }
        scattered.sort_unstable();
        Ok(Self {
            chunks,
            scattered,
            len,
        })
    }

    /// Creates the slots of an index whose chunks occupy slots `0..chunks.len()` in order.
    pub fn dense(chunks: &'a [ChunkRecord]) -> Self {
        Self {
            len: chunks.len(),
            chunks,
            scattered: Vec::new(),
        }
    }

    /// Returns the number of slots the index spans, occupied or not.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Returns whether any of the `count` slots from `start` holds a chunk.
    ///
    /// An Extensible Array writer allocates only the blocks that hold a chunk. The cost is a
    /// binary search over the scattered slots, whatever `count` is.
    pub(crate) fn any_occupied(&self, start: u64, count: u64) -> bool {
        // A slot past `usize::MAX` is unoccupied. An Extensible Array spans about 2^33 slots, more
        // than a 32-bit `usize` holds.
        let Ok(start) = usize::try_from(start) else {
            return false;
        };
        let end = usize::try_from(count)
            .ok()
            .and_then(|c| start.checked_add(c))
            .unwrap_or(usize::MAX);
        if self.scattered.is_empty() {
            return start < self.chunks.len().min(end);
        }
        let from = self.scattered.partition_point(|&(s, _)| s < start);
        self.scattered.get(from).is_some_and(|&(s, _)| s < end)
    }

    /// Returns the chunk at `slot`, or `None` where no chunk occupies it.
    pub(crate) fn at(&self, slot: usize) -> Option<&ChunkRecord> {
        if self.scattered.is_empty() {
            return self.chunks.get(slot);
        }
        self.scattered
            .binary_search_by_key(&slot, |&(s, _)| s)
            .ok()
            .map(|i| &self.chunks[self.scattered[i].1])
    }
}

/// The encoding of one element of a Fixed Array or an Extensible Array chunk index.
///
/// The two arrays encode an element the same way, and the specification defines the same client
/// IDs for both.
#[derive(Debug, Clone, Copy)]
pub struct ChunkElementEncoding {
    /// The width in bytes of the chunk size field of a filtered element, or 0 for an unfiltered
    /// index.
    pub(crate) chunk_size_bytes: usize,
    /// The width in bytes of one element: the address, and for a filtered index the chunk size
    /// and the 4-byte filter mask.
    pub elem_size: usize,
    /// The client ID the header and every block store: 1 for a filtered index and 0 for an
    /// unfiltered one.
    pub(crate) client_id: u8,
}

/// Derive the element encoding for a chunk set.
///
/// `chunk_bytes` is the product of the chunk dimensions and the element size. The chunk size
/// field of a filtered element is one byte wider than the fewest bytes that hold `chunk_bytes`,
/// and at most 8 bytes wide, the width the specification gives for a dataset with a version 4
/// layout message. The C library computes the width in `H5D_FARRAY_FILT_COMPUTE_CHUNK_SIZE_LEN`
/// (`H5Dfarray.c`, HDF5 2.2.0).
///
/// It is taken from the geometry, not from the chunks that happen to have
/// been written, because the two part company exactly where it matters. Every
/// chunk `split_into_chunks` produces is padded to the full chunk size,
/// so for one chunk or a thousand the largest `raw_size` *is* `chunk_bytes` and
/// the two agree byte for byte. For **zero** chunks there is no `raw_size` to
/// take a maximum of, and the fallback this used to apply, treating the largest
/// chunk as 1 byte, declared a 2-byte compressed-size field for a dataset whose
/// chunks need up to 8. Nothing catches that later: the width is a header field
/// every reader honours, so an empty filtered dataset handed to the reference C
/// library came back with chunks encoded to a width our reader then decoded as
/// truncated deflate streams, and our own append rejected a chunk that no longer
/// fit the width its own index had declared.
pub fn chunk_element_encoding(
    chunk_bytes: u64,
    offset_size: u8,
    has_filters: bool,
) -> ChunkElementEncoding {
    let os = offset_size as usize;
    let chunk_size_bytes: usize = if has_filters {
        let log2_val = if chunk_bytes <= 1 {
            0
        } else {
            63 - chunk_bytes.leading_zeros()
        };
        let len = 1 + ((log2_val + 8) / 8) as usize;
        len.min(8)
    } else {
        0
    };
    ChunkElementEncoding {
        chunk_size_bytes,
        elem_size: if has_filters {
            os + chunk_size_bytes + 4
        } else {
            os
        },
        client_id: u8::from(has_filters),
    }
}

/// Appends the element of `chunk` to `buf`: the address, and for a filtered index the stored size
/// in `chunk_size_bytes` bytes and the filter mask.
pub(crate) fn write_chunk_element(
    buf: &mut Vec<u8>,
    chunk: &ChunkRecord,
    offset_size: u8,
    has_filters: bool,
    chunk_size_bytes: usize,
) {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "chunk address written into the on-disk offset width selected for this file"
    )]
    match offset_size {
        4 => buf.extend_from_slice(&(chunk.address.get() as u32).to_le_bytes()),
        8 => buf.extend_from_slice(&chunk.address.get().to_le_bytes()),
        _ => buf.extend_from_slice(&chunk.address.get().to_le_bytes()),
    }
    if has_filters {
        let cs_bytes = chunk.stored_size.to_le_bytes();
        buf.extend_from_slice(&cs_bytes[..chunk_size_bytes]);
        buf.extend_from_slice(&chunk.filter_mask.to_le_bytes());
    }
}

pub(crate) fn write_undefined_element(
    buf: &mut Vec<u8>,
    offset_size: u8,
    has_filters: bool,
    chunk_size_bytes: usize,
) {
    let os = offset_size as usize;
    buf.extend_from_slice(&vec![0xFF; os]);
    if has_filters {
        buf.extend_from_slice(&vec![0x00; chunk_size_bytes]);
        buf.extend_from_slice(&0u32.to_le_bytes());
    }
}

/// Chunk sizes before filtering that select four different chunk size field widths, for the
/// length tests of both arrays.
///
/// `extensible_array_len_matches_what_it_builds` asserts that the four widths differ.
#[cfg(test)]
pub(crate) const CHUNK_BYTES: [u64; 4] = [8, 300, 100_000, 1 << 32];
