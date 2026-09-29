//! The chunk records of the Fixed Array and Extensible Array chunk indexes, and the element
//! encoding the two share.
//!
//! Both indexes store one element per slot: the address of the chunk, and for a filtered dataset
//! its stored size and filter mask. A slot no chunk occupies stores the undefined address. The
//! elements are defined in "[The Fixed Array Index][fixed]" and "[The Extensible Array
//! Index][extensible]" of the format specification, version 4.0.
//!
//! [fixed]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_fixedarr
//! [extensible]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::bytes;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::OffsetWidth;

/// The address, stored size, and filter mask of one chunk, as an element of a chunk index
/// records them.
///
/// A reader of a Fixed Array or an Extensible Array reports one record per occupied slot, and a
/// writer encodes one per chunk it places.
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
    /// Returns [`FormatError::Internal`] if `slot_of` and `chunks` differ in length, a slot is
    /// past `len`, or two chunks share a slot, and [`FormatError::ValueTooLargeForPlatform`] if
    /// `len` or a slot does not fit a `usize`.
    pub fn new(chunks: &'a [ChunkRecord], slot_of: &[u64], len: u64) -> Result<Self, FormatError> {
        if chunks.len() != slot_of.len() {
            return Err(FormatError::Internal(format!(
                "{} chunks were given {} index slots",
                chunks.len(),
                slot_of.len()
            )));
        }
        let len = len.to_usize()?;
        // Chunks already in slot order need no table.
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
        if let Some(&(slot, _)) = scattered.last().filter(|&&(slot, _)| slot >= len) {
            return Err(FormatError::Internal(format!(
                "a chunk occupies slot {slot} of an index spanning {len} slots"
            )));
        }
        if let Some(pair) = scattered.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            return Err(FormatError::Internal(format!(
                "two chunks occupy slot {} of the index",
                pair[0].0
            )));
        }
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

impl ChunkElementEncoding {
    /// Returns whether the chunk size field of an element is wide enough for `stored_size`.
    ///
    /// An unfiltered index has no size field, and only a stored size of 0 fits it.
    pub fn holds_stored_size(self, stored_size: u64) -> bool {
        low_bytes(&stored_size.to_le_bytes(), self.chunk_size_bytes).is_some()
    }
}

/// Returns the element encoding of an index over chunks of `chunk_bytes` bytes each before
/// filtering.
///
/// `chunk_bytes` is the product of the chunk dimensions and the element size. The chunk size
/// field of a filtered element is one byte wider than the fewest bytes that hold `chunk_bytes`,
/// and at most 8 bytes wide, the width the specification gives for a dataset with a version 4
/// layout message. The C library computes the width in `H5D_FARRAY_FILT_COMPUTE_CHUNK_SIZE_LEN`
/// (`H5Dfarray.c`, HDF5 2.2.0).
///
/// The width depends on `chunk_bytes` alone and not on the stored sizes, so an index of no
/// chunks has the width its first chunk needs.
pub fn chunk_element_encoding(
    chunk_bytes: u64,
    offset_size: OffsetWidth,
    has_filters: bool,
) -> ChunkElementEncoding {
    let os = usize::from(offset_size.get());
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
///
/// # Errors
///
/// Returns [`FormatError::Internal`] if the index is filtered and `chunk_size_bytes` is more than 8
/// or the stored size does not fit `chunk_size_bytes` bytes.
pub(crate) fn write_chunk_element(
    buf: &mut Vec<u8>,
    chunk: &ChunkRecord,
    offset_size: OffsetWidth,
    has_filters: bool,
    chunk_size_bytes: usize,
) -> Result<(), FormatError> {
    bytes::write_offset(buf, chunk.address.get(), offset_size);
    if has_filters {
        let stored_size = chunk.stored_size.to_le_bytes();
        let low = low_bytes(&stored_size, chunk_size_bytes).ok_or_else(|| {
            FormatError::Internal(format!(
                "a chunk of {} bytes does not fit a {chunk_size_bytes}-byte chunk size field",
                chunk.stored_size
            ))
        })?;
        buf.extend_from_slice(low);
        buf.extend_from_slice(&chunk.filter_mask.to_le_bytes());
    }
    Ok(())
}

/// Returns the low `field_bytes` bytes of `bytes`, or `None` if a byte above them is not zero or
/// `field_bytes` is more than 8.
fn low_bytes(bytes: &[u8; 8], field_bytes: usize) -> Option<&[u8]> {
    let (low, high) = bytes.split_at_checked(field_bytes)?;
    high.iter().all(|&byte| byte == 0).then_some(low)
}

/// Appends the element of a slot no chunk occupies to `buf`: the undefined address, and for a
/// filtered index a zero size and a zero filter mask.
pub(crate) fn write_undefined_element(
    buf: &mut Vec<u8>,
    offset_size: OffsetWidth,
    has_filters: bool,
    chunk_size_bytes: usize,
) {
    bytes::write_offset(buf, u64::MAX, offset_size);
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

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn record(address: u64) -> ChunkRecord {
        ChunkRecord {
            address: StoredAddress::new(address),
            stored_size: 8,
            filter_mask: 0,
        }
    }

    #[test]
    fn scattered_slots_hold_their_chunks() {
        let chunks = [record(0x100), record(0x200)];
        let slots = IndexSlots::new(&chunks, &[3, 1], 4).unwrap();
        assert_eq!(slots.len(), 4);
        assert_eq!(
            (0..4).map(|slot| slots.at(slot)).collect::<Vec<_>>(),
            vec![None, Some(&chunks[1]), None, Some(&chunks[0])]
        );
    }

    #[rstest]
    #[case::a_slot_short(&[0], 2, "2 chunks were given 1 index slots")]
    #[case::a_slot_past_the_span(&[0, 2], 2, "a chunk occupies slot 2 of an index spanning 2 slots")]
    #[case::a_shared_slot(&[1, 1], 2, "two chunks occupy slot 1 of the index")]
    fn slots_that_misplace_a_chunk_are_an_internal_error(
        #[case] slot_of: &[u64],
        #[case] len: u64,
        #[case] expected: &str,
    ) {
        let chunks = [record(0x100), record(0x200)];
        let err = IndexSlots::new(&chunks, slot_of, len).err().unwrap();
        let FormatError::Internal(detail) = &err else {
            panic!("expected Internal, got {err:?}");
        };
        assert_eq!(detail, expected);
    }

    #[rstest]
    #[case::a_size_past_the_field(1 << 16, 2)]
    #[case::a_field_past_8_bytes(8, 9)]
    fn a_stored_size_that_does_not_fit_its_field_is_an_error(
        #[case] stored_size: u64,
        #[case] chunk_size_bytes: usize,
    ) {
        let chunk = ChunkRecord {
            stored_size,
            ..record(0x100)
        };
        let err = write_chunk_element(
            &mut Vec::new(),
            &chunk,
            OffsetWidth::Eight,
            true,
            chunk_size_bytes,
        )
        .unwrap_err();
        let FormatError::Internal(detail) = &err else {
            panic!("expected Internal, got {err:?}");
        };
        assert_eq!(
            detail,
            &format!(
                "a chunk of {stored_size} bytes does not fit a {chunk_size_bytes}-byte chunk size \
                 field"
            )
        );
    }
}
