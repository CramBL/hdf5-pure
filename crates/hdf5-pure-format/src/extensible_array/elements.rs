use super::block::Owner;
use crate::chunk_record::{ChunkRecord, ElementEncoding};
use crate::convert::Narrow;
use crate::error::FormatError;

/// The elements of an Extensible Array index block, unpaged data block, or page.
///
/// Only slots below the header's "Max Index Set" are included.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtensibleArrayElements<'a> {
    bytes: &'a [u8],
    first_slot: u64,
    encoding: ElementEncoding,
}

impl<'a> ExtensibleArrayElements<'a> {
    pub(super) fn below_max_index_set(owner: Owner, bytes: &'a [u8], first_slot: u64) -> Self {
        let encoding = owner.layout().encoding();
        let len = owner
            .max_index_set()
            .saturating_sub(first_slot)
            .to_usize()
            .ok()
            .and_then(|count| count.checked_mul(usize::from(encoding.stride().get())))
            .map_or(bytes.len(), |len| len.min(bytes.len()));
        // `len` is at most `bytes.len()`, so the split is inside `bytes`.
        let (bytes, _at_or_past_max_index_set) = bytes.split_at(len);
        Self {
            bytes,
            first_slot,
            encoding,
        }
    }

    /// Returns the slot and the record of each element that stores a chunk address.
    ///
    /// The slot is the element's index in the whole array. A record of an unfiltered chunk has
    /// `chunk_byte_size` as its stored size and a filter mask of 0. An element that stores the
    /// undefined address has no record.
    pub fn records(
        &self,
        chunk_byte_size: u64,
    ) -> impl Iterator<Item = Result<(u64, ChunkRecord), FormatError>> + 'a {
        let Self {
            bytes,
            first_slot,
            encoding,
        } = *self;
        encoding.records(bytes, first_slot, chunk_byte_size)
    }
}
