//! The parsed chunk index a [`ChunkCache`](crate::chunk_cache::ChunkCache) retains between reads
//! of a dataset.

#[cfg(not(feature = "std"))]
use alloc::format;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::chunked_read::ChunkInfo;
use crate::error::FormatError;

/// The chunks a dataset's chunk index lists, in ascending order of address.
///
/// A `ParsedChunkIndex` holds each chunk once, as the [`ChunkInfo`] the walk of the chunk index
/// returned, and a chunk's coordinate borrows the first `rank` of its offsets. A key of a version
/// 1 B-tree stores one offset more than the dataset has dimensions, the offset within the
/// datatype, as defined in "Version 1 B-trees" of the [format specification, version 4.0][spec].
///
/// The cached readers visit the chunks in this order, in which a
/// [`ChunkSpanReader`](crate::chunk_span::ChunkSpanReader) reads each span once.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v1
#[derive(Debug)]
pub(crate) struct ParsedChunkIndex {
    chunks: Vec<ChunkInfo>,
    rank: usize,
}

impl ParsedChunkIndex {
    /// Sorts `chunks` by address into an index whose coordinates have `rank` dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if a chunk has fewer than `rank` offsets.
    pub(crate) fn new(mut chunks: Vec<ChunkInfo>, rank: usize) -> Result<Self, FormatError> {
        if let Some(chunk) = chunks.iter().find(|chunk| chunk.offsets.len() < rank) {
            return Err(FormatError::Internal(format!(
                "a chunk index of rank {rank} lists a chunk at offsets {:?}",
                chunk.offsets
            )));
        }
        chunks.sort_unstable_by_key(|chunk| chunk.address);
        Ok(Self { chunks, rank })
    }

    /// Returns an iterator over the chunks in ascending order of address, each with its
    /// coordinate.
    pub(crate) fn chunks(&self) -> impl Iterator<Item = IndexedChunk<'_>> {
        self.chunks.iter().map(|info| IndexedChunk {
            info,
            // `new` rejects a chunk with fewer than `rank` offsets.
            coord: &info.offsets[..self.rank],
        })
    }
}

/// A chunk of a [`ParsedChunkIndex`] and its coordinate, both borrowed from it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IndexedChunk<'a> {
    /// The chunk as the walk of the chunk index returned it, with every offset the chunk index in
    /// the file stores.
    pub(crate) info: &'a ChunkInfo,
    /// The chunk's offset in each dimension of the dataset.
    pub(crate) coord: &'a [u64],
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::address::StoredAddress;
    use crate::address::StoredAddressExt;
    use crate::chunked_read::StoredChunkSize;

    fn chunk(offsets: &[u64], address: u64) -> ChunkInfo {
        ChunkInfo {
            chunk_size: StoredChunkSize::v4(16),
            filter_mask: 0,
            offsets: offsets.to_vec(),
            address: StoredAddress::new(address),
        }
    }

    #[rstest]
    #[case::in_address_order(&[0x1000, 0x2000, 0x3000])]
    #[case::reversed(&[0x3000, 0x2000, 0x1000])]
    #[case::interleaved(&[0x2000, 0x3000, 0x1000])]
    fn chunks_are_listed_in_ascending_address_order(#[case] addresses: &[u64]) {
        let chunks = addresses
            .iter()
            .zip(0u64..)
            .map(|(&address, row)| chunk(&[row * 4, 0], address))
            .collect();
        let index = ParsedChunkIndex::new(chunks, 2).unwrap();

        let listed: Vec<_> = index
            .chunks()
            .map(|chunk| chunk.info.address.get())
            .collect();

        assert_eq!(listed, [0x1000, 0x2000, 0x3000]);
    }

    #[rstest]
    #[case::version_1_btree_key(&[4, 8, 0], &[4, 8])]
    #[case::version_4_index(&[4, 8], &[4, 8])]
    fn a_chunk_coordinate_holds_the_offsets_of_the_dataset_rank(
        #[case] offsets: &[u64],
        #[case] coord: &[u64],
    ) {
        let index = ParsedChunkIndex::new(vec![chunk(offsets, 0x1000)], 2).unwrap();

        let chunks: Vec<_> = index.chunks().collect();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].coord, coord);
        assert_eq!(chunks[0].info.offsets, offsets);
    }

    #[test]
    fn a_chunk_with_fewer_offsets_than_the_rank_is_rejected() {
        let err = ParsedChunkIndex::new(vec![chunk(&[4, 8], 0x1000), chunk(&[4], 0x2000)], 2)
            .unwrap_err();

        assert_eq!(
            err,
            FormatError::Internal("a chunk index of rank 2 lists a chunk at offsets [4]".into())
        );
    }
}
