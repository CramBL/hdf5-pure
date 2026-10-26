//! The parsed chunk index a [`ChunkCache`](crate::chunk_cache::ChunkCache) retains between reads
//! of a dataset.

#[cfg(not(feature = "std"))]
use alloc::format;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::chunked_read::ChunkInfo;
use crate::chunked_read::DenseCoverage;
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
    coverage: Option<DenseCoverage>,
}

impl ParsedChunkIndex {
    /// Sorts `chunks` by address and retains dense coverage where it can be proven.
    ///
    /// The coordinate rank is `ds_dims.len()`. The retained proof applies to the supplied chunk
    /// dimensions and dataset extent.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if a chunk has fewer than `ds_dims.len()` offsets.
    pub(crate) fn new(
        mut chunks: Vec<ChunkInfo>,
        chunk_dims: &[u64],
        ds_dims: &[u64],
    ) -> Result<Self, FormatError> {
        let rank = ds_dims.len();
        if let Some(chunk) = chunks.iter().find(|chunk| chunk.offsets.len() < rank) {
            return Err(FormatError::Internal(format!(
                "a chunk index of rank {rank} lists a chunk at offsets {:?}",
                chunk.offsets
            )));
        }
        chunks.sort_unstable_by_key(|chunk| chunk.address);
        let mut index = Self {
            chunks,
            rank,
            coverage: None,
        };
        index.coverage = DenseCoverage::new(&index, chunk_dims, ds_dims);
        Ok(index)
    }

    /// Returns the retained proof if both dimension slices match its geometry.
    ///
    /// Returns `None` if density was unproven or the dimensions differ. Borrows the proof
    /// without allocating or revisiting the chunk descriptors.
    pub(crate) fn dense_coverage(
        &self,
        chunk_dims: &[u64],
        ds_dims: &[u64],
    ) -> Option<&DenseCoverage> {
        self.coverage
            .as_ref()
            .filter(|coverage| coverage.matches(chunk_dims, ds_dims))
    }

    /// Borrows the complete descriptor slice in the same address order as [`Self::chunks`].
    pub(crate) fn descriptors(&self) -> &[ChunkInfo] {
        &self.chunks
    }

    /// Returns an iterator over the chunks in ascending order of address, each with its
    /// coordinate.
    pub(crate) fn chunks(&self) -> impl ExactSizeIterator<Item = IndexedChunk<'_>> {
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
        let index = ParsedChunkIndex::new(chunks, &[4, 1], &[12, 1]).unwrap();

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
        let index = ParsedChunkIndex::new(vec![chunk(offsets, 0x1000)], &[4, 8], &[8, 16]).unwrap();

        let chunks: Vec<_> = index.chunks().collect();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].coord, coord);
        assert_eq!(chunks[0].info.offsets, offsets);
    }

    #[test]
    fn a_chunk_with_fewer_offsets_than_the_rank_is_rejected() {
        let err = ParsedChunkIndex::new(
            vec![chunk(&[4, 8], 0x1000), chunk(&[4], 0x2000)],
            &[4, 8],
            &[8, 16],
        )
        .unwrap_err();

        assert_eq!(
            err,
            FormatError::Internal("a chunk index of rank 2 lists a chunk at offsets [4]".into())
        );
    }

    #[rstest]
    #[case::dense(&[2][..], &[6][..], &[&[4][..], &[0][..], &[2][..]], true)]
    #[case::edge(&[2][..], &[5][..], &[&[4][..], &[0][..], &[2][..]], true)]
    #[case::edges_in_two_dimensions(&[2, 3][..], &[3, 5][..], &[&[2, 3][..], &[0, 0][..], &[0, 3][..], &[2, 0][..]], true)]
    #[case::edges_in_three_dimensions(&[2, 3, 4][..], &[1, 2, 5][..], &[&[0, 0, 4][..], &[0, 0, 0][..]], true)]
    #[case::version_1_key(&[2][..], &[4][..], &[&[0, 0][..], &[2, 0][..]], true)]
    #[case::interior_hole(&[2][..], &[6][..], &[&[0][..], &[4][..]], false)]
    #[case::duplicate(&[2][..], &[6][..], &[&[0][..], &[0][..], &[4][..]], false)]
    #[case::misaligned(&[2][..], &[6][..], &[&[0][..], &[3][..], &[4][..]], false)]
    #[case::out_of_range(&[2][..], &[6][..], &[&[0][..], &[2][..], &[6][..]], false)]
    #[case::past_partial_edge(&[2][..], &[5][..], &[&[0][..], &[2][..], &[6][..]], false)]
    #[case::zero_chunk(&[0][..], &[6][..], &[][..], false)]
    #[case::rank_mismatch(&[2, 2][..], &[4][..], &[&[0][..], &[2][..]], false)]
    #[case::empty_extent(&[2][..], &[0][..], &[][..], true)]
    #[case::chunk_product_overflow(&[u64::MAX, 2][..], &[1, 1][..], &[&[0, 0][..]], false)]
    #[case::extent_product_overflow(&[u64::MAX, 2][..], &[u64::MAX, 2][..], &[&[0, 0][..]], false)]
    #[case::grid_product_overflow(&[1, 1][..], &[u64::MAX, 2][..], &[][..], false)]
    fn dense_coverage_requires_every_aligned_grid_slot_once(
        #[case] chunk_dims: &[u64],
        #[case] ds_dims: &[u64],
        #[case] offsets: &[&[u64]],
        #[case] dense: bool,
    ) {
        let chunks = offsets
            .iter()
            .enumerate()
            .map(|(i, offsets)| chunk(offsets, 0x1000 + i as u64 * 16))
            .collect();
        let index = ParsedChunkIndex::new(chunks, chunk_dims, ds_dims).unwrap();

        assert_eq!(index.dense_coverage(chunk_dims, ds_dims).is_some(), dense);
    }

    #[rstest]
    #[case::changed_chunks(&[1], &[4])]
    #[case::changed_extent(&[2], &[3])]
    #[case::changed_rank(&[2, 1], &[4, 1])]
    fn dense_coverage_is_associated_with_its_geometry(
        #[case] chunk_dims: &[u64],
        #[case] ds_dims: &[u64],
    ) {
        let index =
            ParsedChunkIndex::new(vec![chunk(&[2], 0x1000), chunk(&[0], 0x2000)], &[2], &[4])
                .unwrap();
        let first = index.dense_coverage(&[2], &[4]).unwrap();
        let second = index.dense_coverage(&[2], &[4]).unwrap();

        assert!(core::ptr::eq(first, second));
        assert!(index.dense_coverage(chunk_dims, ds_dims).is_none());
        assert_eq!(
            index.chunks().map(|chunk| chunk.coord).collect::<Vec<_>>(),
            [&[2][..], &[0][..]],
        );
    }
}
