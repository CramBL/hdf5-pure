//! Parsed chunk descriptors and region selection plans.

#[cfg(not(feature = "std"))]
use alloc::format;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::chunked_read::ChunkInfo;
use crate::chunked_read::DenseCoverage;
use crate::error::FormatError;
use crate::read_region::ReadRegion;
use crate::read_region::RegionAxis;

#[cfg(all(test, feature = "std"))]
#[path = "parsed_chunk_index_test_util.rs"]
pub(crate) mod test_util;

/// The chunks a dataset's chunk index lists, in ascending order of address.
///
/// A `ParsedChunkIndex` holds each chunk once, as the [`ChunkInfo`] the walk of the chunk index
/// returned, and a chunk's coordinate borrows the first `rank` of its offsets. A key of a version
/// 1 B-tree stores one offset more than the dataset has dimensions, the offset within the
/// datatype, as defined in "Version 1 B-trees" of the [format specification, version 4.0][spec].
///
/// The cached readers visit the chunks in this order, in which a
/// [`ChunkSpanReader`](crate::chunk_span::ChunkSpanReader) reads each span once.
/// A logical lookup holds descriptor ordinals in leading-coordinate order for region selection.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v1
#[derive(Debug)]
pub(crate) struct ParsedChunkIndex {
    chunks: Vec<ChunkInfo>,
    rank: usize,
    chunk_dims: Vec<u64>,
    ds_dims: Vec<u64>,
    /// Ordinals in `chunks`, sorted by the first offset.
    logical: Vec<usize>,
    coverage: Option<DenseCoverage>,
}

impl ParsedChunkIndex {
    /// Sorts `chunks` by address and retains dense coverage where it can be proven.
    ///
    /// The coordinate rank is `ds_dims.len()`. The retained proof applies to the supplied chunk
    /// dimensions and dataset extent. Stores these dimensions with a logical lookup of descriptor
    /// ordinals for [`Self::select`].
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
        #[cfg(all(test, feature = "std"))]
        test_util::constructed();
        chunks.sort_unstable_by_key(|chunk| chunk.address);
        let mut logical: Vec<_> = (0..chunks.len()).collect();
        logical.sort_unstable_by_key(|&ordinal| chunks[ordinal].offsets.first().copied());
        let mut index = Self {
            chunks,
            rank,
            chunk_dims: chunk_dims.to_vec(),
            ds_dims: ds_dims.to_vec(),
            logical,
            coverage: None,
        };
        index.coverage = DenseCoverage::new(&index, chunk_dims, ds_dims);
        Ok(index)
    }

    /// Plans clipped contributions to `region` in ascending order of chunk address.
    ///
    /// Searches the logical lookup for overlapping leading intervals before testing the other
    /// dimensions. Complete coverage requires either the retained [`DenseCoverage`] proof or
    /// unique, aligned selected chunks whose clipped contributions cover the region exactly.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the dataset extent differs from the index's extent,
    /// the chunk rank differs from the dataset rank, or any chunk dimension is zero.
    pub(crate) fn select<'a, 'spec>(
        &'a self,
        region: &'a ReadRegion<'spec>,
    ) -> Result<ChunkSelection<'a, 'spec>, FormatError> {
        let Self {
            chunks,
            rank,
            chunk_dims,
            ds_dims,
            logical: _,
            coverage,
        } = self;
        if *ds_dims != region.spec().dataspace().dimensions
            || chunk_dims.len() != *rank
            || chunk_dims.contains(&0)
        {
            return Err(FormatError::Internal(
                "selection geometry differs from parsed chunk index".into(),
            ));
        }
        if region.is_empty() {
            return Ok(ChunkSelection {
                index: self,
                region,
                selected: Vec::new(),
                intersections: Vec::new(),
                coverage: SelectionCoverage::Empty,
            });
        }
        let axes: Vec<_> = region.axes().collect();
        let mut selected: Vec<_> = self
            .candidates(&axes)
            .iter()
            .copied()
            .filter(|&ordinal| {
                #[cfg(all(test, feature = "std"))]
                test_util::candidate();
                region.overlaps(&chunks[ordinal].offsets, chunk_dims)
            })
            .collect();
        selected.sort_unstable();
        let mut intersections = Vec::new();
        for &ordinal in &selected {
            for ((axis, &offset), &edge) in
                axes.iter().zip(&chunks[ordinal].offsets).zip(chunk_dims)
            {
                intersections.push(ClippedAxis {
                    start: axis.start().max(offset),
                    end: axis.end().min(offset.saturating_add(edge)),
                });
            }
        }
        let coverage =
            if coverage.is_some() || self.covers_selection(&selected, &intersections, &axes) {
                SelectionCoverage::Complete
            } else {
                SelectionCoverage::PotentiallyUncovered
            };
        Ok(ChunkSelection {
            index: self,
            region,
            selected,
            intersections,
            coverage,
        })
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
    /// Returns descriptor ordinals whose leading intervals overlap the first selected axis.
    ///
    /// Returns the complete logical lookup for a scalar selection.
    fn candidates(&self, axes: &[RegionAxis]) -> &[usize] {
        match (axes.first(), self.chunk_dims.first()) {
            (Some(axis), Some(&edge)) => {
                let start = axis.start().saturating_sub(edge - 1);
                let first = self
                    .logical
                    .partition_point(|&ordinal| self.chunks[ordinal].offsets[0] < start);
                let end = self
                    .logical
                    .partition_point(|&ordinal| self.chunks[ordinal].offsets[0] < axis.end());
                &self.logical[first..end]
            }
            _ => &self.logical,
        }
    }

    /// Returns whether unique, aligned selected chunks partition the region.
    ///
    /// Returns `false` if an element count overflows or a chunk starts outside the dataset.
    fn covers_selection(
        &self,
        selected: &[usize],
        intersections: &[ClippedAxis],
        axes: &[RegionAxis],
    ) -> bool {
        let Self {
            chunks,
            rank,
            chunk_dims,
            ds_dims,
            logical: _,
            coverage: _,
        } = self;
        let Some(elements) = axes
            .iter()
            .try_fold(1u64, |count, axis| count.checked_mul(axis.count()))
        else {
            return false;
        };
        let mut logical = selected.to_vec();
        logical
            .sort_unstable_by(|&a, &b| chunks[a].offsets[..*rank].cmp(&chunks[b].offsets[..*rank]));
        if logical
            .windows(2)
            .any(|pair| chunks[pair[0]].offsets[..*rank] == chunks[pair[1]].offsets[..*rank])
        {
            return false;
        }
        let mut covered = 0u64;
        for (position, &ordinal) in selected.iter().enumerate() {
            if chunks[ordinal]
                .offsets
                .iter()
                .zip(chunk_dims)
                .zip(ds_dims)
                .any(|((&offset, &edge), &extent)| !offset.is_multiple_of(edge) || offset >= extent)
            {
                return false;
            }
            let Some(contribution) = intersections
                .iter()
                .skip(position * *rank)
                .take(*rank)
                .try_fold(1u64, |count, axis| count.checked_mul(axis.end - axis.start))
            else {
                return false;
            };
            let Some(total) = covered.checked_add(contribution) else {
                return false;
            };
            covered = total;
        }
        covered == elements
    }
}

/// Descriptor ordinals, clipped intersections, and coverage for a region of one parsed index.
///
/// Constructed by [`ParsedChunkIndex::select`]. Selected ordinals are in ascending descriptor
/// order, and each intersection has one [`ClippedAxis`] per dataset dimension. Borrows the region
/// used to establish these intersections and coverage.
pub(crate) struct ChunkSelection<'a, 'spec> {
    index: &'a ParsedChunkIndex,
    region: &'a ReadRegion<'spec>,
    selected: Vec<usize>,
    intersections: Vec<ClippedAxis>,
    coverage: SelectionCoverage,
}

impl<'a, 'spec> ChunkSelection<'a, 'spec> {
    /// Borrows the descriptor at `position` in the address-ordered selection.
    ///
    /// Returns `None` if `position` is past the selected descriptors.
    pub(crate) fn chunk(&self, position: usize) -> Option<IndexedChunk<'a>> {
        let &ordinal = self.selected.get(position)?;
        let info = &self.index.chunks[ordinal];
        Some(IndexedChunk {
            info,
            coord: &info.offsets[..self.index.rank],
        })
    }

    /// Borrows the region used to establish the selected contributions.
    pub(crate) fn region(&self) -> &'a ReadRegion<'spec> {
        self.region
    }

    /// Borrows the chunk dimensions used to establish coverage and intersections.
    pub(crate) fn chunk_dims(&self) -> &'a [u64] {
        &self.index.chunk_dims
    }

    /// Returns the coverage established for this region and index.
    pub(crate) fn coverage(&self) -> &SelectionCoverage {
        &self.coverage
    }

    /// Borrows the selected descriptors and their intersections in ascending address order.
    pub(crate) fn chunks(&self) -> impl ExactSizeIterator<Item = SelectedChunk<'_>> {
        let Self {
            index,
            region: _,
            selected,
            intersections,
            coverage,
        } = self;
        let count = match coverage {
            SelectionCoverage::Empty => 0,
            SelectionCoverage::Complete | SelectionCoverage::PotentiallyUncovered => selected.len(),
        };
        selected[..count]
            .iter()
            .enumerate()
            .map(|(position, &ordinal)| {
                let info = &index.chunks[ordinal];
                let rank = index.rank;
                SelectedChunk {
                    chunk: IndexedChunk {
                        info,
                        coord: &info.offsets[..rank],
                    },
                    intersection: &intersections[position * rank..(position + 1) * rank],
                }
            })
    }
}

/// The coverage established from a region and its selected chunk descriptors.
#[derive(Debug, PartialEq)]
pub(crate) enum SelectionCoverage {
    /// Every selected element has exactly one contribution.
    Complete,
    /// The region has no elements.
    Empty,
    /// Complete coverage could not be established.
    PotentiallyUncovered,
}

/// A borrowed descriptor and its intersection with the selected region.
pub(crate) struct SelectedChunk<'a> {
    pub(crate) chunk: IndexedChunk<'a>,
    /// Global coordinate bounds in dataset dimension order.
    pub(crate) intersection: &'a [ClippedAxis],
}

/// Inclusive start and exclusive end of a chunk's intersection along one selected axis.
pub(crate) struct ClippedAxis {
    start: u64,
    end: u64,
}

impl ClippedAxis {
    pub(crate) fn start(&self) -> u64 {
        self.start
    }

    pub(crate) fn end(&self) -> u64 {
        self.end
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
    use crate::data_layout::DataLayout;
    use crate::dataspace::{Dataspace, DataspaceType};
    use crate::fill_value::FillPattern;
    use crate::read_spec::RawReadSpec;
    use crate::type_builders;

    fn chunk(offsets: &[u64], address: u64) -> ChunkInfo {
        ChunkInfo {
            chunk_size: StoredChunkSize::v4(16),
            filter_mask: 0,
            offsets: offsets.to_vec(),
            address: StoredAddress::new(address),
        }
    }

    #[rstest]
    #[case::single_row(3_000, 1, 3_000, 1)]
    #[case::crosses_edge(2_999, 2, 2_998, 2)]
    #[case::tail(8_190, 2, 8_190, 1)]
    fn logical_lookup_limits_candidates_to_intersecting_leading_intervals(
        #[case] start: u64,
        #[case] count: u64,
        #[case] first: u64,
        #[case] candidates: usize,
    ) {
        let layout = DataLayout::Contiguous {
            address: None,
            size: 0,
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![8_192],
            max_dimensions: None,
        };
        let datatype = type_builders::make_u8_type();
        let spec =
            RawReadSpec::parse(&layout, &dataspace, &datatype, None, FillPattern::ZERO).unwrap();
        let region = ReadRegion::rows(spec, start, count).unwrap();
        let index = ParsedChunkIndex::new(
            (0..4_096)
                .map(|row| chunk(&[row * 2], 0x1000 + (4_096 - row) * 16))
                .collect(),
            &[2],
            &[8_192],
        )
        .unwrap();
        let axes: Vec<_> = region.axes().collect();
        let ordinals = index.candidates(&axes);
        assert_eq!(ordinals.len(), candidates);
        assert_eq!(index.descriptors()[ordinals[0]].offsets, [first]);
        let selection = index.select(&region).unwrap();
        assert_eq!(selection.chunks().len(), candidates);
        assert_eq!(selection.coverage, SelectionCoverage::Complete);
    }

    #[test]
    fn selection_intersections_saturate_at_file_width() {
        let layout = DataLayout::Contiguous {
            address: None,
            size: 0,
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![u64::MAX],
            max_dimensions: None,
        };
        let datatype = type_builders::make_u8_type();
        let spec =
            RawReadSpec::parse(&layout, &dataspace, &datatype, None, FillPattern::ZERO).unwrap();
        let region = ReadRegion::rows(spec, u64::MAX - 2, 2).unwrap();
        let index =
            ParsedChunkIndex::new(vec![chunk(&[u64::MAX - 3], 0x1000)], &[4], &[u64::MAX]).unwrap();
        let selection = index.select(&region).unwrap();
        let clipped: Vec<_> = selection
            .chunks()
            .flat_map(|selected| {
                selected
                    .intersection
                    .iter()
                    .map(|axis| (axis.start(), axis.end()))
            })
            .collect();
        assert_eq!(clipped, [(u64::MAX - 2, u64::MAX)]);
        assert_eq!(selection.coverage, SelectionCoverage::Complete);
    }

    #[rstest]
    #[case::covered_subset(&[0, 0], &[1, 2], &[&[0, 0][..]][..], SelectionCoverage::Complete)]
    #[case::clipped_edge(&[2, 3], &[1, 2], &[&[2, 2][..], &[2, 4][..]][..], SelectionCoverage::Complete)]
    #[case::interior_hole(&[0, 0], &[2, 5], &[&[0, 0][..], &[0, 4][..]][..], SelectionCoverage::PotentiallyUncovered)]
    #[case::duplicate(&[0, 0], &[1, 2], &[&[0, 0][..], &[0, 0][..]][..], SelectionCoverage::PotentiallyUncovered)]
    #[case::misaligned(&[1, 1], &[1, 2], &[&[1, 1][..]][..], SelectionCoverage::PotentiallyUncovered)]
    #[case::empty(&[1, 1], &[0, 2], &[&[0, 0][..]][..], SelectionCoverage::Empty)]
    fn selections_prove_only_aligned_unique_complete_coverage(
        #[case] start: &[u64],
        #[case] count: &[u64],
        #[case] offsets: &[&[u64]],
        #[case] expected: SelectionCoverage,
    ) {
        let layout = DataLayout::Contiguous {
            address: None,
            size: 0,
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: vec![3, 5],
            max_dimensions: None,
        };
        let datatype = type_builders::make_u8_type();
        let spec =
            RawReadSpec::parse(&layout, &dataspace, &datatype, None, FillPattern::ZERO).unwrap();
        let region = ReadRegion::parse(spec, start, count).unwrap();
        let chunks = offsets
            .iter()
            .enumerate()
            .map(|(ordinal, offsets)| chunk(offsets, 0x1000 + ordinal as u64 * 16))
            .collect();
        let index = ParsedChunkIndex::new(chunks, &[2, 2], &[3, 5]).unwrap();
        let selection = index.select(&region).unwrap();
        assert_eq!(selection.coverage, expected);
        assert_eq!(
            selection.chunks().len(),
            if region.is_empty() { 0 } else { offsets.len() }
        );
    }

    #[test]
    fn selections_retain_address_order_and_clipped_destinations() {
        let layout = DataLayout::Contiguous {
            address: None,
            size: 0,
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: vec![4, 5],
            max_dimensions: None,
        };
        let datatype = type_builders::make_u8_type();
        let spec =
            RawReadSpec::parse(&layout, &dataspace, &datatype, None, FillPattern::ZERO).unwrap();
        let region = ReadRegion::parse(spec, &[1, 1], &[2, 3]).unwrap();
        let index = ParsedChunkIndex::new(
            vec![
                chunk(&[0, 0, 0], 0x4000),
                chunk(&[2, 2, 0], 0x1000),
                chunk(&[0, 2, 0], 0x3000),
                chunk(&[2, 0, 0], 0x2000),
                chunk(&[0, 4, 0], 0x5000),
            ],
            &[2, 2],
            &[4, 5],
        )
        .unwrap();
        let selection = index.select(&region).unwrap();
        assert_eq!(selection.coverage, SelectionCoverage::Complete);
        let contributions: Vec<_> = selection
            .chunks()
            .map(|selected| {
                (
                    selected.chunk.info.address.get(),
                    selected
                        .intersection
                        .iter()
                        .map(|axis| (axis.start(), axis.end()))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        assert_eq!(
            contributions,
            vec![
                (0x1000, vec![(2, 3), (2, 4)]),
                (0x2000, vec![(2, 3), (1, 2)]),
                (0x3000, vec![(1, 2), (2, 4)]),
                (0x4000, vec![(1, 2), (1, 2)]),
            ]
        );
        assert!(core::ptr::eq(
            selection.chunks().next().unwrap().chunk.info,
            &index.descriptors()[0]
        ));
    }

    #[rstest]
    #[case::different_extent(&[4], &[2])]
    #[case::different_rank(&[3, 5], &[2])]
    #[case::zero_edge(&[3, 5], &[0, 2])]
    fn selections_reject_different_or_invalid_index_geometry(
        #[case] index_extent: &[u64],
        #[case] chunk_dims: &[u64],
    ) {
        let layout = DataLayout::Contiguous {
            address: None,
            size: 0,
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: vec![3, 5],
            max_dimensions: None,
        };
        let datatype = type_builders::make_u8_type();
        let spec =
            RawReadSpec::parse(&layout, &dataspace, &datatype, None, FillPattern::ZERO).unwrap();
        let region = ReadRegion::parse(spec, &[0, 0], &[1, 1]).unwrap();
        let index = ParsedChunkIndex::new(Vec::new(), chunk_dims, index_extent).unwrap();
        let Err(error) = index.select(&region) else {
            panic!("invalid geometry accepted")
        };
        assert_eq!(
            error,
            FormatError::Internal("selection geometry differs from parsed chunk index".into())
        );
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
