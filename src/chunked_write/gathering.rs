//! Checked traversal and byte gathering for dense chunk grids.

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::convert::Narrow;
use crate::error::FormatError;
use crate::fill_value::FillPattern;

/// A dense row-major chunk grid with reusable coordinate storage.
///
/// Dimensions, strides, and logical offsets are in elements and use [`u64`]. Buffer lengths
/// and indices are converted to [`usize`] when needed for allocation or copying.
#[derive(Debug)]
pub(crate) struct ChunkTraversal<'a> {
    shape: &'a [u64],
    chunk_dims: &'a [u64],
    counts: Vec<u64>,
    dataset_strides: Vec<u64>,
    chunk_strides: Vec<u64>,
    /// The current chunk's offsets in dataset elements.
    offsets: Vec<u64>,
    /// Coordinates within the chunk along every dimension except the innermost.
    row: Vec<u64>,
    count: u64,
    rows: u64,
    byte_len: usize,
    element_size: NonZeroUsize,
}

impl<'a> ChunkTraversal<'a> {
    /// Validates a non-scalar chunk grid and allocates reusable traversal state.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidChunkGeometry`] if the ranks differ, the rank or a chunk
    /// dimension is zero, or a derived count, stride, or byte length exceeds [`u64::MAX`].
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the full chunk byte length exceeds
    /// [`usize::MAX`].
    pub(crate) fn new(
        shape: &'a [u64],
        chunk_dims: &'a [u64],
        element_size: NonZeroUsize,
    ) -> Result<Self, FormatError> {
        if shape.is_empty() || shape.len() != chunk_dims.len() {
            return Err(FormatError::InvalidChunkGeometry(
                "chunk dimensions must match a non-scalar dataset's rank",
            ));
        }
        if chunk_dims.contains(&0) {
            return Err(FormatError::InvalidChunkGeometry(
                "chunk dimensions must all be non-zero",
            ));
        }
        let counts = shape
            .iter()
            .zip(chunk_dims)
            .map(|(dimension, chunk)| dimension.div_ceil(*chunk))
            .collect::<Vec<_>>();
        let count = product(&counts)?;
        let byte_len = checked_mul(product(chunk_dims)?, element_size.get().to_u64())?.narrow()?;
        checked_mul(product(shape)?, element_size.get().to_u64())?;
        Ok(Self {
            shape,
            chunk_dims,
            counts,
            dataset_strides: strides(shape)?,
            chunk_strides: strides(chunk_dims)?,
            offsets: vec![0; shape.len()],
            row: vec![0; shape.len() - 1],
            count,
            rows: product(&chunk_dims[..shape.len() - 1])?,
            byte_len,
            element_size,
        })
    }

    /// Calls `visitor` for each chunk in dense row-major order, starting from the first chunk.
    ///
    /// A grid with a zero extent is empty. The visitor may gather the current chunk into a
    /// reusable buffer with [`gather`](Self::gather).
    ///
    /// # Errors
    ///
    /// Returns an error from `visitor` immediately.
    pub(crate) fn visit(
        &mut self,
        mut visitor: impl FnMut(&mut Self) -> Result<(), FormatError>,
    ) -> Result<(), FormatError> {
        for linear in 0..self.count {
            let mut remaining = linear;
            for dimension in (0..self.shape.len()).rev() {
                self.offsets[dimension] = checked_mul(
                    remaining % self.counts[dimension],
                    self.chunk_dims[dimension],
                )?;
                remaining /= self.counts[dimension];
            }
            visitor(self)?;
        }
        Ok(())
    }

    /// Copies available whole elements from `raw` into `destination` for the current chunk.
    ///
    /// Partial edge chunks are initialized with `fill`. Full chunks with incomplete input are
    /// initialized with zeros. With complete input, every byte of a full chunk is overwritten.
    /// The caller may reuse `destination` between visits.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if `destination` has a length other than
    /// [`byte_len`](Self::byte_len). Returns [`FormatError::UnreadableFillValue`] if this is a
    /// partial edge chunk and `fill` is unknown.
    pub(crate) fn gather(
        &mut self,
        raw: &[u8],
        fill: FillPattern<'_>,
        destination: &mut [u8],
    ) -> Result<(), FormatError> {
        let &mut Self {
            shape,
            chunk_dims,
            counts: _,
            dataset_strides: _,
            chunk_strides: _,
            ref offsets,
            row: _,
            count: _,
            rows,
            byte_len,
            element_size,
        } = self;
        if destination.len() != byte_len {
            return Err(FormatError::Internal(
                "chunk gathering destination has an incorrect length".into(),
            ));
        }
        if !self.is_whole() {
            destination.fill(0);
            fill.apply(destination)?;
        } else if self.input_end()? > raw.len().to_u64() {
            destination.fill(0);
        }

        let inner = shape.len() - 1;
        let row_bytes = checked_mul(
            chunk_dims[inner].min(shape[inner] - offsets[inner]),
            element_size.get().to_u64(),
        )?
        .narrow::<usize>()?;
        self.row.fill(0);
        for _ in 0..rows {
            if let Some((source, destination_start)) = self.row_offsets()? {
                if source < raw.len().to_u64() {
                    let source = source.narrow::<usize>()?;
                    let destination_start = destination_start.narrow::<usize>()?;
                    let available = row_bytes.min(raw.len() - source);
                    let available = available - available % element_size;
                    destination[destination_start..destination_start + available]
                        .copy_from_slice(&raw[source..source + available]);
                }
            }
            let row = &mut self.row;
            for dimension in (0..inner).rev() {
                row[dimension] += 1;
                if row[dimension] < chunk_dims[dimension] {
                    break;
                }
                row[dimension] = 0;
            }
        }
        Ok(())
    }

    /// Borrows the current chunk from `raw` when its full contents occupy one available slice.
    ///
    /// Returns `None` for a partial edge chunk, a chunk with gaps between its rows, or input
    /// that ends before the chunk's last byte.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidChunkGeometry`] if a derived byte offset exceeds
    /// [`u64::MAX`].
    pub(crate) fn contiguous<'b>(&self, raw: &'b [u8]) -> Result<Option<&'b [u8]>, FormatError> {
        if !self.is_whole() {
            return Ok(None);
        }
        let start = self
            .offsets
            .iter()
            .zip(&self.dataset_strides)
            .try_fold(0, |sum, (offset, stride)| {
                checked_add(sum, checked_mul(*offset, *stride)?)
            })?;
        let start = checked_mul(start, self.element_size.get().to_u64())?;
        let end = self.input_end()?;
        if end.checked_sub(start) != Some(self.byte_len.to_u64()) || end > raw.len().to_u64() {
            return Ok(None);
        }
        Ok(raw.get(start.narrow::<usize>()?..end.narrow::<usize>()?))
    }

    /// Returns the current chunk's dataset offsets in elements.
    pub(crate) fn offsets(&self) -> &[u64] {
        &self.offsets
    }

    /// Returns the number of chunks in the grid, including partial edge chunks.
    pub(crate) fn chunk_count(&self) -> u64 {
        self.count
    }

    /// Returns the full chunk byte length, including edge padding.
    pub(crate) fn byte_len(&self) -> usize {
        self.byte_len
    }

    fn is_whole(&self) -> bool {
        self.shape
            .iter()
            .zip(self.chunk_dims)
            .zip(&self.offsets)
            .all(|((&dimension, &chunk), &offset)| chunk <= dimension - offset)
    }

    /// Returns the byte offset past the last element of the current full chunk in dataset order.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidChunkGeometry`] if a derived offset exceeds [`u64::MAX`].
    fn input_end(&self) -> Result<u64, FormatError> {
        let last = self
            .offsets
            .iter()
            .zip(self.chunk_dims)
            .zip(&self.dataset_strides)
            .try_fold(0, |sum, ((offset, chunk), stride)| {
                checked_add(sum, checked_mul(checked_add(*offset, chunk - 1)?, *stride)?)
            })?;
        checked_mul(checked_add(last, 1)?, self.element_size.get().to_u64())
    }

    /// Returns source and destination byte offsets for the current row, or `None` outside the
    /// dataset extent.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidChunkGeometry`] if a derived offset exceeds [`u64::MAX`].
    fn row_offsets(&self) -> Result<Option<(u64, u64)>, FormatError> {
        let Self {
            shape,
            chunk_dims: _,
            counts: _,
            dataset_strides,
            chunk_strides,
            offsets,
            row,
            count: _,
            rows: _,
            byte_len: _,
            element_size,
        } = self;
        let inner = shape.len() - 1;
        let mut source = offsets[inner];
        let mut destination = 0;
        for (dimension, coordinate) in row.iter().enumerate() {
            let Some(global) = offsets[dimension]
                .checked_add(*coordinate)
                .filter(|global| *global < shape[dimension])
            else {
                return Ok(None);
            };
            source = checked_add(source, checked_mul(global, dataset_strides[dimension])?)?;
            destination = checked_add(
                destination,
                checked_mul(*coordinate, chunk_strides[dimension])?,
            )?;
        }
        Ok(Some((
            checked_mul(source, element_size.get().to_u64())?,
            checked_mul(destination, element_size.get().to_u64())?,
        )))
    }
}

fn strides(dimensions: &[u64]) -> Result<Vec<u64>, FormatError> {
    let mut strides = vec![1; dimensions.len()];
    for dimension in (0..dimensions.len() - 1).rev() {
        strides[dimension] = checked_mul(strides[dimension + 1], dimensions[dimension + 1])?;
    }
    Ok(strides)
}

fn product(dimensions: &[u64]) -> Result<u64, FormatError> {
    dimensions
        .iter()
        .try_fold(1, |product, dimension| checked_mul(product, *dimension))
}

fn checked_mul(left: u64, right: u64) -> Result<u64, FormatError> {
    left.checked_mul(right)
        .ok_or(FormatError::InvalidChunkGeometry(
            "chunk gathering geometry exceeds the addressable range",
        ))
}

fn checked_add(left: u64, right: u64) -> Result<u64, FormatError> {
    left.checked_add(right)
        .ok_or(FormatError::InvalidChunkGeometry(
            "chunk gathering geometry exceeds the addressable range",
        ))
}

#[cfg(test)]
mod tests {
    use hdf5_pure_core::FillTime;
    use rstest::rstest;

    use super::*;
    use crate::convert;

    #[rstest]
    #[case::rank_one(&[7], &[3], 14, &[true, true, false])]
    #[case::short_rank_one(&[7], &[3], 9, &[true, false, false])]
    #[case::full_width(&[4, 6], &[2, 6], 48, &[true, true])]
    #[case::short_full_width(&[4, 6], &[2, 6], 47, &[true, false])]
    #[case::partial_rows(&[4, 6], &[2, 3], 48, &[false; 4])]
    #[case::single_rows(&[2, 6], &[1, 3], 24, &[true; 4])]
    #[case::slabs(&[2, 3, 4], &[1, 3, 4], 48, &[true, true])]
    #[case::frames(&[2, 3, 4], &[2, 1, 4], 48, &[false; 3])]
    #[case::trailing_runs(&[1, 2, 4], &[1, 1, 2], 16, &[true; 4])]
    #[case::degenerate(&[2, 1, 4], &[1, 1, 4], 16, &[true, true])]
    #[case::oversized(&[2, 2], &[3, 2], 24, &[false])]
    fn borrows_only_complete_contiguous_chunks(
        #[case] shape: &[u64],
        #[case] chunks: &[u64],
        #[case] input_len: usize,
        #[case] expected: &[bool],
    ) {
        let raw = (0..input_len)
            .map(|byte| u8::try_from(byte).unwrap())
            .collect::<Vec<_>>();
        let mut traversal = ChunkTraversal::new(shape, chunks, convert::nz(2)).unwrap();
        let mut borrowed = Vec::new();
        traversal
            .visit(|chunk| {
                let contiguous = chunk.contiguous(&raw)?;
                borrowed.push(contiguous.is_some());
                if let Some(contiguous) = contiguous {
                    let mut gathered = vec![0; chunk.byte_len()];
                    chunk.gather(&raw, FillPattern::ZERO, &mut gathered)?;
                    assert_eq!(contiguous, gathered);
                    let start = chunk
                        .offsets
                        .iter()
                        .zip(&chunk.dataset_strides)
                        .map(|(offset, stride)| offset * stride)
                        .sum::<u64>()
                        * 2;
                    assert_eq!(
                        contiguous.as_ptr(),
                        raw[usize::try_from(start).unwrap()..].as_ptr()
                    );
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(borrowed, expected);
    }

    #[rstest]
    #[case::rank_one(&[7], &[3], &[0, 3, 6])]
    #[case::rank_two(&[3, 5], &[2, 3], &[0, 0, 0, 3, 2, 0, 2, 3])]
    #[case::rank_three(
        &[2, 3, 4],
        &[1, 2, 3],
        &[0, 0, 0, 0, 0, 3, 0, 2, 0, 0, 2, 3, 1, 0, 0, 1, 0, 3, 1, 2, 0, 1, 2, 3]
    )]
    fn visits_chunks_in_dense_row_major_order(
        #[case] shape: &[u64],
        #[case] chunks: &[u64],
        #[case] expected: &[u64],
    ) {
        let mut traversal = ChunkTraversal::new(shape, chunks, convert::nz(1)).unwrap();
        let mut offsets = Vec::new();
        traversal
            .visit(|chunk| {
                offsets.extend_from_slice(&chunk.offsets);
                Ok(())
            })
            .unwrap();
        assert_eq!(offsets, expected);
    }

    #[rstest]
    #[case::full(&[2, 4], &[1, 2], 16, false)]
    #[case::edge(&[3, 5], &[2, 3], 30, true)]
    #[case::short_full(&[2, 4], &[1, 2], 7, false)]
    #[case::short_edge(&[3, 5], &[2, 3], 13, true)]
    #[case::rank_one(&[5], &[3], 9, true)]
    #[case::rank_three(&[3, 4, 5], &[2, 3, 2], 119, true)]
    fn gathering_reuses_destination_and_preserves_element_aligned_input(
        #[case] shape: &[u64],
        #[case] chunks: &[u64],
        #[case] input_len: usize,
        #[case] custom_fill: bool,
    ) {
        let raw = (0..input_len)
            .map(|byte| u8::try_from(byte).unwrap())
            .collect::<Vec<_>>();
        let fill = if custom_fill {
            FillPattern::new(FillTime::IfSet, Some(&[241, 242]), convert::nz(2))
        } else {
            FillPattern::ZERO
        };
        let mut traversal = ChunkTraversal::new(shape, chunks, convert::nz(2)).unwrap();
        let mut destination = vec![233; traversal.byte_len()];
        let pointer = destination.as_ptr();
        traversal
            .visit(|chunk| {
                chunk.gather(&raw, fill, &mut destination)?;
                let whole = shape
                    .iter()
                    .zip(chunks)
                    .zip(&chunk.offsets)
                    .all(|((dimension, extent), offset)| offset + extent <= *dimension);
                let mut expected = Vec::new();
                for element in 0..product(chunks).unwrap() {
                    let mut remaining = element;
                    let mut source = 0;
                    let mut stride = 1;
                    let mut in_bounds = true;
                    for dimension in (0..shape.len()).rev() {
                        let coordinate = remaining % chunks[dimension];
                        remaining /= chunks[dimension];
                        let global = coordinate + chunk.offsets[dimension];
                        in_bounds &= global < shape[dimension];
                        source += global * stride;
                        stride *= shape[dimension];
                    }
                    let source = usize::try_from(source).unwrap() * 2;
                    if in_bounds && source + 2 <= raw.len() {
                        expected.extend_from_slice(&raw[source..source + 2]);
                    } else if !whole && custom_fill {
                        expected.extend_from_slice(&[241, 242]);
                    } else {
                        expected.extend_from_slice(&[0, 0]);
                    }
                }
                assert_eq!(destination, expected, "offsets {:?}", chunk.offsets);
                assert_eq!(destination.as_ptr(), pointer);
                destination.fill(233);
                Ok(())
            })
            .unwrap();
    }

    #[rstest]
    #[case::zero_leading(&[0, 3])]
    #[case::zero_trailing(&[3, 0])]
    fn empty_datasets_visit_no_chunks(#[case] shape: &[u64]) {
        let mut traversal = ChunkTraversal::new(shape, &[2, 2], convert::nz(1)).unwrap();
        assert_eq!(traversal.chunk_count(), 0);
        traversal
            .visit(|_| panic!("empty dataset visited a chunk"))
            .unwrap();
    }

    #[rstest]
    #[case::scalar(&[], &[], "chunk dimensions must match a non-scalar dataset's rank")]
    #[case::rank(&[2, 3], &[2], "chunk dimensions must match a non-scalar dataset's rank")]
    #[case::zero(&[2], &[0], "chunk dimensions must all be non-zero")]
    #[case::chunk_size(&[1, 1], &[u64::MAX, 2], "chunk gathering geometry exceeds the addressable range")]
    #[case::dataset_size(&[u64::MAX, 2], &[1, 2], "chunk gathering geometry exceeds the addressable range")]
    #[case::chunk_count(&[u64::MAX, 2], &[1, 1], "chunk gathering geometry exceeds the addressable range")]
    fn rejects_invalid_geometry(
        #[case] shape: &[u64],
        #[case] chunks: &[u64],
        #[case] reason: &'static str,
    ) {
        let error = ChunkTraversal::new(shape, chunks, convert::nz(1)).unwrap_err();
        assert_eq!(error, FormatError::InvalidChunkGeometry(reason));
    }

    #[test]
    fn rejects_a_destination_with_the_wrong_length() {
        let mut traversal = ChunkTraversal::new(&[2], &[2], convert::nz(1)).unwrap();
        let error = traversal
            .gather(&[1, 2], FillPattern::ZERO, &mut [0])
            .unwrap_err();
        assert_eq!(
            error,
            FormatError::Internal("chunk gathering destination has an incorrect length".into())
        );
    }
}
