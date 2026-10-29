//! Checked unit-stride dataset selections and their byte geometry.

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::convert::Narrow;
use crate::dataspace::DataspaceType;
use crate::error::FormatError;
use crate::read_spec::RawReadSpec;

/// A caller's region bounds or clamped leading-dimension row request.
pub(crate) enum RegionRequest<'a> {
    Bounds { start: &'a [u64], count: &'a [u64] },
    Rows { start: u64, count: u64 },
}

impl RegionRequest<'_> {
    pub(crate) fn parse<'a>(self, spec: RawReadSpec<'a>) -> Result<ReadRegion<'a>, FormatError> {
        match self {
            Self::Bounds { start, count } => ReadRegion::parse(spec, start, count),
            Self::Rows { start, count } => ReadRegion::rows(spec, start, count),
        }
    }
}

/// A selection within a dataset's current dimensions.
///
/// Nonempty selections have checked byte strides and a result length at most [`isize::MAX`].
/// Coordinates remain file-width integers until a backend converts them to memory indices.
#[derive(Clone)]
pub(crate) struct ReadRegion<'a> {
    spec: RawReadSpec<'a>,
    selection: Selection,
}

impl<'a> ReadRegion<'a> {
    /// Parses unit-stride bounds against the current dataspace.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::RegionRankMismatch`] if either slice has a different length from
    /// the rank, and [`FormatError::RegionOutOfBounds`] if any dimension exceeds its extent.
    /// Returns [`FormatError::OffsetOverflow`] if a byte stride overflows, and
    /// [`FormatError::ValueTooLargeForPlatform`] if the result exceeds the platform's buffer limit.
    /// Returns errors from reading the datatype's element size.
    pub(crate) fn parse(
        spec: RawReadSpec<'a>,
        start: &[u64],
        count: &[u64],
    ) -> Result<Self, FormatError> {
        let dimensions = &spec.dataspace().dimensions;
        let rank = dimensions.len();
        if start.len() != rank || count.len() != rank {
            return Err(FormatError::RegionRankMismatch {
                rank,
                start: start.len(),
                count: count.len(),
            });
        }
        for (dimension, ((&start, &count), &extent)) in
            start.iter().zip(count).zip(dimensions).enumerate()
        {
            if start > extent || count > extent - start {
                return Err(FormatError::RegionOutOfBounds {
                    dimension,
                    start,
                    count,
                    extent,
                });
            }
        }
        let element_size = crate::datatype::element_size_usize(spec.datatype())?;
        if count.contains(&0) || spec.dataspace().space_type == DataspaceType::Null {
            return Ok(Self {
                spec,
                selection: Selection::Empty,
            });
        }
        if start.iter().skip(1).all(|&coordinate| coordinate == 0)
            && count.iter().skip(1).eq(dimensions.iter().skip(1))
        {
            return Self::rows(
                spec,
                start.first().copied().unwrap_or(0),
                count.first().copied().unwrap_or(1),
            );
        }
        let mut axes = Vec::with_capacity(rank);
        let mut input_stride = element_size.get() as u64;
        let mut output_stride = input_stride;
        for ((&start, &count), &extent) in start.iter().zip(count).zip(dimensions).rev() {
            axes.push(RegionAxis {
                start,
                end: start + count,
                count,
                input_stride,
                output_stride,
            });
            input_stride = product(input_stride, extent)?;
            output_stride = product(output_stride, count)?;
        }
        axes.reverse();
        let byte_len = buffer_len(output_stride)?;
        Ok(Self {
            spec,
            selection: Selection::Elements { axes, byte_len },
        })
    }

    /// Selects complete rows after clamping the request to the leading dimension.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if byte strides overflow, and
    /// [`FormatError::ValueTooLargeForPlatform`] if the result exceeds the platform's buffer limit.
    /// Returns errors from reading the datatype's element size.
    pub(crate) fn rows(spec: RawReadSpec<'a>, start: u64, count: u64) -> Result<Self, FormatError> {
        let dimensions = &spec.dataspace().dimensions;
        let extent = dimensions.first().copied().unwrap_or(1);
        let start = start.min(extent);
        let count = count.min(extent - start);
        let element_size = crate::datatype::element_size_usize(spec.datatype())?;
        if count == 0
            || dimensions.contains(&0)
            || spec.dataspace().space_type == DataspaceType::Null
        {
            return Ok(Self {
                spec,
                selection: Selection::Empty,
            });
        }
        let row_bytes = dimensions
            .iter()
            .skip(1)
            .rev()
            .try_fold(element_size.get() as u64, |stride, &dimension| {
                product(stride, dimension)
            })?;
        product(row_bytes, extent)?;
        let byte_len = buffer_len(product(row_bytes, count)?)?;
        Ok(Self {
            spec,
            selection: Selection::Rows(RowRegion {
                start,
                end: start + count,
                row_bytes: row_bytes.to_usize()?,
                byte_len,
            }),
        })
    }

    pub(crate) fn spec(&self) -> RawReadSpec<'a> {
        self.spec
    }

    /// Returns the selected coordinates and byte strides in dimension order.
    pub(crate) fn axes(&self) -> impl Iterator<Item = RegionAxis> + '_ {
        let mut owned = match &self.selection {
            Selection::Elements { axes, .. } => axes.iter(),
            _ => [].iter(),
        };
        let mut dimensions = self.spec.dataspace().dimensions.iter().enumerate();
        let mut stride = self.row_band().map_or(0, |rows| rows.row_bytes as u64);
        core::iter::from_fn(move || match &self.selection {
            Selection::Empty => None,
            Selection::Elements { .. } => owned.next().cloned(),
            Selection::Rows(rows) => {
                let (dimension, &extent) = dimensions.next()?;
                let (start, count) = if dimension == 0 {
                    (rows.start, rows.end - rows.start)
                } else {
                    stride /= extent;
                    (0, extent)
                };
                Some(RegionAxis {
                    start,
                    end: start + count,
                    count,
                    input_stride: stride,
                    output_stride: stride,
                })
            }
        })
    }

    pub(crate) fn row_band(&self) -> Option<&RowRegion> {
        match &self.selection {
            Selection::Rows(rows) => Some(rows),
            _ => None,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        matches!(self.selection, Selection::Empty)
    }

    pub(crate) fn is_whole(&self) -> bool {
        match &self.selection {
            Selection::Empty => false,
            Selection::Elements { axes, .. } => axes
                .iter()
                .zip(&self.spec.dataspace().dimensions)
                .all(|(axis, &extent)| axis.start == 0 && axis.count == extent),
            Selection::Rows(rows) => {
                rows.start == 0
                    && rows.end
                        == self
                            .spec
                            .dataspace()
                            .dimensions
                            .first()
                            .copied()
                            .unwrap_or(1)
            }
        }
    }

    pub(crate) fn byte_len(&self) -> usize {
        match self.selection {
            Selection::Empty => 0,
            Selection::Rows(ref rows) => rows.byte_len,
            Selection::Elements { byte_len, .. } => byte_len,
        }
    }

    /// Returns the exclusive byte offset of the last selected element in contiguous storage.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if the offset exceeds [`u64::MAX`].
    pub(crate) fn end_offset(&self) -> Result<u64, FormatError> {
        if self.is_empty() {
            return Ok(0);
        }
        self.axes()
            .try_fold(u64::from(self.spec.datatype().type_size()), |end, axis| {
                sum(end, product(axis.end - 1, axis.input_stride)?)
            })
    }

    pub(crate) fn overlaps(&self, offsets: &[u64], dimensions: &[u64]) -> bool {
        let rank = self.spec.dataspace().dimensions.len();
        !self.is_empty()
            && offsets.len() >= rank
            && dimensions.len() == rank
            && self
                .axes()
                .zip(offsets)
                .zip(dimensions)
                .all(|((axis, &offset), &dimension)| {
                    offset < axis.end && offset.saturating_add(dimension) > axis.start
                })
    }

    /// Visits contiguous byte runs in row-major order, with offsets relative to the payload.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if an offset exceeds [`u64::MAX`], or an error
    /// returned by `visit`.
    pub(crate) fn visit_runs(
        &self,
        mut visit: impl FnMut(u64, usize) -> Result<(), FormatError>,
    ) -> Result<(), FormatError> {
        let axes = match &self.selection {
            Selection::Empty => return Ok(()),
            Selection::Rows(rows) => {
                return visit(product(rows.start, rows.row_bytes as u64)?, rows.byte_len);
            }
            Selection::Elements { axes, .. } => axes,
        };
        let mut split = axes.len();
        let mut run_len = self.spec.datatype().type_size() as u64;
        while let Some(axis) = split.checked_sub(1).and_then(|i| axes.get(i)) {
            if axis.input_stride != run_len {
                break;
            }
            run_len = product(run_len, axis.count)?;
            split -= 1;
        }
        let mut offset = axes.iter().try_fold(0u64, |offset, axis| {
            sum(offset, product(axis.start, axis.input_stride)?)
        })?;
        let run_len = run_len.to_usize()?;
        let mut coordinates = vec![0; split];
        loop {
            visit(offset, run_len)?;
            let mut advanced = false;
            for (coordinate, axis) in coordinates.iter_mut().zip(axes.iter()).rev() {
                *coordinate += 1;
                if *coordinate < axis.count {
                    offset = sum(offset, axis.input_stride)?;
                    advanced = true;
                    break;
                }
                offset -= product(*coordinate - 1, axis.input_stride)?;
                *coordinate = 0;
            }
            if !advanced {
                return Ok(());
            }
        }
    }
}

#[derive(Clone)]
enum Selection {
    Elements {
        axes: Vec<RegionAxis>,
        byte_len: usize,
    },
    Empty,
    Rows(RowRegion),
}

/// Complete rows represented without allocating per-dimension geometry.
#[derive(Clone)]
pub(crate) struct RowRegion {
    start: u64,
    end: u64,
    row_bytes: usize,
    byte_len: usize,
}

impl RowRegion {
    pub(crate) fn start(&self) -> u64 {
        self.start
    }
    pub(crate) fn end(&self) -> u64 {
        self.end
    }
    pub(crate) fn row_bytes(&self) -> usize {
        self.row_bytes
    }
}

/// Checked coordinates and byte strides for one selected dimension.
#[derive(Clone)]
pub(crate) struct RegionAxis {
    start: u64,
    end: u64,
    count: u64,
    input_stride: u64,
    output_stride: u64,
}

impl RegionAxis {
    pub(crate) fn start(&self) -> u64 {
        self.start
    }

    pub(crate) fn end(&self) -> u64 {
        self.end
    }

    pub(crate) fn count(&self) -> u64 {
        self.count
    }

    pub(crate) fn output_stride(&self) -> Result<usize, FormatError> {
        self.output_stride.to_usize()
    }
}

fn buffer_len(length: u64) -> Result<usize, FormatError> {
    let size = length.to_usize()?;
    if size > isize::MAX as usize {
        return Err(FormatError::ValueTooLargeForPlatform {
            value: length,
            target: "isize",
        });
    }
    Ok(size)
}

fn product(left: u64, right: u64) -> Result<u64, FormatError> {
    left.checked_mul(right).ok_or(FormatError::OffsetOverflow {
        offset: left,
        length: right,
    })
}

fn sum(left: u64, right: u64) -> Result<u64, FormatError> {
    left.checked_add(right).ok_or(FormatError::OffsetOverflow {
        offset: left,
        length: right,
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::ReadRegion;
    use crate::data_layout::DataLayout;
    use crate::dataspace::{Dataspace, DataspaceType};
    use crate::error::FormatError;
    use crate::fill_value::FillPattern;
    use crate::read_spec::RawReadSpec;
    use crate::type_builders;

    #[rstest]
    #[case::start_rank_mismatch(&[3, 4], &[1], &[1, 1], FormatError::RegionRankMismatch { rank: 2, start: 1, count: 2 })]
    #[case::count_rank_mismatch(&[3, 4], &[1, 1], &[1], FormatError::RegionRankMismatch { rank: 2, start: 2, count: 1 })]
    #[case::start_beyond_extent(&[3, 4], &[4, 1], &[0, 1], FormatError::RegionOutOfBounds { dimension: 0, start: 4, count: 0, extent: 3 })]
    #[case::count_beyond_extent(&[3, 4], &[1, 3], &[0, 2], FormatError::RegionOutOfBounds { dimension: 1, start: 3, count: 2, extent: 4 })]
    #[case::maximum_start(&[3, 4], &[1, u64::MAX], &[0, 1], FormatError::RegionOutOfBounds { dimension: 1, start: u64::MAX, count: 1, extent: 4 })]
    #[case::stride_product_overflow(&[u64::MAX, 2], &[0, 0], &[1, 1], FormatError::OffsetOverflow { offset: 2, length: u64::MAX })]
    fn regions_reject_rank_bounds_and_stride_errors(
        #[case] shape: &[u64],
        #[case] start: &[u64],
        #[case] count: &[u64],
        #[case] expected: FormatError,
    ) {
        let layout = DataLayout::Contiguous {
            address: None,
            size: 0,
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: shape.to_vec(),
            max_dimensions: None,
        };
        let datatype = type_builders::make_u8_type();
        let spec =
            RawReadSpec::parse(&layout, &dataspace, &datatype, None, FillPattern::ZERO).unwrap();
        let Err(error) = ReadRegion::parse(spec, start, count) else {
            panic!("invalid region accepted")
        };
        assert_eq!(error, expected);
    }

    #[rstest]
    #[case::partial_rows(&[3, 4], &[1, 1], &[2, 2], &[(5, 2), (9, 2)])]
    #[case::complete_rows(&[3, 4], &[1, 0], &[2, 4], &[(4, 8)])]
    #[case::three_dimensional(&[3, 4, 5], &[1, 1, 2], &[2, 2, 2], &[(27, 2), (32, 2), (47, 2), (52, 2)])]
    #[case::maximum_row_offsets(&[u64::MAX, 1], &[u64::MAX - 2, 0], &[2, 1], &[(u64::MAX - 2, 2)])]
    #[case::zero_columns(&[4_294_967_424, 0], &[4_294_967_360, 0], &[1, 0], &[])]
    fn regions_visit_contiguous_runs(
        #[case] shape: &[u64],
        #[case] start: &[u64],
        #[case] count: &[u64],
        #[case] expected: &[(u64, usize)],
    ) {
        let layout = DataLayout::Contiguous {
            address: None,
            size: 0,
        };
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: u8::try_from(shape.len()).unwrap(),
            dimensions: shape.to_vec(),
            max_dimensions: None,
        };
        let datatype = type_builders::make_u8_type();
        let spec =
            RawReadSpec::parse(&layout, &dataspace, &datatype, None, FillPattern::ZERO).unwrap();
        let region = ReadRegion::parse(spec, start, count).unwrap();
        let mut runs = Vec::new();
        region
            .visit_runs(|offset, length| {
                runs.push((offset, length));
                Ok(())
            })
            .unwrap();
        assert_eq!(runs, expected);
        assert_eq!(
            region.end_offset().unwrap(),
            expected
                .last()
                .map_or(0, |&(offset, length)| offset + length as u64)
        );
    }
}
