#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::convert::Narrow;
use crate::dataspace::DataspaceType;
use crate::error::FormatError;
use crate::read_spec::RawReadSpec;

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

#[derive(Clone)]
pub(crate) struct ReadRegion<'a> {
    spec: RawReadSpec<'a>,
    selection: Selection,
}

impl<'a> ReadRegion<'a> {
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
        Ok(Self {
            spec,
            selection: Selection::Elements {
                axes,
                byte_len: output_stride,
            },
        })
    }

    pub(crate) fn rows(spec: RawReadSpec<'a>, start: u64, count: u64) -> Result<Self, FormatError> {
        let dimensions = &spec.dataspace().dimensions;
        let extent = dimensions.first().copied().unwrap_or(1);
        let start = start.min(extent);
        let count = count.min(extent - start);
        if dimensions.is_empty() && count == 0 {
            crate::datatype::element_size_usize(spec.datatype())?;
            return Ok(Self {
                spec,
                selection: Selection::Empty,
            });
        }
        let mut starts = vec![0; dimensions.len()];
        let mut counts = dimensions.clone();
        if let (Some(first_start), Some(first_count)) = (starts.first_mut(), counts.first_mut()) {
            *first_start = start;
            *first_count = count;
        }
        Self::parse(spec, &starts, &counts)
    }

    pub(crate) fn spec(&self) -> RawReadSpec<'a> {
        self.spec
    }

    pub(crate) fn axes(&self) -> &[RegionAxis] {
        match &self.selection {
            Selection::Empty => &[],
            Selection::Elements { axes, .. } => axes,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        matches!(self.selection, Selection::Empty)
    }

    pub(crate) fn is_whole(&self) -> bool {
        !self.is_empty()
            && self
                .axes()
                .iter()
                .zip(&self.spec.dataspace().dimensions)
                .all(|(axis, &extent)| axis.start == 0 && axis.count == extent)
    }

    pub(crate) fn byte_len(&self) -> Result<usize, FormatError> {
        match self.selection {
            Selection::Empty => Ok(0),
            Selection::Elements { byte_len, .. } => byte_len.to_usize(),
        }
    }

    pub(crate) fn end_offset(&self) -> Result<u64, FormatError> {
        if self.is_empty() {
            return Ok(0);
        }
        self.axes()
            .iter()
            .try_fold(u64::from(self.spec.datatype().type_size()), |end, axis| {
                sum(end, product(axis.end - 1, axis.input_stride)?)
            })
    }

    pub(crate) fn overlaps(&self, offsets: &[u64], dimensions: &[u64]) -> bool {
        !self.is_empty()
            && offsets.len() >= self.axes().len()
            && dimensions.len() == self.axes().len()
            && self.axes().iter().zip(offsets).zip(dimensions).all(
                |((axis, &offset), &dimension)| {
                    offset < axis.end && offset.saturating_add(dimension) > axis.start
                },
            )
    }

    pub(crate) fn visit_runs(
        &self,
        mut visit: impl FnMut(u64, usize) -> Result<(), FormatError>,
    ) -> Result<(), FormatError> {
        if self.is_empty() {
            return Ok(());
        }
        let axes = self.axes();
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
            for (coordinate, axis) in coordinates.iter_mut().zip(axes).rev() {
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
        byte_len: u64,
    },
    Empty,
}

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
    #[case(&[3, 4], &[1], &[1, 1], FormatError::RegionRankMismatch { rank: 2, start: 1, count: 2 })]
    #[case(&[3, 4], &[1, 1], &[1], FormatError::RegionRankMismatch { rank: 2, start: 2, count: 1 })]
    #[case(&[3, 4], &[4, 1], &[0, 1], FormatError::RegionOutOfBounds { dimension: 0, start: 4, count: 0, extent: 3 })]
    #[case(&[3, 4], &[1, 3], &[0, 2], FormatError::RegionOutOfBounds { dimension: 1, start: 3, count: 2, extent: 4 })]
    #[case(&[3, 4], &[1, u64::MAX], &[0, 1], FormatError::RegionOutOfBounds { dimension: 1, start: u64::MAX, count: 1, extent: 4 })]
    #[case(&[u64::MAX, 2], &[0, 0], &[1, 1], FormatError::OffsetOverflow { offset: 2, length: u64::MAX })]
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
    #[case(&[3, 4], &[1, 1], &[2, 2], &[(5, 2), (9, 2)])]
    #[case(&[3, 4], &[1, 0], &[2, 4], &[(4, 8)])]
    #[case(&[3, 4, 5], &[1, 1, 2], &[2, 2, 2], &[(27, 2), (32, 2), (47, 2), (52, 2)])]
    #[case(&[u64::MAX, 1], &[u64::MAX - 2, 0], &[2, 1], &[(u64::MAX - 2, 2)])]
    #[case(&[4_294_967_424, 0], &[4_294_967_360, 0], &[1, 0], &[])]
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
