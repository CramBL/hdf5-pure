use crate::FormatError;

/// The recorded skip bits for a chunk's filter pipeline.
///
/// Bit `i` is set when the filter at pipeline position `i` was skipped. Bits outside the
/// pipeline are preserved. The recorded mask alone does not indicate whether the chunk
/// bypasses filtering because of its layout's partial-edge policy.
///
/// The positional encoding is defined in "The Extensible Array Index" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct FilterMask(u32);

impl FilterMask {
    /// Returns the recorded bits, including bits outside the filter pipeline.
    pub const fn raw_bits(self) -> u32 {
        self.0
    }

    /// A recorded mask with every skip bit clear.
    pub const NONE: Self = Self(0);
}

/// A recorded mask paired with a checked filter count of at most 32.
///
/// This type checks positional bounds. The caller associates the mask with the ordered
/// pipeline. Bits beyond the count remain available through [`Self::uninterpreted_bits`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PipelineFilterMask {
    mask: FilterMask,
    count: usize,
}

impl PipelineFilterMask {
    /// Reads recorded bits for a pipeline of `count` filters.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFilterPipelineField`] if `count` exceeds 32.
    pub fn from_raw_bits(bits: u32, count: usize) -> Result<Self, FormatError> {
        if count > FILTER_COUNT_MAX {
            return Err(FormatError::InvalidFilterPipelineField {
                field: "filter count",
                value: count,
                maximum: FILTER_COUNT_MAX,
            });
        }
        Ok(Self {
            mask: FilterMask(bits),
            count,
        })
    }

    /// Sets the skip bit for each of `count` pipeline positions.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFilterPipelineField`] if `count` exceeds 32.
    pub fn all_skipped(count: usize) -> Result<Self, FormatError> {
        let mut mask = Self::from_raw_bits(0, count)?;
        mask.mask = FilterMask(mask.position_bits());
        Ok(mask)
    }

    /// Returns the complete recorded mask.
    pub const fn recorded_mask(self) -> FilterMask {
        self.mask
    }

    /// Returns bits outside the pipeline's positions.
    // `H5Z_pipeline` in `src/H5Z.c`, HDF5 1.14.6, reads only the pipeline's indexed bits.
    pub const fn uninterpreted_bits(self) -> u32 {
        self.mask.0 & !self.position_bits()
    }

    /// Iterates over pipeline positions in application order, with `true` for a skipped filter.
    pub fn positions(
        self,
    ) -> impl Clone + DoubleEndedIterator<Item = (usize, bool)> + ExactSizeIterator {
        (0..self.count).map(move |index| (index, self.mask.0 & (1 << index) != 0))
    }

    /// Returns whether the filter at `index` was skipped, or `None` outside the pipeline.
    pub fn is_skipped(self, index: usize) -> Option<bool> {
        if index < self.count {
            Some(self.mask.0 & (1 << index) != 0)
        } else {
            None
        }
    }

    const fn position_bits(self) -> u32 {
        if self.count == FILTER_COUNT_MAX {
            u32::MAX
        } else {
            (1 << self.count) - 1
        }
    }
}

/// The maximum number of filters in a pipeline.
///
/// Defined in "The Data Storage - Filter Pipeline Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_filter
pub const FILTER_COUNT_MAX: usize = u32::BITS as usize;

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::empty(0, 0)]
    #[case::one(1, 0b1)]
    #[case::two(2, 0b11)]
    #[case::full(32, u32::MAX)]
    fn every_position_can_be_skipped(#[case] count: usize, #[case] bits: u32) {
        let mask = PipelineFilterMask::all_skipped(count).unwrap();
        assert_eq!(mask.recorded_mask().raw_bits(), bits);
        assert_eq!(mask.positions().len(), count);
        assert!(mask.positions().all(|(_, skipped)| skipped));
        assert_eq!(mask.is_skipped(count), None);
        assert_eq!(mask.is_skipped(usize::MAX), None);
        assert_eq!(mask.uninterpreted_bits(), 0);
    }

    #[rstest]
    #[case::empty(u32::MAX, 0, u32::MAX, vec![])]
    #[case::first(0b1, 1, 0, vec![(0, true)])]
    #[case::second(0b10, 2, 0, vec![(0, false), (1, true)])]
    #[case::high(0x8000_0002, 2, 0x8000_0000, vec![(0, false), (1, true)])]
    fn recorded_bits_are_interpreted_only_at_existing_positions(
        #[case] bits: u32,
        #[case] count: usize,
        #[case] uninterpreted: u32,
        #[case] positions: Vec<(usize, bool)>,
    ) {
        let mask = PipelineFilterMask::from_raw_bits(bits, count).unwrap();
        assert_eq!(mask.recorded_mask().raw_bits(), bits);
        assert_eq!(mask.uninterpreted_bits(), uninterpreted);
        assert_eq!(mask.positions().collect::<Vec<_>>(), positions);
        for (index, skipped) in positions {
            assert_eq!(mask.is_skipped(index), Some(skipped));
        }
        assert_eq!(mask.is_skipped(count), None);
        assert_eq!(mask.is_skipped(32), None);
    }

    #[rstest]
    #[case(33)]
    #[case(usize::MAX)]
    fn a_pipeline_larger_than_the_mask_is_rejected(#[case] count: usize) {
        let expected = FormatError::InvalidFilterPipelineField {
            field: "filter count",
            value: count,
            maximum: 32,
        };
        assert_eq!(
            PipelineFilterMask::from_raw_bits(0, count),
            Err(expected.clone())
        );
        assert_eq!(PipelineFilterMask::all_skipped(count), Err(expected));
    }
}
