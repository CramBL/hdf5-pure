//! Dataset storage, shape, filters, fill values, and stored element sizes for raw reads.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::num::NonZeroU32;
use core::num::NonZeroUsize;

use crate::address::StoredAddress;
#[cfg(test)]
use crate::address::StoredAddressExt;
use crate::convert::Narrow;
use crate::data_layout::{ChunkIndexLayout, ChunkedLayoutFlags, DataLayout};
use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::error::FormatError;
use crate::fill_value::FillPattern;
use crate::filter_pipeline::FilterPipeline;
use crate::filters::ChunkContext;
use crate::width::OffsetWidth;

/// The storage and element geometry of a raw dataset read.
///
/// Construction derives the stored element size from the datatype and selected offset width.
/// The reader resolves file addresses with its file context.
#[derive(Clone, Copy)]
pub(crate) struct RawReadSpec<'a> {
    /// Storage properties after dataset-level consistency has been established.
    storage: RawReadStorage<'a>,
    /// The dataset's shape, which fixes how many elements the read produces.
    dataspace: &'a Dataspace,
    /// The datatype used to interpret stored elements and select filter scalar types.
    datatype: &'a Datatype,
    /// The filter pipeline stored bytes passed through, if any.
    pipeline: Option<&'a FilterPipeline>,
    /// What storage that was never allocated reads as.
    fill: FillPattern<'a>,
    /// Logical dataset byte length derived from the dataspace and datatype.
    byte_len: DatasetByteLen,
    /// Stored bytes per element at the selected file offset width.
    element_size: u32,
}

impl<'a> RawReadSpec<'a> {
    /// Parses dataset properties with eight-byte heap addresses.
    ///
    /// For allocated compact and contiguous storage, construction compares the
    /// storage extent with the byte length implied by the dataspace and datatype.
    /// [`RawReadStorage`] has no field for a contiguous extent, so downstream
    /// readers cannot use an unchecked extent after construction succeeds.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::OffsetOverflow`] if multiplying the element count by
    /// the datatype size overflows, or [`FormatError::LogicalDataSizeMismatch`] if
    /// a nonempty dataset's compact or allocated contiguous storage differs from
    /// its logical byte length. Returns the errors of [`Self::parse_with_offset_width`] for an
    /// array size that exceeds its encoded field.
    pub(crate) fn parse(
        layout: &'a DataLayout,
        dataspace: &'a Dataspace,
        datatype: &'a Datatype,
        pipeline: Option<&'a FilterPipeline>,
        fill: FillPattern<'a>,
    ) -> Result<Self, FormatError> {
        Self::parse_with_offset_width(
            layout,
            dataspace,
            datatype,
            pipeline,
            fill,
            OffsetWidth::Eight,
        )
    }

    /// Parses stored element geometry and allocated storage extents at `offset_width`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::LengthTooLarge`] if an array size exceeds its four-byte field,
    /// [`FormatError::OffsetOverflow`] if the logical dataset byte count exceeds [`u64::MAX`], or
    /// [`FormatError::LogicalDataSizeMismatch`] if nonempty compact or allocated contiguous
    /// storage differs from the logical byte count.
    pub(crate) fn parse_with_offset_width(
        layout: &'a DataLayout,
        dataspace: &'a Dataspace,
        datatype: &'a Datatype,
        pipeline: Option<&'a FilterPipeline>,
        fill: FillPattern<'a>,
        offset_width: OffsetWidth,
    ) -> Result<Self, FormatError> {
        let element_size = crate::datatype::stored_type_size(datatype, offset_width)?;
        let byte_len = DatasetByteLen::parse(dataspace, element_size)?;
        let storage = RawReadStorage::parse(layout, byte_len, dataspace.num_elements() != 0)?;
        Ok(Self {
            storage,
            dataspace,
            datatype,
            pipeline,
            fill,
            byte_len,
            element_size,
        })
    }

    /// An unchecked spec for tests that need to drive malformed metadata farther
    /// down the read stack. Production readers use [`Self::parse`] or
    /// [`Self::parse_with_offset_width`].
    #[cfg(test)]
    pub(crate) fn plain(
        layout: &'a DataLayout,
        dataspace: &'a Dataspace,
        datatype: &'a Datatype,
    ) -> Self {
        Self {
            storage: RawReadStorage::plain(layout),
            dataspace,
            datatype,
            pipeline: None,
            fill: FillPattern::ZERO,
            element_size: datatype.type_size(),
            byte_len: DatasetByteLen(
                dataspace
                    .num_elements()
                    .saturating_mul(u64::from(datatype.type_size())),
            ),
        }
    }

    /// Returns the stored element width established during construction.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ZeroSizedDatatype`] if the datatype occupies zero bytes.
    pub(crate) fn element_size(&self) -> Result<NonZeroU32, FormatError> {
        NonZeroU32::new(self.element_size).map_or_else(|| self.datatype.element_size(), Ok)
    }

    /// Returns the stored element width for a materialized host buffer.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::element_size`], or [`FormatError::ValueTooLargeForPlatform`]
    /// if the element width exceeds the target's [`usize`] range.
    pub(crate) fn element_size_usize(&self) -> Result<NonZeroUsize, FormatError> {
        self.element_size()?.narrow()
    }

    /// Combines the stored element width with datatype-specific filter parameters.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::element_size`] for a zero-sized datatype.
    pub(crate) fn chunk_context<'b>(
        &self,
        dimensions: &'b [u64],
    ) -> Result<ChunkContext<'b>, FormatError> {
        Ok(ChunkContext::from_stored_datatype(
            dimensions,
            self.datatype,
            self.element_size()?,
        ))
    }

    pub(crate) fn storage(&self) -> RawReadStorage<'a> {
        self.storage
    }

    pub(crate) fn dataspace(&self) -> &'a Dataspace {
        self.dataspace
    }

    pub(crate) fn pipeline(&self) -> Option<&'a FilterPipeline> {
        self.pipeline
    }

    pub(crate) fn fill(&self) -> FillPattern<'a> {
        self.fill
    }

    /// The logical dataset byte length, narrowed only when a concrete whole
    /// dataset buffer is about to be materialized.
    pub(crate) fn byte_len(&self) -> Result<usize, FormatError> {
        self.byte_len.to_usize()
    }

    pub(crate) fn byte_len_u64(&self) -> u64 {
        self.byte_len.0
    }

    /// Returns fill bytes for a dataset whose storage is unallocated.
    ///
    /// For a zero-element dataset, returns an empty vector without consulting the fill pattern.
    /// Checks the datatype element size and total buffer size before returning.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnreadableFillValue`] if the output is nonempty and the fill pattern
    /// is unreadable, [`FormatError::ZeroSizedDatatype`] if the datatype occupies zero bytes per
    /// element, and [`FormatError::ValueTooLargeForPlatform`] or [`FormatError::OffsetOverflow`]
    /// if the materialized dataset would be larger than this target can address.
    pub(crate) fn unallocated_buffer(&self) -> Result<Vec<u8>, FormatError> {
        let elem_size = self.element_size_usize()?;
        let total = self
            .dataspace
            .num_elements()
            .to_usize()?
            .checked_mul(elem_size.get())
            .ok_or(FormatError::OffsetOverflow {
                offset: self.dataspace.num_elements(),
                length: elem_size.get() as u64,
            })?;
        if total == 0 {
            Ok(Vec::new())
        } else {
            self.fill.buffer(total)
        }
    }
}

/// Storage facts that remain after [`RawReadSpec::parse`] establishes the
/// dataset-level extent invariant.
///
/// The raw contiguous `size` is deliberately absent. Construction returns an
/// allocated contiguous variant only when that size equals the dataset's logical
/// byte length. Downstream readers therefore have no unchecked storage extent.
#[derive(Clone, Copy)]
pub(crate) enum RawReadStorage<'a> {
    Compact {
        data: &'a [u8],
    },
    Contiguous {
        address: StoredAddress,
    },
    ContiguousUnallocated,
    Chunked {
        flags: ChunkedLayoutFlags,
        chunk_dimensions: &'a [u64],
        index: ChunkIndexLayout,
    },
    Virtual,
}

impl<'a> RawReadStorage<'a> {
    fn parse(
        layout: &'a DataLayout,
        byte_len: DatasetByteLen,
        has_elements: bool,
    ) -> Result<Self, FormatError> {
        Ok(match layout {
            DataLayout::Compact { data } => {
                if has_elements {
                    byte_len.require_storage_extent(data.len() as u64)?;
                }
                Self::Compact { data }
            }
            DataLayout::Contiguous {
                address: Some(address),
                size,
            } => {
                if has_elements {
                    byte_len.require_storage_extent(*size)?;
                }
                Self::Contiguous { address: *address }
            }
            DataLayout::Contiguous { address: None, .. } => Self::ContiguousUnallocated,
            DataLayout::Chunked {
                flags,
                chunk_dimensions,
                index,
            } => Self::Chunked {
                flags: *flags,
                chunk_dimensions,
                index: *index,
            },
            DataLayout::Virtual => Self::Virtual,
        })
    }

    #[cfg(test)]
    fn plain(layout: &'a DataLayout) -> Self {
        match layout {
            DataLayout::Compact { data } => Self::Compact { data },
            DataLayout::Contiguous {
                address: Some(address),
                ..
            } => Self::Contiguous { address: *address },
            DataLayout::Contiguous { address: None, .. } => Self::ContiguousUnallocated,
            DataLayout::Chunked {
                flags,
                chunk_dimensions,
                index,
            } => Self::Chunked {
                flags: *flags,
                chunk_dimensions,
                index: *index,
            },
            DataLayout::Virtual => Self::Virtual,
        }
    }
}

/// Logical bytes implied by a dataset's dataspace and datatype.
#[derive(Clone, Copy)]
struct DatasetByteLen(u64);

impl DatasetByteLen {
    fn parse(dataspace: &Dataspace, element_size: u32) -> Result<Self, FormatError> {
        let num_elements = dataspace.num_elements();
        let elem_size = u64::from(element_size);
        Ok(Self(num_elements.checked_mul(elem_size).ok_or(
            FormatError::OffsetOverflow {
                offset: num_elements,
                length: elem_size,
            },
        )?))
    }

    fn require_storage_extent(self, actual: u64) -> Result<(), FormatError> {
        if actual == self.0 {
            return Ok(());
        }
        Err(FormatError::LogicalDataSizeMismatch {
            expected: self.0,
            actual,
        })
    }

    fn to_usize(self) -> Result<usize, FormatError> {
        self.0.to_usize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataspace::DataspaceType;
    use crate::type_builders;

    fn four_i32s() -> Dataspace {
        Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![4],
            max_dimensions: None,
        }
    }

    #[test]
    fn compact_extent_mismatch_is_rejected_while_parsing_the_read_spec() {
        let layout = DataLayout::Compact { data: vec![0; 8] };
        let Err(err) = RawReadSpec::parse(
            &layout,
            &four_i32s(),
            &type_builders::make_i32_type(),
            None,
            FillPattern::ZERO,
        ) else {
            panic!("mismatched compact storage extent was accepted");
        };
        let FormatError::LogicalDataSizeMismatch { expected, actual } = err else {
            panic!("expected LogicalDataSizeMismatch, got {err:?}");
        };
        assert_eq!(expected, 16);
        assert_eq!(actual, 8);
    }

    #[test]
    fn contiguous_extent_mismatch_is_rejected_while_parsing_the_read_spec() {
        let layout = DataLayout::Contiguous {
            address: Some(StoredAddress::new(128)),
            size: 32,
        };
        let Err(err) = RawReadSpec::parse(
            &layout,
            &four_i32s(),
            &type_builders::make_i32_type(),
            None,
            FillPattern::ZERO,
        ) else {
            panic!("mismatched contiguous storage extent was accepted");
        };
        let FormatError::LogicalDataSizeMismatch { expected, actual } = err else {
            panic!("expected LogicalDataSizeMismatch, got {err:?}");
        };
        assert_eq!(expected, 16);
        assert_eq!(actual, 32);
    }

    #[rstest::rstest]
    #[case::larger_storage(4_294_967_312, 4_294_967_328)]
    #[case::smaller_storage(4_294_967_328, 4_294_967_312)]
    #[case::equal_storage(4_294_967_312, 4_294_967_312)]
    fn logical_storage_extents_preserve_full_width_counts(
        #[case] expected: u64,
        #[case] actual: u64,
    ) {
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 1,
            dimensions: vec![expected],
            max_dimensions: None,
        };
        let datatype = type_builders::make_u8_type();
        let layout = DataLayout::Contiguous {
            address: Some(StoredAddress::new(128)),
            size: actual,
        };
        let result = RawReadSpec::parse(&layout, &dataspace, &datatype, None, FillPattern::ZERO);
        if actual != expected {
            let Err(err) = result else {
                panic!("mismatched contiguous storage extent was accepted");
            };
            assert_eq!(
                err,
                FormatError::LogicalDataSizeMismatch { expected, actual }
            );
            return;
        }
        let spec = result.unwrap();
        assert_eq!(spec.byte_len_u64(), expected);
        #[cfg(target_pointer_width = "32")]
        assert_eq!(
            spec.byte_len(),
            Err(FormatError::ValueTooLargeForPlatform {
                value: expected,
                target: "usize"
            }),
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(spec.byte_len().unwrap() as u64, expected);
    }
}
