//! Chunk dimensions and byte sizes interpreted from dataset metadata.
//!
//! Spatial chunk dimensions and the element-size dimension are defined in "The Data Layout
//! Message" of the [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout

use core::num::{NonZeroU32, NonZeroUsize};

use crate::convert::Narrow;
use crate::data_layout::DataLayout;
use crate::dataspace::{Dataspace, DataspaceType};
use crate::datatype;
use crate::width::OffsetWidth;
use crate::{Datatype, FormatError, MaxExtent};

/// The chunk dimensions and byte sizes derived from dataset metadata.
#[derive(Clone, Copy, Debug)]
pub struct ChunkedGeometry<'a> {
    /// The chunk dimensions in elements, one per dataspace dimension.
    pub spatial: &'a [u64],
    /// The nonzero stored element size in bytes, representable as a [`usize`].
    pub element_size: NonZeroUsize,
    /// The uncompressed chunk size in bytes, including any edge overhang.
    pub raw_size: u64,
    /// The maximum dimensions, if at least one maximum exceeds its current dimension.
    pub maxshape: Option<&'a [MaxExtent]>,
}

impl<'a> ChunkedGeometry<'a> {
    /// Borrows the spatial and maximum dimensions and checks the uncompressed chunk size.
    ///
    /// Assumes eight-byte file addresses when sizing variable-length elements. The chunk byte
    /// count is a [`u64`], even when it exceeds the platform's [`usize`] range.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidChunkGeometry`] if the layout is not chunked, the
    /// dataspace is not simple with a consistent nonzero rank, the chunk dimensions have the
    /// wrong rank or include zero, the layout's element size differs from the datatype's,
    /// the maximum dimensions are invalid, or the chunk byte count exceeds [`u64::MAX`].
    /// Returns [`FormatError::ZeroSizedDatatype`] if the datatype occupies zero bytes per
    /// element, or [`FormatError::ValueTooLargeForPlatform`] if its element size exceeds
    /// [`usize::MAX`]. Propagates stored-element-size errors from
    /// [`datatype::stored_element_size`].
    pub fn new(
        datatype: &Datatype,
        dataspace: &'a Dataspace,
        layout: &'a DataLayout,
    ) -> Result<Self, FormatError> {
        Self::with_offset_width(OffsetWidth::Eight, datatype, dataspace, layout)
    }

    /// Borrows chunk dimensions and computes byte sizes from the stored element representation.
    ///
    /// Sizes variable-length references and arrays of them at `offsets`. The layout's final
    /// chunk dimension must equal this stored size. Borrows the spatial dimensions and any
    /// extensible maximum dimensions from the supplied metadata.
    ///
    /// # Errors
    ///
    /// Returns the geometry and element-size errors of [`Self::new`], using `offsets` for the
    /// stored element size.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_format::{__private as format, Datatype, FormatError, OffsetWidth,
    ///     VariableLengthType};
    ///
    /// let datatype = Datatype::VariableLength(VariableLengthType::sequence(Datatype::Opaque {
    ///     size: 1,
    ///     tag: vec![],
    /// }));
    /// let dataspace = format::Dataspace::simple(vec![8], None)?;
    /// let layout = format::DataLayout::Chunked {
    ///     flags: format::ChunkedLayoutFlags::NONE,
    ///     chunk_dimensions: vec![4, 10],
    ///     index: format::ChunkIndexLayout::FixedArray { address: None },
    /// };
    /// let geometry = format::ChunkedGeometry::with_offset_width(
    ///     OffsetWidth::Two, &datatype, &dataspace, &layout,
    /// )?;
    /// assert_eq!(datatype.type_size(), 16);
    /// assert_eq!(geometry.element_size.get(), 10);
    /// assert_eq!(geometry.raw_size, 40);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn with_offset_width(
        offsets: OffsetWidth,
        datatype: &Datatype,
        dataspace: &'a Dataspace,
        layout: &'a DataLayout,
    ) -> Result<Self, FormatError> {
        Self::parse(dataspace, layout, || {
            datatype::stored_element_size_usize(datatype, offsets)
        })
    }

    /// Checks the chunk dimensions against the stored size of one datatype element.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidChunkGeometry`] if the dataspace or layout has invalid
    /// chunk geometry, including a final chunk dimension that differs from `element_size`.
    /// Returns [`FormatError::ValueTooLargeForPlatform`] if the element size exceeds [`usize::MAX`].
    pub fn with_element_size(
        element_size: NonZeroU32,
        dataspace: &'a Dataspace,
        layout: &'a DataLayout,
    ) -> Result<Self, FormatError> {
        Self::parse(dataspace, layout, || element_size.narrow())
    }

    fn parse(
        dataspace: &'a Dataspace,
        layout: &'a DataLayout,
        element_size: impl FnOnce() -> Result<NonZeroUsize, FormatError>,
    ) -> Result<Self, FormatError> {
        let DataLayout::Chunked {
            chunk_dimensions, ..
        } = layout
        else {
            return Err(FormatError::InvalidChunkGeometry("dataset is not chunked"));
        };
        let rank = dataspace.dimensions().len();
        if dataspace.space_type() != DataspaceType::Simple {
            return Err(FormatError::InvalidChunkGeometry(
                "chunking requires a simple dataspace of nonzero rank",
            ));
        }
        let Some((stored_element_size, spatial)) = chunk_dimensions.split_last() else {
            return Err(FormatError::InvalidChunkGeometry(
                "chunked layout has malformed dimensions",
            ));
        };
        if spatial.len() != rank {
            return Err(FormatError::InvalidChunkGeometry(
                "chunked layout has malformed dimensions",
            ));
        }
        if spatial.contains(&0) {
            return Err(FormatError::InvalidChunkGeometry(
                "chunk dimensions must be nonzero",
            ));
        }
        let element_size = element_size()?;
        if *stored_element_size != element_size.get() as u64 {
            return Err(FormatError::InvalidChunkGeometry(
                "chunk element size differs from the datatype size",
            ));
        }
        let raw_size = spatial
            .iter()
            .try_fold(element_size.get() as u64, |size, &dimension| {
                size.checked_mul(dimension)
                    .ok_or(FormatError::InvalidChunkGeometry(
                        "chunk logical byte size exceeds the addressable range",
                    ))
            })?;
        Ok(Self {
            spatial,
            element_size,
            raw_size,
            maxshape: dataspace.extensible_max_dimensions(),
        })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use rstest::rstest;
    use test_util::datatype::Class;

    use super::*;
    use crate::VariableLengthType;
    use crate::data_layout::{ChunkIndexLayout, ChunkedLayoutFlags};

    #[rstest]
    #[case::fixed(None, None)]
    #[case::fixed_at_current(Some(vec![MaxExtent::Fixed(6), MaxExtent::Fixed(8)]), None)]
    #[case::finite_maximum(
        Some(vec![MaxExtent::Fixed(12), MaxExtent::Fixed(8)]),
        Some(vec![MaxExtent::Fixed(12), MaxExtent::Fixed(8)]),
    )]
    #[case::unlimited(
        Some(vec![MaxExtent::Unlimited, MaxExtent::Fixed(8)]),
        Some(vec![MaxExtent::Unlimited, MaxExtent::Fixed(8)]),
    )]
    fn geometry_borrows_spatial_and_extensible_maximum_dimensions(
        #[case] maximum: Option<Vec<MaxExtent>>,
        #[case] expected: Option<Vec<MaxExtent>>,
    ) {
        let dataspace = Dataspace::simple(vec![6, 8], maximum).unwrap();
        let layout = layout(vec![3, 4, 8]);
        let geometry = ChunkedGeometry::new(&datatype(8), &dataspace, &layout).unwrap();
        assert_eq!(geometry.spatial, &[3, 4]);
        assert_eq!(geometry.element_size.get(), 8);
        assert_eq!(geometry.raw_size, 96);
        assert_eq!(geometry.maxshape, expected.as_deref());
        let DataLayout::Chunked {
            chunk_dimensions, ..
        } = &layout
        else {
            panic!("expected a chunked layout");
        };
        assert_eq!(geometry.spatial.as_ptr(), chunk_dimensions.as_ptr());
        assert_eq!(
            geometry.maxshape.map(<[MaxExtent]>::as_ptr),
            expected
                .as_ref()
                .map(|_| dataspace.max_dimensions().unwrap().as_ptr()),
        );
    }

    #[rstest]
    #[case::dimension_product(vec![u64::MAX, 2, 1], 1)]
    #[case::byte_count(vec![u64::MAX / 2 + 1, 1, 2], 2)]
    fn overflowing_chunk_byte_counts_are_errors(#[case] dimensions: Vec<u64>, #[case] size: u32) {
        let err =
            ChunkedGeometry::new(&datatype(size), &dataspace(), &layout(dimensions)).unwrap_err();
        assert_eq!(
            err,
            FormatError::InvalidChunkGeometry(
                "chunk logical byte size exceeds the addressable range",
            ),
        );
    }

    #[rstest]
    #[case::no_dimensions(vec![])]
    #[case::missing_element_dimension(vec![2, 8])]
    #[case::extra_dimensions(vec![2, 3, 4, 8])]
    fn mismatched_chunk_dimension_ranks_are_errors(#[case] dimensions: Vec<u64>) {
        let err =
            ChunkedGeometry::new(&datatype(8), &dataspace(), &layout(dimensions)).unwrap_err();
        assert_eq!(
            err,
            FormatError::InvalidChunkGeometry("chunked layout has malformed dimensions"),
        );
    }

    #[rstest]
    #[case::zero_dimension(vec![0, 4, 8], 8, "chunk dimensions must be nonzero")]
    #[case::different_element_size(
        vec![2, 4, 4], 8, "chunk element size differs from the datatype size",
    )]
    fn invalid_chunk_dimensions_are_errors(
        #[case] dimensions: Vec<u64>,
        #[case] size: u32,
        #[case] reason: &'static str,
    ) {
        let err =
            ChunkedGeometry::new(&datatype(size), &dataspace(), &layout(dimensions)).unwrap_err();
        assert_eq!(err, FormatError::InvalidChunkGeometry(reason));
    }

    #[rstest]
    #[case::scalar(DataspaceType::Scalar)]
    #[case::null(DataspaceType::Null)]
    fn non_simple_dataspaces_are_errors(#[case] space_type: DataspaceType) {
        let dataspace = Dataspace::new(space_type, vec![], None).unwrap();
        let err =
            ChunkedGeometry::new(&datatype(8), &dataspace, &layout(vec![2, 4, 8])).unwrap_err();
        assert_eq!(
            err,
            FormatError::InvalidChunkGeometry(
                "chunking requires a simple dataspace of nonzero rank",
            )
        );
    }

    #[test]
    fn non_chunked_layouts_are_errors() {
        let err =
            ChunkedGeometry::new(&datatype(8), &dataspace(), &DataLayout::Virtual).unwrap_err();
        assert_eq!(
            err,
            FormatError::InvalidChunkGeometry("dataset is not chunked")
        );
    }

    #[test]
    fn zero_element_sizes_are_errors() {
        let err =
            ChunkedGeometry::new(&datatype(0), &dataspace(), &layout(vec![2, 4, 0])).unwrap_err();
        assert_eq!(
            err,
            FormatError::ZeroSizedDatatype {
                class: Class::OPAQUE.0
            }
        );
    }

    #[test]
    fn chunk_byte_counts_can_exceed_u32_max() {
        let dataspace = dataspace();
        let layout = layout(vec![u64::from(u32::MAX), 1, 8]);
        let geometry = ChunkedGeometry::new(&datatype(8), &dataspace, &layout).unwrap();
        assert_eq!(geometry.raw_size, u64::from(u32::MAX) * 8);
    }

    #[rstest]
    #[case::two(OffsetWidth::Two, 10)]
    #[case::four(OffsetWidth::Four, 12)]
    #[case::eight(OffsetWidth::Eight, 16)]
    fn variable_length_geometry_uses_stored_element_width(
        #[case] offsets: OffsetWidth,
        #[case] stored_size: u64,
    ) {
        let datatype = Datatype::VariableLength(VariableLengthType::sequence(datatype(1)));
        let dataspace = dataspace();
        let layout = layout(vec![2, 4, stored_size]);
        let geometry =
            ChunkedGeometry::with_offset_width(offsets, &datatype, &dataspace, &layout).unwrap();
        assert_eq!(geometry.element_size.get() as u64, stored_size);
        assert_eq!(geometry.raw_size, 8 * stored_size);
        assert_eq!(datatype.type_size(), 16);
    }

    #[rstest]
    #[case::two(OffsetWidth::Two)]
    #[case::four(OffsetWidth::Four)]
    fn narrow_variable_length_geometry_rejects_canonical_element_width(
        #[case] offsets: OffsetWidth,
    ) {
        let datatype = Datatype::VariableLength(VariableLengthType::sequence(datatype(1)));
        let error = ChunkedGeometry::with_offset_width(
            offsets,
            &datatype,
            &dataspace(),
            &layout(vec![2, 4, 16]),
        )
        .unwrap_err();
        assert_eq!(
            error,
            FormatError::InvalidChunkGeometry("chunk element size differs from the datatype size")
        );
    }

    fn datatype(size: u32) -> Datatype {
        Datatype::Opaque { size, tag: vec![] }
    }

    fn dataspace() -> Dataspace {
        Dataspace::simple(vec![6, 8], None).unwrap()
    }

    fn layout(chunk_dimensions: Vec<u64>) -> DataLayout {
        DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions,
            index: ChunkIndexLayout::FixedArray { address: None },
        }
    }
}
