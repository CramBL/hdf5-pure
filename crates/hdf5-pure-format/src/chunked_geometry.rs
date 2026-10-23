//! Chunk dimensions and byte sizes interpreted from dataset metadata.
//!
//! Spatial chunk dimensions and the element-size dimension are defined in "The Data Layout
//! Message" of the [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout

use core::num::NonZeroUsize;

use crate::data_layout::DataLayout;
use crate::dataspace::{Dataspace, DataspaceType, Extent};
use crate::datatype;
use crate::{Datatype, FormatError, MaxExtent};

/// The chunk dimensions and byte sizes derived from dataset metadata.
#[derive(Clone, Copy, Debug)]
pub struct ChunkedGeometry<'a> {
    /// The chunk dimensions in elements, one per dataspace dimension.
    pub spatial: &'a [u64],
    /// The nonzero element size in bytes, representable as a [`usize`].
    pub element_size: NonZeroUsize,
    /// The uncompressed chunk size in bytes, including any edge overhang.
    pub raw_size: u64,
    /// The maximum dimensions, if at least one maximum exceeds its current dimension.
    pub maxshape: Option<&'a [MaxExtent]>,
}

impl<'a> ChunkedGeometry<'a> {
    /// Borrows the spatial and maximum dimensions and checks the uncompressed chunk size.
    ///
    /// The chunk byte count is a [`u64`], even when it exceeds the platform's [`usize`] range.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidChunkGeometry`] if the layout is not chunked, the
    /// dataspace is not simple with a consistent nonzero rank, the chunk dimensions have the
    /// wrong rank or include zero, the layout's element size differs from the datatype's,
    /// the maximum dimensions are invalid, or the chunk byte count exceeds [`u64::MAX`].
    /// Returns [`FormatError::ZeroSizedDatatype`] if the datatype occupies zero bytes per
    /// element, or [`FormatError::ValueTooLargeForPlatform`] if its element size exceeds
    /// [`usize::MAX`].
    pub fn new(
        datatype: &Datatype,
        dataspace: &'a Dataspace,
        layout: &'a DataLayout,
    ) -> Result<Self, FormatError> {
        let DataLayout::Chunked {
            chunk_dimensions, ..
        } = layout
        else {
            return Err(FormatError::InvalidChunkGeometry("dataset is not chunked"));
        };
        let rank = dataspace.dimensions.len();
        if dataspace.space_type != DataspaceType::Simple
            || rank == 0
            || rank != usize::from(dataspace.rank)
        {
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
        let element_size = datatype::element_size_usize(datatype)?;
        if *stored_element_size != u64::from(datatype.type_size()) {
            return Err(FormatError::InvalidChunkGeometry(
                "chunk element size differs from the datatype size",
            ));
        }
        Extent::new(&dataspace.dimensions, dataspace.max_dimensions.as_deref())
            .map_err(FormatError::InvalidChunkGeometry)?;
        let raw_size =
            spatial
                .iter()
                .try_fold(u64::from(datatype.type_size()), |size, &dimension| {
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
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: vec![6, 8],
            max_dimensions: maximum,
        };
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
                .map(|_| dataspace.max_dimensions.as_ref().unwrap().as_ptr()),
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
    #[case::scalar(DataspaceType::Scalar, 0, vec![])]
    #[case::null(DataspaceType::Null, 0, vec![])]
    #[case::empty_simple(DataspaceType::Simple, 0, vec![])]
    #[case::rank_mismatch(DataspaceType::Simple, 1, vec![6, 8])]
    fn invalid_dataspaces_are_errors(
        #[case] space_type: DataspaceType,
        #[case] rank: u8,
        #[case] dimensions: Vec<u64>,
    ) {
        let dataspace = Dataspace {
            space_type,
            rank,
            dimensions,
            max_dimensions: None,
        };
        let err =
            ChunkedGeometry::new(&datatype(8), &dataspace, &layout(vec![2, 4, 8])).unwrap_err();
        assert_eq!(
            err,
            FormatError::InvalidChunkGeometry(
                "chunking requires a simple dataspace of nonzero rank",
            )
        );
    }

    #[rstest]
    #[case::rank(vec![MaxExtent::Fixed(6)], "maxshape must have the same rank as the dataset shape")]
    #[case::too_small(
        vec![MaxExtent::Fixed(5), MaxExtent::Fixed(8)],
        "maxshape must be at least the current shape in every dimension",
    )]
    fn invalid_maximum_dimensions_are_errors(
        #[case] maximum: Vec<MaxExtent>,
        #[case] reason: &'static str,
    ) {
        let mut dataspace = dataspace();
        dataspace.max_dimensions = Some(maximum);
        let err =
            ChunkedGeometry::new(&datatype(8), &dataspace, &layout(vec![2, 4, 8])).unwrap_err();
        assert_eq!(err, FormatError::InvalidChunkGeometry(reason));
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

    fn datatype(size: u32) -> Datatype {
        Datatype::Opaque { size, tag: vec![] }
    }

    fn dataspace() -> Dataspace {
        Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: vec![6, 8],
            max_dimensions: None,
        }
    }

    fn layout(chunk_dimensions: Vec<u64>) -> DataLayout {
        DataLayout::Chunked {
            flags: ChunkedLayoutFlags::NONE,
            chunk_dimensions,
            index: ChunkIndexLayout::FixedArray { address: None },
        }
    }
}
