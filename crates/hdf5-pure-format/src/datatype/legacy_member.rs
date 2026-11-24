use alloc::boxed::Box;
use core::num::NonZeroU32;

use byteorder::ByteOrder;
use byteorder::LittleEndian;

use hdf5_pure_core::Datatype;
use hdf5_pure_core::FormatError;

use crate::bytes::ByteCursor;

/// A version 1 compound member's scalar shape or one to four nonzero array dimensions.
///
/// The shape is parsed from "The Datatype Message" of the [format specification, version
/// 4.0][spec]. The specification defines zero dimensionality for scalars and requires a zero
/// dimension permutation.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) struct MemberShape(Shape);

impl MemberShape {
    /// Reads the complete 28-byte version 1 dimension block before checking its fields.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the block is truncated,
    /// [`FormatError::InvalidCompoundV1Rank`] if the rank exceeds four,
    /// [`FormatError::UnsupportedCompoundV1Permutation`] if the permutation is nonzero, or
    /// [`FormatError::InvalidCompoundV1Extent`] if an active dimension is zero.
    pub(super) fn parse(cursor: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let block = cursor.take_array::<DIMENSION_BLOCK_SIZE>()?;
        let rank = block[0];
        if rank > MAX_RANK {
            return Err(FormatError::InvalidCompoundV1Rank(rank));
        }
        let permutation = LittleEndian::read_u32(&block[4..8]);
        if permutation != 0 {
            return Err(FormatError::UnsupportedCompoundV1Permutation(permutation));
        }
        let extent = |dimension: u8| {
            let start = DIMENSION_SIZES_OFFSET + usize::from(dimension) * 4;
            NonZeroU32::new(LittleEndian::read_u32(&block[start..start + 4]))
                .ok_or(FormatError::InvalidCompoundV1Extent { dimension })
        };
        Ok(Self(match rank {
            0 => Shape::Scalar,
            1 => Shape::One([extent(0)?]),
            2 => Shape::Two([extent(0)?, extent(1)?]),
            3 => Shape::Three([extent(0)?, extent(1)?, extent(2)?]),
            4 => Shape::Four([extent(0)?, extent(1)?, extent(2)?, extent(3)?]),
            _ => return Err(FormatError::InvalidCompoundV1Rank(rank)),
        }))
    }

    /// Multiplies the stored base size by each implicit array extent.
    ///
    /// Returns `base_size` for a scalar member.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::CompoundArraySizeOverflow`] if the product exceeds [`u32::MAX`].
    pub(super) fn stored_size(&self, base_size: NonZeroU32) -> Result<NonZeroU32, FormatError> {
        self.dimensions()
            .iter()
            .try_fold(base_size, |size, &extent| {
                size.checked_mul(extent)
                    .ok_or(FormatError::CompoundArraySizeOverflow {
                        size: size.get(),
                        extent: extent.get(),
                    })
            })
    }

    /// Returns [`Datatype::Array`] for an array member or `base_type` for a scalar member.
    pub(super) fn into_datatype(self, base_type: Datatype) -> Datatype {
        let dimensions = self.dimensions();
        if dimensions.is_empty() {
            base_type
        } else {
            Datatype::Array {
                base_type: Box::new(base_type),
                dimensions: dimensions.iter().map(|extent| extent.get()).collect(),
            }
        }
    }

    fn dimensions(&self) -> &[NonZeroU32] {
        match &self.0 {
            Shape::Scalar => &[],
            Shape::One(extents) => extents,
            Shape::Two(extents) => extents,
            Shape::Three(extents) => extents,
            Shape::Four(extents) => extents,
        }
    }
}

enum Shape {
    Four([NonZeroU32; 4]),
    One([NonZeroU32; 1]),
    Scalar,
    Three([NonZeroU32; 3]),
    Two([NonZeroU32; 2]),
}

/// The dimension block's byte length in "The Datatype Message" of the [format specification,
/// version 4.0][spec] for a version 1 compound member.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const DIMENSION_BLOCK_SIZE: usize = 28;
/// The dimension sizes' byte offset within the dimension block.
const DIMENSION_SIZES_OFFSET: usize = 12;
/// The four dimension slots available to a version 1 compound member.
const MAX_RANK: u8 = 4;
