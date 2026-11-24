use alloc::boxed::Box;
use alloc::string::ToString;
use core::num::NonZeroU32;

use byteorder::ByteOrder;
use byteorder::LittleEndian;

use hdf5_pure_core::Datatype;
use hdf5_pure_core::FormatError;

use crate::bytes::ByteCursor;
use crate::datatype::message::decode::Decoded;

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

    /// Checks the member's stored byte range and applies any implicit array shape.
    ///
    /// Computes an implicit array's size from the decoded base size and the member's extents.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::CompoundArraySizeOverflow`] if an implicit array size exceeds
    /// [`u32::MAX`], or [`FormatError::CompoundFieldOutOfBounds`] if the stored member range
    /// extends past `compound_size`.
    pub(super) fn into_datatype(
        self,
        base: Decoded<Datatype>,
        name: &str,
        offset: u64,
        compound_size: u32,
    ) -> Result<Datatype, FormatError> {
        let Decoded {
            datatype: base_type,
            stored_size,
        } = base;
        let dimensions = match &self.0 {
            Shape::Scalar => None,
            Shape::One(extents) => Some(extents.as_slice()),
            Shape::Two(extents) => Some(extents.as_slice()),
            Shape::Three(extents) => Some(extents.as_slice()),
            Shape::Four(extents) => Some(extents.as_slice()),
        };
        let size =
            dimensions
                .unwrap_or_default()
                .iter()
                .try_fold(stored_size.get(), |size, extent| {
                    let extent = extent.get();
                    size.checked_mul(extent)
                        .ok_or(FormatError::CompoundArraySizeOverflow { size, extent })
                })?;
        if offset
            .checked_add(u64::from(size))
            .is_none_or(|end| end > u64::from(compound_size))
        {
            return Err(FormatError::CompoundFieldOutOfBounds {
                name: name.to_string(),
                offset,
                field_size: size,
                compound_size,
            });
        }
        Ok(match dimensions {
            Some(extents) => Datatype::Array {
                base_type: Box::new(base_type),
                dimensions: extents.iter().map(|extent| extent.get()).collect(),
            },
            None => base_type,
        })
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
