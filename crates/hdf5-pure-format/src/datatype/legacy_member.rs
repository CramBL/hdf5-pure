use alloc::boxed::Box;
use alloc::string::ToString;
use core::num::NonZeroU32;

use byteorder::ByteOrder;
use byteorder::LittleEndian;

use hdf5_pure_core::Datatype;
use hdf5_pure_core::FormatError;

use crate::bytes;

/// A version 1 compound member's scalar shape or one to four nonzero array dimensions.
///
/// The shape is parsed from "The Datatype Message" of the [format specification, version
/// 4.0][spec]. The specification defines zero dimensionality for scalars and requires a zero
/// dimension permutation.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) struct MemberShape(Shape);

impl MemberShape {
    /// Parses the dimension block at `pos` and returns the shape and number of bytes read.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the block is truncated,
    /// [`FormatError::InvalidCompoundV1Rank`] if the rank exceeds four,
    /// [`FormatError::UnsupportedCompoundV1Permutation`] if the permutation is nonzero, or
    /// [`FormatError::InvalidCompoundV1Extent`] if an active dimension is zero.
    pub(super) fn parse(data: &[u8], pos: usize) -> Result<(Self, usize), FormatError> {
        bytes::ensure_len(data, pos, DIMENSION_BLOCK_SIZE)?;
        let block = &data[pos..pos + DIMENSION_BLOCK_SIZE];
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
        Ok((
            Self(match rank {
                0 => Shape::Scalar,
                1 => Shape::One([extent(0)?]),
                2 => Shape::Two([extent(0)?, extent(1)?]),
                3 => Shape::Three([extent(0)?, extent(1)?, extent(2)?]),
                4 => Shape::Four([extent(0)?, extent(1)?, extent(2)?, extent(3)?]),
                _ => return Err(FormatError::InvalidCompoundV1Rank(rank)),
            }),
            DIMENSION_BLOCK_SIZE,
        ))
    }

    /// Applies the shape to `base_type` within the declared compound byte range.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::CompoundArraySizeOverflow`] if an array size exceeds [`u32::MAX`],
    /// or [`FormatError::CompoundFieldOutOfBounds`] if the member extends past `compound_size`.
    pub(super) fn into_datatype(
        self,
        base_type: Datatype,
        name: &str,
        offset: u64,
        compound_size: u32,
    ) -> Result<Datatype, FormatError> {
        let dimensions = match &self.0 {
            Shape::Scalar => None,
            Shape::One(extents) => Some(extents.as_slice()),
            Shape::Two(extents) => Some(extents.as_slice()),
            Shape::Three(extents) => Some(extents.as_slice()),
            Shape::Four(extents) => Some(extents.as_slice()),
        };
        let size = dimensions
            .unwrap_or_default()
            .iter()
            .try_fold(checked_size(&base_type)?, |size, extent| {
                multiply_size(size, extent.get())
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

fn checked_size(datatype: &Datatype) -> Result<u32, FormatError> {
    match datatype {
        Datatype::Array {
            base_type,
            dimensions,
        } => dimensions
            .iter()
            .try_fold(checked_size(base_type)?, |size, &extent| {
                multiply_size(size, extent)
            }),
        _ => Ok(datatype.type_size()),
    }
}

fn multiply_size(size: u32, extent: u32) -> Result<u32, FormatError> {
    size.checked_mul(extent)
        .ok_or(FormatError::CompoundArraySizeOverflow { size, extent })
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
