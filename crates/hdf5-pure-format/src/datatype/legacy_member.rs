use alloc::boxed::Box;
use core::num::NonZeroU32;

use byteorder::ByteOrder;
use byteorder::LittleEndian;

use hdf5_pure_core::Datatype;
use hdf5_pure_core::FormatError;

use crate::bytes::ByteCursor;

/// The dimension fields of a version 1 compound member.
///
/// Rank zero denotes a scalar. An array has one to four active extents, each nonzero.
/// All four extent slots are preserved, including the unused slots of a scalar or
/// an array with fewer than four dimensions.
///
/// The dimension block is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompoundV1Shape(Shape);

impl CompoundV1Shape {
    /// Constructs dimension fields from a rank and the four stored extent slots.
    ///
    /// Pass the active extents from slowest to fastest changing dimension.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidCompoundV1Rank`] if `rank` exceeds four, or
    /// [`FormatError::InvalidCompoundV1Extent`] if any of the first `rank` extents is zero.
    pub fn new(rank: u8, extents: [u32; 4]) -> Result<Self, FormatError> {
        let [first, second, third, fourth] = extents;
        let extent = |size, dimension| {
            NonZeroU32::new(size).ok_or(FormatError::InvalidCompoundV1Extent { dimension })
        };
        Ok(Self(match rank {
            0 => Shape::Scalar(extents),
            1 => Shape::One([extent(first, 0)?], [second, third, fourth]),
            2 => Shape::Two([extent(first, 0)?, extent(second, 1)?], [third, fourth]),
            3 => Shape::Three(
                [extent(first, 0)?, extent(second, 1)?, extent(third, 2)?],
                [fourth],
            ),
            4 => Shape::Four([
                extent(first, 0)?,
                extent(second, 1)?,
                extent(third, 2)?,
                extent(fourth, 3)?,
            ]),
            _ => return Err(FormatError::InvalidCompoundV1Rank(rank)),
        }))
    }

    /// Returns the number of active dimensions, or zero for a scalar.
    pub const fn rank(self) -> u8 {
        match self.0 {
            Shape::Scalar(_) => 0,
            Shape::One(..) => 1,
            Shape::Two(..) => 2,
            Shape::Three(..) => 3,
            Shape::Four(_) => 4,
        }
    }

    /// Returns all four extent slots in stored order.
    pub fn extents(self) -> [u32; 4] {
        let (dimensions, inactive) = self.extent_slices();
        let mut extents = [0; 4];
        let (active, rest) = extents.split_at_mut(dimensions.len());
        for (extent, dimension) in active.iter_mut().zip(dimensions) {
            *extent = dimension.get();
        }
        rest.copy_from_slice(inactive);
        extents
    }

    /// Reads the complete 28-byte version 1 dimension block before checking its fields.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the block is truncated,
    /// [`FormatError::InvalidCompoundV1Rank`] if the rank exceeds four,
    /// [`FormatError::InvalidCompoundV1ReservedByte`] if `reject_reserved` is set and a reserved
    /// byte is nonzero, [`FormatError::UnsupportedCompoundV1Permutation`] if the permutation
    /// is nonzero, or [`FormatError::InvalidCompoundV1Extent`] if an active extent is zero.
    pub(super) fn parse(
        cursor: &mut ByteCursor<'_>,
        reject_reserved: bool,
    ) -> Result<Self, FormatError> {
        let offset = cursor.position();
        let block = cursor.take_array::<DIMENSION_BLOCK_SIZE>()?;
        let rank = block[0];
        if rank > MAX_RANK {
            return Err(FormatError::InvalidCompoundV1Rank(rank));
        }
        let check_reserved = |indices| {
            if reject_reserved {
                for index in indices {
                    let byte = block[index];
                    if byte != 0 {
                        return Err(FormatError::InvalidCompoundV1ReservedByte {
                            offset: offset + index,
                            byte,
                        });
                    }
                }
            }
            Ok(())
        };
        check_reserved(RESERVED_BEFORE_PERMUTATION)?;
        let permutation = LittleEndian::read_u32(&block[PERMUTATION]);
        if permutation != 0 {
            return Err(FormatError::UnsupportedCompoundV1Permutation(permutation));
        }
        check_reserved(RESERVED_AFTER_PERMUTATION)?;
        let mut extents = [0; 4];
        LittleEndian::read_u32_into(&block[DIMENSION_SIZES_OFFSET..], &mut extents);
        Self::new(rank, extents)
    }

    /// Multiplies the stored base size by each implicit array extent.
    ///
    /// Returns `base_size` for a scalar member.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::CompoundArraySizeOverflow`] if the product exceeds [`u32::MAX`].
    pub(super) fn stored_size(&self, base_size: NonZeroU32) -> Result<NonZeroU32, FormatError> {
        self.extent_slices()
            .0
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
        let (dimensions, _) = self.extent_slices();
        if dimensions.is_empty() {
            base_type
        } else {
            Datatype::Array {
                base_type: Box::new(base_type),
                dimensions: dimensions.iter().map(|extent| extent.get()).collect(),
            }
        }
    }

    fn extent_slices(&self) -> (&[NonZeroU32], &[u32]) {
        match &self.0 {
            Shape::Scalar(inactive) => (&[], inactive),
            Shape::One(active, inactive) => (active, inactive),
            Shape::Two(active, inactive) => (active, inactive),
            Shape::Three(active, inactive) => (active, inactive),
            Shape::Four(active) => (active, &[]),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Shape {
    Four([NonZeroU32; 4]),
    One([NonZeroU32; 1], [u32; 3]),
    Scalar([u32; 4]),
    Three([NonZeroU32; 3], [u32; 1]),
    Two([NonZeroU32; 2], [u32; 2]),
}

// "The Datatype Message" in format specification version 4.0 places the permutation
// between the two reserved fields in a version 1 compound member's dimension block.
const RESERVED_BEFORE_PERMUTATION: core::ops::Range<usize> = 1..4;
const PERMUTATION: core::ops::Range<usize> = 4..8;
const RESERVED_AFTER_PERMUTATION: core::ops::Range<usize> = 8..12;

/// The dimension block's byte length in "The Datatype Message" of the [format specification,
/// version 4.0][spec] for a version 1 compound member.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const DIMENSION_BLOCK_SIZE: usize = 28;
/// The dimension sizes' byte offset within the dimension block.
const DIMENSION_SIZES_OFFSET: usize = 12;
/// The four dimension slots available to a version 1 compound member.
const MAX_RANK: u8 = 4;
