use alloc::vec::Vec;
use core::ops::Range;

use crate::bytes::{ByteCursor, UnexpectedEof};
use crate::codec::{self, DecodeFrame, DecodePrefix, Parsed};
use crate::error::FormatError;
use crate::width::LengthWidth;

use super::shape::{self, Dataspace, DataspaceType};

/// A dataspace header and its complete dimension lists borrowed from the input.
///
/// Version 1 permutation indices retain their stored values until conversion to a shape.
#[derive(Debug)]
pub(crate) struct StoredDataspace<'a> {
    header: Header,
    current: DimensionFields<'a>,
    maxima: Option<DimensionFields<'a>>,
    permutation: Option<DimensionFields<'a>>,
}

impl StoredDataspace<'_> {
    /// Returns the range of current dimension bytes relative to the start of the message body.
    pub(super) fn dimension_range(&self) -> Range<usize> {
        let start = match self.header.kind {
            Kind::VersionOne => V1_HEADER_SIZE,
            Kind::VersionTwo(_) => V2_HEADER_SIZE,
        };
        start..start + self.current.byte_len()
    }

    /// Copies the current and maximum dimensions into a shape.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnsupportedDataspacePermutation`] at the first dimension whose
    /// stored permutation index differs from its position.
    pub(crate) fn into_dataspace(self) -> Result<Dataspace, FormatError> {
        let Self {
            header,
            current,
            maxima,
            permutation,
        } = self;
        if let Some(indices) = permutation {
            indices.require_identity(header.rank)?;
        }
        Ok(Dataspace {
            space_type: match header.kind {
                Kind::VersionOne if header.rank == 0 => DataspaceType::Scalar,
                Kind::VersionOne => DataspaceType::Simple,
                Kind::VersionTwo(space_type) => space_type,
            },
            rank: header.rank,
            dimensions: current.collect(|value| value),
            max_dimensions: maxima.map(|fields| fields.collect(shape::max_extent_from_length)),
        })
    }
}

impl<'a> DecodePrefix<'a> for StoredDataspace<'a> {
    type Context = LengthWidth;
    type Error = FormatError;

    fn decode_prefix(data: &'a [u8], width: LengthWidth) -> Result<Parsed<Self>, FormatError> {
        let mut cursor = ByteCursor::new(data);
        let header = Header::parse(&mut cursor)?;
        let mut read_list = || DimensionFields::read(&mut cursor, header.rank, width);
        let current = read_list()?;
        let maxima = (header.flags & MAX_DIMENSIONS_FLAG != 0)
            .then(&mut read_list)
            .transpose()?;
        let permutation = (matches!(header.kind, Kind::VersionOne)
            && header.flags & PERMUTATION_FLAG != 0)
            .then(read_list)
            .transpose()?;
        Ok(Parsed::from_cursor(
            Self {
                header,
                current,
                maxima,
                permutation,
            },
            &cursor,
        ))
    }
}

impl<'a> DecodeFrame<'a> for StoredDataspace<'a> {
    type Context = LengthWidth;
    type Error = FormatError;

    fn decode_frame(data: &'a [u8], width: LengthWidth) -> Result<Self, FormatError> {
        Ok(codec::decode_exact::<Self>(data, width)?)
    }
}

#[derive(Debug)]
struct Header {
    kind: Kind,
    rank: u8,
    flags: u8,
}

impl Header {
    fn parse(cursor: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let [version, rank, flags, fourth] = cursor.array::<V2_HEADER_SIZE>()?;
        let kind = match version {
            VERSION_ONE => {
                cursor.take_array::<{ V1_HEADER_SIZE - V2_HEADER_SIZE }>()?;
                Kind::VersionOne
            }
            VERSION_TWO => Kind::VersionTwo(match fourth {
                TYPE_SCALAR => DataspaceType::Scalar,
                TYPE_SIMPLE => DataspaceType::Simple,
                TYPE_NULL => DataspaceType::Null,
                _ => return Err(FormatError::InvalidDataspaceType(fourth)),
            }),
            _ => return Err(FormatError::InvalidDataspaceVersion(version)),
        };
        Ok(Self { kind, rank, flags })
    }
}

#[derive(Debug)]
enum Kind {
    VersionOne,
    VersionTwo(DataspaceType),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DimensionFields<'a> {
    Two(&'a [[u8; 2]]),
    Four(&'a [[u8; 4]]),
    Eight(&'a [[u8; 8]]),
}

impl<'a> DimensionFields<'a> {
    fn read(
        cursor: &mut ByteCursor<'a>,
        rank: u8,
        width: LengthWidth,
    ) -> Result<Self, UnexpectedEof> {
        let count = usize::from(rank);
        let start = cursor.position();
        match width {
            LengthWidth::Two => cursor.take_arrays(count).map(Self::Two),
            LengthWidth::Four => cursor.take_arrays(count).map(Self::Four),
            LengthWidth::Eight => cursor.take_arrays(count).map(Self::Eight),
        }
        .map_err(|error| {
            let width = usize::from(width.get());
            let complete = (error.available - start) / width;
            UnexpectedEof::new(start + (complete + 1) * width, error.available)
        })
    }

    fn byte_len(self) -> usize {
        match self {
            Self::Two(fields) => size_of_val(fields),
            Self::Four(fields) => size_of_val(fields),
            Self::Eight(fields) => size_of_val(fields),
        }
    }

    fn collect<T>(self, convert: impl Fn(u64) -> T) -> Vec<T> {
        fn values<T, const N: usize>(fields: &[[u8; N]], decode: impl Fn([u8; N]) -> T) -> Vec<T> {
            match fields {
                [one] => Vec::from([decode(*one)]),
                [one, two] => Vec::from([decode(*one), decode(*two)]),
                _ => fields.iter().copied().map(decode).collect(),
            }
        }
        match self {
            Self::Two(fields) => values(fields, |bytes| {
                convert(u64::from(u16::from_le_bytes(bytes)))
            }),
            Self::Four(fields) => values(fields, |bytes| {
                convert(u64::from(u32::from_le_bytes(bytes)))
            }),
            Self::Eight(fields) => values(fields, |bytes| convert(u64::from_le_bytes(bytes))),
        }
    }

    fn require_identity(self, rank: u8) -> Result<(), FormatError> {
        fn identity(rank: u8, indices: impl Iterator<Item = u64>) -> Result<(), FormatError> {
            for (dimension, index) in (0..rank).zip(indices) {
                if index != u64::from(dimension) {
                    return Err(FormatError::UnsupportedDataspacePermutation { dimension, index });
                }
            }
            Ok(())
        }
        match self {
            Self::Two(fields) => identity(
                rank,
                fields
                    .iter()
                    .map(|&bytes| u64::from(u16::from_le_bytes(bytes))),
            ),
            Self::Four(fields) => identity(
                rank,
                fields
                    .iter()
                    .map(|&bytes| u64::from(u32::from_le_bytes(bytes))),
            ),
            Self::Eight(fields) => {
                identity(rank, fields.iter().map(|&bytes| u64::from_le_bytes(bytes)))
            }
        }
    }
}

// Section `subsubsec_fmt4_dataobject_hdr_msg_simple`, version 4.0, defines the dataspace headers, types and maximum-dimensions flag.
const VERSION_ONE: u8 = 1;
pub(super) const VERSION_TWO: u8 = 2;
const V1_HEADER_SIZE: usize = 8;
pub(super) const V2_HEADER_SIZE: usize = 4;
pub(super) const TYPE_SCALAR: u8 = 0;
pub(super) const TYPE_SIMPLE: u8 = 1;
pub(super) const TYPE_NULL: u8 = 2;
pub(super) const MAX_DIMENSIONS_FLAG: u8 = 0x01;

// Section `subsubsec_fmt4_dataobject_hdr_msg_simple`, version 4.0, assigns bit 1 to
// the version 1 permutation index list.
const PERMUTATION_FLAG: u8 = 0x02;

#[cfg(test)]
mod tests {
    use super::{DimensionFields, StoredDataspace};
    use crate::codec::DecodeFrame;
    use crate::error::FormatError;
    use crate::width::LengthWidth;

    #[test]
    fn version_one_permutation_is_retained_before_semantic_conversion() {
        let raw = [1, 2, 2, 0, 0, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0xfe, 0xff];
        let stored = StoredDataspace::decode_frame(&raw, LengthWidth::Two).unwrap();
        assert_eq!(
            stored.permutation,
            Some(DimensionFields::Two(&[[0, 0], [0xfe, 0xff]]))
        );
        assert_eq!(
            stored.into_dataspace().unwrap_err(),
            FormatError::UnsupportedDataspacePermutation {
                dimension: 1,
                index: 0xfffe
            }
        );
    }
}
