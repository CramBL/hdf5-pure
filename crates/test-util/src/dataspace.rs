//! Dataspace message bodies: section
//! `subsubsec_fmt4_dataobject_hdr_msg_simple`, version 4.0.

use crate::bytes;
use crate::widths::Widths;

/// A version 2 body with dimensions that fit the selected length width.
#[derive(Clone, Copy, Debug)]
pub struct VersionTwo<'a> {
    widths: Widths,
    kind: Kind,
    rank: u8,
    dimensions: &'a [u64],
    maximum_dimensions: Option<&'a [u64]>,
}

impl<'a> VersionTwo<'a> {
    /// Describes a dataspace for one element with no dimensions.
    pub const fn scalar() -> Self {
        Self::dimensionless(Kind::SCALAR)
    }

    /// Describes a dataspace with no elements or dimensions.
    pub const fn null() -> Self {
        Self::dimensionless(Kind::NULL)
    }

    /// Describes an array with the supplied current and optional maximum dimensions.
    ///
    /// Pass dimension values as stored in the file.
    ///
    /// # Panics
    ///
    /// Panics if the rank is outside `1..=255`, a present maximum dimension count differs from
    /// the rank, the length width is unsupported, or a dimension exceeds that width.
    #[track_caller]
    pub fn simple(
        widths: Widths,
        dimensions: &'a [u64],
        maximum_dimensions: Option<&'a [u64]>,
    ) -> Self {
        let rank = u8::try_from(dimensions.len()).expect("the dataspace rank fits in a byte");
        assert_ne!(rank, 0, "a simple dataspace has at least one dimension");
        if let Some(maxima) = maximum_dimensions {
            assert_eq!(maxima.len(), dimensions.len(), "the dimension lists match");
        }
        let limit = match widths.length {
            2 => u64::from(u16::MAX),
            4 => u64::from(u32::MAX),
            8 => u64::MAX,
            width => panic!("a dataspace length width is 2, 4, or 8 bytes, not {width}"),
        };
        for &dimension in dimensions
            .iter()
            .chain(maximum_dimensions.unwrap_or_default())
        {
            assert!(
                dimension <= limit,
                "dimension {dimension} exceeds the maximum {limit} for this length width"
            );
        }
        Self {
            widths,
            kind: Kind::SIMPLE,
            rank,
            dimensions,
            maximum_dimensions,
        }
    }

    /// Writes the version 2 message body to a new vector.
    pub fn build(self) -> Vec<u8> {
        let Self {
            widths,
            kind,
            rank,
            dimensions,
            maximum_dimensions,
        } = self;
        v2_with_widths(
            rank,
            maximum_dimensions.map_or(Flags::NONE, |_| Flags::MAX_DIMS),
            kind,
            dimensions,
            maximum_dimensions,
            widths,
        )
    }

    const fn dimensionless(kind: Kind) -> Self {
        Self {
            widths: Widths::EIGHT,
            kind,
            rank: 0,
            dimensions: &[],
            maximum_dimensions: None,
        }
    }
}

/// Encodes a version 1 dataspace body with eight-byte dimension fields.
pub fn v1(rank: u8, flags: Flags, dims: &[u64], max_dims: Option<&[u64]>) -> Vec<u8> {
    v1_with_widths(rank, flags, dims, max_dims, Widths::EIGHT)
}

/// Encodes a version 1 dataspace body at `widths.length`.
///
/// Writes rank, flags, and dimension lists as supplied, allowing malformed combinations.
///
/// # Panics
///
/// Panics if the length width is unsupported or a dimension does not fit it.
pub fn v1_with_widths(
    rank: u8,
    flags: Flags,
    dims: &[u64],
    max_dims: Option<&[u64]>,
    widths: Widths,
) -> Vec<u8> {
    let mut message = Vec::new();
    message.push(V1_VERSION);
    message.push(rank);
    message.push(flags.0);
    message.push(0);
    message.extend_from_slice(&[0; 4]);
    push_dims(&mut message, dims, max_dims, widths);
    message
}

/// Encodes a version 2 dataspace body with eight-byte dimension fields.
pub fn v2(rank: u8, flags: Flags, kind: Kind, dims: &[u64], max_dims: Option<&[u64]>) -> Vec<u8> {
    v2_with_widths(rank, flags, kind, dims, max_dims, Widths::EIGHT)
}

/// Encodes a version 2 dataspace body at `widths.length`.
///
/// Writes rank, flags, type, and dimension lists as supplied, allowing malformed combinations.
///
/// # Panics
///
/// Panics if the length width is unsupported or a dimension does not fit it.
pub fn v2_with_widths(
    rank: u8,
    flags: Flags,
    kind: Kind,
    dims: &[u64],
    max_dims: Option<&[u64]>,
    widths: Widths,
) -> Vec<u8> {
    let mut message = Vec::new();
    message.push(V2_VERSION);
    message.push(rank);
    message.push(flags.0);
    message.push(kind.0);
    push_dims(&mut message, dims, max_dims, widths);
    message
}

/// A version 2 scalar dataspace, which holds one element and has no dimension.
pub fn scalar() -> Vec<u8> {
    v2(0, Flags::NONE, Kind::SCALAR, &[], None)
}

fn push_dims(message: &mut Vec<u8>, dims: &[u64], max_dims: Option<&[u64]>, widths: Widths) {
    for &dimension in dims.iter().chain(max_dims.unwrap_or_default()) {
        bytes::push_uint(message, dimension, widths.length);
    }
}

/// Which optional dimension lists follow the fixed fields.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Flags(pub u8);

impl Flags {
    pub const NONE: Self = Self(0x00);
    /// A maximum size per dimension follows the sizes.
    pub const MAX_DIMS: Self = Self(0x01);
    /// A permutation index per dimension follows those, which no release of
    /// the reference library has ever written.
    pub const PERMUTATION: Self = Self(0x02);
}

/// What a version 2 dataspace describes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Kind(pub u8);

impl Kind {
    /// One element and no dimension.
    pub const SCALAR: Self = Self(0);
    /// A rectangular array of elements.
    pub const SIMPLE: Self = Self(1);
    /// No element at all.
    pub const NULL: Self = Self(2);
}

const V1_VERSION: u8 = 1;

const V2_VERSION: u8 = 2;

/// The length in bytes of a version 1 dataspace prefix before its current dimensions.
///
/// Defined in "The Dataspace Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_simple
pub const V1_HEADER_LEN: usize = 8;
/// The length in bytes of a version 2 dataspace prefix before its current dimensions.
///
/// Defined in "The Dataspace Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_simple
pub const V2_HEADER_LEN: usize = 4;
