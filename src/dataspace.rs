//! HDF5 Dataspace message parsing (message type 0x0001).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::bytes;
use crate::error::FormatError;

/// Type of dataspace.
#[derive(Debug, Clone, PartialEq)]
pub enum DataspaceType {
    /// Scalar (single element).
    Scalar,
    /// Simple (N-dimensional array).
    Simple,
    /// Null (no data).
    Null,
}

/// Parsed HDF5 dataspace message.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataspace {
    /// The type of this dataspace.
    pub space_type: DataspaceType,
    /// Number of dimensions (0 for scalar).
    pub rank: u8,
    /// Current dimension sizes.
    pub dimensions: Vec<u64>,
    /// The maximum size of each dimension, when the dataspace message stores them.
    pub max_dimensions: Option<Vec<MaxExtent>>,
}

impl Dataspace {
    /// Parse a dataspace message from raw message bytes.
    ///
    /// `length_size` is needed for dimension value width (from superblock).
    pub fn parse(data: &[u8], length_size: u8) -> Result<Dataspace, FormatError> {
        bytes::ensure_len(data, 0, 4)?;

        let version = data[0];
        let rank = data[1];
        let flags = data[2];

        let (space_type, header_size) = match version {
            1 => {
                // v1: byte 3 is reserved, then 4 reserved bytes
                bytes::ensure_len(data, 0, 8)?;
                let st = if rank == 0 {
                    DataspaceType::Scalar
                } else {
                    DataspaceType::Simple
                };
                (st, 8usize)
            }
            2 => {
                // v2: byte 3 is type
                let type_byte = data[3];
                let st = match type_byte {
                    0 => DataspaceType::Scalar,
                    1 => DataspaceType::Simple,
                    2 => DataspaceType::Null,
                    _ => return Err(FormatError::InvalidDataspaceType(type_byte)),
                };
                (st, 4usize)
            }
            _ => return Err(FormatError::InvalidDataspaceVersion(version)),
        };

        let ls = length_size as usize;
        let mut pos = header_size;

        // Read current dimensions
        let mut dimensions = Vec::with_capacity(rank as usize);
        for _ in 0..rank {
            let dim = bytes::read_length(data, pos, length_size)?;
            dimensions.push(dim);
            pos += ls;
        }

        // Read max dimensions if flags bit 0 is set
        let max_dimensions = if flags & 0x01 != 0 {
            let mut max_dims = Vec::with_capacity(rank as usize);
            for _ in 0..rank {
                max_dims.push(MaxExtent::from_length(bytes::read_length(
                    data,
                    pos,
                    length_size,
                )?));
                pos += ls;
            }
            Some(max_dims)
        } else {
            None
        };

        // A v1 dataspace with flags bit 1 set carries rank × length_size bytes
        // of permutation indices here. Nothing below reads `pos`, so they need
        // no skipping; this crate does not expose them.

        Ok(Dataspace {
            space_type,
            rank,
            dimensions,
            max_dimensions,
        })
    }

    /// Serialize dataspace to HDF5 message bytes (v2 format).
    pub fn serialize(&self, length_size: u8) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(2); // version 2
        buf.push(self.rank);
        let flags = if self.max_dimensions.is_some() {
            0x01
        } else {
            0x00
        };
        buf.push(flags);
        let type_byte = match self.space_type {
            DataspaceType::Scalar => 0,
            DataspaceType::Simple => 1,
            DataspaceType::Null => 2,
        };
        buf.push(type_byte);
        for &dim in &self.dimensions {
            Self::write_length(&mut buf, dim, length_size);
        }
        if let Some(ref max_dims) = self.max_dimensions {
            for &md in max_dims {
                Self::write_length(&mut buf, md.to_length(), length_size);
            }
        }
        buf
    }

    fn write_length(buf: &mut Vec<u8>, val: u64, size: u8) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "size is the length byte width chosen to hold val, so each arm casts to a width that fits by construction"
        )]
        match size {
            2 => buf.extend_from_slice(&(val as u16).to_le_bytes()),
            4 => buf.extend_from_slice(&(val as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&val.to_le_bytes()),
            _ => {}
        }
    }

    /// Total number of elements. Scalar = 1, Null = 0.
    ///
    /// The count saturates at `u64::MAX` rather than panicking when a malformed
    /// dataspace declares dimensions whose product overflows `u64`. Every caller
    /// treats the saturated value as an impossibly large element count, so the
    /// downstream size and limit checks reject it as an error instead.
    pub fn num_elements(&self) -> u64 {
        match self.space_type {
            DataspaceType::Null => 0,
            DataspaceType::Scalar => 1,
            DataspaceType::Simple => {
                if self.dimensions.is_empty() {
                    0
                } else {
                    self.dimensions
                        .iter()
                        .fold(1u64, |acc, &dim| acc.saturating_mul(dim))
                }
            }
        }
    }

    /// Returns the maximum dimensions, when some maximum exceeds its current dimension.
    ///
    /// A dataspace with no maximum dimensions describes a dataset of a fixed shape, and so
    /// does one whose every maximum is fixed at the current dimension: both give `None`.
    pub(crate) fn extensible_max_dimensions(&self) -> Option<&[MaxExtent]> {
        let max_dims = self.max_dimensions.as_deref()?;
        let fixed_at_current = max_dims.len() == self.dimensions.len()
            && max_dims
                .iter()
                .zip(&self.dimensions)
                .all(|(max, &dim)| max.size() == Some(dim));
        (!fixed_at_current).then_some(max_dims)
    }
}

/// The maximum size of one dimension of a dataspace.
///
/// A dataspace message stores a maximum size for each dimension, and the special "unlimited
/// size" marks a dimension the data may expand along indefinitely. The maximum sizes are
/// defined in "The Dataspace Message" of the [format specification, version 4.0][spec]. The C
/// library calls the unlimited size `H5S_UNLIMITED`.
///
/// # Examples
///
/// ```
/// use hdf5_pure::MaxExtent;
///
/// // A dataset that grows along its first dimension and keeps three columns.
/// let maxshape = [MaxExtent::Unlimited, MaxExtent::Fixed(3)];
/// assert_eq!(maxshape[0].size(), None);
/// assert_eq!(maxshape[1].size(), Some(3));
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_simple
#[allow(clippy::exhaustive_enums)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MaxExtent {
    /// The dimension grows to this many elements and no further.
    Fixed(u64),
    /// The dimension grows without bound.
    Unlimited,
}

impl MaxExtent {
    /// Interprets a maximum dimension size read from a dataspace message.
    ///
    /// Only an eight-byte field holds the unlimited size, so a maximum read from a narrower
    /// one is always fixed.
    pub(crate) const fn from_length(length: u64) -> Self {
        if length == UNLIMITED_LENGTH {
            Self::Unlimited
        } else {
            Self::Fixed(length)
        }
    }

    /// Returns the number of elements the dimension reaches at most, or `None` for a
    /// dimension that grows without bound.
    pub const fn size(self) -> Option<u64> {
        match self {
            Self::Fixed(size) => Some(size),
            Self::Unlimited => None,
        }
    }

    /// Returns `true` if a dimension of `dim` elements stays within this maximum.
    pub(crate) const fn admits(self, dim: u64) -> bool {
        match self {
            Self::Fixed(size) => size >= dim,
            Self::Unlimited => true,
        }
    }

    /// Returns the maximum size as a dataspace message stores it, all ones for
    /// [`Self::Unlimited`].
    pub(crate) const fn to_length(self) -> u64 {
        match self {
            Self::Fixed(size) => size,
            Self::Unlimited => UNLIMITED_LENGTH,
        }
    }

    const fn is_unlimited(self) -> bool {
        matches!(self, Self::Unlimited)
    }
}

/// A dataset's shape beside the maximum shape that bounds it.
///
/// [`Extent::new`] checks the two against each other, so a writer that takes an `Extent` has a
/// maximum shape of the shape's own rank, with every dimension inside its maximum.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Extent<'a> {
    /// The current size of each dimension.
    dims: &'a [u64],
    /// The maximum size of each dimension, `None` for a dataset of a fixed shape.
    max_dims: Option<&'a [MaxExtent]>,
}

impl<'a> Extent<'a> {
    /// Pairs a shape with the maximum shape that bounds it.
    ///
    /// # Errors
    ///
    /// Returns a reason for the caller to report if `max_dims` and `dims` differ in rank, if a
    /// maximum is smaller than the dimension it bounds, or if a maximum is
    /// [`MaxExtent::Fixed`] at the unlimited size, which [`Dataspace::serialize`] writes as
    /// [`MaxExtent::Unlimited`].
    pub(crate) fn new(
        dims: &'a [u64],
        max_dims: Option<&'a [MaxExtent]>,
    ) -> Result<Self, &'static str> {
        if let Some(max_dims) = max_dims {
            if max_dims.len() != dims.len() {
                return Err("maxshape must have the same rank as the dataset shape");
            }
            if max_dims
                .iter()
                .zip(dims)
                .any(|(max, &dim)| !max.admits(dim))
            {
                return Err("maxshape must be at least the current shape in every dimension");
            }
            if max_dims.contains(&MaxExtent::Fixed(UNLIMITED_LENGTH)) {
                return Err(
                    "a fixed maximum of u64::MAX is the format's unlimited marker; \
                     name MaxExtent::Unlimited for a dimension that grows without bound",
                );
            }
        }
        Ok(Self { dims, max_dims })
    }

    pub(crate) const fn dims(self) -> &'a [u64] {
        self.dims
    }

    /// Returns how many dimensions grow without bound.
    ///
    /// A dataset of a fixed shape has none.
    pub(crate) fn unlimited_dimensions(self) -> usize {
        self.max_dims.map_or(0, |max_dims| {
            max_dims.iter().filter(|max| max.is_unlimited()).count()
        })
    }
}

// The maximum dimension size a dataspace message stores for a dimension that grows
// without bound, the format's "unlimited size". The C library defines `H5S_UNLIMITED` as
// `HSIZE_UNDEF`, which is `UINT64_MAX` (`H5Spublic.h` and `H5public.h`, HDF5 2.2.0). A
// message written with a narrower "Size of Lengths" cannot hold it: `H5O__sdspace_decode`
// zero-extends the field it reads (`H5F_DECODE_LENGTH` in `H5Osdspace.c`, HDF5 2.2.0), so
// only an eight-byte field decodes to this value.
const UNLIMITED_LENGTH: u64 = u64::MAX;

#[cfg(test)]
mod tests {
    use super::*;

    fn build_v1_dataspace(rank: u8, flags: u8, dims: &[u64], max_dims: Option<&[u64]>) -> Vec<u8> {
        let length_size = 8u8;
        let mut buf = Vec::new();
        buf.push(1); // version
        buf.push(rank);
        buf.push(flags);
        buf.push(0); // reserved
        buf.extend_from_slice(&[0u8; 4]); // reserved(4)
        for &d in dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        if let Some(md) = max_dims {
            for &d in md {
                buf.extend_from_slice(&d.to_le_bytes());
            }
        }
        let _ = length_size;
        buf
    }

    fn build_v2_dataspace(
        rank: u8,
        flags: u8,
        type_byte: u8,
        dims: &[u64],
        max_dims: Option<&[u64]>,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(2); // version
        buf.push(rank);
        buf.push(flags);
        buf.push(type_byte);
        for &d in dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        if let Some(md) = max_dims {
            for &d in md {
                buf.extend_from_slice(&d.to_le_bytes());
            }
        }
        buf
    }

    #[test]
    fn scalar_v1() {
        let data = build_v1_dataspace(0, 0, &[], None);
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.space_type, DataspaceType::Scalar);
        assert_eq!(ds.rank, 0);
        assert!(ds.dimensions.is_empty());
        assert!(ds.max_dimensions.is_none());
        assert_eq!(ds.num_elements(), 1);
    }

    #[test]
    fn null_v2() {
        let data = build_v2_dataspace(0, 0, 2, &[], None);
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.space_type, DataspaceType::Null);
        assert_eq!(ds.num_elements(), 0);
    }

    #[test]
    fn simple_1d() {
        let data = build_v1_dataspace(1, 0, &[5], None);
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.space_type, DataspaceType::Simple);
        assert_eq!(ds.rank, 1);
        assert_eq!(ds.dimensions, vec![5]);
        assert!(ds.max_dimensions.is_none());
        assert_eq!(ds.num_elements(), 5);
    }

    #[test]
    fn simple_2d() {
        let data = build_v1_dataspace(2, 0, &[3, 4], None);
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.rank, 2);
        assert_eq!(ds.dimensions, vec![3, 4]);
        assert_eq!(ds.num_elements(), 12);
    }

    #[test]
    fn simple_3d_with_max_dims_unlimited() {
        let data = build_v1_dataspace(3, 0x01, &[2, 3, 4], Some(&[10, u64::MAX, 100]));
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.rank, 3);
        assert_eq!(ds.dimensions, vec![2, 3, 4]);
        let md = ds.max_dimensions.clone().unwrap();
        assert_eq!(
            md,
            vec![
                MaxExtent::Fixed(10),
                MaxExtent::Unlimited,
                MaxExtent::Fixed(100)
            ]
        );
        assert_eq!(ds.num_elements(), 24);
    }

    #[test]
    fn num_elements_saturates_on_overflow() {
        // A malformed dataspace can declare dimensions whose product exceeds
        // u64::MAX. num_elements() must saturate instead of panicking, so the
        // callers can reject the impossibly large count as an error.
        let data = build_v1_dataspace(3, 0, &[1 << 40, 1 << 40, 1 << 40], None);
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.num_elements(), u64::MAX);
    }

    #[test]
    fn v2_simple() {
        let data = build_v2_dataspace(1, 0, 1, &[7], None);
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.space_type, DataspaceType::Simple);
        assert_eq!(ds.dimensions, vec![7]);
    }

    #[test]
    fn v2_scalar() {
        let data = build_v2_dataspace(0, 0, 0, &[], None);
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.space_type, DataspaceType::Scalar);
        assert_eq!(ds.num_elements(), 1);
    }

    #[test]
    fn v1_with_4byte_length() {
        let mut buf = Vec::new();
        buf.push(1); // version
        buf.push(1); // rank
        buf.push(0); // flags
        buf.push(0); // reserved
        buf.extend_from_slice(&[0u8; 4]); // reserved(4)
        buf.extend_from_slice(&10u32.to_le_bytes()); // dim with length_size=4
        let ds = Dataspace::parse(&buf, 4).unwrap();
        assert_eq!(ds.dimensions, vec![10]);
    }

    #[test]
    fn truncated_data_error() {
        let data = [1u8, 2]; // too short
        let err = Dataspace::parse(&data, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn invalid_version_error() {
        let data = [5u8, 0, 0, 0, 0, 0, 0, 0];
        let err = Dataspace::parse(&data, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidDataspaceVersion(5));
    }

    #[test]
    fn invalid_v2_type_error() {
        let data = build_v2_dataspace(0, 0, 5, &[], None);
        let err = Dataspace::parse(&data, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidDataspaceType(5));
    }

    #[test]
    fn v1_with_max_dims() {
        let data = build_v1_dataspace(1, 0x01, &[5], Some(&[10]));
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.max_dimensions, Some(vec![MaxExtent::Fixed(10)]));
    }

    #[test]
    fn all_ones_max_dimension_is_unlimited_only_at_eight_bytes() {
        let eight = build_v1_dataspace(1, 0x01, &[5], Some(&[u64::MAX]));
        let ds = Dataspace::parse(&eight, 8).unwrap();
        assert_eq!(ds.max_dimensions, Some(vec![MaxExtent::Unlimited]));

        // Four bytes cannot hold `H5S_UNLIMITED`: the C library zero-extends
        // the field, so all ones in four bytes is the finite maximum 2^32 - 1.
        let mut four = Vec::new();
        four.extend_from_slice(&[1, 1, 0x01, 0]);
        four.extend_from_slice(&[0u8; 4]);
        four.extend_from_slice(&5u32.to_le_bytes());
        four.extend_from_slice(&u32::MAX.to_le_bytes());
        let ds = Dataspace::parse(&four, 4).unwrap();
        assert_eq!(
            ds.max_dimensions,
            Some(vec![MaxExtent::Fixed(u64::from(u32::MAX))])
        );
    }

    #[test]
    fn serialize_writes_all_ones_for_an_unlimited_maximum() {
        let ds = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: vec![4, 6],
            max_dimensions: Some(vec![MaxExtent::Unlimited, MaxExtent::Fixed(6)]),
        };
        let bytes = ds.serialize(8);
        assert_eq!(bytes[..4], [2, 2, 0x01, 1]);
        assert_eq!(bytes[20..28], u64::MAX.to_le_bytes());
        assert_eq!(bytes[28..36], 6u64.to_le_bytes());
        assert_eq!(Dataspace::parse(&bytes, 8).unwrap(), ds);
    }

    #[test]
    fn extensible_max_dimensions_skips_a_maximum_at_the_current_shape() {
        let fixed = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: vec![4, 6],
            max_dimensions: Some(vec![MaxExtent::Fixed(4), MaxExtent::Fixed(6)]),
        };
        assert_eq!(fixed.extensible_max_dimensions(), None);

        let none = Dataspace {
            max_dimensions: None,
            ..fixed.clone()
        };
        assert_eq!(none.extensible_max_dimensions(), None);

        let grown = Dataspace {
            max_dimensions: Some(vec![MaxExtent::Fixed(4), MaxExtent::Unlimited]),
            ..fixed
        };
        assert_eq!(
            grown.extensible_max_dimensions(),
            Some(&[MaxExtent::Fixed(4), MaxExtent::Unlimited][..])
        );
    }

    #[test]
    fn an_extent_joins_a_shape_with_a_maximum_that_bounds_it() {
        let extent = Extent::new(&[4, 6], Some(&[MaxExtent::Unlimited, MaxExtent::Fixed(6)]))
            .expect("an unlimited dimension bounds anything, and 6 bounds 6");
        assert_eq!(extent.dims(), &[4, 6]);
        assert_eq!(extent.unlimited_dimensions(), 1);

        let bounded = Extent::new(&[4, 6], None).unwrap();
        assert_eq!(bounded.unlimited_dimensions(), 0);
    }

    #[test]
    fn an_extent_rejects_a_maximum_shape_that_disagrees_with_its_shape() {
        assert_eq!(
            Extent::new(&[4], Some(&[MaxExtent::Fixed(4), MaxExtent::Fixed(4)])).unwrap_err(),
            "maxshape must have the same rank as the dataset shape"
        );
        assert_eq!(
            Extent::new(&[4], Some(&[MaxExtent::Fixed(3)])).unwrap_err(),
            "maxshape must be at least the current shape in every dimension"
        );
    }

    #[test]
    fn an_extent_rejects_a_fixed_maximum_at_the_unlimited_size() {
        let err = Extent::new(&[4], Some(&[MaxExtent::Fixed(u64::MAX)])).unwrap_err();
        assert!(err.contains("the format's unlimited marker"), "{err}");
    }

    #[test]
    fn max_extent_reports_its_size_and_bounds_a_dimension() {
        assert_eq!(MaxExtent::Fixed(7).size(), Some(7));
        assert_eq!(MaxExtent::Unlimited.size(), None);
        assert!(!MaxExtent::Fixed(7).is_unlimited());
        assert!(MaxExtent::Unlimited.is_unlimited());
        assert!(MaxExtent::Fixed(7).admits(7));
        assert!(!MaxExtent::Fixed(7).admits(8));
        assert!(MaxExtent::Unlimited.admits(u64::MAX));
    }
}
