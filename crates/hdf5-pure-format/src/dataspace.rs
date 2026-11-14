//! HDF5 Dataspace message parsing (message type 0x0001).

use alloc::vec::Vec;
use core::ops::Range;

use hdf5_pure_core::__private::ElementCount;

use crate::bytes;
use crate::bytes::Fields;
use crate::error::FormatError;
use crate::width;
use crate::width::LengthWidth;

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
    /// Parses a dataspace message body from raw bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidLengthSize`] if `length_size` is not 2, 4, or 8, and otherwise
    /// returns the errors [`parse_with_length_width`](Self::parse_with_length_width) returns.
    pub fn parse(data: &[u8], length_size: u8) -> Result<Dataspace, FormatError> {
        Self::parse_with_length_width(data, LengthWidth::try_from(length_size)?)
    }

    /// Parses a dataspace message body using the file's validated length width.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDataspaceVersion`] if the version is unsupported,
    /// [`FormatError::InvalidDataspaceType`] if a version 2 type byte is invalid, or
    /// [`FormatError::UnexpectedEof`] if the message ends within a field.
    pub fn parse_with_length_width(
        data: &[u8],
        length_width: LengthWidth,
    ) -> Result<Dataspace, FormatError> {
        Self::parse_with_dimension_fields(data, length_width).map(|(dataspace, _)| dataspace)
    }

    /// Parses a dataspace and returns the byte range of its current dimension fields.
    ///
    /// The range is relative to the message body's start and excludes maximum dimensions and
    /// version 1 permutation fields. It is empty for a rank-zero dataspace.
    /// Message versions 1 and 2 are defined in "The Dataspace Message" of the
    /// [format specification, version 4.0][spec].
    ///
    /// # Errors
    ///
    /// Returns the parsing errors of [`Self::parse_with_length_width`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_format::{__private as format, FormatError, LengthWidth, MaxExtent};
    ///
    /// let dataspace = format::Dataspace {
    ///     space_type: format::DataspaceType::Simple,
    ///     rank: 1,
    ///     dimensions: vec![3],
    ///     max_dimensions: Some(vec![MaxExtent::Fixed(7)]),
    /// };
    /// let mut body = dataspace.serialize(2)?;
    /// let (parsed, dimensions) = format::Dataspace::parse_with_dimension_fields(
    ///     &body, LengthWidth::Two,
    /// )?;
    /// assert_eq!(parsed, dataspace);
    /// assert_eq!(dimensions, 4..6);
    /// format::write_length_at(&mut body, dimensions.start, 6, LengthWidth::Two)?;
    /// let grown = format::Dataspace::parse_with_length_width(&body, LengthWidth::Two)?;
    /// assert_eq!(grown.dimensions, [6]);
    /// assert_eq!(grown.max_dimensions, Some(vec![MaxExtent::Fixed(7)]));
    /// # Ok::<(), FormatError>(())
    /// ```
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_simple
    pub fn parse_with_dimension_fields(
        data: &[u8],
        length_width: LengthWidth,
    ) -> Result<(Dataspace, Range<usize>), FormatError> {
        bytes::ensure_len(data, 0, V2_HEADER_SIZE)?;

        let version = data[0];
        let rank = data[1];
        let flags = data[2];

        let (space_type, header_size) = match version {
            VERSION_ONE => {
                // v1: byte 3 is reserved, then 4 reserved bytes
                bytes::ensure_len(data, 0, V1_HEADER_SIZE)?;
                let st = if rank == 0 {
                    DataspaceType::Scalar
                } else {
                    DataspaceType::Simple
                };
                (st, V1_HEADER_SIZE)
            }
            VERSION_TWO => {
                // v2: byte 3 is type
                let type_byte = data[3];
                let st = match type_byte {
                    TYPE_SCALAR => DataspaceType::Scalar,
                    TYPE_SIMPLE => DataspaceType::Simple,
                    TYPE_NULL => DataspaceType::Null,
                    _ => return Err(FormatError::InvalidDataspaceType(type_byte)),
                };
                (st, V2_HEADER_SIZE)
            }
            _ => return Err(FormatError::InvalidDataspaceVersion(version)),
        };

        let mut fields = Fields::new(data, header_size);
        let dimensions =
            Self::read_dimensions(&mut fields, rank, length_width, |dimension| dimension)?;
        let dimension_fields = header_size..fields.pos();
        let max_dimensions = if flags & MAX_DIMENSIONS_FLAG != 0 {
            Some(Self::read_dimensions(
                &mut fields,
                rank,
                length_width,
                max_extent_from_length,
            )?)
        } else {
            None
        };

        // A v1 dataspace with flags bit 1 set carries rank × `length_size` bytes
        // of permutation indices here. Nothing below reads the cursor position, so they need
        // no skipping. This crate does not expose them.

        Ok((
            Dataspace {
                space_type,
                rank,
                dimensions,
                max_dimensions,
            },
            dimension_fields,
        ))
    }

    /// Serializes a version 2 dataspace message with `length_size`-byte dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidLengthSize`] if `length_size` is not 2, 4, or 8, or
    /// [`FormatError::LengthTooLarge`] if a dimension or maximum dimension exceeds that width.
    /// An unlimited maximum requires eight-byte lengths, and [`stored_max_extent`] returns the
    /// maximum to store in its place at 2 and 4 bytes.
    pub fn serialize(&self, length_size: u8) -> Result<Vec<u8>, FormatError> {
        let length_width = LengthWidth::try_from(length_size)?;
        let mut buf = Vec::new();
        buf.push(VERSION_TWO);
        buf.push(self.rank);
        let flags = if self.max_dimensions.is_some() {
            MAX_DIMENSIONS_FLAG
        } else {
            0x00
        };
        buf.push(flags);
        let type_byte = match self.space_type {
            DataspaceType::Scalar => TYPE_SCALAR,
            DataspaceType::Simple => TYPE_SIMPLE,
            DataspaceType::Null => TYPE_NULL,
        };
        buf.push(type_byte);
        for &dim in &self.dimensions {
            bytes::try_write_length(&mut buf, dim, length_width)?;
        }
        if let Some(ref max_dims) = self.max_dimensions {
            for &md in max_dims {
                bytes::try_write_length(&mut buf, max_extent_to_length(md), length_width)?;
            }
        }
        Ok(buf)
    }

    /// Returns the element count, saturating dimension products at [`u64::MAX`].
    ///
    /// Returns zero for a null dataspace or a simple dataspace with no dimensions or a zero
    /// dimension. Returns one for a scalar dataspace.
    pub fn num_elements(&self) -> ElementCount {
        ElementCount::new(match self.space_type {
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
        })
    }

    /// Returns the maximum dimensions, when some maximum exceeds its current dimension.
    ///
    /// A dataspace with no maximum dimensions describes a dataset of a fixed shape, and so
    /// does one whose every maximum is fixed at the current dimension: both give `None`.
    pub fn extensible_max_dimensions(&self) -> Option<&[MaxExtent]> {
        let max_dims = self.max_dimensions.as_deref()?;
        let fixed_at_current = max_dims.len() == self.dimensions.len()
            && max_dims
                .iter()
                .zip(&self.dimensions)
                .all(|(max, &dim)| max.size() == Some(dim));
        (!fixed_at_current).then_some(max_dims)
    }

    fn read_dimensions<T>(
        fields: &mut Fields<'_>,
        rank: u8,
        width: LengthWidth,
        from_length: impl Fn(u64) -> T,
    ) -> Result<Vec<T>, FormatError> {
        if rank == 0 {
            return Ok(Vec::new());
        }
        let count = usize::from(rank);
        let mut dimensions = Vec::with_capacity(count);
        match width {
            LengthWidth::Two => dimensions.extend(
                Self::dimension_arrays::<2>(fields, count)?
                    .iter()
                    .map(|&bytes| from_length(u64::from(u16::from_le_bytes(bytes)))),
            ),
            LengthWidth::Four => dimensions.extend(
                Self::dimension_arrays::<4>(fields, count)?
                    .iter()
                    .map(|&bytes| from_length(u64::from(u32::from_le_bytes(bytes)))),
            ),
            LengthWidth::Eight => dimensions.extend(
                Self::dimension_arrays::<8>(fields, count)?
                    .iter()
                    .map(|&bytes| from_length(u64::from_le_bytes(bytes))),
            ),
        }
        Ok(dimensions)
    }

    /// Borrows a dimension list before its fields are decoded.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if a dimension field is incomplete. The `expected`
    /// offset is the end of the first incomplete field in the original input.
    fn dimension_arrays<'a, const N: usize>(
        fields: &mut Fields<'a>,
        count: usize,
    ) -> Result<&'a [[u8; N]], FormatError> {
        let start = fields.pos();
        fields
            .take_arrays(count)
            .map_err(|error| Self::dimension_array_error::<N>(start, error))
    }

    /// Reports the end of the first incomplete field for [`FormatError::UnexpectedEof`].
    ///
    /// Returns other errors unchanged.
    #[cold]
    fn dimension_array_error<const N: usize>(start: usize, error: FormatError) -> FormatError {
        match error {
            FormatError::UnexpectedEof { available, .. } => FormatError::UnexpectedEof {
                expected: start.saturating_add(
                    (available.saturating_sub(start) / N)
                        .saturating_add(1)
                        .saturating_mul(N),
                ),
                available,
            },
            error => error,
        }
    }
}

pub(crate) use hdf5_pure_core::MaxExtent;

/// A dataset's shape beside the maximum shape that bounds it.
///
/// [`Extent::new`] checks the two against each other, so a writer that takes an `Extent` has a
/// maximum shape of the shape's own rank, with every dimension inside its maximum.
#[derive(Clone, Copy, Debug)]
pub struct Extent<'a> {
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
    pub fn new(dims: &'a [u64], max_dims: Option<&'a [MaxExtent]>) -> Result<Self, &'static str> {
        if let Some(max_dims) = max_dims {
            if max_dims.len() != dims.len() {
                return Err("maxshape must have the same rank as the dataset shape");
            }
            if max_dims
                .iter()
                .zip(dims)
                .any(|(max, &dim)| !max_extent_admits(*max, dim))
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

    /// Returns the current size of each dimension.
    pub const fn dims(self) -> &'a [u64] {
        self.dims
    }

    /// Returns how many dimensions grow without bound.
    ///
    /// A dataset of a fixed shape has none.
    pub fn unlimited_dimensions(self) -> usize {
        self.max_dims.map_or(0, |max_dims| {
            max_dims
                .iter()
                .filter(|max| matches!(max, MaxExtent::Unlimited))
                .count()
        })
    }
}

/// Returns the maxima a dataspace message with `length_width`-byte lengths stores for
/// `max_dims`, each as [`stored_max_extent`] returns it.
///
/// # Errors
///
/// Returns [`FormatError::LengthTooLarge`] if a dimension in `dims` exceeds `length_width`. The
/// dimensions are checked first, since a dimension past the width also exceeds the maximum
/// stored for an unlimited one.
pub fn stored_max_dimensions(
    length_width: LengthWidth,
    dims: &[u64],
    max_dims: Vec<MaxExtent>,
) -> Result<Vec<MaxExtent>, FormatError> {
    dims.iter()
        .try_for_each(|&dim| bytes::require_length(dim, length_width))?;
    Ok(max_dims
        .into_iter()
        .map(|max| stored_max_extent(length_width, max))
        .collect())
}

/// Returns the maximum a dataspace message with `length_width`-byte lengths stores for `max`.
///
/// A fixed maximum, and any maximum at eight-byte lengths, is stored as it is. At 2- and 4-byte
/// lengths an unlimited maximum is stored as [`MaxExtent::Fixed`] at the largest length the
/// width holds, the value [`Dataspace::parse`] reads from an all-ones field of that width.
///
/// # Examples
///
/// ```
/// use hdf5_pure_format::{__private as format, LengthWidth, MaxExtent};
///
/// assert_eq!(
///     format::stored_max_extent(LengthWidth::Four, MaxExtent::Unlimited),
///     MaxExtent::Fixed(0xffff_ffff),
/// );
/// assert_eq!(
///     format::stored_max_extent(LengthWidth::Eight, MaxExtent::Unlimited),
///     MaxExtent::Unlimited,
/// );
/// ```
pub fn stored_max_extent(length_width: LengthWidth, max: MaxExtent) -> MaxExtent {
    match (length_width, max) {
        (LengthWidth::Eight, _) | (_, MaxExtent::Fixed(_)) => max,
        // libhdf5 reads the all-ones field back as this finite size and cannot open an extensible
        // array over it (https://github.com/HDFGroup/hdf5/issues/6864).
        (LengthWidth::Two | LengthWidth::Four, MaxExtent::Unlimited) => {
            MaxExtent::Fixed(width::length_width_max(length_width))
        }
    }
}

// The maximum dimension size libhdf5 uses for a dimension that grows without bound:
// `H5S_UNLIMITED` is `HSIZE_UNDEF`, which is `UINT64_MAX` (`H5Spublic.h` and `H5public.h`, HDF5
// 2.2.0). `H5O__sdspace_encode` keeps its low "Size of Lengths" bytes and `H5O__sdspace_decode`
// zero-extends them (`H5Osdspace.c`, and `H5_ENCODE_LENGTH_LEN` and `H5_DECODE_LENGTH_LEN` in
// `H5encode.h`, HDF5 2.2.0), so only an eight-byte field decodes back to this value
// (https://github.com/HDFGroup/hdf5/issues/6864).
const UNLIMITED_LENGTH: u64 = u64::MAX;

fn max_extent_from_length(length: u64) -> MaxExtent {
    if length == UNLIMITED_LENGTH {
        MaxExtent::Unlimited
    } else {
        MaxExtent::Fixed(length)
    }
}

fn max_extent_to_length(max: MaxExtent) -> u64 {
    match max {
        MaxExtent::Fixed(size) => size,
        MaxExtent::Unlimited => UNLIMITED_LENGTH,
    }
}

fn max_extent_admits(max: MaxExtent, dim: u64) -> bool {
    match max {
        MaxExtent::Fixed(size) => size >= dim,
        MaxExtent::Unlimited => true,
    }
}

// Section `subsubsec_fmt4_dataobject_hdr_msg_simple`, version 4.0, defines the dataspace headers, types and maximum-dimensions flag.
const VERSION_ONE: u8 = 1;
const VERSION_TWO: u8 = 2;
const V1_HEADER_SIZE: usize = 8;
const V2_HEADER_SIZE: usize = 4;
const TYPE_SCALAR: u8 = 0;
const TYPE_SIMPLE: u8 = 1;
const TYPE_NULL: u8 = 2;
const MAX_DIMENSIONS_FLAG: u8 = 0x01;

#[cfg(test)]
mod tests {
    use test_util::dataspace;

    use super::*;

    #[rstest::rstest]
    #[case::rank_zero(&[])]
    #[case::one_dimension(&[0])]
    #[case::multiple_dimensions(&[3, 5, u64::from(u16::MAX)])]
    fn dimension_arrays_preserve_values_ranges_and_suffixes(
        #[case] dimensions: &[u64],
        #[values(Version::One, Version::Two)] version: Version,
        #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] width: LengthWidth,
        #[values(false, true)] maxima: bool,
        #[values(false, true)] suffix: bool,
    ) {
        let maximum = width::length_width_max(width);
        let max_dimensions = maxima.then(|| vec![maximum; dimensions.len()]);
        let mut raw = version.body(dimensions, max_dimensions.as_deref(), width);
        if matches!(version, Version::One) {
            raw[3..8].fill(0xa5);
        }
        raw[2] |= 0xfc;
        if suffix {
            raw.extend_from_slice(&[0xef, 0xbe, 0xad, 0xde]);
        }
        let expected = Dataspace {
            space_type: if matches!(version, Version::One) && dimensions.is_empty() {
                DataspaceType::Scalar
            } else {
                DataspaceType::Simple
            },
            rank: u8::try_from(dimensions.len()).unwrap(),
            dimensions: dimensions.to_vec(),
            max_dimensions: maxima.then(|| {
                vec![
                    if maximum == u64::MAX {
                        MaxExtent::Unlimited
                    } else {
                        MaxExtent::Fixed(maximum)
                    };
                    dimensions.len()
                ]
            }),
        };
        let range = version.header_size()
            ..version.header_size() + dimensions.len() * usize::from(width.get());
        assert_eq!(
            Dataspace::parse_with_dimension_fields(&raw, width).unwrap(),
            (expected.clone(), range)
        );
        assert_eq!(Dataspace::parse(&raw, width.get()).unwrap(), expected);
    }

    #[rstest::rstest]
    #[case::two_start(LengthWidth::Two, 0)]
    #[case::two_partial(LengthWidth::Two, 1)]
    #[case::four_start(LengthWidth::Four, 0)]
    #[case::four_one_byte(LengthWidth::Four, 1)]
    #[case::four_two_bytes(LengthWidth::Four, 2)]
    #[case::four_three_bytes(LengthWidth::Four, 3)]
    #[case::eight_start(LengthWidth::Eight, 0)]
    #[case::eight_one_byte(LengthWidth::Eight, 1)]
    #[case::eight_two_bytes(LengthWidth::Eight, 2)]
    #[case::eight_three_bytes(LengthWidth::Eight, 3)]
    #[case::eight_four_bytes(LengthWidth::Eight, 4)]
    #[case::eight_five_bytes(LengthWidth::Eight, 5)]
    #[case::eight_six_bytes(LengthWidth::Eight, 6)]
    #[case::eight_seven_bytes(LengthWidth::Eight, 7)]
    fn truncated_dimension_arrays_report_the_first_incomplete_field(
        #[case] width: LengthWidth,
        #[case] partial: usize,
        #[values(Version::One, Version::Two)] version: Version,
        #[values(0, 1, 2, 3, 4, 5)] complete: usize,
    ) {
        let raw = version.body(&[3, 5, 7], Some(&[11, 13, 17]), width);
        let available = version.header_size() + complete * usize::from(width.get()) + partial;
        let expected = version.header_size() + (complete + 1) * usize::from(width.get());
        assert_eq!(
            Dataspace::parse_with_dimension_fields(&raw[..available], width).unwrap_err(),
            FormatError::UnexpectedEof {
                expected,
                available,
            }
        );
    }

    #[rstest::rstest]
    #[case::empty_header(&[], 2, FormatError::UnexpectedEof { expected: 4, available: 0 })]
    #[case::one_header_byte(&[1], 4, FormatError::UnexpectedEof { expected: 4, available: 1 })]
    #[case::two_header_bytes(&[2, 3], 8, FormatError::UnexpectedEof { expected: 4, available: 2 })]
    #[case::three_header_bytes(&[1, 3, 0xff], 2, FormatError::UnexpectedEof { expected: 4, available: 3 })]
    #[case::five_version_one_bytes(&[1, 3, 0xff, 9, 0], 4, FormatError::UnexpectedEof { expected: 8, available: 5 })]
    #[case::six_version_one_bytes(&[1, 3, 0xff, 9, 0, 0], 8, FormatError::UnexpectedEof { expected: 8, available: 6 })]
    #[case::seven_version_one_bytes(&[1, 3, 0xff, 9, 0, 0, 0], 2, FormatError::UnexpectedEof { expected: 8, available: 7 })]
    #[case::empty_invalid_width(&[], 0, FormatError::InvalidLengthSize(0))]
    #[case::version_and_type_with_invalid_width(&[9, 3, 0xff, 9], 3, FormatError::InvalidLengthSize(3))]
    #[case::short_before_version(&[9, 3, 0xff], 2, FormatError::UnexpectedEof { expected: 4, available: 3 })]
    #[case::version_before_type(&[9, 3, 0xff, 9], 4, FormatError::InvalidDataspaceVersion(9))]
    #[case::version_one_reserved_header(&[1, 3, 0xff, 9], 8, FormatError::UnexpectedEof { expected: 8, available: 4 })]
    #[case::type_before_dimensions(&[2, 3, 0xff, 9], 2, FormatError::InvalidDataspaceType(9))]
    fn header_errors_precede_dimension_acquisition(
        #[case] raw: &[u8],
        #[case] width: u8,
        #[case] expected: FormatError,
    ) {
        assert_eq!(Dataspace::parse(raw, width).unwrap_err(), expected);
    }

    #[rstest::rstest]
    #[case::scalar(0, DataspaceType::Scalar)]
    #[case::simple(1, DataspaceType::Simple)]
    #[case::null(2, DataspaceType::Null)]
    fn version_two_types_retain_the_stored_dimensions(
        #[case] type_byte: u8,
        #[case] space_type: DataspaceType,
        #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] width: LengthWidth,
    ) {
        let mut raw = Version::Two.body(&[3, 5], None, width);
        raw[3] = type_byte;
        assert_eq!(
            Dataspace::parse(&raw, width.get()).unwrap(),
            Dataspace {
                space_type,
                rank: 2,
                dimensions: vec![3, 5],
                max_dimensions: None,
            }
        );
    }

    #[rstest::rstest]
    #[case::absent(0)]
    #[case::partial(1)]
    #[case::complete(3)]
    fn version_one_permutation_fields_remain_unread(
        #[case] permutation_fields: usize,
        #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] width: LengthWidth,
        #[values(false, true)] maxima: bool,
    ) {
        let max_dimensions = maxima.then_some([7, 9, 11]);
        let mut raw = Version::One.body(
            &[3, 5, 7],
            max_dimensions
                .as_ref()
                .map(|dimensions| dimensions.as_slice()),
            width,
        );
        raw[2] |= dataspace::Flags::PERMUTATION.0;
        raw.resize(
            raw.len() + permutation_fields * usize::from(width.get()),
            0xff,
        );
        let (parsed, range) = Dataspace::parse_with_dimension_fields(&raw, width).unwrap();
        assert_eq!(range, 8..8 + 3 * usize::from(width.get()));
        assert_eq!(
            parsed,
            Dataspace {
                space_type: DataspaceType::Simple,
                rank: 3,
                dimensions: vec![3, 5, 7],
                max_dimensions: maxima.then(|| {
                    vec![
                        MaxExtent::Fixed(7),
                        MaxExtent::Fixed(9),
                        MaxExtent::Fixed(11),
                    ]
                }),
            }
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum Version {
        One,
        Two,
    }

    impl Version {
        fn body(self, dimensions: &[u64], maxima: Option<&[u64]>, width: LengthWidth) -> Vec<u8> {
            let rank = u8::try_from(dimensions.len()).unwrap();
            let widths = test_util::widths::Widths::new(8, usize::from(width.get()));
            let flags = if maxima.is_some() {
                dataspace::Flags::MAX_DIMS
            } else {
                dataspace::Flags::NONE
            };
            match self {
                Self::One => dataspace::v1_with_widths(rank, flags, dimensions, maxima, widths),
                Self::Two => dataspace::v2_with_widths(
                    rank,
                    flags,
                    dataspace::Kind::SIMPLE,
                    dimensions,
                    maxima,
                    widths,
                ),
            }
        }

        fn header_size(self) -> usize {
            match self {
                Self::One => dataspace::V1_HEADER_LEN,
                Self::Two => dataspace::V2_HEADER_LEN,
            }
        }
    }

    #[rstest::rstest]
    #[case::version_one(true, dataspace::V1_HEADER_LEN)]
    #[case::version_two(false, dataspace::V2_HEADER_LEN)]
    fn dimension_fields_exclude_maximum_extents(
        #[case] version_one: bool,
        #[case] start: usize,
        #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] lengths: LengthWidth,
    ) {
        let widths = test_util::widths::Widths::new(8, usize::from(lengths.get()));
        let mut encoded = if version_one {
            dataspace::v1_with_widths(
                2,
                dataspace::Flags::MAX_DIMS,
                &[3, 5],
                Some(&[7, 9]),
                widths,
            )
        } else {
            dataspace::v2_with_widths(
                2,
                dataspace::Flags::MAX_DIMS,
                dataspace::Kind::SIMPLE,
                &[3, 5],
                Some(&[7, 9]),
                widths,
            )
        };
        let (parsed, fields) = Dataspace::parse_with_dimension_fields(&encoded, lengths).unwrap();
        assert_eq!(parsed.dimensions, [3, 5]);
        assert_eq!(
            parsed.max_dimensions,
            Some(vec![MaxExtent::Fixed(7), MaxExtent::Fixed(9)])
        );
        assert_eq!(fields, start..start + 2 * usize::from(lengths.get()));
        bytes::write_length_at(&mut encoded, fields.start, 6, lengths).unwrap();
        let changed = Dataspace::parse_with_length_width(&encoded, lengths).unwrap();
        assert_eq!(changed.dimensions, [6, 5]);
        assert_eq!(changed.max_dimensions, parsed.max_dimensions);
    }

    fn build_v1_dataspace(rank: u8, flags: u8, dims: &[u64], max_dims: Option<&[u64]>) -> Vec<u8> {
        dataspace::v1(rank, dataspace::Flags(flags), dims, max_dims)
    }

    fn build_v2_dataspace(
        rank: u8,
        flags: u8,
        type_byte: u8,
        dims: &[u64],
        max_dims: Option<&[u64]>,
    ) -> Vec<u8> {
        dataspace::v2(
            rank,
            dataspace::Flags(flags),
            dataspace::Kind(type_byte),
            dims,
            max_dims,
        )
    }

    #[test]
    fn scalar_v1() {
        let data = build_v1_dataspace(0, 0, &[], None);
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.space_type, DataspaceType::Scalar);
        assert_eq!(ds.rank, 0);
        assert!(ds.dimensions.is_empty());
        assert!(ds.max_dimensions.is_none());
        assert_eq!(ds.num_elements(), ElementCount::new(1));
    }

    #[test]
    fn null_v2() {
        let data = build_v2_dataspace(0, 0, 2, &[], None);
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.space_type, DataspaceType::Null);
        assert_eq!(ds.num_elements(), ElementCount::new(0));
    }

    #[test]
    fn simple_1d() {
        let data = build_v1_dataspace(1, 0, &[5], None);
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.space_type, DataspaceType::Simple);
        assert_eq!(ds.rank, 1);
        assert_eq!(ds.dimensions, vec![5]);
        assert!(ds.max_dimensions.is_none());
        assert_eq!(ds.num_elements(), ElementCount::new(5));
    }

    #[test]
    fn simple_2d() {
        let data = build_v1_dataspace(2, 0, &[3, 4], None);
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.rank, 2);
        assert_eq!(ds.dimensions, vec![3, 4]);
        assert_eq!(ds.num_elements(), ElementCount::new(12));
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
        assert_eq!(ds.num_elements(), ElementCount::new(24));
    }

    #[rstest::rstest]
    #[case::empty_simple(&[], 0)]
    #[case::zero(&[0], 0)]
    #[case::finite(&[3, 5], 15)]
    #[case::exact_limit(&[u64::MAX, 1], u64::MAX)]
    #[case::overflow(&[u64::MAX, 2], u64::MAX)]
    #[case::zero_after_overflow(&[u64::MAX, 2, 0], 0)]
    #[case::zero_before_overflow(&[0, u64::MAX, 2], 0)]
    fn cardinality_preserves_empty_and_saturated_products(
        #[case] dimensions: &[u64],
        #[case] expected: u64,
    ) {
        let data = dataspace::v2(
            u8::try_from(dimensions.len()).unwrap(),
            dataspace::Flags::NONE,
            dataspace::Kind::SIMPLE,
            dimensions,
            None,
        );
        let dataspace = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(dataspace.space_type, DataspaceType::Simple);
        assert_eq!(dataspace.num_elements(), ElementCount::new(expected));
    }

    #[test]
    fn num_elements_saturates_on_overflow() {
        // A malformed dataspace can declare dimensions whose product exceeds
        // u64::MAX. `num_elements()` must saturate on overflow, so the
        // callers can reject the impossibly large count as an error.
        let data = build_v1_dataspace(3, 0, &[1 << 40, 1 << 40, 1 << 40], None);
        let ds = Dataspace::parse(&data, 8).unwrap();
        assert_eq!(ds.num_elements(), ElementCount::new(u64::MAX));
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
        assert_eq!(ds.num_elements(), ElementCount::new(1));
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
        assert_eq!(
            err,
            FormatError::UnexpectedEof {
                expected: 4,
                available: 2
            }
        );
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
        let bytes = ds.serialize(8).unwrap();
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
        assert_eq!(
            err,
            "a fixed maximum of u64::MAX is the format's unlimited marker; \
                     name MaxExtent::Unlimited for a dimension that grows without bound"
        );
    }

    #[test]
    fn max_extent_reports_its_size_and_bounds_a_dimension() {
        assert_eq!(MaxExtent::Fixed(7).size(), Some(7));
        assert_eq!(MaxExtent::Unlimited.size(), None);
        assert!(!matches!(MaxExtent::Fixed(7), MaxExtent::Unlimited));
        assert!(matches!(MaxExtent::Unlimited, MaxExtent::Unlimited));
        assert!(max_extent_admits(MaxExtent::Fixed(7), 7));
        assert!(!max_extent_admits(MaxExtent::Fixed(7), 8));
        assert!(max_extent_admits(MaxExtent::Unlimited, u64::MAX));
    }
}
