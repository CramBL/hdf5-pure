use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use hdf5_pure_core::__private::{ByteLength, ElementCount, ElementSize};

use crate::bytes;
use crate::codec::DecodePrefix;
use crate::codec::Encode;
use crate::error::FormatError;
use crate::width;
use crate::width::LengthWidth;

use super::stored::{
    MAX_DIMENSIONS_FLAG, StoredDataspace, TYPE_NULL, TYPE_SCALAR, TYPE_SIMPLE, V2_HEADER_SIZE,
    VERSION_TWO,
};

/// Type of dataspace.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataspaceType {
    /// Scalar (single element).
    Scalar,
    /// Simple (N-dimensional array).
    Simple,
    /// Null (no data).
    Null,
}

impl DataspaceType {
    fn type_byte(self) -> u8 {
        match self {
            DataspaceType::Scalar => TYPE_SCALAR,
            DataspaceType::Simple => TYPE_SIMPLE,
            DataspaceType::Null => TYPE_NULL,
        }
    }
}

/// The shape of a dataset or attribute.
///
/// A simple dataspace has one to 255 dimensions. Each current dimension is finite
/// and at most its maximum. Scalar and null dataspaces have rank zero.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataspace {
    /// The type of this dataspace.
    space_type: DataspaceType,
    /// Current dimension sizes.
    dimensions: Vec<u64>,
    /// The maximum size of each dimension, when the dataspace message stores them.
    max_dimensions: Option<Vec<MaxExtent>>,
}

impl Dataspace {
    /// Creates a rank-zero dataspace containing one element.
    pub const fn scalar() -> Self {
        Self {
            space_type: DataspaceType::Scalar,
            dimensions: Vec::new(),
            max_dimensions: None,
        }
    }

    /// Creates a rank-zero dataspace containing no elements.
    pub const fn null() -> Self {
        Self {
            space_type: DataspaceType::Null,
            dimensions: Vec::new(),
            max_dimensions: None,
        }
    }

    /// Creates a simple dataspace from a nonempty list of dimensions and optional maxima.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::new`] when the rank or extents are invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_format::{__private::Dataspace, FormatError, MaxExtent};
    ///
    /// let maxima = vec![MaxExtent::Unlimited, MaxExtent::Fixed(3)];
    /// let mut space = Dataspace::simple(vec![0, 3], Some(maxima))?;
    /// assert!(space.is_empty());
    /// space.set_first_dimension(2)?;
    /// assert_eq!(space.num_elements()?.get(), 6);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn simple(
        dimensions: Vec<u64>,
        maxima: Option<Vec<MaxExtent>>,
    ) -> Result<Self, FormatError> {
        Self::new(DataspaceType::Simple, dimensions, maxima)
    }

    /// Creates a dataspace of `space_type` from its current dimensions and optional maxima.
    ///
    /// Omitting the maxima describes a fixed shape. A current dimension can be zero.
    /// The element count is computed on request by [`Self::num_elements`].
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DatasetRankUnsupported`] if the rank exceeds 255, or
    /// [`FormatError::InvalidDataspaceRank`] if the rank is invalid for the kind.
    /// Returns [`FormatError::InvalidDataspaceDimensionCount`] if the two lists differ in length.
    /// Returns [`FormatError::DatasetDimensionUnlimited`] for a current dimension of [`u64::MAX`],
    /// [`FormatError::InvalidDataspaceMaximum`] for a finite maximum of [`u64::MAX`], or
    /// [`FormatError::DataspaceMaximumTooSmall`] for a maximum below its current dimension.
    pub fn new(
        space_type: DataspaceType,
        dimensions: Vec<u64>,
        max_dimensions: Option<Vec<MaxExtent>>,
    ) -> Result<Self, FormatError> {
        let rank = u8::try_from(dimensions.len()).map_err(|_error| {
            FormatError::DatasetRankUnsupported {
                rank: dimensions.len(),
                maximum: u8::MAX,
            }
        })?;
        if matches!(space_type, DataspaceType::Simple) != (rank != 0) {
            return Err(FormatError::InvalidDataspaceRank {
                space_type: space_type.type_byte(),
                rank,
            });
        }
        if let Some(maxima) = max_dimensions.as_deref()
            && maxima.len() != dimensions.len()
        {
            return Err(FormatError::InvalidDataspaceDimensionCount {
                rank,
                dimensions: dimensions.len(),
                maximum_dimensions: Some(maxima.len()),
            });
        }
        for (dimension, &current) in dimensions.iter().enumerate() {
            Self::require_dimension(
                dimension,
                current,
                max_dimensions
                    .as_deref()
                    .and_then(|maxima| maxima.get(dimension))
                    .copied(),
            )?;
        }
        Ok(Self {
            space_type,
            dimensions,
            max_dimensions,
        })
    }

    /// Sets the current size of the first dimension, preserving any explicit maximum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DataspaceHasNoDimensions`] for a scalar or null dataspace,
    /// [`FormatError::DatasetDimensionUnlimited`] if `current` is [`u64::MAX`], or
    /// [`FormatError::DataspaceMaximumTooSmall`] if `current` exceeds the explicit maximum.
    /// Leaves the shape unchanged on error.
    pub fn set_first_dimension(&mut self, current: u64) -> Result<(), FormatError> {
        let maximum = self
            .max_dimensions
            .as_ref()
            .and_then(|maxima| maxima.first())
            .copied();
        let first =
            self.dimensions
                .first_mut()
                .ok_or_else(|| FormatError::DataspaceHasNoDimensions {
                    space_type: self.space_type.type_byte(),
                })?;
        Self::require_dimension(0, current, maximum)?;
        *first = current;
        Ok(())
    }

    /// Returns the scalar, simple, or null kind of this dataspace.
    pub const fn space_type(&self) -> DataspaceType {
        self.space_type
    }

    /// Returns the current size of each dimension in storage order.
    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }

    /// Returns the explicit maximum sizes, or [`None`] if the shape omits them.
    pub fn max_dimensions(&self) -> Option<&[MaxExtent]> {
        self.max_dimensions.as_deref()
    }

    /// Returns the number of dimensions with an unlimited maximum.
    pub fn unlimited_dimensions(&self) -> usize {
        self.max_dimensions()
            .unwrap_or_default()
            .iter()
            .filter(|max| matches!(max, MaxExtent::Unlimited))
            .count()
    }

    /// Parses a dataspace message body from raw bytes.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidLengthSize`] if `length_size` is not 2, 4, or 8, and otherwise
    /// returns the errors [`parse_with_length_width`](Self::parse_with_length_width) returns.
    pub fn parse(data: &[u8], length_size: u8) -> Result<Dataspace, FormatError> {
        Self::parse_with_length_width(data, LengthWidth::try_from(length_size)?)
    }

    /// Parses the dataspace at the start of `data` using the file's length width.
    ///
    /// Reads each declared dimension list, including version 1 permutation indices.
    /// Any bytes after the last field remain uninterpreted.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDataspaceVersion`] if the version is unsupported,
    /// [`FormatError::InvalidDataspaceType`] if a version 2 type byte is invalid, or
    /// [`FormatError::UnexpectedEof`] if the message ends within a field. Returns
    /// [`FormatError::UnsupportedDataspacePermutation`] if a version 1 permutation index
    /// differs from its zero-based dimension number. Propagates the shape errors of [`Self::new`].
    #[inline]
    pub fn parse_with_length_width(
        data: &[u8],
        length_width: LengthWidth,
    ) -> Result<Dataspace, FormatError> {
        StoredDataspace::decode_prefix(data, length_width)?
            .into_value()
            .into_dataspace()
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
    /// let dataspace = format::Dataspace::simple(vec![3], Some(vec![MaxExtent::Fixed(7)]))?;
    /// let mut body = dataspace.prepare(LengthWidth::Two)?.to_bytes();
    /// let (parsed, dimensions) = format::Dataspace::parse_with_dimension_fields(
    ///     &body, LengthWidth::Two,
    /// )?;
    /// assert_eq!(parsed, dataspace);
    /// assert_eq!(dimensions, 4..6);
    /// format::write_length_at(&mut body, dimensions.start, 6, LengthWidth::Two)?;
    /// let grown = format::Dataspace::parse_with_length_width(&body, LengthWidth::Two)?;
    /// assert_eq!(grown.dimensions(), [6]);
    /// assert_eq!(grown.max_dimensions(), Some([MaxExtent::Fixed(7)].as_slice()));
    /// # Ok::<(), FormatError>(())
    /// ```
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_simple
    pub fn parse_with_dimension_fields(
        data: &[u8],
        length_width: LengthWidth,
    ) -> Result<(Dataspace, Range<usize>), FormatError> {
        let stored = StoredDataspace::decode_prefix(data, length_width)?.into_value();
        let range = stored.dimension_range();
        Ok((stored.into_dataspace()?, range))
    }

    /// Borrows the dataspace for version 2 encoding at the file's length width.
    ///
    /// Checks the fields without allocating or writing bytes.
    ///
    /// An unlimited maximum requires eight-byte lengths. At two- and four-byte lengths,
    /// [`stored_max_extent`] converts an unlimited maximum to the largest fixed maximum.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::LengthTooLarge`] for the first dimension or maximum that
    /// exceeds `length_width`. Checks current dimensions first, then maximum dimensions.
    pub fn prepare(&self, length_width: LengthWidth) -> Result<PreparedDataspace<'_>, FormatError> {
        for length in self.lengths() {
            bytes::require_length(length, length_width)?;
        }
        Ok(PreparedDataspace {
            dataspace: self,
            length_width,
        })
    }

    /// Returns the number of elements in the dataspace.
    ///
    /// A scalar contains one element. A null dataspace or a simple dataspace with a zero
    /// dimension contains none.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::DataspaceElementCountOverflow`] if the product of the current
    /// dimensions exceeds [`u64::MAX`].
    pub fn num_elements(&self) -> Result<ElementCount, FormatError> {
        if self.is_empty() {
            return Ok(ElementCount::new(0));
        }
        self.dimensions
            .iter()
            .try_fold(ElementCount::new(1), |count, &dimension| {
                count.checked_mul(ElementCount::new(dimension)).ok_or(
                    FormatError::DataspaceElementCountOverflow {
                        count: count.get(),
                        dimension,
                    },
                )
            })
    }

    /// Returns the size in bytes of all elements stored at `element_size`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::num_elements`], or [`FormatError::OffsetOverflow`] if the
    /// element count times the element size exceeds [`u64::MAX`].
    pub fn byte_length(&self, element_size: ElementSize) -> Result<ByteLength, FormatError> {
        let count = self.num_elements()?;
        element_size
            .byte_length(count)
            .ok_or(FormatError::OffsetOverflow {
                offset: count.get(),
                length: element_size.get(),
            })
    }

    /// Returns `true` for a null dataspace or a simple dataspace with a zero dimension.
    pub fn is_empty(&self) -> bool {
        self.space_type == DataspaceType::Null || self.dimensions.contains(&0)
    }

    /// Returns the maximum dimensions, when some maximum exceeds its current dimension.
    ///
    /// A dataspace with no maximum dimensions describes a dataset of a fixed shape, and so
    /// does one whose every maximum is fixed at the current dimension: both give `None`.
    pub fn extensible_max_dimensions(&self) -> Option<&[MaxExtent]> {
        let max_dims = self.max_dimensions.as_deref()?;
        let fixed_at_current = max_dims
            .iter()
            .zip(&self.dimensions)
            .all(|(max, &dim)| max.size() == Some(dim));
        (!fixed_at_current).then_some(max_dims)
    }

    fn require_dimension(
        dimension: usize,
        current: u64,
        maximum: Option<MaxExtent>,
    ) -> Result<(), FormatError> {
        if current == UNLIMITED_LENGTH {
            return Err(FormatError::DatasetDimensionUnlimited { dimension });
        }
        match maximum {
            Some(MaxExtent::Fixed(UNLIMITED_LENGTH)) => {
                Err(FormatError::InvalidDataspaceMaximum { dimension })
            }
            Some(MaxExtent::Fixed(maximum)) if maximum < current => {
                Err(FormatError::DataspaceMaximumTooSmall {
                    dimension,
                    current,
                    maximum,
                })
            }
            _ => Ok(()),
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "construction bounds the number of dimensions to one byte and resizing preserves it"
    )]
    fn rank(&self) -> u8 {
        self.dimensions.len() as u8
    }

    fn lengths(&self) -> impl Iterator<Item = u64> + '_ {
        self.dimensions.iter().copied().chain(
            self.max_dimensions
                .as_deref()
                .unwrap_or_default()
                .iter()
                .copied()
                .map(max_extent_to_length),
        )
    }
}

/// A version 2 dataspace body ready to write at the selected length width.
///
/// [`Dataspace::prepare`] checks that each current and maximum dimension fits the selected
/// width. The writer can write the same body to multiple destinations.
#[derive(Debug)]
pub struct PreparedDataspace<'a> {
    dataspace: &'a Dataspace,
    length_width: LengthWidth,
}

impl PreparedDataspace<'_> {
    /// Returns the exact body size in bytes, including the four-byte version 2 prefix.
    pub fn encoded_size(&self) -> usize {
        self.encoded_len()
    }

    /// Writes the body to the start of `output` and returns the number of bytes written.
    ///
    /// Leaves bytes after the body untouched.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if `output` is shorter than [`Self::encoded_size`],
    /// leaving the entire slice untouched.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_format::{__private as format, FormatError, LengthWidth, MaxExtent};
    ///
    /// let dataspace = format::Dataspace::simple(vec![258], Some(vec![MaxExtent::Fixed(513)]))?;
    /// let prepared = dataspace.prepare(LengthWidth::Two)?;
    /// let mut output = [0xff; 9];
    /// let written = prepared.write_into(&mut output)?;
    ///
    /// assert_eq!(written, prepared.encoded_size());
    /// assert_eq!(output, [2, 1, 1, 1, 2, 1, 1, 2, 0xff]);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn write_into(&self, output: &mut [u8]) -> Result<usize, FormatError> {
        Ok(self.encode_into(output)?)
    }

    /// Writes the body to a newly allocated vector of [`Self::encoded_size`] bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut output = vec![0; self.encoded_len()];
        self.write_body(&mut output);
        output
    }

    /// Writes the dataspace into the first [`Self::encoded_size`] bytes of `body`.
    ///
    /// # Panics
    ///
    /// Panics if `body` is shorter than [`Self::encoded_size`].
    pub(crate) fn write_body(&self, body: &mut [u8]) {
        assert!(
            self.encode_into(body).is_ok(),
            "caller reserves the prepared dataspace size"
        );
    }

    fn write_lengths<const N: usize>(&self, fields: &mut [u8]) {
        const { assert!(N > 0 && N <= size_of::<u64>()) };
        let (fields, _) = fields.as_chunks_mut::<N>();
        for (field, length) in fields.iter_mut().zip(self.dataspace.lengths()) {
            field.copy_from_slice(&length.to_le_bytes()[..N]);
        }
    }
}

impl Encode for PreparedDataspace<'_> {
    type Error = bytes::UnexpectedEof;

    fn encoded_len(&self) -> usize {
        V2_HEADER_SIZE
            + usize::from(self.length_width.get())
                * usize::from(self.dataspace.rank())
                * (1 + usize::from(self.dataspace.max_dimensions.is_some()))
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        let expected = self.encoded_len();
        let available = output.len();
        let body = output
            .get_mut(..expected)
            .ok_or_else(|| bytes::UnexpectedEof::new(expected, available))?;
        let (header, fields) = body.split_at_mut(V2_HEADER_SIZE);
        let flags = if self.dataspace.max_dimensions.is_some() {
            MAX_DIMENSIONS_FLAG
        } else {
            0
        };
        header.copy_from_slice(&[
            VERSION_TWO,
            self.dataspace.rank(),
            flags,
            self.dataspace.space_type.type_byte(),
        ]);
        match self.length_width {
            LengthWidth::Two => self.write_lengths::<2>(fields),
            LengthWidth::Four => self.write_lengths::<4>(fields),
            LengthWidth::Eight => self.write_lengths::<8>(fields),
        }
        Ok(expected)
    }
}

pub(crate) use hdf5_pure_core::MaxExtent;

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

pub(super) fn max_extent_from_length(length: u64) -> MaxExtent {
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

#[cfg(test)]
mod tests {
    use test_util::dataspace;

    use super::*;

    #[rstest::rstest]
    #[case::empty(&[])]
    #[case::zero_dimension(&[0])]
    #[case::two_dimensions(&[3, 5])]
    #[case::three_dimensions(&[3, 5, u64::from(u16::MAX)])]
    fn dimension_arrays_preserve_values_ranges_and_suffixes(
        #[case] dimensions: &[u64],
        #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] width: LengthWidth,
        #[values(Version::One, Version::Two)] version: Version,
        #[values(false, true)] maxima: bool,
    ) {
        let maximum = width::length_width_max(width);
        let max_dimensions = maxima.then(|| vec![maximum; dimensions.len()]);
        let mut raw = version.body(dimensions, max_dimensions.as_deref(), width);
        if matches!(version, Version::One) {
            raw[3..8].fill(0xa5);
        }
        raw[2] |= 0xfc;
        let expected = Dataspace::new(
            if dimensions.is_empty() {
                DataspaceType::Scalar
            } else {
                DataspaceType::Simple
            },
            dimensions.to_vec(),
            maxima.then(|| {
                vec![
                    if maximum == u64::MAX {
                        MaxExtent::Unlimited
                    } else {
                        MaxExtent::Fixed(maximum)
                    };
                    dimensions.len()
                ]
            }),
        )
        .unwrap();
        let range = version.header_size()
            ..version.header_size() + dimensions.len() * usize::from(width.get());
        assert_eq!(
            Dataspace::parse_with_dimension_fields(&raw, width).unwrap(),
            (expected.clone(), range.clone())
        );
        raw.extend_from_slice(&[0xef, 0xbe, 0xad, 0xde]);
        assert_eq!(
            Dataspace::parse_with_dimension_fields(&raw, width).unwrap(),
            (expected, range)
        );
    }

    #[rstest::rstest]
    #[case::two(LengthWidth::Two)]
    #[case::four(LengthWidth::Four)]
    #[case::eight(LengthWidth::Eight)]
    fn truncated_dimension_arrays_report_the_first_incomplete_field(
        #[case] width: LengthWidth,
        #[values(Version::One, Version::Two)] version: Version,
    ) {
        let raw = version.body(&[3, 5, 7], Some(&[11, 13, 17]), width);
        let size = usize::from(width.get());
        for available in version.header_size()..raw.len() {
            let complete = (available - version.header_size()) / size;
            assert_eq!(
                Dataspace::parse_with_length_width(&raw[..available], width).unwrap_err(),
                FormatError::UnexpectedEof {
                    expected: version.header_size() + (complete + 1) * size,
                    available,
                }
            );
        }
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
    #[case::scalar(0)]
    #[case::null(2)]
    fn version_two_ranked_scalar_and_null_are_rejected(
        #[case] type_byte: u8,
        #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] width: LengthWidth,
    ) {
        let mut raw = Version::Two.body(&[3, 5], None, width);
        raw[3] = type_byte;
        assert_eq!(
            Dataspace::parse_with_length_width(&raw, width).unwrap_err(),
            FormatError::InvalidDataspaceRank {
                space_type: type_byte,
                rank: 2
            }
        );
    }

    #[rstest::rstest]
    fn version_one_permutation_completes_the_prefix(
        #[values(LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight)] width: LengthWidth,
        #[values(false, true)] maxima: bool,
    ) {
        let maxima = maxima.then_some([7, 9, 11]);
        let mut raw =
            Version::One.body(&[3, 5, 7], maxima.as_ref().map(<[u64; 3]>::as_slice), width);
        raw[2] |= dataspace::Flags::PERMUTATION.0;
        let size = usize::from(width.get());
        let permutation_at = raw.len();
        for index in [0u64, 1, 2] {
            raw.extend_from_slice(&index.to_le_bytes()[..size]);
        }
        let end = raw.len();
        raw.push(0xff);
        let parsed = StoredDataspace::decode_prefix(&raw, width).unwrap();
        assert_eq!(parsed.consumed(), end);
        assert_eq!(parsed.value().dimension_range(), 8..8 + 3 * size);
        let space = parsed.into_value().into_dataspace().unwrap();
        assert_eq!(space.dimensions, [3, 5, 7]);
        assert_eq!(
            space.max_dimensions,
            maxima.map(|values| values.map(MaxExtent::Fixed).to_vec())
        );
        for available in permutation_at..end {
            let complete = (available - permutation_at) / size;
            assert_eq!(
                Dataspace::parse_with_length_width(&raw[..available], width).unwrap_err(),
                FormatError::UnexpectedEof {
                    expected: permutation_at + (complete + 1) * size,
                    available
                }
            );
        }
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
                    if dimensions.is_empty() {
                        dataspace::Kind::SCALAR
                    } else {
                        dataspace::Kind::SIMPLE
                    },
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
        let ds = Dataspace::parse_with_length_width(&data, LengthWidth::Eight).unwrap();
        assert_eq!(ds.space_type, DataspaceType::Scalar);
        assert_eq!(ds.rank(), 0);
        assert!(ds.dimensions.is_empty());
        assert!(ds.max_dimensions.is_none());
        assert_eq!(ds.num_elements().unwrap(), ElementCount::new(1));
    }

    #[test]
    fn null_v2() {
        let data = build_v2_dataspace(0, 0, 2, &[], None);
        let ds = Dataspace::parse_with_length_width(&data, LengthWidth::Eight).unwrap();
        assert_eq!(ds.space_type, DataspaceType::Null);
        assert_eq!(ds.num_elements().unwrap(), ElementCount::new(0));
    }

    #[test]
    fn simple_1d() {
        let data = build_v1_dataspace(1, 0, &[5], None);
        let ds = Dataspace::parse_with_length_width(&data, LengthWidth::Eight).unwrap();
        assert_eq!(ds.space_type, DataspaceType::Simple);
        assert_eq!(ds.rank(), 1);
        assert_eq!(ds.dimensions, vec![5]);
        assert!(ds.max_dimensions.is_none());
        assert_eq!(ds.num_elements().unwrap(), ElementCount::new(5));
    }

    #[test]
    fn simple_2d() {
        let data = build_v1_dataspace(2, 0, &[3, 4], None);
        let ds = Dataspace::parse_with_length_width(&data, LengthWidth::Eight).unwrap();
        assert_eq!(ds.rank(), 2);
        assert_eq!(ds.dimensions, vec![3, 4]);
        assert_eq!(ds.num_elements().unwrap(), ElementCount::new(12));
    }

    #[test]
    fn simple_3d_with_max_dims_unlimited() {
        let data = build_v1_dataspace(3, 0x01, &[2, 3, 4], Some(&[10, u64::MAX, 100]));
        let ds = Dataspace::parse_with_length_width(&data, LengthWidth::Eight).unwrap();
        assert_eq!(ds.rank(), 3);
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
        assert_eq!(ds.num_elements().unwrap(), ElementCount::new(24));
    }

    #[rstest::rstest]
    #[case::zero(&[0], 0)]
    #[case::finite(&[3, 5], 15)]
    #[case::exact_limit(&[u64::MAX / 3, 3], u64::MAX)]
    #[case::zero_after_overflow(&[u64::MAX - 1, 2, 0], 0)]
    #[case::zero_before_overflow(&[0, u64::MAX - 1, 2], 0)]
    fn cardinality_preserves_zero_and_exact_products(
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
        let dataspace = Dataspace::parse_with_length_width(&data, LengthWidth::Eight).unwrap();
        assert_eq!(dataspace.space_type, DataspaceType::Simple);
        assert_eq!(
            dataspace.num_elements().unwrap(),
            ElementCount::new(expected)
        );
    }

    #[test]
    fn num_elements_reports_the_first_overflow() {
        let data = build_v1_dataspace(3, 0, &[1 << 40, 1 << 40, 1 << 40], None);
        let ds = Dataspace::parse_with_length_width(&data, LengthWidth::Eight).unwrap();
        assert_eq!(
            ds.num_elements().unwrap_err(),
            FormatError::DataspaceElementCountOverflow {
                count: 1 << 40,
                dimension: 1 << 40
            }
        );
    }

    #[test]
    fn v2_simple() {
        let data = build_v2_dataspace(1, 0, 1, &[7], None);
        let ds = Dataspace::parse_with_length_width(&data, LengthWidth::Eight).unwrap();
        assert_eq!(ds.space_type, DataspaceType::Simple);
        assert_eq!(ds.dimensions, vec![7]);
    }

    #[test]
    fn v2_scalar() {
        let data = build_v2_dataspace(0, 0, 0, &[], None);
        let ds = Dataspace::parse_with_length_width(&data, LengthWidth::Eight).unwrap();
        assert_eq!(ds.space_type, DataspaceType::Scalar);
        assert_eq!(ds.num_elements().unwrap(), ElementCount::new(1));
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
        let ds = Dataspace::parse_with_length_width(&buf, LengthWidth::Four).unwrap();
        assert_eq!(ds.dimensions, vec![10]);
    }

    #[test]
    fn truncated_data_error() {
        let data = [1u8, 2]; // too short
        let err = Dataspace::parse_with_length_width(&data, LengthWidth::Eight).unwrap_err();
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
        let err = Dataspace::parse_with_length_width(&data, LengthWidth::Eight).unwrap_err();
        assert_eq!(err, FormatError::InvalidDataspaceVersion(5));
    }

    #[test]
    fn invalid_v2_type_error() {
        let data = build_v2_dataspace(0, 0, 5, &[], None);
        let err = Dataspace::parse_with_length_width(&data, LengthWidth::Eight).unwrap_err();
        assert_eq!(err, FormatError::InvalidDataspaceType(5));
    }

    #[test]
    fn v1_with_max_dims() {
        let data = build_v1_dataspace(1, 0x01, &[5], Some(&[10]));
        let ds = Dataspace::parse_with_length_width(&data, LengthWidth::Eight).unwrap();
        assert_eq!(ds.max_dimensions, Some(vec![MaxExtent::Fixed(10)]));
    }

    #[test]
    fn all_ones_max_dimension_is_unlimited_only_at_eight_bytes() {
        let eight = build_v1_dataspace(1, 0x01, &[5], Some(&[u64::MAX]));
        let ds = Dataspace::parse_with_length_width(&eight, LengthWidth::Eight).unwrap();
        assert_eq!(ds.max_dimensions, Some(vec![MaxExtent::Unlimited]));

        // Four bytes cannot hold `H5S_UNLIMITED`: the C library zero-extends
        // the field, so all ones in four bytes is the finite maximum 2^32 - 1.
        let mut four = Vec::new();
        four.extend_from_slice(&[1, 1, 0x01, 0]);
        four.extend_from_slice(&[0u8; 4]);
        four.extend_from_slice(&5u32.to_le_bytes());
        four.extend_from_slice(&u32::MAX.to_le_bytes());
        let ds = Dataspace::parse_with_length_width(&four, LengthWidth::Four).unwrap();
        assert_eq!(
            ds.max_dimensions,
            Some(vec![MaxExtent::Fixed(u64::from(u32::MAX))])
        );
    }

    #[test]
    fn serialize_writes_all_ones_for_an_unlimited_maximum() {
        let ds = Dataspace::simple(
            vec![4, 6],
            Some(vec![MaxExtent::Unlimited, MaxExtent::Fixed(6)]),
        )
        .unwrap();
        let bytes = ds.prepare(LengthWidth::Eight).unwrap().to_bytes();
        assert_eq!(bytes[..4], [2, 2, 0x01, 1]);
        assert_eq!(bytes[20..28], u64::MAX.to_le_bytes());
        assert_eq!(bytes[28..36], 6u64.to_le_bytes());
        assert_eq!(
            Dataspace::parse_with_length_width(&bytes, LengthWidth::Eight).unwrap(),
            ds
        );
    }

    #[test]
    fn extensible_max_dimensions_skips_a_maximum_at_the_current_shape() {
        let fixed = Dataspace::simple(
            vec![4, 6],
            Some(vec![MaxExtent::Fixed(4), MaxExtent::Fixed(6)]),
        )
        .unwrap();
        assert_eq!(fixed.extensible_max_dimensions(), None);

        let none = Dataspace::simple(vec![4, 6], None).unwrap();
        assert_eq!(none.extensible_max_dimensions(), None);

        let grown = Dataspace::simple(
            vec![4, 6],
            Some(vec![MaxExtent::Fixed(4), MaxExtent::Unlimited]),
        )
        .unwrap();
        assert_eq!(
            grown.extensible_max_dimensions(),
            Some(&[MaxExtent::Fixed(4), MaxExtent::Unlimited][..])
        );
    }

    #[rstest::rstest]
    #[case::empty_simple(&[], None, FormatError::InvalidDataspaceRank { space_type: TYPE_SIMPLE, rank: 0 })]
    #[case::unlimited_current(&[3, u64::MAX], None, FormatError::DatasetDimensionUnlimited { dimension: 1 })]
    #[case::small_maximum(&[3, 5], Some([4, 2].as_slice()), FormatError::DataspaceMaximumTooSmall { dimension: 1, current: 5, maximum: 2 })]
    fn semantic_conversion_rejects_invalid_simple_extents(
        #[case] dimensions: &[u64],
        #[case] maxima: Option<&[u64]>,
        #[case] expected: FormatError,
    ) {
        let flags = if maxima.is_some() {
            dataspace::Flags::MAX_DIMS
        } else {
            dataspace::Flags::NONE
        };
        let bytes = dataspace::v2(
            u8::try_from(dimensions.len()).unwrap(),
            flags,
            dataspace::Kind::SIMPLE,
            dimensions,
            maxima,
        );
        let stored = StoredDataspace::decode_prefix(&bytes, LengthWidth::Eight).unwrap();
        assert_eq!(stored.consumed(), bytes.len());
        assert_eq!(stored.into_value().into_dataspace().unwrap_err(), expected);
    }

    #[test]
    fn finite_maximum_rejects_the_unlimited_marker() {
        assert_eq!(
            Dataspace::simple(vec![0], Some(vec![MaxExtent::Fixed(u64::MAX)])).unwrap_err(),
            FormatError::InvalidDataspaceMaximum { dimension: 0 }
        );
    }

    #[test]
    fn simple_dataspace_accepts_rank_255_and_rejects_rank_256() {
        assert_eq!(Dataspace::simple(vec![1; 255], None).unwrap().rank(), 255);
        assert_eq!(
            Dataspace::simple(vec![1; 256], None).unwrap_err(),
            FormatError::DatasetRankUnsupported {
                rank: 256,
                maximum: 255
            }
        );
    }

    #[rstest::rstest]
    #[case::scalar(Dataspace::scalar(), TYPE_SCALAR)]
    #[case::null(Dataspace::null(), TYPE_NULL)]
    fn updating_a_missing_dimension_reports_the_dataspace_kind(
        #[case] mut dataspace: Dataspace,
        #[case] space_type: u8,
    ) {
        assert_eq!(
            dataspace.set_first_dimension(1).unwrap_err(),
            FormatError::DataspaceHasNoDimensions { space_type }
        );
    }

    #[test]
    fn updating_the_first_dimension_preserves_the_shape_on_error() {
        let mut shape = Dataspace::simple(
            vec![3, 4],
            Some(vec![MaxExtent::Fixed(5), MaxExtent::Unlimited]),
        )
        .unwrap();
        assert_eq!(
            shape.set_first_dimension(6).unwrap_err(),
            FormatError::DataspaceMaximumTooSmall {
                dimension: 0,
                current: 6,
                maximum: 5
            }
        );
        assert_eq!(shape.dimensions(), [3, 4]);
        shape.set_first_dimension(5).unwrap();
        assert_eq!(shape.dimensions(), [5, 4]);
    }

    #[test]
    fn max_extent_reports_its_size_and_bounds_a_dimension() {
        assert_eq!(MaxExtent::Fixed(7).size(), Some(7));
        assert_eq!(MaxExtent::Unlimited.size(), None);
        assert!(!matches!(MaxExtent::Fixed(7), MaxExtent::Unlimited));
        assert!(matches!(MaxExtent::Unlimited, MaxExtent::Unlimited));
    }
}
