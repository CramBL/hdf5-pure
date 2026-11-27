//! Metadata decoding and bounded encoding.

use alloc::format;
use core::cmp::Ordering;

use hdf5_pure_core::FormatError;

use crate::bytes::ByteCursor;

/// Decodes one value at the start of a byte slice.
///
/// Reads only the bytes needed to decode the value and returns their count. The value may
/// borrow from the input. Count all fields, internal alignment, and trailers of the selected
/// unit, including fields omitted from the result. Measure the count and retained ranges from
/// the start of the supplied unit.
///
/// A consuming caller checks the count and advances its cursor after decoding and any fallible
/// projection succeed. A caller reading a declared field with a known end can use
/// [`DecodeFrame`] to check the whole field.
pub(crate) trait DecodePrefix<'input>: Sized {
    /// The field widths or other information required to interpret the input.
    type Context;

    /// The error returned when decoding fails.
    type Error;

    /// Reads a value at the start of `input` and returns it with the number of bytes consumed.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if the prefix is incomplete or invalid for `context`.
    fn decode_prefix(
        input: &'input [u8],
        context: Self::Context,
    ) -> Result<Parsed<Self>, Self::Error>;
}

/// Decodes a value from a slice whose boundaries are known to the caller.
///
/// The decoder accounts for every byte in the frame, including any payload, checksum, or
/// padding defined by the format. A structural decoder can retain a delimited payload for
/// later interpretation. Document which fields are checked during decoding and which checks
/// require a subsequent operation. The value may borrow from the frame or a layout supplied
/// through [`Self::Context`].
///
/// Implement this trait independently of [`DecodePrefix`]. Use [`decode_exact`] when the
/// frame consists of one prefix and trailing bytes are checked after prefix decoding.
pub(crate) trait DecodeFrame<'input>: Sized {
    /// The field widths or other information required to interpret the input.
    type Context;

    /// The error returned when decoding fails.
    type Error;

    /// Reads a value using the framing rules for its type.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if the frame is incomplete, contains bytes outside its layout,
    /// or is invalid for `context`.
    fn decode_frame(frame: &'input [u8], context: Self::Context) -> Result<Self, Self::Error>;
}

/// Decodes a prefix and checks that it occupies the entire frame.
///
/// # Errors
///
/// Returns [`ExactError::Decode`] if the prefix decoder fails, [`ExactError::TrailingBytes`]
/// if bytes follow the value, or [`ExactError::InvalidConsumption`] if the decoder reports
/// more bytes than the caller supplied.
pub(crate) fn decode_exact<'input, T>(
    frame: &'input [u8],
    context: T::Context,
) -> Result<T, ExactError<T::Error>>
where
    T: DecodePrefix<'input>,
{
    let parsed = T::decode_prefix(frame, context).map_err(ExactError::Decode)?;
    let consumed = parsed.consumed();
    match consumed.cmp(&frame.len()) {
        Ordering::Equal => Ok(parsed.into_value()),
        Ordering::Less => Err(ExactError::TrailingBytes {
            consumed,
            frame_len: frame.len(),
        }),
        Ordering::Greater => Err(ExactError::InvalidConsumption(InvalidConsumption {
            reported: consumed,
            available: frame.len(),
        })),
    }
}

/// An error decoding a single value from an exact frame.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum ExactError<E> {
    Decode(E),
    InvalidConsumption(InvalidConsumption),
    TrailingBytes { consumed: usize, frame_len: usize },
}

impl From<ExactError<FormatError>> for FormatError {
    fn from(error: ExactError<FormatError>) -> Self {
        match error {
            ExactError::Decode(error) => error,
            ExactError::TrailingBytes {
                consumed,
                frame_len,
            } => Self::TrailingMetadataBytes {
                consumed,
                frame_len,
            },
            ExactError::InvalidConsumption(error) => Self::Internal(format!(
                "metadata decoder consumed {} bytes from a {}-byte frame",
                error.reported, error.available
            )),
        }
    }
}

/// A decoder reported consuming more bytes than its input contained.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct InvalidConsumption {
    pub(crate) reported: usize,
    pub(crate) available: usize,
}

/// A value and the number of bytes consumed from the original input.
///
/// Construct this from the value and the slice consumed by its decoder, or use
/// [`Self::from_cursor`] for a cursor positioned at the end of the value. The decoder constructs
/// it after reading the whole value, from a cursor or slice over its own input. [`Self::map`]
/// and [`Self::try_map`] preserve the input count through conversion, even if the new value has
/// a different encoded length.
#[must_use]
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Parsed<T> {
    value: T,
    consumed: usize,
}

impl<T> Parsed<T> {
    /// Pairs `value` with the cursor position measured from the start of the encoded value.
    ///
    /// Initialize the cursor at the first byte of the value and pass it after reading the last.
    pub(crate) fn from_cursor(value: T, cursor: &ByteCursor<'_>) -> Self {
        Self {
            value,
            consumed: cursor.position(),
        }
    }

    pub(crate) fn value(&self) -> &T {
        &self.value
    }

    pub(crate) fn consumed(&self) -> usize {
        self.consumed
    }

    pub(crate) fn into_value(self) -> T {
        self.value
    }

    pub(crate) fn into_parts(self) -> (T, usize) {
        (self.value, self.consumed)
    }

    /// Applies `project` to the value and preserves the consumed byte count.
    pub(crate) fn map<U>(self, project: impl FnOnce(T) -> U) -> Parsed<U> {
        let (value, consumed) = self.into_parts();
        Parsed {
            value: project(value),
            consumed,
        }
    }

    /// Applies a fallible conversion to the value and preserves the consumed byte count on success.
    ///
    /// # Errors
    ///
    /// Returns the error from `project` if the conversion fails.
    pub(crate) fn try_map<U, E>(
        self,
        project: impl FnOnce(T) -> Result<U, E>,
    ) -> Result<Parsed<U>, E> {
        let (value, consumed) = self.into_parts();
        Ok(Parsed {
            value: project(value)?,
            consumed,
        })
    }
}

impl<T> From<(T, &[u8])> for Parsed<T> {
    fn from((value, prefix): (T, &[u8])) -> Self {
        Self {
            value,
            consumed: prefix.len(),
        }
    }
}

/// Interprets a byte region according to the decoded type's framing policy.
///
/// Each implementation defines framing, byte order, alignment requirements, and trailing-byte
/// handling. Successful decoding can leave input bytes unexamined. The caller receives the decoded
/// value without a separate byte count. A decoded value can contain references with the input
/// lifetime `'input`.
///
/// See the implementations for [`SymbolTableMessage`], [`Dataspace`], and [`FileSpaceInfo`] for
/// message framing policies.
///
/// [`SymbolTableMessage`]: crate::symbol_table::SymbolTableMessage
/// [`Dataspace`]: crate::dataspace::Dataspace
/// [`FileSpaceInfo`]: crate::FileSpaceInfo
pub(crate) trait Decode<'input>: Sized {
    /// The field widths or other information required to interpret the input.
    type Context;

    /// The error returned when decoding fails.
    type Error;

    /// Applies the type's framing policy to `input` with `context`.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if the input cannot be decoded with the supplied context.
    fn decode(input: &'input [u8], context: Self::Context) -> Result<Self, Self::Error>;
}

/// Writes complete encodings into bounded byte slices.
///
/// Construction or fallible preparation establishes that numeric fields fit their retained widths.
/// Preparation also establishes counts, geometry, and placement when the layout requires them.
/// Each implementation defines byte order and wire layout. An aggregate encoder must leave
/// the complete output unchanged on error, including bytes reserved for earlier children.
///
/// See [`VariableLengthDescriptor`] and [`SymbolTableMessage::write_into`] for encoding at checked
/// field widths.
///
/// [`VariableLengthDescriptor`]: crate::variable_length_descriptor::VariableLengthDescriptor
/// [`SymbolTableMessage::write_into`]: crate::symbol_table::SymbolTableMessage::write_into
pub(crate) trait Encode {
    /// The error returned when encoding fails.
    type Error;

    /// Returns the exact number of bytes written by a successful encoding.
    ///
    /// This operation is infallible.
    fn encoded_len(&self) -> usize;

    /// Writes the complete encoding at the start of `output`.
    ///
    /// Returns the length reported by [`Self::encoded_len`] and preserves any output suffix.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if the complete encoding cannot be written, leaving `output`
    /// unchanged.
    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error>;
}

impl<T: Encode + ?Sized> Encode for &T {
    type Error = T::Error;

    fn encoded_len(&self) -> usize {
        T::encoded_len(self)
    }

    fn encode_into(&self, output: &mut [u8]) -> Result<usize, Self::Error> {
        T::encode_into(self, output)
    }
}

#[cfg(test)]
mod tests {
    use core::convert::Infallible;

    use super::{DecodePrefix, ExactError, InvalidConsumption, Parsed};
    use crate::link::LinkMessageView;
    use crate::width::OffsetWidth;

    #[test]
    fn link_prefix_length_and_address_range_are_relative_to_the_body() {
        let envelope = b"head\x01\x00\x01x\x34\x12tail";
        let body = &envelope[4..10];
        let exact = super::decode_exact::<LinkMessageView<'_>>(body, OffsetWidth::Two).unwrap();
        let parsed = LinkMessageView::decode_prefix(&envelope[4..], OffsetWidth::Two).unwrap();
        assert_eq!(parsed.value(), &exact);
        assert_eq!(parsed.consumed(), body.len());
        assert_eq!(exact.hard_link_address_range(), Some(4..6));
        assert_eq!(
            super::decode_exact::<LinkMessageView<'_>>(&envelope[4..], OffsetWidth::Two)
                .unwrap_err(),
            ExactError::TrailingBytes {
                consumed: 6,
                frame_len: 10
            }
        );
    }

    #[test]
    fn projecting_a_value_preserves_the_acquired_prefix_extent() {
        let prefix = b"\x01x";
        let parsed = Parsed::from((&prefix[1..], prefix.as_slice()));
        let text = parsed.try_map(core::str::from_utf8).unwrap();
        assert_eq!(text.into_parts(), ("x", 2));
    }

    #[test]
    fn exact_completion_reports_consumption_beyond_the_frame() {
        #[derive(Debug)]
        struct Overclaim;

        impl DecodePrefix<'_> for Overclaim {
            type Context = ();
            type Error = Infallible;

            fn decode_prefix(_input: &[u8], (): ()) -> Result<Parsed<Self>, Self::Error> {
                Ok(Parsed::from((Self, b"other".as_slice())))
            }
        }

        assert_eq!(
            super::decode_exact::<Overclaim>(b"ab", ()).unwrap_err(),
            ExactError::InvalidConsumption(InvalidConsumption {
                reported: 5,
                available: 2
            })
        );
    }
}
