//! Metadata decoding and bounded encoding.

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
/// Each implementation defines byte order and wire layout.
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
