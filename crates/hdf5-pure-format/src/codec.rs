//! Metadata decoding from caller-supplied byte regions.

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
