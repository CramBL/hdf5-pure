//! Decoding from borrowed byte cursors.

use crate::bytes::Fields;

/// A value decoded from the current position in a byte cursor.
///
/// Implementations leave following bytes available. The caller measures consumption from
/// [`Fields::pos`]. Individual [`Fields`] acquisitions leave the cursor unchanged on failure.
pub(crate) trait Decode<'de>: Sized {
    /// Information supplied by the caller to interpret the encoded fields.
    type Context;
    /// The error returned if decoding fails.
    type Error;

    /// Decodes one value and advances `fields` past its encoded bytes.
    ///
    /// # Errors
    ///
    /// Returns an implementation-specific error if decoding fails. The cursor may have advanced.
    fn decode(fields: &mut Fields<'de>, context: Self::Context) -> Result<Self, Self::Error>;
}
