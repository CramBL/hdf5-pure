use super::frame::FractalHeapIdLayout;
use crate::error::FormatError;

/// A borrowed heap ID parsed against its owning heap's layout.
///
/// Retains the original bytes, including any padding.
pub(crate) struct PreparedHeapId<'a> {
    bytes: &'a [u8],
}

impl<'a> PreparedHeapId<'a> {
    /// Parses the ID with `layout` and borrows its original encoding.
    ///
    /// # Errors
    ///
    /// Returns errors from [`FractalHeapIdLayout::parse`].
    pub(crate) fn new(layout: &FractalHeapIdLayout, bytes: &'a [u8]) -> Result<Self, FormatError> {
        layout.parse(bytes)?;
        Ok(Self { bytes })
    }

    /// Returns the complete encoded ID.
    pub(crate) fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

#[cfg(test)]
#[path = "tests/prepared.rs"]
mod tests;
