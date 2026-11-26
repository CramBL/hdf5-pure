use crate::error::FormatError;

/// The number of enclosing datatype descriptions, from zero through 63.
#[derive(Clone, Copy)]
pub(super) struct MessageDepth(u8);

impl MessageDepth {
    /// Returns the depth of a child description within the checked nesting limit.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::NestingDepthExceeded`] if the child would exceed 64 descriptions
    /// on one parent chain, including the outer description.
    pub(super) fn child(self) -> Result<Self, FormatError> {
        self.0
            .checked_add(1)
            .filter(|&depth| depth < MAX_MESSAGE_DEPTH)
            .map(Self)
            .ok_or(FormatError::NestingDepthExceeded)
    }

    /// The depth of the outer description.
    pub(super) const ROOT: Self = Self(0);
}

/// The maximum number of descriptions in a chain during parsing or retained encoding.
const MAX_MESSAGE_DEPTH: u8 = 64;
