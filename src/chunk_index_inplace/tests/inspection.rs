use crate::chunk_index_inplace::{AppendIndex, Located};
use crate::error::Error;

impl AppendIndex {
    /// Returns allocated Extensible Array metadata by value.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] for a finite index or an unallocated Extensible Array.
    pub(super) fn into_allocated(self) -> Result<Located, Error> {
        match self {
            Self::Allocated(located) => Ok(located),
            Self::Fixed(_) | Self::Single(_) | Self::Unallocated(_) => Err(
                Error::AppendUnsupported("allocated Extensible Array metadata is required"),
            ),
        }
    }

    /// Borrows allocated Extensible Array metadata.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AppendUnsupported`] for a finite index or an unallocated Extensible Array.
    pub(crate) fn allocated(&self) -> Result<&Located, Error> {
        match self {
            Self::Allocated(located) => Ok(located),
            Self::Fixed(_) | Self::Single(_) | Self::Unallocated(_) => Err(
                Error::AppendUnsupported("allocated Extensible Array metadata is required"),
            ),
        }
    }
}
