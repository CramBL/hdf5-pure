use crate::address::AddressContext;

use super::Source;

/// An opened backend in its original source coordinates, with its effective address context.
///
/// Only the opened-file backend wrapper implements this sealed interface. Reads use offsets
/// from the beginning of the original source, and the address context belongs to the owning
/// file or write engine. Resident bytes include the complete source and any userblock prefix.
pub(crate) trait OriginalSource: sealed::Sealed + Source {
    /// Returns the owning image's effective stored-to-physical mapping.
    fn addresses(&self) -> AddressContext;

    /// Borrows the complete original image when the backend exposes resident bytes.
    fn resident_bytes(&self) -> Option<&[u8]>;
}

mod sealed {
    #[cfg(feature = "std")]
    use crate::reader::OpenedSource;

    pub trait Sealed {}

    #[cfg(feature = "std")]
    impl Sealed for OpenedSource<'_> {}
}
