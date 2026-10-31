use crate::address::AddressContext;

use super::Source;

/// An opened backend in its original source coordinates, with its effective address context.
///
/// Implementations use offsets from the beginning of the original source and obtain the address
/// context from the owning file or write engine. Resident bytes cover that same complete source,
/// including any userblock prefix. Typed source-view construction requires this interface.
pub(crate) trait OriginalSource: Source {
    /// Returns the owning image's effective stored-to-physical mapping.
    fn addresses(&self) -> AddressContext;

    /// Borrows the complete original image when the backend exposes resident bytes.
    fn resident_bytes(&self) -> Option<&[u8]>;
}
