use core::fmt;

use crate::address::PhysicalAddress;
use crate::address_range::AddressRange;

/// A physical bounds, overlap, or truncation conflict during editing.
///
/// All addresses are byte offsets from the beginning of the file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EditStorageConflict {
    /// A range extends past the permitted end.
    Bounds {
        /// The range whose end exceeds `end`.
        range: AddressRange<PhysicalAddress>,
        /// The upper bound on the range end.
        end: PhysicalAddress,
    },
    /// Two physical ranges overlap.
    Overlap {
        first: AddressRange<PhysicalAddress>,
        second: AddressRange<PhysicalAddress>,
    },
    /// A proposed file end is below the required minimum.
    Truncation {
        /// The proposed end of the file.
        proposed: PhysicalAddress,
        /// The minimum permitted file end.
        required: PhysicalAddress,
    },
}

impl fmt::Display for EditStorageConflict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bounds { range, end } => write!(
                f,
                "range {}..{} extends past {}",
                range.start().get(),
                range.end().get(),
                end.get()
            ),
            Self::Overlap { first, second } => write!(
                f,
                "ranges {}..{} and {}..{} overlap",
                first.start().get(),
                first.end().get(),
                second.start().get(),
                second.end().get()
            ),
            Self::Truncation { proposed, required } => write!(
                f,
                "proposed end {} precedes required end {}",
                proposed.get(),
                required.get()
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for EditStorageConflict {}
