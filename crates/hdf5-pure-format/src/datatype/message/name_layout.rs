//! Name field lengths for compound and enumeration members.

use super::version::DatatypeVersion;

/// The name field layout for compound and enumeration members.
///
/// libhdf5 omits name padding from version 3 onward (`H5O__dtype_decode_helper` and
/// `H5O__dtype_encode_helper` in `H5Odtype.c`, HDF5 2.2.0).
#[derive(Clone, Copy)]
pub(super) enum NameLayout {
    /// The field length is a multiple of eight bytes, including the terminator and padding.
    Padded,
    /// The field ends at the NUL terminator.
    Unpadded,
}

impl NameLayout {
    /// Selects padding for versions 1 and 2.
    pub(super) fn for_version(version: DatatypeVersion) -> Self {
        match version {
            DatatypeVersion::One | DatatypeVersion::Two => Self::Padded,
            DatatypeVersion::Three | DatatypeVersion::Four | DatatypeVersion::Five => {
                Self::Unpadded
            }
        }
    }

    /// Returns the field length, including the terminator and any padding.
    pub(super) fn field_len(self, name_len: usize) -> usize {
        name_len + self.suffix_len(name_len)
    }

    /// Returns the number of bytes for the terminator and any padding.
    pub(super) fn suffix_len(self, name_len: usize) -> usize {
        match self {
            Self::Padded => NAME_ALIGNMENT - name_len % NAME_ALIGNMENT,
            Self::Unpadded => 1,
        }
    }
}

// "The Datatype Message" in format specification version 4.0 defines this alignment in
// bytes for terminated compound and enumeration names in versions 1 and 2.
pub(super) const NAME_ALIGNMENT: usize = 8;
