//! String padding and character encodings.
//!
//! "The Datatype Message" of the [format specification, version 4.0][spec] defines the padding
//! and character-set fields for fixed-length and variable-length strings.
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage

use core::fmt;

/// The treatment of unused bytes in a string element.
#[derive(Debug, Clone, PartialEq)]
pub enum StringPadding {
    /// Terminates a shortened string with a null byte.
    NullTerminate,
    /// Pads unused bytes with null bytes.
    NullPad,
    /// Pads unused bytes with spaces.
    SpacePad,
}

impl fmt::Display for StringPadding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Self::NullTerminate => "null-term",
            Self::NullPad => "null-pad",
            Self::SpacePad => "space-pad",
        })
    }
}

/// The character encoding of a string element.
#[derive(Debug, Clone, PartialEq)]
pub enum CharacterSet {
    /// Uses ASCII encoding.
    Ascii,
    /// Uses UTF-8 encoding.
    Utf8,
}

impl fmt::Display for CharacterSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Self::Ascii => "ascii",
            Self::Utf8 => "utf8",
        })
    }
}
