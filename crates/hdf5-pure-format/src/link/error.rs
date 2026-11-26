use core::error::Error;
use core::fmt;

use crate::error::FormatError;
use crate::width::UintWidth;

/// An error constructing, decoding, or encoding a version 1 Link message.
///
/// [`Error::source`] returns the wrapped [`FormatError`] for [`Self::Format`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LinkWireError {
    /// An error in a field, the message framing, or the encoded length.
    Format(FormatError),
    /// The name length does not fit the selected field width.
    NameWidth {
        /// The name length in bytes.
        length: usize,
        /// The selected width of the name-length field.
        width: UintWidth,
    },
    /// The encoded target value exceeds its two-byte length field. Holds its length in bytes.
    TargetLength(usize),
    /// Bytes follow the target in a complete Link message body.
    TrailingBytes,
}

impl fmt::Display for LinkWireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NameWidth { length, width } => write!(
                f,
                "link name length {length} exceeds its {}-byte field",
                width.get()
            ),
            Self::Format(error) => fmt::Display::fmt(error, f),
            Self::TargetLength(length) => {
                write!(f, "link target length {length} exceeds its two-byte field")
            }
            Self::TrailingBytes => f.write_str("trailing bytes after link message fields"),
        }
    }
}

impl Error for LinkWireError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::NameWidth { .. } | Self::TargetLength(_) | Self::TrailingBytes => None,
            Self::Format(error) => Some(error),
        }
    }
}

impl From<FormatError> for LinkWireError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}
