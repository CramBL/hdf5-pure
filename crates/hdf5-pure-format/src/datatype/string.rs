//! String padding and character-set fields shared by fixed- and variable-length strings.
//!
//! "The Datatype Message" of the [format specification, version 4.0][spec] defines the codes.
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage

use hdf5_pure_core::{CharacterSet, StringPadding};

use crate::error::FormatError;

/// Parses a four-bit padding code for a fixed- or variable-length string.
///
/// # Errors
///
/// Returns [`FormatError::InvalidStringPadding`] for a code other than 0, 1, or 2.
pub(super) fn parse_string_padding(val: u8) -> Result<StringPadding, FormatError> {
    match val {
        0 => Ok(StringPadding::NullTerminate),
        1 => Ok(StringPadding::NullPad),
        2 => Ok(StringPadding::SpacePad),
        _ => Err(FormatError::InvalidStringPadding(val)),
    }
}

/// Parses a four-bit character-set code for a fixed- or variable-length string.
///
/// # Errors
///
/// Returns [`FormatError::InvalidCharacterSet`] for a code other than 0 or 1.
pub(super) fn parse_charset(val: u8) -> Result<CharacterSet, FormatError> {
    match val {
        0 => Ok(CharacterSet::Ascii),
        1 => Ok(CharacterSet::Utf8),
        _ => Err(FormatError::InvalidCharacterSet(val)),
    }
}

/// Encodes the padding code for fixed- and variable-length strings.
///
/// Returns 0, 1, or 2 for [`StringPadding::NullTerminate`], [`StringPadding::NullPad`],
/// or [`StringPadding::SpacePad`], respectively. The fields are defined in "The Datatype
/// Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) fn padding_code(padding: &StringPadding) -> u8 {
    match padding {
        StringPadding::NullTerminate => NULL_TERMINATE,
        StringPadding::NullPad => NULL_PAD,
        StringPadding::SpacePad => SPACE_PAD,
    }
}

/// Encodes the character-set code for fixed- and variable-length strings.
///
/// Returns 0 for [`CharacterSet::Ascii`] and 1 for [`CharacterSet::Utf8`]. The fields
/// are defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) fn charset_code(charset: &CharacterSet) -> u8 {
    match charset {
        CharacterSet::Ascii => ASCII_CHARSET,
        CharacterSet::Utf8 => UTF8_CHARSET,
    }
}

/// The string padding code for NUL termination.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const NULL_TERMINATE: u8 = 0;
/// The string padding code for NUL padding.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const NULL_PAD: u8 = 1;
/// The string padding code for space padding.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const SPACE_PAD: u8 = 2;
/// The ASCII string character-set code.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const ASCII_CHARSET: u8 = 0;
/// The UTF-8 string character-set code.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const UTF8_CHARSET: u8 = 1;
