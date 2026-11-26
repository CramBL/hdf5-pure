use hdf5_pure_core::{__private, CharacterSet, Datatype, StringPadding};

use super::string;
use crate::error::FormatError;
use crate::width::OffsetWidth;

/// Returns `true` if the datatype contains a variable-length string or sequence.
///
/// Examines array elements, enumeration bases, and compound members recursively.
pub fn datatype_holds_variable_length(datatype: &Datatype) -> bool {
    match datatype {
        Datatype::VariableLength(_) => true,
        Datatype::Array { base_type, .. } => datatype_holds_variable_length(base_type),
        Datatype::Enumeration(datatype) => datatype_holds_variable_length(datatype.base_type()),
        Datatype::Compound { members, .. } => members
            .iter()
            .any(|member| datatype_holds_variable_length(&member.datatype)),
        _ => false,
    }
}

/// Derives the heap address width from the variable-length descriptor's declared size.
///
/// # Errors
///
/// Returns [`FormatError::InvalidVariableLengthSize`] if the descriptor size is not 10, 12,
/// or 16 bytes.
pub(super) fn descriptor_width(size: u32) -> Result<OffsetWidth, FormatError> {
    [OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight]
        .into_iter()
        .find(|&offsets| __private::variable_length_descriptor_size(offsets) == size)
        .ok_or(FormatError::InvalidVariableLengthSize { size })
}

/// A checked subtype with padding and character-set metadata for strings.
pub(super) enum VariableLengthMetadata {
    Sequence,
    String {
        padding: StringPadding,
        charset: CharacterSet,
    },
}

/// Parses a variable-length class bit field.
///
/// # Errors
///
/// Returns [`FormatError::InvalidVariableLengthSubtype`] for a subtype other than sequence or
/// string, [`FormatError::InvalidVariableLengthReservedBits`] if a reserved bit is set, or
/// [`FormatError::InvalidVariableLengthSequenceMetadata`] if a sequence has nonzero string
/// metadata. Propagates padding and character-set errors for strings.
pub(super) fn parse_variable_length_metadata(
    flags: [u8; 3],
) -> Result<VariableLengthMetadata, FormatError> {
    let [bf0, bf1, bf2] = flags;
    let subtype = bf0 & VARIABLE_LENGTH_FIELD_MASK;
    if !matches!(subtype, VARIABLE_LENGTH_SEQUENCE | VARIABLE_LENGTH_STRING) {
        return Err(FormatError::InvalidVariableLengthSubtype(subtype));
    }
    let bits = u32::from_le_bytes([bf0, bf1, bf2, 0]) & VARIABLE_LENGTH_RESERVED_MASK;
    if bits != 0 {
        return Err(FormatError::InvalidVariableLengthReservedBits(bits));
    }
    let padding = bf0 >> VARIABLE_LENGTH_PADDING_SHIFT;
    let charset = bf1 & VARIABLE_LENGTH_FIELD_MASK;
    if subtype == VARIABLE_LENGTH_STRING {
        return Ok(VariableLengthMetadata::String {
            padding: string::parse_string_padding(padding)?,
            charset: string::parse_charset(charset)?,
        });
    }
    if padding != 0 || charset != 0 {
        return Err(FormatError::InvalidVariableLengthSequenceMetadata { padding, charset });
    }
    Ok(VariableLengthMetadata::Sequence)
}

/// The mask for a four-bit variable-length subtype or character-set code.
///
/// The fields for datatype message versions 1 through 5 are defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const VARIABLE_LENGTH_FIELD_MASK: u8 = 0x0F;
/// The variable-length sequence subtype code.
///
/// The code for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) const VARIABLE_LENGTH_SEQUENCE: u8 = 0;
/// The variable-length string subtype code.
///
/// The code for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) const VARIABLE_LENGTH_STRING: u8 = 1;
/// The bit position of string padding in the variable-length class bit field.
///
/// The field for datatype message versions 1 through 5 is defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
pub(super) const VARIABLE_LENGTH_PADDING_SHIFT: u8 = 4;
/// The mask for reserved bits 12 through 23 of the variable-length class bit field.
///
/// The bits for datatype message versions 1 through 5 are defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const VARIABLE_LENGTH_RESERVED_MASK: u32 = 0xFF_F000;
