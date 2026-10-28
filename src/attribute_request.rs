use crate::attribute::AttributeMessage;
use crate::error::FormatError;
use crate::type_builders::AttrSpec;

/// A nonempty attribute name without null bytes.
///
/// The name and its null terminator fit in the two-byte "Name Size" field of attribute message
/// versions 1, 2, and 3, defined in "The Attribute Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_attribute
pub(crate) struct AttributeName<'a>(&'a str);

impl AttributeName<'_> {
    /// Encodes or copies an attribute through [`AttrSpec::to_message`].
    ///
    /// Uses this name for decoded values and preserves the name of a copied message.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ValueTooLargeForRepresentation`] if a variable-length string
    /// exceeds [`u32::MAX`] bytes.
    pub(crate) fn to_message(&self, spec: &AttrSpec) -> Result<AttributeMessage, FormatError> {
        spec.to_message(self.0)
    }
}

impl<'a> TryFrom<&'a str> for AttributeName<'a> {
    type Error = FormatError;

    /// Parses an attribute name for a new file.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidAttributeName`] if `name` is empty or contains a null byte.
    /// Returns [`FormatError::AttributeFieldTooLong`] if `name` and its null terminator exceed
    /// [`u16::MAX`] bytes.
    fn try_from(name: &'a str) -> Result<Self, FormatError> {
        if name.is_empty() || name.as_bytes().contains(&0) {
            return Err(FormatError::InvalidAttributeName { name: name.into() });
        }
        if name.len() >= usize::from(u16::MAX) {
            return Err(FormatError::AttributeFieldTooLong {
                name: name.into(),
                field: "name",
                size: name.len().saturating_add(1),
                limit: usize::from(u16::MAX),
            });
        }
        Ok(Self(name))
    }
}
