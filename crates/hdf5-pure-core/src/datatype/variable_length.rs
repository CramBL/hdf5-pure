use alloc::boxed::Box;

use super::definition::Datatype;
use super::layout::FixedPointType;
use super::string::{CharacterSet, StringPadding};
use crate::FormatError;

/// A variable-length sequence or string.
///
/// Strings have required padding and character-set metadata. Sequences hold their element
/// datatype.
///
/// The variable-length class for datatype message versions 1 through 5 is defined in "The
/// Datatype Message" of the [format specification, version 4.0][spec].
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::{CharacterSet, Datatype, StringPadding, VariableLengthType};
///
/// let element = Datatype::String {
///     size: 1,
///     padding: StringPadding::NullTerminate,
///     charset: CharacterSet::Ascii,
/// };
/// let sequence = VariableLengthType::sequence(element);
/// let VariableLengthType::Sequence { element_type } = sequence else {
///     panic!("expected a sequence");
/// };
/// assert_eq!(element_type.type_size(), 1);
///
/// let datatype = Datatype::VariableLength(VariableLengthType::string(
///     StringPadding::NullTerminate,
///     CharacterSet::Utf8,
/// ));
/// let Datatype::VariableLength(VariableLengthType::String(string)) = datatype else {
///     panic!("expected a variable-length string");
/// };
/// assert_eq!(string.charset(), &CharacterSet::Utf8);
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Debug, PartialEq)]
pub enum VariableLengthType {
    /// A sequence whose elements share one datatype.
    Sequence {
        /// The datatype of one element.
        element_type: Box<Datatype>,
    },
    /// A string with padding and character-set metadata.
    String(VariableLengthStringType),
}

impl VariableLengthType {
    /// Constructs a sequence with the supplied element datatype.
    pub fn sequence(element_type: Datatype) -> Self {
        Self::Sequence {
            element_type: Box::new(element_type),
        }
    }

    /// Constructs a string with an unsigned one-byte integer parent datatype.
    pub fn string(padding: StringPadding, charset: CharacterSet) -> Self {
        Self::String(VariableLengthStringType::new(padding, charset))
    }

    /// Returns the element datatype of a sequence or the parent datatype of a string.
    pub fn base_type(&self) -> &Datatype {
        match self {
            Self::Sequence { element_type } => element_type,
            Self::String(string) => string.base_type(),
        }
    }
}

/// A variable-length string's padding, character set, and parent datatype.
///
/// The parent datatype has a positive element size. [`Self::new`] uses an unsigned one-byte
/// integer parent, and [`Self::with_base_type`] retains a supplied parent datatype.
///
/// The variable-length class for datatype message versions 1 through 5 is defined in "The
/// Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Debug, PartialEq)]
pub struct VariableLengthStringType {
    padding: StringPadding,
    charset: CharacterSet,
    base_type: Box<Datatype>,
}

impl VariableLengthStringType {
    /// Constructs a string with an unsigned one-byte integer parent datatype.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{CharacterSet, StringPadding, VariableLengthStringType};
    ///
    /// let string = VariableLengthStringType::new(StringPadding::SpacePad, CharacterSet::Ascii);
    /// assert_eq!(string.padding(), &StringPadding::SpacePad);
    /// assert_eq!(string.charset(), &CharacterSet::Ascii);
    /// assert_eq!(string.base_type().type_size(), 1);
    /// ```
    pub fn new(padding: StringPadding, charset: CharacterSet) -> Self {
        Self {
            padding,
            charset,
            base_type: Box::new(Datatype::FixedPoint(FixedPointType::u8())),
        }
    }

    /// Constructs a string with the supplied parent datatype.
    ///
    /// The parent datatype is retained as part of the string metadata.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ZeroSizedDatatype`] if `base_type` occupies zero bytes per element.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{
    ///     CharacterSet, Datatype, FormatError, StringPadding, VariableLengthStringType,
    /// };
    ///
    /// let base = Datatype::String {
    ///     size: 1,
    ///     padding: StringPadding::NullPad,
    ///     charset: CharacterSet::Ascii,
    /// };
    /// let string = VariableLengthStringType::with_base_type(
    ///     StringPadding::SpacePad,
    ///     CharacterSet::Utf8,
    ///     base.clone(),
    /// )?;
    /// assert_eq!(string.base_type(), &base);
    /// # Ok::<(), FormatError>(())
    /// ```
    pub fn with_base_type(
        padding: StringPadding,
        charset: CharacterSet,
        base_type: Datatype,
    ) -> Result<Self, FormatError> {
        // H5O__dtype_decode_helper in H5Odtype.c, libhdf5 1.14.6, rejects a zero-sized datatype.
        base_type.element_size()?;
        Ok(Self {
            padding,
            charset,
            base_type: Box::new(base_type),
        })
    }

    /// Returns the treatment of unused string bytes.
    pub fn padding(&self) -> &StringPadding {
        &self.padding
    }

    /// Returns the encoding of the string bytes.
    pub fn charset(&self) -> &CharacterSet {
        &self.charset
    }

    /// Returns the parent datatype stored with the string metadata.
    pub fn base_type(&self) -> &Datatype {
        &self.base_type
    }
}

#[cfg(test)]
#[path = "variable_length_tests.rs"]
mod tests;
