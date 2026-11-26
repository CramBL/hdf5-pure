use core::fmt;

/// A required property reported by [`FormatError::MissingNumericDatatypeProperty`][error].
///
/// [error]: crate::FormatError::MissingNumericDatatypeProperty
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumericDatatypeProperty {
    /// The first significant bit within each element.
    BitOffset,
    /// The number of significant bits in each element.
    BitPrecision,
    /// The order of bytes within each element.
    ByteOrder,
    /// The bias applied to the stored exponent.
    ExponentBias,
    /// The exponent field's first bit within each element.
    ExponentLocation,
    /// The number of bits occupied by the exponent field.
    ExponentSize,
    /// The mantissa field's first bit within each element.
    MantissaLocation,
    /// The representation of the most significant mantissa bit.
    MantissaNormalization,
    /// The number of bits occupied by the mantissa field.
    MantissaSize,
    /// The sign bit's position within each element.
    SignLocation,
    /// Whether integer values use two's complement representation.
    Signed,
    /// The number of bytes occupied by one element.
    Size,
}

impl fmt::Display for NumericDatatypeProperty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::BitOffset => "bit_offset",
            Self::BitPrecision => "bit_precision",
            Self::ByteOrder => "byte_order",
            Self::ExponentBias => "exponent_bias",
            Self::ExponentLocation => "exponent_location",
            Self::ExponentSize => "exponent_size",
            Self::MantissaLocation => "mantissa_location",
            Self::MantissaNormalization => "mantissa_normalization",
            Self::MantissaSize => "mantissa_size",
            Self::SignLocation => "sign_location",
            Self::Signed => "signed",
            Self::Size => "size",
        })
    }
}
