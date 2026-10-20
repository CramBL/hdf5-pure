/// The sign and bit range stored in a fixed-point datatype message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedPointLayout {
    /// Whether the stored integer has a sign bit.
    pub signed: bool,
    /// The first significant bit within each element.
    pub bit_offset: u16,
    /// The number of significant bits within each element.
    pub bit_precision: u16,
}

/// The bit fields stored in a floating-point datatype message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FloatingPointLayout {
    /// How the most significant mantissa bit is represented.
    pub mantissa_normalization: MantissaNormalization,
    /// The bit position of the sign, counting from the least significant bit.
    pub sign_location: u8,
    /// The first significant bit within each element.
    pub bit_offset: u16,
    /// The number of significant bits within each element.
    pub bit_precision: u16,
    /// The first bit of the exponent field.
    pub exponent_location: u8,
    /// The number of exponent bits.
    pub exponent_size: u8,
    /// The first bit of the mantissa field.
    pub mantissa_location: u8,
    /// The number of mantissa bits.
    pub mantissa_size: u8,
    /// The exponent bias stored in the message.
    pub exponent_bias: u32,
}

impl FloatingPointLayout {
    /// The bit fields of an IEEE 754 binary32 element.
    pub const IEEE754_BINARY32: Self = Self {
        mantissa_normalization: MantissaNormalization::ImpliedLeadingOne,
        sign_location: 31,
        bit_offset: 0,
        bit_precision: 32,
        exponent_location: 23,
        exponent_size: 8,
        mantissa_location: 0,
        mantissa_size: 23,
        exponent_bias: 127,
    };
    /// The bit fields of an IEEE 754 binary64 element.
    pub const IEEE754_BINARY64: Self = Self {
        mantissa_normalization: MantissaNormalization::ImpliedLeadingOne,
        sign_location: 63,
        bit_offset: 0,
        bit_precision: 64,
        exponent_location: 52,
        exponent_size: 11,
        mantissa_location: 0,
        mantissa_size: 52,
        exponent_bias: 1023,
    };
}

/// The representation of the most significant mantissa bit.
///
/// The values are defined in "The Datatype Message" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MantissaNormalization {
    /// The mantissa is stored without normalization.
    Unnormalized,
    /// The most significant mantissa bit is stored and set, except for zero.
    LeadingOne,
    /// The most significant mantissa bit is implied to be set and omitted from storage.
    ImpliedLeadingOne,
    /// The datatype stores the reserved normalization value.
    Reserved,
}
