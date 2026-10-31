//! Named construction of checked numeric datatypes.

use core::fmt;

use super::{
    BIT_FIELD_CLASS, BitFieldType, FIXED_POINT_CLASS, FLOATING_POINT_CLASS, FixedPointType,
    FloatingPointType, MantissaNormalization,
};
use crate::{DatatypeByteOrder, FormatError};

/// A description of an integer datatype with named properties.
///
/// Set the [`size`](Self::size), [`byte_order`](Self::byte_order), [`signed`](Self::signed),
/// [`bit_offset`](Self::bit_offset), and [`bit_precision`](Self::bit_precision) before calling
/// [`build`](Self::build). Padding defaults to zero. Calling a setter again replaces its value.
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::{DatatypeByteOrder, FixedPointType};
///
/// let integer = FixedPointType::builder()
///     .size(4)
///     .byte_order(DatatypeByteOrder::LittleEndian)
///     .signed(true)
///     .bit_offset(3)
///     .bit_precision(21)
///     .low_padding(false)
///     .high_padding(false)
///     .build()?;
/// assert_eq!(integer.size().get(), 4);
/// assert_eq!(integer.layout().bit_offset(), 3);
/// assert_eq!(integer.layout().bit_precision(), 21);
/// # Ok::<(), hdf5_pure_core::FormatError>(())
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FixedPointTypeBuilder {
    size: Option<u32>,
    byte_order: Option<DatatypeByteOrder>,
    signed: Option<bool>,
    bit_offset: Option<u16>,
    bit_precision: Option<u16>,
    low_padding: bool,
    high_padding: bool,
}

impl FixedPointTypeBuilder {
    /// Creates an incomplete description with zero padding.
    pub const fn new() -> Self {
        Self {
            size: None,
            byte_order: None,
            signed: None,
            bit_offset: None,
            bit_precision: None,
            low_padding: false,
            high_padding: false,
        }
    }

    /// Sets the number of bytes occupied by one element.
    pub const fn size(mut self, size: u32) -> Self {
        self.size = Some(size);
        self
    }

    /// Sets the order of bytes within each element.
    pub const fn byte_order(mut self, byte_order: DatatypeByteOrder) -> Self {
        self.byte_order = Some(byte_order);
        self
    }

    /// Selects two's complement representation when `signed` is `true`.
    pub const fn signed(mut self, signed: bool) -> Self {
        self.signed = Some(signed);
        self
    }

    /// Sets the first significant bit, counting from the least significant bit.
    pub const fn bit_offset(mut self, bit_offset: u16) -> Self {
        self.bit_offset = Some(bit_offset);
        self
    }

    /// Sets the number of significant bits in each element.
    pub const fn bit_precision(mut self, bit_precision: u16) -> Self {
        self.bit_precision = Some(bit_precision);
        self
    }

    /// Sets the padding value below the significant range, with `true` representing one.
    pub const fn low_padding(mut self, low_padding: bool) -> Self {
        self.low_padding = low_padding;
        self
    }

    /// Sets the padding value above the significant range, with `true` representing one.
    pub const fn high_padding(mut self, high_padding: bool) -> Self {
        self.high_padding = high_padding;
        self
    }

    /// Constructs an integer datatype after checking its required properties and bit range.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::MissingNumericDatatypeProperty`] if a required property is unset,
    /// [`FormatError::ZeroSizedDatatype`] if the element size is zero,
    /// [`FormatError::InvalidNumericByteOrder`] if the byte order is [`DatatypeByteOrder::Vax`], or
    /// [`FormatError::InvalidFixedPointRange`] if the significant bit range is empty or extends
    /// past the element.
    pub fn build(self) -> Result<FixedPointType, FormatError> {
        let Self {
            size,
            byte_order,
            signed,
            bit_offset,
            bit_precision,
            low_padding,
            high_padding,
        } = self;
        Ok(FixedPointType::new(
            size.ok_or(FormatError::MissingNumericDatatypeProperty {
                class: FIXED_POINT_CLASS,
                property: NumericDatatypeProperty::Size,
            })?,
            byte_order.ok_or(FormatError::MissingNumericDatatypeProperty {
                class: FIXED_POINT_CLASS,
                property: NumericDatatypeProperty::ByteOrder,
            })?,
            signed.ok_or(FormatError::MissingNumericDatatypeProperty {
                class: FIXED_POINT_CLASS,
                property: NumericDatatypeProperty::Signed,
            })?,
            bit_offset.ok_or(FormatError::MissingNumericDatatypeProperty {
                class: FIXED_POINT_CLASS,
                property: NumericDatatypeProperty::BitOffset,
            })?,
            bit_precision.ok_or(FormatError::MissingNumericDatatypeProperty {
                class: FIXED_POINT_CLASS,
                property: NumericDatatypeProperty::BitPrecision,
            })?,
        )?
        .with_padding(low_padding, high_padding))
    }
}

/// A description of a bit-field datatype with named properties.
///
/// Set the [`size`](Self::size), [`byte_order`](Self::byte_order),
/// [`bit_offset`](Self::bit_offset), and [`bit_precision`](Self::bit_precision) before calling
/// [`build`](Self::build). Padding defaults to zero. Calling a setter again replaces its value.
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::{BitFieldType, DatatypeByteOrder};
///
/// let bits = BitFieldType::builder()
///     .size(2)
///     .byte_order(DatatypeByteOrder::BigEndian)
///     .bit_offset(3)
///     .bit_precision(9)
///     .high_padding(true)
///     .build()?;
/// assert_eq!(bits.bit_offset(), 3);
/// assert_eq!(bits.bit_precision().get(), 9);
/// assert!(!bits.low_padding());
/// assert!(bits.high_padding());
/// # Ok::<(), hdf5_pure_core::FormatError>(())
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BitFieldTypeBuilder {
    size: Option<u32>,
    byte_order: Option<DatatypeByteOrder>,
    bit_offset: Option<u16>,
    bit_precision: Option<u16>,
    low_padding: bool,
    high_padding: bool,
}

impl BitFieldTypeBuilder {
    /// Creates an incomplete description with zero padding.
    pub const fn new() -> Self {
        Self {
            size: None,
            byte_order: None,
            bit_offset: None,
            bit_precision: None,
            low_padding: false,
            high_padding: false,
        }
    }

    /// Sets the number of bytes occupied by one element.
    pub const fn size(mut self, size: u32) -> Self {
        self.size = Some(size);
        self
    }

    /// Sets the order of bytes within each element.
    pub const fn byte_order(mut self, byte_order: DatatypeByteOrder) -> Self {
        self.byte_order = Some(byte_order);
        self
    }

    /// Sets the first significant bit, counting from the least significant bit.
    pub const fn bit_offset(mut self, bit_offset: u16) -> Self {
        self.bit_offset = Some(bit_offset);
        self
    }

    /// Sets the number of significant bits in each element.
    pub const fn bit_precision(mut self, bit_precision: u16) -> Self {
        self.bit_precision = Some(bit_precision);
        self
    }

    /// Sets the padding value below the significant range, with `true` representing one.
    pub const fn low_padding(mut self, low_padding: bool) -> Self {
        self.low_padding = low_padding;
        self
    }

    /// Sets the padding value above the significant range, with `true` representing one.
    pub const fn high_padding(mut self, high_padding: bool) -> Self {
        self.high_padding = high_padding;
        self
    }

    /// Constructs a bit-field datatype after checking its required properties and bit range.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::MissingNumericDatatypeProperty`] if a required property is unset,
    /// [`FormatError::ZeroSizedDatatype`] if the element size is zero,
    /// [`FormatError::InvalidNumericByteOrder`] if the byte order is [`DatatypeByteOrder::Vax`], or
    /// [`FormatError::InvalidNumericBitRange`] if the significant bit range is empty or extends
    /// past the element.
    pub fn build(self) -> Result<BitFieldType, FormatError> {
        let Self {
            size,
            byte_order,
            bit_offset,
            bit_precision,
            low_padding,
            high_padding,
        } = self;
        Ok(BitFieldType::new(
            size.ok_or(FormatError::MissingNumericDatatypeProperty {
                class: BIT_FIELD_CLASS,
                property: NumericDatatypeProperty::Size,
            })?,
            byte_order.ok_or(FormatError::MissingNumericDatatypeProperty {
                class: BIT_FIELD_CLASS,
                property: NumericDatatypeProperty::ByteOrder,
            })?,
            bit_offset.ok_or(FormatError::MissingNumericDatatypeProperty {
                class: BIT_FIELD_CLASS,
                property: NumericDatatypeProperty::BitOffset,
            })?,
            bit_precision.ok_or(FormatError::MissingNumericDatatypeProperty {
                class: BIT_FIELD_CLASS,
                property: NumericDatatypeProperty::BitPrecision,
            })?,
        )?
        .with_padding(low_padding, high_padding))
    }
}

/// A description of a floating-point datatype with named properties.
///
/// Set the [`size`](Self::size), [`byte_order`](Self::byte_order),
/// [`mantissa_normalization`](Self::mantissa_normalization), [`sign_location`](Self::sign_location),
/// [`bit_offset`](Self::bit_offset), [`bit_precision`](Self::bit_precision),
/// [`exponent_location`](Self::exponent_location), [`exponent_size`](Self::exponent_size),
/// [`mantissa_location`](Self::mantissa_location), [`mantissa_size`](Self::mantissa_size), and
/// [`exponent_bias`](Self::exponent_bias) before calling [`build`](Self::build). Padding defaults
/// to zero. Calling a setter again replaces its value.
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::{DatatypeByteOrder, FloatingPointType, MantissaNormalization};
///
/// let binary16 = FloatingPointType::builder()
///     .size(2)
///     .byte_order(DatatypeByteOrder::LittleEndian)
///     .mantissa_normalization(MantissaNormalization::ImpliedLeadingOne)
///     .sign_location(15)
///     .bit_offset(0)
///     .bit_precision(16)
///     .exponent_location(10)
///     .exponent_size(5)
///     .mantissa_location(0)
///     .mantissa_size(10)
///     .exponent_bias(15)
///     .build()?;
/// assert_eq!(binary16.size().get(), 2);
/// assert_eq!(binary16.layout().sign_location(), 15);
/// assert_eq!(binary16.layout().exponent_location(), 10);
/// assert_eq!(binary16.layout().exponent_size(), 5);
/// assert_eq!(binary16.layout().mantissa_location(), 0);
/// assert_eq!(binary16.layout().mantissa_size(), 10);
/// assert_eq!(binary16.layout().exponent_bias(), 15);
/// # Ok::<(), hdf5_pure_core::FormatError>(())
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FloatingPointTypeBuilder {
    size: Option<u32>,
    byte_order: Option<DatatypeByteOrder>,
    mantissa_normalization: Option<MantissaNormalization>,
    sign_location: Option<u8>,
    bit_offset: Option<u16>,
    bit_precision: Option<u16>,
    exponent_location: Option<u8>,
    exponent_size: Option<u8>,
    mantissa_location: Option<u8>,
    mantissa_size: Option<u8>,
    exponent_bias: Option<u32>,
    low_padding: bool,
    high_padding: bool,
    internal_padding: bool,
}

impl FloatingPointTypeBuilder {
    /// Creates an incomplete description with zero padding.
    pub const fn new() -> Self {
        Self {
            size: None,
            byte_order: None,
            mantissa_normalization: None,
            sign_location: None,
            bit_offset: None,
            bit_precision: None,
            exponent_location: None,
            exponent_size: None,
            mantissa_location: None,
            mantissa_size: None,
            exponent_bias: None,
            low_padding: false,
            high_padding: false,
            internal_padding: false,
        }
    }

    /// Sets the number of bytes occupied by one element.
    pub const fn size(mut self, size: u32) -> Self {
        self.size = Some(size);
        self
    }

    /// Sets the order of bytes within each element.
    pub const fn byte_order(mut self, byte_order: DatatypeByteOrder) -> Self {
        self.byte_order = Some(byte_order);
        self
    }

    /// Sets how the most significant mantissa bit is represented.
    pub const fn mantissa_normalization(
        mut self,
        mantissa_normalization: MantissaNormalization,
    ) -> Self {
        self.mantissa_normalization = Some(mantissa_normalization);
        self
    }

    /// Sets the sign bit's position, counting from the least significant bit.
    pub const fn sign_location(mut self, sign_location: u8) -> Self {
        self.sign_location = Some(sign_location);
        self
    }

    /// Sets the first significant bit, counting from the least significant bit.
    pub const fn bit_offset(mut self, bit_offset: u16) -> Self {
        self.bit_offset = Some(bit_offset);
        self
    }

    /// Sets the number of significant bits in each element.
    pub const fn bit_precision(mut self, bit_precision: u16) -> Self {
        self.bit_precision = Some(bit_precision);
        self
    }

    /// Sets the exponent field's first bit, counting from the least significant bit.
    pub const fn exponent_location(mut self, exponent_location: u8) -> Self {
        self.exponent_location = Some(exponent_location);
        self
    }

    /// Sets the number of bits occupied by the exponent field.
    pub const fn exponent_size(mut self, exponent_size: u8) -> Self {
        self.exponent_size = Some(exponent_size);
        self
    }

    /// Sets the mantissa field's first bit, counting from the least significant bit.
    pub const fn mantissa_location(mut self, mantissa_location: u8) -> Self {
        self.mantissa_location = Some(mantissa_location);
        self
    }

    /// Sets the number of bits occupied by the mantissa field.
    pub const fn mantissa_size(mut self, mantissa_size: u8) -> Self {
        self.mantissa_size = Some(mantissa_size);
        self
    }

    /// Sets the bias applied to the stored exponent.
    pub const fn exponent_bias(mut self, exponent_bias: u32) -> Self {
        self.exponent_bias = Some(exponent_bias);
        self
    }

    /// Sets the padding value below the significant range, with `true` representing one.
    pub const fn low_padding(mut self, low_padding: bool) -> Self {
        self.low_padding = low_padding;
        self
    }

    /// Sets the padding value above the significant range, with `true` representing one.
    pub const fn high_padding(mut self, high_padding: bool) -> Self {
        self.high_padding = high_padding;
        self
    }

    /// Sets the padding value between the sign, exponent, and mantissa fields.
    ///
    /// Pass `true` for one or `false` for zero.
    pub const fn internal_padding(mut self, internal_padding: bool) -> Self {
        self.internal_padding = internal_padding;
        self
    }

    /// Constructs a floating-point datatype after checking its required properties and fields.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::MissingNumericDatatypeProperty`] if a required property is unset,
    /// [`FormatError::ZeroSizedDatatype`] if the element size is zero,
    /// [`FormatError::InvalidFloatingPointRange`] if the significant bit range is empty or extends
    /// past the element, [`FormatError::InvalidFloatingPointField`] if a field is empty or extends
    /// outside that range, or [`FormatError::OverlappingFloatingPointFields`] if two fields overlap.
    pub fn build(self) -> Result<FloatingPointType, FormatError> {
        let Self {
            size,
            byte_order,
            mantissa_normalization,
            sign_location,
            bit_offset,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_location,
            mantissa_size,
            exponent_bias,
            low_padding,
            high_padding,
            internal_padding,
        } = self;
        Ok(FloatingPointType::new(
            size.ok_or(FormatError::MissingNumericDatatypeProperty {
                class: FLOATING_POINT_CLASS,
                property: NumericDatatypeProperty::Size,
            })?,
            byte_order.ok_or(FormatError::MissingNumericDatatypeProperty {
                class: FLOATING_POINT_CLASS,
                property: NumericDatatypeProperty::ByteOrder,
            })?,
            mantissa_normalization.ok_or(FormatError::MissingNumericDatatypeProperty {
                class: FLOATING_POINT_CLASS,
                property: NumericDatatypeProperty::MantissaNormalization,
            })?,
            sign_location.ok_or(FormatError::MissingNumericDatatypeProperty {
                class: FLOATING_POINT_CLASS,
                property: NumericDatatypeProperty::SignLocation,
            })?,
            bit_offset.ok_or(FormatError::MissingNumericDatatypeProperty {
                class: FLOATING_POINT_CLASS,
                property: NumericDatatypeProperty::BitOffset,
            })?,
            bit_precision.ok_or(FormatError::MissingNumericDatatypeProperty {
                class: FLOATING_POINT_CLASS,
                property: NumericDatatypeProperty::BitPrecision,
            })?,
            exponent_location.ok_or(FormatError::MissingNumericDatatypeProperty {
                class: FLOATING_POINT_CLASS,
                property: NumericDatatypeProperty::ExponentLocation,
            })?,
            exponent_size.ok_or(FormatError::MissingNumericDatatypeProperty {
                class: FLOATING_POINT_CLASS,
                property: NumericDatatypeProperty::ExponentSize,
            })?,
            mantissa_location.ok_or(FormatError::MissingNumericDatatypeProperty {
                class: FLOATING_POINT_CLASS,
                property: NumericDatatypeProperty::MantissaLocation,
            })?,
            mantissa_size.ok_or(FormatError::MissingNumericDatatypeProperty {
                class: FLOATING_POINT_CLASS,
                property: NumericDatatypeProperty::MantissaSize,
            })?,
            exponent_bias.ok_or(FormatError::MissingNumericDatatypeProperty {
                class: FLOATING_POINT_CLASS,
                property: NumericDatatypeProperty::ExponentBias,
            })?,
        )?
        .with_padding(low_padding, high_padding, internal_padding))
    }
}

/// A required property reported by [`FormatError::MissingNumericDatatypeProperty`].
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
