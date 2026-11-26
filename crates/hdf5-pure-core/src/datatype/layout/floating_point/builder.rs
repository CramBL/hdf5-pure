use super::definition::{FLOATING_POINT_CLASS, FloatingPointType, MantissaNormalization};
use crate::{DatatypeByteOrder, FormatError, NumericDatatypeProperty};

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
