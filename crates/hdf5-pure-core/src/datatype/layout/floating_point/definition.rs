use core::num::{NonZeroU16, NonZeroU32};

use super::builder::FloatingPointTypeBuilder;
use crate::{DatatypeByteOrder, FormatError};

/// A floating-point datatype with separate sign, exponent, and mantissa fields.
///
/// Each field is nonempty, lies within the significant bit range, and is disjoint from the
/// others. The significant bit range fits within the positive element size. Numeric value
/// conversions support a subset of these layouts and byte orders.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FloatingPointType {
    size: NonZeroU32,
    byte_order: DatatypeByteOrder,
    layout: FloatingPointLayout,
}

impl FloatingPointType {
    /// Creates an incomplete floating-point description with zero padding.
    ///
    /// See [`FloatingPointTypeBuilder`] for the required properties and an example.
    pub const fn builder() -> FloatingPointTypeBuilder {
        FloatingPointTypeBuilder::new()
    }

    /// Constructs a floating-point datatype from its element size and bit fields.
    ///
    /// `size` is in bytes. Bit locations count from the least significant bit of the element.
    /// The significant bit range starts at `bit_offset` and contains `bit_precision` bits.
    /// The sign, exponent, and mantissa fields must lie within that range.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ZeroSizedDatatype`] if `size` is zero,
    /// [`FormatError::InvalidFloatingPointRange`] if the bit range is empty or exceeds `size * 8`,
    /// [`FormatError::InvalidFloatingPointField`] if a field is empty or extends outside that
    /// range, or [`FormatError::OverlappingFloatingPointFields`] if two fields overlap.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{DatatypeByteOrder, FloatingPointType, MantissaNormalization};
    ///
    /// let binary16 = FloatingPointType::new(
    ///     2,
    ///     DatatypeByteOrder::LittleEndian,
    ///     MantissaNormalization::ImpliedLeadingOne,
    ///     15,
    ///     0,
    ///     16,
    ///     10,
    ///     5,
    ///     0,
    ///     10,
    ///     15,
    /// )?;
    /// assert_eq!(binary16.size().get(), 2);
    /// assert_eq!(binary16.layout().sign_location(), 15);
    /// assert_eq!(binary16.layout().exponent_location(), 10);
    /// assert_eq!(binary16.layout().exponent_size(), 5);
    /// assert_eq!(binary16.layout().mantissa_location(), 0);
    /// assert_eq!(binary16.layout().mantissa_size(), 10);
    /// assert_eq!(binary16.layout().exponent_bias(), 15);
    /// # Ok::<(), hdf5_pure_core::FormatError>(())
    /// ```
    pub fn new(
        size: u32,
        byte_order: DatatypeByteOrder,
        mantissa_normalization: MantissaNormalization,
        sign_location: u8,
        bit_offset: u16,
        bit_precision: u16,
        exponent_location: u8,
        exponent_size: u8,
        mantissa_location: u8,
        mantissa_size: u8,
        exponent_bias: u32,
    ) -> Result<Self, FormatError> {
        let size = NonZeroU32::new(size).ok_or(FormatError::ZeroSizedDatatype {
            class: FLOATING_POINT_CLASS,
        })?;
        let bit_precision = NonZeroU16::new(bit_precision)
            .filter(|precision| {
                u64::from(bit_offset) + u64::from(precision.get()) <= u64::from(size.get()) * 8
            })
            .ok_or(FormatError::InvalidFloatingPointRange {
                size: size.get(),
                bit_offset,
                bit_precision,
            })?;
        let layout = FloatingPointLayout::parse(
            mantissa_normalization,
            sign_location,
            bit_offset,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_location,
            mantissa_size,
            exponent_bias,
        )?;
        Ok(Self {
            size,
            byte_order,
            layout,
        })
    }

    /// Constructs a floating-point datatype from existing bit fields.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ZeroSizedDatatype`] if `size` is zero, or
    /// [`FormatError::InvalidFloatingPointRange`] if the layout's bit range exceeds `size * 8`.
    pub fn from_layout(
        size: u32,
        byte_order: DatatypeByteOrder,
        layout: FloatingPointLayout,
    ) -> Result<Self, FormatError> {
        let size = NonZeroU32::new(size).ok_or(FormatError::ZeroSizedDatatype {
            class: FLOATING_POINT_CLASS,
        })?;
        if u64::from(layout.bit_offset()) + u64::from(layout.bit_precision())
            > u64::from(size.get()) * 8
        {
            return Err(FormatError::InvalidFloatingPointRange {
                size: size.get(),
                bit_offset: layout.bit_offset(),
                bit_precision: layout.bit_precision(),
            });
        }
        Ok(Self {
            size,
            byte_order,
            layout,
        })
    }

    /// Constructs a four-byte IEEE 754 binary32 datatype.
    pub const fn ieee754_binary32(byte_order: DatatypeByteOrder) -> Self {
        Self {
            size: BINARY32_SIZE,
            byte_order,
            layout: FloatingPointLayout::IEEE754_BINARY32,
        }
    }

    /// Constructs an eight-byte IEEE 754 binary64 datatype.
    pub const fn ieee754_binary64(byte_order: DatatypeByteOrder) -> Self {
        Self {
            size: BINARY64_SIZE,
            byte_order,
            layout: FloatingPointLayout::IEEE754_BINARY64,
        }
    }

    /// Sets the low, high, and internal padding bits.
    ///
    /// Pass `true` for one or `false` for zero. The internal value is used for unused bits
    /// between the sign, exponent, and mantissa fields.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{DatatypeByteOrder, FloatingPointType};
    ///
    /// let datatype = FloatingPointType::ieee754_binary32(DatatypeByteOrder::LittleEndian)
    ///     .with_padding(true, false, true);
    /// assert!(datatype.layout().low_padding());
    /// assert!(!datatype.layout().high_padding());
    /// assert!(datatype.layout().internal_padding());
    /// ```
    pub const fn with_padding(mut self, low: bool, high: bool, internal: bool) -> Self {
        self.layout.low_padding = low;
        self.layout.high_padding = high;
        self.layout.internal_padding = internal;
        self
    }

    /// Returns the number of bytes occupied by one element.
    pub const fn size(self) -> NonZeroU32 {
        self.size
    }
    /// Returns the order of bytes within each element.
    pub const fn byte_order(self) -> DatatypeByteOrder {
        self.byte_order
    }
    /// Returns the floating-point fields and significant bit range of each element.
    pub const fn layout(self) -> FloatingPointLayout {
        self.layout
    }
}

/// The disjoint bit fields and significant bit range of a floating-point element.
///
/// Use [`FloatingPointType::from_layout`] to pair these fields with an element size and byte
/// order.
///
/// The fields for datatype message versions 1 through 5 are defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FloatingPointLayout {
    /// The value of unused bits below the significant range, with `true` representing one.
    low_padding: bool,
    /// The value of unused bits above the significant range, with `true` representing one.
    high_padding: bool,
    /// The value of unused bits between the sign, exponent, and mantissa fields.
    ///
    /// `true` represents one and `false` represents zero.
    internal_padding: bool,
    /// How the most significant mantissa bit is represented.
    mantissa_normalization: MantissaNormalization,
    /// The bit position of the sign, counting from the least significant bit.
    sign_location: u8,
    /// The first significant bit within each element.
    bit_offset: u16,
    /// The number of significant bits within each element.
    bit_precision: u16,
    /// The first bit of the exponent field.
    exponent_location: u8,
    /// The number of exponent bits.
    exponent_size: u8,
    /// The first bit of the mantissa field.
    mantissa_location: u8,
    /// The number of mantissa bits.
    mantissa_size: u8,
    /// The exponent bias stored in the message.
    exponent_bias: u32,
}

impl FloatingPointLayout {
    /// Returns the padding bit below the significant range, with `true` representing one.
    pub const fn low_padding(self) -> bool {
        self.low_padding
    }

    /// Returns the padding bit above the significant range, with `true` representing one.
    pub const fn high_padding(self) -> bool {
        self.high_padding
    }

    /// Returns the padding bit between the sign, exponent, and mantissa fields.
    ///
    /// `true` represents one and `false` represents zero.
    pub const fn internal_padding(self) -> bool {
        self.internal_padding
    }

    /// Parses a layout with nonempty, disjoint fields within the significant bit range.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidFloatingPointField`] if any field is empty or extends
    /// outside the significant bit range, or [`FormatError::OverlappingFloatingPointFields`]
    /// if two fields overlap.
    fn parse(
        mantissa_normalization: MantissaNormalization,
        sign_location: u8,
        bit_offset: u16,
        bit_precision: NonZeroU16,
        exponent_location: u8,
        exponent_size: u8,
        mantissa_location: u8,
        mantissa_size: u8,
        exponent_bias: u32,
    ) -> Result<Self, FormatError> {
        let bit_precision = bit_precision.get();
        let end = u32::from(bit_offset) + u32::from(bit_precision);
        for (field, location, size) in [
            (FloatingPointField::Sign, sign_location, 1),
            (
                FloatingPointField::Exponent,
                exponent_location,
                exponent_size,
            ),
            (
                FloatingPointField::Mantissa,
                mantissa_location,
                mantissa_size,
            ),
        ] {
            if size == 0
                || u16::from(location) < bit_offset
                || u32::from(location) + u32::from(size) > end
            {
                return Err(FormatError::InvalidFloatingPointField {
                    field,
                    location,
                    size,
                    bit_offset,
                    bit_precision,
                });
            }
        }
        for (first, first_start, first_size, second, second_start, second_size) in [
            (
                FloatingPointField::Sign,
                sign_location,
                1,
                FloatingPointField::Exponent,
                exponent_location,
                exponent_size,
            ),
            (
                FloatingPointField::Sign,
                sign_location,
                1,
                FloatingPointField::Mantissa,
                mantissa_location,
                mantissa_size,
            ),
            (
                FloatingPointField::Exponent,
                exponent_location,
                exponent_size,
                FloatingPointField::Mantissa,
                mantissa_location,
                mantissa_size,
            ),
        ] {
            if u16::from(first_start) < u16::from(second_start) + u16::from(second_size)
                && u16::from(second_start) < u16::from(first_start) + u16::from(first_size)
            {
                return Err(FormatError::OverlappingFloatingPointFields {
                    first,
                    second,
                    first_location: first_start,
                    first_size,
                    second_location: second_start,
                    second_size,
                });
            }
        }
        Ok(Self {
            low_padding: false,
            high_padding: false,
            internal_padding: false,
            mantissa_normalization,
            sign_location,
            bit_offset,
            bit_precision,
            exponent_location,
            exponent_size,
            mantissa_location,
            mantissa_size,
            exponent_bias,
        })
    }

    /// Returns the representation of the most significant mantissa bit.
    pub const fn mantissa_normalization(self) -> MantissaNormalization {
        self.mantissa_normalization
    }
    /// Returns the sign bit's position, counting from the least significant bit.
    pub const fn sign_location(self) -> u8 {
        self.sign_location
    }
    /// Returns the position of the first significant bit, counting from the least significant bit.
    pub const fn bit_offset(self) -> u16 {
        self.bit_offset
    }
    /// Returns the number of bits in the significant bit range.
    pub const fn bit_precision(self) -> u16 {
        self.bit_precision
    }
    /// Returns the exponent field's first bit, counting from the least significant bit.
    pub const fn exponent_location(self) -> u8 {
        self.exponent_location
    }
    /// Returns the number of bits occupied by the exponent field.
    pub const fn exponent_size(self) -> u8 {
        self.exponent_size
    }
    /// Returns the mantissa field's first bit, counting from the least significant bit.
    pub const fn mantissa_location(self) -> u8 {
        self.mantissa_location
    }
    /// Returns the number of bits occupied by the mantissa field.
    pub const fn mantissa_size(self) -> u8 {
        self.mantissa_size
    }
    /// Returns the bias applied to the stored exponent.
    pub const fn exponent_bias(self) -> u32 {
        self.exponent_bias
    }

    /// The bit fields of an IEEE 754 binary32 element.
    pub const IEEE754_BINARY32: Self = Self {
        low_padding: false,
        high_padding: false,
        internal_padding: false,
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
        low_padding: false,
        high_padding: false,
        internal_padding: false,
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
/// The values for datatype message versions 1 through 5 are defined in "The Datatype Message" of
/// the [format specification, version 4.0][spec].
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
}

/// A component of a floating-point element, used to report invalid field geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FloatingPointField {
    /// The sign bit.
    Sign,
    /// The exponent field.
    Exponent,
    /// The mantissa field.
    Mantissa,
}

// The format specification, version 4.0, defines this class code in "The Datatype Message".
pub(super) const FLOATING_POINT_CLASS: u8 = 1;

const BINARY32_SIZE: NonZeroU32 = NonZeroU32::new(4).unwrap();
const BINARY64_SIZE: NonZeroU32 = NonZeroU32::new(8).unwrap();

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
