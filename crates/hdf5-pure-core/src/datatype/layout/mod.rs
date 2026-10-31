//! Checked layouts for numeric datatype elements.
//!
//! The numeric classes of datatype message versions 1 through 5 are defined in "The Datatype
//! Message" of the [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage

use core::num::{NonZeroU16, NonZeroU32};

use self::builder::{BitFieldTypeBuilder, FixedPointTypeBuilder, FloatingPointTypeBuilder};
use super::byte_order::DatatypeByteOrder;
use crate::FormatError;

pub(crate) mod builder;

/// A fixed-point datatype with a nonempty bit range contained within each element.
///
/// Elements have a positive size and little-endian or big-endian byte order. The size may
/// exceed the widths supported by numeric value conversions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedPointType {
    size: NonZeroU32,
    byte_order: DatatypeByteOrder,
    layout: FixedPointLayout,
}

impl FixedPointType {
    /// Creates an incomplete integer description with zero padding.
    ///
    /// See [`FixedPointTypeBuilder`] for the required properties and an example.
    pub const fn builder() -> FixedPointTypeBuilder {
        FixedPointTypeBuilder::new()
    }

    /// Constructs an integer datatype with a nonempty significant bit range.
    ///
    /// `size` is the element size in bytes. `bit_offset` counts from the least significant bit,
    /// and `bit_precision` is the number of significant bits. Signed values use two's complement.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ZeroSizedDatatype`] if `size` is zero,
    /// [`FormatError::InvalidNumericByteOrder`] if `byte_order` is [`DatatypeByteOrder::Vax`], or
    /// [`FormatError::InvalidFixedPointRange`] if the bit range is empty or exceeds `size * 8`.
    pub fn new(
        size: u32,
        byte_order: DatatypeByteOrder,
        signed: bool,
        bit_offset: u16,
        bit_precision: u16,
    ) -> Result<Self, FormatError> {
        let size = NonZeroU32::new(size).ok_or(FormatError::ZeroSizedDatatype {
            class: FIXED_POINT_CLASS,
        })?;
        if byte_order == DatatypeByteOrder::Vax {
            return Err(FormatError::InvalidNumericByteOrder {
                class: FIXED_POINT_CLASS,
                byte_order,
            });
        }
        let bit_precision = NonZeroU16::new(bit_precision)
            .filter(|precision| {
                u64::from(bit_offset) + u64::from(precision.get()) <= u64::from(size.get()) * 8
            })
            .ok_or(FormatError::InvalidFixedPointRange {
                size: size.get(),
                bit_offset,
                bit_precision,
            })?;
        Ok(Self {
            size,
            byte_order,
            layout: FixedPointLayout {
                low_padding: false,
                high_padding: false,
                signed,
                bit_offset,
                bit_precision,
            },
        })
    }

    /// Constructs an integer datatype from an existing sign and bit range.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ZeroSizedDatatype`] if `size` is zero,
    /// [`FormatError::InvalidNumericByteOrder`] if `byte_order` is [`DatatypeByteOrder::Vax`], or
    /// [`FormatError::InvalidFixedPointRange`] if the layout's bit range exceeds `size * 8`.
    pub fn from_layout(
        size: u32,
        byte_order: DatatypeByteOrder,
        layout: FixedPointLayout,
    ) -> Result<Self, FormatError> {
        let FixedPointLayout {
            low_padding,
            high_padding,
            signed,
            bit_offset,
            bit_precision,
        } = layout;
        Ok(
            Self::new(size, byte_order, signed, bit_offset, bit_precision.get())?
                .with_padding(low_padding, high_padding),
        )
    }

    /// Constructs a signed 8-bit integer datatype in little-endian byte order.
    pub const fn i8() -> Self {
        Self {
            size: NonZeroU32::MIN,
            byte_order: DatatypeByteOrder::LittleEndian,
            layout: FixedPointLayout {
                low_padding: false,
                high_padding: false,
                signed: true,
                bit_offset: 0,
                bit_precision: standard_precision(8),
            },
        }
    }

    /// Constructs a signed 16-bit integer datatype in little-endian byte order.
    pub const fn i16() -> Self {
        Self {
            size: standard_size(2),
            byte_order: DatatypeByteOrder::LittleEndian,
            layout: FixedPointLayout {
                low_padding: false,
                high_padding: false,
                signed: true,
                bit_offset: 0,
                bit_precision: standard_precision(16),
            },
        }
    }

    /// Constructs a signed 32-bit integer datatype in little-endian byte order.
    pub const fn i32() -> Self {
        Self {
            size: standard_size(4),
            byte_order: DatatypeByteOrder::LittleEndian,
            layout: FixedPointLayout {
                low_padding: false,
                high_padding: false,
                signed: true,
                bit_offset: 0,
                bit_precision: standard_precision(32),
            },
        }
    }

    /// Constructs a signed 64-bit integer datatype in little-endian byte order.
    pub const fn i64() -> Self {
        Self {
            size: standard_size(8),
            byte_order: DatatypeByteOrder::LittleEndian,
            layout: FixedPointLayout {
                low_padding: false,
                high_padding: false,
                signed: true,
                bit_offset: 0,
                bit_precision: standard_precision(64),
            },
        }
    }

    /// Constructs an unsigned 8-bit integer datatype in little-endian byte order.
    pub const fn u8() -> Self {
        Self {
            size: NonZeroU32::MIN,
            byte_order: DatatypeByteOrder::LittleEndian,
            layout: FixedPointLayout {
                low_padding: false,
                high_padding: false,
                signed: false,
                bit_offset: 0,
                bit_precision: standard_precision(8),
            },
        }
    }

    /// Constructs an unsigned 16-bit integer datatype in little-endian byte order.
    pub const fn u16() -> Self {
        Self {
            size: standard_size(2),
            byte_order: DatatypeByteOrder::LittleEndian,
            layout: FixedPointLayout {
                low_padding: false,
                high_padding: false,
                signed: false,
                bit_offset: 0,
                bit_precision: standard_precision(16),
            },
        }
    }

    /// Constructs an unsigned 32-bit integer datatype in little-endian byte order.
    pub const fn u32() -> Self {
        Self {
            size: standard_size(4),
            byte_order: DatatypeByteOrder::LittleEndian,
            layout: FixedPointLayout {
                low_padding: false,
                high_padding: false,
                signed: false,
                bit_offset: 0,
                bit_precision: standard_precision(32),
            },
        }
    }

    /// Constructs an unsigned 64-bit integer datatype in little-endian byte order.
    pub const fn u64() -> Self {
        Self {
            size: standard_size(8),
            byte_order: DatatypeByteOrder::LittleEndian,
            layout: FixedPointLayout {
                low_padding: false,
                high_padding: false,
                signed: false,
                bit_offset: 0,
                bit_precision: standard_precision(64),
            },
        }
    }

    /// Sets the padding bits below and above the significant range.
    ///
    /// Pass `true` for one or `false` for zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::FixedPointType;
    ///
    /// let datatype = FixedPointType::u16().with_padding(true, false);
    /// assert!(datatype.layout().low_padding());
    /// assert!(!datatype.layout().high_padding());
    /// ```
    pub const fn with_padding(mut self, low: bool, high: bool) -> Self {
        self.layout.low_padding = low;
        self.layout.high_padding = high;
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
    /// Returns the padding, sign, and significant bit range of each element.
    pub const fn layout(self) -> FixedPointLayout {
        self.layout
    }
}

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
            size: standard_size(4),
            byte_order,
            layout: FloatingPointLayout::IEEE754_BINARY32,
        }
    }

    /// Constructs an eight-byte IEEE 754 binary64 datatype.
    pub const fn ieee754_binary64(byte_order: DatatypeByteOrder) -> Self {
        Self {
            size: standard_size(8),
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

/// A time datatype with positive element size and precision.
///
/// The precision fits within the element size, and the byte order is little-endian or
/// big-endian.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TimeType {
    size: NonZeroU32,
    byte_order: DatatypeByteOrder,
    bit_precision: NonZeroU16,
}

impl TimeType {
    /// Constructs a time datatype whose precision fits within its element size.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ZeroSizedDatatype`] if `size` is zero,
    /// [`FormatError::InvalidNumericByteOrder`] if `byte_order` is [`DatatypeByteOrder::Vax`], or
    /// [`FormatError::InvalidNumericBitRange`] if `bit_precision` is zero or exceeds `size * 8`.
    pub fn new(
        size: u32,
        byte_order: DatatypeByteOrder,
        bit_precision: u16,
    ) -> Result<Self, FormatError> {
        let size =
            NonZeroU32::new(size).ok_or(FormatError::ZeroSizedDatatype { class: TIME_CLASS })?;
        if byte_order == DatatypeByteOrder::Vax {
            return Err(FormatError::InvalidNumericByteOrder {
                class: TIME_CLASS,
                byte_order,
            });
        }
        let bit_precision = NonZeroU16::new(bit_precision)
            .filter(|precision| u64::from(precision.get()) <= u64::from(size.get()) * 8)
            .ok_or(FormatError::InvalidNumericBitRange {
                class: TIME_CLASS,
                size: size.get(),
                bit_offset: 0,
                bit_precision,
            })?;
        Ok(Self {
            size,
            byte_order,
            bit_precision,
        })
    }

    /// Returns the number of bytes occupied by one element.
    pub const fn size(self) -> NonZeroU32 {
        self.size
    }
    /// Returns the order of bytes within each element.
    pub const fn byte_order(self) -> DatatypeByteOrder {
        self.byte_order
    }
    /// Returns the number of significant bits in each element.
    pub const fn bit_precision(self) -> NonZeroU16 {
        self.bit_precision
    }
}

/// A bit-field datatype with a nonempty bit range contained within each element.
///
/// Elements have a positive size and little-endian or big-endian byte order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BitFieldType {
    /// The value of unused bits below the significant range, with `true` representing one.
    low_padding: bool,
    /// The value of unused bits above the significant range, with `true` representing one.
    high_padding: bool,
    size: NonZeroU32,
    byte_order: DatatypeByteOrder,
    bit_offset: u16,
    bit_precision: NonZeroU16,
}

impl BitFieldType {
    /// Creates an incomplete bit-field description with zero padding.
    ///
    /// See [`BitFieldTypeBuilder`] for the required properties and an example.
    pub const fn builder() -> BitFieldTypeBuilder {
        BitFieldTypeBuilder::new()
    }

    /// Constructs a bit-field datatype with a nonempty significant bit range.
    ///
    /// `size` is the element size in bytes. `bit_offset` counts from the least significant bit.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ZeroSizedDatatype`] if `size` is zero,
    /// [`FormatError::InvalidNumericByteOrder`] if `byte_order` is [`DatatypeByteOrder::Vax`], or
    /// [`FormatError::InvalidNumericBitRange`] if the bit range is empty or exceeds `size * 8`.
    pub fn new(
        size: u32,
        byte_order: DatatypeByteOrder,
        bit_offset: u16,
        bit_precision: u16,
    ) -> Result<Self, FormatError> {
        let size = NonZeroU32::new(size).ok_or(FormatError::ZeroSizedDatatype {
            class: BIT_FIELD_CLASS,
        })?;
        if byte_order == DatatypeByteOrder::Vax {
            return Err(FormatError::InvalidNumericByteOrder {
                class: BIT_FIELD_CLASS,
                byte_order,
            });
        }
        let bit_precision = NonZeroU16::new(bit_precision)
            .filter(|precision| {
                u64::from(bit_offset) + u64::from(precision.get()) <= u64::from(size.get()) * 8
            })
            .ok_or(FormatError::InvalidNumericBitRange {
                class: BIT_FIELD_CLASS,
                size: size.get(),
                bit_offset,
                bit_precision,
            })?;
        Ok(Self {
            low_padding: false,
            high_padding: false,
            size,
            byte_order,
            bit_offset,
            bit_precision,
        })
    }

    /// Sets the padding bits below and above the significant range.
    ///
    /// Pass `true` for one or `false` for zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{BitFieldType, DatatypeByteOrder};
    ///
    /// let datatype = BitFieldType::new(2, DatatypeByteOrder::LittleEndian, 3, 9)?
    ///     .with_padding(false, true);
    /// assert!(!datatype.low_padding());
    /// assert!(datatype.high_padding());
    /// # Ok::<(), hdf5_pure_core::FormatError>(())
    /// ```
    pub const fn with_padding(mut self, low: bool, high: bool) -> Self {
        self.low_padding = low;
        self.high_padding = high;
        self
    }

    /// Returns the padding bit below the significant range, with `true` representing one.
    pub const fn low_padding(self) -> bool {
        self.low_padding
    }

    /// Returns the padding bit above the significant range, with `true` representing one.
    pub const fn high_padding(self) -> bool {
        self.high_padding
    }

    /// Returns the number of bytes occupied by one element.
    pub const fn size(self) -> NonZeroU32 {
        self.size
    }
    /// Returns the order of bytes within each element.
    pub const fn byte_order(self) -> DatatypeByteOrder {
        self.byte_order
    }
    /// Returns the position of the first significant bit, counting from the least significant bit.
    pub const fn bit_offset(self) -> u16 {
        self.bit_offset
    }
    /// Returns the number of significant bits in each element.
    pub const fn bit_precision(self) -> NonZeroU16 {
        self.bit_precision
    }
}

/// The padding, sign, and bit range stored in a fixed-point datatype message.
///
/// The fields for datatype message versions 1 through 5 are defined in "The Datatype Message"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedPointLayout {
    /// The value of unused bits below the significant range, with `true` representing one.
    low_padding: bool,
    /// The value of unused bits above the significant range, with `true` representing one.
    high_padding: bool,
    /// Whether the stored integer has a sign bit.
    signed: bool,
    /// The first significant bit within each element.
    bit_offset: u16,
    /// The number of significant bits within each element.
    bit_precision: NonZeroU16,
}

impl FixedPointLayout {
    /// Returns the padding bit below the significant range, with `true` representing one.
    pub const fn low_padding(self) -> bool {
        self.low_padding
    }

    /// Returns the padding bit above the significant range, with `true` representing one.
    pub const fn high_padding(self) -> bool {
        self.high_padding
    }

    /// Returns the nonzero number of significant bits in each element.
    pub const fn precision(self) -> NonZeroU16 {
        self.bit_precision
    }

    /// Returns whether values use two's complement representation.
    pub const fn signed(self) -> bool {
        self.signed
    }
    /// Returns the position of the first significant bit, counting from the least significant bit.
    pub const fn bit_offset(self) -> u16 {
        self.bit_offset
    }
    /// Returns the number of significant bits in each element.
    pub const fn bit_precision(self) -> u16 {
        self.bit_precision.get()
    }
}

/// The disjoint bit fields and significant bit range of a floating-point element.
///
/// The fields are constructed through [`FloatingPointType::new`], which also checks that the
/// significant bit range fits within the element size.
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

// The class codes are defined in "The Datatype Message" of the format specification, version 4.0.
const FIXED_POINT_CLASS: u8 = 0;
const FLOATING_POINT_CLASS: u8 = 1;
const TIME_CLASS: u8 = 2;
const BIT_FIELD_CLASS: u8 = 4;

const fn standard_size(size: u32) -> NonZeroU32 {
    // Every caller supplies a positive standard numeric width.
    NonZeroU32::new(size).expect("standard numeric widths are positive")
}

const fn standard_precision(precision: u16) -> NonZeroU16 {
    // Every caller supplies a positive standard integer precision.
    NonZeroU16::new(precision).expect("standard integer precisions are positive")
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::zero_size(0, 0, 8, FormatError::ZeroSizedDatatype { class: FIXED_POINT_CLASS })]
    #[case::zero_precision(1, 0, 0, FormatError::InvalidFixedPointRange { size: 1, bit_offset: 0, bit_precision: 0 })]
    #[case::offset_at_storage_end(1, 8, 1, FormatError::InvalidFixedPointRange { size: 1, bit_offset: 8, bit_precision: 1 })]
    #[case::range_past_one_byte(1, 4, 5, FormatError::InvalidFixedPointRange { size: 1, bit_offset: 4, bit_precision: 5 })]
    #[case::precision_past_two_bytes(2, 0, 17, FormatError::InvalidFixedPointRange { size: 2, bit_offset: 0, bit_precision: 17 })]
    #[case::offset_at_eight_byte_end(8, 64, 1, FormatError::InvalidFixedPointRange { size: 8, bit_offset: 64, bit_precision: 1 })]
    #[case::maximum_offset_and_precision(8, u16::MAX, u16::MAX, FormatError::InvalidFixedPointRange { size: 8, bit_offset: u16::MAX, bit_precision: u16::MAX })]
    fn rejects_fixed_point_range(
        #[case] size: u32,
        #[case] offset: u16,
        #[case] precision: u16,
        #[case] expected: FormatError,
    ) {
        assert_eq!(
            FixedPointType::new(
                size,
                DatatypeByteOrder::LittleEndian,
                false,
                offset,
                precision
            )
            .unwrap_err(),
            expected
        );
    }

    #[rstest]
    #[case::one_byte_high_nibble(1, 4, 4)]
    #[case::three_byte_offset_range(3, 7, 13)]
    #[case::sixteen_byte_offset_range(16, 1, 127)]
    #[case::maximum_size_and_fields(u32::MAX, u16::MAX, u16::MAX)]
    fn accepts_fixed_point_range(
        #[case] size: u32,
        #[case] offset: u16,
        #[case] precision: u16,
        #[values(false, true)] signed: bool,
        #[values(DatatypeByteOrder::LittleEndian, DatatypeByteOrder::BigEndian)]
        order: DatatypeByteOrder,
    ) {
        let numeric = FixedPointType::new(size, order, signed, offset, precision).unwrap();
        assert_eq!(numeric.size().get(), size);
        assert_eq!(numeric.byte_order(), order);
        assert_eq!(
            (
                numeric.layout().signed(),
                numeric.layout().bit_offset(),
                numeric.layout().bit_precision()
            ),
            (signed, offset, precision)
        );
    }

    #[rstest]
    fn preserves_fixed_point_layout_padding(
        #[values(false, true)] low: bool,
        #[values(false, true)] high: bool,
    ) {
        let layout = FixedPointType::u16().with_padding(low, high).layout();
        assert_eq!(
            FixedPointType::from_layout(2, DatatypeByteOrder::BigEndian, layout).unwrap(),
            FixedPointType::new(2, DatatypeByteOrder::BigEndian, false, 0, 16)
                .unwrap()
                .with_padding(low, high),
        );
    }

    #[rstest]
    fn preserves_floating_point_layout_padding(
        #[values(false, true)] low: bool,
        #[values(false, true)] high: bool,
        #[values(false, true)] internal: bool,
    ) {
        let layout = FloatingPointType::ieee754_binary32(DatatypeByteOrder::LittleEndian)
            .with_padding(low, high, internal)
            .layout();
        assert_eq!(
            FloatingPointType::from_layout(4, DatatypeByteOrder::BigEndian, layout).unwrap(),
            FloatingPointType::ieee754_binary32(DatatypeByteOrder::BigEndian)
                .with_padding(low, high, internal),
        );
    }

    #[test]
    fn rejects_fixed_point_layout_with_a_smaller_size() {
        let layout = FixedPointType::u64().layout();
        assert_eq!(
            FixedPointType::from_layout(1, DatatypeByteOrder::LittleEndian, layout).unwrap_err(),
            FormatError::InvalidFixedPointRange {
                size: 1,
                bit_offset: 0,
                bit_precision: 64
            }
        );
    }

    #[rstest]
    #[case::zero_size(0, FloatingFields::BINARY32, FormatError::ZeroSizedDatatype { class: FLOATING_POINT_CLASS })]
    #[case::zero_precision(4, FloatingFields { bit_precision: 0, ..FloatingFields::BINARY32 }, FormatError::InvalidFloatingPointRange { size: 4, bit_offset: 0, bit_precision: 0 })]
    #[case::precision_past_offset_end(4, FloatingFields { bit_offset: 1, ..FloatingFields::BINARY32 }, FormatError::InvalidFloatingPointRange { size: 4, bit_offset: 1, bit_precision: 32 })]
    #[case::binary32_in_two_bytes(2, FloatingFields::BINARY32, FormatError::InvalidFloatingPointRange { size: 2, bit_offset: 0, bit_precision: 32 })]
    #[case::sign_at_precision_end(4, FloatingFields { sign_location: 32, ..FloatingFields::BINARY32 }, FormatError::InvalidFloatingPointField { field: FloatingPointField::Sign, location: 32, size: 1, bit_offset: 0, bit_precision: 32 })]
    #[case::zero_exponent_size(4, FloatingFields { exponent_size: 0, ..FloatingFields::BINARY32 }, FormatError::InvalidFloatingPointField { field: FloatingPointField::Exponent, location: 23, size: 0, bit_offset: 0, bit_precision: 32 })]
    #[case::exponent_past_precision_end(4, FloatingFields { exponent_size: 10, ..FloatingFields::BINARY32 }, FormatError::InvalidFloatingPointField { field: FloatingPointField::Exponent, location: 23, size: 10, bit_offset: 0, bit_precision: 32 })]
    #[case::zero_mantissa_size(4, FloatingFields { mantissa_size: 0, ..FloatingFields::BINARY32 }, FormatError::InvalidFloatingPointField { field: FloatingPointField::Mantissa, location: 0, size: 0, bit_offset: 0, bit_precision: 32 })]
    #[case::mantissa_past_precision_end(4, FloatingFields { mantissa_location: 250, mantissa_size: 255, ..FloatingFields::BINARY32 }, FormatError::InvalidFloatingPointField { field: FloatingPointField::Mantissa, location: 250, size: 255, bit_offset: 0, bit_precision: 32 })]
    #[case::mantissa_below_bit_offset(5, FloatingFields { bit_offset: 1, ..FloatingFields::BINARY32 }, FormatError::InvalidFloatingPointField { field: FloatingPointField::Mantissa, location: 0, size: 23, bit_offset: 1, bit_precision: 32 })]
    fn rejects_floating_point_range_or_field(
        #[case] size: u32,
        #[case] fields: FloatingFields,
        #[case] expected: FormatError,
    ) {
        assert_eq!(
            fields
                .parse(size, DatatypeByteOrder::LittleEndian)
                .unwrap_err(),
            expected
        );
    }

    #[rstest]
    #[case::sign_overlaps_exponent(FloatingFields { sign_location: 23, ..FloatingFields::BINARY32 }, FloatingPointField::Sign, 23, 1, FloatingPointField::Exponent, 23, 8)]
    #[case::sign_overlaps_mantissa(FloatingFields { sign_location: 0, ..FloatingFields::BINARY32 }, FloatingPointField::Sign, 0, 1, FloatingPointField::Mantissa, 0, 23)]
    #[case::mantissa_overlaps_exponent_start(FloatingFields { mantissa_size: 24, ..FloatingFields::BINARY32 }, FloatingPointField::Exponent, 23, 8, FloatingPointField::Mantissa, 0, 24)]
    #[case::mantissa_matches_exponent(FloatingFields { mantissa_location: 23, mantissa_size: 8, ..FloatingFields::BINARY32 }, FloatingPointField::Exponent, 23, 8, FloatingPointField::Mantissa, 23, 8)]
    fn rejects_overlapping_floating_point_fields(
        #[case] fields: FloatingFields,
        #[case] first: FloatingPointField,
        #[case] first_location: u8,
        #[case] first_size: u8,
        #[case] second: FloatingPointField,
        #[case] second_location: u8,
        #[case] second_size: u8,
    ) {
        assert_eq!(
            fields
                .parse(4, DatatypeByteOrder::LittleEndian)
                .unwrap_err(),
            FormatError::OverlappingFloatingPointFields {
                first,
                first_location,
                first_size,
                second,
                second_location,
                second_size
            }
        );
    }

    #[rstest]
    #[case::binary32(4, FloatingFields::BINARY32)]
    #[case::binary64(8, FloatingFields::BINARY64)]
    #[case::binary64_in_sixteen_bytes(16, FloatingFields::BINARY64)]
    #[case::binary64_in_maximum_size(u32::MAX, FloatingFields::BINARY64)]
    #[case::offset_binary32(5, FloatingFields { sign_location: 32, bit_offset: 1, exponent_location: 24, mantissa_location: 1, ..FloatingFields::BINARY32 })]
    #[case::reduced_mantissa_and_exponent(4, FloatingFields { mantissa_size: 20, exponent_size: 7, ..FloatingFields::BINARY32 })]
    fn accepts_floating_point_fields(
        #[case] size: u32,
        #[case] fields: FloatingFields,
        #[values(
            DatatypeByteOrder::LittleEndian,
            DatatypeByteOrder::BigEndian,
            DatatypeByteOrder::Vax
        )]
        order: DatatypeByteOrder,
    ) {
        let numeric = fields.parse(size, order).unwrap();
        assert_eq!(numeric.size().get(), size);
        assert_eq!(numeric.byte_order(), order);
        fields.assert_layout(numeric.layout());
    }

    #[rstest]
    #[case::zero_size(0, 0, 1)]
    #[case::zero_precision(1, 0, 0)]
    #[case::range_past_one_byte(1, 4, 5)]
    #[case::maximum_offset_and_precision(8, u16::MAX, u16::MAX)]
    fn rejects_bit_field_range(
        #[case] size: u32,
        #[case] bit_offset: u16,
        #[case] bit_precision: u16,
    ) {
        let expected = if size == 0 {
            FormatError::ZeroSizedDatatype {
                class: BIT_FIELD_CLASS,
            }
        } else {
            FormatError::InvalidNumericBitRange {
                class: BIT_FIELD_CLASS,
                size,
                bit_offset,
                bit_precision,
            }
        };
        assert_eq!(
            BitFieldType::new(
                size,
                DatatypeByteOrder::LittleEndian,
                bit_offset,
                bit_precision
            )
            .unwrap_err(),
            expected
        );
    }

    #[rstest]
    #[case::zero_size(0, 1)]
    #[case::zero_precision(1, 0)]
    #[case::precision_past_one_byte(1, 9)]
    fn rejects_time_precision(#[case] size: u32, #[case] bit_precision: u16) {
        let expected = if size == 0 {
            FormatError::ZeroSizedDatatype { class: TIME_CLASS }
        } else {
            FormatError::InvalidNumericBitRange {
                class: TIME_CLASS,
                size,
                bit_offset: 0,
                bit_precision,
            }
        };
        assert_eq!(
            TimeType::new(size, DatatypeByteOrder::LittleEndian, bit_precision).unwrap_err(),
            expected
        );
    }
    #[rstest]
    #[case::fixed_point_vax(FixedPointType::new(1, DatatypeByteOrder::Vax, false, 0, 8).unwrap_err(), FIXED_POINT_CLASS)]
    #[case::time_vax(TimeType::new(1, DatatypeByteOrder::Vax, 8).unwrap_err(), TIME_CLASS)]
    #[case::bit_field_vax(BitFieldType::new(1, DatatypeByteOrder::Vax, 0, 8).unwrap_err(), BIT_FIELD_CLASS)]
    fn rejects_invalid_numeric_byte_order(#[case] error: FormatError, #[case] class: u8) {
        assert_eq!(
            error,
            FormatError::InvalidNumericByteOrder {
                class,
                byte_order: DatatypeByteOrder::Vax
            }
        );
    }
    #[test]
    fn rejects_floating_point_layout_with_a_smaller_size() {
        assert_eq!(
            FloatingPointType::from_layout(
                1,
                DatatypeByteOrder::LittleEndian,
                FloatingPointLayout::IEEE754_BINARY64
            )
            .unwrap_err(),
            FormatError::InvalidFloatingPointRange {
                size: 1,
                bit_offset: 0,
                bit_precision: 64
            }
        );
    }

    #[derive(Clone, Copy)]
    struct FloatingFields {
        mantissa_normalization: MantissaNormalization,
        sign_location: u8,
        bit_offset: u16,
        bit_precision: u16,
        exponent_location: u8,
        exponent_size: u8,
        mantissa_location: u8,
        mantissa_size: u8,
        exponent_bias: u32,
    }

    impl FloatingFields {
        const fn from_layout(layout: FloatingPointLayout) -> Self {
            let FloatingPointLayout {
                low_padding: _,
                high_padding: _,
                internal_padding: _,
                mantissa_normalization,
                sign_location,
                bit_offset,
                bit_precision,
                exponent_location,
                exponent_size,
                mantissa_location,
                mantissa_size,
                exponent_bias,
            } = layout;
            Self {
                mantissa_normalization,
                sign_location,
                bit_offset,
                bit_precision,
                exponent_location,
                exponent_size,
                mantissa_location,
                mantissa_size,
                exponent_bias,
            }
        }

        fn parse(
            self,
            size: u32,
            order: DatatypeByteOrder,
        ) -> Result<FloatingPointType, FormatError> {
            let Self {
                mantissa_normalization,
                sign_location,
                bit_offset,
                bit_precision,
                exponent_location,
                exponent_size,
                mantissa_location,
                mantissa_size,
                exponent_bias,
            } = self;
            FloatingPointType::new(
                size,
                order,
                mantissa_normalization,
                sign_location,
                bit_offset,
                bit_precision,
                exponent_location,
                exponent_size,
                mantissa_location,
                mantissa_size,
                exponent_bias,
            )
        }

        fn assert_layout(self, layout: FloatingPointLayout) {
            let Self {
                mantissa_normalization,
                sign_location,
                bit_offset,
                bit_precision,
                exponent_location,
                exponent_size,
                mantissa_location,
                mantissa_size,
                exponent_bias,
            } = self;
            assert_eq!(
                (
                    layout.mantissa_normalization(),
                    layout.sign_location(),
                    layout.bit_offset(),
                    layout.bit_precision(),
                    layout.exponent_location(),
                    layout.exponent_size(),
                    layout.mantissa_location(),
                    layout.mantissa_size(),
                    layout.exponent_bias()
                ),
                (
                    mantissa_normalization,
                    sign_location,
                    bit_offset,
                    bit_precision,
                    exponent_location,
                    exponent_size,
                    mantissa_location,
                    mantissa_size,
                    exponent_bias
                )
            );
        }

        const BINARY32: Self = Self::from_layout(FloatingPointLayout::IEEE754_BINARY32);
        const BINARY64: Self = Self::from_layout(FloatingPointLayout::IEEE754_BINARY64);
    }
}
