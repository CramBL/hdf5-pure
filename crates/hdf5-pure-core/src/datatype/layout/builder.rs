//! Named construction of checked fixed-point and bit-field datatypes.

use core::fmt;

use super::{BIT_FIELD_CLASS, BitFieldType, FIXED_POINT_CLASS, FixedPointType};
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

/// A required property reported by [`FormatError::MissingNumericDatatypeProperty`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumericDatatypeProperty {
    /// The first significant bit within each element.
    BitOffset,
    /// The number of significant bits in each element.
    BitPrecision,
    /// The order of bytes within each element.
    ByteOrder,
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
            Self::Signed => "signed",
            Self::Size => "size",
        })
    }
}
