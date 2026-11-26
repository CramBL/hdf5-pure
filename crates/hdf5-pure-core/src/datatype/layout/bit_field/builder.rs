use super::definition::{BIT_FIELD_CLASS, BitFieldType};
use crate::{DatatypeByteOrder, FormatError, NumericDatatypeProperty};

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
