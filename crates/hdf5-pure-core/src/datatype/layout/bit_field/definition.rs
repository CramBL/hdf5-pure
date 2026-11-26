use core::num::{NonZeroU16, NonZeroU32};

use super::builder::BitFieldTypeBuilder;
use crate::{DatatypeByteOrder, FormatError};

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

// The format specification, version 4.0, defines this class code in "The Datatype Message".
pub(super) const BIT_FIELD_CLASS: u8 = 4;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
