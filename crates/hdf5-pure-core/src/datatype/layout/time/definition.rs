use core::num::{NonZeroU16, NonZeroU32};

use crate::{DatatypeByteOrder, FormatError};

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

// The format specification, version 4.0, defines this class code in "The Datatype Message".
const TIME_CLASS: u8 = 2;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
