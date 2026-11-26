use core::num::{NonZeroU16, NonZeroU32};

use super::builder::FixedPointTypeBuilder;
use crate::{DatatypeByteOrder, FormatError};

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
                bit_precision: PRECISION_8,
            },
        }
    }

    /// Constructs a signed 16-bit integer datatype in little-endian byte order.
    pub const fn i16() -> Self {
        Self {
            size: SIZE_16,
            byte_order: DatatypeByteOrder::LittleEndian,
            layout: FixedPointLayout {
                low_padding: false,
                high_padding: false,
                signed: true,
                bit_offset: 0,
                bit_precision: PRECISION_16,
            },
        }
    }

    /// Constructs a signed 32-bit integer datatype in little-endian byte order.
    pub const fn i32() -> Self {
        Self {
            size: SIZE_32,
            byte_order: DatatypeByteOrder::LittleEndian,
            layout: FixedPointLayout {
                low_padding: false,
                high_padding: false,
                signed: true,
                bit_offset: 0,
                bit_precision: PRECISION_32,
            },
        }
    }

    /// Constructs a signed 64-bit integer datatype in little-endian byte order.
    pub const fn i64() -> Self {
        Self {
            size: SIZE_64,
            byte_order: DatatypeByteOrder::LittleEndian,
            layout: FixedPointLayout {
                low_padding: false,
                high_padding: false,
                signed: true,
                bit_offset: 0,
                bit_precision: PRECISION_64,
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
                bit_precision: PRECISION_8,
            },
        }
    }

    /// Constructs an unsigned 16-bit integer datatype in little-endian byte order.
    pub const fn u16() -> Self {
        Self {
            size: SIZE_16,
            byte_order: DatatypeByteOrder::LittleEndian,
            layout: FixedPointLayout {
                low_padding: false,
                high_padding: false,
                signed: false,
                bit_offset: 0,
                bit_precision: PRECISION_16,
            },
        }
    }

    /// Constructs an unsigned 32-bit integer datatype in little-endian byte order.
    pub const fn u32() -> Self {
        Self {
            size: SIZE_32,
            byte_order: DatatypeByteOrder::LittleEndian,
            layout: FixedPointLayout {
                low_padding: false,
                high_padding: false,
                signed: false,
                bit_offset: 0,
                bit_precision: PRECISION_32,
            },
        }
    }

    /// Constructs an unsigned 64-bit integer datatype in little-endian byte order.
    pub const fn u64() -> Self {
        Self {
            size: SIZE_64,
            byte_order: DatatypeByteOrder::LittleEndian,
            layout: FixedPointLayout {
                low_padding: false,
                high_padding: false,
                signed: false,
                bit_offset: 0,
                bit_precision: PRECISION_64,
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

// The format specification, version 4.0, defines this class code in "The Datatype Message".
pub(super) const FIXED_POINT_CLASS: u8 = 0;

const SIZE_16: NonZeroU32 = NonZeroU32::new(2).unwrap();
const SIZE_32: NonZeroU32 = NonZeroU32::new(4).unwrap();
const SIZE_64: NonZeroU32 = NonZeroU32::new(8).unwrap();

const PRECISION_8: NonZeroU16 = NonZeroU16::new(8).unwrap();
const PRECISION_16: NonZeroU16 = NonZeroU16::new(16).unwrap();
const PRECISION_32: NonZeroU16 = NonZeroU16::new(32).unwrap();
const PRECISION_64: NonZeroU16 = NonZeroU16::new(64).unwrap();

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
