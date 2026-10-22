use core::num::NonZeroU16;

use crate::datatype;
use crate::datatype::byte_order::FixedWidthByteOrder;
use crate::datatype::layout::{
    FloatingPointLayout, NumericElementSize, StandardNumericLayout, StandardWidth,
};
use crate::{Datatype, FixedPointType, FormatError};

/// A supported numeric encoding and its element width.
///
/// Fixed-point ranges are nonempty and fit within one to eight bytes. Floating-point encodings
/// are IEEE 754 binary32 or binary64. Byte order is little-endian or big-endian, and enumeration
/// sizes match their base types.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NumericDecodePlan {
    element_size: NumericElementSize,
    encoding: NumericEncoding,
}

impl NumericDecodePlan {
    fn fixed_point(datatype: FixedPointType) -> Result<Self, FormatError> {
        let element_size = NumericElementSize::new(datatype.size())?;
        let order = FixedWidthByteOrder::try_from(datatype.byte_order())
            .map_err(|_order| FormatError::UnsupportedVaxByteOrder)?;
        let layout = datatype.layout();
        let signed = layout.signed();
        let bit_offset = layout.bit_offset();
        let precision = layout.precision();
        let width = match (element_size.get(), bit_offset, precision.get()) {
            (1, 0, 8) => Some(StandardWidth::OneByte),
            (2, 0, 16) => Some(StandardWidth::TwoBytes),
            (4, 0, 32) => Some(StandardWidth::FourBytes),
            (8, 0, 64) => Some(StandardWidth::EightBytes),
            _ => None,
        };
        let encoding = match width {
            Some(width) => NumericEncoding::StandardFixedPoint {
                signed,
                layout: StandardNumericLayout { width, order },
            },
            None => NumericEncoding::FixedPoint(FixedPointDecode {
                order,
                signed,
                bit_offset,
                precision,
            }),
        };
        Ok(Self {
            element_size,
            encoding,
        })
    }

    pub(crate) const fn element_size(self) -> usize {
        self.element_size.get()
    }

    pub(crate) const fn encoding(self) -> NumericEncoding {
        self.encoding
    }
}

impl TryFrom<&Datatype> for NumericDecodePlan {
    type Error = FormatError;

    fn try_from(mut datatype: &Datatype) -> Result<Self, Self::Error> {
        while let Datatype::Enumeration {
            size,
            base_type,
            members: _,
        } = datatype
        {
            let base_size = base_type.type_size();
            if *size != base_size {
                return Err(FormatError::EnumBaseSizeMismatch {
                    size: *size,
                    base_size,
                });
            }
            datatype = base_type;
        }

        match datatype {
            Datatype::FixedPoint(numeric) => Self::fixed_point(*numeric),
            Datatype::FloatingPoint(numeric) => {
                let element_size = NumericElementSize::new(numeric.size())?;
                let order = FixedWidthByteOrder::try_from(numeric.byte_order())
                    .map_err(|_order| FormatError::UnsupportedVaxByteOrder)?;
                let encoding = match (element_size.get(), numeric.layout()) {
                    (4, FloatingPointLayout::IEEE754_BINARY32) => NumericEncoding::Float32(order),
                    (8, FloatingPointLayout::IEEE754_BINARY64) => NumericEncoding::Float64(order),
                    _ => {
                        return Err(FormatError::UnsupportedFloatingPointLayout {
                            size: element_size.get(),
                            layout: numeric.layout(),
                        });
                    }
                };
                Ok(Self {
                    element_size,
                    encoding,
                })
            }
            other => Err(FormatError::TypeMismatch {
                expected: "numeric",
                actual: datatype::datatype_class_name(other),
            }),
        }
    }
}

/// The encodings supported by numeric reads, including fixed-point ranges within an element.
#[derive(Clone, Copy, Debug)]
pub(crate) enum NumericEncoding {
    FixedPoint(FixedPointDecode),
    Float32(FixedWidthByteOrder),
    Float64(FixedWidthByteOrder),
    StandardFixedPoint {
        signed: bool,
        layout: StandardNumericLayout,
    },
}

/// A nonempty fixed-point bit range within the decode plan's element width.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FixedPointDecode {
    order: FixedWidthByteOrder,
    signed: bool,
    bit_offset: u16,
    precision: NonZeroU16,
}

impl FixedPointDecode {
    pub(crate) const fn signed(self) -> bool {
        self.signed
    }

    pub(crate) fn read_unsigned(self, bytes: &[u8]) -> u64 {
        let Self {
            order,
            signed: _,
            bit_offset,
            precision,
        } = self;
        let word = match order {
            FixedWidthByteOrder::LittleEndian => bytes
                .iter()
                .rev()
                .fold(0, |word, byte| (word << 8) | u64::from(*byte)),
            FixedWidthByteOrder::BigEndian => bytes
                .iter()
                .fold(0, |word, byte| (word << 8) | u64::from(*byte)),
        };
        let mask = u64::MAX >> (64 - precision.get());
        (word >> bit_offset) & mask
    }

    pub(crate) fn read_signed(self, bytes: &[u8]) -> i64 {
        let shift = 64 - self.precision.get();
        i64::from_ne_bytes((self.read_unsigned(bytes) << shift).to_ne_bytes()) >> shift
    }
}
