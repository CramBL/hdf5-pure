use crate::{
    Datatype, FormatError,
    datatype::{
        byte_order::{DatatypeByteOrder, FixedWidthByteOrder},
        layout::{
            FixedPointLayout, FloatingPointLayout, NumericElementSize, StandardNumericLayout,
            StandardWidth,
        },
    },
};

#[derive(Clone, Copy, Debug)]
pub(crate) enum NumericDatatype {
    FixedPoint {
        element_size: NumericElementSize,
        byte_order: DatatypeByteOrder,
        layout: FixedPointLayout,
    },
    FloatingPoint {
        element_size: NumericElementSize,
        byte_order: DatatypeByteOrder,
        layout: FloatingPointLayout,
    },
}

impl NumericDatatype {
    pub(crate) const fn element_size(self) -> NumericElementSize {
        match self {
            Self::FixedPoint { element_size, .. } | Self::FloatingPoint { element_size, .. } => {
                element_size
            }
        }
    }

    pub(crate) const fn byte_order(self) -> DatatypeByteOrder {
        match self {
            Self::FixedPoint { byte_order, .. } | Self::FloatingPoint { byte_order, .. } => {
                byte_order
            }
        }
    }

    pub(crate) const fn bit_offset(self) -> u16 {
        match self {
            Self::FixedPoint {
                layout: FixedPointLayout { bit_offset, .. },
                ..
            }
            | Self::FloatingPoint {
                layout: FloatingPointLayout { bit_offset, .. },
                ..
            } => bit_offset,
        }
    }

    pub(crate) const fn bit_precision(self) -> u16 {
        match self {
            Self::FixedPoint {
                layout: FixedPointLayout { bit_precision, .. },
                ..
            }
            | Self::FloatingPoint {
                layout: FloatingPointLayout { bit_precision, .. },
                ..
            } => bit_precision,
        }
    }

    pub(crate) fn standard_layout(self) -> Option<StandardNumericLayout> {
        let width = match self.element_size().get() {
            1 => StandardWidth::OneByte,
            2 => StandardWidth::TwoBytes,
            4 => StandardWidth::FourBytes,
            8 => StandardWidth::EightBytes,
            _ => return None,
        };

        let order = match self.byte_order() {
            DatatypeByteOrder::LittleEndian => FixedWidthByteOrder::LittleEndian,
            DatatypeByteOrder::BigEndian => FixedWidthByteOrder::BigEndian,
            DatatypeByteOrder::Vax => return None,
        };

        let is_standard = match self {
            Self::FixedPoint {
                element_size: _,
                byte_order: _,
                layout:
                    FixedPointLayout {
                        signed: _,
                        bit_offset,
                        bit_precision,
                    },
            } => bit_offset == 0 && bit_precision == width.bits(),

            Self::FloatingPoint {
                element_size: _,
                byte_order: _,
                layout,
            } => match width {
                StandardWidth::FourBytes => layout == FloatingPointLayout::IEEE754_BINARY32,
                StandardWidth::EightBytes => layout == FloatingPointLayout::IEEE754_BINARY64,
                StandardWidth::OneByte | StandardWidth::TwoBytes => false,
            },
        };

        is_standard.then_some(StandardNumericLayout { width, order })
    }
}

impl TryFrom<&Datatype> for NumericDatatype {
    type Error = FormatError;

    fn try_from(datatype: &Datatype) -> Result<Self, Self::Error> {
        match datatype {
            // HDF5 enumerations are represented numerically by their integer
            // base type, so parse that base as the effective numeric datatype.
            Datatype::Enumeration { base_type, .. } => Self::try_from(base_type.as_ref()),

            Datatype::FixedPoint {
                size,
                byte_order,
                layout:
                    FixedPointLayout {
                        signed,
                        bit_offset,
                        bit_precision,
                    },
            } => Ok(Self::FixedPoint {
                element_size: NumericElementSize::new(*size)?,
                byte_order: *byte_order,
                layout: FixedPointLayout {
                    signed: *signed,
                    bit_offset: *bit_offset,
                    bit_precision: *bit_precision,
                },
            }),

            Datatype::FloatingPoint {
                size,
                byte_order,
                layout,
            } => Ok(Self::FloatingPoint {
                element_size: NumericElementSize::new(*size)?,
                byte_order: *byte_order,
                layout: *layout,
            }),

            other => Err(FormatError::TypeMismatch {
                expected: "numeric",
                actual: other.name(),
            }),
        }
    }
}

impl TryFrom<Datatype> for NumericDatatype {
    type Error = FormatError;
    fn try_from(dt: Datatype) -> Result<Self, Self::Error> {
        NumericDatatype::try_from(&dt)
    }
}
