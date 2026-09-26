pub use hdf5_pure_core::FixedPointLayout;
pub use hdf5_pure_core::FloatingPointLayout;

use crate::datatype::byte_order::FixedWidthByteOrder;

/// The width of a numeric element in the standard layout: 1, 2, 4, or 8 bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StandardWidth {
    OneByte,
    TwoBytes,
    FourBytes,
    EightBytes,
}

impl StandardWidth {
    /// Returns the number of bits in one element of this width.
    pub const fn bits(self) -> u16 {
        match self {
            Self::OneByte => 8,
            Self::TwoBytes => 16,
            Self::FourBytes => 32,
            Self::EightBytes => 64,
        }
    }
}

/// The width and byte order of a standard numeric element.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StandardNumericLayout {
    /// The element width.
    pub width: StandardWidth,
    /// The order of its bytes.
    pub order: FixedWidthByteOrder,
}
