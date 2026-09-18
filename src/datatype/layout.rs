use core::num::NonZeroUsize;

use crate::{FormatError, datatype::byte_order::FixedWidthByteOrder};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedPointLayout {
    pub signed: bool,
    pub bit_offset: u16,
    pub bit_precision: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FloatingPointLayout {
    pub bit_offset: u16,
    pub bit_precision: u16,
    pub exponent_location: u8,
    pub exponent_size: u8,
    pub mantissa_location: u8,
    pub mantissa_size: u8,
    pub exponent_bias: u32,
}

impl FloatingPointLayout {
    pub const IEEE754_BINARY32: Self = Self {
        bit_offset: 0,
        bit_precision: 32,
        exponent_location: 23,
        exponent_size: 8,
        mantissa_location: 0,
        mantissa_size: 23,
        exponent_bias: 127,
    };
    pub const IEEE754_BINARY64: Self = Self {
        bit_offset: 0,
        bit_precision: 64,
        exponent_location: 52,
        exponent_size: 11,
        mantissa_location: 0,
        mantissa_size: 52,
        exponent_bias: 1023,
    };
}

/// The width of a numeric element in the standard layout: 1, 2, 4, or 8 bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StandardWidth {
    OneByte,
    TwoBytes,
    FourBytes,
    EightBytes,
}

impl StandardWidth {
    pub(crate) const fn bits(self) -> u16 {
        match self {
            Self::OneByte => 8,
            Self::TwoBytes => 16,
            Self::FourBytes => 32,
            Self::EightBytes => 64,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StandardNumericLayout {
    pub(crate) width: StandardWidth,
    pub(crate) order: FixedWidthByteOrder,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NumericElementSize(NonZeroUsize);

impl NumericElementSize {
    const MAX: usize = 8;

    pub fn new(size: u32) -> Result<Self, FormatError> {
        #[expect(
            clippy::map_err_ignore,
            reason = "The semantics are fully communicated through the format error"
        )]
        let size = usize::try_from(size).map_err(|_| FormatError::ValueTooLargeForPlatform {
            value: size.into(),
            target: "usize",
        })?;

        let size = NonZeroUsize::new(size).expect("parsed HDF5 datatypes are never zero-sized");

        if size.get() > Self::MAX {
            return Err(FormatError::NumericElementTooWide { size: size.get() });
        }

        Ok(Self(size))
    }

    pub(crate) const fn get(self) -> usize {
        self.0.get()
    }
}
