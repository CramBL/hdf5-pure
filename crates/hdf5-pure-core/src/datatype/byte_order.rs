use core::fmt;

/// Byte order of numeric data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DatatypeByteOrder {
    /// Stores the least significant byte first.
    LittleEndian,
    /// Stores the most significant byte first.
    BigEndian,
    /// Uses VAX byte order.
    Vax,
}

impl fmt::Display for DatatypeByteOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Self::LittleEndian => "le",
            Self::BigEndian => "be",
            Self::Vax => "vax",
        })
    }
}
