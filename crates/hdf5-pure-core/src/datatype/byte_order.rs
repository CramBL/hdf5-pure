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
