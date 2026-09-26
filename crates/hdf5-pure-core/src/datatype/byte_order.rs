/// Byte order of numeric data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DatatypeByteOrder {
    /// Stores the least significant byte first.
    LittleEndian,
    /// Stores the most significant byte first.
    BigEndian,
    /// Uses VAX byte order.
    Vax,
}
