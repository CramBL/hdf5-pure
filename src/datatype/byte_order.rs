/// Byte orders handled by the fixed-width primitive fast path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FixedWidthByteOrder {
    LittleEndian,
    BigEndian,
}

impl TryFrom<DatatypeByteOrder> for FixedWidthByteOrder {
    type Error = DatatypeByteOrder;

    fn try_from(order: DatatypeByteOrder) -> Result<Self, Self::Error> {
        match order {
            DatatypeByteOrder::LittleEndian => Ok(Self::LittleEndian),
            DatatypeByteOrder::BigEndian => Ok(Self::BigEndian),
            DatatypeByteOrder::Vax => Err(DatatypeByteOrder::Vax),
        }
    }
}

/// Byte order of numeric data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DatatypeByteOrder {
    LittleEndian,
    BigEndian,
    Vax,
}
