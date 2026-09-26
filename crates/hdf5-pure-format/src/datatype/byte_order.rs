pub use hdf5_pure_core::DatatypeByteOrder;

/// Byte orders handled by the fixed-width primitive fast path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixedWidthByteOrder {
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
