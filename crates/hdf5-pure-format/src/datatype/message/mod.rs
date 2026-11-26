pub(super) use self::codec::MessageDepth;
pub use self::codec::PreparedDatatype;
pub(super) use self::compound_layout::CompoundLayout;
pub(super) use self::description::positive_size;
pub use self::description::{
    CompoundV1Member, DatatypeMessage, DatatypeMessageMember, DatatypeProperties, DatatypeVersion,
    FloatingPointByteOrder,
};
#[cfg(test)]
pub(super) use self::description::{FLOAT_BYTE_ORDER_HIGH_BIT, FLOAT_BYTE_ORDER_LOW_BIT};

pub(super) mod atomic;
mod codec;
pub(super) mod compound_layout;
pub(super) mod decode;
mod description;
