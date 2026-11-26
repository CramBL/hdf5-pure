pub(super) use self::codec::MessageDepth;
pub use self::codec::PreparedDatatype;
pub(super) use self::compound_layout::CompoundLayout;
pub(super) use self::description::positive_size;
pub use self::description::{
    DatatypeMessage, DatatypeMessageMember, DatatypeProperties, DatatypeVersion,
    FloatingPointByteOrder,
};

pub(super) mod atomic;
mod codec;
mod compound_layout;
pub(super) mod decode;
mod description;
