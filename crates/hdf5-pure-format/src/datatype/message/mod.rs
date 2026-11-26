//! Datatype message layouts, retained descriptions, and encoding.

pub use self::codec::PreparedDatatype;
pub use self::representation::{
    CompoundV1Member, DatatypeMessage, DatatypeMessageMember, DatatypeProperties,
};
pub use self::version::{DatatypeVersion, FloatingPointByteOrder};
#[cfg(test)]
pub(super) use self::version::{FLOAT_BYTE_ORDER_HIGH_BIT, FLOAT_BYTE_ORDER_LOW_BIT};

pub(super) mod atomic;
mod codec;
pub(super) mod compound_layout;
pub(super) mod decode;
mod name_layout;
mod parse;
mod representation;
pub(super) mod version;
