//! Bit-field datatypes with configurable significant bit ranges and padding.

mod builder;
mod definition;

pub use self::builder::BitFieldTypeBuilder;
pub use self::definition::BitFieldType;
