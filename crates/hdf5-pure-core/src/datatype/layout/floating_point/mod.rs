//! Floating-point datatypes with IEEE 754 layouts and configurable bit fields.

mod builder;
mod definition;

pub use self::builder::FloatingPointTypeBuilder;
pub use self::definition::{
    FloatingPointField, FloatingPointLayout, FloatingPointType, MantissaNormalization,
};
