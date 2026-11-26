//! Integer datatypes with standard widths and configurable significant bit ranges.

mod builder;
mod definition;

pub use self::builder::FixedPointTypeBuilder;
pub use self::definition::{FixedPointLayout, FixedPointType};
