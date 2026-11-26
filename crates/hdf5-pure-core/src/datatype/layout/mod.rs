//! Checked layouts for numeric datatype elements.
//!
//! The numeric classes of datatype message versions 1 through 5 are defined in "The Datatype
//! Message" of the [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage

mod bit_field;
mod fixed_point;
mod floating_point;
mod property;
mod time;

pub use self::bit_field::{BitFieldType, BitFieldTypeBuilder};
pub use self::fixed_point::{FixedPointLayout, FixedPointType, FixedPointTypeBuilder};
pub use self::floating_point::{
    FloatingPointField, FloatingPointLayout, FloatingPointType, FloatingPointTypeBuilder,
    MantissaNormalization,
};
pub use self::property::NumericDatatypeProperty;
pub use self::time::TimeType;
