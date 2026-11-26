//! Datatype message encoding with canonical or retained versions.
//!
//! The property layouts are defined in "The Datatype Message" of the
//! [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage

pub(super) use self::output::{NodeHeader, Output};
pub use self::prepared::PreparedDatatype;

mod container;
mod output;
mod prepared;
mod retained;
mod source;
