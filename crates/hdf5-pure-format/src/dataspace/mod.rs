//! Parses versions 1 and 2 of the HDF5 Dataspace message and writes version 2.
//!
//! Both layouts are defined in "The Dataspace Message" of the
//! [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_simple

mod shape;
mod stored;

pub use self::shape::{
    Dataspace, DataspaceType, PreparedDataspace, stored_max_dimensions, stored_max_extent,
};
pub(crate) use self::stored::StoredDataspace;
