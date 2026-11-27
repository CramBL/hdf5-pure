//! Count observations, physical field associations, and prepared count updates.
//!
//! Version 1 headers use the prefix count defined in "Version 1 Data Object Header Prefix" of the
//! [format specification, version 4.0][v1]. Version 2 headers are defined in "Version 2 Data Object
//! Header Prefix" of [the same document][v2], and their version 0 count message in "The Object
//! Reference Count Message" of [the same document][count].
//!
//! [v1]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_one
//! [v2]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
//! [count]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_refcount

mod count;
mod observer;
mod update;

pub use count::{
    ObjectReferenceCount, ObjectReferenceCountField, ObjectReferenceCountIssue,
    ObjectReferenceCountMessage,
};
pub use observer::ObjectReferenceCountObserver;
pub use update::ObjectReferenceCountUpdate;
pub use update::ObjectReferenceCountUpdateError;

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;
