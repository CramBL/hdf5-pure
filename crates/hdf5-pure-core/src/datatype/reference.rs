use core::fmt;

/// The kind of target a reference identifies.
///
/// The format also defines attribute references in "The Datatype Message" of
/// the [format specification, version 4.0][spec], so this enum is non-exhaustive.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ReferenceType {
    /// Refers to an object in the file.
    Object,
    /// Refers to a region of a dataset.
    DatasetRegion,
}

impl fmt::Display for ReferenceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Self::Object => "object_ref",
            Self::DatasetRegion => "region_ref",
        })
    }
}
