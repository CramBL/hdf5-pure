/// The maximum size of one dimension of a dataspace.
///
/// A dataspace message stores a maximum size for each dimension, and the special "unlimited
/// size" marks a dimension the data may expand along indefinitely. The maximum sizes are
/// defined in "The Dataspace Message" of the [format specification, version 4.0][spec]. The C
/// library calls the unlimited size `H5S_UNLIMITED`.
///
/// # Examples
///
/// ```
/// use hdf5_pure_core::MaxExtent;
///
/// // A dataset that grows along its first dimension and keeps three columns.
/// let maxshape = [MaxExtent::Unlimited, MaxExtent::Fixed(3)];
/// assert_eq!(maxshape[0].size(), None);
/// assert_eq!(maxshape[1].size(), Some(3));
/// ```
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_simple
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MaxExtent {
    /// The dimension grows to this many elements and no further.
    Fixed(u64),
    /// The dimension grows without bound.
    Unlimited,
}

impl MaxExtent {
    /// Returns the number of elements the dimension reaches at most, or `None` for a
    /// dimension that grows without bound.
    pub const fn size(self) -> Option<u64> {
        match self {
            Self::Fixed(size) => Some(size),
            Self::Unlimited => None,
        }
    }
}
