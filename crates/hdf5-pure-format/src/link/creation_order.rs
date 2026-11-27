use super::info::LinkInfoMessage;
use super::storage::LinkStorageError;

/// A group's link creation-order policy and checked next-index counter.
///
/// The policy is encoded by the tracking and indexing flags of "The Link Info Message" in
/// the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_linkinfo
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinkCreationPolicy {
    /// Tracks link creation indexes and maintains a creation-order index.
    Indexed {
        /// The index assigned to the next inserted link.
        next: LinkCreationCounter,
    },
    /// Tracks link creation indexes without a creation-order index.
    Tracked {
        /// The index assigned to the next inserted link.
        next: LinkCreationCounter,
    },
    /// Omits link creation indexes and the counter.
    Untracked,
}

impl LinkCreationPolicy {
    /// Reads tracking, indexing, and the next-index counter from Link Info metadata.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::IndexingWithoutTracking`] if indexing is enabled without
    /// tracking, and [`LinkStorageError::CreationOrderCounterRange`] if the counter exceeds
    /// the range accepted by [`LinkCreationCounter::new`].
    pub fn from_info(info: &LinkInfoMessage) -> Result<Self, LinkStorageError> {
        let next = info
            .max_creation_order()
            .map(LinkCreationCounter::new)
            .transpose()?;
        match (next, info.indexes_creation_order()) {
            (None, false) => Ok(Self::Untracked),
            (Some(next), false) => Ok(Self::Tracked { next }),
            (Some(next), true) => Ok(Self::Indexed { next }),
            (None, true) => Err(LinkStorageError::IndexingWithoutTracking),
        }
    }

    /// Assigns the next tracked index and increments the counter, or returns `None` when untracked.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::CreationOrderExhausted`] if the counter is [`i64::MAX`].
    /// An error leaves the policy unchanged.
    pub fn take(&mut self) -> Result<Option<u64>, LinkStorageError> {
        let (Self::Tracked { next } | Self::Indexed { next }) = self else {
            return Ok(None);
        };
        let index = next.get();
        let following = index
            .checked_add(1)
            .filter(|&value| value <= MAX_LINK_CREATION_COUNTER)
            .ok_or(LinkStorageError::CreationOrderExhausted)?;
        *next = LinkCreationCounter(following);
        Ok(Some(index))
    }

    /// Returns the next tracked creation index, or `None` when untracked.
    pub fn next_index(self) -> Option<u64> {
        match self {
            Self::Untracked => None,
            Self::Tracked { next } | Self::Indexed { next } => Some(next.get()),
        }
    }

    /// Returns whether creation-order indexing is enabled.
    pub fn is_indexed(self) -> bool {
        matches!(self, Self::Indexed { .. })
    }
}

/// A next-link creation counter in `0..=i64::MAX`.
///
/// libhdf5 decodes the Link Info counter as a signed value and rejects negative values in
/// `H5O__linfo_decode` in `H5Olinfo.c` (HDF5 1.14.6 and 2.2.0). A counter equal to [`i64::MAX`]
/// is representable but cannot be incremented for another link.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinkCreationCounter(u64);

impl LinkCreationCounter {
    /// Creates a counter that libhdf5 can decode as a nonnegative signed value.
    ///
    /// # Errors
    ///
    /// Returns [`LinkStorageError::CreationOrderCounterRange`] if `next` exceeds [`i64::MAX`].
    pub fn new(next: u64) -> Result<Self, LinkStorageError> {
        if next > MAX_LINK_CREATION_COUNTER {
            return Err(LinkStorageError::CreationOrderCounterRange {
                next,
                maximum: MAX_LINK_CREATION_COUNTER,
            });
        }
        Ok(Self(next))
    }

    /// Returns the stored next-index value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// The largest counter accepted by `H5O__linfo_decode` in `H5Olinfo.c` (HDF5 1.14.6 and 2.2.0).
const MAX_LINK_CREATION_COUNTER: u64 = 0x7fff_ffff_ffff_ffff;
