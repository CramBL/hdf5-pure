use crate::address::StoredAddress;
use crate::width::UintWidth;

/// Limits continuation traversal to protect the parser from cycles.
pub(super) const MAX_CONTINUATIONS: u16 = 256;

/// Interprets the flags byte in a version 2 object header prefix.
///
/// The byte encodes the chunk size field width and the presence of optional
/// creation order, attribute phase change, and timestamp fields. Unused and
/// reserved bits are preserved in the raw value. The layout is defined in
/// "Version 2 Data Object Header Prefix" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
#[repr(transparent)]
#[derive(Clone, Copy)]
pub(super) struct HeaderFlags(u8);

impl HeaderFlags {
    const TRACKS_CREATION_ORDER: u8 = 0x04;
    const STORES_ATTRIBUTE_PHASE_CHANGE: u8 = 0x10;
    const STORES_TIMES: u8 = 0x20;

    /// Preserves every bit from the on-disk flags byte.
    pub(super) const fn new(flags: u8) -> Self {
        Self(flags)
    }

    /// Returns the on-disk flags byte unchanged.
    pub(super) const fn raw(self) -> u8 {
        self.0
    }

    /// Reports whether messages carry creation order values.
    pub(super) const fn tracks_creation_order(self) -> bool {
        self.0 & Self::TRACKS_CREATION_ORDER != 0
    }

    /// Reports whether non-default attribute phase change values are present.
    pub(super) const fn stores_attribute_phase_change(self) -> bool {
        self.0 & Self::STORES_ATTRIBUTE_PHASE_CHANGE != 0
    }

    /// Reports whether the four object timestamps are present.
    pub(super) const fn stores_times(self) -> bool {
        self.0 & Self::STORES_TIMES != 0
    }

    /// Returns the width encoded for the chunk size field.
    pub(super) fn chunk_size_width(self) -> UintWidth {
        UintWidth::from_flags(self.0)
    }
}

/// Describes the location and size of an object header continuation block.
///
/// The Object Header Continuation message stores a file address and a byte
/// length for a block containing additional header messages. The fields are
/// defined in "The Object Header Continuation Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_continuation
#[derive(Clone, Copy)]
pub(super) struct Continuation {
    pub(super) address: StoredAddress,
    pub(super) length: u64,
}
