use test_util::object_header::Message;
use test_util::object_header::v2::{self as independent, HeaderFlags};

use crate::object_header::MessageRecordLayout;

/// Selects a four-byte Size of Chunk #0 field, as defined in "Version 2 Data Object Header Prefix"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
pub(super) const FOUR_BYTE_CHUNK_SIZE: HeaderFlags = HeaderFlags(0x02);

#[derive(Clone, Copy)]
pub(super) enum EmptyRecords {
    Empty,
    NilOnly,
}

impl EmptyRecords {
    pub(super) fn records(self, flags: HeaderFlags) -> Vec<u8> {
        match self {
            Self::Empty => Vec::new(),
            Self::NilOnly => {
                independent::message_records(&[Message::nil(17), Message::nil(1)], flags)
            }
        }
    }
}

pub(super) const TRACKED_FLAGS: HeaderFlags =
    HeaderFlags(HeaderFlags::TRACKS_CREATION_ORDER.0 | HeaderFlags::INDEXES_CREATION_ORDER.0);
pub(super) const TRACKED: MessageRecordLayout =
    MessageRecordLayout::from_header_flags(TRACKED_FLAGS.0);
/// Selects a two-byte Size of Chunk #0 field, as defined in "Version 2 Data Object Header Prefix"
/// of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
pub(super) const TWO_BYTE_CHUNK_SIZE: HeaderFlags = HeaderFlags(0x01);
