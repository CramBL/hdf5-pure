//! Object-header bytes, in both the versions the format defines.

use crate::bytes;
use crate::widths::Widths;

pub mod v1;
pub mod v2;

/// One header message record, in whichever of the two record layouts the
/// header carrying it uses.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Message {
    pub msg_type: MessageType,
    pub flags: MessageFlags,
    /// Written only by a version 2 header whose flags say its messages carry
    /// one. A version 1 header has no such field and ignores this.
    pub creation_order: u16,
    pub data: Vec<u8>,
}

impl Message {
    /// A message of `msg_type` carrying `data`, with no flag set.
    pub fn new(msg_type: MessageType, data: &[u8]) -> Self {
        Self {
            msg_type,
            flags: MessageFlags::NONE,
            creation_order: 0,
            data: data.to_vec(),
        }
    }

    /// A Nil message `size` bytes wide, which a reader skips: the padding a
    /// header carries where a message was deleted or space was reserved.
    pub fn nil(size: usize) -> Self {
        Self::new(MessageType::NIL, &vec![0; size])
    }

    /// An Object Header Continuation message pointing at the chunk of
    /// `length` bytes at `at`.
    pub fn continuation(at: u64, length: u64, widths: Widths) -> Self {
        let mut pointer = Vec::new();
        bytes::push_uint(&mut pointer, at, widths.offset);
        bytes::push_uint(&mut pointer, length, widths.length);
        Self::new(MessageType::OBJECT_HEADER_CONTINUATION, &pointer)
    }

    pub fn with_flags(mut self, flags: MessageFlags) -> Self {
        self.flags = flags;
        self
    }

    pub fn with_creation_order(mut self, creation_order: u16) -> Self {
        self.creation_order = creation_order;
        self
    }
}

/// The type a header message record declares, as the two bytes a version 1
/// record stores and the low byte of which a version 2 record stores.
///
/// A test states a type the format does not define by constructing one:
/// `MessageType(0x00FF)`.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct MessageType(pub u16);

impl MessageType {
    /// Section `subsubsec_fmt4_dataobject_hdr_msg_nil`, version 4.0, and the
    /// sections following it in the order the specification lists them.
    pub const NIL: Self = Self(0x0000);
    pub const DATASPACE: Self = Self(0x0001);
    pub const LINK_INFO: Self = Self(0x0002);
    pub const DATATYPE: Self = Self(0x0003);
    pub const FILL_VALUE_OLD: Self = Self(0x0004);
    pub const FILL_VALUE: Self = Self(0x0005);
    pub const LINK: Self = Self(0x0006);
    pub const EXTERNAL_DATA_FILES: Self = Self(0x0007);
    pub const DATA_LAYOUT: Self = Self(0x0008);
    pub const BOGUS: Self = Self(0x0009);
    pub const GROUP_INFO: Self = Self(0x000A);
    pub const FILTER_PIPELINE: Self = Self(0x000B);
    pub const ATTRIBUTE: Self = Self(0x000C);
    pub const OBJECT_COMMENT: Self = Self(0x000D);
    pub const OBJECT_MODIFICATION_TIME_OLD: Self = Self(0x000E);
    pub const SHARED_MESSAGE_TABLE: Self = Self(0x000F);
    pub const OBJECT_HEADER_CONTINUATION: Self = Self(0x0010);
    pub const SYMBOL_TABLE: Self = Self(0x0011);
    pub const OBJECT_MODIFICATION_TIME: Self = Self(0x0012);
    pub const BTREE_K_VALUES: Self = Self(0x0013);
    pub const DRIVER_INFO: Self = Self(0x0014);
    pub const ATTRIBUTE_INFO: Self = Self(0x0015);
    pub const OBJECT_REFERENCE_COUNT: Self = Self(0x0016);
    pub const FILE_SPACE_INFO: Self = Self(0x0017);

    /// A type the specification assigns to no message, for the tests about
    /// what a reader does with one it cannot name: the highest it assigns is
    /// [`Self::FILE_SPACE_INFO`].
    pub const UNKNOWN: Self = Self(0x00FF);
}

/// The flags byte of a header message record, one bit per flag.
///
/// Section `subsubsec_fmt4_dataobject_hdr_prefix_one`, version 4.0, lists the
/// bits, and both header versions use the same byte.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MessageFlags(pub u8);

impl MessageFlags {
    pub const NONE: Self = Self(0x00);
    pub const CONSTANT: Self = Self(0x01);
    pub const SHARED: Self = Self(0x02);
    pub const FORBID_SHARING: Self = Self(0x04);
    pub const FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE: Self = Self(0x08);
    pub const MARK_IF_UNKNOWN: Self = Self(0x10);
    pub const WAS_UNKNOWN: Self = Self(0x20);
    pub const SHAREABLE: Self = Self(0x40);
    pub const FAIL_IF_UNKNOWN_ALWAYS: Self = Self(0x80);
}

impl core::ops::BitOr for MessageFlags {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}
