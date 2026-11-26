use alloc::vec::Vec;

/// One record of a version 2 B-tree, with its bytes as the node stores them.
///
/// The caller interprets the bytes according to the header's
/// [`tree_type`](crate::btree::v2::BTreeV2Header::tree_type).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BTreeV2Record {
    /// The `record_size` bytes of the record.
    pub data: Vec<u8>,
}

/// The width of the name hash that opens a type 5 record and closes a type 8 record.
pub(super) const NAME_HASH_LEN: u16 = 4;
