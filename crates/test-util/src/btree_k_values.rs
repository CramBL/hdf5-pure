//! Fixtures for the B-tree K Values message.

use core::num::NonZeroU16;

use crate::bytes;

/// The three positive K values of a version 0 message.
#[derive(Clone, Copy, Debug)]
pub struct BTreeKValues {
    pub indexed_storage_internal_node_k: NonZeroU16,
    pub group_internal_node_k: NonZeroU16,
    pub group_leaf_node_k: NonZeroU16,
}

impl BTreeKValues {
    /// Builds the seven-byte body in wire order.
    pub fn build(self) -> [u8; BODY_SIZE] {
        let mut body = [0; BODY_SIZE];
        bytes::set_u8_at(&mut body, VERSION_OFFSET, VERSION);
        bytes::set_u16_at(
            &mut body,
            INDEXED_STORAGE_INTERNAL_NODE_K_OFFSET,
            self.indexed_storage_internal_node_k.get(),
        );
        bytes::set_u16_at(
            &mut body,
            GROUP_INTERNAL_NODE_K_OFFSET,
            self.group_internal_node_k.get(),
        );
        bytes::set_u16_at(
            &mut body,
            GROUP_LEAF_NODE_K_OFFSET,
            self.group_leaf_node_k.get(),
        );
        body
    }
}

// Section `subsubsec_fmt4_dataobject_hdr_msg_btreek`, version 4.0, defines the version byte
// and the three two-byte fields that follow it.
const VERSION: u8 = 0;
const VERSION_OFFSET: usize = 0;
const INDEXED_STORAGE_INTERNAL_NODE_K_OFFSET: usize = 1;
const GROUP_INTERNAL_NODE_K_OFFSET: usize = 3;
const GROUP_LEAF_NODE_K_OFFSET: usize = 5;
const BODY_SIZE: usize = 7;
