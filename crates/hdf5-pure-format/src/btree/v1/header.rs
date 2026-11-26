use crate::bytes;
use crate::error::FormatError;

/// The size in bytes of the prefix of a version 1 B-tree node that does not depend on the
/// offset width: the signature (4), the node type (1), the node level (1), and the entries
/// used (2). The two sibling addresses that follow are each `offset_size` bytes wide, see
/// [`btree_v1_node_header_size`]. The node is defined in "Version 1 B-trees" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v1
pub(super) const BTREE_V1_NODE_PREFIX_LEN: usize = 8;

/// Returns the size in bytes of a version 1 B-tree node header, the offset of the node's first key.
///
/// The header is the 8-byte prefix and the left and right sibling addresses, `offset_size` bytes
/// each.
pub const fn btree_v1_node_header_size(offset_size: u8) -> usize {
    BTREE_V1_NODE_PREFIX_LEN + (offset_size as usize) * 2
}

/// The fields preceding the sibling addresses in a version 1 B-tree node.
pub(super) struct NodePrefix {
    pub(super) node_type: u8,
    pub(super) node_level: u8,
    pub(super) entries_used: u16,
}

impl NodePrefix {
    /// Reads the first eight bytes of the node header and checks the signature.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnexpectedEof`] if the prefix is truncated or
    /// [`FormatError::InvalidBTreeSignature`] if it does not begin with `TREE`.
    pub(super) fn parse(header: &[u8]) -> Result<Self, FormatError> {
        let [s0, s1, s2, s3, node_type, node_level, used0, used1] =
            *bytes::array_at::<BTREE_V1_NODE_PREFIX_LEN>(header, 0)?;
        if [s0, s1, s2, s3] != *BTREE_V1_SIGNATURE {
            return Err(FormatError::InvalidBTreeSignature);
        }
        Ok(Self {
            node_type,
            node_level,
            entries_used: u16::from_le_bytes([used0, used1]),
        })
    }
}

/// The signature of a version 1 B-tree node, defined in "Version 1 B-trees" of the [format
/// specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v1
const BTREE_V1_SIGNATURE: &[u8; 4] = b"TREE";
