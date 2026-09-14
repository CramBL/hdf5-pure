//! HDF5 Link Info message parsing (message type 0x0002).

use crate::address::StoredAddress;
use crate::bytes::{ensure_len, read_optional_offset};
use crate::error::FormatError;

/// Parsed Link Info message from a v2 group object header.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkInfoMessage {
    /// Maximum creation order value (if tracking is enabled).
    pub max_creation_order: Option<u64>,
    /// Address of fractal heap for dense link storage. None means compact storage only.
    pub fractal_heap_address: Option<StoredAddress>,
    /// Address of B-tree v2 for name-ordered link index.
    pub btree_name_index_address: Option<StoredAddress>,
    /// Address of B-tree v2 for creation-order link index.
    pub btree_creation_order_address: Option<StoredAddress>,
}

impl LinkInfoMessage {
    /// Parse a Link Info message from raw message data.
    pub fn parse(data: &[u8], offset_size: u8) -> Result<LinkInfoMessage, FormatError> {
        ensure_len(data, 0, 2)?;

        let version = data[0];
        if version != 0 {
            return Err(FormatError::InvalidLinkInfoVersion(version));
        }

        let flags = data[1];
        let has_max_creation_order = flags & 0x01 != 0;
        let has_creation_order_index = flags & 0x02 != 0;

        let mut pos = 2;

        let max_creation_order = if has_max_creation_order {
            ensure_len(data, pos, 8)?;
            let v = u64::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]);
            pos += 8;
            Some(v)
        } else {
            None
        };

        let fractal_heap_address =
            read_optional_offset(data, pos, offset_size)?.map(StoredAddress::new);
        pos += offset_size as usize;

        let btree_name_index_address =
            read_optional_offset(data, pos, offset_size)?.map(StoredAddress::new);
        pos += offset_size as usize;

        let btree_creation_order_address = if has_creation_order_index {
            read_optional_offset(data, pos, offset_size)?.map(StoredAddress::new)
        } else {
            None
        };

        Ok(LinkInfoMessage {
            max_creation_order,
            fractal_heap_address,
            btree_name_index_address,
            btree_creation_order_address,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_storage() {
        // version=0, flags=0, fractal_heap=undef, btree=undef
        let mut data = vec![0u8; 2 + 8 + 8];
        data[0] = 0; // version
        data[1] = 0; // flags
        // fractal heap address = undefined
        data[2..10].copy_from_slice(&0xFFFF_FFFF_FFFF_FFFFu64.to_le_bytes());
        // btree name index = undefined
        data[10..18].copy_from_slice(&0xFFFF_FFFF_FFFF_FFFFu64.to_le_bytes());

        let msg = LinkInfoMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.fractal_heap_address, None);
        assert_eq!(msg.btree_name_index_address, None);
        assert_eq!(msg.max_creation_order, None);
        assert_eq!(msg.btree_creation_order_address, None);
    }

    #[test]
    fn dense_storage_with_creation_order() {
        // flags: bit 0 (max creation order) + bit 1 (creation order index)
        let mut data = Vec::new();
        data.push(0); // version
        data.push(0x03); // flags
        data.extend_from_slice(&42u64.to_le_bytes()); // max_creation_order
        data.extend_from_slice(&0x1000u64.to_le_bytes()); // fractal heap
        data.extend_from_slice(&0x2000u64.to_le_bytes()); // btree name
        data.extend_from_slice(&0x3000u64.to_le_bytes()); // btree creation order

        let msg = LinkInfoMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.max_creation_order, Some(42));
        assert_eq!(msg.fractal_heap_address, Some(StoredAddress::new(0x1000)));
        assert_eq!(
            msg.btree_name_index_address,
            Some(StoredAddress::new(0x2000))
        );
        assert_eq!(
            msg.btree_creation_order_address,
            Some(StoredAddress::new(0x3000))
        );
    }

    #[test]
    fn no_creation_order_tracking() {
        let mut data = Vec::new();
        data.push(0); // version
        data.push(0x00); // flags: nothing
        data.extend_from_slice(&0x500u64.to_le_bytes()); // fractal heap
        data.extend_from_slice(&0x600u64.to_le_bytes()); // btree name

        let msg = LinkInfoMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.max_creation_order, None);
        assert_eq!(msg.fractal_heap_address, Some(StoredAddress::new(0x500)));
        assert_eq!(
            msg.btree_name_index_address,
            Some(StoredAddress::new(0x600))
        );
        assert_eq!(msg.btree_creation_order_address, None);
    }

    #[test]
    fn invalid_version() {
        let data = vec![1, 0, 0, 0];
        let err = LinkInfoMessage::parse(&data, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidLinkInfoVersion(1));
    }

    #[test]
    fn four_byte_offsets() {
        let mut data = Vec::new();
        data.push(0); // version
        data.push(0x00); // flags
        data.extend_from_slice(&0x100u32.to_le_bytes()); // fractal heap
        data.extend_from_slice(&0x200u32.to_le_bytes()); // btree name

        let msg = LinkInfoMessage::parse(&data, 4).unwrap();
        assert_eq!(msg.fractal_heap_address, Some(StoredAddress::new(0x100)));
        assert_eq!(
            msg.btree_name_index_address,
            Some(StoredAddress::new(0x200))
        );
    }
}
