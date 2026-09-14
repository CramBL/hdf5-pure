//! HDF5 Attribute Info message parsing (message type 0x0015).
//!
//! The Attribute Info message describes dense attribute storage: a fractal heap
//! and B-tree v2 indexes for attribute lookup by name or creation order.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::bytes::{ensure_len, read_optional_offset};
use crate::error::FormatError;

/// Parsed Attribute Info message from an object header.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeInfoMessage {
    /// Maximum creation order index — the next one the object would hand out —
    /// present exactly when the object tracks attribute creation order.
    pub max_creation_index: Option<u16>,
    /// Whether the object *indexes* attributes by creation order (flags bit 1).
    ///
    /// Distinct from [`btree_creation_order_address`](Self::btree_creation_order_address)
    /// being `Some`: an object that indexes creation order while storing its
    /// attributes compactly declares the index and leaves its address undefined,
    /// which is what the reference C library writes before a set goes dense.
    pub indexes_creation_order: bool,
    /// Address of the fractal heap storing attribute messages.
    pub fractal_heap_address: Option<StoredAddress>,
    /// Address of B-tree v2 (type 8) for name-ordered attribute index.
    pub btree_name_index_address: Option<StoredAddress>,
    /// Address of B-tree v2 (type 9) for creation-order attribute index.
    pub btree_creation_order_address: Option<StoredAddress>,
}

impl AttributeInfoMessage {
    /// Parse an Attribute Info message from raw message data.
    ///
    /// Layout: version(1) + flags(1) + [max_creation_index(2)] +
    ///         fractal_heap_address(os) + btree_name_index(os) +
    ///         [btree_creation_order(os)]
    pub fn parse(data: &[u8], offset_size: u8) -> Result<AttributeInfoMessage, FormatError> {
        ensure_len(data, 0, 2)?;

        let version = data[0];
        if version != 0 {
            return Err(FormatError::InvalidAttributeInfoVersion(version));
        }

        let flags = data[1];
        let has_max_creation_index = flags & 0x01 != 0;
        let has_creation_order_index = flags & 0x02 != 0;

        let mut pos = 2;

        let max_creation_index = if has_max_creation_index {
            ensure_len(data, pos, 2)?;
            let v = u16::from_le_bytes([data[pos], data[pos + 1]]);
            pos += 2;
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

        Ok(AttributeInfoMessage {
            max_creation_index,
            indexes_creation_order: has_creation_order_index,
            fractal_heap_address,
            btree_name_index_address,
            btree_creation_order_address,
        })
    }

    /// Encode this message's body, the inverse of [`parse`](Self::parse).
    ///
    /// The two optional fields are written exactly when the flags say so:
    /// [`max_creation_index`](Self::max_creation_index) being `Some` is what
    /// "attribute creation order is tracked" means on the wire, and
    /// [`indexes_creation_order`](Self::indexes_creation_order) is what adds the
    /// creation-order B-tree address — undefined while the attributes are still
    /// stored in the object header.
    pub(crate) fn serialize(&self, offset_size: u8) -> Vec<u8> {
        let mut data = Vec::with_capacity(2 + 2 + 3 * offset_size as usize);
        data.push(0); // version
        let flags = u8::from(self.max_creation_index.is_some())
            | (u8::from(self.indexes_creation_order) << 1);
        data.push(flags);
        if let Some(max) = self.max_creation_index {
            data.extend_from_slice(&max.to_le_bytes());
        }
        write_offset(&mut data, self.fractal_heap_address, offset_size);
        write_offset(&mut data, self.btree_name_index_address, offset_size);
        if self.indexes_creation_order {
            write_offset(&mut data, self.btree_creation_order_address, offset_size);
        }
        data
    }
}

/// Write an address, or the all-ones "undefined" value for `None`, in
/// `offset_size` little-endian bytes.
fn write_offset(data: &mut Vec<u8>, address: Option<StoredAddress>, offset_size: u8) {
    let value = address.map_or(u64::MAX, StoredAddress::get);
    for i in 0..offset_size as usize {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "one byte of the address at a time"
        )]
        data.push((value >> (8 * i)) as u8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_compact_storage() {
        // version=0, flags=0, fractal_heap=undef, btree=undef
        let mut data = vec![0u8; 2 + 8 + 8];
        data[0] = 0; // version
        data[1] = 0; // flags
        data[2..10].copy_from_slice(&0xFFFF_FFFF_FFFF_FFFFu64.to_le_bytes());
        data[10..18].copy_from_slice(&0xFFFF_FFFF_FFFF_FFFFu64.to_le_bytes());

        let msg = AttributeInfoMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.fractal_heap_address, None);
        assert_eq!(msg.btree_name_index_address, None);
        assert_eq!(msg.max_creation_index, None);
        assert_eq!(msg.btree_creation_order_address, None);
    }

    #[test]
    fn parse_dense_storage() {
        let mut data = Vec::new();
        data.push(0); // version
        data.push(0x00); // flags: no creation order
        data.extend_from_slice(&0x1000u64.to_le_bytes()); // fractal heap
        data.extend_from_slice(&0x2000u64.to_le_bytes()); // btree name

        let msg = AttributeInfoMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.fractal_heap_address, Some(StoredAddress::new(0x1000)));
        assert_eq!(
            msg.btree_name_index_address,
            Some(StoredAddress::new(0x2000))
        );
        assert_eq!(msg.max_creation_index, None);
        assert_eq!(msg.btree_creation_order_address, None);
    }

    #[test]
    fn parse_dense_with_creation_order() {
        let mut data = Vec::new();
        data.push(0); // version
        data.push(0x03); // flags: max_creation_index + creation_order_index
        data.extend_from_slice(&42u16.to_le_bytes()); // max_creation_index
        data.extend_from_slice(&0x1000u64.to_le_bytes()); // fractal heap
        data.extend_from_slice(&0x2000u64.to_le_bytes()); // btree name
        data.extend_from_slice(&0x3000u64.to_le_bytes()); // btree creation order

        let msg = AttributeInfoMessage::parse(&data, 8).unwrap();
        assert_eq!(msg.max_creation_index, Some(42));
        assert!(msg.indexes_creation_order);
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

    /// Every shape of the message survives a round trip through
    /// [`AttributeInfoMessage::serialize`], which is what lets the editor rewrite
    /// one it read rather than rebuilding it from assumptions.
    #[test]
    fn serialize_round_trips_every_flag_combination() {
        for max in [None, Some(7u16)] {
            for indexed in [false, true] {
                for heap in [None, Some(StoredAddress::new(0x1000))] {
                    let msg = AttributeInfoMessage {
                        max_creation_index: max,
                        indexes_creation_order: indexed,
                        fractal_heap_address: heap,
                        btree_name_index_address: heap.map(|_| StoredAddress::new(0x2000)),
                        btree_creation_order_address: (indexed && heap.is_some())
                            .then_some(StoredAddress::new(0x3000)),
                    };
                    let bytes = msg.serialize(8);
                    assert_eq!(
                        AttributeInfoMessage::parse(&bytes, 8).unwrap(),
                        msg,
                        "round trip for max={max:?} indexed={indexed} heap={heap:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn parse_four_byte_offsets() {
        let mut data = Vec::new();
        data.push(0); // version
        data.push(0x00); // flags
        data.extend_from_slice(&0x100u32.to_le_bytes()); // fractal heap
        data.extend_from_slice(&0x200u32.to_le_bytes()); // btree name

        let msg = AttributeInfoMessage::parse(&data, 4).unwrap();
        assert_eq!(msg.fractal_heap_address, Some(StoredAddress::new(0x100)));
        assert_eq!(
            msg.btree_name_index_address,
            Some(StoredAddress::new(0x200))
        );
    }

    #[test]
    fn invalid_version() {
        let data = vec![1, 0, 0, 0];
        let err = AttributeInfoMessage::parse(&data, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidAttributeInfoVersion(1));
    }

    #[test]
    fn truncated_data() {
        let data = vec![0u8]; // too short
        let err = AttributeInfoMessage::parse(&data, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn undefined_addresses_four_byte() {
        let mut data = Vec::new();
        data.push(0); // version
        data.push(0x00); // flags
        data.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // fractal heap undef
        data.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes()); // btree undef

        let msg = AttributeInfoMessage::parse(&data, 4).unwrap();
        assert_eq!(msg.fractal_heap_address, None);
        assert_eq!(msg.btree_name_index_address, None);
    }
}
