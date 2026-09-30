//! The Link Info message (type 0x0002) of a group: the dense storage of its links, and whether it
//! tracks and indexes their creation order.
//!
//! The message is defined in "The Link Info Message" of the [format specification, version
//! 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_linkinfo

use alloc::vec::Vec;

use crate::address::StoredAddress;
use crate::bytes;
use crate::error::FormatError;
use crate::width::OffsetWidth;

/// Parsed Link Info message from a v2 group object header.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkInfoMessage {
    /// Maximum creation order value (if tracking is enabled).
    pub max_creation_order: Option<u64>,
    /// Address of fractal heap for dense link storage. None means compact storage only.
    pub fractal_heap_address: Option<StoredAddress>,
    /// Address of B-tree v2 for name-ordered link index.
    pub btree_name_index_address: Option<StoredAddress>,
    /// The address of the version 2 B-tree that indexes the links by creation order: `None` where
    /// the group does not index creation order, and `Some(None)` where it does and the message
    /// stores the undefined address.
    pub btree_creation_order_address: Option<Option<StoredAddress>>,
}

impl LinkInfoMessage {
    /// Parses the body of a Link Info message.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidLinkInfoVersion`] if the version is not 0,
    /// [`FormatError::InvalidOffsetSize`] if `offset_size` is not 2, 4, or 8, and
    /// [`FormatError::UnexpectedEof`] if `data` ends inside a field.
    pub fn parse(data: &[u8], offset_size: u8) -> Result<LinkInfoMessage, FormatError> {
        bytes::ensure_len(data, 0, 2)?;

        let version = data[0];
        if version != LINK_INFO_VERSION {
            return Err(FormatError::InvalidLinkInfoVersion(version));
        }

        let flags = data[1];
        let has_max_creation_order = flags & TRACKS_CREATION_ORDER != 0;
        let has_creation_order_index = flags & INDEXES_CREATION_ORDER != 0;

        let mut pos = 2;

        let max_creation_order = if has_max_creation_order {
            bytes::ensure_len(data, pos, 8)?;
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
            bytes::read_optional_offset(data, pos, offset_size)?.map(StoredAddress::new);
        pos += offset_size as usize;

        let btree_name_index_address =
            bytes::read_optional_offset(data, pos, offset_size)?.map(StoredAddress::new);
        pos += offset_size as usize;

        let btree_creation_order_address = if has_creation_order_index {
            Some(bytes::read_optional_offset(data, pos, offset_size)?.map(StoredAddress::new))
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

    /// Returns the body of the message, the inverse of [`parse`](Self::parse).
    ///
    /// The Flags byte has bit 0 set where [`max_creation_order`](Self::max_creation_order) is
    /// `Some`, and bit 1 where [`btree_creation_order_address`](Self::btree_creation_order_address)
    /// is. An address of `None` is stored as the undefined address.
    pub fn serialize(&self, offset_width: OffsetWidth) -> Vec<u8> {
        let Self {
            max_creation_order,
            fractal_heap_address,
            btree_name_index_address,
            btree_creation_order_address,
        } = *self;
        let mut data = Vec::with_capacity(2 + 8 + 3 * usize::from(offset_width.get()));
        data.push(LINK_INFO_VERSION);
        let mut flags = 0;
        if max_creation_order.is_some() {
            flags |= TRACKS_CREATION_ORDER;
        }
        if btree_creation_order_address.is_some() {
            flags |= INDEXES_CREATION_ORDER;
        }
        data.push(flags);
        if let Some(max) = max_creation_order {
            data.extend_from_slice(&max.to_le_bytes());
        }
        let mut address = |address: Option<StoredAddress>| {
            bytes::write_offset(
                &mut data,
                address.map_or(u64::MAX, StoredAddress::get),
                offset_width,
            );
        };
        address(fractal_heap_address);
        address(btree_name_index_address);
        if let Some(btree_creation_order_address) = btree_creation_order_address {
            address(btree_creation_order_address);
        }
        data
    }
}

/// The Link Info message version, the one version "The Link Info Message", version 4.0, defines.
const LINK_INFO_VERSION: u8 = 0;

/// Bit 0 of the Flags field, set where the group tracks the creation order of its links, from the
/// same section as [`LINK_INFO_VERSION`].
const TRACKS_CREATION_ORDER: u8 = 0x01;

/// Bit 1 of the Flags field, set where the group indexes the creation order of its links, from the
/// same section as [`LINK_INFO_VERSION`].
const INDEXES_CREATION_ORDER: u8 = 0x02;

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::eight(8)]
    #[case::two(2)]
    fn compact_storage(#[case] offset_size: u8) {
        // version=0, flags=0, fractal_heap=undef, btree=undef
        let mut data = vec![0u8; 2];
        data.resize(2 + 2 * usize::from(offset_size), 0xFF);

        assert_eq!(
            LinkInfoMessage::parse(&data, offset_size),
            Ok(LinkInfoMessage {
                max_creation_order: None,
                fractal_heap_address: None,
                btree_name_index_address: None,
                btree_creation_order_address: None,
            })
        );
    }

    #[test]
    fn dense_storage_with_creation_order() {
        // flags: bit 0 (max creation order) + bit 1 (creation order index)
        let mut data = Vec::new();
        data.push(0); // version
        data.push(0x03); // flags
        data.extend_from_slice(&42u64.to_le_bytes()); // `max_creation_order`
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
            Some(Some(StoredAddress::new(0x3000)))
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

    #[rstest]
    #[case::compact_storage(LinkInfoMessage {
        max_creation_order: None,
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: None,
    })]
    #[case::tracked_and_indexed_dense_storage(LinkInfoMessage {
        max_creation_order: Some(42),
        fractal_heap_address: Some(StoredAddress::new(0x1000)),
        btree_name_index_address: Some(StoredAddress::new(0x2000)),
        btree_creation_order_address: Some(Some(StoredAddress::new(0x3000))),
    })]
    #[case::indexed_with_no_index_yet(LinkInfoMessage {
        max_creation_order: Some(0),
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: Some(None),
    })]
    fn a_serialized_message_parses_back_to_itself(
        #[values(OffsetWidth::Four, OffsetWidth::Eight)] offset_width: OffsetWidth,
        #[case] message: LinkInfoMessage,
    ) {
        assert_eq!(
            LinkInfoMessage::parse(&message.serialize(offset_width), offset_width.get()),
            Ok(message)
        );
    }

    #[test]
    fn a_compact_message_stores_two_undefined_addresses() {
        let message = LinkInfoMessage {
            max_creation_order: None,
            fractal_heap_address: None,
            btree_name_index_address: None,
            btree_creation_order_address: None,
        };
        let mut expected = vec![0, 0];
        expected.extend_from_slice(&[0xFF; 16]);
        assert_eq!(message.serialize(OffsetWidth::Eight), expected);
    }
}
