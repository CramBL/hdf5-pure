use alloc::vec::Vec;

use crate::address::{StoredAddress, StoredAddressExt};
use crate::datatype::CharacterSet;
use crate::link::storage::{OwnedLink, OwnedLinkTarget};

pub(super) fn hard_link(name: Vec<u8>, creation_order: Option<u64>) -> OwnedLink {
    OwnedLink::new(
        name,
        CharacterSet::Ascii,
        OwnedLinkTarget::Hard {
            object_header_address: StoredAddress::new(512),
        },
        creation_order,
    )
    .unwrap()
}
