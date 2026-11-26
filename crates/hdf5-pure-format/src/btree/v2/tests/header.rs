use test_util::btree::v2 as btree_v2;
use test_util::widths::Widths;

use super::*;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::error::FormatError;
use crate::width::{LengthWidth, OffsetWidth};

#[test]
fn header_size_follows_file_widths() {
    assert_eq!(
        btree_v2_header_size(OffsetWidth::Four, LengthWidth::Four),
        30
    );
    assert_eq!(
        btree_v2_header_size(OffsetWidth::Four, LengthWidth::Eight),
        34
    );
    assert_eq!(
        btree_v2_header_size(OffsetWidth::Eight, LengthWidth::Eight),
        38
    );
}

#[test]
fn a_header_parses_to_its_fields() {
    let data = btree_v2::Header::new(5, 11, 0x1000, 3).build(Widths::EIGHT);
    let hdr = BTreeV2Header::parse(&data, 0, 8, 8).unwrap();
    assert_eq!(hdr.tree_type, 5);
    assert_eq!(hdr.node_size, 512);
    assert_eq!(hdr.record_size, 11);
    assert_eq!(hdr.depth, 0);
    assert_eq!(hdr.root_node_address, StoredAddress::new(0x1000));
    assert_eq!(hdr.num_records_in_root, 3);
    assert_eq!(hdr.total_records, 3);
}

#[test]
fn a_header_without_bthd_is_an_invalid_signature() {
    let mut data = btree_v2::Header::new(5, 11, 0, 0).build(Widths::EIGHT);
    data[0] = b'X';
    let err = BTreeV2Header::parse(&data, 0, 8, 8).unwrap_err();
    assert_eq!(err, FormatError::InvalidBTreeV2Signature);
}

#[test]
fn a_header_of_version_1_is_an_invalid_version() {
    let mut data = btree_v2::Header::new(5, 11, 0, 0).build(Widths::EIGHT);
    data[4] = 1;
    let err = BTreeV2Header::parse(&data, 0, 8, 8).unwrap_err();
    assert_eq!(err, FormatError::InvalidBTreeV2Version(1));
}
