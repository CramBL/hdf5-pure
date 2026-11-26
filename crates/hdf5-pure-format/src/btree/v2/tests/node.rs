use alloc::vec;
use alloc::vec::Vec;
use core::num::NonZeroU16;

use rstest::rstest;
use test_util::btree::v2 as btree_v2;
use test_util::widths::Widths;

use super::*;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::btree::v2::BTREE_V2_FILTERED_CHUNK;
use crate::btree::v2::BTREE_V2_LINK_NAME;
use crate::btree::v2::BTreeV2Header;
use crate::error::FormatError;
use crate::width::OffsetWidth;

#[rstest]
#[case::two_byte_address(OffsetWidth::Two, 0x1_0000)]
#[case::four_byte_address(OffsetWidth::Four, 0x1_0000_0000)]
fn layouts_reject_roots_outside_the_address_width(
    #[case] width: OffsetWidth,
    #[case] address: u64,
) {
    let mut header = BTreeV2Header::parse(
        &btree_v2::Header::new(BTREE_V2_LINK_NAME, 11, 256, 1).build(Widths::EIGHT),
        0,
        8,
        8,
    )
    .unwrap();
    header.root_node_address = StoredAddress::new(address);
    let Err(error) = BTreeV2Layout::new(&header, width) else {
        panic!("expected out-of-width root");
    };
    assert_eq!(error, FormatError::InvalidBTreeV2Signature);
    let Err(error) = BTreeV2Layout::for_collection(&header, width) else {
        panic!("expected out-of-width collection root");
    };
    assert_eq!(error, FormatError::InvalidBTreeV2Signature);
}

#[rstest]
#[case::undefined_address(u64::MAX, 1, 3)]
#[case::empty_child(512, 0, 3)]
#[case::excessive_records(512, 60, 60)]
#[case::empty_subtree(512, 1, 0)]
#[case::excessive_subtree(512, 1, 100)]
fn indexed_child_rejects_invalid_selected_pointer(
    #[case] address: u64,
    #[case] records: u16,
    #[case] total: u64,
) {
    let header = BTreeV2Header::parse(
        &btree_v2::Header::new(BTREE_V2_LINK_NAME, 11, 256, 1)
            .depth(2)
            .total_records(7)
            .build(Widths::EIGHT),
        0,
        8,
        8,
    )
    .unwrap();
    let layout = BTreeV2Layout::new(&header, OffsetWidth::Eight).unwrap();
    let child = btree_v2::Child {
        address,
        records,
        records_width: 1,
        subtree: Some(btree_v2::Subtree {
            records: total,
            width: 2,
        }),
    };
    let bytes = btree_v2::internal(
        BTREE_V2_LINK_NAME,
        &[vec![0; 11]],
        &[child, child],
        Widths::EIGHT,
    );
    let view = layout.root().unwrap().parse(&bytes).unwrap();
    assert_eq!(
        view.child(0).unwrap_err(),
        FormatError::InvalidBTreeV2Signature
    );
}

#[rstest]
#[case::two_byte_addresses(OffsetWidth::Two, 512, 1, 2)]
#[case::four_byte_addresses(OffsetWidth::Four, 4096, 2, 3)]
#[case::eight_byte_addresses(OffsetWidth::Eight, 512, 1, 2)]
fn indexed_node_uses_the_layout_count_and_address_widths(
    #[case] offset_width: OffsetWidth,
    #[case] node_size: u32,
    #[case] count_width: usize,
    #[case] total_width: usize,
) {
    let widths = Widths {
        offset: usize::from(offset_width.get()),
        length: 8,
    };
    let bytes = btree_v2::Header::new(BTREE_V2_LINK_NAME, 11, 256, 1)
        .node_size(node_size)
        .depth(2)
        .total_records(7)
        .build(widths);
    let header = BTreeV2Header::parse(&bytes, 0, offset_width.get(), 8).unwrap();
    let layout = BTreeV2Layout::new(&header, offset_width).unwrap();
    let children = [512, 1024].map(|address| btree_v2::Child {
        address,
        records: 1,
        records_width: count_width,
        subtree: Some(btree_v2::Subtree {
            records: 3,
            width: total_width,
        }),
    });
    let bytes = btree_v2::internal(BTREE_V2_LINK_NAME, &[vec![7; 11]], &children, widths);
    let view = layout.root().unwrap().parse(&bytes).unwrap();
    assert_eq!(view.record_count(), 1);
    assert_eq!(view.record(0).unwrap().as_ptr(), bytes[6..].as_ptr());
    assert_eq!(view.record(0).unwrap(), &[7; 11]);
    assert_eq!(
        view.child(1).unwrap().unwrap(),
        BTreeV2Node {
            layout: &layout,
            frame: BTreeV2NodeFrame {
                address: StoredAddress::new(1024),
                kind: BTreeV2NodeKind::Internal(NonZeroU16::new(1).unwrap()),
                records: 1,
                total_records: 3,
            },
        }
    );
}

#[rstest]
#[case::empty(false, 0, 0)]
#[case::legacy_empty(true, 0, 0)]
#[case::nonempty(true, 1, 1)]
#[case::positive_inaccurate_total(true, 2, 1)]
fn collection_layout_classifies_root_compatibility(
    #[case] defined: bool,
    #[case] records: u16,
    #[case] total: u64,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
    #[values(LegacyEmptyRoot::Allow, LegacyEmptyRoot::Reject)] policy: LegacyEmptyRoot,
) {
    let header = BTreeV2Header {
        tree_type: BTREE_V2_LINK_NAME,
        node_size: 512,
        record_size: 11,
        depth: 0,
        root_node_address: if defined {
            StoredAddress::new(256)
        } else {
            StoredAddress::undefined(width)
        },
        num_records_in_root: records,
        total_records: total,
    };
    let result = BTreeV2Layout::for_collection_with_policy(&header, width, policy);
    if total == 0 && defined && matches!(policy, LegacyEmptyRoot::Reject) {
        let Err(error) = result else {
            panic!("expected rejected legacy root");
        };
        assert_eq!(error, FormatError::InvalidBTreeV2Signature);
        return;
    }
    let layout = result.unwrap();
    match layout.collection_root() {
        BTreeV2CollectionRoot::Empty => {
            assert_eq!((total, records, defined), (0, 0, false));
            assert_eq!(layout.root(), None);
        }
        BTreeV2CollectionRoot::LegacyEmpty => {
            assert_eq!((total, records, defined), (0, 0, true));
            assert_eq!(layout.root().unwrap().address(), header.root_node_address);
        }
        BTreeV2CollectionRoot::NonEmpty(root) => {
            assert!(total > 0);
            assert_eq!(root.address(), header.root_node_address);
            assert_eq!(root.total_records(), total);
        }
    }
}

#[rstest]
#[case::missing_root_records(0, 1, 0)]
#[case::missing_total_records(1, 0, 0)]
#[case::internal_empty(0, 0, 1)]
fn collection_layout_rejects_inconsistent_root_fields(
    #[case] records: u16,
    #[case] total: u64,
    #[case] depth: u16,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
    #[values(false, true)] defined: bool,
) {
    let header = BTreeV2Header {
        tree_type: BTREE_V2_LINK_NAME,
        node_size: 512,
        record_size: 11,
        depth,
        root_node_address: if defined {
            StoredAddress::new(256)
        } else {
            StoredAddress::undefined(width)
        },
        num_records_in_root: records,
        total_records: total,
    };
    let Err(error) = BTreeV2Layout::for_collection(&header, width) else {
        panic!("expected inconsistent root fields");
    };
    assert_eq!(error, FormatError::InvalidBTreeV2Signature);
}

#[rstest]
#[case::zero_record_size(512, 0, 1)]
#[case::short_node(9, 11, 1)]
#[case::records_exceed_node(512, 11, 600)]
fn collection_layout_rejects_invalid_record_geometry(
    #[case] node_size: u32,
    #[case] record_size: u16,
    #[case] records: u16,
) {
    let header = BTreeV2Header {
        tree_type: BTREE_V2_LINK_NAME,
        node_size,
        record_size,
        depth: 0,
        root_node_address: StoredAddress::new(256),
        num_records_in_root: records,
        total_records: 1,
    };
    let Err(error) = BTreeV2Layout::for_collection(&header, OffsetWidth::Eight) else {
        panic!("expected invalid collection geometry");
    };
    assert_eq!(error, FormatError::InvalidBTreeV2Signature);
}

#[test]
fn collection_layout_accepts_maximum_declared_depth_with_bounded_geometry() {
    let header = BTreeV2Header {
        tree_type: BTREE_V2_LINK_NAME,
        node_size: 512,
        record_size: 11,
        depth: u16::MAX,
        root_node_address: StoredAddress::new(256),
        num_records_in_root: 1,
        total_records: 1,
    };
    let layout = BTreeV2Layout::for_collection(&header, OffsetWidth::Eight).unwrap();
    let root = layout.root().unwrap();
    assert_eq!(root.address(), header.root_node_address);
    let bytes = btree_v2::internal(
        BTREE_V2_LINK_NAME,
        &[vec![7; 11]],
        &[btree_v2::Child {
            address: 512,
            records: 1,
            records_width: 1,
            subtree: Some(btree_v2::Subtree {
                records: 1,
                width: 8,
            }),
        }; 2],
        Widths::EIGHT,
    );
    let view = root.parse(&bytes).unwrap();
    assert_eq!(view.record(0).unwrap(), &[7; 11]);
    assert_eq!(
        view.child(0).unwrap().unwrap().address(),
        StoredAddress::new(512)
    );
}

#[rstest]
#[case::zero_record_size(|header: &mut BTreeV2Header| header.record_size = 0)]
#[case::short_node(|header: &mut BTreeV2Header| header.node_size = 9)]
#[case::record_does_not_fit(|header: &mut BTreeV2Header| header.node_size = 20)]
#[case::excessive_depth(|header: &mut BTreeV2Header| header.depth = 64)]
#[case::excessive_root_count(|header: &mut BTreeV2Header| header.num_records_in_root = 600)]
#[case::leaf_total_disagrees(|header: &mut BTreeV2Header| header.total_records = 2)]
#[case::undefined_root(|header: &mut BTreeV2Header| header.root_node_address = StoredAddress::new(u64::MAX))]
fn indexed_layout_rejects_invalid_header_geometry(#[case] change: fn(&mut BTreeV2Header)) {
    let bytes = btree_v2::Header::new(BTREE_V2_LINK_NAME, 11, 256, 1).build(Widths::EIGHT);
    let mut header = BTreeV2Header::parse(&bytes, 0, 8, 8).unwrap();
    change(&mut header);
    let Err(error) = BTreeV2Layout::new(&header, OffsetWidth::Eight) else {
        panic!("expected invalid geometry");
    };
    assert_eq!(error, FormatError::InvalidBTreeV2Signature);
}

#[rstest]
#[case::empty(u64::MAX, 0, 0, true)]
#[case::defined_root(256, 0, 0, false)]
#[case::records_in_root(u64::MAX, 1, 0, false)]
#[case::internal_root(u64::MAX, 0, 1, false)]
fn indexed_empty_layout_requires_consistent_root_fields(
    #[case] address: u64,
    #[case] records: u16,
    #[case] depth: u16,
    #[case] valid: bool,
) {
    let bytes = btree_v2::Header::new(BTREE_V2_LINK_NAME, 11, address, records)
        .depth(depth)
        .total_records(0)
        .build(Widths::EIGHT);
    let header = BTreeV2Header::parse(&bytes, 0, 8, 8).unwrap();
    match BTreeV2Layout::new(&header, OffsetWidth::Eight) {
        Ok(layout) => {
            assert!(valid);
            assert_eq!(layout.root(), None);
        }
        Err(error) => {
            assert!(!valid);
            assert_eq!(error, FormatError::InvalidBTreeV2Signature);
        }
    }
}

#[rstest]
#[case::bad_signature(0, b'X', FormatError::InvalidBTreeV2Signature)]
#[case::bad_version(4, 1, FormatError::InvalidBTreeV2Version(1))]
#[case::bad_type(5, 8, FormatError::InvalidBTreeNodeType(8))]
fn indexed_node_rejects_invalid_prefix(
    #[values(NodeFixture::Leaf, NodeFixture::Internal)] fixture: NodeFixture,
    #[case] at: usize,
    #[case] value: u8,
    #[case] expected: FormatError,
) {
    let layout = BTreeV2Layout::new(&fixture.header(), OffsetWidth::Eight).unwrap();
    let mut bytes = fixture.bytes();
    bytes[at] = value;
    let Err(error) = layout.root().unwrap().parse(&bytes) else {
        panic!("expected invalid node");
    };
    assert_eq!(error, expected);
}

#[rstest]
#[case::leaf(NodeFixture::Leaf)]
#[case::internal(NodeFixture::Internal)]
fn indexed_node_requires_the_complete_checksum_field(
    #[case] fixture: NodeFixture,
    #[values(1, 2, 3, 4)] missing: usize,
) {
    let layout = BTreeV2Layout::new(&fixture.header(), OffsetWidth::Eight).unwrap();
    let bytes = fixture.bytes();
    let Err(error) = layout
        .root()
        .unwrap()
        .parse(&bytes[..bytes.len() - missing])
    else {
        panic!("expected truncated checksum");
    };
    assert_eq!(
        error,
        FormatError::UnexpectedEof {
            expected: bytes.len(),
            available: bytes.len() - missing,
        }
    );
}

#[rstest]
#[case::root(NodeFixture::Leaf)]
#[case::child(NodeFixture::Internal)]
fn node_references_retain_their_owner_after_parent_bytes_are_dropped(#[case] fixture: NodeFixture) {
    let layout = BTreeV2Layout::new(&fixture.header(), OffsetWidth::Eight).unwrap();
    let node = {
        let bytes = fixture.bytes();
        match fixture {
            NodeFixture::Internal => layout
                .root()
                .unwrap()
                .parse(&bytes)
                .unwrap()
                .child(0)
                .unwrap()
                .unwrap(),
            NodeFixture::Leaf => layout.root().unwrap(),
        }
    };
    let foreign = btree_v2::leaf(BTREE_V2_FILTERED_CHUNK, &[vec![7; 11]]);
    let Err(error) = node.parse(&foreign) else {
        panic!("expected the retained owner type");
    };
    assert_eq!(
        error,
        FormatError::InvalidBTreeNodeType(BTREE_V2_FILTERED_CHUNK)
    );
    let bytes = btree_v2::leaf(BTREE_V2_LINK_NAME, &[vec![7; 11]]);
    let view = node.parse(&bytes).unwrap();
    assert_eq!(view.record_count(), 1);
    assert_eq!(view.record(0).unwrap(), &[7; 11]);
    assert_eq!(
        view.record(0).unwrap().as_ptr(),
        bytes[NODE_PREFIX_LEN..].as_ptr()
    );
    assert_eq!(view.child(0).unwrap(), None);
}

#[rstest]
#[case::leaf(NodeFixture::Leaf)]
#[case::internal(NodeFixture::Internal)]
fn indexed_nodes_exclude_unused_allocation_padding(#[case] fixture: NodeFixture) {
    let layout = BTreeV2Layout::new(&fixture.header(), OffsetWidth::Eight).unwrap();
    let node = layout.root().unwrap();
    let mut bytes = fixture.bytes();
    assert_eq!(node.encoded_len().unwrap(), bytes.len());
    bytes.resize(512, 0xff);
    let view = node.parse(&bytes).unwrap();
    assert_eq!(view.record_count(), 1);
    assert_eq!(view.record(0).unwrap(), &[7; 11]);
    assert_eq!(
        view.record(0).unwrap().as_ptr(),
        bytes[NODE_PREFIX_LEN..].as_ptr()
    );
    match fixture {
        NodeFixture::Internal => {
            let child = view.child(1).unwrap().unwrap();
            assert_eq!(child.address(), StoredAddress::new(1024));
            assert_eq!(child.total_records(), 1);
        }
        NodeFixture::Leaf => assert_eq!(view.child(0).unwrap(), None),
    }
}

#[derive(Clone, Copy)]
enum NodeFixture {
    Internal,
    Leaf,
}

impl NodeFixture {
    fn header(self) -> BTreeV2Header {
        let (depth, total) = match self {
            Self::Internal => (1, 3),
            Self::Leaf => (0, 1),
        };
        BTreeV2Header::parse(
            &btree_v2::Header::new(BTREE_V2_LINK_NAME, 11, 256, 1)
                .depth(depth)
                .total_records(total)
                .build(Widths::EIGHT),
            0,
            8,
            8,
        )
        .unwrap()
    }

    fn bytes(self) -> Vec<u8> {
        let records = [vec![7; 11]];
        match self {
            Self::Internal => btree_v2::internal(
                BTREE_V2_LINK_NAME,
                &records,
                &[512, 1024].map(|address| btree_v2::Child {
                    address,
                    records: 1,
                    records_width: 1,
                    subtree: None,
                }),
                Widths::EIGHT,
            ),
            Self::Leaf => btree_v2::leaf(BTREE_V2_LINK_NAME, &records),
        }
    }
}

#[cfg(feature = "checksum")]
#[test]
fn indexed_node_reports_checksum_mismatch() {
    let header = BTreeV2Header::parse(
        &btree_v2::Header::new(BTREE_V2_LINK_NAME, 11, 256, 1).build(Widths::EIGHT),
        0,
        8,
        8,
    )
    .unwrap();
    let layout = BTreeV2Layout::new(&header, OffsetWidth::Eight).unwrap();
    let mut bytes = btree_v2::leaf(BTREE_V2_LINK_NAME, &[vec![0; 11]]);
    let checksum_at = bytes.len() - 4;
    let expected = u32::from_le_bytes(bytes[checksum_at..].try_into().unwrap());
    bytes[6] ^= 1;
    let computed = crate::checksum::jenkins_lookup3(&bytes[..checksum_at]);
    let Err(error) = layout.root().unwrap().parse(&bytes) else {
        panic!("expected checksum mismatch");
    };
    assert_eq!(error, FormatError::ChecksumMismatch { expected, computed });
}

#[test]
fn nodes_with_4_byte_addresses_parse_to_their_fields() {
    let widths = Widths::new(4, 8);
    let header = btree_v2::Header::new(5, 11, 0x200, 1)
        .depth(1)
        .total_records(3)
        .build(widths);
    let leaf = btree_v2::leaf(5, &[vec![1; 11], vec![2; 11]]);
    let child = |address| btree_v2::Child {
        address,
        records: 1,
        records_width: 1,
        subtree: None,
    };
    let internal = btree_v2::internal(5, &[vec![3; 11]], &[child(0x300), child(0x400)], widths);
    let expected = BTreeV2Header {
        tree_type: 5,
        node_size: 512,
        record_size: 11,
        depth: 1,
        root_node_address: StoredAddress::new(0x200),
        num_records_in_root: 1,
        total_records: 3,
    };

    assert_eq!(BTreeV2Header::parse(&header, 0, 4, 8), Ok(expected.clone()));
    assert_eq!(
        BTreeV2Header::parse_from_source(header.as_slice(), 0, 4, 8),
        Ok(expected.clone())
    );
    let layout = BTreeV2Layout::new(&expected, OffsetWidth::Four).unwrap();
    let root = layout.root().unwrap().parse(&internal).unwrap();
    assert_eq!(root.record_count(), 1);
    assert_eq!(root.record(0).unwrap(), &[3; 11]);
    let first = root.child(0).unwrap().unwrap();
    assert_eq!(first.address(), StoredAddress::new(0x300));
    assert_eq!(first.total_records(), 1);
    let second = root.child(1).unwrap().unwrap();
    assert_eq!(second.address(), StoredAddress::new(0x400));
    assert_eq!(second.total_records(), 1);
    let header = BTreeV2Header {
        depth: 0,
        num_records_in_root: 2,
        total_records: 2,
        ..expected
    };
    let layout = BTreeV2Layout::new(&header, OffsetWidth::Four).unwrap();
    let view = layout.root().unwrap().parse(&leaf).unwrap();
    assert_eq!(view.record_count(), 2);
    assert_eq!(view.record(0).unwrap(), &[1; 11]);
    assert_eq!(view.record(1).unwrap(), &[2; 11]);
}
