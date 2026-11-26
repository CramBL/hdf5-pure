use rstest::rstest;
use test_util::btree::v1 as btree_v1;
use test_util::widths::Widths;

use crate::address::{StoredAddress, StoredAddressExt};
use crate::btree::v1::BTreeV1Node;
use crate::btree::v1::header;
use crate::error::FormatError;
use crate::width::FormatWidths;

#[test]
fn parse_leaf_node() {
    let data = btree_v1::node_with_group_keys(
        btree_v1::NodeType::GROUP,
        0,
        &[0, 5, 10],
        &[0x100, 0x200],
        WIDTHS,
    );
    let node = BTreeV1Node::parse(&data, 0, 8, 8).unwrap();
    assert_eq!(node.node_type, 0);
    assert_eq!(node.node_level, 0);
    assert_eq!(node.entries_used, 2);
    assert_eq!(node.keys, vec![0, 5, 10]);
    assert_eq!(
        node.children,
        vec![StoredAddress::new(0x100), StoredAddress::new(0x200)]
    );
    assert_eq!(node.left_sibling, None);
    assert_eq!(node.right_sibling, None);
}

#[test]
fn parse_with_siblings_none() {
    let data =
        btree_v1::node_with_group_keys(btree_v1::NodeType::GROUP, 0, &[0, 8], &[0x300], WIDTHS);
    let node = BTreeV1Node::parse(&data, 0, 8, 8).unwrap();
    assert_eq!(node.left_sibling, None);
    assert_eq!(node.right_sibling, None);
}

#[test]
fn parse_leaf_node_differing_offset_and_length_sizes() {
    let data = btree_v1::node_with_group_keys(
        btree_v1::NodeType::GROUP,
        0,
        &[0x10, 0x20],
        &[0x300],
        Widths::new(4, 8),
    );
    let node = BTreeV1Node::parse(&data, 0, 4, 8).unwrap();
    assert_eq!(node.entries_used, 1);
    assert_eq!(node.keys, vec![0x10, 0x20]);
    assert_eq!(node.children, vec![StoredAddress::new(0x300)]);
}

#[test]
fn invalid_signature() {
    let mut data =
        btree_v1::node_with_group_keys(btree_v1::NodeType::GROUP, 0, &[0, 1], &[0x100], WIDTHS);
    data[0] = b'X';
    let err = BTreeV1Node::parse(&data, 0, 8, 8).unwrap_err();
    assert_eq!(err, FormatError::InvalidBTreeSignature);
}

#[test]
fn parse_4byte_offsets() {
    let data = btree_v1::node_with_group_keys(
        btree_v1::NodeType::GROUP,
        0,
        &[0, 4],
        &[0x50],
        Widths::new(4, 4),
    );
    let node = BTreeV1Node::parse(&data, 0, 4, 4).unwrap();
    assert_eq!(node.entries_used, 1);
    assert_eq!(node.children, vec![StoredAddress::new(0x50)]);
}

const WIDTHS: Widths = Widths::EIGHT;

#[test]
fn a_group_node_parses_from_bytes_and_from_a_source() {
    let widths = Widths::new(4, 8);
    let keys = [
        btree_v1::group_key(0, widths),
        btree_v1::group_key(6, widths),
    ];
    let node = btree_v1::node(btree_v1::NodeType::GROUP, 0, &keys, &[0x300], widths);
    let expected = BTreeV1Node {
        node_type: 0,
        node_level: 0,
        entries_used: 1,
        left_sibling: None,
        right_sibling: None,
        keys: vec![0, 6],
        children: vec![StoredAddress::new(0x300)],
    };

    assert_eq!(header::btree_v1_node_header_size(4), 16);
    assert_eq!(BTreeV1Node::parse(&node, 0, 4, 8), Ok(expected.clone()));
    assert_eq!(
        BTreeV1Node::parse_from_source(node.as_slice(), 0, 4, 8),
        Ok(expected)
    );
}

#[rstest]
#[case::truncated_invalid_signature(
    false,
    GroupNodeExtent::ShortHeader,
    GroupNodeOutcome::TruncatedHeader
)]
#[case::contained_invalid_signature(
    false,
    GroupNodeExtent::Header,
    GroupNodeOutcome::InvalidSignature
)]
#[case::complete_node(true, GroupNodeExtent::Complete, GroupNodeOutcome::Parsed)]
#[case::truncated_valid_header(
    true,
    GroupNodeExtent::ShortHeader,
    GroupNodeOutcome::TruncatedHeader
)]
#[case::truncated_valid_body(true, GroupNodeExtent::ShortBody, GroupNodeOutcome::TruncatedBody)]
fn group_nodes_preserve_header_and_signature_error_precedence(
    #[case] signature: bool,
    #[case] extent: GroupNodeExtent,
    #[case] outcome: GroupNodeOutcome,
    #[values(Widths::new(2, 2), Widths::new(4, 8), Widths::EIGHT)] widths: Widths,
    #[values(0, 7)] offset: usize,
) {
    let mut node =
        btree_v1::node_with_group_keys(btree_v1::NodeType::GROUP, 0, &[0, 6], &[0x300], widths);
    if !signature {
        node[..4].copy_from_slice(&[0xf0, 0x3f, 0, 0]);
    }
    let header_size = header::btree_v1_node_header_size(widths.offset.try_into().unwrap());
    let available = match extent {
        GroupNodeExtent::Complete => node.len(),
        GroupNodeExtent::Header => header_size,
        GroupNodeExtent::ShortBody => node.len() - 1,
        GroupNodeExtent::ShortHeader => header_size - widths.offset.min(6),
    };
    let mut bytes = vec![0; offset];
    bytes.extend_from_slice(&node[..available]);
    let formats = FormatWidths::from_sizes(
        widths.offset.try_into().unwrap(),
        widths.length.try_into().unwrap(),
    )
    .unwrap();
    let buffered = BTreeV1Node::parse_with_widths(&bytes, offset, formats);
    let streamed =
        BTreeV1Node::parse_from_source_with_widths(bytes.as_slice(), offset as u64, formats);
    match outcome {
        GroupNodeOutcome::InvalidSignature => {
            assert_eq!(buffered, Err(FormatError::InvalidBTreeSignature));
            assert_eq!(streamed, Err(FormatError::InvalidBTreeSignature));
        }
        GroupNodeOutcome::Parsed => {
            let expected = BTreeV1Node {
                node_type: btree_v1::NodeType::GROUP.0,
                node_level: 0,
                entries_used: 1,
                left_sibling: None,
                right_sibling: None,
                keys: vec![0, 6],
                children: vec![StoredAddress::new(0x300)],
            };
            assert_eq!(buffered.unwrap(), expected);
            assert_eq!(streamed.unwrap(), expected);
        }
        GroupNodeOutcome::TruncatedBody | GroupNodeOutcome::TruncatedHeader => {
            let required = offset
                + if matches!(outcome, GroupNodeOutcome::TruncatedHeader) {
                    header_size
                } else {
                    node.len()
                };
            assert_eq!(
                buffered,
                Err(FormatError::UnexpectedEof {
                    expected: required,
                    available: bytes.len()
                })
            );
            assert_eq!(
                streamed,
                Err(FormatError::UnexpectedSourceEof {
                    expected: required as u64,
                    available: bytes.len() as u64
                })
            );
        }
    }
}

enum GroupNodeExtent {
    Complete,
    Header,
    ShortBody,
    ShortHeader,
}
enum GroupNodeOutcome {
    InvalidSignature,
    Parsed,
    TruncatedBody,
    TruncatedHeader,
}
