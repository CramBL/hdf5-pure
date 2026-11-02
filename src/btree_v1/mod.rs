//! The walk of a group's version 1 B-tree to its symbol table nodes.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use hdf5_pure_format::__private::BTreeV1Node;

#[cfg(test)]
use crate::address::AddressContext;
use crate::address::StoredAddress;
#[cfg(test)]
use crate::address::StoredAddressExt;
#[cfg(test)]
use crate::address::{BaseAddress, BaseAddressExt};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source::PhysicalBytes;
use crate::source::StoredMetadata;
use crate::source::StoredRead;
use crate::width::FormatWidths;

use self::visit::{AdmittedV1Node, V1InternalNode, V1NodeLevel, V1TreeVisitState};

pub(crate) mod visit;

/// Returns the stored symbol table node addresses in B-tree traversal order.
///
/// Resolves node addresses using the context captured in `physical` and parses nodes in the
/// complete original image. Reports [`FormatError::UnexpectedEof`] with physical positions and
/// the complete image's byte length.
///
/// # Errors
///
/// Returns [`FormatError::InvalidBTreeNodeType`] if a node's type is not 0,
/// [`FormatError::InvalidBTreeSignature`] for a repeated or undefined node address, inconsistent
/// entry counts, or an incorrect child level, and [`FormatError::ResourceLimit`] if the traversal
/// exceeds its work limit.
/// Returns [`FormatError::AddressTooLarge`] if a node address exceeds the offset width.
/// Returns errors from address resolution and [`BTreeV1Node::parse_with_widths`].
pub fn collect_symbol_table_nodes(
    physical: PhysicalBytes<'_>,
    btree_address: StoredAddress,
    widths: FormatWidths,
) -> Result<Vec<StoredAddress>, FormatError> {
    collect_group_nodes(
        |address| {
            BTreeV1Node::parse_with_widths(
                physical.bytes(),
                physical.addresses().resolve(address)?.get().to_usize()?,
                widths,
            )
        },
        widths,
        btree_address,
    )
}

/// Returns the stored symbol table node addresses in B-tree traversal order from `source`.
///
/// Reads each node's complete header before checking the signature, then its entries in use and
/// trailing key, through [`StoredRead::read_metadata`]. Uses the source's captured effective base
/// and preserves its metadata ownership, physical ranges, and cache behavior.
///
/// # Errors
///
/// Returns [`FormatError::InvalidBTreeNodeType`] if a node's type is not 0,
/// [`FormatError::InvalidBTreeSignature`] for a repeated or undefined node address, inconsistent
/// entry counts, or an incorrect child level, and [`FormatError::ResourceLimit`] if the traversal
/// exceeds its work limit.
/// Returns [`FormatError::AddressTooLarge`] if a node address exceeds the offset width.
/// Returns errors from address resolution, node parsing, and source metadata reads.
pub fn collect_symbol_table_nodes_from_source(
    source: &(impl StoredRead + ?Sized),
    btree_address: StoredAddress,
    widths: FormatWidths,
) -> Result<Vec<StoredAddress>, FormatError> {
    collect_group_nodes(
        |address| {
            BTreeV1Node::parse_from_source_with_widths(
                &StoredMetadata(source),
                address.get(),
                widths,
            )
        },
        widths,
        btree_address,
    )
}

fn collect_group_nodes(
    mut read: impl FnMut(StoredAddress) -> Result<BTreeV1Node, FormatError>,
    widths: FormatWidths,
    root: StoredAddress,
) -> Result<Vec<StoredAddress>, FormatError> {
    let mut visits = V1TreeVisitState::new(root, widths.offsets);
    let mut result = Vec::new();
    while let Some(admitted) = visits.next()? {
        let node = GroupNode::parse(read(admitted.address())?, admitted)?;
        match node {
            GroupNode::Internal { level, children } => {
                visits.push_children(level, children.into_iter().map(Ok))?;
            }
            GroupNode::Leaf(children) => result.extend(children),
        }
    }
    Ok(result)
}

/// A group node with checked entry counts and a checked level.
enum GroupNode {
    Internal {
        level: V1InternalNode,
        children: Vec<StoredAddress>,
    },
    Leaf(Vec<StoredAddress>),
}

impl GroupNode {
    fn parse(node: BTreeV1Node, admitted: AdmittedV1Node) -> Result<Self, FormatError> {
        if node.node_type != GROUP_NODE_TYPE {
            return Err(FormatError::InvalidBTreeNodeType(node.node_type));
        }
        if node.children.len() != usize::from(node.entries_used)
            || node.keys.len() != usize::from(node.entries_used) + 1
        {
            return Err(FormatError::InvalidBTreeSignature);
        }
        Ok(match admitted.parse_level(node.node_level)? {
            V1NodeLevel::Internal(level) => Self::Internal {
                level,
                children: node.children,
            },
            V1NodeLevel::Leaf => Self::Leaf(node.children),
        })
    }
}

/// The node type for group B-trees, defined in "Version 1 B-trees" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_infra_btrees_v1
const GROUP_NODE_TYPE: u8 = 0;

#[cfg(test)]
mod tests {
    use hdf5_pure_format::__private as format;
    use rstest::rstest;
    use test_util::btree_v1;
    use test_util::widths::Widths;

    use super::*;
    use crate::placed_source::PlacedSource;
    use crate::source::BytesSource;
    use crate::source::PhysicalSource;

    #[rstest]
    #[case::zero_base(0)]
    #[case::userblock(512)]
    fn collect_symbol_table_nodes_in_tree_order(
        #[case] base: u64,
        #[values((2, 4), (4, 8), (8, 2))] sizes: (u8, u8),
    ) {
        // One internal node points to two leaf nodes.
        let (os, ls) = sizes;
        let widths = Widths::new(usize::from(os), usize::from(ls));
        let leaf1_offset: usize = 0;
        let leaf2_offset: usize = 256;
        let internal_offset: usize = 512;

        let leaf1 =
            btree_v1::node_with_group_keys(btree_v1::NodeType::GROUP, 0, &[0, 5], &[0xA00], widths);
        let leaf2 = btree_v1::node_with_group_keys(
            btree_v1::NodeType::GROUP,
            0,
            &[5, 10],
            &[0xB00],
            widths,
        );
        let internal = btree_v1::node_with_group_keys(
            btree_v1::NodeType::GROUP,
            1,
            &[0, 5, 10],
            &[leaf1_offset as u64, leaf2_offset as u64],
            widths,
        );

        let mut file = vec![0u8; 1024];
        file[leaf1_offset..leaf1_offset + leaf1.len()].copy_from_slice(&leaf1);
        file[leaf2_offset..leaf2_offset + leaf2.len()].copy_from_slice(&leaf2);
        file[internal_offset..internal_offset + internal.len()].copy_from_slice(&internal);

        let source = PlacedSource::new(base, file.clone());
        let expected = vec![StoredAddress::new(0xA00), StoredAddress::new(0xB00)];
        assert_eq!(
            collect_symbol_table_nodes_from_source(
                &source.framed(),
                StoredAddress::new(internal_offset as u64),
                FormatWidths::from_sizes(os, ls).unwrap(),
            )
            .unwrap(),
            expected,
        );
        let header_size = format::btree_v1_node_header_size(os);
        assert_eq!(
            source.reads.into_inner(),
            vec![
                (base + 512, header_size),
                (base + 512, internal.len()),
                (base, header_size),
                (base, leaf1.len()),
                (base + 256, header_size),
                (base + 256, leaf2.len())
            ],
        );
        file.splice(..0, vec![0; base.to_usize().unwrap()]);
        let snods = collect_symbol_table_nodes(
            PhysicalBytes::for_test(
                &file,
                AddressContext::from_effective_base(BaseAddress::new(base)),
            ),
            StoredAddress::new(internal_offset as u64),
            FormatWidths::from_sizes(os, ls).unwrap(),
        )
        .unwrap();
        assert_eq!(
            snods,
            vec![StoredAddress::new(0xA00), StoredAddress::new(0xB00)]
        );
    }

    #[rstest]
    #[case::zero_base(0)]
    #[case::userblock(512)]
    fn collect_rejects_wrong_node_type(#[case] base: u64) {
        let data =
            btree_v1::node_with_group_keys(btree_v1::NodeType::CHUNK, 0, &[0, 1], &[0x100], WIDTHS);
        let mut file = vec![0u8; 512];
        file[..data.len()].copy_from_slice(&data);
        assert_eq!(
            collect_symbol_table_nodes_from_source(
                &PlacedSource::new(base, file.clone()).framed(),
                StoredAddress::new(0),
                FormatWidths::from_sizes(8, 8).unwrap(),
            )
            .unwrap_err(),
            FormatError::InvalidBTreeNodeType(1),
        );
        file.splice(..0, vec![0; base.to_usize().unwrap()]);
        let err = collect_symbol_table_nodes(
            PhysicalBytes::for_test(
                &file,
                AddressContext::from_effective_base(BaseAddress::new(base)),
            ),
            StoredAddress::new(0),
            FormatWidths::from_sizes(8, 8).unwrap(),
        )
        .unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeNodeType(1));
    }

    #[rstest]
    #[case::zero_base(0)]
    #[case::userblock(512)]
    fn collect_symbol_table_nodes_rejects_cyclic_btree(#[case] base: u64) {
        // An internal node (level 1) whose only child points back at itself.
        let os: u8 = 8;
        let ls: u8 = 8;
        let node =
            btree_v1::node_with_group_keys(btree_v1::NodeType::GROUP, 1, &[0, 0], &[0], WIDTHS);
        let mut file = vec![0u8; 1024];
        file[..node.len()].copy_from_slice(&node);

        let source = PlacedSource::new(base, file.clone());
        assert_eq!(
            collect_symbol_table_nodes_from_source(
                &source.framed(),
                StoredAddress::new(0),
                FormatWidths::from_sizes(os, ls).unwrap(),
            )
            .unwrap_err(),
            FormatError::InvalidBTreeSignature,
        );
        assert_eq!(
            source.reads.into_inner(),
            vec![(base, 24), (base, node.len())]
        );
        file.splice(..0, vec![0; base.to_usize().unwrap()]);
        let err = collect_symbol_table_nodes(
            PhysicalBytes::for_test(
                &file,
                AddressContext::from_effective_base(BaseAddress::new(base)),
            ),
            StoredAddress::new(0),
            FormatWidths::from_sizes(os, ls).unwrap(),
        )
        .unwrap_err();
        assert_eq!(err, FormatError::InvalidBTreeSignature);
    }

    #[rstest]
    #[case::repeated_siblings(GroupTreeError::RepeatedSiblings, vec![0, 256])]
    #[case::diamond(GroupTreeError::Diamond, vec![0, 256, 768, 512])]
    #[case::ancestor_cycle(GroupTreeError::AncestorCycle, vec![0, 256])]
    #[case::wrong_level(GroupTreeError::WrongLevel, vec![0, 256])]
    fn group_walk_rejects_malformed_tree_before_repeated_reads(
        #[case] corruption: GroupTreeError,
        #[case] expected_reads: Vec<u64>,
        #[values(0, 512)] base: u64,
    ) {
        let (root_level, root_children, left_level, left_children, right_level, right_children) =
            match corruption {
                GroupTreeError::AncestorCycle => (2, vec![256], 1, vec![0], 0, vec![]),
                GroupTreeError::Diamond => (2, vec![256, 512], 1, vec![768], 1, vec![768]),
                GroupTreeError::RepeatedSiblings => (1, vec![256, 256], 0, vec![0xA00], 0, vec![]),
                GroupTreeError::WrongLevel => (2, vec![256], 0, vec![0xA00], 0, vec![]),
            };
        let mut image = vec![0; 1024];
        for (address, level, children) in [
            (0, root_level, root_children),
            (256, left_level, left_children),
            (512, right_level, right_children),
            (768, 0, vec![0xB00]),
        ] {
            let keys = vec![0; children.len() + 1];
            let node = btree_v1::node_with_group_keys(
                btree_v1::NodeType::GROUP,
                level,
                &keys,
                &children,
                WIDTHS,
            );
            image[address..address + node.len()].copy_from_slice(&node);
        }
        let source = PlacedSource::new(base, image.clone());
        assert_eq!(
            collect_symbol_table_nodes_from_source(
                &source.framed(),
                StoredAddress::ZERO,
                FormatWidths::from_sizes(8, 8).unwrap(),
            )
            .unwrap_err(),
            FormatError::InvalidBTreeSignature,
        );
        let reads = source.reads.into_inner();
        assert_eq!(reads.len(), expected_reads.len() * 2);
        assert_eq!(
            reads
                .iter()
                .step_by(2)
                .map(|&(address, len)| {
                    assert_eq!(len, 24);
                    address - base
                })
                .collect::<Vec<_>>(),
            expected_reads,
        );
        image.splice(..0, vec![0; base.to_usize().unwrap()]);
        assert_eq!(
            collect_symbol_table_nodes(
                PhysicalBytes::for_test(
                    &image,
                    AddressContext::from_effective_base(BaseAddress::new(base))
                ),
                StoredAddress::ZERO,
                FormatWidths::from_sizes(8, 8).unwrap(),
            )
            .unwrap_err(),
            FormatError::InvalidBTreeSignature,
        );
    }

    enum GroupTreeError {
        AncestorCycle,
        Diamond,
        RepeatedSiblings,
        WrongLevel,
    }

    #[rstest]
    #[case::short_header(7, 24, btree_v1::NodeType::GROUP)]
    #[case::short_body(47, 48, btree_v1::NodeType::GROUP)]
    #[case::short_body_before_node_type(47, 48, btree_v1::NodeType::CHUNK)]
    fn collect_preserves_physical_eof_coordinates(
        #[case] available: usize,
        #[case] required: usize,
        #[case] node_type: btree_v1::NodeType,
        #[values(0, 512)] base: usize,
    ) {
        let mut image = vec![0; base];
        let node = btree_v1::node_with_group_keys(node_type, 0, &[0, 1], &[0x100], WIDTHS);
        assert_eq!(node.len(), 48);
        image.extend_from_slice(&node[..available]);
        let addresses = AddressContext::from_effective_base(BaseAddress::new(base.to_u64()));
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        assert_eq!(
            collect_symbol_table_nodes(
                PhysicalBytes::for_test(&image, addresses),
                StoredAddress::ZERO,
                widths
            )
            .unwrap_err(),
            FormatError::UnexpectedEof {
                expected: base + required,
                available: image.len()
            },
        );
        let source = BytesSource::new(&image);
        let stored = PhysicalSource::for_test(&source, addresses).stored();
        assert_eq!(
            collect_symbol_table_nodes_from_source(&stored, StoredAddress::ZERO, widths)
                .unwrap_err(),
            FormatError::UnexpectedSourceEof {
                expected: (base + required).to_u64(),
                available: image.len().to_u64(),
            },
        );
    }

    #[test]
    fn collect_preserves_address_overflow() {
        let addresses = AddressContext::from_effective_base(BaseAddress::new(512));
        let widths = FormatWidths::from_sizes(8, 8).unwrap();
        let source = BytesSource::new(&[]);
        let stored = PhysicalSource::for_test(&source, addresses).stored();
        let expected = FormatError::OffsetOverflow {
            offset: u64::MAX - 1,
            length: 512,
        };
        assert_eq!(
            collect_symbol_table_nodes(
                PhysicalBytes::for_test(&[], addresses),
                StoredAddress::new(u64::MAX - 1),
                widths
            )
            .unwrap_err(),
            expected,
        );
        assert_eq!(
            collect_symbol_table_nodes_from_source(
                &stored,
                StoredAddress::new(u64::MAX - 1),
                widths
            )
            .unwrap_err(),
            expected,
        );
    }

    const WIDTHS: Widths = Widths::EIGHT;
}
