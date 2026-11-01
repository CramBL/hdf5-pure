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

/// The recursion-depth limit for cyclic or deeply nested group B-trees.
const MAX_SYMBOL_TABLE_BTREE_DEPTH: u32 = 64;

/// Returns the stored symbol table node addresses in B-tree traversal order.
///
/// Resolves node addresses using the context captured in `physical` and parses nodes in the
/// complete original image. Reports [`FormatError::UnexpectedEof`] with physical positions and
/// the complete image's byte length.
///
/// # Errors
///
/// Returns [`FormatError::InvalidBTreeNodeType`] if a node's type is not 0, and
/// [`FormatError::NestingDepthExceeded`] if traversal exceeds [`MAX_SYMBOL_TABLE_BTREE_DEPTH`].
/// Returns errors from address resolution and [`BTreeV1Node::parse_with_widths`].
pub fn collect_symbol_table_nodes(
    physical: PhysicalBytes<'_>,
    btree_address: StoredAddress,
    widths: FormatWidths,
) -> Result<Vec<StoredAddress>, FormatError> {
    collect_symbol_table_nodes_inner(physical, btree_address, widths, 0)
}

/// Collects stored symbol table node addresses while checking the recursion depth.
///
/// # Errors
///
/// Returns the errors of [`collect_symbol_table_nodes`].
fn collect_symbol_table_nodes_inner(
    physical: PhysicalBytes<'_>,
    btree_address: StoredAddress,
    widths: FormatWidths,
    depth: u32,
) -> Result<Vec<StoredAddress>, FormatError> {
    if depth > MAX_SYMBOL_TABLE_BTREE_DEPTH {
        return Err(FormatError::NestingDepthExceeded);
    }
    let node_offset = physical
        .addresses()
        .resolve(btree_address)?
        .get()
        .to_usize()?;
    let node = BTreeV1Node::parse_with_widths(physical.bytes(), node_offset, widths)?;

    if node.node_type != 0 {
        return Err(FormatError::InvalidBTreeNodeType(node.node_type));
    }

    if node.node_level == 0 {
        // A leaf's children are the symbol table nodes themselves.
        Ok(node.children)
    } else {
        // An internal node's children are B-tree nodes one level down.
        let mut result = Vec::new();
        for &child_addr in &node.children {
            let child_snods =
                collect_symbol_table_nodes_inner(physical, child_addr, widths, depth + 1)?;
            result.extend(child_snods);
        }
        Ok(result)
    }
}

/// Returns the stored symbol table node addresses in B-tree traversal order from `source`.
///
/// Reads each node's eight-byte prefix, then its header, entries in use, and trailing key,
/// through [`StoredRead::read_metadata`]. Uses the source's captured effective base and preserves its
/// metadata ownership, physical ranges, and cache behavior.
///
/// # Errors
///
/// Returns [`FormatError::InvalidBTreeNodeType`] if a node's type is not 0, and
/// [`FormatError::NestingDepthExceeded`] if traversal exceeds [`MAX_SYMBOL_TABLE_BTREE_DEPTH`].
/// Returns errors from address resolution, node parsing, and source metadata reads.
pub fn collect_symbol_table_nodes_from_source(
    source: &(impl StoredRead + ?Sized),
    btree_address: StoredAddress,
    widths: FormatWidths,
) -> Result<Vec<StoredAddress>, FormatError> {
    collect_symbol_table_nodes_from_source_inner(source, btree_address, widths, 0)
}

/// Reads B-tree nodes through stored metadata reads while checking the recursion depth.
///
/// # Errors
///
/// Returns the errors of [`collect_symbol_table_nodes_from_source`].
fn collect_symbol_table_nodes_from_source_inner(
    source: &(impl StoredRead + ?Sized),
    btree_address: StoredAddress,
    widths: FormatWidths,
    depth: u32,
) -> Result<Vec<StoredAddress>, FormatError> {
    if depth > MAX_SYMBOL_TABLE_BTREE_DEPTH {
        return Err(FormatError::NestingDepthExceeded);
    }
    let node = BTreeV1Node::parse_from_source_with_widths(
        &StoredMetadata(source),
        btree_address.get(),
        widths,
    )?;

    if node.node_type != 0 {
        return Err(FormatError::InvalidBTreeNodeType(node.node_type));
    }

    if node.node_level == 0 {
        Ok(node.children)
    } else {
        let mut result = Vec::new();
        for &child_addr in &node.children {
            let child_snods = collect_symbol_table_nodes_from_source_inner(
                source,
                child_addr,
                widths,
                depth + 1,
            )?;
            result.extend(child_snods);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
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
    fn collect_symbol_table_nodes_in_tree_order(#[case] base: u64) {
        // One internal node points to two leaf nodes.
        let os: u8 = 8;
        let ls: u8 = 8;
        let leaf1_offset: usize = 0;
        let leaf2_offset: usize = 256;
        let internal_offset: usize = 512;

        let leaf1 =
            btree_v1::node_with_group_keys(btree_v1::NodeType::GROUP, 0, &[0, 5], &[0xA00], WIDTHS);
        let leaf2 = btree_v1::node_with_group_keys(
            btree_v1::NodeType::GROUP,
            0,
            &[5, 10],
            &[0xB00],
            WIDTHS,
        );
        let internal = btree_v1::node_with_group_keys(
            btree_v1::NodeType::GROUP,
            1,
            &[0, 5, 10],
            &[leaf1_offset as u64, leaf2_offset as u64],
            WIDTHS,
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
        assert_eq!(
            source.reads.into_inner(),
            vec![
                (base + 512, 8),
                (base + 512, internal.len()),
                (base, 8),
                (base, leaf1.len()),
                (base + 256, 8),
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

        assert_eq!(
            collect_symbol_table_nodes_from_source(
                &PlacedSource::new(base, file.clone()).framed(),
                StoredAddress::new(0),
                FormatWidths::from_sizes(os, ls).unwrap(),
            )
            .unwrap_err(),
            FormatError::NestingDepthExceeded,
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
        assert_eq!(err, FormatError::NestingDepthExceeded);
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
                expected: (base + if available < 8 { 8 } else { required }).to_u64(),
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
            offset: u64::MAX,
            length: 512,
        };
        assert_eq!(
            collect_symbol_table_nodes(
                PhysicalBytes::for_test(&[], addresses),
                StoredAddress::new(u64::MAX),
                widths
            )
            .unwrap_err(),
            expected,
        );
        assert_eq!(
            collect_symbol_table_nodes_from_source(&stored, StoredAddress::new(u64::MAX), widths)
                .unwrap_err(),
            expected,
        );
    }

    const WIDTHS: Widths = Widths::EIGHT;
}
