//! Decodes link names and object-header addresses from controlled symbol-table group fixtures.

use std::collections::BTreeSet;
use std::num::NonZeroU16;

use hdf5_pure::BaseAddress;
use hdf5_pure::File;
use hdf5_pure::MessageType;
use hdf5_pure::StoredAddress;
use hdf5_pure_core::__private::{BaseAddressExt, FileRange, FileRegion, PhysicalAddress};
use hdf5_pure_format::__private::{
    AccessMode, BTreeV1Node, FormatWidths, LocalHeapDataSegment, LocalHeapLocator, ObjectHeader,
    SymbolTableMessage, SymbolTableNode, SymbolTableNodeLocator,
};
use test_util::btree_v1;

/// A complete file image and format parameters for symbol-table group fixtures.
pub struct Fixture<'a> {
    region: FileRegion<'a>,
    base: BaseAddress,
    widths: FormatWidths,
    leaf_node_k: Option<NonZeroU16>,
}

impl<'a> Fixture<'a> {
    /// Uses `file`'s format widths, file base, and symbol-table leaf-node K to interpret `bytes`.
    ///
    /// `bytes` contains the complete physical file, including any userblock.
    ///
    /// # Panics
    ///
    /// Panics if the image length differs from [`File::file_size`] or the leaf-node K is zero.
    pub fn new(file: &File, bytes: &'a [u8]) -> Self {
        let superblock = file.superblock();
        assert_eq!(u64::try_from(bytes.len()).unwrap(), file.file_size());
        Self {
            region: FileRegion::new(
                FileRange::new(PhysicalAddress::new(0), u64::try_from(bytes.len()).unwrap())
                    .unwrap(),
                bytes,
            )
            .unwrap(),
            base: superblock.base_address,
            widths: FormatWidths {
                offsets: superblock.offset_size,
                lengths: superblock.length_size,
            },
            leaf_node_k: superblock
                .group_leaf_node_k
                .map(|k| NonZeroU16::new(k).unwrap()),
        }
    }

    /// Returns the group's link names and encoded object-header addresses in link-name order.
    ///
    /// Decodes the group at `address` through its Symbol Table message, version 1 B-tree,
    /// symbol-table nodes, and local heap. The returned [`StoredAddress`] values remain relative
    /// to the file base.
    ///
    /// # Panics
    ///
    /// Panics if metadata cannot be parsed or fails the fixture's structural assertions, including
    /// invalid addresses, repeated nodes, or invalid or duplicate link names. Soft links and other
    /// unsupported entry cache types cause a panic.
    pub fn links(&self, address: StoredAddress) -> Vec<(String, StoredAddress)> {
        let table = self.table(address);
        let names = self.names(self.address(table.local_heap_address));
        let mut pending = vec![(table.btree_address, None)];
        let mut seen = BTreeSet::new();
        let mut symbol_nodes = BTreeSet::new();
        let mut links = Vec::new();
        while let Some((address, level)) = pending.pop() {
            let address = self.address(address);
            assert!(
                seen.insert(address.encoded),
                "repeated B-tree node at {:?}",
                address.encoded
            );
            let node = BTreeV1Node::parse_with_widths(
                self.region.bytes(),
                usize::try_from(address.physical.get()).unwrap(),
                self.widths,
            )
            .unwrap();
            assert_eq!(node.node_type, btree_v1::NodeType::GROUP.0);
            if let Some(level) = level {
                assert_eq!(node.node_level, level);
            }
            for child in node.children {
                let child = self.address(child);
                if node.node_level > 0 {
                    pending.push((child.encoded, Some(node.node_level.checked_sub(1).unwrap())));
                    continue;
                }
                assert!(
                    symbol_nodes.insert(child.encoded),
                    "repeated symbol table node at {:?}",
                    child.encoded
                );
                let symbols = self.symbols(child);
                for entry in symbols.entries {
                    assert!(
                        entry.cache_type == CACHE_NONE || entry.cache_type == CACHE_GROUP,
                        "unsupported fixture link cache type {}",
                        entry.cache_type
                    );
                    let address = self.address(entry.object_header_address);
                    let name = names.name(entry.link_name_offset).unwrap();
                    assert!(
                        !name.is_empty() && !name.contains('/'),
                        "invalid fixture link name {name:?}"
                    );
                    links.push((name.to_owned(), address.encoded));
                }
            }
        }
        links.sort();
        assert!(
            links.windows(2).all(|pair| pair[0].0 != pair[1].0),
            "duplicate fixture link name"
        );
        links
    }

    fn table(&self, address: StoredAddress) -> SymbolTableMessage {
        let address = self.address(address);
        let header = ObjectHeader::parse_with_base_and_widths(
            self.region.bytes(),
            AccessMode::ReadOnly,
            usize::try_from(address.physical.get()).unwrap(),
            self.widths,
            self.base,
        )
        .unwrap();
        let mut messages = header
            .messages
            .iter()
            .filter(|message| message.msg_type == MessageType::SYMBOL_TABLE);
        let message = messages
            .next()
            .expect("fixture group has a Symbol Table message");
        assert!(
            messages.next().is_none(),
            "fixture group has one Symbol Table message"
        );
        SymbolTableMessage::parse_with_offset_width(&message.data, self.widths.offsets).unwrap()
    }

    fn names(&self, heap_address: Address) -> LocalHeapDataSegment<'a> {
        let locator = LocalHeapLocator::new(self.widths, self.base, heap_address.encoded);
        let heap = locator
            .parse(
                self.region
                    .subregion(locator.parse_range().unwrap())
                    .unwrap(),
            )
            .unwrap();
        heap.data_segment(self.region.subregion(heap.data_segment_range()).unwrap())
            .unwrap()
    }

    fn symbols(&self, address: Address) -> SymbolTableNode {
        let locator =
            SymbolTableNodeLocator::new(self.widths, self.base, address.encoded, self.leaf_node_k);
        let entries = locator
            .parse(
                self.region
                    .subregion(locator.parse_range().unwrap())
                    .unwrap(),
            )
            .unwrap();
        entries
            .parse(self.region.subregion(entries.parse_range()).unwrap())
            .unwrap()
    }

    fn address(&self, encoded: StoredAddress) -> Address {
        assert!(
            encoded.get() <= StoredAddress::undefined(self.widths.offsets).get(),
            "fixture address exceeds its encoded width"
        );
        assert!(
            !encoded.is_undefined(self.widths.offsets),
            "undefined fixture address"
        );
        let physical = PhysicalAddress::new(self.base.absolute(encoded).unwrap());
        self.region
            .subregion(FileRange::new(physical, 1).unwrap())
            .unwrap();
        Address { encoded, physical }
    }
}

/// A defined address within the file, in encoded and physical coordinates.
struct Address {
    encoded: StoredAddress,
    physical: PhysicalAddress,
}

/// The cache type of an entry without cached object metadata.
///
/// Defined in "Symbol Table Entry" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_symboltableentry
const CACHE_NONE: u32 = 0;

/// The cache type of an entry with cached group object metadata.
///
/// Defined in "Symbol Table Entry" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_symboltableentry
const CACHE_GROUP: u32 = 1;

#[cfg(test)]
mod tests {
    use hdf5_pure::FileBuilder;
    use hdf5_pure::OffsetWidth;
    use hdf5_pure_core::__private::StoredAddressExt;

    use super::*;

    #[test]
    #[should_panic(expected = "undefined fixture address")]
    fn an_undefined_address_is_rejected_before_reading_metadata() {
        let bytes = FileBuilder::new().finish().unwrap();
        let file = File::from_bytes(bytes.clone()).unwrap();
        let fixture = Fixture::new(&file, &bytes);
        fixture.address(StoredAddress::undefined(fixture.widths.offsets));
    }

    #[test]
    #[should_panic(expected = "fixture address exceeds its encoded width")]
    fn an_address_wider_than_its_encoded_field_is_rejected() {
        let bytes = FileBuilder::new().finish().unwrap();
        let file = File::from_bytes(bytes.clone()).unwrap();
        let mut fixture = Fixture::new(&file, &bytes);
        fixture.widths.offsets = OffsetWidth::Two;
        fixture.address(StoredAddress::new(0x1_0000));
    }

    #[test]
    #[should_panic(expected = "InsufficientRegion")]
    fn an_address_at_the_end_of_the_file_is_rejected() {
        let bytes = FileBuilder::new().finish().unwrap();
        let file = File::from_bytes(bytes.clone()).unwrap();
        let fixture = Fixture::new(&file, &bytes);
        fixture.address(StoredAddress::new(u64::try_from(bytes.len()).unwrap()));
    }

    #[test]
    #[should_panic(expected = "OffsetOverflow")]
    fn an_address_that_overflows_the_file_base_is_rejected() {
        let bytes = FileBuilder::new().finish().unwrap();
        let file = File::from_bytes(bytes.clone()).unwrap();
        let mut fixture = Fixture::new(&file, &bytes);
        fixture.base = BaseAddress::new(u64::MAX);
        fixture.address(StoredAddress::new(1));
    }
}
