//! Version 1 group and chunk trees in files written by libhdf5.

use std::collections::BTreeMap;

use hdf5_pure_format::__private::{BTreeV1ChunkNode, BTreeV1Node};
use rstest::rstest;
use test_util::btree::v1 as btree_v1;
use test_util::temp;
use test_util_hdf5::__private::c_object;
use test_util_hdf5::file;
use test_util_hdf5::reader::ReadBackend;
use test_util_hdf5::soft_link;
use test_util_hdf5::soft_link::GroupFormat;
use test_util_hdf5::soft_link::NativeLink;

use super::*;
use crate::PhysicalAddress;
use crate::access_mode::AccessMode;
use crate::address::OpenedImageContext;
use crate::data_layout::{ChunkIndexLayout, DataLayout};
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::placed_source::PlacedSource;
use crate::source::ReadSeekSource;
use crate::superblock::{self, LocatedSuperblock};

#[test]
fn both_readers_read_the_links_libhdf5_writes_in_a_symbol_table_group() {
    let path = temp::temp_path("symbol_table_links.h5");
    GroupFormat::SymbolTable.generate(&path);
    let (native, group_address, value_address) = {
        let file = file::libhdf5_open(&path);
        (
            soft_link::native_links(&file, "links"),
            c_object::header_address(&file, "links"),
            c_object::header_address(&file, "links/value"),
        )
    };
    let soft = |name: &str, target: &str| {
        (
            String::from(name),
            NativeLink::Soft {
                target: String::from(target),
            },
        )
    };
    assert_eq!(
        native,
        [
            soft("absolute", "/g/value"),
            soft("chain", "relative"),
            soft("dangling", "missing"),
            soft("group", "/g"),
            soft("relative", "value"),
            soft("self", "self"),
            (
                String::from("value"),
                NativeLink::Hard {
                    address: value_address,
                },
            ),
        ]
    );

    let bytes = std::fs::read(&path).unwrap();
    let LocatedSuperblock {
        address,
        superblock,
        widths,
    } = superblock::read(bytes.as_slice()).unwrap();
    let addresses = OpenedImageContext::reconcile(
        superblock.base_address,
        superblock.eof_address,
        address.get(),
    )
    .unwrap()
    .addresses();
    let header = ObjectHeader::parse_with_widths(
        &bytes,
        AccessMode::ReadOnly,
        usize::try_from(group_address).unwrap(),
        widths,
    )
    .unwrap();
    let message = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::SYMBOL_TABLE)
        .unwrap();
    let table = SymbolTableMessage::parse_with_offset_width(&message.data, widths.offsets).unwrap();
    let k = group_leaf_node_k(&superblock);
    let source = ReadSeekSource::new(std::fs::File::open(&path).unwrap()).unwrap();
    let physical = PhysicalBytes::for_test(&bytes, addresses);
    let source = PhysicalSource::for_test(&source, addresses);

    let mut buffered = Vec::new();
    visit_v1_group_links(physical, &table, widths, k, |link| {
        buffered.push(native_link(addresses, link));
        Ok::<_, FormatError>(())
    })
    .unwrap();
    let mut streaming = Vec::new();
    visit_v1_group_links_from_source(source, &table, widths, k, |link| {
        streaming.push(native_link(addresses, link));
        Ok::<_, FormatError>(())
    })
    .unwrap();
    assert_eq!(buffered, native);
    assert_eq!(streaming, native);

    let value = addresses
        .relative(PhysicalAddress::new(value_address))
        .unwrap();
    let listed = |entries: Vec<GroupEntry>| -> Vec<(String, StoredAddress)> {
        entries
            .into_iter()
            .map(|entry| (entry.name, entry.object_header_address))
            .collect()
    };
    let expected = vec![(String::from("value"), value)];
    assert_eq!(
        listed(resolve_v1_group_entries(physical, &table, widths, k).unwrap()),
        expected
    );
    assert_eq!(
        listed(resolve_v1_group_entries_from_source(source, &table, widths, k).unwrap()),
        expected
    );
    for (name, link) in &native {
        let expected = Some(match link {
            NativeLink::Hard { .. } => LinkTarget::Hard {
                object_header_address: value,
            },
            NativeLink::Soft { target } => LinkTarget::soft(target.clone()),
            NativeLink::External {
                filename,
                object_path,
            } => LinkTarget::external(filename.clone(), object_path.clone()),
        });
        let name = LinkName::new(name).unwrap();
        let buffered = find_link(physical, &table, widths, k, name);
        let streaming = find_link_from_source(source, &table, widths, k, name);
        assert_eq!([buffered, streaming], [Ok(expected.clone()), Ok(expected)]);
    }
}

#[rstest]
#[case::buffered(ReadBackend::Buffered)]
#[case::streaming(ReadBackend::Streaming)]
fn native_multilevel_group_and_chunk_trees_preserve_values(#[case] backend: ReadBackend) {
    let path = temp::temp_path("multilevel-v1-trees.h5");
    let expected: Vec<u32> = (0..512).map(|index| index * 3).collect();
    let names: Vec<_> = (0..512).map(|index| format!("item{index:04}")).collect();
    let (group_address, dataset_address) = {
        let native = file::libhdf5_create_earliest(&path);
        let dataset = native
            .new_dataset::<u32>()
            .shape([512])
            .chunk([1])
            .create("data")
            .unwrap();
        dataset.write_raw(&expected).unwrap();
        assert_eq!(dataset.read_raw::<u32>().unwrap(), expected);
        let group = native.create_group("links").unwrap();
        for name in &names {
            group.link_hard("/data", name).unwrap();
        }
        let mut listed = group.member_names().unwrap();
        listed.sort();
        assert_eq!(listed, names);
        let addresses = (
            c_object::header_address(&native, "links"),
            c_object::header_address(&native, "data"),
        );
        drop(group);
        drop(dataset);
        native.close().unwrap();
        addresses
    };
    let bytes = std::fs::read(&path).unwrap();
    let LocatedSuperblock {
        address,
        superblock,
        widths,
    } = superblock::read(bytes.as_slice()).unwrap();
    let addresses = OpenedImageContext::reconcile(
        superblock.base_address,
        superblock.eof_address,
        address.get(),
    )
    .unwrap()
    .addresses();
    let group_header = ObjectHeader::parse_with_widths(
        &bytes,
        AccessMode::ReadOnly,
        usize::try_from(group_address).unwrap(),
        widths,
    )
    .unwrap();
    let message = group_header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::SYMBOL_TABLE)
        .unwrap();
    let table = SymbolTableMessage::parse_with_offset_width(&message.data, widths.offsets).unwrap();
    let group_node = BTreeV1Node::parse_with_widths(
        &bytes,
        usize::try_from(addresses.resolve(table.btree_address).unwrap().get()).unwrap(),
        widths,
    )
    .unwrap();
    assert_eq!(group_node.node_type, btree_v1::NodeType::GROUP.0);
    assert!(group_node.node_level > 0);
    let dataset_header = ObjectHeader::parse_with_widths(
        &bytes,
        AccessMode::ReadOnly,
        usize::try_from(dataset_address).unwrap(),
        widths,
    )
    .unwrap();
    let message = dataset_header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
        .unwrap();
    let layout = DataLayout::parse_with_widths(&message.data, widths).unwrap();
    let DataLayout::Chunked {
        index: ChunkIndexLayout::BTreeV1 {
            address: Some(index),
        },
        ref chunk_dimensions,
        ..
    } = layout
    else {
        panic!("expected a version 1 chunk B-tree, got {layout:?}");
    };
    let chunk_node = BTreeV1ChunkNode::parse_with_offset_width(
        &bytes,
        usize::try_from(addresses.resolve(index).unwrap().get()).unwrap(),
        widths.offsets,
        chunk_dimensions.len(),
    )
    .unwrap();
    assert!(chunk_node.node_level() > 0);

    let source = PlacedSource::new(0, bytes.clone());
    let entries = resolve_v1_group_entries_from_source(
        PhysicalSource::for_test(&source, addresses),
        &table,
        widths,
        group_leaf_node_k(&superblock),
    )
    .unwrap();
    let target = addresses
        .relative(PhysicalAddress::new(dataset_address))
        .unwrap();
    assert_eq!(
        entries
            .into_iter()
            .map(|entry| (entry.name, entry.object_header_address))
            .collect::<Vec<_>>(),
        names
            .iter()
            .map(|name| (name.clone(), target))
            .collect::<Vec<_>>(),
    );
    let reads = source.reads.take();
    let mut visits = BTreeMap::<u64, usize>::new();
    for &(offset, _) in &reads {
        let start = usize::try_from(offset).unwrap();
        if bytes.get(start..start + btree_v1::SIGNATURE.len()) == Some(btree_v1::SIGNATURE) {
            *visits.entry(offset).or_default() += 1;
        }
    }
    assert!(visits.len() > 1);
    for count in visits.values() {
        assert_eq!(*count, 2);
    }
    assert!(reads.len() >= 2 * visits.len() + 2);
    assert!(reads.len() <= 2 * visits.len() + 2 * names.len() + 2);

    let pure = backend.open(&path).unwrap();
    assert_eq!(pure.root().groups().unwrap(), ["links"]);
    assert_eq!(pure.root().datasets().unwrap(), ["data"]);
    let group = pure.group("links").unwrap();
    assert_eq!(group.datasets().unwrap(), names);
    assert_eq!(pure.dataset("data").unwrap().read_u32().unwrap(), expected);
    assert_eq!(
        group.dataset("item0256").unwrap().read_u32().unwrap(),
        expected
    );
    drop(group);
    pure.close().unwrap();
}

fn native_link(addresses: AddressContext, link: LinkRef<'_>) -> (String, NativeLink) {
    (
        String::from(link.name),
        match link.target {
            TargetRef::Hard(object_header_address) => NativeLink::Hard {
                address: addresses.resolve(object_header_address).unwrap().get(),
            },
            TargetRef::Soft(value) => NativeLink::Soft {
                target: String::from(value.path().unwrap()),
            },
        },
    )
}
