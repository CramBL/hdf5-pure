//! Version 1 group links in files written by libhdf5.

use test_util::temp;
use test_util_hdf5::__private::c_object;
use test_util_hdf5::file;
use test_util_hdf5::soft_link;
use test_util_hdf5::soft_link::GroupFormat;
use test_util_hdf5::soft_link::NativeLink;

use super::*;
use crate::access_mode::AccessMode;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::source::ReadSeekSource;

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
    let superblock = crate::superblock::read(bytes.as_slice())
        .unwrap()
        .superblock;
    let widths = FormatWidths {
        offsets: superblock.offset_size,
        lengths: superblock.length_size,
    };
    let addresses = AddressContext::from_effective_base(superblock.base_address);
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

    let mut buffered = Vec::new();
    visit_v1_group_links(&bytes, &table, widths, addresses, k, |link| {
        buffered.push(native_link(addresses, link));
        Ok(())
    })
    .unwrap();
    let mut streaming = Vec::new();
    visit_v1_group_links_from_source(&source, &table, widths, addresses, k, |link| {
        streaming.push(native_link(addresses, link));
        Ok(())
    })
    .unwrap();
    assert_eq!(buffered, native);
    assert_eq!(streaming, native);

    let value = addresses.relative(value_address).unwrap();
    let listed = |entries: Vec<GroupEntry>| -> Vec<(String, StoredAddress)> {
        entries
            .into_iter()
            .map(|entry| (entry.name, entry.object_header_address))
            .collect()
    };
    let expected = vec![(String::from("value"), value)];
    assert_eq!(
        listed(resolve_v1_group_entries(&bytes, &table, widths, addresses, k).unwrap()),
        expected
    );
    assert_eq!(
        listed(
            resolve_v1_group_entries_from_source(&source, &table, widths, addresses, k).unwrap()
        ),
        expected
    );
    for (name, link) in &native {
        let expected = Some(match link {
            NativeLink::Hard { .. } => LinkTarget::Hard {
                object_header_address: value,
            },
            NativeLink::Soft { target } => LinkTarget::soft(target.clone()),
        });
        let name = LinkName::new(name).unwrap();
        let buffered = find_link(bytes.as_slice(), &table, widths, addresses, k, name, || {
            btree_v1::collect_symbol_table_nodes(&bytes, table.btree_address, widths, addresses)
        });
        let streaming = find_link(&source, &table, widths, addresses, k, name, || {
            btree_v1::collect_symbol_table_nodes_from_source(
                &source,
                table.btree_address,
                widths,
                addresses,
            )
        });
        assert_eq!([buffered, streaming], [Ok(expected.clone()), Ok(expected)]);
    }
}

fn native_link(addresses: AddressContext, link: LinkRef<'_>) -> (String, NativeLink) {
    (
        String::from(link.name),
        match link.target {
            TargetRef::Hard(object_header_address) => NativeLink::Hard {
                address: addresses.absolute(object_header_address).unwrap(),
            },
            TargetRef::Soft(value) => NativeLink::Soft {
                target: String::from(value.path().unwrap()),
            },
        },
    )
}
