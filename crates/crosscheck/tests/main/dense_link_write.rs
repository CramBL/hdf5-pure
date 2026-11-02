#![cfg(feature = "__hdf5-1.10")]

use std::path::Path;

use hdf5::IndexType;
use hdf5::IterationOrder;
use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::Sizeof;
use hdf5::plist::file_create::SizeofInfo;
use hdf5::plist::group_create::LinkCreationOrder;
use hdf5_pure::CharacterSet;
use hdf5_pure::File;
use hdf5_pure::MessageType;
use hdf5_pure::Superblock;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::LengthWidth;
use hdf5_pure_core::OffsetWidth;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::BTreeV2Header;
use hdf5_pure_format::__private::BTreeV2Layout;
use hdf5_pure_format::__private::BTreeV2Node;
use hdf5_pure_format::__private::BTreeV2Plan;
use hdf5_pure_format::__private::BTreeV2WriteDialect;
use hdf5_pure_format::__private::DenseLinkPlan;
use hdf5_pure_format::__private::FractalHeapBlock;
use hdf5_pure_format::__private::FractalHeapBlockLayout;
use hdf5_pure_format::__private::FractalHeapHeader;
use hdf5_pure_format::__private::FractalHeapIdKind;
use hdf5_pure_format::__private::FractalHeapIdLayout;
use hdf5_pure_format::__private::GroupLinkStorage;
use hdf5_pure_format::__private::LinkInfoMessage;
use hdf5_pure_format::__private::ObjectHeader;
use hdf5_pure_format::__private::ObjectHeaderWriter;
use hdf5_pure_format::__private::OwnedLink;
use hdf5_pure_format::__private::OwnedLinkTarget;
use rstest::rstest;

#[rstest]
#[case::empty(0, Sizeof::Bytes8, Sizeof::Bytes8, false)]
#[case::two_four(16, Sizeof::Bytes2, Sizeof::Bytes4, false)]
#[case::two_eight(16, Sizeof::Bytes2, Sizeof::Bytes8, false)]
#[case::four_four(16, Sizeof::Bytes4, Sizeof::Bytes4, false)]
#[case::four_eight(16, Sizeof::Bytes4, Sizeof::Bytes8, false)]
#[case::eight_four(16, Sizeof::Bytes8, Sizeof::Bytes4, false)]
#[case::eight_eight(16, Sizeof::Bytes8, Sizeof::Bytes8, false)]
#[case::large(2000, Sizeof::Bytes8, Sizeof::Bytes8, false)]
#[case::managed_limit(16, Sizeof::Bytes8, Sizeof::Bytes8, true)]
fn libhdf5_reads_planned_dense_links_and_creation_indexes(
    #[case] count: usize,
    #[case] address_size: Sizeof,
    #[case] length_size: Sizeof,
    #[case] maximum_object: bool,
    #[values(
        LinkCreationOrder::Untracked,
        LinkCreationOrder::Tracked,
        LinkCreationOrder::Indexed
    )]
    policy: LinkCreationOrder,
    #[values(0, 512)] userblock: u64,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("planned.h5");
    let external = dir.path().join("external.h5");
    let reference = hdf5::File::create(&external).unwrap();
    reference
        .new_dataset::<i32>()
        .shape(1)
        .create("values")
        .unwrap()
        .write(&[17])
        .unwrap();
    reference.close().unwrap();
    let reference = fixture(&path, policy, address_size, length_size, userblock);
    reference
        .new_dataset::<i32>()
        .shape(3)
        .create("payload")
        .unwrap()
        .write(&[1, 2, 3])
        .unwrap();
    reference.close().unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let pure = File::open(&path).unwrap();
    let mut superblock = pure.superblock().clone();
    drop(pure);
    let header = root_header(&bytes, &superblock);
    let payload = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::LINK)
        .unwrap();
    let target = OwnedLink::parse(&payload.data, superblock.offset_size)
        .unwrap()
        .target()
        .clone();
    let storage = planned_storage(
        count,
        policy,
        superblock.offset_size,
        target,
        maximum_object,
    );
    let expected = storage.links().to_vec();
    let expected_counter = storage.info().max_creation_order;
    let plan = DenseLinkPlan::new(storage, superblock.length_size).unwrap();
    let stored = StoredAddress::new(bytes.len() as u64 - userblock);
    let image = plan.serialize(stored).unwrap();
    assert_eq!(image.bytes.len() as u64, plan.serialized_len());
    bytes.extend_from_slice(&image.bytes);
    let mut writer = ObjectHeaderWriter::new();
    writer.add_message(
        MessageType::LINK_INFO,
        image.info.serialize(superblock.offset_size),
    );
    writer.add_message(MessageType::GROUP_INFO, GROUP_INFO.to_vec());
    let new_root = bytes.len() as u64 - userblock;
    bytes.extend_from_slice(&writer.serialize().unwrap());
    superblock.root_group_address = StoredAddress::new(new_root);
    superblock.eof_address = bytes.len() as u64;
    let encoded = format::serialize_superblock(&superblock).unwrap();
    let at = usize::try_from(userblock).unwrap();
    bytes[at..at + encoded.len()].copy_from_slice(&encoded);
    std::fs::write(&path, bytes).unwrap();
    let reference = hdf5::File::open(&path).unwrap();
    let mut actual = reference.member_names().unwrap();
    let mut names = expected
        .iter()
        .map(|link| String::from_utf8(link.name().to_vec()).unwrap())
        .collect::<Vec<_>>();
    actual.sort();
    names.sort();
    assert_eq!(actual, names);
    assert_eq!(
        reference.group("/").unwrap().info().unwrap().nlinks,
        count as u64
    );
    if let Some(counter) = expected_counter {
        assert_eq!(
            reference.group("/").unwrap().info().unwrap().max_corder,
            i64::try_from(counter).unwrap()
        );
        let ordered = reference
            .group("/")
            .unwrap()
            .links(IndexType::CreationOrder, IterationOrder::Increasing)
            .unwrap();
        let mut expected = expected.clone();
        expected.sort_by_key(OwnedLink::creation_order);
        assert_eq!(
            ordered
                .into_iter()
                .map(|(name, info)| (name.into_bytes(), info.creation_order.unwrap()))
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|link| (link.name().to_vec(), link.creation_order().unwrap() as i64))
                .collect::<Vec<_>>()
        );
    }
    if count > 0 {
        for name in ["payload", "k69209", "别名", "utf8_链"] {
            assert_eq!(
                reference.dataset(name).unwrap().read_raw::<i32>().unwrap(),
                vec![1, 2, 3]
            );
        }
        assert_eq!(
            reference
                .dataset("k155448")
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            vec![17]
        );
    }
    reference.close().unwrap();
    let pure = File::open(&path).unwrap();
    assert_eq!(
        pure.group("/").unwrap().datasets().unwrap(),
        if count > 0 {
            vec!["payload".to_string()]
        } else {
            Vec::<String>::new()
        }
    );
    if count > 0 {
        assert_eq!(
            pure.dataset("k69209").unwrap().read_i32().unwrap(),
            vec![1, 2, 3]
        );
    }
}

#[rstest]
#[case::two(OffsetWidth::Two, Sizeof::Bytes2)]
#[case::four(OffsetWidth::Four, Sizeof::Bytes4)]
#[case::eight(OffsetWidth::Eight, Sizeof::Bytes8)]
fn rebuilt_libhdf5_dense_links_preserve_metadata_and_creation_policy(
    #[case] offsets: OffsetWidth,
    #[case] address_size: Sizeof,
    #[values(Sizeof::Bytes4, Sizeof::Bytes8)] length_size: Sizeof,
    #[values(
        LinkCreationOrder::Untracked,
        LinkCreationOrder::Tracked,
        LinkCreationOrder::Indexed
    )]
    policy: LinkCreationOrder,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("libhdf5.h5");
    let reference = fixture(&path, policy, address_size, length_size, 0);
    reference
        .new_dataset::<i32>()
        .shape(1)
        .create("payload")
        .unwrap()
        .write(&[23])
        .unwrap();
    for index in 0..15 {
        reference
            .link_soft("/payload", &format!("link_{index}"))
            .unwrap();
    }
    reference.unlink("link_7").unwrap();
    reference
        .link_external("missing.h5", "/value", "external")
        .unwrap();
    reference.close().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let pure = File::open(&path).unwrap();
    let superblock = pure.superblock().clone();
    drop(pure);
    let lengths = superblock.length_size;
    let header = root_header(&bytes, &superblock);
    let info = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::LINK_INFO)
        .unwrap();
    let info = LinkInfoMessage::parse_with_offset_width(&info.data, offsets).unwrap();
    assert!(info.fractal_heap_address.is_some());
    let links = dense_links(&bytes, &info, offsets, lengths);
    let storage = GroupLinkStorage::new(&info, links.clone(), offsets).unwrap();
    let plan = DenseLinkPlan::new(storage, lengths).unwrap();
    let image = plan.serialize(StoredAddress::new(0)).unwrap();
    let mut rebuilt = dense_links(&image.bytes, &image.info, offsets, lengths);
    let mut original = links;
    rebuilt.sort_by(|a, b| a.name().cmp(b.name()));
    original.sort_by(|a, b| a.name().cmp(b.name()));
    assert_eq!(rebuilt, original);
    assert_eq!(image.info.max_creation_order, info.max_creation_order);
    assert_eq!(
        image.info.btree_creation_order_address.is_some(),
        info.btree_creation_order_address.is_some()
    );
}

#[rstest]
#[case::leaf(16, Sizeof::Bytes2, 0)]
#[case::internal(200, Sizeof::Bytes4, 512)]
#[case::depth_two(2000, Sizeof::Bytes8, 0)]
fn libhdf5_reads_independently_placed_btree_nodes(
    #[case] count: usize,
    #[case] address_size: Sizeof,
    #[case] userblock: u64,
    #[values(LinkLookup::FirstLink, LinkLookup::LastLink, LinkLookup::Payload)] lookup: LinkLookup,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("placed_nodes.h5");
    let reference = fixture(
        &path,
        LinkCreationOrder::Untracked,
        address_size,
        Sizeof::Bytes8,
        userblock,
    );
    let dataset = reference
        .new_dataset::<i32>()
        .shape(3)
        .create("payload")
        .unwrap();
    dataset.write(&[3, 5, 7]).unwrap();
    drop(dataset);
    let names: Vec<_> = (1..count).map(|index| format!("item_{index:05}")).collect();
    for name in &names {
        reference.link_soft("/payload", name).unwrap();
    }
    reference.close().unwrap();
    let pure = File::open(&path).unwrap();
    let mut superblock = pure.superblock().clone();
    drop(pure);
    let mut bytes = std::fs::read(&path).unwrap();
    let root = root_header(&bytes, &superblock);
    let info = root
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::LINK_INFO)
        .unwrap();
    let info =
        LinkInfoMessage::parse_with_offset_width(&info.data, superblock.offset_size).unwrap();
    let at = usize::try_from(info.btree_name_index_address.unwrap().get() + userblock).unwrap();
    let header = BTreeV2Header::parse(
        &bytes,
        at,
        superblock.offset_size.get(),
        superblock.length_size.get(),
    )
    .unwrap();
    let layout = BTreeV2Layout::new(&header, superblock.offset_size).unwrap();
    let records = ordered_btree_records(
        &bytes[usize::try_from(userblock).unwrap()..],
        layout.root().unwrap(),
    );
    let plan = BTreeV2Plan::new_with_undefined_empty_root(
        BTreeV2WriteDialect::Libhdf5Global,
        header.tree_type,
        count,
        header.record_size,
        header.node_size,
        superblock.offset_size,
    )
    .unwrap();
    let base = bytes.len() as u64 - userblock;
    let addresses: Vec<_> = (0..plan.block_lengths().len())
        .rev()
        .map(|index| StoredAddress::new(base + index as u64 * (u64::from(header.node_size) + 37)))
        .collect();
    let placement = plan.place(&addresses).unwrap();
    let emitted_header = placement.serialize_header(superblock.length_size).unwrap();
    let rebuilt = BTreeV2Header::parse(
        &emitted_header,
        0,
        superblock.offset_size.get(),
        superblock.length_size.get(),
    )
    .unwrap();
    assert_eq!(
        rebuilt.depth,
        match count {
            16 => 0,
            200 => 1,
            _ => 2,
        }
    );
    assert_eq!(rebuilt.root_node_address, *addresses.last().unwrap());
    for node in placement.serialize_nodes(&records).unwrap() {
        let (address, data) = node.unwrap();
        let start = usize::try_from(address.get() + userblock).unwrap();
        bytes.resize(bytes.len().max(start + data.len()), 0);
        bytes[start..start + data.len()].copy_from_slice(&data);
    }
    bytes[at..at + emitted_header.len()].copy_from_slice(&emitted_header);
    superblock.eof_address = bytes.len() as u64;
    let encoded = format::serialize_superblock(&superblock).unwrap();
    let start = usize::try_from(userblock).unwrap();
    bytes[start..start + encoded.len()].copy_from_slice(&encoded);
    std::fs::write(&path, bytes).unwrap();
    let reference = hdf5::File::open(&path).unwrap();
    let mut expected = names;
    expected.push("payload".into());
    expected.sort();
    let mut actual = reference.member_names().unwrap();
    actual.sort();
    assert_eq!(actual, expected);
    let link_name = match lookup {
        LinkLookup::FirstLink => "item_00001".to_owned(),
        LinkLookup::LastLink => format!("item_{:05}", count - 1),
        LinkLookup::Payload => "payload".to_owned(),
    };
    let dataset = reference.dataset(&link_name).unwrap();
    assert_eq!(dataset.read_raw::<i32>().unwrap(), [3, 5, 7]);
    drop(dataset);
    reference.close().unwrap();
}

#[derive(Clone, Copy)]
enum LinkLookup {
    FirstLink,
    LastLink,
    Payload,
}

fn ordered_btree_records(bytes: &[u8], node: BTreeV2Node) -> Vec<u8> {
    let view = node
        .parse(&bytes[usize::try_from(node.address().get()).unwrap()..])
        .unwrap();
    let mut records = Vec::new();
    for index in 0..=view.record_count() {
        if let Some(child) = view.child(index).unwrap() {
            records.extend(ordered_btree_records(bytes, child));
        }
        if index < view.record_count() {
            records.extend_from_slice(view.record(index).unwrap());
        }
    }
    records
}

fn fixture(
    path: &Path,
    policy: LinkCreationOrder,
    addresses: Sizeof,
    lengths: Sizeof,
    userblock: u64,
) -> hdf5::File {
    hdf5::File::with_options()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::V18, LibraryVersion::latest())
        })
        .with_fcpl(|properties| {
            properties.sizes(SizeofInfo {
                sizeof_addr: addresses,
                sizeof_size: lengths,
            });
            properties.link_creation_order(policy).userblock(userblock)
        })
        .create(path)
        .unwrap()
}

fn root_header(bytes: &[u8], superblock: &Superblock) -> ObjectHeader {
    ObjectHeader::parse(
        bytes,
        AccessMode::ReadOnly,
        usize::try_from(superblock.root_group_address.get() + superblock.base_address.get())
            .unwrap(),
        superblock.offset_size.get(),
        superblock.length_size.get(),
    )
    .unwrap()
}

fn planned_storage(
    count: usize,
    policy: LinkCreationOrder,
    offsets: OffsetWidth,
    target: OwnedLinkTarget,
    maximum_object: bool,
) -> GroupLinkStorage {
    let tracked = policy != LinkCreationOrder::Untracked;
    let info = LinkInfoMessage {
        max_creation_order: tracked.then_some(CREATION_BASE + count as u64 * 3 + 7),
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: (policy == LinkCreationOrder::Indexed).then_some(None),
    };
    let links = (0..count)
        .rev()
        .map(|index| {
            let (name, charset, target) = match index {
                0 => (b"payload".to_vec(), CharacterSet::Ascii, target.clone()),
                1 => (
                    b"k69209".to_vec(),
                    CharacterSet::Ascii,
                    OwnedLinkTarget::soft(b"/payload".to_vec()),
                ),
                2 => (
                    b"k155448".to_vec(),
                    CharacterSet::Ascii,
                    OwnedLinkTarget::external(b"external.h5".to_vec(), b"/values".to_vec()),
                ),
                3 => (
                    "别名".as_bytes().to_vec(),
                    CharacterSet::Ascii,
                    OwnedLinkTarget::soft(b"/payload".to_vec()),
                ),
                4 => (
                    "utf8_链".as_bytes().to_vec(),
                    CharacterSet::Utf8,
                    OwnedLinkTarget::soft(b"/payload".to_vec()),
                ),
                _ => (
                    format!(
                        "link_{index:06}{}",
                        "x".repeat(if count > 1000 { 1200 } else { 0 })
                    )
                    .into_bytes(),
                    CharacterSet::Ascii,
                    OwnedLinkTarget::soft(b"/payload".to_vec()),
                ),
            };
            let order = tracked.then_some(CREATION_BASE + index as u64 * 3 + 1);
            let mut link = OwnedLink::new(name, charset.clone(), target.clone(), order).unwrap();
            if maximum_object && index == 5 {
                let prefix = link.encoded_len(offsets).unwrap() - link.name().len();
                let name_len =
                    usize::try_from(format::GROUP_HEAP_MAX_MANAGED_OBJECT).unwrap() - prefix - 1;
                link = OwnedLink::new(vec![b'x'; name_len], charset, target, order).unwrap();
                assert_eq!(
                    link.encoded_len(offsets).unwrap() as u64,
                    format::GROUP_HEAP_MAX_MANAGED_OBJECT
                );
            }
            link
        })
        .collect();
    GroupLinkStorage::new(&info, links, offsets).unwrap()
}

fn dense_links(
    bytes: &[u8],
    info: &LinkInfoMessage,
    offsets: OffsetWidth,
    lengths: LengthWidth,
) -> Vec<OwnedLink> {
    let owner = info.fractal_heap_address.unwrap();
    let header = FractalHeapHeader::parse(
        bytes,
        usize::try_from(owner.get()).unwrap(),
        offsets.get(),
        lengths.get(),
    )
    .unwrap();
    let heap = FractalHeapBlockLayout::new(&header, offsets, lengths, owner).unwrap();
    let region = FileRegion::new(
        FileRange::new(PhysicalAddress::new(0), bytes.len() as u64).unwrap(),
        bytes,
    )
    .unwrap();
    let mut pending = heap.root().unwrap().into_iter().collect::<Vec<_>>();
    let mut direct = Vec::new();
    while let Some(block) = pending.pop() {
        match block {
            FractalHeapBlock::Direct(block) => {
                block
                    .at(PhysicalAddress::new(block.address().get()))
                    .unwrap()
                    .parse_checked(region)
                    .unwrap();
                direct.push(block);
            }
            FractalHeapBlock::Indirect(block) => {
                let reference = block
                    .at(PhysicalAddress::new(block.address().get()))
                    .unwrap();
                pending.extend(
                    reference
                        .parse(region)
                        .unwrap()
                        .children()
                        .map(Result::unwrap),
                );
            }
        }
    }
    let index = BTreeV2Header::parse(
        bytes,
        usize::try_from(info.btree_name_index_address.unwrap().get()).unwrap(),
        offsets.get(),
        lengths.get(),
    )
    .unwrap();
    assert_eq!(index.depth, 0);
    let layout = BTreeV2Layout::new(&index, offsets).unwrap();
    let root = layout.root().unwrap();
    let view = root
        .parse(&bytes[usize::try_from(root.address().get()).unwrap()..])
        .unwrap();
    let ids = FractalHeapIdLayout::new(&header, offsets, lengths);
    (0..view.record_count())
        .map(|index| {
            let record = view.record(index).unwrap();
            let id = &record[NAME_HASH_LEN..];
            let FractalHeapIdKind::Managed {
                heap_offset,
                object_length,
            } = ids.parse(id).unwrap().kind()
            else {
                panic!("expected managed heap ID");
            };
            let block = direct
                .iter()
                .find(|block| {
                    heap_offset >= block.heap_offset()
                        && heap_offset < block.heap_offset() + block.decoded_size()
                })
                .unwrap();
            let object = block
                .at(PhysicalAddress::new(block.address().get()))
                .unwrap()
                .parse_object(region, heap_offset, object_length)
                .unwrap();
            OwnedLink::parse(object.parse(region).unwrap(), offsets).unwrap()
        })
        .collect()
}

// "The Group Info Message", format specification version 4.0, defines the version and flags.
const GROUP_INFO: [u8; 2] = [0, 0];
// "Version 2 B-trees", format specification version 4.0, defines the type-5 hash field.
const NAME_HASH_LEN: usize = 4;

const CREATION_BASE: u64 = 1 << 40;
