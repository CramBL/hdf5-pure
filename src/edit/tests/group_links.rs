use core::ops::Range;
use std::path::Path;

use hdf5_pure_format::__private::BTreeV2Layout;
use hdf5_pure_format::__private::DenseLinkPlan;
use hdf5_pure_format::__private::FractalHeapHeader;
use hdf5_pure_format::__private::GroupLinkStorage;
use hdf5_pure_format::__private::LinkInfoMessage;
use hdf5_pure_format::__private::ObjectHeaderPrefix;
use hdf5_pure_format::__private::OwnedLink;
use hdf5_pure_format::__private::OwnedLinkTarget;
use rstest::rstest;

use crate::File;
use crate::address::AddressContext;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::btree_v2::BTreeV2Header;
use crate::datatype::CharacterSet;
use crate::edit::WriteEngine;
use crate::edit::group;
use crate::edit::group_links;
use crate::edit::object_header::OhRegion;
use crate::error::Error;
use crate::error::FormatError;
use crate::file_lock::FileLocking;
use crate::link_message::LinkMessage;
use crate::message_type::MessageType;
use crate::object_path::ObjectPathBuf;
use crate::source::BytesSource;
use crate::width::FormatWidths;
use crate::width::LengthWidth;
use crate::width::OffsetWidth;
use crate::writer::FileBuilder;

#[rstest]
#[case::untracked(false, false)]
#[case::tracked(true, false)]
#[case::indexed(true, true)]
fn inspection_preserves_encoded_names_and_targets(
    #[case] tracked: bool,
    #[case] indexed: bool,
    #[values(false, true)] dense: bool,
    #[values(OffsetWidth::Four, OffsetWidth::Eight)] offset_width: OffsetWidth,
    #[values(512, 1024)] effective_base: u64,
) {
    let widths = FormatWidths {
        offsets: offset_width,
        lengths: LengthWidth::Eight,
    };
    let info = LinkInfoMessage {
        max_creation_order: tracked.then_some(99),
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: indexed.then_some(None),
    };
    let links = vec![
        OwnedLink::new(
            vec![0xFE],
            CharacterSet::Ascii,
            OwnedLinkTarget::Hard {
                object_header_address: StoredAddress::new(42),
            },
            tracked.then_some(2),
        )
        .unwrap(),
        OwnedLink::new(
            vec![0xFF],
            CharacterSet::Utf8,
            OwnedLinkTarget::soft(vec![b'/', 0xFF]),
            tracked.then_some(7),
        )
        .unwrap(),
        OwnedLink::new(
            b"external".to_vec(),
            CharacterSet::Ascii,
            OwnedLinkTarget::external(vec![0xFE], vec![b'/', 0xFF]),
            tracked.then_some(31),
        )
        .unwrap(),
    ];
    let storage = GroupLinkStorage::new(&info, links.clone(), offset_width).unwrap();
    let (bytes, region) = fixture(storage, dense);
    let mut source = vec![0xAC; usize::try_from(effective_base - 512).unwrap()];
    source.extend_from_slice(&bytes);
    let inspected = group_links::inspect(
        &BytesSource::new(&source),
        AddressContext::from_effective_base(BaseAddress::new(effective_base)),
        widths,
        &region,
    )
    .unwrap();
    let mut actual = inspected.links().to_vec();
    let mut expected = links;
    actual.sort_by(|a, b| a.name().cmp(b.name()));
    expected.sort_by(|a, b| a.name().cmp(b.name()));
    assert_eq!(actual, expected);
    assert_eq!(
        inspected.creation_order().next_index(),
        tracked.then_some(99)
    );
    assert_eq!(inspected.creation_order().is_indexed(), indexed);
    if !dense {
        let error = group::GroupInfo {
            region,
            links: inspected,
        }
        .into_compact()
        .err()
        .unwrap();
        assert!(
            matches!(
                error,
                Error::EditUnsupported(
                    "a group's link name cannot be represented by the editor's UTF-8 paths"
                )
            ),
            "{error:?}"
        );
    }
}

#[rstest]
#[case::huge(|header: &mut FractalHeapHeader| header.huge_objects_count = 1, "a group's fractal heap uses filters, huge objects, or tiny objects")]
#[case::tiny(|header: &mut FractalHeapHeader| header.tiny_objects_count = 1, "a group's fractal heap uses filters, huge objects, or tiny objects")]
#[case::reserved_flags(|header: &mut FractalHeapHeader| header.flags |= 0x80, "a group's fractal heap uses unsupported flags")]
#[case::heap_id_width(|header: &mut FractalHeapHeader| header.heap_id_length = 6, "a group's name index uses an unsupported record encoding")]
#[case::huge_index(|header: &mut FractalHeapHeader| header.btree_huge_objects_address = StoredAddress::new(42), "a group's fractal heap uses filters, huge objects, or tiny objects")]
fn dense_inspection_rejects_unsupported_heap_configurations(
    #[case] change: fn(&mut FractalHeapHeader),
    #[case] reason: &str,
) {
    let (mut bytes, region) = fixture(one_link_storage(), true);
    let mut header = FractalHeapHeader::parse(&bytes, 512, 8, 8).unwrap();
    change(&mut header);
    let encoded = header
        .serialize(OffsetWidth::Eight, LengthWidth::Eight)
        .unwrap();
    bytes[512..512 + encoded.len()].copy_from_slice(&encoded);
    let error =
        group_links::inspect(&BytesSource::new(bytes), addresses(), widths(), &region).unwrap_err();
    assert!(
        matches!(error, Error::EditUnsupported(message) if message == reason),
        "{error:?}"
    );
}

#[rstest]
#[case::heap_count(1, |bytes: &mut Vec<u8>, _info: &LinkInfoMessage| {
    let mut header = FractalHeapHeader::parse(bytes, 512, 8, 8).unwrap();
    header.managed_objects_count += 1;
    let encoded = header.serialize(OffsetWidth::Eight, LengthWidth::Eight).unwrap();
    bytes[512..512 + encoded.len()].copy_from_slice(&encoded);
}, Some("a group's heap and name index have different link counts"))]
#[case::name_hash(1, |bytes: &mut Vec<u8>, info: &LinkInfoMessage| {
    let at = name_leaf(bytes, info) + 6;
    bytes[at] ^= 1;
    update_checksum(bytes, at - 6, 11);
}, Some("a group's name index has inconsistent keys"))]
#[case::heap_id_flags(1, |bytes: &mut Vec<u8>, info: &LinkInfoMessage| {
    let at = name_leaf(bytes, info) + 10;
    bytes[at] |= 1;
    update_checksum(bytes, at - 10, 11);
}, Some("a group's link uses a non-managed heap ID"))]
#[case::creation_id(1, |bytes: &mut Vec<u8>, info: &LinkInfoMessage| {
    let address = info.btree_creation_order_address.flatten().unwrap();
    let header = BTreeV2Header::parse(bytes, 512 + usize::try_from(address.get()).unwrap(), 8, 8).unwrap();
    let at = 512 + usize::try_from(header.root_node_address.get()).unwrap();
    bytes[at + 14] ^= 1;
    update_checksum(bytes, at, 15);
}, Some("a group's creation-order index disagrees with its links"))]
#[case::child_subtree_total(2000, |bytes: &mut Vec<u8>, info: &LinkInfoMessage| {
    let fields = RootChildFields::parse(bytes, info);
    let at = fields.children + 8 + fields.record_count_size;
    let width = fields.pointer_size - 8 - fields.record_count_size;
    let total = test_util::bytes::uint_at(bytes, at, width);
    test_util::bytes::set_uint_at(bytes, at, width, total - 1);
    fields.restamp(bytes);
}, None)]
#[case::repeated_child(2000, |bytes: &mut Vec<u8>, info: &LinkInfoMessage| {
    let fields = RootChildFields::parse(bytes, info);
    let first = test_util::bytes::uint_at(bytes, fields.children, 8);
    test_util::bytes::set_uint_at(bytes, fields.children + fields.pointer_size, 8, first);
    fields.restamp(bytes);
}, None)]
fn dense_inspection_rejects_inconsistent_index_metadata(
    #[case] count: usize,
    #[case] change: fn(&mut Vec<u8>, &LinkInfoMessage),
    #[case] reason: Option<&str>,
) {
    let (mut bytes, region) = fixture(
        if count == 1 {
            one_link_storage()
        } else {
            multilevel_storage(count)
        },
        true,
    );
    let (_, start, end) = find_message(&region, MessageType::LINK_INFO);
    let info = LinkInfoMessage::parse(&region[start..end], 8).unwrap();
    change(&mut bytes, &info);
    let error =
        group_links::inspect(&BytesSource::new(bytes), addresses(), widths(), &region).unwrap_err();
    match reason {
        Some(reason) => assert!(
            matches!(error, Error::EditUnsupported(message) if message == reason),
            "{error:?}"
        ),
        None => assert!(
            matches!(error, Error::Format(FormatError::InvalidBTreeV2Signature)),
            "{error:?}"
        ),
    }
}

#[rstest]
#[case::duplicate_info(|region: &mut OhRegion| region.push(MessageType::LINK_INFO, &crate::file_writer::COMPACT_LINK_INFO.serialize(OffsetWidth::Eight)), "a group has repeated link-info messages")]
#[case::duplicate_group_info(|region: &mut OhRegion| region.push(MessageType::GROUP_INFO, &[0, 0]), "a group has repeated group-info messages")]
#[case::symbol_table(|region: &mut OhRegion| region.push(MessageType::SYMBOL_TABLE, &[0; 16]), "a group mixes symbol-table storage with modern link storage")]
#[case::group_info_flags(|region: &mut OhRegion| {
    let (_, body, _) = find_message(region, MessageType::GROUP_INFO);
    region.bytes_mut()[body + 1] = 0x80;
}, "a group's group-info message uses an unsupported encoding")]
#[case::link_info_flags(|region: &mut OhRegion| {
    let (_, body, _) = find_message(region, MessageType::LINK_INFO);
    region.bytes_mut()[body + 1] |= 0x80;
}, "a group's links or link-info metadata cannot be reproduced")]
#[case::duplicate_name(|region: &mut OhRegion| {
    let (_, body, end) = find_message(region, MessageType::LINK);
    let bytes = region[body..end].to_vec();
    region.push(MessageType::LINK, &bytes);
}, "a group's links or link-info metadata cannot be reproduced")]
#[case::creation_counter(|region: &mut OhRegion| {
    let (_, body, _) = find_message(region, MessageType::LINK_INFO);
    region.bytes_mut()[body + 2..body + 10].copy_from_slice(&0u64.to_le_bytes());
}, "a group's links or link-info metadata cannot be reproduced")]
fn compact_inspection_rejects_inconsistent_metadata(
    #[case] change: fn(&mut OhRegion),
    #[case] reason: &str,
) {
    let (bytes, mut region) = fixture(one_link_storage(), false);
    change(&mut region);
    let error =
        group_links::inspect(&BytesSource::new(bytes), addresses(), widths(), &region).unwrap_err();
    assert!(
        matches!(error, Error::EditUnsupported(message) if message == reason),
        "{error:?}"
    );
}

#[test]
fn bounded_dense_mutation_preserves_existing_children() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dense.h5");
    let mut builder = FileBuilder::new();
    for index in 0..12 {
        builder
            .create_group_with(&format!("group_{index}"), |_| Ok(()))
            .unwrap();
    }
    builder.write(&path).unwrap();
    let mut engine = super::common::open_bounded_session(&path);
    let address = engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap()
        .get();
    let info = engine.inspect_group(address).unwrap();
    assert_eq!(info.links.links().len(), 12);
    engine.create_group(&ObjectPathBuf::parse("added")).unwrap();
    engine.commit().unwrap();
    let address = engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap()
        .get();
    let info = engine.inspect_group(address).unwrap();
    assert!(info.links.info().fractal_heap_address.is_some());
    let mut names = info
        .links
        .links()
        .iter()
        .map(|link| link.name().to_vec())
        .collect::<Vec<_>>();
    names.sort();
    let mut expected = (0..12)
        .map(|index| format!("group_{index}").into_bytes())
        .collect::<Vec<_>>();
    expected.push(b"added".to_vec());
    expected.sort();
    assert_eq!(names, expected);
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().groups().unwrap().len(), 13);
}

#[test]
fn deleting_a_replacement_character_name_rejects_invalid_utf8_before_writing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("names.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("a").with_i32_data(&[1]);
    builder.create_dataset("�").with_i32_data(&[2]);
    builder.write(&path).unwrap();
    let file = File::open(&path).unwrap();
    let root = usize::try_from(file.superblock().root_group_address.get()).unwrap();
    drop(file);
    let mut original = std::fs::read(&path).unwrap();
    let prefix = ObjectHeaderPrefix::parse(&original[root..]).unwrap();
    let end = root + prefix.len + usize::try_from(prefix.chunk0_size).unwrap();
    let mut pos = root + prefix.len;
    let mut replaced = false;
    while let Some(record) = prefix
        .prefix
        .layout
        .next_message(&original[..end], pos)
        .unwrap()
    {
        if record.msg_type == MessageType::LINK
            && OwnedLink::parse(record.body, OffsetWidth::Eight)
                .unwrap()
                .name()
                == b"a"
        {
            let target = LinkMessage::hard_link_address_range(record.body, OffsetWidth::Eight)
                .unwrap()
                .unwrap();
            let at = record.body_range.start + target.start - 1;
            assert_eq!(original[at], b'a');
            original[at] = 0xFF;
            replaced = true;
            break;
        }
        pos = record.body_range.end;
    }
    assert!(replaced);
    let checksum = crate::checksum::jenkins_lookup3(&original[root..end]);
    original[end..end + 4].copy_from_slice(&checksum.to_le_bytes());
    std::fs::write(&path, &original).unwrap();
    let file = File::open_rw(&path).unwrap();
    file.root().delete("�").unwrap();
    let error = file.commit().unwrap_err();
    assert!(
        matches!(
            error,
            Error::EditUnsupported(
                "a group's link name cannot be represented by the editor's UTF-8 paths"
            )
        ),
        "{error:?}"
    );
    drop(file);
    assert_eq!(std::fs::read(&path).unwrap(), original);
}

struct RootChildFields {
    root: Range<usize>,
    children: usize,
    pointer_size: usize,
    record_count_size: usize,
}

impl RootChildFields {
    fn parse(bytes: &[u8], info: &LinkInfoMessage) -> Self {
        let header = BTreeV2Header::parse(
            bytes,
            512 + usize::try_from(info.btree_name_index_address.unwrap().get()).unwrap(),
            8,
            8,
        )
        .unwrap();
        assert_eq!(header.depth, 2);
        let layout = BTreeV2Layout::new(&header, OffsetWidth::Eight).unwrap();
        let node = layout.root().unwrap();
        let start = 512 + usize::try_from(node.address().get()).unwrap();
        let length = layout.encoded_len(node).unwrap();
        let view = layout.parse(&bytes[start..], node).unwrap();
        let records = view.record_count();
        let prefix = test_util::btree_v2::NODE_PREFIX;
        let records_length = records * usize::from(header.record_size);
        let children_length = length - prefix - records_length - test_util::checksum::CHECKSUM;
        let leaf_capacity = (header.node_size
            - u32::try_from(prefix + test_util::checksum::CHECKSUM).unwrap())
            / u32::from(header.record_size);
        Self {
            root: start..start + length,
            children: start + prefix + records_length,
            pointer_size: children_length / (records + 1),
            record_count_size: usize::try_from(
                (u32::BITS - leaf_capacity.leading_zeros()).div_ceil(8),
            )
            .unwrap(),
        }
    }

    fn restamp(&self, bytes: &mut [u8]) {
        test_util::checksum::restamp(bytes, self.root.start, self.root.len());
    }
}

fn multilevel_storage(count: usize) -> GroupLinkStorage {
    let info = LinkInfoMessage {
        max_creation_order: Some(u64::try_from(count).unwrap() + 5),
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: Some(None),
    };
    let links = (0..count)
        .map(|index| {
            OwnedLink::new(
                format!("link_{index:04}").into_bytes(),
                CharacterSet::Ascii,
                OwnedLinkTarget::soft(b"/target".to_vec()),
                Some(u64::try_from(index).unwrap() + 3),
            )
            .unwrap()
        })
        .collect();
    GroupLinkStorage::new(&info, links, OffsetWidth::Eight).unwrap()
}

fn one_link_storage() -> GroupLinkStorage {
    let info = LinkInfoMessage {
        max_creation_order: Some(5),
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: Some(None),
    };
    GroupLinkStorage::new(
        &info,
        vec![
            OwnedLink::new(
                b"link".to_vec(),
                CharacterSet::Ascii,
                OwnedLinkTarget::soft(b"/target".to_vec()),
                Some(3),
            )
            .unwrap(),
        ],
        OffsetWidth::Eight,
    )
    .unwrap()
}

fn addresses() -> AddressContext {
    AddressContext::from_effective_base(BaseAddress::new(512))
}

fn widths() -> FormatWidths {
    FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    }
}

fn name_leaf(bytes: &[u8], info: &LinkInfoMessage) -> usize {
    let header = BTreeV2Header::parse(
        bytes,
        512 + usize::try_from(info.btree_name_index_address.unwrap().get()).unwrap(),
        8,
        8,
    )
    .unwrap();
    assert_eq!(header.depth, 0);
    512 + usize::try_from(header.root_node_address.get()).unwrap()
}

fn update_checksum(bytes: &mut [u8], at: usize, record_size: usize) {
    let checksum_at = at + 6 + record_size;
    let checksum = crate::checksum::jenkins_lookup3(&bytes[at..checksum_at]);
    bytes[checksum_at..checksum_at + 4].copy_from_slice(&checksum.to_le_bytes());
}

fn fixture(storage: GroupLinkStorage, dense: bool) -> (Vec<u8>, OhRegion) {
    let offset_width = storage.offset_width();
    let mut region = OhRegion::empty(ObjectHeaderPrefix::PLAIN);
    let mut source = vec![0xAC; 512];
    if dense {
        let image = DenseLinkPlan::new(storage, LengthWidth::Eight)
            .unwrap()
            .serialize(StoredAddress::new(0))
            .unwrap();
        source.extend_from_slice(&image.bytes);
        region.push(MessageType::LINK_INFO, &image.info.serialize(offset_width));
    } else {
        region.push(
            MessageType::LINK_INFO,
            &storage.info().serialize(storage.offset_width()),
        );
        for link in storage.links() {
            region.push(
                MessageType::LINK,
                &link.serialize(storage.offset_width()).unwrap(),
            );
        }
    }
    region.push(MessageType::GROUP_INFO, &[0, 3, 12, 0, 4, 0, 7, 0, 20, 0]);
    (source, region)
}

fn find_message(region: &OhRegion, wanted: MessageType) -> (usize, usize, usize) {
    let mut pos = 0;
    while let Some((kind, body, end)) = region.next_message(pos).unwrap() {
        if kind == wanted {
            return (pos, body, end);
        }
        pos = end;
    }
    panic!("missing message: {wanted:?}");
}

#[rstest]
fn replaced_dense_metadata_is_reclaimed_after_publication(
    #[values(false, true)] bounded: bool,
    #[values(None, Some(false), Some(true))] indexing: Option<bool>,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reclaim.h5");
    let mut builder = FileBuilder::new();
    for index in 0..12 {
        builder
            .create_group_with(&format!("group_{index}"), |_| Ok(()))
            .unwrap();
    }
    builder.create_dataset("payload").with_i32_data(&[7; 4096]);
    builder.with_file_space_settings(
        crate::FileSpaceSettings::new(crate::FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(0),
    );
    builder.write(&path).unwrap();
    if let Some(indexed) = indexing {
        track_dense_root(&path, indexed);
    }
    let mut engine = if bounded {
        super::common::open_bounded_session(&path)
    } else {
        crate::edit::WriteEngine::open_with_locking(&path, crate::file_lock::FileLocking::Enabled)
            .unwrap()
    };
    let old_root = engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap()
        .get();
    let old = engine.inspect_group(old_root).unwrap();
    let header = engine.oh_chunk_spans(old_root).unwrap();
    let storage = engine
        .dense_group_storage_spans(&header, None, &old.links.info())
        .unwrap();
    assert!(!storage.is_empty());
    engine.create_group(&ObjectPathBuf::parse("added")).unwrap();
    engine.commit().unwrap();
    for &(start, length, _) in &storage {
        assert!(
            engine
                .space
                .reusable_sections()
                .any(|extent| extent.start() <= start && extent.end() >= start + length),
            "missing span {start}..{}",
            start + length
        );
    }
    let first_len = engine.image().len();
    for _ in 0..6 {
        engine.delete(&ObjectPathBuf::parse("added")).unwrap();
        engine.commit().unwrap();
        engine.create_group(&ObjectPathBuf::parse("added")).unwrap();
        engine.commit().unwrap();
    }
    assert!(engine.image().len() < first_len + 32 * 1024);
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().groups().unwrap().len(), 13);
    assert_eq!(
        file.dataset("payload").unwrap().read_i32().unwrap(),
        [7; 4096]
    );
    drop(file);
    dir.close().unwrap();
}

#[rstest]
fn dense_metadata_write_failure_preserves_the_published_tree(
    #[values(false, true)] bounded: bool,
    #[values(0, 17)] prefix: usize,
    #[values(None, Some(false), Some(true))] indexing: Option<bool>,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rollback.h5");
    let mut builder = FileBuilder::new();
    for index in 0..12 {
        builder
            .create_group_with(&format!("group_{index}"), |_| Ok(()))
            .unwrap();
    }
    builder.write(&path).unwrap();
    if let Some(indexed) = indexing {
        track_dense_root(&path, indexed);
    }
    let original = std::fs::read(&path).unwrap();
    let mut engine = if bounded {
        super::common::open_bounded_session(&path)
    } else {
        crate::edit::WriteEngine::open_with_locking(&path, crate::file_lock::FileLocking::Enabled)
            .unwrap()
    };
    engine
        .set_write_buffering(crate::image::WriteBuffering::Unbuffered)
        .unwrap();
    let root = engine.superblock().root_group_address;
    let free = engine.reusable_free_sections().unwrap();
    engine.create_group(&ObjectPathBuf::parse("added")).unwrap();
    test_util::disk_log::fail_write_after(1, prefix);
    let error = engine.commit().unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected an injected I/O error, got {error:?}")
    };
    assert_eq!(error.to_string(), "injected disk write failure");
    assert_eq!(engine.superblock().root_group_address, root);
    assert_eq!(engine.reusable_free_sections().unwrap(), free);
    drop(engine);
    let bytes = std::fs::read(&path).unwrap();
    let start = usize::try_from(root.get()).unwrap();
    assert_eq!(&bytes[start..original.len()], &original[start..]);
    let file = File::open(&path).unwrap();
    let mut names = file.root().groups().unwrap();
    names.sort();
    let mut expected = (0..12)
        .map(|index| format!("group_{index}"))
        .collect::<Vec<_>>();
    expected.sort();
    assert_eq!(names, expected);
    drop(file);
    dir.close().unwrap();
}

fn track_dense_root(path: &Path, indexed: bool) {
    let engine = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
    let mut superblock = engine.superblock().clone();
    let root = engine
        .addresses()
        .resolve(superblock.root_group_address)
        .unwrap()
        .get();
    let inspected = engine.inspect_group(root).unwrap();
    drop(engine);
    let links = inspected
        .links
        .links()
        .iter()
        .enumerate()
        .map(|(index, link)| {
            OwnedLink::new(
                link.name().to_vec(),
                link.charset(),
                link.target().clone(),
                Some(u64::try_from(index).unwrap()),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let info = LinkInfoMessage {
        max_creation_order: Some(u64::try_from(links.len()).unwrap()),
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: indexed.then_some(None),
    };
    let storage = GroupLinkStorage::new(&info, links, OffsetWidth::Eight).unwrap();
    let mut bytes = std::fs::read(path).unwrap();
    let plan = DenseLinkPlan::new(storage, LengthWidth::Eight).unwrap();
    let image = plan
        .serialize(StoredAddress::new(u64::try_from(bytes.len()).unwrap()))
        .unwrap();
    bytes.extend_from_slice(&image.bytes);
    let region = inspected.region;
    let splice = region
        .splice_first(MessageType::LINK_INFO)
        .unwrap()
        .unwrap();
    let region = region.with_bytes(
        splice
            .replace_body(&image.info.serialize(OffsetWidth::Eight))
            .unwrap(),
    );
    let header = crate::edit::object_header::build_v2_object_header(&region).unwrap();
    superblock.root_group_address = StoredAddress::new(u64::try_from(bytes.len()).unwrap());
    bytes.extend_from_slice(&header);
    superblock.eof_address = u64::try_from(bytes.len()).unwrap();
    let header = hdf5_pure_format::__private::serialize_superblock(&superblock).unwrap();
    bytes[..header.len()].copy_from_slice(&header);
    std::fs::write(path, bytes).unwrap();
}
