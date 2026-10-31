use ::test_util;
use test_util::btree_v2;
use test_util::widths::Widths;

use super::*;
use crate::SuperblockVersion;
use crate::data_layout::DataLayout;
use crate::data_read;
use crate::dataspace::Dataspace;
use crate::datatype::{CharacterSet, Datatype};
use crate::message_flags::MessageFlags;
#[cfg(feature = "std")]
use crate::reader::File;
use crate::source::BytesSource;
#[cfg(feature = "std")]
use crate::writer::FileBuilder;

/// Dense-link walks over indexed huge heap objects.
mod huge_links;

#[rstest::rstest]
#[case::present(true)]
#[case::absent(false)]
fn indexed_hash_collisions_compare_names_before_link_targets(#[case] present: bool) {
    let names = ["k155448", "k69209"];
    let hash = checksum::jenkins_lookup3(names[0].as_bytes());
    assert_eq!(hash, checksum::jenkins_lookup3(names[1].as_bytes()));
    let hard = LinkMessage {
        name: names[0].into(),
        link_target: LinkTarget::Hard {
            object_header_address: StoredAddress::new(42),
        },
        creation_order: None,
        charset: CharacterSet::Ascii,
    }
    .serialize(OffsetWidth::Eight);
    let mut soft = LinkMessage {
        name: names[1].into(),
        link_target: LinkTarget::soft("x".into()),
        creation_order: None,
        charset: CharacterSet::Ascii,
    }
    .serialize(OffsetWidth::Eight);
    *soft.last_mut().unwrap() = 0xff;
    let records: Vec<_> = (usize::from(!present)..2)
        .map(|index| {
            let mut record = hash.to_le_bytes().to_vec();
            record.push(u8::try_from(index).unwrap());
            record
        })
        .collect();
    let node = btree_v2::leaf(BTREE_V2_LINK_NAME, &records);
    let header = BTreeV2Header::parse(
        &btree_v2::Header::new(
            BTREE_V2_LINK_NAME,
            5,
            64,
            u16::try_from(records.len()).unwrap(),
        )
        .build(Widths::EIGHT),
        0,
        8,
        8,
    )
    .unwrap();
    let index = BTreeV2LinkNameIndex::new(&header, OffsetWidth::Eight, 1).unwrap();
    let mut bytes = vec![0; 64];
    bytes.extend_from_slice(&node);
    let messages = [hard, soft];
    let mut reads = Vec::new();
    let target = find_indexed_dense_link(
        bytes.as_slice(),
        &index,
        AddressContext::from_effective_base(BaseAddress::ZERO),
        OffsetWidth::Eight,
        LinkName::new(names[0]).unwrap(),
        |id| {
            reads.push(id[0]);
            Ok(messages[usize::from(id[0])].clone())
        },
    )
    .unwrap();
    assert_eq!(
        target,
        present.then_some(LinkTarget::Hard {
            object_header_address: StoredAddress::new(42)
        })
    );
    assert_eq!(reads, if present { vec![1, 0] } else { vec![1] });
}

#[rstest::rstest]
#[case::relative("alias/value", 3)]
#[case::absolute("absolute/value", 4)]
#[case::chain("chain/value", 3)]
#[case::group("alias", 2)]
#[case::current_group_target("alias/again/value", 3)]
fn soft_link_targets_use_relative_or_absolute_starting_groups(
    #[case] path: &str,
    #[case] expected: u64,
) {
    let result = resolve_link_path(
        0,
        1,
        &ObjectPathBuf::parse("g"),
        &ObjectPath::parse(path),
        |address, name| {
            Ok(match (address, name.as_str()) {
                (1, "alias") => {
                    ChildLookup::Found(ChildTarget::Soft(SoftLinkPath::parse("target").unwrap()))
                }
                (1, "absolute") => {
                    ChildLookup::Found(ChildTarget::Soft(SoftLinkPath::parse("/target").unwrap()))
                }
                (1, "chain") => {
                    ChildLookup::Found(ChildTarget::Soft(SoftLinkPath::parse("alias").unwrap()))
                }
                (1, "target") => ChildLookup::Found(ChildTarget::Hard(PhysicalAddress::new(2))),
                (0, "target") => ChildLookup::Found(ChildTarget::Hard(PhysicalAddress::new(5))),
                (2, "value") => ChildLookup::Found(ChildTarget::Hard(PhysicalAddress::new(3))),
                (5, "value") => ChildLookup::Found(ChildTarget::Hard(PhysicalAddress::new(4))),
                (2, "again") => {
                    ChildLookup::Found(ChildTarget::Soft(SoftLinkPath::parse(".").unwrap()))
                }
                _ => ChildLookup::Absent,
            })
        },
    )
    .unwrap();
    assert_eq!(result, expected);
}

#[rstest::rstest]
#[case::self_link("loop")]
#[case::expanding_target("loop/suffix")]
fn soft_link_cycles_stop_after_a_bounded_number_of_lookups(#[case] target: &str) {
    let mut reads = 0;
    let err = resolve_link_path(
        0,
        0,
        &ObjectPathBuf::root(),
        &ObjectPath::parse("loop"),
        |_, _| {
            reads += 1;
            Ok(ChildLookup::Found(ChildTarget::Soft(
                SoftLinkPath::parse(target).unwrap(),
            )))
        },
    )
    .unwrap_err();
    let ResolveError::Format(error) = err else {
        panic!("expected a format error, got {err:?}");
    };
    assert_eq!(
        error,
        FormatError::SoftLinkTraversalLimit {
            path: String::from("loop"),
            limit: 64
        }
    );
    assert_eq!(reads, 65);
}

#[test]
fn structure_errors_preserve_their_format_variant() {
    let expected = FormatError::PathNotFound(String::from("no B-tree v2 name index"));
    let error = resolve_link_path(
        0,
        0,
        &ObjectPathBuf::root(),
        &ObjectPath::parse("alias"),
        |_, name| {
            if name.as_str() == "alias" {
                Ok(ChildLookup::Found(ChildTarget::Soft(
                    SoftLinkPath::parse("target").unwrap(),
                )))
            } else {
                Err(expected.clone())
            }
        },
    )
    .unwrap_err();
    let ResolveError::Format(error) = error else {
        panic!("expected format error, got {error:?}");
    };
    assert_eq!(error, expected);
}

fn test_widths(superblock: &Superblock) -> FormatWidths {
    FormatWidths {
        offsets: superblock.offset_size,
        lengths: superblock.length_size,
    }
}

#[cfg(feature = "std")]
#[rstest::rstest]
#[case::buffered(false)]
#[case::source(true)]
fn path_navigation_uses_the_opened_image_context(#[case] source_backed: bool) {
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::MIN);
    let mut group = builder.create_group("group");
    group.create_dataset("data").with_i32_data(&[1, 2, 3]);
    builder.add_group(group.finish());
    let bytes = builder.finish().unwrap();
    let file = File::from_bytes(bytes.clone()).unwrap();
    let mut superblock = file.superblock().clone();
    let addresses = AddressContext::from_effective_base(superblock.base_address);
    superblock.base_address = BaseAddress::new(2048);
    let path = ObjectPath::parse("group/data");

    let object_at = if source_backed {
        resolve_path_any_from_source(
            &BytesSource::new(&bytes[..]),
            AccessMode::ReadOnly,
            &superblock,
            test_widths(&superblock),
            addresses,
            &path,
        )
    } else {
        resolve_path_any(
            &bytes,
            AccessMode::ReadOnly,
            &superblock,
            test_widths(&superblock),
            addresses,
            &path,
        )
    }
    .unwrap();

    let header = ObjectHeader::parse_with_base_and_widths(
        &bytes,
        AccessMode::ReadOnly,
        object_at.to_usize().unwrap(),
        FormatWidths {
            offsets: superblock.offset_size,
            lengths: superblock.length_size,
        },
        addresses.effective_base(),
    )
    .unwrap();
    let (datatype, dataspace, layout) = extract_dataset(&bytes, &header, test_widths(&superblock));
    let raw = data_read::read_raw_data(
        source::frame(&bytes, addresses.effective_base()).unwrap(),
        &layout,
        &dataspace,
        &datatype,
    )
    .unwrap();
    assert_eq!(data_read::read_as_i32(&raw, &datatype).unwrap(), [1, 2, 3]);
}

const WIDTHS: Widths = Widths::EIGHT;
const NODE_SIZE: u64 = 512;
const RECORD_SIZE: u16 = 16;

fn place_link_name_index(file: &mut [u8], header_at: usize, root_at: usize) -> Vec<Extent> {
    let record = vec![0; usize::from(RECORD_SIZE)];
    let leaf = btree_v2::leaf(BTREE_V2_LINK_NAME, &[record]);
    let header = btree_v2::Header::new(
        BTREE_V2_LINK_NAME,
        RECORD_SIZE,
        u64::try_from(root_at).unwrap(),
        1,
    )
    .build(WIDTHS);
    file[header_at..header_at + header.len()].copy_from_slice(&header);
    file[root_at..root_at + leaf.len()].copy_from_slice(&leaf);
    vec![
        Extent::new(
            StoredAddress::new(u64::try_from(header_at).unwrap()),
            u64::try_from(header.len()).unwrap(),
        )
        .unwrap(),
        Extent::new(
            StoredAddress::new(u64::try_from(root_at).unwrap()),
            NODE_SIZE,
        )
        .unwrap(),
    ]
}

#[test]
fn dense_group_storage_walk_does_not_infer_an_index_from_tracking() {
    let mut file = vec![0; 0x800];
    let expected = place_link_name_index(&mut file, 0x100, 0x200);
    let info = LinkInfoMessage {
        max_creation_order: Some(2),
        fractal_heap_address: Some(StoredAddress::new(0x80)),
        btree_name_index_address: Some(StoredAddress::new(0x100)),
        btree_creation_order_address: None,
    };

    let extents = collect_dense_group_index_storage_extents(
        &BytesSource::new(file),
        &info,
        crate::file_writer::WIDTHS,
    )
    .unwrap();

    assert_eq!(extents, expected);
}

#[test]
fn dense_group_storage_walk_omits_an_undefined_indexed_creation_order_tree() {
    let mut file = vec![0; 0x800];
    let expected = place_link_name_index(&mut file, 0x100, 0x200);
    let info = LinkInfoMessage {
        max_creation_order: Some(2),
        fractal_heap_address: Some(StoredAddress::new(0x80)),
        btree_name_index_address: Some(StoredAddress::new(0x100)),
        btree_creation_order_address: Some(None),
    };

    let extents = collect_dense_group_index_storage_extents(
        &BytesSource::new(file),
        &info,
        crate::file_writer::WIDTHS,
    )
    .unwrap();

    assert_eq!(extents, expected);
}

fn extract_dataset(
    _file_data: &[u8],
    hdr: &ObjectHeader,
    widths: FormatWidths,
) -> (Datatype, Dataspace, DataLayout) {
    let dt_data = &hdr
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::DATATYPE)
        .unwrap()
        .data;
    let ds_data = &hdr
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::DATASPACE)
        .unwrap()
        .data;
    let dl_data = &hdr
        .messages
        .iter()
        .find(|m| m.msg_type == MessageType::DATA_LAYOUT)
        .unwrap()
        .data;
    let (dt, _) = hdf5_pure_format::__private::parse_datatype(dt_data).unwrap();
    let ds = Dataspace::parse_with_length_width(ds_data, widths.lengths).unwrap();
    let dl = DataLayout::parse_with_widths(dl_data, widths).unwrap();
    (dt, ds, dl)
}

#[test]
fn compact_storage_link_messages() {
    let link_data = {
        // Build a Link message: hard link, name="test", addr=0x1000
        let mut d = Vec::new();
        d.push(1); // version
        d.push(0x00); // flags: no creation order, no link type (=hard), no charset, name_size=1byte
        d.push(4); // name length = 4
        d.extend_from_slice(b"test");
        d.extend_from_slice(&0x1000u64.to_le_bytes()); // address
        d
    };

    let oh = ObjectHeader {
        version: 2,
        messages: vec![
            crate::object_header::HeaderMessage {
                msg_type: MessageType::LINK_INFO,
                size: 18,
                flags: MessageFlags::NONE,
                creation_order: None,
                data: {
                    let mut d = Vec::new();
                    d.push(0); // version
                    d.push(0); // flags
                    d.extend_from_slice(&0xFFFF_FFFF_FFFF_FFFFu64.to_le_bytes()); // fractal heap: undefined
                    d.extend_from_slice(&0xFFFF_FFFF_FFFF_FFFFu64.to_le_bytes()); // B-tree: undefined
                    d
                },
            },
            crate::object_header::HeaderMessage {
                msg_type: MessageType::LINK,
                size: link_data.len(),
                flags: MessageFlags::NONE,
                creation_order: None,
                data: link_data,
            },
        ],
        reference_count: None,
        flags: 0,
        access_time: None,
        modification_time: None,
        change_time: None,
        birth_time: None,
    };

    let entries = resolve_v2_group_entries(
        &[],
        AccessMode::ReadOnly,
        &oh,
        FormatWidths::from_sizes(8, 8).unwrap(),
        AddressContext::from_effective_base(BaseAddress::new(0)),
    )
    .unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].name, "test");
    assert_eq!(entries[0].object_header_address, StoredAddress::new(0x1000));
}

/// Builds a version 2 header from filtered messages.
fn header_of(messages: Vec<(MessageType, Vec<u8>)>) -> ObjectHeader {
    ObjectHeader {
        version: 2,
        messages: messages
            .into_iter()
            .map(|(msg_type, data)| crate::object_header::HeaderMessage {
                msg_type,
                size: data.len(),
                flags: MessageFlags::NONE,
                creation_order: None,
                data,
            })
            .collect(),
        reference_count: None,
        flags: 0,
        access_time: None,
        modification_time: None,
        change_time: None,
        birth_time: None,
    }
}

/// Builds a compact Link Info message with an undefined heap address.
fn compact_link_info() -> (MessageType, Vec<u8>) {
    let mut d = vec![0, 0]; // version, flags
    d.extend_from_slice(&u64::MAX.to_le_bytes()); // fractal heap: undefined
    d.extend_from_slice(&u64::MAX.to_le_bytes()); // name index: undefined
    (MessageType::LINK_INFO, d)
}

/// Builds a dense Link Info message with defined heap and name-index addresses.
fn dense_link_info_message() -> (MessageType, Vec<u8>) {
    let mut d = vec![0, 0];
    d.extend_from_slice(&0x1000u64.to_le_bytes()); // fractal heap address
    d.extend_from_slice(&0x2000u64.to_le_bytes()); // name index address
    (MessageType::LINK_INFO, d)
}

// Link messages can identify a compact group even when a lookup filters them all out.
#[test]
fn a_filtered_out_link_still_makes_the_header_a_compact_group() {
    let no_link_info = header_of(vec![]);
    assert!(
        filtered_is_group(&no_link_info, true),
        "a header whose links the filter dropped is still a group"
    );
    assert!(
        holds_compact_links(&no_link_info, OffsetWidth::Eight).unwrap(),
        "and one holding its links compactly, so the scan answers it"
    );
    assert!(
        !filtered_is_group(&no_link_info, false),
        "a header that held no link and no link info is no group of ours"
    );
}

/// Dense storage keeps its links in a fractal heap, so its header holds none
/// to scan whatever the filter saw.
#[test]
fn a_dense_group_is_never_scanned_for_compact_links() {
    let dense = header_of(vec![dense_link_info_message()]);
    assert!(
        filtered_is_group(&dense, true),
        "a dense group is a group, so a lookup goes on to ask the question below"
    );
    assert!(
        !holds_compact_links(&dense, OffsetWidth::Eight).unwrap(),
        "a fractal heap decides, not the presence of a link message"
    );
}

#[test]
fn scanning_compact_links_distinguishes_hard_soft_and_absent_names() {
    let mut hard = vec![1u8, 0x00, 4];
    hard.extend_from_slice(b"data");
    hard.extend_from_slice(&0x4000u64.to_le_bytes());

    let mut soft = vec![1u8, 0x08, 1, 4];
    soft.extend_from_slice(b"soft");
    soft.extend_from_slice(&4u16.to_le_bytes());
    soft.extend_from_slice(b"/abc");

    let header = header_of(vec![
        compact_link_info(),
        (MessageType::LINK, soft),
        (MessageType::LINK, hard),
    ]);

    assert_eq!(
        scan_compact_links(&header, OffsetWidth::Eight, LinkName::new("data").unwrap(),).unwrap(),
        Some(LinkTarget::Hard {
            object_header_address: StoredAddress::new(0x4000)
        })
    );
    assert_eq!(
        scan_compact_links(&header, OffsetWidth::Eight, LinkName::new("soft").unwrap(),).unwrap(),
        Some(LinkTarget::soft(String::from("/abc")))
    );
    assert_eq!(
        scan_compact_links(
            &header,
            OffsetWidth::Eight,
            LinkName::new("absent").unwrap(),
        )
        .unwrap(),
        None
    );
}

#[test]
fn integration_v2_groups_temperature() {
    let bytes = test_util::modern::v2_groups();
    let file_data = bytes.as_slice();
    let sb = crate::superblock::read(file_data).unwrap().superblock;
    assert!(matches!(
        sb.version(),
        SuperblockVersion::Two | SuperblockVersion::Three
    ));

    let addr = resolve_path_any(
        file_data,
        AccessMode::ReadOnly,
        &sb,
        test_widths(&sb),
        AddressContext::from_effective_base(sb.base_address),
        &ObjectPath::parse("sensors/temperature"),
    )
    .unwrap();
    let hdr = ObjectHeader::parse_with_widths(
        file_data,
        AccessMode::ReadOnly,
        addr as usize,
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    )
    .unwrap();
    let (dt, ds, dl) = extract_dataset(
        file_data,
        &hdr,
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    );
    let raw = data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
    let values = data_read::read_as_f64(&raw, &dt).unwrap();
    assert_eq!(values, vec![22.5, 23.1, 21.8]);
}

#[test]
fn integration_v2_groups_humidity() {
    let bytes = test_util::modern::v2_groups();
    let file_data = bytes.as_slice();
    let sb = crate::superblock::read(file_data).unwrap().superblock;

    let addr = resolve_path_any(
        file_data,
        AccessMode::ReadOnly,
        &sb,
        test_widths(&sb),
        AddressContext::from_effective_base(sb.base_address),
        &ObjectPath::parse("sensors/humidity"),
    )
    .unwrap();
    let hdr = ObjectHeader::parse_with_widths(
        file_data,
        AccessMode::ReadOnly,
        addr as usize,
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    )
    .unwrap();
    let (dt, ds, dl) = extract_dataset(
        file_data,
        &hdr,
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    );
    let raw = data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
    let values = data_read::read_as_i32(&raw, &dt).unwrap();
    assert_eq!(values, vec![45, 50, 55]);
}

#[test]
fn integration_v2_many_links() {
    let bytes = test_util::modern::v2_many_links();
    let file_data = bytes.as_slice();
    let sb = crate::superblock::read(file_data).unwrap().superblock;

    let addr = resolve_path_any(
        file_data,
        AccessMode::ReadOnly,
        &sb,
        test_widths(&sb),
        AddressContext::from_effective_base(sb.base_address),
        &ObjectPath::parse("dataset_015"),
    )
    .unwrap();
    let hdr = ObjectHeader::parse_with_widths(
        file_data,
        AccessMode::ReadOnly,
        addr as usize,
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    )
    .unwrap();
    let (dt, ds, dl) = extract_dataset(
        file_data,
        &hdr,
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    );
    let raw = data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
    let values = data_read::read_as_f64(&raw, &dt).unwrap();
    assert_eq!(values, vec![15.0]);
}

#[test]
fn integration_resolve_path_any_v1() {
    let bytes = test_util::legacy::two_groups();
    let file_data = bytes.as_slice();
    let sb = crate::superblock::read(file_data).unwrap().superblock;

    let addr = resolve_path_any(
        file_data,
        AccessMode::ReadOnly,
        &sb,
        test_widths(&sb),
        AddressContext::from_effective_base(sb.base_address),
        &ObjectPath::parse("group1/values"),
    )
    .unwrap();
    let hdr = ObjectHeader::parse_with_widths(
        file_data,
        AccessMode::ReadOnly,
        addr as usize,
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    )
    .unwrap();
    let (dt, ds, dl) = extract_dataset(
        file_data,
        &hdr,
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    );
    let raw = data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
    let values = data_read::read_as_i32(&raw, &dt).unwrap();
    assert_eq!(values, vec![10, 20, 30]);
}

#[test]
fn integration_resolve_path_any_v2() {
    let bytes = test_util::modern::v2_groups();
    let file_data = bytes.as_slice();
    let sb = crate::superblock::read(file_data).unwrap().superblock;

    let addr = resolve_path_any(
        file_data,
        AccessMode::ReadOnly,
        &sb,
        test_widths(&sb),
        AddressContext::from_effective_base(sb.base_address),
        &ObjectPath::parse("sensors/temperature"),
    )
    .unwrap();
    let hdr = ObjectHeader::parse_with_widths(
        file_data,
        AccessMode::ReadOnly,
        addr as usize,
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    )
    .unwrap();
    let (dt, ds, dl) = extract_dataset(
        file_data,
        &hdr,
        FormatWidths {
            offsets: sb.offset_size,
            lengths: sb.length_size,
        },
    );
    let raw = data_read::read_raw_data(file_data, &dl, &ds, &dt).unwrap();
    let values = data_read::read_as_f64(&raw, &dt).unwrap();
    assert_eq!(values, vec![22.5, 23.1, 21.8]);
}

#[test]
fn path_not_found_v2() {
    let bytes = test_util::modern::v2_groups();
    let file_data = bytes.as_slice();
    let sb = crate::superblock::read(file_data).unwrap().superblock;

    let err = resolve_path_any(
        file_data,
        AccessMode::ReadOnly,
        &sb,
        test_widths(&sb),
        AddressContext::from_effective_base(sb.base_address),
        &ObjectPath::parse("nonexistent"),
    )
    .unwrap_err();
    let ResolveError::Format(FormatError::PathNotFound(path)) = err else {
        panic!("expected PathNotFound, got {err:?}");
    };
    assert_eq!(path, "nonexistent");
}

// A nonzero userblock base makes adding the undefined stored address overflow.
#[cfg(feature = "std")]
#[test]
fn a_link_target_that_overflows_the_base_address_is_refused() {
    const UB: u64 = 512;

    let mut b = crate::writer::FileBuilder::new();
    b.with_userblock(UserblockSize::try_from(UB).unwrap());
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3])
        .with_shape(&[3]);
    let mut bytes = b.finish().unwrap();

    // Locate the root header and its one-character link independently of allocator offsets.
    let root_oh = bytes
        .windows(4)
        .position(|w| w == b"OHDR")
        .expect("a latest-format file has a version 2 root object header");
    let link = bytes
        .windows(2)
        .position(|w| w == [1, b'd'])
        .expect("the root header holds a one-character link name");
    let target = link + 2;
    let stored = u64::from_le_bytes(bytes[target..target + 8].try_into().unwrap());
    assert_eq!(
        stored + UB,
        bytes
            .windows(4)
            .enumerate()
            .filter(|(_, w)| *w == b"OHDR")
            .nth(1)
            .expect("the dataset has an object header too")
            .0 as u64,
        "the bytes after the link name must be the dataset's stored address"
    );

    // The fixture resolves before it is corrupted, through both backends:
    // what each assertion below observes is the corruption, not a file that
    // never worked.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ub.h5");
    std::fs::write(&path, &bytes).unwrap();
    for read in [
        crate::reader::File::from_bytes(bytes.clone()).unwrap(),
        crate::reader::File::open_streaming(&path).unwrap(),
    ] {
        assert_eq!(read.dataset("d").unwrap().read::<i32>().unwrap(), [1, 2, 3]);
    }

    // Repair the checksum so parsing reaches the overflowing link target.
    bytes[target..target + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    let (chunk_at, chunk_len) = crate::edit::object_header::read_oh_chunks(
        &crate::source::BytesSource::new(&bytes[..]),
        root_oh as u64,
        AddressContext::from_effective_base(BaseAddress::new(0)),
    )
    .expect("the root header is readable")[0]
        .span;
    let end = (chunk_at + chunk_len).to_usize().unwrap() - 4;
    let repaired = crate::checksum::jenkins_lookup3(&bytes[chunk_at.to_usize().unwrap()..end]);
    bytes[end..end + 4].copy_from_slice(&repaired.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

    // Each backend reaches `ChildLookup::of` through its own lookup
    // (`find_child_target` and `find_child_target_from_source`), so both
    // are asserted.
    for (backend, file) in [
        ("buffered", crate::reader::File::from_bytes(bytes).unwrap()),
        (
            "streaming",
            crate::reader::File::open_streaming(&path).unwrap(),
        ),
    ] {
        let err = file.dataset("d").unwrap_err();
        assert!(
            matches!(
                err,
                crate::Error::Format(FormatError::OffsetOverflow {
                    offset: u64::MAX,
                    length: UB,
                })
            ),
            "{backend}: expected the base-address addition to be refused, got {err:?}"
        );
    }
}
