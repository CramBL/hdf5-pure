use alloc::vec;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::FileRegion;
use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;

use crate::btree_v2::BTreeV2Header;
use crate::btree_v2::BTreeV2Layout;
use crate::btree_v2::BTreeV2Node;
use crate::datatype::CharacterSet;
use crate::fractal_heap::FractalHeapIdKind;
use crate::fractal_heap::FractalHeapIdLayout;
use crate::fractal_heap_block::FractalHeapBlock;
use crate::fractal_heap_block::FractalHeapBlockLayout;
use crate::link_storage::OwnedLink;
use crate::link_storage::OwnedLinkTarget;

use super::*;

#[rstest]
#[case::untracked(0)]
#[case::tracked(1)]
#[case::indexed(2)]
fn planned_links_parse_at_each_supported_address_width(
    #[case] policy: u8,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] offsets: OffsetWidth,
    #[values(LengthWidth::Four, LengthWidth::Eight)] lengths: LengthWidth,
    #[values(0, 512)] physical_base: u64,
) {
    let storage = storage(12, policy, offsets, 0);
    let mut expected = storage.links().to_vec();
    let plan = DenseLinkPlan::new(storage.clone(), lengths).unwrap();
    let base = StoredAddress::new(1024);
    let image = plan.serialize(base).unwrap();
    assert_eq!(image.bytes.len() as u64, plan.serialized_len());
    assert_eq!(image.info, plan.link_info(base).unwrap());
    assert_eq!(
        image.info.max_creation_order,
        storage.info().max_creation_order
    );
    assert_eq!(
        image.info.btree_creation_order_address.is_some(),
        policy == 2
    );
    let (mut links, _, _) = parse(&image, base, physical_base, offsets, lengths);
    links.sort_by(|a, b| a.name().cmp(b.name()));
    expected.sort_by(|a, b| a.name().cmp(b.name()));
    assert_eq!(links, expected);
}

#[rstest]
#[case::empty(0, 0, 0)]
#[case::leaf(10, 0, 0)]
#[case::internal(150, 0, 1)]
#[case::deep(3000, 1200, 2)]
fn heap_blocks_indexes_and_statistics_conform_to_the_parser(
    #[case] count: usize,
    #[case] name_len: usize,
    #[case] minimum_depth: u16,
) {
    let storage = storage(count, 2, OffsetWidth::Eight, name_len);
    let plan = DenseLinkPlan::new(storage.clone(), LengthWidth::Eight).unwrap();
    let base = StoredAddress::new(0x1_0000_1000);
    let image = plan.serialize(base).unwrap();
    let (links, direct, indirect) =
        parse(&image, base, 512, OffsetWidth::Eight, LengthWidth::Eight);
    assert_eq!(links.len(), count);
    assert!(direct >= 1);
    assert!(count != 3000 || indirect > 1);
    let mut expected = storage.links().to_vec();
    expected.sort_by_key(|link| (checksum::jenkins_lookup3(link.name()), link.name().to_vec()));
    assert_eq!(links, expected);
    for address in [
        image.info.btree_name_index_address.unwrap(),
        image.info.btree_creation_order_address.unwrap().unwrap(),
    ] {
        let header = BTreeV2Header::parse(
            &image.bytes,
            usize::try_from(address.get() - base.get()).unwrap(),
            8,
            8,
        )
        .unwrap();
        assert!(header.depth >= minimum_depth);
        assert_eq!(header.total_records, count as u64);
        if count == 0 {
            assert_eq!(
                header.root_node_address,
                StoredAddress::undefined(OffsetWidth::Eight)
            );
            assert_eq!(header.num_records_in_root, 0);
        }
    }
    let heap_header =
        FractalHeapHeader::parse_bytes(&image.bytes, OffsetWidth::Eight, LengthWidth::Eight)
            .unwrap();
    assert_eq!(heap_header.heap_id_length, HEAP_ID_LEN);
    assert_eq!(heap_header.table_width, GROUP_HEAP_TABLE_WIDTH);
    assert_eq!(
        heap_header.starting_block_size,
        GROUP_HEAP_STARTING_BLOCK_SIZE
    );
    assert_eq!(
        heap_header.max_direct_block_size,
        GROUP_HEAP_MAX_DIRECT_BLOCK_SIZE
    );
    assert_eq!(heap_header.max_heap_size, GROUP_HEAP_MAX_HEAP_SIZE_BITS);
    assert_eq!(heap_header.managed_objects_count, count as u64);
    assert_eq!(heap_header.huge_objects_count, 0);
    assert_eq!(heap_header.tiny_objects_count, 0);
    assert_eq!(heap_header.managed_space, plan.heap.managed_space());
    assert_eq!(
        heap_header.allocated_managed_space,
        plan.heap.allocated_space()
    );
    assert_eq!(
        heap_header.free_space_in_managed_blocks,
        plan.heap.free_space()
    );
    assert_eq!(
        heap_header.direct_block_allocation_iterator_offset,
        plan.heap.allocation_iterator()
    );
}

#[test]
fn real_name_hash_collisions_are_sorted_by_raw_name_bytes() {
    let names = [b"k69209".to_vec(), b"k155448".to_vec()];
    assert_eq!(
        checksum::jenkins_lookup3(&names[0]),
        checksum::jenkins_lookup3(&names[1])
    );
    let info = LinkInfoMessage {
        max_creation_order: None,
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: None,
    };
    let links = names
        .into_iter()
        .map(|name| {
            OwnedLink::new(
                name,
                CharacterSet::Ascii,
                OwnedLinkTarget::soft(b"/target".to_vec()),
                None,
            )
            .unwrap()
        })
        .collect();
    let storage = GroupLinkStorage::new(&info, links, OffsetWidth::Eight).unwrap();
    let plan = DenseLinkPlan::new(storage, LengthWidth::Eight).unwrap();
    let image = plan.serialize(StoredAddress::new(0)).unwrap();
    let (links, _, _) = parse(
        &image,
        StoredAddress::new(0),
        0,
        OffsetWidth::Eight,
        LengthWidth::Eight,
    );
    assert_eq!(
        links.iter().map(OwnedLink::name).collect::<Vec<_>>(),
        vec![b"k155448".as_slice(), b"k69209".as_slice()]
    );
}

#[rstest]
#[case::undefined(OffsetWidth::Two, u64::from(u16::MAX), FormatError::DefinedAddressIsUndefined { address: u64::from(u16::MAX), offset_size: 2 })]
#[case::too_wide(OffsetWidth::Two, 65_536, FormatError::AddressTooLarge { address: 65_536, offset_size: 2 })]
#[case::overflow(OffsetWidth::Eight, u64::MAX - 1, FormatError::OffsetOverflow { offset: u64::MAX - 1, length: 0 })]
fn invalid_output_addresses_are_rejected_before_emission(
    #[case] offsets: OffsetWidth,
    #[case] base: u64,
    #[case] mut expected: FormatError,
) {
    let plan = DenseLinkPlan::new(storage(1, 2, offsets, 0), LengthWidth::Eight).unwrap();
    if let FormatError::OffsetOverflow { length, .. } = &mut expected {
        *length = plan.serialized_len();
    }
    assert_eq!(
        plan.link_info(StoredAddress::new(base)).unwrap_err(),
        DenseLinkPlanError::Format(expected)
    );
}

#[test]
fn output_extent_is_checked_at_the_address_field_boundary() {
    let plan = DenseLinkPlan::new(storage(1, 0, OffsetWidth::Two, 0), LengthWidth::Eight).unwrap();
    let base = u64::from(u16::MAX) - plan.serialized_len() + 1;
    assert_eq!(
        plan.link_info(StoredAddress::new(base)).unwrap_err(),
        DenseLinkPlanError::Format(FormatError::AddressTooLarge {
            address: u64::from(u16::MAX) + 1,
            offset_size: 2
        })
    );
    let plan = DenseLinkPlan::new(storage(0, 2, OffsetWidth::Two, 0), LengthWidth::Eight).unwrap();
    let base = StoredAddress::new(u64::from(u16::MAX) - plan.serialized_len());
    assert_eq!(
        plan.serialize(base).unwrap().bytes.len() as u64,
        plan.serialized_len()
    );
}

#[test]
fn two_byte_lengths_reject_the_group_heap_maximum_direct_block_size() {
    let error = DenseLinkPlan::new(storage(0, 0, OffsetWidth::Eight, 0), LengthWidth::Two)
        .err()
        .unwrap();
    assert_eq!(
        error,
        DenseLinkPlanError::Format(FormatError::LengthTooLarge {
            length: GROUP_HEAP_MAX_DIRECT_BLOCK_SIZE,
            length_size: 2
        })
    );
}

#[test]
fn oversized_link_messages_are_rejected_during_planning() {
    let storage = storage(1, 0, OffsetWidth::Eight, 5000);
    let size = storage.links()[0].encoded_len(OffsetWidth::Eight).unwrap() as u64;
    let error = DenseLinkPlan::new(storage, LengthWidth::Eight)
        .err()
        .unwrap();
    assert_eq!(
        error,
        DenseLinkPlanError::Heap(ManagedHeapPlanError::ObjectSize {
            size,
            maximum: GROUP_HEAP_MAX_MANAGED_OBJECT
        })
    );
}

#[test]
fn creation_records_preserve_all_eight_bytes_and_preexisting_addresses_are_replaced() {
    let next = i64::MAX as u64;
    let info = LinkInfoMessage {
        max_creation_order: Some(next),
        fractal_heap_address: Some(StoredAddress::new(111)),
        btree_name_index_address: Some(StoredAddress::new(222)),
        btree_creation_order_address: Some(Some(StoredAddress::new(333))),
    };
    let links = [next - 1, 0x0001_0203_0405_0607]
        .into_iter()
        .enumerate()
        .map(|(index, order)| {
            OwnedLink::new(
                vec![b'a' + u8::try_from(index).unwrap()],
                CharacterSet::Ascii,
                OwnedLinkTarget::Hard {
                    object_header_address: StoredAddress::new(42),
                },
                Some(order),
            )
            .unwrap()
        })
        .collect();
    let storage = GroupLinkStorage::new(&info, links, OffsetWidth::Eight).unwrap();
    let plan = DenseLinkPlan::new(storage, LengthWidth::Eight).unwrap();
    let image = plan.serialize(StoredAddress::new(4096)).unwrap();
    assert_eq!(image.info.max_creation_order, Some(next));
    assert_eq!(
        image.info.fractal_heap_address,
        Some(StoredAddress::new(4096))
    );
    assert_ne!(
        image.info.btree_name_index_address,
        info.btree_name_index_address
    );
    assert_ne!(
        image.info.btree_creation_order_address,
        info.btree_creation_order_address
    );
    let (links, _, _) = parse(
        &image,
        StoredAddress::new(4096),
        512,
        OffsetWidth::Eight,
        LengthWidth::Eight,
    );
    let mut recovered = links
        .iter()
        .map(|link| (link.name().to_vec(), link.creation_order()))
        .collect::<Vec<_>>();
    recovered.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        recovered,
        vec![
            (b"a".to_vec(), Some(next - 1)),
            (b"b".to_vec(), Some(0x0001_0203_0405_0607))
        ]
    );
}

#[test]
fn index_layout_arithmetic_reports_overflow() {
    let mut cursor = u64::MAX;
    let result = PlannedIndex::new(
        FormatWidths {
            offsets: OffsetWidth::Eight,
            lengths: LengthWidth::Eight,
        },
        BTREE_V2_LINK_NAME,
        0,
        NAME_RECORD_LEN,
        Vec::new(),
        &mut cursor,
    );
    assert_eq!(result.err().unwrap(), DenseLinkPlanError::LengthOverflow);
}

#[test]
fn maximum_size_managed_hard_link_round_trips() {
    let target = OwnedLinkTarget::Hard {
        object_header_address: StoredAddress::new(42),
    };
    let link = OwnedLink::new(vec![b'x'], CharacterSet::Ascii, target.clone(), None).unwrap();
    let prefix_and_target = link.encoded_len(OffsetWidth::Eight).unwrap() - link.name().len();
    let name_len = usize::try_from(GROUP_HEAP_MAX_MANAGED_OBJECT).unwrap() - prefix_and_target - 1;
    let link = OwnedLink::new(vec![b'x'; name_len], CharacterSet::Ascii, target, None).unwrap();
    assert_eq!(
        link.encoded_len(OffsetWidth::Eight).unwrap() as u64,
        GROUP_HEAP_MAX_MANAGED_OBJECT
    );
    let info = LinkInfoMessage {
        max_creation_order: None,
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: None,
    };
    let storage = GroupLinkStorage::new(&info, vec![link.clone()], OffsetWidth::Eight).unwrap();
    let plan = DenseLinkPlan::new(storage, LengthWidth::Eight).unwrap();
    let image = plan.serialize(StoredAddress::new(0)).unwrap();
    let (links, _, _) = parse(
        &image,
        StoredAddress::new(0),
        0,
        OffsetWidth::Eight,
        LengthWidth::Eight,
    );
    assert_eq!(links, vec![link]);
}

fn storage(count: usize, policy: u8, offsets: OffsetWidth, name_len: usize) -> GroupLinkStorage {
    let info = LinkInfoMessage {
        max_creation_order: (policy > 0).then_some(count as u64 * 3 + 10),
        fractal_heap_address: None,
        btree_name_index_address: None,
        btree_creation_order_address: (policy == 2).then_some(None),
    };
    let links = (0..count)
        .rev()
        .map(|index| {
            let mut name = format!("link_{index:06}").into_bytes();
            if name_len > name.len() {
                name.resize(name_len, b'x');
            }
            if index % 4 == 1 {
                name.push(0xFF);
            }
            let (charset, target) = match index % 4 {
                0 => (
                    CharacterSet::Ascii,
                    OwnedLinkTarget::Hard {
                        object_header_address: StoredAddress::new(42),
                    },
                ),
                1 => (CharacterSet::Ascii, OwnedLinkTarget::soft(vec![b'/', 0xFF])),
                2 => (
                    CharacterSet::Utf8,
                    OwnedLinkTarget::external(vec![0xFE], b"/object".to_vec()),
                ),
                _ => (
                    CharacterSet::Utf8,
                    OwnedLinkTarget::Hard {
                        object_header_address: StoredAddress::new(84),
                    },
                ),
            };
            OwnedLink::new(
                name,
                charset,
                target,
                (policy > 0).then_some(index as u64 * 3 + 2),
            )
            .unwrap()
        })
        .collect();
    GroupLinkStorage::new(&info, links, offsets).unwrap()
}

fn parse(
    image: &DenseLinkImage,
    base: StoredAddress,
    physical_base: u64,
    offsets: OffsetWidth,
    lengths: LengthWidth,
) -> (Vec<OwnedLink>, usize, usize) {
    let header = FractalHeapHeader::parse_bytes(&image.bytes, offsets, lengths).unwrap();
    let heap = FractalHeapBlockLayout::new(&header, offsets, lengths, base).unwrap();
    let region = FileRegion::new(
        FileRange::new(
            PhysicalAddress::new(physical_base + base.get()),
            image.bytes.len() as u64,
        )
        .unwrap(),
        &image.bytes,
    )
    .unwrap();
    let mut pending = heap.root().unwrap().into_iter().collect::<Vec<_>>();
    let mut direct = Vec::new();
    let mut indirect = 0;
    while let Some(block) = pending.pop() {
        match block {
            FractalHeapBlock::Direct(block) => {
                let reference = block
                    .at(PhysicalAddress::new(physical_base + block.address().get()))
                    .unwrap();
                reference.parse_checked(region).unwrap();
                direct.push(block);
            }
            FractalHeapBlock::Indirect(block) => {
                indirect += 1;
                let reference = block
                    .at(PhysicalAddress::new(physical_base + block.address().get()))
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
    let id_layout = FractalHeapIdLayout::new(&header, offsets, lengths);
    let names = records(
        &image.bytes,
        base,
        image.info.btree_name_index_address.unwrap(),
        offsets,
        lengths,
    );
    let mut links = Vec::new();
    let mut ids = Vec::new();
    for record in names {
        let (hash, id) = record.split_at(usize::from(NAME_RECORD_LEN - HEAP_ID_LEN));
        let FractalHeapIdKind::Managed {
            heap_offset,
            object_length,
        } = id_layout.parse(id).unwrap().kind()
        else {
            panic!("expected a managed heap ID");
        };
        let block = direct
            .iter()
            .find(|block| {
                heap_offset >= block.heap_offset()
                    && heap_offset < block.heap_offset() + block.decoded_size()
            })
            .unwrap();
        let reference = block
            .at(PhysicalAddress::new(physical_base + block.address().get()))
            .unwrap();
        let object = reference
            .parse_object(region, heap_offset, object_length)
            .unwrap();
        let link = OwnedLink::parse(object.parse(region).unwrap(), offsets).unwrap();
        assert_eq!(hash, checksum::jenkins_lookup3(link.name()).to_le_bytes());
        ids.push((link.creation_order(), id.to_vec()));
        links.push(link);
    }
    if let Some(Some(address)) = image.info.btree_creation_order_address {
        let creation = records(&image.bytes, base, address, offsets, lengths);
        ids.sort_by_key(|&(order, _)| order);
        assert_eq!(creation.len(), ids.len());
        for (record, (order, id)) in creation.into_iter().zip(ids) {
            let (index, encoded_id) = record.split_at(core::mem::size_of::<u64>());
            assert_eq!(index, order.unwrap().to_le_bytes());
            assert_eq!(encoded_id, id);
        }
    }
    (links, direct.len(), indirect)
}

fn records(
    bytes: &[u8],
    base: StoredAddress,
    address: StoredAddress,
    offsets: OffsetWidth,
    lengths: LengthWidth,
) -> Vec<Vec<u8>> {
    let header = BTreeV2Header::parse(
        bytes,
        usize::try_from(address.get() - base.get()).unwrap(),
        offsets.get(),
        lengths.get(),
    )
    .unwrap();
    let layout = BTreeV2Layout::new(&header, offsets).unwrap();
    let mut records = Vec::new();
    if let Some(root) = layout.root() {
        collect(root, bytes, base, &mut records);
    }
    records
}

fn collect(node: BTreeV2Node<'_>, bytes: &[u8], base: StoredAddress, records: &mut Vec<Vec<u8>>) {
    let view = node
        .parse(&bytes[usize::try_from(node.address().get() - base.get()).unwrap()..])
        .unwrap();
    for index in 0..=view.record_count() {
        if let Some(child) = view.child(index).unwrap() {
            collect(child, bytes, base, records);
        }
        if index < view.record_count() {
            records.push(view.record(index).unwrap().to_vec());
        }
    }
}
