#![cfg(feature = "__hdf5-1.10")]

use hdf5::file::LibraryVersion;
use hdf5::plist::group_create::AttrCreationOrder;
use hdf5::{IndexType, IterationOrder};
use hdf5_pure::{Datatype, DatatypeByteOrder, File, FixedPointType, MessageType};
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_format::__private::{
    AttributeMessage, Dataspace, DataspaceType, DatatypeLocation, DenseAttrCreationOrder,
    LinkInfoMessage, ObjectHeaderPrefix, ObjectHeaderWriter, PreparedDenseAttrs,
};
use rstest::rstest;

#[rstest]
#[case::empty(vec![])]
#[case::managed(vec![64; 80])]
#[case::huge(vec![10_000])]
#[case::huge_index(vec![9_000; 40])]
#[case::mixed(vec![2, 10_000, 8, 12_000])]
fn libhdf5_reads_independently_placed_dense_attributes(
    #[case] sizes: Vec<usize>,
    #[values(
        AttrCreationOrder::Untracked,
        AttrCreationOrder::Tracked,
        AttrCreationOrder::Indexed
    )]
    policy: AttrCreationOrder,
    #[values(0, 512)] userblock: u64,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("placed.h5");
    let native = hdf5::File::with_options()
        .with_fapl(|properties| {
            properties.libver_bounds(LibraryVersion::V18, LibraryVersion::latest())
        })
        .with_fcpl(|properties| {
            properties
                .attr_phase_change(0, 0)
                .attr_creation_order(policy)
                .userblock(userblock)
        })
        .create(&path)
        .unwrap();
    let expected: Vec<_> = sizes
        .iter()
        .enumerate()
        .map(|(index, size)| {
            let values: Vec<_> = (0..*size)
                .map(|position| index as i64 * 100_000 + position as i64)
                .collect();
            let attribute = native
                .new_attr::<i64>()
                .shape(*size)
                .create(format!("attribute_{index}").as_str())
                .unwrap();
            attribute.write_raw(&values).unwrap();
            assert_eq!(attribute.read_raw::<i64>().unwrap(), values);
            values
        })
        .collect();
    native.close().unwrap();
    let pure = File::open(&path).unwrap();
    let mut superblock = pure.superblock().clone();
    drop(pure);
    let attrs: Vec<_> = expected
        .iter()
        .enumerate()
        .map(|(index, values)| AttributeMessage {
            name: format!("attribute_{index}"),
            datatype: Datatype::FixedPoint(
                FixedPointType::new(8, DatatypeByteOrder::LittleEndian, true, 0, 64).unwrap(),
            ),
            dataspace: Dataspace {
                space_type: DataspaceType::Simple,
                rank: 1,
                dimensions: vec![values.len() as u64],
                max_dimensions: None,
            },
            raw_data: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            datatype_location: DatatypeLocation::Inline,
        })
        .collect();
    let tracked = policy != AttrCreationOrder::Untracked;
    let creation = if tracked {
        DenseAttrCreationOrder::Tracked {
            indices: (0..attrs.len()).map(|index| index as u16 * 3 + 2).collect(),
            max: attrs.len() as u16 * 3 + 9,
            indexed: policy == AttrCreationOrder::Indexed,
        }
    } else {
        DenseAttrCreationOrder::Untracked
    };
    let prepared = PreparedDenseAttrs::new(&attrs, creation, []).unwrap();
    let plan = prepared
        .finalize(
            |reference| panic!("unexpected reference {reference:?}"),
            |attribute| panic!("unexpected payload {attribute}"),
        )
        .unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let mut cursor = bytes.len() as u64;
    let lengths: Vec<_> = plan.block_lengths().collect();
    let mut addresses = vec![StoredAddress::ZERO; lengths.len()];
    for index in (0..lengths.len()).rev() {
        let length = lengths[index];
        if length >= 512 || cursor % 512 + length > 512 {
            cursor = cursor.div_ceil(512) * 512;
        }
        addresses[index] = StoredAddress::new(cursor - userblock);
        cursor += length;
    }
    let placement = plan.place(&addresses).unwrap();
    let mut emitted = Vec::new();
    for ((address, length), block) in placement
        .blocks()
        .zip(placement.serialize_blocks().unwrap())
    {
        let block = block.unwrap();
        let (stored, data) = (block.address, block.bytes);
        assert_eq!((stored, data.len() as u64), (address, length));
        let at = usize::try_from(address.get() + userblock).unwrap();
        if length >= 512 {
            assert_eq!(at % 512, 0);
        } else {
            assert!(at as u64 % 512 + length <= 512);
        }
        bytes.resize(bytes.len().max(at + data.len()), 0);
        bytes[at..at + data.len()].copy_from_slice(&data);
        emitted.push(stored);
    }
    assert_eq!(emitted, addresses);
    assert_eq!(
        placement.attribute_info().max_creation_index,
        tracked.then_some(attrs.len() as u16 * 3 + 9)
    );
    let root_at = usize::try_from(superblock.root_group_address.get() + userblock).unwrap();
    let prefix = ObjectHeaderPrefix::parse(&bytes[root_at..]).unwrap().prefix;
    let mut header = ObjectHeaderWriter::with_prefix(prefix);
    header.add_message(
        MessageType::ATTRIBUTE_INFO,
        placement.attribute_info().serialize(superblock.offset_size),
    );
    header.add_message(MessageType::GROUP_INFO, vec![0, 0]);
    header.add_message(
        MessageType::LINK_INFO,
        LinkInfoMessage {
            max_creation_order: None,
            fractal_heap_address: None,
            btree_name_index_address: None,
            btree_creation_order_address: None,
        }
        .serialize(superblock.offset_size),
    );
    superblock.root_group_address = StoredAddress::new(bytes.len() as u64 - userblock);
    bytes.extend(header.serialize().unwrap());
    superblock.eof_address = bytes.len() as u64;
    let encoded = hdf5_pure_format::__private::serialize_superblock(&superblock).unwrap();
    let at = usize::try_from(userblock).unwrap();
    bytes[at..at + encoded.len()].copy_from_slice(&encoded);
    std::fs::write(&path, bytes).unwrap();
    let native = hdf5::File::open(&path).unwrap();
    let mut names = native.attr_names().unwrap();
    names.sort();
    let mut expected_names: Vec<_> = attrs
        .iter()
        .map(|attribute| attribute.name.clone())
        .collect();
    expected_names.sort();
    assert_eq!(names, expected_names);
    for (index, values) in expected.iter().enumerate() {
        assert_eq!(
            native
                .attr(&attrs[index].name)
                .unwrap()
                .read_raw::<i64>()
                .unwrap(),
            *values
        );
        if tracked {
            let attribute = native
                .attr_by_index(
                    IndexType::CreationOrder,
                    IterationOrder::Increasing,
                    index as u64,
                )
                .unwrap();
            assert_eq!(attribute.name(), attrs[index].name);
            assert_eq!(
                native.attr_info(&attrs[index].name).unwrap().creation_order,
                Some(index as u32 * 3 + 2)
            );
        }
    }
    native.close().unwrap();
}
