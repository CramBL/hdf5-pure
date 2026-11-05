use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util_hdf5::file;

use crate::address::AddressContext;
use crate::attribute_info::AttributeInfoMessage;
use crate::edit::WriteEngine;
use crate::edit::context;
use crate::edit::group::LinkPatches;
use crate::edit::object_header;
use crate::link_message::{LinkMessage, LinkTarget};
use crate::message_type::MessageType;
use crate::source::BytesSource;
use crate::superblock;
use crate::width::{LengthWidth, OffsetWidth};

#[rstest]
#[case::two_two(OffsetWidth::Two, LengthWidth::Two)]
#[case::two_four(OffsetWidth::Two, LengthWidth::Four)]
#[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
#[case::four_two(OffsetWidth::Four, LengthWidth::Two)]
#[case::four_four(OffsetWidth::Four, LengthWidth::Four)]
#[case::four_eight(OffsetWidth::Four, LengthWidth::Eight)]
#[case::eight_two(OffsetWidth::Eight, LengthWidth::Two)]
#[case::eight_four(OffsetWidth::Eight, LengthWidth::Four)]
#[case::eight_eight(OffsetWidth::Eight, LengthWidth::Eight)]
fn native_header_reconstruction_preserves_independent_widths(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[values(0, 512)] base: u64,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("header.h5");
    let expected_sizes = {
        let native = file::libhdf5_create_with_widths(&path, offsets, lengths, base);
        native
            .new_dataset_builder()
            .with_data(&[11i32, 13, 17])
            .create("data")
            .unwrap();
        native.link_hard("data", "alias").unwrap();
        for index in 0..7 {
            native
                .new_attr::<u8>()
                .shape([512])
                .create(format!("attribute_{index}").as_str())
                .unwrap()
                .write_raw(&vec![u8::try_from(index).unwrap(); 512])
                .unwrap();
        }
        native.create_plist().unwrap().get_sizes().unwrap()
    };
    assert_eq!(
        usize::from(expected_sizes.sizeof_addr),
        usize::from(offsets.get())
    );
    assert_eq!(
        usize::from(expected_sizes.sizeof_size),
        usize::from(lengths.get())
    );
    let mut bytes = std::fs::read(&path).unwrap();
    let mut block = {
        let source = BytesSource::new(bytes.as_slice());
        superblock::read(&source).unwrap()
    };
    let widths = block.widths;
    let addresses = AddressContext::from_effective_base(block.superblock.base_address);
    let root = addresses
        .resolve(block.superblock.root_group_address)
        .unwrap()
        .get();
    let encoded = {
        let source = BytesSource::new(bytes.as_slice());
        let chunks =
            object_header::read_oh_chunks(&source, PhysicalAddress::new(root), addresses, widths)
                .unwrap();
        assert!(
            chunks.len() > 1,
            "native root has no continuation: {widths:?}"
        );
        let mut region =
            WriteEngine::gather_oh_messages(&source, PhysicalAddress::new(root), addresses, widths)
                .unwrap();
        assert_eq!(region.widths(), widths);
        let mut position = 0;
        let mut data = None;
        let mut attributes = 0;
        let mut retained = Vec::new();
        while let Some(record) = region.layout().next_message(&region, position).unwrap() {
            if record.msg_type == MessageType::LINK {
                let link = LinkMessage::parse_with_offset_width(record.body, offsets).unwrap();
                if link.name == "data" {
                    let LinkTarget::Hard {
                        object_header_address,
                    } = link.link_target
                    else {
                        panic!("data is not a hard link");
                    };
                    data = Some(object_header_address);
                }
            }
            if record.msg_type == MessageType::ATTRIBUTE_INFO {
                let info =
                    AttributeInfoMessage::parse_with_offset_width(record.body, offsets).unwrap();
                assert_eq!(info.fractal_heap_address, None);
                assert_eq!(info.btree_name_index_address, None);
                assert_eq!(info.btree_creation_order_address, None);
            } else {
                retained.extend_from_slice(&region[position..record.body_range.end]);
            }
            attributes += usize::from(record.msg_type == MessageType::ATTRIBUTE);
            position = record.body_range.end;
        }
        assert_eq!(attributes, 7);
        region = region.with_bytes(retained);
        let data = data.unwrap();
        let mut patches = LinkPatches::new(&mut region, ["data", "alias"]);
        patches.patch_next(data).unwrap();
        patches.patch_next(data).unwrap();
        object_header::build_v2_object_header(&region).unwrap()
    };
    let start = u64::try_from(bytes.len()).unwrap();
    let allocation = FileRange::new(
        PhysicalAddress::new(start),
        u64::try_from(encoded.len()).unwrap(),
    )
    .unwrap();
    context::require_range(offsets, addresses, allocation).unwrap();
    block.superblock.root_group_address = addresses.relative(allocation.start()).unwrap();
    block.superblock.eof_address = allocation.end().get();
    bytes.extend(encoded);
    let encoded = hdf5_pure_format::__private::serialize_superblock(&block.superblock).unwrap();
    let signature = usize::try_from(base).unwrap();
    bytes[signature..signature + encoded.len()].copy_from_slice(&encoded);
    std::fs::write(&path, bytes).unwrap();
    let native = file::libhdf5_open(&path);
    assert_eq!(
        native.create_plist().unwrap().get_sizes().unwrap(),
        expected_sizes
    );
    assert_eq!(
        native.dataset("data").unwrap().read_raw::<i32>().unwrap(),
        [11, 13, 17]
    );
    assert_eq!(
        native.dataset("alias").unwrap().read_raw::<i32>().unwrap(),
        [11, 13, 17]
    );
    for index in 0..7 {
        assert_eq!(
            native
                .attr(&format!("attribute_{index}"))
                .unwrap()
                .read_raw::<u8>()
                .unwrap(),
            vec![u8::try_from(index).unwrap(); 512]
        );
    }
    drop(native);
    let source = BytesSource::new(std::fs::read(&path).unwrap());
    let reopened = superblock::read(&source).unwrap();
    assert_eq!(reopened.widths, widths);
    assert_eq!(
        reopened.superblock.root_group_address,
        block.superblock.root_group_address
    );
    assert_eq!(reopened.superblock.eof_address, allocation.end().get());
    directory.close().unwrap();
}
