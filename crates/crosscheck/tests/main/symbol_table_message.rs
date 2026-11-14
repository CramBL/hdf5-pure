#![cfg(feature = "hdf5")]

use hdf5::plist::file_create::{Sizeof, SizeofInfo};
use hdf5_pure::File;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::{MessageType, OffsetWidth, StoredAddress};
use hdf5_pure_format::__private::{AccessMode, FormatWidths, ObjectHeader, SymbolTableMessage};
use rstest::rstest;
use test_util::bytes;
use test_util_hdf5::__private::c_object;

#[rstest]
#[case::two(OffsetWidth::Two, Sizeof::Bytes2)]
#[case::four(OffsetWidth::Four, Sizeof::Bytes4)]
#[case::eight(OffsetWidth::Eight, Sizeof::Bytes8)]
fn symbol_table_messages_preserve_native_addresses_and_bytes(
    #[case] width: OffsetWidth,
    #[case] native_width: Sizeof,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("symbol-table.h5");
    let native = hdf5::File::with_options()
        .with_fapl(|properties| properties.libver_earliest())
        .with_fcpl(|properties| {
            properties.sizes(SizeofInfo {
                sizeof_addr: native_width,
                sizeof_size: Sizeof::Bytes8,
            })
        })
        .create(&path)
        .unwrap();
    let group = native.create_group("links").unwrap();
    let dataset = group
        .new_dataset::<u32>()
        .shape([2])
        .create("values")
        .unwrap();
    dataset.write_raw(&[7, 11]).unwrap();
    assert_eq!(group.member_names().unwrap(), ["values"]);
    assert_eq!(dataset.read_raw::<u32>().unwrap(), [7, 11]);
    let address = c_object::header_address(&native, "links");
    drop(dataset);
    drop(group);
    native.close().unwrap();

    let image = std::fs::read(&path).unwrap();
    let pure = File::from_bytes(image.clone()).unwrap();
    assert_eq!(pure.superblock().offset_size, width);
    let header = ObjectHeader::parse_with_base_and_widths(
        &image,
        AccessMode::ReadOnly,
        usize::try_from(address).unwrap(),
        FormatWidths {
            offsets: width,
            lengths: pure.superblock().length_size,
        },
        pure.superblock().base_address,
    )
    .unwrap();
    assert_eq!(header.version, 1);
    let body = &header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::SYMBOL_TABLE)
        .unwrap()
        .data;
    let field_size = usize::from(width.get());
    let message = SymbolTableMessage::parse_with_offset_width(body, width).unwrap();
    assert_eq!(
        message,
        SymbolTableMessage {
            btree_address: StoredAddress::new(bytes::uint_at(body, 0, field_size)),
            local_heap_address: StoredAddress::new(bytes::uint_at(body, field_size, field_size)),
        },
    );
    let mut output = body.clone();
    assert_eq!(
        message.write_into(&mut output, width).unwrap(),
        2 * field_size
    );
    assert_eq!(output, *body);
    let group = pure.group("links").unwrap();
    assert_eq!(group.datasets().unwrap(), ["values"]);
    assert_eq!(
        group.dataset("values").unwrap().read::<u32>().unwrap(),
        [7, 11]
    );
    drop(group);
    drop(pure);
    directory.close().unwrap();
}
