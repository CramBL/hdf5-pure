//! Native read-write access to version 2 headers with owned padding.

#![cfg(feature = "hdf5")]

use std::fs;

use hdf5_pure::AttrValue;
use hdf5_pure::File;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_format::__private::{
    FormatWidths, MessageRecordLayout, ObjectHeaderAllocation, ObjectHeaderPrefix,
};
use rstest::rstest;
use test_util::object_header::v2::{self as header, HeaderFlags};
use test_util::object_header::{Message, MessageType};
use test_util::superblock::v2::Superblock;
use test_util::widths::Widths;

#[rstest]
#[case::plain(HeaderFlags::default())]
#[case::tracked(HeaderFlags::TRACKS_CREATION_ORDER | HeaderFlags::INDEXES_CREATION_ORDER)]
fn owned_padding_survives_native_read_write_flush(
    #[case] flags: HeaderFlags,
    #[values(
        0, 1, 2, 3, 4, 5, 17, 65_539, 65_540, 65_541, 65_542, 65_543, 65_544, 65_545, 131_123,
        1_048_576
    )]
    extra: usize,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("padded.h5");
    let mut link_info = vec![LINK_INFO_VERSION, NO_FLAGS];
    link_info.extend_from_slice(&u64::MAX.to_le_bytes());
    link_info.extend_from_slice(&u64::MAX.to_le_bytes());
    let records = header::message_records(
        &[
            Message::new(MessageType::GROUP_INFO, &GROUP_INFO_DEFAULT),
            Message::new(MessageType::LINK_INFO, &link_info),
        ],
        flags,
    );
    let original = header::Header::new()
        .flags(flags)
        .message(Message::new(MessageType::GROUP_INFO, &GROUP_INFO_DEFAULT))
        .message(Message::new(MessageType::LINK_INFO, &link_info))
        .build();
    let allocation = u64::try_from(original.len() + extra).unwrap();
    let plan = ObjectHeaderAllocation::new(
        ObjectHeaderPrefix::with_layout(MessageRecordLayout::from_header_flags(flags.0)),
        FormatWidths::from_sizes(8, 8).unwrap(),
        StoredAddress::new(ROOT_ADDRESS),
        allocation,
        &records,
    )
    .unwrap();
    let mut bytes = Superblock::new(Widths::EIGHT)
        .root_header_address(ROOT_ADDRESS)
        .eof_address(ROOT_ADDRESS + allocation)
        .build();
    bytes.resize(usize::try_from(ROOT_ADDRESS).unwrap(), 0);
    bytes.extend(plan.encode());
    assert_eq!(
        u64::try_from(bytes.len()).unwrap(),
        ROOT_ADDRESS + allocation
    );
    fs::write(&path, bytes).unwrap();
    let native = hdf5::File::open_rw(&path).unwrap();
    assert_eq!(native.member_names().unwrap(), Vec::<String>::new());
    assert_eq!(native.attr_names().unwrap(), Vec::<String>::new());
    native
        .new_attr::<u32>()
        .shape(())
        .create("audit_counter")
        .unwrap()
        .write_scalar(&7)
        .unwrap();
    native.flush().unwrap();
    native.close().unwrap();
    let native = hdf5::File::open(&path).unwrap();
    assert_eq!(
        native
            .attr("audit_counter")
            .unwrap()
            .read_scalar::<u32>()
            .unwrap(),
        7
    );
    native.close().unwrap();
    let pure = File::open_streaming(&path).unwrap();
    assert_eq!(
        pure.root().attrs().unwrap().get("audit_counter"),
        Some(&AttrValue::U32(7))
    );
    assert_eq!(pure.root().groups().unwrap(), Vec::<String>::new());
    pure.close().unwrap();
}

const ROOT_ADDRESS: u64 = 128;
/// The version and flags of a Group Info message with default settings, defined in "The Group
/// Info Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_groupinfo
const GROUP_INFO_DEFAULT: [u8; 2] = [0, 0];

#[test]
fn a_native_header_preserves_its_records_in_a_bounded_allocation() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("native.h5");
    let output = directory.path().join("padded.h5");
    let native = hdf5::File::with_options()
        .with_fapl(|properties| properties.libver_latest())
        .create(&source)
        .unwrap();
    native
        .new_attr::<u32>()
        .shape(())
        .create("source_counter")
        .unwrap()
        .write_scalar(&11)
        .unwrap();
    native.close().unwrap();
    let pure = File::open_streaming(&source).unwrap();
    let root = usize::try_from(pure.superblock().root_group_address.get()).unwrap();
    let widths = FormatWidths {
        offsets: pure.superblock().offset_size,
        lengths: pure.superblock().length_size,
    };
    pure.close().unwrap();
    let source_bytes = fs::read(&source).unwrap();
    let parsed = ObjectHeaderPrefix::parse(&source_bytes[root..]).unwrap();
    let messages = &source_bytes
        [root + parsed.len..root + parsed.len + usize::try_from(parsed.chunk0_size).unwrap()];
    let plan = ObjectHeaderAllocation::new(
        parsed.prefix,
        widths,
        StoredAddress::new(ROOT_ADDRESS),
        131_123,
        messages,
    )
    .unwrap();
    let mut bytes = Superblock::new(Widths::EIGHT)
        .root_header_address(ROOT_ADDRESS)
        .eof_address(ROOT_ADDRESS + 131_123)
        .build();
    bytes.resize(usize::try_from(ROOT_ADDRESS).unwrap(), 0);
    bytes.extend(plan.encode());
    fs::write(&output, bytes).unwrap();
    let native = hdf5::File::open_rw(&output).unwrap();
    assert_eq!(
        native
            .attr("source_counter")
            .unwrap()
            .read_scalar::<u32>()
            .unwrap(),
        11
    );
    native
        .new_attr::<u32>()
        .shape(())
        .create("audit_counter")
        .unwrap()
        .write_scalar(&7)
        .unwrap();
    native.flush().unwrap();
    native.close().unwrap();
    let pure = File::open_streaming(&output).unwrap();
    let attrs = pure.root().attrs().unwrap();
    assert_eq!(attrs.len(), 2);
    assert_eq!(attrs.get("source_counter"), Some(&AttrValue::U32(11)));
    assert_eq!(attrs.get("audit_counter"), Some(&AttrValue::U32(7)));
    pure.close().unwrap();
}

/// Version 0 of Link Info messages, defined in "The Link Info Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_linkinfo
const LINK_INFO_VERSION: u8 = 0;
const NO_FLAGS: u8 = 0;

#[rstest]
#[case::plain_empty(HeaderFlags::default(), &[])]
#[case::plain_nil_only(HeaderFlags::default(), &[17, 1])]
#[case::tracked_empty(HeaderFlags::TRACKS_CREATION_ORDER | HeaderFlags::INDEXES_CREATION_ORDER, &[])]
#[case::tracked_nil_only(HeaderFlags::TRACKS_CREATION_ORDER | HeaderFlags::INDEXES_CREATION_ORDER, &[17, 1])]
fn empty_extension_chunks_survive_native_read_write_flush(
    #[case] flags: HeaderFlags,
    #[case] nil_bodies: &[usize],
    #[values(11, 12, 14, 17)] capacity: u64,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("empty-extension.h5");
    let native = hdf5::File::with_options()
        .with_fapl(|properties| properties.libver_latest())
        .create(&path)
        .unwrap();
    native
        .new_attr::<u32>()
        .shape(())
        .create("audit_counter")
        .unwrap()
        .write_scalar(&7)
        .unwrap();
    native.close().unwrap();
    let pure = File::open_streaming(&path).unwrap();
    let root = pure.superblock().root_group_address.get();
    let widths = FormatWidths {
        offsets: pure.superblock().offset_size,
        lengths: pure.superblock().length_size,
    };
    assert_eq!(pure.superblock().base_address.get(), 0);
    pure.close().unwrap();
    let mut bytes = fs::read(&path).unwrap();
    let address = u64::try_from(bytes.len()).unwrap();
    let records = header::message_records(
        &nil_bodies
            .iter()
            .map(|&len| Message::nil(len))
            .collect::<Vec<_>>(),
        flags,
    );
    let plan = ObjectHeaderAllocation::new(
        ObjectHeaderPrefix::with_layout(MessageRecordLayout::from_header_flags(flags.0)),
        widths,
        StoredAddress::new(address),
        capacity,
        &records,
    )
    .unwrap();
    bytes.extend(plan.encode());
    let superblock = Superblock::new(Widths::new(
        usize::from(widths.offsets.get()),
        usize::from(widths.lengths.get()),
    ))
    .root_header_address(root)
    .extension_address(address)
    .eof_address(address + capacity)
    .build();
    bytes[..superblock.len()].copy_from_slice(&superblock);
    fs::write(&path, bytes).unwrap();
    let native = hdf5::File::open_rw(&path).unwrap();
    assert_eq!(
        native
            .attr("audit_counter")
            .unwrap()
            .read_scalar::<u32>()
            .unwrap(),
        7
    );
    native
        .attr("audit_counter")
        .unwrap()
        .write_scalar(&8_u32)
        .unwrap();
    native.flush().unwrap();
    native.close().unwrap();
    let pure = File::open_streaming(&path).unwrap();
    assert_eq!(
        pure.root().attrs().unwrap().get("audit_counter"),
        Some(&AttrValue::U32(8))
    );
    pure.close().unwrap();
}
