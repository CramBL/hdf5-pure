use std::fs;

use rstest::rstest;
use test_util::bytes;
use test_util::data_layout;
use test_util::dataspace;
use test_util::datatype;
use test_util::disk_log::{self, DiskOp};
use test_util::image::Image;
use test_util::link_message::HardLink;
use test_util::object_header::{Message, MessageType, v2};
use test_util::superblock::v2::Superblock;
use test_util::widths::Widths;

use crate::{File, FileAccessProperties, MemoryStrategy};

#[rstest]
#[case::two_bytes(Widths::new(2, 2))]
#[case::four_bytes(Widths::FOUR)]
fn continued_count_decrease_uses_physical_addresses_and_preserves_the_allocation(
    #[case] widths: Widths,
) {
    const BASE: usize = 512;
    let path = test_util::temp::temp_path("continued-count.h5");
    let userblock = [0xa5; BASE];
    let mut image = Image::starting_with(&userblock);
    image.append(&[0; 64]);
    let original = count_chunk(2);
    let continuation = image.append(&original);
    let header = v2::Header::new()
        .messages([
            Message::new(MessageType::DATASPACE, &dataspace::scalar()),
            Message::new(MessageType::DATATYPE, &datatype::f64_le()),
            Message::new(
                MessageType::DATA_LAYOUT,
                &data_layout::compact(&17_f64.to_le_bytes()),
            ),
        ])
        .continuation(continuation - BASE, &original, widths)
        .build();
    let target = image.append(&header);
    let mut info = vec![0, 0];
    bytes::push_undefined_address(&mut info, widths.offset);
    bytes::push_undefined_address(&mut info, widths.offset);
    let root = image.append(
        &v2::Header::new()
            .messages([
                Message::new(MessageType::LINK_INFO, &info),
                Message::new(
                    MessageType::LINK,
                    &HardLink::new("alias", (target - BASE) as u64).build(widths),
                ),
                Message::new(
                    MessageType::LINK,
                    &HardLink::new("kept", (target - BASE) as u64).build(widths),
                ),
            ])
            .build(),
    );
    let superblock = Superblock::new(widths)
        .version(3)
        .base_address(BASE as u64)
        .root_header_address((root - BASE) as u64)
        .eof_address(image.as_bytes().len() as u64)
        .build();
    image.place(BASE, &superblock);
    fs::write(&path, image.as_bytes()).unwrap();

    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Mirrored),
    )
    .unwrap();
    assert_eq!(file.superblock().base_address.get(), BASE as u64);
    assert_eq!(
        file.dataset("kept").unwrap().header_address().unwrap(),
        target as u64
    );
    assert_eq!(file.dataset("alias").unwrap().read_f64().unwrap(), [17.0]);

    file.root().delete("alias").unwrap();
    disk_log::start();
    file.commit().unwrap();
    file.sync().unwrap();
    let recording = disk_log::finish();
    let covers = |operation: &DiskOp, address: usize, length: usize| {
        matches!(operation, DiskOp::Write { offset, bytes }
            if *offset <= address as u64
                && (address + length) as u64 <= *offset + bytes.len() as u64)
    };
    let published = recording
        .operations
        .iter()
        .position(|op| covers(op, BASE, superblock.len()))
        .unwrap();
    for address in [continuation + 9, continuation + original.len() - 4] {
        assert!(
            recording.operations[..=published]
                .iter()
                .all(|op| !covers(op, address, 4))
        );
        assert!(
            recording.operations[published + 1..]
                .iter()
                .any(|op| covers(op, address, 4))
        );
    }
    assert!(
        file.reusable_free_sections()
            .unwrap()
            .sections()
            .iter()
            .all(|section| {
                section.end_address().get() <= continuation as u64
                    || section.address().get() >= (continuation + original.len()) as u64
            })
    );
    file.close().unwrap();

    let after = fs::read(&path).unwrap();
    assert_eq!(after[..BASE], userblock);
    assert_eq!(after[target..target + header.len()], header);
    assert_eq!(
        after[continuation..continuation + original.len()],
        count_chunk(1)
    );
    let reopened = File::open(&path).unwrap();
    assert_eq!(
        usize::from(reopened.superblock().offset_size.get()),
        widths.offset
    );
    assert_eq!(
        usize::from(reopened.superblock().length_size.get()),
        widths.length
    );
    assert_eq!(reopened.root().datasets().unwrap(), ["kept"]);
    assert_eq!(
        reopened.dataset("kept").unwrap().read_f64().unwrap(),
        [17.0]
    );
}

fn count_chunk(count: u32) -> Vec<u8> {
    let mut padding = Message::nil(15);
    padding.data.fill(0xa7);
    v2::continuation_chunk(
        &[Message::reference_count(count), padding],
        v2::HeaderFlags::default(),
    )
}
