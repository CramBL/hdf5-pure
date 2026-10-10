#![cfg(feature = "hdf5")]

use std::path::Path;

use hdf5::IndexType;
use hdf5::IterationOrder;
use hdf5_pure::File;
use hdf5_pure::MessageType;
use hdf5_pure_format::__private;
use hdf5_pure_format::__private::MessageRecordLayout;
use hdf5_pure_format::__private::ObjectHeaderContinuation;
use hdf5_pure_format::__private::ObjectHeaderPrefix;
use test_util::temp;
use test_util_hdf5::file;

#[test]
fn compact_links_spread_over_continuation_blocks_list_in_the_native_order_of_the_c_library() {
    let path = temp::temp_path("continuations.h5");
    write_links(&path);
    let file = File::open(&path).unwrap();
    assert_root_refers_to_nested_continuation_blocks(&path, &file);

    let c_names = hdf5::File::open(&path)
        .unwrap()
        .links(IndexType::Name, IterationOrder::Native)
        .unwrap()
        .into_iter()
        .map(|(name, _)| name)
        .collect::<Vec<_>>();
    assert_eq!(file.root().groups().unwrap(), c_names);
}

fn write_links(path: &Path) {
    let c_file = file::libhdf5_create_v18(path);
    for index in 0..LINKS {
        let name = format!("{index:02}{}", "n".repeat(LINK_NAME_LEN - 2));
        c_file.create_group(&name).unwrap();
    }
    c_file.close().unwrap();
}

fn assert_root_refers_to_nested_continuation_blocks(path: &Path, file: &File) {
    let bytes = std::fs::read(path).unwrap();
    let superblock = file.superblock();
    let root_at = usize::try_from(superblock.root_group_address).unwrap();
    let prefix = ObjectHeaderPrefix::parse(&bytes[root_at..]).unwrap();
    let layout = prefix.prefix.layout;
    let chunk0_end = root_at + prefix.len + usize::try_from(prefix.chunk0_size).unwrap();
    let referenced = |region: &[u8], records_start: usize| {
        referenced_blocks(
            layout,
            superblock.offset_size,
            superblock.length_size,
            region,
            records_start,
        )
    };

    let root_blocks = referenced(&bytes[..chunk0_end], root_at + prefix.len);
    let nested_blocks = root_blocks
        .iter()
        .flat_map(|continuation| {
            let at = usize::try_from(continuation.address().get()).unwrap();
            let block = &bytes[at..at + usize::try_from(continuation.length()).unwrap()];
            let records = __private::continuation_block_messages(block).unwrap();
            referenced(&block[..records.end], records.start)
        })
        .collect::<Vec<_>>();
    assert!(
        root_blocks.len() >= 2 && !nested_blocks.is_empty(),
        "chunk 0 refers to {} blocks, which refer to {} more",
        root_blocks.len(),
        nested_blocks.len()
    );
}

fn referenced_blocks(
    layout: MessageRecordLayout,
    offset_size: u8,
    length_size: u8,
    region: &[u8],
    records_start: usize,
) -> Vec<ObjectHeaderContinuation> {
    std::iter::successors(
        layout.next_message(region, records_start).unwrap(),
        |record| layout.next_message(region, record.body_range.end).unwrap(),
    )
    .filter(|record| record.msg_type == MessageType::OBJECT_HEADER_CONTINUATION)
    .map(|record| ObjectHeaderContinuation::parse(record.body, offset_size, length_size).unwrap())
    .collect()
}

const LINKS: usize = 8;
const LINK_NAME_LEN: usize = 300;
