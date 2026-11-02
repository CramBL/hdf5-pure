#![cfg(feature = "hdf5")]

use std::path::Path;

use hdf5::File as NativeFile;
use hdf5::IndexType;
use hdf5::IterationOrder;
use hdf5_pure::File;
use hdf5_pure::MessageType;
use hdf5_pure_core::__private::BaseAddressExt;
use hdf5_pure_format::__private;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::MessageFilter;
use hdf5_pure_format::__private::MessageRecordLayout;
use hdf5_pure_format::__private::ObjectHeader;
use hdf5_pure_format::__private::ObjectHeaderContinuation;
use hdf5_pure_format::__private::ObjectHeaderPrefix;
use rstest::rstest;
use test_util::object_header::v1;
use test_util::temp;
use test_util::widths::Widths;
use test_util_hdf5::file;
use test_util_hdf5::storage;

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

#[rstest]
#[case::version_1(HeaderVersion::One)]
#[case::version_2(HeaderVersion::Two)]
fn chunk_observations_match_native_written_header_storage(#[case] version: HeaderVersion) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("observed-header.h5");
    let native = version.create(&path);
    let modern = matches!(version, HeaderVersion::Two);
    for index in 0..LINKS {
        let link_name = format!("{index:02}{}", "n".repeat(LINK_NAME_LEN - 2));
        native.create_group(&link_name).unwrap();
        native
            .new_attr::<u32>()
            .shape(())
            .create(format!("audit_{index:02}{}", "a".repeat(LINK_NAME_LEN)).as_str())
            .unwrap()
            .write_scalar(&u32::try_from(index).unwrap())
            .unwrap();
    }
    native.flush().unwrap();
    let mut expected_names = native.member_names().unwrap();
    expected_names.sort();
    native.close().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let pure = File::open_streaming(&path).unwrap();
    let superblock = pure.superblock();
    let address = superblock
        .base_address
        .resolve(superblock.root_group_address)
        .unwrap();
    let widths = FormatWidths {
        offsets: superblock.offset_size,
        lengths: superblock.length_size,
    };
    let mut expected = if modern {
        storage::header_extents(&pure, &bytes, address)
            .into_iter()
            .map(|range| (range.start().get(), range.length()))
            .collect::<Vec<_>>()
    } else {
        let at = usize::try_from(address.get()).unwrap();
        v1::chunks(
            &bytes,
            at,
            Widths {
                offset: usize::from(widths.offsets.get()),
                length: usize::from(widths.lengths.get()),
            },
        )
        .into_iter()
        .enumerate()
        .map(|(index, chunk)| {
            let prefix = if index == 0 { v1::PREFIX } else { 0 };
            (
                u64::try_from(chunk.range.start - prefix).unwrap(),
                u64::try_from(chunk.range.len() + prefix).unwrap(),
            )
        })
        .collect::<Vec<_>>()
    };
    assert!(expected.len() > 1);
    let mut observed = Vec::new();
    let parsed = ObjectHeader::parse_from_source_with_chunk_observer(
        bytes.as_slice(),
        AccessMode::ReadOnly,
        address,
        widths,
        superblock.base_address,
        MessageFilter::Only(&mut |kind, _| {
            matches!(kind, MessageType::LINK | MessageType::SYMBOL_TABLE)
        }),
        &mut |chunk| {
            assert_eq!(chunk.v2_layout().is_some(), modern);
            assert!(chunk.allocation().contains(chunk.records().range()));
            let start = usize::try_from(chunk.records().range().start().get()).unwrap();
            let end = usize::try_from(chunk.records().range().end().get()).unwrap();
            assert_eq!(chunk.records().bytes().as_ptr(), bytes[start..end].as_ptr());
            observed.push((
                chunk.allocation().start().get(),
                chunk.allocation().length(),
            ));
        },
    )
    .unwrap();
    assert_eq!(
        parsed
            .messages
            .iter()
            .map(|message| message.msg_type)
            .collect::<Vec<_>>(),
        if modern {
            vec![MessageType::LINK; LINKS]
        } else {
            vec![MessageType::SYMBOL_TABLE]
        }
    );
    expected.sort();
    observed.sort();
    assert_eq!(observed, expected);
    let mut actual_names = pure.root().groups().unwrap();
    actual_names.sort();
    assert_eq!(actual_names, expected_names);
    pure.close().unwrap();
}

#[derive(Clone, Copy)]
enum HeaderVersion {
    One,
    Two,
}

impl HeaderVersion {
    fn create(self, path: &Path) -> NativeFile {
        match self {
            Self::One => file::libhdf5_create_earliest(path),
            Self::Two => file::libhdf5_create_v18(path),
        }
    }
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
    let root_at = usize::try_from(
        superblock
            .base_address
            .resolve(superblock.root_group_address)
            .unwrap()
            .get(),
    )
    .unwrap();
    let prefix = ObjectHeaderPrefix::parse(&bytes[root_at..]).unwrap();
    let layout = prefix.prefix.layout;
    let chunk0_end = root_at + prefix.len + usize::try_from(prefix.chunk0_size).unwrap();
    let referenced = |region: &[u8], records_start: usize| {
        referenced_blocks(
            region,
            layout,
            FormatWidths {
                offsets: superblock.offset_size,
                lengths: superblock.length_size,
            },
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
    region: &[u8],
    layout: MessageRecordLayout,
    widths: FormatWidths,
    records_start: usize,
) -> Vec<ObjectHeaderContinuation> {
    std::iter::successors(
        layout.next_message(region, records_start).unwrap(),
        |record| layout.next_message(region, record.body_range.end).unwrap(),
    )
    .filter(|record| record.msg_type == MessageType::OBJECT_HEADER_CONTINUATION)
    .map(|record| ObjectHeaderContinuation::parse_with_widths(record.body, widths).unwrap())
    .collect()
}

const LINKS: usize = 8;
const LINK_NAME_LEN: usize = 300;
