#![cfg(feature = "libhdf5")]

use std::path::Path;

use hdf5_pure::{Error, File, FormatError, MessageType};
use hdf5_pure_core::__private::BaseAddressExt;
use hdf5_pure_format::__private;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::MessageFilter;
use hdf5_pure_format::__private::MessageRecordLayout;
use hdf5_pure_format::__private::ObjectHeader;
use hdf5_pure_format::__private::ObjectHeaderContinuation;
use hdf5_pure_format::__private::ObjectHeaderPrefix;
use libhdf5::File as NativeFile;
use libhdf5::{IndexType, IterationOrder, MajorErrorCode, MinorErrorCode};
use rstest::rstest;
use test_util::checksum;
use test_util::object_header::v1;
use test_util::object_header::v2;
use test_util::temp;
use test_util::widths::Widths;
use test_util_hdf5::file;
use test_util_hdf5::reader::ReadBackend;
use test_util_hdf5::storage;

#[test]
fn compact_links_spread_over_continuation_blocks_list_in_the_native_order_of_the_c_library() {
    let path = temp::temp_path("continuations.h5");
    write_links(&path);
    let file = File::open(&path).unwrap();
    assert_root_refers_to_nested_continuation_blocks(&path, &file);

    let c_names = libhdf5::File::open(&path)
        .unwrap()
        .links(IndexType::Name, IterationOrder::Native)
        .unwrap()
        .into_iter()
        .map(|(name, _)| name)
        .collect::<Vec<_>>();
    assert_eq!(file.root().groups().unwrap(), c_names);
}

#[rstest]
#[case::root(false)]
#[case::continuation(true)]
fn a_v2_header_record_body_overrun_is_rejected_by_buffered_streaming_and_native_reads(
    #[case] continuation: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let valid_path = directory.path().join("valid.h5");
    let malformed_path = directory.path().join("malformed.h5");
    write_links(&valid_path);

    let native = NativeFile::open(&valid_path).unwrap();
    let mut native_names = native.member_names().unwrap();
    native_names.sort();
    assert_eq!(native_names.len(), LINKS);
    native.close().unwrap();

    let backends = [ReadBackend::Buffered, ReadBackend::Streaming];
    for backend in backends {
        let file = backend.open(&valid_path).unwrap();
        let mut names = file.root().groups().unwrap();
        names.sort();
        assert_eq!(names, native_names);
        file.close().unwrap();
    }

    let mut image = std::fs::read(&valid_path).unwrap();
    let error = extend_last_record_past_chunk(&valid_path, &mut image, continuation);
    std::fs::write(&malformed_path, image).unwrap();

    let error_from_native = NativeFile::open(&malformed_path)
        .and_then(|file| {
            let names = file.member_names();
            file.close()?;
            names
        })
        .unwrap_err();
    assert_native_record_overrun(&error_from_native);

    for backend in backends {
        let result = backend.open(&malformed_path).and_then(|file| {
            let names = file.root().groups();
            file.close()?;
            names
        });
        let error_from_pure = result.unwrap_err();
        let Error::Format(actual) = error_from_pure else {
            panic!("expected a format error from {backend:?}, got {error_from_pure:?}");
        };
        assert_eq!(actual, error);
    }
}

fn extend_last_record_past_chunk(path: &Path, image: &mut [u8], continuation: bool) -> FormatError {
    let file = File::open(path).unwrap();
    let superblock = file.superblock();
    let root = superblock
        .base_address
        .resolve(superblock.root_group_address)
        .unwrap();
    let root_at = usize::try_from(root.get()).unwrap();
    let prefix = ObjectHeaderPrefix::parse(&image[root_at..]).unwrap();
    let layout = prefix.prefix.layout;
    let (at, len, records_start) = if continuation {
        let extents = storage::header_extents(&file, image, root);
        let block = extents
            .iter()
            .find(|range| range.start() != root)
            .expect("the root header refers to a continuation block");
        let at = usize::try_from(block.start().get()).unwrap();
        let len = usize::try_from(block.length()).unwrap();
        let records = __private::continuation_block_messages(&image[at..at + len]).unwrap();
        (at, len, records.start)
    } else {
        (
            root_at,
            prefix.len + usize::try_from(prefix.chunk0_size).unwrap() + checksum::CHECKSUM,
            prefix.len,
        )
    };
    file.close().unwrap();

    let region = &image[at..at + len - checksum::CHECKSUM];
    let mut record_start = records_start;
    let mut last_start = None;
    while let Some(record) = layout.next_message(region, record_start).unwrap() {
        last_start = Some(record_start);
        record_start = record.body_range.end;
    }
    assert_eq!(record_start, region.len());
    let last_start = last_start.expect("the target chunk contains an object-header record");
    let available = region.len();
    let body_len = available - last_start - layout.prefix_len();
    v2::set_record_body_size(image, at + last_start, u16::try_from(body_len + 1).unwrap());
    checksum::restamp(image, at, len);

    FormatError::UnexpectedEof {
        expected: available + 1,
        available,
    }
}

// The chunk decoder in `H5Ocache.c` rejects the overrun before decoding the message body
// in HDF5 1.8.23 (`H5O_chunk_deserialize`) and HDF5 1.10.11, 1.12.3, and 1.14.6
// (`H5O__chunk_deserialize`). In HDF5 2.2.0, `H5O__chunk_deserialize` rejects the chunk
// when it checks that the checksum bytes are available.
fn assert_native_record_overrun(error: &libhdf5::Error) {
    let version = libhdf5::library_version();
    let function = if version < (1, 10, 0) {
        "H5O_chunk_deserialize"
    } else {
        "H5O__chunk_deserialize"
    };
    let (minor, description) = if version >= (2, 2, 0) {
        (
            MinorErrorCode::Overflow,
            "ran off end of input buffer while decoding",
        )
    } else {
        (MinorErrorCode::CantInit, "corrupt object header")
    };
    let stack = error.stack().expect("expected a native HDF5 error stack");
    let frame = stack.last().expect("native rejection has no error frames");
    assert_eq!(
        (
            frame.major_code(),
            frame.minor_code(),
            frame.func(),
            frame.desc()
        ),
        (MajorErrorCode::ObjectHeader, minor, function, description),
        "libhdf5 {version:?}: {error:?}"
    );
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
