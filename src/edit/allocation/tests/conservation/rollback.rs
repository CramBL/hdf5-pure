#[cfg(feature = "deflate")]
use std::collections::BTreeSet;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use rstest::rstest;
use test_util::allocation_oracle::AllocationKind;
use test_util::data_layout;
use test_util::disk_log;
use test_util::object_header::MessageType as FixtureMessage;
use test_util::object_header::v2;

#[cfg(feature = "deflate")]
use crate::DeflateLevel;
use crate::Error;
use crate::File;
use crate::FileAccessProperties;
use crate::FileBuilder;
use crate::FileSpacePageSize;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::MaxExtent;
use crate::MemoryStrategy;
use crate::SyncPolicy;
use crate::access_mode::AccessMode;
use crate::address::DecodedStoredAddress;
use crate::data_layout::ChunkIndexLayout;
use crate::data_layout::DataLayout;
use crate::edit::WriteEngine;
use crate::edit::context::FileContext;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::source::BytesSource;
use crate::width::FormatWidths;

use super::CHUNK;
use super::Observation;

#[rstest]
#[case::fletcher32(Encoding::Fletcher32)]
#[cfg_attr(feature = "deflate", case::deflate(Encoding::Deflate))]
fn failed_appends_restore_complete_space_before_different_sized_retries(
    #[case] encoding: Encoding,
    #[values(
        WritePath::Immediate,
        WritePath::BorrowedWrite,
        WritePath::BorrowedFlush,
        WritePath::OwnedWrite,
        WritePath::OwnedFlush
    )]
    writer: WritePath,
    #[values(false, true)] paged: bool,
    #[values(false, true)] deferred: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] policy: SyncPolicy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("rollback.h5");
    fixture(&path, paged, deferred, encoding);
    run(&path, writer, deferred, backing, policy, false, encoding);
}

#[cfg(feature = "deflate")]
#[rstest]
#[case::immediate(WritePath::Immediate)]
#[case::borrowed_write(WritePath::BorrowedWrite)]
#[case::borrowed_flush(WritePath::BorrowedFlush)]
#[case::owned_write(WritePath::OwnedWrite)]
#[case::owned_flush(WritePath::OwnedFlush)]
fn explicit_page_buffers_drain_append_publication_after_rollback(
    #[case] writer: WritePath,
    #[values(false, true)] paged: bool,
    #[values(false, true)] deferred: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("buffered-rollback.h5");
    fixture(&path, paged, deferred, Encoding::Deflate);
    run(
        &path,
        writer,
        deferred,
        backing,
        SyncPolicy::OnClose,
        true,
        Encoding::Deflate,
    );
}

#[rstest]
#[case::below_page(
    SyncPolicy::OnClose,
    usize::try_from(PAGE_BYTES - 1).unwrap(),
    "a page buffer must be at least the file's file-space page size"
)]
#[case::durable(
    SyncPolicy::Always,
    usize::try_from(PAGE_BYTES).unwrap(),
    "a page buffer does nothing under SyncPolicy::Always, whose every barrier is \
     an fsync that flushes it; pair with_page_buffer_size with \
     with_sync_policy(SyncPolicy::OnClose), or drop it"
)]
fn rejected_page_buffers_preserve_physical_append_fixtures(
    #[case] policy: SyncPolicy,
    #[case] budget: usize,
    #[case] reason: &str,
    #[values(false, true)] paged: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("rejected-buffer.h5");
    fixture(&path, paged, true, Encoding::Fletcher32);
    let before = fs::read(&path).unwrap();
    disk_log::start();
    let error = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_memory_strategy(backing)
            .with_sync_policy(policy)
            .with_page_buffer_size(budget),
    )
    .unwrap_err();
    let recording = disk_log::finish();
    let Error::EditUnsupported(actual) = error else {
        panic!("expected the page-buffer rejection, got {error:?}");
    };
    assert_eq!(actual, reason);
    assert!(recording.operations.is_empty());
    assert!(recording.events.is_empty());
    assert_eq!(fs::read(&path).unwrap(), before);
}

/// The filter used to produce stored chunks smaller or larger than a file-space page.
#[derive(Clone, Copy, Debug)]
pub(super) enum Encoding {
    #[cfg(feature = "deflate")]
    Deflate,
    Fletcher32,
}

/// The append or flush operation targeted by an injected write failure.
#[derive(Clone, Copy, Debug)]
pub(super) enum WritePath {
    BorrowedFlush,
    BorrowedWrite,
    Immediate,
    OwnedFlush,
    OwnedWrite,
}

impl WritePath {
    fn fail(self, file: &File, start: usize) {
        let before = file.disk_bytes().unwrap();
        let length = match self {
            Self::BorrowedFlush | Self::OwnedFlush => 137,
            _ => 2 * CHUNK,
        };
        let values = values(start, length);
        let expected_bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let mut dataset = file.dataset("d").unwrap();
        let error = match self {
            Self::Immediate => {
                dataset.append::<i32>(&[]).unwrap();
                assert_eq!(file.disk_bytes().unwrap(), before);
                disk_log::fail_write_after(0, 0);
                dataset.append(&values).unwrap_err()
            }
            Self::BorrowedFlush | Self::BorrowedWrite => {
                let mut appender = dataset.buffered_appender().unwrap();
                appender.append::<i32>(&[]).unwrap();
                appender.flush().unwrap();
                assert_eq!(file.disk_bytes().unwrap(), before);
                if matches!(self, Self::BorrowedFlush) {
                    appender.append(&values).unwrap();
                    assert_eq!(file.disk_bytes().unwrap(), before);
                    disk_log::fail_write_after(0, 0);
                    let error = appender.flush().unwrap_err();
                    assert_eq!(file.disk_bytes().unwrap(), before);
                    assert_eq!(appender.discard(), expected_bytes);
                    error
                } else {
                    disk_log::fail_write_after(0, 0);
                    let error = appender.append(&values).unwrap_err();
                    assert_eq!(file.disk_bytes().unwrap(), before);
                    assert_eq!(appender.discard(), expected_bytes);
                    error
                }
            }
            Self::OwnedFlush | Self::OwnedWrite => {
                let mut appender = dataset.into_buffered_appender().unwrap();
                appender.append::<i32>(&[]).unwrap();
                appender.flush().unwrap();
                assert_eq!(file.disk_bytes().unwrap(), before);
                if matches!(self, Self::OwnedFlush) {
                    appender.append(&values).unwrap();
                    assert_eq!(file.disk_bytes().unwrap(), before);
                    disk_log::fail_write_after(0, 0);
                    let error = appender.flush().unwrap_err();
                    assert_eq!(file.disk_bytes().unwrap(), before);
                    assert_eq!(appender.discard(), expected_bytes);
                    error
                } else {
                    disk_log::fail_write_after(0, 0);
                    let error = appender.append(&values).unwrap_err();
                    assert_eq!(file.disk_bytes().unwrap(), before);
                    assert_eq!(appender.discard(), expected_bytes);
                    error
                }
            }
        };
        disk_log::clear_failures();
        let Error::Io(error) = error else {
            panic!("expected the injected write error, got {error:?}");
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(error.to_string(), "injected disk write failure");
        assert_eq!(file.disk_bytes().unwrap(), before);
    }

    fn append(self, file: &File, values: &[i32]) {
        let mut dataset = file.dataset("d").unwrap();
        match self {
            Self::Immediate => dataset.append(values).unwrap(),
            Self::BorrowedFlush | Self::BorrowedWrite => {
                let mut appender = dataset.buffered_appender().unwrap();
                appender.append(values).unwrap();
                if matches!(self, Self::BorrowedFlush) {
                    appender.flush().unwrap();
                }
                appender.finish().unwrap();
            }
            Self::OwnedFlush | Self::OwnedWrite => {
                let mut appender = dataset.into_buffered_appender().unwrap();
                appender.append(values).unwrap();
                if matches!(self, Self::OwnedFlush) {
                    appender.flush().unwrap();
                }
                appender.finish().unwrap();
            }
        }
    }
}

/// Checks physical rollback, differently sized retries, and a reopened editing session.
pub(super) fn run(
    path: &Path,
    writer: WritePath,
    deferred: bool,
    backing: MemoryStrategy,
    policy: SyncPolicy,
    page_buffer: bool,
    encoding: Encoding,
) {
    let original = fs::read(path).unwrap();
    let before = Observation::read(&original);
    assert_index_address(&original, deferred);
    assert_eq!(before.index_defined, !deferred);
    assert_eq!(before.values.len(), if deferred { 0 } else { CHUNK });
    assert!(before.free.is_empty());
    let properties = FileAccessProperties::new()
        .with_memory_strategy(backing)
        .with_sync_policy(policy)
        .with_page_buffer_size(if page_buffer {
            usize::try_from(PAGE_BYTES).unwrap()
        } else {
            0
        });
    let file = File::open_rw_with_options(path, properties).unwrap();
    let baseline = file.disk_bytes().unwrap();
    let initial = before.values.len();
    writer.fail(&file, initial);
    let failed = file.disk_bytes().unwrap();
    assert_eq!(failed, baseline);
    assert_index_address(&failed, deferred);
    let after_failure = Observation::read(&failed);
    assert_eq!(after_failure.index_defined, !deferred);
    assert_eq!(after_failure.chunks, before.chunks);
    assert_eq!(after_failure.values, before.values);
    assert_eq!(after_failure.free, before.free);
    assert_eq!(after_failure.live, before.live);
    assert_eq!(
        file.dataset("d").unwrap().shape().unwrap(),
        [u64::try_from(initial).unwrap()]
    );
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        before.values
    );

    let retry_length = 9 * CHUNK + 113;
    disk_log::start();
    writer.append(&file, &values(initial, retry_length));
    let recording = disk_log::finish();
    assert!(!recording.operations.is_empty());
    assert!(!recording.events.is_empty());
    let retry = file.disk_bytes().unwrap();
    let retried = inspect(&retry, initial + retry_length);
    assert_activity(&before, &retried, encoding);
    writer.fail(&file, initial + retry_length);
    inspect(&file.disk_bytes().unwrap(), initial + retry_length);
    let completed_length = initial + retry_length + 2 * CHUNK + 19;
    writer.append(&file, &values(initial + retry_length, 2 * CHUNK + 19));
    inspect(&file.disk_bytes().unwrap(), completed_length);
    file.close().unwrap();
    inspect(&fs::read(path).unwrap(), completed_length);
    let file = File::open_rw_with_options(path, properties).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        values(0, completed_length)
    );
    writer.fail(&file, completed_length);
    inspect(&file.disk_bytes().unwrap(), completed_length);
    let final_length = completed_length + 3 * CHUNK + 17;
    writer.append(&file, &values(completed_length, 3 * CHUNK + 17));
    inspect(&file.disk_bytes().unwrap(), final_length);
    file.close().unwrap();
    inspect(&fs::read(path).unwrap(), final_length);
}

/// Removes persisted free-space references to force allocation beyond the original file end.
pub(super) fn force_tail(path: &Path) {
    let mut bytes = fs::read(path).unwrap();
    let context = FileContext::new(crate::superblock::read(bytes.as_slice()).unwrap()).unwrap();
    let (extension, mut info) = WriteEngine::extension_fsinfo(&BytesSource::new(&bytes), &context)
        .unwrap()
        .unwrap();
    info.manager_addrs.fill(DecodedStoredAddress::Undefined);
    let encoded = super::format::serialize_file_space_info_v1(context.widths(), &info).unwrap();
    super::extents::replace_space_info(&mut bytes, &context, extension, &encoded);
    fs::write(path, bytes).unwrap();
    assert!(Observation::read(&fs::read(path).unwrap()).free.is_empty());
}

fn fixture(path: &Path, paged: bool, deferred: bool, encoding: Encoding) {
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(if paged {
            FileSpaceStrategy::Page
        } else {
            FileSpaceStrategy::FsmAggr
        })
        .with_page_size(FileSpacePageSize::try_from(PAGE_BYTES).unwrap())
        .with_persistence(true)
        .with_threshold(1),
    );
    let target = if deferred {
        builder
            .create_dataset("d")
            .with_unallocated_storage(crate::make_i32_type(), &[0])
    } else {
        builder.create_dataset("d").with_i32_data(&values(0, CHUNK))
    };
    target
        .with_chunks(&[u64::try_from(CHUNK).unwrap()])
        .with_maxshape(&[MaxExtent::Unlimited]);
    match encoding {
        #[cfg(feature = "deflate")]
        Encoding::Deflate => {
            target.with_deflate(DeflateLevel::Six);
        }
        Encoding::Fletcher32 => {
            target.with_fletcher32();
        }
    }
    builder
        .create_dataset("ceiling")
        .with_u8_data(&[9; super::CEILING]);
    builder.write(path).unwrap();
    if deferred {
        let mut bytes = fs::read(path).unwrap();
        let (at, mut layout) = test_util::bytes::signature_offsets(&bytes, v2::SIGNATURE)
            .into_iter()
            .find_map(|at| {
                ObjectHeader::parse(&bytes, AccessMode::ReadOnly, at, 8, 8)
                    .unwrap()
                    .messages
                    .into_iter()
                    .find(|message| {
                        message.msg_type == MessageType::DATA_LAYOUT
                            && matches!(
                                DataLayout::parse(&message.data, 8, 8).unwrap(),
                                DataLayout::Chunked { .. }
                            )
                    })
                    .map(|message| (at, message.data))
            })
            .unwrap();
        let (_, pointer) =
            DataLayout::parse_with_ea_address(&layout, FormatWidths::from_sizes(8, 8).unwrap())
                .unwrap();
        layout[pointer.unwrap()].fill(u8::MAX);
        v2::replace_message(&mut bytes, at, FixtureMessage::DATA_LAYOUT, &layout);
        fs::write(path, bytes).unwrap();
    }
    force_tail(path);
    let file = File::from_bytes(fs::read(path).unwrap()).unwrap();
    let DataLayout::Chunked {
        index: ChunkIndexLayout::ExtensibleArray { address, .. },
        ..
    } = file.dataset("d").unwrap().data_layout().unwrap()
    else {
        panic!("expected extensible array layout");
    };
    assert_eq!(address.is_none(), deferred);
}

// The index address is the last field of a version 4 chunked layout ("The Data Layout Message",
// format specification, version 4.0).
fn assert_index_address(bytes: &[u8], deferred: bool) {
    let layout = test_util::bytes::signature_offsets(bytes, v2::SIGNATURE)
        .into_iter()
        .find_map(|at| {
            ObjectHeader::parse(bytes, AccessMode::ReadOnly, at, 8, 8)
                .unwrap()
                .messages
                .into_iter()
                .find(|message| {
                    message.msg_type == MessageType::DATA_LAYOUT
                        && message
                            .data
                            .starts_with(&[data_layout::VERSION_FOUR, data_layout::CHUNKED_CLASS])
                })
                .map(|message| message.data)
        })
        .unwrap();
    let address = u64::from_le_bytes(layout[layout.len() - 8..].try_into().unwrap());
    assert_eq!(address == data_layout::UNDEFINED_ADDRESS, deferred);
}

fn inspect(bytes: &[u8], length: usize) -> Observation {
    let observation = Observation::read(bytes);
    assert_eq!(observation.values, values(0, length));
    let file = File::from_bytes(bytes.to_vec()).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().shape().unwrap(),
        [u64::try_from(length).unwrap()]
    );
    assert!(
        observation
            .reachable()
            .intersection(&observation.free)
            .is_empty()
    );
    observation
}

fn assert_activity(before: &Observation, after: &Observation, encoding: Encoding) {
    let new_chunks = after
        .chunks
        .iter()
        .filter(|chunk| !before.chunks.contains(chunk))
        .collect::<Vec<_>>();
    assert!(new_chunks.len() >= 9);
    assert!(new_chunks.iter().all(|chunk| chunk.1 >= before.eof));
    let indexes = after
        .live
        .iter()
        .filter(|fact| fact.kind == AllocationKind::Index && fact.range.start() >= before.eof)
        .collect::<Vec<_>>();
    assert!(!indexes.is_empty());
    assert!(
        after
            .live
            .iter()
            .any(|fact| fact.kind == AllocationKind::Manager)
    );
    if before.page_size.is_some() {
        assert!(!after.free.is_empty());
    }
    match encoding {
        #[cfg(feature = "deflate")]
        Encoding::Deflate => {
            let lengths = new_chunks
                .iter()
                .map(|chunk| chunk.2)
                .collect::<BTreeSet<_>>();
            assert!(lengths.len() >= 2, "stored lengths {lengths:?}");
            assert!(new_chunks.iter().all(|chunk| chunk.2 < PAGE_BYTES));
            if before.page_size.is_some() {
                assert!(
                    new_chunks
                        .windows(2)
                        .any(|chunks| chunks[0].1 / PAGE_BYTES == chunks[1].1 / PAGE_BYTES)
                );
                assert!(
                    new_chunks
                        .windows(2)
                        .any(|chunks| chunks[0].1 / PAGE_BYTES != chunks[1].1 / PAGE_BYTES)
                );
                let metadata = after
                    .live
                    .iter()
                    .filter(|fact| {
                        fact.kind == AllocationKind::Manager && fact.range.start() >= before.eof
                    })
                    .collect::<Vec<_>>();
                assert!(!metadata.is_empty());
                for manager in metadata {
                    assert!(
                        new_chunks
                            .iter()
                            .all(|chunk| manager.range.start() / PAGE_BYTES != chunk.1 / PAGE_BYTES)
                    );
                }
            }
        }
        Encoding::Fletcher32 => assert!(
            new_chunks
                .iter()
                .all(|chunk| chunk.2 == u64::try_from(CHUNK).unwrap() * 4 + 4)
        ),
    }
}

fn values(start: usize, length: usize) -> Vec<i32> {
    (start..start + length)
        .map(|value| i32::try_from(value).unwrap())
        .collect()
}

const PAGE_BYTES: u64 = 4096;
