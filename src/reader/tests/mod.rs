use std::io::Cursor;
use std::sync::atomic::AtomicUsize;

use ::test_util;
use rstest::rstest;
use test_util::disk_log::{self, DiskOp};
use test_util::file_space_info;
use test_util::object_header;
use test_util::temp;
use test_util::widths::Widths;

use super::*;
use crate::PhysicalAddress;
use crate::address::BaseAddressExt;
use crate::address::DecodedStoredAddress;
use crate::data_layout::{ChunkIndexLayout, ChunkedLayoutFlags};
use crate::dataspace::DataspaceType;
use crate::image::WriteBuffering;
use crate::source;
use crate::source::PhysicalBytes;

use crate::message_flags::MessageFlags;
use crate::{DatatypeByteOrder, FileBuilder};

mod opening;
mod storage;

#[rstest]
#[case::source(true)]
#[case::address(false)]
fn parsed_headers_reject_a_different_source_or_address(#[case] other_source: bool) {
    let mut builder = FileBuilder::new();
    builder.create_dataset("values").with_i32_data(&[7]);
    let bytes = builder.finish().unwrap();
    let file = File::from_bytes(bytes.clone()).unwrap();
    let other = File::from_bytes(bytes).unwrap();
    let address = PhysicalAddress::new(file.dataset("values").unwrap().header_address().unwrap());
    let (_, _, parsed) = file.hard_links_at(address).unwrap();
    let (recipient, address) = if other_source {
        (&other, address)
    } else {
        (
            &file,
            PhysicalAddress::new(file.root().header_address().unwrap()),
        )
    };
    let err = recipient
        .dataset_at(address, ObjectPathBuf::parse("values"), Some(parsed))
        .unwrap_err();
    let Error::Format(FormatError::Internal(reason)) = err else {
        panic!("expected Internal, got {err:?}");
    };
    assert_eq!(
        reason,
        "a parsed object header belongs to another source or address"
    );
}

#[test]
fn parsed_headers_read_metadata_again_after_an_in_place_change() {
    let path = temp::temp_path("source.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("values")
        .with_i32_data(&[])
        .with_chunks(&[2])
        .with_maxshape(&[MaxExtent::Unlimited]);
    builder.write(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    let mut dataset = file.dataset("values").unwrap();
    let address = PhysicalAddress::new(dataset.header_address().unwrap());
    let (_, _, parsed) = file.hard_links_at(address).unwrap();
    assert_eq!(dataset.shape().unwrap(), [0]);
    dataset.append(&[17_i32, 29]).unwrap();
    let fresh = file
        .dataset_at(address, ObjectPathBuf::parse("values"), Some(parsed))
        .unwrap();
    assert_eq!(fresh.shape().unwrap(), [2]);
    assert_eq!(fresh.read_i32().unwrap(), [17, 29]);
    drop(fresh);
    drop(dataset);
    file.close().unwrap();
}

#[test]
fn page_publication_failure_preserves_dataset_order_after_retry() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("publication.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Page));
    builder.create_dataset("keep").with_u8_data(&[7; 64]);
    builder.write(&path).unwrap();
    let file = File::open_page_publication_failure(
        &path,
        PagePublicationFailure::AfterPublish,
        SyncPolicy::OnClose,
        true,
    )
    .unwrap();
    file.root()
        .create_dataset("added", |dataset| {
            dataset.with_u8_data(&[9; 64]);
        })
        .unwrap();
    let error = file.commit().unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected injected publication failure, got {error:?}");
    };
    assert_eq!(error.kind(), std::io::ErrorKind::Other);
    assert_eq!(error.to_string(), "injected PAGE publication failure");
    assert_eq!(file.dataset("added").unwrap().read_u8().unwrap(), [9; 64]);
    file.root()
        .create_dataset("second", |dataset| {
            dataset.with_u8_data(&[13; 64]);
        })
        .unwrap();
    file.commit().unwrap();
    drop(file);
    let reopened = File::open(&path).unwrap();
    assert_eq!(
        reopened.root().datasets().unwrap(),
        ["keep", "added", "second"]
    );
    assert_eq!(
        reopened.dataset("keep").unwrap().read_u8().unwrap(),
        [7; 64]
    );
    assert_eq!(
        reopened.dataset("added").unwrap().read_u8().unwrap(),
        [9; 64]
    );
    assert_eq!(
        reopened.dataset("second").unwrap().read_u8().unwrap(),
        [13; 64]
    );
    drop(reopened);
    directory.close().unwrap();
}

#[test]
fn page_publication_snapshot_preserves_buffered_and_issued_states() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("snapshot.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Page));
    builder.create_dataset("keep").with_u8_data(&[7; 64]);
    builder.write(&path).unwrap();
    let original = std::fs::read(&path).unwrap();
    let file = File::open_page_publication_failure(
        &path,
        PagePublicationFailure::AfterDrain,
        SyncPolicy::OnClose,
        true,
    )
    .unwrap();
    file.root()
        .create_dataset("added", |dataset| {
            dataset.with_u8_data(&[9; 64]);
        })
        .unwrap();
    assert_eq!(file.page_publication_snapshot().unwrap(), original);
    let error = file.commit().unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected injected publication failure, got {error:?}");
    };
    assert_eq!(error.kind(), std::io::ErrorKind::Other);
    assert_eq!(error.to_string(), "injected PAGE publication failure");
    let published = file.page_publication_snapshot().unwrap();
    let snapshot = File::from_bytes(published.clone()).unwrap();
    assert_eq!(snapshot.root().datasets().unwrap(), ["keep", "added"]);
    assert_eq!(
        snapshot.dataset("keep").unwrap().read_u8().unwrap(),
        [7; 64]
    );
    assert_eq!(
        snapshot.dataset("added").unwrap().read_u8().unwrap(),
        [9; 64]
    );
    assert_eq!(file.page_publication_snapshot().unwrap(), published);
    drop(file);
    assert_eq!(std::fs::read(&path).unwrap(), published);
    drop(snapshot);
    directory.close().unwrap();
}

#[rstest]
#[case::index_before(false, 0, 0)]
#[case::extent_before(true, 0, 0)]
#[case::extent_after(true, usize::MAX, 509)]
fn buffered_replacement_failure_keeps_landed_prefix_and_live_storage(
    #[case] extent: bool,
    #[case] prefix: usize,
    #[case] landed: usize,
    #[values(false, true)] owned: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("buffered-retirement-failure.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
    );
    builder
        .create_dataset("d")
        .with_i32_data(&[0, 1, 2])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[512]);
    builder.write(&path).unwrap();
    let original = std::fs::read(&path).unwrap();
    let control_path = directory.path().join("control.h5");
    std::fs::write(&control_path, &original).unwrap();
    let control = File::open_rw(&control_path).unwrap();
    control
        .with_mirror_session(Change::Nothing, |engine| {
            engine.set_write_buffering(WriteBuffering::Unbuffered)
        })
        .unwrap();
    let index_offset =
        u64::try_from(test_util::bytes::find_signature(&original, EA_INDEX_SIGNATURE).unwrap())
            .unwrap();
    let values = (3..603).collect::<Vec<i32>>();
    disk_log::start();
    control
        .dataset("d")
        .unwrap()
        .append(&values[..509])
        .unwrap();
    let operations = disk_log::take();
    let dimension_offset = operations
        .iter()
        .find_map(|operation| match operation {
            DiskOp::Write { offset, bytes }
                if bytes
                    .windows(8)
                    .any(|window| window == 512_u64.to_le_bytes()) =>
            {
                Some(*offset)
            }
            _ => None,
        })
        .unwrap();
    let failure_write = operations
        .iter()
        .filter_map(|operation| match operation {
            DiskOp::Write { offset, .. } => Some(*offset),
            DiskOp::SetLen(_) => None,
        })
        .position(|offset| {
            if extent {
                offset == dimension_offset
            } else {
                offset >= index_offset && offset < index_offset + 200
            }
        })
        .unwrap();
    control.close().unwrap();
    let file = File::open_rw(&path).unwrap();
    file.with_mirror_session(Change::Nothing, |engine| {
        engine.set_write_buffering(WriteBuffering::Unbuffered)
    })
    .unwrap();
    let mut dataset = file.dataset("d").unwrap();
    let predecessor = dataset.chunks().unwrap().remove(0);
    let expected_unwritten = values[landed..]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    disk_log::fail_write_after(failure_write, prefix);
    let (error, unwritten, blocked) = if owned {
        let mut appender = dataset.into_buffered_appender().unwrap();
        let error = appender.append(&values).unwrap_err();
        disk_log::start();
        let unwritten = appender.unwritten().to_vec();
        let blocked = appender.flush().unwrap_err();
        assert_eq!(appender.discard(), unwritten);
        dataset = file.dataset("d").unwrap();
        (error, unwritten, blocked)
    } else {
        let mut appender = dataset.buffered_appender().unwrap();
        let error = appender.append(&values).unwrap_err();
        disk_log::start();
        let unwritten = appender.unwritten().to_vec();
        let blocked = appender.flush().unwrap_err();
        assert_eq!(appender.discard(), unwritten);
        (error, unwritten, blocked)
    };
    let Error::Io(error) = error else {
        panic!("expected injected write error, got {error:?}")
    };
    assert_eq!(error.kind(), std::io::ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    assert!(
        matches!(blocked, Error::AllocationPublicationUncertain),
        "{blocked:?}"
    );
    assert_eq!(unwritten, expected_unwritten);
    assert_eq!(
        dataset.read_i32().unwrap(),
        (0..i32::try_from(3 + landed).unwrap()).collect::<Vec<_>>()
    );
    file.with_mirror_session(Change::Nothing, |engine| {
        let ownership = engine.space_ownership()?;
        assert!(ownership.analysis_complete);
        let predecessor_ranges = if extent {
            &ownership.reusable_free_space
        } else {
            &ownership.live_raw_space
        };
        assert!(
            predecessor_ranges
                .iter()
                .any(|&(start, length)| start <= predecessor.address
                    && predecessor.address + predecessor.storage_size <= start + length)
        );
        for &(start, length) in &ownership.reusable_free_space {
            for &(address, size) in ownership
                .live_raw_space
                .iter()
                .chain(&ownership.live_metadata_space)
            {
                assert!(start + length <= address || address + size <= start);
            }
        }
        Ok(())
    })
    .unwrap();
    let interrupted = file.disk_bytes().unwrap();
    drop(dataset);
    let error = file.close().unwrap_err();
    assert!(
        matches!(error, Error::AllocationPublicationUncertain),
        "{error:?}"
    );
    assert!(disk_log::finish().events.is_empty());
    assert_eq!(std::fs::read(&path).unwrap(), interrupted);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        (0..i32::try_from(3 + landed).unwrap()).collect::<Vec<_>>()
    );
    drop(file);
    directory.close().unwrap();
}

#[rstest]
#[case::plain(0)]
#[case::userblock(4_096)]
fn sync_completes_pending_page_publication_and_refreshes_handles(
    #[case] userblock: u64,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] sync: SyncPolicy,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sync_page.h5");
    let mut builder = FileBuilder::new();
    builder
        .with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Page))
        .with_userblock(UserblockSize::try_from(userblock).unwrap());
    if userblock > 0 {
        builder.with_userblock_content(b"preserve this userblock through explicit sync");
    }
    builder.create_dataset("keep").with_u8_data(&[7; 64]);
    builder.create_dataset("replace").with_u8_data(&[3; 64]);
    builder.create_dataset("remove").with_u8_data(&[5; 64]);
    let mut group = builder.create_group("group");
    group.create_dataset("old").with_u8_data(&[19; 64]);
    builder.add_group(group.finish());
    builder.write(&path).unwrap();
    let original = std::fs::read(&path).unwrap();
    let control_path = dir.path().join("control.h5");
    std::fs::write(&control_path, &original).unwrap();
    let control = File::open_rw_with_options(
        &control_path,
        FileAccessProperties::new().with_sync_policy(sync),
    )
    .unwrap();
    stage_sync_page_edits(&control);
    disk_log::start();
    control.commit().unwrap();
    let operations = disk_log::take();
    let publication_write = operations
        .iter()
        .filter_map(|operation| match operation {
            DiskOp::Write { offset, .. } => Some(*offset),
            DiskOp::SetLen(_) => None,
        })
        .position(|offset| offset == userblock)
        .unwrap();
    drop(control);

    let file =
        File::open_rw_with_options(&path, FileAccessProperties::new().with_sync_policy(sync))
            .unwrap();
    let retained_dataset = file.dataset("replace").unwrap();
    let retained_group = file.group("group").unwrap();
    stage_sync_page_edits(&file);
    disk_log::start();
    disk_log::fail_write_after(publication_write, 17);
    let error = file.commit().unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected partial publication write failure, got {error:?}");
    };
    assert_eq!(error.kind(), std::io::ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    let mut issued = original.clone();
    for operation in disk_log::take() {
        operation.apply(&mut issued);
    }
    assert_eq!(issued, file.page_publication_snapshot().unwrap());
    assert_eq!(retained_dataset.read_u8().unwrap(), [3; 64]);
    assert_eq!(retained_group.datasets().unwrap(), ["old"]);
    file.root()
        .create_dataset("second", |dataset| {
            dataset.with_u8_data(&[11; 64]);
        })
        .unwrap();
    assert_eq!(retained_dataset.read_u8().unwrap(), [3; 64]);
    assert_eq!(retained_group.datasets().unwrap(), ["old"]);
    let by_address_dataset = FileInner::object_at_relative(
        &file.inner,
        file.inner.revisions(),
        file.addresses()
            .relative(PhysicalAddress::new(
                retained_dataset.header_address().unwrap(),
            ))
            .unwrap()
            .get(),
    )
    .unwrap();
    let Object::Dataset(by_address_dataset) = by_address_dataset else {
        panic!("expected dataset handle");
    };
    let by_address_group = FileInner::object_at_relative(
        &file.inner,
        file.inner.revisions(),
        file.addresses()
            .relative(PhysicalAddress::new(
                retained_group.header_address().unwrap(),
            ))
            .unwrap()
            .get(),
    )
    .unwrap();
    let Object::Group(by_address_group) = by_address_group else {
        panic!("expected group handle");
    };
    assert_eq!(by_address_dataset.read_u8().unwrap(), [3; 64]);
    assert_eq!(by_address_group.datasets().unwrap(), ["old"]);
    assert_pending_recovery_write_failure(&file, &retained_dataset, &retained_group, userblock, 0);
    assert_pending_recovery_write_failure(&file, &retained_dataset, &retained_group, userblock, 17);
    file.sync().unwrap();
    assert!(file.has_staged_edits());
    let published = file.page_publication_snapshot().unwrap();
    assert_sync_page_state(&File::from_bytes(published.clone()).unwrap(), false);
    assert_eq!(file.dataset("keep").unwrap().read_u8().unwrap(), [7; 64]);
    assert_eq!(
        file.dataset("replace").unwrap().read_u8().unwrap(),
        [17; 64]
    );
    assert_eq!(file.dataset("added").unwrap().read_u8().unwrap(), [9; 64]);
    let error = file.dataset("remove").unwrap_err();
    assert!(
        matches!(error, Error::Format(FormatError::PathNotFound(ref path)) if path == "remove"),
        "{error:?}"
    );
    let mut names = file.root().datasets().unwrap();
    names.sort_unstable();
    assert_eq!(names, ["added", "keep", "replace", "second"]);
    assert_eq!(retained_dataset.read_u8().unwrap(), [17; 64]);
    assert_eq!(retained_group.datasets().unwrap(), ["old", "child"]);
    assert_eq!(
        retained_group.dataset("child").unwrap().read_u8().unwrap(),
        [31; 64]
    );
    let error = by_address_dataset.read_u8().unwrap_err();
    assert!(matches!(error, Error::StaleHandle), "{error:?}");
    let error = by_address_group.datasets().unwrap_err();
    assert!(matches!(error, Error::StaleHandle), "{error:?}");
    let address_revision = file.inner.address_revision();
    let content_revision = file.inner.content_revision();
    file.sync().unwrap();
    assert_eq!(file.inner.address_revision(), address_revision);
    assert_eq!(file.inner.content_revision(), content_revision);
    assert_eq!(file.page_publication_snapshot().unwrap(), published);
    file.commit().unwrap();
    assert!(!file.has_staged_edits());
    assert_sync_page_state(&file, true);
    test_util::publication::assert_preserved_bytes(
        &original,
        &file.page_publication_snapshot().unwrap(),
        usize::try_from(userblock).unwrap(),
        hdf5_pure_format::__private::serialize_superblock(file.superblock())
            .unwrap()
            .len(),
    );
    file.close().unwrap();
}

fn assert_pending_recovery_write_failure(
    file: &File,
    dataset: &Dataset,
    group: &Group,
    publication_offset: u64,
    prefix: usize,
) {
    let address_revision = file.inner.address_revision();
    let content_revision = file.inner.content_revision();
    disk_log::start();
    disk_log::fail_write_after(0, prefix);
    let error = file.sync().unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected recovery write failure, got {error:?}");
    };
    assert_eq!(error.kind(), std::io::ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    let operations = disk_log::take();
    assert_eq!(operations.len(), usize::from(prefix > 0));
    for operation in operations {
        let DiskOp::Write { offset, bytes } = operation else {
            panic!("expected pending publication write");
        };
        assert_eq!(offset, publication_offset);
        assert_eq!(bytes.len(), prefix);
    }
    assert_eq!(file.inner.address_revision(), address_revision);
    assert_eq!(file.inner.content_revision(), content_revision);
    assert!(file.has_staged_edits());
    assert_eq!(dataset.read_u8().unwrap(), [3; 64]);
    assert_eq!(group.datasets().unwrap(), ["old"]);
}

fn stage_sync_page_edits(file: &File) {
    file.root()
        .create_dataset("added", |dataset| {
            dataset.with_u8_data(&[9; 64]);
        })
        .unwrap();
    file.dataset("replace")
        .unwrap()
        .write_staged(|dataset| {
            dataset.with_u8_data(&[17; 64]);
        })
        .unwrap();
    file.root().delete("remove").unwrap();
    file.group("group")
        .unwrap()
        .create_dataset("child", |dataset| {
            dataset.with_u8_data(&[31; 64]);
        })
        .unwrap();
}

fn assert_sync_page_state(file: &File, second: bool) {
    let mut expected = vec![("keep", 7), ("replace", 17), ("added", 9)];
    if second {
        expected.push(("second", 11));
    }
    let mut names = file.root().datasets().unwrap();
    names.sort_unstable();
    let mut expected_names = expected.iter().map(|(path, _)| *path).collect::<Vec<_>>();
    expected_names.sort_unstable();
    assert_eq!(names, expected_names);
    for (path, value) in expected {
        assert_eq!(file.dataset(path).unwrap().read_u8().unwrap(), [value; 64]);
    }
    assert_eq!(file.root().groups().unwrap(), ["group"]);
    assert_eq!(
        file.group("group").unwrap().datasets().unwrap(),
        ["old", "child"]
    );
    assert_eq!(
        file.dataset("group/old").unwrap().read_u8().unwrap(),
        [19; 64]
    );
    assert_eq!(
        file.dataset("group/child").unwrap().read_u8().unwrap(),
        [31; 64]
    );
    assert!(!file.file_space_info().unwrap().unwrap().persist);
    assert!(
        file.file_space_info()
            .unwrap()
            .unwrap()
            .manager_addrs
            .is_empty()
    );
    if matches!(&file.inner.backend, Backend::Edit(_)) {
        let error = file.persisted_free_sections().unwrap_err();
        assert!(
            matches!(error, Error::FreeSectionInspectionUnsupported),
            "{error:?}"
        );
    } else {
        assert_eq!(file.persisted_free_sections().unwrap(), None);
    }
}

#[rstest]
#[case::barrier(PagePublicationFailure::RepeatedPublishedSync)]
#[case::read(PagePublicationFailure::PublishedRead)]
fn failed_page_recovery_sync_preserves_handle_revisions_and_staged_edits(
    #[case] failure: PagePublicationFailure,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] sync: SyncPolicy,
    #[values(false, true)] gathered: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("failed_sync.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Page));
    builder.create_dataset("keep").with_u8_data(&[7; 64]);
    builder.write(&path).unwrap();
    let file = File::open_page_publication_failure(&path, failure, sync, gathered).unwrap();
    file.root()
        .create_dataset("added", |dataset| {
            dataset.with_u8_data(&[9; 64]);
        })
        .unwrap();
    let error = file.commit().unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected publication barrier failure, got {error:?}");
    };
    assert_eq!(error.kind(), std::io::ErrorKind::Other);
    assert_eq!(error.to_string(), "injected PAGE publication failure");
    file.root()
        .create_dataset("second", |dataset| {
            dataset.with_u8_data(&[11; 64]);
        })
        .unwrap();
    let retained = file.dataset("keep").unwrap();
    assert_eq!(retained.read_u8().unwrap(), [7; 64]);
    assert_failed_page_recovery_sync(&file, &retained, failure);
    assert_failed_page_recovery_sync(&file, &retained, failure);
}

fn assert_failed_page_recovery_sync(
    file: &File,
    retained: &Dataset,
    failure: PagePublicationFailure,
) {
    let address_revision = file.inner.address_revision();
    let content_revision = file.inner.content_revision();
    disk_log::start();
    let error = file.sync().unwrap_err();
    match (failure, error) {
        (PagePublicationFailure::RepeatedPublishedSync, Error::Io(error)) => {
            assert_eq!(error.kind(), std::io::ErrorKind::Other);
            assert_eq!(error.to_string(), "injected PAGE publication failure");
        }
        (PagePublicationFailure::PublishedRead, Error::Format(FormatError::Source(reason))) => {
            assert_eq!(reason, "injected PAGE publication read failure");
        }
        (_, error) => panic!("expected PAGE recovery failure, got {error:?}"),
    }
    assert!(disk_log::take().is_empty());
    assert_eq!(file.inner.address_revision(), address_revision);
    assert_eq!(file.inner.content_revision(), content_revision);
    assert!(file.has_staged_edits());
    assert_eq!(retained.read_u8().unwrap(), [7; 64]);
    assert_eq!(file.dataset("second").unwrap().shape().unwrap(), [64]);
}

#[rstest]
fn ordinary_sync_preserves_reference_handles_and_staged_edits(
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] sync: SyncPolicy,
    #[values(MemoryStrategy::Bounded, MemoryStrategy::Mirrored)] memory: MemoryStrategy,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ordinary_sync.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("keep").with_u8_data(&[7; 64]);
    builder
        .create_dataset("refs")
        .with_path_references(&["keep"]);
    builder.write(&path).unwrap();
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_sync_policy(sync)
            .with_memory_strategy(memory),
    )
    .unwrap();
    let reference = file
        .dataset("refs")
        .unwrap()
        .dereference()
        .unwrap()
        .remove(0);
    let Object::Dataset(reference) = reference else {
        panic!("expected dataset reference");
    };
    assert_eq!(reference.read_u8().unwrap(), [7; 64]);
    let address_revision = file.inner.address_revision();
    let content_revision = file.inner.content_revision();
    disk_log::start();
    file.sync().unwrap();
    assert!(disk_log::take().is_empty());
    assert_eq!(file.inner.address_revision(), address_revision);
    assert_eq!(file.inner.content_revision(), content_revision);
    assert_eq!(reference.read_u8().unwrap(), [7; 64]);
    file.root()
        .create_dataset("later", |dataset| {
            dataset.with_u8_data(&[29; 64]);
        })
        .unwrap();
    file.sync().unwrap();
    assert!(file.has_staged_edits());
    let issued = File::from_bytes(file.page_publication_snapshot().unwrap()).unwrap();
    assert_eq!(issued.root().datasets().unwrap(), ["keep", "refs"]);
    assert_eq!(issued.dataset("keep").unwrap().read_u8().unwrap(), [7; 64]);
    file.close().unwrap();
}

#[rstest]
#[case::plain(0, 0)]
#[case::userblock(512, 512)]
#[case::jammed(0, 512)]
#[case::unjammed(512, 0)]
fn version_zero_file_space_eoa_is_invariant_under_relocation(
    #[case] stored_base: usize,
    #[case] signature_offset: usize,
    #[values(false, true)] source_backed: bool,
) {
    let mut builder = FileBuilder::new();
    builder
        .with_userblock(UserblockSize::try_from(stored_base as u64).unwrap())
        .with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
                .with_persistence(true)
                .with_threshold(64),
        );
    builder.create_dataset("data").with_i32_data(&[1, 2, 3]);
    let mut original = builder.finish().unwrap();
    let canonical = File::from_bytes(original.clone()).unwrap();
    let extension_at = canonical
        .addresses()
        .resolve(canonical.superblock().superblock_extension_address.unwrap())
        .unwrap()
        .get();
    let body = file_space_info::version_0(
        Widths::EIGHT,
        file_space_info::ALL_PERSIST,
        64,
        &[u64::MAX; file_space_info::VERSION_0_MANAGERS],
    );
    object_header::v2::replace_message(
        &mut original,
        usize::try_from(extension_at).unwrap(),
        object_header::MessageType::FILE_SPACE_INFO,
        &body,
    );
    let mut relocated = vec![0; signature_offset];
    relocated.extend_from_slice(&original[stored_base..]);
    let file = if source_backed {
        File::from_source(BytesSource::new(relocated)).unwrap()
    } else {
        File::from_bytes(relocated).unwrap()
    };
    let info = file.file_space_info().unwrap().unwrap();
    assert_eq!(
        file.file_space_settings().unwrap(),
        Some(
            FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
                .with_persistence(true)
                .with_threshold(64)
        )
    );
    assert_eq!(info.eoa_pre_fsm, None);
    assert_eq!(info.strategy, FileSpaceStrategy::FsmAggr);
    assert!(info.persist);
    assert_eq!(info.threshold, 64);
    assert_eq!(
        info.manager_addrs.len(),
        file_space_info::VERSION_0_MANAGERS
    );
    assert!(
        info.manager_addrs
            .iter()
            .all(|&a| a == DecodedStoredAddress::Undefined)
    );
    assert_eq!(file.dataset("data").unwrap().read_i32().unwrap(), [1, 2, 3]);
}

// -----------------------------------------------------------------------
// Reporting the metadata cache (issue #353)
// -----------------------------------------------------------------------

#[rstest]
#[case::owned_adapter(false)]
#[case::shared_owner(true)]
fn the_source_view_reports_the_cache_it_reads_through(#[case] shared_owner: bool) {
    let backend = MetadataCachingSource::new(
        ReadSeekSource::new(Cursor::new((0..=255u8).collect::<Vec<u8>>())).unwrap(),
        MetadataCacheConfig::new(4096),
    );
    let view = SourceView::Stream(&backend);

    assert_eq!(view.metadata_cache_stats().unwrap().reads(), 0);
    let expected: Vec<u8> = (0..64).collect();
    if shared_owner {
        let first = view.read_metadata(0, 64).unwrap();
        let second = view.read_metadata(0, 64).unwrap();
        assert_eq!(first.as_ref(), expected);
        assert_eq!(second.as_ref(), expected);
        assert_eq!(first.as_ptr(), second.as_ptr());
    } else {
        assert_eq!(view.read_metadata_at(0, 64).unwrap(), expected);
        assert_eq!(view.read_metadata_at(0, 64).unwrap(), expected);
    }
    let stats = view
        .metadata_cache_stats()
        .expect("the backend has a cache, so the view reports it");
    assert_eq!((stats.hits(), stats.misses()), (1, 1));

    view.reset_metadata_cache_stats();
    let cleared = view.metadata_cache_stats().unwrap();
    assert_eq!(cleared.hits(), 0);
    assert_eq!(cleared.entries(), 1, "a reset evicts nothing");

    // The memory backend has no metadata cache.
    assert_eq!(SourceView::Mem(&[0u8; 16]).metadata_cache_stats(), None);
}

// -----------------------------------------------------------------------
// Handle re-validation across an edit (issue #351)
// -----------------------------------------------------------------------

/// Writes two chunked datasets with partial tails, a contiguous dataset, and a subgroup.
fn revalidation_fixture(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    for ds in ["log", "other"] {
        b.create_dataset(ds)
            .with_i32_data(&[0, 1])
            .with_shape(&[2])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[4]);
    }
    b.create_dataset("plain").with_i32_data(&[7, 8, 9]);
    let g = b.create_group("g");
    b.add_group(g.finish());
    b.write(path).unwrap();
}

#[test]
fn a_handle_follows_its_object_across_a_commit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("follow.h5");
    revalidation_fixture(&path);

    let file = File::open_rw(&path).unwrap();
    let mut ds = file.dataset("plain").unwrap();
    let group = file.group("g").unwrap();

    // The attribute and child edits relocate both object headers.
    ds.set_attr("units", AttrValue::AsciiString("m".into()))
        .unwrap();
    group
        .create_dataset("child", |b| {
            b.with_i32_data(&[4, 5]);
        })
        .unwrap();
    file.commit().unwrap();

    assert_eq!(
        sorted(ds.attrs()),
        vec![r#"units=AsciiString("m")"#.to_string()],
        "the dataset handle must report the attribute the commit added"
    );
    assert_eq!(
        group.datasets().unwrap(),
        vec!["child".to_string()],
        "the group handle must report the child the commit added"
    );
    assert_eq!(ds.read_i32().unwrap(), vec![7, 8, 9]);
    assert_eq!(
        group.dataset("child").unwrap().read_i32().unwrap(),
        vec![4, 5]
    );
    file.close().unwrap();
}

// The appended elements share a chunk with the reader's cached values.
#[test]
fn an_append_through_one_handle_is_what_another_reads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two_handles.h5");
    revalidation_fixture(&path);

    let file = File::open_rw(&path).unwrap();
    let reader = file.dataset("log").unwrap();
    let mut writer = file.dataset("log").unwrap();

    assert_eq!(reader.read_i32().unwrap(), vec![0, 1]);
    assert!(
        reader.chunk_cache_stats().cached_chunks() > 0,
        "the read must leave the partial trailing chunk cached, or this \
         test cannot tell a retained chunk from a re-read one"
    );

    writer.append(&[2i32, 3]).unwrap();

    assert_eq!(
        reader.chunk_cache_stats().cached_chunks(),
        0,
        "the snapshot must report what the cache holds for a read, and the \
         append left it holding nothing a read will be served"
    );
    assert_eq!(reader.shape().unwrap(), vec![4]);
    assert_eq!(reader.read_i32().unwrap(), vec![0, 1, 2, 3]);
    file.close().unwrap();
}

// Cloned handles share their chunk cache.
#[test]
fn a_clone_is_a_second_handle_to_the_same_object() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("clone.h5");
    revalidation_fixture(&path);

    let file = File::open_rw(&path).unwrap();
    let mut ds = file.dataset("log").unwrap();
    let copy = ds.clone();
    let root = file.root();
    let root_copy = root.clone();

    assert_eq!(copy.read_i32().unwrap(), vec![0, 1]);
    assert!(copy.chunk_cache_stats().cached_chunks() > 0);
    assert_eq!(
        ds.chunk_cache_stats().cached_chunks(),
        copy.chunk_cache_stats().cached_chunks(),
        "clones share one cache: a chunk read through either is warm for both"
    );

    ds.append(&[2i32, 3]).unwrap();
    assert_eq!(copy.read_i32().unwrap(), vec![0, 1, 2, 3]);

    root.create_group("later").unwrap();
    file.commit().unwrap();
    assert!(
        root_copy.groups().unwrap().contains(&"later".to_string()),
        "a cloned group handle follows its group across a commit too"
    );
    file.close().unwrap();
}

#[test]
fn a_handle_to_a_deleted_object_refuses_rather_than_reading_what_it_left() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("deleted.h5");
    revalidation_fixture(&path);

    let file = File::open_rw(&path).unwrap();
    let ds = file.dataset("plain").unwrap();
    assert_eq!(ds.read_i32().unwrap(), vec![7, 8, 9]);

    file.root().delete("plain").unwrap();
    file.commit().unwrap();

    assert!(
        matches!(
            ds.read_i32(),
            Err(Error::Format(FormatError::PathNotFound(ref p))) if p == "plain"
        ),
        "reading a deleted dataset must fail the way opening it does, got {:?}",
        ds.read_i32()
    );
    file.close().unwrap();
}

#[test]
fn a_reference_handle_reads_until_a_commit_could_have_moved_its_object() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("by_ref.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("log")
        .with_i32_data(&[0, 1])
        .with_shape(&[2])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[4]);
    b.create_dataset("refs").with_path_references(&["log"]);
    b.write(&path).unwrap();

    let file = File::open_rw(&path).unwrap();
    let mut by_ref = match file
        .dataset("refs")
        .unwrap()
        .dereference()
        .unwrap()
        .remove(0)
    {
        Object::Dataset(ds) => *ds,
        other => panic!("expected a dataset, got {other:?}"),
    };
    assert_eq!(by_ref.read_i32().unwrap(), vec![0, 1]);

    // An in-place append rewrites the header where it stands, so the address
    // is still the object's and the handle reads its own new elements.
    by_ref.append(&[2i32, 3]).unwrap();
    assert_eq!(by_ref.read_i32().unwrap(), vec![0, 1, 2, 3]);

    // A commit can put the header somewhere else, and nothing on disk marks
    // the bytes it vacated as dead.
    file.root().create_group("g").unwrap();
    file.commit().unwrap();
    assert!(
        matches!(by_ref.read_i32(), Err(Error::StaleHandle)),
        "unexpected: {:?}",
        by_ref.read_i32()
    );

    // And the recovery the error names: dereference again, against the file
    // the commit left.
    let fresh = match file
        .dataset("refs")
        .unwrap()
        .dereference()
        .unwrap()
        .remove(0)
    {
        Object::Dataset(ds) => *ds,
        other => panic!("expected a dataset, got {other:?}"),
    };
    assert_eq!(fresh.read_i32().unwrap(), vec![0, 1, 2, 3]);
    file.close().unwrap();
}

// `Relocating` advances both revisions, `InPlace` content only, and `Nothing` neither.
#[test]
fn every_write_entry_point_declares_what_it_changes() {
    let dir = tempfile::tempdir().unwrap();

    type Step = (&'static str, Change, fn(&File));
    let steps: Vec<Step> = vec![
        ("File::sync", Change::Nothing, |f| {
            f.sync().unwrap();
        }),
        ("Dataset::chunk_cache_stats", Change::Nothing, |f| {
            let _ = f.dataset("log").unwrap().chunk_cache_stats();
        }),
        ("Dataset::reset_chunk_cache_stats", Change::Nothing, |f| {
            f.dataset("log").unwrap().reset_chunk_cache_stats();
        }),
        ("BufferedAppender::new", Change::Nothing, |f| {
            let mut ds = f.dataset("log").unwrap();
            let app = ds.buffered_appender().unwrap();
            app.discard();
        }),
        ("Dataset::append", Change::InPlace, |f| {
            f.dataset("log").unwrap().append(&[9i32]).unwrap();
        }),
        ("Dataset::append_raw", Change::InPlace, |f| {
            f.dataset("log")
                .unwrap()
                .append_raw(&7i32.to_le_bytes())
                .unwrap();
        }),
        ("BufferedAppender::flush", Change::InPlace, |f| {
            let mut ds = f.dataset("log").unwrap();
            let mut app = ds.buffered_appender().unwrap();
            app.append(&[5i32, 6, 7, 8]).unwrap();
            app.flush().unwrap();
        }),
        ("Dataset::write", Change::Relocating, |f| {
            f.dataset("plain").unwrap().write(&[1i32, 2, 3]).unwrap();
        }),
        ("Dataset::set_attr", Change::Relocating, |f| {
            f.dataset("plain")
                .unwrap()
                .set_attr("a", AttrValue::I32(1))
                .unwrap();
        }),
        ("Dataset::remove_attr", Change::Relocating, |f| {
            let mut ds = f.dataset("plain").unwrap();
            ds.set_attr("gone", AttrValue::I32(1)).unwrap();
            f.commit().unwrap();
            ds.remove_attr("gone").unwrap();
        }),
        ("Dataset::write_staged", Change::Relocating, |f| {
            f.dataset("plain")
                .unwrap()
                .write_staged(|b| {
                    b.with_i32_data(&[4, 5, 6]);
                })
                .unwrap();
        }),
        ("Dataset::append_staged", Change::Relocating, |f| {
            f.dataset("log")
                .unwrap()
                .append_staged(|b| {
                    b.append_i32(&[3]);
                })
                .unwrap();
        }),
        ("Group::create_group", Change::Relocating, |f| {
            f.root().create_group("fresh").unwrap();
        }),
        ("Group::create_dataset", Change::Relocating, |f| {
            f.root()
                .create_dataset("made", |b| {
                    b.with_i32_data(&[1]);
                })
                .unwrap();
        }),
        ("Group::delete", Change::Relocating, |f| {
            f.root().delete("plain").unwrap();
        }),
        ("Group::set_attr", Change::Relocating, |f| {
            f.root().set_attr("a", AttrValue::I32(1)).unwrap();
        }),
        ("Group::remove_attr", Change::Relocating, |f| {
            f.root().set_attr("gone", AttrValue::I32(1)).unwrap();
            f.commit().unwrap();
            f.root().remove_attr("gone").unwrap();
        }),
        ("Group::create_group_with", Change::Relocating, |f| {
            f.root()
                .create_group_with("built", |g| {
                    g.set_attr("a", AttrValue::I32(1));
                })
                .unwrap();
        }),
        ("File::copy", Change::Relocating, |f| {
            f.copy("plain", "copied").unwrap();
        }),
        ("File::commit", Change::Relocating, |f| {
            f.root().create_group("committed").unwrap();
            f.commit().unwrap();
        }),
    ];

    let revisions = |f: &File| (f.inner.content_revision(), f.inner.address_revision());
    let declared = |c: Change| match c {
        Change::Relocating => "Relocating",
        Change::InPlace => "InPlace",
        Change::Nothing => "Nothing",
    };
    for (name, change, step) in steps {
        let path = dir.path().join(format!("{}.h5", name.replace("::", "_")));
        revalidation_fixture(&path);
        let file = File::open_rw(&path).unwrap();
        let before = revisions(&file);
        step(&file);
        let after = revisions(&file);
        let observed = match (after.0 > before.0, after.1 > before.1) {
            (true, true) => "Relocating",
            (true, false) => "InPlace",
            (false, false) => "Nothing",
            (false, true) => "an address move with no content change",
        };
        assert_eq!(
            observed,
            declared(change),
            "{name} is declared here as one thing and behaves as another \
             ({before:?} -> {after:?})"
        );
        file.close().unwrap();
    }

    // `close` consumes the file, so it cannot be a row above. It is two
    // operations: the commit it makes, which relocates like any other, and
    // the teardown, which re-homes free space and releases status flags
    // where they stand. Both counters therefore move, and the content one
    // moves twice.
    let path = dir.path().join("File_close.h5");
    revalidation_fixture(&path);
    let file = File::open_rw(&path).unwrap();
    let before = revisions(&file);
    let by_path = file.dataset("plain").unwrap();
    let by_ref = {
        let mut b = FileBuilder::new();
        b.create_dataset("d").with_i32_data(&[1, 2]);
        b.create_dataset("refs").with_path_references(&["d"]);
        let refs_path = dir.path().join("File_close_refs.h5");
        b.write(&refs_path).unwrap();
        let refs = File::open_rw(&refs_path).unwrap();
        let handle = match refs
            .dataset("refs")
            .unwrap()
            .dereference()
            .unwrap()
            .remove(0)
        {
            Object::Dataset(ds) => *ds,
            other => panic!("expected a dataset, got {other:?}"),
        };
        refs.close().unwrap();
        handle
    };
    let inner = Arc::clone(&file.inner);
    file.close().unwrap();
    let after = (inner.content_revision(), inner.address_revision());
    assert_eq!(
        (after.0 - before.0, after.1 - before.1),
        (2, 1),
        "File::close is a Relocating commit and an InPlace teardown"
    );
    assert_eq!(
        by_path.read_i32().unwrap(),
        vec![7, 8, 9],
        "`close` promises reads through surviving handles still work"
    );
    assert!(
        matches!(by_ref.read_i32(), Err(Error::StaleHandle)),
        "the one handle that cannot follow the commit `close` makes: {:?}",
        by_ref.read_i32()
    );
}

#[rstest::rstest]
#[case::same_dataset(|file: &File| { file.dataset("log").unwrap().append(&[9i32]).unwrap(); }, &[0, 1, 9])]
#[case::other_dataset(|file: &File| { file.dataset("other").unwrap().append(&[9i32]).unwrap(); }, &[0, 1])]
#[case::sync(|file: &File| { file.sync().unwrap(); }, &[0, 1])]
fn an_edit_that_moves_no_header_leaves_a_reference_handle_reading(
    #[case] edit: fn(&File),
    #[case] expected: &[i32],
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("reference.h5");
    let mut b = FileBuilder::new();
    for ds in ["log", "other"] {
        b.create_dataset(ds)
            .with_i32_data(&[0, 1])
            .with_shape(&[2])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[4]);
    }
    b.create_dataset("refs").with_path_references(&["log"]);
    b.write(&path).unwrap();

    let file = File::open_rw(&path).unwrap();
    let by_ref = match file
        .dataset("refs")
        .unwrap()
        .dereference()
        .unwrap()
        .remove(0)
    {
        Object::Dataset(ds) => *ds,
        other => panic!("expected a dataset, got {other:?}"),
    };
    assert_eq!(by_ref.read_i32().unwrap(), vec![0, 1], "before the edit");
    edit(&file);
    assert_eq!(by_ref.read_i32().unwrap(), expected);
    drop(by_ref);
    file.close().unwrap();
}

#[test]
fn a_handle_whose_path_becomes_a_group_keeps_refusing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("replaced.h5");
    revalidation_fixture(&path);

    let file = File::open_rw(&path).unwrap();
    let ds = file.dataset("plain").unwrap();
    assert_eq!(ds.read_i32().unwrap(), vec![7, 8, 9]);

    file.root().delete("plain").unwrap();
    file.root()
        .create_group_with("plain", |g| {
            g.set_attr("i_am_a_group", AttrValue::I32(42));
        })
        .unwrap();
    file.commit().unwrap();

    for call in 1..=3 {
        assert!(
            matches!(ds.attrs(), Err(Error::NotADataset(ref p)) if p == "plain"),
            "call {call} answered {:?}",
            ds.attrs()
        );
        assert!(matches!(ds.read_i32(), Err(Error::NotADataset(ref path)) if path == "plain"));
        assert!(matches!(ds.shape(), Err(Error::NotADataset(ref path)) if path == "plain"));
    }
    file.close().unwrap();
}

#[test]
fn reading_never_moves_the_file_on() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("readonly.h5");
    revalidation_fixture(&path);

    let file = File::open(&path).unwrap();
    let _ = read_everything(&file);
    let ds = file.dataset("log").unwrap();
    let _ = ds.read_i32().unwrap();
    let _ = ds.attrs().unwrap();
    let _ = ds.layout().unwrap();
    let _ = file.root().groups().unwrap();
    assert_eq!(
        (file.inner.content_revision(), file.inner.address_revision()),
        (0, 0),
        "a read must not tell every handle its memo has expired"
    );
}

// -----------------------------------------------------------------------
// Opening an object as the wrong kind (issue #352)
// -----------------------------------------------------------------------

/// Builds a root dataset and a group with a dataset child.
fn nested_dataset_bytes() -> Vec<u8> {
    let mut b = FileBuilder::new();
    b.create_dataset("plain").with_i32_data(&[1]);
    let mut g = b.create_group("g");
    g.create_dataset("inner").with_i32_data(&[2]);
    b.add_group(g.finish());
    b.finish().unwrap()
}

#[test]
fn opening_a_dataset_as_a_group_is_refused() {
    let file = File::from_bytes(nested_dataset_bytes()).unwrap();
    let nested = file.group("g").unwrap();

    for (label, named, got) in [
        ("File::group", "plain", file.group("plain")),
        ("Group::group", "plain", file.root().group("plain")),
        ("File::group nested", "g/inner", file.group("g/inner")),
        (
            "Group::group from a subgroup",
            "g/inner",
            nested.group("inner"),
        ),
    ] {
        assert!(
            matches!(&got, Err(Error::NotAGroup(p)) if p == named),
            "{label} answered {:?}",
            got.map(|_| "a group")
        );
    }

    // Error paths use normalized object names.
    assert!(matches!(file.group("/plain/"), Err(Error::NotAGroup(ref p)) if p == "plain"));

    // A name that resolves to nothing stays distinct from one that resolves
    // to the wrong kind: the second reports what is there.
    assert!(matches!(
        file.group("absent"),
        Err(Error::Format(FormatError::PathNotFound(ref path))) if path == "absent"
    ));
    assert!(matches!(
        file.root().group("absent"),
        Err(Error::Format(FormatError::PathNotFound(ref path))) if path == "absent"
    ));

    // And a real group still opens, by either form.
    assert_eq!(file.group("g").unwrap().datasets().unwrap(), ["inner"]);
    assert_eq!(
        file.root().group("g").unwrap().datasets().unwrap(),
        ["inner"]
    );
}

// The independent image uses legacy symbol-table groups.
#[test]
fn a_symbol_table_group_still_opens_by_name() {
    let file = File::from_bytes(test_util::legacy::two_groups()).unwrap();

    // `group1` contains `values`, while its sibling contains `temps`.
    for lookup in [file.group("group1"), file.root().group("group1")] {
        assert_eq!(lookup.unwrap().datasets().unwrap(), ["values"]);
    }
}

#[test]
fn opening_a_named_datatype_as_a_group_is_refused() {
    let mut b = FileBuilder::new();
    b.commit_datatype("mytype", crate::make_i32_type());
    let file = File::from_bytes(b.finish().unwrap()).unwrap();

    assert_eq!(file.root().named_datatypes().unwrap(), vec!["mytype"]);
    assert!(matches!(file.group("mytype"), Err(Error::NotAGroup(ref path)) if path == "mytype"));
    assert!(matches!(
        file.root().group("mytype"),
        Err(Error::NotAGroup(ref path)) if path == "mytype"
    ));
}

// -----------------------------------------------------------------------
// Opening something else as a named datatype (issue #364)
// -----------------------------------------------------------------------

/// Builds a synthetic header with message types from `types` and empty message bodies.
fn header_of(types: &[MessageType]) -> ObjectHeader {
    ObjectHeader {
        version: 2,
        messages: types
            .iter()
            .map(|&msg_type| crate::object_header::HeaderMessage {
                msg_type,
                size: 0,
                flags: MessageFlags::NONE,
                creation_order: None,
                data: Vec::new(),
            })
            .collect(),
        reference_count: None,
        flags: 0,
        access_time: None,
        modification_time: None,
        change_time: None,
        birth_time: None,
    }
}

// Mixed and empty headers exercise the object-kind predicate directly.
#[test]
fn a_committed_datatype_is_a_datatype_that_is_neither_dataset_nor_group() {
    for (label, types, expected) in [
        ("a committed datatype", &[MessageType::DATATYPE][..], true),
        (
            "a dataset, whose element type is a datatype message too",
            &[MessageType::DATATYPE, MessageType::DATA_LAYOUT],
            false,
        ),
        (
            "a group with a link table",
            &[MessageType::LINK_INFO],
            false,
        ),
        (
            "a group with a symbol table",
            &[MessageType::SYMBOL_TABLE],
            false,
        ),
        (
            "a group carrying a datatype",
            &[MessageType::LINK_INFO, MessageType::DATATYPE],
            false,
        ),
        (
            "a header with no datatype at all",
            &[MessageType::DATASPACE],
            false,
        ),
    ] {
        assert_eq!(is_named_datatype(&header_of(types)), expected, "{label}");
    }
}

#[test]
fn a_child_that_is_not_a_committed_datatype_is_refused_by_name() {
    let mut b = FileBuilder::new();
    b.commit_datatype("mytype", crate::make_i32_type());
    b.create_dataset("typed")
        .with_i32_data(&[1, 2, 3])
        .with_committed_datatype("mytype");
    b.create_dataset("plain").with_f64_data(&[1.0]);
    let g = b.create_group("g").finish();
    b.add_group(g);
    let file = File::from_bytes(b.finish().unwrap()).unwrap();
    let root = file.root();

    assert_eq!(root.named_datatypes().unwrap(), ["mytype"]);

    // Both entry points reject plain datasets, datasets using the committed type, and groups.
    for name in ["typed", "plain", "g"] {
        let got = root.named_datatype(name);
        assert!(
            matches!(&got, Err(Error::NotANamedDatatype(p)) if p == name),
            "named_datatype({name:?}) answered {got:?}"
        );
        let got = root.named_datatype_references(name);
        assert!(
            matches!(&got, Err(Error::NotANamedDatatype(p)) if p == name),
            "named_datatype_references({name:?}) answered {got:?}"
        );
    }

    // A name that reaches nothing stays distinct from one that reaches the
    // wrong kind, as it is for `group` and `dataset`.
    assert!(matches!(
        root.named_datatype("absent"),
        Err(Error::Format(FormatError::PathNotFound(ref path))) if path == "absent"
    ));
    assert!(matches!(
        root.named_datatype_references("absent"),
        Err(Error::Format(FormatError::PathNotFound(ref path))) if path == "absent"
    ));

    // A committed type remains readable through both entry points.
    assert_eq!(
        root.named_datatype("mytype").unwrap(),
        crate::make_i32_type()
    );
    assert_eq!(root.named_datatype_references("mytype").unwrap(), 2);
}

#[test]
fn a_group_handle_whose_path_becomes_a_dataset_keeps_refusing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("replaced_group.h5");
    revalidation_fixture(&path);

    let file = File::open_rw(&path).unwrap();
    let group = file.group("g").unwrap();
    assert!(group.datasets().unwrap().is_empty());

    file.root().delete("g").unwrap();
    file.root()
        .create_dataset("g", |b| {
            b.with_i32_data(&[7]);
        })
        .unwrap();
    file.commit().unwrap();

    for call in 1..=3 {
        assert!(
            matches!(group.attrs(), Err(Error::NotAGroup(ref p)) if p == "g"),
            "call {call} answered {:?}",
            group.attrs()
        );
    }
    assert!(matches!(group.datasets(), Err(Error::NotAGroup(ref path)) if path == "g"));
    assert!(matches!(group.groups(), Err(Error::NotAGroup(ref path)) if path == "g"));
    assert!(matches!(
        group.dataset("anything"),
        Err(Error::NotAGroup(ref path)) if path == "g"
    ));

    // The path still names something, and that something still opens as
    // what it now is.
    assert_eq!(file.dataset("g").unwrap().read_i32().unwrap(), vec![7]);
    file.close().unwrap();
}

// -----------------------------------------------------------------------
// Resolving a path *through* something that is not a group (issue #365)
// -----------------------------------------------------------------------

// Errors report the complete prefix through the first non-group component.
#[rstest]
#[case::plain_group("plain/sub", "plain", PathLookup::Group)]
#[case::plain_dataset("plain/sub", "plain", PathLookup::Dataset)]
#[case::nested_group("g/inner/deeper", "g/inner", PathLookup::Group)]
#[case::nested_dataset("g/inner/deeper", "g/inner", PathLookup::Dataset)]
fn a_path_through_a_non_group_reports_the_object_that_stopped_it(
    #[case] asked: &str,
    #[case] stopper: &str,
    #[case] lookup: PathLookup,
) {
    let file = File::from_bytes(nested_dataset_bytes()).unwrap();
    let result = match lookup {
        PathLookup::Dataset => file.dataset(asked).map(|_| ()),
        PathLookup::Group => file.group(asked).map(|_| ()),
    };
    assert!(
        matches!(&result, Err(Error::NotAGroup(path)) if path == stopper),
        "{asked:?}: {result:?}"
    );

    assert!(matches!(
        file.group("absent/sub"),
        Err(Error::Format(FormatError::PathNotFound(ref p))) if p == "absent"
    ));

    // Empty components are dropped before the walk, so the object is named
    // the same way however the path was spelled.
    assert!(matches!(
        file.group("/plain//sub/"),
        Err(Error::NotAGroup(ref p)) if p == "plain"
    ));

    // The name reaches a caller that only prints the error, too.
    let printed = file
        .group("g/inner/deeper")
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert_eq!(printed, "not a group: g/inner");

    assert_eq!(file.dataset("g/inner").unwrap().read_i32().unwrap(), [2]);

    // A committed datatype is neither a dataset nor a group, so it separates
    // "is not a group" from "is a dataset" for an intermediate component the
    // way it does for a final one.
    let mut b = FileBuilder::new();
    b.commit_datatype("mytype", crate::make_i32_type());
    let typed = File::from_bytes(b.finish().unwrap()).unwrap();
    assert!(matches!(
        typed.group("mytype/sub"),
        Err(Error::NotAGroup(ref p)) if p == "mytype"
    ));
}

enum PathLookup {
    Dataset,
    Group,
}

#[test]
fn the_streaming_walk_names_the_object_that_stopped_it_too() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nested.h5");
    std::fs::write(&path, nested_dataset_bytes()).unwrap();

    let file = File::open_streaming(&path).unwrap();
    for (asked, stopper) in [("plain/sub", "plain"), ("g/inner/deeper", "g/inner")] {
        let got = file.group(asked).map(|_| ());
        assert!(
            matches!(&got, Err(Error::NotAGroup(p)) if p == stopper),
            "group({asked:?}) answered {got:?}"
        );
    }
    assert!(matches!(
        file.group("absent/sub"),
        Err(Error::Format(FormatError::PathNotFound(ref p))) if p == "absent"
    ));
    assert_eq!(file.dataset("g/inner").unwrap().read_i32().unwrap(), [2]);
    // Before the directory goes: an open file blocks its removal on Windows.
    drop(file);
}

#[test]
fn a_root_that_is_not_a_group_reports_an_empty_path() {
    let mut bytes = nested_dataset_bytes();
    let LocatedSuperblock {
        address,
        superblock: mut sb,
        ..
    } = crate::superblock::read(bytes.as_slice()).unwrap();
    let sig = usize::try_from(address.get()).unwrap();
    assert_eq!(sb.base_address, BaseAddress::ZERO);
    let addresses = AddressContext::from_effective_base(sb.base_address);
    sb.root_group_address = addresses
        .relative(PhysicalAddress::new(
            group_v2::resolve_read_path_any(
                PhysicalBytes::for_test(
                    &bytes,
                    AddressContext::from_effective_base(sb.base_address),
                ),
                AccessMode::ReadOnly,
                &sb,
                FormatWidths {
                    offsets: sb.offset_size,
                    lengths: sb.length_size,
                },
                &ObjectPath::parse("plain"),
            )
            .unwrap(),
        ))
        .unwrap();
    let rewritten = hdf5_pure_format::__private::serialize_superblock(&sb).unwrap();
    bytes[sig..sig + rewritten.len()].copy_from_slice(&rewritten);

    let file = File::from_bytes(bytes).unwrap();
    let got = file.root().dataset("anything").map(|_| ());
    assert!(
        matches!(&got, Err(Error::NotAGroup(p)) if p.is_empty()),
        "a root that is not a group must refuse, got {got:?}"
    );
}

/// Returns metadata, value, and error observations as comparable text.
fn read_everything(file: &File) -> Vec<String> {
    let mut out = Vec::new();
    out.push(format!("root groups: {:?}", file.root().groups()));
    out.push(format!("root datasets: {:?}", file.root().datasets()));
    out.push(format!("root attrs: {:?}", sorted(file.root().attrs())));

    for path in [
        "plain",
        "g/nested",
        "many_attrs",
        "missing",
        "g/missing",
        "plain/nope",
    ] {
        match file.dataset(path) {
            Ok(ds) => {
                out.push(format!("{path}: shape {:?}", ds.shape()));
                out.push(format!("{path}: attrs {:?}", sorted(ds.attrs())));
                out.push(format!("{path}: all {:?}", ds.read_i32()));
                out.push(format!("{path}: rows {:?}", ds.read_i32_rows(1, 2)));
                out.push(format!("{path}: raw rows {:?}", ds.read_raw_rows(0, 1)));
            }
            Err(e) => out.push(format!("{path}: error {e}")),
        }
    }
    out
}

/// Formats attribute values in name order and preserves read errors.
fn sorted(attrs: Result<HashMap<String, AttrValue>, Error>) -> Vec<String> {
    match attrs {
        Ok(map) => {
            let mut v: Vec<String> = map.iter().map(|(k, val)| format!("{k}={val:?}")).collect();
            v.sort();
            v
        }
        Err(e) => vec![format!("error {e}")],
    }
}

/// Checks identical observations from mirrored and source-backed editing sessions.
fn assert_both_read_paths_agree(bytes: &[u8], what: &str) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("both.h5");
    std::fs::write(&path, bytes).unwrap();

    // One session at a time: `open_rw` takes an exclusive lock, and holding
    // two over one path fails outright where OS locks are mandatory.
    let via_mirror = {
        let f = File::open_rw(&path).unwrap();
        read_everything(&f)
    };
    let via_source = {
        let f = File::open_rw_source_only(&path).unwrap();
        read_everything(&f)
    };

    assert_eq!(
        via_mirror.len(),
        via_source.len(),
        "{what}: the two read paths produced different numbers of results"
    );
    for (m, s) in via_mirror.iter().zip(&via_source) {
        assert_eq!(m, s, "{what}: slice and Source read paths disagree");
    }
    // Guard the guard: a helper that read nothing would make the comparison
    // vacuous, and a file whose datasets all failed to open would too.
    assert!(
        via_mirror.iter().any(|r| r.contains("all Ok(")),
        "{what}: no dataset read succeeded, so this compared nothing"
    );
}

/// Builds plain, nested, and dense-attribute datasets with an optional user block.
fn both_paths_file_bytes(userblock: Option<u64>) -> Vec<u8> {
    let mut b = FileBuilder::new();
    if let Some(ub) = userblock {
        b.with_userblock(UserblockSize::try_from(ub).unwrap());
    }
    b.create_dataset("plain")
        .with_i32_data(&(0..24).collect::<Vec<i32>>())
        .with_shape(&[6, 4])
        .set_attr("units", AttrValue::String("m".into()));
    // Well past the eight-attribute compact limit, so the header converts to
    // the dense layout and the dense extraction path is the one that runs.
    {
        let ds = b
            .create_dataset("many_attrs")
            .with_i32_data(&(0..8).collect::<Vec<i32>>());
        for i in 0..24 {
            ds.set_attr(&format!("attr_{i:02}"), AttrValue::I64(i));
        }
    }
    let mut g = b.create_group("g");
    g.create_dataset("nested")
        .with_i32_data(&(100..112).collect::<Vec<i32>>())
        .with_shape(&[3, 4]);
    b.add_group(g.finish());
    b.finish().unwrap()
}

#[test]
fn both_read_paths_agree() {
    assert_both_read_paths_agree(&both_paths_file_bytes(None), "no userblock");
}

// The user block sets a nonzero source base for both read paths.
#[test]
fn both_read_paths_agree_with_a_userblock() {
    assert_both_read_paths_agree(&both_paths_file_bytes(Some(512)), "512-byte userblock");
}

#[rstest]
#[case::no_retention(ChunkCacheConfig::disabled())]
#[case::index_only(ChunkCacheConfig::disabled().with_index_cache(true))]
#[case::payload_only(ChunkCacheConfig::new().with_index_cache(false))]
#[case::all_retained(ChunkCacheConfig::new())]
fn typed_sweeps_reuse_one_index_without_changing_retention(#[case] config: ChunkCacheConfig) {
    use crate::parsed_chunk_index::test_util as chunk_index_test_util;

    let n = 3 * 131_072 + 5;
    let edge = 128;
    let (bytes, expected) = chunked_f64_file(n, edge);
    let file =
        File::from_bytes_with_options(bytes, FileAccessProperties::new().with_chunk_cache(config))
            .unwrap();
    let dataset = file.dataset("d").unwrap();
    let layout = dataset.read_layout().unwrap();
    let dataspace = dataset.dataspace().unwrap();
    let rows = typed_window_rows(&layout, &dataspace, NonZeroUsize::new(8).unwrap())
        .unwrap()
        .get();
    assert!(rows < n as u64);
    let descriptors = n.div_ceil(edge as usize);
    chunk_index_test_util::reset();
    assert_eq!(dataset.read_f64().unwrap(), expected);
    assert_eq!(chunk_index_test_util::counts(), (1, descriptors));
    let stats = dataset.chunk_cache_stats();
    assert_eq!(stats.index_loaded(), config.index_cache_enabled());
    assert_eq!(stats.cached_chunks() != 0, config.max_slots() != 0);
    chunk_index_test_util::reset();
    assert_eq!(dataset.read_f64().unwrap(), expected);
    assert_eq!(
        chunk_index_test_util::counts(),
        (usize::from(!config.index_cache_enabled()), descriptors)
    );
    assert_eq!(
        dataset.chunk_cache_stats().index_loaded(),
        config.index_cache_enabled()
    );
}

#[rstest]
#[case::buffered(None)]
#[case::mirrored(Some(MemoryStrategy::Mirrored))]
#[case::bounded(Some(MemoryStrategy::Bounded))]
fn typed_sweeps_preserve_backend_address_framing(#[case] strategy: Option<MemoryStrategy>) {
    let path = temp::temp_path("sweep.h5");
    let n = 131_077;
    let expected: Vec<f64> = (0..n).map(f64::from).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("d")
        .with_f64_data(&expected)
        .with_shape(&[n as u64])
        .with_chunks(&[128]);
    if strategy != Some(MemoryStrategy::Bounded) {
        builder.with_userblock(UserblockSize::try_from(512).unwrap());
    }
    builder.write(&path).unwrap();
    let options = FileAccessProperties::new().with_chunk_cache(ChunkCacheConfig::disabled());
    let file = match strategy {
        Some(strategy) => {
            File::open_rw_with_options(&path, options.with_memory_strategy(strategy)).unwrap()
        }
        None => File::open_with_options(&path, options).unwrap(),
    };
    assert_eq!(file.dataset("d").unwrap().read_f64().unwrap(), expected);
    file.close().unwrap();
}

/// Builds an in-memory dataset of 256 [`i32`] values in 32-element chunks.
fn chunked_file_bytes() -> Vec<u8> {
    let data: Vec<i32> = (0..256).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("chunked")
        .with_i32_data(&data)
        .with_shape(&[256])
        .with_chunks(&[32]);
    b.finish().unwrap()
}

// The DAPL override must drive the *live* `ChunkCache`, not merely the value
// reported by `chunk_cache_config()`. These assertions reach the crate's
// `#[cfg(test)]` cache introspection (unavailable to integration tests), so
// they fail if the resolved config ever stops flowing into the real cache.

#[test]
fn enabled_override_populates_live_cache_over_disabled_file_default() {
    let file = File::from_bytes_with_options(
        chunked_file_bytes(),
        FileAccessProperties::new().with_chunk_cache(ChunkCacheConfig::disabled()),
    )
    .unwrap();

    let ds = file
        .dataset_with_options(
            "chunked",
            DatasetAccessProperties::new().with_chunk_cache(ChunkCacheConfig::new()),
        )
        .unwrap();
    assert_eq!(ds.read_i32().unwrap(), (0..256).collect::<Vec<i32>>());

    assert!(ds.chunk_cache_stats().index_loaded());
    assert!(ds.chunk_cache_stats().cached_chunks() > 0);
}

#[test]
fn disabled_override_suppresses_live_cache_over_enabled_file_default() {
    let file = File::from_bytes_with_options(
        chunked_file_bytes(),
        FileAccessProperties::new().with_chunk_cache(ChunkCacheConfig::new()),
    )
    .unwrap();

    let ds = file
        .dataset_with_options(
            "chunked",
            DatasetAccessProperties::new().with_chunk_cache(ChunkCacheConfig::disabled()),
        )
        .unwrap();
    assert_eq!(ds.read_i32().unwrap(), (0..256).collect::<Vec<i32>>());

    assert!(!ds.chunk_cache_stats().index_loaded());
    assert_eq!(ds.chunk_cache_stats().cached_chunks(), 0);
}

// Adding the nonzero base to the rewritten child address overflows `u64`.
#[test]
fn group_child_address_base_overflow_is_rejected() {
    const UB: u64 = 512;
    let mut b = FileBuilder::new();
    b.with_userblock(UserblockSize::try_from(UB).unwrap());
    let mut child = b.create_group("child");
    child.create_dataset("inner").with_i32_data(&[1, 2, 3]);
    b.add_group(child.finish());
    let mut bytes = b.finish().unwrap();

    // Baseline: the file reads and the subgroup is listed.
    let file = File::from_bytes(bytes.clone()).unwrap();
    assert_eq!(file.root().groups().unwrap(), vec!["child".to_string()]);

    // Rewrite the child's stored object-header address to HADDR_UNDEF. It is
    // stored base-relative (absolute minus the userblock base) and, for this
    // single-child file, appears exactly once in the bytes. The link lives in
    // the root object header's chunk-0.
    let stored = file
        .root()
        .group("child")
        .unwrap()
        .header_address()
        .unwrap()
        - UB;
    let needle = stored.to_le_bytes();
    let matches: Vec<usize> = bytes
        .windows(8)
        .enumerate()
        .filter(|(_, w)| *w == needle)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "stored child address {stored:#x} was not uniquely locatable: {matches:?}"
    );
    bytes[matches[0]..matches[0] + 8].copy_from_slice(&u64::MAX.to_le_bytes());

    // Refresh the checksum so address conversion checks the edited link field.
    #[cfg(feature = "checksum")]
    {
        let root_addr = file.root().header_address().unwrap() as usize;
        assert_eq!(&bytes[root_addr..root_addr + 4], b"OHDR");
        let flags = bytes[root_addr + 5];
        let mut pos = root_addr + 6;
        if flags & 0x20 != 0 {
            pos += 16;
        }
        if flags & 0x10 != 0 {
            pos += 4;
        }
        let width = 1usize << (flags & 0x03);
        let chunk0 = (0..width).fold(0usize, |acc, i| {
            acc | ((bytes[pos + i] as usize) << (8 * i))
        });
        pos += width;
        let chunk0_end = pos + chunk0;
        assert!(
            matches[0] < chunk0_end,
            "patched link address is outside the root header's chunk-0"
        );
        let cs = crate::checksum::jenkins_lookup3(&bytes[root_addr..chunk0_end]);
        bytes[chunk0_end..chunk0_end + 4].copy_from_slice(&cs.to_le_bytes());
    }

    let file = File::from_bytes(bytes).unwrap();
    match file.root().groups() {
        Err(Error::Format(FormatError::OffsetOverflow { offset, length })) => {
            assert_eq!(offset, u64::MAX);
            assert_eq!(length, UB);
        }
        other => panic!("expected group-child address overflow, got {other:?}"),
    }
}

/// Updates chunk 0's checksum and returns the checksummed header range.
#[cfg(feature = "checksum")]
fn refresh_v2_header_checksum(bytes: &mut [u8], header_addr: usize) -> std::ops::Range<usize> {
    assert_eq!(&bytes[header_addr..header_addr + 4], b"OHDR");
    let flags = bytes[header_addr + 5];
    let mut pos = header_addr + 6;
    if flags & 0x20 != 0 {
        pos += 16;
    }
    if flags & 0x10 != 0 {
        pos += 4;
    }
    let width = 1usize << (flags & 0x03);
    let chunk0 = (0..width).fold(0usize, |acc, i| {
        acc | ((bytes[pos + i] as usize) << (8 * i))
    });
    pos += width;
    let chunk0_end = pos + chunk0;
    let cs = crate::checksum::jenkins_lookup3(&bytes[header_addr..chunk0_end]);
    bytes[chunk0_end..chunk0_end + 4].copy_from_slice(&cs.to_le_bytes());
    header_addr..chunk0_end
}

// Dataset sizes straddle the typed reader's window threshold.
#[cfg(feature = "checksum")]
#[test]
fn a_layout_size_disagreeing_with_the_dataspace_is_refused_at_every_dataset_size() {
    for n in [1000usize, 200_000] {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut b = crate::writer::FileBuilder::new();
        b.create_dataset("t")
            .with_f64_data(&data)
            .with_shape(&[n as u64]);
        let mut bytes = b.finish().unwrap();
        // Taken before the edit: an edited header fails its checksum, and the
        // address is needed to recompute it.
        let header_addr = {
            let file = File::from_bytes(bytes.clone()).unwrap();
            file.dataset("t").unwrap().header_address().unwrap() as usize
        };

        // The stored byte size occurs exactly once inside the checksummed header.
        let declared = (n * 8) as u64;
        let needle = declared.to_le_bytes();
        let at: Vec<usize> = bytes
            .windows(8)
            .enumerate()
            .filter(|(_, w)| *w == needle)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            at.len(),
            1,
            "the stored size {declared} was not uniquely locatable in a {n}-element file: {at:?}"
        );
        bytes[at[0]..at[0] + 8].copy_from_slice(&(declared * 2).to_le_bytes());

        let chunk0 = refresh_v2_header_checksum(&mut bytes, header_addr);
        assert!(
            chunk0.contains(&at[0]),
            "the patched size is outside the header chunk whose checksum was refreshed"
        );

        let file = File::from_bytes(bytes).unwrap();
        let ds = file.dataset("t").unwrap();
        for (what, err) in [
            ("read_raw", ds.read_raw().unwrap_err()),
            ("read_raw_rows", ds.read_raw_rows(0, 1).unwrap_err()),
            (
                "read_raw_region",
                ds.read_raw_region(&[0], &[1]).unwrap_err(),
            ),
            (
                "empty read_raw_region",
                ds.read_raw_region(&[n as u64], &[0]).unwrap_err(),
            ),
            (
                "empty read_raw_rows",
                ds.read_raw_rows(n as u64, 0).unwrap_err(),
            ),
            ("read_f64", ds.read_f64().unwrap_err()),
            ("read_f64_rows", ds.read_f64_rows(0, 1).unwrap_err()),
            ("read_i32", ds.read_i32().unwrap_err()),
        ] {
            let Error::Format(FormatError::LogicalDataSizeMismatch { expected, actual }) = &err
            else {
                panic!("{what} over {n} elements: expected LogicalDataSizeMismatch, got {err:?}");
            };
            assert_eq!(
                *expected,
                (n * 8) as u64,
                "{what} reported the wrong expected size"
            );
            assert_eq!(
                *actual,
                (n * 16) as u64,
                "{what} reported the wrong stored size"
            );
        }
    }
}

#[rstest::rstest]
#[case::unallocated_contiguous(DataLayout::Contiguous { address: None, size: 0 }, None)]
#[case::virtual_layout(DataLayout::Virtual, Some(FormatError::UnsupportedVirtualLayout))]
fn empty_region_reads_preserve_layout_errors(
    #[case] layout: DataLayout,
    #[case] expected: Option<FormatError>,
) {
    let dataspace = Dataspace {
        space_type: DataspaceType::Simple,
        rank: 1,
        dimensions: vec![0],
        max_dimensions: None,
    };
    let datatype = crate::make_u8_type();
    let region = ReadRegion::parse(
        RawReadSpec::plain(&layout, &dataspace, &datatype),
        &[0],
        &[0],
    )
    .unwrap();
    let result = read_region_framed(
        &source::zero_base_source(&BytesSource::new(&[])),
        &region,
        FormatWidths::from_sizes(8, 8).unwrap(),
        &ChunkCache::new(),
        CachePass::LRU,
    );
    match expected {
        Some(error) => assert_eq!(result.unwrap_err(), error),
        None => assert_eq!(result.unwrap(), Vec::<u8>::new()),
    }
}

#[test]
fn compact_regions_read_selected_columns() {
    let layout = DataLayout::Compact {
        data: (0..16).collect(),
    };
    let dataspace = Dataspace {
        space_type: DataspaceType::Simple,
        rank: 2,
        dimensions: vec![4, 4],
        max_dimensions: None,
    };
    let datatype = crate::make_u8_type();
    let region = ReadRegion::parse(
        RawReadSpec::parse(&layout, &dataspace, &datatype, None, FillPattern::ZERO).unwrap(),
        &[1, 2],
        &[2, 2],
    )
    .unwrap();
    assert_eq!(
        read_region_framed(
            &source::zero_base_source(&BytesSource::new(&[])),
            &region,
            FormatWidths::from_sizes(8, 8).unwrap(),
            &ChunkCache::new(),
            CachePass::LRU,
        )
        .unwrap(),
        [6, 7, 10, 11]
    );
}

#[test]
fn a_typed_read_windows_in_whole_chunk_bands() {
    const BUDGET: u64 = TYPED_READ_WINDOW_BYTES;
    let ds1 = |dims: &[u64]| Dataspace {
        space_type: crate::dataspace::DataspaceType::Simple,
        rank: dims.len() as u8,
        dimensions: dims.to_vec(),
        max_dimensions: None,
    };
    let contiguous = DataLayout::Contiguous {
        address: Some(StoredAddress::new(0)),
        size: 0,
    };
    let chunked = |band: u32| DataLayout::Chunked {
        flags: ChunkedLayoutFlags::NONE,
        chunk_dimensions: vec![u64::from(band), 8],
        index: ChunkIndexLayout::BTreeV1 {
            address: Some(StoredAddress::new(0)),
        },
    };
    let elem = NonZeroUsize::new(8).unwrap();

    // Contiguous reads divide the byte budget by row width.
    assert_eq!(
        typed_window_rows(&contiguous, &ds1(&[1 << 20]), elem)
            .unwrap()
            .get(),
        BUDGET / 8
    );

    assert_eq!(
        typed_window_rows(&chunked(512), &ds1(&[1 << 20]), elem)
            .unwrap()
            .get(),
        BUDGET / 8
    );
    let rows = typed_window_rows(&chunked(300), &ds1(&[1 << 20]), elem)
        .unwrap()
        .get();
    assert_eq!(rows % 300, 0, "a window must end on a chunk band");
    assert!(
        rows <= BUDGET / 8 && rows > BUDGET / 8 - 300,
        "a window must be the largest whole number of bands within the \
         budget, not a smaller one: {rows} rows against {} in budget",
        BUDGET / 8
    );

    // One band over budget: the window is that band, since a narrower one
    // would decode it twice.
    assert_eq!(
        typed_window_rows(&chunked(1 << 20), &ds1(&[1 << 21]), elem)
            .unwrap()
            .get(),
        1 << 20
    );

    // Rows wider than the whole budget: one row, which is the least a window
    // can be and still make progress.
    assert_eq!(
        typed_window_rows(&contiguous, &ds1(&[4, 1 << 20]), elem)
            .unwrap()
            .get(),
        1
    );

    // A scalar has one logical row, including with chunked layout metadata.
    let scalar = Dataspace {
        space_type: crate::dataspace::DataspaceType::Scalar,
        rank: 0,
        dimensions: Vec::new(),
        max_dimensions: None,
    };
    assert!(typed_window_rows(&chunked(8), &scalar, elem).unwrap().get() >= 1);

    // A zero inner dimension makes a row zero bytes wide, and the dataset
    // has no elements at all: one window covers it, and nothing divides by
    // zero on the way there.
    assert_eq!(
        typed_window_rows(&contiguous, &ds1(&[4, 0]), elem)
            .unwrap()
            .get(),
        u64::MAX
    );
}

/// The two channels a caller has for one object's attributes.
struct AttrChannels {
    owner: &'static str,
    /// Decoded attribute values.
    values: HashMap<String, AttrValue>,
    /// Exact stored datatype descriptions.
    datatypes: HashMap<String, Datatype>,
}

/// One written file: its bytes, and both channels read back from each owner.
struct AttrFile {
    /// The file as written, so a test can assert which storage form it got.
    bytes: Vec<u8>,
    owners: Vec<AttrChannels>,
}

/// Writes the same attributes on the root and dataset and reads values and datatypes.
fn attr_channels(
    values: &[(&str, AttrValue)],
    verbatim: &[crate::attribute::AttributeMessage],
) -> AttrFile {
    let mut b = FileBuilder::new();
    for (name, value) in values {
        b.set_attr(name, value.clone());
    }
    for message in verbatim {
        b.set_attr_verbatim(message.clone(), None);
    }
    {
        let ds = b.create_dataset("data").with_f64_data(&[1.0]);
        for (name, value) in values {
            ds.set_attr(name, value.clone());
        }
        for message in verbatim {
            ds.set_attr_verbatim(message.clone(), None);
        }
    }
    let bytes = b.finish().unwrap();
    let file = File::from_bytes(bytes.clone()).unwrap();
    let root = file.root();
    let dataset = file.dataset("data").unwrap();
    AttrFile {
        bytes,
        owners: vec![
            AttrChannels {
                owner: "root group",
                values: root.attrs().unwrap(),
                datatypes: root.attr_datatypes().unwrap(),
            },
            AttrChannels {
                owner: "dataset",
                values: dataset.attrs().unwrap(),
                datatypes: dataset.attr_datatypes().unwrap(),
            },
        ],
    }
}

// Both integer attributes decode to `I32`, despite different byte orders.
#[test]
fn attr_datatypes_reports_the_byte_order_attrs_normalizes() {
    let be = crate::attribute::AttributeMessage {
        name: "be".into(),
        datatype: Datatype::FixedPoint(
            FixedPointType::new(4, DatatypeByteOrder::BigEndian, true, 0, 32).unwrap(),
        ),
        dataspace: Dataspace {
            space_type: crate::dataspace::DataspaceType::Scalar,
            rank: 0,
            dimensions: vec![],
            max_dimensions: None,
        },
        raw_data: (-7i32).to_be_bytes().to_vec(),
        datatype_location: crate::shared_message::DatatypeLocation::Inline,
    };

    for c in attr_channels(&[("count", AttrValue::I32(-7))], std::slice::from_ref(&be)).owners {
        assert_eq!(
            c.values.get("count"),
            Some(&AttrValue::I32(-7)),
            "{}: the value channel keeps the width the attribute was written at",
            c.owner
        );
        let Some(Datatype::FixedPoint(numeric)) = c.datatypes.get("count") else {
            panic!(
                "{}: expected a fixed-point datatype, got {:?}",
                c.owner,
                c.datatypes.get("count")
            );
        };
        assert_eq!(
            (numeric.size().get(), numeric.layout().signed()),
            (4, true),
            "{}: the datatype channel must report the width on disk",
            c.owner
        );

        assert_eq!(
            c.values.get("be"),
            Some(&AttrValue::I32(-7)),
            "{}: a big-endian attribute decodes to the value it holds",
            c.owner
        );
        let Some(Datatype::FixedPoint(numeric)) = c.datatypes.get("be") else {
            panic!(
                "{}: expected a fixed-point datatype, got {:?}",
                c.owner,
                c.datatypes.get("be")
            );
        };
        assert_eq!(
            numeric.byte_order(),
            DatatypeByteOrder::BigEndian,
            "{}: the datatype channel is the only record of the byte order",
            c.owner
        );
    }
}

// The opaque attribute has a datatype but no decoded `AttrValue`.
#[test]
fn attr_datatypes_reports_an_attribute_attrs_omits() {
    let opaque = Datatype::Opaque {
        size: 3,
        tag: b"rgb".to_vec(),
    };
    let raw = crate::attribute::AttributeMessage {
        name: "raw".into(),
        datatype: opaque.clone(),
        dataspace: Dataspace {
            space_type: crate::dataspace::DataspaceType::Scalar,
            rank: 0,
            dimensions: vec![],
            max_dimensions: None,
        },
        raw_data: vec![1, 2, 3],
        datatype_location: crate::shared_message::DatatypeLocation::Inline,
    };

    for c in attr_channels(&[("count", AttrValue::I32(1))], std::slice::from_ref(&raw)).owners {
        assert!(
            !c.values.contains_key("raw"),
            "{}: an opaque attribute has no `AttrValue`, so `attrs` omits it — \
             if that changes, this test is measuring the wrong thing",
            c.owner
        );
        assert_eq!(
            c.datatypes.get("raw"),
            Some(&opaque),
            "{}: the datatype channel must report an attribute `attrs` omits",
            c.owner
        );
        // Specific to the one attribute: a channel that reported only the
        // undecodable one, or dropped its neighbour, would pass the above.
        assert!(
            c.values.contains_key("count") && c.datatypes.contains_key("count"),
            "{}: the attribute beside it must appear in both channels",
            c.owner
        );
    }
}

// Whole and row-window typed reads use the same width limit.
#[test]
fn a_wide_dataset_is_refused_by_the_typed_readers() {
    let dt = Datatype::FixedPoint(
        FixedPointType::new(16, DatatypeByteOrder::LittleEndian, false, 0, 128).unwrap(),
    );
    let mut b = FileBuilder::new();
    b.create_dataset("wide")
        .with_raw_data(dt.clone(), vec![0xFF; 32], 2);
    let file = File::from_bytes(b.finish().unwrap()).unwrap();
    let ds = file.dataset("wide").unwrap();

    assert_eq!(
        ds.datatype().unwrap(),
        dt,
        "the datatype still reads: it is the values that have no answer"
    );
    assert!(matches!(
        ds.read_u64(),
        Err(Error::Format(FormatError::NumericElementTooWide {
            size: 16
        }))
    ));
    assert!(matches!(
        ds.read_u64_rows(0, 1),
        Err(Error::Format(FormatError::NumericElementTooWide {
            size: 16
        }))
    ));
    // Raw reads preserve the full 32-byte payload.
    assert_eq!(ds.read_raw().unwrap().len(), 32);
}

#[test]
fn an_attribute_too_wide_to_decode_is_omitted_rather_than_truncated() {
    let wide = Datatype::FixedPoint(
        FixedPointType::new(9, DatatypeByteOrder::LittleEndian, false, 0, 72).unwrap(),
    );
    // 2^64 exactly, so every one of the low 64 bits is zero.
    let mut raw_data = vec![0u8; 9];
    raw_data[8] = 1;
    let huge = crate::attribute::AttributeMessage {
        name: "huge".into(),
        datatype: wide.clone(),
        dataspace: Dataspace {
            space_type: crate::dataspace::DataspaceType::Scalar,
            rank: 0,
            dimensions: vec![],
            max_dimensions: None,
        },
        raw_data,
        datatype_location: crate::shared_message::DatatypeLocation::Inline,
    };

    for c in attr_channels(&[("count", AttrValue::I32(1))], std::slice::from_ref(&huge)).owners {
        assert!(
            !c.values.contains_key("huge"),
            "{}: decoding this attribute would report 0 for a value of 2^64",
            c.owner
        );
        assert_eq!(
            c.datatypes.get("huge"),
            Some(&wide),
            "{}: the datatype channel must still report the width on disk",
            c.owner
        );
        assert!(
            c.values.contains_key("count") && c.datatypes.contains_key("count"),
            "{}: the attribute beside it must still decode",
            c.owner
        );
    }
}

// Twelve attributes force dense storage, confirmed by the fractal-heap signature.
#[test]
fn attr_datatypes_covers_dense_attribute_storage() {
    let names: Vec<String> = (0..12).map(|i| format!("a{i:02}")).collect();
    let values: Vec<(&str, AttrValue)> = names
        .iter()
        .enumerate()
        .map(|(i, n)| {
            (
                n.as_str(),
                #[expect(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                AttrValue::I32(i as i32),
            )
        })
        .collect();

    let f = attr_channels(&values, &[]);
    assert!(
        f.bytes.windows(4).any(|w| w == b"FRHP"),
        "the fixture must really use dense storage, or this test proves \
         nothing beyond the compact path the other tests already cover"
    );

    for c in f.owners {
        assert_eq!(
            c.datatypes.len(),
            names.len(),
            "{}: every attribute in the heap must be reported, got {:?}",
            c.owner,
            c.datatypes.keys().collect::<Vec<_>>()
        );
        // The two channels must agree on which attributes exist: all of these
        // decode, so neither one has anything to omit here.
        let mut from_values: Vec<&String> = c.values.keys().collect();
        let mut from_types: Vec<&String> = c.datatypes.keys().collect();
        from_values.sort();
        from_types.sort();
        assert_eq!(
            from_values, from_types,
            "{}: the channels disagree",
            c.owner
        );
        for name in &names {
            assert!(
                matches!(
                    c.datatypes.get(name),
                    Some(Datatype::FixedPoint(numeric)) if numeric.size().get() == 4
                ),
                "{}: {name} must keep its 4-byte width through the heap, got {:?}",
                c.owner,
                c.datatypes.get(name)
            );
        }
    }
}

// -----------------------------------------------------------------------
// Coalesced chunk reads (see `crate::chunk_span`)
// -----------------------------------------------------------------------

/// A [`Source`] with counters for read calls and requested byte lengths.
struct CountingSource {
    bytes: Vec<u8>,
    reads: Arc<AtomicUsize>,
    bytes_read: Arc<AtomicUsize>,
}

impl Source for CountingSource {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.bytes_read.fetch_add(buf.len(), Ordering::Relaxed);
        BytesSource::new(&self.bytes).read_at(offset, buf)
    }
}

/// Counters for one measured read.
#[derive(Default)]
struct ReadCounts {
    reads: Arc<AtomicUsize>,
    bytes: Arc<AtomicUsize>,
}

impl ReadCounts {
    /// Runs `f` and returns the change in read-call and byte counters.
    fn measure<R>(&self, f: impl FnOnce() -> R) -> (usize, usize) {
        let r0 = self.reads.load(Ordering::Relaxed);
        let b0 = self.bytes.load(Ordering::Relaxed);
        f();
        (
            self.reads.load(Ordering::Relaxed) - r0,
            self.bytes.load(Ordering::Relaxed) - b0,
        )
    }
}

/// Builds a streaming [`File`] over `bytes` with shared read counters.
fn counting_streaming_file(
    bytes: Vec<u8>,
    counts: &ReadCounts,
    access: FileAccessProperties,
) -> File {
    File::from_source_with_options(
        CountingSource {
            bytes,
            reads: Arc::clone(&counts.reads),
            bytes_read: Arc::clone(&counts.bytes),
        },
        access,
    )
    .expect("open from source")
}

/// Builds file bytes and sequential values for an `n`-element [`f64`] dataset.
fn chunked_f64_file(n: usize, chunk: u64) -> (Vec<u8>, Vec<f64>) {
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("d")
        .with_f64_data(&data)
        .with_shape(&[n as u64])
        .with_chunks(&[chunk]);
    (builder.finish().expect("write file"), data)
}

// The dataset has 1,024 adjacent single-row chunks.
#[test]
fn a_streaming_read_coalesces_a_run_of_small_chunks() {
    let n = 1024usize;
    let (bytes, data) = chunked_f64_file(n, 1);
    let counts = ReadCounts::default();
    let file = counting_streaming_file(bytes, &counts, FileAccessProperties::new());

    let mut got = Vec::new();
    let (reads, _) = counts.measure(|| got = file.dataset("d").unwrap().read_f64().unwrap());
    assert_eq!(got, data, "the coalesced read must return the same values");

    assert!(
        reads < n / 8,
        "expected the {n} chunks to be coalesced into few reads, got {reads}"
    );
}

#[test]
fn a_windowed_streaming_read_fetches_only_its_own_window() {
    let rows = 4096usize;
    let (bytes, data) = chunked_f64_file(rows, 4);
    let counts = ReadCounts::default();
    let file = counting_streaming_file(bytes, &counts, FileAccessProperties::new());
    let ds = file.dataset("d").unwrap();

    // The first window walks (and caches on this handle) the chunk index,
    // so measure the second: what it reads is the window's own chunks.
    assert_eq!(ds.read_f64_rows(0, 8).unwrap(), data[0..8]);
    let (_, window_bytes) =
        counts.measure(|| assert_eq!(ds.read_f64_rows(2048, 8).unwrap(), data[2048..2056]));

    // Eight rows are two 32-byte chunks.
    assert!(
        window_bytes < 256,
        "an 8-row window read {window_bytes} bytes; it needs 64"
    );
}

/// The second read, through the retained index, fetches at most the dataset's bytes.
#[test]
fn a_second_whole_read_costs_what_the_first_did() {
    // 512 KiB of f64 in 1 KiB chunks: more than one 256 KiB span, so an
    // unordered walk has somewhere to thrash between.
    let n = 64 * 1024usize;
    let dataset_bytes = n * 8;
    let (bytes, data) = chunked_f64_file(n, 128);
    let counts = ReadCounts::default();
    let file = counting_streaming_file(bytes, &counts, FileAccessProperties::new());
    let ds = file.dataset("d").unwrap();

    counts.measure(|| assert_eq!(ds.read_f64().unwrap(), data));
    let (_, second) = counts.measure(|| assert_eq!(ds.read_f64().unwrap(), data));

    assert!(
        second <= dataset_bytes,
        "the second read fetched {second} bytes of a {dataset_bytes}-byte dataset"
    );
}

/// The loop, through the retained index, fetches at most twice the dataset's bytes.
#[test]
fn a_window_loop_costs_the_dataset_once() {
    // Read 2 MiB in 512 KiB windows across 1 KiB chunks.
    let rows = 256 * 1024usize;
    let dataset_bytes = rows * 8;
    let (bytes, data) = chunked_f64_file(rows, 128);
    let counts = ReadCounts::default();
    let file = counting_streaming_file(bytes, &counts, FileAccessProperties::new());
    let ds = file.dataset("d").unwrap();

    let window = 64 * 1024;
    let (_, total) = counts.measure(|| {
        for lo in (0..rows).step_by(window) {
            let got = ds.read_f64_rows(lo as u64, window as u64).unwrap();
            assert_eq!(got, data[lo..lo + window]);
        }
    });

    assert!(
        total <= 2 * dataset_bytes,
        "a window loop over a {dataset_bytes}-byte dataset read {total} bytes"
    );
}

// The cache can retain every chunk across both reads.
#[test]
fn a_warm_chunk_cache_is_not_re_fetched() {
    // 32 KiB of f64 in 32 chunks of 1 KiB, laid end to end.
    let n = 4096usize;
    let chunk_bytes = 1024usize;
    let (bytes, data) = chunked_f64_file(n, 128);
    let counts = ReadCounts::default();
    let file = counting_streaming_file(
        bytes,
        &counts,
        FileAccessProperties::new().with_chunk_cache(ChunkCacheConfig::new().with_max_slots(64)),
    );
    let ds = file.dataset("d").unwrap();

    // Warm the second half of the dataset: chunks 16..31.
    assert_eq!(ds.read_f64_rows(2048, 2048).unwrap(), data[2048..]);
    assert_eq!(
        ds.chunk_cache_stats().cached_chunks(),
        16,
        "the window's own chunks stay cached"
    );

    let (_, second) = counts.measure(|| assert_eq!(ds.read_f64().unwrap(), data));
    assert_eq!(
        second,
        16 * chunk_bytes,
        "only the cold half was needed; a span over the whole dataset would \
         have fetched {} bytes",
        32 * chunk_bytes
    );
}

// -----------------------------------------------------------------------
// Group member iterators (`iter_datasets` / `iter_groups`)
// -----------------------------------------------------------------------

/// Builds root datasets, subgroups with two datasets each, and a committed datatype.
///
/// Names omit zero-padding, so lexical and insertion order differ at two-digit indices.
fn mixed_member_file(datasets: usize, groups: usize) -> Vec<u8> {
    let mut b = FileBuilder::new();
    b.commit_datatype("a_type", crate::make_i32_type());
    for i in 0..datasets {
        b.create_dataset(&format!("ds{i}"))
            .with_i32_data(&[i as i32, -(i as i32)]);
    }
    for i in 0..groups {
        let mut g = b.create_group(&format!("g{i}"));
        g.create_dataset("inner").with_i32_data(&[i as i32]);
        g.create_dataset("other").with_i32_data(&[-1]);
        b.add_group(g.finish());
    }
    b.finish().expect("write the fixture")
}

#[test]
fn iter_datasets_agrees_with_opening_each_name() {
    let file = File::from_bytes(mixed_member_file(12, 3)).unwrap();

    for group in [file.root(), file.group("g1").unwrap()] {
        let names = group.datasets().unwrap();
        assert!(
            !names.is_empty(),
            "the fixture must have members to compare"
        );

        let iterated: Vec<(String, Dataset)> = group.iter_datasets().unwrap().collect();
        assert_eq!(
            iterated.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
            names,
            "the iterator must yield the members `datasets` lists, in that order"
        );

        for (name, ds) in &iterated {
            let opened = group.dataset(name).unwrap();
            assert_eq!(
                ds.header_address().unwrap(),
                opened.header_address().unwrap(),
                "{name}"
            );
            assert_eq!(ds.shape().unwrap(), opened.shape().unwrap(), "{name}");
            assert_eq!(ds.read_i32().unwrap(), opened.read_i32().unwrap(), "{name}");
        }
    }
}

#[test]
fn iter_groups_agrees_with_opening_each_name() {
    let file = File::from_bytes(mixed_member_file(4, 5)).unwrap();
    let root = file.root();

    let names = root.groups().unwrap();
    let iterated: Vec<(String, Group)> = root.iter_groups().unwrap().collect();
    assert_eq!(
        iterated.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
        names,
        "the iterator must yield the subgroups `groups` lists, in that order"
    );
    assert_eq!(names.len(), 5);

    for (name, group) in &iterated {
        assert_eq!(
            group.header_address().unwrap(),
            root.group(name).unwrap().header_address().unwrap(),
            "{name}"
        );
        let mut inner = group
            .iter_datasets()
            .unwrap()
            .map(|(n, _)| n)
            .collect::<Vec<_>>();
        inner.sort();
        assert_eq!(inner, ["inner", "other"], "{name} must be walkable in turn");
    }
}

// A committed datatype is excluded from both dataset and group iterators.
#[test]
fn the_member_iterators_sort_the_child_kinds_apart() {
    let file = File::from_bytes(mixed_member_file(3, 2)).unwrap();
    let root = file.root();

    let datasets: Vec<String> = root.iter_datasets().unwrap().map(|(n, _)| n).collect();
    let groups: Vec<String> = root.iter_groups().unwrap().map(|(n, _)| n).collect();

    assert_eq!(datasets, ["ds0", "ds1", "ds2"]);
    assert_eq!(groups, ["g0", "g1"]);
    assert_eq!(root.named_datatypes().unwrap(), ["a_type"]);
    assert!(
        !datasets.contains(&"a_type".to_string()) && !groups.contains(&"a_type".to_string()),
        "a committed datatype is neither a dataset nor a group"
    );
}

/// Returns bytes read by per-name lookup and by handle iteration for `n` members.
fn member_walk_bytes(n: usize) -> (usize, usize) {
    let mut b = FileBuilder::new();
    for i in 0..n {
        b.create_dataset(&format!("ds{i}"))
            .with_i32_data(&[i as i32]);
    }
    let bytes = b.finish().expect("write the fixture");

    let counts = ReadCounts::default();
    let file = counting_streaming_file(bytes.clone(), &counts, FileAccessProperties::new());
    let (_, by_name) = counts.measure(|| {
        let root = file.root();
        for name in root.datasets().unwrap() {
            root.dataset(&name).unwrap();
        }
    });

    let counts = ReadCounts::default();
    let file = counting_streaming_file(bytes, &counts, FileAccessProperties::new());
    let (_, iterated) = counts.measure(|| for (_, _ds) in file.root().iter_datasets().unwrap() {});

    (by_name, iterated)
}

#[test]
fn iterating_members_enumerates_the_group_once() {
    let (_, iterated_16) = member_walk_bytes(16);
    let (by_name_64, iterated_64) = member_walk_bytes(64);

    assert!(
        iterated_64 <= 5 * iterated_16,
        "one enumeration plus one header per member is linear, so four times \
         the members must cost about four times the bytes: {iterated_16} -> \
         {iterated_64}"
    );

    assert!(
        iterated_64 < by_name_64,
        "at 64 members the one-enumeration walk should still be the cheaper: \
         {iterated_64} bytes against {by_name_64}"
    );
}

#[test]
fn a_member_handle_carries_the_files_chunk_cache_config() {
    let configured = ChunkCacheConfig::new()
        .with_max_slots(17)
        .with_max_bytes(4096)
        .with_index_cache(false);
    let file = File::from_bytes_with_options(
        mixed_member_file(3, 0),
        FileAccessProperties::new().with_chunk_cache(configured),
    )
    .unwrap();
    let root = file.root();

    let members: Vec<(String, Dataset)> = root.iter_datasets().unwrap().collect();
    assert_eq!(members.len(), 3);
    for (name, ds) in &members {
        assert_eq!(
            ds.chunk_cache_config(),
            root.dataset(name).unwrap().chunk_cache_config(),
            "{name} must resolve its cache the way `dataset` does"
        );
        assert_eq!(
            ds.chunk_cache_config(),
            configured,
            "{name} must carry the configuration the file was opened with"
        );
    }
}

/// Builds root and subgroup children with matching names and a reference to the subgroup.
fn dereferenced_group_file() -> Vec<u8> {
    let mut b = FileBuilder::new();
    b.create_dataset("inner").with_i32_data(&[0]);
    let mut root_sub = b.create_group("sub");
    root_sub.create_dataset("x").with_i32_data(&[0]);
    b.add_group(root_sub.finish());

    let mut g = b.create_group("g0");
    g.create_dataset("inner").with_i32_data(&[1]);
    let mut nested = g.create_group("sub");
    nested.create_dataset("x").with_i32_data(&[1]);
    g.add_group(nested.finish());
    b.add_group(g.finish());

    b.create_dataset("refs").with_path_references(&["g0"]);
    b.finish().expect("write the fixture")
}

#[test]
fn members_of_a_dereferenced_group_have_no_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("refs.h5");
    std::fs::write(&path, dereferenced_group_file()).unwrap();

    let file = File::open_rw(&path).unwrap();
    let mut objects = file.dataset("refs").unwrap().dereference().unwrap();
    let group = match objects.remove(0) {
        Object::Group(g) => g,
        other => panic!("expected a group, got {other:?}"),
    };

    // Writable root objects share these member names.
    let (name, mut member) = group.iter_datasets().unwrap().next().unwrap();
    assert_eq!(name, "inner");
    assert!(
        matches!(
            member.set_attr("tag", AttrValue::I64(1)),
            Err(Error::ReadOnly)
        ),
        "a member of a path-less group must refuse a write, not address `/inner`"
    );

    let (name, subgroup) = group.iter_groups().unwrap().next().unwrap();
    assert_eq!(name, "sub");
    assert!(
        matches!(
            subgroup.set_attr("tag", AttrValue::I64(1)),
            Err(Error::ReadOnly)
        ),
        "and so must a subgroup of one, not address `/sub`"
    );
}

// Subgroup members must retain their full root-relative paths.
#[test]
fn a_member_handle_resolves_to_its_own_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("members.h5");
    std::fs::write(&path, mixed_member_file(2, 2)).unwrap();

    let file = File::open_rw(&path).unwrap();
    let group = file.group("g1").unwrap();
    let (name, mut ds) = group
        .iter_datasets()
        .unwrap()
        .find(|(n, _)| n == "inner")
        .expect("g1/inner");
    assert_eq!(name, "inner");
    ds.set_attr("tag", AttrValue::I64(7)).unwrap();
    file.commit().unwrap();
    // Windows holds the write lock until the session is dropped.
    drop(ds);
    drop(group);
    drop(file);

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("g1/inner")
            .unwrap()
            .attrs()
            .unwrap()
            .get("tag")
            .and_then(AttrValue::as_i64),
        Some(7),
        "the attribute must land on the member the handle came from"
    );
    assert!(
        !file
            .dataset("g0/inner")
            .unwrap()
            .attrs()
            .unwrap()
            .contains_key("tag"),
        "and on no other group's member of the same name"
    );
}

#[test]
fn every_read_write_open_carries_the_fapl_sync_policy() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = |name: &str| {
        let path = dir.path().join(name);
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&[1, 2, 3, 4])
            .with_shape(&[4])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[2]);
        b.write(&path).unwrap();
        path
    };
    let props = || FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose);
    let policy_of = |file: &File| match &file.inner.backend {
        Backend::Edit(m) => m
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .sync_policy(),
        _ => panic!("a read-write open must build an editing session"),
    };

    let opened = File::open_rw_with_options(fixture("open_rw.h5"), props()).unwrap();
    assert_eq!(policy_of(&opened), SyncPolicy::OnClose, "File::open_rw");
    drop(opened);

    let created = File::create_with_options(
        dir.path().join("created.h5"),
        crate::FileCreateProperties::new(),
        props(),
    )
    .unwrap();
    assert_eq!(policy_of(&created), SyncPolicy::OnClose, "File::create");
    drop(created);

    let swmr = File::open_swmr_writer_with_options(fixture("swmr.h5"), props()).unwrap();
    assert_eq!(
        policy_of(&swmr),
        SyncPolicy::OnClose,
        "File::open_swmr_writer"
    );
    drop(swmr);

    let bounded = File::open_rw_with_options(
        fixture("bounded.h5"),
        props().with_memory_strategy(MemoryStrategy::Bounded),
    )
    .unwrap();
    assert_eq!(
        policy_of(&bounded),
        SyncPolicy::OnClose,
        "File::open_rw_with_options (bounded)"
    );
}

#[test]
fn a_page_buffer_is_rejected_where_it_cannot_be_honored() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = |name: &str, paged: bool| {
        let path = dir.path().join(name);
        let mut b = FileBuilder::new();
        if paged {
            b.with_file_space_settings(
                FileSpaceSettings::new(FileSpaceStrategy::Page)
                    .with_persistence(true)
                    .with_page_size(FileSpacePageSize::try_from(16 * 1024).unwrap()),
            );
        }
        b.create_dataset("d")
            .with_i32_data(&[1, 2, 3, 4])
            .with_shape(&[4]);
        b.write(&path).unwrap();
        path
    };
    let buffered = |bytes| {
        FileAccessProperties::new()
            .with_sync_policy(SyncPolicy::OnClose)
            .with_page_buffer_size(bytes)
    };

    // Check file bytes and reopening after each rejected open.
    let untouched = |label: &str, path: &std::path::Path, before: &[u8]| {
        assert_eq!(
            std::fs::read(path).unwrap(),
            before,
            "{label}: a refused open changed the file"
        );
        let file = File::open(path).unwrap();
        assert_eq!(
            file.dataset("d").unwrap().read_i32().unwrap(),
            [1, 2, 3, 4],
            "{label}"
        );
    };

    // An unpaged file is a supported control.
    let unpaged = fixture("unpaged.h5", false);
    let accepted = File::open_rw_with_options(&unpaged, buffered(1 << 20)).unwrap();
    assert_eq!(
        accepted.dataset("d").unwrap().read_i32().unwrap(),
        [1, 2, 3, 4]
    );
    accepted.close().unwrap();

    // The supported 256 KiB budget is below the session's 1 MiB gather budget.
    let small_budget = fixture("small_budget.h5", true);
    let accepted = File::open_rw_with_options(&small_budget, buffered(256 * 1024)).unwrap();
    assert_eq!(
        accepted.dataset("d").unwrap().read_i32().unwrap(),
        [1, 2, 3, 4]
    );
    accepted.close().unwrap();

    let paged = fixture("paged.h5", true);
    let before = std::fs::read(&paged).unwrap();
    let refused = File::open_rw_with_options(&paged, buffered(8192));
    assert!(
        matches!(
            &refused,
            Err(Error::EditUnsupported(
                "a page buffer must be at least the file's file-space page size"
            ))
        ),
        "a budget below one page must be refused, got {refused:?}"
    );
    untouched("budget under one page", &paged, &before);

    // And the pairing a caller reaches by doing nothing, since Always is the
    // default: every barrier there is an fsync that flushes the buffer.
    let refused = File::open_rw_with_options(
        &paged,
        FileAccessProperties::new().with_page_buffer_size(1 << 20),
    );
    assert!(
        matches!(
            &refused,
            Err(Error::EditUnsupported(
                "a page buffer does nothing under SyncPolicy::Always, whose every barrier is an fsync that flushes it; pair with_page_buffer_size with with_sync_policy(SyncPolicy::OnClose), or drop it"
            ))
        ),
        "a page buffer under SyncPolicy::Always must be refused, got {refused:?}"
    );
    untouched("page buffer under Always", &paged, &before);

    // The builder writes PAGE only at version 3 or later.
    let old_format = dir.path().join("v2.h5");
    std::fs::copy(&paged, &old_format).unwrap();
    {
        let mut bytes = std::fs::read(&old_format).unwrap();
        let LocatedSuperblock {
            address,
            superblock: mut sb,
            ..
        } = crate::superblock::read(bytes.as_slice()).unwrap();
        let sig = usize::try_from(address.get()).unwrap();
        assert_eq!(
            sb.version(),
            SuperblockVersion::Three,
            "the fixture must start at the newer format"
        );
        sb.set_version(SuperblockVersion::Two).unwrap();
        let rewritten = hdf5_pure_format::__private::serialize_superblock(&sb).unwrap();
        bytes[sig..sig + rewritten.len()].copy_from_slice(&rewritten);
        std::fs::write(&old_format, &bytes).unwrap();
    }
    let before = std::fs::read(&old_format).unwrap();
    {
        let file = File::open(&old_format).unwrap();
        assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [1, 2, 3, 4]);
    }
    let refused = File::open_rw_with_options(&old_format, buffered(1 << 20));
    assert!(
        matches!(
            &refused,
            Err(Error::EditUnsupported(
                "a page buffer marks the file for the life of the session, so that a session that crashes leaves one every reader refuses rather than one that reads clean; only a version-3 superblock carries a status-flags byte any library reads back, and this file's is older. Rewrite it at the 1.10 format (repack, or FileBuilder's default bounds), or drop the page buffer"
            ))
        ),
        "a page buffer on a pre-v3 superblock must be refused, got {refused:?}"
    );
    untouched("version-2 superblock", &old_format, &before);

    let no_persist = dir.path().join("no_persist.h5");
    let mut b = FileBuilder::new();
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page).with_page_size(FileSpacePageSize::DEFAULT),
    );
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3, 4])
        .with_shape(&[4]);
    b.write(&no_persist).unwrap();
    let before = std::fs::read(&no_persist).unwrap();
    let refused = File::open_rw_with_options(
        &no_persist,
        buffered(1 << 20).with_memory_strategy(MemoryStrategy::Mirrored),
    );
    assert!(
        matches!(
            &refused,
            Err(Error::EditUnsupported(
                "a page buffer on a paged file needs its free space persisted: staged \
                 commits without persisted free space require ordered publication without \
                 page buffering; recreate the file with \
                 with_file_space_settings(FileSpaceSettings::new(\
                 FileSpaceStrategy::Page).with_persistence(true)), or drop \
                 the page buffer"
            ))
        ),
        "a page buffer on a PAGE file without persisted free space must be rejected, got {refused:?}"
    );
    untouched("paged, not persisting", &no_persist, &before);

    let paged_userblock = dir.path().join("paged_userblock.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UserblockSize::try_from(16 * 1024).unwrap());
    b.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_page_size(FileSpacePageSize::try_from(16 * 1024).unwrap()),
    );
    b.create_dataset("d")
        .with_i32_data(&[1, 2, 3, 4])
        .with_shape(&[4]);
    b.write(&paged_userblock).unwrap();
    let accepted = File::open_rw_with_options(&paged_userblock, buffered(1 << 20)).unwrap();
    assert_eq!(accepted.edit_backing(), Some(EditBacking::Mirrored));
    assert_eq!(
        accepted.dataset("d").unwrap().read_i32().unwrap(),
        [1, 2, 3, 4]
    );
    accepted.close().unwrap();

    let swmr = fixture("swmr.h5", false);
    let before = std::fs::read(&swmr).unwrap();
    let refused = File::open_swmr_writer_with_options(&swmr, buffered(1 << 20));
    assert!(
        matches!(
            &refused,
            Err(Error::EditUnsupported(
                "the SWMR writer cannot buffer its writes: its readers observe the order they become visible in; leave FileAccessProperties::with_page_buffer_size unset to open it"
            ))
        ),
        "the SWMR writer must refuse a page buffer, got {refused:?}"
    );
    // A rejected SWMR open must leave the write-status flags clear.
    untouched("swmr", &swmr, &before);
}

#[derive(Clone, Copy)]
enum DefaultOpen {
    Paged,
    Swmr,
    Unpaged,
}

#[rstest]
#[case::paged(DefaultOpen::Paged)]
#[case::swmr(DefaultOpen::Swmr)]
#[case::unpaged(DefaultOpen::Unpaged)]
fn default_access_properties_open_without_a_page_buffer(#[case] mode: DefaultOpen) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("default_access.h5");
    let mut builder = FileBuilder::new();
    if matches!(mode, DefaultOpen::Paged) {
        builder.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::Page)
                .with_persistence(true)
                .with_page_size(FileSpacePageSize::try_from(16 * 1024).unwrap()),
        );
    }
    builder
        .create_dataset("d")
        .with_i32_data(&[1, 2, 3, 4])
        .with_shape(&[4]);
    builder.write(&path).unwrap();
    let file = match mode {
        DefaultOpen::Swmr => {
            File::open_swmr_writer_with_options(&path, FileAccessProperties::new())
        }
        DefaultOpen::Paged | DefaultOpen::Unpaged => {
            File::open_rw_with_options(&path, FileAccessProperties::new())
        }
    }
    .unwrap();
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [1, 2, 3, 4]);
    file.close().unwrap();
}

// Invalid creation/access pairs must be rejected before creating the path.
#[rstest]
#[case::page_larger_than_budget(
    paged_create_properties(FileSpacePageSize::try_from(2 << 20).unwrap()),
    1 << 20, SyncPolicy::OnClose,
    "a page buffer smaller than the file's file-space page size would be refused at open — so creating one this way would write the file and then fail to open it; raise with_page_buffer_size",
)]
#[case::v18(
    crate::FileCreateProperties::new().with_libver_bounds(crate::LibVer::Earliest, crate::LibVer::V18),
    1 << 20, SyncPolicy::OnClose,
    "a page buffer marks the file in a version-3 superblock, and these bounds write the 1.8 format, so creating one this way would write the file and then fail to open it; raise with_libver_bounds to admit LibVer::V110, or drop the page buffer",
)]
#[case::budget_under_one_page(
    paged_create_properties(FileSpacePageSize::try_from(16 * 1024).unwrap()),
    8192, SyncPolicy::OnClose,
    "a page buffer smaller than the file's file-space page size would be refused at open — so creating one this way would write the file and then fail to open it; raise with_page_buffer_size",
)]
#[case::unpaged_budget_under_default_page(
    crate::FileCreateProperties::new(),
    2048,
    SyncPolicy::OnClose,
    "a page buffer smaller than the file's file-space page size would be refused at open — so creating one this way would write the file and then fail to open it; raise with_page_buffer_size"
)]
#[case::default_sync_policy(
    paged_create_properties(FileSpacePageSize::try_from(16 * 1024).unwrap()),
    1 << 20, SyncPolicy::Always,
    "a page buffer does nothing under the default SyncPolicy::Always, whose every barrier is an fsync that flushes it, so creating one this way would write the file and then fail to open it; add with_sync_policy(SyncPolicy::OnClose) to the access properties, or drop the page buffer",
)]
fn create_with_options_rejects_a_page_buffer_it_could_not_reopen_with(
    #[case] create: crate::FileCreateProperties,
    #[case] budget: usize,
    #[case] policy: SyncPolicy,
    #[case] reason: &str,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rejected.h5");
    let err = File::create_with_options(
        &path,
        create,
        FileAccessProperties::new()
            .with_sync_policy(policy)
            .with_page_buffer_size(budget),
    )
    .unwrap_err();
    let Error::EditUnsupported(message) = err else {
        panic!("expected EditUnsupported, got {err:?}");
    };
    assert_eq!(message, reason);
    assert!(!path.exists(), "the rejected creation wrote a file");
}

#[rstest]
#[case::paged(paged_create_properties(FileSpacePageSize::try_from(16 * 1024).unwrap()), 1 << 20)]
#[case::unpaged(crate::FileCreateProperties::new(), 1 << 20)]
#[case::paged_small_budget(paged_create_properties(FileSpacePageSize::try_from(16 * 1024).unwrap()), 256 * 1024)]
fn create_with_options_accepts_a_supported_page_buffer(
    #[case] create: crate::FileCreateProperties,
    #[case] budget: usize,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("accepted.h5");
    let file = File::create_with_options(
        &path,
        create,
        FileAccessProperties::new()
            .with_sync_policy(SyncPolicy::OnClose)
            .with_page_buffer_size(budget),
    )
    .unwrap();
    assert_eq!(file.root().datasets().unwrap(), Vec::<String>::new());
    assert_eq!(file.root().groups().unwrap(), Vec::<String>::new());
    file.close().unwrap();
    let reopened = File::open(&path).unwrap();
    assert_eq!(reopened.root().datasets().unwrap(), Vec::<String>::new());
    assert_eq!(reopened.root().groups().unwrap(), Vec::<String>::new());
}

fn paged_create_properties(page_size: FileSpacePageSize) -> crate::FileCreateProperties {
    crate::FileCreateProperties::new().with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::Page)
            .with_persistence(true)
            .with_page_size(page_size),
    )
}

// Count synchronization calls because a missing barrier does not change file bytes.
#[test]
fn close_and_drop_force_their_barrier_under_every_policy() {
    use crate::edit::WriteEngine;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    let dir = tempfile::tempdir().unwrap();
    let teardown = |name: &str, swmr: bool, explicit: bool| -> u64 {
        let path = dir.path().join(name);
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..8).collect::<Vec<_>>())
            .with_shape(&[8])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[4]);
        // The persisting fixture requires manager settlement during ordinary teardown.
        b.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
        );
        b.write(&path).unwrap();

        let syncs = Arc::new(AtomicU64::new(0));
        let session =
            WriteEngine::open_sync_counting(&path, SyncPolicy::OnClose, Arc::clone(&syncs))
                .unwrap();
        let mut inner = FileInner::from_rw_session(
            session,
            FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
        )
        .unwrap();
        inner.swmr_write = swmr;
        let file = File {
            inner: Arc::new(inner),
        };
        if !swmr {
            // Give ordinary teardown pending append work.
            file.dataset("d")
                .unwrap()
                .append(&[8i32, 9, 10, 11])
                .unwrap();
        }
        assert_eq!(
            syncs.load(AtomicOrdering::Relaxed),
            0,
            "nothing before teardown may sync under OnClose ({name})"
        );

        if explicit {
            file.close().unwrap();
        } else {
            drop(file);
        }
        syncs.load(AtomicOrdering::Relaxed)
    };

    for (name, swmr, explicit) in [
        ("close_plain.h5", false, true),
        ("close_swmr.h5", true, true),
        ("drop_plain.h5", false, false),
        ("drop_swmr.h5", true, false),
    ] {
        assert!(
            teardown(name, swmr, explicit) > 0,
            "{name} must force its barrier: the writes it makes are past the \
             last point a caller could have ordered them"
        );
    }
}
/// The signature of a version 0 Extensible Array index block, defined in "The Extensible Array
/// Index" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_appendixc_extarr
const EA_INDEX_SIGNATURE: &[u8; 4] = b"EAIB";
