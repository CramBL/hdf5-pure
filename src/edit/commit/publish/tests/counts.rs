use core::iter;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use hdf5_pure_format::__private::ObjectReferenceCount;
use hdf5_pure_format::__private::ObjectReferenceCountField;
use rstest::rstest;
use test_util::datatype;
use test_util::disk_log;
use test_util::disk_log::DiskOp;
use test_util::disk_log::FailurePoint;
use test_util::disk_log::IoEvent;
use test_util::disk_log::Recording;
use test_util::file_space_info;
use test_util::global_heap::Object;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags;
use test_util::object_header::MessageType;
use test_util::object_header::v2::Header;
use test_util::superblock::v2::Superblock;
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use crate::File;
use crate::PhysicalAddress;
use crate::edit::SyncPolicy;
use crate::edit::WriteEngine;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::commit::PublicationState;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::error::FormatError;
use crate::image::WriteBuffering;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;
use crate::type_builders;

#[rstest]
#[case::remaining_alias(2, false, 1)]
#[case::final_aliases(2, true, 0)]
#[case::manual_pin(3, true, 1)]
fn active_alias_commits_retire_only_zero_count_original_storage(
    #[case] original: u32,
    #[case] all: bool,
    #[case] remaining: u32,
) {
    let path = test_util::temp::temp_path("active-alias-counts.h5");
    let mut fixture = Fixture::new();
    let raw = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    let mut messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &type_builders::make_i32_type(),
        &[7_i32.to_le_bytes(), 11_i32.to_le_bytes()].concat(),
    );
    messages.push(Message::reference_count(original));
    let target = fixture.append_header(messages);
    fixture.link("a", target);
    fixture.link("b", target);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let header = engine.oh_chunk_spans(target).unwrap();
    engine.delete(&ObjectPathBuf::parse("a")).unwrap();
    if all {
        engine.delete(&ObjectPathBuf::parse("b")).unwrap();
    }
    engine.commit().unwrap();
    let reusable = engine.space.reusable_sections().collect::<Vec<_>>();
    for (start, end) in iter::once((raw, raw + 8)).chain(
        header
            .iter()
            .map(|range| (range.start().get(), range.end().get())),
    ) {
        assert_eq!(
            reusable
                .iter()
                .any(|extent| { extent.start().get() <= start && end <= extent.end().get() }),
            remaining == 0,
            "{start}..{end}: {reusable:?}"
        );
    }
    if remaining > 0 {
        assert_eq!(field(&engine, target).value(), remaining);
    }
    drop(engine);
    let reader = File::open(&path).unwrap();
    assert_absent(reader.dataset("a").unwrap_err(), "a");
    if all {
        assert_absent(reader.dataset("b").unwrap_err(), "b");
    } else {
        assert_eq!(reader.dataset("b").unwrap().read_i32().unwrap(), [7, 11]);
    }
}

#[rstest]
#[case::increase(false)]
#[case::decrease(true)]
fn active_count_writes_surround_the_reference_publication(#[case] decrease: bool) {
    let path = test_util::temp::temp_path("active-count-order.h5");
    let target = copy_fixture(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine.set_sync_policy(SyncPolicy::Always);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let before = field(&engine, target);
    if decrease {
        engine.delete(&ObjectPathBuf::parse("type")).unwrap();
    } else {
        engine
            .copy(
                &ObjectPathBuf::parse("source"),
                &ObjectPathBuf::parse("copy"),
            )
            .unwrap();
    }
    disk_log::start();
    engine.commit().unwrap();
    let recording = disk_log::finish();
    let update = event_index(&recording, before.range().start().get());
    let reference = event_index(&recording, 0);
    assert_eq!(update < reference, !decrease);
    assert_eq!(field(&engine, target).value(), if decrease { 1 } else { 3 });
    assert_eq!(engine.publication, PublicationState::Published);
    drop(engine);
    let reader = File::open(&path).unwrap();
    assert_eq!(reader.dataset("source").unwrap().read_i32().unwrap(), [17]);
    if !decrease {
        assert_eq!(reader.dataset("copy").unwrap().read_i32().unwrap(), [17]);
    }
}

#[rstest]
#[case::before_reference(false)]
#[case::uncertain_reference(true)]
fn active_count_failures_restore_or_protect_source_increases(
    #[case] publication: bool,
    #[values(false, true)] page: bool,
) {
    let path = test_util::temp::temp_path("active-count-failure.h5");
    let target = recovery_fixture(&path, page);
    let recording = {
        let mut engine = edit_test_util::open_engine(&path);
        engine.set_sync_policy(SyncPolicy::Always);
        engine
            .set_write_buffering(WriteBuffering::Unbuffered)
            .unwrap();
        engine
            .copy(
                &ObjectPathBuf::parse("source"),
                &ObjectPathBuf::parse("copy"),
            )
            .unwrap();
        let count = field(&engine, target);
        disk_log::start();
        engine.commit().unwrap();
        (disk_log::finish(), count)
    };
    recovery_fixture(&path, page);
    let mut engine = edit_test_util::open_engine(&path);
    engine.set_sync_policy(SyncPolicy::Always);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine
        .copy(
            &ObjectPathBuf::parse("source"),
            &ObjectPathBuf::parse("copy"),
        )
        .unwrap();
    let free = engine.space.reusable_sections().collect::<Vec<_>>();
    let address = if publication {
        0
    } else {
        recording.1.range().start().get()
    };
    let index = recording
        .0
        .operations
        .iter()
        .filter(|operation| matches!(operation, DiskOp::Write { .. }))
        .position(
            |operation| matches!(operation, DiskOp::Write { offset, .. } if *offset == address),
        )
        .unwrap();
    disk_log::start();
    if page && publication {
        let sync = recording
            .0
            .events
            .iter()
            .take(event_index(&recording.0, 0))
            .filter(|event| matches!(event, IoEvent::Synchronize { .. }))
            .count();
        disk_log::fail_sync_after(sync, FailurePoint::Before);
    } else {
        disk_log::fail_write_after(index, 0);
    }
    let error = engine.commit().unwrap_err();
    disk_log::finish();
    disk_log::clear_failures();
    let Error::Io(error) = error else {
        panic!("expected injected I/O error, got {error:?}")
    };
    assert_eq!(error.kind(), std::io::ErrorKind::Other);
    assert_eq!(
        error.to_string(),
        if page && publication {
            "injected disk synchronization failure"
        } else {
            "injected disk write failure"
        }
    );
    assert_eq!(engine.publication.attempted(), publication);
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), free);
    assert_eq!(
        field(&engine, target).value(),
        if publication { 3 } else { 2 }
    );
    if publication && !page {
        assert!(matches!(
            engine.context.allocations.before_reference(),
            Err(Error::AllocationPublicationUncertain)
        ));
    } else {
        engine.context.allocations.before_reference().unwrap();
        if page && publication {
            engine.force_sync().unwrap();
            assert_eq!(engine.publication, PublicationState::Ready);
            assert_eq!(field(&engine, target).value(), 3);
            drop(engine);
            let reader = File::open(&path).unwrap();
            assert_eq!(reader.dataset("copy").unwrap().read_i32().unwrap(), [17]);
        }
    }
}

#[rstest]
#[case::eligible(CollectionOwner::Eligible)]
#[case::manual_pin(CollectionOwner::Retained)]
#[case::unknown_count(CollectionOwner::Unknown)]
fn active_collection_union_protects_retained_and_unknown_count_owners(
    #[case] owner: CollectionOwner,
) {
    let path = test_util::temp::temp_path("active-count-collections.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"value")]);
    let mut messages = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, collection, 1),
    );
    messages.push(Message::reference_count(
        if owner == CollectionOwner::Retained {
            3
        } else {
            2
        },
    ));
    if owner == CollectionOwner::Unknown {
        messages.push(Message::reference_count(2));
    }
    let target = fixture.append_header(messages);
    fixture.link("a", target);
    fixture.link("b", target);
    let mut kept = Form::CompactDataset.messages(
        &mut fixture,
        &type_builders::make_i32_type(),
        &29_i32.to_le_bytes(),
    );
    kept.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, collection, 1),
    ));
    fixture.object("kept", kept);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let range = engine.collection_span(collection).unwrap();
    engine.delete(&ObjectPathBuf::parse("a")).unwrap();
    engine.delete(&ObjectPathBuf::parse("b")).unwrap();
    engine
        .remove_dataset_attr(&ObjectPathBuf::parse("kept"), "a")
        .unwrap();
    engine.commit().unwrap();
    assert_eq!(
        engine
            .space
            .reusable_sections()
            .any(|extent| extent.start().get() <= range.start().get()
                && range.end().get() <= extent.end().get()),
        owner == CollectionOwner::Eligible
    );
    if owner == CollectionOwner::Retained {
        assert_eq!(field(&engine, target).value(), 1);
    }
    drop(engine);
    let reader = File::open(&path).unwrap();
    assert_eq!(reader.dataset("kept").unwrap().read_i32().unwrap(), [29]);
    assert_eq!(
        reader.dataset("kept").unwrap().attrs().unwrap(),
        HashMap::new()
    );
}

#[derive(Clone, Copy, PartialEq)]
enum CollectionOwner {
    Eligible,
    Retained,
    Unknown,
}

fn recovery_fixture(path: &Path, page: bool) -> u64 {
    let target = copy_fixture(path);
    if page {
        let mut image = fs::read(path).unwrap();
        let extension = u64::try_from(image.len()).unwrap();
        image.extend(
            Header::new()
                .message(Message::new(
                    MessageType::FILE_SPACE_INFO,
                    &file_space_info::non_persistent_version_1(
                        Widths::EIGHT,
                        file_space_info::PAGE,
                        1,
                        4_096,
                    ),
                ))
                .build(),
        );
        image.resize(image.len().next_multiple_of(4_096), 0);
        let prefix = Superblock::new(Widths::EIGHT)
            .version(3)
            .extension_address(extension)
            .root_header_address(Fixture::ROOT)
            .eof_address(u64::try_from(image.len()).unwrap())
            .build();
        image[..prefix.len()].copy_from_slice(&prefix);
        fs::write(path, image).unwrap();
    }
    target
}

fn copy_fixture(path: &Path) -> u64 {
    let mut fixture = Fixture::new();
    let target = fixture.append_header(vec![
        Message::new(
            MessageType::DATATYPE,
            &hdf5_pure_format::__private::serialize_datatype(&type_builders::make_i32_type())
                .unwrap(),
        ),
        Message::reference_count(2),
    ]);
    fixture.link("type", target);
    let mut messages = Form::CompactDataset.messages(
        &mut fixture,
        &type_builders::make_i32_type(),
        &17_i32.to_le_bytes(),
    );
    *messages
        .iter_mut()
        .find(|message| message.msg_type == MessageType::DATATYPE)
        .unwrap() = Message::new(
        MessageType::DATATYPE,
        &datatype::committed_reference(target),
    )
    .with_flags(MessageFlags::SHARED);
    fixture.object("source", messages);
    fixture.write(path);
    target
}

fn field(engine: &WriteEngine, address: u64) -> ObjectReferenceCountField {
    let observed = AnalysisPhase::new(engine)
        .object_reference_count(ObjectId::new(PhysicalAddress::new(address)))
        .unwrap();
    let ObjectReferenceCount::Version2(field) = observed else {
        panic!("expected explicit version 2 count, got {observed:?}")
    };
    field
}

fn event_index(recording: &Recording, address: u64) -> usize {
    recording
        .events
        .iter()
        .position(|event| matches!(event, IoEvent::Write { offset, .. } if *offset == address))
        .unwrap()
}

fn assert_absent(error: Error, path: &str) {
    let Error::Format(FormatError::PathNotFound(actual)) = error else {
        panic!("expected a missing path, got {error:?}")
    };
    assert_eq!(actual, path);
}
