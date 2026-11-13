use std::path::Path;

use rstest::rstest;
use test_util::datatype;
use test_util::disk_log::{self, DiskOp, FailurePoint};
use test_util::global_heap::Object;
use test_util::object_header::v2::Header;
use test_util::object_header::{Message, MessageFlags, MessageType};
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use crate::edit::test_util as edit_test_util;
use crate::edit::{SyncPolicy, WriteEngine};
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;
use crate::{DatasetBuilder, Error, File};

mod public_outcomes;

#[rstest]
#[case::increase(2, false, false, 3)]
#[case::decrease(1, true, false, 2)]
#[case::balanced(1, false, false, 2)]
#[case::removed_dataset_user(1, false, true, 2)]
fn committed_type_updates_follow_reference_publication_order(
    #[case] copies: usize,
    #[case] extra_alias: bool,
    #[case] remove_source: bool,
    #[case] count: u32,
) {
    let path = test_util::temp::temp_path("ordered-source-counts.h5");
    let original = if extra_alias { 3 } else { 2 };
    let target = fixture(&path, extra_alias);
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    stage(&mut engine, copies, remove_source);
    if extra_alias {
        engine.delete(&ObjectPathBuf::parse("alias")).unwrap();
    }
    disk_log::start();
    engine.commit().unwrap();
    let writes = disk_log::take();
    let publication = writes
        .iter()
        .position(|write| matches!(write, DiskOp::Write { offset: 0, .. }))
        .unwrap();
    let update = writes.iter().position(|write| {
        matches!(write, DiskOp::Write { offset, bytes }
            if *offset == target + 12 && bytes == &count.to_le_bytes())
    });
    assert_eq!(update.is_some(), count != original);
    if let Some(update) = update {
        assert_eq!(update < publication, count > original);
    }
    let expected = type_header(count);
    assert_eq!(
        engine
            .image()
            .read_exact_at(target, expected.len())
            .unwrap(),
        expected
    );
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("copy0").unwrap().read_f64().unwrap(), [17.0]);
    if !remove_source {
        assert_eq!(file.dataset("value").unwrap().read_f64().unwrap(), [31.0]);
    }
}

#[rstest]
#[case::content_barrier(false)]
#[case::attempted_root(true)]
fn count_bearing_failures_preserve_the_commit_journal_boundary(#[case] publication: bool) {
    let path = test_util::temp::temp_path("count-journal-boundary.h5");
    let target = fixture(&path, false);
    let before = std::fs::read(&path).unwrap();
    let control = test_util::temp::temp_path("count-journal-control.h5");
    std::fs::write(&control, &before).unwrap();
    let mut reference = edit_test_util::open_engine(&control);
    reference
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    stage(&mut reference, 2, false);
    disk_log::start();
    reference.commit().unwrap();
    let writes = disk_log::take();
    let root = writes
        .iter()
        .filter_map(|operation| match operation {
            DiskOp::Write { offset, .. } => Some(*offset),
            DiskOp::SetLen(_) => None,
        })
        .position(|offset| offset == 0)
        .unwrap();
    drop(reference);
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine.set_sync_policy(SyncPolicy::Always);
    let free = engine
        .space
        .admitted_snapshot()
        .sections()
        .collect::<Vec<_>>();
    stage(&mut engine, 2, false);
    if publication {
        disk_log::fail_write_after(root, 0);
    } else {
        disk_log::fail_sync_after(0, FailurePoint::Before);
    }
    let error = engine.commit().unwrap_err();
    disk_log::clear_failures();
    let Error::Io(error) = error else {
        panic!("expected injected I/O failure, got {error:?}");
    };
    assert_eq!(error.kind(), std::io::ErrorKind::Other);
    assert_eq!(
        error.to_string(),
        if publication {
            "injected disk write failure"
        } else {
            "injected disk synchronization failure"
        }
    );
    assert_eq!(
        engine
            .space
            .admitted_snapshot()
            .sections()
            .collect::<Vec<_>>(),
        free
    );
    assert!(engine.staged.is_empty());
    assert_eq!(engine.publication.attempted(), publication);
    let expected = type_header(if publication { 3 } else { 2 });
    assert_eq!(
        engine
            .image()
            .read_exact_at(target, expected.len())
            .unwrap(),
        expected
    );
    if publication {
        let error = engine.commit().unwrap_err();
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{error:?}"
        );
    } else {
        assert_eq!(&engine.disk_bytes().unwrap()[..before.len()], before);
        stage(&mut engine, 2, false);
        engine.commit().unwrap();
    }
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("value").unwrap().read_f64().unwrap(), [31.0]);
}

#[test]
fn detached_retained_owners_preserve_shared_heap_collections() {
    let path = test_util::temp::temp_path("retained-count-owner.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"value")]);
    let mut kept = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, parent, 7),
    );
    kept.push(Message::reference_count(2));
    fixture.object("retained", kept.clone());
    kept.pop();
    fixture.object("deleted", kept);
    fixture.write(&path);
    let before = std::fs::read(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("retained")).unwrap();
    engine.delete(&ObjectPathBuf::parse("deleted")).unwrap();
    engine.commit().unwrap();
    let start = usize::try_from(parent).unwrap();
    assert_eq!(
        engine.image().read_exact_at(parent, 4_096).unwrap(),
        before[start..start + 4_096]
    );
    assert!(
        engine
            .space
            .admitted_snapshot()
            .sections()
            .all(|(extent, _)| {
                let start = engine.addresses().resolve(extent.start()).unwrap().get();
                start + extent.len().get() <= parent || start >= parent + 4_096
            })
    );
}

fn stage(engine: &mut WriteEngine, copies: usize, remove_source: bool) {
    for index in 0..copies {
        engine
            .copy(
                &ObjectPathBuf::parse("source"),
                &ObjectPathBuf::parse(&format!("copy{index}")),
            )
            .unwrap();
    }
    engine.delete(&ObjectPathBuf::parse("type")).unwrap();
    engine.delete(&ObjectPathBuf::parse("dead")).unwrap();
    if remove_source {
        engine.delete(&ObjectPathBuf::parse("source")).unwrap();
    } else {
        let mut replacement = DatasetBuilder::new("");
        replacement.with_f64_data(&[31.0]);
        engine
            .stage_dataset_write(&ObjectPathBuf::parse("value"), replacement)
            .unwrap();
    }
}

fn fixture(path: &Path, extra_alias: bool) -> u64 {
    let mut fixture = Fixture::new();
    let target = fixture.append_header(vec![
        Message::reference_count(if extra_alias { 3 } else { 2 }),
        Message::new(MessageType::DATATYPE, &datatype::f64_le()),
    ]);
    fixture.link("type", target);
    if extra_alias {
        fixture.link("alias", target);
    }
    let mut records =
        Form::CompactDataset.messages(&mut fixture, &crate::make_f64_type(), &17_f64.to_le_bytes());
    *records
        .iter_mut()
        .find(|message| message.msg_type == MessageType::DATATYPE)
        .unwrap() = Message::new(
        MessageType::DATATYPE,
        &datatype::committed_reference(target),
    )
    .with_flags(MessageFlags::SHARED);
    fixture.object("source", records);
    let records = Form::ContiguousDataset.messages(
        &mut fixture,
        &crate::make_f64_type(),
        &17_f64.to_le_bytes(),
    );
    fixture.object("value", records);
    let records = Form::ContiguousDataset.messages(
        &mut fixture,
        &crate::make_i32_type(),
        &vec![7_i32.to_le_bytes(); 1_024].concat(),
    );
    fixture.object("dead", records);
    fixture.write(path);
    target
}

fn type_header(count: u32) -> Vec<u8> {
    Header::new()
        .messages([
            Message::reference_count(count),
            Message::new(MessageType::DATATYPE, &datatype::f64_le()),
        ])
        .build()
}
