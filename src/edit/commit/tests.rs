use std::path::Path;

use rstest::rstest;
use test_util::__private::durability::FaultSchedule;
use test_util::datatype;
use test_util::disk_log::{self, DiskOp, FailurePoint};
use test_util::global_heap::{self, Object};
use test_util::object_header::v2::Header;
use test_util::object_header::{Message, MessageFlags, MessageType};
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::members::ParentImage;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use crate::edit::test_util::durability;
use crate::{AttrValue, File, FileAccessProperties, FileLocking, SyncPolicy};

mod count_edges;
mod public_failures;
mod public_outcomes;
mod published_counts;

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
    let original_bytes = std::fs::read(&path).unwrap();
    let parent = global_heap::collection_ranges(&original_bytes, Widths::EIGHT)[0].start;
    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new()
            .with_locking(FileLocking::Disabled)
            .with_page_buffer_size(0),
    )
    .unwrap();
    stage(&file, copies, remove_source, extra_alias);
    disk_log::start();
    file.commit().unwrap();
    let writes = disk_log::take();
    let publication = writes
        .iter()
        .position(|write| matches!(write, DiskOp::Write { offset: 0, .. }))
        .unwrap();
    let update = writes.iter().position(|write| match write {
        DiskOp::Write { offset, bytes } => {
            target
                .checked_add(12)
                .and_then(|field| field.checked_sub(*offset))
                .and_then(|relative| usize::try_from(relative).ok())
                .and_then(|relative| bytes.get(relative..relative + 4))
                == Some(count.to_le_bytes().as_slice())
        }
        DiskOp::SetLen(_) => false,
    });
    assert_eq!(update.is_some(), count != original);
    if let Some(update) = update {
        assert_eq!(update < publication, count > original);
    }
    let bytes = std::fs::read(&path).unwrap();
    let expected = type_header(count);
    ParentImage::read(&original_bytes, parent).assert_unchanged(&bytes);
    assert_eq!(
        &bytes[target as usize..target as usize + expected.len()],
        expected
    );
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("copy0").unwrap().read_f64().unwrap(), [17.0]);
    assert_eq!(
        file.dataset("keep").unwrap().read_string().unwrap(),
        ["retained"]
    );
    assert_eq!(
        file.dataset("heap").unwrap().attrs().unwrap()["a"],
        AttrValue::I64(9)
    );
    if !remove_source {
        assert_eq!(file.dataset("value").unwrap().read_f64().unwrap(), [31.0]);
    }
    file.close().unwrap();
}

#[rstest]
#[case::rollback(CountFailure::BeforePublication)]
#[case::uncertain(CountFailure::Publication)]
fn count_bearing_heap_failures_preserve_the_commit_journal_boundary(
    #[case] fault: CountFailure,
    #[values(false, true)] decrease: bool,
) {
    let path = test_util::temp::temp_path("count-heap-boundary.h5");
    let target = fixture(&path, decrease);
    let original = std::fs::read(&path).unwrap();
    let parent = global_heap::collection_ranges(&original, Widths::EIGHT)[0].start;
    let control = test_util::temp::temp_path("count-heap-control.h5");
    std::fs::write(&control, &original).unwrap();
    let properties = FileAccessProperties::new()
        .with_locking(FileLocking::Disabled)
        .with_page_buffer_size(0)
        .with_sync_policy(SyncPolicy::Always);
    let reference = File::open_rw_with_options(&control, properties).unwrap();
    stage(&reference, if decrease { 1 } else { 2 }, false, decrease);
    disk_log::start();
    reference.commit().unwrap();
    let recording = disk_log::finish();
    let publication = recording
        .events
        .iter()
        .position(|event| matches!(event, disk_log::IoEvent::Write { offset: 0, .. }))
        .unwrap();
    let schedule = match fault {
        CountFailure::BeforePublication => {
            let barrier = recording.events[..publication]
                .iter()
                .rposition(|event| matches!(event, disk_log::IoEvent::Synchronize { .. }))
                .unwrap();
            FaultSchedule::synchronize(&recording, barrier, FailurePoint::Before)
        }
        CountFailure::Publication => FaultSchedule::write(&recording, publication, 0),
    };
    reference.close().unwrap();
    let file = File::open_rw_with_options(&path, properties).unwrap();
    stage(&file, if decrease { 1 } else { 2 }, false, decrease);
    disk_log::start();
    schedule.inject();
    let error = file.commit().unwrap_err();
    disk_log::clear_failures();
    let failed = disk_log::finish();
    schedule.assert_failed(&failed);
    durability::assert_io(
        error,
        if matches!(fault, CountFailure::BeforePublication) {
            "injected disk synchronization failure"
        } else {
            "injected disk write failure"
        },
    );
    assert!(!file.has_staged_edits());
    let after = std::fs::read(&path).unwrap();
    let expected_count = match fault {
        CountFailure::BeforePublication => {
            if decrease {
                3
            } else {
                2
            }
        }
        CountFailure::Publication => 3,
    };
    let expected = type_header(expected_count);
    assert_eq!(
        &after[target as usize..target as usize + expected.len()],
        expected
    );
    ParentImage::read(&original, parent).assert_unchanged(&after);
    if matches!(fault, CountFailure::BeforePublication) {
        assert_eq!(&after[..original.len()], original);
        stage(&file, if decrease { 1 } else { 2 }, false, decrease);
        file.commit().unwrap();
        file.close().unwrap();
    } else if matches!(fault, CountFailure::Publication) {
        durability::assert_uncertain(file.commit().unwrap_err());
        drop(file);
        File::clear_status_flags(&path).unwrap();
    }
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("keep").unwrap().read_string().unwrap(),
        ["retained"]
    );
    assert_eq!(file.dataset("value").unwrap().read_f64().unwrap(), [31.0]);
    let mut names = file.root().datasets().unwrap();
    names.sort();
    assert_eq!(
        names,
        if matches!(fault, CountFailure::Publication) {
            vec!["dead", "heap", "keep", "source", "value"]
        } else if decrease {
            vec!["copy0", "heap", "keep", "source", "value"]
        } else {
            vec!["copy0", "copy1", "heap", "keep", "source", "value"]
        }
    );
    assert_eq!(
        file.dataset("heap").unwrap().attrs().unwrap()["a"],
        if matches!(fault, CountFailure::Publication) {
            AttrValue::VarLenAsciiStringArray(vec!["removed".into()])
        } else {
            AttrValue::I64(9)
        }
    );
    file.close().unwrap();
}

#[derive(Clone, Copy, Debug)]
enum CountFailure {
    BeforePublication,
    Publication,
}

fn stage(file: &File, copies: usize, remove_source: bool, extra_alias: bool) {
    for index in 0..copies {
        file.copy("source", &format!("copy{index}")).unwrap();
    }
    file.root().delete("type").unwrap();
    file.root().delete("dead").unwrap();
    if extra_alias {
        file.root().delete("alias").unwrap();
    }
    file.dataset("heap")
        .unwrap()
        .set_attr("a", AttrValue::I64(9))
        .unwrap();
    if remove_source {
        file.root().delete("source").unwrap();
    } else {
        file.dataset("value")
            .unwrap()
            .write_staged(|builder| {
                builder.with_f64_data(&[31.0]);
            })
            .unwrap();
    }
}

fn fixture(path: &Path, extra_alias: bool) -> u64 {
    fixture_counts(
        path,
        extra_alias,
        published_counts::Version::Two,
        1,
        if extra_alias { 3 } else { 2 },
    )[0]
}

fn fixture_counts(
    path: &Path,
    extra_alias: bool,
    version: published_counts::Version,
    targets: usize,
    original: u32,
) -> Vec<u64> {
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"removed"), Object::new(91, b"retained")]);
    let mut addresses = Vec::new();
    for index in 0..targets {
        let suffix = if index == 0 {
            String::new()
        } else {
            index.to_string()
        };
        let header = version.header(original);
        let target = if index == 0 {
            fixture.image.append(&header)
        } else {
            fixture.image.append_aligned(&header, 4096)
        } as u64;
        addresses.push(target);
        fixture.link(&format!("type{suffix}"), target);
        if extra_alias {
            fixture.link(&format!("alias{suffix}"), target);
        }
        let mut records = Form::CompactDataset.messages(
            &mut fixture,
            &crate::make_f64_type(),
            &17_f64.to_le_bytes(),
        );
        *records
            .iter_mut()
            .find(|message| message.msg_type == MessageType::DATATYPE)
            .unwrap() = Message::new(
            MessageType::DATATYPE,
            &datatype::committed_reference(target),
        )
        .with_flags(MessageFlags::SHARED);
        fixture.object(&format!("source{suffix}"), records);
    }
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

    let mut records =
        Form::CompactDataset.messages(&mut fixture, &crate::make_i32_type(), &17_i32.to_le_bytes());
    records.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(7, parent, 7),
    ));
    fixture.object("heap", records);
    let records = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(8, parent, 91),
    );
    fixture.object("keep", records);
    fixture.write(path);
    addresses
}

fn type_header(count: u32) -> Vec<u8> {
    Header::new()
        .messages([
            Message::reference_count(count),
            Message::new(MessageType::DATATYPE, &datatype::f64_le()),
        ])
        .build()
}
