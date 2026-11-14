use std::fs;
use std::io::ErrorKind;

use test_util::disk_log;
use test_util::disk_log::IoEvent;
use test_util::global_heap::MIN_COLLECTION_SIZE;
use test_util::global_heap::Object;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use crate::AttrValue;
use crate::Error;
use crate::File;
use crate::HeapCompactionState;
use crate::PhysicalAddress;
use crate::source::BytesSource;
use crate::width::LengthWidth;

#[test]
fn public_commit_reports_a_later_compaction_failure_without_replaying_edits() {
    let path = test_util::temp::temp_path("public-later-compaction-failure.h5");
    let mut fixture = Fixture::new();
    let parents = [
        fixture.collection(&[Object::new(7, b"removed"), Object::new(91, b"retained")]),
        fixture.collection(&[Object::new(7, b"removed"), Object::new(91, b"retained")]),
        fixture.collection(&[Object::new(7, b"removed"), Object::new(91, b"retained")]),
    ];
    let references = parents
        .iter()
        .flat_map(|&parent| global_heap_fixture::reference(7, parent, 7))
        .collect::<Vec<_>>();
    let mut records = Form::ContiguousDataset.messages(
        &mut fixture,
        &crate::make_i32_type(),
        &17_i32.to_le_bytes(),
    );
    records.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &references,
    ));
    fixture.object("value", records);
    let records = Form::ContiguousDataset.messages(
        &mut fixture,
        &crate::make_i32_type(),
        &17_i32.to_le_bytes(),
    );
    fixture.object("numbers", records);
    fixture.write(&path);
    let original = fs::read(&path).unwrap();
    let file = File::open_rw(&path).unwrap();
    stage(&file);
    disk_log::start();
    file.commit().unwrap();
    let recording = disk_log::finish();
    let parent_length = u64::try_from(MIN_COLLECTION_SIZE).unwrap();
    let failed = recording.events.iter().position(|event| matches!(event, IoEvent::Write { offset, .. } if *offset >= parents[1] && *offset < parents[1].checked_add(parent_length).unwrap())).unwrap();
    let writes = recording.events[..failed]
        .iter()
        .filter(|event| matches!(event, IoEvent::Write { .. }))
        .count();
    file.close().unwrap();
    fs::write(&path, &original).unwrap();
    let file = File::open_rw(&path).unwrap();
    stage(&file);
    disk_log::start();
    disk_log::fail_write_after(writes, 2);
    let error = file.commit().unwrap_err();
    disk_log::finish();
    let Error::HeapCompactionFailed {
        completed,
        state,
        source,
    } = error
    else {
        panic!("unexpected public commit error: {error:?}");
    };
    assert_eq!(completed, 1);
    assert_eq!(state, HeapCompactionState::Restored);
    let Error::Io(error) = *source else {
        panic!("unexpected compaction source: {source:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    assert_eq!(file.dataset("numbers").unwrap().read_i32().unwrap(), [23]);
    assert_eq!(file.dataset("value").unwrap().read_i32().unwrap(), [17]);
    assert_eq!(
        file.dataset("value").unwrap().attrs().unwrap().remove("a"),
        Some(AttrValue::I64(9))
    );
    let after = fs::read(&path).unwrap();
    let first = usize::try_from(parents[0]).unwrap();
    assert_ne!(
        &after[first..first + MIN_COLLECTION_SIZE],
        &original[first..first + MIN_COLLECTION_SIZE]
    );
    let source = BytesSource::new(&after);
    let compacted = crate::global_heap::collect_members(
        &source,
        LengthWidth::Eight,
        PhysicalAddress::new(parents[0]),
        |_| {},
    )
    .unwrap();
    assert_eq!(
        compacted
            .members()
            .iter()
            .map(|member| member.index().get())
            .collect::<Vec<_>>(),
        [91]
    );
    let rest_start = usize::try_from(parents[1]).unwrap();
    let rest_length = MIN_COLLECTION_SIZE.checked_mul(2).unwrap();
    assert_eq!(
        &after[rest_start..rest_start + rest_length],
        &original[rest_start..rest_start + rest_length]
    );
    file.commit().unwrap();
    assert_eq!(
        &fs::read(&path).unwrap()[rest_start..rest_start + rest_length],
        &after[rest_start..rest_start + rest_length]
    );
    file.close().unwrap();
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("numbers").unwrap().read_i32().unwrap(), [23]);
    assert_eq!(
        file.dataset("value").unwrap().attrs().unwrap().remove("a"),
        Some(AttrValue::I64(9))
    );
}

/// Stages a numeric overwrite and an attribute replacement.
///
/// # Panics
///
/// Panics if either dataset cannot be opened or the edits cannot be staged.
fn stage(file: &File) {
    let mut dataset = file.dataset("value").unwrap();
    dataset.set_attr("a", AttrValue::I64(9)).unwrap();
    file.dataset("numbers")
        .unwrap()
        .write_staged(|data| {
            data.with_i32_data(&[23]);
        })
        .unwrap();
}
