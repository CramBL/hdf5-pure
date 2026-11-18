use std::fs;

use test_util::disk_log::{self, IoEvent, IoOutcome, Synchronization};
use test_util::global_heap;
use test_util::widths::Widths;
use test_util_hdf5::file;
use test_util_hdf5::global_heap::attributes;
use test_util_hdf5::numeric::ByteOrder;

use crate::page_publication_failure::PagePublicationFailure;
use crate::{AttrValue, File, FileSpacePageSize, FileSpaceStrategy, SyncPolicy, UserblockSize};

use super::{HEADER, HeapWrite, RECORD, assert_io, editor_options, heap_write, replay_prefix};

#[test]
fn native_page_heap_candidates_survive_publication_recovery() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("recover.h5");
    let control_path = directory.path().join("control.h5");
    let native = file::libhdf5_page_builder(FileSpacePageSize::DEFAULT, UserblockSize::NONE, false)
        .with_fcpl(|properties| properties.obj_track_times(false))
        .create(&path)
        .unwrap();
    native
        .new_dataset_builder()
        .empty_as(ByteOrder::LittleEndian.i32_datatype())
        .shape([1])
        .create("numbers")
        .unwrap()
        .write(&[17_i32])
        .unwrap();
    attributes::write_ascii(&native, "first", "removed");
    attributes::write_ascii(&native, "second", "retained");
    native.close().unwrap();
    fs::copy(&path, &control_path).unwrap();

    let original = fs::read(&path).unwrap();
    let ranges = global_heap::collection_ranges(&original, Widths::EIGHT);
    let [parent] = ranges.as_slice() else {
        panic!("expected one shared parent: {ranges:?}");
    };
    let range = usize::try_from(parent.start).unwrap()..usize::try_from(parent.end).unwrap();
    let original_parent = &original[range.clone()];
    assert_eq!(original_parent.len(), global_heap::MIN_COLLECTION_SIZE);
    assert_eq!(&original_parent[2 * HEADER..2 * HEADER + 7], b"removed");
    assert_eq!(
        &original_parent[2 * HEADER + RECORD..2 * HEADER + RECORD + 8],
        b"retained"
    );

    let control = File::open_rw_with_options(&control_path, editor_options()).unwrap();
    let settings = control.file_space_settings().unwrap().unwrap();
    assert_eq!(settings.strategy(), FileSpaceStrategy::Page);
    assert!(!settings.persist());
    stage(&control);
    disk_log::start();
    control.commit().unwrap();
    let recording = disk_log::finish();
    let packed = control.disk_bytes().unwrap();
    assert_eq!(
        &packed[range.start + HEADER..range.start + HEADER + RECORD],
        &original_parent[HEADER + RECORD..HEADER + 2 * RECORD]
    );
    assert_values(&control, false);
    control.close().unwrap();
    let compaction = heap_write(&recording, parent.start, HeapWrite::Gathered);
    let publication = recording
        .events
        .iter()
        .position(|event| matches!(event, IoEvent::Write { offset: 0, .. }))
        .unwrap();
    assert!(publication < compaction);

    let file = File::open_page_publication_failure(
        &path,
        PagePublicationFailure::AfterPublish,
        SyncPolicy::Always,
        true,
    )
    .unwrap();
    assert_eq!(file.disk_bytes().unwrap(), original);
    assert_eq!(file.dataset("numbers").unwrap().read_i32().unwrap(), [17]);
    let before = file.root().attrs().unwrap();
    assert_eq!(before.len(), 2);
    assert_eq!(
        before["first"],
        AttrValue::VarLenAsciiString("removed".into())
    );
    assert_eq!(
        before["second"],
        AttrValue::VarLenAsciiString("retained".into())
    );
    stage(&file);
    disk_log::start();
    assert_io(
        file.commit().unwrap_err(),
        "injected PAGE publication failure",
    );
    let failed = disk_log::finish();
    assert_eq!(failed.events, recording.events[..=publication]);
    let published = file.disk_bytes().unwrap();
    assert_eq!(
        published,
        replay_prefix(&recording, &original, publication + 1)
    );
    assert_eq!(&published[range.clone()], original_parent);
    assert!(!file.has_staged_edits());
    let image = File::from_bytes(published.clone()).unwrap();
    assert_values(&image, false);
    image.close().unwrap();

    file.root().set_attr("after", AttrValue::I64(31)).unwrap();
    disk_log::start();
    file.sync().unwrap();
    let recovery = disk_log::finish();
    assert_eq!(
        recovery.events,
        [IoEvent::Synchronize {
            kind: Synchronization::All,
            outcome: IoOutcome::Succeeded,
        }]
    );
    assert!(recovery.operations.is_empty());
    assert_eq!(file.disk_bytes().unwrap(), published);
    assert!(file.has_staged_edits());
    assert_values(&file, false);
    file.commit().unwrap();
    assert!(!file.has_staged_edits());
    assert_values(&file, true);
    assert_eq!(&file.disk_bytes().unwrap()[range.clone()], original_parent);
    file.close().unwrap();
    assert_eq!(&fs::read(&path).unwrap()[range], original_parent);

    let native = file::libhdf5_open(&path);
    let mut names = native.attr_names().unwrap();
    names.sort();
    assert_eq!(names, ["after", "second"]);
    assert_eq!(
        native.attr("after").unwrap().read_scalar::<i64>().unwrap(),
        31
    );
    assert_eq!(
        attributes::ascii_values(&native.attr("second").unwrap()),
        ["retained"]
    );
    assert_eq!(
        native
            .dataset("numbers")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [23]
    );
    native.close().unwrap();
}

fn stage(file: &File) {
    file.root().remove_attr("first").unwrap();
    file.dataset("numbers")
        .unwrap()
        .write_staged(|builder| {
            builder.with_i32_data(&[23]);
        })
        .unwrap();
}

fn assert_values(file: &File, has_followup: bool) {
    let attrs = file.root().attrs().unwrap();
    assert_eq!(attrs.len(), usize::from(has_followup) + 1);
    assert_eq!(
        attrs["second"],
        AttrValue::VarLenAsciiString("retained".into())
    );
    assert_eq!(
        attrs.get("after"),
        has_followup.then_some(&AttrValue::I64(31))
    );
    assert_eq!(file.dataset("numbers").unwrap().read_i32().unwrap(), [23]);
}
