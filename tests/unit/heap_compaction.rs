use std::fs;
use std::io::ErrorKind;
use std::ops::Range;
use std::path::Path;

use rstest::rstest;
use test_util::disk_log::{self, FailurePoint, IoEvent, IoOutcome, Recording};
use test_util::global_heap::{self, Object};
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use crate::{AttrValue, Error, File, FileAccessProperties, FileLocking, HeapCompactionState};
use crate::{MemoryStrategy, SyncPolicy};

const HEADER: usize = 16;
const RECORD: usize = 24;
const PARTIAL: usize = 2;

#[derive(Clone, Copy)]
enum Failure {
    FirstParent,
    Restoration,
    LaterParent,
}

impl Failure {
    fn parent(self) -> usize {
        match self {
            Self::FirstParent | Self::Restoration => 0,
            Self::LaterParent => 1,
        }
    }

    fn assert_error(self, error: Error) {
        let Error::HeapCompactionFailed {
            completed,
            state,
            source,
        } = error
        else {
            panic!("expected a heap compaction failure, got {error:?}");
        };
        assert_eq!(completed, self.parent());
        if matches!(self, Self::Restoration) {
            assert_eq!(state, HeapCompactionState::Uncertain);
            let Error::CommitPartiallyApplied { refusal, restore } = *source else {
                panic!("expected a restoration failure, got {source:?}");
            };
            assert_io(*restore, "injected disk synchronization failure");
            assert_io(*refusal, "injected disk write failure");
        } else {
            assert_eq!(state, HeapCompactionState::Restored);
            assert_io(*source, "injected disk write failure");
        }
    }
}

#[rstest]
#[case::first_parent(Failure::FirstParent)]
#[case::unconfirmed_restoration(Failure::Restoration)]
#[case::later_parent(Failure::LaterParent)]
fn public_heap_compaction_failure_keeps_the_published_edit(#[case] failure: Failure) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failure.h5");
    let control_path = directory.path().join("control.h5");
    let parents = write_source(&path);
    fs::copy(&path, &control_path).unwrap();

    let control = open_editor(&control_path);
    let original = fs::read(&control_path).unwrap();
    stage(&control);
    disk_log::start();
    control.commit().unwrap();
    let recording = disk_log::finish();
    let published = fs::read(&control_path).unwrap();
    assert_published_values(&control);
    assert_parents(&parents, &original, &published, parents.len());
    assert_allocated(&control, &parents);
    let event = movement(&recording, parents[failure.parent()]);
    control.close().unwrap();

    let file = open_editor(&path);
    let before = fs::read(&path).unwrap();
    assert_eq!(before, original);
    assert_eq!(file.dataset("numbers").unwrap().read_i32().unwrap(), [17]);
    let attr = file.dataset("value").unwrap().attrs().unwrap();
    assert_eq!(
        attr.get("a"),
        Some(&AttrValue::VarLenAsciiStringArray(vec![
            "removed".into();
            3
        ]))
    );
    stage(&file);
    disk_log::start();
    let prefix = &recording.events[..event];
    disk_log::fail_write_after(
        prefix
            .iter()
            .filter(|event| matches!(event, IoEvent::Write { .. }))
            .count(),
        PARTIAL,
    );
    if matches!(failure, Failure::Restoration) {
        disk_log::fail_sync_after(
            prefix
                .iter()
                .filter(|event| matches!(event, IoEvent::Synchronize { .. }))
                .count(),
            FailurePoint::Before,
        );
    }
    let error = file.commit().unwrap_err();
    disk_log::clear_failures();
    let failed = disk_log::finish();
    failure.assert_error(error);
    assert!(matches!(
        failed.events[event],
        IoEvent::Write { offset, written: PARTIAL, outcome: IoOutcome::Failed(ErrorKind::Other), .. }
            if offset == parents[failure.parent()] + HEADER as u64
    ));
    if matches!(failure, Failure::Restoration) {
        assert!(matches!(
            failed.events.last(),
            Some(IoEvent::Synchronize {
                outcome: IoOutcome::Failed(ErrorKind::Other),
                ..
            })
        ));
    }

    let range = parent_range(parents[failure.parent()]);
    let mut issued = before;
    for operation in failed.events[..=event]
        .iter()
        .filter_map(IoEvent::operation)
    {
        failed.operations[operation].apply(&mut issued);
    }
    let changed_index = range.start + HEADER..range.start + HEADER + PARTIAL;
    assert_eq!(issued[changed_index.clone()], 91_u16.to_le_bytes());
    assert_ne!(issued[changed_index.clone()], original[changed_index]);

    let after = fs::read(&path).unwrap();
    assert_eq!(
        after[..Fixture::ROOT as usize],
        published[..Fixture::ROOT as usize]
    );
    assert_parents(&parents, &original, &after, failure.parent());
    assert_published_values(&file);
    let stored = File::from_bytes(after.clone()).unwrap();
    assert_published_values(&stored);
    stored.close().unwrap();
    assert!(!file.has_staged_edits());
    if matches!(failure, Failure::Restoration) {
        file.root().set_attr("after", AttrValue::I64(31)).unwrap();
        disk_log::start();
        for result in [file.commit(), file.sync()] {
            assert!(matches!(
                result.unwrap_err(),
                Error::AllocationPublicationUncertain
            ));
        }
        assert!(disk_log::finish().events.is_empty());
        assert_eq!(fs::read(&path).unwrap(), after);
        drop(file);
        File::clear_status_flags(&path).unwrap();
    } else {
        assert_allocated(&file, &parents);
        file.root().set_attr("after", AttrValue::I64(31)).unwrap();
        file.commit().unwrap();
        assert_parents(
            &parents,
            &original,
            &fs::read(&path).unwrap(),
            failure.parent(),
        );
        file.close().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_published_values(&file);
    let attrs = file.root().attrs().unwrap();
    let fresh = if matches!(failure, Failure::Restoration) {
        None
    } else {
        Some(AttrValue::I64(31))
    };
    assert_eq!(attrs.get("after"), fresh.as_ref());
    file.close().unwrap();
}

fn write_source(path: &Path) -> [u64; 3] {
    let mut fixture = Fixture::new();
    let parents = std::array::from_fn(|_| {
        fixture.collection(&[Object::new(7, b"removed"), Object::new(91, b"retained")])
    });
    let removed = parents
        .iter()
        .flat_map(|&parent| global_heap_fixture::reference(7, parent, 7))
        .collect::<Vec<_>>();
    let retained = parents
        .iter()
        .flat_map(|&parent| global_heap_fixture::reference(8, parent, 91))
        .collect::<Vec<_>>();
    let mut messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &crate::make_i32_type(),
        &17_i32.to_le_bytes(),
    );
    messages.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &removed,
    ));
    fixture.object("value", messages);
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &crate::make_i32_type(),
        &17_i32.to_le_bytes(),
    );
    fixture.object("numbers", messages);
    let messages =
        Form::CompactDataset.messages(&mut fixture, &global_heap_fixture::vlen_string(), &retained);
    fixture.object("keep", messages);
    fixture.write(path);
    parents
}

fn open_editor(path: &Path) -> File {
    File::open_rw_with_options(
        path,
        FileAccessProperties::new()
            .with_locking(FileLocking::Disabled)
            .with_memory_strategy(MemoryStrategy::Mirrored)
            .with_page_buffer_size(0)
            .with_sync_policy(SyncPolicy::Always),
    )
    .unwrap()
}

fn stage(file: &File) {
    file.dataset("value")
        .unwrap()
        .set_attr("a", AttrValue::I64(9))
        .unwrap();
    file.dataset("numbers")
        .unwrap()
        .write_staged(|builder| {
            builder.with_i32_data(&[23]);
        })
        .unwrap();
}

fn assert_published_values(file: &File) {
    assert_eq!(file.dataset("numbers").unwrap().read_i32().unwrap(), [23]);
    assert_eq!(
        file.dataset("keep").unwrap().read_string().unwrap(),
        ["retained"; 3]
    );
    let value = file.dataset("value").unwrap();
    assert_eq!(value.read_i32().unwrap(), [17]);
    assert_eq!(value.attrs().unwrap().get("a"), Some(&AttrValue::I64(9)));
}

fn movement(recording: &Recording, parent: u64) -> usize {
    let publication = recording
        .events
        .iter()
        .position(|event| matches!(event, IoEvent::Write { offset: 0, .. }))
        .unwrap();
    let movement = recording.events.iter().position(|event| matches!(
        event,
        IoEvent::Write { offset, requested, written, outcome: IoOutcome::Succeeded, .. }
            if *offset == parent + HEADER as u64 && *requested >= RECORD && written == requested
    )).unwrap_or_else(|| panic!("no compaction write for heap at {parent}: {:?}", recording.events));
    assert!(publication < movement);
    movement
}

fn parent_range(address: u64) -> Range<usize> {
    let start = usize::try_from(address).unwrap();
    start..start + global_heap::MIN_COLLECTION_SIZE
}

fn assert_parents(parents: &[u64], original: &[u8], observed: &[u8], completed: usize) {
    let packed = global_heap::collection(&[Object::new(91, b"retained")], Widths::EIGHT);
    for (index, &address) in parents.iter().enumerate() {
        let range = parent_range(address);
        let mut expected = original[range.clone()].to_vec();
        if index < completed {
            let written = HEADER + RECORD + HEADER;
            expected[..written].copy_from_slice(&packed[..written]);
        }
        assert_eq!(observed[range], expected, "heap parent {index}");
    }
}

fn assert_allocated(file: &File, parents: &[u64]) {
    let free = file.reusable_free_sections().unwrap();
    for &address in parents {
        let end = address + global_heap::MIN_COLLECTION_SIZE as u64;
        assert!(free.sections().iter().all(|section| {
            section.end_address().get() <= address || section.address().get() >= end
        }));
    }
}

fn assert_io(error: Error, message: &str) {
    let Error::Io(error) = error else {
        panic!("expected an I/O error, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), message);
}
