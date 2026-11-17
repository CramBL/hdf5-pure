use std::collections::HashMap;
use std::error::Error as StdError;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use test_util::__private::durability::FaultSchedule;
use test_util::disk_log;
use test_util::disk_log::FailurePoint;
use test_util::disk_log::IoEvent;
use test_util::disk_log::Recording;
use test_util::global_heap;
use test_util_hdf5::__private::FailureSource;
use test_util_hdf5::global_heap_fixture::members::Buffering;
use test_util_hdf5::global_heap_fixture::members::ParentImage;

use crate::AttrValue;
use crate::EditBacking;
use crate::Error;
use crate::File;
use crate::FileAccessProperties;
use crate::FileLocking;
use crate::HeapCompactionState;
use crate::PersistedFreeSpacePolicy;
use crate::SyncPolicy;

test_util_hdf5::hdf5_1_10_tests! {
    pub(in crate::edit) mod native;
}

/// Original source bytes and parent addresses for commit-failure scenarios.
///
/// The source may include an aliased numeric owner.
pub(in crate::edit) struct Scenario {
    parents: [u64; 3],
    pub(in crate::edit) owner: Option<u64>,
    original: Vec<u8>,
}

impl Scenario {
    pub(in crate::edit) fn write(path: &Path, contradiction: bool, timestamp: u32) -> Self {
        Self::from_source(FailureSource::aliased(path, contradiction, timestamp))
    }

    fn from_source(source: FailureSource) -> Self {
        let FailureSource {
            parents,
            owner,
            raw: _,
            original,
        } = source;
        Self {
            parents,
            owner,
            original,
        }
    }
}

/// Selects an operation or synchronization boundary from a successful public commit.
#[derive(Clone, Copy, Debug)]
pub(in crate::edit) enum Fault {
    CollectionBarrier,
    Content,
    ContentBarrier,
    Free,
    /// Fails a move in the parent at this zero-based position.
    Move(usize),
    /// Fails a superblock write after this many bytes, capped at the request length.
    Publication(usize),
    PublicationBarrier,
    RestorationBarrier,
}

impl Fault {
    pub(in crate::edit) fn run(self, backing: EditBacking, buffering: Buffering, sync: SyncPolicy) {
        let directory = tempfile::tempdir().unwrap();
        let control = directory.path().join("control.h5");
        let path = directory.path().join("failure.h5");
        let reference = Scenario::write(&control, false, 1);
        let fixture = Scenario::write(&path, false, 2);
        assert_eq!(reference.parents, fixture.parents);
        assert_eq!(reference.owner, fixture.owner);
        assert_eq!(reference.original.len(), fixture.original.len());
        assert_ne!(reference.original, fixture.original);
        self.execute(
            &reference,
            &fixture,
            &control,
            &path,
            Opening::Selected(backing, buffering, sync),
            Retry::FreshEdit,
        );
        directory.close().unwrap();
    }

    /// Runs a control commit, failed commit, retry, and reopen through source-specific checks.
    ///
    /// Schedules the fault from the control recording and compares failed-attempt bytes with the
    /// fixture's original image. `retry` selects immediate close or a fresh edit after empty retry.
    pub(in crate::edit) fn execute(
        self,
        reference: &impl HeapScenario,
        fixture: &impl HeapScenario,
        control_path: &Path,
        path: &Path,
        opening: Opening,
        retry: Retry,
    ) {
        let file = opening.open(control_path);
        reference.assert_open(&file);
        reference.stage(&file);
        disk_log::start();
        file.commit().unwrap();
        let control = disk_log::finish();
        if matches!(retry, Retry::FreshEdit) {
            reference.assert_parents(&file.disk_bytes().unwrap(), reference.parents().len());
        }
        let schedule = self.schedule(&control, fixture.parents());
        file.close().unwrap();
        let file = opening.open(path);
        fixture.assert_open(&file);
        fixture.stage(&file);
        let original_free = file.reusable_free_sections().unwrap();
        disk_log::start();
        schedule.inject();
        let error = file.commit().unwrap_err();
        disk_log::clear_failures();
        let failed = disk_log::finish();
        schedule.assert_failed(&failed);
        self.assert_error(error);
        assert!(!file.has_staged_edits());
        let number = if self.before() { 17 } else { 23 };
        fixture.assert_values(&file, self.confirmed(), number, None);
        let after = file.disk_bytes().unwrap();
        let stored = File::from_bytes(after.clone()).unwrap();
        let on_disk = self.confirmed()
            || matches!(
                self,
                Self::Publication(usize::MAX) | Self::PublicationBarrier
            );
        fixture.assert_values(&stored, on_disk, number, None);
        stored.close().unwrap();
        fixture.assert_parents(&after, self.completed());
        if !self.uncertain() {
            fixture.assert_allocated(&file);
        }
        self.assert_publication(&failed, fixture.original(), &after);
        if self.before() {
            assert_eq!(file.reusable_free_sections().unwrap(), original_free);
        }
        let fresh = if self.uncertain() {
            disk_log::start();
            assert_uncertain(file.commit().unwrap_err());
            assert_uncertain(file.sync().unwrap_err());
            assert!(disk_log::finish().events.is_empty());
            assert_eq!(file.disk_bytes().unwrap(), after);
            file.root().set_attr("after", AttrValue::I64(31)).unwrap();
            assert!(file.has_staged_edits());
            assert_uncertain(file.commit().unwrap_err());
            assert_eq!(file.disk_bytes().unwrap(), after);
            drop(file);
            File::clear_status_flags(path).unwrap();
            None
        } else {
            disk_log::start();
            file.commit().unwrap();
            assert!(disk_log::finish().events.is_empty());
            assert_eq!(file.disk_bytes().unwrap(), after);
            if matches!(retry, Retry::FreshEdit) {
                fixture.advance(&file);
                file.root().set_attr("after", AttrValue::I64(31)).unwrap();
                file.commit().unwrap();
                fixture.assert_allocated(&file);
                fixture.assert_values(
                    &file,
                    self.confirmed(),
                    fixture.fresh_number(number),
                    Some(31),
                );
                fixture.assert_parents(&file.disk_bytes().unwrap(), self.completed());
            }
            file.close().unwrap();
            matches!(retry, Retry::FreshEdit).then_some(31)
        };
        let reader = File::open(path).unwrap();
        let reopened_number = if fresh.is_some() {
            fixture.fresh_number(number)
        } else {
            number
        };
        fixture.assert_values(&reader, on_disk, reopened_number, fresh);
        reader.close().unwrap();
        fixture.assert_closed(path, on_disk, reopened_number, fresh);
    }

    /// Locates the selected failure in a successful recording for the ordered parent addresses.
    ///
    /// # Panics
    ///
    /// Panics if a required event or parent is absent, or an event range cannot be represented.
    fn schedule<'a>(self, recording: &'a Recording, parents: &[u64]) -> FaultSchedule<'a> {
        let events = &recording.events;
        let publication = events
            .iter()
            .position(|event| matches!(event, IoEvent::Write { offset: 0, .. }))
            .unwrap();
        let movement = |index: usize| {
            parents[index]
                .checked_add(u64::try_from(global_heap::HEADER_SIZE).unwrap())
                .unwrap()
        };
        match self {
            Self::Content => {
                let event = events[..publication]
                    .iter()
                    .rposition(|event| matches!(event, IoEvent::Write { .. }))
                    .unwrap();
                FaultSchedule::write(recording, event, 2)
            }
            Self::ContentBarrier => {
                let event = events[..publication]
                    .iter()
                    .rposition(|event| matches!(event, IoEvent::Synchronize { .. }))
                    .unwrap();
                FaultSchedule::synchronize(recording, event, FailurePoint::Before)
            }
            Self::Publication(prefix) => FaultSchedule::write(recording, publication, prefix),
            Self::PublicationBarrier => {
                FaultSchedule::synchronize_after(recording, publication, FailurePoint::Before)
            }
            Self::Move(index) => FaultSchedule::write_containing(recording, movement(index), 2),
            Self::Free => {
                let target = parents[0]
                    .checked_add(
                        u64::try_from(
                            2 * global_heap::HEADER_SIZE
                                + b"retained".len().next_multiple_of(global_heap::ALIGNMENT),
                        )
                        .unwrap(),
                    )
                    .unwrap();
                FaultSchedule::write_containing(recording, target, 2)
            }
            Self::CollectionBarrier => {
                let movement = FaultSchedule::write_containing(recording, movement(0), 2);
                FaultSchedule::synchronize_after(recording, movement.event(), FailurePoint::Before)
            }
            Self::RestorationBarrier => FaultSchedule::write_containing(recording, movement(1), 2)
                .with_recovery_synchronization(FailurePoint::Before),
        }
    }

    /// Returns whether the selected operation precedes a publication attempt.
    fn before(self) -> bool {
        matches!(self, Self::Content | Self::ContentBarrier)
    }

    /// Returns whether publication is confirmed before the selected failure.
    fn confirmed(self) -> bool {
        matches!(
            self,
            Self::Move(_) | Self::Free | Self::CollectionBarrier | Self::RestorationBarrier
        )
    }

    /// Returns whether allocation publication is uncertain after the selected failure.
    fn uncertain(self) -> bool {
        matches!(
            self,
            Self::Publication(_) | Self::PublicationBarrier | Self::RestorationBarrier
        )
    }

    /// Returns the expected completed parent count after the selected failure.
    fn completed(self) -> usize {
        match self {
            Self::Move(index) => index,
            Self::RestorationBarrier => 1,
            _ => 0,
        }
    }

    /// Checks exact nested I/O errors, completed progress, restoration state, and error sources.
    fn assert_error(self, error: Error) {
        let source = if self.confirmed() {
            let chained = StdError::source(&error).unwrap().to_string();
            let Error::HeapCompactionFailed {
                completed,
                state,
                source,
            } = error
            else {
                panic!("unexpected public compaction error: {error:?}");
            };
            assert_eq!(chained, source.to_string());
            assert_eq!(completed, self.completed());
            assert_eq!(
                state,
                if self.uncertain() {
                    HeapCompactionState::Uncertain
                } else {
                    HeapCompactionState::Restored
                }
            );
            *source
        } else {
            error
        };
        let source = if matches!(self, Self::RestorationBarrier) {
            let chained = StdError::source(&source).unwrap().to_string();
            let Error::CommitPartiallyApplied { refusal, restore } = source else {
                panic!("unexpected restoration error: {source:?}");
            };
            assert_eq!(chained, restore.to_string());
            assert_io(*restore, "injected disk synchronization failure");
            *refusal
        } else {
            source
        };
        assert_io(
            source,
            if matches!(
                self,
                Self::ContentBarrier | Self::PublicationBarrier | Self::CollectionBarrier
            ) {
                "injected disk synchronization failure"
            } else {
                "injected disk write failure"
            },
        );
    }

    /// Checks the publication boundary and retention after an uncertain superblock write or
    /// barrier.
    ///
    /// The superblock interval is established from the independent fixture builder and source
    /// prefix. Reconstructs expected bytes after that interval from the failed attempt's completed
    /// prepublication operations.
    fn assert_publication(self, failed: &Recording, original: &[u8], after: &[u8]) {
        let superblock = FailureSource::superblock_size(original);
        let publication = failed
            .events
            .iter()
            .position(|event| matches!(event, IoEvent::Write { offset: 0, .. }));
        assert_eq!(publication.is_none(), self.before());
        if self.uncertain() && !self.confirmed() {
            let publication = publication.unwrap();
            failed.assert_publication_retained(original, after, publication, 0..superblock);
        }
        let parent_end = original.len();
        assert!(after.len() >= parent_end);
    }
}

/// Opens an editor with strict persisted free-space handling and checks its selected backing.
///
/// # Panics
///
/// Panics if opening fails or the selected backing is unavailable.
pub(in crate::edit) fn open(
    path: &Path,
    backing: EditBacking,
    buffering: Buffering,
    sync: SyncPolicy,
) -> File {
    let file = File::open_rw_with_options(
        path,
        FileAccessProperties::new()
            .with_memory_strategy(backing.into())
            .with_sync_policy(sync)
            .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict)
            .with_locking(if matches!(buffering, Buffering::Unbuffered) {
                FileLocking::Disabled
            } else {
                FileLocking::Enabled
            })
            .with_page_buffer_size(if matches!(buffering, Buffering::Session) {
                65_536
            } else {
                0
            }),
    )
    .unwrap();
    assert_eq!(file.edit_backing(), Some(backing));
    file
}

pub(in crate::edit) fn assert_contradiction(error: Error, owner: u64) {
    let Error::InconsistentReferenceCount {
        address,
        observed,
        minimum,
    } = error
    else {
        panic!("unexpected preparation rejection: {error:?}");
    };
    assert_eq!((address, observed, minimum), (owner, 1, 2));
}

pub(in crate::edit) fn assert_io(error: Error, message: &str) {
    let Error::Io(error) = error else {
        panic!("unexpected I/O error: {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), message);
}

pub(in crate::edit) fn assert_uncertain(error: Error) {
    assert!(
        matches!(error, Error::AllocationPublicationUncertain),
        "{error:?}"
    );
}

#[derive(Clone, Copy)]
pub(in crate::edit) enum Opening {
    Default,
    Selected(EditBacking, Buffering, SyncPolicy),
}

impl Opening {
    fn open(self, path: &Path) -> File {
        match self {
            Self::Default => File::open_rw(path).unwrap(),
            Self::Selected(backing, buffering, sync) => open(path, backing, buffering, sync),
        }
    }
}

#[derive(Clone, Copy)]
pub(in crate::edit) enum Retry {
    CloseImmediately,
    FreshEdit,
}

/// Source-specific edits and observations for shared commit-failure execution.
pub(in crate::edit) trait HeapScenario {
    fn parents(&self) -> &[u64];
    fn parent_range(&self, index: usize) -> FileRange;
    fn assert_parents(&self, image: &[u8], completed: usize);
    fn original(&self) -> &[u8];
    fn stage(&self, file: &File);
    /// Checks published or original values, the numeric value, and an optional fresh attribute.
    fn assert_values(&self, file: &File, published: bool, number: i32, after: Option<i64>);
    fn assert_open(&self, _file: &File) {}
    fn assert_closed(&self, _path: &Path, _published: bool, _number: i32, _after: Option<i64>) {}
    fn advance(&self, file: &File);
    fn fresh_number(&self, number: i32) -> i32;

    fn assert_allocated(&self, file: &File) {
        let free = file.reusable_free_sections().unwrap();
        for index in 0..self.parents().len() {
            let range = self.parent_range(index);
            assert!(
                free.sections()
                    .iter()
                    .all(|section| section.end_address().get() <= range.start().get()
                        || section.address().get() >= range.end().get())
            );
        }
    }
}

impl HeapScenario for Scenario {
    fn parent_range(&self, index: usize) -> FileRange {
        FileRange::new(
            PhysicalAddress::new(self.parents[index]),
            u64::try_from(global_heap::MIN_COLLECTION_SIZE).unwrap(),
        )
        .unwrap()
    }
    fn assert_parents(&self, image: &[u8], completed: usize) {
        for (index, &address) in self.parents.iter().enumerate() {
            let parent = ParentImage::read(&self.original, address);
            assert_eq!(parent.indices(), [7, 91]);
            if index < completed {
                parent.assert_compacted(image, &[91]);
            } else {
                parent.assert_unchanged(image);
            }
        }
    }
    fn parents(&self) -> &[u64] {
        &self.parents
    }
    fn original(&self) -> &[u8] {
        &self.original
    }
    /// Stages replacement of the heap-backed attribute with a scalar and numeric data with 23.
    ///
    /// # Panics
    ///
    /// Panics if the datasets cannot be opened or their edits cannot be staged.
    fn stage(&self, file: &File) {
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

    /// Checks the numeric values and complete attribute map before or after publication.
    ///
    /// `published` selects the expected attribute: the replacement scalar when true and original
    /// heap values when false.
    fn assert_values(&self, file: &File, published: bool, number: i32, after: Option<i64>) {
        assert_eq!(file.dataset("value").unwrap().read_i32().unwrap(), [17]);
        assert_eq!(
            file.dataset("numbers").unwrap().read_i32().unwrap(),
            [number]
        );
        assert_eq!(
            file.dataset("value").unwrap().attrs().unwrap(),
            HashMap::from([(
                "a".into(),
                if published {
                    AttrValue::I64(9)
                } else {
                    AttrValue::VarLenAsciiStringArray(vec!["removed".into(); self.parents.len()])
                }
            )])
        );
        assert_eq!(
            file.root().attrs().unwrap(),
            after
                .map(|value| HashMap::from([("after".into(), AttrValue::I64(value))]))
                .unwrap_or_default()
        );
    }
    fn advance(&self, file: &File) {
        file.dataset("numbers")
            .unwrap()
            .write_staged(|builder| {
                builder.with_i32_data(&[29]);
            })
            .unwrap();
    }
    fn fresh_number(&self, _number: i32) -> i32 {
        29
    }
}

pub(in crate::edit) fn later_parent() {
    let directory = tempfile::tempdir().unwrap();
    let control = directory.path().join("control.h5");
    let path = directory.path().join("failure.h5");
    let fixture = Scenario::from_source(FailureSource::later(&control, true));
    fs::write(&path, &fixture.original).unwrap();
    Fault::Move(1).execute(
        &fixture,
        &fixture,
        &control,
        &path,
        Opening::Default,
        Retry::CloseImmediately,
    );
    directory.close().unwrap();
}
