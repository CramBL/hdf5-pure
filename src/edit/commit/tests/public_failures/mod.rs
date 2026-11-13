use std::collections::HashMap;
use std::error::Error as StdError;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use rstest::rstest;
use test_util::disk_log::{self, FailurePoint, IoEvent, IoOutcome, Recording};
use test_util::global_heap::{self, Object};
use test_util::object_header::Message;
use test_util::superblock;
use test_util::superblock::v2::Superblock;
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::members::{Buffering, ParentImage};
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use crate::{
    AttrValue, EditBacking, Error, File, FileAccessProperties, FileLocking, HeapCompactionState,
    PersistedFreeSpacePolicy, SyncPolicy,
};

test_util_hdf5::hdf5_1_10_tests! {
    mod native_crosschecks;
}

#[rstest]
fn public_heap_preparation_rejection_keeps_the_complete_staged_batch(
    #[values(EditBacking::Bounded, EditBacking::Mirrored)] backing: EditBacking,
    #[values(Buffering::Unbuffered, Buffering::Operation, Buffering::Session)] buffering: Buffering,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("rejected.h5");
    let fixture = Scenario::write(&path, true);
    let file = open(&path, backing, buffering, SyncPolicy::OnClose);
    fixture.stage(&file);
    file.root().delete("alias").unwrap();
    let before = file.disk_bytes().unwrap();
    let free = file.reusable_free_sections().unwrap();
    disk_log::start();
    let error = file.commit().unwrap_err();
    assert_contradiction(error, fixture.owner);
    assert!(disk_log::finish().events.is_empty());
    assert!(file.has_staged_edits());
    assert_eq!(file.disk_bytes().unwrap(), before);
    assert_eq!(file.reusable_free_sections().unwrap(), free);
    fixture.assert_values(&file, false, 17);
    disk_log::start();
    assert_contradiction(file.commit().unwrap_err(), fixture.owner);
    assert!(disk_log::finish().events.is_empty());
    assert!(file.has_staged_edits());
    assert_eq!(file.disk_bytes().unwrap(), before);
    drop(file);
    File::clear_status_flags(&path).unwrap();
    let reader = File::open(&path).unwrap();
    fixture.assert_values(&reader, false, 17);
    reader.close().unwrap();
    directory.close().unwrap();
}

#[rstest]
#[case::content(Fault::Content)]
#[case::unwritten_publication(Fault::Publication(0))]
#[case::written_publication(Fault::Publication(usize::MAX))]
#[case::first_move(Fault::Move(0))]
#[case::first_free_header(Fault::Free)]
#[case::later_move(Fault::Move(1))]
fn public_heap_write_failures_preserve_their_phase_and_completed_prefix(
    #[case] fault: Fault,
    #[values(EditBacking::Bounded, EditBacking::Mirrored)] backing: EditBacking,
    #[values(Buffering::Unbuffered, Buffering::Operation)] buffering: Buffering,
) {
    fault.run(backing, buffering, SyncPolicy::OnClose);
}

#[rstest]
#[case::first_move(Fault::Move(0))]
#[case::first_free_header(Fault::Free)]
#[case::later_move(Fault::Move(1))]
fn public_heap_session_buffer_failures_keep_the_published_tree(
    #[case] fault: Fault,
    #[values(EditBacking::Bounded, EditBacking::Mirrored)] backing: EditBacking,
) {
    fault.run(backing, Buffering::Session, SyncPolicy::OnClose);
}

#[rstest]
#[case::content(Fault::ContentBarrier)]
#[case::publication(Fault::PublicationBarrier)]
#[case::collection(Fault::CollectionBarrier)]
#[case::restoration(Fault::RestorationBarrier)]
fn public_heap_barrier_failures_preserve_restoration_and_allocation_evidence(
    #[case] fault: Fault,
    #[values(EditBacking::Bounded, EditBacking::Mirrored)] backing: EditBacking,
    #[values(Buffering::Unbuffered, Buffering::Operation)] buffering: Buffering,
) {
    fault.run(backing, buffering, SyncPolicy::Always);
}

/// Stores original parent images and the aliased numeric owner for public commit failures.
struct Scenario {
    parents: [u64; 3],
    owner: u64,
    original: Vec<u8>,
}

impl Scenario {
    /// Writes heap-backed attributes, numeric data, and an aliased owner with a selected count.
    ///
    /// `contradiction` selects a stored owner count of one for two known hard links.
    ///
    /// # Panics
    ///
    /// Panics if fixture construction, writing, or reading fails.
    fn write(path: &Path, contradiction: bool) -> Self {
        let mut fixture = Fixture::new();
        let parents = std::array::from_fn(|_| {
            fixture.collection(&[Object::new(7, b"removed"), Object::new(91, b"retained")])
        });
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
        let mut records = Form::CompactDataset.messages(
            &mut fixture,
            &crate::make_i32_type(),
            &19_i32.to_le_bytes(),
        );
        records.push(Message::reference_count(if contradiction { 1 } else { 2 }));
        let owner = fixture.append_header(records);
        fixture.link("alias", owner);
        fixture.link("other", owner);
        let records = Form::ContiguousDataset.messages(
            &mut fixture,
            &crate::make_i32_type(),
            &17_i32.to_le_bytes(),
        );
        fixture.object("numbers", records);
        fixture.write(path);
        Self {
            parents,
            owner,
            original: fs::read(path).unwrap(),
        }
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
    fn assert_values(&self, file: &File, published: bool, number: i32) {
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
    }

    /// Checks that every complete parent range is excluded from reusable free sections.
    fn assert_allocated(&self, file: &File) {
        let free = file.reusable_free_sections().unwrap();
        for &parent in &self.parents {
            let end = parent
                .checked_add(u64::try_from(global_heap::MIN_COLLECTION_SIZE).unwrap())
                .unwrap();
            assert!(
                free.sections()
                    .iter()
                    .all(|section| section.end_address().get() <= parent
                        || section.address().get() >= end)
            );
        }
    }

    /// Checks compacted bytes for the completed prefix and original bytes for every later parent.
    fn assert_parents(&self, image: &[u8], completed: usize) {
        let Self {
            parents,
            owner: _,
            original,
        } = self;
        for (index, &address) in parents.iter().enumerate() {
            let parent = ParentImage::read(original, address);
            assert_eq!(parent.indices(), [7, 91]);
            if index < completed {
                parent.assert_compacted(image, &[91]);
            } else {
                parent.assert_unchanged(image);
            }
        }
    }
}

/// Selects an operation or synchronization boundary from a successful public commit.
#[derive(Clone, Copy, Debug)]
enum Fault {
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
    /// Executes a control commit, injects the selected failure, and checks public outcomes.
    ///
    /// Checks issued bytes before retry or close, then checks permitted edits and reopened values.
    ///
    /// # Panics
    ///
    /// Panics if fixture operations fail or any byte, error, allocation, or public value differs.
    fn run(self, backing: EditBacking, buffering: Buffering, sync: SyncPolicy) {
        let fault = self;
        let directory = tempfile::tempdir().unwrap();
        let control = directory.path().join("control.h5");
        let path = directory.path().join("failure.h5");
        let fixture = Scenario::write(&control, false);
        fs::write(&path, &fixture.original).unwrap();
        let file = open(&control, backing, buffering, sync);
        fixture.stage(&file);
        disk_log::start();
        file.commit().unwrap();
        let control = disk_log::finish();
        fixture.assert_parents(&file.disk_bytes().unwrap(), 3);
        let schedule = fault.schedule(&control, &fixture.parents);
        file.close().unwrap();
        let file = open(&path, backing, buffering, sync);
        fixture.stage(&file);
        let original_free = file.reusable_free_sections().unwrap();
        disk_log::start();
        schedule.inject();
        let error = file.commit().unwrap_err();
        disk_log::clear_failures();
        let failed = disk_log::finish();
        schedule.assert_failed(&failed);
        fault.assert_error(error);
        assert!(!file.has_staged_edits());
        let published = fault.confirmed();
        fixture.assert_values(&file, published, if fault.before() { 17 } else { 23 });
        let after = file.disk_bytes().unwrap();
        let stored = File::from_bytes(after.clone()).unwrap();
        fixture.assert_values(
            &stored,
            fault.confirmed()
                || matches!(
                    fault,
                    Fault::Publication(usize::MAX) | Fault::PublicationBarrier
                ),
            if fault.before() { 17 } else { 23 },
        );
        stored.close().unwrap();
        fixture.assert_parents(&after, fault.completed());
        if !fault.uncertain() {
            fixture.assert_allocated(&file);
        }
        fault.assert_publication(&control, &failed, &fixture.original, &after);
        if fault.before() {
            assert_eq!(file.reusable_free_sections().unwrap(), original_free);
        }
        if fault.uncertain() {
            let before = file.disk_bytes().unwrap();
            disk_log::start();
            assert_uncertain(file.commit().unwrap_err());
            assert_uncertain(file.sync().unwrap_err());
            assert!(disk_log::finish().events.is_empty());
            assert_eq!(file.disk_bytes().unwrap(), before);
            file.root().set_attr("after", AttrValue::I64(31)).unwrap();
            assert!(file.has_staged_edits());
            assert_uncertain(file.commit().unwrap_err());
            assert_eq!(file.disk_bytes().unwrap(), before);
            drop(file);
            File::clear_status_flags(&path).unwrap();
        } else {
            disk_log::start();
            file.commit().unwrap();
            assert!(disk_log::finish().events.is_empty());
            assert_eq!(file.disk_bytes().unwrap(), after);
            file.dataset("numbers")
                .unwrap()
                .write_staged(|builder| {
                    builder.with_i32_data(&[29]);
                })
                .unwrap();
            file.root().set_attr("after", AttrValue::I64(31)).unwrap();
            file.commit().unwrap();
            fixture.assert_allocated(&file);
            fixture.assert_values(&file, published, 29);
            assert_eq!(
                file.root().attrs().unwrap(),
                HashMap::from([("after".into(), AttrValue::I64(31))])
            );
            fixture.assert_parents(&file.disk_bytes().unwrap(), fault.completed());
            file.close().unwrap();
        }
        let reader = File::open(&path).unwrap();
        let on_disk = fault.confirmed()
            || matches!(
                fault,
                Fault::Publication(usize::MAX) | Fault::PublicationBarrier
            );
        fixture.assert_values(
            &reader,
            on_disk,
            if fault.before() {
                29
            } else if fault.uncertain() {
                23
            } else {
                29
            },
        );
        reader.close().unwrap();
        directory.close().unwrap();
    }

    /// Locates the selected failure in a successful recording for the ordered parent addresses.
    ///
    /// Write prefixes include bytes before the target within a merged request.
    ///
    /// # Panics
    ///
    /// Panics if a required event or parent is absent, or an event range cannot be represented.
    fn schedule(self, recording: &Recording, parents: &[u64]) -> Schedule {
        let events = &recording.events;
        let publication = events
            .iter()
            .position(|event| matches!(event, IoEvent::Write { offset: 0, .. }))
            .unwrap();
        let movement = |index: usize| {
            events
                .iter()
                .position(|event| {
                    contains(
                        event,
                        parents[index]
                            .checked_add(u64::try_from(global_heap::HEADER_SIZE).unwrap())
                            .unwrap(),
                    )
                })
                .unwrap()
        };
        let next_sync = |after: usize| {
            after
                + 1
                + events[after + 1..]
                    .iter()
                    .position(|event| matches!(event, IoEvent::Synchronize { .. }))
                    .unwrap()
        };
        let event = match self {
            Self::Content => events[..publication]
                .iter()
                .rposition(|event| matches!(event, IoEvent::Write { .. }))
                .unwrap(),
            Self::ContentBarrier => events[..publication]
                .iter()
                .rposition(|event| matches!(event, IoEvent::Synchronize { .. }))
                .unwrap(),
            Self::Publication(_) => publication,
            Self::PublicationBarrier => next_sync(publication),
            Self::Move(index) => movement(index),
            Self::Free => events
                .iter()
                .position(|event| {
                    contains(
                        event,
                        parents[0]
                            .checked_add(
                                u64::try_from(
                                    2 * global_heap::HEADER_SIZE
                                        + b"retained"
                                            .len()
                                            .next_multiple_of(global_heap::ALIGNMENT),
                                )
                                .unwrap(),
                            )
                            .unwrap(),
                    )
                })
                .unwrap(),
            Self::CollectionBarrier => next_sync(movement(0)),
            Self::RestorationBarrier => movement(1),
        };
        let is_write = matches!(events[event], IoEvent::Write { .. });
        let ordinal = events[..event]
            .iter()
            .filter(|event| {
                if is_write {
                    matches!(event, IoEvent::Write { .. })
                } else {
                    matches!(event, IoEvent::Synchronize { .. })
                }
            })
            .count();
        Schedule {
            event,
            write: if let IoEvent::Write { offset, .. } = events[event] {
                let prefix = if let Self::Publication(prefix) = self {
                    prefix
                } else {
                    let target = match self {
                        Self::Free => parents[0]
                            .checked_add(
                                u64::try_from(
                                    2 * global_heap::HEADER_SIZE
                                        + b"retained"
                                            .len()
                                            .next_multiple_of(global_heap::ALIGNMENT),
                                )
                                .unwrap(),
                            )
                            .unwrap(),
                        Self::Move(index) => parents[index]
                            .checked_add(u64::try_from(global_heap::HEADER_SIZE).unwrap())
                            .unwrap(),
                        _ => offset,
                    };
                    usize::try_from(target - offset).unwrap() + 2
                };
                Some((ordinal, prefix))
            } else {
                None
            },
            sync: if matches!(self, Self::RestorationBarrier) {
                Some(
                    events[..event]
                        .iter()
                        .filter(|event| matches!(event, IoEvent::Synchronize { .. }))
                        .count(),
                )
            } else if !is_write {
                Some(ordinal)
            } else {
                None
            },
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
    /// prefix. Bytes after that interval are compared with the successful prepublication writes.
    fn assert_publication(
        self,
        control: &Recording,
        failed: &Recording,
        original: &[u8],
        after: &[u8],
    ) {
        let superblock = fixture_superblock_size(original);
        let publication = failed
            .events
            .iter()
            .position(|event| matches!(event, IoEvent::Write { offset: 0, .. }));
        assert_eq!(publication.is_none(), self.before());
        if self.uncertain() && !self.confirmed() {
            let publication = publication.unwrap();
            assert!(
                failed.events[publication + 1..]
                    .iter()
                    .all(|event| !matches!(event, IoEvent::Write { .. } | IoEvent::Resize { .. }))
            );
            let mut retained = original.to_vec();
            let root = control
                .events
                .iter()
                .position(|event| matches!(event, IoEvent::Write { offset: 0, .. }))
                .unwrap();
            for event in &control.events[..root] {
                if let Some(operation) = event.operation() {
                    control.operations[operation].apply(&mut retained);
                }
            }
            assert!(after.len() >= retained.len());
            let IoEvent::Write { requested, .. } = control.events[root] else {
                panic!("expected a superblock write");
            };
            assert_eq!(requested, superblock);
            assert_eq!(&after[superblock..retained.len()], &retained[superblock..]);
        }
        let parent_end = original.len();
        assert!(after.len() >= parent_end);
    }
}

/// Stores zero-based event and operation positions for an injected failure.
struct Schedule {
    /// The expected failed event's position among all recorded events.
    event: usize,
    /// The write ordinal and the byte prefix issued before its failure.
    write: Option<(usize, usize)>,
    /// The synchronization ordinal that fails before completion.
    sync: Option<usize>,
}

impl Schedule {
    /// Schedules the selected write and synchronization failures.
    fn inject(&self) {
        let Self {
            event: _,
            write,
            sync,
        } = self;
        if let Some((ordinal, prefix)) = *write {
            disk_log::fail_write_after(ordinal, prefix);
        }
        if let Some(ordinal) = *sync {
            disk_log::fail_sync_after(ordinal, FailurePoint::Before);
        }
    }

    /// Checks the selected failed event and the exact number of failed operations.
    fn assert_failed(&self, recording: &Recording) {
        let Self { event, write, sync } = self;
        assert_eq!(
            recording.events[*event].outcome(),
            IoOutcome::Failed(ErrorKind::Other)
        );
        assert_eq!(
            recording
                .events
                .iter()
                .filter(|event| matches!(event.outcome(), IoOutcome::Failed(_)))
                .count(),
            usize::from(write.is_some()) + usize::from(sync.is_some())
        );
    }
}

/// Opens an editor with strict persisted free-space handling and checks its selected backing.
///
/// # Panics
///
/// Panics if opening fails or the selected backing is unavailable.
fn open(path: &Path, backing: EditBacking, buffering: Buffering, sync: SyncPolicy) -> File {
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

fn assert_contradiction(error: Error, owner: u64) {
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

fn assert_io(error: Error, message: &str) {
    let Error::Io(error) = error else {
        panic!("unexpected I/O error: {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), message);
}

fn assert_uncertain(error: Error) {
    assert!(
        matches!(error, Error::AllocationPublicationUncertain),
        "{error:?}"
    );
}

/// Returns whether a write request includes `target` in its half-open byte range.
fn contains(event: &IoEvent, target: u64) -> bool {
    matches!(event, IoEvent::Write { offset, requested, .. } if *offset <= target && target < offset.checked_add(u64::try_from(*requested).unwrap()).unwrap())
}

/// Returns the independent fixture superblock length after checking its source prefix.
///
/// # Panics
///
/// Panics if the source signature, version, or widths differ from the version 3 eight-byte fixture.
fn fixture_superblock_size(image: &[u8]) -> usize {
    let widths = Widths::EIGHT;
    let Widths { offset, length } = widths;
    let prefix = [
        superblock::SIGNATURE.as_slice(),
        &[
            superblock::VERSION_3,
            u8::try_from(offset).unwrap(),
            u8::try_from(length).unwrap(),
        ],
    ]
    .concat();
    assert_eq!(image.get(..prefix.len()), Some(prefix.as_slice()));
    Superblock::new(widths)
        .version(superblock::VERSION_3)
        .build()
        .len()
}
