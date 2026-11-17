//! Retirement, reuse, and publication-failure checks for fragmented chunked creations.

use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use ::test_util::__private::durability::FaultSchedule;
use ::test_util::disk_log;
use ::test_util::disk_log::FailurePoint;
use ::test_util::disk_log::IoEvent;
use ::test_util::disk_log::Recording;
use ::test_util::range;
use rstest::rstest;
use test_util_hdf5::__private::fragmented_creation;
use test_util_hdf5::file_space::chunk_holes;
use test_util_hdf5::file_space::persisted::PersistedStrategy;

use crate::File;
use crate::SyncPolicy;
use crate::dataspace::MaxExtent;
use crate::edit::WriteEngine;
use crate::edit::dataset::test_util::lifecycle;
use crate::edit::test_util;
use crate::error::Error;
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;
use crate::type_builders::DatasetBuilder;

#[rstest]
fn reopened_fragmented_creations_retire_complete_allocations_and_reuse_payload(
    #[values(PersistedStrategy::FsmAggr, PersistedStrategy::Page)] strategy: PersistedStrategy,
    #[values(false, true)] extensible: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("lifecycle.h5");
    let expected = fragmented_creation::write(&path, strategy, extensible);
    drop(lifecycle::retire_and_reuse(&path, "replacement", &expected));
}

#[rstest]
#[case::content_barrier(Fault::ContentBarrier)]
#[case::before_root(Fault::Root(0))]
#[case::partial_root(Fault::Root(17))]
#[case::completed_root(Fault::Root(usize::MAX))]
#[case::publication_barrier(Fault::PublicationBarrier)]
fn fragmented_replacements_preserve_journals_and_uncertain_ownership(
    #[case] fault: Fault,
    #[values(PersistedStrategy::FsmAggr, PersistedStrategy::Page)] strategy: PersistedStrategy,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failed.h5");
    let (file, _) = chunk_holes::prepare(&path, strategy);
    let free = file
        .reusable_free_sections()
        .unwrap()
        .sections()
        .iter()
        .map(|section| (section.address().get(), section.length().get()))
        .collect::<Vec<_>>();
    file.close().unwrap();
    let before = fs::read(&path).unwrap();
    let predecessors = lifecycle::owned_ranges(&path, "spacer0");
    let control = directory.path().join("control.h5");
    fs::copy(&path, &control).unwrap();
    let mut reference = open(&control);
    stage(&mut reference);
    disk_log::start();
    reference.commit().unwrap();
    let recording = disk_log::finish();
    drop(reference);
    let file = File::open(&control).unwrap();
    let chunks = file.dataset("spacer0").unwrap().chunks().unwrap();
    assert_eq!(chunks.len(), 16);
    assert!(chunks.iter().all(
        |chunk| free.iter().any(|&(start, length)| start <= chunk.address
            && chunk.address + chunk.storage_size <= start + length)
    ));
    assert!(free.iter().all(|&(_, length)| length < 4_096));
    assert_eq!(
        file.dataset("spacer0").unwrap().read_i32().unwrap(),
        values()
    );
    assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [53; 64]);
    let journal = file.dataset("d").unwrap().chunks().unwrap()[0].address;
    file.close().unwrap();
    let claimed = lifecycle::owned_ranges(&control, "spacer0");
    let root = recording
        .events
        .iter()
        .position(|event| matches!(event, IoEvent::Write { offset: 0, .. }))
        .unwrap();
    let content = recording.events[..root]
        .iter()
        .rposition(|event| matches!(event, IoEvent::Synchronize { .. }))
        .unwrap();
    let publication = root
        + 1
        + recording.events[root + 1..]
            .iter()
            .position(|event| matches!(event, IoEvent::Synchronize { .. }))
            .unwrap();
    assert!(content < root && root < publication);
    let mut engine = open(&path);
    let original = engine
        .space
        .admitted_snapshot()
        .sections()
        .collect::<Vec<_>>();
    stage(&mut engine);
    disk_log::start();
    let schedule = fault.schedule(&recording, content, root, publication);
    schedule.inject();
    let error = engine.commit().unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected the injected I/O error, got {error:?}")
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(
        error.to_string(),
        if matches!(fault, Fault::Root(_)) {
            "injected disk write failure"
        } else {
            "injected disk synchronization failure"
        }
    );
    disk_log::clear_failures();
    let actual = disk_log::finish();
    schedule.assert_failed(&actual);
    for range in &predecessors {
        let start = usize::try_from(range.start().get()).unwrap();
        let end = usize::try_from(range.end().get()).unwrap();
        assert_eq!(
            engine
                .image()
                .read_exact_at(range.start().get(), end - start)
                .unwrap(),
            before[start..end]
        );
    }
    if matches!(fault, Fault::ContentBarrier) {
        assert_eq!(
            engine
                .space
                .admitted_snapshot()
                .sections()
                .collect::<Vec<_>>(),
            original
        );
        let writes = actual
            .operations
            .iter()
            .filter_map(|operation| match operation {
                disk_log::DiskOp::Write { offset, bytes } if *offset == journal => {
                    Some(bytes.clone())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            writes,
            [
                vec![53_i32; 64]
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
                (0_i32..64).flat_map(i32::to_le_bytes).collect::<Vec<_>>()
            ]
        );
        stage(&mut engine);
        engine.commit().unwrap();
        drop(engine);
        let file = File::open(&path).unwrap();
        assert_eq!(
            file.dataset("spacer0").unwrap().read_i32().unwrap(),
            values()
        );
        assert_eq!(file.dataset("d").unwrap().read_i32().unwrap(), [53; 64]);
        file.close().unwrap();
    } else {
        let reusable = engine
            .space
            .reusable_sections()
            .map(|extent| (extent.start().get(), extent.len().get()))
            .collect::<Vec<_>>();
        for allocation in claimed.iter().chain(&predecessors) {
            assert_eq!(
                range::covered_len(
                    &reusable,
                    &(allocation.start().get()..allocation.end().get())
                ),
                0
            );
        }
        let issued = engine.image.disk_bytes().unwrap();
        disk_log::start();
        assert_uncertain(engine.commit().unwrap_err());
        assert_uncertain(engine.force_sync().unwrap_err());
        assert_uncertain(engine.finalize_persist().unwrap_err());
        drop(engine);
        assert_eq!(disk_log::finish(), Recording::default());
        assert_eq!(fs::read(&path).unwrap(), issued);
    }
}

/// Stages a fragmented replacement and an in-place value update.
fn stage(engine: &mut WriteEngine) {
    engine.delete(&ObjectPathBuf::parse("spacer0")).unwrap();
    let mut builder = DatasetBuilder::new("spacer0");
    builder
        .with_i32_data(&values())
        .with_chunks(&[64])
        .with_maxshape(&[MaxExtent::Unlimited]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("spacer0"), builder)
        .unwrap();
    let mut update = DatasetBuilder::new("d");
    update.with_i32_data(&[53; 64]);
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("d"), update)
        .unwrap();
}

/// Opens an unbuffered test engine with [`SyncPolicy::Always`].
fn open(path: &Path) -> WriteEngine {
    let mut engine = test_util::open_engine(path);
    engine.sync_policy = SyncPolicy::Always;
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine
}

fn values() -> Vec<i32> {
    (0..1_024).collect()
}

#[derive(Clone, Copy)]
enum Fault {
    ContentBarrier,
    PublicationBarrier,
    /// The root-write byte prefix, or [`usize::MAX`] for the complete write.
    Root(usize),
}

impl Fault {
    /// Selects the content barrier, root write, or publication barrier.
    ///
    /// Uses `content`, `root`, and `publication` as indices into [`Recording::events`].
    ///
    /// # Panics
    ///
    /// Panics if the selected index is out of bounds or the event kind does not match the fault.
    fn schedule(
        self,
        recording: &Recording,
        content: usize,
        root: usize,
        publication: usize,
    ) -> FaultSchedule<'_> {
        match self {
            Self::ContentBarrier | Self::PublicationBarrier => FaultSchedule::synchronize(
                recording,
                if matches!(self, Self::ContentBarrier) {
                    content
                } else {
                    publication
                },
                FailurePoint::Before,
            ),
            Self::Root(prefix) => FaultSchedule::write(recording, root, prefix),
        }
    }
}

fn assert_uncertain(error: Error) {
    assert!(
        matches!(error, Error::AllocationPublicationUncertain),
        "{error:?}"
    );
}
