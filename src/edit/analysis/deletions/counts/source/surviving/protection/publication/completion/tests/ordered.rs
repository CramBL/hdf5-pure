use core::num::NonZeroU32;
use std::collections::BTreeMap;
use std::path::Path;

use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::ObjectReferenceCountUpdateError;
use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::SessionSpace;
use rstest::rstest;
use test_util::disk_log;
use test_util::disk_log::DiskOp;
use test_util::disk_log::FailurePoint;
use test_util::disk_log::IoEvent;
use test_util::disk_log::IoOutcome;
use test_util::disk_log::Recording;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;

use super::super::super::super::tests as protection_tests;
use super::super::super::prepare::CountPreparationError;
use super::super::super::test_support::Form;
use super::super::CountCompletionError;
use super::super::CountProtectionError;
use super::super::CountPublicationBounds;
use super::super::CountPublicationError;
use super::super::CountPublicationUpdateError;
use crate::UserblockSize;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::edit::SyncPolicy;
use crate::edit::WriteEngine;
use crate::edit::commit::PublicationState;
use crate::edit::group;
use crate::edit::object_header;
use crate::edit::retirement::RootRetirement;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::file_space_info::FileSpacePageSize;
use crate::file_space_info::FileSpaceStrategy;
use crate::image::WriteBuffering;
use crate::message_type::MessageType;
use crate::object_path::LinkNameBuf;

#[rstest]
#[case::v1(Form::Version1)]
#[case::v2(Form::Version2)]
fn ordered_counts_surround_root_publication_and_preserve_current_headers(
    #[case] form: Form,
    #[values(0, 512)] base: u64,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] policy: SyncPolicy,
) {
    let path = test_util::temp::temp_path("ordered-counts.h5");
    let objects = fixture(&path, form, base);
    let mut engine = edit_test_util::open_engine(&path);
    engine.set_sync_policy(policy);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let bounds = bounds(&mut engine, objects, base);
    for object in objects {
        bounds
            .phase
            .engine
            .write_inplace_journaled(object + base, &form.chunk(2, c"new"))
            .unwrap();
    }
    let prior = bounds
        .phase
        .engine
        .inplace_undo
        .iter()
        .map(|(at, bytes)| (*at, bytes.as_ptr()))
        .collect::<Vec<_>>();
    let root = bounds.root();
    let increase = bounds
        .updates()
        .iter()
        .find(|update| update.count().get() == 3)
        .unwrap()
        .field();
    let decrease = bounds
        .updates()
        .iter()
        .find(|update| update.count().get() == 1)
        .unwrap()
        .field();
    disk_log::start();
    bounds.publish_counts(BTreeMap::new()).unwrap();
    let recording = disk_log::finish();
    let publication = recording
        .events
        .iter()
        .position(|event| matches!(event, IoEvent::Write { offset, .. } if *offset == base))
        .unwrap();
    let first = recording.events.iter().position(|event| matches!(event, IoEvent::Write { offset, .. } if *offset == increase.range().start().get())).unwrap();
    let last = recording.events.iter().position(|event| matches!(event, IoEvent::Write { offset, .. } if *offset == decrease.range().start().get())).unwrap();
    assert!(first < publication && publication < last);
    assert_eq!(engine.publication, PublicationState::Published);
    assert_eq!(
        engine
            .addresses()
            .resolve(engine.superblock().root_group_address)
            .unwrap(),
        root
    );
    assert_eq!(
        engine
            .inplace_undo
            .iter()
            .map(|(at, bytes)| (*at, bytes.as_ptr()))
            .collect::<Vec<_>>(),
        prior
    );
    assert_chunk(&engine, objects[0] + base, &form.chunk(1, c"new"));
    assert_chunk(&engine, objects[1] + base, &form.chunk(3, c"new"));
    engine.context.allocations.before_reference().unwrap();
    if matches!(policy, SyncPolicy::Always) {
        assert!(
            recording.events[first..publication]
                .iter()
                .any(|event| matches!(
                    event,
                    IoEvent::Synchronize {
                        outcome: IoOutcome::Succeeded,
                        ..
                    }
                ))
        );
    }
}

#[rstest]
#[case::increase_before(Fault::Increase(0))]
#[case::increase_partial(Fault::Increase(2))]
#[case::increase_complete(Fault::Increase(usize::MAX))]
#[case::increase_checksum(Fault::IncreaseChecksum)]
#[case::increase_barrier_before(Fault::IncreaseBarrier(FailurePoint::Before))]
#[case::increase_barrier_after(Fault::IncreaseBarrier(FailurePoint::After))]
#[case::publication_before(Fault::Publication(0))]
#[case::publication_partial(Fault::Publication(1))]
#[case::publication_complete(Fault::Publication(usize::MAX))]
#[case::decrease_before(Fault::Decrease(0))]
#[case::decrease_partial(Fault::Decrease(2))]
#[case::decrease_complete(Fault::Decrease(usize::MAX))]
#[case::decrease_checksum(Fault::DecreaseChecksum)]
#[case::decrease_restore(Fault::DecreaseRestore)]
fn ordered_count_failures_preserve_the_applicable_counts_and_ownership(#[case] fault: Fault) {
    let control = test_util::temp::temp_path("ordered-count-control.h5");
    let path = test_util::temp::temp_path("ordered-count-fault.h5");
    let objects = fixture(&control, Form::Version2, 0);
    std::fs::copy(&control, &path).unwrap();
    let recording = {
        let mut engine = edit_test_util::open_engine(&control);
        engine.set_sync_policy(SyncPolicy::Always);
        engine
            .set_write_buffering(WriteBuffering::Unbuffered)
            .unwrap();
        let bounds = bounds(&mut engine, objects, 0);
        disk_log::start();
        bounds.publish_counts(BTreeMap::new()).unwrap();
        disk_log::finish()
    };
    let mut engine = edit_test_util::open_engine(&path);
    engine.set_sync_policy(SyncPolicy::Always);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let bounds = bounds(&mut engine, objects, 0);
    let increase = bounds
        .updates()
        .iter()
        .find(|update| update.count().get() == 3)
        .unwrap()
        .field();
    let decrease = bounds
        .updates()
        .iter()
        .find(|update| update.count().get() == 1)
        .unwrap()
        .field();
    let root = bounds.root();
    let free = bounds
        .phase
        .engine
        .space
        .reusable_sections()
        .collect::<Vec<_>>();
    let indices = [
        increase.range().start().get(),
        increase.chunk().end().get() - u64::try_from(test_util::checksum::CHECKSUM).unwrap(),
        0,
        decrease.range().start().get(),
        decrease.chunk().end().get() - u64::try_from(test_util::checksum::CHECKSUM).unwrap(),
    ]
    .map(|address| write_index(&recording, address));
    disk_log::start();
    fault.inject(indices, &recording);
    let error = bounds.publish_counts(BTreeMap::new()).unwrap_err();
    let failed = disk_log::finish();
    disk_log::clear_failures();
    let CountPublicationUpdateError::Operation(error) = error else {
        panic!("expected an operation error: {error:?}")
    };
    if matches!(fault, Fault::DecreaseRestore) {
        let Error::CommitPartiallyApplied { refusal, restore } = error else {
            panic!("expected restoration failure: {error:?}")
        };
        assert_io(*refusal, "injected disk synchronization failure");
        assert_io(*restore, "injected disk write failure");
    } else {
        assert_io(error, fault.message());
    }
    let published = !matches!(
        fault,
        Fault::Increase(_) | Fault::IncreaseChecksum | Fault::IncreaseBarrier(_)
    );
    assert_eq!(engine.publication.attempted(), published);
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), free);
    assert_chunk(
        &engine,
        objects[1],
        &Form::Version2.chunk(if published { 3 } else { 2 }, c"old"),
    );
    if !matches!(fault, Fault::DecreaseRestore) {
        assert_chunk(&engine, objects[0], &Form::Version2.chunk(2, c"old"));
    }
    assert!(!failed.events.is_empty());
    if published {
        assert!(matches!(
            engine.context.allocations.before_reference(),
            Err(Error::AllocationPublicationUncertain)
        ));
        if matches!(
            fault,
            Fault::Decrease(_) | Fault::DecreaseChecksum | Fault::DecreaseRestore
        ) {
            assert_eq!(
                engine
                    .addresses()
                    .resolve(engine.superblock().root_group_address)
                    .unwrap(),
                root
            );
        }
        disk_log::start();
        assert!(matches!(
            engine.force_sync(),
            Err(Error::AllocationPublicationUncertain)
        ));
        assert!(matches!(
            engine.commit(),
            Err(Error::AllocationPublicationUncertain)
        ));
        drop(engine);
        assert!(disk_log::finish().operations.is_empty());
    } else {
        engine.context.allocations.before_reference().unwrap();
        assert!(engine.inplace_undo.is_empty());
    }
}

#[rstest]
#[case::increase(Change::Increase)]
#[case::decrease(Change::Decrease)]
#[case::ownership(Change::Ownership)]
fn changed_count_evidence_preserves_error_details_and_publication_ownership(
    #[case] change: Change,
) {
    let path = test_util::temp::temp_path("ordered-count-evidence.h5");
    let objects = fixture(&path, Form::Version2, 0);
    let mut engine = edit_test_util::open_engine(&path);
    let bounds = bounds(&mut engine, objects, 0);
    let index = usize::from(matches!(change, Change::Increase));
    let field = bounds
        .updates()
        .iter()
        .find(|update| update.object().header_address().get() == objects[index])
        .unwrap()
        .field()
        .range();
    let allocation = bounds
        .updates()
        .iter()
        .find(|update| update.object().header_address().get() == objects[index])
        .unwrap()
        .allocation()
        .range();
    if matches!(change, Change::Ownership) {
        bounds.phase.engine.space = SessionSpace::from_persisted(
            Some(FileSpaceStrategy::FsmAggr),
            1,
            FileSpacePageSize::DEFAULT,
            [(
                None,
                Extent::new(StoredAddress::new(allocation.start().get()), 1).unwrap(),
            )],
        )
        .unwrap();
    } else {
        bounds
            .phase
            .engine
            .write_inplace_journaled(objects[index], &Form::Version2.chunk(4, c"old"))
            .unwrap();
    }
    let error = bounds.publish_counts(BTreeMap::new()).unwrap_err();
    match (change, error) {
        (
            Change::Ownership,
            CountPublicationUpdateError::Completion(CountCompletionError::Protection(
                CountPublicationError::Protection(CountProtectionError::Reusable {
                    object,
                    allocation: actual,
                    reusable,
                }),
            )),
        ) => {
            assert_eq!(object.header_address().get(), objects[0]);
            assert_eq!(actual, allocation);
            assert_eq!(reusable.start(), allocation.start());
            assert_eq!(reusable.length(), 1);
        }
        (
            Change::Increase | Change::Decrease,
            CountPublicationUpdateError::Preparation(CountPreparationError::Update {
                object,
                error,
            }),
        ) => {
            assert_eq!(object.header_address().get(), objects[index]);
            assert_eq!(
                error,
                ObjectReferenceCountUpdateError::Value {
                    field: field.start(),
                    expected: 2,
                    actual: 4
                }
            );
        }
        (_, error) => panic!("unexpected evidence error: {error:?}"),
    }
    assert_eq!(
        engine.publication.attempted(),
        !matches!(change, Change::Increase)
    );
    if matches!(change, Change::Increase) {
        engine.context.allocations.before_reference().unwrap();
    } else {
        assert!(matches!(
            engine.context.allocations.before_reference(),
            Err(Error::AllocationPublicationUncertain)
        ));
    }
}

#[derive(Clone, Copy)]
enum Change {
    Decrease,
    Increase,
    Ownership,
}

fn fixture(path: &Path, form: Form, base: u64) -> [u64; 2] {
    let mut fixture = Fixture::new();
    let objects =
        [0, 1].map(|_| u64::try_from(fixture.image.append(&form.chunk(2, c"old"))).unwrap());
    for (name, object) in [
        ("a", objects[0]),
        ("b", objects[0]),
        ("c", objects[1]),
        ("d", objects[1]),
    ] {
        fixture.link(name, object);
    }
    protection_tests::write_fixture(fixture, path, UserblockSize::try_from(base).unwrap());
    objects
}

fn bounds(engine: &mut WriteEngine, objects: [u64; 2], base: u64) -> CountPublicationBounds<'_> {
    let (mut counts, _) = protection_tests::counts(engine, &["a", "c"]);
    counts
        .updates
        .iter_mut()
        .find(|update| update.object.header_address().get() == objects[1] + base)
        .unwrap()
        .count = NonZeroU32::new(3).unwrap();
    let protected = counts.protect(engine, []).unwrap();
    let root = engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap();
    let region = engine.inspect_group(root.get()).unwrap().region;
    let mut region =
        group::remove_links_from_region(&region, [&LinkNameBuf::new("a").unwrap()]).unwrap();
    region.push(
        MessageType::LINK,
        &global_heap_fixture::hard_link("e", objects[1]).data,
    );
    let encoded = object_header::build_v2_object_header(&region).unwrap();
    let root = PhysicalAddress::new(engine.place_object_header(&encoded).unwrap());
    let end = PhysicalAddress::new(engine.image.len());
    let pending = RootRetirement::new(root, vec![]).into_pending_frees(vec![], engine.image.len());
    protected
        .before_publication(engine, pending)
        .unwrap()
        .at_end(end)
        .unwrap_or_else(|failure| panic!("{:?}", failure.error))
}

fn assert_chunk(engine: &WriteEngine, address: u64, expected: &[u8]) {
    assert_eq!(
        engine
            .image()
            .read_exact_at(address, expected.len())
            .unwrap(),
        expected
    );
}

fn write_index(recording: &Recording, address: u64) -> usize {
    recording
        .operations
        .iter()
        .filter(|operation| matches!(operation, DiskOp::Write { .. }))
        .position(
            |operation| matches!(operation, DiskOp::Write { offset, .. } if *offset == address),
        )
        .unwrap()
}

fn assert_io(error: Error, message: &str) {
    let Error::Io(error) = error else {
        panic!("expected I/O error: {error:?}")
    };
    assert_eq!(error.kind(), std::io::ErrorKind::Other);
    assert_eq!(error.to_string(), message);
}

#[derive(Clone, Copy)]
enum Fault {
    Decrease(usize),
    DecreaseChecksum,
    DecreaseRestore,
    Increase(usize),
    IncreaseBarrier(FailurePoint),
    IncreaseChecksum,
    Publication(usize),
}

impl Fault {
    fn message(self) -> &'static str {
        match self {
            Self::IncreaseBarrier(_) | Self::DecreaseRestore => {
                "injected disk synchronization failure"
            }
            _ => "injected disk write failure",
        }
    }

    fn inject(self, indices: [usize; 5], recording: &Recording) {
        match self {
            Self::Increase(prefix) => disk_log::fail_write_after(indices[0], prefix),
            Self::IncreaseChecksum => disk_log::fail_write_after(indices[1], 2),
            Self::IncreaseBarrier(point) => disk_log::fail_sync_after(0, point),
            Self::Publication(prefix) => disk_log::fail_write_after(indices[2], prefix),
            Self::Decrease(prefix) => disk_log::fail_write_after(indices[3], prefix),
            Self::DecreaseChecksum => disk_log::fail_write_after(indices[4], 2),
            Self::DecreaseRestore => {
                let DiskOp::Write { offset, .. } = recording
                    .operations
                    .iter()
                    .filter(|operation| matches!(operation, DiskOp::Write { .. }))
                    .nth(indices[4])
                    .unwrap()
                else {
                    panic!("expected the recorded decrease checksum write")
                };
                let last = recording.events.iter().position(|event| matches!(event, IoEvent::Write { offset: actual, .. } if actual == offset)).unwrap();
                let sync = recording.events[..last]
                    .iter()
                    .filter(|event| matches!(event, IoEvent::Synchronize { .. }))
                    .count();
                disk_log::fail_sync_after(sync, FailurePoint::Before);
                disk_log::fail_write_after(indices[4] + 1, 0);
            }
        }
    }
}
