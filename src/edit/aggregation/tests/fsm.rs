use core::num::NonZeroU64;

use std::path::Path;

use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::SessionSpace;
use rstest::rstest;

use crate::Error;
use crate::File;
use crate::FileBuilder;
use crate::FileLocking;
use crate::FileSpacePageSize;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::FormatError;
use crate::MaxExtent;
use crate::MemoryStrategy;
use crate::PersistedFreeSpacePolicy;
use crate::PhysicalAddress;
use crate::SyncPolicy;
use crate::UserblockSize;
use crate::address::AddressContext;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::address::{StoredAddress, StoredAddressExt};
use crate::chunk_index_inplace::Store;
use crate::edit::AppendBuilder;
use crate::edit::AppendTarget;
use crate::object_path::ObjectPathBuf;
use crate::source::MetadataCacheConfig;

use crate::edit::aggregation::AggregateSpace;
use crate::edit::aggregation::AllocationClass;
use crate::edit::aggregation::ClassicAggregators;
use crate::edit::aggregation::WriteEngine;

#[rstest]
#[case::staged_mirror(false, MemoryStrategy::Mirrored, 0)]
#[case::staged_bounded(false, MemoryStrategy::Bounded, 0)]
#[case::staged_userblock(false, MemoryStrategy::Mirrored, 512)]
#[case::immediate_mirror(true, MemoryStrategy::Mirrored, 0)]
#[case::immediate_bounded(true, MemoryStrategy::Bounded, 0)]
fn reusable_sections_precede_retained_aggregate_windows(
    #[case] immediate: bool,
    #[case] backing: MemoryStrategy,
    #[case] userblock: u64,
    #[values(false, true)] explicit: bool,
    #[values(AllocationClass::Metadata, AllocationClass::Raw)] class: AllocationClass,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("reuse.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
    if explicit {
        builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::FsmAggr));
    }
    builder.create_dataset("scratch").with_u8_data(&[7; 4096]);
    builder.create_dataset("keep").with_u8_data(&[11; 4096]);
    builder.write(&path).unwrap();
    let mut engine = open_engine(&path, backing);
    engine.delete(&ObjectPathBuf::parse("scratch")).unwrap();
    engine.commit().unwrap();
    let expected = engine.space.allocate(64, class).unwrap();
    engine.space.return_allocation(expected, class);
    let state = engine.aggregation.unwrap();
    let physical = engine.image.len();
    let address = if immediate {
        let mut store = engine.store();
        let allocation = store.allocate(&[19; 64], class).unwrap();
        let address = allocation.range().start().get();
        store
            .publish_allocations()
            .unwrap()
            .confirm(allocation)
            .unwrap();
        address
    } else {
        let reservation = engine.reserve(64, class).unwrap();
        engine.place(reservation, &[19; 64]).unwrap()
    };
    assert_eq!(
        address,
        engine.addresses().absolute(expected.start()).unwrap()
    );
    assert_eq!(engine.aggregation, Some(state));
    assert_eq!(engine.image.len(), physical);
    assert_eq!(engine.image.read_exact_at(address, 64).unwrap(), [19; 64]);
    engine.finalize_persist().unwrap();
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), ["keep"]);
    assert_eq!(file.dataset("keep").unwrap().read_u8().unwrap(), [11; 4096]);
    assert_eq!(file.persisted_free_sections().unwrap(), None);
}

#[rstest]
#[case::below_threshold(49, false)]
#[case::at_threshold(48, true)]
#[case::above_threshold(47, true)]
fn abandoned_spare_uses_the_session_threshold(
    #[case] threshold: u64,
    #[case] tracked: bool,
    #[values(AllocationClass::Metadata, AllocationClass::Raw)] class: AllocationClass,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("threshold.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_threshold(threshold),
    );
    builder.write(&path).unwrap();
    let mut engine = open_engine(&path, MemoryStrategy::Mirrored);
    let start = engine.image.len();
    let other = if class == AllocationClass::Metadata {
        AllocationClass::Raw
    } else {
        AllocationClass::Metadata
    };
    for (class, bytes) in [
        (class, vec![19; 2000]),
        (other, vec![23; 64]),
        (class, vec![29; 64]),
    ] {
        let reservation = engine.reserve(bytes.len() as u64, class).unwrap();
        engine.place(reservation, &bytes).unwrap();
    }
    let expected = Extent::new(StoredAddress::new(start + 2000), 48).unwrap();
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        if tracked { vec![expected] } else { vec![] }
    );
    let policy = engine.aggregation.unwrap();
    let reservation = engine.reserve(48, class).unwrap();
    if tracked {
        assert_eq!(
            reservation.address(),
            engine.addresses().absolute(expected.start()).unwrap()
        );
        assert_eq!(engine.aggregation, Some(policy));
    } else {
        assert_ne!(
            reservation.address(),
            engine.addresses().absolute(expected.start()).unwrap()
        );
        assert_ne!(engine.aggregation, Some(policy));
    }
    engine.place(reservation, &[31; 48]).unwrap();
}

#[test]
fn subthreshold_spare_merges_into_an_adjacent_tracked_section() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("merge.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_threshold(100),
    );
    builder.write(&path).unwrap();
    let mut engine = open_engine(&path, MemoryStrategy::Mirrored);
    let reservation = engine.reserve(2000, AllocationClass::Metadata).unwrap();
    let start = engine.place(reservation, &[19; 2000]).unwrap();
    let reservation = engine.reserve(64, AllocationClass::Raw).unwrap();
    engine.place(reservation, &[23; 64]).unwrap();
    engine.space.return_allocation(
        Extent::new(StoredAddress::new(start + 1900), 100).unwrap(),
        AllocationClass::Metadata,
    );
    let reservation = engine.reserve(149, AllocationClass::Metadata).unwrap();
    engine.place(reservation, &[29; 149]).unwrap();
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        [Extent::new(StoredAddress::new(start + 1900), 148).unwrap()]
    );
    let state = engine.aggregation;
    let reservation = engine.reserve(148, AllocationClass::Raw).unwrap();
    assert_eq!(reservation.address(), start + 1900);
    assert_eq!(engine.aggregation, state);
    engine.place(reservation, &[31; 148]).unwrap();
}

#[rstest]
fn fsm_aggregates_remain_disabled_for_swmr(#[values(false, true)] explicit: bool) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("swmr.h5");
    let mut builder = FileBuilder::new();
    if explicit {
        builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::FsmAggr));
    }
    builder.write(&path).unwrap();
    let engine = WriteEngine::open_swmr_writer(&path, SyncPolicy::Always).unwrap();
    assert!(engine.aggregation.is_none());
    assert!(engine.swmr_mode);
    assert!(!engine.immediate_reuse_allowed());
}

#[test]
fn accounting_coalesces_spare_with_adjacent_reusable_bytes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("accounting.h5");
    FileBuilder::new().write(&path).unwrap();
    let mut engine = open_engine(&path, MemoryStrategy::Mirrored);
    let reservation = engine.reserve(64, AllocationClass::Raw).unwrap();
    let address = engine.place(reservation, &[19; 64]).unwrap();
    let extent = Extent::new(StoredAddress::new(address), 64).unwrap();
    engine.space.return_allocation(extent, AllocationClass::Raw);
    test_util_hdf5::assert_free_sections!(
        &engine.reusable_free_sections().unwrap(),
        &[(address, 2048)],
        PhysicalAddress
    );
    assert_eq!(
        engine.reusable_space_stats().unwrap().reusable_free_bytes(),
        2048
    );
    engine.finalize_persist().unwrap();
    assert_eq!(engine.image.len(), address);
    assert_eq!(
        engine.aggregation.unwrap().policy,
        ClassicAggregators::new()
    );
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), []);
}

#[test]
fn failed_reservation_preserves_abandoned_spare_and_policy() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("overflow.h5");
    FileBuilder::new().write(&path).unwrap();
    let mut engine = open_engine(&path, MemoryStrategy::Mirrored);
    let reservation = engine.reserve(2000, AllocationClass::Metadata).unwrap();
    engine.place(reservation, &[19; 2000]).unwrap();
    let reservation = engine.reserve(64, AllocationClass::Raw).unwrap();
    engine.place(reservation, &[23; 64]).unwrap();
    let state = engine.aggregation;
    let space = engine.reusable_free_sections().unwrap();
    let error = engine
        .reserve(u64::MAX, AllocationClass::Metadata)
        .unwrap_err();
    let Error::Format(FormatError::OffsetOverflow { offset, length }) = error else {
        panic!("expected overflowing reservation, got {error:?}");
    };
    assert_eq!(offset, engine.image.len());
    assert_eq!(length, u64::MAX);
    assert_eq!(engine.aggregation, state);
    assert_eq!(engine.reusable_free_sections().unwrap(), space);
}

#[test]
fn rollback_restores_spare_returned_during_refill() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("rollback.h5");
    FileBuilder::new().write(&path).unwrap();
    let mut engine = open_engine(&path, MemoryStrategy::Mirrored);
    let reservation = engine.reserve(2000, AllocationClass::Metadata).unwrap();
    engine.place(reservation, &[19; 2000]).unwrap();
    let reservation = engine.reserve(64, AllocationClass::Raw).unwrap();
    engine.place(reservation, &[23; 64]).unwrap();
    let snapshot = engine.snapshot_free();
    let state = engine.aggregation;
    let free = engine.reusable_free_sections().unwrap();
    let placement = engine.reserve(64, AllocationClass::Metadata).unwrap();
    assert_eq!(engine.space.reusable_sections().count(), 1);
    engine.restore_free(snapshot);
    assert_eq!(engine.aggregation, state);
    assert_eq!(engine.reusable_free_sections().unwrap(), free);
    let retry = engine.reserve(64, AllocationClass::Metadata).unwrap();
    assert_eq!(retry.address(), placement.address());
}

#[rstest]
fn teardown_preserves_retained_interior_spare(
    #[values((MemoryStrategy::Mirrored, 0), (MemoryStrategy::Bounded, 0), (MemoryStrategy::Mirrored, 512))]
    (backing, userblock): (MemoryStrategy, u64),
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("teardown.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(userblock).unwrap());
    builder.write(&path).unwrap();
    let mut engine = open_engine(&path, backing);
    let first = engine.reserve(64, AllocationClass::Metadata).unwrap();
    engine.place(first, &[19; 64]).unwrap();
    let second = engine.reserve(64, AllocationClass::Raw).unwrap();
    let end = second.address() + 64;
    engine.place(second, &[23; 64]).unwrap();
    let interior = engine
        .aggregation
        .unwrap()
        .policy
        .unused(AllocationClass::Metadata)
        .unwrap();
    engine.finalize_persist().unwrap();
    assert_eq!(engine.image.len(), end);
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        [interior]
    );
    let reservation = engine.reserve(64, AllocationClass::Metadata).unwrap();
    assert_eq!(
        reservation.address(),
        engine.addresses().absolute(interior.start()).unwrap()
    );
    engine.place(reservation, &[29; 64]).unwrap();
}

#[test]
fn stored_aggregate_positions_remain_u64() {
    let mut aggregate = AggregateSpace::new(StoredAddress::new(4_294_967_296));
    let mut space = SessionSpace::from_strategy(
        Some(FileSpaceStrategy::FsmAggr),
        1,
        FileSpacePageSize::DEFAULT,
    );
    let placement = aggregate
        .reserve(
            AddressContext::from_effective_base(BaseAddress::ZERO),
            &mut space,
            NonZeroU64::new(64).unwrap(),
            AllocationClass::Raw,
        )
        .unwrap();
    assert_eq!(placement.address(), 4_294_967_296);
    assert_eq!(aggregate.eoa(), StoredAddress::new(4_294_969_344));
}

#[test]
fn large_chunk_appends_retain_only_metadata_spare() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("large_chunks.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("values")
        .with_f64_data(&[0.0; 512])
        .with_shape(&[512])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[512]);
    builder.write(&path).unwrap();
    let mut engine = open_engine(&path, MemoryStrategy::Mirrored);
    let mut batch = AppendBuilder::new();
    batch.append_f64(&[1.0; 512]);
    for _ in 0..16 {
        engine
            .append_inplace_gathered(
                AppendTarget::Path(&ObjectPathBuf::parse("values")),
                &batch,
                4,
            )
            .unwrap();
    }
    let state = engine.aggregation.unwrap();
    assert_eq!(state.policy.unused(AllocationClass::Raw), None);
    assert_eq!(
        state
            .policy
            .unused(AllocationClass::Metadata)
            .unwrap()
            .len(),
        1898
    );
}

fn open_engine(path: &Path, backing: MemoryStrategy) -> WriteEngine {
    WriteEngine::open_rw_with_strategy(
        path,
        MetadataCacheConfig::default(),
        FileLocking::Enabled,
        backing,
        PersistedFreeSpacePolicy::Tolerant,
    )
    .unwrap()
}
