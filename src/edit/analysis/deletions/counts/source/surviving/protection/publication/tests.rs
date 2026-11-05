use std::sync::Arc;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::sync::atomic::Ordering;

use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::__private::NonZeroByteLength;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::PageType;
use hdf5_pure_space::__private::RetainedKind;
use hdf5_pure_space::__private::SessionSpace;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util_hdf5::global_heap_fixture::Fixture;

use crate::UserblockSize;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::edit::aggregation::AggregateSpace;
use crate::edit::allocation::AllocationPublication;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::edit::analysis::DISCOVERY_WORK;
use crate::edit::retirement::RootRetirement;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::file_space_info::FileSpacePageSize;
use crate::file_space_info::FileSpaceStrategy;

use super::super::tests as protection_tests;
use super::super::tests::CountStorage;
use super::super::tests::RetirementFixture;
use super::CountEndFailure;
use super::CountProtectionError;
use super::CountPublicationError;

mod completion;
mod current;
mod prepare;
mod write;

#[rstest]
#[case::prefix(Relation::Prefix, FreeClass::Page(PageType::Meta))]
#[case::suffix(Relation::Suffix, FreeClass::Page(PageType::Raw))]
#[case::whole(Relation::Whole, FreeClass::Dead)]
#[case::enclosing(Relation::Enclosing, FreeClass::Dead)]
#[case::before(Relation::Before, FreeClass::Dead)]
#[case::after(Relation::After, FreeClass::Dead)]
fn publication_excludes_complete_retirement_allocations(
    #[case] relation: Relation,
    #[case] class: FreeClass,
    #[values(CountStorage::Header, CountStorage::Continuation)] storage: CountStorage,
    #[values(0, 512)] base: u64,
) {
    let path = test_util::temp::temp_path("count-retirement-bounds.h5");
    let userblock = UserblockSize::try_from(base).unwrap();
    let RetirementFixture {
        objects: (address, _),
        retired: original_retirements,
    } = protection_tests::fixture_with_retirements(&path, userblock, storage);
    let mut engine = edit_test_util::open_engine(&path);
    let (counts, origins) = protection_tests::counts(&engine, &["a"]);
    let protected = counts.protect(&engine, []).unwrap();
    let update = &protected.updates()[0];
    let allocation = update.allocation().range();
    let object = update.object();
    assert_eq!(allocation, update.field().chunk());
    assert_eq!(object.header_address().get(), address + base);
    let expected = match storage {
        CountStorage::Header => FileRange::new(PhysicalAddress::new(address + base), 20),
        CountStorage::Continuation => FileRange::new(PhysicalAddress::new(address + base - 17), 17),
    }
    .unwrap();
    assert_eq!(allocation, expected);
    assert_eq!(
        allocation.start() == object.header_address(),
        matches!(storage, CountStorage::Header)
    );
    let (start, length) = match relation {
        Relation::Prefix => (allocation.start().get(), 1),
        Relation::Suffix => (allocation.end().get() - 1, 1),
        Relation::Whole => (allocation.start().get(), allocation.length()),
        Relation::Enclosing => (allocation.start().get() - 1, allocation.length() + 2),
        Relation::Before => (allocation.start().get() - 1, 1),
        Relation::After => (allocation.end().get(), 1),
    };
    let retired = FileRange::new(PhysicalAddress::new(start), length).unwrap();
    let root = PhysicalAddress::new(engine.image.len() + 16);
    let preceding = original_retirements[0];
    assert!(preceding.0.end() < retired.start());
    let expected_retirement = [preceding, (retired, class)];
    let pending = RootRetirement::new(root, vec![])
        .into_pending_frees(expected_retirement.to_vec(), engine.image.len() + 16);
    assert_eq!(pending.batch.entries(), expected_retirement);
    let before = engine.disk_bytes().unwrap();
    let addresses = engine.addresses();
    let result = protected.before_publication(&mut engine, pending);
    if matches!(relation, Relation::Before | Relation::After) {
        let phase = result.unwrap();
        assert!(Arc::ptr_eq(
            &phase.counts.source.as_ref().unwrap().identity,
            &origins[0].identity
        ));
        let bounds = phase
            .at_end(allocation.end())
            .unwrap_or_else(|failure| panic!("{:?}", failure.error));
        assert_eq!(
            bounds.original_root().unwrap().header_address().get(),
            base + Fixture::ROOT
        );
        assert_eq!(bounds.root(), root);
        assert_eq!(bounds.end(), allocation.end());
        assert_eq!(bounds.updates().len(), 1);
        assert_eq!(bounds.updates()[0].allocation().range(), allocation);
        assert_eq!(bounds.updates()[0].count().get(), 1);
        assert_eq!(bounds.phase.engine.disk_bytes().unwrap(), before);
        let returned = bounds
            .into_phase()
            .retirement
            .batch
            .into_stored(addresses)
            .unwrap();
        assert_eq!(
            returned.entries(),
            expected_retirement.map(|(range, class)| (
                Extent::new(
                    StoredAddress::new(range.start().get() - base),
                    ByteLength::new(range.length())
                )
                .unwrap(),
                class,
            ))
        );
    } else {
        let Err(CountPublicationError::Retirement {
            object: actual,
            allocation: owned,
            retired: actual_retired,
            class: actual_class,
        }) = result
        else {
            panic!("expected a retirement overlap")
        };
        assert_eq!(actual, object);
        assert_eq!(owned, allocation);
        assert_eq!(actual_retired, retired);
        assert_eq!(actual_class, class);
    }
    assert_eq!(engine.disk_bytes().unwrap(), before);
}

#[rstest]
#[case::inside(-1)]
#[case::exact(0)]
#[case::after(1)]
fn publication_checks_eoa_against_the_complete_last_allocation(
    #[case] offset: i64,
    #[values(CountStorage::Header, CountStorage::Continuation)] storage: CountStorage,
    #[values(0, 512)] base: u64,
) {
    let path = test_util::temp::temp_path("count-eoa-bounds.h5");
    let fixture = protection_tests::fixture_with_retirements(
        &path,
        UserblockSize::try_from(base).unwrap(),
        storage,
    );
    let mut engine = edit_test_util::open_engine(&path);
    let (counts, _) = protection_tests::counts(&engine, &["a", "c"]);
    let protected = counts.protect(&engine, []).unwrap();
    let original_root = protected.root();
    let expected = protected
        .updates()
        .iter()
        .map(|update| {
            (
                update.object(),
                update.field(),
                update.count(),
                update.allocation().range(),
            )
        })
        .collect::<Vec<_>>();
    let last = protected.updates().last().unwrap();
    let object = last.object();
    let allocation = last.allocation().range();
    let end = PhysicalAddress::new(allocation.end().get().checked_add_signed(offset).unwrap());
    assert_eq!(fixture.retired[0].0.end(), fixture.retired[1].0.start());
    assert_eq!(fixture.retired[0].0.length(), 16);
    assert_eq!(fixture.retired[1].0.length(), 24);
    assert_eq!(
        fixture.retired[1].0.end(),
        protected.updates()[0].allocation().range().start()
    );
    let addresses = engine.addresses();
    let pending = RootRetirement::new(PhysicalAddress::new(engine.image.len()), vec![])
        .into_pending_frees(fixture.retired.to_vec(), engine.image.len());
    let replacement_root = pending.root;
    let phase = protected.before_publication(&mut engine, pending).unwrap();
    let phase = match phase.at_end(end) {
        Err(failure) if offset < 0 => {
            let CountEndFailure { phase, error } = *failure;
            let CountPublicationError::Truncation {
                object: actual,
                allocation: owned,
                end: actual_end,
            } = error
            else {
                panic!("expected truncated count allocation")
            };
            assert_eq!(actual, object);
            assert_eq!(owned, allocation);
            assert_eq!(actual_end, end);
            phase
        }
        Ok(bounds) if offset >= 0 => {
            assert_eq!(bounds.end(), end);
            assert_eq!(bounds.updates().len(), 2);
            bounds.into_phase()
        }
        Err(failure) => panic!("unexpected count EOA failure: {:?}", failure.error),
        Ok(_) => panic!("expected truncated count allocation"),
    };
    let bounds = phase
        .at_end(allocation.end())
        .unwrap_or_else(|failure| panic!("{:?}", failure.error));
    assert_eq!(bounds.end(), allocation.end());
    assert_eq!(bounds.original_root(), original_root);
    assert_eq!(bounds.root(), replacement_root);
    assert_eq!(
        bounds
            .updates()
            .iter()
            .map(|update| (
                update.object(),
                update.field(),
                update.count(),
                update.allocation().range(),
            ))
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(bounds.phase.retirement.batch.entries(), fixture.retired);
    let returned = bounds
        .into_phase()
        .retirement
        .batch
        .into_stored(addresses)
        .unwrap();
    assert_eq!(
        returned.entries(),
        fixture.retired.map(|(range, class)| (
            Extent::new(
                StoredAddress::new(range.start().get() - base),
                ByteLength::new(range.length())
            )
            .unwrap(),
            class,
        ))
    );
}

#[rstest]
#[case::session(Mutation::Session)]
#[case::root(Mutation::Root)]
#[case::uncertain(Mutation::Uncertain)]
#[case::reusable(Mutation::Reusable)]
#[case::retained(Mutation::Retained)]
#[case::aggregate(Mutation::Aggregate)]
#[case::restored(Mutation::Restored)]
#[case::provisional(Mutation::Provisional)]
fn publication_rechecks_mutable_allocator_and_session_facts(#[case] mutation: Mutation) {
    let path = test_util::temp::temp_path("count-phase-mutation.h5");
    protection_tests::fixture_with_base(&path, UserblockSize::NONE);
    let mut engine = edit_test_util::open_engine(&path);
    let (counts, _) = protection_tests::counts(&engine, &["a"]);
    let protected = counts.protect(&engine, []).unwrap();
    let allocation = protected.updates()[0].allocation().range();
    let object = protected.updates()[0].object();
    let extent = Extent::new(
        engine.addresses().relative(allocation.start()).unwrap(),
        ByteLength::new(1),
    )
    .unwrap();
    let pending = RootRetirement::new(PhysicalAddress::new(engine.image.len()), vec![])
        .into_pending_frees(vec![], engine.image.len());
    let prior = engine.snapshot_free();
    match mutation {
        Mutation::Session => engine.context.allocations = AllocationPublication::new(),
        Mutation::Root => {
            let mut replacement = engine.superblock().clone();
            replacement.root_group_address = engine
                .addresses()
                .relative(object.header_address())
                .unwrap();
            engine
                .context
                .adopt(engine.context.propose(replacement).unwrap())
                .unwrap();
        }
        Mutation::Uncertain => engine.make_allocation_uncertain(),
        Mutation::Reusable | Mutation::Restored | Mutation::Retained => {
            engine.aggregation = None;
            engine.space = SessionSpace::from_persisted(
                Some(if matches!(mutation, Mutation::Retained) {
                    FileSpaceStrategy::Page
                } else {
                    FileSpaceStrategy::FsmAggr
                }),
                1,
                FileSpacePageSize::MIN,
                [(None, extent)],
            )
            .unwrap();
            if matches!(mutation, Mutation::Restored) {
                engine.restore_free(prior);
            }
        }
        Mutation::Aggregate => {
            let mut spare = AggregateSpace::new(StoredAddress::new(extent.start().get() - 1));
            spare
                .reserve_with(
                    engine.widths().offsets,
                    engine.addresses(),
                    NonZeroByteLength::new(1).unwrap(),
                    AllocationClass::Metadata,
                    |_, _| panic!("a fresh aggregate has no previous spare"),
                )
                .unwrap();
            engine.aggregation = Some(spare);
        }
        Mutation::Provisional => {
            engine
                .context
                .allocations
                .provision(FileRange::new(pending.root, 1).unwrap())
                .unwrap();
        }
    }
    let before = engine.disk_bytes().unwrap();
    match (mutation, protected.before_publication(&mut engine, pending)) {
        (Mutation::Restored | Mutation::Provisional, Ok(phase)) => {
            assert_eq!(
                phase
                    .at_end(allocation.end())
                    .unwrap_or_else(|failure| panic!("{:?}", failure.error))
                    .updates()
                    .len(),
                1
            );
        }
        (
            Mutation::Session,
            Err(CountPublicationError::Protection(CountProtectionError::Session)),
        ) => {}
        (
            Mutation::Root,
            Err(CountPublicationError::Protection(CountProtectionError::Root { expected, actual })),
        ) => {
            assert_eq!(expected.header_address().get(), Fixture::ROOT);
            assert_eq!(actual, object);
        }
        (
            Mutation::Uncertain,
            Err(CountPublicationError::Protection(CountProtectionError::Publication(
                Error::AllocationPublicationUncertain,
            ))),
        ) => {}
        (
            Mutation::Reusable,
            Err(CountPublicationError::Protection(CountProtectionError::Reusable {
                object: actual,
                allocation: owned,
                reusable,
            })),
        ) => {
            assert_eq!(actual, object);
            assert_eq!(owned, allocation);
            assert_eq!(reusable.start(), allocation.start());
            assert_eq!(reusable.length(), 1);
        }
        (
            Mutation::Retained,
            Err(CountPublicationError::Protection(CountProtectionError::Retained {
                object: actual,
                allocation: owned,
                retained,
                kind,
            })),
        ) => {
            assert_eq!(actual, object);
            assert_eq!(owned, allocation);
            assert_eq!(retained.start(), allocation.start());
            assert_eq!(retained.length(), 1);
            assert_eq!(kind, RetainedKind::Unclassified);
        }
        (
            Mutation::Aggregate,
            Err(CountPublicationError::Protection(CountProtectionError::Aggregate {
                object: actual,
                allocation: owned,
                spare,
            })),
        ) => {
            assert_eq!(actual, object);
            assert_eq!(owned, allocation);
            assert_eq!(spare.start(), allocation.start());
            assert_eq!(spare.length(), 2_047);
        }
        (_, Err(error)) => panic!("unexpected publication protection error: {error:?}"),
        (_, Ok(_)) => panic!("expected a publication protection error"),
    }
    assert_eq!(engine.disk_bytes().unwrap(), before);
}

#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn publication_bounds_reuse_one_phase_with_zero_allocations_and_source_reads() {
    let path = test_util::temp::temp_path("count-bounds-cost.h5");
    protection_tests::fixture_with_base(&path, UserblockSize::NONE);
    let (mut engine, calls) = protection_tests::engine_with_read_call_counter(&path);
    engine.space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::FsmAggr),
        1,
        FileSpacePageSize::MIN,
        (0..256).map(|at| {
            (
                None,
                Extent::new(StoredAddress::new(at * 2), ByteLength::new(1)).unwrap(),
            )
        }),
    )
    .unwrap();
    let (counts, origins) = protection_tests::counts(&engine, &["a", "c"]);
    let (protected, control) = allocation::measure("count-publication-control", || {
        counts.protect(&engine, [&origins[0]]).unwrap()
    });
    assert!(control.blocks > 0 && control.bytes > 0, "{control}");
    let end = protected.updates()[0].allocation().range().end();
    let pending = RootRetirement::new(PhysicalAddress::new(engine.image.len()), vec![])
        .into_pending_frees(vec![], engine.image.len());
    assert!(calls.load(Ordering::Relaxed) > 0);
    calls.store(0, Ordering::Relaxed);
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    let ((), measured) = allocation::measure("count-publication-bounds", || {
        let mut phase = protected.before_publication(&mut engine, pending).unwrap();
        for _ in 0..256 {
            let bounds = phase
                .at_end(end)
                .unwrap_or_else(|failure| panic!("{:?}", failure.error));
            assert_eq!(bounds.end(), end);
            phase = bounds.into_phase();
        }
    });
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    DISCOVERY_WORK.with(|work| assert_eq!(work.get(), (0, 0)));
    let pending = RootRetirement::new(PhysicalAddress::new(engine.image.len()), vec![])
        .into_pending_frees(vec![], engine.image.len());
    let (empty, _) = protection_tests::counts(&engine, &["a", "b"]);
    let empty = empty.protect(&engine, []).unwrap();
    calls.store(0, Ordering::Relaxed);
    engine.make_allocation_uncertain();
    let ((), measured) = allocation::measure("empty-count-publication", || {
        let phase = empty.before_publication(&mut engine, pending).unwrap();
        assert!(
            phase
                .at_end(PhysicalAddress::new(0))
                .unwrap_or_else(|failure| panic!("{:?}", failure.error))
                .updates()
                .is_empty()
        );
    });
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[derive(Clone, Copy)]
enum Relation {
    After,
    Before,
    Enclosing,
    Prefix,
    Suffix,
    Whole,
}

#[derive(Clone, Copy)]
enum Mutation {
    Aggregate,
    Provisional,
    Restored,
    Retained,
    Reusable,
    Root,
    Session,
    Uncertain,
}
