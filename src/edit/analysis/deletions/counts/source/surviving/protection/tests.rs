use core::cell::Cell;
use core::iter;
use core::num::NonZeroU64;
use std::path::Path;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::sync::Arc;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::sync::atomic::AtomicU64;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use std::sync::atomic::Ordering;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::AllocationClass;
use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::ManagerKind;
use hdf5_pure_space::__private::PageTransition;
use hdf5_pure_space::__private::PageType;
use hdf5_pure_space::__private::RetainedKind;
use hdf5_pure_space::__private::SessionSpace;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::object_header::Message;
use test_util::object_header::v2;
use test_util::superblock::v2::Superblock;
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::Fixture;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::FileLocking;
use crate::UserblockSize;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::edit::WriteEngine;
use crate::edit::aggregation::AggregateSpace;
use crate::edit::analysis::AnalysisPhase;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::edit::analysis::DISCOVERY_WORK;
use crate::edit::analysis::ReconstructionSource;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::error::FormatError;
use crate::file_space_info::FileSpacePageSize;
use crate::file_space_info::FileSpaceStrategy;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::image::CountingImage;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::image::HandleImage;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::source::MetadataCacheConfig;

use super::CountProtectionError;
use super::SurvivingCounts;

#[rstest]
#[case::retained(0, false)]
#[case::one_reconstructed(1, false)]
#[case::all_reconstructed(2, false)]
#[case::unsupported_reconstructed(2, true)]
fn protection_excludes_matching_reconstructions_and_orders_complete_allocations(
    #[case] rebuilt: usize,
    #[case] unsupported: bool,
) {
    let path = test_util::temp::temp_path("protected-counts.h5");
    let (a, c) = fixture(&path);
    let engine = edit_test_util::open_engine(&path);
    let (mut batch, origins) = counts(&engine, &["a", "c"]);
    if unsupported {
        batch.updates[0].allocation = None;
    }
    assert_eq!(
        batch
            .updates()
            .iter()
            .map(|update| update.object().header_address().get())
            .collect::<Vec<_>>(),
        [a, c]
    );
    let protected = batch
        .protect(&engine, origins.iter().take(rebuilt))
        .unwrap();
    let expected = if rebuilt == 2 {
        vec![]
    } else if rebuilt == 1 {
        vec![c]
    } else {
        vec![c, a]
    };
    assert_eq!(
        protected
            .updates()
            .iter()
            .map(|update| update.object().header_address().get())
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(
        protected.root(),
        (rebuilt != 2).then_some(ObjectId::new(PhysicalAddress::new(Fixture::ROOT)))
    );
    for update in protected.updates() {
        assert_eq!(update.count().get(), 1);
        assert_eq!(update.field().value(), 2);
        assert_eq!(update.allocation().range(), update.field().chunk());
        assert_eq!(update.allocation().range().length(), 20);
    }
}

#[rstest]
#[case::phase(Failure::Phase)]
#[case::repeated(Failure::Repeated)]
#[case::allocation(Failure::Allocation)]
#[case::root(Failure::Root)]
#[case::uncertain(Failure::Uncertain)]
fn protection_rejects_inapplicable_or_incomplete_facts_without_writes(#[case] failure: Failure) {
    let path = test_util::temp::temp_path("protected-count-failure.h5");
    let (a, _) = fixture(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let (mut batch, origins) = counts(&engine, &["a"]);
    let (_, foreign) = counts(&engine, &["a"]);
    let before = engine.disk_bytes().unwrap();
    let reconstructed = match failure {
        Failure::Phase => vec![&foreign[0]],
        Failure::Repeated => vec![&origins[0], &origins[0]],
        Failure::Allocation => {
            batch.updates[0].allocation = None;
            vec![]
        }
        Failure::Root => {
            let mut proposed = engine.superblock().clone();
            proposed.root_group_address = StoredAddress::new(a);
            engine
                .context
                .adopt(engine.context.propose(proposed).unwrap())
                .unwrap();
            vec![]
        }
        Failure::Uncertain => {
            engine.make_allocation_uncertain();
            vec![]
        }
    };
    let Err(error) = batch.protect(&engine, reconstructed) else {
        panic!("expected protection to fail")
    };
    let object = ObjectId::new(PhysicalAddress::new(a));
    match (failure, error) {
        (Failure::Phase, CountProtectionError::Phase { object: actual })
        | (Failure::Repeated, CountProtectionError::Repeated { object: actual })
        | (Failure::Allocation, CountProtectionError::Allocation { object: actual }) => {
            assert_eq!(actual, object)
        }
        (Failure::Root, CountProtectionError::Root { expected, actual }) => {
            assert_eq!(expected, ObjectId::new(PhysicalAddress::new(Fixture::ROOT)));
            assert_eq!(actual, object);
        }
        (
            Failure::Uncertain,
            CountProtectionError::Publication(Error::AllocationPublicationUncertain),
        ) => {}
        (_, other) => panic!("unexpected protection error: {other:?}"),
    }
    assert_eq!(engine.disk_bytes().unwrap(), before);
}

#[test]
fn protection_rejects_a_different_session_at_identical_addresses() {
    let path = test_util::temp::temp_path("protected-count-session.h5");
    let other_path = test_util::temp::temp_path("protected-count-other-session.h5");
    fixture(&path);
    std::fs::copy(&path, &other_path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let other = edit_test_util::open_engine(&other_path);
    let (batch, _) = counts(&engine, &["a"]);
    let Err(error) = batch.protect(&other, []) else {
        panic!("expected another session")
    };
    assert!(matches!(error, CountProtectionError::Session), "{error:?}");
}

#[rstest]
#[case::prefix(0, true)]
#[case::count(12, true)]
#[case::checksum(19, true)]
#[case::before(-1, false)]
#[case::after(20, false)]
fn protection_excludes_every_reusable_class_from_complete_allocations(
    #[case] offset: i64,
    #[case] conflict: bool,
    #[values(
        None,
        Some(ManagerKind::PagedSmallMetadata),
        Some(ManagerKind::PagedSmallRaw),
        Some(ManagerKind::PagedLargeGeneric)
    )]
    kind: Option<ManagerKind>,
) {
    let path = test_util::temp::temp_path("protected-count-reusable.h5");
    let (a, _) = fixture(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let (batch, _) = counts(&engine, &["a"]);
    let allocation = batch.updates()[0].header_allocation().unwrap().range();
    let generic = kind == Some(ManagerKind::PagedLargeGeneric);
    let (start, length) = if generic {
        let page = FileSpacePageSize::MIN.get();
        (a / page * page, page)
    } else {
        (a.checked_add_signed(offset).unwrap(), 1)
    };
    let reusable = FileRange::new(PhysicalAddress::new(start), length).unwrap();
    engine.aggregation = None;
    engine.space = SessionSpace::from_persisted(
        Some(if kind.is_some() {
            FileSpaceStrategy::Page
        } else {
            FileSpaceStrategy::FsmAggr
        }),
        1,
        FileSpacePageSize::MIN,
        [(
            kind,
            Extent::new(StoredAddress::new(start), length).unwrap(),
        )],
    )
    .unwrap();
    let result = batch.protect(&engine, []);
    if conflict || generic {
        let Err(CountProtectionError::Reusable {
            object,
            allocation: actual,
            reusable: free,
        }) = result
        else {
            panic!("expected complete-allocation reuse conflict")
        };
        assert_eq!(object, ObjectId::new(PhysicalAddress::new(a)));
        assert_eq!(actual, allocation);
        assert_eq!(free, reusable);
    } else {
        let protected = result.unwrap();
        assert_eq!(protected.updates().len(), 1);
        assert_eq!(protected.updates()[0].allocation().range(), allocation);
    }
}

#[rstest]
#[case::prefix(0, true)]
#[case::count(12, true)]
#[case::checksum(19, true)]
#[case::before(-1, false)]
#[case::after(20, false)]
fn protection_excludes_retained_claims_without_inferring_page_ownership(
    #[case] offset: i64,
    #[case] conflict: bool,
    #[values(
        RetainedKind::Dead,
        RetainedKind::Unclassified,
        RetainedKind::Padding(PageType::Meta),
        RetainedKind::Padding(PageType::Raw)
    )]
    kind: RetainedKind,
    #[values(0, 512)] userblock: u64,
) {
    let path = test_util::temp::temp_path("protected-count-retained.h5");
    let userblock = UserblockSize::try_from(userblock).unwrap();
    let (a, _) = fixture_with_base(&path, userblock);
    let mut engine = edit_test_util::open_engine(&path);
    let (batch, _) = counts(&engine, &["a"]);
    let allocation = batch.updates()[0].header_allocation().unwrap().range();
    let start = a.checked_add_signed(offset).unwrap();
    let extent = Extent::new(StoredAddress::new(start), 1).unwrap();
    engine.aggregation = None;
    engine.space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::Page),
        1,
        FileSpacePageSize::MIN,
        if kind == RetainedKind::Unclassified {
            Some((None, extent))
        } else {
            None
        },
    )
    .unwrap();
    match kind {
        RetainedKind::Dead => {
            engine.space.record_freed(
                StoredAddress::new(engine.image.len() - userblock.get()),
                extent,
                FreeClass::Dead,
            );
        }
        RetainedKind::Padding(page_type) => {
            engine.space.record_page_padding(PageTransition::Pad {
                extent,
                track_as: Some(page_type),
            });
        }
        RetainedKind::Unclassified => {}
    }
    assert_eq!(engine.space.reusable_bytes().unwrap(), 0);
    let before = engine.disk_bytes().unwrap();
    let result = batch.protect(&engine, []);
    if conflict {
        let Err(CountProtectionError::Retained {
            object,
            allocation: actual,
            retained,
            kind: actual_kind,
        }) = result
        else {
            panic!("expected a retained-allocation conflict")
        };
        assert_eq!(
            object,
            ObjectId::new(PhysicalAddress::new(a + userblock.get()))
        );
        assert_eq!(actual, allocation);
        assert_eq!(
            retained,
            FileRange::new(PhysicalAddress::new(start + userblock.get()), 1).unwrap()
        );
        assert_eq!(actual_kind, kind);
    } else {
        let protected = result.unwrap();
        assert_eq!(protected.updates().len(), 1);
        assert_eq!(protected.updates()[0].allocation().range(), allocation);
    }
    assert_eq!(engine.disk_bytes().unwrap(), before);
    let retained = engine.space.retained_extents().collect::<Vec<_>>();
    assert_eq!(retained.len(), 1);
    assert_eq!(retained[0].extent(), extent);
    assert_eq!(retained[0].kind(), kind);
}

#[rstest]
#[case::metadata(AllocationClass::Metadata)]
#[case::raw(AllocationClass::Raw)]
fn protection_excludes_aggregate_spare_without_combining_free_lists(
    #[case] class: AllocationClass,
    #[values(false, true)] conflict: bool,
) {
    let path = test_util::temp::temp_path("protected-count-aggregate.h5");
    let (a, _) = fixture(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let (batch, _) = counts(&engine, &["a"]);
    let allocation = batch.updates()[0].header_allocation().unwrap().range();
    let start = if conflict { a - 1 } else { a + 20 };
    let mut aggregate = AggregateSpace::new(StoredAddress::new(start));
    aggregate
        .reserve_with(
            engine.widths().offsets,
            engine.addresses(),
            NonZeroU64::new(1).unwrap(),
            class,
            |_, _| panic!("a fresh aggregate returns no previous spare"),
        )
        .unwrap();
    let spare = aggregate.unused().next().unwrap();
    assert_eq!(spare.start().get(), start + 1);
    engine.aggregation = Some(aggregate);
    engine.space = SessionSpace::Disabled;
    let before = engine.disk_bytes().unwrap();
    let result = batch.protect(&engine, []);
    if conflict {
        let Err(CountProtectionError::Aggregate {
            object,
            allocation: actual,
            spare: unused,
        }) = result
        else {
            panic!("expected an aggregate-spare conflict")
        };
        assert_eq!(object, ObjectId::new(PhysicalAddress::new(a)));
        assert_eq!(actual, allocation);
        assert_eq!(
            unused,
            FileRange::new(
                engine.addresses().resolve(spare.start()).unwrap(),
                spare.len()
            )
            .unwrap()
        );
    } else {
        assert_eq!(result.unwrap().updates().len(), 1);
    }
    assert_eq!(engine.aggregation, Some(aggregate));
    assert_eq!(engine.disk_bytes().unwrap(), before);
}

#[rstest]
#[case::empty(0)]
#[case::wide(128)]
fn unordered_retained_exclusion_visits_each_claim_once(#[case] count: u64) {
    let path = test_util::temp::temp_path("protected-count-retained-visits.h5");
    fixture(&path);
    let engine = edit_test_util::open_engine(&path);
    let (batch, _) = counts(&engine, &["a", "c"]);
    let protected = batch.protect(&engine, []).unwrap();
    let visits = Cell::new(0);
    assert!(
        count.saturating_sub(1) * 2 < protected.updates()[0].allocation().range().start().get()
    );
    let ranges = (0..count)
        .rev()
        .map(|index| {
            Ok((
                FileRange::new(PhysicalAddress::new(index * 2), 1).unwrap(),
                RetainedKind::Dead,
            ))
        })
        .inspect(|_| visits.set(visits.get() + 1));
    protected.exclude_retained(ranges).unwrap();
    assert_eq!(visits.get(), count);
    let failure = FormatError::OffsetOverflow {
        offset: u64::MAX,
        length: 1,
    };
    let error = protected
        .exclude_retained(iter::once(Err(failure.clone())))
        .unwrap_err();
    let CountProtectionError::Source(actual) = error else {
        panic!("expected the retained-range conversion error")
    };
    assert_eq!(actual, failure);
}

#[test]
fn protection_rejects_distinct_objects_sharing_a_continuation_allocation() {
    let path = test_util::temp::temp_path("protected-count-overlap.h5");
    let mut fixture = Fixture::new();
    let child = v2::continuation_chunk(&[Message::reference_count(2)], v2::HeaderFlags::default());
    let child_at = fixture.image.append(&child);
    let header = v2::Header::new()
        .continuation(child_at, &child, Widths::EIGHT)
        .build();
    let first = u64::try_from(fixture.image.append(&header)).unwrap();
    let second = u64::try_from(fixture.image.append(&header)).unwrap();
    fixture.link("a", first);
    fixture.link("b", first);
    fixture.link("c", second);
    fixture.link("d", second);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let (batch, _) = counts(&engine, &["a", "c"]);
    let range = FileRange::new(
        PhysicalAddress::new(u64::try_from(child_at).unwrap()),
        u64::try_from(child.len()).unwrap(),
    )
    .unwrap();
    let Err(CountProtectionError::Overlap {
        first: a,
        second: b,
    }) = batch.protect(&engine, [])
    else {
        panic!("expected overlapping object allocations")
    };
    assert_eq!(
        [a, b],
        [
            (ObjectId::new(PhysicalAddress::new(first)), range),
            (ObjectId::new(PhysicalAddress::new(second)), range)
        ]
    );
}

#[rstest]
#[case::empty(0)]
#[case::wide(128)]
fn reusable_exclusion_visits_each_needed_extent_once_and_preserves_errors(#[case] count: u64) {
    let path = test_util::temp::temp_path("protected-count-visits.h5");
    fixture(&path);
    let engine = edit_test_util::open_engine(&path);
    let (batch, _) = counts(&engine, &["a", "c"]);
    let protected = batch.protect(&engine, []).unwrap();
    let visits = Cell::new(0);
    assert!(
        count.saturating_sub(1) * 2 < protected.updates()[0].allocation().range().start().get()
    );
    let last = protected
        .updates()
        .last()
        .unwrap()
        .allocation()
        .range()
        .end();
    let ranges = (0..count)
        .map(|index| Ok(FileRange::new(PhysicalAddress::new(index * 2), 1).unwrap()))
        .chain([Ok(FileRange::new(last, 1).unwrap())])
        .inspect(|_| visits.set(visits.get() + 1));
    protected.exclude_reusable(ranges).unwrap();
    assert_eq!(visits.get(), count + 1);
    let failure = FormatError::OffsetOverflow {
        offset: u64::MAX,
        length: 1,
    };
    let error = protected
        .exclude_reusable(iter::once(Err(failure.clone())))
        .unwrap_err();
    let CountProtectionError::Source(actual) = error else {
        panic!("expected the range conversion error")
    };
    assert_eq!(actual, failure);
}

#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn protection_keeps_empty_work_allocation_and_read_free() {
    let path = test_util::temp::temp_path("protected-count-cost.h5");
    fixture(&path);
    let (mut engine, calls) = engine_with_read_call_counter(&path);
    engine.space = SessionSpace::from_persisted(
        Some(FileSpaceStrategy::FsmAggr),
        1,
        FileSpacePageSize::MIN,
        (0..256).map(|index| (None, Extent::new(StoredAddress::new(index * 2), 1).unwrap())),
    )
    .unwrap();
    assert!(engine.aggregation.is_some());
    assert_eq!(engine.space.reusable_sections().count(), 256);
    let (batch, origins) = counts(&engine, &["a", "c"]);
    assert!(calls.load(Ordering::Relaxed) > 0);
    calls.store(0, Ordering::Relaxed);
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    let (protected, control) = allocation::measure("protect-counts-indexed-control", || {
        batch.protect(&engine, [&origins[0]]).unwrap()
    });
    assert_eq!(protected.updates().len(), 1);
    assert_eq!(protected.updates()[0].count().get(), 1);
    assert!(control.blocks > 0 && control.bytes > 0, "{control}");
    assert!(
        control.bytes <= 4_096 && control.peak_bytes <= 4_096,
        "{control}"
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    DISCOVERY_WORK.with(|work| assert_eq!(work.get(), (0, 0)));
    let (batch, origins) = counts(&engine, &["a", "b"]);
    calls.store(0, Ordering::Relaxed);
    let untouched = origins
        .iter()
        .inspect(|_| panic!("empty protection inspects no origins"));
    let (empty, measured) = allocation::measure("protect-counts-empty", || {
        batch.protect(&engine, untouched).unwrap()
    });
    assert!(empty.updates().is_empty());
    assert_eq!(empty.root(), None);
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

fn fixture(path: &Path) -> (u64, u64) {
    fixture_with_base(path, UserblockSize::NONE)
}

pub(super) fn fixture_with_base(path: &Path, userblock: UserblockSize) -> (u64, u64) {
    fixture_with_storage(path, userblock, CountStorage::Header)
}

/// Writes counted objects and returns their stored header offsets.
pub(super) fn fixture_with_storage(
    path: &Path,
    userblock: UserblockSize,
    storage: CountStorage,
) -> (u64, u64) {
    write_count_fixture(Fixture::new(), path, userblock, storage)
}

/// Writes adjacent retirement allocations before the counted object storage.
pub(super) fn fixture_with_retirements(
    path: &Path,
    userblock: UserblockSize,
    storage: CountStorage,
) -> RetirementFixture {
    let mut fixture = Fixture::new();
    let first = u64::try_from(fixture.image.append(&[7; 16])).unwrap();
    let second = u64::try_from(fixture.image.append(&[11; 24])).unwrap();
    let retired = [
        (
            FileRange::new(PhysicalAddress::new(first + userblock.get()), 16).unwrap(),
            FreeClass::Page(PageType::Meta),
        ),
        (
            FileRange::new(PhysicalAddress::new(second + userblock.get()), 24).unwrap(),
            FreeClass::Page(PageType::Raw),
        ),
    ];
    RetirementFixture {
        objects: write_count_fixture(fixture, path, userblock, storage),
        retired,
    }
}

/// Counted objects and adjacent original retirements in one byte fixture.
pub(super) struct RetirementFixture {
    /// The stored header offsets of the objects named `a` and `c`.
    pub(super) objects: (u64, u64),
    /// The original physical allocation ranges and their free-space classes.
    pub(super) retired: [(FileRange, FreeClass); 2],
}

fn write_count_fixture(
    mut fixture: Fixture,
    path: &Path,
    userblock: UserblockSize,
    storage: CountStorage,
) -> (u64, u64) {
    let c = append_count_header(&mut fixture, storage);
    let a = append_count_header(&mut fixture, storage);
    fixture.link("a", a);
    fixture.link("b", a);
    fixture.link("c", c);
    fixture.link("d", c);
    fixture.write(path);
    if userblock.get() != 0 {
        let mut image = vec![0; usize::try_from(userblock.get()).unwrap()];
        image.extend_from_slice(&std::fs::read(path).unwrap());
        let header = Superblock::new(Widths::EIGHT)
            .version(SUPERBLOCK_VERSION)
            .base_address(userblock.get())
            .eof_address(u64::try_from(image.len()).unwrap())
            .root_header_address(Fixture::ROOT)
            .build();
        let start = usize::try_from(userblock.get()).unwrap();
        image[start..start + header.len()].copy_from_slice(&header);
        std::fs::write(path, image).unwrap();
    }
    (a, c)
}

pub(super) fn counts(
    engine: &WriteEngine,
    names: &[&str],
) -> (SurvivingCounts, Vec<ReconstructionSource>) {
    let mut phase = AnalysisPhase::new(engine);
    let mut origins = Vec::new();
    let mut deletions = Vec::new();
    for name in names {
        let path = ObjectPathBuf::parse(name);
        let (_, object) = phase.topology().unwrap().find_link(&path).unwrap();
        origins.push(phase.reconstruction_source(engine, object).unwrap());
        deletions.push(phase.topology().unwrap().deletion(&path).unwrap());
    }
    let plan = phase.source_count_plan(&deletions).unwrap().unwrap();
    (plan.surviving_counts(engine).unwrap(), origins)
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(super) fn engine_with_read_call_counter(path: &Path) -> (WriteEngine, Arc<AtomicU64>) {
    let calls = Arc::new(AtomicU64::new(0));
    let engine = WriteEngine::open_imaged(path, Some(FileLocking::Enabled), |handle, len| {
        Ok(Box::new(
            CountingImage::new(
                Box::new(HandleImage::new(
                    handle,
                    len,
                    MetadataCacheConfig::disabled(),
                )),
                Arc::default(),
                Arc::default(),
            )
            .with_read_calls(calls.clone()),
        ))
    })
    .unwrap();
    (engine, calls)
}

fn append_count_header(fixture: &mut Fixture, storage: CountStorage) -> u64 {
    match storage {
        CountStorage::Header => fixture.append_header(vec![Message::reference_count(2)]),
        CountStorage::Continuation => {
            let child =
                v2::continuation_chunk(&[Message::reference_count(2)], v2::HeaderFlags::default());
            let at = fixture.image.append(&child);
            let header = v2::Header::new()
                .continuation(at, &child, Widths::EIGHT)
                .build();
            u64::try_from(fixture.image.append(&header)).unwrap()
        }
    }
}

/// The object-header chunk containing a fixture count message.
#[derive(Clone, Copy, Debug)]
pub(super) enum CountStorage {
    Continuation,
    Header,
}

#[derive(Clone, Copy)]
enum Failure {
    Allocation,
    Phase,
    Repeated,
    Root,
    Uncertain,
}

/// The version used for the relocated fixture's superblock.
const SUPERBLOCK_VERSION: u8 = 3;
