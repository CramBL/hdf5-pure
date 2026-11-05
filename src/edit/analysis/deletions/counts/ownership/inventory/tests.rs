#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use core::cell::Cell;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use core::mem;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use hdf5_pure_core::__private::FileRange;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::datatype;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags;
use test_util::object_header::MessageType;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::object_header::v2;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use super::CountPreflightError;
use super::CountProtectionError;
use crate::Datatype;
use crate::DatatypeByteOrder;
use crate::FloatingPointType;
use crate::PhysicalAddress;
use crate::access_mode::AccessMode;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::edit::WriteEngine;
use crate::edit::analysis::AnalysisPhase;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::edit::analysis::DISCOVERY_WORK;
use crate::edit::analysis::counts::test_util as count_test_util;
use crate::edit::analysis::deletions::counts::tests::source::net as count_tests;
use crate::edit::copy::CopyTree;
use crate::error::Error;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::source(Work::Source, Some(2), false, false)]
#[case::reconstructed(Work::Source, Some(2), false, true)]
#[case::implicit_reconstructed(Work::Source, None, false, true)]
#[case::unchanged(Work::Unchanged, Some(2), true, false)]
#[case::ordinary_zero(Work::Retired, Some(1), true, false)]
#[case::implicit_zero(Work::Retired, None, true, false)]
fn preflight_preserves_original_inventory_and_selected_source_allocations(
    #[case] work: Work,
    #[case] original: Option<u32>,
    #[case] named: bool,
    #[case] rebuilt: bool,
) {
    let path = test_util::temp::temp_path("owned-count-preflight.h5");
    let (engine, tree, target, calls) = fixture(&path, original, named, work);
    let before = engine.disk_bytes().unwrap();
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase.reconstruction_source(&engine, target).unwrap();
    let copy = phase.copy_users(&tree).unwrap().unwrap();
    let added = phase
        .added_copy_users((!matches!(work, Work::Retired)).then_some(Some(&copy)))
        .unwrap()
        .unwrap();
    let deletion = named.then(|| {
        phase
            .topology()
            .unwrap()
            .deletion(&ObjectPathBuf::parse("target"))
            .unwrap()
    });
    let plan = phase
        .net_count_plan(deletion.as_ref(), &added)
        .unwrap()
        .unwrap();
    let inventory = plan.owned_counts(&engine).unwrap();
    let expected = inventory
        .changes()
        .iter()
        .map(|change| {
            (
                change.object(),
                change.original(),
                change.remaining(),
                change.field(),
                change.header_allocation(),
            )
        })
        .collect::<Vec<_>>();
    drop(plan);
    drop(phase);
    assert!(calls.load(Ordering::Relaxed) > 0);
    calls.store(0, Ordering::Relaxed);
    let checked = inventory
        .preflight(&engine, rebuilt.then_some(&source))
        .unwrap();
    assert_eq!(checked.root(), Some(id(Fixture::ROOT)));
    assert!(checked.shares_phase(&source));
    assert_eq!(
        checked
            .changes()
            .iter()
            .map(|change| (
                change.object(),
                change.original(),
                change.remaining(),
                change.field(),
                change.header_allocation()
            ))
            .collect::<Vec<_>>(),
        expected
    );
    let updates = checked.protected().updates();
    let changed = matches!(work, Work::Source);
    assert_eq!(updates.len(), usize::from(changed && !rebuilt));
    if let Some(update) = updates.first() {
        assert_eq!(update.object(), target);
        assert_eq!(update.field().value(), 2);
        assert_eq!(update.count().get(), 3);
        assert_eq!(update.allocation().range(), update.field().chunk());
    }
    assert_eq!(checked.changes().len(), 1);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(engine.disk_bytes().unwrap(), before);
}

#[rstest]
#[case::original_zero(Rejection::OriginalZero)]
#[case::retained_zero(Rejection::RetainedZero)]
#[case::missing_field(Rejection::MissingField)]
#[case::allocation(Rejection::Allocation)]
#[case::phase(Rejection::Phase)]
#[case::repeated(Rejection::Repeated)]
#[case::session(Rejection::Session)]
#[case::empty_session(Rejection::EmptySession)]
#[case::root(Rejection::Root)]
#[case::uncertain(Rejection::Uncertain)]
fn preflight_rejects_unsupported_counts_and_inapplicable_source_facts_before_writes(
    #[case] rejection: Rejection,
) {
    let path = test_util::temp::temp_path("owned-count-rejection.h5");
    let other_path = test_util::temp::temp_path("owned-count-other.h5");
    let original = match rejection {
        Rejection::OriginalZero => Some(0),
        Rejection::MissingField => None,
        Rejection::RetainedZero => Some(1),
        _ => Some(2),
    };
    let named = matches!(rejection, Rejection::RetainedZero);
    let (mut engine, tree, target, _) = fixture(&path, original, named, Work::Source);
    let (other, _, _, _) = fixture(&other_path, original, named, Work::Source);
    let before = engine.disk_bytes().unwrap();
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase.reconstruction_source(&engine, target).unwrap();
    if matches!(rejection, Rejection::Allocation) {
        let observed = phase.object_reference_count(target).unwrap();
        phase.counts.insert(
            target,
            Ok(count_test_util::observation_without_allocation(observed)),
        );
    }
    let copy = phase.copy_users(&tree).unwrap().unwrap();
    let added = phase
        .added_copy_users(
            (!named && !matches!(rejection, Rejection::EmptySession)).then_some(Some(&copy)),
        )
        .unwrap()
        .unwrap();
    let deletion = named.then(|| {
        phase
            .topology()
            .unwrap()
            .deletion(&ObjectPathBuf::parse("target"))
            .unwrap()
    });
    let plan = phase
        .net_count_plan(deletion.as_ref(), &added)
        .unwrap()
        .unwrap();
    let inventory = plan.owned_counts(&engine).unwrap();
    drop(plan);
    drop(phase);
    let mut foreign = AnalysisPhase::new(&engine);
    let foreign_source = foreign.reconstruction_source(&engine, target).unwrap();
    drop(foreign);
    if matches!(rejection, Rejection::Root) {
        let mut proposed = engine.superblock().clone();
        proposed.root_group_address = StoredAddress::new(target.header_address().get());
        engine
            .context
            .adopt(engine.context.propose(proposed).unwrap())
            .unwrap();
    }
    if matches!(rejection, Rejection::Uncertain) {
        engine.make_allocation_uncertain();
    }
    let origins = match rejection {
        Rejection::Phase => vec![&foreign_source],
        Rejection::Repeated => vec![&source, &source],
        _ => Vec::new(),
    };
    let Err(error) = inventory.preflight(
        if matches!(rejection, Rejection::Session | Rejection::EmptySession) {
            &other
        } else {
            &engine
        },
        origins,
    ) else {
        panic!("expected count preflight rejection")
    };
    match (rejection, error) {
        (Rejection::OriginalZero, CountPreflightError::OriginalZero { object })
        | (Rejection::RetainedZero, CountPreflightError::RetainedZero { object }) => {
            assert_eq!(object, target)
        }
        (
            Rejection::MissingField,
            CountPreflightError::MissingField {
                object,
                original,
                remaining,
            },
        ) => assert_eq!((object, original, remaining), (target, 1, 2)),
        (
            Rejection::Allocation,
            CountPreflightError::Protection(CountProtectionError::Allocation { object }),
        )
        | (
            Rejection::Phase,
            CountPreflightError::Protection(CountProtectionError::Phase { object }),
        )
        | (
            Rejection::Repeated,
            CountPreflightError::Protection(CountProtectionError::Repeated { object }),
        ) => assert_eq!(object, target),
        (
            Rejection::Session | Rejection::EmptySession,
            CountPreflightError::Protection(CountProtectionError::Session),
        )
        | (
            Rejection::Uncertain,
            CountPreflightError::Protection(CountProtectionError::Publication(
                Error::AllocationPublicationUncertain,
            )),
        ) => {}
        (
            Rejection::Root,
            CountPreflightError::Protection(CountProtectionError::Root { expected, actual }),
        ) => assert_eq!((expected, actual), (id(Fixture::ROOT), target)),
        (_, other) => panic!("unexpected preflight error: {other:?}"),
    }
    assert_eq!(engine.disk_bytes().unwrap(), before);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::empty(false)]
#[case::unchanged(true)]
fn preflight_skips_source_and_reconstruction_work_without_changed_counts(#[case] populated: bool) {
    let path = test_util::temp::temp_path("owned-count-empty.h5");
    let (engine, tree, _, calls) = fixture(&path, Some(2), true, Work::Unchanged);
    let mut phase = AnalysisPhase::new(&engine);
    let copy = phase.copy_users(&tree).unwrap().unwrap();
    let added = phase
        .added_copy_users(populated.then_some(Some(&copy)))
        .unwrap()
        .unwrap();
    let deletion = populated.then(|| {
        phase
            .topology()
            .unwrap()
            .deletion(&ObjectPathBuf::parse("target"))
            .unwrap()
    });
    let plan = phase
        .net_count_plan(deletion.as_ref(), &added)
        .unwrap()
        .unwrap();
    let inventory = plan.owned_counts(&engine).unwrap();
    drop(plan);
    drop(phase);
    let mut foreign = AnalysisPhase::new(&engine);
    let foreign_source = foreign
        .reconstruction_source(&engine, id(Fixture::ROOT))
        .unwrap();
    drop(foreign);
    let before = DISCOVERY_WORK.with(Cell::get);
    let visits = Cell::new(0);
    assert!(calls.load(Ordering::Relaxed) > 0);
    calls.store(0, Ordering::Relaxed);
    let (checked, measured) = allocation::measure("owned_count_preflight", || {
        inventory
            .preflight(
                &engine,
                std::iter::once(&foreign_source).inspect(|_| visits.set(visits.get() + 1)),
            )
            .unwrap()
    });
    assert_eq!(checked.changes().len(), usize::from(populated));
    assert_eq!(visits.get(), 0);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), before);
    assert_eq!((measured.blocks, measured.bytes), (0, 0), "{measured}");
    assert!(checked.protected().updates().is_empty());
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::one(1)]
#[case::wide(64)]
fn preflight_protects_each_source_field_with_bounded_storage_and_no_reads(#[case] count: usize) {
    let path = test_util::temp::temp_path("owned-count-preflight-wide.h5");
    let mut fixture = Fixture::new();
    let mut links = Vec::new();
    let expected = (0..count)
        .map(|index| {
            let mut messages = Form::CompactDataset.messages(
                &mut fixture,
                &Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(
                    DatatypeByteOrder::LittleEndian,
                )),
                &17_f64.to_le_bytes(),
            );
            messages.push(Message::reference_count(2));
            let header = v2::Header::new()
                .flags(v2::HeaderFlags::CHUNK_SIZE_WIDTH)
                .messages(messages)
                .build();
            let target = u64::try_from(fixture.image.append(&header)).unwrap();
            let record = v2::message_record(
                &Message::reference_count(2),
                v2::HeaderFlags::CHUNK_SIZE_WIDTH,
            );
            let positions = header
                .windows(record.len())
                .enumerate()
                .filter_map(|(at, bytes)| (bytes == record).then_some(at))
                .collect::<Vec<_>>();
            let &[at] = positions.as_slice() else {
                panic!("expected one count record")
            };
            let field = target + u64::try_from(at + record.len() - mem::size_of::<u32>()).unwrap();
            links.extend([
                global_heap_fixture::hard_link(&format!("a{index}"), target),
                global_heap_fixture::hard_link(&format!("b{index}"), target),
            ]);
            (
                id(target),
                FileRange::new(
                    PhysicalAddress::new(field),
                    u64::try_from(mem::size_of::<u32>()).unwrap(),
                )
                .unwrap(),
                FileRange::new(
                    PhysicalAddress::new(target),
                    u64::try_from(header.len()).unwrap(),
                )
                .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let owner = u64::try_from(
        fixture.image.append(
            &v2::Header::new()
                .flags(v2::HeaderFlags::CHUNK_SIZE_WIDTH)
                .messages(links)
                .build(),
        ),
    )
    .unwrap();
    fixture.link("owner", owner);
    fixture.write(&path);
    let (engine, calls) = count_tests::open_counted_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let added = phase.added_copy_users([]).unwrap().unwrap();
    let deletions = (0..count)
        .map(|index| {
            phase
                .topology()
                .unwrap()
                .deletion(&ObjectPathBuf::parse(&format!("owner/a{index}")))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let plan = phase.net_count_plan(&deletions, &added).unwrap().unwrap();
    let inventory = plan.owned_counts(&engine).unwrap();
    drop(plan);
    drop(phase);
    let before = DISCOVERY_WORK.with(Cell::get);
    assert!(calls.load(Ordering::Relaxed) > 0);
    calls.store(0, Ordering::Relaxed);
    let (checked, measured) = allocation::measure("owned_count_preflight_source", || {
        inventory.preflight(&engine, []).unwrap()
    });
    let protected = checked.protected();
    assert_eq!(checked.changes().len(), count);
    assert_eq!(protected.updates().len(), count);
    assert_eq!(
        protected
            .updates()
            .iter()
            .map(|update| {
                assert_eq!((update.field().value(), update.count().get()), (2, 1));
                assert_eq!(update.field().chunk(), update.allocation().range());
                (
                    update.object(),
                    update.field().range(),
                    update.allocation().range(),
                )
            })
            .collect::<Vec<_>>(),
        expected
    );
    let floor = u64::try_from(mem::size_of_val(protected.updates())).unwrap();
    assert!(
        floor > 0 && measured.blocks > 0 && measured.bytes >= floor && measured.live_bytes >= floor,
        "{measured}"
    );
    assert!(
        measured.bytes <= 8 * floor + 4_096 && measured.live_bytes <= 4 * floor + 4_096,
        "{measured}"
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), before);
}

fn fixture(
    path: &Path,
    original: Option<u32>,
    named: bool,
    work: Work,
) -> (WriteEngine, CopyTree, ObjectId, Arc<AtomicU64>) {
    let mut fixture = Fixture::new();
    let records = if matches!(work, Work::Retired) {
        let mut records = Form::CompactDataset.messages(
            &mut fixture,
            &Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(
                DatatypeByteOrder::LittleEndian,
            )),
            &17_f64.to_le_bytes(),
        );
        records.extend(original.map(Message::reference_count));
        records
    } else {
        let mut records = vec![Message::new(MessageType::DATATYPE, &datatype::f64_le())];
        records.extend(original.map(Message::reference_count));
        records
    };
    let target = fixture.append_header(records);
    if named {
        fixture.link("target", target);
    }
    let mut records = Form::CompactDataset.messages(
        &mut fixture,
        &Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(
            DatatypeByteOrder::LittleEndian,
        )),
        &19_f64.to_le_bytes(),
    );
    if !matches!(work, Work::Retired) {
        *records
            .iter_mut()
            .find(|record| record.msg_type == MessageType::DATATYPE)
            .unwrap() = Message::new(
            MessageType::DATATYPE,
            &datatype::committed_reference(target),
        )
        .with_flags(MessageFlags::SHARED);
    }
    let source = fixture.append_header(records);
    fixture.link("source", source);
    fixture.write(path);
    let (engine, calls) = count_tests::open_counted_engine(path);
    let tree = WriteEngine::read_copy_subtree(
        &engine.opened_source(),
        AccessMode::ReadWrite,
        ObjectId::new(PhysicalAddress::new(source)),
        false,
        engine.widths(),
    )
    .unwrap();
    (engine, tree, id(target), calls)
}

fn id(address: u64) -> ObjectId {
    ObjectId::new(PhysicalAddress::new(address))
}

#[derive(Clone, Copy)]
enum Work {
    Retired,
    Source,
    Unchanged,
}

#[derive(Clone, Copy)]
enum Rejection {
    Allocation,
    EmptySession,
    MissingField,
    OriginalZero,
    Phase,
    Repeated,
    RetainedZero,
    Root,
    Session,
    Uncertain,
}
