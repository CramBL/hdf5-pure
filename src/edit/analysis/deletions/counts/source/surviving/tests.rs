use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::object_header::v2;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;

use crate::FileLocking;
use crate::edit::WriteEngine;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::DISCOVERY_WORK;
use crate::edit::test_util as edit_test_util;
use crate::image::CountingImage;
use crate::image::HandleImage;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;
use crate::source::MetadataCacheConfig;

use super::CountHandoffError;

#[rstest]
#[case::alias(Some(2), false, Some(1))]
#[case::excess(Some(7), true, Some(5))]
#[case::zero(Some(2), true, None)]
#[case::implicit(None, false, None)]
fn detached_counts_retain_only_explicit_positive_remainders(
    #[case] original: Option<u32>,
    #[case] remove_all: bool,
    #[case] remaining: Option<u32>,
) {
    let path = test_util::temp::temp_path("detached-counts.h5");
    let mut fixture = Fixture::new();
    let records = original.into_iter().map(Message::reference_count).collect();
    let address = fixture.append_header(records);
    fixture.link("a", address);
    if original.is_some() {
        fixture.link("b", address);
    }
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let object = ObjectId::new(PhysicalAddress::new(address));
    let origin = phase.reconstruction_source(&engine, object).unwrap();
    let topology = phase.topology().unwrap();
    let mut deletions = vec![topology.deletion(&ObjectPathBuf::parse("a")).unwrap()];
    if remove_all {
        deletions.push(topology.deletion(&ObjectPathBuf::parse("b")).unwrap());
    }
    let plan = phase.source_count_plan(&deletions).unwrap().unwrap();
    let detached = plan.surviving_counts(&engine).unwrap();
    drop(plan);
    drop(phase);
    assert!(detached.same_session(&engine));
    assert_eq!(
        detached.root(),
        remaining.map(|_| ObjectId::new(PhysicalAddress::new(Fixture::ROOT)))
    );
    assert_eq!(detached.shares_phase(&origin), remaining.is_some());
    assert_eq!(detached.updates().len(), usize::from(remaining.is_some()));
    if let Some(count) = remaining {
        let update = &detached.updates()[0];
        assert_eq!(update.object(), object);
        assert_eq!(update.field().value(), original.unwrap());
        assert_eq!(update.field().range().start().get(), address + 12);
        assert_eq!(update.field().chunk().start(), object.header_address());
        assert_eq!(update.count().get(), count);
        engine
            .write_at(update.field().range().start().get(), &9_u32.to_le_bytes())
            .unwrap();
        assert!(detached.same_session(&engine));
        assert_eq!(update.field().value(), original.unwrap());
        assert_eq!(update.count().get(), count);
    }
    let mut next_phase = AnalysisPhase::new(&engine);
    let other_origin = next_phase.reconstruction_source(&engine, object).unwrap();
    assert!(!detached.shares_phase(&other_origin));
    drop(next_phase);
    drop(engine);
    let reopened = edit_test_util::open_engine(&path);
    assert_eq!(detached.same_session(&reopened), remaining.is_none());
}

#[rstest]
fn detachment_rejects_other_engines_and_prospective_roots() {
    let path = test_util::temp::temp_path("detached-count-root.h5");
    let other_path = test_util::temp::temp_path("detached-count-other.h5");
    let mut fixture = Fixture::new();
    let address = fixture.append_header(vec![Message::reference_count(3)]);
    let prospective_root = fixture.append_header(vec![
        Message::new(MessageType::GROUP_INFO, &[0, 0]),
        global_heap_fixture::hard_link("a", address),
        global_heap_fixture::hard_link("b", address),
    ]);
    fixture.link("a", address);
    fixture.link("b", address);
    fixture.write(&path);
    std::fs::copy(&path, &other_path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let other = edit_test_util::open_engine(&other_path);
    let mut phase = AnalysisPhase::new(&engine);
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("a"))
        .unwrap();
    let plan = phase.source_count_plan([&deletion]).unwrap().unwrap();
    let Err(error) = plan.surviving_counts(&other) else {
        panic!("expected a different source engine")
    };
    assert_eq!(error, CountHandoffError::Engine);
    drop(plan);
    drop(phase);
    let mut phase = AnalysisPhase::prospective(&engine, PhysicalAddress::new(prospective_root));
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("a"))
        .unwrap();
    let plan = phase.source_count_plan([&deletion]).unwrap().unwrap();
    let Err(error) = plan.surviving_counts(&engine) else {
        panic!("expected the prospective root to differ")
    };
    assert_eq!(
        error,
        CountHandoffError::Root {
            expected: ObjectId::new(PhysicalAddress::new(prospective_root)),
            actual: ObjectId::new(PhysicalAddress::new(Fixture::ROOT)),
        }
    );
}

#[rstest]
#[case::empty(false)]
#[case::survivor(true)]
fn detachment_does_not_read_or_rediscover_source(#[case] survivor: bool) {
    let path = test_util::temp::temp_path("detached-count-reads.h5");
    let mut fixture = Fixture::new();
    let address = fixture.append_header(vec![Message::reference_count(2)]);
    fixture.link("a", address);
    fixture.link("b", address);
    fixture.write(&path);
    let calls = Arc::new(AtomicU64::new(0));
    let engine = WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, len| {
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
    let mut phase = AnalysisPhase::new(&engine);
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("a"))
        .unwrap();
    let mut deletions = vec![deletion];
    if !survivor {
        deletions.push(
            phase
                .topology()
                .unwrap()
                .deletion(&ObjectPathBuf::parse("b"))
                .unwrap(),
        );
    }
    let plan = phase.source_count_plan(&deletions).unwrap().unwrap();
    assert!(calls.load(Ordering::Relaxed) > 0);
    calls.store(0, Ordering::Relaxed);
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    let detached = plan.surviving_counts(&engine).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    DISCOVERY_WORK.with(|work| assert_eq!(work.get(), (0, 0)));
    assert_eq!(detached.updates().len(), usize::from(survivor));
}

#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn detachment_retains_only_bounded_count_metadata_and_empty_batches_allocate_nothing() {
    let path = test_util::temp::temp_path("detached-count-allocation.h5");
    let mut fixture = Fixture::new();
    let mut links = Vec::new();
    for index in 0..64 {
        let address = fixture.append_header(vec![Message::reference_count(2)]);
        links.push(global_heap_fixture::hard_link(
            &format!("a{index}"),
            address,
        ));
        links.push(global_heap_fixture::hard_link(
            &format!("b{index}"),
            address,
        ));
    }
    let group = u64::try_from(
        fixture.image.append(
            &v2::Header::new()
                .flags(v2::HeaderFlags::CHUNK_SIZE_WIDTH)
                .message(Message::new(MessageType::GROUP_INFO, &[0, 0]))
                .messages(links)
                .build(),
        ),
    )
    .unwrap();
    fixture.link("g", group);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let topology = phase.topology().unwrap();
    let deletions = (0..64)
        .map(|index| {
            topology
                .deletion(&ObjectPathBuf::parse(&format!("g/a{index}")))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let plan = phase.source_count_plan(&deletions).unwrap().unwrap();
    let expected = plan
        .changes()
        .map(|change| {
            (
                change.bound().object(),
                change.bound().field().unwrap(),
                change.remaining(),
            )
        })
        .collect::<Vec<_>>();
    let (detached, measured) = allocation::measure("detach-surviving-counts-wide", || {
        plan.surviving_counts(&engine).unwrap()
    });
    assert_eq!(
        detached
            .updates()
            .iter()
            .map(|update| (update.object(), update.field(), update.count().get()))
            .collect::<Vec<_>>(),
        expected
    );
    let retained = u64::try_from(64 * core::mem::size_of::<super::SurvivingCount>()).unwrap();
    assert!(
        measured.blocks > 0 && measured.bytes >= retained,
        "{measured}"
    );
    assert!(
        measured.bytes <= 4 * retained && measured.live_bytes <= 2 * retained,
        "{measured}"
    );
    drop(plan);
    let mut deletions = deletions;
    let topology = phase.topology().unwrap();
    deletions.extend((0..64).map(|index| {
        topology
            .deletion(&ObjectPathBuf::parse(&format!("g/b{index}")))
            .unwrap()
    }));
    let plan = phase.source_count_plan(&deletions).unwrap().unwrap();
    let (empty, measured) = allocation::measure("detach-surviving-counts-empty", || {
        plan.surviving_counts(&engine).unwrap()
    });
    assert!(empty.updates().is_empty());
    assert_eq!(empty.root(), None);
    assert_eq!(measured.blocks, 0, "{measured}");
}

#[test]
fn a_positive_changed_implicit_count_reports_a_broken_plan_invariant() {
    let path = test_util::temp::temp_path("detached-count-invariant.h5");
    let mut fixture = Fixture::new();
    let address = fixture.append_header(vec![]);
    fixture.link("a", address);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("a"))
        .unwrap();
    let mut plan = phase.source_count_plan([&deletion]).unwrap().unwrap();
    plan.plan.remaining.values_mut().next().unwrap().declared = Some(1);
    let Err(error) = plan.surviving_counts(&engine) else {
        panic!("expected the missing explicit field")
    };
    assert_eq!(
        error,
        CountHandoffError::Source(crate::FormatError::Internal(
            "a positive changed count has no explicit source field".into()
        ))
    );
}

#[rstest]
#[case::ambiguous(vec![Message::reference_count(2), Message::reference_count(3)])]
#[case::unsupported(vec![Message::new(MessageType::OBJECT_REFERENCE_COUNT, &[1, 2, 0, 0, 0])])]
fn unresolved_count_forms_leave_no_detached_updates(#[case] records: Vec<Message>) {
    let path = test_util::temp::temp_path("detached-count-unsupported.h5");
    let mut fixture = Fixture::new();
    let address = fixture.append_header(records);
    fixture.link("a", address);
    fixture.link("b", address);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let deletion = phase
        .topology()
        .unwrap()
        .deletion(&ObjectPathBuf::parse("a"))
        .unwrap();
    let plan = phase.source_count_plan([&deletion]).unwrap().unwrap();
    let detached = plan.surviving_counts(&engine).unwrap();
    assert!(detached.updates().is_empty());
    assert_eq!(detached.root(), None);
}
