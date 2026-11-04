use core::cell::Cell;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use hdf5_pure_format::__private::ObjectReferenceCountIssue;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::datatype;
use test_util::object_header::{Message, MessageFlags, MessageType, v2};
use test_util_hdf5::global_heap_fixture::{Fixture, Form};

use super::super::super::ReplacedHeader;
use super::super::CountProjectionError;
use super::super::{CountDisposition, HeaderCountError, PreparedCounts, PreparedGroup};
use crate::access_mode::AccessMode;
use crate::edit::WriteEngine;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::counts::CountEvidence;
use crate::edit::analysis::counts::changes::OriginalCount;
use crate::edit::copy::CopyTree;
use crate::image::{CountingImage, HandleImage};
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;
use crate::source::MetadataCacheConfig;
use crate::{Datatype, DatatypeByteOrder, FileLocking, FloatingPointType};

#[rstest]
#[case::named(Some(2), true, false, true, 3, Local::Matched)]
#[case::outside(Some(1), false, false, true, 2, Local::Matched)]
#[case::implicit_insert(None, false, false, true, 2, Local::Matched)]
#[case::original_zero(Some(0), false, false, true, 1, Local::Matched)]
#[case::unchanged(Some(2), true, true, true, 2, Local::Matched)]
#[case::implicit_unchanged(None, true, true, true, 1, Local::Matched)]
#[case::maximum_unchanged(Some(u32::MAX), true, true, true, u32::MAX, Local::Matched)]
#[case::retained_zero(Some(1), true, true, false, 0, Local::Matched)]
#[case::source(Some(2), true, false, true, 3, Local::Source)]
#[case::ambiguous(Some(2), true, false, true, 3, Local::Ambiguous)]
#[case::shared(Some(2), true, false, true, 3, Local::Shared)]
fn combined_projection_retains_original_facts_and_exact_dispositions(
    #[case] original: Option<u32>,
    #[case] named: bool,
    #[case] removed: bool,
    #[case] copied: bool,
    #[case] remaining: u32,
    #[case] local: Local,
) {
    let path = test_util::temp::temp_path("combined-count-records.h5");
    let (engine, tree, target, calls) = fixture(&path, original, named);
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase.reconstruction_source(&engine, target).unwrap();
    let mut records = match local {
        Local::Ambiguous => vec![Message::reference_count(2), Message::reference_count(2)],
        Local::Shared => vec![Message::reference_count(2).with_flags(MessageFlags::SHARED)],
        _ => original.into_iter().map(Message::reference_count).collect(),
    };
    records.push(Message::new(MessageType::UNKNOWN, &[0xD5; 33]));
    let mut group = PreparedGroup {
        region: super::gather(&engine, target.header_address().get())
            .with_bytes(v2::message_records(&records, v2::HeaderFlags::default())),
        replaces: (!matches!(local, Local::Source)).then_some(ReplacedHeader::Header(source)),
        ..PreparedGroup::default()
    };
    let before = group.region.to_vec();
    let copy = phase.copy_users(&tree).unwrap().unwrap();
    let added = phase
        .added_copy_users(copied.then_some(Some(&copy)))
        .unwrap()
        .unwrap();
    let deletions = removed.then(|| {
        phase
            .topology()
            .unwrap()
            .deletion(&ObjectPathBuf::parse("type"))
            .unwrap()
    });
    let plan = phase
        .net_count_plan(deletions.as_ref(), &added)
        .unwrap()
        .unwrap();
    let expected = plan.evidence_changes().next().unwrap();
    assert!(calls.load(Ordering::Relaxed) > 0);
    calls.store(0, Ordering::Relaxed);
    let projection = PreparedCounts::combined(&plan, [&mut group]).unwrap();
    let CountDisposition {
        rebuilt,
        source,
        opaque,
        unchanged,
    } = projection.apply().unwrap();
    let same_count = original.unwrap_or(1) == remaining;
    let opaque_count = matches!(local, Local::Ambiguous | Local::Shared);
    assert_eq!(unchanged.len(), usize::from(same_count));
    assert_eq!(
        rebuilt.len(),
        usize::from(!same_count && matches!(local, Local::Matched))
    );
    assert_eq!(source.len(), usize::from(matches!(local, Local::Source)));
    assert_eq!(opaque.len(), usize::from(opaque_count));
    let actual = if same_count {
        &unchanged[0]
    } else if opaque_count {
        let (change, reason) = &opaque[0];
        assert_eq!(
            *reason,
            if matches!(local, Local::Ambiguous) {
                HeaderCountError::Ambiguous
            } else {
                HeaderCountError::Unsupported(ObjectReferenceCountIssue::Shared)
            }
        );
        change
    } else if matches!(local, Local::Source) {
        &source[0]
    } else {
        &rebuilt[0]
    };
    assert_eq!(actual.remaining(), remaining);
    match (actual.original(), expected.original()) {
        (OriginalCount::Topology(actual), OriginalCount::Topology(expected)) => {
            assert!(removed);
            assert!(matches!(actual, CountEvidence::Retained(_)));
            assert!(std::ptr::eq(actual, expected));
            assert_eq!(actual.bound().object(), target);
            assert_eq!(actual.bound().original(), original.unwrap_or(1));
            assert_eq!(actual.bound().field().map(|field| field.value()), original);
        }
        (OriginalCount::Referenced(actual), OriginalCount::Referenced(expected)) => {
            assert!(!removed);
            assert!(std::ptr::eq(actual, expected));
            assert_eq!(actual.object(), target);
            assert_eq!(actual.original(), original.unwrap_or(1));
            assert_eq!(actual.field().map(|field| field.value()), original);
            assert_eq!(actual.minimum(), None);
            assert_eq!(actual.header_allocation().is_none(), original.is_none());
        }
        _ => panic!("original count evidence changed class"),
    }
    if !same_count && matches!(local, Local::Matched) {
        let count = Message::reference_count(remaining);
        if original.is_some() {
            records[0] = count;
        } else {
            records.push(count);
        }
        assert_eq!(
            &group.region[..],
            v2::message_records(&records, v2::HeaderFlags::default())
        );
    } else {
        assert_eq!(&group.region[..], before);
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::empty(Work::Empty)]
#[case::unchanged(Work::Unchanged)]
#[case::implicit_unchanged(Work::Implicit)]
#[case::changed(Work::Changed)]
fn combined_projection_skips_predecessors_for_empty_and_unchanged_work(#[case] work: Work) {
    let path = test_util::temp::temp_path("combined-count-work.h5");
    let original = if matches!(work, Work::Implicit) {
        None
    } else {
        Some(2)
    };
    let (engine, tree, target, calls) = fixture(&path, original, true);
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase.reconstruction_source(&engine, target).unwrap();
    let mut group = PreparedGroup {
        region: super::gather(&engine, target.header_address().get()),
        replaces: Some(ReplacedHeader::Header(source)),
        ..PreparedGroup::default()
    };
    let copy = phase.copy_users(&tree).unwrap().unwrap();
    let added = phase
        .added_copy_users((!matches!(work, Work::Empty)).then_some(Some(&copy)))
        .unwrap()
        .unwrap();
    let deletions = matches!(work, Work::Unchanged | Work::Implicit).then(|| {
        phase
            .topology()
            .unwrap()
            .deletion(&ObjectPathBuf::parse("type"))
            .unwrap()
    });
    let plan = phase
        .net_count_plan(deletions.as_ref(), &added)
        .unwrap()
        .unwrap();
    let changed = matches!(work, Work::Changed);
    if !changed {
        let mut other = AnalysisPhase::new(&engine);
        group.replaces = Some(ReplacedHeader::Header(
            other.reconstruction_source(&engine, target).unwrap(),
        ));
        group.region = group.region.with_bytes(vec![0x30, 20, 0, 0]);
    }
    let before = group.region.to_vec();
    let visits = Cell::new(0);
    calls.store(0, Ordering::Relaxed);
    let (projection, measured) = allocation::measure("combined_count_inventory", || {
        PreparedCounts::combined(
            &plan,
            core::iter::once(&mut group).inspect(|_| visits.set(visits.get() + 1)),
        )
        .unwrap()
    });
    assert_eq!(projection.updates.len(), usize::from(changed));
    let CountDisposition {
        rebuilt,
        source,
        opaque,
        unchanged,
    } = projection.apply().unwrap();
    assert_eq!(visits.get(), usize::from(changed));
    assert_eq!(
        unchanged.len(),
        usize::from(matches!(work, Work::Unchanged | Work::Implicit))
    );
    assert_eq!(rebuilt.len(), usize::from(changed));
    assert!(source.is_empty() && opaque.is_empty());
    if matches!(work, Work::Empty) {
        assert_eq!((measured.blocks, measured.bytes), (0, 0), "{measured}");
    } else {
        assert!(
            measured.blocks > 0 && measured.live_bytes > 0 && measured.bytes < 4_096,
            "{measured}"
        );
    }
    if !changed {
        assert_eq!(&group.region[..], before);
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn combined_projection_partitions_changed_and_unchanged_targets_in_one_batch() {
    let path = test_util::temp::temp_path("mixed-combined-counts.h5");
    let (engine, tree, target, calls) = fixture(&path, Some(2), true);
    let mut phase = AnalysisPhase::new(&engine);
    let (_, other) = phase
        .topology()
        .unwrap()
        .find_link(&ObjectPathBuf::parse("other"))
        .unwrap();
    let mut groups = [target, other].map(|object| PreparedGroup {
        region: super::gather(&engine, object.header_address().get()),
        replaces: Some(ReplacedHeader::Header(
            phase.reconstruction_source(&engine, object).unwrap(),
        )),
        ..PreparedGroup::default()
    });
    let before = groups[0].region.to_vec();
    let copy = phase.copy_users(&tree).unwrap().unwrap();
    let added = phase.added_copy_users([Some(&copy)]).unwrap().unwrap();
    let deletions = ["type", "other"].map(|path| {
        phase
            .topology()
            .unwrap()
            .deletion(&ObjectPathBuf::parse(path))
            .unwrap()
    });
    let plan = phase.net_count_plan(&deletions, &added).unwrap().unwrap();
    calls.store(0, Ordering::Relaxed);
    let visits = Cell::new(0);
    let prepared = PreparedCounts::combined(
        &plan,
        groups.iter_mut().inspect(|_| visits.set(visits.get() + 1)),
    )
    .unwrap();
    assert_eq!(prepared.updates.len(), 1);
    let CountDisposition {
        rebuilt,
        source,
        opaque,
        unchanged,
    } = prepared.apply().unwrap();
    assert_eq!(visits.get(), 2);
    assert!(source.is_empty() && opaque.is_empty());
    assert_eq!(unchanged.len(), 1);
    assert_eq!(rebuilt.len(), 1);
    let OriginalCount::Topology(unchanged_original) = unchanged[0].original() else {
        panic!("expected topology evidence");
    };
    let OriginalCount::Topology(changed_original) = rebuilt[0].original() else {
        panic!("expected topology evidence");
    };
    assert_eq!(
        (
            unchanged_original.bound().object(),
            unchanged[0].remaining()
        ),
        (target, 2)
    );
    assert_eq!(
        (changed_original.bound().object(), rebuilt[0].remaining()),
        (other, 1)
    );
    assert_eq!(&groups[0].region[..], before);
    assert_eq!(groups[1].region.reference_count_mut().unwrap().count(), 1);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[rstest]
#[case::explicit(Some(2), false, Local::Matched)]
#[case::implicit(None, false, Local::Matched)]
#[case::unchanged(Some(2), true, Local::Matched)]
#[case::source(Some(2), false, Local::Source)]
#[case::ambiguous(Some(2), false, Local::Ambiguous)]
#[case::shared(Some(2), false, Local::Shared)]
#[case::omitted_explicit(Some(2), false, Local::Omitted)]
#[case::omitted_implicit(None, false, Local::Omitted)]
#[case::added_explicit(Some(2), false, Local::Added)]
fn owned_projection_uses_preflight_facts_and_rejects_opaque_replacements(
    #[case] original: Option<u32>,
    #[case] unchanged: bool,
    #[case] local: Local,
) {
    let path = test_util::temp::temp_path("owned-preflight-count-records.h5");
    let (engine, tree, target, calls) = fixture(&path, original, unchanged);
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase.reconstruction_source(&engine, target).unwrap();
    let copy = phase.copy_users(&tree).unwrap().unwrap();
    let added = phase.added_copy_users([Some(&copy)]).unwrap().unwrap();
    let deletion = unchanged.then(|| {
        phase
            .topology()
            .unwrap()
            .deletion(&ObjectPathBuf::parse("type"))
            .unwrap()
    });
    let plan = phase
        .net_count_plan(deletion.as_ref(), &added)
        .unwrap()
        .unwrap();
    let inventory = plan.owned_counts(&engine).unwrap();
    drop(plan);
    drop(phase);
    let checked = inventory
        .preflight(
            &engine,
            (!matches!(local, Local::Source | Local::Added)).then_some(&source),
        )
        .unwrap();
    let mut records = match local {
        Local::Ambiguous => vec![Message::reference_count(2), Message::reference_count(2)],
        Local::Shared => vec![Message::reference_count(2).with_flags(MessageFlags::SHARED)],
        _ => original.into_iter().map(Message::reference_count).collect(),
    };
    records.push(Message::new(MessageType::UNKNOWN, &[0xD5; 33]));
    let mut group = PreparedGroup {
        region: super::gather(&engine, target.header_address().get())
            .with_bytes(v2::message_records(&records, v2::HeaderFlags::default())),
        replaces: (!matches!(local, Local::Source | Local::Omitted))
            .then_some(ReplacedHeader::Header(source)),
        ..PreparedGroup::default()
    };
    let before = group.region.to_vec();
    calls.store(0, Ordering::Relaxed);
    let projection = PreparedCounts::owned(&checked, [&mut group]);
    if matches!(local, Local::Added | Local::Omitted) {
        let Err(CountProjectionError::Reconstruction {
            object,
            expected,
            actual,
        }) = projection
        else {
            panic!("expected a reconstruction decision mismatch")
        };
        assert_eq!(
            (object, expected, actual),
            (
                target,
                matches!(local, Local::Omitted),
                matches!(local, Local::Added)
            )
        );
        assert_eq!(&group.region[..], before);
        assert_eq!(
            checked.protected().updates().len(),
            usize::from(matches!(local, Local::Added))
        );
    } else if matches!(local, Local::Ambiguous | Local::Shared) {
        let Err(CountProjectionError::Region { object, error }) = projection else {
            panic!("expected an opaque replacement count rejection")
        };
        assert_eq!(object, target);
        assert_eq!(
            error,
            if matches!(local, Local::Ambiguous) {
                HeaderCountError::Ambiguous
            } else {
                HeaderCountError::Unsupported(ObjectReferenceCountIssue::Shared)
            }
        );
        assert_eq!(&group.region[..], before);
    } else {
        let (disposition, proof) = projection.unwrap().apply().unwrap();
        assert_eq!(disposition.unchanged.len(), usize::from(unchanged));
        assert_eq!(
            disposition.source.len(),
            usize::from(matches!(local, Local::Source))
        );
        assert_eq!(
            disposition.rebuilt.len(),
            usize::from(!unchanged && matches!(local, Local::Matched))
        );
        assert!(disposition.opaque.is_empty());
        let actual = disposition
            .unchanged
            .iter()
            .chain(&disposition.source)
            .chain(&disposition.rebuilt)
            .next()
            .unwrap();
        assert!(std::ptr::eq(*actual, &checked.changes()[0]));
        if !unchanged && matches!(local, Local::Matched) {
            let remaining = original.unwrap_or(1) + 1;
            if original.is_some() {
                records[0] = Message::reference_count(remaining);
            } else {
                records.push(Message::reference_count(remaining));
            }
            assert_eq!(
                &group.region[..],
                v2::message_records(&records, v2::HeaderFlags::default())
            );
        } else {
            assert_eq!(&group.region[..], before);
        }
        drop(disposition);
        let (inventory, protected) = checked.into_parts(proof).unwrap();
        assert_eq!(inventory.changes().len(), 1);
        assert_eq!(
            protected.updates().len(),
            usize::from(matches!(local, Local::Source))
        );
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn preflight_rejects_a_projection_proof_from_another_batch() {
    let path = test_util::temp::temp_path("owned-count-foreign-proof.h5");
    let (engine, tree, _, _) = fixture(&path, Some(2), false);
    let mut phase = AnalysisPhase::new(&engine);
    let copy = phase.copy_users(&tree).unwrap().unwrap();
    let added = phase.added_copy_users([Some(&copy)]).unwrap().unwrap();
    let plan = phase.net_count_plan([], &added).unwrap().unwrap();
    let first = plan
        .owned_counts(&engine)
        .unwrap()
        .preflight(&engine, [])
        .unwrap();
    let second = plan
        .owned_counts(&engine)
        .unwrap()
        .preflight(&engine, [])
        .unwrap();
    drop(plan);
    drop(phase);
    let (disposition, proof) = PreparedCounts::owned(&first, []).unwrap().apply().unwrap();
    assert_eq!(disposition.source.len(), 1);
    drop(disposition);
    let Err(error) = second.into_parts(proof) else {
        panic!("expected a foreign preflight proof rejection")
    };
    assert_eq!(error, CountProjectionError::Preflight);
    assert_eq!(first.protected().updates().len(), 1);
}

fn fixture(
    path: &Path,
    original: Option<u32>,
    named: bool,
) -> (WriteEngine, CopyTree, ObjectId, Arc<AtomicU64>) {
    let mut fixture = Fixture::new();
    let mut records = vec![Message::new(MessageType::DATATYPE, &datatype::f64_le())];
    records.extend(original.map(Message::reference_count));
    let target = fixture.append_header(records);
    if named {
        fixture.link("type", target);
    }
    let other = fixture.append_header(vec![Message::reference_count(2)]);
    fixture.link("other", other);
    let mut records = Form::CompactDataset.messages(
        &mut fixture,
        &Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(
            DatatypeByteOrder::LittleEndian,
        )),
        &17_f64.to_le_bytes(),
    );
    let datatype = records
        .iter_mut()
        .find(|message| message.msg_type == MessageType::DATATYPE)
        .unwrap();
    *datatype = Message::new(
        MessageType::DATATYPE,
        &datatype::committed_reference(target),
    )
    .with_flags(MessageFlags::SHARED);
    let source = fixture.append_header(records);
    fixture.link("source", source);
    fixture.write(path);
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
            .with_read_calls(Arc::clone(&calls)),
        ))
    })
    .unwrap();
    let tree = WriteEngine::read_copy_subtree(
        &engine.opened_source(),
        AccessMode::ReadWrite,
        source,
        false,
        engine.widths(),
    )
    .unwrap();
    (engine, tree, super::id(target), calls)
}

#[derive(Clone, Copy)]
enum Local {
    Added,
    Ambiguous,
    Matched,
    Omitted,
    Shared,
    Source,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[derive(Clone, Copy)]
enum Work {
    Changed,
    Empty,
    Implicit,
    Unchanged,
}
