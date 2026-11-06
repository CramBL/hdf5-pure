use core::num::NonZeroU32;
use std::collections::BTreeMap;
use std::io::ErrorKind;

use rstest::rstest;
use test_util::disk_log;
use test_util::object_header::Message;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use super::{
    Disposition, PlacementError, ReconstructionDispositions, TransitionWithheld, Withheld,
};
use crate::edit::WriteEngine;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::counts::{CountContradiction, CountEvidenceError};
use crate::edit::commit::plan::inventory::{InventoryPreparation, PreparedEditInventory};
use crate::edit::commit::plan::{CommitPlan, ReplacedHeader};
use crate::edit::retirement::RootRetirement;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::image::WriteBuffering;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;
use crate::{AttrValue, Datatype, File, FileBuilder, FixedPointType, PhysicalAddress};

#[rstest]
#[case::root_only(false)]
#[case::dataset_and_ancestors(true)]
fn all_predecessors_are_reconciled_even_without_count_changes(#[case] dataset: bool) {
    let path = test_util::temp::temp_path("reconstruction-values.h5");
    let mut builder = FileBuilder::new();
    builder
        .create_group_with("a", |a| {
            a.create_group_with("b", |b| {
                b.create_dataset("value").with_i32_data(&[11]);
                Ok(())
            })
        })
        .unwrap();
    builder.create_dataset("untouched").with_i32_data(&[23]);
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    let (_, untouched) = AnalysisPhase::new(&engine)
        .topology()
        .unwrap()
        .find_link(&ObjectPathBuf::parse("untouched"))
        .unwrap();
    if dataset {
        engine
            .set_dataset_attr(
                &ObjectPathBuf::parse("a/b/value"),
                "answer",
                AttrValue::I64(19),
            )
            .unwrap();
        engine
            .create_group(&ObjectPathBuf::parse("a/b/new"))
            .unwrap();
    } else {
        engine
            .set_group_attr(&ObjectPathBuf::root(), "answer", AttrValue::I64(19))
            .unwrap();
    }
    let prepared = prepare(&mut engine);
    assert!(
        prepared
            .original()
            .facts()
            .unwrap()
            .counts()
            .changes()
            .is_empty()
    );
    let predecessors = prepared.dispositions().predecessors();
    assert_eq!(
        predecessors
            .keys()
            .map(|path| path.to_string())
            .collect::<Vec<_>>(),
        if dataset {
            vec!["", "a", "a/b", "a/b/value"]
        } else {
            vec![""]
        }
    );
    assert!(
        predecessors
            .values()
            .all(|old| old.disposition == Disposition::Supersede)
    );
    let expected: Vec<_> = predecessors
        .iter()
        .map(|(path, old)| (path.clone(), old.source.object()))
        .collect();
    let applied = prepared.apply(&mut engine).unwrap();
    let replacements = applied.dispositions().replacements().unwrap();
    assert_eq!(replacements, &applied.tree().relocations);
    assert_eq!(replacements.len(), expected.len());
    for (path, old) in expected {
        let placed = replacements[&old];
        assert_ne!(placed, old);
        if path.is_empty() {
            assert_eq!(placed.header_address(), applied.tree().retirement.root());
        } else {
            let root = applied.tree().retirement.root();
            let (_, actual) = AnalysisPhase::prospective(&engine, root)
                .topology()
                .unwrap()
                .find_link(&path)
                .unwrap();
            assert_eq!(placed, actual);
        }
    }
    assert_eq!(
        applied.dispositions().translate(untouched).unwrap(),
        untouched
    );
    engine.publish_tree(applied.into_tree()).unwrap();
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("a/b/value").unwrap().read_i32().unwrap(), [11]);
    let attributes = if dataset {
        file.dataset("a/b/value").unwrap().attrs().unwrap()
    } else {
        file.root().attrs().unwrap()
    };
    assert_eq!(attributes.len(), 1);
    assert_eq!(attributes["answer"], AttrValue::I64(19));
}

#[rstest]
#[case::owner_excess(1, 2, false, false, "g", Withheld::ExcessCount(2))]
#[case::ancestor_excess(2, 1, false, false, "g", Withheld::Parent(ObjectPathBuf::root()))]
#[case::root_excess(2, 1, false, false, "", Withheld::ExcessCount(2))]
#[case::root_backlink(2, 1, true, false, "", Withheld::RootLinks(1))]
#[case::descendant_root_backlink(2, 1, false, true, "", Withheld::RootLinks(1))]
fn retained_original_owners_withhold_the_complete_transition(
    #[case] root_count: u32,
    #[case] group_count: u32,
    #[case] backlink: bool,
    #[case] descendant_backlink: bool,
    #[case] withheld_path: &str,
    #[case] reason: Withheld,
) {
    let path = test_util::temp::temp_path("retained-reconstruction.h5");
    let mut fixture = Fixture::new();
    let records = Form::CompactDataset.messages(
        &mut fixture,
        &Datatype::FixedPoint(FixedPointType::i32()),
        &11_i32.to_le_bytes(),
    );
    let target = fixture.append_header(records);
    let mut records = vec![
        global_heap_fixture::hard_link("remove", target),
        Message::reference_count(group_count),
    ];
    if descendant_backlink {
        records.push(global_heap_fixture::hard_link("root", Fixture::ROOT));
    }
    let group = fixture.append_header(test_util::group_info::compact_messages(&records));
    fixture.link("g", group);
    if backlink {
        fixture.link("root", Fixture::ROOT);
    }
    let fixture = fixture.with_root_reference_count(NonZeroU32::new(root_count).unwrap());
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("g/remove")).unwrap();
    let prepared = prepare(&mut engine);
    assert_eq!(
        prepared.dispositions().predecessors()[&ObjectPathBuf::parse(withheld_path)].disposition,
        Disposition::Withhold(reason)
    );
    let group_old = &prepared.dispositions().predecessors()[&ObjectPathBuf::parse("g")];
    assert_eq!(
        group_old
            .counts
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap()
            .original,
        group_count
    );
    let applied = prepared.apply(&mut engine).unwrap();
    assert_eq!(
        applied.dispositions().replacements(),
        Err(&TransitionWithheld::Predecessors)
    );
    let (_, original_child) =
        AnalysisPhase::prospective(&engine, PhysicalAddress::new(Fixture::ROOT))
            .topology()
            .unwrap()
            .find_link(&ObjectPathBuf::parse("g/remove"))
            .unwrap();
    assert_eq!(original_child, ObjectId::new(PhysicalAddress::new(target)));
    let root = applied.tree().retirement.root();
    assert!(
        AnalysisPhase::prospective(&engine, root)
            .topology()
            .unwrap()
            .find_link(&ObjectPathBuf::parse("g/remove"))
            .is_none()
    );
}

#[rstest]
#[case::repeated(false)]
#[case::wrong_phase(true)]
fn invalid_predecessor_associations_are_withheld(#[case] wrong_phase: bool) {
    let path = test_util::temp::temp_path("predecessor-associations.h5");
    let mut builder = FileBuilder::new();
    builder.create_group_with("g", |_| Ok(())).unwrap();
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_group_attr(&ObjectPathBuf::parse("g"), "a", AttrValue::I64(1))
        .unwrap();
    let mut prepared = prepare(&mut engine);
    let CommitPlan::Tree(tree) = &mut prepared.prepared.plan else {
        panic!("expected tree");
    };
    let old = tree.groups[&ObjectPathBuf::root()]
        .prepared
        .replaces
        .as_ref()
        .unwrap()
        .source()
        .object();
    let (_, group) = AnalysisPhase::new(&engine)
        .topology()
        .unwrap()
        .find_link(&ObjectPathBuf::parse("g"))
        .unwrap();
    let mut phase = AnalysisPhase::new(&engine);
    let root_source = phase.reconstruction_source(&engine, old).unwrap();
    let child_source = if wrong_phase {
        AnalysisPhase::new(&engine)
            .reconstruction_source(&engine, group)
            .unwrap()
    } else {
        root_source.clone()
    };
    tree.groups
        .get_mut(&ObjectPathBuf::root())
        .unwrap()
        .prepared
        .replaces = Some(ReplacedHeader::Header(root_source));
    tree.groups
        .get_mut(&ObjectPathBuf::parse("g"))
        .unwrap()
        .prepared
        .replaces = Some(ReplacedHeader::Header(child_source));
    let dispositions = ReconstructionDispositions::collect(&engine, &mut phase, tree);
    assert_eq!(
        dispositions.predecessors()[&ObjectPathBuf::parse("g")].disposition,
        Disposition::Withhold(if wrong_phase {
            Withheld::Phase
        } else {
            Withheld::RepeatedPredecessor(old)
        })
    );
}

#[rstest]
#[case::missing(PlacementError::Missing(ObjectPathBuf::root()))]
#[case::unexpected(PlacementError::Unexpected(ObjectPathBuf::parse("unexpected")))]
#[case::repeated(PlacementError::Repeated(ObjectPathBuf::root()))]
#[case::source(PlacementError::Source(ObjectPathBuf::root()))]
#[case::relocations(PlacementError::Relocations)]
#[case::root(PlacementError::Root)]
fn placement_reconciliation_rejects_incomplete_or_disagreeing_events(
    #[case] error: PlacementError,
) {
    let path = test_util::temp::temp_path("placement-correspondence.h5");
    FileBuilder::new().write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_group_attr(&ObjectPathBuf::root(), "a", AttrValue::I64(1))
        .unwrap();
    let mut prepared = prepare(&mut engine);
    let mut dispositions = std::mem::replace(&mut prepared.dispositions, empty_dispositions());
    let mut applied = prepared.apply(&mut engine).unwrap();
    let root = ObjectPathBuf::root();
    let source = dispositions.predecessors()[&root].source.object();
    let placed = applied.tree().relocations[&source];
    match &error {
        PlacementError::Missing(_) => {}
        PlacementError::Unexpected(path) => dispositions.record(path, Some(source), placed),
        PlacementError::Source(_) => dispositions.record(&root, None, placed),
        PlacementError::Repeated(_) => {
            dispositions.record(&root, Some(source), placed);
            dispositions.record(&root, Some(source), placed);
        }
        PlacementError::Relocations => {
            dispositions.record(&root, Some(source), placed);
            applied.tree.relocations.clear();
        }
        PlacementError::Root => {
            dispositions.record(&root, Some(source), placed);
            applied.tree.retirement = RootRetirement::new(source.header_address(), vec![]);
        }
    }
    assert_eq!(
        dispositions.finish(applied.tree()).replacements(),
        Err(&TransitionWithheld::Placement(error))
    );
}

#[test]
fn a_count_only_retained_target_keeps_its_original_identity() {
    let path = test_util::temp::temp_path("count-only-owner.h5");
    let mut fixture = Fixture::new();
    let mut records = Form::CompactDataset.messages(
        &mut fixture,
        &Datatype::FixedPoint(FixedPointType::i32()),
        &11_i32.to_le_bytes(),
    );
    records.push(Message::reference_count(2));
    let target = fixture.append_header(records);
    fixture.link("remove", target);
    fixture.link("retain", target);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("remove")).unwrap();
    let prepared = prepare(&mut engine);
    let target = ObjectId::new(PhysicalAddress::new(target));
    let changes = prepared.original().facts().unwrap().counts().changes();
    assert_eq!(
        (changes.len(), changes[0].object(), changes[0].remaining()),
        (1, target, 1)
    );
    let applied = prepared.apply(&mut engine).unwrap();
    assert_eq!(applied.dispositions().translate(target).unwrap(), target);
    assert!(
        !applied
            .dispositions()
            .replacements()
            .unwrap()
            .contains_key(&target)
    );
}

#[rstest]
#[case::ambiguous(false, true)]
#[case::contradictory(false, false)]
#[case::unavailable_topology(true, false)]
fn unavailable_original_observations_have_explicit_dispositions(
    #[case] topology: bool,
    #[case] ambiguous: bool,
) {
    let path = test_util::temp::temp_path("unavailable-reconstruction.h5");
    let mut fixture = Fixture::new();
    if topology {
        fixture.link("dangling", 0xdead);
    } else {
        let counts = if ambiguous {
            vec![Message::reference_count(1), Message::reference_count(1)]
        } else {
            vec![Message::reference_count(0)]
        };
        let group = fixture.append_header(test_util::group_info::compact_messages(&counts));
        fixture.link("g", group);
    }
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let edited = if topology {
        ObjectPathBuf::root()
    } else {
        ObjectPathBuf::parse("g")
    };
    engine
        .set_group_attr(&edited, "a", AttrValue::I64(1))
        .unwrap();
    let prepared = prepare(&mut engine);
    assert_eq!(
        prepared.dispositions().predecessors()[&edited].disposition,
        Disposition::Withhold(if topology {
            Withheld::Topology
        } else if ambiguous {
            Withheld::CountUnavailable
        } else {
            Withheld::CountError
        })
    );
    if ambiguous {
        assert_eq!(
            prepared.dispositions().predecessors()[&edited].counts,
            Ok(None)
        );
    }
    if !topology && !ambiguous {
        let predecessor = &prepared.dispositions().predecessors()[&edited];
        assert_eq!(
            predecessor.counts,
            Err(CountEvidenceError::Contradiction(CountContradiction {
                object: predecessor.source.object(),
                observed: 0,
                minimum: 1,
            }))
        );
    }
    let applied = prepared.apply(&mut engine).unwrap();
    assert_eq!(
        applied.dispositions().replacements(),
        Err(&TransitionWithheld::Predecessors)
    );
}

#[test]
fn a_failure_after_a_group_placement_returns_no_applied_inventory() {
    let path = test_util::temp::temp_path("failed-reconstruction.h5");
    let mut builder = FileBuilder::new();
    builder.create_group_with("g", |_| Ok(())).unwrap();
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_group_attr(&ObjectPathBuf::parse("g"), "a", AttrValue::I64(1))
        .unwrap();
    let prepared = prepare(&mut engine);
    let source = prepared.dispositions().source().as_ref().unwrap().object();
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    disk_log::start();
    disk_log::fail_write_after(1, 0);
    let error = prepared.apply(&mut engine).err().unwrap();
    let recording = disk_log::finish();
    let Error::Io(error) = error else {
        panic!("unexpected error: {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    assert_eq!(
        recording
            .events
            .iter()
            .filter(|event| matches!(
                event,
                disk_log::IoEvent::Write {
                    outcome: disk_log::IoOutcome::Succeeded,
                    ..
                }
            ))
            .count(),
        1
    );
    assert_eq!(
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap(),
        source
    );
}

fn empty_dispositions() -> ReconstructionDispositions {
    ReconstructionDispositions {
        source: Err(Error::StaleHandle),
        predecessors: BTreeMap::new(),
        expected: BTreeMap::new(),
        placed: BTreeMap::new(),
        placement_error: None,
    }
}

fn prepare(engine: &mut WriteEngine) -> PreparedEditInventory {
    let staged = std::mem::take(&mut engine.staged);
    engine.begin_commit(&staged);
    match PreparedEditInventory::prepare(engine, staged) {
        InventoryPreparation::Tree(prepared) => prepared,
        InventoryPreparation::Rejected { error, .. } => panic!("preparation failed: {error:?}"),
        InventoryPreparation::InPlace(_) => panic!("expected a tree edit"),
    }
}
