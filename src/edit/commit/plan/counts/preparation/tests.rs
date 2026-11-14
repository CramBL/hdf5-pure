use std::path::Path;

use rstest::rstest;
use test_util::datatype;
use test_util::disk_log;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use super::PreparedTreeCounts;
use crate::edit::analysis::DISCOVERY_WORK;
use crate::edit::analysis::counts::changes::owned::CountDisposition;
use crate::edit::commit::plan::{CommitPlan, CommitPreparation, PreparedCommit, PreparedTree};
use crate::edit::test_util as edit_test_util;
use crate::edit::{REFERENCE_INTO_RECLAIMED_SPACE, WriteEngine};
use crate::error::Error;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;
use crate::{DatasetBuilder, Datatype, File, FixedPointType, PhysicalAddress};

#[rstest]
#[case::last_aliases(Layout::Aliases, 2, 2, true)]
#[case::surviving_alias(Layout::Aliases, 2, 1, false)]
#[case::excess_count(Layout::Aliases, 3, 2, false)]
#[case::retired_group(Layout::Group, 2, 1, true)]
fn count_aware_preparation_selects_retirement_allocations_at_zero_remaining_count(
    #[case] layout: Layout,
    #[case] original: u32,
    #[case] removed: usize,
    #[case] retired: bool,
) {
    let path = test_util::temp::temp_path("counted-preparation.h5");
    let (object, expected) = fixture(&path, layout, original, false);
    let before = std::fs::read(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    stage_deletions(&mut engine, layout, removed);
    disk_log::start();
    let tree = prepare(&mut engine);
    assert!(disk_log::take().is_empty());
    let Some(PreparedTreeCounts::Complete {
        inventory,
        protected,
        retained_roots,
        eligible_owners,
        retirement,
    }) = tree.counts
    else {
        panic!("expected complete count preparation");
    };
    let change = inventory
        .changes()
        .iter()
        .find(|change| change.object() == object)
        .unwrap();
    let removed_edges = if matches!(layout, Layout::Group) {
        2
    } else {
        removed as u32
    };
    assert_eq!(
        (change.original().value(), change.remaining()),
        (original, original - removed_edges)
    );
    assert_eq!(eligible_owners.contains(&object), retired);
    assert_eq!(retained_roots.contains(&object), !retired);
    assert_eq!(protected.updates().len(), usize::from(!retired));
    assert!(retirement.is_empty());
    let mut actual = tree
        .deleted_free
        .iter()
        .map(|(range, _)| (range.start().get(), range.end().get()))
        .collect::<Vec<_>>();
    actual.sort_unstable();
    assert_eq!(actual, if retired { expected } else { Vec::new() });
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[rstest]
#[case::explicit(Some(1))]
#[case::implicit(None)]
fn retained_zero_counts_preserve_reference_roots_without_retirement(#[case] count: Option<u32>) {
    let path = test_util::temp::temp_path("retained-count-preparation.h5");
    let mut fixture = Fixture::new();
    let mut messages = vec![Message::new(MessageType::DATATYPE, &datatype::f64_le())];
    messages.extend(count.map(Message::reference_count));
    let header = fixture.append_header(messages);
    fixture.link("type", header);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("type")).unwrap();
    let tree = prepare(&mut engine);
    let Some(PreparedTreeCounts::Complete {
        inventory,
        protected,
        retained_roots,
        eligible_owners,
        retirement,
    }) = tree.counts
    else {
        panic!("expected complete retained-count preparation");
    };
    let [change] = inventory.changes() else {
        panic!("expected the retained target");
    };
    let CountDisposition::Retain(retained) = change.disposition() else {
        panic!("expected an unlinked retained object");
    };
    assert!(std::ptr::eq(change, retained));
    assert_eq!((change.original().value(), change.remaining()), (1, 0));
    assert_eq!(
        retained_roots,
        [ObjectId::new(PhysicalAddress::new(header))]
    );
    assert!(protected.updates().is_empty());
    assert!(eligible_owners.is_empty());
    assert!(retirement.is_empty());
    assert!(tree.deleted_free.is_empty());
}

#[rstest]
#[case::zero(0)]
#[case::undercount(1)]
fn contradictory_original_counts_preserve_staging_before_writes(#[case] original: u32) {
    let path = test_util::temp::temp_path("counted-contradiction.h5");
    let (object, _) = fixture(&path, Layout::Aliases, original, false);
    let before = std::fs::read(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    stage_deletions(&mut engine, Layout::Aliases, 2);
    let staged = std::mem::take(&mut engine.staged);
    engine.begin_commit(&staged);
    disk_log::start();
    let CommitPreparation::Rejected { staged, error } =
        PreparedCommit::prepare(&mut engine, staged)
    else {
        panic!("expected original count rejection");
    };
    let Error::InconsistentReferenceCount {
        address,
        observed,
        minimum,
    } = error
    else {
        panic!("expected a structured count contradiction, got {error:?}");
    };
    assert_eq!(
        (address, observed, minimum),
        (object.header_address().get(), original, 2)
    );
    assert_eq!(
        staged.deletes(),
        &[ObjectPathBuf::parse("a"), ObjectPathBuf::parse("b")]
    );
    assert!(disk_log::take().is_empty());
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("a").unwrap().read_i32().unwrap(), [7, 11]);
    assert_eq!(file.dataset("b").unwrap().read_i32().unwrap(), [7, 11]);
}

#[rstest]
#[case::copied(true)]
#[case::supplied(false)]
fn count_eligible_allocations_reach_reference_screening(#[case] copied: bool) {
    let path = test_util::temp::temp_path("counted-references.h5");
    let (object, _) = fixture(&path, Layout::Aliases, 2, false);
    let file = File::open_rw(&path).unwrap();
    file.root()
        .create_dataset("references", |dataset| {
            dataset.with_reference_data(&[object.header_address().get()]);
        })
        .unwrap();
    file.commit().unwrap();
    file.close().unwrap();
    let before = std::fs::read(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    stage_deletions(&mut engine, Layout::Aliases, 2);
    if copied {
        engine
            .copy(
                &ObjectPathBuf::parse("references"),
                &ObjectPathBuf::parse("copy"),
            )
            .unwrap();
    } else {
        let mut dataset = DatasetBuilder::new("");
        dataset.with_reference_data(&[object.header_address().get()]);
        engine
            .stage_created_dataset(&ObjectPathBuf::parse("supplied"), dataset)
            .unwrap();
    }
    let staged = std::mem::take(&mut engine.staged);
    engine.begin_commit(&staged);
    let CommitPreparation::Rejected { error, .. } = PreparedCommit::prepare(&mut engine, staged)
    else {
        panic!("expected reclaimed reference rejection");
    };
    let Error::EditUnsupported(reason) = error else {
        panic!("expected unsupported reference, got {error:?}");
    };
    assert_eq!(reason, REFERENCE_INTO_RECLAIMED_SPACE);
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

#[test]
fn unavailable_counts_withhold_deletion_allocations() {
    let path = test_util::temp::temp_path("counted-unavailable.h5");
    fixture(&path, Layout::Aliases, 2, true);
    let mut engine = edit_test_util::open_engine(&path);
    stage_deletions(&mut engine, Layout::Aliases, 2);
    let tree = prepare(&mut engine);
    assert!(matches!(tree.counts, Some(PreparedTreeCounts::Unavailable)));
    assert!(tree.deleted_free.is_empty());
}

#[test]
fn counted_preparation_projects_the_reconstructed_root_count() {
    let path = test_util::temp::temp_path("counted-root.h5");
    let mut fixture =
        Fixture::new().with_root_reference_count(core::num::NonZeroU32::new(2).unwrap());
    fixture.link("self", Fixture::ROOT);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("self")).unwrap();
    let mut tree = prepare(&mut engine);
    assert_eq!(
        tree.groups
            .get_mut(&ObjectPathBuf::root())
            .unwrap()
            .prepared
            .region
            .reference_count_mut()
            .unwrap()
            .count(),
        1
    );
    let Some(PreparedTreeCounts::Complete {
        inventory,
        protected,
        eligible_owners,
        ..
    }) = tree.counts
    else {
        panic!("expected root count preparation");
    };
    assert_eq!(inventory.changes().len(), 1);
    assert_eq!(inventory.changes()[0].remaining(), 1);
    assert!(protected.updates().is_empty());
    assert!(eligible_owners.is_empty());
    assert!(tree.deleted_free.is_empty());
}

#[test]
fn count_aware_value_preparation_skips_topology_discovery() {
    let path = test_util::temp::temp_path("counted-values.h5");
    fixture(&path, Layout::Aliases, 2, false);
    let mut engine = edit_test_util::open_engine(&path);
    let mut replacement = DatasetBuilder::new("");
    replacement.with_i32_data(&[13, 17]);
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("a"), replacement)
        .unwrap();
    let staged = std::mem::take(&mut engine.staged);
    engine.begin_commit(&staged);
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    let CommitPreparation::Prepared(prepared) = PreparedCommit::prepare(&mut engine, staged) else {
        panic!("expected in-place values");
    };
    assert!(matches!(prepared.plan, CommitPlan::InPlace(_)));
    assert_eq!(DISCOVERY_WORK.with(|work| work.get()), (0, 0));
    engine
        .create_group(&ObjectPathBuf::parse("control"))
        .unwrap();
    engine.delete(&ObjectPathBuf::parse("a")).unwrap();
    let _ = prepare(&mut engine);
    assert!(DISCOVERY_WORK.with(|work| work.get().0) > 0);
}

fn prepare(engine: &mut WriteEngine) -> PreparedTree {
    let staged = std::mem::take(&mut engine.staged);
    engine.begin_commit(&staged);
    let CommitPreparation::Prepared(prepared) = PreparedCommit::prepare(engine, staged) else {
        panic!("expected count-aware tree preparation");
    };
    let CommitPlan::Tree(tree) = prepared.plan else {
        panic!("expected tree preparation");
    };
    tree
}

fn stage_deletions(engine: &mut WriteEngine, layout: Layout, removed: usize) {
    let paths = if matches!(layout, Layout::Group) {
        &["group"][..]
    } else {
        &["a", "b"][..removed]
    };
    for path in paths {
        engine.delete(&ObjectPathBuf::parse(path)).unwrap();
    }
}

fn fixture(
    path: &Path,
    layout: Layout,
    count: u32,
    ambiguous: bool,
) -> (ObjectId, Vec<(u64, u64)>) {
    let mut fixture = Fixture::new();
    let raw = fixture.image.as_bytes().len() as u64;
    let mut messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &Datatype::FixedPoint(FixedPointType::i32()),
        &[7_i32.to_le_bytes(), 11_i32.to_le_bytes()].concat(),
    );
    messages.push(Message::reference_count(count));
    if ambiguous {
        messages.push(Message::reference_count(count));
    }
    let target = fixture.append_header(messages);
    let mut ranges = vec![
        (raw, raw + 8),
        (target, fixture.image.as_bytes().len() as u64),
    ];
    match layout {
        Layout::Aliases => {
            fixture.link("a", target);
            fixture.link("b", target);
        }
        Layout::Group => {
            let group = fixture.append_header(test_util::group_info::compact_messages(&[
                global_heap_fixture::hard_link("a", target),
                global_heap_fixture::hard_link("b", target),
            ]));
            ranges.push((group, fixture.image.as_bytes().len() as u64));
            fixture.link("group", group);
        }
    }
    fixture.write(path);
    ranges.sort_unstable();
    (ObjectId::new(PhysicalAddress::new(target)), ranges)
}

#[derive(Clone, Copy)]
enum Layout {
    Aliases,
    Group,
}
