use std::io::ErrorKind;
use std::path::Path;
use std::sync::atomic::Ordering;

use rstest::rstest;
use test_util::datatype;
use test_util::disk_log::{self, DiskOp};
use test_util::object_header::{Message, MessageFlags, MessageType};
use test_util_hdf5::global_heap_fixture::{Fixture, Form};

use super::storage::StorageOrigin;
use super::{EditInputError, InventoryPreparation, OriginalEdits, PreparedEditInventory};
use crate::DatasetBuilder;
use crate::edit::analysis::physical::claims::domain::OwnerDomain;
use crate::edit::analysis::{AnalysisPhase, DISCOVERY_WORK, PlannedDeletion, ReconstructionSource};
use crate::edit::commit::plan::{CommitPlan, CommitPreparation, PreparedCommit};
use crate::edit::staging::StagedEdits;
use crate::edit::test_util as edit_test_util;
use crate::edit::{SyncPolicy, WriteEngine};
use crate::error::{Error, FormatError};
use crate::image::WriteBuffering;
use crate::object_graph::ObjectId;
use crate::object_path::{LinkNameBuf, ObjectPathBuf};
use crate::{
    Datatype, DatatypeByteOrder, File, FileBuilder, FixedPointType, FloatingPointType,
    PhysicalAddress,
};

mod selection;

#[rstest]
#[case::empty(&[], 0, Some(2), vec![])]
#[case::deletion(&["type"], 0, Some(2), vec![(2, 1)])]
#[case::copy(&[], 1, Some(2), vec![(2, 3)])]
#[case::unchanged(&["type"], 1, Some(2), vec![(2, 2)])]
#[case::repeated_emission(&[], 2, Some(2), vec![(2, 4)])]
#[case::implicit_with_unavailable_domain(&[], 1, None, vec![(1, 2)])]
fn actual_prepared_inputs_determine_original_counts(
    #[case] removed: &[&str],
    #[case] emissions: usize,
    #[case] original: Option<u32>,
    #[case] expected: Vec<(u32, u32)>,
) {
    let path = test_util::temp::temp_path("actual-edits.h5");
    let target = fixture(&path, Target::from(original));
    let mut engine = edit_test_util::open_engine(&path);
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    for path in removed {
        engine.delete(&ObjectPathBuf::parse(path)).unwrap();
    }
    for index in 0..emissions {
        engine
            .copy(
                &ObjectPathBuf::parse("source"),
                &ObjectPathBuf::parse(&format!("copy{index}")),
            )
            .unwrap();
    }
    if original.is_none() {
        let staged = std::mem::take(&mut engine.staged);
        let InventoryPreparation::Rejected { staged, error } =
            PreparedEditInventory::prepare_observed(&mut engine, staged)
        else {
            panic!("expected unprotected count rejection");
        };
        assert!(
            matches!(
                error,
                Error::EditUnsupported("the original reference-count updates cannot be protected")
            ),
            "{error:?}"
        );
        assert_eq!(staged.copies().len(), emissions);
        return;
    }
    let prepared = prepare(&mut engine);
    let facts = prepared.original().facts().unwrap();
    assert_eq!(
        facts
            .counts()
            .changes()
            .iter()
            .map(|change| (
                change.object(),
                change.original().value(),
                change.remaining(),
                change.field().map(|field| field.value())
            ))
            .collect::<Vec<_>>(),
        expected
            .into_iter()
            .map(|(before, after)| (target, before, after, original))
            .collect::<Vec<_>>()
    );
    assert!(facts.eligible_owners().is_empty());
    if original.is_some() {
        assert!(prepared.storage().allocations().iter().any(|allocation| {
            matches!(allocation.origin, StorageOrigin::Owner { object, .. } if object == target)
        }));
    }
    assert_eq!(
        original_source(prepared.original()).object(),
        ObjectId::new(PhysicalAddress::new(Fixture::ROOT))
    );
}

#[rstest]
#[case::distinct_edges(false, &["type", "alias"], 1)]
#[case::aliased_group(true, &["group/type", "alias/type"], 1)]
fn actual_deletion_inputs_count_distinct_edges_or_reject_aliased_groups(
    #[case] group_alias: bool,
    #[case] removed: &[&str],
    #[case] expected: u32,
) {
    let path = test_util::temp::temp_path("actual-aliases.h5");
    let target = fixture(
        &path,
        if group_alias {
            Target::AliasedGroup
        } else {
            Target::AliasedType
        },
    );
    let mut engine = edit_test_util::open_engine(&path);
    for path in removed {
        engine.delete(&ObjectPathBuf::parse(path)).unwrap();
    }
    if group_alias {
        let staged = std::mem::take(&mut engine.staged);
        let InventoryPreparation::Rejected { error, .. } =
            PreparedEditInventory::prepare_observed(&mut engine, staged)
        else {
            panic!("expected an aliased group edit to be rejected");
        };
        assert!(
            matches!(
                error,
                Error::EditUnsupported(
                    "editing a group relocates its object header; only supported when it has a single hard link"
                )
            ),
            "{error:?}"
        );
        return;
    }
    let prepared = prepare(&mut engine);
    let changes = prepared.original().facts().unwrap().counts().changes();
    assert_eq!(changes.len(), 1);
    assert_eq!(
        (changes[0].object(), changes[0].remaining()),
        (target, expected)
    );
    let CommitPlan::Tree(tree) = &prepared.prepared.plan else {
        panic!("expected tree edits");
    };
    assert_eq!(
        tree.groups
            .values()
            .map(|node| node.prepared.deletes.len())
            .sum::<usize>(),
        2
    );
}

#[rstest]
#[case::matching_session(false)]
#[case::different_session(true)]
fn separate_prepared_edits_keep_their_original_inputs_and_successful_placements(
    #[case] different_session: bool,
) {
    let path = test_util::temp::temp_path("actual-correspondence.h5");
    let target = fixture(&path, Target::Named(2));
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("type")).unwrap();
    let first = prepare(&mut engine);
    engine
        .copy(
            &ObjectPathBuf::parse("source"),
            &ObjectPathBuf::parse("copy"),
        )
        .unwrap();
    let second = prepare(&mut engine);
    let first_source = original_source(first.original());
    let second_source = original_source(second.original());
    assert_eq!(first_source.object(), second_source.object());
    assert_eq!(
        first.original().facts().unwrap().counts().changes()[0].remaining(),
        1
    );
    assert_eq!(
        second.original().facts().unwrap().counts().changes()[0].remaining(),
        3
    );
    let original = first_source.object();
    let pointer = first
        .original()
        .facts()
        .unwrap()
        .counts()
        .changes()
        .as_ptr();
    drop(second);
    if different_session {
        let other_path = test_util::temp::temp_path("other-session.h5");
        std::fs::write(&other_path, engine.disk_bytes().unwrap()).unwrap();
        let mut other = edit_test_util::open_engine(&other_path);
        let error = first.apply(&mut other).err().unwrap();
        let Error::Format(FormatError::Internal(reason)) = error else {
            panic!("unexpected association error: {error:?}");
        };
        assert_eq!(
            reason,
            "a prepared edit inventory belongs to another allocation session"
        );
        return;
    }
    let applied = first.apply(&mut engine).unwrap();
    assert_eq!(
        applied
            .original()
            .facts()
            .unwrap()
            .counts()
            .changes()
            .as_ptr(),
        pointer
    );
    assert_eq!(
        applied.original().facts().unwrap().counts().changes()[0].object(),
        target
    );
    assert_eq!(applied.tree().relocations.len(), 1);
    let placed_root = applied.tree().relocations[&original];
    assert_ne!(placed_root, original);
    engine.publish_tree(applied.into_tree()).unwrap();
    assert_eq!(
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap(),
        placed_root
    );
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), ["source"]);
    assert_eq!(file.dataset("source").unwrap().read_f64().unwrap(), [17.0]);
}

#[rstest]
#[case::empty(false)]
#[case::deletion(true)]
fn qualified_original_inputs_stay_with_successful_application(#[case] removal: bool) {
    let path = test_util::temp::temp_path("qualified-application.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("source").with_i32_data(&[11]);
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    let (_, target) = AnalysisPhase::new(&engine)
        .topology()
        .unwrap()
        .find_link(&ObjectPathBuf::parse("source"))
        .unwrap();
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    if removal {
        engine.delete(&ObjectPathBuf::parse("source")).unwrap();
    }
    let prepared = prepare(&mut engine);
    let owners = prepared.original().owners().unwrap();
    let source = owners.source().object();
    assert_eq!(
        owners
            .facts()
            .counts()
            .changes()
            .iter()
            .map(|change| (
                change.object(),
                change.original().value(),
                change.remaining()
            ))
            .collect::<Vec<_>>(),
        if removal {
            vec![(target, 1, 0)]
        } else {
            vec![]
        }
    );
    assert_eq!(
        owners.facts().eligible_owners(),
        if removal { vec![target] } else { vec![] }
    );
    let pointer = owners.facts().counts().changes().as_ptr();
    let applied = prepared.apply(&mut engine).unwrap();
    let owners = applied.original().owners().unwrap();
    assert_eq!(owners.source().object(), source);
    assert_eq!(owners.facts().counts().changes().as_ptr(), pointer);
    assert_ne!(applied.tree().relocations[&source], source);
}

#[test]
fn incomplete_copy_inventory_preserves_rejected_staging() {
    let path = test_util::temp::temp_path("incomplete-copy.h5");
    let mut fixture = Fixture::new();
    let mut records = Form::CompactDataset.messages(
        &mut fixture,
        &Datatype::FixedPoint(FixedPointType::i32()),
        &11_i32.to_le_bytes(),
    );
    records.push(Message::new(MessageType(0xfe), &[]));
    fixture.object("source", records);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .copy(
            &ObjectPathBuf::parse("source"),
            &ObjectPathBuf::parse("copy"),
        )
        .unwrap();
    let staged = std::mem::take(&mut engine.staged);
    let before = engine.disk_bytes().unwrap();
    let InventoryPreparation::Rejected { staged, error } =
        PreparedEditInventory::prepare_observed(&mut engine, staged)
    else {
        panic!("expected incomplete copy rejection");
    };
    assert!(
        matches!(
            error,
            Error::EditUnsupported("the copy's committed-datatype users cannot be established")
        ),
        "{error:?}"
    );
    assert_eq!(
        staged.copies(),
        [(ObjectPathBuf::parse("source"), ObjectPathBuf::parse("copy"))]
    );
    assert_eq!(engine.disk_bytes().unwrap(), before);
}

#[test]
fn unresolved_deletion_inventory_keeps_independent_domain_outcome() {
    let path = test_util::temp::temp_path("unresolved-edits.h5");
    let mut fixture = Fixture::new();
    let records = Form::CompactDataset.messages(
        &mut fixture,
        &Datatype::FixedPoint(FixedPointType::i32()),
        &11_i32.to_le_bytes(),
    );
    let target = fixture.append_header(records);
    fixture.link("type", target);
    fixture.write(&path);
    let target = ObjectId::new(PhysicalAddress::new(target));
    let mut engine = edit_test_util::open_engine(&path);
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    let staged = std::mem::take(&mut engine.staged);
    let CommitPlan::Tree(mut tree) = engine.plan_commit(&staged).unwrap() else {
        panic!("expected tree edits");
    };
    tree.groups
        .get_mut(&ObjectPathBuf::root())
        .unwrap()
        .prepared
        .deletes
        .push(PlannedDeletion::unresolved(
            LinkNameBuf::new("type").unwrap(),
            target.header_address(),
        ));
    let mut phase = AnalysisPhase::new(&engine);
    let OriginalEdits::Unavailable {
        counts,
        source,
        domain,
    } = OriginalEdits::collect(&engine, &mut phase, &tree, &staged)
    else {
        panic!("expected unavailable deletion evidence");
    };
    assert!(
        matches!(counts, Err(EditInputError::IncompleteCounts)),
        "unexpected count outcome"
    );
    assert_eq!(
        source.unwrap().object(),
        ObjectId::new(PhysicalAddress::new(Fixture::ROOT))
    );
    assert!(
        matches!(domain, Some(Ok(OwnerDomain::Available(_)))),
        "unexpected domain outcome"
    );
}

#[test]
fn mixed_same_file_foreign_and_created_inputs_have_distinct_count_contributions() {
    let path = test_util::temp::temp_path("mixed-inputs.h5");
    let foreign_path = test_util::temp::temp_path("foreign-inputs.h5");
    let target = fixture(&path, Target::Named(2));
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("source")
        .with_u64_data(&[target.header_address().get()]);
    builder.write(&foreign_path).unwrap();
    let foreign = File::open(&foreign_path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .copy(
            &ObjectPathBuf::parse("source"),
            &ObjectPathBuf::parse("same"),
        )
        .unwrap();
    engine
        .copy_from(
            &foreign,
            &ObjectPathBuf::parse("source"),
            &ObjectPathBuf::parse("foreign"),
        )
        .unwrap();
    let mut values = DatasetBuilder::new("");
    values.with_u64_data(&[target.header_address().get()]);
    engine
        .stage_created_dataset(&ObjectPathBuf::parse("created"), values)
        .unwrap();
    drop(foreign);
    let prepared = prepare(&mut engine);
    let facts = prepared.original().facts().unwrap();
    assert_eq!(
        facts
            .counts()
            .changes()
            .iter()
            .map(|change| (change.object(), change.remaining()))
            .collect::<Vec<_>>(),
        [(target, 3)]
    );
    let applied = prepared.apply(&mut engine).unwrap();
    assert_eq!(
        applied.original().facts().unwrap().counts().changes()[0].remaining(),
        3
    );
    engine.publish_tree(applied.into_tree()).unwrap();
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.root().datasets().unwrap(),
        ["source", "same", "foreign", "created"]
    );
    assert_eq!(
        file.dataset("foreign").unwrap().read_u64().unwrap(),
        [target.header_address().get()]
    );
    assert_eq!(
        file.dataset("created").unwrap().read_u64().unwrap(),
        [target.header_address().get()]
    );
    assert_eq!(file.dataset("same").unwrap().read_f64().unwrap(), [17.0]);
}

#[test]
fn copy_application_failure_preserves_source_bytes_through_rollback() {
    let path = test_util::temp::temp_path("failed-application.h5");
    fixture(&path, Target::Named(2));
    let mut engine = edit_test_util::open_engine(&path);
    let (_, source) = AnalysisPhase::new(&engine)
        .topology()
        .unwrap()
        .find_link(&ObjectPathBuf::parse("source"))
        .unwrap();
    let original = WriteEngine::gather_oh_messages(
        &engine.image(),
        source.header_address(),
        engine.addresses(),
        engine.widths(),
    )
    .unwrap();
    engine.set_sync_policy(SyncPolicy::Always);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine
        .copy(
            &ObjectPathBuf::parse("source"),
            &ObjectPathBuf::parse("first"),
        )
        .unwrap();
    engine
        .copy(
            &ObjectPathBuf::parse("source"),
            &ObjectPathBuf::parse("second"),
        )
        .unwrap();
    let disk = engine.disk_bytes().unwrap();
    let snapshot = engine.snapshot_free();
    let root = engine.superblock().root_group_address;
    let prepared = prepare(&mut engine);
    disk_log::start();
    disk_log::fail_write_after(1, 0);
    let error = prepared.apply(&mut engine).err().unwrap();
    disk_log::clear_failures();
    let writes = disk_log::take();
    let [DiskOp::Write { offset, .. }] = writes.as_slice() else {
        panic!("expected one complete emitted header: {writes:?}");
    };
    assert!(*offset >= u64::try_from(disk.len()).unwrap());
    let emitted = WriteEngine::gather_oh_messages(
        &engine.image(),
        PhysicalAddress::new(*offset),
        engine.addresses(),
        engine.widths(),
    )
    .unwrap();
    assert_eq!(emitted.to_vec(), original.to_vec());
    assert!(!engine.publication.attempted());
    let error = engine.rollback_commit(snapshot, StagedEdits::default(), error);
    let Error::Io(error) = error else {
        panic!("unexpected application error: {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    assert_eq!(engine.superblock().root_group_address, root);
    assert!(!engine.has_staged_edits());
    assert_eq!(
        engine.disk_bytes().unwrap().get(..disk.len()),
        Some(disk.as_slice())
    );
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), ["source"]);
    assert_eq!(file.dataset("source").unwrap().read_f64().unwrap(), [17.0]);
}

#[test]
fn ordinary_numeric_preparation_keeps_domain_and_count_discovery_lazy() {
    let path = test_util::temp::temp_path("lazy-inventory.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("source").with_i32_data(&[11]);
    builder.write(&path).unwrap();
    let (mut engine, calls) = edit_test_util::open_counted_engine(&path);
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    let staged = std::mem::take(&mut engine.staged);
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    calls.store(0, Ordering::Relaxed);
    let CommitPreparation::Prepared(prepared) = PreparedCommit::prepare(&mut engine, staged) else {
        panic!("expected ordinary preparation");
    };
    assert_eq!(DISCOVERY_WORK.with(|work| work.get()), (0, 0));
    let baseline = calls.load(Ordering::Relaxed);
    assert!(baseline > 0);
    drop(prepared);
    engine.create_group(&ObjectPathBuf::parse("new")).unwrap();
    calls.store(0, Ordering::Relaxed);
    let checked = prepare(&mut engine);
    let owners = checked.original().owners().unwrap();
    assert!(owners.facts().counts().changes().is_empty());
    assert!(owners.facts().eligible_owners().is_empty());
    assert_eq!(DISCOVERY_WORK.with(|work| work.get()), (1, 2));
    assert!(calls.load(Ordering::Relaxed) > baseline);
}

fn original_source(original: &OriginalEdits) -> &ReconstructionSource {
    match original {
        OriginalEdits::Complete(owners) => owners.source(),
        OriginalEdits::Unavailable { source, .. } => source.as_ref().unwrap(),
        OriginalEdits::Association(error) => panic!("unexpected association error: {error:?}"),
    }
}

/// Begins a commit and prepares its staged inputs as a tree inventory.
///
/// # Panics
///
/// Panics if preparation rejects the staged set or returns an in-place plan.
pub(super) fn prepare(engine: &mut WriteEngine) -> PreparedEditInventory {
    let staged = std::mem::take(&mut engine.staged);
    engine.begin_commit(&staged);
    match PreparedEditInventory::prepare_observed(engine, staged) {
        InventoryPreparation::Tree(prepared) => prepared,
        InventoryPreparation::Rejected { error, .. } => panic!("preparation failed: {error:?}"),
        InventoryPreparation::Direct(_) => panic!("expected a tree edit"),
    }
}

/// Writes a dataset using a committed datatype and returns the datatype's physical identity.
///
/// The constructor uses `description` to choose the datatype's count record and hard-link
/// arrangement.
///
/// # Panics
///
/// Panics if the fixture cannot be written.
pub(super) fn fixture(path: &Path, description: Target) -> ObjectId {
    let mut fixture = Fixture::new();
    let mut records = vec![Message::new(MessageType::DATATYPE, &datatype::f64_le())];
    records.extend(description.count().map(Message::reference_count));
    let target = fixture.append_header(records);
    match description {
        Target::Implicit => {}
        Target::AliasedGroup => {
            let group = fixture.append_header(vec![
                test_util_hdf5::global_heap_fixture::hard_link("type", target),
                Message::reference_count(2),
            ]);
            fixture.link("group", group);
            fixture.link("alias", group);
        }
        Target::AliasedType | Target::Named(_) => {
            fixture.link("type", target);
            if matches!(description, Target::AliasedType) {
                fixture.link("alias", target);
            }
        }
    }
    let mut records = Form::CompactDataset.messages(
        &mut fixture,
        &Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(
            DatatypeByteOrder::LittleEndian,
        )),
        &17_f64.to_le_bytes(),
    );
    let message = records
        .iter_mut()
        .find(|message| message.msg_type == MessageType::DATATYPE)
        .unwrap();
    *message = Message::new(
        MessageType::DATATYPE,
        &datatype::committed_reference(target),
    );
    message.flags = MessageFlags::SHARED;
    fixture.object("source", records);
    fixture.write(path);
    ObjectId::new(PhysicalAddress::new(target))
}

/// The count record and hard-link arrangement of a committed-datatype fixture.
pub(super) enum Target {
    /// A type in a group with two aliases, with stored count two.
    AliasedGroup,
    /// A type with two direct aliases and stored count three.
    AliasedType,
    /// A type referenced only by the dataset, without an explicit count record.
    Implicit,
    /// A named type with the supplied stored count.
    Named(u32),
}

impl Target {
    fn count(&self) -> Option<u32> {
        match self {
            Self::AliasedGroup => Some(2),
            Self::AliasedType => Some(3),
            Self::Implicit => None,
            Self::Named(count) => Some(*count),
        }
    }
}

impl From<Option<u32>> for Target {
    fn from(count: Option<u32>) -> Self {
        count.map_or(Self::Implicit, Self::Named)
    }
}
