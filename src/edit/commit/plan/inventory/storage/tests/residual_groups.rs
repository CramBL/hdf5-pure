use rstest::rstest;
use test_util::global_heap::MIN_COLLECTION_SIZE;

use crate::AttrValue;
use crate::File;
use crate::PhysicalAddress;
use crate::edit::analysis::physical::claims::domain::residual::tests as residual_fixture;
use crate::edit::analysis::physical::claims::domain::residual::tests::Extra;
use crate::edit::commit::plan::inventory::members::ParentDisposition;
use crate::edit::commit::plan::inventory::{InventoryPreparation, PreparedEditInventory};
use crate::edit::test_util as edit_test_util;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::reclaim(Extra::Keep, Edit::Delete)]
#[case::retain(Extra::Gone, Edit::Delete)]
#[case::reconstructed_child(Extra::Keep, Edit::ReplaceKeeper)]
#[case::unknown_storage(Extra::Unknown, Edit::Delete)]
fn checked_publication_accounts_for_residual_group_references(
    #[case] extra: Extra,
    #[case] edit: Edit,
) {
    let path = test_util::temp::temp_path("residual-group-publication.h5");
    let (parent, group, keeper) = residual_fixture::fixture(&path, extra);
    let original = std::fs::read(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    let snapshot = engine.snapshot_free();
    engine.delete(&ObjectPathBuf::parse("gone")).unwrap();
    if matches!(edit, Edit::ReplaceKeeper) {
        engine
            .set_dataset_attr(&ObjectPathBuf::parse("keep"), "a", AttrValue::I64(9))
            .unwrap();
    }
    let staged = std::mem::take(&mut engine.staged);
    let prepared = match PreparedEditInventory::prepare_observed(&mut engine, staged) {
        InventoryPreparation::Tree(prepared) => prepared,
        InventoryPreparation::Rejected { error, .. } => panic!("preparation failed: {error:?}"),
        InventoryPreparation::Direct(_) => panic!("expected checked tree preparation"),
    };
    assert_eq!(
        prepared.collections().residual_groups().is_some(),
        !matches!(extra, Extra::Unknown)
    );
    let pending = prepared.apply_excluding(&mut engine).ok().unwrap();
    assert!(pending.prospective().is_complete());
    assert_eq!(
        pending
            .prospective()
            .roots()
            .contains(&ObjectId::new(PhysicalAddress::new(group))),
        !matches!(extra, Extra::Unknown)
    );
    let completion = pending.publish_compacting_members().ok().unwrap();
    completion.mutation.as_ref().unwrap();
    match (
        &completion.decisions.dispositions()[&PhysicalAddress::new(parent)],
        extra,
    ) {
        (ParentDisposition::Retire(_), Extra::Keep)
        | (ParentDisposition::Retain, Extra::Gone | Extra::Unknown) => {}
        _ => panic!("unexpected parent disposition"),
    }
    engine
        .finish_commit(snapshot, completion.into_outcome())
        .unwrap();
    drop(engine);
    let after = std::fs::read(&path).unwrap();
    let start = usize::try_from(group).unwrap();
    assert_eq!(&after[start..original.len()], &original[start..]);
    if matches!(extra, Extra::Gone | Extra::Unknown) {
        let start = usize::try_from(parent).unwrap();
        let end = start.checked_add(MIN_COLLECTION_SIZE).unwrap();
        assert_eq!(&after[start..end], &original[start..end]);
    }
    let keeper = usize::try_from(keeper.start().get()).unwrap()
        ..usize::try_from(keeper.end().get()).unwrap();
    assert_eq!(&after[keeper.clone()], &original[keeper]);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), ["keep"]);
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [17]);
    if matches!(edit, Edit::ReplaceKeeper) {
        assert_eq!(
            file.dataset("keep").unwrap().attrs().unwrap()["a"],
            AttrValue::I64(9)
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum Edit {
    Delete,
    ReplaceKeeper,
}

test_util_hdf5::hdf5_1_10_tests! {
    mod native_crosschecks;
}
