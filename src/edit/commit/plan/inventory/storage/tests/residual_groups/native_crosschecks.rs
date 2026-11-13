use rstest::rstest;
use test_util_hdf5::file;

use crate::PhysicalAddress;
use crate::edit::analysis::physical::claims::domain::residual::tests as residual_fixture;
use crate::edit::analysis::physical::claims::domain::residual::tests::Extra;
use crate::edit::commit::plan::inventory::members::ParentDisposition;
use crate::edit::test_util as edit_test_util;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::reclaim(Extra::Keep)]
#[case::retain(Extra::Gone)]
fn residual_group_publication_preserves_native_readability(#[case] extra: Extra) {
    let path = test_util::temp::temp_path("native-residual-group.h5");
    let (parent, _, _) = residual_fixture::fixture(&path, extra);
    let mut engine = edit_test_util::open_engine(&path);
    let snapshot = engine.snapshot_free();
    engine.delete(&ObjectPathBuf::parse("gone")).unwrap();
    let prepared = super::super::prepare(&mut engine);
    let pending = prepared.apply_excluding(&mut engine).ok().unwrap();
    let completion = pending.publish_compacting_members().ok().unwrap();
    match (
        &completion.decisions.dispositions()[&PhysicalAddress::new(parent)],
        extra,
    ) {
        (ParentDisposition::Retire(_), Extra::Keep) | (ParentDisposition::Retain, Extra::Gone) => {}
        _ => panic!("unexpected parent disposition"),
    }
    completion.mutation.as_ref().unwrap();
    engine
        .finish_commit(snapshot, completion.into_outcome())
        .unwrap();
    drop(engine);
    let native = file::libhdf5_open(&path);
    assert_eq!(native.member_names().unwrap(), ["keep"]);
    assert_eq!(
        native.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
        [17]
    );
    native.close().unwrap();
}
