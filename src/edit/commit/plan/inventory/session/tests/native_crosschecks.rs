use rstest::rstest;
use test_util_hdf5::dataset;
use test_util_hdf5::file;

use crate::edit::commit::plan::inventory::members::ParentDisposition;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::flat(false)]
#[case::page(true)]
fn session_parent_retirement_preserves_native_reads_after_reopen(#[case] page: bool) {
    let path = test_util::temp::temp_path("native-session-parent.h5");
    let (mut engine, _) = super::fixture(&path, page, false);
    let parent = engine.vl_overwrite_heaps[&ObjectPathBuf::parse("labels")][0];
    super::stage(&mut engine);
    let snapshot = engine.snapshot_free();
    let prepared = super::prepare(&mut engine);
    assert_eq!(prepared.dispositions.session_parents.len(), 1);
    let pending = prepared.apply_excluding(&mut engine).ok().unwrap();
    let completion = pending.publish_compacting_members().ok().unwrap();
    completion.mutation.as_ref().unwrap();
    assert!(matches!(
        completion.decisions.dispositions()[&parent.start()],
        ParentDisposition::Retire(_)
    ));
    engine
        .finish_commit(snapshot, completion.into_outcome())
        .unwrap();
    drop(engine);
    let native = file::libhdf5_open(&path);
    assert_eq!(native.attr_names().unwrap(), ["changed"]);
    assert_eq!(
        native
            .attr("changed")
            .unwrap()
            .read_scalar::<i64>()
            .unwrap(),
        19
    );
    native.close().unwrap();
    assert_eq!(dataset::read_libhdf5_strings(&path, "labels"), ["second"]);
}
