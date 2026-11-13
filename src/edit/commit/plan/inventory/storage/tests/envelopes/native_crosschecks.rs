use test_util_hdf5::file;

use crate::PhysicalAddress;
use crate::edit::analysis::heap::qualification::tests::envelopes as envelope_fixture;
use crate::edit::analysis::heap::qualification::tests::envelopes::Damage;
use crate::edit::commit::plan::inventory::members::ParentDisposition;
use crate::edit::test_util as edit_test_util;
use crate::object_path::ObjectPathBuf;

#[test]
fn collection_envelope_publication_preserves_native_readability() {
    let path = test_util::temp::temp_path("native-collection-envelope.h5");
    let (healthy, _, _, _) = envelope_fixture::fixture(&path, Damage::Records);
    let mut engine = edit_test_util::open_engine(&path);
    let snapshot = engine.snapshot_free();
    engine.delete(&ObjectPathBuf::parse("gone")).unwrap();
    let prepared = super::super::prepare(&mut engine);
    let completion = prepared
        .apply_excluding(&mut engine)
        .ok()
        .unwrap()
        .publish_compacting_members()
        .ok()
        .unwrap();
    assert!(matches!(
        completion.decisions.dispositions()[&PhysicalAddress::new(healthy)],
        ParentDisposition::Retire(_)
    ));
    completion.mutation.as_ref().unwrap();
    engine
        .finish_commit(snapshot, completion.into_outcome())
        .unwrap();
    drop(engine);
    let native = file::libhdf5_open(&path);
    assert_eq!(native.member_names().unwrap(), ["bad", "keep"]);
    assert_eq!(
        native.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
        [17]
    );
    native.close().unwrap();
}
