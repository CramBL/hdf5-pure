use rstest::rstest;
use test_util::global_heap::MIN_COLLECTION_SIZE;

use super::super::{
    StorageApplyError, StorageApplyFailure, StorageConflict, StorageIntegrityError,
};
use crate::File;
use crate::PhysicalAddress;
use crate::edit::analysis::heap::qualification::tests::envelopes as envelope_fixture;
use crate::edit::analysis::heap::qualification::tests::envelopes::Damage;
use crate::edit::analysis::physical::claims::ClaimOrigin;
use crate::edit::commit::plan::inventory::members::ParentDisposition;
use crate::edit::test_util as edit_test_util;
use crate::error::Error;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::records(Damage::Records)]
#[case::version(Damage::Version)]
#[case::unknown(Damage::Unknown)]
fn collection_envelope_publication_retains_damaged_source_bytes(#[case] damage: Damage) {
    let path = test_util::temp::temp_path("collection-envelope-publication.h5");
    let (healthy, damaged, _, opaque) = envelope_fixture::fixture(&path, damage);
    let original = std::fs::read(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    let snapshot = engine.snapshot_free();
    engine.delete(&ObjectPathBuf::parse("gone")).unwrap();
    let prepared = super::prepare(&mut engine);
    let completion = prepared
        .apply_excluding(&mut engine)
        .ok()
        .unwrap()
        .publish_compacting_members()
        .ok()
        .unwrap();
    completion.mutation.as_ref().unwrap();
    match (
        &completion.decisions.dispositions()[&PhysicalAddress::new(healthy)],
        damage,
    ) {
        (ParentDisposition::Retire(_), Damage::Records)
        | (ParentDisposition::Retain, Damage::Version | Damage::Unknown) => {}
        _ => panic!("unexpected healthy parent disposition"),
    }
    engine
        .finish_commit(snapshot, completion.into_outcome())
        .unwrap();
    drop(engine);
    let after = std::fs::read(&path).unwrap();
    let start = usize::try_from(damaged).unwrap();
    let end = start.checked_add(MIN_COLLECTION_SIZE).unwrap();
    assert_eq!(&after[start..end], &original[start..end]);
    if !matches!(damage, Damage::Records) {
        let start = usize::try_from(healthy).unwrap();
        let end = start.checked_add(MIN_COLLECTION_SIZE).unwrap();
        assert_eq!(&after[start..end], &original[start..end]);
    }
    if let Some(opaque) = opaque {
        let start = usize::try_from(opaque.start().get()).unwrap();
        let end = usize::try_from(opaque.end().get()).unwrap();
        assert_eq!(&original[start..end], &envelope_fixture::OPAQUE_BYTES);
        assert_eq!(&after[start..end], &original[start..end]);
    }
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().datasets().unwrap(), ["bad", "keep"]);
    assert_eq!(file.dataset("keep").unwrap().read_i32().unwrap(), [17]);
    let error = file.dataset("bad").unwrap().read_string().unwrap_err();
    let Error::Format(error) = error else {
        panic!("unexpected damaged collection error: {error:?}");
    };
    assert_eq!(
        error,
        damage.expected_error(damaged, u64::try_from(after.len()).unwrap())
    );
}

#[rstest]
#[case::overlap(Damage::Overlap)]
#[case::source_bounds(Damage::BeyondSource)]
fn collection_envelope_conflicts_reject_placement(#[case] damage: Damage) {
    let path = test_util::temp::temp_path("collection-envelope-conflict.h5");
    let (healthy, damaged, _, _) = envelope_fixture::fixture(&path, damage);
    let original = std::fs::read(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("gone")).unwrap();
    let prepared = super::prepare(&mut engine);
    let failure = prepared.apply_excluding(&mut engine).err().unwrap();
    let StorageApplyFailure::BeforePlacement {
        error: StorageApplyError::Conflict(StorageConflict::Original(error)),
        ..
    } = failure
    else {
        panic!("expected original source conflict");
    };
    match (damage, error) {
        (Damage::Overlap, StorageIntegrityError::ConflictingClaims { first, second }) => {
            assert_eq!(
                first.origin(),
                ClaimOrigin::Heap(PhysicalAddress::new(damaged))
            );
            assert_eq!(
                second.origin(),
                ClaimOrigin::Heap(PhysicalAddress::new(healthy))
            );
            assert!(first.range().overlaps(second.range()));
        }
        (Damage::BeyondSource, StorageIntegrityError::BeyondSource { claim, end }) => {
            assert_eq!(
                claim.origin(),
                ClaimOrigin::Heap(PhysicalAddress::new(damaged))
            );
            assert_eq!(end.get(), u64::try_from(original.len()).unwrap());
            assert!(claim.range().end() > end);
        }
        (_, error) => panic!("unexpected geometry error: {error:?}"),
    }
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), original);
}

test_util_hdf5::hdf5_1_10_tests! {
    mod native_crosschecks;
}
