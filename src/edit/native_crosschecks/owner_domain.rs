use std::sync::Arc;
use std::sync::Mutex;

use rstest::rstest;
use test_util_hdf5::__private::heap_members;
use test_util_hdf5::__private::ownership_append;
use test_util_hdf5::attr;
use test_util_hdf5::file;
use test_util_hdf5::file_space_oracle;

use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::ownership::AllocationKind;
use crate::edit::analysis::physical::claims::ClaimOrigin;
use crate::edit::analysis::physical::claims::domain::OwnerDomain;
use crate::edit::append::{AppendBuilder, AppendTarget};
use crate::edit::test_util as edit_test_util;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;
use crate::{DatasetBuilder, File};

#[test]
fn native_persistent_owner_domain_keeps_member_qualification_separate() {
    let path = test_util::temp::temp_path("native-owner-domain-persistent.h5");
    heap_members::write_persisted(&path);
    let original = std::fs::read(&path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let root =
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap();
    let source = phase.reconstruction_source(&engine, root).unwrap();
    let OwnerDomain::Available(proof) = phase.supported_owner_domain(&source).unwrap() else {
        panic!("expected native persistent source qualification");
    };
    assert!(proof.matches(&engine, phase.physical_claims(&source).unwrap()));
    let claims = phase.physical_claims(&source).unwrap();
    assert!(claims.problems().is_empty(), "{:?}", claims.problems());
    assert_eq!(
        claims
            .claims()
            .iter()
            .filter(|claim| matches!(claim.origin(), ClaimOrigin::Heap(_)))
            .count(),
        1
    );
    assert!(
        claims
            .claims()
            .iter()
            .any(|claim| matches!(claim.origin(), ClaimOrigin::NonLive { .. }))
    );
    drop(phase);
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), original);
    heap_members::assert_values(&path, None);
}

#[test]
fn native_owner_domain_preserves_unmatched_member_facts() {
    let path = test_util::temp::temp_path("native-owner-domain-unmatched.h5");
    let token = heap_members::write(&path, true).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let root =
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap();
    let source = phase.reconstruction_source(&engine, root).unwrap();
    let OwnerDomain::Available(proof) = phase.supported_owner_domain(&source).unwrap() else {
        panic!("expected complete source domain with an independent unmatched member");
    };
    assert!(proof.matches(&engine, phase.physical_claims(&source).unwrap()));
    let associations = phase.member_associations(&source).unwrap();
    assert_eq!(associations.collections().len(), 1);
    assert_eq!(
        associations
            .collections()
            .values()
            .next()
            .unwrap()
            .unmatched()
            .len(),
        1
    );
    drop(phase);
    drop(engine);
    heap_members::assert_values(&path, Some(&token));
}

#[rstest]
#[case::hole_immediate(Mode::Immediate, false)]
#[case::eof_immediate(Mode::Immediate, true)]
#[case::existing_staged(Mode::Staged, false)]
#[case::partial_tail(Mode::PartialTail, false)]
#[case::pending_creation(Mode::Creation, false)]
fn native_append_owner_domain_excludes_only_materialized_source_storage(
    #[case] mode: Mode,
    #[case] exhaust_hole: bool,
) {
    let path = test_util::temp::temp_path("native-owner-domain-append.h5");
    ownership_append::Fixture::write(&path, false, false, false);
    let native = file::libhdf5_open_rw(&path);
    attr::write_unicode_array(&native, "labels", &["predecessor"]);
    if matches!(mode, Mode::PartialTail) {
        native.dataset("d").unwrap().resize([8_191]).unwrap();
    }
    native.close().unwrap();
    let free = file_space_oracle::observe(&path, "owner-domain", "pre-append")
        .c
        .sections()
        .to_vec();
    let reads = Arc::new(Mutex::new(Vec::new()));
    let mut engine = edit_test_util::open_observed_engine(&path, Arc::clone(&reads));
    let before = engine.image().len();
    let count = if exhaust_hole { 81_920 } else { 8_192 };
    let mut batch = AppendBuilder::new();
    batch.append_u8(&vec![0x5a; count]);
    let target = ObjectPathBuf::parse("d");
    match mode {
        Mode::Immediate => engine
            .append_inplace_gathered(AppendTarget::Path(&target), &batch, 4)
            .unwrap(),
        Mode::Staged | Mode::PartialTail => engine.stage_dataset_append(&target, batch).unwrap(),
        Mode::Creation => {
            let pending = ObjectPathBuf::parse("pending");
            let mut dataset = DatasetBuilder::new("");
            dataset.with_u8_data(&[11]);
            engine.stage_created_dataset(&pending, dataset).unwrap();
            engine
                .stage_dataset_append_pending(&pending, batch)
                .unwrap();
        }
    }
    engine
        .remove_group_attr(&ObjectPathBuf::root(), "labels")
        .unwrap();
    reads.lock().unwrap().clear();
    let mut phase = AnalysisPhase::new(&engine);
    let root =
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap();
    let source = phase.reconstruction_source(&engine, root).unwrap();
    let OwnerDomain::Available(proof) = phase.supported_owner_domain(&source).unwrap() else {
        panic!("expected supported append predecessor");
    };
    let claims = phase.physical_claims(&source).unwrap();
    assert!(proof.matches(&engine, claims));
    let payload = claims
        .claims()
        .iter()
        .filter(|claim| {
            matches!(
                claim.origin(),
                ClaimOrigin::Object {
                    kind: AllocationKind::Data,
                    ..
                }
            )
        })
        .map(|claim| claim.range())
        .collect::<Vec<_>>();
    assert_eq!(
        payload.iter().map(|range| range.length()).sum::<u64>(),
        if !matches!(mode, Mode::Immediate) {
            16_384
        } else {
            16_384 + count as u64
        }
    );
    let available = claims
        .claims()
        .iter()
        .filter(|claim| matches!(claim.origin(), ClaimOrigin::NonLive { .. }))
        .map(|claim| claim.range())
        .collect::<Vec<_>>();
    let observations = reads.lock().unwrap();
    assert!(!observations.is_empty());
    assert!(
        observations
            .iter()
            .all(|read| payload.iter().all(|allocation| !read.overlaps(*allocation))),
        "{observations:?}: {payload:?}"
    );
    assert!(
        observations
            .iter()
            .all(|read| available.iter().all(|range| !read.overlaps(*range))),
        "{observations:?}: {available:?}"
    );
    drop(observations);
    if exhaust_hole {
        assert!(engine.image().len() > before);
        assert!(payload.iter().any(|range| range.start().get() >= before));
    } else if matches!(mode, Mode::Immediate) {
        assert!(
            payload.iter().any(|range| {
                range.length() == 8_192
                    && free.iter().any(|&(start, length)| {
                        start <= range.start().get() && range.end().get() <= start + length
                    })
            }),
            "{payload:?}: {free:?}"
        );
    } else {
        assert_eq!(engine.image().len(), before);
    }
    drop(phase);
    engine.commit().unwrap();
    drop(engine);
    let pure = File::open(&path).unwrap();
    let expected = if matches!(mode, Mode::Creation) {
        vec![0; 8_192]
    } else {
        [
            vec![
                0;
                if matches!(mode, Mode::PartialTail) {
                    8_191
                } else {
                    8_192
                }
            ],
            vec![0x5a; count],
        ]
        .concat()
    };
    if matches!(mode, Mode::Creation) {
        assert_eq!(
            pure.dataset("pending").unwrap().read_u8().unwrap(),
            [vec![11], vec![0x5a; count]].concat()
        );
    }
    assert_eq!(pure.dataset("d").unwrap().read_u8().unwrap(), expected);
    pure.close().unwrap();
    let native = file::libhdf5_open(&path);
    assert_eq!(
        native.dataset("d").unwrap().read_raw::<u8>().unwrap(),
        expected
    );
    if matches!(mode, Mode::Creation) {
        assert_eq!(
            native.dataset("pending").unwrap().read_raw::<u8>().unwrap(),
            [vec![11], vec![0x5a; count]].concat()
        );
    }
    native.close().unwrap();
}

#[derive(Clone, Copy)]
enum Mode {
    Creation,
    Immediate,
    PartialTail,
    Staged,
}
