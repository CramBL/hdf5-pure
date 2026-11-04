use core::cell::Cell;
use core::num::NonZeroU64;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use hdf5_pure_format::__private::ObjectHeaderPrefix;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::attribute;
use test_util::dataspace;
use test_util::datatype;
use test_util::object_header::v2::{self, HeaderFlags};
use test_util::object_header::{Message, MessageType};
use test_util::temp::{self, TempPath};

use crate::edit::WriteEngine;
use crate::edit::analysis::{AnalysisPhase, DISCOVERY_WORK};
use crate::edit::attributes::DenseAttrSet;
use crate::edit::copy::{CopyNode, CopyTree, CopyTreeBuilder};
use crate::edit::group;
use crate::edit::object_header::OhRegion;
use crate::error::FormatError;
use crate::image::{CountingImage, HandleImage};
use crate::object_graph::ObjectId;
use crate::source::MetadataCacheConfig;
use crate::{FileBuilder, FileLocking, PhysicalAddress, UserblockSize};

use super::{AddedCopyUsers, CopyUsersError, CopyUsersUnion};

#[rstest]
#[case::absolute(0)]
#[case::userblock(512)]
fn added_users_count_actual_emissions_in_a_lazy_source_phase(#[case] base: u64) {
    let (_path, engine, calls, bytes) = source(base);
    let first = tree(&[1_000_000, 1_000_000]);
    let second = tree(&[2_000_000]);
    let mut phase = AnalysisPhase::new(&engine);
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    let first = phase.copy_users(&first).unwrap().unwrap();
    let second = phase.copy_users(&second).unwrap().unwrap();
    let added = phase
        .added_copy_users([Some(&first), Some(&first), Some(&second)])
        .unwrap()
        .unwrap();
    assert_eq!(
        counts(&added),
        vec![(base + 1_000_000, 4), (base + 2_000_000, 1)]
    );
    assert_eq!(
        added.root(),
        Some(
            ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap()
        )
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(bytes.load(Ordering::Relaxed), 0);
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (0, 0));
    assert!(phase.topology.is_none() && phase.counts.is_empty() && phase.allocations.is_empty());
    let topology = phase.topology().unwrap();
    assert!(Arc::ptr_eq(
        &added.source.as_ref().unwrap().identity,
        &topology.identity
    ));
    assert!(calls.load(Ordering::Relaxed) > 0 && bytes.load(Ordering::Relaxed) > 0);
}

#[rstest]
#[case::engine(Relation::Engine)]
#[case::phase(Relation::Phase)]
#[case::root(Relation::Root)]
fn added_users_require_matching_nonempty_phase_inputs(#[case] relation: Relation) {
    let (_path, engine, _, _) = source(0);
    let (_other_path, other, _, _) = source(512);
    let tree = tree(&[1_000_000]);
    let mut phase = AnalysisPhase::new(&engine);
    let bound = phase.copy_users(&tree).unwrap().unwrap();
    let current_root = bound.source.as_ref().unwrap().root;
    let foreign_root = ObjectId::new(PhysicalAddress::new(
        current_root.header_address().get() + 1,
    ));
    let mut wrong = match relation {
        Relation::Engine => AnalysisPhase::new(&other),
        Relation::Phase => AnalysisPhase::new(&engine),
        Relation::Root => AnalysisPhase::prospective(&engine, foreign_root.header_address()),
    };
    let Err(error) = wrong.added_copy_users([Some(&bound)]) else {
        panic!("expected a foreign count phase to be rejected")
    };
    assert_eq!(
        error,
        match relation {
            Relation::Engine => CopyUsersError::Engine,
            Relation::Phase => CopyUsersError::Phase,
            Relation::Root => CopyUsersError::Root {
                expected: foreign_root,
                actual: current_root
            },
        }
    );
    let second = wrong.copy_users(&tree).unwrap().unwrap();
    assert_eq!(
        counts(&wrong.added_copy_users([Some(&second)]).unwrap().unwrap()),
        vec![(
            if matches!(relation, Relation::Engine) {
                1_000_512
            } else {
                1_000_000
            },
            1
        )]
    );
    let Err(error) = phase.added_copy_users([Some(&bound), Some(&second)]) else {
        panic!("expected mixed count phases to be rejected")
    };
    assert_eq!(
        error,
        match relation {
            Relation::Engine => CopyUsersError::Engine,
            Relation::Phase => CopyUsersError::Phase,
            Relation::Root => CopyUsersError::Root {
                expected: current_root,
                actual: foreign_root
            },
        }
    );
    let prefix = CopyUsersUnion::new()
        .add(&phase, Some(&bound))
        .unwrap()
        .unwrap();
    let Err(incremental) = prefix.add(&wrong, Some(&second)) else {
        panic!("expected an incremental union to reject a changed phase")
    };
    assert_eq!(incremental, error);
    assert!(phase.topology.is_none() && phase.counts.is_empty());
}

#[test]
fn incomplete_copies_end_union_construction() {
    let (_path, engine, calls, bytes) = source(0);
    let known = tree(&[1_000_000]);
    let unknown = CopyTreeBuilder::default().finish(CopyNode::DatasetVerbatim {
        region: OhRegion::new(
            crate::file_writer::WIDTHS,
            v2::message_records(
                &[Message::new(MessageType(0xfe), &[])],
                HeaderFlags::default(),
            ),
            ObjectHeaderPrefix::PLAIN,
        ),
        dense_attrs: DenseAttrSet::default(),
    });
    let mut phase = AnalysisPhase::new(&engine);
    assert!(phase.copy_users(&unknown).unwrap().is_none());
    assert!(phase.identity.is_none());
    let known = phase.copy_users(&known).unwrap().unwrap();
    assert!(
        phase
            .added_copy_users([Some(&known), None])
            .unwrap()
            .is_none()
    );
    assert_eq!(
        counts(&phase.added_copy_users([Some(&known)]).unwrap().unwrap()),
        vec![(1_000_000, 1)]
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(bytes.load(Ordering::Relaxed), 0);
    assert!(phase.counts.is_empty() && phase.topology.is_none());
}

#[test]
fn checked_target_conversion_preserves_the_phase_after_a_failed_prefix() {
    let (_path, engine, calls, bytes) = source(512);
    let valid = tree(&[1_000_000]);
    let overflow = tree(&[u64::MAX - 1]);
    let mut phase = AnalysisPhase::new(&engine);
    let valid = phase.copy_users(&valid).unwrap().unwrap();
    let overflow = phase.copy_users(&overflow).unwrap().unwrap();
    let Err(error) = phase.added_copy_users([Some(&valid), Some(&overflow)]) else {
        panic!("expected overflowing physical coordinates to be rejected")
    };
    assert_eq!(
        error,
        CopyUsersError::Address(FormatError::OffsetOverflow {
            offset: u64::MAX - 1,
            length: 512
        })
    );
    assert_eq!(
        counts(&phase.added_copy_users([Some(&valid)]).unwrap().unwrap()),
        vec![(1_000_512, 1)]
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(bytes.load(Ordering::Relaxed), 0);
}

#[test]
fn added_user_overflow_keeps_the_original_multiplicity() {
    let object = ObjectId::new(PhysicalAddress::new(900));
    let mut users = BTreeMap::from([(object, NonZeroU64::MAX)]);
    assert_eq!(
        super::add_users(&mut users, object, NonZeroU64::MIN).unwrap_err(),
        CopyUsersError::Overflow {
            object,
            current: NonZeroU64::MAX,
            added: NonZeroU64::MIN
        }
    );
    assert_eq!(users, BTreeMap::from([(object, NonZeroU64::MAX)]));
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn added_user_lookup_storage_starts_with_a_nonempty_copy() {
    let (_path, engine, calls, bytes) = source(0);
    let empty = tree(&[]);
    let nonempty = tree(&[1_000_000]);
    let mut phase = AnalysisPhase::new(&engine);
    let (result, measured) = allocation::measure("empty_added_copy_users", || {
        assert_eq!(phase.added_copy_users([]).unwrap().unwrap().root(), None);
        let bound = phase.copy_users(&empty).unwrap().unwrap();
        phase.added_copy_users([Some(&bound)]).unwrap().unwrap()
    });
    assert_eq!(counts(&result), Vec::new());
    assert_eq!(result.root(), None);
    assert!(phase.identity.is_none());
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(measured.bytes, 0, "{measured}");
    let (control, positive) = allocation::measure("nonempty_added_copy_users", || {
        let bound = phase.copy_users(&nonempty).unwrap().unwrap();
        phase
            .added_copy_users([Some(&bound), Some(&bound)])
            .unwrap()
            .unwrap()
    });
    assert_eq!(counts(&control), vec![(1_000_000, 2)]);
    assert!(
        positive.blocks > 0 && positive.bytes > 0 && positive.bytes <= 2_048,
        "{positive}"
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(bytes.load(Ordering::Relaxed), 0);
    assert!(phase.topology.is_none() && phase.counts.is_empty());
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::one(1)]
#[case::many(128)]
fn incremental_users_retain_one_union_and_the_current_summary(#[case] copies: u64) {
    let (_path, engine, calls, bytes) = source(0);
    let trees = (0..copies).map(|_| tree(&[1_000_000])).collect::<Vec<_>>();
    let mut phase = AnalysisPhase::new(&engine);
    let (added, measured) =
        allocation::measure(&format!("incremental_copy_users_{copies}"), || {
            let mut union = CopyUsersUnion::new();
            for tree in &trees {
                let bound = phase.copy_users(tree).unwrap().unwrap();
                union = union.add(&phase, Some(&bound)).unwrap().unwrap();
            }
            union.finish()
        });
    drop(trees);
    assert_eq!(counts(&added), vec![(1_000_000, copies)]);
    assert!(
        measured.blocks > 0 && measured.peak_bytes <= 2_048 && measured.live_bytes <= 1_024,
        "{measured}"
    );
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(bytes.load(Ordering::Relaxed), 0);
    assert!(phase.topology.is_none() && phase.counts.is_empty());
}

#[derive(Clone, Copy)]
enum Relation {
    Engine,
    Phase,
    Root,
}

fn counts(users: &AddedCopyUsers<'_>) -> Vec<(u64, u64)> {
    users
        .users()
        .map(|(object, count)| (object.header_address().get(), count.get()))
        .collect()
}

fn tree(targets: &[u64]) -> CopyTree {
    let mut region = group::fresh_group_region(crate::file_writer::WIDTHS).unwrap();
    for (index, &target) in targets.iter().enumerate() {
        let body = attribute::Attribute::new(
            &format!("a{index}"),
            &datatype::committed_reference(target),
            &dataspace::scalar(),
            &7i32.to_le_bytes(),
        )
        .flags(attribute::Flags::SHARED_DATATYPE)
        .build();
        region.push_bytes(&v2::message_record(
            &Message::new(MessageType::ATTRIBUTE, &body),
            HeaderFlags::default(),
        ));
    }
    CopyTreeBuilder::default().finish(CopyNode::Group {
        region,
        links: Vec::new(),
        dense_attrs: DenseAttrSet::default(),
    })
}

fn source(base: u64) -> (TempPath, WriteEngine, Arc<AtomicU64>, Arc<AtomicU64>) {
    let path = temp::temp_path("added-copy-users.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(base).unwrap());
    builder.write(&path).unwrap();
    let calls = Arc::new(AtomicU64::new(0));
    let bytes = Arc::new(AtomicU64::new(0));
    let engine = WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, len| {
        Ok(Box::new(
            CountingImage::new(
                Box::new(HandleImage::new(
                    handle,
                    len,
                    MetadataCacheConfig::disabled(),
                )),
                Arc::clone(&bytes),
                Arc::default(),
            )
            .with_read_calls(Arc::clone(&calls)),
        ))
    })
    .unwrap();
    calls.store(0, Ordering::Relaxed);
    bytes.store(0, Ordering::Relaxed);
    (path, engine, calls, bytes)
}
