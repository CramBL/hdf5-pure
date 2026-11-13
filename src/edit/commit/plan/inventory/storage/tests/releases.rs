use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::Extent;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::PageType;
use hdf5_pure_space::__private::SessionSpace;
use rstest::rstest;
use test_util::global_heap::Object;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use super::super::PendingEditInventory;
use super::super::SpaceClaim;
use super::super::StorageConflict;
use super::super::StorageOrigin;
use super::super::parents::ParentPreparationError;
use crate::AttrValue;
use crate::FileSpacePageSize;
use crate::FileSpaceStrategy;
use crate::edit::allocation::AllocationPublication;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::retirement::RetirementMergeError;
use crate::edit::retirement::RootRetirement;
use crate::edit::test_util as edit_test_util;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::reusable(Restriction::Reusable)]
#[case::session(Restriction::Session)]
#[case::source(Restriction::Source)]
#[case::original_root(Restriction::OriginalRoot)]
#[case::prospective_root(Restriction::Root)]
#[case::missing_protection(Restriction::Missing)]
#[case::short_protection(Restriction::Short)]
#[case::wrong_class(Restriction::Class)]
#[case::overlapping_retirement(Restriction::Overlap)]
#[case::conflicting_retirement_class(Restriction::RetirementClass)]
fn parent_rejection_preserves_protection_and_retirement_batch(#[case] restriction: Restriction) {
    let path = test_util::temp::temp_path("parent-release-rejection.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"removed")]);
    let mut messages =
        Form::CompactDataset.messages(&mut fixture, &crate::make_i32_type(), &11_i32.to_le_bytes());
    messages.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(7, parent, 7),
    ));
    let owner = fixture.append_header(messages);
    fixture.link("value", owner);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_dataset_attr(&ObjectPathBuf::parse("value"), "a", AttrValue::I64(9))
        .unwrap();
    let prepared = super::prepare(&mut engine);
    let applied = prepared.apply(&mut engine).unwrap();
    let mut pending = PendingEditInventory::new(&mut engine, applied);
    let parent = PhysicalAddress::new(parent);
    let range = super::range(parent.get(), 4096);
    let expected_source = pending.storage.source.clone();
    let original_root = expected_source.as_ref().unwrap().object();
    let prospective_root = pending.pending.root;
    match restriction {
        Restriction::Reusable => {
            pending.engine.aggregation = None;
            pending.engine.space = SessionSpace::from_persisted(
                Some(FileSpaceStrategy::FsmAggr),
                ByteLength::new(1),
                FileSpacePageSize::MIN,
                [(
                    None,
                    Extent::new(
                        pending.engine.addresses().relative(parent).unwrap(),
                        ByteLength::new(4096),
                    )
                    .unwrap(),
                )],
            )
            .unwrap();
        }
        Restriction::Session => pending.engine.context.allocations = AllocationPublication::new(),
        Restriction::Source => {
            let mut phase = AnalysisPhase::new(pending.engine);
            pending.storage.source = Some(
                phase
                    .reconstruction_source(pending.engine, original_root)
                    .unwrap(),
            );
        }
        Restriction::OriginalRoot => {
            let mut root = pending.engine.superblock().clone();
            root.root_group_address = pending
                .engine
                .addresses()
                .relative(PhysicalAddress::new(owner))
                .unwrap();
            pending
                .engine
                .context
                .adopt(pending.engine.context.propose(root).unwrap())
                .unwrap();
        }
        Restriction::Root => {
            pending.pending.root = PhysicalAddress::new(prospective_root.get() + 8)
        }
        Restriction::Missing => pending
            .storage
            .allocations
            .retain(|allocation| allocation.range != range),
        Restriction::Short | Restriction::Class => {
            let allocation = pending
                .storage
                .allocations
                .iter_mut()
                .find(|allocation| allocation.range == range)
                .unwrap();
            if matches!(restriction, Restriction::Short) {
                allocation.range = super::range(parent.get(), 4088);
            } else {
                allocation.origin = StorageOrigin::Parent {
                    address: parent,
                    class: Some(FreeClass::Page(PageType::Meta)),
                };
            }
        }
        Restriction::Overlap | Restriction::RetirementClass => {
            let mut entries = pending.pending.batch.entries().to_vec();
            entries.push(if matches!(restriction, Restriction::Overlap) {
                (
                    super::range(parent.get() + 8, 16),
                    FreeClass::Page(PageType::Raw),
                )
            } else {
                (range, FreeClass::Page(PageType::Meta))
            });
            pending.pending = RootRetirement::new(prospective_root, Vec::new())
                .into_pending_frees(entries, pending.engine.image.len());
        }
    }
    let allocations = pending.storage.allocations.clone();
    let entries = pending.pending.batch.entries().to_vec();
    let bytes = pending.engine.disk_bytes().unwrap();
    let error = match pending.prepare_parents() {
        Ok(_) => panic!("expected parent rejection"),
        Err(error) => error,
    };
    match (restriction, error) {
        (
            Restriction::Reusable,
            ParentPreparationError::Storage(StorageConflict::Overlap {
                allocation,
                range: actual,
                claim,
            }),
        ) => {
            assert_eq!(
                allocation,
                *allocations
                    .iter()
                    .find(|allocation| allocation.range == range)
                    .unwrap()
            );
            assert_eq!(actual, range);
            assert_eq!(claim, SpaceClaim::Reusable);
        }
        (Restriction::Session, ParentPreparationError::Session) => {}
        (Restriction::Source, ParentPreparationError::Source { expected, actual }) => {
            assert_eq!(expected, pending.storage.source);
            assert_eq!(actual, expected_source);
        }
        (Restriction::OriginalRoot, ParentPreparationError::OriginalRoot { expected, actual }) => {
            assert_eq!(expected, original_root);
            assert_eq!(actual, ObjectId::new(PhysicalAddress::new(owner)));
        }
        (Restriction::Root, ParentPreparationError::Root { expected, actual }) => {
            assert_eq!(expected, pending.pending.root);
            assert_eq!(actual, prospective_root);
        }
        (
            Restriction::Missing | Restriction::Short | Restriction::Class,
            ParentPreparationError::Protection {
                range: actual,
                class,
                actual: allocation,
            },
        ) => {
            assert_eq!(actual, range);
            assert_eq!(
                class,
                matches!(restriction, Restriction::Class).then_some(FreeClass::Page(PageType::Raw))
            );
            assert_eq!(
                allocation,
                allocations
                    .iter()
                    .find(|allocation| allocation.range.start() == parent)
                    .copied()
            );
        }
        (
            Restriction::Overlap,
            ParentPreparationError::Batch(RetirementMergeError::Overlap { first, second }),
        ) => {
            assert_eq!(first, range);
            assert_eq!(second, super::range(parent.get() + 8, 16));
        }
        (
            Restriction::RetirementClass,
            ParentPreparationError::Batch(RetirementMergeError::Class {
                range: actual,
                first,
                second,
            }),
        ) => {
            assert_eq!(actual, range);
            assert_eq!(first, FreeClass::Page(PageType::Meta));
            assert_eq!(second, FreeClass::Page(PageType::Raw));
        }
        (_, error) => panic!("unexpected rejection: {error:?}"),
    }
    assert_eq!(pending.storage.allocations, allocations);
    assert_eq!(pending.pending.batch.entries(), entries);
    assert_eq!(pending.engine.disk_bytes().unwrap(), bytes);
}

#[derive(Clone, Copy)]
enum Restriction {
    Class,
    Missing,
    OriginalRoot,
    Overlap,
    RetirementClass,
    Reusable,
    Root,
    Session,
    Short,
    Source,
}
