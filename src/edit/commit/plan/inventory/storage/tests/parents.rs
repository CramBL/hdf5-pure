use core::num::NonZeroU16;

use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::PageType;
use rstest::rstest;
use test_util::global_heap;
use test_util::global_heap::Object;
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use super::super::PendingEditInventory;
use crate::AttrValue;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::commit::plan::inventory::members::ParentDisposition;
use crate::edit::commit::plan::inventory::prospective::ReferencePresence;
use crate::edit::test_util as edit_test_util;
use crate::global_heap::GlobalHeapId;
use crate::object_graph::ObjectId;
use crate::object_path::ObjectPathBuf;

#[rstest]
#[case::whole_dead(Disposition::Retire)]
#[case::unobserved_member(Disposition::CompactUnobserved)]
#[case::surviving_member(Disposition::CompactLive)]
#[case::shared_live_member(Disposition::RetainLive)]
#[case::unavailable_qualification(Disposition::RetainUnavailable)]
fn each_parent_has_one_checked_disposition(#[case] disposition: Disposition) {
    let path = test_util::temp::temp_path("exclusive-parent-disposition.h5");
    let mut fixture = Fixture::new();
    let objects = if matches!(
        disposition,
        Disposition::CompactUnobserved | Disposition::CompactLive
    ) {
        vec![
            Object::new(7, b"removed"),
            Object {
                index: 91,
                reference_count: 0,
                data: b"retained",
            },
        ]
    } else {
        vec![Object::new(7, b"removed")]
    };
    let parent = fixture.collection(&objects);
    let mut messages =
        Form::CompactDataset.messages(&mut fixture, &crate::make_i32_type(), &11_i32.to_le_bytes());
    messages.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(7, parent, 7),
    ));
    let owner = fixture.append_header(messages);
    fixture.link("value", owner);
    if matches!(
        disposition,
        Disposition::CompactLive | Disposition::RetainLive
    ) {
        let messages = Form::FillValue.messages(
            &mut fixture,
            &global_heap_fixture::vlen_string(),
            &global_heap_fixture::reference(
                if matches!(disposition, Disposition::CompactLive) {
                    8
                } else {
                    7
                },
                parent,
                if matches!(disposition, Disposition::CompactLive) {
                    91
                } else {
                    7
                },
            ),
        );
        fixture.object("keep", messages);
    }
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_dataset_attr(&ObjectPathBuf::parse("value"), "a", AttrValue::I64(9))
        .unwrap();
    let mut prepared = super::prepare(&mut engine);
    if matches!(disposition, Disposition::RetainUnavailable) {
        let mut phase = AnalysisPhase::new(&engine);
        let candidates = phase.deleted_collections(&[ObjectId::new(PhysicalAddress::new(owner))]);
        prepared.collections = phase.capture_collections(None, candidates);
    }
    let applied = prepared.apply(&mut engine).unwrap();
    let pending = PendingEditInventory::new(&mut engine, applied);
    let decisions = pending.member_decisions();
    assert_eq!(decisions.dispositions().len(), 1);
    let parent = PhysicalAddress::new(parent);
    match (&decisions.dispositions()[&parent], disposition) {
        (ParentDisposition::Retire(retirement), Disposition::Retire) => {
            assert_eq!(retirement.range().start(), parent);
            assert_eq!(retirement.range().length(), 4096);
            assert_eq!(retirement.class(), FreeClass::Page(PageType::Raw));
            assert_eq!(retirement.root(), pending.pending.root);
            assert_ne!(
                retirement.root(),
                retirement.source().object().header_address()
            );
            assert_eq!(
                Some(retirement.source()),
                pending.dispositions.original().source().as_ref().ok()
            );
            assert!(
                retirement
                    .session()
                    .same(&pending.engine.context.allocations.session())
            );
            assert_eq!(decisions.plans().count(), 0);
        }
        (
            ParentDisposition::Compact(plan),
            Disposition::CompactUnobserved | Disposition::CompactLive,
        ) => {
            assert_eq!(plan.range().start(), parent);
            assert_eq!(plan.range().length(), 4096);
            assert_eq!(plan.retained(), 1);
            assert_eq!(plan.moves().len(), 1);
            assert_eq!(
                plan.moves()[0].object().index(),
                NonZeroU16::new(91).unwrap()
            );
            assert_eq!(plan.moves()[0].object().reference_count(), 0);
            assert_eq!(decisions.plans().count(), 1);
        }
        (ParentDisposition::Retain, Disposition::RetainLive | Disposition::RetainUnavailable) => {
            assert_eq!(decisions.plans().count(), 0)
        }
        _ => panic!("unexpected parent disposition"),
    }
    if !matches!(disposition, Disposition::RetainUnavailable) {
        let id = GlobalHeapId {
            collection: parent,
            index: NonZeroU16::new(7).unwrap(),
        };
        assert_eq!(
            pending.prospective().parent_absence(parent).is_some(),
            matches!(
                disposition,
                Disposition::Retire | Disposition::CompactUnobserved
            )
        );
        if matches!(disposition, Disposition::RetainLive) {
            assert_eq!(
                pending.prospective().member_presence(id),
                ReferencePresence::ObservedLive
            );
        }
    }
}

#[rstest]
#[case::full(4064, 0)]
#[case::implied(4056, 0)]
#[case::explicit(4048, 9)]
fn whole_retirement_preserves_parent_extent_and_stored_counts(
    #[case] length: usize,
    #[case] reference_count: u16,
) {
    let path = test_util::temp::temp_path("whole-parent-geometry.h5");
    let mut fixture = Fixture::new();
    let payload = vec![b'x'; length];
    let bytes = global_heap::collection_of_size(
        &[Object {
            index: 91,
            reference_count,
            data: &payload,
        }],
        Widths::EIGHT,
        4096,
    );
    let parent = u64::try_from(fixture.image.append(&bytes)).unwrap();
    let mut messages =
        Form::CompactDataset.messages(&mut fixture, &crate::make_i32_type(), &7_i32.to_le_bytes());
    messages.extend(Form::CompactAttribute.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(u32::try_from(length).unwrap(), parent, 91),
    ));
    fixture.object("value", messages);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .set_dataset_attr(&ObjectPathBuf::parse("value"), "a", AttrValue::I64(9))
        .unwrap();
    let prepared = super::prepare(&mut engine);
    let applied = prepared.apply(&mut engine).unwrap();
    let pending = PendingEditInventory::new(&mut engine, applied);
    let decisions = pending.member_decisions();
    let ParentDisposition::Retire(retirement) =
        &decisions.dispositions()[&PhysicalAddress::new(parent)]
    else {
        panic!("expected whole-parent retirement");
    };
    assert_eq!(retirement.range().start().get(), parent);
    assert_eq!(
        retirement.range().length(),
        u64::try_from(bytes.len()).unwrap()
    );
    assert_eq!(retirement.class(), FreeClass::Page(PageType::Raw));
    assert_eq!(decisions.plans().count(), 0);
    assert_eq!(
        pending
            .engine
            .image()
            .read_exact_at(parent, bytes.len())
            .unwrap(),
        bytes
    );
    let members = crate::global_heap::collect_members(
        pending.engine.image(),
        pending.engine.widths().lengths,
        PhysicalAddress::new(parent),
    )
    .unwrap();
    assert_eq!(members.members().len(), 1);
    assert_eq!(
        (
            members.members()[0].index().get(),
            members.members()[0].reference_count()
        ),
        (91, reference_count)
    );
}

#[derive(Clone, Copy)]
enum Disposition {
    CompactLive,
    CompactUnobserved,
    RetainLive,
    RetainUnavailable,
    Retire,
}
