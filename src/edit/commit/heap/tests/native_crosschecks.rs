use core::num::NonZeroU16;

use rstest::rstest;
use test_util::global_heap;
use test_util::widths::Widths;
use test_util_hdf5::__private::heap_members;
use test_util_hdf5::file;
use test_util_hdf5::global_heap::attributes;

use crate::edit::commit::heap;
use crate::edit::commit::plan::inventory::members::ParentDisposition;
use crate::edit::commit::plan::inventory::prospective::ReferencePresence;
use crate::edit::commit::plan::inventory::{InventoryPreparation, PreparedEditInventory};
use crate::edit::test_util as edit_test_util;
use crate::global_heap::GlobalHeapId;
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;

use super::*;

#[rstest]
#[case::unbuffered(false)]
#[case::session(true)]
fn native_readers_observe_preserved_records_after_protected_compaction(#[case] session: bool) {
    let path = test_util::temp::temp_path("native-heap-compaction.h5");
    assert!(heap_members::write(&path, false).is_none());
    let original = std::fs::read(&path).unwrap();
    let ranges = global_heap::collection_ranges(&original, Widths::EIGHT);
    assert_eq!(ranges.len(), 1);
    let range = ranges[0].clone();
    let bytes =
        &original[usize::try_from(range.start).unwrap()..usize::try_from(range.end).unwrap()];
    let start = PhysicalAddress::new(range.start);
    let plan = super::compaction(start, bytes, &[1]);
    assert_eq!(plan.retained(), 1);
    assert_eq!(plan.moves().len(), 1);
    let member = plan.moves()[0].object();
    assert_eq!(member.index().get(), 2);
    assert_eq!(
        &bytes[super::local(member.payload().range(), start)],
        b"right"
    );
    let mut engine = edit_test_util::open_engine(&path);
    engine
        .remove_group_attr(&ObjectPathBuf::root(), "first")
        .unwrap();
    engine.commit().unwrap();
    drop(engine);
    let mut engine = edit_test_util::open_engine(&path);
    engine.set_sync_policy(SyncPolicy::OnClose);
    engine
        .set_write_buffering(if session {
            WriteBuffering::Session {
                page_size: 4096,
                max_bytes: 1_000_000,
            }
        } else {
            WriteBuffering::Unbuffered
        })
        .unwrap();
    heap::rootless(&mut engine, &[plan]).unwrap();
    drop(engine);
    let native = file::libhdf5_open(&path);
    assert_eq!(native.attr_names().unwrap(), ["second"]);
    assert_eq!(
        attributes::ascii_values(&native.attr("second").unwrap()),
        ["right"]
    );
    assert_eq!(
        native.dataset("data").unwrap().read_raw::<i32>().unwrap(),
        [7, 11, 13, 17]
    );
    native.close().unwrap();
}

#[rstest]
#[case::compact_unbuffered(false, false)]
#[case::compact_session(true, false)]
#[case::retire_unbuffered(false, true)]
#[case::retire_session(true, true)]
fn native_source_parent_dispositions_survive_checked_publication(
    #[case] session: bool,
    #[case] retire: bool,
) {
    let path = test_util::temp::temp_path("native-checked-heap-compaction.h5");
    assert!(heap_members::write(&path, false).is_none());
    let original = std::fs::read(&path).unwrap();
    let ranges = global_heap::collection_ranges(&original, Widths::EIGHT);
    assert_eq!(ranges.len(), 1);
    let parent = PhysicalAddress::new(ranges[0].start);
    let mut engine = edit_test_util::open_engine(&path);
    engine.set_sync_policy(SyncPolicy::OnClose);
    engine
        .set_write_buffering(if session {
            WriteBuffering::Session {
                page_size: 4096,
                max_bytes: 1_000_000,
            }
        } else {
            WriteBuffering::Unbuffered
        })
        .unwrap();
    engine
        .remove_group_attr(&ObjectPathBuf::root(), "first")
        .unwrap();
    if retire {
        engine
            .remove_group_attr(&ObjectPathBuf::root(), "second")
            .unwrap();
    }
    let snapshot = engine.snapshot_free();
    let staged = std::mem::take(&mut engine.staged);
    let InventoryPreparation::Tree(prepared) =
        PreparedEditInventory::prepare_observed(&mut engine, staged)
    else {
        panic!("expected checked tree preparation");
    };
    let pending = prepared.apply_excluding(&mut engine).ok().unwrap();
    assert!(pending.prospective().is_complete());
    assert_eq!(
        pending.prospective().member_presence(GlobalHeapId {
            collection: parent,
            index: NonZeroU16::new(1).unwrap()
        }),
        ReferencePresence::Unreferenced
    );
    assert_eq!(
        pending.prospective().member_presence(GlobalHeapId {
            collection: parent,
            index: NonZeroU16::new(2).unwrap()
        }),
        if retire {
            ReferencePresence::Unreferenced
        } else {
            ReferencePresence::ObservedLive
        }
    );
    assert_eq!(
        pending.prospective().parent_presence(parent),
        if retire {
            ReferencePresence::Unreferenced
        } else {
            ReferencePresence::ObservedLive
        }
    );
    assert_eq!(
        pending.member_decisions().physical_members(),
        if retire { 2 } else { 1 }
    );
    assert_eq!(
        pending.member_decisions().plans().count(),
        usize::from(!retire)
    );
    let completion = pending.publish_compacting_members().ok().unwrap();
    completion.mutation.as_ref().unwrap();
    if retire {
        let ParentDisposition::Retire(retirement) = &completion.decisions.dispositions()[&parent]
        else {
            panic!("expected whole-parent retirement");
        };
        assert_eq!(retirement.range().start(), parent);
        assert_eq!(Some(retirement.root()), completion.prospective.root());
        assert_eq!(completion.decisions.plans().count(), 0);
        assert_eq!(retirement.range().end().get(), ranges[0].end);
    } else {
        assert_eq!(
            completion.decisions.plans().next().unwrap().range().start(),
            parent
        );
        assert_eq!(completion.decisions.plans().next().unwrap().retained(), 1);
    }
    engine
        .finish_commit(snapshot, completion.into_outcome())
        .unwrap();
    drop(engine);
    let native = file::libhdf5_open(&path);
    if retire {
        assert_eq!(native.attr_names().unwrap(), Vec::<String>::new());
    } else {
        assert_eq!(native.attr_names().unwrap(), ["second"]);
        assert_eq!(
            attributes::ascii_values(&native.attr("second").unwrap()),
            ["right"]
        );
    }
    assert_eq!(
        native.dataset("data").unwrap().read_raw::<i32>().unwrap(),
        [7, 11, 13, 17]
    );
    native.close().unwrap();
}
