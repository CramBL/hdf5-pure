use rstest::rstest;
use test_util::global_heap;
use test_util::widths::Widths;
use test_util_hdf5::__private::heap_members;
use test_util_hdf5::file;
use test_util_hdf5::global_heap::attributes;

use crate::edit::commit::heap;
use crate::edit::commit::plan::inventory::{InventoryPreparation, PreparedEditInventory};
use crate::edit::test_util as edit_test_util;
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
    let staged = std::mem::take(&mut engine.staged);
    let InventoryPreparation::Tree(prepared) = PreparedEditInventory::prepare(&mut engine, staged)
    else {
        panic!("expected checked tree preparation");
    };
    prepared
        .apply_excluding(&mut engine)
        .ok()
        .unwrap()
        .publish_with_completion(|_| Ok(()))
        .ok()
        .unwrap();
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
#[case::unbuffered(false)]
#[case::session(true)]
fn native_source_members_compact_through_checked_publication(#[case] session: bool) {
    let path = test_util::temp::temp_path("native-checked-heap-compaction.h5");
    assert!(heap_members::write(&path, false).is_none());
    let original = std::fs::read(&path).unwrap();
    let ranges = global_heap::collection_ranges(&original, Widths::EIGHT);
    assert_eq!(ranges.len(), 1);
    let range = usize::try_from(ranges[0].start).unwrap()..usize::try_from(ranges[0].end).unwrap();
    let parent = &original[range.clone()];
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
    let staged = std::mem::take(&mut engine.staged);
    let InventoryPreparation::Tree(prepared) = PreparedEditInventory::prepare(&mut engine, staged)
    else {
        panic!("expected checked tree preparation");
    };
    let pending = prepared.apply_excluding(&mut engine).ok().unwrap();
    assert!(pending.prospective().is_complete());
    assert_eq!(pending.member_decisions().plans().count(), 1);
    pending.publish_compacting_members().unwrap().unwrap();
    drop(engine);

    let compacted = std::fs::read(&path).unwrap();
    let compacted = &compacted[range];
    const HEADER_LEN: usize = 16;
    const RECORD_LEN: usize = 24;
    let free = HEADER_LEN + RECORD_LEN;
    assert_eq!(&compacted[..HEADER_LEN], &parent[..HEADER_LEN]);
    assert_eq!(
        &compacted[HEADER_LEN..free],
        &parent[free..free + RECORD_LEN]
    );
    assert_eq!(&compacted[free..free + 8], &[0; 8]);
    assert_eq!(
        &compacted[free + 8..free + HEADER_LEN],
        &u64::try_from(compacted.len() - free).unwrap().to_le_bytes()
    );

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
