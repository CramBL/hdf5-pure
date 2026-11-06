use rstest::rstest;
use test_util::global_heap;
use test_util::widths::Widths;
use test_util_hdf5::__private::heap_members;
use test_util_hdf5::file;
use test_util_hdf5::global_heap::attributes;

use crate::edit::commit::heap;
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
