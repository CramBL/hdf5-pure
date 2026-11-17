use test_util::global_heap::Object;
use test_util::object_header::Message;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use crate::edit::test_util as edit_test_util;
use crate::object_path::ObjectPathBuf;

#[test]
fn detached_retained_owners_preserve_shared_heap_collections() {
    let path = test_util::temp::temp_path("retained-count-owner.h5");
    let mut fixture = Fixture::new();
    let parent = fixture.collection(&[Object::new(7, b"value")]);
    let mut kept = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        &global_heap_fixture::reference(5, parent, 7),
    );
    kept.push(Message::reference_count(2));
    fixture.object("retained", kept.clone());
    kept.pop();
    fixture.object("deleted", kept);
    fixture.write(&path);
    let before = std::fs::read(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    engine.delete(&ObjectPathBuf::parse("retained")).unwrap();
    engine.delete(&ObjectPathBuf::parse("deleted")).unwrap();
    engine.commit().unwrap();
    let start = usize::try_from(parent).unwrap();
    assert_eq!(
        engine.image().read_exact_at(parent, 4_096).unwrap(),
        before[start..start + 4_096]
    );
    assert!(
        engine
            .space
            .admitted_snapshot()
            .sections()
            .all(|(extent, _)| {
                let start = engine.addresses().resolve(extent.start()).unwrap().get();
                start + extent.len().get() <= parent || start >= parent + 4_096
            })
    );
}
