use std::fs;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::PageType;
use rstest::rstest;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;
use test_util_hdf5::userblock;

use super::ObjectStorageKind;
#[cfg(feature = "deflate")]
use crate::DeflateLevel;
#[cfg(feature = "deflate")]
use crate::File;
#[cfg(feature = "deflate")]
use crate::FileBuilder;
#[cfg(feature = "deflate")]
use crate::FileSpaceSettings;
#[cfg(feature = "deflate")]
use crate::FileSpaceStrategy;
use crate::PhysicalAddress;
use crate::edit::group;
use crate::edit::test_util as edit_test_util;
use crate::file_writer;
use crate::object_graph::ObjectId;
use crate::type_builders;

#[rstest]
#[case::zero(0)]
#[case::userblock(512)]
fn one_object_retains_complete_header_allocations_and_stored_page_classes(#[case] base: usize) {
    let path = test_util::temp::temp_path("complete-object-storage.h5");
    let bytes = test_util::generic_large::generic_large();
    fs::write(&path, &bytes).unwrap();
    if base != 0 {
        userblock::prepend(&path, &vec![0xA5; base]);
    }
    let bytes = fs::read(&path).unwrap();
    let base = u64::try_from(base).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let before = engine.space.reusable_sections().collect::<Vec<_>>();
    let mut storage = Vec::new();
    assert_eq!(
        engine.collect_object_storage(id(512 + base), &mut storage),
        Some(ObjectStorageKind::Dataset)
    );
    assert_eq!(
        storage,
        [
            (range(512 + base, 268), FreeClass::Page(PageType::Meta)),
            (range(1_536 + base, 4_158), FreeClass::Dead),
            (range(1_024 + base, 512), FreeClass::Page(PageType::Raw)),
        ]
    );
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), before);
    drop(engine);
    assert_eq!(fs::read(&path).unwrap(), bytes);
}

#[test]
fn group_storage_excludes_children_and_aliases_prevent_retirement() {
    let path = test_util::temp::temp_path("group-object-storage.h5");
    let mut fixture = Fixture::new();
    let data = u64::try_from(fixture.image.as_bytes().len()).unwrap();
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &type_builders::make_i32_type(),
        &[7_i32.to_le_bytes(), 11_i32.to_le_bytes()].concat(),
    );
    let value = fixture.append_header(messages);
    let value_len = u64::try_from(fixture.image.as_bytes().len()).unwrap() - value;
    let owner = fixture.append_header(vec![
        Message::new(
            MessageType::LINK_INFO,
            &file_writer::COMPACT_LINK_INFO
                .serialize(file_writer::OFFSET_WIDTH)
                .unwrap(),
        ),
        Message::new(MessageType::GROUP_INFO, &group::GROUP_INFO_BODY),
        global_heap_fixture::hard_link("value", value),
    ]);
    let owner_len = u64::try_from(fixture.image.as_bytes().len()).unwrap() - owner;
    fixture.link("first", owner);
    fixture.link("second", owner);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let topology = engine.count_incoming_hard_links().unwrap();
    assert_eq!(topology.incoming(id(owner)), Some(2));
    assert_eq!(topology.incoming(id(value)), Some(1));
    let mut own_storage = Vec::new();
    assert_eq!(
        engine.collect_object_storage(id(owner), &mut own_storage),
        Some(ObjectStorageKind::Group)
    );
    assert_eq!(
        own_storage,
        [(range(owner, owner_len), FreeClass::Page(PageType::Meta))]
    );
    let mut child_storage = Vec::new();
    assert_eq!(
        engine.collect_object_storage(id(value), &mut child_storage),
        Some(ObjectStorageKind::Dataset)
    );
    assert_eq!(
        child_storage,
        [
            (range(value, value_len), FreeClass::Page(PageType::Meta)),
            (range(data, 8), FreeClass::Page(PageType::Raw)),
        ]
    );
    let mut retirement = Vec::new();
    topology.collect_free_spans(id(owner), 0, &mut retirement);
    assert!(retirement.is_empty());
}

#[rstest]
#[case::absent(ObjectFailure::Absent)]
#[case::continuation(ObjectFailure::Continuation)]
#[case::unsupported(ObjectFailure::Unsupported)]
fn unaccounted_objects_append_no_owned_allocations(#[case] failure: ObjectFailure) {
    let path = test_util::temp::temp_path("unsupported-object-storage.h5");
    let address = match failure {
        ObjectFailure::Absent => {
            let bytes = test_util::generic_large::generic_large();
            let address = u64::try_from(bytes.len()).unwrap();
            fs::write(&path, bytes).unwrap();
            address
        }
        ObjectFailure::Continuation => {
            let mut bytes = test_util::generic_large::generic_large();
            bytes[1_536] ^= 1;
            fs::write(&path, bytes).unwrap();
            512
        }
        ObjectFailure::Unsupported => {
            let mut fixture = Fixture::new();
            let address = fixture.append_header(vec![Message::new(MessageType(0x7F), &[])]);
            fixture.link("unsupported", address);
            fixture.write(&path);
            address
        }
    };
    let engine = edit_test_util::open_engine(&path);
    let sentinel = (range(1, 1), FreeClass::Dead);
    let mut storage = vec![sentinel];
    assert_eq!(
        engine.collect_object_storage(id(address), &mut storage),
        None
    );
    assert_eq!(storage, [sentinel]);
}

#[cfg(feature = "deflate")]
#[rstest]
#[case::plain(false)]
#[case::filtered(true)]
fn chunk_storage_uses_encoded_lengths_and_keeps_index_allocations(#[case] filtered: bool) {
    let path = test_util::temp::temp_path("chunk-object-storage.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(FileSpaceStrategy::Page));
    let dataset = builder.create_dataset("d");
    dataset.with_i32_data(&[0; 256]).with_chunks(&[64]);
    if filtered {
        dataset.with_deflate(DeflateLevel::Six);
    }
    builder.write(&path).unwrap();
    let reader = File::open(&path).unwrap();
    let address = reader.dataset("d").unwrap().header_address().unwrap();
    let chunks = reader.dataset("d").unwrap().chunks().unwrap();
    assert_eq!(chunks.len(), 4);
    let encoded_bytes = chunks.iter().map(|chunk| chunk.storage_size).sum::<u64>();
    if filtered {
        assert!(encoded_bytes < 1_024);
    } else {
        assert_eq!(encoded_bytes, 1_024);
    }
    drop(reader);
    let engine = edit_test_util::open_engine(&path);
    let mut storage = Vec::new();
    assert_eq!(
        engine.collect_object_storage(id(address), &mut storage),
        Some(ObjectStorageKind::Dataset)
    );
    assert_eq!(storage.len(), 7);
    for chunk in chunks {
        assert_eq!(
            storage
                .iter()
                .filter(|&&(range, class)| {
                    range.start().get() == chunk.address
                        && range.length() == chunk.storage_size
                        && class == FreeClass::Page(PageType::Raw)
                })
                .count(),
            1
        );
    }
}

#[derive(Clone, Copy)]
enum ObjectFailure {
    Absent,
    Continuation,
    Unsupported,
}

fn id(address: u64) -> ObjectId {
    ObjectId::new(PhysicalAddress::new(address))
}

fn range(address: u64, length: u64) -> FileRange {
    FileRange::new(PhysicalAddress::new(address), length).unwrap()
}
