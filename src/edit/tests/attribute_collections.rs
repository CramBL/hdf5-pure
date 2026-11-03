use std::io::ErrorKind;

use hdf5_pure_format::__private as format;
use hdf5_pure_space::__private::AllocationClass;
use rstest::rstest;
use test_util::attribute::Attribute;
use test_util::dataspace;
use test_util::disk_log;
use test_util::global_heap;
use test_util::global_heap::Object;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use super::common;
use crate::AttrValue;
use crate::CharacterSet;
use crate::Error;
use crate::File;
use crate::FileBuilder;
use crate::FileSpaceSettings;
use crate::FileSpaceStrategy;
use crate::StoredAddress;
use crate::address::StoredAddressExt;
use crate::chunk_index_inplace::Store;
use crate::edit::WriteEngine;
use crate::edit::test_util as edit_test_util;
use crate::image::WriteBuffering;
use crate::object_path::ObjectPathBuf;
use crate::type_builders;
use crate::type_builders::DatasetBuilder;

#[rstest]
#[case::shared_attribute(Survivor::Attribute)]
#[case::shared_dataset(Survivor::Dataset)]
#[case::incomplete_references(Survivor::Unknown)]
fn attribute_retirement_preserves_shared_and_unresolved_collections(#[case] survivor: Survivor) {
    let path = test_util::temp::temp_path("retained-attribute-collection.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"before")]);
    let mut messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &crate::make_i32_type(),
        &23i32.to_le_bytes(),
    );
    messages.push(attribute("changed", collection));
    match survivor {
        Survivor::Attribute => messages.push(attribute("kept", collection)),
        Survivor::Dataset => {
            let stored = Form::ContiguousDataset.messages(
                &mut fixture,
                &global_heap_fixture::vlen_string(),
                &global_heap_fixture::reference(6, collection, 1),
            );
            fixture.object("kept", stored);
        }
        Survivor::Unknown => fixture.object("unknown", vec![Message::new(MessageType(0x7f), &[])]),
    }
    fixture.object("target", messages);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let length = global_heap::MIN_COLLECTION_SIZE;
    let before = engine.image().read_exact_at(collection, length).unwrap();
    engine
        .set_dataset_attr(
            &ObjectPathBuf::parse("target"),
            "changed",
            AttrValue::I64(19),
        )
        .unwrap();
    engine.commit().unwrap();
    assert!(
        engine
            .reusable_free_sections()
            .unwrap()
            .sections()
            .iter()
            .all(|section| {
                section.end_address().get() <= collection
                    || collection + u64::try_from(length).unwrap() <= section.address().get()
            })
    );
    assert_eq!(
        engine.image().read_exact_at(collection, length).unwrap(),
        before
    );
    let liveness = engine.global_heap_liveness().unwrap();
    assert_eq!(
        liveness.analysis_complete,
        !matches!(survivor, Survivor::Unknown)
    );
    assert_eq!(
        liveness.referenced_collections.contains(&collection),
        !matches!(survivor, Survivor::Unknown)
    );
    drop(engine);
    let file = File::open(&path).unwrap();
    let target = file.dataset("target").unwrap();
    assert_eq!(target.read_i32().unwrap(), [23]);
    assert_eq!(target.attrs().unwrap()["changed"], AttrValue::I64(19));
    if matches!(survivor, Survivor::Attribute) {
        assert_eq!(
            target.attrs().unwrap()["kept"],
            AttrValue::VarLenAsciiString("before".into())
        );
    }
}

#[test]
fn failed_attribute_publication_keeps_predecessors_until_a_fresh_successful_commit() {
    let path = test_util::temp::temp_path("failed-attribute-publication.h5");
    let mut builder = FileBuilder::new();
    builder.set_attr("label", AttrValue::VarLenAsciiString("before".into()));
    builder
        .create_dataset("guard")
        .with_u8_data(&vec![0x5a; 8_192]);
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(true)
            .with_threshold(1),
    );
    builder.write(&path).unwrap();
    let ranges = global_heap::collection_ranges(&std::fs::read(&path).unwrap(), Widths::EIGHT);
    let [collection] = ranges.as_slice() else {
        panic!("expected one original collection: {ranges:?}");
    };
    let mut engine = edit_test_util::open_engine(&path);
    let bytes = engine
        .image()
        .read_exact_at(
            collection.start,
            usize::try_from(collection.end - collection.start).unwrap(),
        )
        .unwrap();
    let extension = engine.context.superblock().superblock_extension_address;
    let root = engine.context.superblock().root_group_address;
    let mut invalid = engine.context.superblock().clone();
    invalid.superblock_extension_address = Some(StoredAddress::ZERO);
    let proposed = engine.context.propose(invalid).unwrap();
    engine.context.adopt(proposed).unwrap();
    engine
        .set_group_attr(&ObjectPathBuf::root(), "label", AttrValue::I64(19))
        .unwrap();
    common::assert_unsupported(
        engine.commit().unwrap_err(),
        "an object does not use a version 2 object header",
    );
    assert_eq!(engine.context.superblock().root_group_address, root);
    assert!(
        engine
            .reusable_free_sections()
            .unwrap()
            .sections()
            .iter()
            .all(|section| {
                section.end_address().get() <= collection.start
                    || collection.end <= section.address().get()
            })
    );
    assert_eq!(
        engine
            .image()
            .read_exact_at(collection.start, bytes.len())
            .unwrap(),
        bytes
    );
    let mut restored = engine.context.superblock().clone();
    restored.superblock_extension_address = extension;
    let proposed = engine.context.propose(restored).unwrap();
    engine.context.adopt(proposed).unwrap();
    engine
        .set_group_attr(&ObjectPathBuf::root(), "label", AttrValue::I64(19))
        .unwrap();
    engine.commit().unwrap();
    let free = engine.reusable_free_sections().unwrap();
    let retained_end = collection.end.min(engine.image.len());
    assert!(
        free.sections().iter().any(|section| {
            section.address().get() <= collection.start
                && retained_end <= section.end_address().get()
        }) || engine.image.len() <= collection.start,
        "collection {collection:?} is neither reusable in {free:?} nor beyond EOF {}",
        engine.image.len()
    );
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().attrs().unwrap()["label"], AttrValue::I64(19));
    assert_eq!(
        file.dataset("guard").unwrap().read_u8().unwrap(),
        vec![0x5a; 8_192]
    );
}

#[rstest]
#[case::content(false)]
#[case::root(true)]
fn failed_attribute_io_preserves_original_allocations(
    #[case] root: bool,
    #[values(false, true)] persistent: bool,
) {
    let path = test_util::temp::temp_path("attribute-publication-io.h5");
    let mut builder = FileBuilder::new();
    builder.set_attr("label", AttrValue::VarLenAsciiString("before".into()));
    builder
        .create_dataset("guard")
        .with_u8_data(&vec![0x5a; 8_192]);
    builder.with_file_space_settings(
        FileSpaceSettings::new(FileSpaceStrategy::FsmAggr)
            .with_persistence(persistent)
            .with_threshold(1),
    );
    builder.write(&path).unwrap();
    let ranges = global_heap::collection_ranges(&std::fs::read(&path).unwrap(), Widths::EIGHT);
    let [collection] = ranges.as_slice() else {
        panic!("expected one collection: {ranges:?}");
    };
    let mut engine = if root {
        WriteEngine::open_torn_writes(&path, 0..48).unwrap()
    } else {
        edit_test_util::open_engine(&path)
    };
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let bytes = engine
        .image()
        .read_exact_at(
            collection.start,
            usize::try_from(collection.end - collection.start).unwrap(),
        )
        .unwrap();
    engine
        .set_group_attr(&ObjectPathBuf::root(), "label", AttrValue::I64(19))
        .unwrap();
    if !root {
        disk_log::fail_write_after(0, 0);
    }
    let error = engine.commit().unwrap_err();
    disk_log::clear_failures();
    let Error::Io(error) = error else {
        panic!("expected injected I/O error: {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(
        error.to_string(),
        if root {
            "simulated device write failure"
        } else {
            "injected disk write failure"
        }
    );
    assert_eq!(engine.publication.attempted(), root);
    assert!(engine.reusable_ranges().all(|range| {
        let range = range.unwrap();
        range.end().get() <= collection.start || collection.end <= range.start().get()
    }));
    assert_eq!(
        engine
            .image()
            .read_exact_at(collection.start, bytes.len())
            .unwrap(),
        bytes
    );
    if root {
        let issued = engine.disk_bytes().unwrap();
        assert!(matches!(
            engine.commit().unwrap_err(),
            Error::AllocationPublicationUncertain
        ));
        let Err(error) = engine.store().allocate(&[7; 16], AllocationClass::Raw) else {
            panic!("an uncertain attribute publication permits another allocation");
        };
        assert!(
            matches!(error, Error::AllocationPublicationUncertain),
            "{error:?}"
        );
        assert_eq!(engine.disk_bytes().unwrap(), issued);
        drop(engine);
        assert_eq!(std::fs::read(&path).unwrap(), issued);
    } else {
        engine
            .set_group_attr(&ObjectPathBuf::root(), "label", AttrValue::I64(19))
            .unwrap();
        engine.commit().unwrap();
        let end = collection.end.min(engine.image.len());
        assert!(
            engine.image.len() <= collection.start
                || engine.reusable_ranges().any(|range| {
                    let range = range.unwrap();
                    range.start().get() <= collection.start && end <= range.end().get()
                })
        );
        drop(engine);
    }
    let file = File::open(&path).unwrap();
    assert_eq!(file.root().attrs().unwrap()["label"], AttrValue::I64(19));
    assert_eq!(
        file.dataset("guard").unwrap().read_u8().unwrap(),
        vec![0x5a; 8_192]
    );
}

#[test]
fn incomplete_attribute_liveness_preserves_exclusive_dataset_overwrite_retirement() {
    let path = test_util::temp::temp_path("attribute-exclusive-overwrite.h5");
    let mut fixture = Fixture::new();
    let attribute_collection = fixture.collection(&[Object::new(1, b"before")]);
    let strings = fixture.collection(&[Object::new(1, b"seed")]);
    let messages = Form::ContiguousDataset.messages(
        &mut fixture,
        &type_builders::make_vlen_string_type(CharacterSet::Utf8),
        &global_heap_fixture::reference(4, strings, 1),
    );
    fixture.object("labels", messages);
    fixture.object(
        "owner",
        vec![
            Message::new(
                MessageType::LINK_INFO,
                &format::LinkInfoMessage {
                    max_creation_order: None,
                    fractal_heap_address: None,
                    btree_name_index_address: None,
                    btree_creation_order_address: None,
                }
                .serialize(format::OffsetWidth::Eight),
            ),
            attribute("changed", attribute_collection),
        ],
    );
    fixture.object("unknown", vec![Message::new(MessageType(0x7f), &[])]);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let dataset = ObjectPathBuf::parse("labels");
    let mut replacement = DatasetBuilder::new("");
    replacement.with_vlen_strings(&["first"]);
    engine.stage_dataset_write(&dataset, replacement).unwrap();
    engine.commit().unwrap();
    let exclusive = engine.vl_overwrite_heaps[&dataset].clone();
    assert_eq!(exclusive.len(), 1);
    assert!(!engine.global_heap_liveness().unwrap().analysis_complete);
    let mut replacement = DatasetBuilder::new("");
    replacement.with_vlen_strings(&["second"]);
    engine.stage_dataset_write(&dataset, replacement).unwrap();
    engine
        .set_group_attr(
            &ObjectPathBuf::parse("owner"),
            "changed",
            AttrValue::I64(19),
        )
        .unwrap();
    engine.commit().unwrap();
    assert!(!engine.global_heap_liveness().unwrap().analysis_complete);
    let retired = exclusive[0];
    assert!(engine.reusable_ranges().any(|range| {
        let range = range.unwrap();
        range.start() <= retired.start() && retired.end() <= range.end()
    }));
    assert!(engine.reusable_ranges().all(|range| {
        let range = range.unwrap();
        range.end().get() <= attribute_collection
            || attribute_collection + u64::try_from(global_heap::MIN_COLLECTION_SIZE).unwrap()
                <= range.start().get()
    }));
    assert_ne!(engine.vl_overwrite_heaps[&dataset], exclusive);
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("labels").unwrap().read_string().unwrap(),
        ["second"]
    );
    assert_eq!(
        file.group("owner").unwrap().attrs().unwrap()["changed"],
        AttrValue::I64(19)
    );
}

#[derive(Clone, Copy, Debug)]
enum Survivor {
    Attribute,
    Dataset,
    Unknown,
}

fn attribute(name: &str, collection: u64) -> Message {
    Message::new(
        MessageType::ATTRIBUTE,
        &Attribute::new(
            name,
            &format::serialize_datatype(&global_heap_fixture::vlen_string()).unwrap(),
            &dataspace::scalar(),
            &global_heap_fixture::reference(6, collection, 1),
        )
        .build(),
    )
}
