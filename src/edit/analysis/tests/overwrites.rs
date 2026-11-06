use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private as format;
use hdf5_pure_space::__private::{FreeClass, PageType};
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::attribute::Attribute;
use test_util::dataspace;
use test_util::global_heap;
use test_util::global_heap::Object;
use test_util::object_header::{Message, MessageType};
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::{self, Fixture, Form};

use super::super::{AnalysisPhase, CollectionContent, CollectionProvenance};
use crate::AttrValue;
use crate::CharacterSet;
use crate::File;
use crate::FileBuilder;
use crate::FillTime;
use crate::data_layout::{ChunkIndexLayout, DataLayout};
use crate::edit::WriteEngine;
use crate::edit::analysis::heap::HeapReferenceError;
use crate::edit::attributes::{self, AttrOp};
use crate::edit::test_util as edit_test_util;
use crate::error::FormatError;
use crate::global_heap::GlobalHeapId;
use crate::image::{CountingImage, HandleImage};
use crate::object_path::ObjectPathBuf;
use crate::source::MetadataCacheConfig;
use crate::type_builders::{self, DatasetBuilder};

#[rstest]
#[case::compact_raw(Layout::Compact, false)]
#[case::contiguous_raw(Layout::Contiguous, false)]
#[case::chunked_raw(Layout::Chunked, false)]
#[case::compact_deferred(Layout::Compact, true)]
#[case::contiguous_deferred(Layout::Contiguous, true)]
#[case::chunked_deferred(Layout::Chunked, true)]
fn overwrite_candidates_preserve_original_values_before_either_commit_route(
    #[case] layout: Layout,
    #[case] deferred: bool,
    #[values(false, true)] values_first: bool,
) {
    let path = test_util::temp::temp_path("overwrite-candidates.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[
        Object::new(7, b"before"),
        Object::new(11, b"other"),
        Object::new(13, b"fill"),
    ]);
    let mut messages = layout.messages(
        &mut fixture,
        &global_heap_fixture::reference(6, collection, 7),
    );
    messages.push(Message::new(
        MessageType::ATTRIBUTE,
        &Attribute::new(
            "unchanged",
            &format::serialize_datatype(&global_heap_fixture::vlen_string()).unwrap(),
            &dataspace::scalar(),
            &global_heap_fixture::reference(5, collection, 11),
        )
        .build(),
    ));
    messages.push(Message::new(
        MessageType::FILL_VALUE,
        &format::serialize_fill_value_message_v3(
            FillTime::IfSet,
            Some(&global_heap_fixture::reference(4, collection, 13)),
        )
        .unwrap(),
    ));
    let object = fixture.append_header(messages);
    fixture.link("d", object);
    fixture.write(&path);
    let original = std::fs::read(&path).unwrap();
    {
        let file = File::open(&path).unwrap();
        let dataset = file.dataset("d").unwrap();
        let stored = dataset.data_layout().unwrap();
        match (layout, stored) {
            (Layout::Compact, DataLayout::Compact { data }) => {
                assert_eq!(data, global_heap_fixture::reference(6, collection, 7));
            }
            (Layout::Contiguous, DataLayout::Contiguous { address, size }) => {
                assert!(address.is_some());
                assert_eq!(size, 16);
            }
            (
                Layout::Chunked,
                DataLayout::Chunked {
                    chunk_dimensions,
                    index,
                    ..
                },
            ) => {
                assert_eq!(chunk_dimensions, [1, 16]);
                let ChunkIndexLayout::SingleChunk { filtered, address } = index else {
                    panic!("expected a single-chunk index, got {index:?}");
                };
                assert!(filtered.is_none());
                assert!(address.is_some());
            }
            (_, stored) => panic!("unexpected original layout: {stored:?}"),
        }
        assert_eq!(
            dataset.read_raw().unwrap(),
            global_heap_fixture::reference(6, collection, 7)
        );
    }
    let mut engine = edit_test_util::open_engine(&path);
    let mut replacement = DatasetBuilder::new("");
    if deferred {
        replacement.with_vlen_strings(&["after"]);
    } else {
        replacement.with_raw_data(
            type_builders::make_vlen_string_type(CharacterSet::Utf8),
            global_heap_fixture::reference(5, collection, 11),
            1,
        );
    }
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("d"), replacement)
        .unwrap();
    let mut phase = AnalysisPhase::new(&engine);
    crate::edit::global_heap_liveness::DATASET_VALUE_WORK.with(|work| work.set(0));
    if !values_first {
        phase
            .read_reference_facts(super::object(object), false)
            .unwrap();
    }
    let mut candidates = phase.overwrite_collections(&engine.staged).unwrap();
    let origin = CollectionProvenance {
        object: super::object(object),
        content: CollectionContent::DatasetValues,
    };
    let expected = BTreeSet::from([GlobalHeapId {
        collection: PhysicalAddress::new(collection),
        index: 7.try_into().unwrap(),
    }]);
    assert!(Arc::ptr_eq(
        candidates
            .observations
            .as_ref()
            .unwrap()
            .identity
            .as_ref()
            .unwrap(),
        phase.identity.as_ref().unwrap(),
    ));
    let facts = &candidates.observations.as_ref().unwrap().predecessors[&origin];
    assert_eq!(facts.objects, expected);
    assert!(facts.complete);
    assert!(facts.error.is_none());
    assert_eq!(candidates.collections.len(), 1);
    let allocation = &candidates.collections[&PhysicalAddress::new(collection)];
    assert_eq!(allocation.range.start(), PhysicalAddress::new(collection));
    assert_eq!(
        allocation.range.length(),
        u64::try_from(global_heap::MIN_COLLECTION_SIZE).unwrap(),
    );
    assert_eq!(allocation.class, Some(FreeClass::Page(PageType::Raw)));
    assert_eq!(allocation.origins, BTreeSet::from([origin.clone()]));
    candidates
        .extend(phase.overwrite_collections(&engine.staged).unwrap())
        .unwrap();
    assert_eq!(phase.allocations.len(), 1);
    assert_eq!(phase.value_references.len(), 1);
    let all = phase
        .read_reference_facts(super::object(object), false)
        .unwrap();
    assert_eq!(
        all.objects,
        BTreeSet::from([
            GlobalHeapId {
                collection: PhysicalAddress::new(collection),
                index: 7.try_into().unwrap()
            },
            GlobalHeapId {
                collection: PhysicalAddress::new(collection),
                index: 11.try_into().unwrap()
            },
            GlobalHeapId {
                collection: PhysicalAddress::new(collection),
                index: 13.try_into().unwrap(),
            },
        ])
    );
    assert!(all.complete);
    assert!(all.error.is_none());
    assert_eq!(
        crate::edit::global_heap_liveness::DATASET_VALUE_WORK.with(|work| work.get()),
        1
    );
    assert!(phase.topology.is_none());
    assert!(phase.liveness.is_none());
    assert_eq!(
        candidates.observations.as_ref().unwrap().predecessors.len(),
        1
    );
    assert_eq!(
        candidates.observations.as_ref().unwrap().predecessors[&origin].objects,
        expected
    );
    candidates
        .extend(phase.deleted_collections(&[super::object(object)]))
        .unwrap();
    let observations = candidates.observations.as_ref().unwrap();
    assert_eq!(observations.predecessors.len(), 2);
    assert_eq!(observations.predecessors[&origin].objects, expected);
    assert_eq!(
        observations.predecessors[&CollectionProvenance {
            object: super::object(object),
            content: CollectionContent::Object,
        }]
            .objects
            .len(),
        3
    );
    assert_eq!(candidates.collections.len(), 1);
    assert_eq!(
        candidates.collections[&PhysicalAddress::new(collection)]
            .origins
            .len(),
        2
    );
    assert_eq!(
        crate::edit::global_heap_liveness::DATASET_VALUE_WORK.with(|work| work.get()),
        1
    );
    let region = WriteEngine::gather_oh_messages(
        engine.image(),
        PhysicalAddress::new(object),
        engine.addresses(),
        engine.widths(),
    )
    .unwrap();
    let operations = [AttrOp::Remove {
        name: "unchanged".into(),
    }];
    let edited = attributes::plan_attr_ops(
        &engine.opened_source(),
        &region,
        &operations.iter().collect::<Vec<_>>(),
    )
    .unwrap();
    candidates
        .extend(
            phase
                .attribute_collections(super::object(object), edited.predecessors)
                .unwrap(),
        )
        .unwrap();
    let observations = candidates.observations.as_ref().unwrap();
    assert_eq!(observations.predecessors.len(), 3);
    assert!(Arc::ptr_eq(
        observations.identity.as_ref().unwrap(),
        phase.identity.as_ref().unwrap()
    ));
    assert_eq!(
        observations.predecessors[&CollectionProvenance {
            object: super::object(object),
            content: CollectionContent::Attribute("unchanged".into()),
        }]
            .objects,
        BTreeSet::from([GlobalHeapId {
            collection: PhysicalAddress::new(collection),
            index: 11.try_into().unwrap(),
        }])
    );
    assert_eq!(
        candidates.collections[&PhysicalAddress::new(collection)]
            .origins
            .len(),
        3
    );
    drop(phase);
    assert_eq!(engine.image.disk_bytes().unwrap(), original);
    assert!(engine.vl_overwrite_heaps.is_empty());
    let root = engine.superblock().root_group_address;
    engine.commit().unwrap();
    assert_eq!(
        engine.superblock().root_group_address == root,
        !matches!(layout, Layout::Compact) && !deferred,
    );
    assert_eq!(engine.vl_overwrite_heaps.is_empty(), !deferred);
    drop(engine);
    let file = File::open(&path).unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_string().unwrap(),
        [if deferred { "after" } else { "other" }]
    );
    assert_eq!(
        file.dataset("d").unwrap().attrs().unwrap()["unchanged"],
        AttrValue::VarLenAsciiString("other".into())
    );
}

#[rstest]
#[case::unknown_parent(false)]
#[case::malformed_descriptor(true)]
fn overwrite_candidates_preserve_unknown_or_partial_original_evidence(
    #[case] malformed: bool,
    #[values(false, true)] values_first: bool,
) {
    let path = test_util::temp::temp_path("unknown-overwrite.h5");
    let mut fixture = Fixture::new();
    let collection = if malformed {
        fixture.collection(&[Object::new(7, b"before")])
    } else {
        Fixture::ROOT
    };
    let mut raw = global_heap_fixture::reference(6, collection, 7);
    raw.extend(global_heap_fixture::reference(6, u64::MAX, 7));
    let messages = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::vlen_string(),
        if malformed { &raw } else { &raw[..16] },
    );
    let object = fixture.append_header(messages);
    fixture.link("d", object);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let mut replacement = DatasetBuilder::new("");
    replacement.with_vlen_strings(if malformed {
        &["after", "after"]
    } else {
        &["after"]
    });
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("d"), replacement)
        .unwrap();
    let mut phase = AnalysisPhase::new(&engine);
    crate::edit::global_heap_liveness::DATASET_VALUE_WORK.with(|work| work.set(0));
    if !values_first {
        phase
            .read_reference_facts(super::object(object), false)
            .unwrap();
    }
    let candidates = phase.overwrite_collections(&engine.staged).unwrap();
    assert!(candidates.collections.is_empty());
    assert!(Arc::ptr_eq(
        candidates
            .observations
            .as_ref()
            .unwrap()
            .identity
            .as_ref()
            .unwrap(),
        phase.identity.as_ref().unwrap()
    ));
    let facts = &candidates.observations.as_ref().unwrap().predecessors[&CollectionProvenance {
        object: super::object(object),
        content: CollectionContent::DatasetValues,
    }];
    assert_eq!(
        facts.objects,
        BTreeSet::from([GlobalHeapId {
            collection: PhysicalAddress::new(collection),
            index: 7.try_into().unwrap(),
        }])
    );
    assert_eq!(facts.complete, !malformed);
    if malformed {
        let HeapReferenceError::Format(FormatError::VlDataError(reason)) =
            facts.error.as_ref().unwrap()
        else {
            panic!("expected an invalid descriptor, got {:?}", facts.error);
        };
        assert_eq!(reason, "non-empty VL element has an undefined heap address");
    } else {
        assert!(facts.error.is_none());
    }
    let all = phase
        .read_reference_facts(super::object(object), false)
        .unwrap();
    assert_eq!(all.objects, facts.objects);
    assert_eq!(all.complete, facts.complete);
    if let Some(HeapReferenceError::Format(expected)) = &facts.error {
        let Some(HeapReferenceError::Format(actual)) = &all.error else {
            panic!("expected the cached descriptor error, got {:?}", all.error);
        };
        assert_eq!(actual, expected);
    } else {
        assert!(all.error.is_none());
    }
    assert_eq!(
        crate::edit::global_heap_liveness::DATASET_VALUE_WORK.with(|work| work.get()),
        1
    );
    assert!(phase.liveness.is_none());
}

#[test]
fn overwrite_candidates_retain_unsupported_reference_interpretation() {
    let path = test_util::temp::temp_path("unsupported-overwrite.h5");
    let mut fixture = Fixture::new();
    let messages = Form::CompactDataset.messages(
        &mut fixture,
        &global_heap_fixture::object_reference(),
        &[0; 8],
    );
    let object = fixture.append_header(messages);
    fixture.link("d", object);
    fixture.write(&path);
    let mut engine = edit_test_util::open_engine(&path);
    let mut replacement = DatasetBuilder::new("");
    replacement.with_raw_data(global_heap_fixture::object_reference(), vec![0; 8], 1);
    engine
        .stage_dataset_write(&ObjectPathBuf::parse("d"), replacement)
        .unwrap();
    let mut phase = AnalysisPhase::new(&engine);
    let candidates = phase.overwrite_collections(&engine.staged).unwrap();
    assert!(candidates.collections.is_empty());
    let observations = candidates.observations.as_ref().unwrap();
    assert!(Arc::ptr_eq(
        observations.identity.as_ref().unwrap(),
        phase.identity.as_ref().unwrap()
    ));
    assert_eq!(observations.predecessors.len(), 1);
    let facts = &observations.predecessors[&CollectionProvenance {
        object: super::object(object),
        content: CollectionContent::DatasetValues,
    }];
    assert!(facts.objects.is_empty());
    assert!(!facts.complete);
    assert!(facts.error.is_none());
    assert!(phase.allocations.is_empty());
    assert!(phase.liveness.is_none());
}

#[rstest]
#[case::empty(false)]
#[case::numeric(true)]
fn overwrite_candidates_skip_reference_free_inputs(#[case] numeric: bool) {
    let path = test_util::temp::temp_path("numeric-overwrite.h5");
    let mut builder = FileBuilder::new();
    builder.create_dataset("d").with_i32_data(&[1]);
    builder.write(&path).unwrap();
    let calls = Arc::new(AtomicU64::new(0));
    let mut engine = WriteEngine::open_imaged(&path, None, |handle, len| {
        Ok(Box::new(
            CountingImage::new(
                Box::new(HandleImage::new(
                    handle,
                    len,
                    MetadataCacheConfig::disabled(),
                )),
                Arc::default(),
                Arc::default(),
            )
            .with_read_calls(Arc::clone(&calls)),
        ))
    })
    .unwrap();
    if numeric {
        let mut replacement = DatasetBuilder::new("");
        replacement.with_i32_data(&[2]);
        engine
            .stage_dataset_write(&ObjectPathBuf::parse("d"), replacement)
            .unwrap();
    }
    let mut phase = AnalysisPhase::new(&engine);
    let mut positive = [0; 8];
    engine.image.read_at(0, &mut positive).unwrap();
    assert_eq!(&positive, b"\x89HDF\r\n\x1a\n");
    assert!(calls.load(Ordering::Relaxed) > 0);
    calls.store(0, Ordering::Relaxed);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let candidates = {
        let region = if numeric {
            "numeric_overwrite_reference_free"
        } else {
            "empty_overwrite_reference_free"
        };
        let (control, positive) =
            allocation::measure(&format!("{region}_control"), || vec![0_u8; 64]);
        assert_eq!(control, [0; 64]);
        assert!(positive.blocks > 0 && positive.bytes >= 64, "{positive}");
        let (candidates, measured) = allocation::measure(region, || {
            phase.overwrite_collections(&engine.staged).unwrap()
        });
        assert_eq!(measured.blocks, 0);
        assert_eq!(measured.bytes, 0);
        candidates
    };
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let candidates = phase.overwrite_collections(&engine.staged).unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert!(candidates.observations.is_none());
    assert!(candidates.collections.is_empty());
    assert!(phase.identity.is_none());
    assert!(phase.value_references.is_empty());
    assert!(phase.allocations.is_empty());
    assert!(phase.references.is_empty());
    assert!(phase.topology.is_none());
    assert!(phase.liveness.is_none());
}

#[derive(Clone, Copy)]
enum Layout {
    Chunked,
    Compact,
    Contiguous,
}

impl Layout {
    fn messages(self, fixture: &mut Fixture, raw: &[u8]) -> Vec<Message> {
        let form = if matches!(self, Self::Compact) {
            Form::CompactDataset
        } else {
            Form::ContiguousDataset
        };
        let mut messages = form.messages(
            fixture,
            &type_builders::make_vlen_string_type(CharacterSet::Utf8),
            raw,
        );
        if matches!(self, Self::Chunked) {
            let address = fixture.image.append(raw) as u64;
            messages.retain(|message| message.msg_type != MessageType::DATA_LAYOUT);
            messages.push(Message::new(
                MessageType::DATA_LAYOUT,
                &::test_util::data_layout::single_chunk_with_geometry(
                    Widths::EIGHT,
                    None,
                    Some(address),
                    1,
                    16,
                ),
            ));
        }
        messages
    }
}
