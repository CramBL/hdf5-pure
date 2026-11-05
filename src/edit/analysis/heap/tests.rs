use core::cell::Cell;
use core::num::NonZeroU16;
use std::collections::BTreeSet;
use std::sync::atomic::Ordering;

use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::bytes;
use test_util::global_heap;
use test_util::global_heap::Object;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture;
use test_util_hdf5::global_heap_fixture::Fixture;
use test_util_hdf5::global_heap_fixture::Form;

use super::HeapReferenceError;
use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::COLLECTION_ALLOCATIONS;
use crate::edit::analysis::COLLECTION_WORK;
use crate::edit::test_util as edit_test_util;
use crate::error::FormatError;
use crate::global_heap::GlobalHeapId;
use crate::object_graph::ObjectId;

#[rstest]
#[case::attribute(Form::CompactAttribute)]
#[case::compact(Form::CompactDataset)]
#[case::contiguous(Form::ContiguousDataset)]
fn address_queries_retain_sparse_member_ids_without_decoding_again(#[case] form: Form) {
    let path = test_util::temp::temp_path("heap-member-facts.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[
        Object::new(1, b"object01"),
        Object::new(7, b"object07"),
        Object {
            index: u16::MAX,
            reference_count: 0,
            data: b"lastslot",
        },
    ]);
    let raw = [
        global_heap_fixture::reference(8, collection, 1),
        global_heap_fixture::reference(8, collection, 7),
        global_heap_fixture::reference(8, collection, 7),
        global_heap_fixture::reference(8, collection, u32::from(u16::MAX)),
        global_heap_fixture::reference(0, 0, 0),
        global_heap_fixture::reference(0, collection, 7),
    ]
    .concat();
    let records = form.messages(&mut fixture, &global_heap_fixture::vlen_string(), &raw);
    let owner = fixture.append_header(records);
    fixture.link("a", owner);
    fixture.link("b", owner);
    fixture.write(&path);
    let original = std::fs::read(&path).unwrap();
    let (engine, calls) = edit_test_util::open_counted_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase
        .reconstruction_source(&engine, ObjectId::new(PhysicalAddress::new(owner)))
        .unwrap();
    COLLECTION_WORK.with(|work| work.set((0, 0)));
    let live = phase
        .collections_reachable_from([ObjectId::new(PhysicalAddress::new(Fixture::ROOT))])
        .unwrap();
    assert!(live.analysis_complete);
    assert_eq!(live.referenced_collections, BTreeSet::from([collection]));
    let read_calls = calls.load(Ordering::Relaxed);
    let references = phase.heap_reference_facts(&source).unwrap();
    assert!(references.is_complete());
    assert!(references.error().is_none());
    assert_eq!(references.collections(), &live.referenced_collections);
    assert_eq!(
        references.objects(),
        &BTreeSet::from([
            id(collection, 1),
            id(collection, 7),
            id(collection, u16::MAX)
        ])
    );
    assert_eq!(calls.load(Ordering::Relaxed), read_calls);
    assert_eq!(COLLECTION_WORK.with(Cell::get), (2, 2));
    COLLECTION_ALLOCATIONS.with(|count| count.set(0));
    let allocation = phase
        .collection_allocation(PhysicalAddress::new(collection))
        .unwrap();
    let read_calls = calls.load(Ordering::Relaxed);
    let contents = phase
        .heap_members(&source, PhysicalAddress::new(collection))
        .unwrap();
    assert_eq!(contents.range(), allocation.range);
    assert_eq!(
        contents.range().length(),
        u64::try_from(global_heap::MIN_COLLECTION_SIZE).unwrap()
    );
    let members = contents
        .members()
        .iter()
        .map(|member| {
            let payload = member.payload().range();
            (
                member.index().get(),
                member.reference_count(),
                member.range(),
                payload,
            )
        })
        .collect::<Vec<_>>();
    let header = u64::try_from(global_heap::SIZE_OFFSET + Widths::EIGHT.length).unwrap();
    for ((index, references, allocation, payload), (expected_index, expected_references, value)) in
        members.into_iter().zip([
            (1, 1, b"object01"),
            (7, 1, b"object07"),
            (u16::MAX, 0, b"lastslot"),
        ])
    {
        let at = u64::try_from(bytes::sole_signature(&original, value)).unwrap();
        assert_eq!((index, references), (expected_index, expected_references));
        assert_eq!(
            (payload.start().get(), payload.length()),
            (at, u64::try_from(value.len()).unwrap())
        );
        assert_eq!(
            (allocation.start().get(), allocation.length()),
            (at - header, header + payload.length())
        );
    }
    assert_eq!(contents.members().len(), 3);
    assert_eq!(calls.load(Ordering::Relaxed), read_calls);
    assert_eq!(COLLECTION_ALLOCATIONS.with(Cell::get), 1);
    drop(phase);
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), original);
}

#[rstest]
#[case::header(ReferenceInput::Header)]
#[case::link(ReferenceInput::Link)]
#[case::empty(ReferenceInput::Empty)]
#[case::null(ReferenceInput::Null)]
#[case::unsupported(ReferenceInput::Unsupported)]
#[case::unknown(ReferenceInput::Unknown)]
#[case::malformed(ReferenceInput::Malformed)]
#[case::missing_datatype(ReferenceInput::MissingDatatype)]
fn reference_completeness_and_failures_remain_cached(#[case] input: ReferenceInput) {
    let path = test_util::temp::temp_path("heap-reference-completeness.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"value")]);
    let raw = match input {
        ReferenceInput::Empty => Vec::new(),
        ReferenceInput::Null => global_heap_fixture::reference(0, u64::MAX, 0),
        ReferenceInput::Malformed => [
            global_heap_fixture::reference(5, collection, 1),
            global_heap_fixture::reference(1, collection, 0),
        ]
        .concat(),
        _ => global_heap_fixture::reference(5, collection, 1),
    };
    let datatype = if matches!(input, ReferenceInput::Unsupported) {
        global_heap_fixture::object_reference()
    } else {
        global_heap_fixture::vlen_string()
    };
    let mut records = Form::CompactDataset.messages(&mut fixture, &datatype, &raw);
    if matches!(input, ReferenceInput::Unknown) {
        records.push(Message::new(MessageType(0x7f), &[]));
    }
    if matches!(input, ReferenceInput::MissingDatatype) {
        records.retain(|record| record.msg_type != MessageType::DATATYPE);
    }
    if matches!(input, ReferenceInput::Link) {
        records.push(Message::new(MessageType::LINK, &[0xff, 0]));
    }
    let owner = if matches!(input, ReferenceInput::Header) {
        0
    } else {
        fixture.append_header(records)
    };
    fixture.link("a", owner);
    fixture.write(&path);
    let (engine, calls) = edit_test_util::open_counted_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase
        .reconstruction_source(&engine, ObjectId::new(PhysicalAddress::new(owner)))
        .unwrap();
    let facts = phase.heap_reference_facts(&source).unwrap();
    assert_eq!(
        facts.is_complete(),
        matches!(
            input,
            ReferenceInput::Empty | ReferenceInput::Null | ReferenceInput::Link
        )
    );
    let expected = if matches!(
        input,
        ReferenceInput::Unknown | ReferenceInput::Malformed | ReferenceInput::Link
    ) {
        BTreeSet::from([id(collection, 1)])
    } else {
        BTreeSet::new()
    };
    assert_eq!(facts.objects(), &expected);
    match input {
        ReferenceInput::Header => {
            let Some(HeapReferenceError::Format(error)) = facts.error() else {
                panic!("expected header error, got {:?}", facts.error());
            };
            assert_eq!(
                error,
                &FormatError::InvalidObjectHeaderVersion(test_util::superblock::SIGNATURE[0])
            );
        }
        ReferenceInput::Malformed => {
            let Some(HeapReferenceError::Format(error)) = facts.error() else {
                panic!("expected malformed descriptor, got {:?}", facts.error());
            };
            assert_eq!(
                error,
                &FormatError::GlobalHeapObjectNotFound {
                    collection_address: collection,
                    index: 0
                }
            );
        }
        ReferenceInput::MissingDatatype => assert!(
            matches!(
                facts.error(),
                Some(HeapReferenceError::MissingMessage(
                    crate::MessageType::DATATYPE
                ))
            ),
            "{:?}",
            facts.error()
        ),
        _ => assert!(facts.error().is_none()),
    }
    let before = calls.load(Ordering::Relaxed);
    assert_eq!(
        phase.heap_reference_facts(&source).unwrap().objects(),
        &expected
    );
    assert_eq!(calls.load(Ordering::Relaxed), before);
    if matches!(input, ReferenceInput::Link) {
        let error = phase
            .collections_reachable_from([source.object()])
            .unwrap_err();
        let crate::error::Error::Format(error) = error else {
            panic!("expected link error, got {error:?}");
        };
        assert_eq!(error, FormatError::InvalidLinkVersion(0xff));
        assert!(phase.heap_reference_facts(&source).unwrap().is_complete());
    }
}

#[test]
fn collection_framing_failure_is_independent_of_complete_reference_facts() {
    let path = test_util::temp::temp_path("heap-member-error.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"one"), Object::new(1, b"two")]);
    let raw = global_heap_fixture::reference(3, collection, 1);
    let records =
        Form::CompactDataset.messages(&mut fixture, &global_heap_fixture::vlen_string(), &raw);
    let owner = fixture.append_header(records);
    fixture.link("a", owner);
    fixture.write(&path);
    let other_path = test_util::temp::temp_path("heap-member-other-session.h5");
    std::fs::copy(&path, &other_path).unwrap();
    let (engine, calls) = edit_test_util::open_counted_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let source = phase
        .reconstruction_source(&engine, ObjectId::new(PhysicalAddress::new(owner)))
        .unwrap();
    assert!(phase.heap_reference_facts(&source).unwrap().is_complete());
    let error = phase
        .heap_members(&source, PhysicalAddress::new(collection))
        .err()
        .unwrap();
    assert_eq!(
        error,
        FormatError::DuplicateGlobalHeapObjectIndex {
            collection_address: collection,
            index: 1
        }
    );
    let before = calls.load(Ordering::Relaxed);
    assert_eq!(
        phase
            .heap_members(&source, PhysicalAddress::new(collection))
            .err()
            .unwrap(),
        error
    );
    assert_eq!(calls.load(Ordering::Relaxed), before);
    assert_eq!(
        phase.heap_reference_facts(&source).unwrap().objects(),
        &BTreeSet::from([id(collection, 1)])
    );
    let mut other = AnalysisPhase::new(&engine);
    let error = other.heap_reference_facts(&source).err().unwrap();
    let crate::error::Error::Format(error) = error else {
        panic!("expected phase error, got {error:?}");
    };
    assert_eq!(
        error,
        FormatError::Internal("heap query belongs to another source phase".into())
    );
    assert_eq!(
        other
            .heap_members(&source, PhysicalAddress::new(collection))
            .err()
            .unwrap(),
        error
    );
    assert_eq!(calls.load(Ordering::Relaxed), before);
    let (other_engine, other_calls) = edit_test_util::open_counted_engine(&other_path);
    other_calls.store(0, Ordering::Relaxed);
    let mut other_session = AnalysisPhase::new(&other_engine);
    let session_error = other_session.heap_reference_facts(&source).err().unwrap();
    let crate::error::Error::Format(session_error) = session_error else {
        panic!("expected session error, got {session_error:?}");
    };
    assert_eq!(session_error, error);
    assert_eq!(
        other_session
            .heap_members(&source, PhysicalAddress::new(collection))
            .err()
            .unwrap(),
        error
    );
    assert_eq!(other_calls.load(Ordering::Relaxed), 0);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::numeric(false)]
#[case::variable_length(true)]
fn heap_identity_queries_bound_source_reads_and_allocations(#[case] variable: bool) {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;

    use test_util::allocation;

    use crate::FileLocking;
    use crate::edit::WriteEngine;
    use crate::image::CountingImage;
    use crate::image::HandleImage;
    use crate::source::MetadataCacheConfig;
    let path = test_util::temp::temp_path("heap-member-resources.h5");
    let payload = vec![0x5a; 1_048_576];
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(u16::MAX, &payload)]);
    let descriptor = global_heap_fixture::reference(262_144, collection, u32::from(u16::MAX));
    let datatype = if variable {
        global_heap_fixture::vlen_sequence()
    } else {
        crate::type_builders::make_i32_type()
    };
    let records = Form::ContiguousDataset.messages(
        &mut fixture,
        &datatype,
        if variable { &descriptor } else { &payload },
    );
    let owner = fixture.append_header(records);
    fixture.link("a", owner);
    fixture.write(&path);
    let calls = Arc::new(AtomicU64::new(0));
    let read_bytes = Arc::new(AtomicU64::new(0));
    let engine = WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, length| {
        Ok(Box::new(
            CountingImage::new(
                Box::new(HandleImage::new(
                    handle,
                    length,
                    MetadataCacheConfig::disabled(),
                )),
                Arc::clone(&read_bytes),
                Arc::default(),
            )
            .with_read_calls(Arc::clone(&calls)),
        ))
    })
    .unwrap();
    calls.store(0, Ordering::Relaxed);
    read_bytes.store(0, Ordering::Relaxed);
    let (mut phase, empty) =
        allocation::measure("heap_empty_phase", || AnalysisPhase::new(&engine));
    assert_eq!((empty.blocks, empty.bytes), (0, 0));
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let source = phase
        .reconstruction_source(&engine, ObjectId::new(PhysicalAddress::new(owner)))
        .unwrap();
    let (_, positive) = allocation::measure("heap_payload_control", || payload.clone());
    assert!(positive.bytes >= u64::try_from(payload.len()).unwrap());
    let ((), first) = allocation::measure("heap_sparse_identity", || {
        let references = phase.heap_reference_facts(&source).unwrap();
        assert!(references.is_complete());
        assert_eq!(references.objects().len(), usize::from(variable));
        if variable {
            assert_eq!(
                references.objects().first(),
                Some(&id(collection, u16::MAX))
            );
            assert_eq!(
                phase
                    .heap_members(&source, PhysicalAddress::new(collection))
                    .unwrap()
                    .members()
                    .len(),
                1
            );
        }
    });
    assert!(
        first.blocks > 0 && first.bytes > 0 && first.bytes < 32_768,
        "{first}"
    );
    assert!(first.live_bytes > 0 && first.live_bytes < 8_192, "{first}");
    let before = calls.load(Ordering::Relaxed);
    let bytes_before = read_bytes.load(Ordering::Relaxed);
    assert!(before > 0 && before < 16, "{before}");
    assert!(bytes_before > 0 && bytes_before < 16_384, "{bytes_before}");
    let ((), cached) = allocation::measure("heap_cached_identities", || {
        assert!(phase.heap_reference_facts(&source).unwrap().is_complete());
        if variable {
            assert_eq!(
                phase
                    .heap_members(&source, PhysicalAddress::new(collection))
                    .unwrap()
                    .members()[0]
                    .payload()
                    .range()
                    .length(),
                u64::try_from(payload.len()).unwrap()
            );
        } else {
            assert!(phase.allocations.is_empty());
        }
    });
    assert_eq!((cached.blocks, cached.bytes), (0, 0));
    assert_eq!(calls.load(Ordering::Relaxed), before);
    assert_eq!(read_bytes.load(Ordering::Relaxed), bytes_before);
}

fn id(collection: u64, index: u16) -> GlobalHeapId {
    GlobalHeapId {
        collection: PhysicalAddress::new(collection),
        index: NonZeroU16::new(index).unwrap(),
    }
}

#[derive(Clone, Copy)]
enum ReferenceInput {
    Empty,
    Header,
    Link,
    Malformed,
    MissingDatatype,
    Null,
    Unknown,
    Unsupported,
}
