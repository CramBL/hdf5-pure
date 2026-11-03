use core::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::ObjectReferenceCount;
use hdf5_pure_format::__private::ObjectReferenceCountIssue;
use rstest::rstest;
use test_util::global_heap::Object;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags;
use test_util::object_header::MessageType;
use test_util::object_header::v2;
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::Fixture;

use super::super::AnalysisPhase;
use super::super::DISCOVERY_WORK;
use crate::FileLocking;
use crate::edit::WriteEngine;
use crate::edit::test_util as edit_test_util;
use crate::error::FormatError;
use crate::image::CountingImage;
use crate::image::FileImage;
use crate::image::HandleImage;
use crate::object_graph::ObjectId;
use crate::source::MetadataCacheConfig;

#[rstest]
fn count_queries_are_lazy_cached_and_bound_to_source_contents(
    #[values(false, true)] mirrored: bool,
) {
    let path = test_util::temp::temp_path("count-cache.h5");
    let mut fixture = Fixture::new();
    let address = fixture.append_header(vec![Message::reference_count(2)]);
    fixture.link("a", address);
    fixture.link("b", address);
    fixture.write(&path);
    let calls = Arc::new(AtomicU64::new(0));
    let bytes = Arc::new(AtomicU64::new(0));
    let mut engine = WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, len| {
        let image: Box<dyn FileImage> = if mirrored {
            Box::new(WriteEngine::read_mirror(handle)?)
        } else {
            Box::new(HandleImage::new(
                handle,
                len,
                MetadataCacheConfig::disabled(),
            ))
        };
        Ok(Box::new(
            CountingImage::new(image, bytes.clone(), Arc::default()).with_read_calls(calls.clone()),
        ))
    })
    .unwrap();
    calls.store(0, Ordering::Relaxed);
    bytes.store(0, Ordering::Relaxed);
    let object = ObjectId::new(PhysicalAddress::new(address));
    let mut phase = AnalysisPhase::new(&engine);
    assert!(phase.counts.is_empty());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let observed = phase.object_reference_count(object).unwrap();
    let ObjectReferenceCount::Version2(field) = observed else {
        panic!("expected an explicit count")
    };
    assert_eq!(field.value(), 2);
    assert_eq!(field.range().start().get(), address + 12);
    assert_eq!(field.chunk().start().get(), address);
    assert_eq!(field.chunk().length(), 20);
    let read_calls = calls.load(Ordering::Relaxed);
    let read_bytes = bytes.load(Ordering::Relaxed);
    assert!(read_calls > 0 && read_bytes >= 20);
    assert_eq!(phase.object_reference_count(object).unwrap(), observed);
    assert_eq!(calls.load(Ordering::Relaxed), read_calls);
    assert_eq!(bytes.load(Ordering::Relaxed), read_bytes);
    assert!(phase.topology.is_none() && phase.links.is_empty() && phase.references.is_empty());
    assert_eq!(phase.topology().unwrap().incoming(address), Some(2));
    let references = phase.collections_reachable_from([Fixture::ROOT]).unwrap();
    assert!(references.analysis_complete && references.referenced_collections.is_empty());
    assert_eq!(phase.counts.len(), 1);
    let after_enrichment = calls.load(Ordering::Relaxed);
    assert!(after_enrichment > read_calls);
    assert_eq!(phase.object_reference_count(object).unwrap(), observed);
    assert_eq!(calls.load(Ordering::Relaxed), after_enrichment);
    drop(phase);
    let replacement = v2::Header::new()
        .message(Message::reference_count(9))
        .build();
    engine.write_at(address, &replacement).unwrap();
    let mut phase = AnalysisPhase::prospective(&engine, PhysicalAddress::new(Fixture::ROOT));
    assert!(phase.counts.is_empty());
    let ObjectReferenceCount::Version2(updated) = phase.object_reference_count(object).unwrap()
    else {
        panic!("expected an explicit count")
    };
    assert_eq!(updated.value(), 9);
    assert_eq!(updated.range(), field.range());
    assert!(calls.load(Ordering::Relaxed) > after_enrichment);
    assert!(phase.topology.is_none());
}

#[rstest]
#[case::implicit(CountForm::Implicit)]
#[case::unsupported(CountForm::Unsupported)]
#[case::ambiguous(CountForm::Ambiguous)]
fn count_completeness_does_not_replace_other_phase_facts(#[case] form: CountForm) {
    let path = test_util::temp::temp_path("count-completeness.h5");
    let mut fixture = Fixture::new();
    let collection = fixture.collection(&[Object::new(1, b"value")]);
    let records = match form {
        CountForm::Ambiguous => vec![Message::reference_count(2), Message::reference_count(3)],
        CountForm::Implicit => vec![],
        CountForm::Unsupported => vec![Message::new(
            MessageType::OBJECT_REFERENCE_COUNT,
            &[1, 2, 0, 0, 0],
        )],
    };
    let address = fixture.append_header(records);
    fixture.link("a", address);
    fixture.write(&path);
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    assert_eq!(phase.topology().unwrap().incoming(address), Some(1));
    let references = phase.collections_reachable_from([Fixture::ROOT]).unwrap();
    assert!(references.analysis_complete && references.referenced_collections.is_empty());
    let allocation = phase
        .collection_allocation(PhysicalAddress::new(collection))
        .unwrap();
    assert!(phase.counts.is_empty());
    let expected = match form {
        CountForm::Ambiguous => ObjectReferenceCount::Ambiguous,
        CountForm::Implicit => ObjectReferenceCount::ImplicitOne,
        CountForm::Unsupported => {
            ObjectReferenceCount::Unsupported(ObjectReferenceCountIssue::MessageVersion(1))
        }
    };
    assert_eq!(
        phase
            .object_reference_count(ObjectId::new(PhysicalAddress::new(address)))
            .unwrap(),
        expected
    );
    assert_eq!(phase.topology().unwrap().incoming(address), Some(1));
    assert_eq!(
        phase
            .collections_reachable_from([Fixture::ROOT])
            .unwrap()
            .analysis_complete,
        references.analysis_complete
    );
    assert_eq!(
        phase
            .collection_allocation(PhysicalAddress::new(collection))
            .unwrap()
            .range,
        allocation.range
    );
    assert_eq!(DISCOVERY_WORK.with(Cell::get), (1, 2));
    assert_eq!(phase.counts.len(), 1);
    assert_eq!(phase.references.len(), 2);
    assert_eq!(phase.allocations.len(), 1);
}

#[rstest]
#[case::source(Failure::Source)]
#[case::message(Failure::Message)]
fn incomplete_parses_cache_failures_without_promoting_count_scratch(#[case] failure: Failure) {
    let path = test_util::temp::temp_path("count-parse-failure.h5");
    let mut fixture = Fixture::new();
    let continuation = match failure {
        Failure::Message => {
            let bytes = v2::continuation_chunk(
                &[Message::new(MessageType::UNKNOWN, &[])
                    .with_flags(MessageFlags::FAIL_IF_UNKNOWN_ALWAYS)],
                v2::HeaderFlags::default(),
            );
            let address = fixture.image.append(&bytes);
            Message::continuation(
                u64::try_from(address).unwrap(),
                u64::try_from(bytes.len()).unwrap(),
                Widths::EIGHT,
            )
        }
        Failure::Source => Message::continuation(1_000_000, 64, Widths::EIGHT),
    };
    let address = fixture.append_header(vec![Message::reference_count(2), continuation]);
    fixture.link("a", address);
    fixture.write(&path);
    let calls = Arc::new(AtomicU64::new(0));
    let engine = WriteEngine::open_imaged(&path, Some(FileLocking::Enabled), |handle, len| {
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
            .with_read_calls(calls.clone()),
        ))
    })
    .unwrap();
    let expected = match failure {
        Failure::Message => FormatError::UnsupportedMessage(MessageType::UNKNOWN.0),
        Failure::Source => FormatError::UnexpectedSourceEof {
            expected: 1_000_064,
            available: engine.image.len(),
        },
    };
    let mut phase = AnalysisPhase::new(&engine);
    let object = ObjectId::new(PhysicalAddress::new(address));
    calls.store(0, Ordering::Relaxed);
    assert_eq!(phase.object_reference_count(object).unwrap_err(), expected);
    let reads = calls.load(Ordering::Relaxed);
    assert!(reads > 0);
    assert_eq!(phase.counts.len(), 1);
    assert_eq!(phase.counts[&object], Err(expected.clone()));
    assert_eq!(phase.object_reference_count(object).unwrap_err(), expected);
    assert_eq!(calls.load(Ordering::Relaxed), reads);
    assert!(
        phase.topology.is_none() && phase.references.is_empty() && phase.allocations.is_empty()
    );
}

#[derive(Clone, Copy)]
enum CountForm {
    Ambiguous,
    Implicit,
    Unsupported,
}

#[derive(Clone, Copy)]
enum Failure {
    Message,
    Source,
}
