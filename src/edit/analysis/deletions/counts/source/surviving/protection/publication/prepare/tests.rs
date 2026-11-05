use core::num::NonZeroU32;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use core::num::NonZeroUsize;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::MetadataBytes;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::ObjectReferenceCount;
use hdf5_pure_format::__private::ObjectReferenceCountIssue;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use hdf5_pure_format::__private::ObjectReferenceCountUpdate;
use hdf5_pure_format::__private::ObjectReferenceCountUpdateError;
use hdf5_pure_space::__private::FreeClass;
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags;
use test_util::object_header::MessageType;
use test_util::object_header::v2;
use test_util::widths::Widths;
use test_util_hdf5::global_heap_fixture::Fixture;

use super::super::super::tests as protection_tests;
use super::super::CountPublicationBounds;
use super::super::test_util as count_test_util;
use super::super::test_util::Form;
use super::CountPreparationError;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use super::CurrentCountUpdates;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::UserblockSize;
use crate::address::BaseAddress;
use crate::address::BaseAddressExt;
use crate::convert;
use crate::edit::WriteEngine;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::edit::analysis::DISCOVERY_WORK;
use crate::edit::analysis::counts::CountObservation;
use crate::edit::retirement::RootRetirement;
use crate::edit::test_util as edit_test_util;
use crate::error::FormatError;
use crate::object_graph::ObjectId;
use crate::source::BytesSource;
use crate::source::Source;
use crate::source::SourceMetadata;
use crate::width::FormatWidths;

#[rstest]
#[case::v1(Form::Version1)]
#[case::v2(Form::Version2)]
#[case::continuation(Form::Continuation)]
fn preparation_uses_current_values_and_preserves_complete_predecessor_ownership(
    #[case] form: Form,
    #[values(0, 512)] base: u64,
    #[values(2, u32::MAX)] count: u32,
) {
    let path = test_util::temp::temp_path("current-count-values.h5");
    let fixture =
        count_test_util::fixture(&path, form, base, NonZeroU32::new(count).unwrap(), c"old");
    let mut engine = edit_test_util::open_engine(&path);
    let mut bounds = publication_bounds(&mut engine, &fixture.retired, &["a"]);
    let root = bounds.original_root();
    let old_field = bounds.updates()[0].field();
    let current = form.chunk(count, c"new");
    assert_eq!(
        u64::try_from(current.len()).unwrap(),
        fixture.allocation.length()
    );
    bounds
        .phase
        .engine
        .write_at(fixture.allocation.start().get(), &current)
        .unwrap();
    let before = bounds.phase.engine.disk_bytes().unwrap();
    let prepared = bounds.prepare_counts().unwrap();
    let (original, update) = prepared.updates().next().unwrap();
    assert_eq!(original.object(), fixture.object);
    assert_eq!(original.field(), old_field);
    assert_eq!(original.count().get(), count - 1);
    assert_eq!(original.allocation().range(), fixture.allocation);
    let mut patched = current;
    for (range, bytes) in update.writes() {
        assert!(range.start() >= fixture.allocation.start());
        assert!(range.end() <= fixture.allocation.end());
        let offset = range.start().get() - fixture.allocation.start().get();
        patched[convert::slice_range(offset, range.length()).unwrap()].copy_from_slice(bytes);
    }
    assert_eq!(patched, form.chunk(count - 1, c"new"));
    assert_eq!(
        update.writes().count(),
        if matches!(form, Form::Version1) { 1 } else { 2 }
    );
    assert_eq!(prepared.counts.root(), root);
    assert_eq!(prepared.engine.disk_bytes().unwrap(), before);
    drop(prepared);
    assert_eq!(bounds.original_root(), root);
    assert_eq!(bounds.phase.retirement.batch.entries(), fixture.retired);
}

#[rstest]
#[case::moved(Change::Moved)]
#[case::removed(Change::Removed)]
#[case::duplicate(Change::Duplicate)]
#[case::shared(Change::Shared)]
#[case::resized(Change::Resized)]
#[case::count(Change::Count)]
fn preparation_rejects_changed_framing_and_counts(#[case] change: Change) {
    let path = test_util::temp::temp_path("current-count-framing.h5");
    let fixture = count_test_util::fixture(
        &path,
        Form::Version2,
        0,
        NonZeroU32::new(2).unwrap(),
        c"oldx",
    );
    let mut engine = edit_test_util::open_engine(&path);
    let mut bounds = publication_bounds(&mut engine, &fixture.retired, &["a"]);
    let expected = bounds.updates()[0].field();
    let count = Message::reference_count(2);
    let comment = Message::new(MessageType::OBJECT_COMMENT, c"oldx".to_bytes_with_nul());
    let messages = match change {
        Change::Moved => vec![
            Message::new(MessageType::OBJECT_COMMENT, &count.data),
            count,
        ],
        Change::Removed => vec![Message::nil(count.data.len()), comment],
        Change::Duplicate => vec![count, Message::reference_count(2)],
        Change::Shared => vec![count.with_flags(MessageFlags::SHARED), comment],
        Change::Resized => vec![
            count,
            Message::new(MessageType::OBJECT_COMMENT, c"old".to_bytes_with_nul()),
        ],
        Change::Count => vec![Message::reference_count(3), comment],
    };
    let current = v2::Header::new().messages(messages).build();
    bounds
        .phase
        .engine
        .write_at(fixture.allocation.start().get(), &current)
        .unwrap();
    let before = bounds.phase.engine.disk_bytes().unwrap();
    let Err(error) = bounds.prepare_counts() else {
        panic!("expected changed count evidence")
    };
    match (change, error) {
        (Change::Count, CountPreparationError::Update { object, error }) => {
            assert_eq!(object, fixture.object);
            assert_eq!(
                error,
                ObjectReferenceCountUpdateError::Value {
                    field: expected.range().start(),
                    expected: 2,
                    actual: 3
                }
            );
        }
        (
            change,
            CountPreparationError::Field {
                object,
                expected: actual_expected,
                actual,
            },
        ) => {
            assert_eq!(object, fixture.object);
            assert_eq!(actual_expected, expected);
            match change {
                Change::Removed => assert_eq!(actual, ObjectReferenceCount::ImplicitOne),
                Change::Duplicate => assert_eq!(actual, ObjectReferenceCount::Ambiguous),
                Change::Shared => assert_eq!(
                    actual,
                    ObjectReferenceCount::Unsupported(ObjectReferenceCountIssue::Shared)
                ),
                Change::Moved | Change::Resized => {
                    let ObjectReferenceCount::Version2(field) = actual else {
                        panic!("expected an explicit count")
                    };
                    assert_eq!(field.value(), 2);
                    assert!(field.range() != expected.range() || field.chunk() != expected.chunk());
                    if matches!(change, Change::Moved) {
                        let bytes = &before[convert::slice_range(
                            expected.range().start().get(),
                            expected.range().length(),
                        )
                        .unwrap()];
                        assert_eq!(bytes, 2_u32.to_le_bytes());
                    }
                }
                Change::Count => panic!("expected a scalar count mismatch"),
            }
        }
        (_, other) => panic!("unexpected preparation error: {other:?}"),
    }
    assert_eq!(bounds.phase.retirement.batch.entries(), fixture.retired);
    assert_eq!(bounds.phase.engine.disk_bytes().unwrap(), before);
}

#[rstest]
#[case::initial(ReadFailure::Initial)]
#[case::ancestor(ReadFailure::Ancestor)]
#[case::allocation_eof(ReadFailure::AllocationEof)]
fn preparation_preserves_exact_source_failures(#[case] failure: ReadFailure) {
    let path = test_util::temp::temp_path("current-count-io.h5");
    let fixture = count_test_util::fixture(
        &path,
        Form::Continuation,
        512,
        NonZeroU32::new(2).unwrap(),
        c"old",
    );
    let mut engine = edit_test_util::open_engine(&path);
    let bounds = publication_bounds(&mut engine, &fixture.retired, &["a"]);
    let original = &bounds.updates()[0];
    let end = fixture.allocation.end().get() - 1;
    let error = match failure {
        ReadFailure::AllocationEof => {
            assert!(original.field().range().end().get() <= end);
            let bytes = bounds.phase.engine.disk_bytes().unwrap();
            let source = BytesSource(&bytes[..usize::try_from(end).unwrap()]);
            original
                .prepare_source(
                    &source,
                    bounds.phase.engine.widths(),
                    bounds.phase.engine.addresses().effective_base(),
                )
                .unwrap_err()
        }
        ReadFailure::Initial | ReadFailure::Ancestor => {
            let offset = if matches!(failure, ReadFailure::Initial) {
                fixture.allocation.start().get()
            } else {
                fixture.object.header_address().get()
            };
            let source = FailingSource {
                source: bounds.phase.engine.image(),
                offset,
            };
            original
                .prepare_source(
                    &source,
                    bounds.phase.engine.widths(),
                    bounds.phase.engine.addresses().effective_base(),
                )
                .unwrap_err()
        }
    };
    let expected = match failure {
        ReadFailure::AllocationEof => FormatError::UnexpectedSourceEof {
            expected: fixture.allocation.end().get(),
            available: end,
        },
        ReadFailure::Initial | ReadFailure::Ancestor => {
            FormatError::Source("injected metadata read".into())
        }
    };
    match error {
        CountPreparationError::Read { object, error }
            if !matches!(failure, ReadFailure::Ancestor) =>
        {
            assert_eq!(object, fixture.object);
            assert_eq!(error, expected);
        }
        CountPreparationError::Parse { object, error }
            if matches!(failure, ReadFailure::Ancestor) =>
        {
            assert_eq!(object, fixture.object);
            assert_eq!(error, expected);
        }
        other => panic!("unexpected preparation error: {other:?}"),
    }
    assert_eq!(bounds.phase.retirement.batch.entries(), fixture.retired);
}

#[test]
fn preparation_rejects_misassociated_source_ranges() {
    let path = test_util::temp::temp_path("current-count-range.h5");
    let fixture = count_test_util::fixture(
        &path,
        Form::Version2,
        0,
        NonZeroU32::new(2).unwrap(),
        c"old",
    );
    let mut engine = edit_test_util::open_engine(&path);
    let bounds = publication_bounds(&mut engine, &fixture.retired, &["a"]);
    let bytes = bounds.phase.engine.disk_bytes().unwrap();
    let source = MislocatedSource(&bytes);
    let original = &bounds.updates()[0];
    let Err(CountPreparationError::Range {
        object,
        expected,
        provided,
    }) = original.prepare_source(
        &source,
        bounds.phase.engine.widths(),
        bounds.phase.engine.addresses().effective_base(),
    )
    else {
        panic!("expected a misassociated source range")
    };
    assert_eq!(object, fixture.object);
    assert_eq!(expected, fixture.allocation);
    assert_eq!(
        provided,
        FileRange::new(
            PhysicalAddress::new(expected.start().get() + 1),
            expected.length()
        )
        .unwrap()
    );
}

#[rstest]
#[case::header(Form::Version2)]
#[case::continuation(Form::Continuation)]
fn preparation_preserves_read_checksum_policy_and_emits_valid_checksums(#[case] form: Form) {
    let path = test_util::temp::temp_path("current-count-checksum.h5");
    let fixture = count_test_util::fixture(&path, form, 0, NonZeroU32::new(2).unwrap(), c"old");
    let mut engine = edit_test_util::open_engine(&path);
    let mut bounds = publication_bounds(&mut engine, &fixture.retired, &["a"]);
    let mut current = form.chunk(2, c"new");
    let footer = current.len() - test_util::checksum::CHECKSUM;
    let computed = test_util::checksum::lookup3(&current[..footer]);
    let expected = computed ^ 1;
    current[footer..].copy_from_slice(&expected.to_le_bytes());
    bounds
        .phase
        .engine
        .write_at(fixture.allocation.start().get(), &current)
        .unwrap();
    if cfg!(feature = "checksum") {
        let Err(CountPreparationError::Parse { object, error }) = bounds.prepare_counts() else {
            panic!("expected read checksum error")
        };
        assert_eq!(object, fixture.object);
        assert_eq!(error, FormatError::ChecksumMismatch { expected, computed });
    } else {
        let prepared = bounds.prepare_counts().unwrap();
        for (range, bytes) in prepared.updates().next().unwrap().1.writes() {
            current[convert::slice_range(
                range.start().get() - fixture.allocation.start().get(),
                range.length(),
            )
            .unwrap()]
            .copy_from_slice(bytes);
        }
        assert_eq!(current, form.chunk(1, c"new"));
    }
}

#[rstest]
#[case::scalar(1)]
#[case::batch(6)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn preparation_skips_empty_work_and_retains_only_compact_updates(
    #[case] objects: usize,
    #[values(1_024, 32_768)] body: usize,
) {
    let path = test_util::temp::temp_path("current-count-cost.h5");
    let comment = std::ffi::CString::new(vec![b'x'; body]).unwrap();
    let mut fixture = Fixture::new();
    let retired = protection_tests::append_retirements(&mut fixture, UserblockSize::NONE);
    let chunk = Form::Version2.chunk(2, &comment);
    let names = (0..NonZeroUsize::new(objects).unwrap().get())
        .map(|index| {
            let at = u64::try_from(fixture.image.append(&chunk)).unwrap();
            let name = format!("a{index}");
            fixture.link(&name, at);
            fixture.link(&format!("b{index}"), at);
            name
        })
        .collect::<Vec<_>>();
    fixture.write(&path);
    let names = names.iter().map(String::as_str).collect::<Vec<_>>();
    let (mut engine, calls) = protection_tests::engine_with_read_call_counter(&path);
    let mut bounds = publication_bounds(&mut engine, &retired, &names);
    assert!(calls.load(std::sync::atomic::Ordering::Relaxed) > 0);
    calls.store(0, std::sync::atomic::Ordering::Relaxed);
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    let (prepared, measured) = allocation::measure(
        &format!("prepare-current-count-control-{objects}-{body}"),
        || bounds.prepare_counts().unwrap(),
    );
    assert_eq!(prepared.updates().len(), objects);
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::Relaxed),
        u64::try_from(objects).unwrap()
    );
    assert!(measured.blocks > 0 && measured.live_bytes > 0, "{measured}");
    assert!(
        measured.live_bytes
            <= u64::try_from(core::mem::size_of::<ObjectReferenceCountUpdate>() * objects).unwrap(),
        "{measured}"
    );
    assert!(
        measured.peak_bytes
            < u64::try_from(
                chunk.len() * 3 + objects * core::mem::size_of::<ObjectReferenceCountUpdate>()
            )
            .unwrap(),
        "{measured}"
    );
    DISCOVERY_WORK.with(|work| assert_eq!(work.get(), (0, 0)));
    drop(prepared);
    drop(bounds);
    let mut empty = publication_bounds(&mut engine, &retired, &["a0", "b0"]);
    calls.store(0, std::sync::atomic::Ordering::Relaxed);
    let (prepared, measured) = allocation::measure(
        &format!("prepare-current-count-empty-{objects}-{body}"),
        || empty.prepare_counts().unwrap(),
    );
    assert_eq!(prepared.updates().len(), 0);
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    drop(prepared);
    assert_eq!(empty.phase.retirement.batch.entries(), retired);
}

#[test]
fn preparation_rejects_new_overlapping_chunk_allocations() {
    let path = test_util::temp::temp_path("current-count-overlap.h5");
    let mut fixture = Fixture::new();
    let child = v2::continuation_chunk(&[], v2::HeaderFlags::default());
    let child_at = fixture.image.append(&child);
    let comment = Message::new(MessageType::OBJECT_COMMENT, &child);
    let original = v2::Header::new()
        .messages([Message::reference_count(2), comment.clone()])
        .continuation(child_at, &child, Widths::EIGHT)
        .build();
    let at = fixture.image.append(&original);
    fixture.link("a", u64::try_from(at).unwrap());
    fixture.link("b", u64::try_from(at).unwrap());
    fixture.write(&path);
    let embedded = original
        .windows(child.len())
        .position(|bytes| bytes == child)
        .unwrap();
    let current = v2::Header::new()
        .messages([Message::reference_count(2), comment])
        .continuation(at + embedded, &child, Widths::EIGHT)
        .build();
    assert_eq!(current.len(), original.len());
    let mut engine = edit_test_util::open_engine(&path);
    let mut bounds = publication_bounds(&mut engine, &[], &["a"]);
    let expected = bounds.updates()[0].allocation().range();
    let field = bounds.updates()[0].field();
    bounds
        .phase
        .engine
        .write_at(u64::try_from(at).unwrap(), &current)
        .unwrap();
    let Err(CountPreparationError::Allocation {
        object,
        expected: actual_expected,
        actual,
    }) = bounds.prepare_counts()
    else {
        panic!("expected overlapping chunk allocations")
    };
    assert_eq!(object.header_address().get(), u64::try_from(at).unwrap());
    assert_eq!(actual_expected, expected);
    assert_eq!(actual, None);
    assert_eq!(bounds.updates()[0].field(), field);
}

#[rstest]
#[case::four(Widths::FOUR)]
#[case::eight(Widths::EIGHT)]
fn current_observations_keep_encoded_widths_and_physical_fields(
    #[case] widths: Widths,
    #[values(0, 512)] base: u64,
) {
    let child = v2::continuation_chunk(&[Message::reference_count(7)], v2::HeaderFlags::default());
    let child_at = 64_usize;
    let root_at = child_at + child.len();
    let header = v2::Header::new()
        .continuation(child_at, &child, widths)
        .build();
    let mut bytes = vec![0; usize::try_from(base).unwrap() + root_at];
    let physical_child = base + u64::try_from(child_at).unwrap();
    bytes[convert::slice_range(physical_child, u64::try_from(child.len()).unwrap()).unwrap()]
        .copy_from_slice(&child);
    bytes.extend_from_slice(&header);
    let source = SourceMetadata(&BytesSource(bytes));
    let object = ObjectId::new(PhysicalAddress::new(base + u64::try_from(root_at).unwrap()));
    let (count, allocation) = CountObservation::read_source(
        &source,
        object,
        FormatWidths::from_sizes(
            u8::try_from(widths.offset).unwrap(),
            u8::try_from(widths.length).unwrap(),
        )
        .unwrap(),
        BaseAddress::new(base),
    )
    .unwrap()
    .into_parts();
    let ObjectReferenceCount::Version2(field) = count else {
        panic!("expected continuation count")
    };
    assert_eq!(field.value(), 7);
    assert_eq!(
        field.chunk(),
        FileRange::new(
            PhysicalAddress::new(physical_child),
            u64::try_from(child.len()).unwrap()
        )
        .unwrap()
    );
    assert_eq!(allocation.unwrap().range(), field.chunk());
    assert!(field.range().start() > field.chunk().start());
    assert!(field.range().end() < field.chunk().end());
}

test_util_hdf5::hdf5_1_10_tests! {
    mod native_crosschecks;
}

#[rstest]
#[case::original(Association::Original)]
#[case::replacement_root(Association::ReplacementRoot)]
#[case::other_session(Association::OtherSession)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn preparation_preserves_source_facts_across_the_engine_loan(#[case] association: Association) {
    let path = test_util::temp::temp_path("count-preparation-loan.h5");
    let other_path = test_util::temp::temp_path("count-preparation-other-loan.h5");
    let fixture = count_test_util::fixture(
        &path,
        Form::Version2,
        0,
        NonZeroU32::new(2).unwrap(),
        c"old",
    );
    std::fs::copy(&path, &other_path).unwrap();
    let (mut engine, calls) = protection_tests::engine_with_read_call_counter(&path);
    let (mut other, other_calls) = protection_tests::engine_with_read_call_counter(&other_path);
    let (counts, _) = protection_tests::counts(&engine, &["a"]);
    let counts = counts.protect(&engine, []).unwrap();
    let original_root = counts.root().unwrap();
    assert!(calls.load(std::sync::atomic::Ordering::Relaxed) > 0);
    if matches!(association, Association::ReplacementRoot) {
        let spans = engine
            .oh_chunk_spans(original_root.header_address())
            .unwrap();
        let [range] = spans.as_slice() else {
            panic!("expected one original root allocation")
        };
        let bytes = engine
            .image()
            .read_exact_at(
                range.start().get(),
                usize::try_from(range.length()).unwrap(),
            )
            .unwrap();
        let root = engine.place_object_header(&bytes).unwrap().header_address();
        assert_ne!(root, original_root.header_address());
        engine.publish_nonpersisting(root, None).unwrap();
    }
    let (receiver, calls) = if matches!(association, Association::OtherSession) {
        (&mut other, &other_calls)
    } else {
        (&mut engine, &calls)
    };
    let before = receiver.disk_bytes().unwrap();
    calls.store(0, std::sync::atomic::Ordering::Relaxed);
    DISCOVERY_WORK.with(|work| work.set((0, 0)));
    let (result, measured) =
        allocation::measure(&format!("prepare-count-loan-{association:?}"), || {
            CurrentCountUpdates::new(receiver, &counts)
        });
    if matches!(association, Association::OtherSession) {
        let Err(error) = result else {
            panic!("expected another allocation session")
        };
        assert!(matches!(error, CountPreparationError::Session), "{error:?}");
        assert_eq!(measured.blocks, 0, "{measured}");
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    } else {
        let prepared = result.unwrap();
        assert!(std::ptr::eq(prepared.counts, &counts));
        assert_eq!(prepared.counts.root(), Some(original_root));
        let (original, update) = prepared.updates().next().unwrap();
        assert_eq!(original.object(), fixture.object);
        assert_eq!(original.allocation().range(), fixture.allocation);
        assert_eq!(original.field().value(), 2);
        assert_eq!(original.count().get(), 1);
        let mut patched = Form::Version2.chunk(2, c"old");
        for (range, bytes) in update.writes() {
            patched[convert::slice_range(
                range.start().get() - fixture.allocation.start().get(),
                range.length(),
            )
            .unwrap()]
            .copy_from_slice(bytes);
        }
        assert_eq!(patched, Form::Version2.chunk(1, c"old"));
        let reads = calls.load(std::sync::atomic::Ordering::Relaxed);
        assert!(reads > 0);
        assert_eq!(prepared.updates().len(), 1);
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), reads);
        assert!(measured.blocks > 0 && measured.live_bytes > 0, "{measured}");
        assert!(
            measured.live_bytes
                <= u64::try_from(core::mem::size_of::<ObjectReferenceCountUpdate>()).unwrap(),
            "{measured}"
        );
        assert_eq!(prepared.engine.disk_bytes().unwrap(), before);
    }
    assert_eq!(receiver.disk_bytes().unwrap(), before);
    assert_eq!(counts.root(), Some(original_root));
    DISCOVERY_WORK.with(|work| assert_eq!(work.get(), (0, 0)));
}

fn publication_bounds<'a>(
    engine: &'a mut WriteEngine,
    retired: &[(FileRange, FreeClass)],
    names: &[&str],
) -> CountPublicationBounds<'a> {
    let (counts, _) = protection_tests::counts(engine, names);
    let counts = counts.protect(engine, []).unwrap();
    let end = PhysicalAddress::new(engine.image().len());
    let root = engine
        .addresses()
        .resolve(engine.superblock().root_group_address)
        .unwrap();
    let pending = RootRetirement::new(root, vec![])
        .into_pending_frees(retired.to_vec(), engine.image().len());
    counts
        .before_publication(engine, pending)
        .unwrap()
        .at_end(end)
        .unwrap_or_else(|failure| panic!("{:?}", failure.error))
}

#[derive(Clone, Copy)]
enum Change {
    Count,
    Duplicate,
    Moved,
    Removed,
    Resized,
    Shared,
}

#[derive(Clone, Copy)]
enum ReadFailure {
    AllocationEof,
    Ancestor,
    Initial,
}

struct FailingSource<'a, S: Source + ?Sized> {
    source: &'a S,
    offset: u64,
}

impl<S: Source + ?Sized> Source for FailingSource<'_, S> {
    fn len(&self) -> u64 {
        self.source.len()
    }
    fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        if offset <= self.offset && self.offset < offset + u64::try_from(bytes.len()).unwrap() {
            Err(FormatError::Source("injected metadata read".into()))
        } else {
            self.source.read_at(offset, bytes)
        }
    }
}

struct MislocatedSource<'a>(&'a [u8]);

impl Source for MislocatedSource<'_> {
    fn len(&self) -> u64 {
        u64::try_from(self.0.len()).unwrap()
    }
    fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        BytesSource(self.0).read_at(offset, bytes)
    }
    fn read_metadata(&self, offset: u64, length: usize) -> Result<MetadataBytes<'_>, FormatError> {
        let length = u64::try_from(length).unwrap();
        MetadataBytes::borrowed(
            FileRange::new(PhysicalAddress::new(offset + 1), length)?,
            &self.0[convert::slice_range(offset, length)?],
        )
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[derive(Clone, Copy, Debug)]
enum Association {
    Original,
    OtherSession,
    ReplacementRoot,
}
