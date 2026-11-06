use core::cell::Cell;
use core::num::NonZeroU8;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::object_header::v1::Header as HeaderV1;
use test_util::object_header::v2::Header as HeaderV2;

use super::{READ_BUFFER_LEN, ResidualContents, ResidualOutcome};
use crate::access_mode::AccessMode;
use crate::address::{AddressContext, BaseAddress, BaseAddressExt};
use crate::edit::analysis::AnalysisPhase;
use crate::edit::test_util as edit_test_util;
use crate::error::{Error, FormatError};
use crate::object_graph::ObjectId;
use crate::object_header::ObjectHeader;
use crate::source::{BytesSource, PhysicalSource, Source};
use crate::{File, FileBuilder, UserblockSize};

#[rstest]
#[case::empty(1, 0, None)]
#[case::one(1, 1, None)]
#[case::zero_batches(3, 16_385, None)]
#[case::first(1, 16_385, Some(0))]
#[case::buffer_end(3, 16_385, Some(8_191))]
#[case::buffer_start(511, 16_385, Some(8_192))]
#[case::last(1, 16_385, Some(16_384))]
#[case::address_limit(u64::MAX - 8_193, 8_193, None)]
fn content_observations_keep_zero_prefixes_and_unexamined_suffixes(
    #[case] start: u64,
    #[case] length: u64,
    #[case] nonzero: Option<u64>,
) {
    let source = SyntheticSource::new(start + length, nonzero.map(|at| start + at), None);
    let domain = range(start, length);
    let contents = ResidualContents::read(&physical(&source), &[domain]).unwrap();
    let [observed] = contents.intervals() else {
        panic!("expected one residual interval");
    };
    assert_eq!(observed.range(), domain);
    let expected_calls = nonzero.map_or(length.div_ceil(8_192), |at| at / 8_192 + 1);
    assert_eq!(
        (observed.read_calls(), observed.requested_bytes()),
        (expected_calls, (expected_calls * 8_192).min(length))
    );
    match nonzero {
        None => {
            assert_eq!(observed.outcome(), &ResidualOutcome::AllZero);
            assert_eq!(observed.zero_prefix(), domain);
            assert_eq!(observed.unexamined(), range(start + length, 0));
        }
        Some(at) => {
            assert_eq!(
                observed.outcome(),
                &ResidualOutcome::NonZero {
                    address: PhysicalAddress::new(start + at),
                    value: NonZeroU8::new(0x5a).unwrap()
                }
            );
            assert_eq!(observed.zero_prefix(), range(start, at));
            assert_eq!(
                observed.unexamined(),
                range(start + at + 1, length - at - 1)
            );
        }
    }
    let (calls, bytes, peak) = source.work.get();
    assert_eq!(
        (calls, bytes),
        (observed.read_calls(), observed.requested_bytes())
    );
    assert!(peak <= READ_BUFFER_LEN);
}

#[test]
fn read_failures_keep_completed_prefixes_and_other_intervals() {
    let source = SyntheticSource::new(20_003, None, Some(8_192));
    let contents =
        ResidualContents::read(&physical(&source), &[range(0, 16_387), range(20_000, 3)]).unwrap();
    let [failed, complete] = contents.intervals() else {
        panic!("expected two residual intervals");
    };
    assert_eq!(
        failed.outcome(),
        &ResidualOutcome::ReadFailed(FormatError::Source("injected residual read failure".into()))
    );
    assert_eq!(failed.zero_prefix(), range(0, 8_192));
    assert_eq!(failed.unexamined(), range(8_192, 8_195));
    assert_eq!((failed.read_calls(), failed.requested_bytes()), (2, 16_384));
    assert_eq!(complete.outcome(), &ResidualOutcome::AllZero);
    assert_eq!(complete.zero_prefix(), range(20_000, 3));
    assert_eq!(complete.unexamined(), range(20_003, 0));
    assert_eq!(source.work.get(), (3, 16_387, 8_192));
}

#[rstest]
#[case::version_one(HeaderV1::new().build(), OBJECT_HEADER_VERSION_ONE)]
#[case::version_two(HeaderV2::new().build(), OBJECT_HEADER_VERSION_TWO)]
fn supported_headers_at_unaligned_and_buffer_boundaries_remain_nonzero(
    #[case] header: Vec<u8>,
    #[case] version: u8,
    #[values(1, 8_191)] at: usize,
) {
    let mut bytes = vec![0; at + header.len() + 3];
    bytes[at..at + header.len()].copy_from_slice(&header);
    let parsed = ObjectHeader::parse(&bytes, AccessMode::ReadOnly, at, 8, 8).unwrap();
    assert_eq!(parsed.version, version);
    let source = BytesSource::new(&bytes);
    let contents =
        ResidualContents::read(&physical(&source), &[range(0, bytes.len() as u64)]).unwrap();
    assert_eq!(contents.intervals()[0].zero_prefix(), range(0, at as u64));
    assert_eq!(
        contents.intervals()[0].outcome(),
        &ResidualOutcome::NonZero {
            address: PhysicalAddress::new(at as u64),
            value: NonZeroU8::new(header[0]).unwrap()
        }
    );
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::small(8_193)]
#[case::large(8_388_609)]
fn residual_reads_bound_scratch_and_retained_state(#[case] length: u64) {
    use test_util::allocation;

    let source = SyntheticSource::new(length, None, None);
    let (contents, measured) = allocation::measure("residual_content_scan", || {
        ResidualContents::read(&physical(&source), &[range(0, length)]).unwrap()
    });
    assert_eq!(contents.intervals()[0].outcome(), &ResidualOutcome::AllZero);
    assert_eq!(
        source.work.get(),
        (length.div_ceil(8_192), length, READ_BUFFER_LEN)
    );
    assert!(
        measured.blocks > 0
            && measured.bytes > 0
            && measured.bytes < 4_096
            && measured.peak_bytes < 4_096
            && measured.live_bytes < 4_096,
        "{measured}"
    );
    let (control, positive) =
        allocation::measure("residual_content_control", || vec![0_u8; 1_048_576]);
    assert_eq!(control.len(), 1_048_576);
    assert!(positive.bytes >= 1_048_576);
}

#[rstest]
#[case::plain(0)]
#[case::userblock(512)]
fn phase_content_queries_skip_numeric_payloads_and_reexamine_changed_bytes(#[case] base: u64) {
    let path = test_util::temp::temp_path("residual-content-phase.h5");
    let mut builder = FileBuilder::new();
    builder.with_userblock(UserblockSize::try_from(base).unwrap());
    builder
        .create_dataset("data")
        .with_u8_data(&vec![0x5a; 1_048_576]);
    builder.write(&path).unwrap();
    let requested_bytes = Arc::new(AtomicU64::new(0));
    let (mut engine, calls) =
        edit_test_util::open_counted_engine_with_bytes(&path, Arc::clone(&requested_bytes));
    let mut phase = AnalysisPhase::new(&engine);
    let root =
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap();
    let source = phase.reconstruction_source(&engine, root).unwrap();
    let coverage = phase.physical_coverage(&source).unwrap();
    assert!(coverage.content.is_none());
    let ranges = coverage.residuals().to_vec();
    assert_eq!(ranges.len(), 1);
    calls.store(0, Ordering::Relaxed);
    requested_bytes.store(0, Ordering::Relaxed);
    let contents = phase.residual_content(&source).unwrap();
    assert_eq!(
        contents
            .intervals()
            .iter()
            .map(|entry| entry.range())
            .collect::<Vec<_>>(),
        ranges
    );
    assert_eq!(contents.intervals()[0].outcome(), &ResidualOutcome::AllZero);
    assert_eq!(requested_bytes.load(Ordering::Relaxed), ranges[0].length());
    assert_eq!(
        calls.load(Ordering::Relaxed),
        ranges[0].length().div_ceil(8_192)
    );
    assert!(requested_bytes.load(Ordering::Relaxed) < 4_096);
    let pointer = core::ptr::from_ref(contents);
    assert_eq!(
        core::ptr::from_ref(phase.residual_content(&source).unwrap()),
        pointer
    );
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let ((), cached) = test_util::allocation::measure("residual_content_cached", || {
            assert_eq!(
                core::ptr::from_ref(phase.residual_content(&source).unwrap()),
                pointer
            );
        });
        assert_eq!((cached.blocks, cached.bytes), (0, 0));
    }
    let before = calls.load(Ordering::Relaxed);
    let mut foreign = AnalysisPhase::new(&engine);
    let foreign_source = foreign.reconstruction_source(&engine, root).unwrap();
    let Error::Format(error) = phase.residual_content(&foreign_source).err().unwrap() else {
        panic!("expected foreign phase error");
    };
    assert_eq!(
        error,
        FormatError::Internal("physical claims belong to another source phase".into())
    );
    assert_eq!(calls.load(Ordering::Relaxed), before);
    drop(foreign);
    drop(phase);
    let snapshot = engine.snapshot_free();
    engine
        .image
        .write_at(ranges[0].start().get(), &[0x5a])
        .unwrap();
    engine.restore_free(snapshot);
    let mut changed = AnalysisPhase::new(&engine);
    let fresh = changed.reconstruction_source(&engine, root).unwrap();
    let error = changed.residual_content(&source).err().unwrap();
    assert!(
        matches!(error, Error::Format(FormatError::Internal(ref message)) if message == "physical claims belong to another source phase"),
        "{error:?}"
    );
    let contents = changed.residual_content(&fresh).unwrap();
    assert_eq!(
        contents.intervals()[0].outcome(),
        &ResidualOutcome::NonZero {
            address: ranges[0].start(),
            value: NonZeroU8::new(0x5a).unwrap()
        }
    );
    drop(changed);
    engine
        .image
        .write_at(ranges[0].start().get(), &[0])
        .unwrap();
    let mut restored = AnalysisPhase::new(&engine);
    let fresh = restored.reconstruction_source(&engine, root).unwrap();
    assert_eq!(
        restored.residual_content(&fresh).unwrap().intervals()[0].outcome(),
        &ResidualOutcome::AllZero
    );
    drop(restored);
    drop(engine);
    let reader = File::open(&path).unwrap();
    assert_eq!(
        reader.dataset("data").unwrap().read_u8().unwrap(),
        vec![0x5a; 1_048_576]
    );
    reader.close().unwrap();
}

fn range(start: u64, length: u64) -> FileRange {
    FileRange::new(PhysicalAddress::new(start), length).unwrap()
}
fn physical<S: Source + ?Sized>(source: &S) -> PhysicalSource<'_, S> {
    PhysicalSource::for_test(
        source,
        AddressContext::from_effective_base(BaseAddress::ZERO),
    )
}

struct SyntheticSource {
    length: u64,
    nonzero: Option<u64>,
    failure: Option<u64>,
    work: Cell<(u64, u64, usize)>,
}
impl SyntheticSource {
    fn new(length: u64, nonzero: Option<u64>, failure: Option<u64>) -> Self {
        Self {
            length,
            nonzero,
            failure,
            work: Cell::new((0, 0, 0)),
        }
    }
}
impl Source for SyntheticSource {
    fn len(&self) -> u64 {
        self.length
    }
    fn read_at(&self, offset: u64, bytes: &mut [u8]) -> Result<(), FormatError> {
        let Self {
            length,
            nonzero,
            failure,
            work,
        } = self;
        let end = offset.checked_add(bytes.len() as u64).unwrap();
        if end > *length {
            return Err(FormatError::UnexpectedSourceEof {
                expected: end,
                available: *length,
            });
        }
        let (calls, requested, peak) = work.get();
        work.set((
            calls + 1,
            requested + bytes.len() as u64,
            peak.max(bytes.len()),
        ));
        bytes.fill(0);
        if *failure == Some(offset) {
            bytes.fill(0xff);
            return Err(FormatError::Source("injected residual read failure".into()));
        }
        if let Some(at) = nonzero.filter(|&at| offset <= at && at < end) {
            bytes[usize::try_from(at - offset).unwrap()] = 0x5a;
        }
        Ok(())
    }
}

/// Expected version byte in "Version 1 Data Object Header Prefix" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_one
const OBJECT_HEADER_VERSION_ONE: u8 = 1;
/// Expected version byte in "Version 2 Data Object Header Prefix" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_two
const OBJECT_HEADER_VERSION_TWO: u8 = 2;
