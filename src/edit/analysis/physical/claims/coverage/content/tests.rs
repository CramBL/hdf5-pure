use core::cell::Cell;
use core::num::NonZeroU8;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::object_header::v1::Header as HeaderV1;
use test_util::object_header::v2::Header as HeaderV2;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use super::READ_BUFFER_LEN;
use super::{ResidualContents, ResidualOutcome};
use crate::access_mode::AccessMode;
use crate::address::{AddressContext, BaseAddress, BaseAddressExt};
use crate::error::FormatError;
use crate::object_header::ObjectHeader;
use crate::source::{BytesSource, PhysicalSource, Source};

#[rstest]
#[case::empty(1, 0, None)]
#[case::one(1, 1, None)]
#[case::zero_batches(3, 16_385, None)]
#[case::first(1, 16_385, Some(0))]
#[case::buffer_end(3, 16_385, Some(8_191))]
#[case::buffer_start(511, 16_385, Some(8_192))]
#[case::last(1, 16_385, Some(16_384))]
#[case::address_limit(u64::MAX - 8_193, 8_193, None)]
#[case::empty_at_limit(u64::MAX, 0, None)]
#[case::nonzero_at_limit(u64::MAX - 8_193, 8_193, Some(8_192))]
fn residual_scans_stop_at_the_first_nonzero_byte(
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
    match nonzero {
        None => {
            assert_eq!(observed.outcome(), &ResidualOutcome::AllZero);
        }
        Some(at) => {
            assert_eq!(
                observed.outcome(),
                &ResidualOutcome::NonZero {
                    address: PhysicalAddress::new(start + at),
                    value: NonZeroU8::new(0x5a).unwrap()
                }
            );
        }
    }
    let (calls, bytes, peak) = source.work.get();
    assert_eq!(
        (calls, bytes),
        (expected_calls, (expected_calls * 8_192).min(length))
    );
    assert_eq!(peak, usize::try_from(length.min(8_192)).unwrap());
}

#[test]
fn read_failures_preserve_other_intervals() {
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
    assert_eq!(complete.outcome(), &ResidualOutcome::AllZero);
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
    let (scan_region, control_region) = if length == 8_193 {
        (
            "residual_content_small_scan",
            "residual_content_small_control",
        )
    } else {
        (
            "residual_content_large_scan",
            "residual_content_large_control",
        )
    };
    let (contents, measured) = allocation::measure(scan_region, || {
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
    let (control, positive) = allocation::measure(control_region, || vec![0_u8; 1_048_576]);
    assert_eq!(control.len(), 1_048_576);
    assert!(positive.bytes >= 1_048_576);
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
