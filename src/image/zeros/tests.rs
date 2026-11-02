use std::fs::{self, OpenOptions};
use std::io::{Error as IoError, ErrorKind};
use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::disk_log;

use super::ZERO_BATCH_BYTES;
use crate::error::{Error, FormatError};
use crate::image::zeros;
use crate::image::{FileImage, HandleImage, MirrorImage, WriteBuffering};
use crate::source::MetadataCacheConfig;

#[rstest]
#[case::empty(0)]
#[case::single_byte(1)]
#[case::small(63)]
#[case::below_batch(ZERO_BATCH_BYTES - 1)]
#[case::one_batch(ZERO_BATCH_BYTES)]
#[case::below_budget(2 * ZERO_BATCH_BYTES - 1)]
#[case::at_budget(2 * ZERO_BATCH_BYTES)]
#[case::several_batches(3 * ZERO_BATCH_BYTES + 7)]
fn zero_ranges_preserve_contents_lengths_and_visibility(
    #[case] length: usize,
    #[values(Backing::Bounded, Backing::Mirrored)] backing: Backing,
    #[values(Buffering::Unbuffered, Buffering::Operation, Buffering::Session)] buffering: Buffering,
    #[values(false, true)] append: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("zeros.bin");
    let initial = vec![7; 4 * ZERO_BATCH_BYTES + 32];
    let mut image = image(&path, &initial, backing);
    image
        .set_write_buffering(buffering.mode(2 * ZERO_BATCH_BYTES))
        .unwrap();
    assert_eq!(image.read_metadata(0, 32).unwrap().as_ref(), &[7; 32]);
    image.write_at(0, &[9]).unwrap();
    let start = if append { initial.len() } else { 16 };
    image
        .fill_zeros(range(start as u64, length as u64))
        .unwrap();
    let mut expected = initial.clone();
    expected[0] = 9;
    expected.resize(expected.len().max(start + length), 0);
    expected[start..start + length].fill(0);
    assert_eq!(image.len(), expected.len() as u64);
    assert_eq!(image.read_exact_at(0, expected.len()).unwrap(), expected);
    assert_eq!(
        image.read_metadata(0, 32).unwrap().as_ref(),
        &expected[..32]
    );
    let before = image.disk_bytes().unwrap();
    if matches!(buffering, Buffering::Unbuffered) || length >= 2 * ZERO_BATCH_BYTES {
        assert_eq!(before, expected);
        assert_eq!(image.buffering_work().pending_bytes, 0);
        assert_eq!(
            image.buffering_work().write_calls,
            1 + length.div_ceil(ZERO_BATCH_BYTES) as u64
        );
    } else {
        assert_eq!(before, initial);
        assert!(image.buffering_work().pending_bytes > 0);
    }
    image.ordering_barrier().unwrap();
    if matches!(buffering, Buffering::Session) {
        assert_eq!(image.disk_bytes().unwrap(), before);
    } else {
        assert_eq!(image.disk_bytes().unwrap(), expected);
    }
    image.flush().unwrap();
    assert_eq!(image.disk_bytes().unwrap(), expected);
    let work = image.buffering_work();
    assert_eq!(work.pending_bytes, 0);
    assert_eq!(
        work.clean_gap_bytes,
        if length > 0
            && length < 2 * ZERO_BATCH_BYTES
            && !matches!(buffering, Buffering::Unbuffered)
        {
            (start - 1) as u64
        } else {
            0
        }
    );
    assert_eq!(
        work.mirror_bytes,
        if matches!(backing, Backing::Mirrored) {
            expected.len() as u64
        } else {
            0
        }
    );
    drop(image);
    assert_eq!(fs::read(&path).unwrap(), expected);
}

#[rstest]
#[case::crosses_end(8, 9)]
#[case::starts_after_end(17, 1)]
fn zero_ranges_reject_growth_outside_append(
    #[case] start: u64,
    #[case] length: u64,
    #[values(Backing::Bounded, Backing::Mirrored)] backing: Backing,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bounds.bin");
    let mut image = image(&path, &[7; 16], backing);
    let error = image.fill_zeros(range(start, length)).unwrap_err();
    assert!(
        matches!(error, Error::Format(FormatError::UnexpectedSourceEof { expected, available: 16 }) if expected == start + length),
        "{error:?}"
    );
    assert_eq!(image.len(), 16);
    assert_eq!(image.disk_bytes().unwrap(), [7; 16]);
    assert_eq!(image.buffering_work().write_calls, 0);
    drop(image);
}

#[rstest]
#[case::empty(0, 0)]
#[case::single_byte(1, 1)]
#[case::several_batches(3 * ZERO_BATCH_BYTES as u64 + 7, 4)]
fn zero_generation_reuses_bounded_scratch(#[case] length: u64, #[case] calls: usize) {
    let mut chunks = Vec::new();
    zeros::emit(range(19, length), |offset, bytes| {
        assert!(bytes.iter().all(|&byte| byte == 0));
        chunks.push((offset, bytes.len(), bytes.as_ptr() as usize));
        Ok(())
    })
    .unwrap();
    assert_eq!(chunks.len(), calls);
    assert_eq!(
        chunks.iter().map(|chunk| chunk.1 as u64).sum::<u64>(),
        length
    );
    assert!(chunks.iter().all(|chunk| chunk.1 <= ZERO_BATCH_BYTES));
    assert!(
        chunks
            .windows(2)
            .all(|pair| pair[0].2 == pair[1].2 && pair[1].0 == pair[0].0 + pair[0].1 as u64)
    );
    if length != 0 {
        assert_eq!(chunks[0].1 as u64, length.min(ZERO_BATCH_BYTES as u64));
    }
}

#[rstest]
#[case::wide_length(0, u64::from(u32::MAX) + 1)]
#[case::near_endpoint(u64::MAX - 3 * ZERO_BATCH_BYTES as u64, 3 * ZERO_BATCH_BYTES as u64)]
fn zero_generation_keeps_geometry_in_u64(#[case] start: u64, #[case] length: u64) {
    let mut calls = 0;
    let error = zeros::emit(range(start, length), |offset, bytes| {
        calls += 1;
        assert_eq!(offset, start);
        assert_eq!(bytes.len(), ZERO_BATCH_BYTES);
        Err(IoError::other("stop after bounded generation").into())
    })
    .unwrap_err();
    assert_injected(error, "stop after bounded generation");
    assert_eq!(calls, 1);
    let error = FileRange::new(PhysicalAddress::new(u64::MAX), 1).unwrap_err();
    assert!(
        matches!(
            error,
            FormatError::OffsetOverflow {
                offset: u64::MAX,
                length: 1
            }
        ),
        "{error:?}"
    );
}

#[rstest]
#[case::before_write(0)]
#[case::partial_write(7)]
#[case::completed_write(usize::MAX)]
fn zero_failure_keeps_successful_prefix_progress(
    #[case] prefix: usize,
    #[values(Backing::Bounded, Backing::Mirrored)] backing: Backing,
    #[values(Buffering::Unbuffered, Buffering::Operation, Buffering::Session)] buffering: Buffering,
    #[values(false, true)] append: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("failure.bin");
    let initial = vec![7; 4 * ZERO_BATCH_BYTES];
    let mut image = image(&path, &initial, backing);
    image
        .set_write_buffering(buffering.mode(ZERO_BATCH_BYTES * 2))
        .unwrap();
    let start = if append { initial.len() } else { 0 };
    disk_log::fail_write_after(1, prefix);
    let error = image
        .fill_zeros(range(start as u64, 3 * ZERO_BATCH_BYTES as u64))
        .unwrap_err();
    assert_injected(error, "injected disk write failure");
    assert_eq!(
        image.len(),
        (initial.len() + if append { ZERO_BATCH_BYTES } else { 0 }) as u64
    );
    assert_eq!(image.buffering_work().write_calls, 1);
    assert_eq!(image.buffering_work().pending_bytes, 0);
    let mut expected = initial;
    let written = ZERO_BATCH_BYTES + prefix.min(ZERO_BATCH_BYTES);
    expected.resize(expected.len().max(start + written), 0);
    expected[start..start + written].fill(0);
    assert_eq!(image.disk_bytes().unwrap(), expected);
    if let Some(mirror) = image.as_slice() {
        assert_eq!(
            &mirror[start..start + ZERO_BATCH_BYTES],
            vec![0; ZERO_BATCH_BYTES]
        );
        if !append {
            assert_eq!(&mirror[ZERO_BATCH_BYTES..], vec![7; 3 * ZERO_BATCH_BYTES]);
        }
    }
    image.discard_pending_writes();
    drop(image);
    assert_eq!(fs::read(&path).unwrap(), expected);
}

#[rstest]
#[case::bounded(Backing::Bounded)]
#[case::mirrored(Backing::Mirrored)]
fn zero_batches_record_one_explicit_pending_copy_per_byte(#[case] backing: Backing) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("copies.bin");
    let mut image = image(&path, &[], backing);
    let length = ZERO_BATCH_BYTES * 8;
    image
        .set_write_buffering(Buffering::Session.mode(length * 2))
        .unwrap();
    image.fill_zeros(range(0, length as u64)).unwrap();
    let extended = image.buffering_work();
    assert_eq!(extended.pending_runs, 1);
    assert_eq!(extended.pending_bytes, length as u64);
    assert!(extended.pending_capacity >= length as u64);
    assert_eq!(extended.copied_bytes, length as u64);
    assert_eq!(extended.write_calls, 0);
    image.fill_zeros(range(1, (length - 2) as u64)).unwrap();
    let contained = image.buffering_work();
    assert_eq!(contained.pending_runs, 1);
    assert_eq!(contained.copied_bytes, (2 * length - 2) as u64);
    assert_eq!(contained.pending_capacity, extended.pending_capacity);
    image.flush().unwrap();
    let flushed = image.buffering_work();
    assert_eq!(flushed.write_calls, 1);
    assert_eq!(flushed.write_bytes, length as u64);
    assert_eq!(flushed.copied_bytes, contained.copied_bytes);
    assert_eq!(flushed.clean_gap_bytes, 0);
    assert_eq!(flushed.flush_peak_bytes, length as u64);
    assert_eq!(image.read_exact_at(0, length).unwrap(), vec![0; length]);
    drop(image);
}

#[rstest]
#[case::bounded(Backing::Bounded)]
#[case::mirrored(Backing::Mirrored)]
fn empty_zero_ranges_leave_pending_writes_unchanged(#[case] backing: Backing) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("empty.bin");
    let mut image = image(&path, &[7; 16], backing);
    image
        .set_write_buffering(Buffering::Session.mode(32))
        .unwrap();
    image.write_at(0, &[9]).unwrap();
    let before = image.buffering_work();
    image.fill_zeros(range(u64::MAX, 0)).unwrap();
    assert_eq!(image.buffering_work(), before);
    assert_eq!(image.disk_bytes().unwrap(), [7; 16]);
    image.discard_pending_writes();
    drop(image);
}

#[rstest]
#[case::before_write(0)]
#[case::partial_write(7)]
fn failed_zero_flush_keeps_pending_ownership(
    #[case] prefix: usize,
    #[values(Backing::Bounded, Backing::Mirrored)] backing: Backing,
    #[values(Buffering::Operation, Buffering::Session)] buffering: Buffering,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("pending.bin");
    let mut image = image(&path, &[7; 32], backing);
    image.set_write_buffering(buffering.mode(64)).unwrap();
    image.fill_zeros(range(8, 16)).unwrap();
    disk_log::fail_write_after(0, prefix);
    assert_injected(image.flush().unwrap_err(), "injected disk write failure");
    assert_eq!(image.len(), 32);
    assert_eq!(image.buffering_work().pending_bytes, 16);
    assert_eq!(image.read_exact_at(8, 16).unwrap(), [0; 16]);
    let mut expected = vec![7; 32];
    expected[8..8 + prefix].fill(0);
    assert_eq!(image.disk_bytes().unwrap(), expected);
    image.flush().unwrap();
    expected[8..24].fill(0);
    assert_eq!(image.disk_bytes().unwrap(), expected);
    assert_eq!(image.buffering_work().pending_bytes, 0);
    drop(image);
    assert_eq!(fs::read(&path).unwrap(), expected);
}

#[derive(Clone, Copy, Debug)]
enum Backing {
    Bounded,
    Mirrored,
}

#[derive(Clone, Copy, Debug)]
enum Buffering {
    Operation,
    Session,
    Unbuffered,
}

impl Buffering {
    fn mode(self, budget: usize) -> WriteBuffering {
        match self {
            Self::Operation => WriteBuffering::Operation {
                page_size: 1 << 24,
                max_bytes: budget,
            },
            Self::Session => WriteBuffering::Session {
                page_size: 1 << 24,
                max_bytes: budget,
            },
            Self::Unbuffered => WriteBuffering::Unbuffered,
        }
    }
}

fn image(path: &Path, initial: &[u8], backing: Backing) -> Box<dyn FileImage> {
    fs::write(path, initial).unwrap();
    let handle = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    match backing {
        Backing::Bounded => Box::new(HandleImage::new(
            handle,
            initial.len() as u64,
            MetadataCacheConfig::new(64 * 1024),
        )),
        Backing::Mirrored => Box::new(MirrorImage::new(handle, initial.to_vec())),
    }
}

fn range(start: u64, length: u64) -> FileRange {
    FileRange::new(PhysicalAddress::new(start), length).unwrap()
}

fn assert_injected(error: Error, message: &str) {
    let Error::Io(error) = error else {
        panic!("expected injected I/O failure, got {error:?}");
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), message);
}
