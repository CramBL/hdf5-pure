use std::fs::{self, OpenOptions};
use std::io::{Error as IoError, ErrorKind};
use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use rstest::rstest;
use test_util::disk_log::{self, DiskOp, IoEvent};

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
    disk_log::start();
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
    let issued = disk_log::take();
    let direct = matches!(buffering, Buffering::Unbuffered) || length >= 2 * ZERO_BATCH_BYTES;
    if direct {
        assert_eq!(before, expected);
        assert_eq!(issued.len(), 1 + length.div_ceil(ZERO_BATCH_BYTES));
    } else {
        assert_eq!(before, initial);
        assert!(issued.is_empty());
    }

    disk_log::start();
    image.ordering_barrier().unwrap();
    if matches!(buffering, Buffering::Session) {
        assert_eq!(image.disk_bytes().unwrap(), before);
    } else {
        assert_eq!(image.disk_bytes().unwrap(), expected);
    }
    image.flush().unwrap();
    assert_eq!(image.disk_bytes().unwrap(), expected);
    let writes = disk_log::take();
    if direct {
        assert!(writes.is_empty());
    } else {
        let end = if length == 0 { 1 } else { start + length };
        assert_eq!(
            writes,
            [DiskOp::Write {
                offset: 0,
                bytes: expected[..end].to_vec()
            }]
        );
    }
    assert_eq!(
        image.as_slice().map(|mirror| mirror.len()),
        matches!(backing, Backing::Mirrored).then_some(expected.len())
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
    disk_log::start();
    let error = image.fill_zeros(range(start, length)).unwrap_err();
    assert!(
        matches!(error, Error::Format(FormatError::UnexpectedSourceEof { expected, available: 16 }) if expected == start + length),
        "{error:?}"
    );
    assert_eq!(image.len(), 16);
    assert_eq!(image.disk_bytes().unwrap(), [7; 16]);
    assert!(disk_log::take().is_empty());
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
    disk_log::start();
    disk_log::fail_write_after(1, prefix);
    let error = image
        .fill_zeros(range(start as u64, 3 * ZERO_BATCH_BYTES as u64))
        .unwrap_err();
    assert_injected(error, "injected disk write failure");
    assert_eq!(
        image.len(),
        (initial.len() + if append { ZERO_BATCH_BYTES } else { 0 }) as u64
    );
    let issued = disk_log::finish();
    assert_eq!(
        issued
            .events
            .iter()
            .filter(|event| matches!(event, IoEvent::Write { .. }))
            .count(),
        2
    );
    assert_eq!(
        issued.operations[0],
        DiskOp::Write {
            offset: start as u64,
            bytes: vec![0; ZERO_BATCH_BYTES]
        }
    );
    disk_log::start();
    image.flush().unwrap();
    assert!(disk_log::take().is_empty());
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
fn buffered_zero_rewrites_issue_one_contiguous_run(#[case] backing: Backing) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("copies.bin");
    let mut image = image(&path, &[], backing);
    let length = ZERO_BATCH_BYTES * 8;
    image
        .set_write_buffering(Buffering::Session.mode(length * 2))
        .unwrap();
    image.fill_zeros(range(0, length as u64)).unwrap();
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let region = format!("buffered_zero_rewrite_{backing:?}");
        let ((), rewrite) = test_util::allocation::measure(&region, || {
            image.fill_zeros(range(1, (length - 2) as u64)).unwrap();
        });
        assert!(rewrite.bytes <= 2 * ZERO_BATCH_BYTES as u64, "{rewrite}");
        assert!(rewrite.live_bytes < ZERO_BATCH_BYTES as u64, "{rewrite}");
        let (control, copy) =
            test_util::allocation::measure(&format!("{region}_control"), || vec![0_u8; length]);
        assert_eq!(control.len(), length);
        assert!(copy.peak_bytes >= length as u64);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    image.fill_zeros(range(1, (length - 2) as u64)).unwrap();

    assert!(image.disk_bytes().unwrap().is_empty());
    disk_log::start_write_ranges(1);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let region = format!("buffered_zero_flush_{backing:?}");
        let ((), flushed) = test_util::allocation::measure(&region, || image.flush().unwrap());
        assert!(flushed.peak_bytes < ZERO_BATCH_BYTES as u64, "{flushed}");
        assert!(flushed.live_bytes < ZERO_BATCH_BYTES as u64, "{flushed}");
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    image.flush().unwrap();
    assert_eq!(disk_log::take_write_ranges(), [(0, length as u64)]);
    drop(image);
    assert_eq!(fs::read(&path).unwrap(), vec![0; length]);
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
    image.fill_zeros(range(u64::MAX, 0)).unwrap();
    assert_eq!(image.disk_bytes().unwrap(), [7; 16]);
    disk_log::start();
    image.flush().unwrap();
    assert_eq!(
        disk_log::take(),
        [DiskOp::Write {
            offset: 0,
            bytes: vec![9]
        }]
    );
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
    assert_eq!(image.read_exact_at(8, 16).unwrap(), [0; 16]);
    let mut expected = vec![7; 32];
    expected[8..8 + prefix].fill(0);
    assert_eq!(image.disk_bytes().unwrap(), expected);
    disk_log::start();
    image.flush().unwrap();
    assert_eq!(
        disk_log::take(),
        [DiskOp::Write {
            offset: 8,
            bytes: vec![0; 16]
        }]
    );
    expected[8..24].fill(0);
    assert_eq!(image.disk_bytes().unwrap(), expected);
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
