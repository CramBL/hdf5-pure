use std::fs::{self, OpenOptions};

use rstest::rstest;
use test_util::snapshot::text::{Report, Value};

use super::ImageWork;
use crate::image::{FileImage, HandleImage, MirrorImage, WriteBuffering};
use crate::source::MetadataCacheConfig;

#[derive(Clone, Copy)]
enum Backing {
    Bounded,
    Mirrored,
}

#[rstest]
#[case::bounded(Backing::Bounded, "foundation_buffering_bounded")]
#[case::mirrored(Backing::Mirrored, "foundation_buffering_mirrored")]
fn foundation_buffering_observations_preserve_clean_gaps(
    #[case] backing: Backing,
    #[case] snapshot: &str,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("buffering.bin");
    let initial: Vec<u8> = (0_u8..128).collect();
    fs::write(&path, &initial).unwrap();
    let handle = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    let mut image: Box<dyn FileImage> = match backing {
        Backing::Bounded => Box::new(HandleImage::new(
            handle,
            initial.len() as u64,
            MetadataCacheConfig::disabled(),
        )),
        Backing::Mirrored => Box::new(MirrorImage::new(handle, initial.clone())),
    };
    image
        .set_write_buffering(WriteBuffering::Session {
            page_size: 4096,
            max_bytes: 8,
        })
        .unwrap();
    assert_eq!(image.buffering_work().write_calls, 0);
    image.write_at(2, &[201, 202]).unwrap();
    image.write_at(42, &[203, 204]).unwrap();
    let pending = image.buffering_work();
    assert_eq!(pending.pending_runs, 2);
    assert_eq!(pending.pending_bytes, 4);
    assert!(pending.pending_capacity >= pending.pending_bytes);
    assert_eq!(pending.write_calls, 0);
    image.flush().unwrap();
    let flushed = image.buffering_work();
    assert_eq!(flushed.pending_runs, 0);
    assert_eq!(flushed.pending_bytes, 0);
    assert_eq!(flushed.clean_gap_calls, 1);
    assert_eq!(flushed.clean_gap_bytes, 38);
    assert_eq!(flushed.write_calls, 1);
    assert_eq!(flushed.write_bytes, 42);
    assert_eq!(flushed.flush_peak_bytes, 44);
    assert!(flushed.flush_peak_bytes > 8);
    assert!(flushed.flush_peak_capacity >= flushed.flush_peak_bytes);
    assert_eq!(flushed.copied_bytes, 6);
    let mut report = Report::new();
    report.metrics(
        "Buffering configuration",
        "Bytes",
        [("Pending budget", 8), ("File growth", 0)],
    );
    let pending = pending.values();
    let flushed = flushed.values();
    let metrics = [
        "Pending runs (count)",
        "Pending bytes",
        "Pending capacity (bytes)",
        "Flush peak (bytes)",
        "Flush peak capacity (bytes)",
        "Clean-gap reads (calls)",
        "Clean-gap reads (bytes)",
        "Explicit buffer copies (bytes)",
        "Issued writes (calls)",
        "Issued writes (bytes)",
        "Mirror bytes",
        "Mirror capacity (bytes)",
    ];
    report.table(
        "Buffering observations",
        ["Metric", "Pending", "Flushed"],
        metrics.into_iter().enumerate().map(|(index, metric)| {
            [
                Value::text(metric),
                Value::Number(pending[index]),
                Value::Number(flushed[index]),
            ]
        }),
    );
    let report = report.finish();
    drop(image);
    let mut expected = initial;
    expected[2..4].copy_from_slice(&[201, 202]);
    expected[42..44].copy_from_slice(&[203, 204]);
    assert_eq!(fs::read(&path).unwrap(), expected);
    insta::assert_snapshot!(snapshot, report);
}

impl ImageWork {
    /// Returns values in the order of the report's metric rows.
    fn values(self) -> [u64; 12] {
        let Self {
            pending_runs,
            pending_bytes,
            pending_capacity,
            flush_peak_bytes,
            flush_peak_capacity,
            clean_gap_calls,
            clean_gap_bytes,
            copied_bytes,
            write_calls,
            write_bytes,
            mirror_bytes,
            mirror_capacity,
        } = self;
        [
            pending_runs,
            pending_bytes,
            pending_capacity,
            flush_peak_bytes,
            flush_peak_capacity,
            clean_gap_calls,
            clean_gap_bytes,
            copied_bytes,
            write_calls,
            write_bytes,
            mirror_bytes,
            mirror_capacity,
        ]
    }
}
