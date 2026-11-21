use std::fs::{self, OpenOptions};

use rstest::rstest;
use test_util::disk_log::{self, DiskOp};
use test_util::snapshot::text::Report;

use crate::image::{FileImage, HandleImage, MirrorImage, WriteBuffering};
use crate::source::MetadataCacheConfig;

#[derive(Clone, Copy, Debug)]
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
    let mut image = buffered_image(&path, &initial, backing);
    let mut expected = initial.clone();
    expected[2..4].copy_from_slice(&[201, 202]);
    expected[42..44].copy_from_slice(&[203, 204]);
    disk_log::start();
    image.write_at(2, &[201, 202]).unwrap();
    image.write_at(42, &[203, 204]).unwrap();
    assert!(disk_log::take().is_empty());
    assert_eq!(image.disk_bytes().unwrap(), initial);

    disk_log::start();
    disk_log::start_read_ranges();
    image.flush().unwrap();
    assert_eq!(disk_log::take_read_ranges(), [(4, 38)]);
    assert_eq!(
        disk_log::take(),
        [DiskOp::Write {
            offset: 2,
            bytes: expected[2..44].to_vec()
        }]
    );
    drop(image);
    assert_eq!(fs::read(&path).unwrap(), expected);
    let mut report = Report::new();
    report.metrics(
        "Buffering configuration",
        "Bytes",
        [("Pending budget", 8), ("File growth", 0)],
    );
    report.metrics(
        "Issued span",
        "Bytes",
        [("Offset", 2), ("Length", 42), ("Preserved gap", 38)],
    );
    insta::assert_snapshot!(snapshot, report.finish());
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::bounded(Backing::Bounded)]
#[case::mirrored(Backing::Mirrored)]
fn flushing_sparse_writes_bounds_clean_gap_scratch(#[case] backing: Backing) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("buffering.bin");
    let initial: Vec<u8> = (0_u8..128).collect();
    let mut image = buffered_image(&path, &initial, backing);
    image.write_at(2, &[201, 202]).unwrap();
    image.write_at(42, &[203, 204]).unwrap();

    let region = format!("flush_clean_gap_{backing:?}");
    let ((), flushed) = test_util::allocation::measure(&region, || image.flush().unwrap());
    let (control, copied) =
        test_util::allocation::measure(&format!("{region}_control"), || initial.clone());
    assert_eq!(control, initial);
    assert!(copied.peak_bytes >= 128, "{copied}");
    assert!(flushed.peak_bytes < 192, "{flushed}");
    assert!(flushed.bytes < 224, "{flushed}");

    let mut expected = initial;
    expected[2..4].copy_from_slice(&[201, 202]);
    expected[42..44].copy_from_slice(&[203, 204]);
    drop(image);
    assert_eq!(fs::read(&path).unwrap(), expected);
}

fn buffered_image(path: &std::path::Path, initial: &[u8], backing: Backing) -> Box<dyn FileImage> {
    fs::write(path, initial).unwrap();
    let handle = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let mut image: Box<dyn FileImage> = match backing {
        Backing::Bounded => Box::new(HandleImage::new(
            handle,
            initial.len() as u64,
            MetadataCacheConfig::disabled(),
        )),
        Backing::Mirrored => Box::new(MirrorImage::new(handle, initial.to_vec())),
    };
    image
        .set_write_buffering(WriteBuffering::Session {
            page_size: 4096,
            max_bytes: 8,
        })
        .unwrap();
    image
}
