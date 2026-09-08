//! End-to-end SWMR interop with h5py: hdf5-pure acts as the single writer (with
//! the superblock SWMR flag set) while h5py opens the same file with
//! `swmr=True` and reads the appended data concurrently.
//!
//! Skipped automatically when python3 / h5py are unavailable, so this is an
//! optional interop probe rather than a hard dependency (CI does not install
//! h5py). To run it deterministically, use the version the probe was validated
//! against (h5py 3.16.0 on CPython 3.13):
//!
//!     uv run --with h5py==3.16.0 -- cargo test --test swmr_concurrent
//!
//! `uv run` puts a pinned python3 + h5py on PATH for the duration of the test.

use hdf5_pure::{File, FileBuilder};
use tempfile::tempdir;

/// Open `path` with h5py and return `(len, first, last)` of dataset `d`, or
/// `None` if python3/h5py are unavailable. `mode` is "swmr" or "plain".
fn h5py_read(path: &std::path::Path, mode: &str) -> Option<(usize, i64, i64)> {
    let script = r"
import sys, h5py
path, mode = sys.argv[1], sys.argv[2]
f = h5py.File(path, 'r', swmr=(mode == 'swmr'), libver='latest')
d = f['d']
if mode == 'swmr':
    d.refresh()
n = d.shape[0]
print(n, int(d[0]) if n else 0, int(d[-1]) if n else 0)
";
    let out = std::process::Command::new("python3")
        .args(["-c", script, &path.to_string_lossy(), mode])
        // SWMR readers run without HDF5's file locking so they don't conflict
        // with the writer that still holds the file open.
        .env("HDF5_USE_FILE_LOCKING", "FALSE")
        .output()
        .ok()?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        if err.contains("No module named") {
            return None; // h5py not installed — skip
        }
        panic!("h5py read ({mode}) failed: {err}");
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let mut it = s.split_whitespace();
    let n: usize = it.next()?.parse().ok()?;
    let first: i64 = it.next()?.parse().ok()?;
    let last: i64 = it.next()?.parse().ok()?;
    Some((n, first, last))
}

#[test]
fn h5py_swmr_reads_pure_appends_concurrently() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");

    // Create the dataset (unlimited, chunked) with hdf5-pure.
    {
        let data: Vec<i32> = (0..5).collect();
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&data)
            .with_shape(&[5])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[1]);
        b.write(&path).unwrap();
    }

    // Open as a SWMR writer (sets the superblock SWMR flag) and keep it open
    // while h5py reads concurrently.
    let w = File::open_swmr_writer(&path).unwrap();
    w.dataset("d")
        .unwrap()
        .append(&(5..50).collect::<Vec<_>>())
        .unwrap();

    let Some((n1, first1, last1)) = h5py_read(&path, "swmr") else {
        eprintln!("h5py unavailable; skipping");
        return;
    };
    assert_eq!(
        (n1, first1, last1),
        (50, 0, 49),
        "h5py swmr read after first append"
    );

    // Append more while still open; a fresh h5py swmr reader sees it.
    w.dataset("d")
        .unwrap()
        .append(&(50..200).collect::<Vec<_>>())
        .unwrap();
    let (n2, first2, last2) = h5py_read(&path, "swmr").unwrap();
    assert_eq!(
        (n2, first2, last2),
        (200, 0, 199),
        "h5py swmr read after second append"
    );

    // Clean close clears the SWMR flag; the file then opens normally.
    w.close().unwrap();
    let (n3, first3, last3) = h5py_read(&path, "plain").unwrap();
    assert_eq!(
        (n3, first3, last3),
        (200, 0, 199),
        "h5py plain read after close"
    );
}

/// The superblock SWMR-write flag (0x05) is set while a writer is open and
/// cleared on a clean close (matching h5py/C); `clear_swmr_flag` recovers a
/// file left flagged by a crashed writer.
#[test]
fn swmr_flag_lifecycle() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    {
        let data: Vec<i32> = (0..5).collect();
        let mut b = FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&data)
            .with_shape(&[5])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[1]);
        b.write(&path).unwrap();
    }
    let flag = |p: &std::path::Path| -> u8 {
        let bytes = std::fs::read(p).unwrap();
        let sig = b"\x89HDF\r\n\x1a\n";
        let off = bytes.windows(8).position(|w| w == sig).unwrap();
        bytes[off + 11] // v3 superblock consistency-flags byte
    };

    assert_eq!(
        flag(&path),
        0x00,
        "freshly created file is not SWMR-flagged"
    );
    {
        let w = File::open_swmr_writer(&path).unwrap();
        assert_eq!(flag(&path), 0x05, "flag set while writer is open");
        w.dataset("d").unwrap().append(&[5, 6, 7]).unwrap();
        assert_eq!(flag(&path), 0x05, "flag stays set across appends");
        w.close().unwrap();
    }
    assert_eq!(flag(&path), 0x00, "flag cleared on clean close");

    // Simulate a crashed writer (flag left set), then recover.
    {
        let w = File::open_swmr_writer(&path).unwrap();
        w.dataset("d").unwrap().append(&[8]).unwrap();
        // Skipping Drop leaves the flag set, as a crash would.
        #[expect(clippy::mem_forget, reason = "the test models a writer that crashed")]
        std::mem::forget(w);
    }
    assert_eq!(flag(&path), 0x05, "flag left set after simulated crash");
    File::clear_swmr_flag(&path).unwrap();
    assert_eq!(flag(&path), 0x00, "clear_swmr_flag recovers the file");
}
