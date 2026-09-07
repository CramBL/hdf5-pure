// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
//! Reference-C-library interop for the HDF5 1.8 output format
//! (`FileBuilder::with_libver_bounds` with an upper bound of `LibVer::V18`).
//!
//! The 1.8 format exists because a reader older than 1.10 cannot open what this
//! crate writes by default: MATLAB linked HDF5 1.8.12 before R2021b, and a
//! version 3 superblock is a 1.10 addition it does not understand. What this
//! module checks is that the *older encoding is still a correct file* — the
//! version 2 superblock and version 3 data-layout messages carry the same
//! content, and the current C library reads and writes every part of it.
//!
//! It cannot check the boundary itself. The `hdf5-metno` dev-dependency builds a
//! current libhdf5, which reads both formats without complaint, so nothing here
//! would fail if the bound stopped working. `tests/libver_matrix_crosscheck.rs`
//! covers that half against every release the interop workflow builds, 1.8.23
//! included.
//!
//! Measured there when this landed, so the boundary is observed rather than
//! inferred from the format specification:
//!
//! ```text
//! superblock 3 (the default through 0.33.0)  h5dump: unable to open file, exit 1
//! superblock 2 (the 1.8 format)              h5dump: reads data, groups, attributes
//! ```
//!
//! Note what the first line is not: 1.8 does not degrade, warn, or read part of
//! the file. It cannot open it at all — which is exactly the reported symptom of
//! `h5disp` working while `load` fails, since those reach different HDF5
//! versions inside MATLAB.
//!
//! The last test covers the other end of the bound rather than the 1.8 format: a
//! *lower* bound of `LibVer::LATEST`, which selects the same 1.10 format the
//! writer emits by default and must produce a file the C library reads.

use hdf5_pure::{AttrValue, File, FileBuilder, LibVer};
#[cfg(feature = "__hdf5-1.10")]
use hdf5_pure::{FileAccessProperties, FileCreateProperties};
use tempfile::tempdir;

/// A file exercising everything the 1.8 format still carries: contiguous
/// datasets of two types, compact attributes on all three kinds of object, a
/// subgroup, and a userblock (the MAT v7.3 shape).
fn write_v18(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    b.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    b.with_userblock(512);
    b.set_attr("root_attr", AttrValue::AsciiString("r".into()));
    b.create_dataset("values")
        .with_f64_data(&[1.0, 2.0, 3.0])
        .set_attr("MATLAB_class", AttrValue::AsciiString("double".into()));
    let mut g = b.create_group("grp");
    g.set_attr("tag", AttrValue::I64(7));
    g.create_dataset("inner").with_i32_data(&[7, 8]);
    let g = g.finish();
    b.add_group(g);
    b.write(path).unwrap();
}

#[test]
fn c_reads_every_part_of_a_1_8_format_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("v18.h5");
    write_v18(&path);

    // The superblock version is the whole point, so read it out of the bytes
    // rather than trusting a library to report it.
    let bytes = std::fs::read(&path).unwrap();
    let sig = bytes
        .windows(8)
        .position(|w| w == b"\x89HDF\r\n\x1a\n")
        .expect("the file carries an HDF5 signature");
    assert_eq!(sig, 512, "the userblock precedes the superblock");
    assert_eq!(bytes[sig + 8], 2, "version 2 superblock");

    let f = hdf5::File::open(&path).unwrap();
    let d = f.dataset("values").unwrap();
    assert_eq!(d.read_raw::<f64>().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(d.attr_names().unwrap(), vec!["MATLAB_class".to_string()]);
    assert_eq!(d.loc_info().unwrap().num_attrs, 1);

    let g = f.group("grp").unwrap();
    assert_eq!(g.attr_names().unwrap(), vec!["tag".to_string()]);
    assert_eq!(g.loc_info().unwrap().num_attrs, 1);
    assert_eq!(
        g.dataset("inner").unwrap().read_raw::<i32>().unwrap(),
        vec![7, 8]
    );

    assert_eq!(f.attr_names().unwrap(), vec!["root_attr".to_string()]);
    assert_eq!(f.loc_info().unwrap().num_attrs, 1);
}

/// The C library must also be able to *modify* the file, not merely read it —
/// the check that caught the missing Group Info message in earlier releases.
///
/// Written without a userblock, unlike the read test above. A userblock file the
/// C library has modified is unreadable here for reasons that have nothing to do
/// with the format version (it reproduces identically on a version 3 superblock,
/// so it predates this bound), and pinning that failure into this module would
/// make a 1.8-format test fail for an unrelated repair.
#[test]
fn c_writes_into_a_1_8_format_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("v18_rw.h5");
    let mut b = FileBuilder::new();
    b.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    b.create_dataset("values").with_f64_data(&[1.0, 2.0, 3.0]);
    let mut g = b.create_group("grp");
    g.create_dataset("inner").with_i32_data(&[7, 8]);
    let g = g.finish();
    b.add_group(g);
    b.write(&path).unwrap();

    {
        let f = hdf5::File::open_rw(&path).unwrap();
        f.new_dataset::<i32>()
            .shape([3])
            .create("added")
            .unwrap()
            .write(&[4, 5, 6])
            .unwrap();
        f.group("grp")
            .unwrap()
            .new_dataset::<f64>()
            .shape([2])
            .create("added_inner")
            .unwrap()
            .write(&[1.5, 2.5])
            .unwrap();
    }

    // The C library's additions land in a file this crate still reads whole.
    let f = File::open(&path).unwrap();
    assert_eq!(f.libver_bound(), LibVer::V18);
    assert_eq!(
        f.dataset("added").unwrap().read_i32().unwrap(),
        vec![4, 5, 6]
    );
    assert_eq!(
        f.dataset("values").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        f.dataset("grp/added_inner").unwrap().read_f64().unwrap(),
        vec![1.5, 2.5]
    );
}

/// An edit session preserves the superblock version it opened, and writes its
/// contiguous additions in that file's own format rather than the writer's
/// default. So a `.mat` file that gains a variable through `open_rw` is still
/// the 1.8 file it was, which is the thing the MAT default would otherwise lose
/// on the first edit.
///
/// Chunked additions are the documented exception and are *not* refused: a
/// chunked dataset needs a 1.10 chunk index whatever the superblock says, and
/// refusing would take away in-place editing of every file the C library wrote
/// under its own default bounds (issue #101, `edit_crosscheck.rs`).
#[test]
fn an_edit_session_keeps_a_1_8_file_in_the_1_8_format() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("v18_edit.h5");
    write_v18(&path);

    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("flat", |d| {
                d.with_i32_data(&[1, 2, 3]);
            })
            .unwrap();
        s.dataset("values")
            .unwrap()
            .set_attr("added", AttrValue::I64(5))
            .unwrap();
        s.commit().unwrap();
    }

    let bytes = std::fs::read(&path).unwrap();
    let sig = bytes
        .windows(8)
        .position(|w| w == b"\x89HDF\r\n\x1a\n")
        .unwrap();
    assert_eq!(
        bytes[sig + 8],
        2,
        "the commit kept the version 2 superblock"
    );

    let f = File::open(&path).unwrap();
    assert_eq!(f.libver_bound(), LibVer::V18, "the file stayed 1.8");
    assert_eq!(
        f.dataset("flat").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    drop(f);

    // The C library reads the edited file whole, including the attribute the
    // edit added — which it can only count because the rewritten header carries
    // an Attribute Info message.
    let f = hdf5::File::open(&path).unwrap();
    assert_eq!(
        f.dataset("flat").unwrap().read_raw::<i32>().unwrap(),
        vec![1, 2, 3]
    );
    let d = f.dataset("values").unwrap();
    assert_eq!(d.read_raw::<f64>().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(d.loc_info().unwrap().num_attrs, 2);
}

/// A *lower* bound of `LibVer::LATEST` licenses newer encodings without
/// requiring any, so both entry points write the 1.10 format under it — the
/// reading `H5Pset_libver_bounds` has, where the low bound is a floor and the
/// library is free to stay below the ceiling. The C library reading the result
/// is what says the file is a real one and not merely one this crate accepted.
#[cfg(feature = "__hdf5-1.10")]
#[test]
fn c_reads_a_file_written_at_a_lower_bound_of_latest() {
    let dir = tempdir().unwrap();

    let via_builder = dir.path().join("latest_builder.h5");
    let mut b = FileBuilder::new();
    b.with_libver_bounds(LibVer::LATEST, LibVer::LATEST);
    b.set_attr("root_attr", AttrValue::AsciiString("r".into()));
    b.create_dataset("values").with_f64_data(&[1.0, 2.0, 3.0]);
    b.write(&via_builder).unwrap();

    let via_properties = dir.path().join("latest_created.h5");
    {
        let f = File::create_with_options(
            &via_properties,
            FileCreateProperties::new().with_libver_bounds(LibVer::LATEST, LibVer::LATEST),
            FileAccessProperties::new(),
        )
        .expect("a lower bound of LATEST is satisfied by the 1.10 format");
        f.root()
            .create_dataset("values", |d| {
                d.with_f64_data(&[1.0, 2.0, 3.0]);
            })
            .unwrap();
        f.root()
            .set_attr("root_attr", AttrValue::AsciiString("r".into()))
            .unwrap();
        f.commit().unwrap();
    }

    for path in [via_builder, via_properties] {
        let bytes = std::fs::read(&path).unwrap();
        let sig = bytes
            .windows(8)
            .position(|w| w == b"\x89HDF\r\n\x1a\n")
            .expect("the file carries an HDF5 signature");
        assert_eq!(
            bytes[sig + 8],
            3,
            "{}: the 1.10 format, not one 1.12 or 1.14 would have named",
            path.display()
        );

        // Released before the C library opens the same file: Windows OS locks
        // are mandatory.
        let f = File::open(&path).unwrap();
        assert_eq!(f.libver_bound(), LibVer::V110);
        assert_eq!(f.libver_bound(), LibVer::WRITER_DEFAULT);
        drop(f);

        let f = hdf5::File::open(&path).unwrap();
        assert_eq!(
            f.dataset("values").unwrap().read_raw::<f64>().unwrap(),
            vec![1.0, 2.0, 3.0],
            "{}",
            path.display()
        );
        assert_eq!(f.attr_names().unwrap(), vec!["root_attr".to_string()]);
    }
}
