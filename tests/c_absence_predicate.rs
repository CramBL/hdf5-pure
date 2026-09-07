// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit little-endian targets.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Guards the discriminating power of [`common::assert_c_absent`], which the
//! delete crosschecks use to prove a removed object is *absent* rather than
//! merely unreachable.
//!
//! Without this file the helper could silently decay. Its conditions are tuned
//! to which minor error codes the reference C library reports, and a libhdf5
//! upgrade can change that: if a future version stopped reporting a load
//! failure when metadata is damaged, the helper would keep passing while no
//! longer distinguishing damage from absence, and every delete crosscheck would
//! quietly weaken back to `is_err()`.
//!
//! So this asserts both directions, and asserts that the extra conditions are
//! still doing work.
//!
//! Every call goes through the safe `hdf5-metno` API, which serializes its own C
//! calls through an internal lock, so no extra guard is needed.

use hdf5_pure::FileBuilder;
use tempfile::tempdir;

mod common;
use common::{assert_c_absent, c_reports_absent};

/// A file whose bytes are mostly metadata: 40 two-element datasets, so a
/// corruption sweep lands on object headers and link tables rather than on raw
/// data that nothing has to parse.
fn metadata_heavy(path: &std::path::Path) {
    let mut b = FileBuilder::new();
    for i in 0..40i32 {
        b.create_dataset(&format!("d{i}")).with_data(&[i, 7]);
    }
    b.write(path).unwrap();
}

/// The helper must accept a genuine absence in every shape a delete test can
/// produce. A false negative here is a flaky crosscheck.
#[test]
fn genuine_absences_are_accepted() {
    let dir = tempdir().unwrap();

    // A compact group (<= 8 links, symbol-table/compact storage).
    let compact = dir.path().join("compact.h5");
    let mut b = FileBuilder::new();
    for i in 0..3i32 {
        b.create_dataset(&format!("c{i}")).with_data(&[i]);
    }
    b.write(&compact).unwrap();
    {
        let c = hdf5::File::open(&compact).unwrap();
        assert_c_absent(&c.dataset("nope").unwrap_err(), "missing dataset");
        assert_c_absent(&c.group("nope").unwrap_err(), "missing group");
        assert_c_absent(
            &c.dataset("c0").unwrap().attr("nope").unwrap_err(),
            "missing attribute",
        );
    }

    // A dense group (40 links).
    let dense = dir.path().join("dense.h5");
    metadata_heavy(&dense);
    {
        let c = hdf5::File::open(&dense).unwrap();
        assert_c_absent(&c.dataset("nope").unwrap_err(), "missing dataset (dense)");
    }

    // Absence inside a subgroup, so traversal crosses a link first.
    let nested = dir.path().join("nested.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("sub");
    g.create_dataset("inner").with_data(&[1i32]);
    let finished = g.finish();
    b.add_group(finished);
    b.write(&nested).unwrap();
    {
        let c = hdf5::File::open(&nested).unwrap();
        assert_eq!(
            c.dataset("sub/inner").unwrap().read_raw::<i32>().unwrap(),
            vec![1]
        );
        assert_c_absent(&c.dataset("sub/nope").unwrap_err(), "missing in subgroup");
    }
}

/// Corrupt a metadata-heavy file at every offset and several widths, then ask
/// for datasets that *do* exist. The helper must reject every resulting failure:
/// damage is not absence.
///
/// The second assertion is the one that keeps this test honest. A damaged link
/// table reports `H5E_NOTFOUND` just as a missing link does, so if the bare code
/// check ever stops producing false accepts, the helper's extra conditions have
/// become untestable here and this file no longer proves anything.
#[test]
fn corruption_is_never_mistaken_for_absence() {
    let dir = tempdir().unwrap();
    let good = dir.path().join("good.h5");
    metadata_heavy(&good);
    let baseline = std::fs::read(&good).unwrap();

    let mut damaged = 0;
    let mut accepted_by_bare_code = 0;

    // Past the superblock, since rewriting it is a different failure class the
    // delete crosschecks never produce.
    for width in [1usize, 4, 8, 32] {
        for at in (96..baseline.len()).step_by(16) {
            let mut bytes = baseline.clone();
            for i in 0..width {
                if at + i < bytes.len() {
                    bytes[at + i] ^= 0xFF;
                }
            }
            let path = dir.path().join(format!("corrupt_{width}_{at}.h5"));
            std::fs::write(&path, &bytes).unwrap();

            let failure = match hdf5::File::open(&path) {
                Err(e) => Some(e),
                Ok(c) => (0..40).find_map(|i| match c.dataset(&format!("d{i}")) {
                    Ok(ds) => ds.read_raw::<i32>().err(),
                    Err(e) => Some(e),
                }),
            };

            let Some(err) = failure else { continue };
            damaged += 1;
            if err.contains_minor(hdf5::MinorErrorCode::NotFound) {
                accepted_by_bare_code += 1;
            }
            assert!(
                !c_reports_absent(&err),
                "corrupting {width} byte(s) at offset {at} produced a failure the \
                 absence helper accepted, so it cannot tell damage from absence: {err}"
            );
        }
    }

    assert!(
        damaged > 100,
        "the sweep must actually break the file to prove anything (only {damaged} \
         of the corruptions were detectable)"
    );
    assert!(
        accepted_by_bare_code > 0,
        "no corruption reported H5E_NOTFOUND, so this sweep no longer exercises the \
         helper's extra conditions — either the sweep or the helper needs revisiting \
         ({damaged} corruptions were detectable)"
    );
}
