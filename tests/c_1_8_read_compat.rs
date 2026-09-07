//! Reading files written by an actual HDF5 1.8 library, from committed bytes.
//!
//! This is not the crate's first coverage of these formats, and does not claim
//! to be. `tests/owned_swmr_crosscheck.rs` already asks libhdf5 for a version 1
//! superblock and checks the parsed K values and status flags against it, and
//! `tests/edit_crosscheck.rs` does the same for a version 2 one. What both cost
//! is the `hdf5-metno` dev-dependency, which requires 64-bit little-endian — so every
//! file using it opens with `#![cfg(not(target_pointer_width = "32"))]` and
//! compiles out on the i686 target, which is where address arithmetic is most
//! likely to be wrong. Reading committed bytes needs no dev-dependency, so this
//! runs there; it is listed in the `test-32bit` recipe in the justfile.
//!
//! The committed corpus also had nothing at these versions: of the 80 tracked
//! `.h5`/`.mat` fixtures before these two, 69 were superblock 0 and 11 were
//! superblock 3.
//!
//! What the pair uniquely holds is `v2_superblock.h5`: a **version 2 superblock
//! carrying a version 1 B-tree chunk index**. This crate's writer cannot produce
//! that combination — it refuses chunked storage under a 1.8 bound, because the
//! only chunk indices it writes arrived in 1.10 — so no round trip through it
//! can stand in for the file.
//!
//! Both files hold the same objects, written by the same program, so a failure
//! in one and not the other names the format rather than the reader. See
//! `tests/fixtures/c_1_8/NOTICE.md`.

use hdf5_pure::{AttrValue, File};

const V1: &str = "tests/fixtures/c_1_8/v1_superblock.h5";
const V2: &str = "tests/fixtures/c_1_8/v2_superblock.h5";

/// The values the fixtures hold, read back through the public API.
fn assert_contents(path: &str) {
    let f = File::open(path).unwrap_or_else(|e| panic!("{path}: {e:?}"));

    // Contiguous dataset with an attribute.
    assert_eq!(
        f.dataset("values").unwrap().read_f64().unwrap(),
        vec![1.5, 2.5, 3.5, 4.5],
        "{path}: /values"
    );
    assert_eq!(
        f.dataset("values").unwrap().attrs().unwrap().get("units"),
        Some(&AttrValue::AsciiString("m/s".into())),
        "{path}: /values units"
    );

    // Chunked and deflated, indexed by a version 1 B-tree in both files —
    // both were written under 1.8 bounds, and 1.8 had no other chunk index.
    let expected: Vec<i32> = (0..1000).map(|i| i % 97).collect();
    assert_eq!(
        f.dataset("chunked").unwrap().read_i32().unwrap(),
        expected,
        "{path}: /chunked"
    );

    // A group, its attribute, and a dataset inside it. The two files differ here
    // in a way the superblock version does not cause: the fixture builds the
    // version 2 file under `H5F_LIBVER_LATEST`, so its root is a link-message
    // group, where the version 1 file's is a v1 symbol table.
    assert_eq!(
        f.group("grp").unwrap().attrs().unwrap().get("tag"),
        Some(&AttrValue::AsciiString("group".into())),
        "{path}: /grp tag"
    );
    assert_eq!(
        f.dataset("grp/inner").unwrap().read_f64().unwrap(),
        vec![1.5, 2.5, 3.5, 4.5],
        "{path}: /grp/inner"
    );

    // A root-group attribute.
    assert_eq!(
        f.root().attrs().unwrap().get("root_attr"),
        Some(&AttrValue::AsciiString("r".into())),
        "{path}: / root_attr"
    );
}

/// The superblock version byte. Both fixtures have base address 0, so the
/// signature is at offset 0 and this needs no scan.
fn superblock_version(path: &str) -> u8 {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(
        &bytes[..8],
        b"\x89HDF\r\n\x1a\n",
        "{path}: no signature at offset 0; the fixture grew a userblock"
    );
    bytes[8]
}

#[test]
fn reads_a_c_written_version_1_superblock() {
    // Guarded before the content checks: a fixture that stopped being a version
    // 1 superblock should say so, rather than surfacing as a content failure.
    assert_eq!(superblock_version(V1), 1);
    assert_contents(V1);
}

#[test]
fn reads_a_c_written_version_2_superblock() {
    assert_eq!(superblock_version(V2), 2);
    assert_contents(V2);
}
