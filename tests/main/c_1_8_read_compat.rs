//! Reading files written by an actual HDF5 1.8 library, from committed bytes.
//!
//! This is not the crate's first coverage of these formats, and does not claim
//! to be. `crates/crosscheck/tests/main/owned_swmr.rs` already asks libhdf5 for a version 1
//! superblock and checks the parsed K values and status flags against it, and
//! `crates/crosscheck/tests/main/edit.rs` does the same for a version 2 one. Both need the
//! `hdf5-metno` dev-dependency, which `test-32bit` and `test-big-endian` never
//! enable. Those recipes stay scoped to this crate's own pointer-width and
//! byte-order handling, where address arithmetic is most likely to be wrong,
//! and `test-32bit-hdf5`/`test-big-endian-hdf5` cover the pair separately,
//! under QEMU. Reading committed bytes needs no dev-dependency, so this file
//! runs under the plain `test-32bit` too, without a C toolchain, and is listed
//! in its recipe in the `justfile`.
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
//! `tests/data/c/1.8/NOTICE.md`.

use hdf5_pure::{AttrValue, File};
use test_util::superblock;

const V1: &str = "tests/data/c/1.8/v1_superblock.h5";
const V2: &str = "tests/data/c/1.8/v2_superblock.h5";

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

#[test]
fn reads_a_c_written_version_1_superblock() {
    // Guarded before the content checks: a fixture that stopped being a version
    // 1 superblock should say so, rather than surfacing as a content failure.
    assert_eq!(superblock::version(V1), 1);
    assert_contents(V1);
}

#[test]
fn reads_a_c_written_version_2_superblock() {
    assert_eq!(superblock::version(V2), 2);
    assert_contents(V2);
}
