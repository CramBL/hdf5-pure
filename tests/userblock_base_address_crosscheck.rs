// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Reading the structures the C library places in a file that has a userblock.
//!
//! The format specification says of the superblock's base address that "unless
//! otherwise noted, all other file addresses are relative to this base address".
//! A userblock makes that base non-zero — 512 bytes for a `.mat` file — so every
//! stored address is short of its real position by exactly that much, and a
//! reader that forgets to add it back lands inside the userblock.
//!
//! Two structures this crate does not itself write reach that path, so nothing
//! in the suite exercised them until now:
//!
//! - an **object header continuation block**, which the C library creates when a
//!   header outgrows its chunk. This crate always writes single-chunk headers.
//! - **dense link storage** (fractal heap + version 2 B-tree), which the C
//!   library switches a group to past eight links. This crate writes compact
//!   Link messages.
//!
//! Both appear the moment a C-based tool touches a `.mat` file, since every one
//! carries a userblock: `h5repack`, `h5copy`, or MATLAB writing over the file.
//! The failure was total rather than partial — the root group would not list at
//! all — and it needed no edit from us to happen, as
//! [`c_written_dense_links_in_a_userblock_file`] shows.
//!
//! Each test reads through every backend, because they resolve addresses by
//! different code: [`File::open`] indexes one buffer, [`File::open_streaming`]
//! reads windows through a `Source`, and [`File::from_bytes`] takes an image the
//! caller owns.
//!
//! [`every_c_written_shape_reads_under_a_userblock`] is a different kind of test
//! from the rest. It fixes no defect: every shape in it already read correctly.
//! It exists because a missing base address has now been a bug three separate
//! times — dense attributes in #214, then the two above — which makes it a
//! *class* rather than three incidents, and a class is worth covering by its
//! ground rather than by its instances.
//!
//! The read path applies the base in four independent places, and this file is
//! sized to catch each. Measured by removing the base from one place at a time
//! and recording which tests notice:
//!
//! | base removed from | tests that fail |
//! | --- | --- |
//! | raw data reads (contiguous, chunked, chunk indices, global heap) | all six |
//! | dense attribute storage | `every_c_written_shape` only |
//! | object header continuation | the two `c_modified`/`c_added_attribute` tests |
//! | dense link storage | the three dense-link tests |
//!
//! Worth reading the second and third rows together. The class test is the only
//! guard on dense attribute storage, and it is *not* a guard on continuations —
//! its six-deep group nesting does not produce one, which the measurement showed
//! and an unchecked assumption would not have. Continuations stay covered by the
//! two tests that provoke them directly.

use hdf5_pure::{File, FileBuilder};
use tempfile::tempdir;

/// The size MATLAB uses, and the one every `.mat` this crate writes carries.
const UB: u64 = 512;

/// Read `path` through every backend and return each one's view of the root
/// group's dataset names, sorted.
fn names_via_every_backend(path: &std::path::Path) -> Vec<(&'static str, Vec<String>)> {
    let sorted = |f: &File| -> Vec<String> {
        let mut v = f.root().datasets().expect("list the root group");
        v.sort();
        v
    };
    vec![
        ("open", sorted(&File::open(path).unwrap())),
        (
            "open_streaming",
            sorted(&File::open_streaming(path).unwrap()),
        ),
        (
            "from_bytes",
            sorted(&File::from_bytes(std::fs::read(path).unwrap()).unwrap()),
        ),
    ]
}

fn assert_every_backend_sees(path: &std::path::Path, expected: &[&str]) {
    let mut want: Vec<String> = expected.iter().map(|s| (*s).to_string()).collect();
    want.sort();
    for (backend, got) in names_via_every_backend(path) {
        assert_eq!(got, want, "backend `{backend}` read the wrong link set");
    }
}

/// Create a file with the C library, with or without a userblock.
fn c_file(path: &std::path::Path, userblock: u64, build: impl FnOnce(&hdf5::File)) {
    let file = hdf5::File::with_options()
        .with_fcpl(|p| {
            if userblock > 0 {
                p.userblock(userblock);
            }
            p
        })
        .create(path)
        .unwrap();
    build(&file);
    file.close().unwrap();
    assert_eq!(
        hdf5::File::open(path).unwrap().userblock(),
        userblock,
        "the C library did not produce the requested userblock"
    );
}

fn c_add_dataset(file: &hdf5::File, name: &str, value: i32) {
    file.new_dataset::<i32>()
        .shape([1])
        .create(name)
        .unwrap()
        .write(&[value])
        .unwrap();
}

/// The reported case: a file this crate wrote, then modified by the C library.
///
/// Adding one link overflows the root header's chunk, so the C library spills the
/// rest into a continuation block and records its address relative to the base.
/// Run against both userblock sizes: the zero case passed throughout and pins
/// that the base is what makes the difference, not the C edit.
#[test]
fn c_modified_file_with_a_userblock_reads_whole() {
    let dir = tempdir().unwrap();
    for ub in [0, UB] {
        let path = dir.path().join(format!("ours_then_c_{ub}.h5"));
        let mut b = FileBuilder::new();
        if ub > 0 {
            b.with_userblock(ub);
        }
        b.create_dataset("values").with_f64_data(&[1.0, 2.0, 3.0]);
        b.write(&path).unwrap();

        {
            let f = hdf5::File::open_rw(&path).unwrap();
            c_add_dataset(&f, "added", 4);
        }

        assert_every_backend_sees(&path, &["values", "added"]);

        // The data itself, not merely the names: a continuation block carries the
        // data-layout message of whichever dataset was pushed out of chunk 0.
        let f = File::open(&path).unwrap();
        assert_eq!(
            f.dataset("values").unwrap().read_f64().unwrap(),
            vec![1.0, 2.0, 3.0],
            "userblock {ub}"
        );
        assert_eq!(
            f.dataset("added").unwrap().read_i32().unwrap(),
            vec![4],
            "userblock {ub}"
        );
    }
}

/// An attribute added by the C library also grows the header past its chunk, so
/// this reaches the continuation block without adding any link.
#[test]
fn c_added_attribute_in_a_userblock_file_reads_whole() {
    let dir = tempdir().unwrap();
    for ub in [0, UB] {
        let path = dir.path().join(format!("attr_{ub}.h5"));
        let mut b = FileBuilder::new();
        if ub > 0 {
            b.with_userblock(ub);
        }
        b.create_dataset("values").with_f64_data(&[1.5]);
        b.write(&path).unwrap();

        {
            let f = hdf5::File::open_rw(&path).unwrap();
            let d = f.dataset("values").unwrap();
            d.new_attr::<i32>()
                .shape([1])
                .create("tag")
                .unwrap()
                .write(&[7])
                .unwrap();
        }

        assert_every_backend_sees(&path, &["values"]);
        let f = File::open(&path).unwrap();
        assert_eq!(f.dataset("values").unwrap().read_f64().unwrap(), vec![1.5]);
        let attrs = f.dataset("values").unwrap().attrs().unwrap();
        assert_eq!(
            attrs.keys().collect::<Vec<_>>(),
            vec!["tag"],
            "userblock {ub}"
        );
    }
}

/// Dense link storage in a userblock file the C library wrote start to finish.
///
/// Nothing this crate wrote is involved, which is what makes this the widest of
/// the three: the Link Info message names a fractal heap and a version 2 B-tree
/// by base-relative address, and reading either at its stored value lands in the
/// userblock. Twenty-four links is past the eight at which the C library converts
/// a group to dense storage.
#[test]
fn c_written_dense_links_in_a_userblock_file() {
    let dir = tempdir().unwrap();
    for ub in [0, UB] {
        let path = dir.path().join(format!("dense_{ub}.h5"));
        c_file(&path, ub, |f| {
            for i in 0..24 {
                c_add_dataset(f, &format!("d{i:02}"), i);
            }
        });

        let names: Vec<String> = (0..24).map(|i| format!("d{i:02}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        assert_every_backend_sees(&path, &refs);

        let f = File::open(&path).unwrap();
        assert_eq!(
            f.dataset("d17").unwrap().read_i32().unwrap(),
            vec![17],
            "userblock {ub}"
        );
    }
}

/// A group *nested* inside a userblock file, dense on both levels. The base
/// applies at every hop, not only at the root, so this catches a fix applied
/// only where the root group is resolved.
#[test]
fn c_written_nested_dense_group_in_a_userblock_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("nested.h5");
    c_file(&path, UB, |f| {
        let g = f.create_group("inner").unwrap();
        for i in 0..24 {
            g.new_dataset::<i32>()
                .shape([1])
                .create(format!("n{i:02}").as_str())
                .unwrap()
                .write(&[i])
                .unwrap();
        }
        for i in 0..24 {
            c_add_dataset(f, &format!("r{i:02}"), i);
        }
    });

    let f = File::open(&path).unwrap();
    let mut inner = f.group("inner").unwrap().datasets().unwrap();
    inner.sort();
    let mut want: Vec<String> = (0..24).map(|i| format!("n{i:02}")).collect();
    want.sort();
    assert_eq!(inner, want);
    assert_eq!(
        f.dataset("inner/n23").unwrap().read_i32().unwrap(),
        vec![23]
    );
}

/// The whole point of the fix, stated as the round trip it protects: a `.mat`
/// file that a C-based tool has rewritten still reads here.
///
/// Twelve variables rather than two, because that is what makes this a test
/// rather than a demonstration. A `.mat` with a couple of variables keeps its
/// root links compact and its headers in one chunk, so it never reaches either
/// half of the defect and passes with the whole fix reverted; past eight, the
/// copy writes dense link storage and the base address starts to matter.
#[cfg(feature = "serde")]
#[test]
fn a_mat_file_repacked_by_the_c_library_still_reads() {
    use hdf5_pure::mat;
    use std::collections::BTreeMap;

    const VARS: usize = 12;

    let doc: BTreeMap<String, Vec<f64>> = (0..VARS)
        .map(|i| (format!("var{i:02}"), vec![i as f64, i as f64 + 0.5]))
        .collect();

    let dir = tempdir().unwrap();
    let src = dir.path().join("in.mat");
    mat::to_file(&doc, &src).unwrap();
    assert_eq!(
        hdf5::File::open(&src).unwrap().userblock(),
        UB,
        "a .mat file carries a userblock, which is what puts it on this path"
    );

    // Stand in for `h5repack`: the C library copies every object into a fresh
    // file that keeps the userblock, which is the shape a repacked `.mat` has.
    let dst = dir.path().join("out.mat");
    {
        let input = hdf5::File::open(&src).unwrap();
        let output = hdf5::File::with_options()
            .with_fcpl(|p| {
                p.userblock(UB);
                p
            })
            .create(&dst)
            .unwrap();
        for name in input.member_names().unwrap() {
            input
                .dataset(&name)
                .unwrap()
                .copy_to(&output, &name)
                .unwrap();
        }
    }

    let f = File::open(&dst).unwrap();
    let mut got = f.root().datasets().unwrap();
    got.sort();
    let mut want: Vec<String> = doc.keys().cloned().collect();
    want.sort();
    assert_eq!(got, want, "every variable survived the repack");
    assert_eq!(
        f.dataset("var07").unwrap().read_f64().unwrap(),
        vec![7.0, 7.5]
    );
}

// ---------------------------------------------------------------------------
// The defect class, rather than the two instances of it
// ---------------------------------------------------------------------------

/// Read `path` through every backend and check it with `read`.
fn every_backend(path: &std::path::Path, what: &str, read: impl Fn(&File) -> String) {
    let expected = read(&File::open(path).unwrap());
    for (backend, file) in [
        ("open", File::open(path).unwrap()),
        ("open_streaming", File::open_streaming(path).unwrap()),
        (
            "from_bytes",
            File::from_bytes(std::fs::read(path).unwrap()).unwrap(),
        ),
    ] {
        assert_eq!(
            read(&file),
            expected,
            "{what}: backend `{backend}` disagrees with `open`"
        );
    }
}

/// Every shape the C library writes that makes this crate resolve a stored
/// address, read back from a file with a userblock and from one without.
///
/// A stored address is short of its real position by the base, so a reader that
/// forgets to add it back lands inside the userblock — and each of these shapes
/// is read by a *different* one. The `ub = 0` half is the control: it holds the
/// content constant so a failure can only be the base address, not the shape.
///
/// The module documentation carries the measured coverage table for the whole
/// file. What this test adds to it is the only guard on dense attribute storage,
/// and the widest guard on raw data reads: every shape below stops reading when
/// the base is removed from the data path, so a regression there cannot hide in
/// whichever storage kind happens to be untested.
///
/// Some of these have userblock coverage elsewhere — dense attributes in
/// `userblock_dense_attrs.rs`, chunked editing in `edit_userblock_chunked.rs` —
/// and are kept anyway. The point is one place where the class is visible at
/// once, so the next address-consuming reader added to this crate has an obvious
/// home for its case.
#[test]
fn every_c_written_shape_reads_under_a_userblock() {
    let dir = tempdir().unwrap();

    for ub in [0, UB] {
        // Chunked with a filter: the chunk addresses live in a chunk index, and
        // the data must be found before it can be inflated.
        let path = dir.path().join(format!("chunked_deflate_{ub}.h5"));
        c_file(&path, ub, |f| {
            f.new_dataset::<f64>()
                .shape([64])
                .chunk([8])
                .deflate(4)
                .create("c")
                .unwrap()
                .write(&(0..64).map(f64::from).collect::<Vec<_>>())
                .unwrap();
        });
        every_backend(&path, "chunked+deflate", |f| {
            format!("{:?}", f.dataset("c").unwrap().read_f64().unwrap())
        });
        assert_eq!(
            File::open(&path)
                .unwrap()
                .dataset("c")
                .unwrap()
                .read_f64()
                .unwrap(),
            (0..64).map(f64::from).collect::<Vec<_>>(),
            "chunked+deflate under userblock {ub}"
        );

        // Variable-length strings: the element data is a reference into the
        // global heap, which is reached by its own stored address.
        let path = dir.path().join(format!("vlen_{ub}.h5"));
        c_file(&path, ub, |f| {
            let words: Vec<hdf5::types::VarLenUnicode> = ["alpha", "beta", "gamma"]
                .iter()
                .map(|s| s.parse().unwrap())
                .collect();
            f.new_dataset::<hdf5::types::VarLenUnicode>()
                .shape([3])
                .create("s")
                .unwrap()
                .write(&words)
                .unwrap();
        });
        every_backend(&path, "vlen strings", |f| {
            format!("{:?}", f.dataset("s").unwrap().read_string().unwrap())
        });
        assert_eq!(
            File::open(&path)
                .unwrap()
                .dataset("s")
                .unwrap()
                .read_string()
                .unwrap(),
            vec!["alpha", "beta", "gamma"],
            "vlen strings under userblock {ub}"
        );

        // Enough attributes to force dense storage: a fractal heap and a version
        // 2 B-tree, each named by a stored address in the Attribute Info message.
        let path = dir.path().join(format!("dense_attrs_{ub}.h5"));
        c_file(&path, ub, |f| {
            let d = f.new_dataset::<i32>().shape([1]).create("d").unwrap();
            d.write(&[1]).unwrap();
            for i in 0..24 {
                d.new_attr::<i32>()
                    .shape([1])
                    .create(format!("a{i:02}").as_str())
                    .unwrap()
                    .write(&[i])
                    .unwrap();
            }
        });
        every_backend(&path, "dense attributes", |f| {
            let mut names: Vec<String> = f
                .dataset("d")
                .unwrap()
                .attrs()
                .unwrap()
                .into_keys()
                .collect();
            names.sort();
            names.join(",")
        });
        assert_eq!(
            File::open(&path)
                .unwrap()
                .dataset("d")
                .unwrap()
                .attrs()
                .unwrap()
                .len(),
            24,
            "dense attributes under userblock {ub}"
        );

        // Six levels of nesting: the base applies at every hop, not only at the
        // root, and a header this deep in a C-written file carries a
        // continuation block.
        let path = dir.path().join(format!("nested_{ub}.h5"));
        c_file(&path, ub, |f| {
            let mut g = f.create_group("l0").unwrap();
            for i in 1..6 {
                g = g.create_group(&format!("l{i}")).unwrap();
            }
            g.new_dataset::<i32>()
                .shape([1])
                .create("leaf")
                .unwrap()
                .write(&[42])
                .unwrap();
        });
        every_backend(&path, "nested groups", |f| {
            format!(
                "{:?}",
                f.dataset("l0/l1/l2/l3/l4/l5/leaf")
                    .unwrap()
                    .read_i32()
                    .unwrap()
            )
        });
        assert_eq!(
            File::open(&path)
                .unwrap()
                .dataset("l0/l1/l2/l3/l4/l5/leaf")
                .unwrap()
                .read_i32()
                .unwrap(),
            vec![42],
            "nested groups under userblock {ub}"
        );

        // Sixty-four chunks: enough that the index is a real structure with
        // internal nodes rather than a single-chunk shortcut.
        let path = dir.path().join(format!("many_chunks_{ub}.h5"));
        c_file(&path, ub, |f| {
            f.new_dataset::<i32>()
                .shape([1024])
                .chunk([16])
                .create("b")
                .unwrap()
                .write(&(0..1024).collect::<Vec<i32>>())
                .unwrap();
        });
        every_backend(&path, "64-chunk index", |f| {
            let v = f.dataset("b").unwrap().read_i32().unwrap();
            format!("{} {} {}", v.len(), v[0], v[v.len() - 1])
        });
        assert_eq!(
            File::open(&path)
                .unwrap()
                .dataset("b")
                .unwrap()
                .read_i32()
                .unwrap(),
            (0..1024).collect::<Vec<i32>>(),
            "64-chunk index under userblock {ub}"
        );

        // An unlimited dimension, which the C library indexes with an
        // extensible array rather than the fixed-array form above.
        let path = dir.path().join(format!("unlimited_{ub}.h5"));
        c_file(&path, ub, |f| {
            let d = f
                .new_dataset::<i32>()
                .shape((0..,))
                .chunk([8])
                .create("u")
                .unwrap();
            d.resize([32]).unwrap();
            d.write(&(0..32).collect::<Vec<i32>>()).unwrap();
        });
        every_backend(&path, "extensible-array index", |f| {
            format!("{:?}", f.dataset("u").unwrap().read_i32().unwrap())
        });
        assert_eq!(
            File::open(&path)
                .unwrap()
                .dataset("u")
                .unwrap()
                .read_i32()
                .unwrap(),
            (0..32).collect::<Vec<i32>>(),
            "extensible-array index under userblock {ub}"
        );
    }
}
