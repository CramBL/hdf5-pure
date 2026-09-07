// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Reference-C-library interop for issue #146: the unified append + edit session.
//!
//! Covers the fast, immediate `Dataset::append` and the staged
//! `set_dataset_attr` / `remove_dataset_attr` against files the C library *wrote*,
//! reading the result back with both the C library and this crate. The
//! make-or-break case is `set_dataset_attr` / `set_group_attr` on a C-written
//! object carrying an *undefined-address* Attribute Info message (which nearly
//! every real-world object has, for attribute creation-order metadata): the shared
//! compact-attribute walkers must accept it rather than mistake it for dense
//! storage.

use hdf5::Extent;
use hdf5::file::LibraryVersion;
use hdf5_pure::{AttrValue, Error, File};
use tempfile::tempdir;

mod common;
use common::assert_c_absent;

/// Create a rank-1 unlimited (Extensible-Array indexed) i32 dataset `name` with the
/// C library under the latest format, seeded with `0..n`, chunk length `chunk`.
fn c_create_unlimited(path: &std::path::Path, name: &str, n: i32, chunk: usize) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
    let ds = file
        .new_dataset::<i32>()
        .chunk((chunk,))
        .shape((Extent::resizable(n as usize),))
        .create(name)
        .unwrap();
    ds.write(&(0..n).collect::<Vec<_>>()).unwrap();
    file.close().unwrap();
}

fn read_c(path: &std::path::Path, name: &str) -> Vec<i32> {
    let f = hdf5::File::open(path).unwrap();
    let v = f.dataset(name).unwrap().read_raw::<i32>().unwrap();
    f.close().unwrap();
    v
}

fn read_pure(path: &std::path::Path, name: &str) -> Vec<i32> {
    File::open(path)
        .unwrap()
        .dataset(name)
        .unwrap()
        .read_i32()
        .unwrap()
}

// ---- in-place append against C-written files --------------------------------

#[test]
fn append_inplace_to_c_dataset_both_read() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    c_create_unlimited(&path, "d", 8, 4);

    {
        let s = File::open_rw(&path).unwrap();
        s.dataset("d").unwrap().append(&[8, 9, 10, 11, 12]).unwrap(); // any length (unfiltered)
    }

    let expected: Vec<i32> = (0..13).collect();
    assert_eq!(read_pure(&path, "d"), expected);
    assert_eq!(read_c(&path, "d"), expected);
}

/// Create a rank-1 unlimited, shuffle+deflate i32 dataset with the C library,
/// seeded with `0..n` and left on a partial trailing chunk when `n % chunk != 0`.
fn c_create_filtered_unlimited(path: &std::path::Path, name: &str, n: i32, chunk: usize) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
    let ds = file
        .new_dataset::<i32>()
        .chunk((chunk,))
        .shuffle()
        .deflate(4)
        .shape((Extent::resizable(n as usize),))
        .create(name)
        .unwrap();
    ds.write(&(0..n).collect::<Vec<_>>()).unwrap();
    file.close().unwrap();
}

/// A filtered dataset this crate leaves on a partial trailing chunk, then grows
/// again (issue #393). The re-encoded chunk is zero-padded to the full chunk
/// size before the pipeline runs and the dataspace dimension is what bounds the
/// live elements, so a reader that took the chunk size for the live length would
/// read the padding back as data — which is what the C library is here to rule
/// out.
#[test]
fn append_inplace_grows_a_filtered_partial_tail_both_read() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("filtered_tail.h5");
    c_create_filtered_unlimited(&path, "d", 8, 4);

    {
        let s = File::open_rw(&path).unwrap();
        let mut ds = s.dataset("d").unwrap();
        ds.append(&[8, 9]).unwrap(); // leaves a partial trailing chunk
        ds.append(&[10, 11]).unwrap(); // grows it
        ds.append(&[12, 13, 14]).unwrap(); // grows it again, crossing a boundary
    }

    let expected: Vec<i32> = (0..15).collect();
    assert_eq!(read_pure(&path, "d"), expected);
    assert_eq!(read_c(&path, "d"), expected);
}

/// The same growth onto a partial trailing chunk the **C library itself** wrote,
/// whose stored chunk may carry a non-zero filter mask. This crate decodes it
/// with that mask and re-encodes with none, which is what an aligned filtered
/// append has always done.
#[test]
fn append_inplace_grows_a_c_written_filtered_partial_tail() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_filtered_tail.h5");
    c_create_filtered_unlimited(&path, "d", 10, 4); // 10 % 4 != 0

    {
        let s = File::open_rw(&path).unwrap();
        let mut ds = s.dataset("d").unwrap();
        ds.append(&[10, 11]).unwrap(); // completes the C-written partial chunk
        ds.append(&[12]).unwrap(); // opens a new one
        ds.append(&(13..20).collect::<Vec<_>>()).unwrap(); // grows across chunks
    }

    let expected: Vec<i32> = (0..20).collect();
    assert_eq!(read_pure(&path, "d"), expected);
    assert_eq!(read_c(&path, "d"), expected);
}

#[test]
fn hard_link_aliasing_append_inplace_stays_coherent() {
    // Two hard links to one dataset: appending in place via either path must stay
    // coherent, because the geometry cache is keyed by object-header address, not
    // by path (both links share the one header).
    let dir = tempdir().unwrap();
    let path = dir.path().join("alias.h5");
    c_create_unlimited(&path, "d", 4, 4);
    {
        let file = hdf5::File::open_rw(&path).unwrap();
        file.link_hard("d", "alias").unwrap();
        file.close().unwrap();
    }

    {
        let s = File::open_rw(&path).unwrap();
        s.dataset("d").unwrap().append(&[4, 5]).unwrap(); // via "d"    -> 0..6
        s.dataset("alias").unwrap().append(&[6, 7]).unwrap(); // via alias -> 0..8
        s.dataset("d").unwrap().append(&[8]).unwrap(); // via "d"    -> 0..9
    }

    let expected: Vec<i32> = (0..9).collect();
    assert_eq!(read_pure(&path, "d"), expected);
    assert_eq!(read_c(&path, "alias"), expected); // same object, both names
}

// ---- dataset attributes on C-written objects (undefined-AttributeInfo) -------

#[test]
fn c_dataset_with_attribute_info_accepts_set_dataset_attr() {
    // A C-written dataset carrying attributes has an *undefined-address* Attribute
    // Info message (creation-order metadata, not dense storage). `set_dataset_attr`
    // must accept it, preserve the existing attribute, and add the new one — read
    // back by both libraries.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_ds_attr.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        let ds = file.new_dataset::<i32>().shape((4,)).create("d").unwrap();
        ds.write(&[1i32, 2, 3, 4]).unwrap();
        let a = ds.new_attr::<i64>().shape(()).create("orig").unwrap();
        a.write_scalar(&7i64).unwrap();
        file.close().unwrap();
    }

    {
        let s = File::open_rw(&path).unwrap();
        s.dataset("d")
            .unwrap()
            .set_attr("added", AttrValue::I64(3))
            .unwrap();
        s.commit().unwrap(); // must NOT be refused as "dense attribute storage"
    }

    // Reference C library sees the data and both attributes.
    let c = hdf5::File::open(&path).unwrap();
    let d = c.dataset("d").unwrap();
    assert_eq!(d.read_raw::<i32>().unwrap(), vec![1, 2, 3, 4]);
    let orig: i64 = d.attr("orig").unwrap().read_scalar().unwrap();
    let added: i64 = d.attr("added").unwrap().read_scalar().unwrap();
    assert_eq!((orig, added), (7, 3));
    c.close().unwrap();

    // Pure reader agrees.
    let f = File::open(&path).unwrap();
    let attrs = f.dataset("d").unwrap().attrs().unwrap();
    assert_eq!(attrs.get("orig"), Some(&AttrValue::I64(7)));
    assert_eq!(attrs.get("added"), Some(&AttrValue::I64(3)));
}

#[test]
fn c_group_with_attribute_info_accepts_set_group_attr() {
    // Regression for the shared-walker fix on the *group* path: a C-written group
    // carrying an attribute also has an undefined-address Attribute Info message,
    // and `set_group_attr` (which pre-dates this fix) must no longer spuriously
    // refuse it.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_grp_attr.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        let g = file.create_group("grp").unwrap();
        let a = g.new_attr::<i64>().shape(()).create("orig").unwrap();
        a.write_scalar(&11i64).unwrap();
        // A dataset so the file has content besides the group.
        file.new_dataset::<i32>()
            .shape((2,))
            .create("d")
            .unwrap()
            .write(&[5i32, 6])
            .unwrap();
        file.close().unwrap();
    }

    {
        let s = File::open_rw(&path).unwrap();
        s.group("grp")
            .unwrap()
            .set_attr("added", AttrValue::I64(22))
            .unwrap();
        s.commit().unwrap();
    }

    let c = hdf5::File::open(&path).unwrap();
    let g = c.group("grp").unwrap();
    let orig: i64 = g.attr("orig").unwrap().read_scalar().unwrap();
    let added: i64 = g.attr("added").unwrap().read_scalar().unwrap();
    assert_eq!((orig, added), (11, 22));
}

#[test]
fn set_dataset_attr_on_chunked_dataset_c_reads() {
    // An attribute edit on a *chunked* (Extensible-Array) dataset relocates the
    // header while preserving the data-layout message verbatim, so the chunk data
    // and index stay in place. Both libraries must read the grown-attribute dataset
    // and its data correctly, and a subsequent in-place append must still work
    // (the header address changed, so the geometry cache re-locates).
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_chunked_attr.h5");
    c_create_unlimited(&path, "d", 8, 4);

    {
        let s = File::open_rw(&path).unwrap();
        s.dataset("d")
            .unwrap()
            .set_attr("checked", AttrValue::I64(1))
            .unwrap();
        s.commit().unwrap();
        s.dataset("d").unwrap().append(&[8, 9, 10, 11]).unwrap(); // re-locates, then grows
    }

    let expected: Vec<i32> = (0..12).collect();
    assert_eq!(read_pure(&path, "d"), expected);
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(c.dataset("d").unwrap().read_raw::<i32>().unwrap(), expected);
    let checked: i64 = c
        .dataset("d")
        .unwrap()
        .attr("checked")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(checked, 1);
}

#[test]
fn set_dataset_attr_multi_hard_link_refused() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_ds_multilink.h5");
    c_create_unlimited(&path, "d", 4, 4);
    {
        let file = hdf5::File::open_rw(&path).unwrap();
        file.link_hard("d", "alias").unwrap();
        file.close().unwrap();
    }
    let before = std::fs::read(&path).unwrap();

    {
        let s = File::open_rw(&path).unwrap();
        s.dataset("d")
            .unwrap()
            .set_attr("x", AttrValue::I64(1))
            .unwrap();
        let err = s.commit().unwrap_err();
        assert!(
            matches!(err, Error::EditUnsupported(_))
                && err.to_string().contains("single hard link"),
            "expected single-hard-link refusal, got: {err}"
        );
    }
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "file modified on refusal"
    );
}

// ---- combined mixed edits ---------------------------------------------------

#[test]
fn combined_mixed_edits_c_readable() {
    // One long-lived session mixes an immediate in-place append, a staged group
    // creation, a staged dataset-attribute edit, and a staged recursive delete —
    // then more in-place appends after the commit. The reference C library reads
    // every result correctly.
    let dir = tempdir().unwrap();
    let path = dir.path().join("combined.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .chunk((4,))
            .shape((Extent::resizable(4),))
            .create("log")
            .unwrap()
            .write(&[0i32, 1, 2, 3])
            .unwrap();
        file.new_dataset::<i32>()
            .shape((3,))
            .create("keep")
            .unwrap()
            .write(&[7i32, 8, 9])
            .unwrap();
        let g = file.create_group("old").unwrap();
        g.new_dataset::<i32>()
            .shape((2,))
            .create("inner")
            .unwrap()
            .write(&[1i32, 2])
            .unwrap();
        file.close().unwrap();
    }

    {
        let s = File::open_rw(&path).unwrap();
        s.dataset("log").unwrap().append(&[4, 5, 6, 7]).unwrap(); // immediate -> 0..8
        s.root().create_group("run").unwrap(); // staged
        s.dataset("keep")
            .unwrap()
            .set_attr("checked", AttrValue::I64(1))
            .unwrap(); // staged
        s.root().delete("old").unwrap(); // staged recursive delete
        s.commit().unwrap();
        s.dataset("log").unwrap().append(&[8, 9]).unwrap(); // immediate -> 0..10
    }

    // Reference C library.
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("log").unwrap().read_raw::<i32>().unwrap(),
        (0..10).collect::<Vec<_>>()
    );
    assert_eq!(
        c.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
        vec![7, 8, 9]
    );
    let checked: i64 = c
        .dataset("keep")
        .unwrap()
        .attr("checked")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(checked, 1);
    assert!(c.group("run").is_ok(), "created group missing");
    assert_c_absent(&c.group("old").unwrap_err(), "old");
    c.close().unwrap();

    // Pure reader agrees on the grown dataset.
    assert_eq!(read_pure(&path, "log"), (0..10).collect::<Vec<_>>());
}

/// The shape that used to wedge a `BufferedAppender`: a *filtered* dataset
/// sitting on a partial trailing chunk with a second hard link (issue #316).
///
/// The appender used to realign such a dataset with an internal
/// `append_staged` + `commit`, and a commit relocates the object header, which
/// the append preflight refuses for a dataset with more than one hard link — so
/// the appender could never write and the session could never be sealed. Since
/// issue #393 the trailing chunk is re-encoded in place instead, which moves no
/// header, so both links keep naming the grown dataset.
#[test]
fn a_filtered_partial_tail_with_two_hard_links_appends_in_place() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("aliased_tail.h5");
    {
        let mut b = hdf5_pure::FileBuilder::new();
        b.create_dataset("d")
            .with_i32_data(&(0..10).collect::<Vec<i32>>())
            .with_shape(&[10])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[8])
            .with_deflate(1);
        b.write(&path).unwrap();
    }
    {
        let file = hdf5::File::open_rw(&path).unwrap();
        file.link_hard("d", "alias").unwrap();
        file.close().unwrap();
    }

    let session = File::open_rw(&path).unwrap();
    {
        let mut ds = session.dataset("d").unwrap();
        let mut app = ds.buffered_appender().unwrap();
        app.append(&(10..30i32).collect::<Vec<_>>()).unwrap();
        app.finish().unwrap();
    }
    session.close().unwrap();

    let expected: Vec<i32> = (0..30).collect();
    assert_eq!(read_pure(&path, "d"), expected);
    assert_eq!(read_pure(&path, "alias"), expected);
    assert_eq!(read_c(&path, "alias"), expected);
}

/// An immediate append that draws its chunks from freed space (issue #349)
/// produces a file the reference C library reads in full — the appended
/// dataset, and the neighbours whose bytes the reuse must not have reached.
///
/// The free-space assertion is what stops this from passing vacuously: a C read
/// succeeds just as well if the append quietly extended end-of-file instead, so
/// the session is asked to confirm it spent the hole first.
#[test]
fn an_append_into_freed_space_stays_c_readable() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("reuse.h5");
    let payload: Vec<i32> = (4..8196).collect();
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .chunk((1024,))
            .shape((Extent::resizable(4),))
            .create("log")
            .unwrap()
            .write(&[0i32, 1, 2, 3])
            .unwrap();
        // Vacated below. Sized so its hole dominates what the append allocates,
        // not so it swallows it whole: the appended chunks total more than this
        // (a partial leading chunk is rewritten too) and the extensible-array
        // blocks are on top, so some of the append still reaches end-of-file.
        file.new_dataset::<i32>()
            .shape((payload.len(),))
            .create("scratch")
            .unwrap()
            .write(&vec![5i32; payload.len()])
            .unwrap();
        // Live, and after the hole, so the delete cannot simply truncate.
        file.new_dataset::<i32>()
            .shape((3,))
            .create("ceiling")
            .unwrap()
            .write(&[7i32, 8, 9])
            .unwrap();
        file.close().unwrap();
    }

    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("scratch").unwrap();
        s.commit().unwrap();
        let before = s.space_accounting().unwrap().reusable_free_bytes;
        s.dataset("log").unwrap().append(&payload).unwrap();
        let after = s.space_accounting().unwrap().reusable_free_bytes;
        // Most of the hole, rather than a byte count that happens to match the
        // payload: the point is that the append drew on the free list at all,
        // and an exact figure would be a coincidence of this fixture's sizes.
        assert!(
            after < before / 2,
            "the append should have spent the hole on its chunks \
             ({before} -> {after} reusable)"
        );
        s.close().unwrap();
    }

    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("log").unwrap().read_raw::<i32>().unwrap(),
        (0..8196).collect::<Vec<_>>()
    );
    assert_eq!(
        c.dataset("ceiling").unwrap().read_raw::<i32>().unwrap(),
        vec![7, 8, 9]
    );
    assert_c_absent(&c.dataset("scratch").unwrap_err(), "scratch");
    c.close().unwrap();

    assert_eq!(read_pure(&path, "log"), (0..8196).collect::<Vec<_>>());
}
