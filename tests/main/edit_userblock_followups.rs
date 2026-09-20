//! Delete and copy editing on files that carry a userblock (non-zero base
//! address) — the follow-up parity work to the userblock slice of issue #104.
//!
//! Both operations rewrite or reclaim on-disk addresses, all of which are stored
//! relative to the userblock base on such a file. Each test edits a synthetic
//! userblock file and reads the result back through the pure-Rust reader, and
//! every test asserts the 512-byte userblock survives the edit byte-for-byte.
//!
//! (The resizing-overwrite parity for the contiguous and compact layouts is
//! exercised in `crates/crosscheck/tests/edit_userblock.rs`: only the reference C library can
//! create the never-written-contiguous and compact-layout fixtures those paths
//! need.)

use hdf5_pure::{AttrValue, File, FileBuilder};

use temp::temp_path;
use test_util::temp;

const UB: usize = 512;

const MARKER: &[u8] = b"USERBLOCK-FOLLOWUP-104";

/// Stamp a recognizable marker across the userblock region of `bytes` and return
/// the 512-byte userblock as written, for later byte-for-byte comparison.
fn stamp_userblock(bytes: &mut [u8]) -> Vec<u8> {
    bytes[..MARKER.len()].copy_from_slice(MARKER);
    bytes[UB - 1] = 0xAB;
    bytes[..UB].to_vec()
}

fn assert_userblock_unchanged(path: &std::path::Path, original: &[u8]) {
    let after = std::fs::read(path).unwrap();
    assert_eq!(
        &after[..UB],
        original,
        "userblock bytes changed across the edit"
    );
}

/// Build a userblock file with two root datasets and a nested group+dataset.
fn build_userblock_file(path: &std::path::Path) -> Vec<u8> {
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("alpha")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0]);
    b.create_dataset("beta").with_i32_data(&[10, 20, 30]);
    let mut g = b.create_group("grp");
    g.create_dataset("inner").with_f64_data(&[7.5, 8.5]);
    b.add_group(g.finish());
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(path, &bytes).unwrap();
    userblock
}

// ---- delete ----

#[test]
fn userblock_delete_dataset_roundtrip() {
    let path = temp_path("hdf5_pure_ub_fu_delete_ds.h5");
    let userblock = build_userblock_file(&path);

    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("alpha").unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert!(file.dataset("alpha").is_err(), "alpha should be deleted");
    // Neighbours survive.
    assert_eq!(
        file.dataset("beta").unwrap().read_i32().unwrap(),
        vec![10, 20, 30]
    );
    assert_eq!(
        file.dataset("grp/inner").unwrap().read_f64().unwrap(),
        vec![7.5, 8.5]
    );
    let mut datasets = file.root().datasets().unwrap();
    datasets.sort();
    assert_eq!(datasets, vec!["beta".to_string()]);
    assert_userblock_unchanged(&path, &userblock);
}

#[test]
fn userblock_delete_group_subtree_roundtrip() {
    // Deleting a group reclaims its whole subtree (its header plus the nested
    // dataset's header and data). On a userblock file every child link and data
    // address is base-relative, so the subtree walk must re-absolutize them.
    let path = temp_path("hdf5_pure_ub_fu_delete_grp.h5");
    let userblock = build_userblock_file(&path);

    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("grp").unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert!(file.group("grp").is_err(), "grp should be deleted");
    assert!(file.dataset("grp/inner").is_err());
    assert!(file.root().groups().unwrap().is_empty());
    assert_eq!(
        file.dataset("alpha").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_userblock_unchanged(&path, &userblock);
}

#[test]
fn userblock_delete_chunked_dataset_roundtrip() {
    // Deleting a chunked/filtered dataset reclaims its chunk index and chunk data
    // blocks via the base-aware `chunked_storage_spans`.
    let path = temp_path("hdf5_pure_ub_fu_delete_chunk.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    let chunked: Vec<f64> = (0..800).map(|i| (i % 7) as f64 * 0.5).collect();
    b.create_dataset("c")
        .with_f64_data(&chunked)
        .with_shape(&[800])
        .with_chunks(&[50])
        .with_deflate(6);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();

    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("c").unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert!(file.dataset("c").is_err(), "c should be deleted");
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert_userblock_unchanged(&path, &userblock);
}

#[test]
fn userblock_delete_then_reuse_freed_space() {
    // Deleting a sizeable contiguous dataset frees its data extent; a later commit
    // in the same session should reuse that freed hole for a contiguous add rather
    // than only appending. The reused write lands at the freed *absolute* offset,
    // so a base mistake in `collect_free_spans` would either leak (no reuse) or, far
    // worse, free a still-live region the reuse then corrupts.
    let path = temp_path("hdf5_pure_ub_fu_delete_reuse.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    let big: Vec<f64> = (0..256).map(|i| i as f64).collect();
    b.create_dataset("big").with_f64_data(&big);
    b.create_dataset("keep").with_i32_data(&[7, 8, 9]);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();

    let reuse: Vec<f64> = (0..64).map(|i| (i as f64) * -1.5).collect();
    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("big").unwrap();
        s.commit().unwrap();
        let len_after_delete = std::fs::metadata(&path).unwrap().len();
        s.root()
            .create_dataset("reuse", |b| {
                b.with_f64_data(&reuse);
            })
            .unwrap();
        s.commit().unwrap();
        let len_after_reuse = std::fs::metadata(&path).unwrap().len();
        assert!(
            len_after_reuse < len_after_delete + (reuse.len() * 8) as u64,
            "reuse commit did not reclaim freed delete space \
             (delete={len_after_delete}, reuse={len_after_reuse})"
        );
    }

    let file = File::open(&path).unwrap();
    assert!(file.dataset("big").is_err());
    assert_eq!(file.dataset("reuse").unwrap().read_f64().unwrap(), reuse);
    assert_eq!(
        file.dataset("keep").unwrap().read_i32().unwrap(),
        vec![7, 8, 9]
    );
    assert_userblock_unchanged(&path, &userblock);
}

#[test]
fn userblock_delete_one_of_several_then_read_attr() {
    // A delete rewrites the parent group's header (relinking survivors) and frees
    // the removed object. A sibling group's compact attribute must remain readable.
    let path = temp_path("hdf5_pure_ub_fu_delete_attr.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("doomed").with_f64_data(&[1.0, 2.0]);
    b.create_dataset("survivor").with_i32_data(&[5, 6]);
    let mut g = b.create_group("grp");
    g.set_attr("tag", AttrValue::AsciiString("kept".into()));
    b.add_group(g.finish());
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();

    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("doomed").unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert!(file.dataset("doomed").is_err());
    assert_eq!(
        file.dataset("survivor").unwrap().read_i32().unwrap(),
        vec![5, 6]
    );
    let attrs = file.group("grp").unwrap().attrs().unwrap();
    assert_eq!(
        attrs.get("tag"),
        Some(&AttrValue::AsciiString("kept".into()))
    );
    assert_userblock_unchanged(&path, &userblock);
}

// ---- copy (in-file) ----

#[test]
fn userblock_copy_dataset_roundtrip() {
    // Copying a contiguous dataset writes a fresh data block and header; on a
    // userblock file the new data address and the parent link to the copy must both
    // be stored base-relative.
    let path = temp_path("hdf5_pure_ub_fu_copy_ds.h5");
    let userblock = build_userblock_file(&path);

    {
        let s = File::open_rw(&path).unwrap();
        s.copy("alpha", "alpha_copy").unwrap();
        s.copy("grp/inner", "grp/inner_copy").unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    // Original untouched, copy reads identical values.
    assert_eq!(
        file.dataset("alpha").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        file.dataset("alpha_copy").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        file.dataset("grp/inner_copy").unwrap().read_f64().unwrap(),
        vec![7.5, 8.5]
    );
    assert_userblock_unchanged(&path, &userblock);
}

#[test]
fn userblock_copy_group_subtree_roundtrip() {
    // Copying a whole group deep-copies its nested dataset too; every child link
    // and data address in the copy is written base-relative.
    let path = temp_path("hdf5_pure_ub_fu_copy_grp.h5");
    let userblock = build_userblock_file(&path);

    {
        let s = File::open_rw(&path).unwrap();
        s.copy("grp", "grp_copy").unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    // Both the original subtree and the copy are present and identical.
    assert_eq!(
        file.dataset("grp/inner").unwrap().read_f64().unwrap(),
        vec![7.5, 8.5]
    );
    assert_eq!(
        file.dataset("grp_copy/inner").unwrap().read_f64().unwrap(),
        vec![7.5, 8.5]
    );
    let mut groups = file.root().groups().unwrap();
    groups.sort();
    assert_eq!(groups, vec!["grp".to_string(), "grp_copy".to_string()]);
    assert_userblock_unchanged(&path, &userblock);
}

#[test]
fn userblock_copy_chunked_dataset_roundtrip() {
    // Copying a chunked/filtered dataset enumerates the source chunks (on a
    // base-relative view of the file) and rebuilds the index at the new location.
    let path = temp_path("hdf5_pure_ub_fu_copy_chunk.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    let data: Vec<f64> = (0..600).map(|i| (i % 9) as f64 * 0.25).collect();
    b.create_dataset("c")
        .with_f64_data(&data)
        .with_shape(&[600])
        .with_chunks(&[40])
        .with_deflate(6);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();

    {
        let s = File::open_rw(&path).unwrap();
        s.copy("c", "c_copy").unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("c").unwrap().read_f64().unwrap(), data);
    assert_eq!(file.dataset("c_copy").unwrap().read_f64().unwrap(), data);
    assert_userblock_unchanged(&path, &userblock);
}

// ---- copy (cross-file, into a userblock destination) ----

#[test]
fn userblock_cross_file_copy_into_userblock_dest() {
    // A base-0 source file is copied into a userblock destination. The destination
    // writes the copy base-relative even though the source was read base-0.
    let dst_path = temp_path("hdf5_pure_ub_fu_xcopy_dst.h5");
    let src_path = temp_path("hdf5_pure_ub_fu_xcopy_src.h5");
    let userblock = build_userblock_file(&dst_path);

    // A plain (no-userblock) source file.
    {
        let mut b = FileBuilder::new();
        b.create_dataset("payload").with_i32_data(&[100, 200, 300]);
        let mut g = b.create_group("sub");
        g.create_dataset("leaf").with_f64_data(&[1.25, 2.5]);
        b.add_group(g.finish());
        b.write(&src_path).unwrap();
    }

    {
        let source = File::open(&src_path).unwrap();
        let s = File::open_rw(&dst_path).unwrap();
        s.copy_from(&source, "payload", "imported").unwrap();
        s.copy_from(&source, "sub", "imported_grp").unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&dst_path).unwrap();
    assert_eq!(
        file.dataset("imported").unwrap().read_i32().unwrap(),
        vec![100, 200, 300]
    );
    assert_eq!(
        file.dataset("imported_grp/leaf")
            .unwrap()
            .read_f64()
            .unwrap(),
        vec![1.25, 2.5]
    );
    // Destination originals untouched.
    assert_eq!(
        file.dataset("alpha").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_userblock_unchanged(&dst_path, &userblock);
}

// ---- reference screening (issue #317) ----

/// An object reference this commit writes is screened against the space the
/// same commit reclaims, and on a userblock file the two are in different
/// coordinates: a stored reference is relative to the base address, while the
/// reclaimed spans are absolute file offsets. A screen that compared them
/// directly would miss every hazard on a `.mat`-shaped file and catch a
/// harmless address 512 bytes further on.
///
/// `refs` names `grp/inner`, `grp` is what goes away, and the address reaches
/// the commit as an address rather than as a path — the form that skips
/// `resolve_reference_target` entirely (issue #317).
#[test]
fn userblock_reference_into_deleted_space_is_refused() {
    let path = temp_path("hdf5_pure_ub_fu_ref_delete.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("alpha").with_f64_data(&[1.0, 2.0]);
    let mut g = b.create_group("grp");
    g.create_dataset("inner").with_f64_data(&[7.5, 8.5]);
    b.add_group(g.finish());
    b.create_dataset("refs")
        .with_path_references(&["grp/inner"]);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();

    let stored = {
        let file = File::open(&path).unwrap();
        let raw = file.dataset("refs").unwrap().read_raw().unwrap();
        u64::from_le_bytes(raw[..8].try_into().unwrap())
    };
    // Base-relative: the object is past the 512-byte userblock, and the value
    // stored for it is not.
    assert!(
        stored < UB as u64,
        "expected a base-relative address, got {stored}"
    );

    let before = std::fs::read(&path).unwrap();
    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("added", |b| {
                b.with_reference_data(&[stored]);
            })
            .unwrap();
        s.root().delete("grp").unwrap();
        let err = s.commit().unwrap_err();
        assert!(
            err.to_string()
                .contains("holds the address of an object this commit deletes"),
            "got: {err}"
        );
    }

    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_userblock_unchanged(&path, &userblock);
}

/// A path reference to an object the *same commit* places, on a userblock file.
///
/// The commit preflight replays the apply loop's placement order against
/// placeholder addresses, and those placeholders go through the same
/// `address - base` that converts a real address to its stored, base-relative
/// form. A zero placeholder underflows that on any file with a userblock, so
/// this edit — legal, and fine on a base-0 file — panicked in a debug build.
#[test]
fn userblock_reference_to_an_object_the_same_commit_places() {
    let path = temp_path("hdf5_pure_ub_fu_ref_same_commit.h5");
    let mut b = FileBuilder::new();
    b.with_userblock(UB as u64);
    b.create_dataset("alpha").with_f64_data(&[1.0, 2.0]);
    let mut bytes = b.finish().unwrap();
    let userblock = stamp_userblock(&mut bytes);
    std::fs::write(&path, &bytes).unwrap();

    {
        let s = File::open_rw(&path).unwrap();
        s.root().create_group("g").unwrap();
        s.root()
            .create_dataset("g/inner", |b| {
                b.with_i32_data(&[7, 8, 9]);
            })
            .unwrap();
        s.root()
            .create_dataset("refs", |b| {
                b.with_path_references(&["g/inner"]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let file = File::open(&path).unwrap();
    // The stored value is base-relative, and it resolves to the object this
    // same commit placed.
    let raw = file.dataset("refs").unwrap().read_raw().unwrap();
    let stored = u64::from_le_bytes(raw[..8].try_into().unwrap());
    // Base-relative, so it is smaller than the absolute offset the object sits
    // at. (It is under `UB` here only because this fixture is small; the
    // dereference below is what proves the value is the right one.)
    let absolute = stored + UB as u64;
    assert!(
        stored > 0 && absolute > UB as u64,
        "base-relative: {stored}"
    );
    let targets = file.dataset("refs").unwrap().dereference().unwrap();
    match &targets[0] {
        hdf5_pure::Object::Dataset(ds) => assert_eq!(ds.read_i32().unwrap(), vec![7, 8, 9]),
        other => panic!("expected a dataset reference, got {other:?}"),
    }
    drop(file);
    assert_userblock_unchanged(&path, &userblock);
}
