// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Cross-validation for editing files with a userblock (the userblock slice of
//! issue #104) against the reference C library. A single off-by-base address in
//! an edited file makes the C library error or read garbage, so reading the
//! result back through `hdf5` is the external tripwire that the editor stored
//! every root/link/layout address relative to the base address correctly.

use hdf5::dataset::Layout;
use hdf5::file::LibraryVersion;
use hdf5_pure::{File, FileBuilder};
use tempfile::tempdir;

mod common;
use common::assert_c_absent;

const UB: u64 = 512;

/// Create a userblock file with the reference C library. The closure adds
/// datasets; the file is created with a 512-byte userblock and the modern library
/// bounds so its on-disk format matches what a real `.mat`-style file carries.
fn c_userblock_file(path: &std::path::Path, build: impl FnOnce(&hdf5::File)) {
    let file = hdf5::File::with_options()
        .with_fcpl(|p| {
            p.userblock(UB);
            p
        })
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
    build(&file);
    file.close().unwrap();
    // Sanity: the file really carries a userblock.
    assert_eq!(
        hdf5::File::open(path).unwrap().userblock(),
        UB,
        "C library did not produce a userblock file"
    );
}

#[test]
fn pure_userblock_file_edited_then_read_by_c_library() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("pure_ub.h5");

    let mut b = FileBuilder::new();
    b.with_userblock(UB);
    b.create_dataset("alpha").with_f64_data(&[1.0, 2.0, 3.0]);
    let mut g = b.create_group("grp");
    g.create_dataset("beta").with_i32_data(&[10, 20, 30, 40]);
    b.add_group(g.finish());
    std::fs::write(&path, b.finish().unwrap()).unwrap();

    {
        let s = File::open_rw(&path).unwrap();
        s.dataset("alpha")
            .unwrap()
            .write_staged(|b| {
                b.with_f64_data(&[9.0, 8.0, 7.0]);
            })
            .unwrap();
        s.root()
            .create_dataset("added", |b| {
                b.with_f64_data(&[100.0, 200.0]);
            })
            .unwrap();
        s.root()
            .create_dataset("grp/gamma", |b| {
                b.with_i32_data(&[1, 2, 3]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    // pure reader
    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("alpha").unwrap().read_f64().unwrap(),
        vec![9.0, 8.0, 7.0]
    );
    assert_eq!(
        f.dataset("added").unwrap().read_f64().unwrap(),
        vec![100.0, 200.0]
    );
    assert_eq!(
        f.dataset("grp/gamma").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );

    // reference C library reader — the real interop proof.
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("alpha").unwrap().read_raw::<f64>().unwrap(),
        vec![9.0, 8.0, 7.0]
    );
    assert_eq!(
        c.dataset("added").unwrap().read_raw::<f64>().unwrap(),
        vec![100.0, 200.0]
    );
    assert_eq!(
        c.dataset("grp/beta").unwrap().read_raw::<i32>().unwrap(),
        vec![10, 20, 30, 40]
    );
    assert_eq!(
        c.dataset("grp/gamma").unwrap().read_raw::<i32>().unwrap(),
        vec![1, 2, 3]
    );
}

#[test]
fn real_mat_edited_then_read_by_c_library() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("real.mat");
    std::fs::copy("tests/fixtures/mat_real/test_string_v73.mat", &path).unwrap();

    let before = File::open(&path)
        .unwrap()
        .dataset("string_scalar")
        .unwrap()
        .read_u32()
        .unwrap();

    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("hdf5_pure_probe", |b| {
                b.with_f64_data(&[42.0, -1.0, 3.5]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    // The C library opens the edited real .mat file and reads both the added
    // dataset and an untouched original — proving every address survived the edit
    // relative to the 512-byte MATLAB userblock.
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("hdf5_pure_probe")
            .unwrap()
            .read_raw::<f64>()
            .unwrap(),
        vec![42.0, -1.0, 3.5]
    );
    assert_eq!(
        c.dataset("string_scalar")
            .unwrap()
            .read_raw::<u32>()
            .unwrap(),
        before
    );
}

#[test]
fn userblock_chunked_add_and_overwrite_read_by_c_library() {
    // Adding a chunked/filtered dataset and overwriting an existing one on a
    // userblock file both write chunk-index and chunk-data addresses relative to
    // the base. The C library reading the result back is the external proof that
    // every such address was stored correctly.
    let dir = tempdir().unwrap();
    let path = dir.path().join("pure_ub_chunk.h5");

    let seed = vec![2.0f64; 300];
    let mut b = FileBuilder::new();
    b.with_userblock(UB);
    b.create_dataset("contig").with_i32_data(&[1, 2, 3]);
    b.create_dataset("c")
        .with_f64_data(&seed)
        .with_shape(&[300])
        .with_chunks(&[30])
        .with_deflate(6);
    std::fs::write(&path, b.finish().unwrap()).unwrap();

    let added: Vec<f64> = (0..400).map(|i| (i % 11) as f64 * 0.5).collect();
    let updated: Vec<f64> = (0..300).map(|i| (i as f64).cos()).collect();
    {
        let s = File::open_rw(&path).unwrap();
        // Add a new chunked dataset and relocate-overwrite the existing one.
        s.root()
            .create_dataset("added", |b| {
                b.with_f64_data(&added)
                    .with_shape(&[400])
                    .with_chunks(&[50])
                    .with_deflate(6);
            })
            .unwrap();
        s.dataset("c")
            .unwrap()
            .write_staged(|b| {
                b.with_f64_data(&updated);
            })
            .unwrap();
        s.commit().unwrap();
    }

    // pure reader
    let f = File::open(&path).unwrap();
    assert_eq!(f.dataset("c").unwrap().read_f64().unwrap(), updated);
    assert_eq!(f.dataset("added").unwrap().read_f64().unwrap(), added);

    // reference C library reader — the real interop proof.
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(c.dataset("c").unwrap().read_raw::<f64>().unwrap(), updated);
    assert_eq!(
        c.dataset("added").unwrap().read_raw::<f64>().unwrap(),
        added
    );
    assert_eq!(
        c.dataset("contig").unwrap().read_raw::<i32>().unwrap(),
        vec![1, 2, 3]
    );
}

#[test]
fn userblock_chunked_reclaimed_space_reused_read_by_c_library() {
    // A relocating chunked overwrite frees the old chunk storage; a later commit
    // in the same session reuses it. The C library reading every dataset back
    // confirms the reclaim freed only dead bytes (a base-address mistake in the
    // span enumerators would have corrupted a live region the reuse then wrote).
    let dir = tempdir().unwrap();
    let path = dir.path().join("pure_ub_reclaim.h5");

    let seed = vec![5.0f64; 600];
    let mut b = FileBuilder::new();
    b.with_userblock(UB);
    b.create_dataset("keep").with_f64_data(&[10.0, 20.0, 30.0]);
    b.create_dataset("c")
        .with_f64_data(&seed)
        .with_shape(&[600])
        .with_chunks(&[40])
        .with_deflate(6);
    std::fs::write(&path, b.finish().unwrap()).unwrap();

    let updated: Vec<f64> = (0..600).map(|i| (i as f64) * 0.01).collect();
    let reuse: Vec<f64> = (0..80).map(|i| (i as f64) + 0.5).collect();
    {
        let s = File::open_rw(&path).unwrap();
        s.dataset("c")
            .unwrap()
            .write_staged(|b| {
                b.with_f64_data(&updated);
            })
            .unwrap();
        s.commit().unwrap();
        s.root()
            .create_dataset("reuse", |b| {
                b.with_f64_data(&reuse);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(c.dataset("c").unwrap().read_raw::<f64>().unwrap(), updated);
    assert_eq!(
        c.dataset("reuse").unwrap().read_raw::<f64>().unwrap(),
        reuse
    );
    assert_eq!(
        c.dataset("keep").unwrap().read_raw::<f64>().unwrap(),
        vec![10.0, 20.0, 30.0]
    );
}

#[test]
fn userblock_contiguous_undefined_address_relocates_read_by_c_library() {
    // A contiguous dataset the C library created but never wrote has an undefined
    // data address, so overwriting it relocates the header and writes a fresh data
    // block. On a userblock file the new data address must be stored relative to
    // the base; the C library reading it back confirms it was.
    let dir = tempdir().unwrap();
    let path = dir.path().join("ub_contig_reloc.h5");
    c_userblock_file(&path, |file| {
        // Created without a write: the contiguous data address stays undefined.
        file.new_dataset::<i32>()
            .shape((3,))
            .create("blank")
            .unwrap();
        file.new_dataset::<f64>()
            .shape((2,))
            .create("keep")
            .unwrap()
            .write(&[1.0f64, 2.0])
            .unwrap();
    });

    {
        let s = File::open_rw(&path).unwrap();
        s.dataset("blank")
            .unwrap()
            .write_staged(|b| {
                b.with_i32_data(&[5, 6, 7]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("blank").unwrap().read_i32().unwrap(),
        vec![5, 6, 7]
    );
    assert_eq!(
        f.dataset("keep").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0]
    );

    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("blank").unwrap().read_raw::<i32>().unwrap(),
        vec![5, 6, 7]
    );
    assert_eq!(
        c.dataset("keep").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0]
    );
}

#[test]
fn userblock_compact_overwrite_read_by_c_library() {
    // A compact dataset carries its data inline in the header, so any overwrite
    // rewrites and relocates the header and relinks the parent. On a userblock file
    // that relink must be stored relative to the base; the C library reading the
    // new values back confirms it was.
    let dir = tempdir().unwrap();
    let path = dir.path().join("ub_compact.h5");
    c_userblock_file(&path, |file| {
        file.new_dataset::<i32>()
            .shape((4,))
            .layout(Layout::Compact)
            .create("cpt")
            .unwrap()
            .write(&[1i32, 2, 3, 4])
            .unwrap();
        file.new_dataset::<f64>()
            .shape((2,))
            .create("keep")
            .unwrap()
            .write(&[1.0f64, 2.0])
            .unwrap();
    });

    {
        let s = File::open_rw(&path).unwrap();
        // Same shape, but a compact overwrite always relocates the header.
        s.dataset("cpt")
            .unwrap()
            .write_staged(|b| {
                b.with_i32_data(&[9, 8, 7, 6]);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("cpt").unwrap().read_i32().unwrap(),
        vec![9, 8, 7, 6]
    );

    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("cpt").unwrap().read_raw::<i32>().unwrap(),
        vec![9, 8, 7, 6]
    );
    assert_eq!(
        c.dataset("keep").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0]
    );
}

#[test]
fn userblock_delete_then_reuse_read_by_c_library() {
    // Deleting objects from a userblock file reclaims their storage base-relative;
    // a later add reuses a freed hole. The C library reading every survivor back
    // confirms the reclaim freed only dead bytes — a base mistake in the subtree
    // walk would have freed a live region the reuse then overwrote, corrupting it.
    let dir = tempdir().unwrap();
    let path = dir.path().join("ub_delete_reuse.h5");
    c_userblock_file(&path, |file| {
        let big: Vec<f64> = (0..256).map(|i| i as f64).collect();
        file.new_dataset::<f64>()
            .shape((256,))
            .create("doomed")
            .unwrap()
            .write(&big)
            .unwrap();
        file.new_dataset::<f64>()
            .shape((3,))
            .create("keep")
            .unwrap()
            .write(&[10.0f64, 20.0, 30.0])
            .unwrap();
        // A chunked dataset whose storage is reclaimed via the index walker.
        let chunked: Vec<i32> = (0..512).collect();
        file.new_dataset::<i32>()
            .shape((512,))
            .chunk((64,))
            .create("c")
            .unwrap()
            .write(&chunked)
            .unwrap();
    });

    let reuse: Vec<f64> = (0..64).map(|i| (i as f64) * -1.5).collect();
    {
        let s = File::open_rw(&path).unwrap();
        s.root().delete("doomed").unwrap();
        s.root().delete("c").unwrap();
        s.commit().unwrap();
        s.root()
            .create_dataset("reuse", |b| {
                b.with_f64_data(&reuse);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let f = File::open(&path).unwrap();
    assert!(f.dataset("doomed").is_err());
    assert!(f.dataset("c").is_err());
    assert_eq!(f.dataset("reuse").unwrap().read_f64().unwrap(), reuse);

    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("reuse").unwrap().read_raw::<f64>().unwrap(),
        reuse
    );
    assert_eq!(
        c.dataset("keep").unwrap().read_raw::<f64>().unwrap(),
        vec![10.0, 20.0, 30.0]
    );
    assert_c_absent(&c.dataset("doomed").unwrap_err(), "doomed");
    assert_c_absent(&c.dataset("c").unwrap_err(), "c");
}

#[test]
fn userblock_copy_read_by_c_library() {
    // Copying objects within a userblock file writes fresh data, chunk storage, and
    // link addresses, all base-relative. The C library reading every copy back is
    // the external proof those addresses were stored correctly (a base mistake
    // would dangle the copy's data/index pointer and error or read garbage).
    let dir = tempdir().unwrap();
    let path = dir.path().join("ub_copy.h5");
    c_userblock_file(&path, |file| {
        file.new_dataset::<f64>()
            .shape((4,))
            .create("alpha")
            .unwrap()
            .write(&[1.0f64, 2.0, 3.0, 4.0])
            .unwrap();
        let chunked: Vec<i32> = (0..400).collect();
        file.new_dataset::<i32>()
            .shape((400,))
            .chunk((50,))
            .create("c")
            .unwrap()
            .write(&chunked)
            .unwrap();
        let grp = file.create_group("grp").unwrap();
        grp.new_dataset::<f64>()
            .shape((2,))
            .create("leaf")
            .unwrap()
            .write(&[7.5f64, 8.5])
            .unwrap();
    });

    {
        let s = File::open_rw(&path).unwrap();
        s.copy("alpha", "alpha_copy").unwrap();
        s.copy("c", "c_copy").unwrap();
        s.copy("grp", "grp_copy").unwrap();
        s.commit().unwrap();
    }

    let expected_c: Vec<i32> = (0..400).collect();
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("alpha_copy").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        c.dataset("c_copy").unwrap().read_raw::<i32>().unwrap(),
        expected_c
    );
    assert_eq!(
        c.dataset("grp_copy/leaf")
            .unwrap()
            .read_raw::<f64>()
            .unwrap(),
        vec![7.5, 8.5]
    );
    // Originals untouched.
    assert_eq!(
        c.dataset("alpha").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        c.dataset("c").unwrap().read_raw::<i32>().unwrap(),
        expected_c
    );
}

#[test]
fn userblock_extensible_array_add_read_by_c_library() {
    // An unlimited-maxshape dataset uses the extensible-array chunk index, which
    // embeds many internal addresses built off the planner base. Adding one to a
    // userblock file and reading it back through the C library confirms every EA
    // address was stored relative to the base correctly.
    let dir = tempdir().unwrap();
    let path = dir.path().join("pure_ub_ea.h5");

    let mut b = FileBuilder::new();
    b.with_userblock(UB);
    b.create_dataset("keep").with_i32_data(&[7, 8, 9]);
    std::fs::write(&path, b.finish().unwrap()).unwrap();

    let added: Vec<f64> = (0..500).map(|i| (i as f64).sin() * 1e3).collect();
    {
        let s = File::open_rw(&path).unwrap();
        s.root()
            .create_dataset("ea", |b| {
                b.with_f64_data(&added)
                    .with_shape(&[500])
                    .with_chunks(&[40])
                    .with_maxshape(&[u64::MAX])
                    .with_deflate(6);
            })
            .unwrap();
        s.commit().unwrap();
    }

    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(c.dataset("ea").unwrap().read_raw::<f64>().unwrap(), added);
    assert_eq!(
        c.dataset("keep").unwrap().read_raw::<i32>().unwrap(),
        vec![7, 8, 9]
    );
}
