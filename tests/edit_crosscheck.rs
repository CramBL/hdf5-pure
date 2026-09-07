// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// which is gated to 64-bit little-endian targets; skip them elsewhere so the pure-Rust
// suite can run under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
//! Cross-validation for in-place editing against the reference C library
//! (issue #32): files the C library *writes* are edited in place by
//! `File::open_rw`, and the result is read back by both `hdf5-pure` and the C
//! library. This proves the editor works on files it did not itself produce, in
//! both the HDF5 1.8 (v2 superblock) and 1.10+ (v3 superblock) formats, and
//! including headers the C library splits across multiple chunks (which the
//! editor collapses into a single chunk on rewrite).
//!
//! The default earliest format (version 0 superblock with symbol-table groups)
//! is also editable: each group on the edited path is converted to the latest
//! compact-link format on rewrite, the superblock's root symbol-table entry is
//! repointed, and the result is read back correctly by the C library.

use hdf5::file::LibraryVersion;
use hdf5_pure::{AttrValue, File, FileBuilder, ScaleOffset};
use tempfile::tempdir;

mod common;
use common::{assert_c_absent, create_v18};

/// Stage an add, an add-into-a-group, a delete, and a copy — the full op set.
fn stage_edits(session: &File) {
    session
        .root()
        .create_dataset("added", |b| {
            b.with_f64_data(&[100.0, 200.0]);
        })
        .unwrap();
    session
        .root()
        .create_dataset("grp/gamma", |b| {
            b.with_i32_data(&[1, 2, 3]);
        })
        .unwrap();
    session.root().delete("doomed").unwrap();
    session.copy("alpha", "alpha_copy").unwrap();
}

/// Read the edited file back through both readers and assert every object is
/// correct (and the deleted one is gone).
fn assert_edits_applied(path: &std::path::Path) {
    // hdf5-pure reader.
    let f = File::open(path).unwrap();
    assert_eq!(
        f.dataset("alpha").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        f.dataset("alpha_copy").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        f.dataset("added").unwrap().read_f64().unwrap(),
        vec![100.0, 200.0]
    );
    assert_eq!(
        f.dataset("grp/beta").unwrap().read_i32().unwrap(),
        vec![10, 20, 30, 40]
    );
    assert_eq!(
        f.dataset("grp/gamma").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert!(
        f.dataset("doomed").is_err(),
        "deleted dataset still present (pure)"
    );

    // Reference C library reader — the real interop proof.
    let c = hdf5::File::open(path).unwrap();
    assert_eq!(
        c.dataset("alpha_copy").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0]
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
    assert_c_absent(&c.dataset("doomed").unwrap_err(), "doomed");
}

/// Write the starter file (two root datasets + a group with a dataset) with the
/// C library at the given library-version bounds.
fn write_c_starter(path: &std::path::Path, low: LibraryVersion, high: LibraryVersion) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(low, high))
        .create(path)
        .unwrap();
    file.new_dataset::<f64>()
        .shape((3,))
        .create("alpha")
        .unwrap()
        .write(&[1.0f64, 2.0, 3.0])
        .unwrap();
    file.new_dataset::<i32>()
        .shape((2,))
        .create("doomed")
        .unwrap()
        .write(&[7i32, 8])
        .unwrap();
    let grp = file.create_group("grp").unwrap();
    grp.new_dataset::<i32>()
        .shape((4,))
        .create("beta")
        .unwrap()
        .write(&[10i32, 20, 30, 40])
        .unwrap();
    file.close().unwrap();
}

#[test]
fn pure_written_file_edited_then_read_by_c_library() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("pure.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("alpha").with_f64_data(&[1.0, 2.0, 3.0]);
    b.create_dataset("doomed").with_i32_data(&[7, 8]);
    let mut g = b.create_group("grp");
    g.create_dataset("beta").with_i32_data(&[10, 20, 30, 40]);
    b.add_group(g.finish());
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        stage_edits(&session);
        session.commit().unwrap();
    } // drop the editor (release its exclusive lock) before reading back

    assert_edits_applied(&path);
}

#[test]
fn c_written_v2_file_edited_in_place() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_v2.h5");
    write_c_starter(&path, LibraryVersion::V18, LibraryVersion::V18);
    assert_eq!(File::open(&path).unwrap().superblock().version, 2);

    {
        let session = File::open_rw(&path).unwrap();
        stage_edits(&session);
        session.commit().unwrap();
    } // drop the editor (release its exclusive lock) before reading back

    assert_edits_applied(&path);
}

#[test]
fn c_written_v3_file_edited_in_place() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_v3.h5");
    write_c_starter(&path, LibraryVersion::V110, LibraryVersion::latest());
    assert_eq!(File::open(&path).unwrap().superblock().version, 3);

    {
        let session = File::open_rw(&path).unwrap();
        stage_edits(&session);
        session.commit().unwrap();
    } // drop the editor (release its exclusive lock) before reading back

    assert_edits_applied(&path);
}

#[test]
fn c_multichunk_group_header_is_collapsed_and_edited() {
    // The C library lays a group header out across multiple chunks once it holds
    // enough messages; several root attributes reliably force that. The editor
    // must collapse the continuation chunks into one header on rewrite, preserve
    // the existing messages (the attributes), and apply the edit.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_multichunk.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        file.new_dataset::<f64>()
            .shape((3,))
            .create("alpha")
            .unwrap()
            .write(&[1.0f64, 2.0, 3.0])
            .unwrap();
        file.new_dataset::<i32>()
            .shape((2,))
            .create("doomed")
            .unwrap()
            .write(&[7i32, 8])
            .unwrap();
        let grp = file.create_group("grp").unwrap();
        grp.new_dataset::<i32>()
            .shape((4,))
            .create("beta")
            .unwrap()
            .write(&[10i32, 20, 30, 40])
            .unwrap();
        // Several root-group attributes push the root header past one chunk.
        for i in 0..6 {
            let a = file
                .new_attr::<i64>()
                .shape(())
                .create(format!("meta{i}").as_str())
                .unwrap();
            a.write_scalar(&(i as i64 * 100)).unwrap();
        }
        file.close().unwrap();
    }

    {
        let session = File::open_rw(&path).unwrap();
        stage_edits(&session);
        session.commit().unwrap();
    }

    assert_edits_applied(&path);

    // The C-written root attributes survived the multi-chunk -> single-chunk
    // rewrite of the root header (verified by the C library).
    let c = hdf5::File::open(&path).unwrap();
    for i in 0..6 {
        let v: i64 = c.attr(&format!("meta{i}")).unwrap().read_scalar().unwrap();
        assert_eq!(
            v,
            i as i64 * 100,
            "root attribute meta{i} lost or corrupted"
        );
    }
}

#[test]
fn c_v0_symboltable_file_edited_then_read_by_c_library() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_v0.h5");
    // Earliest low bound yields a version 0 superblock with symbol-table groups.
    write_c_starter(&path, LibraryVersion::Earliest, LibraryVersion::V18);
    assert!(
        File::open(&path).unwrap().superblock().version <= 1,
        "expected a v0/v1 superblock from the earliest libver bound"
    );

    // Add at root, add into the (symbol-table) group, and delete a root dataset.
    // (Copy of the existing v1 objects is not supported, so it is not staged.)
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("added", |b| {
                b.with_f64_data(&[100.0, 200.0]);
            })
            .unwrap();
        session
            .root()
            .create_dataset("grp/gamma", |b| {
                b.with_i32_data(&[1, 2, 3]);
            })
            .unwrap();
        session.root().delete("doomed").unwrap();
        session.commit().unwrap();
    }

    // The superblock stays version 0; the edited groups were converted to v2.
    let f = File::open(&path).unwrap();
    assert!(f.superblock().version <= 1);
    assert_eq!(
        f.dataset("alpha").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        f.dataset("added").unwrap().read_f64().unwrap(),
        vec![100.0, 200.0]
    );
    assert_eq!(
        f.dataset("grp/beta").unwrap().read_i32().unwrap(),
        vec![10, 20, 30, 40]
    );
    assert_eq!(
        f.dataset("grp/gamma").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert!(f.dataset("doomed").is_err());

    // The reference C library reads the edited version-0 file and agrees.
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("alpha").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0]
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
    assert_c_absent(&c.dataset("doomed").unwrap_err(), "doomed");
}

#[test]
fn c_v0_root_attributes_survive_conversion() {
    // A version-0 root is a symbol-table (v1) group; editing converts it to v2.
    // Its existing attributes must survive that conversion, verified by the C
    // library reading them back.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_v0_attrs.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::Earliest, LibraryVersion::V18))
            .create(&path)
            .unwrap();
        file.new_dataset::<f64>()
            .shape((2,))
            .create("d")
            .unwrap()
            .write(&[1.0f64, 2.0])
            .unwrap();
        file.new_attr::<i64>()
            .shape(())
            .create("tag")
            .unwrap()
            .write_scalar(&77i64)
            .unwrap();
        file.close().unwrap();
    }
    assert!(File::open(&path).unwrap().superblock().version <= 1);

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("extra", |b| {
                b.with_i32_data(&[9]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let c = hdf5::File::open(&path).unwrap();
    let tag: i64 = c.attr("tag").unwrap().read_scalar().unwrap();
    assert_eq!(tag, 77, "root attribute lost converting the v0 group to v2");
    assert_eq!(
        c.dataset("d").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        c.dataset("extra").unwrap().read_raw::<i32>().unwrap(),
        vec![9]
    );
}

#[test]
fn c_library_reads_group_attributes_edited_in_place() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_group_attrs.h5");
    let mut b = FileBuilder::new();
    let mut g = b.create_group("grp");
    g.set_attr("count", AttrValue::I64(1));
    g.set_attr("drop", AttrValue::I64(9));
    b.add_group(g.finish());
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .group("grp")
            .unwrap()
            .set_attr("count", AttrValue::I64(2))
            .unwrap();
        session
            .group("grp")
            .unwrap()
            .set_attr("added", AttrValue::I64(3))
            .unwrap();
        session.group("grp").unwrap().remove_attr("drop").unwrap();
        session
            .root()
            .create_group_with("new_grp", |g| {
                g.set_attr("tag", AttrValue::I64(55));
            })
            .unwrap();
        session.commit().unwrap();
    }

    let c = hdf5::File::open(&path).unwrap();
    let grp = c.group("grp").unwrap();
    let count: i64 = grp.attr("count").unwrap().read_scalar().unwrap();
    let added: i64 = grp.attr("added").unwrap().read_scalar().unwrap();
    assert_eq!(count, 2);
    assert_eq!(added, 3);
    assert_c_absent(&grp.attr("drop").unwrap_err(), "grp/@drop");
    let tag: i64 = c
        .group("new_grp")
        .unwrap()
        .attr("tag")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(tag, 55);
}

#[test]
fn free_space_reuse_and_truncation_stay_c_readable() {
    // Free-space management (issue #21): within one session, add a large dataset
    // then delete it. The freed blocks and superseded headers are reclaimed and
    // the file is truncated. The reference C library must still read the survivor
    // correctly from the shrunken file, and its end-of-file must be consistent.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_freespace.h5");
    write_c_starter(&path, LibraryVersion::V110, LibraryVersion::latest());
    let size_start = std::fs::metadata(&path).unwrap().len();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("bulk", |b| {
                b.with_f64_data(&vec![9.0; 2048]);
            })
            .unwrap();
        session.commit().unwrap();
        let size_with_bulk = std::fs::metadata(&path).unwrap().len();
        assert!(size_with_bulk > size_start);

        session.root().delete("bulk").unwrap();
        session.commit().unwrap();
        let size_after = std::fs::metadata(&path).unwrap().len();
        assert!(
            size_after < size_with_bulk,
            "deleting the bulk dataset should shrink the file (was {size_with_bulk}, now {size_after})"
        );
    }

    // hdf5-pure: end-of-file matches the truncated physical size.
    let f = File::open(&path).unwrap();
    assert_eq!(f.file_size(), std::fs::metadata(&path).unwrap().len());
    assert!(f.dataset("bulk").is_err());

    // The reference C library reads the shrunken file and the survivors intact.
    let c = hdf5::File::open(&path).unwrap();
    assert_c_absent(&c.dataset("bulk").unwrap_err(), "bulk");
    assert_eq!(
        c.dataset("alpha").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        c.dataset("grp/beta").unwrap().read_raw::<i32>().unwrap(),
        vec![10, 20, 30, 40]
    );
}

#[test]
fn a_chunked_dataset_written_into_a_freed_hole_stays_c_readable() {
    // Issue #261: a chunked dataset's data region is sized before it is placed, so
    // it can land in a region an earlier commit freed instead of at end-of-file.
    // Every address it carries — each chunk's, the index's, the one in the
    // data-layout message — is then computed from that interior address rather
    // than from the end of the file, which is exactly the class of mistake a
    // reader catches and a size assertion does not. So make the reference C
    // library resolve them: it reads the relocated dataset, and the survivors
    // above and below the hole, from a file it wrote itself.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_chunked_hole.h5");
    write_c_starter(&path, LibraryVersion::V110, LibraryVersion::latest());

    let filtered: Vec<f64> = (0..4096).map(|i| (i % 37) as f64).collect();
    let size_before;
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("victim", |b| {
                b.with_f64_data(&vec![9.0; 4096]).with_chunks(&[512]);
            })
            .unwrap();
        session.commit().unwrap();
        // A dataset *above* the victim, so deleting it leaves an interior hole
        // rather than a trailing run the commit would simply truncate away.
        session
            .root()
            .create_dataset("ceiling", |b| {
                b.with_i32_data(&[4; 256]);
            })
            .unwrap();
        session.commit().unwrap();
        size_before = std::fs::metadata(&path).unwrap().len();

        session.root().delete("victim").unwrap();
        session.commit().unwrap();
        session
            .root()
            .create_dataset("replacement", |b| {
                b.with_f64_data(&filtered)
                    .with_chunks(&[512])
                    .with_deflate(4);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let size_after = std::fs::metadata(&path).unwrap().len();
    assert!(
        size_after <= size_before,
        "the replacement fits the freed hole, so the file must not grow \
         (was {size_before}, now {size_after})"
    );

    let c = hdf5::File::open(&path).unwrap();
    assert_c_absent(&c.dataset("victim").unwrap_err(), "victim");
    assert_eq!(
        c.dataset("replacement").unwrap().read_raw::<f64>().unwrap(),
        filtered,
        "the C library resolves the chunk index of a dataset placed mid-file"
    );
    assert_eq!(
        c.dataset("ceiling").unwrap().read_raw::<i32>().unwrap(),
        vec![4; 256]
    );
    assert_eq!(
        c.dataset("alpha").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
}

#[test]
fn chunked_and_filtered_datasets_added_in_place_are_c_readable() {
    // Issue #76: chunked / filtered / extensible datasets added in place to a
    // file the editor did not write must be read back faithfully by the
    // reference C library — the real interop proof that their chunk data, index,
    // and filter pipeline are emitted correctly.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_chunked_edit.h5");
    write_c_starter(&path, LibraryVersion::V110, LibraryVersion::latest());

    let f64_data: Vec<f64> = (0..400).map(|i| i as f64 * 0.25).collect();
    let i32_data: Vec<i32> = (0..256).map(|i| 1000 + (i % 11)).collect();
    let ext_data: Vec<i32> = (0..128).collect();

    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("deflated", |b| {
                b.with_f64_data(&f64_data)
                    .with_chunks(&[100])
                    .with_deflate(6);
            })
            .unwrap();
        session
            .root()
            .create_dataset("shuffled", |b| {
                b.with_f64_data(&f64_data)
                    .with_chunks(&[64])
                    .with_shuffle()
                    .with_deflate(4);
            })
            .unwrap();
        session
            .root()
            .create_dataset("checked", |b| {
                b.with_i32_data(&i32_data)
                    .with_chunks(&[80])
                    .with_fletcher32();
            })
            .unwrap();
        session
            .root()
            .create_dataset("scaled", |b| {
                b.with_i32_data(&i32_data)
                    .with_chunks(&[80])
                    .with_scale_offset(ScaleOffset::Integer(0));
            })
            .unwrap();
        // Into a group the C library wrote, exercising header relocation.
        session
            .root()
            .create_dataset("grp/grid", |b| {
                b.with_i32_data(&(0..(6 * 4)).collect::<Vec<i32>>())
                    .with_shape(&[6, 4])
                    .with_chunks(&[4, 3]);
            })
            .unwrap();
        // Extensible (unlimited) dataset → Extensible-Array chunk index.
        session
            .root()
            .create_dataset("stream", |b| {
                b.with_i32_data(&ext_data)
                    .with_shape(&[128])
                    .with_maxshape(&[u64::MAX])
                    .with_chunks(&[32]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    // hdf5-pure reads everything back.
    {
        let f = File::open(&path).unwrap();
        assert_eq!(f.dataset("deflated").unwrap().read_f64().unwrap(), f64_data);
        assert_eq!(f.dataset("shuffled").unwrap().read_f64().unwrap(), f64_data);
        assert_eq!(f.dataset("checked").unwrap().read_i32().unwrap(), i32_data);
        assert_eq!(f.dataset("scaled").unwrap().read_i32().unwrap(), i32_data);
        assert_eq!(
            f.dataset("grp/grid").unwrap().read_i32().unwrap(),
            (0..(6 * 4)).collect::<Vec<i32>>()
        );
        assert_eq!(f.dataset("stream").unwrap().read_i32().unwrap(), ext_data);
        // The C-written survivors are intact.
        assert_eq!(
            f.dataset("alpha").unwrap().read_f64().unwrap(),
            vec![1.0, 2.0, 3.0]
        );
    }

    // The reference C library reads the filtered/chunked/extensible additions and
    // sees them as chunked storage.
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("deflated").unwrap().read_raw::<f64>().unwrap(),
        f64_data
    );
    assert_eq!(
        c.dataset("shuffled").unwrap().read_raw::<f64>().unwrap(),
        f64_data
    );
    assert_eq!(
        c.dataset("checked").unwrap().read_raw::<i32>().unwrap(),
        i32_data
    );
    assert_eq!(
        c.dataset("scaled").unwrap().read_raw::<i32>().unwrap(),
        i32_data
    );
    assert_eq!(
        c.dataset("grp/grid").unwrap().read_raw::<i32>().unwrap(),
        (0..(6 * 4)).collect::<Vec<i32>>()
    );
    assert_eq!(
        c.dataset("stream").unwrap().read_raw::<i32>().unwrap(),
        ext_data
    );
    for name in [
        "deflated", "shuffled", "checked", "scaled", "grp/grid", "stream",
    ] {
        assert!(
            c.dataset(name).unwrap().chunk().is_some(),
            "C library does not see {name} as chunked"
        );
    }
    // The C-written original is untouched.
    assert_eq!(
        c.dataset("alpha").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
}

#[test]
fn deleting_chunked_datasets_in_place_stays_c_readable() {
    // Reclaiming chunked storage on delete (issue #77) must leave a file the
    // reference C library still reads. The starter is written by the C library
    // with HDF5 1.8 bounds, so its chunked dataset uses a *B-tree v1* index — the
    // foreign layout the editor's own writer never emits. The editor deletes it
    // in place, then churns an editor-written (Fixed Array) chunked dataset to
    // prove its storage is reclaimed and reused rather than leaked. Finally both
    // readers see only the contiguous survivor in the shrunken, valid file.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_chunked_delete.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V18, LibraryVersion::V18))
            .create(&path)
            .unwrap();
        file.new_dataset::<f64>()
            .shape((3,))
            .create("keep")
            .unwrap()
            .write(&[1.0f64, 2.0, 3.0])
            .unwrap();
        let chunked: Vec<i32> = (0..2048).collect();
        file.new_dataset::<i32>()
            .shape((2048,))
            .chunk((256,)) // 8 chunks of 256 i32 = 1024 bytes each, B-tree v1
            .create("c_chunked")
            .unwrap()
            .write(&chunked)
            .unwrap();
        file.close().unwrap();
    }

    let with_c_chunked = std::fs::metadata(&path).unwrap().len();

    {
        let session = File::open_rw(&path).unwrap();
        // Reclaim the C-written B-tree-v1 chunked dataset.
        session.root().delete("c_chunked").unwrap();
        session.commit().unwrap();
        // Re-add *contiguous* datasets sized to the reclaimed 1024-byte chunk
        // holes. (Chunked adds always append their blob, so contiguous adds are
        // what exercise reuse of a freed interior hole.) Six 1024-byte blocks fit
        // within the ~8 KB the delete freed, so the file barely grows; if the
        // B-tree-v1 storage had leaked instead, they would all append and the
        // file would grow by ~6 KB, failing the bound below.
        for i in 0..6 {
            session
                .root()
                .create_dataset(&format!("r{i}"), |b| {
                    b.with_i32_data(&vec![i; 256]);
                })
                .unwrap(); // 256 i32 = 1024 bytes, one chunk hole
            session.commit().unwrap();
        }
    }
    let after = std::fs::metadata(&path).unwrap().len();
    assert!(
        after < with_c_chunked + 4096,
        "deleting the C-written B-tree-v1 chunked dataset must reclaim its storage \
         for the contiguous re-adds to reuse, not leak it (was {with_c_chunked}, \
         after six 1 KiB re-adds {after})"
    );

    // hdf5-pure: the deleted dataset is gone, the survivor and re-adds read back,
    // and the recorded end-of-file matches the physical size.
    let f = File::open(&path).unwrap();
    assert_eq!(f.file_size(), std::fs::metadata(&path).unwrap().len());
    assert!(f.dataset("c_chunked").is_err());
    assert_eq!(
        f.dataset("keep").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    for i in 0..6 {
        assert_eq!(
            f.dataset(&format!("r{i}")).unwrap().read_i32().unwrap(),
            vec![i; 256]
        );
    }

    // The reference C library reads the reclaimed file too.
    let c = hdf5::File::open(&path).unwrap();
    assert_c_absent(&c.dataset("c_chunked").unwrap_err(), "c_chunked");
    assert_eq!(
        c.dataset("keep").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    for i in 0..6 {
        assert_eq!(
            c.dataset(&format!("r{i}"))
                .unwrap()
                .read_raw::<i32>()
                .unwrap(),
            vec![i; 256]
        );
    }
}

#[test]
fn overwriting_chunked_datasets_in_place_stays_c_readable() {
    // Issue #101: overwriting a chunked dataset's values in place must leave a
    // file the reference C library still reads. Two paths are exercised:
    //  - a C-written HDF5-1.8 dataset uses a *B-tree v1* chunk index (the foreign
    //    layout the editor never emits); an unfiltered same-shape overwrite reuses
    //    its chunk slots without rewriting the index, so it stays B-tree v1; and
    //  - an editor-written deflate dataset whose values change compressibility is
    //    rebuilt and relocated, staying chunked + compressed.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_chunked_overwrite.h5");
    let btree: Vec<i32> = (0..2048).collect();
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V18, LibraryVersion::V18))
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape((2048,))
            .chunk((256,)) // 8 chunks, B-tree v1 index
            .create("c_chunked")
            .unwrap()
            .write(&btree)
            .unwrap();
        file.close().unwrap();
    }

    // Add an editor-written deflate dataset (highly compressible to start).
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("deflated", |b| {
                b.with_i32_data(&vec![0i32; 2048])
                    .with_shape(&[2048])
                    .with_chunks(&[256])
                    .with_deflate(6);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let size_before = std::fs::metadata(&path).unwrap().len();

    // Overwrite both: the B-tree-v1 one in place (unfiltered, same slots), the
    // deflate one with incompressible values (forces a relocate).
    let btree_new: Vec<i32> = btree.iter().rev().copied().collect();
    let deflate_new: Vec<i32> = (0..2048i32)
        .map(|i| i.wrapping_mul(2_654_435_761u32 as i32) ^ (i << 3))
        .collect();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("c_chunked")
            .unwrap()
            .write_staged(|b| {
                b.with_i32_data(&btree_new).with_shape(&[2048]);
            })
            .unwrap();
        session
            .dataset("deflated")
            .unwrap()
            .write_staged(|b| {
                b.with_i32_data(&deflate_new).with_shape(&[2048]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    // The unfiltered B-tree-v1 overwrite touched only chunk payload bytes, so the
    // file grew only by the relocated deflate dataset's storage.
    assert!(std::fs::metadata(&path).unwrap().len() >= size_before);

    // hdf5-pure reads both back.
    {
        let f = File::open(&path).unwrap();
        assert_eq!(
            f.dataset("c_chunked").unwrap().read_i32().unwrap(),
            btree_new
        );
        assert_eq!(
            f.dataset("deflated").unwrap().read_i32().unwrap(),
            deflate_new
        );
    }
    // The reference C library reads both and still sees them as chunked.
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("c_chunked").unwrap().read_raw::<i32>().unwrap(),
        btree_new
    );
    assert_eq!(
        c.dataset("deflated").unwrap().read_raw::<i32>().unwrap(),
        deflate_new
    );
    for name in ["c_chunked", "deflated"] {
        assert!(
            c.dataset(name).unwrap().chunk().is_some(),
            "C library does not see {name} as chunked after overwrite"
        );
    }
}

#[test]
fn fits_with_slack_filtered_overwrite_stays_c_readable() {
    // Issue #101 follow-up: when a filtered chunked dataset is overwritten with
    // more-compressible values, the re-encoded chunks shrink and fit their slots,
    // so the chunk index is rebuilt *in place* to record the new sizes. The
    // reference C library must still read the rebuilt index and the new values.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_fits_with_slack.h5");
    write_c_starter(&path, LibraryVersion::V110, LibraryVersion::latest());

    // Editor writes an incompressible deflate dataset (large chunk slots, v4
    // Fixed-Array index).
    let orig: Vec<i32> = (0..2048i32)
        .map(|i| i.wrapping_mul(2_654_435_761u32 as i32) ^ (i << 3))
        .collect();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("d", |b| {
                b.with_i32_data(&orig)
                    .with_shape(&[2048])
                    .with_chunks(&[512])
                    .with_deflate(6);
            })
            .unwrap();
        session.commit().unwrap();
    }
    let size_before = std::fs::metadata(&path).unwrap().len();

    // Overwrite with highly compressible values: chunks shrink and fit with slack.
    let updated: Vec<i32> = vec![5; 2048];
    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("d")
            .unwrap()
            .write_staged(|b| {
                b.with_i32_data(&updated).with_shape(&[2048]);
            })
            .unwrap();
        session.commit().unwrap();
    }
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        size_before,
        "fits-with-slack overwrite should reuse slots and rebuild the index in place"
    );

    // hdf5-pure reads the new values.
    assert_eq!(
        File::open(&path)
            .unwrap()
            .dataset("d")
            .unwrap()
            .read_i32()
            .unwrap(),
        updated
    );
    // The reference C library reads the in-place-rebuilt index + new values, and
    // still sees the dataset as chunked.
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(c.dataset("d").unwrap().read_raw::<i32>().unwrap(), updated);
    assert!(
        c.dataset("d").unwrap().chunk().is_some(),
        "C library does not see the dataset as chunked after in-place index rebuild"
    );
}

#[test]
fn copying_chunked_datasets_in_place_stays_c_readable() {
    // Issue #101: copying a chunked/filtered dataset in place must leave a file
    // the reference C library still reads, with the copy chunked + compressed. A
    // C-written B-tree-v1 source is reproduced with a v4 index (the writer emits
    // only single/fixed/extensible), which the C library reads as chunked too.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_chunked_copy.h5");
    let btree: Vec<i32> = (0..2048).collect();
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V18, LibraryVersion::V18))
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape((2048,))
            .chunk((256,)) // B-tree v1 index
            .create("c_chunked")
            .unwrap()
            .write(&btree)
            .unwrap();
        file.close().unwrap();
    }
    // An editor-written, highly compressible deflate dataset to copy too.
    let deflate: Vec<i32> = (0..4096).map(|i| i % 4).collect();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("deflated", |b| {
                b.with_i32_data(&deflate)
                    .with_shape(&[4096])
                    .with_chunks(&[512])
                    .with_shuffle()
                    .with_deflate(6);
            })
            .unwrap();
        session.commit().unwrap();
    }
    {
        let session = File::open_rw(&path).unwrap();
        session.copy("c_chunked", "c_chunked_copy").unwrap();
        session.copy("deflated", "deflated_copy").unwrap();
        session.commit().unwrap();
    }

    // hdf5-pure: sources untouched, copies correct.
    {
        let f = File::open(&path).unwrap();
        assert_eq!(f.dataset("c_chunked").unwrap().read_i32().unwrap(), btree);
        assert_eq!(
            f.dataset("c_chunked_copy").unwrap().read_i32().unwrap(),
            btree
        );
        assert_eq!(f.dataset("deflated").unwrap().read_i32().unwrap(), deflate);
        assert_eq!(
            f.dataset("deflated_copy").unwrap().read_i32().unwrap(),
            deflate
        );
    }
    // The reference C library reads the copies and sees them as chunked.
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("c_chunked_copy")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        btree
    );
    assert_eq!(
        c.dataset("deflated_copy")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        deflate
    );
    for name in ["c_chunked_copy", "deflated_copy"] {
        assert!(
            c.dataset(name).unwrap().chunk().is_some(),
            "C library does not see {name} as chunked after copy"
        );
    }
}

#[test]
fn overwriting_multiply_linked_chunked_that_relocates_is_refused() {
    // A relocating chunked overwrite (a filtered dataset whose re-encoded chunks
    // change size) moves the object header, so only one of several hard links
    // could be repointed; it is refused when the object has more than one, and the
    // file is left untouched.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_chunked_multilink.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        file.new_dataset::<f64>()
            .shape((3,))
            .create("anchor")
            .unwrap();
        file.close().unwrap();
    }
    // Editor adds a deflate dataset (highly compressible).
    {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("d", |b| {
                b.with_i32_data(&vec![0i32; 1024])
                    .with_shape(&[1024])
                    .with_chunks(&[256])
                    .with_deflate(6);
            })
            .unwrap();
        session.commit().unwrap();
    }
    // C library adds a second hard link to it.
    {
        let file = hdf5::File::open_rw(&path).unwrap();
        file.link_hard("d", "d_alias").unwrap();
        file.close().unwrap();
    }
    let before = std::fs::read(&path).unwrap();
    {
        let session = File::open_rw(&path).unwrap();
        // Incompressible new values change the stored size => relocate.
        let updated: Vec<i32> = (0..1024i32)
            .map(|i| i.wrapping_mul(2_654_435_761u32 as i32) ^ (i << 3))
            .collect();
        session
            .dataset("d")
            .unwrap()
            .write_staged(|b| {
                b.with_i32_data(&updated).with_shape(&[1024]);
            })
            .unwrap();
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string().contains("single hard link"),
            "expected multi-link relocate refusal, got: {err}"
        );
    }
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "file modified on refusal"
    );
}

#[test]
fn deleting_one_of_several_hard_links_keeps_the_survivor() {
    // Issue #77 review (finding #1): an HDF5 object can have several hard links.
    // Deleting ONE link must not reclaim the object's storage while another link
    // still references it — freeing it would corrupt the survivor once the bytes
    // are reused. Covers a chunked (Fixed Array) object and a contiguous one,
    // each linked twice; one link of each is deleted, then churn forces the
    // allocator to reuse anything wrongly freed.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_hardlink_survivor.h5");
    let chunked: Vec<i32> = (0..512).collect();
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape((512,))
            .chunk((64,))
            .create("chunked_orig")
            .unwrap()
            .write(&chunked)
            .unwrap();
        file.link_hard("/chunked_orig", "chunked_alias").unwrap();
        file.new_dataset::<f64>()
            .shape((4,))
            .create("contig_orig")
            .unwrap()
            .write(&[1.0f64, 2.0, 3.0, 4.0])
            .unwrap();
        file.link_hard("/contig_orig", "contig_alias").unwrap();
        file.close().unwrap();
    }

    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("chunked_orig").unwrap();
        session.root().delete("contig_orig").unwrap();
        session.commit().unwrap();
        for i in 0..6 {
            session
                .root()
                .create_dataset(&format!("f{i}"), |b| {
                    b.with_i32_data(&vec![i; 300]);
                })
                .unwrap();
            session.commit().unwrap();
            session.root().delete(&format!("f{i}")).unwrap();
            session.commit().unwrap();
        }
    }

    // The surviving aliases must still read their data through both readers; the
    // deleted link names are gone.
    let f = File::open(&path).unwrap();
    assert!(f.dataset("chunked_orig").is_err());
    assert!(f.dataset("contig_orig").is_err());
    assert_eq!(
        f.dataset("chunked_alias").unwrap().read_i32().unwrap(),
        chunked
    );
    assert_eq!(
        f.dataset("contig_alias").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );

    let c = hdf5::File::open(&path).unwrap();
    assert_c_absent(&c.dataset("chunked_orig").unwrap_err(), "chunked_orig");
    assert_eq!(
        c.dataset("chunked_alias")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        chunked
    );
    assert_eq!(
        c.dataset("contig_alias")
            .unwrap()
            .read_raw::<f64>()
            .unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn deleting_all_hard_links_to_an_object_in_one_commit_is_safe() {
    // Issue #77 review (finding #2): deleting every hard link to a chunked object
    // in a single commit must not double-free its storage (a debug-build panic in
    // the free list) nor corrupt the file. The storage is conservatively left as
    // dead bytes; the file stays valid and the unrelated survivor reads back.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_hardlink_all.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape((512,))
            .chunk((64,))
            .create("orig")
            .unwrap()
            .write(&(0..512).collect::<Vec<i32>>())
            .unwrap();
        file.link_hard("/orig", "alias").unwrap();
        file.new_dataset::<f64>()
            .shape((2,))
            .create("keep")
            .unwrap()
            .write(&[9.0f64, 8.0])
            .unwrap();
        file.close().unwrap();
    }
    {
        let session = File::open_rw(&path).unwrap();
        session.root().delete("orig").unwrap();
        session.root().delete("alias").unwrap(); // both links to the same object, one commit
        session.commit().unwrap();
    }
    let f = File::open(&path).unwrap();
    assert!(f.dataset("orig").is_err());
    assert!(f.dataset("alias").is_err());
    assert_eq!(
        f.dataset("keep").unwrap().read_f64().unwrap(),
        vec![9.0, 8.0]
    );
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("keep").unwrap().read_raw::<f64>().unwrap(),
        vec![9.0, 8.0]
    );
}

#[test]
fn cross_file_copy_from_read_by_c_library() {
    // A cross-file `copy_from` must produce a destination the reference C library
    // can read: the copied object's verbatim headers, data, and attributes have to
    // be valid in the new file, not just resolvable by this crate's own reader.
    let dir = tempdir().unwrap();
    let src_path = dir.path().join("xsrc.h5");
    let dst_path = dir.path().join("xdst.h5");

    // Source: an attributed dataset and a group subtree. Written with this crate
    // (compact attributes, which the verbatim copy reproduces — the C library
    // stores even a single latest-format attribute densely, which the copy path
    // does not yet handle for either same- or cross-file copies).
    {
        let mut b = FileBuilder::new();
        let ds = b.create_dataset("calibration");
        ds.with_f64_data(&[0.99, 1.0, 1.01]);
        ds.set_attr("revision", AttrValue::I64(7));
        let mut bundle = b.create_group("bundle");
        bundle.create_dataset("inner").with_i32_data(&[5, 6]);
        b.add_group(bundle.finish());
        b.write(&src_path).unwrap();
    }

    // Destination: a C-written starter (alpha, doomed, grp/beta).
    write_c_starter(&dst_path, LibraryVersion::V110, LibraryVersion::latest());

    {
        let source = File::open(&src_path).unwrap();
        let session = File::open_rw(&dst_path).unwrap();
        session
            .copy_from(&source, "calibration", "calibration")
            .unwrap();
        session.copy_from(&source, "bundle", "bundle").unwrap();
        session.commit().unwrap();
    } // drop the editor (release its exclusive lock) before reading back

    // hdf5-pure reader.
    let f = File::open(&dst_path).unwrap();
    assert_eq!(
        f.dataset("calibration").unwrap().read_f64().unwrap(),
        vec![0.99, 1.0, 1.01]
    );
    assert_eq!(
        f.dataset("bundle/inner").unwrap().read_i32().unwrap(),
        vec![5, 6]
    );

    // Reference C library reader — the interop proof.
    let c = hdf5::File::open(&dst_path).unwrap();
    assert_eq!(
        c.dataset("calibration").unwrap().read_raw::<f64>().unwrap(),
        vec![0.99, 1.0, 1.01]
    );
    let revision: i64 = c
        .dataset("calibration")
        .unwrap()
        .attr("revision")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(revision, 7);
    assert_eq!(
        c.dataset("bundle/inner")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        vec![5, 6]
    );
    // The destination's pre-existing objects survive.
    assert_eq!(
        c.dataset("alpha").unwrap().read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
}

#[test]
fn cross_file_copy_from_c_written_attributed_dataset() {
    // A C-library-written object with a few attributes stores them compactly but
    // also carries an Attribute Info message with an *undefined* heap address; the
    // copy path must treat that as compact (not dense), so `copy_from` succeeds and
    // the C library reads the copy back — attributes included.
    let dir = tempdir().unwrap();
    let src_path = dir.path().join("c_attr_src.h5");
    let dst_path = dir.path().join("c_attr_dst.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&src_path)
            .unwrap();
        let ds = file
            .new_dataset::<f64>()
            .shape((3,))
            .create("calibration")
            .unwrap();
        ds.write(&[0.99f64, 1.0, 1.01]).unwrap();
        ds.new_attr::<i64>()
            .shape(())
            .create("revision")
            .unwrap()
            .write_scalar(&7i64)
            .unwrap();
        ds.new_attr::<f64>()
            .shape(())
            .create("gain")
            .unwrap()
            .write_scalar(&2.5f64)
            .unwrap();
        file.close().unwrap();
    }
    write_c_starter(&dst_path, LibraryVersion::V110, LibraryVersion::latest());

    {
        let source = File::open(&src_path).unwrap();
        let session = File::open_rw(&dst_path).unwrap();
        session
            .copy_from(&source, "calibration", "calibration")
            .unwrap();
        session.commit().unwrap();
    }

    let c = hdf5::File::open(&dst_path).unwrap();
    assert_eq!(
        c.dataset("calibration").unwrap().read_raw::<f64>().unwrap(),
        vec![0.99, 1.0, 1.01]
    );
    let revision: i64 = c
        .dataset("calibration")
        .unwrap()
        .attr("revision")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(revision, 7);
    let gain: f64 = c
        .dataset("calibration")
        .unwrap()
        .attr("gain")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(gain, 2.5);
}

#[test]
fn same_file_copy_of_c_written_attributed_object() {
    // The same compact-attribute fix applies to the in-file `copy`: a C-written
    // attributed object is now copyable in place and read back by the C library.
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_attr_infile.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        let ds = file.new_dataset::<i32>().shape((2,)).create("src").unwrap();
        ds.write(&[10i32, 20]).unwrap();
        ds.new_attr::<i64>()
            .shape(())
            .create("tag")
            .unwrap()
            .write_scalar(&99i64)
            .unwrap();
        file.close().unwrap();
    }

    {
        let session = File::open_rw(&path).unwrap();
        session.copy("src", "dup").unwrap();
        session.commit().unwrap();
    }

    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("dup").unwrap().read_raw::<i32>().unwrap(),
        vec![10, 20]
    );
    let tag: i64 = c
        .dataset("dup")
        .unwrap()
        .attr("tag")
        .unwrap()
        .read_scalar()
        .unwrap();
    assert_eq!(tag, 99);
}

#[test]
fn cross_file_copy_from_reproduces_c_written_dense_attributes() {
    // Above the compact threshold (8 attributes) the C library stores attributes
    // densely — a real fractal heap. A cross-file copy of a *fixed-size* dense
    // attribute set is now reproduced (issue #87): the attributes are read out of
    // the source heap and a fresh heap is built in the destination, which the C
    // library then reads back. (A variable-length/reference dense set is still
    // refused cross-file; see `edit_dense_attr_copy.rs`.)
    let dir = tempdir().unwrap();
    let src_path = dir.path().join("dense_src.h5");
    let dst_path = dir.path().join("dense_dst.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&src_path)
            .unwrap();
        let ds = file.new_dataset::<i32>().shape((1,)).create("ds").unwrap();
        ds.write(&[1i32]).unwrap();
        for i in 0..12 {
            ds.new_attr::<i64>()
                .shape(())
                .create(format!("a{i}").as_str())
                .unwrap()
                .write_scalar(&(i as i64))
                .unwrap();
        }
        file.close().unwrap();
    }
    write_c_starter(&dst_path, LibraryVersion::V110, LibraryVersion::latest());

    {
        let source = File::open(&src_path).unwrap();
        let session = File::open_rw(&dst_path).unwrap();
        session.copy_from(&source, "ds", "dup").unwrap();
        session.commit().unwrap();
    }

    let c = hdf5::File::open(&dst_path).unwrap();
    let ds = c.dataset("dup").unwrap();
    assert_eq!(ds.read_raw::<i32>().unwrap(), vec![1]);
    for i in 0..12 {
        let got: i64 = ds.attr(&format!("a{i}")).unwrap().read_scalar().unwrap();
        assert_eq!(got, i as i64, "dense attr a{i} mismatch");
    }
}

// ---- write_dataset (issue #79): value-overwrite crosschecks ----

/// Overwrite a C-library-written contiguous dataset in place (same size) and read
/// the new values back through both readers. Covers the no-relocation fast path,
/// in both the HDF5 1.8 (v2 superblock) and 1.10+ (v3 superblock) formats.
#[test]
fn write_dataset_same_size_crosscheck() {
    for (low, high) in [
        (LibraryVersion::V18, LibraryVersion::V18),
        (LibraryVersion::V110, LibraryVersion::latest()),
    ] {
        let dir = tempdir().unwrap();
        let path = dir.path().join("w.h5");
        {
            let file = hdf5::File::with_options()
                .with_fapl(|p| p.libver_bounds(low, high))
                .create(&path)
                .unwrap();
            file.new_dataset::<f64>()
                .shape((4,))
                .create("d")
                .unwrap()
                .write(&[1.0f64, 2.0, 3.0, 4.0])
                .unwrap();
            file.close().unwrap();
        }

        {
            let session = File::open_rw(&path).unwrap();
            session
                .dataset("d")
                .unwrap()
                .write_staged(|b| {
                    b.with_f64_data(&[9.0, 8.0, 7.0, 6.0]);
                })
                .unwrap();
            session.commit().unwrap();
        }

        let pure = File::open(&path).unwrap();
        assert_eq!(
            pure.dataset("d").unwrap().read_f64().unwrap(),
            vec![9.0, 8.0, 7.0, 6.0]
        );
        let c = hdf5::File::open(&path).unwrap();
        assert_eq!(
            c.dataset("d").unwrap().read_raw::<f64>().unwrap(),
            vec![9.0, 8.0, 7.0, 6.0]
        );
    }
}

/// Overwrite a dataset the C library created but never wrote — its contiguous
/// data address is undefined, so the overwrite relocates the header, repoints the
/// data-layout message, and relinks the parent. Read the result back with both
/// readers (the relocation path proven against the reference library).
#[test]
fn write_dataset_undefined_address_relocates_crosscheck() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        // `new_dataset(...).create(...)` creates without writing: a contiguous
        // dataset whose data address stays undefined until data is written.
        file.new_dataset::<i32>()
            .shape((3,))
            .create("blank")
            .unwrap();
        // A neighbour dataset so the group keeps a stable, re-linkable parent.
        file.new_dataset::<f64>()
            .shape((2,))
            .create("keep")
            .unwrap()
            .write(&[1.0f64, 2.0])
            .unwrap();
        file.close().unwrap();
    }

    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("blank")
            .unwrap()
            .write_staged(|b| {
                b.with_i32_data(&[5, 6, 7]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    let pure = File::open(&path).unwrap();
    assert_eq!(
        pure.dataset("blank").unwrap().read_i32().unwrap(),
        vec![5, 6, 7]
    );
    assert_eq!(
        pure.dataset("keep").unwrap().read_f64().unwrap(),
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

/// A dataset reachable by two hard links shares a single object header. A
/// same-size in-place overwrite does not relocate that header, so both names see
/// the new data — verified through both readers.
#[test]
fn write_dataset_shared_hard_link_crosscheck() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("shared.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        file.new_dataset::<i32>()
            .shape((3,))
            .create("a")
            .unwrap()
            .write(&[1i32, 2, 3])
            .unwrap();
        // Second hard link to the very same object header.
        file.link_hard("a", "b").unwrap();
        file.close().unwrap();
    }

    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("a")
            .unwrap()
            .write_staged(|b| {
                b.with_i32_data(&[40, 50, 60]);
            })
            .unwrap();
        session.commit().unwrap();
    }

    // Both links see the new data through both readers.
    let pure = File::open(&path).unwrap();
    assert_eq!(
        pure.dataset("a").unwrap().read_i32().unwrap(),
        vec![40, 50, 60]
    );
    assert_eq!(
        pure.dataset("b").unwrap().read_i32().unwrap(),
        vec![40, 50, 60]
    );
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("a").unwrap().read_raw::<i32>().unwrap(),
        vec![40, 50, 60]
    );
    assert_eq!(
        c.dataset("b").unwrap().read_raw::<i32>().unwrap(),
        vec![40, 50, 60]
    );
}

/// A relocating overwrite (here: filling a never-written, undefined-address
/// dataset) of a multiply-hard-linked object is refused, because only one of its
/// parent links could be repointed at the moved header — the others would diverge.
/// The file is left untouched.
#[test]
fn write_dataset_relocate_with_multiple_hard_links_is_refused() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("shared_relocate.h5");
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        // Created but never written: contiguous address undefined, so an overwrite
        // relocates the header.
        file.new_dataset::<i32>().shape((3,)).create("a").unwrap();
        file.link_hard("a", "b").unwrap();
        file.close().unwrap();
    }
    let before = std::fs::read(&path).unwrap();
    {
        let session = File::open_rw(&path).unwrap();
        session
            .dataset("a")
            .unwrap()
            .write_staged(|b| {
                b.with_i32_data(&[1, 2, 3]);
            })
            .unwrap();
        let err = session.commit().unwrap_err();
        assert!(
            err.to_string().contains("single hard link"),
            "expected multi-link refusal, got: {err}"
        );
    }
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "file modified on refusal"
    );
}

/// A delete and a create at the same path in one commit (issue #305), on files
/// the C library wrote in both formats, read back by the C library.
///
/// The replacement changes the dataset's length, so a file where the original
/// survived — or where its link was left beside the replacement's — reads as the
/// wrong length rather than as plausible data. The *datatype* does no work here:
/// the C library converts silently, and `read_raw::<i32>()` on a surviving f64
/// original returns values rather than an error. `h5py` and `h5dump` go through
/// this same reader.
#[test]
fn a_replaced_object_is_read_by_the_c_library() {
    for (name, low, high, want_sb) in [
        ("replace_v2.h5", LibraryVersion::V18, LibraryVersion::V18, 2),
        (
            "replace_v3.h5",
            LibraryVersion::V110,
            LibraryVersion::latest(),
            3,
        ),
    ] {
        let dir = tempdir().unwrap();
        let path = dir.path().join(name);
        write_c_starter(&path, low, high);
        assert_eq!(File::open(&path).unwrap().superblock().version, want_sb);

        {
            let session = File::open_rw(&path).unwrap();
            // A dataset replaced at its own path, and a group replaced with its
            // whole subtree — the two shapes a rotation takes.
            session.root().delete("alpha").unwrap();
            session
                .root()
                .create_dataset("alpha", |b| {
                    b.with_i32_data(&[11, 22]);
                })
                .unwrap();
            session.root().delete("grp").unwrap();
            session
                .root()
                .create_group_with("grp", |g| {
                    g.set_attr("generation", AttrValue::I64(2));
                    g.create_dataset("delta", |b| {
                        b.with_f64_data(&[9.5]);
                    });
                })
                .unwrap();
            session.commit().unwrap();
        } // drop the editor (release its exclusive lock) before reading back

        let c = hdf5::File::open(&path).unwrap();
        assert_eq!(
            c.dataset("alpha").unwrap().read_raw::<i32>().unwrap(),
            vec![11, 22],
            "{name}: the C library sees the original rather than the replacement"
        );
        assert_eq!(
            c.dataset("grp/delta").unwrap().read_raw::<f64>().unwrap(),
            vec![9.5]
        );
        assert_c_absent(&c.dataset("grp/beta").unwrap_err(), "grp/beta");
        assert_eq!(
            c.group("grp")
                .unwrap()
                .attr("generation")
                .unwrap()
                .read_scalar::<i64>()
                .unwrap(),
            2
        );
        // The untouched neighbour, to show the rotation did not disturb the rest
        // of the group it rebuilt.
        assert_eq!(
            c.dataset("doomed").unwrap().read_raw::<i32>().unwrap(),
            vec![7, 8]
        );
    }
}

// ---- hard-linked groups (issue #327) ----

/// A file the C library writes: `/g` holding one dataset, hard-linked as
/// `/alias`, plus an unrelated `/keep` so the refusals below can be shown to
/// leave the rest of the file alone.
fn write_hard_linked_group(path: &std::path::Path) {
    let file = hdf5::File::create(path).unwrap();
    let g = file.create_group("g").unwrap();
    g.new_dataset::<i32>()
        .shape((3,))
        .create("inner")
        .unwrap()
        .write(&[1i32, 2, 3])
        .unwrap();
    file.link_hard("/g", "/alias").unwrap();
    file.new_dataset::<f64>()
        .shape((2,))
        .create("keep")
        .unwrap()
        .write(&[9.0f64, 8.0])
        .unwrap();
    file.close().unwrap();
}

/// Editing a group with more than one hard link is refused rather than
/// silently diverging its aliases (issue #327).
///
/// A commit rebuilds a dirty group's object header at a fresh address and
/// patches the one link it resolved the group through. Every other link was
/// left naming the old header, which the same commit freed — so `/alias` showed
/// the pre-commit group, and the next commit in the session reused the span and
/// made it unreadable to both libraries. This is the rule the three relocating
/// *dataset* writes have always had, applied where it was missing.
#[test]
fn editing_a_hard_linked_group_is_refused() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_hardlink_group.h5");
    write_hard_linked_group(&path);
    let before = std::fs::read(&path).unwrap();

    let session = File::open_rw(&path).unwrap();
    session
        .root()
        .create_dataset("g/extra", |b| {
            b.with_i32_data(&[9]);
        })
        .unwrap();
    let refused = session.commit().unwrap_err();
    assert!(
        refused.to_string().contains("single hard link"),
        "expected the hard-link refusal, got: {refused}"
    );
    drop(session);

    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "a refused commit must leave the file byte-identical"
    );
    // And both readers still see one object through both of its links.
    let c = hdf5::File::open(&path).unwrap();
    for name in ["g", "alias"] {
        assert_eq!(
            c.group(name).unwrap().member_names().unwrap(),
            vec!["inner".to_string()],
            "{name} must be untouched"
        );
    }
}

/// The refusal reaches an edit *below* a hard-linked group, because every
/// ancestor on the edited path is rebuilt too.
#[test]
fn editing_below_a_hard_linked_group_is_refused() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_hardlink_group_deep.h5");
    {
        let file = hdf5::File::create(&path).unwrap();
        let g = file.create_group("g").unwrap();
        g.create_group("child").unwrap();
        file.link_hard("/g", "/alias").unwrap();
        file.close().unwrap();
    }
    let before = std::fs::read(&path).unwrap();

    let session = File::open_rw(&path).unwrap();
    session
        .root()
        .create_dataset("g/child/deep", |b| {
            b.with_i32_data(&[1]);
        })
        .unwrap();
    assert!(
        session.commit().is_err(),
        "rebuilding `child` rebuilds `g` above it, which is the hard-linked one"
    );
    drop(session);
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// The rule is about *hard* links, and about groups that actually have more
/// than one. Everything else still edits.
#[test]
fn the_hard_link_rule_does_not_refuse_an_ordinary_group() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_hardlink_group_ok.h5");
    {
        let file = create_v18(&path);
        let g = file.create_group("g").unwrap();
        g.create_group("child").unwrap();
        // A *soft* link is not a hard link: it resolves by path, so it still
        // finds the group after the rebuild and nothing is stranded.
        file.link_soft("/g", "/pointer").unwrap();
        file.close().unwrap();
    }

    let session = File::open_rw(&path).unwrap();
    // The root group is rebuilt by every commit and is named by the superblock
    // rather than by a link, so it must never be caught by this rule.
    session
        .root()
        .create_dataset("at_root", |b| {
            b.with_i32_data(&[1]);
        })
        .unwrap();
    session
        .root()
        .create_dataset("g/child/deep", |b| {
            b.with_i32_data(&[2]);
        })
        .unwrap();
    session.commit().unwrap();
    drop(session);

    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("at_root").unwrap().read_raw::<i32>().unwrap(),
        [1]
    );
    assert_eq!(
        c.dataset("g/child/deep")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [2]
    );
    assert_eq!(
        c.dataset("pointer/child/deep")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        [2],
        "the soft link still resolves to the rebuilt group"
    );
}

/// A hard-linked *dataset* under a hard-linked group, deleted rather than
/// edited: deletion does not relocate a header, so it is unaffected.
#[test]
fn deleting_a_link_to_a_hard_linked_group_still_works() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_hardlink_group_delete.h5");
    write_hard_linked_group(&path);

    let session = File::open_rw(&path).unwrap();
    session.root().delete("alias").unwrap();
    session.commit().unwrap();
    drop(session);

    let c = hdf5::File::open(&path).unwrap();
    assert!(c.group("alias").is_err(), "the removed link is gone");
    assert_eq!(
        c.group("g").unwrap().member_names().unwrap(),
        vec!["inner".to_string()],
        "the object survives its other link"
    );
}

/// When the link graph cannot be walked, the same edit is refused — with a
/// different message, because it is a different problem.
///
/// The count that proves a group has one hard link is a file-wide walk, and it
/// gives up on an object header it cannot parse. A file damaged that way was
/// previously edited around; there is no way to establish that nothing else
/// names the group being rebuilt, so it is refused rather than rebuilt on an
/// assumption. Saying "it has more than one hard link" would send the reader
/// looking for a second link that may not exist.
#[test]
fn editing_a_group_is_refused_when_the_links_cannot_be_walked() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c_hardlink_group_damaged.h5");
    {
        let file = create_v18(&path);
        let g = file.create_group("g").unwrap();
        g.new_dataset::<i32>()
            .shape((3,))
            .create("inner")
            .unwrap()
            .write(&[1i32, 2, 3])
            .unwrap();
        file.new_dataset::<i32>()
            .shape((1,))
            .create("damaged")
            .unwrap()
            .write(&[1i32])
            .unwrap();
        file.close().unwrap();
    }

    // Break one object header that is neither the root nor the edited group, by
    // giving it a version no reader accepts. The link walk stops there.
    let mut bytes = std::fs::read(&path).unwrap();
    let last = (0..bytes.len() - 4)
        .rfind(|&i| &bytes[i..i + 4] == b"OHDR")
        .expect("the fixture must carry version 2 object headers");
    bytes[last + 4] = 9;
    std::fs::write(&path, &bytes).unwrap();

    let session = File::open_rw(&path).unwrap();
    session
        .root()
        .create_dataset("g/extra", |b| {
            b.with_i32_data(&[9]);
        })
        .unwrap();
    let refused = session.commit().unwrap_err().to_string();
    assert!(
        refused.contains("could not be walked"),
        "expected the unwalkable-graph refusal, got: {refused}"
    );
    assert!(
        !refused.contains("single hard link"),
        "and not the one about aliases, which would misdirect: {refused}"
    );
    drop(session);
    assert_eq!(
        std::fs::read(&path).unwrap(),
        bytes,
        "a refused commit must leave the file byte-identical"
    );
}

/// A dataset the reference library created and never wrote is copied as the
/// storage it has — none — rather than refused (issue #336).
///
/// The C library does not allocate a contiguous dataset's data until something
/// is written to it, so one created and never written stores no data block at
/// all and its layout message carries the undefined address. Reading it answers
/// the fill value for every element (#284), which is exactly what makes a copy
/// that reproduced what it *read* look correct: a destination storing a grid of
/// fill values reads identically to one storing nothing. So the copy is checked
/// for what it **stores** — no data address, and a file too small to hold the
/// elements — as well as for what it reads. `repack` learned the same lesson in
/// #293.
///
/// The fill value is deliberately not zero: a copy that lost it would still read
/// as a plausible run of zeros.
#[test]
fn a_never_written_dataset_is_copied_as_the_storage_it_never_had() {
    const N: usize = 100_000;
    const FILL: i32 = -12_345;

    let dir = tempdir().unwrap();
    let path = dir.path().join("never_written_copy.h5");
    {
        // The copy path needs a version 2 object header, so the format is named
        // rather than left to the linked library's default (which has moved).
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        let ds = file
            .new_dataset::<i32>()
            .shape((N,))
            .fill_value(FILL)
            .create("d")
            .unwrap();
        ds.new_attr::<i32>()
            .shape(())
            .create("tag")
            .unwrap()
            .write_scalar(&99i32)
            .unwrap();
        // A group holding another one, to copy the same shape recursively.
        let g = file.create_group("g").unwrap();
        g.new_dataset::<f64>()
            .shape((4, 5))
            .create("inner")
            .unwrap();
        file.close().unwrap();
    }
    // The source stores no data block: this is the shape under test, not an
    // assumption about it.
    {
        let before = File::open(&path).unwrap();
        assert!(
            matches!(
                before.dataset("d").unwrap().layout().unwrap(),
                hdf5_pure::Layout::Contiguous { address: None, .. }
            ),
            "the fixture is meant to be a never-written dataset"
        );
    }

    let session = File::open_rw(&path).unwrap();
    session.copy("d", "d_copy").unwrap();
    session.copy("g", "g_copy").unwrap();
    session.commit().unwrap();
    drop(session);

    let f = File::open(&path).unwrap();
    for name in ["d_copy", "g_copy/inner"] {
        let ds = f.dataset(name).unwrap();
        assert!(
            matches!(
                ds.layout().unwrap(),
                hdf5_pure::Layout::Contiguous { address: None, .. }
            ),
            "{name}: the copy materialized storage the source never had: {:?}",
            ds.layout().unwrap()
        );
    }
    let copied = f.dataset("d_copy").unwrap();
    assert_eq!(copied.read_i32().unwrap(), vec![FILL; N], "values match");
    assert_eq!(
        copied.fill_value::<i32>().unwrap(),
        Some(FILL),
        "the fill value carried across"
    );
    assert_eq!(copied.shape().unwrap(), vec![N as u64], "shape carried");
    assert_eq!(
        copied.attrs().unwrap().get("tag"),
        Some(&AttrValue::I32(99)),
        "the attribute carried across"
    );
    assert_eq!(
        f.dataset("g_copy/inner").unwrap().read_f64().unwrap(),
        vec![0.0; 20],
        "the group's never-written child copied too"
    );
    drop(f);

    // The elements are absent rather than merely small: the whole file is
    // smaller than the run of bytes one copy of them would occupy. Stated as a
    // rule against the dataset's own size so it holds whatever the surrounding
    // metadata costs.
    let materialized = (N * core::mem::size_of::<i32>()) as u64;
    let len = std::fs::metadata(&path).unwrap().len();
    assert!(
        len < materialized,
        "the file ({len} B) carries the {materialized} B of elements neither the \
         source nor its copy ever stored"
    );

    // The reference library reads its own convention back out of the copy.
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("d_copy").unwrap().read_raw::<i32>().unwrap(),
        vec![FILL; N],
        "the C library reads the copy as its fill value"
    );
    assert_eq!(
        c.dataset("g_copy/inner")
            .unwrap()
            .read_raw::<f64>()
            .unwrap(),
        vec![0.0; 20]
    );
    c.close().unwrap();

    // The cross-file path reads the source at staging time and screens each
    // copied header for addresses that would dangle in another file, so it
    // reaches the same dataset through a different set of checks.
    let dst = dir.path().join("cross.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
        b.write(&dst).unwrap();
    }
    let source = File::open(&path).unwrap();
    let session = File::open_rw(&dst).unwrap();
    session.copy_from(&source, "d", "d_copy").unwrap();
    session.commit().unwrap();
    drop(session);
    drop(source);

    let f = File::open(&dst).unwrap();
    let ds = f.dataset("d_copy").unwrap();
    assert!(
        matches!(
            ds.layout().unwrap(),
            hdf5_pure::Layout::Contiguous { address: None, .. }
        ),
        "the cross-file copy materialized storage the source never had: {:?}",
        ds.layout().unwrap()
    );
    assert_eq!(ds.read_i32().unwrap(), vec![FILL; N]);
    drop(f);
    let len = std::fs::metadata(&dst).unwrap().len();
    assert!(
        len < materialized,
        "the cross-file destination ({len} B) carries the {materialized} B of \
         elements the source never stored"
    );
}

/// A never-written dataset whose *datatype* declares an object reference stores
/// no address at all, so copying it beside a delete has nothing that could point
/// into the space the delete reclaims (issues #317, #336).
///
/// The screen that enforces #317 reads a copied dataset's element bytes and
/// refuses any address landing in a vacated span. A dataset with no storage has
/// no element bytes to read, so it has to be skipped rather than screened as an
/// empty run — because when that screen cannot *map* a datatype's addresses it
/// refuses unconditionally, before reading a byte, and an empty run does not
/// save it.
///
/// Which is why all three unmappable shapes are here and not just the one this
/// crate can write. A dataset-region reference and a variable length of object
/// references are exactly the datatypes that refusal names, and an
/// object-reference dataset — the only one with mappable addresses — passes for
/// a reason that tells you nothing about the other two. Before #336 the whole
/// combination was unreachable: the copy was refused before any screen ran.
#[test]
fn a_never_written_reference_dataset_copies_beside_a_delete() {
    use hdf5::types::{Reference, TypeDescriptor};

    for (label, desc) in [
        ("object", TypeDescriptor::Reference(Reference::Object)),
        ("region", TypeDescriptor::Reference(Reference::Region)),
        (
            "vlen-of-object",
            TypeDescriptor::VarLenArray(Box::new(TypeDescriptor::Reference(Reference::Object))),
        ),
    ] {
        let dir = tempdir().unwrap();
        let path = dir.path().join("never_written_refs.h5");
        {
            // As above: the copy path needs a version 2 object header.
            let file = hdf5::File::with_options()
                .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
                .create(&path)
                .unwrap();
            file.new_dataset_builder()
                .empty_as(&desc)
                .shape((4,))
                .create("refs")
                .unwrap();
            file.new_dataset::<i32>()
                .shape((3,))
                .create("doomed")
                .unwrap()
                .write(&[1i32, 2, 3])
                .unwrap();
            file.close().unwrap();
        }

        let session = File::open_rw(&path).unwrap();
        session.copy("refs", "refs_copy").unwrap();
        session.root().delete("doomed").unwrap();
        session
            .commit()
            .unwrap_or_else(|e| panic!("{label}: a dataset storing no reference was refused: {e}"));
        drop(session);

        let f = File::open(&path).unwrap();
        let ds = f.dataset("refs_copy").unwrap();
        assert!(
            matches!(
                ds.layout().unwrap(),
                hdf5_pure::Layout::Contiguous { address: None, .. }
            ),
            "{label}: the copy materialized storage the source never had: {:?}",
            ds.layout().unwrap()
        );
        assert!(
            f.dataset("doomed").is_err(),
            "{label}: the delete went through"
        );
    }
}

#[test]
fn appending_into_a_staged_contiguous_dataset_stays_c_readable() {
    // `Dataset::append_staged` on a dataset staged in the *same* session folds
    // the elements into the pending creation, growing its raw bytes and its
    // leading dimension together. That pairing is reachable no other way — a
    // builder is handed its whole shape at once — and what it produces is an
    // ordinary fixed-shape contiguous dataset, so the reference library is what
    // proves the two halves agree (issue #392).
    let dir = tempdir().unwrap();
    let path = dir.path().join("staged_append.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1]);
    b.write(&path).unwrap();

    {
        let session = File::open_rw(&path).unwrap();
        let mut col = session
            .root()
            .create_dataset("col", |b| {
                b.with_f64_data(&[1.0, 2.0]);
            })
            .unwrap();
        col.append_staged(|a| {
            a.append_f64(&[3.0, 4.0, 5.0]);
        })
        .unwrap();
        assert_eq!(col.shape().unwrap(), vec![5]);
        session.commit().unwrap();
    } // drop the editor (release its exclusive lock) before reading back

    let f = File::open(&path).unwrap();
    assert_eq!(
        f.dataset("col").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0]
    );

    let c = hdf5::File::open(&path).unwrap();
    let ds = c.dataset("col").unwrap();
    assert_eq!(ds.shape(), vec![5], "the C library reads the grown shape");
    assert_eq!(
        ds.read_raw::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        "and every element the fold appended"
    );
}
