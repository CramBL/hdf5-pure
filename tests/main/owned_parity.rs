//! Owned-handle parity with `File::open_rw` (issue #148, PR A).
//!
//! The capabilities that were previously reachable only through `File::open_rw` —
//! a staged, index-rebuilding append (filtered / any-length), cross-file copy,
//! group attributes, live space accounting, staged-edit introspection, and a
//! per-open file-locking policy — now work through owned `File` / `Dataset` /
//! `Group` handles. Plus the post-`close` seal (`Error::FileClosed`).

use hdf5_pure::{AttrValue, Error, File, FileAccessProperties, FileBuilder, FileLocking};
use tempfile::tempdir;
use test_util_hdf5::dataset::{Filter, Unlimited};

fn build_simple(path: &std::path::Path, data: &[i32]) {
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(data)
        .with_shape(&[data.len() as u64]);
    b.write(path).unwrap();
}

/// Growing a **filtered** dataset that sits on a partial trailing chunk, both
/// ways: the immediate `Dataset::append` (fast in-place path, which re-encodes
/// that chunk since issue #393) and `Dataset::append_staged` (index-rebuild
/// path, applied on `commit`). What separates them here is *when* the elements
/// land, which is what the rest of this test pins.
#[test]
fn append_staged_grows_filtered_any_length() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("f.h5");
    Unlimited::new("d", &(0..3).collect::<Vec<i32>>(), 4)
        .filters(&[Filter::Deflate(6)])
        .pure_create(&path); // length 3, chunk 4 -> partial tail

    let file = File::open_rw(&path).unwrap();
    {
        let mut ds = file.dataset("d").unwrap();
        // Immediate in-place append: applied at once, nothing staged.
        ds.append(&[3i32, 4]).unwrap();
        assert!(!file.has_staged_edits());
        // The staged rebuild path takes the same shape, deferred to `commit`.
        ds.append_staged(|b| {
            b.append_i32(&[5, 6, 7]);
        })
        .unwrap();
    }
    // Staged: not applied until commit.
    assert!(file.has_staged_edits());
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        (0..5).collect::<Vec<_>>()
    );
    file.commit().unwrap();
    assert!(!file.has_staged_edits());
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        (0..8).collect::<Vec<_>>()
    );
}

#[test]
fn copy_from_another_file() {
    let dir = tempdir().unwrap();
    let src_path = dir.path().join("src.h5");
    let dst_path = dir.path().join("dst.h5");
    {
        let mut b = FileBuilder::new();
        b.create_dataset("cal")
            .with_f64_data(&[1.0, 2.0, 3.0])
            .with_shape(&[3]);
        b.write(&src_path).unwrap();
    }
    build_simple(&dst_path, &[1, 2, 3]);

    // Source is a separate, buffered read-only file (no lock conflict).
    let source = File::open(&src_path).unwrap();
    let file = File::open_rw(&dst_path).unwrap();
    file.copy_from(&source, "cal", "cal_copy").unwrap();
    file.commit().unwrap();
    assert_eq!(
        file.dataset("cal_copy").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0]
    );

    // A streaming source has no in-memory image and is refused.
    let streaming = File::open_streaming(&src_path).unwrap();
    assert!(matches!(
        file.copy_from(&streaming, "cal", "cal_copy2"),
        Err(Error::EditUnsupported(_))
    ));
}

#[test]
fn group_attrs_on_subgroup_and_root() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("g.h5");
    build_simple(&path, &[1]);

    let file = File::open_rw(&path).unwrap();
    file.root().create_group("grp").unwrap();
    file.commit().unwrap();

    file.group("grp")
        .unwrap()
        .set_attr("kind", AttrValue::String("trial".into()))
        .unwrap();
    file.root().set_attr("version", AttrValue::I64(7)).unwrap();
    file.commit().unwrap();

    assert_eq!(
        file.group("grp").unwrap().attrs().unwrap().get("kind"),
        Some(&AttrValue::String("trial".into()))
    );
    assert_eq!(
        file.root().attrs().unwrap().get("version"),
        Some(&AttrValue::I64(7))
    );

    file.group("grp").unwrap().remove_attr("kind").unwrap();
    file.commit().unwrap();
    assert!(
        !file
            .group("grp")
            .unwrap()
            .attrs()
            .unwrap()
            .contains_key("kind")
    );
}

#[test]
fn space_accounting_and_read_only_refusal() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("s.h5");
    build_simple(&path, &[1, 2, 3, 4, 5]);

    let file = File::open_rw(&path).unwrap();
    let acct = file.space_accounting().unwrap();
    assert!(acct.logical_size > 0);
    // Release the exclusive lock before opening the file another way.
    drop(file);

    let ro = File::from_bytes(std::fs::read(&path).unwrap()).unwrap();
    assert!(matches!(ro.space_accounting(), Err(Error::ReadOnly)));
    assert!(!ro.has_staged_edits());
}

#[test]
fn open_rw_with_locking_disabled_edits() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("l.h5");
    build_simple(&path, &[1, 2, 3]);

    let file = File::open_rw_with_options(
        &path,
        FileAccessProperties::new().with_locking(FileLocking::Disabled),
    )
    .unwrap();
    file.root().create_group("g").unwrap();
    file.commit().unwrap();
    assert!(file.group("g").is_ok());
}

/// After `File::close`, a surviving handle's writes are refused with
/// `Error::FileClosed`, while reads still work.
#[test]
fn writes_after_close_are_sealed() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("c.h5");
    build_simple(&path, &[1, 2, 3]);

    let file = File::open_rw(&path).unwrap();
    let mut ds = file.dataset("d").unwrap();
    let root = file.root();
    file.close().unwrap();

    assert!(matches!(ds.write(&[9i32, 9, 9]), Err(Error::FileClosed)));
    assert!(matches!(
        ds.set_attr("x", AttrValue::I64(1)),
        Err(Error::FileClosed)
    ));
    assert!(matches!(root.create_group("g"), Err(Error::FileClosed)));
    assert!(matches!(
        root.set_attr("v", AttrValue::I64(1)),
        Err(Error::FileClosed)
    ));
    // Reads through a surviving handle still work.
    assert_eq!(ds.read_i32().unwrap(), vec![1, 2, 3]);
}
