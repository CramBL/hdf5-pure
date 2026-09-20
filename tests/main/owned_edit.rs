//! Owned-handle write surface (issue #148, phase 2b): overwrite values, edit
//! attributes, create subgroups, and delete objects through owned handles on a
//! `File::open_rw` file, applied by `File::commit`. Immediate appends are covered
//! in `owned_append.rs`.

use hdf5_pure::{AttrValue, Error, File, FileBuilder};
use tempfile::tempdir;

fn create_i32(path: &std::path::Path, data: &[i32]) {
    let mut b = FileBuilder::new();
    b.create_dataset("d")
        .with_i32_data(data)
        .with_shape(&[data.len() as u64]);
    b.write(path).unwrap();
}

#[test]
fn write_overwrites_values_after_commit() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, &[1, 2, 3]);
    let file = File::open_rw(&path).unwrap();
    {
        let mut ds = file.dataset("d").unwrap();
        ds.write(&[10i32, 20, 30]).unwrap(); // staged (same shape)
    }
    // Staged: not visible until commit.
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    file.commit().unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        vec![10, 20, 30]
    );
}

#[test]
fn set_and_remove_attr() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, &[1, 2, 3]);
    let file = File::open_rw(&path).unwrap();
    {
        let mut ds = file.dataset("d").unwrap();
        ds.set_attr("count", AttrValue::I64(42)).unwrap();
        ds.set_attr("unit", AttrValue::String("m/s".into()))
            .unwrap();
    }
    file.commit().unwrap();
    let attrs = file.dataset("d").unwrap().attrs().unwrap();
    assert_eq!(attrs.get("count"), Some(&AttrValue::I64(42)));
    assert_eq!(attrs.get("unit"), Some(&AttrValue::String("m/s".into())));

    {
        let mut ds = file.dataset("d").unwrap();
        ds.remove_attr("count").unwrap();
    }
    file.commit().unwrap();
    assert!(
        !file
            .dataset("d")
            .unwrap()
            .attrs()
            .unwrap()
            .contains_key("count")
    );
}

#[test]
fn create_group_and_delete_object() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, &[1, 2, 3]);
    let file = File::open_rw(&path).unwrap();

    file.root().create_group("newgrp").unwrap();
    file.commit().unwrap();
    assert!(file.group("newgrp").is_ok());

    file.root().delete("d").unwrap();
    file.commit().unwrap();
    assert!(file.dataset("d").is_err());
}

#[test]
fn writes_on_readonly_file_are_refused() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, &[1, 2, 3]);
    let bytes = std::fs::read(&path).unwrap();
    let file = File::from_bytes(bytes).unwrap();
    let mut ds = file.dataset("d").unwrap();
    assert!(matches!(ds.write(&[9i32]), Err(Error::ReadOnly)));
    assert!(matches!(
        ds.set_attr("x", AttrValue::I64(1)),
        Err(Error::ReadOnly)
    ));
    assert!(matches!(file.commit(), Err(Error::ReadOnly)));
    assert!(matches!(
        file.root().create_group("g"),
        Err(Error::ReadOnly)
    ));
}

#[test]
fn create_dataset_and_read_back() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, &[1, 2, 3]);
    let file = File::open_rw(&path).unwrap();
    file.root()
        .create_dataset("new", |b| {
            b.with_i32_data(&[7, 8, 9]).with_shape(&[3]);
        })
        .unwrap();
    file.commit().unwrap();
    assert_eq!(
        file.dataset("new").unwrap().read_i32().unwrap(),
        vec![7, 8, 9]
    );
}

#[test]
fn create_file_and_build_through_handles() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("new.h5");
    let file = File::create(&path).unwrap();
    file.root()
        .create_dataset("d", |b| {
            b.with_i32_data(&[1, 2, 3]).with_shape(&[3]);
        })
        .unwrap();
    file.root().create_group("grp").unwrap();
    file.commit().unwrap();
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert!(file.group("grp").is_ok());
    // Reopen read-only to confirm it persisted as a valid file.
    drop(file);
    let ro = File::open(&path).unwrap();
    assert_eq!(ro.dataset("d").unwrap().read_i32().unwrap(), vec![1, 2, 3]);
}

#[test]
fn copy_dataset_within_file() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    create_i32(&path, &[1, 2, 3]);
    let file = File::open_rw(&path).unwrap();
    file.copy("d", "d_copy").unwrap();
    file.commit().unwrap();
    assert_eq!(
        file.dataset("d_copy").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    // Original is untouched.
    assert_eq!(
        file.dataset("d").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
}
