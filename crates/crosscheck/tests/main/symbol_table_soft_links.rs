#![cfg(feature = "__hdf5-1.10")]
//! A soft link in a version 1 (symbol-table) group, as libhdf5
//! (libhdf5) writes it under the earliest format bounds. Its symbol table entry
//! carries the undefined address in place of an object header, so the group
//! only lists its hard links. An edit that would convert the group to the latest
//! format would drop the link, so it is rejected.

use std::path::Path;

use hdf5_pure::{Error, File};

/// A version 0 file whose root holds a dataset `d` and a soft link `pointer`
/// to it, both in the root's symbol table.
fn write_soft_link_file(path: &Path) {
    let f = hdf5::File::with_options()
        .with_fapl(|p| p.libver_earliest())
        .create(path)
        .unwrap();
    f.new_dataset::<i32>()
        .shape([2])
        .create("d")
        .unwrap()
        .write_raw(&[1, 2])
        .unwrap();
    f.link_soft("/d", "pointer").unwrap();
    f.close().unwrap();
}

#[test]
fn a_soft_link_does_not_hide_hard_links_in_a_symbol_table_group() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("soft_v1.h5");
    write_soft_link_file(&path);

    let f = File::open(&path).unwrap();
    assert!(
        f.superblock().version <= 1,
        "expected a v0/v1 superblock from the earliest libver bound"
    );
    assert_eq!(f.root().datasets().unwrap(), ["d"]);
    assert_eq!(f.dataset("d").unwrap().read_i32().unwrap(), [1, 2]);
}

#[test]
fn converting_a_symbol_table_group_holding_a_soft_link_refuses() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("soft_v1_edit.h5");
    write_soft_link_file(&path);
    let before = std::fs::read(&path).unwrap();

    // Scoped so the session, and the file lock it holds, is gone before the
    // bytes are read back.
    let err = {
        let session = File::open_rw(&path).unwrap();
        session
            .root()
            .create_dataset("added", |b| {
                b.with_i32_data(&[3]);
            })
            .unwrap();
        session.commit().unwrap_err()
    };
    let Error::EditUnsupported(reason) = &err else {
        panic!("expected EditUnsupported, got {err:?}");
    };
    assert_eq!(
        *reason,
        "a version 1 group holding a soft link cannot be converted in place"
    );
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "the file is unchanged"
    );

    // The reference library still follows the link.
    let c = hdf5::File::open(&path).unwrap();
    assert_eq!(
        c.dataset("pointer").unwrap().read_raw::<i32>().unwrap(),
        [1, 2]
    );
}
