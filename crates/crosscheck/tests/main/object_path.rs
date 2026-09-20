#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Object paths, resolved over one file by the reference C library and by hdf5-pure.
//!
//! The C library's traversal is the oracle for the grammar: which spellings name one object, where
//! an absolute path opened on a group starts, that `..` is an ordinary link name, and which path
//! neither library accepts as the target of a write.

use hdf5::{MajorErrorCode, MinorErrorCode};
use hdf5_pure::{Error, File};
use rstest::rstest;
use tempfile::tempdir;

/// Writes a file holding `/b`, `/a/b` and `/a`, where `/b` reads 2 and `/a/b` reads 1, so that a
/// path resolved from the root and the same path resolved from `a` reach different data.
fn nested_file(path: &std::path::Path) {
    let f = hdf5::File::create(path).unwrap();
    f.new_dataset::<i32>()
        .shape((1,))
        .create("b")
        .unwrap()
        .write(&[2i32])
        .unwrap();
    let a = f.create_group("a").unwrap();
    a.new_dataset::<i32>()
        .shape((1,))
        .create("b")
        .unwrap()
        .write(&[1i32])
        .unwrap();
    drop(a);
    f.close().unwrap();
}

#[rstest]
#[case("a/b")]
#[case("/a/b")]
#[case("a/b/")]
#[case("/a/b/")]
#[case("a//b")]
#[case("///a///b///")]
#[case("./a/b")]
#[case("a/./b")]
#[case("a/b/.")]
fn both_libraries_resolve_every_spelling_of_a_path_to_one_dataset(#[case] spelling: &str) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("nested.h5");
    nested_file(&path);

    let c_read = hdf5::File::open(&path)
        .unwrap()
        .dataset(spelling)
        .unwrap_or_else(|e| panic!("the C library rejected {spelling:?}: {e}"))
        .read_raw::<i32>()
        .unwrap();
    assert_eq!(c_read, vec![1], "the C library read {spelling:?}");

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset(spelling).unwrap().read_i32().unwrap(), c_read);
    assert_eq!(
        file.root().dataset(spelling).unwrap().read_i32().unwrap(),
        c_read
    );
}

#[rstest]
#[case("/b", 2)]
#[case("//b", 2)]
#[case("/a/b", 1)]
#[case("b", 1)]
fn both_libraries_walk_a_path_opened_on_a_group_from_the_same_place(
    #[case] spelling: &str,
    #[case] value: i32,
) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("nested.h5");
    nested_file(&path);

    let c_file = hdf5::File::open(&path).unwrap();
    let c_read = c_file
        .group("a")
        .unwrap()
        .dataset(spelling)
        .unwrap_or_else(|e| panic!("the C library rejected {spelling:?}: {e}"))
        .read_raw::<i32>()
        .unwrap();
    assert_eq!(c_read, vec![value], "the C library read {spelling:?}");

    let file = File::open(&path).unwrap();
    assert_eq!(
        file.group("a")
            .unwrap()
            .dataset(spelling)
            .unwrap()
            .read_i32()
            .unwrap(),
        c_read
    );
}

#[test]
fn both_libraries_read_two_dots_as_an_ordinary_link_name() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("two_dots.h5");

    {
        let f = hdf5::File::create(&path).unwrap();
        f.new_dataset::<i32>()
            .shape((1,))
            .create("b")
            .unwrap()
            .write(&[2i32])
            .unwrap();
        let a = f.create_group("a").unwrap();
        a.new_dataset::<i32>()
            .shape((1,))
            .create("b")
            .unwrap()
            .write(&[1i32])
            .unwrap();
        let up = a.create_group("..").unwrap();
        up.new_dataset::<i32>()
            .shape((1,))
            .create("b")
            .unwrap()
            .write(&[3i32])
            .unwrap();
        drop(up);
        drop(a);
        f.close().unwrap();
    }

    let c_file = hdf5::File::open(&path).unwrap();
    let mut members = c_file.group("a").unwrap().member_names().unwrap();
    members.sort();
    assert_eq!(members, ["..", "b"]);
    assert_eq!(
        c_file.dataset("a/../b").unwrap().read_raw::<i32>().unwrap(),
        vec![3],
        "the C library reads `..` as a link name, not as the parent group"
    );

    let file = File::open(&path).unwrap();
    assert_eq!(file.dataset("a/../b").unwrap().read_i32().unwrap(), vec![3]);
    assert_eq!(
        file.root().dataset("a/../b").unwrap().read_i32().unwrap(),
        vec![3]
    );
}

#[rstest]
#[case("", MajorErrorCode::Args, MinorErrorCode::BadValue)]
#[case(".", MajorErrorCode::Link, MinorErrorCode::CantDelete)]
#[case("./.", MajorErrorCode::Link, MinorErrorCode::CantDelete)]
fn neither_library_deletes_a_group_through_a_path_that_spells_no_link(
    #[case] spelling: &str,
    #[case] major: MajorErrorCode,
    #[case] minor: MinorErrorCode,
) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("nested.h5");
    nested_file(&path);

    {
        let c_file = hdf5::File::open_rw(&path).unwrap();
        let a = c_file.group("a").unwrap();
        let c_err = a.unlink(spelling).unwrap_err();
        assert!(c_err.contains_major(major), "{c_err}");
        assert!(c_err.contains_minor(minor), "{c_err}");
        assert_eq!(a.member_names().unwrap(), ["b"]);
        drop(a);
        c_file.close().unwrap();
    }

    let file = File::open_rw(&path).unwrap();
    let a = file.group("a").unwrap();
    let err = a.delete(spelling).unwrap_err();
    let Error::EditUnsupported(reason) = &err else {
        panic!("expected EditUnsupported, got {err:?}");
    };
    assert_eq!(
        *reason,
        "a write needs a link name, and this path holds none"
    );
    file.commit().unwrap();
    assert_eq!(a.datasets().unwrap(), ["b"]);
    assert_eq!(file.root().groups().unwrap(), ["a"]);
}
